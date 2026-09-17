"""Deterministic tonic ending materialization for the solo-piano profile."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import gcd, lcm
from typing import cast

from .projection import PlanChoice, project_solo_piano_3m
from .realization_workspace import (
    AccompanimentValue,
    HarmonyChord,
    MelodyNote,
    MelodyValue,
    RealizationWorkspace,
)
from .solo_piano_accompaniment import (
    AccompanimentRequestEvent,
    place_workspace_accompaniment,
)

_MINIMUM_ENDING_HOLD_MS = 2_000


@dataclass(frozen=True, slots=True)
class EndingMaterialization:
    """One jointly generated harmony, melody, and accompaniment ending."""

    material_key: str
    previous_material_key: str
    length_units: int
    harmonies: tuple[HarmonyChord, ...]
    melody: MelodyValue
    accompaniment: AccompanimentValue
    common_grid_units_per_weight: int
    material_scale_factors: tuple[tuple[str, int], ...]
    nominal_hold_ms: int
    placement_candidate_evaluation_count: int


def ending_materialization_diagnostic(
    materialization: EndingMaterialization,
) -> dict[str, object]:
    """Return the stable diagnostic saved beside a deterministic ending run."""
    return {
        "schema_version": 1,
        "material_key": materialization.material_key,
        "previous_material_key": materialization.previous_material_key,
        "length_units": materialization.length_units,
        "common_grid_units_per_weight": materialization.common_grid_units_per_weight,
        "material_scale_factors": dict(materialization.material_scale_factors),
        "nominal_hold_ms": materialization.nominal_hold_ms,
        "placement_candidate_evaluation_count": (
            materialization.placement_candidate_evaluation_count
        ),
    }


def materialize_solo_piano_ending(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
) -> EndingMaterialization:
    """Create the profile's single tonic ending from verified upstream values."""
    if workspace.schema_version not in {3, 4}:
        raise ValueError("Ending materialization requires a version 3 or 4 workspace")
    if workspace.plan is None:
        raise ValueError("Ending materialization requires a current overall plan")
    projection = project_solo_piano_3m(
        cast(dict[str, object], document),
        plan_choice=PlanChoice(
            tonal_center=workspace.plan.tonal_center,
            mode=workspace.plan.mode,
            harmonic_focus_by_section={},
            contrasts_with_by_section={},
        ),
    )
    if not projection.projected or projection.piece_plan is None:
        message = projection.issues[0].message if projection.issues else "Ending projection failed"
        raise ValueError(message)
    plan = projection.piece_plan
    leaves = tuple(node for node in plan.nodes if node.score_material_id is not None)
    if len(leaves) < 2:
        raise ValueError("Ending materialization requires music before a final release")
    final = leaves[-1]
    if final.role != "release" or final.duration_weight is None:
        raise ValueError("The final leaf must be a release with a positive relative length")
    script = cast(Mapping[str, object], document["script"])
    materials = cast(Mapping[str, Mapping[str, object]], script["materials"])
    material_key = cast(str, final.score_material_id)
    ending_keys = {key for key, value in materials.items() if value["kind"] == "ending"}
    if ending_keys != {material_key}:
        raise ValueError("The profile requires exactly one placed ending material")

    previous = next(
        (
            node
            for node in reversed(leaves[:-1])
            if materials[cast(str, node.score_material_id)]["kind"] not in {"transition", "ending"}
        ),
        None,
    )
    if previous is None:
        raise ValueError("The final release has no preceding ordinary melody")
    previous_material_key = previous.score_material_id
    melodies = dict(workspace.melodies)
    previous_melody = melodies.get(previous_material_key)
    if previous_melody is None or not previous_melody.notes:
        raise ValueError("The melody before the final release is missing")
    final_onset = max(note.at_units for note in previous_melody.notes)
    boundary_pitches = tuple(
        note.pitch for note in previous_melody.notes if note.at_units == final_onset
    )
    select = max if previous_melody.foreground_voice == "upper" else min
    previous_pitch = select(boundary_pitches)
    tonic_pitch = min(
        (pitch for pitch in range(21, 109) if pitch % 12 == plan.tonal_center),
        key=lambda pitch: (abs(pitch - previous_pitch), pitch),
    )

    length_units = final.duration_weight
    harmony = HarmonyChord(length_units, plan.tonal_center, plan.mode)
    ending_melody = MelodyValue(
        previous_melody.foreground_voice,
        (MelodyNote(0, length_units, tonic_pitch),),
    )
    zones = ("bass", "low") if previous_melody.foreground_voice == "upper" else ("middle", "high")
    requests = tuple(
        AccompanimentRequestEvent(
            at_units=0,
            preferred_duration_units=length_units,
            degree=degree,
            preferred_register_zone=zone,
            articulations=("tenuto",),
        )
        for degree, zone in zip(("root", "fifth"), zones, strict=True)
    )
    placement = place_workspace_accompaniment(
        material_key=material_key,
        tonal_center=plan.tonal_center,
        mode=plan.mode,
        harmonies=(harmony,),
        melody=ending_melody,
        requests=requests,
    )
    if placement.value is None:
        raise ValueError(placement.reason or placement.status)
    ending_accompaniment = placement.value
    if any(
        note.at_units != 0 or note.duration_units != length_units
        for note in ending_accompaniment.notes
    ):
        raise ValueError("The ending accompaniment does not share the full hold")
    if {note.degree for note in ending_accompaniment.notes} != {"root", "fifth"}:
        raise ValueError("The ending accompaniment must contain root and fifth")

    lengths = {
        key: sum(chord.duration_units for chord in chords) for key, chords in workspace.harmonies
    }
    lengths[material_key] = length_units
    weights: dict[str, int] = {}
    for leaf in leaves:
        assert leaf.score_material_id is not None and leaf.duration_weight is not None
        weights[leaf.score_material_id] = leaf.duration_weight
    missing_lengths = sorted(set(weights) - set(lengths))
    if missing_lengths:
        raise ValueError(f"A placed material has no current harmony: {missing_lengths[0]}")
    common_grid = lcm(
        *(lengths[key] // gcd(lengths[key], weight) for key, weight in weights.items())
    )
    scale_factors = tuple(
        sorted((key, weights[key] * common_grid // lengths[key]) for key in weights)
    )
    setup = cast(Mapping[str, object], script["performance_setup"])
    target_duration_ms = cast(int, setup["target_duration_seconds"]) * 1_000
    total_weight = sum(cast(int, leaf.duration_weight) for leaf in leaves)
    nominal_hold_ms = target_duration_ms * length_units // total_weight
    if target_duration_ms * length_units < _MINIMUM_ENDING_HOLD_MS * total_weight:
        raise ValueError("The final release is too short for the minimum ending hold")

    return EndingMaterialization(
        material_key=material_key,
        previous_material_key=previous_material_key,
        length_units=length_units,
        harmonies=(harmony,),
        melody=ending_melody,
        accompaniment=ending_accompaniment,
        common_grid_units_per_weight=common_grid,
        material_scale_factors=scale_factors,
        nominal_hold_ms=nominal_hold_ms,
        placement_candidate_evaluation_count=placement.candidate_evaluation_count,
    )
