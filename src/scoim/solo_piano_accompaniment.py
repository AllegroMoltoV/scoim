"""Deterministic solo-piano placement for coarse accompaniment choices."""

from dataclasses import dataclass

from llm_musical_composer.performance_pipeline import (
    HARMONY_INTERVALS,
    PiecePlan,
    PlanNode,
    ScoreHarmony,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
)
from llm_musical_composer.piano_texture_register_placement import (
    PianoTextureEventV2,
    PianoTextureSpecV2,
    place_piano_texture_v8,
)

from .realization_workspace import (
    AccompanimentNote,
    AccompanimentValue,
    HarmonyChord,
    MelodyValue,
)


@dataclass(frozen=True, slots=True)
class AccompanimentRequestEvent:
    """One model-selected accompaniment preference without fixed identifiers or pitch."""

    at_units: int
    preferred_duration_units: int
    degree: str
    preferred_register_zone: str
    articulations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AccompanimentPlacement:
    """A placed value or one deterministic placement failure."""

    status: str
    value: AccompanimentValue | None
    reason: str | None
    candidate_evaluation_count: int


def place_workspace_accompaniment(
    *,
    material_key: str,
    tonal_center: int,
    mode: str,
    harmonies: tuple[HarmonyChord, ...],
    melody: MelodyValue,
    requests: tuple[AccompanimentRequestEvent, ...],
) -> AccompanimentPlacement:
    """Compile coarse accompaniment preferences through the existing V8 placer."""
    length_units = sum(chord.duration_units for chord in harmonies)
    plan = PiecePlan(
        plan_id=f"workspace-plan-{material_key}",
        title="SCoIM workspace accompaniment",
        tonal_center=tonal_center,
        mode=mode,
        root_node_id="workspace-root",
        ending_intent="tonic",
        nodes=(
            PlanNode("workspace-root", None, 0, "whole"),
            PlanNode(
                "workspace-leaf",
                "workspace-root",
                0,
                "statement",
                duration_weight=length_units,
                score_material_id=material_key,
            ),
        ),
    )
    score_harmonies: list[ScoreHarmony] = []
    cursor = 0
    for index, chord in enumerate(harmonies):
        score_harmonies.append(
            ScoreHarmony(
                f"workspace-h-{index:03d}",
                cursor,
                chord.duration_units,
                chord.root_pitch_class,
                chord.quality,
            )
        )
        cursor += chord.duration_units
    score_notes = tuple(
        ScoreNote(
            f"workspace-m-{index:03d}",
            note.at_units,
            note.duration_units,
            note.pitch,
            melody.foreground_voice,
        )
        for index, note in enumerate(melody.notes)
    )
    score = ScoreSpec(
        score_id=f"workspace-score-{material_key}",
        divisions=12,
        materials=(
            ScoreMaterial(
                material_key,
                length_units,
                score_notes,
                harmonies=tuple(score_harmonies),
                foreground_voice=melody.foreground_voice,
            ),
        ),
    )
    accompaniment_voice = "lower" if melody.foreground_voice == "upper" else "upper"
    texture_events: list[PianoTextureEventV2] = []
    for index, request in enumerate(requests):
        harmony = next(
            (
                item
                for item in score_harmonies
                if item.at_units <= request.at_units
                and request.at_units + request.preferred_duration_units
                <= item.at_units + item.duration_units
            ),
            None,
        )
        harmony_id = harmony.harmony_id if harmony is not None else "workspace-unresolved"
        texture_events.append(
            PianoTextureEventV2(
                f"workspace-a-{index:03d}",
                harmony_id,
                "accompaniment",
                accompaniment_voice,
                request.at_units,
                request.preferred_duration_units,
                request.degree,
                request.preferred_register_zone,
                request.articulations,
            )
        )
    placement = place_piano_texture_v8(
        plan,
        score,
        PianoTextureSpecV2(material_key, tuple(texture_events)),
    )
    if placement.status != "placed":
        return AccompanimentPlacement(
            placement.status,
            None,
            placement.reason,
            placement.total_candidate_evaluation_count,
        )
    placed_by_id = {note.event_id: note for note in placement.notes}
    expected_ids = {f"workspace-a-{index:03d}" for index in range(len(requests))}
    if len(placement.notes) != len(requests) or set(placed_by_id) != expected_ids:
        raise ValueError("The placer changed the accompaniment event set")
    for index, request in enumerate(requests):
        note = placed_by_id[f"workspace-a-{index:03d}"]
        if note.at_units != request.at_units:
            raise ValueError("The placer changed an accompaniment onset")
        if note.voice != accompaniment_voice:
            raise ValueError("The placer changed an accompaniment voice")
        if note.articulations != request.articulations:
            raise ValueError("The placer changed accompaniment articulations")
        if not 1 <= note.duration_units <= request.preferred_duration_units:
            raise ValueError("The placer changed an accompaniment duration outside its preference")
        harmony = next(
            item
            for item in score_harmonies
            if item.at_units <= request.at_units
            and request.at_units + request.preferred_duration_units
            <= item.at_units + item.duration_units
        )
        degree_index = {"root": 0, "third": 1, "fifth": 2, "seventh": 3}[request.degree]
        intervals = HARMONY_INTERVALS[harmony.quality]
        if (
            degree_index >= len(intervals)
            or note.pitch % 12 != (harmony.root_pitch_class + intervals[degree_index]) % 12
        ):
            raise ValueError("The placer changed an accompaniment degree")
    notes = tuple(
        AccompanimentNote(
            request.at_units,
            request.preferred_duration_units,
            placed_by_id[f"workspace-a-{index:03d}"].duration_units,
            request.degree,
            request.preferred_register_zone,
            placed_by_id[f"workspace-a-{index:03d}"].pitch,
            request.articulations,
        )
        for index, request in enumerate(requests)
    )
    return AccompanimentPlacement(
        "placed",
        AccompanimentValue(melody.foreground_voice, notes),
        None,
        placement.total_candidate_evaluation_count,
    )
