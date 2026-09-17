"""Load and validate the complete phase-3 boundary state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from .generation_script_validation import check_generation_script_document
from .phase3_model_contracts import HarmonicPlan, SectionHarmonicIntent
from .phase3_realization import expected_phase2_projection_targets
from .projection_ledger import (
    ProjectionLedgerEntry,
    validate_expected_projection_targets,
    validate_projection_ledger,
)
from .score_ir import PiecePlan, PlanNode, ScoreHarmony, validate_piece_plan
from .score_projection import PlanChoice, build_piece_plan


@dataclass(frozen=True, slots=True)
class LoadedPhase3State:
    """Typed phase-3 values proven to belong to the supplied inputs."""

    plan: HarmonicPlan
    harmonies_by_score_unit: dict[str, tuple[ScoreHarmony, ...]]
    cumulative_projection_ledger: tuple[ProjectionLedgerEntry, ...]


def load_complete_phase3_state(
    validated_script: Mapping[str, object],
    phase3_state: Mapping[str, object],
    projection_ledger: Sequence[ProjectionLedgerEntry | Mapping[str, object]],
    *,
    target_profile: str = "solo_piano_3m_v2",
) -> LoadedPhase3State:
    """Return phase-3 values only after structural and lineage checks pass."""
    if target_profile != "solo_piano_3m_v2":
        raise ValueError("the phase-4 profile is unsupported")
    script_check = check_generation_script_document(validated_script)
    if not script_check.valid:
        raise ValueError(script_check.issues[0].message)
    if phase3_state.get("outcome") != "complete":
        raise ValueError("phase 4 requires a complete phase-3 state")
    raw_plan = cast(Mapping[str, object], phase3_state["harmonic_plan"])
    raw_piece_plan = cast(Mapping[str, object], raw_plan["piece_plan"])
    piece_plan = PiecePlan(
        plan_id=cast(str, raw_piece_plan["plan_id"]),
        title=cast(str, raw_piece_plan["title"]),
        tonal_center=cast(int, raw_piece_plan["tonal_center"]),
        mode=cast(str, raw_piece_plan["mode"]),
        root_section_id=cast(str, raw_piece_plan["root_section_id"]),
        nodes=tuple(
            PlanNode(**cast(dict[str, object], node))
            for node in cast(list[object], raw_piece_plan["nodes"])
        ),
    )
    validate_piece_plan(piece_plan)
    expected_plan, _ = build_piece_plan(
        validated_script,
        PlanChoice(piece_plan.tonal_center, piece_plan.mode),
    )
    if piece_plan != expected_plan:
        raise ValueError("the phase-3 piece plan does not match the script")
    length_units = {
        cast(str, key): cast(int, value)
        for key, value in cast(Mapping[str, object], raw_plan["length_units_by_score_unit"]).items()
    }
    expected_unit_ids = {
        cast(str, node.score_unit_id) for node in piece_plan.nodes if node.score_unit_id is not None
    }
    if set(length_units) != expected_unit_ids or any(value <= 0 for value in length_units.values()):
        raise ValueError("phase-3 lengths do not exactly cover score units")
    intents = tuple(
        SectionHarmonicIntent(**cast(dict[str, object], value))
        for value in cast(list[object], raw_plan["section_intents"])
    )
    if {intent.score_unit_id for intent in intents} != expected_unit_ids:
        raise ValueError("phase-3 intents do not exactly cover score units")
    local_ledger = tuple(
        ProjectionLedgerEntry(**cast(dict[str, object], value))
        for value in cast(list[object], raw_plan["projection_ledger"])
    )
    plan = HarmonicPlan(
        piece_plan=piece_plan,
        overall_harmonic_story=cast(str, raw_plan["overall_harmonic_story"]),
        section_intents=intents,
        divisions=cast(int, raw_plan["divisions"]),
        length_units_by_score_unit=length_units,
        projection_ledger=local_ledger,
    )
    raw_harmonies = cast(
        Mapping[str, list[dict[str, object]]], phase3_state["harmonies_by_score_unit"]
    )
    harmonies = {
        score_unit_id: tuple(ScoreHarmony(**value) for value in values)
        for score_unit_id, values in raw_harmonies.items()
    }
    if set(harmonies) != expected_unit_ids:
        raise ValueError("phase-3 harmonies do not exactly cover score units")
    for score_unit_id, values in harmonies.items():
        cursor = 0
        for harmony in values:
            if harmony.at_units != cursor or harmony.duration_units <= 0:
                raise ValueError("phase-3 harmony has a gap or overlap")
            cursor += harmony.duration_units
        if cursor != length_units[score_unit_id]:
            raise ValueError("phase-3 harmony does not cover its score unit")
    input_ledger = tuple(
        entry
        if isinstance(entry, ProjectionLedgerEntry)
        else ProjectionLedgerEntry(**cast(dict[str, object], entry))
        for entry in projection_ledger
    )
    validate_projection_ledger(input_ledger)
    phase3_ledger = tuple(
        ProjectionLedgerEntry(**cast(dict[str, object], value))
        for value in cast(list[object], phase3_state["projection_ledger"])
    )
    validate_projection_ledger(phase3_ledger)
    if (
        not phase3_ledger
        or tuple(input_ledger[-len(phase3_ledger) :]) != phase3_ledger
        or phase3_ledger[: len(local_ledger)] != local_ledger
    ):
        raise ValueError("phase-3 state does not match the cumulative projection ledger")
    expected_targets = expected_phase2_projection_targets(validated_script)
    expected_targets.update(
        {
            "plan_node": frozenset(node.section_id for node in piece_plan.nodes),
            "score_unit": frozenset(expected_unit_ids),
            "score_harmony": frozenset(
                harmony.harmony_id for values in harmonies.values() for harmony in values
            ),
        }
    )
    validate_expected_projection_targets(input_ledger, expected_targets)
    return LoadedPhase3State(plan, harmonies, input_ledger)
