"""Deterministic diagnostic SMF for a phase-4 foreground-only state."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

from .neutral_score_preview import NeutralPreviewSegment, write_neutral_score_preview
from .phase3_model_contracts import HarmonicPlan, ordered_leaf_section_ids
from .score_ir import ScoreNote


def write_phase4_foreground_preview(
    document: Mapping[str, object],
    plan: HarmonicPlan,
    notes_by_material_placement: Mapping[str, tuple[ScoreNote, ...]],
    output_path: Path,
) -> Path:
    """Write only accepted foreground notes while retaining the full piece duration."""
    script = cast(Mapping[str, object], document["script"])
    setup = cast(Mapping[str, object], script["performance_setup"])
    placements = cast(Mapping[str, Mapping[str, object]], script["material_placements"])
    leaf_ids = ordered_leaf_section_ids(document)
    unit_id_by_section = {
        node.section_id: cast(str, node.score_unit_id)
        for node in plan.piece_plan.nodes
        if node.score_unit_id is not None
    }
    target_seconds = cast(float, setup["target_duration_seconds"])
    notes_by_section: dict[str, list[ScoreNote]] = {}
    for placement_id, notes in notes_by_material_placement.items():
        section_id = cast(str, placements[placement_id]["section_id"])
        notes_by_section.setdefault(section_id, []).extend(notes)
    segments = tuple(
        NeutralPreviewSegment(
            plan.length_units_by_score_unit[unit_id_by_section[section_id]],
            tuple(notes_by_section.get(section_id, ())),
        )
        for section_id in leaf_ids
    )
    return write_neutral_score_preview(
        segments,
        target_seconds,
        output_path,
        meta_track_name="SCoIM phase 4 foreground",
        note_track_name="Foreground",
    )
