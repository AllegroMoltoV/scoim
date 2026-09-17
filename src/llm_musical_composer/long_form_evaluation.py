"""宣言した大区分が実音の活動量と回帰に現れるかを軸別に記述する。"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

from llm_musical_composer.composition_ir import Composition, Material, Note
from llm_musical_composer.section_contrast import _features, _secondary_changes

ACTIVITY_METRICS = ("note_density", "velocity_median", "polyphony_mean")


def _expanded_part_material(composition: Composition, part_index: int) -> Material:
    part = composition.parts[part_index]
    material_map = composition.material_by_id
    offset = 0
    notes: list[Note] = []
    for use in composition.form[part.start_use_index : part.end_use_index]:
        material = material_map[use.material_id]
        notes.extend(
            Note(
                event_id=note.event_id,
                at_ms=offset + note.at_ms,
                duration_ms=note.duration_ms,
                pitch=note.pitch,
                velocity=note.velocity,
            )
            for note in material.notes
        )
        offset += material.duration_ms
    return Material(part.part_id, offset, tuple(notes))


def _expanded_use_range_material(
    composition: Composition, material_id: str, start_use_index: int, end_use_index: int
) -> Material:
    material_map = composition.material_by_id
    offset = 0
    notes: list[Note] = []
    for use in composition.form[start_use_index:end_use_index]:
        material = material_map[use.material_id]
        notes.extend(
            Note(
                event_id=note.event_id,
                at_ms=offset + note.at_ms,
                duration_ms=note.duration_ms,
                pitch=note.pitch,
                velocity=note.velocity,
            )
            for note in material.notes
        )
        offset += material.duration_ms
    return Material(material_id, offset, tuple(notes))


def evaluate_long_form_structure(composition: Composition) -> dict[str, Any]:
    """大区分の実音特徴、最高潮、下降、正確な素材回帰を合算せず返す。"""
    if not composition.parts:
        return {
            "status": "unable_to_investigate",
            "issues": ["parts are missing"],
            "parts": [],
            "phrases": [],
            "boundaries": [],
            "climax": None,
            "post_climax_decline": None,
            "return": None,
        }
    reports = []
    for index, part in enumerate(composition.parts):
        material = _expanded_part_material(composition, index)
        reports.append(
            {
                "index": index,
                "part_id": part.part_id,
                "role": part.role,
                "energy": part.energy,
                "start_use_index": part.start_use_index,
                "end_use_index": part.end_use_index,
                "duration_ms": material.duration_ms,
                "features": _features(material),
            }
        )
    boundaries = [
        {
            "after_part": left["index"],
            "changed_metrics": _secondary_changes(left["features"], right["features"]),
        }
        for left, right in pairwise(reports)
    ]
    climax = next(item for item in reports if item["role"] == "climax")
    climax_support = [
        metric
        for metric in ACTIVITY_METRICS
        if all(
            float(climax["features"][metric]) > float(other["features"][metric])
            for other in reports
            if other is not climax
        )
    ]
    later = reports[climax["index"] + 1 :]
    post_climax = later[-1]
    decline_support = [
        metric
        for metric in ACTIVITY_METRICS
        if float(post_climax["features"][metric]) < float(climax["features"][metric])
    ]
    opening_materials = {
        use.material_id
        for use in composition.form[
            composition.parts[0].start_use_index : composition.parts[0].end_use_index
        ]
    }
    later_materials = {
        use.material_id
        for part in composition.parts[climax["index"] + 1 :]
        for use in composition.form[part.start_use_index : part.end_use_index]
    }
    reused = sorted(opening_materials & later_materials)
    phrase_by_id = {phrase.phrase_id: phrase for phrase in composition.phrases}
    phrase_reports = []
    for index, phrase in enumerate(composition.phrases):
        material = _expanded_use_range_material(
            composition,
            phrase.phrase_id,
            phrase.start_use_index,
            phrase.end_use_index,
        )
        phrase_materials = {
            use.material_id
            for use in composition.form[phrase.start_use_index : phrase.end_use_index]
            if use.role != "transition"
        }
        source_materials: set[str] = set()
        if phrase.derived_from is not None:
            source = phrase_by_id[phrase.derived_from]
            source_materials = {
                use.material_id
                for use in composition.form[source.start_use_index : source.end_use_index]
                if use.role != "transition"
            }
        phrase_reports.append(
            {
                "index": index,
                "phrase_id": phrase.phrase_id,
                "part_id": phrase.part_id,
                "role": phrase.role,
                "derived_from": phrase.derived_from,
                "variation_kind": phrase.variation_kind,
                "start_use_index": phrase.start_use_index,
                "end_use_index": phrase.end_use_index,
                "duration_ms": material.duration_ms,
                "features": _features(material),
                "reused_source_material_ids": sorted(phrase_materials & source_materials),
                "derived_material_ids": sorted(
                    material_id
                    for material_id in phrase_materials
                    if composition.material_by_id[material_id].derived_from in source_materials
                ),
            }
        )
    issues = []
    if len(climax_support) < 2:
        issues.append("declared climax has fewer than two actual activity maxima")
    if len(decline_support) < 2:
        issues.append("post-climax part declines on fewer than two activity metrics")
    if not reused:
        issues.append("no opening material is reused after the climax")
    return {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "parts": reports,
        "phrases": phrase_reports,
        "boundaries": boundaries,
        "climax": {
            "part_id": climax["part_id"],
            "supporting_metrics": climax_support,
            "passes": len(climax_support) >= 2,
        },
        "post_climax_decline": {
            "part_id": post_climax["part_id"],
            "supporting_metrics": decline_support,
            "passes": len(decline_support) >= 2,
        },
        "return": {
            "reused_material_ids": reused,
            "passes": bool(reused),
        },
    }
