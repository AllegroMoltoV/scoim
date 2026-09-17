"""宣言した区間差が実際の音符へ現れているかを検査する。"""

from __future__ import annotations

import statistics
from collections import Counter
from itertools import pairwise
from typing import Any

from llm_musical_composer.composition_ir import Composition, Material

CLUSTERED_MIN_RATIO = 0.55
DISTRIBUTED_MAX_RATIO = 0.35
MIN_DENSITY_RATIO = 1.35
MIN_POLYPHONY_RATIO = 1.35
MIN_REGISTER_SHIFT = 6.0
MIN_VELOCITY_SHIFT = 8.0
MIN_CLUSTERED_RATIO_SHIFT = 0.2


def _round(value: float) -> float:
    return round(float(value), 8)


def _features(material: Material) -> dict[str, float | str]:
    if not material.notes:
        return {
            "note_density": 0.0,
            "velocity_median": 0.0,
            "pitch_median": 0.0,
            "polyphony_mean": 0.0,
            "clustered_note_ratio": 0.0,
            "observed_attack_style": "ambiguous",
        }
    onset_counts = Counter(note.at_ms for note in material.notes)
    clustered_count = sum(1 for note in material.notes if onset_counts[note.at_ms] >= 3)
    clustered_ratio = clustered_count / len(material.notes)
    if clustered_ratio >= CLUSTERED_MIN_RATIO:
        observed_style = "clustered"
    elif clustered_ratio <= DISTRIBUTED_MAX_RATIO:
        observed_style = "distributed"
    else:
        observed_style = "ambiguous"
    return {
        "note_density": _round(len(material.notes) / (material.duration_ms / 1000)),
        "velocity_median": _round(statistics.median(note.velocity for note in material.notes)),
        "pitch_median": _round(statistics.median(note.pitch for note in material.notes)),
        "polyphony_mean": _round(
            sum(note.duration_ms for note in material.notes) / material.duration_ms
        ),
        "clustered_note_ratio": _round(clustered_ratio),
        "observed_attack_style": observed_style,
    }


def _ratio(first: float, second: float) -> float:
    lower = min(first, second)
    if lower <= 0:
        return float("inf") if max(first, second) > 0 else 1.0
    return max(first, second) / lower


def _secondary_changes(left: dict[str, float | str], right: dict[str, float | str]) -> list[str]:
    changes: list[str] = []
    if _ratio(float(left["note_density"]), float(right["note_density"])) >= MIN_DENSITY_RATIO:
        changes.append("note_density")
    if abs(float(left["pitch_median"]) - float(right["pitch_median"])) >= MIN_REGISTER_SHIFT:
        changes.append("pitch_median")
    if abs(float(left["velocity_median"]) - float(right["velocity_median"])) >= MIN_VELOCITY_SHIFT:
        changes.append("velocity_median")
    if _ratio(float(left["polyphony_mean"]), float(right["polyphony_mean"])) >= (
        MIN_POLYPHONY_RATIO
    ):
        changes.append("polyphony_mean")
    return changes


def _higher_energy_support(
    left_energy: int,
    right_energy: int,
    left: dict[str, float | str],
    right: dict[str, float | str],
) -> list[str]:
    if left_energy == right_energy:
        return []
    higher, lower = (left, right) if left_energy > right_energy else (right, left)
    return [
        name
        for name in ("note_density", "velocity_median", "polyphony_mean")
        if float(higher[name]) > float(lower[name])
    ]


def evaluate_section_contrast(composition: Composition) -> dict[str, Any]:
    """区間契約、実音上の場面差、最高潮を合算せず返す。"""
    if any(section.role is None or section.energy is None for section in composition.form):
        return {
            "status": "unable_to_investigate",
            "issues": ["section contract is missing"],
            "sections": [],
            "boundaries": [],
            "climax": None,
        }

    materials = composition.material_by_id
    sections: list[dict[str, Any]] = []
    issues: list[str] = []
    for index, use in enumerate(composition.form):
        features = _features(materials[use.material_id])
        sections.append(
            {
                "index": index,
                "material_id": use.material_id,
                "role": use.role,
                "energy": use.energy,
                "declared_attack_style": use.attack_style,
                "features": features,
            }
        )

    boundaries: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(pairwise(sections)):
        secondary = _secondary_changes(left["features"], right["features"])
        clustered_shift = abs(
            float(left["features"]["clustered_note_ratio"])
            - float(right["features"]["clustered_note_ratio"])
        )
        attack_pattern_changed = clustered_shift >= MIN_CLUSTERED_RATIO_SHIFT
        energy_support = _higher_energy_support(
            int(left["energy"]),
            int(right["energy"]),
            left["features"],
            right["features"],
        )
        energy_direction_matches = left["energy"] == right["energy"] or len(energy_support) >= 2
        if not energy_direction_matches:
            issues.append(f"boundary {index} energy direction has fewer than two supports")
        boundaries.append(
            {
                "after_section": index,
                "clustered_note_ratio_shift": _round(clustered_shift),
                "attack_pattern_changed": attack_pattern_changed,
                "secondary_changes": secondary,
                "energy_direction_supporting_metrics": energy_support,
                "energy_direction_matches": energy_direction_matches,
                "scene_change_pass": (attack_pattern_changed and bool(secondary)),
            }
        )
    if not any(boundary["scene_change_pass"] for boundary in boundaries):
        issues.append("no boundary has both an attack pattern and a secondary feature change")

    climax_index = next(
        index for index, section in enumerate(composition.form) if section.role == "climax"
    )
    climax_features = sections[climax_index]["features"]
    climax_support = [
        name
        for name in ("note_density", "velocity_median", "polyphony_mean")
        if all(
            float(climax_features[name]) > float(section["features"][name])
            for index, section in enumerate(sections)
            if index != climax_index
        )
    ]
    if len(climax_support) < 2:
        issues.append("climax has fewer than two actual activity maxima")
    climax = {
        "section_index": climax_index,
        "supporting_metrics": climax_support,
        "passes": len(climax_support) >= 2,
    }
    return {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "sections": sections,
        "boundaries": boundaries,
        "climax": climax,
    }
