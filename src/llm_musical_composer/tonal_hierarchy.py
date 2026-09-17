"""明示的なあかるさの主音相対調性契約を段階ごとに検査する。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from llm_musical_composer.harmonic_skeleton import HarmonicSkeletonV0
from llm_musical_composer.performance_pipeline import PiecePlan


def _not_applicable() -> dict[str, object]:
    return {"status": "not_applicable", "passes": True, "failures": []}


def evaluate_piece_plan_tonal_hierarchy(
    plan: PiecePlan,
    target: Mapping[str, Any],
) -> dict[str, object]:
    """PiecePlanが明示3値のmode契約へ一致するかを返す。"""

    if target.get("status") != "specified":
        return _not_applicable()
    failures = []
    if plan.mode != target.get("plan_mode"):
        failures.append("plan-mode")
    if plan.ending_intent != "tonic":
        failures.append("ending-intent")
    return {
        "status": "assessed",
        "passes": not failures,
        "failures": failures,
        "plan_mode": plan.mode,
        "required_plan_mode": target.get("plan_mode"),
        "ending_intent": plan.ending_intent,
    }


def evaluate_brightness_resolution(
    achieved_normalized: float,
    target: Mapping[str, Any],
) -> dict[str, object]:
    """RenderedPerformanceのあかるさが明示3値の受入帯へ入るかを返す。"""

    if target.get("status") != "specified":
        return _not_applicable()
    target_value = target.get("target")
    if not isinstance(target_value, Mapping):
        raise ValueError("tonal hierarchy brightness target is invalid")
    raw_range = target_value.get("acceptable_normalized_range")
    if (
        not isinstance(raw_range, list)
        or len(raw_range) != 2
        or float(raw_range[0]) > float(raw_range[1])
    ):
        raise ValueError("tonal hierarchy brightness range is invalid")
    acceptable = [float(raw_range[0]), float(raw_range[1])]
    achieved = float(achieved_normalized)
    target_normalized = float(target_value["normalized"])
    passes = acceptable[0] <= achieved <= acceptable[1]
    return {
        "status": "achieved" if passes else "not_reached",
        "passes": passes,
        "requested": int(target["requested"]),
        "target_normalized": target_normalized,
        "achieved_normalized": achieved,
        "normalized_residual": achieved - target_normalized,
        "acceptable_normalized_range": acceptable,
    }


def _materials_by_role(plan: PiecePlan, roles: tuple[str, ...]) -> dict[str, set[str]]:
    by_id = {node.node_id: node for node in plan.nodes}
    result = {role: set() for role in roles}
    for node in plan.nodes:
        if node.score_material_id is None:
            continue
        current = node
        seen: set[str] = set()
        while True:
            if current.node_id in seen:
                break
            seen.add(current.node_id)
            if current.role in result:
                result[current.role].add(node.score_material_id)
            if current.parent_id is None:
                break
            current = by_id[current.parent_id]
    return result


def _harmony_key(tonic: int, root: int, quality: str) -> tuple[int, str]:
    return (root - tonic) % 12, quality


def evaluate_harmonic_skeleton_tonal_hierarchy(
    plan: PiecePlan,
    skeleton: HarmonicSkeletonV0,
    target: Mapping[str, Any],
) -> dict[str, object]:
    """和声骨格が指定旋法、特徴和音、終止契約へ一致するかを返す。"""

    if target.get("status") != "specified":
        return _not_applicable()
    allowed_raw = target.get("allowed_harmonies")
    characteristic = target.get("characteristic_harmony")
    ending = target.get("ending")
    if not isinstance(allowed_raw, list) or not isinstance(ending, Mapping):
        raise ValueError("tonal hierarchy harmony contract is invalid")
    allowed = {
        (int(item["root_degree"]), str(item["quality"]))
        for item in allowed_raw
        if isinstance(item, Mapping)
    }
    if len(allowed) != len(allowed_raw):
        raise ValueError("tonal hierarchy allowed harmonies are invalid")

    required_roles = (
        tuple(str(role) for role in characteristic.get("required_roles", ()))
        if isinstance(characteristic, Mapping)
        else ()
    )
    material_roles = _materials_by_role(plan, (*required_roles, "release"))
    release_ids = material_roles["release"]
    by_material = {material.material_id: material for material in skeleton.materials}
    outside = []
    for material in skeleton.materials:
        if material.material_id in release_ids:
            continue
        for harmony in material.harmonies:
            key = _harmony_key(
                plan.tonal_center,
                harmony.root_pitch_class,
                harmony.quality,
            )
            if key not in allowed:
                outside.append(
                    {
                        "material_id": material.material_id,
                        "harmony_id": harmony.harmony_id,
                        "root_degree": key[0],
                        "quality": key[1],
                    }
                )

    characteristic_counts: dict[str, int] = {}
    if isinstance(characteristic, Mapping):
        characteristic_key = (
            int(characteristic["root_degree"]),
            str(characteristic["quality"]),
        )
        for role in required_roles:
            characteristic_counts[role] = sum(
                _harmony_key(
                    plan.tonal_center,
                    harmony.root_pitch_class,
                    harmony.quality,
                )
                == characteristic_key
                for material_id in material_roles[role]
                for harmony in by_material[material_id].harmonies
            )

    ending_key = (int(ending["root_degree"]), str(ending["quality"]))
    ending_harmonies = [
        harmony
        for material_id in release_ids
        for harmony in by_material[material_id].harmonies
    ]
    ending_passes = bool(ending_harmonies) and all(
        _harmony_key(plan.tonal_center, harmony.root_pitch_class, harmony.quality)
        == ending_key
        for harmony in ending_harmonies
    )
    failures = []
    if outside:
        failures.append("outside-harmony")
    if isinstance(characteristic, Mapping):
        minimum = int(characteristic["minimum_per_role"])
        failures.extend(
            f"characteristic-harmony:{role}"
            for role, count in characteristic_counts.items()
            if count < minimum
        )
    if not ending_passes:
        failures.append("ending-harmony")
    return {
        "status": "assessed",
        "passes": not failures,
        "failures": failures,
        "scale_policy": target.get("scale_policy"),
        "outside_harmonies": outside,
        "characteristic_counts": characteristic_counts,
        "ending_passes": ending_passes,
    }
