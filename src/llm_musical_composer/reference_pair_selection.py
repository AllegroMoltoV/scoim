"""制御値が近く、匿名生成目標が分離した参照曲対を決定的に選ぶ。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from llm_musical_composer.reference_generation_target import (
    ReferenceGenerationTargetError,
    build_reference_generation_target,
)
from llm_musical_composer.reference_profile import (
    FEATURE_GROUPS,
    copy_fingerprint_similarity,
    profile_distances,
)
from llm_musical_composer.run_state import (
    atomic_write_bytes,
    atomic_write_json,
    sha256_file,
    sha256_json,
)

CONTROL_LABELS = ("あかるさ", "高さ", "発音頻度")
MAXIMUM_CONTROL_DIFFERENCE = 0.1


class ReferencePairSelectionError(ValueError):
    """検証済み成果物から識別可能な参照曲対を選べない。"""


@dataclass(frozen=True)
class ReferencePairSelection:
    """参照曲対の出所、選択根拠、決定的fingerprint。"""

    artifact: dict[str, Any]
    sha256: str


def _round(value: float) -> float:
    return round(float(value), 8)


def _number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReferencePairSelectionError(f"finite number required: {location}")
    result = float(value)
    if not math.isfinite(result):
        raise ReferencePairSelectionError(f"finite number required: {location}")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReferencePairSelectionError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise ReferencePairSelectionError(f"JSON object required: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ReferencePairSelectionError(f"cannot read JSONL: {path}") from error
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ReferencePairSelectionError(
                f"invalid JSONL at {path}:{line_number}"
            ) from error
        if not isinstance(value, dict):
            raise ReferencePairSelectionError(
                f"JSONL object required at {path}:{line_number}"
            )
        result.append(value)
    return result


def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row.get("name")
        if not isinstance(name, str) or not name:
            raise ReferencePairSelectionError(f"{label} name is invalid")
        if name in result:
            raise ReferencePairSelectionError(f"duplicate {label}: {name}")
        result[name] = row
    return result


def descriptor_separation(
    first: dict[str, Any], second: dict[str, Any]
) -> dict[str, Any]:
    """二つの匿名目標について、中心方向と近傍球の分離を返す。"""

    item_id = first.get("id")
    if not isinstance(item_id, str) or second.get("id") != item_id or "." not in item_id:
        raise ReferencePairSelectionError("descriptor ids do not match")
    kind = first.get("kind")
    if kind not in {"scalar", "distribution"} or second.get("kind") != kind:
        raise ReferencePairSelectionError(f"descriptor kinds do not match: {item_id}")
    first_center = first.get("neighborhood_center")
    second_center = second.get("neighborhood_center")
    if (
        not isinstance(first_center, list)
        or not isinstance(second_center, list)
        or not first_center
        or len(first_center) != len(second_center)
    ):
        raise ReferencePairSelectionError(f"descriptor centers do not match: {item_id}")
    left = [_number(value, f"{item_id}:first-center") for value in first_center]
    right = [_number(value, f"{item_id}:second-center") for value in second_center]
    if kind == "scalar" and len(left) != 1:
        raise ReferencePairSelectionError(f"scalar descriptor has multiple values: {item_id}")
    first_radius = _number(first.get("neighborhood_radius"), f"{item_id}:first-radius")
    second_radius = _number(second.get("neighborhood_radius"), f"{item_id}:second-radius")
    if first_radius < 0 or second_radius < 0:
        raise ReferencePairSelectionError(f"descriptor radius is negative: {item_id}")
    delta = [_round(after - before) for before, after in zip(left, right, strict=True)]
    distance = sum(abs(value) for value in delta)
    if kind == "distribution":
        distance /= 2
    distance = _round(distance)
    radius_sum = _round(first_radius + second_radius)
    margin = _round(distance - radius_sum)
    return {
        "id": item_id,
        "group": item_id.split(".", 1)[0],
        "kind": kind,
        "center_delta": delta,
        "center_distance": distance,
        "radius_sum": radius_sum,
        "separation_margin": margin,
        "separated": margin > 0,
    }


def _flatten_targets(target: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stage_targets = target.get("stage_targets")
    if not isinstance(stage_targets, dict):
        raise ReferencePairSelectionError("prompt target has no stage targets")
    result: dict[str, dict[str, Any]] = {}
    for values in stage_targets.values():
        if not isinstance(values, list):
            raise ReferencePairSelectionError("prompt stage target must be an array")
        for value in values:
            if not isinstance(value, dict) or not isinstance(value.get("id"), str):
                raise ReferencePairSelectionError("prompt descriptor is invalid")
            if value["id"] in result:
                raise ReferencePairSelectionError(f"duplicate prompt descriptor: {value['id']}")
            result[value["id"]] = value
    return result


def _group_medians(summary: dict[str, Any]) -> dict[str, float]:
    try:
        groups = summary["controls"]["groups"]
    except (KeyError, TypeError) as error:
        raise ReferencePairSelectionError("reference summary group controls are missing") from error
    return {
        group: _number(
            groups[group]["dispersion"]["median_distance"],
            f"{group}:median-distance",
        )
        for group in FEATURE_GROUPS
    }


def _preflight_target(
    reference_dir: Path,
    control_dir: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    try:
        selected = summary["default_reference"]["selected"]
        name = selected["name"]
        source_hash = manifest["inputs"][name]
    except (KeyError, TypeError) as error:
        raise ReferencePairSelectionError("validated default reference is missing") from error
    build_reference_generation_target(
        {
            "reference": {
                "state": "resolved",
                "name": name,
                "sha256": source_hash,
                "selection_method": "automatic_medoid",
            },
            "controls": {},
        },
        reference_dir=reference_dir,
        control_dir=control_dir,
    )


def select_reference_pair(
    reference_dir: Path,
    control_dir: Path,
) -> ReferencePairSelection:
    """検証済み232曲から、初回smoke testに使う参照曲対を選ぶ。"""

    reference_dir = Path(reference_dir)
    control_dir = Path(control_dir)
    paths = {
        "reference_manifest": reference_dir / "manifest.json",
        "reference_records": reference_dir / "files.jsonl",
        "reference_summary": reference_dir / "summary.json",
        "control_manifest": control_dir / "manifest.json",
        "control_records": control_dir / "records.jsonl",
        "control_summary": control_dir / "summary.json",
    }
    reference_manifest = _read_json(paths["reference_manifest"])
    reference_summary = _read_json(paths["reference_summary"])
    _preflight_target(reference_dir, control_dir, reference_manifest, reference_summary)
    profile_rows = _read_jsonl(paths["reference_records"])
    control_rows = _read_jsonl(paths["control_records"])
    profiles = _index(profile_rows, "reference profile")
    controls = _index(control_rows, "control record")
    if set(profiles) != set(controls):
        raise ReferencePairSelectionError("reference and control record names differ")
    if any(row.get("status") != "pass" for row in profiles.values()):
        raise ReferencePairSelectionError("reference profile has a non-passing record")
    medians = _group_medians(reference_summary)
    targets: dict[str, dict[str, dict[str, Any]]] = {}
    target_errors: dict[str, str] = {}

    def target_for(name: str) -> dict[str, dict[str, Any]] | None:
        if name in target_errors:
            return None
        if name not in targets:
            try:
                source_hash = reference_manifest["inputs"][name]
                target = build_reference_generation_target(
                    {
                        "reference": {
                            "state": "resolved",
                            "name": name,
                            "sha256": source_hash,
                            "selection_method": "explicit",
                        },
                        "controls": {},
                    },
                    reference_dir=reference_dir,
                    control_dir=control_dir,
                )
                targets[name] = _flatten_targets(target.prompt_target)
            except (KeyError, ReferenceGenerationTargetError) as error:
                target_errors[name] = str(error)
                return None
        return targets[name]

    eligible: list[dict[str, Any]] = []
    for first_name, second_name in combinations(
        sorted(profiles, key=lambda value: (value.casefold(), value)), 2
    ):
        first_control = controls[first_name].get("normalized")
        second_control = controls[second_name].get("normalized")
        if not isinstance(first_control, dict) or not isinstance(second_control, dict):
            raise ReferencePairSelectionError("normalized control values are missing")
        control_differences = {
            label: _round(
                abs(
                    _number(first_control.get(label), f"{first_name}:{label}")
                    - _number(second_control.get(label), f"{second_name}:{label}")
                )
            )
            for label in CONTROL_LABELS
        }
        if max(control_differences.values()) > MAXIMUM_CONTROL_DIFFERENCE:
            continue
        distances = profile_distances(
            profiles[first_name]["profile"], profiles[second_name]["profile"]
        )
        if any(distances[group] < medians[group] for group in FEATURE_GROUPS):
            continue
        first_target = target_for(first_name)
        second_target = target_for(second_name)
        if first_target is None or second_target is None:
            continue
        if set(first_target) != set(second_target):
            raise ReferencePairSelectionError("reference prompt descriptor names differ")
        descriptors = [
            descriptor_separation(first_target[item_id], second_target[item_id])
            for item_id in sorted(first_target)
        ]
        separated = [item for item in descriptors if item["separated"]]
        separated_groups = sorted({item["group"] for item in separated})
        eligible.append(
            {
                "names": [first_name, second_name],
                "reference_sha256": [
                    reference_manifest["inputs"][first_name],
                    reference_manifest["inputs"][second_name],
                ],
                "controls": {
                    first_name: {label: first_control[label] for label in CONTROL_LABELS},
                    second_name: {label: second_control[label] for label in CONTROL_LABELS},
                },
                "control_differences": control_differences,
                "control_l2": _round(
                    math.sqrt(sum(value**2 for value in control_differences.values()))
                ),
                "group_distances": distances,
                "group_median_multiples": {
                    group: _round(distances[group] / medians[group])
                    for group in FEATURE_GROUPS
                },
                "descriptors": descriptors,
                "separated_descriptor_count": len(separated),
                "separated_groups": separated_groups,
                "separation_margin_sum": _round(
                    sum(item["separation_margin"] for item in separated)
                ),
                "copy": {
                    "exact": profiles[first_name]["copy_fingerprint"]["sequence_sha256"]
                    == profiles[second_name]["copy_fingerprint"]["sequence_sha256"],
                    "similarity": copy_fingerprint_similarity(
                        profiles[first_name]["copy_fingerprint"],
                        profiles[second_name]["copy_fingerprint"],
                    ),
                },
            }
        )
    target_eligible_count = len(eligible)
    eligible = [
        item
        for item in eligible
        if set(item["separated_groups"]) == set(FEATURE_GROUPS) and not item["copy"]["exact"]
    ]
    if not eligible:
        raise ReferencePairSelectionError("no reference pair has separable prompt targets")
    eligible.sort(
        key=lambda item: (
            -item["separated_descriptor_count"],
            -item["separation_margin_sum"],
            -min(item["group_median_multiples"].values()),
            item["control_l2"],
            tuple((name.casefold(), name) for name in item["names"]),
        )
    )
    selected = eligible[0]
    artifact = {
        "schema_version": 1,
        "selection_version": "reference-variance-pair-v1",
        "status": "selected",
        "criteria": {
            "maximum_control_difference": MAXIMUM_CONTROL_DIFFERENCE,
            "minimum_group_distances": medians,
            "required_separated_groups": list(FEATURE_GROUPS),
            "copy_exact_action": "exclude",
        },
        "input_sha256": {name: sha256_file(path) for name, path in sorted(paths.items())},
        "eligible_pair_count": target_eligible_count,
        "fully_separated_pair_count": len(eligible),
        "target_errors": dict(sorted(target_errors.items())),
        "selected": selected,
    }
    return ReferencePairSelection(artifact, sha256_json(artifact))


def write_reference_pair_selection(
    reference_dir: Path,
    control_dir: Path,
    output_path: Path,
) -> ReferencePairSelection:
    """参照曲対の成果物と、その内容fingerprintを確定書き込みする。"""

    result = select_reference_pair(reference_dir, control_dir)
    output_path = Path(output_path)
    atomic_write_json(output_path, result.artifact)
    atomic_write_bytes(
        output_path.with_suffix(".sha256"), (result.sha256 + "\n").encode("ascii")
    )
    return result
