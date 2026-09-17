"""検証済み参照成果物から、匿名の段階別生成目標を構築する。"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_musical_composer.control_reference_baseline import AXIS_VERSION
from llm_musical_composer.run_state import sha256_file, sha256_json

TARGET_VERSION = "reference-generation-target-v7"
LEGACY_TARGET_VERSION = "reference-generation-target-v1"
REGISTER_ENVELOPE_POLICY_ID = "melody-containing-reference-span-v2"

CONTROL_SPECS = {
    "brightness": "あかるさ",
    "height": "高さ",
    "attack_frequency": "発音頻度",
}

DESCRIPTOR_SPECS = (
    ("score_spec", "performance_texture", "notes_per_attack"),
    ("score_spec", "performance_texture", "relative_register"),
    ("score_spec", "performance_texture", "pitch_range"),
    ("score_spec", "rhythm_time", "attack_size"),
    ("score_spec", "pitch_harmony", "transition_interval_class"),
    ("score_spec", "pitch_harmony", "transition_direction"),
    ("score_spec", "pitch_harmony", "transition_magnitude"),
    ("score_spec", "pitch_harmony", "vertical_interval_class"),
    ("score_spec", "pitch_harmony", "monophonic_attack_ratio"),
    ("performance_spec", "performance_texture", "velocity"),
    ("performance_spec", "performance_texture", "pedal_on_ratio"),
    ("performance_spec", "performance_texture", "attack_under_pedal_ratio"),
    ("performance_spec", "performance_texture", "pedal_changes_per_attack"),
    ("rendered_surface", "performance_texture", "polyphony_duration"),
    ("rendered_surface", "rhythm_time", "ioi_ratio"),
    ("rendered_surface", "rhythm_time", "duration_ratio"),
    ("rendered_surface", "rhythm_time", "rest_ratio"),
)


class ReferenceGenerationTargetError(ValueError):
    """参照生成目標を安全に構築できない場合に送出する。"""


@dataclass(frozen=True)
class ReferenceGenerationTarget:
    artifact: dict[str, Any]
    sha256: str
    prompt_target: dict[str, Any]
    prompt_sha256: str


def _round(value: float) -> float:
    return round(float(value), 8)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReferenceGenerationTargetError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise ReferenceGenerationTargetError(f"JSON object required: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ReferenceGenerationTargetError(f"cannot read JSONL: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ReferenceGenerationTargetError(
                f"invalid JSONL at {path}:{line_number}"
            ) from error
        if not isinstance(value, dict):
            raise ReferenceGenerationTargetError(f"JSONL object required at {path}:{line_number}")
        rows.append(value)
    return rows


def _verify_output_hashes(
    directory: Path, manifest: Mapping[str, object], names: Sequence[str]
) -> None:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ReferenceGenerationTargetError(f"manifest outputs are invalid: {directory}")
    for name in names:
        expected = outputs.get(name)
        path = directory / name
        if not isinstance(expected, str) or not path.is_file():
            raise ReferenceGenerationTargetError(f"manifest output is missing: {path}")
        if sha256_file(path).casefold() != expected.casefold():
            raise ReferenceGenerationTargetError(f"hash mismatch: {path}")


def _number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReferenceGenerationTargetError(f"finite number required: {location}")
    result = float(value)
    if not math.isfinite(result):
        raise ReferenceGenerationTargetError(f"finite number required: {location}")
    return result


def _metric(record: Mapping[str, Any], group: str, metric: str) -> tuple[str, list[float]]:
    try:
        raw = record["profile"]["feature_groups"][group]["metrics"][metric]
        kind = raw["kind"]
        raw_values = raw["values"]
    except (KeyError, TypeError) as error:
        raise ReferenceGenerationTargetError(
            f"reference metric is missing: {group}.{metric}"
        ) from error
    if kind not in {"scalar", "distribution"} or not isinstance(raw_values, list):
        raise ReferenceGenerationTargetError(f"reference metric is invalid: {group}.{metric}")
    values = [_number(value, f"{group}.{metric}") for value in raw_values]
    if not values or (kind == "scalar" and len(values) != 1):
        raise ReferenceGenerationTargetError(f"reference metric shape is invalid: {group}.{metric}")
    if kind == "distribution":
        if any(value < 0 for value in values) or sum(values) <= 0:
            raise ReferenceGenerationTargetError(
                f"reference distribution is invalid: {group}.{metric}"
            )
        total = sum(values)
        values = [value / total for value in values]
    return kind, values


def _center_and_radius(kind: str, rows: Sequence[list[float]]) -> tuple[list[float], float]:
    dimensions = {len(row) for row in rows}
    if len(dimensions) != 1:
        raise ReferenceGenerationTargetError("reference neighborhood metric dimensions differ")
    center = [statistics.fmean(row[index] for row in rows) for index in range(len(rows[0]))]
    if kind == "distribution":
        total = sum(center)
        if total <= 0:
            raise ReferenceGenerationTargetError("reference neighborhood cannot be normalized")
        center = [value / total for value in center]
        radius = max(
            sum(abs(value - mean) for value, mean in zip(row, center, strict=True)) / 2
            for row in rows
        )
    else:
        radius = max(abs(row[0] - center[0]) for row in rows)
    return [_round(value) for value in center], _round(radius)


def _ratio_mapping(
    record: Mapping[str, Any],
    field: str,
    *,
    minimum_key: int,
) -> dict[int, float]:
    try:
        raw = record["diagnostics"]["overlap"][field]
    except (KeyError, TypeError) as error:
        raise ReferenceGenerationTargetError(
            f"control overlap diagnostic is missing: {field}"
        ) from error
    if not isinstance(raw, Mapping) or not raw:
        raise ReferenceGenerationTargetError(
            f"control overlap diagnostic is invalid: {field}"
        )
    result: dict[int, float] = {}
    for raw_key, raw_value in raw.items():
        try:
            key = int(raw_key)
        except (TypeError, ValueError) as error:
            raise ReferenceGenerationTargetError(
                f"control overlap diagnostic key is invalid: {field}"
            ) from error
        if key < minimum_key or str(key) != str(raw_key):
            raise ReferenceGenerationTargetError(
                f"control overlap diagnostic key is invalid: {field}"
            )
        value = _number(raw_value, f"diagnostics.overlap.{field}.{raw_key}")
        if value < 0:
            raise ReferenceGenerationTargetError(
                f"control overlap diagnostic is negative: {field}"
            )
        result[key] = value
    total = sum(result.values())
    if total <= 0 or not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ReferenceGenerationTargetError(
            f"control overlap diagnostic is not normalized: {field}"
        )
    return result


def _binned_overlap_distribution(
    record: Mapping[str, Any],
    field: str,
    *,
    minimum_key: int,
    upper_bin: int,
) -> tuple[list[float], dict[int, float]]:
    raw = _ratio_mapping(record, field, minimum_key=minimum_key)
    values = [0.0] * (upper_bin - minimum_key + 1)
    for key, value in raw.items():
        values[min(key, upper_bin) - minimum_key] += value
    total = sum(values)
    return ([_round(value / total) for value in values], raw)


def _overlap_scalar(record: Mapping[str, Any], field: str) -> float:
    try:
        raw = record["diagnostics"]["overlap"][field]
    except (KeyError, TypeError) as error:
        raise ReferenceGenerationTargetError(
            f"control overlap diagnostic is missing: {field}"
        ) from error
    return _number(raw, f"diagnostics.overlap.{field}")


def _control_axis_value(record: Mapping[str, Any], field: str, label: str) -> float:
    try:
        raw = record[field][label]
    except (KeyError, TypeError) as error:
        raise ReferenceGenerationTargetError(
            f"control baseline value is missing: {field}.{label}"
        ) from error
    return _number(raw, f"{field}.{label}")


def _attack_frequency_target(
    anchor: Mapping[str, Any],
    neighborhood: Sequence[Mapping[str, Any]],
    *,
    control: Mapping[str, Any],
    control_summary: Mapping[str, Any],
    legacy_reference: bool = False,
) -> dict[str, Any]:
    raw_anchor = _control_axis_value(anchor, "raw", "発音頻度")
    normalized_anchor = _control_axis_value(anchor, "normalized", "発音頻度")
    raw_neighbors = [
        _control_axis_value(record, "raw", "発音頻度") for record in neighborhood
    ]
    normalized_neighbors = [
        _control_axis_value(record, "normalized", "発音頻度")
        for record in neighborhood
    ]
    if not raw_neighbors or min(raw_neighbors) <= 0:
        raise ReferenceGenerationTargetError(
            "attack frequency neighborhood must contain positive values"
        )
    duration_seconds = 180
    raw_minimum = min(raw_neighbors)
    raw_maximum = max(raw_neighbors)
    reference_target = {
        "id": "attack_frequency",
        "kind": "scalar",
        "generation_stage": "score_and_performance",
        "grouping": {
            "anchor": "first_note_on",
            "chain": False,
            "tolerance_ms": 30,
        },
        "anchor": {
            "normalized": _round(normalized_anchor),
            "raw_groups_per_second": _round(raw_anchor),
        },
        "neighborhood": {
            "normalized": {
                "center": _round(statistics.fmean(normalized_neighbors)),
                "minimum": _round(min(normalized_neighbors)),
                "maximum": _round(max(normalized_neighbors)),
            },
            "raw_groups_per_second": {
                "center": _round(statistics.fmean(raw_neighbors)),
                "minimum": _round(raw_minimum),
                "maximum": _round(raw_maximum),
            },
        },
        "groups_for_180_seconds": {
            "anchor_half_up": math.floor(raw_anchor * duration_seconds + 0.5),
            "minimum_ceil": math.ceil(raw_minimum * duration_seconds),
            "maximum_floor": math.floor(raw_maximum * duration_seconds),
        },
        "strict_score_budget": {
            "groups": math.ceil(raw_minimum * duration_seconds),
            "source": "neighborhood_minimum_ceil",
        },
        "promotion_range": {
            "minimum_groups_per_second": _round(raw_minimum),
            "maximum_groups_per_second": _round(raw_maximum),
            "inclusive": True,
        },
        "source_observation_ids": [
            "control_baseline.raw.発音頻度",
            "control_baseline.normalized.発音頻度",
        ],
        "instructions": [
            "発音群数と発音群サイズを共同予算として扱う",
            "全体BPMだけで達成しない",
            "30 ms以内の打鍵を分裂させて発音回数を水増ししない",
        ],
        "reachability": {"status": "unverified"},
    }
    if control.get("source") != "explicit" or legacy_reference:
        return reference_target

    requested = _number(control.get("value"), "controls.attack_frequency.value")
    try:
        axis = control_summary["axes"]["発音頻度"]
        axis_minimum = _number(axis["minimum"], "axes.発音頻度.minimum")
        axis_maximum = _number(axis["maximum"], "axes.発音頻度.maximum")
    except (KeyError, TypeError) as error:
        raise ReferenceGenerationTargetError(
            "attack frequency corpus axis is missing"
        ) from error
    if not 0 < axis_minimum < axis_maximum:
        raise ReferenceGenerationTargetError(
            "attack frequency corpus axis is invalid"
        )
    raw_target = axis_minimum + ((requested + 1.0) / 2.0) * (
        axis_maximum - axis_minimum
    )
    strict_groups = math.floor(raw_target * duration_seconds + 0.5)
    if strict_groups <= 0:
        raise ReferenceGenerationTargetError(
            "explicit attack frequency score budget is empty"
        )
    return {
        **reference_target,
        "anchor": {
            "normalized": _round(requested),
            "raw_groups_per_second": _round(raw_target),
        },
        "groups_for_180_seconds": {
            "requested_half_up": strict_groups,
            "reference_anchor_half_up": reference_target[
                "groups_for_180_seconds"
            ]["anchor_half_up"],
        },
        "strict_score_budget": {
            "groups": strict_groups,
            "source": "explicit_corpus_min_max_half_up",
        },
        "promotion_range": {
            "minimum_groups_per_second": _round(
                (strict_groups - 0.5) / duration_seconds
            ),
            "maximum_groups_per_second": _round(
                (strict_groups + 0.5) / duration_seconds
            ),
            "inclusive": True,
        },
        "resolution": {
            "source": "explicit_corpus_min_max_v1",
            "duration_seconds": duration_seconds,
            "axis_minimum_groups_per_second": _round(axis_minimum),
            "axis_maximum_groups_per_second": _round(axis_maximum),
            "rounding": "half_up_before_canonical_round",
        },
    }


def _per_group_counts(values: Sequence[float], total: int) -> list[int]:
    raw = [value * total for value in values]
    counts = [math.floor(value) for value in raw]
    remaining = total - sum(counts)
    order = sorted(
        range(len(values)),
        key=lambda index: (-(raw[index] - counts[index]), index),
    )
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def _tonal_hierarchy_target(
    brightness: Mapping[str, Any],
    control_summary: Mapping[str, Any],
) -> dict[str, Any]:
    source = brightness.get("source")
    requested = _number(brightness.get("value"), "controls.brightness.value")
    if source != "explicit":
        return {
            "id": "tonal_hierarchy",
            "kind": "unverified_control",
            "generation_stage": "piece_plan_score_and_melody",
            "status": "unverified_continuous_reference",
            "requested_normalized": _round(requested),
            "reachability": {"status": "unverified"},
        }
    if requested not in {-1.0, 0.0, 1.0}:
        raise ReferenceGenerationTargetError("explicit brightness must be -1, 0, or 1")
    try:
        axis = control_summary["axes"]["あかるさ"]
        minimum = _number(axis["minimum"], "axes.あかるさ.minimum")
        maximum = _number(axis["maximum"], "axes.あかるさ.maximum")
    except (KeyError, TypeError) as error:
        raise ReferenceGenerationTargetError("brightness control baseline is missing") from error
    if not minimum < maximum:
        raise ReferenceGenerationTargetError("brightness control baseline is degenerate")
    level = int(requested)
    specs = {
        -1: {
            "plan_mode": "minor",
            "scale_policy": "minor_functional",
            "scale_degrees": [0, 2, 3, 5, 7, 8, 10],
            "optional_scale_degrees": [11],
            "allowed_harmonies": [
                (0, "minor"),
                (2, "diminished"),
                (3, "major"),
                (5, "minor"),
                (7, "minor"),
                (7, "major"),
                (8, "major"),
                (10, "major"),
                (11, "diminished"),
            ],
            "ending_quality": "minor",
            "acceptable": [-1.0, -0.5],
            "instructions": [
                "短調の主音、前属、属、回帰を曲全体で一貫させる",
                "区分ごとの長短調交替だけで暗さを作らない",
            ],
        },
        0: {
            "plan_mode": "minor",
            "scale_policy": "dorian",
            "scale_degrees": [0, 2, 3, 5, 7, 9, 10],
            "optional_scale_degrees": [],
            "allowed_harmonies": [
                (0, "minor"),
                (2, "minor"),
                (3, "major"),
                (5, "major"),
                (7, "minor"),
                (9, "diminished"),
                (10, "major"),
            ],
            "ending_quality": "minor",
            "acceptable": [-0.5, 0.5],
            "instructions": [
                "短主和音、長III、長IV、短v、長VIIを主音相対で使う",
                "statement、contrast、returnの各役割に長IVを少なくとも一度置く",
                "区分ごとの長短調交替ではなくDorianを曲全体で一貫させる",
            ],
        },
        1: {
            "plan_mode": "major",
            "scale_policy": "major_functional",
            "scale_degrees": [0, 2, 4, 5, 7, 9, 11],
            "optional_scale_degrees": [],
            "allowed_harmonies": [
                (0, "major"),
                (2, "minor"),
                (4, "minor"),
                (5, "major"),
                (7, "major"),
                (9, "minor"),
                (11, "diminished"),
            ],
            "ending_quality": "major",
            "acceptable": [0.5, 1.0],
            "instructions": [
                "長調の主音、前属、属、回帰を曲全体で一貫させる",
                "区分ごとの長短調交替だけで明るさを作らない",
            ],
        },
    }
    spec = specs[level]
    target_raw = minimum if level == -1 else maximum if level == 1 else (minimum + maximum) / 2
    return {
        "id": "tonal_hierarchy",
        "kind": "quality_rule",
        "generation_stage": "piece_plan_score_and_melody",
        "status": "specified",
        "requested": level,
        "target": {
            "raw": _round(target_raw),
            "normalized": float(level),
            "acceptable_normalized_range": spec["acceptable"],
        },
        "tonic_policy": "piece_plan_relative",
        "plan_mode": spec["plan_mode"],
        "scale_policy": spec["scale_policy"],
        "scale_degrees": spec["scale_degrees"],
        "optional_scale_degrees": spec["optional_scale_degrees"],
        "allowed_harmonies": [
            {"root_degree": degree, "quality": quality}
            for degree, quality in spec["allowed_harmonies"]
        ],
        "characteristic_harmony": (
            {
                "required_roles": ["statement", "contrast", "return"],
                "root_degree": 5,
                "quality": "major",
                "minimum_per_role": 1,
            }
            if level == 0
            else None
        ),
        "ending": {"root_degree": 0, "quality": spec["ending_quality"]},
        "instructions": spec["instructions"],
        "reachability": {"status": "unverified"},
    }


def _semantic_targets(
    anchor: Mapping[str, Any],
    neighborhood: Sequence[Mapping[str, Any]],
    stage_targets: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    artifact_controls: Mapping[str, Mapping[str, Any]],
    control_summary: Mapping[str, Any],
    legacy_reference_attack_frequency: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    attack_anchor, raw_attack_anchor = _binned_overlap_distribution(
        anchor,
        "attack_size_distribution",
        minimum_key=1,
        upper_bin=4,
    )
    attack_neighbors = [
        _binned_overlap_distribution(
            row,
            "attack_size_distribution",
            minimum_key=1,
            upper_bin=4,
        )[0]
        for row in neighborhood
    ]
    attack_center, attack_radius = _center_and_radius(
        "distribution", attack_neighbors
    )
    notes_anchor = _overlap_scalar(anchor, "notes_per_attack")
    expected_notes = sum(size * ratio for size, ratio in raw_attack_anchor.items())
    if not math.isclose(notes_anchor, expected_notes, abs_tol=1e-6):
        raise ReferenceGenerationTargetError(
            "30 ms attack-size distribution conflicts with notes per attack"
        )
    neighbor_notes = [
        [_overlap_scalar(row, "notes_per_attack")] for row in neighborhood
    ]
    notes_center, notes_radius = _center_and_radius("scalar", neighbor_notes)

    held_anchor, _ = _binned_overlap_distribution(
        anchor,
        "active_polyphony_duration_distribution",
        minimum_key=0,
        upper_bin=4,
    )
    held_neighbors = [
        _binned_overlap_distribution(
            row,
            "active_polyphony_duration_distribution",
            minimum_key=0,
            upper_bin=4,
        )[0]
        for row in neighborhood
    ]
    held_center, held_radius = _center_and_radius("distribution", held_neighbors)

    indexed = {
        str(item["id"]): item
        for items in stage_targets.values()
        for item in items
    }
    register = indexed["performance_texture.relative_register"]
    pitch_range = indexed["performance_texture.pitch_range"]
    velocity = indexed["performance_texture.velocity"]
    range_center = float(pitch_range["neighborhood_center"][0])
    range_radius = float(pitch_range["neighborhood_radius"])
    target_span = math.floor(range_center * 88 + 0.5)
    maximum_span = math.floor((range_center + range_radius) * 88)
    if not 0 <= target_span <= maximum_span <= 87:
        raise ReferenceGenerationTargetError("register envelope span is invalid")
    unverified = {"status": "unverified"}
    return {
        "piece_plan": [
            _tonal_hierarchy_target(artifact_controls["brightness"], control_summary)
        ],
        "score_spec": [
            _attack_frequency_target(
                anchor,
                neighborhood,
                control=artifact_controls["attack_frequency"],
                control_summary=control_summary,
                legacy_reference=legacy_reference_attack_frequency,
            ),
            {
                "id": "attack_texture",
                "kind": "distribution",
                "generation_stage": "texture_collection",
                "grouping": {
                    "anchor": "first_note_on",
                    "chain": False,
                    "tolerance_ms": 30,
                },
                "bins": [
                    {"id": "one", "meaning": "1音"},
                    {"id": "two", "meaning": "2音"},
                    {"id": "three", "meaning": "3音"},
                    {"id": "four_or_more", "meaning": "4音以上"},
                ],
                "anchor": attack_anchor,
                "neighborhood_center": attack_center,
                "neighborhood_radius": attack_radius,
                "per_25_groups": _per_group_counts(attack_center, 25),
                "notes_per_attack": {
                    "anchor": _round(notes_anchor),
                    "neighborhood_center": notes_center[0],
                    "neighborhood_radius": notes_radius,
                },
                "maximum_group_size": {"anchor": max(raw_attack_anchor)},
                "source_observation_ids": [
                    "control_baseline.diagnostics.overlap.attack_size_distribution",
                    "control_baseline.diagnostics.overlap.notes_per_attack",
                ],
                "instructions": [
                    "全素材を通した配分として扱い、各素材へ同じ比率を強制しない",
                    "発音頻度を増やす指示として使わない",
                    "3音以上の打鍵を展開部と頂点へ配分できる",
                ],
                "reachability": unverified,
            },
            {
                "id": "foreground_accompaniment_coordination",
                "kind": "quality_rule",
                "generation_stage": "texture_collection",
                "reference_voice_labels": "unavailable",
                "instructions": [
                    "伴奏の独立発音を残す",
                    "句頭、和声転換、強調点、終止では確定旋律と合流する",
                    "2音以上の打鍵を伴奏だけの塊へ偏らせない",
                ],
                "reachability": unverified,
            },
            {
                "id": "register_envelope",
                "policy_id": REGISTER_ENVELOPE_POLICY_ID,
                "kind": "compound",
                "generation_stage": "melody_and_texture_collection",
                "target_span_semitones": target_span,
                "maximum_span_semitones": maximum_span,
                "pitch_range": {
                    "anchor": pitch_range["anchor"],
                    "neighborhood_center": pitch_range["neighborhood_center"],
                    "neighborhood_radius": pitch_range["neighborhood_radius"],
                },
                "relative_register_diagnostic": {
                    "anchor": register["anchor"],
                    "neighborhood_center": register["neighborhood_center"],
                    "neighborhood_radius": register["neighborhood_radius"],
                },
                "source_observation_ids": [
                    "reference_profile_v1.performance_texture.relative_register",
                    "reference_profile_v1.performance_texture.pitch_range",
                ],
                "instructions": [
                    "全曲の音域幅として扱い、各素材へ同じ幅を強制しない",
                    "少数の孤立した極端音だけで幅を作らない",
                    "一括移調や低域間隔違反で達成しない",
                ],
                "reachability": unverified,
            },
        ],
        "performance_spec": [
            {
                "id": "coordination_preservation",
                "kind": "quality_rule",
                "generation_stage": "performance_spec",
                "preserve_rolled_45ms": True,
                "instructions": [
                    "厚い発音群を保つ区間ではscoreまたはalignedを使う",
                    "rolledは崩しを意図する限定区間だけに使う",
                    "楽譜上で独立した両手の発音は吸着させない",
                ],
                "reachability": unverified,
            },
            {
                "id": "velocity_shape",
                "kind": "distribution",
                "generation_stage": "performance_spec",
                "bins": [
                    {"id": f"{index * 16}-{min(127, index * 16 + 15)}"}
                    for index in range(8)
                ],
                "anchor": velocity["anchor"],
                "neighborhood_center": velocity["neighborhood_center"],
                "neighborhood_radius": velocity["neighborhood_radius"],
                "source_observation_ids": [
                    "reference_profile_v1.performance_texture.velocity"
                ],
                "instructions": [
                    "区分ごとの強弱形を残す",
                    "全音を同じvelocityに固定しない",
                    "clipで強弱差を消さない",
                ],
                "reachability": unverified,
            },
        ],
        "rendered_surface": [
            {
                "id": "key_held_texture",
                "kind": "distribution",
                "generation_stage": "score_and_performance",
                "bins": [
                    {"id": "silent", "meaning": "無音"},
                    {"id": "one", "meaning": "1音"},
                    {"id": "two", "meaning": "2音"},
                    {"id": "three", "meaning": "3音"},
                    {"id": "four_or_more", "meaning": "4音以上"},
                ],
                "anchor": held_anchor,
                "neighborhood_center": held_center,
                "neighborhood_radius": held_radius,
                "includes_pedal_extension": False,
                "source_observation_ids": [
                    "control_baseline.diagnostics.overlap.active_polyphony_duration_distribution"
                ],
                "instructions": [
                    "鍵盤を押している音価上の厚みとして扱う",
                    "ペダルまたは発音頻度だけで達成しない",
                ],
                "reachability": unverified,
            }
        ],
    }


def _index_by_name(rows: Sequence[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row.get("name")
        if not isinstance(name, str) or not name:
            raise ReferenceGenerationTargetError(f"{label} record name is invalid")
        folded = name.casefold()
        if folded in result:
            raise ReferenceGenerationTargetError(f"duplicate {label} record: {name}")
        result[folded] = row
    return result


def _resolve_controls(
    request_controls: object, control_record: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    if not isinstance(request_controls, Mapping):
        raise ReferenceGenerationTargetError("resolved request controls must be an object")
    unknown = set(request_controls) - set(CONTROL_SPECS)
    if unknown:
        raise ReferenceGenerationTargetError(f"unknown control: {sorted(unknown)[0]}")
    normalized = control_record.get("normalized")
    if not isinstance(normalized, Mapping):
        raise ReferenceGenerationTargetError("control baseline normalized values are missing")
    artifact_controls: dict[str, dict[str, Any]] = {}
    prompt_controls: dict[str, float] = {}
    for control_id, label in CONTROL_SPECS.items():
        if control_id in request_controls:
            requested = request_controls[control_id]
            if not isinstance(requested, Mapping) or requested.get("label") != label:
                raise ReferenceGenerationTargetError(f"resolved control is invalid: {control_id}")
            value = _number(requested.get("value"), control_id)
            source = "explicit"
        else:
            value = _number(normalized.get(label), f"normalized.{label}")
            source = "reference"
        if not -1 <= value <= 1:
            raise ReferenceGenerationTargetError(f"control is outside corpus coordinates: {label}")
        if control_id == "brightness" and value not in {-1.0, 0.0, 1.0} and source == "explicit":
            raise ReferenceGenerationTargetError("explicit brightness must be -1, 0, or 1")
        resolved = {"label": label, "source": source, "value": _round(value)}
        artifact_controls[control_id] = resolved
        prompt_controls[control_id] = resolved["value"]
    return artifact_controls, prompt_controls


def build_reference_generation_target(
    resolved_request: Mapping[str, object],
    *,
    reference_dir: Path,
    control_dir: Path,
    legacy_reference_attack_frequency: bool = False,
) -> ReferenceGenerationTarget:
    """検証済み成果物と解決済み要求から、匿名の段階別目標を返す。"""

    reference_dir = Path(reference_dir)
    control_dir = Path(control_dir)
    reference_manifest = _read_json(reference_dir / "manifest.json")
    control_manifest = _read_json(control_dir / "manifest.json")
    if reference_manifest.get("status") != "pass":
        raise ReferenceGenerationTargetError("reference profile manifest has not passed")
    if control_manifest.get("status") != "complete":
        raise ReferenceGenerationTargetError("control baseline manifest is incomplete")
    if control_manifest.get("axis_version") != AXIS_VERSION:
        raise ReferenceGenerationTargetError("control baseline axis version is incompatible")
    _verify_output_hashes(reference_dir, reference_manifest, ("files.jsonl", "summary.json"))
    _verify_output_hashes(control_dir, control_manifest, ("records.jsonl", "summary.json"))
    if control_manifest.get("reference_manifest_sha256") != sha256_file(
        reference_dir / "manifest.json"
    ):
        raise ReferenceGenerationTargetError("reference manifest hash mismatch")

    reference_summary = _read_json(reference_dir / "summary.json")
    control_summary = _read_json(control_dir / "summary.json")
    if reference_summary.get("status") != "pass":
        raise ReferenceGenerationTargetError("reference profile summary has not passed")
    if control_summary.get("axis_version") != AXIS_VERSION:
        raise ReferenceGenerationTargetError(
            "control baseline summary axis version is incompatible"
        )

    reference = resolved_request.get("reference")
    if not isinstance(reference, Mapping) or reference.get("state") != "resolved":
        raise ReferenceGenerationTargetError("reference must be resolved before target creation")
    reference_name = reference.get("name")
    reference_sha256 = reference.get("sha256")
    if not isinstance(reference_name, str) or not isinstance(reference_sha256, str):
        raise ReferenceGenerationTargetError("resolved reference provenance is invalid")
    source_hashes = reference_manifest.get("inputs")
    if not isinstance(source_hashes, Mapping):
        raise ReferenceGenerationTargetError("reference manifest inputs are invalid")
    validated_source_sha256 = source_hashes.get(reference_name)
    if (
        not isinstance(validated_source_sha256, str)
        or validated_source_sha256.casefold() != reference_sha256.casefold()
    ):
        raise ReferenceGenerationTargetError("resolved reference source hash mismatch")

    reference_index = _index_by_name(
        _read_jsonl(reference_dir / "files.jsonl"), "reference profile"
    )
    control_index = _index_by_name(
        _read_jsonl(control_dir / "records.jsonl"), "control baseline"
    )
    anchor = reference_index.get(reference_name.casefold())
    control_record = control_index.get(reference_name.casefold())
    if anchor is None or control_record is None:
        raise ReferenceGenerationTargetError(
            "selected reference is missing from validated artifacts"
        )
    if anchor.get("status") != "pass":
        raise ReferenceGenerationTargetError("selected reference profile has not passed")

    try:
        neighbor_entries = anchor["neighborhood"]["neighbors"]
    except (KeyError, TypeError) as error:
        raise ReferenceGenerationTargetError(
            "selected reference neighborhood is missing"
        ) from error
    if not isinstance(neighbor_entries, list) or not neighbor_entries:
        raise ReferenceGenerationTargetError("selected reference neighborhood is empty")
    neighborhood: list[dict[str, Any]] = []
    control_neighborhood: list[dict[str, Any]] = []
    for entry in neighbor_entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("name"), str):
            raise ReferenceGenerationTargetError("selected reference neighborhood is invalid")
        row = reference_index.get(entry["name"].casefold())
        if row is None or row.get("status") != "pass":
            raise ReferenceGenerationTargetError("selected reference neighbor is unavailable")
        neighborhood.append(row)
        control_row = control_index.get(entry["name"].casefold())
        if control_row is None:
            raise ReferenceGenerationTargetError(
                "selected reference neighbor control baseline is unavailable"
            )
        control_neighborhood.append(control_row)

    artifact_controls, prompt_controls = _resolve_controls(
        resolved_request.get("controls", {}), control_record
    )
    stage_targets: dict[str, list[dict[str, Any]]] = {
        "score_spec": [],
        "performance_spec": [],
        "rendered_surface": [],
    }
    for stage, group, metric_name in DESCRIPTOR_SPECS:
        kind, anchor_values = _metric(anchor, group, metric_name)
        neighborhood_values: list[list[float]] = []
        for row in neighborhood:
            neighbor_kind, values = _metric(row, group, metric_name)
            if neighbor_kind != kind:
                raise ReferenceGenerationTargetError(
                    f"reference metric kind differs: {group}.{metric_name}"
                )
            neighborhood_values.append(values)
        center, radius = _center_and_radius(kind, neighborhood_values)
        stage_targets[stage].append(
            {
                "id": f"{group}.{metric_name}",
                "kind": kind,
                "anchor": [_round(value) for value in anchor_values],
                "neighborhood_center": center,
                "neighborhood_radius": radius,
                "final_observation_stage": "rendered_performance",
                "reachability": {"status": "unverified"},
            }
        )

    prompt_target = {
        "schema_version": 1,
        "target_version": TARGET_VERSION,
        "controls": prompt_controls,
        "piece_plan": {
            "reference_specific_targets": [],
            "long_term_structure": {"status": "unverified"},
        },
        "stage_targets": stage_targets,
        "semantic_targets": _semantic_targets(
            control_record,
            control_neighborhood,
            stage_targets,
            artifact_controls=artifact_controls,
            control_summary=control_summary,
            legacy_reference_attack_frequency=legacy_reference_attack_frequency,
        ),
    }
    artifact = {
        "schema_version": 1,
        "target_version": TARGET_VERSION,
        "reference": {
            "name": reference_name,
            "sha256": reference_sha256.lower(),
            "selection_method": reference.get("selection_method"),
        },
        "controls": artifact_controls,
        "inputs": {
            "reference_manifest_sha256": sha256_file(reference_dir / "manifest.json"),
            "control_manifest_sha256": sha256_file(control_dir / "manifest.json"),
        },
        "prompt_target": prompt_target,
        "prompt_target_sha256": sha256_json(prompt_target),
    }
    return ReferenceGenerationTarget(
        artifact=artifact,
        sha256=sha256_json(artifact),
        prompt_target=prompt_target,
        prompt_sha256=sha256_json(prompt_target),
    )
