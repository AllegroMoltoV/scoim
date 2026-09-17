"""`はげしさ`制御の橋渡し観測量と成立可能性を検証する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from llm_musical_composer.composition_ir import Composition, Material
from llm_musical_composer.reference_profile import (
    PedalEvent,
    ReferencePiece,
    extract_reference_profile,
)
from llm_musical_composer.smf_notes import SmfNote

OBSERVABLES = (
    "note_rate_hz",
    "attack_rate_hz",
    "mean_active_polyphony",
    "velocity_level",
)
EXCLUDED_NAMES = frozenset({"aimusic01.mid", "rut.mid"})


class IntensityControlError(ValueError):
    """`はげしさ`の入力を通常値として扱えない場合に送出する。"""


def load_reference_records(path: Path) -> list[dict[str, Any]]:
    """参照曲プロファイルのJSONLを行番号付きで厳密に読む。"""

    records: list[dict[str, Any]] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise IntensityControlError(f"unable to read reference records: {error}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise IntensityControlError(f"reference records line {line_number} is empty")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise IntensityControlError(
                f"reference records line {line_number} is invalid JSON"
            ) from error
        if not isinstance(raw, dict):
            raise IntensityControlError(f"reference records line {line_number} must be an object")
        records.append(raw)
    if not records:
        raise IntensityControlError("reference records are empty")
    return records


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IntensityControlError(f"{location} must be an object")
    return value


def _positive_int(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise IntensityControlError(f"{location} must be a positive integer")
    return value


def _distribution(value: object, *, size: int, location: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise IntensityControlError(f"{location} must contain {size} values")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise IntensityControlError(f"{location} must contain finite numbers")
        number = float(item)
        if not math.isfinite(number) or number < 0:
            raise IntensityControlError(f"{location} must contain non-negative finite numbers")
        result.append(number)
    if not math.isclose(sum(result), 1.0, rel_tol=0, abs_tol=1e-6):
        raise IntensityControlError(f"{location} must sum to 1")
    return result


def _metric_distribution(
    performance: Mapping[str, Any], metric_name: str, *, size: int
) -> list[float]:
    metric = _mapping(performance.get(metric_name), f"performance_texture.{metric_name}")
    if metric.get("kind") != "distribution":
        raise IntensityControlError(f"performance_texture.{metric_name}.kind is invalid")
    return _distribution(
        metric.get("values"),
        size=size,
        location=f"performance_texture.{metric_name}.values",
    )


def extract_intensity_observables(record: Mapping[str, Any]) -> dict[str, float]:
    """曲別参照レコードから三つの制御観測量を抽出する。"""

    if record.get("status") != "pass":
        raise IntensityControlError("record status must be pass")
    name = record.get("name")
    if not isinstance(name, str) or not name:
        raise IntensityControlError("record name is required")
    if name.casefold() in {item.casefold() for item in EXCLUDED_NAMES}:
        raise IntensityControlError(f"excluded reference is not allowed: {name}")

    profile = _mapping(record.get("profile"), "profile")
    if profile.get("status") != "pass":
        raise IntensityControlError("profile status must be pass")
    attack_count = _positive_int(profile.get("attack_count"), "profile.attack_count")
    note_count = _positive_int(profile.get("note_count"), "profile.note_count")
    if note_count < attack_count:
        raise IntensityControlError("profile.note_count must not be smaller than attack_count")
    duration_ms = _positive_int(profile.get("duration_ms"), "profile.duration_ms")
    groups = _mapping(profile.get("feature_groups"), "profile.feature_groups")
    performance_group = _mapping(groups.get("performance_texture"), "performance_texture")
    performance = _mapping(performance_group.get("metrics"), "performance_texture.metrics")
    polyphony = _metric_distribution(performance, "polyphony_duration", size=5)
    velocity = _metric_distribution(performance, "velocity", size=8)

    neighborhood = _mapping(record.get("neighborhood"), "neighborhood")
    if neighborhood.get("anchor") != name:
        raise IntensityControlError("neighborhood anchor must match record name")

    return {
        "note_rate_hz": round(note_count / (duration_ms / 1000), 8),
        "attack_rate_hz": round(attack_count / (duration_ms / 1000), 8),
        "mean_active_polyphony": round(
            sum(index * weight for index, weight in enumerate(polyphony)), 8
        ),
        "velocity_level": round(
            sum(index * weight for index, weight in enumerate(velocity)) / 7, 8
        ),
    }


def extract_composition_intensity(composition: Composition) -> dict[str, Any]:
    """再利用を展開したComposition本体を参照曲と同じ定義で測る。"""

    materials = composition.material_by_id
    notes: list[SmfNote] = []
    pedals: list[PedalEvent] = []
    offset = 0
    for use in composition.form:
        material = materials[use.material_id]
        notes.extend(
            SmfNote(
                pitch=note.pitch,
                onset_ms=offset + note.at_ms,
                duration_ms=note.duration_ms,
                velocity=note.velocity,
            )
            for note in material.notes
        )
        pedals.extend(
            PedalEvent(at_ms=offset + pedal.at_ms, value=pedal.value) for pedal in material.pedals
        )
        offset += material.duration_ms
    if not notes:
        raise IntensityControlError("composition body must contain a completed note")
    try:
        piece = ReferencePiece(
            name="expanded-composition-body",
            notes=tuple(notes),
            pedals=tuple(pedals),
        )
        profile = extract_reference_profile(piece)
    except (ValueError, statistics.StatisticsError) as error:
        raise IntensityControlError(f"unable to measure composition body: {error}") from error
    record = {
        "name": "expanded-composition-body",
        "status": "pass",
        "profile": profile,
        "neighborhood": {"anchor": "expanded-composition-body"},
    }
    return {
        "status": "pass",
        "basis": "expanded composition body without generated tonic ending",
        "expanded_note_count": len(notes),
        "expanded_duration_ms": offset,
        "observables": extract_intensity_observables(record),
    }


def extract_material_intensity(material: Material) -> dict[str, Any]:
    """素材の宣言時間を分母にし、全曲と同じ四観測量を測る。"""

    if material.duration_ms <= 0 or not material.notes:
        raise IntensityControlError("material must contain a completed note in positive duration")
    if any(
        note.at_ms < 0
        or note.duration_ms <= 0
        or note.at_ms >= material.duration_ms
        or note.at_ms + note.duration_ms > material.duration_ms
        for note in material.notes
    ):
        raise IntensityControlError("material notes must fit inside the declared duration")
    attacks = len({note.at_ms for note in material.notes})
    velocity_level = statistics.mean(min(note.velocity // 16, 7) for note in material.notes) / 7
    changes: dict[int, int] = {0: 0, material.duration_ms: 0}
    for note in material.notes:
        changes[note.at_ms] = changes.get(note.at_ms, 0) + 1
        end = note.at_ms + note.duration_ms
        changes[end] = changes.get(end, 0) - 1
    active = 0
    previous = 0
    weighted_polyphony = 0
    for at_ms in sorted(changes):
        weighted_polyphony += min(active, 4) * (at_ms - previous)
        active += changes[at_ms]
        previous = at_ms
    duration_seconds = material.duration_ms / 1000
    observables = {
        "note_rate_hz": round(len(material.notes) / duration_seconds, 8),
        "attack_rate_hz": round(attacks / duration_seconds, 8),
        "mean_active_polyphony": round(weighted_polyphony / material.duration_ms, 8),
        "velocity_level": round(velocity_level, 8),
    }
    return {
        "status": "pass",
        "basis": "declared material duration without generated ending",
        "material_id": material.material_id,
        "duration_ms": material.duration_ms,
        "note_count": len(material.notes),
        "attack_count": attacks,
        "observables": observables,
    }


def _relative_acceptance(center: float, target: float) -> dict[str, float]:
    delta = target - center
    if math.isclose(delta, 0, rel_tol=0, abs_tol=1e-12):
        return {"minimum": round(center, 8), "maximum": round(center, 8)}
    midpoint = center + delta * 0.5
    outer = target + delta * 0.5
    return {
        "minimum": round(min(midpoint, outer), 8),
        "maximum": round(max(midpoint, outer), 8),
    }


def resolve_material_intensity_target(
    material: Material,
    records: Iterable[Mapping[str, Any]],
    *,
    reference_name: str,
    value: float,
    duration_seconds: float = 180.0,
    max_notes: int = 950,
) -> dict[str, Any]:
    """参照近傍の相対変化を、同じ局所素材の観測値へ写像する。"""

    record_list = list(records)
    center = resolve_intensity_target(
        record_list,
        reference_name=reference_name,
        value=0.0,
        duration_seconds=duration_seconds,
        max_notes=max_notes,
    )
    endpoint = resolve_intensity_target(
        record_list,
        reference_name=reference_name,
        value=value,
        duration_seconds=duration_seconds,
        max_notes=max_notes,
    )
    if center["status"] != "reachable" or endpoint["status"] != "reachable":
        return {
            "schema_version": 1,
            "status": "unreachable",
            "reference": reference_name,
            "value": float(value),
            "reason": endpoint.get("reason", center.get("reason", "reference is unreachable")),
        }
    baseline = extract_material_intensity(material)
    baseline_observables = baseline["observables"]
    targets = {
        "note_rate_hz": baseline_observables["note_rate_hz"]
        * endpoint["targets"]["note_rate_hz"]
        / center["targets"]["note_rate_hz"],
        "attack_rate_hz": baseline_observables["attack_rate_hz"]
        * endpoint["targets"]["attack_rate_hz"]
        / center["targets"]["attack_rate_hz"],
        "velocity_level": baseline_observables["velocity_level"]
        + endpoint["targets"]["velocity_level"]
        - center["targets"]["velocity_level"],
    }
    if not 0 <= targets["velocity_level"] <= 1:
        return {
            "schema_version": 1,
            "status": "unreachable",
            "reference": reference_name,
            "value": float(value),
            "reason": "relative velocity target is outside the representable range",
        }
    if targets["attack_rate_hz"] > targets["note_rate_hz"]:
        return {
            "schema_version": 1,
            "status": "unreachable",
            "reference": reference_name,
            "value": float(value),
            "reason": "relative attack target exceeds note target",
        }

    by_name = {str(record.get("name")): record for record in record_list}
    neighbor_names = _neighbor_names(by_name[reference_name], set(by_name))
    polyphony_values = [
        extract_intensity_observables(by_name[name])["mean_active_polyphony"]
        for name in neighbor_names
    ]
    polyphony_center = statistics.median(polyphony_values)
    baseline_polyphony = baseline_observables["mean_active_polyphony"]
    hold = {
        "minimum": baseline_polyphony * min(polyphony_values) / polyphony_center,
        "maximum": baseline_polyphony * max(polyphony_values) / polyphony_center,
    }
    rounded_targets = {name: round(number, 8) for name, number in targets.items()}
    duration = material.duration_ms / 1000
    return {
        "schema_version": 1,
        "status": "reachable",
        "reference": reference_name,
        "value": float(value),
        "baseline": baseline,
        "targets": rounded_targets,
        "target_counts": {
            "note_count": round(targets["note_rate_hz"] * duration),
            "attack_count": round(targets["attack_rate_hz"] * duration),
        },
        "acceptance": {
            name: _relative_acceptance(baseline_observables[name], rounded_targets[name])
            for name in ("note_rate_hz", "attack_rate_hz", "velocity_level")
        },
        "holds": {
            "mean_active_polyphony": {
                "minimum": round(hold["minimum"], 8),
                "maximum": round(hold["maximum"], 8),
            }
        },
        "evidence": {
            "basis": "reference endpoint relative to the same material baseline",
            "reference_center": center["targets"],
            "reference_endpoint": endpoint["targets"],
        },
    }


def evaluate_material_intensity_triplet(
    materials: Mapping[float, Material], targets: Mapping[float, Mapping[str, Any]]
) -> dict[str, Any]:
    """同じ素材の低・中・高候補について方向、進捗、重なり保持を検査する。"""

    values = (-1.0, 0.0, 1.0)
    if set(materials) != set(values) or set(targets) != set(values):
        raise IntensityControlError("material triplet must contain exactly -1, 0, and 1")
    reports = {value: extract_material_intensity(materials[value]) for value in values}
    issues: list[str] = []
    driver_axes = ("note_rate_hz", "attack_rate_hz", "velocity_level")
    for axis in driver_axes:
        observed = [reports[value]["observables"][axis] for value in values]
        if not observed[0] < observed[1] < observed[2]:
            issues.append(axis)
            continue
        for value in (-1.0, 1.0):
            target = targets[value]
            if target.get("status") != "reachable":
                issues.append(axis)
                break
            accepted = target["acceptance"][axis]
            candidate = reports[value]["observables"][axis]
            if not accepted["minimum"] <= candidate <= accepted["maximum"]:
                issues.append(axis)
                break
    direction_issues = list(dict.fromkeys(issues))
    hold_issues: list[str] = []
    for value in (-1.0, 1.0):
        if targets[value].get("status") != "reachable":
            hold_issues.append("mean_active_polyphony")
            continue
        accepted = targets[value]["holds"]["mean_active_polyphony"]
        candidate = reports[value]["observables"]["mean_active_polyphony"]
        if not accepted["minimum"] <= candidate <= accepted["maximum"]:
            hold_issues.append("mean_active_polyphony")
    all_issues = list(dict.fromkeys([*direction_issues, *hold_issues]))
    return {
        "status": "pass" if not all_issues else "fail",
        "direction_status": "pass" if not direction_issues else "fail",
        "hold_status": "pass" if not hold_issues else "fail",
        "issues": all_issues,
        "candidates": {str(value): reports[value] for value in values},
    }


def _validated_observables(value: Mapping[str, Any], location: str) -> dict[str, float]:
    if set(value) != set(OBSERVABLES):
        raise IntensityControlError(f"{location} observable set is invalid")
    result: dict[str, float] = {}
    for name in OBSERVABLES:
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise IntensityControlError(f"{location}.{name} must be finite")
        number = float(item)
        if not math.isfinite(number) or number < 0:
            raise IntensityControlError(f"{location}.{name} must be non-negative and finite")
        result[name] = number
    return result


def find_direction_candidates(
    anchor: Mapping[str, Any],
    candidates: Mapping[str, Mapping[str, Any]],
    *,
    direction_axes: Sequence[str] = OBSERVABLES,
) -> dict[str, Any]:
    """指定した全軸が厳密に同方向へ動く候補だけを返す。"""

    baseline = _validated_observables(anchor, "anchor")
    if not direction_axes or any(axis not in OBSERVABLES for axis in direction_axes):
        raise IntensityControlError("direction_axes must be a non-empty observable subset")
    if len(direction_axes) != len(set(direction_axes)):
        raise IntensityControlError("direction_axes contains duplicates")
    lower: list[dict[str, Any]] = []
    higher: list[dict[str, Any]] = []
    mixed_or_equal = 0
    for name in sorted(candidates, key=lambda item: (item.casefold(), item)):
        if not isinstance(name, str) or not name:
            raise IntensityControlError("candidate name is required")
        values = _validated_observables(candidates[name], f"candidates[{name!r}]")
        deltas = {key: round(values[key] - baseline[key], 8) for key in OBSERVABLES}
        item = {"name": name, "observables": values, "deltas": deltas}
        if all(deltas[key] < 0 for key in direction_axes):
            lower.append(item)
        elif all(deltas[key] > 0 for key in direction_axes):
            higher.append(item)
        else:
            mixed_or_equal += 1
    return {
        "lower": lower,
        "higher": higher,
        "mixed_or_equal_count": mixed_or_equal,
    }


def _neighbor_names(record: Mapping[str, Any], known_names: set[str]) -> list[str]:
    neighborhood = _mapping(record.get("neighborhood"), "neighborhood")
    raw_neighbors = neighborhood.get("neighbors")
    if not isinstance(raw_neighbors, list) or len(raw_neighbors) < 3:
        raise IntensityControlError("neighborhood must contain at least three references")
    names: list[str] = []
    for index, raw in enumerate(raw_neighbors):
        item = _mapping(raw, f"neighborhood.neighbors[{index}]")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise IntensityControlError("neighborhood neighbor name is required")
        if name not in known_names:
            raise IntensityControlError(f"unknown neighborhood reference: {name}")
        names.append(name)
    if len(names) != len(set(names)):
        raise IntensityControlError("neighborhood contains duplicate references")
    anchor_name = record["name"]
    if names.count(anchor_name) != 1:
        raise IntensityControlError("neighborhood must contain its anchor exactly once")
    return names


def _capable_default_reference(
    requested_default: str,
    by_name: Mapping[str, Mapping[str, Any]],
    eligible_names: set[str],
) -> tuple[str, str]:
    if requested_default in eligible_names:
        return requested_default, "reference_profile_medoid_is_preset_reachable"
    neighborhood = _mapping(by_name[requested_default].get("neighborhood"), "neighborhood")
    raw_neighbors = neighborhood.get("neighbors")
    if not isinstance(raw_neighbors, list):
        raise IntensityControlError("default reference neighborhood is invalid")
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    for index, raw in enumerate(raw_neighbors):
        item = _mapping(raw, f"neighborhood.neighbors[{index}]")
        name = item.get("name")
        if isinstance(name, str) and name in eligible_names:
            ranks = _mapping(item.get("group_ranks"), "neighborhood group_ranks")
            if not ranks or not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in ranks.values()
            ):
                raise IntensityControlError("neighborhood group ranks are invalid")
            candidates.append((name, ranks))
    if not candidates:
        raise IntensityControlError("no preset-reachable reference near the default medoid")
    selected = min(
        candidates,
        key=lambda item: (
            max(item[1].values()),
            statistics.median(item[1].values()),
            item[0].casefold(),
            item[0],
        ),
    )[0]
    return selected, "nearest_preset_reachable_neighbor_of_reference_profile_medoid"


def _ordinal_ranks(values: Mapping[str, float]) -> dict[str, int]:
    ordered = sorted(values, key=lambda name: (values[name], name.casefold(), name))
    return {name: index for index, name in enumerate(ordered)}


def _representative(
    observables: Mapping[str, Mapping[str, float]], target_index: int, excluded: set[str]
) -> str:
    ranks = {
        axis: _ordinal_ranks({name: values[axis] for name, values in observables.items()})
        for axis in OBSERVABLES
    }
    candidates = [name for name in observables if name not in excluded]
    if not candidates:
        raise IntensityControlError("not enough distinct representative anchors")
    return min(
        candidates,
        key=lambda name: (
            max(abs(ranks[axis][name] - target_index) for axis in OBSERVABLES),
            statistics.median(abs(ranks[axis][name] - target_index) for axis in OBSERVABLES),
            name.casefold(),
            name,
        ),
    )


def _average_ranks(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=lambda name: (values[name], name.casefold(), name))
    result: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[index]]:
            end += 1
        rank = (index + end - 1) / 2
        for name in ordered[index:end]:
            result[name] = rank
        index = end
    return result


def _spearman(first: Mapping[str, float], second: Mapping[str, float]) -> float | None:
    first_ranks = _average_ranks(first)
    second_ranks = _average_ranks(second)
    names = sorted(first)
    first_values = [first_ranks[name] for name in names]
    second_values = [second_ranks[name] for name in names]
    first_mean = statistics.mean(first_values)
    second_mean = statistics.mean(second_values)
    numerator = sum(
        (left - first_mean) * (right - second_mean)
        for left, right in zip(first_values, second_values, strict=True)
    )
    first_scale = sum((value - first_mean) ** 2 for value in first_values)
    second_scale = sum((value - second_mean) ** 2 for value in second_values)
    if first_scale == 0 or second_scale == 0:
        return None
    return round(numerator / math.sqrt(first_scale * second_scale), 8)


def _balanced_endpoint(
    names: Sequence[str],
    observables: Mapping[str, Mapping[str, float]],
    center: Mapping[str, float],
    *,
    direction: str,
) -> str | None:
    axes = ("note_rate_hz", "attack_rate_hz", "velocity_level")
    ranges = {
        axis: max(observables[name][axis] for name in names)
        - min(observables[name][axis] for name in names)
        for axis in axes
    }
    candidates: list[tuple[str, list[float]]] = []
    for name in names:
        if direction == "lower":
            differences = [center[axis] - observables[name][axis] for axis in axes]
        else:
            differences = [observables[name][axis] - center[axis] for axis in axes]
        if all(value > 0 for value in differences):
            progress = [
                difference / ranges[axis]
                for axis, difference in zip(axes, differences, strict=True)
                if ranges[axis] > 0
            ]
            if len(progress) == len(axes):
                candidates.append((name, progress))
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            -min(item[1]),
            -statistics.mean(item[1]),
            item[0].casefold(),
            item[0],
        ),
    )[0]


def resolve_intensity_target(
    records: Iterable[Mapping[str, Any]],
    *,
    reference_name: str,
    value: float,
    duration_seconds: float = 180.0,
    max_notes: int = 950,
) -> dict[str, Any]:
    """二駆動軸の参照相対目標と重なり保持範囲を決定的に解決する。"""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not -1 <= float(value) <= 1
    ):
        raise IntensityControlError("intensity value must be finite from -1 through 1")
    control_value = float(value)
    if (
        isinstance(duration_seconds, bool)
        or not isinstance(duration_seconds, (int, float))
        or not math.isfinite(float(duration_seconds))
        or duration_seconds <= 0
    ):
        raise IntensityControlError("duration_seconds must be positive and finite")
    if isinstance(max_notes, bool) or not isinstance(max_notes, int) or max_notes <= 0:
        raise IntensityControlError("max_notes must be a positive integer")
    attack_capacity = max_notes / float(duration_seconds)
    by_name: dict[str, Mapping[str, Any]] = {}
    observables: dict[str, dict[str, float]] = {}
    for record in records:
        name = record.get("name")
        if not isinstance(name, str) or not name:
            raise IntensityControlError("record name is required")
        if name in by_name:
            raise IntensityControlError(f"duplicate record name: {name}")
        by_name[name] = record
        observables[name] = extract_intensity_observables(record)
    if reference_name not in by_name:
        raise IntensityControlError("reference is unavailable")
    neighbor_names = _neighbor_names(by_name[reference_name], set(by_name))
    axes = ("note_rate_hz", "attack_rate_hz", "velocity_level")
    center = {
        axis: statistics.median(observables[name][axis] for name in neighbor_names) for axis in axes
    }
    if center["note_rate_hz"] > attack_capacity:
        return {
            "schema_version": 1,
            "status": "unreachable",
            "reference": reference_name,
            "value": control_value,
            "reason": "reference center exceeds preset note capacity",
            "preset_note_capacity_hz": round(attack_capacity, 8),
        }
    lower_name = _balanced_endpoint(neighbor_names, observables, center, direction="lower")
    if lower_name is None:
        return {
            "schema_version": 1,
            "status": "unreachable",
            "reference": reference_name,
            "value": control_value,
            "reason": "local neighborhood has no strictly lower two-driver endpoint",
        }
    higher_name = _balanced_endpoint(neighbor_names, observables, center, direction="higher")
    if higher_name is None:
        return {
            "schema_version": 1,
            "status": "unreachable",
            "reference": reference_name,
            "value": control_value,
            "reason": "local neighborhood has no strictly higher two-driver endpoint",
        }

    if control_value < 0:
        start = observables[lower_name]
        amount = control_value + 1
        targets = {axis: start[axis] + (center[axis] - start[axis]) * amount for axis in axes}
    else:
        amount = control_value
        higher_values = dict(observables[higher_name])
        higher_values["note_rate_hz"] = min(higher_values["note_rate_hz"], attack_capacity)
        higher_values["attack_rate_hz"] = min(
            higher_values["attack_rate_hz"], higher_values["note_rate_hz"]
        )
        if higher_values["note_rate_hz"] <= center["note_rate_hz"] and control_value > 0:
            return {
                "schema_version": 1,
                "status": "unreachable",
                "reference": reference_name,
                "value": control_value,
                "reason": "preset note capacity leaves no higher direction",
                "preset_note_capacity_hz": round(attack_capacity, 8),
            }
        targets = {
            axis: center[axis] + (higher_values[axis] - center[axis]) * amount for axis in axes
        }
    polyphony_values = [observables[name]["mean_active_polyphony"] for name in neighbor_names]
    note_rate_clipped = observables[higher_name]["note_rate_hz"] > attack_capacity
    return {
        "schema_version": 1,
        "status": "reachable",
        "reference": reference_name,
        "value": control_value,
        "targets": {axis: round(targets[axis], 8) for axis in axes},
        "holds": {
            "mean_active_polyphony": {
                "minimum": round(min(polyphony_values), 8),
                "maximum": round(max(polyphony_values), 8),
            }
        },
        "evidence": {
            "basis": "validated seven-reference local neighborhood",
            "lower_reference": lower_name,
            "center": {axis: round(center[axis], 8) for axis in axes},
            "higher_reference": higher_name,
            "preset_note_capacity_hz": round(attack_capacity, 8),
            "note_rate_clipped_to_preset": note_rate_clipped,
        },
        "unverified_meaning": "perceived intensity",
    }


def analyze_intensity_feasibility(
    records: Iterable[Mapping[str, Any]], *, default_reference: str
) -> dict[str, Any]:
    """三軸同時方向が代表参照の局所近傍で成立するかを要約する。"""

    by_name: dict[str, Mapping[str, Any]] = {}
    observables: dict[str, dict[str, float]] = {}
    for record in records:
        name = record.get("name")
        if not isinstance(name, str) or not name:
            raise IntensityControlError("record name is required")
        if name in by_name:
            raise IntensityControlError(f"duplicate record name: {name}")
        by_name[name] = record
        observables[name] = extract_intensity_observables(record)
    if len(by_name) < 7:
        raise IntensityControlError("at least seven reference records are required")
    if default_reference not in by_name:
        raise IntensityControlError("default reference is unavailable")

    names = set(by_name)
    count = len(names)
    lower_target = math.floor((count - 1) * 0.25)
    upper_target = math.ceil((count - 1) * 0.75)
    representatives = [default_reference]
    representatives.append(_representative(observables, lower_target, set(representatives)))
    representatives.append(_representative(observables, upper_target, set(representatives)))

    def build_controls(direction_axes: Sequence[str]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for anchor_name in representatives:
            neighbor_names = _neighbor_names(by_name[anchor_name], names)
            candidates = {name: observables[name] for name in neighbor_names if name != anchor_name}
            directions = find_direction_candidates(
                observables[anchor_name], candidates, direction_axes=direction_axes
            )
            result.append(
                {
                    "anchor": anchor_name,
                    "observables": observables[anchor_name],
                    "lower_count": len(directions["lower"]),
                    "higher_count": len(directions["higher"]),
                    "lower_candidates": directions["lower"],
                    "higher_candidates": directions["higher"],
                    "mixed_or_equal_count": directions["mixed_or_equal_count"],
                    "status": ("pass" if directions["lower"] and directions["higher"] else "fail"),
                }
            )
        return result

    original_axes = ("note_rate_hz", "mean_active_polyphony", "velocity_level")
    controls = build_controls(original_axes)
    driver_axes = ("note_rate_hz", "attack_rate_hz", "velocity_level")
    driver_controls = build_controls(driver_axes)

    correlations: dict[str, float | None] = {}
    for index, first in enumerate(OBSERVABLES):
        for second in OBSERVABLES[index + 1 :]:
            correlations[f"{first}__{second}"] = _spearman(
                {name: values[first] for name, values in observables.items()},
                {name: values[second] for name, values in observables.items()},
            )
    status = "pass" if all(item["status"] == "pass" for item in controls) else "fail"
    driver_status = "pass" if all(item["status"] == "pass" for item in driver_controls) else "fail"
    target_resolution = {
        anchor: [
            resolve_intensity_target(by_name.values(), reference_name=anchor, value=value)
            for value in (-1.0, -0.5, 0.0, 0.5, 1.0)
        ]
        for anchor in representatives
    }
    eligible_names: list[str] = []
    for name in sorted(by_name, key=lambda item: (item.casefold(), item)):
        endpoints = [
            resolve_intensity_target(by_name.values(), reference_name=name, value=value)
            for value in (-1.0, 0.0, 1.0)
        ]
        if all(item["status"] == "reachable" for item in endpoints):
            eligible_names.append(name)
    eligible_representatives: list[str] = []
    eligible_target_resolution: dict[str, list[dict[str, Any]]] = {}
    control_default_reference: str | None = None
    control_default_basis: str | None = None
    if len(eligible_names) >= 3:
        control_default_reference, control_default_basis = _capable_default_reference(
            default_reference, by_name, set(eligible_names)
        )
        eligible_observables = {name: observables[name] for name in eligible_names}
        eligible_representatives = [control_default_reference]
        eligible_representatives.append(
            _representative(
                eligible_observables,
                math.floor((len(eligible_names) - 1) * 0.25),
                set(eligible_representatives),
            )
        )
        eligible_representatives.append(
            _representative(
                eligible_observables,
                math.ceil((len(eligible_names) - 1) * 0.75),
                set(eligible_representatives),
            )
        )
        eligible_target_resolution = {
            anchor: [
                resolve_intensity_target(by_name.values(), reference_name=anchor, value=value)
                for value in (-1.0, -0.5, 0.0, 0.5, 1.0)
            ]
            for anchor in eligible_representatives
        }
    resolution_status = (
        "pass"
        if len(eligible_representatives) == 3
        and all(
            item["status"] == "reachable"
            for values in eligible_target_resolution.values()
            for item in values
        )
        else "fail"
    )
    return {
        "schema_version": 1,
        "status": status,
        "source_count": count,
        "observables": list(OBSERVABLES),
        "reference_style_distance_is_separate": True,
        "publication_state": "not_published",
        "representative_selection": (
            "validated default plus rank-balanced lower and upper quartile representatives"
        ),
        "representative_anchors": representatives,
        "anchor_controls": controls,
        "alternative_hypotheses": {
            "note_attack_velocity_drivers": {
                "status": driver_status,
                "direction_observables": list(driver_axes),
                "held_observable": "mean_active_polyphony",
                "anchor_controls": driver_controls,
                "limitation": (
                    "polyphony direction is not part of intensity; a separate reference-relative "
                    "guard is required before generation"
                ),
            }
        },
        "recommended_hypothesis": (
            "strict_three_axis"
            if status == "pass"
            else "note_attack_velocity_drivers"
            if driver_status == "pass"
            else "unresolved"
        ),
        "target_resolution": target_resolution,
        "preset_reachability": {
            "duration_seconds": 180,
            "max_notes": 950,
            "eligible_reference_count": len(eligible_names),
            "requested_default_reference": default_reference,
            "control_default_reference": control_default_reference,
            "control_default_selection_basis": control_default_basis,
            "eligible_representative_anchors": eligible_representatives,
            "status": resolution_status,
        },
        "eligible_target_resolution": eligible_target_resolution,
        "spearman_correlations": correlations,
        "decision_rule": (
            "every representative anchor requires at least one local neighbor strictly lower "
            "and one strictly higher on all three observables"
        ),
    }


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_intensity_feasibility(
    records: Sequence[Mapping[str, Any]],
    *,
    default_reference: str,
    output_dir: Path,
    input_sha256: str,
    implementation_sha256: str,
) -> dict[str, Any]:
    """決定的な成立可能性要約と入力・実装ハッシュを保存する。"""

    for location, digest in (
        ("input_sha256", input_sha256),
        ("implementation_sha256", implementation_sha256),
    ):
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in digest)
        ):
            raise IntensityControlError(f"{location} is invalid")
    summary = analyze_intensity_feasibility(records, default_reference=default_reference)
    summary_bytes = _canonical_json(summary)
    manifest = {
        "schema_version": 1,
        "status": summary["status"],
        "inputs": {"reference_profile_files_jsonl": input_sha256.lower()},
        "implementations": {"intensity_control.py": implementation_sha256.lower()},
        "outputs": {"summary.json": _sha256_bytes(summary_bytes)},
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_bytes(summary_bytes)
    (output_dir / "manifest.json").write_bytes(_canonical_json(manifest))
    return summary


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise IntensityControlError(f"unable to hash input: {error}") from error
    return digest.hexdigest()


def _default_reference(path: Path) -> str:
    try:
        summary = json.loads(Path(path).read_text(encoding="utf-8"))
        selected = summary["default_reference"]["selected"]
        if summary.get("status") != "pass" or selected.get("kind") != "real_medoid":
            raise KeyError("reference summary has not passed")
        name = selected["name"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise IntensityControlError("validated default reference is unavailable") from error
    if not isinstance(name, str) or not name:
        raise IntensityControlError("validated default reference name is invalid")
    return name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze the feasibility of the intensity fader")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    implementation_path = Path(__file__)
    summary = write_intensity_feasibility(
        load_reference_records(args.input_jsonl),
        default_reference=_default_reference(args.reference_summary),
        output_dir=args.output_dir,
        input_sha256=_sha256_file(args.input_jsonl),
        implementation_sha256=_sha256_file(implementation_path),
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "source_count": summary["source_count"],
                "representative_anchors": summary["representative_anchors"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
