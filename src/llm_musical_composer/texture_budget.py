"""発音頻度と30 ms発音群テクスチャーを同時に満たす素材別予算。"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from llm_musical_composer.harmonic_skeleton import HarmonicSkeletonV0, ScorePayloadV0
from llm_musical_composer.performance_pipeline import (
    PiecePlan,
    RenderedPerformance,
    ScoreSpec,
)


class TextureBudgetError(ValueError):
    """共同テクスチャー予算を決定できない場合に送出する。"""


_BUCKETS = ("one", "two", "three", "four_or_more")
_MINIMUM_BUCKET_SIZES = (1, 2, 3, 4)
_TARGET_DURATION_MS = 180_000
_MINIMUM_ENDING_HOLD_MS = 2_000


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TextureBudgetError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise TextureBudgetError(f"{label} must be finite")
    return result


def _largest_remainder(probabilities: Sequence[float], total: int) -> list[int]:
    raw = [value * total for value in probabilities]
    result = [math.floor(value) for value in raw]
    missing = total - sum(result)
    order = sorted(range(len(raw)), key=lambda index: (-(raw[index] % 1), index))
    for index in order[:missing]:
        result[index] += 1
    return result


def _weighted_allocation(
    total: int,
    records: Sequence[dict[str, Any]],
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
) -> list[int]:
    expanded_length = sum(
        record["occurrence_count"] * record["length_units"] for record in records
    )
    if expanded_length <= 0:
        raise TextureBudgetError("expanded material length must be positive")
    ideals = [total * record["length_units"] / expanded_length for record in records]
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for index, record in enumerate(records):
        weight = record["occurrence_count"]
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        for subtotal, (cost, values) in states.items():
            for value in range(lower_bounds[index], upper_bounds[index] + 1):
                candidate_total = subtotal + weight * value
                if candidate_total > total:
                    break
                candidate = (cost + (value - ideals[index]) ** 2, (*values, value))
                current = next_states.get(candidate_total)
                if current is None or candidate < current:
                    next_states[candidate_total] = candidate
        states = next_states
    if total not in states:
        raise TextureBudgetError("weighted material allocation has no integer solution")
    return list(states[total][1])


def _material_bucket_counts(
    records: Sequence[dict[str, Any]],
    group_counts: Sequence[int],
    probabilities: Sequence[float],
    global_counts: Sequence[int],
) -> list[list[int]]:
    counts = [
        _largest_remainder(probabilities, group_count) for group_count in group_counts
    ]
    expanded = [
        sum(
            record["occurrence_count"] * row[bucket_index]
            for record, row in zip(records, counts, strict=True)
        )
        for bucket_index in range(len(_BUCKETS))
    ]
    differences = [
        target - actual
        for target, actual in zip(global_counts, expanded, strict=True)
    ]
    while any(differences):
        destination = next(index for index, value in enumerate(differences) if value > 0)
        source = next(index for index, value in enumerate(differences) if value < 0)
        candidates = [
            index
            for index, (record, row) in enumerate(zip(records, counts, strict=True))
            if row[source] > 0
            and record["occurrence_count"] <= differences[destination]
            and record["occurrence_count"] <= -differences[source]
        ]
        if not candidates:
            raise TextureBudgetError("attack-size buckets have no integer allocation")
        selected = min(
            candidates,
            key=lambda index: (
                records[index]["occurrence_count"] != 1,
                records[index]["material_id"],
            ),
        )
        weight = records[selected]["occurrence_count"]
        counts[selected][source] -= 1
        counts[selected][destination] += 1
        differences[source] += weight
        differences[destination] -= weight
    return counts


def _distribution_distance(actual: Sequence[float], expected: Sequence[float]) -> float:
    return sum(abs(left - right) for left, right in zip(actual, expected, strict=True)) / 2


def material_attack_group_capacities(
    plan: PiecePlan,
    skeleton: HarmonicSkeletonV0,
    *,
    duration_ms: int = _TARGET_DURATION_MS,
    minimum_ending_hold_ms: int = _MINIMUM_ENDING_HOLD_MS,
) -> dict[str, dict[str, int]]:
    """終止保持用の末尾位置を除いた素材別発音群容量を返す。"""

    if duration_ms <= 0 or minimum_ending_hold_ms <= 0:
        raise TextureBudgetError("ending capacity durations must be positive")
    nodes = {item.node_id: item for item in plan.nodes}
    children = {item.node_id: [] for item in plan.nodes}
    for node in plan.nodes:
        if node.parent_id is not None:
            children[node.parent_id].append(node)

    def visit(node_id: str) -> tuple[Any, ...]:
        descendants = sorted(children[node_id], key=lambda item: item.order)
        if not descendants:
            return (nodes[node_id],)
        return tuple(leaf for child in descendants for leaf in visit(child.node_id))

    leaves = visit(plan.root_node_id)
    if not leaves or leaves[-1].role != "release":
        raise TextureBudgetError("final occurrence must be a release")
    final_material_id = leaves[-1].score_material_id
    if final_material_id is None:
        raise TextureBudgetError("final release material is missing")
    if sum(leaf.score_material_id == final_material_id for leaf in leaves) != 1:
        raise TextureBudgetError("final release material must be dedicated")
    materials = {item.material_id: item for item in skeleton.materials}
    try:
        total_units = sum(materials[leaf.score_material_id].length_units for leaf in leaves)
    except KeyError as error:
        raise TextureBudgetError("plan material is missing from skeleton") from error
    minimum_final_duration_units = math.ceil(
        total_units * minimum_ending_hold_ms / duration_ms
    )
    reserved_ending_units = minimum_final_duration_units - 1
    result: dict[str, dict[str, int]] = {}
    for material in skeleton.materials:
        reserved = (
            reserved_ending_units if material.material_id == final_material_id else 0
        )
        capacity = material.length_units - reserved
        if capacity <= 0:
            raise TextureBudgetError("ending reserve exhausts attack capacity")
        result[material.material_id] = {
            "length_units": material.length_units,
            "attack_group_capacity": capacity,
            "reserved_ending_units": reserved,
            "minimum_final_duration_units": (
                minimum_final_duration_units
                if material.material_id == final_material_id
                else 0
            ),
        }
    return result


def allocate_texture_budget(
    plan: PiecePlan,
    skeleton: HarmonicSkeletonV0,
    melodies: ScorePayloadV0,
    attack_target: Mapping[str, object],
    *,
    minimum_attack_group_count: int,
) -> dict[str, Any]:
    """確定旋律を差し引いたScoreMaterial別の伴奏event予算を返す。"""

    if minimum_attack_group_count <= 0:
        raise TextureBudgetError("minimum attack group count must be positive")
    center_raw = attack_target.get("neighborhood_center")
    if not isinstance(center_raw, Sequence) or isinstance(center_raw, (str, bytes)):
        raise TextureBudgetError("attack texture center is invalid")
    center = [_number(value, "attack texture center") for value in center_raw]
    if len(center) != 4 or any(value < 0 for value in center):
        raise TextureBudgetError("attack texture center is invalid")
    center_total = sum(center)
    if center_total <= 0:
        raise TextureBudgetError("attack texture center is empty")
    center = [value / center_total for value in center]
    radius = _number(attack_target.get("neighborhood_radius"), "attack texture radius")
    notes_target = attack_target.get("notes_per_attack")
    if not isinstance(notes_target, Mapping):
        raise TextureBudgetError("notes-per-attack target is invalid")
    notes_center = _number(
        notes_target.get("neighborhood_center"), "notes-per-attack center"
    )
    notes_radius = _number(
        notes_target.get("neighborhood_radius"), "notes-per-attack radius"
    )
    notes_lower = notes_center - notes_radius
    maximum_target = attack_target.get("maximum_group_size")
    if not isinstance(maximum_target, Mapping):
        raise TextureBudgetError("maximum group size target is missing")
    maximum_group_size = int(
        _number(maximum_target.get("anchor"), "maximum group size")
    )
    if maximum_group_size < 4:
        raise TextureBudgetError("maximum group size must allow the upper bucket")

    occurrence_counts = Counter(
        node.score_material_id
        for node in plan.nodes
        if node.score_material_id is not None
    )
    payloads = {item.material_id: item for item in melodies.materials}
    capacities = material_attack_group_capacities(plan, skeleton)
    records: list[dict[str, Any]] = []
    for material in skeleton.materials:
        payload = payloads.get(material.material_id)
        occurrence_count = occurrence_counts.get(material.material_id, 0)
        if payload is None or occurrence_count <= 0:
            raise TextureBudgetError("material payload or occurrence is missing")
        melody_groups = Counter(note.at_units for note in payload.notes)
        if not melody_groups:
            raise TextureBudgetError("melody must contain attacks")
        records.append(
            {
                "material_id": material.material_id,
                "length_units": material.length_units,
                **capacities[material.material_id],
                "occurrence_count": occurrence_count,
                "melody_attack_group_count": len(melody_groups),
                "melody_note_event_count": len(payload.notes),
                "melody_maximum_group_size": max(melody_groups.values()),
            }
        )

    group_counts = _weighted_allocation(
        minimum_attack_group_count,
        records,
        [record["melody_attack_group_count"] for record in records],
        [record["attack_group_capacity"] for record in records],
    )
    global_bucket_counts = _largest_remainder(center, minimum_attack_group_count)
    bucket_counts = _material_bucket_counts(
        records,
        group_counts,
        center,
        global_bucket_counts,
    )
    distribution = [value / minimum_attack_group_count for value in global_bucket_counts]
    distance = _distribution_distance(distribution, center)
    if distance > radius:
        raise TextureBudgetError("integer attack-size distribution is outside neighborhood")

    minimum_note_count = sum(
        size * count
        for size, count in zip(_MINIMUM_BUCKET_SIZES, global_bucket_counts, strict=True)
    )
    required_note_count = max(
        minimum_note_count,
        math.ceil(notes_lower * minimum_attack_group_count),
    )
    maximum_note_count = sum(
        size * count
        for size, count in zip(
            (*_MINIMUM_BUCKET_SIZES[:-1], maximum_group_size),
            global_bucket_counts,
            strict=True,
        )
    )
    if required_note_count > maximum_note_count:
        raise TextureBudgetError("notes-per-attack target exceeds maximum group size")
    extra_notes = required_note_count - minimum_note_count
    if extra_notes:
        raise TextureBudgetError(
            "notes-per-attack target needs within-bucket allocation not implemented"
        )

    material_results = []
    for record, group_count, counts in zip(
        records, group_counts, bucket_counts, strict=True
    ):
        combined_note_count = sum(
            size * count
            for size, count in zip(_MINIMUM_BUCKET_SIZES, counts, strict=True)
        )
        required_texture_events = combined_note_count - record["melody_note_event_count"]
        if required_texture_events <= 0:
            raise TextureBudgetError("texture event budget must be positive")
        if record["melody_maximum_group_size"] > maximum_group_size:
            raise TextureBudgetError("melody already exceeds maximum group size")
        material_results.append(
            {
                **record,
                "combined_attack_group_count": group_count,
                "combined_note_event_count": combined_note_count,
                "attack_size_counts": dict(zip(_BUCKETS, counts, strict=True)),
                "required_texture_event_count": required_texture_events,
                "minimum_texture_event_count": required_texture_events,
                "maximum_texture_event_count": required_texture_events,
            }
        )

    expanded_note_count = sum(
        item["occurrence_count"] * item["combined_note_event_count"]
        for item in material_results
    )
    if expanded_note_count != required_note_count:
        raise TextureBudgetError("material note-event allocation does not match whole score")
    return {
        "schema_version": 1,
        "basis": "score_material_integer_allocation",
        "score_spec_target": {
            "attack_group_count": minimum_attack_group_count,
            "attack_size_counts": dict(
                zip(_BUCKETS, global_bucket_counts, strict=True)
            ),
            "attack_size_distribution": distribution,
            "distribution_center": center,
            "distribution_distance": distance,
            "distribution_radius": radius,
            "note_event_count": required_note_count,
            "notes_per_attack": required_note_count / minimum_attack_group_count,
            "notes_per_attack_minimum": notes_lower,
            "maximum_group_size": maximum_group_size,
        },
        "materials": material_results,
    }


def _measurement(
    group_sizes: Sequence[int],
    target: Mapping[str, object],
) -> dict[str, Any]:
    if not group_sizes:
        raise TextureBudgetError("texture measurement has no attack groups")
    maximum_group_size = int(target["maximum_group_size"])
    counts = [0, 0, 0, 0]
    for size in group_sizes:
        if size <= 0:
            raise TextureBudgetError("attack group size must be positive")
        counts[min(size, 4) - 1] += 1
    group_count = len(group_sizes)
    distribution = [count / group_count for count in counts]
    center = [float(value) for value in target["distribution_center"]]
    distance = _distribution_distance(distribution, center)
    note_count = sum(group_sizes)
    notes_per_attack = note_count / group_count
    return {
        "attack_group_count": group_count,
        "attack_size_counts": dict(zip(_BUCKETS, counts, strict=True)),
        "attack_size_distribution": distribution,
        "distribution_distance": distance,
        "distribution_radius": float(target["distribution_radius"]),
        "note_event_count": note_count,
        "notes_per_attack": notes_per_attack,
        "notes_per_attack_minimum": float(target["notes_per_attack_minimum"]),
        "maximum_group_size": max(group_sizes),
        "maximum_group_size_limit": maximum_group_size,
    }


def measure_score_texture_budget(
    plan: PiecePlan,
    score: ScoreSpec,
    budget: Mapping[str, object],
) -> dict[str, Any]:
    """ScoreMaterialを出現回数で展開した名目発音群を測る。"""

    target = budget.get("score_spec_target")
    material_targets = budget.get("materials")
    if not isinstance(target, Mapping) or not isinstance(material_targets, Sequence):
        raise TextureBudgetError("texture budget is invalid")
    expected = {
        str(item["material_id"]): item
        for item in material_targets
        if isinstance(item, Mapping) and "material_id" in item
    }
    occurrence_counts = Counter(
        node.score_material_id
        for node in plan.nodes
        if node.score_material_id is not None
    )
    expanded_sizes: list[int] = []
    materials = []
    for material in score.materials:
        target_item = expected.get(material.material_id)
        if target_item is None:
            raise TextureBudgetError("score material is missing from texture budget")
        by_onset = Counter(note.at_units for note in material.notes)
        sizes = list(by_onset.values())
        item = _measurement(sizes, target)
        item["material_id"] = material.material_id
        item["occurrence_count"] = occurrence_counts[material.material_id]
        expected_counts = target_item.get("attack_size_counts")
        item["matches_budget"] = (
            item["attack_group_count"]
            == target_item.get("combined_attack_group_count")
            and item["note_event_count"]
            == target_item.get("combined_note_event_count")
            and item["attack_size_counts"] == expected_counts
        )
        materials.append(item)
        expanded_sizes.extend(sizes * occurrence_counts[material.material_id])
    whole = _measurement(expanded_sizes, target)
    whole["matches_budget"] = (
        whole["attack_group_count"] == target.get("attack_group_count")
        and whole["note_event_count"] == target.get("note_event_count")
        and whole["attack_size_counts"] == target.get("attack_size_counts")
        and whole["distribution_distance"] <= whole["distribution_radius"]
        and whole["notes_per_attack"] >= whole["notes_per_attack_minimum"]
        and whole["maximum_group_size"] <= whole["maximum_group_size_limit"]
        and all(item["matches_budget"] for item in materials)
    )
    return {"schema_version": 1, "whole_score": whole, "materials": materials}


def measure_rendered_texture_budget(
    rendered: RenderedPerformance,
    budget: Mapping[str, object],
    *,
    tolerance_ms: int = 30,
) -> dict[str, Any]:
    """演奏時刻上の非連鎖30 ms発音群を共同予算へ照合する。"""

    if tolerance_ms < 0:
        raise TextureBudgetError("attack group tolerance must not be negative")
    target = budget.get("score_spec_target")
    if not isinstance(target, Mapping):
        raise TextureBudgetError("texture budget target is invalid")
    ordered = sorted(rendered.notes, key=lambda note: (note.at_ms, note.pitch))
    groups: list[list[Any]] = []
    anchor: int | None = None
    current: list[Any] = []
    for note in ordered:
        if anchor is None or note.at_ms - anchor > tolerance_ms:
            if current:
                groups.append(current)
            anchor = note.at_ms
            current = [note]
        else:
            current.append(note)
    if current:
        groups.append(current)
    result = _measurement([len(group) for group in groups], target)
    result["group_tolerance_ms"] = tolerance_ms
    result["frequency_matches_budget"] = (
        result["attack_group_count"] == target.get("attack_group_count", 0)
    )
    result["minimum_frequency_matches_budget"] = (
        result["attack_group_count"] >= target.get("attack_group_count", 0)
    )
    result["texture_shape_matches_budget"] = (
        result["distribution_distance"] <= result["distribution_radius"]
        and result["notes_per_attack"] >= result["notes_per_attack_minimum"]
        and result["maximum_group_size"] <= result["maximum_group_size_limit"]
    )
    result["matches_budget"] = (
        result["minimum_frequency_matches_budget"]
        and result["texture_shape_matches_budget"]
    )
    return {"schema_version": 1, "rendered_performance": result}
