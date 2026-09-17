"""観測上同じSMFになる構成候補を、明示的な有限集合として診断する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import deque
from dataclasses import dataclass, fields, is_dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

from llm_musical_composer.attack_frequency_control import scale_score_resolution
from llm_musical_composer.performance_pipeline import (
    PerformanceSpec,
    PiecePlan,
    ScoreSpec,
    ordered_leaf_schedule,
    render_performance,
    render_performance_smf,
    validate_pipeline,
)
from llm_musical_composer.reference_decomposition import (
    build_observed_performance,
    load_observed_smf,
)
from llm_musical_composer.run_state import atomic_write_bytes, atomic_write_json, sha256_file
from llm_musical_composer.score_timing_known_fixtures import (
    KnownTimingFixture,
    build_known_timing_fixtures,
)
from llm_musical_composer.structure_identifiability import (
    flatten_inert_internal_node,
    inert_internal_node_ids,
    merge_neutral_adjacent_materials,
    neutral_adjacent_leaf_pairs,
    scale_divisions_only,
    verify_known_timing_fixture,
)


@dataclass(frozen=True, order=True)
class StructureSpanV0:
    start: Fraction
    end: Fraction


@dataclass(frozen=True, order=True)
class StructureNodeV0:
    local_id: str
    node_kind: str
    parent_local_id: str | None
    sibling_order: int
    start: Fraction
    end: Fraction


@dataclass(frozen=True, order=True)
class StructureMaterialV0:
    local_id: str
    occurrence_leaf_ids: tuple[str, ...]
    occurrence_spans: tuple[StructureSpanV0, ...]


@dataclass(frozen=True, order=True)
class StructureDerivedRelationV0:
    subject_kind: str
    subject_local_id: str
    source_local_id: str
    subject_span: tuple[StructureSpanV0, ...]
    source_span: tuple[StructureSpanV0, ...]


@dataclass(frozen=True)
class CompactStructureDescriptorV0:
    schema_version: int
    nodes: tuple[StructureNodeV0, ...]
    materials: tuple[StructureMaterialV0, ...]
    derived_relations: tuple[StructureDerivedRelationV0, ...]
    terminal_tail_ratio: Fraction
    terminal_tail_source: str
    absolute_unit_scale: str


@dataclass(frozen=True)
class _CandidateState:
    plan: PiecePlan
    score: ScoreSpec
    performance: PerformanceSpec


def _jsonable(value: Any) -> Any:
    if isinstance(value, Fraction):
        return [value.numerator, value.denominator]
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in sorted(value.items())}
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _fraction_from_json(value: Any) -> Fraction:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, int) for item in value)
        or value[1] <= 0
    ):
        raise ValueError(f"invalid rational value: {value!r}")
    return Fraction(value[0], value[1])


def _span_from_json(value: Any) -> StructureSpanV0:
    if not isinstance(value, dict) or set(value) != {"start", "end"}:
        raise ValueError("invalid structure span")
    return StructureSpanV0(
        start=_fraction_from_json(value["start"]),
        end=_fraction_from_json(value["end"]),
    )


def compact_structure_descriptor_from_dict(
    value: dict[str, Any],
) -> CompactStructureDescriptorV0:
    """JSON化したV0 descriptorを型へ戻し、構造を検査する。"""
    if value.get("schema_version") != 0:
        raise ValueError("unsupported compact structure descriptor version")
    nodes = tuple(
        StructureNodeV0(
            local_id=item["local_id"],
            node_kind=item["node_kind"],
            parent_local_id=item["parent_local_id"],
            sibling_order=item["sibling_order"],
            start=_fraction_from_json(item["start"]),
            end=_fraction_from_json(item["end"]),
        )
        for item in value["nodes"]
    )
    materials = tuple(
        StructureMaterialV0(
            local_id=item["local_id"],
            occurrence_leaf_ids=tuple(item["occurrence_leaf_ids"]),
            occurrence_spans=tuple(_span_from_json(span) for span in item["occurrence_spans"]),
        )
        for item in value["materials"]
    )
    relations = tuple(
        StructureDerivedRelationV0(
            subject_kind=item["subject_kind"],
            subject_local_id=item["subject_local_id"],
            source_local_id=item["source_local_id"],
            subject_span=tuple(_span_from_json(span) for span in item["subject_span"]),
            source_span=tuple(_span_from_json(span) for span in item["source_span"]),
        )
        for item in value["derived_relations"]
    )
    descriptor = CompactStructureDescriptorV0(
        schema_version=0,
        nodes=nodes,
        materials=materials,
        derived_relations=relations,
        terminal_tail_ratio=_fraction_from_json(value["terminal_tail_ratio"]),
        terminal_tail_source=value["terminal_tail_source"],
        absolute_unit_scale=value["absolute_unit_scale"],
    )
    if _jsonable(descriptor) != value:
        raise ValueError("compact structure descriptor is not canonical")
    return descriptor


def _node_local_ids(plan: PiecePlan) -> dict[str, str]:
    children = {node.node_id: [] for node in plan.nodes}
    by_id = {node.node_id: node for node in plan.nodes}
    for node in plan.nodes:
        if node.parent_id is not None:
            children[node.parent_id].append(node)
    result: dict[str, str] = {}

    def visit(node_id: str, path: tuple[int, ...]) -> None:
        node_children = sorted(children[node_id], key=lambda item: item.order)
        kind = "internal" if node_children else "leaf"
        path_text = "root" if not path else ".".join(str(item) for item in path)
        result[node_id] = f"{kind}-{path_text}"
        for ordinal, child in enumerate(node_children):
            visit(child.node_id, (*path, ordinal))

    visit(by_id[plan.root_node_id].node_id, ())
    return result


def _material_local_ids(
    plan: PiecePlan,
    score: ScoreSpec,
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    node_ids = _node_local_ids(plan)
    leaves, _ = ordered_leaf_schedule(plan, score)
    ordered_materials: list[str] = []
    occurrences: dict[str, list[str]] = {}
    for node, _, _ in leaves:
        assert node.score_material_id is not None
        if node.score_material_id not in occurrences:
            ordered_materials.append(node.score_material_id)
            occurrences[node.score_material_id] = []
        occurrences[node.score_material_id].append(node_ids[node.node_id])
    local_ids = {
        material_id: f"material-{index}" for index, material_id in enumerate(ordered_materials)
    }
    return local_ids, {
        material_id: tuple(occurrences[material_id]) for material_id in ordered_materials
    }


def compact_structure_descriptor(
    plan: PiecePlan,
    score: ScoreSpec,
    *,
    terminal_tail_source: str = "oracle_only",
) -> CompactStructureDescriptorV0:
    """絶対unit尺度と元IRのIDを除いた構成記述を作る。"""
    leaves, intervals = ordered_leaf_schedule(plan, score)
    total_units = intervals[plan.root_node_id][1]
    if total_units <= 0:
        raise ValueError("root interval must have positive length")
    node_ids = _node_local_ids(plan)
    children = {node.node_id: [] for node in plan.nodes}
    for node in plan.nodes:
        if node.parent_id is not None:
            children[node.parent_id].append(node)
    nodes = tuple(
        StructureNodeV0(
            local_id=node_ids[node.node_id],
            node_kind="internal" if children[node.node_id] else "leaf",
            parent_local_id=(node_ids[node.parent_id] if node.parent_id is not None else None),
            sibling_order=node.order,
            start=Fraction(intervals[node.node_id][0], total_units),
            end=Fraction(intervals[node.node_id][1], total_units),
        )
        for node in plan.nodes
    )
    node_spans = {
        node.node_id: (
            StructureSpanV0(
                Fraction(intervals[node.node_id][0], total_units),
                Fraction(intervals[node.node_id][1], total_units),
            ),
        )
        for node in plan.nodes
    }
    material_ids, material_occurrences = _material_local_ids(plan, score)
    leaf_spans = {
        node_ids[node.node_id]: StructureSpanV0(
            Fraction(start, total_units),
            Fraction(end, total_units),
        )
        for node, start, end in leaves
    }
    materials = tuple(
        StructureMaterialV0(
            local_id=material_ids[material_id],
            occurrence_leaf_ids=material_occurrences[material_id],
            occurrence_spans=tuple(
                leaf_spans[leaf_id] for leaf_id in material_occurrences[material_id]
            ),
        )
        for material_id in material_ids
    )
    material_spans = {
        material_id: tuple(leaf_spans[leaf_id] for leaf_id in material_occurrences[material_id])
        for material_id in material_ids
    }
    relations: list[StructureDerivedRelationV0] = []
    for node in plan.nodes:
        if node.derived_from is not None:
            relations.append(
                StructureDerivedRelationV0(
                    subject_kind="node",
                    subject_local_id=node_ids[node.node_id],
                    source_local_id=node_ids[node.derived_from],
                    subject_span=node_spans[node.node_id],
                    source_span=node_spans[node.derived_from],
                )
            )
    material_by_id = {material.material_id: material for material in score.materials}
    for material_id in material_ids:
        material = material_by_id[material_id]
        if material.derived_from is not None:
            relations.append(
                StructureDerivedRelationV0(
                    subject_kind="material",
                    subject_local_id=material_ids[material_id],
                    source_local_id=material_ids[material.derived_from],
                    subject_span=material_spans[material_id],
                    source_span=material_spans[material.derived_from],
                )
            )
    material_lookup = {material.material_id: material for material in score.materials}
    attack_positions = [
        start + note.at_units
        for node, start, _ in leaves
        for note in material_lookup[node.score_material_id].notes
    ]
    if not attack_positions:
        raise ValueError("structure descriptor requires at least one note onset")
    terminal_tail_ratio = Fraction(total_units - max(attack_positions), total_units)
    return CompactStructureDescriptorV0(
        schema_version=0,
        nodes=nodes,
        materials=materials,
        derived_relations=tuple(sorted(relations)),
        terminal_tail_ratio=terminal_tail_ratio,
        terminal_tail_source=terminal_tail_source,
        absolute_unit_scale="quotiented_out",
    )


def _state_fingerprint(state: _CandidateState) -> str:
    return hashlib.sha256(repr((state.plan, state.score, state.performance)).encode()).hexdigest()


def _erase_node_relation(state: _CandidateState, node_id: str) -> _CandidateState:
    nodes = tuple(
        replace(
            node,
            role="release" if node.node_id == node_id and node.role == "return" else node.role,
            derived_from=None if node.node_id == node_id else node.derived_from,
        )
        for node in state.plan.nodes
    )
    return replace(state, plan=replace(state.plan, nodes=nodes))


def _erase_material_relation(state: _CandidateState, material_id: str) -> _CandidateState:
    materials = tuple(
        replace(material, derived_from=None) if material.material_id == material_id else material
        for material in state.score.materials
    )
    return replace(state, score=replace(state.score, materials=materials))


def _structure_variants(state: _CandidateState) -> tuple[tuple[str, _CandidateState], ...]:
    node_ids = _node_local_ids(state.plan)
    material_ids, _ = _material_local_ids(state.plan, state.score)
    result: list[tuple[str, _CandidateState]] = []
    for node in state.plan.nodes:
        if node.derived_from is not None:
            result.append(
                (
                    f"erase-node-derived:{node_ids[node.node_id]}",
                    _erase_node_relation(state, node.node_id),
                )
            )
    for material in state.score.materials:
        if material.derived_from is not None and material.material_id in material_ids:
            result.append(
                (
                    f"erase-material-derived:{material_ids[material.material_id]}",
                    _erase_material_relation(state, material.material_id),
                )
            )
    for target_id in inert_internal_node_ids(state.plan, state.performance):
        plan, performance, _ = flatten_inert_internal_node(
            state.plan,
            state.performance,
            target_id,
        )
        result.append(
            (
                f"flatten-node:{node_ids[target_id]}",
                _CandidateState(plan, state.score, performance),
            )
        )
    for left_id, right_id in neutral_adjacent_leaf_pairs(
        state.plan,
        state.score,
        state.performance,
    ):
        plan, score, performance, _ = merge_neutral_adjacent_materials(
            state.plan,
            state.score,
            state.performance,
            left_id,
            right_id,
        )
        result.append(
            (
                f"merge-boundary:{node_ids[left_id]}+{node_ids[right_id]}",
                _CandidateState(plan, score, performance),
            )
        )
    return tuple(result)


def _rendered_bytes(state: _CandidateState, path: Path) -> tuple[bytes, object]:
    validate_pipeline(state.plan, state.score, state.performance)
    rendered = render_performance(state.plan, state.score, state.performance)
    render_performance_smf(rendered, path)
    return path.read_bytes(), rendered


def _maximum_note_time_difference_ms(base: object, variant: object) -> int | None:
    if len(base.notes) != len(variant.notes):
        return None
    differences: list[int] = []
    for left, right in zip(base.notes, variant.notes, strict=True):
        if (left.pitch, left.velocity) != (right.pitch, right.velocity):
            return None
        differences.extend(
            (abs(left.at_ms - right.at_ms), abs(left.duration_ms - right.duration_ms))
        )
    return max(differences, default=0)


def _initial_upper_bound(fixture: KnownTimingFixture) -> int:
    relation_count = sum(node.derived_from is not None for node in fixture.plan.nodes) + sum(
        material.derived_from is not None for material in fixture.score.materials
    )
    internal_count = sum(
        node_id != fixture.plan.root_node_id
        for node_id in inert_internal_node_ids(fixture.plan, fixture.performance)
    )
    boundary_count = max(0, len(ordered_leaf_schedule(fixture.plan, fixture.score)[0]) - 1)
    return 2 ** (relation_count + internal_count + boundary_count)


def _candidate_closure(
    fixture: KnownTimingFixture,
    temporary_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[_CandidateState], str, int]:
    initial = _CandidateState(fixture.plan, fixture.score, fixture.performance)
    initial_candidate_id = _sha256_json(compact_structure_descriptor(initial.plan, initial.score))
    baseline_bytes, _ = _rendered_bytes(initial, temporary_root / "baseline.mid")
    if baseline_bytes != fixture.source.smf_path.read_bytes():
        raise ValueError(f"fixture SMF mismatch: {fixture.fixture_id}")
    queue: deque[_CandidateState] = deque([initial])
    seen_states = {_state_fingerprint(initial)}
    descriptor_states: dict[str, _CandidateState] = {}
    edges: set[tuple[str, str, str]] = set()
    upper_bound = _initial_upper_bound(fixture)
    render_index = 0
    while queue:
        state = queue.popleft()
        descriptor = compact_structure_descriptor(state.plan, state.score)
        source_id = _sha256_json(descriptor)
        descriptor_states.setdefault(source_id, state)
        for transform_id, variant in _structure_variants(state):
            render_index += 1
            variant_bytes, _ = _rendered_bytes(
                variant,
                temporary_root / f"candidate-{render_index}.mid",
            )
            if variant_bytes != baseline_bytes:
                continue
            variant_descriptor = compact_structure_descriptor(variant.plan, variant.score)
            target_id = _sha256_json(variant_descriptor)
            edges.add((source_id, target_id, transform_id))
            descriptor_states.setdefault(target_id, variant)
            state_id = _state_fingerprint(variant)
            if state_id not in seen_states:
                seen_states.add(state_id)
                queue.append(variant)
        if len(descriptor_states) > upper_bound:
            return [], [], [], "not_converged", upper_bound
    candidates = [
        {
            "candidate_id": candidate_id,
            "origin_status": (
                "oracle_fixture_baseline"
                if candidate_id == initial_candidate_id
                else "oracle_generated_exact_smf_variant"
            ),
            "descriptor": _jsonable(compact_structure_descriptor(state.plan, state.score)),
        }
        for candidate_id, state in sorted(descriptor_states.items())
    ]
    edge_records = [
        {
            "from_candidate_id": source,
            "to_candidate_id": target,
            "transform_id": transform_id,
            "assumptions": [
                "known_ir_is_used_only_for_schema_coverage",
                "targeted_monotone_rewrite",
            ],
            "evidence_status": "oracle_verified_exact_smf",
            "validator_status": "passed",
            "renderer_status": "exact_smf_bytes",
        }
        for source, target, transform_id in sorted(edges)
    ]
    ordered_states = [state for _, state in sorted(descriptor_states.items())]
    return candidates, edge_records, ordered_states, "complete", upper_bound


def score_timing_dependency(fixture_record: dict[str, Any] | None) -> dict[str, Any]:
    """既知位置とScoreTiming候補空間の依存状態を返す。"""
    if fixture_record is None:
        return {
            "status": "oracle_only",
            "candidate_ids": [],
            "span_source": "oracle_only",
        }
    if fixture_record["score_position_result"] == "recovered":
        return {
            "status": "matched_candidate",
            "candidate_ids": sorted(
                match["candidate_id"]
                for match in fixture_record["best_matches"]
                if match["best_scale_interval_match_rate"] == 1.0
            ),
            "span_source": "oracle_only",
        }
    return {
        "status": "outside_candidate_space",
        "candidate_ids": [],
        "span_source": "oracle_only",
    }


def _terminal_observation(
    fixture: KnownTimingFixture,
    candidates: list[dict[str, Any]],
    dependency: dict[str, Any],
) -> dict[str, Any]:
    observed = load_observed_smf(fixture.source.smf_path)
    performance = build_observed_performance(observed)
    event_times = {item.event_id: item.at_us for item in performance.event_times}
    pedal_off_times = [
        event_times[event.event_id]
        for event in observed.events
        if event.message_type == "control_change"
        and int(event.field("control", -1)) == 64
        and int(event.field("value", 0)) < 64
    ]
    end_times = [
        event_times[event.event_id]
        for event in observed.events
        if event.message_type == "end_of_track"
    ]
    last_attack_positions = sorted(
        {
            (position.numerator, position.denominator)
            for candidate in candidates
            for position in (
                Fraction(1) - Fraction(*candidate["descriptor"]["terminal_tail_ratio"]),
            )
        }
    )
    return {
        "last_attack_score_position_candidates": [list(item) for item in last_attack_positions],
        "position_source": "oracle_only",
        "score_timing_match_status": dependency["status"],
        "terminal_tail_assumption_status": "assumed_from_candidate",
        "last_note_off_us": max(note.offset_us for note in performance.notes),
        "last_cc64_off_us": max(pedal_off_times) if pedal_off_times else None,
        "end_of_track_us": max(end_times) if end_times else None,
        "composition_end_status": "unresolved",
    }


def _scale_controls(
    fixture: KnownTimingFixture,
    states: list[_CandidateState],
    temporary_root: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, state in enumerate(states):
        base_descriptor = compact_structure_descriptor(state.plan, state.score)
        base_bytes, base_rendered = _rendered_bytes(
            state,
            temporary_root / f"scale-{index}-base.mid",
        )
        for transform_id, score in (
            ("divisions-only-x2", scale_divisions_only(state.score)),
            ("full-score-resolution-x2", scale_score_resolution(state.score, factor=2)),
        ):
            variant = replace(state, score=score)
            variant_bytes, variant_rendered = _rendered_bytes(
                variant,
                temporary_root / f"scale-{index}-{transform_id}.mid",
            )
            maximum_difference = _maximum_note_time_difference_ms(
                base_rendered,
                variant_rendered,
            )
            pedals_equal = base_rendered.pedals == variant_rendered.pedals
            exact = base_bytes == variant_bytes
            within_tick = (
                not exact
                and maximum_difference is not None
                and maximum_difference <= 1
                and pedals_equal
            )
            records.append(
                {
                    "candidate_id": _sha256_json(base_descriptor),
                    "transform_id": transform_id,
                    "descriptor_equal": compact_structure_descriptor(
                        variant.plan,
                        variant.score,
                    )
                    == base_descriptor,
                    "status": (
                        "exact_smf_collision"
                        if exact
                        else "within_one_transport_tick"
                        if within_tick
                        else "different"
                    ),
                    "maximum_note_time_difference_ms": maximum_difference,
                    "rendered_pedals_literal_equal": pedals_equal,
                }
            )
    return records


def _negative_control(fixture: KnownTimingFixture, temporary_root: Path) -> dict[str, Any]:
    first_material = fixture.score.materials[0]
    first_note = first_material.notes[0]
    changed_material = replace(
        first_material,
        notes=(replace(first_note, pitch=first_note.pitch + 1), *first_material.notes[1:]),
    )
    changed_score = replace(
        fixture.score,
        materials=(changed_material, *fixture.score.materials[1:]),
    )
    original = _CandidateState(fixture.plan, fixture.score, fixture.performance)
    changed = replace(original, score=changed_score)
    original_path = temporary_root / "negative-original.mid"
    changed_path = temporary_root / "negative-changed.mid"
    original_bytes, _ = _rendered_bytes(original, original_path)
    changed_bytes, _ = _rendered_bytes(changed, changed_path)
    original_ledger = load_observed_smf(original_path).ledger_sha256
    changed_ledger = load_observed_smf(changed_path).ledger_sha256
    return {
        "fixture_id": fixture.fixture_id,
        "structure_descriptor_equal": compact_structure_descriptor(
            fixture.plan,
            fixture.score,
        )
        == compact_structure_descriptor(fixture.plan, changed_score),
        "pitch_change_detected": original_bytes != changed_bytes,
        "original_ledger_sha256": original_ledger,
        "changed_ledger_sha256": changed_ledger,
        "ledger_changed": original_ledger != changed_ledger,
    }


def _all_keys(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        result.update(value)
        for child in value.values():
            result.update(_all_keys(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_all_keys(child))
    return result


def _descriptor_measure(descriptor: dict[str, Any]) -> int:
    relation_count = len(descriptor["derived_relations"])
    internal_count = sum(
        node["node_kind"] == "internal" and node["parent_local_id"] is not None
        for node in descriptor["nodes"]
    )
    leaf_count = sum(node["node_kind"] == "leaf" for node in descriptor["nodes"])
    return relation_count + internal_count + max(0, leaf_count - 1)


def _descriptor_roundtrips(descriptor: dict[str, Any]) -> bool:
    try:
        restored = compact_structure_descriptor_from_dict(descriptor)
    except (KeyError, TypeError, ValueError):
        return False
    return _jsonable(restored) == descriptor


def structure_candidate_set_contract_checks(
    record: dict[str, Any],
    *,
    expected_timing_dependency: dict[str, Any],
) -> dict[str, bool]:
    """候補集合の採用前契約を、独立した項目ごとに検査する。"""
    candidates = record["candidates"]
    candidate_ids = {candidate["candidate_id"] for candidate in candidates}
    by_id = {candidate["candidate_id"]: candidate["descriptor"] for candidate in candidates}
    edges = record["candidate_edges"]
    scale_controls = record["scale_controls"]
    timing = record["score_timing_dependency"]
    terminal = record["terminal_observation"]
    baseline_ids = {
        candidate["candidate_id"]
        for candidate in candidates
        if candidate["origin_status"] == "oracle_fixture_baseline"
    }
    reachable = set(baseline_ids)
    changed = True
    while changed:
        changed = False
        for edge in edges:
            if edge["from_candidate_id"] in reachable and edge["to_candidate_id"] not in reachable:
                reachable.add(edge["to_candidate_id"])
                changed = True
    expected_scale_keys = {
        (candidate_id, transform_id)
        for candidate_id in candidate_ids
        for transform_id in ("divisions-only-x2", "full-score-resolution-x2")
    }
    actual_scale_keys = {
        (control["candidate_id"], control["transform_id"]) for control in scale_controls
    }
    position_candidates = terminal["last_attack_score_position_candidates"]
    positions_are_valid = bool(position_candidates) and all(
        isinstance(item, list)
        and len(item) == 2
        and isinstance(item[0], int)
        and isinstance(item[1], int)
        and item[1] > 0
        and 0 <= Fraction(item[0], item[1]) <= 1
        for item in position_candidates
    )
    observed_end_times_are_valid = (
        isinstance(terminal["last_note_off_us"], int)
        and terminal["last_note_off_us"] >= 0
        and isinstance(terminal["end_of_track_us"], int)
        and terminal["end_of_track_us"] >= terminal["last_note_off_us"]
        and (
            terminal["last_cc64_off_us"] is None
            or (
                isinstance(terminal["last_cc64_off_us"], int)
                and 0 <= terminal["last_cc64_off_us"] <= terminal["end_of_track_us"]
            )
        )
    )
    return {
        "candidate_ids_match_descriptors": len(candidate_ids) == len(candidates)
        and all(
            candidate["candidate_id"] == _sha256_json(candidate["descriptor"])
            for candidate in candidates
        ),
        "descriptors_roundtrip": all(
            _descriptor_roundtrips(candidate["descriptor"]) for candidate in candidates
        ),
        "candidate_origins_are_explicit": sum(
            candidate["origin_status"] == "oracle_fixture_baseline" for candidate in candidates
        )
        == 1
        and all(
            candidate["origin_status"]
            in {"oracle_fixture_baseline", "oracle_generated_exact_smf_variant"}
            for candidate in candidates
        ),
        "edge_endpoints_exist": all(
            edge["from_candidate_id"] in candidate_ids and edge["to_candidate_id"] in candidate_ids
            for edge in edges
        ),
        "all_candidates_are_reachable": bool(edges) and reachable == candidate_ids,
        "edge_evidence_is_explicit": all(
            edge["assumptions"]
            and edge["evidence_status"] == "oracle_verified_exact_smf"
            and edge["validator_status"] == "passed"
            and edge["renderer_status"] == "exact_smf_bytes"
            for edge in edges
        ),
        "edges_reduce_structure_measure": all(
            _descriptor_measure(by_id[edge["to_candidate_id"]])
            < _descriptor_measure(by_id[edge["from_candidate_id"]])
            for edge in edges
        ),
        "scale_controls_are_separate_and_valid": len(scale_controls) == len(expected_scale_keys)
        and actual_scale_keys == expected_scale_keys
        and all(control["descriptor_equal"] is True for control in scale_controls)
        and all(
            control["status"] == "exact_smf_collision"
            for control in scale_controls
            if control["transform_id"] == "divisions-only-x2"
        )
        and all(
            control["status"] in {"exact_smf_collision", "within_one_transport_tick"}
            for control in scale_controls
            if control["transform_id"] == "full-score-resolution-x2"
        ),
        "score_timing_dependency_is_valid": timing == expected_timing_dependency
        and timing["status"] in {"matched_candidate", "outside_candidate_space", "oracle_only"}
        and bool(timing["candidate_ids"]) == (timing["status"] == "matched_candidate"),
        "terminal_states_are_separate": positions_are_valid
        and observed_end_times_are_valid
        and terminal["position_source"] == "oracle_only"
        and terminal["terminal_tail_assumption_status"] == "assumed_from_candidate"
        and terminal["composition_end_status"] == "unresolved",
        "candidate_count_is_bounded": 0 < len(candidates) <= record["naive_upper_bound"],
        "forbidden_fields_absent": not record["forbidden_fields"],
    }


def _representation_metrics(record: dict[str, Any]) -> dict[str, Any]:
    descriptor_payloads = [item["descriptor"] for item in record["candidates"]]
    descriptor_bytes = [_canonical_json_bytes(item) for item in descriptor_payloads]
    candidate_count = len(descriptor_bytes)

    def scalar_fields(value: Any, path: tuple[str, ...] = ()) -> dict[tuple[str, ...], bytes]:
        if isinstance(value, dict):
            result: dict[tuple[str, ...], bytes] = {}
            for key, child in sorted(value.items()):
                result.update(scalar_fields(child, (*path, key)))
            return result
        if isinstance(value, list):
            result = {}
            for index, child in enumerate(value):
                result.update(scalar_fields(child, (*path, str(index))))
            return result
        return {path: _canonical_json_bytes(value)}

    common: dict[tuple[str, ...], bytes] = (
        scalar_fields(descriptor_payloads[0]) if descriptor_payloads else {}
    )
    for descriptor in descriptor_payloads[1:]:
        current = scalar_fields(descriptor)
        common = {path: encoded for path, encoded in common.items() if current.get(path) == encoded}
    common_scalar_value_bytes = sum(len(encoded) for encoded in common.values())
    return {
        "fixture_id": record["fixture_id"],
        "candidate_count": candidate_count,
        "explicit_descriptor_bytes": sum(len(item) for item in descriptor_bytes),
        "mean_descriptor_bytes": (
            sum(len(item) for item in descriptor_bytes) / candidate_count
            if candidate_count
            else 0.0
        ),
        "common_scalar_value_bytes_per_descriptor": common_scalar_value_bytes,
        "repeated_common_scalar_value_bytes": common_scalar_value_bytes
        * max(0, candidate_count - 1),
        "set_bytes": len(_canonical_json_bytes(record)),
    }


def _jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(record) + b"\n" for record in records)


def _verify_fixture_root(fixture_root: Path) -> None:
    result = json.loads((fixture_root / "result.json").read_text(encoding="utf-8"))
    manifest = json.loads((fixture_root / "manifest.json").read_text(encoding="utf-8"))
    if result.get("status") != "passed" or manifest.get("status") != "passed":
        raise ValueError("known fixture root is not passed")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("known fixture manifest outputs are invalid")
    for name, expected in outputs.items():
        path = fixture_root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"known fixture manifest hash mismatch: {name}")


def run_structure_equivalence_schema(*, workspace: Path, output_dir: Path) -> dict[str, Any]:
    """4人工設計族の同一SMF構造候補を閉包し、決定的な成果物を書く。"""
    workspace = Path(workspace)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    fixture_root = workspace / ".appendix/score-timing-known-fixtures-v1"
    _verify_fixture_root(fixture_root)
    fixture_records = {
        item["fixture_id"]: item
        for item in (
            json.loads(line)
            for line in (fixture_root / "fixtures.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    fixtures = build_known_timing_fixtures(fixture_root)
    for fixture in fixtures:
        verify_known_timing_fixture(fixture)
    candidate_sets: list[dict[str, Any]] = []
    negative_controls: list[dict[str, Any]] = []
    forbidden = {"notes", "events", "attack_groups", "pitch", "velocity", "duration_units"}
    with tempfile.TemporaryDirectory(prefix="structure-equivalence-") as temporary:
        temporary_root = Path(temporary)
        for fixture in fixtures:
            fixture_temporary = temporary_root / fixture.fixture_id
            fixture_temporary.mkdir(parents=True, exist_ok=True)
            candidates, edges, states, search_status, upper_bound = _candidate_closure(
                fixture,
                fixture_temporary,
            )
            dependency = score_timing_dependency(fixture_records[fixture.fixture_id])
            record = {
                "fixture_id": fixture.fixture_id,
                "source_kind": "oracle_coverage_fixture",
                "source_ledger_sha256": load_observed_smf(fixture.source.smf_path).ledger_sha256,
                "attack_group_rule": "anchored-window-30000us-v1",
                "score_timing_dependency": dependency,
                "search_status": search_status,
                "naive_upper_bound": upper_bound,
                "candidate_count": len(candidates),
                "candidates": candidates,
                "candidate_edges": edges,
                "scale_controls": _scale_controls(
                    fixture,
                    states,
                    fixture_temporary,
                ),
                "terminal_observation": _terminal_observation(
                    fixture,
                    candidates,
                    dependency,
                ),
            }
            record["forbidden_fields"] = sorted(forbidden.intersection(_all_keys(record)))
            record["contract_checks"] = structure_candidate_set_contract_checks(
                record,
                expected_timing_dependency=dependency,
            )
            candidate_sets.append(record)
            negative_controls.append(_negative_control(fixture, fixture_temporary))

    metrics = [_representation_metrics(record) for record in candidate_sets]
    forbidden_violation_count = sum(bool(record["forbidden_fields"]) for record in candidate_sets)
    contract_check_count = sum(len(record["contract_checks"]) for record in candidate_sets)
    contract_check_pass_count = sum(
        value is True for record in candidate_sets for value in record["contract_checks"].values()
    )
    descriptor_hash_verified_count = sum(
        record["contract_checks"]["candidate_ids_match_descriptors"] * record["candidate_count"]
        for record in candidate_sets
    )
    descriptor_roundtrip_verified_count = sum(
        record["contract_checks"]["descriptors_roundtrip"] * record["candidate_count"]
        for record in candidate_sets
    )
    negative_controls_pass = all(
        item["pitch_change_detected"]
        and item["ledger_changed"]
        and item["structure_descriptor_equal"]
        for item in negative_controls
    )
    complete = (
        len(candidate_sets) == 4
        and all(record["search_status"] == "complete" for record in candidate_sets)
        and contract_check_pass_count == contract_check_count
        and negative_controls_pass
        and forbidden_violation_count == 0
    )
    explicit_descriptor_bytes = sum(metric["explicit_descriptor_bytes"] for metric in metrics)
    result = {
        "schema_version": 1,
        "status": "pass" if complete else "partial",
        "run_status": "pass" if complete else "partial",
        "adoption_status": "pending_repeat_comparison" if complete else "rejected",
        "adoption_scope": "oracle_coverage_experiment_only",
        "repeat_comparison_status": "external_repeat_required",
        "fixture_count": len(candidate_sets),
        "converged_fixture_count": sum(
            record["search_status"] == "complete" for record in candidate_sets
        ),
        "candidate_count": sum(record["candidate_count"] for record in candidate_sets),
        "candidate_edge_count": sum(len(record["candidate_edges"]) for record in candidate_sets),
        "descriptor_hash_verified_count": descriptor_hash_verified_count,
        "descriptor_roundtrip_verified_count": descriptor_roundtrip_verified_count,
        "contract_check_count": contract_check_count,
        "contract_check_pass_count": contract_check_pass_count,
        "explicit_descriptor_bytes": explicit_descriptor_bytes,
        "forbidden_field_violation_count": forbidden_violation_count,
        "discovery_status": "not_implemented_or_claimed",
        "terminal_tail_status": "unresolved",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(output_dir / "candidate-sets.jsonl", _jsonl_bytes(candidate_sets))
    atomic_write_bytes(output_dir / "representation-metrics.jsonl", _jsonl_bytes(metrics))
    atomic_write_bytes(
        output_dir / "negative-controls.jsonl",
        _jsonl_bytes(negative_controls),
    )
    atomic_write_json(output_dir / "experiment-result.json", result)
    output_names = (
        "candidate-sets.jsonl",
        "experiment-result.json",
        "negative-controls.jsonl",
        "representation-metrics.jsonl",
    )
    input_paths = {
        "attack_frequency_control.py": workspace
        / "src/llm_musical_composer/attack_frequency_control.py",
        "collision-results.jsonl": workspace
        / ".appendix/structure-identifiability-collisions-v3/collision-results.jsonl",
        "fixtures.jsonl": fixture_root / "fixtures.jsonl",
        "performance_pipeline.py": workspace / "src/llm_musical_composer/performance_pipeline.py",
        "reference_decomposition.py": workspace
        / "src/llm_musical_composer/reference_decomposition.py",
        "score_timing_known_fixtures.py": workspace
        / "src/llm_musical_composer/score_timing_known_fixtures.py",
        "structure_equivalence_schema.py": workspace
        / "src/llm_musical_composer/structure_equivalence_schema.py",
        "structure_identifiability.py": workspace
        / "src/llm_musical_composer/structure_identifiability.py",
    }
    for fixture in fixtures:
        input_paths[f"{fixture.fixture_id}/final.mid"] = fixture.source.smf_path
        input_paths[f"{fixture.fixture_id}/piece-plan.dsl"] = fixture.source.piece_path
        input_paths[f"{fixture.fixture_id}/score-spec.dsl"] = fixture.source.score_path
        input_paths[f"{fixture.fixture_id}/performance-spec.dsl"] = fixture.source.performance_path
    manifest = {
        "schema_version": 1,
        "status": result["status"],
        "inputs": {name: sha256_file(path) for name, path in sorted(input_paths.items())},
        "outputs": {name: sha256_file(output_dir / name) for name in output_names},
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return result


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    result = run_structure_equivalence_schema(
        workspace=arguments.workspace,
        output_dir=arguments.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
