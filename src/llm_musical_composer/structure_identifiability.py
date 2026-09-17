"""異なる潜在構造が同じSMFを生成する固定反例を診断する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import Any

from llm_musical_composer.attack_frequency_control import scale_score_resolution
from llm_musical_composer.performance_pipeline import (
    NodePerformance,
    PerformanceSpec,
    PiecePlan,
    PlanNode,
    ScoreSpec,
    ordered_leaf_schedule,
    render_performance,
    render_performance_smf,
    resolve_effective_profile,
    validate_pipeline,
)
from llm_musical_composer.pipeline_dsl import (
    parse_performance_spec,
    parse_piece_plan,
    parse_score_spec,
)
from llm_musical_composer.reference_decomposition import load_observed_smf
from llm_musical_composer.run_state import atomic_write_bytes, atomic_write_json, sha256_file
from llm_musical_composer.score_timing_known_fixtures import (
    KnownTimingFixture,
    build_known_timing_fixtures,
)

_TRANSFORM_IDS = (
    "erase-recurrence",
    "flatten-inert-node",
    "merge-neutral-adjacent",
    "divisions-only-x2",
    "full-score-resolution-x2",
)


def _jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")


def _tree_shape(plan: PiecePlan) -> object:
    children: dict[str, list[PlanNode]] = {node.node_id: [] for node in plan.nodes}
    for node in plan.nodes:
        if node.parent_id is not None:
            children[node.parent_id].append(node)

    def visit(node_id: str) -> object:
        ordered = sorted(children[node_id], key=lambda item: item.order)
        return "leaf" if not ordered else [visit(node.node_id) for node in ordered]

    return visit(plan.root_node_id)


def _material_boundaries(plan: PiecePlan, score: ScoreSpec) -> tuple[int, ...]:
    leaves, _ = ordered_leaf_schedule(plan, score)
    return tuple(end for _, _, end in leaves[:-1])


def _depths(plan: PiecePlan) -> dict[str, int]:
    by_id = {node.node_id: node for node in plan.nodes}
    result: dict[str, int] = {}
    for node in plan.nodes:
        depth = 0
        current = node
        while current.parent_id is not None:
            depth += 1
            current = by_id[current.parent_id]
        result[node.node_id] = depth
    return result


def _repair_plan_references(plan: PiecePlan, replacement_id: str | None = None) -> PiecePlan:
    """平坦化後の参照を、存在、先行順、深さの契約へ合わせる。"""
    nodes = list(plan.nodes)
    positions = {node.node_id: index for index, node in enumerate(nodes)}
    depths = _depths(plan)
    valid_ids = set(positions)
    repaired: list[PlanNode] = []
    for node in nodes:
        derived = node.derived_from
        contrast = node.contrasts_with
        if derived not in valid_ids and derived is not None:
            derived = replacement_id
        if contrast not in valid_ids and contrast is not None:
            contrast = replacement_id

        def valid(reference: str | None, node_id: str = node.node_id) -> bool:
            return (
                reference is not None
                and reference in valid_ids
                and positions[reference] < positions[node_id]
                and depths[reference] == depths[node_id]
            )

        derived = derived if valid(derived) else None
        contrast = contrast if valid(contrast) else None
        role = "release" if node.role == "return" and derived is None else node.role
        repaired.append(
            replace(
                node,
                role=role,
                derived_from=derived,
                contrasts_with=contrast,
            )
        )
    return replace(plan, nodes=tuple(repaired))


def erase_recurrence_annotations(
    plan: PiecePlan,
    score: ScoreSpec,
) -> tuple[PiecePlan, ScoreSpec] | None:
    """rendererが消費しない反復注釈を消す。"""
    applicable = any(node.derived_from is not None for node in plan.nodes) or any(
        material.derived_from is not None for material in score.materials
    )
    if not applicable:
        return None
    nodes = tuple(
        replace(
            node,
            role="release" if node.role == "return" else node.role,
            derived_from=None,
        )
        for node in plan.nodes
    )
    materials = tuple(replace(material, derived_from=None) for material in score.materials)
    return replace(plan, nodes=nodes), replace(score, materials=materials)


def inert_internal_node_ids(
    plan: PiecePlan,
    performance: PerformanceSpec,
) -> tuple[str, ...]:
    """明示演奏指定を持たない内部nodeをplan順で返す。"""
    children: dict[str, list[PlanNode]] = {node.node_id: [] for node in plan.nodes}
    for node in plan.nodes:
        if node.parent_id is not None:
            children[node.parent_id].append(node)
    explicit_ids = {item.node_id for item in performance.node_performances}
    return tuple(
        node.node_id
        for node in plan.nodes
        if node.node_id != plan.root_node_id
        and children[node.node_id]
        and node.score_material_id is None
        and node.node_id not in explicit_ids
    )


def flatten_inert_internal_node(
    plan: PiecePlan,
    performance: PerformanceSpec,
    node_id: str,
) -> tuple[PiecePlan, PerformanceSpec, str]:
    """指定した演奏非依存の内部nodeを一段平坦化する。"""
    if node_id not in inert_internal_node_ids(plan, performance):
        raise ValueError(f"node is not an inert internal node: {node_id}")
    children: dict[str, list[PlanNode]] = {node.node_id: [] for node in plan.nodes}
    for node in plan.nodes:
        if node.parent_id is not None:
            children[node.parent_id].append(node)
    target = next(node for node in plan.nodes if node.node_id == node_id)
    assert target.parent_id is not None
    target_children = sorted(children[target.node_id], key=lambda item: item.order)
    parent_children = sorted(children[target.parent_id], key=lambda item: item.order)
    expanded: list[str] = []
    for child in parent_children:
        if child.node_id == target.node_id:
            expanded.extend(item.node_id for item in target_children)
        else:
            expanded.append(child.node_id)
    parent_orders = {node_id: order for order, node_id in enumerate(expanded)}
    target_child_ids = {item.node_id for item in target_children}
    changed_nodes: list[PlanNode] = []
    for node in plan.nodes:
        if node.node_id == target.node_id:
            continue
        if node.node_id in target_child_ids:
            changed_nodes.append(
                replace(
                    node,
                    parent_id=target.parent_id,
                    order=parent_orders[node.node_id],
                )
            )
        elif node.parent_id == target.parent_id:
            changed_nodes.append(replace(node, order=parent_orders[node.node_id]))
        else:
            changed_nodes.append(node)
    changed_plan = replace(plan, nodes=tuple(changed_nodes))
    changed_plan = _repair_plan_references(
        changed_plan,
        replacement_id=target_children[0].node_id,
    )
    changed_performance = replace(
        performance,
        node_performances=tuple(
            item for item in performance.node_performances if item.node_id != target.node_id
        ),
    )
    return changed_plan, changed_performance, target.node_id


def flatten_first_inert_internal_node(
    plan: PiecePlan,
    performance: PerformanceSpec,
) -> tuple[PiecePlan, PerformanceSpec, str] | None:
    """明示演奏指定を持たない最初の内部nodeを一段平坦化する。"""
    candidates = inert_internal_node_ids(plan, performance)
    if not candidates:
        return None
    return flatten_inert_internal_node(plan, performance, candidates[0])


def _effective_profile_tuple(
    node: PlanNode,
    nodes: dict[str, PlanNode],
    settings: dict[str, NodePerformance],
) -> tuple[str, ...]:
    return tuple(
        resolve_effective_profile(node, nodes, settings, attribute, default)[1]
        for attribute, default in (
            ("timing_profile", "neutral"),
            ("dynamics_profile", "steady"),
            ("articulation_profile", "score"),
            ("coordination_profile", "score"),
            ("pedal_profile", "none"),
        )
    )


def _merge_eligible(
    left: PlanNode,
    right: PlanNode,
    *,
    plan: PiecePlan,
    score: ScoreSpec,
    performance: PerformanceSpec,
) -> bool:
    if left.parent_id != right.parent_id or right.order != left.order + 1:
        return False
    assert left.score_material_id is not None and right.score_material_id is not None
    material_by_id = {material.material_id: material for material in score.materials}
    left_material = material_by_id[left.score_material_id]
    right_material = material_by_id[right.score_material_id]
    reference_counts = Counter(
        node.score_material_id for node in plan.nodes if node.score_material_id is not None
    )
    referenced_materials = {
        material.derived_from for material in score.materials if material.derived_from is not None
    }
    referenced_nodes = {
        reference
        for node in plan.nodes
        for reference in (node.derived_from, node.contrasts_with)
        if reference is not None
    }
    if (
        reference_counts[left.score_material_id] != 1
        or reference_counts[right.score_material_id] != 1
        or left.score_material_id in referenced_materials
        or right.score_material_id in referenced_materials
        or left.node_id in referenced_nodes
        or right.node_id in referenced_nodes
        or left_material.derived_from is not None
        or right_material.derived_from is not None
        or left_material.directions
        or right_material.directions
        or left_material.harmonies
        or right_material.harmonies
        or left_material.foreground_voice != right_material.foreground_voice
    ):
        return False
    nodes = {node.node_id: node for node in plan.nodes}
    settings = {item.node_id: item for item in performance.node_performances}
    left_profiles = _effective_profile_tuple(left, nodes, settings)
    right_profiles = _effective_profile_tuple(right, nodes, settings)
    allowed = (
        left_profiles[0] == "neutral"
        and left_profiles[1] == "steady"
        and left_profiles[2] == "score"
        and left_profiles[3] in {"score", "aligned"}
        and left_profiles[4] == "none"
    )
    return allowed and left_profiles == right_profiles


def neutral_adjacent_leaf_pairs(
    plan: PiecePlan,
    score: ScoreSpec,
    performance: PerformanceSpec,
) -> tuple[tuple[str, str], ...]:
    """狭い中立条件を満たす隣接素材境界を葉順で返す。"""
    leaves, _ = ordered_leaf_schedule(plan, score)
    leaf_nodes = [node for node, _, _ in leaves]
    result: list[tuple[str, str]] = []
    for left, right in pairwise(leaf_nodes):
        if _merge_eligible(
            left,
            right,
            plan=plan,
            score=score,
            performance=performance,
        ):
            result.append((left.node_id, right.node_id))
    return tuple(result)


def merge_neutral_adjacent_materials(
    plan: PiecePlan,
    score: ScoreSpec,
    performance: PerformanceSpec,
    left_node_id: str,
    right_node_id: str,
) -> tuple[PiecePlan, ScoreSpec, PerformanceSpec, tuple[str, str]]:
    """指定した中立な隣接素材境界を一つ消す。"""
    if (left_node_id, right_node_id) not in neutral_adjacent_leaf_pairs(
        plan,
        score,
        performance,
    ):
        raise ValueError(
            f"nodes are not an eligible neutral adjacent pair: {left_node_id}, {right_node_id}"
        )
    by_id = {node.node_id: node for node in plan.nodes}
    left, right = by_id[left_node_id], by_id[right_node_id]
    assert left.score_material_id is not None and right.score_material_id is not None
    material_by_id = {material.material_id: material for material in score.materials}
    left_material = material_by_id[left.score_material_id]
    right_material = material_by_id[right.score_material_id]
    shifted_notes = tuple(
        replace(note, at_units=note.at_units + left_material.length_units)
        for note in right_material.notes
    )
    merged_material = replace(
        left_material,
        length_units=left_material.length_units + right_material.length_units,
        notes=left_material.notes + shifted_notes,
    )
    changed_materials = tuple(
        merged_material if material.material_id == left_material.material_id else material
        for material in score.materials
        if material.material_id != right_material.material_id
    )

    siblings = sorted(
        (node for node in plan.nodes if node.parent_id == left.parent_id),
        key=lambda item: item.order,
    )
    sibling_ids = [node.node_id for node in siblings if node.node_id != right.node_id]
    orders = {node_id: index for index, node_id in enumerate(sibling_ids)}
    changed_nodes = tuple(
        replace(node, order=orders[node.node_id]) if node.parent_id == left.parent_id else node
        for node in plan.nodes
        if node.node_id != right.node_id
    )
    changed_plan = replace(plan, nodes=changed_nodes)
    changed_score = replace(score, materials=changed_materials)
    changed_performance = replace(
        performance,
        node_performances=tuple(
            item for item in performance.node_performances if item.node_id != right.node_id
        ),
    )
    return (
        changed_plan,
        changed_score,
        changed_performance,
        (left.node_id, right.node_id),
    )


def merge_first_neutral_adjacent_materials(
    plan: PiecePlan,
    score: ScoreSpec,
    performance: PerformanceSpec,
) -> tuple[PiecePlan, ScoreSpec, PerformanceSpec, tuple[str, str]] | None:
    """狭い中立条件を満たす最初の隣接素材境界を一つ消す。"""
    candidates = neutral_adjacent_leaf_pairs(plan, score, performance)
    if not candidates:
        return None
    return merge_neutral_adjacent_materials(
        plan,
        score,
        performance,
        *candidates[0],
    )


def scale_divisions_only(score: ScoreSpec, *, factor: int = 2) -> ScoreSpec:
    """rendererが消費しないdivisionsだけを整数倍する。"""
    if factor <= 1:
        raise ValueError("scale factor must be greater than one")
    return replace(score, divisions=score.divisions * factor)


def _sounding_notes(rendered: object) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(
        (note.at_ms, note.duration_ms, note.pitch, note.velocity) for note in rendered.notes
    )


def _sounding_pedals(rendered: object) -> tuple[tuple[int, int], ...]:
    return tuple((pedal.at_ms, pedal.value) for pedal in rendered.pedals)


def _provenance(rendered: object) -> tuple[tuple[str, str], ...]:
    return tuple((note.event_id, note.occurrence_node_id) for note in rendered.notes)


def _maximum_note_time_difference_ms(base: object, variant: object) -> int | None:
    if len(base.notes) != len(variant.notes):
        return None
    paired = zip(base.notes, variant.notes, strict=True)
    differences = []
    for left, right in paired:
        if (left.pitch, left.velocity) != (right.pitch, right.velocity):
            return None
        differences.extend(
            (abs(left.at_ms - right.at_ms), abs(left.duration_ms - right.duration_ms))
        )
    return max(differences, default=0)


def compare_structure_variant(
    fixture: KnownTimingFixture,
    *,
    transform_id: str,
    variant_plan: PiecePlan,
    variant_score: ScoreSpec,
    variant_performance: PerformanceSpec,
    temporary_root: Path,
) -> dict[str, Any]:
    """潜在IRの差とSMFへ出る意味の差を別々に比較する。"""
    validate_pipeline(variant_plan, variant_score, variant_performance)
    original = render_performance(fixture.plan, fixture.score, fixture.performance)
    variant = render_performance(variant_plan, variant_score, variant_performance)
    temporary_root = Path(temporary_root)
    temporary_root.mkdir(parents=True, exist_ok=True)
    original_path = temporary_root / f"{fixture.fixture_id}-{transform_id}-original.mid"
    variant_path = temporary_root / f"{fixture.fixture_id}-{transform_id}-variant.mid"
    render_performance_smf(original, original_path)
    render_performance_smf(variant, variant_path)
    original_bytes = original_path.read_bytes()
    variant_bytes = variant_path.read_bytes()
    original_observed = load_observed_smf(original_path)
    variant_observed = load_observed_smf(variant_path)
    maximum_difference = _maximum_note_time_difference_ms(original, variant)
    exact = original_bytes == variant_bytes
    sounding_pedals_equal = _sounding_pedals(original) == _sounding_pedals(variant)
    within_tick = (
        not exact
        and maximum_difference is not None
        and maximum_difference <= 1
        and sounding_pedals_equal
    )
    return {
        "fixture_id": fixture.fixture_id,
        "source_family_id": fixture.fixture_id,
        "transform_id": transform_id,
        "status": (
            "exact_smf_collision"
            if exact
            else "within_one_transport_tick"
            if within_tick
            else "different"
        ),
        "smf_bytes_equal": exact,
        "smf_sha256_equal": hashlib.sha256(original_bytes).digest()
        == hashlib.sha256(variant_bytes).digest(),
        "smf_performance_meaning_equal": (
            _sounding_notes(original) == _sounding_notes(variant) and sounding_pedals_equal
        ),
        "rendered_notes_literal_equal": original.notes == variant.notes,
        "rendered_pedals_literal_equal": original.pedals == variant.pedals,
        "performed_provenance_equal": _provenance(original) == _provenance(variant),
        "event_ledger_equal": original_observed.events == variant_observed.events,
        "node_intervals_equal": original.node_intervals == variant.node_intervals,
        "lineage_equal": original.lineage == variant.lineage,
        "maximum_note_time_difference_ms": maximum_difference,
        "latent_plan_equal": fixture.plan == variant_plan,
        "latent_score_equal": fixture.score == variant_score,
        "latent_performance_equal": fixture.performance == variant_performance,
        "tree_shape_equal": _tree_shape(fixture.plan) == _tree_shape(variant_plan),
        "material_boundaries_equal": _material_boundaries(fixture.plan, fixture.score)
        == _material_boundaries(variant_plan, variant_score),
        "original_material_count": len(fixture.score.materials),
        "variant_material_count": len(variant_score.materials),
    }


def _not_applicable(fixture: KnownTimingFixture, transform_id: str, reason: str) -> dict[str, Any]:
    return {
        "fixture_id": fixture.fixture_id,
        "source_family_id": fixture.fixture_id,
        "transform_id": transform_id,
        "status": "not_applicable",
        "reason": reason,
    }


def _pitch_negative_variant(fixture: KnownTimingFixture) -> ScoreSpec:
    first_material = fixture.score.materials[0]
    first_note = first_material.notes[0]
    changed_material = replace(
        first_material,
        notes=(replace(first_note, pitch=first_note.pitch + 1), *first_material.notes[1:]),
    )
    return replace(
        fixture.score,
        materials=(changed_material, *fixture.score.materials[1:]),
    )


def verify_known_timing_fixture(fixture: KnownTimingFixture) -> None:
    """保存済みfixtureと現在の既知IRのDSL、lineage、SMFを照合する。"""
    if parse_piece_plan(fixture.source.piece_path.read_text(encoding="utf-8")) != fixture.plan:
        raise ValueError(f"fixture PiecePlan mismatch: {fixture.fixture_id}")
    if parse_score_spec(fixture.source.score_path.read_text(encoding="utf-8")) != fixture.score:
        raise ValueError(f"fixture ScoreSpec mismatch: {fixture.fixture_id}")
    if (
        parse_performance_spec(fixture.source.performance_path.read_text(encoding="utf-8"))
        != fixture.performance
    ):
        raise ValueError(f"fixture PerformanceSpec mismatch: {fixture.fixture_id}")
    evidence = json.loads(fixture.source.evidence_path.read_text(encoding="utf-8"))
    rendered = render_performance(fixture.plan, fixture.score, fixture.performance)
    if evidence.get("status") != "passed" or evidence.get("lineage") != list(rendered.lineage):
        raise ValueError(f"fixture lineage mismatch: {fixture.fixture_id}")
    with tempfile.TemporaryDirectory(prefix="structure-fixture-check-") as temporary:
        regenerated = Path(temporary) / "regenerated.mid"
        render_performance_smf(rendered, regenerated)
        if regenerated.read_bytes() != fixture.source.smf_path.read_bytes():
            raise ValueError(f"fixture SMF mismatch: {fixture.fixture_id}")


def _fixture_records(
    fixture: KnownTimingFixture,
    temporary_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    verify_known_timing_fixture(fixture)
    records: list[dict[str, Any]] = []

    recurrence = erase_recurrence_annotations(fixture.plan, fixture.score)
    if recurrence is None:
        records.append(_not_applicable(fixture, "erase-recurrence", "no recurrence annotation"))
    else:
        plan, score = recurrence
        records.append(
            compare_structure_variant(
                fixture,
                transform_id="erase-recurrence",
                variant_plan=plan,
                variant_score=score,
                variant_performance=fixture.performance,
                temporary_root=temporary_root,
            )
        )

    flattened = flatten_first_inert_internal_node(fixture.plan, fixture.performance)
    if flattened is None:
        records.append(
            _not_applicable(fixture, "flatten-inert-node", "no inert non-root internal node")
        )
    else:
        plan, performance, removed_node_id = flattened
        record = compare_structure_variant(
            fixture,
            transform_id="flatten-inert-node",
            variant_plan=plan,
            variant_score=fixture.score,
            variant_performance=performance,
            temporary_root=temporary_root,
        )
        record["removed_node_id"] = removed_node_id
        records.append(record)

    merged = merge_first_neutral_adjacent_materials(
        fixture.plan,
        fixture.score,
        fixture.performance,
    )
    if merged is None:
        records.append(
            _not_applicable(
                fixture,
                "merge-neutral-adjacent",
                "no adjacent leaves meet the fixed neutral merge contract",
            )
        )
    else:
        plan, score, performance, merged_leaf_ids = merged
        record = compare_structure_variant(
            fixture,
            transform_id="merge-neutral-adjacent",
            variant_plan=plan,
            variant_score=score,
            variant_performance=performance,
            temporary_root=temporary_root,
        )
        record["merged_leaf_ids"] = list(merged_leaf_ids)
        records.append(record)

    records.append(
        compare_structure_variant(
            fixture,
            transform_id="divisions-only-x2",
            variant_plan=fixture.plan,
            variant_score=scale_divisions_only(fixture.score),
            variant_performance=fixture.performance,
            temporary_root=temporary_root,
        )
    )
    records.append(
        compare_structure_variant(
            fixture,
            transform_id="full-score-resolution-x2",
            variant_plan=fixture.plan,
            variant_score=scale_score_resolution(fixture.score, factor=2),
            variant_performance=fixture.performance,
            temporary_root=temporary_root,
        )
    )

    negative_score = _pitch_negative_variant(fixture)
    negative = compare_structure_variant(
        fixture,
        transform_id="pitch-plus-one-negative-control",
        variant_plan=fixture.plan,
        variant_score=negative_score,
        variant_performance=fixture.performance,
        temporary_root=temporary_root,
    )
    negative["detected"] = (
        negative["status"] == "different"
        and negative["smf_bytes_equal"] is False
        and negative["smf_performance_meaning_equal"] is False
    )
    return records, negative


def run_structure_identifiability(*, workspace: Path, output_dir: Path) -> dict[str, Any]:
    """4人工設計族へ固定変換を適用し、決定的な成果物を書く。"""
    workspace = Path(workspace)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    fixture_root = workspace / ".appendix/score-timing-known-fixtures-v1"
    root_result = json.loads((fixture_root / "result.json").read_text(encoding="utf-8"))
    root_manifest = json.loads((fixture_root / "manifest.json").read_text(encoding="utf-8"))
    if root_result.get("status") != "passed" or root_manifest.get("status") != "passed":
        raise ValueError("known fixture root is not passed")
    fixtures = build_known_timing_fixtures(fixture_root)
    records: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="structure-identifiability-") as temporary:
        temporary_root = Path(temporary)
        for fixture in fixtures:
            fixture_rows, negative = _fixture_records(fixture, temporary_root)
            records.extend(fixture_rows)
            negatives.append(negative)

    transform_summary: dict[str, dict[str, int]] = {}
    for transform_id in _TRANSFORM_IDS:
        items = [record for record in records if record["transform_id"] == transform_id]
        transform_summary[transform_id] = {
            "applicable_count": sum(item["status"] != "not_applicable" for item in items),
            "exact_smf_collision_count": sum(
                item["status"] == "exact_smf_collision" for item in items
            ),
            "not_applicable_count": sum(item["status"] == "not_applicable" for item in items),
            "within_one_transport_tick_count": sum(
                item["status"] == "within_one_transport_tick" for item in items
            ),
        }
    negative_detected_count = sum(item["detected"] is True for item in negatives)
    allowed_statuses = {
        "exact_smf_collision",
        "within_one_transport_tick",
        "different",
        "not_applicable",
    }
    complete = (
        len(fixtures) == 4
        and len(records) == 20
        and all(
            sum(record["transform_id"] == transform_id for record in records) == 4
            for transform_id in _TRANSFORM_IDS
        )
        and negative_detected_count == 4
        and all(record["status"] in allowed_statuses for record in records)
    )
    result = {
        "schema_version": 1,
        "status": "pass" if complete else "partial",
        "source_fixture_count": len(fixtures),
        "artificial_design_family_count": len({fixture.fixture_id for fixture in fixtures}),
        "independence_claim": "not_statistical_independence",
        "comparison_count": len(records),
        "negative_control_count": len(negatives),
        "negative_control_detected_count": negative_detected_count,
        "transform_summary": transform_summary,
        "conclusion_scope": (
            "exact collisions refute universal unique recovery for the changed latent fields; "
            "they do not identify a preferred alternative"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(output_dir / "collision-results.jsonl", _jsonl_bytes(records))
    atomic_write_bytes(output_dir / "negative-controls.jsonl", _jsonl_bytes(negatives))
    atomic_write_json(output_dir / "experiment-result.json", result)
    output_names = (
        "collision-results.jsonl",
        "experiment-result.json",
        "negative-controls.jsonl",
    )
    input_paths = {
        "attack_frequency_control.py": workspace
        / "src/llm_musical_composer/attack_frequency_control.py",
        "fixtures.jsonl": fixture_root / "fixtures.jsonl",
        "fixtures-manifest.json": fixture_root / "manifest.json",
        "performance_pipeline.py": workspace / "src/llm_musical_composer/performance_pipeline.py",
        "pipeline_dsl.py": workspace / "src/llm_musical_composer/pipeline_dsl.py",
        "reference_decomposition.py": workspace
        / "src/llm_musical_composer/reference_decomposition.py",
        "run_state.py": workspace / "src/llm_musical_composer/run_state.py",
        "score_timing_known_fixtures.py": workspace
        / "src/llm_musical_composer/score_timing_known_fixtures.py",
        "structure_identifiability.py": workspace
        / "src/llm_musical_composer/structure_identifiability.py",
    }
    for fixture in fixtures:
        prefix = fixture.fixture_id
        input_paths[f"{prefix}/piece-plan.dsl"] = fixture.source.piece_path
        input_paths[f"{prefix}/score-spec.dsl"] = fixture.source.score_path
        input_paths[f"{prefix}/performance-spec.dsl"] = fixture.source.performance_path
        input_paths[f"{prefix}/final.mid"] = fixture.source.smf_path
        input_paths[f"{prefix}/result.json"] = fixture.source.evidence_path
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
    result = run_structure_identifiability(
        workspace=arguments.workspace,
        output_dir=arguments.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
