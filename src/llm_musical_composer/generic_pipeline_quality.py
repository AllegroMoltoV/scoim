"""固有区分IDへ依存しない、3分間の独奏鍵盤曲生成に必要な最低品質を検査する。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Any

from llm_musical_composer.performance_pipeline import (
    PerformanceSpec,
    PiecePlan,
    PlanNode,
    RenderedPerformance,
    ScoreSpec,
    ordered_leaf_schedule,
    resolve_effective_profile,
    validate_pipeline,
    validate_score_spec,
)
from llm_musical_composer.recurrence_analysis import analyze_recurrences
from llm_musical_composer.recurrence_quality import (
    analyze_foreground_dissonance,
    analyze_material_harmony,
    analyze_rendered_harmony,
)

TARGET_DURATION_MS = 180_000
MINIMUM_ENDING_HOLD_MS = 2_000
LOW_PITCH_BOUNDARY = 48
MINIMUM_LOW_SPACING_SEMITONES = 7


def _plan_children(plan: PiecePlan) -> dict[str, tuple[PlanNode, ...]]:
    children: dict[str, list[PlanNode]] = defaultdict(list)
    for node in plan.nodes:
        if node.parent_id is not None:
            children[node.parent_id].append(node)
    return {
        node_id: tuple(sorted(items, key=lambda item: item.order))
        for node_id, items in children.items()
    }


def _subtree_leaf_material_ids(
    node: PlanNode,
    children: dict[str, tuple[PlanNode, ...]],
) -> frozenset[str]:
    descendants = children.get(node.node_id, ())
    if not descendants:
        assert node.score_material_id is not None
        return frozenset({node.score_material_id})
    return frozenset(
        material_id
        for child in descendants
        for material_id in _subtree_leaf_material_ids(child, children)
    )


def _ordered_plan_leaves(plan: PiecePlan) -> tuple[PlanNode, ...]:
    children = _plan_children(plan)
    node_by_id = {node.node_id: node for node in plan.nodes}
    leaves: list[PlanNode] = []

    def visit(node: PlanNode) -> None:
        descendants = children.get(node.node_id, ())
        if not descendants:
            leaves.append(node)
            return
        for child in descendants:
            visit(child)

    visit(node_by_id[plan.root_node_id])
    return tuple(leaves)


def evaluate_generic_piece_plan_quality(
    plan: PiecePlan,
    *,
    declared_cues: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """後段IRなしで判定できる構成上の最低条件を検査する。"""

    cue_map = declared_cues or {}
    failures: list[str] = []
    if not any(node.derived_from is not None for node in plan.nodes):
        failures.append("missing-recurrence")
    if not any(node.contrasts_with is not None for node in plan.nodes):
        failures.append("missing-contrast")
    children = _plan_children(plan)
    node_by_id = {node.node_id: node for node in plan.nodes}
    for target in plan.nodes:
        if target.derived_from is None:
            continue
        source = node_by_id[target.derived_from]
        target_is_leaf = not children.get(target.node_id)
        source_is_leaf = not children.get(source.node_id)
        if target_is_leaf != source_is_leaf:
            failures.append(f"unassessable-recurrence:{target.node_id}")
        elif target_is_leaf:
            if (
                target.score_material_id != source.score_material_id
                and target.node_id not in cue_map
            ):
                failures.append(f"unassessable-recurrence:{target.node_id}")
        elif (
            not (
                _subtree_leaf_material_ids(target, children)
                & _subtree_leaf_material_ids(source, children)
            )
            and target.node_id not in cue_map
        ):
            failures.append(f"unassessable-recurrence:{target.node_id}")
    leaves = _ordered_plan_leaves(plan)
    if leaves and leaves[-1].derived_from is not None:
        failures.append(f"final-leaf-recurrence:{leaves[-1].node_id}")
    return {"schema_version": 1, "passes": not failures, "failures": failures}


def _score_ending_metrics(plan: PiecePlan, score: ScoreSpec) -> dict[str, Any]:
    leaves, intervals = ordered_leaf_schedule(plan, score)
    materials = {material.material_id: material for material in score.materials}
    total_units = intervals[plan.root_node_id][1]
    required = sorted({plan.tonal_center, (plan.tonal_center + 7) % 12})
    if not leaves:
        return {
            "leaf_node_id": None,
            "material_id": None,
            "dedicated_single_harmony_material": False,
            "attack_units": None,
            "note_count": 0,
            "pitch_classes": [],
            "minimum_duration_units": 0,
            "minimum_nominal_duration_ms": 0,
            "notes_reach_material_end": False,
            "prior_key_carryover_count": 0,
            "total_occurrence_units": total_units,
            "required_pitch_classes": required,
        }
    leaf = leaves[-1][0]
    assert leaf.score_material_id is not None
    material = materials[leaf.score_material_id]
    usage_count = sum(node.score_material_id == material.material_id for node, _, _ in leaves)
    dedicated = (
        usage_count == 1
        and len(material.harmonies) == 1
        and material.harmonies[0].at_units == 0
        and material.harmonies[0].duration_units == material.length_units
        and material.harmonies[0].root_pitch_class == plan.tonal_center
    )
    attack_units = max((note.at_units for note in material.notes), default=None)
    chord = (
        ()
        if attack_units is None
        else tuple(note for note in material.notes if note.at_units == attack_units)
    )
    minimum_duration_units = min((note.duration_units for note in chord), default=0)
    minimum_nominal_duration_ms = (
        minimum_duration_units * TARGET_DURATION_MS // total_units if total_units else 0
    )
    return {
        "leaf_node_id": leaf.node_id,
        "material_id": material.material_id,
        "dedicated_single_harmony_material": dedicated,
        "attack_units": attack_units,
        "note_count": len(chord),
        "pitch_classes": sorted({note.pitch % 12 for note in chord}),
        "minimum_duration_units": minimum_duration_units,
        "minimum_nominal_duration_ms": minimum_nominal_duration_ms,
        "notes_reach_material_end": bool(chord)
        and all(note.at_units + note.duration_units == material.length_units for note in chord),
        "prior_key_carryover_count": 0
        if attack_units is None
        else sum(
            note.at_units < attack_units < note.at_units + note.duration_units
            for note in material.notes
        ),
        "total_occurrence_units": total_units,
        "required_pitch_classes": required,
    }


def evaluate_generic_score_quality(plan: PiecePlan, score: ScoreSpec) -> dict[str, Any]:
    """ScoreSpecだけで判定できる一般品質を返す。"""

    validate_score_spec(plan, score)
    failures: list[str] = []
    voices = {note.voice for material in score.materials for note in material.notes}
    if "upper" not in voices:
        failures.append("missing-upper-voice")
    if "lower" not in voices:
        failures.append("missing-lower-voice")
    has_compound_texture = any(
        len({note.pitch for note in material.notes if note.at_units == onset}) >= 2
        for material in score.materials
        for onset in {note.at_units for note in material.notes}
    )
    if not has_compound_texture:
        failures.append("missing-piano-texture")

    harmony = tuple(
        analyze_material_harmony(
            score,
            material.material_id,
            low_pitch_boundary=LOW_PITCH_BOUNDARY,
            minimum_low_spacing_semitones=MINIMUM_LOW_SPACING_SEMITONES,
        )
        for material in score.materials
        if material.harmonies
    )
    failures.extend(f"harmony:{item.material_id}" for item in harmony if not item.passes)
    dissonance = tuple(
        analyze_foreground_dissonance(score, material.material_id)
        for material in score.materials
        if material.harmonies
    )
    failures.extend(
        f"unsupported-dissonance:{item.material_id}"
        for item in dissonance
        if item.status != "assessed" or item.unsupported_tones
    )

    ending = _score_ending_metrics(plan, score)
    required = set(ending["required_pitch_classes"])
    if not ending["dedicated_single_harmony_material"]:
        failures.append("final-tonic-dedicated-material")
    if not required.issubset(ending["pitch_classes"]):
        failures.append("final-tonic-pitches")
    if not ending["notes_reach_material_end"]:
        failures.append("final-tonic-material-duration")
    if ending["prior_key_carryover_count"]:
        failures.append("ending-key-carryover")
    if (
        ending["minimum_duration_units"] * TARGET_DURATION_MS
        < ending["total_occurrence_units"] * MINIMUM_ENDING_HOLD_MS
    ):
        failures.append("final-tonic-nominal-duration")

    return {
        "schema_version": 1,
        "passes": not failures,
        "failures": failures,
        "harmony": [asdict(item) for item in harmony],
        "dissonance": [asdict(item) for item in dissonance],
        "unassessed_material_ids": [
            material.material_id for material in score.materials if not material.harmonies
        ],
        "ending_score": ending,
    }


def _ending_metrics(plan: PiecePlan, rendered: RenderedPerformance) -> dict[str, Any]:
    if not rendered.notes:
        return {
            "attack_ms": None,
            "note_count": 0,
            "pitch_classes": [],
            "minimum_note_duration_ms": 0,
            "pedal_hold_after_attack_ms": 0,
            "pedal_down_at_attack": False,
            "prior_key_carryover_count": 0,
        }
    attack = max(note.at_ms for note in rendered.notes)
    chord = tuple(note for note in rendered.notes if note.at_ms == attack)
    pedal_state = 0
    for pedal in sorted(
        (pedal for pedal in rendered.pedals if pedal.at_ms <= attack),
        key=lambda item: (item.at_ms, item.value, item.event_id),
    ):
        pedal_state = pedal.value
    releases = tuple(
        pedal.at_ms for pedal in rendered.pedals if pedal.value < 64 and pedal.at_ms > attack
    )
    release = min(releases, default=attack)
    return {
        "attack_ms": attack,
        "note_count": len(chord),
        "pitch_classes": sorted({note.pitch % 12 for note in chord}),
        "minimum_note_duration_ms": min(note.duration_ms for note in chord),
        "pedal_hold_after_attack_ms": release - attack,
        "pedal_down_at_attack": pedal_state >= 64,
        "prior_key_carryover_count": sum(
            note.at_ms < attack < note.at_ms + note.duration_ms for note in rendered.notes
        ),
        "required_pitch_classes": sorted({plan.tonal_center, (plan.tonal_center + 7) % 12}),
    }


def evaluate_generic_pipeline_quality(
    plan: PiecePlan,
    score: ScoreSpec,
    performance: PerformanceSpec,
    rendered: RenderedPerformance,
    *,
    declared_cues: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """参照類似とは分離した、形式非依存の品質結果を返す。"""

    validate_pipeline(plan, score, performance)
    failures = list(
        evaluate_generic_piece_plan_quality(plan, declared_cues=declared_cues)["failures"]
    )
    score_quality = evaluate_generic_score_quality(plan, score)
    failures.extend(score_quality["failures"])

    recurrences = analyze_recurrences(
        plan,
        score,
        rendered,
        declared_cues=declared_cues,
    )
    for item in recurrences:
        if item.exact_surface_copy:
            failures.append(f"exact-recurrence:{item.target_node_id}")
        elif item.relation_status != "related":
            failures.append(f"unassessed-recurrence:{item.target_node_id}")

    rendered_harmony = analyze_rendered_harmony(plan, score, rendered)
    if not rendered_harmony.passes:
        failures.append("rendered-harmony")

    if rendered.duration_ms != TARGET_DURATION_MS:
        failures.append("duration")
    if not rendered.notes:
        failures.append("no-notes")
    if any(not 1 <= note.velocity <= 127 for note in rendered.notes):
        failures.append("velocity")
    if not rendered.pedals or rendered.pedals[-1].value >= 64:
        failures.append("final-pedal-release")

    ending = _ending_metrics(plan, rendered)
    leaves, _ = ordered_leaf_schedule(plan, score)
    node_by_id = {node.node_id: node for node in plan.nodes}
    performance_by_node = {item.node_id: item for item in performance.node_performances}
    _, ending_coordination = resolve_effective_profile(
        leaves[-1][0],
        node_by_id,
        performance_by_node,
        "coordination_profile",
        "score",
    )
    ending["effective_coordination_profile"] = ending_coordination
    required = set(ending.get("required_pitch_classes", []))
    if not required.issubset(ending["pitch_classes"]):
        failures.append("final-tonic-pitches")
    if ending["minimum_note_duration_ms"] < MINIMUM_ENDING_HOLD_MS:
        failures.append("final-tonic-note-duration")
    if ending["pedal_hold_after_attack_ms"] < MINIMUM_ENDING_HOLD_MS:
        failures.append("final-tonic-pedal-duration")
    if not ending["pedal_down_at_attack"]:
        failures.append("final-tonic-pedal-down")
    if ending_coordination not in {"score", "aligned"}:
        failures.append("final-tonic-coordination")
    if ending["prior_key_carryover_count"]:
        failures.append("ending-key-carryover")

    failures = list(dict.fromkeys(failures))

    return {
        "schema_version": 1,
        "passes": not failures,
        "failures": failures,
        "recurrences": [asdict(item) for item in recurrences],
        "score_quality": score_quality,
        "harmony": score_quality["harmony"],
        "dissonance": score_quality["dissonance"],
        "rendered_harmony": asdict(rendered_harmony),
        "ending": ending,
    }
