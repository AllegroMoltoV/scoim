"""三段階の中間表現を、Pythonを実行せず制限付き記法として解析する。"""

from __future__ import annotations

import ast
from collections.abc import Callable
from typing import Any

from llm_musical_composer.performance_pipeline import (
    NodePerformance,
    PerformanceSpec,
    PiecePlan,
    PlanNode,
    ScoreDirection,
    ScoreHarmony,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
)


class PipelineDslError(ValueError):
    """段階別DSLが構文または許可した語彙に違反したことを表す。"""


def _root(source: str, expected_name: str) -> ast.Call:
    try:
        parsed = ast.parse(source, mode="eval")
    except SyntaxError as error:
        raise PipelineDslError(f"invalid syntax at line {error.lineno}: {error.msg}") from error
    return _call(parsed.body, expected_name)


def _call(node: ast.AST, expected_name: str) -> ast.Call:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise PipelineDslError("only direct calls to allowed DSL functions are accepted")
    if node.func.id != expected_name:
        raise PipelineDslError(f"unknown DSL function: {node.func.id}")
    if node.args:
        raise PipelineDslError("positional arguments are not accepted")
    if any(keyword.arg is None for keyword in node.keywords):
        raise PipelineDslError("keyword expansion is not accepted")
    return node


def _arguments(
    call: ast.Call,
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> dict[str, ast.AST]:
    result: dict[str, ast.AST] = {}
    for keyword in call.keywords:
        assert keyword.arg is not None
        if keyword.arg in result:
            raise PipelineDslError(f"duplicate argument: {keyword.arg}")
        result[keyword.arg] = keyword.value
    unknown = set(result) - required - optional
    if unknown:
        raise PipelineDslError(f"unknown argument: {sorted(unknown)[0]}")
    missing = required - result.keys()
    if missing:
        raise PipelineDslError(f"missing argument: {sorted(missing)[0]}")
    return result


def _value(node: ast.AST, expected: type[int] | type[str]) -> int | str:
    if not isinstance(node, ast.Constant) or type(node.value) is not expected:
        raise PipelineDslError(f"expected a literal {expected.__name__}")
    return node.value


def _optional_string(node: ast.AST | None) -> str | None:
    if node is None or (isinstance(node, ast.Constant) and node.value is None):
        return None
    return str(_value(node, str))


def _optional_int(node: ast.AST | None) -> int | None:
    if node is None or (isinstance(node, ast.Constant) and node.value is None):
        return None
    return int(_value(node, int))


def _list(node: ast.AST, builder: Callable[[ast.AST], Any]) -> tuple[Any, ...]:
    if not isinstance(node, ast.List):
        raise PipelineDslError("expected a list literal")
    return tuple(builder(item) for item in node.elts)


def _string_list(node: ast.AST) -> tuple[str, ...]:
    return _list(node, lambda item: str(_value(item, str)))


def _plan_node(node: ast.AST) -> PlanNode:
    args = _arguments(
        _call(node, "plan_node"),
        required=frozenset({"node_id", "parent_id", "order", "role"}),
        optional=frozenset(
            {
                "derived_from",
                "contrasts_with",
                "harmonic_focus",
                "duration_weight",
                "score_material_id",
            }
        ),
    )
    return PlanNode(
        node_id=str(_value(args["node_id"], str)),
        parent_id=_optional_string(args["parent_id"]),
        order=int(_value(args["order"], int)),
        role=str(_value(args["role"], str)),
        derived_from=_optional_string(args.get("derived_from")),
        contrasts_with=_optional_string(args.get("contrasts_with")),
        harmonic_focus=_optional_int(args.get("harmonic_focus")),
        duration_weight=_optional_int(args.get("duration_weight")),
        score_material_id=_optional_string(args.get("score_material_id")),
    )


def parse_piece_plan(source: str) -> PiecePlan:
    """PiecePlanだけを受理するDSLを解析する。"""
    args = _arguments(
        _root(source, "piece_plan"),
        required=frozenset(
            {
                "plan_id",
                "title",
                "tonal_center",
                "mode",
                "root_node_id",
                "ending_intent",
                "nodes",
            }
        ),
    )
    return PiecePlan(
        plan_id=str(_value(args["plan_id"], str)),
        title=str(_value(args["title"], str)),
        tonal_center=int(_value(args["tonal_center"], int)),
        mode=str(_value(args["mode"], str)),
        root_node_id=str(_value(args["root_node_id"], str)),
        ending_intent=str(_value(args["ending_intent"], str)),
        nodes=_list(args["nodes"], _plan_node),
    )


def _score_note(node: ast.AST) -> ScoreNote:
    args = _arguments(
        _call(node, "score_note"),
        required=frozenset({"event_id", "at_units", "duration_units", "pitch", "voice"}),
        optional=frozenset({"tie", "articulations"}),
    )
    articulations = args.get("articulations")
    return ScoreNote(
        event_id=str(_value(args["event_id"], str)),
        at_units=int(_value(args["at_units"], int)),
        duration_units=int(_value(args["duration_units"], int)),
        pitch=int(_value(args["pitch"], int)),
        voice=str(_value(args["voice"], str)),
        tie=_optional_string(args.get("tie")),
        articulations=() if articulations is None else _string_list(articulations),
    )


def _score_direction(node: ast.AST) -> ScoreDirection:
    args = _arguments(
        _call(node, "score_direction"),
        required=frozenset({"direction_id", "at_units", "kind", "value"}),
    )
    return ScoreDirection(
        direction_id=str(_value(args["direction_id"], str)),
        at_units=int(_value(args["at_units"], int)),
        kind=str(_value(args["kind"], str)),
        value=str(_value(args["value"], str)),
    )


def _score_harmony(node: ast.AST) -> ScoreHarmony:
    args = _arguments(
        _call(node, "score_harmony"),
        required=frozenset(
            {"harmony_id", "at_units", "duration_units", "root_pitch_class", "quality"}
        ),
    )
    return ScoreHarmony(
        harmony_id=str(_value(args["harmony_id"], str)),
        at_units=int(_value(args["at_units"], int)),
        duration_units=int(_value(args["duration_units"], int)),
        root_pitch_class=int(_value(args["root_pitch_class"], int)),
        quality=str(_value(args["quality"], str)),
    )


def _score_material(node: ast.AST) -> ScoreMaterial:
    args = _arguments(
        _call(node, "score_material"),
        required=frozenset({"material_id", "length_units", "notes"}),
        optional=frozenset({"derived_from", "directions", "harmonies", "foreground_voice"}),
    )
    directions = args.get("directions")
    harmonies = args.get("harmonies")
    return ScoreMaterial(
        material_id=str(_value(args["material_id"], str)),
        length_units=int(_value(args["length_units"], int)),
        notes=_list(args["notes"], _score_note),
        derived_from=_optional_string(args.get("derived_from")),
        directions=() if directions is None else _list(directions, _score_direction),
        harmonies=() if harmonies is None else _list(harmonies, _score_harmony),
        foreground_voice=_optional_string(args.get("foreground_voice")),
    )


def parse_score_spec(source: str) -> ScoreSpec:
    """ScoreSpecだけを受理するDSLを解析する。"""
    args = _arguments(
        _root(source, "score_spec"),
        required=frozenset({"score_id", "divisions", "materials"}),
    )
    return ScoreSpec(
        score_id=str(_value(args["score_id"], str)),
        divisions=int(_value(args["divisions"], int)),
        materials=_list(args["materials"], _score_material),
    )


def _node_performance(node: ast.AST) -> NodePerformance:
    args = _arguments(
        _call(node, "node_performance"),
        required=frozenset({"node_id"}),
        optional=frozenset(
            {
                "timing_profile",
                "timing_amount",
                "dynamics_profile",
                "articulation_profile",
                "coordination_profile",
                "pedal_profile",
            }
        ),
    )
    return NodePerformance(
        node_id=str(_value(args["node_id"], str)),
        timing_profile=_optional_string(args.get("timing_profile")),
        timing_amount=_optional_string(args.get("timing_amount")),
        dynamics_profile=_optional_string(args.get("dynamics_profile")),
        articulation_profile=_optional_string(args.get("articulation_profile")),
        coordination_profile=_optional_string(args.get("coordination_profile")),
        pedal_profile=_optional_string(args.get("pedal_profile")),
    )


def parse_performance_spec(source: str) -> PerformanceSpec:
    """PerformanceSpecだけを受理するDSLを解析する。"""
    args = _arguments(
        _root(source, "performance_spec"),
        required=frozenset(
            {
                "performance_id",
                "target_duration_ms",
                "default_velocity",
                "timing_budget_id",
                "node_performances",
            }
        ),
        optional=frozenset({"key_release_percent", "velocity_policy_id"}),
    )
    return PerformanceSpec(
        performance_id=str(_value(args["performance_id"], str)),
        target_duration_ms=int(_value(args["target_duration_ms"], int)),
        default_velocity=int(_value(args["default_velocity"], int)),
        timing_budget_id=str(_value(args["timing_budget_id"], str)),
        node_performances=_list(args["node_performances"], _node_performance),
        key_release_percent=int(_value(args["key_release_percent"], int))
        if "key_release_percent" in args
        else 100,
        velocity_policy_id=str(_value(args["velocity_policy_id"], str))
        if "velocity_policy_id" in args
        else "legacy-unison-v1",
    )


def _call_text(name: str, fields: tuple[tuple[str, str], ...]) -> str:
    return f"{name}(" + ", ".join(f"{key}={value}" for key, value in fields) + ")"


def _optional_field(name: str, value: object) -> tuple[tuple[str, str], ...]:
    return () if value is None else ((name, repr(value)),)


def dump_piece_plan(plan: PiecePlan) -> str:
    """PiecePlanを決定的な一行DSLへ直列化する。"""
    nodes = []
    for node in plan.nodes:
        fields = (
            ("node_id", repr(node.node_id)),
            ("parent_id", repr(node.parent_id)),
            ("order", repr(node.order)),
            ("role", repr(node.role)),
            *_optional_field("derived_from", node.derived_from),
            *_optional_field("contrasts_with", node.contrasts_with),
            *_optional_field("harmonic_focus", node.harmonic_focus),
            *_optional_field("duration_weight", node.duration_weight),
            *_optional_field("score_material_id", node.score_material_id),
        )
        nodes.append(_call_text("plan_node", fields))
    return _call_text(
        "piece_plan",
        (
            ("plan_id", repr(plan.plan_id)),
            ("title", repr(plan.title)),
            ("tonal_center", repr(plan.tonal_center)),
            ("mode", repr(plan.mode)),
            ("root_node_id", repr(plan.root_node_id)),
            ("ending_intent", repr(plan.ending_intent)),
            ("nodes", "[" + ", ".join(nodes) + "]"),
        ),
    )


def dump_score_spec(score: ScoreSpec) -> str:
    """ScoreSpecを決定的な一行DSLへ直列化する。"""
    materials = []
    for material in score.materials:
        notes = [
            _call_text(
                "score_note",
                (
                    ("event_id", repr(note.event_id)),
                    ("at_units", repr(note.at_units)),
                    ("duration_units", repr(note.duration_units)),
                    ("pitch", repr(note.pitch)),
                    ("voice", repr(note.voice)),
                    *_optional_field("tie", note.tie),
                    *(
                        (("articulations", repr(list(note.articulations))),)
                        if note.articulations
                        else ()
                    ),
                ),
            )
            for note in material.notes
        ]
        directions = [
            _call_text(
                "score_direction",
                (
                    ("direction_id", repr(direction.direction_id)),
                    ("at_units", repr(direction.at_units)),
                    ("kind", repr(direction.kind)),
                    ("value", repr(direction.value)),
                ),
            )
            for direction in material.directions
        ]
        harmonies = [
            _call_text(
                "score_harmony",
                (
                    ("harmony_id", repr(harmony.harmony_id)),
                    ("at_units", repr(harmony.at_units)),
                    ("duration_units", repr(harmony.duration_units)),
                    ("root_pitch_class", repr(harmony.root_pitch_class)),
                    ("quality", repr(harmony.quality)),
                ),
            )
            for harmony in material.harmonies
        ]
        fields = (
            ("material_id", repr(material.material_id)),
            ("length_units", repr(material.length_units)),
            ("notes", "[" + ", ".join(notes) + "]"),
            *_optional_field("derived_from", material.derived_from),
            *((("harmonies", "[" + ", ".join(harmonies) + "]"),) if harmonies else ()),
            *_optional_field("foreground_voice", material.foreground_voice),
            *((("directions", "[" + ", ".join(directions) + "]"),) if directions else ()),
        )
        materials.append(_call_text("score_material", fields))
    return _call_text(
        "score_spec",
        (
            ("score_id", repr(score.score_id)),
            ("divisions", repr(score.divisions)),
            ("materials", "[" + ", ".join(materials) + "]"),
        ),
    )


def dump_performance_spec(performance: PerformanceSpec) -> str:
    """PerformanceSpecを決定的な一行DSLへ直列化する。"""
    items = []
    for item in performance.node_performances:
        fields = (
            ("node_id", repr(item.node_id)),
            *_optional_field("timing_profile", item.timing_profile),
            *_optional_field("timing_amount", item.timing_amount),
            *_optional_field("dynamics_profile", item.dynamics_profile),
            *_optional_field("articulation_profile", item.articulation_profile),
            *_optional_field("coordination_profile", item.coordination_profile),
            *_optional_field("pedal_profile", item.pedal_profile),
        )
        items.append(_call_text("node_performance", fields))
    return _call_text(
        "performance_spec",
        (
            ("performance_id", repr(performance.performance_id)),
            ("target_duration_ms", repr(performance.target_duration_ms)),
            ("default_velocity", repr(performance.default_velocity)),
            ("timing_budget_id", repr(performance.timing_budget_id)),
            *(
                (("key_release_percent", repr(performance.key_release_percent)),)
                if performance.key_release_percent != 100
                else ()
            ),
            *(
                (("velocity_policy_id", repr(performance.velocity_policy_id)),)
                if performance.velocity_policy_id != "legacy-unison-v1"
                else ()
            ),
            ("node_performances", "[" + ", ".join(items) + "]"),
        ),
    )
