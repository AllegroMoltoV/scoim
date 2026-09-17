"""ScoreSpecから和声の時間骨格を分離する実験IR。"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from llm_musical_composer.performance_pipeline import (
    HARMONY_INTERVALS,
    PiecePlan,
    ScoreDirection,
    ScoreHarmony,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
    validate_piece_plan,
    validate_score_spec,
)


class HarmonicSkeletonError(ValueError):
    """HarmonicSkeletonV0の構文または意味契約違反。"""


@dataclass(frozen=True)
class HarmonicMaterialV0:
    material_id: str
    length_units: int
    derived_from: str | None
    harmonies: tuple[ScoreHarmony, ...]


@dataclass(frozen=True)
class HarmonicSkeletonV0:
    score_id: str
    divisions: int
    materials: tuple[HarmonicMaterialV0, ...]


@dataclass(frozen=True)
class MaterialPayloadV0:
    material_id: str
    notes: tuple[ScoreNote, ...]
    directions: tuple[ScoreDirection, ...]
    foreground_voice: str | None


@dataclass(frozen=True)
class ScorePayloadV0:
    materials: tuple[MaterialPayloadV0, ...]


def _fail(message: str) -> None:
    raise HarmonicSkeletonError(message)


def _call(node: ast.AST, expected_name: str) -> ast.Call:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        _fail("only direct calls to allowed harmonic skeleton functions are accepted")
    if node.func.id != expected_name:
        _fail(f"unknown harmonic skeleton function: {node.func.id}")
    if node.args:
        _fail("positional arguments are not accepted")
    if any(keyword.arg is None for keyword in node.keywords):
        _fail("keyword expansion is not accepted")
    return node


def _arguments(call: ast.Call, required: frozenset[str]) -> dict[str, ast.AST]:
    result: dict[str, ast.AST] = {}
    for keyword in call.keywords:
        assert keyword.arg is not None
        if keyword.arg in result:
            _fail(f"duplicate argument: {keyword.arg}")
        result[keyword.arg] = keyword.value
    unknown = set(result) - required
    if unknown:
        _fail(f"unknown argument: {sorted(unknown)[0]}")
    missing = required - result.keys()
    if missing:
        _fail(f"missing argument: {sorted(missing)[0]}")
    return result


def _literal(node: ast.AST, expected: type[int] | type[str]) -> int | str:
    if not isinstance(node, ast.Constant) or type(node.value) is not expected:
        _fail(f"expected a literal {expected.__name__}")
    return node.value


def _optional_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and node.value is None:
        return None
    return str(_literal(node, str))


def _list(node: ast.AST, builder: Callable[[ast.AST], Any]) -> tuple[Any, ...]:
    if not isinstance(node, ast.List):
        _fail("expected a list literal")
    return tuple(builder(item) for item in node.elts)


def _harmony(node: ast.AST) -> ScoreHarmony:
    args = _arguments(
        _call(node, "harmonic_interval_v0"),
        frozenset(
            {"harmony_id", "at_units", "duration_units", "root_pitch_class", "quality"}
        ),
    )
    return ScoreHarmony(
        str(_literal(args["harmony_id"], str)),
        int(_literal(args["at_units"], int)),
        int(_literal(args["duration_units"], int)),
        int(_literal(args["root_pitch_class"], int)),
        str(_literal(args["quality"], str)),
    )


def _material(node: ast.AST) -> HarmonicMaterialV0:
    args = _arguments(
        _call(node, "harmonic_material_v0"),
        frozenset({"material_id", "length_units", "derived_from", "harmonies"}),
    )
    return HarmonicMaterialV0(
        str(_literal(args["material_id"], str)),
        int(_literal(args["length_units"], int)),
        _optional_string(args["derived_from"]),
        _list(args["harmonies"], _harmony),
    )


def parse_harmonic_skeleton(source: str) -> HarmonicSkeletonV0:
    """HarmonicSkeletonV0をPythonとして実行せず解析する。"""

    try:
        parsed = ast.parse(source, mode="eval")
    except SyntaxError as error:
        raise HarmonicSkeletonError(
            f"invalid harmonic skeleton syntax at line {error.lineno}: {error.msg}"
        ) from error
    args = _arguments(
        _call(parsed.body, "harmonic_skeleton_v0"),
        frozenset({"score_id", "divisions", "materials"}),
    )
    return HarmonicSkeletonV0(
        str(_literal(args["score_id"], str)),
        int(_literal(args["divisions"], int)),
        _list(args["materials"], _material),
    )


def _call_text(name: str, fields: tuple[tuple[str, str], ...]) -> str:
    return f"{name}(" + ", ".join(f"{key}={value}" for key, value in fields) + ")"


def dump_harmonic_skeleton(skeleton: HarmonicSkeletonV0) -> str:
    """HarmonicSkeletonV0を決定的な一行DSLへ直列化する。"""

    materials: list[str] = []
    for material in skeleton.materials:
        harmonies = [
            _call_text(
                "harmonic_interval_v0",
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
        materials.append(
            _call_text(
                "harmonic_material_v0",
                (
                    ("material_id", repr(material.material_id)),
                    ("length_units", repr(material.length_units)),
                    ("derived_from", repr(material.derived_from)),
                    ("harmonies", "[" + ", ".join(harmonies) + "]"),
                ),
            )
        )
    return _call_text(
        "harmonic_skeleton_v0",
        (
            ("score_id", repr(skeleton.score_id)),
            ("divisions", repr(skeleton.divisions)),
            ("materials", "[" + ", ".join(materials) + "]"),
        ),
    )


def validate_harmonic_skeleton(plan: PiecePlan, skeleton: HarmonicSkeletonV0) -> None:
    """PiecePlanとの参照と和声時間契約を検査する。"""

    validate_piece_plan(plan)
    if not skeleton.score_id or skeleton.divisions <= 0 or not skeleton.materials:
        _fail("harmonic skeleton metadata is invalid")
    expected_ids = {
        node.score_material_id for node in plan.nodes if node.score_material_id is not None
    }
    material_ids: set[str] = set()
    harmony_ids: set[str] = set()
    for material in skeleton.materials:
        if not material.material_id or material.material_id in material_ids:
            _fail("harmonic material IDs must be non-empty and unique")
        if material.derived_from is not None and material.derived_from not in material_ids:
            _fail("derived_from must reference an earlier harmonic material")
        material_ids.add(material.material_id)
        if material.length_units <= 0:
            _fail("harmonic material length must be positive")
        cursor = 0
        for harmony in material.harmonies:
            if not harmony.harmony_id or harmony.harmony_id in harmony_ids:
                _fail("harmony IDs must be non-empty and unique")
            harmony_ids.add(harmony.harmony_id)
            if (
                harmony.at_units != cursor
                or harmony.duration_units <= 0
                or not 0 <= harmony.root_pitch_class <= 11
                or harmony.quality not in HARMONY_INTERVALS
            ):
                _fail("harmony coverage or vocabulary is invalid")
            cursor += harmony.duration_units
        if material.harmonies and cursor != material.length_units:
            _fail("harmony coverage must fill the material")
    if material_ids != expected_ids:
        _fail("harmonic skeleton material IDs must match PiecePlan")


def split_score_spec(
    plan: PiecePlan,
    score: ScoreSpec,
) -> tuple[HarmonicSkeletonV0, ScorePayloadV0]:
    """ScoreSpecを和声骨格と非和声payloadへ損失なく分ける。"""

    validate_score_spec(plan, score)
    skeleton = HarmonicSkeletonV0(
        score.score_id,
        score.divisions,
        tuple(
            HarmonicMaterialV0(
                material.material_id,
                material.length_units,
                material.derived_from,
                material.harmonies,
            )
            for material in score.materials
        ),
    )
    payload = ScorePayloadV0(
        tuple(
            MaterialPayloadV0(
                material.material_id,
                material.notes,
                material.directions,
                material.foreground_voice,
            )
            for material in score.materials
        )
    )
    validate_harmonic_skeleton(plan, skeleton)
    return skeleton, payload


def apply_harmonic_skeleton(
    plan: PiecePlan,
    payload: ScorePayloadV0,
    skeleton: HarmonicSkeletonV0,
) -> ScoreSpec:
    """和声骨格と参照payloadからScoreSpecを再合成する。"""

    validate_harmonic_skeleton(plan, skeleton)
    payload_by_id = {material.material_id: material for material in payload.materials}
    if len(payload_by_id) != len(payload.materials) or set(payload_by_id) != {
        material.material_id for material in skeleton.materials
    }:
        _fail("payload material IDs must match harmonic skeleton material IDs")
    score = ScoreSpec(
        skeleton.score_id,
        skeleton.divisions,
        tuple(
            ScoreMaterial(
                material.material_id,
                material.length_units,
                payload_by_id[material.material_id].notes,
                derived_from=material.derived_from,
                directions=payload_by_id[material.material_id].directions,
                harmonies=material.harmonies,
                foreground_voice=payload_by_id[material.material_id].foreground_voice,
            )
            for material in skeleton.materials
        ),
    )
    validate_score_spec(plan, score)
    return score
