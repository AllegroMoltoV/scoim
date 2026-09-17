"""曲全体の段階生成で使うIDなし集合DSLを安全に解析する。"""

from __future__ import annotations

import ast
from collections.abc import Callable, Sequence

from llm_musical_composer.staged_material_pilot import (
    MelodyDraft,
    TextureDraft,
    dump_harmonic_draft,
    dump_melody_draft,
    dump_texture_draft,
    parse_harmonic_draft,
    parse_melody_draft,
    parse_texture_draft,
)
from llm_musical_composer.whole_score_staged_generation import (
    PerformanceOccurrenceDraftV0,
    WholeHarmonicMaterialDraftV0,
)


class WholeScoreCollectionDslError(ValueError):
    """集合DSLの構文または制限に違反した入力。"""


def _fail(message: str) -> None:
    raise WholeScoreCollectionDslError(message)


def _collection_elements(source: str, function: str, keyword: str) -> tuple[ast.AST, ...]:
    try:
        expression = ast.parse(source.strip(), mode="eval")
    except SyntaxError as error:
        raise WholeScoreCollectionDslError("collection source is not one expression") from error
    call = expression.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        _fail("collection must be a direct function call")
    if call.func.id != function or call.args:
        _fail("collection function or positional arguments are invalid")
    if (
        len(call.keywords) != 1
        or call.keywords[0].arg != keyword
        or not isinstance(call.keywords[0].value, ast.List)
    ):
        _fail("collection keyword arguments are invalid")
    elements = tuple(call.keywords[0].value.elts)
    if not elements:
        _fail("collection must not be empty")
    return elements


def _parse_collection[T](
    source: str,
    function: str,
    keyword: str,
    parser: Callable[[str], T],
) -> tuple[T, ...]:
    result: list[T] = []
    for element in _collection_elements(source, function, keyword):
        try:
            result.append(parser(ast.unparse(element)))
        except ValueError as error:
            raise WholeScoreCollectionDslError(str(error)) from error
    return tuple(result)


def _dump_collection[T](
    function: str,
    keyword: str,
    items: Sequence[T],
    dump: Callable[[T], str],
) -> str:
    if not items:
        _fail("collection must not be empty")
    return f"{function}({keyword}=[{', '.join(dump(item) for item in items)}])"


def _parse_harmonic_material(source: str) -> WholeHarmonicMaterialDraftV0:
    try:
        expression = ast.parse(source, mode="eval")
    except SyntaxError as error:
        raise WholeScoreCollectionDslError("harmonic material is invalid") from error
    call = expression.body
    if (
        not isinstance(call, ast.Call)
        or not isinstance(call.func, ast.Name)
        or call.func.id != "harmonic_material"
        or call.args
        or any(item.arg is None for item in call.keywords)
    ):
        _fail("harmonic material must be a direct keyword-only call")
    values = {item.arg: item.value for item in call.keywords}
    if len(values) != len(call.keywords) or set(values) != {"length_units", "events"}:
        _fail("harmonic material fields are invalid")
    length = values["length_units"]
    if (
        not isinstance(length, ast.Constant)
        or isinstance(length.value, bool)
        or not isinstance(length.value, int)
        or length.value <= 0
    ):
        _fail("harmonic material length_units must be a positive integer")
    events = values["events"]
    if not isinstance(events, ast.List):
        _fail("harmonic material events must be a list")
    draft = parse_harmonic_draft(f"harmonic_draft(events={ast.unparse(events)})")
    return WholeHarmonicMaterialDraftV0(length.value, draft)


def _dump_harmonic_material(item: WholeHarmonicMaterialDraftV0) -> str:
    if item.length_units <= 0:
        _fail("harmonic material length_units must be a positive integer")
    draft = dump_harmonic_draft(item.draft)
    events = draft.removeprefix("harmonic_draft(events=").removesuffix(")")
    return f"harmonic_material(length_units={item.length_units}, events={events})"


def parse_harmonic_collection(
    source: str,
) -> tuple[WholeHarmonicMaterialDraftV0, ...]:
    return _parse_collection(
        source, "harmonic_collection", "materials", _parse_harmonic_material
    )


def dump_harmonic_collection(items: Sequence[WholeHarmonicMaterialDraftV0]) -> str:
    return _dump_collection(
        "harmonic_collection", "materials", items, _dump_harmonic_material
    )


def parse_melody_collection(source: str) -> tuple[MelodyDraft, ...]:
    return _parse_collection(source, "melody_collection", "materials", parse_melody_draft)


def dump_melody_collection(items: Sequence[MelodyDraft]) -> str:
    return _dump_collection("melody_collection", "materials", items, dump_melody_draft)


def parse_texture_collection(source: str) -> tuple[TextureDraft, ...]:
    return _parse_collection(
        source, "texture_collection", "materials", parse_texture_draft
    )


def dump_texture_collection(items: Sequence[TextureDraft]) -> str:
    return _dump_collection("texture_collection", "materials", items, dump_texture_draft)


_PERFORMANCE_FIELDS = (
    "timing_profile",
    "timing_amount",
    "dynamics_profile",
    "articulation_profile",
    "coordination_profile",
    "pedal_profile",
)


def _parse_performance_occurrence(source: str) -> PerformanceOccurrenceDraftV0:
    try:
        expression = ast.parse(source, mode="eval")
    except SyntaxError as error:
        raise WholeScoreCollectionDslError("performance occurrence is invalid") from error
    call = expression.body
    if (
        not isinstance(call, ast.Call)
        or not isinstance(call.func, ast.Name)
        or call.func.id != "performance_occurrence"
        or call.args
        or any(item.arg is None for item in call.keywords)
    ):
        _fail("performance occurrence must be a direct keyword-only call")
    values = {item.arg: item.value for item in call.keywords}
    if len(values) != len(call.keywords) or set(values) != set(_PERFORMANCE_FIELDS):
        _fail("performance occurrence fields are invalid")
    prepared: list[str | None] = []
    for field in _PERFORMANCE_FIELDS:
        node = values[field]
        if not isinstance(node, ast.Constant) or (
            node.value is not None and not isinstance(node.value, str)
        ):
            _fail("performance occurrence values must be literal strings or None")
        prepared.append(node.value)
    return PerformanceOccurrenceDraftV0(*prepared)


def parse_performance_collection(source: str) -> tuple[PerformanceOccurrenceDraftV0, ...]:
    return _parse_collection(
        source,
        "performance_collection",
        "occurrences",
        _parse_performance_occurrence,
    )


def _dump_performance_occurrence(item: PerformanceOccurrenceDraftV0) -> str:
    values = vars(item)
    return "performance_occurrence(" + ", ".join(
        f"{field}={values[field]!r}" for field in _PERFORMANCE_FIELDS
    ) + ")"


def dump_performance_collection(items: Sequence[PerformanceOccurrenceDraftV0]) -> str:
    return _dump_collection(
        "performance_collection",
        "occurrences",
        items,
        _dump_performance_occurrence,
    )
