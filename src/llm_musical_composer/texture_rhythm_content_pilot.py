"""伴奏の発音骨格と内容を分ける限定パイロット。"""

from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from llm_musical_composer.performance_pipeline import ScoreHarmony, ScoreNote
from llm_musical_composer.piano_texture_register_placement import REGISTER_ZONES
from llm_musical_composer.staged_material_pilot import (
    TextureDraft,
    TextureEventDraft,
    harmony_start_accompaniment_policy,
)

_ARTICULATIONS = frozenset({"normal", "staccato", "tenuto", "accent"})
_DEGREES = frozenset({"root", "third", "fifth", "seventh"})


class TextureRhythmContentPilotError(ValueError):
    """二段階伴奏パイロットの入力または実行契約違反。"""


@dataclass(frozen=True)
class TextureRhythmGroupV0:
    at_units: int
    new_accompaniment_attack_count: int


@dataclass(frozen=True)
class TextureRhythmDraftV0:
    groups: tuple[TextureRhythmGroupV0, ...]


@dataclass(frozen=True)
class TextureContentEventV0:
    duration_units: int
    degree: str
    register_zone: str
    articulations: tuple[str, ...] = ()


@dataclass(frozen=True)
class TextureContentDraftV0:
    groups: tuple[tuple[TextureContentEventV0, ...], ...]


class TextureRhythmContentRunner(Protocol):
    @property
    def call_number(self) -> int: ...

    def run(
        self, step_id: str, prompt: str, input_hashes: dict[str, str]
    ) -> Mapping[str, object]: ...


def _fail(message: str) -> None:
    raise TextureRhythmContentPilotError(message)


def _direct_call(node: ast.AST, name: str) -> ast.Call:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        _fail("only direct calls are accepted")
    if node.func.id != name or node.args or any(item.arg is None for item in node.keywords):
        _fail(f"{name} must be a direct keyword-only call")
    return node


def _arguments(call: ast.Call, expected: frozenset[str]) -> dict[str, ast.AST]:
    result: dict[str, ast.AST] = {}
    for item in call.keywords:
        assert item.arg is not None
        if item.arg in result:
            _fail(f"duplicate argument: {item.arg}")
        result[item.arg] = item.value
    if set(result) != expected:
        _fail("arguments do not match the contract")
    return result


def _expression(source: str) -> ast.AST:
    try:
        return ast.parse(source.strip(), mode="eval").body
    except SyntaxError as error:
        raise TextureRhythmContentPilotError("source is not one expression") from error


def _integer(node: ast.AST) -> int:
    if not isinstance(node, ast.Constant) or type(node.value) is not int:
        _fail("expected a literal int")
    return node.value


def _string(node: ast.AST) -> str:
    if not isinstance(node, ast.Constant) or type(node.value) is not str:
        _fail("expected a literal str")
    return node.value


def _list(node: ast.AST) -> tuple[ast.AST, ...]:
    if not isinstance(node, ast.List):
        _fail("expected a list literal")
    return tuple(node.elts)


def _rhythm_group(node: ast.AST) -> TextureRhythmGroupV0:
    arguments = _arguments(
        _direct_call(node, "texture_rhythm_group"),
        frozenset({"at_units", "new_accompaniment_attack_count"}),
    )
    return TextureRhythmGroupV0(
        _integer(arguments["at_units"]),
        _integer(arguments["new_accompaniment_attack_count"]),
    )


def parse_texture_rhythm_draft(source: str) -> TextureRhythmDraftV0:
    call = _direct_call(_expression(source), "texture_rhythm_draft")
    arguments = _arguments(call, frozenset({"groups"}))
    groups = tuple(_rhythm_group(item) for item in _list(arguments["groups"]))
    if not groups:
        _fail("texture rhythm groups must not be empty")
    return TextureRhythmDraftV0(groups)


def _content_event(node: ast.AST) -> TextureContentEventV0:
    arguments = _arguments(
        _direct_call(node, "texture_content_event"),
        frozenset({"duration_units", "degree", "register_zone", "articulations"}),
    )
    articulations = tuple(_string(item) for item in _list(arguments["articulations"]))
    return TextureContentEventV0(
        _integer(arguments["duration_units"]),
        _string(arguments["degree"]),
        _string(arguments["register_zone"]),
        articulations,
    )


def parse_texture_content_draft(source: str) -> TextureContentDraftV0:
    call = _direct_call(_expression(source), "texture_content_draft")
    arguments = _arguments(call, frozenset({"groups"}))
    groups = tuple(
        tuple(_content_event(event) for event in _list(group))
        for group in _list(arguments["groups"])
    )
    if not groups or any(not group for group in groups):
        _fail("texture content groups and events must not be empty")
    return TextureContentDraftV0(groups)


def dump_texture_rhythm_draft(draft: TextureRhythmDraftV0) -> str:
    if not draft.groups:
        _fail("texture rhythm groups must not be empty")
    groups = ", ".join(
        "texture_rhythm_group("
        f"at_units={item.at_units!r}, "
        "new_accompaniment_attack_count="
        f"{item.new_accompaniment_attack_count!r})"
        for item in draft.groups
    )
    return f"texture_rhythm_draft(groups=[{groups}])"


def dump_texture_content_draft(draft: TextureContentDraftV0) -> str:
    if not draft.groups or any(not group for group in draft.groups):
        _fail("texture content groups and events must not be empty")
    groups = []
    for group in draft.groups:
        events = ", ".join(
            "texture_content_event("
            f"duration_units={item.duration_units!r}, "
            f"degree={item.degree!r}, "
            f"register_zone={item.register_zone!r}, "
            f"articulations={list(item.articulations)!r})"
            for item in group
        )
        groups.append(f"[{events}]")
    return f"texture_content_draft(groups=[{', '.join(groups)}])"


def _harmony_index_at(harmonies: Sequence[ScoreHarmony], at_units: int) -> int:
    matches = tuple(
        index
        for index, harmony in enumerate(harmonies)
        if harmony.at_units <= at_units < harmony.at_units + harmony.duration_units
    )
    if len(matches) != 1:
        _fail("rhythm attack does not belong to exactly one harmony interval")
    return matches[0]


def _attack_size_counts(
    rhythm: TextureRhythmDraftV0, melody: Sequence[ScoreNote]
) -> tuple[dict[str, int], int, int]:
    melody_counts = Counter(note.at_units for note in melody)
    texture_counts = {
        group.at_units: group.new_accompaniment_attack_count for group in rhythm.groups
    }
    onsets = sorted(set(melody_counts) | set(texture_counts))
    sizes = [melody_counts[at] + texture_counts.get(at, 0) for at in onsets]
    return (
        {
            "one": sum(size == 1 for size in sizes),
            "two": sum(size == 2 for size in sizes),
            "three": sum(size == 3 for size in sizes),
            "four_or_more": sum(size >= 4 for size in sizes),
        },
        len(onsets),
        max(sizes, default=0),
    )


def validate_texture_rhythm_draft(
    draft: TextureRhythmDraftV0,
    *,
    material_length_units: int,
    harmonies: Sequence[ScoreHarmony],
    melody: Sequence[ScoreNote],
    budget: Mapping[str, object],
    feasibility: Mapping[str, object],
    maximum_group_size: int,
    required_final_attack_units: int | None = None,
    minimum_required_final_new_accompaniment_attack_count: int = 0,
) -> dict[str, object]:
    """内容call前に発音位置と整数予算のhard条件を検査する。"""

    if not draft.groups:
        _fail("texture rhythm groups must not be empty")
    previous: int | None = None
    for group in draft.groups:
        if not 0 <= group.at_units < material_length_units:
            _fail("texture rhythm attack is outside material range")
        if group.new_accompaniment_attack_count <= 0:
            _fail("new accompaniment attack count must be positive")
        if previous is not None and group.at_units <= previous:
            _fail("texture rhythm groups must be strictly ordered")
        previous = group.at_units

    saved_policy = feasibility.get("harmony_start_accompaniment")
    if saved_policy is None:
        policy = harmony_start_accompaniment_policy(harmonies, melody, feasibility)
    elif isinstance(saved_policy, Mapping):
        policy = saved_policy
    else:
        _fail("harmony start accompaniment policy is invalid")
    raw_minimums = policy.get("minimum_new_accompaniment_attacks")
    if not isinstance(raw_minimums, Mapping):
        _fail("harmony start accompaniment minimums are missing")
    try:
        harmony_start_minimums = {
            int(at_units): int(minimum) for at_units, minimum in raw_minimums.items()
        }
    except (TypeError, ValueError) as error:
        raise TextureRhythmContentPilotError(
            "harmony start accompaniment minimums are invalid"
        ) from error
    harmony_starts = {item.at_units for item in harmonies}
    if set(harmony_start_minimums) != harmony_starts or any(
        minimum not in {0, 1} for minimum in harmony_start_minimums.values()
    ):
        _fail("harmony start accompaniment minimums do not cover harmonies")
    required_harmony_starts = {
        at_units for at_units, minimum in harmony_start_minimums.items() if minimum > 0
    }
    rhythm_starts = {item.at_units for item in draft.groups}
    if not required_harmony_starts.issubset(rhythm_starts):
        _fail("texture rhythm must attack at every required harmony interval start")

    capacity_rows = tuple(
        item for item in feasibility.get("onset_capacities", ()) if isinstance(item, Mapping)
    )
    capacity_checks = []
    for group in draft.groups:
        harmony_index = _harmony_index_at(harmonies, group.at_units)
        row = next(
            (
                item
                for item in capacity_rows
                if int(item.get("harmony_index", -1)) == harmony_index
                and int(item.get("start_units", -1))
                <= group.at_units
                < int(item.get("end_units", -1))
            ),
            None,
        )
        if row is None:
            _fail("texture rhythm onset capacity is missing")
        maximum = int(row["maximum_new_accompaniment_attack_count"])
        if group.new_accompaniment_attack_count > maximum:
            _fail("texture rhythm exceeds onset capacity")
        capacity_checks.append(
            {
                "at_units": group.at_units,
                "harmony_index": harmony_index,
                "actual": group.new_accompaniment_attack_count,
                "maximum": maximum,
            }
        )

    if required_final_attack_units is not None:
        if any(group.at_units > required_final_attack_units for group in draft.groups):
            _fail("texture rhythm has an attack after shared ending")
        final = next(
            (group for group in draft.groups if group.at_units == required_final_attack_units),
            None,
        )
        if final is None:
            _fail("texture rhythm is missing shared ending attack")
        if (
            final.new_accompaniment_attack_count
            < minimum_required_final_new_accompaniment_attack_count
        ):
            _fail("texture rhythm has too few final accompaniment attacks")

    event_count = sum(item.new_accompaniment_attack_count for item in draft.groups)
    size_counts, combined_group_count, actual_maximum = _attack_size_counts(draft, melody)
    expected = {
        "required_texture_event_count": int(budget["required_texture_event_count"]),
        "combined_attack_group_count": int(budget["combined_attack_group_count"]),
        "combined_note_event_count": int(budget["combined_note_event_count"]),
        "attack_size_counts": dict(budget["attack_size_counts"]),
    }
    actual = {
        "required_texture_event_count": event_count,
        "combined_attack_group_count": combined_group_count,
        "combined_note_event_count": len(melody) + event_count,
        "attack_size_counts": size_counts,
    }
    if actual != expected:
        _fail("texture rhythm does not match material budget")
    if actual_maximum > maximum_group_size:
        _fail("texture rhythm exceeds maximum attack group size")

    return {
        "status": "pass",
        "expected": expected,
        "actual": actual,
        "maximum_group_size": actual_maximum,
        "capacity_checks": capacity_checks,
        "harmony_start_accompaniment": policy,
    }


def join_texture_rhythm_content(
    rhythm: TextureRhythmDraftV0,
    content: TextureContentDraftV0,
    harmonies: Sequence[ScoreHarmony],
) -> TextureDraft:
    """確定骨格を変えず、位置対応の内容を現行TextureDraftへ戻す。"""

    if len(content.groups) != len(rhythm.groups):
        _fail("texture content group count does not match fixed rhythm")
    events: list[TextureEventDraft] = []
    for group_index, (rhythm_group, content_group) in enumerate(
        zip(rhythm.groups, content.groups, strict=True), 1
    ):
        if len(content_group) != rhythm_group.new_accompaniment_attack_count:
            _fail(f"texture content event count differs at group {group_index}")
        harmony_index = _harmony_index_at(harmonies, rhythm_group.at_units)
        for item in content_group:
            if item.duration_units <= 0:
                _fail("texture content duration must be positive")
            if item.degree not in _DEGREES:
                _fail("texture content degree is invalid")
            if item.register_zone not in REGISTER_ZONES:
                _fail("texture content register zone is invalid")
            if any(value not in _ARTICULATIONS for value in item.articulations):
                _fail("texture content articulation is invalid")
            events.append(
                TextureEventDraft(
                    harmony_index,
                    rhythm_group.at_units,
                    item.duration_units,
                    item.degree,
                    item.register_zone,
                    item.articulations,
                )
            )
    return TextureDraft(tuple(events))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def _write_json(path: Path, value: object) -> None:
    _write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _response_source(response: Mapping[str, object], stage: str) -> str:
    source = response.get("composition_source")
    if not isinstance(source, str) or not source.strip():
        _fail(f"{stage} response has no composition_source")
    return source.strip()


def _save_failure(
    run_dir: Path,
    stage: str,
    error: Exception,
    call_number: int,
) -> None:
    _write_json(
        run_dir / f"failures/{stage}.json",
        {
            "stage": stage,
            "error_type": type(error).__name__,
            "detail": str(error),
            "external_call_number": call_number,
        },
    )


def execute_texture_rhythm_content_pilot(
    run_dir: Path,
    runner: TextureRhythmContentRunner,
    *,
    rhythm_prompt: str,
    content_prompt: Callable[[TextureRhythmDraftV0], str],
    input_hashes: dict[str, str],
    rhythm_validation_arguments: dict[str, object],
    harmonies: Sequence[ScoreHarmony],
    validate_joined: Callable[[TextureDraft], Mapping[str, object]],
) -> dict[str, object]:
    """1素材を最大二段階で生成し、成功した骨格を再利用する。"""

    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = {"schema_version": 1, "input_hashes": input_hashes}
    if manifest_path.is_file():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved != manifest:
            _fail("pilot manifest differs from fixed inputs")
    else:
        _write_json(manifest_path, manifest)

    rhythm_output = run_dir / "outputs/texture-rhythm.dsl"
    if rhythm_output.is_file():
        rhythm = parse_texture_rhythm_draft(rhythm_output.read_text(encoding="utf-8"))
    else:
        _write_text(run_dir / "prompts/texture-rhythm.md", rhythm_prompt)
        try:
            response = runner.run("texture-rhythm", rhythm_prompt, input_hashes)
            source = _response_source(response, "texture-rhythm")
            _write_text(run_dir / "responses/texture-rhythm.dsl", source)
            rhythm = parse_texture_rhythm_draft(source)
            validation = validate_texture_rhythm_draft(rhythm, **rhythm_validation_arguments)
            _write_json(run_dir / "outputs/texture-rhythm-validation.json", validation)
            _write_text(rhythm_output, dump_texture_rhythm_draft(rhythm))
        except Exception as error:
            _save_failure(run_dir, "texture-rhythm", error, runner.call_number)
            raise

    validate_texture_rhythm_draft(rhythm, **rhythm_validation_arguments)
    rendered_content_prompt = content_prompt(rhythm)
    _write_text(run_dir / "prompts/texture-content.md", rendered_content_prompt)
    try:
        response = runner.run("texture-content", rendered_content_prompt, input_hashes)
        source = _response_source(response, "texture-content")
        _write_text(run_dir / "responses/texture-content.dsl", source)
        content = parse_texture_content_draft(source)
        draft = join_texture_rhythm_content(rhythm, content, harmonies)
        validation = dict(validate_joined(draft))
    except Exception as error:
        _save_failure(run_dir, "texture-content", error, runner.call_number)
        raise

    _write_text(run_dir / "outputs/texture-content.dsl", dump_texture_content_draft(content))
    _write_json(run_dir / "outputs/texture-validation.json", validation)
    summary = {
        "schema_version": 1,
        "status": "completed",
        "external_call_count": runner.call_number,
        "rhythm_sha256": _sha256_text(dump_texture_rhythm_draft(rhythm)),
        "content_sha256": _sha256_text(dump_texture_content_draft(content)),
        "texture_event_count": len(draft.events),
        "validation": validation,
    }
    _write_json(run_dir / "summary.json", summary)
    return summary
