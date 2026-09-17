"""和声と前景を固定し、伴奏生成境界だけを検証する実験用パイロット。"""

from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from llm_musical_composer.generic_pipeline_quality import evaluate_generic_score_quality
from llm_musical_composer.performance_pipeline import (
    HARMONY_INTERVALS,
    PiecePlan,
    ScoreHarmony,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
    validate_score_spec,
)
from llm_musical_composer.pipeline_dsl import (
    dump_score_spec,
    parse_piece_plan,
    parse_score_spec,
)
from llm_musical_composer.run_state import (
    RunLock,
    RunStore,
    atomic_write_bytes,
    atomic_write_json,
    sha256_file,
    sha256_json,
    sha256_text,
)

TARGET_MATERIAL_ID = "material_4"
TEXTURE_EVENT_COUNT = 8
TEXTURE_ATTACK_COUNT = 4
MODEL_ID = "gpt-5.6-sol"
REASONING_EFFORT = "high"
PROTOCOL_ID = "piano-texture-boundary-pilot-v1"
STEP_ID = "generate-piano-texture"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "prompts" / "pipeline-piano-texture-spec.md"
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "codex-composition-response.schema.json"
_ARTICULATIONS = frozenset({"normal", "staccato", "tenuto", "accent"})
_DEGREE_INDEX = {"root": 0, "third": 1, "fifth": 2, "seventh": 3}


class PianoTextureValidationError(ValueError):
    """PianoTextureSpecまたは固定範囲の契約違反を表す。"""


@dataclass(frozen=True)
class PianoTextureEvent:
    event_id: str
    harmony_id: str
    role: str
    voice: str
    at_units: int
    duration_units: int
    degree: str
    octave: int
    articulations: tuple[str, ...] = ()


@dataclass(frozen=True)
class PianoTextureSpec:
    material_id: str
    events: tuple[PianoTextureEvent, ...]


class TextureRunner(Protocol):
    @property
    def call_number(self) -> int: ...

    def run(
        self, step_id: str, prompt: str, input_hashes: dict[str, str] | None = None
    ) -> dict[str, object]: ...


def _fail(message: str) -> None:
    raise PianoTextureValidationError(message)


def _call(node: ast.AST, expected_name: str) -> ast.Call:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        _fail("only direct calls to allowed texture DSL functions are accepted")
    if node.func.id != expected_name:
        _fail(f"unknown texture DSL function: {node.func.id}")
    if node.args:
        _fail("positional arguments are not accepted")
    if any(keyword.arg is None for keyword in node.keywords):
        _fail("keyword expansion is not accepted")
    return node


def _root(source: str) -> ast.Call:
    try:
        parsed = ast.parse(source, mode="eval")
    except SyntaxError as error:
        raise PianoTextureValidationError(
            f"invalid texture syntax at line {error.lineno}: {error.msg}"
        ) from error
    return _call(parsed.body, "piano_texture_spec")


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
            _fail(f"duplicate argument: {keyword.arg}")
        result[keyword.arg] = keyword.value
    unknown = set(result) - required - optional
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


def _list(node: ast.AST, builder: Callable[[ast.AST], Any]) -> tuple[Any, ...]:
    if not isinstance(node, ast.List):
        _fail("expected a list literal")
    return tuple(builder(item) for item in node.elts)


def _event(node: ast.AST) -> PianoTextureEvent:
    args = _arguments(
        _call(node, "piano_texture_note"),
        required=frozenset(
            {
                "event_id",
                "harmony_id",
                "role",
                "voice",
                "at_units",
                "duration_units",
                "degree",
                "octave",
                "articulations",
            }
        ),
    )
    return PianoTextureEvent(
        event_id=str(_literal(args["event_id"], str)),
        harmony_id=str(_literal(args["harmony_id"], str)),
        role=str(_literal(args["role"], str)),
        voice=str(_literal(args["voice"], str)),
        at_units=int(_literal(args["at_units"], int)),
        duration_units=int(_literal(args["duration_units"], int)),
        degree=str(_literal(args["degree"], str)),
        octave=int(_literal(args["octave"], int)),
        articulations=_list(args["articulations"], lambda item: str(_literal(item, str))),
    )


def parse_piano_texture_spec(source: str) -> PianoTextureSpec:
    """PianoTextureSpec専用DSLをPythonとして実行せず解析する。"""

    args = _arguments(
        _root(source),
        required=frozenset({"material_id", "events"}),
    )
    return PianoTextureSpec(
        material_id=str(_literal(args["material_id"], str)),
        events=_list(args["events"], _event),
    )


def _call_text(name: str, fields: tuple[tuple[str, str], ...]) -> str:
    return f"{name}(" + ", ".join(f"{key}={value}" for key, value in fields) + ")"


def dump_piano_texture_spec(spec: PianoTextureSpec) -> str:
    """PianoTextureSpecを決定的な一行DSLへ直列化する。"""

    events = [
        _call_text(
            "piano_texture_note",
            (
                ("event_id", repr(event.event_id)),
                ("harmony_id", repr(event.harmony_id)),
                ("role", repr(event.role)),
                ("voice", repr(event.voice)),
                ("at_units", repr(event.at_units)),
                ("duration_units", repr(event.duration_units)),
                ("degree", repr(event.degree)),
                ("octave", repr(event.octave)),
                ("articulations", repr(list(event.articulations))),
            ),
        )
        for event in spec.events
    ]
    return _call_text(
        "piano_texture_spec",
        (
            ("material_id", repr(spec.material_id)),
            ("events", "[" + ", ".join(events) + "]"),
        ),
    )


def _material(score: ScoreSpec, material_id: str) -> ScoreMaterial:
    matches = tuple(item for item in score.materials if item.material_id == material_id)
    if len(matches) != 1:
        _fail(f"score must contain exactly one target material: {material_id}")
    return matches[0]


def build_fixed_material_view(
    plan: PiecePlan,
    score: ScoreSpec,
    material_id: str = TARGET_MATERIAL_ID,
) -> dict[str, object]:
    """旧伴奏を除き、LLMへ渡せる固定情報だけを投影する。"""

    validate_score_spec(plan, score)
    material = _material(score, material_id)
    if material.foreground_voice is None or not material.harmonies:
        _fail("target material must declare harmony and foreground voice")
    foreground = tuple(note for note in material.notes if note.voice == material.foreground_voice)
    if not foreground:
        _fail("target material must contain foreground notes")
    return {
        "schema_version": 1,
        "piece_plan": {"tonal_center": plan.tonal_center, "mode": plan.mode},
        "score": {"score_id": score.score_id, "divisions": score.divisions},
        "material": {
            "material_id": material.material_id,
            "length_units": material.length_units,
            "derived_from": material.derived_from,
            "harmonies": [asdict(item) for item in material.harmonies],
            "foreground_voice": material.foreground_voice,
            "foreground_notes": [asdict(item) for item in foreground],
            "directions": [asdict(item) for item in material.directions],
            "comparison_conditions": {
                "event_count": TEXTURE_EVENT_COUNT,
                "attack_count": TEXTURE_ATTACK_COUNT,
                "attack_cell_units": score.divisions,
            },
        },
    }


def _overlaps(start_a: int, duration_a: int, start_b: int, duration_b: int) -> bool:
    return start_a < start_b + duration_b and start_b < start_a + duration_a


def _harmony_for_event(material: ScoreMaterial, event: PianoTextureEvent) -> ScoreHarmony:
    matches = tuple(item for item in material.harmonies if item.harmony_id == event.harmony_id)
    if len(matches) != 1:
        _fail(f"texture harmony_id is unknown: {event.harmony_id}")
    harmony = matches[0]
    if (
        event.at_units < harmony.at_units
        or event.duration_units <= 0
        or event.at_units + event.duration_units > harmony.at_units + harmony.duration_units
    ):
        _fail("texture event must fit inside one declared harmony interval")
    return harmony


def _degree_pitch(harmony: ScoreHarmony, degree: str, octave: int) -> int:
    index = _DEGREE_INDEX.get(degree)
    intervals = HARMONY_INTERVALS[harmony.quality]
    if index is None or index >= len(intervals):
        _fail(f"texture degree is not available in {harmony.quality}: {degree}")
    pitch_class = (harmony.root_pitch_class + intervals[index]) % 12
    pitch = 12 * (octave + 1) + pitch_class
    if not 21 <= pitch <= 108:
        _fail("converted texture pitch is outside MIDI piano range")
    return pitch


def validate_and_convert_piano_texture(
    plan: PiecePlan,
    score: ScoreSpec,
    texture: PianoTextureSpec,
) -> tuple[ScoreNote, ...]:
    """実験固有の条件を検査し、伴奏ScoreNoteへ決定的に変換する。"""

    validate_score_spec(plan, score)
    if texture.material_id != TARGET_MATERIAL_ID:
        _fail(f"texture material_id must be {TARGET_MATERIAL_ID}")
    material = _material(score, texture.material_id)
    if material.foreground_voice is None or not material.harmonies:
        _fail("target material must declare harmony and foreground voice")
    if len(texture.events) != TEXTURE_EVENT_COUNT:
        _fail(f"texture must contain exactly {TEXTURE_EVENT_COUNT} events")
    attacks = {event.at_units for event in texture.events}
    if len(attacks) != TEXTURE_ATTACK_COUNT:
        _fail(f"texture must contain exactly {TEXTURE_ATTACK_COUNT} attack positions")
    for start in range(0, material.length_units, score.divisions):
        end = min(start + score.divisions, material.length_units)
        if not any(start <= attack < end for attack in attacks):
            _fail("texture must attack in each divisions-sized cell")

    original_ids = {note.event_id for item in score.materials for note in item.notes}
    seen_ids: set[str] = set()
    accompaniment_voice = "lower" if material.foreground_voice == "upper" else "upper"
    foreground = tuple(note for note in material.notes if note.voice == material.foreground_voice)
    converted: list[ScoreNote] = []
    for event in texture.events:
        if not event.event_id or event.event_id in seen_ids or event.event_id in original_ids:
            _fail("texture event IDs must be new, non-empty, and unique")
        seen_ids.add(event.event_id)
        if event.role != "accompaniment":
            _fail("texture role must be accompaniment")
        if event.voice != accompaniment_voice:
            _fail("texture voice must be the accompaniment voice")
        if event.at_units < 0:
            _fail("texture event position must be non-negative")
        if len(set(event.articulations)) != len(event.articulations) or any(
            item not in _ARTICULATIONS for item in event.articulations
        ):
            _fail("texture articulations are outside the ScoreNote vocabulary")
        harmony = _harmony_for_event(material, event)
        pitch = _degree_pitch(harmony, event.degree, event.octave)
        overlapping_foreground = tuple(
            note
            for note in foreground
            if _overlaps(
                event.at_units,
                event.duration_units,
                note.at_units,
                note.duration_units,
            )
        )
        if material.foreground_voice == "upper" and any(
            pitch >= note.pitch for note in overlapping_foreground
        ):
            _fail("texture voice crossing is not allowed")
        if material.foreground_voice == "lower" and any(
            pitch <= note.pitch for note in overlapping_foreground
        ):
            _fail("texture voice crossing is not allowed")
        converted.append(
            ScoreNote(
                event_id=event.event_id,
                at_units=event.at_units,
                duration_units=event.duration_units,
                pitch=pitch,
                voice=event.voice,
                tie=None,
                articulations=event.articulations,
            )
        )

    for index, note in enumerate(converted):
        for other in converted[index + 1 :]:
            if (
                note.voice == other.voice
                and note.pitch == other.pitch
                and _overlaps(
                    note.at_units,
                    note.duration_units,
                    other.at_units,
                    other.duration_units,
                )
            ):
                _fail("the same texture pitch cannot overlap within one voice")
    return tuple(converted)


def combine_piano_texture(
    plan: PiecePlan,
    score: ScoreSpec,
    texture: PianoTextureSpec,
) -> ScoreSpec:
    """対象素材の伴奏全体だけをPianoTextureSpecで置き換える。"""

    converted = validate_and_convert_piano_texture(plan, score, texture)
    target = _material(score, texture.material_id)
    assert target.foreground_voice is not None
    foreground = tuple(note for note in target.notes if note.voice == target.foreground_voice)
    notes = tuple(
        sorted(
            (*foreground, *converted),
            key=lambda item: (item.at_units, item.voice, item.pitch, item.event_id),
        )
    )
    replacement = replace(target, notes=notes)
    combined = replace(
        score,
        materials=tuple(
            replacement if item.material_id == target.material_id else item
            for item in score.materials
        ),
    )
    validate_combination_scope(score, combined, target.material_id)
    validate_score_spec(plan, combined)
    return combined


def validate_combination_scope(
    before: ScoreSpec,
    after: ScoreSpec,
    material_id: str = TARGET_MATERIAL_ID,
) -> None:
    """対象伴奏以外のScoreSpec固定情報が意味的に同一か検査する。"""

    if before.score_id != after.score_id or before.divisions != after.divisions:
        _fail("fixed score content changed outside the texture replacement")
    if tuple(item.material_id for item in before.materials) != tuple(
        item.material_id for item in after.materials
    ):
        _fail("fixed score content changed outside the texture replacement")
    for old, new in zip(before.materials, after.materials, strict=True):
        if old.material_id != material_id:
            if old != new:
                _fail("fixed score content changed outside the texture replacement")
            continue
        if old.foreground_voice is None or new.foreground_voice != old.foreground_voice:
            _fail("fixed score content changed outside the texture replacement")
        old_fixed = replace(
            old,
            notes=tuple(note for note in old.notes if note.voice == old.foreground_voice),
        )
        new_fixed = replace(
            new,
            notes=tuple(note for note in new.notes if note.voice == new.foreground_voice),
        )
        if old_fixed != new_fixed:
            _fail("fixed score content changed outside the texture replacement")


def _immutable_text(path: Path, content: str) -> Path:
    encoded = content.encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            _fail(f"saved pilot artifact conflicts with current content: {path}")
        return path
    atomic_write_bytes(path, encoded)
    return path


def _artifact_record(path: Path, output_dir: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(output_dir)).replace("\\", "/"),
        "sha256": sha256_file(path),
    }


def _render_prompt(template: str, fixed_view: dict[str, object]) -> str:
    placeholder = "{{FIXED_MATERIAL_JSON}}"
    if template.count(placeholder) != 1:
        _fail("texture prompt must contain the fixed material placeholder exactly once")
    fixed_json = json.dumps(fixed_view, ensure_ascii=False, indent=2, sort_keys=True)
    return template.replace(placeholder, fixed_json)


def prepare_piano_texture_pilot(
    *,
    piece_plan_path: Path,
    score_spec_path: Path,
    output_dir: Path,
    prompt_template_path: Path = DEFAULT_PROMPT_PATH,
) -> dict[str, object]:
    """外部呼出し前の不変入力、LLM view、プロンプトを準備する。"""

    piece_plan_path = Path(piece_plan_path).resolve()
    score_spec_path = Path(score_spec_path).resolve()
    output_dir = Path(output_dir).resolve()
    prompt_template_path = Path(prompt_template_path).resolve()
    plan_source = piece_plan_path.read_text(encoding="utf-8")
    score_source = score_spec_path.read_text(encoding="utf-8")
    plan = parse_piece_plan(plan_source)
    score = parse_score_spec(score_source)
    validate_score_spec(plan, score)
    fixed_view = build_fixed_material_view(plan, score, TARGET_MATERIAL_ID)
    template = prompt_template_path.read_text(encoding="utf-8")
    prompt = _render_prompt(template, fixed_view)
    implementation_path = Path(__file__).resolve()
    run_spec = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "target_material_id": TARGET_MATERIAL_ID,
        "model": MODEL_ID,
        "reasoning_effort": REASONING_EFFORT,
        "max_external_calls": 1,
        "inputs": {
            "piece_plan_sha256": sha256_text(plan_source),
            "score_spec_sha256": sha256_text(score_source),
            "prompt_template_sha256": sha256_file(prompt_template_path),
            "implementation_sha256": sha256_file(implementation_path),
        },
    }
    store = RunStore(output_dir, max_calls=1)
    store.initialize(run_spec)
    plan_snapshot = store.snapshot_file("input-piece-plan.dsl", piece_plan_path)
    score_snapshot = store.snapshot_file("input-score-spec.dsl", score_spec_path)
    fixed_path = store.snapshot_json("fixed-material.json", fixed_view)
    prompt_path = output_dir / "prompt.txt"
    store.promote_bytes(prompt.encode("utf-8"), prompt_path, sha256_text(prompt))
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") != "prepared":
            raise PianoTextureValidationError("completed pilot cannot be prepared again")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "prepared",
        "passes": False,
        "source_kind": None,
        "confirmed_external_call_count": 0,
        "model": MODEL_ID,
        "reasoning_effort": REASONING_EFFORT,
        "files": {
            "input_piece_plan": _artifact_record(plan_snapshot, output_dir),
            "input_score_spec": _artifact_record(score_snapshot, output_dir),
            "fixed_material": _artifact_record(fixed_path, output_dir),
            "prompt": _artifact_record(prompt_path, output_dir),
            "prompt_template": {
                "path": str(prompt_template_path),
                "sha256": sha256_file(prompt_template_path),
            },
            "implementation": {
                "path": str(implementation_path),
                "sha256": sha256_file(implementation_path),
            },
        },
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def _confirmed_external_call_count(runner: TextureRunner, fallback: int) -> int:
    store = getattr(runner, "run_store", None)
    if store is None:
        return fallback
    state = store.rebuild_state()
    calls = state.get("calls")
    if isinstance(calls, dict):
        value = calls.get("confirmed_external_call_count")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return fallback


def _failed_manifest(
    manifest: dict[str, object],
    *,
    runner: TextureRunner,
    fallback_calls: int,
    error: Exception,
) -> dict[str, object]:
    manifest.update(
        {
            "status": "failed",
            "passes": False,
            "source_kind": "external",
            "confirmed_external_call_count": _confirmed_external_call_count(runner, fallback_calls),
            "error": {"type": type(error).__name__, "detail": str(error)},
        }
    )
    return manifest


def execute_piano_texture_pilot(
    *,
    output_dir: Path,
    runner: TextureRunner,
) -> dict[str, object]:
    """準備済み入力から外部応答を一度だけ取得し、検査結果を確定する。"""

    output_dir = Path(output_dir).resolve()
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "prepared":
        return manifest
    plan_source = (output_dir / "input-piece-plan.dsl").read_text(encoding="utf-8")
    score_source = (output_dir / "input-score-spec.dsl").read_text(encoding="utf-8")
    prompt = (output_dir / "prompt.txt").read_text(encoding="utf-8")
    fixed_view = json.loads((output_dir / "fixed-material.json").read_text(encoding="utf-8"))
    plan = parse_piece_plan(plan_source)
    score = parse_score_spec(score_source)
    validate_score_spec(plan, score)
    before_calls = runner.call_number
    input_hashes = {
        "piece_plan": sha256_text(plan_source),
        "score_spec": sha256_text(score_source),
        "fixed_material": sha256_json(fixed_view),
        "prompt": sha256_text(prompt),
    }
    try:
        response = runner.run(STEP_ID, prompt, input_hashes)
        after_calls = runner.call_number
        call_delta = after_calls - before_calls
        if call_delta != 1:
            _fail("pilot must reserve exactly one external attempt")
        source = response.get("composition_source")
        if not isinstance(source, str) or not source:
            _fail("external response has no PianoTextureSpec source")
        response_path = _immutable_text(output_dir / "response.txt", source)
        files = manifest.setdefault("files", {})
        assert isinstance(files, dict)
        files["response"] = _artifact_record(response_path, output_dir)
        texture = parse_piano_texture_spec(source)
        canonical = dump_piano_texture_spec(texture)
        texture_path = _immutable_text(output_dir / "piano-texture.dsl", canonical)
        files["piano_texture"] = _artifact_record(texture_path, output_dir)
        combined = combine_piano_texture(plan, score, texture)
        combined_source = dump_score_spec(combined)
        combined_path = _immutable_text(output_dir / "combined-score-spec.dsl", combined_source)
        files["combined_score_spec"] = _artifact_record(combined_path, output_dir)
        quality = evaluate_generic_score_quality(plan, combined)
        quality_path = output_dir / "score-quality.json"
        atomic_write_json(quality_path, quality)
        files["score_quality"] = _artifact_record(quality_path, output_dir)
        passes = quality.get("passes") is True
        manifest.update(
            {
                "status": "completed" if passes else "failed",
                "passes": passes,
                "source_kind": "external",
                "confirmed_external_call_count": _confirmed_external_call_count(
                    runner, after_calls
                ),
            }
        )
        if not passes:
            manifest["error"] = {
                "type": "ScoreQualityFailure",
                "detail": "combined ScoreSpec failed existing generic quality",
            }
    except Exception as error:
        after_calls = runner.call_number
        manifest = _failed_manifest(
            manifest,
            runner=runner,
            fallback_calls=max(after_calls, before_calls),
            error=error,
        )
    atomic_write_json(manifest_path, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--piece-plan", type=Path, required=True)
    prepare.add_argument("--score-spec", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--prompt-template", type=Path, default=DEFAULT_PROMPT_PATH)
    run = subparsers.add_parser("run")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--timeout-seconds", type=int, default=1200)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        prepare_piano_texture_pilot(
            piece_plan_path=args.piece_plan,
            score_spec_path=args.score_spec,
            output_dir=args.output_dir,
            prompt_template_path=args.prompt_template,
        )
        return 0
    from llm_musical_composer.pilot_loop import (
        CodexExecRunner,
        isolated_codex_working_directory,
    )

    output_dir = Path(args.output_dir).resolve()
    store = RunStore(output_dir, max_calls=1)
    runner = CodexExecRunner(
        run_store=store,
        schema_path=DEFAULT_SCHEMA_PATH,
        model=MODEL_ID,
        reasoning_effort=REASONING_EFFORT,
        working_directory=isolated_codex_working_directory(),
        timeout_seconds=args.timeout_seconds,
    )
    with RunLock(output_dir / ".lock"):
        result = execute_piano_texture_pilot(output_dir=output_dir, runner=runner)
    return 0 if result.get("passes") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
