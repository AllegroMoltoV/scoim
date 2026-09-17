"""曲構成を観測し、既存部分を保持して終止素材だけを追加する反復。"""

from __future__ import annotations

import argparse
import enum
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from llm_musical_composer.composition_ir import Composition
from llm_musical_composer.ending_closure import (
    build_closure_profile,
    evaluate_closure_against_profile,
    evaluate_ending_discrimination,
    extract_closure_features,
    run_closure_controls,
)
from llm_musical_composer.listening_evaluation import create_listening_package
from llm_musical_composer.music_dsl import DslError, parse_composition
from llm_musical_composer.pilot_features import NoteEvent
from llm_musical_composer.pilot_loop import MODEL_ID, CodexExecRunner
from llm_musical_composer.smf_notes import load_smf_notes
from llm_musical_composer.smf_render import render_composition
from llm_musical_composer.structure_features import (
    build_structure_reference_profile,
    extract_structure_features,
    run_structure_controls,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_CALLS = 2
EXCLUDED_NAMES = frozenset({"rut.mid", "aimusic01.mid"})


class Runner(Protocol):
    def run(self, prompt: str, response_path: Path) -> dict[str, object]: ...


class StructureRunStatus(enum.Enum):
    COMPLETED = "completed"
    CLOSURE_CONTROL_FAILED = "closure_control_failed"
    REVISION_FAILED = "revision_failed"


@dataclass(frozen=True)
class StructureIterationResult:
    status: StructureRunStatus
    call_count: int
    listening_directory: Path | None
    error: str | None = None


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _notes(path: Path) -> list[NoteEvent]:
    return [
        NoteEvent(note.pitch, note.onset_ms, note.duration_ms, note.velocity)
        for note in load_smf_notes(path)
    ]


def _expanded_last_pedal(composition: Composition) -> tuple[int, int] | None:
    offset = 0
    latest: tuple[int, int] | None = None
    materials = composition.material_by_id
    for section in composition.form:
        material = materials[section.material_id]
        for pedal in material.pedals:
            event = (offset + pedal.at_ms, pedal.value)
            if latest is None or event[0] >= latest[0]:
                latest = event
        offset += material.duration_ms
    return latest


def validate_closing_revision(before: Composition, after: Composition) -> list[str]:
    """既存内容を変更せず closing だけを追加したか検査する。"""
    issues: list[str] = []
    if before.title != after.title:
        issues.append("title changed")
    if after.form[:-1] != before.form or not after.form or after.form[-1].material_id != "closing":
        issues.append("form must append closing exactly once")
    before_materials = before.material_by_id
    after_materials = after.material_by_id
    if set(after_materials) != {*before_materials, "closing"}:
        issues.append("material set must add only closing")
    if any(section.material_id not in after_materials for section in after.form):
        return issues
    for material_id, material in before_materials.items():
        if after_materials.get(material_id) != material:
            issues.append(f"material {material_id} changed")
    closing = after_materials.get("closing")
    if closing is not None and not 4_000 <= closing.duration_ms <= 8_000:
        issues.append("closing duration must be between 4000 and 8000 ms")
    if not 52_000 <= after.duration_ms <= 56_000:
        issues.append("expanded duration must be between 52000 and 56000 ms")
    if after.note_count > 512:
        issues.append("expanded note count must not exceed 512")
    last_note_end = 0
    offset = 0
    for section in after.form:
        material = after_materials.get(section.material_id)
        if material is None:
            continue
        last_note_end = max(
            last_note_end,
            *(offset + note.at_ms + note.duration_ms for note in material.notes),
        )
        offset += material.duration_ms
    last_pedal = _expanded_last_pedal(after)
    if last_pedal is not None and (last_pedal[1] != 0 or last_pedal[0] > last_note_end):
        issues.append("pedal must be off by the final note ending")
    return issues


def _response_source(response: dict[str, object]) -> str:
    source = response.get("composition_source")
    if not isinstance(source, str):
        raise ValueError("response has no string composition_source")
    return source


def _prompt(name: str, replacements: dict[str, str]) -> str:
    value = (PROJECT_ROOT / "prompts" / name).read_text(encoding="utf-8")
    for key, replacement in replacements.items():
        value = value.replace("{{" + key + "}}", replacement)
    return value


def _revision_error(before: Composition, source: str) -> tuple[Composition | None, str | None]:
    try:
        after = parse_composition(source)
    except (DslError, ValueError) as error:
        return None, f"{type(error).__name__}: {error}"
    issues = validate_closing_revision(before, after)
    return (None, "; ".join(issues)) if issues else (after, None)


def _declared_form_evidence(
    composition: Composition, notes: list[NoteEvent], *, window_count: int = 16
) -> tuple[list[str], list[dict[str, Any]]]:
    sounding_start = min(note.onset_ms for note in notes)
    sounding_end = max(note.onset_ms + note.duration_ms for note in notes)
    sounding_span = sounding_end - sounding_start
    materials = composition.material_by_id
    sections: list[tuple[int, int, str]] = []
    offset = 0
    for use in composition.form:
        ending = offset + materials[use.material_id].duration_ms
        sections.append((offset, ending, use.material_id))
        offset = ending
    window_materials: list[str] = []
    for index in range(window_count):
        center = sounding_start + sounding_span * ((index + 0.5) / window_count)
        material_id = next(
            (material_id for start, end, material_id in sections if start <= center < end),
            sections[-1][2],
        )
        window_materials.append(material_id)
    boundaries = [
        {
            "after_section": index,
            "position": round((end - sounding_start) / sounding_span, 8),
            "left_material": sections[index][2],
            "right_material": sections[index + 1][2],
        }
        for index, (_, end, _) in enumerate(sections[:-1])
        if sounding_start < end < sounding_end
    ]
    return window_materials, boundaries


def _structure_report(
    composition: Composition, notes: list[NoteEvent], closure_profile: dict[str, Any]
) -> dict[str, Any]:
    declared_materials, declared_boundaries = _declared_form_evidence(composition, notes)
    structure = extract_structure_features(notes, declared_material_ids=declared_materials)
    candidates = structure["novelty"]["consensus_candidates"]
    for boundary in declared_boundaries:
        nearest = min(
            (abs(boundary["position"] - candidate["position"]) for candidate in candidates),
            default=None,
        )
        boundary["nearest_consensus_distance"] = round(nearest, 8) if nearest is not None else None
        boundary["matched_within_one_window"] = nearest is not None and nearest <= 1 / 16
    closure = extract_closure_features(notes)
    return {
        "structure": structure,
        "declared_form": {
            "window_material_ids": declared_materials,
            "boundaries": declared_boundaries,
        },
        "closure": closure,
        "closure_reference_comparison": evaluate_closure_against_profile(closure, closure_profile)
        if closure_profile.get("metrics")
        else {"metrics": {}},
    }


def _write_structure_listening_form(path: Path) -> None:
    _write_json(
        path,
        {
            "schema_version": 1,
            "allowed_values": [1, 2, 3, 4, 5, "判定不能", "該当なし"],
            "blind_axes": [
                {"name": "再生上の正常性", "question": "音の鳴りっぱなしや不自然な途切れがないか"},
                {"name": "区間の切り替わり", "question": "区間の変化を自然に感じられるか"},
                {"name": "反復と変化の釣り合い", "question": "繰り返しと変化が適切か"},
                {"name": "盛り上がり位置と強さ", "question": "高まりの位置と程度が自然か"},
                {"name": "終止感", "question": "最後に曲が閉じたと感じられるか"},
                {"name": "全体構成と展開", "question": "始まりから終わりまで展開が成立しているか"},
                {"name": "総合的な音楽品質", "question": "楽曲として成立しているか"},
            ],
            "revealed_axes": [
                {
                    "name": "終止感の変化",
                    "values": ["改善", "変化なし", "悪化", "判定不能"],
                },
                {
                    "name": "対象外の保持",
                    "values": ["保持", "知覚できる不要変更あり", "判定不能"],
                },
                {"name": "全体比較", "values": ["修正版が良い", "同程度", "修正版が悪い"]},
            ],
            "rule": "値を合算しない",
        },
    )


def _write_manifest(
    output_dir: Path,
    result: StructureIterationResult,
    *,
    control: dict[str, Any],
) -> None:
    _write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "status": result.status.value,
            "requested_model": MODEL_ID,
            "reported_model": None,
            "reported_model_status": "unable_to_investigate",
            "call_count": result.call_count,
            "listening_directory": str(result.listening_directory)
            if result.listening_directory
            else None,
            "error": result.error,
            "ending_discrimination": control,
        },
    )


def run_structure_iteration(
    before_source: str,
    closure_profile: dict[str, Any],
    ending_control: dict[str, Any],
    runner: Runner,
    output_dir: Path,
    *,
    random_seed: int | None = None,
) -> StructureIterationResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    before = parse_composition(before_source)
    (output_dir / "before.music.py").write_text(before_source + "\n", encoding="utf-8")
    before_path = output_dir / "before.mid"
    render_composition(before, before_path)
    before_report = _structure_report(before, _notes(before_path), closure_profile)
    _write_json(output_dir / "before-structure.json", before_report)
    if ending_control.get("status") != "pass":
        result = StructureIterationResult(
            StructureRunStatus.CLOSURE_CONTROL_FAILED,
            0,
            None,
            "ending discrimination did not pass",
        )
        _write_manifest(output_dir, result, control=ending_control)
        return result

    response_path = output_dir / "revision-response.json"
    response = runner.run(
        _prompt(
            "pilot-ending-revise.md",
            {
                "reference_closure_profile": json.dumps(
                    closure_profile, ensure_ascii=False, sort_keys=True
                ),
                "current_structure": json.dumps(before_report, ensure_ascii=False, sort_keys=True),
                "source": before_source,
            },
        ),
        response_path,
    )
    call_count = 1
    try:
        revised_source = _response_source(response)
        after, error = _revision_error(before, revised_source)
    except ValueError as caught:
        revised_source = ""
        after, error = None, f"{type(caught).__name__}: {caught}"
    if error:
        repair = runner.run(
            _prompt(
                "pilot-ending-repair.md",
                {"error": error, "source": revised_source, "original_source": before_source},
            ),
            output_dir / "repair-response.json",
        )
        call_count += 1
        try:
            revised_source = _response_source(repair)
            after, error = _revision_error(before, revised_source)
        except ValueError as caught:
            after, error = None, f"{type(caught).__name__}: {caught}"
    if error or after is None:
        result = StructureIterationResult(
            StructureRunStatus.REVISION_FAILED, call_count, None, error or "unknown revision error"
        )
        _write_manifest(output_dir, result, control=ending_control)
        return result

    (output_dir / "after.music.py").write_text(revised_source + "\n", encoding="utf-8")
    after_path = output_dir / "after.mid"
    render_composition(after, after_path)
    after_report = _structure_report(after, _notes(after_path), closure_profile)
    _write_json(output_dir / "after-structure.json", after_report)
    _write_json(
        output_dir / "similarity-matrices.json",
        {
            "before": before_report["structure"]["similarity_matrices"],
            "after": after_report["structure"]["similarity_matrices"],
        },
    )
    _write_json(
        output_dir / "activity-curves.json",
        {
            "before": before_report["structure"]["activity"],
            "after": after_report["structure"]["activity"],
        },
    )
    listening_directory = output_dir / "listen"
    package = create_listening_package(
        before_path,
        after_path,
        listening_directory,
        target_axis="ending_closure",
        target_material="closing",
        rng=random.Random(random_seed),
    )
    _write_structure_listening_form(package.evaluation_form)
    result = StructureIterationResult(StructureRunStatus.COMPLETED, call_count, listening_directory)
    _write_manifest(output_dir, result, control=ending_control)
    return result


def _reference_data(
    source_dir: Path,
) -> tuple[list[list[NoteEvent]], list[str], list[dict[str, str]]]:
    corpus: list[list[NoteEvent]] = []
    source_files: list[str] = []
    unable: list[dict[str, str]] = []
    excluded = {name.casefold() for name in EXCLUDED_NAMES}
    for path in sorted(Path(source_dir).glob("*.mid"), key=lambda item: item.name.casefold()):
        if path.name.casefold() in excluded:
            continue
        try:
            notes = _notes(path)
            if not notes:
                raise ValueError("no complete pitched notes")
            corpus.append(notes)
            source_files.append(path.name)
        except (OSError, ValueError, EOFError) as error:
            unable.append({"name": path.name, "error": f"{type(error).__name__}: {error}"})
    return corpus, source_files, unable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="曲構成を評価し、終止素材だけを修正します。")
    parser.add_argument("--source-dir", type=Path, default=Path(".appendix/source-smf"))
    parser.add_argument(
        "--baseline-source",
        type=Path,
        default=Path(".appendix/minimal-loop/20260811-172616/candidates/candidate-1.music.py"),
    )
    parser.add_argument("--output-root", type=Path, default=Path(".appendix/structure-evaluation"))
    parser.add_argument(
        "--schema", type=Path, default=Path("schemas/codex-composition-response.schema.json")
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)

    corpus, source_files, unable = _reference_data(args.source_dir)
    closure_features = [extract_closure_features(notes) for notes in corpus]
    closure_profile = build_closure_profile(closure_features)
    structure_reports = [extract_structure_features(notes) for notes in corpus]
    structure_profile = build_structure_reference_profile(structure_reports)
    profile = {
        "schema_version": 1,
        "source_count": len(corpus),
        "structure": structure_profile,
        "closure": closure_profile,
        "source_files": source_files,
        "excluded_files": sorted(EXCLUDED_NAMES),
        "unable_to_investigate": unable,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "reference-profile.json", profile)
    ending_control = evaluate_ending_discrimination(corpus)
    controls = {
        "structure": run_structure_controls(),
        "closure": run_closure_controls(corpus[0]) if corpus else {},
        "ending_discrimination": ending_control,
    }
    _write_json(args.output_root / "reference-controls.json", controls)

    run_id = args.run_id or datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    runner = CodexExecRunner(
        schema_path=args.schema,
        log_dir=Path(".logs") / f"structure-iteration-{run_id}",
        model=args.model,
        working_directory=Path.cwd(),
        max_calls=MAX_CALLS,
    )
    result = run_structure_iteration(
        args.baseline_source.read_text(encoding="utf-8"),
        closure_profile,
        ending_control,
        runner,
        args.output_root / run_id,
    )
    print(json.dumps({"status": result.status.value, "run_id": run_id}, ensure_ascii=False))
    return 0 if result.status is StructureRunStatus.COMPLETED else 2


if __name__ == "__main__":
    raise SystemExit(main())
