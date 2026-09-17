"""参照作風目標を固定し、三分曲を段階生成して機械検証する。"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import mido

from llm_musical_composer import (
    composition_ir as composition_ir_module,
)
from llm_musical_composer import (
    long_form_evaluation as long_form_evaluation_module,
)
from llm_musical_composer import (
    long_form_generation as long_form_generation_module,
)
from llm_musical_composer import music_dsl as music_dsl_module
from llm_musical_composer import piano_texture as piano_texture_module
from llm_musical_composer import style_target as style_target_module
from llm_musical_composer.long_form_evaluation import evaluate_long_form_structure
from llm_musical_composer.long_form_generation import (
    MAX_NATURAL_NOTE_COUNT,
    LongFormGenerationError,
    _validate_long_form_phrase_structure,
    _validate_natural_long_form_plan,
    _validate_natural_materials,
    run_staged_generation,
)
from llm_musical_composer.material_development import (
    build_material_development_reference_profile_from_directory,
    evaluate_material_development,
)
from llm_musical_composer.piano_texture import (
    evaluate_piano_texture,
    evaluate_variation_contracts,
)
from llm_musical_composer.pilot_features import NoteEvent
from llm_musical_composer.pilot_loop import (
    MODEL_ID,
    CodexExecRunner,
    _validate_rendered_ending,
    isolated_codex_working_directory,
)
from llm_musical_composer.reference_structure import analyze_note_structure
from llm_musical_composer.run_state import (
    RunLock,
    RunStore,
    atomic_write_json,
    sha256_file,
    sha256_json,
)
from llm_musical_composer.smf_notes import load_smf_notes
from llm_musical_composer.smf_render import render_composition
from llm_musical_composer.style_target import (
    EXCLUDED_NAMES,
    evaluate_style_features,
    evaluate_style_target_feasibility,
    extract_style_features,
)
from llm_musical_composer.sustain_profile import evaluate_sustain_profile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_CALLS = 6
DEFAULT_PLAN_BRIEF = (
    "既存の完全例と同じ大区分数、素材時間、フレーズ配列を複写せず、"
    "制約内で非対称な構成と離れた位置での回帰を作る。"
)
_VALIDATION_EXAMPLE = re.compile(
    r"<!-- INTERNAL_VALIDATION_EXAMPLE_BEGIN -->.*?"
    r"<!-- INTERNAL_VALIDATION_EXAMPLE_END -->",
    flags=re.DOTALL,
)
STEP_IDS = (
    "plan",
    "material-batch-01",
    "material-batch-02",
    "material-batch-03",
    "material-batch-04",
    "material-repair-variations",
    "generate-candidate",
    "evaluate-candidate",
    "publish-final",
)


def load_plan_source(path: Path) -> str:
    """保存済みの構造化応答または DSL ファイルから計画原文を読む。"""
    path = Path(path)
    if path.suffix.casefold() != ".json":
        return path.read_text(encoding="utf-8")
    value = _read_object(path)
    source = value.get("composition_source")
    if not isinstance(source, str) or not source:
        raise ValueError("saved plan has no string composition_source")
    return source


class PresetPlanRunner:
    """保存済み計画だけを外部呼び出しなしで返し、素材生成を委譲する。"""

    def __init__(self, delegate: Any, plan_source: str) -> None:
        self.delegate = delegate
        self.plan_source = plan_source

    def run(
        self, step_id: str, prompt: str, input_hashes: dict[str, str] | None = None
    ) -> dict[str, object]:
        if step_id == "plan":
            return {
                "composition_source": self.plan_source,
                "intent_summary": "reused validated saved plan response",
            }
        return self.delegate.run(step_id, prompt, input_hashes)


def _validate_saved_response(step_id: str, value: dict[str, Any]) -> dict[str, Any]:
    for key in ("composition_source", "intent_summary"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f"saved response {step_id} has no string {key}")
    return value


def load_preset_responses(directory: Path) -> dict[str, dict[str, Any]]:
    """既知の生成段階に対応する保存済み構造化応答を読む。"""
    directory = Path(directory)
    allowed = frozenset(STEP_IDS[:6])
    responses: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        step_id = path.stem
        if step_id not in allowed:
            raise ValueError(f"unknown preset response step: {step_id}")
        responses[step_id] = _validate_saved_response(step_id, _read_object(path))
    if not responses:
        raise ValueError(f"preset response directory contains no usable responses: {directory}")
    return responses


class PresetResponseRunner:
    """保存済み応答を返し、存在しない段階だけを外部生成へ委譲する。"""

    def __init__(self, delegate: Any, responses: dict[str, dict[str, Any]]) -> None:
        self.delegate = delegate
        self.responses = responses

    def run(
        self, step_id: str, prompt: str, input_hashes: dict[str, str] | None = None
    ) -> dict[str, Any]:
        if step_id in self.responses:
            return self.responses[step_id]
        return self.delegate.run(step_id, prompt, input_hashes)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must be an object: {path}")
    return value


def load_verified_style_target(
    target_path: Path, manifest_path: Path, reference_records_path: Path
) -> dict[str, Any]:
    """目標と参照記録が作成時から変わっていないことを確認する。"""
    target_path = Path(target_path)
    manifest_path = Path(manifest_path)
    reference_records_path = Path(reference_records_path)
    target = _read_object(target_path)
    manifest = _read_object(manifest_path)
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict):
        raise ValueError("style target manifest has no sha256 object")
    target_hash = sha256_file(target_path)
    if hashes.get("style_target") != target_hash:
        raise ValueError("style target fingerprint differs from its manifest")
    records_hash = sha256_file(reference_records_path)
    if hashes.get("input_jsonl") != records_hash:
        raise ValueError("reference records fingerprint differs from the style target manifest")
    if target.get("status") != "pass" or not isinstance(target.get("prompt_target"), dict):
        raise ValueError("style target is not usable")
    feasibility = evaluate_style_target_feasibility(
        target["prompt_target"],
        duration_ms=180_000,
        max_note_count=MAX_NATURAL_NOTE_COUNT,
    )
    return {
        "target": target,
        "manifest": manifest,
        "prompt_target": target["prompt_target"],
        "feasibility": feasibility,
        "hashes": {
            "style_target": target_hash,
            "style_manifest": sha256_file(manifest_path),
            "reference_records": records_hash,
            "prompt_target": sha256_json(target["prompt_target"]),
        },
    }


def build_plan_prompt(
    template: str,
    prompt_target: dict[str, Any],
    *,
    plan_brief: str = DEFAULT_PLAN_BRIEF,
) -> str:
    """曲名や音列を含めず、数値範囲だけを計画用指示へ埋め込む。"""
    if not plan_brief.strip():
        raise ValueError("plan brief must not be empty")
    live_template = _VALIDATION_EXAMPLE.sub("", template)
    rendered = live_template.replace(
        "{{style_target}}",
        json.dumps(prompt_target, ensure_ascii=False, indent=2, sort_keys=True),
    ).replace("{{plan_brief}}", plan_brief.strip())
    if re.search(r"\{\{[A-Za-z0-9_]+\}\}", rendered):
        raise ValueError("plan prompt contains unresolved variables")
    return rendered


def _smf_report(path: Path, expected_note_count: int) -> dict[str, Any]:
    midi = mido.MidiFile(path, charset="utf-8")
    absolute_tick = 0
    note_on_count = 0
    note_off_count = 0
    cc64_count = 0
    for message in mido.merge_tracks(midi.tracks):
        absolute_tick += message.time
        if message.type == "note_on" and message.velocity > 0:
            note_on_count += 1
        elif message.type == "note_off" or (message.type == "note_on" and message.velocity == 0):
            note_off_count += 1
        elif message.type == "control_change" and message.control == 64:
            cc64_count += 1
    loaded_note_count = len(load_smf_notes(path))
    issues = []
    if midi.ticks_per_beat != 500:
        issues.append("ticks_per_beat differs from 500")
    if absolute_tick != 180_000:
        issues.append(f"SMF ends at {absolute_tick} ticks instead of 180000")
    if note_on_count != expected_note_count or loaded_note_count != expected_note_count:
        issues.append("SMF reread note count differs from the rendered composition")
    if note_off_count != note_on_count:
        issues.append("SMF note-on and note-off counts differ")
    if cc64_count == 0:
        issues.append("SMF contains no CC64 events")
    return {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "ticks_per_beat": midi.ticks_per_beat,
        "end_tick": absolute_tick,
        "track_count": len(midi.tracks),
        "note_on_count": note_on_count,
        "note_off_count": note_off_count,
        "loaded_note_count": loaded_note_count,
        "cc64_count": cc64_count,
    }


def evaluate_long_form_candidate(
    composition: Any,
    midi_path: Path,
    *,
    style_target: dict[str, Any],
    material_development_profile: dict[str, Any],
    require_naturalness: bool = False,
    require_voice_structure: bool = False,
) -> dict[str, Any]:
    """必須契約と未校正の参考指標を混ぜずに評価する。"""
    sustain = evaluate_sustain_profile(composition.materials)
    structure = evaluate_long_form_structure(composition)
    ending: dict[str, Any]
    try:
        _validate_rendered_ending(composition, midi_path)
        ending = {"status": "pass", "issues": []}
    except ValueError as error:
        ending = {"status": "fail", "issues": [str(error)]}
    smf = _smf_report(midi_path, composition.note_count)
    notes = load_smf_notes(midi_path)
    analysis = analyze_note_structure(
        [NoteEvent(note.pitch, note.onset_ms, note.duration_ms, note.velocity) for note in notes],
        resolutions=(4,),
    )
    style_record = {"name": midi_path.name, **analysis}
    try:
        style = evaluate_style_features(extract_style_features(style_record), style_target)
        style["status"] = "pass"
    except ValueError as error:
        style = {"status": "unable_to_investigate", "error": str(error)}
    development = evaluate_material_development(composition.materials, material_development_profile)
    piano_texture = evaluate_piano_texture(composition)
    variation_contracts = evaluate_variation_contracts(composition)
    naturalness: dict[str, Any] | None = None
    if require_naturalness:
        naturalness_issues: list[str] = []
        try:
            _validate_natural_long_form_plan(composition)
            _validate_natural_materials(composition, composition.materials)
            if composition.phrases:
                _validate_long_form_phrase_structure(composition)
        except LongFormGenerationError as error:
            naturalness_issues.append(str(error))
        if composition.note_count > MAX_NATURAL_NOTE_COUNT:
            naturalness_issues.append(f"composition exceeds {MAX_NATURAL_NOTE_COUNT} notes")
        early_releases = []
        for material in composition.materials:
            last_pedal = max(
                material.pedals,
                key=lambda pedal: (pedal.at_ms, pedal.event_id),
                default=None,
            )
            if (
                last_pedal is None
                or last_pedal.value != 0
                or last_pedal.at_ms != material.duration_ms
            ):
                early_releases.append(material.material_id)
        if early_releases:
            naturalness_issues.append(
                "materials do not release the final pedal at their boundary: "
                + ", ".join(early_releases)
            )
        use_counts: dict[str, int] = {}
        for use in composition.form:
            use_counts[use.material_id] = use_counts.get(use.material_id, 0) + 1
        naturalness = {
            "status": "pass" if not naturalness_issues else "fail",
            "issues": naturalness_issues,
            "note_count": composition.note_count,
            "use_count": len(composition.form),
            "unique_material_count": len(use_counts),
            "unique_material_ratio": round(len(use_counts) / len(composition.form), 6),
            "maximum_material_use_count": max(use_counts.values(), default=0),
            "transition_use_count": sum(use.role == "transition" for use in composition.form),
            "phrase_count": len(composition.phrases),
            "early_release_materials": early_releases,
        }
    hard_gates = {
        "duration": composition.duration_ms == 180_000,
        "sustain": sustain["status"] == "pass",
        "macro_structure": structure["status"] == "pass",
        "tonic_ending": ending["status"] == "pass",
        "smf_reread": smf["status"] == "pass",
        **({"naturalness": naturalness["status"] == "pass"} if naturalness is not None else {}),
        **(
            {
                "piano_texture": piano_texture["status"] == "pass",
                "typed_variations": variation_contracts["status"] == "pass",
            }
            if require_voice_structure
            else {}
        ),
    }
    return {
        "status": "pass" if all(hard_gates.values()) else "fail",
        "hard_gates": hard_gates,
        "duration_ms": composition.duration_ms,
        "composition_note_count": composition.note_count,
        "sustain": sustain,
        "macro_structure": structure,
        "tonic_ending": ending,
        "smf_reread": smf,
        "style_target": style,
        "material_development": development,
        "piano_texture": piano_texture,
        "variation_contracts": variation_contracts,
        **({"naturalness": naturalness} if naturalness is not None else {}),
        "metric_policy": {
            "hard": list(hard_gates),
            "diagnostic_only": ["style_target", "material_development"],
        },
    }


def _implementation_hashes() -> dict[str, str]:
    modules = {
        "composition_ir.py": Path(composition_ir_module.__file__),
        "music_dsl.py": Path(music_dsl_module.__file__),
        "long_form_loop.py": Path(__file__),
        "long_form_generation.py": Path(long_form_generation_module.__file__),
        "long_form_evaluation.py": Path(long_form_evaluation_module.__file__),
        "piano_texture.py": Path(piano_texture_module.__file__),
        "style_target.py": Path(style_target_module.__file__),
    }
    return {name: sha256_file(path) for name, path in modules.items()}


def run_long_form(
    *,
    source_dir: Path,
    target_path: Path,
    manifest_path: Path,
    reference_records_path: Path,
    schema_path: Path,
    output_root: Path,
    run_id: str,
    model: str,
    plan_source_path: Path | None = None,
    preset_response_dir: Path | None = None,
    plan_brief_path: Path | None = None,
) -> dict[str, Any]:
    """ひとつの三分候補を生成し、合格した成果物だけを公開する。"""
    verified = load_verified_style_target(target_path, manifest_path, reference_records_path)
    if verified["feasibility"]["status"] != "pass":
        density = verified["feasibility"]["checks"]["density_level"]
        raise ValueError(
            "style density target is unreachable under the naturalness limit: "
            f"requires at least {density['minimum_required_note_count']} notes, "
            f"but the limit is {MAX_NATURAL_NOTE_COUNT}"
        )
    run_dir = Path(output_root).resolve() / run_id
    store = RunStore(run_dir, max_calls=MAX_CALLS)
    with RunLock(run_dir / ".run.lock"):
        target_snapshot = store.snapshot_file("inputs/style-target.json", target_path)
        manifest_snapshot = store.snapshot_file("inputs/style-target-manifest.json", manifest_path)
        records_snapshot = store.snapshot_file(
            "inputs/reference-files.jsonl", reference_records_path
        )
        plan_snapshot = store.snapshot_file(
            "inputs/long-form-plan.md", PROJECT_ROOT / "prompts/long-form-plan.md"
        )
        material_snapshot = store.snapshot_file(
            "inputs/long-form-material-batch.md",
            PROJECT_ROOT / "prompts/long-form-material-batch.md",
        )
        schema_snapshot = store.snapshot_file("inputs/response.schema.json", schema_path)
        plan_brief_snapshot = (
            store.snapshot_file("inputs/plan-brief.txt", plan_brief_path)
            if plan_brief_path is not None
            else None
        )
        plan_brief = (
            plan_brief_snapshot.read_text(encoding="utf-8")
            if plan_brief_snapshot is not None
            else DEFAULT_PLAN_BRIEF
        )
        plan_brief_hash = (
            sha256_file(plan_brief_snapshot)
            if plan_brief_snapshot is not None
            else sha256_json(plan_brief)
        )
        saved_plan_snapshot = (
            store.snapshot_file(
                f"inputs/preset-plan{Path(plan_source_path).suffix}", plan_source_path
            )
            if plan_source_path is not None
            else None
        )
        preset_response_snapshots: dict[str, Path] = {}
        if preset_response_dir is not None:
            for step_id in load_preset_responses(preset_response_dir):
                source = Path(preset_response_dir) / f"{step_id}.json"
                preset_response_snapshots[step_id] = store.snapshot_file(
                    f"inputs/preset-responses/{step_id}.json", source
                )
        spec = {
            "schema_version": 1,
            "run_id": run_id,
            "requested_model": model,
            "max_calls": MAX_CALLS,
            "step_ids": list(STEP_IDS),
            "input_hashes": verified["hashes"],
            "implementation_sha256": _implementation_hashes(),
            "style_target_feasibility": verified["feasibility"],
            "plan_brief_sha256": plan_brief_hash,
            "preset_plan_sha256": (
                sha256_file(saved_plan_snapshot) if saved_plan_snapshot is not None else None
            ),
            "preset_response_sha256": {
                step_id: sha256_file(path)
                for step_id, path in sorted(preset_response_snapshots.items())
            },
        }
        store.initialize(spec)
        plan_prompt = build_plan_prompt(
            plan_snapshot.read_text(encoding="utf-8"),
            verified["prompt_target"],
            plan_brief=plan_brief,
        )
        base_hashes = {
            **verified["hashes"],
            "plan_prompt_template": sha256_file(plan_snapshot),
            "material_prompt_template": sha256_file(material_snapshot),
            "schema": sha256_file(schema_snapshot),
            "plan_brief": plan_brief_hash,
        }
        codex_runner = CodexExecRunner(
            run_store=store,
            schema_path=schema_snapshot,
            model=model,
            working_directory=isolated_codex_working_directory(),
        )
        runner: Any = codex_runner
        if saved_plan_snapshot is not None:
            runner = PresetPlanRunner(codex_runner, load_plan_source(saved_plan_snapshot))
        if preset_response_snapshots:
            responses = {
                step_id: _validate_saved_response(step_id, _read_object(path))
                for step_id, path in preset_response_snapshots.items()
            }
            runner = PresetResponseRunner(runner, responses)
        staging = run_dir / "staged"
        try:
            composition = run_staged_generation(
                runner,
                plan_prompt=plan_prompt,
                material_prompt_template=material_snapshot.read_text(encoding="utf-8"),
                output_dir=staging,
                base_input_hashes=base_hashes,
                style_prompt_target=verified["prompt_target"],
                require_inner_structure=True,
                require_naturalness=True,
                require_phrase_structure=True,
                require_voice_structure=True,
            )
        except LongFormGenerationError as error:
            store.record_step(
                "generate-candidate",
                "failed",
                base_hashes,
                {"error": str(error), "error_type": type(error).__name__},
            )
            raise
        store.record_step(
            "generate-candidate",
            "completed",
            base_hashes,
            {"composition_sha256": sha256_file(staging / "final.music.py")},
        )
        staged_midi = staging / "final.mid"
        render_composition(composition, staged_midi)
        material_profile = build_material_development_reference_profile_from_directory(
            source_dir, excluded_names=EXCLUDED_NAMES
        )
        verification = evaluate_long_form_candidate(
            composition,
            staged_midi,
            style_target=verified["target"],
            material_development_profile=material_profile,
            require_naturalness=True,
            require_voice_structure=True,
        )
        atomic_write_json(run_dir / "verification.json", verification)
        evaluation_hashes = {
            "composition": sha256_file(staging / "final.music.py"),
            "smf": sha256_file(staged_midi),
            "style_target": verified["hashes"]["style_target"],
        }
        store.record_step(
            "evaluate-candidate",
            "completed",
            evaluation_hashes,
            {"verification_sha256": sha256_file(run_dir / "verification.json")},
        )
        final_source: Path | None = None
        final_midi: Path | None = None
        if verification["status"] == "pass":
            final_source = store.promote_file(
                staging / "final.music.py",
                run_dir / "final.music.py",
                sha256_file(staging / "final.music.py"),
            )
            final_midi = store.promote_file(
                staged_midi, run_dir / "final.mid", sha256_file(staged_midi)
            )
            store.record_step(
                "publish-final",
                "completed",
                evaluation_hashes,
                {
                    "source": str(final_source),
                    "smf": str(final_midi),
                },
            )
        else:
            store.record_step(
                "publish-final",
                "failed",
                evaluation_hashes,
                {"reason": "required deterministic gate failed"},
            )
        state = store.rebuild_state()
    return {
        "status": verification["status"],
        "run_id": run_id,
        "run_dir": str(run_dir),
        "final_source": str(final_source) if final_source else None,
        "final_midi": str(final_midi) if final_midi else None,
        "call_count": state["calls"]["confirmed_external_call_count"],
        "verification": verification,
        "snapshots": {
            "style_target": str(target_snapshot),
            "manifest": str(manifest_snapshot),
            "reference_records": str(records_snapshot),
            "plan_brief": str(plan_brief_snapshot) if plan_brief_snapshot else None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="参照目標を使って三分のピアノ SMF を生成します。")
    parser.add_argument("--source-dir", type=Path, default=Path(".appendix/source-smf"))
    parser.add_argument(
        "--style-target", type=Path, default=Path(".appendix/style-target/style-target.json")
    )
    parser.add_argument(
        "--style-manifest", type=Path, default=Path(".appendix/style-target/manifest.json")
    )
    parser.add_argument(
        "--reference-records",
        type=Path,
        default=Path(".appendix/reference-structure-analysis/files.jsonl"),
    )
    parser.add_argument(
        "--schema", type=Path, default=Path("schemas/codex-composition-response.schema.json")
    )
    parser.add_argument("--output-root", type=Path, default=Path(".appendix/long-form-runs"))
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--plan-source", type=Path)
    parser.add_argument("--preset-responses", type=Path)
    parser.add_argument("--plan-brief", type=Path)
    args = parser.parse_args(argv)
    result = run_long_form(
        source_dir=args.source_dir,
        target_path=args.style_target,
        manifest_path=args.style_manifest,
        reference_records_path=args.reference_records,
        schema_path=args.schema,
        output_root=args.output_root,
        run_id=args.run_id,
        model=args.model,
        plan_source_path=args.plan_source,
        preset_response_dir=args.preset_responses,
        plan_brief_path=args.plan_brief,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
