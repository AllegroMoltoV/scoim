"""2設計族の段階素材パイロットを準備・実行するCLI。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any, Protocol

from llm_musical_composer.generic_pipeline_quality import (
    evaluate_generic_pipeline_quality,
    evaluate_generic_score_quality,
)
from llm_musical_composer.performance_pipeline import (
    render_performance,
    render_performance_smf,
)
from llm_musical_composer.pilot_loop import (
    CodexExecRunner,
    isolated_codex_working_directory,
)
from llm_musical_composer.pipeline_dsl import (
    dump_score_spec,
    parse_performance_spec,
    parse_piece_plan,
    parse_score_spec,
)
from llm_musical_composer.recurrence_quality import analyze_foreground_dissonance
from llm_musical_composer.reference_timing_run import (
    KnownTimingSource,
    _calibration_lineage_status,
    _completed_run_evidence_status,
    default_known_timing_sources,
)
from llm_musical_composer.run_state import (
    RunStore,
    atomic_write_bytes,
    atomic_write_json,
    sha256_file,
    sha256_text,
)
from llm_musical_composer.staged_material_pilot import (
    assemble_harmonies,
    assemble_melody,
    assemble_texture,
    build_fixed_context,
    build_texture_feasibility,
    dump_harmonic_draft,
    dump_melody_draft,
    dump_texture_draft,
    full_low_spacing_violations,
    parse_harmonic_draft,
    parse_melody_draft,
    parse_texture_draft,
    texture_feasibility_violations,
    texture_onset_capacity_violations,
)

MODEL_ID = "gpt-5.6-sol"
REASONING_EFFORT = "high"
PROTOCOL_ID = "staged-material-pilot-v3"


@dataclass(frozen=True)
class PilotCase:
    case_id: str
    source_case_id: str
    target_material_id: str


CASES = (
    PilotCase("multiscale-v8-b-return", "multiscale-v8", "b-return"),
    PilotCase("reference-v7-a2-material-7", "reference-variance-v7-a2", "material_7"),
)


class PilotRunner(Protocol):
    @property
    def call_number(self) -> int: ...

    def run(
        self, step_id: str, prompt: str, input_hashes: dict[str, str] | None = None
    ) -> dict[str, object]: ...


def _file_record(workspace: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.resolve().relative_to(workspace.resolve()).as_posix(),
        "sha256": sha256_file(path),
    }


def _source_evidence(source: KnownTimingSource) -> tuple[str, tuple[str, str, str]]:
    plan = parse_piece_plan(source.piece_path.read_text(encoding="utf-8"))
    score = parse_score_spec(source.score_path.read_text(encoding="utf-8"))
    performance = parse_performance_spec(source.performance_path.read_text(encoding="utf-8"))
    rendered = render_performance(plan, score, performance)
    if source.evidence_kind == "calibration_result":
        status = _calibration_lineage_status(source.evidence_path, rendered.lineage)
    else:
        status = _completed_run_evidence_status(source)
    return status, rendered.lineage


def _render(template: str, context: dict[str, Any], placeholder: str) -> str:
    if template.count(placeholder) != 1:
        raise ValueError(f"prompt must contain {placeholder} exactly once")
    value = json.dumps(context, ensure_ascii=False, indent=2, sort_keys=True)
    return template.replace(placeholder, value)


def prepare_staged_material_pilot(workspace: Path, output_root: Path) -> dict[str, Any]:
    """固定2ケースの証跡、入力、和声promptを外部呼出し前に保存する。"""

    workspace = Path(workspace).resolve()
    output_root = Path(output_root).resolve()
    available = {source.case_id: source for source in default_known_timing_sources(workspace)}
    template_paths = {
        "harmony": workspace / "prompts/pipeline-harmonic-material-spec.md",
        "melody": workspace / "prompts/pipeline-melody-spec.md",
        "texture": workspace / "prompts/pipeline-piano-texture-spec-v2-thin.md",
    }
    case_results: list[dict[str, Any]] = []
    for case in CASES:
        source = available[case.source_case_id]
        evidence_status, lineage = _source_evidence(source)
        case_dir = output_root / case.case_id
        if evidence_status != "pass":
            case_results.append(
                {
                    "case_id": case.case_id,
                    "status": "unable_to_investigate",
                    "reason": "source evidence did not pass",
                }
            )
            continue
        plan_source = source.piece_path.read_text(encoding="utf-8")
        score_source = source.score_path.read_text(encoding="utf-8")
        plan = parse_piece_plan(plan_source)
        score = parse_score_spec(score_source)
        context = build_fixed_context(case.case_id, plan, score, case.target_material_id)
        prompt = _render(
            template_paths["harmony"].read_text(encoding="utf-8"),
            context,
            "{{FIXED_CONTEXT_JSON}}",
        )
        run_spec = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "case_id": case.case_id,
            "source_case_id": case.source_case_id,
            "target_material_id": case.target_material_id,
            "model": MODEL_ID,
            "reasoning_effort": REASONING_EFFORT,
            "max_external_calls": 3,
            "source_evidence_kind": source.evidence_kind,
            "source_evidence_status": evidence_status,
            "source_lineage": list(lineage),
            "inputs": {
                "piece_plan": _file_record(workspace, source.piece_path),
                "score_spec": _file_record(workspace, source.score_path),
                "performance_spec": _file_record(workspace, source.performance_path),
                "smf": _file_record(workspace, source.smf_path),
                "evidence": _file_record(workspace, source.evidence_path),
                "implementation": _file_record(
                    workspace,
                    workspace / "src/llm_musical_composer/staged_material_pilot.py",
                ),
                "placement_implementation": _file_record(
                    workspace,
                    workspace
                    / "src/llm_musical_composer/piano_texture_register_placement.py",
                ),
                "performance_contract": _file_record(
                    workspace,
                    workspace / "src/llm_musical_composer/performance_pipeline.py",
                ),
                "runner": _file_record(workspace, Path(__file__)),
                "prompts": {
                    name: _file_record(workspace, path) for name, path in template_paths.items()
                },
            },
        }
        store = RunStore(case_dir, max_calls=3)
        store.initialize(run_spec)
        store.snapshot_file("input-piece-plan.dsl", source.piece_path)
        store.snapshot_file("input-score-spec.dsl", source.score_path)
        store.snapshot_file("input-performance-spec.dsl", source.performance_path)
        store.snapshot_json("fixed-context.json", context)
        store.promote_bytes(
            prompt.encode("utf-8"),
            case_dir / "prompt-harmony.txt",
            sha256_text(prompt),
        )
        manifest = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "case_id": case.case_id,
            "target_material_id": case.target_material_id,
            "status": "prepared",
            "passes": False,
            "confirmed_external_call_count": 0,
            "field_provenance": {
                "runner_fixed": [
                    "material_id",
                    "length_units",
                    "derived_from",
                    "directions",
                    "event_ids",
                    "harmony_ids",
                    "accompaniment_role",
                    "accompaniment_voice",
                ],
                "externally_generated": [
                    "harmonies",
                    "foreground_voice",
                    "melody_events",
                    "texture_events",
                ],
            },
        }
        atomic_write_json(case_dir / "manifest.json", manifest)
        case_results.append({"case_id": case.case_id, "status": "prepared"})
    prepared_count = sum(case["status"] == "prepared" for case in case_results)
    result = {
        "schema_version": 1,
        "status": "prepared" if prepared_count == len(CASES) else "unable_to_investigate",
        "case_count": len(CASES),
        "prepared_case_count": prepared_count,
        "maximum_external_call_count": prepared_count * 3,
        "cases": case_results,
    }
    atomic_write_json(output_root / "result.json", result)
    return result


def _response_source(response: dict[str, object], stage: str) -> str:
    value = response.get("composition_source")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{stage} response has no composition_source")
    return value


def _save_text(path: Path, value: str) -> None:
    content = value.encode("utf-8")
    atomic_write_bytes(path, content)


def execute_prepared_case(case_dir: Path, runner: PilotRunner) -> dict[str, Any]:
    """準備済み1ケースを三段階で実行し、修復せず結果を確定する。"""

    case_dir = Path(case_dir).resolve()
    manifest_path = case_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "prepared":
        return manifest
    plan = parse_piece_plan((case_dir / "input-piece-plan.dsl").read_text(encoding="utf-8"))
    score = parse_score_spec((case_dir / "input-score-spec.dsl").read_text(encoding="utf-8"))
    performance = parse_performance_spec(
        (case_dir / "input-performance-spec.dsl").read_text(encoding="utf-8")
    )
    context = json.loads((case_dir / "fixed-context.json").read_text(encoding="utf-8"))
    case_id = str(manifest["case_id"])
    material_id = str(manifest["target_material_id"])
    target = next(material for material in score.materials if material.material_id == material_id)
    baseline_rendered = render_performance(plan, score, performance)
    baseline_score_quality = evaluate_generic_score_quality(plan, score)
    baseline_quality = evaluate_generic_pipeline_quality(
        plan, score, performance, baseline_rendered
    )
    templates = Path(__file__).resolve().parents[2] / "prompts"
    try:
        harmony_prompt = (case_dir / "prompt-harmony.txt").read_text(encoding="utf-8")
        harmony_response = _response_source(
            runner.run("harmony", harmony_prompt, {"prompt": sha256_text(harmony_prompt)}),
            "harmony",
        )
        harmony_draft = parse_harmonic_draft(harmony_response)
        harmonies = assemble_harmonies(case_id, target, harmony_draft, score)
        harmony_source = dump_harmonic_draft(harmony_draft)
        _save_text(case_dir / "response-harmony.txt", harmony_response)
        _save_text(case_dir / "harmonic-draft.dsl", harmony_source)

        melody_context = {
            **context,
            "confirmed_harmonies": [asdict(item) for item in harmonies],
        }
        melody_prompt = _render(
            (templates / "pipeline-melody-spec.md").read_text(encoding="utf-8"),
            melody_context,
            "{{STAGE_CONTEXT_JSON}}",
        )
        _save_text(case_dir / "prompt-melody.txt", melody_prompt)
        melody_response = _response_source(
            runner.run("melody", melody_prompt, {"prompt": sha256_text(melody_prompt)}),
            "melody",
        )
        melody_draft = parse_melody_draft(melody_response)
        melody = assemble_melody(case_id, score.divisions, target, harmonies, melody_draft, score)
        melody_source = dump_melody_draft(melody_draft)
        _save_text(case_dir / "response-melody.txt", melody_response)
        _save_text(case_dir / "melody-draft.dsl", melody_source)

        provisional = dataclass_replace(
            target,
            notes=melody,
            harmonies=harmonies,
            foreground_voice=melody_draft.foreground_voice,
        )
        provisional_score = dataclass_replace(
            score,
            materials=tuple(
                provisional if material.material_id == material_id else material
                for material in score.materials
            ),
        )
        dissonance = analyze_foreground_dissonance(provisional_score, material_id)
        if not dissonance.passes:
            raise ValueError("melody has unsupported foreground dissonance")

        texture_context = {
            **melody_context,
            "confirmed_foreground_voice": melody_draft.foreground_voice,
            "confirmed_melody": [asdict(item) for item in melody],
            "texture_feasibility": build_texture_feasibility(harmonies, melody),
        }
        texture_prompt = _render(
            (templates / "pipeline-piano-texture-spec-v2-thin.md").read_text(encoding="utf-8"),
            texture_context,
            "{{STAGE_CONTEXT_JSON}}",
        )
        _save_text(case_dir / "prompt-texture.txt", texture_prompt)
        texture_response = _response_source(
            runner.run("texture", texture_prompt, {"prompt": sha256_text(texture_prompt)}),
            "texture",
        )
        texture_draft = parse_texture_draft(texture_response)
        _save_text(case_dir / "response-texture.txt", texture_response)
        _save_text(case_dir / "texture-draft.dsl", dump_texture_draft(texture_draft))
        feasibility = texture_context["texture_feasibility"]
        advisory_violations = texture_feasibility_violations(
            texture_draft, feasibility
        )
        onset_violations = texture_onset_capacity_violations(
            texture_draft, feasibility
        )
        atomic_write_json(
            case_dir / "texture-feasibility.json",
            {
                "schema_version": 2,
                "status": (
                    "onset_capacity_exceeded"
                    if onset_violations
                    else "declared_zone_infeasible"
                    if advisory_violations
                    else "passed"
                ),
                "contract": feasibility,
                "violations": advisory_violations,
                "advisory_violations": advisory_violations,
                "onset_capacity_violations": onset_violations,
            },
        )
        if advisory_violations:
            raise ValueError("texture has declared_zone_infeasible events")
        if onset_violations:
            raise ValueError("texture exceeds onset capacity")
        placement, material = assemble_texture(
            case_id, plan, score, target, harmonies, melody, texture_draft
        )
        placement_record = {
            "schema_version": 1,
            "status": placement.status,
            "failed_onset": placement.failed_onset,
            "reason": placement.reason,
            "accompaniment_low_spacing_violations": placement.low_spacing_violations,
            "combined_low_spacing_violations": None,
        }
        atomic_write_json(case_dir / "texture-placement.json", placement_record)
        if placement.status != "placed":
            raise ValueError(f"texture placement failed: {placement.status}")
        combined_low_spacing = full_low_spacing_violations(material)
        placement_record["combined_low_spacing_violations"] = combined_low_spacing
        atomic_write_json(case_dir / "texture-placement.json", placement_record)
        if combined_low_spacing:
            raise ValueError("combined material has low spacing violations")

        combined = dataclass_replace(
            score,
            materials=tuple(
                material if item.material_id == material_id else item for item in score.materials
            ),
        )
        after_score_quality = evaluate_generic_score_quality(plan, combined)
        new_score_failures = sorted(
            set(after_score_quality["failures"]) - set(baseline_score_quality["failures"])
        )
        rendered = render_performance(plan, combined, performance)
        after_quality = evaluate_generic_pipeline_quality(plan, combined, performance, rendered)
        new_pipeline_failures = sorted(
            set(after_quality["failures"]) - set(baseline_quality["failures"])
        )
        if new_score_failures or new_pipeline_failures:
            raise ValueError(
                "new quality failures: "
                f"score={new_score_failures}, pipeline={new_pipeline_failures}"
            )
        _save_text(case_dir / "combined-score-spec.dsl", dump_score_spec(combined))
        atomic_write_json(
            case_dir / "score-quality.json",
            {"baseline": baseline_score_quality, "after": after_score_quality},
        )
        atomic_write_json(
            case_dir / "performance-quality.json",
            {"baseline": baseline_quality, "after": after_quality},
        )
        smf_result = render_performance_smf(rendered, case_dir / "outputs/final.mid")
        manifest.update(
            {
                "status": "completed",
                "passes": True,
                "confirmed_external_call_count": runner.call_number,
                "smf_sha256": sha256_file(smf_result.path),
            }
        )
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "passes": False,
                "confirmed_external_call_count": runner.call_number,
                "error": {"type": type(error).__name__, "detail": str(error)},
            }
        )
    atomic_write_json(manifest_path, manifest)
    return manifest


def create_default_pilot_runner(workspace: Path, case_dir: Path) -> PilotRunner:
    """成果物rootから分離した既存のCodex作業領域を使う。"""

    return CodexExecRunner(
        run_store=RunStore(case_dir, max_calls=3),
        schema_path=Path(workspace).resolve()
        / "schemas/codex-composition-response.schema.json",
        model=MODEL_ID,
        reasoning_effort=REASONING_EFFORT,
        working_directory=isolated_codex_working_directory(),
        timeout_seconds=900,
    )


def execute_prepared_pilot(
    workspace: Path,
    output_root: Path,
    runner_factory: Callable[[Path], PilotRunner] | None = None,
) -> dict[str, Any]:
    """準備済み2ケースを独立実行し、親manifestへ集約する。"""

    workspace = Path(workspace).resolve()
    output_root = Path(output_root).resolve()

    factory = runner_factory or (
        lambda case_dir: create_default_pilot_runner(workspace, case_dir)
    )
    cases: list[dict[str, Any]] = []
    for case in CASES:
        case_dir = output_root / case.case_id
        if not (case_dir / "manifest.json").is_file():
            cases.append(
                {
                    "case_id": case.case_id,
                    "status": "unable_to_investigate",
                    "passes": False,
                    "confirmed_external_call_count": 0,
                    "error": {
                        "type": "FileNotFoundError",
                        "detail": "prepared case manifest is missing",
                    },
                }
            )
            continue
        run_spec = json.loads((case_dir / "run-spec.json").read_text(encoding="utf-8"))
        input_records = run_spec.get("inputs", {})
        records = [
            value
            for value in input_records.values()
            if isinstance(value, dict) and "path" in value
        ]
        prompt_records = input_records.get("prompts", {})
        if isinstance(prompt_records, dict):
            records.extend(
                value
                for value in prompt_records.values()
                if isinstance(value, dict) and "path" in value
            )
        drifted = [
            str(record["path"])
            for record in records
            if not (workspace / str(record["path"])).is_file()
            or sha256_file(workspace / str(record["path"])) != record.get("sha256")
        ]
        if drifted:
            cases.append(
                {
                    "case_id": case.case_id,
                    "status": "unable_to_investigate",
                    "passes": False,
                    "confirmed_external_call_count": 0,
                    "error": {
                        "type": "InputDriftError",
                        "detail": f"prepared inputs changed: {sorted(drifted)}",
                    },
                }
            )
            continue
        try:
            runner = factory(case_dir)
        except Exception as error:
            failed = json.loads(
                (case_dir / "manifest.json").read_text(encoding="utf-8")
            )
            failed.update(
                {
                    "status": "unable_to_investigate",
                    "passes": False,
                    "confirmed_external_call_count": 0,
                    "error": {"type": type(error).__name__, "detail": str(error)},
                }
            )
            atomic_write_json(case_dir / "manifest.json", failed)
            cases.append(failed)
            continue
        cases.append(execute_prepared_case(case_dir, runner))
    passes = len(cases) == len(CASES) and all(item.get("passes") is True for item in cases)
    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "completed" if passes else "failed",
        "passes": passes,
        "confirmed_external_call_count": sum(
            int(item.get("confirmed_external_call_count", 0)) for item in cases
        ),
        "cases": cases,
    }
    atomic_write_json(output_root / "result.json", result)
    atomic_write_json(output_root / "manifest.json", result)
    return result


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    if arguments.command == "prepare":
        result = prepare_staged_material_pilot(arguments.workspace, arguments.output_root)
        success = result["status"] == "prepared"
    else:
        result = execute_prepared_pilot(arguments.workspace, arguments.output_root)
        success = result["passes"] is True
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
