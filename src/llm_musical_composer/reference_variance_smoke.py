"""参照差smoke testの外部呼び出し前固定と順次実行を扱う。"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from llm_musical_composer.composition_request import (
    normalize_composition_request,
    resolve_composition_request,
)
from llm_musical_composer.pilot_loop import CodexExecRunner
from llm_musical_composer.reference_generation_target import (
    ReferenceGenerationTarget,
    build_reference_generation_target,
)
from llm_musical_composer.reference_pair_selection import select_reference_pair
from llm_musical_composer.run_state import (
    RunStore,
    StateConflictError,
    atomic_write_json,
    sha256_json,
)
from llm_musical_composer.staged_pipeline_generation import (
    StagedPipelineResult,
    StageRunner,
    run_staged_pipeline_generation,
    staged_pipeline_fingerprints,
)

MODEL_CONFIG = {
    "model": "gpt-5.6-sol",
    "reasoning_effort": "high",
    "timeout_seconds": 600,
}
SMOKE_VERSION = "reference-variance-smoke-v8"
DEFAULT_ARTIFACT_ROOT = Path(f".appendix/{SMOKE_VERSION}")
MAXIMUM_CALLS_PER_RUN = 7
STOP_STAGES = ("piece_plan", "score_spec", "performance_spec")

RunnerFactory = Callable[[Path, Mapping[str, object]], StageRunner]


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StateConflictError(f"cannot read prepared JSON: {path}") from error
    if not isinstance(value, dict):
        raise StateConflictError(f"prepared JSON must be an object: {path}")
    return value


def _write_immutable_json(path: Path, value: dict[str, Any], label: str) -> None:
    if path.is_file():
        if _read_object(path) != value:
            raise StateConflictError(f"saved {label} conflicts with current inputs: {path}")
        return
    atomic_write_json(path, value)


def _request_inputs(
    project_root: Path,
    reference_name: str,
) -> tuple[dict[str, Any], dict[str, Any], ReferenceGenerationTarget]:
    reference_dir = project_root / ".appendix" / "reference-profile-v1"
    control_dir = project_root / ".appendix" / "control-reference-baseline-v3"
    reference_summary = _read_object(reference_dir / "summary.json")
    reference_manifest = _read_object(reference_dir / "manifest.json")
    normalized = normalize_composition_request(
        {
            "schema_version": 1,
            "preset": "solo_piano_3m_v1",
            "reference": reference_name,
            "controls": {},
        }
    )
    resolved = resolve_composition_request(
        normalized,
        reference_summary=reference_summary,
        reference_hashes=reference_manifest["inputs"],
    )
    target = build_reference_generation_target(
        resolved.value,
        reference_dir=reference_dir,
        control_dir=control_dir,
    )
    return normalized.value, resolved.value, target


def prepare_reference_variance_smoke(
    project_root: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    """参照曲対、4run、モデル、入力fingerprintを外部呼び出し前に固定する。"""

    project_root = Path(project_root).resolve()
    artifact_root = Path(artifact_root).resolve()
    selection = select_reference_pair(
        project_root / ".appendix" / "reference-profile-v1",
        project_root / ".appendix" / "control-reference-baseline-v3",
    )
    _write_immutable_json(artifact_root / "selection.json", selection.artifact, "selection")
    runs: list[dict[str, Any]] = []
    for reference_index, reference_name in enumerate(
        selection.artifact["selected"]["names"], start=1
    ):
        reference_label = "a" if reference_index == 1 else "b"
        normalized, resolved, target = _request_inputs(project_root, reference_name)
        for candidate_number in (1, 2):
            runs.append(
                {
                    "run_id": f"reference-{reference_label}-candidate-{candidate_number}",
                    "reference_name": reference_name,
                    "candidate_number": candidate_number,
                    "normalized_request_sha256": sha256_json(normalized),
                    "resolved_request_sha256": sha256_json(resolved),
                    "target_artifact_sha256": target.sha256,
                    "prompt_target_sha256": target.prompt_sha256,
                }
            )
    preflight = {
        "schema_version": 1,
        "smoke_version": SMOKE_VERSION,
        "status": "prepared",
        "selection_sha256": selection.sha256,
        "model_config": MODEL_CONFIG,
        "maximum_calls_per_run": MAXIMUM_CALLS_PER_RUN,
        "maximum_external_call_count": MAXIMUM_CALLS_PER_RUN * len(runs),
        "execution_policy": (
            "execute one explicitly selected candidate through one explicitly selected stage"
        ),
        "implementation_hashes": staged_pipeline_fingerprints(),
        "runs": runs,
    }
    _write_immutable_json(artifact_root / "preflight.json", preflight, "preflight")
    return preflight


def _default_runner_factory(
    run_dir: Path,
    model_config: Mapping[str, object],
) -> StageRunner:
    working_directory = (
        Path(tempfile.gettempdir()) / "llm-musical-composer-reference-variance" / run_dir.name
    )
    return CodexExecRunner(
        run_store=RunStore(run_dir, max_calls=MAXIMUM_CALLS_PER_RUN),
        schema_path=Path(__file__).resolve().parents[2]
        / "schemas"
        / "codex-composition-response.schema.json",
        model=str(model_config["model"]),
        reasoning_effort=str(model_config["reasoning_effort"]),
        working_directory=working_directory,
        timeout_seconds=int(model_config["timeout_seconds"]),
    )


def execute_prepared_smoke(
    project_root: Path,
    artifact_root: Path,
    *,
    run_id: str,
    stop_after: str,
    runner_factory: RunnerFactory = _default_runner_factory,
) -> StagedPipelineResult:
    """固定済みの1runを、指定した1段階まで実行する。"""

    project_root = Path(project_root).resolve()
    artifact_root = Path(artifact_root).resolve()
    expected = prepare_reference_variance_smoke(project_root, artifact_root)
    preflight = _read_object(artifact_root / "preflight.json")
    if preflight != expected:
        raise StateConflictError("saved preflight conflicts with current inputs")
    if stop_after not in STOP_STAGES:
        raise ValueError(f"unknown stop stage: {stop_after}")
    selected_runs = [run for run in preflight["runs"] if run["run_id"] == run_id]
    if len(selected_runs) != 1:
        raise ValueError(f"unknown prepared run: {run_id}")
    run = selected_runs[0]
    model_config = preflight["model_config"]
    normalized, resolved, target = _request_inputs(project_root, run["reference_name"])
    current_hashes = {
        "normalized_request_sha256": sha256_json(normalized),
        "resolved_request_sha256": sha256_json(resolved),
        "target_artifact_sha256": target.sha256,
        "prompt_target_sha256": target.prompt_sha256,
    }
    if any(run[key] != value for key, value in current_hashes.items()):
        raise StateConflictError(f"prepared run inputs changed: {run['run_id']}")
    run_dir = artifact_root / "runs" / run["run_id"]
    runner = runner_factory(run_dir, model_config)
    return run_staged_pipeline_generation(
        run_dir=run_dir,
        normalized_request=normalized,
        resolved_request=resolved,
        prompt_target=target.prompt_target,
        model_config=model_config,
        runner=runner,
        stop_after=stop_after,
    )


def main() -> int:
    """固定済みsmoke testを準備または実行する。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "execute"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
    )
    parser.add_argument("--run-id")
    parser.add_argument("--stop-after", choices=STOP_STAGES)
    arguments = parser.parse_args()
    if arguments.action == "prepare":
        result: object = prepare_reference_variance_smoke(
            arguments.project_root, arguments.artifact_root
        )
    else:
        if arguments.run_id is None or arguments.stop_after is None:
            parser.error("execute requires --run-id and --stop-after")
        item = execute_prepared_smoke(
            arguments.project_root,
            arguments.artifact_root,
            run_id=arguments.run_id,
            stop_after=arguments.stop_after,
        )
        result = {
            "status": item.status,
            "stopped_after": item.stopped_after,
            "smf_path": str(item.smf_path) if item.smf_path is not None else None,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI入口
    raise SystemExit(main())
