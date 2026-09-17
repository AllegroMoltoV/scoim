"""Codex を使った最小の候補生成、評価、局所修正ループ。"""

from __future__ import annotations

import argparse
import enum
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from llm_musical_composer.activity_selection import (
    build_activity_reference_profile_from_directory,
    extract_activity_features,
)
from llm_musical_composer.composition_ir import Composition
from llm_musical_composer.material_development import (
    build_material_development_reference_profile_from_directory,
    evaluate_material_development,
)
from llm_musical_composer.music_dsl import DslError, parse_composition
from llm_musical_composer.pilot_features import (
    FeatureStatus,
    NoteEvent,
    build_reference_profile_from_directory,
    evaluate_features,
    extract_smf_features,
    run_minimal_controls,
)
from llm_musical_composer.run_state import (
    InterruptedAttemptError,
    RunLock,
    RunStore,
    StateConflictError,
    atomic_write_bytes,
    atomic_write_json,
    event_summary,
    sha256_bytes,
    sha256_file,
    sha256_json,
    sha256_text,
)
from llm_musical_composer.section_contrast import evaluate_section_contrast
from llm_musical_composer.smf_notes import load_smf_notes
from llm_musical_composer.smf_render import (
    ENDING_BREATH_MS,
    render_composition,
    tonic_ending_pitches,
)
from llm_musical_composer.sustain_profile import (
    SustainContractError,
    SustainRepairScopeError,
    evaluate_sustain_profile,
    validate_sustain_repair_scope,
)

MODEL_ID = "gpt-5.6-sol"
MAX_CALLS = 7
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVISION_AXES = frozenset({"pitch_order", "relative_timing", "texture"})
PROMPT_NAMES = ("pilot-compose.md", "pilot-repair.md", "pilot-revise.md")
STEP_IDS = (
    "compose-candidate-1",
    "repair-candidate-1",
    "compose-candidate-2",
    "repair-candidate-2",
    "compose-candidate-3",
    "repair-candidate-3",
    "evaluate-candidates",
    "revise-selected",
    "publish-final",
)


class CodexProcessRecoveryError(RuntimeError):
    """タイムアウトしたCodexプロセスを終了・回収できなかった。"""


def _resolve_codex_executable() -> tuple[Path, Path | None]:
    codex_command = shutil.which("codex")
    if codex_command is None:
        raise FileNotFoundError("codex command was not found on PATH")
    if sys.platform != "win32":
        return Path(codex_command), None

    configured_root = os.environ.get("CODEX_MANAGED_PACKAGE_ROOT")
    package_root = (
        Path(configured_root)
        if configured_root
        else Path(codex_command).resolve().parent / "node_modules" / "@openai" / "codex"
    )
    executable = (
        package_root
        / "node_modules"
        / "@openai"
        / "codex-win32-x64"
        / "vendor"
        / "x86_64-pc-windows-msvc"
        / "bin"
        / "codex.exe"
    ).resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"managed native codex executable was not found: {executable}")
    return executable, package_root.resolve()


def _run_codex_process(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    prompt = kwargs.pop("input")
    timeout = kwargs.pop("timeout")
    process = subprocess.Popen(command, stdin=subprocess.PIPE, **kwargs)
    try:
        process.communicate(input=prompt, timeout=timeout)
    except subprocess.TimeoutExpired as timeout_error:
        try:
            process.kill()
            process.communicate()
        except Exception as recovery_error:
            raise CodexProcessRecoveryError(
                f"timed out Codex process could not be terminated and reaped: "
                f"{type(recovery_error).__name__}: {recovery_error}"
            ) from recovery_error
        raise timeout_error
    return subprocess.CompletedProcess(command, process.returncode)


def isolated_codex_working_directory(temp_root: Path | None = None) -> Path:
    root = Path(tempfile.gettempdir()) if temp_root is None else Path(temp_root)
    path = (root / "llm-musical-composer-codex-compose").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


class Runner(Protocol):
    def run(
        self, step_id: str, prompt: str, input_hashes: dict[str, str] | None = None
    ) -> dict[str, object]: ...


class RunStatus(enum.Enum):
    COMPLETED = "completed"
    NO_VALID_CANDIDATE = "no_valid_candidate"
    REVISION_FAILED = "revision_failed"


@dataclass
class Candidate:
    candidate_id: str
    source: str
    midi_path: Path | None
    evaluation: dict[str, Any] | None
    error: str | None
    repaired: bool


@dataclass
class PilotResult:
    status: RunStatus
    call_count: int
    candidates: list[Candidate]
    selected_candidate: str | None
    final_source_path: Path | None
    final_midi_path: Path | None
    revision_status: str | None


class CodexExecRunner:
    def __init__(
        self,
        *,
        run_store: RunStore,
        schema_path: Path,
        model: str = MODEL_ID,
        working_directory: Path | None = None,
        reasoning_effort: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        self.run_store = run_store
        self.schema_path = Path(schema_path).resolve()
        self.model = model
        if reasoning_effort not in {None, "none", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("unsupported Codex reasoning effort")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("Codex timeout must be positive")
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self.working_directory = (
            Path(working_directory).resolve() if working_directory is not None else None
        )
        if self.working_directory is not None:
            self.working_directory.mkdir(parents=True, exist_ok=True)

    @property
    def call_number(self) -> int:
        return self.run_store.call_attempt_count

    def request_for(
        self, step_id: str, prompt: str, input_hashes: dict[str, str]
    ) -> dict[str, object]:
        request = {
            "schema_version": 1,
            "step_id": step_id,
            "requested_model": self.model,
            "prompt_sha256": sha256_text(prompt),
            "schema_sha256": sha256_file(self.schema_path),
            "input_hashes": dict(sorted(input_hashes.items())),
        }
        if self.reasoning_effort is not None:
            request["reasoning_effort"] = self.reasoning_effort
        if self.timeout_seconds is not None:
            request["timeout_seconds"] = self.timeout_seconds
        return request

    @staticmethod
    def _validated_response_bytes(content: bytes) -> dict[str, object]:
        try:
            value = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid structured response: {error}") from error
        if not isinstance(value, dict) or set(value) != {"composition_source", "intent_summary"}:
            raise ValueError("response does not match the required object properties")
        source = value.get("composition_source")
        summary = value.get("intent_summary")
        if not isinstance(source, str) or not source:
            raise ValueError("response has no non-empty string composition_source")
        if not isinstance(summary, str) or len(summary) > 300:
            raise ValueError("response has an invalid intent_summary")
        return value

    @staticmethod
    def _validated_response(path: Path) -> dict[str, object]:
        try:
            content = path.read_bytes()
        except OSError as error:
            raise ValueError(f"invalid structured response: {error}") from error
        return CodexExecRunner._validated_response_bytes(content)

    def _promote_validated_response(
        self,
        step_id: str,
        attempt_dir: Path,
        *,
        returncode: int | None,
    ) -> tuple[dict[str, object], str]:
        response_path = attempt_dir / "response.staged.json"
        try:
            response_content = response_path.read_bytes()
            response = self._validated_response_bytes(response_content)
            response_hash = sha256_bytes(response_content)
            self.run_store.promote_bytes(
                response_content,
                self.run_store.run_dir / "responses" / f"{step_id}.json",
                response_hash,
            )
        except (OSError, ValueError, StateConflictError) as error:
            self.run_store.finalize_attempt(
                attempt_dir,
                "failed",
                returncode=returncode,
                detail=f"invalid or uncommitted structured response: {error}",
            )
            raise
        return response, response_hash

    def _commit_completed_attempt(self, step_id: str, attempt_dir: Path) -> dict[str, object]:
        terminal_path = attempt_dir / "terminal.json"
        if terminal_path.is_file():
            terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
            if terminal.get("status") != "completed":
                raise StateConflictError(f"attempt is not completed: {attempt_dir}")
            response_path = attempt_dir / "response.staged.json"
            response_content = response_path.read_bytes()
            response = self._validated_response_bytes(response_content)
            response_hash = sha256_bytes(response_content)
            if terminal.get("response_sha256") != response_hash:
                raise StateConflictError(f"response fingerprint changed: {attempt_dir}")
            self.run_store.promote_bytes(
                response_content,
                self.run_store.run_dir / "responses" / f"{step_id}.json",
                response_hash,
            )
        else:
            response, response_hash = self._promote_validated_response(
                step_id, attempt_dir, returncode=0
            )
            self.run_store.finalize_attempt(
                attempt_dir,
                "completed",
                returncode=0,
                response_sha256=response_hash,
                detail="recovered from persisted turn.completed event",
            )
        return response

    def _recover_existing(
        self, step_id: str, request: dict[str, object]
    ) -> dict[str, object] | None:
        records = list(self.run_store.iter_attempt_records(step_id))
        if not records:
            return None
        if len(records) != 1:
            raise StateConflictError(f"multiple attempts require manual review: {step_id}")
        attempt_dir, saved_request = records[0]
        if saved_request != request:
            raise StateConflictError(f"saved request conflicts with current input: {step_id}")
        recovery_path = attempt_dir / "recovery.json"
        if recovery_path.is_file():
            raise InterruptedAttemptError(
                f"Codex process recovery is required before retrying: {step_id}"
            )
        terminal_path = attempt_dir / "terminal.json"
        if terminal_path.is_file():
            terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
            status = terminal.get("status")
            if status == "completed":
                response = self._commit_completed_attempt(step_id, attempt_dir)
                self.run_store.record_reuse(step_id, attempt_dir)
                return response
            if status == "interrupted":
                raise InterruptedAttemptError(f"attempt outcome is ambiguous: {step_id}")
            raise RuntimeError(f"saved Codex attempt failed: {step_id}")
        summary = event_summary(attempt_dir / "stdout.jsonl")
        if (
            summary["thread_started"]
            and summary["turn_completed"]
            and not summary["turn_failed"]
            and not summary["parse_errors"]
        ):
            return self._commit_completed_attempt(step_id, attempt_dir)
        if summary["turn_failed"]:
            self.run_store.finalize_attempt(
                attempt_dir, "failed", returncode=None, detail="persisted turn.failed event"
            )
            raise RuntimeError(f"saved Codex attempt failed: {step_id}")
        self.run_store.finalize_attempt(
            attempt_dir,
            "interrupted",
            returncode=None,
            detail="no terminal Codex event was persisted",
        )
        raise InterruptedAttemptError(f"attempt outcome is ambiguous: {step_id}")

    def run(
        self, step_id: str, prompt: str, input_hashes: dict[str, str] | None = None
    ) -> dict[str, object]:
        request = self.request_for(step_id, prompt, input_hashes or {})
        recovered = self._recover_existing(step_id, request)
        if recovered is not None:
            return recovered
        attempt_dir = self.run_store.reserve_attempt(step_id, request, prompt)
        response_path = attempt_dir / "response.staged.json"
        try:
            codex_executable, managed_package_root = _resolve_codex_executable()
        except FileNotFoundError as error:
            self.run_store.finalize_attempt(
                attempt_dir, "failed", returncode=None, detail=str(error)
            )
            raise
        environment = os.environ.copy()
        if managed_package_root is not None:
            environment["CODEX_MANAGED_PACKAGE_ROOT"] = str(managed_package_root)
            environment["CODEX_MANAGED_BY_NPM"] = "1"
        command = [
            str(codex_executable),
            "exec",
            "--model",
            self.model,
            "--sandbox",
            "read-only",
            "--ephemeral",
            "--ignore-user-config",
            "--skip-git-repo-check",
        ]
        if self.reasoning_effort is not None:
            command.extend(["-c", f'model_reasoning_effort="{self.reasoning_effort}"'])
        if self.working_directory is not None:
            command.extend(["-C", str(self.working_directory)])
        command.extend(
            [
                "--json",
                "--output-schema",
                str(self.schema_path),
                "-o",
                str(response_path),
                "-",
            ]
        )
        try:
            with (
                (attempt_dir / "stdout.jsonl").open("w", encoding="utf-8") as stdout_file,
                (attempt_dir / "stderr.log").open("w", encoding="utf-8") as stderr_file,
            ):
                completed = _run_codex_process(
                    command,
                    input=prompt,
                    text=True,
                    encoding="utf-8",
                    stdout=stdout_file,
                    stderr=stderr_file,
                    cwd=self.working_directory,
                    timeout=self.timeout_seconds,
                    env=environment,
                )
        except CodexProcessRecoveryError as error:
            self.run_store.record_recovery_failure(attempt_dir, detail=str(error))
            raise InterruptedAttemptError(
                f"Codex process recovery is required before retrying: {step_id}"
            ) from error
        except subprocess.TimeoutExpired as error:
            self.run_store.finalize_attempt(
                attempt_dir,
                "interrupted",
                returncode=None,
                detail=f"codex exec timed out after {self.timeout_seconds} seconds",
            )
            raise InterruptedAttemptError(
                f"codex exec timed out after {self.timeout_seconds} seconds: {step_id}"
            ) from error
        except OSError as error:
            self.run_store.finalize_attempt(
                attempt_dir,
                "failed",
                returncode=None,
                detail=f"Codex process could not start: {type(error).__name__}: {error}",
            )
            raise
        if completed.returncode != 0:
            self.run_store.finalize_attempt(
                attempt_dir,
                "failed",
                returncode=completed.returncode,
                detail="codex exec returned a non-zero exit code",
            )
            raise RuntimeError(f"codex exec failed; see {attempt_dir / 'stderr.log'}")
        summary = event_summary(attempt_dir / "stdout.jsonl")
        if (
            not summary["thread_started"]
            or not summary["turn_completed"]
            or summary["turn_failed"]
            or summary["parse_errors"]
        ):
            self.run_store.finalize_attempt(
                attempt_dir,
                "failed",
                returncode=completed.returncode,
                detail="Codex JSONL has no unambiguous turn.completed event",
            )
            raise RuntimeError(f"codex exec returned no valid completion; see {attempt_dir}")
        response, response_hash = self._promote_validated_response(
            step_id, attempt_dir, returncode=completed.returncode
        )
        self.run_store.finalize_attempt(
            attempt_dir,
            "completed",
            returncode=completed.returncode,
            response_sha256=response_hash,
            detail=(
                f"completed with {summary['error_event_count']} warning error events"
                if summary["error_event_count"]
                else None
            ),
        )
        return response


def _write_json(path: Path, value: object) -> None:
    atomic_write_json(path, value)


def _material_map(composition: Composition) -> dict[str, object]:
    return {material.material_id: material for material in composition.materials}


def validate_revision_scope(
    before: Composition, after: Composition, target_material: str
) -> list[str]:
    issues: list[str] = []
    if before.title != after.title:
        issues.append("title changed")
    if before.form != after.form:
        issues.append("form changed")
    if before.tonal_center != after.tonal_center:
        issues.append("tonal_center changed")
    if before.mode != after.mode:
        issues.append("mode changed")
    if before.ending != after.ending:
        issues.append("ending changed")
    before_materials = _material_map(before)
    after_materials = _material_map(after)
    if set(before_materials) != set(after_materials):
        issues.append("material set changed")
        return issues
    for material_id in before_materials:
        if (
            material_id != target_material
            and before_materials[material_id] != after_materials[material_id]
        ):
            issues.append(f"material {material_id} changed outside target")
    if before_materials.get(target_material) == after_materials.get(target_material):
        issues.append("target material did not change")
    return issues


def _extreme_material_count(evaluation: dict[str, Any]) -> int:
    report = evaluation.get("material_development")
    return int(report["extreme_material_count"]) if report is not None else 0


def _candidate_sort_key(candidate: Candidate) -> tuple[int, float, float, str]:
    assert candidate.evaluation is not None
    return (
        _extreme_material_count(candidate.evaluation),
        -candidate.evaluation["inside_axis_count"],
        candidate.evaluation["worst_axis_distance"],
        candidate.candidate_id,
    )


def _evaluation_sort_key(evaluation: dict[str, Any]) -> tuple[int, float, float]:
    return (
        _extreme_material_count(evaluation),
        -evaluation["inside_axis_count"],
        evaluation["worst_axis_distance"],
    )


def _revision_improves(before: dict[str, Any], after: dict[str, Any], target_axis: str) -> bool:
    if target_axis == "material_development":
        return _extreme_material_count(after) < _extreme_material_count(
            before
        ) and _evaluation_sort_key(after) < _evaluation_sort_key(before)
    before_target = before["axes"][target_axis]["normalized_distance"]
    after_target = after["axes"][target_axis]["normalized_distance"]
    return after_target < before_target and _evaluation_sort_key(after) <= _evaluation_sort_key(
        before
    )


def _evaluate_source(
    candidate_id: str,
    source: str,
    output_dir: Path,
    profile: dict[str, Any],
    *,
    material_development_profile: dict[str, Any] | None = None,
    repaired: bool,
    sustain_repair_baseline: Composition | None = None,
) -> Candidate:
    source_path = output_dir / f"{candidate_id}.music.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        composition = parse_composition(source, require_section_contract=True)
        if sustain_repair_baseline is not None:
            scope_issues = validate_sustain_repair_scope(sustain_repair_baseline, composition)
            if scope_issues:
                raise SustainRepairScopeError("; ".join(scope_issues))
        if composition.ending is None:
            raise DslError("tonic ending contract is required for new candidates")
        if not 45_000 <= composition.duration_ms <= 60_000:
            raise DslError("pilot duration must be between 45000 and 60000 ms")
        section_contrast = evaluate_section_contrast(composition)
        if section_contrast["status"] != "pass":
            raise DslError("section contrast failed: " + "; ".join(section_contrast["issues"]))
        sustain_profile = evaluate_sustain_profile(composition.materials)
        if sustain_profile["status"] != "pass":
            raise SustainContractError("; ".join(sustain_profile["issues"]))
        midi_path = output_dir / f"{candidate_id}.mid"
        with tempfile.TemporaryDirectory(prefix=f".{candidate_id}-", dir=output_dir) as stage_name:
            stage = Path(stage_name)
            staged_source = stage / source_path.name
            staged_midi = stage / midi_path.name
            staged_source.write_text(source + "\n", encoding="utf-8")
            render_composition(composition, staged_midi)
            _validate_rendered_ending(composition, staged_midi)
            atomic_write_bytes(source_path, staged_source.read_bytes())
            atomic_write_bytes(midi_path, staged_midi.read_bytes())
        features = extract_smf_features(midi_path)
        smf_notes = load_smf_notes(midi_path)
        features.update(
            extract_activity_features(
                [
                    NoteEvent(note.pitch, note.onset_ms, note.duration_ms, note.velocity)
                    for note in smf_notes
                ]
            )
        )
        evaluation = evaluate_features(features, profile)
        evaluation["section_contrast"] = section_contrast
        evaluation["sustain_profile"] = sustain_profile
        if material_development_profile is not None:
            evaluation["material_development"] = evaluate_material_development(
                composition.materials, material_development_profile
            )
        return Candidate(candidate_id, source, midi_path, evaluation, None, repaired)
    except (DslError, OSError, ValueError, EOFError) as error:
        return Candidate(
            candidate_id,
            source,
            None,
            None,
            f"{type(error).__name__}: {error}",
            repaired,
        )


def _validate_rendered_ending(composition: Composition, midi_path: Path) -> None:
    assert composition.ending is not None
    assert composition.tonal_center is not None
    assert composition.mode is not None
    expected_pitches = set(tonic_ending_pitches(composition))
    expected_onset = composition.body_duration_ms + ENDING_BREATH_MS
    expected_duration = composition.duration_ms - expected_onset
    ending_notes = {
        note.pitch
        for note in load_smf_notes(midi_path)
        if note.onset_ms == expected_onset and note.duration_ms == expected_duration
    }
    if not expected_pitches.issubset(ending_notes):
        raise ValueError("rendered SMF did not preserve the tonic ending contract")


def _response_source(response: dict[str, object]) -> str:
    value = response.get("composition_source")
    if not isinstance(value, str):
        raise ValueError("response has no string composition_source")
    return value


def _prompt(name: str, replacements: dict[str, str], prompt_dir: Path | None = None) -> str:
    root = PROJECT_ROOT / "prompts" if prompt_dir is None else Path(prompt_dir)
    value = (root / name).read_text(encoding="utf-8")
    for key, replacement in replacements.items():
        value = value.replace("{{" + key + "}}", replacement)
    return value


def _compose_prompt(
    profile: dict[str, Any],
    candidate_number: int,
    prompt_dir: Path | None = None,
    material_development_profile: dict[str, Any] | None = None,
) -> str:
    generation_axes = {
        axis: metrics for axis, metrics in profile["axes"].items() if axis != "activity"
    }
    return _prompt(
        "pilot-compose.md",
        {
            "candidate_number": str(candidate_number),
            "reference_axes": json.dumps(generation_axes, ensure_ascii=False, sort_keys=True),
            "material_development_reference": json.dumps(
                (material_development_profile or {}).get("metrics", {}),
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
        prompt_dir,
    )


def _repair_prompt(source: str, error: str, prompt_dir: Path | None = None) -> str:
    return _prompt("pilot-repair.md", {"error": error, "source": source}, prompt_dir)


def _revision_prompt(
    source: str,
    axis: str,
    material_id: str,
    evaluation: dict[str, Any],
    prompt_dir: Path | None = None,
) -> str:
    target_evaluation = (
        evaluation["material_development"]
        if axis == "material_development"
        else evaluation["axes"][axis]
    )
    return _prompt(
        "pilot-revise.md",
        {
            "axis": axis,
            "material_id": material_id,
            "evaluation": json.dumps(target_evaluation, ensure_ascii=False, sort_keys=True),
            "source": source,
        },
        prompt_dir,
    )


def _manifest_candidate(candidate: Candidate) -> dict[str, Any]:
    result = asdict(candidate)
    result["midi_path"] = str(candidate.midi_path) if candidate.midi_path else None
    return result


def _publish_final(output_dir: Path, candidate: Candidate) -> tuple[Path, Path]:
    assert candidate.midi_path is not None
    final_source_path = output_dir / "final.music.py"
    final_midi_path = output_dir / "final.mid"
    atomic_write_bytes(final_source_path, (candidate.source + "\n").encode("utf-8"))
    atomic_write_bytes(final_midi_path, candidate.midi_path.read_bytes())
    return final_source_path, final_midi_path


def run_pilot(
    profile: dict[str, Any],
    runner: Runner,
    output_dir: Path,
    *,
    random_seed: int | None = None,
    run_store: RunStore | None = None,
    prompt_dir: Path | None = None,
    material_development_profile: dict[str, Any] | None = None,
) -> PilotResult:
    del random_seed
    output_dir = Path(output_dir)
    candidates_dir = output_dir / "candidates"
    candidates: list[Candidate] = []
    call_count = 0
    for number in range(1, 4):
        candidate_id = f"candidate-{number}"
        compose_step = f"compose-{candidate_id}"
        compose_prompt = _compose_prompt(
            profile,
            number,
            prompt_dir,
            material_development_profile=material_development_profile,
        )
        response = runner.run(
            compose_step,
            compose_prompt,
            {
                "profile": sha256_json(profile),
                "material_development_profile": sha256_json(material_development_profile or {}),
            },
        )
        call_count += 1
        if run_store is not None:
            run_store.record_step(
                compose_step,
                "completed",
                {
                    "profile": sha256_json(profile),
                    "material_development_profile": sha256_json(material_development_profile or {}),
                    "prompt": sha256_text(compose_prompt),
                },
                {"response_sha256": sha256_json(response)},
            )
        candidate = _evaluate_source(
            candidate_id,
            _response_source(response),
            candidates_dir,
            profile,
            material_development_profile=material_development_profile,
            repaired=False,
        )
        if candidate.error:
            sustain_repair_baseline = (
                parse_composition(candidate.source, require_section_contract=True)
                if candidate.error.startswith("SustainContractError:")
                else None
            )
            repair_step = f"repair-{candidate_id}"
            repair_prompt = _repair_prompt(candidate.source, candidate.error, prompt_dir)
            repair = runner.run(
                repair_step,
                repair_prompt,
                {
                    "source": sha256_text(candidate.source),
                    "error": sha256_text(candidate.error),
                },
            )
            call_count += 1
            if run_store is not None:
                run_store.record_step(
                    repair_step,
                    "completed",
                    {
                        "source": sha256_text(candidate.source),
                        "error": sha256_text(candidate.error),
                        "prompt": sha256_text(repair_prompt),
                    },
                    {"response_sha256": sha256_json(repair)},
                )
            candidate = _evaluate_source(
                candidate_id,
                _response_source(repair),
                candidates_dir,
                profile,
                material_development_profile=material_development_profile,
                repaired=True,
                sustain_repair_baseline=sustain_repair_baseline,
            )
        elif run_store is not None:
            run_store.record_step(
                f"repair-{candidate_id}",
                "skipped",
                {"candidate_source": sha256_text(candidate.source)},
                {"reason": "initial candidate passed deterministic validation"},
            )
        candidates.append(candidate)

    valid = [candidate for candidate in candidates if candidate.evaluation is not None]
    if run_store is not None:
        run_store.record_step(
            "evaluate-candidates",
            "completed",
            {candidate.candidate_id: sha256_text(candidate.source) for candidate in candidates},
            {
                "valid_candidate_ids": [candidate.candidate_id for candidate in valid],
                "candidate_evaluations_sha256": sha256_json(
                    [candidate.evaluation for candidate in candidates]
                ),
            },
        )
    if not valid:
        result = PilotResult(
            RunStatus.NO_VALID_CANDIDATE,
            call_count,
            candidates,
            None,
            None,
            None,
            None,
        )
        if run_store is not None:
            run_store.record_step(
                "revise-selected",
                "skipped",
                {"candidate_evaluations": sha256_json([None for _ in candidates])},
                {"reason": "no valid candidate"},
            )
            run_store.record_step(
                "publish-final",
                "failed",
                {"candidate_sources": sha256_json([item.source for item in candidates])},
                {"reason": "no valid candidate"},
            )
        _write_manifest(output_dir, result, run_store=run_store)
        return result
    selected = min(valid, key=_candidate_sort_key)
    assert selected.evaluation is not None
    before = parse_composition(selected.source, require_section_contract=True)
    development = selected.evaluation.get("material_development")
    if development is not None and development["extreme_material_count"] > 0:
        target_axis = "material_development"
        target_material = str(development["worst_material_id"])
        revisable_distance = float(development["extreme_material_count"])
    else:
        target_axis = max(
            (axis for axis in selected.evaluation["axes"] if axis in REVISION_AXES),
            key=lambda axis: selected.evaluation["axes"][axis]["normalized_distance"],
            default=None,
        )
        target_material = max(
            before.materials, key=lambda material: len(material.notes)
        ).material_id
        revisable_distance = (
            selected.evaluation["axes"][target_axis]["normalized_distance"]
            if target_axis is not None
            else 0.0
        )
    if revisable_distance <= 0:
        if run_store is not None:
            run_store.record_step(
                "revise-selected",
                "skipped",
                {"selected_source": sha256_text(selected.source)},
                {"reason": "selected candidate has no revisable outlier"},
            )
        final_source_path, final_midi_path = _publish_final(output_dir, selected)
        revision_status = (
            "skipped_no_outlier"
            if selected.evaluation["worst_axis_distance"] <= 0
            else "skipped_no_revisable_axis"
        )
        _write_json(
            output_dir / "evaluations.json",
            {
                "selected_before": selected.evaluation,
                "revision_after": None,
                "target_axis": target_axis,
                "target_material": target_material,
                "revision_status": revision_status,
                "revision_error": None,
            },
        )
        result = PilotResult(
            RunStatus.COMPLETED,
            call_count,
            candidates,
            selected.candidate_id,
            final_source_path,
            final_midi_path,
            revision_status,
        )
        if run_store is not None:
            run_store.record_step(
                "publish-final",
                "completed",
                {"selected_source": sha256_text(selected.source)},
                {
                    "source_sha256": sha256_file(final_source_path),
                    "midi_sha256": sha256_file(final_midi_path),
                },
            )
        _write_manifest(output_dir, result, run_store=run_store)
        return result
    assert target_axis is not None
    revision_prompt = _revision_prompt(
        selected.source, target_axis, target_material, selected.evaluation, prompt_dir
    )
    revision_response = runner.run(
        "revise-selected",
        revision_prompt,
        {
            "selected_source": sha256_text(selected.source),
            "evaluation": sha256_json(selected.evaluation),
            "target": sha256_json({"axis": target_axis, "material": target_material}),
        },
    )
    call_count += 1
    if run_store is not None:
        run_store.record_step(
            "revise-selected",
            "completed",
            {
                "selected_source": sha256_text(selected.source),
                "evaluation": sha256_json(selected.evaluation),
                "prompt": sha256_text(revision_prompt),
            },
            {"response_sha256": sha256_json(revision_response)},
        )
    revision = _evaluate_source(
        "revision",
        _response_source(revision_response),
        candidates_dir,
        profile,
        material_development_profile=material_development_profile,
        repaired=False,
    )
    revision_error: str | None = None
    if revision.error:
        final_candidate = selected
        revision_status = "rejected_invalid"
        revision_error = revision.error
    else:
        after = parse_composition(revision.source, require_section_contract=True)
        scope_issues = validate_revision_scope(before, after, target_material)
        if scope_issues:
            final_candidate = selected
            revision_status = "rejected_scope"
            revision_error = "; ".join(scope_issues)
        else:
            assert revision.evaluation is not None
            if _revision_improves(selected.evaluation, revision.evaluation, target_axis):
                final_candidate = revision
                revision_status = "accepted"
            else:
                final_candidate = selected
                revision_status = "rejected_no_improvement"
                revision_error = "revision did not improve automatic selection criteria"
    final_source_path, final_midi_path = _publish_final(output_dir, final_candidate)
    _write_json(
        output_dir / "evaluations.json",
        {
            "selected_before": selected.evaluation,
            "revision_after": revision.evaluation,
            "target_axis": target_axis,
            "target_material": target_material,
            "revision_status": revision_status,
            "revision_error": revision_error,
        },
    )
    result = PilotResult(
        RunStatus.COMPLETED,
        call_count,
        candidates,
        selected.candidate_id,
        final_source_path,
        final_midi_path,
        revision_status,
    )
    if run_store is not None:
        run_store.record_step(
            "publish-final",
            "completed",
            {"selected_source": sha256_text(final_candidate.source)},
            {
                "source_sha256": sha256_file(final_source_path),
                "midi_sha256": sha256_file(final_midi_path),
            },
        )
    _write_manifest(output_dir, result, revision_error=revision_error, run_store=run_store)
    return result


def _write_manifest(
    output_dir: Path,
    result: PilotResult,
    *,
    revision_error: str | None = None,
    run_store: RunStore | None = None,
) -> None:
    calls = run_store.rebuild_state()["calls"] if run_store is not None else None
    requested_model = (
        run_store.read_spec().get("requested_model", MODEL_ID)
        if run_store is not None
        else MODEL_ID
    )
    _write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "status": result.status.value,
            "requested_model": requested_model,
            "reported_model": None,
            "reported_model_status": "unable_to_investigate",
            "call_count": (calls["call_attempt_count"] if calls is not None else result.call_count),
            "call_count_definition": (
                "reserved external call attempts; may conservatively overcount interrupted starts"
                if calls is not None
                else "logical runner calls in a non-durable test run"
            ),
            **(calls or {}),
            "selected_candidate": result.selected_candidate,
            "final_source_path": str(result.final_source_path)
            if result.final_source_path
            else None,
            "final_midi_path": str(result.final_midi_path) if result.final_midi_path else None,
            "revision_status": result.revision_status,
            "revision_error": revision_error,
            "candidates": [_manifest_candidate(candidate) for candidate in result.candidates],
        },
    )


def _resolved_input_path(path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate.resolve()
    project_candidate = PROJECT_ROOT / candidate
    return project_candidate.resolve()


def _source_corpus_fingerprints(source_dir: Path) -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted(Path(source_dir).glob("*.mid"), key=lambda item: item.name.casefold())
    }


def _implementation_fingerprints() -> dict[str, str]:
    names = (
        "activity_selection.py",
        "material_development.py",
        "music_dsl.py",
        "pilot_features.py",
        "pilot_loop.py",
        "run_state.py",
        "section_contrast.py",
        "smf_render.py",
        "sustain_profile.py",
    )
    source_root = PROJECT_ROOT / "src" / "llm_musical_composer"
    return {name: sha256_file(source_root / name) for name in names}


def _snapshot_run_inputs(
    store: RunStore,
    *,
    profile: dict[str, Any],
    activity_profile: dict[str, Any],
    material_development_profile: dict[str, Any],
    controls: dict[str, Any],
    selection_profile: dict[str, Any],
    schema_path: Path,
) -> tuple[Path, Path, dict[str, str]]:
    snapshots: dict[str, Path] = {}
    for name, value in (
        ("reference-profile.json", profile),
        ("activity-reference-profile.json", activity_profile),
        ("material-development-reference-profile.json", material_development_profile),
        ("reference-controls.json", controls),
        ("effective-reference-profile.json", selection_profile),
    ):
        relative = Path("inputs") / name
        snapshots[str(relative)] = store.snapshot_json(relative, value)
    prompt_dir = store.run_dir / "inputs" / "prompts"
    for name in PROMPT_NAMES:
        relative = Path("inputs") / "prompts" / name
        snapshots[str(relative)] = store.snapshot_file(relative, PROJECT_ROOT / "prompts" / name)
    schema_snapshot = store.snapshot_file(Path("inputs") / "schema.json", schema_path)
    snapshots[str(Path("inputs") / "schema.json")] = schema_snapshot
    return (
        prompt_dir,
        schema_snapshot,
        {name: sha256_file(path) for name, path in sorted(snapshots.items())},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Codex で最小縦断作曲ループを実行します。")
    parser.add_argument("--source-dir", type=Path, default=Path(".appendix/source-smf"))
    parser.add_argument("--output-root", type=Path, default=Path(".appendix/minimal-loop"))
    parser.add_argument(
        "--schema", type=Path, default=Path("schemas/codex-composition-response.schema.json")
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    run_id = args.run_id or datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    source_dir = _resolved_input_path(args.source_dir)
    schema_path = _resolved_input_path(args.schema)
    run_dir = Path(args.output_root).resolve() / run_id
    store = RunStore(run_dir, max_calls=MAX_CALLS)
    excluded = frozenset({"rut.mid", "aimusic01.mid"})
    with RunLock(run_dir / ".run.lock"):
        profile = build_reference_profile_from_directory(source_dir, excluded_names=excluded)
        activity_profile = build_activity_reference_profile_from_directory(
            source_dir, excluded_names=excluded
        )
        material_development_profile = build_material_development_reference_profile_from_directory(
            source_dir, excluded_names=excluded
        )
        if activity_profile["unable_to_investigate"]:
            raise RuntimeError("activity reference profile contains unreadable source files")
        if material_development_profile["unable_to_investigate"]:
            raise RuntimeError(
                "material development reference profile contains unreadable source files"
            )
        controls: dict[str, Any]
        if profile.get("source_files"):
            representative = source_dir / profile["source_files"][0]
            smf_notes = load_smf_notes(representative)
            controls = run_minimal_controls(
                [
                    NoteEvent(note.pitch, note.onset_ms, note.duration_ms, note.velocity)
                    for note in smf_notes
                ]
            )
        else:
            controls = {
                "status": FeatureStatus.UNABLE.value,
                "detail": "reference profile did not identify a usable source file",
            }
        selection_profile = {**profile, "axes": dict(profile["axes"])}
        selection_profile["axes"]["activity"] = activity_profile["axes"]["activity"]
        if (
            "time_stretch" in controls
            and controls["time_stretch"]["status"] != FeatureStatus.PASS.value
        ):
            selection_profile["axes"].pop("relative_timing", None)
        if "order_disruption" in controls:
            changed_axes = set(controls["order_disruption"].get("changed_axes", []))
            for axis in ("pitch_order", "relative_timing"):
                if axis not in changed_axes:
                    selection_profile["axes"].pop(axis, None)
        prompt_dir, schema_snapshot, snapshot_hashes = _snapshot_run_inputs(
            store,
            profile=profile,
            activity_profile=activity_profile,
            material_development_profile=material_development_profile,
            controls=controls,
            selection_profile=selection_profile,
            schema_path=schema_path,
        )
        spec = {
            "schema_version": 1,
            "run_id": run_id,
            "requested_model": args.model,
            "max_calls": MAX_CALLS,
            "step_ids": list(STEP_IDS),
            "excluded_source_names": sorted(excluded),
            "source_corpus_sha256": sha256_json(_source_corpus_fingerprints(source_dir)),
            "input_snapshots": snapshot_hashes,
            "implementation_sha256": _implementation_fingerprints(),
        }
        store.initialize(spec)
        runner = CodexExecRunner(
            run_store=store,
            schema_path=schema_snapshot,
            model=args.model,
            working_directory=isolated_codex_working_directory(),
        )
        result = run_pilot(
            selection_profile,
            runner,
            run_dir,
            run_store=store,
            prompt_dir=prompt_dir,
            material_development_profile=material_development_profile,
        )
    print(json.dumps({"status": result.status.value, "run_id": run_id}, ensure_ascii=False))
    return 0 if result.status is RunStatus.COMPLETED else 2


if __name__ == "__main__":
    raise SystemExit(main())
