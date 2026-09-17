"""Coordinate the fixed solo-piano realization stages as one resumable run."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from llm_musical_composer.run_state import RunStore, atomic_write_json

from .projection import check_solo_piano_3m_structure
from .proposal import ProposalRunner
from .realization_operations import (
    advance_accompaniment_realization_run,
    advance_ending_realization_run,
    advance_melody_realization_run,
    advance_minimal_realization_run,
    advance_performance_realization_run,
    initialize_accompaniment_realization_run,
    initialize_ending_realization_run,
    initialize_melody_realization_run,
    initialize_minimal_realization_run,
    initialize_performance_realization_run,
)
from .realization_run import RealizationRunStatus, RealizationRunStore
from .realization_script import check_realization_script, realization_script_sha256
from .realization_workspace import RealizationWorkspace, workspace_to_dict
from .validation import ValidationIssue

_PROFILE = "solo_piano_3m_v1"
_STAGES = (
    ("plan-harmony", "stages/01-plan-harmony"),
    ("melody", "stages/02-melody"),
    ("accompaniment", "stages/03-accompaniment"),
    ("ending", "stages/04-ending"),
    ("performance", "stages/05-performance"),
)


@dataclass(frozen=True, slots=True)
class StagedRealizationStatus:
    """Derived state of one complete sequence of realization stages."""

    status: str
    current_stage: str | None
    completed_stages: tuple[str, ...]
    workspace: RealizationWorkspace | None
    awaiting_review: str | None
    issues: tuple[ValidationIssue, ...]


def _review_spec(review_after: Mapping[str, frozenset[str]]) -> dict[str, list[str]]:
    known = {stage_id for stage_id, _ in _STAGES}
    unknown = set(review_after) - known
    if unknown:
        raise ValueError(f"Unknown staged realization review stage: {sorted(unknown)[0]}")
    return {stage_id: sorted(review_after.get(stage_id, frozenset())) for stage_id, _ in _STAGES}


def initialize_staged_realization(
    run_dir: Path,
    document: dict[str, object],
    *,
    profile: str,
    model: str,
    review_after: Mapping[str, frozenset[str]],
) -> StagedRealizationStatus:
    """Initialize or verify the immutable inputs for one staged realization."""
    validation = check_realization_script(document)
    if not validation.valid:
        raise ValueError(validation.issues[0].message)
    if profile != _PROFILE:
        raise ValueError(f"Unsupported staged realization profile: {profile}")
    profile_validation = check_solo_piano_3m_structure(document)
    if not profile_validation.valid:
        raise ValueError(profile_validation.issues[0].message)
    review_spec = _review_spec(review_after)
    spec = {
        "schema_version": 1,
        "approved_script_sha256": realization_script_sha256(document),
        "profile": profile,
        "model": model,
        "stages": [{"stage_id": stage_id, "run_path": run_path} for stage_id, run_path in _STAGES],
        "review_after": review_spec,
        "call_policy": "bounded_content_repair_v1",
    }
    root = Path(run_dir).resolve()
    store = RunStore(root, max_calls=0)
    store.initialize(spec)
    store.snapshot_json("inputs/approved-script.json", document)
    return rebuild_staged_realization(root)


def rebuild_staged_realization(run_dir: Path) -> StagedRealizationStatus:
    """Reconstruct parent state only from immutable inputs and child runs."""
    root = Path(run_dir).resolve()
    spec = RunStore(root, max_calls=0).read_spec()
    document = _read_document(root / "inputs" / "approved-script.json")
    if spec.get("approved_script_sha256") != realization_script_sha256(document):
        raise ValueError("The staged run script does not match its immutable specification")
    completed: list[str] = []
    workspace: RealizationWorkspace | None = None
    awaiting_review: str | None = None
    issues: tuple[ValidationIssue, ...] = ()
    status_name = "initialized"
    current_stage: str | None = cast(list[dict[str, object]], spec["stages"])[0]["stage_id"]
    expected_source_hash: str | None = None
    stage_states: list[dict[str, object]] = []
    for raw_stage in cast(list[dict[str, object]], spec["stages"]):
        stage_id = cast(str, raw_stage["stage_id"])
        child_dir = root / cast(str, raw_stage["run_path"])
        if not child_dir.is_dir():
            current_stage = stage_id
            break
        child_spec = RunStore(child_dir).read_spec()
        if child_spec.get("approved_script_sha256") != spec["approved_script_sha256"]:
            raise ValueError(f"Stage script hash does not match the parent: {stage_id}")
        if child_spec.get("source_workspace_record_sha256") != expected_source_hash:
            raise ValueError(
                f"Stage input workspace does not continue the parent chain: {stage_id}"
            )
        child = RealizationRunStore(child_dir).rebuild()
        stage_states.append(_stage_state(stage_id, child_spec, child))
        workspace = child.workspace
        if child.status == "completed":
            completed.append(stage_id)
            expected_source_hash = workspace.workspace_record_sha256
            status_name = "running"
            continue
        current_stage = stage_id
        status_name = child.status
        issues = child.issues
        if child.awaiting_review is not None:
            awaiting_review = f"{stage_id}:{child.awaiting_review}"
        break
    else:
        status_name = "completed"
        current_stage = None
    status = StagedRealizationStatus(
        status=status_name,
        current_stage=cast(str | None, current_stage),
        completed_stages=tuple(completed),
        workspace=workspace,
        awaiting_review=awaiting_review,
        issues=issues,
    )
    _save_state(root, status, stage_states=stage_states)
    return status


def advance_staged_realization(
    run_dir: Path,
    runner: ProposalRunner,
) -> StagedRealizationStatus:
    """Advance stages until completion, failure, or a configured review point."""
    root = Path(run_dir).resolve()
    spec = RunStore(root, max_calls=0).read_spec()
    document = _read_document(root / "inputs" / "approved-script.json")
    review_after = cast(dict[str, list[str]], spec["review_after"])
    model = cast(str, spec["model"])
    while True:
        parent = rebuild_staged_realization(root)
        if parent.status in {"awaiting_review", "failed", "completed"}:
            return parent
        assert parent.current_stage is not None
        stage_id = parent.current_stage
        stage_path = next(
            cast(str, stage["run_path"])
            for stage in cast(list[dict[str, object]], spec["stages"])
            if stage["stage_id"] == stage_id
        )
        child_dir = root / stage_path
        if not child_dir.exists():
            _initialize_stage(
                stage_id,
                child_dir,
                document,
                parent.workspace,
                review_after=frozenset(review_after[stage_id]),
                model=model,
            )
        child = _advance_stage(stage_id, child_dir, runner)
        if child.status != "completed":
            return rebuild_staged_realization(root)


def _initialize_stage(
    stage_id: str,
    run_dir: Path,
    document: dict[str, object],
    workspace: RealizationWorkspace | None,
    *,
    review_after: frozenset[str],
    model: str,
) -> None:
    if stage_id == "plan-harmony":
        if workspace is not None:
            raise ValueError("The first stage cannot start from an existing workspace")
        initialize_minimal_realization_run(
            run_dir,
            document,
            review_after=review_after,
            model=model,
            max_calls=2,
        )
        return
    if workspace is None:
        raise ValueError(f"Stage requires a preceding workspace: {stage_id}")
    source_hash = workspace.workspace_record_sha256
    script = cast(dict[str, object], document["script"])
    has_transitions = bool(cast(dict[str, object], script["transitions"]))
    model_calls = 1 + int(has_transitions)
    if stage_id == "melody":
        initialize_melody_realization_run(
            run_dir,
            document,
            workspace,
            source_workspace_record_sha256=source_hash,
            review_after=review_after,
            model=model,
            max_calls=model_calls,
        )
    elif stage_id == "accompaniment":
        content_repair_limit = 1
        initialize_accompaniment_realization_run(
            run_dir,
            document,
            workspace,
            source_workspace_record_sha256=source_hash,
            review_after=review_after,
            model=model,
            max_calls=model_calls * (content_repair_limit + 1),
            content_repair_limit=content_repair_limit,
        )
    elif stage_id == "ending":
        initialize_ending_realization_run(
            run_dir,
            document,
            workspace,
            source_workspace_record_sha256=source_hash,
            review_after=review_after,
        )
    elif stage_id == "performance":
        initialize_performance_realization_run(
            run_dir,
            document,
            workspace,
            source_workspace_record_sha256=source_hash,
            review_after=review_after,
            model=model,
        )
    else:
        raise ValueError(f"Unknown staged realization stage: {stage_id}")


def _advance_stage(
    stage_id: str,
    run_dir: Path,
    runner: ProposalRunner,
) -> RealizationRunStatus:
    if stage_id == "plan-harmony":
        return advance_minimal_realization_run(run_dir, runner)
    if stage_id == "melody":
        return advance_melody_realization_run(run_dir, runner)
    if stage_id == "accompaniment":
        return advance_accompaniment_realization_run(run_dir, runner)
    if stage_id == "ending":
        return advance_ending_realization_run(run_dir)
    if stage_id == "performance":
        return advance_performance_realization_run(run_dir, runner)
    raise ValueError(f"Unknown staged realization stage: {stage_id}")


def _stage_state(
    stage_id: str,
    spec: Mapping[str, object],
    status: RealizationRunStatus,
) -> dict[str, object]:
    return {
        "stage_id": stage_id,
        "status": status.status,
        "input_workspace_record_sha256": spec["initial_workspace_record_sha256"],
        "output_workspace_record_sha256": status.workspace.workspace_record_sha256,
        "awaiting_review": status.awaiting_review,
        "issues": [
            {"code": issue.code.value, "message": issue.message, "path": issue.path}
            for issue in status.issues
        ],
    }


def _save_state(
    run_dir: Path,
    status: StagedRealizationStatus,
    *,
    stage_states: list[dict[str, object]] | None = None,
) -> None:
    atomic_write_json(
        run_dir / "staged-realization-state.json",
        {
            "schema_version": 1,
            "status": status.status,
            "current_stage": status.current_stage,
            "completed_stages": list(status.completed_stages),
            "awaiting_review": status.awaiting_review,
            "issues": [
                {"code": issue.code.value, "message": issue.message, "path": issue.path}
                for issue in status.issues
            ],
            "stages": stage_states or [],
            "workspace": workspace_to_dict(status.workspace)
            if status.workspace is not None
            else None,
        },
    )


def _read_document(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return cast(dict[str, object], value)
