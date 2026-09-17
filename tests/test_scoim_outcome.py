from __future__ import annotations

import pytest

from scoim.outcome import ArtifactDisposition, ExecutionOutcome, TerminalState
from scoim.validation import IssueCode, ValidationIssue


def test_completed_outcome_cannot_have_no_artifacts() -> None:
    with pytest.raises(ValueError, match="completed outcome must have artifacts"):
        ExecutionOutcome(
            terminal_state=TerminalState.COMPLETED,
            artifact_disposition=ArtifactDisposition.NONE,
            issues=(),
        )


def test_candidate_outcome_cannot_have_issues() -> None:
    issue = ValidationIssue(IssueCode.STORAGE_ERROR, "failed", "/output")

    with pytest.raises(ValueError, match="candidate outcome cannot have issues"):
        ExecutionOutcome(
            terminal_state=TerminalState.COMPLETED,
            artifact_disposition=ArtifactDisposition.CANDIDATE,
            issues=(issue,),
        )


def test_failed_outcome_cannot_expose_artifacts() -> None:
    issue = ValidationIssue(IssueCode.STORAGE_ERROR, "failed", "/output")

    with pytest.raises(ValueError, match="non-completed outcome cannot have artifacts"):
        ExecutionOutcome(
            terminal_state=TerminalState.FAILED,
            artifact_disposition=ArtifactDisposition.DIAGNOSTIC_ONLY,
            issues=(issue,),
        )


def test_failed_outcome_requires_an_issue() -> None:
    with pytest.raises(ValueError, match="failed outcome requires an issue"):
        ExecutionOutcome(
            terminal_state=TerminalState.FAILED,
            artifact_disposition=ArtifactDisposition.NONE,
            issues=(),
        )


def test_diagnostic_outcome_requires_projection_target_issue() -> None:
    issue = ValidationIssue(IssueCode.MODEL_OUTPUT_INVALID, "failed", "/response")

    with pytest.raises(ValueError, match="diagnostic outcome requires target issues"):
        ExecutionOutcome(
            terminal_state=TerminalState.COMPLETED,
            artifact_disposition=ArtifactDisposition.DIAGNOSTIC_ONLY,
            issues=(issue,),
        )


def test_candidate_outcome_serializes_without_a_duplicate_promotion_flag() -> None:
    outcome = ExecutionOutcome(
        terminal_state=TerminalState.COMPLETED,
        artifact_disposition=ArtifactDisposition.CANDIDATE,
        issues=(),
    )

    assert outcome.promotion_eligible
    assert outcome.to_dict() == {
        "terminal_state": "completed",
        "artifact_disposition": "candidate",
        "issues": [],
    }


def test_diagnostic_outcome_round_trips_through_persisted_data() -> None:
    issue = ValidationIssue(
        IssueCode.PROJECTION_TARGET_UNMET,
        "target was not met",
        "/quality",
    )
    outcome = ExecutionOutcome(
        terminal_state=TerminalState.COMPLETED,
        artifact_disposition=ArtifactDisposition.DIAGNOSTIC_ONLY,
        issues=(issue,),
    )

    assert ExecutionOutcome.from_dict(outcome.to_dict()) == outcome


@pytest.mark.parametrize(
    "value",
    (
        {},
        {"terminal_state": "failed", "artifact_disposition": "none", "issues": {}},
        {"terminal_state": "failed", "artifact_disposition": "none", "issues": [None]},
        {
            "terminal_state": "failed",
            "artifact_disposition": "none",
            "issues": [{"code": "storage_error", "message": 1, "path": "/output"}],
        },
        {
            "terminal_state": "failed",
            "artifact_disposition": "none",
            "issues": [{"code": "unknown", "message": "failed", "path": "/output"}],
        },
        {"terminal_state": "unknown", "artifact_disposition": "none", "issues": []},
    ),
)
def test_persisted_outcome_rejects_values_outside_its_contract(
    value: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        ExecutionOutcome.from_dict(value)
