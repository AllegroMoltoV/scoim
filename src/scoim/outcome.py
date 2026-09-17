"""Typed terminal outcomes for generation attempts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from .validation import IssueCode, ValidationIssue


class TerminalState(StrEnum):
    """How an execution attempt ended."""

    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class ArtifactDisposition(StrEnum):
    """How generated artifacts may be used."""

    NONE = "none"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    CANDIDATE = "candidate"


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    """Orthogonal terminal state, artifact use, and typed reasons."""

    terminal_state: TerminalState
    artifact_disposition: ArtifactDisposition
    issues: tuple[ValidationIssue, ...]

    def __post_init__(self) -> None:
        if (
            self.terminal_state is TerminalState.COMPLETED
            and self.artifact_disposition is ArtifactDisposition.NONE
        ):
            raise ValueError("completed outcome must have artifacts")
        if (
            self.terminal_state is not TerminalState.COMPLETED
            and self.artifact_disposition is not ArtifactDisposition.NONE
        ):
            raise ValueError("non-completed outcome cannot have artifacts")
        if self.terminal_state is TerminalState.FAILED and not self.issues:
            raise ValueError("failed outcome requires an issue")
        if self.artifact_disposition is ArtifactDisposition.CANDIDATE and self.issues:
            raise ValueError("candidate outcome cannot have issues")
        if self.artifact_disposition is ArtifactDisposition.DIAGNOSTIC_ONLY and (
            not self.issues
            or any(issue.code is not IssueCode.PROJECTION_TARGET_UNMET for issue in self.issues)
        ):
            raise ValueError("diagnostic outcome requires target issues")

    @property
    def promotion_eligible(self) -> bool:
        """Return whether the artifacts may become a formal candidate."""

        return self.artifact_disposition is ArtifactDisposition.CANDIDATE

    def to_dict(self) -> dict[str, object]:
        """Return the stable persisted representation."""

        return {
            "terminal_state": self.terminal_state.value,
            "artifact_disposition": self.artifact_disposition.value,
            "issues": [
                {
                    "code": issue.code.value,
                    "message": issue.message,
                    "path": issue.path,
                }
                for issue in self.issues
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ExecutionOutcome:
        """Parse and validate the persisted representation."""

        if set(value) != {"terminal_state", "artifact_disposition", "issues"}:
            raise ValueError("outcome fields do not match the contract")
        raw_issues = value["issues"]
        if not isinstance(raw_issues, list):
            raise ValueError("outcome issues must be an array")
        issues = []
        for raw_issue in raw_issues:
            if not isinstance(raw_issue, Mapping) or set(raw_issue) != {
                "code",
                "message",
                "path",
            }:
                raise ValueError("outcome issue fields do not match the contract")
            message = raw_issue["message"]
            path = raw_issue["path"]
            if not isinstance(message, str) or not isinstance(path, str):
                raise ValueError("outcome issue text must be strings")
            try:
                code = IssueCode(raw_issue["code"])
            except (TypeError, ValueError) as error:
                raise ValueError("outcome issue code is invalid") from error
            issues.append(ValidationIssue(code=code, message=message, path=path))
        try:
            terminal_state = TerminalState(value["terminal_state"])
            artifact_disposition = ArtifactDisposition(value["artifact_disposition"])
        except (TypeError, ValueError) as error:
            raise ValueError("outcome state is invalid") from error
        return cls(
            terminal_state=terminal_state,
            artifact_disposition=artifact_disposition,
            issues=tuple(issues),
        )
