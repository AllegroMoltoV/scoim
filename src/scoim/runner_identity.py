"""Pre-call runner identity and post-call runner-record verification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from .proposal import ProposalRun, ProposalRunner
from .validation import IssueCode, ValidationIssue


@dataclass(frozen=True, slots=True)
class RunnerIdentity:
    """Generation-affecting runner values fixed before a public run starts."""

    provider: str
    model: str
    model_settings: Mapping[str, object]


def read_runner_identity(runner: ProposalRunner) -> RunnerIdentity:
    """Read and validate the identity exposed by one public v2 runner."""
    identity_method = getattr(runner, "identity", None)
    if identity_method is None:
        raise TypeError("The runner does not expose its identity")
    identity = identity_method()
    if not isinstance(identity, RunnerIdentity):
        raise TypeError("The runner identity has an invalid type")
    if not identity.provider.strip() or not identity.model.strip():
        raise ValueError("The runner identity is incomplete")
    if any(not isinstance(key, str) for key in identity.model_settings):
        raise ValueError("The runner model setting keys must be strings")
    return identity


class IdentityCheckingRunner:
    """Delegate calls while rejecting records that differ from fixed conditions."""

    def __init__(self, runner: ProposalRunner, identity: RunnerIdentity) -> None:
        self._runner = runner
        self._identity = identity

    def preflight(self) -> tuple[ValidationIssue, ...]:
        preflight = getattr(self._runner, "preflight", None)
        return () if preflight is None else preflight()

    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        result = self._runner.run(prompt, response_schema_path)
        differences: list[str] = []
        if result.provider != self._identity.provider:
            differences.append("provider")
        if result.model != self._identity.model:
            differences.append("model")
        for key, expected in self._identity.model_settings.items():
            if result.model_settings.get(key) != expected:
                differences.append(f"model_settings/{key}")
        if not differences:
            return result
        issue = ValidationIssue(
            IssueCode.LINEAGE_MISMATCH,
            f"Runner record differs from its declared identity: {', '.join(differences)}",
            "/runner",
        )
        return replace(result, issues=(*result.issues, issue))
