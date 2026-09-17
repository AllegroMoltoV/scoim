from pathlib import Path

from scoim.proposal import ProposalRun
from scoim.runner_identity import (
    IdentityCheckingRunner,
    RunnerIdentity,
    read_runner_identity,
)
from scoim.validation import IssueCode


class MismatchedRunner:
    def identity(self) -> RunnerIdentity:
        return RunnerIdentity(
            provider="fake",
            model="requested-model",
            model_settings={"reasoning_effort": "medium"},
        )

    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        return ProposalRun(
            provider="fake",
            model="different-model",
            model_settings={"reasoning_effort": "medium"},
            started=True,
            terminal_state="completed",
            raw_response=b"{}",
            events=None,
            stderr=None,
            issues=(),
        )


def test_identity_checking_runner_rejects_a_different_reported_model(tmp_path: Path) -> None:
    runner = MismatchedRunner()
    identity = read_runner_identity(runner)
    checked = IdentityCheckingRunner(runner, identity)

    result = checked.run("prompt", tmp_path / "schema.json")

    assert result.issues[0].code is IssueCode.LINEAGE_MISMATCH
    assert result.model == "different-model"
