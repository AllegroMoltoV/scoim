import json
import math
from pathlib import Path

import pytest

from scoim.flow_proposal import FlowProposalRequest, propose_flow
from scoim.proposal import ProposalRun
from scoim.validation import IssueCode, ValidationIssue


def _valid_response() -> dict[str, object]:
    return {
        "title": "窓辺の午後",
        "overall_flow": "明るく始まり、少し陰ってから穏やかに戻る。",
        "scenes": [
            {
                "name": "やわらかな始まり",
                "length_class": "short",
                "heard_as": "軽やかな音が現れる。",
                "relation_to_previous": "曲の始まり。",
                "transition_to_next": "動きを少し広げる。",
            },
            {
                "name": "穏やかな帰り道",
                "length_class": "long",
                "heard_as": "冒頭を思い出しながら落ち着く。",
                "relation_to_previous": "最初の感じが少し変わって戻る。",
                "transition_to_next": "余韻を残して終わる。",
            },
        ],
    }


class FixedRunner:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, object]]] = []

    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        schema = json.loads(response_schema_path.read_text(encoding="utf-8"))
        self.calls.append((prompt, schema))
        response = self.responses[len(self.calls) - 1]
        return ProposalRun(
            provider="fixed",
            model="fixed-model",
            model_settings={},
            started=True,
            terminal_state="completed",
            raw_response=json.dumps(response, ensure_ascii=False).encode("utf-8"),
            events=b'{"type":"turn.completed"}\n',
            stderr=b"",
            issues=(),
        )


def test_propose_flow_normalizes_scene_ids_and_records_the_run(tmp_path: Path) -> None:
    runner = FixedRunner([_valid_response()])
    request = FlowProposalRequest(
        instruction="明るい感じのピアノ曲",
        document_id="flow-001",
        instrumentation="solo_piano",
        target_duration=180,
    )

    result = propose_flow(request, runner, tmp_path / "run")

    assert result.proposed is True
    assert result.document is not None
    assert result.document["document_type"] == "flow"
    assert result.document["schema_version"] == "0.1.0"
    assert [scene["scene_id"] for scene in result.document["scenes"]] == [
        "scene-001",
        "scene-002",
    ]
    assert len(runner.calls) == 1
    assert (tmp_path / "run" / "outputs" / "flow-draft.json").is_file()
    state = json.loads((tmp_path / "run" / "run-state.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["calls"]["call_attempt_count"] == 1


def test_propose_flow_records_relative_run_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    runner = FixedRunner([_valid_response()])
    request = FlowProposalRequest(
        instruction="明るい感じのピアノ曲",
        document_id="flow-001",
        instrumentation="solo_piano",
        target_duration=180,
    )

    result = propose_flow(request, runner, Path("run"))

    assert result.proposed is True
    assert result.run_dir == (tmp_path / "run").resolve()
    state = json.loads((tmp_path / "run" / "run-state.json").read_text(encoding="utf-8"))
    assert Path(state["steps"]["publish-final"]["outputs"]["document_path"]) == (
        Path("outputs/flow-draft.json")
    )


class PreflightFailureRunner:
    def __init__(self) -> None:
        self.called = False

    def preflight(self) -> tuple[ValidationIssue, ...]:
        return (ValidationIssue(IssueCode.RUNNER_UNAVAILABLE, "Runner is unavailable", "/runner"),)

    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        self.called = True
        raise AssertionError("run must not be called")


def test_propose_flow_does_not_create_a_run_for_preflight_failure(tmp_path: Path) -> None:
    runner = PreflightFailureRunner()
    request = FlowProposalRequest("明るい曲", "flow-001", "solo_piano", 180)

    result = propose_flow(request, runner, tmp_path / "run")

    assert result.proposed is False
    assert result.issues[0].code is IssueCode.RUNNER_UNAVAILABLE
    assert runner.called is False
    assert not (tmp_path / "run").exists()


class StartedFailureRunner:
    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        return ProposalRun(
            provider="fixed",
            model="fixed-model",
            model_settings={},
            started=True,
            terminal_state="failed",
            raw_response=None,
            events=b'{"type":"thread.started"}\n{"type":"turn.failed"}\n',
            stderr=b"failed\n",
            issues=(ValidationIssue(IssueCode.RUNNER_FAILED, "Runner failed", "/runner"),),
        )


def test_propose_flow_records_a_started_runner_failure(tmp_path: Path) -> None:
    request = FlowProposalRequest("明るい曲", "flow-001", "solo_piano", 180)

    result = propose_flow(request, StartedFailureRunner(), tmp_path / "run")

    assert result.proposed is False
    assert result.run_dir == tmp_path / "run"
    assert result.issues[0].code is IssueCode.RUNNER_FAILED
    state = json.loads((tmp_path / "run" / "run-state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["calls"]["call_attempt_count"] == 1
    assert state["calls"]["failed_external_call_count"] == 1


def test_propose_flow_does_not_repair_an_unlocalizable_missing_field(tmp_path: Path) -> None:
    response = _valid_response()
    del response["title"]
    runner = FixedRunner([response])
    request = FlowProposalRequest("明るい曲", "flow-001", "solo_piano", 180)

    result = propose_flow(request, runner, tmp_path / "run")

    assert result.proposed is False
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert len(runner.calls) == 1
    state = json.loads((tmp_path / "run" / "run-state.json").read_text(encoding="utf-8"))
    assert state["calls"]["call_attempt_count"] == 1


def test_propose_flow_repairs_only_a_localized_scalar_leaf_once(tmp_path: Path) -> None:
    response = _valid_response()
    response["scenes"][0]["length_class"] = "brief"
    runner = FixedRunner(
        [
            response,
            {
                "replacements": [
                    {"path": "/scenes/0/length_class", "value": "short"},
                ]
            },
        ]
    )
    request = FlowProposalRequest("明るい曲", "flow-001", "solo_piano", 180)

    result = propose_flow(request, runner, tmp_path / "run")

    assert result.proposed is True
    assert result.document is not None
    assert result.document["scenes"][0]["length_class"] == "short"
    assert result.document["scenes"][1]["name"] == "穏やかな帰り道"
    assert len(runner.calls) == 2
    assert runner.calls[1][1]["properties"]["replacements"]["minItems"] == 1


def test_propose_flow_stops_after_one_failed_repair(tmp_path: Path) -> None:
    response = _valid_response()
    response["scenes"][0]["length_class"] = "brief"
    runner = FixedRunner(
        [
            response,
            {
                "replacements": [
                    {"path": "/scenes/0/length_class", "value": "tiny"},
                ]
            },
        ]
    )
    request = FlowProposalRequest("明るい曲", "flow-001", "solo_piano", 180)

    result = propose_flow(request, runner, tmp_path / "run")

    assert result.proposed is False
    assert result.issues[0].path == "/scenes/0/length_class"
    assert len(runner.calls) == 2
    state = json.loads((tmp_path / "run" / "run-state.json").read_text(encoding="utf-8"))
    assert state["calls"]["call_attempt_count"] == 2


@pytest.mark.parametrize(
    ("proposal_request", "path"),
    [
        (FlowProposalRequest(" ", "flow-001", "solo_piano", 180), "/instruction"),
        (FlowProposalRequest("曲", "Flow 1", "solo_piano", 180), "/document_id"),
        (FlowProposalRequest("曲", "flow-001", "piano quartet", 180), "/instrumentation"),
        (FlowProposalRequest("曲", "flow-001", "solo_piano", math.nan), "/target_duration"),
    ],
)
def test_propose_flow_rejects_invalid_requests_before_creating_a_run(
    tmp_path: Path,
    proposal_request: FlowProposalRequest,
    path: str,
) -> None:
    runner = FixedRunner([_valid_response()])

    result = propose_flow(proposal_request, runner, tmp_path / "run")

    assert result.proposed is False
    assert result.issues[0].path == path
    assert runner.calls == []
    assert not (tmp_path / "run").exists()


def test_propose_flow_rejects_an_existing_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    runner = FixedRunner([_valid_response()])

    result = propose_flow(
        FlowProposalRequest("明るい曲", "flow-001", "solo_piano", 180), runner, run_dir
    )

    assert result.proposed is False
    assert result.issues[0].code is IssueCode.STORAGE_CONFLICT
    assert runner.calls == []


class RawRunner:
    def __init__(self, response: bytes) -> None:
        self.response = response

    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        return ProposalRun(
            "fixed",
            "fixed-model",
            {},
            True,
            "completed",
            self.response,
            None,
            None,
            (),
        )


@pytest.mark.parametrize("response", [b"not json", b"[]"])
def test_propose_flow_records_a_non_object_json_response(tmp_path: Path, response: bytes) -> None:
    result = propose_flow(
        FlowProposalRequest("明るい曲", "flow-001", "solo_piano", 180),
        RawRunner(response),
        tmp_path / "run",
    )

    assert result.proposed is False
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert result.run_dir == tmp_path / "run"


class RepairFailureRunner(FixedRunner):
    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        if self.calls:
            self.calls.append((prompt, json.loads(response_schema_path.read_text("utf-8"))))
            return ProposalRun(
                "fixed",
                "fixed-model",
                {},
                True,
                "failed",
                None,
                b'{"type":"turn.failed"}\n',
                b"failed\n",
                (ValidationIssue(IssueCode.RUNNER_FAILED, "Repair failed", "/runner"),),
            )
        return super().run(prompt, response_schema_path)


def test_propose_flow_records_a_failed_repair_attempt(tmp_path: Path) -> None:
    response = _valid_response()
    response["scenes"][0]["length_class"] = "brief"
    runner = RepairFailureRunner([response])

    result = propose_flow(
        FlowProposalRequest("明るい曲", "flow-001", "solo_piano", 180),
        runner,
        tmp_path / "run",
    )

    assert result.proposed is False
    assert result.issues[0].code is IssueCode.RUNNER_FAILED
    assert len(runner.calls) == 2


def test_propose_flow_rejects_a_malformed_repair_response(tmp_path: Path) -> None:
    response = _valid_response()
    response["scenes"][0]["length_class"] = "brief"
    runner = FixedRunner([response, {"wrong": []}])

    result = propose_flow(
        FlowProposalRequest("明るい曲", "flow-001", "solo_piano", 180),
        runner,
        tmp_path / "run",
    )

    assert result.proposed is False
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert len(runner.calls) == 2


def test_propose_flow_requires_each_allowed_repair_path_once(tmp_path: Path) -> None:
    response = _valid_response()
    response["scenes"][0]["name"] = ""
    response["scenes"][0]["length_class"] = "brief"
    runner = FixedRunner(
        [
            response,
            {
                "replacements": [
                    {"path": "/scenes/0/name", "value": "始まり"},
                    {"path": "/scenes/0/name", "value": "別名"},
                ]
            },
        ]
    )

    result = propose_flow(
        FlowProposalRequest("明るい曲", "flow-001", "solo_piano", 180),
        runner,
        tmp_path / "run",
    )

    assert result.proposed is False
    assert result.issues[0].path == "/replacements"
    assert len(runner.calls) == 2


def test_propose_flow_does_not_repair_a_non_scalar_value(tmp_path: Path) -> None:
    response = _valid_response()
    response["scenes"][0]["name"] = {"wrong": "shape"}
    runner = FixedRunner([response])

    result = propose_flow(
        FlowProposalRequest("明るい曲", "flow-001", "solo_piano", 180),
        runner,
        tmp_path / "run",
    )

    assert result.proposed is False
    assert result.issues[0].path == "/scenes/0/name"
    assert len(runner.calls) == 1
