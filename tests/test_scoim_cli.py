import copy
import json
from pathlib import Path

from test_scoim_script_compilation import _approved_flow, _response
from test_scoim_staged_realization import _complete_responses

import scoim.cli as cli_module
from scoim.cli import main
from scoim.proposal import ProposalRun
from scoim.public_realization import PublicRealizationResult
from scoim.validation import IssueCode, ValidationIssue


def _draft_flow() -> dict[str, object]:
    document = copy.deepcopy(_approved_flow())
    document["status"] = "draft"
    document["approval"] = None
    return document


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _proposal_response() -> dict[str, object]:
    return {
        "title": "窓辺の午後",
        "overall_flow": "主題が現れ、景色が広がり、少し変えて戻って閉じる。",
        "scenes": [
            {
                "name": "主題",
                "length_class": "short",
                "heard_as": "静かに始まる。",
                "relation_to_previous": "曲の始まり。",
                "transition_to_next": "短いつなぎへ進む。",
            },
            {
                "name": "広がり",
                "length_class": "medium",
                "heard_as": "景色が広がる。",
                "relation_to_previous": "少し動きを増す。",
                "transition_to_next": "主題へ戻る。",
            },
            {
                "name": "帰還",
                "length_class": "long",
                "heard_as": "主題が変化して戻る。",
                "relation_to_previous": "広がりを受けて落ち着く。",
                "transition_to_next": "余韻を残して閉じる。",
            },
        ],
    }


def _fixed_aba_stage_responses() -> list[dict[str, object]]:
    responses = _complete_responses()
    responses[0]["contrast_descriptions"] = ["Open the register"]
    responses[3]["transition_melodies"] = responses[3]["transition_melodies"][:1]
    responses[5]["transition_accompaniments"] = responses[5]["transition_accompaniments"][:1]
    return [*responses[:7], responses[6]]


def test_check_cli_returns_a_typed_flow_validation_failure(tmp_path: Path, capsys) -> None:
    document = _draft_flow()
    document["schema_version"] = "9.9.9"
    flow_path = tmp_path / "flow.json"
    _write_json(flow_path, document)

    exit_code = main(["check", str(flow_path)])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert output == {
        "valid": False,
        "issues": [
            {
                "code": "unsupported_schema_version",
                "message": "Unsupported flow schema version: '9.9.9'",
                "path": "/schema_version",
            }
        ],
    }


def test_show_cli_returns_the_scene_and_immediate_neighbors(tmp_path: Path, capsys) -> None:
    flow_path = tmp_path / "flow.json"
    document = _draft_flow()
    _write_json(flow_path, document)

    exit_code = main(["show", str(flow_path), "scene-002"])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["shown"] is True
    assert output["scene"] == document["scenes"][1]
    assert output["position"] == 2
    assert output["previous_scene"] == document["scenes"][0]
    assert output["next_scene"] == document["scenes"][2]
    assert output["issues"] == []


def test_show_cli_returns_a_typed_issue_for_an_unknown_scene(tmp_path: Path, capsys) -> None:
    flow_path = tmp_path / "flow.json"
    _write_json(flow_path, _draft_flow())

    exit_code = main(["show", str(flow_path), "scene-999"])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert output["shown"] is False
    assert output["issues"][0]["code"] == "not_found"


def test_apply_cli_atomically_saves_the_updated_document(tmp_path: Path, capsys) -> None:
    script_path = tmp_path / "flow.json"
    patch_path = tmp_path / "patch.json"
    output_path = tmp_path / "updated.json"
    _write_json(script_path, _draft_flow())
    _write_json(
        patch_path,
        [
            {"op": "test", "path": "/revision", "value": 1},
            {"op": "replace", "path": "/revision", "value": 2},
            {"op": "replace", "path": "/title", "value": "更新したCLIの曲"},
        ],
    )

    exit_code = main(["apply", str(script_path), str(patch_path), "--output", str(output_path)])

    output = json.loads(capsys.readouterr().out)
    saved_document = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert output["applied"] is True
    assert output["document"] == saved_document
    assert output["issues"] == []
    assert saved_document["revision"] == 2
    assert saved_document["title"] == "更新したCLIの曲"


def test_approve_cli_atomically_replaces_the_input_draft(tmp_path: Path, capsys) -> None:
    script_path = tmp_path / "flow.json"
    _write_json(script_path, _draft_flow())

    exit_code = main(["approve", str(script_path), "--output", str(script_path)])

    output = json.loads(capsys.readouterr().out)
    saved_document = json.loads(script_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert output["approved"] is True
    assert output["document"] == saved_document
    assert output["issues"] == []
    assert saved_document["status"] == "approved"
    assert len(saved_document["approval"]["content_sha256"]) == 64


def test_check_cli_reports_an_unreadable_input_as_storage_error(tmp_path: Path, capsys) -> None:
    exit_code = main(["check", str(tmp_path / "missing.json")])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert output["issues"][0]["code"] == "storage_error"


def test_check_cli_does_not_infer_a_legacy_script_as_a_flow(tmp_path: Path, capsys) -> None:
    legacy_path = Path("tests/fixtures/scoim/fixed-aba/approved-script.json")

    exit_code = main(["check", str(legacy_path)])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert output["valid"] is False
    assert output["issues"][0]["code"] == "schema_invalid"


def test_apply_cli_rejects_a_patch_that_is_not_an_array(tmp_path: Path, capsys) -> None:
    script_path = tmp_path / "flow.json"
    patch_path = tmp_path / "patch.json"
    output_path = tmp_path / "updated.json"
    _write_json(script_path, _draft_flow())
    _write_json(patch_path, {"op": "replace"})

    exit_code = main(["apply", str(script_path), str(patch_path), "--output", str(output_path)])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert output["applied"] is False
    assert output["issues"][0]["code"] == "patch_invalid"
    assert not output_path.exists()


def test_apply_cli_does_not_overwrite_a_different_output_document(tmp_path: Path, capsys) -> None:
    script_path = tmp_path / "flow.json"
    patch_path = tmp_path / "patch.json"
    output_path = tmp_path / "existing.json"
    _write_json(script_path, _draft_flow())
    existing = _draft_flow()
    existing["document_id"] = "other-flow"
    _write_json(output_path, existing)
    original = output_path.read_bytes()
    _write_json(
        patch_path,
        [
            {"op": "test", "path": "/revision", "value": 1},
            {"op": "replace", "path": "/revision", "value": 2},
        ],
    )

    exit_code = main(["apply", str(script_path), str(patch_path), "--output", str(output_path)])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert output["applied"] is False
    assert output["issues"][0]["code"] == "storage_conflict"
    assert output_path.read_bytes() == original


def test_propose_cli_saves_a_valid_draft_with_an_explicit_model(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    runner_arguments: list[dict[str, object]] = []

    class FakeCodexStructuredRunner:
        def __init__(self, **kwargs: object) -> None:
            runner_arguments.append(kwargs)

        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            return ProposalRun(
                provider="fake",
                model="gpt-5.6-sol",
                model_settings={"reasoning_effort": "medium"},
                started=True,
                terminal_state="completed",
                raw_response=json.dumps(_proposal_response(), ensure_ascii=False).encode(),
                events=None,
                stderr=None,
                issues=(),
            )

    monkeypatch.setattr(
        cli_module,
        "CodexStructuredRunner",
        FakeCodexStructuredRunner,
        raising=False,
    )
    output_dir = tmp_path / "proposal"

    exit_code = main(
        [
            "propose",
            "明るい感じの曲",
            "--document-id",
            "bright_piece",
            "--instrumentation",
            "solo_piano",
            "--duration",
            "180",
            "--model",
            "gpt-5.6-sol",
            "--output",
            str(output_dir),
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["proposed"] is True
    assert output["document"]["document_id"] == "bright_piece"
    assert output["document"]["document_type"] == "flow"
    assert output["document"]["instrumentation"] == "solo_piano"
    assert output["document"]["target_duration"] == 180.0
    assert Path(output["run_dir"]) == output_dir
    assert (output_dir / "outputs" / "flow-draft.json").is_file()
    request = json.loads((output_dir / "inputs" / "request.json").read_text(encoding="utf-8"))
    assert request == {
        "instruction": "明るい感じの曲",
        "document_id": "bright_piece",
        "instrumentation": "solo_piano",
        "target_duration": 180.0,
    }
    assert runner_arguments == [
        {
            "model": "gpt-5.6-sol",
            "working_directory": Path.cwd(),
        }
    ]


def test_propose_cli_returns_a_typed_preflight_failure_without_output(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    class UnauthenticatedRunner:
        def __init__(self, **kwargs: object) -> None:
            pass

        def preflight(self) -> tuple[ValidationIssue, ...]:
            return (
                ValidationIssue(
                    IssueCode.RUNNER_UNAUTHENTICATED,
                    "Codex CLI is not authenticated",
                    "/runner",
                ),
            )

        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            raise AssertionError("run must not be called after a failed preflight")

    monkeypatch.setattr(cli_module, "CodexStructuredRunner", UnauthenticatedRunner)
    output_dir = tmp_path / "proposal"

    exit_code = main(
        [
            "propose",
            "明るい感じの曲",
            "--document-id",
            "bright_piece",
            "--instrumentation",
            "solo_piano",
            "--duration",
            "180",
            "--model",
            "gpt-5.6-sol",
            "--output",
            str(output_dir),
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert output["proposed"] is False
    assert output["run_dir"] is None
    assert output["issues"][0]["code"] == "runner_unauthenticated"
    assert not output_dir.exists()


def test_realize_cli_routes_the_public_contract(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    runner_arguments: list[dict[str, object]] = []
    captured: dict[str, object] = {}

    class FakeCodexRunner:
        def __init__(self, **kwargs: object) -> None:
            runner_arguments.append(kwargs)

    def fake_realize(
        source: Path,
        destination: Path,
        **kwargs: object,
    ) -> PublicRealizationResult:
        captured.update(source=source, destination=destination, **kwargs)
        return PublicRealizationResult(
            True,
            True,
            "flow",
            destination,
            Path("composition"),
            Path("trial"),
            None,
            {"final_smf": "trial/artifacts/final.mid"},
            (),
        )

    monkeypatch.setattr(cli_module, "CodexStructuredRunner", FakeCodexRunner)
    monkeypatch.setattr(cli_module, "realize", fake_realize)
    source = tmp_path / "approved-flow.json"
    output_dir = tmp_path / "output"

    exit_code = main(
        [
            "realize",
            str(source),
            "--model",
            "gpt-5.6-sol",
            "--trial-id",
            "trial-001",
            "--composition-id",
            "composition-001",
            "--output",
            str(output_dir),
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["input_kind"] == "flow"
    assert output["artifacts"]["final_smf"] == "trial/artifacts/final.mid"
    assert runner_arguments == [{"model": "gpt-5.6-sol", "working_directory": Path.cwd()}]
    assert captured.pop("runner").__class__ is FakeCodexRunner
    assert captured == {
        "source": source,
        "destination": output_dir,
        "model": "gpt-5.6-sol",
        "trial_id": "trial-001",
        "composition_id": "composition-001",
        "profile": None,
    }


def test_cli_runs_from_flow_proposal_through_offline_trial_replay(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    calls: list[str] = []
    responses = [_proposal_response(), _response(), *_fixed_aba_stage_responses()]

    class FakeCodexRunner:
        def __init__(self, **kwargs: object) -> None:
            pass

        def preflight(self) -> tuple[ValidationIssue, ...]:
            return ()

        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            calls.append(response_schema_path.name)
            response = responses.pop(0)
            return ProposalRun(
                provider="fake",
                model="gpt-5.6-sol",
                model_settings={"reasoning_effort": "medium"},
                started=True,
                terminal_state="completed",
                raw_response=json.dumps(response, ensure_ascii=False).encode("utf-8"),
                events=b'{"type":"turn.completed"}\n',
                stderr=b"",
                issues=(),
            )

    monkeypatch.setattr(cli_module, "CodexStructuredRunner", FakeCodexRunner)
    proposal_dir = tmp_path / "proposal"
    assert (
        main(
            [
                "propose",
                "穏やかに始まり、明るく広がって戻るピアノ曲",
                "--document-id",
                "end-to-end-flow",
                "--instrumentation",
                "solo_piano",
                "--duration",
                "180",
                "--model",
                "gpt-5.6-sol",
                "--output",
                str(proposal_dir),
            ]
        )
        == 0
    )
    proposal = json.loads(capsys.readouterr().out)
    draft_path = proposal_dir / "outputs" / "flow-draft.json"
    assert main(["check", str(draft_path)]) == 0
    checked = json.loads(capsys.readouterr().out)
    assert main(["show", str(draft_path), "scene-002"]) == 0
    shown = json.loads(capsys.readouterr().out)
    patch_path = tmp_path / "patch.json"
    revised_path = tmp_path / "revised-flow.json"
    _write_json(
        patch_path,
        [
            {"op": "test", "path": "/revision", "value": 1},
            {"op": "replace", "path": "/revision", "value": 2},
        ],
    )
    assert main(["apply", str(draft_path), str(patch_path), "--output", str(revised_path)]) == 0
    applied = json.loads(capsys.readouterr().out)
    approved_path = tmp_path / "approved-flow.json"
    assert main(["approve", str(revised_path), "--output", str(approved_path)]) == 0
    approved = json.loads(capsys.readouterr().out)
    output_dir = tmp_path / "output"
    assert (
        main(
            [
                "realize",
                str(approved_path),
                "--model",
                "gpt-5.6-sol",
                "--trial-id",
                "end-to-end",
                "--profile",
                "solo_piano_3m_v1",
                "--output",
                str(output_dir),
            ]
        )
        == 0
    )
    realized = json.loads(capsys.readouterr().out)
    assert (
        main(
            [
                "realize",
                str(output_dir / "trial"),
                "--output",
                str(tmp_path / "replayed"),
            ]
        )
        == 0
    )
    replayed = json.loads(capsys.readouterr().out)

    assert proposal["proposed"] is True
    assert checked["valid"] is True
    assert shown["scene"]["scene_id"] == "scene-002"
    assert applied["document"]["revision"] == 2
    assert approved["document"]["status"] == "approved"
    assert realized["succeeded"] is True
    assert replayed["succeeded"] is True
    assert responses == []
    assert len(calls) == 10
    assert (output_dir / "trial" / "artifacts" / "phase-04-melody.mid").is_file()
    assert (output_dir / "trial" / "artifacts" / "phase-06-score.mid").is_file()
    assert (output_dir / "trial" / "artifacts" / "final.mid").is_file()
    assert (tmp_path / "replayed" / "artifacts" / "final.mid").is_file()
