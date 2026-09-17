import copy
import json
from pathlib import Path

from scoim.flow_operations import approve_flow
from scoim.proposal import ProposalRun
from scoim.script_compilation import CompilationRequest, _validate_response, compile_script


class FixedRunner:
    def __init__(
        self,
        response: dict[str, object] | bytes | list[dict[str, object] | bytes],
    ) -> None:
        self.responses = response if isinstance(response, list) else [response]
        self.calls = 0

    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        del prompt, response_schema_path
        self.calls += 1
        response = self.responses[self.calls - 1]
        raw_response = (
            response
            if isinstance(response, bytes)
            else json.dumps(response, ensure_ascii=False).encode()
        )
        return ProposalRun(
            provider="fixed",
            model="fixed-model",
            model_settings={},
            started=True,
            terminal_state="completed",
            raw_response=raw_response,
            events=b'{"type":"turn.completed"}\n',
            stderr=b"",
            issues=(),
        )


class FailureRunner:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        del prompt, response_schema_path
        self.calls += 1
        return ProposalRun(
            provider="fixed",
            model="fixed-model",
            model_settings={},
            started=True,
            terminal_state="failed",
            raw_response=None,
            events=b"",
            stderr=b"failure",
            issues=(),
        )


class RecordingRunner(FixedRunner):
    def __init__(self, response: dict[str, object] | list[dict[str, object]]) -> None:
        super().__init__(response)
        self.prompts: list[str] = []
        self.schemas: list[dict[str, object]] = []

    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        self.prompts.append(prompt)
        self.schemas.append(json.loads(response_schema_path.read_text(encoding="utf-8")))
        return super().run(prompt, response_schema_path)


class RepairFailureRunner(FixedRunner):
    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        if self.calls == 0:
            return super().run(prompt, response_schema_path)
        self.calls += 1
        return ProposalRun(
            provider="fixed",
            model="fixed-model",
            model_settings={},
            started=True,
            terminal_state="failed",
            raw_response=None,
            events=b"",
            stderr=b"repair failure",
            issues=(),
        )


def _approved_flow() -> dict[str, object]:
    draft = {
        "document_type": "flow",
        "schema_version": "0.1.0",
        "document_id": "flow-001",
        "revision": 1,
        "status": "draft",
        "approval": None,
        "title": "窓辺の午後",
        "overall_flow": "主題が現れ、景色が広がり、少し変えて戻って閉じる。",
        "instrumentation": "solo_piano",
        "target_duration": 180,
        "scenes": [
            {
                "scene_id": "scene-001",
                "name": "主題",
                "length_class": "short",
                "heard_as": "静かに始まる。",
                "relation_to_previous": "曲の始まり。",
                "transition_to_next": "短いつなぎへ進む。",
            },
            {
                "scene_id": "scene-002",
                "name": "広がり",
                "length_class": "medium",
                "heard_as": "景色が広がる。",
                "relation_to_previous": "少し動きを増す。",
                "transition_to_next": "主題へ戻る。",
            },
            {
                "scene_id": "scene-003",
                "name": "帰還",
                "length_class": "long",
                "heard_as": "主題が変化して戻る。",
                "relation_to_previous": "広がりを受けて落ち着く。",
                "transition_to_next": "余韻を残して閉じる。",
            },
        ],
    }
    result = approve_flow(draft)
    assert result.document is not None
    return result.document


def _response() -> dict[str, object]:
    legacy = json.loads(
        Path("tests/fixtures/scoim/fixed-aba/approved-script.json").read_text(encoding="utf-8")
    )
    script = copy.deepcopy(legacy["script"])
    sections = []
    for section_id, section in sorted(
        script["sections"].items(),
        key=lambda item: (item[1]["parent_section_id"] is not None, item[1]["order"]),
    ):
        transferred = {"id": section_id, **section}
        transferred.pop("order")
        transferred.setdefault("relative_length", None)
        sections.append(transferred)
    directions = []
    for direction_id, direction in script["performance_setup"]["performance_directions"].items():
        relative = direction.get("relative_to")
        directions.append(
            {
                "id": direction_id,
                "target_type": direction["target"]["type"],
                "target_id": direction["target"]["id"],
                "relative_to_type": None if relative is None else relative["type"],
                "relative_to_id": None if relative is None else relative["id"],
                "description": direction["description"],
            }
        )
    transfer = {
        "title": "窓辺の午後",
        "brief": script["brief"],
        "root_section_id": script["root_section_id"],
        "sections": sections,
        "materials": [{"id": key, **value} for key, value in script["materials"].items()],
        "placements": [{"id": key, **value} for key, value in script["placements"].items()],
        "variations": [{"id": key, **value} for key, value in script["variations"].items()],
        "transitions": [{"id": key, **value} for key, value in script["transitions"].items()],
        "performance_directions": directions,
        "requirements": [
            {"id": key, **value} for key, value in script.get("requirements", {}).items()
        ],
    }
    return {
        "script": transfer,
        "scene_section_mappings": [
            {"scene_id": "scene-001", "section_ids": ["statement"]},
            {"scene_id": "scene-002", "section_ids": ["bridge", "contrast"]},
            {"scene_id": "scene-003", "section_ids": ["return", "release"]},
        ],
    }


def test_compile_script_uses_a_codex_supported_output_schema(tmp_path: Path) -> None:
    runner = RecordingRunner(_response())

    result = compile_script(
        CompilationRequest(_approved_flow(), "composition-001", "solo_piano_3m_v1"),
        runner,
        tmp_path / "run",
    )

    assert result.compiled is True
    assert "uniqueItems" not in json.dumps(runner.schemas[0])


def test_compile_script_rejects_duplicate_scene_section_ids_in_python() -> None:
    response = _response()
    response["scene_section_mappings"][0]["section_ids"] = ["statement", "statement"]

    result = _validate_response(
        CompilationRequest(_approved_flow(), "composition-001", "solo_piano_3m_v1"),
        response,
    )

    assert result.compiled is False
    assert any("mapped once" in issue.message for issue in result.issues)


def test_compilation_validation_reports_an_unknown_scene_without_raising() -> None:
    response = _response()
    response["scene_section_mappings"][0]["scene_id"] = "unknown-scene"

    result = _validate_response(
        CompilationRequest(_approved_flow(), "composition-001", "solo_piano_3m_v1"),
        response,
    )

    assert result.compiled is False
    assert any("mapped once and in order" in issue.message for issue in result.issues)


def test_compile_script_creates_a_validated_document_and_frozen_call_budget(tmp_path: Path) -> None:
    runner = FixedRunner(_response())

    result = compile_script(
        CompilationRequest(_approved_flow(), "composition-001", "solo_piano_3m_v1"),
        runner,
        tmp_path / "run",
    )

    assert result.compiled is True
    assert result.document is not None
    assert result.document["status"] == "validated"
    assert result.document["source_flow"]["document_id"] == "flow-001"
    assert result.metrics is not None
    assert result.metrics.normal_calls == 8
    assert result.metrics.maximum_calls == 16
    assert runner.calls == 1
    assert len(result.projection_targets) > 0


def test_compile_script_repairs_a_missing_return_relation_with_a_full_response_once(
    tmp_path: Path,
) -> None:
    response = _response()
    response["script"]["variations"] = []
    runner = FixedRunner([response, _response()])

    result = compile_script(
        CompilationRequest(_approved_flow(), "composition-002", "solo_piano_3m_v1"),
        runner,
        tmp_path / "run",
    )

    assert result.compiled is True
    assert result.response_record["attempts"] == 2
    assert runner.calls == 2


def test_compile_script_stops_after_one_unsuccessful_full_response_repair(
    tmp_path: Path,
) -> None:
    response = _response()
    response["script"]["variations"] = []
    second_response = _response()
    second_response["script"]["variations"] = []
    runner = FixedRunner([response, second_response])

    result = compile_script(
        CompilationRequest(_approved_flow(), "composition-006", "solo_piano_3m_v1"),
        runner,
        tmp_path / "run",
    )

    assert result.compiled is False
    assert result.response_record["attempts"] == 2
    assert runner.calls == 2


def test_compile_script_repairs_multiple_content_problems_with_a_full_response(
    tmp_path: Path,
) -> None:
    response = _response()
    response["script"]["variations"] = []
    response["scene_section_mappings"][0]["section_ids"] = ["contrast"]
    runner = RecordingRunner([response, _response()])

    result = compile_script(
        CompilationRequest(_approved_flow(), "composition-003", "solo_piano_3m_v1"),
        runner,
        tmp_path / "run",
    )

    assert result.compiled is True
    assert runner.calls == 2
    assert runner.schemas[1] == runner.schemas[0]
    assert "承認済み楽曲台本" in runner.prompts[1]
    assert "元の応答" in runner.prompts[1]
    assert "変更しない" in runner.prompts[1]
    assert "validation_issues" in runner.prompts[1]
    assert len(result.response_record["initial_issues"]) >= 2
    for issue in result.response_record["initial_issues"]:
        assert issue["message"] in runner.prompts[1]


def test_compile_script_repairs_invalid_json_with_a_full_response(tmp_path: Path) -> None:
    runner = FixedRunner([b"not json", _response()])

    result = compile_script(
        CompilationRequest(_approved_flow(), "composition-003", "solo_piano_3m_v1"),
        runner,
        tmp_path / "run",
    )

    assert result.compiled is True
    assert result.response_record["attempts"] == 2
    assert result.response_record["initial_issues"][0]["code"] == "model_output_invalid"
    assert runner.calls == 2


def test_compile_script_repairs_non_object_json_with_a_full_response(tmp_path: Path) -> None:
    runner = FixedRunner([b"[]", _response()])

    result = compile_script(
        CompilationRequest(_approved_flow(), "composition-003", "solo_piano_3m_v1"),
        runner,
        tmp_path / "run",
    )

    assert result.compiled is True
    assert result.response_record["attempts"] == 2
    assert result.response_record["initial_issues"][0]["code"] == "model_output_invalid"
    assert runner.calls == 2


def test_compile_script_stops_when_the_single_repair_call_fails(tmp_path: Path) -> None:
    response = _response()
    response["script"]["variations"] = []
    runner = RepairFailureRunner(response)

    result = compile_script(
        CompilationRequest(_approved_flow(), "composition-003", "solo_piano_3m_v1"),
        runner,
        tmp_path / "run",
    )

    assert result.compiled is False
    assert result.response_record["attempts"] == 2
    assert len(result.response_record["attempt_paths"]) == 2
    assert result.issues[0].code.value == "runner_failed"
    assert runner.calls == 2


def test_compile_script_does_not_hash_json_null_when_both_responses_are_invalid(
    tmp_path: Path,
) -> None:
    runner = FixedRunner([b"not json", b"still not json"])

    result = compile_script(
        CompilationRequest(_approved_flow(), "composition-003", "solo_piano_3m_v1"),
        runner,
        tmp_path / "run",
    )

    assert result.compiled is False
    assert "response_sha256" not in result.response_record
    assert result.response_record["attempts"] == 2
    assert runner.calls == 2


def test_compile_script_rejects_scene_lengths_that_reverse_the_flow_classes() -> None:
    response = _response()
    response["scene_section_mappings"] = [
        {"scene_id": "scene-001", "section_ids": ["statement", "bridge"]},
        {"scene_id": "scene-002", "section_ids": ["contrast"]},
        {"scene_id": "scene-003", "section_ids": ["return", "release"]},
    ]
    result = _validate_response(
        CompilationRequest(_approved_flow(), "composition-004", "solo_piano_3m_v1"),
        response,
    )

    assert result.compiled is False
    assert any("strictly ordered" in issue.message for issue in result.issues)


def test_projection_ledger_contains_each_human_flow_field_once(tmp_path: Path) -> None:
    result = compile_script(
        CompilationRequest(_approved_flow(), "composition-005", "solo_piano_3m_v1"),
        FixedRunner(_response()),
        tmp_path / "run",
    )

    flow_paths = [
        target["source_path"]
        for target in result.projection_targets
        if not target["source_path"].startswith("/script/")
    ]
    assert len(flow_paths) == len(set(flow_paths)) == 22


def test_compile_script_rejects_a_title_that_changes_the_approved_flow() -> None:
    response = _response()
    response["script"]["title"] = "別の題名"

    result = _validate_response(
        CompilationRequest(_approved_flow(), "composition-007", "solo_piano_3m_v1"),
        response,
    )

    assert result.compiled is False
    assert any(issue.path == "/script/title" for issue in result.issues)


def test_compile_script_records_a_runner_failure_after_start(tmp_path: Path) -> None:
    runner = FailureRunner()

    result = compile_script(
        CompilationRequest(_approved_flow(), "composition-008", "solo_piano_3m_v1"),
        runner,
        tmp_path / "run",
    )

    assert result.compiled is False
    assert result.run_dir == tmp_path / "run"
    assert runner.calls == 1


def test_compile_script_rejects_an_invalid_transfer_schema(tmp_path: Path) -> None:
    response = _response()
    del response["script"]["title"]
    runner = FixedRunner([response, response])

    result = compile_script(
        CompilationRequest(_approved_flow(), "composition-009", "solo_piano_3m_v1"),
        runner,
        tmp_path / "run",
    )

    assert result.compiled is False
    assert result.issues[0].code.value == "model_output_invalid"
    assert result.response_record["attempts"] == 2
    assert runner.calls == 2


def test_compile_script_rejects_duplicate_transfer_ids() -> None:
    response = _response()
    response["script"]["materials"].append(copy.deepcopy(response["script"]["materials"][0]))

    result = _validate_response(
        CompilationRequest(_approved_flow(), "composition-010", "solo_piano_3m_v1"),
        response,
    )

    assert result.compiled is False
    assert "Duplicate ID" in result.issues[0].message


def test_compile_script_rejects_a_malformed_full_response_repair(tmp_path: Path) -> None:
    response = _response()
    response["script"]["variations"] = []
    runner = FixedRunner([response, {"script": {"title": "incomplete"}}])

    result = compile_script(
        CompilationRequest(_approved_flow(), "composition-011", "solo_piano_3m_v1"),
        runner,
        tmp_path / "run",
    )

    assert result.compiled is False
    assert runner.calls == 2


def test_compile_script_rejects_an_existing_run_before_a_model_call(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    runner = FixedRunner(_response())

    result = compile_script(
        CompilationRequest(_approved_flow(), "composition-012", "solo_piano_3m_v1"),
        runner,
        run_dir,
    )

    assert result.run_dir is None
    assert runner.calls == 0
