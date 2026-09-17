import hashlib
import json
import math
from importlib.resources import as_file, files
from pathlib import Path
from typing import NoReturn

import scoim.proposal as proposal_module
from scoim.operations import apply_patch, approve
from scoim.proposal import ProposalRequest, ProposalRun, propose_script
from scoim.validation import IssueCode, ValidationIssue


def _valid_response() -> dict[str, object]:
    return {
        "title": "夕暮れの帰還",
        "brief": "静かな夕暮れから景色が開け、最後は冒頭へ戻る。",
        "root_section_id": "whole",
        "sections": [
            {
                "id": "whole",
                "parent_section_id": None,
                "role": "whole",
                "relative_length": None,
                "description": "曲全体。",
            },
            {
                "id": "a",
                "parent_section_id": "whole",
                "role": "statement",
                "relative_length": 4,
                "description": "主題を提示する。",
            },
            {
                "id": "bridge",
                "parent_section_id": "whole",
                "role": "transition",
                "relative_length": 1,
                "description": "対照部へ導く。",
            },
            {
                "id": "b",
                "parent_section_id": "whole",
                "role": "contrast",
                "relative_length": 4,
                "description": "景色が開ける。",
            },
        ],
        "materials": [
            {"id": "theme", "kind": "theme", "description": "夕暮れの中心素材。"},
            {"id": "fill", "kind": "transition", "description": "場面を結ぶ素材。"},
            {"id": "contrast", "kind": "contrast", "description": "開けた対照素材。"},
        ],
        "placements": [
            {"id": "theme_a", "section_id": "a", "material_id": "theme"},
            {"id": "fill_ab", "section_id": "bridge", "material_id": "fill"},
            {"id": "contrast_b", "section_id": "b", "material_id": "contrast"},
        ],
        "variations": [],
        "transitions": [
            {
                "id": "a_to_b",
                "from_placement_id": "theme_a",
                "connector_placement_id": "fill_ab",
                "to_placement_id": "contrast_b",
                "description": "主題から対照部へ移る。",
            }
        ],
        "performance_directions": [
            {
                "id": "quiet_opening",
                "target_type": "section",
                "target_id": "a",
                "relative_to_type": None,
                "relative_to_id": None,
                "description": "静かに、少し時間を揺らす。",
            }
        ],
        "requirements": [],
    }


def _profile_response() -> dict[str, object]:
    response = _valid_response()
    sections = response["sections"]
    materials = response["materials"]
    placements = response["placements"]
    assert isinstance(sections, list)
    assert isinstance(materials, list)
    assert isinstance(placements, list)
    sections.append(
        {
            "id": "release",
            "parent_section_id": "whole",
            "role": "release",
            "relative_length": 2,
            "description": "十分な余韻で閉じる。",
        }
    )
    materials.append({"id": "cadence", "kind": "ending", "description": "終止素材。"})
    placements.append({"id": "cadence_last", "section_id": "release", "material_id": "cadence"})
    return response


class FakeRunner:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[tuple[str, Path]] = []
        self.schemas: list[dict[str, object]] = []

    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        self.calls.append((prompt, response_schema_path))
        self.schemas.append(json.loads(response_schema_path.read_text(encoding="utf-8")))
        return ProposalRun(
            provider="fake",
            model="fake-model",
            model_settings={"reasoning_effort": "low"},
            started=True,
            terminal_state="completed",
            raw_response=json.dumps(self.response, ensure_ascii=False).encode("utf-8"),
            events=None,
            stderr=None,
            issues=(),
        )


def test_profile_aware_proposal_constrains_profile_vocabulary(tmp_path: Path) -> None:
    runner = FakeRunner(_profile_response())
    request = ProposalRequest(
        instruction="静かな夜から頂点へ進み、余韻へほどける曲",
        document_id="night_arc",
        instrumentation="solo_piano",
        target_duration_seconds=180,
        target_profile="solo_piano_3m_v1",
    )

    result = propose_script(request, runner, tmp_path / "proposal")

    assert result.proposed is True
    section_role = runner.schemas[0]["$defs"]["section"]["properties"]["role"]
    material_kind = runner.schemas[0]["$defs"]["material"]["properties"]["kind"]
    assert section_role == {
        "type": "string",
        "enum": [
            "climax",
            "contrast",
            "opening",
            "release",
            "return",
            "statement",
            "transition",
            "variation",
            "whole",
        ],
    }
    assert material_kind == {
        "type": "string",
        "enum": ["contrast", "ending", "theme", "transition"],
    }
    assert "最後のrelease" in runner.calls[0][0]
    assert "requirementsへ使う演奏指示" in runner.calls[0][0]
    assert "同じ階層の深さ" in runner.calls[0][0]
    assert "同じ素材" in runner.calls[0][0]
    assert "すべての素材" in runner.calls[0][0]
    assert "一つの変奏元" in runner.calls[0][0]
    assert (
        json.loads((tmp_path / "proposal" / "request.json").read_text(encoding="utf-8"))[
            "target_profile"
        ]
        == "solo_piano_3m_v1"
    )


def test_profile_aware_proposal_rejects_conflicting_setup_before_model_call(
    tmp_path: Path,
) -> None:
    runner = FakeRunner(_profile_response())
    request = ProposalRequest(
        instruction="弦楽四重奏曲",
        document_id="wrong_setup",
        instrumentation="string_quartet",
        target_duration_seconds=180,
        target_profile="solo_piano_3m_v1",
    )

    result = propose_script(request, runner, tmp_path / "proposal")

    assert result.proposed is False
    assert result.issues[0].code is IssueCode.UNSUPPORTED_PROFILE
    assert result.issues[0].path == "/target_profile"
    assert runner.calls == []
    assert not (tmp_path / "proposal").exists()


def test_profile_aware_proposal_records_an_incompatible_structure(
    tmp_path: Path,
) -> None:
    runner = FakeRunner(_valid_response())
    request = ProposalRequest(
        instruction="静かな夜から頂点へ進み、余韻へほどける曲",
        document_id="night_arc",
        instrumentation="solo_piano",
        target_duration_seconds=180,
        target_profile="solo_piano_3m_v1",
    )

    result = propose_script(request, runner, tmp_path / "proposal")

    assert result.proposed is False
    assert result.proposal_path == tmp_path / "proposal"
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert result.issues[0].path == "/script/sections/b/role"


class FakeRawRunner:
    def __init__(self, raw_response: bytes | None) -> None:
        self.raw_response = raw_response

    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        return ProposalRun(
            provider="fake",
            model="fake-model",
            model_settings={},
            started=True,
            terminal_state="completed",
            raw_response=self.raw_response,
            events=b'{"type":"turn.completed"}\n',
            stderr=b"",
            issues=(),
        )


class FakeFailureRunner:
    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        return ProposalRun(
            provider="fake",
            model="fake-model",
            model_settings={},
            started=True,
            terminal_state="failed",
            raw_response=None,
            events=b'{"type":"turn.failed"}\n',
            stderr=b"runner failed\n",
            issues=(
                ValidationIssue(
                    IssueCode.RUNNER_FAILED,
                    "The proposal runner failed",
                    "/runner",
                ),
            ),
        )


class FakePreflightFailureRunner:
    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        return ProposalRun(
            provider="fake",
            model="fake-model",
            model_settings={},
            started=False,
            terminal_state="failed",
            raw_response=None,
            events=None,
            stderr=None,
            issues=(
                ValidationIssue(
                    IssueCode.RUNNER_FAILED,
                    "The proposal runner preflight failed",
                    "/runner",
                ),
            ),
        )


def test_proposal_response_schema_does_not_mix_ref_with_sibling_keywords() -> None:
    schema_resource = files("scoim").joinpath("schemas", "proposal-response-1.schema.json")
    with as_file(schema_resource) as schema_path:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

    def check_node(value: object) -> None:
        if isinstance(value, dict):
            if "$ref" in value:
                assert set(value) == {"$ref"}
            for member in value.values():
                check_node(member)
        elif isinstance(value, list):
            for member in value:
                check_node(member)

    check_node(schema)


def test_propose_script_normalizes_a_valid_runner_response_to_a_draft(tmp_path: Path) -> None:
    runner = FakeRunner(_valid_response())
    request = ProposalRequest(
        instruction="明るい夕暮れの曲",
        document_id="evening_return",
        instrumentation="solo_piano",
        target_duration_seconds=180,
    )

    result = propose_script(request, runner, tmp_path / "proposal")

    assert result.proposed is True
    assert result.document is not None
    assert result.document["schema_version"] == "0.1.0"
    assert result.document["document_id"] == "evening_return"
    assert result.document["revision"] == 1
    assert result.document["status"] == "draft"
    assert result.document["approval"] is None
    script = result.document["script"]
    assert isinstance(script, dict)
    assert script["performance_setup"] == {
        "instrumentation": "solo_piano",
        "target_duration_seconds": 180,
        "performance_directions": {
            "quiet_opening": {
                "target": {"type": "section", "id": "a"},
                "description": "静かに、少し時間を揺らす。",
            }
        },
    }
    assert script["requirements"] == {}
    assert "明るい夕暮れの曲" in runner.calls[0][0]
    assert runner.calls[0][1].name == "proposal-response-1.schema.json"


def test_propose_script_derives_sibling_order_from_section_array_position(
    tmp_path: Path,
) -> None:
    runner = FakeRunner(_valid_response())
    request = ProposalRequest(
        instruction="明るい夕暮れの曲",
        document_id="array_order_piece",
        instrumentation="solo_piano",
        target_duration_seconds=180,
    )

    result = propose_script(request, runner, tmp_path / "proposal")

    assert result.proposed is True
    assert result.document is not None
    sections = result.document["script"]["sections"]
    assert sections["whole"]["order"] == 0
    assert sections["a"]["order"] == 0
    assert sections["bridge"]["order"] == 1
    assert sections["b"]["order"] == 2


def test_propose_script_derives_branch_length_from_its_children(tmp_path: Path) -> None:
    response = _valid_response()
    response["sections"][0]["relative_length"] = 1
    runner = FakeRunner(response)
    request = ProposalRequest(
        instruction="明るい夕暮れの曲",
        document_id="derived_branch_length",
        instrumentation="solo_piano",
        target_duration_seconds=180,
    )

    result = propose_script(request, runner, tmp_path / "proposal")

    assert result.proposed is True
    assert result.document is not None
    sections = result.document["script"]["sections"]
    assert "relative_length" not in sections["whole"]
    assert sections["a"]["relative_length"] == 4
    assert result.issues == ()


def test_propose_script_normalizes_a_typed_requirement_to_an_id_keyed_object(
    tmp_path: Path,
) -> None:
    response = _valid_response()
    directions = response["performance_directions"]
    assert isinstance(directions, list)
    directions.append(
        {
            "id": "stronger_contrast",
            "target_type": "section",
            "target_id": "b",
            "relative_to_type": "section",
            "relative_to_id": "a",
            "description": "対照部を提示部より強くする。",
        }
    )
    requirements = response["requirements"]
    assert isinstance(requirements, list)
    requirements.append(
        {
            "id": "contrast_louder",
            "performance_direction_id": "stronger_contrast",
            "feature": "loudness",
            "relation": "more",
        }
    )
    runner = FakeRunner(response)
    request = ProposalRequest("対照部を強く", "typed_piece", "solo_piano", 180)

    result = propose_script(request, runner, tmp_path / "proposal")

    assert result.proposed is True
    assert result.document is not None
    assert result.document["script"]["requirements"] == {
        "contrast_louder": {
            "performance_direction_id": "stronger_contrast",
            "feature": "loudness",
            "relation": "more",
        }
    }


def test_propose_script_prompt_requires_a_dedicated_transition_leaf(tmp_path: Path) -> None:
    runner = FakeRunner(_valid_response())
    request = ProposalRequest(
        instruction="明るい夕暮れの曲",
        document_id="transition_leaf_guidance",
        instrumentation="solo_piano",
        target_duration_seconds=180,
    )

    result = propose_script(request, runner, tmp_path / "proposal")

    assert result.proposed is True
    prompt = runner.calls[0][0]
    assert "つなぎ素材の配置だけを持つ専用の葉区分" in prompt
    assert "前の配置、つなぎ配置、次の配置" in prompt


def test_propose_script_prompt_only_types_explicit_comparisons(tmp_path: Path) -> None:
    runner = FakeRunner(_valid_response())
    request = ProposalRequest("静かな曲", "typed_guidance", "solo_piano", 180)

    result = propose_script(request, runner, tmp_path / "proposal")

    assert result.proposed is True
    prompt = runner.calls[0][0]
    assert "onset_alignment" in prompt
    assert "loudness" in prompt
    assert "more" in prompt
    assert "less" in prompt
    assert "曖昧な指示を推測で変換せず" in prompt


def test_propose_script_saves_the_complete_success_record(tmp_path: Path) -> None:
    response = _valid_response()
    runner = FakeRunner(response)
    request = ProposalRequest(
        instruction="明るい夕暮れの曲",
        document_id="evening_return",
        instrumentation="solo_piano",
        target_duration_seconds=180,
    )
    output_dir = tmp_path / "proposal"

    result = propose_script(request, runner, output_dir)

    assert result.proposal_path == output_dir
    assert result.draft_path == output_dir / "normalized-draft.json"
    expected_files = {
        "manifest.json",
        "request.json",
        "prompt.md",
        "response-schema.json",
        "raw-response.txt",
        "normalized-draft.json",
        "validation.json",
    }
    assert {path.name for path in output_dir.iterdir()} == expected_files
    assert (output_dir / "raw-response.txt").read_bytes() == json.dumps(
        response, ensure_ascii=False
    ).encode("utf-8")
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["runner"] == {
        "provider": "fake",
        "model": "fake-model",
        "model_settings": {"reasoning_effort": "low"},
        "terminal_state": "completed",
    }
    assert manifest["draft"] == {
        "document_id": "evening_return",
        "revision": 1,
        "content_sha256": result.content_sha256,
    }
    for relative_path, expected_hash in manifest["files"].items():
        actual_hash = hashlib.sha256((output_dir / relative_path).read_bytes()).hexdigest()
        assert actual_hash == expected_hash
    assert "manifest.json" not in manifest["files"]


def test_propose_script_rejects_an_empty_instruction_before_calling_runner(
    tmp_path: Path,
) -> None:
    runner = FakeRunner(_valid_response())
    output_dir = tmp_path / "proposal"

    result = propose_script(
        ProposalRequest(
            instruction="   ",
            document_id="evening_return",
            instrumentation="solo_piano",
            target_duration_seconds=180,
        ),
        runner,
        output_dir,
    )

    assert result.proposed is False
    assert result.issues[0].code.value == "schema_invalid"
    assert result.issues[0].path == "/instruction"
    assert runner.calls == []
    assert output_dir.exists() is False


def test_propose_script_rejects_an_invalid_document_id_before_calling_runner(
    tmp_path: Path,
) -> None:
    runner = FakeRunner(_valid_response())

    result = propose_script(
        ProposalRequest(
            instruction="明るい曲",
            document_id="Evening Return",
            instrumentation="solo_piano",
            target_duration_seconds=180,
        ),
        runner,
        tmp_path / "proposal",
    )

    assert result.proposed is False
    assert result.issues[0].code.value == "schema_invalid"
    assert result.issues[0].path == "/document_id"
    assert runner.calls == []


def test_propose_script_rejects_an_invalid_instrumentation_before_calling_runner(
    tmp_path: Path,
) -> None:
    runner = FakeRunner(_valid_response())

    result = propose_script(
        ProposalRequest(
            instruction="明るい曲",
            document_id="evening_return",
            instrumentation="Solo Piano",
            target_duration_seconds=180,
        ),
        runner,
        tmp_path / "proposal",
    )

    assert result.proposed is False
    assert result.issues[0].code.value == "schema_invalid"
    assert result.issues[0].path == "/instrumentation"
    assert runner.calls == []


def test_propose_script_rejects_a_nonpositive_duration_before_calling_runner(
    tmp_path: Path,
) -> None:
    runner = FakeRunner(_valid_response())

    result = propose_script(
        ProposalRequest(
            instruction="明るい曲",
            document_id="evening_return",
            instrumentation="solo_piano",
            target_duration_seconds=0,
        ),
        runner,
        tmp_path / "proposal",
    )

    assert result.proposed is False
    assert result.issues[0].code.value == "schema_invalid"
    assert result.issues[0].path == "/target_duration_seconds"
    assert runner.calls == []


def test_propose_script_saves_an_invalid_json_response_as_a_failed_record(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "proposal"

    result = propose_script(
        ProposalRequest(
            instruction="明るい曲",
            document_id="bright_piece",
            instrumentation="solo_piano",
            target_duration_seconds=180,
        ),
        FakeRawRunner(b"not json"),
        output_dir,
    )

    assert result.proposed is False
    assert result.proposal_path == output_dir
    assert result.draft_path is None
    assert result.issues[0].code.value == "model_output_invalid"
    assert (output_dir / "raw-response.txt").read_bytes() == b"not json"
    assert (output_dir / "normalized-draft.json").exists() is False
    assert (output_dir / "stderr.log").read_bytes() == b""
    assert (output_dir / "runner-events.jsonl").read_bytes() == b'{"type":"turn.completed"}\n'
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    validation = json.loads((output_dir / "validation.json").read_text(encoding="utf-8"))
    assert validation["valid"] is False
    assert validation["issues"][0]["code"] == "model_output_invalid"


def test_propose_script_rejects_duplicate_ids_in_a_transfer_collection(
    tmp_path: Path,
) -> None:
    response = _valid_response()
    sections = response["sections"]
    assert isinstance(sections, list)
    sections.append(dict(sections[1]))

    result = propose_script(
        ProposalRequest(
            instruction="明るい曲",
            document_id="bright_piece",
            instrumentation="solo_piano",
            target_duration_seconds=180,
        ),
        FakeRunner(response),
        tmp_path / "proposal",
    )

    assert result.proposed is False
    assert result.issues[0].code.value == "model_output_invalid"
    assert result.issues[0].path == "/sections/4/id"
    assert (tmp_path / "proposal" / "normalized-draft.json").exists() is False


def test_propose_script_rejects_duplicate_requirement_ids(tmp_path: Path) -> None:
    response = _valid_response()
    requirements = response["requirements"]
    assert isinstance(requirements, list)
    requirement = {
        "id": "same_requirement",
        "performance_direction_id": "quiet_opening",
        "feature": "loudness",
        "relation": "less",
    }
    requirements.extend((requirement, dict(requirement)))

    result = propose_script(
        ProposalRequest("静かな曲", "duplicate_requirement", "solo_piano", 180),
        FakeRunner(response),
        tmp_path / "proposal",
    )

    assert result.proposed is False
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert result.issues[0].path == "/requirements/1/id"


def test_propose_script_rejects_a_partial_relative_direction_reference(
    tmp_path: Path,
) -> None:
    response = _valid_response()
    directions = response["performance_directions"]
    assert isinstance(directions, list)
    direction = directions[0]
    assert isinstance(direction, dict)
    direction["relative_to_type"] = "section"

    result = propose_script(
        ProposalRequest(
            instruction="明るい曲",
            document_id="bright_piece",
            instrumentation="solo_piano",
            target_duration_seconds=180,
        ),
        FakeRunner(response),
        tmp_path / "proposal",
    )

    assert result.proposed is False
    assert result.issues[0].code.value == "model_output_invalid"
    assert result.issues[0].path == "/performance_directions/0/relative_to_id"


def test_propose_script_rejects_a_response_that_violates_the_transfer_schema(
    tmp_path: Path,
) -> None:
    response = _valid_response()
    del response["title"]

    result = propose_script(
        ProposalRequest(
            instruction="明るい曲",
            document_id="bright_piece",
            instrumentation="solo_piano",
            target_duration_seconds=180,
        ),
        FakeRunner(response),
        tmp_path / "proposal",
    )

    assert result.proposed is False
    assert result.issues[0].code.value == "model_output_invalid"
    assert result.issues[0].path == ""
    assert (tmp_path / "proposal" / "raw-response.txt").is_file()


def test_propose_script_rejects_an_existing_output_before_calling_runner(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "proposal"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    runner = FakeRunner(_valid_response())

    result = propose_script(
        ProposalRequest(
            instruction="明るい曲",
            document_id="bright_piece",
            instrumentation="solo_piano",
            target_duration_seconds=180,
        ),
        runner,
        output_dir,
    )

    assert result.proposed is False
    assert result.issues[0].code.value == "storage_conflict"
    assert result.issues[0].path == "/output_dir"
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert runner.calls == []


def test_propose_script_reports_storage_error_without_leaving_partial_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "proposal"

    def fail_replace(source: object, destination: object) -> NoReturn:
        raise OSError("write failed")

    monkeypatch.setattr(proposal_module.os, "replace", fail_replace)

    result = propose_script(
        ProposalRequest(
            instruction="明るい曲",
            document_id="bright_piece",
            instrumentation="solo_piano",
            target_duration_seconds=180,
        ),
        FakeRunner(_valid_response()),
        output_dir,
    )

    assert result.proposed is False
    assert result.issues[0].code.value == "storage_error"
    assert result.issues[0].path == "/output_dir"
    assert output_dir.exists() is False
    assert list(tmp_path.iterdir()) == []


def test_proposed_draft_can_be_patched_and_explicitly_approved(tmp_path: Path) -> None:
    proposal = propose_script(
        ProposalRequest(
            instruction="明るい曲",
            document_id="bright_piece",
            instrumentation="solo_piano",
            target_duration_seconds=180,
        ),
        FakeRunner(_valid_response()),
        tmp_path / "proposal",
    )
    assert proposal.document is not None

    applied = apply_patch(
        proposal.document,
        [
            {"op": "test", "path": "/revision", "value": 1},
            {"op": "replace", "path": "/revision", "value": 2},
            {"op": "replace", "path": "/script/title", "value": "光の帰還"},
        ],
    )
    assert applied.document is not None
    approved = approve(applied.document)

    assert approved.approved is True
    assert approved.document is not None
    assert approved.document["status"] == "approved"
    assert approved.document["revision"] == 2
    assert approved.document["approval"] is not None


def test_propose_script_preserves_a_complete_relative_direction_reference(
    tmp_path: Path,
) -> None:
    response = _valid_response()
    directions = response["performance_directions"]
    assert isinstance(directions, list)
    direction = directions[0]
    assert isinstance(direction, dict)
    direction["relative_to_type"] = "section"
    direction["relative_to_id"] = "b"

    result = propose_script(
        ProposalRequest("明るい曲", "bright_piece", "solo_piano", 180),
        FakeRunner(response),
        tmp_path / "proposal",
    )

    assert result.document is not None
    script = result.document["script"]
    assert isinstance(script, dict)
    setup = script["performance_setup"]
    assert isinstance(setup, dict)
    saved_directions = setup["performance_directions"]
    assert isinstance(saved_directions, dict)
    assert saved_directions["quiet_opening"]["relative_to"] == {
        "type": "section",
        "id": "b",
    }


def test_propose_script_records_a_missing_final_response(tmp_path: Path) -> None:
    output_dir = tmp_path / "proposal"

    result = propose_script(
        ProposalRequest("明るい曲", "bright_piece", "solo_piano", 180),
        FakeRawRunner(None),
        output_dir,
    )

    assert result.proposed is False
    assert result.issues[0].code.value == "model_output_invalid"
    assert (output_dir / "raw-response.txt").exists() is False
    assert (output_dir / "manifest.json").is_file()


def test_propose_script_records_a_semantically_invalid_normalized_draft(
    tmp_path: Path,
) -> None:
    response = _valid_response()
    placements = response["placements"]
    assert isinstance(placements, list)
    placement = placements[0]
    assert isinstance(placement, dict)
    placement["material_id"] = "missing"

    result = propose_script(
        ProposalRequest("明るい曲", "bright_piece", "solo_piano", 180),
        FakeRunner(response),
        tmp_path / "proposal",
    )

    assert result.proposed is False
    assert result.issues[0].code.value == "model_output_invalid"
    assert result.issues[0].path == "/script/placements/theme_a/material_id"
    assert (tmp_path / "proposal" / "normalized-draft.json").exists() is False


def test_propose_script_records_a_runner_failure_after_execution_started(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "proposal"

    result = propose_script(
        ProposalRequest("明るい曲", "bright_piece", "solo_piano", 180),
        FakeFailureRunner(),
        output_dir,
    )

    assert result.proposed is False
    assert result.proposal_path == output_dir
    assert result.issues[0].code is IssueCode.RUNNER_FAILED
    assert (output_dir / "stderr.log").read_bytes() == b"runner failed\n"
    assert (output_dir / "runner-events.jsonl").read_bytes() == b'{"type":"turn.failed"}\n'
    assert (output_dir / "raw-response.txt").exists() is False


def test_propose_script_rejects_a_nonfinite_duration_before_calling_runner(
    tmp_path: Path,
) -> None:
    runner = FakeRunner(_valid_response())

    result = propose_script(
        ProposalRequest("明るい曲", "bright_piece", "solo_piano", math.nan),
        runner,
        tmp_path / "proposal",
    )

    assert result.proposed is False
    assert result.issues[0].code.value == "schema_invalid"
    assert result.issues[0].path == "/target_duration_seconds"
    assert runner.calls == []


def test_propose_script_does_not_create_a_record_for_a_preflight_failure(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "proposal"

    result = propose_script(
        ProposalRequest("明るい曲", "bright_piece", "solo_piano", 180),
        FakePreflightFailureRunner(),
        output_dir,
    )

    assert result.proposed is False
    assert result.proposal_path is None
    assert result.issues[0].code is IssueCode.RUNNER_FAILED
    assert output_dir.exists() is False


def test_propose_script_reports_storage_error_while_saving_a_failed_record(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail_replace(source: object, destination: object) -> NoReturn:
        raise OSError("write failed")

    monkeypatch.setattr(proposal_module.os, "replace", fail_replace)

    result = propose_script(
        ProposalRequest("明るい曲", "bright_piece", "solo_piano", 180),
        FakeRawRunner(b"not json"),
        tmp_path / "proposal",
    )

    assert result.proposed is False
    assert result.issues[0].code is IssueCode.STORAGE_ERROR
    assert list(tmp_path.iterdir()) == []
