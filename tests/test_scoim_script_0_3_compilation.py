import json
from pathlib import Path

from scoim.flow_operations import approve_flow
from scoim.proposal import ProposalRun
from scoim.score_projection import PlanChoice, build_piece_plan
from scoim.script_0_3_compilation import (
    Script03CompilationRequest,
    compile_script_0_3,
)
from scoim.validation import IssueCode, ValidationIssue

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "scoim" / "script-0.3-compilation"


def _fixture(name: str) -> dict[str, object]:
    return json.loads((_FIXTURE_DIR / name).read_text(encoding="utf-8"))


class SequencedRunner:
    def __init__(self, responses: list[dict[str, object] | bytes]) -> None:
        self.responses = responses
        self.prompts: list[str] = []
        self.schemas: list[dict[str, object]] = []

    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        self.prompts.append(prompt)
        self.schemas.append(json.loads(response_schema_path.read_text(encoding="utf-8")))
        response = self.responses[len(self.prompts) - 1]
        raw_response = (
            response
            if isinstance(response, bytes)
            else json.dumps(response, ensure_ascii=False).encode("utf-8")
        )
        return ProposalRun(
            provider="fixed",
            model="fixed",
            model_settings={},
            started=True,
            terminal_state="completed",
            raw_response=raw_response,
            events=b"",
            stderr=b"",
            issues=(),
        )


class FailingRunner:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        self.calls += 1
        return ProposalRun(
            provider="fixed",
            model="fixed",
            model_settings={},
            started=True,
            terminal_state="failed",
            raw_response=None,
            events=b"",
            stderr=b"provider failure",
            issues=(ValidationIssue(IssueCode.RUNNER_FAILED, "provider failure", "/runner"),),
        )


def _approved_flow(
    instrumentation: str = "solo_piano", target_duration: int = 180
) -> dict[str, object]:
    result = approve_flow(
        {
            "document_type": "flow",
            "schema_version": "0.1.0",
            "document_id": "flow-001",
            "revision": 1,
            "status": "draft",
            "approval": None,
            "title": "小さな曲",
            "overall_flow": "主題を示して閉じる。",
            "instrumentation": instrumentation,
            "target_duration": target_duration,
            "scenes": [
                {
                    "scene_id": "scene-001",
                    "name": "主題",
                    "length_class": "medium",
                    "heard_as": "穏やかな主題。",
                    "relation_to_previous": "曲の始まり。",
                    "transition_to_next": "静かに閉じる。",
                }
            ],
        }
    )
    assert result.document is not None
    return result.document


def _structure_response() -> dict[str, object]:
    return {
        "materials": [{"description": "短い主題"}],
        "scenes": [
            {
                "sections": [
                    {
                        "depth": 0,
                        "role": "statement",
                        "description": "主題を示す",
                        "relative_length": 1,
                        "placements": [{"material_index": 0, "role": "foreground"}],
                    }
                ]
            }
        ],
    }


def _relations_response() -> dict[str, object]:
    return {
        "outcome": "complete",
        "structure_insufficient_reason": None,
        "variation_relations": [],
        "material_placement_transitions": [],
        "performance_directions": [],
        "performance_comparison_requirements": [],
    }


def test_compilation_calls_two_distinct_operations_and_publishes_only_the_valid_script(
    tmp_path: Path,
) -> None:
    runner = SequencedRunner(
        [_fixture("structure-response.json"), _fixture("relations-response.json")]
    )

    result = compile_script_0_3(
        Script03CompilationRequest(
            approved_flow=_fixture("approved-flow.json"),
            composition_id="composition-001",
        ),
        runner,
        tmp_path / "run",
    )

    assert result.compiled is True
    assert result.outcome == "complete"
    assert result.document is not None
    placements = result.document["script"]["material_placements"]
    first_section = [
        placement for placement in placements.values() if placement["section_id"] == "section-001"
    ]
    assert {placement["role"] for placement in first_section} == {
        "foreground",
        "accompaniment",
    }
    assert sum(placement["material_id"] == "material-001" for placement in placements.values()) == 5
    plan, _ = build_piece_plan(result.document, PlanChoice(tonal_center=0, mode="major"))
    assert plan.plan_id == "composition-001"
    assert len(runner.prompts) == 2
    assert runner.schemas[0] != runner.schemas[1]
    assert '"section-001"' in runner.prompts[1]
    assert (tmp_path / "run/outputs/indexed-structure.json").is_file()
    assert (tmp_path / "run/outputs/projection-ledger.json").is_file()
    assert (tmp_path / "run/outputs/validated-script.json").is_file()
    compilation = json.loads(
        (tmp_path / "run/outputs/compilation.json").read_text(encoding="utf-8")
    )
    assert compilation["call_count"] == 2
    assert compilation["repair_count"] == 0


def test_structure_content_failure_is_repaired_once_before_relations(
    tmp_path: Path,
) -> None:
    invalid_structure = _structure_response()
    invalid_structure["scenes"][0]["sections"][0]["placements"][0]["material_index"] = 99
    runner = SequencedRunner([invalid_structure, _structure_response(), _relations_response()])

    result = compile_script_0_3(
        Script03CompilationRequest(
            approved_flow=_approved_flow(),
            composition_id="composition-001",
        ),
        runner,
        tmp_path / "run",
    )

    assert result.compiled is True
    assert len(runner.prompts) == 3
    assert "unknown material index" in runner.prompts[1]
    assert runner.schemas[0] == runner.schemas[1]
    assert (tmp_path / "run/repairs/script-structure.json").is_file()
    compilation = json.loads(
        (tmp_path / "run/outputs/compilation.json").read_text(encoding="utf-8")
    )
    assert compilation["call_count"] == 3
    assert compilation["repair_count"] == 1


def test_relations_content_failure_is_repaired_once_with_the_structure_frozen(
    tmp_path: Path,
) -> None:
    invalid_relations = _relations_response()
    invalid_relations["structure_insufficient_reason"] = "completeなのに理由がある"
    runner = SequencedRunner([_structure_response(), invalid_relations, _relations_response()])

    result = compile_script_0_3(
        Script03CompilationRequest(
            approved_flow=_approved_flow(),
            composition_id="composition-001",
        ),
        runner,
        tmp_path / "run",
    )

    assert result.compiled is True
    assert len(runner.prompts) == 3
    assert "complete response cannot contain" in runner.prompts[2]
    assert '"document_type": "indexed_structure"' in runner.prompts[2]
    assert runner.schemas[1] == runner.schemas[2]
    assert (tmp_path / "run/repairs/script-relations.json").is_file()
    compilation = json.loads(
        (tmp_path / "run/outputs/compilation.json").read_text(encoding="utf-8")
    )
    assert compilation["call_count"] == 3
    assert compilation["repair_count"] == 1


def test_unreadable_json_is_a_content_failure_and_gets_one_repair(tmp_path: Path) -> None:
    runner = SequencedRunner([b"not-json", _structure_response(), _relations_response()])

    result = compile_script_0_3(
        Script03CompilationRequest(
            approved_flow=_approved_flow(),
            composition_id="composition-001",
        ),
        runner,
        tmp_path / "run",
    )

    assert result.compiled is True
    assert len(runner.prompts) == 3
    assert "not-json" in runner.prompts[1]
    assert (tmp_path / "run/repairs/script-structure.json").is_file()


def test_unreadable_relations_json_gets_one_repair(tmp_path: Path) -> None:
    runner = SequencedRunner([_structure_response(), b"not-json", _relations_response()])

    result = compile_script_0_3(
        Script03CompilationRequest(
            approved_flow=_approved_flow(),
            composition_id="composition-001",
        ),
        runner,
        tmp_path / "run",
    )

    assert result.compiled is True
    assert len(runner.prompts) == 3
    assert "not-json" in runner.prompts[2]
    assert '"document_type": "indexed_structure"' in runner.prompts[2]
    assert (tmp_path / "run/repairs/script-relations.json").is_file()


def test_runner_failure_is_not_repaired_and_does_not_start_relations(
    tmp_path: Path,
) -> None:
    runner = FailingRunner()

    result = compile_script_0_3(
        Script03CompilationRequest(
            approved_flow=_approved_flow(),
            composition_id="composition-001",
        ),
        runner,
        tmp_path / "run",
    )

    assert result.compiled is False
    assert result.outcome == "runner_failed"
    assert runner.calls == 1
    assert not (tmp_path / "run/repairs").exists()
    assert not (tmp_path / "run/outputs/indexed-structure.json").exists()
    assert not (tmp_path / "run/outputs/validated-script.json").exists()


def test_structure_insufficient_stops_without_repair_or_validated_script(
    tmp_path: Path,
) -> None:
    insufficient = _relations_response()
    insufficient.update(
        {
            "outcome": "structure_insufficient",
            "structure_insufficient_reason": "必要な再提示関係を現在の構造では表せない",
        }
    )
    runner = SequencedRunner([_structure_response(), insufficient])

    result = compile_script_0_3(
        Script03CompilationRequest(
            approved_flow=_approved_flow(),
            composition_id="composition-001",
        ),
        runner,
        tmp_path / "run",
    )

    assert result.compiled is False
    assert result.outcome == "structure_insufficient"
    assert len(runner.prompts) == 2
    assert not (tmp_path / "run/repairs").exists()
    assert not (tmp_path / "run/outputs/validated-script.json").exists()
    compilation = json.loads(
        (tmp_path / "run/outputs/compilation.json").read_text(encoding="utf-8")
    )
    assert compilation["outcome"] == "structure_insufficient"
    assert compilation["repair_count"] == 0


def test_each_operation_gets_at_most_one_content_repair(tmp_path: Path) -> None:
    first = _structure_response()
    first["scenes"][0]["sections"][0]["placements"][0]["material_index"] = 98
    second = _structure_response()
    second["scenes"][0]["sections"][0]["placements"][0]["material_index"] = 99
    runner = SequencedRunner([first, second])

    result = compile_script_0_3(
        Script03CompilationRequest(
            approved_flow=_approved_flow(),
            composition_id="composition-001",
        ),
        runner,
        tmp_path / "run",
    )

    assert result.compiled is False
    assert result.outcome == "content_invalid"
    assert len(runner.prompts) == 2
    assert not (tmp_path / "run/outputs/indexed-structure.json").exists()
    assert not (tmp_path / "run/outputs/validated-script.json").exists()
    compilation = json.loads(
        (tmp_path / "run/outputs/compilation.json").read_text(encoding="utf-8")
    )
    assert compilation["repair_count"] == 1


def test_profile_incompatible_flow_is_rejected_before_any_model_call(
    tmp_path: Path,
) -> None:
    runner = SequencedRunner([])

    result = compile_script_0_3(
        Script03CompilationRequest(
            approved_flow=_approved_flow(instrumentation="string_quartet"),
            composition_id="composition-001",
        ),
        runner,
        tmp_path / "run",
    )

    assert result.compiled is False
    assert result.outcome == "request_invalid"
    assert runner.prompts == []
    assert any(issue.code == IssueCode.UNREPRESENTABLE for issue in result.issues)
    assert not (tmp_path / "run").exists()


def test_parse_and_semantic_failures_share_one_repair_budget(tmp_path: Path) -> None:
    still_invalid = _structure_response()
    still_invalid["scenes"][0]["sections"][0]["placements"][0]["material_index"] = 99
    runner = SequencedRunner([b"not-json", still_invalid])

    result = compile_script_0_3(
        Script03CompilationRequest(
            approved_flow=_approved_flow(),
            composition_id="composition-001",
        ),
        runner,
        tmp_path / "run",
    )

    assert result.compiled is False
    assert result.outcome == "content_invalid"
    assert len(runner.prompts) == 2
    compilation = json.loads(
        (tmp_path / "run/outputs/compilation.json").read_text(encoding="utf-8")
    )
    assert compilation["repair_count"] == 1
