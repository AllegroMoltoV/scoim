import json
from pathlib import Path

from llm_musical_composer.pipeline_dsl import (
    parse_performance_spec,
    parse_score_spec,
)
from scoim.operations import apply_patch, approve
from scoim.proposal import ProposalRequest, ProposalRun, propose_script
from scoim.realization_generation import _create_single_response_model_trial as create_model_trial
from scoim.trial_bundle import replay_trial_bundle
from scoim.validation import IssueCode, ValidationIssue

_FIXED_ABA = Path(__file__).parent / "fixtures" / "scoim" / "fixed-aba"


def _fixture(name: str) -> dict[str, object]:
    return json.loads((_FIXED_ABA / name).read_text(encoding="utf-8"))


def _model_response() -> dict[str, object]:
    legacy = _fixture("frozen-response.json")
    legacy_score = parse_score_spec(legacy["score_spec"]["source"])
    legacy_performance = parse_performance_spec(legacy["performance_spec"]["source"])
    section_ids = ("whole", "statement", "bridge", "contrast", "return", "release")
    performances = {
        section_id: {
            "timing_profile": None,
            "timing_amount": None,
            "dynamics_profile": None,
            "articulation_profile": None,
            "coordination_profile": None,
            "pedal_profile": None,
        }
        for section_id in section_ids
    }
    for item in legacy_performance.node_performances:
        performances[item.node_id] = {
            "timing_profile": item.timing_profile,
            "timing_amount": item.timing_amount,
            "dynamics_profile": item.dynamics_profile,
            "articulation_profile": item.articulation_profile,
            "coordination_profile": item.coordination_profile,
            "pedal_profile": item.pedal_profile,
        }
    plan_choice = legacy["plan_choice"]
    assert isinstance(plan_choice, dict)
    harmonic_focus = plan_choice["harmonic_focus_by_section"]
    assert isinstance(harmonic_focus, dict)
    return {
        "plan_choice": {
            "tonal_center": plan_choice["tonal_center"],
            "mode": plan_choice["mode"],
            "harmonic_focus_by_section": {
                section_id: harmonic_focus.get(section_id) for section_id in section_ids
            },
            "contrasts_with_by_section": {"contrast": 0},
        },
        "score_materials": {
            material.material_id: {
                "foreground_voice": material.foreground_voice,
                "notes": [
                    {
                        "at_units": note.at_units * 3,
                        "duration_units": note.duration_units * 3,
                        "pitch": note.pitch,
                        "voice": note.voice,
                        "tie": note.tie,
                        "articulations": list(note.articulations),
                    }
                    for note in material.notes
                ],
                "harmonies": [
                    {
                        "at_units": harmony.at_units * 3,
                        "duration_units": harmony.duration_units * 3,
                        "root_pitch_class": harmony.root_pitch_class,
                        "quality": harmony.quality,
                    }
                    for harmony in material.harmonies
                ],
                "directions": [
                    {
                        "at_units": direction.at_units * 3,
                        "kind": direction.kind,
                        "value": direction.value,
                    }
                    for direction in material.directions
                ],
            }
            for material in legacy_score.materials
        },
        "node_performances": performances,
    }


def _approved_with_requirement() -> dict[str, object]:
    result = apply_patch(
        _fixture("approved-script.json"),
        [
            {"op": "test", "path": "/revision", "value": 1},
            {"op": "replace", "path": "/revision", "value": 2},
            {
                "op": "add",
                "path": "/script/requirements",
                "value": {
                    "return_more_aligned": {
                        "performance_direction_id": "clear_return",
                        "feature": "onset_alignment",
                        "relation": "more",
                    }
                },
            },
        ],
        new_draft=True,
    )
    assert result.document is not None
    approved = approve(result.document)
    assert approved.document is not None
    return approved.document


def _proposal_response() -> dict[str, object]:
    script = _fixture("approved-script.json")["script"]
    assert isinstance(script, dict)
    sections = script["sections"]
    materials = script["materials"]
    placements = script["placements"]
    variations = script["variations"]
    transitions = script["transitions"]
    setup = script["performance_setup"]
    assert all(
        isinstance(value, dict)
        for value in (sections, materials, placements, variations, transitions, setup)
    )
    section_order = ["whole", "statement", "bridge", "contrast", "return", "release"]
    return {
        "title": script["title"],
        "brief": script["brief"],
        "root_section_id": script["root_section_id"],
        "sections": [
            {
                "id": section_id,
                "parent_section_id": sections[section_id]["parent_section_id"],
                "role": sections[section_id]["role"],
                "relative_length": sections[section_id].get("relative_length"),
                "description": sections[section_id]["description"],
            }
            for section_id in section_order
        ],
        "materials": [{"id": item_id, **item} for item_id, item in materials.items()],
        "placements": [{"id": item_id, **item} for item_id, item in placements.items()],
        "variations": [{"id": item_id, **item} for item_id, item in variations.items()],
        "transitions": [{"id": item_id, **item} for item_id, item in transitions.items()],
        "performance_directions": [
            {
                "id": item_id,
                "target_type": item["target"]["type"],
                "target_id": item["target"]["id"],
                "relative_to_type": (
                    item.get("relative_to", {}).get("type")
                    if isinstance(item.get("relative_to"), dict)
                    else None
                ),
                "relative_to_id": (
                    item.get("relative_to", {}).get("id")
                    if isinstance(item.get("relative_to"), dict)
                    else None
                ),
                "description": item["description"],
            }
            for item_id, item in setup["performance_directions"].items()
        ],
        "requirements": [],
    }


class _FakeRunner:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.prompts: list[str] = []
        self.schema_paths: list[Path] = []
        self.schemas: list[dict[str, object]] = []

    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        self.prompts.append(prompt)
        self.schema_paths.append(response_schema_path)
        self.schemas.append(json.loads(response_schema_path.read_text(encoding="utf-8")))
        return ProposalRun(
            provider="fake",
            model="test-model",
            model_settings={"reasoning_effort": "medium"},
            started=True,
            terminal_state="completed",
            raw_response=json.dumps(self.response, ensure_ascii=False).encode("utf-8"),
            events=b'{"type":"turn.completed"}\n',
            stderr=b"",
            issues=(),
        )


class _RunRunner:
    def __init__(self, run: ProposalRun) -> None:
        self.run_result = run
        self.prompts: list[str] = []

    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        self.prompts.append(prompt)
        return self.run_result


def test_create_model_trial_uses_one_structured_call_and_creates_a_bundle(
    tmp_path: Path,
) -> None:
    runner = _FakeRunner(_model_response())
    output = tmp_path / "model-trial"

    result = create_model_trial(
        _fixture("approved-script.json"),
        runner,
        output,
        trial_id="model-trial",
    )

    assert result.created is True
    assert result.persisted is True
    assert result.bundle_path == output
    assert result.outcome.promotion_eligible is True
    assert len(runner.prompts) == 1
    assert "statement" in runner.prompts[0]
    assert "pipeline-dsl-v1" not in runner.prompts[0]
    assert "target_assertions" not in runner.prompts[0]
    assert "音楽上の値" in runner.prompts[0]
    assert runner.schema_paths[0].name == "typed-realization-response-2.schema.json"
    assert set(runner.schemas[0]["properties"]) == {
        "plan_choice",
        "score_materials",
        "node_performances",
    }
    assert (output / "artifacts" / "final.mid").is_file()
    assert (output / "model-runs" / "realization.json").is_file()
    frozen_response = json.loads((output / "frozen-response.json").read_text(encoding="utf-8"))
    assert frozen_response["schema_version"] == 2
    assert set(frozen_response) == {
        "schema_version",
        "profile",
        "plan_choice",
        "score_materials",
        "node_performances",
    }
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 3
    assert manifest["files"]["realization_model_run"]["path"] == ("model-runs/realization.json")


def test_create_model_trial_prompts_for_typed_requirements_without_assertions(
    tmp_path: Path,
) -> None:
    runner = _FakeRunner(_model_response())

    result = create_model_trial(
        _approved_with_requirement(),
        runner,
        tmp_path / "typed-trial",
        trial_id="typed-trial",
    )

    assert result.created is True
    prompt = runner.prompts[0]
    assert '"target_section_id": "return"' in prompt
    assert '"reference_section_id": "statement"' in prompt
    assert '"feature": "onset_alignment"' in prompt
    assert '"relation": "more"' in prompt
    assert "型付き検査要件について" in prompt
    assert "target_assertions" not in prompt


def test_create_model_trial_includes_a_validated_proposal_record(tmp_path: Path) -> None:
    proposal_dir = tmp_path / "proposal"
    proposed = propose_script(
        ProposalRequest(
            instruction="静かに始まり、主題へ戻って閉じる曲",
            document_id="evening_return",
            instrumentation="solo_piano",
            target_duration_seconds=180,
        ),
        _FakeRunner(_proposal_response()),
        proposal_dir,
    )
    assert proposed.proposed is True
    assert proposed.document is not None
    approved = approve(proposed.document)
    assert approved.document is not None

    output = tmp_path / "model-trial"
    result = create_model_trial(
        approved.document,
        _FakeRunner(_model_response()),
        output,
        trial_id="model-trial",
        proposal_record_dir=proposal_dir,
    )

    assert result.created is True
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"]["proposal_model_run"]["path"] == "model-runs/proposal.json"
    proposal_record = json.loads(
        (output / "model-runs" / "proposal.json").read_text(encoding="utf-8")
    )
    assert proposal_record["schema_version"] == 1
    assert set(proposal_record["files"]) == set(
        json.loads((proposal_dir / "manifest.json").read_text(encoding="utf-8"))["files"]
    ) | {"manifest.json"}


def test_create_model_trial_rejects_a_modified_proposal_record_before_model_call(
    tmp_path: Path,
) -> None:
    proposal_dir = tmp_path / "proposal"
    proposed = propose_script(
        ProposalRequest(
            instruction="静かに始まり、主題へ戻って閉じる曲",
            document_id="evening_return",
            instrumentation="solo_piano",
            target_duration_seconds=180,
        ),
        _FakeRunner(_proposal_response()),
        proposal_dir,
    )
    assert proposed.document is not None
    approved = approve(proposed.document)
    assert approved.document is not None
    (proposal_dir / "prompt.md").write_text("modified", encoding="utf-8")
    runner = _FakeRunner(_model_response())

    result = create_model_trial(
        approved.document,
        runner,
        tmp_path / "trial",
        trial_id="trial",
        proposal_record_dir=proposal_dir,
    )

    assert result.persisted is False
    assert result.issues[0].code is IssueCode.LINEAGE_MISMATCH
    assert result.issues[0].path == "/proposal_record/prompt.md"
    assert runner.prompts == []


def test_create_model_trial_rejects_an_unplaced_material_before_model_call(
    tmp_path: Path,
) -> None:
    changed = apply_patch(
        _fixture("approved-script.json"),
        [
            {"op": "test", "path": "/revision", "value": 1},
            {"op": "replace", "path": "/revision", "value": 2},
            {"op": "add", "path": "/script/requirements", "value": {}},
            {
                "op": "add",
                "path": "/script/materials/unused",
                "value": {"kind": "theme", "description": "配置されない素材。"},
            },
        ],
        new_draft=True,
    )
    assert changed.document is not None
    approved = approve(changed.document)
    assert approved.document is not None
    runner = _FakeRunner(_model_response())

    result = create_model_trial(
        approved.document,
        runner,
        tmp_path / "unplaced-material",
        trial_id="unplaced-material",
    )

    assert result.persisted is False
    assert result.issues[0].code is IssueCode.UNREPRESENTABLE
    assert result.issues[0].path == "/script/materials/unused"
    assert runner.prompts == []


def test_create_model_trial_persists_an_unknown_fixed_key(tmp_path: Path) -> None:
    response = _model_response()
    score_materials = response["score_materials"]
    assert isinstance(score_materials, dict)
    score_materials["unknown"] = score_materials["theme"]
    output = tmp_path / "unknown-fixed-key"

    result = create_model_trial(
        _fixture("approved-script.json"),
        _FakeRunner(response),
        output,
        trial_id="unknown-fixed-key",
    )

    assert result.created is False
    assert result.persisted is True
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert result.issues[0].path == "/runner/response/score_materials/unknown"
    assert not (output / "frozen-response.json").exists()


def test_create_model_trial_rejects_a_draft_before_calling_the_runner(
    tmp_path: Path,
) -> None:
    document = _fixture("approved-script.json")
    document["status"] = "draft"
    document["approval"] = None
    document["script"]["requirements"] = {}
    runner = _FakeRunner(_fixture("frozen-response.json"))

    result = create_model_trial(
        document,
        runner,
        tmp_path / "draft-trial",
        trial_id="draft-trial",
    )

    assert result.created is False
    assert result.persisted is False
    assert result.issues[0].code.value == "semantic_invalid"
    assert result.issues[0].path == "/status"
    assert runner.prompts == []


def test_create_model_trial_persists_a_started_timeout_without_a_fake_response(
    tmp_path: Path,
) -> None:
    class TimeoutRunner:
        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            return ProposalRun(
                provider="fake",
                model="test-model",
                model_settings={"timeout_seconds": 1},
                started=True,
                terminal_state="timeout",
                raw_response=None,
                events=b'{"type":"turn.started"}\n',
                stderr=b"timed out",
                issues=(
                    ValidationIssue(
                        IssueCode.RUNNER_TIMEOUT,
                        "The realization runner timed out",
                        "/runner",
                    ),
                ),
            )

    output = tmp_path / "timeout-trial"

    result = create_model_trial(
        _fixture("approved-script.json"),
        TimeoutRunner(),
        output,
        trial_id="timeout-trial",
    )

    assert result.created is False
    assert result.persisted is True
    assert result.trial_path == output
    assert result.bundle_path is None
    assert result.issues[0].code is IssueCode.RUNNER_TIMEOUT
    assert (output / "terminal.json").is_file()
    assert (output / "model-runs" / "realization.json").is_file()
    assert not (output / "frozen-response.json").exists()


def test_replay_rejects_a_modified_realization_model_record(tmp_path: Path) -> None:
    bundle = tmp_path / "model-trial"
    created = create_model_trial(
        _fixture("approved-script.json"),
        _FakeRunner(_model_response()),
        bundle,
        trial_id="model-trial",
    )
    assert created.created is True
    (bundle / "model-runs" / "realization.json").write_text("{}\n", encoding="utf-8")

    replayed = replay_trial_bundle(bundle, tmp_path / "replay")

    assert replayed.replayed is False
    assert replayed.issues[0].code is IssueCode.LINEAGE_MISMATCH
    assert replayed.issues[0].path == "/files/realization_model_run/sha256"


def test_create_model_trial_persists_an_invalid_started_response(tmp_path: Path) -> None:
    class InvalidRunner:
        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            return ProposalRun(
                provider="fake",
                model="test-model",
                model_settings={},
                started=True,
                terminal_state="completed",
                raw_response=b"{",
                events=None,
                stderr=None,
                issues=(),
            )

    output = tmp_path / "invalid-trial"

    result = create_model_trial(
        _fixture("approved-script.json"),
        InvalidRunner(),
        output,
        trial_id="invalid-trial",
    )

    assert result.created is False
    assert result.persisted is True
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert (output / "terminal.json").is_file()
    assert not (output / "frozen-response.json").exists()


def test_create_model_trial_rejects_an_existing_output_before_calling_the_runner(
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing-trial"
    output.mkdir()
    runner = _FakeRunner(_fixture("frozen-response.json"))

    result = create_model_trial(
        _fixture("approved-script.json"),
        runner,
        output,
        trial_id="existing-trial",
    )

    assert result.created is False
    assert result.persisted is False
    assert result.issues[0].code is IssueCode.STORAGE_CONFLICT
    assert runner.prompts == []


def test_create_model_trial_rejects_an_empty_trial_id_before_calling_the_runner(
    tmp_path: Path,
) -> None:
    runner = _FakeRunner(_fixture("frozen-response.json"))

    result = create_model_trial(
        _fixture("approved-script.json"),
        runner,
        tmp_path / "trial",
        trial_id="",
    )

    assert result.created is False
    assert result.persisted is False
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert result.issues[0].path == "/trial_id"
    assert runner.prompts == []


def test_create_model_trial_rejects_an_invalid_script_before_calling_the_runner(
    tmp_path: Path,
) -> None:
    document = _fixture("approved-script.json")
    document["unknown"] = True
    runner = _FakeRunner(_fixture("frozen-response.json"))

    result = create_model_trial(
        document,
        runner,
        tmp_path / "invalid-script-trial",
        trial_id="invalid-script-trial",
    )

    assert result.persisted is False
    assert result.issues[0].code is IssueCode.SCHEMA_INVALID
    assert runner.prompts == []


def test_create_model_trial_returns_an_unstarted_runner_failure_without_a_trial(
    tmp_path: Path,
) -> None:
    runner = _RunRunner(
        ProposalRun(
            provider="fake",
            model="test-model",
            model_settings={},
            started=False,
            terminal_state="failed",
            raw_response=None,
            events=None,
            stderr=None,
            issues=(
                ValidationIssue(
                    IssueCode.RUNNER_UNAUTHENTICATED,
                    "The runner is not authenticated",
                    "/runner",
                ),
            ),
        )
    )

    result = create_model_trial(
        _fixture("approved-script.json"),
        runner,
        tmp_path / "unstarted-trial",
        trial_id="unstarted-trial",
    )

    assert result.persisted is False
    assert result.issues[0].code is IssueCode.RUNNER_UNAUTHENTICATED
    assert not (tmp_path / "unstarted-trial").exists()


def test_create_model_trial_persists_a_missing_completed_response(tmp_path: Path) -> None:
    runner = _RunRunner(
        ProposalRun(
            provider="fake",
            model="test-model",
            model_settings={},
            started=True,
            terminal_state="completed",
            raw_response=None,
            events=None,
            stderr=None,
            issues=(),
        )
    )

    result = create_model_trial(
        _fixture("approved-script.json"),
        runner,
        tmp_path / "missing-response-trial",
        trial_id="missing-response-trial",
    )

    assert result.persisted is True
    assert result.issues[0].code is IssueCode.RUNNER_FAILED
    assert not (tmp_path / "missing-response-trial" / "frozen-response.json").exists()


def test_create_model_trial_persists_a_schema_invalid_response(tmp_path: Path) -> None:
    runner = _RunRunner(
        ProposalRun(
            provider="fake",
            model="test-model",
            model_settings={},
            started=True,
            terminal_state="completed",
            raw_response=b'{"schema_version":1}',
            events=None,
            stderr=None,
            issues=(),
        )
    )

    result = create_model_trial(
        _fixture("approved-script.json"),
        runner,
        tmp_path / "schema-invalid-trial",
        trial_id="schema-invalid-trial",
    )

    assert result.persisted is True
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert result.issues[0].path.startswith("/runner/response")
