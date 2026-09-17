import copy
import json
from pathlib import Path

import pytest
from test_scoim_proposal import _profile_response

from llm_musical_composer.run_state import StateConflictError
from scoim.operations import approve
from scoim.proposal import ProposalRequest, ProposalRun, propose_script
from scoim.realization_run import approve_pending_review
from scoim.staged_realization import (
    advance_staged_realization,
    initialize_staged_realization,
    rebuild_staged_realization,
)
from scoim.validation import content_sha256

_FIXTURE = Path(__file__).parent / "fixtures" / "scoim" / "nested-aba" / "approved-script.json"


def _approved_script() -> dict[str, object]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


class QueueRunner:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = responses
        self.calls = 0

    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        response = self.responses[self.calls]
        self.calls += 1
        return ProposalRun(
            provider="fake",
            model="test-model",
            model_settings={},
            started=True,
            terminal_state="completed",
            raw_response=json.dumps(response).encode("utf-8"),
            events=b'{"type":"thread.started"}\n{"type":"turn.completed"}\n',
            stderr=b"",
            issues=(),
        )


class NeverRunner:
    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        raise AssertionError("A completed staged run must not call the model again")


def _complete_responses() -> list[dict[str, object]]:
    melody = {
        "foreground_voice": "upper",
        "notes": [{"at_units": 0, "duration_units": 4, "pitch": 67}],
    }
    accompaniment = {
        "events": [
            {
                "at_units": 0,
                "preferred_duration_units": 4,
                "degree": "root",
                "preferred_register_zone": "bass",
                "articulations": [],
            }
        ]
    }
    transition_melody = {
        "foreground_voice": "upper",
        "notes": [{"at_units": 0, "duration_units": 2, "pitch": 69}],
    }
    transition_accompaniment = {
        "events": [
            {
                "at_units": 0,
                "preferred_duration_units": 2,
                "degree": "root",
                "preferred_register_zone": "bass",
                "articulations": [],
            }
        ]
    }

    def performance(count: int) -> dict[str, object]:
        return {
            "performances": [
                {
                    "timing_profile": "flow",
                    "timing_amount": "subtle",
                    "dynamics_profile": "shape",
                    "articulation_profile": "legato",
                    "coordination_profile": "aligned",
                    "pedal_profile": "phrase_legato",
                }
                for _ in range(count)
            ]
        }

    return [
        {
            "tonal_center": 0,
            "mode": "major",
            "contrast_descriptions": ["Open the register", "Add motion"],
        },
        {
            "harmonies": [
                [{"duration_units": 2, "root_pitch_class": 7, "quality": "major"}],
                [{"duration_units": 4, "root_pitch_class": 5, "quality": "major"}],
                [{"duration_units": 4, "root_pitch_class": 0, "quality": "major"}],
            ]
        },
        {"melodies": [melody, melody]},
        {"transition_melodies": [transition_melody, transition_melody]},
        {"accompaniments": [accompaniment, accompaniment]},
        {
            "transition_accompaniments": [
                transition_accompaniment,
                transition_accompaniment,
            ]
        },
        performance(1),
        performance(2),
        performance(1),
    ]


def _non_return_responses() -> list[dict[str, object]]:
    responses = _complete_responses()
    responses[0]["contrast_descriptions"] = ["Open the register"]
    responses[1]["harmonies"] = [
        [{"duration_units": 4, "root_pitch_class": root, "quality": "major"}] for root in (5, 7, 0)
    ]
    responses[3]["transition_melodies"] = responses[3]["transition_melodies"][:1]
    responses[5]["transition_accompaniments"] = responses[5]["transition_accompaniments"][:1]
    return responses[:7]


def test_staged_realization_initializes_an_immutable_parent_run(tmp_path: Path) -> None:
    document = _approved_script()
    run_dir = tmp_path / "staged"

    status = initialize_staged_realization(
        run_dir,
        document,
        profile="solo_piano_3m_v1",
        model="test-model",
        review_after={},
    )

    assert status.status == "initialized"
    assert status.current_stage == "plan-harmony"
    assert status.completed_stages == ()
    spec = json.loads((run_dir / "run-spec.json").read_text(encoding="utf-8"))
    assert spec == {
        "schema_version": 1,
        "approved_script_sha256": content_sha256(document),
        "profile": "solo_piano_3m_v1",
        "model": "test-model",
        "stages": [
            {"stage_id": "plan-harmony", "run_path": "stages/01-plan-harmony"},
            {"stage_id": "melody", "run_path": "stages/02-melody"},
            {"stage_id": "accompaniment", "run_path": "stages/03-accompaniment"},
            {"stage_id": "ending", "run_path": "stages/04-ending"},
            {"stage_id": "performance", "run_path": "stages/05-performance"},
        ],
        "review_after": {
            "accompaniment": [],
            "ending": [],
            "melody": [],
            "performance": [],
            "plan-harmony": [],
        },
        "call_policy": "bounded_content_repair_v1",
    }
    assert (
        json.loads((run_dir / "inputs" / "approved-script.json").read_text(encoding="utf-8"))
        == document
    )


def test_staged_realization_completes_all_five_stages(tmp_path: Path) -> None:
    run_dir = tmp_path / "staged"
    initialize_staged_realization(
        run_dir,
        _approved_script(),
        profile="solo_piano_3m_v1",
        model="test-model",
        review_after={},
    )
    runner = QueueRunner(_complete_responses())

    status = advance_staged_realization(run_dir, runner)

    assert status.status == "completed", status
    assert status.current_stage is None
    assert status.completed_stages == (
        "plan-harmony",
        "melody",
        "accompaniment",
        "ending",
        "performance",
    )
    assert status.workspace is not None
    assert status.workspace.schema_version == 4
    assert runner.calls == 9


def test_staged_realization_resumes_a_completed_run_without_model_calls(tmp_path: Path) -> None:
    run_dir = tmp_path / "staged"
    document = _approved_script()
    initialize_staged_realization(
        run_dir,
        document,
        profile="solo_piano_3m_v1",
        model="test-model",
        review_after={},
    )
    completed = advance_staged_realization(run_dir, QueueRunner(_complete_responses()))
    assert completed.status == "completed"

    resumed = initialize_staged_realization(
        run_dir,
        document,
        profile="solo_piano_3m_v1",
        model="test-model",
        review_after={},
    )
    status = advance_staged_realization(run_dir, NeverRunner())

    assert resumed.status == "completed"
    assert status.status == "completed"
    assert status.workspace == completed.workspace


def test_staged_realization_rejects_changed_resume_settings(tmp_path: Path) -> None:
    run_dir = tmp_path / "staged"
    document = _approved_script()
    initialize_staged_realization(
        run_dir,
        document,
        profile="solo_piano_3m_v1",
        model="test-model",
        review_after={},
    )

    with pytest.raises(StateConflictError, match="saved run-spec conflicts with current input"):
        initialize_staged_realization(
            run_dir,
            document,
            profile="solo_piano_3m_v1",
            model="changed-model",
            review_after={},
        )


def test_staged_realization_rejects_an_unknown_review_stage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown staged realization review stage"):
        initialize_staged_realization(
            tmp_path / "staged",
            _approved_script(),
            profile="solo_piano_3m_v1",
            model="test-model",
            review_after={"unknown": frozenset()},
        )


def test_staged_realization_rejects_an_invalid_script(tmp_path: Path) -> None:
    document = _approved_script()
    del document["document_id"]

    with pytest.raises(ValueError, match="document_id"):
        initialize_staged_realization(
            tmp_path / "staged",
            document,
            profile="solo_piano_3m_v1",
            model="test-model",
            review_after={},
        )


def test_staged_realization_requires_an_approved_script(tmp_path: Path) -> None:
    document = _approved_script()
    document["status"] = "draft"
    document["approval"] = None
    document["script"]["requirements"] = {}

    with pytest.raises(ValueError, match="requires an approved script"):
        initialize_staged_realization(
            tmp_path / "staged",
            document,
            profile="solo_piano_3m_v1",
            model="test-model",
            review_after={},
        )


def test_staged_realization_rejects_an_unsupported_profile(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported staged realization profile"):
        initialize_staged_realization(
            tmp_path / "staged",
            _approved_script(),
            profile="string_quartet_3m_v1",
            model="test-model",
            review_after={},
        )


def test_staged_realization_rejects_a_script_outside_the_profile(tmp_path: Path) -> None:
    document = _approved_script()
    document["script"]["performance_setup"]["target_duration_seconds"] = 120
    document["approval"] = {"content_sha256": content_sha256(document)}

    with pytest.raises(ValueError, match="180-second target"):
        initialize_staged_realization(
            tmp_path / "staged",
            document,
            profile="solo_piano_3m_v1",
            model="test-model",
            review_after={},
        )


def test_staged_realization_rebuild_rejects_a_changed_script_snapshot(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "staged"
    initialize_staged_realization(
        run_dir,
        _approved_script(),
        profile="solo_piano_3m_v1",
        model="test-model",
        review_after={},
    )
    snapshot_path = run_dir / "inputs" / "approved-script.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["script"]["title"] = "変更"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match its immutable specification"):
        rebuild_staged_realization(run_dir)


def test_staged_realization_rebuild_rejects_a_stage_for_another_script(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "staged"
    initialize_staged_realization(
        run_dir,
        _approved_script(),
        profile="solo_piano_3m_v1",
        model="test-model",
        review_after={"plan-harmony": frozenset({"overall-plan"})},
    )
    advance_staged_realization(run_dir, QueueRunner(_complete_responses()))
    spec_path = run_dir / "stages" / "01-plan-harmony" / "run-spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["approved_script_sha256"] = "0" * 64
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(ValueError, match="Stage script hash does not match"):
        rebuild_staged_realization(run_dir)


def test_staged_realization_rebuild_rejects_a_broken_workspace_chain(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "staged"
    initialize_staged_realization(
        run_dir,
        _approved_script(),
        profile="solo_piano_3m_v1",
        model="test-model",
        review_after={"plan-harmony": frozenset({"overall-plan"})},
    )
    advance_staged_realization(run_dir, QueueRunner(_complete_responses()))
    spec_path = run_dir / "stages" / "01-plan-harmony" / "run-spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["source_workspace_record_sha256"] = "0" * 64
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(ValueError, match="does not continue the parent chain"):
        rebuild_staged_realization(run_dir)


def test_staged_realization_rejects_a_changed_script_on_resume(tmp_path: Path) -> None:
    run_dir = tmp_path / "staged"
    document = _approved_script()
    initialize_staged_realization(
        run_dir,
        document,
        profile="solo_piano_3m_v1",
        model="test-model",
        review_after={},
    )
    changed = copy.deepcopy(document)
    changed["script"]["title"] = "変更された曲名"
    changed["approval"] = {"content_sha256": content_sha256(changed)}

    with pytest.raises(StateConflictError, match="saved run-spec conflicts with current input"):
        initialize_staged_realization(
            run_dir,
            changed,
            profile="solo_piano_3m_v1",
            model="test-model",
            review_after={},
        )


def test_staged_realization_stops_at_and_resumes_from_a_review(tmp_path: Path) -> None:
    run_dir = tmp_path / "staged"
    initialize_staged_realization(
        run_dir,
        _approved_script(),
        profile="solo_piano_3m_v1",
        model="test-model",
        review_after={"plan-harmony": frozenset({"overall-plan"})},
    )
    runner = QueueRunner(_complete_responses())

    waiting = advance_staged_realization(run_dir, runner)

    assert waiting.status == "awaiting_review"
    assert waiting.current_stage == "plan-harmony"
    assert waiting.awaiting_review == "plan-harmony:overall-plan"
    assert runner.calls == 1

    approve_pending_review(run_dir / "stages" / "01-plan-harmony", actor="human")
    completed = advance_staged_realization(run_dir, runner)

    assert completed.status == "completed"
    assert runner.calls == 9


def test_staged_realization_does_not_start_later_stages_after_a_failure(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "staged"
    initialize_staged_realization(
        run_dir,
        _approved_script(),
        profile="solo_piano_3m_v1",
        model="test-model",
        review_after={},
    )
    responses = _complete_responses()
    responses[1] = {"harmonies": []}
    runner = QueueRunner(responses)

    status = advance_staged_realization(run_dir, runner)

    assert status.status == "failed"
    assert status.current_stage == "plan-harmony"
    assert status.completed_stages == ()
    assert status.issues
    assert runner.calls == 2
    assert not (run_dir / "stages" / "02-melody").exists()


def test_profile_proposal_without_a_return_completes_the_fixed_stages(tmp_path: Path) -> None:
    proposal = propose_script(
        ProposalRequest(
            instruction="静かな夜から一度だけ頂点を作り、冒頭へ戻らず余韻へほどける曲",
            document_id="night_arc",
            instrumentation="solo_piano",
            target_duration_seconds=180,
            target_profile="solo_piano_3m_v1",
        ),
        QueueRunner([_profile_response()]),
        tmp_path / "proposal",
    )
    assert proposal.document is not None
    approval = approve(proposal.document)
    assert approval.document is not None
    roles = {section["role"] for section in approval.document["script"]["sections"].values()}
    assert "return" not in roles
    run_dir = tmp_path / "staged"
    initialize_staged_realization(
        run_dir,
        approval.document,
        profile="solo_piano_3m_v1",
        model="test-model",
        review_after={},
    )

    status = advance_staged_realization(run_dir, QueueRunner(_non_return_responses()))

    assert status.status == "completed", status
    assert status.workspace is not None
