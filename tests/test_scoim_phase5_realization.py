import json
from pathlib import Path

from scoim.phase3_realization import Phase3Request, realize_phase3
from scoim.phase4_realization import Phase4Request, realize_phase4
from scoim.phase5_realization import Phase5Request, realize_phase5
from scoim.projection_ledger import ProjectionLedgerEntry
from scoim.proposal import ProposalRun
from scoim.validation import IssueCode, ValidationIssue

_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "scoim"
    / "score-unit-layer-vertical"
    / "basic-validated-script.json"
)


class SequencedRunner:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        self.prompts.append(prompt)
        return ProposalRun(
            provider="fixed",
            model="fixed",
            model_settings={},
            started=True,
            terminal_state="completed",
            raw_response=json.dumps(
                self.responses[len(self.prompts) - 1], ensure_ascii=False
            ).encode(),
            events=b"",
            stderr=b"",
            issues=(),
        )


class PreflightFailingRunner:
    def preflight(self) -> tuple[ValidationIssue, ...]:
        return (ValidationIssue(IssueCode.RUNNER_FAILED, "runner unavailable", "/runner"),)

    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        raise AssertionError("preflight failure must prevent a model call")


def _document() -> dict[str, object]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _phase2_ledger(document: dict[str, object]) -> tuple[ProjectionLedgerEntry, ...]:
    script = document["script"]
    assert isinstance(script, dict)
    setup = script["performance_setup"]
    assert isinstance(setup, dict)
    collections = {
        "sections": script["sections"],
        "materials": script["materials"],
        "material_placements": script["material_placements"],
        "script_element_variation_relations": script["script_element_variation_relations"],
        "material_placement_transitions": script["material_placement_transitions"],
        "performance_directions": setup["performance_directions"],
    }
    if "performance_direction_comparison_requirements" in script:
        collections["performance_direction_comparison_requirements"] = script[
            "performance_direction_comparison_requirements"
        ]
    return tuple(
        ProjectionLedgerEntry(
            "phase2_source",
            target_id,
            target_kind,
            target_id,
            "script",
            "direct_id_equality",
            "passed",
            f"target_id={target_id}",
        )
        for target_kind, values in collections.items()
        for target_id in values
    )


def _upstream(tmp_path: Path) -> tuple[dict[str, object], dict[str, object], list[object]]:
    document = _document()
    phase3_dir = tmp_path / "phase3"
    phase3_responses = [
        {
            "tonal_center": 0,
            "mode": "major",
            "overall_harmonic_story": "主調から少し離れて戻る。",
            "section_harmonic_intents": [
                {"harmonic_intent": "主調を示す。", "connection_from_previous": "始める。"},
                {"harmonic_intent": "属へ動く。", "connection_from_previous": "つなぐ。"},
                {"harmonic_intent": "主題へ戻る。", "connection_from_previous": "戻す。"},
                {"harmonic_intent": "着地する。", "connection_from_previous": "閉じる。"},
            ],
        },
        {"harmonies": [{"duration_units": 48, "root_pitch_class": 0, "quality": "major"}]},
        {"harmonies": [{"duration_units": 12, "root_pitch_class": 7, "quality": "major"}]},
        {"harmonies": [{"duration_units": 48, "root_pitch_class": 5, "quality": "major"}]},
        {"harmonies": [{"duration_units": 12, "root_pitch_class": 0, "quality": "major"}]},
    ]
    phase3 = realize_phase3(
        Phase3Request(document, _phase2_ledger(document)),
        SequencedRunner(phase3_responses),
        phase3_dir,
    )
    assert phase3.realized
    phase3_state = json.loads(
        (phase3_dir / "outputs" / "phase3-state.json").read_text(encoding="utf-8")
    )
    phase3_ledger = json.loads(
        (phase3_dir / "outputs" / "projection-ledger.json").read_text(encoding="utf-8")
    )
    phase4_dir = tmp_path / "phase4"
    phase4_responses = [
        {"notes": [{"at_units": 0, "duration_units": 12, "pitch": 72, "voice": "upper"}]},
        {"notes": [{"at_units": 0, "duration_units": 8, "pitch": 74, "voice": "upper"}]},
        {"notes": [{"at_units": 0, "duration_units": 12, "pitch": 60, "voice": "lower"}]},
        {"notes": [{"at_units": 0, "duration_units": 12, "pitch": 71, "voice": "upper"}]},
    ]
    phase4 = realize_phase4(
        Phase4Request(document, phase3_state, phase3_ledger),
        SequencedRunner(phase4_responses),
        phase4_dir,
    )
    assert phase4.realized
    phase4_state = json.loads(
        (phase4_dir / "outputs" / "phase4-state.json").read_text(encoding="utf-8")
    )
    cumulative = json.loads(
        (phase4_dir / "outputs" / "projection-ledger.json").read_text(encoding="utf-8")
    )
    return phase3_state, phase4_state, cumulative


def _response(at_units: int = 0) -> dict[str, object]:
    return {
        "events": [
            {
                "at_units": at_units,
                "preferred_duration_units": 6,
                "degree": "root",
                "preferred_register_zone": "bass",
                "voice": "lower",
                "articulations": ["normal"],
            }
        ]
    }


def _unplaceable_response() -> dict[str, object]:
    response = _response()
    events = response["events"]
    assert isinstance(events, list)
    events[0]["degree"] = "seventh"
    return response


def test_phase5_realization_completes_every_accompaniment_operation(tmp_path: Path) -> None:
    document = _document()
    phase3_state, phase4_state, ledger = _upstream(tmp_path)
    runner = SequencedRunner([_response(), _response(6)])
    run_dir = tmp_path / "phase5"

    result = realize_phase5(
        Phase5Request(document, phase3_state, phase4_state, ledger),
        runner,
        run_dir,
    )

    assert result.realized
    assert result.outcome == "complete"
    assert len(runner.prompts) == 2
    state = json.loads((run_dir / "outputs" / "phase5-state.json").read_text(encoding="utf-8"))
    assert set(state["requests_by_material_placement"]) == {
        "support-first",
        "support-return",
    }
    assert set(state["notes_by_material_placement"]) == {
        "support-first",
        "support-return",
    }
    cumulative = json.loads(
        (run_dir / "outputs" / "projection-ledger.json").read_text(encoding="utf-8")
    )
    assert len(cumulative) > len(ledger)
    assert cumulative[-len(state["projection_ledger"]) :] == state["projection_ledger"]


def test_phase5_repairs_a_symbolic_event_that_cannot_be_placed(tmp_path: Path) -> None:
    document = _document()
    phase3_state, phase4_state, ledger = _upstream(tmp_path)
    runner = SequencedRunner([_unplaceable_response(), _response(), _response(6)])

    result = realize_phase5(
        Phase5Request(document, phase3_state, phase4_state, ledger),
        runner,
        tmp_path / "phase5",
    )

    assert result.realized
    assert len(runner.prompts) == 3
    assert "requested degree is unavailable" in runner.prompts[1]


def test_phase5_stops_after_a_failed_content_repair(tmp_path: Path) -> None:
    document = _document()
    phase3_state, phase4_state, ledger = _upstream(tmp_path)
    runner = SequencedRunner([_unplaceable_response(), _unplaceable_response()])

    result = realize_phase5(
        Phase5Request(document, phase3_state, phase4_state, ledger),
        runner,
        tmp_path / "phase5",
    )

    assert result.outcome == "content_invalid"
    assert len(runner.prompts) == 2
    assert not (tmp_path / "phase5" / "attempts" / "accompaniment-support-return").exists()


def test_phase5_resumes_at_the_first_unaccepted_accompaniment(tmp_path: Path) -> None:
    document = _document()
    phase3_state, phase4_state, ledger = _upstream(tmp_path)
    run_dir = tmp_path / "phase5"

    paused = realize_phase5(
        Phase5Request(document, phase3_state, phase4_state, ledger),
        SequencedRunner([_response()]),
        run_dir,
        max_new_operations=1,
    )
    resumed_runner = SequencedRunner([_response(6)])
    completed = realize_phase5(
        Phase5Request(document, phase3_state, phase4_state, ledger),
        resumed_runner,
        run_dir,
    )

    assert paused.outcome == "paused"
    assert completed.outcome == "complete"
    assert len(resumed_runner.prompts) == 1


def test_phase5_rejects_a_phase4_state_from_a_different_script(tmp_path: Path) -> None:
    document = _document()
    phase3_state, phase4_state, ledger = _upstream(tmp_path)
    phase4_state["input_script_sha256"] = "0" * 64
    runner = SequencedRunner([])

    result = realize_phase5(
        Phase5Request(document, phase3_state, phase4_state, ledger),
        runner,
        tmp_path / "phase5",
    )

    assert result.outcome == "request_invalid"
    assert runner.prompts == []
    assert "does not match the validated script" in result.issues[0].message


def test_phase5_records_preflight_failure_without_model_use(tmp_path: Path) -> None:
    document = _document()
    phase3_state, phase4_state, ledger = _upstream(tmp_path)

    result = realize_phase5(
        Phase5Request(document, phase3_state, phase4_state, ledger),
        PreflightFailingRunner(),
        tmp_path / "phase5",
    )

    assert result.outcome == "runner_failed"
    assert result.issues[0].path == "/runner"


def test_phase5_requires_a_positive_resume_limit(tmp_path: Path) -> None:
    document = _document()

    try:
        realize_phase5(
            Phase5Request(document, {}, {}, ()),
            SequencedRunner([]),
            tmp_path / "phase5",
            max_new_operations=0,
        )
    except ValueError as error:
        assert str(error) == "max_new_operations must be positive"
    else:
        raise AssertionError("a non-positive resume limit must be rejected")
