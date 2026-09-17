import json
from pathlib import Path

import mido
import pytest

from scoim.phase3_realization import Phase3Request, realize_phase3
from scoim.phase3_state import load_complete_phase3_state
from scoim.phase4_realization import Phase4Request, realize_phase4
from scoim.phase4_state import load_complete_phase4_state
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
            ).encode("utf-8"),
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
        for target_kind, raw_values in collections.items()
        for target_id in raw_values
    )


def _phase3_responses() -> list[dict[str, object]]:
    return [
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


def _phase4_responses() -> list[dict[str, object]]:
    return [
        {"notes": [{"at_units": 0, "duration_units": 12, "pitch": 72, "voice": "upper"}]},
        {"notes": [{"at_units": 0, "duration_units": 8, "pitch": 72, "voice": "upper"}]},
        {"notes": [{"at_units": 0, "duration_units": 12, "pitch": 60, "voice": "lower"}]},
        {"notes": [{"at_units": 0, "duration_units": 12, "pitch": 71, "voice": "upper"}]},
    ]


def _phase3_inputs(
    tmp_path: Path, document: dict[str, object] | None = None
) -> tuple[dict[str, object], list[dict[str, object]]]:
    document = document or _document()
    run_dir = tmp_path / "phase3"
    result = realize_phase3(
        Phase3Request(document, _phase2_ledger(document)),
        SequencedRunner(_phase3_responses()),
        run_dir,
    )
    assert result.realized
    state = json.loads((run_dir / "outputs" / "phase3-state.json").read_text("utf-8"))
    ledger = json.loads((run_dir / "outputs" / "projection-ledger.json").read_text("utf-8"))
    return state, ledger


def _phase4_inputs(
    tmp_path: Path,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    document = _document()
    phase3_state, phase3_ledger = _phase3_inputs(tmp_path)
    run_dir = tmp_path / "phase4-inputs"
    result = realize_phase4(
        Phase4Request(document, phase3_state, phase3_ledger),
        SequencedRunner(_phase4_responses()),
        run_dir,
    )
    assert result.realized
    phase4_state = json.loads((run_dir / "outputs" / "phase4-state.json").read_text("utf-8"))
    cumulative = json.loads((run_dir / "outputs" / "projection-ledger.json").read_text("utf-8"))
    return phase3_state, phase3_ledger, phase4_state, cumulative


def _corrupt_phase3_state(state: dict[str, object], case: str) -> str:
    plan = state["harmonic_plan"]
    assert isinstance(plan, dict)
    if case == "piece-plan":
        piece_plan = plan["piece_plan"]
        assert isinstance(piece_plan, dict)
        piece_plan["title"] = "changed"
        return "the phase-3 piece plan does not match the script"
    if case == "lengths":
        lengths = plan["length_units_by_score_unit"]
        assert isinstance(lengths, dict)
        lengths.pop(next(iter(lengths)))
        return "phase-3 lengths do not exactly cover score units"
    if case == "intents":
        intents = plan["section_intents"]
        assert isinstance(intents, list)
        intents.pop()
        return "phase-3 intents do not exactly cover score units"
    harmonies = state["harmonies_by_score_unit"]
    assert isinstance(harmonies, dict)
    if case == "harmony-units":
        harmonies.pop(next(iter(harmonies)))
        return "phase-3 harmonies do not exactly cover score units"
    values = next(iter(harmonies.values()))
    assert isinstance(values, list)
    first = values[0]
    assert isinstance(first, dict)
    if case == "harmony-gap":
        first["at_units"] = 1
        return "phase-3 harmony has a gap or overlap"
    first["duration_units"] -= 1
    return "phase-3 harmony does not cover its score unit"


def test_phase4_realization_completes_every_foreground_operation(tmp_path: Path) -> None:
    document = _document()
    phase3_state, ledger = _phase3_inputs(tmp_path)
    runner = SequencedRunner(_phase4_responses())
    run_dir = tmp_path / "phase4"

    result = realize_phase4(
        Phase4Request(document, phase3_state, ledger),
        runner,
        run_dir,
    )

    assert result.realized is True
    assert result.outcome == "complete"
    assert len(runner.prompts) == 4
    state = json.loads((run_dir / "outputs" / "phase4-state.json").read_text("utf-8"))
    assert set(state["notes_by_material_placement"]) == {
        "theme-first",
        "theme-return",
        "ending-only",
        "bridge-only",
    }
    assert (run_dir / "outputs" / "projection-ledger.json").is_file()
    assert (run_dir / "outputs" / "foreground-preview.mid").read_bytes()[:4] == b"MThd"


def test_phase3_state_loader_returns_checked_shared_harmony(tmp_path: Path) -> None:
    document = _document()
    phase3_state, ledger = _phase3_inputs(tmp_path)

    loaded = load_complete_phase3_state(document, phase3_state, ledger)

    assert loaded.plan.piece_plan.title == "基本縦断"
    assert set(loaded.harmonies_by_score_unit) == {
        "score-unit-statement",
        "score-unit-bridge",
        "score-unit-return",
        "score-unit-release",
    }
    assert len(loaded.cumulative_projection_ledger) == len(ledger)


def test_phase4_state_loader_accepts_the_complete_matching_run(tmp_path: Path) -> None:
    document = _document()
    phase3_state, phase3_ledger = _phase3_inputs(tmp_path)
    run_dir = tmp_path / "phase4"
    realized = realize_phase4(
        Phase4Request(document, phase3_state, phase3_ledger),
        SequencedRunner(_phase4_responses()),
        run_dir,
    )
    assert realized.realized
    phase4_state = json.loads((run_dir / "outputs" / "phase4-state.json").read_text("utf-8"))
    phase4_ledger = json.loads((run_dir / "outputs" / "projection-ledger.json").read_text("utf-8"))

    loaded = load_complete_phase4_state(
        document,
        phase3_state,
        phase4_state,
        phase4_ledger,
    )

    assert set(loaded.notes_by_material_placement) == {
        "theme-first",
        "theme-return",
        "ending-only",
        "bridge-only",
    }
    assert len(loaded.cumulative_projection_ledger) == len(phase4_ledger)


def test_phase4_state_loader_rejects_an_incomplete_state(tmp_path: Path) -> None:
    document = _document()
    phase3_state, _phase3_ledger, phase4_state, cumulative = _phase4_inputs(tmp_path)
    phase4_state["outcome"] = "content_invalid"

    with pytest.raises(ValueError, match="complete phase-4 state"):
        load_complete_phase4_state(document, phase3_state, phase4_state, cumulative)


def test_phase4_state_loader_rejects_a_different_phase3_state(tmp_path: Path) -> None:
    document = _document()
    phase3_state, _phase3_ledger, phase4_state, cumulative = _phase4_inputs(tmp_path)
    plan = phase3_state["harmonic_plan"]
    assert isinstance(plan, dict)
    plan["overall_harmonic_story"] = "changed"

    with pytest.raises(ValueError, match="does not match the phase-3 state"):
        load_complete_phase4_state(document, phase3_state, phase4_state, cumulative)


def test_phase4_state_loader_rejects_a_detached_local_ledger(tmp_path: Path) -> None:
    document = _document()
    phase3_state, _phase3_ledger, phase4_state, cumulative = _phase4_inputs(tmp_path)
    local = phase4_state["projection_ledger"]
    assert isinstance(local, list)
    local[0]["evidence"] = "changed"

    with pytest.raises(ValueError, match="does not match the cumulative projection ledger"):
        load_complete_phase4_state(document, phase3_state, phase4_state, cumulative)


def test_phase4_state_loader_rejects_a_rewritten_local_ledger(tmp_path: Path) -> None:
    document = _document()
    phase3_state, _phase3_ledger, phase4_state, cumulative = _phase4_inputs(tmp_path)
    local = phase4_state["projection_ledger"]
    assert isinstance(local, list)
    local[0]["source_id"] = "other-placement"
    cumulative[-len(local)]["source_id"] = "other-placement"

    with pytest.raises(ValueError, match="phase-4 projection ledger does not match"):
        load_complete_phase4_state(document, phase3_state, phase4_state, cumulative)


def test_phase4_state_loader_rejects_a_missing_foreground_placement(tmp_path: Path) -> None:
    document = _document()
    phase3_state, _phase3_ledger, phase4_state, cumulative = _phase4_inputs(tmp_path)
    notes = phase4_state["notes_by_material_placement"]
    assert isinstance(notes, dict)
    notes.pop("theme-first")

    with pytest.raises(ValueError, match="exactly cover foreground placements"):
        load_complete_phase4_state(document, phase3_state, phase4_state, cumulative)


def test_phase4_state_loader_rejects_duplicate_foreground_note_ids(tmp_path: Path) -> None:
    document = _document()
    phase3_state, _phase3_ledger, phase4_state, cumulative = _phase4_inputs(tmp_path)
    notes = phase4_state["notes_by_material_placement"]
    assert isinstance(notes, dict)
    first = notes["theme-first"][0]
    notes["theme-return"][0]["score_note_id"] = first["score_note_id"]

    with pytest.raises(ValueError, match="globally unique"):
        load_complete_phase4_state(document, phase3_state, phase4_state, cumulative)


def test_phase4_preview_keeps_an_accompaniment_only_tail_silent(tmp_path: Path) -> None:
    document = _document()
    script = document["script"]
    assert isinstance(script, dict)
    placements = script["material_placements"]
    assert isinstance(placements, dict)
    placements["ending-only"]["role"] = "accompaniment"
    phase3_state, ledger = _phase3_inputs(tmp_path, document)
    responses = [_phase4_responses()[0], _phase4_responses()[1], _phase4_responses()[3]]
    run_dir = tmp_path / "phase4"

    result = realize_phase4(
        Phase4Request(document, phase3_state, ledger),
        SequencedRunner(responses),
        run_dir,
    )

    assert result.realized
    midi = mido.MidiFile(run_dir / "outputs" / "foreground-preview.mid")
    assert midi.length == 180.0


def test_phase4_stops_before_later_operations_after_failed_content_repair(
    tmp_path: Path,
) -> None:
    document = _document()
    phase3_state, ledger = _phase3_inputs(tmp_path)
    invalid = {"notes": [{"at_units": 47, "duration_units": 2, "pitch": 72, "voice": "upper"}]}
    runner = SequencedRunner([invalid, invalid])
    run_dir = tmp_path / "phase4"

    result = realize_phase4(Phase4Request(document, phase3_state, ledger), runner, run_dir)

    assert result.outcome == "content_invalid"
    assert len(runner.prompts) == 2
    assert not (run_dir / "attempts" / "foreground-theme-return").exists()


def test_phase4_repairs_a_schema_invalid_response_before_content_validation(
    tmp_path: Path,
) -> None:
    document = _document()
    phase3_state, ledger = _phase3_inputs(tmp_path)
    valid_responses = _phase4_responses()
    runner = SequencedRunner([{}, *valid_responses])
    run_dir = tmp_path / "phase4"

    result = realize_phase4(Phase4Request(document, phase3_state, ledger), runner, run_dir)

    assert result.outcome == "complete"
    assert len(runner.prompts) == 5
    assert "required property" in runner.prompts[1]


def test_phase4_resumes_at_the_first_unaccepted_foreground(tmp_path: Path) -> None:
    document = _document()
    phase3_state, ledger = _phase3_inputs(tmp_path)
    run_dir = tmp_path / "phase4"

    paused = realize_phase4(
        Phase4Request(document, phase3_state, ledger),
        SequencedRunner(_phase4_responses()[:2]),
        run_dir,
        max_new_operations=2,
    )
    second_runner = SequencedRunner(_phase4_responses()[2:])
    completed = realize_phase4(
        Phase4Request(document, phase3_state, ledger),
        second_runner,
        run_dir,
    )

    assert paused.outcome == "paused"
    assert completed.outcome == "complete"
    assert len(second_runner.prompts) == 2


def test_phase4_rejects_a_missing_cumulative_projection_before_model_use(
    tmp_path: Path,
) -> None:
    document = _document()
    phase3_state, ledger = _phase3_inputs(tmp_path)
    runner = SequencedRunner([])

    result = realize_phase4(
        Phase4Request(document, phase3_state, ledger[1:]),
        runner,
        tmp_path / "phase4",
    )

    assert result.outcome == "request_invalid"
    assert runner.prompts == []
    assert "missing projection target" in result.issues[0].message


def test_phase4_records_preflight_failure_without_calling_the_model(tmp_path: Path) -> None:
    document = _document()
    phase3_state, ledger = _phase3_inputs(tmp_path)

    result = realize_phase4(
        Phase4Request(document, phase3_state, ledger),
        PreflightFailingRunner(),
        tmp_path / "phase4",
    )

    assert result.outcome == "runner_failed"
    assert result.issues[0].path == "/runner"


def test_phase4_rejects_an_unsupported_profile_before_model_use(tmp_path: Path) -> None:
    document = _document()
    phase3_state, ledger = _phase3_inputs(tmp_path)
    runner = SequencedRunner([])

    result = realize_phase4(
        Phase4Request(document, phase3_state, ledger, target_profile="unknown"),
        runner,
        tmp_path / "phase4",
    )

    assert result.outcome == "request_invalid"
    assert runner.prompts == []
    assert result.issues[0].message == "the phase-4 profile is unsupported"


def test_phase4_rejects_incomplete_phase3_state_before_model_use(tmp_path: Path) -> None:
    document = _document()
    phase3_state, ledger = _phase3_inputs(tmp_path)
    phase3_state["outcome"] = "content_invalid"
    runner = SequencedRunner([])

    result = realize_phase4(
        Phase4Request(document, phase3_state, ledger),
        runner,
        tmp_path / "phase4",
    )

    assert result.outcome == "request_invalid"
    assert runner.prompts == []
    assert result.issues[0].message == "phase 4 requires a complete phase-3 state"


def test_phase4_requires_a_positive_resume_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"^max_new_operations must be positive$"):
        realize_phase4(
            Phase4Request({}, {}, ()),
            SequencedRunner([]),
            tmp_path / "phase4",
            max_new_operations=0,
        )


@pytest.mark.parametrize(
    "case",
    ("piece-plan", "lengths", "intents", "harmony-units", "harmony-gap", "harmony-cover"),
)
def test_phase4_rejects_corrupt_phase3_musical_state_before_model_use(
    tmp_path: Path, case: str
) -> None:
    document = _document()
    phase3_state, ledger = _phase3_inputs(tmp_path)
    expected_message = _corrupt_phase3_state(phase3_state, case)
    runner = SequencedRunner([])

    result = realize_phase4(
        Phase4Request(document, phase3_state, ledger),
        runner,
        tmp_path / f"phase4-{case}",
    )

    assert result.outcome == "request_invalid"
    assert runner.prompts == []
    assert result.issues[0].message == expected_message


def test_phase4_repairs_a_collision_with_an_accepted_layer(tmp_path: Path) -> None:
    document = _document()
    script = document["script"]
    assert isinstance(script, dict)
    materials = script["materials"]
    placements = script["material_placements"]
    assert isinstance(materials, dict)
    assert isinstance(placements, dict)
    materials["counter"] = {"description": "同じ区分の対旋律。"}
    placements["z-counter"] = {
        "section_id": "statement",
        "material_id": "counter",
        "role": "foreground",
    }
    phase3_state, ledger = _phase3_inputs(tmp_path, document)
    collided = {"notes": [{"at_units": 6, "duration_units": 12, "pitch": 72, "voice": "upper"}]}
    repaired = {"notes": [{"at_units": 6, "duration_units": 12, "pitch": 76, "voice": "upper"}]}
    responses = [
        _phase4_responses()[0],
        collided,
        repaired,
        *_phase4_responses()[1:],
    ]
    runner = SequencedRunner(responses)

    result = realize_phase4(
        Phase4Request(document, phase3_state, ledger),
        runner,
        tmp_path / "phase4",
    )

    assert result.realized
    assert len(runner.prompts) == 6
    assert '"occupied_notes": [{"at_units": 0' in runner.prompts[1]
    repair_context = json.loads(runner.prompts[2].split("入力: ", 1)[1])
    assert '"occupied_notes": [{"at_units": 0' in repair_context["original_prompt"]
    assert repair_context["all_issues"][0]["path"] == "/notes/0"
    assert (
        "candidate=[6,18), occupied=[0,12), pitch=72, voice=upper"
        in repair_context["all_issues"][0]["message"]
    )


def test_phase4_uses_transition_boundaries_as_context_not_copy_constraints(
    tmp_path: Path,
) -> None:
    document = _document()
    phase3_state, ledger = _phase3_inputs(tmp_path)
    responses = _phase4_responses()
    responses[-1] = responses[0]
    runner = SequencedRunner(responses)

    result = realize_phase4(
        Phase4Request(document, phase3_state, ledger),
        runner,
        tmp_path / "phase4",
    )

    assert result.realized
    assert len(runner.prompts) == 4


def test_phase4_rejects_a_phase3_ledger_not_bound_to_the_cumulative_ledger(
    tmp_path: Path,
) -> None:
    document = _document()
    phase3_state, ledger = _phase3_inputs(tmp_path)
    phase3_state["projection_ledger"][0]["evidence"] = "changed"
    runner = SequencedRunner([])

    result = realize_phase4(
        Phase4Request(document, phase3_state, ledger),
        runner,
        tmp_path / "phase4",
    )

    assert result.outcome == "request_invalid"
    assert runner.prompts == []
    assert "cumulative projection ledger" in result.issues[0].message
