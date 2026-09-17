import json
from pathlib import Path
from typing import cast

import pytest

from scoim.phase3_realization import Phase3Request, realize_phase3
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
        response = self.responses[len(self.prompts) - 1]
        return ProposalRun(
            provider="fixed",
            model="fixed",
            model_settings={},
            started=True,
            terminal_state="completed",
            raw_response=json.dumps(response, ensure_ascii=False).encode("utf-8"),
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
    script = cast(dict[str, object], document["script"])
    collections = {
        "sections": cast(dict[str, object], script["sections"]),
        "materials": cast(dict[str, object], script["materials"]),
        "material_placements": cast(dict[str, object], script["material_placements"]),
        "script_element_variation_relations": cast(
            dict[str, object], script["script_element_variation_relations"]
        ),
        "material_placement_transitions": cast(
            dict[str, object], script["material_placement_transitions"]
        ),
        "performance_directions": cast(
            dict[str, object],
            cast(dict[str, object], script["performance_setup"])["performance_directions"],
        ),
    }
    if "performance_direction_comparison_requirements" in script:
        collections["performance_direction_comparison_requirements"] = cast(
            dict[str, object], script["performance_direction_comparison_requirements"]
        )
    return tuple(
        ProjectionLedgerEntry(
            source_kind="phase2_source",
            source_id=target_id,
            target_kind=target_kind,
            target_id=target_id,
            target_stage="script",
            verification="direct_id_equality",
            status="passed",
            evidence=f"target_id={target_id}",
        )
        for target_kind, values in collections.items()
        for target_id in values
    )


def _responses() -> list[dict[str, object]]:
    return [
        {
            "tonal_center": 0,
            "mode": "major",
            "overall_harmonic_story": "主調から少し離れて戻る。",
            "section_harmonic_intents": [
                {
                    "harmonic_intent": "主調を示す。",
                    "connection_from_previous": "静かに始める。",
                },
                {
                    "harmonic_intent": "属和音へ動く。",
                    "connection_from_previous": "緊張を高める。",
                },
                {
                    "harmonic_intent": "主題を支える。",
                    "connection_from_previous": "緊張を解く。",
                },
                {
                    "harmonic_intent": "主調へ着地する。",
                    "connection_from_previous": "余韻を保つ。",
                },
            ],
        },
        {"harmonies": [{"duration_units": 48, "root_pitch_class": 0, "quality": "major"}]},
        {"harmonies": [{"duration_units": 12, "root_pitch_class": 7, "quality": "major"}]},
        {"harmonies": [{"duration_units": 48, "root_pitch_class": 5, "quality": "major"}]},
        {"harmonies": [{"duration_units": 12, "root_pitch_class": 0, "quality": "major"}]},
    ]


def test_phase3_realization_completes_all_registered_operations(tmp_path: Path) -> None:
    document = _document()
    runner = SequencedRunner(_responses())
    run_dir = tmp_path / "phase3"

    result = realize_phase3(
        Phase3Request(document, _phase2_ledger(document)),
        runner,
        run_dir,
    )

    assert result.realized is True
    assert result.outcome == "complete"
    assert len(runner.prompts) == 5
    state = json.loads((run_dir / "outputs" / "phase3-state.json").read_text("utf-8"))
    assert set(state["harmonies_by_score_unit"]) == {
        "score-unit-statement",
        "score-unit-bridge",
        "score-unit-return",
        "score-unit-release",
    }
    spec = json.loads((run_dir / "run-spec.json").read_text("utf-8"))
    assert spec["operation_order"] == [
        "overall-plan",
        "harmony-score-unit-statement",
        "harmony-score-unit-bridge",
        "harmony-score-unit-return",
        "harmony-score-unit-release",
    ]
    assert state["harmonies_by_score_unit"]["score-unit-release"][-1]["root_pitch_class"] == 0


def test_phase3_rejects_a_missing_phase2_projection_before_model_use(
    tmp_path: Path,
) -> None:
    document = _document()
    ledger = _phase2_ledger(document)[1:]
    runner = SequencedRunner([])

    result = realize_phase3(Phase3Request(document, ledger), runner, tmp_path / "run")

    assert result.outcome == "request_invalid"
    assert runner.prompts == []
    assert "missing projection target" in result.issues[0].message


def test_phase3_rejects_an_extra_phase2_projection_before_model_use(
    tmp_path: Path,
) -> None:
    document = _document()
    extra = ProjectionLedgerEntry(
        source_kind="phase2_source",
        source_id="unexpected",
        target_kind="sections",
        target_id="unexpected",
        target_stage="script",
        verification="direct_id_equality",
        status="passed",
        evidence="target_id=unexpected",
    )
    runner = SequencedRunner([])

    result = realize_phase3(
        Phase3Request(document, (*_phase2_ledger(document), extra)),
        runner,
        tmp_path / "run",
    )

    assert result.outcome == "request_invalid"
    assert runner.prompts == []
    assert "unexpected projection target" in result.issues[0].message


def test_phase3_rejects_a_duplicate_phase2_projection_before_model_use(
    tmp_path: Path,
) -> None:
    document = _document()
    ledger = _phase2_ledger(document)
    runner = SequencedRunner([])

    result = realize_phase3(
        Phase3Request(document, (*ledger, ledger[0])),
        runner,
        tmp_path / "run",
    )

    assert result.outcome == "request_invalid"
    assert runner.prompts == []
    assert "duplicate projection target" in result.issues[0].message


def test_phase3_stops_before_later_harmony_operations_after_content_failure(
    tmp_path: Path,
) -> None:
    document = _document()
    invalid_harmony = {
        "harmonies": [{"duration_units": 1, "root_pitch_class": 0, "quality": "major"}]
    }
    runner = SequencedRunner([_responses()[0], invalid_harmony, invalid_harmony])

    result = realize_phase3(
        Phase3Request(document, _phase2_ledger(document)),
        runner,
        tmp_path / "run",
    )

    assert result.outcome == "content_invalid"
    assert len(runner.prompts) == 3
    assert not (tmp_path / "run" / "attempts" / "harmony-score-unit-bridge").exists()


def test_phase3_resumes_from_the_first_unaccepted_operation(tmp_path: Path) -> None:
    document = _document()
    run_dir = tmp_path / "run"
    first_runner = SequencedRunner(_responses()[:2])

    paused = realize_phase3(
        Phase3Request(document, _phase2_ledger(document)),
        first_runner,
        run_dir,
        max_new_operations=2,
    )
    second_runner = SequencedRunner(_responses()[2:])
    completed = realize_phase3(
        Phase3Request(document, _phase2_ledger(document)),
        second_runner,
        run_dir,
    )

    assert paused.outcome == "paused"
    assert completed.outcome == "complete"
    assert len(first_runner.prompts) == 2
    assert len(second_runner.prompts) == 3


@pytest.mark.parametrize(
    ("request_changes", "expected_path"),
    [
        ({"target_profile": "unknown"}, "/target_profile"),
        ({"divisions": 0}, "/time_grid"),
    ],
)
def test_phase3_rejects_unsupported_run_settings(
    tmp_path: Path,
    request_changes: dict[str, object],
    expected_path: str,
) -> None:
    document = _document()
    request_values = {
        "validated_script": document,
        "projection_ledger": _phase2_ledger(document),
        **request_changes,
    }

    result = realize_phase3(
        Phase3Request(**request_values),
        SequencedRunner([]),
        tmp_path / "run",
    )

    assert result.outcome == "request_invalid"
    assert result.issues[0].path == expected_path


def test_phase3_records_preflight_failure_without_calling_the_model(tmp_path: Path) -> None:
    document = _document()
    result = realize_phase3(
        Phase3Request(document, _phase2_ledger(document)),
        PreflightFailingRunner(),
        tmp_path / "run",
    )

    assert result.outcome == "runner_failed"
    assert result.issues[0].message == "runner unavailable"
    assert not (tmp_path / "run" / "attempts").exists()


def test_phase3_stops_when_the_overall_plan_remains_invalid_after_repair(
    tmp_path: Path,
) -> None:
    document = _document()
    invalid_overall = _responses()[0]
    invalid_overall["section_harmonic_intents"] = invalid_overall["section_harmonic_intents"][:-1]
    runner = SequencedRunner([invalid_overall, invalid_overall])

    result = realize_phase3(
        Phase3Request(document, _phase2_ledger(document)),
        runner,
        tmp_path / "run",
    )

    assert result.outcome == "content_invalid"
    assert len(runner.prompts) == 2
    assert not (tmp_path / "run" / "outputs" / "piece-plan.json").exists()
