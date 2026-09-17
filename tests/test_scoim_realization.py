import json
from dataclasses import replace
from pathlib import Path

import jsonpointer
from test_scoim_projection import (
    _approved_document,
    _approved_relational_document,
    _approved_relational_with_requirement,
)

from llm_musical_composer.performance_pipeline import (
    NodePerformance,
    PerformanceSpec,
    ScoreHarmony,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
)
from llm_musical_composer.pipeline_dsl import (
    dump_performance_spec,
    dump_score_spec,
    parse_performance_spec,
    parse_score_spec,
)
from scoim.operations import apply_patch, approve
from scoim.projection import PlanChoice, project_solo_piano_3m
from scoim.realization import realize_solo_piano_3m
from scoim.validation import IssueCode

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "scoim" / "fixed-aba"
_NESTED_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "scoim" / "nested-aba"


def _build_fixed_response(document: dict[str, object]) -> dict[str, object]:
    choice = PlanChoice(
        tonal_center=9,
        mode="minor",
        harmonic_focus_by_section={
            "statement": 9,
            "bridge": 9,
            "contrast": 4,
            "return": 9,
            "release": 9,
        },
        contrasts_with_by_section={"contrast": "statement"},
    )
    theme = ScoreMaterial(
        "theme",
        8,
        (
            ScoreNote("t-l0", 0, 4, 45, "lower"),
            ScoreNote("t-l1", 0, 4, 52, "lower"),
            ScoreNote("t-u0", 0, 4, 69, "upper"),
            ScoreNote("t-u1", 4, 4, 69, "upper", articulations=("tenuto",)),
            ScoreNote("t-l2", 4, 4, 52, "lower", articulations=("tenuto",)),
        ),
        harmonies=(ScoreHarmony("theme-a", 0, 8, 9, "minor"),),
        foreground_voice="upper",
    )
    connector = ScoreMaterial(
        "connector",
        2,
        (
            ScoreNote("b-l0", 0, 2, 45, "lower"),
            ScoreNote("b-l1", 0, 2, 52, "lower"),
            ScoreNote("b-u0", 0, 2, 69, "upper"),
        ),
        harmonies=(ScoreHarmony("bridge-a", 0, 2, 9, "minor"),),
        foreground_voice="upper",
    )
    contrast = ScoreMaterial(
        "contrast",
        8,
        (
            ScoreNote("c-l0", 0, 4, 40, "lower"),
            ScoreNote("c-l1", 0, 4, 47, "lower"),
            ScoreNote("c-u0", 0, 2, 64, "upper"),
            ScoreNote("c-u1", 2, 2, 68, "upper"),
            ScoreNote("c-u2", 4, 4, 71, "upper"),
        ),
        harmonies=(ScoreHarmony("contrast-e", 0, 8, 4, "major"),),
        foreground_voice="upper",
    )
    cadence = ScoreMaterial(
        "cadence",
        4,
        (
            ScoreNote("e-l0", 0, 4, 45, "lower", articulations=("tenuto",)),
            ScoreNote("e-l1", 0, 4, 52, "lower", articulations=("tenuto",)),
            ScoreNote("e-u0", 0, 4, 69, "upper", articulations=("tenuto",)),
        ),
        harmonies=(ScoreHarmony("ending-a", 0, 4, 9, "minor"),),
        foreground_voice="upper",
    )
    score = ScoreSpec("fixed-aba-score", 4, (theme, connector, contrast, cadence))
    performance = PerformanceSpec(
        "fixed-aba-performance",
        180_000,
        64,
        "narrative-v2",
        (
            NodePerformance(
                "statement", "savor", "subtle", "shape", "legato", "rolled", "harmony_legato"
            ),
            NodePerformance(
                "bridge", "neutral", "subtle", "steady", "score", "aligned", "harmony_legato"
            ),
            NodePerformance(
                "contrast", "build", "subtle", "build", "score", "score", "harmony_legato"
            ),
            NodePerformance(
                "return", "flow", "subtle", "steady", "legato", "aligned", "harmony_legato"
            ),
            NodePerformance(
                "release", "release", "subtle", "release", "legato", "aligned", "harmony_legato"
            ),
        ),
    )
    projection = project_solo_piano_3m(document, plan_choice=choice)
    assert projection.projected
    assertions = {}
    for target in projection.targets:
        if target.verification not in {
            "translated_mechanical_claim",
            "targeted_performance_difference",
        }:
            continue
        escaped_target = jsonpointer.escape(target.target_id)
        parts = jsonpointer.JsonPointer(target.source_path).parts
        expected_by_stage: dict[str, tuple[str, object]] = {}
        if len(parts) < 2 or parts[1] == "brief":
            expected_by_stage = {
                "piece_plan": ("has_recurrence", True),
                "score_spec": ("has_upper_and_lower", True),
                "performance_spec": ("duration_ms", 180_000),
            }
        elif parts[1] == "sections":
            expected_by_stage = {
                "piece_plan": ("node/node_id", parts[2]),
            }
        elif parts[1] == "materials":
            expected_by_stage = {
                "score_spec": ("material/material_id", parts[2]),
            }
        elif parts[1] == "variations":
            expected_by_stage = {
                "piece_plan": ("derived_from_matches", True),
                "score_spec": ("same_score_material", True),
                "performance_spec": ("performance_difference", True),
            }
        elif parts[1] == "transitions":
            expected_by_stage = {
                "piece_plan": ("adjacent_in_order", True),
                "score_spec": ("adjacent_in_order", True),
            }
        elif "performance_directions" in parts:
            direction_id = parts[-2]
            if direction_id == "quiet_opening":
                expected_by_stage = {
                    "performance_spec": ("effective_profiles/timing_profile", "savor"),
                }
            elif direction_id == "clear_return":
                expected_by_stage = {
                    "performance_spec": ("different_profile_count", 3),
                }
        assertions[target.target_id] = {
            "interpretation": f"{target.source_path}の意図が各生成段階の事実に反映される。",
            "tests": {
                stage: [
                    {
                        "op": "test",
                        "path": (
                            f"/target_evidence/{escaped_target}/{stage}/"
                            f"{expected_by_stage.get(stage, ('entity_present', True))[0]}"
                        ),
                        "value": expected_by_stage.get(stage, ("entity_present", True))[1],
                    }
                ]
                for stage in target.generation_stages
            },
        }
    return {
        "schema_version": 1,
        "profile": "solo_piano_3m_v1",
        "plan_choice": {
            "tonal_center": choice.tonal_center,
            "mode": choice.mode,
            "harmonic_focus_by_section": dict(choice.harmonic_focus_by_section),
            "contrasts_with_by_section": dict(choice.contrasts_with_by_section),
        },
        "score_spec": {"format": "pipeline-dsl-v1", "source": dump_score_spec(score)},
        "performance_spec": {
            "format": "pipeline-dsl-v1",
            "source": dump_performance_spec(performance),
        },
        "target_assertions": assertions,
    }


def _fixed_response(document: dict[str, object]) -> dict[str, object]:
    approved = json.loads((_FIXTURE_DIR / "approved-script.json").read_text(encoding="utf-8"))
    assert document == approved
    return json.loads((_FIXTURE_DIR / "frozen-response.json").read_text(encoding="utf-8"))


def test_fixed_fixture_matches_the_current_projection_contract() -> None:
    document = _approved_relational_document()

    assert _fixed_response(document) == _build_fixed_response(document)


def test_fixed_revision_2_fixture_is_publicly_derived_and_reuses_the_frozen_response(
    tmp_path: Path,
) -> None:
    revision_1 = _approved_relational_document()
    applied = apply_patch(
        revision_1,
        [
            {"op": "test", "path": "/revision", "value": 1},
            {"op": "replace", "path": "/revision", "value": 2},
            {
                "op": "add",
                "path": "/script/requirements",
                "value": {
                    "return_more_aligned_than_opening": {
                        "performance_direction_id": "clear_return",
                        "feature": "onset_alignment",
                        "relation": "more",
                    }
                },
            },
        ],
        new_draft=True,
    )
    assert applied.document is not None
    approved = approve(applied.document)
    assert approved.document is not None
    fixture = json.loads(
        (_FIXTURE_DIR / "approved-script-revision-2.json").read_text(encoding="utf-8")
    )

    assert fixture == approved.document
    result = realize_solo_piano_3m(
        fixture,
        json.loads((_FIXTURE_DIR / "frozen-response.json").read_text(encoding="utf-8")),
        tmp_path / "output",
    )
    assert result.realized is True
    assert result.outcome.promotion_eligible is True
    assert (
        next(
            item
            for item in result.target_results
            if item["target_id"] == "/script/requirements/return_more_aligned_than_opening"
        )["status"]
        == "passed"
    )


def test_nested_revision_2_fixture_realizes_both_typed_requirements(
    tmp_path: Path,
) -> None:
    revision_1 = json.loads(
        (_NESTED_FIXTURE_DIR / "approved-script.json").read_text(encoding="utf-8")
    )
    applied = apply_patch(
        revision_1,
        [
            {"op": "test", "path": "/revision", "value": 1},
            {"op": "replace", "path": "/revision", "value": 2},
            {
                "op": "add",
                "path": "/script/requirements",
                "value": {
                    "contrast_end_louder_than_start": {
                        "performance_direction_id": "contrast_growth",
                        "feature": "loudness",
                        "relation": "more",
                    },
                    "return_more_aligned_than_opening": {
                        "performance_direction_id": "aligned_return",
                        "feature": "onset_alignment",
                        "relation": "more",
                    },
                },
            },
        ],
        new_draft=True,
    )
    assert applied.document is not None
    approved = approve(applied.document)
    assert approved.document is not None
    fixture = json.loads(
        (_NESTED_FIXTURE_DIR / "approved-script-revision-2.json").read_text(encoding="utf-8")
    )
    response = json.loads(
        (_NESTED_FIXTURE_DIR / "frozen-response-revision-2.json").read_text(encoding="utf-8")
    )

    assert fixture == approved.document
    result = realize_solo_piano_3m(fixture, response, tmp_path / "output-a")
    replay = realize_solo_piano_3m(fixture, response, tmp_path / "output-b")

    assert result.realized is True
    assert result.outcome.promotion_eligible is True
    actual_ids = [item["target_id"] for item in result.target_results]
    assert len(actual_ids) == len(set(actual_ids))
    typed = {
        item["target_id"]: item
        for item in result.target_results
        if item["target_id"].startswith("/script/requirements/")
    }
    assert set(typed) == {
        "/script/requirements/contrast_end_louder_than_start",
        "/script/requirements/return_more_aligned_than_opening",
    }
    assert {item["status"] for item in typed.values()} == {"passed"}
    assert not set(typed).intersection(response["target_assertions"])
    diagnostics = json.loads(result.diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["targets"] == list(result.target_results)
    assert replay.realized is True
    assert replay.target_results == result.target_results
    assert replay.musicxml_path.read_bytes() == result.musicxml_path.read_bytes()
    assert replay.smf_path.read_bytes() == result.smf_path.read_bytes()
    assert replay.diagnostics_path.read_bytes() == result.diagnostics_path.read_bytes()


def test_realization_rejects_an_unknown_frozen_response_version(tmp_path: Path) -> None:
    result = realize_solo_piano_3m(
        _approved_document(),
        {"schema_version": 99},
        tmp_path / "output",
    )

    assert not result.realized
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert result.issues[0].path == "/frozen_response/schema_version"
    assert not (tmp_path / "output").exists()


def test_realization_rejects_unknown_frozen_response_fields(tmp_path: Path) -> None:
    response = {
        "schema_version": 1,
        "profile": "solo_piano_3m_v1",
        "plan_choice": {},
        "score_spec": {},
        "performance_spec": {},
        "target_assertions": {},
        "fallback": "silent",
    }

    result = realize_solo_piano_3m(
        _approved_document(),
        response,
        tmp_path / "output",
    )

    assert not result.realized
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert result.issues[0].path == "/frozen_response/fallback"


def test_realization_rejects_an_unsupported_profile(tmp_path: Path) -> None:
    document = _approved_relational_document()
    response = _fixed_response(document)
    response["profile"] = "string_quartet_3m_v1"

    result = realize_solo_piano_3m(document, response, tmp_path / "output")

    assert not result.realized
    assert result.issues[0].code is IssueCode.UNSUPPORTED_PROFILE
    assert result.issues[0].path == "/frozen_response/profile"


def test_realization_rejects_invalid_plan_choice_fields(tmp_path: Path) -> None:
    document = _approved_relational_document()
    response = _fixed_response(document)
    response["plan_choice"] = {}

    result = realize_solo_piano_3m(document, response, tmp_path / "output")

    assert not result.realized
    assert result.frozen_response_sha256 is not None
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert result.issues[0].path == "/frozen_response/plan_choice"


def test_realization_writes_verified_musicxml_smf_and_diagnostics(tmp_path: Path) -> None:
    document = _approved_relational_document()
    output = tmp_path / "output"

    result = realize_solo_piano_3m(document, _fixed_response(document), output)

    assert result.realized
    assert result.outcome.to_dict() == {
        "terminal_state": "completed",
        "artifact_disposition": "candidate",
        "issues": [],
    }
    assert result.outcome.promotion_eligible
    assert result.issues == ()
    assert result.frozen_response_sha256 is not None
    assert result.musicxml_path == output / "score.musicxml"
    assert result.musicxml_path.is_file()
    assert result.musicxml_sha256 is not None
    assert result.smf_path == output / "final.mid"
    assert result.smf_path.read_bytes()[:4] == b"MThd"
    assert result.smf_sha256 is not None
    assert result.diagnostics_path == output / "realization-diagnostics.json"
    assert result.realized_duration_ms == 180_000
    assert result.target_results
    assert all(item["status"] == "passed" for item in result.target_results)
    diagnostics = json.loads(result.diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["realized_duration_ms"] == 180_000
    assert diagnostics["quality"]["passes"] is True
    assert diagnostics["artifacts"]["musicxml"]["sha256"] == result.musicxml_sha256
    assert diagnostics["artifacts"]["smf"]["sha256"] == result.smf_sha256
    assert all(item["status"] == "passed" for item in diagnostics["targets"])


def test_realization_evaluates_a_typed_requirement_from_the_rendered_performance(
    tmp_path: Path,
) -> None:
    document = _approved_relational_with_requirement()

    result = realize_solo_piano_3m(
        document,
        _build_fixed_response(document),
        tmp_path / "output",
    )

    assert result.realized is True
    typed = next(
        item
        for item in result.target_results
        if item["target_id"] == "/script/requirements/return_more_aligned"
    )
    assert typed["status"] == "passed"
    assert typed["observed_relation"] == "more"
    assert typed["target"]["value"] == 0.0
    assert typed["reference"]["value"] == 45.0
    assert (
        "/script/requirements/return_more_aligned"
        not in _build_fixed_response(document)["target_assertions"]
    )


def test_realization_marks_an_unmet_typed_requirement_as_diagnostic_only(
    tmp_path: Path,
) -> None:
    document = _approved_relational_with_requirement()
    response = _build_fixed_response(document)
    performance = parse_performance_spec(response["performance_spec"]["source"])
    response["performance_spec"]["source"] = dump_performance_spec(
        replace(
            performance,
            node_performances=tuple(
                replace(item, coordination_profile="rolled") if item.node_id == "return" else item
                for item in performance.node_performances
            ),
        )
    )

    result = realize_solo_piano_3m(document, response, tmp_path / "output")

    assert result.realized is True
    assert result.outcome.artifact_disposition.value == "diagnostic_only"
    typed = next(
        item
        for item in result.target_results
        if item["target_id"] == "/script/requirements/return_more_aligned"
    )
    assert typed["status"] == "failed"
    assert typed["observed_relation"] == "equal"
    assert any(issue.path == "/script/requirements/return_more_aligned" for issue in result.issues)


def test_realization_preserves_a_missing_typed_observation_in_diagnostics(
    tmp_path: Path,
) -> None:
    document = _approved_relational_with_requirement()
    response = _build_fixed_response(document)
    score = parse_score_spec(response["score_spec"]["source"])
    response["score_spec"]["source"] = dump_score_spec(
        replace(
            score,
            materials=tuple(
                replace(
                    material,
                    notes=tuple(note for note in material.notes if note.voice == "upper"),
                )
                if material.material_id == "theme"
                else material
                for material in score.materials
            ),
        )
    )

    result = realize_solo_piano_3m(document, response, tmp_path / "output")

    assert result.realized is True
    assert result.outcome.artifact_disposition.value == "diagnostic_only"
    typed = next(
        item
        for item in result.target_results
        if item["target_id"] == "/script/requirements/return_more_aligned"
    )
    assert typed["status"] == "missing"
    assert typed["observed_relation"] == "missing"
    assert typed["target"]["reason"] == "no_score_simultaneous_multivoice_onset"
    assert typed["reference"]["reason"] == "no_score_simultaneous_multivoice_onset"


def test_realization_rejects_a_missing_natural_language_assertion(tmp_path: Path) -> None:
    document = _approved_relational_document()
    response = _fixed_response(document)
    assertions = response["target_assertions"]
    assert isinstance(assertions, dict)
    assertions.pop("/script/brief")

    result = realize_solo_piano_3m(document, response, tmp_path / "output")

    assert not result.realized
    assert result.issues[0].code is IssueCode.UNREPRESENTABLE
    assert result.issues[0].path == "/frozen_response/target_assertions"
    assert not (tmp_path / "output").exists()


def test_realization_rejects_an_assertion_outside_its_target_scope(tmp_path: Path) -> None:
    document = _approved_relational_document()
    response = _fixed_response(document)
    assertion = response["target_assertions"]["/script/brief"]
    assertion["tests"]["piece_plan"][0]["path"] = "/quality/passes"

    result = realize_solo_piano_3m(document, response, tmp_path / "output")

    assert not result.realized
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert not (tmp_path / "output").exists()


def test_realization_reports_a_scoped_assertion_that_is_not_met(tmp_path: Path) -> None:
    document = _approved_relational_document()
    response = _fixed_response(document)
    assertion = response["target_assertions"]["/script/brief"]
    assertion["tests"]["piece_plan"][0]["value"] = False

    result = realize_solo_piano_3m(document, response, tmp_path / "output")

    assert result.realized
    assert result.outcome.to_dict()["terminal_state"] == "completed"
    assert result.outcome.to_dict()["artifact_disposition"] == "diagnostic_only"
    assert not result.outcome.promotion_eligible
    assert result.issues[0].code is IssueCode.PROJECTION_TARGET_UNMET
    assert result.musicxml_path is not None and result.musicxml_path.is_file()
    assert result.smf_path is not None and result.smf_path.is_file()
    assert result.diagnostics_path is not None and result.diagnostics_path.is_file()


def test_realization_keeps_all_targets_when_a_natural_language_assertion_fails(
    tmp_path: Path,
) -> None:
    document = _approved_relational_with_requirement()
    response = _build_fixed_response(document)
    response["target_assertions"]["/script/brief"]["tests"]["piece_plan"][0]["value"] = False

    result = realize_solo_piano_3m(document, response, tmp_path / "output")

    projection = project_solo_piano_3m(
        document,
        plan_choice=PlanChoice(
            tonal_center=9,
            mode="minor",
            harmonic_focus_by_section={
                "statement": 9,
                "bridge": 9,
                "contrast": 4,
                "return": 9,
                "release": 9,
            },
            contrasts_with_by_section={"contrast": "statement"},
        ),
    )
    expected_ids = [target.target_id for target in projection.targets]
    actual_ids = [item["target_id"] for item in result.target_results]
    assert actual_ids == expected_ids
    assert len(actual_ids) == len(set(actual_ids))
    brief = next(item for item in result.target_results if item["target_id"] == "/script/brief")
    assert brief["status"] == "failed"
    typed = next(
        item
        for item in result.target_results
        if item["target_id"] == "/script/requirements/return_more_aligned"
    )
    assert typed["status"] == "passed"


def test_realization_prioritizes_later_assertion_format_errors_over_value_mismatches(
    tmp_path: Path,
) -> None:
    document = _approved_relational_with_requirement()
    response = _build_fixed_response(document)
    response["target_assertions"]["/script/brief"]["tests"]["piece_plan"][0]["value"] = False
    response["target_assertions"]["/script/sections/bridge/description"]["unexpected"] = True

    result = realize_solo_piano_3m(document, response, tmp_path / "output")

    assert result.realized is False
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert result.issues[0].path.endswith("/~1script~1sections~1bridge~1description")
    assert not (tmp_path / "output").exists()


def test_realization_preserves_quality_unfit_artifacts_for_diagnosis(
    tmp_path: Path, monkeypatch
) -> None:
    document = _approved_relational_document()
    response = _fixed_response(document)
    import scoim.realization as realization

    evaluate_quality = realization.evaluate_generic_pipeline_quality
    monkeypatch.setattr(
        "scoim.realization.evaluate_generic_pipeline_quality",
        lambda *args: {
            **evaluate_quality(*args),
            "passes": False,
            "reason": "controlled quality failure",
        },
    )

    result = realize_solo_piano_3m(document, response, tmp_path / "output")

    assert result.realized
    assert result.outcome.artifact_disposition.value == "diagnostic_only"
    assert result.issues[0].code is IssueCode.PROJECTION_TARGET_UNMET
    assert result.issues[0].path == "/quality"
    assert result.smf_path is not None and result.smf_path.is_file()
    diagnostics = json.loads(result.diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["quality"]["passes"] is False
    assert diagnostics["outcome"] == result.outcome.to_dict()


def test_realization_rejects_invalid_pipeline_dsl(tmp_path: Path) -> None:
    document = _approved_relational_document()
    response = _fixed_response(document)
    response["score_spec"]["source"] = "score_spec("

    result = realize_solo_piano_3m(document, response, tmp_path / "output")

    assert not result.realized
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert not (tmp_path / "output").exists()


def test_realization_rejects_invalid_pipeline_dsl_response_fields(tmp_path: Path) -> None:
    document = _approved_relational_document()
    response = _fixed_response(document)
    response["score_spec"] = {"source": "score_spec(...)"}

    result = realize_solo_piano_3m(document, response, tmp_path / "output")

    assert not result.realized
    assert result.frozen_response_sha256 is not None
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert result.issues[0].path == "/frozen_response/score_spec"


def test_realization_rejects_a_referenced_empty_score_material_before_rendering(
    tmp_path: Path,
) -> None:
    document = _approved_relational_document()
    response = _fixed_response(document)
    score = parse_score_spec(response["score_spec"]["source"])
    response["score_spec"]["source"] = dump_score_spec(
        replace(
            score,
            materials=tuple(
                replace(material, notes=()) if material.material_id == "theme" else material
                for material in score.materials
            ),
        )
    )

    result = realize_solo_piano_3m(document, response, tmp_path / "output")

    assert result.realized is False
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert "referenced score material must contain at least one note" in result.issues[0].message
    assert not (tmp_path / "output").exists()


def test_realization_rejects_invalid_target_assertion_fields(tmp_path: Path) -> None:
    document = _approved_relational_document()
    response = _fixed_response(document)
    response["target_assertions"]["/script/brief"] = {"interpretation": "説明だけ"}

    result = realize_solo_piano_3m(document, response, tmp_path / "output")

    assert not result.realized
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert not (tmp_path / "output").exists()


def test_realization_rejects_score_lengths_that_break_script_ratios(tmp_path: Path) -> None:
    document = _approved_relational_document()
    response = _fixed_response(document)
    score_response = response["score_spec"]
    assert isinstance(score_response, dict)
    score = parse_score_spec(score_response["source"])
    connector = score.materials[1]
    changed_connector = replace(
        connector,
        length_units=3,
        notes=tuple(replace(note, duration_units=3) for note in connector.notes),
        harmonies=tuple(replace(harmony, duration_units=3) for harmony in connector.harmonies),
    )
    score_response["source"] = dump_score_spec(
        replace(
            score,
            materials=(score.materials[0], changed_connector, *score.materials[2:]),
        )
    )

    result = realize_solo_piano_3m(document, response, tmp_path / "output")

    assert not result.realized
    assert result.issues[0].code is IssueCode.PROJECTION_TARGET_UNMET
    assert result.issues[0].path == "/script/sections"
    assert not (tmp_path / "output").exists()


def test_realization_does_not_overwrite_an_existing_output_directory(tmp_path: Path) -> None:
    document = _approved_relational_document()
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing\n", encoding="utf-8")

    result = realize_solo_piano_3m(document, _fixed_response(document), output)

    assert not result.realized
    assert result.issues[0].code is IssueCode.STORAGE_CONFLICT
    assert marker.read_text(encoding="utf-8") == "existing\n"
    assert {item.name for item in output.iterdir()} == {"keep.txt"}


def test_realization_reports_an_output_storage_error(tmp_path: Path) -> None:
    document = _approved_relational_document()
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("occupied\n", encoding="utf-8")

    result = realize_solo_piano_3m(
        document,
        _fixed_response(document),
        parent_file / "output",
    )

    assert not result.realized
    assert result.issues[0].code is IssueCode.STORAGE_ERROR
    assert parent_file.read_text(encoding="utf-8") == "occupied\n"


def test_realization_replays_the_same_frozen_response_byte_for_byte(tmp_path: Path) -> None:
    document = _approved_relational_document()
    response = _fixed_response(document)
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    first = realize_solo_piano_3m(document, response, first_output)
    second = realize_solo_piano_3m(document, response, second_output)

    assert first.realized and second.realized
    assert first.frozen_response_sha256 == second.frozen_response_sha256
    for filename in ("score.musicxml", "final.mid", "realization-diagnostics.json"):
        assert (first_output / filename).read_bytes() == (second_output / filename).read_bytes()
