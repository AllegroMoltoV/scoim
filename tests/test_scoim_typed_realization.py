import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from llm_musical_composer.performance_pipeline import (
    NodePerformance,
    PerformanceSpec,
    PiecePlan,
    PlanNode,
    ScoreHarmony,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
    validate_pipeline,
)
from llm_musical_composer.pipeline_dsl import (
    dump_performance_spec,
    dump_score_spec,
    parse_performance_spec,
    parse_score_spec,
)
from scoim.realization import realize_solo_piano_3m
from scoim.trial_bundle import create_trial_bundle, replay_trial_bundle
from scoim.typed_realization import (
    TypedRealizationError,
    build_performance_spec_from_typed_response,
    build_score_spec_from_typed_response,
    build_typed_ir,
    build_typed_response_schema,
    ensure_all_script_materials_are_placed,
    ensure_distinct_event_boundaries,
    ensure_plan_resource_limit,
    normalize_typed_model_response,
    validate_typed_model_response,
)
from scoim.validation import IssueCode

_FIXED_ABA = Path(__file__).parent / "fixtures" / "scoim" / "fixed-aba"
_NESTED_ABA = Path(__file__).parent / "fixtures" / "scoim" / "nested-aba"


def _approved_script() -> dict[str, object]:
    return json.loads((_FIXED_ABA / "approved-script.json").read_text(encoding="utf-8"))


def _reverse_object_keys(value: object) -> object:
    if isinstance(value, dict):
        return {key: _reverse_object_keys(value[key]) for key in reversed(tuple(value.keys()))}
    if isinstance(value, list):
        return [_reverse_object_keys(item) for item in value]
    return value


def _typed_model_response(
    fixture_dir: Path,
    response_name: str,
    section_ids: tuple[str, ...],
    contrast_indexes: dict[str, int | None],
) -> dict[str, object]:
    legacy = json.loads((fixture_dir / response_name).read_text(encoding="utf-8"))
    legacy_score = parse_score_spec(legacy["score_spec"]["source"])
    legacy_performance = parse_performance_spec(legacy["performance_spec"]["source"])
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
    return {
        "plan_choice": {
            "tonal_center": plan_choice["tonal_center"],
            "mode": plan_choice["mode"],
            "harmonic_focus_by_section": {
                section_id: plan_choice["harmonic_focus_by_section"].get(section_id)
                for section_id in section_ids
            },
            "contrasts_with_by_section": contrast_indexes,
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


def _typed_model_response_from_legacy() -> dict[str, object]:
    return _typed_model_response(
        _FIXED_ABA,
        "frozen-response.json",
        ("whole", "statement", "bridge", "contrast", "return", "release"),
        {"contrast": 0},
    )


def _one_material_plan() -> PiecePlan:
    return PiecePlan(
        plan_id="typed-example",
        title="型付き転送例",
        tonal_center=0,
        mode="major",
        root_node_id="whole",
        ending_intent="tonic",
        nodes=(
            PlanNode("whole", None, 0, "whole"),
            PlanNode(
                "statement",
                "whole",
                0,
                "statement",
                duration_weight=2,
                score_material_id="theme",
            ),
        ),
    )


def test_typed_material_values_build_a_round_trippable_score_spec() -> None:
    score = build_score_spec_from_typed_response(
        _one_material_plan(),
        {
            "theme": {
                "foreground_voice": "upper",
                "notes": [
                    {
                        "at_units": 0,
                        "duration_units": 6,
                        "pitch": 48,
                        "voice": "lower",
                        "tie": None,
                        "articulations": ["tenuto"],
                    },
                    {
                        "at_units": 0,
                        "duration_units": 3,
                        "pitch": 60,
                        "voice": "upper",
                        "tie": None,
                        "articulations": ["normal"],
                    },
                ],
                "harmonies": [
                    {
                        "at_units": 0,
                        "duration_units": 12,
                        "root_pitch_class": 0,
                        "quality": "major",
                    }
                ],
                "directions": [],
            }
        },
        material_derivations={},
    )

    assert score == ScoreSpec(
        score_id="typed-example-score",
        divisions=12,
        materials=(
            ScoreMaterial(
                material_id="theme",
                length_units=12,
                notes=(
                    ScoreNote(
                        event_id="theme-note-000",
                        at_units=0,
                        duration_units=6,
                        pitch=48,
                        voice="lower",
                        articulations=("tenuto",),
                    ),
                    ScoreNote(
                        event_id="theme-note-001",
                        at_units=0,
                        duration_units=3,
                        pitch=60,
                        voice="upper",
                        articulations=("normal",),
                    ),
                ),
                harmonies=(
                    ScoreHarmony(
                        harmony_id="theme-harmony-000",
                        at_units=0,
                        duration_units=12,
                        root_pitch_class=0,
                        quality="major",
                    ),
                ),
                foreground_voice="upper",
            ),
        ),
    )
    assert parse_score_spec(dump_score_spec(score)) == score


def test_typed_node_values_build_a_round_trippable_performance_spec() -> None:
    performance = build_performance_spec_from_typed_response(
        _one_material_plan(),
        {
            "whole": {
                "timing_profile": None,
                "timing_amount": None,
                "dynamics_profile": None,
                "articulation_profile": None,
                "coordination_profile": None,
                "pedal_profile": None,
            },
            "statement": {
                "timing_profile": "savor",
                "timing_amount": "subtle",
                "dynamics_profile": "shape",
                "articulation_profile": "legato",
                "coordination_profile": "aligned",
                "pedal_profile": "none",
            },
        },
    )

    assert performance == PerformanceSpec(
        performance_id="typed-example-performance",
        target_duration_ms=180000,
        default_velocity=64,
        timing_budget_id="narrative-v2",
        node_performances=(
            NodePerformance(node_id="whole"),
            NodePerformance(
                node_id="statement",
                timing_profile="savor",
                timing_amount="subtle",
                dynamics_profile="shape",
                articulation_profile="legato",
                coordination_profile="aligned",
                pedal_profile="none",
            ),
        ),
        key_release_percent=100,
        velocity_policy_id="foreground-accompaniment-harmony-shape-v1",
    )
    assert parse_performance_spec(dump_performance_spec(performance)) == performance


def test_unplaced_material_is_unrepresentable_instead_of_receiving_a_guessed_length() -> None:
    with pytest.raises(TypedRealizationError) as caught:
        ensure_all_script_materials_are_placed(_one_material_plan(), {"theme", "unused"})

    assert caught.value.issue.code is IssueCode.UNREPRESENTABLE
    assert caught.value.issue.path == "/script/materials/unused"


def test_typed_score_rejects_a_missing_fixed_material_key() -> None:
    with pytest.raises(TypedRealizationError) as caught:
        build_score_spec_from_typed_response(
            _one_material_plan(),
            {},
            material_derivations={},
        )

    assert caught.value.issue.code is IssueCode.MODEL_OUTPUT_INVALID
    assert caught.value.issue.path == "/runner/response/score_materials/theme"


def test_plan_resource_limit_is_checked_before_allocating_the_time_map() -> None:
    plan = PiecePlan(
        plan_id="too-large",
        title="大きすぎる例",
        tonal_center=0,
        mode="major",
        root_node_id="whole",
        ending_intent="tonic",
        nodes=(
            PlanNode("whole", None, 0, "whole"),
            PlanNode(
                "statement",
                "whole",
                0,
                "statement",
                duration_weight=30_001,
                score_material_id="theme",
            ),
        ),
    )

    with pytest.raises(TypedRealizationError) as caught:
        ensure_plan_resource_limit(plan)

    assert caught.value.issue.code is IssueCode.UNREPRESENTABLE
    assert caught.value.issue.path == "/script/sections"


def test_distinct_score_boundaries_must_not_collapse_to_the_same_integer_ms() -> None:
    plan = _one_material_plan()
    score = ScoreSpec(
        score_id="typed-example-score",
        divisions=12,
        materials=(
            ScoreMaterial(
                material_id="theme",
                length_units=12,
                notes=(
                    ScoreNote("theme-note-000", 0, 1, 60, "upper"),
                    ScoreNote("theme-note-001", 1, 1, 62, "upper"),
                ),
            ),
        ),
    )
    performance = PerformanceSpec(
        performance_id="typed-example-performance",
        target_duration_ms=1,
        default_velocity=64,
        timing_budget_id="narrative-v2",
        node_performances=(),
    )

    with pytest.raises(TypedRealizationError) as caught:
        ensure_distinct_event_boundaries(plan, score, performance)

    assert caught.value.issue.code is IssueCode.UNREPRESENTABLE
    assert caught.value.issue.path == "/runner/response/score_materials/theme/notes/0"


def test_approved_script_builds_the_same_typed_response_schema_bytes() -> None:
    first = build_typed_response_schema(_approved_script())
    second = build_typed_response_schema(_approved_script())

    first_bytes = json.dumps(
        first, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    second_bytes = json.dumps(
        second, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    assert first_bytes == second_bytes
    assert set(first["properties"]) == {
        "plan_choice",
        "score_materials",
        "node_performances",
    }
    assert "target_assertions" not in first["properties"]
    Draft202012Validator.check_schema(first)
    assert set(first["properties"]["score_materials"]["properties"]) == {
        "cadence",
        "connector",
        "contrast",
        "theme",
    }
    assert set(first["properties"]["node_performances"]["properties"]) == {
        "whole",
        "statement",
        "bridge",
        "contrast",
        "return",
        "release",
    }
    assert set(first["$defs"]) == {
        "direction",
        "harmony",
        "material",
        "note",
        "performance",
    }
    assert all(
        value == {"$ref": "#/$defs/material"}
        for value in first["properties"]["score_materials"]["properties"].values()
    )
    assert all(
        value == {"$ref": "#/$defs/performance"}
        for value in first["properties"]["node_performances"]["properties"].values()
    )

    def assert_structured_subset(node: object) -> None:
        assert isinstance(node, dict)
        assert "type" in node or "anyOf" in node or "$ref" in node
        if node.get("type") == "object":
            properties = node.get("properties")
            assert isinstance(properties, dict)
            assert node.get("additionalProperties") is False
            assert set(node.get("required", [])) == set(properties)
        for child in node.get("properties", {}).values():
            assert_structured_subset(child)
        if "items" in node:
            assert_structured_subset(node["items"])
        for child in node.get("anyOf", []):
            assert_structured_subset(child)
        for child in node.get("$defs", {}).values():
            assert_structured_subset(child)

    assert_structured_subset(first)


def test_typed_response_schema_ignores_json_object_key_order() -> None:
    original = build_typed_response_schema(_approved_script())
    reordered = build_typed_response_schema(_reverse_object_keys(_approved_script()))

    assert json.dumps(original, sort_keys=True) == json.dumps(reordered, sort_keys=True)


def test_typed_model_response_resolves_choice_indexes_without_model_authored_ids() -> None:
    document = _approved_script()
    response = {
        "plan_choice": {
            "tonal_center": 9,
            "mode": "minor",
            "harmonic_focus_by_section": {
                "whole": None,
                "statement": 9,
                "bridge": 9,
                "contrast": 4,
                "return": 9,
                "release": 9,
            },
            "contrasts_with_by_section": {"contrast": 0},
        },
        "score_materials": {
            material_id: {
                "foreground_voice": None,
                "notes": [],
                "harmonies": [],
                "directions": [],
            }
            for material_id in ("theme", "connector", "contrast", "cadence")
        },
        "node_performances": {
            section_id: {
                "timing_profile": None,
                "timing_amount": None,
                "dynamics_profile": None,
                "articulation_profile": None,
                "coordination_profile": None,
                "pedal_profile": None,
            }
            for section_id in (
                "release",
                "return",
                "contrast",
                "bridge",
                "statement",
                "whole",
            )
        },
    }

    normalized = normalize_typed_model_response(document, response)

    assert normalized["schema_version"] == 2
    assert normalized["profile"] == "solo_piano_3m_v1"
    assert normalized["plan_choice"] == {
        "tonal_center": 9,
        "mode": "minor",
        "harmonic_focus_by_section": {
            "whole": None,
            "statement": 9,
            "bridge": 9,
            "contrast": 4,
            "return": 9,
            "release": 9,
        },
        "contrasts_with_by_section": {"contrast": 0},
    }
    assert list(normalized["score_materials"]) == [
        "cadence",
        "connector",
        "contrast",
        "theme",
    ]
    assert list(normalized["node_performances"]) == [
        "whole",
        "statement",
        "bridge",
        "contrast",
        "return",
        "release",
    ]


def test_normalized_schema_2_builds_all_existing_ir_stages() -> None:
    document = _approved_script()
    frozen_response = normalize_typed_model_response(document, _typed_model_response_from_legacy())

    typed_ir = build_typed_ir(document, frozen_response)

    validate_pipeline(typed_ir.plan, typed_ir.score, typed_ir.performance)
    assert parse_score_spec(dump_score_spec(typed_ir.score)) == typed_ir.score
    assert (
        parse_performance_spec(dump_performance_spec(typed_ir.performance)) == typed_ir.performance
    )
    assert typed_ir.score.divisions == 12
    assert typed_ir.performance.target_duration_ms == 180000


def test_typed_model_response_rejects_unknown_fixed_keys_with_a_pointer() -> None:
    document = _approved_script()
    response = _typed_model_response_from_legacy()
    response["score_materials"]["unknown"] = response["score_materials"]["theme"]

    issues = validate_typed_model_response(document, response)

    assert len(issues) == 1
    assert issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert issues[0].path == "/runner/response/score_materials/unknown"


def test_typed_score_rejects_a_note_outside_its_fixed_material_length() -> None:
    document = _approved_script()
    response = _typed_model_response_from_legacy()
    response["score_materials"]["theme"]["notes"][0]["at_units"] = 24
    frozen_response = normalize_typed_model_response(document, response)

    with pytest.raises(TypedRealizationError) as caught:
        build_typed_ir(document, frozen_response)

    assert caught.value.issue.code is IssueCode.MODEL_OUTPUT_INVALID
    assert caught.value.issue.path == "/runner/response/score_materials/theme/notes/0"


def test_schema_2_returns_a_typed_failure_for_incomplete_harmony_coverage(
    tmp_path: Path,
) -> None:
    document = _approved_script()
    response = _typed_model_response_from_legacy()
    response["score_materials"]["theme"]["harmonies"][0]["duration_units"] = 3
    frozen_response = normalize_typed_model_response(document, response)

    result = realize_solo_piano_3m(document, frozen_response, tmp_path / "artifacts")

    assert result.realized is False
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert result.issues[0].path == "/runner/response/score_materials/theme/harmonies/0"


def test_typed_score_rejects_overlapping_same_pitch_with_the_second_note_path() -> None:
    document = _approved_script()
    response = _typed_model_response_from_legacy()
    first_note = response["score_materials"]["theme"]["notes"][0]
    second_note = response["score_materials"]["theme"]["notes"][1]
    second_note["at_units"] = first_note["at_units"]
    second_note["pitch"] = first_note["pitch"]
    second_note["voice"] = first_note["voice"]
    frozen_response = normalize_typed_model_response(document, response)

    with pytest.raises(TypedRealizationError) as caught:
        build_typed_ir(document, frozen_response)

    assert caught.value.issue.code is IssueCode.MODEL_OUTPUT_INVALID
    assert caught.value.issue.path == "/runner/response/score_materials/theme/notes/1"


def test_typed_score_requires_foreground_voice_with_harmony() -> None:
    document = _approved_script()
    response = _typed_model_response_from_legacy()
    response["score_materials"]["theme"]["foreground_voice"] = None
    frozen_response = normalize_typed_model_response(document, response)

    with pytest.raises(TypedRealizationError) as caught:
        build_typed_ir(document, frozen_response)

    assert caught.value.issue.code is IssueCode.MODEL_OUTPUT_INVALID
    assert caught.value.issue.path == ("/runner/response/score_materials/theme/foreground_voice")


def test_typed_performance_rejects_timing_amount_without_a_profile() -> None:
    document = _approved_script()
    response = _typed_model_response_from_legacy()
    response["node_performances"]["whole"]["timing_amount"] = "subtle"
    frozen_response = normalize_typed_model_response(document, response)

    with pytest.raises(TypedRealizationError) as caught:
        build_typed_ir(document, frozen_response)

    assert caught.value.issue.code is IssueCode.MODEL_OUTPUT_INVALID
    assert caught.value.issue.path == ("/runner/response/node_performances/whole/timing_amount")


def test_typed_score_rejects_a_direction_value_for_the_wrong_kind() -> None:
    document = _approved_script()
    response = _typed_model_response_from_legacy()
    response["score_materials"]["theme"]["directions"] = [
        {"at_units": 0, "kind": "dynamic", "value": "light"}
    ]
    frozen_response = normalize_typed_model_response(document, response)

    with pytest.raises(TypedRealizationError) as caught:
        build_typed_ir(document, frozen_response)

    assert caught.value.issue.code is IssueCode.MODEL_OUTPUT_INVALID
    assert caught.value.issue.path == ("/runner/response/score_materials/theme/directions/0/value")


def test_typed_score_rejects_duplicate_note_articulations() -> None:
    document = _approved_script()
    response = _typed_model_response_from_legacy()
    response["score_materials"]["theme"]["notes"][0]["articulations"] = [
        "tenuto",
        "tenuto",
    ]
    frozen_response = normalize_typed_model_response(document, response)

    with pytest.raises(TypedRealizationError) as caught:
        build_typed_ir(document, frozen_response)

    assert caught.value.issue.code is IssueCode.MODEL_OUTPUT_INVALID
    assert caught.value.issue.path == (
        "/runner/response/score_materials/theme/notes/0/articulations/1"
    )


def test_schema_2_realizes_musicxml_and_smf_with_natural_language_unverified(
    tmp_path: Path,
) -> None:
    document = _approved_script()
    frozen_response = normalize_typed_model_response(document, _typed_model_response_from_legacy())

    result = realize_solo_piano_3m(document, frozen_response, tmp_path / "artifacts")

    assert result.realized is True, result.issues
    assert result.outcome.artifact_disposition.value == "candidate"
    assert result.musicxml_path is not None and result.musicxml_path.is_file()
    assert result.smf_path is not None and result.smf_path.is_file()
    statuses = {item["status"] for item in result.target_results}
    assert "unverified" in statuses


def test_schema_2_trial_bundle_uses_realizer_4_and_replays_offline(tmp_path: Path) -> None:
    document = _approved_script()
    frozen_response = normalize_typed_model_response(document, _typed_model_response_from_legacy())
    bundle = tmp_path / "bundle"

    created = create_trial_bundle(
        document,
        frozen_response,
        bundle,
        trial_id="typed-schema-2",
    )
    replayed = replay_trial_bundle(bundle, tmp_path / "replay")

    assert created.created is True, created.issues
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["realizer"]["realizer_version"] == 4
    assert {
        name: manifest["files"][name]["sha256"] for name in ("musicxml", "smf", "diagnostics")
    } == {
        "musicxml": "587c32f3e4f1acf20619482e0c62e6f721cd20cb9685812ed302e263765e9d86",
        "smf": "991d742dad782968051b555690d66b445d90a4c0577f7dcc88bb197799b1e89b",
        "diagnostics": "31f7f2cc89c68835b267512efb6093b6bd339719290593da55eee91499ef5c6e",
    }
    assert replayed.replayed is True, replayed.issues
    assert replayed.artifact_sha256 == {
        "musicxml": manifest["files"]["musicxml"]["sha256"],
        "smf": manifest["files"]["smf"]["sha256"],
        "diagnostics": manifest["files"]["diagnostics"]["sha256"],
    }


def test_nested_schema_2_trial_bundle_replays_offline(tmp_path: Path) -> None:
    document = json.loads(
        (_NESTED_ABA / "approved-script-revision-2.json").read_text(encoding="utf-8")
    )
    response = _typed_model_response(
        _NESTED_ABA,
        "frozen-response-revision-2.json",
        (
            "whole",
            "a1",
            "a1_open",
            "a1_answer",
            "bridge_ab",
            "b",
            "b1",
            "b2",
            "bridge_ba",
            "a2",
            "a2_open",
            "a2_answer",
            "release",
        ),
        {"b": 0, "b1": None},
    )
    frozen_response = normalize_typed_model_response(document, response)
    bundle = tmp_path / "nested-bundle"

    created = create_trial_bundle(
        document,
        frozen_response,
        bundle,
        trial_id="nested-typed-schema-2",
    )
    replayed = replay_trial_bundle(bundle, tmp_path / "nested-replay")

    assert created.created is True, created.issues
    assert replayed.replayed is True, replayed.issues
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert replayed.artifact_sha256 == {
        "musicxml": manifest["files"]["musicxml"]["sha256"],
        "smf": manifest["files"]["smf"]["sha256"],
        "diagnostics": manifest["files"]["diagnostics"]["sha256"],
    }
