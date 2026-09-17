from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from itertools import accumulate
from types import SimpleNamespace

import pytest
from mido import Message, MetaMessage, MidiFile, MidiTrack

from llm_musical_composer.performance_pipeline import (
    NodePerformance,
    PerformanceSpec,
    PiecePlan,
    PlanNode,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
    render_performance,
    render_performance_smf,
)
from llm_musical_composer.pipeline_dsl import (
    dump_performance_spec,
    dump_piece_plan,
    dump_score_spec,
)
from llm_musical_composer.reference_decomposition import (
    ObservedNote,
    build_observed_performance,
    load_observed_smf,
)
from llm_musical_composer.reference_timing_hypothesis import (
    AlignedExpectedObservedAttack,
    ExpectedScoreAttack,
    KnownScoreNoteAlignment,
    align_known_score_note_events,
)
from llm_musical_composer.score_timing_hypothesis import (
    GroupingAttack,
    HypothesisAttackGroupRef,
    ScoreTimingHypothesisV0,
    build_grouping_profiles,
)
from llm_musical_composer.score_timing_upper_evidence import (
    _verified_oracle_observation_dependency,
    build_candidate_upper_evidence,
    classify_candidate_evidence,
    enumerate_pitch_rhythm_recurrences,
    run_upper_evidence_observation,
    run_upper_evidence_oracle_annotation,
    summarize_pitch_rhythm_recurrence,
    verify_observation_artifacts,
)
from llm_musical_composer.texture_edge_diagnostics import (
    build_known_voice_diagnostics,
    build_texture_candidate_joins,
    build_texture_edge_diagnostics,
    matched_pair_weighted_null_rate,
    measure_event_id_references,
)


def _candidate(
    positions: tuple[int, ...],
    *,
    profile_id: str = "profile-a",
    ledger_sha256: str = "a" * 64,
) -> ScoreTimingHypothesisV0:
    return ScoreTimingHypothesisV0(
        candidate_id=f"candidate-{positions[-1]}",
        status="candidate",
        search_status="complete",
        source_ledger_sha256=ledger_sha256,
        grouping_profile_id=profile_id,
        score_grid_id="grid",
        requested_segment_count=1,
        actual_segment_count=1,
        attack_group_refs=tuple(
            HypothesisAttackGroupRef(group_index=index, score_position=position)
            for index, position in enumerate(positions)
        ),
        shared_time_map=(),
        local_coordination=(),
        equivalence=(),
        assumptions=(),
        evidence_event_ids=(),
        metrics={},
    )


def _note(event_id: str, pitch: int) -> ObservedNote:
    return ObservedNote(
        note_on_event_id=event_id,
        note_off_event_id=f"{event_id}-off",
        track=0,
        channel=0,
        pitch=pitch,
        velocity=64,
        note_off_velocity=0,
        onset_tick=0,
        offset_tick=1,
        onset_us=0,
        offset_us=1,
        matching_status="matched",
    )


def test_transposed_recurrence_keeps_pitch_shape_and_rhythm_ratio() -> None:
    pitch_groups = ((60,), (62,), (64,), (67,), (69,), (71,))

    summary = summarize_pitch_rhythm_recurrence(
        pitch_groups=pitch_groups,
        score_positions=(0, 1, 2, 4, 5, 6),
        window_sizes=(3,),
    )

    assert summary["highest"]["3"] == {
        "repeated_pitch_shape_count": 1,
        "repetition_pair_count": 1,
        "same_rhythm_pair_count": 1,
        "different_rhythm_pair_count": 0,
    }
    assert summary["lowest"]["3"] == summary["highest"]["3"]
    assert summary["pitch_set"]["3"] == summary["highest"]["3"]


def test_recurrence_enumerator_exposes_canonical_occurrences() -> None:
    records = enumerate_pitch_rhythm_recurrences(
        pitch_groups=((60,), (62,), (64,), (67,), (69,), (71,)),
        score_positions=(0, 1, 2, 4, 5, 6),
        window_sizes=(3,),
    )

    highest = next(record for record in records if record["view"] == "highest")
    assert highest["window_size"] == 3
    assert highest["occurrences"] == (
        {"start_group_index": 0, "rhythm_ratio": (1, 1)},
        {"start_group_index": 3, "rhythm_ratio": (1, 1)},
    )


def test_same_pitch_shape_can_have_different_rhythm_ratio() -> None:
    summary = summarize_pitch_rhythm_recurrence(
        pitch_groups=((60,), (62,), (64,), (67,), (69,), (71,)),
        score_positions=(0, 1, 2, 4, 5, 7),
        window_sizes=(3,),
    )

    assert summary["highest"]["3"]["same_rhythm_pair_count"] == 0
    assert summary["highest"]["3"]["different_rhythm_pair_count"] == 1


def test_candidate_is_joined_to_group_profile_by_index() -> None:
    groups = tuple(GroupingAttack(index * 100, (f"e{index}",), (0,)) for index in range(6))
    notes = {
        f"e{index}": _note(f"e{index}", pitch)
        for index, pitch in enumerate((60, 62, 64, 67, 69, 71))
    }

    evidence = build_candidate_upper_evidence(
        _candidate((0, 1, 2, 4, 5, 6)),
        grouping_profiles={"profile-a": groups},
        notes_by_id=notes,
        ledger_sha256="a" * 64,
        window_sizes=(3,),
    )

    assert evidence["status"] == "assessed"
    assert evidence["summary"]["highest"]["3"]["same_rhythm_pair_count"] == 1


def test_candidate_join_rejects_profile_index_event_and_ledger_mismatch() -> None:
    groups = (GroupingAttack(0, ("e0",), (0,)), GroupingAttack(100, ("missing",), (0,)))
    notes = {"e0": _note("e0", 60)}

    missing_profile = build_candidate_upper_evidence(
        _candidate((0, 1), profile_id="absent"),
        grouping_profiles={"profile-a": groups},
        notes_by_id=notes,
        ledger_sha256="a" * 64,
    )
    bad_index = build_candidate_upper_evidence(
        replace(
            _candidate((0, 1)),
            attack_group_refs=(
                HypothesisAttackGroupRef(0, 0),
                HypothesisAttackGroupRef(2, 1),
            ),
        ),
        grouping_profiles={"profile-a": groups},
        notes_by_id=notes,
        ledger_sha256="a" * 64,
    )
    missing_event = build_candidate_upper_evidence(
        _candidate((0, 1)),
        grouping_profiles={"profile-a": groups},
        notes_by_id=notes,
        ledger_sha256="a" * 64,
    )
    bad_ledger = build_candidate_upper_evidence(
        _candidate((0, 1)),
        grouping_profiles={"profile-a": groups},
        notes_by_id=notes,
        ledger_sha256="b" * 64,
    )

    assert missing_profile["status"] == "unable_to_investigate"
    assert missing_profile["reason"] == "grouping_profile_missing"
    assert bad_index["reason"] == "group_index_mismatch"
    assert missing_event["reason"] == "source_event_missing"
    assert bad_ledger["reason"] == "ledger_sha256_mismatch"


def test_classification_separates_timing_grouping_and_no_recurrence() -> None:
    timing = classify_candidate_evidence(
        [
            {
                "status": "assessed",
                "grouping_profile_id": "g",
                "fingerprint": "a",
                "repetition_pair_count": 1,
            },
            {
                "status": "assessed",
                "grouping_profile_id": "g",
                "fingerprint": "b",
                "repetition_pair_count": 1,
            },
        ]
    )
    grouping = classify_candidate_evidence(
        [
            {
                "status": "assessed",
                "grouping_profile_id": "g1",
                "fingerprint": "a",
                "repetition_pair_count": 1,
            },
            {
                "status": "assessed",
                "grouping_profile_id": "g2",
                "fingerprint": "b",
                "repetition_pair_count": 1,
            },
        ]
    )
    absent = classify_candidate_evidence(
        [
            {
                "status": "assessed",
                "grouping_profile_id": "g",
                "fingerprint": "a",
                "repetition_pair_count": 0,
            }
        ]
    )

    assert timing == "timing_sensitive"
    assert grouping == "grouping_only"
    assert absent == "unable_to_investigate"


def test_oracle_stage_rejects_modified_observation_artifact(tmp_path) -> None:
    input_bytes = b'{"schema_version":1,"sources":[]}'
    evidence_bytes = b'{"case_id":"source"}\n'
    (tmp_path / "observation-input.json").write_bytes(input_bytes)
    (tmp_path / "observed-upper-evidence.jsonl").write_bytes(evidence_bytes)
    (tmp_path / "observation-manifest.json").write_text(
        json.dumps(
            {
                "outputs": {
                    "observation-input.json": hashlib.sha256(input_bytes).hexdigest(),
                    "observed-upper-evidence.jsonl": hashlib.sha256(evidence_bytes).hexdigest(),
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "observed-upper-evidence.jsonl").write_text(
        '{"case_id":"changed"}\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        verify_observation_artifacts(tmp_path)


def test_texture_diagnostic_aliases_identical_grouping_content_without_event_ids() -> None:
    shared = (
        GroupingAttack(0, ("note-a", "note-b"), (0, 10_000)),
        GroupingAttack(100_000, ("note-c",), (0,)),
    )
    distinct = (
        GroupingAttack(0, ("note-a",), (0,)),
        GroupingAttack(10_000, ("note-b",), (0,)),
        GroupingAttack(100_000, ("note-c",), (0,)),
    )
    notes = {
        "note-a": _note("note-a", 60),
        "note-b": _note("note-b", 64),
        "note-c": _note("note-c", 62),
    }

    diagnostic = build_texture_edge_diagnostics(
        grouping_profiles={"profile-a": shared, "profile-b": shared, "profile-c": distinct},
        notes_by_id=notes,
        ledger_sha256="a" * 64,
    )

    assert diagnostic["surface_profile_count"] == 3
    assert diagnostic["semantic_profile_count"] == 2
    assert diagnostic["alias_map"]["profile-a"] == diagnostic["alias_map"]["profile-b"]
    assert diagnostic["alias_map"]["profile-a"] != diagnostic["alias_map"]["profile-c"]
    assert all(profile["event_id_occurrence_count"] == 0 for profile in diagnostic["profiles"])
    serialized = json.dumps(diagnostic, sort_keys=True)
    assert "note-a" not in serialized
    assert "note-b" not in serialized
    assert "note-c" not in serialized


def test_texture_diagnostic_compresses_equal_pitch_ties_into_relation_blocks() -> None:
    groups = (
        GroupingAttack(0, ("left-a", "left-b", "left-c"), (0, 0, 0)),
        GroupingAttack(100_000, ("right-a", "right-b", "right-c"), (0, 0, 0)),
    )
    notes = {
        "left-a": _note("left-a", 60),
        "left-b": _note("left-b", 60),
        "left-c": _note("left-c", 64),
        "right-a": _note("right-a", 62),
        "right-b": _note("right-b", 62),
        "right-c": _note("right-c", 65),
    }

    diagnostic = build_texture_edge_diagnostics(
        grouping_profiles={"profile": groups},
        notes_by_id=notes,
        ledger_sha256="a" * 64,
    )
    profile = diagnostic["profiles"][0]

    assert profile["vertical_groups"][0]["rank_blocks"] == [[0, 1], [2]]
    assert len(profile["horizontal_relations"]) == 3
    assert (
        sum(relation["relation"] == "mutual" for relation in profile["horizontal_relations"]) == 2
    )
    tied = next(
        relation
        for relation in profile["horizontal_relations"]
        if relation["source_rank_index"] == 0 and relation["target_rank_index"] == 0
    )
    assert tied["relation"] == "mutual"
    assert "source_members" not in tied
    assert "target_members" not in tied
    assert profile["max_virtual_edges_per_relation_block"] == 4


def test_texture_diagnostic_counts_a_two_way_branch_as_ambiguous_paths() -> None:
    groups = (
        GroupingAttack(0, ("source",), (0,)),
        GroupingAttack(100_000, ("lower", "upper"), (0, 0)),
    )
    notes = {
        "source": _note("source", 60),
        "lower": _note("lower", 59),
        "upper": _note("upper", 61),
    }

    diagnostic = build_texture_edge_diagnostics(
        grouping_profiles={"profile": groups},
        notes_by_id=notes,
        ledger_sha256="a" * 64,
    )
    profile = diagnostic["profiles"][0]

    assert profile["max_ambiguous_component_member_count"] == 3
    assert profile["path_count_log10_upper_bound"] == pytest.approx(0.30103)


def test_texture_candidate_join_changes_fingerprint_without_copying_base_graph() -> None:
    groups = tuple(GroupingAttack(index * 100_000, (f"event-{index}",), (0,)) for index in range(4))
    notes = {
        f"event-{index}": _note(f"event-{index}", pitch)
        for index, pitch in enumerate((60, 62, 64, 65))
    }
    diagnostic = build_texture_edge_diagnostics(
        grouping_profiles={"profile-a": groups, "profile-alias": groups},
        notes_by_id=notes,
        ledger_sha256="a" * 64,
    )

    joins = build_texture_candidate_joins(
        (
            _candidate((0, 1, 2, 3), profile_id="profile-a"),
            _candidate((0, 1, 3, 4), profile_id="profile-alias"),
        ),
        texture_diagnostic=diagnostic,
        window_sizes=(3, 4),
    )

    assert joins["base_graph_count"] == 1
    assert joins["candidate_count"] == 2
    assert joins["sensitivity"] == "timing_sensitive"
    assert all(record["supported_window_count"] == 3 for record in joins["records"])
    assert len({record["fingerprint"] for record in joins["records"]}) == 2
    assert all("horizontal_relations" not in record for record in joins["records"])


def test_texture_candidate_join_keeps_window_position_in_the_fingerprint() -> None:
    first_intervals = (1, 1, 1, 1, 1, 2, 2, 2, 2)
    second_intervals = (1, 1, 1, 1, 2, 2, 2, 2, 2)
    first_positions = tuple(accumulate((0, *first_intervals)))
    second_positions = tuple(accumulate((0, *second_intervals)))
    groups = tuple(
        GroupingAttack(index * 100_000, (f"event-{index}",), (0,))
        for index in range(len(first_positions))
    )
    notes = {f"event-{index}": _note(f"event-{index}", 60) for index in range(len(first_positions))}
    diagnostic = build_texture_edge_diagnostics(
        grouping_profiles={"profile": groups},
        notes_by_id=notes,
        ledger_sha256="a" * 64,
    )

    joins = build_texture_candidate_joins(
        (
            _candidate(first_positions, profile_id="profile"),
            _candidate(second_positions, profile_id="profile"),
        ),
        texture_diagnostic=diagnostic,
    )

    assert joins["sensitivity"] == "timing_sensitive"
    assert len({record["fingerprint"] for record in joins["records"]}) == 2


def test_event_id_measurement_counts_actual_string_references() -> None:
    measurement = measure_event_id_references(
        {"first": ["event-a", "event-a"], "second": {"value": "event-b"}},
        event_ids={"event-a", "event-b", "event-c"},
    )

    assert measurement == {"unique_count": 2, "occurrence_count": 3}


def test_known_voice_diagnostic_reports_pitch_rank_bias_without_claiming_voice_accuracy() -> None:
    groups = (
        GroupingAttack(0, ("first-upper", "first-lower"), (0, 0)),
        GroupingAttack(100_000, ("second-upper", "second-lower"), (0, 0)),
    )
    notes = {
        "first-upper": _note("first-upper", 60),
        "first-lower": _note("first-lower", 64),
        "second-upper": _note("second-upper", 62),
        "second-lower": _note("second-lower", 65),
    }
    alignment = KnownScoreNoteAlignment(
        status="assessed",
        matches=tuple(
            AlignedExpectedObservedAttack(
                expected=ExpectedScoreAttack(
                    event_id=f"expected-{event_id}",
                    score_unit=0 if event_id.startswith("first") else 1,
                    pitch=note.pitch,
                    voice="upper" if event_id.endswith("upper") else "lower",
                ),
                observed=note,
            )
            for event_id, note in notes.items()
        ),
        assumptions=("pitch-local non-crossing alignment",),
        unmatched_expected_count=0,
        unmatched_observed_count=0,
    )
    texture = build_texture_edge_diagnostics(
        grouping_profiles={"profile": groups},
        notes_by_id=notes,
        ledger_sha256="a" * 64,
    )

    diagnostic = build_known_voice_diagnostics(
        alignment=alignment,
        grouping_profiles={"profile": groups},
        notes_by_id=notes,
        texture_diagnostic=texture,
        oracle_basis="generator_voice_labels",
    )
    profile = diagnostic["profiles"][0]

    assert diagnostic["status"] == "assessed"
    assert diagnostic["oracle_basis"] == "generator_voice_labels"
    assert diagnostic["claim_scope"] == "generator_ir_compatibility_only"
    assert profile["highest_upper_match_rate"] == 0.0
    assert profile["lowest_lower_match_rate"] == 0.0
    assert profile["upper_below_lower_group_count"] == 2
    assert profile["mutual_nearest_same_label_rate"] == 1.0
    assert profile["adjacent_all_pair_same_label_rate"] == 0.5
    assert profile["matched_pair_weighted_null_same_label_rate"] == 0.5
    assert profile["same_label_rate_lift"] == 0.5
    assert profile["mutual_nearest_virtual_pair_count"] == 2
    assert profile["adjacent_all_pair_count"] == 4
    assert profile["nearest_null_comparison_status"] == "assessed"
    assert profile["same_label_continuation_counts"] == {"zero": 0, "one": 2, "multiple": 0}
    serialized = json.dumps(diagnostic, sort_keys=True)
    assert "first-upper" not in serialized
    assert "second-lower" not in serialized


def test_matched_pair_weighted_null_uses_selected_pair_count_per_boundary() -> None:
    rate = matched_pair_weighted_null_rate(
        (
            (1, 4, 1),
            (12, 16, 1),
        )
    )

    assert rate == 0.5
    assert rate != pytest.approx(13 / 20)


def test_known_voice_diagnostic_can_report_negative_and_unavailable_nearest_lift() -> None:
    crossing_groups = (
        GroupingAttack(0, ("source-upper", "source-lower"), (0, 0)),
        GroupingAttack(100_000, ("target-lower", "target-upper"), (0, 0)),
    )
    notes = {
        "source-upper": _note("source-upper", 60),
        "source-lower": _note("source-lower", 72),
        "target-lower": _note("target-lower", 61),
        "target-upper": _note("target-upper", 71),
    }
    alignment = KnownScoreNoteAlignment(
        status="assessed",
        matches=tuple(
            AlignedExpectedObservedAttack(
                expected=ExpectedScoreAttack(
                    event_id=f"expected-{event_id}",
                    score_unit=0 if event_id.startswith("source") else 1,
                    pitch=note.pitch,
                    voice="upper" if event_id.endswith("upper") else "lower",
                ),
                observed=note,
            )
            for event_id, note in notes.items()
        ),
        assumptions=("pitch-local non-crossing alignment",),
        unmatched_expected_count=0,
        unmatched_observed_count=0,
    )
    profiles = {
        "crossing": crossing_groups,
        "single": (GroupingAttack(0, ("source-upper", "source-lower"), (0, 0)),),
    }
    texture = build_texture_edge_diagnostics(
        grouping_profiles=profiles,
        notes_by_id=notes,
        ledger_sha256="b" * 64,
    )

    diagnostic = build_known_voice_diagnostics(
        alignment=alignment,
        grouping_profiles=profiles,
        notes_by_id=notes,
        texture_diagnostic=texture,
        oracle_basis="generator_voice_labels",
    )
    by_alias = {profile["aliases"][0]: profile for profile in diagnostic["profiles"]}

    assert by_alias["crossing"]["mutual_nearest_same_label_rate"] == 0.0
    assert by_alias["crossing"]["adjacent_all_pair_same_label_rate"] == 0.5
    assert by_alias["crossing"]["matched_pair_weighted_null_same_label_rate"] == 0.5
    assert by_alias["crossing"]["same_label_rate_lift"] == -0.5
    assert by_alias["crossing"]["nearest_null_comparison_status"] == "assessed"
    assert by_alias["single"]["adjacent_all_pair_same_label_rate"] is None
    assert by_alias["single"]["same_label_rate_lift"] is None
    assert by_alias["single"]["nearest_null_comparison_status"] == "unable_to_investigate"


def test_known_voice_oracle_does_not_change_the_frozen_texture_diagnostic() -> None:
    groups = (
        GroupingAttack(0, ("a", "b"), (0, 0)),
        GroupingAttack(100_000, ("c", "d"), (0, 0)),
    )
    notes = {
        "a": _note("a", 60),
        "b": _note("b", 60),
        "c": _note("c", 62),
        "d": _note("d", 62),
    }
    texture = build_texture_edge_diagnostics(
        grouping_profiles={"profile": groups},
        notes_by_id=notes,
        ledger_sha256="c" * 64,
    )
    frozen = json.dumps(texture, sort_keys=True)

    for swapped in (False, True):
        alignment = KnownScoreNoteAlignment(
            status="assessed",
            matches=tuple(
                AlignedExpectedObservedAttack(
                    expected=ExpectedScoreAttack(
                        event_id=f"expected-{event_id}",
                        score_unit=0,
                        pitch=note.pitch,
                        voice=("upper" if (index % 2 == 0) is not swapped else "lower"),
                    ),
                    observed=note,
                )
                for index, (event_id, note) in enumerate(notes.items())
            ),
            assumptions=("pitch-local non-crossing alignment",),
            unmatched_expected_count=0,
            unmatched_observed_count=0,
        )
        diagnostic = build_known_voice_diagnostics(
            alignment=alignment,
            grouping_profiles={"profile": groups},
            notes_by_id=notes,
            texture_diagnostic=texture,
            oracle_basis="generator_voice_labels",
        )
        profile = diagnostic["profiles"][0]
        assert profile["mutual_nearest_virtual_pair_count"] == 4
        assert profile["adjacent_all_pair_count"] == 4
        assert profile["mutual_nearest_same_label_rate"] == 0.5
        assert profile["adjacent_all_pair_same_label_rate"] == 0.5
        assert profile["same_label_rate_lift"] == 0.0

    assert json.dumps(texture, sort_keys=True) == frozen


def test_non_rolled_render_keeps_nearest_null_rates_when_voice_labels_are_swapped(
    tmp_path,
) -> None:
    plan = PiecePlan(
        plan_id="plan",
        title="counterfactual",
        tonal_center=0,
        mode="major",
        root_node_id="root",
        ending_intent="tonic",
        nodes=(
            PlanNode("root", None, 0, "whole"),
            PlanNode(
                "leaf",
                "root",
                0,
                "statement",
                duration_weight=1,
                score_material_id="material",
            ),
        ),
    )
    notes = (
        ScoreNote("a", 0, 1, 60, "upper"),
        ScoreNote("b", 0, 1, 67, "lower"),
        ScoreNote("c", 2, 1, 62, "upper"),
        ScoreNote("d", 2, 1, 65, "lower"),
    )
    score = ScoreSpec(
        "score",
        4,
        (ScoreMaterial("material", 4, notes),),
    )
    swapped = replace(
        score,
        materials=(
            replace(
                score.materials[0],
                notes=tuple(
                    replace(note, voice="lower" if note.voice == "upper" else "upper")
                    for note in notes
                ),
            ),
        ),
    )
    performance = PerformanceSpec(
        "performance",
        4_000,
        64,
        "subtle-v1",
        (NodePerformance("root", coordination_profile="score", pedal_profile="none"),),
    )

    def diagnose(candidate: ScoreSpec, name: str) -> tuple[dict, dict]:
        smf_path = tmp_path / f"{name}.mid"
        render_performance_smf(render_performance(plan, candidate, performance), smf_path)
        observed_smf = load_observed_smf(smf_path)
        observed = build_observed_performance(observed_smf)
        profiles = build_grouping_profiles(observed)
        notes_by_id = {note.note_on_event_id: note for note in observed.notes}
        texture = build_texture_edge_diagnostics(
            grouping_profiles=profiles,
            notes_by_id=notes_by_id,
            ledger_sha256=observed_smf.ledger_sha256,
        )
        voice = build_known_voice_diagnostics(
            alignment=align_known_score_note_events(plan, candidate, observed),
            grouping_profiles=profiles,
            notes_by_id=notes_by_id,
            texture_diagnostic=texture,
            oracle_basis="generator_voice_labels",
        )
        return texture, voice

    original_texture, original_voice = diagnose(score, "original")
    swapped_texture, swapped_voice = diagnose(swapped, "swapped")

    def structural_profiles(texture: dict) -> list[tuple[object, object]]:
        return [
            (profile["vertical_groups"], profile["horizontal_relations"])
            for profile in texture["profiles"]
        ]

    assert structural_profiles(original_texture) == structural_profiles(swapped_texture)
    assert [
        (
            profile["matched_pair_weighted_null_same_label_rate"],
            profile["mutual_nearest_same_label_rate"],
            profile["same_label_rate_lift"],
        )
        for profile in original_voice["profiles"]
    ] == [
        (
            profile["matched_pair_weighted_null_same_label_rate"],
            profile["mutual_nearest_same_label_rate"],
            profile["same_label_rate_lift"],
        )
        for profile in swapped_voice["profiles"]
    ]


def test_oracle_dependency_metadata_is_verified_against_performance_dsl(tmp_path) -> None:
    performance_path = tmp_path / "performance.dsl"
    performance_path.write_text(
        dump_performance_spec(
            PerformanceSpec(
                "performance",
                4_000,
                64,
                "subtle-v1",
                (NodePerformance("root", coordination_profile="rolled"),),
            )
        ),
        encoding="utf-8",
    )
    source = SimpleNamespace(
        case_id="mismatch",
        performance_path=performance_path,
        oracle_observation_dependency="absent",
    )

    with pytest.raises(ValueError, match="oracle observation dependency mismatch"):
        _verified_oracle_observation_dependency(source)


def test_observation_runner_writes_separate_texture_artifact_and_manifest_entry(
    tmp_path,
) -> None:
    smf_path = tmp_path / "source.mid"
    midi = MidiFile(type=0, ticks_per_beat=480)
    track = MidiTrack()
    midi.tracks.append(track)
    track.append(MetaMessage("set_tempo", tempo=500_000, time=0))
    for low, high in ((48, 60), (50, 62), (52, 64), (53, 65)):
        track.append(Message("note_on", note=low, velocity=64, time=0))
        track.append(Message("note_on", note=high, velocity=64, time=0))
        track.append(Message("note_off", note=low, velocity=0, time=240))
        track.append(Message("note_off", note=high, velocity=0, time=0))
        track.append(Message("program_change", program=0, time=240))
    track.append(MetaMessage("end_of_track", time=0))
    midi.save(smf_path)
    output_dir = tmp_path / "observation"

    result = run_upper_evidence_observation(
        sources=(
            SimpleNamespace(
                case_id="source",
                group_id="group",
                smf_path=smf_path,
            ),
        ),
        output_dir=output_dir,
    )

    assert result["status"] == "pass"
    texture_records = [
        json.loads(line)
        for line in (output_dir / "observed-texture-edge-diagnostics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert texture_records[0]["texture_diagnostic"]["semantic_profile_count"] == 1
    assert texture_records[0]["candidate_join"]["base_graph_count"] == 1
    assert texture_records[0]["measurements"]["texture_event_id_unique_count"] == 0
    assert texture_records[0]["measurements"]["texture_event_id_occurrence_count"] == 0
    manifest = json.loads((output_dir / "observation-manifest.json").read_text(encoding="utf-8"))
    assert "observed-texture-edge-diagnostics.jsonl" in manifest["outputs"]


def test_oracle_runner_writes_voice_bias_diagnostics_after_observation_freeze(
    tmp_path,
) -> None:
    smf_path = tmp_path / "known.mid"
    midi = MidiFile(type=0, ticks_per_beat=480)
    track = MidiTrack()
    midi.tracks.append(track)
    track.append(MetaMessage("set_tempo", tempo=500_000, time=0))
    pitch_pairs = ((48, 60), (50, 62), (52, 64), (53, 65))
    for low, high in pitch_pairs:
        track.append(Message("note_on", note=low, velocity=64, time=0))
        track.append(Message("note_on", note=high, velocity=64, time=0))
        track.append(Message("note_off", note=low, velocity=0, time=240))
        track.append(Message("note_off", note=high, velocity=0, time=0))
        track.append(Message("program_change", program=0, time=240))
    track.append(MetaMessage("end_of_track", time=0))
    midi.save(smf_path)
    plan = PiecePlan(
        plan_id="known",
        title="known",
        tonal_center=0,
        mode="major",
        root_node_id="root",
        ending_intent="tonic",
        nodes=(
            PlanNode("root", None, 0, "whole"),
            PlanNode(
                "leaf",
                "root",
                0,
                "statement",
                duration_weight=1,
                score_material_id="material",
            ),
        ),
    )
    score = ScoreSpec(
        score_id="known-score",
        divisions=4,
        materials=(
            ScoreMaterial(
                material_id="material",
                length_units=8,
                notes=tuple(
                    note
                    for index, (low, high) in enumerate(pitch_pairs)
                    for note in (
                        ScoreNote(f"low-{index}", index * 2, 1, low, "lower"),
                        ScoreNote(f"high-{index}", index * 2, 1, high, "upper"),
                    )
                ),
            ),
        ),
    )
    piece_path = tmp_path / "piece.dsl"
    score_path = tmp_path / "score.dsl"
    piece_path.write_text(dump_piece_plan(plan), encoding="utf-8")
    score_path.write_text(dump_score_spec(score), encoding="utf-8")
    source = SimpleNamespace(
        case_id="known",
        group_id="arbitrary-group-name",
        smf_path=smf_path,
        piece_path=piece_path,
        score_path=score_path,
        oracle_basis="pitch_rank_constructed",
        oracle_observation_dependency="absent",
    )
    output_dir = tmp_path / "oracle"
    run_upper_evidence_observation(sources=(source,), output_dir=output_dir)

    result = run_upper_evidence_oracle_annotation(sources=(source,), output_dir=output_dir)

    assert result["status"] == "pass"
    assert result["texture_voice_assessed_source_count"] == 1
    assert result["texture_voice_pitch_rank_constructed_source_count"] == 1
    records = [
        json.loads(line)
        for line in (output_dir / "texture-voice-oracle-annotations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert records[0]["oracle_basis"] == "pitch_rank_constructed"
    assert records[0]["oracle_observation_dependency"] == "absent"
    assert records[0]["positive_evidence_eligible"] is True
    assert records[0]["claim_scope"] == "generator_ir_compatibility_only"
    texture_result = json.loads((output_dir / "texture-result.json").read_text(encoding="utf-8"))
    assert texture_result["status_scope"] == "oracle_annotation_completion_only"
    assert texture_result["perceptual_voice_evidence_status"] == "insufficient"
