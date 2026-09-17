from __future__ import annotations

from dataclasses import asdict, replace
from fractions import Fraction
from itertools import product

from llm_musical_composer.reference_decomposition import (
    ObservedAttackGroup,
    ObservedNote,
    ObservedPerformance,
)
from llm_musical_composer.score_timing_hypothesis import (
    VOCABULARIES,
    _classify_boundary_transition,
    build_grouping_profiles,
    compare_candidate_to_known_score,
    compare_grouping_to_known_score,
    fit_dynamic_score_timing,
    fit_interval_vocabulary_global,
    generate_dynamic_score_timing_candidates,
    generate_score_timing_candidates,
)


def _performance(attacks: list[tuple[int, int]]) -> ObservedPerformance:
    notes = tuple(
        ObservedNote(
            note_on_event_id=f"on-{index}",
            note_off_event_id=f"off-{index}",
            track=0,
            channel=0,
            pitch=pitch,
            velocity=64,
            note_off_velocity=0,
            onset_tick=index * 10,
            offset_tick=index * 10 + 5,
            onset_us=onset_us,
            offset_us=onset_us + 100_000,
            matching_status="assessed",
        )
        for index, (onset_us, pitch) in enumerate(attacks)
    )
    groups: list[ObservedAttackGroup] = []
    index = 0
    while index < len(notes):
        anchor = notes[index].onset_us
        event_ids = []
        while index < len(notes) and notes[index].onset_us - anchor <= 30_000:
            event_ids.append(notes[index].note_on_event_id)
            index += 1
        groups.append(ObservedAttackGroup(anchor, tuple(event_ids)))
    return ObservedPerformance(
        event_times=(),
        notes=notes,
        attack_groups=tuple(groups),
        attack_group_window_us=30_000,
        note_matching_status="assessed",
        ambiguous_note_count=0,
        unmatched_note_off_count=0,
        dangling_note_on_count=0,
    )


def test_grouping_profiles_keep_30ms_and_60ms_rolled_rules_distinct() -> None:
    performance = _performance([(0, 48), (45_000, 60), (500_000, 52), (500_000, 64), (900_000, 55)])

    profiles = build_grouping_profiles(performance)

    assert len(profiles["attack-30ms"]) == 4
    assert len(profiles["rolled-merge-60ms"]) == 3
    assert profiles["rolled-merge-60ms"][0].relative_onsets_us == (0, 45_000)
    assert len(profiles["rolled-separate"]) == 4
    assert profiles["rolled-separate"][2].source_event_ids == ("on-2", "on-3")


def test_fixed_candidate_family_recovers_known_relative_score_intervals() -> None:
    positions = [0, 1, 3, 4, 7, 8]
    performance = _performance(
        [(position * 100_000, 60 + index) for index, position in enumerate(positions)]
    )

    result = generate_score_timing_candidates(performance, source_ledger_sha256="a" * 64)

    assert result.status == "assessed"
    assert len(result.candidates) == 27
    candidate = next(
        item
        for item in result.candidates
        if item.grouping_profile_id == "attack-30ms"
        and item.score_grid_id == "binary-dotted"
        and item.requested_segment_count == 1
    )
    inferred = tuple(
        right.score_position - left.score_position
        for left, right in zip(
            candidate.attack_group_refs, candidate.attack_group_refs[1:], strict=False
        )
    )
    assert inferred == (1, 2, 1, 3, 1)
    assert candidate.search_status == "complete"
    assert candidate.metrics["maximum_group_anchor_error_us"] == 0
    assert "withheld_anchor_mean_error_us" not in candidate.metrics
    assert candidate.metrics["in_sample_interpolation_mean_error_us"] == 0
    profiles = build_grouping_profiles(performance)
    known_units = {f"on-{index}": position for index, position in enumerate(positions)}
    comparison = compare_candidate_to_known_score(
        candidate,
        profiles[candidate.grouping_profile_id],
        known_units,
    )
    assert comparison["status"] == "equivalent"


def test_known_score_comparison_reports_grouping_conflict() -> None:
    performance = _performance([(0, 48), (20_000, 60), (200_000, 62)])
    result = generate_score_timing_candidates(
        performance,
        source_ledger_sha256="e" * 64,
    )
    candidate = next(
        item
        for item in result.candidates
        if item.grouping_profile_id == "attack-30ms"
        and item.score_grid_id == "binary"
        and item.requested_segment_count == 1
    )

    comparison = compare_candidate_to_known_score(
        candidate,
        build_grouping_profiles(performance)["attack-30ms"],
        {"on-0": 0, "on-1": 1, "on-2": 2},
    )

    assert comparison["status"] == "grouping_conflicts_known_score"


def test_grouping_comparison_separates_merge_and_split_failures() -> None:
    performance = _performance([(0, 48), (20_000, 60), (200_000, 62)])
    profiles = build_grouping_profiles(performance)

    compatible = compare_grouping_to_known_score(
        profiles["attack-30ms"],
        {"on-0": 0, "on-1": 0, "on-2": 1},
    )
    split = compare_grouping_to_known_score(
        profiles["rolled-separate"],
        {"on-0": 0, "on-1": 0, "on-2": 1},
    )
    merged = compare_grouping_to_known_score(
        profiles["attack-30ms"],
        {"on-0": 0, "on-1": 1, "on-2": 2},
    )

    assert compatible["status"] == "compatible"
    assert split["split_known_position_count"] == 1
    assert merged["merged_multiple_position_group_count"] == 1


def test_candidates_do_not_copy_pitch_velocity_or_one_time_knot_per_attack() -> None:
    performance = _performance([(index * 120_000, 48 + index % 12) for index in range(20)])

    result = generate_score_timing_candidates(performance, source_ledger_sha256="b" * 64)

    for candidate in result.candidates:
        payload = asdict(candidate)
        assert "pitch" not in str(payload).lower()
        assert "velocity" not in str(payload).lower()
        assert len(candidate.shared_time_map) <= 9
        assert len(candidate.shared_time_map) < len(candidate.attack_group_refs)


def test_negative_controls_are_kept_outside_adoptable_candidates() -> None:
    performance = _performance([(0, 60), (90_000, 62), (250_000, 64), (500_000, 65)])

    result = generate_score_timing_candidates(performance, source_ledger_sha256="c" * 64)

    assert {item.control_id for item in result.negative_controls} == {
        "observed-copy-negative-control",
        "uniform-grid-baseline",
    }
    assert all(item.status == "negative_control" for item in result.negative_controls)
    assert all(item.status == "candidate" for item in result.candidates)
    by_id = {item.control_id: item for item in result.negative_controls}
    assert by_id["observed-copy-negative-control"].metrics["vocabulary_size"] == 3
    assert by_id["uniform-grid-baseline"].metrics["vocabulary_size"] == 1
    assert all(
        item.metrics["normalized_serialization_bytes"] > 0 for item in result.negative_controls
    )


def test_unmatched_notes_are_reported_instead_of_silently_dropped() -> None:
    performance = _performance([(0, 60), (100_000, 62)])
    performance = replace(
        performance,
        attack_groups=(
            *performance.attack_groups,
            ObservedAttackGroup(200_000, ("unmatched-on",)),
        ),
        note_matching_status="unable_to_investigate",
        dangling_note_on_count=1,
    )

    result = generate_score_timing_candidates(performance, source_ledger_sha256="d" * 64)

    assert result.status == "unable_to_investigate"
    assert result.candidates == ()
    assert "note matching" in result.reason


def test_ambiguous_note_off_matching_does_not_block_note_on_candidates() -> None:
    performance = _performance([(0, 60), (100_000, 62), (200_000, 64)])
    performance = replace(
        performance,
        note_matching_status="ambiguous",
        ambiguous_note_count=2,
    )

    result = generate_score_timing_candidates(
        performance,
        source_ledger_sha256="f" * 64,
    )

    assert result.status == "assessed"
    assert len(result.candidates) == 27


def _exhaustive_vocabulary_error(
    observed: tuple[int, ...], vocabulary: tuple[Fraction, ...]
) -> Fraction:
    errors = []
    for intervals in product(vocabulary, repeat=len(observed)):
        scale = sum(
            Fraction(value) * interval for value, interval in zip(observed, intervals, strict=True)
        ) / sum(interval * interval for interval in intervals)
        errors.append(
            sum(
                (Fraction(value) - scale * interval) ** 2
                for value, interval in zip(observed, intervals, strict=True)
            )
        )
    return min(errors)


def test_global_vocabulary_fit_resolves_local_initialization_counterexample() -> None:
    vocabulary = tuple(Fraction(value) for value in (1, 2, 4, 8))

    fit = fit_interval_vocabulary_global((1, 1, 3), vocabulary)

    assert fit.normalized_intervals == (1, 1, 4)
    assert fit.squared_error == Fraction(1, 9)
    assert fit.squared_error == _exhaustive_vocabulary_error((1, 1, 3), vocabulary)


def test_global_vocabulary_fit_matches_small_exhaustive_oracle() -> None:
    vocabulary = tuple(Fraction(value) for value in (1, 2, 4))

    for observed in product(range(1, 5), repeat=3):
        fit = fit_interval_vocabulary_global(observed, vocabulary)
        assert fit.squared_error == _exhaustive_vocabulary_error(observed, vocabulary)


def test_boundary_transition_distinguishes_fixed_point_and_longer_cycle() -> None:
    assert _classify_boundary_transition(((0, 4, 8),), (0, 4, 8)) == "complete"
    assert (
        _classify_boundary_transition(
            ((0, 3, 8), (0, 5, 8), (0, 4, 8)),
            (0, 3, 8),
        )
        == "not_converged"
    )
    assert _classify_boundary_transition(((0, 3, 8),), (0, 4, 8)) == "continue"


def test_dynamic_score_timing_moves_boundary_to_tempo_change() -> None:
    true_intervals = (1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2)
    observed_intervals = tuple(
        interval * (100 if index < 6 else 160) for index, interval in enumerate(true_intervals)
    )

    fit = fit_dynamic_score_timing(
        observed_intervals,
        VOCABULARIES["binary"],
        requested_segment_count=2,
    )

    assert fit.search_status == "complete"
    assert fit.boundaries == (0, 6, len(true_intervals))
    assert fit.boundary_history[-1] == fit.boundaries
    assert fit.normalized_intervals == true_intervals
    assert len(fit.boundaries) <= 9


def test_dynamic_candidate_family_is_separate_from_fixed_27() -> None:
    performance = _performance([(index * 100_000, 60 + index % 5) for index in range(20)])

    result = generate_dynamic_score_timing_candidates(
        performance,
        source_ledger_sha256="a" * 64,
    )

    assert result.status == "assessed"
    assert len(result.candidates) == 6
    assert all(
        candidate.candidate_id.endswith("--dynamic-boundaries") for candidate in result.candidates
    )
    assert all(len(candidate.shared_time_map) <= 9 for candidate in result.candidates)
    assert all(
        candidate.search_status in {"complete", "not_converged"} for candidate in result.candidates
    )
