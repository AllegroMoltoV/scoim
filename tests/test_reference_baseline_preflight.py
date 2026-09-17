import copy

import pytest

from llm_musical_composer.reference_baseline_preflight import (
    PreflightError,
    average_rank_spearman,
    build_pair_distance_model,
    compare_k_configurations,
    control_distance,
    empirical_midrank,
    evaluate_target_eligibility,
    high_precision_profile_distances,
    local_radii,
    select_candidates,
    select_farthest_first_from_pairs,
    set_comparison,
    summarize_coverage,
)
from llm_musical_composer.reference_profile import FEATURE_GROUPS, profile_distances
from llm_musical_composer.score_timing_split import select_development_records


def _profile(value: float) -> dict[str, object]:
    return {
        "status": "pass",
        "feature_groups": {
            group: {
                "metrics": {
                    "scalar": {"kind": "scalar", "values": [value]},
                    "distribution": {
                        "kind": "distribution",
                        "values": [value, 1.0 - value],
                    },
                }
            }
            for group in FEATURE_GROUPS
        },
    }


def test_high_precision_distances_match_existing_rounding_without_early_rounding() -> None:
    first = _profile(0.12345678)
    second = _profile(0.87654321)

    precise = high_precision_profile_distances(first, second)

    assert precise == {group: pytest.approx(0.75308643) for group in FEATURE_GROUPS}
    assert {group: round(value, 8) for group, value in precise.items()} == profile_distances(
        first, second
    )


def test_empirical_midrank_handles_ties_and_rejects_self_distance_shortcuts() -> None:
    values = [0.1, 0.2, 0.2, 0.4]

    assert empirical_midrank(0.2, values) == pytest.approx(0.5)
    assert empirical_midrank(0.1, values) == pytest.approx(0.125)
    with pytest.raises(PreflightError, match="present"):
        empirical_midrank(0.3, values)


def test_pair_model_and_local_radius_identify_an_isolated_piece() -> None:
    profiles = {
        "a.mid": _profile(0.10),
        "b.mid": _profile(0.11),
        "c.mid": _profile(0.12),
        "isolated.mid": _profile(0.90),
    }

    model = build_pair_distance_model(profiles)
    radii = local_radii(model["empirical_quantile"], profiles, k=2)

    assert radii["isolated.mid"] > radii["a.mid"]
    assert all(name != other for name, other in model["empirical_quantile"])
    assert model["group_medians"] == {group: pytest.approx(0.40) for group in FEATURE_GROUPS}


def test_pair_analysis_is_deterministic_under_input_reordering() -> None:
    profiles = {
        "Beta.mid": _profile(0.11),
        "alpha.mid": _profile(0.11),
        "zeta.mid": _profile(0.50),
        "center.mid": _profile(0.10),
    }
    reversed_profiles = dict(reversed(list(profiles.items())))

    first = build_pair_distance_model(profiles)
    second = build_pair_distance_model(reversed_profiles)

    assert first == second
    assert local_radii(first["empirical_quantile"], profiles, k=1) == local_radii(
        second["empirical_quantile"], reversed_profiles, k=1
    )


def test_candidate_boundary_includes_all_ties_at_nearest_rank_cutoff() -> None:
    radii = {"a": 0.1, "b": 0.2, "c": 0.2, "d": 0.9}

    selected = select_candidates(radii, fraction=0.5)

    assert selected["requested_count"] == 2
    assert selected["threshold"] == 0.2
    assert selected["names"] == ["a", "b", "c"]


def test_target_eligibility_excludes_failures_without_losing_the_reason() -> None:
    def builder(name: str) -> None:
        if name == "bad.mid":
            raise ValueError("vertical interval distribution is invalid")

    result = evaluate_target_eligibility(["good.mid", "bad.mid"], builder)

    assert result == [
        {
            "name": "bad.mid",
            "status": "failed",
            "error_type": "ValueError",
            "reason": "vertical interval distribution is invalid",
        },
        {"name": "good.mid", "status": "success"},
    ]


def test_control_distance_keeps_continuous_reference_values_separate_from_user_domain() -> None:
    first = {"あかるさ": -0.25, "高さ": 0.5, "発音頻度": -1.0}
    second = {"あかるさ": 0.25, "高さ": -0.5, "発音頻度": -0.5}

    assert control_distance(first, second) == pytest.approx(0.5)
    with pytest.raises(PreflightError, match="control"):
        control_distance(first, {"あかるさ": 0.25, "高さ": -0.5})


def test_coverage_reports_control_and_feature_spaces_separately() -> None:
    profiles = {
        "a.mid": _profile(0.1),
        "b.mid": _profile(0.2),
        "c.mid": _profile(0.9),
    }
    controls = {
        "a.mid": {"あかるさ": -1.0, "高さ": -1.0, "発音頻度": -1.0},
        "b.mid": {"あかるさ": 0.0, "高さ": 0.0, "発音頻度": 0.0},
        "c.mid": {"あかるさ": 1.0, "高さ": 1.0, "発音頻度": 1.0},
    }
    model = build_pair_distance_model(profiles)

    result = summarize_coverage(
        population_names=list(profiles),
        candidate_names=["a.mid", "b.mid"],
        controls=controls,
        empirical_pairs=model["empirical_quantile"],
        raw_pairs=model["raw"],
    )

    assert result["control_nearest_distance"]["maximum"]["name"] == "c.mid"
    assert result["feature_nearest_composite"]["maximum"]["name"] == "c.mid"
    assert set(result["feature_nearest_by_group"]) == set(FEATURE_GROUPS)
    assert result["reference_controls"]["あかるさ"]["maximum"] == 0.0
    assert result["user_brightness_request_distance"]["1"] == 1.0


def test_distance_methods_and_candidate_sets_can_be_compared_without_hiding_disagreement() -> None:
    first = {"a": 1.0, "b": 2.0, "c": 2.0, "d": 4.0}
    second = {"a": 1.0, "b": 3.0, "c": 2.0, "d": 4.0}

    assert average_rank_spearman(first, second) == pytest.approx(0.9486832980505138)
    assert set_comparison({"a", "b"}, {"b", "c"}) == {
        "intersection_count": 1,
        "union_count": 3,
        "retention": 0.5,
        "jaccard": pytest.approx(1 / 3),
    }


def test_high_precision_distance_rejects_invalid_metric_contract() -> None:
    first = _profile(0.1)
    second = copy.deepcopy(first)
    second["feature_groups"]["rhythm_time"]["metrics"]["scalar"]["values"] = []

    with pytest.raises(PreflightError, match="dimensions"):
        high_precision_profile_distances(first, second)


def test_cached_farthest_first_matches_existing_selector() -> None:
    records = [
        {
            "name": name,
            "status": "pass",
            "profile": _profile(value),
            "copy_fingerprint": {"sequence_sha256": f"hash-{name}"},
        }
        for name, value in {
            "a.mid": 0.10,
            "b.mid": 0.15,
            "c.mid": 0.40,
            "d.mid": 0.90,
        }.items()
    ]
    model = build_pair_distance_model({record["name"]: record["profile"] for record in records})

    cached = select_farthest_first_from_pairs(records, model["raw"], count=3)

    assert [item["name"] for item in cached] == [
        item["name"] for item in select_development_records(records, count=3)
    ]


def test_k_comparison_keeps_rank_and_candidate_set_evidence() -> None:
    radii = {
        5: {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4},
        7: {"a": 0.1, "b": 0.3, "c": 0.2, "d": 0.4},
    }

    result = compare_k_configurations(radii, fractions=(0.5,))

    assert result["k=5_vs_k=7"]["spearman"] == pytest.approx(0.8)
    assert result["k=5_vs_k=7"]["candidate_sets"]["0.50"]["jaccard"] == 1 / 3
