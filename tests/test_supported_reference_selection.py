import json
from pathlib import Path

import pytest

from llm_musical_composer.run_state import sha256_json
from llm_musical_composer.supported_reference_selection import (
    SupportedReferenceSelectionError,
    select_name_from_candidates,
    select_supported_reference,
)

PROJECT_ROOT = Path(__file__).parents[1]
ANALYSIS_DIR = PROJECT_ROOT / ".appendix" / "corpus-baseline-preflight-v1"


def test_fixed_seed_selects_expected_supported_reference() -> None:
    result = select_supported_reference(PROJECT_ROOT, ANALYSIS_DIR)

    assert result.selection["population"] == "eligible_copy_deduplicated"
    assert result.selection["distance_method"] == "empirical_quantile"
    assert result.selection["k"] == 7
    assert result.selection["fraction"] == 0.75
    assert result.selection["candidate_count"] == 171
    assert result.selection["analysis_manifest_sha256"] == (
        "f51fe63c786b5b54555b0cd233799c8b7977f51f7bf0880cf8d9ad0ea0208980"
    )
    assert result.selection["selection_sha256"] == (
        "17e39323a3a1450f535294a837c6d20a54c0df0baa1411d51e2e6d746ddcf952"
    )
    assert result.selection["selected_index"] == 79
    assert result.selection["selected_name"] == "mayodance.mid"
    assert result.selection["seed_hash_method"] == "run-state-sha256-json-v1"
    assert result.resolved_request["reference"]["selection_method"] == (
        "supported_pool_seed_v1"
    )
    assert result.medoid_control["reference"]["name"] == "SonataForHouseMoving.mid"
    assert result.medoid_control["reference"]["selection_method"] == "automatic_medoid"

    serialized_prompt = json.dumps(
        result.target.prompt_target, ensure_ascii=False, sort_keys=True
    )
    assert "mayodance.mid" not in serialized_prompt
    assert result.resolved_request["reference"]["sha256"] not in serialized_prompt
    assert "copy_fingerprint" not in serialized_prompt


def test_candidate_input_order_does_not_change_selection() -> None:
    candidates = json.loads(
        (ANALYSIS_DIR / "candidate-pool-sensitivity.json").read_text(encoding="utf-8")
    )
    names = candidates["candidate_sets"]["eligible_copy_deduplicated"][
        "empirical_quantile"
    ]["k=7:0.75"]["names"]
    digest = "17e39323a3a1450f535294a837c6d20a54c0df0baa1411d51e2e6d746ddcf952"

    assert select_name_from_candidates(names, digest) == (
        79,
        "mayodance.mid",
    )
    assert select_name_from_candidates(list(reversed(names)), digest) == (
        79,
        "mayodance.mid",
    )


def test_internal_explicit_brightness_rebuilds_request_provenance() -> None:
    result = select_supported_reference(
        PROJECT_ROOT,
        ANALYSIS_DIR,
        explicit_brightness=0,
    )

    assert result.normalized_request["controls"] == {
        "brightness": {"label": "あかるさ", "value": 0.0}
    }
    assert result.resolved_request["controls"] == result.normalized_request["controls"]
    assert result.resolved_request["normalized_request_sha256"] == sha256_json(
        result.normalized_request
    )
    assert result.resolved_request["reference"]["name"] == "mayodance.mid"
    assert result.target.artifact["controls"]["brightness"] == {
        "label": "あかるさ",
        "source": "explicit",
        "value": 0.0,
    }
    tonal = result.target.prompt_target["semantic_targets"]["piece_plan"][0]
    assert tonal["status"] == "specified"
    assert tonal["scale_policy"] == "dorian"


def test_internal_explicit_attack_frequency_rebuilds_request_provenance() -> None:
    result = select_supported_reference(
        PROJECT_ROOT,
        ANALYSIS_DIR,
        explicit_attack_frequency=0.0,
    )

    assert result.normalized_request["controls"] == {
        "attack_frequency": {"label": "発音頻度", "value": 0.0}
    }
    assert result.resolved_request["controls"] == result.normalized_request["controls"]
    assert result.resolved_request["normalized_request_sha256"] == sha256_json(
        result.normalized_request
    )
    assert result.target.artifact["controls"]["attack_frequency"] == {
        "label": "発音頻度",
        "source": "explicit",
        "value": 0.0,
    }
    frequency = next(
        item
        for item in result.target.prompt_target["semantic_targets"]["score_spec"]
        if item["id"] == "attack_frequency"
    )
    assert frequency["anchor"] == {
        "normalized": 0.0,
        "raw_groups_per_second": 4.17764351,
    }
    assert frequency["strict_score_budget"] == {
        "groups": 752,
        "source": "explicit_corpus_min_max_half_up",
    }


@pytest.mark.parametrize("value", [-1.01, 1.01, float("inf")])
def test_internal_explicit_attack_frequency_rejects_out_of_domain(
    value: float,
) -> None:
    with pytest.raises(ValueError, match=r"-1\.0 through 1\.0"):
        select_supported_reference(
            PROJECT_ROOT,
            ANALYSIS_DIR,
            explicit_attack_frequency=value,
        )


@pytest.mark.parametrize("value", [-2, -0.5, 0.5, 2])
def test_internal_explicit_brightness_rejects_non_public_domain(value: float) -> None:
    with pytest.raises(ValueError, match="-1, 0, or 1"):
        select_supported_reference(
            PROJECT_ROOT,
            ANALYSIS_DIR,
            explicit_brightness=value,
        )


def test_analysis_output_hash_mismatch_stops_selection(tmp_path: Path) -> None:
    copied = tmp_path / "analysis"
    copied.mkdir()
    for path in ANALYSIS_DIR.iterdir():
        if path.is_file():
            (copied / path.name).write_bytes(path.read_bytes())
    with (copied / "candidate-pool-sensitivity.json").open("ab") as output:
        output.write(b"\n")

    with pytest.raises(SupportedReferenceSelectionError, match="hash mismatch"):
        select_supported_reference(
            PROJECT_ROOT,
            copied,
            expected_analysis_manifest_sha256=None,
            expected_selection_sha256=None,
        )


def test_different_seed_changes_selection_hash() -> None:
    result = select_supported_reference(
        PROJECT_ROOT,
        ANALYSIS_DIR,
        seed="normal-generation-v1-seed-002",
        expected_selection_sha256=None,
    )

    assert result.selection["selection_sha256"] != (
        "17e39323a3a1450f535294a837c6d20a54c0df0baa1411d51e2e6d746ddcf952"
    )
