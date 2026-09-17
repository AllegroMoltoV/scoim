from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_musical_composer.known_structure_evaluation import (
    evaluate_known_source,
    run_known_structure_evaluation,
)
from llm_musical_composer.run_state import sha256_file


def _observed() -> dict[str, object]:
    return {
        "schema_version": 2,
        "name": "fixture-a.mid",
        "source_sha256": "source",
        "semantic_profiles": [
            {
                "semantic_profile_hash": "profile",
                "group_count": 4,
                "boundary": {
                    "candidates": [{"start_group_index": 2}],
                },
                "recurrence": {
                    "candidates": [
                        {
                            "candidate_id": "repeat",
                            "window_size": 2,
                            "occurrences": [[0, False, False], [2, False, False]],
                        }
                    ]
                },
            }
        ],
        "score_timing_aliases": [
            {
                "alias_id": "alias",
                "semantic_profile_hash": "profile",
                "candidate_ids": ["timing-a"],
            }
        ],
    }


def _timing() -> dict[str, object]:
    return {
        "name": "fixture-a.mid",
        "source_sha256": "source",
        "candidates": [
            {
                "candidate_id": "timing-a",
                "attack_group_refs": [
                    {"score_position": value} for value in (0, 1, 2, 3)
                ],
            }
        ],
    }


def _oracle(candidate_ids: list[str]) -> dict[str, object]:
    return {
        "fixture_id": "fixture-a",
        "score_timing_dependency": {
            "status": "matched_candidate" if candidate_ids else "outside_candidate_space",
            "candidate_ids": candidate_ids,
        },
        "candidates": [
            {
                "descriptor": {
                    "nodes": [
                        {"start": [0, 1], "end": [1, 3]},
                        {"start": [1, 3], "end": [2, 3]},
                        {"start": [2, 3], "end": [1, 1]},
                    ],
                    "materials": [],
                }
            },
            {
                "descriptor": {
                    "nodes": [
                        {"start": [0, 1], "end": [2, 3]},
                        {"start": [2, 3], "end": [1, 1]},
                    ],
                    "materials": [],
                }
            },
        ],
    }


def test_known_evaluation_separates_common_allowed_and_recurrence_observation() -> None:
    result = evaluate_known_source(
        observed_source=_observed(),
        timing_source=_timing(),
        oracle_source=_oracle(["timing-a"]),
    )

    alias = result["aliases"][0]
    assert alias["mapping_status"] == "candidate_aligned"
    assert alias["observed_boundaries"] == [
        {"position": [2, 3], "oracle_relation": "common_boundary"}
    ]
    assert alias["oracle_common_boundaries"] == [[2, 3]]
    assert alias["oracle_allowed_boundaries"] == [[1, 3], [2, 3]]
    assert alias["observed_recurrence_spans"] == [
        {
            "candidate_id": "repeat",
            "spans": [[[0, 1], [2, 3]], [[2, 3], [1, 1]]],
        }
    ]
    assert "oracle_material_annotations" in result
    assert "oracle_material_annotations" not in alias


def test_known_evaluation_marks_mappable_candidate_space_miss_without_calling_it_false() -> None:
    result = evaluate_known_source(
        observed_source=_observed(),
        timing_source=_timing(),
        oracle_source=_oracle([]),
    )

    alias = result["aliases"][0]
    assert alias["mapping_status"] == "oracle_mapped"
    assert "false" not in str(alias).lower()


def test_known_evaluation_runner_keeps_raw_inputs_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed_files = tmp_path / "source-candidates.jsonl"
    timing_files = tmp_path / "timing-files.jsonl"
    oracle_files = tmp_path / "candidate-sets.jsonl"
    observed_files.write_text(json.dumps(_observed()) + "\n", encoding="utf-8")
    timing_files.write_text(json.dumps(_timing()) + "\n", encoding="utf-8")
    oracle_files.write_text(json.dumps(_oracle(["timing-a"])) + "\n", encoding="utf-8")
    method_manifest = tmp_path / "method-manifest.json"
    method_manifest.write_text("{}", encoding="utf-8")
    observed_manifest = tmp_path / "observed-manifest.json"
    timing_manifest = tmp_path / "timing-manifest.json"
    oracle_manifest = tmp_path / "oracle-manifest.json"
    observed_manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "pass",
                "inputs": {"method_manifest": "f" * 64},
                "outputs": {"source-candidates.jsonl": sha256_file(observed_files)},
            }
        ),
        encoding="utf-8",
    )
    timing_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "pass",
                "inputs": {"method_manifest": "f" * 64},
                "outputs": {"files.jsonl": sha256_file(timing_files)},
            }
        ),
        encoding="utf-8",
    )
    oracle_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "pass",
                "outputs": {"candidate-sets.jsonl": sha256_file(oracle_files)},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "llm_musical_composer.known_structure_evaluation.verify_method_manifest",
        lambda **_arguments: "f" * 64,
    )

    result = run_known_structure_evaluation(
        observed_files=observed_files,
        observed_manifest=observed_manifest,
        timing_files=timing_files,
        timing_manifest=timing_manifest,
        oracle_files=oracle_files,
        oracle_manifest=oracle_manifest,
        repository_root=Path.cwd(),
        method_manifest=method_manifest,
        output_dir=tmp_path / "evaluation",
    )

    assert result["status"] == "pass"
    assert result["mapping_counts"] == {
        "candidate_aligned": 1,
        "oracle_mapped": 0,
        "unable_to_compare": 0,
    }
    assert (tmp_path / "evaluation" / "evaluation.jsonl").is_file()
    assert not (tmp_path / "evaluation" / "source-candidates.jsonl").exists()
