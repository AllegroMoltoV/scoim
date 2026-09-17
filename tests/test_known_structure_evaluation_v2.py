from __future__ import annotations

import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from llm_musical_composer.known_structure_evaluation_v2 import (
    KnownStructureEvaluationV2Error,
    evaluate_known_source_v2,
    oracle_attack_span,
    run_known_structure_evaluation_v2,
    verify_archived_method_manifest,
)
from llm_musical_composer.performance_pipeline import (
    PiecePlan,
    PlanNode,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
)


def _observed() -> dict[str, object]:
    return {
        "schema_version": 2,
        "name": "fixture-a.mid",
        "source_sha256": "source",
        "source_ledger_sha256": "ledger",
        "semantic_profiles": [
            {
                "semantic_profile_hash": "profile",
                "group_count": 4,
                "boundary": {"candidates": [{"start_group_index": 2}]},
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
        "source_ledger_sha256": "ledger",
        "candidates": [
            {
                "candidate_id": "timing-a",
                "attack_group_refs": [
                    {"score_position": value} for value in (0, 1, 2, 3)
                ],
            }
        ],
    }


def _oracle() -> dict[str, object]:
    return {
        "fixture_id": "fixture-a",
        "source_ledger_sha256": "ledger",
        "score_timing_dependency": {
            "status": "matched_candidate",
            "candidate_ids": ["timing-a"],
        },
        "candidates": [
            {
                "candidate_id": "structure-a",
                "origin_status": "oracle_fixture_baseline",
                "descriptor": {
                    "nodes": [
                        {"start": [0, 1], "end": [1, 2]},
                        {"start": [1, 2], "end": [1, 1]},
                    ],
                    "materials": [],
                    "derived_relations": [],
                    "terminal_tail_ratio": [1, 4],
                },
            }
        ],
    }


def test_v2_maps_score_timing_between_oracle_first_and_last_attacks() -> None:
    result = evaluate_known_source_v2(
        observed_source=_observed(),
        timing_source=_timing(),
        oracle_source=_oracle(),
        oracle_first_attack=Fraction(0),
        oracle_last_attack=Fraction(3, 4),
        expected_source_sha256="source",
    )

    alias = result["aliases"][0]
    assert alias["observed_boundaries"] == [
        {"position": [1, 2], "oracle_relation": "common_boundary"}
    ]
    assert alias["observed_boundaries"][0]["position"] != [2, 3]


def test_v2_preserves_oracle_leading_rest_in_affine_mapping() -> None:
    result = evaluate_known_source_v2(
        observed_source=_observed(),
        timing_source=_timing(),
        oracle_source=_oracle(),
        oracle_first_attack=Fraction(1, 4),
        oracle_last_attack=Fraction(3, 4),
        expected_source_sha256="source",
    )

    occurrence = result["aliases"][0]["observed_recurrences"][0]["occurrences"][0]
    assert occurrence["start"] == [1, 4]


def test_v2_keeps_mapped_occurrence_when_same_candidate_reaches_terminal() -> None:
    result = evaluate_known_source_v2(
        observed_source=_observed(),
        timing_source=_timing(),
        oracle_source=_oracle(),
        oracle_first_attack=Fraction(0),
        oracle_last_attack=Fraction(3, 4),
        expected_source_sha256="source",
    )

    occurrences = result["aliases"][0]["observed_recurrences"][0]["occurrences"]
    assert occurrences == [
        {"start": [0, 1], "end": [1, 2], "endpoint_status": "mapped"},
        {
            "start": [1, 2],
            "end": None,
            "endpoint_status": "terminal_endpoint_unobservable",
        },
    ]


def test_v2_rejects_source_ledger_mismatch() -> None:
    oracle = _oracle()
    oracle["source_ledger_sha256"] = "different"

    with pytest.raises(KnownStructureEvaluationV2Error, match="ledger"):
        evaluate_known_source_v2(
            observed_source=_observed(),
            timing_source=_timing(),
            oracle_source=oracle,
            oracle_first_attack=Fraction(0),
            oracle_last_attack=Fraction(3, 4),
            expected_source_sha256="source",
        )


def _plan_and_score_with_leading_and_terminal_space() -> tuple[PiecePlan, ScoreSpec]:
    plan = PiecePlan(
        plan_id="plan",
        title="fixture",
        tonal_center=0,
        mode="major",
        root_node_id="root",
        ending_intent="tonic",
        nodes=(
            PlanNode("root", None, 0, "whole"),
            PlanNode("a", "root", 0, "statement", duration_weight=1, score_material_id="a"),
            PlanNode("b", "root", 1, "contrast", duration_weight=1, score_material_id="b"),
        ),
    )
    score = ScoreSpec(
        score_id="score",
        divisions=4,
        materials=(
            ScoreMaterial(
                material_id="a",
                length_units=4,
                notes=(ScoreNote("a-note", 1, 1, 60, "upper"),),
            ),
            ScoreMaterial(
                material_id="b",
                length_units=4,
                notes=(ScoreNote("b-note", 2, 1, 64, "upper"),),
            ),
        ),
    )
    return plan, score


def _baseline_descriptor() -> dict[str, object]:
    return {
        "schema_version": 0,
        "nodes": [
            {
                "local_id": "internal-root",
                "node_kind": "internal",
                "parent_local_id": None,
                "sibling_order": 0,
                "start": [0, 1],
                "end": [1, 1],
            },
            {
                "local_id": "leaf-0",
                "node_kind": "leaf",
                "parent_local_id": "internal-root",
                "sibling_order": 0,
                "start": [0, 1],
                "end": [1, 2],
            },
            {
                "local_id": "leaf-1",
                "node_kind": "leaf",
                "parent_local_id": "internal-root",
                "sibling_order": 1,
                "start": [1, 2],
                "end": [1, 1],
            },
        ],
        "materials": [
            {
                "local_id": "material-0",
                "occurrence_leaf_ids": ["leaf-0"],
                "occurrence_spans": [{"start": [0, 1], "end": [1, 2]}],
            },
            {
                "local_id": "material-1",
                "occurrence_leaf_ids": ["leaf-1"],
                "occurrence_spans": [{"start": [1, 2], "end": [1, 1]}],
            },
        ],
        "derived_relations": [],
        "terminal_tail_ratio": [1, 4],
        "terminal_tail_source": "oracle_only",
        "absolute_unit_scale": "quotiented_out",
    }


def test_oracle_attack_span_comes_from_score_attacks_not_root_endpoints() -> None:
    plan, score = _plan_and_score_with_leading_and_terminal_space()
    oracle = {
        "candidates": [
            {
                "origin_status": "oracle_fixture_baseline",
                "descriptor": _baseline_descriptor(),
            },
            {
                "origin_status": "oracle_generated_exact_smf_variant",
                "descriptor": _baseline_descriptor(),
            },
        ]
    }

    assert oracle_attack_span(plan=plan, score=score, oracle_source=oracle) == (
        Fraction(1, 8),
        Fraction(3, 4),
    )


def test_oracle_attack_span_rejects_candidate_tail_disagreement() -> None:
    plan, score = _plan_and_score_with_leading_and_terminal_space()
    different = _baseline_descriptor()
    different["terminal_tail_ratio"] = [1, 8]
    oracle = {
        "candidates": [
            {
                "origin_status": "oracle_fixture_baseline",
                "descriptor": _baseline_descriptor(),
            },
            {
                "origin_status": "oracle_generated_exact_smf_variant",
                "descriptor": different,
            },
        ]
    }

    with pytest.raises(KnownStructureEvaluationV2Error, match="terminal tail"):
        oracle_attack_span(plan=plan, score=score, oracle_source=oracle)


def _run_git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_archived_method_manifest_uses_frozen_commit_after_head_advances(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _run_git(repository, "init", "-b", "main")
    _run_git(repository, "config", "user.email", "test@example.invalid")
    _run_git(repository, "config", "user.name", "Test")
    method = repository / "method.py"
    method.write_text("VALUE = 1\n", encoding="utf-8")
    _run_git(repository, "add", "method.py")
    _run_git(repository, "commit", "-m", "method")
    frozen_commit = _run_git(repository, "rev-parse", "HEAD")
    constants = {"window": [1, 2]}
    payload = {
        "schema_version": 1,
        "status": "frozen",
        "frozen_commit": frozen_commit,
        "files": {
            "method.py": hashlib.sha256(
                subprocess.run(
                    ["git", "show", f"{frozen_commit}:method.py"],
                    cwd=repository,
                    check=True,
                    capture_output=True,
                ).stdout
            ).hexdigest(),
        },
        "constants": constants,
        "constants_sha256": hashlib.sha256(
            json.dumps(
                constants, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
    }
    manifest = tmp_path / "method-manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    expected_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    method.write_text("VALUE = 2\n", encoding="utf-8")
    _run_git(repository, "add", "method.py")
    _run_git(repository, "commit", "-m", "later")

    assert (
        verify_archived_method_manifest(
            repository_root=repository,
            manifest_path=manifest,
            expected_manifest_sha256=expected_hash,
            expected_files=("method.py",),
        )
        == expected_hash
    )


def test_runner_rejects_fixed_input_change_before_creating_output(
    tmp_path: Path,
) -> None:
    inputs = {}
    for name in (
        "observed_files",
        "observed_manifest",
        "timing_files",
        "timing_manifest",
        "oracle_files",
        "oracle_manifest",
    ):
        path = tmp_path / f"{name}.json"
        path.write_text("{}", encoding="utf-8")
        inputs[name] = path
    output_dir = tmp_path / "output"

    with pytest.raises(KnownStructureEvaluationV2Error, match="fixed input"):
        run_known_structure_evaluation_v2(
            **inputs,
            fixtures_root=tmp_path,
            repository_root=tmp_path,
            method_manifest=tmp_path / "method.json",
            evaluator_manifest=tmp_path / "evaluator.json",
            output_dir=output_dir,
        )

    assert not output_dir.exists()
