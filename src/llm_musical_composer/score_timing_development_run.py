"""隔離した開発SMFへ固定ScoreTiming候補族を適用する。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llm_musical_composer.observed_structure_method import (
    METHOD_CONSTANTS,
    verify_method_manifest,
)
from llm_musical_composer.reference_decomposition import (
    build_observed_performance,
    load_observed_smf,
)
from llm_musical_composer.run_state import (
    atomic_write_bytes,
    atomic_write_json,
    sha256_file,
)
from llm_musical_composer.score_timing_hypothesis import (
    generate_score_timing_candidates,
)

PARETO_METRICS = (
    "mean_group_anchor_error_us",
    "in_sample_interpolation_mean_error_us",
    "vocabulary_size",
    "time_map_knot_count",
    "local_coordination_group_count",
    "normalized_serialization_bytes",
)
_SCORE_TIMING_CONSTANTS = METHOD_CONSTANTS["score_timing"]


class ScoreTimingDevelopmentRunError(ValueError):
    """開発実行の入力隔離または成果物契約に違反した。"""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScoreTimingDevelopmentRunError(f"unable to read JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ScoreTimingDevelopmentRunError(f"JSON root must be an object: {path}")
    return value


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")


def pareto_candidate_ids(
    records: Sequence[Mapping[str, Any]], *, metric_names: Sequence[str]
) -> set[str]:
    """全指標を小さくする向きで非劣候補IDを返す。"""
    complete = [record for record in records if record.get("search_status") == "complete"]
    result: set[str] = set()
    for candidate in complete:
        candidate_metrics = candidate.get("metrics")
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_metrics, Mapping) or not isinstance(candidate_id, str):
            raise ScoreTimingDevelopmentRunError("candidate Pareto record is invalid")
        dominated = False
        for other in complete:
            if other is candidate:
                continue
            other_metrics = other.get("metrics")
            if not isinstance(other_metrics, Mapping):
                raise ScoreTimingDevelopmentRunError("candidate Pareto metrics are invalid")
            try:
                weakly_better = all(
                    float(other_metrics[name]) <= float(candidate_metrics[name])
                    for name in metric_names
                )
                strictly_better = any(
                    float(other_metrics[name]) < float(candidate_metrics[name])
                    for name in metric_names
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ScoreTimingDevelopmentRunError(
                    "candidate Pareto metric is missing or non-numeric"
                ) from error
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            result.add(candidate_id)
    return result


def _verified_sources(
    staging_dir: Path, staging_manifest: Path
) -> tuple[dict[str, str], str, str, str, str | None]:
    manifest = _read_json(staging_manifest)
    schema_version = manifest.get("schema_version")
    if schema_version == 1:
        split_role = "development"
        source_key = "development_smf"
    elif schema_version == 2 and manifest.get("split_role") in {
        "development",
        "holdout",
        "known_fixture",
    }:
        split_role = str(manifest["split_role"])
        source_key = "staged_smf"
    else:
        raise ScoreTimingDevelopmentRunError("staging manifest contract is invalid")
    sources = manifest.get(source_key)
    if manifest.get("status") != "pass" or not isinstance(sources, Mapping):
        raise ScoreTimingDevelopmentRunError("staging manifest contract is invalid")
    manifest_inputs = manifest.get("inputs")
    staging_method_hash = (
        manifest_inputs.get("method_manifest")
        if isinstance(manifest_inputs, Mapping)
        else None
    )
    if split_role != "development" and not isinstance(staging_method_hash, str):
        raise ScoreTimingDevelopmentRunError(
            "staging method manifest SHA-256 is missing"
        )
    expected: dict[str, str] = {}
    for name, expected_hash in sources.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(expected_hash, str)
        ):
            raise ScoreTimingDevelopmentRunError("staging source record is invalid")
        expected[name] = expected_hash.lower()
    actual_names = {
        path.name
        for path in Path(staging_dir).iterdir()
        if path.is_file() and path.suffix.casefold() == ".mid"
    }
    if set(expected) != actual_names:
        raise ScoreTimingDevelopmentRunError(
            f"source set mismatch: expected={sorted(expected)}, actual={sorted(actual_names)}"
        )
    for name, expected_hash in expected.items():
        if sha256_file(Path(staging_dir) / name).lower() != expected_hash:
            raise ScoreTimingDevelopmentRunError(f"staged SMF SHA-256 mismatch: {name}")
    return (
        expected,
        sha256_file(staging_manifest),
        split_role,
        source_key,
        staging_method_hash,
    )


def run_score_timing_development(
    *,
    staging_dir: Path,
    staging_manifest: Path,
    output_dir: Path,
    repository_root: Path | None = None,
    method_manifest: Path | None = None,
) -> dict[str, Any]:
    """開発stagingだけを入力に候補生成とPareto診断を保存する。"""
    staging_dir = Path(staging_dir)
    staging_manifest = Path(staging_manifest)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ScoreTimingDevelopmentRunError(f"output directory is not empty: {output_dir}")
    (
        sources,
        manifest_hash,
        split_role,
        source_key,
        staging_method_hash,
    ) = _verified_sources(staging_dir, staging_manifest)
    method_manifest_hash: str | None = None
    if split_role != "development" and (
        repository_root is None or method_manifest is None
    ):
        raise ScoreTimingDevelopmentRunError(
            "method manifest is required for non-development staging"
        )
    if method_manifest is not None:
        if repository_root is None:
            raise ScoreTimingDevelopmentRunError(
                "repository root is required with method manifest"
            )
        method_manifest_hash = verify_method_manifest(
            repository_root=Path(repository_root),
            manifest_path=Path(method_manifest),
        )
        if staging_method_hash is not None and staging_method_hash != method_manifest_hash:
            raise ScoreTimingDevelopmentRunError(
                "staging method manifest SHA-256 mismatch"
            )
    records: list[dict[str, Any]] = []
    for name in sorted(sources, key=lambda value: (value.casefold(), value)):
        try:
            observed = load_observed_smf(staging_dir / name)
            performance = build_observed_performance(observed)
            candidate_set = generate_score_timing_candidates(
                performance,
                source_ledger_sha256=observed.ledger_sha256,
            )
            candidates = [asdict(candidate) for candidate in candidate_set.candidates]
            negative_controls = [
                asdict(control) for control in candidate_set.negative_controls
            ]
            controls_by_id = {
                control["control_id"]: control for control in negative_controls
            }
            for candidate in candidates:
                comparisons: dict[str, dict[str, bool]] = {}
                for control_id, control in controls_by_id.items():
                    candidate_error = float(
                        candidate["metrics"]["mean_group_anchor_error_us"]
                    )
                    control_error = float(
                        control["metrics"]["mean_group_anchor_error_us"]
                    )
                    candidate_size = float(
                        candidate["metrics"]["normalized_serialization_bytes"]
                    )
                    control_size = float(
                        control["metrics"]["normalized_serialization_bytes"]
                    )
                    comparisons[control_id] = {
                        "control_dominates_reconstruction_and_size": (
                            control_error <= candidate_error
                            and control_size <= candidate_size
                            and (
                                control_error < candidate_error
                                or control_size < candidate_size
                            )
                        ),
                        "candidate_improves_reconstruction": (
                            candidate_error < control_error
                        ),
                        "candidate_reduces_serialization": candidate_size < control_size,
                    }
                candidate["negative_control_comparison"] = comparisons
                candidate["viable_tradeoff"] = (
                    not comparisons["observed-copy-negative-control"][
                        "control_dominates_reconstruction_and_size"
                    ]
                    and comparisons["uniform-grid-baseline"][
                        "candidate_improves_reconstruction"
                    ]
                )
            pareto_ids = pareto_candidate_ids(candidates, metric_names=PARETO_METRICS)
            for candidate in candidates:
                candidate["pareto_nondominated"] = candidate["candidate_id"] in pareto_ids
            records.append(
                {
                    "name": name,
                    "status": candidate_set.status,
                    "reason": candidate_set.reason,
                    "source_sha256": observed.source_sha256,
                    "source_ledger_sha256": observed.ledger_sha256,
                    "note_matching_status": performance.note_matching_status,
                    "note_count": len(performance.notes),
                    "attack_group_count": len(performance.attack_groups),
                    "candidates": candidates,
                    "negative_controls": negative_controls,
                    "pareto_candidate_ids": sorted(pareto_ids),
                }
            )
        except (OSError, EOFError, ValueError) as error:
            records.append(
                {
                    "name": name,
                    "status": "unable_to_investigate",
                    "reason": str(error),
                    "candidates": [],
                    "negative_controls": [],
                    "pareto_candidate_ids": [],
                }
            )
    assessed_count = sum(record["status"] == "assessed" for record in records)
    candidate_count = sum(len(record["candidates"]) for record in records)
    negative_control_count = sum(len(record["negative_controls"]) for record in records)
    nonconverged_count = sum(
        candidate["search_status"] != "complete"
        for record in records
        for candidate in record["candidates"]
    )
    source_with_viable_tradeoff_count = sum(
        any(candidate["viable_tradeoff"] for candidate in record["candidates"])
        for record in records
    )
    status = "pass" if assessed_count == len(records) and not nonconverged_count else "partial"
    run_spec = {
        "schema_version": 1,
        "split_role": split_role,
        "candidate_family": {
            "grouping_profiles": list(_SCORE_TIMING_CONSTANTS["grouping_profiles"]),
            "score_grids": list(_SCORE_TIMING_CONSTANTS["score_grids"]),
            "time_map_segment_counts": list(
                _SCORE_TIMING_CONSTANTS["time_map_segment_counts"]
            ),
            "candidate_count_per_source": int(
                _SCORE_TIMING_CONSTANTS["candidate_count_per_source"]
            ),
        },
        "inputs": {
            "staging_manifest": manifest_hash,
            source_key: dict(sorted(sources.items())),
        },
        "pareto_metrics": list(PARETO_METRICS),
        "viable_tradeoff_rule": (
            "not dominated by observed-copy on mean reconstruction and serialization; "
            "improves mean reconstruction over uniform-grid"
        ),
    }
    if method_manifest_hash is not None:
        run_spec["inputs"]["method_manifest"] = method_manifest_hash
    summary = {
        "schema_version": 1,
        "status": status,
        "split_role": split_role,
        "source_count": len(records),
        "assessed_count": assessed_count,
        "candidate_count": candidate_count,
        "negative_control_count": negative_control_count,
        "nonconverged_count": nonconverged_count,
        "source_with_viable_tradeoff_count": source_with_viable_tradeoff_count,
        "sources": [
            {
                "name": record["name"],
                "status": record["status"],
                "candidate_count": len(record["candidates"]),
                "pareto_candidate_count": len(record["pareto_candidate_ids"]),
            }
            for record in records
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "run-spec.json", run_spec)
    atomic_write_bytes(output_dir / "files.jsonl", _jsonl_bytes(records))
    atomic_write_json(output_dir / "summary.json", summary)
    output_names = ("files.jsonl", "run-spec.json", "summary.json")
    manifest = {
        "schema_version": 1,
        "status": status,
        "split_role": split_role,
        "inputs": {"staging_manifest": manifest_hash},
        "outputs": {name: sha256_file(output_dir / name) for name in output_names},
    }
    if method_manifest_hash is not None:
        manifest["inputs"]["method_manifest"] = method_manifest_hash
    atomic_write_json(output_dir / "manifest.json", manifest)
    return {
        "status": status,
        "source_count": len(records),
        "candidate_count": candidate_count,
        "negative_control_count": negative_control_count,
        "nonconverged_count": nonconverged_count,
        "source_with_viable_tradeoff_count": source_with_viable_tradeoff_count,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--staging-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--method-manifest", type=Path)
    arguments = parser.parse_args(argv)
    result = run_score_timing_development(
        staging_dir=arguments.staging_dir,
        staging_manifest=arguments.staging_manifest,
        output_dir=arguments.output_dir,
        repository_root=arguments.repository_root,
        method_manifest=arguments.method_manifest,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
