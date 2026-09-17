"""保存済み窓特徴の完全順列で、構造境界の固定窓依存を調べる。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import statistics
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from llm_musical_composer.reference_structure import _persistent_boundaries
from llm_musical_composer.run_state import atomic_write_bytes, atomic_write_json, sha256_file
from llm_musical_composer.structure_features import _novelty

IDENTITY_PERMUTATION = (0, 1, 2, 3)
QUARTILE_POSITIONS = (0.25, 0.5, 0.75)
POSITION_BIN_COUNT = 32


def _round(value: float) -> float:
    return round(float(value), 8)


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return _round(ordered[0])
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return _round(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"available_count": 0, "status": "unable_to_investigate"}
    return {
        "available_count": len(values),
        "p25": _percentile(values, 0.25),
        "median": _percentile(values, 0.5),
        "p75": _percentile(values, 0.75),
    }


def block_permutations() -> tuple[tuple[int, int, int, int], ...]:
    """恒等配置を先頭にした四ブロックの全順列を返す。"""
    permutations = tuple(itertools.permutations(range(4)))
    return (IDENTITY_PERMUTATION, *(item for item in permutations if item != IDENTITY_PERMUTATION))


def _permuted_windows(
    windows: Sequence[dict[str, Any]], permutation: Sequence[int]
) -> list[dict[str, Any]]:
    if tuple(sorted(permutation)) != IDENTITY_PERMUTATION:
        raise ValueError("permutation must contain each block index from zero to three")
    if len(windows) < 4 or len(windows) % 4:
        raise ValueError("window count must be a positive multiple of four")
    block_size = len(windows) // 4
    source_features = [copy.deepcopy(window["features"]) for window in windows]
    reordered_features = [
        feature
        for source_block in permutation
        for feature in source_features[source_block * block_size : (source_block + 1) * block_size]
    ]
    result = copy.deepcopy(list(windows))
    for window, features in zip(result, reordered_features, strict=True):
        window["features"] = features
    return result


def permute_coordinates(coordinates: dict[str, Any], permutation: Sequence[int]) -> dict[str, Any]:
    """全座標と全解像度へ同じブロック順列を適用し、境界を再計算する。"""
    changed = copy.deepcopy(coordinates)
    for coordinate in changed.values():
        for report in coordinate.get("resolutions", {}).values():
            if report.get("status") != "pass":
                continue
            windows = _permuted_windows(report.get("windows", []), permutation)
            report["windows"] = windows
            report["novelty"] = _novelty(windows)
    return {
        "coordinates": changed,
        "persistent_boundaries": _persistent_boundaries(changed),
    }


def boundary_statistics(boundaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """持続境界の支持量と固定格子への集中を別々に数える。"""
    evidence_count = sum(len(boundary["evidence"]) for boundary in boundaries)
    histogram = [0] * (POSITION_BIN_COUNT + 1)
    quartile_count = 0
    for boundary in boundaries:
        position = float(boundary["position"])
        bin_index = max(0, min(POSITION_BIN_COUNT, round(position * POSITION_BIN_COUNT)))
        histogram[bin_index] += 1
        near_quartile = any(
            abs(position - quartile) <= 1 / POSITION_BIN_COUNT for quartile in QUARTILE_POSITIONS
        )
        if near_quartile:
            quartile_count += 1
    boundary_count = len(boundaries)
    return {
        "boundary_count": boundary_count,
        "evidence_count": evidence_count,
        "extra_support": evidence_count - boundary_count,
        "dual_coordinate_boundary_count": sum(
            int(boundary["coordinate_count"] == 2) for boundary in boundaries
        ),
        "multi_resolution_boundary_count": sum(
            int(boundary["resolution_count"] >= 2) for boundary in boundaries
        ),
        "position_histogram_1_over_32": histogram,
        "quartile_concentration_count": quartile_count,
        "quartile_concentration_rate": _round(quartile_count / boundary_count)
        if boundary_count
        else None,
    }


def _boundary_signature(boundaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """JSONのキー順や支持指標の列挙順を除いた境界の意味を返す。"""
    return [
        {
            "position": boundary["position"],
            "resolution_count": boundary["resolution_count"],
            "resolutions": sorted(boundary["resolutions"]),
            "coordinate_count": boundary["coordinate_count"],
            "coordinates": sorted(boundary["coordinates"]),
            "evidence": sorted(
                (
                    {
                        "coordinate": evidence["coordinate"],
                        "resolution": evidence["resolution"],
                        "position": evidence["position"],
                        "supporting_metrics": sorted(evidence["supporting_metrics"]),
                    }
                    for evidence in boundary["evidence"]
                ),
                key=lambda evidence: (
                    evidence["position"],
                    evidence["coordinate"],
                    evidence["resolution"],
                ),
            ),
        }
        for boundary in boundaries
    ]


def evaluate_record(record: dict[str, Any]) -> dict[str, Any]:
    """一曲の実配置と二十三帰無配置を同じ手順で評価する。"""
    base = {"name": record.get("name"), "status": record.get("status")}
    if record.get("status") != "pass":
        return {**base, "error": copy.deepcopy(record.get("error"))}
    coordinates = record.get("coordinates")
    if not isinstance(coordinates, dict) or not coordinates:
        return {
            "name": record.get("name"),
            "status": "unable_to_investigate",
            "error": {"type": "ValueError", "message": "coordinates are unavailable"},
        }
    available = sum(
        report.get("status") == "pass"
        for coordinate in coordinates.values()
        for report in coordinate.get("resolutions", {}).values()
    )
    unavailable = sum(
        report.get("status") != "pass"
        for coordinate in coordinates.values()
        for report in coordinate.get("resolutions", {}).values()
    )
    try:
        permutation_results = []
        for permutation in block_permutations():
            changed = permute_coordinates(coordinates, permutation)
            boundaries = changed["persistent_boundaries"]
            permutation_results.append(
                {
                    "permutation": list(permutation),
                    "available_resolution_count": available,
                    "unavailable_resolution_count": unavailable,
                    "persistent_boundaries": boundaries,
                    "statistics": boundary_statistics(boundaries),
                }
            )
    except (KeyError, TypeError, ValueError) as error:
        return {
            "name": record.get("name"),
            "status": "unable_to_investigate",
            "error": {"type": type(error).__name__, "message": str(error)},
        }
    saved_boundaries = record.get("persistent_boundaries")
    current_boundaries = permutation_results[0]["persistent_boundaries"]
    saved_signature = (
        _boundary_signature(saved_boundaries) if isinstance(saved_boundaries, list) else None
    )
    current_signature = _boundary_signature(current_boundaries)
    identity_matches_saved = (
        isinstance(saved_boundaries, list) and current_signature == saved_signature
    )
    if not identity_matches_saved:
        saved_bytes = json.dumps(saved_signature, sort_keys=True, separators=(",", ":")).encode()
        current_bytes = json.dumps(
            current_signature, sort_keys=True, separators=(",", ":")
        ).encode()
        return {
            "name": record.get("name"),
            "status": "unable_to_investigate",
            "error": {
                "type": "IdentityMismatch",
                "message": (
                    "identity permutation does not reproduce saved persistent boundaries; "
                    "regenerate the reference analysis with the current implementation"
                ),
                "saved_count": len(saved_boundaries)
                if isinstance(saved_boundaries, list)
                else None,
                "current_count": len(current_boundaries),
                "saved_sha256": hashlib.sha256(saved_bytes).hexdigest(),
                "current_sha256": hashlib.sha256(current_bytes).hexdigest(),
                "saved_positions": [boundary.get("position") for boundary in saved_boundaries]
                if isinstance(saved_boundaries, list)
                else None,
                "current_positions": [boundary.get("position") for boundary in current_boundaries],
            },
        }
    return {
        **base,
        "identity_matches_saved": True,
        "available_resolution_count": available,
        "unavailable_resolution_count": unavailable,
        "permutations": permutation_results,
    }


def build_null_control_summary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """曲別順列結果を集計し、事前に固定した停止条件を判定する。"""
    records = list(records)
    passed = [record for record in records if record.get("status") == "pass"]
    if not passed:
        return {
            "schema_version": 1,
            "source_count": len(records),
            "pass_count": 0,
            "unable_to_investigate_count": len(records),
            "status": "unable_to_investigate",
            "decision": "unable_to_investigate",
            "boundary_use": "diagnostic_only",
            "generation_boundary_target": None,
        }
    expected_count = len(block_permutations())
    if any(len(record.get("permutations", [])) != expected_count for record in passed):
        raise ValueError("each passed record must contain all 24 block permutations")
    corpus_medians = [
        _round(
            statistics.median(
                record["permutations"][index]["statistics"]["extra_support"] for record in passed
            )
        )
        for index in range(expected_count)
    ]
    paired_differences = [
        _round(
            record["permutations"][0]["statistics"]["extra_support"]
            - statistics.median(
                item["statistics"]["extra_support"] for item in record["permutations"][1:]
            )
        )
        for record in passed
    ]
    identity_median = corpus_medians[0]
    exact_p = _round(
        sum(value >= identity_median for value in corpus_medians) / len(corpus_medians)
    )
    identity_exceeds_all_nulls = all(identity_median > value for value in corpus_medians[1:])
    paired_distribution = _distribution(paired_differences)
    material_difference = paired_distribution["median"] >= 1
    passed_stage_one = identity_exceeds_all_nulls and material_difference
    return {
        "schema_version": 1,
        "source_count": len(records),
        "pass_count": len(passed),
        "unable_to_investigate_count": len(records) - len(passed),
        "status": "pass",
        "method": "four_macro_block_complete_permutation",
        "permutation_count": expected_count,
        "null_permutation_count": expected_count - 1,
        "primary_metric": "extra_support",
        "corpus_extra_support_median_by_permutation": corpus_medians,
        "identity_corpus_median": identity_median,
        "identity_exceeds_all_nulls": identity_exceeds_all_nulls,
        "exact_permutation_p": exact_p,
        "paired_extra_support_difference": paired_distribution,
        "minimum_material_paired_median": 1,
        "decision": "pass_stage_one" if passed_stage_one else "fail",
        "boundary_use": "diagnostic_only",
        "generation_boundary_target": None,
        "next_step": (
            "phase_control"
            if passed_stage_one
            else "long_form_generation_without_inferred_boundaries"
        ),
        "claim_limit": (
            "Passing only shows that fixed-grid macro-block exchangeability does not fully "
            "explain cross-resolution and cross-coordinate support. It does not establish "
            "perceptual boundaries."
            if passed_stage_one
            else "The current persistent boundaries must not determine section count, boundary "
            "positions, or hierarchy depth in generation."
        ),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at line {line_number}: {error.msg}") from error
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record at line {line_number} is not an object")
        records.append(value)
    return records


def _jsonl_bytes(records: Sequence[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")


def write_structure_null_controls(records_path: Path, output_dir: Path) -> dict[str, Any]:
    """曲別結果、集計、再現用ハッシュを確定書き込みする。"""
    records_path = Path(records_path)
    output_dir = Path(output_dir)
    evaluated = [evaluate_record(record) for record in _read_jsonl(records_path)]
    summary = build_null_control_summary(evaluated)
    files_path = output_dir / "files.jsonl"
    summary_path = output_dir / "summary.json"
    atomic_write_bytes(files_path, _jsonl_bytes(evaluated))
    atomic_write_json(summary_path, summary)
    configuration = {
        "block_count": 4,
        "permutation_count": len(block_permutations()),
        "primary_metric": "extra_support",
        "minimum_material_paired_median": 1,
    }
    manifest = {
        "schema_version": 1,
        "configuration": configuration,
        "configuration_sha256": hashlib.sha256(
            json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "sha256": {
            "input_jsonl": sha256_file(records_path),
            "implementation": sha256_file(Path(__file__)),
            "files_jsonl": sha256_file(files_path),
            "summary_json": sha256_file(summary_path),
        },
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="参照曲の構造境界を完全順列で反証します。")
    parser.add_argument(
        "--records",
        type=Path,
        default=Path(".appendix/reference-structure-analysis/files.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".appendix/structure-null-controls"),
    )
    args = parser.parse_args(argv)
    summary = write_structure_null_controls(args.records, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
