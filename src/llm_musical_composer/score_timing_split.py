"""観測特徴だけでScoreTiming開発標本を固定し、開発用SMFだけを隔離する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from llm_musical_composer.observed_structure_method import (
    METHOD_CONSTANTS,
    verify_method_manifest,
)
from llm_musical_composer.reference_profile import (
    FEATURE_GROUPS,
    profile_distances,
    select_default_reference,
    select_local_neighborhood,
)
from llm_musical_composer.run_state import atomic_write_bytes, atomic_write_json, sha256_file


class ScoreTimingSplitError(ValueError):
    """開発標本の入力契約または隔離に違反した。"""


def _name_key(name: str) -> tuple[str, str]:
    return name.casefold(), name


def _profile_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    profiles: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        name = record.get("name")
        profile = record.get("profile")
        if not isinstance(name, str) or Path(name).name != name:
            raise ScoreTimingSplitError(f"profile record {index} name is invalid")
        if name in profiles:
            raise ScoreTimingSplitError(f"duplicate profile name: {name}")
        if record.get("status") != "pass" or not isinstance(profile, Mapping):
            raise ScoreTimingSplitError(f"profile record is not usable: {name}")
        profiles[name] = profile
    return profiles


def _fingerprint_groups(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        fingerprint = record.get("copy_fingerprint")
        value = fingerprint.get("sequence_sha256") if isinstance(fingerprint, Mapping) else None
        if not isinstance(value, str) or not value:
            raise ScoreTimingSplitError(f"copy fingerprint is invalid: {record.get('name')}")
        grouped[value].append(str(record["name"]))
    return {
        name: tuple(sorted(grouped[fingerprint], key=_name_key))
        for fingerprint, names in grouped.items()
        for name in names
    }


def select_development_records(
    records: Sequence[Mapping[str, Any]], *, count: int = 8
) -> list[dict[str, Any]]:
    """実在medoidから三特徴群の未被覆距離を広げて開発曲を選ぶ。"""
    profiles = _profile_records(records)
    if count < 1 or count > len(profiles):
        raise ScoreTimingSplitError(f"cannot select {count} development records")
    conflicts = _fingerprint_groups(records)
    medoid = str(select_default_reference(profiles)["selected"]["name"])
    selected = [medoid]
    blocked = set(conflicts[medoid]) - {medoid}
    diagnostics: dict[str, dict[str, float]] = {medoid: {group: 0.0 for group in FEATURE_GROUPS}}
    while len(selected) < count:
        candidates = [name for name in profiles if name not in selected and name not in blocked]
        if not candidates:
            raise ScoreTimingSplitError(
                f"cannot select {count} records without exact-copy conflicts"
            )

        def coverage(name: str) -> dict[str, float]:
            pair_distances = [
                profile_distances(profiles[name], profiles[chosen]) for chosen in selected
            ]
            return {
                group: min(distance[group] for distance in pair_distances)
                for group in FEATURE_GROUPS
            }

        candidate_coverage = {name: coverage(name) for name in candidates}

        def selection_key(
            name: str,
            coverage: dict[str, dict[str, float]] = candidate_coverage,
        ) -> tuple[float, float, float, str, str]:
            values = tuple(coverage[name][group] for group in FEATURE_GROUPS)
            return (
                -min(values),
                -statistics.median(values),
                -max(values),
                name.casefold(),
                name,
            )

        chosen = min(candidates, key=selection_key)
        selected.append(chosen)
        diagnostics[chosen] = candidate_coverage[chosen]
        blocked.update(set(conflicts[chosen]) - {chosen})
    return [
        {
            "name": name,
            "selection_index": index,
            "selection_role": "real_medoid" if index == 0 else "farthest_first",
            "minimum_group_distances_to_prior": diagnostics[name],
            "exact_copy_conflicts": [item for item in conflicts[name] if item != name],
        }
        for index, name in enumerate(selected)
    ]


def select_holdout_records(
    records: Sequence[Mapping[str, Any]], *, development_names: Sequence[str]
) -> list[dict[str, Any]]:
    """各開発曲に最も近い未使用・非同一コピーの保留曲を一つ選ぶ。"""
    profiles = _profile_records(records)
    conflicts = _fingerprint_groups(records)
    development = list(development_names)
    if len(set(development)) != len(development):
        raise ScoreTimingSplitError("development names must be unique")
    missing = [name for name in development if name not in profiles]
    if missing:
        raise ScoreTimingSplitError(f"unknown development record: {missing[0]}")
    if len(profiles) < 3:
        raise ScoreTimingSplitError("at least three profiles are required")

    blocked = set(development)
    for name in development:
        blocked.update(conflicts[name])
    selected: list[dict[str, Any]] = []
    for index, anchor in enumerate(development):
        neighborhood = select_local_neighborhood(
            anchor,
            profiles,
            neighbor_count=len(profiles),
        )
        peers = [
            item
            for item in neighborhood["neighbors"]
            if item["role"] == "neighbor" and item["name"] not in blocked
        ]
        if not peers:
            raise ScoreTimingSplitError(
                f"no unused holdout peer is available for development anchor: {anchor}"
            )
        chosen = peers[0]
        name = str(chosen["name"])
        blocked.update(conflicts[name])
        selected.append(
            {
                "name": name,
                "selection_index": index,
                "selection_role": "nearest_unused_peer",
                "anchor_name": anchor,
                "group_distances": chosen["group_distances"],
                "group_ranks": chosen["group_ranks"],
                "exact_copy_conflicts": [
                    item for item in conflicts[name] if item != name
                ],
            }
        )
    return selected


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ScoreTimingSplitError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    if not all(isinstance(record, dict) for record in records):
        raise ScoreTimingSplitError(f"JSONL records must be objects: {path}")
    return records


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")


def _semantic_input_hash(records: Sequence[Mapping[str, Any]]) -> str:
    normalized = [
        {
            "name": record["name"],
            "profile": record["profile"],
            "copy_fingerprint": {"sequence_sha256": record["copy_fingerprint"]["sequence_sha256"]},
        }
        for record in sorted(records, key=lambda item: _name_key(str(item["name"])))
    ]
    payload = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_development_split(
    *,
    source_dir: Path,
    profile_manifest: Path,
    profile_records: Path,
    profile_summary: Path,
    output_dir: Path,
    count: int = 8,
) -> dict[str, Any]:
    """検証済み参照から開発曲だけを選び、専用stagingへ複製する。"""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ScoreTimingSplitError(f"output directory is not empty: {output_dir}")
    manifest = _read_json(Path(profile_manifest))
    summary = _read_json(Path(profile_summary))
    records = _read_jsonl(Path(profile_records))
    outputs = manifest.get("outputs")
    inputs = manifest.get("inputs")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "pass"
        or not isinstance(outputs, Mapping)
        or not isinstance(inputs, Mapping)
    ):
        raise ScoreTimingSplitError("profile manifest contract is invalid")
    for path, key in (
        (Path(profile_records), "files.jsonl"),
        (Path(profile_summary), "summary.json"),
    ):
        expected = outputs.get(key)
        if not isinstance(expected, str) or sha256_file(path).lower() != expected.lower():
            raise ScoreTimingSplitError(f"profile artifact SHA-256 mismatch: {key}")
    if summary.get("status") != "pass" or summary.get("source_count") != len(records):
        raise ScoreTimingSplitError("profile summary source count is invalid")
    names = {str(record.get("name")) for record in records}
    if set(inputs) != names:
        raise ScoreTimingSplitError("profile manifest source set differs from records")
    for name in sorted(names, key=_name_key):
        path = source_dir / name
        expected = inputs[name]
        if (
            not path.is_file()
            or not isinstance(expected, str)
            or sha256_file(path).lower() != expected.lower()
        ):
            raise ScoreTimingSplitError(f"source SHA-256 mismatch: {name}")

    selected = select_development_records(records, count=count)
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / "development-smf"
    staging.mkdir()
    selected_records = []
    by_name = {str(record["name"]): record for record in records}
    for selection in selected:
        name = selection["name"]
        destination = staging / name
        shutil.copy2(source_dir / name, destination)
        selected_records.append(
            {
                **selection,
                "source_sha256": inputs[name],
                "staged_sha256": sha256_file(destination),
                "profile": by_name[name]["profile"],
            }
        )
    run_spec = {
        "schema_version": 1,
        "selection_version": "score-timing-development-farthest-v1",
        "development_count": count,
        "source_count": len(records),
        "semantic_profile_set_sha256": _semantic_input_hash(records),
        "inputs": {
            "profile_manifest": sha256_file(Path(profile_manifest)),
            "profile_records": sha256_file(Path(profile_records)),
            "profile_summary": sha256_file(Path(profile_summary)),
        },
    }
    selection = {
        "schema_version": 1,
        "selection_method": ("real medoid, then maximize minimum/median/maximum group distance"),
        "development": selected,
    }
    holdout_rule = {
        "schema_version": 1,
        "materialized": False,
        "count": count,
        "selection_method": (
            "after method freeze, choose one unused local-neighborhood peer per development anchor"
        ),
    }
    coverage = {
        "schema_version": 1,
        "development_count": count,
        "records": [
            {
                "name": item["name"],
                "minimum_group_distances_to_prior": item["minimum_group_distances_to_prior"],
            }
            for item in selected
        ],
    }
    atomic_write_json(output_dir / "run-spec.json", run_spec)
    atomic_write_json(output_dir / "selection.json", selection)
    atomic_write_bytes(output_dir / "development.jsonl", _jsonl_bytes(selected_records))
    atomic_write_json(output_dir / "holdout-rule.json", holdout_rule)
    atomic_write_json(output_dir / "coverage.json", coverage)
    output_names = (
        "coverage.json",
        "development.jsonl",
        "holdout-rule.json",
        "run-spec.json",
        "selection.json",
    )
    artifact_manifest = {
        "schema_version": 1,
        "status": "pass",
        "outputs": {name: sha256_file(output_dir / name) for name in output_names},
        "development_smf": {item["name"]: item["staged_sha256"] for item in selected_records},
    }
    atomic_write_json(output_dir / "manifest.json", artifact_manifest)
    return {
        "status": "pass",
        "source_count": len(records),
        "development_count": count,
        "holdout_materialized": False,
    }


def write_holdout_split(
    *,
    source_dir: Path,
    profile_manifest: Path,
    profile_records: Path,
    profile_summary: Path,
    development_selection: Path,
    repository_root: Path,
    method_manifest: Path,
    output_dir: Path,
    expected_count: int = int(METHOD_CONSTANTS["holdout_count"]),
) -> dict[str, Any]:
    """凍結後に各開発曲の近傍から保留曲を一度だけ物質化する。"""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ScoreTimingSplitError(f"output directory is not empty: {output_dir}")
    method_hash = verify_method_manifest(
        repository_root=Path(repository_root),
        manifest_path=Path(method_manifest),
    )
    manifest = _read_json(Path(profile_manifest))
    summary = _read_json(Path(profile_summary))
    records = _read_jsonl(Path(profile_records))
    outputs = manifest.get("outputs")
    inputs = manifest.get("inputs")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "pass"
        or not isinstance(outputs, Mapping)
        or not isinstance(inputs, Mapping)
    ):
        raise ScoreTimingSplitError("profile manifest contract is invalid")
    for path, key in (
        (Path(profile_records), "files.jsonl"),
        (Path(profile_summary), "summary.json"),
    ):
        expected = outputs.get(key)
        if not isinstance(expected, str) or sha256_file(path).lower() != expected.lower():
            raise ScoreTimingSplitError(f"profile artifact SHA-256 mismatch: {key}")
    if summary.get("status") != "pass" or summary.get("source_count") != len(records):
        raise ScoreTimingSplitError("profile summary source count is invalid")
    names = {str(record.get("name")) for record in records}
    if set(inputs) != names:
        raise ScoreTimingSplitError("profile manifest source set differs from records")
    for name in sorted(names, key=_name_key):
        path = source_dir / name
        expected = inputs[name]
        if (
            not path.is_file()
            or not isinstance(expected, str)
            or sha256_file(path).lower() != expected.lower()
        ):
            raise ScoreTimingSplitError(f"source SHA-256 mismatch: {name}")

    development_payload = _read_json(Path(development_selection))
    development_records = development_payload.get("development")
    if development_payload.get("schema_version") != 1 or not isinstance(
        development_records, list
    ):
        raise ScoreTimingSplitError("development selection contract is invalid")
    development_names = []
    for record in development_records:
        name = record.get("name") if isinstance(record, Mapping) else None
        if not isinstance(name, str):
            raise ScoreTimingSplitError("development selection name is invalid")
        development_names.append(name)
    if len(development_names) != expected_count:
        raise ScoreTimingSplitError(
            f"development selection count differs from frozen count: "
            f"expected={expected_count}, actual={len(development_names)}"
        )
    selected = select_holdout_records(records, development_names=development_names)

    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / "staged-smf"
    staging.mkdir()
    by_name = {str(record["name"]): record for record in records}
    selected_records = []
    for selection in selected:
        name = selection["name"]
        destination = staging / name
        shutil.copy2(source_dir / name, destination)
        selected_records.append(
            {
                **selection,
                "source_sha256": inputs[name],
                "staged_sha256": sha256_file(destination),
                "profile": by_name[name]["profile"],
            }
        )
    run_spec = {
        "schema_version": 2,
        "split_role": "holdout",
        "selection_version": "score-timing-holdout-nearest-unused-v1",
        "source_count": len(records),
        "holdout_count": len(selected_records),
        "semantic_profile_set_sha256": _semantic_input_hash(records),
        "inputs": {
            "profile_manifest": sha256_file(Path(profile_manifest)),
            "profile_records": sha256_file(Path(profile_records)),
            "profile_summary": sha256_file(Path(profile_summary)),
            "development_selection": sha256_file(Path(development_selection)),
            "method_manifest": method_hash,
        },
    }
    selection_payload = {
        "schema_version": 2,
        "split_role": "holdout",
        "materialized": True,
        "selection_method": (
            "one nearest unused non-copy peer per development anchor; "
            "minimize worst group rank, median group rank, then name"
        ),
        "holdout": selected,
    }
    atomic_write_json(output_dir / "run-spec.json", run_spec)
    atomic_write_json(output_dir / "selection.json", selection_payload)
    atomic_write_bytes(output_dir / "holdout.jsonl", _jsonl_bytes(selected_records))
    output_names = ("holdout.jsonl", "run-spec.json", "selection.json")
    artifact_manifest = {
        "schema_version": 2,
        "status": "pass",
        "split_role": "holdout",
        "inputs": {"method_manifest": method_hash},
        "outputs": {name: sha256_file(output_dir / name) for name in output_names},
        "staged_smf": {
            item["name"]: item["staged_sha256"] for item in selected_records
        },
    }
    atomic_write_json(output_dir / "manifest.json", artifact_manifest)
    return {
        "status": "pass",
        "source_count": len(records),
        "holdout_count": len(selected_records),
        "holdout_materialized": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """開発分割をCLIから再現する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--profile-manifest", type=Path, required=True)
    parser.add_argument("--profile-records", type=Path, required=True)
    parser.add_argument("--profile-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    arguments = parser.parse_args(argv)
    result = write_development_split(
        source_dir=arguments.source_dir,
        profile_manifest=arguments.profile_manifest,
        profile_records=arguments.profile_records,
        profile_summary=arguments.profile_summary,
        output_dir=arguments.output_dir,
        count=arguments.count,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
