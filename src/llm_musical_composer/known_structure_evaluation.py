"""凍結済み観測候補を既知構造の候補集合と事後比較する。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from llm_musical_composer.observed_structure_method import verify_method_manifest
from llm_musical_composer.run_state import atomic_write_bytes, atomic_write_json, sha256_file


class KnownStructureEvaluationError(ValueError):
    """観測結果と既知構造候補の比較契約に違反した。"""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise KnownStructureEvaluationError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not all(isinstance(value, dict) for value in values):
        raise KnownStructureEvaluationError(f"JSONL records must be objects: {path}")
    return values


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        + b"\n"
        for record in records
    )


def _fraction(value: Sequence[object]) -> Fraction:
    if len(value) != 2:
        raise KnownStructureEvaluationError("fraction must have two integers")
    numerator, denominator = int(value[0]), int(value[1])
    if denominator == 0:
        raise KnownStructureEvaluationError("fraction denominator must not be zero")
    return Fraction(numerator, denominator)


def _fraction_list(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _oracle_boundaries(
    oracle_source: Mapping[str, Any],
) -> tuple[set[Fraction], set[Fraction]]:
    candidate_sets: list[set[Fraction]] = []
    candidates = oracle_source.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise KnownStructureEvaluationError("oracle candidates are missing")
    for candidate in candidates:
        descriptor = candidate.get("descriptor") if isinstance(candidate, Mapping) else None
        if not isinstance(descriptor, Mapping):
            raise KnownStructureEvaluationError("oracle descriptor is invalid")
        boundaries: set[Fraction] = set()
        nodes = descriptor.get("nodes")
        materials = descriptor.get("materials")
        if not isinstance(nodes, list) or not isinstance(materials, list):
            raise KnownStructureEvaluationError("oracle structure lists are invalid")
        spans = []
        spans.extend(nodes)
        for material in materials:
            if not isinstance(material, Mapping) or not isinstance(
                material.get("occurrence_spans"), list
            ):
                raise KnownStructureEvaluationError("oracle material spans are invalid")
            spans.extend(material["occurrence_spans"])
        for span in spans:
            if not isinstance(span, Mapping):
                raise KnownStructureEvaluationError("oracle span is invalid")
            for key in ("start", "end"):
                value = span.get(key)
                if not isinstance(value, list):
                    raise KnownStructureEvaluationError("oracle span endpoint is invalid")
                boundary = _fraction(value)
                if boundary not in {Fraction(0), Fraction(1)}:
                    boundaries.add(boundary)
        candidate_sets.append(boundaries)
    common = set.intersection(*candidate_sets)
    allowed = set.union(*candidate_sets)
    return common, allowed


def _oracle_material_annotations(
    oracle_source: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result = []
    for candidate in oracle_source.get("candidates", []):
        descriptor = candidate.get("descriptor", {})
        materials = []
        for material in descriptor.get("materials", []):
            materials.append(
                {
                    "local_id": material.get("local_id"),
                    "occurrence_spans": material.get("occurrence_spans", []),
                }
            )
        result.append(
            {
                "oracle_candidate_id": candidate.get("candidate_id"),
                "materials": materials,
                "derived_relations": descriptor.get("derived_relations", []),
            }
        )
    return result


def _edge_fraction(
    edge: int, *, positions: Sequence[int], group_count: int
) -> Fraction | None:
    if group_count < 2 or len(positions) != group_count:
        return None
    if edge == 0:
        return Fraction(0)
    if edge == group_count:
        return Fraction(1)
    if edge < 0 or edge >= group_count:
        return None
    first = int(positions[0])
    last = int(positions[-1])
    if last <= first:
        return None
    return Fraction(int(positions[edge]) - first, last - first)


def evaluate_known_source(
    *,
    observed_source: Mapping[str, Any],
    timing_source: Mapping[str, Any],
    oracle_source: Mapping[str, Any],
) -> dict[str, Any]:
    """一つのfixtureをalias別に候補集合と比較する。"""
    if observed_source.get("schema_version") != 2:
        raise KnownStructureEvaluationError("observed source schema is invalid")
    if observed_source.get("name") != timing_source.get("name"):
        raise KnownStructureEvaluationError("observed and timing names differ")
    if observed_source.get("source_sha256") != timing_source.get("source_sha256"):
        raise KnownStructureEvaluationError("observed and timing source hashes differ")
    fixture_id = str(oracle_source.get("fixture_id", ""))
    if observed_source.get("name") != f"{fixture_id}.mid":
        raise KnownStructureEvaluationError("oracle fixture ID differs from source name")
    common, allowed = _oracle_boundaries(oracle_source)
    profiles = {
        str(profile.get("semantic_profile_hash")): profile
        for profile in observed_source.get("semantic_profiles", [])
        if isinstance(profile, Mapping)
    }
    lower = {
        str(candidate.get("candidate_id")): candidate
        for candidate in timing_source.get("candidates", [])
        if isinstance(candidate, Mapping)
    }
    dependency = oracle_source.get("score_timing_dependency")
    aligned_ids = (
        set(dependency.get("candidate_ids", []))
        if isinstance(dependency, Mapping)
        else set()
    )
    aliases = []
    for alias in observed_source.get("score_timing_aliases", []):
        if not isinstance(alias, Mapping):
            raise KnownStructureEvaluationError("observed alias is invalid")
        profile = profiles.get(str(alias.get("semantic_profile_hash")))
        candidate_ids = [str(value) for value in alias.get("candidate_ids", [])]
        lower_candidates = [lower.get(candidate_id) for candidate_id in candidate_ids]
        positions_sets = []
        for candidate in lower_candidates:
            refs = candidate.get("attack_group_refs") if isinstance(candidate, Mapping) else None
            if not isinstance(refs, list):
                continue
            positions_sets.append(tuple(int(ref["score_position"]) for ref in refs))
        if (
            not isinstance(profile, Mapping)
            or not positions_sets
            or len(set(positions_sets)) != 1
        ):
            aliases.append(
                {
                    "alias_id": alias.get("alias_id"),
                    "mapping_status": "unable_to_compare",
                    "reason": "profile or a unique score-position mapping is unavailable",
                }
            )
            continue
        positions = positions_sets[0]
        group_count = int(profile.get("group_count", -1))
        boundary_payload = profile.get("boundary")
        recurrence_payload = profile.get("recurrence")
        if not isinstance(boundary_payload, Mapping) or not isinstance(
            recurrence_payload, Mapping
        ):
            raise KnownStructureEvaluationError("observed profile payload is invalid")
        observed_boundaries = []
        unable = False
        for boundary in boundary_payload.get("candidates", []):
            edge = int(boundary["start_group_index"])
            normalized = _edge_fraction(edge, positions=positions, group_count=group_count)
            if normalized is None:
                unable = True
                break
            relation = (
                "common_boundary"
                if normalized in common
                else "allowed_boundary"
                if normalized in allowed
                else "outside_enumerated_candidates"
            )
            observed_boundaries.append(
                {"position": _fraction_list(normalized), "oracle_relation": relation}
            )
        recurrence_records = []
        for recurrence in recurrence_payload.get("candidates", []):
            spans = []
            size = int(recurrence["window_size"])
            for occurrence in recurrence["occurrences"]:
                start = int(occurrence[0])
                left = _edge_fraction(start, positions=positions, group_count=group_count)
                right = _edge_fraction(
                    start + size, positions=positions, group_count=group_count
                )
                if left is None or right is None:
                    unable = True
                    break
                spans.append([_fraction_list(left), _fraction_list(right)])
            recurrence_records.append(
                {"candidate_id": recurrence["candidate_id"], "spans": spans}
            )
        if unable:
            aliases.append(
                {
                    "alias_id": alias.get("alias_id"),
                    "mapping_status": "unable_to_compare",
                    "reason": "group anchors cannot be normalized to the root score span",
                }
            )
            continue
        aliases.append(
            {
                "alias_id": alias.get("alias_id"),
                "mapping_status": (
                    "candidate_aligned"
                    if aligned_ids.intersection(candidate_ids)
                    else "oracle_mapped"
                ),
                "score_timing_candidate_ids": candidate_ids,
                "oracle_common_boundaries": [
                    _fraction_list(value) for value in sorted(common)
                ],
                "oracle_allowed_boundaries": [
                    _fraction_list(value) for value in sorted(allowed)
                ],
                "observed_boundaries": observed_boundaries,
                "observed_recurrence_spans": recurrence_records,
            }
        )
    return {
        "schema_version": 1,
        "fixture_id": fixture_id,
        "name": observed_source["name"],
        "oracle_material_annotations": _oracle_material_annotations(oracle_source),
        "aliases": aliases,
    }


def run_known_structure_evaluation(
    *,
    observed_files: Path,
    observed_manifest: Path,
    timing_files: Path,
    timing_manifest: Path,
    oracle_files: Path,
    oracle_manifest: Path,
    repository_root: Path,
    method_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """raw観測結果を変更せず、別成果物としてoracle診断を保存する。"""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise KnownStructureEvaluationError(f"output directory is not empty: {output_dir}")
    method_hash = verify_method_manifest(
        repository_root=Path(repository_root),
        manifest_path=Path(method_manifest),
    )
    observed_manifest_payload = _read_json(Path(observed_manifest))
    timing_manifest_payload = _read_json(Path(timing_manifest))
    oracle_manifest_payload = _read_json(Path(oracle_manifest))
    contracts = (
        (
            observed_manifest_payload,
            Path(observed_files),
            "source-candidates.jsonl",
            {2},
            {"pass"},
        ),
        (
            timing_manifest_payload,
            Path(timing_files),
            "files.jsonl",
            {1},
            {"pass"},
        ),
        (
            oracle_manifest_payload,
            Path(oracle_files),
            "candidate-sets.jsonl",
            {1},
            {"pass"},
        ),
    )
    for manifest, path, output_name, schema_versions, statuses in contracts:
        outputs = manifest.get("outputs")
        if (
            manifest.get("schema_version") not in schema_versions
            or manifest.get("status") not in statuses
            or not isinstance(outputs, Mapping)
            or outputs.get(output_name) != sha256_file(path)
        ):
            raise KnownStructureEvaluationError(
                f"input artifact manifest contract is invalid: {output_name}"
            )
    for manifest, label in (
        (observed_manifest_payload, "observed"),
        (timing_manifest_payload, "timing"),
    ):
        inputs = manifest.get("inputs")
        if not isinstance(inputs, Mapping) or inputs.get("method_manifest") != method_hash:
            raise KnownStructureEvaluationError(
                f"{label} method manifest SHA-256 mismatch"
            )
    observed = _read_jsonl(Path(observed_files))
    timing = _read_jsonl(Path(timing_files))
    oracle = _read_jsonl(Path(oracle_files))
    timing_by_name = {str(record.get("name")): record for record in timing}
    oracle_by_id = {str(record.get("fixture_id")): record for record in oracle}
    if len(timing_by_name) != len(timing) or len(oracle_by_id) != len(oracle):
        raise KnownStructureEvaluationError("duplicate input source key")
    results = []
    for source in sorted(observed, key=lambda item: str(item.get("name", "")).casefold()):
        name = str(source.get("name"))
        fixture_id = Path(name).stem
        lower = timing_by_name.get(name)
        expected = oracle_by_id.get(fixture_id)
        if lower is None or expected is None:
            raise KnownStructureEvaluationError(f"source join is incomplete: {name}")
        results.append(
            evaluate_known_source(
                observed_source=source,
                timing_source=lower,
                oracle_source=expected,
            )
        )
    if len(results) != len(timing) or len(results) != len(oracle):
        raise KnownStructureEvaluationError("source sets differ across inputs")
    mapping_counts = {
        status: sum(
            alias.get("mapping_status") == status
            for result in results
            for alias in result["aliases"]
        )
        for status in ("candidate_aligned", "oracle_mapped", "unable_to_compare")
    }
    summary = {
        "schema_version": 1,
        "status": "pass",
        "fixture_count": len(results),
        "mapping_counts": mapping_counts,
        "interpretation": (
            "outside_enumerated_candidates is unenumerated, not a proven false boundary"
        ),
    }
    run_spec = {
        "schema_version": 1,
        "split_role": "known_fixture",
        "inputs": {
            "observed_files": sha256_file(Path(observed_files)),
            "observed_manifest": sha256_file(Path(observed_manifest)),
            "timing_files": sha256_file(Path(timing_files)),
            "timing_manifest": sha256_file(Path(timing_manifest)),
            "oracle_files": sha256_file(Path(oracle_files)),
            "oracle_manifest": sha256_file(Path(oracle_manifest)),
            "method_manifest": method_hash,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(output_dir / "evaluation.jsonl", _jsonl_bytes(results))
    atomic_write_json(output_dir / "summary.json", summary)
    atomic_write_json(output_dir / "run-spec.json", run_spec)
    output_names = ("evaluation.jsonl", "run-spec.json", "summary.json")
    atomic_write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "status": "pass",
            "inputs": {"method_manifest": method_hash},
            "outputs": {
                name: sha256_file(output_dir / name) for name in output_names
            },
        },
    )
    return {
        "status": "pass",
        "fixture_count": len(results),
        "mapping_counts": mapping_counts,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed-files", type=Path, required=True)
    parser.add_argument("--observed-manifest", type=Path, required=True)
    parser.add_argument("--timing-files", type=Path, required=True)
    parser.add_argument("--timing-manifest", type=Path, required=True)
    parser.add_argument("--oracle-files", type=Path, required=True)
    parser.add_argument("--oracle-manifest", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--method-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    result = run_known_structure_evaluation(
        observed_files=arguments.observed_files,
        observed_manifest=arguments.observed_manifest,
        timing_files=arguments.timing_files,
        timing_manifest=arguments.timing_manifest,
        oracle_files=arguments.oracle_files,
        oracle_manifest=arguments.oracle_manifest,
        repository_root=arguments.repository_root,
        method_manifest=arguments.method_manifest,
        output_dir=arguments.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
