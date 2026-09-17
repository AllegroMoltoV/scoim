"""既知生成元の先頭・最終発音を使って観測候補をroot座標へ写す。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from llm_musical_composer.known_structure_evaluation import (
    _jsonl_bytes,
    _oracle_boundaries,
    _oracle_material_annotations,
    _read_json,
    _read_jsonl,
)
from llm_musical_composer.observed_structure_method import FROZEN_METHOD_FILES
from llm_musical_composer.performance_pipeline import (
    PiecePlan,
    ScoreSpec,
    ordered_leaf_schedule,
    validate_score_spec,
)
from llm_musical_composer.pipeline_dsl import parse_piece_plan, parse_score_spec
from llm_musical_composer.run_state import (
    atomic_write_bytes,
    atomic_write_json,
    sha256_file,
)
from llm_musical_composer.structure_equivalence_schema import (
    compact_structure_descriptor,
    compact_structure_descriptor_from_dict,
)

EXPECTED_METHOD_MANIFEST_SHA256 = (
    "6c534ae279cd9d00a78db4ca27924c3a4e654662fd8bf0d2ce7bab012bc8ef16"
)
EXPECTED_INPUT_SHA256 = {
    "observed_files": "113f140f4634a28abdca9b4c5547cbbded6b1d52fd0fef7e2dbca468c1492bb9",
    "observed_manifest": "33eac44e7d5602897c3e2b47c45dffed05b687335304a88bdecb5491e5dba717",
    "timing_files": "fa9caeca2a5d8dcf2ffed77f393a9e56d0ebfc2e7a9f49925766385e680d886a",
    "timing_manifest": "e2e895686f18314e94baf6e06a94640a2de4de5c230a8f985c221f9d3149b0ac",
    "oracle_files": "e9265ac13075ba4d8afaf0bb64755f96049dc12f830e83dd7761ba7ce5f67fb9",
    "oracle_manifest": "2385b083406242d85e3fc42c837ff39cf66753178283b338ac2052b5218c852e",
}
EVALUATOR_STATIC_FILES = frozenset(
    {
        "docs/design/reference-smf-reverse-decomposition.md",
        "pyproject.toml",
        "tests/test_known_structure_evaluation_v2.py",
    }
)


class KnownStructureEvaluationV2Error(ValueError):
    """既知構造比較V2の入力または来歴契約に違反した。"""


class OracleGeometryUnavailable(KnownStructureEvaluationV2Error):
    """入力は読めたが、oracle座標を一意に証明できない。"""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_value(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _git(
    repository_root: Path, *arguments: str, text: bool = True
) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=text,
    )
    if result.returncode:
        stderr = (
            result.stderr.strip()
            if text
            else result.stderr.decode(errors="replace").strip()
        )
        raise KnownStructureEvaluationV2Error(
            f"git {' '.join(arguments)} failed: {stderr}"
        )
    return result.stdout.strip() if text else result.stdout


def _validated_relative_paths(paths: Sequence[str]) -> tuple[str, ...]:
    result = []
    for value in paths:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
            raise KnownStructureEvaluationV2Error(
                f"manifest file path is invalid: {value}"
            )
        result.append(value)
    if len(set(result)) != len(result):
        raise KnownStructureEvaluationV2Error("manifest file paths must be unique")
    return tuple(result)


def _commit_exists(repository_root: Path, commit: str) -> None:
    if not commit:
        raise KnownStructureEvaluationV2Error("manifest commit is missing")
    _git(repository_root, "cat-file", "-e", f"{commit}^{{commit}}")


def _fraction_list(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def verify_archived_method_manifest(
    *,
    repository_root: Path,
    manifest_path: Path,
    expected_manifest_sha256: str = EXPECTED_METHOD_MANIFEST_SHA256,
    expected_files: Sequence[str] = FROZEN_METHOD_FILES,
) -> str:
    """過去commitのblobを使い、抽出方式の凍結manifestを検証する。"""
    repository_root = Path(repository_root).resolve()
    manifest_path = Path(manifest_path)
    actual_manifest_sha256 = sha256_file(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise KnownStructureEvaluationV2Error("method manifest SHA-256 mismatch")
    manifest = _read_json(manifest_path)
    files = manifest.get("files")
    constants = manifest.get("constants")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "frozen"
        or not isinstance(files, Mapping)
        or not isinstance(constants, Mapping)
    ):
        raise KnownStructureEvaluationV2Error("method manifest contract is invalid")
    declared_paths = _validated_relative_paths([str(value) for value in files])
    if set(declared_paths) != set(_validated_relative_paths(expected_files)):
        raise KnownStructureEvaluationV2Error(
            "method manifest file set differs from expected files"
        )
    if manifest.get("constants_sha256") != _sha256_value(constants):
        raise KnownStructureEvaluationV2Error("method constants SHA-256 mismatch")
    frozen_commit = str(manifest.get("frozen_commit", ""))
    _commit_exists(repository_root, frozen_commit)
    for relative in declared_paths:
        declared_hash = str(files[relative]).lower()
        committed = _git(
            repository_root, "show", f"{frozen_commit}:{relative}", text=False
        )
        if hashlib.sha256(committed).hexdigest() != declared_hash:
            raise KnownStructureEvaluationV2Error(
                f"frozen method blob SHA-256 mismatch: {relative}"
            )
    return actual_manifest_sha256


def _evaluator_files_at_commit(repository_root: Path, commit: str) -> tuple[str, ...]:
    tracked = str(
        _git(
            repository_root,
            "ls-tree",
            "-r",
            "--name-only",
            commit,
            "--",
            "src/llm_musical_composer",
        )
    ).splitlines()
    source_files = {value for value in tracked if value.endswith(".py")}
    return tuple(sorted(source_files | set(EVALUATOR_STATIC_FILES)))


def write_evaluator_manifest(
    *, repository_root: Path, output_path: Path
) -> dict[str, Any]:
    """現在のcommitと一致するV2比較器の推移的入力を凍結する。"""
    repository_root = Path(repository_root).resolve()
    commit = str(_git(repository_root, "rev-parse", "HEAD"))
    files = {}
    for relative in _evaluator_files_at_commit(repository_root, commit):
        current = repository_root / relative
        if not current.is_file():
            raise KnownStructureEvaluationV2Error(
                f"evaluator file is missing: {relative}"
            )
        committed = _git(repository_root, "show", f"{commit}:{relative}", text=False)
        committed_hash = hashlib.sha256(committed).hexdigest()
        if sha256_file(current) != committed_hash:
            raise KnownStructureEvaluationV2Error(
                f"evaluator file differs from HEAD: {relative}"
            )
        files[relative] = committed_hash
    payload = {
        "schema_version": 1,
        "status": "frozen",
        "frozen_commit": commit,
        "files": files,
    }
    atomic_write_json(Path(output_path), payload)
    return payload


def verify_evaluator_manifest(
    *, repository_root: Path, manifest_path: Path
) -> str:
    """V2比較器のmanifest、commit blob、現在の作業ツリーを照合する。"""
    repository_root = Path(repository_root).resolve()
    manifest_path = Path(manifest_path)
    manifest = _read_json(manifest_path)
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "frozen"
        or not isinstance(files, Mapping)
    ):
        raise KnownStructureEvaluationV2Error(
            "evaluator manifest contract is invalid"
        )
    commit = str(manifest.get("frozen_commit", ""))
    _commit_exists(repository_root, commit)
    if str(_git(repository_root, "rev-parse", "HEAD")) != commit:
        raise KnownStructureEvaluationV2Error(
            "current HEAD differs from evaluator commit"
        )
    declared_paths = _validated_relative_paths([str(value) for value in files])
    if set(declared_paths) != set(_evaluator_files_at_commit(repository_root, commit)):
        raise KnownStructureEvaluationV2Error("evaluator file set is incomplete")
    for relative in declared_paths:
        declared_hash = str(files[relative]).lower()
        committed = _git(repository_root, "show", f"{commit}:{relative}", text=False)
        current = repository_root / relative
        if (
            hashlib.sha256(committed).hexdigest() != declared_hash
            or not current.is_file()
            or sha256_file(current) != declared_hash
        ):
            raise KnownStructureEvaluationV2Error(
                f"evaluator file SHA-256 mismatch: {relative}"
            )
    return sha256_file(manifest_path)


def _tail_ratios(oracle_source: Mapping[str, Any]) -> set[Fraction]:
    result = set()
    for candidate in oracle_source.get("candidates", []):
        descriptor = candidate.get("descriptor") if isinstance(candidate, Mapping) else None
        value = descriptor.get("terminal_tail_ratio") if isinstance(descriptor, Mapping) else None
        if not isinstance(value, list) or len(value) != 2:
            raise KnownStructureEvaluationV2Error("oracle terminal tail ratio is invalid")
        result.add(Fraction(int(value[0]), int(value[1])))
    return result


def oracle_attack_span(
    *,
    plan: PiecePlan,
    score: ScoreSpec,
    oracle_source: Mapping[str, Any],
) -> tuple[Fraction, Fraction]:
    """既知譜面の実発音範囲をroot区間内の有理数座標で返す。"""
    validate_score_spec(plan, score)
    leaves, intervals = ordered_leaf_schedule(plan, score)
    total_units = intervals[plan.root_node_id][1]
    materials = {material.material_id: material for material in score.materials}
    attacks = [
        start + note.at_units
        for node, start, _ in leaves
        for note in materials[str(node.score_material_id)].notes
    ]
    if total_units <= 0 or not attacks:
        raise OracleGeometryUnavailable("oracle score has no attack span")
    first = Fraction(min(attacks), total_units)
    last = Fraction(max(attacks), total_units)
    if not Fraction(0) <= first < last <= Fraction(1):
        raise OracleGeometryUnavailable("oracle attack span is invalid")

    candidates = oracle_source.get("candidates")
    if not isinstance(candidates, list):
        raise OracleGeometryUnavailable("oracle candidates are missing")
    baselines = [
        candidate
        for candidate in candidates
        if isinstance(candidate, Mapping)
        and candidate.get("origin_status") == "oracle_fixture_baseline"
    ]
    if len(baselines) != 1:
        raise OracleGeometryUnavailable(
            "oracle fixture must have exactly one baseline"
        )
    baseline_payload = baselines[0].get("descriptor")
    if not isinstance(baseline_payload, dict):
        raise OracleGeometryUnavailable("oracle baseline descriptor is invalid")
    try:
        baseline = compact_structure_descriptor_from_dict(baseline_payload)
        calculated = compact_structure_descriptor(plan, score)
    except (KeyError, TypeError, ValueError) as error:
        raise OracleGeometryUnavailable(
            f"oracle baseline descriptor is invalid: {error}"
        ) from error
    if baseline != calculated:
        raise OracleGeometryUnavailable(
            "oracle baseline descriptor differs from source DSL"
        )
    tail_ratios = _tail_ratios(oracle_source)
    if len(tail_ratios) != 1 or next(iter(tail_ratios)) != Fraction(1) - last:
        raise OracleGeometryUnavailable(
            "oracle terminal tail differs across candidates or source DSL"
        )
    return first, last


def _mapped_position(
    edge: int,
    *,
    positions: Sequence[int],
    group_count: int,
    oracle_first_attack: Fraction,
    oracle_last_attack: Fraction,
) -> Fraction | None:
    if (
        group_count < 2
        or len(positions) != group_count
        or edge < 0
        or edge >= group_count
    ):
        return None
    first = int(positions[0])
    last = int(positions[-1])
    if last <= first:
        return None
    ratio = Fraction(int(positions[edge]) - first, last - first)
    return oracle_first_attack + ratio * (oracle_last_attack - oracle_first_attack)


def _validate_source_identity(
    *,
    observed_source: Mapping[str, Any],
    timing_source: Mapping[str, Any],
    oracle_source: Mapping[str, Any],
    expected_source_sha256: str,
) -> tuple[str, str]:
    if observed_source.get("schema_version") != 2:
        raise KnownStructureEvaluationV2Error("observed source schema is invalid")
    fixture_id = str(oracle_source.get("fixture_id", ""))
    expected_name = f"{fixture_id}.mid"
    if observed_source.get("name") != expected_name or timing_source.get(
        "name"
    ) != expected_name:
        raise KnownStructureEvaluationV2Error(
            "source names do not match oracle fixture"
        )
    if (
        observed_source.get("source_sha256") != expected_source_sha256
        or timing_source.get("source_sha256") != expected_source_sha256
    ):
        raise KnownStructureEvaluationV2Error(
            "source SHA-256 does not match oracle SMF"
        )
    ledgers = {
        observed_source.get("source_ledger_sha256"),
        timing_source.get("source_ledger_sha256"),
        oracle_source.get("source_ledger_sha256"),
    }
    if None in ledgers or len(ledgers) != 1:
        raise KnownStructureEvaluationV2Error("source ledger SHA-256 values differ")
    return fixture_id, expected_name


def evaluate_known_source_v2(
    *,
    observed_source: Mapping[str, Any],
    timing_source: Mapping[str, Any],
    oracle_source: Mapping[str, Any],
    oracle_first_attack: Fraction,
    oracle_last_attack: Fraction,
    expected_source_sha256: str,
) -> dict[str, Any]:
    """一つの既知生成元を、root終端と最終発音を分けて比較する。"""
    fixture_id, expected_name = _validate_source_identity(
        observed_source=observed_source,
        timing_source=timing_source,
        oracle_source=oracle_source,
        expected_source_sha256=expected_source_sha256,
    )
    if not Fraction(0) <= oracle_first_attack < oracle_last_attack <= Fraction(1):
        raise KnownStructureEvaluationV2Error("oracle attack span is invalid")

    common, allowed = _oracle_boundaries(oracle_source)
    tail_ratios = _tail_ratios(oracle_source)
    geometry_available = (
        len(tail_ratios) == 1
        and next(iter(tail_ratios)) == Fraction(1) - oracle_last_attack
    )
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
            raise KnownStructureEvaluationV2Error("observed alias is invalid")
        candidate_ids = [str(value) for value in alias.get("candidate_ids", [])]
        profile = profiles.get(str(alias.get("semantic_profile_hash")))
        positions_sets = []
        for candidate_id in candidate_ids:
            candidate = lower.get(candidate_id)
            refs = candidate.get("attack_group_refs") if isinstance(candidate, Mapping) else None
            if isinstance(refs, list):
                positions_sets.append(tuple(int(item["score_position"]) for item in refs))
        if (
            not geometry_available
            or not isinstance(profile, Mapping)
            or not positions_sets
            or len(set(positions_sets)) != 1
        ):
            aliases.append(
                {
                    "alias_id": alias.get("alias_id"),
                    "mapping_status": "unable_to_compare",
                    "reason": "a unique score-to-root mapping is unavailable",
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
            raise KnownStructureEvaluationV2Error("observed profile payload is invalid")
        observed_boundaries = []
        mapping_failed = False
        for boundary in boundary_payload.get("candidates", []):
            position = _mapped_position(
                int(boundary["start_group_index"]),
                positions=positions,
                group_count=group_count,
                oracle_first_attack=oracle_first_attack,
                oracle_last_attack=oracle_last_attack,
            )
            if position is None:
                mapping_failed = True
                break
            relation = (
                "common_boundary"
                if position in common
                else "allowed_boundary"
                if position in allowed
                else "outside_enumerated_candidates"
            )
            observed_boundaries.append(
                {"position": _fraction_list(position), "oracle_relation": relation}
            )
        if mapping_failed:
            aliases.append(
                {
                    "alias_id": alias.get("alias_id"),
                    "mapping_status": "unable_to_compare",
                    "reason": "a boundary group anchor cannot be mapped",
                }
            )
            continue
        recurrence_records = []
        for recurrence in recurrence_payload.get("candidates", []):
            size = int(recurrence["window_size"])
            occurrences = []
            for occurrence in recurrence["occurrences"]:
                start_edge = int(occurrence[0])
                end_edge = start_edge + size
                start = _mapped_position(
                    start_edge,
                    positions=positions,
                    group_count=group_count,
                    oracle_first_attack=oracle_first_attack,
                    oracle_last_attack=oracle_last_attack,
                )
                if start is None:
                    mapping_failed = True
                    break
                if end_edge == group_count:
                    occurrences.append(
                        {
                            "start": _fraction_list(start),
                            "end": None,
                            "endpoint_status": "terminal_endpoint_unobservable",
                        }
                    )
                    continue
                end = _mapped_position(
                    end_edge,
                    positions=positions,
                    group_count=group_count,
                    oracle_first_attack=oracle_first_attack,
                    oracle_last_attack=oracle_last_attack,
                )
                if end is None:
                    mapping_failed = True
                    break
                occurrences.append(
                    {
                        "start": _fraction_list(start),
                        "end": _fraction_list(end),
                        "endpoint_status": "mapped",
                    }
                )
            recurrence_records.append(
                {"candidate_id": recurrence["candidate_id"], "occurrences": occurrences}
            )
        if mapping_failed:
            aliases.append(
                {
                    "alias_id": alias.get("alias_id"),
                    "mapping_status": "unable_to_compare",
                    "reason": "a recurrence group anchor cannot be mapped",
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
                "observed_recurrences": recurrence_records,
            }
        )
    return {
        "schema_version": 2,
        "fixture_id": fixture_id,
        "name": expected_name,
        "oracle_attack_span": {
            "first": _fraction_list(oracle_first_attack),
            "last": _fraction_list(oracle_last_attack),
        },
        "oracle_material_annotations": _oracle_material_annotations(oracle_source),
        "aliases": aliases,
    }


def _oracle_fixture_geometry(
    *,
    fixtures_root: Path,
    fixture_id: str,
    oracle_manifest: Mapping[str, Any],
    oracle_source: Mapping[str, Any],
) -> tuple[Fraction, Fraction, str]:
    inputs = oracle_manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise KnownStructureEvaluationV2Error("oracle manifest inputs are invalid")
    fixture_root = Path(fixtures_root) / fixture_id
    paths = {
        "piece-plan.dsl": fixture_root / "piece-plan.dsl",
        "score-spec.dsl": fixture_root / "score-spec.dsl",
        "final.mid": fixture_root / "final.mid",
    }
    for name, path in paths.items():
        key = f"{fixture_id}/{name}"
        if inputs.get(key) != sha256_file(path):
            raise KnownStructureEvaluationV2Error(
                f"oracle fixture SHA-256 mismatch: {key}"
            )
    plan = parse_piece_plan(paths["piece-plan.dsl"].read_text(encoding="utf-8"))
    score = parse_score_spec(paths["score-spec.dsl"].read_text(encoding="utf-8"))
    first, last = oracle_attack_span(
        plan=plan, score=score, oracle_source=oracle_source
    )
    return first, last, str(inputs[f"{fixture_id}/final.mid"])


def _unable_source_result(
    *,
    observed_source: Mapping[str, Any],
    timing_source: Mapping[str, Any],
    oracle_source: Mapping[str, Any],
    expected_source_sha256: str,
    reason: str,
) -> dict[str, Any]:
    fixture_id, expected_name = _validate_source_identity(
        observed_source=observed_source,
        timing_source=timing_source,
        oracle_source=oracle_source,
        expected_source_sha256=expected_source_sha256,
    )
    aliases = [
        {
            "alias_id": alias.get("alias_id"),
            "mapping_status": "unable_to_compare",
            "reason": reason,
        }
        for alias in observed_source.get("score_timing_aliases", [])
        if isinstance(alias, Mapping)
    ]
    return {
        "schema_version": 2,
        "fixture_id": fixture_id,
        "name": expected_name,
        "oracle_attack_span": None,
        "oracle_material_annotations": _oracle_material_annotations(oracle_source),
        "aliases": aliases,
    }


def _artifact_contract(
    *,
    manifest: Mapping[str, Any],
    artifact_path: Path,
    output_name: str,
    schema_version: int,
) -> None:
    outputs = manifest.get("outputs")
    if (
        manifest.get("schema_version") != schema_version
        or manifest.get("status") != "pass"
        or not isinstance(outputs, Mapping)
        or outputs.get(output_name) != sha256_file(artifact_path)
    ):
        raise KnownStructureEvaluationV2Error(
            f"input artifact manifest contract is invalid: {output_name}"
        )


def _unique_records(
    records: Sequence[Mapping[str, Any]], *, key: str, label: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        value = str(record.get(key, ""))
        if not value or value in result:
            raise KnownStructureEvaluationV2Error(
                f"duplicate or empty {label} source key"
            )
        result[value] = record
    return result


def run_known_structure_evaluation_v2(
    *,
    observed_files: Path,
    observed_manifest: Path,
    timing_files: Path,
    timing_manifest: Path,
    oracle_files: Path,
    oracle_manifest: Path,
    fixtures_root: Path,
    repository_root: Path,
    method_manifest: Path,
    evaluator_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """保存済みraw候補を、発音範囲を補正したoracle座標で比較する。"""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise KnownStructureEvaluationV2Error(
            f"output directory is not empty: {output_dir}"
        )
    input_paths = {
        "observed_files": Path(observed_files),
        "observed_manifest": Path(observed_manifest),
        "timing_files": Path(timing_files),
        "timing_manifest": Path(timing_manifest),
        "oracle_files": Path(oracle_files),
        "oracle_manifest": Path(oracle_manifest),
    }
    input_hashes = {name: sha256_file(path) for name, path in input_paths.items()}
    if input_hashes != EXPECTED_INPUT_SHA256:
        differences = sorted(
            name
            for name in EXPECTED_INPUT_SHA256
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256[name]
        )
        raise KnownStructureEvaluationV2Error(
            f"fixed input SHA-256 mismatch: {', '.join(differences)}"
        )
    method_hash = verify_archived_method_manifest(
        repository_root=Path(repository_root), manifest_path=Path(method_manifest)
    )
    evaluator_hash = verify_evaluator_manifest(
        repository_root=Path(repository_root), manifest_path=Path(evaluator_manifest)
    )
    observed_manifest_payload = _read_json(Path(observed_manifest))
    timing_manifest_payload = _read_json(Path(timing_manifest))
    oracle_manifest_payload = _read_json(Path(oracle_manifest))
    _artifact_contract(
        manifest=observed_manifest_payload,
        artifact_path=Path(observed_files),
        output_name="source-candidates.jsonl",
        schema_version=2,
    )
    _artifact_contract(
        manifest=timing_manifest_payload,
        artifact_path=Path(timing_files),
        output_name="files.jsonl",
        schema_version=1,
    )
    _artifact_contract(
        manifest=oracle_manifest_payload,
        artifact_path=Path(oracle_files),
        output_name="candidate-sets.jsonl",
        schema_version=1,
    )
    for manifest, label in (
        (observed_manifest_payload, "observed"),
        (timing_manifest_payload, "timing"),
    ):
        inputs = manifest.get("inputs")
        if not isinstance(inputs, Mapping) or inputs.get("method_manifest") != method_hash:
            raise KnownStructureEvaluationV2Error(
                f"{label} method manifest SHA-256 mismatch"
            )

    observed = _read_jsonl(Path(observed_files))
    timing = _read_jsonl(Path(timing_files))
    oracle = _read_jsonl(Path(oracle_files))
    observed_by_name = _unique_records(observed, key="name", label="observed")
    timing_by_name = _unique_records(timing, key="name", label="timing")
    oracle_by_id = _unique_records(oracle, key="fixture_id", label="oracle")
    observed_names = set(observed_by_name)
    if observed_names != set(timing_by_name) or {
        Path(name).stem for name in observed_names
    } != set(oracle_by_id):
        raise KnownStructureEvaluationV2Error("source sets differ across inputs")

    results = []
    for name in sorted(observed_names, key=str.casefold):
        source = observed_by_name[name]
        lower = timing_by_name[name]
        fixture_id = Path(name).stem
        expected = oracle_by_id[fixture_id]
        try:
            first, last, source_hash = _oracle_fixture_geometry(
                fixtures_root=Path(fixtures_root),
                fixture_id=fixture_id,
                oracle_manifest=oracle_manifest_payload,
                oracle_source=expected,
            )
        except OracleGeometryUnavailable as error:
            smf_inputs = oracle_manifest_payload.get("inputs")
            assert isinstance(smf_inputs, Mapping)
            source_hash = str(smf_inputs.get(f"{fixture_id}/final.mid", ""))
            results.append(
                _unable_source_result(
                    observed_source=source,
                    timing_source=lower,
                    oracle_source=expected,
                    expected_source_sha256=source_hash,
                    reason=str(error),
                )
            )
            continue
        results.append(
            evaluate_known_source_v2(
                observed_source=source,
                timing_source=lower,
                oracle_source=expected,
                oracle_first_attack=first,
                oracle_last_attack=last,
                expected_source_sha256=source_hash,
            )
        )

    aliases = [alias for result in results for alias in result["aliases"]]
    mapping_counts = {
        status: sum(alias.get("mapping_status") == status for alias in aliases)
        for status in ("candidate_aligned", "oracle_mapped", "unable_to_compare")
    }
    boundary_counts = {
        status: sum(
            boundary.get("oracle_relation") == status
            for alias in aliases
            for boundary in alias.get("observed_boundaries", [])
        )
        for status in (
            "common_boundary",
            "allowed_boundary",
            "outside_enumerated_candidates",
        )
    }
    recurrence_endpoint_counts = {
        status: sum(
            occurrence.get("endpoint_status") == status
            for alias in aliases
            for recurrence in alias.get("observed_recurrences", [])
            for occurrence in recurrence.get("occurrences", [])
        )
        for status in ("mapped", "terminal_endpoint_unobservable")
    }
    summary = {
        "schema_version": 2,
        "status": "pass",
        "fixture_count": len(results),
        "mapping_counts": mapping_counts,
        "boundary_counts": boundary_counts,
        "recurrence_endpoint_counts": recurrence_endpoint_counts,
        "interpretation": (
            "outside_enumerated_candidates is unenumerated, not a proven false boundary"
        ),
    }
    run_spec = {
        "schema_version": 2,
        "split_role": "known_fixture",
        "inputs": {
            **input_hashes,
            "method_manifest": method_hash,
            "evaluator_manifest": evaluator_hash,
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
            "schema_version": 2,
            "status": "pass",
            "inputs": {
                "method_manifest": method_hash,
                "evaluator_manifest": evaluator_hash,
            },
            "outputs": {
                name: sha256_file(output_dir / name) for name in output_names
            },
        },
    )
    return {
        "status": "pass",
        "fixture_count": len(results),
        "mapping_counts": mapping_counts,
        "boundary_counts": boundary_counts,
        "recurrence_endpoint_counts": recurrence_endpoint_counts,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed-files", type=Path, required=True)
    parser.add_argument("--observed-manifest", type=Path, required=True)
    parser.add_argument("--timing-files", type=Path, required=True)
    parser.add_argument("--timing-manifest", type=Path, required=True)
    parser.add_argument("--oracle-files", type=Path, required=True)
    parser.add_argument("--oracle-manifest", type=Path, required=True)
    parser.add_argument("--fixtures-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--method-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    result = run_known_structure_evaluation_v2(
        observed_files=arguments.observed_files,
        observed_manifest=arguments.observed_manifest,
        timing_files=arguments.timing_files,
        timing_manifest=arguments.timing_manifest,
        oracle_files=arguments.oracle_files,
        oracle_manifest=arguments.oracle_manifest,
        fixtures_root=arguments.fixtures_root,
        repository_root=arguments.repository_root,
        method_manifest=arguments.method_manifest,
        evaluator_manifest=arguments.evaluator_manifest,
        output_dir=arguments.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
