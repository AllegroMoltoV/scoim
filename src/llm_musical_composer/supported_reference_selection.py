"""保存済みの広域候補から、再現可能な実験用参照曲を選ぶ。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_musical_composer.composition_request import (
    normalize_composition_request,
    resolve_composition_request,
)
from llm_musical_composer.reference_generation_target import (
    ReferenceGenerationTarget,
    build_reference_generation_target,
)
from llm_musical_composer.run_state import sha256_file, sha256_json

SELECTION_VERSION = "supported-reference-selection-v1"
DEFAULT_SEED = "normal-generation-v1-seed-001"
EXPECTED_ANALYSIS_MANIFEST_SHA256 = (
    "f51fe63c786b5b54555b0cd233799c8b7977f51f7bf0880cf8d9ad0ea0208980"
)
EXPECTED_SELECTION_SHA256 = (
    "17e39323a3a1450f535294a837c6d20a54c0df0baa1411d51e2e6d746ddcf952"
)
POPULATION = "eligible_copy_deduplicated"
DISTANCE_METHOD = "empirical_quantile"
K = 7
FRACTION = 0.75
EXPECTED_CANDIDATE_COUNT = 171


class SupportedReferenceSelectionError(ValueError):
    """候補分析または選択契約を再現できない。"""


@dataclass(frozen=True)
class SupportedReferenceSelection:
    selection: dict[str, Any]
    normalized_request: dict[str, Any]
    resolved_request: dict[str, Any]
    medoid_control: dict[str, Any]
    target: ReferenceGenerationTarget


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SupportedReferenceSelectionError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise SupportedReferenceSelectionError(f"JSON object required: {path}")
    return value


def _verify_digest(actual_path: Path, expected: object, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise SupportedReferenceSelectionError(f"{label} hash is missing")
    if not actual_path.is_file() or sha256_file(actual_path).casefold() != expected.casefold():
        raise SupportedReferenceSelectionError(f"{label} hash mismatch: {actual_path}")


def _verify_analysis(
    project_root: Path,
    analysis_dir: Path,
    *,
    expected_manifest_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    manifest_path = analysis_dir / "manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    if (
        expected_manifest_sha256 is not None
        and manifest_sha256.casefold() != expected_manifest_sha256.casefold()
    ):
        raise SupportedReferenceSelectionError("analysis manifest hash mismatch")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise SupportedReferenceSelectionError("analysis manifest is incomplete")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise SupportedReferenceSelectionError("analysis manifest outputs are invalid")
    for name, digest in outputs.items():
        if not isinstance(name, str):
            raise SupportedReferenceSelectionError("analysis output name is invalid")
        _verify_digest(analysis_dir / name, digest, f"analysis output {name}")

    run_spec = _read_json(analysis_dir / "run-spec.json")
    inputs = run_spec.get("inputs")
    if not isinstance(inputs, Mapping):
        raise SupportedReferenceSelectionError("analysis input hashes are invalid")
    live_inputs = {
        "reference_manifest_sha256": project_root
        / ".appendix"
        / "reference-profile-v1"
        / "manifest.json",
        "control_manifest_sha256": project_root
        / ".appendix"
        / "control-reference-baseline-v3"
        / "manifest.json",
        "evaluation_groups_sha256": project_root
        / ".appendix"
        / "corpus-audit"
        / "evaluation-groups.json",
    }
    for key, path in live_inputs.items():
        _verify_digest(path, inputs.get(key), f"analysis input {key}")
    return manifest, run_spec, manifest_sha256


def _candidate_names(analysis_dir: Path) -> list[str]:
    sensitivity = _read_json(analysis_dir / "candidate-pool-sensitivity.json")
    try:
        record = sensitivity["candidate_sets"][POPULATION][DISTANCE_METHOD][
            f"k={K}:{FRACTION:.2f}"
        ]
        names = record["names"]
    except (KeyError, TypeError) as error:
        raise SupportedReferenceSelectionError("candidate set is unavailable") from error
    if (
        not isinstance(record, Mapping)
        or record.get("actual_count") != EXPECTED_CANDIDATE_COUNT
        or record.get("fraction") != FRACTION
        or not isinstance(names, list)
        or len(names) != EXPECTED_CANDIDATE_COUNT
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise SupportedReferenceSelectionError("candidate set does not match fixed contract")
    if len({name.casefold() for name in names}) != len(names):
        raise SupportedReferenceSelectionError("candidate set contains duplicate names")

    eligibility_path = analysis_dir / "target-eligibility.jsonl"
    try:
        eligibility = {
            row["name"].casefold(): row["status"]
            for row in (
                json.loads(line)
                for line in eligibility_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise SupportedReferenceSelectionError("target eligibility is invalid") from error
    if any(eligibility.get(name.casefold()) != "success" for name in names):
        raise SupportedReferenceSelectionError("candidate set contains an ineligible reference")
    return names


def select_name_from_candidates(names: Sequence[str], selection_sha256: str) -> tuple[int, str]:
    """候補入力順に依存せず、選択ハッシュから一曲を返す。"""
    if not names:
        raise SupportedReferenceSelectionError("candidate set is empty")
    ordered = sorted(names, key=lambda name: (name.casefold(), name))
    if len(set(ordered)) != len(ordered):
        raise SupportedReferenceSelectionError("candidate set contains duplicate names")
    try:
        numeric = int(selection_sha256, 16)
    except (TypeError, ValueError) as error:
        raise SupportedReferenceSelectionError("selection hash is invalid") from error
    index = numeric % len(ordered)
    return index, ordered[index]


def _reference_hash(reference_manifest: Mapping[str, Any], name: str) -> str:
    inputs = reference_manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise SupportedReferenceSelectionError("reference manifest inputs are invalid")
    matches = [
        digest
        for candidate, digest in inputs.items()
        if isinstance(candidate, str) and candidate.casefold() == name.casefold()
    ]
    if len(matches) != 1 or not isinstance(matches[0], str):
        raise SupportedReferenceSelectionError("selected reference hash is unavailable")
    return matches[0].lower()


def select_supported_reference(
    project_root: Path,
    analysis_dir: Path,
    *,
    seed: str = DEFAULT_SEED,
    expected_analysis_manifest_sha256: str | None = EXPECTED_ANALYSIS_MANIFEST_SHA256,
    expected_selection_sha256: str | None = EXPECTED_SELECTION_SHA256,
    explicit_brightness: int | None = None,
    explicit_attack_frequency: float | None = None,
) -> SupportedReferenceSelection:
    """固定済みの広域候補から実験用参照を選び、匿名目標を構築する。"""
    if explicit_brightness is not None and explicit_brightness not in {-1, 0, 1}:
        raise SupportedReferenceSelectionError(
            "internal explicit brightness must be -1, 0, or 1"
        )
    project_root = Path(project_root).resolve()
    analysis_dir = Path(analysis_dir).resolve()
    if not seed:
        raise SupportedReferenceSelectionError("seed must not be empty")
    _, _, manifest_sha256 = _verify_analysis(
        project_root,
        analysis_dir,
        expected_manifest_sha256=expected_analysis_manifest_sha256,
    )
    names = _candidate_names(analysis_dir)
    selection_payload = {
        "analysis_manifest_sha256": manifest_sha256,
        "seed": seed,
    }
    selection_sha256 = sha256_json(selection_payload)
    if (
        expected_selection_sha256 is not None
        and selection_sha256.casefold() != expected_selection_sha256.casefold()
    ):
        raise SupportedReferenceSelectionError("selection hash mismatch")
    selected_index, selected_name = select_name_from_candidates(names, selection_sha256)

    reference_dir = project_root / ".appendix" / "reference-profile-v1"
    control_dir = project_root / ".appendix" / "control-reference-baseline-v3"
    reference_manifest = _read_json(reference_dir / "manifest.json")
    reference_summary = _read_json(reference_dir / "summary.json")
    reference_hashes = reference_manifest.get("inputs")
    if not isinstance(reference_hashes, Mapping):
        raise SupportedReferenceSelectionError("reference hashes are unavailable")
    explicit_controls: dict[str, int | float] = {}
    published_controls: set[str] = set()
    if explicit_brightness is not None:
        explicit_controls["あかるさ"] = explicit_brightness
        published_controls.add("あかるさ")
    if explicit_attack_frequency is not None:
        explicit_controls["発音頻度"] = explicit_attack_frequency
        published_controls.add("発音頻度")
    normalized = normalize_composition_request(
        {
            "schema_version": 1,
            "preset": "solo_piano_3m_v1",
            "controls": explicit_controls,
        },
        published_controls=frozenset(published_controls),
    )
    medoid = resolve_composition_request(
        normalized,
        reference_summary=reference_summary,
        reference_hashes=reference_hashes,
    )
    resolved_request = {
        "schema_version": 1,
        "preset": "solo_piano_3m_v1",
        "normalized_request_sha256": normalized.sha256,
        "reference": {
            "state": "resolved",
            "name": selected_name,
            "sha256": _reference_hash(reference_manifest, selected_name),
            "selection_method": "supported_pool_seed_v1",
        },
        "controls": normalized.value["controls"],
    }
    target = build_reference_generation_target(
        resolved_request,
        reference_dir=reference_dir,
        control_dir=control_dir,
    )
    selection = {
        "schema_version": 1,
        "selection_version": SELECTION_VERSION,
        "population": POPULATION,
        "distance_method": DISTANCE_METHOD,
        "k": K,
        "fraction": FRACTION,
        "candidate_count": len(names),
        "seed": seed,
        "seed_hash_method": "run-state-sha256-json-v1",
        "analysis_manifest_sha256": manifest_sha256,
        "selection_sha256": selection_sha256,
        "selected_index": selected_index,
        "selected_name": selected_name,
        "selected_reference_sha256": resolved_request["reference"]["sha256"],
    }
    return SupportedReferenceSelection(
        selection=selection,
        normalized_request=normalized.value,
        resolved_request=resolved_request,
        medoid_control=medoid.value,
        target=target,
    )
