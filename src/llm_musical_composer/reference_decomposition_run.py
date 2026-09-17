"""参照 SMF 観測層の全件往復を再現可能な成果物へ保存する。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llm_musical_composer.reference_decomposition import (
    build_observed_performance,
    load_observed_smf,
    observed_smf_differences,
    write_observed_smf,
)
from llm_musical_composer.run_state import atomic_write_bytes, atomic_write_json, sha256_file


class ReferenceDecompositionRunError(ValueError):
    """入力または成果物を信頼できない場合に送出する。"""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReferenceDecompositionRunError(f"unable to read JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReferenceDecompositionRunError(f"JSON root must be an object: {path}")
    return value


def _verify_hash(path: Path, expected: object, *, label: str) -> str:
    if not isinstance(expected, str) or not expected:
        raise ReferenceDecompositionRunError(f"{label} SHA-256 is missing")
    if not Path(path).is_file():
        raise ReferenceDecompositionRunError(f"{label} is missing: {path}")
    actual = sha256_file(Path(path)).lower()
    if actual != expected.lower():
        raise ReferenceDecompositionRunError(f"{label} SHA-256 mismatch: {path}")
    return actual


def _source_records(
    reference_manifest: dict[str, Any], structure_manifest: dict[str, Any]
) -> list[dict[str, str]]:
    if reference_manifest.get("schema_version") != 1:
        raise ReferenceDecompositionRunError("reference manifest schema_version must be 1")
    if reference_manifest.get("status") != "pass":
        raise ReferenceDecompositionRunError("reference manifest status must be pass")
    if structure_manifest.get("schema_version") != 1:
        raise ReferenceDecompositionRunError("structure manifest schema_version must be 1")
    inputs = reference_manifest.get("inputs")
    sources = structure_manifest.get("source_files")
    if not isinstance(inputs, dict) or not isinstance(sources, list):
        raise ReferenceDecompositionRunError("manifest source records are invalid")
    records: list[dict[str, str]] = []
    for index, item in enumerate(sources):
        if not isinstance(item, dict):
            raise ReferenceDecompositionRunError(f"source_files[{index}] must be an object")
        name = item.get("name")
        sha256 = item.get("sha256")
        if not isinstance(name, str) or Path(name).name != name:
            raise ReferenceDecompositionRunError(f"source_files[{index}].name is invalid")
        if not isinstance(sha256, str) or not sha256:
            raise ReferenceDecompositionRunError(f"source_files[{index}].sha256 is invalid")
        if inputs.get(name, "").lower() != sha256.lower():
            raise ReferenceDecompositionRunError(f"manifest source disagreement: {name}")
        records.append({"name": name, "sha256": sha256.lower()})
    if set(inputs) != {item["name"] for item in records}:
        raise ReferenceDecompositionRunError("reference and structure source sets differ")
    return sorted(records, key=lambda item: (item["name"].casefold(), item["name"]))


def _verify_manifests(
    *,
    reference_manifest_path: Path,
    reference_files: Path,
    reference_summary: Path,
    structure_manifest_path: Path,
    structure_files: Path,
    structure_summary: Path,
    structure_controls: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    reference_manifest = _read_json(reference_manifest_path)
    structure_manifest = _read_json(structure_manifest_path)
    reference_outputs = reference_manifest.get("outputs")
    structure_outputs = structure_manifest.get("artifact_sha256")
    if not isinstance(reference_outputs, dict) or not isinstance(structure_outputs, dict):
        raise ReferenceDecompositionRunError("manifest artifact records are invalid")
    _verify_hash(
        reference_files,
        reference_outputs.get("files.jsonl"),
        label="reference artifact",
    )
    _verify_hash(
        reference_summary,
        reference_outputs.get("summary.json"),
        label="reference artifact",
    )
    for path, name in (
        (structure_files, "files.jsonl"),
        (structure_summary, "summary.json"),
        (structure_controls, "controls.json"),
    ):
        _verify_hash(path, structure_outputs.get(name), label="structure artifact")
    return (
        reference_manifest,
        structure_manifest,
        _source_records(reference_manifest, structure_manifest),
    )


def _jsonl_bytes(records: Sequence[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")


def _known_source_outputs(known_source_run: Path) -> tuple[dict[str, Any], Path]:
    run_root = Path(known_source_run).resolve()
    state_path = run_root / "run-state.json"
    state = _read_json(state_path)
    if state.get("status") != "completed":
        raise ReferenceDecompositionRunError("known source run status must be completed")
    steps = state.get("steps")
    if not isinstance(steps, dict):
        raise ReferenceDecompositionRunError("known source run steps are invalid")
    specifications = {
        "piece_plan": ("piece-plan", "path", "sha256"),
        "score_spec": ("score-spec-aggregate", "path", "sha256"),
        "performance_spec": ("performance-spec", "path", "sha256"),
        "smf": ("publish-final", "smf_path", "smf_sha256"),
    }
    verified: dict[str, dict[str, str]] = {}
    smf_path: Path | None = None
    for label, (step_name, path_key, hash_key) in specifications.items():
        step = steps.get(step_name)
        if not isinstance(step, dict) or step.get("status") != "completed":
            raise ReferenceDecompositionRunError(f"known source {label} step must be completed")
        outputs = step.get("outputs")
        if not isinstance(outputs, dict) or not isinstance(outputs.get(path_key), str):
            raise ReferenceDecompositionRunError(f"known source {label} output record is invalid")
        path = (run_root / outputs[path_key]).resolve()
        if run_root != path and run_root not in path.parents:
            raise ReferenceDecompositionRunError(
                f"known source {label} output escapes the run directory"
            )
        actual_hash = _verify_hash(
            path,
            outputs.get(hash_key),
            label=f"known source {label}",
        )
        verified[label] = {
            "path": str(path.relative_to(run_root)),
            "sha256": actual_hash,
        }
        if label == "smf":
            smf_path = path
    if smf_path is None:
        raise ReferenceDecompositionRunError("known source SMF is missing")
    return (
        {
            "run_path": str(run_root),
            "run_state_sha256": sha256_file(state_path),
            "verified_outputs": verified,
        },
        smf_path,
    )


def _destructive_control_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase_1_status": "fixture_defined",
        "controls": [
            {
                "name": "pitch_change",
                "expected_primary_difference": "ObservedSmf.fields",
                "verified_by": "test_roundtrip_difference_reports_track_and_field_changes",
            },
            {
                "name": "timing_change",
                "expected_primary_difference": "ObservedSmf.absolute_tick",
                "verified_by": "test_roundtrip_difference_detects_timing_change_and_pedal_removal",
            },
            {
                "name": "pedal_removal",
                "expected_primary_difference": "ObservedSmf.control_change",
                "verified_by": "test_roundtrip_difference_detects_timing_change_and_pedal_removal",
            },
            {
                "name": "section_reordering",
                "expected_primary_difference": "FormHypothesis",
                "status": "not_yet_assessed",
            },
            {
                "name": "pitch_order_destruction",
                "expected_primary_difference": "LineAndHarmonyHypothesis",
                "status": "not_yet_assessed",
            },
        ],
    }


def run_reference_decomposition(
    *,
    source_dir: Path,
    output_dir: Path,
    reference_manifest: Path,
    reference_files: Path,
    reference_summary: Path,
    structure_manifest: Path,
    structure_files: Path,
    structure_summary: Path,
    structure_controls: Path,
    audit_paths: Sequence[Path],
    known_source_run: Path,
) -> dict[str, Any]:
    """検証済み入力全件を観測、再出力し、意味往復を記録する。"""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ReferenceDecompositionRunError(f"output directory is not empty: {output_dir}")
    reference_manifest_value, structure_manifest_value, sources = _verify_manifests(
        reference_manifest_path=Path(reference_manifest),
        reference_files=Path(reference_files),
        reference_summary=Path(reference_summary),
        structure_manifest_path=Path(structure_manifest),
        structure_files=Path(structure_files),
        structure_summary=Path(structure_summary),
        structure_controls=Path(structure_controls),
    )
    expected_names = {item["name"] for item in sources}
    actual_names = {path.name for path in source_dir.glob("*.mid")} | {
        path.name for path in source_dir.glob("*.MID")
    }
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise ReferenceDecompositionRunError(
            f"source set mismatch: missing={missing}, unexpected={unexpected}"
        )
    for item in sources:
        _verify_hash(source_dir / item["name"], item["sha256"], label="source SMF")
    known_source, known_source_smf = _known_source_outputs(Path(known_source_run))
    audit_inputs: dict[str, str] = {}
    for path in audit_paths:
        path = Path(path)
        if not path.is_file():
            raise ReferenceDecompositionRunError(f"corpus audit artifact is missing: {path}")
        audit_inputs[path.name] = sha256_file(path)

    output_dir.mkdir(parents=True, exist_ok=True)
    roundtrip_dir = output_dir / "roundtrip-smf"
    roundtrip_dir.mkdir(parents=True, exist_ok=True)
    observations: list[dict[str, Any]] = []
    roundtrips: list[dict[str, Any]] = []
    losses: list[dict[str, Any]] = []
    event_counts: Counter[str] = Counter()
    roundtrip_failure_count = 0
    for source in sources:
        name = source["name"]
        source_path = source_dir / name
        regenerated_path = roundtrip_dir / name
        try:
            observed = load_observed_smf(source_path)
            performance = build_observed_performance(observed)
            write_observed_smf(observed, regenerated_path)
            regenerated = load_observed_smf(regenerated_path)
            differences = observed_smf_differences(observed, regenerated)
            event_counts.update(event.message_type for event in observed.events)
            event_status = "pass" if not differences else "fail"
            performance_status = performance.note_matching_status
            if differences:
                roundtrip_failure_count += 1
            observations.append(
                {
                    "name": name,
                    "status": "affirmative_evidence"
                    if not differences
                    else "investigated_no_evidence",
                    "source_sha256": observed.source_sha256,
                    "ledger_sha256": observed.ledger_sha256,
                    "track_count": observed.track_count,
                    "event_count": len(observed.events),
                    "note_count": len(performance.notes),
                    "attack_group_count": len(performance.attack_groups),
                    "event_roundtrip_status": event_status,
                    "performance_status": performance_status,
                    "ambiguous_note_count": performance.ambiguous_note_count,
                    "unmatched_note_off_count": performance.unmatched_note_off_count,
                    "dangling_note_on_count": performance.dangling_note_on_count,
                }
            )
            roundtrips.append(
                {
                    "name": name,
                    "status": event_status,
                    "regenerated_sha256": regenerated.source_sha256,
                    "regenerated_ledger_sha256": regenerated.ledger_sha256,
                    "differences": [
                        {
                            "event_id": difference.event_id,
                            "changed_fields": list(difference.changed_fields),
                        }
                        for difference in differences
                    ],
                }
            )
            losses.append(
                {
                    "name": name,
                    "unsupported_event_count": 0,
                    "unsupported": [],
                }
            )
        except (OSError, EOFError, ValueError) as error:
            observations.append(
                {
                    "name": name,
                    "status": "unable_to_investigate",
                    "error": str(error),
                }
            )
            roundtrips.append({"name": name, "status": "unable_to_investigate"})
            losses.append(
                {
                    "name": name,
                    "unsupported_event_count": None,
                    "unsupported": [],
                    "error": str(error),
                }
            )

    observations.sort(key=lambda item: (item["name"].casefold(), item["name"]))
    roundtrips.sort(key=lambda item: (item["name"].casefold(), item["name"]))
    losses.sort(key=lambda item: (item["name"].casefold(), item["name"]))
    known_observed = load_observed_smf(known_source_smf)
    known_performance = build_observed_performance(known_observed)
    known_regenerated_path = output_dir / "known-source-roundtrip.mid"
    write_observed_smf(known_observed, known_regenerated_path)
    known_regenerated = load_observed_smf(known_regenerated_path)
    known_differences = observed_smf_differences(known_observed, known_regenerated)
    known_source["schema_version"] = 1
    known_source["status"] = (
        "affirmative_evidence" if not known_differences else "investigated_no_evidence"
    )
    known_source["observed"] = {
        "event_count": len(known_observed.events),
        "note_count": len(known_performance.notes),
        "performance_status": known_performance.note_matching_status,
        "event_roundtrip_status": "pass" if not known_differences else "fail",
        "regenerated_sha256": known_regenerated.source_sha256,
        "regenerated_ledger_sha256": known_regenerated.ledger_sha256,
        "differences": [
            {
                "event_id": difference.event_id,
                "changed_fields": list(difference.changed_fields),
            }
            for difference in known_differences
        ],
    }
    unable_to_investigate_count = sum(
        item.get("status") == "unable_to_investigate" for item in observations
    )
    status = (
        "pass"
        if not roundtrip_failure_count and not unable_to_investigate_count and not known_differences
        else "partial"
    )
    run_spec = {
        "schema_version": 1,
        "source_count": len(sources),
        "attack_group_window_us": 30_000,
        "inputs": {
            "reference_manifest": sha256_file(Path(reference_manifest)),
            "reference_files": sha256_file(Path(reference_files)),
            "reference_summary": sha256_file(Path(reference_summary)),
            "structure_manifest": sha256_file(Path(structure_manifest)),
            "structure_files": sha256_file(Path(structure_files)),
            "structure_summary": sha256_file(Path(structure_summary)),
            "structure_controls": sha256_file(Path(structure_controls)),
            "corpus_audit": audit_inputs,
            "known_source_run_state": known_source["run_state_sha256"],
        },
        "source_manifest_sha256": {item["name"]: item["sha256"] for item in sources},
        "verified_manifest_status": {
            "reference": reference_manifest_value.get("status"),
            "reference_schema_version": reference_manifest_value.get("schema_version"),
            "structure_schema_version": structure_manifest_value.get("schema_version"),
        },
    }
    event_support = {
        "schema_version": 1,
        "preserved": dict(sorted(event_counts.items())),
        "derived_only": {
            "note_matching": "track, channel and pitch local candidates",
            "attack_groups": "30 ms from the first note-on without chain merging",
            "absolute_time": "tempo-map projection from absolute ticks",
        },
        "unsupported": {},
    }
    result = {
        "status": status,
        "source_count": len(sources),
        "affirmative_evidence_count": sum(
            item.get("status") == "affirmative_evidence" for item in observations
        ),
        "investigated_no_evidence_count": sum(
            item.get("status") == "investigated_no_evidence" for item in observations
        ),
        "unable_to_investigate_count": unable_to_investigate_count,
        "event_roundtrip_pass_count": sum(
            item.get("event_roundtrip_status") == "pass" for item in observations
        ),
    }
    atomic_write_json(output_dir / "run-spec.json", run_spec)
    atomic_write_bytes(output_dir / "corpus-observations.jsonl", _jsonl_bytes(observations))
    atomic_write_json(output_dir / "event-support.json", event_support)
    atomic_write_json(output_dir / "known-source-controls.json", known_source)
    atomic_write_json(output_dir / "destructive-controls.json", _destructive_control_contract())
    atomic_write_bytes(output_dir / "loss-ledger.jsonl", _jsonl_bytes(losses))
    atomic_write_bytes(output_dir / "roundtrip.jsonl", _jsonl_bytes(roundtrips))
    atomic_write_json(output_dir / "result.json", result)
    output_names = (
        "run-spec.json",
        "corpus-observations.jsonl",
        "event-support.json",
        "known-source-controls.json",
        "destructive-controls.json",
        "loss-ledger.jsonl",
        "roundtrip.jsonl",
        "result.json",
    )
    manifest = {
        "schema_version": 1,
        "status": status,
        "outputs": {name: sha256_file(output_dir / name) for name in output_names},
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return result


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--reference-files", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--structure-manifest", type=Path, required=True)
    parser.add_argument("--structure-files", type=Path, required=True)
    parser.add_argument("--structure-summary", type=Path, required=True)
    parser.add_argument("--structure-controls", type=Path, required=True)
    parser.add_argument("--audit-path", type=Path, action="append", default=[])
    parser.add_argument("--known-source-run", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    result = run_reference_decomposition(
        source_dir=arguments.source_dir,
        output_dir=arguments.output_dir,
        reference_manifest=arguments.reference_manifest,
        reference_files=arguments.reference_files,
        reference_summary=arguments.reference_summary,
        structure_manifest=arguments.structure_manifest,
        structure_files=arguments.structure_files,
        structure_summary=arguments.structure_summary,
        structure_controls=arguments.structure_controls,
        audit_paths=arguments.audit_path,
        known_source_run=arguments.known_source_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
