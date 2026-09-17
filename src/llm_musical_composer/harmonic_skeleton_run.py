"""既知生成元でHarmonicSkeletonV0の投影と再合成を診断する。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llm_musical_composer.harmonic_skeleton import (
    apply_harmonic_skeleton,
    dump_harmonic_skeleton,
    parse_harmonic_skeleton,
    split_score_spec,
)
from llm_musical_composer.performance_pipeline import render_performance
from llm_musical_composer.pipeline_dsl import (
    parse_performance_spec,
    parse_piece_plan,
    parse_score_spec,
)
from llm_musical_composer.reference_timing_run import (
    _calibration_lineage_status,
    _completed_run_evidence_status,
    default_known_timing_sources,
)
from llm_musical_composer.run_state import (
    atomic_write_bytes,
    atomic_write_json,
    sha256_file,
)

_CASE_IDS = ("multiscale-v8", "reference-variance-v7-a2")
_OUTPUT_NAMES = (
    "run-spec.json",
    "sources.jsonl",
    "skeletons.jsonl",
    "payloads.jsonl",
    "roundtrip-results.jsonl",
    "result.json",
)


def _jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")


def _file_record(workspace: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.resolve().relative_to(workspace.resolve()).as_posix(),
        "sha256": sha256_file(path),
    }


def run_harmonic_skeleton_diagnostics(
    workspace: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """固定2設計族を診断し、決定的な成果物一式を保存する。"""

    workspace = Path(workspace).resolve()
    output_dir = Path(output_dir)
    available = {
        source.case_id: source for source in default_known_timing_sources(workspace)
    }
    source_records: list[dict[str, Any]] = []
    skeleton_records: list[dict[str, Any]] = []
    payload_records: list[dict[str, Any]] = []
    roundtrip_records: list[dict[str, Any]] = []
    for case_id in _CASE_IDS:
        source = available[case_id]
        plan = parse_piece_plan(source.piece_path.read_text(encoding="utf-8"))
        score = parse_score_spec(source.score_path.read_text(encoding="utf-8"))
        performance = parse_performance_spec(source.performance_path.read_text(encoding="utf-8"))
        rendered = render_performance(plan, score, performance)
        if source.evidence_kind == "calibration_result":
            evidence_status = _calibration_lineage_status(
                source.evidence_path, rendered.lineage
            )
        else:
            evidence_status = _completed_run_evidence_status(source)
        source_records.append(
            {
                "case_id": case_id,
                "family_id": source.group_id,
                "status": (
                    "assessed" if evidence_status == "pass" else "unable_to_investigate"
                ),
                "evidence_kind": source.evidence_kind,
                "evidence_status": evidence_status,
                "semantic_lineage": list(rendered.lineage),
                "files": {
                    "piece_plan": _file_record(workspace, source.piece_path),
                    "score_spec": _file_record(workspace, source.score_path),
                    "performance_spec": _file_record(workspace, source.performance_path),
                    "smf": _file_record(workspace, source.smf_path),
                    "evidence": _file_record(workspace, source.evidence_path),
                },
            }
        )
        if evidence_status != "pass":
            roundtrip_records.append(
                {
                    "case_id": case_id,
                    "family_id": source.group_id,
                    "status": "unable_to_investigate",
                    "reason": "source evidence did not pass",
                }
            )
            continue
        skeleton, payload = split_score_spec(plan, score)
        skeleton_source = dump_harmonic_skeleton(skeleton)
        parsed = parse_harmonic_skeleton(skeleton_source)
        restored = apply_harmonic_skeleton(plan, payload, parsed)
        exact = restored == score
        skeleton_records.append(
            {
                "case_id": case_id,
                "dsl": skeleton_source,
                "skeleton": asdict(skeleton),
            }
        )
        payload_records.append({"case_id": case_id, "payload": asdict(payload)})
        roundtrip_records.append(
            {
                "case_id": case_id,
                "family_id": source.group_id,
                "status": "pass" if exact and parsed == skeleton else "fail",
                "material_count": len(skeleton.materials),
                "harmony_count": sum(
                    len(material.harmonies) for material in skeleton.materials
                ),
                "unharmonized_material_count": sum(
                    not material.harmonies for material in skeleton.materials
                ),
                "dsl_roundtrip_exact": parsed == skeleton,
                "score_roundtrip_exact": exact,
            }
        )
    assessed = [
        record
        for record in roundtrip_records
        if record["status"] != "unable_to_investigate"
    ]
    passed = [record for record in assessed if record["status"] == "pass"]
    families = {record["family_id"] for record in passed}
    status = (
        "pass"
        if len(assessed) == len(_CASE_IDS) and len(passed) == len(_CASE_IDS) and len(families) == 2
        else "fail"
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "case_count": len(_CASE_IDS),
        "assessed_case_count": len(assessed),
        "passed_case_count": len(passed),
        "source_family_count": 2,
        "passed_family_count": len(families),
    }
    atomic_write_json(
        output_dir / "run-spec.json",
        {
            "schema_version": 1,
            "protocol_id": "harmonic-skeleton-projection-v0",
            "case_ids": list(_CASE_IDS),
            "implementation": _file_record(
                workspace, workspace / "src/llm_musical_composer/harmonic_skeleton.py"
            ),
            "runner": _file_record(workspace, Path(__file__)),
        },
    )
    atomic_write_bytes(output_dir / "sources.jsonl", _jsonl_bytes(source_records))
    atomic_write_bytes(output_dir / "skeletons.jsonl", _jsonl_bytes(skeleton_records))
    atomic_write_bytes(output_dir / "payloads.jsonl", _jsonl_bytes(payload_records))
    atomic_write_bytes(
        output_dir / "roundtrip-results.jsonl", _jsonl_bytes(roundtrip_records)
    )
    atomic_write_json(output_dir / "result.json", result)
    atomic_write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "status": status,
            "outputs": {name: sha256_file(output_dir / name) for name in _OUTPUT_NAMES},
        },
    )
    return result


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    result = run_harmonic_skeleton_diagnostics(arguments.workspace, arguments.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
