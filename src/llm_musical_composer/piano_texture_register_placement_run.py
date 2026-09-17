"""PianoTextureSpecV2の固定複数素材診断を保存する。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llm_musical_composer.generic_pipeline_quality import evaluate_generic_score_quality
from llm_musical_composer.performance_pipeline import (
    PiecePlan,
    PlanNode,
    ScoreHarmony,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
    render_performance,
)
from llm_musical_composer.piano_texture_pilot import (
    parse_piano_texture_spec,
    validate_and_convert_piano_texture,
)
from llm_musical_composer.piano_texture_register_placement import (
    PianoTextureEventV2,
    PianoTexturePlacementResult,
    PianoTextureSpecV2,
    combine_piano_texture_v2,
    convert_piano_texture_v1_to_v2,
    place_piano_texture_v2,
    project_accompaniment_to_v2,
)
from llm_musical_composer.pipeline_dsl import (
    parse_performance_spec,
    parse_piece_plan,
    parse_score_spec,
)
from llm_musical_composer.recurrence_quality import analyze_material_harmony
from llm_musical_composer.reference_timing_run import (
    KnownTimingSource,
    _calibration_lineage_status,
    _completed_run_evidence_status,
    default_known_timing_sources,
)
from llm_musical_composer.run_state import (
    atomic_write_bytes,
    atomic_write_json,
    sha256_file,
)

_OUTPUT_NAMES = (
    "run-spec.json",
    "sources.jsonl",
    "projected-textures.jsonl",
    "placed-textures.jsonl",
    "case-results.jsonl",
    "result.json",
)
_KNOWN_CASES = (
    ("multiscale-v8", "a2-return-prime"),
    ("multiscale-v8", "b-theme"),
    ("reference-variance-v7-a2", "material_1"),
    ("reference-variance-v7-a2", "material_7"),
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")


def _relative(workspace: Path, path: Path) -> str:
    return path.resolve().relative_to(workspace.resolve()).as_posix()


def _file_record(workspace: Path, path: Path) -> dict[str, str]:
    return {"path": _relative(workspace, path), "sha256": sha256_file(path)}


def _load_known_source(
    source: KnownTimingSource,
) -> tuple[PiecePlan, ScoreSpec, str, tuple[str, str, str]]:
    plan = parse_piece_plan(source.piece_path.read_text(encoding="utf-8"))
    score = parse_score_spec(source.score_path.read_text(encoding="utf-8"))
    performance = parse_performance_spec(source.performance_path.read_text(encoding="utf-8"))
    rendered = render_performance(plan, score, performance)
    if source.evidence_kind == "calibration_result":
        evidence_status = _calibration_lineage_status(source.evidence_path, rendered.lineage)
    elif source.evidence_kind == "completed_run":
        evidence_status = _completed_run_evidence_status(source)
    else:
        evidence_status = "fail"
    return plan, score, evidence_status, rendered.lineage


def _target_harmony_record(score: ScoreSpec, material_id: str) -> dict[str, Any]:
    result = analyze_material_harmony(
        score,
        material_id,
        low_pitch_boundary=48,
        minimum_low_spacing_semitones=7,
    )
    return asdict(result)


def _diagnose_texture(
    case_id: str,
    family_id: str,
    plan: PiecePlan,
    score: ScoreSpec,
    texture: PianoTextureSpecV2,
    *,
    source_status: str,
) -> tuple[dict[str, Any], PianoTexturePlacementResult | None]:
    baseline = evaluate_generic_score_quality(plan, score)
    if source_status != "pass":
        return (
            {
                "case_id": case_id,
                "family_id": family_id,
                "status": "unable_to_investigate",
                "reason": "source evidence did not pass",
                "baseline_failures": baseline["failures"],
            },
            None,
        )
    placement = place_piano_texture_v2(plan, score, texture)
    if placement.status != "placed":
        return (
            {
                "case_id": case_id,
                "family_id": family_id,
                "status": placement.status,
                "reason": placement.reason,
                "failed_onset": placement.failed_onset,
                "baseline_failures": baseline["failures"],
            },
            placement,
        )
    combined = combine_piano_texture_v2(plan, score, placement)
    after = evaluate_generic_score_quality(plan, combined)
    baseline_failures = set(baseline["failures"])
    after_failures = set(after["failures"])
    harmony = _target_harmony_record(combined, texture.material_id)
    passed = (
        placement.low_spacing_violations == 0
        and harmony["passes"]
        and harmony["accompaniment_chord_tone_ratio"] == 1.0
        and not (after_failures - baseline_failures)
    )
    return (
        {
            "case_id": case_id,
            "family_id": family_id,
            "status": "pass" if passed else "fail",
            "placement_status": placement.status,
            "event_count": len(texture.events),
            "low_spacing_violations": placement.low_spacing_violations,
            "baseline_failures": sorted(baseline_failures),
            "after_failures": sorted(after_failures),
            "new_failures": sorted(after_failures - baseline_failures),
            "target_harmony": harmony,
        },
        placement,
    )


def _verify_v1_inputs(workspace: Path, root: Path) -> tuple[str, dict[str, Any]]:
    manifest = _read_json(root / "manifest.json")
    selected = (
        "input_piece_plan",
        "input_score_spec",
        "piano_texture",
        "prompt",
        "response",
    )
    checks: dict[str, Any] = {}
    for key in selected:
        item = manifest["files"][key]
        path = Path(item["path"])
        if not path.is_absolute():
            path = root / path
        actual = sha256_file(path)
        checks[key] = {
            "path": _relative(workspace, path),
            "expected_sha256": item["sha256"],
            "actual_sha256": actual,
            "status": "pass" if actual == item["sha256"] else "fail",
        }
    status = "pass" if all(item["status"] == "pass" for item in checks.values()) else "fail"
    return status, checks


def _direct_regression(
    workspace: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = workspace / ".appendix/piano-texture-boundary-pilot-v1"
    evidence_status, checks = _verify_v1_inputs(workspace, root)
    plan = parse_piece_plan((root / "input-piece-plan.dsl").read_text(encoding="utf-8"))
    score = parse_score_spec((root / "input-score-spec.dsl").read_text(encoding="utf-8"))
    v1 = parse_piano_texture_spec((root / "piano-texture.dsl").read_text(encoding="utf-8"))
    v1_notes = validate_and_convert_piano_texture(plan, score, v1)
    texture = convert_piano_texture_v1_to_v2(plan, score, v1)
    result, placement = _diagnose_texture(
        "piano-texture-v1-regression",
        "piano-texture-v1-external-response",
        plan,
        score,
        texture,
        source_status=evidence_status,
    )
    assert placement is not None
    v1_by_id = {note.event_id: note for note in v1_notes}
    placed_by_id = {note.event_id: note for note in placement.notes}
    symbolic_preserved = all(
        (
            event.event_id in placed_by_id
            and placed_by_id[event.event_id].at_units == event.at_units
            and placed_by_id[event.event_id].duration_units == event.duration_units
            and placed_by_id[event.event_id].articulations == event.articulations
        )
        for event in v1.events
    )
    original_spacing = sum(
        1
        for onset in sorted({note.at_units for note in v1_notes})
        if (
            len(
                pitches := sorted(
                    {
                        note.pitch
                        for note in v1_notes
                        if note.at_units <= onset < note.at_units + note.duration_units
                    }
                )
            )
            >= 2
            and pitches[0] < 48
            and pitches[1] - pitches[0] < 7
        )
    )
    result.update(
        {
            "source_evidence_status": evidence_status,
            "symbolic_events_preserved": symbolic_preserved,
            "original_low_spacing_violations": original_spacing,
            "pitch_changed_event_count": sum(
                v1_by_id[event_id].pitch != placed_by_id[event_id].pitch
                for event_id in v1_by_id
            ),
        }
    )
    if not symbolic_preserved or original_spacing < 1:
        result["status"] = "fail"
    source = {
        "case_id": "piano-texture-v1-regression",
        "family_id": "piano-texture-v1-external-response",
        "status": "assessed" if evidence_status == "pass" else "unable_to_investigate",
        "evidence_checks": checks,
    }
    return source, {"case_id": result["case_id"], "texture": asdict(texture)}, {
        "case_id": result["case_id"],
        "placement": asdict(placement),
    }, result


def _known_cases(
    workspace: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    available = {source.case_id: source for source in default_known_timing_sources(workspace)}
    loaded: dict[str, tuple[PiecePlan, ScoreSpec, str, tuple[str, str, str]]] = {}
    source_records: list[dict[str, Any]] = []
    projected_records: list[dict[str, Any]] = []
    placed_records: list[dict[str, Any]] = []
    case_records: list[dict[str, Any]] = []
    for case_id, material_id in _KNOWN_CASES:
        source = available[case_id]
        if case_id not in loaded:
            loaded[case_id] = _load_known_source(source)
            _, _, evidence_status, lineage = loaded[case_id]
            source_records.append(
                {
                    "case_id": case_id,
                    "family_id": source.group_id,
                    "status": "assessed" if evidence_status == "pass" else "unable_to_investigate",
                    "evidence_kind": source.evidence_kind,
                    "evidence_status": evidence_status,
                    "lineage": list(lineage),
                    "files": {
                        "piece_plan": _file_record(workspace, source.piece_path),
                        "score_spec": _file_record(workspace, source.score_path),
                        "performance_spec": _file_record(workspace, source.performance_path),
                        "smf": _file_record(workspace, source.smf_path),
                        "evidence": _file_record(workspace, source.evidence_path),
                    },
                }
            )
        plan, score, evidence_status, _ = loaded[case_id]
        projected = project_accompaniment_to_v2(plan, score, material_id)
        projected_records.append(
            {
                "case_id": f"{case_id}:{material_id}",
                "projection": asdict(projected),
            }
        )
        if projected.status != "assessed":
            case_records.append(
                {
                    "case_id": f"{case_id}:{material_id}",
                    "family_id": source.group_id,
                    "status": "unable_to_investigate",
                    "reason": projected.reason,
                }
            )
            continue
        result, placement = _diagnose_texture(
            f"{case_id}:{material_id}",
            source.group_id,
            plan,
            score,
            projected.texture,
            source_status=evidence_status,
        )
        case_records.append(result)
        if placement is not None:
            placed_records.append(
                {"case_id": result["case_id"], "placement": asdict(placement)}
            )
    return source_records, projected_records, placed_records, case_records


def _fixture_plan_score(*, foreground_voice: str, quality: str) -> tuple[PiecePlan, ScoreSpec]:
    plan = PiecePlan(
        "fixture-plan",
        "配置fixture",
        0,
        "major",
        "whole",
        "tonic",
        (
            PlanNode("whole", None, 0, "whole"),
            PlanNode(
                "body",
                "whole",
                0,
                "statement",
                duration_weight=1,
                score_material_id="body",
            ),
        ),
    )
    foreground_pitch = 72 if foreground_voice == "upper" else 36
    accompaniment_voice = "lower" if foreground_voice == "upper" else "upper"
    score = ScoreSpec(
        "fixture-score",
        8,
        (
            ScoreMaterial(
                "body",
                8,
                (
                    ScoreNote("foreground", 0, 8, foreground_pitch, foreground_voice),
                    ScoreNote("old", 0, 8, 48, accompaniment_voice),
                ),
                harmonies=(ScoreHarmony("h1", 0, 8, 0, quality),),
                foreground_voice=foreground_voice,
            ),
        ),
    )
    return plan, score


def _fixtures() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    projected: list[dict[str, Any]] = []
    placed: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    definitions = (
        (
            "fixture-major-seventh",
            *_fixture_plan_score(foreground_voice="upper", quality="major-seventh"),
            PianoTextureSpecV2(
                "body",
                (
                    PianoTextureEventV2(
                        "seventh", "h1", "accompaniment", "lower", 0, 8, "seventh", "low"
                    ),
                ),
            ),
            "placed",
        ),
        (
            "fixture-voice-crossing",
            *_fixture_plan_score(foreground_voice="lower", quality="major"),
            PianoTextureSpecV2(
                "body",
                (
                    PianoTextureEventV2(
                        "crossing", "h1", "accompaniment", "upper", 0, 8, "root", "bass"
                    ),
                ),
            ),
            "constraint_unplaceable",
        ),
    )
    for case_id, plan, score, texture, expected_status in definitions:
        placement = place_piano_texture_v2(plan, score, texture)
        projected.append({"case_id": case_id, "texture": asdict(texture)})
        placed.append({"case_id": case_id, "placement": asdict(placement)})
        results.append(
            {
                "case_id": case_id,
                "family_id": "artificial-fixture",
                "status": "pass" if placement.status == expected_status else "fail",
                "expected_placement_status": expected_status,
                "actual_placement_status": placement.status,
                "counts_as_source_family_evidence": False,
            }
        )
    return projected, placed, results


def run_register_placement_diagnostics(
    workspace: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """固定対照を診断し、決定的な成果物一式を保存する。"""

    workspace = Path(workspace).resolve()
    output_dir = Path(output_dir)
    direct_source, direct_projected, direct_placed, direct_case = _direct_regression(workspace)
    sources, projected, placed, cases = _known_cases(workspace)
    fixture_projected, fixture_placed, fixture_cases = _fixtures()
    sources.insert(0, direct_source)
    projected.insert(0, direct_projected)
    projected.extend(fixture_projected)
    placed.insert(0, direct_placed)
    placed.extend(fixture_placed)
    all_cases = [direct_case, *cases, *fixture_cases]
    known_passes = [case for case in cases if case["status"] == "pass"]
    passed_families = {case["family_id"] for case in known_passes}
    status = (
        "pass"
        if direct_case["status"] == "pass"
        and len(known_passes) == len(_KNOWN_CASES)
        and len(passed_families) == 2
        and all(case["status"] == "pass" for case in fixture_cases)
        else "fail"
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "direct_regression": {
            "case_count": 1,
            "passed_case_count": int(direct_case["status"] == "pass"),
        },
        "known_sources": {
            "case_count": len(_KNOWN_CASES),
            "assessed_case_count": sum(case["status"] != "unable_to_investigate" for case in cases),
            "passed_case_count": len(known_passes),
            "source_family_count": 2,
            "passed_family_count": len(passed_families),
        },
        "fixtures": {
            "case_count": len(fixture_cases),
            "passed_case_count": sum(case["status"] == "pass" for case in fixture_cases),
            "counts_as_source_family_evidence": False,
        },
    }
    run_spec = {
        "schema_version": 1,
        "protocol_id": "piano-texture-register-placement-v2",
        "known_cases": [
            {"source_case_id": case_id, "material_id": material_id}
            for case_id, material_id in _KNOWN_CASES
        ],
        "register_zones": {
            "bass": [21, 47, 36],
            "low": [36, 59, 48],
            "middle": [48, 71, 60],
            "high": [60, 108, 72],
        },
        "implementation": _file_record(
            workspace,
            workspace / "src/llm_musical_composer/piano_texture_register_placement.py",
        ),
        "runner": _file_record(workspace, Path(__file__)),
    }
    atomic_write_json(output_dir / "run-spec.json", run_spec)
    atomic_write_bytes(output_dir / "sources.jsonl", _jsonl_bytes(sources))
    atomic_write_bytes(output_dir / "projected-textures.jsonl", _jsonl_bytes(projected))
    atomic_write_bytes(output_dir / "placed-textures.jsonl", _jsonl_bytes(placed))
    atomic_write_bytes(output_dir / "case-results.jsonl", _jsonl_bytes(all_cases))
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
    result = run_register_placement_diagnostics(arguments.workspace, arguments.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
