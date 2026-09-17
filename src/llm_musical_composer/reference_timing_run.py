"""保存済み生成元IRを現在のレンダラーで再生成し、時間互換性を調べる。"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_musical_composer.performance_pipeline import (
    performance_time_map,
    render_performance,
    render_performance_smf,
)
from llm_musical_composer.pipeline_dsl import (
    parse_performance_spec,
    parse_piece_plan,
    parse_score_spec,
)
from llm_musical_composer.reference_decomposition import (
    build_observed_performance,
    load_observed_smf,
    observed_smf_differences,
)
from llm_musical_composer.reference_timing_hypothesis import (
    align_known_score_timing,
    assess_sequential_performance_stage,
    build_timing_control_artifacts,
)
from llm_musical_composer.run_state import atomic_write_bytes, atomic_write_json, sha256_file
from llm_musical_composer.score_timing_hypothesis import (
    build_grouping_profiles,
    compare_candidate_to_known_score,
    compare_grouping_to_known_score,
    generate_score_timing_candidates,
)
from llm_musical_composer.score_timing_oracle_diagnostics import (
    fit_equal_piecewise_time_map,
    fit_piecewise_time_map,
)


@dataclass(frozen=True)
class KnownTimingSource:
    case_id: str
    group_id: str
    root: Path
    piece_path: Path
    score_path: Path
    performance_path: Path
    smf_path: Path
    evidence_kind: str
    evidence_path: Path
    oracle_basis: str = "generator_voice_labels"
    oracle_observation_dependency: str = "unknown"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _jsonl_bytes(records: Sequence[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")


def _calibration_lineage_status(evidence_path: Path, rendered_lineage: tuple[str, str, str]) -> str:
    evidence = _read_json(evidence_path)
    if evidence.get("status") != "passed":
        return "fail"
    lineage = evidence.get("lineage")
    return "pass" if lineage == list(rendered_lineage) else "fail"


def _completed_run_evidence_status(source: KnownTimingSource) -> str:
    state = _read_json(source.evidence_path)
    if state.get("status") != "completed" or not isinstance(state.get("steps"), dict):
        return "fail"
    expected_paths = {
        str(path.resolve().relative_to(source.root.resolve())).replace("/", "\\").casefold(): path
        for path in (
            source.piece_path,
            source.score_path,
            source.performance_path,
            source.smf_path,
        )
    }
    verified: set[str] = set()
    for step in state["steps"].values():
        if not isinstance(step, dict) or step.get("status") != "completed":
            continue
        outputs = step.get("outputs")
        if not isinstance(outputs, dict):
            continue
        for path_key, hash_key in (("path", "sha256"), ("smf_path", "smf_sha256")):
            relative = outputs.get(path_key)
            expected_hash = outputs.get(hash_key)
            if not isinstance(relative, str) or not isinstance(expected_hash, str):
                continue
            normalized = relative.replace("/", "\\").casefold()
            path = expected_paths.get(normalized)
            if path is not None and sha256_file(path).lower() == expected_hash.lower():
                verified.add(normalized)
    return "pass" if verified == set(expected_paths) else "fail"


def _fraction_record(value: object) -> dict[str, int | float]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "value": round(float(value), 6),
    }


def _time_fit_record(fit: object) -> dict[str, Any]:
    return {
        "requested_segment_count": fit.requested_segment_count,
        "actual_segment_count": fit.actual_segment_count,
        "boundaries": list(fit.boundaries),
        "anchor_squared_error": _fraction_record(fit.anchor_squared_error),
        "anchor_mean_absolute_error": _fraction_record(fit.anchor_mean_absolute_error),
        "anchor_maximum_absolute_error": _fraction_record(fit.anchor_maximum_absolute_error),
    }


def _candidate_recovery(
    timing_evidence: object,
    observed_performance: object,
    ledger_sha256: str,
    *,
    true_time_map_ms: tuple[int, ...],
    renderer_matches_source: bool,
) -> dict[str, Any]:
    known_units: dict[str, int] = {}
    for group in timing_evidence.attack_groups:
        for event_id in group.evidence_event_ids:
            previous = known_units.get(event_id)
            if previous is not None and previous != group.score_unit:
                return {
                    "status": "unable_to_investigate",
                    "reason": "known score evidence maps one event to multiple positions",
                }
            known_units[event_id] = group.score_unit
    candidate_set = generate_score_timing_candidates(
        observed_performance,
        source_ledger_sha256=ledger_sha256,
    )
    if candidate_set.status != "assessed":
        return {
            "status": candidate_set.status,
            "reason": candidate_set.reason,
            "candidate_count": len(candidate_set.candidates),
            "equivalent_candidate_count": 0,
            "equivalent_candidate_ids": [],
        }
    profiles = build_grouping_profiles(observed_performance)
    grouping_diagnostics = {
        profile_id: compare_grouping_to_known_score(groups, known_units)
        for profile_id, groups in profiles.items()
    }
    time_diagnostics: dict[str, Any] = {
        "status": "assessed",
        "renderer_matches_source": renderer_matches_source,
        "profiles": {},
    }
    time_fit_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], dict[str, Any]] = {}
    for profile_id, groups in profiles.items():
        if grouping_diagnostics[profile_id]["status"] != "compatible":
            continue
        positions = tuple(known_units[group.source_event_ids[0]] for group in groups)
        observed_times = tuple(group.onset_us for group in groups)
        oracle_times = tuple(true_time_map_ms[position] * 1_000 for position in positions)
        cache_key = (positions, observed_times)
        if cache_key not in time_fit_cache:
            anchor_errors = tuple(
                abs(observed - oracle)
                for observed, oracle in zip(observed_times, oracle_times, strict=True)
            )
            segment_records = {}
            for count in (1, 4, 8):
                segment_records[str(count)] = {
                    "renderer_equal": _time_fit_record(
                        fit_equal_piecewise_time_map(
                            positions,
                            oracle_times,
                            requested_segment_count=count,
                        )
                    ),
                    "renderer_dynamic": _time_fit_record(
                        fit_piecewise_time_map(
                            positions,
                            oracle_times,
                            requested_segment_count=count,
                        )
                    ),
                    "source_dynamic": _time_fit_record(
                        fit_piecewise_time_map(
                            positions,
                            observed_times,
                            requested_segment_count=count,
                        )
                    ),
                }
            time_fit_cache[cache_key] = {
                "group_count": len(groups),
                "observed_vs_renderer_anchor_mean_absolute_error_us": round(
                    statistics.fmean(anchor_errors), 3
                ),
                "observed_vs_renderer_anchor_maximum_absolute_error_us": max(anchor_errors),
                "segment_counts": segment_records,
            }
        time_diagnostics["profiles"][profile_id] = time_fit_cache[cache_key]
    comparisons = [
        (
            candidate.candidate_id,
            compare_candidate_to_known_score(
                candidate,
                profiles[candidate.grouping_profile_id],
                known_units,
            ),
        )
        for candidate in candidate_set.candidates
    ]
    equivalent = [
        candidate_id
        for candidate_id, comparison in comparisons
        if comparison["status"] == "equivalent"
    ]
    status_counts = Counter(comparison["status"] for _, comparison in comparisons)
    match_rates = [
        float(comparison["best_scale_interval_match_rate"])
        for _, comparison in comparisons
        if "best_scale_interval_match_rate" in comparison
    ]
    best_match_rate = max(match_rates, default=0.0)
    best_match_ids = [
        candidate_id
        for candidate_id, comparison in comparisons
        if comparison.get("best_scale_interval_match_rate") == best_match_rate
    ]
    best_id = best_match_ids[0] if best_match_ids else None
    best_diagnostics = next(
        (comparison for candidate_id, comparison in comparisons if candidate_id == best_id),
        {},
    )
    return {
        "status": "recovered" if equivalent else "not_recovered",
        "candidate_count": len(comparisons),
        "equivalent_candidate_count": len(equivalent),
        "equivalent_candidate_ids": equivalent,
        "comparison_status_counts": dict(sorted(status_counts.items())),
        "grouping_diagnostics": grouping_diagnostics,
        "time_map_oracle_diagnostics": time_diagnostics,
        "best_scale_interval_match_rate": best_match_rate,
        "best_match_candidate_ids": best_match_ids,
        "best_match_diagnostics": {
            "best_scale": best_diagnostics.get("best_scale"),
            "interval_count": best_diagnostics.get("interval_count"),
            "mismatched_intervals": best_diagnostics.get("mismatched_intervals", []),
        },
        "known_score_was_not_search_input": True,
    }


def _source_record(source: KnownTimingSource, regenerated_path: Path) -> dict[str, Any]:
    paths = (
        source.piece_path,
        source.score_path,
        source.performance_path,
        source.smf_path,
        source.evidence_path,
    )
    if any(not Path(path).is_file() for path in paths):
        missing = [str(path) for path in paths if not Path(path).is_file()]
        return {
            "case_id": source.case_id,
            "group_id": source.group_id,
            "status": "unable_to_investigate",
            "missing": missing,
        }
    try:
        plan = parse_piece_plan(source.piece_path.read_text(encoding="utf-8"))
        score = parse_score_spec(source.score_path.read_text(encoding="utf-8"))
        performance = parse_performance_spec(source.performance_path.read_text(encoding="utf-8"))
        rendered = render_performance(plan, score, performance)
        render_performance_smf(rendered, regenerated_path)
        source_observed = load_observed_smf(source.smf_path)
        regenerated_observed = load_observed_smf(regenerated_path)
        differences = observed_smf_differences(source_observed, regenerated_observed)
        source_performance = build_observed_performance(source_observed)
        timing_evidence = align_known_score_timing(
            plan,
            score,
            source_observed,
            source_performance,
        )
        coordination_counts = Counter(
            group.coordination_status for group in timing_evidence.attack_groups
        )
        feature_statuses = {
            "note_on": _event_subset_status(source_observed, regenerated_observed, "note_on"),
            "note_off": _event_subset_status(source_observed, regenerated_observed, "note_off"),
            "cc64": _event_subset_status(source_observed, regenerated_observed, "cc64"),
        }
        if source.evidence_kind == "calibration_result":
            lineage_status = _calibration_lineage_status(source.evidence_path, rendered.lineage)
        elif source.evidence_kind == "completed_run":
            lineage_status = _completed_run_evidence_status(source)
        else:
            raise ValueError(f"unsupported evidence kind: {source.evidence_kind}")
        event_status = "pass" if not differences else "fail"
        candidate_recovery = _candidate_recovery(
            timing_evidence,
            source_performance,
            source_observed.ledger_sha256,
            true_time_map_ms=performance_time_map(plan, score, performance),
            renderer_matches_source=event_status == "pass",
        )
        status = (
            "affirmative_evidence"
            if lineage_status == "pass" and event_status == "pass"
            else "investigated_no_evidence"
        )
        return {
            "case_id": source.case_id,
            "group_id": source.group_id,
            "status": status,
            "evidence_kind": source.evidence_kind,
            "lineage_status": lineage_status,
            "event_roundtrip_status": event_status,
            "timing_roundtrip_status": (
                "pass"
                if feature_statuses["note_on"] == feature_statuses["note_off"] == "pass"
                else "fail"
            ),
            "feature_roundtrip_status": feature_statuses,
            "known_score_timing": {
                "status": timing_evidence.status,
                "note_on_status": timing_evidence.note_on_status,
                "note_off_status": timing_evidence.note_off_status,
                "cc64_status": timing_evidence.cc64_status,
                "attack_group_count": len(timing_evidence.attack_groups),
                "coordination_status_counts": dict(sorted(coordination_counts.items())),
                "maximum_invariant_spread_us": max(
                    (group.invariant_spread_us for group in timing_evidence.attack_groups),
                    default=0,
                ),
                "unmatched_expected_count": timing_evidence.unmatched_expected_count,
                "unmatched_observed_count": timing_evidence.unmatched_observed_count,
                "assumptions": list(timing_evidence.assumptions),
            },
            "score_timing_candidate_recovery": candidate_recovery,
            "source_hashes": {
                "piece": sha256_file(source.piece_path),
                "score": sha256_file(source.score_path),
                "performance": sha256_file(source.performance_path),
                "smf": sha256_file(source.smf_path),
                "evidence": sha256_file(source.evidence_path),
            },
            "regenerated_smf_sha256": sha256_file(regenerated_path),
            "differences": [
                {
                    "event_id": difference.event_id,
                    "changed_fields": list(difference.changed_fields),
                }
                for difference in differences
            ],
        }
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        return {
            "case_id": source.case_id,
            "group_id": source.group_id,
            "status": "unable_to_investigate",
            "error": str(error),
        }


def _event_subset_status(first: object, second: object, subset: str) -> str:
    def selected(observed: object) -> tuple[tuple[object, ...], ...]:
        rows: list[tuple[object, ...]] = []
        for event in observed.events:
            velocity = int(event.field("velocity", 0))
            is_attack = event.message_type == "note_on" and velocity > 0
            is_release = event.message_type == "note_off" or (
                event.message_type == "note_on" and velocity == 0
            )
            is_cc64 = event.message_type == "control_change" and event.field("control") == 64
            if not {
                "note_on": is_attack,
                "note_off": is_release,
                "cc64": is_cc64,
            }[subset]:
                continue
            rows.append(
                (
                    event.track,
                    event.absolute_tick,
                    event.message_type,
                    event.fields,
                )
            )
        return tuple(rows)

    return "pass" if selected(first) == selected(second) else "fail"


def run_known_timing_compatibility(
    *, sources: Sequence[KnownTimingSource], output_dir: Path
) -> dict[str, Any]:
    """複数の既知生成元を同じ処理で再生成して分類する。"""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    if not sources:
        raise ValueError("known timing sources must not be empty")
    case_ids = [source.case_id for source in sources]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("known timing source case IDs must be unique")
    output_dir.mkdir(parents=True, exist_ok=True)
    regenerated_dir = output_dir / "regenerated-smf"
    regenerated_dir.mkdir()
    records = [
        _source_record(source, regenerated_dir / f"{source.case_id}.mid")
        for source in sorted(sources, key=lambda item: item.case_id)
    ]
    controls = build_timing_control_artifacts()
    affirmative = sum(item["status"] == "affirmative_evidence" for item in records)
    no_evidence = sum(item["status"] == "investigated_no_evidence" for item in records)
    unable = sum(item["status"] == "unable_to_investigate" for item in records)
    status = "pass" if affirmative == len(records) else "partial"
    run_spec = {
        "schema_version": 1,
        "source_count": len(records),
        "group_count": len({source.group_id for source in sources}),
        "sources": [
            {
                "case_id": source.case_id,
                "group_id": source.group_id,
                "root": str(source.root.resolve()),
                "evidence_kind": source.evidence_kind,
            }
            for source in sorted(sources, key=lambda item: item.case_id)
        ],
    }
    result = {
        "status": status,
        "source_count": len(records),
        "group_count": run_spec["group_count"],
        "affirmative_evidence_count": affirmative,
        "investigated_no_evidence_count": no_evidence,
        "unable_to_investigate_count": unable,
        "timing_roundtrip_pass_count": sum(
            item.get("timing_roundtrip_status") == "pass" for item in records
        ),
        "cc64_roundtrip_pass_count": sum(
            item.get("feature_roundtrip_status", {}).get("cc64") == "pass" for item in records
        ),
        "known_score_timing_assessed_count": sum(
            item.get("known_score_timing", {}).get("status") == "assessed" for item in records
        ),
        "known_score_note_on_assessed_count": sum(
            item.get("known_score_timing", {}).get("note_on_status") == "assessed"
            for item in records
        ),
        "score_timing_candidate_recovered_count": sum(
            item.get("score_timing_candidate_recovery", {}).get("status") == "recovered"
            for item in records
        ),
    }
    sequential_assessment = {
        **assess_sequential_performance_stage(),
        "known_score_note_on_assessed_count": result["known_score_note_on_assessed_count"],
        "known_score_source_count": len(records),
    }
    atomic_write_json(output_dir / "run-spec.json", run_spec)
    atomic_write_bytes(output_dir / "known-source-availability.jsonl", _jsonl_bytes(records))
    atomic_write_bytes(
        output_dir / "known-score-results.jsonl",
        _jsonl_bytes(
            [
                {
                    "case_id": item["case_id"],
                    "group_id": item["group_id"],
                    **item["known_score_timing"],
                }
                for item in records
                if "known_score_timing" in item
            ]
        ),
    )
    atomic_write_json(
        output_dir / "finite-vocabulary-collisions.json",
        controls["finite_vocabulary_collisions"],
    )
    atomic_write_json(
        output_dir / "non-identifiability-controls.json",
        controls["non_identifiability"],
    )
    atomic_write_json(
        output_dir / "sequential-stage-assessment.json",
        sequential_assessment,
    )
    atomic_write_json(output_dir / "result.json", result)
    output_names = (
        "known-source-availability.jsonl",
        "known-score-results.jsonl",
        "finite-vocabulary-collisions.json",
        "non-identifiability-controls.json",
        "result.json",
        "run-spec.json",
        "sequential-stage-assessment.json",
    )
    manifest = {
        "schema_version": 1,
        "status": status,
        "outputs": {name: sha256_file(output_dir / name) for name in output_names},
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return result


def default_known_timing_sources(workspace: Path = Path(".")) -> tuple[KnownTimingSource, ...]:
    workspace = Path(workspace)
    sources: list[KnownTimingSource] = []
    for version in range(4, 9):
        root = workspace / f".appendix/multiscale-calibration-run-v{version}"
        sources.append(
            KnownTimingSource(
                case_id=f"multiscale-v{version}",
                group_id="multiscale-calibration-family",
                root=root,
                piece_path=root / "inputs/piece-plan.music.py",
                score_path=root / "inputs/score.music.py",
                performance_path=root / "inputs/performance.music.py",
                smf_path=root / "outputs/final.mid",
                evidence_kind="calibration_result",
                evidence_path=root / "result.json",
                oracle_basis="generator_voice_labels",
                oracle_observation_dependency="absent",
            )
        )
    root = workspace / ".appendix/reference-variance-smoke-v7/runs/reference-a-candidate-2"
    sources.append(
        KnownTimingSource(
            case_id="reference-variance-v7-a2",
            group_id="reference-variance-completed-run",
            root=root,
            piece_path=root / "outputs/piece-plan.dsl",
            score_path=root / "outputs/score-spec.dsl",
            performance_path=root / "outputs/performance-spec.dsl",
            smf_path=root / "outputs/final.mid",
            evidence_kind="completed_run",
            evidence_path=root / "run-state.json",
            oracle_basis="generator_voice_labels",
            oracle_observation_dependency="present",
        )
    )
    return tuple(sources)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    result = run_known_timing_compatibility(
        sources=default_known_timing_sources(arguments.workspace),
        output_dir=arguments.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
