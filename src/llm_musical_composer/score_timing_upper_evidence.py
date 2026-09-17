"""ScoreTiming候補から、音高形とリズム反復の候補感度を診断する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any

from llm_musical_composer.pipeline_dsl import (
    parse_performance_spec,
    parse_piece_plan,
    parse_score_spec,
)
from llm_musical_composer.reference_decomposition import (
    ObservedNote,
    build_observed_performance,
    load_observed_smf,
)
from llm_musical_composer.reference_timing_hypothesis import (
    align_known_score_note_events,
    align_known_score_timing,
)
from llm_musical_composer.run_state import atomic_write_bytes, atomic_write_json, sha256_file
from llm_musical_composer.score_timing_hypothesis import (
    GroupingAttack,
    ScoreTimingHypothesisV0,
    build_grouping_profiles,
    compare_candidate_to_known_score,
    generate_score_timing_candidates,
)
from llm_musical_composer.texture_edge_diagnostics import (
    build_known_voice_diagnostics,
    build_texture_candidate_joins,
    build_texture_edge_diagnostics,
)

DEFAULT_WINDOW_SIZES = tuple(range(3, 9))
VIEWS = ("highest", "lowest", "pitch_set")


def _verified_oracle_observation_dependency(source: object) -> str:
    declared = str(source.oracle_observation_dependency)
    performance_path = getattr(source, "performance_path", None)
    if performance_path is None:
        return declared
    performance = parse_performance_spec(Path(performance_path).read_text(encoding="utf-8"))
    actual = (
        "present"
        if any(item.coordination_profile == "rolled" for item in performance.node_performances)
        else "absent"
    )
    if declared != actual:
        raise ValueError(
            f"oracle observation dependency mismatch: {source.case_id}: "
            f"declared={declared}, actual={actual}"
        )
    return declared


def _normalized_intervals(values: Sequence[int]) -> tuple[int, ...]:
    intervals = tuple(right - left for left, right in pairwise(values))
    if not intervals or any(value <= 0 for value in intervals):
        raise ValueError("score positions must be strictly increasing")
    divisor = math.gcd(*intervals)
    return tuple(value // divisor for value in intervals)


def _pitch_key(
    pitch_groups: Sequence[tuple[int, ...]],
    *,
    start: int,
    size: int,
    view: str,
) -> tuple[Any, ...]:
    window = pitch_groups[start : start + size]
    if view == "highest":
        values = tuple(max(group) for group in window)
        return tuple(value - values[0] for value in values)
    if view == "lowest":
        values = tuple(min(group) for group in window)
        return tuple(value - values[0] for value in values)
    if view == "pitch_set":
        basses = tuple(min(group) for group in window)
        return tuple(
            (
                bass - basses[0],
                tuple(sorted(pitch - bass for pitch in group)),
            )
            for bass, group in zip(basses, window, strict=True)
        )
    raise ValueError(f"unknown pitch view: {view}")


def enumerate_pitch_rhythm_recurrences(
    *,
    pitch_groups: Sequence[tuple[int, ...]],
    score_positions: Sequence[int],
    window_sizes: Sequence[int] = DEFAULT_WINDOW_SIZES,
) -> tuple[dict[str, Any], ...]:
    """非重複の音高形反復をcanonicalな出現集合として列挙する。"""
    if len(pitch_groups) != len(score_positions):
        raise ValueError("pitch groups and score positions must have equal length")
    if any(not group for group in pitch_groups):
        raise ValueError("pitch groups must be non-empty")
    positions = tuple(int(value) for value in score_positions)
    if len(positions) > 1:
        _normalized_intervals(positions)
    records: list[dict[str, Any]] = []
    for view in VIEWS:
        for raw_size in window_sizes:
            size = int(raw_size)
            if size < 2:
                raise ValueError("window sizes must be at least two")
            occurrences: dict[tuple[Any, ...], list[tuple[int, tuple[int, ...]]]] = defaultdict(
                list
            )
            for start in range(max(0, len(pitch_groups) - size + 1)):
                key = _pitch_key(pitch_groups, start=start, size=size, view=view)
                rhythm = _normalized_intervals(positions[start : start + size])
                occurrences[key].append((start, rhythm))
            for key, items in sorted(occurrences.items(), key=lambda item: repr(item[0])):
                pairs = tuple(
                    (left, right)
                    for left, right in combinations(items, 2)
                    if right[0] >= left[0] + size
                )
                if not pairs:
                    continue
                participating = sorted(
                    {item for pair in pairs for item in pair},
                    key=lambda item: (item[0], item[1]),
                )
                records.append(
                    {
                        "view": view,
                        "window_size": size,
                        "pitch_shape": key,
                        "occurrences": tuple(
                            {
                                "start_group_index": start,
                                "rhythm_ratio": rhythm,
                            }
                            for start, rhythm in participating
                        ),
                        "repetition_pair_count": len(pairs),
                        "same_rhythm_pair_count": sum(
                            left[1] == right[1] for left, right in pairs
                        ),
                        "different_rhythm_pair_count": sum(
                            left[1] != right[1] for left, right in pairs
                        ),
                    }
                )
    return tuple(records)


def summarize_pitch_rhythm_recurrence(
    *,
    pitch_groups: Sequence[tuple[int, ...]],
    score_positions: Sequence[int],
    window_sizes: Sequence[int] = DEFAULT_WINDOW_SIZES,
) -> dict[str, dict[str, dict[str, int]]]:
    """音高形の非重複反復と対応する音価比の一致を集計する。"""
    enumerated = enumerate_pitch_rhythm_recurrences(
        pitch_groups=pitch_groups,
        score_positions=score_positions,
        window_sizes=window_sizes,
    )
    result: dict[str, dict[str, dict[str, int]]] = {view: {} for view in VIEWS}
    for view in VIEWS:
        for raw_size in window_sizes:
            size = int(raw_size)
            if size < 2:
                raise ValueError("window sizes must be at least two")
            matching = [
                record
                for record in enumerated
                if record["view"] == view and record["window_size"] == size
            ]
            result[view][str(size)] = {
                "repeated_pitch_shape_count": len(matching),
                "repetition_pair_count": sum(
                    int(record["repetition_pair_count"]) for record in matching
                ),
                "same_rhythm_pair_count": sum(
                    int(record["same_rhythm_pair_count"]) for record in matching
                ),
                "different_rhythm_pair_count": sum(
                    int(record["different_rhythm_pair_count"]) for record in matching
                ),
            }
    return result


def _fingerprint(summary: Mapping[str, Any]) -> str:
    payload = json.dumps(summary, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_candidate_upper_evidence(
    candidate: ScoreTimingHypothesisV0,
    *,
    grouping_profiles: Mapping[str, tuple[GroupingAttack, ...]],
    notes_by_id: Mapping[str, ObservedNote],
    ledger_sha256: str,
    window_sizes: Sequence[int] = DEFAULT_WINDOW_SIZES,
) -> dict[str, Any]:
    """候補を元の発音群へ安全に再結合し、集計だけを返す。"""
    base = {
        "candidate_id": candidate.candidate_id,
        "grouping_profile_id": candidate.grouping_profile_id,
    }
    if candidate.source_ledger_sha256 != ledger_sha256:
        return {**base, "status": "unable_to_investigate", "reason": "ledger_sha256_mismatch"}
    groups = grouping_profiles.get(candidate.grouping_profile_id)
    if groups is None:
        return {**base, "status": "unable_to_investigate", "reason": "grouping_profile_missing"}
    refs = candidate.attack_group_refs
    if len(refs) != len(groups) or tuple(ref.group_index for ref in refs) != tuple(
        range(len(groups))
    ):
        return {**base, "status": "unable_to_investigate", "reason": "group_index_mismatch"}
    missing = sorted(
        {
            event_id
            for group in groups
            for event_id in group.source_event_ids
            if event_id not in notes_by_id
        }
    )
    if missing:
        return {
            **base,
            "status": "unable_to_investigate",
            "reason": "source_event_missing",
            "missing_event_count": len(missing),
        }
    pitch_groups = tuple(
        tuple(notes_by_id[event_id].pitch for event_id in group.source_event_ids)
        for group in groups
    )
    positions = tuple(ref.score_position for ref in refs)
    summary = summarize_pitch_rhythm_recurrence(
        pitch_groups=pitch_groups,
        score_positions=positions,
        window_sizes=window_sizes,
    )
    pair_count = sum(
        record["repetition_pair_count"] for view in summary.values() for record in view.values()
    )
    return {
        **base,
        "status": "assessed",
        "fingerprint": _fingerprint(summary),
        "repetition_pair_count": pair_count,
        "summary": summary,
    }


def classify_candidate_evidence(records: Sequence[Mapping[str, Any]]) -> str:
    """候補差が時間、発音群、どちらにもないかを分類する。"""
    assessed = [record for record in records if record.get("status") == "assessed"]
    if not assessed or not any(int(record.get("repetition_pair_count", 0)) for record in assessed):
        return "unable_to_investigate"
    by_profile: dict[str, set[str]] = defaultdict(set)
    for record in assessed:
        by_profile[str(record["grouping_profile_id"])].add(str(record["fingerprint"]))
    if any(len(fingerprints) > 1 for fingerprints in by_profile.values()):
        return "timing_sensitive"
    if len({str(record["fingerprint"]) for record in assessed}) > 1:
        return "grouping_only"
    return "candidate_invariant"


def _jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
        for record in records
    )


def _normalized_json_size(value: object) -> int:
    return len(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def run_upper_evidence_observation(
    *,
    sources: Sequence[object],
    output_dir: Path,
) -> dict[str, Any]:
    """SMFだけを読み、候補別の上位証拠集計を凍結する。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    input_records = [
        {
            "case_id": source.case_id,
            "group_id": source.group_id,
            "smf_path": str(Path(source.smf_path).resolve()),
            "smf_sha256": sha256_file(source.smf_path),
        }
        for source in sources
    ]
    input_path = output_dir / "observation-input.json"
    atomic_write_json(input_path, {"schema_version": 1, "sources": input_records})
    source_records = []
    texture_records = []
    for source, input_record in zip(sources, input_records, strict=True):
        observed_smf = load_observed_smf(Path(input_record["smf_path"]))
        if observed_smf.source_sha256 != input_record["smf_sha256"]:
            raise ValueError(f"SMF SHA-256 mismatch: {source.case_id}")
        performance = build_observed_performance(observed_smf)
        profiles = build_grouping_profiles(performance)
        candidate_set = generate_score_timing_candidates(
            performance,
            source_ledger_sha256=observed_smf.ledger_sha256,
        )
        notes_by_id = {note.note_on_event_id: note for note in performance.notes}
        candidate_records = [
            build_candidate_upper_evidence(
                candidate,
                grouping_profiles=profiles,
                notes_by_id=notes_by_id,
                ledger_sha256=observed_smf.ledger_sha256,
            )
            for candidate in candidate_set.candidates
        ]
        texture_diagnostic = build_texture_edge_diagnostics(
            grouping_profiles=profiles,
            notes_by_id=notes_by_id,
            ledger_sha256=observed_smf.ledger_sha256,
        )
        texture_join = build_texture_candidate_joins(
            candidate_set.candidates,
            texture_diagnostic=texture_diagnostic,
        )
        grouping_payload = {
            profile_id: [asdict(group) for group in groups]
            for profile_id, groups in sorted(profiles.items())
        }
        grouping_event_ids = [
            event_id
            for groups in profiles.values()
            for group in groups
            for event_id in group.source_event_ids
        ]
        texture_records.append(
            {
                "case_id": source.case_id,
                "group_id": source.group_id,
                "status": texture_diagnostic["status"],
                "measurements": {
                    "observed_note_count": len(performance.notes),
                    "observed_performance_normalized_json_bytes": _normalized_json_size(
                        asdict(performance)
                    ),
                    "grouping_unique_event_id_count": len(set(grouping_event_ids)),
                    "grouping_event_id_occurrence_count": len(grouping_event_ids),
                    "grouping_normalized_json_bytes": _normalized_json_size(grouping_payload),
                    "texture_event_id_unique_count": sum(
                        profile["event_id_unique_count"]
                        for profile in texture_diagnostic["profiles"]
                    ),
                    "texture_event_id_occurrence_count": sum(
                        profile["event_id_occurrence_count"]
                        for profile in texture_diagnostic["profiles"]
                    ),
                    "texture_normalized_json_bytes": _normalized_json_size(texture_diagnostic),
                },
                "texture_diagnostic": texture_diagnostic,
                "candidate_join": texture_join,
            }
        )
        source_records.append(
            {
                "case_id": source.case_id,
                "group_id": source.group_id,
                "status": candidate_set.status,
                "candidate_count": len(candidate_records),
                "sensitivity": classify_candidate_evidence(candidate_records),
                "candidate_records": candidate_records,
            }
        )
    evidence_path = output_dir / "observed-upper-evidence.jsonl"
    atomic_write_bytes(evidence_path, _jsonl_bytes(source_records))
    texture_path = output_dir / "observed-texture-edge-diagnostics.jsonl"
    atomic_write_bytes(texture_path, _jsonl_bytes(texture_records))
    result = {
        "status": "pass"
        if all(record["status"] == "assessed" for record in source_records)
        else "partial",
        "source_count": len(source_records),
        "group_count": len({record["group_id"] for record in source_records}),
        "sensitivity_counts": {
            name: sum(record["sensitivity"] == name for record in source_records)
            for name in (
                "timing_sensitive",
                "grouping_only",
                "candidate_invariant",
                "unable_to_investigate",
            )
        },
        "texture_semantic_profile_count": sum(
            record["texture_diagnostic"]["semantic_profile_count"] for record in texture_records
        ),
        "texture_sensitivity_counts": {
            name: sum(record["candidate_join"]["sensitivity"] == name for record in texture_records)
            for name in ("timing_sensitive", "candidate_invariant", "unable_to_investigate")
        },
    }
    manifest = {
        "schema_version": 1,
        "status": result["status"],
        "outputs": {
            "observation-input.json": sha256_file(input_path),
            "observed-upper-evidence.jsonl": sha256_file(evidence_path),
            "observed-texture-edge-diagnostics.jsonl": sha256_file(texture_path),
        },
    }
    atomic_write_json(output_dir / "observation-manifest.json", manifest)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def verify_observation_artifacts(output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """正解付与前に、観測成果物が凍結後から変わっていないことを検証する。"""
    output_dir = Path(output_dir)
    manifest = _read_json(output_dir / "observation-manifest.json")
    for name, expected in manifest.get("outputs", {}).items():
        if sha256_file(output_dir / name) != expected:
            raise ValueError(f"observation artifact SHA-256 mismatch: {name}")
    return (
        _read_jsonl(output_dir / "observed-upper-evidence.jsonl"),
        _read_json(output_dir / "observation-input.json"),
    )


def run_upper_evidence_oracle_annotation(
    *,
    sources: Sequence[object],
    output_dir: Path,
) -> dict[str, Any]:
    """凍結した観測診断へ、別段階で既知楽譜との比較だけを付与する。"""
    observed_records, observation_input = verify_observation_artifacts(output_dir)
    observed_by_id = {record["case_id"]: record for record in observed_records}
    input_by_id = {record["case_id"]: record for record in observation_input["sources"]}
    texture_by_id = {
        record["case_id"]: record
        for record in _read_jsonl(Path(output_dir) / "observed-texture-edge-diagnostics.jsonl")
    }
    annotations = []
    voice_annotations = []
    for source in sources:
        observed_record = observed_by_id[source.case_id]
        input_record = input_by_id[source.case_id]
        if sha256_file(source.smf_path) != input_record["smf_sha256"]:
            raise ValueError(f"SMF SHA-256 mismatch before oracle annotation: {source.case_id}")
        plan = parse_piece_plan(source.piece_path.read_text(encoding="utf-8"))
        score = parse_score_spec(source.score_path.read_text(encoding="utf-8"))
        observed_smf = load_observed_smf(source.smf_path)
        performance = build_observed_performance(observed_smf)
        timing = align_known_score_timing(plan, score, observed_smf, performance)
        profiles = build_grouping_profiles(performance)
        notes_by_id = {note.note_on_event_id: note for note in performance.notes}
        alignment = align_known_score_note_events(plan, score, performance)
        oracle_basis = str(source.oracle_basis)
        oracle_observation_dependency = _verified_oracle_observation_dependency(source)
        voice_annotations.append(
            {
                "case_id": source.case_id,
                "group_id": source.group_id,
                "oracle_observation_dependency": oracle_observation_dependency,
                "positive_evidence_eligible": oracle_observation_dependency == "absent",
                **build_known_voice_diagnostics(
                    alignment=alignment,
                    grouping_profiles=profiles,
                    notes_by_id=notes_by_id,
                    texture_diagnostic=texture_by_id[source.case_id]["texture_diagnostic"],
                    oracle_basis=oracle_basis,
                ),
            }
        )
        if timing.note_on_status != "assessed":
            annotations.append(
                {
                    "case_id": source.case_id,
                    "group_id": source.group_id,
                    "status": "unable_to_investigate",
                    "reason": "known_score_timing_not_assessed",
                }
            )
            continue
        known_units = {
            event_id: group.score_unit
            for group in timing.attack_groups
            for event_id in group.evidence_event_ids
        }
        candidate_set = generate_score_timing_candidates(
            performance,
            source_ledger_sha256=observed_smf.ledger_sha256,
        )
        observed_candidates = {
            record["candidate_id"]: record for record in observed_record["candidate_records"]
        }
        candidate_annotations = []
        for candidate in candidate_set.candidates:
            if candidate.candidate_id not in observed_candidates:
                raise ValueError(
                    f"candidate set changed: {source.case_id}: {candidate.candidate_id}"
                )
            comparison = compare_candidate_to_known_score(
                candidate,
                profiles[candidate.grouping_profile_id],
                known_units,
            )
            candidate_annotations.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "fingerprint": observed_candidates[candidate.candidate_id]["fingerprint"],
                    "comparison_status": comparison["status"],
                    "match_rate": comparison.get("best_scale_interval_match_rate"),
                }
            )
        comparable = [
            record for record in candidate_annotations if record["match_rate"] is not None
        ]
        exact = [record for record in comparable if record["comparison_status"] == "equivalent"]
        exact_collision = any(
            any(
                other["fingerprint"] == record["fingerprint"]
                and other["comparison_status"] != "equivalent"
                for other in comparable
            )
            for record in exact
        )
        best_match_collision: bool | None = None
        best_rate: float | None = None
        if comparable and not exact:
            best_rate = max(float(record["match_rate"]) for record in comparable)
            best_records = [
                record for record in comparable if float(record["match_rate"]) == best_rate
            ]
            best_match_collision = any(
                any(
                    other["fingerprint"] == record["fingerprint"]
                    and float(other["match_rate"]) < best_rate
                    for other in comparable
                )
                for record in best_records
            )
        annotations.append(
            {
                "case_id": source.case_id,
                "group_id": source.group_id,
                "status": "assessed",
                "sensitivity": observed_record["sensitivity"],
                "exact_candidate_count": len(exact),
                "exact_oracle_collision": exact_collision,
                "best_match_rate": best_rate,
                "best_match_oracle_collision": best_match_collision,
                "candidate_annotations": candidate_annotations,
            }
        )
    annotation_path = Path(output_dir) / "oracle-annotations.jsonl"
    atomic_write_bytes(annotation_path, _jsonl_bytes(annotations))
    voice_annotation_path = Path(output_dir) / "texture-voice-oracle-annotations.jsonl"
    atomic_write_bytes(voice_annotation_path, _jsonl_bytes(voice_annotations))
    assessed = [record for record in annotations if record["status"] == "assessed"]
    assessed_groups = {record["group_id"] for record in assessed}
    timing_sensitive_groups = {
        record["group_id"] for record in assessed if record["sensitivity"] == "timing_sensitive"
    }
    result = {
        "status": "pass" if len(assessed) == len(annotations) else "partial",
        "source_count": len(annotations),
        "assessed_source_count": len(assessed),
        "assessed_group_count": len(assessed_groups),
        "assessed_timing_sensitivity_group_count": len(timing_sensitive_groups),
        "exact_oracle_collision_source_count": sum(
            bool(record["exact_oracle_collision"]) for record in assessed
        ),
        "best_match_oracle_collision_source_count": sum(
            record["best_match_oracle_collision"] is True for record in assessed
        ),
        "texture_voice_assessed_source_count": sum(
            record["status"] == "assessed" for record in voice_annotations
        ),
        "texture_voice_pitch_rank_constructed_source_count": sum(
            record["oracle_basis"] == "pitch_rank_constructed" for record in voice_annotations
        ),
    }
    atomic_write_json(Path(output_dir) / "result.json", result)
    atomic_write_json(
        Path(output_dir) / "texture-result.json",
        {
            "schema_version": 0,
            "status": (
                "pass"
                if all(record["status"] == "assessed" for record in voice_annotations)
                else "partial"
            ),
            "status_scope": "oracle_annotation_completion_only",
            "perceptual_voice_evidence_status": "insufficient",
            "source_count": len(voice_annotations),
            "assessed_source_count": result["texture_voice_assessed_source_count"],
            "pitch_rank_constructed_source_count": result[
                "texture_voice_pitch_rank_constructed_source_count"
            ],
            "generator_voice_label_source_count": sum(
                record["oracle_basis"] == "generator_voice_labels" for record in voice_annotations
            ),
        },
    )
    return result


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("observe", "annotate"))
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    from llm_musical_composer.reference_timing_run import default_known_timing_sources

    arguments = _parse_arguments()
    sources = default_known_timing_sources(arguments.workspace)
    if arguments.mode == "observe":
        result = run_upper_evidence_observation(
            sources=sources,
            output_dir=arguments.output_dir,
        )
    else:
        result = run_upper_evidence_oracle_annotation(
            sources=sources,
            output_dir=arguments.output_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
