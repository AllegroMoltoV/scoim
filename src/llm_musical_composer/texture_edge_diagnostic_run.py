"""固定source集合で圧縮texture edgeと生成元ラベルの記述診断を実行する。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_musical_composer.reference_timing_run import default_known_timing_sources
from llm_musical_composer.run_state import atomic_write_json, sha256_file
from llm_musical_composer.score_timing_known_fixtures import build_known_timing_fixtures
from llm_musical_composer.score_timing_upper_evidence import (
    run_upper_evidence_observation,
    run_upper_evidence_oracle_annotation,
)


class TextureEdgeDiagnosticRunError(ValueError):
    """入力固定または実験契約に違反した。"""


@dataclass(frozen=True)
class ObservationSource:
    case_id: str
    group_id: str
    smf_path: Path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TextureEdgeDiagnosticRunError(f"unable to read JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise TextureEdgeDiagnosticRunError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def development_observation_sources(
    *,
    staging_dir: Path,
    staging_manifest: Path,
) -> tuple[ObservationSource, ...]:
    """固定manifestと一致する開発SMFだけを観測入力にする。"""
    staging_dir = Path(staging_dir)
    manifest = _read_json(staging_manifest)
    expected = manifest.get("development_smf")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "pass"
        or not isinstance(expected, dict)
    ):
        raise TextureEdgeDiagnosticRunError("development staging manifest is invalid")
    expected_names = set(expected)
    actual_names = {
        path.name
        for path in staging_dir.iterdir()
        if path.is_file() and path.suffix.casefold() == ".mid"
    }
    if expected_names != actual_names:
        raise TextureEdgeDiagnosticRunError(
            f"development source set mismatch: expected={sorted(expected_names)}, "
            f"actual={sorted(actual_names)}"
        )
    sources: list[ObservationSource] = []
    for name in sorted(expected, key=lambda value: (value.casefold(), value)):
        path = staging_dir / name
        if sha256_file(path) != expected[name]:
            raise TextureEdgeDiagnosticRunError(f"development SMF SHA-256 mismatch: {name}")
        sources.append(
            ObservationSource(
                case_id=f"development-{Path(name).stem}",
                group_id="development-reference-unlabeled",
                smf_path=path,
            )
        )
    return tuple(sources)


def _input_processing_status(
    texture_records: list[dict[str, Any]],
    voice_records: list[dict[str, Any]],
    *,
    expected_source_count: int,
    expected_known_voice_source_count: int,
) -> str:
    if (
        len(texture_records) == expected_source_count
        and len(voice_records) == expected_known_voice_source_count
        and all(record["status"] == "assessed" for record in texture_records)
        and all(record["status"] == "assessed" for record in voice_records)
    ):
        return "pass"
    return "partial"


def _summarize_output(
    output_dir: Path,
    *,
    expected_source_count: int,
    expected_known_voice_source_count: int,
) -> dict[str, Any]:
    texture_records = _read_jsonl(output_dir / "observed-texture-edge-diagnostics.jsonl")
    voice_records = _read_jsonl(output_dir / "texture-voice-oracle-annotations.jsonl")
    all_profiles = [
        profile
        for record in texture_records
        for profile in record["texture_diagnostic"]["profiles"]
    ]
    generator_records = [
        record for record in voice_records if record["oracle_basis"] == "generator_voice_labels"
    ]

    def has_pitch_rank_counterexample(record: dict[str, Any]) -> bool:
        return any(
            profile["highest_upper_match_rate"] != 1.0
            or profile["lowest_lower_match_rate"] != 1.0
            or profile["upper_below_lower_group_count"] > 0
            for profile in record["profiles"]
        )

    rank_counterexamples = [
        record for record in generator_records if has_pitch_rank_counterexample(record)
    ]
    measurements = [record["measurements"] for record in texture_records]
    processing_status = _input_processing_status(
        texture_records,
        voice_records,
        expected_source_count=expected_source_count,
        expected_known_voice_source_count=expected_known_voice_source_count,
    )
    eligible_generator_records = [
        record for record in generator_records if record.get("positive_evidence_eligible") is True
    ]
    eligible_profiles = [
        profile for record in eligible_generator_records for profile in record["profiles"]
    ]
    assessed_null_profiles = [
        profile
        for profile in eligible_profiles
        if profile["nearest_null_comparison_status"] == "assessed"
    ]
    unavailable_null_profile_count = sum(
        profile["nearest_null_comparison_status"] != "assessed" for profile in eligible_profiles
    )
    return {
        "schema_version": 0,
        "status": processing_status,
        "status_scope": "input_processing_only",
        "adoption_status": "rejected_persistent_base_graph",
        "representation_status": "derived_view_only",
        "persistent_voice_representation_status": "pending",
        "perceptual_voice_evidence_status": "insufficient",
        "consumer_contract_status": "not_defined",
        "pitch_rank_baseline_status": (
            "counterexample_confirmed" if rank_counterexamples else "unable_to_investigate"
        ),
        "nearest_relation_null_comparison_status": (
            "descriptive_only"
            if assessed_null_profiles and unavailable_null_profile_count == 0
            else "partial"
            if assessed_null_profiles
            else "unable_to_investigate"
        ),
        "source_count": len(texture_records),
        "known_voice_source_count": len(voice_records),
        "surface_profile_count": sum(
            record["texture_diagnostic"]["surface_profile_count"] for record in texture_records
        ),
        "semantic_profile_count": sum(
            record["texture_diagnostic"]["semantic_profile_count"] for record in texture_records
        ),
        "candidate_count": sum(
            record["candidate_join"]["candidate_count"] for record in texture_records
        ),
        "base_graph_count": sum(
            record["candidate_join"]["base_graph_count"] for record in texture_records
        ),
        "event_id_free_profile_count": sum(
            profile["event_id_unique_count"] == 0 and profile["event_id_occurrence_count"] == 0
            for profile in all_profiles
        ),
        "max_block_member_ratio": max(
            (float(profile["block_member_ratio"]) for profile in all_profiles),
            default=0.0,
        ),
        "max_virtual_edges_per_relation_block": max(
            (int(profile["max_virtual_edges_per_relation_block"]) for profile in all_profiles),
            default=0,
        ),
        "max_ambiguous_component_member_count": max(
            (int(profile["max_ambiguous_component_member_count"]) for profile in all_profiles),
            default=0,
        ),
        "max_path_count_log10_upper_bound": max(
            (float(profile["path_count_log10_upper_bound"]) for profile in all_profiles),
            default=0.0,
        ),
        "texture_to_grouping_byte_ratio_min": min(
            (
                measurement["texture_normalized_json_bytes"]
                / measurement["grouping_normalized_json_bytes"]
                for measurement in measurements
            ),
            default=0.0,
        ),
        "texture_to_grouping_byte_ratio_max": max(
            (
                measurement["texture_normalized_json_bytes"]
                / measurement["grouping_normalized_json_bytes"]
                for measurement in measurements
            ),
            default=0.0,
        ),
        "timing_sensitivity_counts": {
            name: sum(record["candidate_join"]["sensitivity"] == name for record in texture_records)
            for name in ("timing_sensitive", "candidate_invariant", "unable_to_investigate")
        },
        "pitch_rank_constructed_source_count": sum(
            record["oracle_basis"] == "pitch_rank_constructed" for record in voice_records
        ),
        "generator_voice_label_source_count": len(generator_records),
        "generator_voice_group_count": len({record["group_id"] for record in generator_records}),
        "generator_voice_pitch_rank_counterexample_source_count": len(rank_counterexamples),
        "generator_voice_pitch_rank_counterexample_group_count": len(
            {record["group_id"] for record in rank_counterexamples}
        ),
        "oracle_independent_generator_voice_source_count": len(eligible_generator_records),
        "oracle_independent_generator_voice_group_count": len(
            {record["group_id"] for record in eligible_generator_records}
        ),
        "oracle_dependent_generator_voice_source_count": len(generator_records)
        - len(eligible_generator_records),
        "oracle_independent_assessed_profile_count": len(assessed_null_profiles),
        "oracle_independent_unavailable_profile_count": unavailable_null_profile_count,
        "oracle_independent_positive_lift_profile_count": sum(
            float(profile["same_label_rate_lift"]) > 0 for profile in assessed_null_profiles
        ),
        "oracle_independent_zero_lift_profile_count": sum(
            float(profile["same_label_rate_lift"]) == 0 for profile in assessed_null_profiles
        ),
        "oracle_independent_negative_lift_profile_count": sum(
            float(profile["same_label_rate_lift"]) < 0 for profile in assessed_null_profiles
        ),
        "oracle_independent_lift_min": min(
            (float(profile["same_label_rate_lift"]) for profile in assessed_null_profiles),
            default=None,
        ),
        "oracle_independent_lift_max": max(
            (float(profile["same_label_rate_lift"]) for profile in assessed_null_profiles),
            default=None,
        ),
    }


def run_texture_edge_diagnostic_pilot(
    *,
    workspace: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """固定済み18入力を観測し、既知10例だけへ事後oracleを付ける。"""
    workspace = Path(workspace)
    known_sources = default_known_timing_sources(workspace) + tuple(
        fixture.source
        for fixture in build_known_timing_fixtures(
            workspace / ".appendix/score-timing-known-fixtures-v1"
        )
    )
    development_sources = development_observation_sources(
        staging_dir=workspace / ".appendix/score-timing-split-v1/development-smf",
        staging_manifest=workspace / ".appendix/score-timing-split-v1/manifest.json",
    )
    observation = run_upper_evidence_observation(
        sources=known_sources + development_sources,
        output_dir=output_dir,
    )
    oracle = run_upper_evidence_oracle_annotation(
        sources=known_sources,
        output_dir=output_dir,
    )
    summary = _summarize_output(
        Path(output_dir),
        expected_source_count=len(known_sources) + len(development_sources),
        expected_known_voice_source_count=len(known_sources),
    )
    result = {
        **summary,
        "observation_status": observation["status"],
        "oracle_status": oracle["status"],
    }
    atomic_write_json(Path(output_dir) / "experiment-result.json", result)
    output_names = (
        "experiment-result.json",
        "observation-input.json",
        "observation-manifest.json",
        "observed-texture-edge-diagnostics.jsonl",
        "observed-upper-evidence.jsonl",
        "oracle-annotations.jsonl",
        "result.json",
        "texture-result.json",
        "texture-voice-oracle-annotations.jsonl",
    )
    atomic_write_json(
        Path(output_dir) / "manifest.json",
        {
            "schema_version": 1,
            "status": result["status"],
            "outputs": {name: sha256_file(Path(output_dir) / name) for name in output_names},
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
    result = run_texture_edge_diagnostic_pilot(
        workspace=arguments.workspace,
        output_dir=arguments.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
