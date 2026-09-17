"""活動量の参照分布を作り、保存済み候補を API 呼び出しなしで再評価する。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from llm_musical_composer.pilot_features import (
    NoteEvent,
    evaluate_features,
    extract_smf_features,
)
from llm_musical_composer.smf_notes import load_smf_notes
from llm_musical_composer.structure_features import extract_structure_features


def _round(value: float) -> float:
    return round(float(value), 8)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"available_count": 0, "status": "unable_to_investigate"}
    return {
        "available_count": len(values),
        "p25": _round(_percentile(values, 0.25)),
        "median": _round(_percentile(values, 0.5)),
        "p75": _round(_percentile(values, 0.75)),
    }


def extract_activity_features(
    notes: list[NoteEvent], *, window_count: int = 16
) -> dict[str, dict[str, float | None]]:
    """高活動区間の数と最初の位置を、別々の値として返す。"""
    report = extract_structure_features(notes, window_count=window_count)
    if report.get("status") != "pass":
        raise ValueError(report.get("detail", "activity features are unavailable"))
    candidates = report["high_activity_candidates"]
    return {
        "activity": {
            "high_activity_candidate_count": float(len(candidates)),
            "first_high_activity_position": (
                float(candidates[0]["position"]) if candidates else None
            ),
        }
    }


def _extract_smf_activity(path: Path) -> dict[str, dict[str, float | None]]:
    notes = [
        NoteEvent(note.pitch, note.onset_ms, note.duration_ms, note.velocity)
        for note in load_smf_notes(path)
    ]
    return extract_activity_features(notes)


def build_activity_reference_profile(
    feature_sets: list[dict[str, dict[str, float | None]]],
) -> dict[str, Any]:
    if not feature_sets:
        raise ValueError("at least one usable activity feature set is required")
    counts = [
        float(features["activity"]["high_activity_candidate_count"]) for features in feature_sets
    ]
    positions = [
        float(value)
        for features in feature_sets
        if (value := features["activity"]["first_high_activity_position"]) is not None
    ]
    return {
        "schema_version": 1,
        "source_count": len(feature_sets),
        "axes": {
            "activity": {
                "high_activity_candidate_count": _summary(counts),
                "first_high_activity_position": _summary(positions),
            }
        },
    }


def build_activity_reference_profile_from_directory(
    input_dir: Path, *, excluded_names: frozenset[str] = frozenset()
) -> dict[str, Any]:
    feature_sets: list[dict[str, dict[str, float | None]]] = []
    source_files: list[str] = []
    unable: list[dict[str, str]] = []
    excluded_casefold = {name.casefold() for name in excluded_names}
    for path in sorted(Path(input_dir).glob("*.mid"), key=lambda item: item.name.casefold()):
        if path.name.casefold() in excluded_casefold:
            continue
        try:
            feature_sets.append(_extract_smf_activity(path))
            source_files.append(path.name)
        except (OSError, ValueError, EOFError) as error:
            unable.append({"name": path.name, "error": f"{type(error).__name__}: {error}"})
    profile = build_activity_reference_profile(feature_sets)
    profile["source_files"] = source_files
    profile["excluded_files"] = sorted(excluded_names)
    profile["unable_to_investigate"] = unable
    return profile


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, float, str]:
    evaluation = candidate["evaluation"]
    return (
        -float(evaluation["inside_axis_count"]),
        float(evaluation["worst_axis_distance"]),
        str(candidate["candidate_id"]),
    )


def replay_activity_selection(
    *,
    source_dir: Path,
    base_profile_path: Path,
    candidate_dir: Path,
    output_dir: Path,
    excluded_names: frozenset[str] = frozenset({"rut.mid", "aimusic01.mid"}),
) -> dict[str, Any]:
    """保存済み MIDI を再評価し、外部生成 API を呼ばずに順位を付ける。"""
    activity_profile = build_activity_reference_profile_from_directory(
        source_dir, excluded_names=excluded_names
    )
    base_profile = json.loads(Path(base_profile_path).read_text(encoding="utf-8"))
    effective_profile = {**base_profile, "axes": dict(base_profile["axes"])}
    effective_profile["axes"]["activity"] = activity_profile["axes"]["activity"]

    candidates: list[dict[str, Any]] = []
    unable: list[dict[str, str]] = []
    for midi_path in sorted(Path(candidate_dir).glob("candidate-*.mid")):
        try:
            features = extract_smf_features(midi_path)
            features.update(_extract_smf_activity(midi_path))
            candidates.append(
                {
                    "candidate_id": midi_path.stem,
                    "midi_path": str(midi_path),
                    "evaluation": evaluate_features(features, effective_profile),
                }
            )
        except (OSError, ValueError, EOFError) as error:
            unable.append({"name": midi_path.name, "error": f"{type(error).__name__}: {error}"})
    selected = min(candidates, key=_candidate_sort_key) if candidates else None
    result = {
        "schema_version": 1,
        "api_call_count": 0,
        "source_count": activity_profile["source_count"],
        "candidate_count": len(candidates),
        "selected_candidate": selected["candidate_id"] if selected else None,
        "candidates": candidates,
        "unable_to_investigate": unable,
    }
    _write_json(Path(output_dir) / "activity-reference-profile.json", activity_profile)
    _write_json(Path(output_dir) / "effective-reference-profile.json", effective_profile)
    _write_json(Path(output_dir) / "candidate-evaluations.json", candidates)
    _write_json(Path(output_dir) / "manifest.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="保存済み候補へ活動量評価軸を追加します。")
    parser.add_argument("--source-dir", type=Path, default=Path(".appendix/source-smf"))
    parser.add_argument(
        "--base-profile",
        type=Path,
        default=Path(".appendix/minimal-loop/effective-reference-profile.json"),
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=Path(".appendix/minimal-loop/20260811-isolated-context-v1/candidates"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".appendix/activity-selection-replay/20260811-isolated-context-v1"),
    )
    args = parser.parse_args(argv)
    result = replay_activity_selection(
        source_dir=args.source_dir,
        base_profile_path=args.base_profile,
        candidate_dir=args.candidate_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["selected_candidate"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
