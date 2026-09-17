"""鍵盤保持中の垂直音程クラス分布を、移調不変な作風診断として扱う。"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

from llm_musical_composer.run_state import atomic_write_json, sha256_file
from llm_musical_composer.smf_notes import SmfNote, load_smf_notes

INTERVAL_CLASSES = tuple(str(index) for index in range(7))


def _round(value: float) -> float:
    return round(float(value), 8)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return _round(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _interval_class(first: int, second: int) -> int:
    distance = abs(first - second) % 12
    return min(distance, 12 - distance)


def _validate_note(item: SmfNote) -> None:
    if not 0 <= item.pitch <= 127:
        raise ValueError("note pitch must be between 0 and 127")
    if item.onset_ms < 0:
        raise ValueError("note onset_ms must be non-negative")
    if item.duration_ms <= 0:
        raise ValueError("note duration_ms must be positive")


def extract_harmonic_profile(notes: Iterable[SmfNote]) -> dict[str, Any]:
    """鍵盤保持が重なる音符対の時間を、7 種類の音程クラスへ集計する。"""
    prepared = list(notes)
    for item in prepared:
        _validate_note(item)

    starts: dict[int, list[tuple[int, int]]] = {}
    ends: dict[int, list[int]] = {}
    for note_id, item in enumerate(prepared):
        starts.setdefault(item.onset_ms, []).append((note_id, item.pitch))
        ends.setdefault(item.onset_ms + item.duration_ms, []).append(note_id)
    times = sorted(set(starts) | set(ends))
    active: dict[int, int] = {}
    class_pair_ms = {key: 0 for key in INTERVAL_CLASSES}
    previous_time = times[0] if times else 0
    for time in times:
        elapsed = time - previous_time
        if elapsed > 0 and len(active) >= 2:
            for first, second in combinations(active.values(), 2):
                class_pair_ms[str(_interval_class(first, second))] += elapsed
        for note_id in ends.get(time, []):
            active.pop(note_id, None)
        for note_id, pitch in starts.get(time, []):
            active[note_id] = pitch
        previous_time = time

    overlap_pair_ms = sum(class_pair_ms.values())
    if overlap_pair_ms == 0:
        return {
            "schema_version": 1,
            "status": "unable_to_investigate",
            "reason": "no overlapping key-held note pairs",
            "overlap_pair_ms": 0,
        }
    return {
        "schema_version": 1,
        "status": "pass",
        "basis": "key-held overlap duration; sustain extension excluded",
        "overlap_pair_ms": overlap_pair_ms,
        "class_pair_ms": class_pair_ms,
        "distribution": {
            key: _round(value / overlap_pair_ms) for key, value in class_pair_ms.items()
        },
    }


def profile_smf(path: Path) -> dict[str, Any]:
    """SMF を読み、入力ハッシュと垂直音程クラス分布を返す。"""
    path = Path(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "profile": extract_harmonic_profile(load_smf_notes(path)),
    }


def _validated_distribution(profile: dict[str, Any]) -> dict[str, float]:
    if profile.get("status") != "pass":
        raise ValueError("harmonic profile is unavailable")
    distribution = profile.get("distribution")
    if not isinstance(distribution, dict) or set(distribution) != set(INTERVAL_CLASSES):
        raise ValueError("harmonic profile has invalid interval classes")
    try:
        values = {key: float(distribution[key]) for key in INTERVAL_CLASSES}
    except (TypeError, ValueError) as error:
        raise ValueError("harmonic profile distribution is invalid") from error
    if not all(math.isfinite(value) and value >= 0 for value in values.values()):
        raise ValueError("harmonic profile distribution must be finite and non-negative")
    if not math.isclose(sum(values.values()), 1.0, abs_tol=1e-6):
        raise ValueError("harmonic profile distribution must sum to one")
    return values


def _class_summary(values: Sequence[float]) -> dict[str, Any]:
    minimum = _round(min(values))
    p25 = _percentile(values, 0.25)
    median = _percentile(values, 0.5)
    p75 = _percentile(values, 0.75)
    maximum = _round(max(values))
    observed_range = maximum - minimum
    iqr = p75 - p25
    return {
        "available_count": len(values),
        "minimum": minimum,
        "p25": p25,
        "median": median,
        "p75": p75,
        "maximum": maximum,
        "target_low": p25,
        "target_high": p75,
        "iqr_to_observed_range": _round(iqr / observed_range) if observed_range > 1e-12 else None,
    }


def build_harmonic_target(references: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """固定済み近傍の音程クラスごとの範囲を、合算せず返す。"""
    prepared: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for item in sorted(references, key=lambda value: str(value.get("name", "")).casefold()):
        name = str(item.get("name", ""))
        if not name:
            raise ValueError("reference name is required")
        folded = name.casefold()
        if folded in seen:
            raise ValueError(f"duplicate reference name: {seen[folded]} and {name}")
        seen[folded] = name
        try:
            distribution = _validated_distribution(item["profile"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"reference profile is unavailable: {name}: {error}") from error
        prepared.append(
            {
                "name": name,
                "sha256": str(item.get("sha256", "")),
                "distribution": {key: _round(value) for key, value in distribution.items()},
            }
        )
    if len(prepared) < 3:
        raise ValueError("at least three reference profiles are required")
    return {
        "schema_version": 1,
        "status": "pass",
        "semantics": "transposition-invariant key-held vertical interval-class distribution",
        "reference_count": len(prepared),
        "references": prepared,
        "classes": {
            key: _class_summary([item["distribution"][key] for item in prepared])
            for key in INTERVAL_CLASSES
        },
    }


def _class_distance(value: float, bounds: dict[str, Any]) -> float:
    low = float(bounds["target_low"])
    high = float(bounds["target_high"])
    if low <= value <= high:
        return 0.0
    scale = high - low
    if scale <= 1e-9:
        scale = max(
            float(bounds["maximum"]) - float(bounds["minimum"]),
            abs(float(bounds["median"])),
            1e-6,
        )
    return _round((low - value if value < low else value - high) / scale)


def evaluate_harmonic_profile(profile: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    """候補と参照範囲の差を音程クラス別に返す。"""
    distribution = _validated_distribution(profile)
    if target.get("status") != "pass" or set(target.get("classes", {})) != set(INTERVAL_CLASSES):
        raise ValueError("harmonic target is unavailable")
    classes = {}
    for key in INTERVAL_CLASSES:
        bounds = target["classes"][key]
        distance = _class_distance(distribution[key], bounds)
        classes[key] = {
            "candidate_value": _round(distribution[key]),
            "target_range_low": bounds["target_low"],
            "target_median": bounds["median"],
            "target_range_high": bounds["target_high"],
            "inside_target_range": distance == 0,
            "normalized_distance": distance,
        }
    return {
        "schema_version": 1,
        "status": "pass",
        "classes": classes,
        "inside_class_count": sum(item["inside_target_range"] for item in classes.values()),
        "worst_class_distance": _round(
            max(item["normalized_distance"] for item in classes.values())
        ),
        "outside_classes": [
            key for key, item in classes.items() if not item["inside_target_range"]
        ],
    }


def decide_local_repair(
    *, transposition_passes: bool, outside_classes: Sequence[str]
) -> dict[str, Any]:
    """指標対照と範囲外クラス数から、局所修正へ進めるかを返す。"""
    eligible = transposition_passes and len(outside_classes) == 1
    if not transposition_passes:
        reason = "v26 and v34 differ under the transposition-invariant metric"
    elif not outside_classes:
        reason = "v26 is already inside every interval-class target range"
    elif len(outside_classes) > 1:
        reason = "v26 has multiple outside interval classes; a unique local repair is not justified"
    else:
        reason = "one outside interval class can be diagnosed before a bounded local repair"
    return {
        "eligible": eligible,
        "outside_classes": list(outside_classes),
        "reason": reason,
    }


def analyze_harmonic_style(
    *,
    source_dir: Path,
    style_target_path: Path,
    candidate_paths: dict[str, Path],
) -> dict[str, Any]:
    """固定参照近傍と既存候補を同じ音程クラス経路で分析する。"""
    source_dir = Path(source_dir)
    style_target_path = Path(style_target_path)
    style_target = json.loads(style_target_path.read_text(encoding="utf-8"))
    try:
        neighbor_names = [str(item["name"]) for item in style_target["neighbors"]]
    except (KeyError, TypeError) as error:
        raise ValueError("style target has no fixed neighbor list") from error
    if len(neighbor_names) < 3 or len({name.casefold() for name in neighbor_names}) != len(
        neighbor_names
    ):
        raise ValueError("style target neighbor list is too small or duplicated")

    references = []
    for name in neighbor_names:
        item = profile_smf(source_dir / name)
        references.append({"name": name, "sha256": item["sha256"], "profile": item["profile"]})
    target = build_harmonic_target(references)

    candidates = {}
    for candidate_id, path in sorted(candidate_paths.items()):
        item = profile_smf(path)
        candidates[candidate_id] = {
            **item,
            "evaluation": evaluate_harmonic_profile(item["profile"], target),
        }
    try:
        baseline = candidates["v26"]["profile"]["distribution"]
        transposed = candidates["v34"]["profile"]["distribution"]
    except KeyError as error:
        raise ValueError("candidate paths must include v26 and v34") from error
    dimension_differences = {
        key: _round(abs(float(baseline[key]) - float(transposed[key]))) for key in INTERVAL_CLASSES
    }
    transposition_passes = max(dimension_differences.values()) <= 1e-8
    outside_classes = candidates["v26"]["evaluation"]["outside_classes"]
    repair_decision = decide_local_repair(
        transposition_passes=transposition_passes,
        outside_classes=outside_classes,
    )
    return {
        "schema_version": 1,
        "status": "pass" if transposition_passes else "fail",
        "metric_scope": (
            "key-held vertical interval-class distribution only; "
            "not harmonic function, resolution, or aesthetic quality"
        ),
        "inputs": {
            "style_target": {
                "path": str(style_target_path),
                "sha256": sha256_file(style_target_path),
            },
            "source_dir": str(source_dir),
        },
        "target": target,
        "candidates": candidates,
        "controls": {
            "v26_v34_global_transposition": {
                "status": "pass" if transposition_passes else "fail",
                "dimension_differences": dimension_differences,
                "maximum_difference": max(dimension_differences.values()),
            }
        },
        "negative_reference_semantics": (
            "v9 is diagnostic only because it differs from v26 on more than harmony"
        ),
        "local_repair_decision": repair_decision,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="移調不変な垂直音程分布を分析します。")
    parser.add_argument("--source-dir", type=Path, default=Path(".appendix/source-smf"))
    parser.add_argument(
        "--style-target",
        type=Path,
        default=Path(".appendix/style-target-feasible-default-v1/style-target.json"),
    )
    parser.add_argument(
        "--v26",
        type=Path,
        default=Path(
            ".appendix/long-form-runs/20260818-long-form-v26-ending-voice-leading/final.mid"
        ),
    )
    parser.add_argument(
        "--v34",
        type=Path,
        default=Path(".appendix/style-comparisons/20260818-v34-v26-plus-one/final.mid"),
    )
    parser.add_argument(
        "--v9",
        type=Path,
        default=Path(".appendix/long-form-runs/20260816-long-form-v9-motif-style/final.mid"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".appendix/harmonic-style-v1/analysis.json"),
    )
    args = parser.parse_args(argv)
    result = analyze_harmonic_style(
        source_dir=args.source_dir,
        style_target_path=args.style_target,
        candidate_paths={"v26": args.v26, "v34": args.v34, "v9": args.v9},
    )
    atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "local_repair_decision": result["local_repair_decision"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
