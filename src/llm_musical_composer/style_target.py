"""参照曲または直感的な四値から、軸別の作風目標範囲を作る。"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from llm_musical_composer.run_state import (
    atomic_write_json,
    sha256_file,
    sha256_json,
)

AXES = ("density", "polyphony", "velocity", "register")
EXCLUDED_NAMES = frozenset({"rut.mid", "aimusic01.mid"})


def _round(value: float) -> float:
    return round(float(value), 8)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("at least one value is required")
    if len(ordered) == 1:
        return _round(ordered[0])
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return _round(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _summary(values: Sequence[float], *, anchor_value: float) -> dict[str, Any]:
    result = {
        "available_count": len(values),
        "minimum": _round(min(values)),
        "p25": _percentile(values, 0.25),
        "median": _percentile(values, 0.5),
        "p75": _percentile(values, 0.75),
        "maximum": _round(max(values)),
    }
    result["target_low"] = _round(min(result["p25"], anchor_value))
    result["target_high"] = _round(max(result["p75"], anchor_value))
    return result


@dataclass(frozen=True)
class StyleTargetRequest:
    """明示アンカーまたは四つのコーパス分位で目標を指定する。"""

    anchor_name: str | None = None
    density: float | None = None
    polyphony: float | None = None
    velocity: float | None = None
    register: float | None = None
    neighbor_count: int = 7

    def __post_init__(self) -> None:
        provided_controls = [axis for axis in AXES if getattr(self, axis) is not None]
        if self.anchor_name is not None and provided_controls:
            raise ValueError("anchor_name cannot be combined with automatic controls")
        for axis in AXES:
            value = getattr(self, axis)
            if value is None:
                continue
            if not math.isfinite(value):
                raise ValueError(f"{axis} must be finite")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{axis} must be between 0 and 1")
        if self.neighbor_count < 3:
            raise ValueError("neighbor_count must be at least three")

    def effective_quantiles(self) -> dict[str, float]:
        return {
            axis: 0.5 if getattr(self, axis) is None else float(getattr(self, axis))
            for axis in AXES
        }


def _centered(values: Sequence[float]) -> list[float]:
    center = statistics.median(values)
    return [_round(value - center) for value in values]


def extract_style_features(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """全曲で利用可能な経過時間4窓から、水準と曲内推移を分けて返す。"""
    if record.get("status") != "pass":
        raise ValueError("reference record is unavailable")
    try:
        report = record["coordinates"]["elapsed_time"]["resolutions"]["4"]
    except (KeyError, TypeError) as error:
        raise ValueError("elapsed-time four-window report is missing") from error
    if report.get("status") != "pass" or len(report.get("windows", [])) != 4:
        raise ValueError("elapsed-time four-window report is unavailable")
    windows = report["windows"]
    if [window.get("index") for window in windows] != [0, 1, 2, 3]:
        raise ValueError("elapsed-time four-window indices are invalid")
    try:
        onset_counts = [float(window["features"]["onset_count"]) for window in windows]
        polyphony = [float(window["features"]["polyphony_mean"]) for window in windows]
        velocity = [float(window["features"]["velocity_median"]) for window in windows]
        register = [float(window["features"]["pitch_median"]) for window in windows]
        note_count = float(record["note_count"])
        duration_ms = float(record["duration_ms"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("elapsed-time four-window features are incomplete") from error
    values = [*onset_counts, *polyphony, *velocity, *register, note_count, duration_ms]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("elapsed-time four-window features must be finite")
    if any(value < 0 for value in onset_counts):
        raise ValueError("elapsed-time four-window onset counts must be non-negative")
    if duration_ms <= 0 or sum(onset_counts) <= 0:
        raise ValueError("elapsed-time four-window features have no positive duration or density")
    if abs(sum(onset_counts) - note_count) > max(1.0, note_count * 1e-6):
        raise ValueError("elapsed-time four-window onset total differs from note_count")
    density_total = sum(onset_counts)
    return {
        "density": {
            "level": _round(note_count / (duration_ms / 1_000)),
            "shape": [_round(value / density_total) for value in onset_counts],
        },
        "polyphony": {
            "level": _round(statistics.median(polyphony)),
            "shape": _centered(polyphony),
        },
        "velocity": {
            "level": _round(statistics.median(velocity)),
            "shape": _centered(velocity),
        },
        "register": {
            "level": _round(statistics.median(register)),
            "shape": _centered(register),
        },
    }


def _scale(values: Sequence[float]) -> dict[str, float]:
    median = _percentile(values, 0.5)
    iqr = _percentile(values, 0.75) - _percentile(values, 0.25)
    return {"median": median, "scale": _round(iqr if iqr > 1e-9 else max(abs(median), 1.0))}


def _build_scales(feature_sets: Sequence[dict[str, dict[str, Any]]]) -> dict[str, Any]:
    return {
        axis: {
            "level": _scale([features[axis]["level"] for features in feature_sets]),
            "shape": [
                _scale([features[axis]["shape"][index] for features in feature_sets])
                for index in range(4)
            ],
        }
        for axis in AXES
    }


def _axis_distance(first: dict[str, Any], second: dict[str, Any], scale: dict[str, Any]) -> float:
    distances = [abs(first["level"] - second["level"]) / scale["level"]["scale"]]
    distances.extend(
        abs(left - right) / item["scale"]
        for left, right, item in zip(first["shape"], second["shape"], scale["shape"], strict=True)
    )
    return _round(max(distances))


def _distance_report(
    first: dict[str, dict[str, Any]],
    second: dict[str, dict[str, Any]],
    scales: dict[str, Any],
) -> dict[str, float]:
    return {axis: _axis_distance(first[axis], second[axis], scales[axis]) for axis in AXES}


def _rank_key(axis_distances: dict[str, float], name: str) -> tuple[float, float, str]:
    values = list(axis_distances.values())
    return max(values), statistics.median(values), name.casefold()


def _load_usable_records(
    records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    prepared: list[dict[str, Any]] = []
    unable: list[dict[str, str]] = []
    seen: dict[str, str] = {}
    for record in sorted(records, key=lambda item: str(item.get("name", "")).casefold()):
        name = str(record.get("name", ""))
        folded = name.casefold()
        if folded in seen:
            raise ValueError(f"duplicate reference name: {seen[folded]} and {name}")
        seen[folded] = name
        if folded in {excluded.casefold() for excluded in EXCLUDED_NAMES}:
            unable.append({"name": name, "status": "excluded"})
            continue
        try:
            features = extract_style_features(record)
        except ValueError as error:
            unable.append(
                {
                    "name": name,
                    "status": "unable_to_investigate",
                    "error": str(error),
                }
            )
            continue
        prepared.append({"name": name, "features": features})
    if len(prepared) < 2:
        raise ValueError("at least two usable reference records are required")
    return prepared, unable


def _select_anchor(
    prepared: Sequence[dict[str, Any]],
    request: StyleTargetRequest,
    scales: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_name = {item["name"].casefold(): item for item in prepared}
    if request.anchor_name is not None:
        folded = request.anchor_name.casefold()
        if folded in {name.casefold() for name in EXCLUDED_NAMES}:
            raise ValueError(f"excluded reference cannot be selected: {request.anchor_name}")
        if folded not in by_name:
            raise ValueError(f"reference not found among usable records: {request.anchor_name}")
        anchor = by_name[folded]
        return anchor, {"mode": "explicit_anchor", "anchor_name": anchor["name"]}

    requested_quantiles = request.effective_quantiles()
    targets = {
        axis: _percentile(
            [item["features"][axis]["level"] for item in prepared],
            requested_quantiles[axis],
        )
        for axis in AXES
    }
    ranked: list[tuple[tuple[float, float, str], dict[str, Any], dict[str, float]]] = []
    for item in prepared:
        distances = {
            axis: _round(
                abs(item["features"][axis]["level"] - targets[axis])
                / scales[axis]["level"]["scale"]
            )
            for axis in AXES
        }
        ranked.append((_rank_key(distances, item["name"]), item, distances))
    _, anchor, control_distances = min(ranked, key=lambda item: item[0])
    return anchor, {
        "mode": "automatic_controls",
        "anchor_name": anchor["name"],
        "requested_quantiles": requested_quantiles,
        "target_levels": targets,
        "control_axis_distances": control_distances,
        "anchor_basis": "requested level quantiles",
        "neighbor_basis": "anchor level and diagnostic coarse curves",
    }


def _build_axis_ranges(
    neighbors: Sequence[dict[str, Any]], anchor_features: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    return {
        axis: {
            "level": _summary(
                [item["features"][axis]["level"] for item in neighbors],
                anchor_value=anchor_features[axis]["level"],
            ),
            "shape": [
                _summary(
                    [item["features"][axis]["shape"][index] for item in neighbors],
                    anchor_value=anchor_features[axis]["shape"][index],
                )
                for index in range(4)
            ],
        }
        for axis in AXES
    }


def _prompt_target(axes: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scope": "neighbor level ranges only",
        "axis_targets": {
            axis: {
                "level": {
                    "range_low": axes[axis]["level"]["target_low"],
                    "median": axes[axis]["level"]["median"],
                    "range_high": axes[axis]["level"]["target_high"],
                },
            }
            for axis in AXES
        },
    }


def build_style_target(
    records: Iterable[dict[str, Any]], request: StyleTargetRequest
) -> dict[str, Any]:
    """アンカー近傍の範囲を、軸を合算せず決定的に返す。"""
    prepared, unable = _load_usable_records(records)
    feature_sets = [item["features"] for item in prepared]
    scales = _build_scales(feature_sets)
    if request.anchor_name is not None:
        unavailable = {
            item["name"].casefold(): item
            for item in unable
            if item["status"] == "unable_to_investigate"
        }
        if request.anchor_name.casefold() in unavailable:
            detail = unavailable[request.anchor_name.casefold()].get("error", "unknown reason")
            raise ValueError(f"reference is unable_to_investigate: {request.anchor_name}: {detail}")
    anchor, selection = _select_anchor(prepared, request, scales)
    if request.neighbor_count > len(prepared):
        raise ValueError(
            f"neighbor_count {request.neighbor_count} exceeds "
            f"usable reference count {len(prepared)}"
        )
    ranked_neighbors = []
    for item in prepared:
        distances = _distance_report(anchor["features"], item["features"], scales)
        ranked_neighbors.append(
            {
                "name": item["name"],
                "features": item["features"],
                "axis_distances": distances,
                "worst_axis_distance": _round(max(distances.values())),
                "median_axis_distance": _round(statistics.median(distances.values())),
            }
        )
    ranked_neighbors.sort(key=lambda item: _rank_key(item["axis_distances"], item["name"]))
    selected = ranked_neighbors[: request.neighbor_count]
    axes = _build_axis_ranges(selected, anchor["features"])
    return {
        "schema_version": 1,
        "curve_semantics": (
            "diagnostic four equal elapsed-time observations, "
            "not structural sections or prompt targets"
        ),
        "status": "pass",
        "request": asdict(request),
        "selection": selection,
        "source_count": len(prepared),
        "neighbor_count": len(selected),
        "neighbors": selected,
        "axes": axes,
        "prompt_target": _prompt_target(axes),
        "unable_to_investigate": unable,
        "excluded_names": sorted(EXCLUDED_NAMES),
        "prohibited_target_features": [
            "boundary_candidates",
            "hierarchy_depth",
            "repetition",
            "reference_note_sequences",
        ],
        "complementary_global_targets": [
            "pitch_order",
            "relative_timing",
            "texture",
            "activity",
            "material_development",
            "sustain",
        ],
    }


def _metric_distance(value: float, bounds: dict[str, Any]) -> float:
    scale = bounds["p75"] - bounds["p25"]
    if scale <= 1e-9:
        scale = max(abs(bounds["median"]), 1.0)
    if value < bounds["target_low"]:
        return _round((bounds["target_low"] - value) / scale)
    if value > bounds["target_high"]:
        return _round((value - bounds["target_high"]) / scale)
    return 0.0


def evaluate_style_features(
    features: dict[str, dict[str, Any]], target: dict[str, Any]
) -> dict[str, Any]:
    """プロンプト提示水準と非提示の参考曲線を混ぜずに評価する。"""
    level_results: dict[str, Any] = {}
    shape_results: dict[str, Any] = {}
    for axis in AXES:
        level_bounds = target["axes"][axis]["level"]
        level_distance = _metric_distance(features[axis]["level"], level_bounds)
        shape_distances = [
            _metric_distance(value, bounds)
            for value, bounds in zip(
                features[axis]["shape"], target["axes"][axis]["shape"], strict=True
            )
        ]
        level_results[axis] = {
            "candidate_value": _round(features[axis]["level"]),
            "target_range_low": level_bounds["target_low"],
            "target_range_high": level_bounds["target_high"],
            "inside_target_range": level_distance == 0,
            "normalized_distance": level_distance,
        }
        shape_results[axis] = {
            "candidate_values": [_round(value) for value in features[axis]["shape"]],
            "inside_reference_range": all(distance == 0 for distance in shape_distances),
            "normalized_distance": _round(max(shape_distances)),
            "dimension_distances": shape_distances,
        }
    return {
        "schema_version": 2,
        "prompted_level": {
            "axes": level_results,
            "inside_axis_count": sum(
                item["inside_target_range"] for item in level_results.values()
            ),
            "worst_axis_distance": _round(
                max(item["normalized_distance"] for item in level_results.values())
            ),
        },
        "diagnostic_shape": {
            "semantics": "four equal elapsed-time observations not shown to the generator",
            "axes": shape_results,
            "inside_axis_count": sum(
                item["inside_reference_range"] for item in shape_results.values()
            ),
            "worst_axis_distance": _round(
                max(item["normalized_distance"] for item in shape_results.values())
            ),
        },
    }


def evaluate_style_target_feasibility(
    prompt_target: dict[str, Any], *, duration_ms: int, max_note_count: int
) -> dict[str, Any]:
    """固定尺と音数上限から、提示した密度水準へ到達できるかを返す。"""
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    if max_note_count < 0:
        raise ValueError("max_note_count must be non-negative")
    try:
        bounds = prompt_target["axis_targets"]["density"]["level"]
        target_low = float(bounds["range_low"])
        target_high = float(bounds["range_high"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("prompt target has no density level range") from error
    if not all(math.isfinite(value) for value in (target_low, target_high)):
        raise ValueError("density level range must be finite")
    if target_low < 0 or target_high < target_low:
        raise ValueError("density level range is invalid")

    duration_seconds = duration_ms / 1_000
    maximum_reachable = _round(max_note_count / duration_seconds)
    minimum_required = math.ceil(target_low * duration_seconds - 1e-9)
    reachable = maximum_reachable >= target_low
    check = {
        "status": "reachable" if reachable else "unreachable",
        "target_range_low": _round(target_low),
        "target_range_high": _round(target_high),
        "maximum_reachable_level": maximum_reachable,
        "minimum_required_note_count": minimum_required,
        "shortfall_note_count": max(0, minimum_required - max_note_count),
    }
    return {
        "schema_version": 1,
        "status": "pass" if reachable else "conflict",
        "duration_ms": duration_ms,
        "max_note_count": max_note_count,
        "checks": {"density_level": check},
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at line {line_number}: {error.msg}") from error
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record at line {line_number} is not an object")
        records.append(value)
    return records


def write_style_target(
    records_path: Path, output_dir: Path, request: StyleTargetRequest
) -> dict[str, Any]:
    """作風目標と再現用manifestを確定書き込みする。"""
    records_path = Path(records_path)
    output_dir = Path(output_dir)
    target = build_style_target(_read_jsonl(records_path), request)
    target_path = output_dir / "style-target.json"
    atomic_write_json(target_path, target)
    manifest = {
        "schema_version": 1,
        "sha256": {
            "input_jsonl": sha256_file(records_path),
            "implementation": sha256_file(Path(__file__)),
            "request": sha256_json(asdict(request)),
            "style_target": sha256_file(target_path),
        },
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="参照曲の近傍から作風目標を作ります。")
    parser.add_argument(
        "--records",
        type=Path,
        default=Path(".appendix/reference-structure-analysis/files.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path(".appendix/style-target"))
    parser.add_argument("--anchor")
    for axis in AXES:
        parser.add_argument(f"--{axis}", type=float)
    parser.add_argument("--neighbor-count", type=int, default=7)
    args = parser.parse_args(argv)
    request = StyleTargetRequest(
        anchor_name=args.anchor,
        density=args.density,
        polyphony=args.polyphony,
        velocity=args.velocity,
        register=args.register,
        neighbor_count=args.neighbor_count,
    )
    target = write_style_target(args.records, args.output_dir, request)
    print(json.dumps(target["selection"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
