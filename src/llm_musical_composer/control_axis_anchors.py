"""参照SMFから3軸の基準曲候補を再現可能に選ぶ。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any

from llm_musical_composer.evaluator_registry import (
    assert_evaluator_use,
    load_evaluator_registry,
)
from llm_musical_composer.reference_profile import (
    ReferencePiece,
    load_reference_piece,
)
from llm_musical_composer.run_state import (
    atomic_write_bytes,
    atomic_write_json,
    sha256_file,
)

AXES = ("あかるさ", "うごき", "パワー")
AXIS_EVALUATOR_IDS = {
    "あかるさ": "control_axis_brightness_descriptors_v1",
    "うごき": "control_axis_motion_descriptors_v1",
    "パワー": "control_axis_power_descriptors_v1",
}
EXCLUDED_NAMES = frozenset({"rut.mid", "aimusic01.mid"})

_MAJOR_PROFILE = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
_MINOR_PROFILE = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)


def _round(value: float) -> float:
    return round(float(value), 8)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _pearson(first: Sequence[float], second: Sequence[float]) -> float:
    first_mean = statistics.fmean(first)
    second_mean = statistics.fmean(second)
    centered_first = [value - first_mean for value in first]
    centered_second = [value - second_mean for value in second]
    denominator = math.sqrt(
        sum(value * value for value in centered_first)
        * sum(value * value for value in centered_second)
    )
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(centered_first, centered_second, strict=True)) / denominator


def _best_profile(pitch_classes: Sequence[float], template: Sequence[float]) -> tuple[float, int]:
    values = [
        (_pearson([pitch_classes[(root + offset) % 12] for offset in range(12)], template), root)
        for root in range(12)
    ]
    return max(values, key=lambda item: (item[0], -item[1]))


def fixed_tonic_brightness_metrics(
    pitch_classes: Sequence[float],
    tonic_pitch_class: int,
) -> dict[str, dict[str, Any]]:
    """宣言済み主音を基準に、生成候補の調性的な明暗を記述する。"""
    if len(pitch_classes) != 12:
        raise ValueError("pitch_classes must contain 12 duration weights")
    if not 0 <= tonic_pitch_class <= 11:
        raise ValueError("tonic_pitch_class must be from 0 through 11")
    relative = [float(pitch_classes[(tonic_pitch_class + offset) % 12]) for offset in range(12)]
    major_correlation = _pearson(relative, _MAJOR_PROFILE)
    minor_correlation = _pearson(relative, _MINOR_PROFILE)
    bright_weight = sum(relative[degree] for degree in (4, 9, 11))
    dark_weight = sum(relative[degree] for degree in (3, 8, 10))
    support = bright_weight + dark_weight
    modal_balance = (bright_weight - dark_weight) / support if support > 0 else 0.0
    return {
        "major_minor_profile_margin": _available(
            major_correlation - minor_correlation,
            positive_direction="higher",
            unit="correlation_difference",
            support=round(sum(relative)),
        ),
        "modal_degree_balance": _available(
            modal_balance,
            positive_direction="higher",
            unit="weighted_ratio",
            support=round(support),
        ),
    }


def _available(
    value: float,
    *,
    positive_direction: str,
    unit: str,
    support: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "available",
        "value": _round(value),
        "positive_direction": positive_direction,
        "unit": unit,
    }
    if support is not None:
        result["support"] = support
    return result


def _unavailable(*, positive_direction: str, unit: str, reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": reason,
        "positive_direction": positive_direction,
        "unit": unit,
    }


def _brightness(piece: ReferencePiece) -> dict[str, Any]:
    pitch_classes = [0.0] * 12
    for note in piece.notes:
        pitch_classes[note.pitch % 12] += note.duration_ms
    major_correlation, major_root = _best_profile(pitch_classes, _MAJOR_PROFILE)
    minor_correlation, minor_root = _best_profile(pitch_classes, _MINOR_PROFILE)
    root = major_root if major_correlation >= minor_correlation else minor_root
    bright_weight = sum(pitch_classes[(root + degree) % 12] for degree in (4, 9, 11))
    dark_weight = sum(pitch_classes[(root + degree) % 12] for degree in (3, 8, 10))
    modal_support = bright_weight + dark_weight
    modal_metric = (
        _available(
            (bright_weight - dark_weight) / modal_support,
            positive_direction="higher",
            unit="weighted_ratio",
            support=round(modal_support),
        )
        if modal_support > 0
        else _unavailable(
            positive_direction="higher",
            unit="weighted_ratio",
            reason="no major-specific or minor-specific scale-degree evidence",
        )
    )
    return {
        "status": "available",
        "metrics": {
            "major_minor_profile_margin": _available(
                major_correlation - minor_correlation,
                positive_direction="higher",
                unit="correlation_difference",
                support=len(piece.notes),
            ),
            "modal_degree_balance": modal_metric,
        },
        "confidence": {
            "tonal_profile_correlation": _round(max(major_correlation, minor_correlation)),
            "estimated_tonic_pitch_class": root,
            "estimated_mode": "major" if major_correlation >= minor_correlation else "minor",
        },
    }


def _merged_coverage(intervals: Sequence[tuple[int, int]]) -> int:
    total = 0
    start, end = sorted(intervals)[0]
    for next_start, next_end in sorted(intervals)[1:]:
        if next_start > end:
            total += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return total + end - start


def _motion(piece: ReferencePiece) -> dict[str, Any]:
    attacks = sorted({note.onset_ms for note in piece.notes})
    iois = [second - first for first, second in pairwise(attacks)]
    next_attack = dict(pairwise(attacks))
    release_ratios = []
    for onset, following in next_attack.items():
        longest = max(note.duration_ms for note in piece.notes if note.onset_ms == onset)
        release_ratios.append(longest / (following - onset))
    start = min(note.onset_ms for note in piece.notes)
    end = max(note.onset_ms + note.duration_ms for note in piece.notes)
    coverage = _merged_coverage(
        [(note.onset_ms, note.onset_ms + note.duration_ms) for note in piece.notes]
    )
    return {
        "status": "available",
        "metrics": {
            "median_ioi_ms": _available(
                statistics.median(iois),
                positive_direction="lower",
                unit="milliseconds",
                support=len(iois),
            ),
            "silence_ratio": _available(
                max(0.0, 1 - coverage / (end - start)),
                positive_direction="lower",
                unit="time_ratio",
                support=len(piece.notes),
            ),
            "detached_attack_ratio": _available(
                sum(value < 0.85 for value in release_ratios) / len(release_ratios),
                positive_direction="higher",
                unit="attack_ratio",
                support=len(release_ratios),
            ),
        },
    }


def _power(piece: ReferencePiece) -> dict[str, Any]:
    velocities = [note.velocity for note in piece.notes]
    pitches = [note.pitch for note in piece.notes]
    pitch_median = statistics.median(pitches)
    total_duration = sum(note.duration_ms for note in piece.notes)
    bass_duration = sum(note.duration_ms for note in piece.notes if note.pitch <= pitch_median - 7)
    attacks: dict[int, list[int]] = {}
    for note in piece.notes:
        attacks.setdefault(note.onset_ms, []).append(note.pitch)
    consonant_wide_attacks = 0
    for attack_pitches in attacks.values():
        intervals = {abs(first - second) % 12 for first, second in combinations(attack_pitches, 2)}
        if max(attack_pitches) - min(attack_pitches) >= 7 and intervals & {3, 4, 5, 7, 8, 9}:
            consonant_wide_attacks += 1
    return {
        "status": "available",
        "metrics": {
            "velocity_iqr": _available(
                _percentile(velocities, 0.75) - _percentile(velocities, 0.25),
                positive_direction="higher",
                unit="MIDI_velocity_difference",
                support=len(velocities),
            ),
            "accent_contrast": _available(
                _percentile(velocities, 0.90) - _percentile(velocities, 0.50),
                positive_direction="higher",
                unit="MIDI_velocity_difference",
                support=len(velocities),
            ),
            "relative_bass_support": _available(
                bass_duration / total_duration,
                positive_direction="higher",
                unit="note_duration_ratio",
                support=sum(note.pitch <= pitch_median - 7 for note in piece.notes),
            ),
            "wide_consonant_attack_ratio": _available(
                consonant_wide_attacks / len(attacks),
                positive_direction="higher",
                unit="attack_ratio",
                support=len(attacks),
            ),
        },
    }


def extract_control_axis_descriptors(piece: ReferencePiece) -> dict[str, Any]:
    """直接観測できる記述子だけを返し、復元不能な特徴を明示する。"""

    return {
        "schema_version": 1,
        "status": "available",
        "name": piece.name,
        "axes": {
            "あかるさ": _brightness(piece),
            "うごき": _motion(piece),
            "パワー": _power(piece),
        },
        "unavailable_features": [
            {
                "id": "score_relative_rubato",
                "reason": "score-time positions are not present in ReferencePiece",
            },
            {
                "id": "voice_convergence",
                "reason": "track/channel/voice labels are not present in ReferencePiece",
            },
            {
                "id": "performance_intention",
                "reason": "performer intention cannot be observed directly from SMF events",
            },
        ],
    }


def build_descriptor_record(piece: ReferencePiece) -> dict[str, Any]:
    """曲全体の記述子と8分割leave-one-block-out記述子を構築する。"""

    descriptors = extract_control_axis_descriptors(piece)
    attacks = sorted({note.onset_ms for note in piece.notes})
    evidence = {
        "name": piece.name,
        "attack_count": len(attacks),
        "note_count": len(piece.notes),
        "duration_ms": max(note.onset_ms + note.duration_ms for note in piece.notes),
        "pedal_event_count": len(piece.pedals),
        "descriptors": descriptors,
    }
    if len(attacks) < 8:
        return {
            **evidence,
            "status": "unavailable",
            "reason": "piece cannot be split into eight contiguous onset blocks",
            "jackknife": [],
        }
    blocks: list[set[int]] = []
    for index in range(8):
        start = len(attacks) * index // 8
        stop = len(attacks) * (index + 1) // 8
        blocks.append(set(attacks[start:stop]))
    jackknife = []
    for index, omitted in enumerate(blocks):
        reduced = ReferencePiece(
            piece.name,
            tuple(note for note in piece.notes if note.onset_ms not in omitted),
            piece.pedals,
        )
        jackknife.append(
            {
                "omitted_block": index,
                "omitted_attack_count": len(omitted),
                "axes": extract_control_axis_descriptors(reduced)["axes"],
            }
        )
    return {
        **evidence,
        "status": "available",
        "jackknife": jackknife,
    }


def _metric_contract(record: Mapping[str, Any], axis: str) -> Mapping[str, Any]:
    return record["descriptors"]["axes"][axis]["metrics"]


def _is_better(value: float, other: float, direction: str, side: str) -> bool:
    high_is_larger = direction == "higher"
    if side == "low":
        high_is_larger = not high_is_larger
    return value > other if high_is_larger else value < other


def _rank(value: float, population: Sequence[float], direction: str, side: str) -> int:
    if side == "middle":
        center = statistics.median(population)
        distance = abs(value - center)
        return 1 + sum(abs(other - center) < distance - 1e-12 for other in population)
    return 1 + sum(
        _is_better(other, value, direction, side)
        for other in population
        if abs(other - value) > 1e-12
    )


def _candidate_rows(records: Sequence[Mapping[str, Any]], axis: str, side: str) -> list[dict]:
    metric_names = sorted(
        {
            name
            for record in records
            if record.get("status") == "available"
            for name in _metric_contract(record, axis)
        }
    )
    eligible = [
        record
        for record in records
        if record.get("status") == "available"
        and all(
            _metric_contract(record, axis).get(name, {}).get("status") == "available"
            for name in metric_names
        )
        and len(record.get("jackknife", [])) == 8
        and all(
            all(
                variant["axes"][axis]["metrics"].get(name, {}).get("status") == "available"
                for name in metric_names
            )
            for variant in record["jackknife"]
        )
    ]
    populations = {
        name: [float(_metric_contract(record, axis)[name]["value"]) for record in eligible]
        for name in metric_names
    }
    rows = []
    for record in eligible:
        ranks = {
            name: _rank(
                float(_metric_contract(record, axis)[name]["value"]),
                populations[name],
                _metric_contract(record, axis)[name]["positive_direction"],
                side,
            )
            for name in metric_names
        }
        jackknife_worst = [max(ranks.values())]
        for variant in record["jackknife"]:
            variant_ranks = [
                _rank(
                    float(variant["axes"][axis]["metrics"][name]["value"]),
                    populations[name],
                    _metric_contract(record, axis)[name]["positive_direction"],
                    side,
                )
                for name in metric_names
            ]
            jackknife_worst.append(max(variant_ranks))
        rows.append(
            {
                "name": record["name"],
                "directional_ranks": ranks,
                "worst_rank": max(ranks.values()),
                "median_rank": _round(statistics.median(ranks.values())),
                "rank_width": max(jackknife_worst) - min(jackknife_worst),
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            item["worst_rank"],
            item["median_rank"],
            item["rank_width"],
            item["name"].casefold(),
            item["name"],
        ),
    )


def select_axis_candidates(
    records: Sequence[Mapping[str, Any]],
    axis: str,
    *,
    registry: Mapping[str, Any],
    candidate_count: int = 2,
) -> dict[str, list[dict]]:
    """単一総合点を作らず、最悪順位、中央値、順位幅の順で候補を選ぶ。"""

    if axis not in AXES:
        raise ValueError(f"unknown control axis: {axis}")
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    assert_evaluator_use(registry, [AXIS_EVALUATOR_IDS[axis]], purpose="ranking")
    low = _candidate_rows(records, axis, "low")[:candidate_count]
    low_names = {item["name"] for item in low}
    high = [
        item for item in _candidate_rows(records, axis, "high") if item["name"] not in low_names
    ][:candidate_count]
    reserved = low_names | {item["name"] for item in high}
    middle = [
        item for item in _candidate_rows(records, axis, "middle") if item["name"] not in reserved
    ][:candidate_count]
    return {"low": low, "middle": middle, "high": high}


class ControlAxisAnchorError(ValueError):
    """基準曲の校正を通常結果として扱えない場合に送出する。"""


def verify_reference_artifacts(
    *,
    source_dir: Path,
    reference_dir: Path,
    implementation_dir: Path,
    expected_count: int = 232,
) -> dict[str, Any]:
    """保存済み参照プロファイルの全入力と実装と出力をSHA-256で検査する。"""

    try:
        manifest = json.loads((reference_dir / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((reference_dir / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ControlAxisAnchorError(f"unable to load reference artifacts: {error}") from error
    source_paths = sorted(
        (
            path
            for path in source_dir.glob("*.mid")
            if path.name.casefold() not in {name.casefold() for name in EXCLUDED_NAMES}
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )
    if len(source_paths) != expected_count or summary.get("source_count") != expected_count:
        raise ControlAxisAnchorError("reference source count does not match expected count")
    if summary.get("status") != "pass" or summary.get("unable_to_investigate"):
        raise ControlAxisAnchorError("reference summary is not a complete passing result")
    expected_exclusions = sorted(EXCLUDED_NAMES, key=str.casefold)
    actual_exclusions = sorted(
        (item.get("name") for item in summary.get("excluded", [])), key=str.casefold
    )
    if actual_exclusions != expected_exclusions:
        raise ControlAxisAnchorError("reference exclusions do not match the approved exclusions")
    input_hashes = manifest.get("inputs", {})
    if set(input_hashes) != {path.name for path in source_paths}:
        raise ControlAxisAnchorError("reference manifest input names do not match source SMFs")
    for path in source_paths:
        if sha256_file(path).casefold() != str(input_hashes[path.name]).casefold():
            raise ControlAxisAnchorError(f"reference input SHA-256 mismatch: {path.name}")
    implementation_hashes = manifest.get("implementations", {})
    for name in ("reference_profile.py", "reference_controls.py"):
        path = implementation_dir / name
        if sha256_file(path).casefold() != str(implementation_hashes.get(name, "")).casefold():
            raise ControlAxisAnchorError(f"reference implementation SHA-256 mismatch: {name}")
    for name in ("files.jsonl", "summary.json"):
        path = reference_dir / name
        expected = str(manifest.get("outputs", {}).get(name, ""))
        if sha256_file(path).casefold() != expected.casefold():
            raise ControlAxisAnchorError(f"reference output SHA-256 mismatch: {name}")
    return {
        "status": "pass",
        "source_count": len(source_paths),
        "source_paths": source_paths,
        "manifest_sha256": sha256_file(reference_dir / "manifest.json"),
    }


def _average_tie_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(indexed):
        stop = position + 1
        while stop < len(indexed) and abs(indexed[stop][1] - indexed[position][1]) <= 1e-12:
            stop += 1
        average_rank = (position + 1 + stop) / 2
        for original_index, _ in indexed[position:stop]:
            ranks[original_index] = average_rank
        position = stop
    return ranks


def _spearman(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second) or len(first) < 2:
        raise ControlAxisAnchorError("Spearman correlation requires paired values")
    return _pearson(_average_tie_ranks(first), _average_tie_ranks(second))


def analyze_axis_separation(
    records: Sequence[Mapping[str, Any]], candidates: Mapping[str, Mapping[str, Sequence[dict]]]
) -> dict[str, Any]:
    high_rank_maps = {
        axis: {item["name"]: item["median_rank"] for item in _candidate_rows(records, axis, "high")}
        for axis in AXES
    }
    pairs = []
    stop_reasons = []
    for first_axis, second_axis in combinations(AXES, 2):
        names = sorted(
            set(high_rank_maps[first_axis]) & set(high_rank_maps[second_axis]),
            key=lambda name: (name.casefold(), name),
        )
        correlation = _spearman(
            [high_rank_maps[first_axis][name] for name in names],
            [high_rank_maps[second_axis][name] for name in names],
        )
        first_endpoints = {
            item["name"] for side in ("low", "high") for item in candidates[first_axis][side]
        }
        second_endpoints = {
            item["name"] for side in ("low", "high") for item in candidates[second_axis][side]
        }
        overlap = sorted(
            first_endpoints & second_endpoints, key=lambda name: (name.casefold(), name)
        )
        denominator = min(len(first_endpoints), len(second_endpoints))
        pair = {
            "axes": [first_axis, second_axis],
            "eligible_pair_count": len(names),
            "spearman_rank_correlation": _round(correlation),
            "endpoint_overlap": overlap,
            "endpoint_overlap_count": len(overlap),
            "endpoint_denominator": denominator,
        }
        pairs.append(pair)
        if abs(correlation) >= 0.8:
            stop_reasons.append(
                f"{first_axis} and {second_axis} have absolute Spearman correlation >= 0.8"
            )
        if denominator and len(overlap) > denominator / 2:
            stop_reasons.append(
                f"{first_axis} and {second_axis} share more than half of endpoint candidates"
            )
    return {
        "status": "fail" if stop_reasons else "pass",
        "pairs": pairs,
        "stop_reasons": stop_reasons,
        "thresholds_are_diagnostic_only": True,
    }


def _listening_materials(
    *,
    source_dir: Path,
    output_dir: Path,
    candidates: Mapping[str, Mapping[str, Sequence[dict]]],
    axes: Sequence[str] = AXES,
    candidate_index: int = 0,
    alias_prefix: str = "S",
    directory_name: str = "listening",
    title: str = "3軸基準曲の匿名試聴票",
) -> tuple[list[dict[str, str]], str]:
    if not axes or len(set(axes)) != len(axes) or any(axis not in AXES for axis in axes):
        raise ControlAxisAnchorError("listening axes must be unique known control axes")
    if candidate_index < 0:
        raise ControlAxisAnchorError("candidate_index must be non-negative")
    listening_dir = output_dir / directory_name
    listening_dir.mkdir(parents=True, exist_ok=True)
    mapping = []
    lines = [
        f"# {title}",
        "",
        "各組には3曲あります。曲名、機械が予想した項目、機械が予想した順番は隠しています。",
        "3曲を聴き比べて、次の二つだけ答えてください。",
        "",
        "1. 3曲の間で、いちばん違って聞こえる項目を一つ選んでください。",
        "   - あかるさ: 暗い感じから、明るい感じまで",
        "   - うごき: 落ち着いて止まっている感じから、前へ進んで流れる感じまで",
        "   - パワー: 軽く控えめな感じから、力強く重厚な感じまで",
        "   - どれか分からない",
        "2. 選んだ項目について、3曲を弱いほうから強いほうへの順番に並べてください。",
        "   - あかるさなら、暗いほうから明るいほうへ並べます。",
        "   - うごきなら、落ち着いているほうから前へ進むほうへ並べます。",
        "   - パワーなら、控えめなほうから力強いほうへ並べます。",
        "",
        "分からないときは、無理に選んだり並べたりせず、「分からない」と書いてください。",
        "",
    ]
    for set_index, axis in enumerate(axes, start=1):
        if any(
            len(candidates[axis][side]) <= candidate_index for side in ("low", "middle", "high")
        ):
            raise ControlAxisAnchorError(f"axis {axis!r} has no requested listening candidate")
        selected = [
            (side, candidates[axis][side][candidate_index]["name"])
            for side in ("low", "middle", "high")
        ]
        selected.sort(
            key=lambda item: hashlib.sha256(f"{axis}\0{item[0]}\0{item[1]}".encode()).hexdigest()
        )
        lines.extend([f"## 組{set_index}", ""])
        for item_index, (side, name) in enumerate(selected, start=1):
            alias = f"{alias_prefix}{set_index}-{item_index}"
            relative_path = f"{directory_name}/{alias}.mid"
            atomic_write_bytes(output_dir / relative_path, (source_dir / name).read_bytes())
            mapping.append(
                {
                    "alias": alias,
                    "axis": axis,
                    "side": side,
                    "source_name": name,
                    "path": relative_path,
                }
            )
            lines.append(f"- `{alias}`: `{relative_path}`")
        lines.extend(
            [
                "",
                "回答:",
                "",
                "- いちばん違って聞こえる項目。あかるさ・うごき・パワー・分からないから選ぶ:",
                "- 弱いほうから強いほうへの順番:",
                "- 順番を決められない曲:",
                "- 選択肢以外で強く感じた違い:",
                "- 明らかに不自然だと感じた曲:",
                "",
            ]
        )
    return mapping, "\n".join(lines)


def build_backup_listening_round(
    *, source_dir: Path, output_dir: Path, axes: Sequence[str]
) -> dict[str, Any]:
    """初回不合格軸だけを保存済み第2候補で一度再試行する。"""

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    try:
        candidates = json.loads((output_dir / "candidates.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ControlAxisAnchorError(f"unable to load anchor candidates: {error}") from error
    if candidates.get("status") != "pass":
        raise ControlAxisAnchorError("backup listening requires passing machine candidates")
    requested_axes = tuple(axes)
    mapping, sheet = _listening_materials(
        source_dir=source_dir,
        output_dir=output_dir,
        candidates=candidates["axes"],
        axes=requested_axes,
        candidate_index=1,
        alias_prefix="B",
        directory_name="listening-backup",
        title="基準曲の予備候補を使う匿名試聴票",
    )
    result = {
        "schema_version": 1,
        "status": "pass",
        "calibration_status": "backup_listening_pending",
        "axes": list(requested_axes),
        "mapping": mapping,
    }
    mapping_path = output_dir / "backup-listening-mapping.json"
    sheet_path = output_dir / "listening-sheet-backup.md"
    atomic_write_json(mapping_path, result)
    atomic_write_bytes(sheet_path, sheet.encode("utf-8"))
    output_paths = [mapping_path, sheet_path]
    output_paths.extend(sorted((output_dir / "listening-backup").glob("*.mid")))
    selected_sources = sorted(
        {source_dir / item["source_name"] for item in mapping},
        key=lambda path: (path.name.casefold(), path.name),
    )
    manifest = {
        "schema_version": 1,
        "status": "pass",
        "axes": list(requested_axes),
        "inputs": {path.name: sha256_file(path) for path in selected_sources},
        "implementations": {"control_axis_anchors.py": sha256_file(Path(__file__))},
        "outputs": {
            path.relative_to(output_dir).as_posix(): sha256_file(path) for path in output_paths
        },
    }
    atomic_write_json(output_dir / "backup-manifest.json", manifest)
    return result


def build_control_axis_anchors(
    *,
    source_dir: Path,
    reference_dir: Path,
    output_dir: Path,
    registry_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    project_root = Path(project_root)
    source_dir = Path(source_dir)
    reference_dir = Path(reference_dir)
    output_dir = Path(output_dir)
    verification = verify_reference_artifacts(
        source_dir=source_dir,
        reference_dir=reference_dir,
        implementation_dir=project_root / "src" / "llm_musical_composer",
    )
    registry = load_evaluator_registry(registry_path, project_root=project_root)
    records = [
        build_descriptor_record(load_reference_piece(path)) for path in verification["source_paths"]
    ]
    candidates = {axis: select_axis_candidates(records, axis, registry=registry) for axis in AXES}
    incomplete_axes = [
        axis
        for axis in AXES
        if any(len(candidates[axis][side]) < 2 for side in ("low", "middle", "high"))
    ]
    separation = analyze_axis_separation(records, candidates)
    status = "pass" if not incomplete_axes and separation["status"] == "pass" else "fail"
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor_bytes = b"".join(
        (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for record in records
    )
    atomic_write_bytes(output_dir / "descriptors.jsonl", descriptor_bytes)
    candidate_result: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "calibration_status": "uncalibrated",
        "axes": candidates,
        "incomplete_axes": incomplete_axes,
        "separation": separation,
        "signed_values_assigned_to_corpus": False,
    }
    if status == "pass":
        listening_mapping, listening_sheet = _listening_materials(
            source_dir=source_dir,
            output_dir=output_dir,
            candidates=candidates,
        )
        candidate_result["listening_mapping"] = listening_mapping
    else:
        listening_sheet = "\n".join(
            [
                "# 3軸基準曲の匿名試聴票",
                "",
                "機械選別が停止条件に該当したため、試聴候補は公開しません。",
                "`candidates.json`の`incomplete_axes`と`separation.stop_reasons`を確認してください。",
                "",
            ]
        )
    atomic_write_json(output_dir / "candidates.json", candidate_result)
    atomic_write_bytes(output_dir / "listening-sheet.md", listening_sheet.encode("utf-8"))
    output_paths = [
        output_dir / "descriptors.jsonl",
        output_dir / "candidates.json",
        output_dir / "listening-sheet.md",
    ]
    output_paths.extend(sorted((output_dir / "listening").glob("*.mid")))
    manifest = {
        "schema_version": 1,
        "status": status,
        "reference_manifest_sha256": verification["manifest_sha256"],
        "inputs": {path.name: sha256_file(path) for path in verification["source_paths"]},
        "implementations": {
            "control_axis_anchors.py": sha256_file(Path(__file__)),
            "evaluator-registry-v1.json": sha256_file(registry_path),
        },
        "outputs": {
            path.relative_to(output_dir).as_posix(): sha256_file(path) for path in output_paths
        },
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return candidate_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build three control-axis anchor candidates")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--backup-axes", nargs="+", choices=AXES)
    args = parser.parse_args(argv)
    if args.backup_axes:
        result = build_backup_listening_round(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            axes=args.backup_axes,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "calibration_status": result["calibration_status"],
                    "axes": result["axes"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    result = build_control_axis_anchors(
        source_dir=args.source_dir,
        reference_dir=args.reference_dir,
        output_dir=args.output_dir,
        registry_path=args.registry,
        project_root=args.project_root,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "calibration_status": result["calibration_status"],
                "incomplete_axes": result["incomplete_axes"],
                "stop_reasons": result["separation"]["stop_reasons"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
