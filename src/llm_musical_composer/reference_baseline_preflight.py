"""通常生成の参照候補を採用する前に、コーパス内の支持と被覆を測る。"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

from llm_musical_composer.reference_generation_target import (
    build_reference_generation_target,
)
from llm_musical_composer.reference_profile import (
    FEATURE_GROUPS,
    profile_distances,
    select_default_reference,
)
from llm_musical_composer.run_state import (
    atomic_write_bytes,
    atomic_write_json,
    sha256_file,
)

ANALYSIS_VERSION = "corpus-baseline-preflight-v1"
CONTROL_NAMES = ("あかるさ", "高さ", "発音頻度")
K_VALUES = (5, 7, 11)
CANDIDATE_FRACTIONS = (0.05, 0.10, 0.20, 0.25, 0.50, 0.75)
EVALUATION_GROUPS_SHA256 = "3b5c54805a44d0664112f383a0122dc815ecbcb3a05341c1b3ea6a1b54a864be"


class PreflightError(ValueError):
    """入力または分析契約を満たせない。"""


def _name_key(name: str) -> tuple[str, str]:
    return name.casefold(), name


def _pair_key(first: str, second: str) -> tuple[str, str]:
    if first == second:
        raise PreflightError("self distance is not a corpus pair")
    return tuple(sorted((first, second), key=_name_key))  # type: ignore[return-value]


def _finite_number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreflightError(f"{location} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise PreflightError(f"{location} must be finite")
    return result


def high_precision_profile_distances(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, float]:
    """保存値から群距離を再計算し、成果物出力まで丸めない。"""
    if first.get("status") != "pass" or second.get("status") != "pass":
        raise PreflightError("reference profile status is not pass")
    first_groups = first.get("feature_groups")
    second_groups = second.get("feature_groups")
    if not isinstance(first_groups, Mapping) or not isinstance(second_groups, Mapping):
        raise PreflightError("reference profile feature groups are invalid")
    if set(first_groups) != set(FEATURE_GROUPS) or set(second_groups) != set(FEATURE_GROUPS):
        raise PreflightError("reference profile feature groups are invalid")

    result: dict[str, float] = {}
    for group in FEATURE_GROUPS:
        first_group = first_groups[group]
        second_group = second_groups[group]
        if not isinstance(first_group, Mapping) or not isinstance(second_group, Mapping):
            raise PreflightError("reference profile group is invalid")
        first_metrics = first_group.get("metrics")
        second_metrics = second_group.get("metrics")
        if not isinstance(first_metrics, Mapping) or not isinstance(second_metrics, Mapping):
            raise PreflightError("reference profile metrics are invalid")
        if set(first_metrics) != set(second_metrics):
            raise PreflightError("reference profile metric sets differ")
        metric_distances: list[float] = []
        for metric_name in sorted(first_metrics):
            one = first_metrics[metric_name]
            two = second_metrics[metric_name]
            if not isinstance(one, Mapping) or not isinstance(two, Mapping):
                raise PreflightError("reference profile metric is invalid")
            kind = one.get("kind")
            if kind != two.get("kind") or kind not in {"scalar", "distribution"}:
                raise PreflightError("reference profile metric kinds differ")
            one_values = one.get("values")
            two_values = two.get("values")
            if not isinstance(one_values, list) or not isinstance(two_values, list):
                raise PreflightError("reference profile metric values are invalid")
            if not one_values or len(one_values) != len(two_values):
                raise PreflightError("reference profile metric dimensions differ")
            distance = sum(
                abs(
                    _finite_number(left, f"{group}.{metric_name}")
                    - _finite_number(right, f"{group}.{metric_name}")
                )
                for left, right in zip(one_values, two_values, strict=True)
            )
            if kind == "distribution":
                distance /= 2.0
            metric_distances.append(distance)
        result[group] = statistics.mean(metric_distances)

    rounded = {group: round(value, 8) for group, value in result.items()}
    if rounded != profile_distances(first, second):
        raise PreflightError("high precision distance differs from existing profile distance")
    return result


def empirical_midrank(value: float, values: Sequence[float]) -> float:
    """同値へ平均順位を与える経験分位を返す。"""
    if not values:
        raise PreflightError("empirical distribution is empty")
    if value not in values:
        raise PreflightError("value must be present in empirical distribution")
    lower = sum(item < value for item in values)
    equal = sum(item == value for item in values)
    return (lower + 0.5 * equal) / len(values)


def _empirical_lookup(values: Sequence[float]) -> dict[float, float]:
    """同値ごとの経験分位を一度だけ計算する。"""
    if not values:
        raise PreflightError("empirical distribution is empty")
    ordered = sorted(values)
    result: dict[float, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end] == ordered[start]:
            end += 1
        result[ordered[start]] = (start + 0.5 * (end - start)) / len(ordered)
        start = end
    return result


def build_pair_distance_model(
    profiles: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """全曲対について生距離、経験分位最大、中央値倍率最大を作る。"""
    names = sorted(profiles, key=_name_key)
    if len(names) < 3:
        raise PreflightError("at least three profiles are required")
    raw: dict[tuple[str, str], dict[str, float]] = {}
    for first, second in combinations(names, 2):
        raw[(first, second)] = high_precision_profile_distances(profiles[first], profiles[second])
    group_values = {
        group: [distances[group] for distances in raw.values()] for group in FEATURE_GROUPS
    }
    quantile_lookups = {group: _empirical_lookup(values) for group, values in group_values.items()}
    medians = {group: statistics.median(values) for group, values in group_values.items()}
    if any(not math.isfinite(value) or value <= 0 for value in medians.values()):
        raise PreflightError("group median must be finite and positive")

    group_quantiles: dict[tuple[str, str], dict[str, float]] = {}
    empirical: dict[tuple[str, str], float] = {}
    median_normalized: dict[tuple[str, str], float] = {}
    for pair, distances in raw.items():
        quantiles = {group: quantile_lookups[group][distances[group]] for group in FEATURE_GROUPS}
        group_quantiles[pair] = quantiles
        empirical[pair] = max(quantiles.values())
        median_normalized[pair] = max(distances[group] / medians[group] for group in FEATURE_GROUPS)
    return {
        "raw": raw,
        "group_quantiles": group_quantiles,
        "empirical_quantile": empirical,
        "median_normalized": median_normalized,
        "group_medians": medians,
    }


def local_radii(
    pair_distances: Mapping[tuple[str, str], float],
    profiles_or_names: Mapping[str, object] | Sequence[str],
    *,
    k: int,
) -> dict[str, float]:
    """各曲のk番目に近い相手までの半径を返す。"""
    names = sorted(profiles_or_names, key=_name_key)
    if k < 1 or k >= len(names):
        raise PreflightError("k must be smaller than the population")
    result: dict[str, float] = {}
    for name in names:
        distances = sorted(
            pair_distances[_pair_key(name, other)] for other in names if other != name
        )
        result[name] = distances[k - 1]
    return result


def select_candidates(radii: Mapping[str, float], *, fraction: float) -> dict[str, Any]:
    """nearest-rank境界を使い、境界同値を全て含める。"""
    if not radii or not 0 < fraction <= 1:
        raise PreflightError("candidate fraction must be in (0, 1]")
    requested = math.ceil(fraction * len(radii))
    threshold = sorted(radii.values())[requested - 1]
    names = sorted((name for name, value in radii.items() if value <= threshold), key=_name_key)
    return {
        "fraction": fraction,
        "requested_count": requested,
        "actual_count": len(names),
        "threshold": threshold,
        "names": names,
    }


def select_farthest_first_from_pairs(
    records: Sequence[Mapping[str, Any]],
    raw_pairs: Mapping[tuple[str, str], Mapping[str, float]],
    *,
    count: int,
) -> list[dict[str, Any]]:
    """既存farthest-first方式を、計算済み曲対距離から再現する。"""
    profiles: dict[str, Mapping[str, Any]] = {}
    conflicts_by_fingerprint: dict[str, list[str]] = defaultdict(list)
    fingerprint_by_name: dict[str, str] = {}
    for record in records:
        name = record.get("name")
        profile = record.get("profile")
        fingerprint = record.get("copy_fingerprint")
        sequence = fingerprint.get("sequence_sha256") if isinstance(fingerprint, Mapping) else None
        if (
            not isinstance(name, str)
            or record.get("status") != "pass"
            or not isinstance(profile, Mapping)
            or not isinstance(sequence, str)
            or not sequence
        ):
            raise PreflightError("farthest-first record is invalid")
        profiles[name] = profile
        fingerprint_by_name[name] = sequence
        conflicts_by_fingerprint[sequence].append(name)
    if count < 1 or count > len(profiles):
        raise PreflightError("farthest-first count is invalid")

    medoid = str(select_default_reference(profiles)["selected"]["name"])
    selected = [medoid]
    blocked = set(conflicts_by_fingerprint[fingerprint_by_name[medoid]]) - {medoid}
    diagnostics = {medoid: {group: 0.0 for group in FEATURE_GROUPS}}
    coverage: dict[str, dict[str, float]] = {}
    for name in profiles:
        if name != medoid:
            coverage[name] = {
                group: round(raw_pairs[_pair_key(name, medoid)][group], 8)
                for group in FEATURE_GROUPS
            }
    while len(selected) < count:
        candidates = [name for name in profiles if name not in selected and name not in blocked]
        if not candidates:
            raise PreflightError("farthest-first exhausted exact-copy-safe candidates")

        def selection_key(name: str) -> tuple[float, float, float, str, str]:
            values = tuple(coverage[name][group] for group in FEATURE_GROUPS)
            return (
                -min(values),
                -statistics.median(values),
                -max(values),
                name.casefold(),
                name,
            )

        chosen = min(candidates, key=selection_key)
        selected.append(chosen)
        diagnostics[chosen] = dict(coverage[chosen])
        blocked.update(set(conflicts_by_fingerprint[fingerprint_by_name[chosen]]) - {chosen})
        for name in candidates:
            if name == chosen:
                continue
            distances = raw_pairs[_pair_key(name, chosen)]
            for group in FEATURE_GROUPS:
                coverage[name][group] = min(coverage[name][group], round(distances[group], 8))
    return [
        {
            "name": name,
            "selection_index": index,
            "selection_role": "real_medoid" if index == 0 else "farthest_first",
            "minimum_group_distances_to_prior": diagnostics[name],
            "exact_copy_conflicts": sorted(
                set(conflicts_by_fingerprint[fingerprint_by_name[name]]) - {name},
                key=_name_key,
            ),
        }
        for index, name in enumerate(selected)
    ]


def _average_ranks(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=lambda name: (values[name], *_name_key(name)))
    ranks: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        average = ((start + 1) + end) / 2.0
        for name in ordered[start:end]:
            ranks[name] = average
        start = end
    return ranks


def average_rank_spearman(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    """同値に平均順位を与えたSpearman相関を返す。"""
    if set(first) != set(second) or len(first) < 2:
        raise PreflightError("rank vectors must have the same names")
    first_ranks = _average_ranks(first)
    second_ranks = _average_ranks(second)
    names = sorted(first, key=_name_key)
    left = [first_ranks[name] for name in names]
    right = [second_ranks[name] for name in names]
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    if denominator == 0:
        raise PreflightError("rank correlation is undefined for a constant vector")
    return numerator / denominator


def set_comparison(first: set[str], second: set[str]) -> dict[str, float | int]:
    """第一集合を基準とする保持率とJaccard係数を返す。"""
    intersection = first & second
    union = first | second
    return {
        "intersection_count": len(intersection),
        "union_count": len(union),
        "retention": len(intersection) / len(first) if first else 1.0,
        "jaccard": len(intersection) / len(union) if union else 1.0,
    }


def compare_k_configurations(
    radii_by_k: Mapping[int, Mapping[str, float]],
    *,
    fractions: Sequence[float] = CANDIDATE_FRACTIONS,
) -> dict[str, Any]:
    """近傍数ごとの順位と候補集合を直接比較する。"""
    result: dict[str, Any] = {}
    for first_k, second_k in combinations(sorted(radii_by_k), 2):
        first = radii_by_k[first_k]
        second = radii_by_k[second_k]
        if set(first) != set(second):
            raise PreflightError("k configurations must use the same population")
        result[f"k={first_k}_vs_k={second_k}"] = {
            "spearman": average_rank_spearman(first, second),
            "candidate_sets": {
                f"{fraction:.2f}": set_comparison(
                    set(select_candidates(first, fraction=fraction)["names"]),
                    set(select_candidates(second, fraction=fraction)["names"]),
                )
                for fraction in fractions
            },
        }
    return result


def evaluate_target_eligibility(
    names: Iterable[str], builder: Callable[[str], object]
) -> list[dict[str, Any]]:
    """全曲を試し、失敗理由を成功件数へ隠さず保存する。"""
    results: list[dict[str, Any]] = []
    for name in sorted(names, key=_name_key):
        try:
            builder(name)
        except Exception as error:
            results.append(
                {
                    "name": name,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "reason": str(error),
                }
            )
        else:
            results.append({"name": name, "status": "success"})
    return results


def control_distance(first: Mapping[str, object], second: Mapping[str, object]) -> float:
    """3制御座標の最大差を全範囲2で割る。"""
    if set(first) < set(CONTROL_NAMES) or set(second) < set(CONTROL_NAMES):
        raise PreflightError("control values are incomplete")
    return max(
        abs(
            _finite_number(first[name], f"control.{name}")
            - _finite_number(second[name], f"control.{name}")
        )
        / 2.0
        for name in CONTROL_NAMES
    )


def _nearest_rank(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise PreflightError("cannot summarize empty values")
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def _summary(values_by_name: Mapping[str, float]) -> dict[str, Any]:
    if not values_by_name:
        raise PreflightError("cannot summarize empty values")
    values = list(values_by_name.values())
    maximum = max(values)
    maximum_name = min(
        (name for name, value in values_by_name.items() if value == maximum), key=_name_key
    )
    return {
        "minimum": min(values),
        "p05": _nearest_rank(values, 0.05),
        "p25": _nearest_rank(values, 0.25),
        "median": _nearest_rank(values, 0.50),
        "p75": _nearest_rank(values, 0.75),
        "p95": _nearest_rank(values, 0.95),
        "maximum": {"value": maximum, "name": maximum_name},
    }


def _axis_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise PreflightError("cannot summarize empty axis")
    return {
        "minimum": min(values),
        "p05": _nearest_rank(values, 0.05),
        "p25": _nearest_rank(values, 0.25),
        "median": _nearest_rank(values, 0.50),
        "p75": _nearest_rank(values, 0.75),
        "p95": _nearest_rank(values, 0.95),
        "maximum": max(values),
    }


def summarize_coverage(
    *,
    population_names: Sequence[str],
    candidate_names: Sequence[str],
    controls: Mapping[str, Mapping[str, object]],
    empirical_pairs: Mapping[tuple[str, str], float],
    raw_pairs: Mapping[tuple[str, str], Mapping[str, float]],
) -> dict[str, Any]:
    """制御空間と特徴空間を混ぜずに候補集合の被覆を測る。"""
    population = sorted(set(population_names), key=_name_key)
    candidates = sorted(set(candidate_names), key=_name_key)
    if not population or not candidates or not set(candidates) <= set(population):
        raise PreflightError("candidate coverage population is invalid")
    for name in population:
        if name not in controls:
            raise PreflightError(f"control record is missing: {name}")

    control_nearest: dict[str, float] = {}
    feature_nearest: dict[str, float] = {}
    group_nearest: dict[str, dict[str, float]] = {group: {} for group in FEATURE_GROUPS}
    group_distributions = {
        group: [distance[group] for distance in raw_pairs.values()] for group in FEATURE_GROUPS
    }
    group_quantile_lookups = {
        group: _empirical_lookup(values) for group, values in group_distributions.items()
    }
    for name in population:
        control_nearest[name] = min(
            control_distance(controls[name], controls[item]) for item in candidates
        )
        if name in candidates:
            feature_nearest[name] = 0.0
            for group in FEATURE_GROUPS:
                group_nearest[group][name] = 0.0
            continue
        feature_nearest[name] = min(empirical_pairs[_pair_key(name, item)] for item in candidates)
        for group in FEATURE_GROUPS:
            group_nearest[group][name] = min(
                group_quantile_lookups[group][raw_pairs[_pair_key(name, item)][group]]
                for item in candidates
            )

    reference_controls = {
        control: _axis_summary(
            [_finite_number(controls[name][control], f"control.{control}") for name in candidates]
        )
        for control in CONTROL_NAMES
    }
    brightness_values = [
        _finite_number(controls[name]["あかるさ"], "control.あかるさ") for name in candidates
    ]
    return {
        "population_count": len(population),
        "candidate_count": len(candidates),
        "reference_controls": reference_controls,
        "user_brightness_request_distance": {
            str(request): min(abs(request - value) for value in brightness_values)
            for request in (-1, 0, 1)
        },
        "control_nearest_distance": _summary(control_nearest),
        "feature_nearest_composite": _summary(feature_nearest),
        "feature_nearest_by_group": {
            group: _summary(values) for group, values in group_nearest.items()
        },
    }


def _json_safe(value: object) -> object:
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError(f"unable to read JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise PreflightError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise PreflightError(f"unable to read JSONL: {path}: {error}") from error
    for line_number, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise PreflightError(f"invalid JSONL line {line_number}: {path}") from error
        if not isinstance(value, dict):
            raise PreflightError(f"JSONL record must be an object: {path}:{line_number}")
        records.append(value)
    return records


def _verify_manifest(directory: Path, expected_status: str, files: Sequence[str]) -> dict[str, Any]:
    manifest = _read_json(directory / "manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("status") != expected_status:
        raise PreflightError(f"input manifest is not complete: {directory}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise PreflightError(f"input manifest outputs are invalid: {directory}")
    for name in files:
        expected = outputs.get(name)
        if not isinstance(expected, str) or sha256_file(directory / name) != expected:
            raise PreflightError(f"input hash mismatch: {directory / name}")
    return manifest


def _collapse_copy_votes(
    records: Sequence[Mapping[str, Any]], eligible_names: set[str]
) -> tuple[list[str], list[dict[str, Any]]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        name = record.get("name")
        if name not in eligible_names:
            continue
        fingerprint = record.get("copy_fingerprint")
        sequence = fingerprint.get("sequence_sha256") if isinstance(fingerprint, Mapping) else None
        if not isinstance(name, str) or not isinstance(sequence, str) or not sequence:
            raise PreflightError(f"copy fingerprint is invalid: {name}")
        groups[sequence].append(name)
    representatives: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    for fingerprint, names in sorted(groups.items()):
        ordered = sorted(names, key=_name_key)
        representatives.append(ordered[0])
        if len(ordered) > 1:
            diagnostics.append(
                {"sequence_sha256": fingerprint, "representative": ordered[0], "names": ordered}
            )
    return sorted(representatives, key=_name_key), diagnostics


def _condition_populations(
    names: Sequence[str],
    records: Mapping[str, Mapping[str, Any]],
    series: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    duration_values = [float(records[name]["profile"]["duration_ms"]) for name in names]
    note_values = [float(records[name]["profile"]["note_count"]) for name in names]
    duration_p05 = _nearest_rank(duration_values, 0.05)
    note_p05 = _nearest_rank(note_values, 0.05)
    result = {
        "eligible_copy_deduplicated": list(names),
        "exclude_duration_below_p05": [
            name for name in names if float(records[name]["profile"]["duration_ms"]) >= duration_p05
        ],
        "exclude_note_count_below_p05": [
            name for name in names if float(records[name]["profile"]["note_count"]) >= note_p05
        ],
        "exclude_note_count_below_100": [
            name for name in names if int(records[name]["profile"]["note_count"]) >= 100
        ],
    }
    for entry in series:
        label = entry.get("series")
        files = entry.get("files")
        if (
            isinstance(label, str)
            and isinstance(files, list)
            and all(isinstance(item, str) for item in files)
        ):
            excluded = set(files)
            result[f"exclude_series:{label}"] = [name for name in names if name not in excluded]
    return result


def _component_sizes(names: Sequence[str], edges: set[tuple[str, str]]) -> list[int]:
    adjacency: dict[str, set[str]] = {name: set() for name in names}
    for first, second in edges:
        if first in adjacency and second in adjacency:
            adjacency[first].add(second)
            adjacency[second].add(first)
    unseen = set(names)
    sizes: list[int] = []
    while unseen:
        root = min(unseen, key=_name_key)
        stack = [root]
        unseen.remove(root)
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for neighbor in adjacency[current] & unseen:
                unseen.remove(neighbor)
                stack.append(neighbor)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def _graph_reproduction(
    records: Sequence[Mapping[str, Any]], eligible_names: set[str]
) -> dict[str, Any]:
    outgoing: dict[str, set[str]] = {}
    for record in records:
        name = record["name"]
        entries = record.get("neighborhood", {}).get("neighbors", [])
        outgoing[name] = {
            entry["name"]
            for entry in entries
            if isinstance(entry, Mapping)
            and isinstance(entry.get("name"), str)
            and entry["name"] != name
        }
    directed_as_undirected = {
        _pair_key(name, neighbor)
        for name, neighbors in outgoing.items()
        for neighbor in neighbors
        if neighbor in outgoing
    }
    mutual = {
        _pair_key(name, neighbor)
        for name, neighbors in outgoing.items()
        for neighbor in neighbors
        if neighbor in outgoing and name in outgoing[neighbor]
    }

    def result(names: Sequence[str]) -> dict[str, Any]:
        selected = set(names)
        directed_sizes = _component_sizes(
            names,
            {edge for edge in directed_as_undirected if set(edge) <= selected},
        )
        mutual_sizes = _component_sizes(names, {edge for edge in mutual if set(edge) <= selected})
        return {
            "vertex_count": len(names),
            "directed_neighbor_as_undirected": {
                "component_count": len(directed_sizes),
                "component_sizes": directed_sizes,
            },
            "mutual_neighbor_only": {
                "component_count": len(mutual_sizes),
                "component_sizes": mutual_sizes,
            },
        }

    all_names = sorted(outgoing, key=_name_key)
    eligible = sorted(eligible_names, key=_name_key)
    return {
        "edge_source": "saved 232-piece seven-member neighborhoods; no edge refill",
        "all_profiles": result(all_names),
        "eligible_induced_subgraph": result(eligible),
    }


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    content = "".join(
        json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for record in records
    )
    atomic_write_bytes(path, content.encode("utf-8"))


def run_preflight(project_root: Path, output_dir: Path) -> dict[str, Any]:
    """入力を検証し、候補採用を伴わない記述分析を一式保存する。"""
    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    reference_dir = project_root / ".appendix" / "reference-profile-v1"
    control_dir = project_root / ".appendix" / "control-reference-baseline-v3"
    groups_path = project_root / ".appendix" / "corpus-audit" / "evaluation-groups.json"

    reference_manifest = _verify_manifest(reference_dir, "pass", ("files.jsonl", "summary.json"))
    control_manifest = _verify_manifest(control_dir, "complete", ("records.jsonl", "summary.json"))
    if control_manifest.get("reference_manifest_sha256") != sha256_file(
        reference_dir / "manifest.json"
    ):
        raise PreflightError("control baseline reference manifest hash mismatch")
    if sha256_file(groups_path) != EVALUATION_GROUPS_SHA256:
        raise PreflightError("evaluation groups hash mismatch")
    reference_records = _read_jsonl(reference_dir / "files.jsonl")
    control_records = _read_jsonl(control_dir / "records.jsonl")
    profile_by_name = {record["name"]: record for record in reference_records}
    control_by_name = {record["name"]: record for record in control_records}
    if set(profile_by_name) != set(control_by_name):
        raise PreflightError("reference and control record sets differ")
    if {"rut.mid", "aimusic01.mid"} & set(profile_by_name):
        raise PreflightError("known excluded noise files returned to the corpus")

    source_hashes = reference_manifest.get("inputs")
    if not isinstance(source_hashes, Mapping):
        raise PreflightError("reference source hashes are missing")

    def build(name: str) -> object:
        return build_reference_generation_target(
            {
                "reference": {
                    "state": "resolved",
                    "name": name,
                    "sha256": source_hashes[name],
                    "selection_method": "preflight_explicit",
                },
                "controls": {},
            },
            reference_dir=reference_dir,
            control_dir=control_dir,
        )

    eligibility = evaluate_target_eligibility(profile_by_name, build)
    eligible_names = {item["name"] for item in eligibility if item["status"] == "success"}
    representatives, duplicate_groups = _collapse_copy_votes(reference_records, eligible_names)
    representative_profiles = {name: profile_by_name[name]["profile"] for name in representatives}
    controls = {name: control_by_name[name]["normalized"] for name in representatives}
    evaluation_groups = _read_json(groups_path)
    series = evaluation_groups.get("filename_series")
    if not isinstance(series, list):
        raise PreflightError("evaluation filename series are invalid")
    populations = _condition_populations(representatives, profile_by_name, series)

    population_models: dict[str, dict[str, Any]] = {}
    population_radii: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    local_rows: list[dict[str, Any]] = []
    candidate_sets: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for condition, names in populations.items():
        subset_profiles = {name: representative_profiles[name] for name in names}
        model = build_pair_distance_model(subset_profiles)
        population_models[condition] = model
        population_radii[condition] = {"empirical_quantile": {}, "median_normalized": {}}
        candidate_sets[condition] = {"empirical_quantile": {}, "median_normalized": {}}
        for method in ("empirical_quantile", "median_normalized"):
            for k in K_VALUES:
                radii = local_radii(model[method], names, k=k)
                population_radii[condition][method][k] = radii
                ranks = _average_ranks(radii)
                for name in sorted(names, key=_name_key):
                    local_rows.append(
                        {
                            "population": condition,
                            "method": method,
                            "k": k,
                            "name": name,
                            "radius": radii[name],
                            "average_rank": ranks[name],
                        }
                    )
                for fraction in CANDIDATE_FRACTIONS:
                    key = f"{fraction:.2f}"
                    candidate_sets[condition][method][f"k={k}:{key}"] = select_candidates(
                        radii, fraction=fraction
                    )

    base = "eligible_copy_deduplicated"
    base_names = populations[base]
    sensitivity: dict[str, Any] = {
        "population_counts": {name: len(values) for name, values in populations.items()},
        "k_comparisons": {
            method: compare_k_configurations(population_radii[base][method])
            for method in ("empirical_quantile", "median_normalized")
        },
        "rank_correlations": {},
        "candidate_comparisons": {},
        "candidate_sets": candidate_sets,
    }
    for condition, names in populations.items():
        common = set(base_names) & set(names)
        sensitivity["rank_correlations"][condition] = {}
        sensitivity["candidate_comparisons"][condition] = {}
        for method in ("empirical_quantile", "median_normalized"):
            for k in K_VALUES:
                base_values = {name: population_radii[base][method][k][name] for name in common}
                current_values = {
                    name: population_radii[condition][method][k][name] for name in common
                }
                label = f"{method}:k={k}"
                sensitivity["rank_correlations"][condition][label] = average_rank_spearman(
                    base_values, current_values
                )
                for fraction in CANDIDATE_FRACTIONS:
                    key = f"k={k}:{fraction:.2f}"
                    first = set(candidate_sets[base][method][key]["names"]) & common
                    second = set(candidate_sets[condition][method][key]["names"]) & common
                    sensitivity["candidate_comparisons"][condition][f"{method}:{key}"] = (
                        set_comparison(first, second)
                    )

    base_model = population_models[base]
    coverage: dict[str, Any] = {"local_radius": {}, "farthest_first": {}}
    base_records = [profile_by_name[name] for name in base_names]
    local_by_fraction = {
        fraction: candidate_sets[base]["empirical_quantile"][f"k=7:{fraction:.2f}"]
        for fraction in CANDIDATE_FRACTIONS
    }
    maximum_farthest_count = max(len(item["names"]) for item in local_by_fraction.values())
    farthest_order = select_farthest_first_from_pairs(
        base_records, base_model["raw"], count=maximum_farthest_count
    )
    for fraction in CANDIDATE_FRACTIONS:
        local = local_by_fraction[fraction]
        local_names = local["names"]
        coverage["local_radius"][f"{fraction:.2f}"] = summarize_coverage(
            population_names=base_names,
            candidate_names=local_names,
            controls=controls,
            empirical_pairs=base_model["empirical_quantile"],
            raw_pairs=base_model["raw"],
        )
        farthest = farthest_order[: len(local_names)]
        farthest_names = [item["name"] for item in farthest]
        coverage["farthest_first"][f"{fraction:.2f}"] = {
            "names": farthest_names,
            "coverage": summarize_coverage(
                population_names=base_names,
                candidate_names=farthest_names,
                controls=controls,
                empirical_pairs=base_model["empirical_quantile"],
                raw_pairs=base_model["raw"],
            ),
            "local_radius_summary": _summary(
                {
                    name: population_radii[base]["empirical_quantile"][7][name]
                    for name in farthest_names
                }
            ),
        }

    method_comparisons: dict[str, Any] = {}
    for k in K_VALUES:
        empirical_radii = population_radii[base]["empirical_quantile"][k]
        median_radii = population_radii[base]["median_normalized"][k]
        method_comparisons[f"k={k}"] = {
            "spearman": average_rank_spearman(empirical_radii, median_radii),
            "candidate_sets": {
                f"{fraction:.2f}": set_comparison(
                    set(
                        candidate_sets[base]["empirical_quantile"][f"k={k}:{fraction:.2f}"]["names"]
                    ),
                    set(
                        candidate_sets[base]["median_normalized"][f"k={k}:{fraction:.2f}"]["names"]
                    ),
                )
                for fraction in CANDIDATE_FRACTIONS
            },
        }

    graph = _graph_reproduction(reference_records, eligible_names)
    run_spec = {
        "analysis_version": ANALYSIS_VERSION,
        "inputs": {
            "reference_manifest_sha256": sha256_file(reference_dir / "manifest.json"),
            "control_manifest_sha256": sha256_file(control_dir / "manifest.json"),
            "evaluation_groups_sha256": sha256_file(groups_path),
        },
        "distance": {
            "methods": ["empirical_quantile_max", "median_normalized_max"],
            "feature_groups": list(FEATURE_GROUPS),
            "k_values": list(K_VALUES),
            "candidate_fractions": list(CANDIDATE_FRACTIONS),
            "self_pairs_included": False,
            "round_before_comparison": False,
        },
        "sensitivity_conditions": list(populations),
        "automatic_adoption": False,
    }
    result = {
        "status": "complete",
        "positive_evidence": {
            "profile_count": len(profile_by_name),
            "target_success_count": len(eligible_names),
            "target_failure_count": len(profile_by_name) - len(eligible_names),
            "copy_deduplicated_voting_count": len(representatives),
            "analysis_completed": True,
        },
        "evidence_not_found": [
            "local support is not evidence of aesthetic quality",
            "this analysis does not establish generation success or audible style similarity",
        ],
        "unable_to_investigate": [],
        "duplicate_vote_groups": duplicate_groups,
        "method_comparisons": method_comparisons,
        "adoption_decision": None,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "run-spec.json", _json_safe(run_spec))
    _write_jsonl(output_dir / "target-eligibility.jsonl", eligibility)
    _write_jsonl(output_dir / "local-radius.jsonl", local_rows)
    atomic_write_json(output_dir / "candidate-pool-sensitivity.json", _json_safe(sensitivity))
    atomic_write_json(output_dir / "coverage.json", _json_safe(coverage))
    atomic_write_json(output_dir / "graph-reproduction.json", _json_safe(graph))
    atomic_write_json(output_dir / "result.json", _json_safe(result))
    output_names = (
        "run-spec.json",
        "target-eligibility.jsonl",
        "local-radius.jsonl",
        "candidate-pool-sensitivity.json",
        "coverage.json",
        "graph-reproduction.json",
        "result.json",
    )
    manifest = {
        "schema_version": 1,
        "analysis_version": ANALYSIS_VERSION,
        "status": "complete",
        "outputs": {name: sha256_file(output_dir / name) for name in output_names},
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return _json_safe(result)  # type: ignore[return-value]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".appendix/corpus-baseline-preflight-v1")
    )
    arguments = parser.parse_args(argv)
    result = run_preflight(arguments.project_root, arguments.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
