"""発音群の音高順位と隣接関係を、note列を複製せずに診断する。"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from fractions import Fraction
from itertools import pairwise
from typing import Any

from llm_musical_composer.reference_decomposition import ObservedNote
from llm_musical_composer.reference_timing_hypothesis import KnownScoreNoteAlignment
from llm_musical_composer.score_timing_hypothesis import (
    GroupingAttack,
    ScoreTimingHypothesisV0,
)


def _stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def measure_event_id_references(
    value: object,
    *,
    event_ids: set[str],
) -> dict[str, int]:
    """構造化値に実際に再掲された既知event IDを数える。"""
    found: list[str] = []

    def visit(item: object) -> None:
        if isinstance(item, str):
            if item in event_ids:
                found.append(item)
            return
        if isinstance(item, Mapping):
            for key, nested in item.items():
                visit(key)
                visit(nested)
            return
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for nested in item:
                visit(nested)

    visit(value)
    return {"unique_count": len(set(found)), "occurrence_count": len(found)}


def _semantic_grouping_payload(groups: Sequence[GroupingAttack]) -> list[dict[str, object]]:
    return [
        {
            "source_event_ids": list(group.source_event_ids),
            "relative_onsets_us": list(group.relative_onsets_us),
        }
        for group in groups
    ]


def _pitch_rank_blocks(
    groups: Sequence[GroupingAttack],
    notes_by_id: Mapping[str, ObservedNote],
) -> list[list[tuple[int, list[int]]]]:
    result: list[list[tuple[int, list[int]]]] = []
    for group in groups:
        by_pitch: dict[int, list[int]] = defaultdict(list)
        for member_index, event_id in enumerate(group.source_event_ids):
            try:
                pitch = notes_by_id[event_id].pitch
            except KeyError as error:
                raise ValueError(f"source event missing from note ledger: {event_id}") from error
            by_pitch[pitch].append(member_index)
        result.append([(pitch, members) for pitch, members in sorted(by_pitch.items())])
    return result


def _relation_blocks(
    ranked: Sequence[Sequence[tuple[int, list[int]]]],
) -> list[dict[str, object]]:
    relations: list[dict[str, object]] = []
    for source_group, (source_blocks, target_blocks) in enumerate(pairwise(ranked)):
        flags: dict[tuple[int, int], set[str]] = defaultdict(set)
        for source_rank, (source_pitch, _) in enumerate(source_blocks):
            distance = min(abs(target_pitch - source_pitch) for target_pitch, _ in target_blocks)
            for target_rank, (target_pitch, _) in enumerate(target_blocks):
                if abs(target_pitch - source_pitch) == distance:
                    flags[(source_rank, target_rank)].add("forward")
        for target_rank, (target_pitch, _) in enumerate(target_blocks):
            distance = min(abs(source_pitch - target_pitch) for source_pitch, _ in source_blocks)
            for source_rank, (source_pitch, _) in enumerate(source_blocks):
                if abs(source_pitch - target_pitch) == distance:
                    flags[(source_rank, target_rank)].add("backward")
        for (source_rank, target_rank), directions in sorted(flags.items()):
            source_pitch, _ = source_blocks[source_rank]
            target_pitch, _ = target_blocks[target_rank]
            relation = "mutual" if len(directions) == 2 else next(iter(directions))
            relations.append(
                {
                    "source_group_index": source_group,
                    "target_group_index": source_group + 1,
                    "source_rank_index": source_rank,
                    "target_rank_index": target_rank,
                    "semitone_delta": target_pitch - source_pitch,
                    "relation": relation,
                }
            )
    return relations


def _ambiguous_component_size(
    ranked: Sequence[Sequence[tuple[int, list[int]]]],
    relations: Sequence[Mapping[str, object]],
) -> int:
    nodes = {
        (group_index, rank_index)
        for group_index, blocks in enumerate(ranked)
        for rank_index in range(len(blocks))
    }
    parent = {node: node for node in nodes}
    incoming_degree = {node: 0 for node in nodes}
    outgoing_degree = {node: 0 for node in nodes}

    def find(node: tuple[int, int]) -> tuple[int, int]:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: tuple[int, int], right: tuple[int, int]) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for relation in relations:
        if relation["relation"] != "mutual":
            continue
        source = (int(relation["source_group_index"]), int(relation["source_rank_index"]))
        target = (int(relation["target_group_index"]), int(relation["target_rank_index"]))
        union(source, target)
        outgoing_degree[source] += 1
        incoming_degree[target] += 1
    members_by_root: dict[tuple[int, int], int] = defaultdict(int)
    ambiguous_roots: set[tuple[int, int]] = set()
    for group_index, blocks in enumerate(ranked):
        for rank_index, (_, members) in enumerate(blocks):
            node = (group_index, rank_index)
            root = find(node)
            members_by_root[root] += len(members)
            if len(members) > 1 or incoming_degree[node] > 1 or outgoing_degree[node] > 1:
                ambiguous_roots.add(root)
    return max((members_by_root[root] for root in ambiguous_roots), default=0)


def _log10_add(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + math.log10(sum(10 ** (value - maximum) for value in values))


def _path_count_log10_upper_bound(
    ranked: Sequence[Sequence[tuple[int, list[int]]]],
    relations: Sequence[Mapping[str, object]],
) -> float:
    if not ranked:
        return 0.0
    incoming: dict[tuple[int, int], list[int]] = defaultdict(list)
    for relation in relations:
        if relation["relation"] == "mutual":
            incoming[
                (int(relation["target_group_index"]), int(relation["target_rank_index"]))
            ].append(int(relation["source_rank_index"]))
    previous = {
        rank_index: math.log10(len(members)) for rank_index, (_, members) in enumerate(ranked[0])
    }
    maximum = _log10_add(tuple(previous.values())) if previous else 0.0
    for group_index, blocks in enumerate(ranked[1:], start=1):
        current: dict[int, float] = {}
        for rank_index, (_, members) in enumerate(blocks):
            sources = [
                previous[source_rank]
                for source_rank in incoming.get((group_index, rank_index), ())
                if source_rank in previous
            ]
            prefix = _log10_add(sources) if sources else 0.0
            current[rank_index] = prefix + math.log10(len(members))
        if current:
            maximum = max(maximum, _log10_add(tuple(current.values())))
        previous = current
    return round(maximum, 8)


def build_texture_edge_diagnostics(
    *,
    grouping_profiles: Mapping[str, tuple[GroupingAttack, ...]],
    notes_by_id: Mapping[str, ObservedNote],
    ledger_sha256: str,
) -> dict[str, Any]:
    """同じ発音群内容を共有し、indexだけを持つ圧縮診断を作る。"""
    aliases_by_hash: dict[str, list[str]] = defaultdict(list)
    groups_by_hash: dict[str, tuple[GroupingAttack, ...]] = {}
    for profile_id, groups in sorted(grouping_profiles.items()):
        content_hash = _stable_hash(_semantic_grouping_payload(groups))
        aliases_by_hash[content_hash].append(profile_id)
        groups_by_hash.setdefault(content_hash, groups)
    profiles: list[dict[str, Any]] = []
    alias_map: dict[str, str] = {}
    for content_hash in sorted(aliases_by_hash):
        semantic_profile_id = f"semantic-{content_hash}"
        aliases = sorted(aliases_by_hash[content_hash])
        alias_map.update({alias: semantic_profile_id for alias in aliases})
        groups = groups_by_hash[content_hash]
        ranked = _pitch_rank_blocks(groups, notes_by_id)
        relations = _relation_blocks(ranked)
        member_count = sum(len(group.source_event_ids) for group in groups)
        block_count = sum(len(blocks) for blocks in ranked) + len(relations)
        profile: dict[str, Any] = {
            "semantic_profile_id": semantic_profile_id,
            "aliases": aliases,
            "ledger_sha256": ledger_sha256,
            "group_count": len(groups),
            "member_count": member_count,
            "vertical_groups": [
                {
                    "group_index": group_index,
                    "rank_blocks": [members for _, members in blocks],
                }
                for group_index, blocks in enumerate(ranked)
            ],
            "horizontal_relations": relations,
            "block_count": block_count,
            "block_member_ratio": round(block_count / member_count, 8) if member_count else 0.0,
            "max_ambiguous_component_member_count": _ambiguous_component_size(ranked, relations),
            "path_count_log10_upper_bound": _path_count_log10_upper_bound(ranked, relations),
        }
        profile["max_virtual_edges_per_relation_block"] = max(
            (
                len(
                    ranked[int(relation["source_group_index"])][int(relation["source_rank_index"])][
                        1
                    ]
                )
                * len(
                    ranked[int(relation["target_group_index"])][int(relation["target_rank_index"])][
                        1
                    ]
                )
                for relation in relations
            ),
            default=0,
        )
        event_measurement = measure_event_id_references(
            profile,
            event_ids={event_id for group in groups for event_id in group.source_event_ids},
        )
        profile["event_id_unique_count"] = event_measurement["unique_count"]
        profile["event_id_occurrence_count"] = event_measurement["occurrence_count"]
        profile["normalized_json_bytes_without_measurement"] = len(
            json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        profiles.append(profile)
    return {
        "schema_version": 0,
        "status": "assessed",
        "surface_profile_count": len(grouping_profiles),
        "semantic_profile_count": len(profiles),
        "alias_map": dict(sorted(alias_map.items())),
        "profiles": profiles,
    }


def _window_has_mutual_path(
    profile: Mapping[str, Any],
    *,
    start: int,
    size: int,
) -> bool:
    relations_by_pair: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for relation in profile["horizontal_relations"]:
        if relation["relation"] == "mutual":
            relations_by_pair[int(relation["source_group_index"])].append(relation)
    states = {int(relation["source_rank_index"]) for relation in relations_by_pair.get(start, ())}
    for source_group in range(start, start + size - 1):
        states = {
            int(relation["target_rank_index"])
            for relation in relations_by_pair.get(source_group, ())
            if int(relation["source_rank_index"]) in states
        }
        if not states:
            return False
    return bool(states)


def _normalized_score_intervals(positions: Sequence[int]) -> tuple[int, ...]:
    intervals = tuple(right - left for left, right in pairwise(positions))
    if not intervals or any(interval <= 0 for interval in intervals):
        raise ValueError("score positions must be strictly increasing")
    divisor = math.gcd(*intervals)
    return tuple(interval // divisor for interval in intervals)


def build_texture_candidate_joins(
    candidates: Sequence[ScoreTimingHypothesisV0],
    *,
    texture_diagnostic: Mapping[str, Any],
    window_sizes: Sequence[int] = tuple(range(3, 7)),
) -> dict[str, Any]:
    """共有base診断へ、候補別の楽譜間隔fingerprintだけを結合する。"""
    profiles = {
        profile["semantic_profile_id"]: profile for profile in texture_diagnostic["profiles"]
    }
    alias_map = texture_diagnostic["alias_map"]
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        semantic_profile_id = alias_map.get(candidate.grouping_profile_id)
        base = {
            "candidate_id": candidate.candidate_id,
            "semantic_profile_id": semantic_profile_id,
        }
        if semantic_profile_id is None or semantic_profile_id not in profiles:
            records.append(
                {**base, "status": "unable_to_investigate", "reason": "semantic_profile_missing"}
            )
            continue
        profile = profiles[semantic_profile_id]
        positions = tuple(ref.score_position for ref in candidate.attack_group_refs)
        if len(positions) != int(profile["group_count"]):
            records.append(
                {**base, "status": "unable_to_investigate", "reason": "group_count_mismatch"}
            )
            continue
        patterns: list[tuple[int, int, tuple[int, ...]]] = []
        try:
            for raw_size in window_sizes:
                size = int(raw_size)
                if size < 3:
                    raise ValueError("window sizes must be at least three")
                for start in range(max(0, len(positions) - size + 1)):
                    if _window_has_mutual_path(profile, start=start, size=size):
                        pattern = _normalized_score_intervals(positions[start : start + size])
                        patterns.append((start, size, pattern))
        except ValueError as error:
            records.append({**base, "status": "unable_to_investigate", "reason": str(error)})
            continue
        if not patterns:
            records.append(
                {**base, "status": "unable_to_investigate", "reason": "no_mutual_path_window"}
            )
            continue
        records.append(
            {
                **base,
                "status": "assessed",
                "fingerprint": _stable_hash(patterns),
                "supported_window_count": len(patterns),
                "distinct_interval_pattern_count": len({pattern for _, _, pattern in patterns}),
            }
        )
    by_profile: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record["status"] == "assessed":
            by_profile[str(record["semantic_profile_id"])].add(str(record["fingerprint"]))
    if not by_profile:
        sensitivity = "unable_to_investigate"
    elif any(len(fingerprints) > 1 for fingerprints in by_profile.values()):
        sensitivity = "timing_sensitive"
    else:
        sensitivity = "candidate_invariant"
    return {
        "schema_version": 0,
        "base_graph_count": int(texture_diagnostic["semantic_profile_count"]),
        "candidate_count": len(candidates),
        "sensitivity": sensitivity,
        "records": records,
    }


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 8) if denominator else None


def matched_pair_weighted_null_rate(
    boundary_counts: Sequence[tuple[int, int, int]],
) -> float | None:
    """区間別all-pair率を、同じ区間の選択pair数で重み付けする。"""
    selected_total = sum(selected_count for _, _, selected_count in boundary_counts)
    if selected_total == 0:
        return None
    expected_same = Fraction(0, 1)
    for same_count, all_count, selected_count in boundary_counts:
        if all_count <= 0 or not 0 <= same_count <= all_count or selected_count < 0:
            raise ValueError("invalid boundary pair counts")
        expected_same += Fraction(same_count, all_count) * selected_count
    return round(float(expected_same / selected_total), 8)


def build_known_voice_diagnostics(
    *,
    alignment: KnownScoreNoteAlignment,
    grouping_profiles: Mapping[str, tuple[GroupingAttack, ...]],
    notes_by_id: Mapping[str, ObservedNote],
    texture_diagnostic: Mapping[str, Any],
    oracle_basis: str,
) -> dict[str, Any]:
    """既知voiceラベルと音高順位規則の関係を、正解声部とは呼ばずに測る。"""
    base = {
        "schema_version": 0,
        "oracle_basis": oracle_basis,
        "claim_scope": "generator_ir_compatibility_only",
    }
    if alignment.status != "assessed":
        return {
            **base,
            "status": "unable_to_investigate",
            "reason": "known_score_note_alignment_not_assessed",
            "profiles": [],
        }
    label_by_event_id = {
        match.observed.note_on_event_id: match.expected.voice for match in alignment.matches
    }
    profile_records: list[dict[str, Any]] = []
    for texture_profile in texture_diagnostic["profiles"]:
        aliases = list(texture_profile["aliases"])
        if not aliases or aliases[0] not in grouping_profiles:
            raise ValueError("texture diagnostic alias is absent from grouping profiles")
        groups = grouping_profiles[aliases[0]]
        highest_upper_match = 0
        highest_count = 0
        lowest_lower_match = 0
        lowest_count = 0
        crossing_groups = 0
        same_voice_multiple_groups = 0
        member_labels: list[list[str]] = []
        for group in groups:
            pitches = [notes_by_id[event_id].pitch for event_id in group.source_event_ids]
            labels = [label_by_event_id[event_id] for event_id in group.source_event_ids]
            member_labels.append(labels)
            highest = max(pitches)
            lowest = min(pitches)
            highest_indices = [index for index, pitch in enumerate(pitches) if pitch == highest]
            lowest_indices = [index for index, pitch in enumerate(pitches) if pitch == lowest]
            highest_count += len(highest_indices)
            highest_upper_match += sum(labels[index] == "upper" for index in highest_indices)
            lowest_count += len(lowest_indices)
            lowest_lower_match += sum(labels[index] == "lower" for index in lowest_indices)
            upper_pitches = [
                pitch for pitch, label in zip(pitches, labels, strict=True) if label == "upper"
            ]
            lower_pitches = [
                pitch for pitch, label in zip(pitches, labels, strict=True) if label == "lower"
            ]
            crossing_groups += bool(
                upper_pitches and lower_pitches and min(upper_pitches) < max(lower_pitches)
            )
            label_counts = {label: labels.count(label) for label in set(labels)}
            same_voice_multiple_groups += any(count > 1 for count in label_counts.values())

        mutual_relations = [
            relation
            for relation in texture_profile["horizontal_relations"]
            if relation["relation"] == "mutual"
        ]
        same_label_pairs = 0
        all_pairs = 0
        adjacent_same_label_pairs = 0
        adjacent_all_pairs = 0
        adjacent_boundary_counts: list[tuple[int, int]] = []
        for source_labels, target_labels in pairwise(member_labels):
            boundary_all_pairs = len(source_labels) * len(target_labels)
            boundary_same_label_pairs = sum(
                source_label == target_label
                for source_label in source_labels
                for target_label in target_labels
            )
            adjacent_all_pairs += boundary_all_pairs
            adjacent_same_label_pairs += boundary_same_label_pairs
            adjacent_boundary_counts.append((boundary_same_label_pairs, boundary_all_pairs))
        continuation_counts = {"zero": 0, "one": 0, "multiple": 0}
        relations_by_source: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
        mutual_pair_counts_by_boundary: dict[int, int] = defaultdict(int)
        for relation in mutual_relations:
            source_group = int(relation["source_group_index"])
            target_group = int(relation["target_group_index"])
            source_members = [
                int(value)
                for value in texture_profile["vertical_groups"][source_group]["rank_blocks"][
                    int(relation["source_rank_index"])
                ]
            ]
            target_members = [
                int(value)
                for value in texture_profile["vertical_groups"][target_group]["rank_blocks"][
                    int(relation["target_rank_index"])
                ]
            ]
            for source_member in source_members:
                relations_by_source[(source_group, source_member)].append(relation)
            for source_member in source_members:
                for target_member in target_members:
                    all_pairs += 1
                    mutual_pair_counts_by_boundary[source_group] += 1
                    same_label_pairs += (
                        member_labels[source_group][source_member]
                        == member_labels[target_group][target_member]
                    )
        for group_index, labels in enumerate(member_labels[:-1]):
            for member_index, label in enumerate(labels):
                targets = {
                    (int(relation["target_group_index"]), target_member)
                    for relation in relations_by_source.get((group_index, member_index), ())
                    for target_member in (
                        int(value)
                        for value in texture_profile["vertical_groups"][
                            int(relation["target_group_index"])
                        ]["rank_blocks"][int(relation["target_rank_index"])]
                    )
                    if member_labels[int(relation["target_group_index"])][target_member] == label
                }
                bucket = "zero" if not targets else "one" if len(targets) == 1 else "multiple"
                continuation_counts[bucket] += 1
        mutual_rate = _safe_rate(same_label_pairs, all_pairs)
        adjacent_rate = _safe_rate(adjacent_same_label_pairs, adjacent_all_pairs)
        matched_null_rate = matched_pair_weighted_null_rate(
            tuple(
                (same_count, pair_count, mutual_pair_counts_by_boundary[index])
                for index, (same_count, pair_count) in enumerate(adjacent_boundary_counts)
            )
        )
        profile_records.append(
            {
                "semantic_profile_id": texture_profile["semantic_profile_id"],
                "aliases": aliases,
                "highest_upper_match_rate": _safe_rate(highest_upper_match, highest_count),
                "lowest_lower_match_rate": _safe_rate(lowest_lower_match, lowest_count),
                "upper_below_lower_group_count": crossing_groups,
                "same_voice_multiple_note_group_count": same_voice_multiple_groups,
                "mutual_nearest_same_label_rate": mutual_rate,
                "mutual_nearest_virtual_pair_count": all_pairs,
                "adjacent_all_pair_same_label_rate": adjacent_rate,
                "adjacent_all_pair_count": adjacent_all_pairs,
                "matched_pair_weighted_null_same_label_rate": matched_null_rate,
                "same_label_rate_lift": (
                    round(mutual_rate - matched_null_rate, 8)
                    if mutual_rate is not None and matched_null_rate is not None
                    else None
                ),
                "nearest_null_comparison_status": (
                    "assessed"
                    if mutual_rate is not None and matched_null_rate is not None
                    else "unable_to_investigate"
                ),
                "same_label_continuation_counts": continuation_counts,
            }
        )
    return {**base, "status": "assessed", "profiles": profile_records}
