"""実在SMFの観測だけから素材境界と反復の有限候補を作る。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any

from llm_musical_composer.observed_structure_method import (
    METHOD_CONSTANTS,
    verify_method_manifest,
)
from llm_musical_composer.reference_decomposition import (
    build_observed_performance,
    load_observed_smf,
)
from llm_musical_composer.run_state import atomic_write_bytes, atomic_write_json, sha256_file
from llm_musical_composer.score_timing_hypothesis import build_grouping_profiles
from llm_musical_composer.score_timing_upper_evidence import (
    enumerate_pitch_rhythm_recurrences,
)

_OBSERVED_CONSTANTS = METHOD_CONSTANTS["observed_structure"]
BOUNDARY_WINDOW_SIZES = tuple(_OBSERVED_CONSTANTS["boundary_window_sizes"])
RECURRENCE_WINDOW_SIZES = tuple(_OBSERVED_CONSTANTS["recurrence_window_sizes"])
RECURRENCE_EXTENSION_PROBE_SIZE = int(
    _OBSERVED_CONSTANTS["recurrence_extension_probe_size"]
)
MAXIMUM_BOUNDARY_SELECTION_RATE = float(
    _OBSERVED_CONSTANTS["maximum_boundary_selection_rate"]
)
MINIMUM_ASSESSED_SOURCE_COUNT = int(
    _OBSERVED_CONSTANTS["minimum_assessed_source_count_per_candidate_kind"]
)


@dataclass(frozen=True)
class CandidateGroup:
    """上位候補抽出に必要な発音群の最小観測。"""

    group_id: str
    onset_us: int
    source_event_ids: tuple[str, ...]
    relative_onsets_us: tuple[int, ...]
    pitches: tuple[int, ...]
    held_durations_us: tuple[int, ...]


class ObservedStructureCandidateError(ValueError):
    """観測候補の入力契約または成果物契約に違反した。"""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _fraction_record(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _median(values: Sequence[int]) -> Fraction:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        raise ValueError("median requires at least one value")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return Fraction(ordered[middle])
    return Fraction(ordered[middle - 1] + ordered[middle], 2)


def canonical_semantic_profile_hash(groups: Sequence[CandidateGroup]) -> str:
    """表面上のprofile名を除いた発音群列のhashを返す。"""
    return _sha256(
        [
            {
                "onset_us": group.onset_us,
                "source_event_ids": group.source_event_ids,
                "relative_onsets_us": group.relative_onsets_us,
            }
            for group in groups
        ]
    )


def _pitch_shape_hash(value: object) -> str:
    return _sha256(value)


def build_recurrence_candidates(
    groups: Sequence[CandidateGroup],
    score_positions: Sequence[int],
    *,
    window_sizes: Sequence[int] = RECURRENCE_WINDOW_SIZES,
) -> dict[str, Any]:
    """検索上限内で最長のcanonical反復候補を作る。"""
    if len(groups) != len(score_positions):
        raise ValueError("groups and score positions must have equal length")
    pitch_groups = tuple(group.pitches for group in groups)
    raw = enumerate_pitch_rhythm_recurrences(
        pitch_groups=pitch_groups,
        score_positions=score_positions,
        window_sizes=window_sizes,
    )
    maximum_size = max((int(value) for value in window_sizes), default=0)
    extension = (
        enumerate_pitch_rhythm_recurrences(
            pitch_groups=pitch_groups,
            score_positions=score_positions,
            window_sizes=(maximum_size + 1,),
        )
        if maximum_size >= 3
        else ()
    )
    extension_starts: dict[str, set[int]] = {}
    for record in extension:
        extension_starts.setdefault(str(record["view"]), set()).update(
            int(item["start_group_index"]) for item in record["occurrences"]
        )

    raw_starts = {
        id(record): tuple(int(item["start_group_index"]) for item in record["occurrences"])
        for record in raw
    }
    retained = []
    for record in raw:
        starts = raw_starts[id(record)]
        contained = any(
            str(other["view"]) == str(record["view"])
            and int(other["window_size"]) > int(record["window_size"])
            and raw_starts[id(other)] == starts
            for other in raw
        )
        if not contained:
            retained.append(record)

    compact: dict[tuple[int, tuple[tuple[int, bool, bool], ...]], dict[str, Any]] = {}
    for record in retained:
        view = str(record["view"])
        size = int(record["window_size"])
        next_starts = extension_starts.get(view, set()) if size == maximum_size else set()
        occurrences: list[tuple[int, bool, bool]] = []
        for occurrence in record["occurrences"]:
            start = int(occurrence["start_group_index"])
            left_extension = start > 0 and start - 1 in next_starts
            right_extension = start in next_starts
            occurrences.append((start, left_extension, right_extension))
        pitch_hash = _pitch_shape_hash(record["pitch_shape"])
        compact_key = (size, tuple(occurrences))
        compact.setdefault(
            compact_key,
            {
                "window_size": size,
                "occurrences": [list(item) for item in occurrences],
                "view_pitch_shape_sha256": {},
            },
        )["view_pitch_shape_sha256"][view] = pitch_hash

    candidates = []
    for value in compact.values():
        identity = {
            **value,
            "views": sorted(value["view_pitch_shape_sha256"]),
            "view_pitch_shape_sha256": dict(
                sorted(value["view_pitch_shape_sha256"].items())
            ),
        }
        candidates.append(
            {
                "candidate_id": _sha256(identity),
                **identity,
            }
        )
    candidates.sort(key=lambda item: item["candidate_id"])
    return {
        "status": "assessed" if candidates else "insufficient_evidence",
        "candidates": candidates,
    }


def _plateau_peaks(scores: Mapping[int, Fraction]) -> list[dict[str, Any]]:
    ordered = sorted(scores)
    peaks: list[dict[str, Any]] = []
    index = 0
    while index < len(ordered):
        start_index = index
        value = scores[ordered[index]]
        while (
            index + 1 < len(ordered)
            and ordered[index + 1] == ordered[index] + 1
            and scores[ordered[index + 1]] == value
        ):
            index += 1
        left = scores.get(ordered[start_index] - 1)
        right = scores.get(ordered[index] + 1)
        if (left is None or value > left) and (right is None or value > right):
            peaks.append(
                {
                    "start_group_index": ordered[start_index],
                    "end_group_index": ordered[index],
                    "score": _fraction_record(value),
                }
            )
        index += 1
    return peaks


def _merge_ranges(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for record in sorted(
        records,
        key=lambda item: (
            int(item["start_group_index"]),
            int(item["end_group_index"]),
            int(item["window_size"]),
        ),
    ):
        start = int(record["start_group_index"])
        end = int(record["end_group_index"])
        if merged and start <= int(merged[-1]["end_group_index"]):
            merged[-1]["end_group_index"] = max(end, int(merged[-1]["end_group_index"]))
            merged[-1]["window_sizes"].append(int(record["window_size"]))
            merged[-1]["scores"].append(record["score"])
            continue
        merged.append(
            {
                "start_group_index": start,
                "end_group_index": end,
                "window_sizes": [int(record["window_size"])],
                "scores": [record["score"]],
            }
        )
    for item in merged:
        item["window_sizes"] = sorted(set(item["window_sizes"]))
    return merged


def _family_peak_ranges(
    groups: Sequence[CandidateGroup],
    *,
    window_sizes: Sequence[int],
    family: str,
) -> list[dict[str, Any]]:
    n = len(groups)
    onsets = tuple(group.onset_us for group in groups)
    gaps = tuple(right - left for left, right in pairwise(onsets))
    all_pitches = [pitch for group in groups for pitch in group.pitches]
    all_sizes = [len(group.pitches) for group in groups]
    all_durations = [duration for group in groups for duration in group.held_durations_us]
    ranges = {
        "pitch": max(all_pitches) - min(all_pitches) if all_pitches else 0,
        "size": max(all_sizes) - min(all_sizes) if all_sizes else 0,
        "duration": max(all_durations) - min(all_durations) if all_durations else 0,
    }
    records: list[dict[str, Any]] = []
    for width_value in window_sizes:
        width = int(width_value)
        if width < 1:
            raise ValueError("window sizes must be positive")
        scores: dict[int, Fraction] = {}
        for edge in range(width, n - width + 1):
            if edge <= 0 or edge >= n:
                continue
            if family == "pause":
                local = [
                    gap
                    for index, gap in enumerate(gaps)
                    if edge - width <= index <= edge + width - 1
                    and index != edge - 1
                    and gap > 0
                ]
                if not local:
                    continue
                baseline = _median(local)
                focal = gaps[edge - 1]
                if focal > baseline:
                    scores[edge] = Fraction(focal, 1) / baseline
                continue
            if family != "texture":
                raise ValueError(f"unknown boundary family: {family}")
            left = groups[edge - width : edge]
            right = groups[edge : edge + width]
            features = (
                (
                    [pitch for group in left for pitch in group.pitches],
                    [pitch for group in right for pitch in group.pitches],
                    ranges["pitch"],
                ),
                (
                    [len(group.pitches) for group in left],
                    [len(group.pitches) for group in right],
                    ranges["size"],
                ),
                (
                    [duration for group in left for duration in group.held_durations_us],
                    [duration for group in right for duration in group.held_durations_us],
                    ranges["duration"],
                ),
            )
            score = sum(
                (
                    abs(_median(left_values) - _median(right_values)) / scale
                    if scale
                    else Fraction(0)
                )
                for left_values, right_values, scale in features
            ) / 3
            if score > 0:
                scores[edge] = score
        records.extend({**peak, "window_size": width} for peak in _plateau_peaks(scores))
    return _merge_ranges(records)


def _recurrence_edge_ranges(
    recurrences: Sequence[Mapping[str, Any]], *, group_count: int
) -> list[dict[str, Any]]:
    edges: dict[int, set[str]] = {}
    for candidate in recurrences:
        candidate_id = str(candidate["candidate_id"])
        size = int(candidate["window_size"])
        for occurrence in candidate["occurrences"]:
            start = int(occurrence[0])
            left_extension = bool(occurrence[1])
            right_extension = bool(occurrence[2])
            if start > 0 and not left_extension:
                edges.setdefault(start, set()).add(candidate_id)
            if start + size < group_count and not right_extension:
                edges.setdefault(start + size, set()).add(candidate_id)
    return [
        {
            "start_group_index": edge,
            "end_group_index": edge,
            "candidate_ids": sorted(ids),
        }
        for edge, ids in sorted(edges.items())
    ]


def build_boundary_candidates(
    groups: Sequence[CandidateGroup],
    *,
    recurrences: Sequence[Mapping[str, Any]],
    window_sizes: Sequence[int] = BOUNDARY_WINDOW_SIZES,
) -> dict[str, Any]:
    """異なる二証拠族が重なる境界だけを返す。"""
    families = {
        "pause": _family_peak_ranges(groups, window_sizes=window_sizes, family="pause"),
        "texture": _family_peak_ranges(groups, window_sizes=window_sizes, family="texture"),
        "recurrence": _recurrence_edge_ranges(recurrences, group_count=len(groups)),
    }
    candidates = []
    for edge in range(1, len(groups)):
        supporting = sorted(
            family
            for family, ranges in families.items()
            if any(
                int(item["start_group_index"]) <= edge <= int(item["end_group_index"])
                for item in ranges
            )
        )
        if "pause" in supporting and "texture" in supporting:
            candidates.append(
                {
                    "start_group_index": edge,
                    "end_group_index": edge,
                    "evidence_families": supporting,
                }
            )
    return {
        "status": "assessed" if candidates else "insufficient_evidence",
        "families": families,
        "candidates": candidates,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ObservedStructureCandidateError(f"unable to read JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ObservedStructureCandidateError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        values = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ObservedStructureCandidateError(f"unable to read JSONL: {path}: {error}") from error
    if not all(isinstance(value, dict) for value in values):
        raise ObservedStructureCandidateError(f"JSONL records must be objects: {path}")
    return values


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(record) + b"\n" for record in records)


def _verified_inputs(
    *,
    staging_dir: Path,
    staging_manifest: Path,
    score_timing_files: Path,
    score_timing_manifest: Path,
) -> tuple[dict[str, str], list[dict[str, Any]], str, str]:
    staging = _read_json(staging_manifest)
    schema_version = staging.get("schema_version")
    if schema_version == 1:
        split_role = "development"
        source_key = "development_smf"
    elif schema_version == 2 and staging.get("split_role") in {
        "development",
        "holdout",
        "known_fixture",
    }:
        split_role = str(staging["split_role"])
        source_key = "staged_smf"
    else:
        raise ObservedStructureCandidateError("staging manifest contract is invalid")
    sources = staging.get(source_key)
    if staging.get("status") != "pass" or not isinstance(sources, Mapping):
        raise ObservedStructureCandidateError("staging manifest contract is invalid")
    staging_inputs = staging.get("inputs")
    staging_method_hash = (
        staging_inputs.get("method_manifest")
        if isinstance(staging_inputs, Mapping)
        else None
    )
    if split_role != "development" and not isinstance(staging_method_hash, str):
        raise ObservedStructureCandidateError(
            "staging method manifest SHA-256 is missing"
        )
    expected_sources = {
        str(name): str(digest).lower() for name, digest in sources.items()
    }
    actual_names = {
        path.name
        for path in Path(staging_dir).iterdir()
        if path.is_file() and path.suffix.casefold() == ".mid"
    }
    if set(expected_sources) != actual_names:
        raise ObservedStructureCandidateError("staging source set mismatch")
    for name, digest in expected_sources.items():
        if sha256_file(Path(staging_dir) / name).lower() != digest:
            raise ObservedStructureCandidateError(f"staging SMF SHA-256 mismatch: {name}")

    timing_manifest = _read_json(score_timing_manifest)
    timing_outputs = timing_manifest.get("outputs")
    timing_inputs = timing_manifest.get("inputs")
    if (
        timing_manifest.get("schema_version") != 1
        or timing_manifest.get("status") != "pass"
        or timing_manifest.get("split_role", "development") != split_role
        or not isinstance(timing_outputs, Mapping)
        or not isinstance(timing_inputs, Mapping)
    ):
        raise ObservedStructureCandidateError("ScoreTiming manifest contract is invalid")
    expected_files_hash = timing_outputs.get("files.jsonl")
    if (
        not isinstance(expected_files_hash, str)
        or sha256_file(score_timing_files).lower() != expected_files_hash.lower()
    ):
        raise ObservedStructureCandidateError("ScoreTiming files SHA-256 mismatch")
    expected_staging_hash = timing_inputs.get("staging_manifest")
    if (
        not isinstance(expected_staging_hash, str)
        or sha256_file(staging_manifest).lower() != expected_staging_hash.lower()
    ):
        raise ObservedStructureCandidateError("staging manifest SHA-256 mismatch")
    if (
        staging_method_hash is not None
        and timing_inputs.get("method_manifest") != staging_method_hash
    ):
        raise ObservedStructureCandidateError(
            "ScoreTiming and staging method manifest SHA-256 differ"
        )
    records = _read_jsonl(score_timing_files)
    by_name = {str(record.get("name")): record for record in records}
    if len(by_name) != len(records) or set(by_name) != set(expected_sources):
        raise ObservedStructureCandidateError("ScoreTiming source set mismatch")
    for name, record in by_name.items():
        if str(record.get("source_sha256", "")).lower() != expected_sources[name]:
            raise ObservedStructureCandidateError(f"ScoreTiming source SHA-256 mismatch: {name}")
    return (
        expected_sources,
        [by_name[name] for name in sorted(by_name, key=str.casefold)],
        split_role,
        source_key,
    )


def _candidate_groups(
    groups: Sequence[object], notes_by_id: Mapping[str, object]
) -> tuple[CandidateGroup, ...]:
    result = []
    for index, group in enumerate(groups):
        missing = [event_id for event_id in group.source_event_ids if event_id not in notes_by_id]
        if missing:
            raise ObservedStructureCandidateError(
                f"grouping source event is missing: {missing[0]}"
            )
        notes = [notes_by_id[event_id] for event_id in group.source_event_ids]
        result.append(
            CandidateGroup(
                group_id=f"group-{index}",
                onset_us=int(group.onset_us),
                source_event_ids=tuple(group.source_event_ids),
                relative_onsets_us=tuple(int(value) for value in group.relative_onsets_us),
                pitches=tuple(int(note.pitch) for note in notes),
                held_durations_us=tuple(int(note.offset_us - note.onset_us) for note in notes),
            )
        )
    return tuple(result)


def _boundary_anchor(groups: Sequence[CandidateGroup], edge: int) -> dict[str, Any]:
    return {
        "before_group_id": groups[edge - 1].group_id,
        "after_group_id": groups[edge].group_id,
        "before_source_event_ids": groups[edge - 1].source_event_ids,
        "after_source_event_ids": groups[edge].source_event_ids,
        "before_onset_us": groups[edge - 1].onset_us,
        "after_onset_us": groups[edge].onset_us,
    }


def _with_boundary_anchors(
    groups: Sequence[CandidateGroup], result: Mapping[str, Any]
) -> dict[str, Any]:
    candidates = []
    for item in result["candidates"]:
        start = int(item["start_group_index"])
        candidates.append(
            {
                **item,
                "anchor": _boundary_anchor(groups, start),
            }
        )
    return {**result, "candidates": candidates}


def _strip_recurrence_rhythm(result: Mapping[str, Any]) -> dict[str, Any]:
    return dict(result)


def _recurrence_rhythm_sha256(
    recurrence_candidates: Sequence[Mapping[str, Any]], score_positions: Sequence[int]
) -> str:
    positions = tuple(int(value) for value in score_positions)
    normalized = []
    for candidate in recurrence_candidates:
        size = int(candidate["window_size"])
        occurrences = []
        for occurrence in candidate["occurrences"]:
            start = int(occurrence[0])
            values = positions[start : start + size]
            intervals = tuple(right - left for left, right in pairwise(values))
            if not intervals or any(value <= 0 for value in intervals):
                raise ObservedStructureCandidateError("score positions must be increasing")
            divisor = math.gcd(*intervals)
            occurrences.append([start, [value // divisor for value in intervals]])
        normalized.append(
            {
                "candidate_id": candidate["candidate_id"],
                "view_pitch_shape_sha256": candidate["view_pitch_shape_sha256"],
                "window_size": size,
                "occurrences": occurrences,
            }
        )
    return _sha256(normalized)


def _boundary_comparison_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": result["status"],
        "pause": result["families"]["pause"],
        "texture": result["families"]["texture"],
        "candidates": [
            {
                "start_group_index": item["start_group_index"],
                "end_group_index": item["end_group_index"],
                "evidence_families": item["evidence_families"],
            }
            for item in result["candidates"]
        ],
    }


def _control_groups(
    groups: Sequence[CandidateGroup], *, kind: str
) -> tuple[CandidateGroup, ...]:
    if kind == "transpose":
        return tuple(
            CandidateGroup(
                **{
                    **group.__dict__,
                    "pitches": tuple(pitch + 5 for pitch in group.pitches),
                }
            )
            for group in groups
        )
    if kind == "pitch-order":
        amount = max(1, len(groups) // 3)
        payloads = [group.pitches for group in groups]
        payloads = payloads[amount:] + payloads[:amount]
        return tuple(
            CandidateGroup(**{**group.__dict__, "pitches": tuple(payloads[index])})
            for index, group in enumerate(groups)
        )
    if kind == "gap-rotation":
        gaps = [right.onset_us - left.onset_us for left, right in pairwise(groups)]
        if not gaps:
            return tuple(groups)
        rotated = gaps[1:] + gaps[:1]
        onsets = [groups[0].onset_us]
        for gap in rotated:
            onsets.append(onsets[-1] + gap)
        return tuple(
            CandidateGroup(**{**group.__dict__, "onset_us": onsets[index]})
            for index, group in enumerate(groups)
        )
    raise ValueError(f"unknown control kind: {kind}")


def _positive_controls() -> dict[str, dict[str, Any]]:
    gaps = (100_000,) * 5 + (500_000,) + (100_000,) * 5
    onsets = [0]
    for gap in gaps:
        onsets.append(onsets[-1] + gap)
    groups = tuple(
        CandidateGroup(
            group_id=f"g{index}",
            onset_us=onsets[index],
            source_event_ids=(f"e{index}",),
            relative_onsets_us=(0,),
            pitches=((index + 12,) if index >= 6 else (index,)),
            held_durations_us=(80_000,),
        )
        for index in range(12)
    )
    boundary = build_boundary_candidates(groups, recurrences=(), window_sizes=(4,))
    boundary_actual = [
        (int(item["start_group_index"]), int(item["end_group_index"]))
        for item in boundary["candidates"]
    ]
    pause_actual = [
        (int(item["start_group_index"]), int(item["end_group_index"]))
        for item in boundary["families"]["pause"]
        if int(item["start_group_index"]) <= 6 <= int(item["end_group_index"])
    ]
    texture_actual = [
        (int(item["start_group_index"]), int(item["end_group_index"]))
        for item in boundary["families"]["texture"]
        if int(item["start_group_index"]) <= 6 <= int(item["end_group_index"])
    ]
    recurrence_actual = [
        int(item["start_group_index"])
        for item in _recurrence_edge_ranges(
            (
                {
                    "candidate_id": "known-long-recurrence",
                    "window_size": 4,
                    "occurrences": ((2, False, True), (10, True, False)),
                },
            ),
            group_count=16,
        )
    ]
    return {
        "pause": {
            "status": "pass" if pause_actual and boundary_actual == [(6, 6)] else "fail",
            "actual": pause_actual,
        },
        "texture": {
            "status": "pass" if texture_actual and boundary_actual == [(6, 6)] else "fail",
            "actual": texture_actual,
        },
        "recurrence_endpoints": {
            "status": "pass" if recurrence_actual == [2, 14] else "fail",
            "actual": recurrence_actual,
        },
    }


def _source_record(
    *,
    name: str,
    path: Path,
    timing_record: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observed = load_observed_smf(path)
    performance = build_observed_performance(observed)
    profiles = build_grouping_profiles(performance)
    notes_by_id = {note.note_on_event_id: note for note in performance.notes}
    semantic: dict[str, dict[str, Any]] = {}
    surface_to_semantic: dict[str, str] = {}
    for profile_id, raw_groups in sorted(profiles.items()):
        groups = _candidate_groups(raw_groups, notes_by_id)
        profile_hash = canonical_semantic_profile_hash(groups)
        surface_to_semantic[profile_id] = profile_hash
        if profile_hash in semantic:
            semantic[profile_hash]["surface_profile_ids"].append(profile_id)
            continue
        uniform_positions = tuple(range(len(groups)))
        recurrence = build_recurrence_candidates(groups, uniform_positions)
        boundary = build_boundary_candidates(
            groups,
            recurrences=recurrence["candidates"],
        )
        boundary = _with_boundary_anchors(groups, boundary)
        semantic[profile_hash] = {
            "semantic_profile_hash": profile_hash,
            "surface_profile_ids": [profile_id],
            "group_count": len(groups),
            "groups": groups,
            "recurrence": recurrence,
            "boundary": boundary,
        }

    alias_groups: dict[str, dict[str, Any]] = {}
    candidates = timing_record.get("candidates")
    if not isinstance(candidates, list):
        raise ObservedStructureCandidateError(f"ScoreTiming candidates are invalid: {name}")
    for candidate in candidates:
        profile_id = str(candidate.get("grouping_profile_id"))
        if profile_id not in surface_to_semantic:
            raise ObservedStructureCandidateError(
                f"ScoreTiming grouping profile is missing: {name}: {profile_id}"
            )
        profile_hash = surface_to_semantic[profile_id]
        groups = semantic[profile_hash]["groups"]
        refs = candidate.get("attack_group_refs")
        if not isinstance(refs, list) or len(refs) != len(groups):
            raise ObservedStructureCandidateError(
                f"ScoreTiming group count mismatch: {name}: {candidate.get('candidate_id')}"
            )
        if [int(ref.get("group_index", -1)) for ref in refs] != list(range(len(groups))):
            raise ObservedStructureCandidateError(
                f"ScoreTiming group index mismatch: {name}: {candidate.get('candidate_id')}"
            )
        positions = tuple(int(ref["score_position"]) for ref in refs)
        recurrence = build_recurrence_candidates(groups, positions)
        signature = {
            "source_sha256": observed.source_sha256,
            "semantic_profile_hash": profile_hash,
            "score_positions_sha256": _sha256(positions),
            "recurrence_rhythm_sha256": _recurrence_rhythm_sha256(
                recurrence["candidates"], positions
            ),
        }
        alias_id = _sha256(signature)
        alias_groups.setdefault(
            alias_id,
            {
                "alias_id": alias_id,
                **signature,
                "candidate_ids": [],
                "recurrence_status": recurrence["status"],
                "recurrence_candidate_count": len(recurrence["candidates"]),
            },
        )["candidate_ids"].append(str(candidate["candidate_id"]))

    controls = []
    public_profiles = []
    for profile_hash, profile in sorted(semantic.items()):
        groups = profile.pop("groups")
        base_recurrence = profile["recurrence"]
        base_boundary = profile["boundary"]
        selection_count = len(
            {
                edge
                for item in base_boundary["candidates"]
                for edge in range(
                    int(item["start_group_index"]), int(item["end_group_index"]) + 1
                )
            }
        )
        internal_count = max(0, len(groups) - 1)
        profile["boundary_selection_rate"] = (
            selection_count / internal_count if internal_count else 0.0
        )
        profile["recurrence"] = _strip_recurrence_rhythm(base_recurrence)
        profile["boundary"] = {
            **base_boundary,
            "families": {
                "pause": base_boundary["families"]["pause"],
                "texture": base_boundary["families"]["texture"],
            },
        }
        public_profiles.append(profile)

        transposed = _control_groups(groups, kind="transpose")
        transposed_recurrence = build_recurrence_candidates(
            transposed, tuple(range(len(transposed)))
        )
        transposed_boundary = build_boundary_candidates(
            transposed,
            recurrences=transposed_recurrence["candidates"],
        )
        controls.append(
            {
                "name": name,
                "semantic_profile_hash": profile_hash,
                "control": "transpose-5",
                "status": "pass"
                if _sha256(base_recurrence) == _sha256(transposed_recurrence)
                and _sha256(_boundary_comparison_payload(base_boundary))
                == _sha256(_boundary_comparison_payload(transposed_boundary))
                else "fail",
            }
        )
        pitch_order = _control_groups(groups, kind="pitch-order")
        pitch_result = build_recurrence_candidates(
            pitch_order, tuple(range(len(pitch_order)))
        )
        controls.append(
            {
                "name": name,
                "semantic_profile_hash": profile_hash,
                "control": "pitch-order",
                "status": "detected"
                if _sha256(base_recurrence) != _sha256(pitch_result)
                else "not_detected",
            }
        )
        gaps = [right.onset_us - left.onset_us for left, right in pairwise(groups)]
        if len(set(gaps)) <= 1:
            gap_status = "not_applicable"
        else:
            gap_groups = _control_groups(groups, kind="gap-rotation")
            gap_recurrence = build_recurrence_candidates(
                gap_groups, tuple(range(len(gap_groups)))
            )
            gap_boundary = build_boundary_candidates(
                gap_groups,
                recurrences=gap_recurrence["candidates"],
            )
            gap_status = (
                "detected"
                if _sha256(base_boundary["families"]["pause"])
                != _sha256(gap_boundary["families"]["pause"])
                else "not_detected"
            )
        controls.append(
            {
                "name": name,
                "semantic_profile_hash": profile_hash,
                "control": "gap-rotation",
                "status": gap_status,
            }
        )

    return (
        {
            "schema_version": 2,
            "name": name,
            "status": "assessed",
            "source_sha256": observed.source_sha256,
            "source_ledger_sha256": observed.ledger_sha256,
            "semantic_profiles": public_profiles,
            "score_timing_candidate_count": len(candidates),
            "score_timing_aliases": [
                {**alias, "candidate_ids": sorted(alias["candidate_ids"])}
                for alias in sorted(alias_groups.values(), key=lambda item: item["alias_id"])
            ],
        },
        controls,
    )


def _eligible_recurrence_edges(
    recurrence_candidates: Sequence[Mapping[str, Any]], *, group_count: int
) -> dict[str, list[str]]:
    edges: dict[int, set[str]] = {}
    for candidate in recurrence_candidates:
        candidate_id = str(candidate["candidate_id"])
        size = int(candidate["window_size"])
        for occurrence in candidate["occurrences"]:
            start = int(occurrence[0])
            if start > 0 and not bool(occurrence[1]):
                edges.setdefault(start, set()).add(candidate_id)
            if start + size < group_count and not bool(occurrence[2]):
                edges.setdefault(start + size, set()).add(candidate_id)
    return {str(edge): sorted(ids) for edge, ids in sorted(edges.items())}


def decode_source_candidates(
    source_record: Mapping[str, Any], timing_record: Mapping[str, Any]
) -> dict[str, Any]:
    """V2と検証済み下位recordだけから省略した関係を復元する。"""
    if source_record.get("schema_version") != 2:
        raise ObservedStructureCandidateError("source candidate schema version is invalid")
    source_sha256 = str(source_record.get("source_sha256", ""))
    if str(timing_record.get("source_sha256", "")) != source_sha256:
        raise ObservedStructureCandidateError("decoder source SHA-256 mismatch")
    lower_candidates = timing_record.get("candidates")
    if not isinstance(lower_candidates, list):
        raise ObservedStructureCandidateError("decoder lower candidates are invalid")
    lower_by_id = {str(item.get("candidate_id")): item for item in lower_candidates}
    profiles = {
        str(profile["semantic_profile_hash"]): profile
        for profile in source_record.get("semantic_profiles", [])
    }
    aliases = []
    for alias in source_record.get("score_timing_aliases", []):
        profile_hash = str(alias["semantic_profile_hash"])
        if profile_hash not in profiles:
            raise ObservedStructureCandidateError("decoder semantic profile is missing")
        profile = profiles[profile_hash]
        decoded_members = []
        for candidate_id in alias["candidate_ids"]:
            lower = lower_by_id.get(str(candidate_id))
            if lower is None:
                raise ObservedStructureCandidateError(
                    f"decoder lower candidate is missing: {candidate_id}"
                )
            positions = tuple(
                int(item["score_position"]) for item in lower["attack_group_refs"]
            )
            if _sha256(positions) != alias["score_positions_sha256"]:
                raise ObservedStructureCandidateError("decoder score positions SHA-256 mismatch")
            rhythm_hash = _recurrence_rhythm_sha256(
                profile["recurrence"]["candidates"], positions
            )
            if rhythm_hash != alias["recurrence_rhythm_sha256"]:
                raise ObservedStructureCandidateError("decoder rhythm SHA-256 mismatch")
            decoded_members.append(
                {
                    "join_key": [source_sha256, str(candidate_id)],
                    "score_positions": list(positions),
                    "recurrence_rhythm_sha256": rhythm_hash,
                }
            )
        aliases.append(
            {
                "alias_id": alias["alias_id"],
                "members": decoded_members,
            }
        )
    return {
        "aliases": aliases,
        "eligible_recurrence_edges": {
            profile_hash: _eligible_recurrence_edges(
                profile["recurrence"]["candidates"],
                group_count=int(profile["group_count"]),
            )
            for profile_hash, profile in profiles.items()
        },
    }


def canonical_recurrence_relations(source_record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """V1/V2の反復payloadをcandidate IDに依存しない同じ関係へ正規化する。"""
    relations = []
    schema_version = source_record.get("schema_version", 1)
    for profile in source_record.get("semantic_profiles", []):
        profile_hash = str(profile["semantic_profile_hash"])
        for candidate in profile["recurrence"]["candidates"]:
            if schema_version == 2:
                occurrences = [
                    [int(item[0]), bool(item[1]), bool(item[2])]
                    for item in candidate["occurrences"]
                ]
                views = [
                    (str(view), str(candidate["view_pitch_shape_sha256"][view]))
                    for view in candidate["views"]
                ]
            else:
                occurrences = [
                    [
                        int(item["start_group_index"]),
                        bool(item["left_extension_beyond_cap"]),
                        bool(item["right_extension_beyond_cap"]),
                    ]
                    for item in candidate["occurrences"]
                ]
                views = [
                    (str(candidate["view"]), str(candidate["pitch_shape_sha256"]))
                ]
            for view, pitch_hash in views:
                relations.append(
                    {
                        "semantic_profile_hash": profile_hash,
                        "view": view,
                        "pitch_shape_sha256": pitch_hash,
                        "window_size": int(candidate["window_size"]),
                        "occurrences": occurrences,
                    }
                )
    return sorted(
        relations,
        key=lambda item: (
            item["semantic_profile_hash"],
            item["view"],
            item["pitch_shape_sha256"],
            item["window_size"],
            item["occurrences"],
        ),
    )


def run_observed_structure_candidates(
    *,
    staging_dir: Path,
    staging_manifest: Path,
    score_timing_files: Path,
    score_timing_manifest: Path,
    output_dir: Path,
    repository_root: Path | None = None,
    method_manifest: Path | None = None,
) -> dict[str, Any]:
    """固定済み開発標本へ観測候補器を適用し、決定的成果物を保存する。"""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ObservedStructureCandidateError(f"output directory is not empty: {output_dir}")
    sources, timing_records, split_role, source_key = _verified_inputs(
        staging_dir=Path(staging_dir),
        staging_manifest=Path(staging_manifest),
        score_timing_files=Path(score_timing_files),
        score_timing_manifest=Path(score_timing_manifest),
    )
    method_manifest_hash: str | None = None
    if split_role != "development" and (
        repository_root is None or method_manifest is None
    ):
        raise ObservedStructureCandidateError(
            "method manifest is required for non-development staging"
        )
    if method_manifest is not None:
        if repository_root is None:
            raise ObservedStructureCandidateError(
                "repository root is required with method manifest"
            )
        method_manifest_hash = verify_method_manifest(
            repository_root=Path(repository_root),
            manifest_path=Path(method_manifest),
        )
        timing_method_hash = _read_json(Path(score_timing_manifest)).get("inputs", {}).get(
            "method_manifest"
        )
        if timing_method_hash != method_manifest_hash:
            raise ObservedStructureCandidateError(
                "ScoreTiming method manifest SHA-256 mismatch"
            )
    source_records = []
    controls = []
    for timing_record in timing_records:
        name = str(timing_record["name"])
        try:
            source, source_controls = _source_record(
                name=name,
                path=Path(staging_dir) / name,
                timing_record=timing_record,
            )
            source_records.append(source)
            controls.extend(source_controls)
        except (OSError, EOFError, ValueError) as error:
            source_records.append(
                {
                    "name": name,
                    "status": "unable_to_investigate",
                    "reason": str(error),
                    "semantic_profiles": [],
                    "score_timing_candidate_count": 0,
                    "score_timing_aliases": [],
                }
            )
    source_bytes = _jsonl_bytes(source_records)
    control_bytes = _jsonl_bytes(controls)
    positive_controls = _positive_controls()
    assessed = [record for record in source_records if record["status"] == "assessed"]
    boundary_source_count = sum(
        any(profile["boundary"]["status"] == "assessed" for profile in record["semantic_profiles"])
        for record in assessed
    )
    recurrence_source_count = sum(
        any(
            profile["recurrence"]["status"] == "assessed"
            for profile in record["semantic_profiles"]
        )
        for record in assessed
    )
    selection_rates = [
        float(profile["boundary_selection_rate"])
        for record in assessed
        for profile in record["semantic_profiles"]
        if profile["boundary"]["status"] == "assessed"
    ]
    transpose_pass = all(
        record["status"] == "pass"
        for record in controls
        if record["control"] == "transpose-5"
    )
    pitch_detected = any(
        record["status"] == "detected"
        for record in controls
        if record["control"] == "pitch-order"
    )
    gap_detected = any(
        record["status"] == "detected"
        for record in controls
        if record["control"] == "gap-rotation"
    )
    timing_size = Path(score_timing_files).stat().st_size
    adoption_checks = {
        "positive_controls": all(
            control["status"] == "pass" for control in positive_controls.values()
        ),
        "boundary_multiple_sources": boundary_source_count >= MINIMUM_ASSESSED_SOURCE_COUNT,
        "recurrence_multiple_sources": (
            recurrence_source_count >= MINIMUM_ASSESSED_SOURCE_COUNT
        ),
        "boundary_selection_rate": all(
            rate <= MAXIMUM_BOUNDARY_SELECTION_RATE for rate in selection_rates
        ),
        "upper_output_not_larger_than_lower_input": len(source_bytes) <= timing_size,
        "transposition_invariance": transpose_pass,
        "pitch_order_sensitivity": pitch_detected,
        "gap_rotation_sensitivity": gap_detected,
    }
    status = "pass" if len(assessed) == len(source_records) else "partial"
    summary = {
        "schema_version": 2,
        "status": status,
        "split_role": split_role,
        "source_count": len(source_records),
        "assessed_source_count": len(assessed),
        "semantic_profile_count": sum(
            len(record["semantic_profiles"]) for record in source_records
        ),
        "boundary_assessed_source_count": boundary_source_count,
        "recurrence_assessed_source_count": recurrence_source_count,
        "source_candidates_bytes": len(source_bytes),
        "score_timing_files_bytes": timing_size,
        "positive_controls": positive_controls,
        "adoption_checks": adoption_checks,
        "adoption_status": "pass" if all(adoption_checks.values()) else "rejected",
    }
    run_spec = {
        "schema_version": 2,
        "split_role": split_role,
        "inputs": {
            "staging_manifest": sha256_file(staging_manifest),
            "score_timing_manifest": sha256_file(score_timing_manifest),
            "score_timing_files": sha256_file(score_timing_files),
            source_key: dict(sorted(sources.items())),
        },
        "boundary_window_sizes": list(BOUNDARY_WINDOW_SIZES),
        "recurrence_window_sizes": list(RECURRENCE_WINDOW_SIZES),
        "recurrence_extension_probe_size": RECURRENCE_EXTENSION_PROBE_SIZE,
        "minimum_boundary_evidence_family_count": 2,
        "maximum_boundary_selection_rate": MAXIMUM_BOUNDARY_SELECTION_RATE,
        "minimum_assessed_source_count_per_candidate_kind": (
            MINIMUM_ASSESSED_SOURCE_COUNT
        ),
        "output_size_bound": "source-candidates.jsonl <= score-timing files.jsonl",
    }
    if method_manifest_hash is not None:
        run_spec["inputs"]["method_manifest"] = method_manifest_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(output_dir / "source-candidates.jsonl", source_bytes)
    atomic_write_bytes(output_dir / "negative-controls.jsonl", control_bytes)
    atomic_write_json(output_dir / "summary.json", summary)
    atomic_write_json(output_dir / "run-spec.json", run_spec)
    output_names = (
        "negative-controls.jsonl",
        "run-spec.json",
        "source-candidates.jsonl",
        "summary.json",
    )
    atomic_write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 2,
            "status": status,
            "split_role": split_role,
            "outputs": {
                name: sha256_file(output_dir / name) for name in output_names
            },
            "inputs": (
                {"method_manifest": method_manifest_hash}
                if method_manifest_hash is not None
                else {}
            ),
        },
    )
    return {
        "status": status,
        "source_count": len(source_records),
        "boundary_assessed_source_count": boundary_source_count,
        "recurrence_assessed_source_count": recurrence_source_count,
        "adoption_status": summary["adoption_status"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """固定済み開発標本の観測候補をCLIで再現する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--staging-manifest", type=Path, required=True)
    parser.add_argument("--score-timing-files", type=Path, required=True)
    parser.add_argument("--score-timing-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--method-manifest", type=Path)
    arguments = parser.parse_args(argv)
    result = run_observed_structure_candidates(
        staging_dir=arguments.staging_dir,
        staging_manifest=arguments.staging_manifest,
        score_timing_files=arguments.score_timing_files,
        score_timing_manifest=arguments.score_timing_manifest,
        output_dir=arguments.output_dir,
        repository_root=arguments.repository_root,
        method_manifest=arguments.method_manifest,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
