"""前景や和声を確定せず、SMF観測が各和声をどれだけ支持するか記述する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_musical_composer.performance_pipeline import (
    ScoreSpec,
    harmony_pitch_classes,
    ordered_leaf_schedule,
)
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
from llm_musical_composer.reference_timing_hypothesis import align_known_score_note_events
from llm_musical_composer.reference_timing_run import KnownTimingSource
from llm_musical_composer.run_state import atomic_write_bytes, atomic_write_json, sha256_file
from llm_musical_composer.score_timing_hypothesis import GroupingAttack

HARMONY_QUALITIES = ("major", "minor", "diminished", "major-seventh")
LOW_REGISTER_LIMIT = 48


@dataclass(frozen=True)
class VerifiedRoundtripSource:
    """受動exportの二段hashで検証したroundtrip SMF。"""

    name: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class ObservationDiagnosticSource:
    """観測診断へ渡す、hash付きのSMF入力。"""

    case_id: str
    group_id: str
    path: Path
    expected_sha256: str


def harmony_keys() -> tuple[tuple[int, str], ...]:
    """現行正方向契約の48和声を固定順で返す。"""
    return tuple(
        (root_pitch_class, quality)
        for root_pitch_class in range(12)
        for quality in HARMONY_QUALITIES
    )


def pitch_class_inclusion_table() -> list[dict[str, Any]]:
    """48和声と12音高クラスの静的包含表を返す。"""
    return [
        {
            "harmony_index": harmony_index,
            "root_pitch_class": root_pitch_class,
            "quality": quality,
            "pitch_classes": sorted(harmony_pitch_classes(root_pitch_class, quality)),
        }
        for harmony_index, (root_pitch_class, quality) in enumerate(harmony_keys())
    ]


def one_or_more_support_ids(pitch_classes: set[int]) -> frozenset[int]:
    """観測音高クラスを一つ以上含む和声index集合を返す。"""
    normalized = {int(value) % 12 for value in pitch_classes}
    return frozenset(
        record["harmony_index"]
        for record in pitch_class_inclusion_table()
        if normalized.intersection(record["pitch_classes"])
    )


def _fraction_record(numerator: int, denominator: int) -> dict[str, int] | None:
    if denominator == 0:
        return None
    divisor = math.gcd(numerator, denominator)
    return {
        "numerator": numerator // divisor,
        "denominator": denominator // divisor,
    }


def harmony_support(
    pitches: Sequence[int],
    *,
    root_pitch_class: int,
    quality: str,
) -> dict[str, Any]:
    """一つの和声が観測音を説明する件数を、順位付けせず返す。"""
    pitch_values = tuple(int(pitch) for pitch in pitches)
    chord = harmony_pitch_classes(int(root_pitch_class), quality)
    chord_tone_count = sum(pitch % 12 in chord for pitch in pitch_values)
    root = int(root_pitch_class) % 12
    intervals = tuple((pitch % 12 - root) % 12 for pitch in pitch_values)
    low_intervals = tuple(
        interval
        for pitch, interval in zip(pitch_values, intervals, strict=True)
        if pitch < LOW_REGISTER_LIMIT
    )
    third_intervals = {3, 4}
    return {
        "status": "assessed",
        "pitch_count": len(pitch_values),
        "harmony_pitch_class_count": len(chord),
        "chord_tone_count": chord_tone_count,
        "non_chord_tone_count": len(pitch_values) - chord_tone_count,
        "chord_tone_rate": _fraction_record(chord_tone_count, len(pitch_values)),
        "low_register": {
            "root_or_fifth_count": sum(interval in {0, 7} for interval in low_intervals),
            "third_count": sum(interval in third_intervals for interval in low_intervals),
            "other_count": sum(
                interval not in {0, 7} | third_intervals for interval in low_intervals
            ),
        },
    }


def _all_harmony_support(pitches: Sequence[int]) -> list[dict[str, Any]]:
    return [
        {
            "harmony_index": harmony_index,
            "root_pitch_class": root_pitch_class,
            "quality": quality,
            **harmony_support(
                pitches,
                root_pitch_class=root_pitch_class,
                quality=quality,
            ),
        }
        for harmony_index, (root_pitch_class, quality) in enumerate(harmony_keys())
    ]


def build_group_support(
    *,
    groups: Sequence[GroupingAttack],
    notes_by_id: Mapping[str, ObservedNote],
    note_matching_status: str,
) -> list[dict[str, Any]]:
    """発音群ごとにattackとkey-heldの48和声支持を分離して返す。"""
    all_notes = tuple(notes_by_id.values())
    result: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        try:
            attack_pitches = tuple(
                notes_by_id[event_id].pitch for event_id in group.source_event_ids
            )
        except KeyError as error:
            raise ValueError(f"source event missing from note ledger: {error.args[0]}") from error
        if not attack_pitches:
            raise ValueError("attack group must contain at least one member")
        attack_support = _all_harmony_support(attack_pitches)
        if note_matching_status == "assessed":
            key_held_pitches = tuple(
                note.pitch
                for note in all_notes
                if note.onset_us <= group.onset_us < note.offset_us
            )
            key_held_support: list[dict[str, Any]] | dict[str, str] = _all_harmony_support(
                key_held_pitches
            )
        else:
            key_held_support = {
                "status": "unable_to_investigate",
                "reason": "note_matching_not_assessed",
            }
        harmony_records: list[dict[str, Any]] = []
        for harmony_index, (root_pitch_class, quality) in enumerate(harmony_keys()):
            record: dict[str, Any] = {
                "harmony_index": harmony_index,
                "root_pitch_class": root_pitch_class,
                "quality": quality,
                "attack_support": {
                    key: value
                    for key, value in attack_support[harmony_index].items()
                    if key not in {"harmony_index", "root_pitch_class", "quality"}
                },
            }
            if isinstance(key_held_support, list):
                record["key_held_support"] = {
                    key: value
                    for key, value in key_held_support[harmony_index].items()
                    if key not in {"harmony_index", "root_pitch_class", "quality"}
                }
            else:
                record["key_held_support"] = key_held_support
            harmony_records.append(record)
        result.append(
            {
                "group_index": group_index,
                "onset_us": group.onset_us,
                "member_count": len(group.source_event_ids),
                "harmony_support": harmony_records,
            }
        )
    return result


def stepwise_motion_evidence(pitches: Sequence[int]) -> str | None:
    """音価を使わず、三音の半音・全音進行形だけを記述する。"""
    values = tuple(int(pitch) for pitch in pitches)
    if len(values) != 3:
        raise ValueError("stepwise motion evidence requires exactly three pitches")
    first = values[1] - values[0]
    second = values[2] - values[1]
    if 0 < abs(first) <= 2 and first == second:
        return "monotonic_step"
    if 0 < abs(first) <= 2 and second == -first:
        return "returning_step"
    return None


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record must be an object: {path}:{line_number}")
        records.append(value)
    return records


def verify_roundtrip_sources(
    *,
    root: Path,
    excluded_names: frozenset[str] = frozenset({"rut.mid", "aimusic01.mid"}),
) -> tuple[VerifiedRoundtripSource, ...]:
    """manifestからJSONL、JSONLからSMFへ二段でhashを照合する。"""
    root = Path(root)
    manifest_path = root / "manifest.json"
    roundtrip_path = root / "roundtrip.jsonl"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "pass":
        raise ValueError("roundtrip manifest status is not pass")
    expected_jsonl_hash = manifest.get("outputs", {}).get("roundtrip.jsonl")
    if expected_jsonl_hash != sha256_file(roundtrip_path):
        raise ValueError("roundtrip.jsonl SHA-256 mismatch")
    sources: list[VerifiedRoundtripSource] = []
    for record in _read_jsonl(roundtrip_path):
        name = record.get("name")
        regenerated_hash = record.get("regenerated_sha256")
        if not isinstance(name, str) or not isinstance(regenerated_hash, str):
            raise ValueError("roundtrip record identity is invalid")
        if name in excluded_names:
            continue
        path = root / "roundtrip-smf" / name
        if not path.is_file():
            raise ValueError(f"roundtrip SMF is missing: {name}")
        actual_hash = sha256_file(path)
        if actual_hash != regenerated_hash:
            raise ValueError(f"roundtrip SMF SHA-256 mismatch: {name}")
        sources.append(VerifiedRoundtripSource(name=name, path=path, sha256=actual_hash))
    names = [source.name for source in sources]
    if len(names) != len(set(names)):
        raise ValueError("roundtrip source names must be unique")
    return tuple(sorted(sources, key=lambda source: source.name.casefold()))


def canonical_sha256(value: object) -> str:
    """診断payloadの決定的hashを返す。"""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")


def _pitch_class_counts(pitches: Sequence[int]) -> list[int]:
    counts = [0] * 12
    for pitch in pitches:
        counts[int(pitch) % 12] += 1
    return counts


def _support_histograms(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    attack_tone_counts: Counter[int] = Counter()
    key_held_tone_counts: Counter[int] = Counter()
    attack_supporting_harmony_counts: Counter[int] = Counter()
    key_held_supporting_harmony_counts: Counter[int] = Counter()
    key_held_assessed_group_count = 0
    for record in records:
        for group in record.get("groups", []):
            attack_counts = list(group["attack_pitch_class_counts"])
            attack_values = [
                sum(attack_counts[pitch_class] for pitch_class in item["pitch_classes"])
                for item in pitch_class_inclusion_table()
            ]
            attack_tone_counts.update(attack_values)
            attack_supporting_harmony_counts[sum(value > 0 for value in attack_values)] += 1
            key_held = group["key_held_pitch_class_counts"]
            if not isinstance(key_held, list):
                continue
            key_held_assessed_group_count += 1
            key_held_values = [
                sum(key_held[pitch_class] for pitch_class in item["pitch_classes"])
                for item in pitch_class_inclusion_table()
            ]
            key_held_tone_counts.update(key_held_values)
            key_held_supporting_harmony_counts[sum(value > 0 for value in key_held_values)] += 1
    return {
        "attack_chord_tone_count_histogram": {
            str(key): value for key, value in sorted(attack_tone_counts.items())
        },
        "attack_supporting_harmony_count_histogram": {
            str(key): value for key, value in sorted(attack_supporting_harmony_counts.items())
        },
        "key_held_assessed_group_count": key_held_assessed_group_count,
        "key_held_chord_tone_count_histogram": {
            str(key): value for key, value in sorted(key_held_tone_counts.items())
        },
        "key_held_supporting_harmony_count_histogram": {
            str(key): value
            for key, value in sorted(key_held_supporting_harmony_counts.items())
        },
    }


def _observe_source(
    source: ObservationDiagnosticSource,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = {
        "case_id": source.case_id,
        "group_id": source.group_id,
        "source_name": source.path.name,
        "expected_source_sha256": source.expected_sha256,
    }
    if not source.path.is_file():
        unable = {**base, "status": "unable_to_investigate", "reason": "source_missing"}
        return unable, {**unable, "groups": []}
    actual_hash = sha256_file(source.path)
    if actual_hash != source.expected_sha256:
        unable = {
            **base,
            "status": "unable_to_investigate",
            "reason": "source_sha256_mismatch",
            "actual_source_sha256": actual_hash,
        }
        return unable, {**unable, "groups": []}
    try:
        observed = load_observed_smf(source.path)
        performance = build_observed_performance(observed)
        notes_by_id = {note.note_on_event_id: note for note in performance.notes}
        groups: list[dict[str, Any]] = []
        for group_index, group in enumerate(performance.attack_groups):
            try:
                attack_pitches = tuple(
                    notes_by_id[event_id].pitch for event_id in group.note_on_event_ids
                )
            except KeyError as error:
                raise ValueError(
                    f"attack member is absent from matched notes: {error.args[0]}"
                ) from error
            compact: dict[str, Any] = {
                "group_index": group_index,
                "attack_pitch_class_counts": _pitch_class_counts(attack_pitches),
            }
            if performance.note_matching_status == "assessed":
                compact["key_held_pitch_class_counts"] = _pitch_class_counts(
                    tuple(
                        note.pitch
                        for note in performance.notes
                        if note.onset_us <= group.onset_us < note.offset_us
                    )
                )
            else:
                compact["key_held_pitch_class_counts"] = {
                    "status": "unable_to_investigate",
                    "reason": "note_matching_not_assessed",
                }
            groups.append(compact)
        source_record = {
            **base,
            "status": "assessed",
            "source_sha256": actual_hash,
            "ledger_sha256": observed.ledger_sha256,
            "note_matching_status": performance.note_matching_status,
            "ambiguous_note_count": performance.ambiguous_note_count,
            "unmatched_note_off_count": performance.unmatched_note_off_count,
            "dangling_note_on_count": performance.dangling_note_on_count,
            "attack_group_count": len(groups),
        }
        return source_record, {
            "case_id": source.case_id,
            "group_id": source.group_id,
            "status": "assessed",
            "source_sha256": actual_hash,
            "ledger_sha256": observed.ledger_sha256,
            "groups": groups,
        }
    except (OSError, ValueError) as error:
        unable = {
            **base,
            "status": "unable_to_investigate",
            "reason": str(error),
        }
        return unable, {**unable, "groups": []}


def run_observation_diagnostics(
    *,
    sources: Sequence[ObservationDiagnosticSource],
    output_dir: Path,
    split_role: str,
) -> dict[str, Any]:
    """固定入力をSMFだけから処理し、compactな支持根拠を保存する。"""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    if not sources:
        raise ValueError("observation sources must not be empty")
    case_ids = [source.case_id for source in sources]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("observation case IDs must be unique")
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = [_observe_source(source) for source in sorted(sources, key=lambda item: item.case_id)]
    source_records = [pair[0] for pair in pairs]
    support_records = [pair[1] for pair in pairs]
    assessed_count = sum(record["status"] == "assessed" for record in source_records)
    unable_count = len(source_records) - assessed_count
    result = {
        "schema_version": 1,
        "status": "pass" if unable_count == 0 else "partial",
        "split_role": split_role,
        "source_count": len(source_records),
        "assessed_source_count": assessed_count,
        "unable_to_investigate_source_count": unable_count,
        "attack_group_count": sum(
            int(record.get("attack_group_count", 0)) for record in source_records
        ),
        "hard_domain_status": "not_adopted",
        "static_one_or_more_support_is_monotonic": True,
        "support_histograms": _support_histograms(support_records),
    }
    run_spec = {
        "schema_version": 1,
        "split_role": split_role,
        "attack_group_window_us": 30_000,
        "harmony_qualities": list(HARMONY_QUALITIES),
        "low_register_limit": LOW_REGISTER_LIMIT,
        "support_kinds": ["attack_support", "key_held_support"],
        "pedal_sounding_status": "not_assessed",
        "sources": [
            {
                "case_id": source.case_id,
                "group_id": source.group_id,
                "source_name": source.path.name,
                "expected_sha256": source.expected_sha256,
            }
            for source in sorted(sources, key=lambda item: item.case_id)
        ],
    }
    atomic_write_bytes(output_dir / "observation-sources.jsonl", _jsonl_bytes(source_records))
    atomic_write_bytes(output_dir / "harmony-support.jsonl", _jsonl_bytes(support_records))
    atomic_write_json(output_dir / "observation-result.json", result)
    atomic_write_json(output_dir / "run-spec.json", run_spec)
    observation_names = (
        "observation-sources.jsonl",
        "harmony-support.jsonl",
        "observation-result.json",
        "run-spec.json",
    )
    observation_manifest = {
        "schema_version": 1,
        "status": result["status"],
        "outputs": {name: sha256_file(output_dir / name) for name in observation_names},
    }
    atomic_write_json(output_dir / "observation-manifest.json", observation_manifest)
    manifest_names = (*observation_names, "observation-manifest.json")
    manifest = {
        "schema_version": 1,
        "status": result["status"],
        "outputs": {name: sha256_file(output_dir / name) for name in manifest_names},
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return result


def _verify_observation_manifest(
    observation_dir: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    path = Path(observation_dir) / "observation-manifest.json"
    if sha256_file(path) != expected_sha256:
        raise ValueError("observation manifest SHA-256 mismatch")
    manifest = _read_json(path)
    for name, expected in manifest.get("outputs", {}).items():
        artifact = Path(observation_dir) / str(name)
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise ValueError(f"observation artifact SHA-256 mismatch: {name}")
    return manifest


def _expected_event_oracles(plan: object, score: ScoreSpec) -> dict[str, dict[str, Any]]:
    materials = {material.material_id: material for material in score.materials}
    result: dict[str, dict[str, Any]] = {}
    leaves, _ = ordered_leaf_schedule(plan, score)
    for leaf, _, _ in leaves:
        assert leaf.score_material_id is not None
        material = materials[leaf.score_material_id]
        for note in material.notes:
            harmonies = tuple(
                harmony
                for harmony in material.harmonies
                if harmony.at_units <= note.at_units < harmony.at_units + harmony.duration_units
            )
            if len(harmonies) > 1:
                raise ValueError(f"multiple harmonies cover score note: {note.event_id}")
            harmony = harmonies[0] if harmonies else None
            result[f"{leaf.node_id}:{note.event_id}"] = {
                "voice": note.voice,
                "foreground_voice": material.foreground_voice,
                "harmony": (
                    None
                    if harmony is None
                    else {
                        "root_pitch_class": harmony.root_pitch_class,
                        "quality": harmony.quality,
                    }
                ),
            }
    return result


def _counts_to_pitches(counts: Sequence[int]) -> tuple[int, ...]:
    if len(counts) != 12 or any(int(value) < 0 for value in counts):
        raise ValueError("pitch-class counts must contain 12 non-negative integers")
    return tuple(
        pitch_class
        for pitch_class, count in enumerate(counts)
        for _ in range(int(count))
    )


def _support_comparison(
    counts: Sequence[int],
    *,
    root_pitch_class: int,
    quality: str,
) -> dict[str, Any]:
    pitches = _counts_to_pitches(counts)
    scores = [
        int(
            harmony_support(
                pitches,
                root_pitch_class=root,
                quality=item_quality,
            )["chord_tone_count"]
        )
        for root, item_quality in harmony_keys()
    ]
    oracle_index = harmony_keys().index((int(root_pitch_class), str(quality)))
    oracle_score = scores[oracle_index]
    maximum = max(scores)
    return {
        "oracle_chord_tone_count": oracle_score,
        "maximum_chord_tone_count": maximum,
        "greater_harmony_count": sum(score > oracle_score for score in scores),
        "equal_harmony_count": sum(score == oracle_score for score in scores),
        "oracle_is_unique_maximum": oracle_score == maximum and scores.count(maximum) == 1,
        "oracle_is_tied_maximum": oracle_score == maximum and scores.count(maximum) > 1,
    }


def _annotate_known_source(
    source: KnownTimingSource,
    *,
    support_record: Mapping[str, Any],
) -> dict[str, Any]:
    base = {
        "case_id": source.case_id,
        "group_id": source.group_id,
        "oracle_basis": source.oracle_basis,
        "oracle_observation_dependency": source.oracle_observation_dependency,
        "positive_evidence_eligible": source.oracle_observation_dependency == "absent",
    }
    required = (source.piece_path, source.score_path, source.performance_path, source.smf_path)
    if any(not Path(path).is_file() for path in required):
        return {**base, "status": "unable_to_investigate", "reason": "known_IR_missing"}
    try:
        plan = parse_piece_plan(source.piece_path.read_text(encoding="utf-8"))
        score = parse_score_spec(source.score_path.read_text(encoding="utf-8"))
        parse_performance_spec(source.performance_path.read_text(encoding="utf-8"))
        if not any(material.harmonies for material in score.materials):
            return {**base, "status": "oracle_not_declared", "reason": "harmony_not_declared"}
        observed = load_observed_smf(source.smf_path)
        performance = build_observed_performance(observed)
        alignment = align_known_score_note_events(plan, score, performance)
        if alignment.status != "assessed":
            return {
                **base,
                "status": "unable_to_investigate",
                "reason": "known_score_note_alignment_not_assessed",
            }
        if support_record.get("source_sha256") != observed.source_sha256:
            return {
                **base,
                "status": "unable_to_investigate",
                "reason": "observation_source_sha256_mismatch",
            }
        compact_groups = list(support_record.get("groups", []))
        if len(compact_groups) != len(performance.attack_groups):
            return {
                **base,
                "status": "unable_to_investigate",
                "reason": "observation_group_count_mismatch",
            }
        expected_oracles = _expected_event_oracles(plan, score)
        oracle_by_observed = {
            match.observed.note_on_event_id: expected_oracles[match.expected.event_id]
            for match in alignment.matches
        }
        group_annotations: list[dict[str, Any]] = []
        accompaniment_chord_tones = 0
        accompaniment_total = 0
        multiple_harmony_group_count = 0
        for group_index, (group, compact) in enumerate(
            zip(performance.attack_groups, compact_groups, strict=True)
        ):
            members = [oracle_by_observed[event_id] for event_id in group.note_on_event_ids]
            declared = {
                (int(item["harmony"]["root_pitch_class"]), str(item["harmony"]["quality"]))
                for item in members
                if item["harmony"] is not None
            }
            if not declared:
                continue
            if len(declared) != 1:
                multiple_harmony_group_count += 1
                continue
            root_pitch_class, quality = next(iter(declared))
            attack_comparison = _support_comparison(
                compact["attack_pitch_class_counts"],
                root_pitch_class=root_pitch_class,
                quality=quality,
            )
            key_held_counts = compact["key_held_pitch_class_counts"]
            key_held_comparison = (
                _support_comparison(
                    key_held_counts,
                    root_pitch_class=root_pitch_class,
                    quality=quality,
                )
                if isinstance(key_held_counts, list)
                else key_held_counts
            )
            chord = harmony_pitch_classes(root_pitch_class, quality)
            for event_id in group.note_on_event_ids:
                item = oracle_by_observed[event_id]
                foreground_voice = item["foreground_voice"]
                if foreground_voice is None or item["voice"] == foreground_voice:
                    continue
                accompaniment_total += 1
                accompaniment_chord_tones += int(
                    next(
                        note.pitch
                        for note in performance.notes
                        if note.note_on_event_id == event_id
                    )
                    % 12
                    in chord
                )
            group_annotations.append(
                {
                    "group_index": group_index,
                    "declared_harmony": {
                        "root_pitch_class": root_pitch_class,
                        "quality": quality,
                    },
                    "attack_support": attack_comparison,
                    "key_held_support": key_held_comparison,
                }
            )
        if not group_annotations:
            return {
                **base,
                "status": "unable_to_investigate",
                "reason": "no_comparable_declared_harmony_group",
                "multiple_declared_harmony_group_count": multiple_harmony_group_count,
            }
        return {
            **base,
            "status": "assessed",
            "declared_harmony_group_count": len(group_annotations),
            "multiple_declared_harmony_group_count": multiple_harmony_group_count,
            "attack_oracle_unique_maximum_count": sum(
                item["attack_support"]["oracle_is_unique_maximum"]
                for item in group_annotations
            ),
            "attack_oracle_tied_maximum_count": sum(
                item["attack_support"]["oracle_is_tied_maximum"]
                for item in group_annotations
            ),
            "key_held_oracle_unique_maximum_count": sum(
                isinstance(item["key_held_support"], dict)
                and item["key_held_support"].get("oracle_is_unique_maximum", False)
                for item in group_annotations
            ),
            "key_held_oracle_tied_maximum_count": sum(
                isinstance(item["key_held_support"], dict)
                and item["key_held_support"].get("oracle_is_tied_maximum", False)
                for item in group_annotations
            ),
            "accompaniment_chord_tone_rate": _fraction_record(
                accompaniment_chord_tones, accompaniment_total
            ),
            "groups": group_annotations,
        }
    except (KeyError, OSError, UnicodeDecodeError, ValueError) as error:
        return {**base, "status": "unable_to_investigate", "reason": str(error)}


def run_known_harmony_oracle(
    *,
    sources: Sequence[KnownTimingSource],
    observation_dir: Path,
    expected_observation_manifest_sha256: str,
) -> dict[str, Any]:
    """凍結済み観測成果物へだけ既知ScoreHarmonyを注釈する。"""
    observation_dir = Path(observation_dir)
    _verify_observation_manifest(
        observation_dir,
        expected_sha256=expected_observation_manifest_sha256,
    )
    support_by_case = {
        str(record["case_id"]): record
        for record in _read_jsonl(observation_dir / "harmony-support.jsonl")
    }
    annotations = [
        _annotate_known_source(source, support_record=support_by_case.get(source.case_id, {}))
        for source in sorted(sources, key=lambda item: item.case_id)
    ]
    assessed_count = sum(item["status"] == "assessed" for item in annotations)
    not_declared_count = sum(item["status"] == "oracle_not_declared" for item in annotations)
    unable_count = sum(item["status"] == "unable_to_investigate" for item in annotations)
    result = {
        "schema_version": 1,
        "status": "pass" if unable_count == 0 and assessed_count > 0 else "partial",
        "source_count": len(annotations),
        "assessed_source_count": assessed_count,
        "oracle_not_declared_source_count": not_declared_count,
        "unable_to_investigate_source_count": unable_count,
        "positive_evidence_eligible_assessed_source_count": sum(
            item["status"] == "assessed" and item["positive_evidence_eligible"]
            for item in annotations
        ),
        "declared_harmony_group_count": sum(
            int(item.get("declared_harmony_group_count", 0)) for item in annotations
        ),
        "attack_oracle_unique_maximum_count": sum(
            int(item.get("attack_oracle_unique_maximum_count", 0)) for item in annotations
        ),
        "attack_oracle_tied_maximum_count": sum(
            int(item.get("attack_oracle_tied_maximum_count", 0)) for item in annotations
        ),
        "key_held_oracle_unique_maximum_count": sum(
            int(item.get("key_held_oracle_unique_maximum_count", 0)) for item in annotations
        ),
        "key_held_oracle_tied_maximum_count": sum(
            int(item.get("key_held_oracle_tied_maximum_count", 0)) for item in annotations
        ),
        "hard_domain_status": "not_adopted",
    }
    atomic_write_bytes(
        observation_dir / "oracle-annotations.jsonl",
        _jsonl_bytes(annotations),
    )
    atomic_write_json(observation_dir / "oracle-result.json", result)
    manifest = _read_json(observation_dir / "manifest.json")
    manifest["status"] = result["status"]
    manifest["observation_manifest_sha256"] = expected_observation_manifest_sha256
    manifest["outputs"]["oracle-annotations.jsonl"] = sha256_file(
        observation_dir / "oracle-annotations.jsonl"
    )
    manifest["outputs"]["oracle-result.json"] = sha256_file(
        observation_dir / "oracle-result.json"
    )
    atomic_write_json(observation_dir / "manifest.json", manifest)
    return result


def observation_sources_from_known(
    sources: Sequence[KnownTimingSource],
) -> tuple[ObservationDiagnosticSource, ...]:
    """既知生成元を、IRを含まない観測入力へ変換する。"""
    return tuple(
        ObservationDiagnosticSource(
            case_id=source.case_id,
            group_id=source.group_id,
            path=source.smf_path,
            expected_sha256=(
                sha256_file(source.smf_path) if source.smf_path.is_file() else "missing"
            ),
        )
        for source in sorted(sources, key=lambda item: item.case_id)
    )


def default_known_harmony_sources(workspace: Path = Path(".")) -> tuple[KnownTimingSource, ...]:
    """保存済み6件と和声未宣言の人工4件を返す。"""
    from llm_musical_composer.reference_timing_run import default_known_timing_sources
    from llm_musical_composer.score_timing_known_fixtures import build_known_timing_fixtures

    workspace = Path(workspace)
    existing = default_known_timing_sources(workspace)
    fixtures = build_known_timing_fixtures(
        workspace / ".appendix/score-timing-known-fixtures-v1"
    )
    return (*existing, *(fixture.source for fixture in fixtures))


def reference_observation_sources(
    root: Path,
) -> tuple[ObservationDiagnosticSource, ...]:
    """検証済みroundtrip SMFを参照全件の観測入力へ変換する。"""
    return tuple(
        ObservationDiagnosticSource(
            case_id=f"reference-{source.name}",
            group_id="reference-corpus-unlabeled",
            path=source.path,
            expected_sha256=source.sha256,
        )
        for source in verify_roundtrip_sources(root=Path(root))
    )


def run_default_known_diagnostics(
    *,
    workspace: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """既知入力を観測、凍結、oracle注釈の順で処理する。"""
    sources = default_known_harmony_sources(workspace)
    observation = run_observation_diagnostics(
        sources=observation_sources_from_known(sources),
        output_dir=output_dir,
        split_role="known_development",
    )
    observation_manifest_sha256 = sha256_file(
        Path(output_dir) / "observation-manifest.json"
    )
    oracle = run_known_harmony_oracle(
        sources=sources,
        observation_dir=output_dir,
        expected_observation_manifest_sha256=observation_manifest_sha256,
    )
    return {"observation": observation, "oracle": oracle}


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    known = subparsers.add_parser("known")
    known.add_argument("--workspace", type=Path, default=Path("."))
    known.add_argument("--output-dir", type=Path, required=True)
    reference = subparsers.add_parser("reference")
    reference.add_argument("--input-root", type=Path, required=True)
    reference.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    if arguments.command == "known":
        result = run_default_known_diagnostics(
            workspace=arguments.workspace,
            output_dir=arguments.output_dir,
        )
        status = result["oracle"]["status"]
    else:
        result = run_observation_diagnostics(
            sources=reference_observation_sources(arguments.input_root),
            output_dir=arguments.output_dir,
            split_role="reference_corpus_unlabeled",
        )
        status = result["status"]
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
