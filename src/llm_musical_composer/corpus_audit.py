"""参照SMF群を読み取り専用で診断する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import struct
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mido

from llm_musical_composer.smf_notes import (
    collect_events as _collect_events,
)
from llm_musical_composer.smf_notes import (
    match_notes as _match_notes,
)
from llm_musical_composer.smf_notes import (
    maximum_polyphony as _maximum_polyphony,
)

FINGERPRINT_VERSION = 1
FINGERPRINT_GRID_PER_QUARTER = 24
FINGERPRINT_NGRAM_SIZE = 4
TOP_CANDIDATES_PER_FILE = 5


class InvalidSmfStructureError(ValueError):
    """SMFのバイト構造を解釈できない場合の例外。"""


class UnsupportedTimingError(ValueError):
    """この段階で拍単位の集計ができない時間形式。"""


@dataclass(frozen=True)
class RawInspection:
    header: dict[str, Any]
    issues: tuple[dict[str, Any], ...]
    track_chunks: tuple[bytes, ...]
    trailing_bytes: int


def _issue(code: str, message: str, *, offset: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"code": code, "message": message}
    if offset is not None:
        result["offset"] = offset
    return result


def inspect_smf_bytes(data: bytes) -> RawInspection:
    """Midoで正規化される前のSMFチャンク構造を検査する。"""
    if len(data) < 8:
        raise InvalidSmfStructureError("File is shorter than an SMF chunk header.")
    if data[:4] != b"MThd":
        raise InvalidSmfStructureError("The first chunk is not MThd.")

    header_length = struct.unpack(">I", data[4:8])[0]
    if header_length < 6:
        raise InvalidSmfStructureError(f"MThd length is shorter than 6: {header_length}")
    if len(data) < 8 + header_length:
        raise InvalidSmfStructureError("MThd data is truncated.")

    smf_format, declared_tracks, division = struct.unpack(">HHH", data[8:14])
    issues: list[dict[str, Any]] = []
    if header_length != 6:
        issues.append(
            _issue(
                "noncanonical_header_length",
                f"MThd length is {header_length}; the canonical length is 6.",
                offset=4,
            )
        )
    if smf_format not in (0, 1, 2):
        raise InvalidSmfStructureError(f"Unsupported SMF format value: {smf_format}")
    if declared_tracks == 0:
        issues.append(_issue("zero_declared_tracks", "The header declares zero tracks."))
    if smf_format == 0 and declared_tracks != 1:
        issues.append(
            _issue(
                "format_track_count_mismatch",
                f"Format 0 declares {declared_tracks} tracks instead of one.",
            )
        )

    if division & 0x8000:
        frames_raw = (division >> 8) & 0xFF
        frames_per_second = 256 - frames_raw
        ticks_per_frame = division & 0xFF
        timing_kind = "smpte"
        if frames_per_second not in (24, 25, 29, 30) or ticks_per_frame == 0:
            issues.append(
                _issue(
                    "invalid_smpte_division",
                    (
                        f"SMPTE division has rate {frames_per_second} "
                        f"and {ticks_per_frame} ticks/frame."
                    ),
                )
            )
        header = {
            "signature": "MThd",
            "header_length": header_length,
            "format": smf_format,
            "declared_tracks": declared_tracks,
            "timing_kind": timing_kind,
            "frames_per_second": frames_per_second,
            "ticks_per_frame": ticks_per_frame,
        }
    else:
        timing_kind = "ppqn"
        if division == 0:
            issues.append(_issue("invalid_ppqn", "PPQN division is zero."))
        header = {
            "signature": "MThd",
            "header_length": header_length,
            "format": smf_format,
            "declared_tracks": declared_tracks,
            "timing_kind": timing_kind,
            "ticks_per_beat": division,
        }

    offset = 8 + header_length
    track_chunks: list[bytes] = []
    for track_index in range(declared_tracks):
        if offset + 8 > len(data):
            raise InvalidSmfStructureError(
                f"Track {track_index} chunk header is missing or truncated at byte {offset}."
            )
        if data[offset : offset + 4] != b"MTrk":
            raise InvalidSmfStructureError(
                f"Track {track_index} does not start with MTrk at byte {offset}."
            )
        track_length = struct.unpack(">I", data[offset + 4 : offset + 8])[0]
        track_start = offset + 8
        track_end = track_start + track_length
        if track_end > len(data):
            raise InvalidSmfStructureError(
                f"Track {track_index} declares {track_length} bytes beyond end of file."
            )
        track_chunks.append(data[track_start:track_end])
        offset = track_end

    trailing_bytes = len(data) - offset
    if trailing_bytes:
        issues.append(
            _issue(
                "trailing_data",
                f"{trailing_bytes} byte(s) follow the declared track chunks.",
                offset=offset,
            )
        )
    return RawInspection(header, tuple(issues), tuple(track_chunks), trailing_bytes)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def _map_messages(events: list[dict[str, Any]], message_type: str) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for event in events:
        message = event["message"]
        if message.type != message_type:
            continue
        base: dict[str, Any] = {
            "tick": event["tick"],
            "track": event["track"],
            "event_index": event["event_index"],
        }
        if message_type == "set_tempo":
            base["microseconds_per_beat"] = message.tempo
            base["bpm"] = _round(mido.tempo2bpm(message.tempo), 3)
        elif message_type == "time_signature":
            base.update(
                numerator=message.numerator,
                denominator=message.denominator,
                clocks_per_click=message.clocks_per_click,
                notated_32nd_notes_per_beat=message.notated_32nd_notes_per_beat,
            )
        elif message_type == "key_signature":
            base["key"] = message.key
        mapped.append(base)
    return mapped


def _duration_seconds(midi: mido.MidiFile) -> float:
    tempo = 500_000
    seconds = 0.0
    for message in mido.merge_tracks(midi.tracks):
        seconds += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == "set_tempo":
            tempo = message.tempo
    return seconds


def _estimated_measures(
    final_tick: int, ticks_per_beat: int, time_signatures: list[dict[str, Any]]
) -> float:
    changes = sorted(time_signatures, key=lambda item: item["tick"])
    if not changes or changes[0]["tick"] != 0:
        changes.insert(0, {"tick": 0, "numerator": 4, "denominator": 4})
    total = 0.0
    for index, current in enumerate(changes):
        start = min(current["tick"], final_tick)
        end = final_tick
        if index + 1 < len(changes):
            end = min(changes[index + 1]["tick"], final_tick)
        if end <= start:
            continue
        quarter_notes = (end - start) / ticks_per_beat
        quarter_notes_per_measure = current["numerator"] * 4 / current["denominator"]
        total += quarter_notes / quarter_notes_per_measure
    return total


def _grid_offset(tick: int, ticks_per_beat: int, grid: int) -> float:
    scaled = tick * grid / ticks_per_beat
    return abs(scaled - round(scaled)) / grid


def build_musical_fingerprint(
    notes: Iterable[dict[str, Any]], *, ticks_per_beat: int
) -> dict[str, Any]:
    """移調、テンポメタイベント、PPQN差に不変な説明可能なfingerprintを作る。"""
    pitched = sorted(
        (note for note in notes if note.get("channel") != 9),
        key=lambda note: (note["onset_tick"], note["pitch"], note.get("track", 0)),
    )
    tokens: list[str] = []
    previous_pitch: int | None = None
    previous_onset: int | None = None
    for note in pitched:
        pitch_delta = 0 if previous_pitch is None else note["pitch"] - previous_pitch
        onset_delta = (
            0
            if previous_onset is None
            else round(
                (note["onset_tick"] - previous_onset)
                * FINGERPRINT_GRID_PER_QUARTER
                / ticks_per_beat
            )
        )
        raw_duration = note.get("duration_ticks")
        duration = (
            -1
            if raw_duration is None
            else round(raw_duration * FINGERPRINT_GRID_PER_QUARTER / ticks_per_beat)
        )
        tokens.append(f"{pitch_delta}:{onset_delta}:{duration}")
        previous_pitch = note["pitch"]
        previous_onset = note["onset_tick"]

    encoded = "|".join(tokens).encode("utf-8")
    ngram_size = min(FINGERPRINT_NGRAM_SIZE, len(tokens))
    ngrams: list[str] = []
    if ngram_size:
        for index in range(len(tokens) - ngram_size + 1):
            chunk = "|".join(tokens[index : index + ngram_size]).encode("utf-8")
            ngrams.append(hashlib.sha256(chunk).hexdigest()[:16])
    return {
        "version": FINGERPRINT_VERSION,
        "grid_per_quarter": FINGERPRINT_GRID_PER_QUARTER,
        "ngram_size": ngram_size,
        "excluded_channels": [9],
        "token_count": len(tokens),
        "sha256": hashlib.sha256(encoded).hexdigest().upper(),
        "ngrams": sorted(set(ngrams)),
    }


def fingerprint_similarity(first: Iterable[str], second: Iterable[str]) -> float:
    """fingerprint n-gram集合のJaccard類似度を返す。"""
    first_set = set(first)
    second_set = set(second)
    if not first_set and not second_set:
        return 1.0
    union = first_set | second_set
    if not union:
        return 0.0
    return len(first_set & second_set) / len(union)


def _failed_result(path: Path, data: bytes, error: Exception, header: Any = None) -> dict[str, Any]:
    return {
        "name": path.name,
        "relative_path": path.name,
        "extension": path.suffix.lower(),
        "bytes": len(data),
        "sha256": _sha256_bytes(data),
        "status": "unable_to_investigate",
        "source_provenance_status": "content_found_source_unproven",
        "header": header,
        "conformance": None,
        "structure": None,
        "performance": None,
        "maps": None,
        "anomalies": None,
        "fingerprint": None,
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def audit_file(path: Path) -> dict[str, Any]:
    """ひとつのSMFを診断し、失敗を0件へ変換せず返す。"""
    path = Path(path)
    data = path.read_bytes()
    try:
        raw = inspect_smf_bytes(data)
    except Exception as error:
        return _failed_result(path, data, error)

    if raw.header["timing_kind"] == "smpte":
        return _failed_result(
            path,
            data,
            UnsupportedTimingError("SMPTE timing is not evaluated in beat-based metrics."),
            raw.header,
        )
    if raw.header.get("ticks_per_beat") == 0:
        return _failed_result(
            path,
            data,
            InvalidSmfStructureError("PPQN division is zero."),
            raw.header,
        )

    try:
        midi = mido.MidiFile(filename=str(path), clip=False)
        events, track_reports = _collect_events(midi)
    except Exception as error:
        return _failed_result(path, data, error, raw.header)

    conformance_issues = list(raw.issues)
    for track_report in track_reports:
        conformance_issues.extend(
            {**issue, "track": track_report["track"]} for issue in track_report["issues"]
        )
    if midi.type != raw.header["format"]:
        conformance_issues.append(
            _issue("parser_header_mismatch", "Mido and byte-level format values differ.")
        )
    if len(midi.tracks) != raw.header["declared_tracks"]:
        conformance_issues.append(
            _issue("parser_track_count_mismatch", "Mido and header track counts differ.")
        )

    final_tick = max((track["last_tick"] for track in track_reports), default=0)
    notes, note_anomalies = _match_notes(events, final_tick)
    pitched_notes = [note for note in notes if note["channel"] != 9]
    velocities = [note["velocity"] for note in notes]
    pitches = [note["pitch"] for note in pitched_notes]
    event_counts = Counter(event["message"].type for event in events)
    channels = sorted(
        {event["message"].channel for event in events if hasattr(event["message"], "channel")}
    )
    programs = [
        {
            "tick": event["tick"],
            "track": event["track"],
            "channel": event["message"].channel,
            "program": event["message"].program,
        }
        for event in events
        if event["message"].type == "program_change"
    ]
    controls = [event["message"] for event in events if event["message"].type == "control_change"]
    sustain_events = [message for message in controls if message.control == 64]
    other_controls = [message for message in controls if message.control != 64]
    time_signatures = _map_messages(events, "time_signature")
    tempos = _map_messages(events, "set_tempo")
    key_signatures = _map_messages(events, "key_signature")

    if midi.type == 2:
        duration_seconds = None
        estimated_measures = None
        density = None
        conformance_issues.append(
            _issue(
                "format_2_global_metrics_not_applicable",
                (
                    "Format 2 tracks are asynchronous; global duration and measure metrics "
                    "are omitted."
                ),
            )
        )
    else:
        duration_seconds = _duration_seconds(midi)
        estimated_measures = _estimated_measures(final_tick, midi.ticks_per_beat, time_signatures)
        quarter_notes = final_tick / midi.ticks_per_beat
        density = len(notes) / quarter_notes if quarter_notes > 0 else None

    offsets_16 = [_grid_offset(note["onset_tick"], midi.ticks_per_beat, 4) for note in notes]
    offsets_12 = [_grid_offset(note["onset_tick"], midi.ticks_per_beat, 3) for note in notes]
    fingerprint = build_musical_fingerprint(notes, ticks_per_beat=midi.ticks_per_beat)
    anomaly_result: dict[str, Any] = {
        **note_anomalies,
        "empty_song": not notes,
        "unterminated_sustain": any(message.value >= 64 for message in sustain_events[-1:]),
    }

    return {
        "name": path.name,
        "relative_path": path.name,
        "extension": path.suffix.lower(),
        "bytes": len(data),
        "sha256": _sha256_bytes(data),
        "status": "affirmative_evidence",
        "source_provenance_status": "content_found_source_unproven",
        "header": raw.header,
        "conformance": {
            "parser_success": True,
            "is_structurally_conformant": not conformance_issues,
            "issues": conformance_issues,
            "trailing_bytes": raw.trailing_bytes,
        },
        "structure": {
            "parsed_tracks": len(midi.tracks),
            "track_reports": track_reports,
            "channels": channels,
            "program_changes": programs,
            "event_counts": dict(sorted(event_counts.items())),
            "final_tick": final_tick,
            "duration_seconds": _round(duration_seconds),
            "estimated_measures": _round(estimated_measures),
            "note_count": len(notes),
            "pitched_note_count": len(pitched_notes),
            "drum_note_count": len(notes) - len(pitched_notes),
            "pitch_min": min(pitches) if pitches else None,
            "pitch_max": max(pitches) if pitches else None,
            "max_polyphony": _maximum_polyphony(notes, final_tick),
            "max_pitched_polyphony": _maximum_polyphony(pitched_notes, final_tick),
            "notes_per_quarter": _round(density),
        },
        "performance": {
            "velocity_min": min(velocities) if velocities else None,
            "velocity_max": max(velocities) if velocities else None,
            "velocity_mean": _round(statistics.mean(velocities)) if velocities else None,
            "velocity_standard_deviation": (
                _round(statistics.pstdev(velocities)) if velocities else None
            ),
            "velocity_unique": len(set(velocities)),
            "sustain_event_count": len(sustain_events),
            "other_control_change_count": len(other_controls),
            "control_numbers": dict(
                sorted(Counter(message.control for message in controls).items())
            ),
            "pitchwheel_event_count": event_counts.get("pitchwheel", 0),
            "aftertouch_event_count": event_counts.get("aftertouch", 0)
            + event_counts.get("polytouch", 0),
            "sysex_event_count": event_counts.get("sysex", 0),
            "onset_offset_from_1_16_mean_quarters": (
                _round(statistics.mean(offsets_16)) if offsets_16 else None
            ),
            "onset_offset_from_triplet_1_8_mean_quarters": (
                _round(statistics.mean(offsets_12)) if offsets_12 else None
            ),
        },
        "maps": {
            "tempos": tempos,
            "tempo_default_assumed": not tempos,
            "time_signatures": time_signatures,
            "time_signature_default_assumed": not time_signatures,
            "key_signatures": key_signatures,
        },
        "anomalies": anomaly_result,
        "fingerprint": fingerprint,
        "error": None,
    }


def _filename_series(name: str) -> str | None:
    stem = Path(name).stem.casefold()
    numbered_movement = re.match(r"^(.+?)第\d+曲", stem)
    if numbered_movement:
        return numbered_movement.group(1)
    trailing = re.sub(r"[-_ ]?\d+$", "", stem)
    if trailing != stem and trailing:
        return trailing
    match = re.match(r"^(.+?)[-_ ]\d+(?:[-_ ].*)?$", stem)
    if match:
        return match.group(1)
    return None


def _groups_by_value(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        value = record.get(key)
        if value:
            grouped[value].append(record["name"])
    return [
        {"value": value, "files": sorted(files)}
        for value, files in sorted(grouped.items())
        if len(files) > 1
    ]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _numeric_summary(values: Iterable[int | float | None]) -> dict[str, Any]:
    available = sorted(float(value) for value in values if value is not None)
    if not available:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(available),
        "min": _round(available[0]),
        "p25": _round(_percentile(available, 0.25)),
        "median": _round(_percentile(available, 0.5)),
        "p75": _round(_percentile(available, 0.75)),
        "p95": _round(_percentile(available, 0.95)),
        "max": _round(available[-1]),
        "mean": _round(statistics.mean(available)),
    }


def audit_corpus(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    """ディレクトリ直下を全件診断し、決定的な構造化出力を保存する。"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for path in sorted(input_dir.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file():
            records.append(
                {
                    "name": path.name,
                    "relative_path": path.name,
                    "extension": "",
                    "status": "unable_to_investigate",
                    "error": {"type": "UnexpectedDirectory", "message": "Entry is a directory."},
                }
            )
            continue
        if path.suffix.lower() != ".mid":
            data = path.read_bytes()
            records.append(
                {
                    "name": path.name,
                    "relative_path": path.name,
                    "extension": path.suffix.lower(),
                    "bytes": len(data),
                    "sha256": _sha256_bytes(data),
                    "status": "excluded_non_midi",
                    "source_provenance_status": "content_found_source_unproven",
                    "container_detected": "zip" if data.startswith(b"PK\x03\x04") else None,
                    "error": None,
                }
            )
            continue
        records.append(audit_file(path))

    successful = [record for record in records if record["status"] == "affirmative_evidence"]
    exact_groups = _groups_by_value(successful, "sha256")
    fingerprint_records = [
        {**record, "musical_sha256": record["fingerprint"]["sha256"]}
        for record in successful
        if record.get("fingerprint") and record["fingerprint"]["token_count"]
    ]
    musical_groups = _groups_by_value(fingerprint_records, "musical_sha256")

    pairwise_evidence: list[dict[str, Any]] = []
    candidates_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    compared_pair_count = 0
    for first_index, first in enumerate(fingerprint_records):
        for second in fingerprint_records[first_index + 1 :]:
            compared_pair_count += 1
            similarity = fingerprint_similarity(
                first["fingerprint"]["ngrams"], second["fingerprint"]["ngrams"]
            )
            evidence = {
                "first": first["name"],
                "second": second["name"],
                "jaccard": _round(similarity),
                "status": (
                    "content_found_source_unproven"
                    if similarity > 0
                    else "investigated_no_evidence"
                ),
            }
            if similarity > 0:
                pairwise_evidence.append(evidence)
            candidates_by_file[first["name"]].append(
                {"file": second["name"], "jaccard": evidence["jaccard"]}
            )
            candidates_by_file[second["name"]].append(
                {"file": first["name"], "jaccard": evidence["jaccard"]}
            )
    pairwise_evidence.sort(key=lambda item: (-item["jaccard"], item["first"], item["second"]))
    top_candidates = {
        name: sorted(values, key=lambda item: (-item["jaccard"], item["file"]))[
            :TOP_CANDIDATES_PER_FILE
        ]
        for name, values in sorted(candidates_by_file.items())
    }

    filename_series: dict[str, list[str]] = defaultdict(list)
    for record in records:
        series = _filename_series(record["name"])
        if series:
            filename_series[series].append(record["name"])
    filename_groups = [
        {"series": series, "files": sorted(files)}
        for series, files in sorted(filename_series.items())
        if len(files) > 1
    ]

    evaluation_groups = {
        "filename_series": filename_groups,
        "track_configuration": {
            "single_track": sorted(
                record["name"] for record in successful if record["structure"]["parsed_tracks"] == 1
            ),
            "multi_track": sorted(
                record["name"] for record in successful if record["structure"]["parsed_tracks"] > 1
            ),
        },
        "timing": {
            "tempo_changes": sorted(
                record["name"] for record in successful if len(record["maps"]["tempos"]) > 1
            ),
            "time_signature_changes": sorted(
                record["name"]
                for record in successful
                if len(record["maps"]["time_signatures"]) > 1
            ),
        },
        "performance_evidence": {
            "velocity_variation": sorted(
                record["name"]
                for record in successful
                if record["performance"]["velocity_unique"] > 1
            ),
            "sustain": sorted(
                record["name"]
                for record in successful
                if record["performance"]["sustain_event_count"] > 0
            ),
            "timing_offset_from_1_16": sorted(
                record["name"]
                for record in successful
                if (record["performance"]["onset_offset_from_1_16_mean_quarters"] or 0) > 0
            ),
        },
        "not_evidence_of_human_performance": (
            "Velocity variation, sustain, and grid offsets are observations, "
            "not proof of human performance."
        ),
    }

    event_type_totals: Counter[str] = Counter()
    control_number_totals: Counter[int] = Counter()
    anomaly_totals: Counter[str] = Counter()
    for record in successful:
        event_type_totals.update(record["structure"]["event_counts"])
        control_number_totals.update(
            {int(key): value for key, value in record["performance"]["control_numbers"].items()}
        )
        for name, value in record["anomalies"].items():
            if isinstance(value, bool):
                anomaly_totals[name] += int(value)
            elif isinstance(value, int):
                anomaly_totals[name] += value

    summary = {
        "schema_version": 1,
        "input_file_count": len(records),
        "midi_file_count": sum(record.get("extension") == ".mid" for record in records),
        "affirmative_evidence_count": len(successful),
        "unable_to_investigate_count": sum(
            record["status"] == "unable_to_investigate" for record in records
        ),
        "excluded_non_midi_count": sum(
            record["status"] == "excluded_non_midi" for record in records
        ),
        "structurally_conformant_count": sum(
            bool(record.get("conformance", {}).get("is_structurally_conformant"))
            for record in successful
        ),
        "exact_duplicate_group_count": len(exact_groups),
        "musical_fingerprint_group_count": len(musical_groups),
        "pairwise_compared_count": compared_pair_count,
        "pairwise_positive_evidence_count": len(pairwise_evidence),
        "distributions": {
            "formats": dict(
                sorted(Counter(str(record["header"]["format"]) for record in successful).items())
            ),
            "ticks_per_beat": dict(
                sorted(
                    Counter(
                        str(record["header"]["ticks_per_beat"]) for record in successful
                    ).items()
                )
            ),
            "parsed_tracks": dict(
                sorted(
                    Counter(
                        str(record["structure"]["parsed_tracks"]) for record in successful
                    ).items()
                )
            ),
            "note_count": _numeric_summary(
                record["structure"]["note_count"] for record in successful
            ),
            "duration_seconds": _numeric_summary(
                record["structure"]["duration_seconds"] for record in successful
            ),
            "estimated_measures": _numeric_summary(
                record["structure"]["estimated_measures"] for record in successful
            ),
            "notes_per_quarter": _numeric_summary(
                record["structure"]["notes_per_quarter"] for record in successful
            ),
            "max_polyphony": _numeric_summary(
                record["structure"]["max_polyphony"] for record in successful
            ),
        },
        "feature_file_counts": {
            "explicit_tempo": sum(bool(record["maps"]["tempos"]) for record in successful),
            "tempo_default_assumed": sum(
                record["maps"]["tempo_default_assumed"] for record in successful
            ),
            "explicit_time_signature": sum(
                bool(record["maps"]["time_signatures"]) for record in successful
            ),
            "time_signature_default_assumed": sum(
                record["maps"]["time_signature_default_assumed"] for record in successful
            ),
            "explicit_program_change": sum(
                bool(record["structure"]["program_changes"]) for record in successful
            ),
            "multiple_tempos": sum(len(record["maps"]["tempos"]) > 1 for record in successful),
            "multiple_time_signatures": sum(
                len(record["maps"]["time_signatures"]) > 1 for record in successful
            ),
            "key_signature_present": sum(
                bool(record["maps"]["key_signatures"]) for record in successful
            ),
            "velocity_variation": sum(
                record["performance"]["velocity_unique"] > 1 for record in successful
            ),
            "sustain_present": sum(
                record["performance"]["sustain_event_count"] > 0 for record in successful
            ),
            "other_control_change_present": sum(
                record["performance"]["other_control_change_count"] > 0 for record in successful
            ),
            "pitchwheel_present": sum(
                record["performance"]["pitchwheel_event_count"] > 0 for record in successful
            ),
            "aftertouch_present": sum(
                record["performance"]["aftertouch_event_count"] > 0 for record in successful
            ),
            "sysex_present": sum(
                record["performance"]["sysex_event_count"] > 0 for record in successful
            ),
            "onset_offset_from_1_16_present": sum(
                (record["performance"]["onset_offset_from_1_16_mean_quarters"] or 0) > 0
                for record in successful
            ),
        },
        "event_type_totals": dict(sorted(event_type_totals.items())),
        "channel_file_counts": {
            str(channel): sum(channel in record["structure"]["channels"] for record in successful)
            for channel in sorted(
                {channel for record in successful for channel in record["structure"]["channels"]}
            )
        },
        "control_number_totals": {
            str(key): value for key, value in sorted(control_number_totals.items())
        },
        "anomaly_totals": dict(sorted(anomaly_totals.items())),
    }
    duplicate_candidates = {
        "fingerprint_version": FINGERPRINT_VERSION,
        "warning": (
            "Similarity values are review evidence only. "
            "No threshold proves a common work or source."
        ),
        "exact_groups": [
            {"sha256": group["value"], "files": group["files"]} for group in exact_groups
        ],
        "musical_fingerprint_groups": [
            {"sha256": group["value"], "files": group["files"]} for group in musical_groups
        ],
        "pairwise_positive_evidence": pairwise_evidence,
        "top_candidates_per_file": top_candidates,
    }

    files_path = output_dir / "files.jsonl"
    files_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
        ),
        encoding="utf-8",
    )
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "duplicate-candidates.json", duplicate_candidates)
    _write_json(output_dir / "evaluation-groups.json", evaluation_groups)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="参照SMF群を読み取り専用で診断します。")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    summary = audit_corpus(args.input_dir, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
