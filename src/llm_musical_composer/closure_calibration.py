"""人間ラベル付き終止判定校正セットの決定的な部品。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mido

from llm_musical_composer.ending_closure import COMPARISON_METRICS, extract_closure_features
from llm_musical_composer.pilot_features import NoteEvent

TICKS_PER_BEAT = 500
TEMPO = 500_000
ALLOWED_ANSWERS = ("A", "B", "同程度", "判定不能")


@dataclass(frozen=True)
class PedalEvent:
    at_ms: int
    value: int


@dataclass(frozen=True)
class Performance:
    notes: tuple[NoteEvent, ...]
    pedals: tuple[PedalEvent, ...]
    duration_ms: int


@dataclass(frozen=True)
class Candidate:
    name: str
    path: Path
    features: tuple[float, ...]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_performance(path: Path) -> Performance:
    """SMF を拍に依存しないミリ秒単位のピアノ演奏へ変換する。"""
    midi = mido.MidiFile(path)
    if midi.type == 2:
        raise ValueError("SMF format 2 has no single global timeline")
    tempo = 500_000
    elapsed_seconds = 0.0
    active: dict[tuple[int, int], deque[tuple[int, int]]] = defaultdict(deque)
    notes: list[NoteEvent] = []
    pedals: list[PedalEvent] = []
    for message in mido.merge_tracks(midi.tracks, skip_checks=True):
        elapsed_seconds += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        at_ms = round(elapsed_seconds * 1_000)
        if message.type == "set_tempo":
            tempo = message.tempo
            continue
        if not hasattr(message, "channel") or message.channel == 9:
            continue
        if message.type == "control_change" and message.control == 64:
            pedals.append(PedalEvent(at_ms, message.value))
            continue
        is_on = message.type == "note_on" and message.velocity > 0
        is_off = message.type == "note_off" or (message.type == "note_on" and message.velocity == 0)
        key = (message.channel, message.note) if is_on or is_off else None
        if is_on and key is not None:
            active[key].append((at_ms, message.velocity))
        elif is_off and key is not None and active[key]:
            onset_ms, velocity = active[key].popleft()
            if at_ms > onset_ms:
                notes.append(NoteEvent(message.note, onset_ms, at_ms - onset_ms, velocity))
    duration_ms = round(elapsed_seconds * 1_000)
    ordered_notes = tuple(
        sorted(notes, key=lambda note: (note.onset_ms, note.pitch, note.duration_ms))
    )
    ordered_pedals = tuple(sorted(pedals, key=lambda event: (event.at_ms, event.value)))
    return Performance(ordered_notes, ordered_pedals, duration_ms)


def slice_performance(performance: Performance, start_ms: int, end_ms: int) -> Performance:
    """指定区間を切り出し、境界をまたぐノートとペダルを閉じる。"""
    if start_ms < 0:
        raise ValueError("start must not be negative")
    if end_ms <= start_ms:
        raise ValueError("end must be greater than start")
    duration_ms = end_ms - start_ms
    notes = tuple(
        NoteEvent(
            note.pitch,
            max(note.onset_ms, start_ms) - start_ms,
            min(note.onset_ms + note.duration_ms, end_ms) - max(note.onset_ms, start_ms),
            note.velocity,
        )
        for note in performance.notes
        if note.onset_ms < end_ms and note.onset_ms + note.duration_ms > start_ms
    )
    previous_pedal = next(
        (event.value for event in reversed(performance.pedals) if event.at_ms < start_ms),
        0,
    )
    pedal_values: list[PedalEvent] = []
    if previous_pedal:
        pedal_values.append(PedalEvent(0, previous_pedal))
    pedal_values.extend(
        PedalEvent(event.at_ms - start_ms, event.value)
        for event in performance.pedals
        if start_ms <= event.at_ms < end_ms
    )
    pedals: list[PedalEvent] = []
    for event in (*pedal_values, PedalEvent(duration_ms, 0)):
        if pedals and pedals[-1] == event:
            continue
        pedals.append(event)
    return Performance(notes, tuple(pedals), duration_ms)


def write_performance(performance: Performance, path: Path) -> None:
    """1 tick を 1 ms とする単一ピアノ SMF を書く。"""
    midi = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name="Calibration sample", time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=TEMPO, time=0))
    meta.append(mido.MetaMessage("end_of_track", time=performance.duration_ms))
    midi.tracks.append(meta)
    piano = mido.MidiTrack()
    piano.append(mido.MetaMessage("track_name", name="Piano", time=0))
    piano.append(mido.Message("program_change", channel=0, program=0, time=0))
    events: list[tuple[int, int, mido.Message]] = []
    for note in performance.notes:
        events.append(
            (
                note.onset_ms,
                2,
                mido.Message("note_on", channel=0, note=note.pitch, velocity=note.velocity),
            )
        )
        events.append(
            (
                note.onset_ms + note.duration_ms,
                0,
                mido.Message("note_off", channel=0, note=note.pitch, velocity=0),
            )
        )
    for pedal in performance.pedals:
        events.append(
            (
                pedal.at_ms,
                1,
                mido.Message("control_change", channel=0, control=64, value=pedal.value),
            )
        )
    previous_tick = 0
    for tick, _, message in sorted(events, key=lambda item: (item[0], item[1])):
        message.time = tick - previous_tick
        piano.append(message)
        previous_tick = tick
    piano.append(mido.MetaMessage("end_of_track", time=performance.duration_ms - previous_tick))
    midi.tracks.append(piano)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(path)


def find_natural_middle_end(
    performance: Performance,
    *,
    target_fraction: float = 0.65,
    search_radius: float = 0.05,
) -> int | None:
    """目標位置付近で、直後の空きが長いノート終端を選ぶ。"""
    if not performance.notes:
        return None
    sounding_start = min(note.onset_ms for note in performance.notes)
    sounding_end = max(note.onset_ms + note.duration_ms for note in performance.notes)
    span = sounding_end - sounding_start
    if span <= 0 or not 0 <= target_fraction - search_radius < target_fraction + search_radius <= 1:
        return None
    target = sounding_start + span * target_fraction
    lower = math.ceil(sounding_start + span * (target_fraction - search_radius))
    upper = math.floor(sounding_start + span * (target_fraction + search_radius))
    candidates = sorted(
        {
            note.onset_ms + note.duration_ms
            for note in performance.notes
            if lower <= note.onset_ms + note.duration_ms <= upper
        }
    )
    if not candidates:
        return None

    def score(ending: int) -> tuple[int, int, int, float, int]:
        active_count = sum(
            note.onset_ms < ending < note.onset_ms + note.duration_ms for note in performance.notes
        )
        next_onset = min(
            (note.onset_ms for note in performance.notes if note.onset_ms >= ending),
            default=sounding_end,
        )
        following_gap = max(0, next_onset - ending)
        radius_ms = max(1, round(span * 0.02))
        local_onsets = sum(
            ending - radius_ms <= note.onset_ms <= ending + radius_ms for note in performance.notes
        )
        return active_count, -following_gap, local_onsets, abs(ending - target), ending

    return min(candidates, key=score)


def build_conflicts(
    evaluation_groups: dict[str, Any], duplicate_candidates: dict[str, Any]
) -> dict[str, frozenset[str]]:
    """系列、完全一致、近似候補を同時選択しないための関係を作る。"""
    groups: list[list[str]] = [
        list(group["files"]) for group in evaluation_groups.get("filename_series", [])
    ]
    for key in ("exact_groups", "musical_fingerprint_groups"):
        groups.extend(list(group["files"]) for group in duplicate_candidates.get(key, []))
    groups.extend(
        [item["first"], item["second"]]
        for item in duplicate_candidates.get("pairwise_positive_evidence", [])
    )
    conflicts: dict[str, set[str]] = defaultdict(set)
    for group in groups:
        for name in group:
            conflicts[name].update(other for other in group if other != name)
    return {name: frozenset(values) for name, values in conflicts.items()}


def _normalized_features(candidates: Sequence[Candidate]) -> dict[str, tuple[float, ...]]:
    lengths = {len(candidate.features) for candidate in candidates}
    if len(lengths) != 1:
        raise ValueError("candidate feature vectors must have equal length")
    dimensions = next(iter(lengths), 0)
    bounds = [
        (
            min(candidate.features[index] for candidate in candidates),
            max(candidate.features[index] for candidate in candidates),
        )
        for index in range(dimensions)
    ]
    return {
        candidate.name: tuple(
            0.5 if maximum == minimum else (value - minimum) / (maximum - minimum)
            for value, (minimum, maximum) in zip(candidate.features, bounds, strict=True)
        )
        for candidate in candidates
    }


def _tie_key(seed: int, name: str) -> str:
    return hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()


def select_diverse_candidates(
    candidates: Sequence[Candidate],
    conflicts: dict[str, frozenset[str]],
    *,
    count: int,
    seed: int,
) -> list[Candidate]:
    """特徴空間の距離を広げつつ、競合関係を避けて選ぶ。"""
    if count < 1 or count > len(candidates):
        raise ValueError(f"cannot select {count} candidates")
    normalized = _normalized_features(candidates)
    remaining = {candidate.name: candidate for candidate in candidates}
    first_name = min(remaining, key=lambda name: _tie_key(seed, name))
    selected = [remaining.pop(first_name)]
    while len(selected) < count:
        blocked = {
            name for candidate in selected for name in conflicts.get(candidate.name, frozenset())
        }
        available = [candidate for name, candidate in remaining.items() if name not in blocked]
        if not available:
            raise ValueError(f"cannot select {count} candidates without conflicts")

        def minimum_distance(candidate: Candidate) -> float:
            vector = normalized[candidate.name]
            return min(math.dist(vector, normalized[chosen.name]) for chosen in selected)

        chosen = min(
            available,
            key=lambda candidate: (-minimum_distance(candidate), _tie_key(seed, candidate.name)),
        )
        selected.append(chosen)
        remaining.pop(chosen.name)
    return selected


def assign_pair_roles(name: str, *, seed: int) -> dict[str, str]:
    actual_first = int(_tie_key(seed, name)[:2], 16) % 2 == 0
    return {"A": "actual", "B": "middle"} if actual_first else {"A": "middle", "B": "actual"}


def assign_splits(pair_ids: Sequence[str]) -> dict[str, str]:
    if len(pair_ids) % 2:
        raise ValueError("pair count must be even")
    ordered = sorted(pair_ids, key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    midpoint = len(ordered) // 2
    return {
        pair_id: "few_shot" if index < midpoint else "holdout"
        for index, pair_id in enumerate(ordered)
    }


def _mean(values: Iterable[int]) -> float | None:
    collected = list(values)
    return round(sum(collected) / len(collected), 4) if collected else None


def summarize_performance(
    performance: Performance, *, window_count: int = 16, terminal_limit: int = 64
) -> dict[str, Any]:
    if window_count < 1 or terminal_limit < 1:
        raise ValueError("window_count and terminal_limit must be positive")
    windows: list[dict[str, Any]] = []
    for index in range(window_count):
        start = performance.duration_ms * index / window_count
        end = performance.duration_ms * (index + 1) / window_count
        notes = [note for note in performance.notes if start <= note.onset_ms < end]
        windows.append(
            {
                "index": index,
                "onset_count": len(notes),
                "pitch_mean": _mean(note.pitch for note in notes),
                "velocity_mean": _mean(note.velocity for note in notes),
            }
        )
    terminal = sorted(
        performance.notes,
        key=lambda note: (note.onset_ms, note.pitch, note.duration_ms),
    )[-terminal_limit:]
    denominator = max(1, performance.duration_ms)
    return {
        "duration_ms": performance.duration_ms,
        "note_count": len(performance.notes),
        "windows": windows,
        "terminal_events": [
            {
                "onset_ratio": round(note.onset_ms / denominator, 6),
                "duration_ratio": round(note.duration_ms / denominator, 6),
                "onset_ms": note.onset_ms,
                "duration_ms": note.duration_ms,
                "pitch": note.pitch,
                "velocity": note.velocity,
            }
            for note in terminal
        ],
        "pedal_events": [
            {
                "at_ratio": round(event.at_ms / denominator, 6),
                "at_ms": event.at_ms,
                "value": event.value,
            }
            for event in performance.pedals
        ],
    }


def validate_answer(value: str) -> str:
    if value not in ALLOWED_ANSWERS:
        raise ValueError(f"answer must be one of the allowed values: {ALLOWED_ANSWERS}")
    return value


def create_evaluation_form(pair_ids: Sequence[str], path: Path) -> None:
    _write_json(
        Path(path),
        {
            "schema_version": 1,
            "scope": "局所的な終止感",
            "scope_note": (
                "末尾付近で音楽が区切れたように感じるかを比較する。"
                "曲全体の構成や総合品質の評価には使わない。"
            ),
            "allowed_values": list(ALLOWED_ANSWERS),
            "procedure": [
                "対応表を開かず sample-A.mid と sample-B.mid を聴く",
                "終止感が強い方、同程度、判定不能のいずれかを記入する",
                "好みや音質ではなく、その位置で曲が区切れたように感じるかを比べる",
            ],
            "answers": [
                {"pair_id": pair_id, "answer": None, "reason": None} for pair_id in pair_ids
            ],
        },
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _maximum_polyphony(notes: Sequence[NoteEvent]) -> int:
    changes = [
        change
        for note in notes
        for change in (
            (note.onset_ms, 1),
            (note.onset_ms + note.duration_ms, -1),
        )
    ]
    current = maximum = 0
    for _, delta in sorted(changes, key=lambda item: (item[0], item[1])):
        current += delta
        maximum = max(maximum, current)
    return maximum


def _candidate_features(performance: Performance) -> tuple[float, ...]:
    sounding_start = min(note.onset_ms for note in performance.notes)
    sounding_end = max(note.onset_ms + note.duration_ms for note in performance.notes)
    span_seconds = max(0.001, (sounding_end - sounding_start) / 1_000)
    pitches = [note.pitch for note in performance.notes]
    closure = extract_closure_features(list(performance.notes), smf_end_ms=performance.duration_ms)[
        "comparison_metrics"
    ]
    return (
        span_seconds,
        len(performance.notes) / span_seconds,
        float(_maximum_polyphony(performance.notes)),
        float(max(pitches) - min(pitches)),
        *(float(closure.get(metric) or 0.0) for metric in COMPARISON_METRICS),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def build_calibration_set(
    source_root: Path,
    audit_root: Path,
    output_root: Path,
    *,
    count: int = 16,
    seed: int = 20_260_811,
) -> dict[str, Any]:
    """監査済み曲から匿名の局所的終止感比較セットを作る。"""
    source_root = Path(source_root)
    audit_root = Path(audit_root)
    output_root = Path(output_root)
    records = _read_records(audit_root / "files.jsonl")
    evaluation_groups = _read_json(audit_root / "evaluation-groups.json")
    duplicate_candidates = _read_json(audit_root / "duplicate-candidates.json")
    conflicts = build_conflicts(evaluation_groups, duplicate_candidates)
    excluded = {"rut.mid", "aimusic01.mid"}
    diagnostics: dict[str, list[str]] = {
        "excluded_by_user": [],
        "not_affirmative_or_conformant": [],
        "missing_source": [],
        "no_complete_notes": [],
        "no_natural_middle_end": [],
    }
    performances: dict[str, Performance] = {}
    middle_endings: dict[str, int] = {}
    candidates: list[Candidate] = []
    for record in records:
        name = str(record["name"])
        path = source_root / str(record.get("relative_path", name))
        if name.casefold() in excluded:
            diagnostics["excluded_by_user"].append(name)
            continue
        if record.get("status") != "affirmative_evidence" or not record.get("conformance", {}).get(
            "is_structurally_conformant"
        ):
            diagnostics["not_affirmative_or_conformant"].append(name)
            continue
        if not path.is_file():
            diagnostics["missing_source"].append(name)
            continue
        performance = read_performance(path)
        if not performance.notes:
            diagnostics["no_complete_notes"].append(name)
            continue
        middle_end = find_natural_middle_end(performance)
        if middle_end is None:
            diagnostics["no_natural_middle_end"].append(name)
            continue
        performances[name] = performance
        middle_endings[name] = middle_end
        candidates.append(Candidate(name, path, _candidate_features(performance)))
    selected = select_diverse_candidates(candidates, conflicts, count=count, seed=seed)
    pair_ids = [f"pair-{index:02d}" for index in range(1, count + 1)]
    splits = assign_splits(pair_ids)
    pairs: list[dict[str, Any]] = []
    for pair_id, candidate in zip(pair_ids, selected, strict=True):
        performance = performances[candidate.name]
        sounding_start = min(note.onset_ms for note in performance.notes)
        sounding_end = max(note.onset_ms + note.duration_ms for note in performance.notes)
        span = sounding_end - sounding_start
        excerpt_duration = round(span * 0.30)
        actual_interval = (sounding_end - excerpt_duration, sounding_end)
        middle_end = middle_endings[candidate.name]
        middle_interval = (middle_end - excerpt_duration, middle_end)
        excerpts = {
            "actual": slice_performance(performance, *actual_interval),
            "middle": slice_performance(performance, *middle_interval),
        }
        roles = assign_pair_roles(candidate.name, seed=seed)
        labeled = {label: excerpts[role] for label, role in roles.items()}
        listen_root = output_root / "listen" / pair_id
        artifacts: dict[str, dict[str, str]] = {}
        for label in ("A", "B"):
            midi_path = listen_root / f"sample-{label}.mid"
            write_performance(labeled[label], midi_path)
            artifacts[label] = {
                "path": midi_path.relative_to(output_root).as_posix(),
                "sha256": _sha256(midi_path),
            }
        from llm_musical_composer.piano_roll import render_pair_piano_roll

        image_path = output_root / "judge" / f"{pair_id}.png"
        image = render_pair_piano_roll(labeled["A"], labeled["B"], image_path)
        judge_path = output_root / "judge" / f"{pair_id}.json"
        _write_json(
            judge_path,
            {
                "schema_version": 1,
                "pair_id": pair_id,
                "scope": "局所的な終止感",
                "samples": {label: summarize_performance(labeled[label]) for label in ("A", "B")},
                "full_structure": summarize_performance(
                    performance, window_count=32, terminal_limit=16
                )["windows"],
                "piano_roll": image,
            },
        )
        pairs.append(
            {
                "pair_id": pair_id,
                "source": candidate.name,
                "split": splits[pair_id],
                "roles": roles,
                "intervals_ms": {
                    "actual": list(actual_interval),
                    "middle": list(middle_interval),
                },
                "selection_features": list(candidate.features),
                "artifacts": artifacts,
                "judge_sha256": _sha256(judge_path),
                "piano_roll_sha256": _sha256(image_path),
            }
        )
    _write_json(
        output_root / "selection-manifest.json",
        {
            "schema_version": 1,
            "seed": seed,
            "excluded": sorted(excluded),
            "selection_rule": "normalized farthest-first with conflict exclusion",
            "diagnostics": diagnostics,
            "pairs": pairs,
        },
    )
    _write_json(
        output_root / "split-manifest.json",
        {"schema_version": 1, "splits": splits},
    )
    create_evaluation_form(pair_ids, output_root / "evaluation-form.json")
    return {
        "status": "affirmative_evidence",
        "candidate_count": len(candidates),
        "diagnostics": diagnostics,
        "pair_count": len(pairs),
        "few_shot_count": list(splits.values()).count("few_shot"),
        "holdout_count": list(splits.values()).count("holdout"),
        "output_root": str(output_root.resolve()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("audit_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20_260_811)
    arguments = parser.parse_args(argv)
    print(
        json.dumps(
            build_calibration_set(
                arguments.source_root,
                arguments.audit_root,
                arguments.output_root,
                count=arguments.count,
                seed=arguments.seed,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
