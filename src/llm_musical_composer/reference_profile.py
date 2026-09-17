"""拍子と明示BPMに依存しない参照曲プロファイルを構築する。"""

from __future__ import annotations

import hashlib
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any

import mido

from llm_musical_composer.smf_notes import SmfNote, collect_events, load_smf_notes, match_notes

FEATURE_GROUPS = ("performance_texture", "rhythm_time", "pitch_harmony")


class ReferenceProfileError(ValueError):
    """参照プロファイルを通常値として扱えない場合に送出する。"""


def _round(value: float) -> float:
    return round(float(value), 8)


def _number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReferenceProfileError(f"{location} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ReferenceProfileError(f"{location} must be finite")
    return result


@dataclass(frozen=True)
class PedalEvent:
    at_ms: int
    value: int


@dataclass(frozen=True)
class ReferencePiece:
    name: str
    notes: tuple[SmfNote, ...]
    pedals: tuple[PedalEvent, ...]

    @classmethod
    def from_dicts(
        cls,
        *,
        name: str,
        notes: Iterable[Mapping[str, object]],
        pedals: Iterable[Mapping[str, object]],
    ) -> ReferencePiece:
        if not isinstance(name, str) or not name.strip():
            raise ReferenceProfileError("piece name is required")
        prepared_notes: list[SmfNote] = []
        for index, raw in enumerate(notes):
            pitch = _number(raw.get("pitch"), f"notes[{index}].pitch")
            onset = _number(raw.get("onset_ms"), f"notes[{index}].onset_ms")
            duration = _number(raw.get("duration_ms"), f"notes[{index}].duration_ms")
            velocity = _number(raw.get("velocity"), f"notes[{index}].velocity")
            if not pitch.is_integer() or not 0 <= pitch <= 127:
                raise ReferenceProfileError("note pitch must be an integer from 0 through 127")
            if onset < 0:
                raise ReferenceProfileError("note onset_ms must be non-negative")
            if duration <= 0:
                raise ReferenceProfileError("note duration_ms must be positive")
            if not velocity.is_integer() or not 1 <= velocity <= 127:
                raise ReferenceProfileError("note velocity must be an integer from 1 through 127")
            prepared_notes.append(
                SmfNote(
                    pitch=int(pitch),
                    onset_ms=round(onset),
                    duration_ms=max(1, round(duration)),
                    velocity=int(velocity),
                )
            )
        if not prepared_notes:
            raise ReferenceProfileError("piece must contain at least one completed note")
        if len({item.onset_ms for item in prepared_notes}) < 2:
            raise ReferenceProfileError("piece must contain at least two distinct note onsets")

        prepared_pedals: list[PedalEvent] = []
        for index, raw in enumerate(pedals):
            at_ms = _number(raw.get("at_ms"), f"pedals[{index}].at_ms")
            value = _number(raw.get("value"), f"pedals[{index}].value")
            if at_ms < 0:
                raise ReferenceProfileError("pedal at_ms must be non-negative")
            if not value.is_integer() or not 0 <= value <= 127:
                raise ReferenceProfileError("pedal value must be an integer from 0 through 127")
            prepared_pedals.append(PedalEvent(round(at_ms), int(value)))
        return cls(name.strip(), tuple(prepared_notes), tuple(prepared_pedals))


def _load_pedals(midi: mido.MidiFile) -> list[dict[str, int]]:
    tempo = 500_000
    elapsed_seconds = 0.0
    pedals: list[dict[str, int]] = []
    for message in mido.merge_tracks(midi.tracks):
        elapsed_seconds += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == "set_tempo":
            tempo = message.tempo
        elif message.type == "control_change" and message.control == 64 and message.channel != 9:
            pedals.append({"at_ms": round(elapsed_seconds * 1000), "value": message.value})
    return pedals


def load_reference_piece(path: Path) -> ReferencePiece:
    path = Path(path)
    try:
        midi = mido.MidiFile(path)
    except (OSError, EOFError, ValueError) as error:
        raise ReferenceProfileError(f"unable to read SMF: {error}") from error
    if midi.type == 2:
        raise ReferenceProfileError("SMF format 2 has no single global timeline")
    events, reports = collect_events(midi)
    final_tick = max((report["last_tick"] for report in reports), default=0)
    _, note_status = match_notes(events, final_tick)
    if note_status["dangling_note_on"]:
        raise ReferenceProfileError("SMF contains dangling note_on events")
    if note_status["unmatched_note_off"]:
        raise ReferenceProfileError("SMF contains unmatched note_off events")
    notes = load_smf_notes(path)
    return ReferencePiece.from_dicts(
        name=path.name,
        notes=[
            {
                "pitch": note.pitch,
                "onset_ms": note.onset_ms,
                "duration_ms": note.duration_ms,
                "velocity": note.velocity,
            }
            for note in notes
        ],
        pedals=_load_pedals(midi),
    )


def _distribution(values: Iterable[int], size: int) -> list[float]:
    counts = [0] * size
    total = 0
    for value in values:
        counts[value] += 1
        total += 1
    if total == 0:
        return [0.0] * size
    return [_round(value / total) for value in counts]


def _weighted_distribution(weighted_bins: Iterable[tuple[int, int]], size: int) -> list[float]:
    counts = [0] * size
    total = 0
    for bin_index, weight in weighted_bins:
        counts[bin_index] += weight
        total += weight
    if total == 0:
        return [0.0] * size
    return [_round(value / total) for value in counts]


def _metric(kind: str, values: Sequence[float]) -> dict[str, Any]:
    return {"kind": kind, "values": [_round(value) for value in values]}


def _attacks(notes: Sequence[SmfNote]) -> list[tuple[int, list[SmfNote]]]:
    grouped: dict[int, list[SmfNote]] = {}
    for note in sorted(notes, key=lambda item: (item.onset_ms, item.pitch, item.duration_ms)):
        grouped.setdefault(note.onset_ms, []).append(note)
    return sorted(grouped.items())


def _ratio_bin(value: float) -> int:
    log_ratio = math.log2(max(value, 1e-9))
    for index, boundary in enumerate((-1.0, -0.5, 0.0, 0.5, 1.0)):
        if log_ratio <= boundary:
            return index
    return 5


def _rest_bin(value: float) -> int:
    if value <= 1e-9:
        return 0
    if value <= 0.5:
        return 1
    if value <= 1.0:
        return 2
    if value <= 2.0:
        return 3
    return 4


def _attack_size_bin(size: int) -> int:
    return min(size, 4) - 1


def _relative_pitch_bin(value: float) -> int:
    for index, boundary in enumerate((-12, -7, -3, 3, 7, 12)):
        if value < boundary:
            return index
    return 6


def _magnitude_bin(value: float) -> int:
    value = abs(value)
    if value < 0.5:
        return 0
    if value <= 2:
        return 1
    if value <= 5:
        return 2
    if value <= 11:
        return 3
    return 4


def _interval_class(value: float) -> int:
    distance = abs(round(value)) % 12
    return min(distance, 12 - distance)


def _polyphony_distribution(notes: Sequence[SmfNote], start: int, end: int) -> list[float]:
    changes: dict[int, int] = {}
    for note in notes:
        changes[note.onset_ms] = changes.get(note.onset_ms, 0) + 1
        note_end = note.onset_ms + note.duration_ms
        changes[note_end] = changes.get(note_end, 0) - 1
    active = 0
    previous = start
    weighted: list[tuple[int, int]] = []
    for time in sorted(set(changes) | {start, end}):
        if time > previous:
            weighted.append((min(active, 4), time - previous))
        active += changes.get(time, 0)
        previous = time
    return _weighted_distribution(weighted, 5)


def _pedal_features(
    pedals: Sequence[PedalEvent], attacks: Sequence[tuple[int, list[SmfNote]]], start: int, end: int
) -> tuple[float, float, float]:
    ordered = sorted(pedals, key=lambda item: (item.at_ms, item.value))
    state = False
    previous = start
    on_ms = 0
    changes = 0
    for event in ordered:
        if event.at_ms <= start:
            state = event.value >= 64
            continue
        if event.at_ms >= end:
            break
        if state:
            on_ms += event.at_ms - previous
        new_state = event.value >= 64
        if new_state != state:
            changes += 1
        state = new_state
        previous = event.at_ms
    if state:
        on_ms += end - previous

    pedal_index = 0
    state = False
    under_pedal = 0
    for onset, _ in attacks:
        while pedal_index < len(ordered) and ordered[pedal_index].at_ms <= onset:
            state = ordered[pedal_index].value >= 64
            pedal_index += 1
        under_pedal += state
    duration = max(1, end - start)
    return (
        _round(on_ms / duration),
        _round(under_pedal / len(attacks)),
        _round(min(changes / max(1, len(attacks)), 1.0)),
    )


def extract_reference_profile(piece: ReferencePiece) -> dict[str, Any]:
    notes = sorted(piece.notes, key=lambda item: (item.onset_ms, item.pitch, item.duration_ms))
    attacks = _attacks(notes)
    start = min(note.onset_ms for note in notes)
    end = max(note.onset_ms + note.duration_ms for note in notes)
    duration = max(1, end - start)
    onset_values = [onset for onset, _ in attacks]
    iois = [second - first for first, second in pairwise(onset_values)]
    median_ioi = statistics.median(iois)
    if median_ioi <= 0:
        raise ReferenceProfileError("median inter-onset interval must be positive")

    center_pitch = statistics.median(note.pitch for note in notes)
    pedal_on, attack_under_pedal, pedal_changes = _pedal_features(piece.pedals, attacks, start, end)
    performance = {
        "notes_per_attack": _metric("scalar", [min(len(notes) / len(attacks) / 4.0, 1.0)]),
        "polyphony_duration": _metric("distribution", _polyphony_distribution(notes, start, end)),
        "velocity": _metric(
            "distribution", _distribution((min(note.velocity // 16, 7) for note in notes), 8)
        ),
        "relative_register": _metric(
            "distribution",
            _distribution((_relative_pitch_bin(note.pitch - center_pitch) for note in notes), 7),
        ),
        "pitch_range": _metric(
            "scalar",
            [min((max(note.pitch for note in notes) - min(note.pitch for note in notes)) / 88, 1)],
        ),
        "pedal_on_ratio": _metric("scalar", [pedal_on]),
        "attack_under_pedal_ratio": _metric("scalar", [attack_under_pedal]),
        "pedal_changes_per_attack": _metric("scalar", [pedal_changes]),
    }

    previous_end = max(note.onset_ms + note.duration_ms for note in attacks[0][1])
    rests: list[float] = []
    for onset, attack_notes in attacks[1:]:
        rests.append(max(0, onset - previous_end) / median_ioi)
        previous_end = max(
            previous_end,
            max(note.onset_ms + note.duration_ms for note in attack_notes),
        )
    rhythm = {
        "ioi_ratio": _metric(
            "distribution", _distribution((_ratio_bin(value / median_ioi) for value in iois), 6)
        ),
        "duration_ratio": _metric(
            "distribution",
            _distribution((_ratio_bin(note.duration_ms / median_ioi) for note in notes), 6),
        ),
        "rest_ratio": _metric(
            "distribution", _distribution((_rest_bin(value) for value in rests), 5)
        ),
        "attack_size": _metric(
            "distribution", _distribution((_attack_size_bin(len(items)) for _, items in attacks), 4)
        ),
    }

    attack_medians = [statistics.median(note.pitch for note in items) for _, items in attacks]
    pitch_differences = [second - first for first, second in pairwise(attack_medians)]
    vertical_classes = [
        _interval_class(second.pitch - first.pitch)
        for _, items in attacks
        for first, second in combinations(items, 2)
    ]
    pitch = {
        "transition_interval_class": _metric(
            "distribution",
            _distribution((_interval_class(value) for value in pitch_differences), 7),
        ),
        "transition_direction": _metric(
            "distribution",
            _distribution(
                (0 if value < 0 else 2 if value > 0 else 1 for value in pitch_differences), 3
            ),
        ),
        "transition_magnitude": _metric(
            "distribution", _distribution((_magnitude_bin(value) for value in pitch_differences), 5)
        ),
        "vertical_interval_class": _metric("distribution", _distribution(vertical_classes, 7)),
        "monophonic_attack_ratio": _metric(
            "scalar", [sum(len(items) == 1 for _, items in attacks) / len(attacks)]
        ),
    }
    return {
        "schema_version": 1,
        "status": "pass",
        "basis": "elapsed-time note events; meter and declared BPM are not features",
        "note_count": len(notes),
        "attack_count": len(attacks),
        "duration_ms": duration,
        "feature_groups": {
            "performance_texture": {"metrics": performance},
            "rhythm_time": {"metrics": rhythm},
            "pitch_harmony": {"metrics": pitch},
        },
        "long_term_structure": {"status": "unverified"},
    }


def _validated_groups(profile: Mapping[str, Any]) -> Mapping[str, Any]:
    if profile.get("status") != "pass":
        raise ReferenceProfileError("reference profile status is not pass")
    groups = profile.get("feature_groups")
    if not isinstance(groups, Mapping) or set(groups) != set(FEATURE_GROUPS):
        raise ReferenceProfileError("reference profile feature groups are invalid")
    return groups


def profile_distances(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, float]:
    first_groups = _validated_groups(first)
    second_groups = _validated_groups(second)
    result: dict[str, float] = {}
    for group_name in FEATURE_GROUPS:
        first_metrics = first_groups[group_name].get("metrics")
        second_metrics = second_groups[group_name].get("metrics")
        if not isinstance(first_metrics, Mapping) or not isinstance(second_metrics, Mapping):
            raise ReferenceProfileError("reference profile metrics are invalid")
        if set(first_metrics) != set(second_metrics):
            raise ReferenceProfileError("reference profile metric sets differ")
        distances = []
        for metric_name in sorted(first_metrics):
            one = first_metrics[metric_name]
            two = second_metrics[metric_name]
            if one.get("kind") != two.get("kind") or one.get("kind") not in {
                "scalar",
                "distribution",
            }:
                raise ReferenceProfileError("reference profile metric kinds differ")
            one_values = one.get("values")
            two_values = two.get("values")
            if not isinstance(one_values, list) or not isinstance(two_values, list):
                raise ReferenceProfileError("reference profile metric values are invalid")
            if len(one_values) != len(two_values) or not one_values:
                raise ReferenceProfileError("reference profile metric dimensions differ")
            if not all(
                isinstance(value, (int, float)) and math.isfinite(float(value))
                for value in [*one_values, *two_values]
            ):
                raise ReferenceProfileError("reference profile metric values must be finite")
            distance = sum(
                abs(float(left) - float(right))
                for left, right in zip(one_values, two_values, strict=True)
            )
            if one["kind"] == "distribution":
                distance /= 2
            distances.append(distance)
        result[group_name] = _round(statistics.mean(distances))
    return result


def _group_ranks(
    query: Mapping[str, Any], profiles: Mapping[str, Mapping[str, Any]], names: Iterable[str]
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]]]:
    distances = {name: profile_distances(query, profiles[name]) for name in names}
    ranks: dict[str, dict[str, int]] = {name: {} for name in distances}
    for group in FEATURE_GROUPS:
        ordered = sorted(
            distances, key=lambda name: (distances[name][group], name.casefold(), name)
        )
        for rank, name in enumerate(ordered, 1):
            ranks[name][group] = rank
    return distances, ranks


def select_local_neighborhood(
    anchor_name: str,
    profiles: Mapping[str, Mapping[str, Any]],
    *,
    neighbor_count: int = 7,
) -> dict[str, Any]:
    if anchor_name not in profiles:
        raise ReferenceProfileError(f"unknown anchor: {anchor_name}")
    if neighbor_count < 3:
        raise ReferenceProfileError("neighbor_count must be at least 3")
    candidates = [name for name in profiles if name != anchor_name]
    if len(candidates) < neighbor_count - 1:
        raise ReferenceProfileError("not enough non-anchor references for requested neighborhood")
    distances, ranks = _group_ranks(profiles[anchor_name], profiles, candidates)
    ordered = sorted(
        candidates,
        key=lambda name: (
            max(ranks[name].values()),
            statistics.median(ranks[name].values()),
            name.casefold(),
            name,
        ),
    )
    selected = ordered[: neighbor_count - 1]
    return {
        "anchor": anchor_name,
        "selection_basis": "minimize worst group rank, then median group rank, then name",
        "neighbors": [
            {
                "name": anchor_name,
                "role": "anchor",
                "group_distances": {group: 0.0 for group in FEATURE_GROUPS},
                "group_ranks": {group: 0 for group in FEATURE_GROUPS},
            },
            *[
                {
                    "name": name,
                    "role": "neighbor",
                    "group_distances": distances[name],
                    "group_ranks": ranks[name],
                }
                for name in selected
            ],
        ],
    }


def _coordinate_median(profiles: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    first = next(iter(profiles.values()))
    groups = _validated_groups(first)
    synthetic_groups: dict[str, Any] = {}
    for group in FEATURE_GROUPS:
        metrics: dict[str, Any] = {}
        for metric_name, metric in groups[group]["metrics"].items():
            values = []
            for index in range(len(metric["values"])):
                values.append(
                    statistics.median(
                        profile["feature_groups"][group]["metrics"][metric_name]["values"][index]
                        for profile in profiles.values()
                    )
                )
            if metric["kind"] == "distribution" and sum(values) > 0:
                total = sum(values)
                values = [value / total for value in values]
            metrics[metric_name] = _metric(metric["kind"], values)
        synthetic_groups[group] = {"metrics": metrics}
    return {
        "schema_version": 1,
        "status": "pass",
        "feature_groups": synthetic_groups,
    }


def select_default_reference(profiles: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if len(profiles) < 3:
        raise ReferenceProfileError("at least three references are required")
    names = sorted(profiles, key=lambda name: (name.casefold(), name))
    mean_distances: dict[str, dict[str, float]] = {}
    for name in names:
        comparisons = [
            profile_distances(profiles[name], profiles[other]) for other in names if other != name
        ]
        mean_distances[name] = {
            group: _round(statistics.mean(item[group] for item in comparisons))
            for group in FEATURE_GROUPS
        }
    ranks: dict[str, dict[str, int]] = {name: {} for name in names}
    for group in FEATURE_GROUPS:
        ordered = sorted(
            names, key=lambda name: (mean_distances[name][group], name.casefold(), name)
        )
        for rank, name in enumerate(ordered, 1):
            ranks[name][group] = rank
    selected = min(
        names,
        key=lambda name: (
            max(ranks[name].values()),
            statistics.median(ranks[name].values()),
            name.casefold(),
            name,
        ),
    )

    coordinate_median = _coordinate_median(profiles)
    median_distances = {
        name: profile_distances(coordinate_median, profiles[name]) for name in names
    }
    nearest_by_group = {
        group: min(
            names,
            key=lambda name: (median_distances[name][group], name.casefold(), name),
        )
        for group in FEATURE_GROUPS
    }
    return {
        "selection_basis": "real medoid minimizing worst group rank, then median rank, then name",
        "selected": {
            "kind": "real_medoid",
            "name": selected,
            "group_mean_distances": mean_distances[selected],
            "group_ranks": ranks[selected],
        },
        "coordinate_median": {
            "kind": "diagnostic_synthetic",
            "selectable": False,
            "nearest_real_by_group": nearest_by_group,
            "nearest_distances": {
                group: median_distances[nearest_by_group[group]][group] for group in FEATURE_GROUPS
            },
        },
    }


def _quantized_log_ratio(value: float, center: float) -> int:
    return round(math.log2(max(value, 1e-9) / max(center, 1e-9)) * 4)


def build_copy_fingerprint(piece: ReferencePiece) -> dict[str, Any]:
    attacks = _attacks(piece.notes)
    onsets = [onset for onset, _ in attacks]
    iois = [second - first for first, second in pairwise(onsets)]
    median_ioi = statistics.median(iois)
    attack_pitches_twice = [
        round(2 * statistics.median(note.pitch for note in items)) for _, items in attacks
    ]
    attack_durations = [
        statistics.median(note.duration_ms for note in items) for _, items in attacks
    ]
    median_duration = statistics.median(attack_durations)
    tokens: list[str] = []
    for index in range(len(attacks) - 2):
        tokens.append(
            ":".join(
                str(value)
                for value in (
                    attack_pitches_twice[index + 1] - attack_pitches_twice[index],
                    attack_pitches_twice[index + 2] - attack_pitches_twice[index + 1],
                    _quantized_log_ratio(iois[index], median_ioi),
                    _quantized_log_ratio(iois[index + 1], median_ioi),
                    _quantized_log_ratio(attack_durations[index], median_duration),
                    _quantized_log_ratio(attack_durations[index + 1], median_duration),
                    _quantized_log_ratio(attack_durations[index + 2], median_duration),
                )
            )
        )
    encoded = "|".join(tokens).encode("utf-8")
    return {
        "version": 1,
        "basis": "three-attack relative pitch and normalized timing tokens",
        "token_count": len(tokens),
        "sequence_sha256": hashlib.sha256(encoded).hexdigest().upper(),
        "token_hashes": sorted(
            {hashlib.sha256(token.encode("utf-8")).hexdigest()[:16] for token in tokens}
        ),
    }


def copy_fingerprint_similarity(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    first_tokens = set(first.get("token_hashes", []))
    second_tokens = set(second.get("token_hashes", []))
    if not first_tokens and not second_tokens:
        return 1.0
    union = first_tokens | second_tokens
    return _round(len(first_tokens & second_tokens) / len(union)) if union else 0.0


def evaluate_copy_risk(
    candidate: Mapping[str, Any],
    references: Mapping[str, Mapping[str, Any]],
    *,
    review_threshold: float,
) -> dict[str, Any]:
    if not 0 <= review_threshold <= 1:
        raise ReferenceProfileError("copy review threshold must be from 0 through 1")
    comparisons = [
        {
            "name": name,
            "exact": candidate.get("sequence_sha256") == fingerprint.get("sequence_sha256"),
            "similarity": copy_fingerprint_similarity(candidate, fingerprint),
        }
        for name, fingerprint in references.items()
    ]
    if not comparisons:
        raise ReferenceProfileError("at least one copy reference is required")
    nearest = min(
        comparisons,
        key=lambda item: (-int(item["exact"]), -item["similarity"], item["name"].casefold()),
    )
    if nearest["exact"]:
        status = "exact_copy"
    elif nearest["similarity"] >= review_threshold:
        status = "review"
    else:
        status = "pass"
    return {
        "status": status,
        "nearest_reference": nearest["name"],
        "exact": nearest["exact"],
        "similarity": nearest["similarity"],
        "review_threshold": review_threshold,
    }
