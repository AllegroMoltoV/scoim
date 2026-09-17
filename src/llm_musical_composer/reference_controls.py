"""参照プロファイルを決定的な無害変換と破壊対照で反証する。"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import replace
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any

from llm_musical_composer.reference_profile import (
    FEATURE_GROUPS,
    PedalEvent,
    ReferencePiece,
    ReferenceProfileError,
    build_copy_fingerprint,
    copy_fingerprint_similarity,
    extract_reference_profile,
    load_reference_piece,
    profile_distances,
    select_default_reference,
    select_local_neighborhood,
)
from llm_musical_composer.run_state import (
    atomic_write_bytes,
    atomic_write_json,
    sha256_file,
)

EXCLUDED_NAMES = frozenset({"rut.mid", "aimusic01.mid"})


def _round(value: float) -> float:
    return round(float(value), 8)


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ReferenceProfileError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return _round(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def transpose_piece(piece: ReferencePiece, semitones: int) -> ReferencePiece:
    if isinstance(semitones, bool) or not isinstance(semitones, int):
        raise ReferenceProfileError("semitones must be an integer")
    notes = tuple(replace(note, pitch=note.pitch + semitones) for note in piece.notes)
    if any(not 0 <= note.pitch <= 127 for note in notes):
        raise ReferenceProfileError("transposition moves a note outside MIDI pitch range")
    return replace(piece, notes=notes)


def stretch_piece(piece: ReferencePiece, factor: float) -> ReferencePiece:
    if not math.isfinite(factor) or factor <= 0:
        raise ReferenceProfileError("stretch factor must be finite and positive")
    return replace(
        piece,
        notes=tuple(
            replace(
                note,
                onset_ms=round(note.onset_ms * factor),
                duration_ms=max(1, round(note.duration_ms * factor)),
            )
            for note in piece.notes
        ),
        pedals=tuple(
            PedalEvent(round(event.at_ms * factor), event.value) for event in piece.pedals
        ),
    )


def reorder_piece(piece: ReferencePiece) -> ReferencePiece:
    return replace(piece, notes=tuple(reversed(piece.notes)), pedals=tuple(reversed(piece.pedals)))


def destroy_pitch_order(piece: ReferencePiece) -> ReferencePiece:
    ordered_indices = sorted(
        range(len(piece.notes)),
        key=lambda index: (
            piece.notes[index].onset_ms,
            piece.notes[index].pitch,
            piece.notes[index].duration_ms,
        ),
    )
    reversed_pitches = list(reversed([piece.notes[index].pitch for index in ordered_indices]))
    replacements = dict(zip(ordered_indices, reversed_pitches, strict=True))
    return replace(
        piece,
        notes=tuple(
            replace(note, pitch=replacements[index]) for index, note in enumerate(piece.notes)
        ),
    )


def destroy_timing(piece: ReferencePiece) -> ReferencePiece:
    onsets = sorted({note.onset_ms for note in piece.notes})
    iois = [second - first for first, second in pairwise(onsets)]
    center = max(1, round(statistics.median(iois)))
    new_onsets = [0]
    gap_pattern = (1, 4, 1, 5, 2, 6)
    for index in range(len(onsets) - 1):
        new_onsets.append(new_onsets[-1] + center * gap_pattern[index % len(gap_pattern)])
    onset_map = dict(zip(onsets, new_onsets, strict=True))
    duration_pattern = (1, 5, 2, 6)
    attack_index = {onset: index for index, onset in enumerate(onsets)}
    return replace(
        piece,
        notes=tuple(
            replace(
                note,
                onset_ms=onset_map[note.onset_ms],
                duration_ms=max(
                    1,
                    round(center * duration_pattern[attack_index[note.onset_ms] % 4] / 2),
                ),
            )
            for note in piece.notes
        ),
    )


def flatten_velocity(piece: ReferencePiece) -> ReferencePiece:
    velocity = round(statistics.median(note.velocity for note in piece.notes))
    return replace(piece, notes=tuple(replace(note, velocity=velocity) for note in piece.notes))


def destroy_pedal(piece: ReferencePiece) -> ReferencePiece:
    return replace(piece, pedals=())


def _adaptive_transposition(piece: ReferencePiece) -> int:
    pitches = [note.pitch for note in piece.notes]
    if max(pitches) <= 122:
        return 5
    if min(pitches) >= 5:
        return -5
    if max(pitches) < 127:
        return 1
    if min(pitches) > 0:
        return -1
    raise ReferenceProfileError("piece spans the entire MIDI range and cannot be transposed")


def _control_record(base: dict[str, Any], transformed: dict[str, Any]) -> dict[str, float]:
    return profile_distances(base, transformed)


def _pairwise_dispersion(profiles: Sequence[dict[str, Any]], group: str) -> dict[str, Any]:
    values = [
        profile_distances(first, second)[group] for first, second in combinations(profiles, 2)
    ]
    near_tie_ratio = sum(value <= 1e-9 for value in values) / len(values)
    near_separate_ratio = sum(value >= 1 - 1e-9 for value in values) / len(values)
    status = "pass" if near_tie_ratio < 0.95 and near_separate_ratio < 0.95 else "fail"
    return {
        "status": status,
        "pair_count": len(values),
        "median_distance": _round(statistics.median(values)),
        "near_tie_ratio": _round(near_tie_ratio),
        "near_complete_separation_ratio": _round(near_separate_ratio),
    }


def evaluate_reference_controls(pieces: Iterable[ReferencePiece]) -> dict[str, Any]:
    prepared = sorted(pieces, key=lambda item: (item.name.casefold(), item.name))
    if len(prepared) < 3 or len({item.name.casefold() for item in prepared}) != len(prepared):
        raise ReferenceProfileError("reference controls require at least three unique pieces")
    profiles = [extract_reference_profile(piece) for piece in prepared]
    records: dict[str, list[dict[str, float]]] = {
        name: []
        for name in (
            "input_order",
            "transposition",
            "time_stretch",
            "pitch_order_destruction",
            "timing_destruction",
            "velocity_flattening",
            "pedal_removal",
        )
    }
    transformed_profiles: dict[str, list[dict[str, Any]]] = {
        "input_order": [],
        "transposition": [],
        "time_stretch": [],
    }
    for piece, profile in zip(prepared, profiles, strict=True):
        harmless_profiles = {
            "input_order": extract_reference_profile(reorder_piece(piece)),
            "transposition": extract_reference_profile(
                transpose_piece(piece, _adaptive_transposition(piece))
            ),
            "time_stretch": extract_reference_profile(stretch_piece(piece, 1.5)),
        }
        for control_name, transformed in harmless_profiles.items():
            transformed_profiles[control_name].append(transformed)
            records[control_name].append(_control_record(profile, transformed))
        records["pitch_order_destruction"].append(
            _control_record(profile, extract_reference_profile(destroy_pitch_order(piece)))
        )
        records["timing_destruction"].append(
            _control_record(profile, extract_reference_profile(destroy_timing(piece)))
        )
        records["velocity_flattening"].append(
            _control_record(profile, extract_reference_profile(flatten_velocity(piece)))
        )
        if piece.pedals:
            records["pedal_removal"].append(
                _control_record(profile, extract_reference_profile(destroy_pedal(piece)))
            )

    harmless_names = ("input_order", "transposition", "time_stretch")
    control_summary: dict[str, Any] = {}
    for name, values in records.items():
        control_summary[name] = {
            "status": (
                "pass"
                if values
                and all(
                    all(math.isfinite(distance) and distance >= 0 for distance in item.values())
                    for item in values
                )
                else "unable_to_investigate"
            ),
            "available_count": len(values),
            "groups": {
                group: {
                    "median_distance": _round(statistics.median(item[group] for item in values)),
                    "p95_distance": _percentile([item[group] for item in values], 0.95),
                    "maximum_distance": _round(max(item[group] for item in values)),
                }
                for group in FEATURE_GROUPS
            }
            if values
            else {},
        }
    input_order_pass = all(
        item[group] == 0 for item in records["input_order"] for group in FEATURE_GROUPS
    )
    control_summary["input_order"]["status"] = "pass" if input_order_pass else "fail"

    for control_name, queries in transformed_profiles.items():
        ranks: dict[str, list[int]] = {group: [] for group in FEATURE_GROUPS}
        for index, query in enumerate(queries):
            own = profile_distances(query, profiles[index])
            comparisons = [profile_distances(query, candidate) for candidate in profiles]
            for group in FEATURE_GROUPS:
                ranks[group].append(
                    1
                    + sum(
                        item[group] < own[group] - 1e-12
                        for candidate_index, item in enumerate(comparisons)
                        if candidate_index != index
                    )
                )
        control_summary[control_name]["self_retrieval"] = {
            group: {
                "median_tie_aware_rank": _round(statistics.median(values)),
                "p95_tie_aware_rank": _percentile(values, 0.95),
                "top_rank_rate": _round(sum(value == 1 for value in values) / len(values)),
            }
            for group, values in ranks.items()
        }

    harmless_limits = {
        group: _percentile([item[group] for name in harmless_names for item in records[name]], 0.95)
        for group in FEATURE_GROUPS
    }
    group_control = {
        "performance_texture": "velocity_flattening",
        "rhythm_time": "timing_destruction",
        "pitch_harmony": "pitch_order_destruction",
    }
    specificity = {
        "performance_texture": (
            all(
                item["rhythm_time"] == 0 and item["pitch_harmony"] == 0
                for item in records["velocity_flattening"]
            )
            and all(
                item["rhythm_time"] == 0 and item["pitch_harmony"] == 0
                for item in records["pedal_removal"]
            )
        ),
        "rhythm_time": all(item["pitch_harmony"] == 0 for item in records["timing_destruction"]),
        "pitch_harmony": all(
            item["rhythm_time"] == 0 for item in records["pitch_order_destruction"]
        ),
    }
    groups: dict[str, Any] = {}
    for group, destructive_name in group_control.items():
        destructive_median = statistics.median(item[group] for item in records[destructive_name])
        dispersion = _pairwise_dispersion(profiles, group)
        invariance_pass = (
            control_summary["input_order"]["groups"][group]["maximum_distance"] == 0
            and control_summary["transposition"]["groups"][group]["maximum_distance"] == 0
            and harmless_limits[group] < dispersion["median_distance"]
            and control_summary["time_stretch"]["self_retrieval"][group]["p95_tie_aware_rank"] == 1
        )
        discrimination_pass = destructive_median > harmless_limits[group] + 1e-6
        status = (
            "pass"
            if input_order_pass
            and invariance_pass
            and discrimination_pass
            and specificity[group]
            and dispersion["status"] == "pass"
            else "fail"
        )
        groups[group] = {
            "status": status,
            "harmless_p95_distance": _round(harmless_limits[group]),
            "destructive_control": destructive_name,
            "destructive_median_distance": _round(destructive_median),
            "invariance_pass": invariance_pass,
            "discrimination_pass": discrimination_pass,
            "specificity_pass": specificity[group],
            "dispersion": dispersion,
        }
    passing = sum(item["status"] == "pass" for item in groups.values())
    return {
        "schema_version": 1,
        "status": "pass" if passing >= 2 else "fail",
        "passing_group_count": passing,
        "groups": groups,
        "controls": control_summary,
        "decision_rule": "at least two independently reported feature groups must pass",
    }


def _copy_policy(
    fingerprints: dict[str, dict[str, Any]], pieces: dict[str, ReferencePiece]
) -> dict[str, Any]:
    similarities = [
        copy_fingerprint_similarity(fingerprints[first], fingerprints[second])
        for first, second in combinations(sorted(fingerprints), 2)
    ]
    review_threshold = _percentile(similarities, 0.995)
    transposition_matches = 0
    stretch_exact_matches = 0
    stretch_similarities: list[float] = []
    stretch_ranks: list[int] = []
    for name, piece in pieces.items():
        expected = fingerprints[name]
        transposition_matches += (
            build_copy_fingerprint(transpose_piece(piece, _adaptive_transposition(piece)))
            == expected
        )
        stretched = build_copy_fingerprint(stretch_piece(piece, 1.5))
        stretch_exact_matches += stretched == expected
        own_similarity = copy_fingerprint_similarity(stretched, expected)
        stretch_similarities.append(own_similarity)
        stretch_ranks.append(
            1
            + sum(
                copy_fingerprint_similarity(stretched, candidate) > own_similarity + 1e-12
                for candidate_name, candidate in fingerprints.items()
                if candidate_name != name
            )
        )
    controls_pass = (
        transposition_matches == len(pieces)
        and _percentile(stretch_ranks, 0.95) == 1
        and _percentile(stretch_similarities, 0.05) > review_threshold
    )
    return {
        "status": "pass" if controls_pass else "fail",
        "exact_copy_action": "reject",
        "high_similarity_action": "review",
        "review_threshold_basis": "99.5th percentile of non-self corpus pair similarity",
        "review_threshold": review_threshold,
        "pair_count": len(similarities),
        "median_similarity": _round(statistics.median(similarities)),
        "maximum_similarity": _round(max(similarities)),
        "separate_from_style_ranking": True,
        "controls": {
            "transposition_exact_match_rate": _round(transposition_matches / len(pieces)),
            "time_stretch_exact_match_rate": _round(stretch_exact_matches / len(pieces)),
            "time_stretch_self_similarity_p05": _percentile(stretch_similarities, 0.05),
            "time_stretch_self_similarity_median": _round(statistics.median(stretch_similarities)),
            "time_stretch_self_retrieval_p95_rank": _percentile(stretch_ranks, 0.95),
            "time_stretch_top_rank_rate": _round(
                sum(rank == 1 for rank in stretch_ranks) / len(stretch_ranks)
            ),
        },
    }


def build_reference_profile_v1(
    *,
    source_dir: Path,
    output_dir: Path,
    neighbor_count: int = 7,
) -> dict[str, Any]:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    paths = sorted(source_dir.glob("*.mid"), key=lambda path: (path.name.casefold(), path.name))
    excluded = [
        {"name": path.name, "status": "excluded"}
        for path in paths
        if path.name.casefold() in {name.casefold() for name in EXCLUDED_NAMES}
    ]
    pieces: dict[str, ReferencePiece] = {}
    unable: list[dict[str, str]] = []
    for path in paths:
        if path.name.casefold() in {name.casefold() for name in EXCLUDED_NAMES}:
            continue
        try:
            pieces[path.name] = load_reference_piece(path)
        except ReferenceProfileError as error:
            unable.append(
                {
                    "name": path.name,
                    "status": "unable_to_investigate",
                    "error": str(error),
                }
            )
    if len(pieces) < max(3, neighbor_count):
        raise ReferenceProfileError("not enough usable references for requested neighborhood")

    profiles = {name: extract_reference_profile(piece) for name, piece in pieces.items()}
    fingerprints = {name: build_copy_fingerprint(piece) for name, piece in pieces.items()}
    controls = evaluate_reference_controls(pieces.values())
    neighborhoods = {
        name: select_local_neighborhood(name, profiles, neighbor_count=neighbor_count)
        for name in profiles
    }
    default_reference = select_default_reference(profiles)
    copy_policy = _copy_policy(fingerprints, pieces)
    status = (
        "pass"
        if controls["passing_group_count"] >= 2 and copy_policy["status"] == "pass"
        else "fail"
    )
    records = [
        {
            "name": name,
            "status": "pass",
            "profile": profiles[name],
            "copy_fingerprint": fingerprints[name],
            "neighborhood": neighborhoods[name],
        }
        for name in sorted(profiles, key=lambda value: (value.casefold(), value))
    ]
    summary = {
        "schema_version": 1,
        "status": status,
        "source_count": len(records),
        "excluded": excluded,
        "unable_to_investigate": unable,
        "feature_groups": list(FEATURE_GROUPS),
        "controls": controls,
        "default_reference": default_reference,
        "copy_policy": copy_policy,
        "long_term_structure": {
            "status": "unverified",
            "included_in_similarity": False,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    files_path = output_dir / "files.jsonl"
    files_bytes = b"".join(
        (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for record in records
    )
    atomic_write_bytes(files_path, files_bytes)
    summary_path = output_dir / "summary.json"
    atomic_write_json(summary_path, summary)
    implementation_root = Path(__file__).parent
    manifest = {
        "schema_version": 1,
        "status": status,
        "inputs": {
            name: sha256_file(source_dir / name)
            for name in sorted(pieces, key=lambda value: (value.casefold(), value))
        },
        "implementations": {
            "reference_profile.py": sha256_file(implementation_root / "reference_profile.py"),
            "reference_controls.py": sha256_file(implementation_root / "reference_controls.py"),
        },
        "outputs": {
            "files.jsonl": sha256_file(files_path),
            "summary.json": sha256_file(summary_path),
        },
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build reference profile v1")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--neighbor-count", type=int, default=7)
    args = parser.parse_args(argv)
    result = build_reference_profile_v1(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        neighbor_count=args.neighbor_count,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "source_count": result["source_count"],
                "passing_group_count": result["controls"]["passing_group_count"],
                "default_reference": result["default_reference"]["selected"]["name"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
