"""参照曲の構造候補を拍に依存せず、複数解像度で記述する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from llm_musical_composer import smf_notes as smf_notes_module
from llm_musical_composer import structure_features as structure_features_module
from llm_musical_composer.pilot_features import NoteEvent
from llm_musical_composer.run_state import atomic_write_bytes, atomic_write_json, sha256_file
from llm_musical_composer.smf_notes import load_smf_notes
from llm_musical_composer.structure_features import extract_structure_features

DEFAULT_RESOLUTIONS = (4, 8, 16, 32)
DEFAULT_EXCLUSIONS = ("aimusic01.mid", "rut.mid")
BOUNDARY_TOLERANCE = 1 / 16
COORDINATES = ("elapsed_time", "onset_order")


def _round(value: float) -> float:
    return round(float(value), 8)


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return _round(ordered[0])
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return _round(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"available_count": 0, "status": "unable_to_investigate"}
    return {
        "available_count": len(values),
        "p25": _percentile(values, 0.25),
        "median": _percentile(values, 0.5),
        "p75": _percentile(values, 0.75),
    }


def _unavailable(detail: str) -> dict[str, Any]:
    return {"status": "unable_to_investigate", "detail": detail, "windows": []}


def _compact_report(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("status") != "pass":
        return report
    return {
        "status": "pass",
        "window_count": report["window_count"],
        "coordinate": report["coordinate"],
        "phase_fraction": report["phase_fraction"],
        "windows": report["windows"],
        "novelty": report["novelty"],
        "repetition": report["repetition"],
        "activity": report["activity"],
        "high_activity_candidates": report["high_activity_candidates"],
    }


def _resolution_report(
    notes: list[NoteEvent], *, resolution: int, coordinate: str
) -> dict[str, Any]:
    unique_onsets = {note.onset_ms for note in notes if note.duration_ms > 0}
    if len(unique_onsets) < resolution:
        return _unavailable(
            f"unique onset count {len(unique_onsets)} is below window count {resolution}"
        )
    report = extract_structure_features(
        notes,
        window_count=resolution,
        coordinate=coordinate,
    )
    if report.get("status") != "pass":
        return report
    empty_windows = [
        window["index"] for window in report["windows"] if window["features"]["onset_count"] == 0
    ]
    if empty_windows:
        return {
            **_unavailable("one or more windows contain no onsets"),
            "empty_window_indices": empty_windows,
        }
    return _compact_report(report)


def _persistent_boundaries(coordinates: dict[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for coordinate_name, coordinate in coordinates.items():
        for resolution_name, report in coordinate["resolutions"].items():
            if report.get("status") != "pass":
                continue
            for candidate in report["novelty"]["consensus_candidates"]:
                evidence.append(
                    {
                        "coordinate": coordinate_name,
                        "resolution": int(resolution_name),
                        "position": float(candidate["position"]),
                        "supporting_metrics": candidate["supporting_metrics"],
                    }
                )
    clusters: list[list[dict[str, Any]]] = []
    for item in sorted(
        evidence,
        key=lambda value: (value["position"], value["coordinate"], value["resolution"]),
    ):
        if (
            clusters
            and abs(
                item["position"] - statistics.mean(member["position"] for member in clusters[-1])
            )
            <= BOUNDARY_TOLERANCE
        ):
            clusters[-1].append(item)
        else:
            clusters.append([item])
    persistent: list[dict[str, Any]] = []
    for cluster in clusters:
        resolutions = sorted({item["resolution"] for item in cluster})
        coordinate_names = sorted({item["coordinate"] for item in cluster})
        if len(resolutions) < 2 and len(coordinate_names) < 2:
            continue
        persistent.append(
            {
                "position": _round(statistics.median(item["position"] for item in cluster)),
                "resolution_count": len(resolutions),
                "resolutions": resolutions,
                "coordinate_count": len(coordinate_names),
                "coordinates": coordinate_names,
                "evidence": cluster,
            }
        )
    return persistent


def analyze_note_structure(
    notes: Iterable[NoteEvent], *, resolutions: Sequence[int] = DEFAULT_RESOLUTIONS
) -> dict[str, Any]:
    """一曲を二座標かつ複数解像度で記述する。"""
    normalized = [
        NoteEvent(note.pitch, note.onset_ms, note.duration_ms, note.velocity)
        for note in notes
        if note.duration_ms > 0
    ]
    normalized_resolutions = tuple(sorted(set(resolutions)))
    if not normalized_resolutions or any(resolution < 2 for resolution in normalized_resolutions):
        raise ValueError("resolutions must contain window counts of at least two")
    if not normalized:
        return {
            "status": "unable_to_investigate",
            "detail": "no complete pitched notes",
            "coordinates": {},
            "persistent_boundaries": [],
            "repetition_use": "diagnostic_only",
        }
    coordinates = {
        coordinate: {
            "resolutions": {
                str(resolution): _resolution_report(
                    normalized,
                    resolution=resolution,
                    coordinate=coordinate,
                )
                for resolution in normalized_resolutions
            }
        }
        for coordinate in COORDINATES
    }
    return {
        "status": "pass",
        "note_count": len(normalized),
        "unique_onset_count": len({note.onset_ms for note in normalized}),
        "duration_ms": max(note.onset_ms + note.duration_ms for note in normalized)
        - min(note.onset_ms for note in normalized),
        "coordinates": coordinates,
        "persistent_boundaries": _persistent_boundaries(coordinates),
        "repetition_use": "diagnostic_only",
        "repetition_detail": (
            "The existing repetition modalities are retained for diagnosis but are not "
            "generation targets because corpus validation found under-detection and saturation."
        ),
    }


def _candidate_positions(report: dict[str, Any]) -> list[float]:
    if report.get("status") != "pass":
        return []
    return [float(candidate["position"]) for candidate in report["novelty"]["consensus_candidates"]]


def _positions_match(
    first: Sequence[float], second: Sequence[float], *, tolerance: float = BOUNDARY_TOLERANCE
) -> bool:
    return (
        bool(first)
        and bool(second)
        and all(any(abs(left - right) <= tolerance for right in second) for left in first)
    )


def _control_notes() -> list[NoteEvent]:
    sections = (
        (0, [60, 64, 62, 67, 65, 69, 67, 72], 45, 120),
        (1_200, [48, 55, 60, 64, 67, 72, 76, 79], 105, 90),
        (2_400, [67, 71, 69, 74, 72, 76, 74, 79], 45, 120),
        (3_600, [55, 59, 57, 62, 60, 64, 62, 67], 75, 120),
    )
    return [
        NoteEvent(pitch, start + index * spacing, 220, velocity)
        for start, pitches, velocity, spacing in sections
        for index, pitch in enumerate(pitches)
    ]


def _observation_signature(result: dict[str, Any]) -> dict[str, Any]:
    return {
        coordinate: {
            resolution: {
                "boundaries": _candidate_positions(report),
                "peaks": {
                    name: series["peak_positions"]
                    for name, series in report.get("activity", {}).items()
                },
            }
            for resolution, report in value["resolutions"].items()
            if report.get("status") == "pass"
        }
        for coordinate, value in result["coordinates"].items()
    }


def run_reference_structure_controls() -> dict[str, dict[str, Any]]:
    """全曲分析の前に、承認済みの最小反証対照を実行する。"""
    notes = _control_notes()
    original = analyze_note_structure(notes, resolutions=(4, 8))
    stretched = analyze_note_structure(
        [
            NoteEvent(note.pitch, note.onset_ms * 2, note.duration_ms * 2, note.velocity)
            for note in notes
        ],
        resolutions=(4, 8),
    )
    transposed = analyze_note_structure(
        [
            NoteEvent(note.pitch + 7, note.onset_ms, note.duration_ms, note.velocity)
            for note in notes
        ],
        resolutions=(4, 8),
    )
    jittered = analyze_note_structure(
        [
            NoteEvent(
                note.pitch,
                note.onset_ms + ((index % 3) - 1) * 5,
                note.duration_ms,
                note.velocity,
            )
            for index, note in enumerate(notes)
        ],
        resolutions=(4, 8),
    )
    elapsed = extract_structure_features(notes, window_count=8)
    shifted = extract_structure_features(notes, window_count=8, phase_fraction=0.5)
    elapsed_positions = _candidate_positions(elapsed)
    shifted_positions = _candidate_positions(shifted)

    first_section = notes[:8]
    middle_section = notes[8:16]
    repeated_section = [
        NoteEvent(note.pitch + 7, note.onset_ms + 2_400, note.duration_ms, note.velocity)
        for note in first_section
    ]
    broken_section = [
        NoteEvent(60 + index * 2, note.onset_ms + 2_400, note.duration_ms, note.velocity)
        for index, note in enumerate(first_section)
    ]
    aba = extract_structure_features(
        first_section + middle_section + repeated_section, window_count=3
    )
    abc = extract_structure_features(
        first_section + middle_section + broken_section, window_count=3
    )

    reordered_notes = (
        [
            NoteEvent(note.pitch, note.onset_ms - 1_200, note.duration_ms, note.velocity)
            for note in notes[8:16]
        ]
        + [
            NoteEvent(note.pitch, note.onset_ms + 1_200, note.duration_ms, note.velocity)
            for note in notes[:8]
        ]
        + notes[16:]
    )
    reordered = analyze_note_structure(reordered_notes, resolutions=(4,))
    original_four = original["coordinates"]["elapsed_time"]["resolutions"]["4"]
    reordered_four = reordered["coordinates"]["elapsed_time"]["resolutions"]["4"]

    sparse = analyze_note_structure(
        [NoteEvent(60 + index, index * 1_000, 300, 64) for index in range(8)],
        resolutions=(4, 32),
    )
    uniform = analyze_note_structure(
        [NoteEvent(60, index * 100, 180, 64) for index in range(32)],
        resolutions=(4, 8),
    )
    truncated = analyze_note_structure(notes[:16], resolutions=(4, 8))
    doubled = analyze_note_structure(
        notes
        + [
            NoteEvent(note.pitch, note.onset_ms + 4_800, note.duration_ms, note.velocity)
            for note in notes
        ],
        resolutions=(4, 8),
    )
    form_candidate_counts = [
        len(result["persistent_boundaries"]) for result in (uniform, truncated, doubled)
    ]
    forced_form_absent = form_candidate_counts[0] == 0 and len(set(form_candidate_counts)) > 1

    original_signature = _observation_signature(original)
    outcomes = {
        "time_stretch": original_signature == _observation_signature(stretched),
        "transposition": original_signature == _observation_signature(transposed),
        "onset_jitter": all(
            _positions_match(
                value[resolution]["boundaries"],
                _observation_signature(jittered)[coordinate][resolution]["boundaries"],
            )
            for coordinate, value in original_signature.items()
            for resolution in value
            if value[resolution]["boundaries"]
        ),
        "half_window_phase": _positions_match(
            elapsed_positions, shifted_positions, tolerance=0.125
        ),
        "repetition_break": (
            aba["similarity_matrices"]["pitch_interval"][0][2]
            > abc["similarity_matrices"]["pitch_interval"][0][2]
        ),
        "section_reorder": (
            original_four["activity"]["velocity"]["peak_positions"]
            != reordered_four["activity"]["velocity"]["peak_positions"]
        ),
        "no_forced_four_part_form": forced_form_absent,
        "sparse_high_resolution": all(
            coordinate["resolutions"]["32"]["status"] == "unable_to_investigate"
            for coordinate in sparse["coordinates"].values()
        ),
    }
    return {
        name: {
            "status": "pass" if passed else "fail",
            "detail": "counterexample behaved as expected"
            if passed
            else "counterexample did not produce the expected evidence",
        }
        for name, passed in outcomes.items()
    }


NoteLoader = Callable[[Path], Iterable[Any]]


def analyze_reference_files(
    paths: Iterable[Path],
    *,
    note_loader: NoteLoader = load_smf_notes,
    resolutions: Sequence[int] = DEFAULT_RESOLUTIONS,
) -> list[dict[str, Any]]:
    """列挙順に依存せず、失敗を明示した曲別記録を返す。"""
    records: list[dict[str, Any]] = []
    for path in sorted((Path(path) for path in paths), key=lambda item: item.name.casefold()):
        try:
            analysis = analyze_note_structure(note_loader(path), resolutions=resolutions)
            records.append({"name": path.name, **analysis, "error": None})
        except Exception as error:
            records.append(
                {
                    "name": path.name,
                    "status": "unable_to_investigate",
                    "coordinates": {},
                    "persistent_boundaries": [],
                    "repetition_use": "diagnostic_only",
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            )
    return records


def _build_summary(
    records: Sequence[dict[str, Any]],
    *,
    excluded_files: Sequence[str],
    resolutions: Sequence[int],
) -> dict[str, Any]:
    passed = [record for record in records if record["status"] == "pass"]
    persistent_counts = [float(len(record["persistent_boundaries"])) for record in passed]
    unable_counts = []
    boundary_positions: list[float] = []
    activity_peak_positions: list[float] = []
    for record in passed:
        unable_counts.append(
            float(
                sum(
                    report["status"] != "pass"
                    for coordinate in record["coordinates"].values()
                    for report in coordinate["resolutions"].values()
                )
            )
        )
        boundary_positions.extend(
            float(boundary["position"]) for boundary in record["persistent_boundaries"]
        )
        for coordinate in record["coordinates"].values():
            for report in coordinate["resolutions"].values():
                if report["status"] != "pass":
                    continue
                for series in report["activity"].values():
                    activity_peak_positions.extend(
                        float(value) for value in series["peak_positions"]
                    )
    return {
        "schema_version": 1,
        "source_count": len(records),
        "pass_count": len(passed),
        "unable_to_investigate_count": len(records) - len(passed),
        "excluded_files": sorted(excluded_files, key=str.casefold),
        "coordinates": list(COORDINATES),
        "resolutions": sorted(set(resolutions)),
        "boundary_tolerance": BOUNDARY_TOLERANCE,
        "persistent_boundary_count": _distribution(persistent_counts),
        "persistent_boundary_position": _distribution(boundary_positions),
        "activity_peak_position": _distribution(activity_peak_positions),
        "unavailable_resolution_count": _distribution(unable_counts),
        "repetition_use": "diagnostic_only",
    }


def _jsonl_bytes(records: Sequence[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")


def write_reference_structure_analysis(
    input_dir: Path,
    output_dir: Path,
    *,
    note_loader: NoteLoader = load_smf_notes,
    resolutions: Sequence[int] = DEFAULT_RESOLUTIONS,
    exclusions: Sequence[str] = DEFAULT_EXCLUSIONS,
) -> dict[str, Any]:
    """基本対照を通過した設定だけで参照曲を解析し、決定的に保存する。"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    controls = run_reference_structure_controls()
    critical = set(controls) - {"repetition_break"}
    failures = [name for name in sorted(critical) if controls[name]["status"] != "pass"]
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "controls.json", controls)
    if failures:
        raise RuntimeError(f"reference structure controls failed: {', '.join(failures)}")

    excluded_lookup = {name.casefold() for name in exclusions}
    candidates = sorted(input_dir.glob("*.mid"), key=lambda path: path.name.casefold())
    selected = [path for path in candidates if path.name.casefold() not in excluded_lookup]
    present_exclusions = [
        path.name for path in candidates if path.name.casefold() in excluded_lookup
    ]
    records = analyze_reference_files(
        selected,
        note_loader=note_loader,
        resolutions=resolutions,
    )
    summary = _build_summary(
        records,
        excluded_files=present_exclusions,
        resolutions=resolutions,
    )
    atomic_write_bytes(output_dir / "files.jsonl", _jsonl_bytes(records))
    atomic_write_json(output_dir / "summary.json", summary)
    artifact_names = ("controls.json", "files.jsonl", "summary.json")
    manifest = {
        "schema_version": 1,
        "configuration": {
            "coordinates": list(COORDINATES),
            "resolutions": sorted(set(resolutions)),
            "boundary_tolerance": BOUNDARY_TOLERANCE,
            "exclusions": sorted(exclusions, key=str.casefold),
        },
        "source_files": [{"name": path.name, "sha256": sha256_file(path)} for path in selected],
        "artifact_sha256": {name: sha256_file(output_dir / name) for name in artifact_names},
        "implementation_sha256": {
            "reference_structure.py": sha256_file(Path(__file__)),
            "smf_notes.py": sha256_file(Path(smf_notes_module.__file__)),
            "structure_features.py": sha256_file(Path(structure_features_module.__file__)),
        },
        "configuration_sha256": hashlib.sha256(
            json.dumps(
                {
                    "coordinates": COORDINATES,
                    "resolutions": sorted(set(resolutions)),
                    "boundary_tolerance": BOUNDARY_TOLERANCE,
                    "exclusions": sorted(exclusions, key=str.casefold),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="参照SMFの長期構造候補を記述します。")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    summary = write_reference_structure_analysis(args.input_dir, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
