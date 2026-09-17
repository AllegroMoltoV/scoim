"""参照SMF上の4項目を測り、コーパスの最小値・最大値で正規化する。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from llm_musical_composer.control_axis_anchors import extract_control_axis_descriptors
from llm_musical_composer.reference_profile import ReferencePiece, load_reference_piece

AXIS_VERSION = "four-control-corpus-minmax-v3"
ATTACK_GROUP_TOLERANCE_MS = 30
AXES = ("あかるさ", "高さ", "重なり", "発音頻度")
EXCLUDED_NAMES = frozenset({"rut.mid", "aimusic01.mid"})
AXIS_CONTRACTS: dict[str, dict[str, object]] = {
    "あかるさ": {
        "input_domain": [-1, 0, 1],
        "input_kind": "integer",
        "positive_direction": "brighter",
        "raw_metric": "major_minor_profile_margin",
        "unit": "correlation_difference",
    },
    "高さ": {
        "input_domain": [-1.0, 1.0],
        "input_kind": "continuous",
        "positive_direction": "higher",
        "raw_metric": "mean_midi_pitch",
        "unit": "midi_note_number",
    },
    "重なり": {
        "input_domain": [-1.0, 1.0],
        "input_kind": "continuous",
        "positive_direction": "more_active_notes",
        "raw_metric": "mean_active_polyphony",
        "unit": "active_notes",
    },
    "発音頻度": {
        "input_domain": [-1.0, 1.0],
        "input_kind": "continuous",
        "positive_direction": "more_attacks_per_second",
        "raw_metric": "attack_group_count_per_second_30ms",
        "unit": "attacks_per_second",
    },
}


class BaselineError(ValueError):
    """4項目基準を安全に構築できない場合のエラー。"""


@dataclass(frozen=True)
class ReferenceInput:
    """manifestで固定された参照SMF。"""

    name: str
    path: Path
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sort_name(name: str) -> tuple[str, str]:
    return name.casefold(), name


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselineError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise BaselineError(f"{label} must be finite")
    return number


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise BaselineError("percentile requires at least one value")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _ratio_distribution(counts: Mapping[int, int], total: int) -> dict[str, float]:
    if total <= 0:
        raise BaselineError("distribution support must be positive")
    return {str(key): counts[key] / total for key in sorted(counts)}


def _active_polyphony(notes: Sequence[Any], start: int, end: int) -> tuple[dict[str, float], float]:
    if end <= start:
        raise BaselineError("piece duration must be positive")
    deltas: dict[int, int] = {}
    for note in notes:
        deltas[note.onset_ms] = deltas.get(note.onset_ms, 0) + 1
        release = note.onset_ms + note.duration_ms
        deltas[release] = deltas.get(release, 0) - 1
    durations: Counter[int] = Counter()
    active = 0
    previous = start
    for at_ms in sorted(deltas):
        if at_ms < start or at_ms > end:
            continue
        if at_ms > previous:
            durations[active] += at_ms - previous
        active += deltas[at_ms]
        if active < 0:
            raise BaselineError("active polyphony became negative")
        previous = at_ms
    if previous < end:
        durations[active] += end - previous
    total = end - start
    distribution = _ratio_distribution(durations, total)
    mean = sum(polyphony * duration for polyphony, duration in durations.items()) / total
    return distribution, mean


def _group_attacks(
    notes: Sequence[Any], *, tolerance_ms: int = ATTACK_GROUP_TOLERANCE_MS
) -> list[list[Any]]:
    """群先頭から許容幅内の整数ms発音をまとめる。連鎖結合はしない。"""

    if tolerance_ms < 0:
        raise BaselineError("attack group tolerance must not be negative")
    groups: list[list[Any]] = []
    anchor: int | None = None
    current: list[Any] = []
    for note in sorted(
        notes,
        key=lambda item: (item.onset_ms, item.pitch, item.duration_ms, item.velocity),
    ):
        if anchor is None or note.onset_ms - anchor > tolerance_ms:
            if current:
                groups.append(current)
            anchor = note.onset_ms
            current = [note]
        else:
            current.append(note)
    if current:
        groups.append(current)
    return groups


def extract_control_observables(piece: ReferencePiece) -> dict[str, Any]:
    """1曲から4項目の生値と、外れ値検査用の診断値を抽出する。"""

    notes = sorted(
        piece.notes,
        key=lambda note: (note.onset_ms, note.pitch, note.duration_ms, note.velocity),
    )
    if not notes:
        raise BaselineError("piece must contain notes")
    exact_attacks: dict[int, list[Any]] = {}
    for note in notes:
        exact_attacks.setdefault(note.onset_ms, []).append(note)
    if len(exact_attacks) < 2:
        raise BaselineError("piece must contain at least two attacks")
    attack_groups = _group_attacks(notes)
    start = min(note.onset_ms for note in notes)
    end = max(note.onset_ms + note.duration_ms for note in notes)
    if end <= start:
        raise BaselineError("piece duration must be positive")
    duration_seconds = (end - start) / 1000

    brightness_axis = extract_control_axis_descriptors(piece)["axes"]["あかるさ"]
    brightness_metric = brightness_axis["metrics"]["major_minor_profile_margin"]
    if brightness_metric.get("status") != "available":
        raise BaselineError("brightness metric is unavailable")
    brightness = _finite_number(brightness_metric.get("value"), "brightness")
    modal_metric = brightness_axis["metrics"]["modal_degree_balance"]

    pitches = [float(note.pitch) for note in notes]
    attack_sizes = Counter(len(group) for group in attack_groups)
    polyphony_distribution, mean_polyphony = _active_polyphony(notes, start, end)
    attack_times = [group[0].onset_ms for group in attack_groups]
    iois = [float(second - first) for first, second in pairwise(attack_times)]

    return {
        "name": piece.name,
        "raw": {
            "あかるさ": brightness,
            "高さ": statistics.fmean(pitches),
            "重なり": mean_polyphony,
            "発音頻度": len(attack_groups) / duration_seconds,
        },
        "diagnostics": {
            "attack_group_count": len(attack_groups),
            "duration_seconds": duration_seconds,
            "exact_onset_count": len(exact_attacks),
            "note_count": len(notes),
            "brightness": {
                "estimated_mode": brightness_axis["confidence"]["estimated_mode"],
                "estimated_tonic_pitch_class": brightness_axis["confidence"][
                    "estimated_tonic_pitch_class"
                ],
                "modal_degree_balance": (
                    modal_metric.get("value") if modal_metric.get("status") == "available" else None
                ),
                "modal_degree_balance_status": modal_metric.get("status"),
                "tonal_profile_correlation": brightness_axis["confidence"][
                    "tonal_profile_correlation"
                ],
            },
            "height": {
                "maximum_midi_pitch": max(pitches),
                "median_midi_pitch": statistics.median(pitches),
                "minimum_midi_pitch": min(pitches),
                "p10_midi_pitch": _percentile(pitches, 0.10),
                "p90_midi_pitch": _percentile(pitches, 0.90),
            },
            "overlap": {
                "active_polyphony_duration_distribution": polyphony_distribution,
                "attack_size_distribution": _ratio_distribution(attack_sizes, len(attack_groups)),
                "mean_active_polyphony": mean_polyphony,
                "notes_per_attack": len(notes) / len(attack_groups),
            },
            "attack_frequency": {
                "attack_group_tolerance_ms": ATTACK_GROUP_TOLERANCE_MS,
                "median_ioi_ms": statistics.median(iois) if iois else None,
                "notes_per_second": len(notes) / duration_seconds,
            },
        },
        "unavailable_features": [
            {
                "id": "voice_specific_pitch_center",
                "reason": "track/channel/voice labels are not present in ReferencePiece",
                "status": "unavailable",
            }
        ],
    }


def normalize_value(value: float, *, minimum: float, maximum: float) -> float:
    """コーパスの最小値を-1、最大値を1へ写す。範囲外は切り詰めない。"""

    value = _finite_number(value, "value")
    minimum = _finite_number(minimum, "minimum")
    maximum = _finite_number(maximum, "maximum")
    if maximum <= minimum:
        raise BaselineError("axis has zero range or reversed endpoints")
    if value == minimum:
        return -1.0
    if value == maximum:
        return 1.0
    return -1 + 2 * (value - minimum) / (maximum - minimum)


def build_axis_summary(records: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    """全参照曲の項目別端点と分布要約を作る。"""

    if not records:
        raise BaselineError("at least one record is required")
    result: dict[str, Any] = {}
    for axis in AXES:
        rows: list[tuple[str, float]] = []
        for index, record in enumerate(records):
            name = record.get("name")
            raw = record.get("raw")
            if not isinstance(name, str) or not isinstance(raw, Mapping):
                raise BaselineError(f"record {index} has invalid name or raw values")
            rows.append((name, _finite_number(raw.get(axis), f"{name}:{axis}")))
        values = [value for _, value in rows]
        minimum = min(values)
        maximum = max(values)
        if maximum <= minimum:
            raise BaselineError(f"axis {axis} has zero range")
        result[axis] = {
            **AXIS_CONTRACTS[axis],
            "minimum": minimum,
            "maximum": maximum,
            "endpoints": {
                "minimum": sorted(
                    [name for name, value in rows if value == minimum], key=_sort_name
                ),
                "maximum": sorted(
                    [name for name, value in rows if value == maximum], key=_sort_name
                ),
            },
            "distribution": {
                "minimum": minimum,
                "p10": _percentile(values, 0.10),
                "p25": _percentile(values, 0.25),
                "median": _percentile(values, 0.50),
                "mean": statistics.fmean(values),
                "p75": _percentile(values, 0.75),
                "p90": _percentile(values, 0.90),
                "maximum": maximum,
            },
        }
    return result


def normalize_record(
    record: Mapping[str, object],
    summary: Mapping[str, Mapping[str, object]],
    *,
    observation: bool = False,
) -> dict[str, Any]:
    """1曲の生値を基準へ写し、観測対象なら範囲外区分も付ける。"""

    prepared = copy.deepcopy(dict(record))
    raw = prepared.get("raw")
    if not isinstance(raw, Mapping):
        raise BaselineError("record raw values are required")
    normalized: dict[str, float] = {}
    range_status: dict[str, str] = {}
    for axis in AXES:
        contract = summary.get(axis)
        if not isinstance(contract, Mapping):
            raise BaselineError(f"summary is missing axis {axis}")
        minimum = _finite_number(contract.get("minimum"), f"{axis}:minimum")
        maximum = _finite_number(contract.get("maximum"), f"{axis}:maximum")
        value = _finite_number(raw.get(axis), f"{axis}:value")
        normalized[axis] = normalize_value(value, minimum=minimum, maximum=maximum)
        if observation:
            range_status[axis] = (
                "below" if value < minimum else "above" if value > maximum else "within"
            )
    prepared["normalized"] = normalized
    if observation:
        prepared["range_status"] = range_status
    return prepared


def validate_reference_inputs(
    manifest: Mapping[str, object],
    source_dir: Path,
    *,
    expected_count: int = 232,
) -> tuple[ReferenceInput, ...]:
    """manifestの入力だけを、件数・除外名・存在・SHA-256で固定する。"""

    raw_inputs = manifest.get("inputs")
    if not isinstance(raw_inputs, Mapping):
        raise BaselineError("manifest inputs must be an object")
    if len(raw_inputs) != expected_count:
        raise BaselineError(f"manifest input count must be {expected_count}, got {len(raw_inputs)}")
    excluded = {name.casefold() for name in EXCLUDED_NAMES}
    prepared: list[ReferenceInput] = []
    for name in sorted(raw_inputs, key=lambda item: _sort_name(str(item))):
        expected_hash = raw_inputs[name]
        if not isinstance(name, str) or Path(name).name != name or Path(name).is_absolute():
            raise BaselineError("manifest contains an unsafe input name")
        if name.casefold() in excluded:
            raise BaselineError(f"manifest contains excluded input: {name}")
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in expected_hash)
        ):
            raise BaselineError(f"manifest hash is invalid: {name}")
        path = source_dir / name
        if not path.is_file():
            raise BaselineError(f"input file is missing: {name}")
        actual_hash = _sha256(path)
        if actual_hash.casefold() != expected_hash.casefold():
            raise BaselineError(f"input hash mismatch: {name}")
        prepared.append(ReferenceInput(name, path, actual_hash))
    return tuple(prepared)


def _rounded_json(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BaselineError("JSON values must be finite")
        rounded = round(value, 8)
        return 0.0 if rounded == 0 else rounded
    if isinstance(value, Mapping):
        return {str(key): _rounded_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded_json(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    text = json.dumps(
        _rounded_json(payload),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    ordered = sorted(rows, key=lambda row: _sort_name(str(row["name"])))
    lines = [
        json.dumps(
            _rounded_json(row),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        for row in ordered
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")


def write_baseline_artifacts(
    output_dir: Path,
    *,
    records: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    observations: Sequence[Mapping[str, object]],
    provenance: Mapping[str, object],
) -> dict[str, Any]:
    """決定的なJSON成果物4点を書き出す。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    summary_path = output_dir / "summary.json"
    observations_path = output_dir / "observations.jsonl"
    manifest_path = output_dir / "manifest.json"
    _write_jsonl(records_path, records)
    _write_json(
        summary_path,
        {
            "schema_version": 1,
            "axis_version": provenance.get("axis_version", AXIS_VERSION),
            "record_count": len(records),
            "axes": summary,
        },
    )
    _write_jsonl(observations_path, observations)
    outputs = {path.name: _sha256(path) for path in (records_path, summary_path, observations_path)}
    manifest = {
        "schema_version": 1,
        "status": "complete",
        **dict(provenance),
        "outputs": outputs,
    }
    _write_json(manifest_path, manifest)
    return manifest


def build_control_reference_baseline(
    *,
    source_dir: Path,
    reference_manifest_path: Path,
    output_dir: Path,
    observation_paths: Sequence[Path] = (),
    implementation_paths: Sequence[Path] = (),
    expected_count: int = 232,
) -> dict[str, Any]:
    """参照コーパスと任意の候補SMFから4項目基準成果物を構築する。"""

    try:
        reference_manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BaselineError(f"unable to read reference manifest: {error}") from error
    if not isinstance(reference_manifest, Mapping):
        raise BaselineError("reference manifest must be an object")
    inputs = validate_reference_inputs(
        reference_manifest, source_dir, expected_count=expected_count
    )
    raw_records = [extract_control_observables(load_reference_piece(item.path)) for item in inputs]
    summary = build_axis_summary(raw_records)
    records = [normalize_record(record, summary) for record in raw_records]

    observations: list[dict[str, Any]] = []
    observation_hashes: dict[str, str] = {}
    for path in sorted(observation_paths, key=lambda item: _sort_name(str(item))):
        if not path.is_file():
            raise BaselineError(f"observation file is missing: {path}")
        if path.name in observation_hashes:
            raise BaselineError(f"duplicate observation name: {path.name}")
        observation_hashes[path.name] = _sha256(path)
        observations.append(
            normalize_record(
                extract_control_observables(load_reference_piece(path)),
                summary,
                observation=True,
            )
        )

    implementations: dict[str, str] = {}
    for path in sorted(implementation_paths, key=lambda item: str(item).casefold()):
        if not path.is_file():
            raise BaselineError(f"implementation file is missing: {path}")
        try:
            name = path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            name = path.resolve().as_posix()
        implementations[name] = _sha256(path)
    provenance = {
        "axis_version": AXIS_VERSION,
        "environment": {
            "mido": importlib.metadata.version("mido"),
            "python": platform.python_version(),
        },
        "implementations": implementations,
        "inputs": {item.name: item.sha256 for item in inputs},
        "observations": observation_hashes,
        "reference_manifest_sha256": _sha256(reference_manifest_path),
    }
    return write_baseline_artifacts(
        output_dir,
        records=records,
        summary=summary,
        observations=observations,
        provenance=provenance,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--observation", action="append", default=[], type=Path)
    parser.add_argument("--implementation", action="append", default=[], type=Path)
    parser.add_argument("--expected-count", default=232, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI入口。"""

    arguments = _parser().parse_args(argv)
    manifest = build_control_reference_baseline(
        source_dir=arguments.source_dir,
        reference_manifest_path=arguments.reference_manifest,
        output_dir=arguments.output_dir,
        observation_paths=arguments.observation,
        implementation_paths=arguments.implementation,
        expected_count=arguments.expected_count,
    )
    print(
        json.dumps(
            {
                "axis_version": manifest["axis_version"],
                "input_count": len(manifest["inputs"]),
                "observation_count": len(manifest["observations"]),
                "outputs": manifest["outputs"],
                "status": manifest["status"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
