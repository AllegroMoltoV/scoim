"""3分曲全体の`はげしさ`低・中・高候補を同じ構成から生成する。"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from llm_musical_composer.composition_ir import Composition, Material
from llm_musical_composer.intensity_control import (
    extract_composition_intensity,
    extract_material_intensity,
    load_reference_records,
    resolve_intensity_target,
)
from llm_musical_composer.intensity_iteration import (
    _verify_records_manifest,
    evaluate_upper_delay_artifact,
    load_prompt,
)
from llm_musical_composer.intensity_patch import (
    IntensityPatchError,
    apply_material_patch,
    extend_lower_note_durations,
    plan_material_intensity_patch,
    shorten_same_pitch_overlaps,
)
from llm_musical_composer.long_form_evaluation import evaluate_long_form_structure
from llm_musical_composer.long_form_generation import (
    LongFormGenerationError,
    _validate_natural_materials,
    composition_to_source,
)
from llm_musical_composer.music_dsl import THREE_MINUTE_POLICY, DslError, parse_composition
from llm_musical_composer.piano_texture import (
    evaluate_piano_texture,
    evaluate_variation_contracts,
)
from llm_musical_composer.pilot_loop import (
    MODEL_ID,
)
from llm_musical_composer.run_state import (
    RunLock,
    RunStore,
    atomic_write_bytes,
    atomic_write_json,
    sha256_file,
)
from llm_musical_composer.smf_render import render_composition
from llm_musical_composer.sustain_profile import evaluate_sustain_profile


class FullIntensityIterationError(ValueError):
    """全曲強度対照が入力または固定契約に違反した場合の例外。"""


DRIVER_AXES = ("note_rate_hz", "attack_rate_hz", "velocity_level")
ENERGY_AXES = ("note_density", "velocity_median", "polyphony_mean")


def _finite_number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FullIntensityIterationError(f"{location} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise FullIntensityIterationError(f"{location} must be finite")
    return number


def _relative_acceptance(center: float, target: float) -> dict[str, float]:
    delta = target - center
    if math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return {"minimum": round(center, 8), "maximum": round(center, 8)}
    midpoint = center + delta * 0.5
    outer = target + delta * 0.5
    return {
        "minimum": round(max(0.0, min(midpoint, outer)), 8),
        "maximum": round(min(1.0, max(midpoint, outer)), 8)
        if 0 <= center <= 1 and 0 <= target <= 1
        else round(max(midpoint, outer), 8),
    }


def _allocate_expanded_counts(
    raw: Mapping[str, float],
    usage_counts: Mapping[str, int],
    *,
    desired_total: int,
    minimum: int | Mapping[str, int],
    maximum: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """再利用回数を含む全曲合計へ、素材ごとの整数件数を決定的に合わせる。"""

    minimums = (
        {material_id: minimum for material_id in raw} if isinstance(minimum, int) else dict(minimum)
    )
    allocation = {
        material_id: max(
            minimums[material_id],
            min(round(value), maximum[material_id]) if maximum is not None else round(value),
        )
        for material_id, value in raw.items()
    }
    total = sum(allocation[name] * usage_counts[name] for name in allocation)
    guard = 0
    while total != desired_total:
        guard += 1
        if guard > 100_000:
            raise FullIntensityIterationError("unable to allocate expanded intensity counts")
        direction = 1 if total < desired_total else -1
        remaining = abs(desired_total - total)
        candidates = []
        for name in allocation:
            weight = usage_counts[name]
            if weight > remaining:
                continue
            proposed = allocation[name] + direction
            if proposed < minimums[name]:
                continue
            if maximum is not None and proposed > maximum[name]:
                continue
            residual = abs(proposed - raw[name])
            candidates.append((weight != 1, residual, name.casefold(), name))
        if not candidates:
            raise FullIntensityIterationError("expanded intensity target is not representable")
        name = min(candidates)[3]
        allocation[name] += direction
        total += direction * usage_counts[name]
    return allocation


def build_center_material_targets(
    composition: Composition,
    center_target: Mapping[str, Any],
    *,
    preserve_derived_upper_onsets: bool = False,
) -> dict[str, dict[str, Any]]:
    """全曲の中央差を各素材の元値へ適用し、energy順位を平坦化しない。"""

    if center_target.get("status") != "reachable" or center_target.get("value") != 0.0:
        raise FullIntensityIterationError("center target must be reachable and have value 0")
    try:
        requested = {
            axis: _finite_number(center_target["targets"][axis], f"center target {axis}")
            for axis in DRIVER_AXES
        }
        global_hold = center_target["holds"]["mean_active_polyphony"]
        hold_minimum = _finite_number(global_hold["minimum"], "center polyphony minimum")
        hold_maximum = _finite_number(global_hold["maximum"], "center polyphony maximum")
    except (KeyError, TypeError) as error:
        raise FullIntensityIterationError("center target is invalid") from error
    baseline = extract_composition_intensity(composition)["observables"]
    if baseline["note_rate_hz"] <= 0 or baseline["attack_rate_hz"] <= 0:
        raise FullIntensityIterationError("baseline intensity rates must be positive")
    if baseline["mean_active_polyphony"] <= 0:
        raise FullIntensityIterationError("baseline polyphony must be positive")
    note_ratio = requested["note_rate_hz"] / baseline["note_rate_hz"]
    attack_ratio = requested["attack_rate_hz"] / baseline["attack_rate_hz"]
    velocity_delta = requested["velocity_level"] - baseline["velocity_level"]
    hold_factors = (
        hold_minimum / baseline["mean_active_polyphony"],
        hold_maximum / baseline["mean_active_polyphony"],
    )

    reports = {
        material.material_id: extract_material_intensity(material)
        for material in composition.materials
    }
    protected_upper_ids = {
        material.material_id
        for material in composition.materials
        if material.derived_from is not None
    } | {
        material.derived_from
        for material in composition.materials
        if material.derived_from is not None
    }
    usage_counts = {
        material.material_id: sum(
            use.material_id == material.material_id for use in composition.form
        )
        for material in composition.materials
    }
    raw_note_counts = {
        material_id: report["note_count"] * note_ratio for material_id, report in reports.items()
    }
    desired_note_total = round(requested["note_rate_hz"] * composition.body_duration_ms / 1_000)
    note_counts = _allocate_expanded_counts(
        raw_note_counts,
        usage_counts,
        desired_total=desired_note_total,
        minimum={
            material.material_id: (
                max(2, sum(note.voice == "upper" for note in material.notes) + 1)
                if preserve_derived_upper_onsets and material.material_id in protected_upper_ids
                else 2
            )
            for material in composition.materials
        },
    )
    raw_attack_counts = {
        material_id: report["attack_count"] * attack_ratio
        for material_id, report in reports.items()
    }
    desired_attack_total = round(requested["attack_rate_hz"] * composition.body_duration_ms / 1_000)
    attack_counts = _allocate_expanded_counts(
        raw_attack_counts,
        usage_counts,
        desired_total=desired_attack_total,
        minimum={
            material.material_id: (
                max(2, len({note.at_ms for note in material.notes if note.voice == "upper"}) + 1)
                if preserve_derived_upper_onsets and material.material_id in protected_upper_ids
                else 2
            )
            for material in composition.materials
        },
        maximum={
            material.material_id: note_counts[material.material_id]
            - (
                sum(note.voice == "upper" for note in material.notes)
                - len({note.at_ms for note in material.notes if note.voice == "upper"})
                if preserve_derived_upper_onsets and material.material_id in protected_upper_ids
                else 0
            )
            for material in composition.materials
        },
    )

    result: dict[str, dict[str, Any]] = {}
    for material in composition.materials:
        report = reports[material.material_id]
        observed = report["observables"]
        duration_seconds = material.duration_ms / 1_000
        targets = {
            "note_rate_hz": note_counts[material.material_id] / duration_seconds,
            "attack_rate_hz": attack_counts[material.material_id] / duration_seconds,
            "velocity_level": observed["velocity_level"] + velocity_delta,
        }
        if not 0 <= targets["velocity_level"] <= 1:
            raise FullIntensityIterationError(
                f"center velocity target is outside the range for {material.material_id}"
            )
        rounded = {axis: round(targets[axis], 8) for axis in DRIVER_AXES}
        result[material.material_id] = {
            "schema_version": 1,
            "status": "reachable",
            "reference": center_target.get("reference"),
            "value": 0.0,
            "baseline": report,
            "targets": rounded,
            "target_counts": {
                "note_count": note_counts[material.material_id],
                "attack_count": attack_counts[material.material_id],
            },
            "acceptance": {
                axis: _relative_acceptance(observed[axis], rounded[axis]) for axis in DRIVER_AXES
            },
            "holds": {
                "mean_active_polyphony": {
                    "minimum": round(observed["mean_active_polyphony"] * hold_factors[0], 8),
                    "maximum": round(observed["mean_active_polyphony"] * hold_factors[1], 8),
                }
            },
            "evidence": {
                "basis": "global center ratio or delta applied to this material baseline",
                "global_baseline": baseline,
                "global_center": requested,
            },
        }
    return result


def build_endpoint_material_targets(
    composition: Composition,
    records: list[dict[str, Any]],
    *,
    reference_name: str,
    value: float,
    max_notes: int = 950,
) -> dict[str, dict[str, Any]]:
    """全曲端点を先に固定し、再利用回数を含む整数件数を各素材へ配分する。"""

    if value not in (-1.0, 1.0):
        raise FullIntensityIterationError("endpoint value must be -1 or 1")
    global_target = resolve_intensity_target(
        records,
        reference_name=reference_name,
        value=value,
        max_notes=max_notes,
    )
    if global_target.get("status") != "reachable":
        raise FullIntensityIterationError("full intensity endpoint is unreachable")
    result = build_center_material_targets(
        composition,
        {**global_target, "value": 0.0},
        preserve_derived_upper_onsets=True,
    )
    for target in result.values():
        target["value"] = value
        target["evidence"]["basis"] = "global endpoint allocated to material baseline"
    return result


def _validate_material_revision(current: Material, revised: Material) -> None:
    """素材の固定範囲と、基準にない二声上の新規問題を拒否する。"""

    if revised.material_id != current.material_id:
        raise FullIntensityIterationError("intensity material ID changed")
    if revised.duration_ms != current.duration_ms:
        raise FullIntensityIterationError("intensity material duration changed")
    if revised.derived_from != current.derived_from:
        raise FullIntensityIterationError("intensity material derivation changed")
    if revised.pedals != current.pedals:
        raise FullIntensityIterationError("intensity material pedals changed")
    current_pitch_classes = {note.pitch % 12 for note in current.notes}
    revised_pitch_classes = {note.pitch % 12 for note in revised.notes}
    if not revised_pitch_classes or not revised_pitch_classes <= current_pitch_classes:
        raise FullIntensityIterationError("intensity material added an unsupported pitch-class")
    current_range = (
        min(note.pitch for note in current.notes),
        max(note.pitch for note in current.notes),
    )
    if any(not current_range[0] <= note.pitch <= current_range[1] for note in revised.notes):
        raise FullIntensityIterationError("intensity material changed the allowed pitch range")
    current_delay = evaluate_upper_delay_artifact(current.notes)
    revised_delay = evaluate_upper_delay_artifact(revised.notes)
    if current_delay["status"] == "pass" and revised_delay["status"] != "pass":
        raise FullIntensityIterationError(str(revised_delay["issue"]))


def generate_deterministic_intensity_candidate(
    composition: Composition,
    *,
    material_targets: Mapping[str, Mapping[str, Any]],
) -> Composition:
    """全素材へ同じ決定規則を適用し、外部呼び出しなしで候補を作る。"""

    if set(material_targets) != set(composition.material_by_id):
        raise FullIntensityIterationError("material targets do not match the composition")
    revised_materials = []
    protected_upper_ids = {
        material.material_id
        for material in composition.materials
        if material.derived_from is not None
    } | {
        material.derived_from
        for material in composition.materials
        if material.derived_from is not None
    }
    for material in composition.materials:
        target = material_targets[material.material_id]
        try:
            value = float(target.get("value", 0.0))
            patch = plan_material_intensity_patch(
                material,
                target_counts=target["target_counts"],
                target_velocity_level=target["targets"]["velocity_level"],
                preserve_all_upper=material.material_id in protected_upper_ids and value != 0.0,
            )
            revised = apply_material_patch(
                material,
                patch,
                target_counts=target["target_counts"],
            )
            duration_factor = 2.00 if value == -1.0 else 1.10
            revised = shorten_same_pitch_overlaps(
                extend_lower_note_durations(revised, factor=duration_factor)
            )
        except (KeyError, TypeError, IntensityPatchError) as error:
            raise FullIntensityIterationError(
                f"deterministic intensity transform failed for {material.material_id}: {error}"
            ) from error
        revised_materials.append(revised)
    candidate = replace(composition, materials=tuple(revised_materials))
    try:
        reparsed = parse_composition(
            composition_to_source(candidate),
            policy=THREE_MINUTE_POLICY,
            require_parts=True,
            require_section_contract=True,
        )
    except DslError as error:
        raise FullIntensityIterationError(
            f"assembled deterministic intensity candidate is invalid: {error}"
        ) from error
    if reparsed != candidate:
        raise FullIntensityIterationError(
            "deterministic intensity candidate changed during round-trip"
        )
    return candidate


def _triplet_acceptance(
    targets: Mapping[float, Mapping[str, Any]], axis: str, value: float
) -> dict[str, float]:
    low = float(targets[-1.0]["targets"][axis])
    center = float(targets[0.0]["targets"][axis])
    high = float(targets[1.0]["targets"][axis])
    if not low < center < high:
        raise FullIntensityIterationError(f"full intensity target {axis} is not monotone")
    if value == -1.0:
        minimum = max(0.0, low - (center - low) * 0.5)
        maximum = (low + center) / 2
    elif value == 0.0:
        minimum = (low + center) / 2
        maximum = (center + high) / 2
    elif value == 1.0:
        minimum = (center + high) / 2
        maximum = high + (high - center) * 0.5
        if axis == "velocity_level":
            maximum = min(1.0, maximum)
    else:  # pragma: no cover - caller fixes the triplet
        raise FullIntensityIterationError("unknown triplet value")
    return {"minimum": round(minimum, 8), "maximum": round(maximum, 8)}


def evaluate_full_intensity_distribution(
    compositions: Mapping[float, Composition], *, window_ms: int = 10_000
) -> dict[str, Any]:
    """低候補と高候補の差が一部区間だけに局在していないことを検査する。"""

    if window_ms <= 0:
        raise FullIntensityIterationError("distribution window must be positive")

    def windows(composition: Composition) -> list[tuple[int, int, int]]:
        result: list[list[tuple[int, int]]] = [
            [] for _ in range(math.ceil(composition.body_duration_ms / window_ms))
        ]
        offset = 0
        for use in composition.form:
            material = composition.material_by_id[use.material_id]
            for note in material.notes:
                result[(offset + note.at_ms) // window_ms].append(
                    (offset + note.at_ms, note.velocity)
                )
            offset += material.duration_ms
        return [
            (
                len(notes),
                len({onset for onset, _velocity in notes}),
                round(sum(velocity for _onset, velocity in notes) / len(notes)) if notes else 0,
            )
            for notes in result
        ]

    low = windows(compositions[-1.0])
    high = windows(compositions[1.0])
    changed = [
        index for index, pair in enumerate(zip(low, high, strict=True)) if pair[0] != pair[1]
    ]
    unchanged = [index for index in range(len(low)) if index not in changed]
    longest_unchanged = 0
    current = 0
    for index in range(len(low)):
        if index in unchanged:
            current += 1
            longest_unchanged = max(longest_unchanged, current)
        else:
            current = 0
    changed_ratio = len(changed) / len(low)
    status = "pass" if changed_ratio >= 0.80 and longest_unchanged <= 2 else "fail"
    return {
        "status": status,
        "window_ms": window_ms,
        "window_count": len(low),
        "changed_window_count": len(changed),
        "changed_ratio": round(changed_ratio, 8),
        "longest_unchanged_window_run": longest_unchanged,
        "changed_windows": changed,
        "unchanged_windows": unchanged,
    }


def evaluate_full_intensity_triplet(
    compositions: Mapping[float, Composition], targets: Mapping[float, Mapping[str, Any]]
) -> dict[str, Any]:
    """3分全体の低・中・高について方向、到達範囲、重なり保持を検査する。"""

    values = (-1.0, 0.0, 1.0)
    if set(compositions) != set(values) or set(targets) != set(values):
        raise FullIntensityIterationError("full triplet must contain exactly -1, 0, and 1")
    for value in values:
        if targets[value].get("status") != "reachable" or targets[value].get("value") != value:
            raise FullIntensityIterationError("full triplet target is invalid")
    reports = {value: extract_composition_intensity(compositions[value]) for value in values}
    issues: list[str] = []
    acceptances: dict[str, dict[str, dict[str, float]]] = {}
    for axis in DRIVER_AXES:
        observed = [reports[value]["observables"][axis] for value in values]
        axis_acceptance = {
            str(value): _triplet_acceptance(targets, axis, value) for value in values
        }
        acceptances[axis] = axis_acceptance
        if not observed[0] < observed[1] < observed[2]:
            issues.append(axis)
            continue
        for value, candidate in zip(values, observed, strict=True):
            accepted = axis_acceptance[str(value)]
            if not accepted["minimum"] <= candidate <= accepted["maximum"]:
                issues.append(axis)
                break
    for value in values:
        accepted = targets[value]["holds"]["mean_active_polyphony"]
        candidate = reports[value]["observables"]["mean_active_polyphony"]
        if not accepted["minimum"] <= candidate <= accepted["maximum"]:
            issues.append("mean_active_polyphony")
            break
    unique_issues = list(dict.fromkeys(issues))
    distribution = evaluate_full_intensity_distribution(compositions)
    if distribution["status"] != "pass":
        unique_issues.append("whole_song_distribution")
    return {
        "status": "pass" if not unique_issues else "fail",
        "issues": unique_issues,
        "candidates": {str(value): reports[value] for value in values},
        "acceptance": acceptances,
        "distribution": distribution,
    }


def evaluate_energy_order(composition: Composition) -> dict[str, Any]:
    """宣言energyが異なる大区分の実音順序を3軸中2軸で確認する。"""

    structure = evaluate_long_form_structure(composition)
    parts = structure.get("parts")
    if not isinstance(parts, list) or not parts:
        return {"status": "unable_to_investigate", "violations": [], "structure": structure}
    violations: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for left_index, left in enumerate(parts):
        for right in parts[left_index + 1 :]:
            if left["energy"] == right["energy"]:
                continue
            lower, higher = (left, right) if left["energy"] < right["energy"] else (right, left)
            supporting = [
                axis
                for axis in ENERGY_AXES
                if float(lower["features"][axis]) < float(higher["features"][axis])
            ]
            item = {
                "lower_part": lower["part_id"],
                "higher_part": higher["part_id"],
                "lower_energy": lower["energy"],
                "higher_energy": higher["energy"],
                "supporting_metrics": supporting,
            }
            comparisons.append(item)
            if len(supporting) < 2:
                violations.append(item)
    return {
        "status": "pass" if not violations else "fail",
        "violations": violations,
        "comparisons": comparisons,
        "structure": structure,
    }


def evaluate_single_full_intensity(
    composition: Composition,
    targets: Mapping[float, Mapping[str, Any]],
    *,
    value: float,
) -> dict[str, Any]:
    """後続候補を生成する前に一曲だけの到達範囲と重なりを検査する。"""

    if value not in (-1.0, 0.0, 1.0):
        raise FullIntensityIterationError("single intensity value is invalid")
    report = extract_composition_intensity(composition)
    issues: list[str] = []
    acceptance: dict[str, dict[str, float]] = {}
    for axis in DRIVER_AXES:
        accepted = _triplet_acceptance(targets, axis, value)
        acceptance[axis] = accepted
        observed = report["observables"][axis]
        if not accepted["minimum"] <= observed <= accepted["maximum"]:
            issues.append(axis)
    hold = targets[value]["holds"]["mean_active_polyphony"]
    polyphony = report["observables"]["mean_active_polyphony"]
    if not hold["minimum"] <= polyphony <= hold["maximum"]:
        issues.append("mean_active_polyphony")
    return {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "candidate": report,
        "acceptance": acceptance,
    }


def evaluate_full_candidate_quality(
    baseline: Composition, candidate: Composition, *, max_notes: int = 950
) -> dict[str, Any]:
    """はげしさ以外の構成、奏法、安全契約を一曲ごとに検査する。"""

    issues: list[str] = []
    if (
        candidate.title != baseline.title
        or candidate.form != baseline.form
        or candidate.parts != baseline.parts
        or candidate.phrases != baseline.phrases
    ):
        issues.append("structure changed")
    if (
        candidate.tonal_center != baseline.tonal_center
        or candidate.mode != baseline.mode
        or candidate.ending != baseline.ending
    ):
        issues.append("tonal context or ending changed")
    if set(candidate.material_by_id) != set(baseline.material_by_id):
        issues.append("material set changed")
    else:
        for material_id, current in baseline.material_by_id.items():
            try:
                _validate_material_revision(current, candidate.material_by_id[material_id])
            except ValueError as error:
                issues.append(str(error))
    try:
        duration_ms: int | None = candidate.duration_ms
        note_count: int | None = candidate.note_count
    except KeyError:
        duration_ms = None
        note_count = None
        issues.append("composition references a missing material")
    if duration_ms != 180_000:
        issues.append("duration is not 180000 ms")
    if note_count is not None and note_count > max_notes:
        issues.append(f"note count exceeds {max_notes}")
    sustain = evaluate_sustain_profile(candidate.materials)
    if sustain["status"] != "pass":
        issues.extend(str(issue) for issue in sustain["issues"])
    texture = evaluate_piano_texture(candidate)
    if texture["status"] != "pass":
        issues.extend(str(issue) for issue in texture["issues"])
    variations = evaluate_variation_contracts(candidate)
    if variations["status"] != "pass":
        issues.extend(str(issue) for issue in variations["issues"])
    energy = evaluate_energy_order(candidate)
    if energy["status"] != "pass":
        issues.append("declared energy order is not present in the sounding parts")
    structure = energy["structure"]
    if structure.get("status") != "pass":
        issues.extend(str(issue) for issue in structure.get("issues", []))
    try:
        _validate_natural_materials(candidate, candidate.materials)
    except LongFormGenerationError as error:
        issues.append(str(error))
    try:
        reparsed = parse_composition(
            composition_to_source(candidate),
            policy=THREE_MINUTE_POLICY,
            require_parts=True,
            require_section_contract=True,
        )
        if reparsed != candidate:
            issues.append("composition source round-trip changed the candidate")
    except DslError as error:
        issues.append(f"composition source is invalid: {error}")
    unique = list(dict.fromkeys(issues))
    return {
        "status": "pass" if not unique else "fail",
        "issues": unique,
        "duration_ms": duration_ms,
        "note_count": note_count,
        "sustain": sustain,
        "piano_texture": texture,
        "variations": variations,
        "energy_order": energy,
    }


def _write_staged_candidate(run_dir: Path, label: str, composition: Composition) -> dict[str, Path]:
    source_path = run_dir / "staged" / f"{label}.music.py"
    midi_path = run_dir / "staged" / f"{label}.mid"
    atomic_write_bytes(source_path, (composition_to_source(composition) + "\n").encode("utf-8"))
    render_composition(composition, midi_path)
    return {"source": source_path, "midi": midi_path}


def run_full_intensity_iteration(
    *,
    base_source_path: Path,
    run_dir: Path,
    records_path: Path,
    records_manifest_path: Path,
    prompt_path: Path,
    schema_path: Path,
    reference_name: str,
    model: str = MODEL_ID,
    max_notes: int = 950,
) -> dict[str, Any]:
    """中央を先に検査し、合格した場合だけ全曲の低・高候補を生成する。"""

    base_source_path = Path(base_source_path).resolve()
    run_dir = Path(run_dir).resolve()
    records_path = Path(records_path).resolve()
    records_manifest_path = Path(records_manifest_path).resolve()
    prompt_path = Path(prompt_path).resolve()
    schema_path = Path(schema_path).resolve()
    _verify_records_manifest(records_path, records_manifest_path)
    records = load_reference_records(records_path)
    load_prompt(prompt_path)
    try:
        baseline = parse_composition(
            base_source_path.read_text(encoding="utf-8"),
            policy=THREE_MINUTE_POLICY,
            require_parts=True,
            require_section_contract=True,
        )
    except (OSError, DslError) as error:
        raise FullIntensityIterationError(
            f"unable to load full intensity baseline: {error}"
        ) from error

    max_calls = 0
    labels = {-1.0: "intensity-low", 0.0: "intensity-zero", 1.0: "intensity-high"}
    store = RunStore(run_dir, max_calls=max_calls)
    with RunLock(run_dir / ".run.lock"):
        base_snapshot = store.snapshot_file("inputs/base.music.py", base_source_path)
        records_snapshot = store.snapshot_file("inputs/reference-files.jsonl", records_path)
        manifest_snapshot = store.snapshot_file(
            "inputs/reference-manifest.json", records_manifest_path
        )
        prompt_snapshot = store.snapshot_file(
            "inputs/intensity-full-material-batch.md", prompt_path
        )
        schema_snapshot = store.snapshot_file("inputs/response.schema.json", schema_path)
        implementation_snapshot = store.snapshot_file(
            "inputs/full-intensity-iteration.py", Path(__file__)
        )
        hashes = {
            "base_source": sha256_file(base_snapshot),
            "reference_records": sha256_file(records_snapshot),
            "reference_manifest": sha256_file(manifest_snapshot),
            "prompt": sha256_file(prompt_snapshot),
            "schema": sha256_file(schema_snapshot),
            "implementation": sha256_file(implementation_snapshot),
        }
        store.initialize(
            {
                "schema_version": 1,
                "generation_mode": "deterministic_intensity_transform",
                "source_model": model,
                "max_calls": max_calls,
                "step_ids": ["evaluate-center", "evaluate-triplet", "publish-final"],
                "reference": reference_name,
                "input_hashes": hashes,
            }
        )
        targets = {
            value: resolve_intensity_target(
                records,
                reference_name=reference_name,
                value=value,
                max_notes=max_notes,
            )
            for value in (-1.0, 0.0, 1.0)
        }
        if any(target.get("status") != "reachable" for target in targets.values()):
            raise FullIntensityIterationError("full intensity global targets are unreachable")

        material_targets: dict[float, dict[str, dict[str, Any]]] = {
            0.0: build_center_material_targets(baseline, targets[0.0])
        }
        center = generate_deterministic_intensity_candidate(
            baseline,
            material_targets=material_targets[0.0],
        )
        compositions: dict[float, Composition] = {0.0: center}
        staged_paths = {0.0: _write_staged_candidate(run_dir, labels[0.0], center)}
        center_intensity = evaluate_single_full_intensity(center, targets, value=0.0)
        qualities: dict[str, Any] = {
            "0.0": evaluate_full_candidate_quality(baseline, center, max_notes=max_notes)
        }
        center_status = (
            "pass"
            if center_intensity["status"] == "pass" and qualities["0.0"]["status"] == "pass"
            else "fail"
        )
        center_evaluation = {
            "schema_version": 1,
            "status": center_status,
            "reference": reference_name,
            "intensity": center_intensity,
            "target": targets[0.0],
            "material_targets": material_targets[0.0],
            "fixed_quality": qualities["0.0"],
        }
        atomic_write_json(run_dir / "center-evaluation.json", center_evaluation)
        store.record_step(
            "evaluate-center",
            "completed" if center_status == "pass" else "failed",
            hashes,
            {"evaluation_sha256": sha256_file(run_dir / "center-evaluation.json")},
        )
        if center_status != "pass":
            store.record_step(
                "publish-final",
                "skipped",
                hashes,
                {"reason": "center candidate failed; endpoints were not generated"},
            )
            return {
                "status": "fail",
                "stage": "center",
                "run_dir": str(run_dir),
                "reference": reference_name,
                "candidates": {},
                "evaluation_path": str(run_dir / "center-evaluation.json"),
            }

        for value in (-1.0, 1.0):
            material_targets[value] = build_endpoint_material_targets(
                center,
                records,
                reference_name=reference_name,
                value=value,
                max_notes=max_notes,
            )
            candidate = generate_deterministic_intensity_candidate(
                center,
                material_targets=material_targets[value],
            )
            compositions[value] = candidate
            staged_paths[value] = _write_staged_candidate(run_dir, labels[value], candidate)
            qualities[str(value)] = evaluate_full_candidate_quality(
                center, candidate, max_notes=max_notes
            )

        ordered = {value: compositions[value] for value in (-1.0, 0.0, 1.0)}
        intensity = evaluate_full_intensity_triplet(ordered, targets)
        status = (
            "pass"
            if intensity["status"] == "pass"
            and all(report["status"] == "pass" for report in qualities.values())
            else "fail"
        )
        evaluation = {
            "schema_version": 1,
            "status": status,
            "reference": reference_name,
            "intensity": intensity,
            "targets": {str(value): target for value, target in targets.items()},
            "material_targets": {
                str(value): value_targets for value, value_targets in material_targets.items()
            },
            "fixed_quality": qualities,
        }
        atomic_write_json(run_dir / "evaluation.json", evaluation)
        store.record_step(
            "evaluate-triplet",
            "completed" if status == "pass" else "failed",
            hashes,
            {"evaluation_sha256": sha256_file(run_dir / "evaluation.json")},
        )
        published: dict[str, dict[str, str]] = {}
        if status == "pass":
            for value, paths in staged_paths.items():
                label = labels[value]
                source_path = store.promote_file(
                    paths["source"],
                    run_dir / "candidates" / f"{label}.music.py",
                    sha256_file(paths["source"]),
                )
                midi_path = store.promote_file(
                    paths["midi"],
                    run_dir / "candidates" / f"{label}.mid",
                    sha256_file(paths["midi"]),
                )
                published[str(value)] = {
                    "source_path": str(source_path),
                    "midi_path": str(midi_path),
                }
        store.record_step(
            "publish-final",
            "completed" if status == "pass" else "skipped",
            hashes,
            published or {"reason": "automatic full intensity gate failed"},
        )
    return {
        "status": status,
        "stage": "triplet",
        "run_dir": str(run_dir),
        "reference": reference_name,
        "candidates": published,
        "evaluation_path": str(run_dir / "evaluation.json"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="同じ3分構成から、はげしさの低・中・高全曲比較を生成します。"
    )
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-records",
        type=Path,
        default=Path(".appendix/reference-profile-v1/files.jsonl"),
    )
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        default=Path(".appendix/reference-profile-v1/manifest.json"),
    )
    parser.add_argument(
        "--prompt", type=Path, default=Path("prompts/intensity-full-material-batch.md")
    )
    parser.add_argument(
        "--schema", type=Path, default=Path("schemas/codex-composition-response.schema.json")
    )
    parser.add_argument("--reference", default="WayfarersRestStop.mid")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--max-notes", type=int, default=950)
    args = parser.parse_args(argv)
    result = run_full_intensity_iteration(
        base_source_path=args.base_source,
        run_dir=args.run_dir,
        records_path=args.reference_records,
        records_manifest_path=args.reference_manifest,
        prompt_path=args.prompt,
        schema_path=args.schema,
        reference_name=args.reference,
        model=args.model,
        max_notes=args.max_notes,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
