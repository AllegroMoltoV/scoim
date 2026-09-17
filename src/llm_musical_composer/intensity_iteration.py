"""同じ局所素材から`はげしさ`の低・高候補を範囲限定して生成する。"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from llm_musical_composer.composition_ir import Composition, Material
from llm_musical_composer.intensity_control import (
    evaluate_material_intensity_triplet,
    extract_material_intensity,
    load_reference_records,
    resolve_material_intensity_target,
)
from llm_musical_composer.long_form_generation import (
    Runner,
    _material_source,
    _parse_material_batch,
    _response_source,
    composition_to_source,
)
from llm_musical_composer.music_dsl import THREE_MINUTE_POLICY, DslError, parse_composition
from llm_musical_composer.piano_texture import (
    evaluate_piano_texture,
    evaluate_variation_contracts,
    voice_texture_features,
)
from llm_musical_composer.pilot_loop import (
    MODEL_ID,
    CodexExecRunner,
    isolated_codex_working_directory,
)
from llm_musical_composer.run_state import (
    RunLock,
    RunStore,
    atomic_write_bytes,
    atomic_write_json,
    sha256_file,
    sha256_json,
)
from llm_musical_composer.smf_render import render_composition
from llm_musical_composer.sustain_profile import evaluate_sustain_profile


class IntensityIterationError(ValueError):
    """局所強度候補が入力、範囲、または固定品質契約に違反した場合の例外。"""


@dataclass(frozen=True)
class LocalIntensityTriplet:
    """同じ基準素材から作った低・中・高候補と機械評価。"""

    compositions: dict[float, Composition]
    targets: dict[float, dict[str, Any]]
    evaluation: dict[str, Any]
    repairs: dict[str, Any] = field(default_factory=dict)


def _render_prompt(template: str, replacements: dict[str, str]) -> str:
    result = template
    for name, value in replacements.items():
        result = result.replace("{{" + name + "}}", value)
    if re.search(r"\{\{[A-Za-z0-9_]+\}\}", result):
        raise IntensityIterationError("intensity prompt contains unresolved variables")
    return result


def _batch_id(value: float) -> str:
    if value == -1.0:
        return "intensity-low"
    if value == 1.0:
        return "intensity-high"
    raise IntensityIterationError("local intensity generation accepts only -1 or 1")


def evaluate_upper_delay_artifact(
    notes: tuple[Any, ...], *, tolerance_ms: int = 50
) -> dict[str, Any]:
    """下声の直後へ上声を一律に微小遅延する既知の不自然さを検出する。"""

    upper = sorted({note.at_ms for note in notes if note.voice == "upper"})
    lower = sorted({note.at_ms for note in notes if note.voice == "lower"})
    delayed = [
        onset
        for onset in upper
        if any(1 <= onset - lower_onset <= tolerance_ms for lower_onset in lower)
    ]
    delayed_fraction = len(delayed) / len(upper) if upper else 0.0
    systematic = len(delayed) >= 3 and delayed_fraction >= 0.75
    return {
        "status": "fail" if systematic else "pass",
        "issue": "upper voice is systematically micro-delayed after lower voice"
        if systematic
        else None,
        "upper_onset_count": len(upper),
        "delayed_upper_onset_count": len(delayed),
        "delayed_upper_fraction": round(delayed_fraction, 8),
        "tolerance_ms": tolerance_ms,
    }


def _tonal_context(composition: Composition, material: Material) -> dict[str, Any]:
    pitch_classes = sorted({note.pitch % 12 for note in material.notes})
    pitches = [note.pitch for note in material.notes]
    return {
        "tonal_center": composition.tonal_center,
        "mode": composition.mode,
        "allowed_pitch_classes": pitch_classes,
        "minimum_pitch": min(pitches),
        "maximum_pitch": max(pitches),
        "rule": "reuse the existing harmonic vocabulary; do not transpose or add pitch classes",
    }


def _validate_revision(current: Material, revised: Material) -> None:
    if revised.material_id != current.material_id:
        raise IntensityIterationError("intensity material ID changed")
    if revised.duration_ms != current.duration_ms:
        raise IntensityIterationError("intensity material duration changed")
    if revised.derived_from != current.derived_from:
        raise IntensityIterationError("intensity material derivation changed")
    if revised.pedals != current.pedals:
        raise IntensityIterationError("intensity material pedals changed")
    current_pitch_classes = {note.pitch % 12 for note in current.notes}
    revised_pitch_classes = {note.pitch % 12 for note in revised.notes}
    if not revised_pitch_classes or not revised_pitch_classes <= current_pitch_classes:
        raise IntensityIterationError("intensity material added an unsupported pitch-class")
    current_range = (
        min(note.pitch for note in current.notes),
        max(note.pitch for note in current.notes),
    )
    if any(not current_range[0] <= note.pitch <= current_range[1] for note in revised.notes):
        raise IntensityIterationError("intensity material changed the allowed pitch range")
    texture = voice_texture_features(revised.notes)
    if texture["status"] != "pass":
        raise IntensityIterationError(
            f"intensity material broke voice texture: {texture['issues'][0]}"
        )
    delay = evaluate_upper_delay_artifact(revised.notes)
    if delay["status"] != "pass":
        raise IntensityIterationError(str(delay["issue"]))


def repair_material_velocity(
    material: Material, target: dict[str, Any]
) -> tuple[Material, dict[str, Any]]:
    """全音符の強弱順位と幅を保つ一定オフセットだけでvelocityを校正する。"""

    try:
        target_level = float(target["targets"]["velocity_level"])
        accepted = target["acceptance"]["velocity_level"]
        minimum = float(accepted["minimum"])
        maximum = float(accepted["maximum"])
    except (KeyError, TypeError, ValueError) as error:
        raise IntensityIterationError("velocity repair target is invalid") from error
    if target.get("status") != "reachable" or not all(
        math.isfinite(value) for value in (target_level, minimum, maximum)
    ):
        raise IntensityIterationError("velocity repair target is invalid")
    if not 0 <= minimum <= target_level <= maximum <= 1:
        raise IntensityIterationError("velocity repair target range is invalid")
    if not material.notes:
        raise IntensityIterationError("velocity repair requires notes")

    before = extract_material_intensity(material)["observables"]["velocity_level"]
    candidates: list[tuple[tuple[float, int, int], Material, float, int]] = []
    minimum_offset = 1 - min(note.velocity for note in material.notes)
    maximum_offset = 127 - max(note.velocity for note in material.notes)
    for offset in range(minimum_offset, maximum_offset + 1):
        revised = replace(
            material,
            notes=tuple(replace(note, velocity=note.velocity + offset) for note in material.notes),
        )
        level = extract_material_intensity(revised)["observables"]["velocity_level"]
        if minimum <= level <= maximum:
            candidates.append(
                ((abs(offset), abs(level - target_level), offset), revised, level, offset)
            )
    if not candidates:
        raise IntensityIterationError("no constant velocity offset reaches the target range")
    _, revised, after, offset = min(candidates, key=lambda item: item[0])
    return revised, {
        "status": "repaired",
        "method": "constant_integer_velocity_offset_without_clipping",
        "offset": offset,
        "before": before,
        "after": after,
        "target": target_level,
        "acceptance": {"minimum": minimum, "maximum": maximum},
    }


def apply_velocity_only_repair(
    compositions: dict[float, Composition],
    targets: dict[float, dict[str, Any]],
    evaluation: dict[str, Any],
    *,
    material_id: str,
) -> LocalIntensityTriplet:
    """velocityだけが外れた端点へ一定オフセット修復を一度適用する。"""

    if evaluation.get("status") == "pass":
        return LocalIntensityTriplet(compositions, targets, evaluation)
    if evaluation.get("issues") != ["velocity_level"]:
        return LocalIntensityTriplet(compositions, targets, evaluation)
    updated = dict(compositions)
    repairs: dict[str, Any] = {}
    for value in (-1.0, 1.0):
        observed = evaluation["candidates"][str(value)]["observables"]["velocity_level"]
        accepted = targets[value]["acceptance"]["velocity_level"]
        if accepted["minimum"] <= observed <= accepted["maximum"]:
            continue
        composition = updated[value]
        material = composition.material_by_id[material_id]
        revised, report = repair_material_velocity(material, targets[value])
        updated[value] = replace(
            composition,
            materials=tuple(
                revised if item.material_id == material_id else item
                for item in composition.materials
            ),
        )
        repairs[str(value)] = report
    repaired_evaluation = evaluate_material_intensity_triplet(
        {value: candidate.material_by_id[material_id] for value, candidate in updated.items()},
        targets,
    )
    return LocalIntensityTriplet(updated, targets, repaired_evaluation, repairs)


def generate_intensity_material(
    composition: Composition,
    runner: Runner,
    *,
    material_id: str,
    control_value: float,
    target: dict[str, Any],
    prompt_template: str,
    base_input_hashes: dict[str, str] | None = None,
) -> Composition:
    """指定素材の低または高候補だけをCodex応答から受理する。"""

    if (
        isinstance(control_value, bool)
        or not isinstance(control_value, (int, float))
        or not math.isfinite(float(control_value))
    ):
        raise IntensityIterationError("control value must be finite")
    value = float(control_value)
    batch_id = _batch_id(value)
    if target.get("status") != "reachable" or target.get("value") != value:
        raise IntensityIterationError("intensity target must be reachable and match the value")
    current = composition.material_by_id.get(material_id)
    if current is None:
        raise IntensityIterationError(f"unknown intensity material: {material_id}")
    tonal_context = _tonal_context(composition, current)
    prompt = _render_prompt(
        prompt_template,
        {
            "batch_id": batch_id,
            "material_id": material_id,
            "control_value": str(value),
            "current_material": _material_source(current),
            "target": json.dumps(target, ensure_ascii=False, indent=2, sort_keys=True),
            "tonal_context": json.dumps(
                tonal_context, ensure_ascii=False, indent=2, sort_keys=True
            ),
        },
    )
    response = runner.run(
        batch_id,
        prompt,
        {
            **(base_input_hashes or {}),
            "current_material": sha256_json(_material_source(current)),
            "intensity_target": sha256_json(target),
            "tonal_context": sha256_json(tonal_context),
        },
    )
    returned_batch_id, materials = _parse_material_batch(_response_source(response))
    if returned_batch_id != batch_id:
        raise IntensityIterationError("intensity batch ID changed")
    if len(materials) != 1:
        raise IntensityIterationError("intensity batch must return exactly one material")
    revised = materials[0]
    _validate_revision(current, revised)
    return replace(
        composition,
        materials=tuple(
            revised if material.material_id == material_id else material
            for material in composition.materials
        ),
    )


def select_intensity_test_material(composition: Composition) -> str:
    """派生関係と特殊役割を避け、最初の独立した一回使用素材を選ぶ。"""

    parent_ids = {
        material.derived_from
        for material in composition.materials
        if material.derived_from is not None
    }
    disallowed_roles = {"transition", "climax", "release"}
    for material in composition.materials:
        uses = [use for use in composition.form if use.material_id == material.material_id]
        roles = {use.role for use in uses}
        if (
            material.derived_from is None
            and material.material_id not in parent_ids
            and len(uses) == 1
            and roles
            and None not in roles
            and not roles & disallowed_roles
            and {note.voice for note in material.notes} >= {"upper", "lower"}
        ):
            return material.material_id
    raise IntensityIterationError("no independent local intensity test material is available")


def build_local_intensity_triplet(
    composition: Composition,
    runner: Runner,
    records: list[dict[str, Any]],
    *,
    reference_name: str,
    material_id: str,
    prompt_template: str,
    base_input_hashes: dict[str, str] | None = None,
    max_notes: int = 950,
) -> LocalIntensityTriplet:
    """基準素材を再利用し、低・高だけを独立呼び出しで生成する。"""

    if material_id not in composition.material_by_id:
        raise IntensityIterationError(f"unknown intensity material: {material_id}")
    material = composition.material_by_id[material_id]
    targets = {
        value: resolve_material_intensity_target(
            material,
            records,
            reference_name=reference_name,
            value=value,
            max_notes=max_notes,
        )
        for value in (-1.0, 0.0, 1.0)
    }
    unreachable = [value for value, target in targets.items() if target["status"] != "reachable"]
    if unreachable:
        raise IntensityIterationError(
            "local intensity targets are unreachable: "
            + ", ".join(str(value) for value in unreachable)
        )
    low = generate_intensity_material(
        composition,
        runner,
        material_id=material_id,
        control_value=-1.0,
        target=targets[-1.0],
        prompt_template=prompt_template,
        base_input_hashes=base_input_hashes,
    )
    high = generate_intensity_material(
        composition,
        runner,
        material_id=material_id,
        control_value=1.0,
        target=targets[1.0],
        prompt_template=prompt_template,
        base_input_hashes=base_input_hashes,
    )
    compositions = {-1.0: low, 0.0: composition, 1.0: high}
    evaluation = evaluate_material_intensity_triplet(
        {value: candidate.material_by_id[material_id] for value, candidate in compositions.items()},
        targets,
    )
    return apply_velocity_only_repair(compositions, targets, evaluation, material_id=material_id)


def load_prompt(path: Path) -> str:
    """局所強度プロンプトを読み、空ファイルを拒否する。"""

    try:
        prompt = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise IntensityIterationError(f"unable to read intensity prompt: {error}") from error
    if not prompt.strip():
        raise IntensityIterationError("intensity prompt is empty")
    return prompt


def _verify_records_manifest(records_path: Path, manifest_path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntensityIterationError(f"unable to read reference manifest: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("status") != "pass":
        raise IntensityIterationError("reference manifest has not passed")
    outputs = manifest.get("outputs")
    expected = outputs.get("files.jsonl") if isinstance(outputs, dict) else None
    if not isinstance(expected, str) or sha256_file(records_path) != expected:
        raise IntensityIterationError("reference manifest does not match files.jsonl")
    return manifest


def _evaluate_full_quality(
    baseline: Composition, candidate: Composition, *, material_id: str, max_notes: int
) -> dict[str, Any]:
    issues: list[str] = []
    if candidate.title != baseline.title:
        issues.append("title changed")
    if candidate.form != baseline.form or candidate.parts != baseline.parts:
        issues.append("structure changed")
    if candidate.phrases != baseline.phrases:
        issues.append("phrase plan changed")
    if (
        candidate.tonal_center != baseline.tonal_center
        or candidate.mode != baseline.mode
        or candidate.ending != baseline.ending
    ):
        issues.append("tonal context or ending changed")
    for current in baseline.materials:
        if current.material_id == material_id:
            continue
        if candidate.material_by_id.get(current.material_id) != current:
            issues.append(f"material {current.material_id} changed outside scope")
    if candidate.duration_ms != baseline.duration_ms:
        issues.append("duration changed")
    if candidate.note_count > max_notes:
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
    return {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "duration_ms": candidate.duration_ms,
        "note_count": candidate.note_count,
        "sustain": sustain,
        "piano_texture": texture,
        "variations": variations,
    }


def run_local_intensity_iteration(
    *,
    base_source_path: Path,
    run_dir: Path,
    records_path: Path,
    records_manifest_path: Path,
    prompt_path: Path,
    schema_path: Path,
    reference_name: str,
    model: str = MODEL_ID,
    material_id: str | None = None,
    max_notes: int = 950,
) -> dict[str, Any]:
    """局所素材の低・中・高を作り、自動検査通過時だけ比較候補を公開する。"""

    base_source_path = Path(base_source_path).resolve()
    run_dir = Path(run_dir).resolve()
    records_path = Path(records_path).resolve()
    records_manifest_path = Path(records_manifest_path).resolve()
    prompt_path = Path(prompt_path).resolve()
    schema_path = Path(schema_path).resolve()
    _verify_records_manifest(records_path, records_manifest_path)
    records = load_reference_records(records_path)
    try:
        composition = parse_composition(
            base_source_path.read_text(encoding="utf-8"),
            policy=THREE_MINUTE_POLICY,
            require_parts=True,
            require_section_contract=True,
        )
    except (OSError, DslError) as error:
        raise IntensityIterationError(f"unable to load baseline composition: {error}") from error
    selected_material_id = material_id or select_intensity_test_material(composition)
    if selected_material_id not in composition.material_by_id:
        raise IntensityIterationError(f"unknown intensity material: {selected_material_id}")
    prompt_template = load_prompt(prompt_path)

    store = RunStore(run_dir, max_calls=2)
    with RunLock(run_dir / ".run.lock"):
        base_snapshot = store.snapshot_file("inputs/base.music.py", base_source_path)
        records_snapshot = store.snapshot_file("inputs/reference-files.jsonl", records_path)
        manifest_snapshot = store.snapshot_file(
            "inputs/reference-manifest.json", records_manifest_path
        )
        prompt_snapshot = store.snapshot_file("inputs/intensity-local-material.md", prompt_path)
        schema_snapshot = store.snapshot_file("inputs/response.schema.json", schema_path)
        hashes = {
            "base_source": sha256_file(base_snapshot),
            "reference_records": sha256_file(records_snapshot),
            "reference_manifest": sha256_file(manifest_snapshot),
            "prompt": sha256_file(prompt_snapshot),
            "schema": sha256_file(schema_snapshot),
        }
        store.initialize(
            {
                "schema_version": 1,
                "requested_model": model,
                "max_calls": 2,
                "step_ids": [
                    "intensity-low",
                    "intensity-high",
                    "evaluate-triplet",
                    "publish-final",
                ],
                "reference": reference_name,
                "material_id": selected_material_id,
                "input_hashes": hashes,
            }
        )
        runner = CodexExecRunner(
            run_store=store,
            schema_path=schema_snapshot,
            model=model,
            working_directory=isolated_codex_working_directory(),
        )
        triplet = build_local_intensity_triplet(
            composition,
            runner,
            records,
            reference_name=reference_name,
            material_id=selected_material_id,
            prompt_template=prompt_template,
            base_input_hashes=hashes,
            max_notes=max_notes,
        )
        labels = {-1.0: "intensity-low", 0.0: "intensity-zero", 1.0: "intensity-high"}
        staged_paths: dict[float, dict[str, Path]] = {}
        qualities: dict[str, Any] = {}
        for value, candidate in triplet.compositions.items():
            label = labels[value]
            source_path = run_dir / "staged" / f"{label}.music.py"
            midi_path = run_dir / "staged" / f"{label}.mid"
            atomic_write_bytes(source_path, (composition_to_source(candidate) + "\n").encode())
            render_composition(candidate, midi_path)
            staged_paths[value] = {"source": source_path, "midi": midi_path}
            qualities[str(value)] = _evaluate_full_quality(
                composition,
                candidate,
                material_id=selected_material_id,
                max_notes=max_notes,
            )
        status = (
            "pass"
            if triplet.evaluation["status"] == "pass"
            and all(report["status"] == "pass" for report in qualities.values())
            else "fail"
        )
        evaluation = {
            "schema_version": 1,
            "status": status,
            "reference": reference_name,
            "material_id": selected_material_id,
            "intensity": triplet.evaluation,
            "repairs": triplet.repairs,
            "targets": {str(value): target for value, target in triplet.targets.items()},
            "fixed_quality": qualities,
        }
        atomic_write_json(run_dir / "evaluation.json", evaluation)
        store.record_step(
            "evaluate-triplet",
            "completed",
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
            store.record_step("publish-final", "completed", hashes, published)
        else:
            store.record_step(
                "publish-final",
                "skipped",
                hashes,
                {"reason": "automatic intensity or fixed-quality gate failed"},
            )
    return {
        "status": status,
        "run_dir": str(run_dir),
        "reference": reference_name,
        "material_id": selected_material_id,
        "candidates": published,
        "evaluation_path": str(run_dir / "evaluation.json"),
    }


def repair_saved_local_intensity_iteration(
    *, failed_run: Path, run_dir: Path, max_notes: int = 950
) -> dict[str, Any]:
    """保存済みのvelocity単独失敗を外部呼び出しなしで限定修復する。"""

    failed_run = Path(failed_run).resolve()
    run_dir = Path(run_dir).resolve()
    evaluation_path = failed_run / "evaluation.json"
    try:
        failed_evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntensityIterationError(f"unable to read failed intensity run: {error}") from error
    if (
        not isinstance(failed_evaluation, dict)
        or failed_evaluation.get("status") != "fail"
        or failed_evaluation.get("intensity", {}).get("issues") != ["velocity_level"]
    ):
        raise IntensityIterationError("saved run is not a velocity-only failure")
    material_id = failed_evaluation.get("material_id")
    reference = failed_evaluation.get("reference")
    if not isinstance(material_id, str) or not isinstance(reference, str):
        raise IntensityIterationError("saved run identity is invalid")
    labels = {-1.0: "intensity-low", 0.0: "intensity-zero", 1.0: "intensity-high"}
    compositions: dict[float, Composition] = {}
    source_paths: dict[float, Path] = {}
    try:
        for value, label in labels.items():
            source_path = failed_run / "staged" / f"{label}.music.py"
            source_paths[value] = source_path
            compositions[value] = parse_composition(
                source_path.read_text(encoding="utf-8"),
                policy=THREE_MINUTE_POLICY,
                require_parts=True,
                require_section_contract=True,
            )
        targets = {value: failed_evaluation["targets"][str(value)] for value in labels}
    except (OSError, DslError, KeyError, TypeError) as error:
        raise IntensityIterationError(f"saved intensity artifacts are invalid: {error}") from error
    repaired = apply_velocity_only_repair(
        compositions,
        targets,
        failed_evaluation["intensity"],
        material_id=material_id,
    )
    qualities = {
        str(value): _evaluate_full_quality(
            compositions[0.0],
            candidate,
            material_id=material_id,
            max_notes=max_notes,
        )
        for value, candidate in repaired.compositions.items()
    }
    status = (
        "pass"
        if repaired.evaluation["status"] == "pass"
        and repaired.repairs
        and all(report["status"] == "pass" for report in qualities.values())
        else "fail"
    )
    result_evaluation = {
        "schema_version": 1,
        "status": status,
        "source_run": str(failed_run),
        "reference": reference,
        "material_id": material_id,
        "intensity": repaired.evaluation,
        "targets": {str(value): target for value, target in repaired.targets.items()},
        "repairs": repaired.repairs,
        "fixed_quality": qualities,
    }

    store = RunStore(run_dir, max_calls=0)
    with RunLock(run_dir / ".run.lock"):
        failed_evaluation_snapshot = store.snapshot_file(
            "inputs/failed-evaluation.json", evaluation_path
        )
        source_snapshots = {
            value: store.snapshot_file(f"inputs/{labels[value]}.music.py", source_path)
            for value, source_path in source_paths.items()
        }
        implementation_snapshot = store.snapshot_file(
            "inputs/intensity-iteration.py", Path(__file__)
        )
        hashes = {
            "failed_evaluation": sha256_file(failed_evaluation_snapshot),
            "implementation": sha256_file(implementation_snapshot),
            **{
                f"source_{labels[value]}": sha256_file(snapshot)
                for value, snapshot in source_snapshots.items()
            },
        }
        store.initialize(
            {
                "schema_version": 1,
                "max_calls": 0,
                "step_ids": ["repair-velocity", "evaluate-triplet", "publish-final"],
                "source_run": str(failed_run),
                "material_id": material_id,
                "input_hashes": hashes,
            }
        )
        atomic_write_json(run_dir / "evaluation.json", result_evaluation)
        published: dict[str, dict[str, str]] = {}
        for value, candidate in repaired.compositions.items():
            label = labels[value]
            staged_source = run_dir / "staged" / f"{label}.music.py"
            staged_midi = run_dir / "staged" / f"{label}.mid"
            atomic_write_bytes(
                staged_source, (composition_to_source(candidate) + "\n").encode("utf-8")
            )
            render_composition(candidate, staged_midi)
            if status == "pass":
                source_path = store.promote_file(
                    staged_source,
                    run_dir / "candidates" / f"{label}.music.py",
                    sha256_file(staged_source),
                )
                midi_path = store.promote_file(
                    staged_midi,
                    run_dir / "candidates" / f"{label}.mid",
                    sha256_file(staged_midi),
                )
                published[str(value)] = {
                    "source_path": str(source_path),
                    "midi_path": str(midi_path),
                }
        store.record_step(
            "repair-velocity",
            "completed" if repaired.repairs else "skipped",
            hashes,
            repaired.repairs or {"reason": "no repair was applied"},
        )
        store.record_step(
            "evaluate-triplet",
            "completed",
            hashes,
            {"evaluation_sha256": sha256_file(run_dir / "evaluation.json")},
        )
        store.record_step(
            "publish-final",
            "completed" if status == "pass" else "skipped",
            hashes,
            published or {"reason": "automatic gate failed"},
        )
    return {
        "status": status,
        "run_dir": str(run_dir),
        "reference": reference,
        "material_id": material_id,
        "candidates": published,
        "evaluation_path": str(run_dir / "evaluation.json"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="同じ局所素材から、はげしさの低・中・高比較を生成します。"
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
    parser.add_argument("--prompt", type=Path, default=Path("prompts/intensity-local-material.md"))
    parser.add_argument(
        "--schema", type=Path, default=Path("schemas/codex-composition-response.schema.json")
    )
    parser.add_argument("--reference", default="WayfarersRestStop.mid")
    parser.add_argument("--material-id")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--max-notes", type=int, default=950)
    args = parser.parse_args(argv)
    result = run_local_intensity_iteration(
        base_source_path=args.base_source,
        run_dir=args.run_dir,
        records_path=args.reference_records,
        records_manifest_path=args.reference_manifest,
        prompt_path=args.prompt,
        schema_path=args.schema,
        reference_name=args.reference,
        model=args.model,
        material_id=args.material_id,
        max_notes=args.max_notes,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


def repair_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="保存済み局所比較のvelocity単独失敗を決定的に修復します。"
    )
    parser.add_argument("--failed-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-notes", type=int, default=950)
    args = parser.parse_args(argv)
    result = repair_saved_local_intensity_iteration(
        failed_run=args.failed_run,
        run_dir=args.run_dir,
        max_notes=args.max_notes,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
