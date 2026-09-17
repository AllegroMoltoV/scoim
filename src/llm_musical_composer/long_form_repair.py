"""三分曲の確定計画を保ち、指定した局所素材だけを一度修正する。"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from llm_musical_composer.long_form_generation import (
    Runner,
    _material_source,
    _parse_material_batch,
    _response_source,
    composition_to_source,
)
from llm_musical_composer.long_form_loop import (
    PROJECT_ROOT,
    evaluate_long_form_candidate,
    load_verified_style_target,
)
from llm_musical_composer.material_development import (
    build_material_development_reference_profile_from_directory,
)
from llm_musical_composer.music_dsl import THREE_MINUTE_POLICY, DslError, parse_composition
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
from llm_musical_composer.style_target import EXCLUDED_NAMES


class LongFormRepairError(ValueError):
    """局所修正が対象素材の境界を越えた場合の例外。"""


def _prompt(template: str, replacements: dict[str, str]) -> str:
    result = template
    for name, value in replacements.items():
        result = result.replace("{{" + name + "}}", value)
    if re.search(r"\{\{[A-Za-z0-9_]+\}\}", result):
        raise LongFormRepairError("repair prompt contains unresolved variables")
    return result


def repair_material(
    composition: Any,
    runner: Runner,
    *,
    material_id: str,
    prompt_template: str,
    evaluation: dict[str, Any],
    base_input_hashes: dict[str, str] | None = None,
) -> Any:
    """指定素材だけを応答から差し替え、全曲契約を再検査する。"""
    if material_id not in composition.material_by_id:
        raise LongFormRepairError(f"unknown repair material: {material_id}")
    current = composition.material_by_id[material_id]
    prompt = _prompt(
        prompt_template,
        {
            "material_id": material_id,
            "duration_ms": str(current.duration_ms),
            "current_material": _material_source(current),
            "evaluation": json.dumps(evaluation, ensure_ascii=False, indent=2, sort_keys=True),
        },
    )
    response = runner.run(
        "repair-climax",
        prompt,
        {
            **(base_input_hashes or {}),
            "current_material": sha256_json(_material_source(current)),
            "evaluation": sha256_json(evaluation),
        },
    )
    batch_id, materials = _parse_material_batch(_response_source(response))
    if batch_id != "repair-climax":
        raise LongFormRepairError("repair batch ID changed")
    if len(materials) != 1:
        raise LongFormRepairError("repair must return exactly one material")
    revised = materials[0]
    if revised.material_id != material_id:
        raise LongFormRepairError("repair material ID changed")
    if revised.duration_ms != current.duration_ms:
        raise LongFormRepairError("repair material duration changed")
    result = replace(
        composition,
        materials=tuple(
            revised if material.material_id == material_id else material
            for material in composition.materials
        ),
    )
    try:
        return parse_composition(
            composition_to_source(result),
            policy=THREE_MINUTE_POLICY,
            require_parts=True,
            require_section_contract=True,
        )
    except DslError as error:
        raise LongFormRepairError(f"repaired composition is invalid: {error}") from error


def run_repair(
    *,
    base_run: Path,
    run_dir: Path,
    source_dir: Path,
    target_path: Path,
    manifest_path: Path,
    reference_records_path: Path,
    schema_path: Path,
    material_id: str,
    model: str,
) -> dict[str, Any]:
    """保存済み候補のひとつの素材を一回だけ修正し、再評価する。"""
    base_run = Path(base_run).resolve()
    run_dir = Path(run_dir).resolve()
    verified = load_verified_style_target(target_path, manifest_path, reference_records_path)
    source_path = base_run / "staged/final.music.py"
    verification_path = base_run / "verification.json"
    composition = parse_composition(
        source_path.read_text(encoding="utf-8"),
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )
    original_verification = json.loads(verification_path.read_text(encoding="utf-8"))
    macro = original_verification["macro_structure"]
    repair_evaluation = {
        "issue": macro["issues"],
        "climax": macro["climax"],
        "parts": [
            {"part_id": part["part_id"], "role": part["role"], "features": part["features"]}
            for part in macro["parts"]
        ],
        "required_outcome": (
            "The declared climax must be the strict maximum on at least two of "
            "note_density, velocity_median, and polyphony_mean. Preserve musical development."
        ),
    }
    store = RunStore(run_dir, max_calls=1)
    with RunLock(run_dir / ".run.lock"):
        source_snapshot = store.snapshot_file("inputs/base.music.py", source_path)
        verification_snapshot = store.snapshot_file(
            "inputs/base-verification.json", verification_path
        )
        target_snapshot = store.snapshot_file("inputs/style-target.json", target_path)
        manifest_snapshot = store.snapshot_file("inputs/style-target-manifest.json", manifest_path)
        records_snapshot = store.snapshot_file(
            "inputs/reference-files.jsonl", reference_records_path
        )
        prompt_snapshot = store.snapshot_file(
            "inputs/long-form-repair.md", PROJECT_ROOT / "prompts/long-form-repair.md"
        )
        schema_snapshot = store.snapshot_file("inputs/response.schema.json", schema_path)
        hashes = {
            "base_source": sha256_file(source_snapshot),
            "base_verification": sha256_file(verification_snapshot),
            "style_target": sha256_file(target_snapshot),
            "style_manifest": sha256_file(manifest_snapshot),
            "reference_records": sha256_file(records_snapshot),
            "prompt": sha256_file(prompt_snapshot),
            "schema": sha256_file(schema_snapshot),
        }
        store.initialize(
            {
                "schema_version": 1,
                "requested_model": model,
                "max_calls": 1,
                "step_ids": ["repair-climax", "evaluate-candidate", "publish-final"],
                "material_id": material_id,
                "input_hashes": hashes,
            }
        )
        runner = CodexExecRunner(
            run_store=store,
            schema_path=schema_snapshot,
            model=model,
            working_directory=isolated_codex_working_directory(),
        )
        repaired = repair_material(
            composition,
            runner,
            material_id=material_id,
            prompt_template=prompt_snapshot.read_text(encoding="utf-8"),
            evaluation=repair_evaluation,
            base_input_hashes=hashes,
        )
        staged = run_dir / "staged"
        source_bytes = (composition_to_source(repaired) + "\n").encode("utf-8")
        atomic_write_bytes(staged / "final.music.py", source_bytes)
        render_composition(repaired, staged / "final.mid")
        profile = build_material_development_reference_profile_from_directory(
            source_dir, excluded_names=EXCLUDED_NAMES
        )
        verification = evaluate_long_form_candidate(
            repaired,
            staged / "final.mid",
            style_target=verified["target"],
            material_development_profile=profile,
        )
        atomic_write_json(run_dir / "verification.json", verification)
        outputs: dict[str, str | None] = {"final_source": None, "final_midi": None}
        store.record_step(
            "evaluate-candidate",
            "completed",
            hashes,
            {"verification_sha256": sha256_file(run_dir / "verification.json")},
        )
        if verification["status"] == "pass":
            final_source = store.promote_file(
                staged / "final.music.py",
                run_dir / "final.music.py",
                sha256_file(staged / "final.music.py"),
            )
            final_midi = store.promote_file(
                staged / "final.mid", run_dir / "final.mid", sha256_file(staged / "final.mid")
            )
            outputs = {"final_source": str(final_source), "final_midi": str(final_midi)}
            store.record_step("publish-final", "completed", hashes, outputs)
        else:
            store.record_step(
                "publish-final",
                "skipped",
                hashes,
                {"reason": "required deterministic gate failed"},
            )
    return {
        "status": verification["status"],
        "run_dir": str(run_dir),
        **outputs,
        "verification": verification,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="三分曲の指定素材だけを一度修正します。")
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--material-id", required=True)
    parser.add_argument("--source-dir", type=Path, default=Path(".appendix/source-smf"))
    parser.add_argument(
        "--style-target", type=Path, default=Path(".appendix/style-target/style-target.json")
    )
    parser.add_argument(
        "--style-manifest", type=Path, default=Path(".appendix/style-target/manifest.json")
    )
    parser.add_argument(
        "--reference-records",
        type=Path,
        default=Path(".appendix/reference-structure-analysis/files.jsonl"),
    )
    parser.add_argument(
        "--schema", type=Path, default=Path("schemas/codex-composition-response.schema.json")
    )
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args(argv)
    result = run_repair(
        base_run=args.base_run,
        run_dir=args.run_dir,
        source_dir=args.source_dir,
        target_path=args.style_target,
        manifest_path=args.style_manifest,
        reference_records_path=args.reference_records,
        schema_path=args.schema,
        material_id=args.material_id,
        model=args.model,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
