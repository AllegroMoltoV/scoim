"""V26固定入力で伴奏リズム・内容二段階パイロットを実行する。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llm_musical_composer.harmonic_skeleton import parse_harmonic_skeleton
from llm_musical_composer.pilot_loop import (
    CodexExecRunner,
    isolated_codex_working_directory,
)
from llm_musical_composer.pipeline_dsl import parse_piece_plan
from llm_musical_composer.run_state import (
    RunStore,
    atomic_write_json,
    sha256_file,
    sha256_text,
)
from llm_musical_composer.staged_material_pilot import (
    build_texture_feasibility,
    dump_texture_draft,
)
from llm_musical_composer.texture_rhythm_content_pilot import (
    TextureRhythmDraftV0,
    dump_texture_rhythm_draft,
    execute_texture_rhythm_content_pilot,
)
from llm_musical_composer.whole_score_staged_generation import (
    assemble_whole_score_melodies,
    build_whole_score_context,
)
from llm_musical_composer.whole_score_staged_generation_dsl import (
    parse_melody_collection,
)
from llm_musical_composer.whole_score_staged_generation_run import (
    _validate_partial_texture_batches,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / (
    ".appendix/supported-normal-generation-v26-capacity-normalization-seed007-network-enabled"
)
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / ".appendix/texture-rhythm-content-two-stage-pilot-v1"
MATERIAL_ID = "material_d"
MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "high"
TIMEOUT_SECONDS = 900
PLACEMENT_POLICY = "search-aware-onset-zone-v8"

_SOURCE_RELATIVE_PATHS = {
    "piece_plan": Path("runs/whole-score/inputs/piece-plan.dsl"),
    "harmonic_skeleton": Path("runs/whole-score/outputs/harmonic-skeleton.dsl"),
    "melody_collection": Path("runs/whole-score/outputs/melody-collection.dsl"),
    "texture_budget": Path("runs/whole-score/outputs/texture-budget.json"),
    "register_range": Path("runs/whole-score/outputs/register-range.json"),
    "prompt_target": Path("runs/whole-score/inputs/prompt-target.json"),
    "run_spec": Path("runs/whole-score/run-spec.json"),
}

_IMPLEMENTATION_PATHS = {
    "pilot": Path("src/llm_musical_composer/texture_rhythm_content_pilot.py"),
    "pilot_run": Path("src/llm_musical_composer/texture_rhythm_content_pilot_run.py"),
    "staged_material": Path("src/llm_musical_composer/staged_material_pilot.py"),
    "whole_score_run": Path("src/llm_musical_composer/whole_score_staged_generation_run.py"),
    "placement": Path("src/llm_musical_composer/piano_texture_register_placement.py"),
    "rhythm_prompt": Path("prompts/pipeline-texture-rhythm-pilot.md"),
    "content_prompt": Path("prompts/pipeline-texture-content-pilot.md"),
    "response_schema": Path("schemas/codex-composition-response.schema.json"),
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _render(template: str, context: dict[str, Any]) -> str:
    marker = "{{STAGE_CONTEXT_JSON}}"
    if template.count(marker) != 1:
        raise ValueError("pilot prompt marker is invalid")
    return template.replace(
        marker,
        json.dumps(context, ensure_ascii=False, indent=2, sort_keys=True),
    )


def _fixed_hashes(project_root: Path, source_root: Path) -> dict[str, str]:
    paths = {
        f"source_{name}": source_root / relative
        for name, relative in _SOURCE_RELATIVE_PATHS.items()
    }
    paths.update(
        {
            f"implementation_{name}": project_root / relative
            for name, relative in _IMPLEMENTATION_PATHS.items()
        }
    )
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"pilot fixed input is missing: {missing}")
    return {name: sha256_file(path) for name, path in sorted(paths.items())}


def prepare_texture_rhythm_content_pilot(
    project_root: Path = PROJECT_ROOT,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Any]:
    """外部0回でV26入力、方針、素材、promptを固定する。"""

    project_root = Path(project_root).resolve()
    source_root = Path(source_root).resolve()
    artifact_root = Path(artifact_root).resolve()
    source_paths = {
        name: source_root / relative for name, relative in _SOURCE_RELATIVE_PATHS.items()
    }
    fixed_hashes = _fixed_hashes(project_root, source_root)
    plan = parse_piece_plan(source_paths["piece_plan"].read_text(encoding="utf-8"))
    skeleton = parse_harmonic_skeleton(
        source_paths["harmonic_skeleton"].read_text(encoding="utf-8")
    )
    melody_drafts = parse_melody_collection(
        source_paths["melody_collection"].read_text(encoding="utf-8")
    )
    melodies = assemble_whole_score_melodies("texture-two-stage-v26", plan, skeleton, melody_drafts)
    texture_budget = _read_json(source_paths["texture_budget"])
    register_range = _read_json(source_paths["register_range"])
    prompt_target = _read_json(source_paths["prompt_target"])
    run_spec = _read_json(source_paths["run_spec"])
    if run_spec.get("generation_profile_id") != "stable-staged-v2":
        raise ValueError("V26 generation profile is not stable-staged-v2")
    if run_spec.get("texture_placement_policy") != PLACEMENT_POLICY:
        raise ValueError("V26 texture placement policy is not fixed V8")
    register_enforcement = run_spec.get("register_enforcement")
    if (
        not isinstance(register_enforcement, dict)
        or register_enforcement.get("effective_mode") != "hard"
    ):
        raise ValueError("V26 register enforcement is not hard")

    context = build_whole_score_context(plan)
    if MATERIAL_ID not in context.material_ids:
        raise ValueError("fixed pilot material does not exist")
    if MATERIAL_ID in context.release_material_ids:
        raise ValueError("fixed pilot material must not be a release material")
    harmonic_material = next(item for item in skeleton.materials if item.material_id == MATERIAL_ID)
    melody = next(item for item in melodies.materials if item.material_id == MATERIAL_ID)
    material_budget = next(
        item for item in texture_budget["materials"] if item["material_id"] == MATERIAL_ID
    )
    allowed_pitch_range = (
        int(register_range["minimum_pitch"]),
        int(register_range["maximum_pitch"]),
    )
    feasibility = build_texture_feasibility(
        harmonic_material.harmonies,
        melody.notes,
        allowed_pitch_range=allowed_pitch_range,
    )
    stage_context = {
        "schema_version": 1,
        "experiment_scope": {
            "material_id": MATERIAL_ID,
            "single_material_only": True,
            "does_not_establish_batch_superiority": True,
        },
        "piece": {
            "tonal_center": plan.tonal_center,
            "mode": plan.mode,
            "divisions": skeleton.divisions,
        },
        "material": {
            "material_id": MATERIAL_ID,
            "length_units": harmonic_material.length_units,
            "harmonies": [asdict(item) for item in harmonic_material.harmonies],
            "foreground_voice": melody.foreground_voice,
            "confirmed_melody": [asdict(item) for item in melody.notes],
            "texture_budget": {
                "required_texture_event_count": material_budget["required_texture_event_count"],
                "combined_attack_group_count": material_budget["combined_attack_group_count"],
                "combined_note_event_count": material_budget["combined_note_event_count"],
                "attack_size_counts": material_budget["attack_size_counts"],
                "maximum_group_size": texture_budget["score_spec_target"]["maximum_group_size"],
            },
            "texture_feasibility": feasibility,
        },
        "allowed_pitch_range": {
            "minimum_pitch": allowed_pitch_range[0],
            "maximum_pitch": allowed_pitch_range[1],
        },
        "controls": prompt_target["controls"],
        "semantic_targets": prompt_target["semantic_targets"],
        "placement_policy": PLACEMENT_POLICY,
        "register_enforcement": "hard",
    }
    rhythm_template = (project_root / _IMPLEMENTATION_PATHS["rhythm_prompt"]).read_text(
        encoding="utf-8"
    )
    rhythm_prompt = _render(rhythm_template, stage_context)
    preflight = {
        "schema_version": 1,
        "status": "prepared",
        "source_root": str(source_root),
        "artifact_root": str(artifact_root),
        "material_id": MATERIAL_ID,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "timeout_seconds": TIMEOUT_SECONDS,
        "maximum_external_call_count": 2,
        "placement_policy": PLACEMENT_POLICY,
        "register_enforcement": "hard",
        "fixed_hashes": fixed_hashes,
        "stage_context": stage_context,
        "rhythm_prompt_sha256": sha256_text(rhythm_prompt),
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(artifact_root / "preflight.json", preflight)
    prompt_path = artifact_root / "prompts/texture-rhythm.md"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(rhythm_prompt, encoding="utf-8", newline="\n")
    return preflight


def execute_fixed_texture_rhythm_content_pilot(
    project_root: Path = PROJECT_ROOT,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, object]:
    """prepare済みV26非終止素材を最大2 callで実行する。"""

    project_root = Path(project_root).resolve()
    source_root = Path(source_root).resolve()
    artifact_root = Path(artifact_root).resolve()
    preflight = prepare_texture_rhythm_content_pilot(project_root, source_root, artifact_root)
    source_paths = {
        name: source_root / relative for name, relative in _SOURCE_RELATIVE_PATHS.items()
    }
    plan = parse_piece_plan(source_paths["piece_plan"].read_text(encoding="utf-8"))
    skeleton = parse_harmonic_skeleton(
        source_paths["harmonic_skeleton"].read_text(encoding="utf-8")
    )
    melodies = assemble_whole_score_melodies(
        "texture-two-stage-v26",
        plan,
        skeleton,
        parse_melody_collection(source_paths["melody_collection"].read_text(encoding="utf-8")),
    )
    texture_budget = _read_json(source_paths["texture_budget"])
    register_range = _read_json(source_paths["register_range"])
    harmonic_material = next(item for item in skeleton.materials if item.material_id == MATERIAL_ID)
    melody = next(item for item in melodies.materials if item.material_id == MATERIAL_ID)
    material_budget = next(
        item for item in texture_budget["materials"] if item["material_id"] == MATERIAL_ID
    )
    allowed_pitch_range = (
        int(register_range["minimum_pitch"]),
        int(register_range["maximum_pitch"]),
    )
    feasibility = preflight["stage_context"]["material"]["texture_feasibility"]
    rhythm_validation_arguments = {
        "material_length_units": harmonic_material.length_units,
        "harmonies": harmonic_material.harmonies,
        "melody": melody.notes,
        "budget": material_budget,
        "feasibility": feasibility,
        "maximum_group_size": int(texture_budget["score_spec_target"]["maximum_group_size"]),
    }
    rhythm_prompt = (artifact_root / "prompts/texture-rhythm.md").read_text(encoding="utf-8")
    content_template = (project_root / _IMPLEMENTATION_PATHS["content_prompt"]).read_text(
        encoding="utf-8"
    )
    stage_context = preflight["stage_context"]

    def content_prompt(rhythm: TextureRhythmDraftV0) -> str:
        return _render(
            content_template,
            {
                **stage_context,
                "fixed_rhythm": {
                    "canonical_dsl": dump_texture_rhythm_draft(rhythm),
                    "groups": [asdict(item) for item in rhythm.groups],
                },
            },
        )

    def validate_joined(draft):
        validation = _validate_partial_texture_batches(
            plan,
            skeleton,
            melodies,
            texture_budget,
            {MATERIAL_ID: draft},
            allowed_pitch_range,
            PLACEMENT_POLICY,
        )
        if validation["whole_score_quality"] is not None:
            raise ValueError("single-material pilot unexpectedly ran whole-score quality")
        output = artifact_root / "outputs/texture-draft.dsl"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(dump_texture_draft(draft), encoding="utf-8", newline="\n")
        return validation

    runner = CodexExecRunner(
        run_store=RunStore(artifact_root / "codex", max_calls=2),
        schema_path=project_root / _IMPLEMENTATION_PATHS["response_schema"],
        model=MODEL,
        reasoning_effort=REASONING_EFFORT,
        working_directory=isolated_codex_working_directory(),
        timeout_seconds=TIMEOUT_SECONDS,
    )
    return execute_texture_rhythm_content_pilot(
        artifact_root,
        runner,
        rhythm_prompt=rhythm_prompt,
        content_prompt=content_prompt,
        input_hashes=preflight["fixed_hashes"],
        rhythm_validation_arguments=rhythm_validation_arguments,
        harmonies=harmonic_material.harmonies,
        validate_joined=validate_joined,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    arguments = parser.parse_args()
    if arguments.action == "prepare":
        result = prepare_texture_rhythm_content_pilot(
            arguments.project_root, arguments.source_root, arguments.artifact_root
        )
    else:
        result = execute_fixed_texture_rhythm_content_pilot(
            arguments.project_root, arguments.source_root, arguments.artifact_root
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
