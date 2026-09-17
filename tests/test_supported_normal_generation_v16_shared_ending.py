import json
from dataclasses import replace
from pathlib import Path

import pytest

from llm_musical_composer import (
    supported_normal_generation_v16_shared_ending as v16_module,
)
from llm_musical_composer.pipeline_dsl import parse_piece_plan
from llm_musical_composer.supported_normal_generation_v16_shared_ending import (
    SupportedNormalGenerationV16Error,
    prepare_supported_normal_generation_v16,
    validate_adopted_melody_projections,
)
from llm_musical_composer.whole_score_staged_generation import (
    assemble_whole_score_skeleton,
)
from llm_musical_composer.whole_score_staged_generation_dsl import (
    parse_harmonic_collection,
    parse_melody_collection,
    parse_texture_collection,
)
from llm_musical_composer.whole_score_staged_generation_run import (
    normalize_melody_ending,
)

PROJECT_ROOT = Path(__file__).parents[1]
V15B_RUN = PROJECT_ROOT / (
    ".appendix/supported-normal-generation-v15b-bidirectional-low-spacing-"
    "v6-network-access/runs/whole-score"
)


def test_v16_preflight_reuses_only_the_first_five_materials(tmp_path: Path) -> None:
    artifact_root = tmp_path / "v16"

    result = prepare_supported_normal_generation_v16(PROJECT_ROOT, artifact_root)

    assert result["status"] == "prepared"
    assert result["external_calls_made"] == 0
    assert result["adopted_material_ids"] == [
        "material_1",
        "material_2",
        "material_3",
        "material_4",
        "material_5",
    ]
    assert result["excluded_material_ids"] == ["material_6"]
    assert result["melody_ending_normalization"]["source_attack_units"] == 47
    assert result["melody_ending_normalization"]["required_final_attack_units"] == 51
    prefix = parse_texture_collection(
        (artifact_root / "inputs/prepared-texture-prefix.dsl").read_text(encoding="utf-8")
    )
    assert len(prefix) == 5
    run_dir = artifact_root / "runs/whole-score"
    assert (run_dir / "inputs/prepared-texture-prefix-provenance.json").is_file()
    assert not (run_dir / "outputs/texture-collection-batch-004.dsl").exists()
    state = json.loads((run_dir / "run-state.json").read_text(encoding="utf-8"))
    assert state["calls"]["confirmed_external_call_count"] == 0


def test_v16_preflight_rejects_a_changed_source_batch_hash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original = v16_module.sha256_file

    def tampered(path: Path) -> str:
        if path.name == "texture-collection-batch-002.dsl" and "outputs" in path.parts:
            return "0" * 64
        return original(path)

    monkeypatch.setattr(v16_module, "sha256_file", tampered)

    with pytest.raises(SupportedNormalGenerationV16Error, match="SHA-256"):
        prepare_supported_normal_generation_v16(PROJECT_ROOT, tmp_path / "v16")

    assert not (tmp_path / "v16").exists()


def test_adopted_melody_projection_rejects_a_non_release_change() -> None:
    plan = parse_piece_plan((V15B_RUN / "inputs/piece-plan.dsl").read_text(encoding="utf-8"))
    harmonies = parse_harmonic_collection(
        (V15B_RUN / "inputs/prepared-harmonic-collection.dsl").read_text(encoding="utf-8")
    )
    skeleton = assemble_whole_score_skeleton(
        "v16-test",
        plan,
        tuple((item.length_units, item.draft) for item in harmonies),
    )
    source = parse_melody_collection(
        (V15B_RUN / "inputs/prepared-melody-collection.dsl").read_text(encoding="utf-8")
    )
    normalized, diagnostic = normalize_melody_ending(plan, skeleton, source)
    changed_first = replace(
        normalized[0],
        events=(
            replace(normalized[0].events[0], pitch=normalized[0].events[0].pitch + 1),
            *normalized[0].events[1:],
        ),
    )

    with pytest.raises(SupportedNormalGenerationV16Error, match="adopted melody"):
        validate_adopted_melody_projections(
            source,
            (changed_first, *normalized[1:]),
            diagnostic,
            adopted_material_ids=(
                "material_1",
                "material_2",
                "material_3",
                "material_4",
                "material_5",
            ),
        )
