import json
from pathlib import Path

from test_scoim_script_0_3_compilation import (
    SequencedRunner,
    _approved_flow,
)
from test_scoim_script_0_4_compilation import _structure_response_0_4

from scoim.script_0_4_compilation import (
    Script04CompilationRequest,
    compile_script_0_4,
)
from scoim.v2_composition_bundle import (
    create_v2_composition_bundle,
    verify_v2_composition_bundle,
)


def _relations_response() -> dict[str, object]:
    return {
        "outcome": "complete",
        "structure_insufficient_reason": None,
        "variation_relations": [],
        "material_placement_transitions": [],
        "performance_directions": [
            {
                "target_type": "section",
                "target_id": "section-001",
                "relative_to_type": None,
                "relative_to_id": None,
                "performance_aspects": ["timing"],
                "description": "句の終わりを少し味わう。",
            }
        ],
    }


def test_v2_composition_bundle_packages_a_complete_phase2_run(tmp_path: Path) -> None:
    phase2 = tmp_path / "phase2"
    compiled = compile_script_0_4(
        Script04CompilationRequest(_approved_flow(), "composition-001"),
        SequencedRunner([_structure_response_0_4(), _relations_response()]),
        phase2,
    )
    assert compiled.compiled is True

    destination = tmp_path / "composition"
    result = create_v2_composition_bundle(phase2, destination)

    assert result.created is True, result.issues
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["bundle_type"] == "composition"
    assert manifest["schema_version"] == 3
    assert manifest["target_profile"] == "solo_piano_3m_v2"
    assert manifest["composition_id"] == "composition-001"
    assert (destination / "approved-flow.json").is_file()
    assert (destination / "validated-script.json").is_file()
    assert (destination / "projection-ledger.json").is_file()
    assert (destination / "model-runs" / "phase2" / "run-spec.json").is_file()


def test_v2_composition_bundle_verifies_without_the_original_run(tmp_path: Path) -> None:
    phase2 = tmp_path / "phase2"
    compiled = compile_script_0_4(
        Script04CompilationRequest(_approved_flow(), "composition-001"),
        SequencedRunner([_structure_response_0_4(), _relations_response()]),
        phase2,
    )
    assert compiled.compiled is True
    destination = tmp_path / "composition"
    created = create_v2_composition_bundle(phase2, destination)
    assert created.created is True

    result = verify_v2_composition_bundle(destination)

    assert result.valid is True, result.issues


def test_v2_composition_bundle_rejects_unverifiable_phase2_model_records(
    tmp_path: Path,
) -> None:
    phase2 = tmp_path / "phase2"
    compiled = compile_script_0_4(
        Script04CompilationRequest(_approved_flow(), "composition-001"),
        SequencedRunner([_structure_response_0_4(), _relations_response()]),
        phase2,
    )
    assert compiled.compiled is True
    validation_path = next((phase2 / "attempts").glob("*/attempt-*/validation.json"))
    validation_path.unlink()

    result = create_v2_composition_bundle(phase2, tmp_path / "composition")

    assert result.created is False
    assert result.issues[0].code.value == "lineage_mismatch"
    assert "model operation record" in result.issues[0].message


def test_v2_composition_bundle_rejects_a_modified_file(tmp_path: Path) -> None:
    phase2 = tmp_path / "phase2"
    compiled = compile_script_0_4(
        Script04CompilationRequest(_approved_flow(), "composition-001"),
        SequencedRunner([_structure_response_0_4(), _relations_response()]),
        phase2,
    )
    assert compiled.compiled is True
    destination = tmp_path / "composition"
    created = create_v2_composition_bundle(phase2, destination)
    assert created.created is True
    script_path = destination / "validated-script.json"
    script_path.write_text(script_path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    result = verify_v2_composition_bundle(destination)

    assert result.valid is False
    assert result.issues[0].path == "/bundle"
