import copy
from pathlib import Path

from test_scoim_script_0_3_compilation import (
    SequencedRunner,
    _approved_flow,
    _structure_response,
)
from test_scoim_script_0_4_model_contracts import _transition_structure_response

from scoim.flow_operations import approve_flow
from scoim.profile_capabilities import solo_piano_3m_v2_capabilities
from scoim.script_0_4_compilation import (
    Script04CompilationRequest,
    compile_script_0_4,
)
from scoim.script_0_4_model_contracts import index_structure_candidate


def _two_scene_flow() -> dict[str, object]:
    draft = _approved_flow()
    draft["status"] = "draft"
    draft["approval"] = None
    draft["scenes"].append(
        {
            "scene_id": "scene-002",
            "name": "回帰",
            "length_class": "medium",
            "heard_as": "主題を変形して戻す。",
            "relation_to_previous": "前の場面を受ける。",
            "transition_to_next": "静かに閉じる。",
        }
    )
    approved = approve_flow(draft)
    assert approved.document is not None
    return approved.document


def _structure_response_0_4() -> dict[str, object]:
    response = copy.deepcopy(_structure_response())
    for scene in response["scenes"]:
        for section in scene["sections"]:
            section["structural_purpose"] = "regular"
    return response


def _two_scene_structure_response() -> dict[str, object]:
    return {
        "materials": [
            {"description": "短い主題。"},
            {"description": "主題を支える低音。"},
        ],
        "scenes": [
            {
                "sections": [
                    {
                        "depth": 0,
                        "role": "statement",
                        "structural_purpose": "regular",
                        "description": "主題を示す。",
                        "relative_length": 1,
                        "placements": [
                            {"material_index": 0, "role": "foreground"},
                            {"material_index": 1, "role": "accompaniment"},
                        ],
                    }
                ]
            },
            {
                "sections": [
                    {
                        "depth": 0,
                        "role": "return",
                        "structural_purpose": "regular",
                        "description": "主題を変形して戻す。",
                        "relative_length": 1,
                        "placements": [
                            {"material_index": 0, "role": "foreground"},
                            {"material_index": 1, "role": "accompaniment"},
                        ],
                    }
                ]
            },
        ],
    }


def _unsupported_relations(indexed: dict[str, object]) -> dict[str, object]:
    placements = indexed["script"]["material_placements"]
    materials = list(indexed["script"]["materials"])
    foreground_ids = [
        placement_id
        for placement_id, placement in placements.items()
        if placement["role"] == "foreground"
    ]
    accompaniment_ids = [
        placement_id
        for placement_id, placement in placements.items()
        if placement["role"] == "accompaniment"
    ]
    return {
        "outcome": "complete",
        "structure_insufficient_reason": None,
        "variation_relations": [
            {
                "source": {"type": "material_placement", "id": foreground_ids[0]},
                "target": {"type": "material_placement", "id": accompaniment_ids[-1]},
                "preserve": ["輪郭"],
                "change": ["役割"],
                "description": "前景を伴奏へ変奏する。",
            },
            {
                "source": {"type": "material", "id": materials[0]},
                "target": {"type": "material", "id": materials[1]},
                "preserve": ["動き"],
                "change": ["音域"],
                "description": "主題を伴奏素材へ変奏する。",
            },
        ],
        "material_placement_transitions": [],
        "performance_directions": [],
    }


def test_script_0_4_compilation_uses_two_operations_and_new_performance_aspects(
    tmp_path: Path,
) -> None:
    relations = {
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
    runner = SequencedRunner([_structure_response_0_4(), relations])

    result = compile_script_0_4(
        Script04CompilationRequest(_approved_flow(), "composition-001"),
        runner,
        tmp_path / "run",
    )

    assert result.compiled is True
    assert result.document is not None
    assert result.document["schema_version"] == "0.4.0"
    assert "performance_direction_comparison_requirements" not in result.document["script"]
    assert len(runner.prompts) == 2


def test_script_0_4_compilation_selects_a_machine_derived_transition_candidate(
    tmp_path: Path,
) -> None:
    relations = {
        "outcome": "complete",
        "structure_insufficient_reason": None,
        "variation_relations": [],
        "material_placement_transitions": [
            {
                "candidate_id": "transition-candidate-001",
                "description": "短く向きを変えて次へ進む。",
            }
        ],
        "performance_directions": [],
    }
    runner = SequencedRunner([_transition_structure_response(), relations])

    result = compile_script_0_4(
        Script04CompilationRequest(_approved_flow(), "composition-transition"),
        runner,
        tmp_path / "run",
    )

    assert result.compiled is True, result.issues
    assert result.document is not None
    transition = next(iter(result.document["script"]["material_placement_transitions"].values()))
    assert transition["transition_material_placement_id"] == "material-placement-002"
    assert len(runner.prompts) == 2


def test_script_0_4_compilation_resumes_after_a_completed_operation(tmp_path: Path) -> None:
    relations = {
        "outcome": "complete",
        "structure_insufficient_reason": None,
        "variation_relations": [],
        "material_placement_transitions": [],
        "performance_directions": [],
    }
    run_dir = tmp_path / "run"
    first_runner = SequencedRunner([_structure_response_0_4()])

    paused = compile_script_0_4(
        Script04CompilationRequest(_approved_flow(), "composition-001"),
        first_runner,
        run_dir,
        max_new_operations=1,
    )

    assert paused.compiled is False
    assert paused.outcome == "paused"
    assert len(first_runner.prompts) == 1

    second_runner = SequencedRunner([relations])
    completed = compile_script_0_4(
        Script04CompilationRequest(_approved_flow(), "composition-001"),
        second_runner,
        run_dir,
    )

    assert completed.compiled is True, completed.issues
    assert len(second_runner.prompts) == 1


def test_script_0_4_repairs_all_downstream_relation_issues_in_one_call(tmp_path: Path) -> None:
    approved_flow = _two_scene_flow()
    structure = _two_scene_structure_response()
    indexed = index_structure_candidate(
        approved_flow,
        "composition-002",
        structure,
        solo_piano_3m_v2_capabilities(),
    ).indexed_structure
    assert indexed is not None
    invalid_relations = _unsupported_relations(indexed)
    repaired_relations = {
        "outcome": "complete",
        "structure_insufficient_reason": None,
        "variation_relations": [],
        "material_placement_transitions": [],
        "performance_directions": [],
    }
    runner = SequencedRunner([structure, invalid_relations, repaired_relations])

    result = compile_script_0_4(
        Script04CompilationRequest(approved_flow, "composition-002"),
        runner,
        tmp_path / "run",
    )

    assert result.compiled is True, result.issues
    assert len(runner.prompts) == 3
    assert "/script/script_element_variation_relations/variation-001" in runner.prompts[2]
    assert "/script/script_element_variation_relations/variation-002" in runner.prompts[2]


def test_script_0_4_stops_after_one_failed_downstream_relation_repair(tmp_path: Path) -> None:
    approved_flow = _two_scene_flow()
    structure = _two_scene_structure_response()
    indexed = index_structure_candidate(
        approved_flow,
        "composition-003",
        structure,
        solo_piano_3m_v2_capabilities(),
    ).indexed_structure
    assert indexed is not None
    invalid_relations = _unsupported_relations(indexed)
    runner = SequencedRunner([structure, invalid_relations, invalid_relations])
    run_dir = tmp_path / "run"

    result = compile_script_0_4(
        Script04CompilationRequest(approved_flow, "composition-003"),
        runner,
        run_dir,
    )

    assert result.compiled is False
    assert result.outcome == "content_invalid"
    assert len(runner.prompts) == 3
    assert not (run_dir / "outputs" / "validated-script.json").exists()
