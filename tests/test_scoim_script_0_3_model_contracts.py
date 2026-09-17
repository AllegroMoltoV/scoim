import json

from scoim.flow_operations import approve_flow
from scoim.profile_capabilities import solo_piano_3m_v2_capabilities
from scoim.script_0_3_model_contracts import (
    build_script_document,
    check_structure_candidate,
    index_structure_candidate,
    relations_prompt,
    relations_response_schema,
    structure_prompt,
    structure_response_schema,
)
from scoim.script_0_3_validation import check_script_0_3_document


def _approved_flow() -> dict[str, object]:
    result = approve_flow(
        {
            "document_type": "flow",
            "schema_version": "0.1.0",
            "document_id": "flow-001",
            "revision": 1,
            "status": "draft",
            "approval": None,
            "title": "小さな曲",
            "overall_flow": "主題を示して閉じる。",
            "instrumentation": "solo_piano",
            "target_duration": 180,
            "scenes": [
                {
                    "scene_id": "scene-001",
                    "name": "主題",
                    "length_class": "medium",
                    "heard_as": "穏やかな主題。",
                    "relation_to_previous": "曲の始まり。",
                    "transition_to_next": "静かに閉じる。",
                }
            ],
        }
    )
    assert result.document is not None
    return result.document


def test_structure_schema_uses_profile_placement_roles_but_not_section_role_vocabulary() -> None:
    capabilities = solo_piano_3m_v2_capabilities()

    schema = structure_response_schema(capabilities)
    section = schema["properties"]["scenes"]["items"]["properties"]["sections"]["items"]
    placement = section["properties"]["placements"]["items"]

    assert placement["properties"]["role"]["enum"] == sorted(capabilities.material_placement_roles)
    assert section["properties"]["role"] == {
        "type": "string",
        "pattern": "^[a-z][a-z0-9_-]*$",
    }
    assert "structural_purpose" not in section["required"]
    assert "structural_purpose" not in section["properties"]


def test_structure_prompt_receives_the_same_profile_capabilities() -> None:
    capabilities = solo_piano_3m_v2_capabilities()

    prompt = structure_prompt(_approved_flow(), capabilities)

    assert '"material_placement_roles": ["accompaniment", "foreground"]' in prompt
    assert '"minimum_material_placements_per_leaf": 1' in prompt
    assert "最終IDを返さない" in prompt


def test_structure_candidate_rejects_model_generated_final_ids() -> None:
    response = {
        "materials": [{"description": "短い主題", "id": "material-made-by-model"}],
        "scenes": [
            {
                "sections": [
                    {
                        "depth": 0,
                        "role": "statement",
                        "description": "主題を示す",
                        "relative_length": 1,
                        "placements": [{"material_index": 0, "role": "foreground"}],
                    }
                ]
            }
        ],
    }

    result = check_structure_candidate(_approved_flow(), response, solo_piano_3m_v2_capabilities())

    assert result.valid is False
    assert any(issue.path == "/materials/0" for issue in result.issues)


def test_python_assigns_stable_ids_and_preserves_material_reuse() -> None:
    response = {
        "materials": [{"description": "繰り返して使う短い主題"}],
        "scenes": [
            {
                "sections": [
                    {
                        "depth": 0,
                        "role": "statement",
                        "description": "主題を示す",
                        "relative_length": 1,
                        "placements": [{"material_index": 0, "role": "foreground"}],
                    },
                    {
                        "depth": 0,
                        "role": "return",
                        "description": "主題を再び示す",
                        "relative_length": 1,
                        "placements": [{"material_index": 0, "role": "foreground"}],
                    },
                ]
            }
        ],
    }

    result = index_structure_candidate(
        _approved_flow(), "composition-001", response, solo_piano_3m_v2_capabilities()
    )

    assert result.valid is True
    assert result.indexed_structure is not None
    script = result.indexed_structure["script"]
    assert list(script["sections"]) == ["section-root", "section-001", "section-002"]
    assert list(script["materials"]) == ["material-001"]
    assert list(script["material_placements"]) == [
        "material-placement-001",
        "material-placement-002",
    ]
    assert {placement["material_id"] for placement in script["material_placements"].values()} == {
        "material-001"
    }


def test_structure_validation_lists_independent_depth_leaf_and_reference_problems() -> None:
    response = {
        "materials": [{"description": "短い主題"}],
        "scenes": [
            {
                "sections": [
                    {
                        "depth": 1,
                        "role": "statement",
                        "description": "誤った枝",
                        "relative_length": 1,
                        "placements": [{"material_index": 4, "role": "foreground"}],
                    },
                    {
                        "depth": 3,
                        "role": "return",
                        "description": "誤った葉",
                        "relative_length": None,
                        "placements": [],
                    },
                ]
            }
        ],
    }

    result = check_structure_candidate(_approved_flow(), response, solo_piano_3m_v2_capabilities())

    assert result.valid is False
    assert len(result.issues) >= 5
    assert {issue.path for issue in result.issues} >= {
        "/scenes/0/sections/0/depth",
        "/scenes/0/sections/1/depth",
        "/scenes/0/sections/0",
        "/scenes/0/sections/0/placements/0/material_index",
        "/scenes/0/sections/1/relative_length",
        "/scenes/0/sections/1/placements",
    }


def test_relations_schema_uses_only_python_assigned_targets_and_no_final_relation_ids() -> None:
    structure = {
        "materials": [{"description": "短い主題"}],
        "scenes": [
            {
                "sections": [
                    {
                        "depth": 0,
                        "role": "statement",
                        "description": "主題を示す",
                        "relative_length": 1,
                        "placements": [{"material_index": 0, "role": "foreground"}],
                    }
                ]
            }
        ],
    }
    indexed = index_structure_candidate(
        _approved_flow(), "composition-001", structure, solo_piano_3m_v2_capabilities()
    ).indexed_structure
    assert indexed is not None

    schema = relations_response_schema(indexed, solo_piano_3m_v2_capabilities())
    direction = schema["properties"]["performance_directions"]["items"]
    variation = schema["properties"]["variation_relations"]["items"]

    assert direction["properties"]["target_type"]["enum"] == ["section"]
    assert direction["properties"]["target_id"]["enum"] == [
        "section-001",
        "section-root",
    ]
    assert "id" not in variation["properties"]


def test_relations_prompt_preserves_the_indexed_structure_boundary() -> None:
    structure = {
        "materials": [{"description": "短い主題"}],
        "scenes": [
            {
                "sections": [
                    {
                        "depth": 0,
                        "role": "statement",
                        "description": "主題を示す",
                        "relative_length": 1,
                        "placements": [{"material_index": 0, "role": "foreground"}],
                    }
                ]
            }
        ],
    }
    indexed = index_structure_candidate(
        _approved_flow(), "composition-001", structure, solo_piano_3m_v2_capabilities()
    ).indexed_structure
    assert indexed is not None

    prompt = relations_prompt(_approved_flow(), indexed, solo_piano_3m_v2_capabilities())

    assert '"material-placement-001"' in prompt
    assert '"performance_direction_target_types": ["section"]' in prompt
    assert "区分とマテリアル配置を変更しない" in prompt
    assert "structure_insufficient" in prompt


def test_model_response_schemas_avoid_unsupported_one_of() -> None:
    capabilities = solo_piano_3m_v2_capabilities()
    structure = {
        "materials": [{"description": "短い主題"}],
        "scenes": [
            {
                "sections": [
                    {
                        "depth": 0,
                        "role": "statement",
                        "description": "主題を示す",
                        "relative_length": 1,
                        "placements": [{"material_index": 0, "role": "foreground"}],
                    }
                ]
            }
        ],
    }
    indexed = index_structure_candidate(
        _approved_flow(), "composition-001", structure, capabilities
    ).indexed_structure
    assert indexed is not None

    schemas = [
        structure_response_schema(capabilities),
        relations_response_schema(indexed, capabilities),
    ]

    assert all('"oneOf"' not in json.dumps(schema) for schema in schemas)


def test_python_builds_final_relation_ids_and_a_valid_script_document() -> None:
    structure = {
        "materials": [
            {"description": "繰り返して使う短い主題"},
            {"description": "主題をつなぐ短い動き"},
        ],
        "scenes": [
            {
                "sections": [
                    {
                        "depth": 0,
                        "role": "statement",
                        "description": "主題を示す",
                        "relative_length": 1,
                        "placements": [{"material_index": 0, "role": "foreground"}],
                    },
                    {
                        "depth": 0,
                        "role": "transition",
                        "description": "再提示へ移る",
                        "relative_length": 0.25,
                        "placements": [{"material_index": 1, "role": "foreground"}],
                    },
                    {
                        "depth": 0,
                        "role": "return",
                        "description": "主題を変えて戻す",
                        "relative_length": 1,
                        "placements": [{"material_index": 0, "role": "foreground"}],
                    },
                ]
            }
        ],
    }
    indexed = index_structure_candidate(
        _approved_flow(), "composition-001", structure, solo_piano_3m_v2_capabilities()
    ).indexed_structure
    assert indexed is not None
    response = {
        "outcome": "complete",
        "structure_insufficient_reason": None,
        "variation_relations": [
            {
                "source": {"type": "material_placement", "id": "material-placement-001"},
                "target": {"type": "material_placement", "id": "material-placement-003"},
                "preserve": ["主題の輪郭"],
                "change": ["終わり方"],
                "description": "主題の輪郭を保ちながら穏やかに戻す",
            }
        ],
        "material_placement_transitions": [
            {
                "source_material_placement_id": "material-placement-001",
                "transition_material_placement_id": "material-placement-002",
                "target_material_placement_id": "material-placement-003",
                "description": "主題から再提示へ移る",
            }
        ],
        "performance_directions": [
            {
                "target_type": "section",
                "target_id": "section-003",
                "relative_to_type": "section",
                "relative_to_id": "section-001",
                "description": "最初より縦の線をそろえる",
            }
        ],
        "performance_comparison_requirements": [
            {
                "performance_direction_index": 0,
                "feature": "onset_alignment",
                "relation": "more",
            }
        ],
    }

    result = build_script_document(
        _approved_flow(), indexed, response, solo_piano_3m_v2_capabilities()
    )

    assert result.valid is True
    assert result.document is not None
    assert check_script_0_3_document(result.document).valid is True
    script = result.document["script"]
    assert list(script["script_element_variation_relations"]) == ["variation-001"]
    assert list(script["material_placement_transitions"]) == ["material-placement-transition-001"]
    assert list(script["performance_setup"]["performance_directions"]) == [
        "performance-direction-001"
    ]
    assert list(script["performance_direction_comparison_requirements"]) == [
        "performance-comparison-001"
    ]
    flow_targets = {
        entry.source_id
        for entry in result.projection_ledger
        if entry.source_kind in {"flow_field", "flow_scene_field"}
    }
    assert flow_targets == {
        "/title",
        "/overall_flow",
        "/instrumentation",
        "/target_duration",
        "/scenes/0/scene_id",
        "/scenes/0/name",
        "/scenes/0/length_class",
        "/scenes/0/heard_as",
        "/scenes/0/relation_to_previous",
        "/scenes/0/transition_to_next",
    }
    assert all(
        entry.status == "passed" and entry.evidence
        for entry in result.projection_ledger
        if entry.verification != "natural_language_claim"
    )


def test_structure_insufficient_is_accepted_only_with_a_reason_and_empty_candidates() -> None:
    structure = {
        "materials": [{"description": "短い主題"}],
        "scenes": [
            {
                "sections": [
                    {
                        "depth": 0,
                        "role": "statement",
                        "description": "主題を示す",
                        "relative_length": 1,
                        "placements": [{"material_index": 0, "role": "foreground"}],
                    }
                ]
            }
        ],
    }
    indexed = index_structure_candidate(
        _approved_flow(), "composition-001", structure, solo_piano_3m_v2_capabilities()
    ).indexed_structure
    assert indexed is not None
    response = {
        "outcome": "structure_insufficient",
        "structure_insufficient_reason": "再提示と元の主題を区別できる構造がない",
        "variation_relations": [],
        "material_placement_transitions": [],
        "performance_directions": [],
        "performance_comparison_requirements": [],
    }

    result = build_script_document(
        _approved_flow(), indexed, response, solo_piano_3m_v2_capabilities()
    )

    assert result.valid is False
    assert result.outcome == "structure_insufficient"
    assert result.document is None
    assert result.issues == ()


def test_relation_validation_lists_independent_direction_problems() -> None:
    structure = {
        "materials": [{"description": "短い主題"}],
        "scenes": [
            {
                "sections": [
                    {
                        "depth": 0,
                        "role": "statement",
                        "description": "主題を示す",
                        "relative_length": 1,
                        "placements": [{"material_index": 0, "role": "foreground"}],
                    }
                ]
            }
        ],
    }
    indexed = index_structure_candidate(
        _approved_flow(), "composition-001", structure, solo_piano_3m_v2_capabilities()
    ).indexed_structure
    assert indexed is not None
    response = {
        "outcome": "complete",
        "structure_insufficient_reason": None,
        "variation_relations": [],
        "material_placement_transitions": [],
        "performance_directions": [
            {
                "target_type": "section",
                "target_id": "section-001",
                "relative_to_type": "section",
                "relative_to_id": None,
                "description": "比較対象が欠けている",
            },
            {
                "target_type": "section",
                "target_id": "section-001",
                "relative_to_type": None,
                "relative_to_id": "section-root",
                "description": "比較種別が欠けている",
            },
        ],
        "performance_comparison_requirements": [
            {
                "performance_direction_index": 99,
                "feature": "onset_alignment",
                "relation": "more",
            }
        ],
    }

    result = build_script_document(
        _approved_flow(), indexed, response, solo_piano_3m_v2_capabilities()
    )

    assert result.valid is False
    assert {issue.path for issue in result.issues} == {
        "/performance_directions/0",
        "/performance_directions/1",
        "/performance_comparison_requirements/0/performance_direction_index",
    }


def test_final_script_is_rechecked_against_the_generation_profile() -> None:
    structure = {
        "materials": [{"description": "短い主題"}],
        "scenes": [
            {
                "sections": [
                    {
                        "depth": 0,
                        "role": "statement",
                        "description": "主題を示す",
                        "relative_length": 1,
                        "placements": [{"material_index": 0, "role": "foreground"}],
                    }
                ]
            }
        ],
    }
    indexed = index_structure_candidate(
        _approved_flow(), "composition-001", structure, solo_piano_3m_v2_capabilities()
    ).indexed_structure
    assert indexed is not None
    indexed["script"]["performance_setup"]["instrumentation"] = "string_quartet"
    response = {
        "outcome": "complete",
        "structure_insufficient_reason": None,
        "variation_relations": [],
        "material_placement_transitions": [],
        "performance_directions": [],
        "performance_comparison_requirements": [],
    }

    result = build_script_document(
        _approved_flow(), indexed, response, solo_piano_3m_v2_capabilities()
    )

    assert result.valid is False
    assert result.outcome == "content_invalid"
    assert any(issue.path == "/script/performance_setup/instrumentation" for issue in result.issues)
