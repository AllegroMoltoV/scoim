import json

from scoim.profile_capabilities import solo_piano_3m_v2_capabilities
from scoim.script_0_4_model_contracts import (
    build_script_document,
    index_structure_candidate,
    relations_prompt,
    relations_response_schema,
    structure_prompt,
    structure_response_schema,
)


def _approved_flow() -> dict[str, object]:
    return {
        "document_type": "flow",
        "schema_version": "0.1.0",
        "document_id": "flow-001",
        "revision": 1,
        "status": "approved",
        "approval": {"content_sha256": "a" * 64},
        "title": "朝の即興曲",
        "overall_flow": "静かな冒頭から少しずつ前へ進む。",
        "instrumentation": "solo_piano",
        "target_duration": 180,
        "scenes": [
            {
                "scene_id": "scene-001",
                "name": "冒頭",
                "length_class": "long",
                "heard_as": "低い音から始まり、旋律の音域が少しずつ広がる。",
                "relation_to_previous": "初登場",
                "transition_to_next": "流れを止めずに進む。",
            }
        ],
    }


def _indexed_structure() -> dict[str, object]:
    result = index_structure_candidate(
        _approved_flow(),
        "composition-001",
        {
            "materials": [
                {"description": "ゆるやかに上がる短い旋律。"},
                {"description": "旋律を支える低音。"},
            ],
            "scenes": [
                {
                    "sections": [
                        {
                            "depth": 0,
                            "role": "statement",
                            "structural_purpose": "regular",
                            "description": "低音から始め、旋律の音域を広げる。",
                            "relative_length": 1,
                            "placements": [
                                {"material_index": 0, "role": "foreground"},
                                {"material_index": 1, "role": "accompaniment"},
                            ],
                        }
                    ]
                }
            ],
        },
        solo_piano_3m_v2_capabilities(),
    )
    assert result.indexed_structure is not None
    return result.indexed_structure


def _indexed_structure_with_return() -> dict[str, object]:
    approved_flow = _approved_flow()
    approved_flow["scenes"].append(
        {
            "scene_id": "scene-002",
            "name": "回帰",
            "length_class": "long",
            "heard_as": "冒頭を思い出しながら閉じる。",
            "relation_to_previous": "冒頭の再現。",
            "transition_to_next": "静かに閉じる。",
        }
    )
    result = index_structure_candidate(
        approved_flow,
        "composition-002",
        {
            "materials": [
                {"description": "ゆるやかに上がる短い旋律。"},
                {"description": "旋律を支える低音。"},
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
                            "description": "主題を再現する。",
                            "relative_length": 1,
                            "placements": [
                                {"material_index": 0, "role": "foreground"},
                                {"material_index": 1, "role": "accompaniment"},
                            ],
                        }
                    ]
                },
            ],
        },
        solo_piano_3m_v2_capabilities(),
    )
    assert result.indexed_structure is not None
    return result.indexed_structure


def _transition_structure_response() -> dict[str, object]:
    return {
        "materials": [
            {"description": "主題。"},
            {"description": "短い遷移。"},
            {"description": "戻った主題。"},
        ],
        "scenes": [
            {
                "sections": [
                    {
                        "depth": 0,
                        "role": "statement",
                        "structural_purpose": "regular",
                        "description": "主題を示す。",
                        "relative_length": 2,
                        "placements": [{"material_index": 0, "role": "foreground"}],
                    },
                    {
                        "depth": 0,
                        "role": "bridge",
                        "structural_purpose": "transition_connector",
                        "description": "短く方向を変える。",
                        "relative_length": 1,
                        "placements": [{"material_index": 1, "role": "foreground"}],
                    },
                    {
                        "depth": 0,
                        "role": "return",
                        "structural_purpose": "regular",
                        "description": "主題へ戻る。",
                        "relative_length": 2,
                        "placements": [{"material_index": 2, "role": "foreground"}],
                    },
                ]
            }
        ],
    }


def _indexed_structure_with_transition() -> dict[str, object]:
    result = index_structure_candidate(
        _approved_flow(),
        "composition-transition",
        _transition_structure_response(),
        solo_piano_3m_v2_capabilities(),
    )
    assert result.indexed_structure is not None
    return result.indexed_structure


def _indexed_structure_with_two_transitions() -> dict[str, object]:
    response = {
        "materials": [{"description": f"素材{index}"} for index in range(5)],
        "scenes": [
            {
                "sections": [
                    {
                        "depth": 0,
                        "role": "statement",
                        "structural_purpose": "regular",
                        "description": "二つの前景で始める。",
                        "relative_length": 2,
                        "placements": [
                            {"material_index": 0, "role": "foreground"},
                            {"material_index": 1, "role": "foreground"},
                        ],
                    },
                    {
                        "depth": 0,
                        "role": "bridge-one",
                        "structural_purpose": "transition_connector",
                        "description": "一つ目の遷移。",
                        "relative_length": 1,
                        "placements": [{"material_index": 2, "role": "foreground"}],
                    },
                    {
                        "depth": 0,
                        "role": "middle",
                        "structural_purpose": "regular",
                        "description": "中間部。",
                        "relative_length": 2,
                        "placements": [{"material_index": 3, "role": "foreground"}],
                    },
                    {
                        "depth": 0,
                        "role": "bridge-two",
                        "structural_purpose": "transition_connector",
                        "description": "二つ目の遷移。",
                        "relative_length": 1,
                        "placements": [{"material_index": 2, "role": "foreground"}],
                    },
                    {
                        "depth": 0,
                        "role": "return",
                        "structural_purpose": "regular",
                        "description": "戻る。",
                        "relative_length": 2,
                        "placements": [{"material_index": 4, "role": "foreground"}],
                    },
                ]
            }
        ],
    }
    result = index_structure_candidate(
        _approved_flow(),
        "composition-two-transitions",
        response,
        solo_piano_3m_v2_capabilities(),
    )
    assert result.indexed_structure is not None
    return result.indexed_structure


def test_phase2_structure_contract_requires_a_typed_structural_purpose() -> None:
    schema = structure_response_schema(solo_piano_3m_v2_capabilities())
    section_schema = schema["properties"]["scenes"]["items"]["properties"]["sections"]["items"]

    assert "structural_purpose" in section_schema["required"]
    assert section_schema["properties"]["structural_purpose"] == {
        "enum": ["regular", "transition_connector"]
    }


def test_phase2_structure_prompt_does_not_treat_every_transition_sentence_as_a_connector() -> None:
    prompt = structure_prompt(_approved_flow(), solo_piano_3m_v2_capabilities())

    assert "transition_to_nextは聞こえ方の文章" in prompt
    assert "それだけを理由にtransition_connectorを作らない" in prompt


def test_phase2_indexes_a_valid_transition_connector_without_publishing_its_purpose() -> None:
    result = index_structure_candidate(
        _approved_flow(),
        "composition-transition",
        _transition_structure_response(),
        solo_piano_3m_v2_capabilities(),
    )

    assert result.valid is True, result.issues
    assert result.indexed_structure is not None
    assert result.indexed_structure["transition_connector_section_ids"] == ["section-002"]
    sections = result.indexed_structure["script"]["sections"]
    assert all("structural_purpose" not in section for section in sections.values())


def test_phase2_rejects_an_accompaniment_beside_a_transition_connector() -> None:
    response = _transition_structure_response()
    response["scenes"][0]["sections"][1]["placements"].append(
        {"material_index": 0, "role": "accompaniment"}
    )

    result = index_structure_candidate(
        _approved_flow(),
        "composition-transition",
        response,
        solo_piano_3m_v2_capabilities(),
    )

    assert result.valid is False
    assert [(issue.message, issue.path) for issue in result.issues] == [
        (
            "A transition connector must contain exactly one foreground placement",
            "/scenes/0/sections/1/placements",
        )
    ]


def test_phase2_rejects_a_branch_marked_as_a_transition_connector() -> None:
    response = _transition_structure_response()
    sections = response["scenes"][0]["sections"]
    sections[0]["structural_purpose"] = "transition_connector"
    sections[0]["relative_length"] = None
    sections[0]["placements"] = []
    sections[1]["depth"] = 1
    sections[1]["structural_purpose"] = "regular"

    result = index_structure_candidate(
        _approved_flow(),
        "composition-transition",
        response,
        solo_piano_3m_v2_capabilities(),
    )

    assert result.valid is False
    assert (result.issues[-1].message, result.issues[-1].path) == (
        "A branch section must have regular structural purpose",
        "/scenes/0/sections/0/structural_purpose",
    )


def test_phase2_rejects_a_transition_connector_without_a_preceding_foreground() -> None:
    response = _transition_structure_response()
    response["scenes"][0]["sections"] = response["scenes"][0]["sections"][1:]

    result = index_structure_candidate(
        _approved_flow(),
        "composition-transition",
        response,
        solo_piano_3m_v2_capabilities(),
    )

    assert result.valid is False
    assert (result.issues[0].message, result.issues[0].path) == (
        "A transition connector requires a preceding regular foreground placement",
        "/scenes/0/sections/0/structural_purpose",
    )


def test_phase2_relations_contract_offers_only_derived_transition_candidates() -> None:
    indexed = _indexed_structure_with_transition()
    capabilities = solo_piano_3m_v2_capabilities()

    prompt = relations_prompt(_approved_flow(), indexed, capabilities)
    prompt_input = json.loads(prompt.split("入力: ", maxsplit=1)[1])
    groups = prompt_input["transition_candidate_groups"]
    schema = relations_response_schema(indexed, capabilities)
    transitions = schema["properties"]["material_placement_transitions"]

    assert groups == [
        {
            "transition_connector_section_id": "section-002",
            "candidates": [
                {
                    "candidate_id": "transition-candidate-001",
                    "source_material_placement_id": "material-placement-001",
                    "transition_material_placement_id": "material-placement-002",
                    "target_material_placement_id": "material-placement-003",
                }
            ],
        }
    ]
    assert transitions["minItems"] == 1
    assert transitions["maxItems"] == 1
    assert transitions["items"]["required"] == ["candidate_id", "description"]
    assert transitions["items"]["properties"]["candidate_id"] == {
        "enum": ["transition-candidate-001"]
    }
    assert "source_material_placement_id" not in transitions["items"]["properties"]


def test_phase2_expands_a_selected_transition_candidate_into_the_public_document() -> None:
    indexed = _indexed_structure_with_transition()
    response = {
        "outcome": "complete",
        "structure_insufficient_reason": None,
        "variation_relations": [],
        "material_placement_transitions": [
            {
                "candidate_id": "transition-candidate-001",
                "description": "主題から回帰へ短く向きを変える。",
            }
        ],
        "performance_directions": [],
    }

    result = build_script_document(
        _approved_flow(), indexed, response, solo_piano_3m_v2_capabilities()
    )

    assert result.valid is True, result.issues
    assert result.document is not None
    transitions = result.document["script"]["material_placement_transitions"]
    assert transitions == {
        "material-placement-transition-001": {
            "source_material_placement_id": "material-placement-001",
            "transition_material_placement_id": "material-placement-002",
            "target_material_placement_id": "material-placement-003",
            "description": "主題から回帰へ短く向きを変える。",
        }
    }


def test_phase2_does_not_allow_a_transition_connector_to_be_left_unselected() -> None:
    indexed = _indexed_structure_with_transition()
    response = {
        "outcome": "complete",
        "structure_insufficient_reason": None,
        "variation_relations": [],
        "material_placement_transitions": [],
        "performance_directions": [],
    }

    result = build_script_document(
        _approved_flow(), indexed, response, solo_piano_3m_v2_capabilities()
    )

    assert result.valid is False
    assert result.outcome == "content_invalid"
    assert [issue.path for issue in result.issues] == ["/material_placement_transitions"]


def test_phase2_excludes_a_transition_connector_from_explicit_variation_relations() -> None:
    indexed = _indexed_structure_with_transition()
    response = {
        "outcome": "complete",
        "structure_insufficient_reason": None,
        "variation_relations": [
            {
                "source": {"type": "material_placement", "id": "material-placement-001"},
                "target": {"type": "material_placement", "id": "material-placement-002"},
                "preserve": ["輪郭"],
                "change": ["勢い"],
                "description": "遷移へ変奏する。",
            }
        ],
        "material_placement_transitions": [
            {
                "candidate_id": "transition-candidate-001",
                "description": "主題から回帰へ短く向きを変える。",
            }
        ],
        "performance_directions": [],
    }

    result = build_script_document(
        _approved_flow(), indexed, response, solo_piano_3m_v2_capabilities()
    )

    assert result.valid is False
    assert result.outcome == "content_invalid"
    assert [issue.path for issue in result.issues] == ["/variation_relations/0/target/id"]


def test_phase2_requires_one_selection_from_each_transition_connector_group() -> None:
    indexed = _indexed_structure_with_two_transitions()
    response = {
        "outcome": "complete",
        "structure_insufficient_reason": None,
        "variation_relations": [],
        "material_placement_transitions": [
            {"candidate_id": "transition-candidate-001", "description": "候補一。"},
            {"candidate_id": "transition-candidate-002", "description": "候補二。"},
        ],
        "performance_directions": [],
    }

    result = build_script_document(
        _approved_flow(), indexed, response, solo_piano_3m_v2_capabilities()
    )

    assert result.valid is False
    assert result.outcome == "content_invalid"
    assert [issue.path for issue in result.issues] == [
        "/material_placement_transitions/1/candidate_id",
        "/material_placement_transitions",
    ]


def test_phase2_lists_all_nearest_foreground_combinations_in_stable_order() -> None:
    indexed = _indexed_structure_with_two_transitions()
    prompt = relations_prompt(_approved_flow(), indexed, solo_piano_3m_v2_capabilities())
    groups = json.loads(prompt.split("入力: ", maxsplit=1)[1])["transition_candidate_groups"]

    assert [
        [candidate["candidate_id"] for candidate in group["candidates"]] for group in groups
    ] == [
        ["transition-candidate-001", "transition-candidate-002"],
        ["transition-candidate-003"],
    ]


def test_phase2_relations_contract_exposes_only_supported_performance_aspects() -> None:
    capabilities = solo_piano_3m_v2_capabilities()
    indexed = _indexed_structure()

    prompt = relations_prompt(_approved_flow(), indexed, capabilities)
    schema = relations_response_schema(indexed, capabilities)
    prompt_input = json.loads(prompt.split("入力: ", maxsplit=1)[1])
    direction_schema = schema["properties"]["performance_directions"]["items"]

    assert prompt_input["profile_capabilities"]["performance_aspects"] == [
        {
            "aspect_id": aspect.aspect_id,
            "description": aspect.description,
        }
        for aspect in capabilities.performance_aspects
    ]
    assert prompt_input["profile_capabilities"]["score_relations"] == {
        "explicit_variation_roles": ["foreground"],
        "transition_roles": ["foreground"],
        "accompaniment_reuse": "implicit_previous_same_material",
        "instructions": [
            "明示的な変奏は、伴奏として使うマテリアルまたはマテリアル配置を含めない。",
            "遷移の元、遷移専用、先は、すべて前景のマテリアル配置にする。",
            "伴奏で同じマテリアルを再利用するときは、明示的な変奏を作らない。",
        ],
    }
    assert direction_schema["required"] == [
        "target_type",
        "target_id",
        "relative_to_type",
        "relative_to_id",
        "performance_aspects",
        "description",
    ]
    assert direction_schema["properties"]["performance_aspects"]["items"]["enum"] == list(
        capabilities.performance_aspect_ids
    )
    assert "音域、旋律、伴奏音、和音の要望を演奏指示へ複写しない" in prompt
    assert "比較条件" not in prompt
    assert "performance_comparison_requirements" not in schema["required"]


def test_phase2_relations_schema_uses_only_codex_supported_array_keywords() -> None:
    schema = relations_response_schema(_indexed_structure(), solo_piano_3m_v2_capabilities())

    direction_schema = schema["properties"]["performance_directions"]["items"]

    assert "uniqueItems" not in direction_schema["properties"]["performance_aspects"]


def test_phase2_builds_a_script_0_4_document_without_comparison_requirements() -> None:
    indexed = _indexed_structure()
    response = {
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
                "performance_aspects": ["timing", "dynamics"],
                "description": "少し溜めながら、音量に穏やかな山を作る。",
            }
        ],
    }

    result = build_script_document(
        _approved_flow(), indexed, response, solo_piano_3m_v2_capabilities()
    )

    assert result.valid is True
    assert result.document is not None
    assert result.document["schema_version"] == "0.4.0"
    script = result.document["script"]
    assert "performance_direction_comparison_requirements" not in script
    assert script["performance_setup"]["performance_directions"] == {
        "performance-direction-001": {
            "target": {"type": "section", "id": "section-001"},
            "performance_aspects": ["timing", "dynamics"],
            "description": "少し溜めながら、音量に穏やかな山を作る。",
        }
    }


def test_phase2_rejects_a_relation_that_phase4_and_phase5_cannot_realize() -> None:
    indexed = _indexed_structure_with_return()
    placements = indexed["script"]["material_placements"]
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
    response = {
        "outcome": "complete",
        "structure_insufficient_reason": None,
        "variation_relations": [
            {
                "source": {"type": "material_placement", "id": foreground_ids[0]},
                "target": {"type": "material_placement", "id": accompaniment_ids[-1]},
                "preserve": ["輪郭"],
                "change": ["リズム"],
                "description": "前景を伴奏へ変奏する。",
            }
        ],
        "material_placement_transitions": [],
        "performance_directions": [],
    }

    result = build_script_document(
        _approved_flow(), indexed, response, solo_piano_3m_v2_capabilities()
    )

    assert result.valid is False
    assert [(issue.code.value, issue.path) for issue in result.issues] == [
        (
            "unrepresentable",
            "/script/script_element_variation_relations/variation-001",
        )
    ]
