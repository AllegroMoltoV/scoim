import hashlib
import json
from importlib.resources import files

import rfc8785
from jsonschema import Draft202012Validator

from scoim.validation import IssueCode, check


def _minimal_document() -> dict[str, object]:
    return {
        "schema_version": "0.1.0",
        "document_id": "small_song",
        "revision": 1,
        "status": "draft",
        "approval": None,
        "script": {
            "title": "小さな曲",
            "brief": "一つの素材を短く提示する。",
            "performance_setup": {
                "instrumentation": "solo_piano",
                "target_duration_seconds": 30,
                "performance_directions": {},
            },
            "root_section_id": "whole",
            "sections": {
                "whole": {
                    "parent_section_id": None,
                    "order": 0,
                    "role": "whole",
                    "description": "曲全体",
                },
                "statement": {
                    "parent_section_id": "whole",
                    "order": 0,
                    "role": "statement",
                    "relative_length": 1,
                    "description": "素材を提示する。",
                },
            },
            "materials": {
                "theme": {"kind": "theme", "description": "中心素材。"},
            },
            "placements": {
                "theme_first": {"section_id": "statement", "material_id": "theme"},
            },
            "variations": {},
            "requirements": {},
            "transitions": {},
        },
    }


def _document_with_requirement() -> dict[str, object]:
    document = _minimal_document()
    script = document["script"]
    assert isinstance(script, dict)
    sections = script["sections"]
    assert isinstance(sections, dict)
    sections["return"] = {
        "parent_section_id": "whole",
        "order": 1,
        "role": "return",
        "relative_length": 1,
        "description": "素材を再提示する。",
    }
    setup = script["performance_setup"]
    assert isinstance(setup, dict)
    directions = setup["performance_directions"]
    assert isinstance(directions, dict)
    directions["clear_return"] = {
        "target": {"type": "section", "id": "return"},
        "relative_to": {"type": "section", "id": "statement"},
        "description": "再提示では縦の線をそろえる。",
    }
    requirements = script["requirements"]
    assert isinstance(requirements, dict)
    requirements["return_more_aligned"] = {
        "performance_direction_id": "clear_return",
        "feature": "onset_alignment",
        "relation": "more",
    }
    return document


def test_minimal_script_document_conforms_to_public_schema() -> None:
    schema_text = files("scoim").joinpath("schemas", "script-0.1.0.schema.json").read_text("utf-8")
    schema = json.loads(schema_text)
    document = _minimal_document()

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(document)

    assert json.loads(json.dumps(document, ensure_ascii=False)) == document


def test_check_accepts_a_typed_requirement_linked_to_a_comparative_direction() -> None:
    document = _document_with_requirement()

    result = check(document)

    assert result.valid is True
    assert result.issues == ()


def test_check_rejects_a_draft_without_requirements() -> None:
    document = _minimal_document()
    script = document["script"]
    assert isinstance(script, dict)
    del script["requirements"]

    result = check(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.SEMANTIC_INVALID
    assert result.issues[0].path == "/script/requirements"


def test_check_rejects_a_requirement_with_a_missing_performance_direction() -> None:
    document = _document_with_requirement()
    script = document["script"]
    assert isinstance(script, dict)
    requirements = script["requirements"]
    assert isinstance(requirements, dict)
    requirement = requirements["return_more_aligned"]
    assert isinstance(requirement, dict)
    requirement["performance_direction_id"] = "missing"

    result = check(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.SEMANTIC_INVALID
    assert result.issues[0].path == (
        "/script/requirements/return_more_aligned/performance_direction_id"
    )


def test_check_rejects_a_requirement_whose_direction_has_no_comparison() -> None:
    document = _document_with_requirement()
    script = document["script"]
    assert isinstance(script, dict)
    setup = script["performance_setup"]
    assert isinstance(setup, dict)
    directions = setup["performance_directions"]
    assert isinstance(directions, dict)
    direction = directions["clear_return"]
    assert isinstance(direction, dict)
    del direction["relative_to"]

    result = check(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.SEMANTIC_INVALID
    assert result.issues[0].path == (
        "/script/requirements/return_more_aligned/performance_direction_id"
    )


def test_check_rejects_a_requirement_whose_direction_targets_a_placement() -> None:
    document = _document_with_requirement()
    script = document["script"]
    assert isinstance(script, dict)
    setup = script["performance_setup"]
    assert isinstance(setup, dict)
    directions = setup["performance_directions"]
    assert isinstance(directions, dict)
    direction = directions["clear_return"]
    assert isinstance(direction, dict)
    direction["target"] = {"type": "placement", "id": "theme_first"}

    result = check(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.SEMANTIC_INVALID
    assert result.issues[0].path == (
        "/script/requirements/return_more_aligned/performance_direction_id"
    )


def test_check_rejects_a_requirement_whose_direction_compares_a_placement() -> None:
    document = _document_with_requirement()
    script = document["script"]
    setup = script["performance_setup"]
    directions = setup["performance_directions"]
    direction = directions["clear_return"]
    direction["relative_to"] = {"type": "placement", "id": "theme_first"}

    result = check(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.SEMANTIC_INVALID
    assert result.issues[0].path == (
        "/script/requirements/return_more_aligned/performance_direction_id"
    )


def test_check_rejects_a_requirement_that_compares_a_section_with_itself() -> None:
    document = _document_with_requirement()
    script = document["script"]
    assert isinstance(script, dict)
    setup = script["performance_setup"]
    assert isinstance(setup, dict)
    directions = setup["performance_directions"]
    assert isinstance(directions, dict)
    direction = directions["clear_return"]
    assert isinstance(direction, dict)
    direction["relative_to"] = {"type": "section", "id": "return"}

    result = check(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.SEMANTIC_INVALID
    assert result.issues[0].path == (
        "/script/requirements/return_more_aligned/performance_direction_id"
    )


def test_check_rejects_a_requirement_between_ancestor_and_descendant_sections() -> None:
    document = _document_with_requirement()
    script = document["script"]
    assert isinstance(script, dict)
    setup = script["performance_setup"]
    assert isinstance(setup, dict)
    directions = setup["performance_directions"]
    assert isinstance(directions, dict)
    direction = directions["clear_return"]
    assert isinstance(direction, dict)
    direction["relative_to"] = {"type": "section", "id": "whole"}

    result = check(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.SEMANTIC_INVALID
    assert result.issues[0].path == (
        "/script/requirements/return_more_aligned/performance_direction_id"
    )


def test_check_rejects_duplicate_requirements_for_the_same_direction_and_feature() -> None:
    document = _document_with_requirement()
    script = document["script"]
    assert isinstance(script, dict)
    requirements = script["requirements"]
    assert isinstance(requirements, dict)
    requirements["same_comparison_again"] = {
        "performance_direction_id": "clear_return",
        "feature": "onset_alignment",
        "relation": "less",
    }

    result = check(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.SEMANTIC_INVALID
    assert result.issues[0].path == "/script/requirements/same_comparison_again/feature"


def test_check_rejects_a_requirement_cycle_after_normalizing_less_relations() -> None:
    document = _document_with_requirement()
    script = document["script"]
    assert isinstance(script, dict)
    setup = script["performance_setup"]
    assert isinstance(setup, dict)
    directions = setup["performance_directions"]
    assert isinstance(directions, dict)
    directions["restrained_statement"] = {
        "target": {"type": "section", "id": "statement"},
        "relative_to": {"type": "section", "id": "return"},
        "description": "提示を再提示より控える。",
    }
    requirements = script["requirements"]
    assert isinstance(requirements, dict)
    first = requirements["return_more_aligned"]
    assert isinstance(first, dict)
    first["relation"] = "less"
    requirements["statement_less_aligned"] = {
        "performance_direction_id": "restrained_statement",
        "feature": "onset_alignment",
        "relation": "less",
    }

    result = check(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.SEMANTIC_INVALID
    assert result.issues[0].path == "/script/requirements/statement_less_aligned/relation"


def test_check_rejects_an_unknown_requirement_feature() -> None:
    document = _document_with_requirement()
    requirement = document["script"]["requirements"]["return_more_aligned"]
    requirement["feature"] = "tempo"

    result = check(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.SCHEMA_INVALID
    assert result.issues[0].path == "/script/requirements/return_more_aligned/feature"


def test_check_rejects_an_unknown_requirement_relation() -> None:
    document = _document_with_requirement()
    requirement = document["script"]["requirements"]["return_more_aligned"]
    requirement["relation"] = "equal"

    result = check(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.SCHEMA_INVALID
    assert result.issues[0].path == "/script/requirements/return_more_aligned/relation"


def test_check_rejects_an_unknown_requirement_field() -> None:
    document = _document_with_requirement()
    requirement = document["script"]["requirements"]["return_more_aligned"]
    requirement["threshold"] = 1

    result = check(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.SCHEMA_INVALID
    assert result.issues[0].path == "/script/requirements/return_more_aligned"


def test_check_rejects_a_requirement_with_a_missing_field() -> None:
    document = _document_with_requirement()
    requirement = document["script"]["requirements"]["return_more_aligned"]
    del requirement["relation"]

    result = check(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.SCHEMA_INVALID
    assert result.issues[0].path == "/script/requirements/return_more_aligned"


def test_check_accepts_a_legacy_approved_document_without_requirements() -> None:
    document = _minimal_document()
    script = document["script"]
    assert isinstance(script, dict)
    del script["requirements"]
    document["status"] = "approved"
    hashed_content = {
        field: document[field] for field in ("schema_version", "document_id", "revision", "script")
    }
    document["approval"] = {
        "content_sha256": hashlib.sha256(rfc8785.dumps(hashed_content)).hexdigest()
    }

    result = check(document)

    assert result.valid is True
    assert result.issues == ()


def test_check_rejects_the_superseded_flat_performance_fields() -> None:
    document = _minimal_document()
    script = document["script"]
    assert isinstance(script, dict)
    performance_setup = script.pop("performance_setup")
    assert isinstance(performance_setup, dict)
    script["target_duration_seconds"] = performance_setup["target_duration_seconds"]
    script["performance_directions"] = performance_setup["performance_directions"]

    result = check(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.SCHEMA_INVALID
    assert result.issues[0].path == "/script"


def test_check_rejects_an_unknown_schema_version_with_a_typed_issue() -> None:
    document = _minimal_document()
    document["schema_version"] = "9.9.9"

    result = check(document)

    assert result.valid is False
    assert len(result.issues) == 1
    assert result.issues[0].code is IssueCode.UNSUPPORTED_SCHEMA_VERSION
    assert result.issues[0].path == "/schema_version"


def test_check_rejects_a_missing_required_field_as_schema_invalid() -> None:
    document = _minimal_document()
    del document["script"]

    result = check(document)

    assert result.valid is False
    assert len(result.issues) == 1
    assert result.issues[0].code is IssueCode.SCHEMA_INVALID
    assert result.issues[0].path == ""


def test_check_rejects_a_value_with_the_wrong_json_type() -> None:
    document = _minimal_document()
    document["revision"] = "1"

    result = check(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.SCHEMA_INVALID
    assert result.issues[0].path == "/revision"


def test_check_rejects_an_unknown_field() -> None:
    document = _minimal_document()
    document["extra"] = True

    result = check(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.SCHEMA_INVALID
    assert result.issues[0].path == ""


def test_check_rejects_a_value_outside_a_closed_vocabulary() -> None:
    document = _minimal_document()
    document["status"] = "review"

    result = check(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.SCHEMA_INVALID
    assert result.issues[0].path == "/status"


def test_check_rejects_a_placement_with_a_missing_section_reference() -> None:
    document = _minimal_document()
    script = document["script"]
    assert isinstance(script, dict)
    placements = script["placements"]
    assert isinstance(placements, dict)
    placement = placements["theme_first"]
    assert isinstance(placement, dict)
    placement["section_id"] = "missing"

    result = check(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.SEMANTIC_INVALID
    assert result.issues[0].path == "/script/placements/theme_first/section_id"


def test_check_rejects_a_cycle_in_section_containment() -> None:
    document = _minimal_document()
    script = document["script"]
    assert isinstance(script, dict)
    sections = script["sections"]
    assert isinstance(sections, dict)
    statement = sections["statement"]
    assert isinstance(statement, dict)
    statement["parent_section_id"] = "statement"

    result = check(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID
        and issue.path == "/script/sections/statement/parent_section_id"
        for issue in result.issues
    )


def test_check_rejects_a_designated_root_that_has_a_parent() -> None:
    document = _minimal_document()
    script = document["script"]
    assert isinstance(script, dict)
    script["root_section_id"] = "statement"

    result = check(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID
        and issue.path == "/script/sections/statement/parent_section_id"
        for issue in result.issues
    )


def test_check_rejects_a_gap_in_sibling_order() -> None:
    document = _minimal_document()
    script = document["script"]
    assert isinstance(script, dict)
    sections = script["sections"]
    assert isinstance(sections, dict)
    sections["second"] = {
        "parent_section_id": "whole",
        "order": 2,
        "role": "statement",
        "relative_length": 1,
        "description": "二つ目の素材を提示する。",
    }

    result = check(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID and issue.path == "/script/sections/second/order"
        for issue in result.issues
    )


def test_check_rejects_a_leaf_section_without_relative_length() -> None:
    document = _minimal_document()
    script = document["script"]
    assert isinstance(script, dict)
    sections = script["sections"]
    assert isinstance(sections, dict)
    statement = sections["statement"]
    assert isinstance(statement, dict)
    del statement["relative_length"]

    result = check(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID
        and issue.path == "/script/sections/statement/relative_length"
        for issue in result.issues
    )


def test_check_rejects_a_placement_on_a_branch_section() -> None:
    document = _minimal_document()
    script = document["script"]
    assert isinstance(script, dict)
    sections = script["sections"]
    assert isinstance(sections, dict)
    statement = sections["statement"]
    assert isinstance(statement, dict)
    del statement["relative_length"]
    sections["detail"] = {
        "parent_section_id": "statement",
        "order": 0,
        "role": "detail",
        "relative_length": 1,
        "description": "素材を詳しく示す。",
    }

    result = check(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID
        and issue.path == "/script/placements/theme_first/section_id"
        for issue in result.issues
    )


def test_check_rejects_a_variation_with_a_missing_source_reference() -> None:
    document = _minimal_document()
    script = document["script"]
    assert isinstance(script, dict)
    variations = script["variations"]
    assert isinstance(variations, dict)
    variations["changed_theme"] = {
        "source": {"type": "material", "id": "missing"},
        "target": {"type": "material", "id": "theme"},
        "preserve": ["中心となる音型"],
        "change": ["終わり方"],
        "description": "素材の終わり方を変える。",
    }

    result = check(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID
        and issue.path == "/script/variations/changed_theme/source/id"
        for issue in result.issues
    )


def test_check_rejects_a_variation_between_different_reference_types() -> None:
    document = _minimal_document()
    script = document["script"]
    assert isinstance(script, dict)
    variations = script["variations"]
    assert isinstance(variations, dict)
    variations["changed_theme"] = {
        "source": {"type": "material", "id": "theme"},
        "target": {"type": "section", "id": "statement"},
        "preserve": ["中心となる音型"],
        "change": ["終わり方"],
        "description": "素材の終わり方を変える。",
    }

    result = check(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID
        and issue.path == "/script/variations/changed_theme/target/type"
        for issue in result.issues
    )


def test_check_rejects_a_variation_that_targets_its_source() -> None:
    document = _minimal_document()
    script = document["script"]
    assert isinstance(script, dict)
    variations = script["variations"]
    assert isinstance(variations, dict)
    variations["unchanged_theme"] = {
        "source": {"type": "material", "id": "theme"},
        "target": {"type": "material", "id": "theme"},
        "preserve": ["中心となる音型"],
        "change": ["終わり方"],
        "description": "同じ素材を変奏として参照する。",
    }

    result = check(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID
        and issue.path == "/script/variations/unchanged_theme/target/id"
        for issue in result.issues
    )


def test_check_rejects_overlapping_preserve_and_change_in_a_variation() -> None:
    document = _minimal_document()
    script = document["script"]
    assert isinstance(script, dict)
    materials = script["materials"]
    assert isinstance(materials, dict)
    materials["theme_varied"] = {"kind": "theme", "description": "変奏した中心素材。"}
    variations = script["variations"]
    assert isinstance(variations, dict)
    variations["changed_theme"] = {
        "source": {"type": "material", "id": "theme"},
        "target": {"type": "material", "id": "theme_varied"},
        "preserve": ["中心となる音型"],
        "change": ["中心となる音型"],
        "description": "素材の終わり方を変える。",
    }

    result = check(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID
        and issue.path == "/script/variations/changed_theme/change/0"
        for issue in result.issues
    )


def test_check_rejects_multiple_variations_with_the_same_target() -> None:
    document = _minimal_document()
    script = document["script"]
    assert isinstance(script, dict)
    materials = script["materials"]
    assert isinstance(materials, dict)
    materials["second_theme"] = {"kind": "theme", "description": "別の中心素材。"}
    materials["theme_varied"] = {"kind": "theme", "description": "変奏した中心素材。"}
    variations = script["variations"]
    assert isinstance(variations, dict)
    variations["from_theme"] = {
        "source": {"type": "material", "id": "theme"},
        "target": {"type": "material", "id": "theme_varied"},
        "preserve": ["中心となる音型"],
        "change": ["終わり方"],
        "description": "中心素材を変奏する。",
    }
    variations["from_second_theme"] = {
        "source": {"type": "material", "id": "second_theme"},
        "target": {"type": "material", "id": "theme_varied"},
        "preserve": ["音域"],
        "change": ["中心となる音型"],
        "description": "別の素材から同じ変奏先を作る。",
    }

    result = check(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID
        and issue.path == "/script/variations/from_second_theme/target/id"
        for issue in result.issues
    )


def test_check_rejects_a_cycle_in_material_variations() -> None:
    document = _minimal_document()
    script = document["script"]
    assert isinstance(script, dict)
    materials = script["materials"]
    assert isinstance(materials, dict)
    materials["second_theme"] = {"kind": "theme", "description": "別の中心素材。"}
    variations = script["variations"]
    assert isinstance(variations, dict)
    variations["theme_to_second"] = {
        "source": {"type": "material", "id": "theme"},
        "target": {"type": "material", "id": "second_theme"},
        "preserve": ["中心となる音型"],
        "change": ["終わり方"],
        "description": "中心素材から別の素材を作る。",
    }
    variations["second_to_theme"] = {
        "source": {"type": "material", "id": "second_theme"},
        "target": {"type": "material", "id": "theme"},
        "preserve": ["終わり方"],
        "change": ["中心となる音型"],
        "description": "別の素材から中心素材へ戻す。",
    }

    result = check(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID
        and issue.path == "/script/variations/second_to_theme/target/id"
        for issue in result.issues
    )


def test_check_rejects_a_section_variation_that_points_backwards() -> None:
    document = _minimal_document()
    script = document["script"]
    assert isinstance(script, dict)
    sections = script["sections"]
    assert isinstance(sections, dict)
    sections["return"] = {
        "parent_section_id": "whole",
        "order": 1,
        "role": "return",
        "relative_length": 1,
        "description": "素材を再提示する。",
    }
    variations = script["variations"]
    assert isinstance(variations, dict)
    variations["backwards_return"] = {
        "source": {"type": "section", "id": "return"},
        "target": {"type": "section", "id": "statement"},
        "preserve": ["中心となる音型"],
        "change": ["終わり方"],
        "description": "後の区間を元に前の区間を作る。",
    }

    result = check(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID
        and issue.path == "/script/variations/backwards_return/target/id"
        for issue in result.issues
    )


def test_check_rejects_a_placement_variation_that_points_backwards() -> None:
    document = _minimal_document()
    script = document["script"]
    assert isinstance(script, dict)
    sections = script["sections"]
    assert isinstance(sections, dict)
    sections["return"] = {
        "parent_section_id": "whole",
        "order": 1,
        "role": "return",
        "relative_length": 1,
        "description": "素材を再提示する。",
    }
    placements = script["placements"]
    assert isinstance(placements, dict)
    placements["theme_return"] = {"section_id": "return", "material_id": "theme"}
    variations = script["variations"]
    assert isinstance(variations, dict)
    variations["backwards_return"] = {
        "source": {"type": "placement", "id": "theme_return"},
        "target": {"type": "placement", "id": "theme_first"},
        "preserve": ["中心となる音型"],
        "change": ["終わり方"],
        "description": "後の配置を元に前の配置を作る。",
    }

    result = check(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID
        and issue.path == "/script/variations/backwards_return/target/id"
        for issue in result.issues
    )


def test_check_rejects_a_transition_with_a_missing_connector_placement() -> None:
    document = _minimal_document()
    script = document["script"]
    assert isinstance(script, dict)
    transitions = script["transitions"]
    assert isinstance(transitions, dict)
    transitions["to_return"] = {
        "from_placement_id": "theme_first",
        "connector_placement_id": "missing",
        "to_placement_id": "theme_first",
        "description": "主題へ戻る。",
    }

    result = check(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID
        and issue.path == "/script/transitions/to_return/connector_placement_id"
        for issue in result.issues
    )


def test_check_rejects_a_transition_that_reuses_a_placement() -> None:
    document = _minimal_document()
    script = document["script"]
    assert isinstance(script, dict)
    sections = script["sections"]
    assert isinstance(sections, dict)
    sections["return"] = {
        "parent_section_id": "whole",
        "order": 1,
        "role": "return",
        "relative_length": 1,
        "description": "素材を再提示する。",
    }
    placements = script["placements"]
    assert isinstance(placements, dict)
    placements["theme_return"] = {"section_id": "return", "material_id": "theme"}
    transitions = script["transitions"]
    assert isinstance(transitions, dict)
    transitions["to_return"] = {
        "from_placement_id": "theme_first",
        "connector_placement_id": "theme_first",
        "to_placement_id": "theme_return",
        "description": "主題の再提示へつなぐ。",
    }

    result = check(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID
        and issue.path == "/script/transitions/to_return/connector_placement_id"
        for issue in result.issues
    )


def test_check_rejects_a_transition_connector_that_shares_its_section() -> None:
    document = _minimal_document()
    script = document["script"]
    assert isinstance(script, dict)
    sections = script["sections"]
    assert isinstance(sections, dict)
    sections["bridge"] = {
        "parent_section_id": "whole",
        "order": 1,
        "role": "transition",
        "relative_length": 0.25,
        "description": "再提示へ導く。",
    }
    sections["return"] = {
        "parent_section_id": "whole",
        "order": 2,
        "role": "return",
        "relative_length": 1,
        "description": "素材を再提示する。",
    }
    materials = script["materials"]
    assert isinstance(materials, dict)
    materials["bridge"] = {"kind": "transition", "description": "再提示への接続素材。"}
    placements = script["placements"]
    assert isinstance(placements, dict)
    placements["bridge_fill"] = {"section_id": "bridge", "material_id": "bridge"}
    placements["bridge_theme"] = {"section_id": "bridge", "material_id": "theme"}
    placements["theme_return"] = {"section_id": "return", "material_id": "theme"}
    transitions = script["transitions"]
    assert isinstance(transitions, dict)
    transitions["to_return"] = {
        "from_placement_id": "theme_first",
        "connector_placement_id": "bridge_fill",
        "to_placement_id": "theme_return",
        "description": "主題の再提示へつなぐ。",
    }

    result = check(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID
        and issue.path == "/script/transitions/to_return/connector_placement_id"
        for issue in result.issues
    )


def test_check_rejects_a_transition_with_the_connector_after_its_target() -> None:
    document = _minimal_document()
    script = document["script"]
    assert isinstance(script, dict)
    sections = script["sections"]
    assert isinstance(sections, dict)
    sections["return"] = {
        "parent_section_id": "whole",
        "order": 1,
        "role": "return",
        "relative_length": 1,
        "description": "素材を再提示する。",
    }
    sections["bridge"] = {
        "parent_section_id": "whole",
        "order": 2,
        "role": "transition",
        "relative_length": 0.25,
        "description": "再提示へ導く。",
    }
    materials = script["materials"]
    assert isinstance(materials, dict)
    materials["bridge"] = {"kind": "transition", "description": "再提示への接続素材。"}
    placements = script["placements"]
    assert isinstance(placements, dict)
    placements["theme_return"] = {"section_id": "return", "material_id": "theme"}
    placements["bridge_fill"] = {"section_id": "bridge", "material_id": "bridge"}
    transitions = script["transitions"]
    assert isinstance(transitions, dict)
    transitions["to_return"] = {
        "from_placement_id": "theme_first",
        "connector_placement_id": "bridge_fill",
        "to_placement_id": "theme_return",
        "description": "主題の再提示へつなぐ。",
    }

    result = check(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID
        and issue.path == "/script/transitions/to_return/connector_placement_id"
        for issue in result.issues
    )


def test_check_rejects_a_performance_direction_with_a_missing_target() -> None:
    document = _minimal_document()
    script = document["script"]
    assert isinstance(script, dict)
    performance_setup = script["performance_setup"]
    assert isinstance(performance_setup, dict)
    directions = performance_setup["performance_directions"]
    assert isinstance(directions, dict)
    directions["quietly"] = {
        "target": {"type": "section", "id": "missing"},
        "description": "静かに演奏する。",
    }

    result = check(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID
        and issue.path == "/script/performance_setup/performance_directions/quietly/target/id"
        for issue in result.issues
    )


def test_check_rejects_a_draft_with_approval_metadata() -> None:
    document = _minimal_document()
    document["approval"] = {"content_sha256": "0" * 64}

    result = check(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID and issue.path == "/approval"
        for issue in result.issues
    )


def test_check_rejects_an_approved_document_without_approval_metadata() -> None:
    document = _minimal_document()
    document["status"] = "approved"

    result = check(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID and issue.path == "/approval"
        for issue in result.issues
    )


def test_check_rejects_an_approved_document_with_the_wrong_content_hash() -> None:
    document = _minimal_document()
    document["status"] = "approved"
    document["approval"] = {"content_sha256": "0" * 64}

    result = check(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID and issue.path == "/approval/content_sha256"
        for issue in result.issues
    )


def test_check_accepts_an_approved_document_with_the_matching_content_hash() -> None:
    document = _minimal_document()
    document["status"] = "approved"
    hashed_content = {
        field: document[field] for field in ("schema_version", "document_id", "revision", "script")
    }
    document["approval"] = {
        "content_sha256": hashlib.sha256(rfc8785.dumps(hashed_content)).hexdigest()
    }

    result = check(document)

    assert result.valid is True
    assert result.issues == ()
