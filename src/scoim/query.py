"""Read-only local queries for SCoIM music-script documents."""

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import jsonpointer

from .validation import IssueCode, ValidationIssue, check

COLLECTION_BY_ITEM_TYPE = {
    "section": "sections",
    "material": "materials",
    "placement": "placements",
    "variation": "variations",
    "transition": "transitions",
    "performance_direction": "performance_directions",
    "requirement": "requirements",
}


@dataclass(frozen=True, slots=True)
class ItemReference:
    """Stable reference to one script item."""

    type: str
    id: str


@dataclass(frozen=True, slots=True)
class ShownItem:
    """One queried script item and its value."""

    type: str
    id: str
    value: dict[str, object]


@dataclass(frozen=True, slots=True)
class ShowRelation:
    """A direct relation viewed from the queried item."""

    role: str
    item: ItemReference


@dataclass(frozen=True, slots=True)
class ShowResult:
    """Result of querying one item and its direct relations."""

    shown: bool
    item: ShownItem | None
    relations: tuple[ShowRelation, ...]
    issues: tuple[ValidationIssue, ...]


def _reference_matches(
    raw_owner: object,
    field: str,
    *,
    item_type: str,
    item_id: str,
) -> bool:
    owner = cast(dict[str, object], raw_owner)
    reference = cast(dict[str, object], owner[field])
    return reference["type"] == item_type and reference["id"] == item_id


def _item_reference_from(owner: dict[str, object], field: str) -> ItemReference:
    reference = cast(dict[str, object], owner[field])
    return ItemReference(cast(str, reference["type"]), cast(str, reference["id"]))


def _variation_relations(
    script: dict[str, object], *, item_type: str, item_id: str
) -> tuple[ShowRelation, ...]:
    relations: list[ShowRelation] = []
    variations = cast(dict[str, object], script["variations"])
    for reference_field, relation_role in (
        ("source", "variation_as_source"),
        ("target", "variation_as_target"),
    ):
        relations.extend(
            ShowRelation(relation_role, ItemReference("variation", variation_id))
            for variation_id, raw_variation in sorted(variations.items())
            if _reference_matches(
                raw_variation,
                reference_field,
                item_type=item_type,
                item_id=item_id,
            )
        )
    return tuple(relations)


def _performance_direction_relations(
    script: dict[str, object], *, item_type: str, item_id: str
) -> tuple[ShowRelation, ...]:
    performance_setup = cast(dict[str, object], script["performance_setup"])
    directions = cast(dict[str, object], performance_setup["performance_directions"])
    return tuple(
        ShowRelation(
            "performance_direction",
            ItemReference("performance_direction", direction_id),
        )
        for direction_id, raw_direction in sorted(directions.items())
        if _reference_matches(
            raw_direction,
            "target",
            item_type=item_type,
            item_id=item_id,
        )
    )


def _section_requirement_relations(
    script: dict[str, object], item_id: str
) -> tuple[ShowRelation, ...]:
    setup = cast(dict[str, object], script["performance_setup"])
    directions = cast(dict[str, object], setup["performance_directions"])
    requirements = cast(dict[str, object], script.get("requirements", {}))
    relations: list[ShowRelation] = []
    for role, field in (
        ("requirement_as_target", "target"),
        ("requirement_as_reference", "relative_to"),
    ):
        for requirement_id, raw_requirement in sorted(requirements.items()):
            requirement = cast(dict[str, object], raw_requirement)
            direction = cast(
                dict[str, object], directions[cast(str, requirement["performance_direction_id"])]
            )
            reference = cast(dict[str, object], direction[field])
            if reference["id"] == item_id:
                relations.append(ShowRelation(role, ItemReference("requirement", requirement_id)))
    return tuple(relations)


def _section_relations(
    script: dict[str, object], item_id: str, section: dict[str, object]
) -> tuple[ShowRelation, ...]:
    sections = cast(dict[str, object], script["sections"])
    relations: list[ShowRelation] = []
    parent_id = section["parent_section_id"]
    if isinstance(parent_id, str):
        relations.append(ShowRelation("parent", ItemReference("section", parent_id)))
    child_ids = sorted(
        (
            child_id
            for child_id, raw_child in sections.items()
            if cast(dict[str, object], raw_child)["parent_section_id"] == item_id
        ),
        key=lambda child_id: cast(dict[str, object], sections[child_id])["order"],
    )
    relations.extend(
        ShowRelation("child", ItemReference("section", child_id)) for child_id in child_ids
    )
    placements = cast(dict[str, object], script["placements"])
    relations.extend(
        ShowRelation("placement", ItemReference("placement", placement_id))
        for placement_id, raw_placement in sorted(placements.items())
        if cast(dict[str, object], raw_placement)["section_id"] == item_id
    )
    relations.extend(_variation_relations(script, item_type="section", item_id=item_id))
    relations.extend(_performance_direction_relations(script, item_type="section", item_id=item_id))
    relations.extend(_section_requirement_relations(script, item_id))
    return tuple(relations)


def _material_relations(script: dict[str, object], item_id: str) -> tuple[ShowRelation, ...]:
    placements = cast(dict[str, object], script["placements"])
    placement_relations = tuple(
        ShowRelation("placement", ItemReference("placement", placement_id))
        for placement_id, raw_placement in sorted(placements.items())
        if cast(dict[str, object], raw_placement)["material_id"] == item_id
    )
    return placement_relations + _variation_relations(script, item_type="material", item_id=item_id)


def _placement_relations(
    script: dict[str, object], item_id: str, placement: dict[str, object]
) -> tuple[ShowRelation, ...]:
    relations = [
        ShowRelation("section", ItemReference("section", cast(str, placement["section_id"]))),
        ShowRelation("material", ItemReference("material", cast(str, placement["material_id"]))),
    ]
    relations.extend(_variation_relations(script, item_type="placement", item_id=item_id))
    transitions = cast(dict[str, object], script["transitions"])
    for field, relation_role in (
        ("from_placement_id", "transition_from"),
        ("connector_placement_id", "transition_connector"),
        ("to_placement_id", "transition_to"),
    ):
        relations.extend(
            ShowRelation(relation_role, ItemReference("transition", transition_id))
            for transition_id, raw_transition in sorted(transitions.items())
            if cast(dict[str, object], raw_transition)[field] == item_id
        )
    relations.extend(
        _performance_direction_relations(script, item_type="placement", item_id=item_id)
    )
    return tuple(relations)


def _variation_item_relations(variation: dict[str, object]) -> tuple[ShowRelation, ...]:
    return tuple(
        ShowRelation(field, _item_reference_from(variation, field))
        for field in ("source", "target")
    )


def _transition_relations(transition: dict[str, object]) -> tuple[ShowRelation, ...]:
    return tuple(
        ShowRelation(
            relation_role,
            ItemReference("placement", cast(str, transition[field])),
        )
        for field, relation_role in (
            ("from_placement_id", "from"),
            ("connector_placement_id", "connector"),
            ("to_placement_id", "to"),
        )
    )


def _performance_direction_item_relations(
    script: dict[str, object], direction_id: str, direction: dict[str, object]
) -> tuple[ShowRelation, ...]:
    fields = ("target", "relative_to") if "relative_to" in direction else ("target",)
    relations = [ShowRelation(field, _item_reference_from(direction, field)) for field in fields]
    requirements = cast(dict[str, object], script.get("requirements", {}))
    relations.extend(
        ShowRelation("requirement", ItemReference("requirement", requirement_id))
        for requirement_id, raw_requirement in sorted(requirements.items())
        if cast(dict[str, object], raw_requirement)["performance_direction_id"] == direction_id
    )
    return tuple(relations)


def _requirement_item_relations(
    script: dict[str, object], requirement: dict[str, object]
) -> tuple[ShowRelation, ...]:
    direction_id = cast(str, requirement["performance_direction_id"])
    setup = cast(dict[str, object], script["performance_setup"])
    directions = cast(dict[str, object], setup["performance_directions"])
    direction = cast(dict[str, object], directions[direction_id])
    return (
        ShowRelation("performance_direction", ItemReference("performance_direction", direction_id)),
        ShowRelation("target", _item_reference_from(direction, "target")),
        ShowRelation("reference", _item_reference_from(direction, "relative_to")),
    )


def show(document: Mapping[str, object], item_type: str, item_id: str) -> ShowResult:
    """Return one script item and its direct relations without mutating the document."""
    validation = check(document)
    if not validation.valid:
        return ShowResult(
            shown=False,
            item=None,
            relations=(),
            issues=validation.issues,
        )
    if item_type not in COLLECTION_BY_ITEM_TYPE:
        return ShowResult(
            shown=False,
            item=None,
            relations=(),
            issues=(
                ValidationIssue(
                    code=IssueCode.QUERY_INVALID,
                    message=f"Unsupported item type: {item_type!r}",
                    path="/item_type",
                ),
            ),
        )
    script = cast(dict[str, object], document["script"])
    collection_name = COLLECTION_BY_ITEM_TYPE[item_type]
    if item_type == "performance_direction":
        performance_setup = cast(dict[str, object], script["performance_setup"])
        collection = cast(dict[str, object], performance_setup[collection_name])
        collection_path = f"/script/performance_setup/{collection_name}"
    else:
        collection = cast(dict[str, object], script[collection_name])
        collection_path = f"/script/{collection_name}"
    if item_id not in collection:
        return ShowResult(
            shown=False,
            item=None,
            relations=(),
            issues=(
                ValidationIssue(
                    code=IssueCode.NOT_FOUND,
                    message=f"{item_type} does not exist: {item_id}",
                    path=f"{collection_path}/{jsonpointer.escape(item_id)}",
                ),
            ),
        )
    item = cast(dict[str, object], collection[item_id])
    if item_type == "section":
        relations = _section_relations(script, item_id, item)
    elif item_type == "material":
        relations = _material_relations(script, item_id)
    elif item_type == "placement":
        relations = _placement_relations(script, item_id, item)
    elif item_type == "variation":
        relations = _variation_item_relations(item)
    elif item_type == "transition":
        relations = _transition_relations(item)
    elif item_type == "requirement":
        relations = _requirement_item_relations(script, item)
    else:
        relations = _performance_direction_item_relations(script, item_id, item)
    return ShowResult(
        shown=True,
        item=ShownItem(item_type, item_id, copy.deepcopy(item)),
        relations=relations,
        issues=(),
    )
