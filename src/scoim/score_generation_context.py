"""Bounded section descriptions and the contract for new score generation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

SECTION_RANGE_CONTEXT_V1 = "section-range-context-v1"
GENERATION_CONTEXT_CONTRACT = SECTION_RANGE_CONTEXT_V1


def require_current_generation_context(value: object) -> None:
    """Reject old generation inputs without restricting completed bundle replay."""
    if value != GENERATION_CONTEXT_CONTRACT:
        raise ValueError(
            "new generation requires the current generation context contract; "
            "start a new trial from the saved composition"
        )


def _section_context(section_id: str, section: Mapping[str, object]) -> dict[str, object]:
    return {
        "section_id": section_id,
        **{
            field: section[field] for field in ("parent_section_id", "order", "role", "description")
        },
    }


def sections_in_hierarchy_order(document: Mapping[str, object]) -> list[dict[str, object]]:
    """Describe all validated sections in root-first performance order."""
    script = cast(Mapping[str, object], document["script"])
    sections = cast(Mapping[str, Mapping[str, object]], script["sections"])
    children: dict[str, list[str]] = {}
    for section_id, section in sections.items():
        parent_id = section["parent_section_id"]
        if isinstance(parent_id, str):
            children.setdefault(parent_id, []).append(section_id)
    for child_ids in children.values():
        child_ids.sort(key=lambda key: cast(int, sections[key]["order"]))
    result: list[dict[str, object]] = []
    pending = [cast(str, script["root_section_id"])]
    while pending:
        section_id = pending.pop()
        result.append(_section_context(section_id, sections[section_id]))
        pending.extend(reversed(children.get(section_id, ())))
    return result


def ancestor_sections(document: Mapping[str, object], section_id: str) -> list[dict[str, object]]:
    """Describe only the target's ancestors, from root to immediate parent."""
    script = cast(Mapping[str, object], document["script"])
    sections = cast(Mapping[str, Mapping[str, object]], script["sections"])
    result: list[dict[str, object]] = []
    parent_id = sections[section_id]["parent_section_id"]
    while isinstance(parent_id, str):
        section = sections[parent_id]
        result.append(_section_context(parent_id, section))
        parent_id = section["parent_section_id"]
    result.reverse()
    return result
