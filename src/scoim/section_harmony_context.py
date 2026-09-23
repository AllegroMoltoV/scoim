"""Deliver complete source harmony ranges to section variation targets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import cast

from .score_ir import ScoreHarmony
from .score_work_plan import ordered_placement_ids, section_leaf_ids


def section_harmony_contexts(
    document: Mapping[str, object],
    target_leaf_id: str,
    *,
    length_units_by_score_unit: Mapping[str, int],
    accepted_harmonies_by_score_unit: Mapping[str, tuple[ScoreHarmony, ...]],
) -> list[dict[str, object]]:
    """Keep every applicable ancestor relation and require all source harmonies."""
    script = cast(Mapping[str, object], document["script"])
    relations = cast(
        Mapping[str, Mapping[str, object]], script["script_element_variation_relations"]
    )
    placements = cast(Mapping[str, Mapping[str, object]], script["material_placements"])
    order = ordered_placement_ids(document)

    def section_range(section_id: str, *, include_harmonies: bool) -> dict[str, object]:
        offset = 0
        leaves: list[dict[str, object]] = []
        for leaf_id in section_leaf_ids(document, section_id):
            score_unit_id = f"score-unit-{leaf_id}"
            length = length_units_by_score_unit[score_unit_id]
            leaf: dict[str, object] = {
                "section_id": leaf_id,
                "score_unit_id": score_unit_id,
                "offset_units": offset,
                "length_units": length,
                "material_placements": [
                    {"material_placement_id": key, **placements[key]}
                    for key in order
                    if placements[key]["section_id"] == leaf_id
                ],
            }
            if include_harmonies:
                if score_unit_id not in accepted_harmonies_by_score_unit:
                    raise ValueError(
                        f"section variation source harmony is not accepted: {score_unit_id}"
                    )
                leaf["harmonies"] = [
                    asdict(value) for value in accepted_harmonies_by_score_unit[score_unit_id]
                ]
            leaves.append(leaf)
            offset += length
        return {"section_id": section_id, "length_units": offset, "leaves": leaves}

    contexts: list[dict[str, object]] = []
    for relation_id, relation in sorted(relations.items()):
        source = cast(Mapping[str, str], relation["source"])
        target = cast(Mapping[str, str], relation["target"])
        if source["type"] != "section" or target_leaf_id not in section_leaf_ids(
            document, target["id"]
        ):
            continue
        contexts.append(
            {
                "relation_id": relation_id,
                "relation": dict(relation),
                "source": section_range(source["id"], include_harmonies=True),
                "target": section_range(target["id"], include_harmonies=False),
                "target_leaf_id": target_leaf_id,
            }
        )
    return contexts
