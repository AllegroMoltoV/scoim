"""Resolution of typed SCoIM requirements independent of lower-stage choices."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import jsonpointer


@dataclass(frozen=True, slots=True)
class ResolvedRequirement:
    """One typed requirement with its section references resolved from the script."""

    requirement_id: str
    source_path: str
    performance_direction_id: str
    target_section_id: str
    reference_section_id: str
    feature: str
    relation: str

    def value(self) -> dict[str, str]:
        """Return the stable value embedded in prompts and projection targets."""
        return {
            "requirement_id": self.requirement_id,
            "performance_direction_id": self.performance_direction_id,
            "target_section_id": self.target_section_id,
            "reference_section_id": self.reference_section_id,
            "feature": self.feature,
            "relation": self.relation,
        }


def resolve_requirements(script: Mapping[str, object]) -> tuple[ResolvedRequirement, ...]:
    """Resolve validated requirement references without requiring a PlanChoice."""
    setup = cast(dict[str, object], script["performance_setup"])
    directions = cast(dict[str, object], setup["performance_directions"])
    requirements = cast(dict[str, object], script.get("requirements", {}))
    resolved: list[ResolvedRequirement] = []
    for requirement_id, raw_requirement in sorted(requirements.items()):
        requirement = cast(dict[str, object], raw_requirement)
        direction_id = cast(str, requirement["performance_direction_id"])
        direction = cast(dict[str, object], directions[direction_id])
        target = cast(dict[str, object], direction["target"])
        reference = cast(dict[str, object], direction["relative_to"])
        resolved.append(
            ResolvedRequirement(
                requirement_id=requirement_id,
                source_path=f"/script/requirements/{jsonpointer.escape(requirement_id)}",
                performance_direction_id=direction_id,
                target_section_id=cast(str, target["id"]),
                reference_section_id=cast(str, reference["id"]),
                feature=cast(str, requirement["feature"]),
                relation=cast(str, requirement["relation"]),
            )
        )
    return tuple(resolved)
