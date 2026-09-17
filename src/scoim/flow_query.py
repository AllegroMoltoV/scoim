"""Read-only queries for human-editable SCoIM flow documents."""

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from .flow_validation import check_flow
from .validation import IssueCode, ValidationIssue


@dataclass(frozen=True, slots=True)
class FlowShowResult:
    """Result of showing one scene and its immediate context."""

    shown: bool
    scene: dict[str, object] | None
    position: int | None
    previous_scene: dict[str, object] | None
    next_scene: dict[str, object] | None
    issues: tuple[ValidationIssue, ...]


def show_flow(document: Mapping[str, object], scene_id: str) -> FlowShowResult:
    """Return one scene and its immediate neighbors without inferring relations."""
    validation = check_flow(document)
    if not validation.valid:
        return FlowShowResult(False, None, None, None, None, validation.issues)
    scenes = cast(list[dict[str, object]], document["scenes"])
    for index, scene in enumerate(scenes):
        if scene["scene_id"] == scene_id:
            return FlowShowResult(
                True,
                copy.deepcopy(scene),
                index + 1,
                copy.deepcopy(scenes[index - 1]) if index > 0 else None,
                copy.deepcopy(scenes[index + 1]) if index + 1 < len(scenes) else None,
                (),
            )
    return FlowShowResult(
        False,
        None,
        None,
        None,
        None,
        (
            ValidationIssue(
                IssueCode.NOT_FOUND,
                f"Unknown scene ID: {scene_id}",
                "/scene_id",
            ),
        ),
    )
