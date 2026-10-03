"""Shared score-ending boundary derived from accepted terminal-layer notes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from .score_ir import PiecePlan, ScoreHarmony, ScoreNote, ScoreSpec

SHARED_TERMINAL = "shared-terminal-v1"


class TerminalBoundaryError(ValueError):
    """The accepted score values cannot establish one shared ending boundary."""


@dataclass(frozen=True, slots=True)
class TerminalBoundary:
    """Last score attack and sounding end shared by score and performance layers."""

    score_unit_id: str
    terminal_attack_units: int
    terminal_end_units: int


def terminal_boundary_from_json(value: Mapping[str, object]) -> TerminalBoundary:
    """Restore a terminal boundary from its stable JSON representation."""
    return TerminalBoundary(
        score_unit_id=cast(str, value["score_unit_id"]),
        terminal_attack_units=cast(int, value["terminal_attack_units"]),
        terminal_end_units=cast(int, value["terminal_end_units"]),
    )


def final_score_unit_id(plan: PiecePlan) -> str:
    """Return the last leaf score unit in structural performance order."""
    nodes = {node.section_id: node for node in plan.nodes}
    children: dict[str, list[str]] = {}
    for node in plan.nodes:
        if node.parent_section_id is not None:
            children.setdefault(node.parent_section_id, []).append(node.section_id)
    leaves: list[str] = []

    def visit(section_id: str) -> None:
        descendants = sorted(
            children.get(section_id, ()),
            key=lambda child_id: nodes[child_id].order,
        )
        if descendants:
            for child_id in descendants:
                visit(child_id)
            return
        score_unit_id = nodes[section_id].score_unit_id
        if score_unit_id is None:
            raise TerminalBoundaryError("the final structural leaf has no score unit")
        leaves.append(score_unit_id)

    if plan.root_section_id not in nodes:
        raise TerminalBoundaryError("the plan root section is missing")
    visit(plan.root_section_id)
    if not leaves:
        raise TerminalBoundaryError("the plan has no score unit")
    return leaves[-1]


def terminal_boundary_role(document: Mapping[str, object], plan: PiecePlan) -> str:
    """Return the role that owns the ending without inventing a score layer."""
    if final_role_placement_ids(document, plan, "foreground"):
        return "foreground"
    if final_role_placement_ids(document, plan, "accompaniment"):
        return "accompaniment"
    raise TerminalBoundaryError("the final score unit has no foreground or accompaniment")


def derive_terminal_boundary(
    document: Mapping[str, object],
    plan: PiecePlan,
    harmonies_by_score_unit: Mapping[str, tuple[ScoreHarmony, ...]],
    notes_by_material_placement: Mapping[str, tuple[ScoreNote, ...]],
    *,
    role: str,
) -> TerminalBoundary:
    """Derive the shared ending from one accepted role in the final unit."""
    if role not in {"foreground", "accompaniment"}:
        raise TerminalBoundaryError(f"unsupported terminal role: {role}")
    final_unit_id = final_score_unit_id(plan)
    placement_ids = final_role_placement_ids(document, plan, role)
    if any(key not in notes_by_material_placement for key in placement_ids):
        raise TerminalBoundaryError("terminal role placements are incomplete")
    notes = tuple(
        note
        for placement_id in placement_ids
        for note in notes_by_material_placement.get(placement_id, ())
    )
    if not notes:
        raise TerminalBoundaryError(f"the final score unit has no accepted {role} note")
    terminal_attack = max(note.at_units for note in notes)
    sounding = tuple(
        note
        for note in notes
        if note.at_units <= terminal_attack < note.at_units + note.duration_units
    )
    if not sounding:
        raise TerminalBoundaryError(f"the last {role} attack has no sounding note")
    terminal_end = max(note.at_units + note.duration_units for note in sounding)
    final_harmonies = harmonies_by_score_unit.get(final_unit_id, ())
    containing_harmony = next(
        (
            harmony
            for harmony in final_harmonies
            if harmony.at_units <= terminal_attack < harmony.at_units + harmony.duration_units
        ),
        None,
    )
    if containing_harmony is None:
        raise TerminalBoundaryError(f"the last {role} attack is outside shared harmony")
    if containing_harmony is not final_harmonies[-1]:
        raise TerminalBoundaryError(f"the last {role} attack is outside the final harmony")
    if terminal_end > containing_harmony.at_units + containing_harmony.duration_units:
        raise TerminalBoundaryError(f"the terminal {role} extends beyond the final harmony")
    return TerminalBoundary(final_unit_id, terminal_attack, terminal_end)


def derive_score_terminal_boundary(
    document: Mapping[str, object],
    plan: PiecePlan,
    score: ScoreSpec,
) -> TerminalBoundary:
    """Reconstruct the terminal boundary from a completed score."""
    script = cast(Mapping[str, object], document["script"])
    placements = cast(Mapping[str, Mapping[str, object]], script["material_placements"])
    role = terminal_boundary_role(document, plan)
    role_notes: dict[str, tuple[ScoreNote, ...]] = {}
    harmonies: dict[str, tuple[ScoreHarmony, ...]] = {}
    for unit in score.score_units:
        harmonies[unit.score_unit_id] = unit.harmonies
        for layer in unit.score_unit_layers:
            placement = placements[layer.source_material_placement_id]
            if placement["role"] == role:
                role_notes[layer.source_material_placement_id] = layer.notes
    return derive_terminal_boundary(document, plan, harmonies, role_notes, role=role)


def final_role_placement_ids(
    document: Mapping[str, object], plan: PiecePlan, role: str
) -> tuple[str, ...]:
    final_unit_id = final_score_unit_id(plan)
    final_section_id = next(
        (node.section_id for node in plan.nodes if node.score_unit_id == final_unit_id),
        None,
    )
    if final_section_id is None:
        raise TerminalBoundaryError("the final score unit has no source section")
    script = cast(Mapping[str, object], document["script"])
    placements = cast(Mapping[str, Mapping[str, object]], script["material_placements"])
    return tuple(
        sorted(
            placement_id
            for placement_id, placement in placements.items()
            if placement["role"] == role and placement["section_id"] == final_section_id
        )
    )


def validate_score_terminal_boundary(
    document: Mapping[str, object],
    plan: PiecePlan,
    score: ScoreSpec,
    boundary: TerminalBoundary,
) -> None:
    """Check the assembled score without changing any accepted note."""
    reconstructed = derive_score_terminal_boundary(document, plan, score)
    if reconstructed != boundary:
        raise TerminalBoundaryError(
            "the score terminal boundary differs from the accepted score boundary"
        )
    final_unit = next(
        (unit for unit in score.score_units if unit.score_unit_id == boundary.score_unit_id),
        None,
    )
    if final_unit is None:
        raise TerminalBoundaryError("the score is missing the terminal score unit")
    validate_terminal_notes(
        document,
        plan,
        {layer.source_material_placement_id: layer.notes for layer in final_unit.score_unit_layers},
        boundary,
    )


def validate_terminal_notes(
    document: Mapping[str, object],
    plan: PiecePlan,
    notes: Mapping[str, tuple[ScoreNote, ...]],
    boundary: TerminalBoundary,
) -> None:
    """Check available terminal placements, requiring support when its owner is present."""
    final_ids = (
        *final_role_placement_ids(document, plan, "foreground"),
        *final_role_placement_ids(document, plan, "accompaniment"),
    )
    for key in final_ids:
        for note in notes.get(key, ()):
            if note.at_units > boundary.terminal_attack_units:
                raise TerminalBoundaryError("a score note attacks after the shared terminal attack")
            if note.at_units + note.duration_units > boundary.terminal_end_units:
                raise TerminalBoundaryError("a score note outlives the shared terminal end")
    support_ids = final_role_placement_ids(document, plan, "accompaniment")
    if (
        support_ids
        and support_ids[-1] in notes
        and not any(
            note.at_units <= boundary.terminal_attack_units < note.at_units + note.duration_units
            and note.at_units + note.duration_units == boundary.terminal_end_units
            for note in notes[support_ids[-1]]
        )
    ):
        raise TerminalBoundaryError(
            "the final accompaniment owner does not support the shared terminal"
        )


def available_terminal_boundary(
    document: Mapping[str, object],
    plan: PiecePlan,
    harmonies: Mapping[str, tuple[ScoreHarmony, ...]],
    notes: Mapping[str, tuple[ScoreNote, ...]],
) -> TerminalBoundary | None:
    """Derive only after every placement of the terminal role has been evaluated."""
    role = terminal_boundary_role(document, plan)
    required = final_role_placement_ids(document, plan, role)
    if not set(required) <= notes.keys():
        return None
    boundary = derive_terminal_boundary(document, plan, harmonies, notes, role=role)
    validate_terminal_notes(document, plan, notes, boundary)
    return boundary
