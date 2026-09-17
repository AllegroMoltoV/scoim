"""Version-separated score intermediate representation for solo_piano_3m_v2."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast


class ScoreIrValidationError(ValueError):
    """A score intermediate representation violates its structural contract."""


@dataclass(frozen=True, slots=True)
class PlanNode:
    section_id: str
    parent_section_id: str | None
    order: int
    duration_weight: float | None = None
    score_unit_id: str | None = None


@dataclass(frozen=True, slots=True)
class PiecePlan:
    plan_id: str
    title: str
    tonal_center: int
    mode: str
    root_section_id: str
    nodes: tuple[PlanNode, ...]


@dataclass(frozen=True, slots=True)
class ScoreNote:
    score_note_id: str
    at_units: int
    duration_units: int
    pitch: int
    voice: str
    tie: str | None = None
    articulations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ScoreDirection:
    direction_id: str
    at_units: int
    kind: str
    value: str


@dataclass(frozen=True, slots=True)
class ScoreHarmony:
    harmony_id: str
    at_units: int
    duration_units: int
    root_pitch_class: int
    quality: str


@dataclass(frozen=True, slots=True)
class ScoreUnitLayer:
    score_unit_layer_id: str
    source_material_placement_id: str
    notes: tuple[ScoreNote, ...]


@dataclass(frozen=True, slots=True)
class ScoreUnit:
    score_unit_id: str
    source_section_id: str
    length_units: int
    harmonies: tuple[ScoreHarmony, ...]
    directions: tuple[ScoreDirection, ...]
    score_unit_layers: tuple[ScoreUnitLayer, ...]


@dataclass(frozen=True, slots=True)
class ScoreSpec:
    score_id: str
    divisions: int
    score_units: tuple[ScoreUnit, ...]


def piece_plan_from_json(value: Mapping[str, object]) -> PiecePlan:
    """Restore a piece plan from its stable JSON representation."""
    return PiecePlan(
        plan_id=cast(str, value["plan_id"]),
        title=cast(str, value["title"]),
        tonal_center=cast(int, value["tonal_center"]),
        mode=cast(str, value["mode"]),
        root_section_id=cast(str, value["root_section_id"]),
        nodes=tuple(
            PlanNode(**cast(dict[str, object], node)) for node in cast(list[object], value["nodes"])
        ),
    )


def score_spec_from_json(value: Mapping[str, object]) -> ScoreSpec:
    """Restore a score specification from its stable JSON representation."""
    return ScoreSpec(
        score_id=cast(str, value["score_id"]),
        divisions=cast(int, value["divisions"]),
        score_units=tuple(
            ScoreUnit(
                score_unit_id=cast(str, unit["score_unit_id"]),
                source_section_id=cast(str, unit["source_section_id"]),
                length_units=cast(int, unit["length_units"]),
                harmonies=tuple(
                    ScoreHarmony(**cast(dict[str, object], harmony))
                    for harmony in cast(list[Mapping[str, object]], unit["harmonies"])
                ),
                directions=tuple(
                    ScoreDirection(**cast(dict[str, object], direction))
                    for direction in cast(list[Mapping[str, object]], unit["directions"])
                ),
                score_unit_layers=tuple(
                    ScoreUnitLayer(
                        score_unit_layer_id=cast(str, layer["score_unit_layer_id"]),
                        source_material_placement_id=cast(
                            str, layer["source_material_placement_id"]
                        ),
                        notes=tuple(
                            ScoreNote(
                                score_note_id=cast(str, note["score_note_id"]),
                                at_units=cast(int, note["at_units"]),
                                duration_units=cast(int, note["duration_units"]),
                                pitch=cast(int, note["pitch"]),
                                voice=cast(str, note["voice"]),
                                tie=cast(str | None, note.get("tie")),
                                articulations=tuple(cast(list[str], note.get("articulations", []))),
                            )
                            for note in cast(list[Mapping[str, object]], layer["notes"])
                        ),
                    )
                    for layer in cast(list[Mapping[str, object]], unit["score_unit_layers"])
                ),
            )
            for unit in cast(list[Mapping[str, object]], value["score_units"])
        ),
    )


def validate_score_ir(plan: PiecePlan, score: ScoreSpec) -> None:
    """Validate the cross-object invariants of one v2 score boundary."""
    if score.divisions <= 0:
        raise ScoreIrValidationError("a score must use positive divisions")
    expected_units = validate_piece_plan(plan)
    _validate_score_spec(score, expected_units)


def validate_piece_plan(plan: PiecePlan) -> dict[str, str]:
    """Validate a piece plan and return its score-unit to section mapping."""
    section_ids = [node.section_id for node in plan.nodes]
    if len(section_ids) != len(set(section_ids)):
        raise ScoreIrValidationError("section IDs must be unique")
    if plan.root_section_id not in section_ids:
        raise ScoreIrValidationError("the designated root section must exist")
    for node in plan.nodes:
        if (node.section_id == plan.root_section_id) != (node.parent_section_id is None):
            raise ScoreIrValidationError("the plan must have exactly one parentless root")
        if node.parent_section_id is not None and node.parent_section_id not in section_ids:
            raise ScoreIrValidationError("a plan node references a missing parent section")
    sibling_orders: dict[str, list[int]] = {}
    for node in plan.nodes:
        if node.parent_section_id is not None:
            sibling_orders.setdefault(node.parent_section_id, []).append(node.order)
    for orders in sibling_orders.values():
        if sorted(orders) != list(range(len(orders))):
            raise ScoreIrValidationError("sibling order must be consecutive from zero")
    children_by_parent: dict[str, list[str]] = {}
    for node in plan.nodes:
        if node.parent_section_id is not None:
            children_by_parent.setdefault(node.parent_section_id, []).append(node.section_id)
    reachable: set[str] = set()
    pending = [plan.root_section_id]
    while pending:
        section_id = pending.pop()
        if section_id in reachable:
            continue
        reachable.add(section_id)
        pending.extend(children_by_parent.get(section_id, ()))
    if reachable != set(section_ids):
        raise ScoreIrValidationError("all plan nodes must be reachable from the root section")
    parent_section_ids = {
        node.parent_section_id for node in plan.nodes if node.parent_section_id is not None
    }
    for node in plan.nodes:
        if node.section_id in parent_section_ids:
            if node.duration_weight is not None or node.score_unit_id is not None:
                raise ScoreIrValidationError("a branch node cannot own a score unit or duration")
        elif node.score_unit_id is None or node.duration_weight is None:
            raise ScoreIrValidationError("a leaf node must own one score unit and duration")
        elif node.duration_weight <= 0:
            raise ScoreIrValidationError("a leaf node must have a positive duration weight")
    expected_units = {
        node.score_unit_id: node.section_id for node in plan.nodes if node.score_unit_id is not None
    }
    leaf_count = sum(node.score_unit_id is not None for node in plan.nodes)
    if len(expected_units) != leaf_count:
        raise ScoreIrValidationError("one score unit cannot be shared by two leaf sections")
    return expected_units


def _validate_score_spec(score: ScoreSpec, expected_units: dict[str, str]) -> None:
    actual_units = {unit.score_unit_id: unit.source_section_id for unit in score.score_units}
    if (
        len(expected_units) != len(score.score_units)
        or len(actual_units) != len(score.score_units)
        or actual_units != expected_units
    ):
        raise ScoreIrValidationError("score units must match leaf plan nodes one to one")
    material_placement_ids: set[str] = set()
    score_unit_layer_ids: set[str] = set()
    score_note_ids: set[str] = set()
    for unit in score.score_units:
        if unit.length_units <= 0:
            raise ScoreIrValidationError("a score unit must have a positive length")
        harmony_cursor = 0
        for harmony in sorted(unit.harmonies, key=lambda item: item.at_units):
            if harmony.at_units != harmony_cursor or harmony.duration_units <= 0:
                raise ScoreIrValidationError(
                    "harmony must cover a score unit without gaps or overlaps"
                )
            harmony_cursor += harmony.duration_units
        if harmony_cursor != unit.length_units:
            raise ScoreIrValidationError("harmony must cover the whole score unit")
        if not unit.score_unit_layers:
            raise ScoreIrValidationError("a score unit must contain at least one layer")
        for layer in unit.score_unit_layers:
            if not layer.notes:
                raise ScoreIrValidationError("a score unit layer must contain at least one note")
            if layer.score_unit_layer_id in score_unit_layer_ids:
                raise ScoreIrValidationError("score unit layer IDs must be globally unique")
            score_unit_layer_ids.add(layer.score_unit_layer_id)
            if layer.source_material_placement_id in material_placement_ids:
                raise ScoreIrValidationError(
                    "a material placement must correspond to exactly one score unit layer"
                )
            material_placement_ids.add(layer.source_material_placement_id)
            for note in layer.notes:
                if (
                    note.at_units < 0
                    or note.duration_units <= 0
                    or note.at_units + note.duration_units > unit.length_units
                ):
                    raise ScoreIrValidationError(
                        "a score note must remain within score unit bounds"
                    )
                if note.voice not in {"upper", "lower"}:
                    raise ScoreIrValidationError("a score note voice must be upper or lower")
                if note.score_note_id in score_note_ids:
                    raise ScoreIrValidationError("score note IDs must be globally unique")
                score_note_ids.add(note.score_note_id)
