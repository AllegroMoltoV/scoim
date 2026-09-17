"""既存素材へ`はげしさ`の小さな音符差分を決定的に適用する。"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Any

from llm_musical_composer.composition_ir import Material, Note
from llm_musical_composer.music_dsl import DslError, _arguments, _call, _list, _literal

MIN_NEW_ONSET_DISTANCE_MS = 100


class IntensityPatchError(ValueError):
    """局所差分の構文または適用結果が固定契約に違反した場合の例外。"""


@dataclass(frozen=True)
class NoteEdit:
    event_id: str
    at_ms: int
    duration_ms: int


@dataclass(frozen=True)
class NoteClone:
    source_event_id: str
    event_id: str
    at_ms: int
    duration_ms: int
    pitch: int


@dataclass(frozen=True)
class MaterialPatch:
    material_id: str
    remove_ids: tuple[str, ...]
    edits: tuple[NoteEdit, ...]
    additions: tuple[NoteClone, ...]
    velocity_offset: int


def _string_list(node: ast.AST) -> tuple[str, ...]:
    return _list(node, lambda item: str(_literal(item, str)))


def _int_literal(node: ast.AST) -> int:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = int(_literal(node.operand, int))
        return -value if isinstance(node.op, ast.USub) else value
    return int(_literal(node, int))


def _parse_note_edit(node: ast.AST) -> NoteEdit:
    args = _arguments(
        _call(node, "note_edit"),
        positional=("event_id",),
        required=frozenset({"at_ms", "duration_ms"}),
    )
    return NoteEdit(
        str(_literal(args["event_id"], str)),
        _int_literal(args["at_ms"]),
        _int_literal(args["duration_ms"]),
    )


def _parse_note_clone(node: ast.AST) -> NoteClone:
    args = _arguments(
        _call(node, "note_clone"),
        positional=("source_event_id", "event_id"),
        required=frozenset({"at_ms", "duration_ms", "pitch"}),
    )
    return NoteClone(
        str(_literal(args["source_event_id"], str)),
        str(_literal(args["event_id"], str)),
        _int_literal(args["at_ms"]),
        _int_literal(args["duration_ms"]),
        _int_literal(args["pitch"]),
    )


def _parse_material_patch(node: ast.AST) -> MaterialPatch:
    args = _arguments(
        _call(node, "material_patch"),
        positional=("material_id",),
        required=frozenset({"remove_ids", "edits", "additions", "velocity_offset"}),
    )
    return MaterialPatch(
        str(_literal(args["material_id"], str)),
        _string_list(args["remove_ids"]),
        _list(args["edits"], _parse_note_edit),
        _list(args["additions"], _parse_note_clone),
        _int_literal(args["velocity_offset"]),
    )


def parse_material_patch_batch(source: str) -> tuple[str, tuple[MaterialPatch, ...]]:
    """許可した直接呼び出しだけを含む局所差分バッチを解析する。"""

    try:
        root = ast.parse(source, mode="eval")
        args = _arguments(
            _call(root.body, "material_patch_batch"),
            positional=("batch_id",),
            required=frozenset({"patches"}),
        )
        batch_id = str(_literal(args["batch_id"], str))
        return batch_id, _list(args["patches"], _parse_material_patch)
    except (SyntaxError, DslError) as error:
        raise IntensityPatchError(f"invalid material patch batch: {error}") from error


def _require_unique(values: tuple[str, ...], location: str) -> None:
    if len(values) != len(set(values)):
        raise IntensityPatchError(f"{location} are not unique")


def _validate_note_bounds(material: Material, *, at_ms: int, duration_ms: int) -> None:
    if at_ms < 0 or duration_ms <= 0 or at_ms + duration_ms > material.duration_ms:
        raise IntensityPatchError("patched note is outside the material duration")


def _validate_new_onset(original_onsets: set[int], at_ms: int) -> None:
    if at_ms in original_onsets:
        return
    nearest = min((abs(at_ms - onset) for onset in original_onsets), default=10**9)
    if nearest < MIN_NEW_ONSET_DISTANCE_MS:
        raise IntensityPatchError("patched onset introduces a microtiming artifact")


def apply_material_patch(
    material: Material,
    patch: MaterialPatch,
    *,
    target_counts: Mapping[str, Any] | None = None,
) -> Material:
    """検証済み差分を適用し、元の素材を変更せず新しい素材を返す。"""

    if patch.material_id != material.material_id:
        raise IntensityPatchError("material ID does not match the patch")
    by_id = {note.event_id: note for note in material.notes}
    remove_ids = patch.remove_ids
    edit_ids = tuple(edit.event_id for edit in patch.edits)
    addition_ids = tuple(addition.event_id for addition in patch.additions)
    _require_unique(remove_ids, "remove_ids")
    _require_unique(edit_ids, "edit event IDs")
    _require_unique(addition_ids, "addition event IDs")
    unknown_removed = set(remove_ids) - by_id.keys()
    if unknown_removed:
        raise IntensityPatchError("patch has an unknown removed event ID")
    unknown_edited = set(edit_ids) - by_id.keys()
    if unknown_edited:
        raise IntensityPatchError("patch has an unknown edited event ID")
    if set(remove_ids) & set(edit_ids):
        raise IntensityPatchError("an event cannot be removed and edited")
    if set(addition_ids) & by_id.keys():
        raise IntensityPatchError("patch has a duplicate event ID")
    original_onsets = {note.at_ms for note in material.notes}
    edits = {edit.event_id: edit for edit in patch.edits}
    for edit in patch.edits:
        original = by_id[edit.event_id]
        if original.voice == "upper" and edit.at_ms != original.at_ms:
            raise IntensityPatchError("upper voice onset cannot be changed")
        _validate_note_bounds(material, at_ms=edit.at_ms, duration_ms=edit.duration_ms)
        if edit.at_ms != original.at_ms:
            _validate_new_onset(original_onsets, edit.at_ms)

    pitch_classes = {note.pitch % 12 for note in material.notes}
    minimum_pitch = min(note.pitch for note in material.notes)
    maximum_pitch = max(note.pitch for note in material.notes)
    for addition in patch.additions:
        if addition.source_event_id not in by_id:
            raise IntensityPatchError("patch has an unknown clone source event ID")
        if addition.pitch % 12 not in pitch_classes:
            raise IntensityPatchError("clone added an unsupported pitch-class")
        if not minimum_pitch <= addition.pitch <= maximum_pitch:
            raise IntensityPatchError("clone changed the allowed pitch range")
        _validate_note_bounds(material, at_ms=addition.at_ms, duration_ms=addition.duration_ms)
        _validate_new_onset(original_onsets, addition.at_ms)

    revised: list[Note] = []
    for note in material.notes:
        if note.event_id in remove_ids:
            continue
        edit = edits.get(note.event_id)
        velocity = note.velocity + patch.velocity_offset
        if not 1 <= velocity <= 127:
            raise IntensityPatchError("velocity offset causes clipping")
        revised.append(
            replace(
                note,
                at_ms=edit.at_ms if edit is not None else note.at_ms,
                duration_ms=edit.duration_ms if edit is not None else note.duration_ms,
                velocity=velocity,
            )
        )
    for addition in patch.additions:
        source = by_id[addition.source_event_id]
        velocity = source.velocity + patch.velocity_offset
        if not 1 <= velocity <= 127:
            raise IntensityPatchError("velocity offset causes clipping")
        revised.append(
            Note(
                addition.event_id,
                addition.at_ms,
                addition.duration_ms,
                addition.pitch,
                velocity,
                source.voice,
            )
        )
    revised.sort(key=lambda note: (note.at_ms, note.pitch, note.event_id))
    result = replace(material, notes=tuple(revised))
    if target_counts is not None:
        note_count = target_counts.get("note_count")
        attack_count = target_counts.get("attack_count")
        if len(result.notes) != note_count:
            raise IntensityPatchError("patched note count does not match the target")
        if len({note.at_ms for note in result.notes}) != attack_count:
            raise IntensityPatchError("patched attack count does not match the target")
    return result


def _evenly_select[T](items: list[T], count: int) -> list[T]:
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    return [items[index * len(items) // count] for index in range(count)]


def _velocity_offset(notes: list[Note], target_level: float) -> int:
    candidates = []
    for offset in range(-126, 127):
        velocities = [note.velocity + offset for note in notes]
        if not velocities or min(velocities) < 1 or max(velocities) > 127:
            continue
        level = sum(min(velocity // 16, 7) for velocity in velocities) / len(velocities) / 7
        candidates.append((abs(level - target_level), abs(offset), offset))
    if not candidates:
        raise IntensityPatchError("velocity target cannot be represented without clipping")
    return min(candidates)[2]


def _new_onset_slots(material: Material, used: set[int], count: int) -> list[int]:
    original = {note.at_ms for note in material.notes}
    candidates = []
    for at_ms in range(0, material.duration_ms, MIN_NEW_ONSET_DISTANCE_MS):
        if at_ms in used:
            continue
        nearest_original = min(
            (abs(at_ms - onset) for onset in original),
            default=10**9,
        )
        if nearest_original < MIN_NEW_ONSET_DISTANCE_MS:
            continue
        if any(abs(at_ms - onset) < MIN_NEW_ONSET_DISTANCE_MS for onset in candidates):
            continue
        candidates.append(at_ms)
        if len(candidates) == count:
            return candidates
    raise IntensityPatchError("target attack count has no non-microtiming onset layout")


def plan_material_intensity_patch(
    material: Material,
    *,
    target_counts: Mapping[str, Any],
    target_velocity_level: float,
    preserve_all_upper: bool = False,
) -> MaterialPatch:
    """上声の発音位置を保ち、目標件数へ一致する差分を決定的に作る。"""

    note_count = target_counts.get("note_count")
    attack_count = target_counts.get("attack_count")
    if (
        type(note_count) is not int
        or type(attack_count) is not int
        or note_count <= 0
        or attack_count <= 0
        or attack_count > note_count
    ):
        raise IntensityPatchError("target counts are invalid")
    upper = sorted(
        (note for note in material.notes if note.voice == "upper"),
        key=lambda note: (note.at_ms, -note.pitch, note.event_id),
    )
    lower = sorted(
        (note for note in material.notes if note.voice != "upper"),
        key=lambda note: (note.at_ms, note.pitch, note.event_id),
    )
    if not upper or not lower or note_count < 2 or attack_count < 2:
        raise IntensityPatchError("target counts cannot preserve upper and lower voices")

    upper_by_onset: dict[int, list[Note]] = {}
    for note in upper:
        upper_by_onset.setdefault(note.at_ms, []).append(note)
    upper_onsets = (
        sorted(upper_by_onset)
        if preserve_all_upper
        else _evenly_select(
            sorted(upper_by_onset), min(len(upper_by_onset), attack_count - 1, note_count - 1)
        )
    )
    lower_attack_count = attack_count - len(upper_onsets)
    selected_upper = (
        list(upper) if preserve_all_upper else [upper_by_onset[onset][0] for onset in upper_onsets]
    )
    lower_note_count = note_count - len(selected_upper)
    if lower_note_count < lower_attack_count:
        raise IntensityPatchError("target counts cannot preserve both voices")
    selected_lower = _evenly_select(lower, min(len(lower), lower_note_count))

    upper_onset_set = set(upper_onsets)
    original_lower_slots = sorted(
        {note.at_ms for note in lower if note.at_ms not in upper_onset_set}
    )
    lower_slots = _evenly_select(
        original_lower_slots, min(len(original_lower_slots), lower_attack_count)
    )
    missing_slots = lower_attack_count - len(lower_slots)
    if missing_slots:
        lower_slots.extend(
            _new_onset_slots(material, upper_onset_set | set(lower_slots), missing_slots)
        )
    lower_slots.sort()

    remove_ids = [
        note.event_id
        for note in material.notes
        if note not in set(selected_upper) | set(selected_lower)
    ]
    allowed_pitches = sorted({note.pitch for note in material.notes})
    slot_pitches: dict[int, set[int]] = {
        onset: {note.pitch for note in selected_upper if note.at_ms == onset}
        for onset in upper_onsets
    }
    for onset in lower_slots:
        slot_pitches.setdefault(onset, set())

    def compatible(at_ms: int, pitch: int) -> bool:
        existing = slot_pitches[at_ms]
        return (
            len(existing) < 3
            and pitch not in existing
            and all(abs(pitch - other) % 12 not in {1, 2, 6, 10, 11} for other in existing)
        )

    def assign_pair(
        source: Note,
        required_slot: int | None = None,
        *,
        prefer_anchor: bool = False,
    ) -> tuple[int, int]:
        slots = [required_slot] if required_slot is not None else [*lower_slots, *upper_onsets]
        choices = {
            (at_ms, pitch)
            for at_ms in slots
            for pitch in allowed_pitches
            if compatible(at_ms, pitch)
        }
        if prefer_anchor and required_slot is None:
            anchor_choices = {pair for pair in choices if pair[0] in upper_onset_set}
            if anchor_choices:
                choices = anchor_choices
        if not choices:
            raise IntensityPatchError("target layout has no consonant lower note position")
        pair = min(
            choices,
            key=lambda item: (
                abs(item[1] - source.pitch),
                abs(item[0] - source.at_ms),
                item,
            ),
        )
        slot_pitches[pair[0]].add(pair[1])
        return pair

    edits = []
    additions = []
    addition_index = 0
    existing_ids = {note.event_id for note in material.notes}

    def next_addition_id() -> str:
        nonlocal addition_index
        while True:
            addition_index += 1
            event_id = f"{material.material_id}-intensity-{addition_index:03d}"
            if event_id not in existing_ids:
                existing_ids.add(event_id)
                return event_id

    for index, note in enumerate(selected_lower):
        required_slot = lower_slots[index] if index < len(lower_slots) else None
        at_ms, pitch = assign_pair(
            note,
            required_slot,
            prefer_anchor=index >= len(lower_slots),
        )
        duration_ms = min(note.duration_ms, material.duration_ms - at_ms)
        if pitch != note.pitch:
            remove_ids.append(note.event_id)
            additions.append(
                NoteClone(
                    note.event_id,
                    next_addition_id(),
                    at_ms,
                    duration_ms,
                    pitch,
                )
            )
        elif at_ms != note.at_ms or duration_ms != note.duration_ms:
            edits.append(NoteEdit(note.event_id, at_ms, duration_ms))

    clone_sources = selected_lower or lower
    for index in range(lower_note_count - len(selected_lower)):
        source = clone_sources[index % len(clone_sources)]
        slot_index = len(selected_lower) + index
        required_slot = lower_slots[slot_index] if slot_index < len(lower_slots) else None
        at_ms, pitch = assign_pair(
            source,
            required_slot,
            prefer_anchor=slot_index >= len(lower_slots),
        )
        additions.append(
            NoteClone(
                source.event_id,
                next_addition_id(),
                at_ms,
                min(source.duration_ms, material.duration_ms - at_ms),
                pitch,
            )
        )
    velocity_sources = [*selected_upper, *selected_lower]
    velocity_sources.extend(
        next(note for note in material.notes if note.event_id == clone.source_event_id)
        for clone in additions
    )
    return MaterialPatch(
        material.material_id,
        tuple(remove_ids),
        tuple(edits),
        tuple(additions),
        _velocity_offset(velocity_sources, float(target_velocity_level)),
    )


def shorten_same_pitch_overlaps(material: Material) -> Material:
    """同じ音高の次の発音までに前の音を閉じ、発音件数を変えない。"""

    by_pitch: dict[int, list[Note]] = {}
    for note in material.notes:
        by_pitch.setdefault(note.pitch, []).append(note)
    durations: dict[str, int] = {}
    for notes in by_pitch.values():
        ordered = sorted(notes, key=lambda note: (note.at_ms, note.event_id))
        for current, following in pairwise(ordered):
            if current.at_ms == following.at_ms:
                raise IntensityPatchError("same pitch is repeated at one onset")
            durations[current.event_id] = min(current.duration_ms, following.at_ms - current.at_ms)
    return replace(
        material,
        notes=tuple(
            replace(note, duration_ms=durations.get(note.event_id, note.duration_ms))
            for note in material.notes
        ),
    )


def extend_lower_note_durations(material: Material, *, factor: float = 1.10) -> Material:
    """下声の余韻を少し延ばし、同音重複は後段の正規化へ委ねる。"""

    if factor < 1:
        raise IntensityPatchError("lower duration factor must not shorten notes")
    return replace(
        material,
        notes=tuple(
            replace(
                note,
                duration_ms=min(
                    material.duration_ms - note.at_ms,
                    max(1, round(note.duration_ms * factor)),
                ),
            )
            if note.voice != "upper"
            else note
            for note in material.notes
        ),
    )
