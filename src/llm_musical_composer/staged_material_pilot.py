"""和声、旋律、伴奏を薄いDSLで順に組み立てる素材パイロット。"""

from __future__ import annotations

import ast
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

from llm_musical_composer.performance_pipeline import (
    HARMONY_INTERVALS,
    PiecePlan,
    ScoreHarmony,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
    ordered_leaf_schedule,
)
from llm_musical_composer.piano_texture_register_placement import (
    REGISTER_ZONES,
    PianoTextureEventV2,
    PianoTexturePlacementResult,
    PianoTextureSpecV2,
    low_spacing_violations,
    place_piano_texture_v2,
    place_piano_texture_v3,
    place_piano_texture_v4,
    place_piano_texture_v5,
    place_piano_texture_v6,
    place_piano_texture_v7,
    place_piano_texture_v8,
)

_ARTICULATIONS = frozenset({"normal", "staccato", "tenuto", "accent"})
_DEGREE_INDEX = {"root": 0, "third": 1, "fifth": 2, "seventh": 3}


class StagedMaterialError(ValueError):
    """段階素材DSLまたは結合契約の違反。"""


@dataclass(frozen=True)
class HarmonicEventDraft:
    at_units: int
    duration_units: int
    root_pitch_class: int
    quality: str


@dataclass(frozen=True)
class HarmonicDraft:
    events: tuple[HarmonicEventDraft, ...]


@dataclass(frozen=True)
class MelodyEventDraft:
    at_units: int
    duration_units: int
    pitch: int
    articulations: tuple[str, ...] = ()


@dataclass(frozen=True)
class MelodyDraft:
    foreground_voice: str
    events: tuple[MelodyEventDraft, ...]


@dataclass(frozen=True)
class TextureEventDraft:
    harmony_index: int
    at_units: int
    duration_units: int
    degree: str
    register_zone: str
    articulations: tuple[str, ...] = ()


@dataclass(frozen=True)
class TextureDraft:
    events: tuple[TextureEventDraft, ...]


def _fail(message: str) -> None:
    raise StagedMaterialError(message)


def _call(node: ast.AST, name: str) -> ast.Call:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        _fail("only direct calls to allowed staged material functions are accepted")
    if node.func.id != name:
        _fail(f"unknown staged material function: {node.func.id}")
    if node.args or any(keyword.arg is None for keyword in node.keywords):
        _fail("positional arguments and keyword expansion are not accepted")
    return node


def _args(call: ast.Call, names: frozenset[str]) -> dict[str, ast.AST]:
    result: dict[str, ast.AST] = {}
    for keyword in call.keywords:
        assert keyword.arg is not None
        if keyword.arg in result:
            _fail(f"duplicate argument: {keyword.arg}")
        result[keyword.arg] = keyword.value
    if set(result) != names:
        _fail("staged material arguments do not match the contract")
    return result


def _literal(node: ast.AST, expected: type[int] | type[str]) -> int | str:
    if not isinstance(node, ast.Constant) or type(node.value) is not expected:
        _fail(f"expected a literal {expected.__name__}")
    return node.value


def _list(node: ast.AST, builder: Callable[[ast.AST], Any]) -> tuple[Any, ...]:
    if not isinstance(node, ast.List):
        _fail("expected a list literal")
    return tuple(builder(item) for item in node.elts)


def _articulations(node: ast.AST) -> tuple[str, ...]:
    return _list(node, lambda item: str(_literal(item, str)))


def _parse(source: str, name: str, event_builder: Callable[[ast.AST], Any]) -> ast.Call:
    try:
        parsed = ast.parse(source, mode="eval")
    except SyntaxError as error:
        raise StagedMaterialError(f"invalid staged material syntax: {error.msg}") from error
    return _call(parsed.body, name)


def _harmonic_event(node: ast.AST) -> HarmonicEventDraft:
    args = _args(
        _call(node, "harmonic_event"),
        frozenset({"at_units", "duration_units", "root_pitch_class", "quality"}),
    )
    return HarmonicEventDraft(
        int(_literal(args["at_units"], int)),
        int(_literal(args["duration_units"], int)),
        int(_literal(args["root_pitch_class"], int)),
        str(_literal(args["quality"], str)),
    )


def parse_harmonic_draft(source: str) -> HarmonicDraft:
    call = _parse(source, "harmonic_draft", _harmonic_event)
    args = _args(call, frozenset({"events"}))
    return HarmonicDraft(_list(args["events"], _harmonic_event))


def _melody_event(node: ast.AST) -> MelodyEventDraft:
    args = _args(
        _call(node, "melody_event"),
        frozenset({"at_units", "duration_units", "pitch", "articulations"}),
    )
    return MelodyEventDraft(
        int(_literal(args["at_units"], int)),
        int(_literal(args["duration_units"], int)),
        int(_literal(args["pitch"], int)),
        _articulations(args["articulations"]),
    )


def parse_melody_draft(source: str) -> MelodyDraft:
    call = _parse(source, "melody_draft", _melody_event)
    args = _args(call, frozenset({"foreground_voice", "events"}))
    return MelodyDraft(
        str(_literal(args["foreground_voice"], str)),
        _list(args["events"], _melody_event),
    )


def _texture_event(node: ast.AST) -> TextureEventDraft:
    args = _args(
        _call(node, "texture_event"),
        frozenset(
            {
                "harmony_index",
                "at_units",
                "duration_units",
                "degree",
                "register_zone",
                "articulations",
            }
        ),
    )
    return TextureEventDraft(
        int(_literal(args["harmony_index"], int)),
        int(_literal(args["at_units"], int)),
        int(_literal(args["duration_units"], int)),
        str(_literal(args["degree"], str)),
        str(_literal(args["register_zone"], str)),
        _articulations(args["articulations"]),
    )


def parse_texture_draft(source: str) -> TextureDraft:
    call = _parse(source, "texture_draft", _texture_event)
    args = _args(call, frozenset({"events"}))
    return TextureDraft(_list(args["events"], _texture_event))


def _text(name: str, fields: Sequence[tuple[str, object]]) -> str:
    return f"{name}(" + ", ".join(f"{key}={value!r}" for key, value in fields) + ")"


def dump_harmonic_draft(draft: HarmonicDraft) -> str:
    events = [_text("harmonic_event", tuple(vars(event).items())) for event in draft.events]
    return f"harmonic_draft(events=[{', '.join(events)}])"


def dump_melody_draft(draft: MelodyDraft) -> str:
    events = [
        _text(
            "melody_event",
            (
                ("at_units", event.at_units),
                ("duration_units", event.duration_units),
                ("pitch", event.pitch),
                ("articulations", list(event.articulations)),
            ),
        )
        for event in draft.events
    ]
    return (
        f"melody_draft(foreground_voice={draft.foreground_voice!r}, "
        f"events=[{', '.join(events)}])"
    )


def dump_texture_draft(draft: TextureDraft) -> str:
    events = [
        _text(
            "texture_event",
            (
                ("harmony_index", event.harmony_index),
                ("at_units", event.at_units),
                ("duration_units", event.duration_units),
                ("degree", event.degree),
                ("register_zone", event.register_zone),
                ("articulations", list(event.articulations)),
            ),
        )
        for event in draft.events
    ]
    return f"texture_draft(events=[{', '.join(events)}])"


def _existing_ids(score: ScoreSpec) -> tuple[set[str], set[str]]:
    return (
        {note.event_id for material in score.materials for note in material.notes},
        {h.harmony_id for material in score.materials for h in material.harmonies},
    )


def _material_by_id(score: ScoreSpec, material_id: str) -> ScoreMaterial:
    matches = tuple(item for item in score.materials if item.material_id == material_id)
    if len(matches) != 1:
        _fail(f"score must contain exactly one target material: {material_id}")
    return matches[0]


def build_fixed_context(
    case_id: str,
    plan: PiecePlan,
    score: ScoreSpec,
    material_id: str,
) -> dict[str, Any]:
    """旧対象音楽を含めず、三段階に共通する固定文脈を作る。"""

    target = _material_by_id(score, material_id)
    children = [
        material.material_id
        for material in score.materials
        if material.derived_from == material_id
    ]
    if children:
        _fail("pilot target must not have fixed derivative children")
    nodes = [
        {
            "node_id": node.node_id,
            "role": node.role,
            "derived_from": node.derived_from,
            "contrasts_with": node.contrasts_with,
            "harmonic_focus": node.harmonic_focus,
        }
        for node in plan.nodes
        if node.score_material_id == material_id
    ]
    if len(nodes) != 1:
        _fail("pilot target material must have exactly one PiecePlan occurrence")
    schedule, _ = ordered_leaf_schedule(plan, score)
    occurrence_index = next(
        index
        for index, (node, _, _) in enumerate(schedule)
        if node.score_material_id == material_id
    )

    def connection(offset: int) -> dict[str, Any] | None:
        neighbor_index = occurrence_index + offset
        if not 0 <= neighbor_index < len(schedule):
            return None
        neighbor_node = schedule[neighbor_index][0]
        assert neighbor_node.score_material_id is not None
        neighbor = _material_by_id(score, neighbor_node.score_material_id)
        if not neighbor.harmonies:
            return None
        harmony = neighbor.harmonies[-1] if offset < 0 else neighbor.harmonies[0]
        return {
            "node_id": neighbor_node.node_id,
            "material_id": neighbor.material_id,
            "harmony": asdict(harmony),
        }

    relation_source: dict[str, Any] | None = None
    if target.derived_from is not None:
        source = _material_by_id(score, target.derived_from)
        relation_source = {
            "material_id": source.material_id,
            "length_units": source.length_units,
            "harmonies": [asdict(harmony) for harmony in source.harmonies],
            "foreground_voice": source.foreground_voice,
            "note_count": len(source.notes),
        }
    return {
        "schema_version": 1,
        "case_id": case_id,
        "piece": {
            "tonal_center": plan.tonal_center,
            "mode": plan.mode,
            "ending_intent": plan.ending_intent,
        },
        "score": {"score_id": score.score_id, "divisions": score.divisions},
        "target": {
            "material_id": target.material_id,
            "length_units": target.length_units,
            "derived_from": target.derived_from,
            "directions": [asdict(direction) for direction in target.directions],
            "maximum_event_count": min(
                64, 6 * math.ceil(target.length_units / score.divisions)
            ),
        },
        "plan_occurrence": nodes[0],
        "relation_source": relation_source,
        "connection_context": {
            "previous": connection(-1),
            "next": connection(1),
        },
    }


def assemble_harmonies(
    case_id: str,
    target: ScoreMaterial,
    draft: HarmonicDraft,
    score: ScoreSpec,
) -> tuple[ScoreHarmony, ...]:
    if not 2 <= len(draft.events) <= 8:
        _fail("harmonic draft must contain between 2 and 8 events")
    _, existing_harmonies = _existing_ids(score)
    cursor = 0
    result: list[ScoreHarmony] = []
    for index, event in enumerate(draft.events, 1):
        harmony_id = f"pilot-{case_id}-h-{index:03d}"
        if harmony_id in existing_harmonies:
            _fail("runner-assigned harmony ID collides with the fixed score")
        if (
            event.at_units != cursor
            or event.duration_units <= 0
            or not 0 <= event.root_pitch_class <= 11
            or event.quality not in HARMONY_INTERVALS
        ):
            _fail("harmonic draft coverage or vocabulary is invalid")
        cursor += event.duration_units
        result.append(
            ScoreHarmony(
                harmony_id,
                event.at_units,
                event.duration_units,
                event.root_pitch_class,
                event.quality,
            )
        )
    if cursor != target.length_units:
        _fail("harmonic draft must fill the target material")
    return tuple(result)


def _overlaps(left: ScoreNote, right: ScoreNote) -> bool:
    return left.at_units < right.at_units + right.duration_units and right.at_units < (
        left.at_units + left.duration_units
    )


def _safe_opposite_exists(
    foreground_voice: str,
    notes: Sequence[ScoreNote],
    harmonies: Sequence[ScoreHarmony],
) -> bool:
    for onset in sorted({note.at_units for note in notes}):
        sounding = [
            note for note in notes if note.at_units <= onset < note.at_units + note.duration_units
        ]
        harmony = next(
            (h for h in harmonies if h.at_units <= onset < h.at_units + h.duration_units),
            None,
        )
        if harmony is None:
            return False
        pitch_classes = {
            (harmony.root_pitch_class + interval) % 12
            for interval in HARMONY_INTERVALS[harmony.quality]
        }
        candidates = [pitch for pitch in range(21, 109) if pitch % 12 in pitch_classes]
        candidates = [
            pitch
            for pitch in candidates
            if (
                (
                    foreground_voice == "upper"
                    and all(pitch < note.pitch for note in sounding)
                )
                or (
                    foreground_voice == "lower"
                    and all(pitch > note.pitch for note in sounding)
                )
            )
        ]
        if not any(
            len(ordered := sorted({pitch, *(note.pitch for note in sounding)})) < 2
            or ordered[0] >= 48
            or ordered[1] - ordered[0] >= 7
            for pitch in candidates
        ):
            return False
    return True


def assemble_melody(
    case_id: str,
    divisions: int,
    target: ScoreMaterial,
    harmonies: tuple[ScoreHarmony, ...],
    draft: MelodyDraft,
    score: ScoreSpec,
) -> tuple[ScoreNote, ...]:
    if draft.foreground_voice not in {"upper", "lower"}:
        _fail("melody foreground voice is invalid")
    maximum = min(64, 6 * math.ceil(target.length_units / divisions))
    if not 4 <= len(draft.events) <= maximum:
        _fail("melody event count is outside the pilot range")
    existing_events, _ = _existing_ids(score)
    notes: list[ScoreNote] = []
    for index, event in enumerate(draft.events, 1):
        event_id = f"pilot-{case_id}-m-{index:03d}"
        if event_id in existing_events:
            _fail("runner-assigned melody ID collides with the fixed score")
        if (
            event.at_units < 0
            or event.duration_units <= 0
            or event.at_units + event.duration_units > target.length_units
            or not 21 <= event.pitch <= 108
            or len(set(event.articulations)) != len(event.articulations)
            or any(item not in _ARTICULATIONS for item in event.articulations)
        ):
            _fail("melody event is outside the contract")
        notes.append(
            ScoreNote(
                event_id,
                event.at_units,
                event.duration_units,
                event.pitch,
                draft.foreground_voice,
                articulations=event.articulations,
            )
        )
    if len({note.at_units for note in notes}) < 4 or len({note.pitch for note in notes}) < 3:
        _fail("melody must have at least four attacks and three pitches")
    if any(
        left.pitch == right.pitch and _overlaps(left, right)
        for index, left in enumerate(notes)
        for right in notes[index + 1 :]
    ):
        _fail("the same melody pitch cannot overlap")
    if any(
        not any(h.at_units <= note.at_units < h.at_units + h.duration_units for note in notes)
        for h in harmonies
    ):
        _fail("melody must attack in every harmony interval")
    if not _safe_opposite_exists(draft.foreground_voice, notes, harmonies):
        _fail("melody has no safe opposite voice placement")
    return tuple(notes)


def build_texture_feasibility(
    harmonies: Sequence[ScoreHarmony],
    melody: Sequence[ScoreNote],
    *,
    allowed_pitch_range: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """助言用zoneと新規伴奏打鍵数のhard必要条件を作る。"""

    if not melody or len({note.voice for note in melody}) != 1:
        _fail("texture feasibility requires one non-empty foreground voice")
    foreground_voice = melody[0].voice
    if foreground_voice not in {"upper", "lower"}:
        _fail("texture feasibility foreground voice is invalid")
    if allowed_pitch_range is not None:
        lower, upper = allowed_pitch_range
        if not 21 <= lower <= upper <= 108:
            _fail("texture feasibility allowed pitch range is invalid")
    degree_names = ("root", "third", "fifth", "seventh")
    rows: list[dict[str, Any]] = []
    onset_capacities: list[dict[str, Any]] = []
    for harmony_index, harmony in enumerate(harmonies):
        harmony_end = harmony.at_units + harmony.duration_units
        overlapping = tuple(
            note
            for note in melody
            if note.at_units < harmony_end
            and harmony.at_units < note.at_units + note.duration_units
        )
        change_points = {harmony.at_units, harmony_end}
        for note in overlapping:
            change_points.add(max(harmony.at_units, note.at_units))
            end = min(harmony_end, note.at_units + note.duration_units)
            if end < harmony_end:
                change_points.add(end)
        ordered_change_points = sorted(change_points)
        sounding_sets = tuple(
            tuple(
                note.pitch
                for note in overlapping
                if note.at_units <= point < note.at_units + note.duration_units
            )
            for point in ordered_change_points[:-1]
        )
        zones: dict[str, list[str]] = {}
        intervals = HARMONY_INTERVALS[harmony.quality]
        for degree_index, interval in enumerate(intervals):
            pitch_class = (harmony.root_pitch_class + interval) % 12
            allowed: list[str] = []
            for zone, (low, high, _) in REGISTER_ZONES.items():
                if allowed_pitch_range is not None:
                    low = max(low, allowed_pitch_range[0])
                    high = min(high, allowed_pitch_range[1])
                candidates = tuple(
                    pitch for pitch in range(low, high + 1) if pitch % 12 == pitch_class
                )
                if any(
                    all(
                        (
                            not sounding
                            or (
                                foreground_voice == "upper"
                                and candidate < min(sounding)
                            )
                            or (
                                foreground_voice == "lower"
                                and candidate > max(sounding)
                            )
                        )
                        and not (
                            len(ordered := sorted({candidate, *sounding})) >= 2
                            and ordered[0] < 48
                            and ordered[1] - ordered[0] < 7
                        )
                        for sounding in sounding_sets
                    )
                    for candidate in candidates
                ):
                    allowed.append(zone)
            zones[degree_names[degree_index]] = allowed
        pitch_range = (
            [min(note.pitch for note in overlapping), max(note.pitch for note in overlapping)]
            if overlapping
            else None
        )
        rows.append(
            {
                "harmony_index": harmony_index,
                "harmony_id": harmony.harmony_id,
                "overlapping_foreground_pitch_range": pitch_range,
                "individually_feasible_register_zones": zones,
            }
        )
        candidate_pitches: set[int] = set()
        for interval in intervals:
            pitch_class = (harmony.root_pitch_class + interval) % 12
            for zone_low, zone_high, _ in REGISTER_ZONES.values():
                if allowed_pitch_range is not None:
                    zone_low = max(zone_low, allowed_pitch_range[0])
                    zone_high = min(zone_high, allowed_pitch_range[1])
                candidate_pitches.update(
                    pitch
                    for pitch in range(zone_low, zone_high + 1)
                    if pitch % 12 == pitch_class
                )
        for start, end, sounding in zip(
            ordered_change_points[:-1],
            ordered_change_points[1:],
            sounding_sets,
            strict=True,
        ):
            voice_safe = tuple(
                pitch
                for pitch in sorted(candidate_pitches)
                if not sounding
                or (foreground_voice == "upper" and pitch < min(sounding))
                or (foreground_voice == "lower" and pitch > max(sounding))
            )
            onset_capacities.append(
                {
                    "harmony_index": harmony_index,
                    "harmony_id": harmony.harmony_id,
                    "start_units": start,
                    "end_units": end,
                    "sounding_foreground_pitches": sorted(set(sounding)),
                    "maximum_new_accompaniment_attack_count": (
                        _maximum_low_spacing_compatible_additions(
                            sounding, voice_safe
                        )
                    ),
                    "allowed_pitch_range": (
                        list(allowed_pitch_range)
                        if allowed_pitch_range is not None
                        else None
                    ),
                    "constraint_id": "new_accompaniment_onset_capacity_v1",
                }
            )
    result: dict[str, Any] = {
        "schema_version": 3,
        "scope": "texture_generation_feasibility_v3",
        "advisory_scope": "whole_harmony_universal_safety_advisory",
        "foreground_voice": foreground_voice,
        "accompaniment_voice": "lower" if foreground_voice == "upper" else "upper",
        "allowed_pitch_range": (
            list(allowed_pitch_range) if allowed_pitch_range is not None else None
        ),
        "harmonies": rows,
        "onset_capacities": onset_capacities,
    }
    result["harmony_start_accompaniment"] = harmony_start_accompaniment_policy(
        harmonies, melody, result
    )
    return result


def harmony_start_accompaniment_policy(
    harmonies: Sequence[ScoreHarmony],
    melody: Sequence[ScoreNote],
    feasibility: Mapping[str, Any],
) -> dict[str, Any]:
    """旋律だけで和声開始を担える位置の伴奏同時打鍵を免除する。"""

    capacity_rows = tuple(
        row
        for row in feasibility.get("onset_capacities", ())
        if isinstance(row, Mapping)
    )
    minimums: dict[str, int] = {}
    melody_led_starts: list[dict[str, Any]] = []
    for harmony_index, harmony in enumerate(harmonies):
        start = harmony.at_units
        row = next(
            (
                candidate
                for candidate in capacity_rows
                if int(candidate.get("harmony_index", -1)) == harmony_index
                and int(candidate.get("start_units", -1)) <= start
                < int(candidate.get("end_units", -1))
            ),
            None,
        )
        if row is None:
            _fail("harmony start onset capacity is missing")
        maximum = int(row["maximum_new_accompaniment_attack_count"])
        melody_pitches = sorted(note.pitch for note in melody if note.at_units == start)
        chord_pitch_classes = sorted(
            {
                (harmony.root_pitch_class + interval) % 12
                for interval in HARMONY_INTERVALS[harmony.quality]
            }
        )
        melody_leads = (
            maximum == 0
            and bool(melody_pitches)
            and all(pitch % 12 in chord_pitch_classes for pitch in melody_pitches)
        )
        minimums[str(start)] = 0 if melody_leads else 1
        if melody_leads:
            melody_led_starts.append(
                {
                    "harmony_index": harmony_index,
                    "harmony_id": harmony.harmony_id,
                    "at_units": start,
                    "maximum_new_accompaniment_attack_count": maximum,
                    "melody_pitches": melody_pitches,
                    "chord_pitch_classes": chord_pitch_classes,
                    "reason": "capacity_zero_and_melody_attacks_chord_tones",
                }
            )
    return {
        "schema_version": 1,
        "policy_id": "melody-led-harmony-start-v1",
        "minimum_new_accompaniment_attacks": minimums,
        "melody_led_starts": melody_led_starts,
    }


def _pitches_pass_low_spacing(pitches: Sequence[int]) -> bool:
    ordered = sorted(set(pitches))
    return len(ordered) < 2 or ordered[0] >= 48 or ordered[1] - ordered[0] >= 7


def _maximum_low_spacing_compatible_additions(
    fixed_pitches: Sequence[int], candidate_pitches: Sequence[int]
) -> int:
    """下二音だけに依存する規則から、追加可能な異音高数を多項式時間で求める。"""

    fixed = tuple(sorted(set(fixed_pitches)))
    candidates = tuple(sorted(set(candidate_pitches) - set(fixed)))
    selections: list[tuple[int, ...]] = [()]
    selections.append(tuple(pitch for pitch in candidates if pitch >= 48))
    for lowest in (pitch for pitch in (*fixed, *candidates) if pitch < 48):
        selected = tuple(
            pitch
            for pitch in candidates
            if pitch == lowest or pitch >= lowest + 7
        )
        selections.append(selected)
    return max(
        (
            len(selected)
            for selected in selections
            if _pitches_pass_low_spacing((*fixed, *selected))
        ),
        default=0,
    )


def texture_onset_capacity_violations(
    draft: TextureDraft, feasibility: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """同じ開始位置の新規伴奏打鍵数がhard必要条件を超えた箇所を返す。"""

    rows = tuple(
        row
        for row in feasibility.get("onset_capacities", [])
        if isinstance(row, Mapping)
    )
    groups = Counter((event.harmony_index, event.at_units) for event in draft.events)
    violations: list[dict[str, Any]] = []
    for (harmony_index, at_units), actual in sorted(groups.items()):
        row = next(
            (
                item
                for item in rows
                if int(item.get("harmony_index", -1)) == harmony_index
                and int(item.get("start_units", -1)) <= at_units
                < int(item.get("end_units", -1))
            ),
            None,
        )
        if row is None:
            violations.append(
                {
                    "harmony_index": harmony_index,
                    "at_units": at_units,
                    "actual_new_accompaniment_attack_count": actual,
                    "maximum_new_accompaniment_attack_count": None,
                    "reason": "onset_capacity_missing",
                }
            )
            continue
        maximum = int(row["maximum_new_accompaniment_attack_count"])
        if actual > maximum:
            violations.append(
                {
                    "harmony_index": harmony_index,
                    "at_units": at_units,
                    "actual_new_accompaniment_attack_count": actual,
                    "maximum_new_accompaniment_attack_count": maximum,
                    "reason": "onset_capacity_exceeded",
                }
            )
    return violations


def texture_budget_onset_capacity_reachability(
    length_units: int,
    melody: Sequence[ScoreNote],
    material_budget: Mapping[str, Any],
    feasibility: Mapping[str, Any],
    *,
    minimum_new_accompaniment_attacks: Mapping[int, int] | None = None,
    last_usable_attack_units: int | None = None,
) -> dict[str, Any]:
    """発音群サイズ予算を局所の新規伴奏打鍵上限へ割り当てられるか調べる。"""

    if length_units <= 0:
        _fail("texture budget reachability length must be positive")
    if last_usable_attack_units is not None and not (
        0 <= last_usable_attack_units < length_units
    ):
        _fail("texture budget reachability last usable attack is invalid")
    counts_raw = material_budget.get("attack_size_counts")
    if not isinstance(counts_raw, Mapping):
        _fail("texture budget reachability attack-size counts are invalid")
    bucket_counts = {
        name: int(counts_raw.get(name, 0))
        for name in ("one", "two", "three", "four_or_more")
    }
    if any(value < 0 for value in bucket_counts.values()):
        _fail("texture budget reachability attack-size counts are invalid")
    minimum_additions = {
        int(at_units): int(count)
        for at_units, count in (minimum_new_accompaniment_attacks or {}).items()
    }
    if any(
        not 0 <= at_units < length_units or count < 0
        for at_units, count in minimum_additions.items()
    ):
        _fail("texture budget reachability minimum additions are invalid")
    minimum_additions_record = {
        str(at_units): count
        for at_units, count in sorted(minimum_additions.items())
    }
    required_group_count = sum(bucket_counts.values())
    declared_group_count = int(
        material_budget.get("combined_attack_group_count", required_group_count)
    )
    if declared_group_count != required_group_count:
        _fail("texture budget reachability group count is inconsistent")
    if required_group_count > length_units:
        return {
            "schema_version": 2,
            "reachable": False,
            "reason": "attack_group_count_exceeds_positions",
            "position_count": length_units,
            "required_attack_group_count": required_group_count,
            "minimum_new_accompaniment_attacks": minimum_additions_record,
            "last_usable_attack_units": last_usable_attack_units,
        }
    capacity_rows = tuple(
        row
        for row in feasibility.get("onset_capacities", [])
        if isinstance(row, Mapping)
    )
    melody_attacks = Counter(note.at_units for note in melody)
    right_slots = [
        f"{bucket}:{index}"
        for bucket, count in bucket_counts.items()
        for index in range(count)
    ]
    right_slots.extend(
        f"unused:{index}" for index in range(length_units - required_group_count)
    )
    edges: list[list[int]] = []
    for at_units in range(length_units):
        row = next(
            (
                item
                for item in capacity_rows
                if int(item.get("start_units", -1)) <= at_units
                < int(item.get("end_units", -1))
            ),
            None,
        )
        if row is None:
            return {
                "schema_version": 2,
                "reachable": False,
                "reason": "onset_capacity_missing",
                "position_count": length_units,
                "required_attack_group_count": required_group_count,
                "failed_at_units": at_units,
                "minimum_new_accompaniment_attacks": minimum_additions_record,
                "last_usable_attack_units": last_usable_attack_units,
            }
        base = melody_attacks[at_units]
        maximum = int(row["maximum_new_accompaniment_attack_count"])
        minimum_addition = minimum_additions.get(at_units, 0)
        minimum_total = base + minimum_addition
        if minimum_total == 0:
            minimum_total = 1
        maximum_total = base + maximum
        after_last_usable_attack = (
            last_usable_attack_units is not None
            and at_units > last_usable_attack_units
        )
        allowed_buckets = (
            set()
            if after_last_usable_attack
            else {
                _attack_size_bucket(total)
                for total in range(minimum_total, maximum_total + 1)
            }
        )
        position_edges = [
            index
            for index, slot in enumerate(right_slots)
            if slot.split(":", 1)[0] in allowed_buckets
            or (
                slot.startswith("unused:")
                and base == 0
                and minimum_addition == 0
            )
        ]
        edges.append(position_edges)

    matched_left_by_right: dict[int, int] = {}

    def assign(left: int, visited: set[int]) -> bool:
        for right in edges[left]:
            if right in visited:
                continue
            visited.add(right)
            previous = matched_left_by_right.get(right)
            if previous is None or assign(previous, visited):
                matched_left_by_right[right] = left
                return True
        return False

    matched = sum(assign(left, set()) for left in range(length_units))
    reachable = matched == length_units
    return {
        "schema_version": 2,
        "reachable": reachable,
        "reason": None if reachable else "attack_size_buckets_unassignable",
        "position_count": length_units,
        "required_attack_group_count": required_group_count,
        "matched_position_count": matched,
        "minimum_new_accompaniment_attacks": minimum_additions_record,
        "last_usable_attack_units": last_usable_attack_units,
    }


def _attack_size_bucket(size: int) -> str:
    if size == 1:
        return "one"
    if size == 2:
        return "two"
    if size == 3:
        return "three"
    return "four_or_more"


def texture_feasibility_violations(
    draft: TextureDraft, feasibility: dict[str, Any]
) -> list[dict[str, Any]]:
    """LLM指定のdegree・zoneが単体必要条件表にないeventを返す。"""

    rows = {
        int(row["harmony_index"]): row
        for row in feasibility.get("harmonies", [])
        if isinstance(row, dict) and "harmony_index" in row
    }
    violations: list[dict[str, Any]] = []
    for event_index, event in enumerate(draft.events):
        row = rows.get(event.harmony_index, {})
        zones = row.get("individually_feasible_register_zones", {})
        allowed = zones.get(event.degree, []) if isinstance(zones, dict) else []
        if event.register_zone not in allowed:
            violations.append(
                {
                    "event_index": event_index,
                    "harmony_index": event.harmony_index,
                    "degree": event.degree,
                    "register_zone": event.register_zone,
                }
            )
    return violations


def texture_event_feasibility_violations(
    draft: TextureDraft,
    harmonies: Sequence[ScoreHarmony],
    melody: Sequence[ScoreNote],
    *,
    allowed_pitch_range: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    """各伴奏eventの実時間区間に候補音があるかをhard検査する。"""

    if not melody or len({note.voice for note in melody}) != 1:
        _fail("texture event feasibility requires one non-empty foreground voice")
    foreground_voice = melody[0].voice
    if foreground_voice not in {"upper", "lower"}:
        _fail("texture event feasibility foreground voice is invalid")
    if allowed_pitch_range is not None:
        lower, upper = allowed_pitch_range
        if not 21 <= lower <= upper <= 108:
            _fail("texture event feasibility allowed pitch range is invalid")
    violations: list[dict[str, Any]] = []
    for event_index, event in enumerate(draft.events):
        if not 0 <= event.harmony_index < len(harmonies):
            violations.append({"event_index": event_index, "reason": "invalid_harmony"})
            continue
        harmony = harmonies[event.harmony_index]
        intervals = HARMONY_INTERVALS[harmony.quality]
        degree_index = _DEGREE_INDEX.get(event.degree)
        if degree_index is None or degree_index >= len(intervals):
            violations.append({"event_index": event_index, "reason": "invalid_degree"})
            continue
        event_start = event.at_units
        event_end = event.at_units + event.duration_units
        harmony_end = harmony.at_units + harmony.duration_units
        if event_start < harmony.at_units or event_end > harmony_end:
            violations.append({"event_index": event_index, "reason": "outside_harmony"})
            continue
        overlapping = tuple(
            note
            for note in melody
            if note.at_units < event_end
            and event_start < note.at_units + note.duration_units
        )
        change_points = {event_start}
        for note in overlapping:
            if event_start < note.at_units < event_end:
                change_points.add(note.at_units)
            note_end = note.at_units + note.duration_units
            if event_start < note_end < event_end:
                change_points.add(note_end)
        sounding_sets = tuple(
            tuple(
                note.pitch
                for note in overlapping
                if note.at_units <= point < note.at_units + note.duration_units
            )
            for point in sorted(change_points)
        )
        low, high, _ = REGISTER_ZONES[event.register_zone]
        if allowed_pitch_range is not None:
            low = max(low, allowed_pitch_range[0])
            high = min(high, allowed_pitch_range[1])
        pitch_class = (harmony.root_pitch_class + intervals[degree_index]) % 12
        candidates = tuple(
            pitch for pitch in range(low, high + 1) if pitch % 12 == pitch_class
        )
        feasible = any(
            all(
                (
                    not sounding
                    or (foreground_voice == "upper" and candidate < min(sounding))
                    or (foreground_voice == "lower" and candidate > max(sounding))
                )
                for sounding in sounding_sets
            )
            for candidate in candidates
        )
        if not feasible:
            violations.append(
                {
                    "event_index": event_index,
                    "harmony_index": event.harmony_index,
                    "degree": event.degree,
                    "register_zone": event.register_zone,
                    "reason": "no_individual_pitch_candidate",
                }
            )
    return violations


def texture_preplacement_violations(
    draft: TextureDraft,
    harmonies: Sequence[ScoreHarmony],
    melody: Sequence[ScoreNote],
    *,
    allowed_pitch_range: tuple[int, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """配置前に確定できるevent条件と発音位置容量を同じ基準で返す。"""

    feasibility = build_texture_feasibility(
        harmonies,
        melody,
        allowed_pitch_range=allowed_pitch_range,
    )
    return {
        "event_feasibility": texture_event_feasibility_violations(
            draft,
            harmonies,
            melody,
            allowed_pitch_range=allowed_pitch_range,
        ),
        "onset_capacity": texture_onset_capacity_violations(draft, feasibility),
    }


def assemble_texture(
    case_id: str,
    plan: PiecePlan,
    score: ScoreSpec,
    target: ScoreMaterial,
    harmonies: tuple[ScoreHarmony, ...],
    melody: tuple[ScoreNote, ...],
    draft: TextureDraft,
    *,
    maximum_event_count: int | None = None,
    placement_policy: str = "strict-v2",
    allowed_pitch_range: tuple[int, int] | None = None,
) -> tuple[PianoTexturePlacementResult, ScoreMaterial]:
    maximum = (
        min(64, 6 * math.ceil(target.length_units / score.divisions))
        if maximum_event_count is None
        else maximum_event_count
    )
    if maximum <= 0:
        _fail("texture event maximum must be positive")
    if not draft.events or len(draft.events) > maximum:
        _fail("texture event count is outside the pilot range")
    has_chord_attack = any(
        sum(event.at_units == onset for event in draft.events) >= 2
        for onset in {event.at_units for event in draft.events}
    )
    has_arpeggio = any(
        len({event.at_units for event in group}) >= 3
        and len({event.degree for event in group}) >= 2
        for harmony_index in {event.harmony_index for event in draft.events}
        if (
            group := tuple(
                event for event in draft.events if event.harmony_index == harmony_index
            )
        )
    )
    if not has_chord_attack and not has_arpeggio:
        _fail("texture must contain a chord attack or a varied arpeggio")
    foreground_voice = melody[0].voice
    accompaniment_voice = "lower" if foreground_voice == "upper" else "upper"
    events: list[PianoTextureEventV2] = []
    for index, event in enumerate(draft.events, 1):
        if not 0 <= event.harmony_index < len(harmonies):
            _fail("texture harmony index is invalid")
        events.append(
            PianoTextureEventV2(
                f"pilot-{case_id}-a-{index:03d}",
                harmonies[event.harmony_index].harmony_id,
                "accompaniment",
                accompaniment_voice,
                event.at_units,
                event.duration_units,
                event.degree,
                event.register_zone,
                event.articulations,
            )
        )
    feasibility = build_texture_feasibility(
        harmonies, melody, allowed_pitch_range=allowed_pitch_range
    )
    start_minimums = feasibility["harmony_start_accompaniment"][
        "minimum_new_accompaniment_attacks"
    ]
    if any(
        not any(event.at_units == onset for event in events)
        for onset, minimum in (
            (int(at_units), int(required))
            for at_units, required in start_minimums.items()
        )
        if minimum > 0
    ):
        _fail("texture must attack at every required harmony interval start")
    provisional = replace(
        target,
        notes=melody,
        harmonies=harmonies,
        foreground_voice=foreground_voice,
    )
    provisional_score = replace(
        score,
        materials=tuple(
            provisional if material.material_id == target.material_id else material
            for material in score.materials
        ),
    )
    texture = PianoTextureSpecV2(target.material_id, tuple(events))
    if placement_policy == "strict-v2":
        placement = place_piano_texture_v2(
            plan,
            provisional_score,
            texture,
            allowed_pitch_range=allowed_pitch_range,
        )
    elif placement_policy == "same-key-rearticulation-v3":
        placement = place_piano_texture_v3(
            plan,
            provisional_score,
            texture,
            allowed_pitch_range=allowed_pitch_range,
        )
    elif placement_policy == "bounded-backtracking-v4":
        placement = place_piano_texture_v4(
            plan,
            provisional_score,
            texture,
            allowed_pitch_range=allowed_pitch_range,
        )
    elif placement_policy == "low-foreground-outward-v5":
        placement = place_piano_texture_v5(
            plan,
            provisional_score,
            texture,
            allowed_pitch_range=allowed_pitch_range,
        )
    elif placement_policy == "bidirectional-low-spacing-v6":
        placement = place_piano_texture_v6(
            plan,
            provisional_score,
            texture,
            allowed_pitch_range=allowed_pitch_range,
        )
    elif placement_policy == "onset-feasible-zone-v7":
        placement = place_piano_texture_v7(
            plan,
            provisional_score,
            texture,
            allowed_pitch_range=allowed_pitch_range,
        )
    elif placement_policy == "search-aware-onset-zone-v8":
        placement = place_piano_texture_v8(
            plan,
            provisional_score,
            texture,
            allowed_pitch_range=allowed_pitch_range,
        )
    else:
        _fail(f"unknown texture placement policy: {placement_policy}")
    if placement.status != "placed":
        return placement, provisional
    material = replace(
        provisional,
        notes=tuple(
            sorted(
                (*melody, *placement.notes),
                key=lambda note: (note.at_units, note.voice, note.pitch, note.event_id),
            )
        ),
    )
    return placement, material


def full_low_spacing_violations(material: ScoreMaterial) -> int:
    """前景と伴奏を合わせ、全発音位置で低域の下二音間隔を数える。"""

    return low_spacing_violations(material.notes)
