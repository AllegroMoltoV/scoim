"""粗い音域帯から伴奏の実音高を決定するPianoTextureSpecV2実験。"""

from __future__ import annotations

import ast
import itertools
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from llm_musical_composer.performance_pipeline import (
    HARMONY_INTERVALS,
    PiecePlan,
    ScoreHarmony,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
    validate_score_spec,
)
from llm_musical_composer.piano_texture_pilot import (
    PianoTextureSpec,
    PianoTextureValidationError,
)

REGISTER_ZONES: dict[str, tuple[int, int, int]] = {
    "bass": (21, 47, 36),
    "low": (36, 59, 48),
    "middle": (48, 71, 60),
    "high": (60, 108, 72),
}
_ZONE_ORDER = tuple(REGISTER_ZONES)
_ARTICULATIONS = frozenset({"normal", "staccato", "tenuto", "accent"})
_DEGREE_INDEX = {"root": 0, "third": 1, "fifth": 2, "seventh": 3}


@dataclass(frozen=True)
class PianoTextureEventV2:
    event_id: str
    harmony_id: str
    role: str
    voice: str
    at_units: int
    duration_units: int
    degree: str
    register_zone: str
    articulations: tuple[str, ...] = ()


@dataclass(frozen=True)
class PianoTextureSpecV2:
    material_id: str
    events: tuple[PianoTextureEventV2, ...]


@dataclass(frozen=True)
class PianoTexturePlacementResult:
    status: str
    material_id: str
    notes: tuple[ScoreNote, ...]
    failed_onset: int | None
    reason: str | None
    low_spacing_violations: int


@dataclass(frozen=True)
class PianoTextureRearticulationV3:
    event_id: str
    pitch: int
    original_duration_units: int
    final_duration_units: int
    rearticulated_at_units: int


@dataclass(frozen=True)
class PianoTexturePlacementResultV3(PianoTexturePlacementResult):
    rearticulations: tuple[PianoTextureRearticulationV3, ...]


@dataclass(frozen=True)
class PianoTexturePlacementResultV4(PianoTexturePlacementResultV3):
    candidate_evaluation_count: int
    backtrack_count: int


@dataclass(frozen=True)
class PianoTextureZoneProjectionV5:
    event_id: str
    original_zone: str
    effective_zone: str
    reason: str


@dataclass(frozen=True)
class PianoTexturePlacementResultV5(PianoTexturePlacementResultV4):
    zone_projections: tuple[PianoTextureZoneProjectionV5, ...]
    primary_status: str
    primary_failed_onset: int | None
    primary_reason: str | None
    primary_candidate_evaluation_count: int
    primary_backtrack_count: int
    fallback_status: str | None
    fallback_candidate_evaluation_count: int | None
    fallback_backtrack_count: int | None


@dataclass(frozen=True)
class PianoTexturePlacementResultV6(PianoTexturePlacementResultV5):
    v6_zone_projections: tuple[PianoTextureZoneProjectionV5, ...]
    v5_status: str
    v5_candidate_evaluation_count: int
    v5_backtrack_count: int
    v6_fallback_status: str | None
    v6_fallback_candidate_evaluation_count: int | None
    v6_fallback_backtrack_count: int | None


@dataclass(frozen=True)
class PianoTexturePlacementResultV7(PianoTexturePlacementResultV6):
    v7_zone_projections: tuple[PianoTextureZoneProjectionV5, ...]
    v6_status: str
    v6_candidate_evaluation_count: int
    v6_backtrack_count: int
    v7_fallback_status: str | None
    v7_fallback_candidate_evaluation_count: int | None
    v7_fallback_backtrack_count: int | None


@dataclass(frozen=True)
class PianoTexturePlacementResultV8(PianoTexturePlacementResultV7):
    v8_zone_projections: tuple[PianoTextureZoneProjectionV5, ...]
    v7_status: str
    v7_candidate_evaluation_count: int
    v7_backtrack_count: int
    v8_fallback_status: str | None
    v8_fallback_candidate_evaluation_count: int | None
    v8_fallback_backtrack_count: int | None
    total_candidate_evaluation_count: int
    maximum_total_candidate_evaluations: int


@dataclass(frozen=True)
class PianoTextureProjectionResult:
    status: str
    material_id: str
    texture: PianoTextureSpecV2
    event_id_map: dict[str, str]
    reason: str | None = None


def _fail(message: str) -> None:
    raise PianoTextureValidationError(message)


def _call(node: ast.AST, expected_name: str) -> ast.Call:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        _fail("only direct calls to allowed texture V2 DSL functions are accepted")
    if node.func.id != expected_name:
        _fail(f"unknown texture V2 DSL function: {node.func.id}")
    if node.args:
        _fail("positional arguments are not accepted")
    if any(keyword.arg is None for keyword in node.keywords):
        _fail("keyword expansion is not accepted")
    return node


def _arguments(call: ast.Call, *, required: frozenset[str]) -> dict[str, ast.AST]:
    result: dict[str, ast.AST] = {}
    for keyword in call.keywords:
        assert keyword.arg is not None
        if keyword.arg in result:
            _fail(f"duplicate argument: {keyword.arg}")
        result[keyword.arg] = keyword.value
    unknown = set(result) - required
    if unknown:
        _fail(f"unknown argument: {sorted(unknown)[0]}")
    missing = required - result.keys()
    if missing:
        _fail(f"missing argument: {sorted(missing)[0]}")
    return result


def _literal(node: ast.AST, expected: type[int] | type[str]) -> int | str:
    if not isinstance(node, ast.Constant) or type(node.value) is not expected:
        _fail(f"expected a literal {expected.__name__}")
    return node.value


def _list(node: ast.AST, builder: Callable[[ast.AST], Any]) -> tuple[Any, ...]:
    if not isinstance(node, ast.List):
        _fail("expected a list literal")
    return tuple(builder(item) for item in node.elts)


def _event(node: ast.AST) -> PianoTextureEventV2:
    args = _arguments(
        _call(node, "piano_texture_note_v2"),
        required=frozenset(
            {
                "event_id",
                "harmony_id",
                "role",
                "voice",
                "at_units",
                "duration_units",
                "degree",
                "register_zone",
                "articulations",
            }
        ),
    )
    zone = str(_literal(args["register_zone"], str))
    if zone not in REGISTER_ZONES:
        _fail(f"unknown register zone: {zone}")
    return PianoTextureEventV2(
        event_id=str(_literal(args["event_id"], str)),
        harmony_id=str(_literal(args["harmony_id"], str)),
        role=str(_literal(args["role"], str)),
        voice=str(_literal(args["voice"], str)),
        at_units=int(_literal(args["at_units"], int)),
        duration_units=int(_literal(args["duration_units"], int)),
        degree=str(_literal(args["degree"], str)),
        register_zone=zone,
        articulations=_list(args["articulations"], lambda item: str(_literal(item, str))),
    )


def parse_piano_texture_spec_v2(source: str) -> PianoTextureSpecV2:
    """PianoTextureSpecV2をPythonとして実行せず解析する。"""

    try:
        parsed = ast.parse(source, mode="eval")
    except SyntaxError as error:
        raise PianoTextureValidationError(
            f"invalid texture V2 syntax at line {error.lineno}: {error.msg}"
        ) from error
    args = _arguments(
        _call(parsed.body, "piano_texture_spec_v2"),
        required=frozenset({"material_id", "events"}),
    )
    return PianoTextureSpecV2(
        material_id=str(_literal(args["material_id"], str)),
        events=_list(args["events"], _event),
    )


def _call_text(name: str, fields: tuple[tuple[str, str], ...]) -> str:
    return f"{name}(" + ", ".join(f"{key}={value}" for key, value in fields) + ")"


def dump_piano_texture_spec_v2(spec: PianoTextureSpecV2) -> str:
    """PianoTextureSpecV2を決定的な一行DSLへ直列化する。"""

    events = [
        _call_text(
            "piano_texture_note_v2",
            (
                ("event_id", repr(event.event_id)),
                ("harmony_id", repr(event.harmony_id)),
                ("role", repr(event.role)),
                ("voice", repr(event.voice)),
                ("at_units", repr(event.at_units)),
                ("duration_units", repr(event.duration_units)),
                ("degree", repr(event.degree)),
                ("register_zone", repr(event.register_zone)),
                ("articulations", repr(list(event.articulations))),
            ),
        )
        for event in spec.events
    ]
    return _call_text(
        "piano_texture_spec_v2",
        (
            ("material_id", repr(spec.material_id)),
            ("events", "[" + ", ".join(events) + "]"),
        ),
    )


def _material(score: ScoreSpec, material_id: str) -> ScoreMaterial:
    matches = tuple(item for item in score.materials if item.material_id == material_id)
    if len(matches) != 1:
        _fail(f"score must contain exactly one target material: {material_id}")
    return matches[0]


def _overlaps(start_a: int, duration_a: int, start_b: int, duration_b: int) -> bool:
    return start_a < start_b + duration_b and start_b < start_a + duration_a


def _harmony_for_event(material: ScoreMaterial, event: PianoTextureEventV2) -> ScoreHarmony:
    matches = tuple(item for item in material.harmonies if item.harmony_id == event.harmony_id)
    if len(matches) != 1:
        _fail(f"texture harmony_id is unknown: {event.harmony_id}")
    harmony = matches[0]
    if (
        event.at_units < harmony.at_units
        or event.duration_units <= 0
        or event.at_units + event.duration_units > harmony.at_units + harmony.duration_units
    ):
        _fail("texture event must fit inside one declared harmony interval")
    return harmony


def _degree_interval(harmony: ScoreHarmony, degree: str) -> int:
    index = _DEGREE_INDEX.get(degree)
    intervals = HARMONY_INTERVALS[harmony.quality]
    if index is None or index >= len(intervals):
        _fail(f"texture degree is not available in {harmony.quality}: {degree}")
    return intervals[index]


def _candidate_pitches(harmony: ScoreHarmony, event: PianoTextureEventV2) -> tuple[int, ...]:
    interval = _degree_interval(harmony, event.degree)
    pitch_class = (harmony.root_pitch_class + interval) % 12
    low, high, _ = REGISTER_ZONES[event.register_zone]
    return tuple(pitch for pitch in range(low, high + 1) if pitch % 12 == pitch_class)


def _validated_allowed_pitch_range(
    value: tuple[int, int] | None,
) -> tuple[int, int] | None:
    if value is None:
        return None
    if (
        len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or not 21 <= value[0] <= value[1] <= 108
    ):
        _fail("allowed pitch range must be an inclusive MIDI interval")
    return value


def _candidate_pitches_in_range(
    harmony: ScoreHarmony,
    event: PianoTextureEventV2,
    allowed_pitch_range: tuple[int, int] | None,
) -> tuple[int, ...]:
    candidates = _candidate_pitches(harmony, event)
    if allowed_pitch_range is None:
        return candidates
    lower, upper = allowed_pitch_range
    return tuple(pitch for pitch in candidates if lower <= pitch <= upper)


def movement_distance(left: Sequence[int], right: Sequence[int]) -> int:
    """音数が異なる二つの音集合を対称最近傍距離で比較する。"""

    if not left or not right:
        return 0
    return sum(min(abs(a - b) for b in right) for a in left) + sum(
        min(abs(b - a) for a in left) for b in right
    )


def _validate_texture(
    score: ScoreSpec,
    material: ScoreMaterial,
    texture: PianoTextureSpecV2,
) -> dict[str, ScoreHarmony]:
    if material.foreground_voice is None or not material.harmonies:
        _fail("target material must declare harmony and foreground voice")
    if not texture.events:
        _fail("texture V2 must contain accompaniment events")
    accompaniment_voice = "lower" if material.foreground_voice == "upper" else "upper"
    original_ids = {note.event_id for item in score.materials for note in item.notes}
    seen: set[str] = set()
    harmonies: dict[str, ScoreHarmony] = {}
    for event in texture.events:
        if not event.event_id or event.event_id in seen or event.event_id in original_ids:
            _fail("texture V2 event IDs must be new, non-empty, and unique")
        seen.add(event.event_id)
        if event.role != "accompaniment":
            _fail("texture V2 role must be accompaniment")
        if event.voice != accompaniment_voice:
            _fail("texture V2 voice must be the accompaniment voice")
        if event.at_units < 0:
            _fail("texture V2 event position must be non-negative")
        if event.register_zone not in REGISTER_ZONES:
            _fail(f"unknown register zone: {event.register_zone}")
        if len(set(event.articulations)) != len(event.articulations) or any(
            item not in _ARTICULATIONS for item in event.articulations
        ):
            _fail("texture V2 articulations are outside the ScoreNote vocabulary")
        harmony = _harmony_for_event(material, event)
        _degree_interval(harmony, event.degree)
        harmonies[event.event_id] = harmony
    return harmonies


def low_spacing_violations(notes: Sequence[ScoreNote]) -> int:
    """全音符の発音状態が変わる各時刻で低域の下二音間隔違反を数える。"""

    violations = 0
    change_points = {
        point
        for note in notes
        for point in (note.at_units, note.at_units + note.duration_units)
    }
    for point in sorted(change_points):
        sounding = sorted(
            {
                note.pitch
                for note in notes
                if note.at_units <= point < note.at_units + note.duration_units
            }
        )
        if len(sounding) >= 2 and sounding[0] < 48 and sounding[1] - sounding[0] < 7:
            violations += 1
    return violations


def _valid_assignment(
    material: ScoreMaterial,
    events: Sequence[PianoTextureEventV2],
    pitches: Sequence[int],
    placed: Sequence[tuple[PianoTextureEventV2, ScoreNote]],
) -> bool:
    foreground = tuple(
        note for note in material.notes if note.voice == material.foreground_voice
    )
    for event, pitch in zip(events, pitches, strict=True):
        overlapping_foreground = tuple(
            note
            for note in foreground
            if _overlaps(
                event.at_units,
                event.duration_units,
                note.at_units,
                note.duration_units,
            )
        )
        if material.foreground_voice == "upper" and any(
            pitch >= note.pitch for note in overlapping_foreground
        ):
            return False
        if material.foreground_voice == "lower" and any(
            pitch <= note.pitch for note in overlapping_foreground
        ):
            return False

    sounding: list[int] = []
    onset = events[0].at_units
    for _, old_note in placed:
        if old_note.at_units <= onset < old_note.at_units + old_note.duration_units:
            sounding.append(old_note.pitch)
        for event, pitch in zip(events, pitches, strict=True):
            if old_note.pitch == pitch and _overlaps(
                old_note.at_units,
                old_note.duration_units,
                event.at_units,
                event.duration_units,
            ):
                return False
    if len(set(pitches)) != len(pitches):
        return False
    sounding.extend(pitches)
    ordered = sorted(set(sounding))
    return not (
        len(ordered) >= 2 and ordered[0] < 48 and ordered[1] - ordered[0] < 7
    )


def _assignment_rank(
    events: Sequence[PianoTextureEventV2],
    pitches: Sequence[int],
    harmonies: dict[str, ScoreHarmony],
    previous_pitches: Sequence[int],
) -> tuple[int, int, int, int, tuple[int, ...]]:
    lowest_index = min(range(len(pitches)), key=lambda index: (pitches[index], index))
    lowest_pitch = pitches[lowest_index]
    lowest_harmony = harmonies[events[lowest_index].event_id]
    safe_lowest = (
        lowest_pitch >= 48
        or (lowest_pitch - lowest_harmony.root_pitch_class) % 12 in {0, 7}
    )
    center_distance = sum(
        abs(pitch - REGISTER_ZONES[event.register_zone][2])
        for event, pitch in zip(events, pitches, strict=True)
    )
    width = max(pitches) - min(pitches) if len(pitches) > 1 else 0
    return (
        0 if safe_lowest else 1,
        movement_distance(previous_pitches, pitches),
        center_distance,
        width,
        tuple(pitches),
    )


def place_piano_texture_v2(
    plan: PiecePlan,
    score: ScoreSpec,
    texture: PianoTextureSpecV2,
    *,
    allowed_pitch_range: tuple[int, int] | None = None,
) -> PianoTexturePlacementResult:
    """V2イベントを発音位置順に決定配置する。"""

    allowed_pitch_range = _validated_allowed_pitch_range(allowed_pitch_range)
    validate_score_spec(plan, score)
    material = _material(score, texture.material_id)
    harmonies = _validate_texture(score, material, texture)
    by_onset: dict[int, list[PianoTextureEventV2]] = defaultdict(list)
    for event in texture.events:
        by_onset[event.at_units].append(event)

    placed: list[tuple[PianoTextureEventV2, ScoreNote]] = []
    previous_pitches: tuple[int, ...] = ()
    for onset in sorted(by_onset):
        events = tuple(sorted(by_onset[onset], key=lambda item: item.event_id))
        candidate_sets = tuple(
            _candidate_pitches_in_range(
                harmonies[event.event_id], event, allowed_pitch_range
            )
            for event in events
        )
        assignments = tuple(
            pitches
            for pitches in itertools.product(*candidate_sets)
            if _valid_assignment(material, events, pitches, placed)
        )
        if not assignments:
            empty_history_assignments = tuple(
                pitches
                for pitches in itertools.product(*candidate_sets)
                if _valid_assignment(material, events, pitches, ())
            )
            status = "greedy_unplaceable" if empty_history_assignments else "constraint_unplaceable"
            return PianoTexturePlacementResult(
                status,
                texture.material_id,
                (),
                onset,
                "no valid pitch assignment at onset",
                0,
            )
        pitches = min(
            assignments,
            key=lambda item: _assignment_rank(events, item, harmonies, previous_pitches),
        )
        for event, pitch in zip(events, pitches, strict=True):
            placed.append(
                (
                    event,
                    ScoreNote(
                        event.event_id,
                        event.at_units,
                        event.duration_units,
                        pitch,
                        event.voice,
                        tie=None,
                        articulations=event.articulations,
                    ),
                )
            )
        previous_pitches = tuple(sorted(pitches))

    notes = tuple(
        sorted(
            (note for _, note in placed),
            key=lambda item: (item.at_units, item.pitch, item.event_id),
        )
    )
    return PianoTexturePlacementResult(
        "placed",
        texture.material_id,
        notes,
        None,
        None,
        low_spacing_violations(notes),
    )


def _rearticulation_history(
    placed: Sequence[tuple[PianoTextureEventV2, ScoreNote]],
    pitches: Sequence[int],
    onset: int,
) -> tuple[tuple[PianoTextureEventV2, ScoreNote], ...]:
    return tuple(
        item
        for item in placed
        if not (
            item[1].pitch in pitches
            and item[1].at_units < onset < item[1].at_units + item[1].duration_units
        )
    )


def _rearticulation_cost(
    placed: Sequence[tuple[PianoTextureEventV2, ScoreNote]],
    pitches: Sequence[int],
    onset: int,
) -> tuple[int, int]:
    affected = tuple(
        note
        for _, note in placed
        if note.pitch in pitches
        and note.at_units < onset < note.at_units + note.duration_units
    )
    return (
        len(affected),
        sum(note.at_units + note.duration_units - onset for note in affected),
    )


def place_piano_texture_v3(
    plan: PiecePlan,
    score: ScoreSpec,
    texture: PianoTextureSpecV2,
    *,
    allowed_pitch_range: tuple[int, int] | None = None,
) -> PianoTexturePlacementResultV3:
    """再打鍵を最小化し、必要な同一鍵だけ前の音価を切って配置する。"""

    allowed_pitch_range = _validated_allowed_pitch_range(allowed_pitch_range)
    validate_score_spec(plan, score)
    material = _material(score, texture.material_id)
    harmonies = _validate_texture(score, material, texture)
    by_onset: dict[int, list[PianoTextureEventV2]] = defaultdict(list)
    for event in texture.events:
        by_onset[event.at_units].append(event)

    placed: list[tuple[PianoTextureEventV2, ScoreNote]] = []
    rearticulations: list[PianoTextureRearticulationV3] = []
    previous_pitches: tuple[int, ...] = ()
    for onset in sorted(by_onset):
        events = tuple(sorted(by_onset[onset], key=lambda item: item.event_id))
        candidate_sets = tuple(
            _candidate_pitches_in_range(
                harmonies[event.event_id], event, allowed_pitch_range
            )
            for event in events
        )
        assignments = tuple(
            pitches
            for pitches in itertools.product(*candidate_sets)
            if _valid_assignment(
                material,
                events,
                pitches,
                _rearticulation_history(placed, pitches, onset),
            )
        )
        if not assignments:
            empty_history_assignments = tuple(
                pitches
                for pitches in itertools.product(*candidate_sets)
                if _valid_assignment(material, events, pitches, ())
            )
            status = (
                "greedy_unplaceable"
                if empty_history_assignments
                else "constraint_unplaceable"
            )
            return PianoTexturePlacementResultV3(
                status,
                texture.material_id,
                (),
                onset,
                "no valid pitch assignment at onset",
                0,
                (),
            )
        pitches = min(
            assignments,
            key=lambda item: (
                *_rearticulation_cost(placed, item, onset),
                *_assignment_rank(events, item, harmonies, previous_pitches),
            ),
        )
        updated: list[tuple[PianoTextureEventV2, ScoreNote]] = []
        for old_event, old_note in placed:
            if (
                old_note.pitch in pitches
                and old_note.at_units < onset < old_note.at_units + old_note.duration_units
            ):
                final_duration = onset - old_note.at_units
                rearticulations.append(
                    PianoTextureRearticulationV3(
                        old_event.event_id,
                        old_note.pitch,
                        old_note.duration_units,
                        final_duration,
                        onset,
                    )
                )
                old_note = replace(old_note, duration_units=final_duration)
            updated.append((old_event, old_note))
        placed = updated
        for event, pitch in zip(events, pitches, strict=True):
            placed.append(
                (
                    event,
                    ScoreNote(
                        event.event_id,
                        event.at_units,
                        event.duration_units,
                        pitch,
                        event.voice,
                        tie=None,
                        articulations=event.articulations,
                    ),
                )
            )
        previous_pitches = tuple(sorted(pitches))

    notes = tuple(
        sorted(
            (note for _, note in placed),
            key=lambda item: (item.at_units, item.pitch, item.event_id),
        )
    )
    return PianoTexturePlacementResultV3(
        "placed",
        texture.material_id,
        notes,
        None,
        None,
        low_spacing_violations(notes),
        tuple(rearticulations),
    )


class _TextureSearchExhausted(RuntimeError):
    def __init__(self, onset: int) -> None:
        super().__init__("texture pitch search evaluation limit reached")
        self.onset = onset


def place_piano_texture_v4(
    plan: PiecePlan,
    score: ScoreSpec,
    texture: PianoTextureSpecV2,
    *,
    maximum_candidate_evaluations: int = 100_000,
    allowed_pitch_range: tuple[int, int] | None = None,
) -> PianoTexturePlacementResultV4:
    """既存順位を保った上限制御DFSで、後続を塞ぐ局所選択だけを戻す。"""

    if maximum_candidate_evaluations <= 0:
        _fail("texture search evaluation maximum must be positive")
    allowed_pitch_range = _validated_allowed_pitch_range(allowed_pitch_range)
    validate_score_spec(plan, score)
    material = _material(score, texture.material_id)
    harmonies = _validate_texture(score, material, texture)
    by_onset: dict[int, list[PianoTextureEventV2]] = defaultdict(list)
    for event in texture.events:
        by_onset[event.at_units].append(event)
    groups = tuple(
        tuple(sorted(by_onset[onset], key=lambda item: item.event_id))
        for onset in sorted(by_onset)
    )
    evaluation_count = 0
    backtrack_count = 0

    def evaluate(onset: int) -> None:
        nonlocal evaluation_count
        if evaluation_count >= maximum_candidate_evaluations:
            raise _TextureSearchExhausted(onset)
        evaluation_count += 1

    try:
        for events in groups:
            onset = events[0].at_units
            candidate_sets = tuple(
                _candidate_pitches_in_range(
                    harmonies[event.event_id], event, allowed_pitch_range
                )
                for event in events
            )
            has_locally_valid_assignment = False
            for pitches in itertools.product(*candidate_sets):
                evaluate(onset)
                if _valid_assignment(material, events, pitches, ()):
                    has_locally_valid_assignment = True
                    break
            if not has_locally_valid_assignment:
                return PianoTexturePlacementResultV4(
                    "constraint_unplaceable",
                    texture.material_id,
                    (),
                    onset,
                    "no locally valid pitch assignment at onset",
                    0,
                    (),
                    evaluation_count,
                    backtrack_count,
                )

        deepest_onset = groups[0][0].at_units
        failed_states: set[tuple[Any, ...]] = set()

        def search(
            index: int,
            placed: tuple[tuple[PianoTextureEventV2, ScoreNote], ...],
            rearticulations: tuple[PianoTextureRearticulationV3, ...],
            previous_pitches: tuple[int, ...],
        ) -> tuple[
            tuple[tuple[PianoTextureEventV2, ScoreNote], ...],
            tuple[PianoTextureRearticulationV3, ...],
        ] | None:
            nonlocal backtrack_count, deepest_onset
            if index == len(groups):
                foreground = tuple(
                    note
                    for note in material.notes
                    if note.voice == material.foreground_voice
                )
                completed_notes = tuple(note for _, note in placed)
                if low_spacing_violations((*foreground, *completed_notes)):
                    return None
                return placed, rearticulations
            events = groups[index]
            onset = events[0].at_units
            deepest_onset = max(deepest_onset, onset)
            state_key = (
                index,
                previous_pitches,
                tuple(
                    sorted(
                        (
                            note.event_id,
                            note.pitch,
                            note.at_units,
                            note.at_units + note.duration_units,
                        )
                        for _, note in placed
                        if onset < note.at_units + note.duration_units
                    )
                ),
            )
            if state_key in failed_states:
                return None
            assignments = []
            candidate_sets = tuple(
                _candidate_pitches_in_range(
                    harmonies[event.event_id], event, allowed_pitch_range
                )
                for event in events
            )
            for pitches in itertools.product(*candidate_sets):
                evaluate(onset)
                if _valid_assignment(
                    material,
                    events,
                    pitches,
                    _rearticulation_history(placed, pitches, onset),
                ):
                    assignments.append(pitches)
            assignments.sort(
                key=lambda item: (
                    *_rearticulation_cost(placed, item, onset),
                    *_assignment_rank(events, item, harmonies, previous_pitches),
                )
            )
            for pitches in assignments:
                updated = []
                new_rearticulations = list(rearticulations)
                for old_event, old_note in placed:
                    if (
                        old_note.pitch in pitches
                        and old_note.at_units < onset
                        < old_note.at_units + old_note.duration_units
                    ):
                        final_duration = onset - old_note.at_units
                        new_rearticulations.append(
                            PianoTextureRearticulationV3(
                                old_event.event_id,
                                old_note.pitch,
                                old_note.duration_units,
                                final_duration,
                                onset,
                            )
                        )
                        old_note = replace(old_note, duration_units=final_duration)
                    updated.append((old_event, old_note))
                updated.extend(
                    (
                        event,
                        ScoreNote(
                            event.event_id,
                            event.at_units,
                            event.duration_units,
                            pitch,
                            event.voice,
                            tie=None,
                            articulations=event.articulations,
                        ),
                    )
                    for event, pitch in zip(events, pitches, strict=True)
                )
                found = search(
                    index + 1,
                    tuple(updated),
                    tuple(new_rearticulations),
                    tuple(sorted(pitches)),
                )
                if found is not None:
                    return found
                backtrack_count += 1
            failed_states.add(state_key)
            return None

        found = search(0, (), (), ())
    except _TextureSearchExhausted as error:
        return PianoTexturePlacementResultV4(
            "search_exhausted",
            texture.material_id,
            (),
            error.onset,
            str(error),
            0,
            (),
            evaluation_count,
            backtrack_count,
        )
    if found is None:
        return PianoTexturePlacementResultV4(
            "search_unplaceable",
            texture.material_id,
            (),
            deepest_onset,
            "no complete pitch assignment exists within the search space",
            0,
            (),
            evaluation_count,
            backtrack_count,
        )
    placed, rearticulations = found
    notes = tuple(
        sorted(
            (note for _, note in placed),
            key=lambda item: (item.at_units, item.pitch, item.event_id),
        )
    )
    return PianoTexturePlacementResultV4(
        "placed",
        texture.material_id,
        notes,
        None,
        None,
        low_spacing_violations(
            (
                *(
                    note
                    for note in material.notes
                    if note.voice == material.foreground_voice
                ),
                *notes,
            )
        ),
        rearticulations,
        evaluation_count,
        backtrack_count,
    )


def place_piano_texture_v5(
    plan: PiecePlan,
    score: ScoreSpec,
    texture: PianoTextureSpecV2,
    *,
    maximum_candidate_evaluations: int = 100_000,
    allowed_pitch_range: tuple[int, int] | None = None,
) -> PianoTexturePlacementResultV5:
    """V4の解なし時だけ、低いlower前景と重なる伴奏zoneを一段外へ移す。"""

    primary = place_piano_texture_v4(
        plan,
        score,
        texture,
        maximum_candidate_evaluations=maximum_candidate_evaluations,
        allowed_pitch_range=allowed_pitch_range,
    )

    def result(
        final: PianoTexturePlacementResultV4,
        projections: tuple[PianoTextureZoneProjectionV5, ...],
        fallback: PianoTexturePlacementResultV4 | None,
    ) -> PianoTexturePlacementResultV5:
        return PianoTexturePlacementResultV5(
            final.status,
            final.material_id,
            final.notes,
            final.failed_onset,
            final.reason,
            final.low_spacing_violations,
            final.rearticulations,
            final.candidate_evaluation_count,
            final.backtrack_count,
            projections,
            primary.status,
            primary.failed_onset,
            primary.reason,
            primary.candidate_evaluation_count,
            primary.backtrack_count,
            fallback.status if fallback is not None else None,
            fallback.candidate_evaluation_count if fallback is not None else None,
            fallback.backtrack_count if fallback is not None else None,
        )

    if primary.status not in {"constraint_unplaceable", "search_unplaceable"}:
        return result(primary, (), None)
    material = _material(score, texture.material_id)
    if material.foreground_voice != "lower":
        return result(primary, (), None)
    low_foreground = tuple(
        note
        for note in material.notes
        if note.voice == "lower" and note.pitch < 48
    )
    outward = {"bass": "low", "low": "middle", "middle": "high"}
    projections: list[PianoTextureZoneProjectionV5] = []
    projected_events: list[PianoTextureEventV2] = []
    for event in texture.events:
        effective_zone = event.register_zone
        if event.register_zone in outward and any(
            note.at_units < event.at_units + event.duration_units
            and event.at_units < note.at_units + note.duration_units
            for note in low_foreground
        ):
            effective_zone = outward[event.register_zone]
        if effective_zone != event.register_zone:
            projections.append(
                PianoTextureZoneProjectionV5(
                    event.event_id,
                    event.register_zone,
                    effective_zone,
                    "overlaps_lower_foreground_below_midi_48",
                )
            )
            event = replace(event, register_zone=effective_zone)
        projected_events.append(event)
    if not projections:
        return result(primary, (), None)
    fallback = place_piano_texture_v4(
        plan,
        score,
        replace(texture, events=tuple(projected_events)),
        maximum_candidate_evaluations=maximum_candidate_evaluations,
        allowed_pitch_range=allowed_pitch_range,
    )
    return result(fallback, tuple(projections), fallback)


def place_piano_texture_v6(
    plan: PiecePlan,
    score: ScoreSpec,
    texture: PianoTextureSpecV2,
    *,
    maximum_candidate_evaluations: int = 100_000,
    allowed_pitch_range: tuple[int, int] | None = None,
) -> PianoTexturePlacementResultV6:
    """上声前景の下で低域候補間隔が不足するlowだけをmiddleへ移す。"""

    allowed_pitch_range = _validated_allowed_pitch_range(allowed_pitch_range)
    primary = place_piano_texture_v5(
        plan,
        score,
        texture,
        maximum_candidate_evaluations=maximum_candidate_evaluations,
        allowed_pitch_range=allowed_pitch_range,
    )

    def result(
        final: PianoTexturePlacementResultV4 | PianoTexturePlacementResultV5,
        projections: tuple[PianoTextureZoneProjectionV5, ...],
        fallback: PianoTexturePlacementResultV4 | None,
    ) -> PianoTexturePlacementResultV6:
        return PianoTexturePlacementResultV6(
            final.status,
            final.material_id,
            final.notes,
            final.failed_onset,
            final.reason,
            final.low_spacing_violations,
            final.rearticulations,
            final.candidate_evaluation_count,
            final.backtrack_count,
            primary.zone_projections,
            primary.primary_status,
            primary.primary_failed_onset,
            primary.primary_reason,
            primary.primary_candidate_evaluation_count,
            primary.primary_backtrack_count,
            primary.fallback_status,
            primary.fallback_candidate_evaluation_count,
            primary.fallback_backtrack_count,
            projections,
            primary.status,
            primary.candidate_evaluation_count,
            primary.backtrack_count,
            fallback.status if fallback is not None else None,
            fallback.candidate_evaluation_count if fallback is not None else None,
            fallback.backtrack_count if fallback is not None else None,
        )

    if primary.status not in {"constraint_unplaceable", "search_unplaceable"}:
        return result(primary, (), None)
    material = _material(score, texture.material_id)
    if material.foreground_voice != "upper":
        return result(primary, (), None)
    harmonies = _validate_texture(score, material, texture)
    bass_events = tuple(event for event in texture.events if event.register_zone == "bass")
    projections: list[PianoTextureZoneProjectionV5] = []
    projected_events: list[PianoTextureEventV2] = []
    for event in texture.events:
        should_project = False
        if event.register_zone == "low":
            low_candidates = _candidate_pitches_in_range(
                harmonies[event.event_id], event, allowed_pitch_range
            )
            if low_candidates:
                for bass in bass_events:
                    if (
                        bass.harmony_id != event.harmony_id
                        or not _overlaps(
                            bass.at_units,
                            bass.duration_units,
                            event.at_units,
                            event.duration_units,
                        )
                    ):
                        continue
                    bass_candidates = _candidate_pitches_in_range(
                        harmonies[bass.event_id], bass, allowed_pitch_range
                    )
                    if bass_candidates and not any(
                        low_pitch - bass_pitch >= 7
                        for bass_pitch in bass_candidates
                        for low_pitch in low_candidates
                    ):
                        should_project = True
                        break
        if should_project:
            projections.append(
                PianoTextureZoneProjectionV5(
                    event.event_id,
                    "low",
                    "middle",
                    "no_seven_semitone_bass_low_candidate_pair",
                )
            )
            event = replace(event, register_zone="middle")
        projected_events.append(event)
    if not projections:
        return result(primary, (), None)
    fallback = place_piano_texture_v4(
        plan,
        score,
        replace(texture, events=tuple(projected_events)),
        maximum_candidate_evaluations=maximum_candidate_evaluations,
        allowed_pitch_range=allowed_pitch_range,
    )
    return result(fallback, tuple(projections), fallback)


def _place_piano_texture_v7_search(
    plan: PiecePlan,
    score: ScoreSpec,
    texture: PianoTextureSpecV2,
    *,
    maximum_candidate_evaluations: int,
    allowed_pitch_range: tuple[int, int] | None,
) -> tuple[
    PianoTexturePlacementResultV4,
    tuple[PianoTextureZoneProjectionV5, ...],
]:
    """元zoneと一段上方zoneを実音高と同じDFSで選ぶ。"""

    if maximum_candidate_evaluations <= 0:
        _fail("texture search evaluation maximum must be positive")
    allowed_pitch_range = _validated_allowed_pitch_range(allowed_pitch_range)
    validate_score_spec(plan, score)
    material = _material(score, texture.material_id)
    harmonies = _validate_texture(score, material, texture)
    original_by_id = {event.event_id: event for event in texture.events}
    by_onset: dict[int, list[PianoTextureEventV2]] = defaultdict(list)
    for event in texture.events:
        by_onset[event.at_units].append(event)
    groups = tuple(
        tuple(sorted(by_onset[onset], key=lambda item: item.event_id))
        for onset in sorted(by_onset)
    )
    evaluation_count = 0
    backtrack_count = 0

    def evaluate(onset: int) -> None:
        nonlocal evaluation_count
        if evaluation_count >= maximum_candidate_evaluations:
            raise _TextureSearchExhausted(onset)
        evaluation_count += 1

    def zone_variants(
        event: PianoTextureEventV2,
    ) -> tuple[PianoTextureEventV2, ...]:
        if event.register_zone == "bass":
            return (event, replace(event, register_zone="low"))
        if event.register_zone == "low":
            return (event, replace(event, register_zone="middle"))
        return (event,)

    def assignments(
        events: tuple[PianoTextureEventV2, ...],
        placed: tuple[tuple[PianoTextureEventV2, ScoreNote], ...],
        previous_pitches: tuple[int, ...],
    ) -> list[tuple[tuple[PianoTextureEventV2, ...], tuple[int, ...]]]:
        onset = events[0].at_units
        result: list[
            tuple[tuple[PianoTextureEventV2, ...], tuple[int, ...]]
        ] = []
        for effective_events in itertools.product(
            *(zone_variants(event) for event in events)
        ):
            candidate_sets = tuple(
                _candidate_pitches_in_range(
                    harmonies[event.event_id], event, allowed_pitch_range
                )
                for event in effective_events
            )
            for pitches in itertools.product(*candidate_sets):
                evaluate(onset)
                if _valid_assignment(
                    material,
                    effective_events,
                    pitches,
                    _rearticulation_history(placed, pitches, onset),
                ):
                    result.append((effective_events, pitches))
        result.sort(
            key=lambda item: (
                sum(
                    effective.register_zone
                    != original_by_id[effective.event_id].register_zone
                    for effective in item[0]
                ),
                sum(
                    abs(
                        _ZONE_ORDER.index(effective.register_zone)
                        - _ZONE_ORDER.index(
                            original_by_id[effective.event_id].register_zone
                        )
                    )
                    for effective in item[0]
                ),
                *_rearticulation_cost(placed, item[1], onset),
                *_assignment_rank(item[0], item[1], harmonies, previous_pitches),
                tuple(event.register_zone for event in item[0]),
            )
        )
        return result

    try:
        for events in groups:
            if not assignments(events, (), ()):
                return (
                    PianoTexturePlacementResultV4(
                        "constraint_unplaceable",
                        texture.material_id,
                        (),
                        events[0].at_units,
                        "no locally valid zone and pitch assignment at onset",
                        0,
                        (),
                        evaluation_count,
                        backtrack_count,
                    ),
                    (),
                )

        deepest_onset = groups[0][0].at_units
        failed_states: set[tuple[Any, ...]] = set()

        def search(
            index: int,
            placed: tuple[tuple[PianoTextureEventV2, ScoreNote], ...],
            rearticulations: tuple[PianoTextureRearticulationV3, ...],
            previous_pitches: tuple[int, ...],
        ) -> tuple[
            tuple[tuple[PianoTextureEventV2, ScoreNote], ...],
            tuple[PianoTextureRearticulationV3, ...],
        ] | None:
            nonlocal backtrack_count, deepest_onset
            if index == len(groups):
                foreground = tuple(
                    note
                    for note in material.notes
                    if note.voice == material.foreground_voice
                )
                completed_notes = tuple(note for _, note in placed)
                if low_spacing_violations((*foreground, *completed_notes)):
                    return None
                return placed, rearticulations
            events = groups[index]
            onset = events[0].at_units
            deepest_onset = max(deepest_onset, onset)
            state_key = (
                index,
                previous_pitches,
                tuple(
                    sorted(
                        (
                            note.event_id,
                            note.pitch,
                            note.at_units,
                            note.at_units + note.duration_units,
                        )
                        for _, note in placed
                        if onset < note.at_units + note.duration_units
                    )
                ),
            )
            if state_key in failed_states:
                return None
            for effective_events, pitches in assignments(
                events, placed, previous_pitches
            ):
                updated = []
                new_rearticulations = list(rearticulations)
                for old_event, old_note in placed:
                    if (
                        old_note.pitch in pitches
                        and old_note.at_units < onset
                        < old_note.at_units + old_note.duration_units
                    ):
                        final_duration = onset - old_note.at_units
                        new_rearticulations.append(
                            PianoTextureRearticulationV3(
                                old_event.event_id,
                                old_note.pitch,
                                old_note.duration_units,
                                final_duration,
                                onset,
                            )
                        )
                        old_note = replace(old_note, duration_units=final_duration)
                    updated.append((old_event, old_note))
                updated.extend(
                    (
                        event,
                        ScoreNote(
                            event.event_id,
                            event.at_units,
                            event.duration_units,
                            pitch,
                            event.voice,
                            tie=None,
                            articulations=event.articulations,
                        ),
                    )
                    for event, pitch in zip(
                        effective_events, pitches, strict=True
                    )
                )
                found = search(
                    index + 1,
                    tuple(updated),
                    tuple(new_rearticulations),
                    tuple(sorted(pitches)),
                )
                if found is not None:
                    return found
                backtrack_count += 1
            failed_states.add(state_key)
            return None

        found = search(0, (), (), ())
    except _TextureSearchExhausted as error:
        return (
            PianoTexturePlacementResultV4(
                "search_exhausted",
                texture.material_id,
                (),
                error.onset,
                str(error),
                0,
                (),
                evaluation_count,
                backtrack_count,
            ),
            (),
        )
    if found is None:
        return (
            PianoTexturePlacementResultV4(
                "search_unplaceable",
                texture.material_id,
                (),
                deepest_onset,
                "no complete zone and pitch assignment exists within the search space",
                0,
                (),
                evaluation_count,
                backtrack_count,
            ),
            (),
        )
    placed, rearticulations = found
    notes = tuple(
        sorted(
            (note for _, note in placed),
            key=lambda item: (item.at_units, item.pitch, item.event_id),
        )
    )
    projections = tuple(
        PianoTextureZoneProjectionV5(
            event.event_id,
            original_by_id[event.event_id].register_zone,
            event.register_zone,
            "onset_feasible_zone_and_pitch_search",
        )
        for event, _ in sorted(placed, key=lambda item: item[0].event_id)
        if event.register_zone != original_by_id[event.event_id].register_zone
    )
    return (
        PianoTexturePlacementResultV4(
            "placed",
            texture.material_id,
            notes,
            None,
            None,
            low_spacing_violations(
                (
                    *(
                        note
                        for note in material.notes
                        if note.voice == material.foreground_voice
                    ),
                    *notes,
                )
            ),
            rearticulations,
            evaluation_count,
            backtrack_count,
        ),
        projections,
    )


def place_piano_texture_v7(
    plan: PiecePlan,
    score: ScoreSpec,
    texture: PianoTextureSpecV2,
    *,
    maximum_candidate_evaluations: int = 100_000,
    allowed_pitch_range: tuple[int, int] | None = None,
) -> PianoTexturePlacementResultV7:
    """V6の局所解なし時だけ、zoneと実音高を一体探索する。"""

    v6 = place_piano_texture_v6(
        plan,
        score,
        texture,
        maximum_candidate_evaluations=maximum_candidate_evaluations,
        allowed_pitch_range=allowed_pitch_range,
    )

    def result(
        final: PianoTexturePlacementResultV4 | PianoTexturePlacementResultV6,
        projections: tuple[PianoTextureZoneProjectionV5, ...],
        fallback: PianoTexturePlacementResultV4 | None,
    ) -> PianoTexturePlacementResultV7:
        return PianoTexturePlacementResultV7(
            final.status,
            final.material_id,
            final.notes,
            final.failed_onset,
            final.reason,
            final.low_spacing_violations,
            final.rearticulations,
            final.candidate_evaluation_count,
            final.backtrack_count,
            v6.zone_projections,
            v6.primary_status,
            v6.primary_failed_onset,
            v6.primary_reason,
            v6.primary_candidate_evaluation_count,
            v6.primary_backtrack_count,
            v6.fallback_status,
            v6.fallback_candidate_evaluation_count,
            v6.fallback_backtrack_count,
            v6.v6_zone_projections,
            v6.v5_status,
            v6.v5_candidate_evaluation_count,
            v6.v5_backtrack_count,
            v6.v6_fallback_status,
            v6.v6_fallback_candidate_evaluation_count,
            v6.v6_fallback_backtrack_count,
            projections,
            v6.status,
            v6.candidate_evaluation_count,
            v6.backtrack_count,
            fallback.status if fallback is not None else None,
            fallback.candidate_evaluation_count if fallback is not None else None,
            fallback.backtrack_count if fallback is not None else None,
        )

    if v6.status != "constraint_unplaceable":
        return result(v6, (), None)
    material = _material(score, texture.material_id)
    if material.foreground_voice != "upper":
        return result(v6, (), None)
    fallback, projections = _place_piano_texture_v7_search(
        plan,
        score,
        texture,
        maximum_candidate_evaluations=maximum_candidate_evaluations,
        allowed_pitch_range=allowed_pitch_range,
    )
    return result(fallback, projections, fallback)


def _v7_total_candidate_evaluation_count(
    result: PianoTexturePlacementResultV7,
) -> int:
    return sum(
        count
        for count in (
            result.primary_candidate_evaluation_count,
            result.fallback_candidate_evaluation_count,
            result.v6_fallback_candidate_evaluation_count,
            result.v7_fallback_candidate_evaluation_count,
        )
        if count is not None
    )


def place_piano_texture_v8(
    plan: PiecePlan,
    score: ScoreSpec,
    texture: PianoTextureSpecV2,
    *,
    maximum_candidate_evaluations: int = 100_000,
    allowed_pitch_range: tuple[int, int] | None = None,
) -> PianoTexturePlacementResultV8:
    """V7が未探索の全体解なし時だけ、同じ一体探索を一度実行する。"""

    v7 = place_piano_texture_v7(
        plan,
        score,
        texture,
        maximum_candidate_evaluations=maximum_candidate_evaluations,
        allowed_pitch_range=allowed_pitch_range,
    )

    def result(
        final: PianoTexturePlacementResultV4 | PianoTexturePlacementResultV7,
        projections: tuple[PianoTextureZoneProjectionV5, ...],
        fallback: PianoTexturePlacementResultV4 | None,
    ) -> PianoTexturePlacementResultV8:
        values = dict(v7.__dict__)
        for name in (
            "status",
            "material_id",
            "notes",
            "failed_onset",
            "reason",
            "low_spacing_violations",
            "rearticulations",
            "candidate_evaluation_count",
            "backtrack_count",
        ):
            values[name] = getattr(final, name)
        total = _v7_total_candidate_evaluation_count(v7)
        if fallback is not None:
            total += fallback.candidate_evaluation_count
        maximum_total = 3 * maximum_candidate_evaluations
        if total > maximum_total:
            _fail("texture pitch search total evaluation maximum exceeded")
        return PianoTexturePlacementResultV8(
            **values,
            v8_zone_projections=projections,
            v7_status=v7.status,
            v7_candidate_evaluation_count=v7.candidate_evaluation_count,
            v7_backtrack_count=v7.backtrack_count,
            v8_fallback_status=fallback.status if fallback is not None else None,
            v8_fallback_candidate_evaluation_count=(
                fallback.candidate_evaluation_count
                if fallback is not None
                else None
            ),
            v8_fallback_backtrack_count=(
                fallback.backtrack_count if fallback is not None else None
            ),
            total_candidate_evaluation_count=total,
            maximum_total_candidate_evaluations=maximum_total,
        )

    if v7.status != "search_unplaceable" or v7.v7_fallback_status is not None:
        return result(v7, (), None)
    material = _material(score, texture.material_id)
    if material.foreground_voice != "upper":
        return result(v7, (), None)
    fallback, projections = _place_piano_texture_v7_search(
        plan,
        score,
        texture,
        maximum_candidate_evaluations=maximum_candidate_evaluations,
        allowed_pitch_range=allowed_pitch_range,
    )
    return result(fallback, projections, fallback)


def _zone_for_pitch(pitch: int) -> str:
    candidates = tuple(
        zone
        for zone in _ZONE_ORDER
        if REGISTER_ZONES[zone][0] <= pitch <= REGISTER_ZONES[zone][1]
    )
    if not candidates:
        _fail(f"pitch is outside all register zones: {pitch}")
    return min(
        candidates,
        key=lambda zone: (abs(pitch - REGISTER_ZONES[zone][2]), _ZONE_ORDER.index(zone)),
    )


def convert_piano_texture_v1_to_v2(
    plan: PiecePlan,
    score: ScoreSpec,
    texture: PianoTextureSpec,
) -> PianoTextureSpecV2:
    """V1の実音高を音域帯へ置き換え、その他の記号情報を保つ。"""

    validate_score_spec(plan, score)
    material = _material(score, texture.material_id)
    events: list[PianoTextureEventV2] = []
    for event in texture.events:
        harmony = _harmony_for_event(
            material,
            PianoTextureEventV2(
                event.event_id,
                event.harmony_id,
                event.role,
                event.voice,
                event.at_units,
                event.duration_units,
                event.degree,
                "middle",
                event.articulations,
            ),
        )
        pitch_class = (harmony.root_pitch_class + _degree_interval(harmony, event.degree)) % 12
        pitch = 12 * (event.octave + 1) + pitch_class
        events.append(
            PianoTextureEventV2(
                event.event_id,
                event.harmony_id,
                event.role,
                event.voice,
                event.at_units,
                event.duration_units,
                event.degree,
                _zone_for_pitch(pitch),
                event.articulations,
            )
        )
    return PianoTextureSpecV2(texture.material_id, tuple(events))


def project_accompaniment_to_v2(
    plan: PiecePlan,
    score: ScoreSpec,
    material_id: str,
) -> PianoTextureProjectionResult:
    """既知ScoreSpecの伴奏を配置器用V2イベントへ決定的に投影する。"""

    validate_score_spec(plan, score)
    material = _material(score, material_id)
    if material.foreground_voice is None or not material.harmonies:
        _fail("target material must declare harmony and foreground voice")
    accompaniment_voice = "lower" if material.foreground_voice == "upper" else "upper"
    events: list[PianoTextureEventV2] = []
    event_id_map: dict[str, str] = {}
    for note in sorted(
        (item for item in material.notes if item.voice == accompaniment_voice),
        key=lambda item: (item.at_units, item.pitch, item.event_id),
    ):
        harmonies = tuple(
            harmony
            for harmony in material.harmonies
            if harmony.at_units <= note.at_units
            and note.at_units + note.duration_units
            <= harmony.at_units + harmony.duration_units
        )
        if len(harmonies) != 1:
            return PianoTextureProjectionResult(
                "unable_to_investigate",
                material_id,
                PianoTextureSpecV2(material_id, ()),
                event_id_map,
                f"note does not fit one harmony: {note.event_id}",
            )
        harmony = harmonies[0]
        matching_degrees = tuple(
            degree
            for degree, index in _DEGREE_INDEX.items()
            if index < len(HARMONY_INTERVALS[harmony.quality])
            and (harmony.root_pitch_class + HARMONY_INTERVALS[harmony.quality][index]) % 12
            == note.pitch % 12
        )
        if len(matching_degrees) != 1:
            return PianoTextureProjectionResult(
                "unable_to_investigate",
                material_id,
                PianoTextureSpecV2(material_id, ()),
                event_id_map,
                f"note is not one declared degree: {note.event_id}",
            )
        projected_id = f"projected-{note.event_id}"
        event_id_map[note.event_id] = projected_id
        events.append(
            PianoTextureEventV2(
                projected_id,
                harmony.harmony_id,
                "accompaniment",
                accompaniment_voice,
                note.at_units,
                note.duration_units,
                matching_degrees[0],
                _zone_for_pitch(note.pitch),
                note.articulations,
            )
        )
    return PianoTextureProjectionResult(
        "assessed",
        material_id,
        PianoTextureSpecV2(material_id, tuple(events)),
        event_id_map,
    )


def combine_piano_texture_v2(
    plan: PiecePlan,
    score: ScoreSpec,
    placement: PianoTexturePlacementResult,
) -> ScoreSpec:
    """配置済み伴奏だけで対象素材の旧伴奏を置き換える。"""

    if placement.status != "placed":
        _fail("only a placed texture V2 result can be combined")
    target = _material(score, placement.material_id)
    if target.foreground_voice is None:
        _fail("target material must declare foreground voice")
    foreground = tuple(note for note in target.notes if note.voice == target.foreground_voice)
    replacement = replace(
        target,
        notes=tuple(
            sorted(
                (*foreground, *placement.notes),
                key=lambda item: (item.at_units, item.voice, item.pitch, item.event_id),
            )
        ),
    )
    combined = replace(
        score,
        materials=tuple(
            replacement if item.material_id == target.material_id else item
            for item in score.materials
        ),
    )
    validate_score_spec(plan, combined)
    return combined
