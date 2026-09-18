"""Version-separated performance intermediate representation for solo_piano_3m_v2."""

import hashlib
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import cast

import rfc8785

from .profile_capabilities import (
    GenerationProfileCapabilities,
    solo_piano_3m_v2_capabilities,
)
from .score_ir import PiecePlan, ScoreSpec


class PerformanceIrValidationError(ValueError):
    """A performance intermediate representation violates its contract."""


@dataclass(frozen=True, slots=True)
class SectionPerformance:
    section_id: str
    timing_profile: str | None = None
    timing_amount: str | None = None
    dynamics_profile: str | None = None
    articulation_profile: str | None = None
    coordination_profile: str | None = None
    pedal_profile: str | None = None


@dataclass(frozen=True, slots=True)
class PerformanceSpec:
    performance_id: str
    target_duration_ms: int
    default_velocity: int
    timing_budget_id: str
    section_performances: tuple[SectionPerformance, ...]
    key_release_percent: int = 100
    velocity_policy_id: str = "legacy-unison-v1"


@dataclass(frozen=True, slots=True)
class PerformedNote:
    performed_note_id: str
    source_score_note_ids: tuple[str, ...]
    at_ms: int
    duration_ms: int
    pitch: int
    velocity: int


@dataclass(frozen=True, slots=True)
class PerformedPedal:
    performed_pedal_id: str
    source_score_unit_id: str
    at_ms: int
    value: int


@dataclass(frozen=True, slots=True)
class SectionInterval:
    section_id: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True, slots=True)
class RenderedPerformance:
    performance_id: str
    title: str
    duration_ms: int
    notes: tuple[PerformedNote, ...]
    pedals: tuple[PerformedPedal, ...]
    section_intervals: tuple[SectionInterval, ...]
    piece_plan_sha256: str
    score_spec_sha256: str
    performance_spec_sha256: str


def validate_performance_spec(
    plan: PiecePlan,
    performance: PerformanceSpec,
    capabilities: GenerationProfileCapabilities | None = None,
) -> None:
    """Validate performance targets against the piece plan."""
    active_capabilities = capabilities or solo_piano_3m_v2_capabilities()
    if (
        type(performance.target_duration_ms) is not int
        or performance.target_duration_ms <= 0
        or not 1 <= performance.default_velocity <= 127
        or not 40 <= performance.key_release_percent <= 100
    ):
        raise PerformanceIrValidationError("performance metadata is invalid")
    section_ids = {node.section_id for node in plan.nodes}
    seen: set[str] = set()
    for item in performance.section_performances:
        if item.section_id not in section_ids:
            raise PerformanceIrValidationError("performance refers to an unknown section")
        if item.section_id in seen:
            raise PerformanceIrValidationError("performance section IDs must be unique")
        seen.add(item.section_id)
        profile_values = {
            "timing_profile": item.timing_profile,
            "timing_amount": item.timing_amount,
            "dynamics_profile": item.dynamics_profile,
            "articulation_profile": item.articulation_profile,
            "coordination_profile": item.coordination_profile,
            "pedal_profile": item.pedal_profile,
        }
        try:
            invalid_profile = any(
                value is not None
                and value not in active_capabilities.performance_choices_for_field(field_name)
                for field_name, value in profile_values.items()
            )
        except KeyError:
            invalid_profile = True
        if invalid_profile or (item.timing_profile is None and item.timing_amount is not None):
            raise PerformanceIrValidationError("performance profile vocabulary is invalid")


def validate_rendered_performance(
    plan: PiecePlan,
    score: ScoreSpec,
    performance: PerformanceSpec,
    rendered: RenderedPerformance,
    *,
    enforce_physical_key_contract: bool = True,
) -> None:
    """Validate lineage and the physical-key projection after rendering."""
    expected_note_ids = {
        note.score_note_id
        for unit in score.score_units
        for layer in unit.score_unit_layers
        for note in layer.notes
    }
    source_note_ids = [
        source_id for note in rendered.notes for source_id in note.source_score_note_ids
    ]
    performed_note_ids = [note.performed_note_id for note in rendered.notes]
    if (
        any(not note.source_score_note_ids for note in rendered.notes)
        or any(note.duration_ms <= 0 for note in rendered.notes)
        or set(source_note_ids) != expected_note_ids
        or len(source_note_ids) != len(expected_note_ids)
        or len(performed_note_ids) != len(set(performed_note_ids))
    ):
        raise PerformanceIrValidationError("rendered notes must cover score notes exactly once")
    if enforce_physical_key_contract:
        notes_by_pitch: dict[int, list[PerformedNote]] = {}
        for note in rendered.notes:
            notes_by_pitch.setdefault(note.pitch, []).append(note)
        for same_key in notes_by_pitch.values():
            ordered = sorted(same_key, key=lambda note: (note.at_ms, note.performed_note_id))
            for previous, following in pairwise(ordered):
                if previous.at_ms + previous.duration_ms > following.at_ms:
                    raise PerformanceIrValidationError(
                        "rendered notes on the same piano key must not overlap"
                    )
    score_unit_ids = {unit.score_unit_id for unit in score.score_units}
    if any(pedal.source_score_unit_id not in score_unit_ids for pedal in rendered.pedals):
        raise PerformanceIrValidationError("rendered pedal refers to an unknown score unit")
    section_ids = {node.section_id for node in plan.nodes}
    interval_ids = [interval.section_id for interval in rendered.section_intervals]
    if set(interval_ids) != section_ids or len(interval_ids) != len(section_ids):
        raise PerformanceIrValidationError("rendered section intervals must cover the plan")
    expected_hashes = (
        dataclass_content_sha256(plan),
        dataclass_content_sha256(score),
        dataclass_content_sha256(performance),
    )
    actual_hashes = (
        rendered.piece_plan_sha256,
        rendered.score_spec_sha256,
        rendered.performance_spec_sha256,
    )
    if actual_hashes != expected_hashes:
        raise PerformanceIrValidationError("rendered performance lineage hash mismatch")


def dataclass_content_sha256(value: object) -> str:
    """Hash one immutable IR dataclass using canonical JSON."""
    return hashlib.sha256(rfc8785.dumps(asdict(value))).hexdigest()


def rendered_performance_to_json(
    rendered: RenderedPerformance,
    score: ScoreSpec,
    *,
    schema_version: int,
) -> dict[str, object]:
    """Serialize a rendered performance using its recorded phase-7 contract."""
    value = cast(dict[str, object], asdict(rendered))
    if schema_version == 2:
        return value
    if schema_version != 1:
        raise ValueError("the rendered-performance schema version is unsupported")
    voice_by_note_id = {
        note.score_note_id: note.voice
        for unit in score.score_units
        for layer in unit.score_unit_layers
        for note in layer.notes
    }
    notes = cast(list[dict[str, object]], value["notes"])
    for note in notes:
        source_ids = cast(list[str] | tuple[str, ...], note.pop("source_score_note_ids"))
        if len(source_ids) != 1:
            raise ValueError("phase-7 schema version 1 requires one source per note")
        source_id = source_ids[0]
        note["source_score_note_id"] = source_id
        note["voice"] = voice_by_note_id[source_id]
    return value
