"""Deterministic observations for typed requirements in ``solo_piano_3m_v1``."""

from collections import defaultdict
from dataclasses import asdict, dataclass
from statistics import fmean, median
from typing import Literal

from llm_musical_composer.performance_pipeline import (
    PiecePlan,
    RenderedPerformance,
    ScoreNote,
    ScoreSpec,
)

from .requirements import ResolvedRequirement


@dataclass(frozen=True, slots=True)
class Observation:
    """Summary of one feature observed for one section."""

    feature: str
    section_id: str
    status: Literal["observed", "missing"]
    value: float | None
    unit: str
    sample_count: int
    more_when: Literal["higher", "lower"]
    reason: str | None = None
    minimum: float | None = None
    median: float | None = None
    maximum: float | None = None


@dataclass(frozen=True, slots=True)
class RequirementObservationResult:
    """Observed result for one resolved typed requirement."""

    requirement_id: str
    performance_direction_id: str
    feature: str
    relation: str
    target_section_id: str
    reference_section_id: str
    status: Literal["passed", "failed", "missing"]
    observed_relation: Literal["more", "less", "equal", "missing"]
    target: Observation
    reference: Observation

    def value(self) -> dict[str, object]:
        """Return a JSON-compatible diagnostic value."""
        return asdict(self)


def _descendant_leaf_ids(plan: PiecePlan, section_id: str) -> tuple[str, ...]:
    by_parent: dict[str, list[str]] = defaultdict(list)
    nodes = {node.node_id: node for node in plan.nodes}
    if section_id not in nodes:
        return ()
    for node in plan.nodes:
        if node.parent_id is not None:
            by_parent[node.parent_id].append(node.node_id)
    for child_ids in by_parent.values():
        child_ids.sort(key=lambda item: nodes[item].order)
    leaves: list[str] = []
    pending = [section_id]
    while pending:
        node_id = pending.pop()
        child_ids = by_parent.get(node_id, [])
        if child_ids:
            pending.extend(reversed(child_ids))
        else:
            leaves.append(node_id)
    return tuple(leaves)


def _summary(
    *,
    feature: str,
    section_id: str,
    samples: list[float],
    unit: str,
    more_when: Literal["higher", "lower"],
) -> Observation:
    if not samples:
        raise ValueError("an observed summary needs at least one sample")
    return Observation(
        feature=feature,
        section_id=section_id,
        status="observed",
        value=fmean(samples),
        unit=unit,
        sample_count=len(samples),
        more_when=more_when,
        minimum=min(samples),
        median=float(median(samples)),
        maximum=max(samples),
    )


def observe_onset_alignment(
    plan: PiecePlan,
    score: ScoreSpec,
    rendered: RenderedPerformance,
    section_id: str,
) -> Observation:
    """Measure mean onset spread for score-simultaneous multi-voice notes."""
    leaves = _descendant_leaf_ids(plan, section_id)
    materials = {material.material_id: material for material in score.materials}
    nodes = {node.node_id: node for node in plan.nodes}
    performed = {note.event_id: note for note in rendered.notes}
    spreads: list[float] = []
    for leaf_id in leaves:
        material_id = nodes[leaf_id].score_material_id
        if material_id is None:
            continue
        by_onset: dict[int, list[ScoreNote]] = defaultdict(list)
        for note in materials[material_id].notes:
            by_onset[note.at_units].append(note)
        for raw_notes in by_onset.values():
            notes = raw_notes
            if len({note.voice for note in notes}) < 2:
                continue
            performed_notes = [performed.get(f"{leaf_id}:{note.event_id}") for note in notes]
            if any(note is None for note in performed_notes):
                return Observation(
                    feature="onset_alignment",
                    section_id=section_id,
                    status="missing",
                    value=None,
                    unit="ms_mean_spread",
                    sample_count=len(spreads),
                    more_when="lower",
                    reason="rendered_note_missing",
                )
            starts = [note.at_ms for note in performed_notes if note is not None]
            spreads.append(float(max(starts) - min(starts)))
    if not spreads:
        return Observation(
            feature="onset_alignment",
            section_id=section_id,
            status="missing",
            value=None,
            unit="ms_mean_spread",
            sample_count=0,
            more_when="lower",
            reason="no_score_simultaneous_multivoice_onset",
        )
    return _summary(
        feature="onset_alignment",
        section_id=section_id,
        samples=spreads,
        unit="ms_mean_spread",
        more_when="lower",
    )


def observe_loudness(
    plan: PiecePlan, rendered: RenderedPerformance, section_id: str
) -> Observation:
    """Measure mean MIDI velocity for all performed notes in a section subtree."""
    leaves = set(_descendant_leaf_ids(plan, section_id))
    velocities = [
        float(note.velocity) for note in rendered.notes if note.occurrence_node_id in leaves
    ]
    if not velocities:
        return Observation(
            feature="loudness",
            section_id=section_id,
            status="missing",
            value=None,
            unit="mean_midi_velocity",
            sample_count=0,
            more_when="higher",
            reason="no_rendered_notes",
        )
    return _summary(
        feature="loudness",
        section_id=section_id,
        samples=velocities,
        unit="mean_midi_velocity",
        more_when="higher",
    )


def compare_observations(
    target: Observation, reference: Observation
) -> Literal["more", "less", "equal", "missing"]:
    """Compare two observations in the feature's semantic direction."""
    if target.status == "missing" or reference.status == "missing":
        return "missing"
    if target.feature != reference.feature or target.more_when != reference.more_when:
        raise ValueError("observations must measure the same feature")
    assert target.value is not None and reference.value is not None
    if target.value == reference.value:
        return "equal"
    target_is_higher = target.value > reference.value
    target_is_more = target_is_higher == (target.more_when == "higher")
    return "more" if target_is_more else "less"


def evaluate_requirement(
    plan: PiecePlan,
    score: ScoreSpec,
    rendered: RenderedPerformance,
    requirement: ResolvedRequirement,
) -> RequirementObservationResult:
    """Observe and evaluate one typed requirement without model-provided assertions."""
    if requirement.feature == "onset_alignment":
        target = observe_onset_alignment(plan, score, rendered, requirement.target_section_id)
        reference = observe_onset_alignment(plan, score, rendered, requirement.reference_section_id)
    elif requirement.feature == "loudness":
        target = observe_loudness(plan, rendered, requirement.target_section_id)
        reference = observe_loudness(plan, rendered, requirement.reference_section_id)
    else:
        raise ValueError(f"unsupported feature: {requirement.feature}")
    observed_relation = compare_observations(target, reference)
    status: Literal["passed", "failed", "missing"]
    if observed_relation == "missing":
        status = "missing"
    elif observed_relation == requirement.relation:
        status = "passed"
    else:
        status = "failed"
    return RequirementObservationResult(
        requirement_id=requirement.requirement_id,
        performance_direction_id=requirement.performance_direction_id,
        feature=requirement.feature,
        relation=requirement.relation,
        target_section_id=requirement.target_section_id,
        reference_section_id=requirement.reference_section_id,
        status=status,
        observed_relation=observed_relation,
        target=target,
        reference=reference,
    )
