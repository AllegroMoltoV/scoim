"""Private one-way adapter from the v2 score boundary to proven low-level rendering."""

from collections.abc import Mapping
from pathlib import Path
from typing import cast

from llm_musical_composer.performance_pipeline import (
    NodePerformance as LegacyNodePerformance,
)
from llm_musical_composer.performance_pipeline import PerformanceSpec as LegacyPerformanceSpec
from llm_musical_composer.performance_pipeline import PerformedNote as LegacyPerformedNote
from llm_musical_composer.performance_pipeline import PerformedPedal as LegacyPerformedPedal
from llm_musical_composer.performance_pipeline import PiecePlan as LegacyPiecePlan
from llm_musical_composer.performance_pipeline import PlanNode as LegacyPlanNode
from llm_musical_composer.performance_pipeline import (
    RenderedPerformance as LegacyRenderedPerformance,
)
from llm_musical_composer.performance_pipeline import ScoreDirection as LegacyScoreDirection
from llm_musical_composer.performance_pipeline import ScoreHarmony as LegacyScoreHarmony
from llm_musical_composer.performance_pipeline import ScoreMaterial as LegacyScoreMaterial
from llm_musical_composer.performance_pipeline import ScoreNote as LegacyScoreNote
from llm_musical_composer.performance_pipeline import ScoreSpec as LegacyScoreSpec
from llm_musical_composer.performance_pipeline import (
    render_performance_smf,
    render_role_neutral_musicxml,
    render_role_neutral_performance_with_pedal_sources,
    validate_musicxml_round_trip,
    validate_smf_round_trip,
)

from .performance_ir import (
    PerformanceSpec,
    PerformedNote,
    PerformedPedal,
    RenderedPerformance,
    SectionInterval,
    dataclass_content_sha256,
    validate_performance_spec,
    validate_rendered_performance,
)
from .profile_capabilities import solo_piano_3m_v2_capabilities
from .score_ir import PiecePlan, ScoreSpec, validate_score_ir
from .validation import IssueCode, ValidationIssue


class ScoreRenderingError(ValueError):
    """The private renderer adapter cannot preserve the v2 input contract."""

    def __init__(self, issue: ValidationIssue | str) -> None:
        if isinstance(issue, str):
            issue = ValidationIssue(IssueCode.SEMANTIC_INVALID, issue, "")
        super().__init__(issue.message)
        self.issue = issue


def render_score_performance(
    script_document: Mapping[str, object],
    plan: PiecePlan,
    score: ScoreSpec,
    performance: PerformanceSpec,
) -> RenderedPerformance:
    """Render v2 score data while preserving an explicit note-source correspondence."""
    capabilities = solo_piano_3m_v2_capabilities()
    if not capabilities.supports_velocity_policy(performance.velocity_policy_id):
        raise ScoreRenderingError(
            ValidationIssue(
                IssueCode.UNREPRESENTABLE,
                f"Unsupported velocity policy: {performance.velocity_policy_id}",
                "/performance/velocity_policy_id",
            )
        )
    validate_performance_spec(plan, performance)
    legacy_plan, legacy_score = _legacy_score_boundary(script_document, plan, score)
    note_source_by_legacy_id = {
        f"{unit.source_section_id}:{note.score_note_id}": note.score_note_id
        for unit in score.score_units
        for layer in unit.score_unit_layers
        for note in layer.notes
    }
    legacy_performance = LegacyPerformanceSpec(
        performance_id=performance.performance_id,
        target_duration_ms=performance.target_duration_ms,
        default_velocity=performance.default_velocity,
        timing_budget_id=performance.timing_budget_id,
        node_performances=tuple(
            LegacyNodePerformance(
                node_id=item.section_id,
                timing_profile=item.timing_profile,
                timing_amount=item.timing_amount,
                dynamics_profile=item.dynamics_profile,
                articulation_profile=item.articulation_profile,
                coordination_profile=item.coordination_profile,
                pedal_profile=item.pedal_profile,
            )
            for item in performance.section_performances
        ),
        key_release_percent=performance.key_release_percent,
        velocity_policy_id=performance.velocity_policy_id,
    )
    legacy_rendered, pedal_source_sections = render_role_neutral_performance_with_pedal_sources(
        legacy_plan,
        legacy_score,
        legacy_performance,
    )
    unit_id_by_section = {unit.source_section_id: unit.score_unit_id for unit in score.score_units}
    notes = tuple(
        PerformedNote(
            performed_note_id=item.event_id,
            source_score_note_id=note_source_by_legacy_id[item.event_id],
            at_ms=item.at_ms,
            duration_ms=item.duration_ms,
            pitch=item.pitch,
            velocity=item.velocity,
            voice=item.voice,
        )
        for item in legacy_rendered.notes
    )
    pedals = tuple(
        PerformedPedal(
            performed_pedal_id=item.event_id,
            source_score_unit_id=unit_id_by_section[pedal_source_sections[item.event_id]],
            at_ms=item.at_ms,
            value=item.value,
        )
        for item in legacy_rendered.pedals
    )
    rendered = RenderedPerformance(
        performance_id=legacy_rendered.performance_id,
        title=legacy_rendered.title,
        duration_ms=legacy_rendered.duration_ms,
        notes=notes,
        pedals=pedals,
        section_intervals=tuple(
            SectionInterval(section_id, start_ms, end_ms)
            for section_id, start_ms, end_ms in legacy_rendered.node_intervals
        ),
        piece_plan_sha256=dataclass_content_sha256(plan),
        score_spec_sha256=dataclass_content_sha256(score),
        performance_spec_sha256=dataclass_content_sha256(performance),
    )
    validate_rendered_performance(plan, score, performance, rendered)
    return rendered


def _legacy_score_boundary(
    script_document: Mapping[str, object],
    plan: PiecePlan,
    score: ScoreSpec,
) -> tuple[LegacyPiecePlan, LegacyScoreSpec]:
    validate_score_ir(plan, score)
    script = cast(dict[str, object], script_document["script"])
    sections = cast(dict[str, dict[str, object]], script["sections"])
    placements = cast(dict[str, dict[str, object]], script["material_placements"])
    if {node.section_id for node in plan.nodes} != set(sections):
        raise ScoreRenderingError("piece plan sections do not match the script")
    layer_placement_ids = {
        layer.source_material_placement_id
        for unit in score.score_units
        for layer in unit.score_unit_layers
    }
    if layer_placement_ids != set(placements):
        raise ScoreRenderingError("score layers do not match the script material placements")
    legacy_plan = LegacyPiecePlan(
        plan.plan_id,
        plan.title,
        plan.tonal_center,
        plan.mode,
        plan.root_section_id,
        "tonic",
        tuple(
            LegacyPlanNode(
                node.section_id,
                node.parent_section_id,
                node.order,
                "whole" if node.section_id == plan.root_section_id else "statement",
                derived_from=None,
                duration_weight=(
                    int(node.duration_weight) if node.duration_weight is not None else None
                ),
                score_material_id=node.score_unit_id,
            )
            for node in plan.nodes
        ),
    )
    legacy_materials: list[LegacyScoreMaterial] = []
    for unit in score.score_units:
        legacy_notes = tuple(
            LegacyScoreNote(
                note.score_note_id,
                note.at_units,
                note.duration_units,
                note.pitch,
                note.voice,
                note.tie,
                note.articulations,
            )
            for layer in unit.score_unit_layers
            for note in layer.notes
        )
        legacy_materials.append(
            LegacyScoreMaterial(
                unit.score_unit_id,
                unit.length_units,
                legacy_notes,
                directions=tuple(
                    LegacyScoreDirection(item.direction_id, item.at_units, item.kind, item.value)
                    for item in unit.directions
                ),
                harmonies=tuple(
                    LegacyScoreHarmony(
                        item.harmony_id,
                        item.at_units,
                        item.duration_units,
                        item.root_pitch_class,
                        item.quality,
                    )
                    for item in unit.harmonies
                ),
                foreground_voice=None,
            )
        )
    legacy_score = LegacyScoreSpec(score.score_id, score.divisions, tuple(legacy_materials))
    return legacy_plan, legacy_score


def write_score_musicxml(
    script_document: Mapping[str, object],
    plan: PiecePlan,
    score: ScoreSpec,
    output_path: Path,
) -> Path:
    """Write v2 score values through the private one-way MusicXML adapter."""
    legacy_plan, legacy_score = _legacy_score_boundary(script_document, plan, score)
    return render_role_neutral_musicxml(legacy_plan, legacy_score, output_path)


def check_score_musicxml(
    script_document: Mapping[str, object],
    plan: PiecePlan,
    score: ScoreSpec,
    path: Path,
) -> dict[str, object]:
    """Reread MusicXML and compare all values preserved by that format."""
    legacy_plan, legacy_score = _legacy_score_boundary(script_document, plan, score)
    return validate_musicxml_round_trip(legacy_plan, legacy_score, path)


def write_rendered_performance_smf(rendered: RenderedPerformance, output_path: Path) -> Path:
    """Write only the values stored by the v2 rendered-performance boundary to SMF."""
    legacy_rendered = _legacy_rendered_performance(rendered)
    return render_performance_smf(legacy_rendered, output_path).path


def check_rendered_performance_smf(rendered: RenderedPerformance, path: Path) -> dict[str, object]:
    """Reread SMF and compare all values preserved by that format."""
    return validate_smf_round_trip(_legacy_rendered_performance(rendered), path)


def _legacy_rendered_performance(
    rendered: RenderedPerformance,
) -> LegacyRenderedPerformance:
    return LegacyRenderedPerformance(
        performance_id=rendered.performance_id,
        title=rendered.title,
        duration_ms=rendered.duration_ms,
        notes=tuple(
            LegacyPerformedNote(
                event_id=item.performed_note_id,
                occurrence_node_id="not-stored-in-smf",
                at_ms=item.at_ms,
                duration_ms=item.duration_ms,
                pitch=item.pitch,
                velocity=item.velocity,
                voice=item.voice,
            )
            for item in rendered.notes
        ),
        pedals=tuple(
            LegacyPerformedPedal(
                event_id=item.performed_pedal_id,
                occurrence_node_id=item.source_score_unit_id,
                at_ms=item.at_ms,
                value=item.value,
            )
            for item in rendered.pedals
        ),
        harmonies=(),
        node_intervals=tuple(
            (item.section_id, item.start_ms, item.end_ms) for item in rendered.section_intervals
        ),
        lineage=(
            rendered.piece_plan_sha256,
            rendered.score_spec_sha256,
            rendered.performance_spec_sha256,
        ),
    )
