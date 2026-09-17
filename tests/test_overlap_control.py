from __future__ import annotations

import math

import pytest

from llm_musical_composer.overlap_control import (
    CANDIDATE_SPECS,
    MeasuredOverlapCandidate,
    build_performance_candidate,
    build_score_candidate,
    classify_note,
    denormalize_overlap,
    resolve_measured_overlap,
)
from llm_musical_composer.performance_pipeline import (
    NodePerformance,
    PerformanceSpec,
    ScoreHarmony,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
)


def _note(
    event_id: str,
    *,
    at: int,
    duration: int,
    pitch: int,
    voice: str,
    articulations: tuple[str, ...] = (),
) -> ScoreNote:
    return ScoreNote(event_id, at, duration, pitch, voice, None, articulations)


def _material(
    *,
    material_id: str = "theme",
    root: int = 0,
    notes: tuple[ScoreNote, ...] | None = None,
) -> ScoreMaterial:
    return ScoreMaterial(
        material_id=material_id,
        length_units=16,
        notes=notes
        or (
            _note("bass", at=0, duration=2, pitch=48, voice="lower"),
            _note("third", at=0, duration=2, pitch=52, voice="lower"),
            _note("next-bass", at=8, duration=2, pitch=48, voice="lower"),
            _note("melody", at=0, duration=2, pitch=67, voice="upper"),
            _note("next-melody", at=4, duration=2, pitch=69, voice="upper"),
        ),
        harmonies=(ScoreHarmony("h", 0, 16, root, "major"),),
        foreground_voice="upper",
    )


def _performance() -> PerformanceSpec:
    return PerformanceSpec(
        performance_id="performance",
        target_duration_ms=180_000,
        default_velocity=64,
        timing_budget_id="timing",
        node_performances=(
            NodePerformance("a1", articulation_profile="score"),
            NodePerformance("a1-child", articulation_profile="light"),
            NodePerformance("b", articulation_profile=None),
            NodePerformance("a2", articulation_profile="legato"),
        ),
    )


def test_denormalize_overlap_uses_corpus_endpoints() -> None:
    assert denormalize_overlap(-1.0, minimum=0.5, maximum=3.5) == 0.5
    assert denormalize_overlap(0.0, minimum=0.5, maximum=3.5) == 2.0
    assert denormalize_overlap(1.0, minimum=0.5, maximum=3.5) == 3.5


@pytest.mark.parametrize("value", [-1.1, 1.1, math.nan, True, "0"])
def test_denormalize_overlap_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError):
        denormalize_overlap(value, minimum=0.5, maximum=3.5)


def test_classification_depends_on_harmony_and_simultaneous_context() -> None:
    c_major = _material(root=0)
    f_major = _material(root=5)
    bass = c_major.notes[0]

    c_role = classify_note(c_major, bass)
    f_role = classify_note(f_major, bass)

    assert c_role.harmony_role == "root"
    assert c_role.bass_anchor is True
    assert f_role.harmony_role == "fifth"
    assert f_role.bass_anchor is True

    raised_companion = _material(
        notes=(
            _note("lower-root", at=0, duration=2, pitch=36, voice="lower"),
            bass,
        )
    )
    assert classify_note(raised_companion, bass).bass_anchor is False


def test_shortening_is_role_ordered_and_protects_bass_anchor() -> None:
    score = ScoreSpec("score", 4, (_material(),))

    stage1 = build_score_candidate(score, direction="short", stage=1)
    stage2 = build_score_candidate(score, direction="short", stage=2)
    stage3 = build_score_candidate(score, direction="short", stage=3)

    by_id_1 = {note.event_id: note for note in stage1.score.materials[0].notes}
    by_id_2 = {note.event_id: note for note in stage2.score.materials[0].notes}
    by_id_3 = {note.event_id: note for note in stage3.score.materials[0].notes}
    assert by_id_1["third"].duration_units == 1
    assert by_id_1["bass"].duration_units == 2
    assert by_id_2["bass"].duration_units == 2
    assert by_id_3["melody"].duration_units == 1


def test_harmonic_extension_never_shortens_and_respects_limits() -> None:
    score = ScoreSpec("score", 4, (_material(),))
    candidate = build_score_candidate(score, direction="harmonic-long", stage=4)
    by_id = {note.event_id: note for note in candidate.score.materials[0].notes}

    assert by_id["bass"].duration_units == 8
    assert by_id["third"].duration_units == 16
    assert by_id["melody"].duration_units == 4
    assert all(
        after.duration_units >= before.duration_units
        for before, after in zip(
            score.materials[0].notes,
            candidate.score.materials[0].notes,
            strict=True,
        )
    )


def test_protected_material_tenuto_and_terminal_note_are_unchanged() -> None:
    protected = _material(material_id="ending")
    ordinary = _material(
        material_id="ordinary",
        notes=(
            _note("tenuto", at=0, duration=2, pitch=48, voice="lower", articulations=("tenuto",)),
            _note("terminal", at=8, duration=8, pitch=52, voice="lower"),
        ),
    )
    score = ScoreSpec("score", 4, (protected, ordinary))
    candidate = build_score_candidate(score, direction="short", stage=3)
    assert candidate.score.materials == score.materials


def test_performance_candidate_only_changes_named_parent_nodes() -> None:
    candidate = build_performance_candidate(_performance(), profile="light")
    by_id = {item.node_id: item for item in candidate.node_performances}
    assert by_id["a1"].articulation_profile == "light"
    assert by_id["b"].articulation_profile == "light"
    assert by_id["a2"].articulation_profile == "light"
    assert by_id["a1-child"].articulation_profile == "light"
    assert by_id["a1-child"] == _performance().node_performances[1]


def test_candidate_catalog_is_the_measured_18_candidates() -> None:
    assert len(CANDIDATE_SPECS) == 18
    assert len({candidate.candidate_id for candidate in CANDIDATE_SPECS}) == 18


def test_resolution_keeps_safety_separate_from_reachability() -> None:
    candidates = (
        MeasuredOverlapCandidate("far-low", 1.0, True, 20, 0),
        MeasuredOverlapCandidate("near-low", 1.5, True, 10, 1),
        MeasuredOverlapCandidate("unsafe", 3.5, False, 0, 0),
    )
    result = resolve_measured_overlap(
        1.0,
        candidates=candidates,
        minimum=0.5,
        maximum=3.5,
    )
    assert result.candidate_id == "near-low"
    assert result.status == "unreachable"
    assert result.safe_raw_range == (1.0, 1.5)
    assert result.achieved_normalized == pytest.approx(-1 / 3)


def test_resolution_uses_documented_deterministic_tie_break() -> None:
    candidates = (
        MeasuredOverlapCandidate("z", 2.0, True, 4, 0),
        MeasuredOverlapCandidate("a", 2.0, True, 2, 1),
        MeasuredOverlapCandidate("b", 2.0, True, 2, 0),
    )
    result = resolve_measured_overlap(
        0.0,
        candidates=candidates,
        minimum=1.0,
        maximum=3.0,
    )
    assert result.candidate_id == "b"
    assert result.status == "exact"
