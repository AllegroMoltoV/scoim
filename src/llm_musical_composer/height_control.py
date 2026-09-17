"""コーパス座標の高さ要求を、安全な整数半音の全体移調へ解決する。"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from llm_musical_composer.control_reference_baseline import normalize_value
from llm_musical_composer.performance_pipeline import (
    HARMONY_INTERVALS,
    PerformanceSpec,
    PiecePlan,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
)

SCHEME_ID = "height-v3-downward-register-revoicing"
PIANO_MIN_PITCH = 21
PIANO_MAX_PITCH = 108
QUANTIZATION_LIMIT_SEMITONES = 1.0


@dataclass(frozen=True)
class HeightResolution:
    """高さ要求と、現在の基準曲から安全に解決した値。"""

    requested: float
    target_mean: float
    semitones: int
    achieved_mean: float
    achieved_normalized: float
    target_error_semitones: float
    safe_mean_range: tuple[float, float]
    safe_normalized_range: tuple[float, float]
    status: str


@dataclass(frozen=True)
class HeightVariant:
    """全体移調した3段IRと、その用途。"""

    scheme_id: str
    purpose: str
    resolution: HeightResolution
    plan: PiecePlan
    score: ScoreSpec
    performance: PerformanceSpec
    diagnostics: RevoicingDiagnostics


@dataclass(frozen=True)
class RevoicingDiagnostics:
    """低域を疎開したイベント数。音符の削除は許可しない。"""

    revoiced_event_ids: tuple[str, ...]
    lifted_event_ids: tuple[str, ...]
    new_or_worsened_voice_collisions: tuple[
        tuple[str, str, int | None, int], ...
    ] = ()

    @property
    def revoiced_note_count(self) -> int:
        return len(self.revoiced_event_ids)

    @property
    def lifted_note_count(self) -> int:
        return len(self.lifted_event_ids)


@dataclass(frozen=True)
class HeightCandidate:
    """品質判定前の、実測可能な高さ候補。"""

    scheme_id: str
    semitones: int
    plan: PiecePlan
    score: ScoreSpec
    performance: PerformanceSpec
    diagnostics: RevoicingDiagnostics


def _normalized_score_for_contract(
    source: ScoreSpec, target: ScoreSpec
) -> ScoreSpec | None:
    """高さ変更を許可した欄だけを元へ戻し、残りの不変性を比較する。"""

    if len(source.materials) != len(target.materials):
        return None
    materials = []
    for source_material, target_material in zip(
        source.materials, target.materials, strict=True
    ):
        if len(source_material.notes) != len(target_material.notes) or len(
            source_material.harmonies
        ) != len(target_material.harmonies):
            return None
        materials.append(
            replace(
                target_material,
                notes=tuple(
                    replace(target_note, pitch=source_note.pitch)
                    for source_note, target_note in zip(
                        source_material.notes, target_material.notes, strict=True
                    )
                ),
                harmonies=tuple(
                    replace(
                        target_harmony,
                        root_pitch_class=source_harmony.root_pitch_class,
                    )
                    for source_harmony, target_harmony in zip(
                        source_material.harmonies,
                        target_material.harmonies,
                        strict=True,
                    )
                ),
            )
        )
    return replace(target, score_id=source.score_id, materials=tuple(materials))


def validate_height_candidate_contract(
    source_plan: PiecePlan,
    source_score: ScoreSpec,
    source_performance: PerformanceSpec,
    candidate: HeightCandidate,
) -> dict[str, object]:
    """高さ候補が許可された音高関連欄以外を変えていないか検査する。"""

    violations: list[str] = []
    if candidate.performance != source_performance:
        violations.append("PerformanceSpec")
    if len(source_plan.nodes) != len(candidate.plan.nodes):
        violations.append("PiecePlan.nodes")
    else:
        normalized_nodes = tuple(
            replace(target, harmonic_focus=source.harmonic_focus)
            for source, target in zip(
                source_plan.nodes, candidate.plan.nodes, strict=True
            )
        )
        normalized_plan = replace(
            candidate.plan,
            plan_id=source_plan.plan_id,
            tonal_center=source_plan.tonal_center,
            nodes=normalized_nodes,
        )
        if normalized_plan != source_plan:
            violations.append("PiecePlan")
        if candidate.plan.tonal_center != (
            source_plan.tonal_center + candidate.semitones
        ) % 12:
            violations.append("PiecePlan.tonal_center:value")
        for source, target in zip(
            source_plan.nodes, candidate.plan.nodes, strict=True
        ):
            expected = (
                None
                if source.harmonic_focus is None
                else (source.harmonic_focus + candidate.semitones) % 12
            )
            if target.harmonic_focus != expected:
                violations.append(
                    f"PlanNode.harmonic_focus:value:{source.node_id}"
                )

    normalized_score = _normalized_score_for_contract(source_score, candidate.score)
    if normalized_score != source_score:
        violations.append("ScoreSpec")
    revoiced = set(candidate.diagnostics.revoiced_event_ids)
    lifted = set(candidate.diagnostics.lifted_event_ids)
    if normalized_score is not None:
        for source_material, target_material in zip(
            source_score.materials, candidate.score.materials, strict=True
        ):
            for source_note, target_note in zip(
                source_material.notes, target_material.notes, strict=True
            ):
                transposed = source_note.pitch + candidate.semitones
                if target_note.event_id in revoiced:
                    allowed = _open_pitch_classes(target_material, target_note)
                    if target_note.pitch % 12 not in allowed:
                        violations.append(
                            f"ScoreNote.pitch:revoicing-value:{target_note.event_id}"
                        )
                elif target_note.event_id in lifted:
                    delta = target_note.pitch - transposed
                    if delta <= 0 or delta % 12:
                        violations.append(
                            f"ScoreNote.pitch:lift-value:{target_note.event_id}"
                        )
                elif target_note.pitch != transposed:
                    violations.append(
                        f"ScoreNote.pitch:transposition-value:{target_note.event_id}"
                    )
            for source_harmony, target_harmony in zip(
                source_material.harmonies,
                target_material.harmonies,
                strict=True,
            ):
                expected = (
                    source_harmony.root_pitch_class + candidate.semitones
                ) % 12
                if target_harmony.root_pitch_class != expected:
                    violations.append(
                        "ScoreHarmony.root_pitch_class:value:"
                        f"{target_harmony.harmony_id}"
                    )
    if any(
        not PIANO_MIN_PITCH <= note.pitch <= PIANO_MAX_PITCH
        for material in candidate.score.materials
        for note in material.notes
    ):
        violations.append("ScoreNote.pitch:piano-range")
    unique = sorted(set(violations))
    return {
        "status": "passed" if not unique else "failed",
        "violations": unique,
        "allowed_field_types": [
            "PiecePlan.plan_id",
            "PiecePlan.tonal_center",
            "PiecePlan.nodes[*].harmonic_focus",
            "ScoreSpec.score_id",
            "ScoreSpec.materials[*].notes[*].pitch",
            "ScoreSpec.materials[*].harmonies[*].root_pitch_class",
        ],
    }


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _validate_endpoints(minimum: object, maximum: object) -> tuple[float, float]:
    prepared_minimum = _finite_number(minimum, "minimum")
    prepared_maximum = _finite_number(maximum, "maximum")
    if prepared_maximum <= prepared_minimum:
        raise ValueError("maximum must be greater than minimum")
    return prepared_minimum, prepared_maximum


def denormalize_height(requested: object, *, minimum: object, maximum: object) -> float:
    """`-1.0`から`1.0`の要求をコーパス上の平均MIDI音高へ戻す。"""

    prepared = _finite_number(requested, "requested")
    if not -1.0 <= prepared <= 1.0:
        raise ValueError("requested height must be between -1.0 and 1.0")
    prepared_minimum, prepared_maximum = _validate_endpoints(minimum, maximum)
    if prepared == -1.0:
        return prepared_minimum
    if prepared == 1.0:
        return prepared_maximum
    return prepared_minimum + ((prepared + 1.0) / 2.0) * (
        prepared_maximum - prepared_minimum
    )


def _score_pitch_range(score: ScoreSpec) -> tuple[int, int]:
    pitches = [note.pitch for material in score.materials for note in material.notes]
    if not pitches:
        raise ValueError("score must contain notes")
    return min(pitches), max(pitches)


def _safe_transpositions(
    score: ScoreSpec,
    *,
    current_mean: float,
    minimum: float,
    maximum: float,
) -> tuple[int, ...]:
    minimum_pitch, maximum_pitch = _score_pitch_range(score)
    physical_minimum = PIANO_MIN_PITCH - minimum_pitch
    physical_maximum = PIANO_MAX_PITCH - maximum_pitch
    safe = tuple(
        semitones
        for semitones in range(physical_minimum, physical_maximum + 1)
        if minimum <= current_mean + semitones <= maximum
    )
    if not safe:
        raise ValueError("score has no safe transposition inside the corpus height range")
    return safe


def candidate_transpositions(score: ScoreSpec) -> tuple[int, ...]:
    """ピアノの物理音域内に収まる全整数半音を返す。"""

    minimum_pitch, maximum_pitch = _score_pitch_range(score)
    return tuple(
        range(
            PIANO_MIN_PITCH - minimum_pitch,
            PIANO_MAX_PITCH - maximum_pitch + 1,
        )
    )


def resolve_measured_height(
    requested: object,
    *,
    candidate_means: dict[int, float],
    minimum: object,
    maximum: object,
) -> HeightResolution:
    """品質を通過した候補の実測平均から、要求に最も近い候補を選ぶ。"""

    prepared_minimum, prepared_maximum = _validate_endpoints(minimum, maximum)
    target = denormalize_height(
        requested, minimum=prepared_minimum, maximum=prepared_maximum
    )
    safe = {
        semitones: _finite_number(mean, f"candidate_means[{semitones}]")
        for semitones, mean in candidate_means.items()
        if prepared_minimum <= float(mean) <= prepared_maximum
    }
    if not safe:
        raise ValueError("no measured height candidate is inside the corpus range")
    semitones, achieved = min(
        safe.items(),
        key=lambda item: (abs(item[1] - target), abs(item[0]), item[0]),
    )
    error = abs(achieved - target)
    status = (
        "exact"
        if math.isclose(error, 0.0, rel_tol=0.0, abs_tol=1e-12)
        else "quantized"
        if error <= QUANTIZATION_LIMIT_SEMITONES
        else "unreachable"
    )
    safe_means = (min(safe.values()), max(safe.values()))
    return HeightResolution(
        requested=float(requested),
        target_mean=target,
        semitones=semitones,
        achieved_mean=achieved,
        achieved_normalized=normalize_value(
            achieved, minimum=prepared_minimum, maximum=prepared_maximum
        ),
        target_error_semitones=error,
        safe_mean_range=safe_means,
        safe_normalized_range=tuple(
            normalize_value(
                value, minimum=prepared_minimum, maximum=prepared_maximum
            )
            for value in safe_means
        ),
        status=status,
    )


def resolve_height(
    requested: object,
    *,
    score: ScoreSpec,
    current_mean: object,
    minimum: object,
    maximum: object,
) -> HeightResolution:
    """高さ要求を、コーパス範囲内の最も近い安全な整数半音へ解決する。"""

    prepared_current = _finite_number(current_mean, "current_mean")
    prepared_minimum, prepared_maximum = _validate_endpoints(minimum, maximum)
    target = denormalize_height(
        requested,
        minimum=prepared_minimum,
        maximum=prepared_maximum,
    )
    safe = _safe_transpositions(
        score,
        current_mean=prepared_current,
        minimum=prepared_minimum,
        maximum=prepared_maximum,
    )
    semitones = min(
        safe,
        key=lambda item: (abs(prepared_current + item - target), abs(item), item),
    )
    achieved = prepared_current + semitones
    error = abs(achieved - target)
    if math.isclose(error, 0.0, rel_tol=0.0, abs_tol=1e-12):
        status = "exact"
    elif error <= QUANTIZATION_LIMIT_SEMITONES:
        status = "quantized"
    else:
        status = "unreachable"
    safe_means = (prepared_current + safe[0], prepared_current + safe[-1])
    return HeightResolution(
        requested=float(requested),
        target_mean=target,
        semitones=semitones,
        achieved_mean=achieved,
        achieved_normalized=normalize_value(
            achieved,
            minimum=prepared_minimum,
            maximum=prepared_maximum,
        ),
        target_error_semitones=error,
        safe_mean_range=safe_means,
        safe_normalized_range=tuple(
            normalize_value(
                value,
                minimum=prepared_minimum,
                maximum=prepared_maximum,
            )
            for value in safe_means
        ),
        status=status,
    )


def _transpose_material(material: ScoreMaterial, semitones: int) -> ScoreMaterial:
    return replace(
        material,
        notes=tuple(
            replace(note, pitch=note.pitch + semitones) for note in material.notes
        ),
        harmonies=tuple(
            replace(
                harmony,
                root_pitch_class=(harmony.root_pitch_class + semitones) % 12,
            )
            for harmony in material.harmonies
        ),
    )


def _overlaps(left: ScoreNote, right: ScoreNote) -> bool:
    return (
        left.at_units < right.at_units + right.duration_units
        and right.at_units < left.at_units + left.duration_units
    )


def _cross_voice_gaps(material: ScoreMaterial) -> dict[tuple[str, str], int]:
    return {
        (lower.event_id, upper.event_id): upper.pitch - lower.pitch
        for lower in material.notes
        if lower.voice == "lower"
        for upper in material.notes
        if upper.voice == "upper" and _overlaps(lower, upper)
    }


def _new_or_worsened_voice_collisions(
    base: ScoreMaterial, candidate: ScoreMaterial
) -> tuple[tuple[str, str, int | None, int], ...]:
    """新規交差と、変換前より狭くなった既存声部対を返す。"""

    base_gaps = _cross_voice_gaps(base)
    failures = []
    for pair, candidate_gap in _cross_voice_gaps(candidate).items():
        base_gap = base_gaps.get(pair)
        new_collision = base_gap is None and candidate_gap <= 0
        healthy_pair_collided = (
            base_gap is not None and base_gap > 0 and candidate_gap <= 0
        )
        existing_collision_worsened = (
            base_gap is not None and base_gap <= 0 and candidate_gap < base_gap
        )
        if new_collision or healthy_pair_collided or existing_collision_worsened:
            failures.append((*pair, base_gap, candidate_gap))
    return tuple(sorted(failures))


def _validate_ending_material(material: ScoreMaterial) -> None:
    """和声宣言のないendingを再配置せず検査する。"""

    gaps = _cross_voice_gaps(material)
    collisions = [pair for pair, gap in gaps.items() if gap <= 0]
    if collisions:
        raise ValueError(
            "ending voice collision: "
            + ", ".join(f"{lower}/{upper}" for lower, upper in collisions)
        )
    notes = tuple(note for note in material.notes if note.voice == "lower")
    for onset in sorted({note.at_units for note in notes}):
        sounding = sorted(
            (
                note
                for note in notes
                if note.at_units <= onset < note.at_units + note.duration_units
            ),
            key=lambda note: note.pitch,
        )
        if (
            len(sounding) >= 2
            and sounding[0].pitch < 48
            and sounding[1].pitch - sounding[0].pitch < 7
        ):
            raise ValueError(
                "ending low-register spacing: "
                f"{sounding[0].event_id}/{sounding[1].event_id}"
            )


def _active_harmonies(material: ScoreMaterial, note: ScoreNote):
    return tuple(
        harmony
        for harmony in material.harmonies
        if harmony.at_units < note.at_units + note.duration_units
        and note.at_units < harmony.at_units + harmony.duration_units
    )


def _harmony_role(material: ScoreMaterial, note: ScoreNote, pitch: int) -> str:
    harmonies = _active_harmonies(material, note)
    if not harmonies:
        return "unknown"
    intervals = {(pitch - harmony.root_pitch_class) % 12 for harmony in harmonies}
    if intervals == {0}:
        return "root"
    if intervals == {7}:
        return "perfect_fifth"
    return "color"


def _valid_pitch(
    note: ScoreNote,
    target: int,
    notes: tuple[ScoreNote, ...],
    pitches: dict[str, int],
) -> bool:
    for other in notes:
        if other.event_id == note.event_id or not _overlaps(note, other):
            continue
        other_pitch = pitches[other.event_id]
        if target == other_pitch:
            return False
        if note.voice == "lower" and other.voice == "upper" and target >= other_pitch:
            return False
        if note.voice == "upper" and other.voice == "lower" and target <= other_pitch:
            return False
    return True


def _open_pitch_classes(material: ScoreMaterial, note: ScoreNote) -> set[int]:
    allowed: set[int] | None = None
    for harmony in _active_harmonies(material, note):
        current = {harmony.root_pitch_class}
        if 7 in HARMONY_INTERVALS[harmony.quality]:
            current.add((harmony.root_pitch_class + 7) % 12)
        allowed = current if allowed is None else allowed & current
    return set() if allowed is None else allowed


def _nearest_open_pitch(
    material: ScoreMaterial,
    note: ScoreNote,
    notes: tuple[ScoreNote, ...],
    pitches: dict[str, int],
) -> int | None:
    current = pitches[note.event_id]
    allowed = _open_pitch_classes(material, note)
    candidates = sorted(
        (
            pitch
            for pitch in range(PIANO_MIN_PITCH, PIANO_MAX_PITCH + 1)
            if pitch != current and pitch % 12 in allowed
        ),
        key=lambda pitch: (abs(pitch - current), pitch),
    )
    return next(
        (pitch for pitch in candidates if _valid_pitch(note, pitch, notes, pitches)),
        None,
    )


def _octave_lift(
    note: ScoreNote,
    notes: tuple[ScoreNote, ...],
    pitches: dict[str, int],
    *,
    minimum: int,
) -> int | None:
    current = pitches[note.event_id]
    for target in range(current + 12, PIANO_MAX_PITCH + 1, 12):
        if target >= minimum and _valid_pitch(note, target, notes, pitches):
            return target
    return None


def _sounding_at(
    notes: tuple[ScoreNote, ...], pitches: dict[str, int], onset: int
) -> list[ScoreNote]:
    return sorted(
        (
            note
            for note in notes
            if note.at_units <= onset < note.at_units + note.duration_units
        ),
        key=lambda note: pitches[note.event_id],
    )


def _repair_harmonic_material(
    material: ScoreMaterial,
) -> tuple[ScoreMaterial, set[str], set[str]]:
    accompaniment = "lower" if material.foreground_voice == "upper" else "upper"
    accompaniment_notes = tuple(
        note for note in material.notes if note.voice == accompaniment
    )
    pitches = {note.event_id: note.pitch for note in material.notes}
    revoiced: set[str] = set()
    lifted: set[str] = set()

    for note in accompaniment_notes:
        pitch = pitches[note.event_id]
        if pitch >= 48 or _harmony_role(material, note, pitch) != "color":
            continue
        target = _nearest_open_pitch(material, note, material.notes, pitches)
        if target is None:
            raise ValueError(f"low color tone cannot be revoiced: {note.event_id}")
        pitches[note.event_id] = target
        revoiced.add(note.event_id)

    for _ in range(len(material.notes) * 4 + 1):
        violation: tuple[ScoreNote, ScoreNote] | None = None
        for onset in sorted({note.at_units for note in accompaniment_notes}):
            sounding = _sounding_at(accompaniment_notes, pitches, onset)
            if (
                len(sounding) >= 2
                and pitches[sounding[0].event_id] < 48
                and pitches[sounding[1].event_id] - pitches[sounding[0].event_id] < 7
            ):
                violation = (sounding[0], sounding[1])
                break
        if violation is None:
            break
        role_priority = {"color": 2, "perfect_fifth": 1, "root": 0, "unknown": 3}
        ordered = sorted(
            violation,
            key=lambda item: (
                role_priority[_harmony_role(material, item, pitches[item.event_id])],
                pitches[item.event_id],
            ),
            reverse=True,
        )
        for note in ordered:
            target = _octave_lift(note, material.notes, pitches, minimum=PIANO_MIN_PITCH)
            if target is not None:
                pitches[note.event_id] = target
                lifted.add(note.event_id)
                break
        else:
            raise ValueError(
                "low-register spacing cannot be repaired: "
                + ", ".join(note.event_id for note in violation)
            )
    else:
        raise ValueError("low-register spacing repair did not converge")

    return (
        replace(
            material,
            notes=tuple(
                replace(note, pitch=pitches[note.event_id]) for note in material.notes
            ),
        ),
        revoiced,
        lifted,
    )


def _open_final_tonic(
    material: ScoreMaterial,
) -> tuple[ScoreMaterial, set[str]]:
    pitches = {note.event_id: note.pitch for note in material.notes}
    lifted: set[str] = set()
    for note in material.notes:
        if note.voice != "upper" or pitches[note.event_id] >= 48:
            continue
        target = _octave_lift(note, material.notes, pitches, minimum=48)
        if target is None:
            raise ValueError(f"final tonic cannot be opened: {note.event_id}")
        pitches[note.event_id] = target
        lifted.add(note.event_id)
    return (
        replace(
            material,
            notes=tuple(
                replace(note, pitch=pitches[note.event_id]) for note in material.notes
            ),
        ),
        lifted,
    )


def build_height_candidate(
    plan: PiecePlan,
    score: ScoreSpec,
    performance: PerformanceSpec,
    *,
    semitones: int,
) -> HeightCandidate:
    """全体移調後、低域の色音を根音・完全五度へ疎開する。"""

    if isinstance(semitones, bool) or not isinstance(semitones, int):
        raise ValueError("semitones must be an integer")
    minimum_pitch, maximum_pitch = _score_pitch_range(score)
    if (
        minimum_pitch + semitones < PIANO_MIN_PITCH
        or maximum_pitch + semitones > PIANO_MAX_PITCH
    ):
        raise ValueError("transposed score is outside the supported piano range")
    suffix = f"{'p' if semitones >= 0 else 'm'}{abs(semitones)}"
    prepared_plan = replace(
        plan,
        plan_id=f"{plan.plan_id}-height-{suffix}",
        tonal_center=(plan.tonal_center + semitones) % 12,
        nodes=tuple(
            replace(
                node,
                harmonic_focus=(
                    None
                    if node.harmonic_focus is None
                    else (node.harmonic_focus + semitones) % 12
                ),
            )
            for node in plan.nodes
        ),
    )
    materials: list[ScoreMaterial] = []
    revoiced: set[str] = set()
    lifted: set[str] = set()
    voice_collisions: list[tuple[str, str, int | None, int]] = []
    for source in score.materials:
        transposed = _transpose_material(source, semitones)
        material = transposed
        if semitones < 0 and material.harmonies and material.foreground_voice:
            material, changed, opened = _repair_harmonic_material(material)
            revoiced.update(changed)
            lifted.update(opened)
        elif semitones < 0 and material.material_id == "final-tonic":
            material, opened = _open_final_tonic(material)
            lifted.update(opened)
        elif semitones < 0 and material.material_id == "ending":
            _validate_ending_material(material)
        worsened = _new_or_worsened_voice_collisions(transposed, material)
        if worsened:
            voice_collisions.extend(worsened)
        materials.append(material)
    if voice_collisions:
        raise ValueError(
            "new or worsened voice collision: "
            + ", ".join(f"{lower}/{upper}" for lower, upper, _, _ in voice_collisions)
        )
    return HeightCandidate(
        scheme_id=SCHEME_ID,
        semitones=semitones,
        plan=prepared_plan,
        score=replace(
            score,
            score_id=f"{score.score_id}-height-{suffix}",
            materials=tuple(materials),
        ),
        performance=performance,
        diagnostics=RevoicingDiagnostics(
            revoiced_event_ids=tuple(sorted(revoiced)),
            lifted_event_ids=tuple(sorted(lifted)),
            new_or_worsened_voice_collisions=tuple(sorted(voice_collisions)),
        ),
    )


def build_height_variant(
    plan: PiecePlan,
    score: ScoreSpec,
    performance: PerformanceSpec,
    resolution: HeightResolution,
    *,
    purpose: str,
) -> HeightVariant:
    """解決済みの整数半音を3段IRへ一貫して適用する。"""

    if purpose not in {"candidate", "diagnostic"}:
        raise ValueError("purpose must be candidate or diagnostic")
    if resolution.status == "unreachable" and purpose != "diagnostic":
        raise ValueError("an unreachable height requires diagnostic purpose")
    if resolution.status != "unreachable" and purpose == "diagnostic":
        raise ValueError("a reachable height must use candidate purpose")
    semitones = resolution.semitones
    suffix = f"{'p' if semitones >= 0 else 'm'}{abs(semitones)}"
    prepared_plan = replace(
        plan,
        plan_id=f"{plan.plan_id}-height-{suffix}",
        tonal_center=(plan.tonal_center + semitones) % 12,
        nodes=tuple(
            replace(
                node,
                harmonic_focus=(
                    None
                    if node.harmonic_focus is None
                    else (node.harmonic_focus + semitones) % 12
                ),
            )
            for node in plan.nodes
        ),
    )
    prepared_score = replace(
        score,
        score_id=f"{score.score_id}-height-{suffix}",
        materials=tuple(_transpose_material(item, semitones) for item in score.materials),
    )
    minimum_pitch, maximum_pitch = _score_pitch_range(prepared_score)
    if minimum_pitch < PIANO_MIN_PITCH or maximum_pitch > PIANO_MAX_PITCH:
        raise ValueError("transposed score is outside the supported piano range")
    return HeightVariant(
        scheme_id=SCHEME_ID,
        purpose="diagnostic_only" if purpose == "diagnostic" else "candidate",
        resolution=resolution,
        plan=prepared_plan,
        score=prepared_score,
        performance=performance,
        diagnostics=RevoicingDiagnostics((), ()),
    )
