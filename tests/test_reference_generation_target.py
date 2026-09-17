import json
from pathlib import Path

import pytest

from llm_musical_composer.performance_pipeline import (
    PerformedNote,
    PerformedPedal,
    RenderedPerformance,
)
from llm_musical_composer.reference_generation_target import (
    ReferenceGenerationTargetError,
    build_reference_generation_target,
)
from llm_musical_composer.reference_profile import ReferencePiece, extract_reference_profile

PROJECT_ROOT = Path(__file__).parents[1]
REFERENCE_DIR = PROJECT_ROOT / ".appendix" / "reference-profile-v1"
CONTROL_DIR = PROJECT_ROOT / ".appendix" / "control-reference-baseline-v3"


def _request(*, controls: dict[str, dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "schema_version": 1,
        "preset": "solo_piano_3m_v1",
        "reference": {
            "state": "resolved",
            "name": "SonataForHouseMoving.mid",
            "sha256": "d0a3cde6718632e555488b5e23ed1294ab56465d6f6b76ac19657733caacebbe",
            "selection_method": "automatic_medoid",
        },
        "controls": controls or {},
    }


def test_real_artifacts_build_anonymous_stage_targets() -> None:
    target = build_reference_generation_target(
        _request(), reference_dir=REFERENCE_DIR, control_dir=CONTROL_DIR
    )

    assert target.artifact["reference"]["name"] == "SonataForHouseMoving.mid"
    assert set(target.artifact["controls"]) == {"brightness", "height", "attack_frequency"}
    assert all(item["source"] == "reference" for item in target.artifact["controls"].values())
    assert target.prompt_target["piece_plan"] == {
        "reference_specific_targets": [],
        "long_term_structure": {"status": "unverified"},
    }
    assert set(target.prompt_target["stage_targets"]) == {
        "score_spec",
        "performance_spec",
        "rendered_surface",
    }
    assert target.prompt_target["target_version"] == "reference-generation-target-v7"
    assert set(target.prompt_target["semantic_targets"]) == {
        "piece_plan",
        "score_spec",
        "performance_spec",
        "rendered_surface",
    }

    semantic = {
        item["id"]: item
        for stage in target.prompt_target["semantic_targets"].values()
        for item in stage
    }
    assert set(semantic) == {
        "tonal_hierarchy",
        "attack_texture",
        "attack_frequency",
        "key_held_texture",
        "foreground_accompaniment_coordination",
        "coordination_preservation",
        "register_envelope",
        "velocity_shape",
    }
    register = semantic["register_envelope"]
    assert register["policy_id"] == "melody-containing-reference-span-v2"
    assert register["target_span_semitones"] <= register["maximum_span_semitones"] <= 87
    assert len(register["relative_register_diagnostic"]["neighborhood_center"]) == 7
    velocity = semantic["velocity_shape"]
    assert len(velocity["neighborhood_center"]) == 8
    assert [item["id"] for item in velocity["bins"]][-1] == "112-127"
    attack = semantic["attack_texture"]
    assert attack["generation_stage"] == "texture_collection"
    assert attack["grouping"] == {
        "anchor": "first_note_on",
        "chain": False,
        "tolerance_ms": 30,
    }
    assert [item["id"] for item in attack["bins"]] == [
        "one",
        "two",
        "three",
        "four_or_more",
    ]
    assert attack["anchor"] == pytest.approx(
        [0.47853535, 0.19318182, 0.11742424, 0.21085858]
    )
    assert attack["per_25_groups"] == [12, 5, 3, 5]
    assert attack["notes_per_attack"]["anchor"] == pytest.approx(2.20075758)
    assert attack["maximum_group_size"] == {"anchor": 7}
    assert attack["source_observation_ids"] == [
        "control_baseline.diagnostics.overlap.attack_size_distribution",
        "control_baseline.diagnostics.overlap.notes_per_attack",
    ]

    held = semantic["key_held_texture"]
    assert [item["id"] for item in held["bins"]] == [
        "silent",
        "one",
        "two",
        "three",
        "four_or_more",
    ]
    assert held["anchor"] == pytest.approx(
        [0.29208602, 0.17815268, 0.13012866, 0.14663429, 0.25299836]
    )
    assert held["includes_pedal_extension"] is False
    assert semantic["foreground_accompaniment_coordination"]["kind"] == "quality_rule"
    assert semantic["coordination_preservation"]["preserve_rolled_45ms"] is True
    assert semantic["tonal_hierarchy"] == {
        "id": "tonal_hierarchy",
        "kind": "unverified_control",
        "generation_stage": "piece_plan_score_and_melody",
        "status": "unverified_continuous_reference",
        "requested_normalized": target.artifact["controls"]["brightness"]["value"],
        "reachability": {"status": "unverified"},
    }


@pytest.mark.parametrize(
    ("requested", "mode", "scale_policy", "ending_quality", "acceptable_range"),
    [
        (-1, "minor", "minor_functional", "minor", [-1.0, -0.5]),
        (0, "minor", "dorian", "minor", [-0.5, 0.5]),
        (1, "major", "major_functional", "major", [0.5, 1.0]),
    ],
)
def test_explicit_brightness_builds_anonymous_tonal_hierarchy(
    requested: int,
    mode: str,
    scale_policy: str,
    ending_quality: str,
    acceptable_range: list[float],
) -> None:
    target = build_reference_generation_target(
        _request(controls={"brightness": {"label": "あかるさ", "value": requested}}),
        reference_dir=REFERENCE_DIR,
        control_dir=CONTROL_DIR,
    )

    tonal = target.prompt_target["semantic_targets"]["piece_plan"][0]
    assert tonal["id"] == "tonal_hierarchy"
    assert tonal["kind"] == "quality_rule"
    assert tonal["status"] == "specified"
    assert tonal["requested"] == requested
    assert tonal["target"] == {
        "raw": pytest.approx(
            {-1: -0.35847272, 0: 0.0423874, 1: 0.44324752}[requested]
        ),
        "normalized": float(requested),
        "acceptable_normalized_range": acceptable_range,
    }
    assert tonal["tonic_policy"] == "piece_plan_relative"
    assert tonal["plan_mode"] == mode
    assert tonal["scale_policy"] == scale_policy
    assert tonal["ending"] == {"root_degree": 0, "quality": ending_quality}
    assert tonal["reachability"] == {"status": "unverified"}
    serialized = json.dumps(tonal, ensure_ascii=False, sort_keys=True)
    assert "SonataForHouseMoving.mid" not in serialized
    assert _request()["reference"]["sha256"] not in serialized
    assert "tonic_pitch_class" not in serialized


def test_attack_frequency_semantic_target_uses_30ms_control_neighborhood() -> None:
    request = _request()
    request["reference"] = {
        "state": "resolved",
        "name": "mayodance.mid",
        "sha256": "d902a9c12372cca53fceb2ac1f487201adb3a826695aecd1d9b9248f72ba3877",
        "selection_method": "supported_pool_seed_v1",
    }

    target = build_reference_generation_target(
        request, reference_dir=REFERENCE_DIR, control_dir=CONTROL_DIR
    )
    frequency = next(
        item
        for item in target.prompt_target["semantic_targets"]["score_spec"]
        if item["id"] == "attack_frequency"
    )

    assert frequency["grouping"] == {
        "anchor": "first_note_on",
        "chain": False,
        "tolerance_ms": 30,
    }
    assert frequency["anchor"] == {
        "normalized": pytest.approx(0.09276148),
        "raw_groups_per_second": pytest.approx(4.5357329),
    }
    assert frequency["neighborhood"]["raw_groups_per_second"] == {
        "center": pytest.approx(4.76504422),
        "minimum": pytest.approx(3.60069793),
        "maximum": pytest.approx(5.87968569),
    }
    assert frequency["groups_for_180_seconds"] == {
        "anchor_half_up": 816,
        "minimum_ceil": 649,
        "maximum_floor": 1058,
    }
    assert frequency["strict_score_budget"] == {
        "groups": 649,
        "source": "neighborhood_minimum_ceil",
    }
    assert frequency["promotion_range"] == {
        "minimum_groups_per_second": pytest.approx(3.60069793),
        "maximum_groups_per_second": pytest.approx(5.87968569),
        "inclusive": True,
    }

    serialized = json.dumps(target.prompt_target, ensure_ascii=False, sort_keys=True)
    assert "SonataForHouseMoving.mid" not in serialized
    assert "challenge.mid" not in serialized
    assert _request()["reference"]["sha256"] not in serialized
    assert "copy_fingerprint" not in serialized

    descriptors = [
        descriptor
        for stage in target.prompt_target["stage_targets"].values()
        for descriptor in stage
    ]
    assert len(descriptors) == 17
    assert all(item["reachability"] == {"status": "unverified"} for item in descriptors)
    for descriptor in descriptors:
        if descriptor["kind"] == "distribution":
            assert sum(descriptor["anchor"]) == pytest.approx(1.0)
            assert sum(descriptor["neighborhood_center"]) == pytest.approx(1.0)


def test_explicit_attack_frequency_zero_builds_strict_middle_score_budget() -> None:
    target = build_reference_generation_target(
        _request(
            controls={
                "attack_frequency": {"label": "発音頻度", "value": 0.0}
            }
        ),
        reference_dir=REFERENCE_DIR,
        control_dir=CONTROL_DIR,
    )
    frequency = next(
        item
        for item in target.prompt_target["semantic_targets"]["score_spec"]
        if item["id"] == "attack_frequency"
    )

    assert target.prompt_target["target_version"] == "reference-generation-target-v7"
    assert frequency["anchor"] == {
        "normalized": 0.0,
        "raw_groups_per_second": 4.17764351,
    }
    assert frequency["strict_score_budget"] == {
        "groups": 752,
        "source": "explicit_corpus_min_max_half_up",
    }
    assert frequency["resolution"] == {
        "source": "explicit_corpus_min_max_v1",
        "duration_seconds": 180,
        "axis_minimum_groups_per_second": 0.31731929,
        "axis_maximum_groups_per_second": 8.03796774,
        "rounding": "half_up_before_canonical_round",
    }
    assert frequency["promotion_range"] == {
        "minimum_groups_per_second": pytest.approx(751.5 / 180),
        "maximum_groups_per_second": pytest.approx(752.5 / 180),
        "inclusive": True,
    }
def test_explicit_zero_is_not_the_same_as_omitting_a_control() -> None:
    omitted = build_reference_generation_target(
        _request(), reference_dir=REFERENCE_DIR, control_dir=CONTROL_DIR
    )
    explicit = build_reference_generation_target(
        _request(controls={"height": {"label": "高さ", "value": 0.0}}),
        reference_dir=REFERENCE_DIR,
        control_dir=CONTROL_DIR,
    )

    assert omitted.artifact["controls"]["height"]["source"] == "reference"
    assert omitted.artifact["controls"]["height"]["value"] != 0.0
    assert explicit.artifact["controls"]["height"] == {
        "label": "高さ",
        "source": "explicit",
        "value": 0.0,
    }


def test_public_overlap_request_is_rejected() -> None:
    with pytest.raises(ReferenceGenerationTargetError, match="unknown control"):
        build_reference_generation_target(
            _request(controls={"overlap": {"label": "重なり", "value": 0.0}}),
            reference_dir=REFERENCE_DIR,
            control_dir=CONTROL_DIR,
        )


def test_resolved_reference_hash_must_match_validated_source() -> None:
    request = _request()
    request["reference"]["sha256"] = "0" * 64

    with pytest.raises(ReferenceGenerationTargetError, match="source hash mismatch"):
        build_reference_generation_target(
            request, reference_dir=REFERENCE_DIR, control_dir=CONTROL_DIR
        )


def test_manifest_hash_mismatch_stops_before_target_creation(tmp_path: Path) -> None:
    reference_dir = tmp_path / "reference"
    control_dir = tmp_path / "control"
    reference_dir.mkdir()
    control_dir.mkdir()
    for name in ("files.jsonl", "summary.json", "manifest.json"):
        (reference_dir / name).write_bytes((REFERENCE_DIR / name).read_bytes())
    for name in ("records.jsonl", "summary.json", "manifest.json"):
        (control_dir / name).write_bytes((CONTROL_DIR / name).read_bytes())
    with (control_dir / "records.jsonl").open("ab") as output:
        output.write(b"\n")

    with pytest.raises(ReferenceGenerationTargetError, match="hash mismatch"):
        build_reference_generation_target(
            _request(), reference_dir=reference_dir, control_dir=control_dir
        )


def _piece_from_rendered(rendered: RenderedPerformance) -> ReferencePiece:
    return ReferencePiece.from_dicts(
        name=rendered.title,
        notes=(
            {
                "pitch": note.pitch,
                "onset_ms": note.at_ms,
                "duration_ms": note.duration_ms,
                "velocity": note.velocity,
            }
            for note in rendered.notes
        ),
        pedals=({"at_ms": pedal.at_ms, "value": pedal.value} for pedal in rendered.pedals),
    )


def _rendered(
    *,
    pitches: tuple[int, ...] = (48, 60, 50, 62, 52, 64),
    onsets: tuple[int, ...] = (0, 0, 500, 500, 1000, 1000),
    durations: tuple[int, ...] = (450, 450, 450, 450, 900, 900),
    velocities: tuple[int, ...] = (60, 72, 62, 74, 64, 76),
    pedals: tuple[tuple[int, int], ...] = ((0, 127), (1300, 0)),
) -> RenderedPerformance:
    notes = tuple(
        PerformedNote(
            event_id=f"n{index}",
            occurrence_node_id="a",
            at_ms=onsets[index],
            duration_ms=durations[index],
            pitch=pitches[index],
            velocity=velocities[index],
            voice="lower" if index % 2 == 0 else "upper",
        )
        for index in range(len(pitches))
    )
    performed_pedals = tuple(
        PerformedPedal(
            event_id=f"p{index}", occurrence_node_id="a", at_ms=at_ms, value=value
        )
        for index, (at_ms, value) in enumerate(pedals)
    )
    return RenderedPerformance(
        performance_id="fixture",
        title="fixture",
        duration_ms=max(note.at_ms + note.duration_ms for note in notes),
        notes=notes,
        pedals=performed_pedals,
        harmonies=(),
        node_intervals=(("a", 0, 2000),),
        lineage=("plan", "score", "performance"),
    )


def test_actual_rendered_ir_separates_score_performance_and_coupled_observations() -> None:
    base = extract_reference_profile(_piece_from_rendered(_rendered()))["feature_groups"]
    score_changed = extract_reference_profile(
        _piece_from_rendered(_rendered(pitches=(48, 60, 55, 67, 52, 64)))
    )["feature_groups"]
    performance_changed = extract_reference_profile(
        _piece_from_rendered(
            _rendered(
                velocities=(28, 40, 32, 44, 36, 48),
                pedals=((0, 0), (1300, 0)),
            )
        )
    )["feature_groups"]
    coupled_changed = extract_reference_profile(
        _piece_from_rendered(
            _rendered(
                onsets=(0, 0, 250, 250, 1000, 1000),
                durations=(180, 180, 600, 600, 300, 300),
            )
        )
    )["feature_groups"]

    assert score_changed["pitch_harmony"] != base["pitch_harmony"]
    assert performance_changed["pitch_harmony"] == base["pitch_harmony"]
    assert performance_changed["performance_texture"] != base["performance_texture"]
    assert coupled_changed["rhythm_time"] != base["rhythm_time"]
    assert coupled_changed["performance_texture"]["metrics"]["polyphony_duration"] != (
        base["performance_texture"]["metrics"]["polyphony_duration"]
    )
