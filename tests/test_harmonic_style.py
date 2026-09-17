from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import mido
import pytest

from llm_musical_composer.harmonic_style import (
    analyze_harmonic_style,
    build_harmonic_target,
    decide_local_repair,
    evaluate_harmonic_profile,
    extract_harmonic_profile,
    main,
    profile_smf,
)
from llm_musical_composer.smf_notes import SmfNote, maximum_polyphony


def note(pitch: int, onset_ms: int, duration_ms: int, velocity: int = 80) -> SmfNote:
    return SmfNote(pitch, onset_ms, duration_ms, velocity)


def profile(*pitches: int, duration_ms: int = 1_000) -> dict:
    return extract_harmonic_profile([note(pitch, 0, duration_ms) for pitch in pitches])


def write_midi(
    path: Path,
    *,
    pedal: bool = False,
    pitches: tuple[int, ...] = (60, 64),
    velocity: int = 80,
) -> None:
    midi = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    if pedal:
        track.append(mido.Message("control_change", control=64, value=127, time=0))
    for pitch in pitches:
        track.append(mido.Message("note_on", note=pitch, velocity=velocity, time=0))
    for index, pitch in enumerate(pitches):
        track.append(
            mido.Message("note_off", note=pitch, velocity=0, time=480 if index == 0 else 0)
        )
    if pedal:
        track.append(mido.Message("control_change", control=64, value=0, time=0))
    track.append(mido.MetaMessage("end_of_track", time=0))
    midi.save(path)


def test_single_notes_have_no_comparable_vertical_intervals() -> None:
    result = extract_harmonic_profile([note(60, 0, 500), note(64, 500, 500), note(67, 1_000, 500)])

    assert result == {
        "schema_version": 1,
        "status": "unable_to_investigate",
        "reason": "no overlapping key-held note pairs",
        "overlap_pair_ms": 0,
    }


def test_triad_is_counted_by_overlap_duration_and_interval_class() -> None:
    result = profile(60, 64, 67)

    assert result["status"] == "pass"
    assert result["overlap_pair_ms"] == 3_000
    assert result["class_pair_ms"] == {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 1_000,
        "4": 1_000,
        "5": 1_000,
        "6": 0,
    }
    assert result["distribution"] == {
        "0": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": pytest.approx(1 / 3),
        "4": pytest.approx(1 / 3),
        "5": pytest.approx(1 / 3),
        "6": 0.0,
    }


def test_transposition_time_stretch_velocity_and_input_order_are_invariant() -> None:
    original_notes = [
        note(60, 0, 1_000, 40),
        note(64, 100, 800, 90),
        note(67, 200, 500, 110),
    ]
    transposed = [
        note(item.pitch + 7, item.onset_ms, item.duration_ms, item.velocity)
        for item in original_notes
    ]
    stretched = [
        note(item.pitch, item.onset_ms * 3, item.duration_ms * 3, item.velocity)
        for item in original_notes
    ]
    velocity_changed = [
        note(item.pitch, item.onset_ms, item.duration_ms, 1 + index)
        for index, item in enumerate(original_notes)
    ]

    expected = extract_harmonic_profile(original_notes)["distribution"]
    assert extract_harmonic_profile(transposed)["distribution"] == expected
    assert extract_harmonic_profile(stretched)["distribution"] == expected
    assert extract_harmonic_profile(velocity_changed)["distribution"] == expected
    assert extract_harmonic_profile(reversed(original_notes))["distribution"] == expected


def test_pedal_events_do_not_change_key_held_profile(tmp_path: Path) -> None:
    without_pedal = tmp_path / "without.mid"
    with_pedal = tmp_path / "with.mid"
    write_midi(without_pedal, pedal=False)
    write_midi(with_pedal, pedal=True)

    first = profile_smf(without_pedal)
    second = profile_smf(with_pedal)

    assert first["profile"] == second["profile"]
    assert first["sha256"] != second["sha256"]


def test_format_two_smf_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "format-two.mid"
    midi = mido.MidiFile(type=2, ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(track)
    midi.save(path)

    with pytest.raises(ValueError, match="format 2"):
        profile_smf(path)


def test_shared_note_parser_handles_an_unknown_effective_end() -> None:
    notes = [{"onset_tick": 10, "effective_end_tick": None}]

    assert maximum_polyphony(notes, final_tick=20) == 1


@pytest.mark.parametrize(
    "invalid_note, message",
    [
        (note(-1, 0, 100), "pitch"),
        (note(60, -1, 100), "onset_ms"),
        (note(60, 0, 0), "duration_ms"),
    ],
)
def test_invalid_notes_are_rejected(invalid_note: SmfNote, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        extract_harmonic_profile([invalid_note])


def test_chord_tone_change_affects_expected_interval_classes() -> None:
    consonant = profile(60, 64, 67)
    changed = profile(60, 61, 67)

    assert changed["distribution"]["1"] > consonant["distribution"]["1"]
    assert changed["distribution"]["6"] > consonant["distribution"]["6"]
    assert changed["distribution"]["3"] < consonant["distribution"]["3"]
    assert changed["distribution"]["4"] < consonant["distribution"]["4"]


def reference_item(name: str, pitches: tuple[int, ...]) -> dict:
    return {"name": name, "sha256": name, "profile": profile(*pitches)}


def reference_profiles() -> list[dict]:
    return [
        reference_item("a.mid", (60, 64, 67)),
        reference_item("b.mid", (60, 63, 67)),
        reference_item("c.mid", (60, 65, 69)),
        reference_item("d.mid", (60, 64, 69)),
        reference_item("e.mid", (60, 62, 67)),
        reference_item("f.mid", (60, 65, 68)),
        reference_item("g.mid", (60, 63, 68)),
    ]


def test_target_keeps_each_interval_class_separate_and_is_order_invariant() -> None:
    first = build_harmonic_target(reference_profiles())
    second = build_harmonic_target(reversed(reference_profiles()))

    assert first == second
    assert first["status"] == "pass"
    assert first["reference_count"] == 7
    assert set(first["classes"]) == {str(index) for index in range(7)}
    assert [item["name"] for item in first["references"]] == [
        "a.mid",
        "b.mid",
        "c.mid",
        "d.mid",
        "e.mid",
        "f.mid",
        "g.mid",
    ]
    assert "total_loss" not in json.dumps(first)


def test_target_rejects_duplicate_unavailable_and_too_small_inputs() -> None:
    duplicate = [*reference_profiles(), reference_item("A.MID", (60, 64, 67))]
    with pytest.raises(ValueError, match="duplicate"):
        build_harmonic_target(duplicate)

    unavailable = reference_item("bad.mid", (60, 64, 67))
    unavailable["profile"] = {
        "schema_version": 1,
        "status": "unable_to_investigate",
        "reason": "no overlap",
        "overlap_pair_ms": 0,
    }
    with pytest.raises(ValueError, match="unavailable"):
        build_harmonic_target([*reference_profiles()[:6], unavailable])

    with pytest.raises(ValueError, match="at least three"):
        build_harmonic_target(reference_profiles()[:2])

    with pytest.raises(ValueError, match="name is required"):
        build_harmonic_target([{"name": "", "profile": profile(60, 64, 67)}] * 3)


def test_candidate_evaluation_reports_dimensions_without_total_loss() -> None:
    target = build_harmonic_target(reference_profiles())

    evaluation = evaluate_harmonic_profile(profile(60, 64, 67), target)

    assert evaluation["status"] == "pass"
    assert set(evaluation["classes"]) == {str(index) for index in range(7)}
    assert 0 <= evaluation["inside_class_count"] <= 7
    assert evaluation["worst_class_distance"] >= 0
    assert "total_loss" not in json.dumps(evaluation)

    with pytest.raises(ValueError, match="unavailable"):
        evaluate_harmonic_profile(
            extract_harmonic_profile([note(60, 0, 100)]),
            target,
        )


@pytest.mark.parametrize(
    "bad_profile, message",
    [
        ({"status": "pass", "distribution": {"0": 1.0}}, "interval classes"),
        (
            {
                "status": "pass",
                "distribution": {str(index): "bad" if index == 0 else 0 for index in range(7)},
            },
            "distribution is invalid",
        ),
        (
            {
                "status": "pass",
                "distribution": {
                    str(index): float("nan") if index == 0 else 0 for index in range(7)
                },
            },
            "finite and non-negative",
        ),
        (
            {
                "status": "pass",
                "distribution": {str(index): 0.1 for index in range(7)},
            },
            "sum to one",
        ),
    ],
)
def test_invalid_profile_distributions_are_rejected(bad_profile: dict, message: str) -> None:
    target = build_harmonic_target(reference_profiles())

    with pytest.raises(ValueError, match=message):
        evaluate_harmonic_profile(bad_profile, target)


def test_unavailable_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="target is unavailable"):
        evaluate_harmonic_profile(profile(60, 64, 67), {"status": "fail"})


def test_zero_iqr_target_uses_an_explicit_fallback_scale() -> None:
    identical = [reference_item(f"{index}.mid", (60, 64, 67)) for index in range(3)]
    target = build_harmonic_target(identical)

    result = evaluate_harmonic_profile(profile(60, 61, 67), target)

    assert result["worst_class_distance"] > 0


def test_nonzero_iqr_is_used_as_the_distance_scale() -> None:
    target = build_harmonic_target(reference_profiles())
    target["classes"]["1"].update(
        minimum=0.1,
        p25=0.2,
        median=0.25,
        p75=0.3,
        maximum=0.4,
        target_low=0.2,
        target_high=0.3,
    )

    result = evaluate_harmonic_profile(profile(60, 64, 67), target)

    assert result["classes"]["1"]["normalized_distance"] == 2.0


@pytest.mark.parametrize(
    "transposition_passes, outside_classes, eligible, reason_fragment",
    [
        (False, ["1"], False, "differ"),
        (True, [], False, "already inside"),
        (True, ["1", "6"], False, "multiple"),
        (True, ["1"], True, "one outside"),
    ],
)
def test_local_repair_decision_requires_one_unique_outside_class(
    transposition_passes: bool,
    outside_classes: list[str],
    eligible: bool,
    reason_fragment: str,
) -> None:
    result = decide_local_repair(
        transposition_passes=transposition_passes,
        outside_classes=outside_classes,
    )

    assert result["eligible"] is eligible
    assert result["outside_classes"] == outside_classes
    assert reason_fragment in result["reason"]


def prepare_analysis_inputs(tmp_path: Path, *, changed_v34: bool = False) -> dict[str, Path]:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    named_chords = {
        "a.mid": (60, 64, 67),
        "b.mid": (60, 63, 67),
        "c.mid": (60, 65, 69),
        "d.mid": (60, 64, 69),
        "e.mid": (60, 62, 67),
        "f.mid": (60, 65, 68),
        "g.mid": (60, 63, 68),
    }
    for name, pitches in named_chords.items():
        write_midi(source_dir / name, pitches=pitches)
    style_target = tmp_path / "style-target.json"
    style_target.write_text(
        json.dumps({"neighbors": [{"name": name} for name in named_chords]}),
        encoding="utf-8",
    )
    v26 = tmp_path / "v26.mid"
    v34 = tmp_path / "v34.mid"
    v9 = tmp_path / "v9.mid"
    write_midi(v26, pitches=(60, 64, 67))
    write_midi(v34, pitches=(61, 65, 68) if not changed_v34 else (61, 62, 68))
    write_midi(v9, pitches=(60, 61, 66))
    return {
        "source_dir": source_dir,
        "style_target": style_target,
        "v26": v26,
        "v34": v34,
        "v9": v9,
    }


def test_analysis_reads_fixed_neighbors_and_proves_real_transposition_invariance(
    tmp_path: Path,
) -> None:
    paths = prepare_analysis_inputs(tmp_path)

    result = analyze_harmonic_style(
        source_dir=paths["source_dir"],
        style_target_path=paths["style_target"],
        candidate_paths={"v26": paths["v26"], "v34": paths["v34"], "v9": paths["v9"]},
    )

    assert result["status"] == "pass"
    assert result["controls"]["v26_v34_global_transposition"]["maximum_difference"] == 0
    assert result["target"]["reference_count"] == 7
    assert result["negative_reference_semantics"].startswith("v9 is diagnostic")
    assert result["local_repair_decision"]["reason"]


def test_main_writes_analysis_and_returns_failure_for_non_transposition(tmp_path: Path) -> None:
    paths = prepare_analysis_inputs(tmp_path, changed_v34=True)
    output = tmp_path / "output" / "analysis.json"

    status = main(
        [
            "--source-dir",
            str(paths["source_dir"]),
            "--style-target",
            str(paths["style_target"]),
            "--v26",
            str(paths["v26"]),
            "--v34",
            str(paths["v34"]),
            "--v9",
            str(paths["v9"]),
            "--output",
            str(output),
        ]
    )

    assert status == 1
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["status"] == "fail"
    assert saved["local_repair_decision"]["reason"].startswith("v26 and v34 differ")


def test_module_entrypoint_runs_the_same_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = prepare_analysis_inputs(tmp_path)
    output = tmp_path / "entrypoint.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "harmonic_style",
            "--source-dir",
            str(paths["source_dir"]),
            "--style-target",
            str(paths["style_target"]),
            "--v26",
            str(paths["v26"]),
            "--v34",
            str(paths["v34"]),
            "--v9",
            str(paths["v9"]),
            "--output",
            str(output),
        ],
    )

    with (
        pytest.warns(RuntimeWarning, match="found in sys.modules"),
        pytest.raises(SystemExit) as stopped,
    ):
        runpy.run_module("llm_musical_composer.harmonic_style", run_name="__main__")

    assert stopped.value.code == 0
    assert output.exists()


@pytest.mark.parametrize(
    "style_target, candidates, message",
    [
        ({}, {"v26": "v26", "v34": "v34"}, "no fixed neighbor"),
        (
            {"neighbors": [{"name": "a.mid"}, {"name": "A.MID"}, {"name": "b.mid"}]},
            {"v26": "v26", "v34": "v34"},
            "too small or duplicated",
        ),
    ],
)
def test_analysis_rejects_invalid_neighbor_lists(
    tmp_path: Path, style_target: dict, candidates: dict, message: str
) -> None:
    target_path = tmp_path / "style.json"
    target_path.write_text(json.dumps(style_target), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        analyze_harmonic_style(
            source_dir=tmp_path,
            style_target_path=target_path,
            candidate_paths={key: tmp_path / value for key, value in candidates.items()},
        )


def test_analysis_requires_v26_and_v34_after_loading_candidates(tmp_path: Path) -> None:
    paths = prepare_analysis_inputs(tmp_path)

    with pytest.raises(ValueError, match="must include v26 and v34"):
        analyze_harmonic_style(
            source_dir=paths["source_dir"],
            style_target_path=paths["style_target"],
            candidate_paths={"v26": paths["v26"]},
        )
