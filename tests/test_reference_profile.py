import copy
from pathlib import Path

import mido
import pytest

from llm_musical_composer.reference_profile import (
    ReferencePiece,
    ReferenceProfileError,
    build_copy_fingerprint,
    copy_fingerprint_similarity,
    evaluate_copy_risk,
    extract_reference_profile,
    load_reference_piece,
    profile_distances,
    select_default_reference,
    select_local_neighborhood,
)


def _piece(name: str = "base.mid", transpose: int = 0) -> ReferencePiece:
    notes = []
    attacks = [
        (0, (48, 60), 80, 600),
        (700, (52,), 84, 280),
        (1100, (55, 64), 88, 500),
        (1800, (57,), 92, 350),
        (2300, (59, 67, 71), 100, 700),
        (3300, (55,), 86, 250),
        (3700, (52, 64), 82, 600),
        (4600, (48, 60), 76, 900),
    ]
    for onset, pitches, velocity, duration in attacks:
        for pitch in pitches:
            notes.append(
                {
                    "pitch": pitch + transpose,
                    "onset_ms": onset,
                    "duration_ms": duration,
                    "velocity": velocity,
                }
            )
    return ReferencePiece.from_dicts(
        name=name,
        notes=notes,
        pedals=[
            {"at_ms": 0, "value": 127},
            {"at_ms": 2100, "value": 0},
            {"at_ms": 2250, "value": 127},
            {"at_ms": 5400, "value": 0},
        ],
    )


def test_profile_is_invariant_to_transposition_time_stretch_and_input_order() -> None:
    from llm_musical_composer.reference_controls import (
        reorder_piece,
        stretch_piece,
        transpose_piece,
    )

    base = _piece()
    expected = extract_reference_profile(base)

    assert (
        extract_reference_profile(transpose_piece(base, 5))["feature_groups"]
        == expected["feature_groups"]
    )
    assert (
        extract_reference_profile(stretch_piece(base, 1.5))["feature_groups"]
        == expected["feature_groups"]
    )
    assert extract_reference_profile(reorder_piece(base)) == expected


def test_targeted_destructions_do_not_collapse_independent_groups() -> None:
    from llm_musical_composer.reference_controls import (
        destroy_pedal,
        destroy_pitch_order,
        destroy_timing,
        flatten_velocity,
    )

    base = _piece()
    profile = extract_reference_profile(base)

    pitch = profile_distances(profile, extract_reference_profile(destroy_pitch_order(base)))
    assert pitch["pitch_harmony"] > 0
    assert pitch["rhythm_time"] == 0

    timing = profile_distances(profile, extract_reference_profile(destroy_timing(base)))
    assert timing["rhythm_time"] > 0
    assert timing["pitch_harmony"] == 0

    velocity = profile_distances(profile, extract_reference_profile(flatten_velocity(base)))
    assert velocity["performance_texture"] > 0
    assert velocity["rhythm_time"] == 0
    assert velocity["pitch_harmony"] == 0

    pedal = profile_distances(profile, extract_reference_profile(destroy_pedal(base)))
    assert pedal["performance_texture"] > 0
    assert pedal["rhythm_time"] == 0
    assert pedal["pitch_harmony"] == 0


@pytest.mark.parametrize(
    "notes, message",
    [
        ([], "at least one"),
        ([{"pitch": 128, "onset_ms": 0, "duration_ms": 1, "velocity": 1}], "pitch"),
        ([{"pitch": 60, "onset_ms": -1, "duration_ms": 1, "velocity": 1}], "onset"),
        ([{"pitch": 60, "onset_ms": 0, "duration_ms": 0, "velocity": 1}], "duration"),
        ([{"pitch": 60, "onset_ms": 0, "duration_ms": 1, "velocity": 0}], "velocity"),
        ([{"pitch": 60, "onset_ms": float("nan"), "duration_ms": 1, "velocity": 1}], "finite"),
    ],
)
def test_invalid_piece_is_not_converted_to_a_normal_profile(
    notes: list[dict[str, object]], message: str
) -> None:
    with pytest.raises(ReferenceProfileError, match=message):
        ReferencePiece.from_dicts(name="invalid.mid", notes=notes, pedals=[])


@pytest.mark.parametrize("name", ["", "   ", None])
def test_piece_name_is_required(name: object) -> None:
    with pytest.raises(ReferenceProfileError, match="name"):
        ReferencePiece.from_dicts(
            name=name,  # type: ignore[arg-type]
            notes=[
                {"pitch": 60, "onset_ms": 0, "duration_ms": 100, "velocity": 80},
                {"pitch": 62, "onset_ms": 200, "duration_ms": 100, "velocity": 80},
            ],
            pedals=[],
        )


def test_piece_requires_two_distinct_onsets() -> None:
    with pytest.raises(ReferenceProfileError, match="two distinct"):
        ReferencePiece.from_dicts(
            name="one-attack.mid",
            notes=[
                {"pitch": 60, "onset_ms": 0, "duration_ms": 100, "velocity": 80},
                {"pitch": 64, "onset_ms": 0, "duration_ms": 100, "velocity": 80},
            ],
            pedals=[],
        )


@pytest.mark.parametrize(
    "pedal, message",
    [
        ({"at_ms": -1, "value": 0}, "at_ms"),
        ({"at_ms": 0, "value": 128}, "value"),
        ({"at_ms": 0, "value": 0.5}, "value"),
        ({"at_ms": float("nan"), "value": 0}, "finite"),
    ],
)
def test_invalid_pedal_is_rejected(pedal: dict[str, object], message: str) -> None:
    with pytest.raises(ReferenceProfileError, match=message):
        ReferencePiece.from_dicts(
            name="invalid-pedal.mid",
            notes=[
                {"pitch": 60, "onset_ms": 0, "duration_ms": 100, "velocity": 80},
                {"pitch": 62, "onset_ms": 200, "duration_ms": 100, "velocity": 80},
            ],
            pedals=[pedal],
        )


def test_dangling_note_in_smf_is_unavailable(tmp_path: Path) -> None:
    midi = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.Message("note_on", note=60, velocity=80, time=0))
    path = tmp_path / "dangling.mid"
    midi.save(path)

    with pytest.raises(ReferenceProfileError, match="dangling"):
        load_reference_piece(path)


def test_valid_smf_loads_tempo_and_pedal(tmp_path: Path) -> None:
    midi = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=600_000, time=0))
    track.append(mido.Message("control_change", control=64, value=127, time=0))
    track.append(mido.Message("control_change", channel=9, control=64, value=127, time=0))
    track.append(mido.Message("note_on", note=60, velocity=80, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=240))
    track.append(mido.Message("note_on", note=62, velocity=82, time=240))
    track.append(mido.Message("note_off", note=62, velocity=0, time=240))
    track.append(mido.Message("control_change", control=64, value=0, time=0))
    path = tmp_path / "valid.mid"
    midi.save(path)

    piece = load_reference_piece(path)

    assert len(piece.notes) == 2
    assert [(event.value, event.at_ms) for event in piece.pedals] == [(127, 0), (0, 900)]


def test_invalid_smf_forms_are_unavailable(tmp_path: Path) -> None:
    bad = tmp_path / "bad.mid"
    bad.write_bytes(b"not midi")
    with pytest.raises(ReferenceProfileError, match="unable to read"):
        load_reference_piece(bad)

    format_two = mido.MidiFile(type=2, ticks_per_beat=480)
    format_two.tracks.append(mido.MidiTrack())
    format_two.tracks.append(mido.MidiTrack())
    format_two_path = tmp_path / "format-two.mid"
    format_two.save(format_two_path)
    with pytest.raises(ReferenceProfileError, match="format 2"):
        load_reference_piece(format_two_path)

    unmatched = mido.MidiFile(type=0, ticks_per_beat=480)
    unmatched_track = mido.MidiTrack()
    unmatched.tracks.append(unmatched_track)
    unmatched_track.append(mido.Message("note_off", note=60, velocity=0, time=0))
    unmatched_path = tmp_path / "unmatched.mid"
    unmatched.save(unmatched_path)
    with pytest.raises(ReferenceProfileError, match="unmatched"):
        load_reference_piece(unmatched_path)


def test_local_neighborhood_excludes_self_from_the_required_neighbors_and_is_stable() -> None:
    profiles = {
        "anchor.mid": extract_reference_profile(_piece("anchor.mid")),
        "Beta.mid": extract_reference_profile(_piece("Beta.mid", 1)),
        "alpha.mid": extract_reference_profile(_piece("alpha.mid", 2)),
        "gamma.mid": extract_reference_profile(_piece("gamma.mid", 3)),
    }

    neighborhood = select_local_neighborhood("anchor.mid", profiles, neighbor_count=3)

    assert neighborhood["anchor"] == "anchor.mid"
    assert neighborhood["neighbors"][0]["name"] == "anchor.mid"
    assert [item["name"] for item in neighborhood["neighbors"][1:]] == [
        "alpha.mid",
        "Beta.mid",
    ]
    with pytest.raises(ReferenceProfileError, match="enough non-anchor"):
        select_local_neighborhood("anchor.mid", profiles, neighbor_count=5)
    with pytest.raises(ReferenceProfileError, match="unknown anchor"):
        select_local_neighborhood("missing.mid", profiles, neighbor_count=3)
    with pytest.raises(ReferenceProfileError, match="at least 3"):
        select_local_neighborhood("anchor.mid", profiles, neighbor_count=2)


def test_default_reference_is_a_real_piece_and_coordinate_median_is_diagnostic() -> None:
    profiles = {
        "a.mid": extract_reference_profile(_piece("a.mid")),
        "b.mid": extract_reference_profile(_piece("b.mid", 2)),
        "c.mid": extract_reference_profile(_piece("c.mid", 4)),
    }

    result = select_default_reference(profiles)

    assert result["selected"]["name"] in profiles
    assert result["selected"]["kind"] == "real_medoid"
    assert result["coordinate_median"]["kind"] == "diagnostic_synthetic"
    with pytest.raises(ReferenceProfileError, match="three references"):
        select_default_reference(dict(list(profiles.items())[:2]))


def test_copy_fingerprint_is_stretch_and_transposition_invariant_but_not_order_blind() -> None:
    from llm_musical_composer.reference_controls import (
        destroy_pitch_order,
        stretch_piece,
        transpose_piece,
    )

    base = _piece()
    fingerprint = build_copy_fingerprint(base)
    assert build_copy_fingerprint(transpose_piece(base, 5)) == fingerprint
    assert build_copy_fingerprint(stretch_piece(base, 1.5)) == fingerprint
    disrupted = build_copy_fingerprint(destroy_pitch_order(base))
    assert copy_fingerprint_similarity(fingerprint, disrupted) < 1.0

    references = {"base.mid": fingerprint, "other.mid": disrupted}
    result = evaluate_copy_risk(fingerprint, references, review_threshold=0.8)
    assert result["status"] == "exact_copy"
    assert result["nearest_reference"] == "base.mid"

    review = evaluate_copy_risk(
        {"sequence_sha256": "candidate", "token_hashes": ["a", "b"]},
        {"near.mid": {"sequence_sha256": "near", "token_hashes": ["a", "b", "c"]}},
        review_threshold=0.5,
    )
    assert review["status"] == "review"
    passed = evaluate_copy_risk(
        {"sequence_sha256": "candidate", "token_hashes": ["x"]},
        {"far.mid": {"sequence_sha256": "far", "token_hashes": ["a"]}},
        review_threshold=0.5,
    )
    assert passed["status"] == "pass"


def test_empty_copy_tokens_and_invalid_copy_request_are_explicit() -> None:
    short = ReferencePiece.from_dicts(
        name="short.mid",
        notes=[
            {"pitch": 60, "onset_ms": 0, "duration_ms": 100, "velocity": 80},
            {"pitch": 62, "onset_ms": 200, "duration_ms": 100, "velocity": 80},
        ],
        pedals=[],
    )
    fingerprint = build_copy_fingerprint(short)
    assert fingerprint["token_count"] == 0
    assert copy_fingerprint_similarity(fingerprint, fingerprint) == 1.0
    with pytest.raises(ReferenceProfileError, match="threshold"):
        evaluate_copy_risk(fingerprint, {"short.mid": fingerprint}, review_threshold=1.1)
    with pytest.raises(ReferenceProfileError, match="at least one"):
        evaluate_copy_risk(fingerprint, {}, review_threshold=0.5)


def test_profile_distance_rejects_unavailable_or_mismatched_profiles() -> None:
    profile = extract_reference_profile(_piece())
    with pytest.raises(ReferenceProfileError, match="status"):
        profile_distances(profile, {"status": "unable_to_investigate"})
    broken = {**profile, "feature_groups": dict(profile["feature_groups"])}
    broken["feature_groups"].pop("rhythm_time")
    with pytest.raises(ReferenceProfileError, match="feature groups"):
        profile_distances(profile, broken)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda value: value["feature_groups"]["rhythm_time"].update(metrics=None), "metrics"),
        (
            lambda value: value["feature_groups"]["rhythm_time"]["metrics"].pop("ioi_ratio"),
            "metric sets",
        ),
        (
            lambda value: value["feature_groups"]["rhythm_time"]["metrics"]["ioi_ratio"].update(
                kind="unknown"
            ),
            "metric kinds",
        ),
        (
            lambda value: value["feature_groups"]["rhythm_time"]["metrics"]["ioi_ratio"].update(
                values="bad"
            ),
            "metric values",
        ),
        (
            lambda value: value["feature_groups"]["rhythm_time"]["metrics"]["ioi_ratio"].update(
                values=[]
            ),
            "dimensions",
        ),
        (
            lambda value: value["feature_groups"]["rhythm_time"]["metrics"]["ioi_ratio"].update(
                values=[float("nan")] * 6
            ),
            "finite",
        ),
    ],
)
def test_profile_distance_rejects_broken_metric_contracts(mutation, message: str) -> None:
    profile = extract_reference_profile(_piece())
    broken = copy.deepcopy(profile)
    mutation(broken)
    with pytest.raises(ReferenceProfileError, match=message):
        profile_distances(profile, broken)
