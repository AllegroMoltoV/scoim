import copy
import json

import pytest

from llm_musical_composer.evaluator_registry import EvaluatorRegistryError
from llm_musical_composer.reference_controls import stretch_piece, transpose_piece
from llm_musical_composer.reference_profile import ReferencePiece


def _piece(name: str = "base.mid") -> ReferencePiece:
    attacks = [
        (0, (48, 60, 64), (66, 78, 82), (750, 520, 480)),
        (600, (52, 62), (70, 88), (460, 280)),
        (1050, (55, 64, 67), (72, 84, 86), (700, 500, 520)),
        (1700, (50, 65), (68, 94), (530, 260)),
        (2200, (53, 67, 72), (74, 90, 96), (680, 420, 390)),
        (2800, (48, 69), (65, 85), (620, 250)),
        (3300, (52, 71, 76), (71, 92, 99), (760, 410, 360)),
        (4000, (55, 72), (69, 87), (540, 310)),
        (4500, (48, 60, 64), (64, 80, 84), (900, 720, 700)),
    ]
    notes = []
    for onset, pitches, velocities, durations in attacks:
        for pitch, velocity, duration in zip(pitches, velocities, durations, strict=True):
            notes.append(
                {
                    "pitch": pitch,
                    "onset_ms": onset,
                    "duration_ms": duration,
                    "velocity": velocity,
                }
            )
    return ReferencePiece.from_dicts(name=name, notes=notes, pedals=[])


def _rebuild(piece: ReferencePiece, *, notes: list[dict[str, int]]) -> ReferencePiece:
    return ReferencePiece.from_dicts(name=piece.name, notes=notes, pedals=[])


def _metric_values(result: dict, axis: str) -> dict[str, float]:
    return {
        name: metric["value"]
        for name, metric in result["axes"][axis]["metrics"].items()
        if metric["status"] == "available"
    }


def test_descriptor_is_independent_of_input_order() -> None:
    from llm_musical_composer.control_axis_anchors import extract_control_axis_descriptors

    piece = _piece()
    reordered = ReferencePiece(piece.name, tuple(reversed(piece.notes)), piece.pedals)

    assert extract_control_axis_descriptors(reordered) == extract_control_axis_descriptors(piece)


def test_brightness_descriptors_are_transposition_invariant() -> None:
    from llm_musical_composer.control_axis_anchors import extract_control_axis_descriptors

    base = extract_control_axis_descriptors(_piece())
    shifted = extract_control_axis_descriptors(transpose_piece(_piece(), 5))

    assert _metric_values(shifted, "あかるさ") == _metric_values(base, "あかるさ")


def test_motion_responds_to_time_but_not_to_extra_chord_tones() -> None:
    from llm_musical_composer.control_axis_anchors import extract_control_axis_descriptors

    piece = _piece()
    base = extract_control_axis_descriptors(piece)
    stretched = extract_control_axis_descriptors(stretch_piece(piece, 1.5))
    assert (
        stretched["axes"]["うごき"]["metrics"]["median_ioi_ms"]["value"]
        > base["axes"]["うごき"]["metrics"]["median_ioi_ms"]["value"]
    )

    extra_notes = [
        {
            "pitch": note.pitch,
            "onset_ms": note.onset_ms,
            "duration_ms": note.duration_ms,
            "velocity": note.velocity,
        }
        for note in piece.notes
    ]
    extra_notes.extend(
        {
            "pitch": min(127, note.pitch + 12),
            "onset_ms": note.onset_ms,
            "duration_ms": note.duration_ms,
            "velocity": note.velocity,
        }
        for note in piece.notes
        if note.pitch >= 60
    )
    thickened = extract_control_axis_descriptors(_rebuild(piece, notes=extra_notes))

    assert _metric_values(thickened, "うごき") == _metric_values(base, "うごき")


def test_power_is_offset_invariant_and_bass_alone_cannot_raise_every_metric() -> None:
    from llm_musical_composer.control_axis_anchors import extract_control_axis_descriptors

    piece = _piece()
    base = extract_control_axis_descriptors(piece)
    offset = _rebuild(
        piece,
        notes=[
            {
                "pitch": note.pitch,
                "onset_ms": note.onset_ms,
                "duration_ms": note.duration_ms,
                "velocity": note.velocity + 5,
            }
            for note in piece.notes
        ],
    )
    assert _metric_values(extract_control_axis_descriptors(offset), "パワー") == _metric_values(
        base, "パワー"
    )

    notes = [
        {
            "pitch": note.pitch,
            "onset_ms": note.onset_ms,
            "duration_ms": note.duration_ms,
            "velocity": note.velocity,
        }
        for note in piece.notes
    ]
    notes.extend(
        {
            "pitch": max(0, min(note.pitch for note in piece.notes if note.onset_ms == onset) - 12),
            "onset_ms": onset,
            "duration_ms": 500,
            "velocity": 72,
        }
        for onset in sorted({note.onset_ms for note in piece.notes})
    )
    bass_added = extract_control_axis_descriptors(_rebuild(piece, notes=notes))
    base_values = _metric_values(base, "パワー")
    bass_values = _metric_values(bass_added, "パワー")

    assert bass_values != base_values
    assert any(bass_values[name] <= value for name, value in base_values.items())


def test_unobservable_score_and_voice_features_are_explicitly_unavailable() -> None:
    from llm_musical_composer.control_axis_anchors import extract_control_axis_descriptors

    result = extract_control_axis_descriptors(_piece())
    unavailable = {item["id"]: item["reason"] for item in result["unavailable_features"]}

    assert "score_relative_rubato" in unavailable
    assert "not present in ReferencePiece" in unavailable["score_relative_rubato"]
    assert "voice_convergence" in unavailable
    assert "track/channel/voice labels" in unavailable["voice_convergence"]


def test_fewer_than_eight_attacks_is_unavailable_for_anchor_stability() -> None:
    from llm_musical_composer.control_axis_anchors import build_descriptor_record

    piece = _piece()
    onsets = sorted({note.onset_ms for note in piece.notes})[:7]
    short = ReferencePiece(piece.name, tuple(n for n in piece.notes if n.onset_ms in onsets), ())

    result = build_descriptor_record(short)

    assert result["status"] == "unavailable"
    assert "eight contiguous onset blocks" in result["reason"]
    assert result["descriptors"]["status"] == "available"
    assert result["note_count"] == len(short.notes)
    assert result["duration_ms"] == max(note.onset_ms + note.duration_ms for note in short.notes)


def _record(name: str, base: float, instability: float = 0.0) -> dict:
    axes = {}
    jackknife = []
    for axis in ("あかるさ", "うごき", "パワー"):
        axes[axis] = {
            "status": "available",
            "metrics": {
                "m1": {
                    "status": "available",
                    "value": base,
                    "positive_direction": "higher",
                },
                "m2": {
                    "status": "available",
                    "value": base * 2,
                    "positive_direction": "higher",
                },
            },
        }
    for index in range(8):
        variation = instability if index % 2 else -instability
        changed = copy.deepcopy(axes)
        for axis in changed.values():
            axis["metrics"]["m1"]["value"] += variation
            axis["metrics"]["m2"]["value"] += variation
        jackknife.append({"omitted_block": index, "axes": changed})
    return {
        "name": name,
        "status": "available",
        "descriptors": {"axes": axes},
        "jackknife": jackknife,
    }


def _registry(status: str) -> dict:
    return {
        "evaluators": [
            {
                "id": "control_axis_brightness_descriptors_v1",
                "status": status,
            },
            {"id": "control_axis_motion_descriptors_v1", "status": status},
            {"id": "control_axis_power_descriptors_v1", "status": status},
        ]
    }


def test_candidate_selection_is_deterministic_and_records_rank_width() -> None:
    from llm_musical_composer.control_axis_anchors import select_axis_candidates

    records = [
        _record("z.mid", 0.0),
        _record("Alpha.mid", 1.0, 0.1),
        _record("beta.mid", 2.0),
        _record("gamma.mid", 3.0),
        _record("delta.mid", 4.0),
        _record("epsilon.mid", 5.0),
    ]

    first = select_axis_candidates(records, "あかるさ", registry=_registry("selection"))
    second = select_axis_candidates(
        list(reversed(records)), "あかるさ", registry=_registry("selection")
    )

    assert first == second
    assert [item["name"] for item in first["low"]] == ["z.mid", "Alpha.mid"]
    assert [item["name"] for item in first["high"]] == ["epsilon.mid", "delta.mid"]
    assert all(item["rank_width"] >= 0 for side in first.values() for item in side)


@pytest.mark.parametrize("status", ["diagnostic_only", "rejected"])
def test_diagnostic_or_rejected_descriptors_cannot_rank(status: str) -> None:
    from llm_musical_composer.control_axis_anchors import select_axis_candidates

    with pytest.raises(EvaluatorRegistryError, match="cannot be used for ranking"):
        select_axis_candidates(
            [_record("a.mid", 0.0), _record("b.mid", 1.0), _record("c.mid", 2.0)],
            "あかるさ",
            registry=_registry(status),
        )


def test_unknown_axis_is_rejected() -> None:
    from llm_musical_composer.control_axis_anchors import select_axis_candidates

    with pytest.raises(ValueError, match="unknown control axis"):
        select_axis_candidates([], "速さ", registry=_registry("selection"))


def test_listening_sheet_asks_concrete_three_choice_question(tmp_path) -> None:
    from llm_musical_composer.control_axis_anchors import _listening_materials

    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    candidates = {}
    for axis_index, axis in enumerate(("あかるさ", "うごき", "パワー")):
        candidates[axis] = {}
        for side_index, side in enumerate(("low", "middle", "high")):
            name = f"piece-{axis_index}-{side_index}.mid"
            (source_dir / name).write_bytes(name.encode())
            candidates[axis][side] = [{"name": name}]

    _, sheet = _listening_materials(
        source_dir=source_dir,
        output_dir=output_dir,
        candidates=candidates,
    )

    assert "いちばん違って聞こえる項目" in sheet
    assert "あかるさ・うごき・パワー" in sheet
    assert "弱いほうから強いほうへの順番" in sheet
    assert "共通して変化している" not in sheet


def test_backup_listening_uses_second_candidates_for_failed_axes(tmp_path) -> None:
    from llm_musical_composer.control_axis_anchors import build_backup_listening_round

    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    output_dir.mkdir()
    axes = {}
    for axis in ("あかるさ", "うごき", "パワー"):
        axes[axis] = {}
        for side in ("low", "middle", "high"):
            axes[axis][side] = []
            for candidate_index in range(2):
                name = f"{axis}-{side}-{candidate_index}.mid"
                (source_dir / name).write_bytes(name.encode())
                axes[axis][side].append({"name": name})
    (output_dir / "candidates.json").write_text(
        json.dumps({"status": "pass", "axes": axes}, ensure_ascii=False),
        encoding="utf-8",
    )

    result = build_backup_listening_round(
        source_dir=source_dir,
        output_dir=output_dir,
        axes=("うごき", "パワー"),
    )

    assert result["status"] == "pass"
    assert result["axes"] == ["うごき", "パワー"]
    assert len(result["mapping"]) == 6
    assert all(item["source_name"].endswith("-1.mid") for item in result["mapping"])
    sheet = (output_dir / "listening-sheet-backup.md").read_text(encoding="utf-8")
    assert "## 組1" in sheet
    assert "## 組2" in sheet
    assert "## 組3" not in sheet
    assert (output_dir / "backup-manifest.json").is_file()
