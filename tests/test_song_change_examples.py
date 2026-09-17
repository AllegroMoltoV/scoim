from __future__ import annotations

import json

import mido

from llm_musical_composer.piano_texture import evaluate_variation_contracts
from llm_musical_composer.song_change import describe_song_change
from llm_musical_composer.song_change_examples import (
    build_song_change_triplet,
    write_song_change_review,
)


def _relation(report: dict[str, object], relation: str) -> dict[str, object]:
    return next(item for item in report["phrase_relations"] if item["relation"] == relation)


def test_triplet_holds_global_confounds_and_moves_relations_independently() -> None:
    triplet = build_song_change_triplet()
    reports = {level: describe_song_change(item) for level, item in triplet.items()}

    assert set(triplet) == {"low", "center", "high"}
    assert all(report["status"] == "measured" for report in reports.values())
    assert reports["low"]["hold"] == reports["center"]["hold"] == reports["high"]["hold"]
    assert reports["low"]["hold"]["duration_ms"] == 30_000
    assert reports["low"]["hold"]["note_count"] == 40
    assert all(evaluate_variation_contracts(item)["status"] == "pass" for item in triplet.values())

    variation_distances = [
        _relation(reports[level], "variation")["upper_rhythm_position_distance"]
        for level in ("low", "center", "high")
    ]
    contrast_rhythm_distances = [
        _relation(reports[level], "contrast")["upper_rhythm_position_distance"]
        for level in ("low", "center", "high")
    ]
    contrast_pitch_distances = [
        _relation(reports[level], "contrast")["upper_interval_distance_semitones"]
        for level in ("low", "center", "high")
    ]
    assert variation_distances == sorted(variation_distances)
    assert len(set(variation_distances)) == 3
    assert contrast_rhythm_distances == sorted(contrast_rhythm_distances)
    assert len(set(contrast_rhythm_distances)) == 3
    assert contrast_pitch_distances == sorted(contrast_pitch_distances)
    assert len(set(contrast_pitch_distances)) == 3
    for report in reports.values():
        returned = _relation(report, "return")
        assert returned["upper_rhythm_position_distance"] == 0.0
        assert returned["upper_interval_distance_semitones"] == 0.0


def test_review_export_is_anonymous_and_rereadable(tmp_path) -> None:
    result = write_song_change_review(tmp_path)

    manifest = json.loads((tmp_path / "review-manifest.json").read_text(encoding="utf-8"))
    mapping = json.loads((tmp_path / "private-mapping.json").read_text(encoding="utf-8"))
    assert result == manifest
    assert set(manifest) == {"schema_version", "question", "clips"}
    assert len(manifest["clips"]) == 3
    assert "level" not in json.dumps(manifest)
    assert len(mapping["clips"]) == 3
    for name in manifest["clips"]:
        assert mido.MidiFile(tmp_path / name).length == 30.0
