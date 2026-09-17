import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import scoim.solo_piano_ending as ending_module
from scoim.realization_operations import (
    build_harmony_checked_diff,
    build_melody_checked_diff,
    build_plan_checked_diff,
)
from scoim.realization_workspace import (
    AccompanimentNote,
    AccompanimentValue,
    RealizationWorkspace,
    apply_checked_diff,
    create_workspace,
)
from scoim.solo_piano_ending import materialize_solo_piano_ending
from scoim.validation import content_sha256

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "scoim"


def _approved_script(fixture: str = "nested-aba") -> dict[str, object]:
    return json.loads((FIXTURE_ROOT / fixture / "approved-script.json").read_text(encoding="utf-8"))


def _must_apply(workspace: RealizationWorkspace, checked_diff: object) -> RealizationWorkspace:
    applied = apply_checked_diff(workspace, checked_diff)  # type: ignore[arg-type]
    assert applied.workspace is not None, applied.issues
    return applied.workspace


def _workspace(document: dict[str, object]) -> RealizationWorkspace:
    workspace = create_workspace(document)
    script = document["script"]
    assert isinstance(script, dict)
    sections = script["sections"]
    assert isinstance(sections, dict)
    contrast_count = sum(
        1
        for section in sections.values()
        if isinstance(section, dict) and section["role"] == "contrast"
    )
    workspace = _must_apply(
        workspace,
        build_plan_checked_diff(
            document,
            workspace,
            {
                "tonal_center": 2,
                "mode": "minor",
                "contrast_descriptions": ["対照"] * contrast_count,
            },
        ),
    )
    harmony_by_material = {
        "connector": [{"duration_units": 4, "root_pitch_class": 9, "quality": "minor"}],
        "contrast": [{"duration_units": 8, "root_pitch_class": 2, "quality": "major"}],
        "theme": [{"duration_units": 8, "root_pitch_class": 2, "quality": "minor"}],
    }
    harmony_targets = tuple(sorted(harmony_by_material))
    workspace = _must_apply(
        workspace,
        build_harmony_checked_diff(
            document,
            workspace,
            harmony_targets,
            {"harmonies": [harmony_by_material[target] for target in harmony_targets]},
        ),
    )
    melody_targets = ("contrast", "theme")
    workspace = _must_apply(
        workspace,
        build_melody_checked_diff(
            document,
            workspace,
            melody_targets,
            {
                "melodies": [
                    {
                        "foreground_voice": "upper",
                        "notes": [{"at_units": 0, "duration_units": 8, "pitch": 69}],
                    },
                    {
                        "foreground_voice": "upper",
                        "notes": [{"at_units": 0, "duration_units": 8, "pitch": 74}],
                    },
                ]
            },
        ),
    )
    return workspace


def _reapprove(document: dict[str, object]) -> dict[str, object]:
    approval = document["approval"]
    assert isinstance(approval, dict)
    approval["content_sha256"] = content_sha256(document)
    return document


@pytest.mark.parametrize(
    ("fixture", "expected_grid", "expected_scales"),
    [
        (
            "fixed-aba",
            4,
            {"cadence": 4, "connector": 1, "contrast": 2, "theme": 2},
        ),
        (
            "nested-aba",
            8,
            {"cadence": 8, "connector": 2, "contrast": 3, "theme": 3},
        ),
    ],
)
def test_materializes_one_shared_tonic_ending_without_rounding_existing_grid(
    fixture: str,
    expected_grid: int,
    expected_scales: dict[str, int],
) -> None:
    document = _approved_script(fixture)
    workspace = _workspace(document)

    result = materialize_solo_piano_ending(document, workspace)

    assert result.material_key == "cadence"
    assert result.previous_material_key == "theme"
    assert result.length_units == 2
    assert result.harmonies[0].root_pitch_class == 2
    assert result.harmonies[0].quality == "minor"
    assert result.harmonies[0].duration_units == 2
    assert result.melody.foreground_voice == "upper"
    assert result.melody.notes[0].pitch == 74
    assert result.melody.notes[0].at_units == 0
    assert result.melody.notes[0].duration_units == 2
    assert {note.degree for note in result.accompaniment.notes} == {"root", "fifth"}
    assert all(note.at_units == 0 for note in result.accompaniment.notes)
    assert all(note.duration_units == 2 for note in result.accompaniment.notes)
    assert result.common_grid_units_per_weight == expected_grid
    assert dict(result.material_scale_factors) == expected_scales
    assert result.nominal_hold_ms >= 2_000


def test_lower_foreground_gets_the_nearest_tonic_with_accompaniment_above() -> None:
    document = _approved_script()
    workspace = _workspace(document)
    workspace = _must_apply(
        workspace,
        build_melody_checked_diff(
            document,
            workspace,
            ("theme",),
            {
                "melodies": [
                    {
                        "foreground_voice": "lower",
                        "notes": [{"at_units": 0, "duration_units": 8, "pitch": 50}],
                    }
                ]
            },
        ),
    )

    result = materialize_solo_piano_ending(document, workspace)

    assert result.melody.foreground_voice == "lower"
    assert result.melody.notes[0].pitch == 50
    assert all(note.pitch > 50 for note in result.accompaniment.notes)


def test_requires_a_version_three_workspace() -> None:
    document = _approved_script()
    workspace = replace(_workspace(document), schema_version=2)

    with pytest.raises(ValueError, match="version 3"):
        materialize_solo_piano_ending(document, workspace)


def test_requires_an_existing_overall_plan() -> None:
    document = _approved_script()
    workspace = replace(_workspace(document), plan=None)

    with pytest.raises(ValueError, match="overall plan"):
        materialize_solo_piano_ending(document, workspace)


def test_reports_a_script_that_can_no_longer_be_projected() -> None:
    document = _approved_script()
    workspace = _workspace(document)
    script = document["script"]
    assert isinstance(script, dict)
    sections = script["sections"]
    assert isinstance(sections, dict)
    release = sections["release"]
    assert isinstance(release, dict)
    release["relative_length"] = 0
    _reapprove(document)

    with pytest.raises(ValueError):
        materialize_solo_piano_ending(document, workspace)


def test_requires_music_before_the_final_release(monkeypatch: pytest.MonkeyPatch) -> None:
    document = _approved_script()
    workspace = _workspace(document)
    monkeypatch.setattr(
        ending_module,
        "project_solo_piano_3m",
        lambda *args, **kwargs: SimpleNamespace(
            projected=True,
            piece_plan=SimpleNamespace(nodes=(SimpleNamespace(score_material_id="cadence"),)),
        ),
    )

    with pytest.raises(ValueError, match="music before"):
        materialize_solo_piano_ending(document, workspace)


def test_requires_an_ordinary_material_before_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _approved_script()
    workspace = _workspace(document)
    monkeypatch.setattr(
        ending_module,
        "project_solo_piano_3m",
        lambda *args, **kwargs: SimpleNamespace(
            projected=True,
            piece_plan=SimpleNamespace(
                nodes=(
                    SimpleNamespace(
                        score_material_id="connector",
                        role="transition",
                        duration_weight=1,
                    ),
                    SimpleNamespace(
                        score_material_id="cadence",
                        role="release",
                        duration_weight=2,
                    ),
                )
            ),
        ),
    )

    with pytest.raises(ValueError, match="preceding ordinary melody"):
        materialize_solo_piano_ending(document, workspace)


def test_rejects_a_non_release_final_leaf() -> None:
    document = deepcopy(_approved_script())
    script = document["script"]
    assert isinstance(script, dict)
    sections = script["sections"]
    assert isinstance(sections, dict)
    release = sections["release"]
    assert isinstance(release, dict)
    release["role"] = "contrast"
    _reapprove(document)
    workspace = _workspace(document)

    with pytest.raises(ValueError, match="final leaf must be a release"):
        materialize_solo_piano_ending(document, workspace)


def test_rejects_more_than_one_ending_material() -> None:
    document = deepcopy(_approved_script())
    script = document["script"]
    assert isinstance(script, dict)
    materials = script["materials"]
    assert isinstance(materials, dict)
    materials["unused_ending"] = {"kind": "ending", "description": "未配置の終止。"}
    _reapprove(document)
    workspace = _workspace(document)

    with pytest.raises(ValueError, match="exactly one"):
        materialize_solo_piano_ending(document, workspace)


def test_requires_the_melody_before_release() -> None:
    document = _approved_script()
    workspace = replace(_workspace(document), melodies=())

    with pytest.raises(ValueError, match="melody before"):
        materialize_solo_piano_ending(document, workspace)


def test_reports_an_unplaceable_shared_ending(monkeypatch: pytest.MonkeyPatch) -> None:
    document = _approved_script()
    workspace = _workspace(document)
    monkeypatch.setattr(
        ending_module,
        "place_workspace_accompaniment",
        lambda **kwargs: SimpleNamespace(
            value=None,
            reason="no safe shared ending",
            status="search_unplaceable",
        ),
    )

    with pytest.raises(ValueError, match="no safe shared ending"):
        materialize_solo_piano_ending(document, workspace)


def test_rejects_a_placer_result_that_shortens_the_shared_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _approved_script()
    workspace = _workspace(document)
    invalid = AccompanimentValue(
        "upper",
        (
            AccompanimentNote(0, 2, 1, "root", "bass", 38, ("tenuto",)),
            AccompanimentNote(0, 2, 2, "fifth", "low", 45, ("tenuto",)),
        ),
    )
    monkeypatch.setattr(
        ending_module,
        "place_workspace_accompaniment",
        lambda **kwargs: SimpleNamespace(value=invalid, candidate_evaluation_count=1),
    )

    with pytest.raises(ValueError, match="full hold"):
        materialize_solo_piano_ending(document, workspace)


def test_rejects_a_placer_result_without_root_and_fifth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _approved_script()
    workspace = _workspace(document)
    invalid = AccompanimentValue(
        "upper",
        (
            AccompanimentNote(0, 2, 2, "root", "bass", 38, ("tenuto",)),
            AccompanimentNote(0, 2, 2, "third", "low", 41, ("tenuto",)),
        ),
    )
    monkeypatch.setattr(
        ending_module,
        "place_workspace_accompaniment",
        lambda **kwargs: SimpleNamespace(value=invalid, candidate_evaluation_count=1),
    )

    with pytest.raises(ValueError, match="root and fifth"):
        materialize_solo_piano_ending(document, workspace)


def test_requires_harmony_for_every_placed_upstream_material() -> None:
    document = _approved_script()
    workspace = _workspace(document)
    workspace = replace(
        workspace,
        harmonies=tuple(item for item in workspace.harmonies if item[0] != "connector"),
    )

    with pytest.raises(ValueError, match="no current harmony: connector"):
        materialize_solo_piano_ending(document, workspace)


def test_rejects_a_release_too_short_for_two_seconds() -> None:
    document = deepcopy(_approved_script())
    script = document["script"]
    assert isinstance(script, dict)
    sections = script["sections"]
    assert isinstance(sections, dict)
    release = sections["release"]
    assert isinstance(release, dict)
    release["relative_length"] = 0.01
    _reapprove(document)
    workspace = _workspace(document)

    with pytest.raises(ValueError, match="too short"):
        materialize_solo_piano_ending(document, workspace)
