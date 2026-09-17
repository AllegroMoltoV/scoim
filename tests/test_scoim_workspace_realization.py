import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

import scoim.realization_workspace as workspace_module
from scoim.realization import realize_solo_piano_3m
from scoim.realization_operations import (
    build_accompaniment_checked_diff,
    build_ending_checked_diff,
    build_harmony_checked_diff,
    build_melody_checked_diff,
    build_performance_checked_diff,
    build_performance_defaults_checked_diff,
    build_plan_checked_diff,
    build_transition_accompaniment_checked_diff,
    build_transition_melody_checked_diff,
)
from scoim.realization_workspace import (
    RealizationWorkspace,
    apply_checked_diff,
    create_workspace,
    workspace_to_dict,
)
from scoim.solo_piano_performance import performance_operation_schedule
from scoim.trial_bundle import create_trial_bundle, replay_trial_bundle
from scoim.validation import content_sha256
from scoim.workspace_realization import (
    WorkspaceRealizationError,
    build_workspace_diagnostic_irs,
    build_workspace_ir,
)

_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "scoim"


def _approved_script(name: str) -> dict[str, object]:
    return json.loads((_FIXTURE_ROOT / name / "approved-script.json").read_text(encoding="utf-8"))


def _apply(workspace: RealizationWorkspace, checked) -> RealizationWorkspace:
    result = apply_checked_diff(workspace, checked)
    assert result.workspace is not None, result.issues
    return result.workspace


def _completed_workspace(document: dict[str, object]) -> RealizationWorkspace:
    script = document["script"]
    assert isinstance(script, dict)
    materials = script["materials"]
    sections = script["sections"]
    transitions = script["transitions"]
    assert isinstance(materials, dict)
    assert isinstance(sections, dict)
    assert isinstance(transitions, dict)

    workspace = create_workspace(document)
    contrast_count = sum(
        isinstance(section, dict) and section["role"] == "contrast" for section in sections.values()
    )
    workspace = _apply(
        workspace,
        build_plan_checked_diff(
            document,
            workspace,
            {
                "tonal_center": 0,
                "mode": "major",
                "contrast_descriptions": ["対照を作る。"] * contrast_count,
            },
        ),
    )
    normal_material_ids = tuple(
        sorted(
            material_id
            for material_id, material in materials.items()
            if isinstance(material_id, str)
            and isinstance(material, dict)
            and material["kind"] != "ending"
        )
    )
    workspace = _apply(
        workspace,
        build_harmony_checked_diff(
            document,
            workspace,
            normal_material_ids,
            {
                "harmonies": [
                    [{"duration_units": 4, "root_pitch_class": 0, "quality": "major"}]
                    for _ in normal_material_ids
                ]
            },
        ),
    )
    melodic_material_ids = tuple(
        material_id
        for material_id in normal_material_ids
        if materials[material_id]["kind"] != "transition"
    )
    workspace = _apply(
        workspace,
        build_melody_checked_diff(
            document,
            workspace,
            melodic_material_ids,
            {
                "melodies": [
                    {
                        "foreground_voice": "upper",
                        "notes": [{"at_units": 0, "duration_units": 4, "pitch": 72}],
                    }
                    for _ in melodic_material_ids
                ]
            },
        ),
    )
    transition_ids = tuple(sorted(transitions))
    if transition_ids:
        workspace = _apply(
            workspace,
            build_transition_melody_checked_diff(
                document,
                workspace,
                transition_ids,
                {
                    "transition_melodies": [
                        {
                            "foreground_voice": "upper",
                            "notes": [{"at_units": 0, "duration_units": 4, "pitch": 72}],
                        }
                        for _ in transition_ids
                    ]
                },
            ),
        )
    accompaniment = {
        "at_units": 0,
        "preferred_duration_units": 4,
        "degree": "root",
        "preferred_register_zone": "bass",
        "articulations": [],
    }
    workspace = _apply(
        workspace,
        build_accompaniment_checked_diff(
            document,
            workspace,
            melodic_material_ids,
            {"accompaniments": [{"events": [accompaniment]} for _ in melodic_material_ids]},
        ),
    )
    if transition_ids:
        workspace = _apply(
            workspace,
            build_transition_accompaniment_checked_diff(
                document,
                workspace,
                transition_ids,
                {
                    "transition_accompaniments": [
                        {"events": [accompaniment]} for _ in transition_ids
                    ]
                },
            ),
        )
    ending_ids = tuple(
        sorted(
            material_id
            for material_id, material in materials.items()
            if isinstance(material_id, str)
            and isinstance(material, dict)
            and material["kind"] == "ending"
        )
    )
    workspace = _apply(
        workspace,
        build_ending_checked_diff(document, workspace, ending_ids),
    )
    workspace = _apply(
        workspace,
        build_performance_defaults_checked_diff(document, workspace),
    )
    schedule = performance_operation_schedule(document, workspace)
    inherited = {
        "timing_profile": None,
        "timing_amount": None,
        "dynamics_profile": None,
        "articulation_profile": None,
        "coordination_profile": None,
        "pedal_profile": None,
    }
    for node_ids in (schedule.absolute_node_ids, *schedule.comparative_waves):
        if node_ids:
            workspace = _apply(
                workspace,
                build_performance_checked_diff(
                    document,
                    workspace,
                    node_ids,
                    {"performances": [inherited for _ in node_ids]},
                ),
            )
    return workspace


def test_completed_nested_workspace_builds_existing_ir_without_rounding() -> None:
    document = _approved_script("nested-aba")
    workspace = _completed_workspace(document)

    result = build_workspace_ir(document, workspace)

    assert result.common_units_per_weight == 4
    assert (
        sum(
            node.duration_weight * result.common_units_per_weight
            for node in result.plan.nodes
            if node.duration_weight is not None
        )
        == 88
    )
    assert result.score.divisions == 12
    assert len(result.performance.node_performances) == len(result.plan.nodes)
    connector_ids = result.material_bindings["connector"]
    assert len(connector_ids) == 2
    assert connector_ids[0] != connector_ids[1]


def test_completed_fixed_aba_workspace_builds_one_transition_material() -> None:
    document = _approved_script("fixed-aba")
    workspace = _completed_workspace(document)

    result = build_workspace_ir(document, workspace)

    assert result.common_units_per_weight == 4
    assert len(result.material_bindings["connector"]) == 1
    assert {material.material_id for material in result.score.materials} == {
        "theme",
        "contrast",
        "cadence",
        *result.material_bindings["connector"],
    }


def test_workspace_diagnostics_exclude_values_from_later_generation_phases() -> None:
    document = _approved_script("nested-aba")
    workspace = _completed_workspace(document)
    final = build_workspace_ir(document, workspace)

    melody, score = build_workspace_diagnostic_irs(document, workspace)

    ending_ids = {
        realized
        for material_id, realized_ids in final.material_bindings.items()
        if document["script"]["materials"][material_id]["kind"] == "ending"
        for realized in realized_ids
    }
    for material in melody.score.materials:
        assert material.material_id not in ending_ids
        assert all(note.voice == material.foreground_voice for note in material.notes)
    assert melody.performance.target_duration_ms < score.performance.target_duration_ms
    assert score.score == final.score
    assert all(
        all(
            value is None
            for value in (
                performance.timing_profile,
                performance.timing_amount,
                performance.dynamics_profile,
                performance.articulation_profile,
                performance.coordination_profile,
                performance.pedal_profile,
            )
        )
        for preview in (melody, score)
        for performance in preview.performance.node_performances
    )


def test_workspace_frozen_response_realizes_verified_outputs(tmp_path: Path) -> None:
    document = _approved_script("nested-aba")
    frozen_response = {
        "schema_version": 3,
        "profile": "solo_piano_3m_v1",
        "workspace": workspace_to_dict(_completed_workspace(document)),
    }

    result = realize_solo_piano_3m(document, frozen_response, tmp_path / "output")

    assert result.realized is True
    assert result.realized_duration_ms == 180_000
    assert result.musicxml_path is not None and result.musicxml_path.is_file()
    assert result.smf_path is not None and result.smf_path.is_file()
    assert (tmp_path / "output" / "phase-04-melody.mid").is_file()
    assert (tmp_path / "output" / "phase-06-score.mid").is_file()
    assert result.outcome.artifact_disposition.value == "diagnostic_only"
    diagnostics = json.loads(result.diagnostics_path.read_text(encoding="utf-8"))
    connector_evidence = diagnostics["target_evidence"]["/script/materials/connector/description"][
        "score_spec"
    ]
    assert connector_evidence["realized_material_ids"] == list(
        build_workspace_ir(document, _completed_workspace(document)).material_bindings["connector"]
    )


def test_workspace_trial_bundle_uses_realizer_v5_and_replays_offline(
    tmp_path: Path,
) -> None:
    document = _approved_script("nested-aba")
    frozen_response = {
        "schema_version": 3,
        "profile": "solo_piano_3m_v1",
        "workspace": workspace_to_dict(_completed_workspace(document)),
    }

    created = create_trial_bundle(
        document,
        frozen_response,
        tmp_path / "bundle",
        trial_id="workspace-v5",
    )

    manifest = json.loads(created.manifest_path.read_text(encoding="utf-8"))
    assert manifest["realizer"]["realizer_version"] == 5
    assert manifest["files"]["phase_04_melody"]["path"] == ("artifacts/phase-04-melody.mid")
    assert manifest["files"]["phase_06_score"]["path"] == ("artifacts/phase-06-score.mid")
    replayed = replay_trial_bundle(tmp_path / "bundle", tmp_path / "replay")
    assert replayed.replayed is True
    assert replayed.outcome == created.outcome
    assert replayed.artifact_sha256 == {
        name: manifest["files"][name]["sha256"]
        for name in (
            "musicxml",
            "smf",
            "diagnostics",
            "phase_04_melody",
            "phase_06_score",
        )
    }


def test_workspace_realization_rejects_an_extra_score_value() -> None:
    document = _approved_script("nested-aba")
    workspace = _completed_workspace(document)
    changed = replace(
        workspace,
        harmonies=(*workspace.harmonies, ("unused", workspace.harmonies[0][1])),
    )
    changed = workspace_module._with_current_hashes(changed)

    with pytest.raises(WorkspaceRealizationError) as caught:
        build_workspace_ir(document, changed)

    assert caught.value.issue.path == "/frozen_response/workspace/harmonies/unused"


def test_workspace_realization_rejects_two_relations_for_one_connector_leaf() -> None:
    document = _approved_script("nested-aba")
    workspace = _completed_workspace(document)
    changed_document = deepcopy(document)
    transitions = changed_document["script"]["transitions"]
    transitions["duplicate_a_to_b"] = deepcopy(transitions["a_to_b"])
    changed_document["approval"]["content_sha256"] = content_sha256(changed_document)
    changed = replace(
        workspace,
        approved_script_sha256=content_sha256(changed_document),
        transition_melodies=(
            *workspace.transition_melodies,
            ("duplicate_a_to_b", dict(workspace.transition_melodies)["a_to_b"]),
        ),
        transition_accompaniments=(
            *workspace.transition_accompaniments,
            (
                "duplicate_a_to_b",
                dict(workspace.transition_accompaniments)["a_to_b"],
            ),
        ),
    )
    changed = workspace_module._with_current_hashes(changed)

    with pytest.raises(WorkspaceRealizationError) as caught:
        build_workspace_ir(changed_document, changed)

    assert "multiple transition relations" in caught.value.issue.message


def test_workspace_realization_rejects_an_invalid_script() -> None:
    document = _approved_script("fixed-aba")
    workspace = _completed_workspace(document)
    document["schema_version"] = 999

    with pytest.raises(WorkspaceRealizationError) as caught:
        build_workspace_ir(document, workspace)

    assert caught.value.issue.path == "/schema_version"


def test_workspace_realization_requires_an_approved_script() -> None:
    document = _approved_script("fixed-aba")
    workspace = _completed_workspace(document)
    document["status"] = "draft"
    document["approval"] = None
    document["script"]["requirements"] = {}

    with pytest.raises(WorkspaceRealizationError) as caught:
        build_workspace_ir(document, workspace)

    assert caught.value.issue.path == "/status"


def test_workspace_realization_rejects_a_workspace_from_another_script() -> None:
    document = _approved_script("fixed-aba")
    workspace = replace(
        _completed_workspace(document),
        approved_script_sha256="0" * 64,
    )
    workspace = workspace_module._with_current_hashes(workspace)

    with pytest.raises(WorkspaceRealizationError) as caught:
        build_workspace_ir(document, workspace)

    assert caught.value.issue.code.value == "lineage_mismatch"


def test_workspace_realization_requires_a_completed_plan() -> None:
    document = _approved_script("fixed-aba")
    workspace = replace(_completed_workspace(document), plan=None)
    workspace = workspace_module._with_current_hashes(workspace)

    with pytest.raises(WorkspaceRealizationError) as caught:
        build_workspace_ir(document, workspace)

    assert caught.value.issue.path == "/frozen_response/workspace/plan"


def test_workspace_realization_rejects_conflicting_foreground_voices() -> None:
    document = _approved_script("fixed-aba")
    workspace = _completed_workspace(document)
    material_id, accompaniment = workspace.accompaniments[0]
    changed_accompaniment = replace(accompaniment, foreground_voice="lower")
    changed = replace(
        workspace,
        accompaniments=(
            (material_id, changed_accompaniment),
            *workspace.accompaniments[1:],
        ),
    )
    changed = workspace_module._with_current_hashes(changed)

    with pytest.raises(WorkspaceRealizationError) as caught:
        build_workspace_ir(document, changed)

    assert "foreground voices do not match" in caught.value.issue.message


def test_workspace_realization_requires_workspace_schema_version_four() -> None:
    document = _approved_script("fixed-aba")
    workspace = replace(
        _completed_workspace(document),
        schema_version=3,
        performances=(),
    )
    workspace = workspace_module._with_current_hashes(workspace)

    with pytest.raises(WorkspaceRealizationError) as caught:
        build_workspace_ir(document, workspace)

    assert caught.value.issue.path == "/frozen_response/workspace/schema_version"


def test_workspace_realization_rejects_a_malformed_workspace_record() -> None:
    document = _approved_script("fixed-aba")
    workspace = replace(_completed_workspace(document), schema_version=999)
    workspace = workspace_module._with_current_hashes(workspace)

    with pytest.raises(WorkspaceRealizationError) as caught:
        build_workspace_ir(document, workspace)

    assert caught.value.issue.path == "/frozen_response/workspace"


def test_workspace_realization_rejects_different_lengths_for_reused_material() -> None:
    document = _approved_script("fixed-aba")
    workspace = _completed_workspace(document)
    document["script"]["sections"]["return"]["relative_length"] = 5
    document["approval"]["content_sha256"] = content_sha256(document)
    workspace = replace(
        workspace,
        approved_script_sha256=content_sha256(document),
    )
    workspace = workspace_module._with_current_hashes(workspace)

    with pytest.raises(WorkspaceRealizationError) as caught:
        build_workspace_ir(document, workspace)

    assert "different score-length ratios" in caught.value.issue.message


def test_workspace_realization_rejects_a_score_grid_over_resource_limit() -> None:
    document = _approved_script("fixed-aba")
    workspace = _completed_workspace(document)
    document["script"]["sections"]["statement"]["relative_length"] = 50_000
    document["script"]["sections"]["return"]["relative_length"] = 50_000
    document["approval"]["content_sha256"] = content_sha256(document)
    workspace = replace(
        workspace,
        approved_script_sha256=content_sha256(document),
    )
    workspace = workspace_module._with_current_hashes(workspace)

    with pytest.raises(WorkspaceRealizationError) as caught:
        build_workspace_ir(document, workspace)

    assert "resource limit" in caught.value.issue.message


def test_workspace_realization_rejects_an_unplaced_material() -> None:
    document = _approved_script("fixed-aba")
    workspace = _completed_workspace(document)
    document["script"]["materials"]["unused"] = {
        "description": "配置されない素材。",
        "kind": "theme",
    }
    document["approval"]["content_sha256"] = content_sha256(document)

    def with_unused(items):
        return (*items, ("unused", dict(items)["theme"]))

    workspace = replace(
        workspace,
        approved_script_sha256=content_sha256(document),
        harmonies=with_unused(workspace.harmonies),
        melodies=with_unused(workspace.melodies),
        accompaniments=with_unused(workspace.accompaniments),
    )
    workspace = workspace_module._with_current_hashes(workspace)

    with pytest.raises(WorkspaceRealizationError) as caught:
        build_workspace_ir(document, workspace)

    assert caught.value.issue.path == "/script/materials/unused"
