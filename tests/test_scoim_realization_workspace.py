import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from scoim.realization_workspace import (
    CheckedWorkspaceDiff,
    PatchOperation,
    PerformanceValue,
    PlanValue,
    RealizationWorkspace,
    apply_checked_diff,
    checked_diff_from_dict,
    checked_diff_to_dict,
    create_workspace,
    replay_checked_diffs,
    state_key_sha256,
    upgrade_workspace_v1_to_v2,
    upgrade_workspace_v2_to_v3,
    upgrade_workspace_v3_to_v4,
    workspace_from_dict,
    workspace_to_dict,
)
from scoim.validation import IssueCode, ValidationIssue, content_sha256

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "scoim"


def _fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURE_ROOT / "nested-aba" / name).read_text(encoding="utf-8"))


def test_upgrade_preserves_a_verified_v1_workspace_and_rehashes_v2() -> None:
    legacy = {
        "schema_version": 1,
        "approved_script_sha256": (
            "d720cc45cfb022b48fd273cac2b3e2d5fead585a13c5b940a49350cd289b0de4"
        ),
        "revision": 0,
        "plan": None,
        "harmonies": {},
        "provenance": {},
        "music_content_sha256": "a6589502c7924a5e90dd353f59f5d6272c1742ac56bd783f7ba20d68845acdca",
        "workspace_record_sha256": (
            "bad8dcfb1b63df05f52e7c76526ec79373af42ac95d1d080ad803aa0658adafc"
        ),
    }

    restored = workspace_from_dict(legacy)
    upgraded = upgrade_workspace_v1_to_v2(restored)

    assert workspace_to_dict(restored) == legacy
    assert workspace_to_dict(upgraded)["schema_version"] == 2
    assert upgraded.plan == restored.plan
    assert upgraded.harmonies == restored.harmonies
    assert upgraded.provenance == restored.provenance
    assert upgraded.melodies == ()
    assert upgraded.transition_melodies == ()
    assert upgraded.music_content_sha256 != restored.music_content_sha256
    assert upgraded.workspace_record_sha256 != restored.workspace_record_sha256


def test_upgrade_preserves_a_verified_v2_workspace_and_rehashes_v3() -> None:
    legacy = upgrade_workspace_v1_to_v2(
        workspace_from_dict(
            {
                "schema_version": 1,
                "approved_script_sha256": (
                    "d720cc45cfb022b48fd273cac2b3e2d5fead585a13c5b940a49350cd289b0de4"
                ),
                "revision": 0,
                "plan": None,
                "harmonies": {},
                "provenance": {},
                "music_content_sha256": (
                    "a6589502c7924a5e90dd353f59f5d6272c1742ac56bd783f7ba20d68845acdca"
                ),
                "workspace_record_sha256": (
                    "bad8dcfb1b63df05f52e7c76526ec79373af42ac95d1d080ad803aa0658adafc"
                ),
            }
        )
    )

    upgraded = upgrade_workspace_v2_to_v3(legacy)

    assert upgraded.schema_version == 3
    assert upgraded.plan == legacy.plan
    assert upgraded.harmonies == legacy.harmonies
    assert upgraded.melodies == legacy.melodies
    assert upgraded.transition_melodies == legacy.transition_melodies
    assert upgraded.provenance == legacy.provenance
    assert upgraded.accompaniments == ()
    assert upgraded.transition_accompaniments == ()
    assert upgraded.music_content_sha256 != legacy.music_content_sha256
    assert upgraded.workspace_record_sha256 != legacy.workspace_record_sha256


def test_upgrade_preserves_a_verified_v3_workspace_and_rehashes_v4() -> None:
    legacy = upgrade_workspace_v2_to_v3(
        upgrade_workspace_v1_to_v2(
            workspace_from_dict(
                {
                    "schema_version": 1,
                    "approved_script_sha256": (
                        "d720cc45cfb022b48fd273cac2b3e2d5fead585a13c5b940a49350cd289b0de4"
                    ),
                    "revision": 0,
                    "plan": None,
                    "harmonies": {},
                    "provenance": {},
                    "music_content_sha256": (
                        "a6589502c7924a5e90dd353f59f5d6272c1742ac56bd783f7ba20d68845acdca"
                    ),
                    "workspace_record_sha256": (
                        "bad8dcfb1b63df05f52e7c76526ec79373af42ac95d1d080ad803aa0658adafc"
                    ),
                }
            )
        )
    )

    upgraded = upgrade_workspace_v3_to_v4(legacy)

    assert upgraded.schema_version == 4
    assert upgraded.plan == legacy.plan
    assert upgraded.harmonies == legacy.harmonies
    assert upgraded.melodies == legacy.melodies
    assert upgraded.transition_melodies == legacy.transition_melodies
    assert upgraded.accompaniments == legacy.accompaniments
    assert upgraded.transition_accompaniments == legacy.transition_accompaniments
    assert upgraded.provenance == legacy.provenance
    assert upgraded.performances == ()
    assert upgraded.music_content_sha256 != legacy.music_content_sha256
    assert upgraded.workspace_record_sha256 != legacy.workspace_record_sha256


def test_create_workspace_is_deterministic_and_does_not_modify_the_script() -> None:
    document = _fixture("approved-script.json")
    original = copy.deepcopy(document)

    first = create_workspace(document)
    second = create_workspace(document)

    assert first == second
    assert first.approved_script_sha256 == content_sha256(document)
    assert first.revision == 0
    assert first.plan is None
    assert first.harmonies == ()
    assert first.provenance == ()
    assert first.music_content_sha256 == second.music_content_sha256
    assert first.workspace_record_sha256 == second.workspace_record_sha256
    assert document == original


def test_apply_checked_diff_saves_a_valid_performance_value() -> None:
    workspace = create_workspace(_fixture("approved-script.json"))
    checked_diff = CheckedWorkspaceDiff(
        base_revision=workspace.revision,
        base_workspace_record_sha256=workspace.workspace_record_sha256,
        read_hashes=(("/plan", state_key_sha256(workspace, "/plan")),),
        write_keys=("/performances/whole",),
        patch=(
            PatchOperation(
                "add",
                "/performances/whole",
                {
                    "timing_profile": "savor",
                    "timing_amount": "subtle",
                    "dynamics_profile": "shape",
                    "articulation_profile": "legato",
                    "coordination_profile": "rolled",
                    "pedal_profile": "phrase_legato",
                },
            ),
        ),
        proposal_sha256="e" * 64,
        validation_issues=(),
    )

    result = apply_checked_diff(workspace, checked_diff)

    assert result.issues == ()
    assert result.workspace is not None
    assert result.workspace.performances == (
        (
            "whole",
            PerformanceValue(
                timing_profile="savor",
                timing_amount="subtle",
                dynamics_profile="shape",
                articulation_profile="legato",
                coordination_profile="rolled",
                pedal_profile="phrase_legato",
            ),
        ),
    )
    assert workspace_from_dict(workspace_to_dict(result.workspace)) == result.workspace


def test_apply_checked_diff_updates_only_a_current_validated_write() -> None:
    workspace = create_workspace(_fixture("approved-script.json"))
    proposal_hash = "a" * 64
    checked_diff = CheckedWorkspaceDiff(
        base_revision=workspace.revision,
        base_workspace_record_sha256=workspace.workspace_record_sha256,
        read_hashes=(("/plan", state_key_sha256(workspace, "/plan")),),
        write_keys=("/plan",),
        patch=(
            PatchOperation(
                op="replace",
                path="/plan",
                value={
                    "tonal_center": 9,
                    "mode": "minor",
                    "contrasts": {"b": "brighter and more active"},
                },
            ),
        ),
        proposal_sha256=proposal_hash,
        validation_issues=(),
    )

    result = apply_checked_diff(workspace, checked_diff)

    assert result.issues == ()
    assert result.workspace is not None
    assert result.workspace.revision == 1
    assert result.workspace.plan == PlanValue(
        tonal_center=9,
        mode="minor",
        contrasts=(("b", "brighter and more active"),),
    )
    assert result.workspace.provenance[0][0] == "/plan"
    assert result.workspace.provenance[0][1].proposal_sha256 == proposal_hash
    assert result.workspace.music_content_sha256 != workspace.music_content_sha256
    assert result.workspace.workspace_record_sha256 != workspace.workspace_record_sha256


def test_apply_checked_diff_rejects_a_stale_revision_without_a_new_state() -> None:
    workspace = create_workspace(_fixture("approved-script.json"))
    checked_diff = CheckedWorkspaceDiff(
        base_revision=workspace.revision + 1,
        base_workspace_record_sha256=workspace.workspace_record_sha256,
        read_hashes=(),
        write_keys=("/plan",),
        patch=(PatchOperation("replace", "/plan", None),),
        proposal_sha256="b" * 64,
        validation_issues=(),
    )

    result = apply_checked_diff(workspace, checked_diff)

    assert result.workspace is None
    assert result.issues == (
        ValidationIssue(
            IssueCode.STALE_REVISION,
            "The workspace revision has changed",
            "/revision",
        ),
    )
    assert workspace.revision == 0


def test_apply_checked_diff_rejects_validation_issues_without_a_new_state() -> None:
    workspace = create_workspace(_fixture("approved-script.json"))
    validation_issue = ValidationIssue(
        IssueCode.MODEL_OUTPUT_INVALID,
        "The proposal contains an invalid mode",
        "/response/mode",
    )
    checked_diff = CheckedWorkspaceDiff(
        base_revision=workspace.revision,
        base_workspace_record_sha256=workspace.workspace_record_sha256,
        read_hashes=(),
        write_keys=("/plan",),
        patch=(PatchOperation("replace", "/plan", None),),
        proposal_sha256="c" * 64,
        validation_issues=(validation_issue,),
    )

    result = apply_checked_diff(workspace, checked_diff)

    assert result.workspace is None
    assert result.issues == (validation_issue,)
    assert workspace.revision == 0


def test_replay_checked_diffs_reconstructs_the_same_workspace_hash() -> None:
    initial = create_workspace(_fixture("approved-script.json"))
    checked_diff = CheckedWorkspaceDiff(
        base_revision=0,
        base_workspace_record_sha256=initial.workspace_record_sha256,
        read_hashes=(("/plan", state_key_sha256(initial, "/plan")),),
        write_keys=("/plan",),
        patch=(
            PatchOperation(
                "replace",
                "/plan",
                {"tonal_center": 0, "mode": "major", "contrasts": {}},
            ),
        ),
        proposal_sha256="d" * 64,
        validation_issues=(),
    )
    applied = apply_checked_diff(initial, checked_diff)
    assert applied.workspace is not None

    replayed = replay_checked_diffs(initial, (checked_diff,))

    assert replayed.issues == ()
    assert replayed.workspace is not None
    assert replayed.workspace.workspace_record_sha256 == (applied.workspace.workspace_record_sha256)


def test_replacing_one_harmony_preserves_other_material_value_and_provenance() -> None:
    workspace = create_workspace(_fixture("approved-script.json"))
    plan_diff = CheckedWorkspaceDiff(
        workspace.revision,
        workspace.workspace_record_sha256,
        (("/plan", state_key_sha256(workspace, "/plan")),),
        ("/plan",),
        (
            PatchOperation(
                "replace",
                "/plan",
                {"tonal_center": 0, "mode": "major", "contrasts": {}},
            ),
        ),
        "1" * 64,
        (),
    )
    workspace = _must_apply(workspace, plan_diff)
    for material, root, source in (("theme", 0, "2"), ("contrast", 7, "3")):
        harmony_diff = CheckedWorkspaceDiff(
            workspace.revision,
            workspace.workspace_record_sha256,
            (("/plan", state_key_sha256(workspace, "/plan")),),
            (f"/harmonies/{material}",),
            (
                PatchOperation(
                    "add",
                    f"/harmonies/{material}",
                    [
                        {
                            "duration_units": 4,
                            "root_pitch_class": root,
                            "quality": "major",
                        }
                    ],
                ),
            ),
            source * 64,
            (),
        )
        workspace = _must_apply(workspace, harmony_diff)
    before_harmonies = dict(workspace.harmonies)
    before_provenance = dict(workspace.provenance)
    replace_theme = CheckedWorkspaceDiff(
        workspace.revision,
        workspace.workspace_record_sha256,
        (
            ("/plan", state_key_sha256(workspace, "/plan")),
            ("/harmonies/theme", state_key_sha256(workspace, "/harmonies/theme")),
        ),
        ("/harmonies/theme",),
        (
            PatchOperation(
                "replace",
                "/harmonies/theme",
                [
                    {
                        "duration_units": 4,
                        "root_pitch_class": 5,
                        "quality": "major",
                    }
                ],
            ),
        ),
        "4" * 64,
        (),
    )

    updated = _must_apply(workspace, replace_theme)

    assert dict(updated.harmonies)["contrast"] == before_harmonies["contrast"]
    assert (
        dict(updated.provenance)["/harmonies/contrast"] == before_provenance["/harmonies/contrast"]
    )
    assert dict(updated.harmonies)["theme"] != before_harmonies["theme"]


def test_replacing_the_plan_invalidates_harmonies_that_read_the_old_plan() -> None:
    workspace = _workspace_with_two_harmonies()
    plan_hash = state_key_sha256(workspace, "/plan")
    replace_plan = CheckedWorkspaceDiff(
        workspace.revision,
        workspace.workspace_record_sha256,
        (("/plan", plan_hash),),
        ("/plan",),
        (
            PatchOperation(
                "replace",
                "/plan",
                {"tonal_center": 9, "mode": "minor", "contrasts": {}},
            ),
        ),
        "5" * 64,
        (),
    )

    updated = _must_apply(workspace, replace_plan)

    assert updated.harmonies == ()
    assert set(dict(updated.provenance)) == {"/plan"}


def test_batch_diff_records_dependencies_for_each_written_value() -> None:
    workspace = _workspace_with_two_harmonies()
    plan_read = ("/plan", state_key_sha256(workspace, "/plan"))
    theme_read = (
        "/harmonies/theme",
        state_key_sha256(workspace, "/harmonies/theme"),
    )
    contrast_read = (
        "/harmonies/contrast",
        state_key_sha256(workspace, "/harmonies/contrast"),
    )
    checked_diff = CheckedWorkspaceDiff(
        base_revision=workspace.revision,
        base_workspace_record_sha256=workspace.workspace_record_sha256,
        read_hashes=tuple(sorted((plan_read, theme_read, contrast_read))),
        write_keys=("/melodies/theme", "/melodies/contrast"),
        patch=(
            PatchOperation("add", "/melodies/theme", _melody_value(72)),
            PatchOperation("add", "/melodies/contrast", _melody_value(67)),
        ),
        proposal_sha256="6" * 64,
        validation_issues=(),
        write_read_hashes=(
            ("/melodies/contrast", tuple(sorted((plan_read, contrast_read)))),
            ("/melodies/theme", tuple(sorted((plan_read, theme_read)))),
        ),
    )

    updated = _must_apply(workspace, checked_diff)

    provenance = dict(updated.provenance)
    assert provenance["/melodies/theme"].read_hashes == tuple(sorted((plan_read, theme_read)))
    assert provenance["/melodies/contrast"].read_hashes == tuple(sorted((plan_read, contrast_read)))
    assert set(dict(updated.melodies)) == {"contrast", "theme"}


def test_harmony_change_transitively_invalidates_only_dependent_melodies() -> None:
    workspace = _workspace_with_melodies_and_transitions()
    contrast_melody = dict(workspace.melodies)["contrast"]
    contrast_transition = dict(workspace.transition_melodies)["contrast_loop"]
    contrast_provenance = dict(workspace.provenance)["/melodies/contrast"]
    transition_provenance = dict(workspace.provenance)["/transition_melodies/contrast_loop"]
    replacement = CheckedWorkspaceDiff(
        base_revision=workspace.revision,
        base_workspace_record_sha256=workspace.workspace_record_sha256,
        read_hashes=(("/harmonies/theme", state_key_sha256(workspace, "/harmonies/theme")),),
        write_keys=("/harmonies/theme",),
        patch=(
            PatchOperation(
                "replace",
                "/harmonies/theme",
                [{"duration_units": 4, "root_pitch_class": 5, "quality": "major"}],
            ),
        ),
        proposal_sha256="7" * 64,
        validation_issues=(),
    )

    updated = _must_apply(workspace, replacement)

    assert set(dict(updated.melodies)) == {"contrast"}
    assert set(dict(updated.transition_melodies)) == {"contrast_loop"}
    assert dict(updated.melodies)["contrast"] == contrast_melody
    assert dict(updated.transition_melodies)["contrast_loop"] == contrast_transition
    assert dict(updated.provenance)["/melodies/contrast"] == contrast_provenance
    assert dict(updated.provenance)["/transition_melodies/contrast_loop"] == transition_provenance


def test_diff_rejects_changing_a_dependency_and_its_dependent_together() -> None:
    workspace = _workspace_with_two_harmonies()
    harmony_read = (
        "/harmonies/theme",
        state_key_sha256(workspace, "/harmonies/theme"),
    )
    checked_diff = CheckedWorkspaceDiff(
        workspace.revision,
        workspace.workspace_record_sha256,
        (harmony_read,),
        ("/harmonies/theme", "/melodies/theme"),
        (
            PatchOperation(
                "replace",
                "/harmonies/theme",
                [{"duration_units": 4, "root_pitch_class": 5, "quality": "major"}],
            ),
            PatchOperation("add", "/melodies/theme", _melody_value(72)),
        ),
        "a" * 64,
        (),
        (
            ("/harmonies/theme", (harmony_read,)),
            ("/melodies/theme", (harmony_read,)),
        ),
    )

    result = apply_checked_diff(workspace, checked_diff)

    assert result.workspace is None
    assert result.issues[0].code is IssueCode.PATCH_INVALID


def test_diff_allows_reading_and_replacing_the_same_value() -> None:
    workspace = _workspace_with_two_harmonies()
    harmony_read = (
        "/harmonies/theme",
        state_key_sha256(workspace, "/harmonies/theme"),
    )
    checked_diff = CheckedWorkspaceDiff(
        workspace.revision,
        workspace.workspace_record_sha256,
        (harmony_read,),
        ("/harmonies/theme",),
        (
            PatchOperation(
                "replace",
                "/harmonies/theme",
                [{"duration_units": 4, "root_pitch_class": 5, "quality": "major"}],
            ),
        ),
        "b" * 64,
        (),
        (("/harmonies/theme", (harmony_read,)),),
    )

    updated = _must_apply(workspace, checked_diff)

    assert dict(updated.harmonies)["theme"][0].root_pitch_class == 5


def test_plan_change_transitively_invalidates_all_downstream_values() -> None:
    workspace = _workspace_with_melodies_and_transitions()
    plan_read = ("/plan", state_key_sha256(workspace, "/plan"))
    checked_diff = CheckedWorkspaceDiff(
        workspace.revision,
        workspace.workspace_record_sha256,
        (plan_read,),
        ("/plan",),
        (
            PatchOperation(
                "replace",
                "/plan",
                {"tonal_center": 9, "mode": "minor", "contrasts": {}},
            ),
        ),
        "c" * 64,
        (),
        (("/plan", (plan_read,)),),
    )

    updated = _must_apply(workspace, checked_diff)

    assert updated.harmonies == ()
    assert updated.melodies == ()
    assert updated.transition_melodies == ()
    assert set(dict(updated.provenance)) == {"/plan"}


def test_workspace_and_checked_diff_round_trip_through_json_values() -> None:
    workspace = _workspace_with_two_harmonies()
    checked_diff = CheckedWorkspaceDiff(
        workspace.revision,
        workspace.workspace_record_sha256,
        (("/harmonies/theme", state_key_sha256(workspace, "/harmonies/theme")),),
        ("/harmonies/theme",),
        (PatchOperation("replace", "/harmonies/theme", []),),
        "6" * 64,
        (
            ValidationIssue(
                IssueCode.MODEL_OUTPUT_INVALID,
                "Empty harmony is not allowed",
                "/response/theme",
            ),
        ),
        (
            (
                "/harmonies/theme",
                (("/harmonies/theme", state_key_sha256(workspace, "/harmonies/theme")),),
            ),
        ),
    )

    restored_workspace = workspace_from_dict(json.loads(json.dumps(workspace_to_dict(workspace))))
    restored_diff = checked_diff_from_dict(
        json.loads(json.dumps(checked_diff_to_dict(checked_diff)))
    )

    assert restored_workspace == workspace
    assert restored_diff == checked_diff


def test_checked_diff_restore_rejects_corrupt_per_write_dependencies() -> None:
    workspace = _workspace_with_two_harmonies()
    checked_diff = CheckedWorkspaceDiff(
        workspace.revision,
        workspace.workspace_record_sha256,
        (("/plan", state_key_sha256(workspace, "/plan")),),
        ("/harmonies/theme",),
        (PatchOperation("replace", "/harmonies/theme", []),),
        "6" * 64,
        (),
        (("/harmonies/theme", (("/plan", state_key_sha256(workspace, "/plan")),)),),
    )
    base = checked_diff_to_dict(checked_diff)
    cases: list[dict[str, object]] = []
    invalid_container = copy.deepcopy(base)
    invalid_container["write_read_hashes"] = []
    cases.append(invalid_container)
    invalid_entry = copy.deepcopy(base)
    invalid_entry["write_read_hashes"] = {1: []}
    cases.append(invalid_entry)
    invalid_hash = copy.deepcopy(base)
    invalid_hash["write_read_hashes"] = {"/harmonies/theme": [{}]}
    cases.append(invalid_hash)

    for value in cases:
        with pytest.raises(ValueError):
            checked_diff_from_dict(value)


def test_workspace_applier_rejects_incomplete_or_foreign_per_write_dependencies() -> None:
    workspace = _workspace_with_two_harmonies()
    plan_read = ("/plan", state_key_sha256(workspace, "/plan"))
    valid = CheckedWorkspaceDiff(
        workspace.revision,
        workspace.workspace_record_sha256,
        (plan_read,),
        ("/harmonies/theme",),
        (
            PatchOperation(
                "replace",
                "/harmonies/theme",
                [{"duration_units": 4, "root_pitch_class": 5, "quality": "major"}],
            ),
        ),
        "6" * 64,
        (),
        (("/harmonies/theme", (plan_read,)),),
    )
    cases = (
        replace(valid, write_read_hashes=()),
        replace(
            valid,
            write_read_hashes=(("/harmonies/theme", (plan_read, plan_read)),),
        ),
        replace(
            valid,
            write_read_hashes=(("/harmonies/theme", (("/missing", "0" * 64),)),),
        ),
    )

    for checked_diff in cases:
        result = apply_checked_diff(workspace, checked_diff)
        assert result.workspace is None
        assert result.issues[0].code is IssueCode.PATCH_INVALID


def test_workspace_applier_rejects_an_invalid_harmony_even_if_marked_checked() -> None:
    workspace = _workspace_with_two_harmonies()
    invalid_diff = CheckedWorkspaceDiff(
        workspace.revision,
        workspace.workspace_record_sha256,
        (("/plan", state_key_sha256(workspace, "/plan")),),
        ("/harmonies/theme",),
        (
            PatchOperation(
                "replace",
                "/harmonies/theme",
                [
                    {
                        "duration_units": 0,
                        "root_pitch_class": 12,
                        "quality": "unknown",
                    }
                ],
            ),
        ),
        "7" * 64,
        (),
    )

    result = apply_checked_diff(workspace, invalid_diff)

    assert result.workspace is None
    assert result.issues[0].code is IssueCode.PATCH_INVALID
    assert workspace == _workspace_with_two_harmonies()


def test_workspace_applier_rejects_a_boolean_tonal_center() -> None:
    workspace = create_workspace(_fixture("approved-script.json"))
    invalid_diff = CheckedWorkspaceDiff(
        workspace.revision,
        workspace.workspace_record_sha256,
        (("/plan", state_key_sha256(workspace, "/plan")),),
        ("/plan",),
        (
            PatchOperation(
                "replace",
                "/plan",
                {"tonal_center": True, "mode": "major", "contrasts": {}},
            ),
        ),
        "8" * 64,
        (),
    )

    result = apply_checked_diff(workspace, invalid_diff)

    assert result.workspace is None
    assert result.issues[0].code is IssueCode.PATCH_INVALID


def test_workspace_restore_rejects_corrupt_identity_hash_and_provenance() -> None:
    base = workspace_to_dict(_workspace_with_two_harmonies())
    cases: list[dict[str, object]] = []
    missing_field = copy.deepcopy(base)
    missing_field.pop("schema_version")
    cases.append(missing_field)
    extra_field = copy.deepcopy(base)
    extra_field["unknown"] = None
    cases.append(extra_field)
    invalid_provenance = copy.deepcopy(base)
    invalid_provenance["provenance"] = []
    cases.append(invalid_provenance)
    invalid_source = copy.deepcopy(base)
    invalid_source["provenance"]["/plan"] = 1  # type: ignore[index]
    cases.append(invalid_source)
    invalid_reads = copy.deepcopy(base)
    invalid_reads["provenance"]["/plan"]["read_hashes"] = "bad"  # type: ignore[index]
    cases.append(invalid_reads)
    invalid_read = copy.deepcopy(base)
    invalid_read["provenance"]["/plan"]["read_hashes"] = [{}]  # type: ignore[index]
    cases.append(invalid_read)
    invalid_proposal = copy.deepcopy(base)
    invalid_proposal["provenance"]["/plan"]["proposal_sha256"] = 1  # type: ignore[index]
    cases.append(invalid_proposal)
    invalid_revision = copy.deepcopy(base)
    invalid_revision["revision"] = -1
    cases.append(invalid_revision)
    invalid_harmonies = copy.deepcopy(base)
    invalid_harmonies["harmonies"] = []
    cases.append(invalid_harmonies)
    invalid_melodies = copy.deepcopy(base)
    invalid_melodies["melodies"] = []
    cases.append(invalid_melodies)
    invalid_melody_entry = copy.deepcopy(base)
    invalid_melody_entry["melodies"] = {"theme": []}
    cases.append(invalid_melody_entry)
    invalid_melody_fields = copy.deepcopy(base)
    invalid_melody_fields["melodies"] = {"theme": {"foreground_voice": "upper"}}
    cases.append(invalid_melody_fields)
    invalid_melody_voice = copy.deepcopy(base)
    invalid_melody_voice["melodies"] = {
        "theme": {
            "foreground_voice": "middle",
            "notes": [{"at_units": 0, "duration_units": 1, "pitch": 60}],
        }
    }
    cases.append(invalid_melody_voice)
    invalid_melody_note = copy.deepcopy(base)
    invalid_melody_note["melodies"] = {"theme": {"foreground_voice": "upper", "notes": [{}]}}
    cases.append(invalid_melody_note)
    invalid_melody_range = copy.deepcopy(base)
    invalid_melody_range["melodies"] = {
        "theme": {
            "foreground_voice": "upper",
            "notes": [{"at_units": 0, "duration_units": 1, "pitch": 109}],
        }
    }
    cases.append(invalid_melody_range)
    invalid_music_hash = copy.deepcopy(base)
    invalid_music_hash["music_content_sha256"] = "0" * 64
    cases.append(invalid_music_hash)
    invalid_record_hash = copy.deepcopy(base)
    invalid_record_hash["workspace_record_sha256"] = "0" * 64
    cases.append(invalid_record_hash)

    for value in cases:
        with pytest.raises(ValueError):
            workspace_from_dict(value)


def test_upgrade_rejects_an_already_current_workspace() -> None:
    with pytest.raises(ValueError, match="version 1"):
        upgrade_workspace_v1_to_v2(create_workspace(_fixture("approved-script.json")))
    with pytest.raises(ValueError, match="version 3"):
        upgrade_workspace_v3_to_v4(create_workspace(_fixture("approved-script.json")))


@pytest.mark.parametrize(
    ("performances", "message"),
    [
        ([], "must be an object"),
        ({1: {}}, "map node IDs"),
        ({"whole": {"timing_profile": None}}, "all supported fields"),
        (
            {
                "whole": {
                    "timing_profile": "impossible",
                    "timing_amount": None,
                    "dynamics_profile": None,
                    "articulation_profile": None,
                    "coordination_profile": None,
                    "pedal_profile": None,
                }
            },
            "timing_profile is unsupported",
        ),
        (
            {
                "whole": {
                    "timing_profile": None,
                    "timing_amount": "subtle",
                    "dynamics_profile": None,
                    "articulation_profile": None,
                    "coordination_profile": None,
                    "pedal_profile": None,
                }
            },
            "timing amount requires a timing profile",
        ),
    ],
)
def test_workspace_rejects_malformed_performance_values(performances: object, message: str) -> None:
    record = workspace_to_dict(create_workspace(_fixture("approved-script.json")))
    record["performances"] = performances

    with pytest.raises(ValueError, match=message):
        workspace_from_dict(record)


def test_workspace_applier_rejects_stale_reads_and_invalid_patch_shape() -> None:
    workspace = create_workspace(_fixture("approved-script.json"))
    valid = CheckedWorkspaceDiff(
        workspace.revision,
        workspace.workspace_record_sha256,
        (("/plan", state_key_sha256(workspace, "/plan")),),
        ("/plan",),
        (PatchOperation("replace", "/plan", None),),
        "9" * 64,
        (),
    )
    cases = (
        replace(valid, base_workspace_record_sha256="0" * 64),
        replace(valid, read_hashes=(("/missing", "0" * 64),)),
        replace(valid, read_hashes=(("/plan", "0" * 64),)),
        replace(valid, write_keys=("/harmonies/theme",)),
        replace(
            valid,
            write_keys=("/missing",),
            patch=(PatchOperation("remove", "/missing", None),),
        ),
    )

    for checked_diff in cases:
        result = apply_checked_diff(workspace, checked_diff)
        assert result.workspace is None

    replayed = replay_checked_diffs(workspace, (cases[0],))
    assert replayed.workspace is None


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("/plan", {"tonal_center": 0, "mode": "major"}),
        ("/plan", {"tonal_center": 0, "mode": "other", "contrasts": {}}),
        ("/plan", {"tonal_center": 0, "mode": "major", "contrasts": []}),
        ("/harmonies/theme", {}),
        ("/harmonies/theme", [{}]),
        (
            "/harmonies/theme",
            [{"duration_units": 1, "root_pitch_class": 12, "quality": "major"}],
        ),
        (
            "/harmonies/theme",
            [{"duration_units": 1, "root_pitch_class": 0, "quality": "other"}],
        ),
        ("/harmonies/theme", []),
    ],
)
def test_workspace_applier_rechecks_plan_and_harmony_shapes(path: str, value: object) -> None:
    workspace = create_workspace(_fixture("approved-script.json"))
    checked_diff = CheckedWorkspaceDiff(
        workspace.revision,
        workspace.workspace_record_sha256,
        (),
        (path,),
        (PatchOperation("replace" if path == "/plan" else "add", path, value),),
        "e" * 64,
        (),
    )

    result = apply_checked_diff(workspace, checked_diff)

    assert result.workspace is None
    assert result.issues[0].code is IssueCode.PATCH_INVALID


def _workspace_with_two_harmonies() -> RealizationWorkspace:
    workspace = create_workspace(_fixture("approved-script.json"))
    plan_diff = CheckedWorkspaceDiff(
        workspace.revision,
        workspace.workspace_record_sha256,
        (("/plan", state_key_sha256(workspace, "/plan")),),
        ("/plan",),
        (
            PatchOperation(
                "replace",
                "/plan",
                {"tonal_center": 0, "mode": "major", "contrasts": {}},
            ),
        ),
        "1" * 64,
        (),
    )
    workspace = _must_apply(workspace, plan_diff)
    for material, root, source in (("theme", 0, "2"), ("contrast", 7, "3")):
        harmony_diff = CheckedWorkspaceDiff(
            workspace.revision,
            workspace.workspace_record_sha256,
            (("/plan", state_key_sha256(workspace, "/plan")),),
            (f"/harmonies/{material}",),
            (
                PatchOperation(
                    "add",
                    f"/harmonies/{material}",
                    [
                        {
                            "duration_units": 4,
                            "root_pitch_class": root,
                            "quality": "major",
                        }
                    ],
                ),
            ),
            source * 64,
            (),
        )
        workspace = _must_apply(workspace, harmony_diff)
    return workspace


def _melody_value(pitch: int) -> dict[str, object]:
    return {
        "foreground_voice": "upper",
        "notes": [{"at_units": 0, "duration_units": 4, "pitch": pitch}],
    }


def _workspace_with_melodies_and_transitions() -> RealizationWorkspace:
    workspace = _workspace_with_two_harmonies()
    plan_read = ("/plan", state_key_sha256(workspace, "/plan"))
    theme_read = ("/harmonies/theme", state_key_sha256(workspace, "/harmonies/theme"))
    contrast_read = (
        "/harmonies/contrast",
        state_key_sha256(workspace, "/harmonies/contrast"),
    )
    melody_diff = CheckedWorkspaceDiff(
        workspace.revision,
        workspace.workspace_record_sha256,
        tuple(sorted((plan_read, theme_read, contrast_read))),
        ("/melodies/theme", "/melodies/contrast"),
        (
            PatchOperation("add", "/melodies/theme", _melody_value(72)),
            PatchOperation("add", "/melodies/contrast", _melody_value(67)),
        ),
        "8" * 64,
        (),
        (
            ("/melodies/contrast", tuple(sorted((plan_read, contrast_read)))),
            ("/melodies/theme", tuple(sorted((plan_read, theme_read)))),
        ),
    )
    workspace = _must_apply(workspace, melody_diff)
    theme_melody_read = (
        "/melodies/theme",
        state_key_sha256(workspace, "/melodies/theme"),
    )
    contrast_melody_read = (
        "/melodies/contrast",
        state_key_sha256(workspace, "/melodies/contrast"),
    )
    transition_diff = CheckedWorkspaceDiff(
        workspace.revision,
        workspace.workspace_record_sha256,
        tuple(sorted((theme_melody_read, contrast_melody_read))),
        (
            "/transition_melodies/theme_to_contrast",
            "/transition_melodies/contrast_loop",
        ),
        (
            PatchOperation(
                "add",
                "/transition_melodies/theme_to_contrast",
                _melody_value(69),
            ),
            PatchOperation(
                "add",
                "/transition_melodies/contrast_loop",
                _melody_value(65),
            ),
        ),
        "9" * 64,
        (),
        (
            (
                "/transition_melodies/contrast_loop",
                (contrast_melody_read,),
            ),
            (
                "/transition_melodies/theme_to_contrast",
                tuple(sorted((theme_melody_read, contrast_melody_read))),
            ),
        ),
    )
    return _must_apply(workspace, transition_diff)


def _must_apply(
    workspace: RealizationWorkspace, checked_diff: CheckedWorkspaceDiff
) -> RealizationWorkspace:
    result = apply_checked_diff(workspace, checked_diff)
    assert result.workspace is not None, result.issues
    return result.workspace
