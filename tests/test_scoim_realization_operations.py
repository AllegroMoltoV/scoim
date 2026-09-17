import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import scoim.realization_operations as operation_module
import scoim.realization_workspace as workspace_module
from llm_musical_composer.run_state import InterruptedAttemptError, RunStore
from scoim.proposal import ProposalRun
from scoim.realization_operations import (
    accompaniment_response_schema,
    advance_accompaniment_realization_run,
    advance_ending_realization_run,
    advance_melody_realization_run,
    advance_minimal_realization_run,
    advance_performance_realization_run,
    build_accompaniment_checked_diff,
    build_ending_checked_diff,
    build_harmony_checked_diff,
    build_melody_checked_diff,
    build_performance_checked_diff,
    build_performance_defaults_checked_diff,
    build_plan_checked_diff,
    build_transition_accompaniment_checked_diff,
    build_transition_melody_checked_diff,
    harmony_response_schema,
    initialize_accompaniment_realization_run,
    initialize_ending_realization_run,
    initialize_melody_realization_run,
    initialize_minimal_realization_run,
    initialize_performance_realization_run,
    melody_response_schema,
    performance_response_schema,
    plan_response_schema,
    replace_pending_review_with_response,
    transition_accompaniment_contexts,
    transition_accompaniment_response_schema,
    transition_melody_contexts,
    transition_melody_response_schema,
)
from scoim.realization_run import RealizationRunStore, approve_pending_review
from scoim.realization_workspace import (
    CheckedWorkspaceDiff,
    PatchOperation,
    PerformanceValue,
    RealizationWorkspace,
    apply_checked_diff,
    create_workspace,
    state_key_sha256,
)
from scoim.solo_piano_performance import (
    performance_contexts,
    performance_operation_schedule,
    performance_piece_plan,
    performance_read_keys,
)
from scoim.validation import IssueCode, content_sha256

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "scoim"


def _approved_script(fixture: str = "nested-aba") -> dict[str, object]:
    return json.loads((FIXTURE_ROOT / fixture / "approved-script.json").read_text(encoding="utf-8"))


def test_plan_operation_maps_ordered_music_choices_to_python_owned_section_keys() -> None:
    document = _approved_script()
    workspace = create_workspace(document)
    schema = plan_response_schema(document)
    assert "section_id" not in json.dumps(schema)

    checked_diff = build_plan_checked_diff(
        document,
        workspace,
        {
            "tonal_center": 9,
            "mode": "minor",
            "contrast_descriptions": [
                "The contrasting region opens the register",
                "The first contrasting leaf introduces motion",
            ],
        },
    )
    applied = apply_checked_diff(workspace, checked_diff)

    assert checked_diff.validation_issues == ()
    assert checked_diff.write_keys == ("/plan",)
    assert applied.workspace is not None
    assert applied.workspace.plan is not None
    assert dict(applied.workspace.plan.contrasts) == {
        "b": "The contrasting region opens the register",
        "b1": "The first contrasting leaf introduces motion",
    }


@pytest.mark.parametrize("fixture", ["fixed-aba", "nested-aba"])
def test_batch_split_and_reordered_harmony_produce_the_same_music_content(
    fixture: str,
) -> None:
    document = _approved_script(fixture)
    initial = _with_plan(document)
    harmony_by_material = {
        "connector": [{"duration_units": 2, "root_pitch_class": 7, "quality": "major"}],
        "contrast": [
            {"duration_units": 4, "root_pitch_class": 5, "quality": "major"},
            {"duration_units": 4, "root_pitch_class": 7, "quality": "major"},
        ],
        "theme": [{"duration_units": 4, "root_pitch_class": 0, "quality": "major"}],
    }
    targets = tuple(sorted(harmony_by_material))
    schema_text = json.dumps(harmony_response_schema(document, targets))
    assert "material_id" not in schema_text
    assert "cadence" not in schema_text

    batch_diff = build_harmony_checked_diff(
        document,
        initial,
        targets,
        {"harmonies": [harmony_by_material[target] for target in targets]},
    )
    batch = _must_apply(initial, batch_diff)
    split = initial
    for target in reversed(targets):
        split_diff = build_harmony_checked_diff(
            document,
            split,
            (target,),
            {"harmonies": [harmony_by_material[target]]},
        )
        split = _must_apply(split, split_diff)

    assert batch.music_content_sha256 == split.music_content_sha256
    assert batch.workspace_record_sha256 != split.workspace_record_sha256
    assert set(dict(batch.harmonies)) == set(harmony_by_material)
    if fixture == "nested-aba":
        script = document["script"]
        assert isinstance(script, dict)
        placements = script["placements"]
        assert isinstance(placements, dict)
        connector_placements = [
            placement
            for placement in placements.values()
            if isinstance(placement, dict) and placement["material_id"] == "connector"
        ]
        assert len(connector_placements) == 2
        assert len(dict(batch.harmonies)["connector"]) == 1


def test_minimal_run_completes_plan_and_harmony_in_two_model_calls(
    tmp_path: Path,
) -> None:
    class QueueRunner:
        def __init__(self, responses: list[dict[str, object]]) -> None:
            self.responses = responses
            self.calls = 0

        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            response = self.responses[self.calls]
            self.calls += 1
            return ProposalRun(
                provider="fake",
                model="test-model",
                model_settings={},
                started=True,
                terminal_state="completed",
                raw_response=json.dumps(response).encode(),
                events=b'{"type":"thread.started"}\n{"type":"turn.completed"}\n',
                stderr=b"",
                issues=(),
            )

    document = _approved_script()
    runner = QueueRunner(
        [
            {
                "tonal_center": 0,
                "mode": "major",
                "contrast_descriptions": ["Open the register", "Add motion"],
            },
            {
                "harmonies": [
                    [{"duration_units": 2, "root_pitch_class": 7, "quality": "major"}],
                    [{"duration_units": 4, "root_pitch_class": 5, "quality": "major"}],
                    [{"duration_units": 4, "root_pitch_class": 0, "quality": "major"}],
                ]
            },
        ]
    )
    run_dir = tmp_path / "run"
    initialize_minimal_realization_run(
        run_dir,
        document,
        review_after=frozenset(),
        model="test-model",
        max_calls=2,
    )

    status = advance_minimal_realization_run(run_dir, runner)

    assert status.status == "completed"
    assert status.completed_operations == ("overall-plan", "harmony-all")
    assert runner.calls == 2
    assert set(dict(status.workspace.harmonies)) == {"connector", "contrast", "theme"}


def test_selective_review_resumes_harmony_with_a_new_runner(tmp_path: Path) -> None:
    class OneResponseRunner:
        def __init__(self, response: dict[str, object]) -> None:
            self.response = response
            self.calls = 0

        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            self.calls += 1
            return ProposalRun(
                "fake",
                "test-model",
                {},
                True,
                "completed",
                json.dumps(self.response).encode(),
                b'{"type":"thread.started"}\n{"type":"turn.completed"}\n',
                b"",
                (),
            )

    document = _approved_script()
    run_dir = tmp_path / "run"
    initialize_minimal_realization_run(
        run_dir,
        document,
        review_after=frozenset({"overall-plan"}),
        model="test-model",
        max_calls=2,
    )
    plan_runner = OneResponseRunner(
        {
            "tonal_center": 0,
            "mode": "major",
            "contrast_descriptions": ["Open the register", "Add motion"],
        }
    )

    waiting = advance_minimal_realization_run(run_dir, plan_runner)

    assert waiting.status == "awaiting_review"
    assert plan_runner.calls == 1
    approval = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; import sys; "
                "from scoim.realization_run import approve_pending_review; "
                "approve_pending_review(Path(sys.argv[1]), actor='human')"
            ),
            str(run_dir),
        ],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert approval.returncode == 0, approval.stderr
    harmony_runner = OneResponseRunner(
        {
            "harmonies": [
                [{"duration_units": 2, "root_pitch_class": 7, "quality": "major"}],
                [{"duration_units": 4, "root_pitch_class": 5, "quality": "major"}],
                [{"duration_units": 4, "root_pitch_class": 0, "quality": "major"}],
            ]
        }
    )

    completed = advance_minimal_realization_run(run_dir, harmony_runner)

    assert completed.status == "completed"
    assert completed.completed_operations == ("overall-plan", "harmony-all")
    assert harmony_runner.calls == 1


def test_human_replacement_uses_the_same_plan_normalizer_and_checker(
    tmp_path: Path,
) -> None:
    class OneResponseRunner:
        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            return ProposalRun(
                "fake",
                "test-model",
                {},
                True,
                "completed",
                json.dumps(
                    {
                        "tonal_center": 0,
                        "mode": "major",
                        "contrast_descriptions": ["Open", "Move"],
                    }
                ).encode(),
                b'{"type":"thread.started"}\n{"type":"turn.completed"}\n',
                b"",
                (),
            )

    document = _approved_script()
    run_dir = tmp_path / "run"
    initialize_minimal_realization_run(
        run_dir,
        document,
        review_after=frozenset({"overall-plan"}),
        model="test-model",
        max_calls=2,
    )
    assert advance_minimal_realization_run(run_dir, OneResponseRunner()).status == (
        "awaiting_review"
    )

    replaced = replace_pending_review_with_response(
        run_dir,
        {
            "tonal_center": 9,
            "mode": "minor",
            "contrast_descriptions": ["Darken", "Increase tension"],
        },
        actor="human",
    )

    assert replaced.status == "running"
    assert replaced.completed_operations == ("overall-plan",)
    assert replaced.workspace.plan is not None
    assert replaced.workspace.plan.tonal_center == 9
    assert replaced.workspace.plan.mode == "minor"


def test_invalid_plan_and_harmony_responses_remain_unapplied() -> None:
    document = _approved_script()
    empty = create_workspace(document)
    invalid_plan = build_plan_checked_diff(
        document,
        empty,
        {"tonal_center": 0, "mode": "major", "contrast_descriptions": []},
    )
    assert invalid_plan.validation_issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert apply_checked_diff(empty, invalid_plan).workspace is None

    planned = _with_plan(document)
    invalid_harmony = build_harmony_checked_diff(
        document,
        planned,
        ("theme",),
        {"harmonies": []},
    )
    assert invalid_harmony.validation_issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert apply_checked_diff(planned, invalid_harmony).workspace is None


def test_harmony_operation_rejects_missing_plan_unknown_duplicate_and_ending_targets() -> None:
    document = _approved_script()
    empty = create_workspace(document)
    without_plan = build_harmony_checked_diff(
        document,
        empty,
        ("theme",),
        {"harmonies": [[]]},
    )
    assert without_plan.validation_issues[0].path == "/plan"

    planned = _with_plan(document)
    for targets, code in (
        ((), IssueCode.MODEL_OUTPUT_INVALID),
        (("theme", "theme"), IssueCode.MODEL_OUTPUT_INVALID),
        (("missing",), IssueCode.NOT_FOUND),
        (("cadence",), IssueCode.MODEL_OUTPUT_INVALID),
    ):
        diff = build_harmony_checked_diff(
            document,
            planned,
            targets,
            {"harmonies": []},
        )
        assert diff.validation_issues[0].code is code
        with pytest.raises(ValueError):
            harmony_response_schema(document, targets)


def test_operation_context_rejects_draft_invalid_and_mismatched_scripts() -> None:
    document = _approved_script()
    workspace = create_workspace(document)
    invalid = json.loads(json.dumps(document))
    invalid["unknown"] = True
    assert build_plan_checked_diff(invalid, workspace, {}).validation_issues[0].code is (
        IssueCode.SCHEMA_INVALID
    )

    draft = json.loads(json.dumps(document))
    draft["status"] = "draft"
    draft["approval"] = None
    draft["script"]["requirements"] = {}
    assert build_plan_checked_diff(draft, workspace, {}).validation_issues[0].path == ("/status")

    mismatched = replace(workspace, approved_script_sha256="0" * 64)
    assert build_plan_checked_diff(document, mismatched, {}).validation_issues[0].code is (
        IssueCode.LINEAGE_MISMATCH
    )


def test_harmony_operation_replaces_an_existing_target() -> None:
    document = _approved_script()
    workspace = _with_plan(document)
    first = build_harmony_checked_diff(
        document,
        workspace,
        ("theme",),
        {"harmonies": [[{"duration_units": 4, "root_pitch_class": 0, "quality": "major"}]]},
    )
    workspace = _must_apply(workspace, first)

    second = build_harmony_checked_diff(
        document,
        workspace,
        ("theme",),
        {"harmonies": [[{"duration_units": 2, "root_pitch_class": 5, "quality": "major"}]]},
    )
    updated = _must_apply(workspace, second)

    assert second.patch[0].op == "replace"
    assert dict(updated.harmonies)["theme"][0].root_pitch_class == 5


def test_melody_operation_maps_positional_values_to_local_material_dependencies() -> None:
    document = _approved_script()
    workspace = _with_harmonies(document)
    targets = ("theme", "contrast")
    schema = melody_response_schema(document, workspace, targets)
    response = {
        "melodies": [
            {
                "foreground_voice": "upper",
                "notes": [{"at_units": 0, "duration_units": 4, "pitch": 72}],
            },
            {
                "foreground_voice": "upper",
                "notes": [{"at_units": 0, "duration_units": 4, "pitch": 67}],
            },
        ]
    }

    checked_diff = build_melody_checked_diff(document, workspace, targets, response)
    updated = _must_apply(workspace, checked_diff)

    assert "material_id" not in json.dumps(schema)
    assert [key for key, _ in updated.melodies] == ["contrast", "theme"]
    dependencies = dict(checked_diff.write_read_hashes or ())
    assert {key for key, _ in dependencies["/melodies/theme"]} == {
        "/approved_script_sha256",
        "/plan",
        "/harmonies/theme",
    }
    assert "/harmonies/contrast" not in {key for key, _ in dependencies["/melodies/theme"]}


def test_shared_connector_gets_separate_transition_values_with_ordered_boundaries() -> None:
    document = _approved_script()
    workspace = _with_melodies(document)
    targets = ("a_to_b", "b_to_a")

    contexts = transition_melody_contexts(document, workspace, targets)
    schema = transition_melody_response_schema(document, workspace, targets)
    checked_diff = build_transition_melody_checked_diff(
        document,
        workspace,
        targets,
        {
            "transition_melodies": [
                {
                    "foreground_voice": "upper",
                    "notes": [{"at_units": 0, "duration_units": 4, "pitch": 69}],
                },
                {
                    "foreground_voice": "upper",
                    "notes": [{"at_units": 0, "duration_units": 4, "pitch": 71}],
                },
            ]
        },
    )
    updated = _must_apply(workspace, checked_diff)

    assert contexts[0]["connector_material_key"] == "connector"
    assert contexts[1]["connector_material_key"] == "connector"
    assert contexts[0]["source_material_key"] == "theme"
    assert contexts[0]["target_material_key"] == "contrast"
    assert contexts[1]["source_material_key"] == "contrast"
    assert contexts[1]["target_material_key"] == "theme"
    assert "transition_id" not in json.dumps(schema)
    assert (
        dict(updated.transition_melodies)["a_to_b"] != dict(updated.transition_melodies)["b_to_a"]
    )


def test_melody_run_reuses_harmony_workspace_and_completes_in_two_calls(
    tmp_path: Path,
) -> None:
    class QueueRunner:
        def __init__(self) -> None:
            self.responses = [
                {
                    "melodies": [
                        {
                            "foreground_voice": "upper",
                            "notes": [{"at_units": 0, "duration_units": 4, "pitch": 67}],
                        },
                        {
                            "foreground_voice": "upper",
                            "notes": [{"at_units": 0, "duration_units": 4, "pitch": 72}],
                        },
                    ]
                },
                {
                    "transition_melodies": [
                        {
                            "foreground_voice": "upper",
                            "notes": [{"at_units": 0, "duration_units": 4, "pitch": 69}],
                        },
                        {
                            "foreground_voice": "upper",
                            "notes": [{"at_units": 0, "duration_units": 4, "pitch": 71}],
                        },
                    ]
                },
            ]
            self.calls = 0

        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            response = self.responses[self.calls]
            self.calls += 1
            return ProposalRun(
                "fake",
                "test-model",
                {},
                True,
                "completed",
                json.dumps(response).encode(),
                b'{"type":"thread.started"}\n{"type":"turn.completed"}\n',
                b"",
                (),
            )

    document = _approved_script()
    starting = _with_harmonies(document)
    runner = QueueRunner()
    run_dir = tmp_path / "run"
    initialize_melody_realization_run(
        run_dir,
        document,
        starting,
        source_workspace_record_sha256="a" * 64,
        review_after=frozenset(),
        model="test-model",
        max_calls=2,
    )

    completed = advance_melody_realization_run(run_dir, runner)

    assert completed.status == "completed"
    assert completed.completed_operations == ("melody-all", "transition-melody-all")
    assert set(dict(completed.workspace.melodies)) == {"contrast", "theme"}
    assert set(dict(completed.workspace.transition_melodies)) == {"a_to_b", "b_to_a"}
    assert runner.calls == 2


def test_melody_run_omits_transition_operation_when_script_has_no_transitions(
    tmp_path: Path,
) -> None:
    document = _approved_script("fixed-aba")
    script = document["script"]
    assert isinstance(script, dict)
    script["transitions"] = {}
    approval = document["approval"]
    assert isinstance(approval, dict)
    approval["content_sha256"] = content_sha256(document)
    workspace = _with_harmonies(document)

    run_dir = tmp_path / "melody-no-transitions"
    initialize_melody_realization_run(
        run_dir,
        document,
        workspace,
        source_workspace_record_sha256=workspace.workspace_record_sha256,
        review_after=frozenset(),
        model="fixed-model",
        max_calls=1,
    )

    spec = RunStore(run_dir, max_calls=1).read_spec()
    assert [operation["instance_id"] for operation in spec["operations"]] == ["melody-all"]


def test_melody_run_initialization_requires_a_current_workspace(tmp_path: Path) -> None:
    document = _approved_script()
    legacy_workspace = replace(_with_harmonies(document), schema_version=1)

    with pytest.raises(ValueError, match="version 2"):
        initialize_melody_realization_run(
            tmp_path / "legacy",
            document,
            legacy_workspace,
            source_workspace_record_sha256=legacy_workspace.workspace_record_sha256,
            review_after=frozenset(),
            model="fixed-model",
            max_calls=2,
        )


def test_melody_schema_rejects_invalid_operation_targets() -> None:
    document = _approved_script()
    current = _with_harmonies(document)
    cases = (
        (current, ()),
        (create_workspace(document), ("theme",)),
        (current, ("missing",)),
        (current, ("connector",)),
    )

    for workspace, targets in cases:
        with pytest.raises(ValueError):
            melody_response_schema(document, workspace, targets)


def test_transition_context_rejects_an_empty_target_set() -> None:
    document = _approved_script()

    with pytest.raises(ValueError, match="non-empty"):
        transition_melody_contexts(document, _with_melodies(document), ())


@pytest.mark.parametrize(
    ("notes", "message"),
    [
        ([{"at_units": 3, "duration_units": 2, "pitch": 72}], "exceeds"),
        (
            [
                {"at_units": 0, "duration_units": 3, "pitch": 72},
                {"at_units": 2, "duration_units": 2, "pitch": 72},
            ],
            "overlap",
        ),
    ],
)
def test_melody_operation_rejects_invalid_timing(notes: list[dict[str, int]], message: str) -> None:
    document = _approved_script()
    workspace = _with_harmonies(document)

    checked_diff = build_melody_checked_diff(
        document,
        workspace,
        ("theme",),
        {"melodies": [{"foreground_voice": "upper", "notes": notes}]},
    )

    assert checked_diff.validation_issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert message in checked_diff.validation_issues[0].message


def test_melody_batch_split_and_reordered_values_have_the_same_music_hash() -> None:
    document = _approved_script()
    initial = _with_harmonies(document)
    values = {
        "contrast": {
            "foreground_voice": "upper",
            "notes": [{"at_units": 0, "duration_units": 4, "pitch": 67}],
        },
        "theme": {
            "foreground_voice": "upper",
            "notes": [{"at_units": 0, "duration_units": 4, "pitch": 72}],
        },
    }
    batch_targets = ("contrast", "theme")
    batch = _must_apply(
        initial,
        build_melody_checked_diff(
            document,
            initial,
            batch_targets,
            {"melodies": [values[target] for target in batch_targets]},
        ),
    )
    split = initial
    for target in reversed(batch_targets):
        split = _must_apply(
            split,
            build_melody_checked_diff(
                document,
                split,
                (target,),
                {"melodies": [values[target]]},
            ),
        )

    assert batch.music_content_sha256 == split.music_content_sha256
    assert batch.workspace_record_sha256 != split.workspace_record_sha256


def test_transition_batch_split_and_reordered_values_have_the_same_music_hash() -> None:
    document = _approved_script()
    initial = _with_melodies(document)
    values = {
        "a_to_b": {
            "foreground_voice": "upper",
            "notes": [{"at_units": 0, "duration_units": 4, "pitch": 69}],
        },
        "b_to_a": {
            "foreground_voice": "upper",
            "notes": [{"at_units": 0, "duration_units": 4, "pitch": 71}],
        },
    }
    batch_targets = ("a_to_b", "b_to_a")
    batch = _must_apply(
        initial,
        build_transition_melody_checked_diff(
            document,
            initial,
            batch_targets,
            {"transition_melodies": [values[target] for target in batch_targets]},
        ),
    )
    split = initial
    for target in reversed(batch_targets):
        split = _must_apply(
            split,
            build_transition_melody_checked_diff(
                document,
                split,
                (target,),
                {"transition_melodies": [values[target]]},
            ),
        )

    assert batch.music_content_sha256 == split.music_content_sha256
    assert batch.workspace_record_sha256 != split.workspace_record_sha256


def test_melody_review_resumes_in_another_process_without_repeating_the_call(
    tmp_path: Path,
) -> None:
    class OneResponseRunner:
        def __init__(self, response: dict[str, object]) -> None:
            self.response = response
            self.calls = 0

        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            self.calls += 1
            return ProposalRun(
                "fake",
                "test-model",
                {},
                True,
                "completed",
                json.dumps(self.response).encode(),
                b'{"type":"thread.started"}\n{"type":"turn.completed"}\n',
                b"",
                (),
            )

    document = _approved_script()
    run_dir = tmp_path / "run"
    initialize_melody_realization_run(
        run_dir,
        document,
        _with_harmonies(document),
        source_workspace_record_sha256="b" * 64,
        review_after=frozenset({"melody-all"}),
        model="test-model",
        max_calls=2,
    )
    melody_runner = OneResponseRunner(
        {
            "melodies": [
                {
                    "foreground_voice": "upper",
                    "notes": [{"at_units": 0, "duration_units": 4, "pitch": 67}],
                },
                {
                    "foreground_voice": "upper",
                    "notes": [{"at_units": 0, "duration_units": 4, "pitch": 72}],
                },
            ]
        }
    )
    waiting = advance_melody_realization_run(run_dir, melody_runner)
    assert waiting.awaiting_review == "melody-all"

    approved = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; import sys; "
                "from scoim.realization_run import approve_pending_review; "
                "approve_pending_review(Path(sys.argv[1]), actor='human')"
            ),
            str(run_dir),
        ],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert approved.returncode == 0, approved.stderr
    transition_runner = OneResponseRunner(
        {
            "transition_melodies": [
                {
                    "foreground_voice": "upper",
                    "notes": [{"at_units": 0, "duration_units": 4, "pitch": 69}],
                },
                {
                    "foreground_voice": "upper",
                    "notes": [{"at_units": 0, "duration_units": 4, "pitch": 71}],
                },
            ]
        }
    )

    completed = advance_melody_realization_run(run_dir, transition_runner)

    assert completed.status == "completed"
    assert melody_runner.calls == 1
    assert transition_runner.calls == 1
    assert RunStore(run_dir, max_calls=2).call_attempt_count == 2


def test_melody_run_stops_before_calling_the_model_when_harmony_is_missing(
    tmp_path: Path,
) -> None:
    class NeverRunner:
        calls = 0

        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            self.calls += 1
            raise AssertionError("The model must not be called")

    document = _approved_script()
    starting = _with_plan(document)
    run_dir = tmp_path / "run"
    initialize_melody_realization_run(
        run_dir,
        document,
        starting,
        source_workspace_record_sha256="c" * 64,
        review_after=frozenset(),
        model="test-model",
        max_calls=2,
    )
    runner = NeverRunner()

    failed = advance_melody_realization_run(run_dir, runner)

    assert failed.status == "failed"
    assert failed.issues[0].code is IssueCode.NOT_FOUND
    assert runner.calls == 0
    assert RunStore(run_dir, max_calls=2).call_attempt_count == 0


def test_transition_melody_stops_before_a_second_call_when_connector_harmony_is_missing(
    tmp_path: Path,
) -> None:
    class MelodyRunner:
        calls = 0

        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            self.calls += 1
            response = {
                "melodies": [
                    {
                        "foreground_voice": "upper",
                        "notes": [{"at_units": 0, "duration_units": 4, "pitch": 67}],
                    },
                    {
                        "foreground_voice": "upper",
                        "notes": [{"at_units": 0, "duration_units": 4, "pitch": 72}],
                    },
                ]
            }
            return ProposalRun(
                "fake",
                "test-model",
                {},
                True,
                "completed",
                json.dumps(response).encode(),
                b'{"type":"thread.started"}\n{"type":"turn.completed"}\n',
                b"",
                (),
            )

    document = _approved_script()
    starting = _with_harmonies(document)
    connector_key = "/harmonies/connector"
    without_connector = _must_apply(
        starting,
        CheckedWorkspaceDiff(
            starting.revision,
            starting.workspace_record_sha256,
            ((connector_key, state_key_sha256(starting, connector_key)),),
            (connector_key,),
            (PatchOperation("remove", connector_key, None),),
            "f" * 64,
            (),
            ((connector_key, ((connector_key, state_key_sha256(starting, connector_key)),)),),
        ),
    )
    run_dir = tmp_path / "missing-connector"
    initialize_melody_realization_run(
        run_dir,
        document,
        without_connector,
        source_workspace_record_sha256=without_connector.workspace_record_sha256,
        review_after=frozenset(),
        model="test-model",
        max_calls=2,
    )
    runner = MelodyRunner()

    failed = advance_melody_realization_run(run_dir, runner)

    assert failed.status == "failed"
    assert failed.issues[0].path == "/harmonies/connector"
    assert runner.calls == 1
    assert RunStore(run_dir, max_calls=2).call_attempt_count == 1


def test_human_melody_replacement_uses_the_regular_melody_checker(tmp_path: Path) -> None:
    class MelodyRunner:
        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            response = {
                "melodies": [
                    {
                        "foreground_voice": "upper",
                        "notes": [{"at_units": 0, "duration_units": 4, "pitch": 67}],
                    },
                    {
                        "foreground_voice": "upper",
                        "notes": [{"at_units": 0, "duration_units": 4, "pitch": 72}],
                    },
                ]
            }
            return ProposalRun(
                "fake",
                "test-model",
                {},
                True,
                "completed",
                json.dumps(response).encode(),
                b'{"type":"thread.started"}\n{"type":"turn.completed"}\n',
                b"",
                (),
            )

    document = _approved_script()
    run_dir = tmp_path / "run"
    initialize_melody_realization_run(
        run_dir,
        document,
        _with_harmonies(document),
        source_workspace_record_sha256="d" * 64,
        review_after=frozenset({"melody-all"}),
        model="test-model",
        max_calls=2,
    )
    waiting = advance_melody_realization_run(run_dir, MelodyRunner())
    assert waiting.status == "awaiting_review"
    invalid = {
        "melodies": [
            {
                "foreground_voice": "upper",
                "notes": [{"at_units": 3, "duration_units": 2, "pitch": 67}],
            },
            {
                "foreground_voice": "upper",
                "notes": [{"at_units": 0, "duration_units": 4, "pitch": 72}],
            },
        ]
    }
    with pytest.raises(ValueError, match="exceeds"):
        replace_pending_review_with_response(run_dir, invalid, actor="human")

    replaced = replace_pending_review_with_response(
        run_dir,
        {
            "melodies": [
                {
                    "foreground_voice": "upper",
                    "notes": [{"at_units": 0, "duration_units": 4, "pitch": 65}],
                },
                {
                    "foreground_voice": "upper",
                    "notes": [{"at_units": 0, "duration_units": 4, "pitch": 74}],
                },
            ]
        },
        actor="human",
    )

    assert replaced.status == "running"
    assert dict(replaced.workspace.melodies)["contrast"].notes[0].pitch == 65
    assert dict(replaced.workspace.melodies)["theme"].notes[0].pitch == 74


def test_harmony_review_replacement_is_checked_before_completion(tmp_path: Path) -> None:
    class QueueRunner:
        def __init__(self) -> None:
            self.responses = [
                {
                    "tonal_center": 0,
                    "mode": "major",
                    "contrast_descriptions": ["Open", "Move"],
                },
                {
                    "harmonies": [
                        [
                            {
                                "duration_units": 2,
                                "root_pitch_class": 7,
                                "quality": "major",
                            }
                        ],
                        [
                            {
                                "duration_units": 4,
                                "root_pitch_class": 5,
                                "quality": "major",
                            }
                        ],
                        [
                            {
                                "duration_units": 4,
                                "root_pitch_class": 0,
                                "quality": "major",
                            }
                        ],
                    ]
                },
            ]

        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            return ProposalRun(
                "fake",
                "test-model",
                {},
                True,
                "completed",
                json.dumps(self.responses.pop(0)).encode(),
                b'{"type":"thread.started"}\n{"type":"turn.completed"}\n',
                b"",
                (),
            )

    run_dir = tmp_path / "run"
    initialize_minimal_realization_run(
        run_dir,
        _approved_script(),
        review_after=frozenset({"harmony-all"}),
        model="test-model",
        max_calls=2,
    )
    waiting = advance_minimal_realization_run(run_dir, QueueRunner())
    assert waiting.awaiting_review == "harmony-all"
    with pytest.raises(ValueError):
        replace_pending_review_with_response(run_dir, {"harmonies": []}, actor="human")
    assert waiting == RealizationRunStore(run_dir).rebuild()

    replacement = {
        "harmonies": [
            [{"duration_units": 2, "root_pitch_class": 2, "quality": "minor"}],
            [{"duration_units": 4, "root_pitch_class": 5, "quality": "major"}],
            [{"duration_units": 4, "root_pitch_class": 0, "quality": "major"}],
        ]
    }
    completed = replace_pending_review_with_response(run_dir, replacement, actor="human")

    assert completed.status == "completed"
    assert dict(completed.workspace.harmonies)["connector"][0].root_pitch_class == 2


def test_accompaniment_operation_places_coarse_values_without_model_owned_ids_or_pitch() -> None:
    document = _approved_script()
    workspace = _with_melodies(document)
    targets = ("contrast", "theme")

    schema_text = json.dumps(accompaniment_response_schema(document, workspace, targets))
    assert "event_id" not in schema_text
    assert '"pitch"' not in schema_text
    assert "uniqueItems" not in schema_text
    checked_diff = build_accompaniment_checked_diff(
        document,
        workspace,
        targets,
        {
            "accompaniments": [
                {
                    "events": [
                        {
                            "at_units": 0,
                            "preferred_duration_units": 4,
                            "degree": "root",
                            "preferred_register_zone": "bass",
                            "articulations": [],
                        }
                    ]
                },
                {
                    "events": [
                        {
                            "at_units": 0,
                            "preferred_duration_units": 4,
                            "degree": "fifth",
                            "preferred_register_zone": "low",
                            "articulations": ["tenuto"],
                        }
                    ]
                },
            ]
        },
    )

    applied = apply_checked_diff(workspace, checked_diff)

    assert checked_diff.validation_issues == ()
    assert applied.workspace is not None
    accompaniments = dict(applied.workspace.accompaniments)
    assert set(accompaniments) == set(targets)
    assert accompaniments["contrast"].notes[0].degree == "root"
    assert accompaniments["theme"].notes[0].pitch % 12 == 11


def test_harmony_change_invalidates_only_its_melody_and_accompaniment() -> None:
    document = _approved_script()
    workspace = _with_accompaniments(document)
    changed = build_harmony_checked_diff(
        document,
        workspace,
        ("contrast",),
        {"harmonies": [[{"duration_units": 4, "root_pitch_class": 9, "quality": "minor"}]]},
    )

    updated = _must_apply(workspace, changed)

    assert "contrast" not in dict(updated.melodies)
    assert "contrast" not in dict(updated.accompaniments)
    assert "theme" in dict(updated.melodies)
    assert "theme" in dict(updated.accompaniments)


def test_shared_connector_gets_separate_transition_accompaniments_with_boundaries() -> None:
    document = _approved_script()
    workspace = _with_accompaniments_and_transition_melodies(document)
    targets = ("a_to_b", "b_to_a")

    contexts = transition_accompaniment_contexts(document, workspace, targets)
    schema_text = json.dumps(transition_accompaniment_response_schema(document, workspace, targets))
    assert "event_id" not in schema_text
    assert '"pitch"' not in schema_text
    event = {
        "at_units": 0,
        "preferred_duration_units": 4,
        "degree": "root",
        "preferred_register_zone": "bass",
        "articulations": [],
    }
    checked_diff = build_transition_accompaniment_checked_diff(
        document,
        workspace,
        targets,
        {"transition_accompaniments": [{"events": [event]}, {"events": [event]}]},
    )

    updated = _must_apply(workspace, checked_diff)

    assert contexts[0]["source_boundary"] != contexts[1]["source_boundary"]
    assert set(dict(updated.transition_accompaniments)) == set(targets)


def test_accompaniment_run_completes_ordinary_and_transition_values_in_two_calls(
    tmp_path: Path,
) -> None:
    class QueueRunner:
        def __init__(self) -> None:
            event = {
                "at_units": 0,
                "preferred_duration_units": 4,
                "degree": "root",
                "preferred_register_zone": "bass",
                "articulations": [],
            }
            self.responses = [
                {"accompaniments": [{"events": [event]}, {"events": [event]}]},
                {
                    "transition_accompaniments": [
                        {"events": [event]},
                        {"events": [event]},
                    ]
                },
            ]
            self.calls = 0

        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            response = self.responses[self.calls]
            self.calls += 1
            return ProposalRun(
                "fake",
                "test-model",
                {},
                True,
                "completed",
                json.dumps(response).encode(),
                b'{"type":"thread.started"}\n{"type":"turn.completed"}\n',
                b"",
                (),
            )

    document = _approved_script()
    starting = _with_all_melodies(document)
    runner = QueueRunner()
    run_dir = tmp_path / "accompaniment-run"
    initialize_accompaniment_realization_run(
        run_dir,
        document,
        starting,
        source_workspace_record_sha256=starting.workspace_record_sha256,
        review_after=frozenset(),
        model="test-model",
        max_calls=2,
    )

    completed = advance_accompaniment_realization_run(run_dir, runner)

    assert completed.status == "completed"
    assert completed.completed_operations == (
        "accompaniment-all",
        "transition-accompaniment-all",
    )
    assert runner.calls == 2
    assert set(dict(completed.workspace.accompaniments)) == {"contrast", "theme"}
    assert set(dict(completed.workspace.transition_accompaniments)) == {"a_to_b", "b_to_a"}


def test_accompaniment_run_repairs_a_wrong_response_count_with_one_full_response(
    tmp_path: Path,
) -> None:
    event = {
        "at_units": 0,
        "preferred_duration_units": 4,
        "degree": "root",
        "preferred_register_zone": "bass",
        "articulations": [],
    }

    class QueueRunner:
        def __init__(self) -> None:
            self.responses = [
                {"accompaniments": [{"events": [event]}]},
                {"accompaniments": [{"events": [event]}, {"events": [event]}]},
            ]
            self.prompts: list[str] = []

        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            self.prompts.append(prompt)
            response = self.responses[len(self.prompts) - 1]
            return ProposalRun(
                "fake",
                "test-model",
                {},
                True,
                "completed",
                json.dumps(response).encode(),
                b'{"type":"turn.completed"}\n',
                b"",
                (),
            )

    document = _approved_script("fixed-aba")
    script = document["script"]
    assert isinstance(script, dict)
    script["transitions"] = {}
    approval = document["approval"]
    assert isinstance(approval, dict)
    approval["content_sha256"] = content_sha256(document)
    starting = _with_melodies(document)
    runner = QueueRunner()
    run_dir = tmp_path / "accompaniment-repair"
    initialize_accompaniment_realization_run(
        run_dir,
        document,
        starting,
        source_workspace_record_sha256=starting.workspace_record_sha256,
        review_after=frozenset(),
        model="test-model",
        max_calls=2,
        content_repair_limit=1,
    )

    completed = advance_accompaniment_realization_run(run_dir, runner)

    assert completed.status == "completed"
    assert completed.completed_operations == ("accompaniment-all",)
    assert len(runner.prompts) == 2
    assert "is too short" in runner.prompts[1]
    assert '"accompaniments"' in runner.prompts[1]
    attempt_root = run_dir / "attempts" / "accompaniment-all"
    first_validation = json.loads(
        (attempt_root / "attempt-001" / "validation.json").read_text(encoding="utf-8")
    )
    second_validation = json.loads(
        (attempt_root / "attempt-002" / "validation.json").read_text(encoding="utf-8")
    )
    assert first_validation["status"] == "invalid"
    assert second_validation["status"] == "valid"
    assert second_validation["issues"] == []
    assert len(dict(completed.workspace.accompaniments)) == 2
    assert completed.workspace.plan == starting.plan
    assert completed.workspace.harmonies == starting.harmonies
    assert completed.workspace.melodies == starting.melodies
    assert completed.workspace.transition_melodies == starting.transition_melodies
    assert completed.workspace.transition_accompaniments == starting.transition_accompaniments


def test_accompaniment_run_repairs_an_unplaceable_response(tmp_path: Path) -> None:
    def response(duration_units: int) -> dict[str, object]:
        event = {
            "at_units": 0,
            "preferred_duration_units": duration_units,
            "degree": "root",
            "preferred_register_zone": "bass",
            "articulations": [],
        }
        return {"accompaniments": [{"events": [event]}, {"events": [event]}]}

    class QueueRunner:
        def __init__(self) -> None:
            self.responses = [response(5), response(4)]
            self.prompts: list[str] = []

        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            self.prompts.append(prompt)
            value = self.responses[len(self.prompts) - 1]
            return ProposalRun(
                "fake",
                "test-model",
                {},
                True,
                "completed",
                json.dumps(value).encode(),
                b'{"type":"turn.completed"}\n',
                b"",
                (),
            )

    document = _approved_script("fixed-aba")
    script = document["script"]
    assert isinstance(script, dict)
    script["transitions"] = {}
    approval = document["approval"]
    assert isinstance(approval, dict)
    approval["content_sha256"] = content_sha256(document)
    starting = _with_melodies(document)
    runner = QueueRunner()
    run_dir = tmp_path / "accompaniment-placement-repair"
    initialize_accompaniment_realization_run(
        run_dir,
        document,
        starting,
        source_workspace_record_sha256=starting.workspace_record_sha256,
        review_after=frozenset(),
        model="test-model",
        max_calls=2,
        content_repair_limit=1,
    )

    completed = advance_accompaniment_realization_run(run_dir, runner)

    assert completed.status == "completed"
    assert len(runner.prompts) == 2
    first_validation = json.loads(
        (run_dir / "attempts" / "accompaniment-all" / "attempt-001" / "validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert first_validation["status"] == "invalid"
    assert "/response/accompaniments/0" in runner.prompts[1]


def test_accompaniment_run_stops_after_one_failed_content_repair(tmp_path: Path) -> None:
    event = {
        "at_units": 0,
        "preferred_duration_units": 4,
        "degree": "root",
        "preferred_register_zone": "bass",
        "articulations": [],
    }

    class AlwaysShortRunner:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            self.calls += 1
            return ProposalRun(
                "fake",
                "test-model",
                {},
                True,
                "completed",
                json.dumps({"accompaniments": [{"events": [event]}]}).encode(),
                b'{"type":"turn.completed"}\n',
                b"",
                (),
            )

    document = _approved_script("fixed-aba")
    script = document["script"]
    assert isinstance(script, dict)
    script["transitions"] = {}
    approval = document["approval"]
    assert isinstance(approval, dict)
    approval["content_sha256"] = content_sha256(document)
    starting = _with_melodies(document)
    runner = AlwaysShortRunner()
    run_dir = tmp_path / "accompaniment-failed-repair"
    initialize_accompaniment_realization_run(
        run_dir,
        document,
        starting,
        source_workspace_record_sha256=starting.workspace_record_sha256,
        review_after=frozenset(),
        model="test-model",
        max_calls=2,
        content_repair_limit=1,
    )

    failed = advance_accompaniment_realization_run(run_dir, runner)

    assert failed.status == "failed"
    assert failed.completed_operations == ()
    assert runner.calls == 2
    attempts = sorted((run_dir / "attempts" / "accompaniment-all").glob("attempt-*"))
    assert [attempt.name for attempt in attempts] == ["attempt-001", "attempt-002"]
    assert all((attempt / "validation.json").is_file() for attempt in attempts)
    assert failed.issues
    assert failed.issues[0].path == "/response/accompaniments"


def test_accompaniment_timeout_does_not_start_content_repair(tmp_path: Path) -> None:
    class TimeoutRunner:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            self.calls += 1
            return ProposalRun(
                "fake",
                "test-model",
                {},
                True,
                "timeout",
                None,
                b'{"type":"thread.started"}\n',
                b"timed out",
                (),
            )

    document = _approved_script("fixed-aba")
    starting = _with_melodies(document)
    runner = TimeoutRunner()
    run_dir = tmp_path / "accompaniment-timeout"
    initialize_accompaniment_realization_run(
        run_dir,
        document,
        starting,
        source_workspace_record_sha256=starting.workspace_record_sha256,
        review_after=frozenset(),
        model="test-model",
        max_calls=2,
        content_repair_limit=1,
    )

    with pytest.raises(InterruptedAttemptError):
        advance_accompaniment_realization_run(run_dir, runner)

    assert runner.calls == 1
    attempts = list((run_dir / "attempts" / "accompaniment-all").glob("attempt-*"))
    assert len(attempts) == 1
    assert not (attempts[0] / "validation.json").exists()


def test_accompaniment_resume_does_not_repeat_a_completed_initial_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    event = {
        "at_units": 0,
        "preferred_duration_units": 4,
        "degree": "root",
        "preferred_register_zone": "bass",
        "articulations": [],
    }

    class QueueRunner:
        def __init__(self) -> None:
            self.calls = 0
            self.responses = [
                {"accompaniments": [{"events": [event]}]},
                {"accompaniments": [{"events": [event]}, {"events": [event]}]},
            ]

        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            response = self.responses[self.calls]
            self.calls += 1
            return ProposalRun(
                "fake",
                "test-model",
                {},
                True,
                "completed",
                json.dumps(response).encode(),
                b'{"type":"turn.completed"}\n',
                b"",
                (),
            )

    document = _approved_script("fixed-aba")
    script = document["script"]
    assert isinstance(script, dict)
    script["transitions"] = {}
    approval = document["approval"]
    assert isinstance(approval, dict)
    approval["content_sha256"] = content_sha256(document)
    starting = _with_melodies(document)
    runner = QueueRunner()
    run_dir = tmp_path / "accompaniment-resume"
    initialize_accompaniment_realization_run(
        run_dir,
        document,
        starting,
        source_workspace_record_sha256=starting.workspace_record_sha256,
        review_after=frozenset(),
        model="test-model",
        max_calls=2,
        content_repair_limit=1,
    )
    original_builder = operation_module.build_accompaniment_checked_diff
    builder_calls = 0

    def interrupt_once(*args: object, **kwargs: object) -> CheckedWorkspaceDiff:
        nonlocal builder_calls
        builder_calls += 1
        if builder_calls == 1:
            raise RuntimeError("simulated process stop after response")
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(operation_module, "build_accompaniment_checked_diff", interrupt_once)

    with pytest.raises(RuntimeError, match="simulated process stop"):
        advance_accompaniment_realization_run(run_dir, runner)
    completed = advance_accompaniment_realization_run(run_dir, runner)

    assert completed.status == "completed"
    assert runner.calls == 2
    assert len(list((run_dir / "attempts" / "accompaniment-all").glob("attempt-*"))) == 2


def test_accompaniment_review_replacement_uses_the_same_placement_checker(
    tmp_path: Path,
) -> None:
    class OneRunner:
        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            event = {
                "at_units": 0,
                "preferred_duration_units": 4,
                "degree": "root",
                "preferred_register_zone": "bass",
                "articulations": [],
            }
            return ProposalRun(
                "fake",
                "test-model",
                {},
                True,
                "completed",
                json.dumps({"accompaniments": [{"events": [event]}, {"events": [event]}]}).encode(),
                b'{"type":"turn.completed"}\n',
                b"",
                (),
            )

    document = _approved_script()
    starting = _with_all_melodies(document)
    run_dir = tmp_path / "accompaniment-review"
    initialize_accompaniment_realization_run(
        run_dir,
        document,
        starting,
        source_workspace_record_sha256=starting.workspace_record_sha256,
        review_after=frozenset({"accompaniment-all"}),
        model="test-model",
        max_calls=2,
    )
    waiting = advance_accompaniment_realization_run(run_dir, OneRunner())
    assert waiting.awaiting_review == "accompaniment-all"

    invalid_event = {
        "at_units": 0,
        "preferred_duration_units": 5,
        "degree": "root",
        "preferred_register_zone": "bass",
        "articulations": [],
    }
    with pytest.raises(ValueError):
        replace_pending_review_with_response(
            run_dir,
            {
                "accompaniments": [
                    {"events": [invalid_event]},
                    {"events": [invalid_event]},
                ]
            },
            actor="human",
        )

    valid_event = dict(invalid_event, preferred_duration_units=4, degree="fifth")
    resumed = replace_pending_review_with_response(
        run_dir,
        {
            "accompaniments": [
                {"events": [valid_event]},
                {"events": [valid_event]},
            ]
        },
        actor="human",
    )

    assert resumed.status == "running"
    assert all(
        accompaniment.notes[0].degree == "fifth"
        for _, accompaniment in resumed.workspace.accompaniments
    )


def test_transition_melody_change_invalidates_only_its_transition_accompaniment() -> None:
    document = _approved_script()
    workspace = _with_all_accompaniments(document)

    changed = build_transition_melody_checked_diff(
        document,
        workspace,
        ("a_to_b",),
        {
            "transition_melodies": [
                {
                    "foreground_voice": "upper",
                    "notes": [{"at_units": 0, "duration_units": 4, "pitch": 65}],
                }
            ]
        },
    )
    updated = _must_apply(workspace, changed)

    assert "a_to_b" not in dict(updated.transition_accompaniments)
    assert "b_to_a" in dict(updated.transition_accompaniments)


def test_accompaniment_run_omits_transition_operation_when_script_has_no_transitions(
    tmp_path: Path,
) -> None:
    class OneRunner:
        calls = 0

        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            self.calls += 1
            event = {
                "at_units": 0,
                "preferred_duration_units": 4,
                "degree": "root",
                "preferred_register_zone": "bass",
                "articulations": [],
            }
            return ProposalRun(
                "fake",
                "test-model",
                {},
                True,
                "completed",
                json.dumps({"accompaniments": [{"events": [event]}, {"events": [event]}]}).encode(),
                b'{"type":"turn.completed"}\n',
                b"",
                (),
            )

    document = _approved_script("fixed-aba")
    script = document["script"]
    assert isinstance(script, dict)
    script["transitions"] = {}
    approval = document["approval"]
    assert isinstance(approval, dict)
    approval["content_sha256"] = content_sha256(document)
    starting = _with_melodies(document)
    runner = OneRunner()
    run_dir = tmp_path / "accompaniment-no-transition"
    initialize_accompaniment_realization_run(
        run_dir,
        document,
        starting,
        source_workspace_record_sha256=starting.workspace_record_sha256,
        review_after=frozenset(),
        model="test-model",
        max_calls=1,
    )

    completed = advance_accompaniment_realization_run(run_dir, runner)

    assert completed.status == "completed"
    assert completed.completed_operations == ("accompaniment-all",)
    assert runner.calls == 1


@pytest.mark.parametrize(
    ("workspace_factory", "targets", "message"),
    [
        (lambda document: create_workspace(document), ("theme",), "overall plan"),
        (lambda document: _with_melodies(document), (), "non-empty unique"),
        (lambda document: _with_melodies(document), ("missing",), "Unknown"),
        (lambda document: _with_melodies(document), ("connector",), "cannot be transition"),
        (lambda document: _with_harmonies(document), ("theme",), "no current melody"),
        (
            lambda document: _with_melodies_missing_theme_harmony(document),
            ("theme",),
            "no current harmony",
        ),
        (
            lambda document: replace(_with_melodies(document), schema_version=2),
            ("theme",),
            "version 3",
        ),
    ],
)
def test_accompaniment_schema_rejects_missing_or_wrong_context(
    workspace_factory: object,
    targets: tuple[str, ...],
    message: str,
) -> None:
    document = _approved_script()
    workspace = workspace_factory(document)  # type: ignore[operator]

    with pytest.raises(ValueError, match=message):
        accompaniment_response_schema(document, workspace, targets)


@pytest.mark.parametrize(
    ("workspace_factory", "targets", "message"),
    [
        (lambda document: _with_accompaniments(document), (), "non-empty unique"),
        (lambda document: _with_accompaniments(document), ("missing",), "Unknown"),
        (
            lambda document: _with_accompaniments(document),
            ("a_to_b",),
            "dependency is missing",
        ),
        (
            lambda document: _with_all_melodies(document),
            ("a_to_b",),
            "dependency is missing",
        ),
        (
            lambda document: replace(
                _with_accompaniments_and_transition_melodies(document), plan=None
            ),
            ("a_to_b",),
            "current overall plan",
        ),
        (
            lambda document: replace(
                _with_accompaniments_and_transition_melodies(document), schema_version=2
            ),
            ("a_to_b",),
            "version 3",
        ),
    ],
)
def test_transition_accompaniment_schema_rejects_missing_or_wrong_context(
    workspace_factory: object,
    targets: tuple[str, ...],
    message: str,
) -> None:
    document = _approved_script()
    workspace = workspace_factory(document)  # type: ignore[operator]

    with pytest.raises(ValueError, match=message):
        transition_accompaniment_response_schema(document, workspace, targets)


@pytest.mark.parametrize("transition", [False, True])
@pytest.mark.parametrize("failure", ["exception", "unplaceable"])
def test_accompaniment_builders_turn_placement_failures_into_checked_issues(
    monkeypatch: pytest.MonkeyPatch,
    transition: bool,
    failure: str,
) -> None:
    document = _approved_script()
    if transition:
        workspace = _with_accompaniments_and_transition_melodies(document)
        targets = ("a_to_b",)
        field = "transition_accompaniments"
        builder = build_transition_accompaniment_checked_diff
    else:
        workspace = _with_melodies(document)
        targets = ("theme",)
        field = "accompaniments"
        builder = build_accompaniment_checked_diff
    if failure == "exception":

        def fail(**kwargs: object) -> object:
            raise ValueError("invalid placement input")

        monkeypatch.setattr(operation_module, "place_workspace_accompaniment", fail)
    else:
        monkeypatch.setattr(
            operation_module,
            "place_workspace_accompaniment",
            lambda **kwargs: SimpleNamespace(
                status="search_unplaceable",
                value=None,
                reason="no joint placement",
            ),
        )
    event = {
        "at_units": 0,
        "preferred_duration_units": 4,
        "degree": "root",
        "preferred_register_zone": "bass",
        "articulations": [],
    }

    checked = builder(document, workspace, targets, {field: [{"events": [event]}]})

    assert checked.validation_issues[0].code == IssueCode.MODEL_OUTPUT_INVALID


@pytest.mark.parametrize(
    ("builder", "field"),
    [
        (build_accompaniment_checked_diff, "accompaniments"),
        (build_transition_accompaniment_checked_diff, "transition_accompaniments"),
    ],
)
def test_accompaniment_builders_preserve_schema_failures(
    builder: object,
    field: str,
) -> None:
    document = _approved_script()
    workspace = (
        _with_melodies(document)
        if field == "accompaniments"
        else _with_accompaniments_and_transition_melodies(document)
    )
    targets = ("theme",) if field == "accompaniments" else ("a_to_b",)

    checked = builder(document, workspace, targets, {})  # type: ignore[operator]

    assert checked.validation_issues[0].code == IssueCode.MODEL_OUTPUT_INVALID


def test_ending_diff_jointly_adds_three_values_with_shared_external_dependencies() -> None:
    document = _approved_script()
    workspace = _with_all_accompaniments(document)

    checked = build_ending_checked_diff(document, workspace, ("cadence",))
    updated = _must_apply(workspace, checked)

    assert checked.validation_issues == ()
    assert checked.write_keys == (
        "/harmonies/cadence",
        "/melodies/cadence",
        "/accompaniments/cadence",
    )
    expected_dependencies = {
        "/approved_script_sha256",
        "/plan",
        "/melodies/theme",
    }
    for write_key, dependencies in checked.write_read_hashes or ():
        assert write_key in checked.write_keys
        assert {key for key, _ in dependencies} == expected_dependencies
    assert "cadence" in dict(updated.harmonies)
    assert "cadence" in dict(updated.melodies)
    assert "cadence" in dict(updated.accompaniments)


def test_performance_operations_follow_comparison_dependencies() -> None:
    document = _approved_script()
    starting = _with_all_accompaniments(document)
    workspace = _must_apply(
        starting,
        build_ending_checked_diff(document, starting, ("cadence",)),
    )

    schedule = performance_operation_schedule(document, workspace)

    assert schedule.default_node_ids == (
        "whole",
        "a1",
        "bridge_ab",
        "b",
        "b1",
        "bridge_ba",
        "a2_open",
        "a2_answer",
        "release",
    )
    assert schedule.absolute_node_ids == ("a1_open",)
    assert schedule.comparative_waves == (("a1_answer", "b2"), ("a2",))


def test_performance_directions_for_the_same_section_and_placement_are_grouped() -> None:
    document = _approved_script()
    script = document["script"]
    assert isinstance(script, dict)
    setup = script["performance_setup"]
    assert isinstance(setup, dict)
    directions = setup["performance_directions"]
    assert isinstance(directions, dict)
    directions["opening_touch"] = {
        "target": {"type": "placement", "id": "a1_open_theme"},
        "description": "冒頭だけ柔らかい打鍵にする。",
    }
    approval = document["approval"]
    assert isinstance(approval, dict)
    approval["content_sha256"] = content_sha256(document)
    workspace = _with_all_accompaniments(document)

    group = next(
        item
        for item in performance_operation_schedule(document, workspace).direction_groups
        if item.node_id == "a1_open"
    )

    assert group.direction_ids == ("opening_freedom", "opening_touch")
    assert group.comparison_node_ids == ()


def test_performance_comparison_cycle_is_rejected_before_a_model_call() -> None:
    document = _approved_script()
    script = document["script"]
    assert isinstance(script, dict)
    setup = script["performance_setup"]
    assert isinstance(setup, dict)
    directions = setup["performance_directions"]
    assert isinstance(directions, dict)
    opening = directions["opening_freedom"]
    assert isinstance(opening, dict)
    opening["relative_to"] = {"type": "section", "id": "a1_answer"}
    approval = document["approval"]
    assert isinstance(approval, dict)
    approval["content_sha256"] = content_sha256(document)
    workspace = _with_all_accompaniments(document)

    with pytest.raises(ValueError, match="cannot be ordered"):
        performance_operation_schedule(document, workspace)


def test_performance_run_saves_a_comparison_cycle_as_a_typed_failure(
    tmp_path: Path,
) -> None:
    class NeverRunner:
        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            raise AssertionError("The model must not be called")

    document = _approved_script()
    script = document["script"]
    assert isinstance(script, dict)
    setup = script["performance_setup"]
    assert isinstance(setup, dict)
    directions = setup["performance_directions"]
    assert isinstance(directions, dict)
    opening = directions["opening_freedom"]
    assert isinstance(opening, dict)
    opening["relative_to"] = {"type": "section", "id": "a1_answer"}
    approval = document["approval"]
    assert isinstance(approval, dict)
    approval["content_sha256"] = content_sha256(document)
    workspace = _with_all_accompaniments(document)
    run_dir = tmp_path / "performance-cycle"

    initialize_performance_realization_run(
        run_dir,
        document,
        workspace,
        source_workspace_record_sha256=workspace.workspace_record_sha256,
        review_after=frozenset(),
        model="test-model",
    )
    failed = advance_performance_realization_run(run_dir, NeverRunner())

    assert failed.status == "failed"
    assert failed.completed_operations == ()
    assert failed.issues[0].code is IssueCode.UNREPRESENTABLE
    assert RunStore(run_dir).rebuild_state()["calls"]["call_attempt_count"] == 0


def test_performance_staging_rejects_wrong_workspace_and_unprojectable_script() -> None:
    document = _approved_script()
    workspace = _with_all_accompaniments(document)

    with pytest.raises(ValueError, match="version 3 or 4"):
        performance_piece_plan(document, replace(workspace, schema_version=2))
    with pytest.raises(ValueError, match="current overall plan"):
        performance_piece_plan(document, replace(workspace, plan=None))

    unsupported = json.loads(json.dumps(document))
    unsupported_setup = unsupported["script"]["performance_setup"]
    unsupported_setup["target_duration_seconds"] = 120
    unsupported["approval"]["content_sha256"] = content_sha256(unsupported)
    with pytest.raises(ValueError, match="180-second target"):
        performance_piece_plan(unsupported, workspace)


def test_performance_context_rejects_unknown_and_unsettled_comparison_nodes() -> None:
    document = _approved_script()
    workspace = _with_all_accompaniments(document)

    with pytest.raises(ValueError, match="Unknown directed"):
        performance_contexts(document, workspace, ("missing",))
    with pytest.raises(ValueError, match="Unknown directed"):
        performance_read_keys(document, workspace, "missing")
    with pytest.raises(ValueError, match="not settled"):
        performance_contexts(document, workspace, ("a1_answer",))
    with pytest.raises(ValueError, match="not settled"):
        performance_read_keys(document, workspace, "a1_answer")


def test_performance_context_summarizes_relation_specific_transition_material() -> None:
    document = _approved_script()
    script = document["script"]
    assert isinstance(script, dict)
    setup = script["performance_setup"]
    assert isinstance(setup, dict)
    directions = setup["performance_directions"]
    assert isinstance(directions, dict)
    directions["bridge_touch"] = {
        "target": {"type": "section", "id": "bridge_ab"},
        "description": "つなぎを軽く進める。",
    }
    approval = document["approval"]
    assert isinstance(approval, dict)
    approval["content_sha256"] = content_sha256(document)
    workspace = _with_all_accompaniments(document)

    context = performance_contexts(document, workspace, ("bridge_ab",))[0]
    read_keys = performance_read_keys(document, workspace, "bridge_ab")

    assert context["score_summary"][0]["material_kind"] == "transition"
    assert context["score_summary"][0]["melody_note_count"] == 1
    assert context["score_summary"][0]["accompaniment_note_count"] == 1
    assert "/transition_melodies/a_to_b" in read_keys
    assert "/transition_accompaniments/a_to_b" in read_keys


def test_performance_builder_reports_target_schema_and_profile_failures() -> None:
    document = _approved_script()
    workspace = _with_all_accompaniments(document)

    with pytest.raises(ValueError, match="non-empty unique"):
        performance_response_schema(document, workspace, ())
    with pytest.raises(ValueError, match="dependency-ordered"):
        performance_response_schema(document, workspace, ("a1_open", "b2"))
    with pytest.raises(ValueError, match="not settled"):
        performance_response_schema(document, workspace, ("a1_answer", "b2"))

    version_3 = workspace_module._with_current_hashes(
        replace(
            workspace,
            schema_version=3,
            performances=(),
            music_content_sha256="",
            workspace_record_sha256="",
        )
    )
    with pytest.raises(ValueError, match="version 4"):
        performance_response_schema(document, version_3, ("a1_open",))

    malformed = build_performance_checked_diff(document, workspace, ("a1_open",), {})
    assert malformed.validation_issues[0].code == IssueCode.MODEL_OUTPUT_INVALID

    invalid_profile = build_performance_checked_diff(
        document,
        workspace,
        ("a1_open",),
        {
            "performances": [
                {
                    "timing_profile": None,
                    "timing_amount": "subtle",
                    "dynamics_profile": None,
                    "articulation_profile": None,
                    "coordination_profile": None,
                    "pedal_profile": None,
                }
            ]
        },
    )
    assert invalid_profile.validation_issues[0].code == IssueCode.MODEL_OUTPUT_INVALID


def test_performance_builders_report_wrong_version_and_dependency_cycle() -> None:
    document = _approved_script()
    workspace = _with_all_accompaniments(document)
    wrong_version = workspace_module._with_current_hashes(
        replace(
            workspace,
            schema_version=3,
            performances=(),
            music_content_sha256="",
            workspace_record_sha256="",
        )
    )
    wrong = build_performance_defaults_checked_diff(document, wrong_version)
    assert wrong.validation_issues[0].code == IssueCode.SEMANTIC_INVALID

    script = document["script"]
    assert isinstance(script, dict)
    setup = script["performance_setup"]
    assert isinstance(setup, dict)
    directions = setup["performance_directions"]
    assert isinstance(directions, dict)
    opening = directions["opening_freedom"]
    assert isinstance(opening, dict)
    opening["relative_to"] = {"type": "section", "id": "a1_answer"}
    approval = document["approval"]
    assert isinstance(approval, dict)
    approval["content_sha256"] = content_sha256(document)
    cyclic_workspace = _with_all_accompaniments(document)
    cyclic = build_performance_defaults_checked_diff(document, cyclic_workspace)
    assert cyclic.validation_issues[0].code == IssueCode.UNREPRESENTABLE

    with pytest.raises(ValueError, match="version 3 or 4"):
        initialize_performance_realization_run(
            Path("unused"),
            document,
            replace(cyclic_workspace, schema_version=2),
            source_workspace_record_sha256="a" * 64,
            review_after=frozenset(),
            model="test-model",
        )


def test_performance_defaults_fill_only_nodes_without_directions() -> None:
    document = _approved_script()
    starting = _with_all_accompaniments(document)
    workspace = _must_apply(
        starting,
        build_ending_checked_diff(document, starting, ("cadence",)),
    )

    checked = build_performance_defaults_checked_diff(document, workspace)
    updated = _must_apply(workspace, checked)

    assert set(dict(updated.performances)) == set(
        performance_operation_schedule(document, workspace).default_node_ids
    )
    inherited = PerformanceValue(None, None, None, None, None, None)
    assert all(performance == inherited for _, performance in updated.performances)
    assert "a1_open" not in dict(updated.performances)


def test_performance_operation_maps_positional_values_to_python_owned_node_ids() -> None:
    document = _approved_script()
    starting = _with_all_accompaniments(document)
    with_ending = _must_apply(
        starting,
        build_ending_checked_diff(document, starting, ("cadence",)),
    )
    workspace = _must_apply(
        with_ending,
        build_performance_defaults_checked_diff(document, with_ending),
    )
    targets = ("a1_open",)

    schema = performance_response_schema(document, workspace, targets)
    assert "node_id" not in json.dumps(schema)
    checked = build_performance_checked_diff(
        document,
        workspace,
        targets,
        {
            "performances": [
                {
                    "timing_profile": "savor",
                    "timing_amount": "moderate",
                    "dynamics_profile": "shape",
                    "articulation_profile": "legato",
                    "coordination_profile": "rolled",
                    "pedal_profile": "phrase_legato",
                }
            ]
        },
    )
    updated = _must_apply(workspace, checked)

    assert dict(updated.performances)["a1_open"] == PerformanceValue(
        "savor", "moderate", "shape", "legato", "rolled", "phrase_legato"
    )


def test_comparative_context_uses_settled_hierarchical_performance_and_score_summary() -> None:
    document = _approved_script()
    starting = _with_all_accompaniments(document)
    with_ending = _must_apply(
        starting,
        build_ending_checked_diff(document, starting, ("cadence",)),
    )
    with_defaults = _must_apply(
        with_ending,
        build_performance_defaults_checked_diff(document, with_ending),
    )
    workspace = _must_apply(
        with_defaults,
        build_performance_checked_diff(
            document,
            with_defaults,
            ("a1_open",),
            {
                "performances": [
                    {
                        "timing_profile": "savor",
                        "timing_amount": "moderate",
                        "dynamics_profile": "shape",
                        "articulation_profile": "legato",
                        "coordination_profile": "rolled",
                        "pedal_profile": "phrase_legato",
                    }
                ]
            },
        ),
    )

    contexts = performance_contexts(document, workspace, ("a1_answer", "b2"))

    opening_source = contexts[0]["comparison_sources"][0]
    assert opening_source["node_id"] == "a1_open"
    assert opening_source["performance_by_leaf"][0]["timing"] == [
        {
            "owner_node_id": "a1_open",
            "covered_leaf_node_ids": ["a1_open"],
            "timing_profile": "savor",
            "timing_amount": "moderate",
        }
    ]
    assert contexts[0]["score_summary"][0]["material_kind"] == "theme"
    assert contexts[0]["score_summary"][0]["melody_note_count"] > 0
    assert contexts[0]["score_summary"][0]["accompaniment_note_count"] > 0


def test_performance_change_invalidates_only_dependent_comparisons() -> None:
    document = _approved_script()
    starting = _with_all_accompaniments(document)
    with_ending = _must_apply(
        starting,
        build_ending_checked_diff(document, starting, ("cadence",)),
    )
    workspace = _must_apply(
        with_ending,
        build_performance_defaults_checked_diff(document, with_ending),
    )

    def response(*timings: str) -> dict[str, object]:
        return {
            "performances": [
                {
                    "timing_profile": timing,
                    "timing_amount": "subtle",
                    "dynamics_profile": "shape",
                    "articulation_profile": "legato",
                    "coordination_profile": "aligned",
                    "pedal_profile": "phrase_legato",
                }
                for timing in timings
            ]
        }

    workspace = _must_apply(
        workspace,
        build_performance_checked_diff(document, workspace, ("a1_open",), response("savor")),
    )
    workspace = _must_apply(
        workspace,
        build_performance_checked_diff(
            document, workspace, ("a1_answer", "b2"), response("flow", "build")
        ),
    )
    workspace = _must_apply(
        workspace,
        build_performance_checked_diff(document, workspace, ("a2",), response("flow")),
    )
    key = "/performances/a1_open"
    replacement = build_performance_checked_diff(
        document, workspace, ("a1_open",), response("neutral")
    )
    updated = _must_apply(workspace, replacement)

    assert "a1_open" in dict(updated.performances)
    assert "a1_answer" not in dict(updated.performances)
    assert "a2" not in dict(updated.performances)
    assert "b2" in dict(updated.performances)
    assert state_key_sha256(updated, key) != state_key_sha256(workspace, key)


def test_performance_run_completes_defaults_and_three_model_waves(tmp_path: Path) -> None:
    class QueueRunner:
        def __init__(self) -> None:
            self.calls = 0
            self.prompts: list[str] = []

        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            self.prompts.append(prompt)
            counts = (1, 2, 1)
            count = counts[self.calls]
            self.calls += 1
            response = {
                "performances": [
                    {
                        "timing_profile": "savor" if self.calls == 1 else "flow",
                        "timing_amount": "subtle",
                        "dynamics_profile": "shape",
                        "articulation_profile": "legato",
                        "coordination_profile": "aligned",
                        "pedal_profile": "phrase_legato",
                    }
                    for _ in range(count)
                ]
            }
            return ProposalRun(
                "fake",
                "test-model",
                {},
                True,
                "completed",
                json.dumps(response).encode(),
                b'{"type":"thread.started"}\n{"type":"turn.completed"}\n',
                b"",
                (),
            )

    document = _approved_script()
    starting = _with_all_accompaniments(document)
    with_ending = _must_apply(
        starting,
        build_ending_checked_diff(document, starting, ("cadence",)),
    )
    run_dir = tmp_path / "performance-run"
    initialize_performance_realization_run(
        run_dir,
        document,
        with_ending,
        source_workspace_record_sha256=with_ending.workspace_record_sha256,
        review_after=frozenset(),
        model="test-model",
    )
    runner = QueueRunner()

    completed = advance_performance_realization_run(run_dir, runner)

    assert completed.status == "completed"
    assert completed.completed_operations == (
        "performance-defaults",
        "performance-absolute",
        "performance-comparative-1",
        "performance-comparative-2",
    )
    assert runner.calls == 3
    assert len(completed.workspace.performances) == 13
    assert '"timing_profile":"savor"' in runner.prompts[1].replace(" ", "")


def test_performance_run_upgrades_a_verified_version_3_workspace(tmp_path: Path) -> None:
    document = _approved_script()
    starting = _with_all_accompaniments(document)
    with_ending = _must_apply(
        starting,
        build_ending_checked_diff(document, starting, ("cadence",)),
    )
    legacy = workspace_module._with_current_hashes(
        replace(
            with_ending,
            schema_version=3,
            performances=(),
            music_content_sha256="",
            workspace_record_sha256="",
        )
    )
    run_dir = tmp_path / "performance-v3-upgrade"

    initialize_performance_realization_run(
        run_dir,
        document,
        legacy,
        source_workspace_record_sha256=legacy.workspace_record_sha256,
        review_after=frozenset(),
        model="test-model",
    )

    assert RealizationRunStore(run_dir).rebuild().workspace.schema_version == 4


def test_performance_run_without_directions_completes_without_a_model_call(
    tmp_path: Path,
) -> None:
    class NeverRunner:
        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            raise AssertionError("The model must not be called")

    document = _approved_script()
    script = document["script"]
    assert isinstance(script, dict)
    setup = script["performance_setup"]
    assert isinstance(setup, dict)
    setup["performance_directions"] = {}
    approval = document["approval"]
    assert isinstance(approval, dict)
    approval["content_sha256"] = content_sha256(document)
    starting = _with_all_accompaniments(document)
    with_ending = _must_apply(
        starting,
        build_ending_checked_diff(document, starting, ("cadence",)),
    )
    run_dir = tmp_path / "performance-default-only"
    initialize_performance_realization_run(
        run_dir,
        document,
        with_ending,
        source_workspace_record_sha256=with_ending.workspace_record_sha256,
        review_after=frozenset(),
        model="test-model",
    )

    completed = advance_performance_realization_run(run_dir, NeverRunner())

    assert completed.status == "completed"
    assert completed.completed_operations == ("performance-defaults",)
    assert len(completed.workspace.performances) == 13


def test_performance_review_replacement_uses_the_regular_checker_and_resumes(
    tmp_path: Path,
) -> None:
    class QueueRunner:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            counts = (1, 2, 1)
            count = counts[self.calls]
            self.calls += 1
            response = {
                "performances": [
                    {
                        "timing_profile": "savor",
                        "timing_amount": "subtle",
                        "dynamics_profile": "shape",
                        "articulation_profile": "legato",
                        "coordination_profile": "aligned",
                        "pedal_profile": "phrase_legato",
                    }
                    for _ in range(count)
                ]
            }
            return ProposalRun(
                "fake",
                "test-model",
                {},
                True,
                "completed",
                json.dumps(response).encode(),
                b'{"type":"thread.started"}\n{"type":"turn.completed"}\n',
                b"",
                (),
            )

    document = _approved_script()
    starting = _with_all_accompaniments(document)
    with_ending = _must_apply(
        starting,
        build_ending_checked_diff(document, starting, ("cadence",)),
    )
    run_dir = tmp_path / "performance-review"
    initialize_performance_realization_run(
        run_dir,
        document,
        with_ending,
        source_workspace_record_sha256=with_ending.workspace_record_sha256,
        review_after=frozenset({"performance-absolute"}),
        model="test-model",
    )
    runner = QueueRunner()
    waiting = advance_performance_realization_run(run_dir, runner)
    assert waiting.awaiting_review == "performance-absolute"

    replacement = {
        "performances": [
            {
                "timing_profile": "neutral",
                "timing_amount": None,
                "dynamics_profile": "steady",
                "articulation_profile": "score",
                "coordination_profile": "aligned",
                "pedal_profile": "phrase_legato",
            }
        ]
    }
    resumed = replace_pending_review_with_response(run_dir, replacement, actor="human")
    assert resumed.status == "running"
    completed = advance_performance_realization_run(run_dir, runner)

    assert completed.status == "completed"
    assert dict(completed.workspace.performances)["a1_open"].timing_profile == "neutral"
    assert runner.calls == 3


def test_ending_run_completes_without_model_request_or_response(tmp_path: Path) -> None:
    document = _approved_script()
    starting = _with_all_accompaniments(document)
    run_dir = tmp_path / "ending-run"
    initialize_ending_realization_run(
        run_dir,
        document,
        starting,
        source_workspace_record_sha256=starting.workspace_record_sha256,
        review_after=frozenset(),
    )

    completed = advance_ending_realization_run(run_dir)

    assert completed.status == "completed"
    assert completed.completed_operations == ("ending-material",)
    assert "cadence" in dict(completed.workspace.harmonies)
    event_dir = run_dir / "events" / "ending-material"
    assert (event_dir / "diff.json").is_file()
    assert not (event_dir / "request.json").exists()
    assert not (event_dir / "response.json").exists()
    assert not (run_dir / "attempts").exists()
    diagnostic = json.loads(
        (run_dir / "diagnostics" / "ending-material.json").read_text(encoding="utf-8")
    )
    assert diagnostic["material_key"] == "cadence"
    assert diagnostic["nominal_hold_ms"] >= 2_000
    assert diagnostic["common_grid_units_per_weight"] == 4
    spec = json.loads((run_dir / "run-spec.json").read_text(encoding="utf-8"))
    assert spec["max_calls"] == 0
    assert spec["model"] == "deterministic-python"


def test_changing_the_preceding_melody_invalidates_all_ending_values() -> None:
    document = _approved_script()
    workspace = _with_all_accompaniments(document)
    workspace = _must_apply(
        workspace,
        build_ending_checked_diff(document, workspace, ("cadence",)),
    )

    changed = build_melody_checked_diff(
        document,
        workspace,
        ("theme",),
        {
            "melodies": [
                {
                    "foreground_voice": "upper",
                    "notes": [{"at_units": 0, "duration_units": 4, "pitch": 74}],
                }
            ]
        },
    )
    updated = _must_apply(workspace, changed)

    assert "cadence" not in dict(updated.harmonies)
    assert "cadence" not in dict(updated.melodies)
    assert "cadence" not in dict(updated.accompaniments)
    assert "contrast" in dict(updated.harmonies)
    assert "contrast" in dict(updated.melodies)
    assert "contrast" in dict(updated.accompaniments)


def test_ending_run_can_pause_for_review_and_resume_without_a_model_call(
    tmp_path: Path,
) -> None:
    document = _approved_script()
    starting = _with_all_accompaniments(document)
    run_dir = tmp_path / "ending-review"
    initialize_ending_realization_run(
        run_dir,
        document,
        starting,
        source_workspace_record_sha256=starting.workspace_record_sha256,
        review_after=frozenset({"ending-material"}),
    )

    waiting = advance_ending_realization_run(run_dir)
    assert waiting.status == "awaiting_review"
    assert waiting.completed_operations == ()

    approved = approve_pending_review(run_dir, actor="human")
    completed = advance_ending_realization_run(run_dir)

    assert approved.status == "completed"
    assert completed.status == "completed"
    assert not (run_dir / "attempts").exists()


def test_ending_failure_is_saved_as_unrepresentable_without_partial_values() -> None:
    document = _approved_script()
    workspace = _with_harmonies(document)

    checked = build_ending_checked_diff(document, workspace, ("cadence",))
    applied = apply_checked_diff(workspace, checked)

    assert checked.validation_issues[0].code is IssueCode.UNREPRESENTABLE
    assert applied.workspace is None
    assert "cadence" not in dict(workspace.harmonies)
    assert "cadence" not in dict(workspace.melodies)
    assert "cadence" not in dict(workspace.accompaniments)


def _with_plan(document: dict[str, object]) -> RealizationWorkspace:
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
    diff = build_plan_checked_diff(
        document,
        workspace,
        {
            "tonal_center": 0,
            "mode": "major",
            "contrast_descriptions": ["More motion"] * contrast_count,
        },
    )
    return _must_apply(workspace, diff)


def _with_harmonies(document: dict[str, object]) -> RealizationWorkspace:
    workspace = _with_plan(document)
    script = document["script"]
    assert isinstance(script, dict)
    materials = script["materials"]
    assert isinstance(materials, dict)
    targets = tuple(
        sorted(
            key
            for key, material in materials.items()
            if isinstance(key, str) and isinstance(material, dict) and material["kind"] != "ending"
        )
    )
    response = {
        "harmonies": [
            [{"duration_units": 4, "root_pitch_class": index * 2, "quality": "major"}]
            for index, _ in enumerate(targets)
        ]
    }
    return _must_apply(
        workspace,
        build_harmony_checked_diff(document, workspace, targets, response),
    )


def _with_melodies(document: dict[str, object]) -> RealizationWorkspace:
    workspace = _with_harmonies(document)
    targets = ("contrast", "theme")
    response = {
        "melodies": [
            {
                "foreground_voice": "upper",
                "notes": [{"at_units": 0, "duration_units": 4, "pitch": 67}],
            },
            {
                "foreground_voice": "upper",
                "notes": [{"at_units": 0, "duration_units": 4, "pitch": 72}],
            },
        ]
    }
    return _must_apply(
        workspace,
        build_melody_checked_diff(document, workspace, targets, response),
    )


def _with_accompaniments(document: dict[str, object]) -> RealizationWorkspace:
    workspace = _with_melodies(document)
    targets = ("contrast", "theme")
    event = {
        "at_units": 0,
        "preferred_duration_units": 4,
        "degree": "root",
        "preferred_register_zone": "bass",
        "articulations": [],
    }
    return _must_apply(
        workspace,
        build_accompaniment_checked_diff(
            document,
            workspace,
            targets,
            {"accompaniments": [{"events": [event]}, {"events": [event]}]},
        ),
    )


def _with_melodies_missing_theme_harmony(
    document: dict[str, object],
) -> RealizationWorkspace:
    workspace = _with_melodies(document)
    return replace(
        workspace,
        harmonies=tuple(item for item in workspace.harmonies if item[0] != "theme"),
    )


def _with_accompaniments_and_transition_melodies(
    document: dict[str, object],
) -> RealizationWorkspace:
    workspace = _with_accompaniments(document)
    targets = ("a_to_b", "b_to_a")
    response = {
        "transition_melodies": [
            {
                "foreground_voice": "upper",
                "notes": [{"at_units": 0, "duration_units": 4, "pitch": pitch}],
            }
            for pitch in (69, 71)
        ]
    }
    return _must_apply(
        workspace,
        build_transition_melody_checked_diff(document, workspace, targets, response),
    )


def _with_all_melodies(document: dict[str, object]) -> RealizationWorkspace:
    workspace = _with_melodies(document)
    targets = ("a_to_b", "b_to_a")
    return _must_apply(
        workspace,
        build_transition_melody_checked_diff(
            document,
            workspace,
            targets,
            {
                "transition_melodies": [
                    {
                        "foreground_voice": "upper",
                        "notes": [{"at_units": 0, "duration_units": 4, "pitch": pitch}],
                    }
                    for pitch in (69, 71)
                ]
            },
        ),
    )


def _with_all_accompaniments(document: dict[str, object]) -> RealizationWorkspace:
    workspace = _with_accompaniments_and_transition_melodies(document)
    event = {
        "at_units": 0,
        "preferred_duration_units": 4,
        "degree": "root",
        "preferred_register_zone": "bass",
        "articulations": [],
    }
    return _must_apply(
        workspace,
        build_transition_accompaniment_checked_diff(
            document,
            workspace,
            ("a_to_b", "b_to_a"),
            {
                "transition_accompaniments": [
                    {"events": [event]},
                    {"events": [event]},
                ]
            },
        ),
    )


def _must_apply(workspace: RealizationWorkspace, checked_diff) -> RealizationWorkspace:
    result = apply_checked_diff(workspace, checked_diff)
    assert result.workspace is not None, result.issues
    return result.workspace
