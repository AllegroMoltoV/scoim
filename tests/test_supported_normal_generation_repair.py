import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_musical_composer import supported_normal_generation_repair as repair_module
from llm_musical_composer import whole_score_staged_generation_run as whole_run_module
from llm_musical_composer.pipeline_dsl import parse_piece_plan
from llm_musical_composer.staged_material_pilot import (
    MelodyDraft,
    MelodyEventDraft,
)
from llm_musical_composer.supported_normal_generation_repair import (
    extend_final_release_capacity,
    prepare_supported_normal_generation_repair,
    repair_harmonic_capacity,
)
from llm_musical_composer.texture_budget import allocate_texture_budget
from llm_musical_composer.whole_score_staged_generation import (
    assemble_whole_score_melodies,
    assemble_whole_score_skeleton,
)
from llm_musical_composer.whole_score_staged_generation_dsl import (
    WholeScoreCollectionDslError,
    parse_harmonic_collection,
    parse_melody_collection,
    parse_texture_collection,
)
from llm_musical_composer.whole_score_staged_generation_run import (
    execute_harmony_stage,
    execute_melody_stages,
    execute_prepared_whole_score_live_run,
    execute_texture_stage,
    partition_texture_batches,
)

PROJECT_ROOT = Path(__file__).parents[1]
V1_ROOT = PROJECT_ROOT / ".appendix/supported-normal-generation-v1"


class _CallOnlyRunner:
    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def call_number(self) -> int:
        return len(self.calls)

    def run(self, step_id: str, prompt: str, input_hashes=None):
        del prompt, input_hashes
        self.calls.append(step_id)
        return {"composition_source": "unused", "intent_summary": "fixture"}


def test_legacy_v3_harmony_capacity_repair_has_fixed_golden_result() -> None:
    plan = parse_piece_plan(
        (V1_ROOT / "runs/piece-plan/outputs/piece-plan.dsl").read_text(
            encoding="utf-8"
        )
    )
    source = parse_harmonic_collection(
        (V1_ROOT / "runs/whole-score/responses/harmonic-collection.dsl").read_text(
            encoding="utf-8"
        )
    )

    repaired, diagnostic = repair_harmonic_capacity(
        plan,
        source,
        minimum_attack_group_count=649,
        reserve_ending_hold=False,
    )

    assert diagnostic["source_capacity"]["maximum_attack_group_count"] == 516
    assert diagnostic["source_capacity"]["maximum_note_event_count"] == 528
    assert diagnostic["source_early_ending"]["status"] == "pass"
    assert diagnostic["repaired_capacity"]["maximum_attack_group_count"] == 649
    assert diagnostic["repaired_capacity"]["maximum_note_event_count"] == 696
    assert diagnostic["repaired_early_ending"]["status"] == "pass"
    assert [item.length_units for item in repaired] == [98, 76, 106, 91, 75, 45, 60]
    assert diagnostic["content_invariants"] == {
        "event_count_unchanged": True,
        "event_order_unchanged": True,
        "root_pitch_class_unchanged": True,
        "quality_unchanged": True,
    }
    for before, after in zip(source, repaired, strict=True):
        assert [event.root_pitch_class for event in after.draft.events] == [
            event.root_pitch_class for event in before.draft.events
        ]
        assert [event.quality for event in after.draft.events] == [
            event.quality for event in before.draft.events
        ]
        assert all(event.duration_units > 0 for event in after.draft.events)
        assert sum(event.duration_units for event in after.draft.events) == (
            after.length_units
        )
        assert [event.at_units for event in after.draft.events] == [
            sum(item.duration_units for item in after.draft.events[:index])
            for index in range(len(after.draft.events))
        ]


def test_capacity_repair_rejects_a_non_positive_budget() -> None:
    plan = parse_piece_plan(
        (V1_ROOT / "runs/piece-plan/outputs/piece-plan.dsl").read_text(
            encoding="utf-8"
        )
    )
    source = parse_harmonic_collection(
        (V1_ROOT / "runs/whole-score/responses/harmonic-collection.dsl").read_text(
            encoding="utf-8"
        )
    )

    try:
        repair_harmonic_capacity(
            plan,
            source,
            minimum_attack_group_count=0,
        )
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("non-positive capacity must fail before an external call")


def test_capacity_repair_reserves_the_ending_hold_by_default() -> None:
    plan = parse_piece_plan(
        (V1_ROOT / "runs/piece-plan/outputs/piece-plan.dsl").read_text(
            encoding="utf-8"
        )
    )
    source = parse_harmonic_collection(
        (V1_ROOT / "runs/whole-score/responses/harmonic-collection.dsl").read_text(
            encoding="utf-8"
        )
    )

    _, diagnostic = repair_harmonic_capacity(
        plan,
        source,
        minimum_attack_group_count=649,
    )

    assert diagnostic["reserve_ending_hold"] is True
    assert diagnostic["repaired_capacity"]["maximum_attack_group_count"] >= 649
    assert any(
        value > 0
        for value in diagnostic["repaired_capacity"][
            "per_material_reserved_ending_units"
        ].values()
    )


def test_prepare_snapshots_repaired_harmony_as_generation_input(tmp_path: Path) -> None:
    artifact_root = tmp_path / "repair"

    result = prepare_supported_normal_generation_repair(PROJECT_ROOT, artifact_root)

    assert result["status"] == "prepared"
    assert result["maximum_external_call_count"] == 2
    assert result["prepared_texture_prefix_sha256"] == (
        "389e381d6d18269aaca3d752860ea7d5d4cddeb6f1f244e902aa2b6685bf0a15"
    )
    whole_run = artifact_root / "runs/whole-score"
    assert (whole_run / "inputs/prepared-harmonic-collection.dsl").is_file()
    assert (whole_run / "inputs/prepared-melody-collection.dsl").is_file()
    assert (whole_run / "inputs/prepared-texture-budget.json").is_file()
    assert (whole_run / "inputs/prepared-texture-prefix.dsl").is_file()
    spec = json.loads((whole_run / "run-spec.json").read_text(encoding="utf-8"))
    assert spec["model_config"]["maximum_external_calls"] == 2
    assert spec["texture_batch_maximum_event_count"] == 250
    assert "prepared_harmonic_collection" in spec["input_hashes"]["generation_inputs"]
    diagnostic = json.loads(
        (artifact_root / "inputs/ending-capacity-repair.json").read_text(
            encoding="utf-8"
        )
    )
    assert diagnostic["source_harmonic_collection_sha256"] == (
        "799006aa97d0e88c4368d826f3a04fdfb3ec43765327f56aa13fb7efee372dc7"
    )
    assert diagnostic["harmony"]["capacity"]["maximum_attack_group_count"] == 649
    assert diagnostic["harmony"]["capacity"]["maximum_note_event_count"] == 708
    assert diagnostic["harmony"]["extended_length_units"] == 67
    assert diagnostic["melody"]["source_attack_units"] == 52
    assert diagnostic["melody"]["aligned_attack_units"] == 59
    assert diagnostic["texture_budget"]["attack_group_count"] == 649
    source_melodies = parse_melody_collection(
        (
            PROJECT_ROOT
            / ".appendix/supported-normal-generation-v3-score-group-capacity/"
            "runs/whole-score/outputs/melody-collection-main.dsl"
        ).read_text(encoding="utf-8")
    )
    repaired_melodies = parse_melody_collection(
        (artifact_root / "inputs/prepared-melody-collection.dsl").read_text(
            encoding="utf-8"
        )
    )
    assert repaired_melodies[:-1] == source_melodies[:-1]
    assert repaired_melodies[-1].events[:-1] == source_melodies[-1].events[:-1]
    assert repaired_melodies[-1].events[-1] == replace(
        source_melodies[-1].events[-1],
        at_units=59,
    )
    budget = json.loads(
        (artifact_root / "inputs/prepared-texture-budget.json").read_text(
            encoding="utf-8"
        )
    )
    assert budget["materials"][-1]["attack_group_capacity"] == 60
    assert budget["materials"][-1]["reserved_ending_units"] == 7
    assert len(
        parse_texture_collection(
            (artifact_root / "inputs/prepared-texture-prefix.dsl").read_text(
                encoding="utf-8"
            )
        )
    ) == 6
    assert len(result["repair_implementation_sha256"]) == 64


def test_prepare_rejects_tampered_v4_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original = repair_module._read_json

    def tampered(path: Path):
        value = original(path)
        if (
            path.name == "run-state.json"
            and "supported-normal-generation-v4-texture-batches" in path.parts
        ):
            value = json.loads(json.dumps(value))
            value["calls"]["confirmed_external_call_count"] = 1
        return value

    monkeypatch.setattr(repair_module, "_read_json", tampered)

    with pytest.raises(
        repair_module.SupportedNormalGenerationRepairError,
        match="provenance",
    ):
        prepare_supported_normal_generation_repair(PROJECT_ROOT, tmp_path / "repair")


def test_prepare_rejects_tampered_v5_ending_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original = repair_module._read_json

    def tampered(path: Path):
        value = original(path)
        if (
            path.name == "run-state.json"
            and "supported-normal-generation-v5-placement-backtracking" in path.parts
        ):
            value = json.loads(json.dumps(value))
            value["calls"]["confirmed_external_call_count"] = 0
        return value

    monkeypatch.setattr(repair_module, "_read_json", tampered)

    with pytest.raises(
        repair_module.SupportedNormalGenerationRepairError,
        match="v5 ending provenance",
    ):
        prepare_supported_normal_generation_repair(PROJECT_ROOT, tmp_path / "repair")


def test_prepare_rejects_tampered_v6_timeout_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original = repair_module._read_json

    def tampered(path: Path):
        value = original(path)
        if (
            path.name == "run-state.json"
            and "supported-normal-generation-v6-ending-slack" in path.parts
        ):
            value = json.loads(json.dumps(value))
            value["calls"]["successful_external_call_count"] = 1
        return value

    monkeypatch.setattr(repair_module, "_read_json", tampered)

    with pytest.raises(
        repair_module.SupportedNormalGenerationRepairError,
        match="v6 timeout provenance",
    ):
        prepare_supported_normal_generation_repair(PROJECT_ROOT, tmp_path / "repair")


def test_prepared_harmony_skips_external_harmony_call(tmp_path: Path) -> None:
    artifact_root = tmp_path / "repair"
    prepare_supported_normal_generation_repair(PROJECT_ROOT, artifact_root)
    run_dir = artifact_root / "runs/whole-score"
    runner = _CallOnlyRunner()

    skeleton = execute_harmony_stage(PROJECT_ROOT, run_dir, runner)

    assert runner.calls == []
    assert [item.length_units for item in skeleton.materials] == [
        98,
        76,
        106,
        91,
        75,
        45,
        67,
    ]
    state = json.loads((run_dir / "run-state.json").read_text(encoding="utf-8"))
    assert state["steps"]["harmonic-collection"]["outputs"]["source_mode"] == (
        "prepared"
    )


def test_prepared_melody_skips_external_melody_call(tmp_path: Path) -> None:
    artifact_root = tmp_path / "repair"
    prepare_supported_normal_generation_repair(PROJECT_ROOT, artifact_root)
    run_dir = artifact_root / "runs/whole-score"
    runner = _CallOnlyRunner()
    skeleton = execute_harmony_stage(PROJECT_ROOT, run_dir, runner)

    payload = execute_melody_stages(PROJECT_ROOT, run_dir, runner, skeleton)

    assert runner.calls == []
    assert len(payload.materials) == 7
    state = json.loads((run_dir / "run-state.json").read_text(encoding="utf-8"))
    assert state["steps"]["melody-collection-main"]["outputs"]["source_mode"] == (
        "prepared"
    )


def test_texture_batches_isolate_final_release_under_250_events() -> None:
    materials = [
        {"material_id": material_id, "required_texture_event_count": count}
        for material_id, count in zip(
            ("a", "a_var", "b", "climax", "return", "prep", "release"),
            (172, 135, 192, 163, 132, 81, 109),
            strict=True,
        )
    ]

    batches = partition_texture_batches(materials, maximum_event_count=250)

    assert [
        (
            [item["material_id"] for item in batch],
            sum(item["required_texture_event_count"] for item in batch),
        )
        for batch in batches
    ] == [
        (["a"], 172),
        (["a_var"], 135),
        (["b"], 192),
        (["climax"], 163),
        (["return", "prep"], 213),
        (["release"], 109),
    ]


def test_prepared_texture_prefix_is_validated_before_the_remaining_call(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "repair"
    prepare_supported_normal_generation_repair(PROJECT_ROOT, artifact_root)
    run_dir = artifact_root / "runs/whole-score"
    runner = _CallOnlyRunner()
    skeleton = execute_harmony_stage(PROJECT_ROOT, run_dir, runner)
    payload = execute_melody_stages(PROJECT_ROOT, run_dir, runner, skeleton)

    with pytest.raises(WholeScoreCollectionDslError):
        execute_texture_stage(PROJECT_ROOT, run_dir, runner, skeleton, payload)

    assert runner.calls == ["texture-collection-batch-006"]
    state = json.loads((run_dir / "run-state.json").read_text(encoding="utf-8"))
    for index in range(1, 6):
        step = state["steps"][f"texture-collection-batch-{index:03d}"]
        assert step["status"] == "completed"
        assert step["outputs"]["source_mode"] == "prepared"
        assert step["outputs"]["external_call_number"] is None
    validation = json.loads(
        (run_dir / "outputs/texture-collection-batch-005-validation.json").read_text(
            encoding="utf-8"
        )
    )
    placements = {item["material_id"]: item for item in validation["placements"]}
    assert set(placements) == {
        "material_a",
        "material_a_variation",
        "material_b",
        "material_climax",
        "material_return_variation",
        "material_release_preparation",
    }
    assert all(item["status"] == "placed" for item in placements.values())
    assert placements["material_b"]["status"] == "placed"
    assert placements["material_climax"]["status"] == "placed"
    assert placements["material_b"]["backtrack_count"] > 0
    assert placements["material_climax"]["backtrack_count"] > 0


def test_prepared_repair_calls_only_remaining_texture_batch_and_performance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact_root = tmp_path / "repair"
    prepare_supported_normal_generation_repair(PROJECT_ROOT, artifact_root)
    run_dir = artifact_root / "runs/whole-score"
    runner = _CallOnlyRunner()

    def melody(project_root, active_run, active_runner, skeleton):
        del project_root, active_run, active_runner, skeleton
        return object()

    def texture(project_root, active_run, active_runner, skeleton, melodies):
        del project_root, active_run, skeleton, melodies
        active_runner.run("texture-collection-batch-006", "fixture", {})
        return SimpleNamespace(score=object())

    def performance(project_root, active_run, active_runner, score):
        del project_root, active_run, score
        active_runner.run("performance-collection", "fixture", {})
        return object()

    monkeypatch.setattr(whole_run_module, "execute_melody_stages", melody)
    monkeypatch.setattr(whole_run_module, "execute_texture_stage", texture)
    monkeypatch.setattr(whole_run_module, "execute_performance_stage", performance)
    monkeypatch.setattr(
        whole_run_module,
        "evaluate_generic_score_quality",
        lambda plan, score: {"passes": True, "failures": []},
    )
    monkeypatch.setattr(
        whole_run_module,
        "evaluate_staged_candidate",
        lambda project_root, active_run: {
            "status": "completed_unfit",
            "promoted": False,
        },
    )

    execute_prepared_whole_score_live_run(PROJECT_ROOT, run_dir, runner)

    assert runner.calls == [
        "texture-collection-batch-006",
        "performance-collection",
    ]


def test_v6_ending_slack_has_an_exact_texture_integer_allocation() -> None:
    plan = parse_piece_plan(
        (V1_ROOT / "runs/piece-plan/outputs/piece-plan.dsl").read_text(
            encoding="utf-8"
        )
    )
    source = parse_harmonic_collection(
        (V1_ROOT / "runs/whole-score/responses/harmonic-collection.dsl").read_text(
            encoding="utf-8"
        )
    )
    repaired, _ = repair_harmonic_capacity(
        plan,
        source,
        minimum_attack_group_count=649,
        reserve_ending_hold=False,
    )
    repaired, ending = extend_final_release_capacity(plan, repaired)
    assert ending["capacity"]["maximum_attack_group_count"] == 649
    skeleton = assemble_whole_score_skeleton(
        "v3-integer-allocation-test",
        plan,
        tuple((item.length_units, item.draft) for item in repaired),
    )
    melody_drafts = []
    for material in skeleton.materials:
        onsets = [harmony.at_units for harmony in material.harmonies]
        onsets.extend(onset for onset in range(material.length_units) if onset not in onsets)
        melody_drafts.append(
            MelodyDraft(
                "upper",
                tuple(
                    MelodyEventDraft(onset, 1, (60, 64, 67)[index % 3])
                    for index, onset in enumerate(onsets[: max(4, len(material.harmonies))])
                ),
            )
        )
    payload = assemble_whole_score_melodies(
        "v3-integer-allocation-test",
        plan,
        skeleton,
        tuple(melody_drafts),
    )
    prompt_target = json.loads(
        (V1_ROOT / "runs/piece-plan/inputs/prompt-target.json").read_text(
            encoding="utf-8"
        )
    )
    attack_target = next(
        item
        for item in prompt_target["semantic_targets"]["score_spec"]
        if item["id"] == "attack_texture"
    )

    budget = allocate_texture_budget(
        plan,
        skeleton,
        payload,
        attack_target,
        minimum_attack_group_count=649,
    )

    assert budget["score_spec_target"]["attack_group_count"] == 649
    assert budget["score_spec_target"]["note_event_count"] == 1_504
    assert sum(
        item["occurrence_count"] * item["combined_attack_group_count"]
        for item in budget["materials"]
    ) == 649
