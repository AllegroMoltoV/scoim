import copy
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
from test_scoim_phase4_realization import SequencedRunner
from test_scoim_score_model_contracts import _accompaniment, _evaluate, _foreground, _setup
from test_scoim_score_realization import _score_run
from test_scoim_score_rendering import _placement_variation_inputs, _score_inputs, _script

import scoim.score_model_contracts as contracts
import scoim.score_rendering as rendering
from scoim.performance_ir import SectionPerformance
from scoim.phase7_realization import Phase7Request, realize_phase7
from scoim.phase8_bundle import replay_phase8_bundle, verify_phase8_bundle
from scoim.score_ir import ScoreHarmony, ScoreNote
from scoim.score_model_contracts import score_group_prompt
from scoim.score_realization import ScoreRequest, realize_score
from scoim.score_state import load_complete_score_run
from scoim.score_work_plan import ScoreWorkOperation
from scoim.terminal_boundary import TerminalBoundary, derive_score_terminal_boundary


def _terminal_setup(*, accompaniment=True):
    document, _, _ = _setup()
    if accompaniment:
        placements = document["script"]["material_placements"]
        placements["support-final"] = {**placements["support-first"], "section_id": "release"}
    return _setup(document)


@pytest.mark.parametrize("joint", [True, False])
def test_terminal_candidate_commits_owner_and_support_in_existing_groups(joint):
    setup = _terminal_setup()
    accepted = {}
    if not joint:
        first = _evaluate(
            setup,
            ScoreWorkOperation("fg", ("ending-only",), ()),
            (),
            {"foregrounds": [_foreground(at=6, duration=6)], "accompaniments": []},
        )
        assert not first.issues
        assert first.terminal_boundary == TerminalBoundary("score-unit-release", 6, 12)
        accepted.update(first.notes_by_placement)
    ids = ("ending-only", "support-final") if joint else ("support-final",)
    response = {
        "foregrounds": [_foreground(at=6, duration=6)] if joint else [],
        "accompaniments": [_accompaniment()],
    }
    result = _evaluate(setup, ScoreWorkOperation("group", ids, ()), (), response, accepted)
    assert not result.issues
    assert result.terminal_boundary == TerminalBoundary("score-unit-release", 6, 12)
    assert set(result.notes_by_placement) == set(ids)
    bad = copy.deepcopy(response)
    bad["accompaniments"][0]["events"][0]["at_units"] = 7
    bad["accompaniments"][0]["events"][0]["preferred_duration_units"] = 5
    rejected = _evaluate(setup, ScoreWorkOperation("group", ids, ()), (), bad, accepted)
    assert rejected.issues and "terminal" in rejected.issues[0].message
    assert not rejected.notes_by_placement


def test_partial_terminal_foregrounds_do_not_fix_an_early_boundary():
    document, _, _ = _terminal_setup(accompaniment=False)
    placements = document["script"]["material_placements"]
    placements["ending-second"] = dict(placements["ending-only"])
    setup = _setup(document)
    first = _evaluate(
        setup,
        ScoreWorkOperation("first", ("ending-only",), ()),
        (),
        {"foregrounds": [_foreground(at=0, duration=3)], "accompaniments": []},
    )
    assert not first.issues and first.terminal_boundary is None
    second = _evaluate(
        setup,
        ScoreWorkOperation("second", ("ending-second",), ()),
        (),
        {"foregrounds": [_foreground(at=6, duration=6)], "accompaniments": []},
        first.notes_by_placement,
    )
    assert not second.issues
    assert second.terminal_boundary == TerminalBoundary("score-unit-release", 6, 12)


def test_terminal_without_foreground_uses_all_accompaniment_and_no_extra_layer():
    document, _, _ = _terminal_setup(accompaniment=False)
    document["script"]["material_placements"]["ending-only"]["role"] = "accompaniment"
    result = _evaluate(
        _setup(document),
        ScoreWorkOperation("ending", ("ending-only",), ()),
        (),
        {"foregrounds": [], "accompaniments": [_accompaniment()]},
    )
    assert not result.issues
    assert result.terminal_boundary == TerminalBoundary("score-unit-release", 0, 12)
    assert set(result.notes_by_placement) == {"ending-only"}


def test_last_attack_must_reach_final_harmony_but_need_not_start_at_its_boundary():
    document, plan, harmonies = _terminal_setup(accompaniment=False)
    harmonies["score-unit-release"] = (
        ScoreHarmony("before", 0, 6, 7, "major"),
        ScoreHarmony("last", 6, 6, 0, "major"),
    )
    operation = ScoreWorkOperation("ending", ("ending-only",), ())
    bad = _evaluate(
        (document, plan, harmonies),
        operation,
        (),
        {"foregrounds": [_foreground(at=0, duration=3)], "accompaniments": []},
    )
    assert bad.issues and "final harmony" in bad.issues[0].message
    good = _evaluate(
        (document, plan, harmonies),
        operation,
        (),
        {"foregrounds": [_foreground(at=8, duration=4)], "accompaniments": []},
    )
    assert not good.issues
    assert good.terminal_boundary.terminal_attack_units == 8


def test_support_is_checked_after_joint_pitch_placement(monkeypatch):
    setup = _terminal_setup()
    original = contracts.place_score_group_accompaniment_events

    def shorten(*args, **kwargs):
        placed = original(*args, **kwargs)
        notes = dict(placed.notes_by_placement)
        notes["support-final"] = tuple(
            replace(note, duration_units=note.duration_units - 1) for note in notes["support-final"]
        )
        return replace(placed, notes_by_placement=notes)

    monkeypatch.setattr(contracts, "place_score_group_accompaniment_events", shorten)
    result = _evaluate(
        setup,
        ScoreWorkOperation("ending", ("ending-only", "support-final"), ()),
        (),
        {"foregrounds": [_foreground(at=6, duration=6)], "accompaniments": [_accompaniment()]},
    )
    assert result.issues and "support" in result.issues[0].message


def test_terminal_prompt_distinguishes_candidate_and_accepted_owners():
    document, plan, harmonies = _terminal_setup()
    for accepted in ({}, {"ending-only": (ScoreNote("last", 6, 6, 72, "upper"),)}):
        operation = ScoreWorkOperation(
            "ending", ("support-final",) if accepted else ("ending-only", "support-final"), ()
        )
        prompt = score_group_prompt(
            document,
            plan,
            operation,
            (),
            harmonies_by_score_unit=harmonies,
            accepted_notes_by_placement=accepted,
            accepted_responses_by_placement={},
        )
        context = json.loads(prompt.split("入力: ", 1)[1])["shared_terminal"]
        assert context["support_placement_id"] == "support-final"
        assert context["owner_placements"][0]["reference_state"] == (
            "accepted" if accepted else "joint_candidate"
        )
        assert (context["accepted_boundary"] is not None) == bool(accepted)


@pytest.mark.parametrize("field", ["terminal_boundary", "terminal_boundary_sha256"])
def test_completed_score_rejects_terminal_tampering(tmp_path, field):
    run = _score_run(tmp_path)
    state_path = run / "outputs/score-state.json"
    state = json.loads(state_path.read_text())
    state[field] = None
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="terminal boundary"):
        load_complete_score_run(run)


def test_previous_complete_bundle_replays_and_cannot_feed_new_stages(tmp_path):
    archive = Path(__file__).parent / "fixtures/scoim/legacy-terminal-9b5e62b/bundle.zip"
    with zipfile.ZipFile(archive) as saved:
        saved.extractall(tmp_path / "old")
    old = tmp_path / "old"
    assert verify_phase8_bundle(old).valid
    assert replay_phase8_bundle(old, tmp_path / "replay").replayed
    for name in ("final.mid", "score.musicxml"):
        assert (old / "artifacts" / name).read_bytes() == (tmp_path / "replay" / name).read_bytes()
    runner = SequencedRunner([])
    result = realize_phase7(Phase7Request(old / "model-runs/score"), runner, tmp_path / "new")
    assert not result.realized and result.outcome == "request_invalid"
    assert not runner.prompts and not (tmp_path / "new").exists()
    run = old / "model-runs/score"
    request = ScoreRequest(
        json.loads((run / "inputs/validated-script.json").read_text()),
        json.loads((run / "inputs/phase3-state.json").read_text()),
        json.loads((run / "inputs/projection-ledger.json").read_text()),
    )
    with pytest.raises(RuntimeError, match="run-spec conflicts"):
        realize_score(request, runner, run)
    assert not runner.prompts


@pytest.mark.parametrize("coordination", ["score", "rolled"])
def test_detached_terminal_notes_keep_their_shared_end(coordination):
    plan, score, performance = _score_inputs()
    unit = score.score_units[0]
    layers = tuple(
        replace(
            layer,
            notes=tuple(
                replace(
                    note,
                    at_units=36 if index == 0 else 0,
                    duration_units=12 if index == 0 else 24,
                    articulations=("staccato",),
                )
                for index, note in enumerate(layer.notes)
            ),
        )
        for layer in unit.score_unit_layers
    )
    score = replace(score, score_units=(replace(unit, score_unit_layers=layers),))
    performance = replace(
        performance,
        section_performances=(
            replace(
                performance.section_performances[0],
                articulation_profile="light",
                coordination_profile=coordination,
            ),
        ),
    )
    boundary = derive_score_terminal_boundary(_script(), plan, score)
    rendered = rendering.render_score_performance(
        _script(), plan, score, performance, terminal_boundary=boundary
    )
    assert max(note.at_ms + note.duration_ms for note in rendered.notes) == 180_000


def test_terminal_pedal_checks_effective_state_across_score_units(monkeypatch):
    document, plan, score, performance = _placement_variation_inputs()
    # One inherited same-harmony phrase spans both leaves.
    score = replace(
        score,
        score_units=tuple(
            replace(
                unit, harmonies=(replace(unit.harmonies[0], root_pitch_class=0, quality="major"),)
            )
            for unit in score.score_units
        ),
    )
    performance = replace(
        performance,
        section_performances=(
            replace(performance.section_performances[0], pedal_profile="phrase_legato"),
        ),
    )
    boundary = derive_score_terminal_boundary(document, plan, score)
    rendered = rendering.render_score_performance(
        document, plan, score, performance, terminal_boundary=boundary
    )
    assert any(
        p.value > 0 and p.source_score_unit_id != boundary.score_unit_id for p in rendered.pedals
    )
    original = rendering.render_role_neutral_performance_with_pedal_sources

    def release_early(*args, **kwargs):
        value, sources = original(*args, **kwargs)
        return replace(
            value,
            pedals=tuple(
                replace(p, at_ms=p.at_ms - 1)
                if p.value == 0 and p.event_id != "pedal-final-up"
                else p
                for p in value.pedals
            ),
        ), sources

    monkeypatch.setattr(
        rendering, "render_role_neutral_performance_with_pedal_sources", release_early
    )
    with pytest.raises(rendering.ScoreRenderingError, match="terminal pedal"):
        rendering.render_score_performance(
            document, plan, score, performance, terminal_boundary=boundary
        )


def test_terminal_without_pedal_accepts_an_earlier_completed_hold():
    document, plan, score, performance = _placement_variation_inputs()
    performance = replace(
        performance,
        section_performances=(
            replace(performance.section_performances[0], pedal_profile="phrase_legato"),
            SectionPerformance("return", pedal_profile="none"),
        ),
    )
    boundary = derive_score_terminal_boundary(document, plan, score)
    rendered = rendering.render_score_performance(
        document, plan, score, performance, terminal_boundary=boundary
    )
    assert any(p.value > 0 for p in rendered.pedals)
    assert not any(p.value > 0 and p.at_ms >= 90_000 for p in rendered.pedals)
