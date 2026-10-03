import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
from test_scoim_score_rendering import _placement_variation_inputs, _score_inputs, _script

from scoim.pedal_contract import LEGACY_PEDAL
from scoim.performance_ir import SectionPerformance
from scoim.score_ir import ScoreHarmony, ScoreNote
from scoim.score_rendering import (
    ScoreRenderingError,
    check_rendered_performance_smf,
    render_score_performance,
    write_rendered_performance_smf,
)
from scoim.score_timing import QUANTIZED_TIMING


def _inputs(profile="phrase_legato"):
    script, plan, score, performance = _placement_variation_inputs()
    first, last = score.score_units
    first = replace(
        first,
        harmonies=(
            ScoreHarmony("c1", 0, 12, 0, "major"),
            ScoreHarmony("c2", 12, 12, 0, "major"),
            ScoreHarmony("g", 24, 24, 7, "major"),
        ),
        score_unit_layers=(
            replace(
                first.score_unit_layers[0],
                notes=(
                    ScoreNote("first", 0, 6, 72, "upper"),
                    ScoreNote("after-rest", 30, 6, 74, "upper"),
                ),
            ),
            replace(first.score_unit_layers[1], notes=(ScoreNote("bass", 0, 6, 48, "lower"),)),
        ),
    )
    last = replace(last, harmonies=(replace(last.harmonies[0], root_pitch_class=7),))
    performance = replace(
        performance,
        section_performances=(replace(performance.section_performances[0], pedal_profile=profile),),
    )
    return script, plan, replace(score, score_units=(first, last)), performance


def _render(inputs, **kwargs):
    return render_score_performance(*inputs, timing_contract=QUANTIZED_TIMING, **kwargs)


def test_phrase_releases_changed_harmony_waits_for_attack_and_preserves_other_values(tmp_path):
    inputs = _inputs()
    new = _render(inputs)
    old = _render(inputs, pedal_contract=LEGACY_PEDAL)
    assert new.notes == old.notes
    assert new.section_intervals == old.section_intervals
    assert new.duration_ms == old.duration_ms
    assert [(p.at_ms, p.value) for p in new.pedals] == [
        (80, 127),
        (45_000, 0),
        (56_330, 127),
        (180_000, 0),
        (180_000, 0),
    ]
    path = write_rendered_performance_smf(new, tmp_path / "pedal.mid")
    assert check_rendered_performance_smf(new, path)["status"] == "passed"


def test_harmony_keeps_repeated_entry_boundaries_but_does_not_depress_without_attack():
    rendered = _render(_inputs("harmony_legato"))
    assert [(p.at_ms, p.value) for p in rendered.pedals] == [
        (80, 127),
        (22_500, 0),
        (45_000, 0),
        (56_330, 127),
        (90_000, 0),
        (90_080, 127),
        (180_000, 0),
        (180_000, 0),
    ]


def test_phrase_detects_harmony_change_in_a_later_leaf():
    script, plan, score, performance = _inputs()
    first, last = score.score_units
    last = replace(last, harmonies=(replace(last.harmonies[0], quality="minor"),))
    rendered = _render((script, plan, replace(score, score_units=(first, last)), performance))
    assert (90_000, 0) in [(p.at_ms, p.value) for p in rendered.pedals]
    assert (90_080, 127) in [(p.at_ms, p.value) for p in rendered.pedals]


@pytest.mark.parametrize("override", ["none", "phrase_legato"])
def test_child_override_ends_the_inherited_hold(override):
    script, plan, score, performance = _inputs()
    performance = replace(
        performance,
        section_performances=(
            *performance.section_performances,
            SectionPerformance("return", pedal_profile=override),
        ),
    )
    rendered = _render((script, plan, score, performance))
    assert any(p.at_ms == 90_000 and p.value == 0 for p in rendered.pedals)
    assert any(p.at_ms > 90_000 and p.value == 127 for p in rendered.pedals) == (override != "none")


def test_pedal_follows_the_actual_rolled_attack_after_a_rest():
    script, plan, score, performance = _inputs()
    root = replace(performance.section_performances[0], coordination_profile="rolled")
    first = score.score_units[0]
    layers = first.score_unit_layers
    first = replace(
        first,
        score_unit_layers=(
            layers[0],
            replace(
                layers[1], notes=(*layers[1].notes, ScoreNote("rolled-bass", 30, 6, 43, "lower"))
            ),
        ),
    )
    rendered = _render(
        (
            script,
            plan,
            replace(score, score_units=(first, score.score_units[1])),
            replace(performance, section_performances=(root,)),
        )
    )
    attacks = [n.at_ms for n in rendered.notes if 45_000 <= n.at_ms < 90_000]
    assert max(attacks) - min(attacks) == 45
    down = next(p for p in rendered.pedals if p.value == 127 and p.at_ms > 45_000)
    assert down.at_ms == min(attacks) + 80


@pytest.mark.parametrize("duration", [1, 2, 3])
def test_pedal_never_moves_before_attack_to_fit_a_short_interval(duration):
    plan, score, performance = _score_inputs()
    script = _script()
    del script["script"]["material_placements"]["support-first"]
    unit = score.score_units[0]
    unit = replace(
        unit,
        score_unit_layers=(
            replace(unit.score_unit_layers[0], notes=(ScoreNote("only", 0, 48, 72, "upper"),)),
        ),
    )
    performance = replace(
        performance,
        target_duration_ms=duration,
        section_performances=(
            replace(performance.section_performances[0], pedal_profile="phrase_legato"),
        ),
    )
    inputs = (script, plan, replace(score, score_units=(unit,)), performance)
    if duration == 1:
        with pytest.raises(ScoreRenderingError, match="pedal depression cannot fit"):
            _render(inputs)
    else:
        rendered = _render(inputs)
        assert rendered.pedals[0].at_ms == duration - 1
        assert rendered.pedals[0].at_ms > rendered.notes[0].at_ms


@pytest.mark.parametrize("profile", ["phrase_legato", "harmony_legato"])
def test_new_pedal_run_resumes_and_replays_without_models(tmp_path, profile):
    from test_scoim_phase4_realization import SequencedRunner
    from test_scoim_phase7_realization import _response

    from scoim.phase7_realization import Phase7Request, realize_phase7
    from scoim.phase7_state import load_complete_phase7_run
    from scoim.phase8_bundle import Phase8BundleRequest, create_phase8_bundle, replay_phase8_bundle
    from scoim.script_0_4_validation import script_0_4_content_sha256

    archive = Path(__file__).parent / f"fixtures/scoim/legacy-pedal-{profile}-480904e/bundle.zip"
    with zipfile.ZipFile(archive) as saved:
        saved.extractall(tmp_path / "legacy")
    runs = tmp_path / "legacy/model-runs"
    response = _response()
    response["performances"][0]["pedal_profile"] = profile
    response["performances"][0]["handled_directions"][0]["field_names"].append("pedal_profile")
    phase7 = tmp_path / "new-phase7"
    runner = SequencedRunner([response])
    request = Phase7Request(runs / "score")
    assert realize_phase7(request, runner, phase7).realized
    resumed = SequencedRunner([])
    assert realize_phase7(request, resumed, phase7).realized
    assert resumed.prompts == []
    new = load_complete_phase7_run(phase7)
    old = load_complete_phase7_run(runs / "phase7")
    assert new.phase7_schema_version == 4
    assert new.rendered.notes == old.rendered.notes
    assert new.rendered.pedals != old.rendered.pedals
    manifest = json.loads((tmp_path / "legacy/lineage/composition-manifest.json").read_bytes())
    manifest["validated_script_content_sha256"] = script_0_4_content_sha256(new.validated_script)
    bundle = tmp_path / "new-bundle"
    created = create_phase8_bundle(
        Phase8BundleRequest(
            phase7,
            {"phase3": runs / "phase3", "score": runs / "score"},
            "pedal-fixture",
            "new-pedal",
            json.dumps(manifest).encode(),
        ),
        bundle,
    )
    assert created.created, created.issues
    assert json.loads((bundle / "manifest.json").read_text())["schema_version"] == 6
    replayed = replay_phase8_bundle(bundle, tmp_path / "replay")
    assert replayed.replayed, replayed.issues
    for filename in ("final.mid", "score.musicxml"):
        assert (bundle / "artifacts" / filename).read_bytes() == (
            tmp_path / "replay" / filename
        ).read_bytes()
    with pytest.raises(RuntimeError, match="run-spec conflicts"):
        realize_phase7(request, resumed, runs / "phase7")
    assert resumed.prompts == []


def test_public_resume_rejects_the_old_pedal_contract(tmp_path):
    from test_scoim_v2_public_run import _request

    from scoim.v2_public_run import ensure_public_v2_run

    assert ensure_public_v2_run(tmp_path / "run", _request()).ready
    marker = tmp_path / "run/public-run.json"
    saved = json.loads(marker.read_text())
    saved["schema_version"] = 4
    del saved["pedal_contract"]
    marker.write_text(json.dumps(saved), encoding="utf-8")
    result = ensure_public_v2_run(tmp_path / "run", _request())
    assert not result.ready
    assert result.issues[0].code.value == "storage_conflict"
