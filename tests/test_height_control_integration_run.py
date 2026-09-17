from __future__ import annotations

import json
from pathlib import Path

import pytest

import llm_musical_composer.height_control_integration_run as integration_run
from llm_musical_composer.height_control_integration_run import (
    HeightIntegrationError,
    build_run_id,
    resolve_accepted_height,
    run_height_integration,
)
from llm_musical_composer.run_state import sha256_file

ROOT = Path(__file__).parents[1]
BASELINE_ROOT = (
    ROOT
    / ".appendix"
    / "key-release-calibration-v1"
    / "reference-v7-a2-joint-texture-v4"
)
REFERENCE_DIR = ROOT / ".appendix" / "reference-profile-v1"
CONTROL_DIR = ROOT / ".appendix" / "control-reference-baseline-v3"


def _write_json(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return path


def _input_spec(tmp_path: Path) -> Path:
    run_spec = BASELINE_ROOT / "run-spec.json"
    manifest = BASELINE_ROOT / "manifest.json"
    return _write_json(
        tmp_path / "input-spec.json",
        {
            "schema_version": 1,
            "baseline_root": BASELINE_ROOT.relative_to(ROOT).as_posix(),
            "run_spec": {
                "path": run_spec.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(run_spec),
            },
            "manifest": {
                "path": manifest.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(manifest),
            },
            "artifacts": {
                "piece_plan": "inputs/piece-plan.dsl",
                "score_spec": "outputs/score-spec.dsl",
                "performance_spec": "outputs/performance-spec.dsl",
                "final_smf": "outputs/final.mid",
            },
            "known_limit": (
                "accepted only for texture, key release, and "
                "melody-accompaniment alignment"
            ),
        },
    )


def _request(tmp_path: Path, value: float) -> Path:
    return _write_json(
        tmp_path / f"request-{value}.json",
        {
            "schema_version": 1,
            "preset": "solo_piano_3m_v1",
            "reference": "TasteOfFall.mid",
            "controls": {"高さ": value},
        },
    )


def test_resolution_uses_only_fully_accepted_candidates() -> None:
    resolution = resolve_accepted_height(
        0.0,
        observations=[
            {"semitones": 0, "height_mean": 58.0, "accepted": False},
            {"semitones": -1, "height_mean": 57.4, "accepted": True},
            {"semitones": 1, "height_mean": 59.4, "accepted": True},
        ],
        minimum=42.0,
        maximum=74.0,
    )

    assert resolution.semitones == -1
    assert resolution.safe_mean_range == (57.4, 59.4)


def test_run_id_uses_semantic_inputs_and_implementation_hashes() -> None:
    semantic = {"request": {"controls": {"height": 0.25}}}
    first = build_run_id(semantic, {"runner.py": "a" * 64})

    assert build_run_id(
        {"request": {"controls": {"height": 0.25}}},
        {"runner.py": "a" * 64},
    ) == first
    assert build_run_id(semantic, {"runner.py": "b" * 64}) != first


def test_input_spec_hash_mismatch_stops_before_output(tmp_path: Path) -> None:
    input_spec = json.loads(_input_spec(tmp_path).read_text(encoding="utf-8"))
    input_spec["run_spec"]["sha256"] = "0" * 64
    input_spec_path = _write_json(tmp_path / "bad-input-spec.json", input_spec)
    output_root = tmp_path / "runs"

    with pytest.raises(HeightIntegrationError, match="run spec hash mismatch"):
        run_height_integration(
            request_path=_request(tmp_path, -0.25),
            input_spec_path=input_spec_path,
            reference_dir=REFERENCE_DIR,
            control_dir=CONTROL_DIR,
            output_root=output_root,
        )

    assert not output_root.exists()


def test_low_and_high_requests_reach_distinct_quality_checked_smf(
    tmp_path: Path,
) -> None:
    input_spec = _input_spec(tmp_path)
    low = run_height_integration(
        request_path=_request(tmp_path, -0.25),
        input_spec_path=input_spec,
        reference_dir=REFERENCE_DIR,
        control_dir=CONTROL_DIR,
        output_root=tmp_path / "runs",
    )
    high = run_height_integration(
        request_path=_request(tmp_path, 0.25),
        input_spec_path=input_spec,
        reference_dir=REFERENCE_DIR,
        control_dir=CONTROL_DIR,
        output_root=tmp_path / "runs",
    )

    assert [low["resolution"]["semitones"], high["resolution"]["semitones"]] == [
        -2,
        6,
    ]
    assert low["status"] == high["status"] == "achieved"
    assert low["height"]["mean"] < high["height"]["mean"]
    assert low["controls"]["raw"]["発音頻度"] == high["controls"]["raw"]["発音頻度"]
    for result in (low, high):
        run_dir = Path(result["run_dir"])
        assert result["artifact_role"] == "normal_candidate"
        assert (run_dir / result["artifacts"]["final_smf"]).is_file()
        assert result["quality"]["passes"]
        assert result["smf_round_trip"]["status"] == "passed"
        assert "path" not in result["smf_round_trip"]


def test_same_meaning_in_different_output_roots_has_same_result(tmp_path: Path) -> None:
    input_spec = _input_spec(tmp_path)
    request = _request(tmp_path, 0.25)

    first = run_height_integration(
        request_path=request,
        input_spec_path=input_spec,
        reference_dir=REFERENCE_DIR,
        control_dir=CONTROL_DIR,
        output_root=tmp_path / "first",
    )
    second = run_height_integration(
        request_path=request,
        input_spec_path=input_spec,
        reference_dir=REFERENCE_DIR,
        control_dir=CONTROL_DIR,
        output_root=tmp_path / "second",
    )

    assert first["run_id"] == second["run_id"]
    first.pop("run_dir")
    second.pop("run_dir")
    assert first == second


def test_unreachable_request_does_not_publish_normal_final_smf(
    tmp_path: Path,
) -> None:
    result = run_height_integration(
        request_path=_request(tmp_path, -1.0),
        input_spec_path=_input_spec(tmp_path),
        reference_dir=REFERENCE_DIR,
        control_dir=CONTROL_DIR,
        output_root=tmp_path / "runs",
    )

    run_dir = Path(result["run_dir"])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert result["status"] == "unreachable"
    assert result["artifact_role"] == "diagnostic_only"
    assert "final_smf" not in result["artifacts"]
    assert not (run_dir / "outputs/final.mid").exists()
    assert "outputs/final.mid" not in manifest["outputs"]
    assert result["resolution"]["safe_normalized_range"][0] > -1.0


def test_all_candidate_rejections_leave_failed_manifest_and_reasons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(integration_run, "candidate_transpositions", lambda _score: (0,))

    def reject_candidate(*_args: object, **_kwargs: object) -> object:
        raise ValueError("synthetic candidate rejection")

    monkeypatch.setattr(integration_run, "build_height_candidate", reject_candidate)
    result = run_height_integration(
        request_path=_request(tmp_path, 0.0),
        input_spec_path=_input_spec(tmp_path),
        reference_dir=REFERENCE_DIR,
        control_dir=CONTROL_DIR,
        output_root=tmp_path / "runs",
    )

    run_dir = Path(result["run_dir"])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    candidates = json.loads((run_dir / "candidates.json").read_text(encoding="utf-8"))

    assert result["status"] == "failed"
    assert result["accepted_candidate_count"] == 0
    assert result["candidate_rejections"] == [
        {
            "reasons": ["synthetic candidate rejection"],
            "semitones": 0,
        }
    ]
    assert candidates[0]["rejection_reasons"] == ["synthetic candidate rejection"]
    assert manifest["status"] == "failed"
    assert manifest["outputs"]["candidates.json"] == sha256_file(
        run_dir / "candidates.json"
    )
    assert manifest["outputs"]["result.json"] == sha256_file(run_dir / "result.json")
    assert not (run_dir / "outputs/final.mid").exists()


def test_cli_reports_failed_run_without_assuming_a_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        integration_run,
        "run_height_integration",
        lambda **_kwargs: {
            "status": "failed",
            "run_id": "failed-run",
            "run_dir": str(tmp_path / "failed-run"),
        },
    )

    exit_code = integration_run.main(
        [
            "--request",
            "request.json",
            "--input-spec",
            "input-spec.json",
            "--reference-dir",
            "reference",
            "--control-dir",
            "control",
            "--output-root",
            "runs",
        ]
    )

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out) == {
        "run_dir": str(tmp_path / "failed-run"),
        "run_id": "failed-run",
        "semitones": None,
        "status": "failed",
    }
