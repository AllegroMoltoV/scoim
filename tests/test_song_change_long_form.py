from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_musical_composer import song_change_long_form
from llm_musical_composer.song_change_arrangement import derive_arrangement_triplet
from llm_musical_composer.song_change_long_form import (
    main,
    publish_arrangement_triplet,
    run_song_change_arrangement,
)
from tests.test_song_change_arrangement import _composition


def test_triplet_is_published_only_when_both_endpoints_pass(tmp_path: Path) -> None:
    triplet = derive_arrangement_triplet(_composition())

    result = publish_arrangement_triplet(
        triplet,
        tmp_path / "pass",
        evaluator=lambda composition, path: {
            "status": "pass",
            "note_count": composition.note_count,
        },
        input_hashes={"base": "hash"},
    )

    assert result["status"] == "pass"
    assert set(result["published"]) == {"low", "high"}
    assert (tmp_path / "pass/low.mid").is_file()
    assert (tmp_path / "pass/high.music.py").is_file()
    assert (tmp_path / "pass/result.json").is_file()

    failed = publish_arrangement_triplet(
        triplet,
        tmp_path / "fail",
        evaluator=lambda composition, path: {
            "status": "fail" if path.name == "high.mid" else "pass"
        },
        input_hashes={"base": "hash"},
    )

    assert failed["status"] == "fail"
    assert failed["published"] == {}
    assert not (tmp_path / "fail/low.mid").exists()


def test_run_requires_a_published_verified_base(tmp_path: Path) -> None:
    kwargs = {
        "base_run": tmp_path / "missing",
        "output_dir": tmp_path / "out",
        "source_dir": tmp_path / "source",
        "target_path": tmp_path / "target.json",
        "manifest_path": tmp_path / "manifest.json",
        "reference_records_path": tmp_path / "records.jsonl",
    }
    with pytest.raises(ValueError, match="no published composition"):
        run_song_change_arrangement(**kwargs)

    kwargs["base_run"].mkdir()
    (kwargs["base_run"] / "final.music.py").write_text("composition()", encoding="utf-8")
    (kwargs["base_run"] / "verification.json").write_text(
        json.dumps({"status": "fail"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="verification is not pass"):
        run_song_change_arrangement(**kwargs)


def test_run_loads_inputs_and_rechecks_both_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_run = tmp_path / "base"
    base_run.mkdir()
    (base_run / "final.music.py").write_text("composition()", encoding="utf-8")
    (base_run / "verification.json").write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    center = _composition()
    evaluated: list[tuple[str, bool, bool]] = []
    monkeypatch.setattr(song_change_long_form, "parse_composition", lambda *args, **kwargs: center)
    monkeypatch.setattr(
        song_change_long_form,
        "load_verified_style_target",
        lambda *args: {"target": {"prompt_target": {}}, "hashes": {"target": "hash"}},
    )
    monkeypatch.setattr(
        song_change_long_form,
        "build_material_development_reference_profile_from_directory",
        lambda *args, **kwargs: {"status": "pass"},
    )

    def evaluate(candidate, midi_path, **kwargs):
        evaluated.append(
            (
                midi_path.name,
                kwargs["require_naturalness"],
                kwargs["require_voice_structure"],
            )
        )
        return {"status": "pass"}

    monkeypatch.setattr(song_change_long_form, "evaluate_long_form_candidate", evaluate)

    result = run_song_change_arrangement(
        base_run=base_run,
        output_dir=tmp_path / "out",
        source_dir=tmp_path / "source",
        target_path=tmp_path / "target.json",
        manifest_path=tmp_path / "manifest.json",
        reference_records_path=tmp_path / "records.jsonl",
    )

    assert result["status"] == "pass"
    assert evaluated == [("low.mid", True, True), ("high.mid", True, True)]
    assert result["input_hashes"]["target"] == "hash"


@pytest.mark.parametrize(("status", "exit_code"), [("pass", 0), ("fail", 2)])
def test_main_returns_endpoint_status(
    status: str,
    exit_code: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        song_change_long_form,
        "run_song_change_arrangement",
        lambda **kwargs: {"status": status, "output": str(kwargs["output_dir"])},
    )

    result = main(
        [
            "--base-run",
            str(tmp_path / "base"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert result == exit_code
    assert json.loads(capsys.readouterr().out)["status"] == status
