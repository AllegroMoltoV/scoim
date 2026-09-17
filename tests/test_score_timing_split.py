from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_musical_composer.run_state import sha256_file
from llm_musical_composer.score_timing_split import (
    ScoreTimingSplitError,
    select_development_records,
    select_holdout_records,
    write_development_split,
    write_holdout_split,
)


def _profile(value: float) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "pass",
        "feature_groups": {
            group: {"metrics": {"value": {"kind": "scalar", "values": [value + offset]}}}
            for group, offset in (
                ("performance_texture", 0.0),
                ("rhythm_time", value * 0.1),
                ("pitch_harmony", value * 0.2),
            )
        },
    }


def _record(name: str, value: float, fingerprint: str | None = None) -> dict[str, object]:
    return {
        "name": name,
        "status": "pass",
        "profile": _profile(value),
        "copy_fingerprint": {
            "sequence_sha256": fingerprint or name,
            "token_hashes": [name],
        },
    }


def test_development_selection_is_input_order_independent_and_starts_at_medoid() -> None:
    records = [_record(f"piece-{index}.mid", float(index)) for index in range(7)]

    first = select_development_records(records, count=4)
    second = select_development_records(tuple(reversed(records)), count=4)

    assert first == second
    assert first[0]["selection_role"] == "real_medoid"
    assert [item["name"] for item in first] == [item["name"] for item in second]


def test_exact_copy_is_recorded_and_not_selected_twice() -> None:
    records = [
        _record("a.mid", 0.0, "same"),
        _record("a-copy.mid", 0.0, "same"),
        _record("b.mid", 1.0),
        _record("c.mid", 2.0),
        _record("d.mid", 3.0),
    ]

    selected = select_development_records(records, count=3)

    names = {item["name"] for item in selected}
    assert not {"a.mid", "a-copy.mid"} <= names
    assert any(item["exact_copy_conflicts"] for item in selected)


def test_holdout_selection_uses_nearest_unused_peer_per_development_anchor() -> None:
    records = [_record(f"piece-{index}.mid", float(index)) for index in range(12)]
    development = ["piece-2.mid", "piece-8.mid"]

    first = select_holdout_records(records, development_names=development)
    second = select_holdout_records(
        tuple(reversed(records)), development_names=development
    )

    assert first == second
    assert [item["anchor_name"] for item in first] == development
    selected = [item["name"] for item in first]
    assert len(selected) == len(set(selected)) == 2
    assert not set(selected) & set(development)
    assert all(item["selection_role"] == "nearest_unused_peer" for item in first)


def test_holdout_selection_excludes_exact_copy_of_development() -> None:
    records = [
        _record("anchor.mid", 0.0, "same"),
        _record("copy.mid", 0.0, "same"),
        _record("near.mid", 0.1),
        _record("far.mid", 10.0),
    ]

    selected = select_holdout_records(records, development_names=["anchor.mid"])

    assert selected[0]["name"] == "near.mid"


def test_split_stages_only_development_files_and_does_not_materialize_holdout(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    records = [_record(f"piece-{index}.mid", float(index)) for index in range(5)]
    source_hashes = {}
    for record in records:
        path = source / str(record["name"])
        path.write_bytes(b"MThd" + str(record["name"]).encode())
        source_hashes[str(record["name"])] = sha256_file(path)
    profiles = tmp_path / "files.jsonl"
    profiles.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"schema_version": 1, "status": "pass", "source_count": 5}),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "pass",
                "inputs": source_hashes,
                "outputs": {
                    "files.jsonl": sha256_file(profiles),
                    "summary.json": sha256_file(summary),
                },
            }
        ),
        encoding="utf-8",
    )

    result = write_development_split(
        source_dir=source,
        profile_manifest=manifest,
        profile_records=profiles,
        profile_summary=summary,
        output_dir=tmp_path / "output",
        count=3,
    )

    assert result["development_count"] == 3
    output = tmp_path / "output"
    assert len(list((output / "development-smf").glob("*.mid"))) == 3
    assert not (output / "holdout.jsonl").exists()
    holdout_rule = json.loads((output / "holdout-rule.json").read_text())
    assert "names" not in holdout_rule
    assert holdout_rule["materialized"] is False


def test_split_rejects_source_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.mid").write_bytes(b"changed")
    records = tmp_path / "files.jsonl"
    records.write_text(json.dumps(_record("a.mid", 0.0)) + "\n", encoding="utf-8")
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"schema_version": 1, "status": "pass", "source_count": 1}),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "pass",
                "inputs": {"a.mid": "0" * 64},
                "outputs": {
                    "files.jsonl": sha256_file(records),
                    "summary.json": sha256_file(summary),
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ScoreTimingSplitError, match="source SHA-256 mismatch"):
        write_development_split(
            source_dir=source,
            profile_manifest=manifest,
            profile_records=records,
            profile_summary=summary,
            output_dir=tmp_path / "output",
            count=1,
        )


def test_holdout_split_materializes_role_aware_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    records = [_record(f"piece-{index}.mid", float(index)) for index in range(6)]
    source_hashes = {}
    for record in records:
        path = source / str(record["name"])
        path.write_bytes(b"MThd" + str(record["name"]).encode())
        source_hashes[str(record["name"])] = sha256_file(path)
    profiles = tmp_path / "files.jsonl"
    profiles.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"schema_version": 1, "status": "pass", "source_count": 6}),
        encoding="utf-8",
    )
    profile_manifest = tmp_path / "profile-manifest.json"
    profile_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "pass",
                "inputs": source_hashes,
                "outputs": {
                    "files.jsonl": sha256_file(profiles),
                    "summary.json": sha256_file(summary),
                },
            }
        ),
        encoding="utf-8",
    )
    development = tmp_path / "development-selection.json"
    development.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "development": [
                    {"name": "piece-1.mid"},
                    {"name": "piece-4.mid"},
                ],
            }
        ),
        encoding="utf-8",
    )
    method_manifest = tmp_path / "method-manifest.json"
    method_manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "llm_musical_composer.score_timing_split.verify_method_manifest",
        lambda **_arguments: "f" * 64,
    )

    result = write_holdout_split(
        source_dir=source,
        profile_manifest=profile_manifest,
        profile_records=profiles,
        profile_summary=summary,
        development_selection=development,
        repository_root=Path.cwd(),
        method_manifest=method_manifest,
        output_dir=tmp_path / "output",
        expected_count=2,
    )

    assert result == {
        "status": "pass",
        "source_count": 6,
        "holdout_count": 2,
        "holdout_materialized": True,
    }
    manifest = json.loads((tmp_path / "output" / "manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["split_role"] == "holdout"
    assert len(manifest["staged_smf"]) == 2
    assert manifest["inputs"]["method_manifest"] == "f" * 64
    assert len(list((tmp_path / "output" / "staged-smf").glob("*.mid"))) == 2
