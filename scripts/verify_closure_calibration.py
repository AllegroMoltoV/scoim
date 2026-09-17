"""生成済み終止感校正セットを実装と独立した入口から検証する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from llm_musical_composer.closure_calibration import (
    ALLOWED_ANSWERS,
    build_conflicts,
    read_performance,
)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest().upper()


def verify(output_root: Path, audit_root: Path) -> dict[str, object]:
    manifest = load_json(output_root / "selection-manifest.json")
    split_manifest = load_json(output_root / "split-manifest.json")
    evaluation = load_json(output_root / "evaluation-form.json")
    pairs = manifest["pairs"]
    assert isinstance(pairs, list) and len(pairs) == 16
    selected = {str(pair["source"]) for pair in pairs}
    assert len(selected) == 16
    assert selected.isdisjoint({"rut.mid", "aimusic01.mid"})
    splits = split_manifest["splits"]
    assert isinstance(splits, dict)
    assert list(splits.values()).count("few_shot") == 8
    assert list(splits.values()).count("holdout") == 8
    evaluation_answers = evaluation["answers"]
    assert isinstance(evaluation_answers, list) and len(evaluation_answers) == 16
    assert evaluation["allowed_values"] == list(ALLOWED_ANSWERS)

    evaluation_groups = load_json(audit_root / "evaluation-groups.json")
    duplicate_candidates = load_json(audit_root / "duplicate-candidates.json")
    conflicts = build_conflicts(evaluation_groups, duplicate_candidates)
    for name in selected:
        assert selected.isdisjoint(conflicts.get(name, frozenset()))

    midi_count = 0
    for pair in pairs:
        pair_id = str(pair["pair_id"])
        assert pair["split"] == splits[pair_id]
        assert set(pair["roles"]) == {"A", "B"}
        judge_path = output_root / "judge" / f"{pair_id}.json"
        image_path = output_root / "judge" / f"{pair_id}.png"
        judge_text = judge_path.read_text(encoding="utf-8")
        assert not any(source in judge_text for source in selected)
        judge = json.loads(judge_text)
        assert set(judge["samples"]) == {"A", "B"}
        assert image_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert file_hash(judge_path) == pair["judge_sha256"]
        assert file_hash(image_path) == pair["piano_roll_sha256"]
        for label in ("A", "B"):
            artifact = pair["artifacts"][label]
            midi_path = output_root / artifact["path"]
            assert file_hash(midi_path) == artifact["sha256"]
            performance = read_performance(midi_path)
            assert performance.notes
            assert all(
                note.onset_ms + note.duration_ms <= performance.duration_ms
                for note in performance.notes
            )
            assert performance.pedals and performance.pedals[-1].value == 0
            midi_count += 1
    return {
        "status": "affirmative_evidence",
        "pair_count": len(pairs),
        "midi_count": midi_count,
        "few_shot_count": list(splits.values()).count("few_shot"),
        "holdout_count": list(splits.values()).count("holdout"),
        "tree_sha256": tree_hash(output_root),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("audit_root", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.output_root, arguments.audit_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
