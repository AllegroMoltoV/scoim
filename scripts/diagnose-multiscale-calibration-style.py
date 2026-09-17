"""3分校正SMFを保存済み参照プロファイルと比較する。"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from llm_musical_composer.reference_profile import (
    extract_reference_profile,
    load_reference_piece,
    profile_distances,
)

DEFAULT_RUN_DIR = Path(".appendix/multiscale-calibration-run")
REFERENCE_FILES = Path(".appendix/reference-profile-v1/files.jsonl")
ANCHOR = "GoodMedicineTastesBitter.mid"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    smf_path = run_dir / "outputs/final.mid"
    piece = load_reference_piece(smf_path)
    candidate = extract_reference_profile(piece)
    references: dict[str, dict[str, object]] = {}
    for line in REFERENCE_FILES.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("status") == "pass":
            references[item["name"]] = item["profile"]
    comparisons = []
    for name, profile in references.items():
        distances = profile_distances(candidate, profile)
        comparisons.append(
            {
                "name": name,
                "distances": distances,
                "mean_distance": statistics.fmean(distances.values()),
            }
        )
    comparisons.sort(key=lambda item: (item["mean_distance"], item["name"]))
    start = min(note.onset_ms for note in piece.notes)
    end = max(note.onset_ms + note.duration_ms for note in piece.notes)
    duration_seconds = (end - start) / 1000
    attacks = len({note.onset_ms for note in piece.notes})
    result = {
        "candidate": str(smf_path.resolve()),
        "duration_seconds": duration_seconds,
        "note_rate_hz": len(piece.notes) / duration_seconds,
        "attack_rate_hz": attacks / duration_seconds,
        "mean_velocity": statistics.fmean(note.velocity for note in piece.notes),
        "anchor": next(item for item in comparisons if item["name"] == ANCHOR),
        "nearest_references": comparisons[:7],
        "limitation": (
            "profile distance is diagnostic only; it does not prove audible style similarity "
            "or long-form narrative quality"
        ),
    }
    output = run_dir / "style-diagnostic.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
