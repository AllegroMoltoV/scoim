"""3分中央候補から曲の変化の低・高端点を決定的に公開する。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from llm_musical_composer.long_form_generation import composition_to_source
from llm_musical_composer.long_form_loop import (
    evaluate_long_form_candidate,
    load_verified_style_target,
)
from llm_musical_composer.material_development import (
    build_material_development_reference_profile_from_directory,
)
from llm_musical_composer.music_dsl import THREE_MINUTE_POLICY, parse_composition
from llm_musical_composer.run_state import (
    atomic_write_bytes,
    atomic_write_json,
    sha256_file,
)
from llm_musical_composer.smf_render import render_composition
from llm_musical_composer.song_change import describe_song_change
from llm_musical_composer.song_change_arrangement import (
    ArrangementTriplet,
    derive_arrangement_triplet,
)
from llm_musical_composer.style_target import EXCLUDED_NAMES


def publish_arrangement_triplet(
    triplet: ArrangementTriplet,
    output_dir: Path,
    *,
    evaluator: Callable[[Any, Path], dict[str, Any]],
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    """端点を一度書き出し、両方が合格した場合だけ最終成果物へ昇格する。"""
    output_dir = Path(output_dir)
    staged = output_dir / "staged"
    reports = {}
    paths = {}
    for label, composition in (("low", triplet.low), ("high", triplet.high)):
        source_path = staged / f"{label}.music.py"
        midi_path = staged / f"{label}.mid"
        atomic_write_bytes(
            source_path,
            (composition_to_source(composition) + "\n").encode("utf-8"),
        )
        render_composition(composition, midi_path)
        reports[label] = {
            "verification": evaluator(composition, midi_path),
            "song_change": describe_song_change(composition),
        }
        paths[label] = (source_path, midi_path)

    status = (
        "pass"
        if all(report["verification"].get("status") == "pass" for report in reports.values())
        else "fail"
    )
    result = {
        "schema_version": 1,
        "status": status,
        "input_hashes": input_hashes,
        "selection": triplet.selection,
        "endpoints": reports,
        "published": {},
    }
    if status == "pass":
        for label, (source_path, midi_path) in paths.items():
            final_source = output_dir / f"{label}.music.py"
            final_midi = output_dir / f"{label}.mid"
            atomic_write_bytes(final_source, source_path.read_bytes())
            atomic_write_bytes(final_midi, midi_path.read_bytes())
            result["published"][label] = {
                "source": str(final_source.resolve()),
                "smf": str(final_midi.resolve()),
                "source_sha256": sha256_file(final_source),
                "smf_sha256": sha256_file(final_midi),
            }
    atomic_write_json(output_dir / "result.json", result)
    return result


def run_song_change_arrangement(
    *,
    base_run: Path,
    output_dir: Path,
    source_dir: Path,
    target_path: Path,
    manifest_path: Path,
    reference_records_path: Path,
) -> dict[str, Any]:
    """保存済み3分曲から低・高端点を作り、全ハードゲートを再検査する。"""
    base_run = Path(base_run).resolve()
    source_path = base_run / "final.music.py"
    verification_path = base_run / "verification.json"
    if not source_path.is_file() or not verification_path.is_file():
        raise ValueError("base run has no published composition and verification")
    base_verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if base_verification.get("status") != "pass":
        raise ValueError("base run verification is not pass")
    composition = parse_composition(
        source_path.read_text(encoding="utf-8"),
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )
    verified = load_verified_style_target(target_path, manifest_path, reference_records_path)
    material_profile = build_material_development_reference_profile_from_directory(
        source_dir, excluded_names=EXCLUDED_NAMES
    )
    triplet = derive_arrangement_triplet(composition)

    def evaluator(candidate: Any, midi_path: Path) -> dict[str, Any]:
        return evaluate_long_form_candidate(
            candidate,
            midi_path,
            style_target=verified["target"],
            material_development_profile=material_profile,
            require_naturalness=True,
            require_voice_structure=True,
        )

    return publish_arrangement_triplet(
        triplet,
        output_dir,
        evaluator=evaluator,
        input_hashes={
            "base_source": sha256_file(source_path),
            "base_verification": sha256_file(verification_path),
            **verified["hashes"],
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="3分曲から曲の変化の低・高端点を作ります。")
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=Path(".appendix/source-smf"))
    parser.add_argument(
        "--style-target",
        type=Path,
        default=Path(".appendix/style-target-feasible-default-v1/style-target.json"),
    )
    parser.add_argument(
        "--style-manifest",
        type=Path,
        default=Path(".appendix/style-target-feasible-default-v1/manifest.json"),
    )
    parser.add_argument(
        "--reference-records",
        type=Path,
        default=Path(".appendix/reference-structure-analysis/files.jsonl"),
    )
    args = parser.parse_args(argv)
    result = run_song_change_arrangement(
        base_run=args.base_run,
        output_dir=args.output_dir,
        source_dir=args.source_dir,
        target_path=args.style_target,
        manifest_path=args.style_manifest,
        reference_records_path=args.reference_records,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
