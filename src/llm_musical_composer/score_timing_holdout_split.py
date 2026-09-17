"""凍結済み方法でScoreTiming保留標本を物質化する。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from llm_musical_composer.score_timing_split import write_holdout_split


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--profile-manifest", type=Path, required=True)
    parser.add_argument("--profile-records", type=Path, required=True)
    parser.add_argument("--profile-summary", type=Path, required=True)
    parser.add_argument("--development-selection", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--method-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    result = write_holdout_split(
        source_dir=arguments.source_dir,
        profile_manifest=arguments.profile_manifest,
        profile_records=arguments.profile_records,
        profile_summary=arguments.profile_summary,
        development_selection=arguments.development_selection,
        repository_root=arguments.repository_root,
        method_manifest=arguments.method_manifest,
        output_dir=arguments.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
