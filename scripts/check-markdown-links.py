from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\((?P<target><[^>]+>|[^\s)]+)")
EXTERNAL_SCHEMES = {"data", "http", "https", "mailto"}


def _tracked_markdown_files(repository_root: Path) -> tuple[Path, ...]:
    completed = subprocess.run(
        ["git", "ls-files", "--", "*.md"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return tuple(repository_root / line for line in completed.stdout.splitlines() if line)


def _local_target(markdown_path: Path, raw_target: str, repository_root: Path) -> Path | None:
    target = (
        raw_target[1:-1] if raw_target.startswith("<") and raw_target.endswith(">") else raw_target
    )
    split = urlsplit(unquote(target))
    if split.scheme in EXTERNAL_SCHEMES or not split.path:
        return None
    if split.scheme:
        return None
    if split.path.startswith("/"):
        return repository_root / split.path.lstrip("/")
    return markdown_path.parent / split.path


def broken_links(markdown_paths: tuple[Path, ...], repository_root: Path) -> tuple[str, ...]:
    failures: list[str] = []
    for markdown_path in markdown_paths:
        text = markdown_path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for match in MARKDOWN_LINK.finditer(line):
                raw_target = match.group("target")
                target = _local_target(markdown_path, raw_target, repository_root)
                if target is not None and not target.resolve().exists():
                    relative = markdown_path.relative_to(repository_root)
                    failures.append(f"{relative}:{line_number}: missing {raw_target}")
    return tuple(failures)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check local targets in tracked Markdown links without network access."
    )
    parser.add_argument("paths", nargs="*", type=Path)
    arguments = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    markdown_paths = (
        tuple(path.resolve() for path in arguments.paths)
        if arguments.paths
        else _tracked_markdown_files(repository_root)
    )
    failures = broken_links(markdown_paths, repository_root)
    if failures:
        print("\n".join(failures))
        return 1
    print(f"Markdown local links: {len(markdown_paths)} files passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
