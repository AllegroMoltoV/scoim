import argparse
import configparser
import io
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath


def _forbidden(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return any(part in {"tests", ".appendix", ".logs", ".prompts", ".tmp"} for part in parts)


def _is_public_music(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return any(
        parts[index : index + 2] == ("examples", "public-music") for index in range(len(parts) - 1)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdist", required=True)
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--version-file", required=True)
    arguments = parser.parse_args()

    with tarfile.open(arguments.sdist, "r:gz") as archive:
        sdist_names = tuple(member.name for member in archive.getmembers() if member.isfile())
    if any(_forbidden(name) for name in sdist_names):
        raise RuntimeError("The sdist contains a private or test path")
    if any(_is_public_music(name) for name in sdist_names):
        raise RuntimeError("The sdist contains a repository-only public music example")
    required_sdist_suffixes = {
        "examples/fixed-aba/approved-script.json",
        "examples/fixed-aba/frozen-response.json",
        "examples/fixed-flow/approved-flow.json",
        "examples/fixed-flow/responses.json",
        "scripts/smoke-installed-public.py",
        "src/scoim/schemas/flow-0.1.0.schema.json",
        "src/scoim/schemas/script-0.2.0.schema.json",
        "src/scoim/schemas/script-compilation-response-1.schema.json",
    }
    for suffix in required_sdist_suffixes:
        if not any(name.endswith(suffix) for name in sdist_names):
            raise RuntimeError(f"The sdist is missing {suffix}")

    with zipfile.ZipFile(arguments.wheel) as archive:
        wheel_names = tuple(archive.namelist())
        entry_points_name = next(
            name for name in wheel_names if name.endswith(".dist-info/entry_points.txt")
        )
        metadata_name = next(name for name in wheel_names if name.endswith(".dist-info/METADATA"))
        entry_points = archive.read(entry_points_name).decode("utf-8")
        metadata = archive.read(metadata_name).decode("utf-8")
    if any(_forbidden(name) for name in wheel_names):
        raise RuntimeError("The wheel contains a private or test path")
    if any(_is_public_music(name) for name in wheel_names):
        raise RuntimeError("The wheel contains a repository-only public music example")
    required_wheel_paths = {
        "scoim/schemas/flow-0.1.0.schema.json",
        "scoim/schemas/script-0.2.0.schema.json",
        "scoim/schemas/script-compilation-response-1.schema.json",
    }
    missing_wheel_paths = required_wheel_paths - set(wheel_names)
    if missing_wheel_paths:
        raise RuntimeError(f"The wheel is missing {sorted(missing_wheel_paths)}")
    allowed_roots = {"scoim", "llm_musical_composer"}
    package_roots = {
        PurePosixPath(name).parts[0]
        for name in wheel_names
        if ".dist-info/" not in name and not name.endswith(".dist-info/")
    }
    if package_roots != allowed_roots:
        raise RuntimeError(f"Unexpected wheel package roots: {sorted(package_roots)}")
    config = configparser.ConfigParser()
    config.read_file(io.StringIO(entry_points))
    scripts = dict(config["console_scripts"])
    if scripts != {"scoim": "scoim.cli:main"}:
        raise RuntimeError(f"Unexpected console scripts: {scripts}")
    parsed_metadata = Parser().parsestr(metadata)
    python_specifiers = {
        item.strip() for item in parsed_metadata.get("Requires-Python", "").split(",")
    }
    if parsed_metadata.get("Name") != "scoim" or python_specifiers != {">=3.13", "<3.14"}:
        raise RuntimeError("The wheel metadata does not match the public package contract")
    version = parsed_metadata.get("Version")
    if not version:
        raise RuntimeError("The wheel metadata has no package version")
    Path(arguments.version_file).write_text(f"{version}\n", encoding="utf-8")
    print(f"sdist files: {len(sdist_names)}")
    print(f"wheel files: {len(wheel_names)}")
    print(f"package version: {version}")
    print("console scripts: scoim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
