import argparse
import hashlib
import io
import json
import xml.etree.ElementTree as ET
from contextlib import redirect_stdout
from pathlib import Path
from typing import ClassVar

import mido

import scoim
import scoim.cli as cli_module
from scoim.proposal import ProposalRun
from scoim.realization import realize_solo_piano_3m
from scoim.runner_identity import RunnerIdentity


class FixedCodexRunner:
    responses: ClassVar[list[dict[str, object]]] = []
    instances: ClassVar[list["FixedCodexRunner"]] = []

    def __init__(self, **kwargs: object) -> None:
        del kwargs
        self.calls = 0
        self.__class__.instances.append(self)

    def preflight(self) -> tuple[scoim.ValidationIssue, ...]:
        return ()

    def identity(self) -> RunnerIdentity:
        return RunnerIdentity("fixed-smoke", "fixed-model", {})

    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        del prompt, response_schema_path
        response = self.responses[self.calls]
        self.calls += 1
        return ProposalRun(
            provider="fixed-smoke",
            model="fixed-model",
            model_settings={},
            started=True,
            terminal_state="completed",
            raw_response=json.dumps(response, ensure_ascii=False).encode("utf-8"),
            events=b"",
            stderr=b"",
            issues=(),
        )


def _run_cli(arguments: list[str]) -> dict[str, object]:
    output = io.StringIO()
    with redirect_stdout(output):
        exit_code = cli_module.main(arguments)
    result = json.loads(output.getvalue())
    if exit_code != 0:
        raise RuntimeError(result)
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_replayed_artifacts(
    generated_root: Path,
    replay_root: Path,
    names: tuple[str, ...],
    *,
    profile: str,
) -> None:
    for name in names:
        generated = generated_root / "trial" / "artifacts" / name
        replay = replay_root / "artifacts" / name
        if not generated.is_file() or not replay.is_file() or _sha256(generated) != _sha256(replay):
            raise RuntimeError(f"The replayed {profile} artifact does not match: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--v2-flow", type=Path, required=True)
    parser.add_argument("--v2-responses", type=Path, required=True)
    parser.add_argument("--legacy-script", type=Path, required=True)
    parser.add_argument("--legacy-response", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--forbid-root", type=Path, required=True)
    arguments = parser.parse_args()

    package_path = Path(scoim.__file__).resolve()
    if package_path.is_relative_to(arguments.forbid_root.resolve()):
        raise RuntimeError(f"SCoIM was imported from the source tree: {package_path}")
    FixedCodexRunner.instances = []
    cli_module.CodexStructuredRunner = FixedCodexRunner

    checked = _run_cli(["check", str(arguments.flow)])
    shown = _run_cli(["show", str(arguments.flow), "scene-002"])
    if checked.get("valid") is not True or shown.get("shown") is not True:
        raise RuntimeError("The installed flow could not be checked and shown")

    FixedCodexRunner.responses = json.loads(arguments.responses.read_text(encoding="utf-8"))
    v1_generated_root = arguments.output / "v1-generated"
    v1_realized = _run_cli(
        [
            "realize",
            str(arguments.flow),
            "--model",
            "fixed-model",
            "--trial-id",
            "installed-public-smoke",
            "--profile",
            "solo_piano_3m_v1",
            "--output",
            str(v1_generated_root),
        ]
    )
    v1_replay_root = arguments.output / "v1-replayed"
    v1_replayed = _run_cli(
        ["realize", str(v1_generated_root / "trial"), "--output", str(v1_replay_root)]
    )
    if v1_realized.get("succeeded") is not True or v1_replayed.get("succeeded") is not True:
        raise RuntimeError("The installed v1 compatibility path did not complete")
    if len(FixedCodexRunner.instances) != 1:
        raise RuntimeError("The v1 offline replay unexpectedly created a model runner")
    v1_runner = FixedCodexRunner.instances[0]
    v1_response_count = len(FixedCodexRunner.responses)
    if v1_runner.calls != v1_response_count:
        raise RuntimeError(f"Unused v1 fixed responses: {v1_runner.calls}/{v1_response_count}")
    _assert_replayed_artifacts(
        v1_generated_root,
        v1_replay_root,
        ("phase-04-melody.mid", "phase-06-score.mid", "final.mid"),
        profile="v1",
    )

    FixedCodexRunner.responses = json.loads(arguments.v2_responses.read_text(encoding="utf-8"))
    v2_generated_root = arguments.output / "v2-generated"
    v2_realized = _run_cli(
        [
            "realize",
            str(arguments.v2_flow),
            "--model",
            "fixed-model",
            "--trial-id",
            "installed-public-v2-smoke",
            "--output",
            str(v2_generated_root),
        ]
    )
    v2_replay_root = arguments.output / "v2-replayed"
    v2_replayed = _run_cli(
        ["realize", str(v2_generated_root / "trial"), "--output", str(v2_replay_root)]
    )
    if v2_realized.get("succeeded") is not True or v2_replayed.get("succeeded") is not True:
        raise RuntimeError("The installed default v2 realization did not complete")
    composition_manifest = json.loads(
        (v2_generated_root / "composition" / "manifest.json").read_text(encoding="utf-8")
    )
    if composition_manifest.get("target_profile") != "solo_piano_3m_v2":
        raise RuntimeError("The installed public smoke did not exercise the default v2 profile")
    if len(FixedCodexRunner.instances) != 2:
        raise RuntimeError("The v2 offline replay unexpectedly created a model runner")
    v2_runner = FixedCodexRunner.instances[1]
    v2_response_count = len(FixedCodexRunner.responses)
    if v2_runner.calls != v2_response_count:
        raise RuntimeError(f"Unused v2 fixed responses: {v2_runner.calls}/{v2_response_count}")
    _assert_replayed_artifacts(
        v2_generated_root,
        v2_replay_root,
        ("score.musicxml", "final.mid"),
        profile="v2",
    )
    ET.parse(v2_generated_root / "trial" / "artifacts" / "score.musicxml")
    midi = mido.MidiFile(v2_generated_root / "trial" / "artifacts" / "final.mid")
    if not midi.tracks:
        raise RuntimeError("The generated v2 SMF has no tracks")

    legacy_document = json.loads(arguments.legacy_script.read_text(encoding="utf-8"))
    legacy_response = json.loads(arguments.legacy_response.read_text(encoding="utf-8"))
    legacy = realize_solo_piano_3m(
        legacy_document,
        legacy_response,
        arguments.output / "legacy",
    )
    if not legacy.realized or legacy.smf_path is None:
        raise RuntimeError("The legacy fixed example could not be replayed")
    expected_legacy_hash = "752e8379b32d9a0908a424d9c20caaca8703d349c40e0f60f6b3d456c5c3a80a"
    if _sha256(legacy.smf_path) != expected_legacy_hash:
        raise RuntimeError("The legacy fixed example SMF hash changed")

    print(
        json.dumps(
            {
                "package_path": str(package_path),
                "v1_model_calls": v1_runner.calls,
                "v1_generated": v1_realized["artifacts"],
                "v1_replayed": v1_replayed["artifacts"],
                "v2_model_calls": v2_runner.calls,
                "v2_generated": v2_realized["artifacts"],
                "v2_replayed": v2_replayed["artifacts"],
                "legacy_smf": str(legacy.smf_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
