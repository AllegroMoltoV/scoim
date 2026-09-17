import subprocess
from pathlib import Path

import pytest

import scoim.codex_proposal as codex_module
from scoim.codex_proposal import CodexStructuredRunner
from scoim.validation import IssueCode

_REQUIRED_FLAGS = (
    "--model",
    "--sandbox",
    "--ephemeral",
    "--ignore-user-config",
    "--skip-git-repo-check",
    "--json",
    "--output-schema",
    "--output-last-message",
)


def test_codex_executable_resolution_uses_path_directly_off_windows(
    monkeypatch,
) -> None:
    monkeypatch.setattr(codex_module.shutil, "which", lambda name: "/usr/bin/codex")
    monkeypatch.setattr(codex_module.sys, "platform", "linux")

    executable, package_root = codex_module._resolve_codex_executable()

    assert executable == Path("/usr/bin/codex")
    assert package_root is None


def test_codex_executable_resolution_rejects_missing_path_command(monkeypatch) -> None:
    monkeypatch.setattr(codex_module.shutil, "which", lambda name: None)

    with pytest.raises(FileNotFoundError, match="not found on PATH"):
        codex_module._resolve_codex_executable()


def test_codex_executable_resolution_uses_configured_windows_package(
    tmp_path: Path,
    monkeypatch,
) -> None:
    executable = (
        tmp_path
        / "node_modules"
        / "@openai"
        / "codex-win32-x64"
        / "vendor"
        / "x86_64-pc-windows-msvc"
        / "bin"
        / "codex.exe"
    )
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"")
    monkeypatch.setattr(codex_module.shutil, "which", lambda name: "codex.CMD")
    monkeypatch.setattr(codex_module.sys, "platform", "win32")
    monkeypatch.setenv("CODEX_MANAGED_PACKAGE_ROOT", str(tmp_path))

    actual_executable, package_root = codex_module._resolve_codex_executable()

    assert actual_executable == executable.resolve()
    assert package_root == tmp_path.resolve()


def test_codex_executable_resolution_rejects_missing_windows_native_binary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(codex_module.shutil, "which", lambda name: "codex.CMD")
    monkeypatch.setattr(codex_module.sys, "platform", "win32")
    monkeypatch.setenv("CODEX_MANAGED_PACKAGE_ROOT", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="managed native codex executable"):
        codex_module._resolve_codex_executable()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model": " "}, "model must not be empty"),
        ({"reasoning_effort": "extreme"}, "reasoning effort"),
        ({"timeout_seconds": 1.5}, "timeout must be an integer"),
        ({"timeout_seconds": 0}, "timeout must be positive"),
    ],
)
def test_codex_runner_rejects_invalid_configuration(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "model": "gpt-5.6-sol",
        "working_directory": tmp_path,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        CodexStructuredRunner(**arguments)  # type: ignore[arg-type]


def test_codex_runner_exposes_its_generation_conditions_before_a_call(tmp_path: Path) -> None:
    runner = CodexStructuredRunner(
        model="gpt-5.6-sol",
        working_directory=tmp_path,
        reasoning_effort="high",
        timeout_seconds=321,
    )

    identity = runner.identity()

    assert identity.provider == "openai-codex-cli"
    assert identity.model == "gpt-5.6-sol"
    assert identity.model_settings == {
        "reasoning_effort": "high",
        "timeout_seconds": 321,
        "working_directory": str(tmp_path.resolve()),
    }


def test_codex_preflight_uses_bounded_captured_subprocess(monkeypatch) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, b"ok", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = codex_module._run_preflight(["codex", "--version"])

    assert result.stdout == b"ok"
    assert calls == [
        (
            ["codex", "--version"],
            {"capture_output": True, "check": False, "timeout": 10},
        )
    ]


def test_codex_process_wraps_creation_failure(monkeypatch) -> None:
    def fail_creation(*args: object, **kwargs: object) -> None:
        raise OSError("cannot create process")

    monkeypatch.setattr(subprocess, "Popen", fail_creation)

    with pytest.raises(codex_module.CodexProcessStartError, match="cannot create process"):
        codex_module._run_codex_process(
            ["codex", "exec"],
            input=b"prompt",
            timeout=1,
            cwd=Path.cwd(),
            env={},
        )


def test_codex_process_returns_completed_streams(monkeypatch) -> None:
    class FakeProcess:
        returncode = 0

        def communicate(
            self,
            *,
            input: bytes | None = None,
            timeout: int | None = None,
        ) -> tuple[bytes, bytes]:
            assert input == b"prompt"
            assert timeout == 5
            return b"events", b"warnings"

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: FakeProcess())

    result = codex_module._run_codex_process(
        ["codex", "exec"],
        input=b"prompt",
        timeout=5,
        cwd=Path.cwd(),
        env={},
    )

    assert result.returncode == 0
    assert result.stdout == b"events"
    assert result.stderr == b"warnings"


def _allow_preflight(
    monkeypatch,
    *,
    package_root: Path | None = None,
) -> None:
    monkeypatch.setattr(
        codex_module,
        "_resolve_codex_executable",
        lambda: (Path("codex.exe"), package_root),
    )

    def fake_preflight(command: list[str]) -> subprocess.CompletedProcess[bytes]:
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, b"codex-cli 0.153.4\n", b"")
        if command[-2:] == ["exec", "--help"]:
            return subprocess.CompletedProcess(
                command,
                0,
                " ".join(_REQUIRED_FLAGS).encode("utf-8"),
                b"",
            )
        return subprocess.CompletedProcess(command, 0, b"Logged in\n", b"")

    monkeypatch.setattr(codex_module, "_run_preflight", fake_preflight)


def test_codex_runner_returns_a_completed_structured_response(
    tmp_path: Path,
    monkeypatch,
) -> None:
    schema_path = tmp_path / "schema.json"
    schema_path.write_text('{"type":"object"}', encoding="utf-8")
    working_directory = tmp_path / "work"
    working_directory.mkdir()
    model_commands: list[tuple[list[str], dict[str, object]]] = []

    monkeypatch.setattr(
        codex_module,
        "_resolve_codex_executable",
        lambda: (Path("codex.exe"), None),
    )

    def fake_preflight(command: list[str]) -> subprocess.CompletedProcess[bytes]:
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, b"codex-cli 0.153.4\n", b"")
        if command[-2:] == ["exec", "--help"]:
            return subprocess.CompletedProcess(
                command,
                0,
                " ".join(_REQUIRED_FLAGS).encode("utf-8"),
                b"",
            )
        return subprocess.CompletedProcess(command, 0, b"Logged in using ChatGPT\n", b"")

    def fake_model_process(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        model_commands.append((command, kwargs))
        response_path = Path(command[command.index("-o") + 1])
        response_path.write_bytes(b'{"title":"draft"}')
        return subprocess.CompletedProcess(
            command,
            0,
            b'{"type":"thread.started"}\n{"type":"turn.completed"}\n',
            b"warning\n",
        )

    monkeypatch.setattr(codex_module, "_run_preflight", fake_preflight)
    monkeypatch.setattr(codex_module, "_run_codex_process", fake_model_process)
    runner = CodexStructuredRunner(
        model="gpt-5.6-sol",
        working_directory=working_directory,
        reasoning_effort="low",
        timeout_seconds=600,
    )

    result = runner.run("proposal prompt", schema_path)

    assert result.started is True
    assert result.terminal_state == "completed"
    assert result.raw_response == b'{"title":"draft"}'
    assert result.events == b'{"type":"thread.started"}\n{"type":"turn.completed"}\n'
    assert result.stderr == b"warning\n"
    assert result.provider == "openai-codex-cli"
    assert result.model == "gpt-5.6-sol"
    assert result.model_settings == {
        "codex_cli_version": "codex-cli 0.153.4",
        "reasoning_effort": "low",
        "timeout_seconds": 600,
        "working_directory": str(working_directory.resolve()),
    }
    command, kwargs = model_commands[0]
    assert command[:3] == ["codex.exe", "exec", "--model"]
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "--ephemeral" in command
    assert "--ignore-user-config" in command
    assert "--skip-git-repo-check" in command
    assert command[command.index("--output-schema") + 1] == str(schema_path.resolve())
    assert command[command.index("-C") + 1] == str(working_directory.resolve())
    assert command[command.index("-c") + 1] == 'model_reasoning_effort="low"'
    assert command[-1] == "-"
    assert kwargs["input"] == b"proposal prompt"
    assert kwargs["timeout"] == 600
    assert kwargs["cwd"] == working_directory.resolve()


def test_codex_runner_reuses_a_successful_explicit_preflight_for_the_next_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    preflight_commands: list[list[str]] = []
    _allow_preflight(monkeypatch)
    original_preflight = codex_module._run_preflight

    def counting_preflight(command: list[str]) -> subprocess.CompletedProcess[bytes]:
        preflight_commands.append(command)
        return original_preflight(command)

    def fake_model_process(command: list[str], **kwargs: object):
        Path(command[command.index("-o") + 1]).write_bytes(b"{}")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(codex_module, "_run_preflight", counting_preflight)
    monkeypatch.setattr(codex_module, "_run_codex_process", fake_model_process)
    runner = CodexStructuredRunner(model="gpt-5.6-sol", working_directory=tmp_path)

    assert runner.preflight() == ()
    assert runner.run("prompt", tmp_path / "schema.json").started is True
    assert len(preflight_commands) == 3


def test_codex_runner_reports_missing_executable_before_start(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def missing_executable() -> tuple[Path, Path | None]:
        raise FileNotFoundError("codex is missing")

    monkeypatch.setattr(codex_module, "_resolve_codex_executable", missing_executable)
    runner = CodexStructuredRunner(
        model="gpt-5.6-sol",
        working_directory=tmp_path,
    )

    result = runner.run("proposal prompt", tmp_path / "schema.json")

    assert result.started is False
    assert result.terminal_state == "failed"
    assert [issue.code for issue in result.issues] == [IssueCode.RUNNER_UNAVAILABLE]


def test_codex_runner_reports_missing_required_flag_before_start(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        codex_module,
        "_resolve_codex_executable",
        lambda: (Path("codex.exe"), None),
    )

    def fake_preflight(command: list[str]) -> subprocess.CompletedProcess[bytes]:
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, b"codex-cli 0.153.4\n", b"")
        return subprocess.CompletedProcess(command, 0, b"--model --sandbox", b"")

    monkeypatch.setattr(codex_module, "_run_preflight", fake_preflight)
    runner = CodexStructuredRunner(model="gpt-5.6-sol", working_directory=tmp_path)

    result = runner.run("proposal prompt", tmp_path / "schema.json")

    assert result.started is False
    assert [issue.code for issue in result.issues] == [IssueCode.RUNNER_INCOMPATIBLE]
    assert "--ephemeral" in result.issues[0].message


def test_codex_runner_reports_missing_authentication_before_start(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        codex_module,
        "_resolve_codex_executable",
        lambda: (Path("codex.exe"), None),
    )

    def fake_preflight(command: list[str]) -> subprocess.CompletedProcess[bytes]:
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, b"codex-cli 0.153.4\n", b"")
        if command[-2:] == ["exec", "--help"]:
            return subprocess.CompletedProcess(
                command,
                0,
                " ".join(_REQUIRED_FLAGS).encode("utf-8"),
                b"",
            )
        return subprocess.CompletedProcess(command, 1, b"", b"Not logged in\n")

    monkeypatch.setattr(codex_module, "_run_preflight", fake_preflight)
    runner = CodexStructuredRunner(model="gpt-5.6-sol", working_directory=tmp_path)

    result = runner.run("proposal prompt", tmp_path / "schema.json")

    assert result.started is False
    assert [issue.code for issue in result.issues] == [IssueCode.RUNNER_UNAUTHENTICATED]


def test_codex_runner_reports_model_process_start_failure_as_not_started(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _allow_preflight(monkeypatch)

    def fail_to_start(command: list[str], **kwargs: object) -> None:
        raise OSError("process creation failed")

    monkeypatch.setattr(codex_module, "_run_codex_process", fail_to_start)
    runner = CodexStructuredRunner(model="gpt-5.6-sol", working_directory=tmp_path)

    result = runner.run("proposal prompt", tmp_path / "schema.json")

    assert result.started is False
    assert [issue.code for issue in result.issues] == [IssueCode.RUNNER_UNAVAILABLE]


def test_codex_runner_preserves_events_and_stderr_on_timeout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _allow_preflight(monkeypatch)

    def time_out(command: list[str], **kwargs: object) -> None:
        raise codex_module.CodexProcessTimeout(b'{"type":"thread.started"}\n', b"late\n")

    monkeypatch.setattr(codex_module, "_run_codex_process", time_out)
    runner = CodexStructuredRunner(model="gpt-5.6-sol", working_directory=tmp_path)

    result = runner.run("proposal prompt", tmp_path / "schema.json")

    assert result.started is True
    assert result.terminal_state == "timeout"
    assert result.events == b'{"type":"thread.started"}\n'
    assert result.stderr == b"late\n"
    assert [issue.code for issue in result.issues] == [IssueCode.RUNNER_TIMEOUT]


def test_codex_process_is_killed_and_reaped_after_timeout(monkeypatch) -> None:
    class FakeProcess:
        returncode = 9

        def __init__(self) -> None:
            self.communicate_calls = 0
            self.killed = False

        def communicate(
            self,
            *,
            input: bytes | None = None,
            timeout: int | None = None,
        ) -> tuple[bytes, bytes]:
            self.communicate_calls += 1
            if self.communicate_calls == 1:
                raise subprocess.TimeoutExpired("codex", timeout or 0)
            return b"partial events", b"partial stderr"

        def kill(self) -> None:
            self.killed = True

    process = FakeProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)

    try:
        codex_module._run_codex_process(
            ["codex.exe", "exec"],
            input=b"prompt",
            timeout=1,
            cwd=Path.cwd(),
            env={},
        )
    except codex_module.CodexProcessTimeout as error:
        assert error.stdout == b"partial events"
        assert error.stderr == b"partial stderr"
    else:
        raise AssertionError("timeout was not reported")

    assert process.killed is True
    assert process.communicate_calls == 2


def test_codex_runner_preserves_evidence_on_failed_exit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _allow_preflight(monkeypatch)

    def fail_after_start(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        response_path = Path(command[command.index("-o") + 1])
        response_path.write_bytes(b'{"partial":true}')
        return subprocess.CompletedProcess(command, 7, b"partial events", b"model failed")

    monkeypatch.setattr(codex_module, "_run_codex_process", fail_after_start)
    runner = CodexStructuredRunner(model="gpt-5.6-sol", working_directory=tmp_path)

    result = runner.run("proposal prompt", tmp_path / "schema.json")

    assert result.started is True
    assert result.terminal_state == "failed"
    assert result.raw_response == b'{"partial":true}'
    assert result.events == b"partial events"
    assert result.stderr == b"model failed"
    assert [issue.code for issue in result.issues] == [IssueCode.RUNNER_FAILED]
    assert "code 7" in result.issues[0].message


def test_codex_runner_rejects_success_without_a_final_response(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _allow_preflight(monkeypatch)
    monkeypatch.setattr(
        codex_module,
        "_run_codex_process",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            0,
            b'{"type":"turn.completed"}\n',
            b"",
        ),
    )
    runner = CodexStructuredRunner(model="gpt-5.6-sol", working_directory=tmp_path)

    result = runner.run("proposal prompt", tmp_path / "schema.json")

    assert result.started is True
    assert result.terminal_state == "failed"
    assert result.raw_response is None
    assert [issue.code for issue in result.issues] == [IssueCode.RUNNER_FAILED]
    assert result.issues[0].message == "Codex returned no final response"


def test_codex_runner_passes_native_package_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package_root = tmp_path / "package"
    _allow_preflight(monkeypatch, package_root=package_root)
    environments: list[dict[str, str]] = []

    def capture_environment(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        environments.append(kwargs["env"])  # type: ignore[arg-type]
        response_path = Path(command[command.index("-o") + 1])
        response_path.write_bytes(b'{"title":"draft"}')
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(codex_module, "_run_codex_process", capture_environment)
    runner = CodexStructuredRunner(model="gpt-5.6-sol", working_directory=tmp_path)

    result = runner.run("proposal prompt", tmp_path / "schema.json")

    assert result.terminal_state == "completed"
    assert environments[0]["CODEX_MANAGED_PACKAGE_ROOT"] == str(package_root)
    assert environments[0]["CODEX_MANAGED_BY_NPM"] == "1"
