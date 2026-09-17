"""Codex CLI adapter for one structured SCoIM model call."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .proposal import ProposalRun
from .runner_identity import RunnerIdentity
from .validation import IssueCode, ValidationIssue

_PROVIDER = "openai-codex-cli"
_REQUIRED_EXEC_FLAGS = (
    "--model",
    "--sandbox",
    "--ephemeral",
    "--ignore-user-config",
    "--skip-git-repo-check",
    "--json",
    "--output-schema",
    "--output-last-message",
)
_REASONING_EFFORTS = frozenset({"none", "low", "medium", "high", "xhigh", "max"})


class CodexProcessTimeout(TimeoutError):
    """A started Codex process timed out after its streams were reaped."""

    def __init__(self, stdout: bytes, stderr: bytes) -> None:
        super().__init__("Codex process timed out")
        self.stdout = stdout
        self.stderr = stderr


class CodexProcessStartError(OSError):
    """Codex process creation failed before a model call could start."""


def _resolve_codex_executable() -> tuple[Path, Path | None]:
    command = shutil.which("codex")
    if command is None:
        raise FileNotFoundError("codex command was not found on PATH")
    if sys.platform != "win32":
        return Path(command), None
    configured_root = os.environ.get("CODEX_MANAGED_PACKAGE_ROOT")
    package_root = (
        Path(configured_root)
        if configured_root
        else Path(command).resolve().parent / "node_modules" / "@openai" / "codex"
    )
    executable = (
        package_root
        / "node_modules"
        / "@openai"
        / "codex-win32-x64"
        / "vendor"
        / "x86_64-pc-windows-msvc"
        / "bin"
        / "codex.exe"
    ).resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"managed native codex executable was not found: {executable}")
    return executable, package_root.resolve()


def _run_preflight(command: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(command, capture_output=True, check=False, timeout=10)


def _run_codex_process(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=kwargs["cwd"],
            env=kwargs["env"],
        )
    except OSError as error:
        raise CodexProcessStartError(str(error)) from error
    try:
        stdout, stderr = process.communicate(input=kwargs["input"], timeout=kwargs["timeout"])
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        raise CodexProcessTimeout(stdout or b"", stderr or b"") from None
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _issue(code: IssueCode, message: str) -> tuple[ValidationIssue, ...]:
    return (ValidationIssue(code, message, "/runner"),)


class CodexStructuredRunner:
    """Run one structured SCoIM request through an authenticated Codex CLI."""

    def __init__(
        self,
        *,
        model: str,
        working_directory: str | Path,
        reasoning_effort: str = "medium",
        timeout_seconds: int = 600,
    ) -> None:
        if not model.strip():
            raise ValueError("Codex model must not be empty")
        if reasoning_effort not in _REASONING_EFFORTS:
            raise ValueError("unsupported Codex reasoning effort")
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int):
            raise ValueError("Codex timeout must be an integer")
        if timeout_seconds <= 0:
            raise ValueError("Codex timeout must be positive")
        self.model = model
        self.working_directory = Path(working_directory).resolve()
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self._prepared: tuple[Path, Path | None, str] | None = None

    def identity(self) -> RunnerIdentity:
        """Return settings fixed before the first public model call."""
        return RunnerIdentity(
            provider=_PROVIDER,
            model=self.model,
            model_settings={
                "reasoning_effort": self.reasoning_effort,
                "timeout_seconds": self.timeout_seconds,
                "working_directory": str(self.working_directory),
            },
        )

    def _preflight(self) -> tuple[Path, Path | None, str] | ProposalRun:
        try:
            executable, package_root = _resolve_codex_executable()
            version = _run_preflight([str(executable), "--version"])
        except (OSError, subprocess.SubprocessError) as error:
            return self._not_started(
                IssueCode.RUNNER_UNAVAILABLE,
                f"Codex CLI is unavailable: {error}",
            )
        if version.returncode != 0:
            return self._not_started(
                IssueCode.RUNNER_INCOMPATIBLE,
                "Codex CLI did not report its version",
            )
        version_text = version.stdout.decode("utf-8", errors="replace").strip()
        try:
            help_result = _run_preflight([str(executable), "exec", "--help"])
        except (OSError, subprocess.SubprocessError) as error:
            return self._not_started(IssueCode.RUNNER_INCOMPATIBLE, str(error))
        help_text = help_result.stdout.decode("utf-8", errors="replace")
        missing_flags = [flag for flag in _REQUIRED_EXEC_FLAGS if flag not in help_text]
        if help_result.returncode != 0 or missing_flags:
            detail = ", ".join(missing_flags) or "exec --help failed"
            return self._not_started(
                IssueCode.RUNNER_INCOMPATIBLE,
                f"Codex CLI lacks required capabilities: {detail}",
            )
        try:
            login = _run_preflight([str(executable), "login", "status"])
        except (OSError, subprocess.SubprocessError) as error:
            return self._not_started(IssueCode.RUNNER_UNAUTHENTICATED, str(error))
        if login.returncode != 0:
            return self._not_started(
                IssueCode.RUNNER_UNAUTHENTICATED,
                "Codex CLI is not authenticated",
            )
        return executable, package_root, version_text

    def _not_started(self, code: IssueCode, message: str) -> ProposalRun:
        return ProposalRun(
            provider=_PROVIDER,
            model=self.model,
            model_settings={
                "reasoning_effort": self.reasoning_effort,
                "timeout_seconds": self.timeout_seconds,
            },
            started=False,
            terminal_state="failed",
            raw_response=None,
            events=None,
            stderr=None,
            issues=_issue(code, message),
        )

    def preflight(self) -> tuple[ValidationIssue, ...]:
        """Check the CLI without starting a model process and prepare one call."""

        if self._prepared is not None:
            return ()
        result = self._preflight()
        if isinstance(result, ProposalRun):
            return result.issues
        self._prepared = result
        return ()

    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        """Preflight Codex, execute one model turn, and capture all streams."""
        issues = self.preflight()
        if issues:
            return self._not_started(issues[0].code, issues[0].message)
        assert self._prepared is not None
        executable, package_root, version = self._prepared
        self._prepared = None
        settings = {
            "codex_cli_version": version,
            "reasoning_effort": self.reasoning_effort,
            "timeout_seconds": self.timeout_seconds,
            "working_directory": str(self.working_directory),
        }
        environment = os.environ.copy()
        if package_root is not None:
            environment["CODEX_MANAGED_PACKAGE_ROOT"] = str(package_root)
            environment["CODEX_MANAGED_BY_NPM"] = "1"
        with tempfile.TemporaryDirectory(prefix="scoim-codex-structured-") as temporary_name:
            response_path = Path(temporary_name) / "response.json"
            command = [
                str(executable),
                "exec",
                "--model",
                self.model,
                "--sandbox",
                "read-only",
                "--ephemeral",
                "--ignore-user-config",
                "--skip-git-repo-check",
                "-c",
                f'model_reasoning_effort="{self.reasoning_effort}"',
                "-C",
                str(self.working_directory),
                "--json",
                "--output-schema",
                str(Path(response_schema_path).resolve()),
                "-o",
                str(response_path),
                "-",
            ]
            try:
                completed = _run_codex_process(
                    command,
                    input=prompt.encode("utf-8"),
                    timeout=self.timeout_seconds,
                    cwd=self.working_directory,
                    env=environment,
                )
            except CodexProcessTimeout as error:
                return ProposalRun(
                    _PROVIDER,
                    self.model,
                    settings,
                    True,
                    "timeout",
                    response_path.read_bytes() if response_path.is_file() else None,
                    error.stdout,
                    error.stderr,
                    _issue(IssueCode.RUNNER_TIMEOUT, "Codex structured call timed out"),
                )
            except (CodexProcessStartError, OSError) as error:
                return self._not_started(
                    IssueCode.RUNNER_UNAVAILABLE,
                    f"Codex process could not start: {error}",
                )
            raw_response = response_path.read_bytes() if response_path.is_file() else None
            if completed.returncode != 0:
                return ProposalRun(
                    _PROVIDER,
                    self.model,
                    settings,
                    True,
                    "failed",
                    raw_response,
                    completed.stdout,
                    completed.stderr,
                    _issue(
                        IssueCode.RUNNER_FAILED,
                        f"Codex structured call exited with code {completed.returncode}",
                    ),
                )
            if raw_response is None:
                return ProposalRun(
                    _PROVIDER,
                    self.model,
                    settings,
                    True,
                    "failed",
                    None,
                    completed.stdout,
                    completed.stderr,
                    _issue(IssueCode.RUNNER_FAILED, "Codex returned no final response"),
                )
            return ProposalRun(
                _PROVIDER,
                self.model,
                settings,
                True,
                "completed",
                raw_response,
                completed.stdout,
                completed.stderr,
                (),
            )
