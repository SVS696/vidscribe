"""CLI provider abstractions."""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class ProviderError(RuntimeError):
    """Raised when a CLI provider cannot complete a correction request."""


@dataclass(frozen=True)
class ProviderResponse:
    """Normalized response returned by all CLI-backed providers."""

    text: str
    raw_json: dict[str, Any]
    cost_estimate: float | None
    duration_s: float


class Provider(Protocol):
    """Protocol for transcript correction providers."""

    def correct(
        self,
        prompt: str,
        frame_paths: list[Path],
        timeout: int,
    ) -> ProviderResponse:
        """Correct a transcript prompt with optional frame references."""


@dataclass(frozen=True)
class ClaudeCLIProvider:
    """Claude Code CLI provider."""

    model: str = "sonnet"

    def correct(
        self,
        prompt: str,
        frame_paths: list[Path],
        timeout: int,
    ) -> ProviderResponse:
        command = [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "json",
            "--max-turns",
            "1",
            "--no-session-persistence",
            "--strict-mcp-config",
            "--mcp-config",
            "{}",
            "--permission-mode",
            "dontAsk",
            "--tools",
            "Read",
            "--disallowed-tools",
            "Bash,Edit,MultiEdit,Write,NotebookEdit",
        ]
        if self.model:
            command.extend(["--model", self.model])
        allowed_dirs = _parent_dirs(frame_paths)
        if allowed_dirs:
            command.append("--add-dir")
            command.extend(str(path) for path in allowed_dirs)
        return _run_isolated_provider(command, timeout=timeout)


@dataclass(frozen=True)
class CodexCLIProvider:
    """Codex CLI provider."""

    model: str = "gpt-5.5"

    def correct(
        self,
        prompt: str,
        frame_paths: list[Path],
        timeout: int,
    ) -> ProviderResponse:
        command = [
            "codex",
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
        ]
        if self.model:
            command.extend(["--model", self.model])
        for frame_path in frame_paths:
            command.extend(["--image", str(frame_path.resolve())])
        command.append(prompt)
        return _run_codex_provider(command, timeout=timeout)


@dataclass(frozen=True)
class OllamaProvider:
    """Local Ollama CLI provider."""

    model: str = "qwen2-vl:7b"

    def correct(
        self,
        prompt: str,
        frame_paths: list[Path],
        timeout: int,
    ) -> ProviderResponse:
        command = ["ollama", "run", self.model, prompt]
        command.extend(str(path) for path in frame_paths)
        return _run_provider(command, timeout=timeout)


def make(name: str, **opts: Any) -> Provider:
    """Create a provider by name."""

    normalized = name.strip().lower()
    if normalized == "claude":
        return ClaudeCLIProvider(**opts)
    if normalized == "codex":
        return CodexCLIProvider(**opts)
    if normalized == "ollama":
        return OllamaProvider(**opts)
    raise ValueError(f"Unsupported provider: {name}")


def _parent_dirs(paths: list[Path]) -> list[Path]:
    dirs: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        parent = path.resolve().parent
        if parent in seen:
            continue
        seen.add(parent)
        dirs.append(parent)
    return dirs


def _run_provider(command: list[str], timeout: int) -> ProviderResponse:
    return _run_provider_in_cwd(command, timeout=timeout, cwd=None)


def _run_isolated_provider(command: list[str], timeout: int) -> ProviderResponse:
    with tempfile.TemporaryDirectory(prefix="vidscribe-provider-") as temp_dir:
        return _run_provider_in_cwd(command, timeout=timeout, cwd=Path(temp_dir))


def _run_codex_provider(command: list[str], timeout: int) -> ProviderResponse:
    with tempfile.TemporaryDirectory(prefix="vidscribe-provider-") as temp_dir:
        output_path = Path(temp_dir) / "last-message.json"
        command_with_output = [
            *command[:-1],
            "--cd",
            temp_dir,
            "--output-last-message",
            str(output_path),
            command[-1],
        ]
        response = _run_provider_in_cwd(
            command_with_output,
            timeout=timeout,
            cwd=Path(temp_dir),
            parse_stdout=False,
        )
        output_text = output_path.read_text(encoding="utf-8").strip()
        if not output_text:
            raise ProviderError("codex did not write an output-last-message.")
        raw_json = _parse_provider_json(output_text)
        return ProviderResponse(
            text=_response_text(raw_json),
            raw_json=raw_json,
            cost_estimate=_cost_estimate(raw_json),
            duration_s=response.duration_s,
        )


def _run_provider_in_cwd(
    command: list[str],
    timeout: int,
    cwd: Path | None,
    *,
    parse_stdout: bool = True,
) -> ProviderResponse:
    started = time.monotonic()
    last_error: ProviderError | None = None
    for attempt in range(2):
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                cwd=cwd,
            )
        except FileNotFoundError as exc:
            binary = command[0]
            raise ProviderError(
                f"{binary} was not found. Install it and make sure it is on PATH."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            last_error = ProviderError(
                f"{command[0]} timed out after {timeout} seconds."
            )
            if attempt == 0:
                continue
            raise last_error from exc

        if result.returncode == 0:
            if not parse_stdout:
                return ProviderResponse(
                    text="",
                    raw_json={},
                    cost_estimate=None,
                    duration_s=time.monotonic() - started,
                )
            raw_json = _parse_provider_json(result.stdout)
            return ProviderResponse(
                text=_response_text(raw_json),
                raw_json=raw_json,
                cost_estimate=_cost_estimate(raw_json),
                duration_s=time.monotonic() - started,
            )

        details = (result.stderr or result.stdout or "").strip()
        message = f"{command[0]} exited with status {result.returncode}"
        if details:
            message = f"{message}: {details}"
        last_error = ProviderError(message)
        if attempt == 0 and _is_transient_error(details):
            continue
        raise last_error

    if last_error is not None:
        raise last_error
    raise ProviderError(f"{command[0]} failed without details.")


def _parse_provider_json(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        raise ProviderError("Provider returned empty stdout.")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = _parse_json_lines(text)

    if not isinstance(parsed, dict):
        raise ProviderError("Provider JSON output must be an object.")
    return parsed


def _parse_json_lines(text: str) -> Any:
    last_json: Any = None
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        try:
            last_json = json.loads(candidate)
        except json.JSONDecodeError:
            continue

    if last_json is not None:
        return last_json
    raise ProviderError("Provider stdout did not contain valid JSON.")


def _response_text(raw_json: dict[str, Any]) -> str:
    for key in ("corrected_text", "text", "result", "response"):
        value = raw_json.get(key)
        if isinstance(value, str):
            return value

    message = raw_json.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content

    raise ProviderError(
        "Provider JSON output must include corrected_text, text, result, or response."
    )


def _cost_estimate(raw_json: dict[str, Any]) -> float | None:
    value = raw_json.get("cost_estimate")
    if value is None:
        value = raw_json.get("cost_usd")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_transient_error(details: str) -> bool:
    lowered = details.lower()
    markers = (
        "temporarily",
        "timeout",
        "timed out",
        "try again",
        "rate limit",
        "rate-limit",
        "overloaded",
        "connection reset",
        "connection refused",
        "unavailable",
    )
    return any(marker in lowered for marker in markers)
