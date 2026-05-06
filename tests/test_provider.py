import subprocess
from pathlib import Path
from unittest.mock import ANY

import pytest

from vidscribe.provider import (
    ClaudeCLIProvider,
    CodexCLIProvider,
    OllamaProvider,
    ProviderError,
    make,
)


def completed(stdout: str, returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess(
        args=["provider"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def codex_completed(output_text: str, stdout: str = '{"event": "done"}'):
    def run(command, **kwargs):
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(output_text, encoding="utf-8")
        return completed(stdout)

    return run


def test_claude_provider_runs_expected_command(mocker) -> None:
    run = mocker.patch(
        "vidscribe.provider.subprocess.run",
        return_value=completed('{"corrected_text": "fixed", "cost_estimate": 0.01}'),
    )

    frame = Path("/tmp/vidscribe-frames/frame.jpg")

    response = ClaudeCLIProvider(model="sonnet").correct(
        "prompt",
        frame_paths=[frame],
        timeout=30,
    )

    assert response.text == "fixed"
    assert response.raw_json["corrected_text"] == "fixed"
    assert response.cost_estimate == 0.01
    run.assert_called_once_with(
        [
            "claude",
            "-p",
            "--input-format",
            "text",
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
            "--model",
            "sonnet",
            "--add-dir",
            str(frame.resolve().parent),
        ],
        capture_output=True,
        text=True,
        input="prompt",
        timeout=30,
        check=False,
        cwd=ANY,
    )
    assert run.call_args.kwargs["cwd"].name.startswith("vidscribe-provider-")


def test_codex_provider_reads_output_last_message(mocker) -> None:
    run = mocker.patch(
        "vidscribe.provider.subprocess.run",
        side_effect=codex_completed('{"corrected_text": "done", "cost_usd": "0.2"}'),
    )
    frame = Path("/tmp/frame.jpg")

    response = CodexCLIProvider(model="gpt-5.5").correct(
        "prompt",
        frame_paths=[frame],
        timeout=45,
    )

    assert response.text == "done"
    assert response.cost_estimate == 0.2
    command = run.call_args.args[0]
    assert command[:10] == [
        "codex",
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--model",
    ]
    assert command[10:17] == [
        "gpt-5.5",
        "--image",
        str(frame.resolve()),
        "--cd",
        ANY,
        "--output-last-message",
        ANY,
    ]
    assert command[17:] == ["-"]
    assert run.call_args.kwargs["input"] == "prompt"
    assert run.call_args.kwargs["cwd"].name.startswith("vidscribe-provider-")


def test_provider_accepts_non_correction_json_schema(mocker) -> None:
    mocker.patch(
        "vidscribe.provider.subprocess.run",
        side_effect=codex_completed('{"speakers": {"SPEAKER_00": "Alice"}}'),
    )

    response = CodexCLIProvider().correct("prompt", frame_paths=[], timeout=10)

    assert response.text == ""
    assert response.raw_json == {"speakers": {"SPEAKER_00": "Alice"}}


def test_claude_provider_deduplicates_frame_parent_dirs(mocker) -> None:
    run = mocker.patch(
        "vidscribe.provider.subprocess.run",
        return_value=completed('{"corrected_text": "fixed"}'),
    )
    first = Path("/tmp/vidscribe-frames/a.jpg")
    second = Path("/tmp/vidscribe-frames/b.jpg")

    ClaudeCLIProvider(model="").correct(
        "prompt",
        frame_paths=[first, second],
        timeout=30,
    )

    command = run.call_args.args[0]
    assert command.count(str(first.resolve().parent)) == 1


def test_ollama_provider_sends_prompt_on_stdin_without_frame_args(mocker) -> None:
    run = mocker.patch(
        "vidscribe.provider.subprocess.run",
        return_value=completed('{"response": "local"}'),
    )
    frames = [Path("/tmp/a.jpg"), Path("/tmp/b.jpg")]

    response = OllamaProvider(model="qwen2-vl:7b").correct(
        "prompt",
        frame_paths=frames,
        timeout=60,
    )

    assert response.text == "local"
    assert run.call_args.args[0] == [
        "ollama",
        "run",
        "qwen2-vl:7b",
    ]
    assert run.call_args.kwargs["input"] == "prompt"
    assert run.call_args.kwargs["cwd"] is None


def test_provider_retries_once_on_transient_nonzero_exit(mocker) -> None:
    calls = 0

    def run(command, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return completed("", returncode=1, stderr="temporarily unavailable")
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text('{"corrected_text": "retry-ok"}', encoding="utf-8")
        return completed('{"event": "done"}')

    run = mocker.patch(
        "vidscribe.provider.subprocess.run",
        side_effect=run,
    )

    response = CodexCLIProvider().correct("prompt", frame_paths=[], timeout=10)

    assert response.text == "retry-ok"
    assert run.call_count == 2


def test_provider_raises_on_non_transient_nonzero_exit(mocker) -> None:
    mocker.patch(
        "vidscribe.provider.subprocess.run",
        return_value=completed("", returncode=2, stderr="bad prompt"),
    )

    with pytest.raises(ProviderError, match="bad prompt"):
        CodexCLIProvider().correct("prompt", frame_paths=[], timeout=10)


def test_provider_retries_timeout_once_then_raises(mocker) -> None:
    mocker.patch(
        "vidscribe.provider.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["codex"], timeout=10),
    )

    with pytest.raises(ProviderError, match="timed out"):
        CodexCLIProvider().correct("prompt", frame_paths=[], timeout=10)


def test_provider_raises_for_invalid_json(mocker) -> None:
    mocker.patch(
        "vidscribe.provider.subprocess.run",
        side_effect=codex_completed("not json"),
    )

    with pytest.raises(ProviderError, match="valid JSON"):
        CodexCLIProvider().correct("prompt", frame_paths=[], timeout=10)


def test_codex_provider_raises_when_output_last_message_missing(mocker) -> None:
    mocker.patch(
        "vidscribe.provider.subprocess.run",
        return_value=completed('{"event": "done"}'),
    )

    with pytest.raises(ProviderError, match="output-last-message file"):
        CodexCLIProvider().correct("prompt", frame_paths=[], timeout=10)


def test_provider_raises_helpful_error_when_binary_missing(mocker) -> None:
    mocker.patch(
        "vidscribe.provider.subprocess.run",
        side_effect=FileNotFoundError,
    )

    with pytest.raises(ProviderError, match="codex was not found"):
        CodexCLIProvider().correct("prompt", frame_paths=[], timeout=10)


def test_provider_factory_creates_known_providers() -> None:
    assert isinstance(make("claude", model="sonnet"), ClaudeCLIProvider)
    assert isinstance(make("codex", model="gpt-5.5"), CodexCLIProvider)
    assert isinstance(make("ollama", model="qwen2-vl:7b"), OllamaProvider)


def test_provider_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported provider"):
        make("gemini")
