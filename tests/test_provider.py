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


def test_claude_provider_runs_expected_command(mocker) -> None:
    run = mocker.patch(
        "vidscribe.provider.subprocess.run",
        return_value=completed('{"corrected_text": "fixed", "cost_estimate": 0.01}'),
    )

    response = ClaudeCLIProvider(model="sonnet").correct(
        "prompt",
        frame_paths=[Path("frame.jpg")],
        timeout=30,
    )

    assert response.text == "fixed"
    assert response.raw_json["corrected_text"] == "fixed"
    assert response.cost_estimate == 0.01
    run.assert_called_once_with(
        [
            "claude",
            "-p",
            "prompt",
            "--output-format",
            "json",
            "--max-turns",
            "1",
            "--model",
            "sonnet",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        cwd=ANY,
    )
    assert run.call_args.kwargs["cwd"].name.startswith("vidscribe-provider-")


def test_codex_provider_accepts_json_lines_stdout(mocker) -> None:
    run = mocker.patch(
        "vidscribe.provider.subprocess.run",
        return_value=completed(
            '{"event": "started"}\n{"corrected_text": "done", "cost_usd": "0.2"}'
        ),
    )

    response = CodexCLIProvider(model="gpt-5.5").correct(
        "prompt",
        frame_paths=[],
        timeout=45,
    )

    assert response.text == "done"
    assert response.cost_estimate == 0.2
    assert run.call_args.args[0] == [
        "codex",
        "exec",
        "--json",
        "--model",
        "gpt-5.5",
        "prompt",
    ]
    assert run.call_args.kwargs["cwd"].name.startswith("vidscribe-provider-")


def test_ollama_provider_passes_frame_paths(mocker) -> None:
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
        "prompt",
        "/tmp/a.jpg",
        "/tmp/b.jpg",
    ]
    assert run.call_args.kwargs["cwd"] is None


def test_provider_retries_once_on_transient_nonzero_exit(mocker) -> None:
    run = mocker.patch(
        "vidscribe.provider.subprocess.run",
        side_effect=[
            completed("", returncode=1, stderr="temporarily unavailable"),
            completed('{"corrected_text": "retry-ok"}'),
        ],
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
        return_value=completed("not json"),
    )

    with pytest.raises(ProviderError, match="valid JSON"):
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
