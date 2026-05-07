"""Tests for the edit module and the `vidscribe edit` CLI command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from typer.testing import CliRunner

from vidscribe.cli import app
from vidscribe.edit import apply_edits
from vidscribe.provider import ProviderResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_TRANSCRIPT = """\
## [00:00:01] **Алексей**

Привет, это тест транскрипта.

## [00:00:10] **s01**

Да, всё работает.
"""

EDITED_TRANSCRIPT = """\
## [00:00:01] **Алексей**

Привет, это тест транскрипта.

## [00:00:10] **Андрей**

Да, всё работает.
"""


def _make_provider_response(text: str) -> ProviderResponse:
    return ProviderResponse(
        text=text,
        raw_json={"text": text},
        cost_estimate=None,
        duration_s=0.5,
    )


def _mock_provider(return_text: str) -> MagicMock:
    p = MagicMock()
    p.correct.return_value = _make_provider_response(return_text)
    return p


# ---------------------------------------------------------------------------
# Unit tests for apply_edits
# ---------------------------------------------------------------------------


def test_apply_edits_passes_numbered_instructions_to_provider() -> None:
    mock_p = _mock_provider(EDITED_TRANSCRIPT)
    result = apply_edits(SAMPLE_TRANSCRIPT, ["replace s01 with Андрей"], mock_p)

    assert result == EDITED_TRANSCRIPT.strip()
    mock_p.correct.assert_called_once()
    call_prompt = mock_p.correct.call_args.args[0]
    assert "1. replace s01 with Андрей" in call_prompt
    assert SAMPLE_TRANSCRIPT in call_prompt


def test_apply_edits_multiple_instructions_form_numbered_list() -> None:
    mock_p = _mock_provider(EDITED_TRANSCRIPT)
    apply_edits(
        SAMPLE_TRANSCRIPT,
        ["replace s00 with Алексей", "replace s01 with Андрей"],
        mock_p,
    )
    call_prompt = mock_p.correct.call_args.args[0]
    assert "1. replace s00 with Алексей" in call_prompt
    assert "2. replace s01 with Андрей" in call_prompt


def test_apply_edits_strips_markdown_fences_from_response() -> None:
    fenced = f"```markdown\n{EDITED_TRANSCRIPT}```"
    mock_p = _mock_provider(fenced)
    result = apply_edits(SAMPLE_TRANSCRIPT, ["some edit"], mock_p)
    assert not result.startswith("```")
    assert "Андрей" in result


def test_apply_edits_returns_text_from_provider_response() -> None:
    mock_p = _mock_provider(EDITED_TRANSCRIPT)
    result = apply_edits(SAMPLE_TRANSCRIPT, ["edit"], mock_p)
    assert result == EDITED_TRANSCRIPT.strip()


def test_apply_edits_uses_no_frame_paths() -> None:
    mock_p = _mock_provider(EDITED_TRANSCRIPT)
    apply_edits(SAMPLE_TRANSCRIPT, ["edit"], mock_p)
    # frame_paths must be an empty list (text-only operation)
    # It is passed as the second positional argument
    call_args = mock_p.correct.call_args
    frame_paths = call_args.kwargs.get("frame_paths")
    if frame_paths is None and len(call_args.args) > 1:
        frame_paths = call_args.args[1]
    assert frame_paths == []


# ---------------------------------------------------------------------------
# CLI integration tests for `vidscribe edit`
# ---------------------------------------------------------------------------


def test_edit_command_writes_output_in_place_by_default(tmp_path: Path, mocker) -> None:
    transcript = tmp_path / "transcript.md"
    transcript.write_text(SAMPLE_TRANSCRIPT, encoding="utf-8")

    mocker.patch(
        "vidscribe.cli.provider.make",
        return_value=_mock_provider(EDITED_TRANSCRIPT),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["--provider", "claude", "edit", str(transcript), "-i", "replace s01 with Андрей"],
    )

    assert result.exit_code == 0, result.output
    assert transcript.read_text(encoding="utf-8") == EDITED_TRANSCRIPT.strip()


def test_edit_command_writes_to_out_path(tmp_path: Path, mocker) -> None:
    transcript = tmp_path / "transcript.md"
    transcript.write_text(SAMPLE_TRANSCRIPT, encoding="utf-8")
    out = tmp_path / "edited.md"

    mocker.patch(
        "vidscribe.cli.provider.make",
        return_value=_mock_provider(EDITED_TRANSCRIPT),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--provider",
            "claude",
            "edit",
            str(transcript),
            "-i",
            "replace s01 with Андрей",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.exists()
    assert out.read_text(encoding="utf-8") == EDITED_TRANSCRIPT.strip()
    # Original should be unchanged
    assert transcript.read_text(encoding="utf-8") == SAMPLE_TRANSCRIPT


def test_edit_command_requires_at_least_one_instruction(tmp_path: Path, mocker) -> None:
    transcript = tmp_path / "transcript.md"
    transcript.write_text(SAMPLE_TRANSCRIPT, encoding="utf-8")

    mocker.patch("vidscribe.cli.provider.make", return_value=_mock_provider(""))

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["--provider", "claude", "edit", str(transcript)],
    )

    assert result.exit_code != 0


def test_edit_command_multiple_instructions(tmp_path: Path, mocker) -> None:
    transcript = tmp_path / "transcript.md"
    transcript.write_text(SAMPLE_TRANSCRIPT, encoding="utf-8")

    mock_p = _mock_provider(EDITED_TRANSCRIPT)
    mocker.patch("vidscribe.cli.provider.make", return_value=mock_p)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--provider",
            "claude",
            "edit",
            str(transcript),
            "-i",
            "replace s00 with Алексей",
            "-i",
            "replace s01 with Андрей",
        ],
    )

    assert result.exit_code == 0, result.output
    # Both instructions must have been passed to apply_edits
    call_prompt = mock_p.correct.call_args.args[0]
    assert "replace s00 with Алексей" in call_prompt
    assert "replace s01 with Андрей" in call_prompt


def test_edit_command_missing_transcript_file(tmp_path: Path, mocker) -> None:
    mocker.patch("vidscribe.cli.provider.make", return_value=_mock_provider(""))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--provider",
            "claude",
            "edit",
            str(tmp_path / "nonexistent.md"),
            "-i",
            "do something",
        ],
    )

    assert result.exit_code != 0
