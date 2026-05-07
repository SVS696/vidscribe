from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vidscribe.audio import AudioExtractionError, extract


def _make_mock_proc(returncode: int = 0, stdout_lines: list[str] | None = None, stderr: str = "") -> MagicMock:
    """Build a mock Popen process for audio tests."""
    proc = MagicMock()
    proc.returncode = returncode
    lines = stdout_lines or []
    proc.stdout.__iter__ = lambda self: iter(lines)
    proc.stdout.read = lambda: ""
    proc.stderr.read = lambda: stderr
    proc.wait = lambda: returncode
    return proc


def test_extract_runs_ffmpeg_with_expected_audio_settings(tmp_path, mocker) -> None:
    mock_proc = _make_mock_proc()
    popen = mocker.patch("vidscribe.audio.subprocess.Popen", return_value=mock_proc)
    video = tmp_path / "input.mp4"
    output = tmp_path / "audio"

    result = extract(video, output)

    assert result == tmp_path / "audio.wav"
    call_args = popen.call_args.args[0]
    assert call_args[:4] == ["ffmpeg", "-y", "-i", str(video)]
    assert "-progress" in call_args
    assert "pipe:1" in call_args
    assert "-nostats" in call_args
    assert call_args[-1] == str(tmp_path / "audio.wav")
    assert "-ac" in call_args
    assert "-ar" in call_args


def test_extract_keeps_explicit_wav_suffix(tmp_path, mocker) -> None:
    mock_proc = _make_mock_proc()
    popen = mocker.patch("vidscribe.audio.subprocess.Popen", return_value=mock_proc)
    output = tmp_path / "custom.wav"

    result = extract(tmp_path / "input.mp4", output)

    assert result == output
    call_args = popen.call_args.args[0]
    assert call_args[-1] == str(output)


def test_extract_raises_helpful_error_when_ffmpeg_is_missing(tmp_path, mocker) -> None:
    mocker.patch("vidscribe.audio.subprocess.Popen", side_effect=FileNotFoundError)

    with pytest.raises(AudioExtractionError, match="ffmpeg was not found"):
        extract(tmp_path / "input.mp4", tmp_path / "audio.wav")


def test_extract_includes_ffmpeg_error_details(tmp_path, mocker) -> None:
    mock_proc = _make_mock_proc(returncode=1, stderr="Invalid data found when processing input")
    mocker.patch("vidscribe.audio.subprocess.Popen", return_value=mock_proc)

    with pytest.raises(AudioExtractionError, match="Invalid data"):
        extract(tmp_path / "broken.mp4", tmp_path / "audio.wav")


def test_extract_tiny_fixture_video(fixtures_path: Path, tmp_path) -> None:
    video = fixtures_path / "short.mp4"
    if not video.exists():
        pytest.skip("short.mp4 fixture is not available")

    output = extract(video, tmp_path / "fixture-audio")

    assert output.exists()
    assert output.suffix == ".wav"
    assert output.stat().st_size > 0
