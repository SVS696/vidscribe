import subprocess
from pathlib import Path

import pytest

from vidscribe.audio import AudioExtractionError, extract


def test_extract_runs_ffmpeg_with_expected_audio_settings(tmp_path, mocker) -> None:
    run = mocker.patch("vidscribe.audio.subprocess.run")
    video = tmp_path / "input.mp4"
    output = tmp_path / "audio"

    result = extract(video, output)

    assert result == tmp_path / "audio.wav"
    run.assert_called_once_with(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-vn",
            str(tmp_path / "audio.wav"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )


def test_extract_keeps_explicit_wav_suffix(tmp_path, mocker) -> None:
    run = mocker.patch("vidscribe.audio.subprocess.run")
    output = tmp_path / "custom.wav"

    result = extract(tmp_path / "input.mp4", output)

    assert result == output
    assert run.call_args.args[0][-1] == str(output)


def test_extract_raises_helpful_error_when_ffmpeg_is_missing(tmp_path, mocker) -> None:
    mocker.patch("vidscribe.audio.subprocess.run", side_effect=FileNotFoundError)

    with pytest.raises(AudioExtractionError, match="ffmpeg was not found"):
        extract(tmp_path / "input.mp4", tmp_path / "audio.wav")


def test_extract_includes_ffmpeg_error_details(tmp_path, mocker) -> None:
    error = subprocess.CalledProcessError(
        returncode=1,
        cmd=["ffmpeg"],
        stderr="Invalid data found when processing input",
    )
    mocker.patch("vidscribe.audio.subprocess.run", side_effect=error)

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
