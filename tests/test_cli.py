from pathlib import Path

from typer.testing import CliRunner

from vidscribe.cache import Cache
from vidscribe.chunker import Chunk
from vidscribe.cli import app
from vidscribe.frames import FrameInfo
from vidscribe.pipeline import CorrectedChunk
from vidscribe.stt import SttResult, SttSegment


def stt_result() -> SttResult:
    return SttResult(
        segments=[
            SttSegment(start=0, end=1, text="hello", speaker="SPEAKER_00"),
        ],
        language="en",
        model="test",
    )


def chunk_item() -> Chunk:
    return Chunk(
        idx=0,
        start=0,
        end=1,
        segments=stt_result().segments,
        frame_paths=[Path("frame.jpg")],
    )


def corrected_item() -> CorrectedChunk:
    return CorrectedChunk(
        idx=0,
        start=0,
        end=1,
        speaker="SPEAKER_00",
        corrected_text="Hello.",
    )


def video_file(tmp_path: Path) -> Path:
    path = tmp_path / "video.mp4"
    path.write_bytes(b"video")
    return path


def test_pipeline_command_wires_full_run_with_command_overrides(
    tmp_path,
    mocker,
) -> None:
    runner = CliRunner()
    video = video_file(tmp_path)
    out = tmp_path / "out.md"
    fake_provider = object()
    fake_frames = [FrameInfo(ts=0, path=tmp_path / "frame.jpg", scene_change=False)]

    extract_mock = mocker.patch(
        "vidscribe.cli._extract",
        return_value=(tmp_path / "audio.wav", fake_frames),
    )
    transcribe_mock = mocker.patch(
        "vidscribe.cli._transcribe_audio",
        return_value=stt_result(),
    )
    chunks_mock = mocker.patch("vidscribe.cli._chunks", return_value=[chunk_item()])
    provider_mock = mocker.patch("vidscribe.cli.provider.make", return_value=fake_provider)
    speakers_mock = mocker.patch(
        "vidscribe.cli.speakers.identify",
        return_value={"SPEAKER_00": "Alice"},
    )
    correct_mock = mocker.patch(
        "vidscribe.cli.correct_chunks",
        return_value=[corrected_item()],
    )
    assemble_mock = mocker.patch("vidscribe.cli.assembler.assemble", return_value="final")

    result = runner.invoke(
        app,
        [
            "--cache-dir",
            str(tmp_path / ".vidscribe"),
            "pipeline",
            str(video),
            "--provider",
            "claude",
            "--model",
            "sonnet",
            "--whisper-model",
            "noscribe-fast",
            "--chunk-strategy",
            "time",
            "--speakers",
            "Alice",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8") == "final"
    provider_mock.assert_called_once_with("claude", model="sonnet")
    assert extract_mock.call_args.args[1].whisper_model == "noscribe-fast"
    assert chunks_mock.call_args.args[2].chunk_strategy == "time"
    speakers_mock.assert_called_once()
    assert speakers_mock.call_args.kwargs["manual"] == ("Alice",)
    correct_mock.assert_called_once()
    assemble_mock.assert_called_once_with([corrected_item()], {"SPEAKER_00": "Alice"})
    transcribe_mock.assert_called_once()


def test_extract_command_runs_audio_and_frames_only(tmp_path, mocker) -> None:
    runner = CliRunner()
    video = video_file(tmp_path)
    extract_mock = mocker.patch(
        "vidscribe.cli._extract",
        return_value=(
            tmp_path / "audio.wav",
            [FrameInfo(ts=0, path=tmp_path / "frame.jpg", scene_change=False)],
        ),
    )

    result = runner.invoke(
        app,
        ["--cache-dir", str(tmp_path / ".vidscribe"), "extract", str(video)],
    )

    assert result.exit_code == 0, result.output
    assert "audio:" in result.output
    assert "frames: 1" in result.output
    extract_mock.assert_called_once()


def test_transcribe_command_runs_stt_without_frames(tmp_path, mocker) -> None:
    runner = CliRunner()
    video = video_file(tmp_path)
    audio_mock = mocker.patch("vidscribe.cli._audio", return_value=tmp_path / "audio.wav")
    stt_mock = mocker.patch("vidscribe.cli._transcribe_audio", return_value=stt_result())

    result = runner.invoke(
        app,
        [
            "--cache-dir",
            str(tmp_path / ".vidscribe"),
            "transcribe",
            str(video),
            "--whisper-model",
            "large-v3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "segments: 1" in result.output
    assert stt_mock.call_args.args[1].whisper_model == "large-v3"
    audio_mock.assert_called_once()


def test_correct_command_uses_cached_stt_and_frames(tmp_path, mocker) -> None:
    runner = CliRunner()
    video = video_file(tmp_path)
    out = tmp_path / "corrected.md"
    fake_provider = object()
    fake_frames = [FrameInfo(ts=0, path=tmp_path / "frame.jpg", scene_change=False)]

    cached_stt = mocker.patch("vidscribe.cli._cached_model", return_value=stt_result())
    cached_frames = mocker.patch("vidscribe.cli._cached_frames", return_value=fake_frames)
    mocker.patch("vidscribe.cli._chunks", return_value=[chunk_item()])
    provider_mock = mocker.patch("vidscribe.cli.provider.make", return_value=fake_provider)
    mocker.patch(
        "vidscribe.cli.speakers.identify",
        return_value={"SPEAKER_00": "Alice"},
    )
    mocker.patch("vidscribe.cli.correct_chunks", return_value=[corrected_item()])
    mocker.patch("vidscribe.cli.assembler.assemble", return_value="corrected")

    result = runner.invoke(
        app,
        [
            "--cache-dir",
            str(tmp_path / ".vidscribe"),
            "correct",
            str(video),
            "--provider",
            "codex",
            "--model",
            "gpt-5.5",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8") == "corrected"
    provider_mock.assert_called_once_with("codex", model="gpt-5.5")
    cached_stt.assert_called_once()
    cached_frames.assert_called_once()


def test_cache_list_and_clear_for_video(tmp_path) -> None:
    runner = CliRunner()
    video = video_file(tmp_path)
    cache_root = tmp_path / ".vidscribe"
    cache = Cache(cache_root)
    video_key = cache.key_for("video", video=video)
    (cache_root / "cache" / video_key / "stt").mkdir(parents=True)
    (cache_root / "cache" / video_key / "frames").mkdir()

    list_result = runner.invoke(
        app,
        ["--cache-dir", str(cache_root), "cache", "list", str(video)],
    )

    assert list_result.exit_code == 0, list_result.output
    assert "frames" in list_result.output
    assert "stt" in list_result.output

    clear_result = runner.invoke(
        app,
        ["--cache-dir", str(cache_root), "cache", "clear", str(video)],
    )

    assert clear_result.exit_code == 0, clear_result.output
    assert not (cache_root / "cache" / video_key).exists()
