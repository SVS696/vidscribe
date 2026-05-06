from pathlib import Path

from typer.testing import CliRunner

from vidscribe.cache import Cache
from vidscribe.chunker import Chunk
from vidscribe.cli import _frames, _transcribe_audio, app
from vidscribe.frames import FrameInfo
from vidscribe.pipeline import CorrectedChunk
from vidscribe.provider import ProviderResponse
from vidscribe.config import AppConfig
from vidscribe.stt import (
    AsrResult,
    AsrSegment,
    AsrWord,
    DiarResult,
    DiarTurn,
    SttResult,
    SttSegment,
)


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
    assemble_mock.assert_called_once_with([corrected_item()], {"SPEAKER_00": "Alice"}, screen_context_mode="off")
    transcribe_mock.assert_called_once()


def test_pipeline_uses_provider_default_model_when_model_unset(tmp_path, mocker) -> None:
    runner = CliRunner()
    video = video_file(tmp_path)

    mocker.patch(
        "vidscribe.cli._extract",
        return_value=(tmp_path / "audio.wav", []),
    )
    mocker.patch("vidscribe.cli._transcribe_audio", return_value=stt_result())
    mocker.patch("vidscribe.cli._chunks", return_value=[chunk_item()])
    provider_mock = mocker.patch("vidscribe.cli.provider.make", return_value=object())
    mocker.patch(
        "vidscribe.cli.speakers.identify",
        return_value={"SPEAKER_00": "Alice"},
    )
    mocker.patch("vidscribe.cli.correct_chunks", return_value=[corrected_item()])
    mocker.patch("vidscribe.cli.assembler.assemble", return_value="final")

    result = runner.invoke(
        app,
        [
            "--cache-dir",
            str(tmp_path / ".vidscribe"),
            "pipeline",
            str(video),
            "--provider",
            "claude",
        ],
    )

    assert result.exit_code == 0, result.output
    provider_mock.assert_called_once_with("claude")


def test_pipeline_command_runs_fixture_e2e_with_mocked_stt_and_provider(
    tmp_path,
    fixtures_path,
    mocker,
) -> None:
    runner = CliRunner()
    video = fixtures_path / "short.mp4"
    out = tmp_path / "fixture.md"

    asr = AsrResult(
        model="large-v3",
        language="en",
        segments=[
            AsrSegment(
                start=0.0,
                end=0.6,
                text="hello bob",
                words=[
                    AsrWord(start=0.0, end=0.2, word="hello"),
                    AsrWord(start=0.2, end=0.6, word="bob"),
                ],
            ),
            AsrSegment(
                start=0.6,
                end=1.0,
                text="hi alice",
                words=[
                    AsrWord(start=0.6, end=0.8, word="hi"),
                    AsrWord(start=0.8, end=1.0, word="alice"),
                ],
            ),
        ],
        words=[],
    )
    diar = DiarResult(
        turns=[
            DiarTurn(start=0.0, end=0.6, speaker="SPEAKER_00"),
            DiarTurn(start=0.6, end=1.0, speaker="SPEAKER_01"),
        ]
    )

    class FakeProvider:
        model = "test-model"

        def __init__(self) -> None:
            self.calls: list[str] = []

        def correct(self, prompt: str, frame_paths: list[Path], timeout: int):
            self.calls.append(prompt)
            if prompt.startswith("You are identifying speaker names"):
                payload = {"speakers": {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}}
            else:
                payload = {
                    "corrected_text": f"Corrected chunk {len(self.calls) - 1}.",
                    "glossary_delta": {},
                    "notes": "",
                }
            return ProviderResponse(
                text="",
                raw_json=payload,
                cost_estimate=None,
                duration_s=0.01,
            )

    fake_provider = FakeProvider()
    mocker.patch("vidscribe.cli.stt.detect_assets", return_value=None)
    transcribe_mock = mocker.patch("vidscribe.cli.stt.transcribe", return_value=asr)
    diarize_mock = mocker.patch("vidscribe.cli.stt.diarize", return_value=diar)
    provider_mock = mocker.patch(
        "vidscribe.cli.provider.make",
        return_value=fake_provider,
    )

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
            "test-model",
            "--whisper-model",
            "large-v3",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.exists()
    transcript = out.read_text(encoding="utf-8")
    assert "Alice" in transcript
    assert "Bob" in transcript
    assert "Corrected chunk 1." in transcript
    assert "Corrected chunk 2." in transcript
    assert transcribe_mock.call_args.args[0].name == "audio.wav"
    diarize_mock.assert_called_once()
    provider_mock.assert_called_once_with("claude", model="test-model")
    assert len(fake_provider.calls) == 3
    cache_root = tmp_path / ".vidscribe" / "cache"
    assert list(cache_root.glob("*/audio/audio.wav"))
    assert list(cache_root.glob("*/frames/frames.json"))
    assert list(cache_root.glob("*/stt/artefact.json"))


def test_pipeline_rejects_unknown_provider_before_expensive_work(tmp_path, mocker) -> None:
    runner = CliRunner()
    video = video_file(tmp_path)
    extract_mock = mocker.patch("vidscribe.cli._extract")

    result = runner.invoke(
        app,
        [
            "--cache-dir",
            str(tmp_path / ".vidscribe"),
            "pipeline",
            str(video),
            "--provider",
            "gemini",
        ],
    )

    assert result.exit_code != 0
    assert "Invalid provider" in result.output
    extract_mock.assert_not_called()


def test_cli_rejects_unknown_no_cache_stage() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["--no-cache", "sttt"])

    assert result.exit_code != 0
    assert "Invalid no_cache.0" in result.output


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


def test_extract_no_cache_disables_audio_and_frames_cache(tmp_path, mocker) -> None:
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
        [
            "--cache-dir",
            str(tmp_path / ".vidscribe"),
            "extract",
            str(video),
            "--no-cache",
        ],
    )

    assert result.exit_code == 0, result.output
    cache = extract_mock.call_args.args[2]
    assert "audio" in cache.disabled_stages
    assert "frames" in cache.disabled_stages


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


def test_correct_no_cache_preserves_stt_and_frame_cache_reads(tmp_path, mocker) -> None:
    runner = CliRunner()
    video = video_file(tmp_path)
    out = tmp_path / "corrected.md"
    fake_frames = [FrameInfo(ts=0, path=tmp_path / "frame.jpg", scene_change=False)]

    def cached_model(cache, stage, key, model_class):
        assert stage not in cache.disabled_stages
        return stt_result()

    def cached_frames(cache, key):
        assert "frames" not in cache.disabled_stages
        return fake_frames

    mocker.patch("vidscribe.cli._cached_model", side_effect=cached_model)
    mocker.patch("vidscribe.cli._cached_frames", side_effect=cached_frames)
    mocker.patch("vidscribe.cli._chunks", return_value=[chunk_item()])
    mocker.patch("vidscribe.cli.provider.make", return_value=object())
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
            "--no-cache",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8") == "corrected"


def test_frames_cache_invalidates_when_frame_rate_changes(tmp_path, mocker) -> None:
    video = video_file(tmp_path)
    cache = Cache(tmp_path / ".vidscribe")
    video_key = cache.key_for("video", video=video)
    calls = []

    def extract(video_path, output_dir, sample_every, **kwargs):
        calls.append(sample_every)
        frame_path = output_dir / f"frame_{len(calls)}.jpg"
        item = FrameInfo(ts=0, path=frame_path, scene_change=False)
        (output_dir / "frames.json").write_text(
            "[" + item.model_dump_json() + "]",
            encoding="utf-8",
        )
        return [item]

    mocker.patch("vidscribe.cli.frames.extract", side_effect=extract)

    first = _frames(video, AppConfig(frame_rate=0.1), cache, video_key)
    second = _frames(video, AppConfig(frame_rate=0.1), cache, video_key)
    third = _frames(video, AppConfig(frame_rate=0.2), cache, video_key)

    assert first == second
    assert third != first
    assert calls == [10.0, 5.0]


def test_transcribe_cache_invalidates_when_stt_config_changes(tmp_path, mocker) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    cache = Cache(tmp_path / ".vidscribe")
    video_key = "video-key"
    calls = []

    def transcribe(audio, model, language, **kwargs):
        calls.append((model, language))
        return AsrResult(model=model, language=language, segments=[], words=[])

    mocker.patch("vidscribe.cli.stt.detect_assets", return_value=None)
    mocker.patch("vidscribe.cli.stt.transcribe", side_effect=transcribe)
    mocker.patch("vidscribe.cli.stt.diarize", return_value=DiarResult(turns=[]))

    first = _transcribe_audio(audio_path, AppConfig(whisper_model="large-v3"), cache, video_key)
    second = _transcribe_audio(audio_path, AppConfig(whisper_model="large-v3"), cache, video_key)
    third = _transcribe_audio(audio_path, AppConfig(whisper_model="medium"), cache, video_key)
    no_asr_cache = Cache(tmp_path / ".vidscribe", disabled_stages={"asr"})
    fourth = _transcribe_audio(
        audio_path,
        AppConfig(whisper_model="medium"),
        no_asr_cache,
        video_key,
    )

    assert first == second
    assert third.model == "medium"
    assert fourth.model == "medium"
    assert calls == [("large-v3", "ru"), ("medium", "ru"), ("medium", "ru")]
    assert cache.get("asr", video_key) is not None
    assert cache.get("diar", video_key) is not None


def test_frames_returns_absolute_paths_from_relative_cache_dir(tmp_path, mocker, monkeypatch) -> None:
    video = video_file(tmp_path)
    cache = Cache(Path(".vidscribe-test"))
    video_key = cache.key_for("video", video=video)
    monkeypatch.chdir(tmp_path)

    def extract(video_path, output_dir, sample_every, **kwargs):
        frame_path = output_dir / "frame.jpg"
        item = FrameInfo(ts=0, path=frame_path, scene_change=False)
        (output_dir / "frames.json").write_text(
            "[" + item.model_dump_json() + "]",
            encoding="utf-8",
        )
        return [item]

    mocker.patch("vidscribe.cli.frames.extract", side_effect=extract)

    frame_items = _frames(video, AppConfig(frame_rate=0.1), cache, video_key)

    assert frame_items[0].path.is_absolute()


def test_pipeline_mix_mode_flags_are_parsed_and_passed(tmp_path, mocker) -> None:
    """--correction-mode mix and related flags are parsed correctly."""
    runner = CliRunner()
    video = video_file(tmp_path)

    mocker.patch(
        "vidscribe.cli._extract",
        return_value=(tmp_path / "audio.wav", []),
    )
    mocker.patch("vidscribe.cli._transcribe_audio", return_value=stt_result())
    mocker.patch("vidscribe.cli._chunks", return_value=[chunk_item()])
    provider_mock = mocker.patch("vidscribe.cli.provider.make", return_value=object())
    mocker.patch(
        "vidscribe.cli.speakers.identify",
        return_value={"SPEAKER_00": "Alice"},
    )
    correct_mock = mocker.patch(
        "vidscribe.cli.correct_chunks",
        return_value=[corrected_item()],
    )
    mocker.patch("vidscribe.cli.assembler.assemble", return_value="final")

    result = runner.invoke(
        app,
        [
            "--cache-dir",
            str(tmp_path / ".vidscribe"),
            "pipeline",
            str(video),
            "--correction-mode",
            "mix",
            "--text-provider",
            "codex",
            "--text-model",
            "gpt-5.5",
            "--visual-provider",
            "claude",
            "--visual-model",
            "sonnet",
        ],
    )

    assert result.exit_code == 0, result.output
    # provider.make should be called at least for text and visual providers
    assert provider_mock.call_count >= 2
    # correct_chunks should be called with a visual_provider
    call_kwargs = correct_mock.call_args.kwargs
    assert call_kwargs.get("visual_provider") is not None


def test_correct_mix_mode_flags_are_parsed_and_passed(tmp_path, mocker) -> None:
    """--correction-mode mix works for the correct command."""
    runner = CliRunner()
    video = video_file(tmp_path)

    mocker.patch("vidscribe.cli._cached_model", return_value=stt_result())
    mocker.patch(
        "vidscribe.cli._cached_frames",
        return_value=[],
    )
    mocker.patch("vidscribe.cli._chunks", return_value=[chunk_item()])
    provider_mock = mocker.patch("vidscribe.cli.provider.make", return_value=object())
    mocker.patch(
        "vidscribe.cli.speakers.identify",
        return_value={"SPEAKER_00": "Alice"},
    )
    correct_mock = mocker.patch(
        "vidscribe.cli.correct_chunks",
        return_value=[corrected_item()],
    )
    mocker.patch("vidscribe.cli.assembler.assemble", return_value="corrected")

    result = runner.invoke(
        app,
        [
            "--cache-dir",
            str(tmp_path / ".vidscribe"),
            "correct",
            str(video),
            "--correction-mode",
            "mix",
            "--text-provider",
            "codex",
            "--visual-provider",
            "claude",
            "--visual-model",
            "sonnet",
        ],
    )

    assert result.exit_code == 0, result.output
    assert provider_mock.call_count >= 2
    call_kwargs = correct_mock.call_args.kwargs
    assert call_kwargs.get("visual_provider") is not None


def test_pipeline_single_mode_does_not_pass_visual_provider(tmp_path, mocker) -> None:
    """Default single mode must not supply visual_provider to correct_chunks."""
    runner = CliRunner()
    video = video_file(tmp_path)

    mocker.patch(
        "vidscribe.cli._extract",
        return_value=(tmp_path / "audio.wav", []),
    )
    mocker.patch("vidscribe.cli._transcribe_audio", return_value=stt_result())
    mocker.patch("vidscribe.cli._chunks", return_value=[chunk_item()])
    mocker.patch("vidscribe.cli.provider.make", return_value=object())
    mocker.patch(
        "vidscribe.cli.speakers.identify",
        return_value={"SPEAKER_00": "Alice"},
    )
    correct_mock = mocker.patch(
        "vidscribe.cli.correct_chunks",
        return_value=[corrected_item()],
    )
    mocker.patch("vidscribe.cli.assembler.assemble", return_value="final")

    result = runner.invoke(
        app,
        [
            "--cache-dir",
            str(tmp_path / ".vidscribe"),
            "pipeline",
            str(video),
        ],
    )

    assert result.exit_code == 0, result.output
    call_kwargs = correct_mock.call_args.kwargs
    assert call_kwargs.get("visual_provider") is None


def test_pipeline_screen_context_flag_is_passed_to_assembler(tmp_path, mocker) -> None:
    """--screen-context inline is parsed and forwarded to assembler.assemble."""
    runner = CliRunner()
    video = video_file(tmp_path)

    mocker.patch(
        "vidscribe.cli._extract",
        return_value=(tmp_path / "audio.wav", []),
    )
    mocker.patch("vidscribe.cli._transcribe_audio", return_value=stt_result())
    mocker.patch("vidscribe.cli._chunks", return_value=[chunk_item()])
    mocker.patch("vidscribe.cli.provider.make", return_value=object())
    mocker.patch(
        "vidscribe.cli.speakers.identify",
        return_value={"SPEAKER_00": "Alice"},
    )
    mocker.patch("vidscribe.cli.correct_chunks", return_value=[corrected_item()])
    assemble_mock = mocker.patch("vidscribe.cli.assembler.assemble", return_value="final")

    result = runner.invoke(
        app,
        [
            "--cache-dir",
            str(tmp_path / ".vidscribe"),
            "pipeline",
            str(video),
            "--screen-context",
            "inline",
        ],
    )

    assert result.exit_code == 0, result.output
    call_kwargs = assemble_mock.call_args.kwargs
    assert call_kwargs.get("screen_context_mode") == "inline"


def test_correct_screen_context_flag_is_passed_to_assembler(tmp_path, mocker) -> None:
    """--screen-context aside works for the correct command."""
    runner = CliRunner()
    video = video_file(tmp_path)

    mocker.patch("vidscribe.cli._cached_model", return_value=stt_result())
    mocker.patch("vidscribe.cli._cached_frames", return_value=[])
    mocker.patch("vidscribe.cli._chunks", return_value=[chunk_item()])
    mocker.patch("vidscribe.cli.provider.make", return_value=object())
    mocker.patch(
        "vidscribe.cli.speakers.identify",
        return_value={"SPEAKER_00": "Alice"},
    )
    mocker.patch("vidscribe.cli.correct_chunks", return_value=[corrected_item()])
    assemble_mock = mocker.patch("vidscribe.cli.assembler.assemble", return_value="corrected")

    result = runner.invoke(
        app,
        [
            "--cache-dir",
            str(tmp_path / ".vidscribe"),
            "correct",
            str(video),
            "--screen-context",
            "aside",
        ],
    )

    assert result.exit_code == 0, result.output
    call_kwargs = assemble_mock.call_args.kwargs
    assert call_kwargs.get("screen_context_mode") == "aside"


def test_cache_list_and_clear_for_video(tmp_path) -> None:
    runner = CliRunner()
    video = video_file(tmp_path)
    cache_root = tmp_path / ".vidscribe"
    cache = Cache(cache_root)
    video_key = cache.key_for("video", video=video)
    (cache_root / "cache" / video_key / "stt").mkdir(parents=True)
    (cache_root / "cache" / video_key / "frames").mkdir()
    derived = cache_root / "cache" / video_key / "derived-key" / "corrected"
    derived.mkdir(parents=True)
    (derived / "artefact.json").write_text('{"text": "secret"}', encoding="utf-8")

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
