from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vidscribe import stt
from vidscribe.stt import (
    AsrResult,
    AsrSegment,
    AsrWord,
    DiarResult,
    DiarTurn,
    SttSegment,
    SttWord,
)


def make_noscribe_bundle(root):
    resources = root / "Resources"
    (resources / "models" / "precise").mkdir(parents=True)
    (resources / "models" / "fast").mkdir(parents=True)
    (resources / "pyannote" / "segmentation").mkdir(parents=True)
    (resources / "pyannote" / "embedding").mkdir(parents=True)
    (resources / "pyannote" / "config.yaml").write_text(
        "\n".join(
            [
                "pipeline:",
                "  name: pyannote.audio.pipelines.SpeakerDiarization",
                "  params:",
                "    segmentation: $model/segmentation",
                "    embedding: $model/embedding",
                "    plda: $model/plda",
            ]
        ),
        encoding="utf-8",
    )
    (resources / "pyannote" / "segmentation" / "pytorch_model.bin").write_bytes(b"seg")
    (resources / "pyannote" / "embedding" / "pytorch_model.bin").write_bytes(b"emb")
    return resources


def test_detect_assets_returns_expected_paths_when_bundle_exists(tmp_path) -> None:
    resources = make_noscribe_bundle(tmp_path)

    assets = stt.detect_assets(resources)

    assert assets is not None
    assert assets.resources_dir == resources
    assert assets.whisper_precise_dir == resources / "models" / "precise"
    assert assets.whisper_fast_dir == resources / "models" / "fast"
    assert assets.pyannote_dir == resources / "pyannote"
    assert assets.config_yaml == resources / "pyannote" / "config.yaml"
    assert assets.segmentation_path.name == "pytorch_model.bin"
    assert assets.embedding_path.name == "pytorch_model.bin"


def test_detect_assets_returns_none_when_required_files_are_missing(tmp_path) -> None:
    (tmp_path / "models" / "precise").mkdir(parents=True)

    assert stt.detect_assets(tmp_path) is None


def test_transcribe_uses_noscribe_precise_assets_and_word_timestamps(
    tmp_path, monkeypatch
) -> None:
    resources = make_noscribe_bundle(tmp_path)
    assets = stt.detect_assets(resources)
    calls = {}

    class FakeWhisperModel:
        def __init__(self, model_path, *, device, compute_type):
            calls["init"] = {
                "model_path": model_path,
                "device": device,
                "compute_type": compute_type,
            }

        def transcribe(self, audio_path, **kwargs):
            calls["transcribe"] = {"audio_path": audio_path, **kwargs}
            segment = SimpleNamespace(
                start=1.0,
                end=2.5,
                text=" hello world ",
                words=[
                    SimpleNamespace(
                        start=1.0,
                        end=1.4,
                        word="hello",
                        probability=0.95,
                    ),
                    SimpleNamespace(
                        start=1.5,
                        end=2.4,
                        word="world",
                        probability=0.9,
                    ),
                ],
            )
            return [segment], SimpleNamespace(language="ru")

    monkeypatch.setattr(stt, "detect_assets", lambda: assets)
    monkeypatch.setattr(stt, "_resolve_device", lambda device: "cpu")
    monkeypatch.setattr(stt, "_whisper_model_class", lambda: FakeWhisperModel)

    result = stt.transcribe(tmp_path / "audio.wav")

    assert calls["init"] == {
        "model_path": str(resources / "models" / "precise"),
        "device": "cpu",
        "compute_type": "int8",
    }
    assert calls["transcribe"] == {
        "audio_path": str(tmp_path / "audio.wav"),
        "language": "ru",
        "word_timestamps": True,
        "vad_filter": True,
    }
    assert result.language == "ru"
    assert result.model == "noscribe-precise"
    assert result.segments == [
        AsrSegment(
            start=1.0,
            end=2.5,
            text="hello world",
            words=[
                AsrWord(start=1.0, end=1.4, word="hello", probability=0.95),
                AsrWord(start=1.5, end=2.4, word="world", probability=0.9),
            ],
        )
    ]
    assert [word.word for word in result.words] == ["hello", "world"]


def test_transcribe_uses_noscribe_fast_assets(tmp_path, monkeypatch) -> None:
    resources = make_noscribe_bundle(tmp_path)
    assets = stt.detect_assets(resources)
    calls = {}

    class FakeWhisperModel:
        def __init__(self, model_path, *, device, compute_type):
            calls["model_path"] = model_path

        def transcribe(self, audio_path, **kwargs):
            return [], SimpleNamespace(language="ru")

    monkeypatch.setattr(stt, "detect_assets", lambda: assets)
    monkeypatch.setattr(stt, "_resolve_device", lambda device: "cpu")
    monkeypatch.setattr(stt, "_whisper_model_class", lambda: FakeWhisperModel)

    stt.transcribe(tmp_path / "audio.wav", model="noscribe-fast")

    assert calls["model_path"] == str(resources / "models" / "fast")


def test_transcribe_uses_regular_faster_whisper_model_name(tmp_path, monkeypatch) -> None:
    calls = {}

    class FakeWhisperModel:
        def __init__(self, model_path, *, device, compute_type):
            calls["model_path"] = model_path

        def transcribe(self, audio_path, **kwargs):
            return [], SimpleNamespace(language="en")

    monkeypatch.setattr(stt, "detect_assets", lambda: None)
    monkeypatch.setattr(stt, "_resolve_device", lambda device: "cpu")
    monkeypatch.setattr(stt, "_whisper_model_class", lambda: FakeWhisperModel)

    result = stt.transcribe(tmp_path / "audio.wav", model="large-v3", language="en")

    assert calls["model_path"] == "large-v3"
    assert result.language == "en"


def test_transcribe_falls_back_to_large_v3_when_noscribe_precise_assets_missing(
    tmp_path, monkeypatch
) -> None:
    calls = {}

    class FakeWhisperModel:
        def __init__(self, model_path, *, device, compute_type):
            calls["model_path"] = model_path

        def transcribe(self, audio_path, **kwargs):
            return [], SimpleNamespace(language="ru")

    monkeypatch.setattr(stt, "detect_assets", lambda: None)
    monkeypatch.setattr(stt, "_resolve_device", lambda device: "cpu")
    monkeypatch.setattr(stt, "_whisper_model_class", lambda: FakeWhisperModel)

    result = stt.transcribe(tmp_path / "audio.wav", model="noscribe-precise")

    assert calls["model_path"] == "large-v3"
    assert result.model == "large-v3"


def test_resolve_device_prefers_cuda_when_available(monkeypatch) -> None:
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    assert stt._resolve_device("auto") == "cuda"


def test_resolve_device_ignores_mps_and_falls_back_to_cpu(monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    assert stt._resolve_device("auto") == "cpu"


def test_asr_result_can_be_written_to_json(tmp_path) -> None:
    result = AsrResult(
        model="large-v3",
        language="ru",
        segments=[
            AsrSegment(
                start=0,
                end=1,
                text="привет",
                words=[AsrWord(start=0, end=1, word="привет")],
            )
        ],
        words=[AsrWord(start=0, end=1, word="привет")],
    )

    output = result.to_json(tmp_path / "asr_segments.json")

    assert output.read_text(encoding="utf-8").startswith("{\n")
    assert "привет" in output.read_text(encoding="utf-8")


def test_diarize_prefers_local_noscribe_assets(tmp_path, monkeypatch) -> None:
    resources = make_noscribe_bundle(tmp_path)
    assets = stt.detect_assets(resources)
    assert assets is not None
    calls = {}

    class FakeSegment:
        start = 0.5
        end = 1.5

    class FakeAnnotation:
        def itertracks(self, *, yield_label):
            assert yield_label is True
            yield FakeSegment(), "track", "SPEAKER_00"

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, config_path, **kwargs):
            config_text = Path(config_path).read_text(encoding="utf-8")
            calls["config_path"] = config_path
            calls["kwargs"] = kwargs
            calls["config_text"] = config_text
            return cls()

        def __call__(self, audio_path):
            calls["audio_path"] = audio_path
            return FakeAnnotation()

    monkeypatch.setattr(stt, "_pyannote_pipeline_class", lambda: FakePipeline)

    result = stt.diarize(tmp_path / "audio.wav", assets)

    assert "$model" not in calls["config_text"]
    assert str(assets.segmentation_path) in calls["config_text"]
    assert str(assets.embedding_path) in calls["config_text"]
    assert calls["kwargs"] == {}
    assert calls["audio_path"] == str(tmp_path / "audio.wav")
    assert result == DiarResult(
        turns=[DiarTurn(start=0.5, end=1.5, speaker="SPEAKER_00")]
    )


def test_diarize_falls_back_to_hugging_face_when_assets_missing(
    tmp_path, monkeypatch
) -> None:
    calls = {}

    class FakeAnnotation:
        def itertracks(self, *, yield_label):
            return iter(())

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls["model_name"] = model_name
            calls["kwargs"] = kwargs
            return cls()

        def __call__(self, audio_path):
            calls["audio_path"] = audio_path
            return FakeAnnotation()

    monkeypatch.setattr(stt, "_pyannote_pipeline_class", lambda: FakePipeline)

    result = stt.diarize(tmp_path / "audio.wav", assets=None, hf_token="hf_test")

    assert calls == {
        "model_name": "pyannote/speaker-diarization-3.1",
        "kwargs": {"use_auth_token": "hf_test"},
        "audio_path": str(tmp_path / "audio.wav"),
    }
    assert result == DiarResult(turns=[])


def test_merge_asr_diar_splits_asr_segments_on_speaker_changes() -> None:
    asr = AsrResult(
        model="large-v3",
        language="ru",
        segments=[
            AsrSegment(
                start=0.0,
                end=4.0,
                text="one two three four",
                words=[
                    AsrWord(start=0.0, end=1.0, word="one"),
                    AsrWord(start=1.0, end=2.0, word="two"),
                    AsrWord(start=2.0, end=3.0, word="three"),
                    AsrWord(start=3.1, end=3.5, word="four"),
                ],
            )
        ],
        words=[],
    )
    diar = DiarResult(
        turns=[
            DiarTurn(start=0.0, end=1.2, speaker="SPEAKER_00"),
            DiarTurn(start=1.2, end=2.1, speaker="SPEAKER_01"),
            DiarTurn(start=2.0, end=2.8, speaker="SPEAKER_01"),
        ]
    )

    result = stt.merge_asr_diar(asr, diar)

    assert result.language == "ru"
    assert result.model == "large-v3"
    assert result.segments == [
        SttSegment(
            start=0.0,
            end=1.0,
            text="one",
            speaker="SPEAKER_00",
            words=[SttWord(start=0.0, end=1.0, word="one", speaker="SPEAKER_00")],
        ),
        SttSegment(
            start=1.0,
            end=3.0,
            text="two three",
            speaker="SPEAKER_01",
            words=[
                SttWord(start=1.0, end=2.0, word="two", speaker="SPEAKER_01"),
                SttWord(start=2.0, end=3.0, word="three", speaker="SPEAKER_01"),
            ],
        ),
        SttSegment(
            start=3.1,
            end=3.5,
            text="four",
            speaker=None,
            words=[SttWord(start=3.1, end=3.5, word="four", speaker=None)],
        ),
    ]


def test_stt_result_can_be_written_to_json(tmp_path) -> None:
    asr = AsrResult(
        model="large-v3",
        language="ru",
        segments=[
            AsrSegment(
                start=0,
                end=1,
                text="привет",
                words=[AsrWord(start=0, end=1, word="привет")],
            )
        ],
        words=[],
    )
    diar = DiarResult(turns=[DiarTurn(start=0, end=1, speaker="SPEAKER_00")])

    output = stt.merge_asr_diar(asr, diar).to_json(tmp_path / "segments.json")

    text = output.read_text(encoding="utf-8")
    assert "SPEAKER_00" in text
    assert "привет" in text


def test_noscribe_transcribe_integration_skips_without_bundle(fixtures_path) -> None:
    if stt.detect_assets() is None:
        pytest.skip("noScribe bundle is not available")

    audio_path = fixtures_path / "short.wav"
    if not audio_path.exists():
        pytest.skip("short.wav fixture is not available")

    result = stt.transcribe(audio_path, model="noscribe-fast")

    assert result.segments


def test_noscribe_diarize_integration_skips_without_bundle(fixtures_path) -> None:
    assets = stt.detect_assets()
    if assets is None:
        pytest.skip("noScribe bundle is not available")

    audio_path = fixtures_path / "short.wav"
    if not audio_path.exists():
        pytest.skip("short.wav fixture is not available")

    result = stt.diarize(audio_path, assets)

    assert isinstance(result.turns, list)


def test_whisper_model_class_raises_helpful_error_when_dependency_missing(mocker) -> None:
    mocker.patch.dict(sys.modules, {"faster_whisper": None})

    with pytest.raises(stt.STTAssetError, match="faster-whisper is not installed"):
        stt._whisper_model_class()


def test_pyannote_pipeline_class_raises_helpful_error_when_dependency_missing(mocker) -> None:
    mocker.patch.dict(sys.modules, {"pyannote.audio": None})

    with pytest.raises(stt.STTAssetError, match="pyannote.audio is not installed"):
        stt._pyannote_pipeline_class()


def test_pyannote_pipeline_class_wraps_binary_load_errors(mocker) -> None:
    original_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "pyannote.audio":
            raise OSError("Could not load torchaudio")
        return original_import(name, *args, **kwargs)

    mocker.patch("builtins.__import__", side_effect=failing_import)

    with pytest.raises(stt.STTAssetError, match="torch and torchaudio"):
        stt._pyannote_pipeline_class()
