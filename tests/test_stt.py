from __future__ import annotations

from types import SimpleNamespace

import pytest

from vidscribe import stt
from vidscribe.stt import AsrResult, AsrSegment, AsrWord, STTAssetError


def make_noscribe_bundle(root):
    resources = root / "Resources"
    (resources / "models" / "precise").mkdir(parents=True)
    (resources / "models" / "fast").mkdir(parents=True)
    (resources / "pyannote" / "segmentation").mkdir(parents=True)
    (resources / "pyannote" / "embedding").mkdir(parents=True)
    (resources / "pyannote" / "config.yaml").write_text("pipeline: local\n")
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


def test_transcribe_raises_helpful_error_when_noscribe_assets_missing(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(stt, "detect_assets", lambda: None)

    with pytest.raises(STTAssetError, match="noScribe model assets were not found"):
        stt.transcribe(tmp_path / "audio.wav", model="noscribe-precise")


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


def test_noscribe_transcribe_integration_skips_without_bundle(fixtures_path) -> None:
    if stt.detect_assets() is None:
        pytest.skip("noScribe bundle is not available")

    audio_path = fixtures_path / "short.wav"
    if not audio_path.exists():
        pytest.skip("short.wav fixture is not available")

    result = stt.transcribe(audio_path, model="noscribe-fast")

    assert result.segments
