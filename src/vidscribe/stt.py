"""Speech-to-text and diarization helpers."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


NOSCRIBE_RESOURCES = Path("/Applications/noScribe.app/Contents/Resources")


class STTAssetError(RuntimeError):
    """Raised when a requested local STT asset is unavailable."""


class AssetPaths(BaseModel):
    """Local noScribe model assets that can be reused by the pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    resources_dir: Path
    whisper_precise_dir: Path
    whisper_fast_dir: Path
    pyannote_dir: Path
    config_yaml: Path
    segmentation_path: Path
    embedding_path: Path


class AsrWord(BaseModel):
    """A word-level ASR timestamp."""

    start: float
    end: float
    word: str
    probability: float | None = None


class AsrSegment(BaseModel):
    """A segment returned by faster-whisper."""

    start: float
    end: float
    text: str
    words: list[AsrWord] = Field(default_factory=list)


class AsrResult(BaseModel):
    """JSON-serializable ASR output."""

    segments: list[AsrSegment]
    words: list[AsrWord]
    language: str | None = None
    model: str

    def to_json(self, path: Path | str) -> Path:
        """Persist the ASR result as UTF-8 JSON."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output


def detect_assets(resources_dir: Path | str = NOSCRIBE_RESOURCES) -> AssetPaths | None:
    """Detect reusable noScribe model assets from the macOS app bundle."""

    resources = Path(resources_dir)
    assets = AssetPaths(
        resources_dir=resources,
        whisper_precise_dir=resources / "models" / "precise",
        whisper_fast_dir=resources / "models" / "fast",
        pyannote_dir=resources / "pyannote",
        config_yaml=resources / "pyannote" / "config.yaml",
        segmentation_path=resources
        / "pyannote"
        / "segmentation"
        / "pytorch_model.bin",
        embedding_path=resources / "pyannote" / "embedding" / "pytorch_model.bin",
    )

    required_paths = [
        assets.whisper_precise_dir,
        assets.whisper_fast_dir,
        assets.pyannote_dir,
        assets.config_yaml,
        assets.segmentation_path,
        assets.embedding_path,
    ]
    if all(path.exists() for path in required_paths):
        return assets
    return None


def transcribe(
    audio_path: Path | str,
    model: str = "noscribe-precise",
    device: str = "auto",
    language: str = "ru",
) -> AsrResult:
    """Transcribe audio with faster-whisper and return JSON-friendly output."""

    resolved_device = _resolve_device(device)
    model_path = _resolve_model_path(model)
    compute_type = _compute_type_for_device(resolved_device)
    whisper_model_class = _whisper_model_class()
    whisper_model = whisper_model_class(
        str(model_path),
        device=resolved_device,
        compute_type=compute_type,
    )

    segments_iter, info = whisper_model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        vad_filter=True,
    )

    segments: list[AsrSegment] = []
    words: list[AsrWord] = []
    for raw_segment in segments_iter:
        segment_words = [_word_from_raw(raw_word) for raw_word in raw_words(raw_segment)]
        words.extend(segment_words)
        segments.append(
            AsrSegment(
                start=float(raw_segment.start),
                end=float(raw_segment.end),
                text=str(raw_segment.text).strip(),
                words=segment_words,
            )
        )

    detected_language = getattr(info, "language", None)
    return AsrResult(
        segments=segments,
        words=words,
        language=detected_language,
        model=model,
    )


def raw_words(raw_segment: Any) -> list[Any]:
    """Return raw faster-whisper words as a list."""

    return list(getattr(raw_segment, "words", None) or [])


def _word_from_raw(raw_word: Any) -> AsrWord:
    return AsrWord(
        start=float(raw_word.start),
        end=float(raw_word.end),
        word=str(raw_word.word),
        probability=getattr(raw_word, "probability", None),
    )


def _resolve_model_path(model: str) -> str | Path:
    if model in {"noscribe-precise", "noscribe-fast"}:
        assets = detect_assets()
        if assets is None:
            raise STTAssetError(
                "noScribe model assets were not found at "
                f"{NOSCRIBE_RESOURCES}. Use a faster-whisper model name such as "
                "'large-v3' or install noScribe."
            )
        if model == "noscribe-precise":
            return assets.whisper_precise_dir
        return assets.whisper_fast_dir
    return model


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _compute_type_for_device(device: str) -> str:
    if platform.system() == "Darwin":
        return "int8"
    return "int8"


def _whisper_model_class() -> type[Any]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise STTAssetError(
            "faster-whisper is not installed. Install the project dependencies first."
        ) from exc
    return WhisperModel
