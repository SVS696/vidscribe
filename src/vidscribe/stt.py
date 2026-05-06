"""Speech-to-text and diarization helpers."""

from __future__ import annotations

import json
import platform
import tempfile
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


class DiarTurn(BaseModel):
    """A speaker turn returned by pyannote."""

    start: float
    end: float
    speaker: str


class DiarResult(BaseModel):
    """JSON-serializable diarization output."""

    turns: list[DiarTurn]

    def to_json(self, path: Path | str) -> Path:
        """Persist the diarization result as UTF-8 JSON."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output


class SttWord(AsrWord):
    """A word-level ASR timestamp with the best matching speaker."""

    speaker: str | None = None


class SttSegment(BaseModel):
    """A diarized transcript segment."""

    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[SttWord] = Field(default_factory=list)


class SttResult(BaseModel):
    """JSON-serializable STT output after ASR and diarization are merged."""

    segments: list[SttSegment]
    language: str | None = None
    model: str

    def to_json(self, path: Path | str) -> Path:
        """Persist the merged STT result as UTF-8 JSON."""

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


def diarize(
    audio_path: Path | str,
    assets: AssetPaths | None,
    hf_token: str | None = None,
) -> DiarResult:
    """Run pyannote diarization and return speaker turns."""

    pipeline_class = _pyannote_pipeline_class()
    if assets is not None:
        with tempfile.TemporaryDirectory(prefix="vidscribe-pyannote-") as temp_dir:
            config_path = _patched_pyannote_config(assets, Path(temp_dir))
            pipeline = pipeline_class.from_pretrained(str(config_path))
            annotation = pipeline(str(audio_path))
    else:
        pipeline = pipeline_class.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        annotation = pipeline(str(audio_path))

    return DiarResult(turns=_turns_from_annotation(annotation))


def merge_asr_diar(asr: AsrResult, diar: DiarResult) -> SttResult:
    """Assign diarization speakers to ASR words by maximum time overlap."""

    segments: list[SttSegment] = []
    for segment in asr.segments:
        words = [
            SttWord(
                start=word.start,
                end=word.end,
                word=word.word,
                probability=word.probability,
                speaker=_speaker_for_interval(word.start, word.end, diar.turns),
            )
            for word in segment.words
        ]
        segments.append(
            SttSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                speaker=_mode_speaker(words),
                words=words,
            )
        )

    return SttResult(segments=segments, language=asr.language, model=asr.model)


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


def _pyannote_pipeline_class() -> type[Any]:
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise STTAssetError(
            "pyannote.audio is not installed. Install the project dependencies first."
        ) from exc
    return Pipeline


def _patched_pyannote_config(assets: AssetPaths, temp_dir: Path) -> Path:
    text = assets.config_yaml.read_text(encoding="utf-8")
    replacements = {
        "$model/segmentation": str(assets.segmentation_path),
        "$model/embedding": str(assets.embedding_path),
        "$model/plda": str(assets.pyannote_dir / "plda"),
    }
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)

    output = temp_dir / "config.yaml"
    output.write_text(text, encoding="utf-8")
    return output


def _turns_from_annotation(annotation: Any) -> list[DiarTurn]:
    turns: list[DiarTurn] = []
    for segment, _track, speaker in annotation.itertracks(yield_label=True):
        turns.append(
            DiarTurn(
                start=float(segment.start),
                end=float(segment.end),
                speaker=str(speaker),
            )
        )
    return turns


def _speaker_for_interval(
    start: float,
    end: float,
    turns: list[DiarTurn],
) -> str | None:
    best_speaker: str | None = None
    best_overlap = 0.0
    for turn in turns:
        overlap = min(end, turn.end) - max(start, turn.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn.speaker
    return best_speaker


def _mode_speaker(words: list[SttWord]) -> str | None:
    counts: dict[str, int] = {}
    for word in words:
        if word.speaker is None:
            continue
        counts[word.speaker] = counts.get(word.speaker, 0) + 1

    if not counts:
        return None
    return max(counts, key=counts.get)
