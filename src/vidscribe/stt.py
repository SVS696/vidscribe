"""Speech-to-text and diarization helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable

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
    *,
    on_progress: Callable[[float, float], None] | None = None,
    total_duration: float | None = None,
) -> AsrResult:
    """Transcribe audio with faster-whisper and return JSON-friendly output.

    Parameters
    ----------
    on_progress:
        Optional callback ``(current_s, total_s)`` called after each segment.
        ``total_s`` will be 0 if *total_duration* is not provided.
    total_duration:
        Audio duration in seconds.  Used to compute progress percentage when
        *on_progress* is supplied.
    """

    resolved_device = _resolve_device(device)
    model_path, effective_model = _resolve_model_path(model)
    compute_type = "int8"
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

    _total = total_duration or 0.0

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
        if on_progress is not None:
            on_progress(float(raw_segment.end), _total)

    detected_language = getattr(info, "language", None)
    return AsrResult(
        segments=segments,
        words=words,
        language=detected_language,
        model=effective_model,
    )


def diarize(
    audio_path: Path | str,
    assets: AssetPaths | None,
    hf_token: str | None = None,
    *,
    progress_hook: Any | None = None,
) -> DiarResult:
    """Run pyannote diarization and return speaker turns.

    Parameters
    ----------
    progress_hook:
        Optional pyannote ``ProgressHook`` instance.  When supplied it is
        passed directly to the pipeline call so pyannote drives its own
        progress reporting.
    """

    pipeline_class = _pyannote_pipeline_class()
    call_kwargs: dict[str, Any] = {}
    if progress_hook is not None:
        call_kwargs["hook"] = progress_hook

    waveform = _load_waveform(audio_path)

    if assets is not None:
        with tempfile.TemporaryDirectory(prefix="vidscribe-pyannote-") as temp_dir:
            config_path = _patched_pyannote_config(assets, Path(temp_dir))
            pipeline = pipeline_class.from_pretrained(str(config_path))
            annotation = pipeline(waveform, **call_kwargs)
    else:
        pipeline = pipeline_class.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        annotation = pipeline(waveform, **call_kwargs)

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
        if words:
            segments.extend(_speaker_runs(words))
        else:
            segments.append(
                SttSegment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    speaker=_speaker_for_interval(segment.start, segment.end, diar.turns),
                    words=[],
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


def _resolve_model_path(model: str) -> tuple[str | Path, str]:
    if model in {"noscribe-precise", "noscribe-fast"}:
        assets = detect_assets()
        if assets is None:
            if model == "noscribe-precise":
                return "large-v3", "large-v3"
            return "small", "small"
        if model == "noscribe-precise":
            return assets.whisper_precise_dir, model
        return assets.whisper_fast_dir, model
    return model, model


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
    except OSError as exc:
        raise STTAssetError(
            "pyannote.audio could not be loaded. Check that torch and torchaudio "
            "versions are compatible, then reinstall the project dependencies."
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



def _load_waveform(audio_path: Path | str) -> dict[str, Any]:
    """Load audio as a waveform dict accepted by pyannote pipelines.

    Using soundfile instead of torchaudio avoids the ``AudioDecoder`` import
    that torchaudio's FFmpeg backend requires (not available on all builds).
    Returns ``{"waveform": Tensor(channels, frames), "sample_rate": int}``.
    """

    import soundfile as sf
    import torch as _torch

    waveform_np, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    # soundfile returns (frames, channels); pyannote expects (channels, frames)
    waveform = _torch.from_numpy(waveform_np.T).contiguous()
    return {"waveform": waveform, "sample_rate": sample_rate}


def _turns_from_annotation(annotation: Any) -> list[DiarTurn]:
    # pyannote 4.x возвращает DiarizeOutput; 3.x — Annotation напрямую.
    if hasattr(annotation, "exclusive_speaker_diarization"):
        annotation = annotation.exclusive_speaker_diarization
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


def _speaker_runs(words: list[SttWord]) -> list[SttSegment]:
    runs: list[SttSegment] = []
    current: list[SttWord] = []
    current_speaker: str | None = None

    for word in words:
        if current and word.speaker != current_speaker:
            runs.append(_segment_from_words(current, current_speaker))
            current = []
        current.append(word)
        current_speaker = word.speaker

    if current:
        runs.append(_segment_from_words(current, current_speaker))

    return runs


def _segment_from_words(words: list[SttWord], speaker: str | None) -> SttSegment:
    text = " ".join(word.word.strip() for word in words if word.word.strip())
    return SttSegment(
        start=words[0].start,
        end=words[-1].end,
        text=text,
        speaker=speaker,
        words=words,
    )
