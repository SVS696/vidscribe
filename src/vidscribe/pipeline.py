"""Pipeline orchestration."""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.progress import Progress

from vidscribe.cache import Cache
from vidscribe.chunker import Chunk
from vidscribe.prompts import render
from vidscribe.provider import Provider, ProviderResponse

if TYPE_CHECKING:
    from vidscribe.progress import PipelineProgress


class CorrectionError(RuntimeError):
    """Raised when a provider response cannot be used as a corrected chunk."""


class CorrectedSegment(BaseModel):
    """A corrected transcript segment preserving speaker attribution."""

    start: float
    end: float
    speaker: str | None = None
    corrected_text: str


class CorrectedChunk(BaseModel):
    """A corrected transcript chunk with provider metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    idx: int
    start: float
    end: float
    speaker: str | None = None
    corrected_text: str
    segments: list[CorrectedSegment] = Field(default_factory=list)
    glossary_delta: dict[str, str] = Field(default_factory=dict)
    notes: str = ""
    raw_json: dict[str, Any] = Field(default_factory=dict)
    cost_estimate: float | None = None
    duration_s: float = 0
    frame_paths: list[Path] = Field(default_factory=list)


def correct_chunks(
    chunks: list[Chunk],
    provider: Provider,
    speakers: Mapping[str, str],
    cache: Cache | None,
    *,
    namespace_key: str | None = None,
    timeout: int = 300,
    console: Console | None = None,
    visual_provider: Provider | None = None,
    pipeline_progress: "PipelineProgress | None" = None,
) -> list[CorrectedChunk]:
    """Correct chunks sequentially while accumulating glossary deltas.

    When ``visual_provider`` is supplied the correction runs in two passes:
    Pass 1 uses ``provider`` for text-only speech corrections; Pass 2 uses
    ``visual_provider`` to apply visual context from frames.
    """

    glossary: dict[str, str] = {}
    corrected: list[CorrectedChunk] = []

    # Build a description that includes provider info
    provider_name = provider.__class__.__name__
    provider_model = getattr(provider, "model", None) or ""
    base_desc = f"[8/9] Correcting chunks ({provider_name}"
    if provider_model:
        base_desc = f"{base_desc}/{provider_model}"
    base_desc = f"{base_desc})"

    if pipeline_progress is not None:
        ctx = pipeline_progress.stage("correct", total=len(chunks), description=base_desc)
    else:
        # Fallback: use the legacy inline Progress (kept for backwards compat)
        output_console = console or Console()
        ctx = _legacy_progress_stage(output_console, len(chunks))

    with ctx as handle:
        for chunk_idx, chunk in enumerate(chunks):
            # For mix mode, update description to show current pass info
            if visual_provider is not None and pipeline_progress is not None:
                handle.update_description(
                    f"{base_desc} — chunk {chunk_idx + 1}/{len(chunks)}"
                )

            glossary_snapshot = dict(glossary)
            cache_key = _cache_key(
                cache,
                chunk,
                provider,
                speakers,
                glossary_snapshot,
                namespace_key=namespace_key,
                visual_provider=visual_provider,
            )
            cached = cache.get("corrected", cache_key) if cache and cache_key else None
            if cached is not None:
                corrected_chunk = CorrectedChunk.model_validate(cached)
            elif visual_provider is not None:
                corrected_chunk = _correct_chunk_mix(
                    chunk=chunk,
                    text_provider=provider,
                    visual_provider=visual_provider,
                    speakers=speakers,
                    glossary=glossary_snapshot,
                    timeout=timeout,
                )
                if cache is not None and cache_key is not None:
                    cache.set("corrected", cache_key, corrected_chunk)
            else:
                corrected_chunk = _correct_chunk(
                    chunk=chunk,
                    provider=provider,
                    speakers=speakers,
                    glossary=glossary_snapshot,
                    timeout=timeout,
                )
                if cache is not None and cache_key is not None:
                    cache.set("corrected", cache_key, corrected_chunk)

            glossary.update(corrected_chunk.glossary_delta)
            corrected.append(corrected_chunk)
            handle.advance()

    return corrected


def _correct_chunk_mix(
    *,
    chunk: Chunk,
    text_provider: Provider,
    visual_provider: Provider,
    speakers: Mapping[str, str],
    glossary: Mapping[str, str],
    timeout: int,
) -> CorrectedChunk:
    """Two-pass mix-mode correction: text-only pass then visual-context pass."""

    # Pass 1: text correction (no frames)
    frame_paths = [path.resolve() for path in chunk.frame_paths]
    text_prompt = render(
        "correct_chunk_text",
        transcript=_chunk_transcript(chunk, speakers),
        glossary=dict(glossary),
        speaker_map=dict(speakers),
    )
    text_response = text_provider.correct(text_prompt, frame_paths=[], timeout=timeout)
    text_payload = _correction_payload(text_response)
    text_segments = _corrected_segments(text_payload, chunk)
    text_corrected = _joined_corrected_text(text_segments) or _string_value(
        text_payload, "corrected_text", required=True
    )

    # Pass 2: visual correction (receives frames + both transcripts)
    visual_prompt = render(
        "correct_chunk_visual",
        asr_transcript=_chunk_transcript(chunk, speakers),
        text_corrected_transcript=text_corrected,
        frame_paths=[str(path) for path in frame_paths],
        glossary=dict(glossary),
        speaker_map=dict(speakers),
    )
    visual_response = visual_provider.correct(
        visual_prompt, frame_paths=frame_paths, timeout=timeout
    )
    visual_payload = _correction_payload(visual_response)
    visual_segments = _corrected_segments(visual_payload, chunk)
    final_text = _joined_corrected_text(visual_segments) or _string_value(
        visual_payload, "corrected_text", required=True
    )

    # Merge glossary deltas from both passes
    merged_glossary = _string_dict(text_payload.get("glossary_delta", {}))
    merged_glossary.update(_string_dict(visual_payload.get("glossary_delta", {})))

    total_cost: float | None = None
    if text_response.cost_estimate is not None or visual_response.cost_estimate is not None:
        total_cost = (text_response.cost_estimate or 0.0) + (
            visual_response.cost_estimate or 0.0
        )

    return CorrectedChunk(
        idx=chunk.idx,
        start=chunk.start,
        end=chunk.end,
        speaker=_chunk_speaker(chunk),
        corrected_text=final_text,
        segments=visual_segments,
        glossary_delta=merged_glossary,
        notes=_string_value(visual_payload, "notes", required=False),
        raw_json=visual_response.raw_json,
        cost_estimate=total_cost,
        duration_s=text_response.duration_s + visual_response.duration_s,
        frame_paths=frame_paths,
    )


def _correct_chunk(
    *,
    chunk: Chunk,
    provider: Provider,
    speakers: Mapping[str, str],
    glossary: Mapping[str, str],
    timeout: int,
) -> CorrectedChunk:
    frame_paths = [path.resolve() for path in chunk.frame_paths]
    prompt = render(
        "correct_chunk",
        transcript=_chunk_transcript(chunk, speakers),
        frame_paths=[str(path) for path in frame_paths],
        glossary=dict(glossary),
        speaker_map=dict(speakers),
    )
    response = provider.correct(prompt, frame_paths=frame_paths, timeout=timeout)
    payload = _correction_payload(response)
    corrected_segments = _corrected_segments(payload, chunk)
    corrected_text = _joined_corrected_text(corrected_segments) or _string_value(
        payload, "corrected_text", required=True
    )
    return CorrectedChunk(
        idx=chunk.idx,
        start=chunk.start,
        end=chunk.end,
        speaker=_chunk_speaker(chunk),
        corrected_text=corrected_text,
        segments=corrected_segments,
        glossary_delta=_string_dict(payload.get("glossary_delta", {})),
        notes=_string_value(payload, "notes", required=False),
        raw_json=response.raw_json,
        cost_estimate=response.cost_estimate,
        duration_s=response.duration_s,
        frame_paths=frame_paths,
    )


def _cache_key(
    cache: Cache | None,
    chunk: Chunk,
    provider: Provider,
    speakers: Mapping[str, str],
    glossary_snapshot: Mapping[str, str],
    *,
    namespace_key: str | None = None,
    visual_provider: Provider | None = None,
) -> str | None:
    if cache is None:
        return None
    key = cache.key_for(
        "corrected",
        chunk=chunk,
        provider=provider.__class__.__name__,
        model=getattr(provider, "model", None),
        visual_provider=visual_provider.__class__.__name__ if visual_provider is not None else None,
        visual_model=getattr(visual_provider, "model", None) if visual_provider is not None else None,
        correction_mode="mix" if visual_provider is not None else "single",
        speakers=dict(speakers),
        glossary_snapshot=dict(glossary_snapshot),
    )
    if namespace_key is None:
        return key
    return f"{namespace_key}/{key}"


def _chunk_transcript(chunk: Chunk, speakers: Mapping[str, str]) -> str:
    lines: list[str] = []
    if chunk.surrounding_context:
        lines.append(f"Context:\n{chunk.surrounding_context.strip()}")
    for segment in chunk.segments:
        speaker_id = segment.speaker or "UNKNOWN"
        speaker_name = speakers.get(speaker_id, speaker_id)
        lines.append(
            f"[{segment.start:.2f}-{segment.end:.2f}] "
            f"{speaker_id} ({speaker_name}): {segment.text.strip()}"
        )
    return "\n".join(lines)


def _chunk_speaker(chunk: Chunk) -> str | None:
    speakers = [
        segment.speaker
        for segment in chunk.segments
        if segment.speaker and segment.speaker.strip()
    ]
    if not speakers:
        return None
    return max(set(speakers), key=speakers.count)


def _correction_payload(response: ProviderResponse) -> dict[str, Any]:
    if isinstance(response.raw_json.get("corrected_text"), str):
        return response.raw_json
    if isinstance(response.raw_json.get("segments"), list):
        return response.raw_json

    text = response.text.strip()
    if text:
        # 1) pure JSON
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # 2) JSON inside markdown code block ```json ... ``` or ``` ... ```
        fence_match = re.search(
            r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE
        )
        if fence_match:
            try:
                parsed = json.loads(fence_match.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        # 3) first balanced { ... } block in the text
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                parsed = json.loads(brace_match.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

    preview = (response.text or "")[:500]
    raise CorrectionError(
        "Provider response is not valid correction JSON. "
        f"raw_json keys={sorted(response.raw_json.keys())} "
        f"text preview={preview!r}"
    )


def _corrected_segments(
    payload: Mapping[str, Any],
    chunk: Chunk,
) -> list[CorrectedSegment]:
    raw_segments = payload.get("segments")
    if isinstance(raw_segments, list):
        segments: list[CorrectedSegment] = []
        for idx, raw_segment in enumerate(raw_segments):
            if not isinstance(raw_segment, Mapping):
                raise CorrectionError("Provider response segments must be objects.")
            source = chunk.segments[min(idx, len(chunk.segments) - 1)]
            text = _string_value(raw_segment, "corrected_text", required=True)
            segments.append(
                CorrectedSegment(
                    start=_float_value(raw_segment.get("start"), source.start),
                    end=_float_value(raw_segment.get("end"), source.end),
                    speaker=_optional_string(raw_segment.get("speaker"), source.speaker),
                    corrected_text=text,
                )
            )
        return segments

    text = _string_value(payload, "corrected_text", required=True)
    return [
        CorrectedSegment(
            start=chunk.start,
            end=chunk.end,
            speaker=_chunk_speaker(chunk),
            corrected_text=text,
        )
    ]


def _joined_corrected_text(segments: list[CorrectedSegment]) -> str:
    return "\n".join(
        segment.corrected_text.strip()
        for segment in segments
        if segment.corrected_text.strip()
    )


def _string_value(
    payload: Mapping[str, Any],
    key: str,
    *,
    required: bool,
) -> str:
    value = payload.get(key)
    if isinstance(value, str):
        return value
    if required:
        raise CorrectionError(f"Provider response is missing string field: {key}")
    return ""


def _optional_string(value: Any, fallback: str | None) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _float_value(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _string_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): str(item).strip()
        for key, item in value.items()
        if str(key).strip() and str(item).strip()
    }


@contextmanager
def _legacy_progress_stage(output_console: Console, total: int):  # type: ignore[return]
    """Backwards-compat fallback: inline Progress when no PipelineProgress."""

    class _LegacyHandle:
        def __init__(self, progress: Progress, task_id: Any) -> None:
            self._progress = progress
            self._task_id = task_id

        def advance(self, delta: float = 1) -> None:
            self._progress.advance(self._task_id, delta)

        def advance_to(self, completed: float) -> None:
            pass

        def update_description(self, description: str) -> None:
            self._progress.update(self._task_id, description=description)

    with Progress(console=output_console, transient=True) as progress:
        task_id = progress.add_task("Correcting chunks", total=total)
        yield _LegacyHandle(progress, task_id)
