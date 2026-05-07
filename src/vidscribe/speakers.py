"""Speaker identification helpers."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from vidscribe.cache import Cache
from vidscribe.frames import FrameInfo
from vidscribe.prompts import render
from vidscribe.provider import Provider
from vidscribe.stt import SttResult, SttSegment

if TYPE_CHECKING:
    from vidscribe.progress import PipelineProgress


SpeakerMap = dict[str, str]

_SPEAKER_SUFFIX_RE = re.compile(r"(\d+)$")


def identify(
    stt: SttResult,
    frames: list[FrameInfo],
    provider: Provider,
    manual: str | Sequence[str] | Mapping[str, str] | None = None,
    *,
    cache: Cache | None = None,
    namespace_key: str | None = None,
    timeout: int = 300,
    pipeline_progress: "PipelineProgress | None" = None,
) -> SpeakerMap:
    """Identify diarized speakers with manual overrides, LLM evidence, and fallbacks."""

    speaker_ids = _speaker_ids(stt)
    if not speaker_ids:
        return {}

    manual_map = _manual_map(manual, speaker_ids)
    if len(manual_map) == len(speaker_ids):
        return _finalize(speaker_ids, manual_map)

    cache_key = None
    if cache is not None:
        cache_key = _cache_key(
            cache,
            namespace_key,
            "speakers",
            stt=stt,
            frame_paths=[frame.path for frame in frames],
            manual=manual_map,
            provider=provider.__class__.__name__,
            model=getattr(provider, "model", None),
        )
        cached = cache.get("speakers", cache_key)
        if isinstance(cached, dict):
            return _finalize(speaker_ids, _string_dict(cached))

    import time as _time

    provider_name = provider.__class__.__name__
    provider_model = getattr(provider, "model", None) or ""
    provider_desc = f"{provider_name}/{provider_model}" if provider_model else provider_name

    if pipeline_progress is not None:
        pipeline_progress.log(f"[7/9] Speaker identification: {provider_desc}")

    ctx = (
        pipeline_progress.stage("speakers")
        if pipeline_progress is not None
        else _null_stage()
    )
    t0 = _time.monotonic()
    with ctx:
        selected = _representative_segments(stt, speaker_ids)
        frame_paths = [path.resolve() for path in _representative_frame_paths(frames, selected)]
        prompt = render(
            "identify_speakers",
            transcript=_transcript_excerpt(selected),
            speakers=speaker_ids,
            frame_paths=frame_paths,
        )
        response = provider.correct(prompt, frame_paths=frame_paths, timeout=timeout)
        provider_map = _provider_speaker_map(response.raw_json, response.text)
        speaker_map = _finalize(speaker_ids, provider_map | manual_map)

    if pipeline_progress is not None:
        elapsed = _time.monotonic() - t0
        mapping_str = ", ".join(f"{k}→{v}" for k, v in sorted(speaker_map.items()))
        pipeline_progress.log(
            f"[7/9] Speakers identified in {elapsed:.1f}s | {mapping_str}"
        )

    if cache is not None and cache_key is not None:
        cache.set("speakers", cache_key, speaker_map)

    return speaker_map


@contextmanager
def _null_stage():  # type: ignore[return]
    """No-op context manager used when no PipelineProgress is provided."""
    yield


def _cache_key(cache: Cache, namespace_key: str | None, stage: str, **inputs: Any) -> str:
    key = cache.key_for(stage, **inputs)
    if namespace_key is None:
        return key
    return f"{namespace_key}/{key}"


def _speaker_ids(stt: SttResult) -> list[str]:
    ids = {
        segment.speaker
        for segment in stt.segments
        if segment.speaker and segment.speaker.strip()
    }
    return sorted(ids, key=_speaker_sort_key)


def _speaker_sort_key(speaker_id: str) -> tuple[str, int, str]:
    match = _SPEAKER_SUFFIX_RE.search(speaker_id)
    if match:
        return (speaker_id[: match.start()], int(match.group(1)), speaker_id)
    return (speaker_id, -1, speaker_id)


def _manual_map(
    manual: str | Sequence[str] | Mapping[str, str] | None,
    speaker_ids: list[str],
) -> SpeakerMap:
    if manual is None:
        return {}

    if isinstance(manual, Mapping):
        return {
            speaker_id: name.strip()
            for speaker_id, name in manual.items()
            if speaker_id in speaker_ids and isinstance(name, str) and name.strip()
        }

    if isinstance(manual, str):
        names = [part.strip() for part in manual.split(",")]
    else:
        names = [str(part).strip() for part in manual]

    return {
        speaker_id: name
        for speaker_id, name in zip(speaker_ids, names, strict=False)
        if name
    }


def _representative_segments(
    stt: SttResult,
    speaker_ids: list[str],
    per_speaker: int = 2,
) -> list[SttSegment]:
    by_speaker: dict[str, list[SttSegment]] = defaultdict(list)
    for segment in sorted(stt.segments, key=lambda item: (item.start, item.end)):
        if segment.speaker in speaker_ids and len(by_speaker[segment.speaker]) < per_speaker:
            by_speaker[segment.speaker].append(segment)

    selected: list[SttSegment] = []
    for speaker_id in speaker_ids:
        selected.extend(by_speaker[speaker_id])
    return sorted(selected, key=lambda item: (item.start, item.end))


def _representative_frame_paths(
    frames: list[FrameInfo],
    segments: list[SttSegment],
) -> list[Path]:
    if not frames or not segments:
        return []

    ordered_frames = sorted(frames, key=lambda frame: frame.ts)
    selected: list[FrameInfo] = []
    for segment in segments:
        in_segment = [
            frame for frame in ordered_frames if segment.start <= frame.ts <= segment.end
        ]
        if in_segment:
            selected.extend(in_segment[:1])
            continue

        midpoint = segment.start + ((segment.end - segment.start) / 2)
        selected.append(min(ordered_frames, key=lambda frame: abs(frame.ts - midpoint)))

    seen: set[Path] = set()
    paths: list[Path] = []
    for frame in sorted(selected, key=lambda item: item.ts):
        if frame.path in seen:
            continue
        seen.add(frame.path)
        paths.append(frame.path)
    return paths


def _transcript_excerpt(segments: list[SttSegment]) -> str:
    lines = []
    for segment in segments:
        speaker = segment.speaker or "UNKNOWN"
        lines.append(
            f"[{segment.start:.2f}-{segment.end:.2f}] {speaker}: {segment.text.strip()}"
        )
    return "\n".join(lines)


def _provider_speaker_map(raw_json: dict[str, Any], text: str) -> SpeakerMap:
    speakers = raw_json.get("speakers")
    if not isinstance(speakers, dict) and text.strip():
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, dict):
            speakers = parsed.get("speakers")

    if not isinstance(speakers, dict):
        return {}
    return _string_dict(speakers)


def _string_dict(value: Mapping[Any, Any]) -> SpeakerMap:
    return {
        str(key): str(item).strip()
        for key, item in value.items()
        if str(key).strip() and str(item).strip()
    }


def _finalize(speaker_ids: list[str], candidates: Mapping[str, str]) -> SpeakerMap:
    return {
        speaker_id: candidates.get(speaker_id, "") or _fallback_name(speaker_id, idx)
        for idx, speaker_id in enumerate(speaker_ids)
    }


def _fallback_name(speaker_id: str, idx: int) -> str:
    match = _SPEAKER_SUFFIX_RE.search(speaker_id)
    number = int(match.group(1)) if match else idx
    return f"s{number:02d}"
