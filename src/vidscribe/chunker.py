"""Transcript chunking helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from vidscribe.frames import FrameInfo
from vidscribe.stt import SttResult, SttSegment

if TYPE_CHECKING:
    from vidscribe.progress import PipelineProgress


ChunkStrategy = Literal["speaker", "time", "scene"]


class Chunk(BaseModel):
    """A provider-sized transcript chunk with relevant visual context."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    idx: int
    start: float
    end: float
    segments: list[SttSegment]
    frame_paths: list[Path] = Field(default_factory=list)
    surrounding_context: str = ""


def chunk(
    stt: SttResult,
    frames: list[FrameInfo],
    strategy: ChunkStrategy,
    window_s: float = 180,
    *,
    pipeline_progress: "PipelineProgress | None" = None,
) -> list[Chunk]:
    """Split diarized STT into correction chunks."""

    import time as _time

    if window_s <= 0:
        raise ValueError("window_s must be greater than 0")

    segments = sorted(stt.segments, key=lambda segment: (segment.start, segment.end))
    ordered_frames = sorted(frames, key=lambda frame: frame.ts)
    if not segments:
        return []

    if pipeline_progress is not None:
        pipeline_progress.log(f"[6/9] Chunking: {strategy} strategy")
    t0 = _time.monotonic()

    if strategy == "speaker":
        groups = _speaker_groups(segments, window_s)
    elif strategy == "time":
        groups = _time_groups(segments, window_s)
    elif strategy == "scene":
        groups = _scene_groups(segments, ordered_frames, window_s)
    else:
        raise ValueError(f"Unsupported chunk strategy: {strategy}")

    result = [
        _build_chunk(
            idx=idx,
            segments=group,
            all_segments=segments,
            frames=ordered_frames,
        )
        for idx, group in enumerate(groups)
        if group
    ]

    if pipeline_progress is not None:
        elapsed = _time.monotonic() - t0
        n = len(result)
        if n > 0:
            total_dur = sum(c.end - c.start for c in result)
            avg_s = total_dur / n
            pipeline_progress.log(
                f"[6/9] Chunking done in {elapsed:.2f}s | {n} chunks (avg {avg_s:.1f}s each)"
            )
        else:
            pipeline_progress.log(f"[6/9] Chunking done in {elapsed:.2f}s | 0 chunks")

    return result


def _speaker_groups(segments: list[SttSegment], window_s: float) -> list[list[SttSegment]]:
    groups: list[list[SttSegment]] = []
    current: list[SttSegment] = []

    for segment in segments:
        if not current:
            current = [segment]
            continue

        speaker_changed = segment.speaker != current[-1].speaker
        too_wide = segment.end - current[0].start > window_s
        if speaker_changed or too_wide:
            groups.append(current)
            current = [segment]
        else:
            current.append(segment)

    if current:
        groups.append(current)
    return groups


def _time_groups(segments: list[SttSegment], window_s: float) -> list[list[SttSegment]]:
    groups: list[list[SttSegment]] = []
    window_start = segments[0].start
    window_end = window_start + window_s
    current: list[SttSegment] = []

    for segment in segments:
        while current and segment.start >= window_end:
            groups.append(current)
            current = []
            window_start = window_end
            window_end = window_start + window_s

        while not current and segment.start >= window_end:
            window_start = window_end
            window_end = window_start + window_s

        current.append(segment)

    if current:
        groups.append(current)
    return groups


def _scene_groups(
    segments: list[SttSegment],
    frames: list[FrameInfo],
    window_s: float,
) -> list[list[SttSegment]]:
    groups: list[list[SttSegment]] = []
    current: list[SttSegment] = []
    scene_boundaries = [
        frame.ts
        for frame in frames
        if frame.scene_change and segments[0].start < frame.ts < segments[-1].end
    ]
    next_boundary_idx = 0
    next_boundary = (
        scene_boundaries[next_boundary_idx]
        if next_boundary_idx < len(scene_boundaries)
        else None
    )

    for segment in segments:
        boundary_crossed = (
            current and next_boundary is not None and segment.start >= next_boundary
        )
        too_wide = current and segment.end - current[0].start > window_s
        if boundary_crossed or too_wide:
            groups.append(current)
            current = []
            while (
                next_boundary_idx < len(scene_boundaries)
                and segment.start >= scene_boundaries[next_boundary_idx]
            ):
                next_boundary_idx += 1
            next_boundary = (
                scene_boundaries[next_boundary_idx]
                if next_boundary_idx < len(scene_boundaries)
                else None
            )

        current.append(segment)

    if current:
        groups.append(current)
    return groups


def _build_chunk(
    idx: int,
    segments: list[SttSegment],
    all_segments: list[SttSegment],
    frames: list[FrameInfo],
) -> Chunk:
    start = min(segment.start for segment in segments)
    end = max(segment.end for segment in segments)
    return Chunk(
        idx=idx,
        start=start,
        end=end,
        segments=segments,
        frame_paths=_frame_paths_for_window(frames, start, end),
        surrounding_context=_surrounding_context(segments, all_segments),
    )


def _frame_paths_for_window(
    frames: list[FrameInfo],
    start: float,
    end: float,
) -> list[Path]:
    if not frames:
        return []

    selected = [frame for frame in frames if start <= frame.ts <= end]
    midpoint = start + ((end - start) / 2)
    reference = min(frames, key=lambda frame: abs(frame.ts - midpoint))
    if reference not in selected:
        selected.append(reference)

    selected.sort(key=lambda frame: frame.ts)
    seen: set[Path] = set()
    paths: list[Path] = []
    for frame in selected:
        if frame.path in seen:
            continue
        seen.add(frame.path)
        paths.append(frame.path)
    return paths


def _surrounding_context(
    chunk_segments: list[SttSegment],
    all_segments: list[SttSegment],
) -> str:
    first_idx = all_segments.index(chunk_segments[0])
    last_idx = all_segments.index(chunk_segments[-1])
    parts: list[str] = []
    if first_idx > 0:
        parts.append(f"Before: {all_segments[first_idx - 1].text}")
    if last_idx + 1 < len(all_segments):
        parts.append(f"After: {all_segments[last_idx + 1].text}")
    return "\n".join(parts)
