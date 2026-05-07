"""Keyframe extraction helpers."""

from __future__ import annotations

import json
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from vidscribe.progress import PipelineProgress


class FrameExtractionError(RuntimeError):
    """Raised when ffmpeg cannot extract video frames."""


class FrameInfo(BaseModel):
    """A keyframe or sampled frame extracted from the source video."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ts: float
    path: Path
    scene_change: bool


_METADATA_FRAME_RE = re.compile(r"frame:\s*(?P<frame>\d+).*pts_time:(?P<ts>[-\d.]+)")
_SCENE_SCORE_RE = re.compile(r"lavfi\.scene_score=(?P<score>[-\d.]+)")


def extract(
    video_path: Path | str,
    out_dir: Path | str,
    scene_threshold: float = 0.3,
    sample_every: float = 10.0,
    *,
    pipeline_progress: "PipelineProgress | None" = None,
) -> list[FrameInfo]:
    """Extract scene-change and sampled frames with ffmpeg."""

    video = Path(video_path)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = output_dir / "frame-%06d.jpg"
    select_filter = (
        f"select='eq(n,0)+gt(scene,{scene_threshold})+"
        f"isnan(prev_selected_t)+gte(t-prev_selected_t,{sample_every})',"
        "metadata=print:key=lavfi.scene_score,showinfo"
    )
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vf",
        select_filter,
        "-fps_mode",
        "vfr",
        str(output_pattern),
    ]

    import time as _time

    ctx = pipeline_progress.stage("frames") if pipeline_progress is not None else _null_stage()

    if pipeline_progress is not None:
        pipeline_progress.log(
            f"[5/9] Frames extraction: scene-detect {scene_threshold}, sample every {sample_every:.0f}s"
        )
    t0 = _time.monotonic()

    with ctx:
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
        except FileNotFoundError as exc:
            raise FrameExtractionError(
                "ffmpeg was not found. Install ffmpeg and make sure it is on PATH."
            ) from exc
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or exc.stdout or "").strip()
            message = f"ffmpeg failed to extract frames from {video}"
            if details:
                message = f"{message}: {details}"
            raise FrameExtractionError(message) from exc

    frames = _frames_from_ffmpeg_log(
        result.stderr or result.stdout or "",
        output_dir=output_dir,
        scene_threshold=scene_threshold,
    )
    _write_frames_json(output_dir / "frames.json", frames)

    if pipeline_progress is not None:
        elapsed = _time.monotonic() - t0
        pipeline_progress.log(f"[5/9] Frames done in {elapsed:.1f}s | {len(frames)} frames")

    return frames


@contextmanager
def _null_stage():  # type: ignore[return]
    """No-op context manager used when no PipelineProgress is provided."""
    yield


def _frames_from_ffmpeg_log(
    log: str,
    output_dir: Path,
    scene_threshold: float,
) -> list[FrameInfo]:
    parsed: list[tuple[float, float | None]] = []
    pending_ts: float | None = None

    for line in log.splitlines():
        frame_match = _METADATA_FRAME_RE.search(line)
        if frame_match:
            pending_ts = float(frame_match.group("ts"))
            continue

        score_match = _SCENE_SCORE_RE.search(line)
        if score_match and pending_ts is not None:
            parsed.append((pending_ts, float(score_match.group("score"))))
            pending_ts = None

    if pending_ts is not None:
        parsed.append((pending_ts, None))

    image_paths = sorted(output_dir.glob("frame-*.jpg"))
    frames: list[FrameInfo] = []
    for idx, (ts, scene_score) in enumerate(parsed, start=1):
        path = (
            image_paths[idx - 1]
            if idx <= len(image_paths)
            else output_dir / f"frame-{idx:06d}.jpg"
        )
        frames.append(
            FrameInfo(
                ts=ts,
                path=path,
                scene_change=bool(
                    scene_score is not None and scene_score >= scene_threshold
                ),
            )
        )
    return frames


def _write_frames_json(path: Path, frames: list[FrameInfo]) -> None:
    path.write_text(
        json.dumps(
            [frame.model_dump(mode="json") for frame in frames],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
