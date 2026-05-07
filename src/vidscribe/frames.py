"""Keyframe extraction helpers."""

from __future__ import annotations

import json
import re
import subprocess
import time as _time
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
_PROGRESS_OTIME_RE = re.compile(r"^out_time_us=(\d+)")


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
        "-progress",
        "pipe:1",
        "-nostats",
        str(output_pattern),
    ]

    ctx = pipeline_progress.stage("frames") if pipeline_progress is not None else _null_stage()

    total_seconds = _probe_duration(video)

    if pipeline_progress is not None:
        if total_seconds > 0:
            tm, ts = divmod(int(total_seconds), 60)
            th, tm = divmod(tm, 60)
            duration_str = f"{th}:{tm:02d}:{ts:02d}" if th else f"{tm}:{ts:02d}"
            pipeline_progress.log(
                f"[5/9] Frames extraction: video {duration_str},"
                f" scene-detect {scene_threshold}, sample every {sample_every:.0f}s"
            )
        else:
            pipeline_progress.log(
                f"[5/9] Frames extraction: scene-detect {scene_threshold}, sample every {sample_every:.0f}s"
            )
    t0 = _time.monotonic()

    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise FrameExtractionError(
            "ffmpeg was not found. Install ffmpeg and make sure it is on PATH."
        ) from exc

    # Read progress from stdout; stderr holds frame metadata — read after completion
    stdout_lines: list[str] = []
    with ctx:
        _read_frames_progress(
            proc,
            total_seconds=total_seconds,
            pipeline_progress=pipeline_progress,
            output_dir=output_dir,
            stdout_lines=stdout_lines,
        )

    stderr_text = proc.stderr.read() if proc.stderr else ""  # type: ignore[union-attr]
    returncode = proc.wait()
    if returncode != 0:
        details = stderr_text.strip()
        message = f"ffmpeg failed to extract frames from {video}"
        if details:
            message = f"{message}: {details}"
        raise FrameExtractionError(message)

    frames = _frames_from_ffmpeg_log(
        stderr_text,
        output_dir=output_dir,
        scene_threshold=scene_threshold,
    )
    _write_frames_json(output_dir / "frames.json", frames)

    if pipeline_progress is not None:
        elapsed = _time.monotonic() - t0
        n_scene = sum(1 for f in frames if f.scene_change)
        n_sampled = len(frames) - n_scene
        pipeline_progress.log(
            f"[5/9] Frames done in {elapsed:.1f}s"
            f" | {len(frames)} frames extracted"
            f" ({n_scene} scene-changes + {n_sampled} sampled)"
        )

    return frames


def _read_frames_progress(
    proc: "subprocess.Popen[str]",
    *,
    total_seconds: float,
    pipeline_progress: "PipelineProgress | None",
    output_dir: Path,
    stdout_lines: list[str],
) -> None:
    """Read ffmpeg progress from stdout, logging every ~2 real-time seconds."""
    if proc.stdout is None:
        return

    out_time_us: int = 0
    _last_log: float = _time.monotonic()
    _log_interval: float = 2.0

    for line in proc.stdout:
        line = line.strip()
        stdout_lines.append(line)
        m = _PROGRESS_OTIME_RE.match(line)
        if m:
            try:
                out_time_us = int(m.group(1))
            except ValueError:
                pass

        if pipeline_progress is not None:
            now = _time.monotonic()
            if now - _last_log >= _log_interval:
                _last_log = now
                audio_s = out_time_us / 1_000_000
                mm, ss = divmod(int(audio_s), 60)
                hh, mm = divmod(mm, 60)
                proc_str = f"{hh}:{mm:02d}:{ss:02d}" if hh else f"{mm}:{ss:02d}"
                # Count extracted frames so far
                n_so_far = len(list(output_dir.glob("frame-*.jpg")))
                if total_seconds > 0:
                    pct = min(100.0, audio_s / total_seconds * 100.0)
                    tm, ts = divmod(int(total_seconds), 60)
                    th, tm = divmod(tm, 60)
                    total_str = f"{th}:{tm:02d}:{ts:02d}" if th else f"{tm}:{ts:02d}"
                    pipeline_progress.log(
                        f"[5/9] Frames: {proc_str}/{total_str} ({pct:.0f}%) | {n_so_far} frames so far"
                    )
                else:
                    pipeline_progress.log(
                        f"[5/9] Frames: {proc_str} processed | {n_so_far} frames so far"
                    )


def _probe_duration(video: Path) -> float:
    """Return video duration in seconds via ffprobe, or 0.0 on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if result.returncode == 0:
            return float(result.stdout.strip() or 0.0)
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        pass
    return 0.0


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
