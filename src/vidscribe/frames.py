"""Keyframe extraction helpers."""

from __future__ import annotations

import json
import re
import select as _select
import subprocess
import time as _time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from vidscribe.progress import PipelineProgress


class FrameExtractionError(RuntimeError):
    """Raised when ffmpeg cannot extract video frames."""


class FFmpegMissingError(FrameExtractionError):
    """Raised when the ffmpeg binary itself is unavailable.

    Distinguished from generic FrameExtractionError so the auto-fallback
    pipeline doesn't pointlessly try other strategies (they all need ffmpeg).
    """


class FrameInfo(BaseModel):
    """A keyframe or sampled frame extracted from the source video."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ts: float
    path: Path
    scene_change: bool


_METADATA_FRAME_RE = re.compile(r"frame:\s*(?P<frame>\d+).*pts_time:(?P<ts>[-\d.]+)")
_SCENE_SCORE_RE = re.compile(r"lavfi\.scene_score=(?P<score>[-\d.]+)")
_PROGRESS_OTIME_RE = re.compile(r"^out_time_us=(\d+)")

STUCK_TIMEOUT: float = 120.0  # seconds without out_time_us change before killing ffmpeg


FramesStrategy = Literal["auto", "scene-detect", "sample-only", "seek"]


def _build_scene_detect_command(
    video: Path,
    output_pattern: Path,
    scene_threshold: float,
    sample_every: float,
) -> list[str]:
    """Return ffmpeg command using scene-detect + uniform sampling filter."""
    select_filter = (
        f"select='eq(n,0)+gt(scene,{scene_threshold})+"
        f"isnan(prev_selected_t)+gte(t-prev_selected_t,{sample_every})',"
        "metadata=print:key=lavfi.scene_score,showinfo"
    )
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-an",  # ignore audio: not needed for frames, avoids demuxer/audio-sync stalls
        "-vf",
        select_filter,
        "-fps_mode",
        "vfr",
        "-progress",
        "pipe:1",
        "-nostats",
        str(output_pattern),
    ]


def _build_sample_only_command(
    video: Path,
    output_pattern: Path,
    sample_every: float,
) -> list[str]:
    """Return simpler ffmpeg command using only uniform sampling (no scene-detect).

    Adds error-tolerant input flags so that corrupt frames are skipped instead
    of stalling decoding (we hit this on partly-broken meeting recordings).
    """
    select_filter = (
        f"select='eq(n,0)+gte(t-prev_selected_t,{sample_every})',showinfo"
    )
    return [
        "ffmpeg",
        "-y",
        # Error-tolerant input options: skip corrupt frames, regenerate PTS,
        # don't abort on minor decode errors. These have to be BEFORE -i.
        "-err_detect",
        "ignore_err",
        "-fflags",
        "+discardcorrupt+genpts",
        # Decode only I-frames (keyframes). 5-50x faster, and skips broken
        # P/B frames that can stall decoding on partly-corrupt recordings.
        # For meetings/screencasts keyframes are dense enough for sampling.
        "-skip_frame",
        "nokey",
        "-i",
        str(video),
        "-an",  # ignore audio: not needed for frames, avoids demuxer/audio-sync stalls
        "-vf",
        select_filter,
        "-fps_mode",
        "vfr",
        "-progress",
        "pipe:1",
        "-nostats",
        str(output_pattern),
    ]


def _seek_extract_frames(
    video: Path,
    output_dir: Path,
    total_seconds: float,
    sample_every: float,
    *,
    pipeline_progress: "PipelineProgress | None",
    per_call_timeout: float = 15.0,
) -> list["FrameInfo"]:
    """Per-segment seek-based frame extraction.

    Each frame is a separate ffmpeg invocation with ``-ss T -frames:v 1``.
    A corrupt segment can fail/timeout one call but other calls keep going.
    Returns the list of FrameInfo for successfully extracted frames.
    """
    if pipeline_progress is not None:
        pipeline_progress.log(
            "[5/9] Frames: switching to seek-based extraction (one ffmpeg per frame)"
        )

    if total_seconds <= 0:
        return []

    timestamps: list[float] = []
    t = 0.0
    while t < total_seconds:
        timestamps.append(t)
        t += sample_every

    frames: list[FrameInfo] = []
    n_failed = 0
    last_log = _time.monotonic()

    for idx, ts in enumerate(timestamps, start=1):
        out_path = output_dir / f"frame-{idx:06d}.jpg"
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{ts:.3f}",
            "-i",
            str(video),
            "-an",
            "-frames:v",
            "1",
            "-q:v",
            "5",
            str(out_path),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=per_call_timeout,
                check=False,
            )
            if result.returncode == 0 and out_path.exists():
                frames.append(FrameInfo(ts=ts, path=out_path, scene_change=False))
            else:
                n_failed += 1
        except FileNotFoundError as exc:
            raise FFmpegMissingError(
                "ffmpeg was not found. Install ffmpeg and make sure it is on PATH."
            ) from exc
        except subprocess.TimeoutExpired:
            n_failed += 1

        if pipeline_progress is not None:
            now = _time.monotonic()
            if now - last_log >= 2.0:
                last_log = now
                mm, ss = divmod(int(ts), 60)
                hh, mm = divmod(mm, 60)
                ts_str = f"{hh}:{mm:02d}:{ss:02d}" if hh else f"{mm}:{ss:02d}"
                pct = ts / total_seconds * 100 if total_seconds > 0 else 0
                pipeline_progress.log(
                    f"[5/9] Frames (seek): {ts_str} processed ({pct:.0f}%)"
                    f" | {len(frames)} ok | {n_failed} skipped"
                )

    if pipeline_progress is not None:
        pipeline_progress.log(
            f"[5/9] Frames (seek): {len(frames)} extracted, {n_failed} skipped (corrupt segments)"
        )

    # If seek-based extraction also produced nothing (all frames failed), the
    # video is unreadable — surface that as an error instead of returning empty.
    if not frames and n_failed > 0:
        raise FrameExtractionError(
            f"seek-based extraction failed for all {n_failed} timestamps; video is likely unreadable"
        )
    return frames


def _run_ffmpeg_once(
    command: list[str],
    video: Path,
    output_dir: Path,
    total_seconds: float,
    pipeline_progress: "PipelineProgress | None",
    stuck_timeout: float,
) -> str:
    """Run one ffmpeg invocation and return stderr text.

    Raises :exc:`FrameExtractionError` on non-zero exit or watchdog trigger.
    """
    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise FFmpegMissingError(
            "ffmpeg was not found. Install ffmpeg and make sure it is on PATH."
        ) from exc

    stdout_lines: list[str] = []
    _read_frames_progress(
        proc,
        total_seconds=total_seconds,
        pipeline_progress=pipeline_progress,
        output_dir=output_dir,
        stdout_lines=stdout_lines,
        stuck_timeout=stuck_timeout,
    )

    stderr_text = proc.stderr.read() if proc.stderr else ""  # type: ignore[union-attr]
    returncode = proc.wait()
    if returncode != 0:
        details = stderr_text.strip()
        message = f"ffmpeg failed to extract frames from {video}"
        if details:
            message = f"{message}: {details}"
        raise FrameExtractionError(message)

    return stderr_text


def extract(
    video_path: Path | str,
    out_dir: Path | str,
    scene_threshold: float = 0.3,
    sample_every: float = 10.0,
    *,
    pipeline_progress: "PipelineProgress | None" = None,
    frames_strategy: "FramesStrategy" = "auto",
    stuck_timeout: float = STUCK_TIMEOUT,
) -> list[FrameInfo]:
    """Extract scene-change and sampled frames with ffmpeg.

    ``frames_strategy`` controls which extraction filter is used:
    - ``"auto"`` (default): try scene-detect first, fall back to sample-only on failure/timeout.
    - ``"scene-detect"``: always use scene-detect filter; no fallback.
    - ``"sample-only"``: skip scene-detect, use uniform sampling only.
    """

    video = Path(video_path)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean stale jpgs from a previous (possibly interrupted) run so that
    # the running counter ("N frames so far") reflects the current attempt.
    for stale in output_dir.glob("frame-*.jpg"):
        try:
            stale.unlink()
        except OSError:
            pass
    stale_json = output_dir / "frames.json"
    if stale_json.exists():
        try:
            stale_json.unlink()
        except OSError:
            pass

    output_pattern = output_dir / "frame-%06d.jpg"

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

    use_scene_detect = frames_strategy in ("auto", "scene-detect")
    use_fallback = frames_strategy == "auto"
    seek_only = frames_strategy == "seek"

    fallback_used = False
    seek_used = False
    stderr_text = ""
    seek_frames: list[FrameInfo] | None = None

    with ctx:
        if seek_only:
            seek_frames = _seek_extract_frames(
                video,
                output_dir,
                total_seconds,
                sample_every,
                pipeline_progress=pipeline_progress,
            )
            seek_used = True
        elif use_scene_detect:
            command = _build_scene_detect_command(video, output_pattern, scene_threshold, sample_every)
            try:
                stderr_text = _run_ffmpeg_once(
                    command, video, output_dir, total_seconds, pipeline_progress, stuck_timeout
                )
            except FFmpegMissingError:
                # ffmpeg binary itself is unavailable; fallbacks won't help
                raise
            except FrameExtractionError:
                if not use_fallback:
                    raise
                # scene-detect failed or timed out — retry with sample-only
                if pipeline_progress is not None:
                    pipeline_progress.log(
                        "[5/9] Frames: scene-detect failed/timed-out, retrying with sample-only strategy"
                    )
                # Clean up any partially extracted frames before retry
                for partial in output_dir.glob("frame-*.jpg"):
                    try:
                        partial.unlink()
                    except OSError:
                        pass
                fallback_cmd = _build_sample_only_command(video, output_pattern, sample_every)
                try:
                    stderr_text = _run_ffmpeg_once(
                        fallback_cmd, video, output_dir, total_seconds, pipeline_progress, stuck_timeout
                    )
                    fallback_used = True
                except FFmpegMissingError:
                    raise
                except FrameExtractionError as sample_error:
                    # Sample-only also stalled — switch to seek-based per-frame extraction.
                    # If video duration was unknown (e.g. ffprobe failed because the
                    # input is bad), seek-based has no timestamps to iterate; surface
                    # the original ffmpeg error instead of silently returning [].
                    if total_seconds <= 0:
                        raise sample_error
                    if pipeline_progress is not None:
                        pipeline_progress.log(
                            "[5/9] Frames: sample-only stalled too, switching to seek-based per-frame extraction"
                        )
                    for partial in output_dir.glob("frame-*.jpg"):
                        try:
                            partial.unlink()
                        except OSError:
                            pass
                    seek_frames = _seek_extract_frames(
                        video,
                        output_dir,
                        total_seconds,
                        sample_every,
                        pipeline_progress=pipeline_progress,
                    )
                    seek_used = True
        else:
            # sample-only, no scene-detect
            command = _build_sample_only_command(video, output_pattern, sample_every)
            stderr_text = _run_ffmpeg_once(
                command, video, output_dir, total_seconds, pipeline_progress, stuck_timeout
            )

    if seek_used:
        frames: list[FrameInfo] = seek_frames or []
    else:
        frames = _frames_from_ffmpeg_log(
            stderr_text,
            output_dir=output_dir,
            scene_threshold=scene_threshold,
        )
    _write_frames_json(output_dir / "frames.json", frames)

    if pipeline_progress is not None:
        elapsed = _time.monotonic() - t0
        if seek_used:
            pipeline_progress.log(
                f"[5/9] Frames done in {elapsed:.1f}s"
                f" | {len(frames)} frames (seek-based per-frame fallback)"
            )
        elif fallback_used:
            pipeline_progress.log(
                f"[5/9] Frames done in {elapsed:.1f}s"
                f" | {len(frames)} frames (sample-only fallback)"
            )
        else:
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
    stuck_timeout: float = STUCK_TIMEOUT,
) -> None:
    """Read ffmpeg progress from stdout, logging every ~2 real-time seconds.

    Uses non-blocking I/O (select) so the loop does not hang when ffmpeg stalls.
    If ``out_time_us`` does not advance for ``stuck_timeout`` seconds the
    subprocess is killed and :exc:`FrameExtractionError` is raised.
    """
    if proc.stdout is None:
        return

    out_time_us: int = 0
    _last_log: float = _time.monotonic()
    _log_interval: float = 2.0
    last_progress_at: float = _time.monotonic()
    # Buffer for partial lines when using non-blocking reads
    _line_buf: str = ""

    while proc.poll() is None:
        now = _time.monotonic()

        # Watchdog: check on EVERY iteration. ffmpeg keeps spamming non-out_time
        # progress lines (bitrate=, speed=, etc.) so the select() ready branch
        # may always be true while out_time_us is stuck. Checking unconditionally
        # is the only way to catch a stall.
        if now - last_progress_at > stuck_timeout:
            audio_s = out_time_us / 1_000_000
            mm, ss = divmod(int(audio_s), 60)
            hh, mm = divmod(mm, 60)
            stuck_at = f"{hh}:{mm:02d}:{ss:02d}" if hh else f"{mm}:{ss:02d}"
            if pipeline_progress is not None:
                pipeline_progress.log(
                    f"[5/9] FFmpeg stuck at {stuck_at} for {stuck_timeout:.0f}s — killing subprocess"
                )
            try:
                proc.kill()
            except OSError:
                pass
            raise FrameExtractionError(
                f"ffmpeg stuck at {stuck_at} for {stuck_timeout:.0f}s, killed"
            )

        # Non-blocking read with 1 s timeout
        try:
            ready, _, _ = _select.select([proc.stdout], [], [], 1.0)
        except (ValueError, OSError):
            # stdout already closed
            break

        if ready:
            chunk = proc.stdout.read(4096)  # type: ignore[arg-type]
            if not chunk:
                break
            _line_buf += chunk
            # Process all complete lines
            while "\n" in _line_buf:
                line, _line_buf = _line_buf.split("\n", 1)
                line = line.strip()
                stdout_lines.append(line)
                m = _PROGRESS_OTIME_RE.match(line)
                if m:
                    try:
                        new_val = int(m.group(1))
                    except ValueError:
                        new_val = out_time_us
                    if new_val != out_time_us:
                        out_time_us = new_val
                        last_progress_at = _time.monotonic()

        # Periodic progress log
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

    # Drain any remaining data after process exited
    if proc.stdout is not None:
        remaining = proc.stdout.read()
        if remaining:
            _line_buf += remaining
        for line in _line_buf.splitlines():
            line = line.strip()
            if line:
                stdout_lines.append(line)
                m = _PROGRESS_OTIME_RE.match(line)
                if m:
                    try:
                        new_val = int(m.group(1))
                        if new_val != out_time_us:
                            out_time_us = new_val
                    except ValueError:
                        pass


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
