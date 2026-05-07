"""Audio extraction helpers."""

from __future__ import annotations

import subprocess
import time as _time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vidscribe.progress import PipelineProgress


class AudioExtractionError(RuntimeError):
    """Raised when ffmpeg cannot extract audio from a video."""


def extract(
    video_path: Path | str,
    out_path: Path | str,
    *,
    pipeline_progress: "PipelineProgress | None" = None,
) -> Path:
    """Extract a mono 16 kHz WAV audio track from a video with ffmpeg."""

    video = Path(video_path)
    output = Path(out_path)
    if output.suffix.lower() != ".wav":
        output = output.with_suffix(".wav")

    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        "-progress",
        "pipe:1",
        "-nostats",
        str(output),
    ]

    ctx = pipeline_progress.stage("audio") if pipeline_progress is not None else _null_stage()

    t0 = _time.monotonic()

    if pipeline_progress is not None:
        pipeline_progress.log(f"[1/9] Audio extraction: {video} → {output}")

    with ctx as handle:
        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as exc:
            raise AudioExtractionError(
                "ffmpeg was not found. Install ffmpeg and make sure it is on PATH."
            ) from exc

        _run_ffmpeg_with_progress(
            proc,
            pipeline_progress=pipeline_progress,
            stage_label="[1/9] Audio extraction",
            handle=handle,
        )

        returncode = proc.wait()
        if returncode != 0:
            stderr_out = (proc.stderr.read() if proc.stderr else "") or ""
            message = f"ffmpeg failed to extract audio from {video}"
            if stderr_out.strip():
                message = f"{message}: {stderr_out.strip()}"
            raise AudioExtractionError(message)

    if pipeline_progress is not None:
        elapsed = _time.monotonic() - t0
        size_mb = output.stat().st_size / 1024 / 1024 if output.exists() else 0.0
        pipeline_progress.log(
            f"[1/9] Audio extraction done in {elapsed:.1f}s"
            f" | wrote {size_mb:.1f} MB to {output}"
        )

    return output


def _run_ffmpeg_with_progress(
    proc: "subprocess.Popen[str]",
    *,
    pipeline_progress: "PipelineProgress | None",
    stage_label: str,
    handle: object,
) -> None:
    """Read ffmpeg -progress pipe:1 stdout and emit periodic progress logs."""
    if pipeline_progress is None or proc.stdout is None:
        # drain stdout silently so the process doesn't block
        if proc.stdout is not None:
            proc.stdout.read()
        return

    out_time_us: int = 0
    total_size: int = 0
    _last_log: float = _time.monotonic()
    _log_interval: float = 2.0  # real-time seconds between progress logs
    _last_audio_s: float = 0.0
    _log_audio_interval: float = 30.0  # audio seconds between progress logs

    for line in proc.stdout:
        line = line.strip()
        if line.startswith("out_time_us="):
            try:
                out_time_us = int(line.split("=", 1)[1])
            except ValueError:
                pass
        elif line.startswith("total_size="):
            try:
                total_size = int(line.split("=", 1)[1])
            except ValueError:
                pass

        now = _time.monotonic()
        audio_s = out_time_us / 1_000_000
        time_since_log = now - _last_log
        audio_since_log = audio_s - _last_audio_s

        if time_since_log >= _log_interval or audio_since_log >= _log_audio_interval:
            _last_log = now
            _last_audio_s = audio_s
            mm = int(audio_s // 60)
            ss = int(audio_s % 60)
            size_mb = total_size / 1024 / 1024
            pipeline_progress.log(
                f"{stage_label}: {mm}:{ss:02d} processed | {size_mb:.1f} MB"
            )



@contextmanager
def _null_stage():  # type: ignore[return]
    """No-op context manager used when no PipelineProgress is provided."""
    yield
