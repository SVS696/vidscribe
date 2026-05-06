"""Audio extraction helpers."""

from __future__ import annotations

import subprocess
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
        str(output),
    ]

    ctx = pipeline_progress.stage("audio") if pipeline_progress is not None else _null_stage()

    with ctx:
        try:
            subprocess.run(command, capture_output=True, text=True, check=True)
        except FileNotFoundError as exc:
            raise AudioExtractionError(
                "ffmpeg was not found. Install ffmpeg and make sure it is on PATH."
            ) from exc
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or exc.stdout or "").strip()
            message = f"ffmpeg failed to extract audio from {video}"
            if details:
                message = f"{message}: {details}"
            raise AudioExtractionError(message) from exc

    return output



@contextmanager
def _null_stage():  # type: ignore[return]
    """No-op context manager used when no PipelineProgress is provided."""
    yield
