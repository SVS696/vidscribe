"""Rich-based pipeline progress reporting.

All output goes to stderr so that stdout (provider JSON) is never polluted.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Generator, TextIO

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

if TYPE_CHECKING:
    pass

_STDERR_CONSOLE = Console(stderr=True)

_STAGE_LABELS = {
    "audio": "[1/9] Extracting audio",
    "stt": "[2/9] Transcribing",
    "diar": "[3/9] Diarizing",
    "merge": "[4/9] Merging ASR + diarization",
    "frames": "[5/9] Extracting frames",
    "chunks": "[6/9] Chunking",
    "speakers": "[7/9] Identifying speakers",
    "correct": "[8/9] Correcting chunks",
    "assembly": "[9/9] Assembling transcript",
}


class PipelineProgress:
    """Single rich.Progress instance that spans all pipeline stages.

    Usage::

        with PipelineProgress() as pp:
            with pp.stage("audio"):
                audio.extract(...)
            with pp.stage("stt", total=duration) as task:
                for seg in segments:
                    task.advance(seg.end)
    """

    def __init__(
        self,
        *,
        quiet: bool = False,
        console: Console | None = None,
        log_file: Path | None = None,
    ) -> None:
        self._quiet = quiet
        self._log_handle: TextIO | None = None

        if log_file is not None:
            from vidscribe.logging_setup import TeeWriter, open_log_file

            self._log_handle = open_log_file(log_file)
            tee = TeeWriter(sys.stderr, self._log_handle)
            self._console = Console(file=tee, highlight=False)
        else:
            self._console = console or _STDERR_CONSOLE

        self._progress: Progress | None = None
        self._stage_tasks: dict[str, TaskID] = {}

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "PipelineProgress":
        if not self._quiet:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=self._console,
                transient=False,
            )
            self._progress.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        if self._progress is not None:
            self._progress.__exit__(*args)
            self._progress = None
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except Exception:  # noqa: BLE001
                pass
            self._log_handle = None

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        total: float | int | None = None,
        description: str | None = None,
    ) -> Generator["_StageHandle", None, None]:
        """Context manager for a single pipeline stage.

        Parameters
        ----------
        name:
            Short stage key (``"audio"``, ``"stt"``, …).
        total:
            Known total units.  ``None`` → indeterminate / spinner.
        description:
            Override the default stage label.
        """
        label = description or _STAGE_LABELS.get(name, name)
        t0 = time.monotonic()

        if self._quiet or self._progress is None:
            yield _StageHandle(None, None, 0)
            return

        task_id = self._progress.add_task(label, total=total)
        handle = _StageHandle(self._progress, task_id, t0)
        try:
            yield handle
        finally:
            elapsed = time.monotonic() - t0
            # Mark complete and update description with elapsed time
            self._progress.update(
                task_id,
                completed=total if total is not None else 1,
                total=total if total is not None else 1,
                description=f"{label} [dim]({elapsed:.1f}s)[/dim]",
            )

    def log(self, message: str) -> None:
        """Write a timestamped log line that stays above the progress bars.

        Always writes to the backing console (and therefore to the log file
        when one is configured), even when *quiet* is True.  This ensures that
        long-running stages produce live output in ``vidscribe logs --follow``
        regardless of the terminal quiet flag.
        """
        if self._progress is not None:
            self._progress.console.log(message)
        else:
            self._console.log(message)

    def print(self, message: str) -> None:
        """Print a message that stays above the progress bars."""
        if self._quiet:
            return
        if self._progress is not None:
            self._progress.console.print(message)
        else:
            self._console.print(message)


class _StageHandle:
    """Handle returned by :meth:`PipelineProgress.stage` to update progress."""

    def __init__(
        self,
        progress: Progress | None,
        task_id: TaskID | None,
        t0: float,
    ) -> None:
        self._progress = progress
        self._task_id = task_id
        self._t0 = t0
        self._last_completed: float = 0.0

    def advance_to(self, completed: float) -> None:
        """Set absolute completed amount (e.g. seconds elapsed in audio)."""
        if self._progress is None or self._task_id is None:
            return
        delta = completed - self._last_completed
        if delta > 0:
            self._progress.advance(self._task_id, delta)
            self._last_completed = completed

    def advance(self, delta: float = 1) -> None:
        """Advance by a relative delta (e.g. one chunk done)."""
        if self._progress is None or self._task_id is None:
            return
        self._progress.advance(self._task_id, delta)
        self._last_completed += delta

    def update_description(self, description: str) -> None:
        """Change the task description mid-run."""
        if self._progress is None or self._task_id is None:
            return
        self._progress.update(self._task_id, description=description)
