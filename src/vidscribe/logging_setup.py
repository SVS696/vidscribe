"""Auto-logging support for vidscribe CLI commands.

Each command invocation creates a timestamped log file under
`<cwd>/.vidscribe/logs/`.  A ``latest.log`` symlink is kept pointing to
the most recent log file.

All rich console output (progress bars, info messages) is tee'd into the log
file so that ``vidscribe logs --follow`` gives a live view in a second terminal.
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO


# ---------------------------------------------------------------------------
# TeeWriter — writes to multiple streams simultaneously
# ---------------------------------------------------------------------------


class TeeWriter:
    """File-like object that mirrors writes to multiple streams.

    Designed to back a rich ``Console`` so that all output goes both to
    stderr (for the user in the primary terminal) and to a log file (so a
    second terminal can tail it).
    """

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        n = 0
        for stream in self.streams:
            try:
                n = stream.write(text)
                stream.flush()
            except Exception:  # noqa: BLE001 – never crash the pipeline
                pass
        return n

    def flush(self) -> None:
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:  # noqa: BLE001
                pass

    # rich uses ``isatty`` to decide whether to emit ANSI codes
    def isatty(self) -> bool:
        return sys.stderr.isatty()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _logs_dir(root: Path | None = None) -> Path:
    """Return `.vidscribe/logs/` relative to *root* (default: cwd)."""
    base = root if root is not None else Path.cwd()
    return base / ".vidscribe" / "logs"


def make_log_path(command_name: str, *, root: Path | None = None) -> Path:
    """Generate a timestamped log file path for *command_name*.

    Example: ``.vidscribe/logs/2026-05-07T03-12-45-pipeline.log``

    The parent directory is created if it does not exist.
    """
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    logs = _logs_dir(root)
    logs.mkdir(parents=True, exist_ok=True)
    return logs / f"{ts}-{command_name}.log"


def update_latest_symlink(log_path: Path) -> None:
    """Atomically update ``.vidscribe/logs/latest.log`` → *log_path*.

    Uses a temp-file + ``os.replace`` trick so the symlink is never in a
    broken intermediate state.
    """
    symlink_target = log_path.parent / "latest.log"
    # Create a temp symlink in the same directory then rename it
    fd, tmp_path_str = tempfile.mkstemp(dir=log_path.parent, prefix=".tmp-latest-")
    os.close(fd)
    tmp_path = Path(tmp_path_str)
    try:
        tmp_path.unlink()  # remove the regular file mkstemp created
        tmp_path.symlink_to(log_path.name)  # relative symlink (same dir)
        os.replace(str(tmp_path), str(symlink_target))
    except Exception:  # noqa: BLE001
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass


def open_log_file(log_path: Path) -> TextIO:
    """Open *log_path* for appending, update ``latest.log`` symlink, return handle."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(log_path, "a", encoding="utf-8")  # noqa: SIM115 – intentional
    update_latest_symlink(log_path)
    return handle


# ---------------------------------------------------------------------------
# Listing helpers (used by `vidscribe logs --list`)
# ---------------------------------------------------------------------------


def list_log_files(root: Path | None = None, limit: int = 10) -> list[Path]:
    """Return up to *limit* most-recent log files (newest first)."""
    logs = _logs_dir(root)
    if not logs.exists():
        return []
    files = sorted(
        (p for p in logs.iterdir() if p.is_file() and p.suffix == ".log" and p.name != "latest.log"),
        reverse=True,
    )
    return files[:limit]


def latest_log_path(root: Path | None = None) -> Path | None:
    """Return the path that ``latest.log`` symlink points to, or None."""
    symlink = _logs_dir(root) / "latest.log"
    if symlink.exists() or symlink.is_symlink():
        # resolve to absolute path
        target = symlink.resolve()
        if target.exists():
            return target
        # symlink exists but target is gone — return symlink itself for display
        return symlink
    # Fall back: newest file
    files = list_log_files(root, limit=1)
    return files[0] if files else None
