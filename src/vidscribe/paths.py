"""Platform-aware default paths for vidscribe user data."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def default_cache_dir() -> Path:
    """Return the platform-specific default cache directory.

    Resolution order:
    1. ``VIDSCRIBE_CACHE_DIR`` environment variable
    2. Platform default:
       - macOS: ``~/Library/Caches/vidscribe``
       - Linux: ``${XDG_CACHE_HOME:-~/.cache}/vidscribe``
       - Windows: ``%LOCALAPPDATA%/vidscribe/cache``
       - Other: ``~/.cache/vidscribe``
    """
    if env := os.environ.get("VIDSCRIBE_CACHE_DIR"):
        return Path(env).expanduser()

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "vidscribe"

    if sys.platform.startswith("linux"):
        xdg = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
        return Path(xdg) / "vidscribe"

    if sys.platform.startswith("win"):
        local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(local) / "vidscribe" / "cache"

    return Path.home() / ".cache" / "vidscribe"


def default_logs_dir() -> Path:
    """Return the default logs directory (inside the cache directory)."""
    return default_cache_dir() / "logs"
