"""Tests for vidscribe.paths — platform-aware default path resolution."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from vidscribe.paths import default_cache_dir, default_logs_dir


# ---------------------------------------------------------------------------
# VIDSCRIBE_CACHE_DIR env override (highest priority, platform-independent)
# ---------------------------------------------------------------------------


def test_env_override_takes_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIDSCRIBE_CACHE_DIR", "/custom/cache")
    result = default_cache_dir()
    assert result == Path("/custom/cache")


def test_env_override_expands_tilde(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIDSCRIBE_CACHE_DIR", "~/vidcache")
    result = default_cache_dir()
    assert result == Path.home() / "vidcache"


def test_env_override_not_set_falls_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VIDSCRIBE_CACHE_DIR", raising=False)
    # Should not raise and should return some absolute-ish path
    result = default_cache_dir()
    assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# macOS platform
# ---------------------------------------------------------------------------


def test_macos_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VIDSCRIBE_CACHE_DIR", raising=False)
    with patch.object(sys, "platform", "darwin"):
        result = default_cache_dir()
    assert result == Path.home() / "Library" / "Caches" / "vidscribe"


# ---------------------------------------------------------------------------
# Linux platform
# ---------------------------------------------------------------------------


def test_linux_default_no_xdg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VIDSCRIBE_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    with patch.object(sys, "platform", "linux"):
        result = default_cache_dir()
    assert result == Path.home() / ".cache" / "vidscribe"


def test_linux_with_xdg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VIDSCRIBE_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", "/xdg/cache")
    with patch.object(sys, "platform", "linux"):
        result = default_cache_dir()
    assert result == Path("/xdg/cache") / "vidscribe"


# ---------------------------------------------------------------------------
# Windows platform
# ---------------------------------------------------------------------------


def test_windows_default_with_localappdata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VIDSCRIBE_CACHE_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", "C:\\Users\\user\\AppData\\Local")
    with patch.object(sys, "platform", "win32"):
        result = default_cache_dir()
    assert result == Path("C:\\Users\\user\\AppData\\Local") / "vidscribe" / "cache"


def test_windows_fallback_no_localappdata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VIDSCRIBE_CACHE_DIR", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    with patch.object(sys, "platform", "win32"):
        result = default_cache_dir()
    assert result == Path.home() / "AppData" / "Local" / "vidscribe" / "cache"


# ---------------------------------------------------------------------------
# Unknown/other platform
# ---------------------------------------------------------------------------


def test_unknown_platform_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VIDSCRIBE_CACHE_DIR", raising=False)
    with patch.object(sys, "platform", "freebsd13"):
        result = default_cache_dir()
    assert result == Path.home() / ".cache" / "vidscribe"


# ---------------------------------------------------------------------------
# default_logs_dir
# ---------------------------------------------------------------------------


def test_logs_dir_is_under_cache_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIDSCRIBE_CACHE_DIR", "/my/cache")
    logs = default_logs_dir()
    assert logs == Path("/my/cache") / "logs"


def test_logs_dir_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VIDSCRIBE_CACHE_DIR", raising=False)
    with patch.object(sys, "platform", "darwin"):
        logs = default_logs_dir()
    assert logs == Path.home() / "Library" / "Caches" / "vidscribe" / "logs"
