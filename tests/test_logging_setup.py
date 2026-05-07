"""Tests for vidscribe.logging_setup."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from vidscribe.logging_setup import (
    TeeWriter,
    list_log_files,
    latest_log_path,
    make_log_path,
    open_log_file,
    update_latest_symlink,
)


# ---------------------------------------------------------------------------
# TeeWriter
# ---------------------------------------------------------------------------


def test_tee_writer_writes_to_all_streams() -> None:
    a = io.StringIO()
    b = io.StringIO()
    tee = TeeWriter(a, b)
    tee.write("hello")
    tee.flush()
    assert a.getvalue() == "hello"
    assert b.getvalue() == "hello"


def test_tee_writer_is_tty_mirrors_stderr() -> None:
    tee = TeeWriter(io.StringIO())
    assert isinstance(tee.isatty(), bool)


def test_tee_writer_survives_broken_stream() -> None:
    """A stream that raises on write must not propagate the exception."""

    class BrokenStream:
        def write(self, text: str) -> int:
            raise OSError("broken")

        def flush(self) -> None:
            raise OSError("broken")

    good = io.StringIO()
    tee = TeeWriter(BrokenStream(), good)
    tee.write("test")  # must not raise
    tee.flush()  # must not raise
    assert good.getvalue() == "test"


# ---------------------------------------------------------------------------
# make_log_path
# ---------------------------------------------------------------------------


def test_make_log_path_creates_directory(tmp_path: Path) -> None:
    log_path = make_log_path("pipeline", root=tmp_path)
    assert log_path.parent.exists()
    assert log_path.parent.name == "logs"
    assert log_path.suffix == ".log"
    assert "pipeline" in log_path.name


def test_make_log_path_includes_timestamp(tmp_path: Path) -> None:
    log_path = make_log_path("correct", root=tmp_path)
    # Timestamp pattern: YYYY-MM-DDTHH-MM-SS
    stem = log_path.stem
    assert stem.count("-") >= 5  # date + time separators


def test_make_log_path_uses_default_logs_dir_when_root_is_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Point VIDSCRIBE_CACHE_DIR to tmp_path so we don't pollute the real cache dir
    monkeypatch.setenv("VIDSCRIBE_CACHE_DIR", str(tmp_path))
    log_path = make_log_path("transcribe")
    assert log_path.is_absolute()
    assert log_path.parent.exists()
    assert "transcribe" in log_path.name
    # Should be inside tmp_path/logs/
    assert log_path.parent == tmp_path / "logs"


# ---------------------------------------------------------------------------
# update_latest_symlink
# ---------------------------------------------------------------------------


def test_update_latest_symlink_creates_symlink(tmp_path: Path) -> None:
    logs_dir = tmp_path / ".vidscribe" / "logs"
    logs_dir.mkdir(parents=True)
    log_file = logs_dir / "2026-05-07T00-00-00-pipeline.log"
    log_file.write_text("content", encoding="utf-8")

    update_latest_symlink(log_file)

    symlink = logs_dir / "latest.log"
    assert symlink.is_symlink()
    assert symlink.resolve() == log_file.resolve()


def test_update_latest_symlink_is_atomic_on_overwrite(tmp_path: Path) -> None:
    """Updating symlink twice should leave it pointing to the newest file."""
    logs_dir = tmp_path / ".vidscribe" / "logs"
    logs_dir.mkdir(parents=True)

    first = logs_dir / "2026-05-07T00-00-01-pipeline.log"
    first.write_text("first", encoding="utf-8")
    second = logs_dir / "2026-05-07T00-00-02-pipeline.log"
    second.write_text("second", encoding="utf-8")

    update_latest_symlink(first)
    update_latest_symlink(second)

    symlink = logs_dir / "latest.log"
    assert symlink.resolve() == second.resolve()


# ---------------------------------------------------------------------------
# open_log_file
# ---------------------------------------------------------------------------


def test_open_log_file_creates_file_and_symlink(tmp_path: Path) -> None:
    log_path = tmp_path / ".vidscribe" / "logs" / "2026-05-07T01-00-00-extract.log"
    handle = open_log_file(log_path)
    try:
        handle.write("line1\n")
    finally:
        handle.close()

    assert log_path.exists()
    assert log_path.read_text(encoding="utf-8") == "line1\n"
    symlink = log_path.parent / "latest.log"
    assert symlink.is_symlink()


# ---------------------------------------------------------------------------
# list_log_files
# ---------------------------------------------------------------------------


def test_list_log_files_returns_newest_first(tmp_path: Path) -> None:
    logs_dir = tmp_path / ".vidscribe" / "logs"
    logs_dir.mkdir(parents=True)
    names = [
        "2026-05-07T00-00-01-pipeline.log",
        "2026-05-07T00-00-03-correct.log",
        "2026-05-07T00-00-02-extract.log",
    ]
    for n in names:
        (logs_dir / n).write_text("x", encoding="utf-8")

    result = list_log_files(root=tmp_path)
    assert [f.name for f in result] == sorted(names, reverse=True)


def test_list_log_files_excludes_latest_symlink(tmp_path: Path) -> None:
    logs_dir = tmp_path / ".vidscribe" / "logs"
    logs_dir.mkdir(parents=True)
    real = logs_dir / "2026-05-07T00-00-01-pipeline.log"
    real.write_text("x", encoding="utf-8")
    (logs_dir / "latest.log").symlink_to(real.name)

    result = list_log_files(root=tmp_path)
    names = [f.name for f in result]
    assert "latest.log" not in names


def test_list_log_files_empty_when_no_logs_dir(tmp_path: Path) -> None:
    result = list_log_files(root=tmp_path)
    assert result == []


def test_list_log_files_respects_limit(tmp_path: Path) -> None:
    logs_dir = tmp_path / ".vidscribe" / "logs"
    logs_dir.mkdir(parents=True)
    for i in range(15):
        (logs_dir / f"2026-05-07T00-00-{i:02d}-pipeline.log").write_text("x", encoding="utf-8")

    result = list_log_files(root=tmp_path, limit=5)
    assert len(result) == 5


# ---------------------------------------------------------------------------
# latest_log_path
# ---------------------------------------------------------------------------


def test_latest_log_path_returns_none_when_empty(tmp_path: Path) -> None:
    result = latest_log_path(root=tmp_path)
    assert result is None


def test_latest_log_path_returns_symlink_target(tmp_path: Path) -> None:
    logs_dir = tmp_path / ".vidscribe" / "logs"
    logs_dir.mkdir(parents=True)
    real = logs_dir / "2026-05-07T00-00-01-pipeline.log"
    real.write_text("content", encoding="utf-8")
    (logs_dir / "latest.log").symlink_to(real.name)

    result = latest_log_path(root=tmp_path)
    assert result is not None
    assert result.name == real.name
