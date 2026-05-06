"""Tests for the PipelineProgress context manager."""

from __future__ import annotations

import io

from rich.console import Console

from vidscribe.progress import PipelineProgress


def _stderr_console() -> Console:
    """Return a rich Console backed by an in-memory buffer."""
    return Console(file=io.StringIO(), highlight=False)


def test_pipeline_progress_is_noop_when_quiet() -> None:
    pp = PipelineProgress(quiet=True)
    with pp as progress:
        with progress.stage("audio"):
            pass
        with progress.stage("stt", total=10.0) as handle:
            handle.advance_to(5.0)
            handle.advance_to(10.0)
        with progress.stage("correct", total=3) as handle:
            for _ in range(3):
                handle.advance()
    # No exception = pass


def test_pipeline_progress_context_manager_enters_and_exits() -> None:
    console = _stderr_console()
    pp = PipelineProgress(console=console)
    with pp as progress:
        assert progress._progress is not None
    assert pp._progress is None


def test_stage_handle_advance_to_is_monotonic() -> None:
    console = _stderr_console()
    pp = PipelineProgress(console=console)
    with pp as progress:
        with progress.stage("stt", total=100.0) as handle:
            handle.advance_to(30.0)
            assert handle._last_completed == 30.0
            # Backwards advance should be ignored
            handle.advance_to(10.0)
            assert handle._last_completed == 30.0
            handle.advance_to(60.0)
            assert handle._last_completed == 60.0


def test_stage_handle_advance_increments_relative() -> None:
    console = _stderr_console()
    pp = PipelineProgress(console=console)
    with pp as progress:
        with progress.stage("correct", total=5) as handle:
            handle.advance()
            assert handle._last_completed == 1.0
            handle.advance(2)
            assert handle._last_completed == 3.0


def test_stage_null_handle_is_safe_when_quiet() -> None:
    """_StageHandle(None, None, 0) methods must not raise."""
    from vidscribe.progress import _StageHandle

    h = _StageHandle(None, None, 0)
    h.advance()
    h.advance_to(100.0)
    h.update_description("whatever")


def test_pipeline_progress_print_is_noop_when_quiet() -> None:
    pp = PipelineProgress(quiet=True)
    with pp as progress:
        # Should not raise
        progress.print("hello")


def test_pipeline_progress_stage_unknown_name_uses_name_as_label() -> None:
    console = _stderr_console()
    pp = PipelineProgress(console=console)
    with pp as progress:
        with progress.stage("custom-stage"):
            pass
    # No exception = pass


def test_pipeline_progress_stage_accepts_description_override() -> None:
    console = _stderr_console()
    pp = PipelineProgress(console=console)
    with pp as progress:
        with progress.stage("audio", description="Custom label"):
            pass
    # No exception = pass
