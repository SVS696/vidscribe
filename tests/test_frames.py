import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vidscribe.frames import FrameExtractionError, FrameInfo, extract


FFMPEG_LOG = """
[Parsed_metadata_1 @ 0x123] frame:0    pts:0       pts_time:0
[Parsed_metadata_1 @ 0x123] lavfi.scene_score=0.000000
[Parsed_metadata_1 @ 0x123] frame:1    pts:10240   pts_time:1.0
[Parsed_metadata_1 @ 0x123] lavfi.scene_score=0.420000
"""

SAMPLE_ONLY_LOG = """
[Parsed_showinfo @ 0x456] n:   0 pts:      0 pts_time:0.000000
[Parsed_showinfo @ 0x456] n:   1 pts:  10240 pts_time:1.000000
"""


def _make_mock_proc(returncode: int = 0, stderr: str = FFMPEG_LOG) -> MagicMock:
    """Build a mock Popen process for frames tests.

    poll() returns the returncode immediately so the _read_frames_progress
    while-loop exits right away (simulates instant process completion).
    """
    proc = MagicMock()
    proc.returncode = returncode
    proc.poll.return_value = returncode  # non-None → loop exits immediately
    proc.stdout.__iter__ = lambda self: iter([])
    proc.stdout.read.return_value = ""
    proc.stderr.read.return_value = stderr
    proc.wait.return_value = returncode
    return proc


def _make_stuck_proc() -> MagicMock:
    """Build a mock process that never advances out_time_us (stuck ffmpeg).

    poll() returns None so the read-loop keeps running, and select always
    times out (no data), triggering the watchdog.
    """
    proc = MagicMock()
    proc.returncode = None
    proc.poll.return_value = None  # process appears to still be running
    proc.stdout.read.return_value = ""
    proc.stderr.read.return_value = ""
    proc.wait.return_value = -9
    return proc


def test_extract_runs_ffmpeg_with_scene_and_sampling_filter(tmp_path, mocker) -> None:
    mock_proc = _make_mock_proc()
    popen = mocker.patch("vidscribe.frames.subprocess.Popen", return_value=mock_proc)
    video = tmp_path / "input.mp4"

    frames = extract(video, tmp_path / "frames", scene_threshold=0.3, sample_every=10.0)

    command = popen.call_args.args[0]
    assert command[:4] == ["ffmpeg", "-y", "-i", str(video)]
    assert command[4] == "-vf"
    assert "gt(scene,0.3)" in command[5]
    assert "gte(t-prev_selected_t,10.0)" in command[5]
    assert "metadata=print:key=lavfi.scene_score" in command[5]
    assert "-progress" in command
    assert "pipe:1" in command
    assert command[-1] == str(tmp_path / "frames" / "frame-%06d.jpg")
    assert frames == [
        FrameInfo(ts=0.0, path=tmp_path / "frames" / "frame-000001.jpg", scene_change=False),
        FrameInfo(ts=1.0, path=tmp_path / "frames" / "frame-000002.jpg", scene_change=True),
    ]


def test_extract_persists_frames_json(tmp_path, mocker) -> None:
    mocker.patch("vidscribe.frames.subprocess.Popen", return_value=_make_mock_proc())
    out_dir = tmp_path / "frames"

    extract(tmp_path / "input.mp4", out_dir)

    data = json.loads((out_dir / "frames.json").read_text(encoding="utf-8"))
    assert data == [
        {"ts": 0.0, "path": str(out_dir / "frame-000001.jpg"), "scene_change": False},
        {"ts": 1.0, "path": str(out_dir / "frame-000002.jpg"), "scene_change": True},
    ]


def test_extract_uses_existing_generated_image_paths(tmp_path, mocker) -> None:
    out_dir = tmp_path / "frames"
    out_dir.mkdir()
    generated = out_dir / "frame-000001.jpg"
    generated.write_bytes(b"jpg")
    stderr_log = (
        "[Parsed_metadata_1 @ 0x123] frame:0 pts:0 pts_time:2.5\n"
        "[Parsed_metadata_1 @ 0x123] lavfi.scene_score=0.100000\n"
    )
    mocker.patch(
        "vidscribe.frames.subprocess.Popen",
        return_value=_make_mock_proc(stderr=stderr_log),
    )

    frames = extract(tmp_path / "input.mp4", out_dir)

    assert frames[0].path == generated
    assert frames[0].ts == 2.5


def test_extract_raises_helpful_error_when_ffmpeg_is_missing(tmp_path, mocker) -> None:
    mocker.patch("vidscribe.frames.subprocess.Popen", side_effect=FileNotFoundError)

    with pytest.raises(FrameExtractionError, match="ffmpeg was not found"):
        extract(tmp_path / "input.mp4", tmp_path / "frames")


def test_extract_includes_ffmpeg_error_details(tmp_path, mocker) -> None:
    mock_proc = _make_mock_proc(returncode=1, stderr="Invalid data found when processing input")
    mocker.patch("vidscribe.frames.subprocess.Popen", return_value=mock_proc)

    with pytest.raises(FrameExtractionError, match="Invalid data"):
        extract(tmp_path / "broken.mp4", tmp_path / "frames")


def test_extract_tiny_fixture_video(fixtures_path: Path, tmp_path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not available")

    video = fixtures_path / "short.mp4"
    if not video.exists():
        pytest.skip("short.mp4 fixture is not available")

    frames = extract(video, tmp_path / "frames")

    assert frames
    assert (tmp_path / "frames" / "frames.json").exists()
    assert frames[0].path.exists()
    assert frames[0].ts == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Watchdog tests
# ---------------------------------------------------------------------------


def test_watchdog_kills_stuck_ffmpeg_and_raises(tmp_path, mocker) -> None:
    """If out_time_us never advances the watchdog must kill the process."""
    stuck_proc = _make_stuck_proc()
    mocker.patch("vidscribe.frames.subprocess.Popen", return_value=stuck_proc)
    mocker.patch("vidscribe.frames._probe_duration", return_value=0.0)
    # select always returns empty ready-list (timeout) — no data ever arrives
    mocker.patch("vidscribe.frames._select.select", return_value=([], [], []))

    with pytest.raises(FrameExtractionError, match="stuck"):
        # stuck_timeout=0 so the watchdog fires on the first idle iteration
        # frames_strategy="scene-detect" disables fallback so the error propagates
        extract(
            tmp_path / "input.mp4",
            tmp_path / "frames",
            stuck_timeout=0,
            frames_strategy="scene-detect",
        )

    stuck_proc.kill.assert_called_once()


def test_watchdog_timeout_configurable_via_kwarg(tmp_path, mocker) -> None:
    """stuck_timeout kwarg overrides the module-level STUCK_TIMEOUT constant."""
    stuck_proc = _make_stuck_proc()
    mocker.patch("vidscribe.frames.subprocess.Popen", return_value=stuck_proc)
    mocker.patch("vidscribe.frames._probe_duration", return_value=0.0)
    mocker.patch("vidscribe.frames._select.select", return_value=([], [], []))

    # Very small timeout — must still raise FrameExtractionError, not hang
    with pytest.raises(FrameExtractionError, match="stuck"):
        extract(
            tmp_path / "input.mp4",
            tmp_path / "frames",
            stuck_timeout=0,
            frames_strategy="scene-detect",
        )


def test_fallback_to_sample_only_when_scene_detect_fails(tmp_path, mocker) -> None:
    """On FrameExtractionError from scene-detect, auto strategy retries sample-only."""
    # First call (scene-detect): mock a failing process
    failing_proc = _make_mock_proc(returncode=1, stderr="filter error")
    # Second call (sample-only): mock a successful process
    success_proc = _make_mock_proc(returncode=0, stderr=SAMPLE_ONLY_LOG)

    popen = mocker.patch(
        "vidscribe.frames.subprocess.Popen",
        side_effect=[failing_proc, success_proc],
    )
    mocker.patch("vidscribe.frames._probe_duration", return_value=0.0)

    # frames_strategy="auto" → should fall back on first failure
    extract(
        tmp_path / "input.mp4",
        tmp_path / "frames",
        frames_strategy="auto",
    )

    assert popen.call_count == 2
    # First call must include scene-detect filter (command[4]=="-vf", command[5]==filter)
    first_cmd = popen.call_args_list[0].args[0]
    vf_idx = first_cmd.index("-vf") + 1
    assert "gt(scene," in first_cmd[vf_idx]
    # Second call must NOT include scene-detect filter
    second_cmd = popen.call_args_list[1].args[0]
    vf_idx2 = second_cmd.index("-vf") + 1
    assert "gt(scene," not in second_cmd[vf_idx2]
    assert "gte(t-prev_selected_t," in second_cmd[vf_idx2]


def test_scene_detect_only_strategy_does_not_fallback(tmp_path, mocker) -> None:
    """frames_strategy='scene-detect' must propagate FrameExtractionError, no fallback."""
    failing_proc = _make_mock_proc(returncode=1, stderr="scene-detect error")
    popen = mocker.patch(
        "vidscribe.frames.subprocess.Popen",
        return_value=failing_proc,
    )
    mocker.patch("vidscribe.frames._probe_duration", return_value=0.0)

    with pytest.raises(FrameExtractionError):
        extract(
            tmp_path / "input.mp4",
            tmp_path / "frames",
            frames_strategy="scene-detect",
        )

    # Must only have tried once (no fallback)
    assert popen.call_count == 1


def test_sample_only_strategy_skips_scene_detect(tmp_path, mocker) -> None:
    """frames_strategy='sample-only' must use simple filter without gt(scene,...)."""
    success_proc = _make_mock_proc(returncode=0, stderr=SAMPLE_ONLY_LOG)
    popen = mocker.patch("vidscribe.frames.subprocess.Popen", return_value=success_proc)
    mocker.patch("vidscribe.frames._probe_duration", return_value=0.0)

    extract(
        tmp_path / "input.mp4",
        tmp_path / "frames",
        frames_strategy="sample-only",
    )

    assert popen.call_count == 1
    cmd = popen.call_args.args[0]
    vf_idx = cmd.index("-vf") + 1
    assert "gt(scene," not in cmd[vf_idx]
    assert "gte(t-prev_selected_t," in cmd[vf_idx]
