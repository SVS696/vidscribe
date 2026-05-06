import json
import shutil
import subprocess
from pathlib import Path

import pytest

from vidscribe.frames import FrameExtractionError, FrameInfo, extract


FFMPEG_LOG = """
[Parsed_metadata_1 @ 0x123] frame:0    pts:0       pts_time:0
[Parsed_metadata_1 @ 0x123] lavfi.scene_score=0.000000
[Parsed_metadata_1 @ 0x123] frame:1    pts:10240   pts_time:1.0
[Parsed_metadata_1 @ 0x123] lavfi.scene_score=0.420000
"""


def test_extract_runs_ffmpeg_with_scene_and_sampling_filter(tmp_path, mocker) -> None:
    run = mocker.patch(
        "vidscribe.frames.subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["ffmpeg"],
            returncode=0,
            stdout="",
            stderr=FFMPEG_LOG,
        ),
    )
    video = tmp_path / "input.mp4"

    frames = extract(video, tmp_path / "frames", scene_threshold=0.3, sample_every=10.0)

    command = run.call_args.args[0]
    assert command[:4] == ["ffmpeg", "-y", "-i", str(video)]
    assert command[4] == "-vf"
    assert "gt(scene,0.3)" in command[5]
    assert "gte(t-prev_selected_t,10.0)" in command[5]
    assert "metadata=print:key=lavfi.scene_score" in command[5]
    assert command[-2:] == ["vfr", str(tmp_path / "frames" / "frame-%06d.jpg")]
    assert frames == [
        FrameInfo(ts=0.0, path=tmp_path / "frames" / "frame-000001.jpg", scene_change=False),
        FrameInfo(ts=1.0, path=tmp_path / "frames" / "frame-000002.jpg", scene_change=True),
    ]


def test_extract_persists_frames_json(tmp_path, mocker) -> None:
    mocker.patch(
        "vidscribe.frames.subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["ffmpeg"],
            returncode=0,
            stdout="",
            stderr=FFMPEG_LOG,
        ),
    )
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
    mocker.patch(
        "vidscribe.frames.subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["ffmpeg"],
            returncode=0,
            stdout="",
            stderr="[Parsed_metadata_1 @ 0x123] frame:0 pts:0 pts_time:2.5\n"
            "[Parsed_metadata_1 @ 0x123] lavfi.scene_score=0.100000\n",
        ),
    )

    frames = extract(tmp_path / "input.mp4", out_dir)

    assert frames[0].path == generated
    assert frames[0].ts == 2.5


def test_extract_raises_helpful_error_when_ffmpeg_is_missing(tmp_path, mocker) -> None:
    mocker.patch("vidscribe.frames.subprocess.run", side_effect=FileNotFoundError)

    with pytest.raises(FrameExtractionError, match="ffmpeg was not found"):
        extract(tmp_path / "input.mp4", tmp_path / "frames")


def test_extract_includes_ffmpeg_error_details(tmp_path, mocker) -> None:
    error = subprocess.CalledProcessError(
        returncode=1,
        cmd=["ffmpeg"],
        stderr="Invalid data found when processing input",
    )
    mocker.patch("vidscribe.frames.subprocess.run", side_effect=error)

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
