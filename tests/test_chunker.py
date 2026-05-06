from pathlib import Path

import pytest

from vidscribe.chunker import Chunk, chunk
from vidscribe.frames import FrameInfo
from vidscribe.stt import SttResult, SttSegment


def segment(
    start: float,
    end: float,
    text: str,
    speaker: str | None = None,
) -> SttSegment:
    return SttSegment(start=start, end=end, text=text, speaker=speaker, words=[])


def frame(ts: float, name: str, scene_change: bool = False) -> FrameInfo:
    return FrameInfo(ts=ts, path=Path(f"/frames/{name}.jpg"), scene_change=scene_change)


def result(*segments: SttSegment) -> SttResult:
    return SttResult(segments=list(segments), language="ru", model="large-v3")


def test_speaker_strategy_splits_on_speaker_turns_and_selects_frames() -> None:
    stt = result(
        segment(0, 10, "hello", "SPEAKER_00"),
        segment(10, 20, "still speaker zero", "SPEAKER_00"),
        segment(20, 30, "speaker one", "SPEAKER_01"),
    )
    frames = [
        frame(1, "first"),
        frame(15, "inside-zero"),
        frame(24, "inside-one"),
        frame(40, "outside"),
    ]

    chunks = chunk(stt, frames, strategy="speaker", window_s=180)

    assert chunks == [
        Chunk(
            idx=0,
            start=0,
            end=20,
            segments=stt.segments[:2],
            frame_paths=[Path("/frames/first.jpg"), Path("/frames/inside-zero.jpg")],
            surrounding_context="After: speaker one",
        ),
        Chunk(
            idx=1,
            start=20,
            end=30,
            segments=stt.segments[2:],
            frame_paths=[Path("/frames/inside-one.jpg")],
            surrounding_context="Before: still speaker zero",
        ),
    ]


def test_time_strategy_uses_fixed_windows_from_first_segment() -> None:
    stt = result(
        segment(5, 12, "a", "SPEAKER_00"),
        segment(60, 72, "b", "SPEAKER_00"),
        segment(126, 130, "c", "SPEAKER_01"),
    )

    chunks = chunk(stt, [], strategy="time", window_s=60)

    assert [chunk.segments for chunk in chunks] == [
        stt.segments[:2],
        stt.segments[2:],
    ]
    assert [(chunk.start, chunk.end) for chunk in chunks] == [(5, 72), (126, 130)]
    assert chunks[0].surrounding_context == "After: c"
    assert chunks[1].surrounding_context == "Before: b"


def test_time_strategy_keeps_speaker_turns_together_within_window() -> None:
    stt = result(
        segment(0, 10, "a", "SPEAKER_00"),
        segment(10, 20, "b", "SPEAKER_01"),
    )

    chunks = chunk(stt, [], strategy="time", window_s=60)

    assert [chunk.segments for chunk in chunks] == [stt.segments]


def test_scene_strategy_splits_at_scene_change_frames() -> None:
    stt = result(
        segment(0, 9, "opening", "SPEAKER_00"),
        segment(11, 19, "new scene", "SPEAKER_01"),
        segment(22, 30, "same scene", "SPEAKER_01"),
    )
    frames = [
        frame(0, "start"),
        frame(10, "scene-a", scene_change=True),
        frame(20, "not-boundary"),
        frame(21, "scene-b", scene_change=True),
    ]

    chunks = chunk(stt, frames, strategy="scene", window_s=180)

    assert [chunk.segments for chunk in chunks] == [
        stt.segments[:1],
        stt.segments[1:2],
        stt.segments[2:],
    ]
    assert chunks[1].frame_paths == [Path("/frames/scene-a.jpg")]
    assert chunks[1].surrounding_context == "Before: opening\nAfter: same scene"


def test_scene_strategy_keeps_speaker_turns_together_within_scene() -> None:
    stt = result(
        segment(0, 9, "opening", "SPEAKER_00"),
        segment(11, 19, "reply", "SPEAKER_01"),
    )

    chunks = chunk(stt, [], strategy="scene", window_s=180)

    assert [chunk.segments for chunk in chunks] == [stt.segments]


def test_strategies_apply_window_limit() -> None:
    stt = result(
        segment(0, 50, "a", "SPEAKER_00"),
        segment(50, 120, "b", "SPEAKER_00"),
    )

    chunks = chunk(stt, [], strategy="speaker", window_s=100)

    assert [chunk.segments for chunk in chunks] == [[stt.segments[0]], [stt.segments[1]]]


def test_chunk_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError, match="window_s"):
        chunk(result(segment(0, 1, "a")), [], strategy="time", window_s=0)
