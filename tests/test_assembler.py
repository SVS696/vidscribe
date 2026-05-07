import pytest

from vidscribe.assembler import assemble
from vidscribe.pipeline import CorrectedChunk, CorrectedSegment, ScreenEvent


def corrected(
    idx: int,
    start: float,
    end: float,
    speaker: str | None,
    text: str,
) -> CorrectedChunk:
    return CorrectedChunk(
        idx=idx,
        start=start,
        end=end,
        speaker=speaker,
        corrected_text=text,
    )


def test_assemble_markdown_merges_adjacent_same_speaker_chunks() -> None:
    transcript = assemble(
        [
            corrected(0, 0, 4, "SPEAKER_00", "Первая реплика."),
            corrected(1, 4, 8, "SPEAKER_00", "Продолжение."),
            corrected(2, 70, 75, "SPEAKER_01", "Ответ."),
        ],
        {"SPEAKER_00": "Иван", "SPEAKER_01": "Алиса"},
    )

    assert transcript == (
        "## [00:00:00] **Иван**\n\n"
        "Первая реплика.\n\n"
        "Продолжение.\n\n"
        "## [00:01:10] **Алиса**\n\n"
        "Ответ.\n"
    )


def test_assemble_markdown_keeps_non_adjacent_same_speaker_separate() -> None:
    transcript = assemble(
        [
            corrected(0, 0, 4, "SPEAKER_00", "Начало."),
            corrected(1, 4, 8, "SPEAKER_01", "Вставка."),
            corrected(2, 8, 12, "SPEAKER_00", "Возврат."),
        ],
        {"SPEAKER_00": "Иван", "SPEAKER_01": "Алиса"},
    )

    assert transcript.count("**Иван**") == 2
    assert transcript.count("## [") == 3


def test_assemble_markdown_preserves_corrected_segment_speakers() -> None:
    transcript = assemble(
        [
            CorrectedChunk(
                idx=0,
                start=0,
                end=8,
                speaker="SPEAKER_00",
                corrected_text="Иван говорит.\nАлиса отвечает.",
                segments=[
                    CorrectedSegment(
                        start=0,
                        end=4,
                        speaker="SPEAKER_00",
                        corrected_text="Иван говорит.",
                    ),
                    CorrectedSegment(
                        start=4,
                        end=8,
                        speaker="SPEAKER_01",
                        corrected_text="Алиса отвечает.",
                    ),
                ],
            )
        ],
        {"SPEAKER_00": "Иван", "SPEAKER_01": "Алиса"},
    )

    assert transcript == (
        "## [00:00:00] **Иван**\n\n"
        "Иван говорит.\n\n"
        "## [00:00:04] **Алиса**\n\n"
        "Алиса отвечает.\n"
    )


def test_assemble_markdown_uses_fallback_speaker_names_and_skips_empty_text() -> None:
    transcript = assemble(
        [
            corrected(0, 0, 2, "SPEAKER_99", "  Raw id  "),
            corrected(1, 2, 4, None, ""),
            corrected(2, 4, 6, None, "Unknown speaker"),
        ],
        {},
    )

    assert transcript == (
        "## [00:00:00] **SPEAKER_99**\n\n"
        "Raw id\n\n"
        "## [00:00:04] **Unknown**\n\n"
        "Unknown speaker\n"
    )


def test_assemble_srt_renders_timestamped_entries_without_merging() -> None:
    transcript = assemble(
        [
            corrected(1, 65.432, 70.9, "SPEAKER_00", "Второй."),
            corrected(0, 0.1, 4.25, "SPEAKER_00", "Первый."),
        ],
        {"SPEAKER_00": "Иван"},
        fmt="srt",
    )

    assert transcript == (
        "1\n"
        "00:00:00,100 --> 00:00:04,250\n"
        "Иван: Первый.\n\n"
        "2\n"
        "00:01:05,432 --> 00:01:10,900\n"
        "Иван: Второй.\n"
    )


def test_assemble_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match="Unsupported output format"):
        assemble([], {}, fmt="txt")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Screen context tests
# ---------------------------------------------------------------------------

def _chunk_with_events(
    idx: int,
    start: float,
    end: float,
    speaker: str,
    text: str,
    events: list[tuple[float, str]] | None = None,
) -> CorrectedChunk:
    return CorrectedChunk(
        idx=idx,
        start=start,
        end=end,
        speaker=speaker,
        corrected_text=text,
        screen_events=[ScreenEvent(ts=ts, description=desc) for ts, desc in (events or [])],
    )


def test_screen_context_off_produces_no_event_markup() -> None:
    transcript = assemble(
        [
            _chunk_with_events(0, 0, 5, "SPEAKER_00", "Первая реплика.", [(2.5, "Tab switched")]),
        ],
        {"SPEAKER_00": "Иван"},
        screen_context_mode="off",
    )

    assert "📺" not in transcript
    assert "scene" not in transcript
    assert "Сцены" not in transcript
    assert "Первая реплика." in transcript


def test_screen_context_inline_inserts_blockquote_before_reply() -> None:
    transcript = assemble(
        [
            _chunk_with_events(0, 83, 90, "SPEAKER_00", "Показываю таблицу.", [(83.0, "Switched to 'Data' tab")]),
        ],
        {"SPEAKER_00": "Иван"},
        screen_context_mode="inline",
    )

    assert "> 📺 [00:01:23] Switched to 'Data' tab" in transcript
    assert "Показываю таблицу." in transcript
    # blockquote must appear before the reply text
    blockquote_pos = transcript.index("> 📺")
    reply_pos = transcript.index("Показываю таблицу.")
    assert blockquote_pos < reply_pos


def test_screen_context_inline_no_events_produces_clean_output() -> None:
    transcript = assemble(
        [
            _chunk_with_events(0, 0, 5, "SPEAKER_00", "Чистая реплика.", []),
        ],
        {"SPEAKER_00": "Иван"},
        screen_context_mode="inline",
    )

    assert "📺" not in transcript
    assert "Чистая реплика." in transcript


def test_screen_context_aside_creates_scenes_section() -> None:
    transcript = assemble(
        [
            _chunk_with_events(0, 0, 10, "SPEAKER_00", "Первый оратор.", [(5.0, "Opened Excel")]),
            _chunk_with_events(1, 10, 20, "SPEAKER_01", "Второй оратор.", [(15.0, "Selected row 5")]),
        ],
        {"SPEAKER_00": "Иван", "SPEAKER_01": "Алиса"},
        screen_context_mode="aside",
    )

    assert "## Сцены" in transcript
    assert "## Транскрипт" in transcript
    assert "- [00:00:05] Opened Excel" in transcript
    assert "- [00:00:15] Selected row 5" in transcript
    # Scenes section must appear before transcript section
    scenes_pos = transcript.index("## Сцены")
    transcript_pos = transcript.index("## Транскрипт")
    assert scenes_pos < transcript_pos
    # Both speakers should appear in transcript
    assert "Иван" in transcript
    assert "Алиса" in transcript


def test_screen_context_aside_no_events_omits_scenes_section() -> None:
    transcript = assemble(
        [
            _chunk_with_events(0, 0, 5, "SPEAKER_00", "Без событий.", []),
        ],
        {"SPEAKER_00": "Иван"},
        screen_context_mode="aside",
    )

    assert "## Сцены" not in transcript
    assert "Без событий." in transcript


def test_screen_context_footer_appends_event_after_reply() -> None:
    transcript = assemble(
        [
            _chunk_with_events(0, 65, 70, "SPEAKER_00", "Выделяю ячейку.", [(65.0, "Selected cell B3")]),
        ],
        {"SPEAKER_00": "Иван"},
        screen_context_mode="footer",
    )

    assert "*[scene 00:01:05: Selected cell B3]*" in transcript
    assert "Выделяю ячейку." in transcript
    # footer must appear after the reply text
    reply_pos = transcript.index("Выделяю ячейку.")
    footer_pos = transcript.index("*[scene")
    assert reply_pos < footer_pos


# ---------------------------------------------------------------------------
# A: spacing normalisation tests
# ---------------------------------------------------------------------------

def test_assemble_collapses_triple_newlines_in_corrected_text() -> None:
    """Triple+ newlines inside a segment's corrected_text are collapsed to \\n\\n."""
    import re

    chunk = corrected(0, 0, 10, "SPEAKER_00", "line1\n\n\nline2\n\n\n\nline3")
    result = assemble([chunk], {"SPEAKER_00": "Иван"})
    assert not re.search(r"\n{3,}", result), "Expected no triple+ newlines"
    assert "line1\n\nline2\n\nline3" in result


def test_assemble_collapses_triple_newlines_across_turns() -> None:
    """Joining two turns that each end with \\n\\n must not produce triple newlines."""
    import re

    # corrected_text with internal double-newline + join via \\n\\n
    c0 = corrected(0, 0, 5, "SPEAKER_00", "para1\n\npara2")
    c1 = corrected(1, 5, 10, "SPEAKER_00", "para3\n\npara4")
    result = assemble([c0, c1], {"SPEAKER_00": "Иван"})
    assert not re.search(r"\n{3,}", result), "Expected no triple+ newlines"


# ---------------------------------------------------------------------------
# B: screen_event ts clamp tests
# ---------------------------------------------------------------------------

def test_parse_screen_events_clamps_ts_to_chunk_window() -> None:
    from vidscribe.pipeline import _parse_screen_events

    # ts slightly outside: clamp expected
    events = _parse_screen_events(
        {"screen_events": [{"ts": -2.0, "description": "early"}, {"ts": 15.0, "description": "late"}]},
        chunk_start=0.0,
        chunk_end=10.0,
    )
    assert len(events) == 2
    assert events[0].ts == 0.0   # clamped from -2
    assert events[1].ts == 10.0  # clamped from 15


def test_parse_screen_events_drops_ts_far_outside_window() -> None:
    from vidscribe.pipeline import _parse_screen_events

    # ts more than 5s outside → dropped
    events = _parse_screen_events(
        {"screen_events": [{"ts": -10.0, "description": "too early"}, {"ts": 100.0, "description": "too late"}]},
        chunk_start=0.0,
        chunk_end=10.0,
    )
    assert events == []


def test_parse_screen_events_keeps_ts_within_window() -> None:
    from vidscribe.pipeline import _parse_screen_events

    events = _parse_screen_events(
        {"screen_events": [{"ts": 5.0, "description": "normal"}]},
        chunk_start=0.0,
        chunk_end=10.0,
    )
    assert len(events) == 1
    assert events[0].ts == 5.0


# ---------------------------------------------------------------------------
# C: speaker_map cross-lookup tests
# ---------------------------------------------------------------------------

def test_speaker_name_maps_speaker_xx_key_to_name() -> None:
    from vidscribe.assembler import _speaker_name

    assert _speaker_name("SPEAKER_01", {"SPEAKER_01": "Алексей"}) == "Алексей"


def test_speaker_name_cross_lookup_sxx_id_against_speaker_xx_key() -> None:
    from vidscribe.assembler import _speaker_name

    # speaker_id uses sXX, map uses SPEAKER_XX
    assert _speaker_name("s01", {"SPEAKER_01": "Алексей"}) == "Алексей"


def test_speaker_name_cross_lookup_speaker_xx_id_against_sxx_key() -> None:
    from vidscribe.assembler import _speaker_name

    # speaker_id uses SPEAKER_XX, map uses sXX
    assert _speaker_name("SPEAKER_01", {"s01": "Алексей"}) == "Алексей"


def test_assemble_markdown_uses_speaker_map_name_in_header() -> None:
    """When speaker_map contains sXX keys, header should show the mapped name."""
    # segments use SPEAKER_XX (raw STT), map uses sXX keys
    chunk = CorrectedChunk(
        idx=0,
        start=0,
        end=5,
        speaker="SPEAKER_00",
        corrected_text="Привет.",
        segments=[
            CorrectedSegment(start=0, end=5, speaker="SPEAKER_00", corrected_text="Привет."),
        ],
    )
    result = assemble([chunk], {"s00": "Иван"})
    assert "**Иван**" in result
    assert "**SPEAKER_00**" not in result
    assert "**s00**" not in result
