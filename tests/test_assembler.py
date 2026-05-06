import pytest

from vidscribe.assembler import assemble
from vidscribe.pipeline import CorrectedChunk, CorrectedSegment


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
