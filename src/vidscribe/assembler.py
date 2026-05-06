"""Final transcript assembly."""

from __future__ import annotations

from typing import Literal, Mapping

from vidscribe.pipeline import CorrectedChunk


OutputFormat = Literal["md", "srt"]


def assemble(
    corrected: list[CorrectedChunk],
    speakers: Mapping[str, str],
    fmt: OutputFormat = "md",
) -> str:
    """Render corrected chunks into a final transcript."""

    chunks = sorted(corrected, key=lambda chunk: (chunk.start, chunk.idx))
    if fmt == "md":
        return _assemble_markdown(chunks, speakers)
    if fmt == "srt":
        return _assemble_srt(chunks, speakers)
    raise ValueError(f"Unsupported output format: {fmt}")


def _assemble_markdown(
    chunks: list[CorrectedChunk],
    speakers: Mapping[str, str],
) -> str:
    blocks: list[_MergedBlock] = []
    for turn in _assembly_turns(chunks):
        text = turn.corrected_text.strip()
        if not text:
            continue

        speaker = _speaker_name(turn.speaker, speakers)
        if blocks and blocks[-1].speaker == speaker:
            blocks[-1].end = max(blocks[-1].end, turn.end)
            blocks[-1].texts.append(text)
            continue

        blocks.append(
            _MergedBlock(
                start=turn.start,
                end=turn.end,
                speaker=speaker,
                texts=[text],
            )
        )

    if not blocks:
        return ""

    rendered = []
    for block in blocks:
        text = "\n\n".join(block.texts)
        rendered.append(
            f"## [{_format_markdown_time(block.start)}] **{block.speaker}**\n\n"
            f"{text}"
        )
    return "\n\n".join(rendered) + "\n"


def _assemble_srt(
    chunks: list[CorrectedChunk],
    speakers: Mapping[str, str],
) -> str:
    entries: list[str] = []
    for turn in _assembly_turns(chunks):
        text = turn.corrected_text.strip()
        if not text:
            continue
        speaker = _speaker_name(turn.speaker, speakers)
        entries.append(
            "\n".join(
                [
                    str(len(entries) + 1),
                    f"{_format_srt_time(turn.start)} --> {_format_srt_time(turn.end)}",
                    f"{speaker}: {text}",
                ]
            )
        )
    return "\n\n".join(entries) + ("\n" if entries else "")


def _assembly_turns(chunks: list[CorrectedChunk]) -> list["_AssemblyTurn"]:
    turns: list[_AssemblyTurn] = []
    for chunk in chunks:
        if chunk.segments:
            turns.extend(
                _AssemblyTurn(
                    start=segment.start,
                    end=segment.end,
                    speaker=segment.speaker,
                    corrected_text=segment.corrected_text,
                )
                for segment in chunk.segments
            )
            continue
        turns.append(
            _AssemblyTurn(
                start=chunk.start,
                end=chunk.end,
                speaker=chunk.speaker,
                corrected_text=chunk.corrected_text,
            )
        )
    return sorted(turns, key=lambda turn: (turn.start, turn.end))


class _AssemblyTurn:
    def __init__(
        self,
        start: float,
        end: float,
        speaker: str | None,
        corrected_text: str,
    ) -> None:
        self.start = start
        self.end = end
        self.speaker = speaker
        self.corrected_text = corrected_text


class _MergedBlock:
    def __init__(
        self,
        *,
        start: float,
        end: float,
        speaker: str,
        texts: list[str],
    ) -> None:
        self.start = start
        self.end = end
        self.speaker = speaker
        self.texts = texts


def _speaker_name(speaker_id: str | None, speakers: Mapping[str, str]) -> str:
    if speaker_id and speaker_id in speakers:
        return speakers[speaker_id]
    if speaker_id:
        return speaker_id
    return "Unknown"


def _format_markdown_time(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_srt_time(seconds: float) -> str:
    milliseconds_total = max(0, int(round(seconds * 1000)))
    seconds_total, milliseconds = divmod(milliseconds_total, 1000)
    hours, remainder = divmod(seconds_total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
