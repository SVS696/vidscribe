"""Final transcript assembly."""

from __future__ import annotations

import time as _time
from typing import TYPE_CHECKING, Literal, Mapping

from vidscribe.pipeline import CorrectedChunk, ScreenEvent

if TYPE_CHECKING:
    from vidscribe.progress import PipelineProgress


OutputFormat = Literal["md", "srt"]
ScreenContextMode = Literal["off", "inline", "aside", "footer"]


def assemble(
    corrected: list[CorrectedChunk],
    speakers: Mapping[str, str],
    fmt: OutputFormat = "md",
    screen_context_mode: ScreenContextMode = "off",
    *,
    pipeline_progress: "PipelineProgress | None" = None,
) -> str:
    """Render corrected chunks into a final transcript."""

    if pipeline_progress is not None:
        pipeline_progress.log(f"[9/9] Final assembly: {fmt} format, screen_context={screen_context_mode}")
    t0 = _time.monotonic()

    chunks = sorted(corrected, key=lambda chunk: (chunk.start, chunk.idx))
    if fmt == "md":
        result = _assemble_markdown(chunks, speakers, screen_context_mode=screen_context_mode)
    elif fmt == "srt":
        result = _assemble_srt(chunks, speakers)
    else:
        raise ValueError(f"Unsupported output format: {fmt}")

    if pipeline_progress is not None:
        elapsed = _time.monotonic() - t0
        n_turns = len([t for t in _assembly_turns(chunks) if t.corrected_text.strip()])
        size_kb = len(result.encode()) / 1024
        pipeline_progress.log(
            f"[9/9] Assembly done in {elapsed:.2f}s | {n_turns} turns, {size_kb:.1f} KB"
        )

    return result


def _assemble_markdown(
    chunks: list[CorrectedChunk],
    speakers: Mapping[str, str],
    *,
    screen_context_mode: ScreenContextMode = "off",
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
            blocks[-1].turn_screen_events.append(turn.screen_events)
            continue

        blocks.append(
            _MergedBlock(
                start=turn.start,
                end=turn.end,
                speaker=speaker,
                texts=[text],
                turn_screen_events=[turn.screen_events],
            )
        )

    if not blocks:
        return ""

    rendered = []

    if screen_context_mode == "aside":
        # Collect all events from all chunks, sorted by ts
        all_events: list[ScreenEvent] = []
        for chunk in chunks:
            all_events.extend(chunk.screen_events)
        all_events.sort(key=lambda e: e.ts)

        if all_events:
            scene_lines = ["## Сцены", ""]
            for event in all_events:
                scene_lines.append(f"- [{_format_markdown_time(event.ts)}] {event.description}")
            scene_lines.extend(["", "## Транскрипт"])
            rendered.append("\n".join(scene_lines))

    for block in blocks:
        if screen_context_mode == "inline":
            # Collect all screen events for this block's turns
            block_events: list[ScreenEvent] = []
            for turn_events in block.turn_screen_events:
                block_events.extend(turn_events)
            block_events.sort(key=lambda e: e.ts)

            prefix_lines: list[str] = []
            for event in block_events:
                prefix_lines.append(f"> 📺 [{_format_markdown_time(event.ts)}] {event.description}")

            header = (
                f"## [{_format_markdown_time(block.start)}] **{block.speaker}**\n\n"
            )
            if prefix_lines:
                prefix = "\n".join(prefix_lines) + "\n\n"
            else:
                prefix = ""
            text = "\n\n".join(block.texts)
            rendered.append(f"{header}{prefix}{text}")

        elif screen_context_mode == "footer":
            header = (
                f"## [{_format_markdown_time(block.start)}] **{block.speaker}**\n\n"
            )
            text_parts: list[str] = []
            for turn_text, turn_events in zip(block.texts, block.turn_screen_events):
                turn_events_sorted = sorted(turn_events, key=lambda e: e.ts)
                footer_lines: list[str] = []
                for event in turn_events_sorted:
                    footer_lines.append(
                        f"*[scene {_format_markdown_time(event.ts)}: {event.description}]*"
                    )
                if footer_lines:
                    text_parts.append(turn_text + "\n" + "\n".join(footer_lines))
                else:
                    text_parts.append(turn_text)
            text = "\n\n".join(text_parts)
            rendered.append(f"{header}{text}")

        else:
            # "off" or "aside" — normal rendering
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
            # Attach screen_events to the first segment of the chunk
            for seg_idx, segment in enumerate(chunk.segments):
                turns.append(
                    _AssemblyTurn(
                        start=segment.start,
                        end=segment.end,
                        speaker=segment.speaker,
                        corrected_text=segment.corrected_text,
                        screen_events=chunk.screen_events if seg_idx == 0 else [],
                    )
                )
            continue
        turns.append(
            _AssemblyTurn(
                start=chunk.start,
                end=chunk.end,
                speaker=chunk.speaker,
                corrected_text=chunk.corrected_text,
                screen_events=chunk.screen_events,
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
        screen_events: list[ScreenEvent] | None = None,
    ) -> None:
        self.start = start
        self.end = end
        self.speaker = speaker
        self.corrected_text = corrected_text
        self.screen_events: list[ScreenEvent] = screen_events or []


class _MergedBlock:
    def __init__(
        self,
        *,
        start: float,
        end: float,
        speaker: str,
        texts: list[str],
        turn_screen_events: list[list[ScreenEvent]] | None = None,
    ) -> None:
        self.start = start
        self.end = end
        self.speaker = speaker
        self.texts = texts
        self.turn_screen_events: list[list[ScreenEvent]] = turn_screen_events or []


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
