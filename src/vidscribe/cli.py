"""Command-line interface for vidscribe."""

from __future__ import annotations

import json
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

if TYPE_CHECKING:
    from vidscribe.progress import PipelineProgress

import typer
from pydantic import ValidationError
from rich.console import Console

from vidscribe import assembler, audio, chunker, frames, provider, speakers, stt
from vidscribe.cache import Cache
from vidscribe.config import AppConfig, CacheStage, ChunkStrategy, CorrectionMode, FramesStrategy, ScreenContextMode, load_config
from vidscribe.frames import FrameInfo
from vidscribe.logging_setup import (
    list_log_files,
    latest_log_path,
    make_log_path,
)
from vidscribe.pipeline import correct_chunks
from vidscribe.progress import PipelineProgress
from vidscribe.stt import SttResult

app = typer.Typer(
    help="Local video transcription with CLI-provider correction.",
    invoke_without_command=True,
)
cache_app = typer.Typer(help="Manage cached pipeline artefacts.")
app.add_typer(cache_app, name="cache")

_ALL_STAGES = {
    "audio",
    "frames",
    "asr",
    "diar",
    "stt",
    "chunks",
    "speakers",
    "corrected",
    "final",
}
_CORRECT_RECOMPUTE_STAGES = {"chunks", "speakers", "corrected", "final"}
console = Console()


@app.callback()
def main(
    ctx: typer.Context,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="CLI provider: claude, codex, or ollama."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Provider model name."),
    ] = None,
    chunk_strategy: Annotated[
        ChunkStrategy | None,
        typer.Option("--chunk-strategy", help="Chunking strategy."),
    ] = None,
    frame_rate: Annotated[
        float | None,
        typer.Option("--frame-rate", min=0.001, help="Frame sampling rate."),
    ] = None,
    whisper_model: Annotated[
        str | None,
        typer.Option("--whisper-model", help="Whisper model or noScribe alias."),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option("--language", help="Transcription language."),
    ] = None,
    hf_token: Annotated[
        str | None,
        typer.Option("--hf-token", help="Hugging Face token for fallback assets."),
    ] = None,
    cache_dir: Annotated[
        Path | None,
        typer.Option("--cache-dir", help="Cache directory."),
    ] = None,
    no_cache: Annotated[
        list[str] | None,
        typer.Option(
            "--no-cache",
            help="Bypass cache for a stage. Repeat for multiple stages.",
        ),
    ] = None,
    speakers: Annotated[
        str | None,
        typer.Option(
            "--speakers",
            help="Comma-separated speaker names, positionally mapped by speaker index.",
        ),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", help="Suppress progress output (for CI/scripts)."),
    ] = False,
    no_log: Annotated[
        bool,
        typer.Option("--no-log", help="Disable automatic log file creation."),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Override log file path."),
    ] = None,
) -> None:
    """Run vidscribe commands."""

    try:
        config = load_config(
            overrides={
                "provider": provider,
                "model": model,
                "chunk_strategy": chunk_strategy,
                "frame_rate": frame_rate,
                "whisper_model": whisper_model,
                "language": language,
                "hf_token": hf_token,
                "cache_dir": cache_dir,
                "no_cache": tuple(no_cache) if no_cache else None,
                "speakers": _parse_speakers(speakers) if speakers is not None else None,
            }
        )
    except ValidationError as exc:
        raise typer.BadParameter(_validation_message(exc)) from exc

    ctx.obj = {"config": config, "quiet": quiet, "no_log": no_log, "log_file": log_file}


def config_from_context(ctx: typer.Context) -> AppConfig:
    """Return callback-loaded config for subcommands."""

    return ctx.obj["config"]


def quiet_from_context(ctx: typer.Context) -> bool:
    """Return the --quiet flag from the root callback."""

    return bool(ctx.obj.get("quiet", False))


def log_file_from_context(ctx: typer.Context, command_name: str) -> Path | None:
    """Resolve the effective log file path for a command invocation.

    Returns ``None`` when ``--no-log`` is set or ctx.obj is not yet populated
    (e.g. during ``--help``).
    """
    obj = ctx.obj
    if not isinstance(obj, dict):
        return None
    if obj.get("no_log"):
        return None
    override: Path | None = obj.get("log_file")
    if override is not None:
        return override
    # Logs live alongside the cache; make_log_path(root=None) resolves via
    # default_logs_dir() which honours VIDSCRIBE_CACHE_DIR and platform defaults.
    return make_log_path(command_name, root=None)


@app.command("pipeline")
def pipeline_command(
    ctx: typer.Context,
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    provider_name: Annotated[
        str | None,
        typer.Option("--provider", help="CLI provider: claude, codex, or ollama."),
    ] = None,
    model: Annotated[str | None, typer.Option("--model", help="Provider model.")] = None,
    whisper_model: Annotated[
        str | None,
        typer.Option("--whisper-model", help="Whisper model or noScribe alias."),
    ] = None,
    chunk_strategy: Annotated[
        ChunkStrategy | None,
        typer.Option("--chunk-strategy", help="Chunking strategy."),
    ] = None,
    speaker_names: Annotated[
        str | None,
        typer.Option(
            "--speakers",
            help="Comma-separated speaker names, positionally mapped by speaker index.",
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Output transcript path."),
    ] = None,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Bypass cache for this run."),
    ] = False,
    correction_mode: Annotated[
        CorrectionMode | None,
        typer.Option("--correction-mode", help="Correction mode: single or mix."),
    ] = None,
    text_provider: Annotated[
        str | None,
        typer.Option("--text-provider", help="Mix-mode: provider for text-only Pass 1."),
    ] = None,
    text_model: Annotated[
        str | None,
        typer.Option("--text-model", help="Mix-mode: model for text-only Pass 1."),
    ] = None,
    visual_provider: Annotated[
        str | None,
        typer.Option("--visual-provider", help="Mix-mode: provider for visual Pass 2."),
    ] = None,
    visual_model: Annotated[
        str | None,
        typer.Option("--visual-model", help="Mix-mode: model for visual Pass 2."),
    ] = None,
    screen_context: Annotated[
        ScreenContextMode | None,
        typer.Option("--screen-context", help="Screen context mode: off, inline, aside, or footer."),
    ] = None,
    frames_strategy: Annotated[
        FramesStrategy | None,
        typer.Option(
            "--frames-strategy",
            help="Frame extraction strategy: auto (scene-detect with sample-only fallback), scene-detect, or sample-only.",
        ),
    ] = None,
) -> None:
    """Run the full transcription and correction pipeline."""

    config = _command_config(
        ctx,
        provider=provider_name,
        model=model,
        whisper_model=whisper_model,
        chunk_strategy=chunk_strategy,
        speakers=_parse_speakers(speaker_names) if speaker_names is not None else None,
        correction_mode=correction_mode,
        text_provider=text_provider,
        text_model=text_model,
        visual_provider=visual_provider,
        visual_model=visual_model,
        screen_context_mode=screen_context,
        frames_strategy=frames_strategy,
    )
    cache = _cache(config, no_cache=no_cache)
    video_key = cache.key_for("video", video=video)
    quiet = quiet_from_context(ctx)
    log_path = log_file_from_context(ctx, "pipeline")
    with PipelineProgress(quiet=quiet, log_file=log_path) as pp:
        audio_path, frame_items = _extract(video, config, cache, video_key, pipeline_progress=pp)
        stt_result = _transcribe_audio(audio_path, config, cache, video_key, pipeline_progress=pp)
        chunk_items = _chunks(stt_result, frame_items, config, cache, video_key, pipeline_progress=pp)
        cli_provider = _make_provider(config)
        speaker_map = speakers.identify(
            stt_result,
            frame_items,
            cli_provider,
            manual=config.speakers,
            cache=cache,
            namespace_key=video_key,
            pipeline_progress=pp,
        )
        corrected = correct_chunks(
            chunk_items,
            cli_provider,
            speaker_map,
            cache,
            namespace_key=video_key,
            visual_provider=_make_visual_provider(config) if config.correction_mode == "mix" else None,
            pipeline_progress=pp,
            screen_context_mode=config.screen_context_mode,
        )
        output_path = out or video.with_suffix(".md")
        transcript = assembler.assemble(
            corrected,
            speaker_map,
            screen_context_mode=config.screen_context_mode,
            pipeline_progress=pp,
            output_path=output_path,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(transcript, encoding="utf-8")
    cache.set("final", video_key, transcript)
    console.print(str(output_path))


@app.command("extract")
def extract_command(
    ctx: typer.Context,
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Bypass cache for this run."),
    ] = False,
    frames_strategy: Annotated[
        FramesStrategy | None,
        typer.Option(
            "--frames-strategy",
            help="Frame extraction strategy: auto (scene-detect with sample-only fallback), scene-detect, or sample-only.",
        ),
    ] = None,
) -> None:
    """Extract audio and keyframes without LLM calls."""

    config = _command_config(ctx, frames_strategy=frames_strategy)
    cache = _cache(config, no_cache=no_cache)
    video_key = cache.key_for("video", video=video)
    quiet = quiet_from_context(ctx)
    log_path = log_file_from_context(ctx, "extract")
    with PipelineProgress(quiet=quiet, log_file=log_path) as pp:
        audio_path, frame_items = _extract(video, config, cache, video_key, pipeline_progress=pp)
    console.print(f"audio: {audio_path}")
    console.print(f"frames: {len(frame_items)}")


@app.command("transcribe")
def transcribe_command(
    ctx: typer.Context,
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    whisper_model: Annotated[
        str | None,
        typer.Option("--whisper-model", help="Whisper model or noScribe alias."),
    ] = None,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Bypass cache for this run."),
    ] = False,
) -> None:
    """Run audio extraction and STT only."""

    config = _command_config(ctx, whisper_model=whisper_model)
    cache = _cache(config, no_cache=no_cache)
    video_key = cache.key_for("video", video=video)
    quiet = quiet_from_context(ctx)
    log_path = log_file_from_context(ctx, "transcribe")
    with PipelineProgress(quiet=quiet, log_file=log_path) as pp:
        audio_path = _audio(video, cache, video_key, pipeline_progress=pp)
        stt_result = _transcribe_audio(audio_path, config, cache, video_key, pipeline_progress=pp)
    console.print(f"segments: {len(stt_result.segments)}")


@app.command("correct")
def correct_command(
    ctx: typer.Context,
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    provider_name: Annotated[
        str | None,
        typer.Option("--provider", help="CLI provider: claude, codex, or ollama."),
    ] = None,
    model: Annotated[str | None, typer.Option("--model", help="Provider model.")] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Output transcript path."),
    ] = None,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Bypass cache for this run."),
    ] = False,
    correction_mode: Annotated[
        CorrectionMode | None,
        typer.Option("--correction-mode", help="Correction mode: single or mix."),
    ] = None,
    text_provider: Annotated[
        str | None,
        typer.Option("--text-provider", help="Mix-mode: provider for text-only Pass 1."),
    ] = None,
    text_model: Annotated[
        str | None,
        typer.Option("--text-model", help="Mix-mode: model for text-only Pass 1."),
    ] = None,
    visual_provider: Annotated[
        str | None,
        typer.Option("--visual-provider", help="Mix-mode: provider for visual Pass 2."),
    ] = None,
    visual_model: Annotated[
        str | None,
        typer.Option("--visual-model", help="Mix-mode: model for visual Pass 2."),
    ] = None,
    screen_context: Annotated[
        ScreenContextMode | None,
        typer.Option("--screen-context", help="Screen context mode: off, inline, aside, or footer."),
    ] = None,
) -> None:
    """Re-run correction from cached STT and frames."""

    config = _command_config(
        ctx,
        provider=provider_name,
        model=model,
        correction_mode=correction_mode,
        text_provider=text_provider,
        text_model=text_model,
        visual_provider=visual_provider,
        visual_model=visual_model,
        screen_context_mode=screen_context,
    )
    if no_cache:
        disabled = set(config.no_cache) | _CORRECT_RECOMPUTE_STAGES
        cache = Cache(config.cache_dir, disabled_stages=disabled, console=console)
    else:
        cache = _cache(config)
    video_key = cache.key_for("video", video=video)
    stt_result = _cached_model(cache, "stt", video_key, SttResult)
    frame_items = _cached_frames(cache, video_key)
    quiet = quiet_from_context(ctx)
    log_path = log_file_from_context(ctx, "correct")
    with PipelineProgress(quiet=quiet, log_file=log_path) as pp:
        chunk_items = _chunks(stt_result, frame_items, config, cache, video_key, pipeline_progress=pp)
        cli_provider = _make_provider(config)
        speaker_map = speakers.identify(
            stt_result,
            frame_items,
            cli_provider,
            manual=config.speakers,
            cache=cache,
            namespace_key=video_key,
            pipeline_progress=pp,
        )
        corrected = correct_chunks(
            chunk_items,
            cli_provider,
            speaker_map,
            cache,
            namespace_key=video_key,
            visual_provider=_make_visual_provider(config) if config.correction_mode == "mix" else None,
            pipeline_progress=pp,
            screen_context_mode=config.screen_context_mode,
        )
        output_path = out or video.with_suffix(".md")
        transcript = assembler.assemble(
            corrected,
            speaker_map,
            screen_context_mode=config.screen_context_mode,
            pipeline_progress=pp,
            output_path=output_path,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(transcript, encoding="utf-8")
    cache.set("final", video_key, transcript)
    console.print(str(output_path))


@cache_app.command("list")
def cache_list(
    ctx: typer.Context,
    video: Annotated[
        Path | None,
        typer.Argument(exists=True, dir_okay=False, help="Optional video to inspect."),
    ] = None,
) -> None:
    """List cached artefact keys or stages for one video."""

    config = config_from_context(ctx)
    cache = _cache(config)
    root = cache.root / "cache"
    if video is not None:
        video_key = cache.key_for("video", video=video)
        video_root = root / video_key
        if not video_root.exists():
            return
        for stage in sorted(path.name for path in video_root.iterdir() if path.is_dir()):
            console.print(stage)
        return

    if not root.exists():
        return
    for key_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        console.print(key_dir.name)


@cache_app.command("clear")
def cache_clear(
    ctx: typer.Context,
    video: Annotated[
        Path | None,
        typer.Argument(exists=True, dir_okay=False, help="Optional video to clear."),
    ] = None,
) -> None:
    """Clear all cached artefacts or the artefacts for one video."""

    config = config_from_context(ctx)
    cache = _cache(config)
    root = cache.root / "cache"
    target = root / cache.key_for("video", video=video) if video is not None else root
    if target.exists():
        shutil.rmtree(target)
    console.print(str(target))


@app.command("logs")
def logs_command(
    follow: Annotated[
        bool,
        typer.Option("--follow", "-f", help="tail -f the latest log file."),
    ] = False,
    list_logs: Annotated[
        bool,
        typer.Option("--list", "-l", help="List last 10 log files."),
    ] = False,
    path: Annotated[
        bool,
        typer.Option("--path", help="Print absolute path to latest.log (default)."),
    ] = False,
) -> None:
    """Show, list, or follow vidscribe run logs.

    By default prints the path to the latest log file.

    Examples:

        vidscribe logs                  # path to latest.log

        vidscribe logs --follow         # tail -f latest.log (Ctrl+C to stop)

        vidscribe logs --list           # last 10 runs
    """

    log_path = latest_log_path()

    if list_logs:
        files = list_log_files(limit=10)
        if not files:
            console.print("No log files found in .vidscribe/logs/")
            return
        for f in files:
            # Extract timestamp and command from filename e.g.
            # 2026-05-07T03-12-45-pipeline.log → 2026-05-07 03:12:45  pipeline
            stem = f.stem  # e.g. "2026-05-07T03-12-45-pipeline"
            parts = stem.split("-", maxsplit=6)
            # parts: ['2026', '05', '07T03', '12', '45', 'pipeline']
            try:
                ts_raw = "-".join(parts[:5])  # "2026-05-07T03-12-45" approx
                ts_display = ts_raw.replace("T", " ").replace("-", ":", 2)
                # Actually reconstruct: date=parts[0]-parts[1]-parts[2 split on T]
                date_part, time_prefix = parts[2].split("T")
                ts_display = f"{parts[0]}-{parts[1]}-{date_part} {time_prefix}:{parts[3]}:{parts[4]}"
                cmd_part = parts[5] if len(parts) > 5 else "?"
            except (IndexError, ValueError):
                ts_display = stem
                cmd_part = ""
            console.print(f"{ts_display}  {cmd_part:<12}  {f}")
        return

    if follow:
        if log_path is None:
            typer.echo("No log files found in vidscribe logs directory", err=True)
            raise typer.Exit(1)
        typer.echo(f"Following: {log_path}", err=True)
        try:
            subprocess.run(["tail", "-f", str(log_path)], check=False)
        except KeyboardInterrupt:
            pass
        return

    # Default / --path: print path
    if log_path is None:
        console.print("No log files found in .vidscribe/logs/")
        raise typer.Exit(1)
    console.print(str(log_path))


def _parse_speakers(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _command_config(ctx: typer.Context, **overrides: Any) -> AppConfig:
    config = config_from_context(ctx)
    compact = {key: value for key, value in overrides.items() if value is not None}
    if not compact:
        return config
    try:
        return AppConfig.model_validate(config.model_dump() | compact)
    except ValidationError as exc:
        raise typer.BadParameter(_validation_message(exc)) from exc


def _make_provider(config: AppConfig) -> provider.Provider:
    """Create the primary (or mix-mode text) provider from config."""
    if config.correction_mode == "mix":
        # In mix-mode: text_provider/text_model override provider/model for Pass 1
        pname = config.text_provider if config.text_provider is not None else config.provider
        pmodel = config.text_model if config.text_model is not None else config.model
    else:
        pname = config.provider
        pmodel = config.model
    opts = {"model": pmodel} if pmodel is not None else {}
    return provider.make(pname, **opts)


def _make_visual_provider(config: AppConfig) -> provider.Provider:
    """Create the mix-mode visual provider (Pass 2) from config."""
    opts = {"model": config.visual_model}
    return provider.make(config.visual_provider, **opts)


def _validation_message(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return str(exc)
    error = errors[0]
    field = ".".join(str(part) for part in error.get("loc", ())) or "config"
    return f"Invalid {field}: {error.get('msg', 'invalid value')}"


def _cache(config: AppConfig, *, no_cache: bool = False) -> Cache:
    disabled: set[CacheStage] = set(config.no_cache)
    if no_cache:
        disabled.update(_ALL_STAGES)
    return Cache(config.cache_dir, disabled_stages=disabled, console=console)


def _extract(
    video: Path,
    config: AppConfig,
    cache: Cache,
    video_key: str,
    *,
    pipeline_progress: "PipelineProgress | None" = None,
) -> tuple[Path, list[FrameInfo]]:
    return (
        _audio(video, cache, video_key, pipeline_progress=pipeline_progress),
        _frames(video, config, cache, video_key, pipeline_progress=pipeline_progress),
    )


def _audio(
    video: Path,
    cache: Cache,
    video_key: str,
    *,
    pipeline_progress: "PipelineProgress | None" = None,
) -> Path:
    stage_dir = _stage_dir(cache, video_key, "audio")
    cached = _cached_file(cache, "audio", video_key)
    if cached is not None:
        return cached
    output = stage_dir / "audio.wav"
    return audio.extract(video, output, pipeline_progress=pipeline_progress)


def _frames(
    video: Path,
    config: AppConfig,
    cache: Cache,
    video_key: str,
    *,
    pipeline_progress: "PipelineProgress | None" = None,
) -> list[FrameInfo]:
    output_dir = _stage_dir(cache, video_key, "frames")
    frames_json = output_dir / "frames.json"
    metadata_path = output_dir / "metadata.json"
    metadata = _frames_metadata(config)
    if "frames" not in cache.disabled_stages and frames_json.exists():
        cached_metadata = _read_json(metadata_path)
        if cached_metadata == metadata:
            cache.console.log(f"cache hit: frames/{video_key}")
            return [
                _absolute_frame_info(FrameInfo.model_validate(item))
                for item in json.loads(frames_json.read_text(encoding="utf-8"))
            ]

    frame_items = frames.extract(
        video,
        output_dir,
        sample_every=1 / config.frame_rate,
        pipeline_progress=pipeline_progress,
        frames_strategy=config.frames_strategy,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return [_absolute_frame_info(frame) for frame in frame_items]


def _transcribe_audio(
    audio_path: Path,
    config: AppConfig,
    cache: Cache,
    video_key: str,
    *,
    pipeline_progress: "PipelineProgress | None" = None,
) -> SttResult:
    assets = stt.detect_assets()
    metadata = _stt_metadata(config, assets)
    metadata_path = _stage_dir(cache, video_key, "stt") / "metadata.json"
    cached = cache.get("stt", video_key)
    dependencies_disabled = bool({"audio", "asr", "diar", "stt"} & cache.disabled_stages)
    if (
        not dependencies_disabled
        and isinstance(cached, dict)
        and _read_json(metadata_path) == metadata
    ):
        return SttResult.model_validate(cached)

    asr = stt.transcribe(
        audio_path,
        model=config.whisper_model,
        language=config.language,
        pipeline_progress=pipeline_progress,
    )
    diar = stt.diarize(
        audio_path,
        assets,
        hf_token=config.hf_token,
        pipeline_progress=pipeline_progress,
    )
    import time as _time

    if pipeline_progress is not None:
        pipeline_progress.log(
            f"[5/9] Merging ASR + diarization | {len(asr.segments)} segments"
        )
    t0_merge = _time.monotonic()
    merge_ctx = (
        pipeline_progress.stage("merge")
        if pipeline_progress is not None
        else _null_pp_stage()
    )
    with merge_ctx:
        result = stt.merge_asr_diar(asr, diar)
    if pipeline_progress is not None:
        elapsed_merge = _time.monotonic() - t0_merge
        n_turns = len({seg.speaker for seg in result.segments if seg.speaker})
        pipeline_progress.log(
            f"[5/9] Merge done in {elapsed_merge:.2f}s | {len(result.segments)} segments, {n_turns} speaker turns"
        )
    cache.set("asr", video_key, asr)
    cache.set("diar", video_key, diar)
    cache.set("stt", video_key, result)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return result


def _chunks(
    stt_result: SttResult,
    frame_items: list[FrameInfo],
    config: AppConfig,
    cache: Cache,
    video_key: str,
    *,
    pipeline_progress: "PipelineProgress | None" = None,
) -> list[chunker.Chunk]:
    cache_key = cache.key_for(
        "chunks",
        video=video_key,
        strategy=config.chunk_strategy,
        stt=stt_result,
        frames=frame_items,
    )
    cache_key = f"{video_key}/{cache_key}"
    cached = cache.get("chunks", cache_key)
    if isinstance(cached, list):
        return [chunker.Chunk.model_validate(item) for item in cached]

    chunk_items = chunker.chunk(
        stt_result,
        frame_items,
        config.chunk_strategy,
        pipeline_progress=pipeline_progress,
    )
    cache.set("chunks", cache_key, chunk_items)
    return chunk_items


def _cached_model(
    cache: Cache,
    stage: str,
    key: str,
    model_class: type[SttResult],
) -> SttResult:
    cached = cache.get(stage, key)
    if not isinstance(cached, dict):
        raise typer.BadParameter(f"Missing cached {stage} artefact for this video.")
    return model_class.model_validate(cached)


def _cached_frames(cache: Cache, video_key: str) -> list[FrameInfo]:
    frames_json = cache.root / "cache" / video_key / "frames" / "frames.json"
    if not frames_json.exists():
        raise typer.BadParameter("Missing cached frames artefact for this video.")
    return [
        _absolute_frame_info(FrameInfo.model_validate(item))
        for item in json.loads(frames_json.read_text(encoding="utf-8"))
    ]


def _absolute_frame_info(frame: FrameInfo) -> FrameInfo:
    return frame.model_copy(update={"path": frame.path.resolve()})


def _frames_metadata(config: AppConfig) -> dict[str, Any]:
    return {
        "frame_rate": config.frame_rate,
        "sample_every": 1 / config.frame_rate,
        "scene_threshold": 0.3,
        "frames_strategy": config.frames_strategy,
    }


def _stt_metadata(config: AppConfig, assets: stt.AssetPaths | None) -> dict[str, Any]:
    return {
        "whisper_model": config.whisper_model,
        "language": config.language,
        "hf_token_present": bool(config.hf_token),
        "diarization_source": "noscribe" if assets is not None else "huggingface",
        "assets_root": str(assets.resources_dir) if assets is not None else None,
    }


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _cached_file(cache: Cache, stage: str, key: str) -> Path | None:
    if stage in cache.disabled_stages:
        return None
    stage_dir = _stage_dir(cache, key, stage)
    if not stage_dir.exists():
        return None
    files = sorted(path for path in stage_dir.iterdir() if path.is_file())
    if not files:
        return None
    cache.console.log(f"cache hit: {stage}/{key}")
    return files[0]


def _stage_dir(cache: Cache, key: str, stage: str) -> Path:
    path = cache.root / "cache" / key / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _null_pp_stage():  # type: ignore[return]
    """No-op context manager used when no PipelineProgress is provided."""
    yield
