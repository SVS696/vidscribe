"""Command-line interface for vidscribe."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from vidscribe import assembler, audio, chunker, frames, provider, speakers, stt
from vidscribe.cache import Cache
from vidscribe.config import AppConfig, ChunkStrategy, load_config
from vidscribe.frames import FrameInfo
from vidscribe.pipeline import correct_chunks
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
) -> None:
    """Run vidscribe commands."""

    ctx.obj = {
        "config": load_config(
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
    }


def config_from_context(ctx: typer.Context) -> AppConfig:
    """Return callback-loaded config for subcommands."""

    return ctx.obj["config"]


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
) -> None:
    """Run the full transcription and correction pipeline."""

    config = _command_config(
        ctx,
        provider=provider_name,
        model=model,
        whisper_model=whisper_model,
        chunk_strategy=chunk_strategy,
        speakers=_parse_speakers(speaker_names) if speaker_names is not None else None,
    )
    cache = _cache(config, no_cache=no_cache)
    video_key = cache.key_for("video", video=video)
    audio_path, frame_items = _extract(video, config, cache, video_key)
    stt_result = _transcribe_audio(audio_path, config, cache, video_key)
    chunk_items = _chunks(stt_result, frame_items, config, cache, video_key)
    cli_provider = provider.make(config.provider, model=config.model)
    speaker_map = speakers.identify(
        stt_result,
        frame_items,
        cli_provider,
        manual=config.speakers,
        cache=cache,
    )
    corrected = correct_chunks(chunk_items, cli_provider, speaker_map, cache)
    transcript = assembler.assemble(corrected, speaker_map)
    output_path = out or video.with_suffix(".md")
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
) -> None:
    """Extract audio and keyframes without LLM calls."""

    config = config_from_context(ctx)
    if no_cache:
        disabled = set(config.no_cache) | _CORRECT_RECOMPUTE_STAGES
        cache = Cache(config.cache_dir, disabled_stages=disabled, console=console)
    else:
        cache = _cache(config)
    video_key = cache.key_for("video", video=video)
    audio_path, frame_items = _extract(video, config, cache, video_key)
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
    audio_path = _audio(video, cache, video_key)
    stt_result = _transcribe_audio(audio_path, config, cache, video_key)
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
) -> None:
    """Re-run correction from cached STT and frames."""

    config = _command_config(ctx, provider=provider_name, model=model)
    if no_cache:
        disabled = set(config.no_cache) | _CORRECT_RECOMPUTE_STAGES
        cache = Cache(config.cache_dir, disabled_stages=disabled, console=console)
    else:
        cache = _cache(config)
    video_key = cache.key_for("video", video=video)
    stt_result = _cached_model(cache, "stt", video_key, SttResult)
    frame_items = _cached_frames(cache, video_key)
    chunk_items = _chunks(stt_result, frame_items, config, cache, video_key)
    cli_provider = provider.make(config.provider, model=config.model)
    speaker_map = speakers.identify(
        stt_result,
        frame_items,
        cli_provider,
        manual=config.speakers,
        cache=cache,
    )
    corrected = correct_chunks(chunk_items, cli_provider, speaker_map, cache)
    transcript = assembler.assemble(corrected, speaker_map)
    output_path = out or video.with_suffix(".md")
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


def _parse_speakers(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _command_config(ctx: typer.Context, **overrides: Any) -> AppConfig:
    config = config_from_context(ctx)
    compact = {key: value for key, value in overrides.items() if value is not None}
    if not compact:
        return config
    return config.model_copy(update=compact)


def _cache(config: AppConfig, *, no_cache: bool = False) -> Cache:
    disabled = set(config.no_cache)
    if no_cache:
        disabled.update(_ALL_STAGES)
    return Cache(config.cache_dir, disabled_stages=disabled, console=console)


def _extract(
    video: Path,
    config: AppConfig,
    cache: Cache,
    video_key: str,
) -> tuple[Path, list[FrameInfo]]:
    return _audio(video, cache, video_key), _frames(video, config, cache, video_key)


def _audio(video: Path, cache: Cache, video_key: str) -> Path:
    stage_dir = _stage_dir(cache, video_key, "audio")
    cached = _cached_file(cache, "audio", video_key)
    if cached is not None:
        return cached
    output = stage_dir / "audio.wav"
    return audio.extract(video, output)


def _frames(
    video: Path,
    config: AppConfig,
    cache: Cache,
    video_key: str,
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
                FrameInfo.model_validate(item)
                for item in json.loads(frames_json.read_text(encoding="utf-8"))
            ]

    frame_items = frames.extract(
        video,
        output_dir,
        sample_every=1 / config.frame_rate,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return frame_items


def _transcribe_audio(
    audio_path: Path,
    config: AppConfig,
    cache: Cache,
    video_key: str,
) -> SttResult:
    assets = stt.detect_assets()
    metadata = _stt_metadata(config, assets)
    metadata_path = _stage_dir(cache, video_key, "stt") / "metadata.json"
    cached = cache.get("stt", video_key)
    if isinstance(cached, dict) and _read_json(metadata_path) == metadata:
        return SttResult.model_validate(cached)

    asr = stt.transcribe(
        audio_path,
        model=config.whisper_model,
        language=config.language,
    )
    diar = stt.diarize(audio_path, assets, hf_token=config.hf_token)
    result = stt.merge_asr_diar(asr, diar)
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
) -> list[chunker.Chunk]:
    cache_key = cache.key_for(
        "chunks",
        video=video_key,
        strategy=config.chunk_strategy,
        stt=stt_result,
        frames=frame_items,
    )
    cached = cache.get("chunks", cache_key)
    if isinstance(cached, list):
        return [chunker.Chunk.model_validate(item) for item in cached]
    chunk_items = chunker.chunk(stt_result, frame_items, config.chunk_strategy)
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
        FrameInfo.model_validate(item)
        for item in json.loads(frames_json.read_text(encoding="utf-8"))
    ]


def _frames_metadata(config: AppConfig) -> dict[str, Any]:
    return {
        "frame_rate": config.frame_rate,
        "sample_every": 1 / config.frame_rate,
        "scene_threshold": 0.3,
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
