"""Command-line interface for vidscribe."""

from pathlib import Path
from typing import Annotated

import typer

from vidscribe.config import AppConfig, ChunkStrategy, load_config

app = typer.Typer(
    help="Local video transcription with CLI-provider correction.",
    invoke_without_command=True,
)


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
            }
        )
    }


def config_from_context(ctx: typer.Context) -> AppConfig:
    """Return callback-loaded config for subcommands."""

    return ctx.obj["config"]
