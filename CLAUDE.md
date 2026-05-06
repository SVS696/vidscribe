# vidscribe development notes

vidscribe is a local-first video transcription pipeline. Media and speech stages
run locally with ffmpeg, faster-whisper, and pyannote; transcript cleanup is
delegated to CLI providers through subprocess calls.

## Common commands

```bash
pip install -e ".[dev]"
pytest -q
pytest --cov
ruff check
vidscribe --help
```

## Architecture conventions

- Typer owns CLI wiring in `src/vidscribe/cli.py`.
- Stage artefacts use pydantic models and the file-backed cache under
  `.vidscribe/cache`.
- Prompt templates live in `src/vidscribe/prompts` and must render with strict
  undefined handling before provider execution.
- Provider CLIs are invoked as subprocesses. Claude and Codex correction runs
  execute from an isolated temporary working directory and should not be given
  edit permissions.

## Model assets

The `noscribe-precise` and `noscribe-fast` aliases load noScribe assets from
`/Applications/noScribe.app/Contents/Resources/`. These aliases should fail fast
if the local assets are missing. Use a standard faster-whisper model name such
as `large-v3` to intentionally use Hugging Face downloads and pyannote fallback
diarization.
