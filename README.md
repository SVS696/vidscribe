# vidscribe

Локальное распознавание видео с LLM-корректировкой через CLI-провайдеров.

Заменяет связку **noScribe + Gemini** на полностью контролируемый pipeline:
ffmpeg → faster-whisper (STT) + pyannote (diarization) → ключевые кадры →
CLI-провайдер (claude/codex/ollama) → выровненный markdown.

## Зачем

- **Без часовых лимитов и галлюцинаций Gemini** — кадры и текст идут к LLM
  выровненными чанками, не «вот тебе час видео, разберись».
- **Локальная транскрипция** через faster-whisper (large-v3 из noScribe.app,
  word-level timestamps) + pyannote diarization напрямую — без HF-токена,
  без лишних alignment-моделей.
- **CLI-провайдеры вместо API** — биллинг через подписки, как у ralphex:
  `claude -p`, `codex exec`, `ollama run`.
- **Кэш по этапам** — переиграть только correction с другой моделью без
  перезапуска STT.

## Status

MVP pipeline реализован: extraction, STT + diarization, keyframes, chunking,
speaker identification, correction loop, final assembly and cache management.
План разработки: `docs/plans/2026-05-06-mvp-pipeline.md`.

## Installation

```bash
cd /path/to/vidscribe
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

External tools:

- `ffmpeg`: `brew install ffmpeg`
- one correction provider CLI:
  - Claude Code CLI for `--provider claude`
  - Codex CLI for `--provider codex`
  - Ollama for `--provider ollama`

Verify the install:

```bash
vidscribe --help
pytest -q
ruff check
```

## First-run setup

By default vidscribe uses `--whisper-model noscribe-precise`, which loads the
local CTranslate2 Whisper large-v3 weights from noScribe.app when present. This
path does not require a Hugging Face token.

If noScribe.app is not installed, choose a faster-whisper model name such as
`large-v3`, `medium`, or `small`. In that mode faster-whisper downloads Whisper
weights into the Hugging Face cache, and pyannote diarization falls back to
`pyannote/speaker-diarization-3.1`.

For the Hugging Face fallback:

```bash
pip install "huggingface_hub[cli]"
huggingface-cli login
```

Use an account that has accepted the model terms for pyannote
`speaker-diarization-3.1`. You can also pass a token per run:

```bash
vidscribe pipeline recording.mp4 --whisper-model large-v3 --hf-token "$HF_TOKEN"
```

## Configuration

Configuration precedence is defaults, then `~/.config/vidscribe/config.toml`,
then environment variables, then CLI flags.

Example config file:

```toml
provider = "codex"
model = "gpt-5.5"
chunk_strategy = "speaker"
frame_rate = 0.1
whisper_model = "noscribe-precise"
language = "ru"
cache_dir = ".vidscribe"
```

Global options may be passed before a subcommand: `--provider`, `--model`,
`--chunk-strategy`, `--frame-rate`, `--whisper-model`, `--language`,
`--hf-token`, `--cache-dir`, and `--speakers`.

`frame_rate` is the sampled frame frequency in frames per second. The default
`0.1` means one sampled frame every 10 seconds. Higher values provide more visual
context and increase provider prompt size.

Supported environment variables:

```bash
export VIDSCRIBE_PROVIDER=claude
export VIDSCRIBE_MODEL=sonnet
export HF_TOKEN=hf_...
```

## Quick start

```bash
# полный прогон с параметрами по умолчанию
vidscribe pipeline recording.mp4 --out transcript.md

# Claude CLI
vidscribe pipeline recording.mp4 --provider claude --model sonnet --out transcript.md

# Codex CLI
vidscribe pipeline recording.mp4 --provider codex --model gpt-5.5 --out transcript.md

# локальная мультимодалка (бесплатно, медленнее)
vidscribe pipeline recording.mp4 --provider ollama --model qwen2-vl:7b

# ручные имена говорящих
vidscribe pipeline recording.mp4 --speakers "Иван,Алиса"

# перепрогнать только correction с другим провайдером по cached STT/frames
vidscribe correct recording.mp4 --provider codex
```

Other commands:

```bash
# audio + keyframes only
vidscribe extract recording.mp4

# STT + diarization only
vidscribe transcribe recording.mp4

# inspect cache keys or stages for one video
vidscribe cache list
vidscribe cache list recording.mp4

# clear all cache or one video's cache
vidscribe cache clear
vidscribe cache clear recording.mp4
```

Use subcommand `--no-cache` on `pipeline`, `extract`, or `transcribe` to bypass
all relevant cached artifacts for that run. On `correct`, subcommand
`--no-cache` recomputes chunking, speaker identification, correction, and final
assembly while still reading the cached STT and frame prerequisites. The global
`--no-cache STAGE` option can be repeated for specific stages: `audio`, `frames`,
`asr`, `diar`, `stt`, `chunks`, `speakers`, `corrected`, and `final`.

## noScribe model reuse

When `/Applications/noScribe.app/Contents/Resources/` exists and contains the
expected files, vidscribe reuses noScribe assets directly:

- precise Whisper model:
  `/Applications/noScribe.app/Contents/Resources/models/precise/`
- fast Whisper model:
  `/Applications/noScribe.app/Contents/Resources/models/fast/`
- pyannote config:
  `/Applications/noScribe.app/Contents/Resources/pyannote/config.yaml`
- pyannote model weights:
  `/Applications/noScribe.app/Contents/Resources/pyannote/segmentation/pytorch_model.bin`
  and
  `/Applications/noScribe.app/Contents/Resources/pyannote/embedding/pytorch_model.bin`

The `noscribe-precise` and `noscribe-fast` aliases require those local files.
If they are unavailable, vidscribe raises an asset error instead of silently
downloading a different model. To use Hugging Face downloads intentionally, pass
a standard faster-whisper model name:

```bash
vidscribe pipeline recording.mp4 --whisper-model large-v3
```

## Providers

The provider is invoked as a subprocess; vidscribe does not use provider SDKs or
store API keys.

- `claude`: runs `claude -p PROMPT --output-format json --max-turns 1`, plus
  `--model MODEL` when configured
- `codex`: runs `codex exec --json ...` from an isolated temporary working
  directory
- `ollama`: runs `ollama run MODEL ...`

The correction prompt requires a JSON object with `corrected_text`,
`glossary_delta`, and `notes`. Provider stdout is parsed and normalized before
the final transcript is assembled.

## Troubleshooting

`ffmpeg was not found`:

Install ffmpeg and make sure it is on `PATH`.

```bash
brew install ffmpeg
ffmpeg -version
```

`noScribe model assets were not found`:

Install noScribe.app in `/Applications`, or pass a standard faster-whisper model
name with `--whisper-model large-v3`. The noScribe custom-model directory under
`~/Library/Application Support/noScribe/whisper_models` is not used by vidscribe.

`pyannote` or Hugging Face access errors:

Accept the pyannote model terms in Hugging Face, run `huggingface-cli login`,
or provide `HF_TOKEN`. The fallback diarization model is
`pyannote/speaker-diarization-3.1`.

Provider command not found:

Install the selected CLI and check that the binary is on `PATH` from the shell
where you run vidscribe. For local Ollama runs, pull the model first:

```bash
ollama pull qwen2-vl:7b
```

`vidscribe correct` says cached STT or frames are missing:

Run `vidscribe pipeline VIDEO` or at least `vidscribe transcribe VIDEO` and
`vidscribe extract VIDEO` first. The correction command intentionally works from
cached STT and frame artifacts.

## Architecture

See `docs/architecture.md` for the pipeline diagram and artifact layout.
