# vidscribe

Локальное распознавание видео с LLM-корректировкой через CLI-провайдеров.

Заменяет связку **[noScribe](https://github.com/kaixxx/noScribe) + Gemini** на
полностью контролируемый pipeline:
[ffmpeg](https://ffmpeg.org/) →
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (STT) +
[pyannote.audio](https://github.com/pyannote/pyannote-audio) (diarization) →
ключевые кадры → CLI-провайдер (claude / codex / ollama) → выровненный markdown.

## Зачем

- **Без часовых лимитов и галлюцинаций Gemini** — кадры и текст идут к LLM
  выровненными чанками, не «вот тебе час видео, разберись».
- **Локальная транскрипция** через faster-whisper (large-v3 из noScribe.app,
  word-level timestamps) + pyannote diarization напрямую — без HF-токена,
  без лишних alignment-моделей.
- **CLI-провайдеры вместо API** — биллинг через подписки:
  `claude -p`, `codex exec`, `ollama run`.
- **Mix-режим** — текст полирует один провайдер (codex), визуальные правки
  делает другой (claude через Read tool на кадрах). Объединяет сильные стороны.
- **Кэш по этапам** — переиграть только correction с другой моделью без
  перезапуска STT (~5 минут на 8 минут видео).

## Зависимости

| Компонент | Зачем | Ссылка |
|-----------|-------|--------|
| **Python 3.11+** | Runtime | <https://www.python.org/> |
| **ffmpeg** | Извлечение аудио + кадры | <https://ffmpeg.org/> |
| **faster-whisper** | STT (CTranslate2 backend) | <https://github.com/SYSTRAN/faster-whisper> |
| **pyannote.audio** | Speaker diarization | <https://github.com/pyannote/pyannote-audio> |
| **soundfile** | Pre-load audio for pyannote | <https://github.com/bastibe/python-soundfile> |
| **rich** | Прогресс-бары + цветной вывод | <https://github.com/Textualize/rich> |
| **Typer** | CLI | <https://typer.tiangolo.com/> |
| **noScribe.app** *(опц.)* | Готовые модели whisper + pyannote | <https://github.com/kaixxx/noScribe> |
| **Claude Code CLI** *(опц.)* | Multimodal correction provider | <https://docs.claude.com/en/docs/claude-code> |
| **Codex CLI** *(опц.)* | Text correction provider | <https://github.com/openai/codex> |
| **Ollama** *(опц.)* | Локальная multimodal correction | <https://ollama.com/> |

## Status

MVP pipeline реализован, mix-mode добавлен, 168 unit-тестов зелёные.
План разработки: `docs/plans/completed/2026-05-06-mvp-pipeline.md`.

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
# cache_dir defaults to the platform cache location; uncomment to override:
# cache_dir = "/path/to/custom/cache"
```

Global options may be passed before a subcommand: `--provider`, `--model`,
`--chunk-strategy`, `--frame-rate`, `--whisper-model`, `--language`,
`--hf-token`, `--cache-dir`, and `--speakers`.

`frame_rate` is the sampled frame frequency in frames per second. The default
`0.1` means one sampled frame every 10 seconds. Higher values provide more visual
context and increase provider prompt size.

### Cache directory

Pipeline artefacts and logs are stored in a global user cache directory.
Default location:

- **macOS:** `~/Library/Caches/vidscribe`
- **Linux:** `${XDG_CACHE_HOME:-~/.cache}/vidscribe`
- **Windows:** `%LOCALAPPDATA%/vidscribe/cache`

Logs live in the same directory under a `logs/` subdirectory.

Override via CLI flag or environment variable:

```bash
vidscribe --cache-dir /path/to/cache pipeline recording.mp4
# or
export VIDSCRIBE_CACHE_DIR=/path/to/cache
```

Supported environment variables:

```bash
export VIDSCRIBE_PROVIDER=claude
export VIDSCRIBE_MODEL=sonnet
export HF_TOKEN=hf_...
export VIDSCRIBE_CACHE_DIR=/path/to/cache   # override cache/logs location
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
store API keys. Auth относится к глобальной CLI provider'a.

| Provider | Multimodal | Команда | Где взять auth |
|----------|------------|---------|----------------|
| `claude` | ✅ через Read tool | `claude -p ... --output-format json` | <https://docs.claude.com/en/docs/claude-code> |
| `codex` | ❌ только текст | `codex exec --json ...` | <https://github.com/openai/codex> |
| `ollama` | ✅ нативно (qwen-vl) | `ollama run MODEL ...` | <https://ollama.com/> (локально, бесплатно) |

The correction prompt requires a JSON object with `corrected_text`,
`glossary_delta`, and `notes`. Provider stdout is parsed and normalized before
the final transcript is assembled.

### Mix-mode (recommended for screen recordings)

Two-pass correction: текст полирует один провайдер, визуал — другой:

```bash
vidscribe pipeline VIDEO \
  --correction-mode mix \
  --text-provider codex --text-model gpt-5.5 \
  --visual-provider claude --visual-model sonnet \
  --out transcript.md
```

Pass 1 (codex): чистит ASR-ошибки в речи, без кадров.
Pass 2 (claude): открывает каждый кадр через Read tool и уточняет ТОЛЬКО
on-screen числа, имена, термины, названия колонок. Не переписывает речь.

Полезно для записей экрана/демо, где важны точные значения с UI (числа в
ячейках Excel, названия колонок, лейблы кнопок). Стоит вдвое больше LLM-вызовов
на чанк, но ~70% дешевле claude-only за счёт codex-pass.

### Screen context (visual events)

Описывайте on-screen действия: переключения вкладок, выделения — всё что не
произносится речью. Опционально через `--screen-context inline|aside|footer`.

```bash
vidscribe pipeline VIDEO --correction-mode mix \
  --visual-provider claude --visual-model sonnet \
  --screen-context inline \
  --out transcript.md
```

Форматы:
- `inline` (рекомендуется) — блок `> 📺 [HH:MM:SS] description` перед репликой
- `aside` — отдельная секция «Сцены» в начале документа
- `footer` — `*[scene HH:MM:SS: ...]*` после реплики

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

## Pipeline flow (9 этапов)

```
[1/9] Audio extraction      ffmpeg → 16kHz mono WAV
[2/9] Frames extraction     scene-detect + sampling, with 3-level fallback (см. ниже)
[3/9] STT                   faster-whisper (по умолчанию ct2-модель из noScribe.app)
[4/9] Diarization           pyannote.audio (локальная или fallback на HF)
[5/9] Merge ASR+diar        слова получают speaker label по максимальному overlap
[6/9] Chunking              speaker / time / scene стратегия
[7/9] Speaker identification    LLM-call (provider) → s00→Иван, s01→Анна, или fallback
[8/9] Correction loop       single (один pass) или mix (text→visual)
[9/9] Final assembly        markdown с timestamps, screen-context, speaker map
```

### Frame extraction fallback (для битых видео)

Запись из Zoom/Telemost иногда содержит битые кадры — ffmpeg застревает на
конкретном таймштампе. Vidscribe имеет 3-уровневый fallback с watchdog'ом
(120s таймаут на застой `out_time_us`):

```
[2/9] scene-detect + sampling
       │
       │ застрял? watchdog kill после 120s
       ▼
[2/9] sample-only (no scene-detect)
       флаги: -err_detect ignore_err -fflags +discardcorrupt+genpts
              -skip_frame nokey -an
       │
       │ застрял? watchdog kill после 120s
       ▼
[2/9] seek-based per-frame
       один ffmpeg -ss T -frames:v 1 на каждый таймштамп
       per-call timeout 15s, битые сегменты пропускаются индивидуально
       — самый надёжный вариант, но медленнее
```

`--frames-strategy seek` сразу пропускает первые два уровня (~4 минуты ожидания).

## CLI flags reference

Глобальные опции (передаются ДО subcommand):

| Флаг | Default | Описание |
|------|---------|----------|
| `--provider` | (config) | CLI провайдер: `claude`, `codex`, `ollama` (для single-mode) |
| `--model` | (config) | Имя модели провайдера |
| `--whisper-model` | `noscribe-precise` | `noscribe-precise` / `noscribe-fast` (из noScribe.app) или `large-v3` / `medium` (HF) |
| `--chunk-strategy` | `speaker` | `speaker` (по сменам говорящего) / `time` (фикс. окно) / `scene` (по кадрам) |
| `--frame-rate` | `0.1` | Частота сэмплирования кадров (FPS). 0.1 = 1 кадр / 10 сек |
| `--language` | `ru` | Язык транскрипции |
| `--hf-token` | env `HF_TOKEN` | HuggingFace токен для fallback на pyannote из HF |
| `--cache-dir` | `~/Library/Caches/vidscribe` (macOS) | Корень кэша артефактов (по умолчанию global per-user) |
| `--no-cache STAGE` | — | Пропустить кэш на стадии: `audio`, `frames`, `asr`, `diar`, `stt`, `chunks`, `speakers`, `corrected`, `final`. Можно повторять. |
| `--speakers "A,B"` | — | Имена говорящих позиционно (s00=A, s01=B). Override для speaker-id |
| `--quiet` | off | Выключить прогресс на stderr (для CI/скриптов) |
| `--no-log` | off | Не создавать файл лога |
| `--log-file PATH` | auto | Переопределить путь лог-файла |

### `vidscribe pipeline VIDEO` — full processing

| Флаг | Default | Описание |
|------|---------|----------|
| `--out FILE` | — | Путь к финальному markdown |
| `--no-cache` | — | Не использовать кэш для всех стадий этого запуска |
| `--correction-mode` | `single` | `single` (один pass) или `mix` (codex текст → claude визуал) |
| `--text-provider` | `--provider` | Провайдер для Pass 1 (mix-mode, текст без кадров) |
| `--text-model` | `--model` | Модель для Pass 1 |
| `--visual-provider` | `claude` | Провайдер для Pass 2 (mix-mode, кадры через Read tool) |
| `--visual-model` | `sonnet` | Модель для Pass 2 |
| `--screen-context` | `off` | `off` / `inline` (`> 📺 [HH:MM:SS] desc` перед репликой) / `aside` (раздел «Сцены» в начале) / `footer` |
| `--frames-strategy` | `auto` | `auto` (3-level fallback) / `scene-detect` / `sample-only` / `seek` |

### Subcommands

| Команда | Когда использовать |
|---------|---------------------|
| `pipeline VIDEO` | Первый запуск — full processing |
| `correct VIDEO` | Перепрогон только correction loop (использует кэш audio/STT/frames) |
| `transcribe VIDEO` | Только STT + diarization, без LLM-correction |
| `extract VIDEO` | Только audio + keyframes |
| `cache list [VIDEO]` | Что в кэше |
| `cache clear [VIDEO]` | Очистить кэш |
| `logs` | Путь к latest.log |
| `logs --follow` | `tail -f` последнего лога (live-эфир в другом окне) |
| `logs --list` | Последние 10 запусков |

## Architecture

See `docs/architecture.md` for the pipeline diagram and artifact layout.

## Claude Code skill

Если используешь Claude Code — есть skill-обёртка которая декодирует
естественный запрос юзера в правильную команду (включая mix-mode,
screen-context, frames-strategy, multi-video concat и т.д.). Установка:

```bash
mkdir -p ~/.claude/skills/vidscribe
cp docs/skill.md ~/.claude/skills/vidscribe/SKILL.md
# или symlink, чтобы git pull обновлял активный skill:
ln -sf "$(pwd)/docs/skill.md" ~/.claude/skills/vidscribe/SKILL.md
```

После установки в Claude Code можно просто говорить «распознай видео X» или
«видео битое, прогони с seek-стратегией» — Claude сам соберёт нужную
`vidscribe` команду по skill recipes.

## Contributing

Issues / PRs welcome. Workflow:

```bash
.venv/bin/pytest -q       # 168 passing
.venv/bin/ruff check      # lint clean
```

## License

MIT — see [LICENSE](./LICENSE).
