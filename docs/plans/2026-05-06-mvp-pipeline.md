# Vidscribe MVP — Local Video Transcription with CLI-Provider Correction

## Overview

CLI-инструмент `vidscribe` для распознавания видео полностью локально (ffmpeg
+ whisperX) с последующей корректировкой через CLI-провайдеров (claude / codex
/ ollama). Заменяет текущий flow noScribe → Gemini, убирая зависимость от
облачной мультимодалки, часовые лимиты и галлюцинации.

**Pipeline:**
```
video.mp4
  ├─ ffmpeg → audio 16k mono
  │           ├─ faster-whisper (word_timestamps=True) → asr_segments.json
  │           └─ pyannote diarization → diar.json
  │              → merge asr+diar → segments.json
  ├─ ffmpeg → keyframes (scene-detect + sampling) → frames.json
  ├─ chunker → chunks.json (по сменам говорящего / окну / сценам)
  ├─ speaker-id → speakers.json (LLM пытается извлечь имена из транскрипта/кадров)
  ├─ correction loop → corrected_chunks/ (LLM правит чанк за чанком + копит глоссарий)
  └─ assembler → final.md
```

**Ключевая идея:** провайдер вызывается как subprocess (как у ralphex), не
через SDK. Это снимает вопрос ключей в коде, использует существующие подписки
и даёт единый интерфейс для claude / codex / локального ollama.

## Context (from discovery)

- **Greenfield**, путь проекта: `~/Projects/vidscribe/`.
- **Skill-обёртка** будет жить отдельно: `~/.claude/skills/vidscribe/SKILL.md`
  (триггеры «распознай видео», «транскрибируй запись»).
- **Reuse noScribe assets (HF-токен НЕ нужен):**
  - папка `~/Library/Application Support/noScribe/whisper_models` — это слот
    для пользовательских custom-моделей, по умолчанию пуста. **НЕ использовать.**
  - реальные веса лежат внутри .app-бандла:
    - whisper (ct2): `/Applications/noScribe.app/Contents/Resources/models/precise/`
      (large-v3, ~1.5GB) и `models/fast/` (~780MB) — полный ct2-комплект:
      `model.bin`, `config.json`, `tokenizer.json`, `preprocessor_config.json`,
      `vocabulary.json`. faster-whisper грузит по абсолютному пути директории.
    - pyannote: `/Applications/noScribe.app/Contents/Resources/pyannote/config.yaml`
      + `segmentation/pytorch_model.bin` + `embedding/pytorch_model.bin`. Pipeline
      собирается вручную через `pyannote.audio.Pipeline.from_pretrained` с
      указанием локальных путей, либо из готового `config.yaml`.
  - bundle-detection: `/Applications/noScribe.app/Contents/Resources/` — если
    есть, по умолчанию используем оттуда; иначе fallback на HF (с токеном).
- **STT-стек: faster-whisper напрямую** (без whisperX-обёртки). Word-level
  timestamps через `word_timestamps=True` (cross-attention) — точности ±50ms
  хватает для chunking по говорящему и привязки кадров. Отказались от
  whisperX, чтобы не тащить wav2vec2 alignment-модели на каждый язык
  (~500MB-1GB) — для русского forced alignment даёт минимальный выигрыш над
  cross-attention timestamps.
- **Default модель:** `noscribe-precise` (large-v3 ct2 из бандла, ~1.5GB).
  `fast` — опция через флаг `--whisper-model fast` для черновых прогонов.
- **Diarization-glue** — пишем сами через pyannote напрямую: запускаем
  diarization pipeline, берём speaker turns, мапим к word-timestamps из
  faster-whisper методом перекрытия интервалов. ~30-50 строк, проще дебажится.
- **macOS Apple Silicon:** faster-whisper через ctranslate2 на CPU+int8.
  Нет полного MPS-ускорения, но Accelerate framework работает; на M-серии
  приемлемая скорость.
- **CLI-providers контракт:**
  - `claude -p "$PROMPT" --output-format json --max-turns 1` — с путями к
    кадрам в промпте (Read tool сам подхватит).
  - `codex exec "$PROMPT"` (или альтернативный non-interactive флаг).
  - `ollama run qwen2-vl:7b "$PROMPT"` — поддерживает картинки в промпте.

## Development Approach

- **Testing**: regular (код первым, тесты сразу за ним), `pytest` + `pytest-mock`.
- **CLI**: `typer` (опции, helps, completion).
- **Конфиги**: `pydantic` v2.
- **Внешние процессы**: `subprocess` через тонкие обёртки, mock в тестах.
- **Языки**: Python 3.11+, минимум зависимостей.
- Каждая задача завершается тестами + прогоном `pytest -q` перед следующей.
- Markdown-задачи (skill, README, prompts) тестов не требуют.
- Сохранять backward-compatibility внутри pipeline между задачами (если
  меняется формат artefact'а — обновить cache key).

## Testing Strategy

- **Unit**: каждый модуль (audio, stt, frames, chunker, provider, assembler,
  cache, speakers) с моками subprocess.
- **Integration**: тестовое короткое видео (10-30 сек, 2 говорящих) в
  `tests/fixtures/short.mp4`.
- **E2E**: один полный прогон pipeline на коротком видео с моком LLM-провайдера.
- Покрытие: целевые 80%+, проверять `pytest --cov`.

## Progress Tracking

- Mark completed items with `[x]` immediately when done
- Add newly discovered tasks with ➕ prefix
- Document issues/blockers with ⚠️ prefix
- Update plan if implementation deviates from original scope

## What Goes Where

- **Implementation Steps** (`[ ]`) — в коде проекта
- **Post-Completion** (без чекбоксов) — ручная проверка на реальных видео,
  настройка skill в Claude Code, документация в MEMORY.md

## Implementation Steps

### Task 1: Project skeleton

- [x] create `pyproject.toml` (project metadata, deps: typer, pydantic, rich,
      ffmpeg-python, faster-whisper, pyannote.audio, jinja2; dev: pytest,
      pytest-mock, pytest-cov, ruff)
- [x] create `src/vidscribe/__init__.py` and submodules: `cli.py`, `config.py`,
      `audio.py`, `stt.py`, `frames.py`, `chunker.py`, `speakers.py`,
      `provider.py`, `pipeline.py`, `assembler.py`, `cache.py`,
      `prompts/__init__.py`
- [x] create `.gitignore` (`.venv/`, `__pycache__/`, `*.egg-info/`,
      `.vidscribe/`, `tests/fixtures/*.mp4` except a tiny one, `.pytest_cache/`)
- [x] create `tests/conftest.py` with fixtures path resolver
- [x] register console script `vidscribe = vidscribe.cli:app`
- [x] write smoke test `tests/test_smoke.py` importing all submodules
- [x] run `pip install -e ".[dev]"` and `pytest -q` — must pass

### Task 2: Config layer

- [x] define `config.AppConfig` (provider, model, chunk_strategy, frame_rate,
      whisper_model='noscribe-precise', language='ru', hf_token, cache_dir)
      with pydantic
- [x] support env vars (`VIDSCRIBE_PROVIDER`, `VIDSCRIBE_MODEL`, `HF_TOKEN`)
      and CLI overrides
- [x] support optional `~/.config/vidscribe/config.toml`
- [x] write tests for env override + file load + CLI precedence
- [x] run tests

### Task 3: Audio extraction

- [ ] implement `audio.extract(video_path, out_path) -> Path` using
      `ffmpeg -y -i {video} -ac 1 -ar 16000 -vn {out}.wav`
- [ ] raise `AudioExtractionError` with helpful message if ffmpeg missing
- [ ] write tests with mocked subprocess + tiny test video
- [ ] run tests

### Task 4: faster-whisper STT (с реюзом noScribe-весов)

- [ ] implement `stt.detect_assets() -> AssetPaths` — ищет noScribe бандл по
      пути `/Applications/noScribe.app/Contents/Resources/`, возвращает
      whisper_precise_dir + whisper_fast_dir + pyannote_dir + config_yaml +
      segmentation_path + embedding_path; иначе None
- [ ] implement `stt.transcribe(audio_path, model='noscribe-precise', device='auto',
      language='ru') -> AsrResult`:
  - `noscribe-precise` (default) → faster-whisper грузит из
    `/Applications/noScribe.app/Contents/Resources/models/precise/`
  - `noscribe-fast` → из `models/fast/`
  - `large-v3` / `medium` / etc → стандартный download через faster-whisper +
    `HF_HOME` cache
- [ ] вызов: `WhisperModel(model_path, device=..., compute_type='int8')` →
      `model.transcribe(audio, language='ru', word_timestamps=True,
      vad_filter=True)` → AsrSegment[] с words[]
- [ ] auto-detect device: cuda > cpu (mps пока не поддерживается ctranslate2)
- [ ] на macOS принудительно `compute_type='int8'` (Accelerate-friendly)
- [ ] output `AsrResult` (segments, words) to JSON
- [ ] write tests с моком faster-whisper + integration-test (skip если
      noScribe бандл отсутствует)
- [ ] run tests

### Task 4b: Pyannote diarization + word-speaker mapping

- [ ] implement `stt.diarize(audio_path, assets: AssetPaths | None,
      hf_token=None) -> DiarResult` — список speaker turns [{start, end,
      speaker}]
- [ ] предпочесть локальный pyannote из noScribe:
  - загрузить `config.yaml` через `Pipeline.from_pretrained(local_yaml_path)`
  - переопределить пути к segmentation/embedding моделям (yaml ссылается
    на HF-имена → подменить на локальные `.bin`)
  - HF-токен НЕ требуется
- [ ] fallback: `Pipeline.from_pretrained('pyannote/speaker-diarization-3.1',
      use_auth_token=hf_token)` — только если бандл отсутствует
- [ ] implement `stt.merge_asr_diar(asr: AsrResult, diar: DiarResult) ->
      SttResult` — для каждого word ищем speaker по максимальному перекрытию
      интервалов; на segment-level берём mode по словам
- [ ] output `SttResult` (segments[{start,end,text,speaker,words[]}]) to JSON
- [ ] write tests:
  - моки pyannote pipeline + проверка merge-логики на синтетических данных
      (overlapping speakers, gaps)
  - integration-test с noScribe бандлом (skip если нет)
- [ ] run tests

### Task 5: Keyframe extraction (renumbered: was Task 5)

- [ ] implement `frames.extract(video_path, out_dir, scene_threshold=0.3,
      sample_every=10.0) -> list[FrameInfo]` через ffmpeg `select` filter +
      `showinfo`
- [ ] FrameInfo: `{ts: float, path: Path, scene_change: bool}`
- [ ] persist `frames.json` рядом с кадрами
- [ ] write tests with short fixture video
- [ ] run tests

### Task 6: Chunking

- [ ] implement `chunker.chunk(stt: SttResult, frames: list[FrameInfo],
      strategy: Literal['speaker', 'time', 'scene'], window_s=180) ->
      list[Chunk]`
- [ ] Chunk = `{idx, start, end, segments[], frame_paths[],
      surrounding_context: str}`
- [ ] frame_paths: все кадры в [start, end] + 1 опорный кадр в середине
- [ ] write tests covering all three strategies (speaker turns / fixed time /
      scene boundaries)
- [ ] run tests

### Task 7: Cache layer

- [ ] implement `cache.Cache(root: Path)` with methods `get(stage, key)`,
      `set(stage, key, artefact)`, `key_for(stage, **inputs)` (sha256)
- [ ] structure: `.vidscribe/cache/{video_hash}/{stage}/...`
- [ ] cache hit logging through rich
- [ ] CLI flag `--no-cache` to bypass for selected stages
- [ ] write tests (in-memory + temp dir)
- [ ] run tests

### Task 8: Prompt templates

- [ ] create `src/vidscribe/prompts/correct_chunk.md` — system prompt + slots
      for transcript, frame paths, glossary, speaker map, JSON schema
- [ ] create `src/vidscribe/prompts/identify_speakers.md` — поиск имён по
      транскрипту (паттерны «Привет, X», «X, расскажи») и по кадрам (имена
      на лоу-терах в Zoom/Teams)
- [ ] implement `prompts.render(name, **kwargs)` через jinja2
- [ ] enforce JSON output instruction (`{"corrected_text": "...",
      "glossary_delta": {...}, "notes": "..."}`)
- [ ] write tests for rendering + missing-slot detection
- [ ] run tests

### Task 9: Provider abstraction (CLI subprocess)

- [ ] define `provider.Provider` protocol with method
      `correct(prompt: str, frame_paths: list[Path], timeout: int) ->
      ProviderResponse`
- [ ] ProviderResponse: `{text: str, raw_json: dict, cost_estimate: float|None,
      duration_s: float}`
- [ ] implement `ClaudeCLIProvider(model='sonnet')`:
      `claude -p "$PROMPT" --output-format json --max-turns 1
      --permission-mode acceptEdits` (frame paths упомянуты в промпте,
      Read tool подхватит сам)
- [ ] implement `CodexCLIProvider(model='gpt-5.5')`:
      `codex exec --json "$PROMPT"` или эквивалент
- [ ] implement `OllamaProvider(model='qwen2-vl:7b')`:
      `ollama run {model} "$PROMPT"` с image-флагами
- [ ] provider factory `provider.make(name, **opts)`
- [ ] handle non-zero exit, parse JSON, retry once on transient errors
- [ ] write tests with mocked subprocess (simulate stdout/stderr/exit codes)
- [ ] run tests

### Task 10: Speaker identification

- [ ] implement `speakers.identify(stt, frames, provider, manual=None) ->
      dict[str, str]` — мапит SPEAKER_00 → 'Иван' / fallback s00
- [ ] strategy: один LLM-вызов на репрезентативные чанки (по 1-2 на каждого
      SPEAKER_*) с инструкцией извлечь имена
- [ ] CLI override: `--speakers "Иван,Алиса"` (позиционно по индексу спикеров)
- [ ] неопознанные → `s00`, `s01` и т.п.
- [ ] persist `speakers.json` в кэш
- [ ] write tests с мок-провайдером
- [ ] run tests

### Task 11: Correction loop

- [ ] implement `pipeline.correct_chunks(chunks, provider, speakers, cache)
      -> list[CorrectedChunk]`
- [ ] sequential по чанкам с накоплением `glossary` (имена, термины,
      повторяющиеся обороты)
- [ ] кэш по каждому чанку отдельно (key = hash(chunk_input + provider +
      model + glossary_snapshot)) — можно перезапустить только провалившиеся
- [ ] прогресс-бар через `rich.progress`
- [ ] write tests с мокнутым провайдером (несколько чанков, проверка
      накопления глоссария)
- [ ] run tests

### Task 12: Final assembly

- [ ] implement `assembler.assemble(corrected: list[CorrectedChunk],
      speakers: dict, fmt: Literal['md', 'srt'] = 'md') -> str`
- [ ] markdown: `## [HH:MM:SS] **Имя**\n\nреплика\n\n` с слиянием соседних
      реплик одного говорящего
- [ ] write tests
- [ ] run tests

### Task 13: CLI commands

- [ ] `vidscribe pipeline VIDEO [--provider X] [--model Y]
      [--whisper-model noscribe-precise|noscribe-fast|large-v3]
      [--chunk-strategy speaker|time|scene] [--speakers "A,B"]
      [--out FILE] [--no-cache]` — full run, default whisper-model =
      `noscribe-precise`
- [ ] `vidscribe extract VIDEO` — только аудио + кадры (no LLM)
- [ ] `vidscribe transcribe VIDEO` — STT only
- [ ] `vidscribe correct VIDEO --provider X` — re-run correction with cached STT/frames
- [ ] `vidscribe cache list|clear [VIDEO]` — управление кэшем
- [ ] write tests via `typer.testing.CliRunner` + mocked pipeline
- [ ] run tests

### Task 14: Verify acceptance criteria

- [ ] прогнать pipeline на тестовом 5-минутном видео с 2 говорящими
- [ ] проверить fallback s00/s01 + override через `--speakers`
- [ ] переключить провайдер (claude → codex) на тех же артефактах через кэш
- [ ] прогнать `pytest -q` (все unit) + e2e
- [ ] прогнать `ruff check` — 0 ошибок
- [ ] проверить покрытие `pytest --cov` — 80%+

### Task 15: Skill wrapper

- [ ] create `~/.claude/skills/vidscribe/SKILL.md` с триггерами «распознай
      видео», «транскрибируй запись», «video transcription»
- [ ] document common flags + provider selection guide (когда claude/codex/ollama)
- [ ] add 3-5 example invocations
- [ ] note: skill вызывает CLI `vidscribe`, никакой логики в скиле быть не должно

### Task 16: Documentation

- [ ] update `README.md` — installation, first-run setup (HF token), provider
      configuration, troubleshooting (ffmpeg not found, HF accept terms)
- [ ] document HF cache reuse from noScribe (если применимо после Task 4)
- [ ] add architecture diagram in `docs/architecture.md` (mermaid)

## Technical Details

**Forms of artefacts (cache contents):**

```
.vidscribe/cache/{video_sha256}/
  audio/audio.wav
  stt/segments.json          # whisperX output
  frames/frames.json
  frames/00_00_05_000.jpg ...
  chunks/chunks.json
  speakers/speakers.json
  corrected/chunk_0001.json  # one per chunk
  corrected/chunk_0002.json
  ...
  final/transcript.md
```

**Provider invocation (Claude example):**

```python
prompt = render("correct_chunk", chunk=chunk, frame_paths=chunk.frame_paths,
                glossary=glossary, speakers=speakers)
result = subprocess.run(
    ["claude", "-p", prompt, "--output-format", "json",
     "--max-turns", "1", "--permission-mode", "acceptEdits"],
    capture_output=True, text=True, timeout=300, check=True,
)
parsed = json.loads(result.stdout)  # claude's wrapper JSON
inner = json.loads(parsed["result"])  # our JSON inside
```

**Speaker map example:**
```json
{ "SPEAKER_00": "Иван", "SPEAKER_01": "Алиса", "SPEAKER_02": "s02" }
```

## Post-Completion

**Manual verification:**
- Полный прогон на реальной записи (лекция/интервью), сравнить качество с
  noScribe + Gemini
- Подтвердить, что Claude CLI subprocess реально читает кадры с диска (если
  нет — добавить `--add-dir` или явную инструкцию в промпт)
- Подобрать оптимальную частоту sampling кадров под типичные видео (1 кадр на
  5/10/15 секунд)

**External system updates:**
- Записать заметку в `~/.claude/projects/-Users-svs/memory/MEMORY.md` про
  vidscribe после первого успешного прогона
- Если skill полезен — добавить в обзор личных скиллов
