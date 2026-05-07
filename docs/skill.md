---
name: vidscribe
version: 0.2.0
---

# vidscribe

Local video transcription through the `vidscribe` CLI: ffmpeg extracts audio
and frames, faster-whisper + pyannote produce speaker-aware transcript chunks,
CLI providers (claude / codex / ollama) correct the text and add visual
context, the assembler emits final markdown.

## Triggers

Use this skill when the user asks to:
- распознай / транскрибируй / расшифруй видео|встречу|запись|интервью|лекцию
- transcribe a video, meeting recording, screen recording, Zoom/Teams recording
- describe what's happening on screen + speech
- "обработай" + указание на видео-файл

Also use when user has an existing transcript artifact in cache and asks for:
- перегенерировать / обновить транскрипт с другой моделью
- доправить имена спикеров / ошибки в готовом транскрипте

## Cardinal rule

This skill **only invokes the CLI**. Do NOT re-implement transcription,
diarization, frame extraction, or LLM calls inside Claude Code. Read the user's
intent, pick flags from the decode table, run `vidscribe`, report the output
path.

## Decode table — user intent → command

The user does not need to remember flags. Translate their phrasing to one of
these recipes:

| Если юзер сказал… | Recipe |
|---|---|
| «распознай видео», «транскрибируй встречу», без уточнений | **DEFAULT** (см. ниже) |
| screen recording / запись экрана / скринкаст / демо / показ интерфейса | **DEFAULT** (mix + screen-context — главный кейс) |
| лекция / подкаст / интервью без экрана / только аудио | `--correction-mode single --provider claude --screen-context off` |
| только текст быстро, без визуала | `--correction-mode single --provider codex --screen-context off` |
| «доправь имена», известные имена спикеров | DEFAULT + `--speakers "Имя1,Имя2"` (порядок = индекс s00, s01) |
| «обнови с другой моделью», «попробуй другую модель» | `vidscribe correct VIDEO --provider X --model Y --out OUT` (использует кэш) |
| «локально, без интернета», privacy concern | `--provider ollama --model qwen2-vl:7b` |
| ограниченный бюджет, простой текст | `--correction-mode single --provider codex` |
| «без кэша», «свежий запуск» | добавь `--no-cache` |
| **«это одно распиленное видео» / «продолжение» / «прогнать N видео в один transcript»** | **MULTI-VIDEO** — concat first (см. ниже), потом DEFAULT pipeline на merged файле |
| **«видео битое / прерывалось / зависает на frame extraction / запись из Zoom/Telemost»** | DEFAULT + `--frames-strategy seek` (per-frame ffmpeg, медленнее но битые сегменты пропускаются) |

## MULTI-VIDEO recipe

Если юзер говорит что несколько файлов — это одно распиленное видео (встреча
прервалась, продолжение, часть 1+2):

```bash
# 1. Создать concat-list (одинаковые форматы и кодеки)
printf "file '%s'\nfile '%s'\n" "/abs/path/v1.webm" "/abs/path/v2.webm" > /tmp/vidscribe-concat.txt

# 2. ffmpeg concat без рекодинга (если форматы и кодеки совпадают)
ffmpeg -y -f concat -safe 0 -i /tmp/vidscribe-concat.txt -c copy /tmp/merged.webm

# 3. Стандартный pipeline на merged файле
vidscribe pipeline /tmp/merged.webm \
  --correction-mode mix \
  --text-provider codex --text-model gpt-5.5 \
  --visual-provider claude --visual-model sonnet \
  --screen-context inline \
  --out /path/transcript.md

# 4. (опц.) убрать временные файлы
rm /tmp/vidscribe-concat.txt /tmp/merged.webm
```

**Если форматы/кодеки разные** (.webm + .mp4, разные fps/кодеки) — используй
filter_complex с re-encoding:
```bash
ffmpeg -i v1.webm -i v2.mp4 \
  -filter_complex "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]" \
  -map "[v]" -map "[a]" /tmp/merged.mp4
```

**Подводный камень:** если у видео РАЗНОЕ разрешение/sample-rate, нужно
сначала привести к общему формату. Для `vidscribe` это не критично — он всё
равно ремэппит аудио в 16kHz mono перед STT, а кадры берёт по таймстампам.
Главное чтобы концат не упал.

**Таймстампы:** после concat все события идут в едином таймлайне (v2 начинается
в момент окончания v1) — никаких "склеек руками" транскрипта. Pipeline видит
один файл.

## DEFAULT recipe (для большинства случаев)

```bash
vidscribe pipeline "VIDEO" \
  --correction-mode mix \
  --text-provider codex --text-model gpt-5.5 \
  --visual-provider claude --visual-model sonnet \
  --screen-context inline \
  --out "OUTPUT.md"
```

**Почему такой default:**
- mix-mode = codex полирует речь, claude отдельно правит on-screen числа/имена
- screen-context inline = `> 📺 [time] description` блоки между репликами для скринкастов
- Output путь = рядом с видео или в той же директории, .md имя из stem видео
- whisper-model `noscribe-precise` (default) = локально из noScribe.app

## Что делать всегда

1. **Перед запуском**: спроси у юзера output путь только если он не указан.
   По умолчанию делай `<video_dir>/<video_stem>-transcript.md`.
2. **Запусти в фоне** через `run_in_background=true` — pipeline на 8-минутном
   видео занимает ~30-40 минут (STT 2 мин + diar 4 мин + correction 30 мин).
3. **Сразу скажи юзеру про `vidscribe logs --follow`** во втором окне для
   live-прогресса. Не предлагай это если юзер сам и так в курсе.
4. **Жди уведомления** о завершении background task — НЕ poll вручную.
5. **После завершения**: покажи юзеру путь к output + краткую статистику
   (длительность, кол-во спикеров, кол-во чанков).

## Особенности и подводные камни

### Имена спикеров
Speaker-id может ошибаться (LLM путает упомянутых vs говорящих). Если юзер
заранее знает имена — добавь `--speakers "A,B"` (позиционно, в порядке
SPEAKER_00, SPEAKER_01). Иначе после прогона юзер обычно скажет какие имена,
и модель сама `sed`-нет (или Read + Edit) готовый markdown.

### Кэш
По дефолту `~/Library/Caches/vidscribe/` (macOS). Аудио + STT + кадры
переживают между запусками — повторный `correct` берёт из кэша,
пересчитывает только correction loop с новыми параметрами.

### Логи
`.vidscribe/logs/` (или соответствующее в global cache dir). Symlink
`latest.log`. Каждый запуск пишет timestamped события всех 9 этапов pipeline.

### Frame extraction fallback (зависания на проблемных видео)

Запись с встреч / Zoom / Telemost иногда содержит битые кадры — ffmpeg
застревает на конкретном таймштампе. Vidscribe имеет 3-уровневый fallback:

1. **scene-detect** (самый умный): scene-detection + uniform sampling
2. **sample-only** (если scene-detect застрял): только uniform sampling с
   `-skip_frame nokey -err_detect ignore_err -fflags +discardcorrupt -an`
3. **seek-based** (если sample-only тоже застрял): один ffmpeg на каждый
   таймштамп с per-call timeout 15s — битые сегменты пропускаются индивидуально

Между уровнями watchdog ждёт 120 секунд застоя `out_time_us`, потом kill +
переход на следующий fallback. Полный путь занимает ~4 минуты ожидания.

**Если юзер заранее знает что видео битое** (или после первого зависания) —
используй `--frames-strategy seek` чтобы сразу пропустить scene-detect и
sample-only:
```bash
vidscribe pipeline VIDEO --frames-strategy seek ...DEFAULT флаги
```

Это медленнее на здоровом видео (per-frame ffmpeg = 1-2с × N таймштампов),
но устойчиво на битых.

В лог будут видны переключения:
```
[5/9] FFmpeg stuck at 19:04 for 120s — killing subprocess
[5/9] Frames: scene-detect failed/timed-out, retrying with sample-only strategy
[5/9] Frames: sample-only stalled too, switching to seek-based per-frame extraction
[5/9] Frames (seek): 1:30:00 processed (58%) | 540 ok | 3 skipped
[5/9] Frames done in 480.2s | 845 frames (seek-based per-frame fallback)
```

## Flag reference (для исключений)

```
--correction-mode single|mix         (default: single при ручном вызове, но skill ставит mix для DEFAULT)
--provider claude|codex|ollama       (single mode)
--model MODEL
--text-provider PROVIDER             (mix mode, Pass 1)
--text-model MODEL
--visual-provider PROVIDER           (mix mode, Pass 2)
--visual-model MODEL
--screen-context off|inline|aside|footer
--whisper-model noscribe-precise|noscribe-fast|large-v3|medium
--chunk-strategy speaker|time|scene
--frames-strategy auto|scene-detect|sample-only|seek    (default: auto = 3-level fallback; seek = robust for broken videos)
--speakers "Name1,Name2"
--out FILE
--no-cache (или --no-cache STAGE)
--cache-dir PATH
--quiet                              (отключить прогресс на stderr)
--no-log
--log-file PATH
```

## Subcommands

| Команда | Когда использовать |
|---------|---------------------|
| `pipeline VIDEO` | первый запуск — full processing |
| `correct VIDEO` | re-run только correction loop из кэша (новый провайдер/режим) |
| `transcribe VIDEO` | только STT + diarization |
| `extract VIDEO` | только audio + keyframes |
| `cache list [VIDEO]` | смотреть что в кэше |
| `cache clear [VIDEO]` | очистить кэш |
| `logs [--follow|--list|--path]` | смотреть логи |

## Editing a finished transcript

**Не делать отдельной CLI-команды для этого.** Если юзер просит правки в готовом
`.md` файле — модель сама правит файл через Read/Edit tool. Простые замены
(`s00 → Алексей`) — `sed`. Семантические правки — Read + Edit. Это локальная
работа модели, не subprocess.

## Examples for the model invoking this skill

| User says | You run |
|---|---|
| «замени s00 на Алексея в transcript X.md» | `sed -i '' 's/\*\*s00\*\*/**Алексей**/g' X.md` (or Read + Edit) |
| «удали все короткие реплики из X.md» | Read X.md, обработай в памяти, Write обратно |
| «распознай видео ~/V/meeting.mp4» | DEFAULT recipe with `--out ~/V/meeting-transcript.md` |
| «расшифруй встречу, спикеры Иван и Анна» | DEFAULT + `--speakers "Иван,Анна"` |
| «транскрибируй лекцию locally only» | `--correction-mode single --provider ollama --model qwen2-vl:7b --screen-context off` |
| «обработай скринкаст с детальным описанием экрана» | DEFAULT (it's already this) |
| «видео из Zoom/Telemost зависает на ffmpeg / битое» | DEFAULT + `--frames-strategy seek` |
| «перегенерируй с claude opus» | `correct VIDEO --provider claude --model opus --out OUT.md` |
| «быстрый черновой транскрипт» | `--correction-mode single --provider codex --screen-context off --whisper-model noscribe-fast` |
