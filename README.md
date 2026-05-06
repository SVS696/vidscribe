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

В разработке. План: `docs/plans/2026-05-06-mvp-pipeline.md`.
Запуск через `ralphex-codex` или `ralphex-claude`.

## Quick start (после реализации)

```bash
# полный прогон с параметрами по умолчанию (claude + sonnet)
vidscribe pipeline recording.mp4 --out transcript.md

# смена провайдера на codex (gpt-5.4)
vidscribe pipeline recording.mp4 --provider codex --out transcript.md

# локальная мультимодалка (бесплатно, медленнее)
vidscribe pipeline recording.mp4 --provider ollama --model qwen2-vl:7b

# ручные имена говорящих
vidscribe pipeline recording.mp4 --speakers "Иван,Алиса"

# перепрогнать только correction на других кадрах/модели
vidscribe correct recording.mp4 --provider codex
```

## Требования

- Python 3.11+
- ffmpeg (`brew install ffmpeg`)

**Веса моделей** — два варианта:

1. **Из noScribe.app (рекомендуется, без HF-токена)** — если приложение
   установлено, переиспользуем веса напрямую из бандла:
   - whisper (ct2): `/Applications/noScribe.app/Contents/Resources/models/{precise,fast}/`
   - pyannote: `/Applications/noScribe.app/Contents/Resources/pyannote/{config.yaml,segmentation,embedding}`
   - HF-токен **не нужен**, всё локально, никаких лицензионных договоров.

2. **Свежее с HuggingFace** — `huggingface-cli login` один раз для
   accept-terms на pyannote 3.1, дальше offline.
