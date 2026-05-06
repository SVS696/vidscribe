from pathlib import Path

import pytest

from vidscribe.cache import Cache
from vidscribe.chunker import Chunk
from vidscribe.pipeline import CorrectionError, correct_chunks
from vidscribe.provider import ProviderResponse
from vidscribe.stt import SttSegment


class FakeProvider:
    def __init__(self, payloads: list[dict] | None = None, *, text: str = "") -> None:
        self.payloads = payloads or []
        self.text = text
        self.calls = []
        self.model = "fake-model"

    def correct(self, prompt: str, frame_paths: list[Path], timeout: int):
        self.calls.append((prompt, frame_paths, timeout))
        payload = self.payloads.pop(0) if self.payloads else {}
        text = self.text or payload.get("corrected_text", "")
        return ProviderResponse(
            text=text,
            raw_json=payload,
            cost_estimate=0.01,
            duration_s=0.2,
        )


def segment(start: float, end: float, speaker: str, text: str) -> SttSegment:
    return SttSegment(start=start, end=end, speaker=speaker, text=text, words=[])


def chunk(
    idx: int,
    start: float,
    end: float,
    speaker: str,
    text: str,
    *,
    frame_paths: list[Path] | None = None,
) -> Chunk:
    return Chunk(
        idx=idx,
        start=start,
        end=end,
        segments=[segment(start, end, speaker, text)],
        frame_paths=frame_paths or [],
        surrounding_context="",
    )


def test_correct_chunks_accumulates_glossary_between_provider_calls() -> None:
    chunks = [
        chunk(0, 0, 5, "SPEAKER_00", "опен эй ай"),
        chunk(1, 5, 10, "SPEAKER_01", "продолжаем про модель"),
    ]
    provider = FakeProvider(
        [
            {
                "corrected_text": "OpenAI",
                "glossary_delta": {"OpenAI": "canonical spelling"},
                "notes": "fixed term",
            },
            {
                "corrected_text": "Продолжаем про модель",
                "glossary_delta": {"GPT": "model family"},
                "notes": "",
            },
        ]
    )

    corrected = correct_chunks(
        chunks,
        provider,
        {"SPEAKER_00": "Иван", "SPEAKER_01": "Алиса"},
        cache=None,
    )

    assert [item.corrected_text for item in corrected] == [
        "OpenAI",
        "Продолжаем про модель",
    ]
    assert corrected[0].glossary_delta == {"OpenAI": "canonical spelling"}
    assert "Known glossary:\n- none" in provider.calls[0][0]
    assert "- OpenAI: canonical spelling" in provider.calls[1][0]
    assert "SPEAKER_00 (Иван)" in provider.calls[0][0]
    assert provider.calls[0][2] == 300


def test_correct_chunks_accepts_corrected_segments() -> None:
    provider = FakeProvider(
        [
            {
                "segments": [
                    {
                        "start": 0,
                        "end": 2,
                        "speaker": "SPEAKER_00",
                        "corrected_text": "Иван говорит.",
                    },
                    {
                        "start": 2,
                        "end": 4,
                        "speaker": "SPEAKER_01",
                        "corrected_text": "Алиса отвечает.",
                    },
                ],
                "glossary_delta": {},
                "notes": "",
            }
        ]
    )
    item = Chunk(
        idx=0,
        start=0,
        end=4,
        segments=[
            segment(0, 2, "SPEAKER_00", "иван говорит"),
            segment(2, 4, "SPEAKER_01", "алиса отвечает"),
        ],
    )

    corrected = correct_chunks(
        [item],
        provider,
        {"SPEAKER_00": "Иван", "SPEAKER_01": "Алиса"},
        cache=None,
    )

    assert corrected[0].corrected_text == "Иван говорит.\nАлиса отвечает."
    assert [segment.speaker for segment in corrected[0].segments] == [
        "SPEAKER_00",
        "SPEAKER_01",
    ]


def test_correct_chunks_passes_absolute_frame_paths_to_prompt_and_provider(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    provider = FakeProvider(
        [
            {
                "corrected_text": "Fixed",
                "glossary_delta": {},
                "notes": "",
            }
        ]
    )

    correct_chunks(
        [chunk(0, 0, 5, "SPEAKER_00", "raw", frame_paths=[Path("frames/a.jpg")])],
        provider,
        {"SPEAKER_00": "Иван"},
        cache=None,
    )

    prompt, frame_paths, _timeout = provider.calls[0]
    assert frame_paths == [(tmp_path / "frames/a.jpg").resolve()]
    assert str((tmp_path / "frames/a.jpg").resolve()) in prompt


def test_correct_chunks_caches_each_chunk_with_glossary_snapshot(tmp_path) -> None:
    chunks = [
        chunk(0, 0, 5, "SPEAKER_00", "первый"),
        chunk(1, 5, 10, "SPEAKER_00", "второй"),
    ]
    cache = Cache(tmp_path)
    first_provider = FakeProvider(
        [
            {
                "corrected_text": "Первый",
                "glossary_delta": {"Term": "from first chunk"},
                "notes": "",
            },
            {
                "corrected_text": "Второй",
                "glossary_delta": {},
                "notes": "",
            },
        ]
    )

    first = correct_chunks(chunks, first_provider, {"SPEAKER_00": "Иван"}, cache)
    second_provider = FakeProvider(
        [
            {
                "corrected_text": "Wrong",
                "glossary_delta": {},
                "notes": "",
            }
        ]
    )
    second = correct_chunks(chunks, second_provider, {"SPEAKER_00": "Иван"}, cache)

    assert [item.corrected_text for item in first] == ["Первый", "Второй"]
    assert [item.corrected_text for item in second] == ["Первый", "Второй"]
    assert second_provider.calls == []


def test_correct_chunks_cache_key_includes_speaker_map(tmp_path) -> None:
    cache = Cache(tmp_path)
    chunks = [chunk(0, 0, 5, "SPEAKER_00", "сырой текст")]
    first_provider = FakeProvider([{"corrected_text": "Иван говорит"}])
    second_provider = FakeProvider([{"corrected_text": "Алиса говорит"}])

    first = correct_chunks(chunks, first_provider, {"SPEAKER_00": "Иван"}, cache)
    second = correct_chunks(chunks, second_provider, {"SPEAKER_00": "Алиса"}, cache)

    assert [item.corrected_text for item in first] == ["Иван говорит"]
    assert [item.corrected_text for item in second] == ["Алиса говорит"]
    assert len(second_provider.calls) == 1
    assert "SPEAKER_00 (Алиса): сырой текст" in second_provider.calls[0][0]


def test_correct_chunks_can_namespace_cache_under_video_key(tmp_path) -> None:
    cache = Cache(tmp_path)
    provider = FakeProvider([{"corrected_text": "Fixed"}])

    correct_chunks(
        [chunk(0, 0, 5, "SPEAKER_00", "raw")],
        provider,
        {"SPEAKER_00": "Иван"},
        cache,
        namespace_key="video-key",
    )

    corrected_root = tmp_path / "cache" / "video-key"
    assert corrected_root.exists()
    assert list(corrected_root.glob("*/corrected/artefact.json"))


def test_correct_chunks_accepts_inner_json_from_provider_text() -> None:
    provider = FakeProvider(
        [{"result": '{"corrected_text": "Fixed", "glossary_delta": {}, "notes": ""}'}],
        text='{"corrected_text": "Fixed", "glossary_delta": {}, "notes": ""}',
    )

    corrected = correct_chunks(
        [chunk(0, 0, 5, "SPEAKER_00", "raw")],
        provider,
        {"SPEAKER_00": "Иван"},
        cache=None,
    )

    assert corrected[0].corrected_text == "Fixed"


def test_correct_chunks_rejects_non_json_provider_text() -> None:
    provider = FakeProvider([{"result": "not json"}], text="not json")

    with pytest.raises(CorrectionError, match="not valid JSON"):
        correct_chunks(
            [chunk(0, 0, 5, "SPEAKER_00", "raw")],
            provider,
            {"SPEAKER_00": "Иван"},
            cache=None,
        )


def test_correct_chunks_rejects_missing_corrected_text() -> None:
    provider = FakeProvider(
        [{"result": '{"glossary_delta": {}, "notes": ""}'}],
        text='{"glossary_delta": {}, "notes": ""}',
    )

    with pytest.raises(CorrectionError, match="corrected_text"):
        correct_chunks(
            [chunk(0, 0, 5, "SPEAKER_00", "raw")],
            provider,
            {"SPEAKER_00": "Иван"},
            cache=None,
        )


def test_correct_chunks_rejects_empty_provider_response() -> None:
    provider = FakeProvider([], text="")

    with pytest.raises(CorrectionError, match="correction JSON object"):
        correct_chunks(
            [chunk(0, 0, 5, "SPEAKER_00", "raw")],
            provider,
            {"SPEAKER_00": "Иван"},
            cache=None,
        )


def test_correct_chunks_rejects_non_object_json_text() -> None:
    provider = FakeProvider([{"result": "[]"}], text="[]")

    with pytest.raises(CorrectionError, match="correction JSON object"):
        correct_chunks(
            [chunk(0, 0, 5, "SPEAKER_00", "raw")],
            provider,
            {"SPEAKER_00": "Иван"},
            cache=None,
        )
