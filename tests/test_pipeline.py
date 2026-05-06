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
