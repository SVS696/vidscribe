from pathlib import Path

from vidscribe.cache import Cache
from vidscribe.frames import FrameInfo
from vidscribe.provider import ProviderResponse
from vidscribe.speakers import identify
from vidscribe.stt import SttResult, SttSegment


class FakeProvider:
    def __init__(self, raw_json=None, text: str = "") -> None:
        self.raw_json = raw_json or {}
        self.text = text
        self.calls = []

    def correct(self, prompt: str, frame_paths: list[Path], timeout: int):
        self.calls.append((prompt, frame_paths, timeout))
        return ProviderResponse(
            text=self.text,
            raw_json=self.raw_json,
            cost_estimate=None,
            duration_s=0.1,
        )


def segment(start: float, end: float, speaker: str, text: str) -> SttSegment:
    return SttSegment(start=start, end=end, speaker=speaker, text=text, words=[])


def stt_result() -> SttResult:
    return SttResult(
        segments=[
            segment(0, 5, "SPEAKER_01", "Алиса, расскажи про план"),
            segment(5, 10, "SPEAKER_00", "Да, сейчас расскажу"),
            segment(10, 15, "SPEAKER_01", "Спасибо"),
            segment(15, 20, "SPEAKER_00", "Продолжаю"),
            segment(20, 25, "SPEAKER_00", "Лишний пример не попадет"),
        ],
        language="ru",
        model="large-v3",
    )


def frame(ts: float, name: str) -> FrameInfo:
    return FrameInfo(ts=ts, path=Path(f"/frames/{name}.jpg"), scene_change=False)


def test_manual_speakers_map_positionally_without_provider_call() -> None:
    provider = FakeProvider(raw_json={"speakers": {"SPEAKER_00": "Wrong"}})

    speakers = identify(stt_result(), [], provider, manual="Иван, Алиса")

    assert speakers == {"SPEAKER_00": "Иван", "SPEAKER_01": "Алиса"}
    assert provider.calls == []


def test_provider_identifies_known_speaker_and_unknown_falls_back() -> None:
    provider = FakeProvider(raw_json={"speakers": {"SPEAKER_00": "Иван"}})
    frames = [frame(1, "a"), frame(7, "b"), frame(18, "c")]

    speakers = identify(stt_result(), frames, provider, timeout=12)

    assert speakers == {"SPEAKER_00": "Иван", "SPEAKER_01": "s01"}
    prompt, frame_paths, timeout = provider.calls[0]
    assert timeout == 12
    assert "SPEAKER_00" in prompt
    assert "SPEAKER_01" in prompt
    assert "Лишний пример не попадет" not in prompt
    assert frame_paths == [
        Path("/frames/a.jpg"),
        Path("/frames/b.jpg"),
        Path("/frames/c.jpg"),
    ]


def test_provider_receives_absolute_frame_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    provider = FakeProvider(raw_json={"speakers": {}})
    frames = [FrameInfo(ts=1, path=Path("frames/a.jpg"), scene_change=False)]

    identify(stt_result(), frames, provider)

    prompt, frame_paths, _timeout = provider.calls[0]
    assert frame_paths == [(tmp_path / "frames/a.jpg").resolve()]
    assert str((tmp_path / "frames/a.jpg").resolve()) in prompt


def test_provider_text_json_is_accepted() -> None:
    provider = FakeProvider(text='{"speakers": {"SPEAKER_01": "Алиса"}}')

    speakers = identify(stt_result(), [], provider)

    assert speakers == {"SPEAKER_00": "s00", "SPEAKER_01": "Алиса"}


def test_manual_mapping_by_speaker_id_overrides_provider() -> None:
    provider = FakeProvider(raw_json={"speakers": {"SPEAKER_00": "Иван"}})

    speakers = identify(
        stt_result(),
        [],
        provider,
        manual={"SPEAKER_01": "Алиса"},
    )

    assert speakers == {"SPEAKER_00": "Иван", "SPEAKER_01": "Алиса"}


def test_speaker_map_is_persisted_and_reused_from_cache(tmp_path) -> None:
    cache = Cache(tmp_path)
    provider = FakeProvider(raw_json={"speakers": {"SPEAKER_00": "Иван"}})

    first = identify(stt_result(), [], provider, cache=cache)
    second_provider = FakeProvider(raw_json={"speakers": {"SPEAKER_00": "Wrong"}})
    second = identify(stt_result(), [], second_provider, cache=cache)

    assert first == {"SPEAKER_00": "Иван", "SPEAKER_01": "s01"}
    assert second == first
    assert second_provider.calls == []


def test_empty_stt_returns_empty_mapping() -> None:
    provider = FakeProvider()

    speakers = identify(SttResult(segments=[], language="ru", model="large-v3"), [], provider)

    assert speakers == {}
    assert provider.calls == []
