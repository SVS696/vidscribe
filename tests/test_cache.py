from pathlib import Path

from rich.console import Console

from vidscribe.cache import Cache


def test_cache_round_trips_json_artefact(tmp_path) -> None:
    cache = Cache(tmp_path)
    key = cache.key_for("stt", video="abc", model="noscribe-precise")

    path = cache.set("stt", key, {"segments": [{"text": "hello"}]})
    cached = cache.get("stt", key)

    assert path == tmp_path / "cache" / key / "stt" / "artefact.json"
    assert cached == {"segments": [{"text": "hello"}]}


def test_cache_key_is_stable_for_structured_inputs(tmp_path) -> None:
    cache = Cache(tmp_path)

    first = cache.key_for("frames", b=2, a={"z": [Path("video.mp4"), b"abc"], 1: True})
    second = cache.key_for("frames", a={1: True, "z": [Path("video.mp4"), b"abc"]}, b=2)
    different_stage = cache.key_for("stt", a={"z": [Path("video.mp4"), b"abc"]}, b=2)

    assert first == second
    assert first != different_stage
    assert len(first) == 64


def test_cache_hashes_file_path_with_content(tmp_path) -> None:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"first")
    cache = Cache(tmp_path)

    first = cache.key_for("audio", video_path=source)
    source.write_bytes(b"second")
    second = cache.key_for("audio", video_path=source)

    assert first != second


def test_cache_round_trips_text_and_binary_artefacts(tmp_path) -> None:
    cache = Cache(tmp_path)
    text_key = cache.key_for("corrected", idx=1)
    binary_key = cache.key_for("audio", video="abc")

    cache.set("corrected", text_key, "Corrected transcript")
    cache.set("audio", binary_key, b"RIFF")

    assert cache.get("corrected", text_key) == "Corrected transcript"
    assert cache.get("audio", binary_key) == b"RIFF"


def test_cache_copies_file_artefact(tmp_path) -> None:
    source = tmp_path / "frames.json"
    source.write_text('{"frames": []}', encoding="utf-8")
    cache = Cache(tmp_path / ".vidscribe")
    key = cache.key_for("frames", video="abc")

    path = cache.set("frames", key, source)

    assert path == tmp_path / ".vidscribe" / "cache" / key / "frames" / "frames.json"
    assert path.read_text(encoding="utf-8") == '{"frames": []}'


def test_cache_hit_logs_through_rich(tmp_path) -> None:
    log_file = tmp_path / "rich.log"
    with log_file.open("w", encoding="utf-8") as output:
        console = Console(file=output, force_terminal=False, width=120)
        cache = Cache(tmp_path / ".vidscribe", console=console)
        key = cache.key_for("chunks", video="abc")

        cache.set("chunks", key, {"chunks": []})
        assert cache.get("chunks", key) == {"chunks": []}

    assert "cache hit: chunks/" in log_file.read_text(encoding="utf-8")


def test_disabled_stage_bypasses_get_and_set(tmp_path) -> None:
    cache = Cache(tmp_path, disabled_stages={"stt"})
    key = cache.key_for("stt", video="abc")

    cache.set("stt", key, {"segments": []})

    assert cache.get("stt", key) is None
    assert not (tmp_path / "cache" / key / "stt").exists()
