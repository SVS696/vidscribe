from pathlib import Path

from typer.testing import CliRunner

from vidscribe.cli import app
from vidscribe.config import AppConfig, load_config


def test_app_config_defaults() -> None:
    config = AppConfig()

    assert config.provider == "codex"
    assert config.model == "gpt-5.5"
    assert config.chunk_strategy == "speaker"
    assert config.frame_rate == 0.1
    assert config.whisper_model == "noscribe-precise"
    assert config.language == "ru"
    assert config.hf_token is None
    assert config.cache_dir == Path(".vidscribe")
    assert config.no_cache == ()
    assert config.speakers == ()


def test_env_overrides_supported_values(monkeypatch) -> None:
    monkeypatch.setenv("VIDSCRIBE_PROVIDER", "claude")
    monkeypatch.setenv("VIDSCRIBE_MODEL", "sonnet")
    monkeypatch.setenv("HF_TOKEN", "hf_test")

    config = load_config(config_path=Path("/missing/config.toml"))

    assert config.provider == "claude"
    assert config.model == "sonnet"
    assert config.hf_token == "hf_test"


def test_loads_optional_config_file(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("VIDSCRIBE_PROVIDER", raising=False)
    monkeypatch.delenv("VIDSCRIBE_MODEL", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                'provider = "ollama"',
                'model = "qwen2-vl:7b"',
                'chunk_strategy = "scene"',
                "frame_rate = 0.25",
                'whisper_model = "noscribe-fast"',
                'language = "en"',
                'hf_token = "from_file"',
                'cache_dir = "/tmp/vidscribe-cache"',
                'no_cache = ["stt", "frames"]',
                'speakers = ["Иван", "Алиса"]',
            ]
        )
    )

    config = load_config(config_path=config_path)

    assert config.provider == "ollama"
    assert config.model == "qwen2-vl:7b"
    assert config.chunk_strategy == "scene"
    assert config.frame_rate == 0.25
    assert config.whisper_model == "noscribe-fast"
    assert config.language == "en"
    assert config.hf_token == "from_file"
    assert config.cache_dir == Path("/tmp/vidscribe-cache")
    assert config.no_cache == ("stt", "frames")
    assert config.speakers == ("Иван", "Алиса")


def test_cli_overrides_env_and_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VIDSCRIBE_PROVIDER", "claude")
    monkeypatch.setenv("VIDSCRIBE_MODEL", "sonnet")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                'provider = "ollama"',
                'model = "qwen2-vl:7b"',
                'chunk_strategy = "scene"',
            ]
        )
    )

    config = load_config(
        config_path=config_path,
        overrides={
            "provider": "codex",
            "model": "gpt-5.5",
            "chunk_strategy": "time",
        },
    )

    assert config.provider == "codex"
    assert config.model == "gpt-5.5"
    assert config.chunk_strategy == "time"


def test_cli_callback_accepts_config_overrides() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "--provider",
            "claude",
            "--model",
            "sonnet",
            "--no-cache",
            "stt",
            "--no-cache",
            "frames",
            "--speakers",
            "Иван, Алиса",
        ],
    )

    assert result.exit_code == 0
