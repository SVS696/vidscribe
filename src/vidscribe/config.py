"""Application configuration loading."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator


ChunkStrategy = Literal["speaker", "time", "scene"]
ProviderName = Literal["claude", "codex", "ollama"]
CacheStage = Literal[
    "audio",
    "frames",
    "asr",
    "diar",
    "stt",
    "chunks",
    "speakers",
    "corrected",
    "final",
]


class AppConfig(BaseModel):
    """Runtime configuration for the transcription pipeline."""

    model_config = ConfigDict(extra="forbid")

    provider: ProviderName = "codex"
    model: str | None = None
    chunk_strategy: ChunkStrategy = "speaker"
    frame_rate: float = Field(default=0.1, gt=0)
    whisper_model: str = "noscribe-precise"
    language: str = "ru"
    hf_token: str | None = None
    cache_dir: Path = Path(".vidscribe")
    no_cache: tuple[CacheStage, ...] = ()
    speakers: tuple[str, ...] = ()

    @field_validator("provider", mode="before")
    @classmethod
    def normalize_provider(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("no_cache", mode="before")
    @classmethod
    def normalize_no_cache(cls, value: Any) -> Any:
        if isinstance(value, str):
            return (value.strip().lower(),)
        if isinstance(value, (list, tuple)):
            return tuple(item.strip().lower() if isinstance(item, str) else item for item in value)
        return value


ENV_MAPPING = {
    "VIDSCRIBE_PROVIDER": "provider",
    "VIDSCRIBE_MODEL": "model",
    "HF_TOKEN": "hf_token",
}


def default_config_path() -> Path:
    """Return the conventional user config path."""

    return Path.home() / ".config" / "vidscribe" / "config.toml"


def load_config_file(path: Path | None = None) -> dict[str, Any]:
    """Load config values from TOML if the file exists."""

    config_path = path or default_config_path()
    if not config_path.exists():
        return {}

    with config_path.open("rb") as config_file:
        data = tomllib.load(config_file)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a TOML table: {config_path}")

    return data


def env_overrides(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Extract supported environment variable overrides."""

    source = env or os.environ
    values: dict[str, Any] = {}
    for env_name, field_name in ENV_MAPPING.items():
        value = source.get(env_name)
        if value:
            values[field_name] = value
    return values


def compact_overrides(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Drop unset CLI override values before applying precedence."""

    if not overrides:
        return {}
    return {key: value for key, value in overrides.items() if value is not None}


def load_config(
    *,
    config_path: Path | None = None,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> AppConfig:
    """Load config using defaults < file < environment < CLI overrides."""

    values = load_config_file(config_path)
    values.update(env_overrides(env))
    values.update(compact_overrides(overrides))
    return AppConfig.model_validate(values)
