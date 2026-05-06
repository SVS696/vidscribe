"""Pipeline artefact cache."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from rich.console import Console


class Cache:
    """File-backed cache for pipeline artefacts."""

    def __init__(
        self,
        root: Path,
        *,
        disabled_stages: set[str] | None = None,
        console: Console | None = None,
    ) -> None:
        self.root = Path(root)
        self.disabled_stages = disabled_stages or set()
        self.console = console or Console()

    def get(self, stage: str, key: str) -> Any | None:
        """Return a cached artefact for stage/key, or None on miss/bypass."""

        if stage in self.disabled_stages:
            return None

        stage_dir = self._stage_dir(stage, key)
        if not stage_dir.exists():
            return None

        artefact_path = self._artefact_path(stage_dir)
        if artefact_path is None:
            return None

        self.console.log(f"cache hit: {stage}/{key}")
        if artefact_path.suffix == ".json":
            return json.loads(artefact_path.read_text(encoding="utf-8"))
        if artefact_path.suffix in {".txt", ".md"}:
            return artefact_path.read_text(encoding="utf-8")
        if artefact_path.suffix == ".bin":
            return artefact_path.read_bytes()
        return artefact_path

    def set(self, stage: str, key: str, artefact: Any) -> Path:
        """Persist an artefact under .vidscribe/cache/{key}/{stage}/."""

        if stage in self.disabled_stages:
            return self._stage_dir(stage, key)

        stage_dir = self._stage_dir(stage, key)
        stage_dir.mkdir(parents=True, exist_ok=True)

        for existing in stage_dir.iterdir():
            if existing.is_file():
                existing.unlink()

        if isinstance(artefact, Path):
            destination = stage_dir / artefact.name
            if artefact.is_file():
                shutil.copy2(artefact, destination)
            else:
                destination.write_text(str(artefact), encoding="utf-8")
            return destination

        if isinstance(artefact, bytes):
            path = stage_dir / "artefact.bin"
            path.write_bytes(artefact)
            return path

        if isinstance(artefact, str):
            suffix = ".md" if stage == "final" else ".txt"
            path = stage_dir / f"artefact{suffix}"
            path.write_text(artefact, encoding="utf-8")
            return path

        path = stage_dir / "artefact.json"
        path.write_text(
            json.dumps(_jsonable(artefact), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def key_for(self, stage: str, **inputs: Any) -> str:
        """Build a deterministic sha256 key from a stage and structured inputs."""

        payload = {"stage": stage, "inputs": _jsonable(inputs)}
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _stage_dir(self, stage: str, key: str) -> Path:
        return self.root / "cache" / key / stage

    @staticmethod
    def _artefact_path(stage_dir: Path) -> Path | None:
        preferred = [
            stage_dir / "artefact.json",
            stage_dir / "artefact.txt",
            stage_dir / "artefact.md",
            stage_dir / "artefact.bin",
        ]
        for path in preferred:
            if path.exists():
                return path
        return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return _path_fingerprint(value)
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _path_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.expanduser()
    if resolved.is_file():
        digest = hashlib.sha256()
        with resolved.open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "path": str(resolved),
            "sha256": digest.hexdigest(),
        }
    return {"path": str(resolved)}
