import importlib


MODULES = [
    "vidscribe",
    "vidscribe.cli",
    "vidscribe.config",
    "vidscribe.audio",
    "vidscribe.stt",
    "vidscribe.frames",
    "vidscribe.chunker",
    "vidscribe.speakers",
    "vidscribe.provider",
    "vidscribe.pipeline",
    "vidscribe.assembler",
    "vidscribe.cache",
    "vidscribe.prompts",
]


def test_imports_all_public_modules() -> None:
    for module in MODULES:
        importlib.import_module(module)

