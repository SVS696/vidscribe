from pathlib import Path

import pytest


@pytest.fixture
def fixtures_path() -> Path:
    return Path(__file__).parent / "fixtures"

