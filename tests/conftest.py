from __future__ import annotations

from pathlib import Path

import pytest
from helpers import ARTIST_PATTERNS, make_tiny_config, write_synthetic_split

from lyricgen.config import ExperimentConfig


@pytest.fixture
def tiny_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    artists = list(ARTIST_PATTERNS)
    write_synthetic_split(data_dir / "train.jsonl", artists, 60)
    write_synthetic_split(data_dir / "val.jsonl", artists, 20)
    write_synthetic_split(data_dir / "test.jsonl", artists, 20)
    return data_dir


@pytest.fixture
def tiny_config(tmp_path: Path, tiny_data_dir: Path) -> ExperimentConfig:
    return make_tiny_config(tiny_data_dir, tmp_path / "run")
