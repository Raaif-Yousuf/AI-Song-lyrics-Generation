from __future__ import annotations

import json
from pathlib import Path

import pytest

from lyricgen.config import DataConfig, ExperimentConfig, ModelConfig, TokenizerConfig, TrainConfig

# Repeating, easily learnable per-artist patterns (not random noise), so a
# tiny model can drive training loss down within a handful of steps.
_ARTIST_PATTERNS = {
    "alpha": "ab cd ab cd ef ab cd ",
    "beta": "xy zw xy zw qr xy zw ",
}


def _pattern_text(artist: str, length: int) -> str:
    pattern = _ARTIST_PATTERNS[artist]
    repeated = pattern * (length // len(pattern) + 1)
    return repeated[:length]


def write_synthetic_split(path: Path, artists: list[str], n_records: int) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n_records):
            artist = artists[i % len(artists)]
            record = {"artist": artist, "text": _pattern_text(artist, 80)}
            fh.write(json.dumps(record) + "\n")


@pytest.fixture
def tiny_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    artists = list(_ARTIST_PATTERNS)
    write_synthetic_split(data_dir / "train.jsonl", artists, 60)
    write_synthetic_split(data_dir / "val.jsonl", artists, 20)
    write_synthetic_split(data_dir / "test.jsonl", artists, 20)
    return data_dir


def make_tiny_config(data_dir: Path, out_dir: Path, **overrides) -> ExperimentConfig:
    train_kwargs = {
        "steps": 40,
        "batch_size": 4,
        "block_size": 16,
        "lr": 0.01,
        "weight_decay": 0.0,
        "warmup_steps": 2,
        "grad_clip": 1.0,
        "eval_interval": 10,
        "eval_batches": 2,
        "artist_dropout": 0.1,
        "seed": 1337,
        "device": "cpu",
        "num_threads": 1,
    }
    train_kwargs.update(overrides.pop("train", {}))
    model_hyperparameters = overrides.pop(
        "model_hyperparameters",
        {"n_layer": 1, "n_head": 1, "d_model": 16, "block_size": 16, "dropout": 0.0},
    )
    return ExperimentConfig(
        name=overrides.pop("name", "tiny-test"),
        data=DataConfig(dir=str(data_dir)),
        tokenizer=TokenizerConfig(type=overrides.pop("tokenizer_type", "char")),
        model=ModelConfig(
            kind=overrides.pop("model_kind", "transformer"), hyperparameters=model_hyperparameters
        ),
        train=TrainConfig(**train_kwargs),
        out_dir=str(out_dir),
    )


@pytest.fixture
def tiny_config(tmp_path: Path, tiny_data_dir: Path) -> ExperimentConfig:
    return make_tiny_config(tiny_data_dir, tmp_path / "run")
