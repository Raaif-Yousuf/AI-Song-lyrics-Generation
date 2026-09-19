"""Experiment configuration schema and YAML loading.

An experiment is fully described by an :class:`ExperimentConfig`, which nests
a :class:`DataConfig`, a tokenizer specification, a :class:`ModelConfig` and a
:class:`TrainConfig`. Configurations are authored as YAML files under
``configs/`` and loaded with :func:`load_config`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

_VALID_MODEL_KINDS = ("lstm", "gru", "transformer")
_VALID_TOKENIZER_TYPES = ("char", "bpe")


class ConfigError(ValueError):
    """Raised when an experiment configuration is missing or invalid."""


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing required key '{key}' in {context}")
    return mapping[key]


def _as_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{context} must be a mapping, got {type(value).__name__}")
    return value


@dataclass
class DataConfig:
    """Location of the prepared train/val/test dataset."""

    dir: str = "data/processed"

    @property
    def train_path(self) -> str:
        return str(Path(self.dir) / "train.jsonl")

    @property
    def val_path(self) -> str:
        return str(Path(self.dir) / "val.jsonl")

    @property
    def test_path(self) -> str:
        return str(Path(self.dir) / "test.jsonl")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DataConfig:
        raw = _as_mapping(raw, "data config")
        unknown = set(raw) - {"dir"}
        if unknown:
            raise ConfigError(f"unknown data config keys: {sorted(unknown)}")
        return cls(dir=raw.get("dir", "data/processed"))

    def to_dict(self) -> dict[str, Any]:
        return {"dir": self.dir}


@dataclass
class TokenizerConfig:
    """Tokenizer choice for an experiment."""

    type: str
    vocab_size: int | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TokenizerConfig:
        raw = _as_mapping(raw, "tokenizer config")
        tokenizer_type = _require(raw, "type", "tokenizer config")
        if tokenizer_type not in _VALID_TOKENIZER_TYPES:
            raise ConfigError(
                f"tokenizer type must be one of {_VALID_TOKENIZER_TYPES}, got {tokenizer_type!r}"
            )
        vocab_size = raw.get("vocab_size")
        if tokenizer_type == "bpe" and vocab_size is None:
            raise ConfigError("tokenizer config of type 'bpe' requires 'vocab_size'")
        unknown = set(raw) - {"type", "vocab_size"}
        if unknown:
            raise ConfigError(f"unknown tokenizer config keys: {sorted(unknown)}")
        return cls(type=tokenizer_type, vocab_size=vocab_size)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type}
        if self.vocab_size is not None:
            out["vocab_size"] = self.vocab_size
        return out


@dataclass
class ModelConfig:
    """Model architecture choice plus free-form hyperparameters.

    `kind` selects the architecture (see `_VALID_MODEL_KINDS`); every other
    key in the YAML `model` mapping is passed through as a hyperparameter, so
    each architecture can define its own knobs without changing this schema.
    """

    kind: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelConfig:
        raw = _as_mapping(raw, "model config")
        kind = _require(raw, "kind", "model config")
        if kind not in _VALID_MODEL_KINDS:
            raise ConfigError(f"model kind must be one of {_VALID_MODEL_KINDS}, got {kind!r}")
        hyperparameters = {k: v for k, v in raw.items() if k != "kind"}
        return cls(kind=kind, hyperparameters=hyperparameters)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, **self.hyperparameters}


@dataclass
class TrainConfig:
    """Training loop hyperparameters."""

    steps: int
    batch_size: int
    block_size: int
    lr: float
    weight_decay: float = 0.0
    warmup_steps: int = 0
    grad_clip: float = 1.0
    eval_interval: int = 200
    eval_batches: int = 20
    artist_dropout: float = 0.1
    seed: int = 1337
    device: str = "cpu"
    num_threads: int | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainConfig:
        raw = _as_mapping(raw, "train config")
        for key in ("steps", "batch_size", "block_size", "lr"):
            _require(raw, key, "train config")
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ConfigError(f"unknown train config keys: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Full description of a single training experiment."""

    name: str
    data: DataConfig
    tokenizer: TokenizerConfig
    model: ModelConfig
    train: TrainConfig
    out_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "data": self.data.to_dict(),
            "tokenizer": self.tokenizer.to_dict(),
            "model": self.model.to_dict(),
            "train": self.train.to_dict(),
            "out_dir": self.out_dir,
        }


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an :class:`ExperimentConfig` from a YAML file.

    Raises :class:`ConfigError` (with a message naming the file and the
    offending key) if the file is missing, is not valid YAML, is not a
    mapping, or is missing/misusing a required key.
    """
    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"could not read config file {config_path}: {exc}") from exc

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc

    context = f"config {config_path}"
    raw = _as_mapping(raw, context)

    name = _require(raw, "name", context)
    out_dir = _require(raw, "out_dir", context)

    try:
        data = DataConfig.from_dict(raw.get("data", {}))
        tokenizer = TokenizerConfig.from_dict(_require(raw, "tokenizer", context))
        model = ModelConfig.from_dict(_require(raw, "model", context))
        train = TrainConfig.from_dict(_require(raw, "train", context))
    except ConfigError as exc:
        raise ConfigError(f"in {context}: {exc}") from exc

    known_top_level = {"name", "data", "tokenizer", "model", "train", "out_dir"}
    unknown = set(raw) - known_top_level
    if unknown:
        raise ConfigError(f"unknown keys in {context}: {sorted(unknown)}")

    return ExperimentConfig(
        name=name,
        data=data,
        tokenizer=tokenizer,
        model=model,
        train=train,
        out_dir=out_dir,
    )
