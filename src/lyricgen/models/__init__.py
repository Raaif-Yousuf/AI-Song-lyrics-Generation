"""Model registry and factory for lyricgen."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import LyricsModel
from .rnn import RNNLyricsModel
from .transformer import TransformerLyricsModel

_REGISTRY: dict[str, Callable[[dict[str, Any]], LyricsModel]] = {
    "lstm": lambda cfg: RNNLyricsModel(kind="lstm", **cfg),
    "gru": lambda cfg: RNNLyricsModel(kind="gru", **cfg),
    "transformer": lambda cfg: TransformerLyricsModel(**cfg),
}


def build_model(config: dict[str, Any]) -> LyricsModel:
    """Build a `LyricsModel` from a config dict.

    `config["kind"]` selects the architecture (`"lstm"`, `"gru"` or
    `"transformer"`); the remaining keys, which must include `vocab_size`
    and `n_artists`, are passed as constructor keyword arguments.
    """
    cfg = dict(config)
    kind = cfg.pop("kind", None)
    if kind not in _REGISTRY:
        raise ValueError(f"unknown model kind: {kind!r}, expected one of {sorted(_REGISTRY)}")
    if "vocab_size" not in cfg or "n_artists" not in cfg:
        raise ValueError("config must include 'vocab_size' and 'n_artists'")
    return _REGISTRY[kind](cfg)


__all__ = ["LyricsModel", "RNNLyricsModel", "TransformerLyricsModel", "build_model"]
