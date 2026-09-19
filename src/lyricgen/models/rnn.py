"""Recurrent (LSTM/GRU) baseline mirroring the original Keras model."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .base import LyricsModel

_RNN_CLASSES: dict[str, type[nn.Module]] = {"lstm": nn.LSTM, "gru": nn.GRU}


class RNNLyricsModel(LyricsModel):
    """Stacked recurrent model: embeddings -> RNN layers -> linear head.

    Mirrors the shape of the original project (LSTM 256 -> LSTM 128 ->
    softmax) but concatenates an artist embedding to the token embedding at
    every step so generation can be conditioned on an artist. `state` is a
    list with one entry per RNN layer, so generation can stream a sequence
    one chunk at a time and resume exactly where it left off.
    """

    def __init__(
        self,
        kind: str,
        vocab_size: int,
        n_artists: int,
        embed_dim: int = 64,
        artist_embed_dim: int = 32,
        hidden_sizes: tuple[int, ...] = (256, 128),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if kind not in _RNN_CLASSES:
            raise ValueError(f"unknown rnn kind: {kind!r}, expected one of {sorted(_RNN_CLASSES)}")
        rnn_cls = _RNN_CLASSES[kind]

        self.config = {
            "kind": kind,
            "vocab_size": vocab_size,
            "n_artists": n_artists,
            "embed_dim": embed_dim,
            "artist_embed_dim": artist_embed_dim,
            "hidden_sizes": list(hidden_sizes),
            "dropout": dropout,
        }
        self.max_context = None

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.artist_embed = nn.Embedding(n_artists, artist_embed_dim)

        input_size = embed_dim + artist_embed_dim
        self.layers = nn.ModuleList()
        for hidden_size in hidden_sizes:
            self.layers.append(rnn_cls(input_size, hidden_size, batch_first=True))
            input_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(input_size, vocab_size)

    def forward(
        self,
        tokens: Tensor,
        artists: Tensor,
        state: list[Any] | None = None,
    ) -> tuple[Tensor, list[Any]]:
        batch, length = tokens.shape
        x = self.token_embed(tokens)
        artist_vec = self.artist_embed(artists).unsqueeze(1).expand(batch, length, -1)
        x = torch.cat([x, artist_vec], dim=-1)

        new_state: list[Any] = []
        for i, layer in enumerate(self.layers):
            layer_state = state[i] if state is not None else None
            x, layer_new_state = layer(x, layer_state)
            new_state.append(layer_new_state)
            x = self.dropout(x)

        logits = self.head(x)
        return logits, new_state
