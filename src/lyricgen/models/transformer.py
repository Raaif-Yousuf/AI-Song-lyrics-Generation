"""Small pre-LayerNorm decoder-only transformer (GPT-style)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import LyricsModel


class _CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention using scaled dot-product attention."""

    def __init__(self, d_model: int, n_head: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, d_model = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        dropout_p = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
        y = y.transpose(1, 2).contiguous().view(batch, length, d_model)
        return self.proj(y)


class _MLP(nn.Module):
    """Position-wise feed-forward block with GELU activation."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class _Block(nn.Module):
    """Pre-LayerNorm transformer block: attention then MLP, both residual."""

    def __init__(self, d_model: int, n_head: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _CausalSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = _MLP(d_model, dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.resid_dropout(self.attn(self.ln1(x)))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLyricsModel(LyricsModel):
    """GPT-style decoder-only transformer with artist conditioning.

    Learned positional embeddings, causal self-attention, and an artist
    embedding added to every position so the whole sequence is conditioned
    on the chosen artist. The output projection shares weights with the
    token embedding. `state` is unused and always returned as None; the
    caller is responsible for cropping input sequences to `block_size`.
    """

    def __init__(
        self,
        vocab_size: int,
        n_artists: int,
        n_layer: int = 4,
        n_head: int = 4,
        d_model: int = 256,
        block_size: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.config = {
            "kind": "transformer",
            "vocab_size": vocab_size,
            "n_artists": n_artists,
            "n_layer": n_layer,
            "n_head": n_head,
            "d_model": d_model,
            "block_size": block_size,
            "dropout": dropout,
        }
        self.max_context = block_size

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(block_size, d_model)
        self.artist_embed = nn.Embedding(n_artists, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([_Block(d_model, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_embed.weight

    def forward(
        self,
        tokens: Tensor,
        artists: Tensor,
        state: Any = None,
    ) -> tuple[Tensor, None]:
        batch, length = tokens.shape
        if length > self.max_context:
            raise ValueError(f"sequence length {length} exceeds block_size {self.max_context}")
        positions = torch.arange(length, device=tokens.device)
        x = self.token_embed(tokens) + self.pos_embed(positions).unsqueeze(0)
        x = x + self.artist_embed(artists).unsqueeze(1)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, None
