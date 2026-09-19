"""Small pre-LayerNorm decoder-only transformer (GPT-style)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import LyricsModel

_LayerKV = tuple[Tensor, Tensor]


@dataclass
class TransformerCache:
    """Per-layer key/value cache for incremental (one-step-at-a-time) decoding.

    `keys[i]` / `values[i]` hold the cached keys/values for block `i`, each
    shaped `[batch, n_head, cached_length, head_dim]`. `length` is the
    number of cached positions, used both to offset new position ids and
    to detect when the cache has filled `block_size` and must be rebuilt.
    """

    keys: list[Tensor]
    values: list[Tensor]
    length: int


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

    def forward(self, x: Tensor, cache: _LayerKV | None = None) -> tuple[Tensor, _LayerKV]:
        batch, length, d_model = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        cached_length = 0
        if cache is not None:
            cached_k, cached_v = cache
            cached_length = cached_k.shape[2]
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        dropout_p = self.dropout if self.training else 0.0
        if cached_length == 0:
            # No cache (or a fresh one): the new queries are the whole
            # sequence, so a plain causal mask applies.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
        else:
            # The new queries are the last `length` positions of the full
            # `cached_length + length` keys/values: unrestricted attention
            # over every cached position, plus a causal mask among the new
            # positions themselves. `is_causal` assumes a top-left-aligned
            # square mask and gives the wrong answer when query and key
            # lengths differ, so this mask is built explicitly.
            total_length = cached_length + length
            mask = torch.ones(length, total_length, dtype=torch.bool, device=x.device)
            mask[:, cached_length:] = torch.tril(
                torch.ones(length, length, dtype=torch.bool, device=x.device)
            )
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)
        y = y.transpose(1, 2).contiguous().view(batch, length, d_model)
        return self.proj(y), (k, v)


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

    def forward(self, x: Tensor, cache: _LayerKV | None = None) -> tuple[Tensor, _LayerKV]:
        attn_out, kv = self.attn(self.ln1(x), cache)
        x = x + self.resid_dropout(attn_out)
        x = x + self.mlp(self.ln2(x))
        return x, kv


class TransformerLyricsModel(LyricsModel):
    """GPT-style decoder-only transformer with artist conditioning.

    Learned positional embeddings, causal self-attention, and an artist
    embedding added to every position so the whole sequence is conditioned
    on the chosen artist. The output projection shares weights with the
    token embedding.

    By default (`use_cache=False`) `state` is unused and always returned
    as None, and the caller is responsible for cropping input sequences to
    `block_size`; this is the path used during training. Passing
    `use_cache=True` enables incremental decoding: `state` becomes a
    `TransformerCache` of per-layer keys/values that `lyricgen.sampling.generate`
    threads through one new token at a time instead of re-encoding the
    whole context on every step. See `forward` for details.
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
        state: TransformerCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, TransformerCache | None]:
        """Compute next-token logits, optionally reading/extending a KV cache.

        With the default `use_cache=False`, `tokens` must be the full
        context (at most `block_size` long) and this is a plain stateless
        forward pass: `state` is ignored and None is always returned,
        matching the base `LyricsModel` training interface exactly.

        With `use_cache=True`, `tokens` are only the *new* tokens to
        encode (the whole prompt on the first call, one token at a time
        after that). `state` is the `TransformerCache` from the previous
        call, or None to start a fresh cache. Positions are offset by the
        cache's current length so each new token gets its correct
        absolute position, and the returned cache is extended with the
        new tokens' keys/values.

        A cache can hold at most `block_size` positions (there is no
        position embedding beyond that); once it reaches `block_size`,
        this raises `ValueError` rather than silently overflowing.
        `lyricgen.sampling.generate` handles that by rebuilding a fresh
        cache from the most recent `block_size - 1` tokens, so it never
        hits this error. That rebuild uses fresh local positions
        (0..block_size-2) rather than the tokens' original absolute
        positions, so generation past `block_size` is not required (and
        is not guaranteed) to match a hypothetical uncached run token for
        token; within `block_size` tokens, cached and uncached forward
        passes are numerically equivalent (see tests).
        """
        batch, length = tokens.shape
        cache_length = state.length if state is not None else 0
        total_length = cache_length + length
        if total_length > self.max_context:
            raise ValueError(
                f"cache length {cache_length} plus {length} new tokens exceeds "
                f"block_size {self.max_context}; start a new cache"
            )
        positions = torch.arange(cache_length, total_length, device=tokens.device)
        x = self.token_embed(tokens) + self.pos_embed(positions).unsqueeze(0)
        x = x + self.artist_embed(artists).unsqueeze(1)
        x = self.drop(x)

        new_keys: list[Tensor] = []
        new_values: list[Tensor] = []
        for i, block in enumerate(self.blocks):
            layer_cache = (state.keys[i], state.values[i]) if state is not None else None
            x, (k, v) = block(x, layer_cache)
            if use_cache:
                new_keys.append(k)
                new_values.append(v)

        x = self.ln_f(x)
        logits = self.head(x)

        new_state = TransformerCache(new_keys, new_values, total_length) if use_cache else None
        return logits, new_state
