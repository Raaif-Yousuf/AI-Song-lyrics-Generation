from __future__ import annotations

import pytest
import torch

from lyricgen.models import build_model
from lyricgen.sampling import generate

KINDS = ["lstm", "gru", "transformer"]


def _small_config(kind: str, vocab_size: int = 13, n_artists: int = 4) -> dict:
    if kind in ("lstm", "gru"):
        return dict(
            kind=kind,
            vocab_size=vocab_size,
            n_artists=n_artists,
            embed_dim=16,
            artist_embed_dim=6,
            hidden_sizes=(24, 12),
            dropout=0.0,
        )
    return dict(
        kind="transformer",
        vocab_size=vocab_size,
        n_artists=n_artists,
        n_layer=2,
        n_head=2,
        d_model=16,
        block_size=32,
        dropout=0.0,
    )


def _tiny_config(kind: str, vocab_size: int, n_artists: int, block_size: int) -> dict:
    if kind in ("lstm", "gru"):
        return dict(
            kind=kind,
            vocab_size=vocab_size,
            n_artists=n_artists,
            embed_dim=16,
            artist_embed_dim=4,
            hidden_sizes=(32, 16),
            dropout=0.0,
        )
    return dict(
        kind="transformer",
        vocab_size=vocab_size,
        n_artists=n_artists,
        n_layer=2,
        n_head=2,
        d_model=32,
        block_size=block_size,
        dropout=0.0,
    )


@pytest.mark.parametrize("kind", KINDS)
def test_build_model_forward_shapes_and_finite(kind: str) -> None:
    model = build_model(_small_config(kind))
    batch, length = 3, 7
    tokens = torch.randint(0, 13, (batch, length))
    artists = torch.randint(0, 4, (batch,))

    logits, state = model(tokens, artists)

    assert logits.shape == (batch, length, 13)
    assert torch.isfinite(logits).all()
    if kind == "transformer":
        assert state is None
        assert model.max_context == 32
    else:
        assert state is not None
        assert model.max_context is None
    assert model.num_parameters() > 0


@pytest.mark.parametrize("kind", KINDS)
def test_gradients_flow(kind: str) -> None:
    model = build_model(_small_config(kind))
    tokens = torch.randint(0, 13, (2, 6))
    artists = torch.randint(0, 4, (2,))
    targets = torch.randint(0, 13, (2, 6))

    logits, _ = model(tokens, artists)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 13), targets.reshape(-1))
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(torch.any(g != 0) for g in grads)


def test_registry_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        build_model({"kind": "nope", "vocab_size": 10, "n_artists": 2})


def test_registry_requires_vocab_and_artists() -> None:
    with pytest.raises(ValueError):
        build_model({"kind": "lstm", "vocab_size": 10})


def test_transformer_causality() -> None:
    model = build_model(_small_config("transformer"))
    model.eval()
    torch.manual_seed(0)
    tokens = torch.randint(0, 13, (1, 10))
    artists = torch.zeros(1, dtype=torch.long)

    with torch.no_grad():
        base_logits, _ = model(tokens, artists)

        changed = tokens.clone()
        changed[0, -1] = (changed[0, -1] + 1) % 13
        changed_logits, _ = model(changed, artists)

    # Logits at every position before the changed (last) token must be
    # unaffected by a change to a future token.
    assert torch.allclose(base_logits[:, :-1, :], changed_logits[:, :-1, :], atol=1e-6)
    assert not torch.allclose(base_logits[:, -1, :], changed_logits[:, -1, :])


def test_transformer_kv_cache_matches_full_forward() -> None:
    model = build_model(_small_config("transformer"))
    model.eval()
    torch.manual_seed(0)
    tokens = torch.randint(0, 13, (1, 9))
    artists = torch.tensor([2])

    with torch.no_grad():
        full_logits, _ = model(tokens, artists)

        # Incrementally encode a chunk, then one token at a time.
        cached_logits, state = model(tokens[:, :4], artists, state=None, use_cache=True)
        for i in range(4, tokens.shape[1]):
            step_logits, state = model(tokens[:, i : i + 1], artists, state=state, use_cache=True)
            cached_logits = torch.cat([cached_logits, step_logits], dim=1)

    assert state.length == tokens.shape[1]
    assert torch.allclose(full_logits, cached_logits, atol=1e-5)


def test_transformer_forward_without_use_cache_returns_none_state() -> None:
    model = build_model(_small_config("transformer"))
    tokens = torch.randint(0, 13, (1, 5))
    artists = torch.zeros(1, dtype=torch.long)

    logits, state = model(tokens, artists, state=None)
    assert state is None
    assert logits.shape == (1, 5, 13)


@pytest.mark.parametrize("kind", ["lstm", "gru"])
def test_rnn_streaming_equivalence(kind: str) -> None:
    model = build_model(_small_config(kind))
    model.eval()
    tokens = torch.randint(0, 13, (2, 10))
    artists = torch.randint(0, 4, (2,))

    with torch.no_grad():
        full_logits, _ = model(tokens, artists)

        first_chunk, second_chunk = tokens[:, :6], tokens[:, 6:]
        chunk1_logits, state = model(first_chunk, artists, state=None)
        chunk2_logits, _ = model(second_chunk, artists, state=state)
        streamed_logits = torch.cat([chunk1_logits, chunk2_logits], dim=1)

    assert torch.allclose(full_logits, streamed_logits, atol=1e-5)


@pytest.mark.parametrize("kind", KINDS)
def test_artist_conditioning_changes_output(kind: str) -> None:
    model = build_model(_small_config(kind))
    model.eval()
    tokens = torch.randint(0, 13, (1, 8))
    artist_a = torch.tensor([1], dtype=torch.long)
    artist_b = torch.tensor([2], dtype=torch.long)

    with torch.no_grad():
        logits_a, _ = model(tokens, artist_a)
        logits_b, _ = model(tokens, artist_b)

    assert not torch.allclose(logits_a, logits_b)


@pytest.mark.parametrize("kind", KINDS)
def test_tiny_model_overfits_and_greedy_generate_reproduces_it(kind: str) -> None:
    vocab_size = 4
    period = 4
    repeats = 12
    seq = [i % period for i in range(period * repeats)]
    tokens_all = torch.tensor(seq, dtype=torch.long)

    torch.manual_seed(0)
    model = build_model(_tiny_config(kind, vocab_size, n_artists=2, block_size=len(seq)))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    x = tokens_all[:-1].unsqueeze(0)
    y = tokens_all[1:].unsqueeze(0)
    artist = torch.zeros(1, dtype=torch.long)

    model.train()
    first_loss = None
    last_loss = None
    for step in range(400):
        optimizer.zero_grad()
        logits, _ = model(x, artist)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        optimizer.step()
        if step == 0:
            first_loss = loss.item()
        last_loss = loss.item()

    assert first_loss is not None and last_loss is not None
    assert last_loss < first_loss * 0.05

    model.eval()
    prompt = seq[:period]
    expected_continuation = seq[period:]
    generated = generate(
        model,
        prompt,
        artist_id=0,
        max_new_tokens=len(expected_continuation),
        temperature=0.0,
        seed=0,
    )
    assert generated == expected_continuation
