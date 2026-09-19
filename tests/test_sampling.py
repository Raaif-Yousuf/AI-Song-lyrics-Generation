from __future__ import annotations

import torch

from lyricgen.models import build_model
from lyricgen.sampling import (
    apply_repetition_penalty,
    generate,
    sample_next,
    top_k_filter,
    top_p_filter,
)


def test_sample_next_temperature_zero_is_argmax() -> None:
    torch.manual_seed(0)
    for _ in range(10):
        logits = torch.randn(20)
        assert sample_next(logits, temperature=0.0) == int(torch.argmax(logits).item())


def test_top_k_filter_keeps_exactly_k() -> None:
    torch.manual_seed(1)
    logits = torch.randn(50)
    k = 7
    filtered = top_k_filter(logits, k)

    finite_mask = torch.isfinite(filtered)
    assert int(finite_mask.sum()) == k

    top_values, top_indices = torch.topk(logits, k)
    expected = torch.full_like(logits, float("-inf"))
    expected[top_indices] = top_values
    assert torch.equal(filtered, expected)


def test_top_k_filter_disabled_for_k_zero_or_full_vocab() -> None:
    logits = torch.randn(10)
    assert torch.equal(top_k_filter(logits, 0), logits)
    assert torch.equal(top_k_filter(logits, 10), logits)


def test_top_p_filter_keeps_minimal_nucleus() -> None:
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    logits = torch.log(probs)
    filtered = top_p_filter(logits, p=0.7)

    finite_mask = torch.isfinite(filtered)
    # cumulative probs are 0.5, 0.8, 0.95, 1.0; the minimal nucleus with
    # cumulative >= 0.7 is the first two tokens.
    assert finite_mask.tolist() == [True, True, False, False]


def test_top_p_filter_disabled_for_p_one() -> None:
    logits = torch.randn(10)
    assert torch.equal(top_p_filter(logits, 1.0), logits)


def test_repetition_penalty_lowers_repeated_tokens_and_handles_negative_logits() -> None:
    logits = torch.tensor([2.0, -3.0, 1.0, -1.0])
    penalized = apply_repetition_penalty(logits, prev_ids=[0, 1], penalty=2.0)

    # Positive logit divided by penalty (lowered), negative logit
    # multiplied by penalty (made more negative, i.e. also lowered).
    assert penalized[0].item() == 1.0
    assert penalized[1].item() == -6.0
    # Untouched entries are unchanged.
    assert penalized[2].item() == 1.0
    assert penalized[3].item() == -1.0
    assert penalized[0] < logits[0]
    assert penalized[1] < logits[1]


def test_repetition_penalty_noop_for_penalty_one_or_empty_history() -> None:
    logits = torch.randn(5)
    assert torch.equal(apply_repetition_penalty(logits, [0, 1], 1.0), logits)
    assert torch.equal(apply_repetition_penalty(logits, [], 2.0), logits)


def _tiny_transformer(vocab_size: int = 8, n_artists: int = 2, block_size: int = 8) -> object:
    return build_model(
        dict(
            kind="transformer",
            vocab_size=vocab_size,
            n_artists=n_artists,
            n_layer=1,
            n_head=1,
            d_model=8,
            block_size=block_size,
            dropout=0.0,
        )
    )


def _tiny_lstm(vocab_size: int = 8, n_artists: int = 2) -> object:
    return build_model(
        dict(
            kind="lstm",
            vocab_size=vocab_size,
            n_artists=n_artists,
            embed_dim=8,
            artist_embed_dim=4,
            hidden_sizes=(8, 8),
            dropout=0.0,
        )
    )


def test_generate_rejects_empty_prompt() -> None:
    model = _tiny_lstm()
    try:
        generate(model, [], artist_id=0, max_new_tokens=3)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty prompt")


def test_seeded_generate_is_reproducible() -> None:
    torch.manual_seed(0)
    model = _tiny_lstm()
    prompt = [1, 2, 3]

    out1 = generate(model, prompt, artist_id=1, max_new_tokens=15, temperature=0.9,
                    top_k=4, seed=42)
    out2 = generate(model, prompt, artist_id=1, max_new_tokens=15, temperature=0.9,
                    top_k=4, seed=42)

    assert out1 == out2


def test_transformer_generate_works_past_block_size() -> None:
    torch.manual_seed(0)
    block_size = 8
    model = _tiny_transformer(block_size=block_size)
    prompt = [0, 1, 2]
    max_new_tokens = 25  # prompt + generated exceeds block_size

    out = generate(
        model, prompt, artist_id=0, max_new_tokens=max_new_tokens, temperature=0.8, seed=1
    )

    assert len(out) == max_new_tokens
    assert all(0 <= tok < 8 for tok in out)


def test_generate_restores_training_mode() -> None:
    model = _tiny_lstm()
    model.train()
    generate(model, [0, 1], artist_id=0, max_new_tokens=2, seed=0)
    assert model.training is True

    model.eval()
    generate(model, [0, 1], artist_id=0, max_new_tokens=2, seed=0)
    assert model.training is False
