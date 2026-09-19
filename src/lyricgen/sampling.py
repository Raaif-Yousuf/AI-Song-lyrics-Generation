"""Decoding utilities: logit filters and autoregressive generation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from lyricgen.models.base import LyricsModel


def apply_repetition_penalty(logits: Tensor, prev_ids: Sequence[int], penalty: float) -> Tensor:
    """Apply a CTRL-style repetition penalty to previously seen token ids.

    For each id that appeared in `prev_ids`, its logit is divided by
    `penalty` if positive or multiplied by `penalty` if negative, which in
    both cases pushes the resulting probability down for `penalty > 1`.
    """
    if penalty == 1.0 or len(prev_ids) == 0:
        return logits
    logits = logits.clone()
    ids = torch.tensor(sorted({int(i) for i in prev_ids}), dtype=torch.long, device=logits.device)
    selected = logits[ids]
    logits[ids] = torch.where(selected > 0, selected / penalty, selected * penalty)
    return logits


def top_k_filter(logits: Tensor, k: int) -> Tensor:
    """Keep exactly the top `k` logits, setting all others to -inf.

    `k <= 0` or `k >= vocab_size` disables filtering (returns `logits`
    unchanged).
    """
    vocab_size = logits.shape[-1]
    if k <= 0 or k >= vocab_size:
        return logits
    top_values, top_indices = torch.topk(logits, k, dim=-1)
    result = torch.full_like(logits, float("-inf"))
    result.scatter_(-1, top_indices, top_values)
    return result


def top_p_filter(logits: Tensor, p: float) -> Tensor:
    """Nucleus filtering: keep the smallest set of top tokens whose
    cumulative probability is at least `p`, setting the rest to -inf.

    `p >= 1.0` disables filtering (returns `logits` unchanged).
    """
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)

    remove_sorted = cumulative > p
    remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
    remove_sorted[..., 0] = False

    remove = torch.zeros_like(remove_sorted)
    remove.scatter_(-1, sorted_indices, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


def sample_next(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> int:
    """Sample the next token id from a 1D logits tensor.

    `temperature == 0` selects the argmax (greedy decoding) and ignores
    `top_k`/`top_p`. Otherwise logits are scaled by `1 / temperature`,
    filtered by `top_k` then `top_p`, and sampled from the resulting
    categorical distribution.
    """
    if temperature == 0.0:
        return int(torch.argmax(logits, dim=-1).item())
    scaled = logits / temperature
    scaled = top_k_filter(scaled, top_k)
    scaled = top_p_filter(scaled, top_p)
    probs = torch.softmax(scaled, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(next_id.item())


def generate(
    model: LyricsModel,
    prompt_ids: Sequence[int],
    artist_id: int,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    repetition_window: int = 64,
    seed: int | None = None,
) -> list[int]:
    """Autoregressively generate `max_new_tokens` ids after `prompt_ids`.

    Returns only the newly generated ids (not the prompt). Recurrent
    models stream their hidden state one step at a time; models with a
    fixed context window (`max_context` not None) are re-fed a cropped
    window of the most recent ids on every step. Runs under
    `torch.inference_mode()` with the model in eval mode, and restores the
    model's previous training mode before returning.
    """
    if len(prompt_ids) == 0:
        raise ValueError("prompt_ids must be non-empty")

    device = next(model.parameters()).device
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            artist_tensor = torch.tensor([artist_id], dtype=torch.long, device=device)
            all_ids = list(prompt_ids)
            generated: list[int] = []
            block_size = model.max_context

            def _run(ids: list[int], state: object) -> tuple[Tensor, object]:
                context = torch.tensor([ids], dtype=torch.long, device=device)
                return model(context, artist_tensor, state)

            if block_size is None:
                logits, state = _run(all_ids, None)
            else:
                logits, state = _run(all_ids[-block_size:], None)

            for _ in range(max_new_tokens):
                next_logits = logits[0, -1, :]
                window = all_ids[-repetition_window:] if repetition_window > 0 else []
                next_logits = apply_repetition_penalty(next_logits, window, repetition_penalty)
                next_id = sample_next(next_logits, temperature, top_k, top_p, generator)
                generated.append(next_id)
                all_ids.append(next_id)

                if block_size is None:
                    logits, state = _run([next_id], state)
                else:
                    logits, state = _run(all_ids[-block_size:], None)
        return generated
    finally:
        model.train(was_training)
