"""Full-split evaluation and qualitative sampling for a trained checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from lyricgen.dataset import count_target_chars, encode_streams, iter_eval_batches, load_split
from lyricgen.metrics import bits_per_char, char_perplexity, longest_copied_run, ngram_novelty_by_n
from lyricgen.tokenizers import PAD_ID

if TYPE_CHECKING:
    from lyricgen.checkpoint import LoadedModel

logger = logging.getLogger(__name__)

DEFAULT_NOVELTY_NS: tuple[int, ...] = (4, 6, 8)


def _compute_full_split_nll(
    model: torch.nn.Module,
    streams: dict[int, torch.Tensor],
    block_size: int,
    batch_size: int,
) -> tuple[float, int]:
    """Total negative log-likelihood (nats, summed) and target token count.

    Runs the model in eval mode under `torch.inference_mode()` over every
    deterministic eval window, restoring the model's previous training mode
    before returning.
    """
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_targets = 0
    try:
        with torch.inference_mode():
            for artists, x, y in iter_eval_batches(streams, block_size, batch_size):
                logits, _ = model(x, artists, None)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    y.reshape(-1),
                    ignore_index=PAD_ID,
                    reduction="sum",
                )
                total_nll += float(loss.item())
                total_targets += int((y != PAD_ID).sum().item())
    finally:
        model.train(was_training)
    return total_nll, total_targets


def _spread_over_artists(artists: Sequence[tuple[str, str]], n_samples: int) -> list[str]:
    """Assign `n_samples` slots to `artists` as evenly as possible.

    Any remainder is given one extra slot each to the first artists (in the
    order `artists` is given, which is by display name), so the assignment
    is deterministic.
    """
    slugs = [slug for slug, _ in artists]
    base, remainder = divmod(n_samples, len(slugs))
    counts = [base + (1 if i < remainder else 0) for i in range(len(slugs))]
    assignment: list[str] = []
    for slug, count in zip(slugs, counts, strict=True):
        assignment.extend([slug] * count)
    return assignment


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line
    return ""


def evaluate(
    loaded: LoadedModel,
    data_dir: str | Path,
    split: str = "test",
    n_samples: int = 24,
    sample_chars: int = 400,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
    seed: int = 1337,
    novelty_ns: Sequence[int] = DEFAULT_NOVELTY_NS,
) -> dict[str, Any]:
    """Evaluate a loaded checkpoint on `split` and generate qualitative samples.

    Computes full-split loss (nats/token), bits-per-char and character
    perplexity with the model in eval mode under `torch.inference_mode()`;
    generates `n_samples` texts spread evenly over the artists present in
    the checkpoint's artist vocabulary, prompted deterministically from the
    first line of a random `split` stanza per artist; and scores those
    samples (and an equal number of random held-out `split` stanzas,
    truncated to `sample_chars`, as a reference point) for n-gram novelty
    and longest verbatim copied run against the training split.
    """
    data_dir = Path(data_dir)
    split_records = load_split(data_dir / f"{split}.jsonl")
    train_records = load_split(data_dir / "train.jsonl")
    train_texts = [r["text"] for r in train_records]

    tokenizer = loaded.tokenizer
    artist_vocab = loaded.artist_vocab
    model = loaded.model

    streams = encode_streams(split_records, tokenizer, artist_vocab)
    n_chars = count_target_chars(streams, tokenizer)
    if n_chars <= 0:
        raise ValueError(f"split '{split}' in {data_dir} has no evaluable target characters")

    block_size = int(loaded.experiment["train"]["block_size"])
    batch_size = int(loaded.experiment["train"]["batch_size"])
    total_nll, total_targets = _compute_full_split_nll(model, streams, block_size, batch_size)

    test_loss = total_nll / total_targets if total_targets else float("nan")
    test_bpc = bits_per_char(total_nll, n_chars)
    test_char_ppl = char_perplexity(total_nll, n_chars)

    artists = loaded.artists()
    if not artists:
        raise ValueError("checkpoint has no artists to sample from")

    records_by_artist: dict[str, list[str]] = {}
    for record in split_records:
        records_by_artist.setdefault(record["artist"], []).append(record["text"])

    rng = random.Random(seed)
    assignment = _spread_over_artists(artists, n_samples)

    sample_artists: list[str] = []
    sample_prompts: list[str] = []
    for slug in assignment:
        candidates = records_by_artist.get(slug, [])
        prompt = _first_line(rng.choice(candidates)) if candidates else ""
        sample_artists.append(slug)
        sample_prompts.append(prompt)

    generated_texts: list[str] = []
    gen_start = time.perf_counter()
    for i, (slug, prompt) in enumerate(zip(sample_artists, sample_prompts, strict=True)):
        text = loaded.generate_text(
            prompt=prompt,
            artist=slug,
            length=sample_chars,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed + i,
        )
        generated_texts.append(text)
    gen_elapsed = time.perf_counter() - gen_start
    total_gen_chars = sum(len(t) for t in generated_texts)
    gen_chars_per_sec = total_gen_chars / gen_elapsed if gen_elapsed > 0 else float("inf")

    all_split_texts = [r["text"] for r in split_records]
    heldout_count = min(n_samples, len(all_split_texts))
    heldout_texts = [
        text[:sample_chars] for text in rng.sample(all_split_texts, heldout_count)
    ]

    novelty = ngram_novelty_by_n(generated_texts, train_texts, novelty_ns)
    novelty_heldout = ngram_novelty_by_n(heldout_texts, train_texts, novelty_ns)

    sample_runs = [longest_copied_run(t, train_texts) for t in generated_texts] or [0]
    heldout_runs = [longest_copied_run(t, train_texts) for t in heldout_texts] or [0]

    samples = [
        {"artist": artist, "prompt": prompt, "text": text[:200]}
        for artist, prompt, text in list(
            zip(sample_artists, sample_prompts, generated_texts, strict=True)
        )[:6]
    ]

    return {
        "split": split,
        "test_loss": test_loss,
        "test_bpc": test_bpc,
        "test_char_ppl": test_char_ppl,
        "gen_chars_per_sec": gen_chars_per_sec,
        "novelty": novelty,
        "novelty_heldout": novelty_heldout,
        "longest_copied_run": {
            "samples": {"mean": statistics.fmean(sample_runs), "max": max(sample_runs)},
            "heldout": {"mean": statistics.fmean(heldout_runs), "max": max(heldout_runs)},
        },
        "sampling": {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        },
        "n_samples": n_samples,
        "seed": seed,
        "samples": samples,
    }


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint (.pt) file.")
    parser.add_argument("--data", required=True, help="Path to the prepared data directory.")
    parser.add_argument("--split", default="test", help="Split to evaluate on.")
    parser.add_argument("--out", required=True, help="Path to write the evaluation JSON to.")
    parser.add_argument("--n-samples", type=int, default=24)
    parser.add_argument("--sample-chars", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--novelty-ns",
        type=int,
        nargs="+",
        default=list(DEFAULT_NOVELTY_NS),
        help="Word n-gram lengths to score novelty for.",
    )
    args = parser.parse_args(argv)

    from lyricgen.checkpoint import load_checkpoint

    loaded = load_checkpoint(args.checkpoint)
    result = evaluate(
        loaded,
        data_dir=args.data,
        split=args.split,
        n_samples=args.n_samples,
        sample_chars=args.sample_chars,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        novelty_ns=tuple(args.novelty_ns),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Wrote evaluation results to %s", out_path)


if __name__ == "__main__":
    main()
