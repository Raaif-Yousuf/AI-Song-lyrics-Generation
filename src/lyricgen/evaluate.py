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

from lyricgen.dataset import (
    ANY_ARTIST_ID,
    count_target_chars,
    encode_streams,
    iter_eval_batches,
    load_split,
)
from lyricgen.metrics import (
    bits_per_char,
    char_perplexity,
    distinct_n,
    longest_copied_run,
    ngram_novelty_by_n,
    repeated_line_rate,
)
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
    artist_map: dict[int, int] | None = None,
) -> tuple[float, int, dict[int, tuple[float, int]]]:
    """One pass over `streams`: total nats/targets plus a per-artist breakdown.

    If `artist_map` is given, each window's artist id is remapped through it
    before conditioning the model (used to score "unconditional" or "wrong
    artist" conditioning), but the resulting loss is still bucketed by the
    window's *original* artist id, so per-artist comparisons across
    conditions stay meaningful. Runs the model in eval mode under
    `torch.inference_mode()`, restoring its previous training mode before
    returning.
    """
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_targets = 0
    per_artist: dict[int, list[float]] = {}
    try:
        with torch.inference_mode():
            for original_artists, x, y in iter_eval_batches(streams, block_size, batch_size):
                artists = original_artists
                if artist_map is not None:
                    artists = original_artists.clone().apply_(artist_map.__getitem__)
                logits, _ = model(x, artists, None)
                per_token_nll = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    y.reshape(-1),
                    ignore_index=PAD_ID,
                    reduction="none",
                ).reshape(y.shape)
                valid = y != PAD_ID
                row_nll = (per_token_nll * valid).sum(dim=1)
                row_targets = valid.sum(dim=1)
                total_nll += float(row_nll.sum().item())
                total_targets += int(row_targets.sum().item())
                for i in range(original_artists.shape[0]):
                    artist_id = int(original_artists[i].item())
                    bucket = per_artist.setdefault(artist_id, [0.0, 0])
                    bucket[0] += float(row_nll[i].item())
                    bucket[1] += int(row_targets[i].item())
    finally:
        model.train(was_training)
    return total_nll, total_targets, {a: (n, c) for a, (n, c) in per_artist.items()}


def _seeded_derangement(items: Sequence[int], seed: int) -> dict[int, int]:
    """A derangement (permutation with no fixed points) of `items`, seeded.

    A derangement of fewer than 2 distinct items does not exist, so with 0
    or 1 items this just maps every item to itself (there is no meaningful
    "wrong artist" to compare against in that case).
    """
    unique_items = list(dict.fromkeys(items))
    if len(unique_items) < 2:
        return {item: item for item in unique_items}

    rng = random.Random(seed)
    shuffled = list(unique_items)
    for _ in range(100):
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(unique_items, shuffled, strict=True)):
            break
    else:
        # Astronomically unlikely with 2+ items, but a rotation by one is
        # always a valid derangement, so fall back to it.
        shuffled = shuffled[1:] + shuffled[:1]
    return dict(zip(unique_items, shuffled, strict=True))


def spread_over_artists(artists: Sequence[tuple[str, str]], n_samples: int) -> list[str]:
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


def first_line(text: str) -> str:
    """The first non-blank line of `text`, or "" if there is none."""
    for line in text.splitlines():
        if line.strip():
            return line
    return ""


def draw_sample_prompts(
    loaded: LoadedModel,
    split_records: list[dict[str, str]],
    n_samples: int,
    seed: int,
) -> tuple[list[str], list[str], random.Random]:
    """Deterministically choose (artist, prompt) pairs for qualitative samples.

    Samples are spread evenly over `loaded.artists()`; each prompt is the
    first non-blank line of a random same-artist stanza from
    `split_records`. Returns the artists, the prompts, and the `random.Random`
    used to draw them (continued from the same draw sequence), so a caller
    that also needs a held-out reference draw (as `evaluate` does) can keep
    it deterministic and non-overlapping with the prompt draws.
    """
    artists = loaded.artists()
    if not artists:
        raise ValueError("checkpoint has no artists to sample from")

    records_by_artist: dict[str, list[str]] = {}
    for record in split_records:
        records_by_artist.setdefault(record["artist"], []).append(record["text"])

    rng = random.Random(seed)
    assignment = spread_over_artists(artists, n_samples)

    sample_artists: list[str] = []
    sample_prompts: list[str] = []
    for slug in assignment:
        candidates = records_by_artist.get(slug, [])
        prompt = first_line(rng.choice(candidates)) if candidates else ""
        sample_artists.append(slug)
        sample_prompts.append(prompt)
    return sample_artists, sample_prompts, rng


def generate_samples(
    loaded: LoadedModel,
    sample_artists: Sequence[str],
    sample_prompts: Sequence[str],
    sample_chars: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    seed: int,
) -> tuple[list[str], float]:
    """Generate one text per `(artist, prompt)` pair; returns `(texts, chars/sec)`.

    Each sample gets its own seed (`seed + index`) derived from the shared
    base `seed`, so a given index's text is reproducible across calls.
    """
    generated_texts: list[str] = []
    start = time.perf_counter()
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
    elapsed = time.perf_counter() - start
    total_chars = sum(len(t) for t in generated_texts)
    chars_per_sec = total_chars / elapsed if elapsed > 0 else float("inf")
    return generated_texts, chars_per_sec


def _conditioning_report(
    model: torch.nn.Module,
    streams: dict[int, torch.Tensor],
    tokenizer: Any,
    artist_vocab: Any,
    block_size: int,
    batch_size: int,
    n_chars: int,
    test_bpc: float,
    per_artist_true: dict[int, tuple[float, int]],
    seed: int,
) -> dict[str, Any]:
    """Artist-conditioning bpc under true, unconditional and wrong artists.

    Each condition is one pass over the split (see `_compute_full_split_nll`);
    "true" reuses the totals and per-artist breakdown already computed for
    the top-level `test_bpc` instead of re-running the model, so this adds
    exactly two more passes ("any" and "wrong"), not one per artist.
    """
    artist_ids = list(streams)
    any_map = {a: ANY_ARTIST_ID for a in artist_ids}
    wrong_map = _seeded_derangement(artist_ids, seed)

    total_any, _, per_artist_any = _compute_full_split_nll(
        model, streams, block_size, batch_size, any_map
    )
    total_wrong, _, _ = _compute_full_split_nll(model, streams, block_size, batch_size, wrong_map)

    per_artist_chars = {
        artist_id: count_target_chars({artist_id: stream}, tokenizer)
        for artist_id, stream in streams.items()
    }

    per_artist_gap: dict[str, dict[str, float]] = {}
    for artist_id, chars in per_artist_chars.items():
        if artist_id == ANY_ARTIST_ID or chars <= 0:
            continue
        true_nll, _ = per_artist_true.get(artist_id, (0.0, 0))
        any_nll, _ = per_artist_any.get(artist_id, (0.0, 0))
        slug = artist_vocab.slug(artist_id)
        per_artist_gap[slug] = {
            "true": bits_per_char(true_nll, chars),
            "any": bits_per_char(any_nll, chars),
        }

    return {
        "true_bpc": test_bpc,
        "any_bpc": bits_per_char(total_any, n_chars),
        "wrong_bpc": bits_per_char(total_wrong, n_chars),
        "per_artist": per_artist_gap,
    }


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
    skip_conditioning: bool = False,
) -> dict[str, Any]:
    """Evaluate a loaded checkpoint on `split` and generate qualitative samples.

    Computes full-split loss (nats/token), bits-per-char and character
    perplexity with the model in eval mode under `torch.inference_mode()`;
    generates `n_samples` texts spread evenly over the artists present in
    the checkpoint's artist vocabulary, prompted deterministically from the
    first line of a random `split` stanza per artist; and scores those
    samples (and an equal number of random held-out `split` stanzas,
    truncated to `sample_chars`, as a reference point) for n-gram novelty,
    distinct n-grams, repeated-line rate and longest verbatim copied run
    against the training split.

    Unless `skip_conditioning` is set, also reports how much the model
    actually uses artist conditioning: bpc with the true artist id, with
    the unconditional ("any") id, and with a wrong (deranged) artist id,
    plus the per-artist true-vs-any bpc gap.
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
    total_nll, total_targets, per_artist_true = _compute_full_split_nll(
        model, streams, block_size, batch_size
    )

    test_loss = total_nll / total_targets if total_targets else float("nan")
    test_bpc = bits_per_char(total_nll, n_chars)
    test_char_ppl = char_perplexity(total_nll, n_chars)

    conditioning = None
    if not skip_conditioning:
        conditioning = _conditioning_report(
            model,
            streams,
            tokenizer,
            artist_vocab,
            block_size,
            batch_size,
            n_chars,
            test_bpc,
            per_artist_true,
            seed,
        )

    sample_artists, sample_prompts, rng = draw_sample_prompts(
        loaded, split_records, n_samples, seed
    )
    generated_texts, gen_chars_per_sec = generate_samples(
        loaded,
        sample_artists,
        sample_prompts,
        sample_chars,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        seed,
    )

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

    result = {
        "split": split,
        "test_loss": test_loss,
        "test_bpc": test_bpc,
        "test_char_ppl": test_char_ppl,
        "gen_chars_per_sec": gen_chars_per_sec,
        "novelty": novelty,
        "novelty_heldout": novelty_heldout,
        "distinct_2": {
            "samples": distinct_n(generated_texts, 2),
            "heldout": distinct_n(heldout_texts, 2),
        },
        "distinct_3": {
            "samples": distinct_n(generated_texts, 3),
            "heldout": distinct_n(heldout_texts, 3),
        },
        "repeated_line_rate": {
            "samples": repeated_line_rate(generated_texts),
            "heldout": repeated_line_rate(heldout_texts),
        },
        "longest_copied_run": {
            "samples": {"mean": statistics.fmean(sample_runs), "max": max(sample_runs)},
            "heldout": {"mean": statistics.fmean(heldout_runs), "max": max(heldout_runs)},
        },
        "conditioning": conditioning,
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
    return result


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
    parser.add_argument(
        "--skip-conditioning",
        action="store_true",
        help="Skip the artist-conditioning bpc breakdown (saves two full passes).",
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
        skip_conditioning=args.skip_conditioning,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Wrote evaluation results to %s", out_path)


if __name__ == "__main__":
    main()
