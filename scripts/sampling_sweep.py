"""Sweep sampling settings for one checkpoint and report diversity metrics."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lyricgen.dataset import load_split
from lyricgen.evaluate import draw_sample_prompts, generate_samples
from lyricgen.metrics import distinct_n, longest_copied_run, ngram_novelty_by_n, repeated_line_rate

if TYPE_CHECKING:
    from lyricgen.checkpoint import LoadedModel

logger = logging.getLogger(__name__)

# temperature 0.5/0.8/1.0/1.2 at top_p=1.0, plus temperature 0.8 with each of
# top_p=0.9, top_k=20 and repetition_penalty=1.2 in isolation.
DEFAULT_GRID: tuple[dict[str, float | int], ...] = (
    {"temperature": 0.5, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0},
    {"temperature": 0.8, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0},
    {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0},
    {"temperature": 1.2, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0},
    {"temperature": 0.8, "top_k": 0, "top_p": 0.9, "repetition_penalty": 1.0},
    {"temperature": 0.8, "top_k": 20, "top_p": 1.0, "repetition_penalty": 1.0},
    {"temperature": 0.8, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.2},
)

TABLE_HEADERS = (
    "temperature",
    "top_k",
    "top_p",
    "repetition_penalty",
    "6-gram novelty",
    "distinct-2",
    "repeated-line rate",
    "longest copied run (mean)",
    "gen chars/s",
)


def run_sweep(
    loaded: LoadedModel,
    data_dir: str | Path,
    split: str = "test",
    n_samples: int = 24,
    sample_chars: int = 400,
    seed: int = 1337,
    grid: Sequence[dict[str, float | int]] = DEFAULT_GRID,
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
    """Generate `n_samples` texts per grid setting and score their diversity.

    Prompts are drawn once (the same way `lyricgen.evaluate.evaluate` draws
    them: spread evenly over the checkpoint's artists, from a random
    same-artist `split` stanza each) and reused for every grid setting, so
    the comparison isolates the effect of the sampling settings rather than
    the prompts. Novelty and longest copied run are scored against the
    training split.
    """
    data_dir = Path(data_dir)
    split_records = load_split(data_dir / f"{split}.jsonl")
    train_records = load_split(data_dir / "train.jsonl")
    train_texts = [r["text"] for r in train_records]

    sample_artists, sample_prompts, _ = draw_sample_prompts(loaded, split_records, n_samples, seed)

    results = []
    for setting in grid:
        generated_texts, chars_per_sec = generate_samples(
            loaded,
            sample_artists,
            sample_prompts,
            sample_chars,
            setting["temperature"],
            setting["top_k"],
            setting["top_p"],
            setting["repetition_penalty"],
            seed,
        )
        novelty_6 = ngram_novelty_by_n(generated_texts, train_texts, (6,))[6]
        copied_runs = [longest_copied_run(t, train_texts) for t in generated_texts] or [0]
        results.append(
            {
                "setting": dict(setting),
                "novelty_6": novelty_6,
                "distinct_2": distinct_n(generated_texts, 2),
                "repeated_line_rate": repeated_line_rate(generated_texts),
                "longest_copied_run_mean": statistics.fmean(copied_runs),
                "gen_chars_per_sec": chars_per_sec,
            }
        )

    return {
        "checkpoint": checkpoint_path,
        "split": split,
        "n_samples": n_samples,
        "sample_chars": sample_chars,
        "seed": seed,
        "results": results,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def render_markdown(sweep: dict[str, Any]) -> str:
    """Render a `run_sweep` result as a Markdown table, one row per setting."""
    lines = [
        "| " + " | ".join(TABLE_HEADERS) + " |",
        "| " + " | ".join(["---"] * len(TABLE_HEADERS)) + " |",
    ]
    for entry in sweep["results"]:
        setting = entry["setting"]
        cells = (
            _fmt(setting["temperature"]),
            _fmt(setting["top_k"]),
            _fmt(setting["top_p"]),
            _fmt(setting["repetition_penalty"]),
            _fmt(entry["novelty_6"]),
            _fmt(entry["distinct_2"]),
            _fmt(entry["repeated_line_rate"]),
            _fmt(entry["longest_copied_run_mean"], 1),
            _fmt(entry["gen_chars_per_sec"], 1),
        )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint (.pt) file.")
    parser.add_argument("--data", required=True, help="Path to the prepared data directory.")
    parser.add_argument("--split", default="test", help="Split to draw prompts from.")
    parser.add_argument("--n-samples", type=int, default=24)
    parser.add_argument("--sample-chars", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out-json", default=None, help="Path to write the sweep JSON to.")
    parser.add_argument("--out-md", default=None, help="Path to write the Markdown table to.")
    args = parser.parse_args(argv)

    from lyricgen.checkpoint import load_checkpoint

    loaded = load_checkpoint(args.checkpoint)
    sweep = run_sweep(
        loaded,
        data_dir=args.data,
        split=args.split,
        n_samples=args.n_samples,
        sample_chars=args.sample_chars,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
    )
    table = render_markdown(sweep)

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(sweep, indent=2), encoding="utf-8")
        logger.info("Wrote sweep results to %s", out_json)

    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(table + "\n", encoding="utf-8")
        logger.info("Wrote sweep table to %s", out_md)

    print(table)


if __name__ == "__main__":
    main()
