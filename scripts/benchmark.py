"""Benchmark training throughput and generation speed for each model config.

For every `configs/<name>.yaml` this builds the model with that config's
hyperparameters and measures (a) forward + backward + optimizer-step
tokens/sec at the config's batch_size x block_size and (b) tokens/sec
generating new tokens from a short random prompt. Numbers are the median
of several timed repeats with untimed warmup steps excluded, so a single
slow rep (e.g. from other processes on a shared machine) does not skew
the result as much as a mean would.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from lyricgen.config import ExperimentConfig, load_config
from lyricgen.models import build_model
from lyricgen.sampling import generate

logger = logging.getLogger(__name__)

_CONFIG_NAMES = ("lstm", "gru", "transformer", "transformer_bpe")

# Small hyperparameter overrides (keyed by model kind, so transformer and
# transformer_bpe share one entry) used only with --tiny, to make the
# benchmark itself fast to smoke-test.
_TINY_HYPERPARAMETERS: dict[str, dict[str, Any]] = {
    "lstm": {"embed_dim": 8, "artist_embed_dim": 4, "hidden_sizes": (8, 8), "dropout": 0.0},
    "gru": {"embed_dim": 8, "artist_embed_dim": 4, "hidden_sizes": (8, 8), "dropout": 0.0},
    "transformer": {"n_layer": 1, "n_head": 1, "d_model": 8, "block_size": 8, "dropout": 0.0},
}
_TINY_BATCH_SIZE = 2
_TINY_BLOCK_SIZE = 8

_HEADERS = ("model", "tokenizer", "vocab", "params", "batch x block", "train tok/s", "gen tok/s")


def _vocab_size_for(config: ExperimentConfig, default_vocab_size: int) -> int:
    """Char configs use `default_vocab_size`; BPE configs use their own."""
    if config.tokenizer.type == "bpe":
        if config.tokenizer.vocab_size is None:
            raise ValueError(f"{config.name}: bpe tokenizer config is missing vocab_size")
        return config.tokenizer.vocab_size
    return default_vocab_size


def _model_dict(
    config: ExperimentConfig, vocab_size: int, n_artists: int, tiny: bool
) -> dict[str, Any]:
    hyperparameters = dict(config.model.hyperparameters)
    if tiny:
        hyperparameters.update(_TINY_HYPERPARAMETERS.get(config.model.kind, {}))
    return {
        "kind": config.model.kind,
        "vocab_size": vocab_size,
        "n_artists": n_artists,
        **hyperparameters,
    }


def _batch_block_for(config: ExperimentConfig, tiny: bool) -> tuple[int, int]:
    if tiny:
        return _TINY_BATCH_SIZE, _TINY_BLOCK_SIZE
    return config.train.batch_size, config.train.block_size


def _median_seconds(step: Callable[[], None], repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        step()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        step()
        timings.append(time.perf_counter() - start)
    return statistics.median(timings)


def benchmark_training(
    model: torch.nn.Module,
    vocab_size: int,
    n_artists: int,
    batch_size: int,
    block_size: int,
    steps: int,
    warmup: int,
) -> float:
    """Median forward + backward + AdamW-step tokens/sec at a fixed shape."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.randint(0, vocab_size, (batch_size, block_size))
    targets = torch.randint(0, vocab_size, (batch_size, block_size))
    artists = torch.randint(0, n_artists, (batch_size,))

    def step() -> None:
        optimizer.zero_grad()
        logits, _ = model(tokens, artists)
        flat_logits = logits.reshape(-1, vocab_size)
        loss = torch.nn.functional.cross_entropy(flat_logits, targets.reshape(-1))
        loss.backward()
        optimizer.step()

    median_s = _median_seconds(step, steps, warmup)
    return (batch_size * block_size) / median_s if median_s > 0 else 0.0


def benchmark_generation(
    model: torch.nn.Module,
    vocab_size: int,
    n_artists: int,
    prompt_len: int,
    new_tokens: int,
    repeats: int,
) -> float:
    """Median new-tokens/sec generating `new_tokens` ids from a random prompt."""
    model.eval()
    rng = random.Random(0)
    prompt = [rng.randrange(vocab_size) for _ in range(prompt_len)]
    artist_id = 1 if n_artists > 1 else 0

    def step() -> None:
        # generate() already runs under torch.inference_mode() internally.
        generate(
            model,
            prompt,
            artist_id=artist_id,
            max_new_tokens=new_tokens,
            temperature=0.8,
            top_k=40,
            seed=0,
        )

    median_s = _median_seconds(step, repeats, warmup=1)
    return new_tokens / median_s if median_s > 0 else 0.0


def run_benchmark(
    configs_dir: Path,
    vocab_size: int,
    n_artists: int,
    threads: int,
    train_steps: int,
    train_warmup: int,
    gen_prompt_len: int,
    gen_tokens: int,
    gen_repeats: int,
    tiny: bool = False,
) -> list[dict[str, Any]]:
    """Load each `configs/<name>.yaml` and benchmark its model; one row per config."""
    torch.set_num_threads(threads)
    torch.set_flush_denormal(True)

    rows: list[dict[str, Any]] = []
    for name in _CONFIG_NAMES:
        config_path = configs_dir / f"{name}.yaml"
        if not config_path.exists():
            logger.warning("skipping missing config %s", config_path)
            continue

        config = load_config(config_path)
        this_vocab_size = _vocab_size_for(config, vocab_size)
        batch_size, block_size = _batch_block_for(config, tiny)

        torch.manual_seed(1337)
        model = build_model(_model_dict(config, this_vocab_size, n_artists, tiny))

        train_tokens_per_sec = benchmark_training(
            model, this_vocab_size, n_artists, batch_size, block_size, train_steps, train_warmup
        )
        gen_tokens_per_sec = benchmark_generation(
            model, this_vocab_size, n_artists, gen_prompt_len, gen_tokens, gen_repeats
        )

        rows.append(
            {
                "name": config.name,
                "kind": config.model.kind,
                "tokenizer": config.tokenizer.type,
                "vocab_size": this_vocab_size,
                "params": model.num_parameters(),
                "batch_size": batch_size,
                "block_size": block_size,
                "train_tokens_per_sec": train_tokens_per_sec,
                "gen_tokens_per_sec": gen_tokens_per_sec,
            }
        )
    return rows


def render_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(_HEADERS) + " |",
        "| " + " | ".join(["---"] * len(_HEADERS)) + " |",
    ]
    for row in rows:
        cells = (
            row["name"],
            row["tokenizer"],
            str(row["vocab_size"]),
            f"{row['params']:,}",
            f"{row['batch_size']}x{row['block_size']}",
            f"{row['train_tokens_per_sec']:.0f}",
            f"{row['gen_tokens_per_sec']:.0f}",
        )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs-dir", default="configs")
    parser.add_argument("--vocab-size", type=int, default=160, help="Vocab size for char configs")
    parser.add_argument("--n-artists", type=int, default=48)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20, help="Timed training steps (median)")
    parser.add_argument(
        "--warmup", type=int, default=5, help="Untimed training steps before timing"
    )
    parser.add_argument("--gen-prompt-len", type=int, default=20)
    parser.add_argument("--gen-tokens", type=int, default=400)
    parser.add_argument(
        "--gen-repeats", type=int, default=3, help="Timed generation repeats (median)"
    )
    parser.add_argument(
        "--tiny", action="store_true", help="Shrink model and batch/block sizes for a quick run"
    )
    parser.add_argument("--out-json", default=None, help="Optional file to also write the rows to")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    rows = run_benchmark(
        configs_dir=Path(args.configs_dir),
        vocab_size=args.vocab_size,
        n_artists=args.n_artists,
        threads=args.threads,
        train_steps=args.steps,
        train_warmup=args.warmup,
        gen_prompt_len=args.gen_prompt_len,
        gen_tokens=args.gen_tokens,
        gen_repeats=args.gen_repeats,
        tiny=args.tiny,
    )

    table = render_table(rows)
    print(table)

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        logger.info("wrote %s", args.out_json)


if __name__ == "__main__":
    main()
