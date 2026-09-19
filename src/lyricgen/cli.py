"""Command-line entry point: `lyricgen prepare|train|generate|evaluate|demo`."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from lyricgen.checkpoint import LoadedModel, load_checkpoint
from lyricgen.config import DataConfig, load_config
from lyricgen.data.prepare import DEFAULT_MIN_STANZA_CHARS
from lyricgen.data.prepare import prepare as run_prepare

logger = logging.getLogger(__name__)


def _cmd_prepare(args: argparse.Namespace) -> None:
    stats = run_prepare(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        min_stanza_chars=args.min_stanza_chars,
    )
    logger.info("Prepared data: %s", json.dumps(stats, indent=2))


def _cmd_train(args: argparse.Namespace) -> None:
    from lyricgen.train import train as run_train

    config = load_config(args.config)
    if args.steps is not None:
        config.train.steps = args.steps
    if args.out_dir is not None:
        config.out_dir = args.out_dir
    if args.data_dir is not None:
        config.data = DataConfig(dir=args.data_dir)

    metrics = run_train(config)
    logger.info("Training complete: %s", json.dumps(metrics, indent=2))


def _print_artist_menu(artists: list[tuple[str, str]]) -> None:
    print("1. Any (unconditional)")
    for i, (_, display_name) in enumerate(artists, start=2):
        print(f"{i}. {display_name}")


def _prompt_artist_choice(artists: list[tuple[str, str]]) -> str | None:
    while True:
        choice = input(f"Enter the number corresponding to an artist (1-{len(artists) + 1}): ")
        choice = choice.strip()
        if not choice.isdigit():
            print("Invalid choice. Please enter a valid number.")
            continue
        index = int(choice)
        if index == 1:
            return None
        if 2 <= index <= len(artists) + 1:
            return artists[index - 2][0]
        print("Invalid choice. Please enter a valid number.")


def _run_interactive(loaded: LoadedModel) -> None:
    artists = loaded.artists()
    while True:
        print("Available artists:")
        _print_artist_menu(artists)
        artist_slug = _prompt_artist_choice(artists)
        temperature = float(input("Enter the temperature (0.0 - 1.5): ").strip())
        length = int(input("Enter the number of characters to generate: ").strip())

        text = loaded.generate_text(artist=artist_slug, length=length, temperature=temperature)
        print(text)

        again = input("Generate again? (y/n): ").strip().lower()
        if again not in ("y", "yes"):
            break


def _cmd_generate(args: argparse.Namespace) -> None:
    if args.checkpoint is None:
        raise SystemExit("lyricgen generate: --checkpoint is required")
    loaded = load_checkpoint(args.checkpoint)

    if args.list_artists:
        for slug, display_name in loaded.artists():
            print(f"{slug}\t{display_name}")
        return

    if args.interactive:
        _run_interactive(loaded)
        return

    for i in range(args.num_samples):
        sample_seed = None if args.seed is None else args.seed + i
        text = loaded.generate_text(
            prompt=args.prompt,
            artist=args.artist,
            length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            seed=sample_seed,
        )
        print(text)


def _cmd_evaluate(args: argparse.Namespace) -> None:
    try:
        from lyricgen import evaluate as evaluate_module
    except ImportError as exc:
        raise SystemExit(f"lyricgen evaluate is not available yet: {exc}") from exc
    evaluate_module.main(args.args)


def _cmd_demo(args: argparse.Namespace) -> None:
    try:
        from lyricgen import app as app_module
    except ImportError as exc:
        raise SystemExit(f"lyricgen demo is not available yet: {exc}") from exc
    app_module.main(args.args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lyricgen", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare", help="Build train/val/test splits from raw lyric files."
    )
    prepare_parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    prepare_parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    prepare_parser.add_argument("--val-frac", type=float, default=0.1)
    prepare_parser.add_argument("--test-frac", type=float, default=0.1)
    prepare_parser.add_argument("--seed", type=int, default=1337)
    prepare_parser.add_argument("--min-stanza-chars", type=int, default=DEFAULT_MIN_STANZA_CHARS)
    prepare_parser.set_defaults(func=_cmd_prepare)

    train_parser = subparsers.add_parser("train", help="Train a model from an experiment config.")
    train_parser.add_argument("--config", required=True, type=Path)
    train_parser.add_argument("--steps", type=int, default=None)
    train_parser.add_argument("--out-dir", type=str, default=None)
    train_parser.add_argument("--data-dir", type=str, default=None)
    train_parser.set_defaults(func=_cmd_train)

    generate_parser = subparsers.add_parser(
        "generate", help="Generate lyrics from a trained checkpoint."
    )
    generate_parser.add_argument("--checkpoint", type=Path, default=None)
    generate_parser.add_argument("--artist", default=None)
    generate_parser.add_argument("--prompt", default="")
    generate_parser.add_argument("--length", type=int, default=400)
    generate_parser.add_argument("--temperature", type=float, default=0.8)
    generate_parser.add_argument("--top-k", type=int, default=0)
    generate_parser.add_argument("--top-p", type=float, default=1.0)
    generate_parser.add_argument("--repetition-penalty", type=float, default=1.0)
    generate_parser.add_argument("--seed", type=int, default=None)
    generate_parser.add_argument("--num-samples", type=int, default=1)
    generate_parser.add_argument("--list-artists", action="store_true")
    generate_parser.add_argument("--interactive", action="store_true")
    generate_parser.set_defaults(func=_cmd_generate)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate a checkpoint (delegates to lyricgen.evaluate)."
    )
    evaluate_parser.add_argument("args", nargs=argparse.REMAINDER)
    evaluate_parser.set_defaults(func=_cmd_evaluate)

    demo_parser = subparsers.add_parser(
        "demo", help="Launch the Gradio demo (delegates to lyricgen.app)."
    )
    demo_parser.add_argument("args", nargs=argparse.REMAINDER)
    demo_parser.set_defaults(func=_cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
