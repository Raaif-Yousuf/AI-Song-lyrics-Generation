"""Render a Markdown comparison table from runs/*/metrics.json and eval.json."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PREFERRED_ORDER = ("lstm", "gru", "transformer", "transformer_bpe")

HEADERS = (
    "model",
    "tokenizer",
    "params",
    "train tok/s",
    "train time (s)",
    "val bpc",
    "test bpc",
    "test char ppl",
    "6-gram novelty (samples/held-out)",
    "gen chars/s",
)


def _sort_key(name: str) -> tuple[int, str]:
    if name in _PREFERRED_ORDER:
        return (_PREFERRED_ORDER.index(name), name)
    return (len(_PREFERRED_ORDER), name)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect_rows(runs_dir: Path) -> list[dict[str, Any]]:
    """One row per `runs/<name>/metrics.json` found, in stable comparison order.

    Runs without a `metrics.json` are skipped (nothing to report); a missing
    `eval.json` just leaves that run's evaluation columns blank.
    """
    if not runs_dir.exists():
        return []

    rows = []
    candidates = (p for p in runs_dir.iterdir() if p.is_dir())
    run_dirs = sorted(candidates, key=lambda p: _sort_key(p.name))
    for run_dir in run_dirs:
        metrics = _load_json(run_dir / "metrics.json")
        if metrics is None:
            continue
        eval_data = _load_json(run_dir / "eval.json") or {}
        rows.append({"name": run_dir.name, "metrics": metrics, "eval": eval_data})
    return rows


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _novelty_pair(eval_data: dict[str, Any], n: int = 6) -> str:
    novelty = eval_data.get("novelty") or {}
    heldout = eval_data.get("novelty_heldout") or {}
    sample_value = novelty.get(str(n), novelty.get(n))
    heldout_value = heldout.get(str(n), heldout.get(n))
    if sample_value is None or heldout_value is None:
        return "-"
    return f"{sample_value:.3f} / {heldout_value:.3f}"


def render_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(HEADERS) + " |",
        "| " + " | ".join(["---"] * len(HEADERS)) + " |",
    ]
    for row in rows:
        metrics = row["metrics"]
        eval_data = row["eval"]
        cells = (
            str(metrics.get("name", row["name"])),
            str(metrics.get("tokenizer", "-")),
            _fmt(metrics.get("params"), 0),
            _fmt(metrics.get("train_tokens_per_sec"), 0),
            _fmt(metrics.get("train_time_s"), 1),
            _fmt(metrics.get("val_bpc")),
            _fmt(eval_data.get("test_bpc")),
            _fmt(eval_data.get("test_char_ppl"), 2),
            _novelty_pair(eval_data),
            _fmt(eval_data.get("gen_chars_per_sec"), 1),
        )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs", help="Directory of run subdirectories")
    parser.add_argument("--out", default=None, help="Optional file to also write the table to.")
    args = parser.parse_args(argv)

    rows = collect_rows(Path(args.runs_dir))
    table = render_table(rows)

    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")
        logger.info("Wrote results table to %s", args.out)

    print(table)


if __name__ == "__main__":
    main()
