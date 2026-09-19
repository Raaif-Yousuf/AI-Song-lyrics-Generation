"""Plot validation bits-per-character curves from runs/*/log.jsonl."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PREFERRED_ORDER = ("lstm", "gru", "transformer", "transformer_bpe")
_COLORS = {
    "lstm": "#1b9e77",
    "gru": "#d95f02",
    "transformer": "#7570b3",
    "transformer_bpe": "#e7298a",
}
_DEFAULT_COLOR = "#666666"

Curve = tuple[list[float], list[float]]


def _sort_key(name: str) -> tuple[int, str]:
    if name in _PREFERRED_ORDER:
        return (_PREFERRED_ORDER.index(name), name)
    return (len(_PREFERRED_ORDER), name)


def load_run_curve(log_path: Path) -> Curve:
    """Read one run's log.jsonl into (training tokens in millions, val bpc).

    Entries are sorted by step. Returns two empty lists if the file does
    not exist or has no entries yet (e.g. a run that has not reached its
    first eval).
    """
    if not log_path.exists():
        return [], []

    entries: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    entries.sort(key=lambda entry: entry.get("step", 0))

    tokens_millions = [entry["tokens_seen"] / 1e6 for entry in entries]
    val_bpc = [entry["val_bpc"] for entry in entries]
    return tokens_millions, val_bpc


def collect_curves(runs_dir: Path) -> dict[str, Curve]:
    """Load every `runs/<name>/log.jsonl` that has at least one entry.

    Runs are returned in a stable order (known model kinds first, then
    alphabetically) so plot colors and legend order are consistent across
    invocations.
    """
    curves: dict[str, Curve] = {}
    if not runs_dir.exists():
        return curves

    run_dirs = sorted(
        (p for p in runs_dir.iterdir() if p.is_dir()), key=lambda p: _sort_key(p.name)
    )
    for run_dir in run_dirs:
        tokens_millions, val_bpc = load_run_curve(run_dir / "log.jsonl")
        if tokens_millions:
            curves[run_dir.name] = (tokens_millions, val_bpc)
    return curves


def plot_curves(
    curves: dict[str, Curve],
    out_path: Path,
    width_px: int = 1000,
    height_px: int = 560,
    dpi: int = 150,
) -> None:
    """Render validation bits-per-char vs. training tokens to a PNG.

    Requires matplotlib (imported lazily so this module can be imported,
    and other functions in it used, without matplotlib installed). Uses
    the non-interactive Agg backend so it runs headless.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for name, (tokens_millions, val_bpc) in curves.items():
        color = _COLORS.get(name, _DEFAULT_COLOR)
        ax.plot(tokens_millions, val_bpc, color=color, linewidth=1.8, solid_capstyle="round")
        ax.annotate(
            name,
            xy=(tokens_millions[-1], val_bpc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=color,
        )

    ax.set_xlabel("training tokens (millions)")
    ax.set_ylabel("validation bits per character")
    ax.set_title("Validation bits-per-character during training")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs", help="Directory of run subdirectories")
    parser.add_argument("--out", default="docs/assets/curves.png", help="Output PNG path")
    parser.add_argument("--width", type=int, default=1000, help="Image width in pixels")
    parser.add_argument("--height", type=int, default=560, help="Image height in pixels")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args(argv)

    curves = collect_curves(Path(args.runs_dir))
    if not curves:
        logger.warning("no run logs with data found under %s", args.runs_dir)

    plot_curves(curves, Path(args.out), args.width, args.height, args.dpi)
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
