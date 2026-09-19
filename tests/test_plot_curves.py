from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "plot_curves.py"
_SPEC = importlib.util.spec_from_file_location("plot_curves", SCRIPT_PATH)
plot_curves = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("plot_curves", plot_curves)
_SPEC.loader.exec_module(plot_curves)


def _entry(step: int, tokens_seen: int, val_bpc: float) -> dict:
    return {
        "step": step,
        "tokens_seen": tokens_seen,
        "train_loss": 2.0,
        "val_loss": 1.9,
        "val_bpc": val_bpc,
        "lr": 0.0003,
        "elapsed_s": 10.0,
        "tokens_per_sec": 1000.0,
    }


def _write_log(run_dir: Path, entries: list[dict]) -> None:
    run_dir.mkdir(parents=True)
    lines = [json.dumps(entry) for entry in entries]
    (run_dir / "log.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_run_curve_reads_tokens_in_millions_and_bpc(tmp_path: Path) -> None:
    run_dir = tmp_path / "lstm"
    _write_log(run_dir, [_entry(100, 1_000_000, 2.0), _entry(200, 2_500_000, 1.5)])

    tokens_millions, val_bpc = plot_curves.load_run_curve(run_dir / "log.jsonl")

    assert tokens_millions == [1.0, 2.5]
    assert val_bpc == [2.0, 1.5]


def test_load_run_curve_sorts_by_step(tmp_path: Path) -> None:
    run_dir = tmp_path / "lstm"
    _write_log(run_dir, [_entry(200, 2_000_000, 1.5), _entry(100, 1_000_000, 2.0)])

    tokens_millions, val_bpc = plot_curves.load_run_curve(run_dir / "log.jsonl")

    assert tokens_millions == [1.0, 2.0]
    assert val_bpc == [2.0, 1.5]


def test_load_run_curve_missing_file_returns_empty(tmp_path: Path) -> None:
    tokens_millions, val_bpc = plot_curves.load_run_curve(tmp_path / "missing" / "log.jsonl")
    assert tokens_millions == []
    assert val_bpc == []


def test_collect_curves_skips_runs_without_entries(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _write_log(runs_dir / "lstm", [_entry(100, 1_000_000, 2.0)])
    (runs_dir / "empty").mkdir(parents=True)

    curves = plot_curves.collect_curves(runs_dir)

    assert set(curves) == {"lstm"}


def test_collect_curves_orders_known_kinds_first(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _write_log(runs_dir / "transformer", [_entry(100, 1_000_000, 2.0)])
    _write_log(runs_dir / "lstm", [_entry(100, 1_000_000, 2.0)])
    _write_log(runs_dir / "gru", [_entry(100, 1_000_000, 2.0)])

    curves = plot_curves.collect_curves(runs_dir)

    assert list(curves) == ["lstm", "gru", "transformer"]


def test_collect_curves_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert plot_curves.collect_curves(tmp_path / "does-not-exist") == {}


def test_plot_curves_writes_a_png(tmp_path: Path) -> None:
    curves = {
        "lstm": ([1.0, 2.0, 3.0], [3.0, 2.0, 1.5]),
        "transformer": ([1.0, 2.0, 3.0], [2.5, 1.8, 1.2]),
    }
    out_path = tmp_path / "curves.png"

    plot_curves.plot_curves(curves, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_main_end_to_end_writes_png(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _write_log(runs_dir / "lstm", [_entry(100, 1_000_000, 2.0), _entry(200, 2_000_000, 1.5)])
    _write_log(runs_dir / "transformer", [_entry(100, 1_000_000, 1.9), _entry(200, 2_000_000, 1.2)])
    out_path = tmp_path / "out" / "curves.png"

    plot_curves.main(["--runs-dir", str(runs_dir), "--out", str(out_path)])

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_main_with_no_runs_still_writes_an_empty_plot(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    out_path = tmp_path / "curves.png"

    plot_curves.main(["--runs-dir", str(runs_dir), "--out", str(out_path)])

    assert out_path.exists()
