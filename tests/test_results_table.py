from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "results_table.py"
_SPEC = importlib.util.spec_from_file_location("results_table", SCRIPT_PATH)
results_table = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("results_table", results_table)
_SPEC.loader.exec_module(results_table)


def _write_run(runs_dir: Path, name: str, metrics: dict, eval_data: dict | None = None) -> None:
    run_dir = runs_dir / name
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    if eval_data is not None:
        (run_dir / "eval.json").write_text(json.dumps(eval_data), encoding="utf-8")


def _metrics(name: str, kind: str) -> dict:
    return {
        "name": name,
        "kind": kind,
        "tokenizer": "char",
        "params": 12345,
        "train_tokens_per_sec": 45000.0,
        "train_time_s": 720.5,
        "val_bpc": 1.25,
    }


def _eval(sample_novelty: float, heldout_novelty: float) -> dict:
    return {
        "test_bpc": 1.5,
        "test_char_ppl": 2.75,
        "gen_chars_per_sec": 500.0,
        "novelty": {"4": 0.9, "6": sample_novelty, "8": 0.95},
        "novelty_heldout": {"4": 0.5, "6": heldout_novelty, "8": 0.6},
    }


def test_collect_rows_orders_known_kinds_first_then_alphabetically(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _write_run(runs_dir, "transformer", _metrics("transformer", "transformer"), _eval(0.8, 0.4))
    _write_run(runs_dir, "zzz-other", _metrics("zzz-other", "transformer"))
    _write_run(runs_dir, "lstm", _metrics("lstm", "lstm"), _eval(0.7, 0.3))
    _write_run(runs_dir, "gru", _metrics("gru", "gru"))
    (runs_dir / "no_metrics").mkdir(parents=True)

    rows = results_table.collect_rows(runs_dir)

    assert [row["name"] for row in rows] == ["lstm", "gru", "transformer", "zzz-other"]


def test_collect_rows_missing_runs_dir_returns_empty(tmp_path: Path) -> None:
    assert results_table.collect_rows(tmp_path / "does-not-exist") == []


def test_render_table_has_header_and_separator_and_rows(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _write_run(runs_dir, "lstm", _metrics("lstm", "lstm"), _eval(0.8, 0.4))
    rows = results_table.collect_rows(runs_dir)

    table = results_table.render_table(rows)
    lines = table.splitlines()

    assert lines[0].startswith("| model")
    assert set(lines[1].replace("|", "").replace(" ", "")) == {"-"}
    assert len(lines) == 3
    assert "lstm" in lines[2]
    assert "0.800 / 0.400" in lines[2]
    assert "1.250" in lines[2]  # val_bpc rounded to 3 decimals


def test_render_table_handles_missing_eval_json(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _write_run(runs_dir, "gru", _metrics("gru", "gru"))
    rows = results_table.collect_rows(runs_dir)

    table = results_table.render_table(rows)

    assert "gru" in table
    assert "| - |" in table


def test_main_writes_table_to_out_file(tmp_path: Path, capsys) -> None:
    runs_dir = tmp_path / "runs"
    _write_run(runs_dir, "lstm", _metrics("lstm", "lstm"), _eval(0.8, 0.4))
    out_path = tmp_path / "table.md"

    results_table.main(["--runs-dir", str(runs_dir), "--out", str(out_path)])

    printed = capsys.readouterr().out.strip()
    written = out_path.read_text(encoding="utf-8").strip()
    assert printed == written
    assert "lstm" in written


def test_main_with_no_runs_prints_header_only(tmp_path: Path, capsys) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    results_table.main(["--runs-dir", str(runs_dir)])

    printed = capsys.readouterr().out.strip().splitlines()
    assert len(printed) == 2
