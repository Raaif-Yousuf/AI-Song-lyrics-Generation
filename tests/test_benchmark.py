from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "benchmark.py"
_SPEC = importlib.util.spec_from_file_location("benchmark", SCRIPT_PATH)
benchmark = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("benchmark", benchmark)
_SPEC.loader.exec_module(benchmark)

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def test_run_benchmark_tiny_covers_every_config() -> None:
    rows = benchmark.run_benchmark(
        configs_dir=CONFIGS_DIR,
        vocab_size=37,
        n_artists=5,
        threads=1,
        train_steps=2,
        train_warmup=1,
        gen_prompt_len=3,
        gen_tokens=5,
        gen_repeats=1,
        tiny=True,
    )

    names = {row["name"] for row in rows}
    assert names == {"lstm", "gru", "transformer", "transformer_bpe"}
    for row in rows:
        assert row["params"] > 0
        assert row["train_tokens_per_sec"] > 0
        assert row["gen_tokens_per_sec"] > 0
        assert row["batch_size"] == benchmark._TINY_BATCH_SIZE
        assert row["block_size"] == benchmark._TINY_BLOCK_SIZE


def test_run_benchmark_uses_bpe_vocab_size_for_transformer_bpe() -> None:
    rows = benchmark.run_benchmark(
        configs_dir=CONFIGS_DIR,
        vocab_size=37,
        n_artists=5,
        threads=1,
        train_steps=1,
        train_warmup=0,
        gen_prompt_len=3,
        gen_tokens=3,
        gen_repeats=1,
        tiny=True,
    )

    by_name = {row["name"]: row for row in rows}
    assert by_name["transformer_bpe"]["vocab_size"] == 2000
    assert by_name["lstm"]["vocab_size"] == 37
    assert by_name["transformer"]["vocab_size"] == 37


def test_run_benchmark_skips_missing_config(tmp_path: Path) -> None:
    (tmp_path / "lstm.yaml").write_text((CONFIGS_DIR / "lstm.yaml").read_text(encoding="utf-8"))

    rows = benchmark.run_benchmark(
        configs_dir=tmp_path,
        vocab_size=37,
        n_artists=5,
        threads=1,
        train_steps=1,
        train_warmup=0,
        gen_prompt_len=3,
        gen_tokens=3,
        gen_repeats=1,
        tiny=True,
    )

    assert [row["name"] for row in rows] == ["lstm"]


def test_render_table_has_header_separator_and_rows() -> None:
    rows = [
        {
            "name": "lstm",
            "kind": "lstm",
            "tokenizer": "char",
            "vocab_size": 160,
            "params": 12345,
            "batch_size": 64,
            "block_size": 256,
            "train_tokens_per_sec": 40000.0,
            "gen_tokens_per_sec": 500.0,
        }
    ]

    table = benchmark.render_table(rows)
    lines = table.splitlines()

    assert lines[0].startswith("| model")
    assert set(lines[1].replace("|", "").replace(" ", "")) == {"-"}
    assert len(lines) == 3
    assert "lstm" in lines[2]
    assert "64x256" in lines[2]
    assert "12,345" in lines[2]


def test_main_tiny_run_prints_table_and_writes_json(tmp_path: Path, capsys) -> None:
    out_json = tmp_path / "bench.json"

    benchmark.main(
        [
            "--configs-dir",
            str(CONFIGS_DIR),
            "--vocab-size",
            "37",
            "--n-artists",
            "5",
            "--threads",
            "1",
            "--steps",
            "2",
            "--warmup",
            "1",
            "--gen-prompt-len",
            "3",
            "--gen-tokens",
            "5",
            "--gen-repeats",
            "1",
            "--tiny",
            "--out-json",
            str(out_json),
        ]
    )

    printed = capsys.readouterr().out
    assert "lstm" in printed
    assert "transformer_bpe" in printed

    rows = json.loads(out_json.read_text(encoding="utf-8"))
    assert {row["name"] for row in rows} == {"lstm", "gru", "transformer", "transformer_bpe"}
