from __future__ import annotations

import json
from pathlib import Path

import torch
from helpers import make_tiny_config

from lyricgen.checkpoint import load_checkpoint
from lyricgen.train import train


def test_train_writes_full_run_directory_layout(tiny_config, tmp_path):
    train(tiny_config)
    out_dir = Path(tiny_config.out_dir)

    expected_files = (
        "config.yaml",
        "tokenizer.json",
        "best.pt",
        "last.pt",
        "log.jsonl",
        "metrics.json",
    )
    for name in expected_files:
        assert (out_dir / name).exists(), f"missing {name}"


def test_train_log_entries_have_expected_fields(tiny_config):
    train(tiny_config)
    out_dir = Path(tiny_config.out_dir)
    lines = (out_dir / "log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) > 0
    expected_keys = {
        "step",
        "tokens_seen",
        "train_loss",
        "val_loss",
        "val_bpc",
        "lr",
        "elapsed_s",
        "tokens_per_sec",
    }
    for line in lines:
        entry = json.loads(line)
        assert expected_keys <= set(entry.keys())


def test_train_metrics_json_has_required_fields(tiny_config):
    metrics = train(tiny_config)
    out_dir = Path(tiny_config.out_dir)
    on_disk = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert on_disk == metrics

    expected_keys = {
        "name",
        "kind",
        "tokenizer",
        "params",
        "steps",
        "tokens_seen",
        "train_time_s",
        "train_tokens_per_sec",
        "best_step",
        "val_loss",
        "val_bpc",
        "val_char_ppl",
    }
    assert expected_keys <= set(metrics.keys())
    assert metrics["steps"] == tiny_config.train.steps
    assert metrics["params"] > 0


def test_train_loss_goes_down(tiny_config):
    train(tiny_config)
    out_dir = Path(tiny_config.out_dir)
    lines = (out_dir / "log.jsonl").read_text(encoding="utf-8").splitlines()
    entries = [json.loads(line) for line in lines]
    assert entries[-1]["val_loss"] < entries[0]["val_loss"]


def test_train_is_deterministic_given_seed(tiny_data_dir, tmp_path):
    cfg_a = make_tiny_config(tiny_data_dir, tmp_path / "run_a")
    cfg_b = make_tiny_config(tiny_data_dir, tmp_path / "run_b")

    metrics_a = train(cfg_a)
    metrics_b = train(cfg_b)

    assert metrics_a["val_loss"] == metrics_b["val_loss"]
    assert metrics_a["val_bpc"] == metrics_b["val_bpc"]

    # Compare log entries excluding wall-clock fields, which legitimately vary.
    def _stable_fields(entry: dict) -> dict:
        return {k: v for k, v in entry.items() if k not in ("elapsed_s", "tokens_per_sec")}

    log_a = [
        _stable_fields(json.loads(line))
        for line in (Path(cfg_a.out_dir) / "log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    log_b = [
        _stable_fields(json.loads(line))
        for line in (Path(cfg_b.out_dir) / "log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert log_a == log_b


def test_checkpoint_round_trip_gives_identical_logits(tiny_config):
    train(tiny_config)
    out_dir = Path(tiny_config.out_dir)
    loaded = load_checkpoint(out_dir / "last.pt")

    tokens = torch.zeros(1, 4, dtype=torch.long)
    artists = torch.zeros(1, dtype=torch.long)
    with torch.inference_mode():
        logits, _ = loaded.model(tokens, artists)

    # Loading the same checkpoint again must reproduce identical weights.
    loaded_again = load_checkpoint(out_dir / "last.pt")
    with torch.inference_mode():
        logits_again, _ = loaded_again.model(tokens, artists)

    assert torch.equal(logits, logits_again)


def test_train_respects_artist_dropout_zero(tiny_data_dir, tmp_path):
    cfg = make_tiny_config(tiny_data_dir, tmp_path / "run", train={"artist_dropout": 0.0})
    metrics = train(cfg)
    assert metrics["steps"] == cfg.train.steps


def test_train_with_bpe_tokenizer(tiny_data_dir, tmp_path):
    cfg = make_tiny_config(
        tiny_data_dir,
        tmp_path / "run",
        tokenizer_type="bpe",
    )
    cfg.tokenizer.vocab_size = 64
    metrics = train(cfg)
    assert metrics["tokenizer"]["type"] == "bpe"
    assert (Path(cfg.out_dir) / "tokenizer.json").exists()


def test_train_with_lstm_model(tiny_data_dir, tmp_path):
    cfg = make_tiny_config(
        tiny_data_dir,
        tmp_path / "run",
        model_kind="lstm",
        model_hyperparameters={
            "embed_dim": 8,
            "artist_embed_dim": 4,
            "hidden_sizes": (8, 8),
            "dropout": 0.0,
        },
    )
    metrics = train(cfg)
    assert metrics["kind"] == "lstm"
