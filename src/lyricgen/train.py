"""Training loop: seeded, artist-conditioned, with periodic evaluation."""

from __future__ import annotations

import itertools
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import nn

from lyricgen.checkpoint import load_checkpoint, save_checkpoint
from lyricgen.config import ExperimentConfig
from lyricgen.dataset import (
    ANY_ARTIST_ID,
    ArtistVocab,
    count_target_chars,
    encode_streams,
    iter_eval_batches,
    load_split,
    sample_batch,
)
from lyricgen.metrics import bits_per_char, char_perplexity
from lyricgen.models import build_model
from lyricgen.tokenizers import BPETokenizer, CharTokenizer

logger = logging.getLogger(__name__)


def _build_tokenizer(kind: str, vocab_size: int | None, texts: list[str]):
    if kind == "char":
        return CharTokenizer.train(texts)
    if kind == "bpe":
        return BPETokenizer.train(texts, vocab_size=vocab_size)
    raise ValueError(f"unknown tokenizer type: {kind!r}")


def _lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    denom = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / denom, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _eval_loss(
    model,
    streams: dict[int, torch.Tensor],
    block_size: int,
    batch_size: int,
    max_batches: int | None,
) -> tuple[float, float, int]:
    """Mean nats/token, total nats (sum) and total valid target count.

    Valid targets exclude padding (target id 0 is never a real token).
    """
    was_training = model.training
    model.eval()
    total_nll_sum = 0.0
    total_valid = 0
    batches = iter_eval_batches(streams, block_size, batch_size)
    if max_batches is not None:
        batches = itertools.islice(batches, max_batches)
    with torch.inference_mode():
        for artists, x, y in batches:
            logits, _ = model(x, artists)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=0,
                reduction="sum",
            )
            valid = int((y != 0).sum().item())
            total_nll_sum += float(loss_sum.item())
            total_valid += valid
    model.train(was_training)
    mean_nats_per_token = total_nll_sum / total_valid if total_valid else 0.0
    return mean_nats_per_token, total_nll_sum, total_valid


def train(config: ExperimentConfig) -> dict[str, Any]:
    """Train a model per `config`, writing the full run directory.

    Returns the same dict that is written to `metrics.json`.
    """
    seed = config.train.seed
    random.seed(seed)
    torch.manual_seed(seed)
    if config.train.num_threads:
        torch.set_num_threads(config.train.num_threads)
    device = torch.device(config.train.device)

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_split(config.data.train_path)
    val_records = load_split(config.data.val_path)
    train_texts = [r["text"] for r in train_records]

    tokenizer = _build_tokenizer(config.tokenizer.type, config.tokenizer.vocab_size, train_texts)
    artist_vocab = ArtistVocab.from_records(train_records)

    train_streams = encode_streams(train_records, tokenizer, artist_vocab)
    val_streams = encode_streams(val_records, tokenizer, artist_vocab)

    model_config = {
        "kind": config.model.kind,
        "vocab_size": tokenizer.vocab_size,
        "n_artists": len(artist_vocab),
        **config.model.hyperparameters,
    }
    model = build_model(model_config)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_lambda(step, config.train.warmup_steps, config.train.steps),
    )

    data_generator = torch.Generator().manual_seed(seed)

    total_chars_full = count_target_chars(val_streams, tokenizer)
    total_tokens_full = sum(max(len(s) - 1, 0) for s in val_streams.values())
    chars_per_token_full = total_chars_full / total_tokens_full if total_tokens_full else 1.0

    log_path = out_dir / "log.jsonl"
    log_entries: list[dict[str, Any]] = []

    best_val_loss = math.inf
    best_step = 0
    tokens_seen = 0
    eval_time_total = 0.0
    train_start = time.perf_counter()

    for step in range(config.train.steps):
        artists, x, y = sample_batch(
            train_streams, config.train.block_size, config.train.batch_size, data_generator
        )
        if config.train.artist_dropout > 0:
            dropout_rate = config.train.artist_dropout
            drop_mask = torch.rand(artists.shape, generator=data_generator) < dropout_rate
            artists = artists.clone()
            artists[drop_mask] = ANY_ARTIST_ID

        artists = artists.to(device)
        x = x.to(device)
        y = y.to(device)

        logits, _ = model(x, artists)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=0)

        optimizer.zero_grad()
        loss.backward()
        if config.train.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        optimizer.step()
        scheduler.step()

        tokens_seen += x.numel()

        step_num = step + 1
        is_last_step = step_num == config.train.steps
        if step_num % config.train.eval_interval == 0 or is_last_step:
            eval_start = time.perf_counter()
            val_loss, _, _ = _eval_loss(
                model,
                val_streams,
                config.train.block_size,
                config.train.batch_size,
                config.train.eval_batches,
            )
            eval_time_total += time.perf_counter() - eval_start

            elapsed = time.perf_counter() - train_start - eval_time_total
            tokens_per_sec = tokens_seen / elapsed if elapsed > 0 else 0.0
            val_bpc_estimate = val_loss / chars_per_token_full / math.log(2)

            entry = {
                "step": step_num,
                "tokens_seen": tokens_seen,
                "train_loss": float(loss.item()),
                "val_loss": val_loss,
                "val_bpc": val_bpc_estimate,
                "lr": scheduler.get_last_lr()[0],
                "elapsed_s": elapsed,
                "tokens_per_sec": tokens_per_sec,
            }
            log_entries.append(entry)
            logger.info(
                "step %d/%d train_loss=%.4f val_loss=%.4f val_bpc=%.4f tok/s=%.0f",
                step_num,
                config.train.steps,
                entry["train_loss"],
                val_loss,
                val_bpc_estimate,
                tokens_per_sec,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step_num
                save_checkpoint(
                    out_dir / "best.pt", model, tokenizer, artist_vocab, config.to_dict()
                )

    train_time_s = time.perf_counter() - train_start - eval_time_total
    train_tokens_per_sec = tokens_seen / train_time_s if train_time_s > 0 else 0.0

    save_checkpoint(out_dir / "last.pt", model, tokenizer, artist_vocab, config.to_dict())

    best_path = out_dir / "best.pt"
    if not best_path.exists():
        # steps < eval_interval and the loop never evaluated; fall back to
        # the final model so a best checkpoint always exists.
        save_checkpoint(best_path, model, tokenizer, artist_vocab, config.to_dict())
        best_step = config.train.steps

    with log_path.open("w", encoding="utf-8") as fh:
        for entry in log_entries:
            fh.write(_json_line(entry))

    tokenizer.save(out_dir / "tokenizer.json")
    (out_dir / "config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )

    loaded = load_checkpoint(best_path, device=str(device))
    full_val_loss, full_val_nll_sum, _ = _eval_loss(
        loaded.model, val_streams, config.train.block_size, config.train.batch_size, None
    )
    val_bpc = bits_per_char(full_val_nll_sum, total_chars_full)
    val_char_ppl = char_perplexity(full_val_nll_sum, total_chars_full)

    metrics = {
        "name": config.name,
        "kind": config.model.kind,
        "tokenizer": config.tokenizer.to_dict(),
        "params": model.num_parameters(),
        "steps": config.train.steps,
        "tokens_seen": tokens_seen,
        "train_time_s": train_time_s,
        "train_tokens_per_sec": train_tokens_per_sec,
        "best_step": best_step,
        "val_loss": full_val_loss,
        "val_bpc": val_bpc,
        "val_char_ppl": val_char_ppl,
    }
    (out_dir / "metrics.json").write_text(_json_dump(metrics), encoding="utf-8")

    save_checkpoint(
        best_path, loaded.model, tokenizer, artist_vocab, config.to_dict(), metrics=metrics
    )

    return metrics


def _json_line(entry: dict[str, Any]) -> str:
    return json.dumps(entry, ensure_ascii=False) + "\n"


def _json_dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)
