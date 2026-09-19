"""Build train/val/test JSONL splits from raw artist lyric files."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import time
from pathlib import Path

from lyricgen.data.artists import ARTISTS
from lyricgen.data.clean import clean_text, split_stanzas

logger = logging.getLogger(__name__)

DEFAULT_MIN_STANZA_CHARS = 20


def _normalize_for_dedup(stanza: str) -> str:
    return " ".join(stanza.split()).lower()


def _stanza_key(stanza: str) -> str:
    normalized = _normalize_for_dedup(stanza)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _load_artist_stanzas(
    raw_dir: Path, slug: str, filenames: tuple[str, ...], min_stanza_chars: int
) -> list[str]:
    stanzas: list[str] = []
    for filename in filenames:
        path = raw_dir / filename
        if not path.exists():
            logger.warning("Missing raw file for %s: %s", slug, path)
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        cleaned = clean_text(raw)
        stanzas.extend(split_stanzas(cleaned))
    return [s for s in stanzas if len(s) >= min_stanza_chars]


def _dedupe_within_artist(stanzas: list[str]) -> tuple[list[str], int]:
    seen: set[str] = set()
    kept: list[str] = []
    removed = 0
    for stanza in stanzas:
        key = _stanza_key(stanza)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        kept.append(stanza)
    return kept, removed


def _split_indices(
    n: int, val_frac: float, test_frac: float, rng: random.Random
) -> tuple[list[int], list[int], list[int]]:
    indices = list(range(n))
    rng.shuffle(indices)
    n_val = int(round(n * val_frac))
    n_test = int(round(n * test_frac))
    val_idx = indices[:n_val]
    test_idx = indices[n_val : n_val + n_test]
    train_idx = indices[n_val + n_test :]
    return train_idx, val_idx, test_idx


def prepare(
    raw_dir: Path,
    out_dir: Path,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 1337,
    min_stanza_chars: int = DEFAULT_MIN_STANZA_CHARS,
) -> dict:
    """Clean, dedupe and split raw artist files into train/val/test JSONL.

    Returns the stats dict that is also written to `out_dir/stats.json`.
    """
    start_time = time.perf_counter()
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_artist_records: dict[str, list[str]] = {}
    stanzas_before_dedupe = 0
    within_artist_removed_total = 0

    for slug, info in sorted(ARTISTS.items()):
        raw_stanzas = _load_artist_stanzas(
            raw_dir, slug, info.filenames, min_stanza_chars
        )
        stanzas_before_dedupe += len(raw_stanzas)
        deduped, removed = _dedupe_within_artist(raw_stanzas)
        within_artist_removed_total += removed
        per_artist_records[slug] = deduped

    key_to_slug: dict[str, str] = {}
    cross_artist_removed = 0
    for slug in sorted(per_artist_records):
        kept: list[str] = []
        for stanza in per_artist_records[slug]:
            key = _stanza_key(stanza)
            owner = key_to_slug.get(key)
            if owner is None:
                key_to_slug[key] = slug
                kept.append(stanza)
            elif owner == slug:
                kept.append(stanza)
            else:
                cross_artist_removed += 1
        per_artist_records[slug] = kept

    stanzas_after_dedupe = sum(len(v) for v in per_artist_records.values())

    rng = random.Random(seed)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    per_artist_split_counts: dict[str, dict[str, int]] = {}

    for slug in sorted(per_artist_records):
        stanzas = per_artist_records[slug]
        artist_rng = random.Random(f"{seed}:{slug}")
        train_idx, val_idx, test_idx = _split_indices(
            len(stanzas), val_frac, test_frac, artist_rng
        )
        per_artist_split_counts[slug] = {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        }
        for split_name, idx_list in (
            ("train", train_idx),
            ("val", val_idx),
            ("test", test_idx),
        ):
            for i in idx_list:
                splits[split_name].append({"artist": slug, "text": stanzas[i]})

    total_chars_per_split: dict[str, int] = {}
    for split_name, records in splits.items():
        rng.shuffle(records)
        total_chars_per_split[split_name] = sum(len(r["text"]) for r in records)
        out_path = out_dir / f"{split_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False))
                fh.write("\n")

    elapsed_seconds = time.perf_counter() - start_time

    stats = {
        "artist_count": len(per_artist_records),
        "stanzas_before_dedupe": stanzas_before_dedupe,
        "stanzas_after_dedupe": stanzas_after_dedupe,
        "within_artist_duplicates_removed": within_artist_removed_total,
        "cross_artist_duplicates_removed": cross_artist_removed,
        "total_chars_per_split": total_chars_per_split,
        "records_per_split": {k: len(v) for k, v in splits.items()},
        "per_artist_split_counts": per_artist_split_counts,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "seed": seed,
        "min_stanza_chars": min_stanza_chars,
        "elapsed_seconds": elapsed_seconds,
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return stats


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--min-stanza-chars", type=int, default=DEFAULT_MIN_STANZA_CHARS
    )
    args = parser.parse_args(argv)

    stats = prepare(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        min_stanza_chars=args.min_stanza_chars,
    )
    logger.info("Prepared data: %s", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
