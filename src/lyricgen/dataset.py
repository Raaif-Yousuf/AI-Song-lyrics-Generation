"""Artist vocabulary and batching utilities over encoded lyric streams."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import torch

ANY_ARTIST = "<any>"
ANY_ARTIST_ID = 0


class ArtistVocab:
    """Bidirectional mapping between artist slugs and integer ids.

    Id 0 is always reserved for `"<any>"`, the unconditional/no-artist id.
    """

    def __init__(self, slugs: Iterable[str]) -> None:
        ordered = [s for s in slugs if s != ANY_ARTIST]
        self._slug_to_id = {ANY_ARTIST: ANY_ARTIST_ID}
        self._id_to_slug = {ANY_ARTIST_ID: ANY_ARTIST}
        for i, slug in enumerate(ordered, start=1):
            self._slug_to_id[slug] = i
            self._id_to_slug[i] = slug

    def id(self, slug: str) -> int:
        return self._slug_to_id[slug]

    def slug(self, artist_id: int) -> str:
        return self._id_to_slug[artist_id]

    def __len__(self) -> int:
        return len(self._slug_to_id)

    def to_dict(self) -> dict:
        ordered_slugs = [
            self._id_to_slug[i] for i in range(1, len(self._id_to_slug))
        ]
        return {"slugs": ordered_slugs}

    @classmethod
    def from_dict(cls, d: dict) -> ArtistVocab:
        return cls(d["slugs"])

    @classmethod
    def from_records(cls, records: Iterable[dict]) -> ArtistVocab:
        slugs = sorted({r["artist"] for r in records})
        return cls(slugs)


def load_split(path: str | Path) -> list[dict]:
    """Read a JSONL split file into a list of `{"artist", "text"}` dicts."""
    records = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def encode_streams(
    records: Iterable[dict], tokenizer, artist_vocab: ArtistVocab
) -> dict[int, torch.Tensor]:
    """Concatenate each artist's stanzas and encode into one long tensor."""
    texts_by_artist: dict[int, list[str]] = {}
    for record in records:
        artist_id = artist_vocab.id(record["artist"])
        texts_by_artist.setdefault(artist_id, []).append(record["text"])

    streams: dict[int, torch.Tensor] = {}
    for artist_id, texts in texts_by_artist.items():
        joined = "\n\n".join(texts)
        ids = tokenizer.encode(joined)
        streams[artist_id] = torch.tensor(ids, dtype=torch.long)
    return streams


def sample_batch(
    streams: dict[int, torch.Tensor],
    block_size: int,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample random windows, one artist stream chosen per example.

    Artists are chosen with probability proportional to their stream
    length. Returns `(artists[B], x[B,T], y[B,T])` where `y` is `x` shifted
    by one position.
    """
    artist_ids = list(streams.keys())
    lengths = torch.tensor(
        [max(len(streams[a]) - 1, 0) for a in artist_ids], dtype=torch.float
    )
    if lengths.sum() <= 0:
        raise ValueError("No stream has enough tokens for a training window")
    probs = lengths / lengths.sum()

    chosen = torch.multinomial(
        probs, num_samples=batch_size, replacement=True, generator=generator
    )

    batch_artists = torch.empty(batch_size, dtype=torch.long)
    x = torch.zeros(batch_size, block_size, dtype=torch.long)
    y = torch.zeros(batch_size, block_size, dtype=torch.long)

    for row, idx in enumerate(chosen.tolist()):
        artist_id = artist_ids[idx]
        stream = streams[artist_id]
        usable_len = len(stream) - 1
        window = min(block_size, usable_len)
        max_start = usable_len - window
        start = int(torch.randint(0, max_start + 1, (1,), generator=generator).item())
        batch_artists[row] = artist_id
        x[row, :window] = stream[start : start + window]
        y[row, :window] = stream[start + 1 : start + 1 + window]

    return batch_artists, x, y


def iter_eval_batches(
    streams: dict[int, torch.Tensor], block_size: int, batch_size: int
):
    """Yield deterministic, non-overlapping windows covering every stream.

    The final partial window of each stream is included, padded with id 0
    (`<pad>`); callers should use `ignore_index=0` in their loss so pad
    positions never contribute (pad id 0 is never a real target).
    """
    windows: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    for artist_id, stream in streams.items():
        usable_len = len(stream) - 1
        for start in range(0, usable_len, block_size):
            window = min(block_size, usable_len - start)
            x = torch.zeros(block_size, dtype=torch.long)
            y = torch.zeros(block_size, dtype=torch.long)
            x[:window] = stream[start : start + window]
            y[:window] = stream[start + 1 : start + 1 + window]
            windows.append((artist_id, x, y))

    for batch_start in range(0, len(windows), batch_size):
        chunk = windows[batch_start : batch_start + batch_size]
        artists = torch.tensor([w[0] for w in chunk], dtype=torch.long)
        x = torch.stack([w[1] for w in chunk])
        y = torch.stack([w[2] for w in chunk])
        yield artists, x, y


def count_target_chars(streams: dict[int, torch.Tensor], tokenizer) -> int:
    """Total number of characters represented by all eval targets.

    Used as the denominator for bits-per-char: the eval targets are every
    token from index 1 onward in each stream, decoded back to text.
    """
    total = 0
    for stream in streams.values():
        usable_len = len(stream) - 1
        if usable_len <= 0:
            continue
        target_ids = stream[1 : 1 + usable_len].tolist()
        total += len(tokenizer.decode(target_ids))
    return total
