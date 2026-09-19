from __future__ import annotations

import json

import pytest

from lyricgen.data import prepare as prepare_module
from lyricgen.data.artists import ArtistInfo


@pytest.fixture
def fake_artists(monkeypatch):
    """Replace the real artist registry with two small synthetic artists."""
    artists = {
        "alpha": ArtistInfo("Alpha", ("alpha.txt",)),
        "beta": ArtistInfo("Beta", ("beta.txt",)),
    }
    monkeypatch.setattr(prepare_module, "ARTISTS", artists)
    return artists


def _write(path, name, content):
    (path / name).write_text(content, encoding="utf-8")


def test_prepare_writes_splits_and_stats(tmp_path, fake_artists):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "processed"

    stanzas = [f"Stanza number {i} has enough characters to pass" for i in range(40)]
    _write(raw_dir, "alpha.txt", "\n\n".join(stanzas))
    _write(raw_dir, "beta.txt", "\n\n".join(stanzas[:20]))

    stats = prepare_module.prepare(
        raw_dir, out_dir, val_frac=0.2, test_frac=0.2, seed=1, min_stanza_chars=5
    )

    assert stats["artist_count"] == 2
    assert (out_dir / "train.jsonl").exists()
    assert (out_dir / "val.jsonl").exists()
    assert (out_dir / "test.jsonl").exists()
    assert (out_dir / "stats.json").exists()

    on_disk_stats = json.loads((out_dir / "stats.json").read_text(encoding="utf-8"))
    assert on_disk_stats == stats

    total_records = sum(stats["records_per_split"].values())
    assert total_records == stats["stanzas_after_dedupe"]


def test_prepare_removes_within_artist_duplicates(tmp_path, fake_artists):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "processed"

    unique = "This stanza is unique and long enough to count"
    dup_block = "\n\n".join([unique] * 3)
    _write(raw_dir, "alpha.txt", dup_block)
    _write(raw_dir, "beta.txt", "A different unique stanza that is long enough")

    stats = prepare_module.prepare(
        raw_dir, out_dir, val_frac=0.0, test_frac=0.0, seed=1, min_stanza_chars=5
    )

    assert stats["stanzas_before_dedupe"] == 4
    assert stats["within_artist_duplicates_removed"] == 2
    assert stats["stanzas_after_dedupe"] == 2


def test_prepare_removes_cross_artist_duplicates_keeping_first_slug(
    tmp_path, fake_artists
):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "processed"

    shared = "This exact stanza appears for two different artists here"
    _write(raw_dir, "alpha.txt", shared)
    _write(raw_dir, "beta.txt", shared)

    stats = prepare_module.prepare(
        raw_dir, out_dir, val_frac=0.0, test_frac=0.0, seed=1, min_stanza_chars=5
    )

    assert stats["cross_artist_duplicates_removed"] == 1
    assert stats["stanzas_after_dedupe"] == 1

    records = [
        json.loads(line)
        for line in (out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["artist"] == "alpha"


def test_prepare_filters_short_stanzas(tmp_path, fake_artists):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "processed"

    _write(raw_dir, "alpha.txt", "hi\n\nThis one is long enough to survive filtering")
    _write(raw_dir, "beta.txt", "yo")

    stats = prepare_module.prepare(
        raw_dir, out_dir, val_frac=0.0, test_frac=0.0, seed=1, min_stanza_chars=20
    )

    assert stats["stanzas_before_dedupe"] == 1


def test_prepare_is_deterministic_given_seed(tmp_path, fake_artists):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    stanzas = [f"Stanza number {i} has plenty of characters in it" for i in range(30)]
    _write(raw_dir, "alpha.txt", "\n\n".join(stanzas))
    _write(raw_dir, "beta.txt", "\n\n".join(stanzas))

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    prepare_module.prepare(raw_dir, out_a, seed=42, min_stanza_chars=5)
    prepare_module.prepare(raw_dir, out_b, seed=42, min_stanza_chars=5)

    train_a = (out_a / "train.jsonl").read_text(encoding="utf-8")
    train_b = (out_b / "train.jsonl").read_text(encoding="utf-8")
    assert train_a == train_b


def test_prepare_missing_raw_file_is_skipped_gracefully(tmp_path, fake_artists):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "processed"
    _write(raw_dir, "alpha.txt", "Only alpha has a file that actually exists here")
    # beta.txt intentionally missing

    stats = prepare_module.prepare(
        raw_dir, out_dir, val_frac=0.0, test_frac=0.0, seed=1, min_stanza_chars=5
    )
    assert stats["stanzas_after_dedupe"] == 1
