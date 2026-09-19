from __future__ import annotations

import json

import torch

from lyricgen.dataset import (
    ANY_ARTIST,
    ANY_ARTIST_ID,
    ArtistVocab,
    count_target_chars,
    encode_streams,
    iter_eval_batches,
    load_split,
    sample_batch,
)
from lyricgen.tokenizers import CharTokenizer


def test_artist_vocab_reserves_any_id_zero():
    vocab = ArtistVocab(["alpha", "beta"])
    assert vocab.id(ANY_ARTIST) == ANY_ARTIST_ID
    assert vocab.slug(ANY_ARTIST_ID) == ANY_ARTIST
    assert len(vocab) == 3


def test_artist_vocab_ids_are_stable_and_bidirectional():
    vocab = ArtistVocab(["alpha", "beta"])
    for slug in ["alpha", "beta"]:
        assert vocab.slug(vocab.id(slug)) == slug


def test_artist_vocab_to_from_dict_roundtrip():
    vocab = ArtistVocab(["alpha", "beta", "gamma"])
    restored = ArtistVocab.from_dict(vocab.to_dict())
    assert len(restored) == len(vocab)
    for slug in ["alpha", "beta", "gamma"]:
        assert restored.id(slug) == vocab.id(slug)


def test_artist_vocab_from_records():
    records = [{"artist": "alpha", "text": "x"}, {"artist": "beta", "text": "y"}]
    vocab = ArtistVocab.from_records(records)
    assert vocab.id("alpha") is not None
    assert vocab.id("beta") is not None
    assert len(vocab) == 3


def test_load_split(tmp_path):
    path = tmp_path / "split.jsonl"
    records = [{"artist": "alpha", "text": "hi"}, {"artist": "beta", "text": "yo"}]
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    loaded = load_split(path)
    assert loaded == records


def test_load_split_skips_blank_lines(tmp_path):
    path = tmp_path / "split.jsonl"
    path.write_text('{"artist": "alpha", "text": "hi"}\n\n', encoding="utf-8")
    loaded = load_split(path)
    assert len(loaded) == 1


def _build_streams():
    tokenizer = CharTokenizer.train(["abcdefghijklmnopqrstuvwxyz "])
    vocab = ArtistVocab(["alpha", "beta"])
    records = [
        {"artist": "alpha", "text": "abcdefghij"},
        {"artist": "alpha", "text": "klmnopqrst"},
        {"artist": "beta", "text": "uvwxyz"},
    ]
    streams = encode_streams(records, tokenizer, vocab)
    return tokenizer, vocab, streams


def test_encode_streams_concatenates_per_artist():
    tokenizer, vocab, streams = _build_streams()
    alpha_id = vocab.id("alpha")
    beta_id = vocab.id("beta")
    assert alpha_id in streams and beta_id in streams
    # alpha's two stanzas joined with a blank-line separator
    expected = tokenizer.encode("abcdefghij\n\nklmnopqrst")
    assert streams[alpha_id].tolist() == expected


def test_sample_batch_shapes_and_targets_are_shifted():
    _, vocab, streams = _build_streams()
    gen = torch.Generator().manual_seed(0)
    artists, x, y = sample_batch(streams, block_size=4, batch_size=8, generator=gen)
    assert artists.shape == (8,)
    assert x.shape == (8, 4)
    assert y.shape == (8, 4)
    for a in artists.tolist():
        assert a in streams


def test_sample_batch_never_picks_empty_stream():
    tokenizer = CharTokenizer.train(["ab"])
    vocab = ArtistVocab(["alpha", "beta"])
    records = [
        {"artist": "alpha", "text": "a"},  # too short to give a training window
        {"artist": "beta", "text": "abababababab"},
    ]
    streams = encode_streams(records, tokenizer, vocab)
    gen = torch.Generator().manual_seed(1)
    artists, x, y = sample_batch(streams, block_size=3, batch_size=16, generator=gen)
    assert set(artists.tolist()) == {vocab.id("beta")}


def test_iter_eval_batches_covers_every_stream_deterministically():
    _, vocab, streams = _build_streams()
    all_targets = []
    for artists, x, y in iter_eval_batches(streams, block_size=4, batch_size=2):
        assert x.shape == y.shape
        all_targets.append((artists.clone(), x.clone(), y.clone()))

    # deterministic: running again gives identical batches
    again = list(iter_eval_batches(streams, block_size=4, batch_size=2))
    assert len(again) == len(all_targets)
    for (a1, x1, y1), (a2, x2, y2) in zip(all_targets, again, strict=True):
        assert torch.equal(a1, a2)
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)


def test_iter_eval_batches_pads_final_partial_window_with_zero():
    tokenizer = CharTokenizer.train(["ab"])
    vocab = ArtistVocab(["alpha"])
    records = [{"artist": "alpha", "text": "ababa"}]
    streams = encode_streams(records, tokenizer, vocab)

    batches = list(iter_eval_batches(streams, block_size=4, batch_size=1))
    assert len(batches) == 1
    _, x, y = batches[0]
    # usable_len = len(stream) - 1 = 4, one window of size 4, no padding needed
    assert x.shape == (1, 4)


def test_iter_eval_batches_pads_when_stream_shorter_than_block():
    tokenizer = CharTokenizer.train(["ab"])
    vocab = ArtistVocab(["alpha"])
    records = [{"artist": "alpha", "text": "aba"}]
    streams = encode_streams(records, tokenizer, vocab)

    batches = list(iter_eval_batches(streams, block_size=8, batch_size=1))
    assert len(batches) == 1
    artists, x, y = batches[0]
    assert x.shape == (1, 8)
    # usable_len = 2, so positions 2..7 are zero padding
    assert x[0, 2:].tolist() == [0] * 6


def test_count_target_chars_matches_decoded_length():
    tokenizer, vocab, streams = _build_streams()
    total = count_target_chars(streams, tokenizer)
    expected = 0
    for stream in streams.values():
        expected += len(tokenizer.decode(stream[1:].tolist()))
    assert total == expected


def test_count_target_chars_handles_too_short_stream():
    tokenizer = CharTokenizer.train(["a"])
    vocab = ArtistVocab(["alpha"])
    records = [{"artist": "alpha", "text": "a"}]
    streams = encode_streams(records, tokenizer, vocab)
    assert count_target_chars(streams, tokenizer) == 0
