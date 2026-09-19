from __future__ import annotations

from lyricgen.tokenizers import (
    PAD_ID,
    UNK_ID,
    BPETokenizer,
    CharTokenizer,
    tokenizer_from_dict,
)

TRAIN_TEXTS = [
    "hello world, this is a small training corpus",
    "another line of text used for training the tokenizer",
    "yet more text so the byte level bpe has something to merge",
]


def test_char_tokenizer_roundtrip():
    tok = CharTokenizer.train(TRAIN_TEXTS)
    text = "hello world"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_char_tokenizer_special_ids():
    tok = CharTokenizer.train(TRAIN_TEXTS)
    assert tok.to_dict()["char_to_id"]["<pad>"] == PAD_ID
    assert tok.to_dict()["char_to_id"]["<unk>"] == UNK_ID


def test_char_tokenizer_maps_unseen_chars_to_unk():
    tok = CharTokenizer.train(["abc"])
    ids = tok.encode("abz")
    assert ids[:2] == tok.encode("ab")
    assert ids[2] == UNK_ID


def test_char_tokenizer_to_from_dict_roundtrip():
    tok = CharTokenizer.train(TRAIN_TEXTS)
    restored = CharTokenizer.from_dict(tok.to_dict())
    text = "hello world"
    assert restored.encode(text) == tok.encode(text)
    assert restored.vocab_size == tok.vocab_size


def test_char_tokenizer_save_load(tmp_path):
    tok = CharTokenizer.train(TRAIN_TEXTS)
    path = tmp_path / "char.json"
    tok.save(path)
    restored = CharTokenizer.load(path)
    assert restored.encode("hello") == tok.encode("hello")


def test_char_tokenizer_from_dict_via_dispatch():
    tok = CharTokenizer.train(TRAIN_TEXTS)
    restored = tokenizer_from_dict(tok.to_dict())
    assert isinstance(restored, CharTokenizer)
    assert restored.encode("hello") == tok.encode("hello")


def test_bpe_tokenizer_special_ids():
    tok = BPETokenizer.train(TRAIN_TEXTS, vocab_size=200, min_frequency=1)
    assert tok.to_dict()["type"] == "bpe"
    # pad/unk must occupy ids 0 and 1 per the shared tokenizer contract
    d = tok.to_dict()
    restored = BPETokenizer.from_dict(d)
    assert restored.vocab_size == tok.vocab_size


def test_bpe_tokenizer_roundtrip_on_seen_text():
    tok = BPETokenizer.train(TRAIN_TEXTS, vocab_size=200, min_frequency=1)
    text = "hello world"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_bpe_tokenizer_to_from_dict_roundtrip():
    tok = BPETokenizer.train(TRAIN_TEXTS, vocab_size=200, min_frequency=1)
    restored = BPETokenizer.from_dict(tok.to_dict())
    text = "another line of text"
    assert restored.encode(text) == tok.encode(text)


def test_bpe_tokenizer_save_load(tmp_path):
    tok = BPETokenizer.train(TRAIN_TEXTS, vocab_size=200, min_frequency=1)
    path = tmp_path / "bpe.json"
    tok.save(path)
    restored = BPETokenizer.load(path)
    text = "hello world"
    assert restored.encode(text) == tok.encode(text)


def test_bpe_tokenizer_from_dict_via_dispatch():
    tok = BPETokenizer.train(TRAIN_TEXTS, vocab_size=200, min_frequency=1)
    restored = tokenizer_from_dict(tok.to_dict())
    assert isinstance(restored, BPETokenizer)


def test_tokenizer_from_dict_rejects_unknown_type():
    import pytest

    with pytest.raises(ValueError):
        tokenizer_from_dict({"type": "not-a-real-type"})
