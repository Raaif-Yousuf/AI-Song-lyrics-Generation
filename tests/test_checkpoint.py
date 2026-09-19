from __future__ import annotations

import pytest
import torch

from lyricgen.checkpoint import LoadedModel, load_checkpoint, save_checkpoint
from lyricgen.dataset import ArtistVocab
from lyricgen.models import build_model
from lyricgen.tokenizers import CharTokenizer


def _build_tiny_model(vocab_size: int, n_artists: int):
    return build_model(
        {
            "kind": "transformer",
            "vocab_size": vocab_size,
            "n_artists": n_artists,
            "n_layer": 1,
            "n_head": 1,
            "d_model": 8,
            "block_size": 8,
            "dropout": 0.0,
        }
    )


def _make_pieces():
    # Seed model init explicitly so these tests do not depend on whatever
    # global torch RNG state earlier tests in the same process left behind
    # (an unseeded tiny weight-tied model can otherwise degenerate into
    # always repeating the same token).
    torch.manual_seed(0)
    tokenizer = CharTokenizer.train(["hello world\ngoodbye world", "goodbye\nworld hello"])
    artist_vocab = ArtistVocab(["alpha", "beta"])
    model = _build_tiny_model(tokenizer.vocab_size, len(artist_vocab))
    return model, tokenizer, artist_vocab


def test_save_and_load_checkpoint_round_trip(tmp_path):
    model, tokenizer, artist_vocab = _make_pieces()
    model.eval()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, tokenizer, artist_vocab, {"name": "test"}, {"val_loss": 1.23})

    loaded = load_checkpoint(path)
    assert isinstance(loaded, LoadedModel)
    assert loaded.experiment == {"name": "test"}
    assert loaded.metrics == {"val_loss": 1.23}
    assert loaded.tokenizer.vocab_size == tokenizer.vocab_size
    assert len(loaded.artist_vocab) == len(artist_vocab)


def test_checkpoint_round_trip_gives_identical_logits(tmp_path):
    model, tokenizer, artist_vocab = _make_pieces()
    model.eval()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, tokenizer, artist_vocab, {})

    tokens = torch.randint(0, tokenizer.vocab_size, (1, 5))
    artists = torch.zeros(1, dtype=torch.long)
    with torch.inference_mode():
        original_logits, _ = model(tokens, artists)

    loaded = load_checkpoint(path)
    with torch.inference_mode():
        loaded_logits, _ = loaded.model(tokens, artists)

    assert torch.allclose(original_logits, loaded_logits)


def test_checkpoint_defaults_metrics_to_empty_dict(tmp_path):
    model, tokenizer, artist_vocab = _make_pieces()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, tokenizer, artist_vocab, {})
    loaded = load_checkpoint(path)
    assert loaded.metrics == {}


def test_artists_excludes_any_and_sorts_by_display_name(tmp_path):
    model, tokenizer, artist_vocab = _make_pieces()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, tokenizer, artist_vocab, {})
    loaded = load_checkpoint(path)

    pairs = loaded.artists()
    slugs = [slug for slug, _ in pairs]
    assert "<any>" not in slugs
    assert set(slugs) == {"alpha", "beta"}
    display_names = [display for _, display in pairs]
    assert display_names == sorted(display_names)


def test_generate_text_respects_length_in_characters(tmp_path):
    model, tokenizer, artist_vocab = _make_pieces()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, tokenizer, artist_vocab, {})
    loaded = load_checkpoint(path)

    text = loaded.generate_text(artist="alpha", length=25, seed=0)
    assert len(text) == 25


def test_generate_text_is_reproducible_with_seed(tmp_path):
    model, tokenizer, artist_vocab = _make_pieces()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, tokenizer, artist_vocab, {})
    loaded = load_checkpoint(path)

    text1 = loaded.generate_text(artist="alpha", length=30, seed=42)
    text2 = loaded.generate_text(artist="alpha", length=30, seed=42)
    assert text1 == text2


def test_generate_text_handles_any_and_none_artist(tmp_path):
    model, tokenizer, artist_vocab = _make_pieces()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, tokenizer, artist_vocab, {})
    loaded = load_checkpoint(path)

    text_none = loaded.generate_text(artist=None, length=10, seed=1)
    text_any = loaded.generate_text(artist="<any>", length=10, seed=1)
    assert text_none == text_any
    assert len(text_none) == 10


def test_generate_text_raises_on_unknown_artist(tmp_path):
    model, tokenizer, artist_vocab = _make_pieces()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, tokenizer, artist_vocab, {})
    loaded = load_checkpoint(path)

    with pytest.raises(ValueError, match="alpha"):
        loaded.generate_text(artist="not-a-real-artist", length=10)


def test_generate_text_empty_prompt_starts_from_newline(tmp_path):
    model, tokenizer, artist_vocab = _make_pieces()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, tokenizer, artist_vocab, {})
    loaded = load_checkpoint(path)

    # Should not raise, and should return only new text (no leading prompt).
    text = loaded.generate_text(prompt="", artist="beta", length=15, seed=5)
    assert len(text) == 15


def test_generate_text_zero_length_returns_empty_string(tmp_path):
    model, tokenizer, artist_vocab = _make_pieces()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, tokenizer, artist_vocab, {})
    loaded = load_checkpoint(path)

    assert loaded.generate_text(artist="alpha", length=0) == ""


def test_load_checkpoint_works_with_lstm_model(tmp_path):
    torch.manual_seed(0)
    tokenizer = CharTokenizer.train(["hello world\ngoodbye world"])
    artist_vocab = ArtistVocab(["alpha"])
    model = build_model(
        {
            "kind": "lstm",
            "vocab_size": tokenizer.vocab_size,
            "n_artists": len(artist_vocab),
            "embed_dim": 4,
            "artist_embed_dim": 2,
            "hidden_sizes": (4, 4),
            "dropout": 0.0,
        }
    )
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, tokenizer, artist_vocab, {})
    loaded = load_checkpoint(path)
    text = loaded.generate_text(artist="alpha", length=12, seed=0)
    assert len(text) == 12
