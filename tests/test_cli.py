from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from helpers import make_tiny_config

from lyricgen import pretrained
from lyricgen.checkpoint import save_checkpoint
from lyricgen.cli import main
from lyricgen.dataset import ArtistVocab
from lyricgen.models import build_model
from lyricgen.tokenizers import CharTokenizer


@pytest.fixture
def tiny_checkpoint(tmp_path: Path) -> Path:
    # Seed model init so generation tests are not sensitive to global torch
    # RNG state left behind by earlier tests in the same process.
    torch.manual_seed(0)
    tokenizer = CharTokenizer.train(["hello world\ngoodbye world", "goodbye\nworld hello"])
    artist_vocab = ArtistVocab(["alpha", "beta"])
    model = build_model(
        {
            "kind": "transformer",
            "vocab_size": tokenizer.vocab_size,
            "n_artists": len(artist_vocab),
            "n_layer": 1,
            "n_head": 1,
            "d_model": 8,
            "block_size": 8,
            "dropout": 0.0,
        }
    )
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, tokenizer, artist_vocab, {})
    return path


def test_cli_generate_prints_text(tiny_checkpoint, capsys):
    main(
        [
            "generate",
            "--checkpoint",
            str(tiny_checkpoint),
            "--artist",
            "alpha",
            "--length",
            "20",
            "--seed",
            "0",
        ]
    )
    out = capsys.readouterr().out
    # print() adds exactly one trailing newline; strip only that one, since
    # generated text may legitimately contain its own newline characters.
    assert len(out[:-1]) == 20


def test_cli_generate_list_artists(tiny_checkpoint, capsys):
    main(["generate", "--checkpoint", str(tiny_checkpoint), "--list-artists"])
    out = capsys.readouterr().out
    assert "alpha" in out
    assert "beta" in out


def test_cli_generate_multiple_samples(tiny_checkpoint, capsys):
    main(
        [
            "generate",
            "--checkpoint",
            str(tiny_checkpoint),
            "--artist",
            "beta",
            "--length",
            "5",
            "--num-samples",
            "3",
            "--seed",
            "1",
        ]
    )
    out = capsys.readouterr().out
    # 3 samples of 5 characters each, one trailing newline per print() call;
    # do not split on "\n" to count them, since generated text may itself
    # contain newline characters.
    assert len(out) == 3 * (5 + 1)


def test_cli_generate_requires_checkpoint():
    with pytest.raises(SystemExit):
        main(["generate"])


def test_cli_generate_unknown_artist_raises(tiny_checkpoint):
    with pytest.raises(ValueError):
        main(["generate", "--checkpoint", str(tiny_checkpoint), "--artist", "nope"])


def test_cli_train_runs_end_to_end(tiny_data_dir, tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    _write_minimal_config(config_path, tiny_data_dir, tmp_path / "run")

    main(["train", "--config", str(config_path), "--steps", "5"])

    out_dir = tmp_path / "run"
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "best.pt").exists()


def test_cli_train_out_dir_override(tiny_data_dir, tmp_path):
    config_path = tmp_path / "config.yaml"
    original_out = tmp_path / "original_run"
    _write_minimal_config(config_path, tiny_data_dir, original_out)

    override_out = tmp_path / "override_run"
    main(
        [
            "train",
            "--config",
            str(config_path),
            "--steps",
            "5",
            "--out-dir",
            str(override_out),
        ]
    )

    assert not original_out.exists()
    assert (override_out / "metrics.json").exists()


def test_cli_prepare_wraps_prepare_module(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "solo.txt").write_text(
        "This stanza has plenty of characters to survive filtering easily",
        encoding="utf-8",
    )

    from lyricgen.data import artists as artists_module
    from lyricgen.data.artists import ArtistInfo

    original = dict(artists_module.ARTISTS)
    artists_module.ARTISTS.clear()
    artists_module.ARTISTS["solo"] = ArtistInfo("Solo", ("solo.txt",))
    try:
        out_dir = tmp_path / "processed"
        main(
            [
                "prepare",
                "--raw-dir",
                str(raw_dir),
                "--out-dir",
                str(out_dir),
                "--val-frac",
                "0.0",
                "--test-frac",
                "0.0",
                "--min-stanza-chars",
                "5",
            ]
        )
        assert (out_dir / "stats.json").exists()
        assert (out_dir / "train.jsonl").exists()
    finally:
        artists_module.ARTISTS.clear()
        artists_module.ARTISTS.update(original)


def test_cli_evaluate_exits_clearly_when_module_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "lyricgen.evaluate", None)
    with pytest.raises(SystemExit, match="could not be loaded"):
        main(["evaluate", "--checkpoint", "x", "--data", "y", "--out", "z"])


def test_cli_demo_exits_clearly_when_gradio_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "lyricgen.app", None)
    with pytest.raises(SystemExit, match="requirements-demo.txt"):
        main(["demo"])


def _write_minimal_config(path: Path, data_dir: Path, out_dir: Path) -> None:
    config = make_tiny_config(data_dir, out_dir)
    import yaml

    path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")


@pytest.fixture
def fake_release(tmp_path, monkeypatch):
    """Point pretrained.RELEASE_URL/cache at local dirs; no network used."""
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(pretrained, "RELEASE_URL", release_dir.as_uri() + "/")
    monkeypatch.setenv("LYRICGEN_CACHE", str(cache_dir))
    return release_dir, cache_dir


def test_cli_download_single_name(fake_release):
    release_dir, cache_dir = fake_release
    (release_dir / "lstm.pt").write_bytes(b"fake lstm checkpoint")

    main(["download", "lstm"])

    assert (cache_dir / "lstm.pt").read_bytes() == b"fake lstm checkpoint"


def test_cli_download_all(fake_release):
    release_dir, cache_dir = fake_release
    for name, entry in pretrained.MANIFEST.items():
        (release_dir / entry["file"]).write_bytes(name.encode())

    main(["download", "--all"])

    for name, entry in pretrained.MANIFEST.items():
        assert (cache_dir / entry["file"]).read_bytes() == name.encode()


def test_cli_download_unknown_name_errors():
    with pytest.raises(SystemExit, match="unknown checkpoint"):
        main(["download", "not-a-real-name"])


def test_cli_download_no_names_errors():
    with pytest.raises(SystemExit, match="specify"):
        main(["download"])


def test_cli_generate_with_pretrained_downloads_then_generates(
    fake_release, torch_seeded_checkpoint_bytes
):
    release_dir, cache_dir = fake_release
    (release_dir / "lstm.pt").write_bytes(torch_seeded_checkpoint_bytes)

    main(["generate", "--pretrained", "lstm", "--artist", "alpha", "--length", "5"])

    assert (cache_dir / "lstm.pt").exists()


@pytest.fixture
def torch_seeded_checkpoint_bytes(tmp_path) -> bytes:
    """Bytes of a real, tiny, loadable checkpoint (an lstm, to match the name)."""
    torch.manual_seed(0)
    tokenizer = CharTokenizer.train(["hello world\ngoodbye world"])
    artist_vocab = ArtistVocab(["alpha", "beta"])
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
    path = tmp_path / "src_ckpt.pt"
    save_checkpoint(path, model, tokenizer, artist_vocab, {})
    return path.read_bytes()
