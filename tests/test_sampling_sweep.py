from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from lyricgen.dataset import ArtistVocab
from lyricgen.models import build_model
from lyricgen.sampling import generate
from lyricgen.tokenizers import CharTokenizer

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "sampling_sweep.py"
_SPEC = importlib.util.spec_from_file_location("sampling_sweep", SCRIPT_PATH)
sampling_sweep = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("sampling_sweep", sampling_sweep)
_SPEC.loader.exec_module(sampling_sweep)

ARTIST_STANZAS = {
    "aa": [
        "the quick brown fox jumps over the lazy dog today\n",
        "a second stanza about the fox and the dog running fast\n",
    ],
    "bb": [
        "roses are red violets are blue sugar is sweet\n",
        "another rhyme about the sky above and the sea below\n",
    ],
}


def _write_split(path: Path, artist_to_stanzas: dict[str, list[str]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for artist, stanzas in artist_to_stanzas.items():
            for stanza in stanzas:
                fh.write(json.dumps({"artist": artist, "text": stanza}) + "\n")


def _build_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_split(data_dir / "train.jsonl", ARTIST_STANZAS)
    _write_split(data_dir / "test.jsonl", ARTIST_STANZAS)
    return data_dir


class _StubLoadedModel:
    """Minimal stand-in for `lyricgen.checkpoint.LoadedModel` in tests."""

    def __init__(self, model, tokenizer, artist_vocab: ArtistVocab) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.artist_vocab = artist_vocab

    def artists(self) -> list[tuple[str, str]]:
        return sorted(((slug, slug.upper()) for slug in ARTIST_STANZAS), key=lambda t: t[1])

    def generate_text(
        self,
        prompt: str = "",
        artist: str | None = None,
        length: int = 400,
        temperature: float = 0.8,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        seed: int | None = None,
    ) -> str:
        artist_id = 0 if artist in (None, "any") else self.artist_vocab.id(artist)
        context = list(self.tokenizer.encode(prompt if prompt else "\n"))
        collected: list[int] = []
        decoded = ""
        step_seed = seed
        for _ in range(20):
            if len(decoded) >= length:
                break
            new_ids = generate(
                self.model,
                context + collected,
                artist_id,
                max_new_tokens=max(length - len(decoded), 1),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=step_seed,
            )
            collected.extend(new_ids)
            decoded = self.tokenizer.decode(collected)
            if step_seed is not None:
                step_seed += 1
        return decoded[:length]


def _build_loaded(tmp_path: Path) -> tuple[_StubLoadedModel, Path]:
    data_dir = _build_data_dir(tmp_path)
    all_records = [
        {"artist": artist, "text": stanza}
        for artist, stanzas in ARTIST_STANZAS.items()
        for stanza in stanzas
    ]
    all_texts = [r["text"] for r in all_records]
    tokenizer = CharTokenizer.train(all_texts)
    artist_vocab = ArtistVocab.from_records(all_records)

    model = build_model(
        dict(
            kind="transformer",
            vocab_size=tokenizer.vocab_size,
            n_artists=len(artist_vocab),
            n_layer=1,
            n_head=1,
            d_model=8,
            block_size=16,
            dropout=0.0,
        )
    )
    loaded = _StubLoadedModel(model, tokenizer, artist_vocab)
    return loaded, data_dir


def test_run_sweep_covers_every_grid_setting_with_valid_ranges(tmp_path: Path) -> None:
    loaded, data_dir = _build_loaded(tmp_path)

    sweep = sampling_sweep.run_sweep(
        loaded,
        data_dir=data_dir,
        split="test",
        n_samples=3,
        sample_chars=20,
        seed=1,
        checkpoint_path="runs/tiny/best.pt",
    )

    assert sweep["checkpoint"] == "runs/tiny/best.pt"
    assert sweep["n_samples"] == 3
    assert len(sweep["results"]) == len(sampling_sweep.DEFAULT_GRID)

    for entry, setting in zip(sweep["results"], sampling_sweep.DEFAULT_GRID, strict=True):
        assert entry["setting"] == setting
        assert 0.0 <= entry["novelty_6"] <= 1.0
        assert 0.0 <= entry["distinct_2"] <= 1.0
        assert 0.0 <= entry["repeated_line_rate"] <= 1.0
        assert entry["longest_copied_run_mean"] >= 0.0
        assert entry["gen_chars_per_sec"] > 0.0


def test_run_sweep_is_deterministic_given_seed(tmp_path: Path) -> None:
    loaded, data_dir = _build_loaded(tmp_path)
    kwargs = dict(data_dir=data_dir, split="test", n_samples=2, sample_chars=15, seed=7)

    sweep1 = sampling_sweep.run_sweep(loaded, **kwargs)
    sweep2 = sampling_sweep.run_sweep(loaded, **kwargs)

    for entry1, entry2 in zip(sweep1["results"], sweep2["results"], strict=True):
        assert entry1["novelty_6"] == pytest.approx(entry2["novelty_6"])
        assert entry1["distinct_2"] == pytest.approx(entry2["distinct_2"])
        assert entry1["repeated_line_rate"] == pytest.approx(entry2["repeated_line_rate"])
        assert entry1["longest_copied_run_mean"] == pytest.approx(entry2["longest_copied_run_mean"])


def test_run_sweep_custom_grid_is_respected(tmp_path: Path) -> None:
    loaded, data_dir = _build_loaded(tmp_path)
    grid = ({"temperature": 0.0, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0},)

    sweep = sampling_sweep.run_sweep(
        loaded, data_dir=data_dir, split="test", n_samples=2, sample_chars=10, seed=1, grid=grid
    )

    assert len(sweep["results"]) == 1
    assert sweep["results"][0]["setting"] == grid[0]


def test_render_markdown_has_header_and_one_row_per_setting(tmp_path: Path) -> None:
    loaded, data_dir = _build_loaded(tmp_path)
    grid = (
        {"temperature": 0.5, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0},
        {"temperature": 0.8, "top_k": 20, "top_p": 1.0, "repetition_penalty": 1.0},
    )

    sweep = sampling_sweep.run_sweep(
        loaded, data_dir=data_dir, split="test", n_samples=2, sample_chars=10, seed=1, grid=grid
    )
    table = sampling_sweep.render_markdown(sweep)
    lines = table.splitlines()

    assert lines[0].startswith("| temperature")
    assert len(lines) == 2 + len(grid)
    assert "0.500" in lines[2]
    assert "0.800" in lines[3]


def test_main_writes_json_and_markdown(tmp_path: Path, monkeypatch, capsys) -> None:
    loaded, data_dir = _build_loaded(tmp_path)
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_text("not a real checkpoint", encoding="utf-8")

    def _fake_load_checkpoint(path, device="cpu"):
        assert str(path) == str(checkpoint_path)
        return loaded

    import lyricgen.checkpoint as checkpoint_module

    monkeypatch.setattr(checkpoint_module, "load_checkpoint", _fake_load_checkpoint)

    out_json = tmp_path / "sweep.json"
    out_md = tmp_path / "sweep.md"
    grid_size = len(sampling_sweep.DEFAULT_GRID)

    sampling_sweep.main(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--data",
            str(data_dir),
            "--n-samples",
            "2",
            "--sample-chars",
            "10",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(payload["results"]) == grid_size

    md_text = out_md.read_text(encoding="utf-8")
    printed = capsys.readouterr().out
    assert md_text.strip() == printed.strip()
