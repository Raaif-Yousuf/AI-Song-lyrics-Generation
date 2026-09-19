from __future__ import annotations

import json
from pathlib import Path

import pytest

from lyricgen.dataset import ArtistVocab
from lyricgen.evaluate import evaluate
from lyricgen.models import build_model
from lyricgen.sampling import generate
from lyricgen.tokenizers import CharTokenizer

ARTIST_STANZAS = {
    "aa": [
        "the quick brown fox jumps over the lazy dog today\n",
        "a second stanza about the fox and the dog running fast\n",
        "yet another stanza with different silly words entirely\n",
    ],
    "bb": [
        "roses are red violets are blue sugar is sweet\n",
        "another rhyme about the sky above and the sea below\n",
        "one more little verse for good measure here\n",
    ],
}

COPY_STANZA = "the quick brown fox jumps over the lazy dog today"


def _write_split(path: Path, artist_to_stanzas: dict[str, list[str]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for artist, stanzas in artist_to_stanzas.items():
            for stanza in stanzas:
                fh.write(json.dumps({"artist": artist, "text": stanza}) + "\n")


def _build_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_split(data_dir / "train.jsonl", ARTIST_STANZAS)
    _write_split(data_dir / "val.jsonl", ARTIST_STANZAS)
    _write_split(data_dir / "test.jsonl", ARTIST_STANZAS)
    return data_dir


class _StubLoadedModel:
    """Minimal stand-in for `lyricgen.checkpoint.LoadedModel` in tests."""

    def __init__(self, model, tokenizer, artist_vocab: ArtistVocab, experiment: dict) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.artist_vocab = artist_vocab
        self.experiment = experiment
        self.metrics: dict = {}

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
        if artist is None or artist == "any":
            artist_id = 0
        else:
            valid = [slug for slug, _ in self.artists()]
            if artist not in valid:
                raise ValueError(f"unknown artist {artist!r}, expected one of {valid}")
            artist_id = self.artist_vocab.id(artist)

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


class _CopyingLoadedModel(_StubLoadedModel):
    """Stand-in whose `generate_text` always returns fixed training text."""

    def generate_text(self, prompt: str = "", artist: str | None = None, length: int = 400, **kw):
        return COPY_STANZA[:length]


def _build_loaded(tmp_path: Path, cls=_StubLoadedModel):
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
    experiment = {"train": {"block_size": 8, "batch_size": 2}}
    loaded = cls(model, tokenizer, artist_vocab, experiment)
    return loaded, data_dir


def test_evaluate_returns_expected_fields_and_ranges(tmp_path: Path) -> None:
    loaded, data_dir = _build_loaded(tmp_path)

    result = evaluate(
        loaded,
        data_dir=data_dir,
        split="test",
        n_samples=4,
        sample_chars=30,
        seed=7,
        novelty_ns=(2, 3),
    )

    for key in (
        "split",
        "test_loss",
        "test_bpc",
        "test_char_ppl",
        "gen_chars_per_sec",
        "novelty",
        "novelty_heldout",
        "distinct_2",
        "distinct_3",
        "repeated_line_rate",
        "longest_copied_run",
        "conditioning",
        "sampling",
        "n_samples",
        "seed",
        "samples",
    ):
        assert key in result, key

    assert result["split"] == "test"
    assert result["n_samples"] == 4
    assert result["seed"] == 7
    assert result["test_bpc"] >= 0.0
    assert result["test_char_ppl"] >= 1.0
    assert result["gen_chars_per_sec"] > 0.0

    assert set(result["novelty"]) == {2, 3}
    for value in result["novelty"].values():
        assert 0.0 <= value <= 1.0
    for value in result["novelty_heldout"].values():
        assert 0.0 <= value <= 1.0

    lcr = result["longest_copied_run"]
    for group in ("samples", "heldout"):
        assert lcr[group]["mean"] >= 0.0
        assert lcr[group]["max"] >= lcr[group]["mean"]

    for metric in (result["distinct_2"], result["distinct_3"], result["repeated_line_rate"]):
        assert set(metric) == {"samples", "heldout"}
        for value in metric.values():
            assert 0.0 <= value <= 1.0

    conditioning = result["conditioning"]
    assert set(conditioning) == {"true_bpc", "any_bpc", "wrong_bpc", "per_artist"}
    assert conditioning["true_bpc"] == pytest.approx(result["test_bpc"])
    assert set(conditioning["per_artist"]) == {"aa", "bb"}
    for gap in conditioning["per_artist"].values():
        assert set(gap) == {"true", "any"}

    assert result["sampling"] == {
        "temperature": 0.8,
        "top_k": 0,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
    }
    # 6 short samples, or fewer if n_samples < 6.
    assert len(result["samples"]) == min(6, 4)
    for sample in result["samples"]:
        assert set(sample) == {"artist", "prompt", "text"}
        assert len(sample["text"]) <= 200


def test_evaluate_is_deterministic_given_seed(tmp_path: Path) -> None:
    loaded, data_dir = _build_loaded(tmp_path)
    kwargs = dict(data_dir=data_dir, split="test", n_samples=4, sample_chars=25, seed=123)

    result1 = evaluate(loaded, **kwargs)
    result2 = evaluate(loaded, **kwargs)

    assert result1["samples"] == result2["samples"]
    assert result1["novelty"] == result2["novelty"]
    assert result1["novelty_heldout"] == result2["novelty_heldout"]
    assert result1["longest_copied_run"] == result2["longest_copied_run"]
    assert result1["test_bpc"] == pytest.approx(result2["test_bpc"])
    assert result1["conditioning"] == result2["conditioning"]


def test_evaluate_copied_text_scores_low_novelty_and_high_copied_run(tmp_path: Path) -> None:
    loaded, data_dir = _build_loaded(tmp_path, cls=_CopyingLoadedModel)
    length = len(COPY_STANZA)

    result = evaluate(
        loaded,
        data_dir=data_dir,
        split="test",
        n_samples=3,
        sample_chars=length,
        seed=1,
        novelty_ns=(2, 3),
    )

    for value in result["novelty"].values():
        assert value == pytest.approx(0.0)

    word_count = len(COPY_STANZA.split())
    assert result["longest_copied_run"]["samples"]["mean"] == pytest.approx(word_count)
    assert result["longest_copied_run"]["samples"]["max"] == word_count


def test_evaluate_unknown_split_raises(tmp_path: Path) -> None:
    loaded, data_dir = _build_loaded(tmp_path)
    with pytest.raises(FileNotFoundError):
        evaluate(loaded, data_dir=data_dir, split="does-not-exist")


def test_evaluate_skip_conditioning_sets_none(tmp_path: Path) -> None:
    loaded, data_dir = _build_loaded(tmp_path)

    result = evaluate(
        loaded,
        data_dir=data_dir,
        split="test",
        n_samples=2,
        sample_chars=20,
        seed=1,
        skip_conditioning=True,
    )

    assert result["conditioning"] is None


def test_evaluate_repeated_line_stub_scores_high_repeated_line_rate(tmp_path: Path) -> None:
    line = "same line every time"

    class _RepeatingLoadedModel(_StubLoadedModel):
        def generate_text(self, prompt="", artist=None, length=400, **kw):
            text = (line + "\n") * 10
            return text[:length]

    loaded, data_dir = _build_loaded(tmp_path, cls=_RepeatingLoadedModel)
    # 5 clean repeats of "line\n" (21 chars each): 4 of them repeat the first.
    sample_chars = (len(line) + 1) * 5

    result = evaluate(
        loaded, data_dir=data_dir, split="test", n_samples=2, sample_chars=sample_chars, seed=1
    )

    assert result["repeated_line_rate"]["samples"] == pytest.approx(4 / 5)
    assert result["distinct_2"]["samples"] < 0.5


def test_seeded_derangement_has_no_fixed_points_and_is_deterministic() -> None:
    from lyricgen.evaluate import _seeded_derangement

    items = [0, 1, 2, 3, 4]
    mapping1 = _seeded_derangement(items, seed=42)
    mapping2 = _seeded_derangement(items, seed=42)

    assert mapping1 == mapping2
    assert set(mapping1.values()) == set(items)
    for item in items:
        assert mapping1[item] != item


def test_seeded_derangement_single_item_maps_to_itself() -> None:
    from lyricgen.evaluate import _seeded_derangement

    assert _seeded_derangement([7], seed=0) == {7: 7}
    assert _seeded_derangement([], seed=0) == {}
