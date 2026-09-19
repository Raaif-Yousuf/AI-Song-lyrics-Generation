from __future__ import annotations

import math

import pytest

from lyricgen.metrics import (
    bits_per_char,
    char_perplexity,
    longest_copied_run,
    ngram_novelty,
    ngram_novelty_by_n,
    ngram_set,
    word_tokens,
)


def test_bits_per_char_known_value() -> None:
    # 1 nat over 1 char == 1/ln(2) bits.
    assert bits_per_char(1.0, 1) == pytest.approx(1 / math.log(2))


def test_bits_per_char_zero_nll() -> None:
    assert bits_per_char(0.0, 100) == 0.0


def test_bits_per_char_rejects_non_positive_chars() -> None:
    with pytest.raises(ValueError):
        bits_per_char(1.0, 0)


def test_char_perplexity_known_value() -> None:
    assert char_perplexity(math.log(2) * 10, 10) == pytest.approx(2.0)


def test_char_perplexity_rejects_non_positive_chars() -> None:
    with pytest.raises(ValueError):
        char_perplexity(1.0, -1)


def test_word_tokens_lowercases_and_strips_punctuation() -> None:
    assert word_tokens("Hello, World! It's Me.") == ["hello", "world", "it's", "me"]


def test_word_tokens_empty_string() -> None:
    assert word_tokens("") == []


def test_ngram_set_rejects_non_positive_n() -> None:
    with pytest.raises(ValueError):
        ngram_set(["a b c"], 0)


def test_ngram_set_counts_unique_ngrams() -> None:
    hashes = ngram_set(["the cat sat"], 2)
    # bigrams: (the, cat), (cat, sat)
    assert len(hashes) == 2


def test_ngram_set_does_not_cross_text_boundaries() -> None:
    together = ngram_set(["the cat sat"], 2)
    split = ngram_set(["the cat", "sat"], 2)
    # "cat sat" bigram only exists when not split across two texts.
    assert len(split) == 1
    assert split != together


def test_ngram_novelty_all_seen() -> None:
    reference = ngram_set(["the cat sat on the mat"], 2)
    assert ngram_novelty(["the cat sat"], reference, 2) == 0.0


def test_ngram_novelty_all_novel() -> None:
    reference = ngram_set(["completely unrelated text here"], 2)
    assert ngram_novelty(["the cat sat"], reference, 2) == 1.0


def test_ngram_novelty_partial() -> None:
    reference = ngram_set(["the cat sat"], 2)
    # generated bigrams: (the, cat) seen, (cat, ran) novel -> 1/2 novel.
    assert ngram_novelty(["the cat ran"], reference, 2) == pytest.approx(0.5)


def test_ngram_novelty_empty_generated_text() -> None:
    reference = ngram_set(["the cat sat"], 2)
    assert ngram_novelty([""], reference, 2) == 0.0


def test_ngram_novelty_by_n_multiple_lengths() -> None:
    reference_texts = ["the cat sat on the mat"]
    generated = ["the cat sat on a rug"]
    result = ngram_novelty_by_n(generated, reference_texts, [1, 2])
    assert set(result) == {1, 2}
    assert 0.0 <= result[1] <= 1.0
    assert 0.0 <= result[2] <= 1.0


def test_longest_copied_run_exact_substring() -> None:
    reference = ["the quick brown fox jumps over the lazy dog"]
    generated = "a quick brown fox jumps over a wall"
    # "quick brown fox jumps over" (5 words) is copied verbatim.
    assert longest_copied_run(generated, reference) == 5


def test_longest_copied_run_no_overlap() -> None:
    reference = ["completely different words entirely"]
    generated = "not a single match here"
    assert longest_copied_run(generated, reference) == 0


def test_longest_copied_run_full_copy() -> None:
    text = "these are the same six words"
    assert longest_copied_run(text, [text]) == 6


def test_longest_copied_run_respects_max_n() -> None:
    text = "one two three four five six seven eight"
    assert longest_copied_run(text, [text], max_n=3) == 3


def test_longest_copied_run_empty_generated() -> None:
    assert longest_copied_run("", ["some reference text"]) == 0


def test_longest_copied_run_empty_reference() -> None:
    assert longest_copied_run("some generated text", []) == 0
    assert longest_copied_run("some generated text", [""]) == 0


def test_longest_copied_run_rejects_non_positive_max_n() -> None:
    with pytest.raises(ValueError):
        longest_copied_run("a b c", ["a b c"], max_n=0)


def test_longest_copied_run_match_at_end() -> None:
    reference = ["completely unrelated preamble then the exact ending phrase"]
    generated = "some other words then the exact ending phrase"
    assert longest_copied_run(generated, reference) == 5
