"""Evaluation metrics: bits-per-char, perplexity, n-gram novelty, copied runs."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence

_WORD_RE = re.compile(r"[a-z0-9']+")

# 61-bit Mersenne prime modulus and an arbitrary odd base for the polynomial
# rolling hash used by `longest_copied_run`. Using a large prime modulus keeps
# hash collisions astronomically unlikely for the text sizes this module
# targets (evaluation batches, not corpora with billions of n-grams).
_HASH_MOD = (1 << 61) - 1
_HASH_BASE = 1_000_003


def word_tokens(text: str) -> list[str]:
    """Lowercase word tokens (alphanumerics and apostrophes) extracted from text."""
    return _WORD_RE.findall(text.lower())


def bits_per_char(total_nll_nats: float, n_chars: int) -> float:
    """Bits-per-character from total negative log-likelihood in nats.

    `total_nll_nats` is the sum of per-target negative log-likelihoods (natural
    log) over `n_chars` characters of held-out text.
    """
    if n_chars <= 0:
        raise ValueError("n_chars must be positive")
    return total_nll_nats / n_chars / math.log(2)


def char_perplexity(total_nll_nats: float, n_chars: int) -> float:
    """Character-level perplexity: exp(nats-per-character)."""
    if n_chars <= 0:
        raise ValueError("n_chars must be positive")
    return math.exp(total_nll_nats / n_chars)


def ngram_set(texts: Iterable[str], n: int) -> set[int]:
    """Hashes of all word n-grams across `texts`.

    Memory-conscious by design: only the integer hash of each n-gram tuple is
    kept, not the tuple itself, so this stays cheap even over large corpora
    where the strings would dominate memory. N-grams do not cross the
    boundary between two entries of `texts` (each string is tokenized and
    windowed independently), so `texts` may be an arbitrary iterable of
    documents or stanzas.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    hashes: set[int] = set()
    for text in texts:
        words = word_tokens(text)
        for i in range(len(words) - n + 1):
            hashes.add(hash(tuple(words[i : i + n])))
    return hashes


def ngram_novelty(texts: Iterable[str], reference: set[int], n: int) -> float:
    """Fraction of `texts`' unique word n-grams absent from `reference`.

    `reference` should be a hash set produced by `ngram_set(reference_texts, n)`
    with the same `n`. Returns 0.0 when `texts` has no n-grams of length `n`.
    """
    generated = ngram_set(texts, n)
    if not generated:
        return 0.0
    novel = sum(1 for h in generated if h not in reference)
    return novel / len(generated)


def ngram_novelty_by_n(
    texts: Iterable[str], reference_texts: Iterable[str], ns: Iterable[int]
) -> dict[int, float]:
    """Novelty scores for several n-gram lengths at once.

    `texts` and `reference_texts` are materialized once so they can be
    re-tokenized for each `n` in `ns`.
    """
    texts = list(texts)
    reference_texts = list(reference_texts)
    return {n: ngram_novelty(texts, ngram_set(reference_texts, n), n) for n in ns}


def distinct_n(texts: Iterable[str], n: int) -> float:
    """Self-diversity: unique word n-grams divided by total word n-grams.

    Unlike `ngram_novelty`, this has no external reference: it measures how
    repetitive `texts` are internally (low distinct-n means the same
    n-grams recur often, e.g. a model stuck in a loop). Returns 0.0 when
    `texts` has no n-grams of length `n`.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    total = 0
    seen: set[int] = set()
    for text in texts:
        words = word_tokens(text)
        for i in range(len(words) - n + 1):
            total += 1
            seen.add(hash(tuple(words[i : i + n])))
    return len(seen) / total if total else 0.0


def repeated_line_rate(texts: Iterable[str]) -> float:
    """Fraction of non-empty lines that exactly repeat an earlier line in
    the same sample.

    Each string in `texts` is checked independently (a line repeating a
    line from a *different* sample does not count); lines are compared
    after stripping surrounding whitespace. Returns 0.0 when there are no
    non-empty lines at all.
    """
    total_lines = 0
    repeated = 0
    for text in texts:
        seen: set[str] = set()
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            total_lines += 1
            if stripped in seen:
                repeated += 1
            else:
                seen.add(stripped)
    return repeated / total_lines if total_lines else 0.0


def _word_hash_prefix(words: Sequence[str]) -> list[int]:
    """Prefix polynomial hash of `words`; `prefix[i]` hashes `words[:i]`."""
    prefix = [0] * (len(words) + 1)
    for i, word in enumerate(words):
        # `hash(str)` is randomized per process by PYTHONHASHSEED, which is
        # fine here: the prefix array and all lookups derived from it are
        # only ever compared within a single call, never persisted or
        # compared across runs.
        word_value = hash(word) & 0xFFFFFFFF
        prefix[i + 1] = (prefix[i] * _HASH_BASE + word_value) % _HASH_MOD
    return prefix


def _powers(count: int) -> list[int]:
    powers = [1] * (count + 1)
    for i in range(1, count + 1):
        powers[i] = (powers[i - 1] * _HASH_BASE) % _HASH_MOD
    return powers


def _window_hash(prefix: list[int], powers: list[int], start: int, length: int) -> int:
    return (prefix[start + length] - prefix[start] * powers[length]) % _HASH_MOD


def _build_reference_index(reference_words: Sequence[str], max_n: int) -> list[set[int]]:
    """Per-length hash sets of every reference n-gram for n in `1..max_n`.

    Complexity: computing the shared prefix-hash array is O(R). Because the
    prefix array supports an O(1) hash lookup for any (start, length) window,
    populating the `max_n` per-length sets costs O(R) work per length, i.e.
    O(R * max_n) time and space overall (R = len(reference_words)). Space is
    `max_n` sets of up to R integers each -- hashes only, no substrings.
    """
    prefix = _word_hash_prefix(reference_words)
    powers = _powers(max_n)
    r = len(reference_words)
    index: list[set[int]] = [set() for _ in range(max_n + 1)]
    for length in range(1, max_n + 1):
        bucket = index[length]
        for start in range(r - length + 1):
            bucket.add(_window_hash(prefix, powers, start, length))
    return index


def longest_copied_run(generated_text: str, reference_texts: Iterable[str], max_n: int = 32) -> int:
    """Length (in words) of the longest run of `generated_text` copied verbatim
    from `reference_texts`.

    Builds a rolling-hash index of the reference for n-gram lengths `1..max_n`
    (see `_build_reference_index`), then for every start position in the
    generated text binary-searches the longest length whose n-gram hash is
    present in the reference index at that length. The search is valid because
    a verbatim match of length L at a given start implies a verbatim match of
    length L-1 at the same start (it is the same words, one shorter): match
    length is monotonically non-increasing as L grows, so binary search finds
    the exact boundary.

    Complexity: O(R * max_n) time and space to build the reference index
    (R = word count of the reference), then O(G * log(max_n)) hash lookups
    (G = word count of `generated_text`), each O(1) after O(G) preprocessing
    of the generated text's own prefix hash array. This is far cheaper than
    the O(R * G) naive comparison for the corpus sizes this module targets.
    The result is approximate in principle (hash collisions could report a
    match that is not truly verbatim) but the collision probability with a
    61-bit modulus is negligible at these text sizes.
    """
    if max_n <= 0:
        raise ValueError("max_n must be a positive integer")

    generated_words = word_tokens(generated_text)
    reference_words: list[str] = []
    for text in reference_texts:
        reference_words.extend(word_tokens(text))

    if not generated_words or not reference_words:
        return 0

    effective_max_n = min(max_n, len(reference_words), len(generated_words))
    ref_index = _build_reference_index(reference_words, effective_max_n)

    gen_prefix = _word_hash_prefix(generated_words)
    gen_powers = _powers(effective_max_n)

    best = 0
    g = len(generated_words)
    for start in range(g):
        limit = min(effective_max_n, g - start)
        if limit <= best:
            # This start position cannot possibly beat the current best.
            continue
        lo, hi = 0, limit
        while lo < hi:
            mid = (lo + hi + 1) // 2
            candidate = _window_hash(gen_prefix, gen_powers, start, mid)
            if candidate in ref_index[mid]:
                lo = mid
            else:
                hi = mid - 1
        best = max(best, lo)
        if best == effective_max_n:
            break
    return best
