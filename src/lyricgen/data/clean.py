"""Cleaning and stanza-splitting for raw scraped lyric files.

The raw files in this project come from several scrapes over the years and
are inconsistent: some wrap whole songs in CSV-style quoting (with `""` as
an escaped quote), some carry Genius-style section tags (`[Chorus]`,
`[Verse 1: Someone]`), some are plain-text poems with hand-indented stanzas
and Project Gutenberg italics markup (`_word_`), and several have no blank
lines at all separating songs. `clean_text` normalizes all of that into
plain text; `split_stanzas` then breaks the cleaned text into paragraph-like
units suitable for training examples.
"""

from __future__ import annotations

import re

_SECTION_TAG_RE = re.compile(r"\[.*?\]", re.DOTALL)
_LEADING_LYRICS_RE = re.compile(r"\A\s*Lyrics\s*\n")
_OPEN_QUOTE_RE = re.compile(r'(\A|\n\n+)"')
_CLOSE_QUOTE_RE = re.compile(r'"(\n\n+|\Z)')
_MULTISPACE_RE = re.compile(r"[ \t]+")
_BLANK_RUN_RE = re.compile(r"\n[ \t]*(?:\n[ \t]*)+")

# Target size (lines) for stanzas synthesized from files with no blank line
# structure at all, chosen to roughly match the size of naturally occurring
# blank-line-delimited stanzas in the better-formatted files.
_FALLBACK_LINES_PER_STANZA = 4

# A blank-line-delimited block longer than this is treated as unstructured
# (a handful of stray blank lines in an otherwise run-together file) and is
# broken up the same way as a file with no blank lines at all.
_MAX_LINES_PER_STANZA = 12


def _unescape_csv_quotes(text: str) -> str:
    return text.replace('""', '"')


def _strip_csv_wrapper_quotes(text: str) -> str:
    text = _OPEN_QUOTE_RE.sub(r"\1", text)
    text = _CLOSE_QUOTE_RE.sub(r"\1", text)
    return text


def clean_text(raw: str, strip_section_tags: bool = True) -> str:
    """Normalize a single raw artist file into plain text.

    Handles CRLF line endings, CSV-style `""`-escaped quoting with stray
    wrapping quotes around whole songs, a stray leading "Lyrics" header
    line, optional `[...]` section/annotation tags, Project Gutenberg
    underscore italics markup, and whitespace (indentation, repeated
    spaces, runs of blank lines).
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = _LEADING_LYRICS_RE.sub("", text)
    text = _unescape_csv_quotes(text)
    text = _strip_csv_wrapper_quotes(text)
    if strip_section_tags:
        text = _SECTION_TAG_RE.sub(" ", text)
    text = text.replace("_", "")

    lines = [_MULTISPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _BLANK_RUN_RE.sub("\n\n", text)
    return text.strip("\n")


def _group_lines(lines: list[str], group_size: int) -> list[str]:
    stanzas = []
    for start in range(0, len(lines), group_size):
        group = [ln for ln in lines[start : start + group_size] if ln]
        if group:
            stanzas.append("\n".join(group))
    return stanzas


def split_stanzas(text: str) -> list[str]:
    """Split cleaned text into stanza-like chunks.

    Files that use blank lines to separate stanzas or songs (poems, and
    several of the better-formatted lyric files) are split on those blank
    lines, trusting the source structure. Files with no blank lines at all
    (common in the messier raw files, where whole songs run together with
    one lyric line per line and no separators) have no usable structure to
    split on. In both cases, any resulting block that is still large (a
    handful of stray blank lines in an otherwise unstructured file, or a
    long block Genius-style with no internal breaks) is broken further into
    small fixed-size groups of consecutive lines, so dedup and training
    examples stay at a useful granularity.
    """
    text = text.strip("\n")
    if not text:
        return []

    blocks = re.split(r"\n\s*\n+", text) if "\n\n" in text else [text]
    stanzas: list[str] = []
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        if len(lines) <= _MAX_LINES_PER_STANZA:
            stanzas.append("\n".join(lines))
        else:
            stanzas.extend(_group_lines(lines, _FALLBACK_LINES_PER_STANZA))
    return stanzas
