from __future__ import annotations

from lyricgen.data.clean import clean_text, split_stanzas


def test_collapses_repeated_spaces_and_trims_lines():
    raw = "  Hello    world  \nSecond   line\n"
    cleaned = clean_text(raw)
    assert cleaned == "Hello world\nSecond line"


def test_normalizes_crlf_line_endings():
    raw = "Line one\r\nLine two\r\n\r\nLine three\r\n"
    cleaned = clean_text(raw)
    assert "\r" not in cleaned
    assert cleaned == "Line one\nLine two\n\nLine three"


def test_unescapes_csv_doubled_quotes():
    raw = 'She said ""hello"" to me'
    cleaned = clean_text(raw)
    assert cleaned == 'She said "hello" to me'


def test_strips_csv_wrapper_quotes_around_a_song():
    raw = '"[Intro]\nFirst line\nSecond line"\n\n"Next song first line\nlast line"'
    cleaned = clean_text(raw)
    stanzas = split_stanzas(cleaned)
    assert stanzas[0].startswith("First line")
    assert not stanzas[0].startswith('"')
    assert stanzas[1].startswith("Next song")
    assert not stanzas[1].endswith('"')


def test_strips_section_tags_by_default():
    raw = "[Verse 1: Someone]\nActual lyric line\n[Chorus]\nAnother line"
    cleaned = clean_text(raw)
    assert "[Verse 1" not in cleaned
    assert "[Chorus]" not in cleaned
    assert "Actual lyric line" in cleaned
    assert "Another line" in cleaned


def test_keeps_section_tags_when_disabled():
    raw = "[Chorus]\nLine one"
    cleaned = clean_text(raw, strip_section_tags=False)
    assert "[Chorus]" in cleaned


def test_strips_multiline_bracketed_notes():
    raw = "Real lyric\n[A long editorial\nnote spanning lines]\nMore lyric"
    cleaned = clean_text(raw)
    assert "editorial" not in cleaned
    assert "Real lyric" in cleaned
    assert "More lyric" in cleaned


def test_strips_leading_lyrics_header():
    raw = "Lyrics\nActual first line\nSecond line"
    cleaned = clean_text(raw)
    assert not cleaned.startswith("Lyrics")
    assert cleaned.startswith("Actual first line")


def test_strips_underscore_italics_markup():
    raw = "This is _emphasized_ text"
    cleaned = clean_text(raw)
    assert "_" not in cleaned
    assert cleaned == "This is emphasized text"


def test_split_stanzas_on_blank_lines():
    text = "Stanza one line one\nStanza one line two\n\nStanza two line one"
    stanzas = split_stanzas(text)
    assert stanzas == ["Stanza one line one\nStanza one line two", "Stanza two line one"]


def test_split_stanzas_falls_back_to_line_grouping_with_no_blank_lines():
    lines = [f"line {i}" for i in range(14)]
    text = "\n".join(lines)
    stanzas = split_stanzas(text)
    assert len(stanzas) == 4
    assert stanzas[0] == "line 0\nline 1\nline 2\nline 3"
    assert stanzas[-1] == "line 12\nline 13"


def test_split_stanzas_breaks_up_oversized_blank_delimited_block():
    lines = [f"line {i}" for i in range(30)]
    text = "\n".join(lines) + "\n\n" + "short stanza"
    stanzas = split_stanzas(text)
    # the 30-line block should be broken into small groups, not kept whole
    assert all(s.count("\n") < 29 for s in stanzas)
    assert stanzas[-1] == "short stanza"


def test_split_stanzas_empty_text():
    assert split_stanzas("") == []


def test_split_stanzas_ignores_blank_only_blocks():
    text = "First stanza\n\n\n\nSecond stanza"
    stanzas = split_stanzas(text)
    assert stanzas == ["First stanza", "Second stanza"]
