# Data

The lyric files are not in the working tree. They are third-party text, about 150 MB of it, and they made
the repository slow to clone, so I removed them from the tree in commit 62de3ca. They are still reachable in
history at commit 0fe1405, and `scripts/fetch_data.py` restores them into an ignored `data/raw/` folder:

```bash
python scripts/fetch_data.py          # writes data/raw/*.txt from git history
lyricgen prepare                      # writes data/processed/{train,val,test}.jsonl + stats.json
```

The two aggregate files in the original folder (`All-songs.txt`, 78 MB, and `Best-songs-1900s.txt`, 70 MB)
are skipped: they mix every artist together, so they cannot be used for artist-conditioned training.

## Cleaning

The raw files come from different sources and are inconsistent, so `lyricgen.data.clean` handles all of it:

- CSV-style exports where each song is wrapped in double quotes with `""` escapes, plus a stray `Lyrics` header
- Genius-style section tags such as `[Verse 1: Eminem]`, including multi-line editorial notes
- Project Gutenberg formatting in the two public-domain files (`_italics_`, hand indentation, roman numerals)
- CRLF line endings and trailing double spaces used as soft line breaks

Text is then split into stanzas on blank lines. Files with no blank lines at all, and blocks longer than 12
lines, fall back to fixed four-line groups so one file does not become a single enormous record.

## Duplicates and splits

Four artists appear under more than one filename (`eminem` + `eminem2`, three Kanye West files, two Lil Wayne
files, two Notorious B.I.G. files), so those sources are merged into one artist. Identical stanzas are then
removed within each artist, and stanzas that appear under more than one artist are kept only for the first
artist alphabetically. Each artist's remaining stanzas are shuffled with a fixed seed and split 80 / 10 / 10,
so every artist is represented in train, validation and test.

Measured on the real files (`data/processed/stats.json`, prepare takes about 1.3 s):

| quantity | value |
| --- | --- |
| artists | 47 |
| stanzas before dedupe | 46,739 |
| duplicates removed within an artist | 5,027 |
| duplicates removed across artists | 19 |
| characters: train / validation / test | 5,266,125 / 657,548 / 657,961 |
| records: train / validation / test | 33,355 / 4,169 / 4,169 |

## Licensing note

The lyrics are copyrighted and are not redistributed here beyond what already exists in this repository's
history. Nothing new was added, the corpus is only used to train small character and subword models, and
`docs/experiments.md` reports how much of the training text the models reproduce verbatim.
