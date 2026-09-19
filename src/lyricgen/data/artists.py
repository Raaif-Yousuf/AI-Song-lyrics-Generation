"""Registry mapping artist slugs to display names and raw source files.

The raw `Artists/` files in project history came from more than one scrape,
so a few artists have more than one source file (for example Eminem's
`eminem.txt` and `eminem2.txt`). This module merges those into a single
canonical slug per artist. The two mixed-artist aggregate files
(`All-songs.txt`, `Best-songs-1900s.txt`) are intentionally excluded.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArtistInfo:
    """Canonical metadata for one artist."""

    display_name: str
    filenames: tuple[str, ...]


ARTISTS: dict[str, ArtistInfo] = {
    "adele": ArtistInfo("Adele", ("adele.txt",)),
    "al-green": ArtistInfo("Al Green", ("al-green.txt",)),
    "alicia-keys": ArtistInfo("Alicia Keys", ("alicia-keys.txt",)),
    "amy-winehouse": ArtistInfo("Amy Winehouse", ("amy-winehouse.txt",)),
    "beatles": ArtistInfo("The Beatles", ("beatles.txt",)),
    "bieber": ArtistInfo("Justin Bieber", ("bieber.txt",)),
    "bjork": ArtistInfo("Bjork", ("bjork.txt",)),
    "blink-182": ArtistInfo("Blink-182", ("blink-182.txt",)),
    "bob-dylan": ArtistInfo("Bob Dylan", ("bob-dylan.txt",)),
    "bob-marley": ArtistInfo("Bob Marley", ("bob-marley.txt",)),
    "britney-spears": ArtistInfo("Britney Spears", ("britney-spears.txt",)),
    "bruce-springsteen": ArtistInfo("Bruce Springsteen", ("bruce-springsteen.txt",)),
    "bruno-mars": ArtistInfo("Bruno Mars", ("bruno-mars.txt",)),
    "cake": ArtistInfo("Cake", ("cake.txt",)),
    "dickinson": ArtistInfo("Emily Dickinson", ("dickinson.txt",)),
    "disney": ArtistInfo("Disney", ("disney.txt",)),
    "dj-khaled": ArtistInfo("DJ Khaled", ("dj-khaled.txt",)),
    "dolly-parton": ArtistInfo("Dolly Parton", ("dolly-parton.txt",)),
    "dr-seuss": ArtistInfo("Dr. Seuss", ("dr-seuss.txt",)),
    "drake": ArtistInfo("Drake", ("drake.txt",)),
    "eminem": ArtistInfo("Eminem", ("eminem.txt", "eminem2.txt")),
    "hamilton": ArtistInfo("Hamilton", ("Hamilton.txt",)),
    "janis-joplin": ArtistInfo("Janis Joplin", ("janisjoplin.txt",)),
    "jimi-hendrix": ArtistInfo("Jimi Hendrix", ("jimi-hendrix.txt",)),
    "johnny-cash": ArtistInfo("Johnny Cash", ("johnny-cash.txt",)),
    "joni-mitchell": ArtistInfo("Joni Mitchell", ("joni-mitchell.txt",)),
    "kanye-west": ArtistInfo(
        "Kanye West", ("kanye.txt", "kanye-west.txt", "Kanye_West.txt")
    ),
    "lady-gaga": ArtistInfo("Lady Gaga", ("lady-gaga.txt",)),
    "leonard-cohen": ArtistInfo("Leonard Cohen", ("leonard-cohen.txt",)),
    "lil-wayne": ArtistInfo("Lil Wayne", ("lil-wayne.txt", "Lil_Wayne.txt")),
    "lin-manuel-miranda": ArtistInfo(
        "Lin-Manuel Miranda", ("lin-manuel-miranda.txt",)
    ),
    "lorde": ArtistInfo("Lorde", ("lorde.txt",)),
    "ludacris": ArtistInfo("Ludacris", ("ludacris.txt",)),
    "michael-jackson": ArtistInfo("Michael Jackson", ("michael-jackson.txt",)),
    "missy-elliott": ArtistInfo("Missy Elliott", ("missy-elliott.txt",)),
    "nickelback": ArtistInfo("Nickelback", ("nickelback.txt",)),
    "nicki-minaj": ArtistInfo("Nicki Minaj", ("nicki-minaj.txt",)),
    "nirvana": ArtistInfo("Nirvana", ("nirvana.txt",)),
    "notorious-big": ArtistInfo(
        "The Notorious B.I.G.", ("notorious-big.txt", "notorious_big.txt")
    ),
    "nursery-rhymes": ArtistInfo("Nursery Rhymes", ("nursery_rhymes.txt",)),
    "patti-smith": ArtistInfo("Patti Smith", ("patti-smith.txt",)),
    "paul-simon": ArtistInfo("Paul Simon", ("paul-simon.txt",)),
    "prince": ArtistInfo("Prince", ("prince.txt",)),
    "r-kelly": ArtistInfo("R. Kelly", ("r-kelly.txt",)),
    "radiohead": ArtistInfo("Radiohead", ("radiohead.txt",)),
    "rihanna": ArtistInfo("Rihanna", ("rihanna.txt",)),
    "taylor-swift": ArtistInfo("Taylor Swift", ("Taylor-swift.txt",)),
}
