"""Restore the raw per-artist lyric files from project history into data/raw/."""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

SOURCE_COMMIT = "0fe1405"
ARTISTS_PATH = "Artists"
AGGREGATE_FILES = {"All-songs.txt", "Best-songs-1900s.txt"}


def _run_git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"git {' '.join(args)} failed: {stderr.strip()}")
    return result.stdout.decode("utf-8", errors="replace")


def _run_git_bytes(args: list[str], cwd: Path) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"git {' '.join(args)} failed: {stderr.strip()}")
    return result.stdout


def _check_commit_exists(repo_root: Path, commit: str) -> None:
    result = subprocess.run(
        ["git", "cat-file", "-e", commit],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Commit {commit} is not available in this clone. "
            "If this is a shallow clone, run `git fetch --unshallow` and try again."
        )


def list_artist_files(repo_root: Path, commit: str = SOURCE_COMMIT) -> list[str]:
    """Return the raw filenames tracked under Artists/ at the given commit."""
    output = _run_git(["ls-tree", "--name-only", commit, f"{ARTISTS_PATH}/"], cwd=repo_root)
    names = [line.split("/", 1)[1] for line in output.splitlines() if line.strip()]
    return sorted(names)


def fetch_data(
    repo_root: Path,
    out_dir: Path,
    commit: str = SOURCE_COMMIT,
    include_aggregates: bool = False,
) -> list[str]:
    """Write each raw artist file from history into out_dir. Returns files written."""
    _check_commit_exists(repo_root, commit)
    filenames = list_artist_files(repo_root, commit)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for filename in filenames:
        if not include_aggregates and filename in AGGREGATE_FILES:
            continue
        data = _run_git_bytes(["show", f"{commit}:{ARTISTS_PATH}/{filename}"], cwd=repo_root)
        dest = out_dir / filename
        dest.write_bytes(data)
        written.append(filename)

    return written


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the git repository (default: project root).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Destination directory (default: <repo-root>/data/raw).",
    )
    parser.add_argument(
        "--commit",
        default=SOURCE_COMMIT,
        help="Commit to restore the raw files from.",
    )
    parser.add_argument(
        "--include-aggregates",
        action="store_true",
        help="Also fetch the mixed-artist aggregate files.",
    )
    args = parser.parse_args(argv)

    out_dir = args.out_dir or (args.repo_root / "data" / "raw")
    written = fetch_data(
        repo_root=args.repo_root,
        out_dir=out_dir,
        commit=args.commit,
        include_aggregates=args.include_aggregates,
    )
    logger.info("Wrote %d raw file(s) to %s", len(written), out_dir)


if __name__ == "__main__":
    main()
