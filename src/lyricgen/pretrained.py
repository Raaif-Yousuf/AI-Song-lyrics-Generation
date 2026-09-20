"""Download and verify pretrained checkpoints published as release assets."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

RELEASE_URL = "https://github.com/Raaif-Yousuf/AI-Song-lyrics-Generation/releases/download/v0.2.0/"

# sha256/size are placeholders until the checkpoints are published; download()
# skips verification for an entry whose sha256 is empty.
MANIFEST: dict[str, dict] = {
    "lstm": {
        "file": "lstm.pt",
        "sha256": "8945fda36107435f4fcb0cb4f935489856c1b36784de64b9b75ca0deb64d31f2",
        "size": 2374921,
    },
    "gru": {
        "file": "gru.pt",
        "sha256": "515d2b17a06eb8644792add09a643ff7e1a3a54651cd53b97898ec47b988e892",
        "size": 1814793,
    },
    "transformer": {
        "file": "transformer.pt",
        "sha256": "ce1bfd8f2e8be4b6d0e1cc3651e4d7b700f8d275fc1b7de4812547f1608409ae",
        "size": 13128403,
    },
    "transformer_bpe": {
        "file": "transformer_bpe.pt",
        "sha256": "a7cf80979159e7109791e4d166896c95f9fddc20a90f31449640c8349bc69c88",
        "size": 15062867,
    },
}

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "lyricgen"
_CHUNK_SIZE = 1 << 16


class PretrainedError(RuntimeError):
    """Raised when a pretrained checkpoint cannot be downloaded or verified."""


def _cache_dir(cache_dir: str | Path | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    env_value = os.environ.get("LYRICGEN_CACHE")
    if env_value:
        return Path(env_value)
    return _DEFAULT_CACHE_DIR


def _sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected_sha256: str) -> None:
    actual = _sha256sum(path)
    if actual != expected_sha256:
        path.unlink(missing_ok=True)
        raise PretrainedError(
            f"checksum mismatch for {path.name}: expected {expected_sha256}, got {actual}"
        )


def download(name: str, cache_dir: str | Path | None = None, force: bool = False) -> Path:
    """Download a pretrained checkpoint, returning its local path.

    A verified copy already in the cache is reused unless `force` is True.
    `cache_dir` defaults to `$LYRICGEN_CACHE`, then `~/.cache/lyricgen`. The
    file is fetched to a temporary file in the destination directory and
    atomically renamed into place only once its sha256 (when the manifest
    has one) has been verified, so a failed or interrupted download never
    leaves a corrupt file at the cached path.
    """
    if name not in MANIFEST:
        raise ValueError(
            f"unknown pretrained checkpoint {name!r}; expected one of {sorted(MANIFEST)}"
        )

    entry = MANIFEST[name]
    expected_sha256 = entry.get("sha256") or None
    dest_dir = _cache_dir(cache_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / entry["file"]

    if not force and dest_path.exists():
        if expected_sha256 is None:
            return dest_path
        try:
            _verify(dest_path, expected_sha256)
            return dest_path
        except PretrainedError:
            logger.warning("cached %s failed verification, re-downloading", dest_path)

    url = RELEASE_URL + entry["file"]
    fd, tmp_name = tempfile.mkstemp(dir=dest_dir, prefix=f".{entry['file']}.", suffix=".part")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as tmp_fh, urllib.request.urlopen(url) as response:
            while True:
                chunk = response.read(_CHUNK_SIZE)
                if not chunk:
                    break
                tmp_fh.write(chunk)

        if expected_sha256 is not None:
            _verify(tmp_path, expected_sha256)

        tmp_path.replace(dest_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return dest_path


def resolve_checkpoint(checkpoint: str | Path | None, pretrained: str | None) -> Path:
    """Resolve a checkpoint path from a `--checkpoint` or `--pretrained` choice.

    Exactly one of `checkpoint`/`pretrained` must be given; raises
    `ValueError` otherwise. Callers such as the CLI should also enforce
    this mutual exclusivity themselves (e.g. an argparse mutually exclusive
    group) so the user gets an earlier, clearer error.
    """
    if checkpoint is not None and pretrained is not None:
        raise ValueError("specify only one of checkpoint or pretrained")
    if pretrained is not None:
        return download(pretrained)
    if checkpoint is not None:
        return Path(checkpoint)
    raise ValueError("either checkpoint or pretrained must be given")
