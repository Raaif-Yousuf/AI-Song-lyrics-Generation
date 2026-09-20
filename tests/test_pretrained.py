from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from lyricgen import pretrained


@pytest.fixture
def fake_release(tmp_path: Path):
    """A fake 'release' directory served over file:// (no network)."""
    release_dir = tmp_path / "release"
    release_dir.mkdir()

    def write(filename: str, content: bytes) -> None:
        (release_dir / filename).write_bytes(content)

    return release_dir, write


@pytest.fixture
def fake_manifest(monkeypatch, fake_release):
    release_dir, write = fake_release
    manifest = {
        "toy": {"file": "toy.pt", "sha256": "", "size": 0},
    }
    monkeypatch.setattr(pretrained, "MANIFEST", manifest)
    monkeypatch.setattr(pretrained, "RELEASE_URL", release_dir.as_uri() + "/")
    write("toy.pt", b"toy checkpoint bytes")
    return manifest


def test_download_fetches_and_caches(tmp_path, fake_manifest):
    cache_dir = tmp_path / "cache"
    path = pretrained.download("toy", cache_dir=cache_dir)

    assert path == cache_dir / "toy.pt"
    assert path.read_bytes() == b"toy checkpoint bytes"


def test_download_unknown_name_raises(tmp_path, fake_manifest):
    with pytest.raises(ValueError, match="unknown pretrained checkpoint"):
        pretrained.download("not-a-real-name", cache_dir=tmp_path / "cache")


def test_download_skips_refetch_when_cached(tmp_path, fake_manifest, fake_release):
    release_dir, write = fake_release
    cache_dir = tmp_path / "cache"

    first = pretrained.download("toy", cache_dir=cache_dir)
    assert first.read_bytes() == b"toy checkpoint bytes"

    # Change the "remote" content; a cached (unverified, but present) file
    # should be reused without re-fetching.
    write("toy.pt", b"CHANGED")
    second = pretrained.download("toy", cache_dir=cache_dir)
    assert second.read_bytes() == b"toy checkpoint bytes"


def test_download_force_refetches(tmp_path, fake_manifest, fake_release):
    release_dir, write = fake_release
    cache_dir = tmp_path / "cache"

    pretrained.download("toy", cache_dir=cache_dir)
    write("toy.pt", b"CHANGED")
    refetched = pretrained.download("toy", cache_dir=cache_dir, force=True)

    assert refetched.read_bytes() == b"CHANGED"


def test_download_verifies_correct_sha256(tmp_path, monkeypatch, fake_release):
    release_dir, write = fake_release
    content = b"verified content"
    write("toy.pt", content)
    digest = hashlib.sha256(content).hexdigest()
    manifest = {"toy": {"file": "toy.pt", "sha256": digest, "size": len(content)}}
    monkeypatch.setattr(pretrained, "MANIFEST", manifest)
    monkeypatch.setattr(pretrained, "RELEASE_URL", release_dir.as_uri() + "/")

    cache_dir = tmp_path / "cache"
    path = pretrained.download("toy", cache_dir=cache_dir)
    assert path.read_bytes() == content


def test_download_raises_clear_error_on_sha256_mismatch(tmp_path, monkeypatch, fake_release):
    release_dir, write = fake_release
    write("toy.pt", b"actual content")
    manifest = {"toy": {"file": "toy.pt", "sha256": "0" * 64, "size": 14}}
    monkeypatch.setattr(pretrained, "MANIFEST", manifest)
    monkeypatch.setattr(pretrained, "RELEASE_URL", release_dir.as_uri() + "/")

    cache_dir = tmp_path / "cache"
    with pytest.raises(pretrained.PretrainedError, match="checksum mismatch"):
        pretrained.download("toy", cache_dir=cache_dir)

    # A failed verification must not leave a corrupt file at the cached path.
    assert not (cache_dir / "toy.pt").exists()


def test_download_redownloads_when_cached_file_fails_verification(
    tmp_path, monkeypatch, fake_release
):
    release_dir, write = fake_release
    good_content = b"good content"
    digest = hashlib.sha256(good_content).hexdigest()
    manifest = {"toy": {"file": "toy.pt", "sha256": digest, "size": len(good_content)}}
    monkeypatch.setattr(pretrained, "MANIFEST", manifest)
    monkeypatch.setattr(pretrained, "RELEASE_URL", release_dir.as_uri() + "/")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    # Pre-seed a corrupt cached file (wrong content for the expected sha256).
    (cache_dir / "toy.pt").write_bytes(b"corrupt")
    write("toy.pt", good_content)

    path = pretrained.download("toy", cache_dir=cache_dir)
    assert path.read_bytes() == good_content


def test_download_uses_lyricgen_cache_env_var(tmp_path, monkeypatch, fake_manifest):
    env_cache_dir = tmp_path / "env_cache"
    monkeypatch.setenv("LYRICGEN_CACHE", str(env_cache_dir))

    path = pretrained.download("toy")
    assert path == env_cache_dir / "toy.pt"
    assert path.read_bytes() == b"toy checkpoint bytes"


def test_download_explicit_cache_dir_overrides_env_var(tmp_path, monkeypatch, fake_manifest):
    env_cache_dir = tmp_path / "env_cache"
    explicit_cache_dir = tmp_path / "explicit_cache"
    monkeypatch.setenv("LYRICGEN_CACHE", str(env_cache_dir))

    path = pretrained.download("toy", cache_dir=explicit_cache_dir)
    assert path == explicit_cache_dir / "toy.pt"
    assert not (env_cache_dir / "toy.pt").exists()


def test_resolve_checkpoint_uses_pretrained_when_only_pretrained_given(
    tmp_path, monkeypatch, fake_manifest
):
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("LYRICGEN_CACHE", str(cache_dir))

    path = pretrained.resolve_checkpoint(checkpoint=None, pretrained="toy")
    assert path == cache_dir / "toy.pt"


def test_resolve_checkpoint_rejects_both_given(fake_manifest):
    with pytest.raises(ValueError, match="only one"):
        pretrained.resolve_checkpoint(checkpoint="/some/path.pt", pretrained="toy")


def test_resolve_checkpoint_uses_checkpoint_when_no_pretrained():
    path = pretrained.resolve_checkpoint(checkpoint="/some/path.pt", pretrained=None)
    assert path == Path("/some/path.pt")


def test_resolve_checkpoint_requires_one_of_them():
    with pytest.raises(ValueError, match="either checkpoint or pretrained"):
        pretrained.resolve_checkpoint(checkpoint=None, pretrained=None)
