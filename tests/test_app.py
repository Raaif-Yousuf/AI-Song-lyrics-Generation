from __future__ import annotations

from typing import Any

import pytest

gr = pytest.importorskip("gradio")

from lyricgen.app import (  # noqa: E402
    ANY_ARTIST_LABEL,
    artist_choices,
    build_demo,
    make_generate_fn,
    slug_for_display,
    status_line,
)


class _FakeModel:
    config = {"kind": "transformer"}

    def num_parameters(self) -> int:
        return 42


class _FakeLoaded:
    """Minimal stand-in for LoadedModel used to test app plumbing."""

    def __init__(self) -> None:
        self.model = _FakeModel()
        self.tokenizer = object()
        self.artist_vocab = object()
        self.experiment: dict[str, Any] = {"model": {"kind": "transformer"}}
        self.metrics: dict[str, Any] = {"val_bpc": 1.5}
        self.calls: list[dict[str, Any]] = []

    def artists(self) -> list[tuple[str, str]]:
        return [("drake", "Drake"), ("eminem", "Eminem")]

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
        self.calls.append(
            dict(
                prompt=prompt,
                artist=artist,
                length=length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
            )
        )
        return "generated text"


def test_artist_choices_lists_any_then_display_names() -> None:
    loaded = _FakeLoaded()
    assert artist_choices(loaded) == [ANY_ARTIST_LABEL, "Drake", "Eminem"]


def test_slug_for_display_maps_name_to_slug() -> None:
    loaded = _FakeLoaded()
    assert slug_for_display(loaded, "Drake") == "drake"
    assert slug_for_display(loaded, "Eminem") == "eminem"


def test_slug_for_display_any_artist_is_none() -> None:
    loaded = _FakeLoaded()
    assert slug_for_display(loaded, ANY_ARTIST_LABEL) is None
    assert slug_for_display(loaded, None) is None


def test_slug_for_display_unknown_raises() -> None:
    loaded = _FakeLoaded()
    with pytest.raises(ValueError):
        slug_for_display(loaded, "Nobody")


def test_status_line_includes_kind_params_and_bpc() -> None:
    loaded = _FakeLoaded()
    line = status_line(loaded)
    assert "transformer" in line
    assert "42" in line
    assert "1.500" in line


def test_status_line_omits_bpc_when_absent() -> None:
    loaded = _FakeLoaded()
    loaded.metrics = {}
    line = status_line(loaded)
    assert "bits-per-char" not in line


def test_build_demo_constructs_blocks() -> None:
    loaded = _FakeLoaded()
    demo = build_demo(loaded)
    assert isinstance(demo, gr.Blocks)


def test_generate_callback_plumbs_artist_display_name_to_slug() -> None:
    loaded = _FakeLoaded()
    generate = make_generate_fn(loaded)

    generate("hello", "Drake", 400.0, 0.8, 10.0, 1.0, 1.0, 3.0)

    assert loaded.calls[-1]["artist"] == "drake"


def test_generate_callback_plumbs_any_artist_to_none() -> None:
    loaded = _FakeLoaded()
    generate = make_generate_fn(loaded)

    generate("hello", ANY_ARTIST_LABEL, 400.0, 0.8, 10.0, 1.0, 1.0, 3.0)

    assert loaded.calls[-1]["artist"] is None


def test_generate_callback_casts_top_k_and_seed_to_int() -> None:
    loaded = _FakeLoaded()
    generate = make_generate_fn(loaded)

    generate("hello", ANY_ARTIST_LABEL, 400.0, 0.8, 10.0, 1.0, 1.0, 3.0)

    call = loaded.calls[-1]
    assert call["top_k"] == 10
    assert isinstance(call["top_k"], int)
    assert call["seed"] == 3
    assert isinstance(call["seed"], int)


def test_generate_callback_negative_seed_means_none() -> None:
    loaded = _FakeLoaded()
    generate = make_generate_fn(loaded)

    generate("hello", ANY_ARTIST_LABEL, 400.0, 0.8, 0.0, 1.0, 1.0, -1.0)

    assert loaded.calls[-1]["seed"] is None


def test_generate_callback_returns_model_output() -> None:
    loaded = _FakeLoaded()
    generate = make_generate_fn(loaded)

    result = generate("hello", ANY_ARTIST_LABEL, 400.0, 0.8, 0.0, 1.0, 1.0, -1.0)

    assert result == "generated text"
