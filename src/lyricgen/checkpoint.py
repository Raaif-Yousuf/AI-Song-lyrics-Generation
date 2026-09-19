"""Checkpoint save/load format shared by training, generation and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from lyricgen.data.artists import ARTISTS
from lyricgen.dataset import ANY_ARTIST, ANY_ARTIST_ID, ArtistVocab
from lyricgen.models import LyricsModel, build_model
from lyricgen.sampling import generate
from lyricgen.tokenizers import BPETokenizer, CharTokenizer, tokenizer_from_dict

FORMAT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    model: LyricsModel,
    tokenizer: CharTokenizer | BPETokenizer,
    artist_vocab: ArtistVocab,
    experiment_config: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> None:
    """Save a full checkpoint (model, tokenizer, artist vocab, config, metrics).

    Everything stored is a plain container, string, number or tensor so the
    file can be loaded back with `torch.load(..., weights_only=True)`.
    """
    payload = {
        "format_version": FORMAT_VERSION,
        "model_config": dict(model.config),
        "model_state": model.state_dict(),
        "tokenizer": tokenizer.to_dict(),
        "artists": artist_vocab.to_dict(),
        "experiment": experiment_config,
        "metrics": dict(metrics) if metrics else {},
    }
    torch.save(payload, path)


@dataclass
class LoadedModel:
    """A trained model plus everything needed to generate text with it."""

    model: LyricsModel
    tokenizer: CharTokenizer | BPETokenizer
    artist_vocab: ArtistVocab
    experiment: dict[str, Any]
    metrics: dict[str, Any]

    def artists(self) -> list[tuple[str, str]]:
        """Known artist (slug, display name) pairs, sorted by display name."""
        pairs = []
        for artist_id in range(1, len(self.artist_vocab)):
            slug = self.artist_vocab.slug(artist_id)
            info = ARTISTS.get(slug)
            display_name = info.display_name if info is not None else slug
            pairs.append((slug, display_name))
        return sorted(pairs, key=lambda pair: pair[1])

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
        """Generate `length` characters of new text (the prompt is not included).

        `artist` of `None` or `"<any>"` means unconditional generation. An
        unknown artist slug raises `ValueError` listing the valid slugs.
        Tokens are generated in one or more passes until the decoded text is
        at least `length` characters long, then trimmed to exactly `length`.
        """
        if artist is None or artist == ANY_ARTIST:
            artist_id = ANY_ARTIST_ID
        else:
            try:
                artist_id = self.artist_vocab.id(artist)
            except KeyError:
                valid = [slug for slug, _ in self.artists()]
                raise ValueError(
                    f"unknown artist {artist!r}; valid artists: {valid}"
                ) from None

        prompt_text = prompt if prompt else "\n"
        prompt_ids = list(self.tokenizer.encode(prompt_text))
        if not prompt_ids:
            raise ValueError("prompt encoded to no tokens")

        if length <= 0:
            return ""

        generated_ids: list[int] = []
        decoded = ""
        max_attempts = 8
        for attempt in range(max_attempts):
            remaining_chars = length - len(decoded)
            if remaining_chars <= 0:
                break
            want_tokens = max(remaining_chars, 1)
            step_seed = None if seed is None else seed + attempt
            new_ids = generate(
                self.model,
                prompt_ids + generated_ids,
                artist_id,
                max_new_tokens=want_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=step_seed,
            )
            generated_ids.extend(new_ids)
            decoded = self.tokenizer.decode(generated_ids)

        return decoded[:length]


def load_checkpoint(path: str | Path, device: str = "cpu") -> LoadedModel:
    """Load a checkpoint written by `save_checkpoint`."""
    payload = torch.load(path, map_location=device, weights_only=True)

    model = build_model(dict(payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()

    tokenizer = tokenizer_from_dict(payload["tokenizer"])
    artist_vocab = ArtistVocab.from_dict(payload["artists"])

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        artist_vocab=artist_vocab,
        experiment=dict(payload.get("experiment", {})),
        metrics=dict(payload.get("metrics", {})),
    )
