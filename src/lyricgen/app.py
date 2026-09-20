"""Gradio demo for interactive, artist-conditioned lyric generation."""

from __future__ import annotations

import argparse
import logging
from typing import Any, Protocol

import gradio as gr

logger = logging.getLogger(__name__)

ANY_ARTIST_LABEL = "Any artist"


class LoadedModelLike(Protocol):
    """Structural interface a loaded checkpoint must provide for the demo."""

    experiment: dict[str, Any]
    metrics: dict[str, Any]

    def artists(self) -> list[tuple[str, str]]: ...

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
    ) -> str: ...


def artist_choices(loaded: LoadedModelLike) -> list[str]:
    """Dropdown labels: the unconditional option followed by display names."""
    return [ANY_ARTIST_LABEL, *(display for _, display in loaded.artists())]


def slug_for_display(loaded: LoadedModelLike, display: str | None) -> str | None:
    """Map a dropdown label back to an artist slug, or None for unconditional."""
    if display is None or display == ANY_ARTIST_LABEL:
        return None
    for slug, name in loaded.artists():
        if name == display:
            return slug
    raise ValueError(f"unknown artist: {display!r}")


def make_generate_fn(loaded: LoadedModelLike):
    """Build the plain callback used by the Generate button, bound to `loaded`."""

    def generate(
        prompt: str,
        artist_display: str,
        length: float,
        temperature: float,
        top_k: float,
        top_p: float,
        repetition_penalty: float,
        seed: float,
    ) -> str:
        artist_slug = slug_for_display(loaded, artist_display)
        seed_int = int(seed)
        seed_value = seed_int if seed_int >= 0 else None
        return loaded.generate_text(
            prompt=prompt or "",
            artist=artist_slug,
            length=int(length),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
            seed=seed_value,
        )

    return generate


def status_line(loaded: LoadedModelLike) -> str:
    """One line summarizing the loaded model: kind, parameter count, val bpc."""
    kind = getattr(loaded.model, "config", {}).get("kind", "unknown")
    parts = [f"model: {kind}"]

    num_parameters = getattr(loaded.model, "num_parameters", None)
    if callable(num_parameters):
        parts.append(f"parameters: {num_parameters():,}")

    metrics = loaded.metrics if isinstance(loaded.metrics, dict) else {}
    val_bpc = metrics.get("val_bpc")
    if val_bpc is not None:
        parts.append(f"validation bits-per-char: {val_bpc:.3f}")

    return " | ".join(parts)


def build_demo(loaded: LoadedModelLike) -> gr.Blocks:
    """Build the Gradio Blocks app for a loaded checkpoint."""
    generate_fn = make_generate_fn(loaded)
    choices = artist_choices(loaded)

    with gr.Blocks(title="lyricgen") as demo:
        gr.Markdown("# lyricgen")
        gr.Markdown(status_line(loaded))
        with gr.Row():
            with gr.Column():
                artist = gr.Dropdown(
                    choices=choices,
                    value=choices[0],
                    label="Artist",
                    elem_id="artist-dropdown",
                )
                prompt = gr.Textbox(
                    label="Prompt",
                    lines=3,
                    placeholder="Start of a verse",
                    elem_id="prompt-input",
                )
                length = gr.Slider(20, 2000, value=400, step=20, label="Length (characters)")
                temperature = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="Temperature")
                top_k = gr.Slider(0, 200, value=0, step=1, label="Top-k (0 disables it)")
                top_p = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Top-p")
                repetition_penalty = gr.Slider(
                    1.0, 2.0, value=1.0, step=0.01, label="Repetition penalty"
                )
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 for random)")
                generate_button = gr.Button("Generate", elem_id="generate-button")
            with gr.Column():
                output = gr.Textbox(
                    label="Generated lyrics", lines=20, elem_id="output-text"
                )

        generate_button.click(
            fn=generate_fn,
            inputs=[prompt, artist, length, temperature, top_k, top_p, repetition_penalty, seed],
            outputs=output,
        )

    return demo


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, load a checkpoint and launch the demo."""
    from lyricgen.pretrained import MANIFEST

    parser = argparse.ArgumentParser(description="lyricgen gradio demo")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--checkpoint", default="runs/transformer/best.pt")
    checkpoint_group.add_argument("--pretrained", choices=sorted(MANIFEST), default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", default=False)
    args = parser.parse_args(argv)

    from lyricgen.checkpoint import load_checkpoint
    from lyricgen.pretrained import resolve_checkpoint

    checkpoint_path = resolve_checkpoint(args.checkpoint, args.pretrained)
    loaded = load_checkpoint(checkpoint_path)
    demo = build_demo(loaded)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
