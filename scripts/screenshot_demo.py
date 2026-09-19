"""Launch the demo and capture a screenshot for documentation.

Starts the Gradio app in the current process (non-blocking launch) on a
free local port, drives it with Playwright using the system-installed
Microsoft Edge, and saves a screenshot. Requires the same environment as
the demo itself plus `playwright` with Edge available (`channel="msedge"`,
no separate browser download).
"""

from __future__ import annotations

import argparse
import logging
import socket
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_OUT = Path("docs/assets/demo.png")


class _FakeModel:
    """Tiny stand-in for a LyricsModel, used only with --fake."""

    config = {"kind": "transformer"}

    def num_parameters(self) -> int:
        return 3_263_488


class _FakeLoaded:
    """Tiny stand-in for LoadedModel, used only with --fake so the demo can
    be screenshotted without a real checkpoint on disk."""

    def __init__(self) -> None:
        self.model = _FakeModel()
        self.tokenizer = object()
        self.artist_vocab = object()
        self.experiment: dict[str, Any] = {"model": {"kind": "transformer"}}
        self.metrics: dict[str, Any] = {"val_bpc": 1.42}

    def artists(self) -> list[tuple[str, str]]:
        return [("drake", "Drake"), ("eminem", "Eminem"), ("kanye-west", "Kanye West")]

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
        del length, temperature, top_k, top_p, repetition_penalty, seed
        body = prompt or "\n"
        who = artist or "no particular artist"
        return f"{body}\nthis is a sample line for {who}\none more line after that"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _build_loaded(checkpoint: str, fake: bool) -> Any:
    if fake:
        return _FakeLoaded()
    from lyricgen.checkpoint import load_checkpoint

    return load_checkpoint(checkpoint)


def _wait_for_server(host: str, port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.2)
    message = f"demo server did not start on {host}:{port} within {timeout_s}s"
    raise RuntimeError(message) from last_error


def _capture(url: str, out_path: Path, width: int, height: int) -> None:
    from playwright.sync_api import sync_playwright

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge")
        try:
            page = browser.new_page(viewport={"width": width, "height": height})
            page.goto(url, wait_until="networkidle")
            page.evaluate("window.scrollTo(0, 0)")

            page.locator("#artist-dropdown input").click()
            page.get_by_role("option", name="Eminem").click()
            page.keyboard.press("Escape")

            prompt_box = page.locator("#prompt-input textarea")
            prompt_box.click()
            prompt_box.fill("Started from the bottom")

            page.locator("#generate-button").click()
            output = page.locator("#output-text textarea")
            page.wait_for_function(
                "el => el.value.trim().length > 0", arg=output.element_handle(), timeout=15_000
            )
            page.wait_for_timeout(500)

            page.screenshot(path=str(out_path))
        finally:
            browser.close()


def capture_screenshot(
    checkpoint: str,
    out_path: Path,
    fake: bool = False,
    host: str = "127.0.0.1",
    port: int | None = None,
    width: int = 1200,
    height: int = 800,
    timeout_s: float = 20.0,
) -> None:
    """Launch the demo, screenshot a generated example, and shut it down."""
    from lyricgen.app import build_demo

    resolved_port = port if port is not None else _free_port()
    loaded = _build_loaded(checkpoint, fake)
    demo = build_demo(loaded)
    demo.launch(
        server_name=host,
        server_port=resolved_port,
        share=False,
        prevent_thread_lock=True,
        show_error=True,
        quiet=True,
    )
    try:
        _wait_for_server(host, resolved_port, timeout_s)
        _capture(f"http://{host}:{resolved_port}", out_path, width, height)
    finally:
        demo.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="capture a screenshot of the lyricgen demo")
    parser.add_argument("--checkpoint", default="runs/transformer/best.pt")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--fake", action="store_true", default=False, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    capture_screenshot(
        checkpoint=args.checkpoint,
        out_path=Path(args.out),
        fake=args.fake,
        host=args.host,
        port=args.port,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
