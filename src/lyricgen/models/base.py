"""Base class shared by all lyric generation models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor, nn


class LyricsModel(nn.Module, ABC):
    """Common interface for lyric generation models.

    Subclasses must set `self.config` (a JSON-serializable dict containing
    everything needed to rebuild the model via `build_model`) and
    `self.max_context` (an int for models with a fixed context window, or
    None for models that can process arbitrarily long sequences, such as
    recurrent networks).
    """

    config: dict[str, Any]
    max_context: int | None

    @abstractmethod
    def forward(
        self,
        tokens: Tensor,
        artists: Tensor,
        state: Any = None,
    ) -> tuple[Tensor, Any]:
        """Compute next-token logits for a batch of token sequences.

        Args:
            tokens: LongTensor of shape [B, T] with token ids.
            artists: LongTensor of shape [B] with artist ids (0 means
                unconditional).
            state: optional recurrent state carried across calls for
                streaming generation. Ignored by stateless models.

        Returns:
            A tuple `(logits, state)` where `logits` has shape [B, T, V]
            and `state` is the updated recurrent state (None for models
            that do not use one).
        """
        raise NotImplementedError

    def num_parameters(self) -> int:
        """Return the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
