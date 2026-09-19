"""Character and byte-level BPE tokenizers with a shared interface."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer as _HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1


class CharTokenizer:
    """Maps individual characters to ids. Unseen characters map to `<unk>`."""

    def __init__(self, char_to_id: dict[str, int]) -> None:
        self._char_to_id = dict(char_to_id)
        self._id_to_char = {i: c for c, i in self._char_to_id.items()}

    @classmethod
    def train(cls, texts: Iterable[str], **kw: object) -> CharTokenizer:
        chars: set[str] = set()
        for text in texts:
            chars.update(text)
        char_to_id = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
        for i, char in enumerate(sorted(chars), start=2):
            char_to_id[char] = i
        return cls(char_to_id)

    def encode(self, text: str) -> list[int]:
        return [self._char_to_id.get(c, UNK_ID) for c in text]

    def decode(self, ids: Iterable[int]) -> str:
        skip = {PAD_ID, UNK_ID}
        return "".join(self._id_to_char.get(i, "") for i in ids if i not in skip)

    @property
    def vocab_size(self) -> int:
        return len(self._char_to_id)

    def to_dict(self) -> dict:
        return {"type": "char", "char_to_id": self._char_to_id}

    @classmethod
    def from_dict(cls, d: dict) -> CharTokenizer:
        return cls(d["char_to_id"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CharTokenizer:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class BPETokenizer:
    """Byte-level BPE tokenizer backed by Hugging Face `tokenizers`."""

    def __init__(self, hf_tokenizer: _HFTokenizer) -> None:
        self._tokenizer = hf_tokenizer

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 4096,
        min_frequency: int = 2,
        **kw: object,
    ) -> BPETokenizer:
        hf_tokenizer = _HFTokenizer(models.BPE(unk_token=UNK_TOKEN))
        hf_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        hf_tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=[PAD_TOKEN, UNK_TOKEN],
        )
        hf_tokenizer.train_from_iterator(texts, trainer=trainer)
        vocab = hf_tokenizer.get_vocab()
        if vocab.get(PAD_TOKEN) != PAD_ID or vocab.get(UNK_TOKEN) != UNK_ID:
            raise RuntimeError(
                "BPE special token ids did not come out as expected "
                f"(pad={vocab.get(PAD_TOKEN)}, unk={vocab.get(UNK_TOKEN)})"
            )
        return cls(hf_tokenizer)

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, ids: Iterable[int]) -> str:
        return self._tokenizer.decode(list(ids), skip_special_tokens=True)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def to_dict(self) -> dict:
        return {"type": "bpe", "tokenizer_json": self._tokenizer.to_str()}

    @classmethod
    def from_dict(cls, d: dict) -> BPETokenizer:
        return cls(_HFTokenizer.from_str(d["tokenizer_json"]))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


_REGISTRY = {"char": CharTokenizer, "bpe": BPETokenizer}


def tokenizer_from_dict(d: dict) -> CharTokenizer | BPETokenizer:
    """Dispatch to the right tokenizer class based on `d["type"]`."""
    tokenizer_type = d.get("type")
    cls = _REGISTRY.get(tokenizer_type)
    if cls is None:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type!r}")
    return cls.from_dict(d)
