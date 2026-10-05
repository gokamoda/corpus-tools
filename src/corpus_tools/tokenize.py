"""Tokenize documents, and cache the token ids of a whole corpus.

Texts are tokenized as they are, without special tokens
(``add_special_tokens=False``) and without adding a leading space. A BOS
token, if wanted, is added when counting (see ``count.py``).

The cache holds the token ids of all documents in one flat array
(``tokens.bin``; uint16 if the vocabulary fits, else uint32) and the start of
each document in ``offsets.npy`` (int64, one more entry than documents).
"""

import json
import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np


def token_dtype(vocab_size: int) -> type[np.unsignedinteger]:
    return np.uint16 if vocab_size <= 2**16 else np.uint32


def tokenize(
    texts: Iterable[str], tokenizer: Any, *, batch_size: int = 1000
) -> Iterator[np.ndarray]:
    """Token ids (int64) of each text, in order."""
    batch: list[str] = []
    for text in texts:
        batch.append(text)
        if len(batch) == batch_size:
            yield from _encode(batch, tokenizer)
            batch = []
    if batch:
        yield from _encode(batch, tokenizer)


def _encode(batch: list[str], tokenizer: Any) -> Iterator[np.ndarray]:
    for ids in tokenizer(batch, add_special_tokens=False)["input_ids"]:
        yield np.asarray(ids, dtype=np.int64)


class TokenizedCorpus:
    """Token ids of a corpus, read lazily from a cache directory."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.meta = json.loads((directory / "meta.json").read_text())
        self.offsets = np.load(directory / "offsets.npy")
        if self.meta["num_tokens"] == 0:  # np.memmap cannot map an empty file
            self.tokens = np.empty(0, dtype=self.meta["dtype"])
        else:
            self.tokens = np.memmap(
                directory / "tokens.bin", dtype=self.meta["dtype"], mode="r"
            )

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, index: int) -> np.ndarray:
        return self.tokens[self.offsets[index] : self.offsets[index + 1]]

    def __iter__(self) -> Iterator[np.ndarray]:
        for start, end in zip(self.offsets[:-1], self.offsets[1:]):
            yield self.tokens[start:end]

    @staticmethod
    def exists(directory: Path) -> bool:
        return (directory / "meta.json").exists()

    @classmethod
    def write(
        cls,
        docs: Iterable[np.ndarray],
        directory: Path,
        *,
        vocab_size: int,
        meta: dict[str, Any] | None = None,
    ) -> "TokenizedCorpus":
        """Save token ids of ``docs`` to ``directory``.

        Writes to a temporary directory first, so an interrupted run leaves no
        cache that looks complete.
        """
        dtype = token_dtype(vocab_size)
        tmp = directory.with_name(directory.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        offsets = [0]
        with (tmp / "tokens.bin").open("wb") as tokens_file:
            for ids in docs:
                if len(ids) and (ids.min() < 0 or ids.max() >= vocab_size):
                    raise ValueError(f"token id out of [0, {vocab_size})")
                tokens_file.write(ids.astype(dtype).tobytes())
                offsets.append(offsets[-1] + len(ids))
        np.save(tmp / "offsets.npy", np.asarray(offsets, dtype=np.int64))
        full_meta = {
            **(meta or {}),
            "vocab_size": vocab_size,
            "dtype": np.dtype(dtype).name,
            "num_documents": len(offsets) - 1,
            "num_tokens": offsets[-1],
        }
        (tmp / "meta.json").write_text(json.dumps(full_meta, indent=2) + "\n")
        if directory.exists():
            shutil.rmtree(directory)
        tmp.rename(directory)
        return cls(directory)
