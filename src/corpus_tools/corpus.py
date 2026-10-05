"""Which corpus to read, and how its rows are read.

A corpus is a split of a Hugging Face dataset. Rows are read either by
streaming (nothing is stored) or from the Hugging Face cache (the dataset is
downloaded once, and later reads skip the download).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Corpus:
    dataset: str  # Hugging Face dataset id, e.g. "Skylion007/openwebtext"
    name: str  # dataset config, e.g. "plain_text"
    split: str = "train"
    text_field: str = "text"

    @property
    def path(self) -> Path:
        """Directory of this corpus under the cache directory."""
        return Path(self.dataset.replace("/", "--"), self.name, self.split)


# alias -> (dataset, config); config None means it has to be given
PRESETS: dict[str, tuple[str, str | None]] = {
    "openwebtext": ("Skylion007/openwebtext", "plain_text"),
    "tinystories": ("roneneldan/TinyStories", "default"),
    "wikipedia": ("wikimedia/wikipedia", None),  # e.g. "20231101.ja", "20231101.en"
}


def preset(alias: str, split: str = "train", name: str | None = None) -> Corpus:
    """The corpus of ``alias``; ``name`` is the config (for wikipedia, the dump)."""
    if alias not in PRESETS:
        raise ValueError(f"Unknown corpus {alias!r}. Choose from: {', '.join(PRESETS)}")
    dataset, default_name = PRESETS[alias]
    name = name or default_name
    if name is None:
        raise ValueError(f"{alias} needs a config name, e.g. 20231101.ja")
    return Corpus(dataset=dataset, name=name, split=split)


@contextmanager
def open_rows(
    corpus: Corpus,
    *,
    revision: str | None = None,
    hf_cache_dir: Path | None = None,
) -> Iterator[tuple[Iterator[dict[str, Any]], int | None]]:
    """Rows of the corpus and their number (None if unknown).

    With ``hf_cache_dir`` the dataset is downloaded into that directory (or
    read from it if already there); otherwise it is streamed. ``revision``
    pins the dataset to a commit of its Hub repository.
    """
    from datasets import load_dataset

    streaming = hf_cache_dir is None
    dataset = load_dataset(
        corpus.dataset,
        name=corpus.name,
        split=corpus.split,
        streaming=streaming,
        revision=revision,
        cache_dir=None if hf_cache_dir is None else str(hf_cache_dir),
    )
    splits = dataset.info.splits
    total = splits[corpus.split].num_examples if splits else None
    rows = iter(dataset)
    try:
        yield rows, total
    finally:
        close = getattr(rows, "close", None)
        if close is not None:
            close()


def dataset_revision(corpus: Corpus) -> str | None:
    """Commit sha of the dataset on the Hub (None if it cannot be reached)."""
    from huggingface_hub import HfApi

    try:
        return HfApi().dataset_info(corpus.dataset).sha
    except Exception:  # offline, or the Hub is down
        return None
