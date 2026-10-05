"""Hash sampling and n-gram counting of Hugging Face corpora."""

from corpus_tools.corpus import PRESETS, Corpus, preset
from corpus_tools.sample import (
    hash_fold,
    hash_key,
    hash_sample,
    iter_hash_fold,
    load_hash_sample,
    save_hash_sample,
)
from corpus_tools.store import Store

__all__ = [
    "PRESETS",
    "Corpus",
    "Store",
    "hash_fold",
    "hash_key",
    "hash_sample",
    "iter_hash_fold",
    "load_hash_sample",
    "preset",
    "save_hash_sample",
]
