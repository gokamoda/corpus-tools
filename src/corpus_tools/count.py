"""Count n-grams of token ids, within each document.

The n-grams of a document never cross into the next document. Counts are
returned as

- n = 1: a dense array [vocab_size] (int64);
- n >= 2: a sparse matrix [vocab_size, vocab_size ** (n - 1)] (CSR, int64),
  where n-gram (a_1, ..., a_n) is at row a_1 and column
  a_2 * V^(n-2) + ... + a_n. For n = 2 this is row a (prefix), column b
  (suffix), as in lm-detokenization's bigram counts.

N-grams are encoded as int64 keys a_1 * V^(n-1) + ... + a_n, aggregated with
np.unique in chunks and added to the sparse total, so memory grows with the
number of distinct n-grams, not with the corpus. This is how
lm-detokenization counted the bigrams of all of OpenWebText (about 9 billion
tokens, 137 million distinct bigrams) in about 2 hours with about 2.4GB.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from scipy import sparse
from tqdm import tqdm

Counts = np.ndarray | sparse.csr_matrix

INT64_MAX = np.iinfo(np.int64).max


class _NgramAccumulator:
    def __init__(self, n: int, vocab_size: int, flush_every: int):
        if vocab_size**n - 1 > INT64_MAX:
            raise ValueError(
                f"{n}-grams of a vocabulary of {vocab_size} do not fit in int64 keys"
            )
        self.n = n
        self.vocab_size = vocab_size
        self.num_columns = vocab_size ** (n - 1)
        self.weights = [vocab_size ** (n - 1 - i) for i in range(n)]
        self.flush_every = flush_every
        self.pending: list[np.ndarray] = []
        self.pending_size = 0
        self.total = sparse.csr_matrix((vocab_size, self.num_columns), dtype=np.int64)

    def add(self, ids: np.ndarray) -> None:
        length = len(ids) - self.n + 1
        if length < 1:
            return
        keys = ids[:length] * self.weights[0]
        for i in range(1, self.n):
            keys += ids[i : i + length] * self.weights[i]
        self.pending.append(keys)
        self.pending_size += length
        if self.pending_size >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        keys, counts = np.unique(np.concatenate(self.pending), return_counts=True)
        chunk = sparse.csr_matrix(
            (
                counts.astype(np.int64),
                (keys // self.num_columns, keys % self.num_columns),
            ),
            shape=self.total.shape,
        )
        self.total = self.total + chunk
        self.pending, self.pending_size = [], 0

    def result(self) -> sparse.csr_matrix:
        self.flush()
        self.total.sum_duplicates()
        self.total.sort_indices()
        return self.total


def count_ngrams(
    docs: Iterable[np.ndarray],
    ns: Sequence[int],
    vocab_size: int,
    *,
    bos_id: int | None = None,
    flush_every: int = 200_000_000,
    total: int | None = None,
) -> dict[int, Counts]:
    """Counts of the n-grams of each n in ``ns`` over ``docs`` (token id arrays).

    With ``bos_id``, that token is put at the start of every document before
    counting.
    """
    if not ns or min(ns) < 1:
        raise ValueError("ns must be a non-empty list of positive integers")
    if bos_id is not None and not 0 <= bos_id < vocab_size:
        raise ValueError(f"bos_id {bos_id} is not in [0, {vocab_size})")
    unigrams = np.zeros(vocab_size, dtype=np.int64) if 1 in ns else None
    accumulators = {
        n: _NgramAccumulator(n, vocab_size, flush_every) for n in set(ns) if n >= 2
    }
    bos = None if bos_id is None else np.array([bos_id], dtype=np.int64)
    for doc in tqdm(docs, total=total, desc="Counting", mininterval=10.0):
        ids = np.asarray(doc, dtype=np.int64)
        if bos is not None:
            ids = np.concatenate([bos, ids])
        if unigrams is not None:
            unigrams += np.bincount(ids, minlength=vocab_size)
        for accumulator in accumulators.values():
            accumulator.add(ids)
    result: dict[int, Counts] = {n: a.result() for n, a in accumulators.items()}
    if unigrams is not None:
        result[1] = unigrams
    return {n: result[n] for n in sorted(result)}


def ngram_table(
    counts: Counts, n: int, vocab_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """The counted n-grams as ids [num_ngrams, n] and their counts [num_ngrams]."""
    if isinstance(counts, np.ndarray):  # n == 1
        ids = np.flatnonzero(counts)
        return ids[:, None], counts[ids]
    coo = sparse.coo_matrix(counts)
    rest = coo.col.astype(np.int64)
    columns = [coo.row.astype(np.int64)]
    for i in range(n - 2, -1, -1):
        columns.append(rest // vocab_size**i)
        rest = rest % vocab_size**i
    return np.stack(columns, axis=1), coo.data


def counts_filename(n: int) -> str:
    return f"{n}-grams.npy" if n == 1 else f"{n}-grams.npz"


def save_counts(counts: Counts, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp" + path.suffix)
    if isinstance(counts, np.ndarray):
        np.save(tmp, counts)
    else:
        sparse.save_npz(tmp, counts)
    tmp.rename(path)


def load_counts(path: Path) -> Counts:
    if path.suffix == ".npy":
        return np.load(path)
    return sparse.load_npz(path).tocsr()
