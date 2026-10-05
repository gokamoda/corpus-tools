from collections import Counter

import numpy as np
import pytest
from scipy import sparse

from corpus_tools.count import (
    count_ngrams,
    counts_filename,
    load_counts,
    ngram_table,
    save_counts,
)

DOCS = [[0, 1, 2, 0, 1], [1, 0], [0], [], [25, 25, 25, 0], [3, 1, 4, 1, 5, 9, 2, 6]] * 7
VOCAB_SIZE = 26


def expected_counts(docs, n, bos_id=None):
    counter = Counter()
    for doc in docs:
        ids = ([bos_id] if bos_id is not None else []) + doc
        counter.update(tuple(ids[i : i + n]) for i in range(len(ids) - n + 1))
    return dict(counter)


def as_dict(counts, n):
    ids, values = ngram_table(counts, n, VOCAB_SIZE)
    return {tuple(int(i) for i in row): int(c) for row, c in zip(ids, values)}


@pytest.mark.parametrize("bos_id", [None, 24])
@pytest.mark.parametrize("flush_every", [1, 5, 10**9])
def test_counts_match_counter_within_documents(bos_id, flush_every):
    docs = [np.array(doc) for doc in DOCS]
    counts = count_ngrams(
        docs, [1, 2, 3, 4], VOCAB_SIZE, bos_id=bos_id, flush_every=flush_every
    )
    assert list(counts) == [1, 2, 3, 4]
    for n in (1, 2, 3, 4):
        assert as_dict(counts[n], n) == expected_counts(DOCS, n, bos_id)


def test_bigram_matrix_is_prefix_by_suffix():
    # As lm-detokenization's Counts.bigrams: row = prefix, column = suffix.
    counts = count_ngrams([np.array([3, 5, 3, 5])], [1, 2], VOCAB_SIZE)
    assert counts[1].shape == (VOCAB_SIZE,)
    assert counts[1].dtype == np.int64
    bigrams = counts[2]
    assert isinstance(bigrams, sparse.csr_matrix)
    assert bigrams.shape == (VOCAB_SIZE, VOCAB_SIZE)
    assert bigrams.dtype == np.int64
    assert bigrams[3, 5] == 2
    assert bigrams[5, 3] == 1
    assert bigrams.nnz == 2


def test_uint16_input_does_not_overflow():
    vocab_size = 60_000
    docs = [np.array([59_999, 59_998, 59_999], dtype=np.uint16)]
    counts = count_ngrams(docs, [2, 3], vocab_size)
    assert counts[2][59_999, 59_998] == 1
    ids, values = ngram_table(counts[3], 3, vocab_size)
    assert ids.tolist() == [[59_999, 59_998, 59_999]]
    assert values.tolist() == [1]


def test_keys_that_do_not_fit_int64_are_refused():
    with pytest.raises(ValueError):
        count_ngrams([np.array([0, 1, 2, 3])], [4], 100_000)


def test_save_and_load(tmp_path):
    counts = count_ngrams([np.array(doc) for doc in DOCS], [1, 2, 3], VOCAB_SIZE)
    for n, value in counts.items():
        path = tmp_path / counts_filename(n)
        save_counts(value, path)
        loaded = load_counts(path)
        if n == 1:
            assert (loaded == value).all()
        else:
            assert (loaded != value).nnz == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["n1.npy", "n2.npz", "n3.npz"]
