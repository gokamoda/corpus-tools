import json
import random
from pathlib import Path

import pytest

from corpus_tools.sample import (
    hash_fold,
    hash_key,
    hash_sample,
    iter_hash_fold,
    load_hash_sample,
    save_hash_sample,
)

TESTDATA = Path(__file__).parent / "testdata"


def make_rows(n: int) -> list[dict]:
    return [{"id": str(i), "text": f"document {i} " + "x" * i} for i in range(n)]


@pytest.mark.parametrize(
    ("num_samples", "max_chars", "expected"),
    [
        (100, None, "expected_hash_n100.jsonl"),
        (100, 20, "expected_hash_n100_chars20.jsonl"),
        (1000, None, "expected_hash_n1000.jsonl"),
    ],
)
def test_same_bytes_as_inseg_attention_and_lm_detokenization(
    tmp_path, num_samples, max_chars, expected
):
    # The expected files were written by inseg-attention's save_hash_sample
    # (src/inseg_attention/data/download/hash_sample.py); lm-detokenization's
    # (src/lm_detokenization/data/hash_sample.py) wrote the same bytes for the
    # cases without max_chars. rows.jsonl has non-ASCII text, escapes, empty
    # texts, duplicate texts and extra fields.
    rows = [json.loads(line) for line in (TESTDATA / "rows.jsonl").open()]
    path = tmp_path / "sample.jsonl"
    save_hash_sample(rows, path, num_samples=num_samples, max_chars=max_chars)
    assert path.read_bytes() == (TESTDATA / expected).read_bytes()


def test_smallest_keys_in_order():
    rows = make_rows(100)
    sample = hash_sample(rows, num_samples=10)
    expected = sorted(hash_key(row["text"]) for row in rows)[:10]
    assert [row["hash"] for row in sample] == expected


def test_sample_is_nested():
    rows = make_rows(1000)
    assert hash_sample(rows, num_samples=100)[:10] == hash_sample(rows, num_samples=10)


def test_independent_of_row_order():
    rows = make_rows(1000)
    shuffled = rows.copy()
    random.Random(0).shuffle(shuffled)
    assert hash_sample(rows, num_samples=50) == hash_sample(shuffled, num_samples=50)


def test_duplicates_kept_once():
    sample = hash_sample(make_rows(100) * 3, num_samples=100)
    assert len(sample) == 100
    assert len({row["text"] for row in sample}) == 100


def test_fewer_rows_than_requested():
    assert len(hash_sample(make_rows(5), num_samples=10)) == 5


def test_max_chars_cuts_text_but_hashes_the_full_text():
    rows = make_rows(200)
    full = hash_sample(rows, num_samples=20)
    cut = hash_sample(rows, num_samples=20, max_chars=12)
    assert [row["hash"] for row in cut] == [row["hash"] for row in full]
    assert all(row["text"] == f["text"][:12] for row, f in zip(cut, full))


def test_min_chars_keeps_the_long_rows_of_the_sample_in_order():
    rows = make_rows(200)  # the text of row i has 10 + i characters or more
    full = hash_sample(rows, num_samples=200)
    long = hash_sample(rows, num_samples=30, min_chars=100)
    assert long == [row for row in full if len(row["text"]) >= 100][:30]
    with pytest.raises(ValueError):
        hash_sample(rows, num_samples=3, min_chars=0)


def test_save_and_load(tmp_path):
    path = tmp_path / "sample.jsonl"
    rows = make_rows(100)
    assert save_hash_sample(rows, path, num_samples=20) == 20
    assert load_hash_sample(path) == hash_sample(rows, num_samples=20)
    assert load_hash_sample(path, num_samples=5) == hash_sample(rows, num_samples=5)
    with pytest.raises(ValueError):
        load_hash_sample(path, num_samples=21)


def test_folds_are_disjoint_and_stable_when_the_sample_grows(tmp_path):
    rows = make_rows(1000)
    small_path, large_path = tmp_path / "small.jsonl", tmp_path / "large.jsonl"
    save_hash_sample(rows, small_path, num_samples=100)
    save_hash_sample(rows, large_path, num_samples=500)
    folds = [list(iter_hash_fold(small_path, fold=k, num_folds=5)) for k in range(5)]
    assert sum(len(fold) for fold in folds) == 100
    assert len({row["id"] for fold in folds for row in fold}) == 100
    for k, fold in enumerate(folds):
        large_fold = list(iter_hash_fold(large_path, fold=k, num_folds=5))
        assert large_fold[: len(fold)] == fold
        assert all(hash_fold(row["hash"], 5) == k for row in fold)


def test_invalid_fold():
    with pytest.raises(ValueError):
        next(iter_hash_fold(Path("unused.jsonl"), fold=5, num_folds=5))
