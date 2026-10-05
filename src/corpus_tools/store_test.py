import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from corpus_tools import store as store_module
from corpus_tools.corpus import Corpus
from corpus_tools.count import count_ngrams
from corpus_tools.sample import hash_sample
from corpus_tools.store import Store
from corpus_tools.tokenize_test import CharTokenizer

CORPUS = Corpus(dataset="someone/letters", name="default", split="train")
TEXTS = ["abcab", "ba", "a", "", "zzza", "hello", "world", "abcab"] * 5


@pytest.fixture
def reads(monkeypatch):
    """Serve TEXTS as the corpus, and record how it was read."""
    calls = []

    @contextmanager
    def fake_open_rows(corpus, *, revision=None, hf_cache_dir=None):
        calls.append({"revision": revision, "hf_cache_dir": hf_cache_dir})
        yield iter([{"text": t, "i": i} for i, t in enumerate(TEXTS)]), len(TEXTS)

    monkeypatch.setattr(store_module, "open_rows", fake_open_rows)
    monkeypatch.setattr(store_module, "dataset_revision", lambda corpus: "abc123")
    monkeypatch.setattr(store_module, "hub_model_id", lambda name: name)
    return calls


def expected(ns, texts=TEXTS, bos_id=None):
    docs = [np.array([ord(c) - ord("a") for c in t]) for t in texts]
    return count_ngrams(docs, ns, 26, bos_id=bos_id)


def assert_same(counts, expected_counts):
    assert list(counts) == list(expected_counts)
    for n, value in counts.items():
        if n == 1:
            assert (value == expected_counts[n]).all()
        else:
            assert (value != expected_counts[n]).nnz == 0


def test_cache_dir():
    assert Store().cache_dir == Path.home() / ".cache/corpus-tools"
    assert Store("~/elsewhere").cache_dir == Path.home() / "elsewhere"


def test_sample_is_made_once(tmp_path, reads):
    store = Store(tmp_path)
    path = store.sample(CORPUS, 3)
    assert path == tmp_path / "someone--letters/default/train/samples/hash_n3.jsonl"
    rows = [json.loads(line) for line in path.open()]
    assert rows == hash_sample(
        [{"text": t, "i": i} for i, t in enumerate(TEXTS)], num_samples=3
    )
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    assert meta["dataset_revision"] == "abc123"
    assert meta["num_rows"] == 3
    assert store.sample(CORPUS, 3) == path
    assert len(reads) == 1


@pytest.mark.parametrize("cache", ["none", "corpus", "tokens"])
@pytest.mark.parametrize("bos", [False, True])
def test_counts_are_the_same_for_every_cache(tmp_path, reads, cache, bos):
    store = Store(tmp_path)
    counts = store.counts(CORPUS, CharTokenizer(), [2, 1, 3], cache=cache, bos=bos)
    assert_same(counts, expected([1, 2, 3], bos_id=25 if bos else None))
    assert reads[0]["revision"] == "abc123"
    assert (reads[0]["hf_cache_dir"] is not None) == (cache == "corpus")

    counts_dir = tmp_path / "someone--letters/default/train/all/char/counts"
    counts_dir = counts_dir / ("bos" if bos else "nobos")
    assert sorted(p.name for p in counts_dir.iterdir()) == [
        "1-grams.json", "1-grams.npy", "2-grams.json", "2-grams.npz", "3-grams.json", "3-grams.npz",
    ]  # fmt: skip
    meta = json.loads((counts_dir / "1-grams.json").read_text())
    assert meta["num_documents"] == len(TEXTS)
    assert meta["num_tokens"] == sum(len(t) for t in TEXTS)
    assert meta["bos"] is bos

    # made once; loaded afterwards
    store.counts(CORPUS, CharTokenizer(), [1, 2], cache=cache, bos=bos)
    assert len(reads) == 1


def test_token_cache_is_shared_by_bos_and_new_n(tmp_path, reads):
    store = Store(tmp_path)
    store.counts(CORPUS, CharTokenizer(), [1], cache="tokens")
    counts = store.counts(CORPUS, CharTokenizer(), [2], cache="tokens", bos=True)
    assert_same(counts, expected([2], bos_id=25))
    assert len(reads) == 1


def test_counts_of_a_sample(tmp_path, reads):
    store = Store(tmp_path)
    store.sample(CORPUS, 4)
    counts = store.counts(CORPUS, CharTokenizer(), [1, 2], source="hash_n4")
    sample = hash_sample([{"text": t} for t in TEXTS], num_samples=4)
    assert_same(counts, expected([1, 2], texts=[row["text"] for row in sample]))
    assert len(reads) == 1  # only to make the sample
    meta = json.loads(
        (
            tmp_path
            / "someone--letters/default/train/hash_n4/char/counts/nobos/1-grams.json"
        ).read_text()
    )
    assert meta["source"] == "hash_n4"
    assert meta["dataset_revision"] == "abc123"


def test_max_documents_is_saved_separately(tmp_path, reads):
    store = Store(tmp_path)
    counts = store.counts(CORPUS, CharTokenizer(), [1], max_documents=3)
    assert_same(counts, expected([1], texts=TEXTS[:3]))
    assert (tmp_path / "someone--letters/default/train/all_head3/char").is_dir()


def test_missing_sample(tmp_path, reads):
    with pytest.raises(FileNotFoundError):
        Store(tmp_path).counts(CORPUS, CharTokenizer(), [1], source="hash_n4")


def test_bos_needs_a_bos_token(tmp_path, reads):
    tokenizer = CharTokenizer()
    tokenizer.bos_token_id = None
    with pytest.raises(ValueError):
        Store(tmp_path).counts(CORPUS, tokenizer, [1], bos=True)


def test_tokenizer_name_sets_the_directory(tmp_path, reads):
    store = Store(tmp_path)
    store.counts(CORPUS, CharTokenizer(), [1], tokenizer_name="org/char-v2")
    corpus_dir = tmp_path / "someone--letters/default/train/all"
    assert (corpus_dir / "org--char-v2/counts/nobos/1-grams.npy").exists()
    assert not (corpus_dir / "char").exists()
    meta = json.loads(
        (corpus_dir / "org--char-v2/counts/nobos/1-grams.json").read_text()
    )
    assert meta["tokenizer"] == "char"
    assert meta["tokenizer_name"] == "org/char-v2"


def test_saved_counts_of_another_vocabulary_are_refused(tmp_path, reads):
    store = Store(tmp_path)
    store.counts(CORPUS, CharTokenizer(), [1])

    class LargerTokenizer(CharTokenizer):
        def __len__(self):
            return 27

    with pytest.raises(ValueError, match="tokenizer_name"):
        store.counts(CORPUS, LargerTokenizer(), [1])
    assert (
        len(store.counts(CORPUS, LargerTokenizer(), [1], tokenizer_name="c27")[1]) == 27
    )


def test_make_sample_to_a_given_file_at_a_revision(tmp_path, reads):
    from corpus_tools.store import make_sample

    path = make_sample(CORPUS, 3, tmp_path / "out" / "s.jsonl", revision="r1")
    assert path == tmp_path / "out" / "s.jsonl"
    assert [json.loads(line) for line in path.open()] == hash_sample(
        [{"text": t, "i": i} for i, t in enumerate(TEXTS)], num_samples=3
    )
    assert (
        json.loads((tmp_path / "out" / "s.meta.json").read_text())["dataset_revision"]
        == "r1"
    )
    assert reads[-1]["revision"] == "r1"


def test_sample_of_another_revision_is_refused(tmp_path, reads):
    store = Store(tmp_path)
    store.sample(CORPUS, 3, revision="r1")
    assert store.sample(CORPUS, 3, revision="r1") == store.sample_path(CORPUS, 3)
    with pytest.raises(ValueError, match="r1"):
        store.sample(CORPUS, 3, revision="r2")
    assert len(reads) == 1


def test_cli_sample_with_output_and_revision(tmp_path, reads, monkeypatch):
    from corpus_tools import cli

    monkeypatch.setattr(cli, "preset", lambda *args, **kwargs: CORPUS)
    output = tmp_path / "data" / "sample.jsonl"
    cli.main(
        ["sample", "--corpus", "openwebtext", "--num-samples", "3",
         "--revision", "r9", "--output", str(output), "--cache-dir", str(tmp_path)]
    )  # fmt: skip
    assert len(output.read_text().splitlines()) == 3
    assert reads[-1] == {"revision": "r9", "hf_cache_dir": None}
