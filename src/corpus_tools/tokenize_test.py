import json

import numpy as np
import pytest

from corpus_tools.tokenize import (
    TokenizedCorpus,
    TokenizedShard,
    shard_name,
    token_dtype,
    tokenize,
)


class CharTokenizer:
    """Tokenizes into character codes 0..25 (a..z), like a HF tokenizer call."""

    name_or_path = "char"
    bos_token_id: int | None = 25

    def __len__(self):
        return 26

    def __call__(self, texts, add_special_tokens=True):
        assert add_special_tokens is False
        return {"input_ids": [[ord(c) - ord("a") for c in t] for t in texts]}


def test_token_dtype():
    assert token_dtype(50_257) == np.uint16
    assert token_dtype(2**16) == np.uint16
    assert token_dtype(2**16 + 1) == np.uint32


def test_tokenize_in_batches():
    texts = ["abc", "", "zz", "b"] * 3
    docs = list(tokenize(texts, CharTokenizer(), batch_size=5))
    assert [d.tolist() for d in docs] == [[0, 1, 2], [], [25, 25], [1]] * 3


def test_write_and_read_a_shard(tmp_path):
    docs = [np.array(d) for d in [[0, 1, 2], [], [25, 25], [1]]]
    directory = tmp_path / "shard-00000"
    corpus = TokenizedShard.write(docs, directory, vocab_size=26, meta={"x": 1})
    assert directory.is_dir()
    assert not (tmp_path / "shard-00000.tmp").exists()
    assert len(corpus) == 4
    assert [d.tolist() for d in corpus] == [d.tolist() for d in docs]
    assert corpus[2].tolist() == [25, 25]
    assert corpus.tokens.dtype == np.uint16
    assert corpus.meta == {
        "x": 1,
        "vocab_size": 26,
        "dtype": "uint16",
        "num_documents": 4,
        "num_tokens": 6,
    }


def test_empty_shard(tmp_path):
    corpus = TokenizedShard.write([], tmp_path / "t", vocab_size=26)
    assert len(corpus) == 0
    assert list(corpus) == []


def test_interrupted_shard_is_not_there(tmp_path):
    def docs():
        yield np.array([0, 1])
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        TokenizedShard.write(docs(), tmp_path / "s", vocab_size=26)
    assert not (tmp_path / "s").exists()
    TokenizedShard.write([np.array([2])], tmp_path / "s", vocab_size=26)
    assert [d.tolist() for d in TokenizedShard(tmp_path / "s")] == [[2]]


def test_corpus_of_shards(tmp_path):
    shards = [[[0, 1], [2]], [], [[3], [4, 5], [6]]]
    (tmp_path / "meta.json").write_text(json.dumps({"num_shards": 3}))
    for i, shard in enumerate(shards[:2]):
        TokenizedShard.write(
            [np.array(d) for d in shard], tmp_path / shard_name(i), vocab_size=26
        )
    assert not TokenizedCorpus.is_complete(tmp_path)
    TokenizedShard.write(
        [np.array(d) for d in shards[2]], tmp_path / shard_name(2), vocab_size=26
    )
    assert TokenizedCorpus.is_complete(tmp_path)

    corpus = TokenizedCorpus(tmp_path)
    flat = [d for shard in shards for d in shard]
    assert len(corpus) == 5
    assert [d.tolist() for d in corpus] == flat
    assert [corpus[i].tolist() for i in range(5)] == flat
    with pytest.raises(IndexError):
        corpus[5]
