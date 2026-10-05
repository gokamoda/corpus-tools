import numpy as np

from corpus_tools.tokenize import TokenizedCorpus, token_dtype, tokenize


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


def test_write_and_read(tmp_path):
    docs = [np.array(d) for d in [[0, 1, 2], [], [25, 25], [1]]]
    directory = tmp_path / "tokenized"
    corpus = TokenizedCorpus.write(docs, directory, vocab_size=26, meta={"x": 1})
    assert TokenizedCorpus.exists(directory)
    assert not (tmp_path / "tokenized.tmp").exists()
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


def test_empty_corpus(tmp_path):
    corpus = TokenizedCorpus.write([], tmp_path / "t", vocab_size=26)
    assert len(corpus) == 0
    assert list(corpus) == []
