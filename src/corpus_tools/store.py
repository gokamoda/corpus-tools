"""Where samples, tokenized corpora and counts are saved, and making them on demand.

Layout under the cache directory (``Store(cache_dir)``, default
~/.cache/corpus-tools):

    hf/                                     Hugging Face cache (cache="corpus")
    <dataset>/<name>/<split>/               e.g. Skylion007--openwebtext/plain_text/train
        samples/hash_n10000.jsonl           hash sample (+ .meta.json)
        <source>/<tokenizer>/               source = all | hash_n10000 (| ..._head100)
            tokenized/                      tokens.bin, offsets.npy, meta.json
            counts/nobos/1-grams.npy, 2-grams.npz   counts (+ 1-grams.json, ...)
            counts/bos/...                  counts with BOS put before each document

A file that already exists is loaded instead of being made again.
"""

import itertools
import json
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from corpus_tools.corpus import Corpus, dataset_revision, open_rows
from corpus_tools.sample import sample_name, save_hash_sample

DEFAULT_CACHE_DIR = "~/.cache/corpus-tools"
ALL = "all"
COUNT_RULES = (
    "texts tokenized as they are with add_special_tokens=False; "
    "n-grams counted within each document; "
    "with bos, the BOS token id is put before each document"
)
CacheMode = Literal["none", "corpus", "tokens"]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def hub_model_id(name: str) -> str:
    """The canonical Hub id of a model, e.g. "gpt2" -> "openai-community/gpt2".

    A local path, or a name the Hub cannot be asked about (offline), is kept.
    """
    if Path(name).expanduser().exists():
        return name
    from huggingface_hub import HfApi

    try:
        return HfApi().model_info(name).id
    except Exception:  # offline, or not on the Hub
        return name


def _corpus_meta(corpus: Corpus) -> dict[str, Any]:
    return {
        "dataset": corpus.dataset,
        "name": corpus.name,
        "split": corpus.split,
        "text_field": corpus.text_field,
    }


def make_sample(
    corpus: Corpus,
    num_samples: int,
    output_path: Path,
    *,
    revision: str | None = None,
    max_chars: int | None = None,
    hf_cache_dir: Path | None = None,
) -> Path:
    """Read the whole split once and save its hash sample to ``output_path``.

    ``revision`` pins the dataset to a commit (default: the latest, which is
    recorded). Writes ``output_path`` (overwritten if it exists) and the meta
    file next to it (``.meta.json``).
    """
    revision = revision or dataset_revision(corpus)
    with open_rows(corpus, revision=revision, hf_cache_dir=hf_cache_dir) as (
        rows,
        total,
    ):
        num_rows = save_hash_sample(
            rows,
            output_path,
            num_samples=num_samples,
            key_field=corpus.text_field,
            max_chars=max_chars,
            total=total,
        )
    _write_json(
        output_path.with_suffix(".meta.json"),
        {
            **_corpus_meta(corpus),
            "dataset_revision": revision,
            "method": "hash",
            "num_samples": num_samples,
            "max_chars": max_chars,
            "num_rows": num_rows,
            "corpus_tools_version": version("corpus-tools"),
            "created": _now(),
        },
    )
    return output_path


class Store:
    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir).expanduser()

    @property
    def hf_cache_dir(self) -> Path:
        return self.cache_dir / "hf"

    def corpus_dir(self, corpus: Corpus) -> Path:
        return self.cache_dir / corpus.path

    # samples

    def sample_path(
        self, corpus: Corpus, num_samples: int, max_chars: int | None = None
    ) -> Path:
        name = sample_name(num_samples, max_chars)
        return self.corpus_dir(corpus) / "samples" / f"{name}.jsonl"

    def sample(
        self,
        corpus: Corpus,
        num_samples: int,
        *,
        max_chars: int | None = None,
        cache_corpus: bool = False,
        revision: str | None = None,
    ) -> Path:
        """Path of the hash sample, made first if it does not exist.

        ``revision`` pins the dataset to a commit (default: the latest). A
        saved sample of another revision is refused, as it has the same path.
        """
        path = self.sample_path(corpus, num_samples, max_chars)
        if path.exists():
            meta_path = path.with_suffix(".meta.json")
            saved = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            if revision is not None and saved.get("dataset_revision") != revision:
                raise ValueError(
                    f"{path} was made from revision {saved.get('dataset_revision')}, "
                    f"not {revision}"
                )
            return path
        return make_sample(
            corpus,
            num_samples,
            path,
            revision=revision,
            max_chars=max_chars,
            hf_cache_dir=self.hf_cache_dir if cache_corpus else None,
        )

    # counts

    def source_dir(
        self, corpus: Corpus, source: str, max_documents: int | None = None
    ) -> Path:
        head = f"_head{max_documents}" if max_documents is not None else ""
        return self.corpus_dir(corpus) / f"{source}{head}"

    def tokenizer_dir(
        self,
        corpus: Corpus,
        tokenizer_name: str,
        source: str = ALL,
        max_documents: int | None = None,
    ) -> Path:
        return self.source_dir(corpus, source, max_documents) / tokenizer_name.replace(
            "/", "--"
        )

    def counts(
        self,
        corpus: Corpus,
        tokenizer: Any,
        ns: Iterable[int],
        *,
        source: str = ALL,
        cache: CacheMode = "none",
        bos: bool = False,
        max_documents: int | None = None,
        tokenizer_name: str | None = None,
        batch_size: int = 1000,
        flush_every: int = 200_000_000,
    ) -> dict[int, Any]:
        """N-gram counts for each n in ``ns`` (see ``count.count_ngrams``).

        ``tokenizer`` is a Hugging Face tokenizer or its name. ``source`` is
        ``"all"`` (the whole split) or the name of a saved hash sample, e.g.
        ``"hash_n10000"``. ``cache`` is how the corpus is read when counts
        have to be made:

        - "none": stream the corpus and count while reading;
        - "corpus": read it through the Hugging Face cache under the cache directory;
        - "tokens": tokenize it once into ``tokenized/`` and count from there.

        Results are saved under ``tokenizer_name`` (default: the canonical Hub
        id of the tokenizer's ``name_or_path``, so "gpt2" and
        "openai-community/gpt2" share their results). Give another name for a tokenizer changed after
        loading, so that its counts are not mixed with the original's.
        """
        try:
            from corpus_tools.count import (
                count_ngrams,
                counts_filename,
                load_counts,
                save_counts,
            )
        except ImportError as error:
            raise ImportError(
                "Counting needs the extra: corpus-tools[count]"
            ) from error
        from corpus_tools.tokenize import TokenizedCorpus, tokenize

        if cache not in ("none", "corpus", "tokens"):
            raise ValueError(f"Unknown cache mode {cache!r}")
        if isinstance(tokenizer, str):
            from transformers import AutoTokenizer

            tokenizer = cast(Any, AutoTokenizer.from_pretrained(tokenizer))
        tokenizer_name = tokenizer_name or hub_model_id(tokenizer.name_or_path)
        vocab_size = len(tokenizer)
        bos_id = None
        if bos:
            bos_id = tokenizer.bos_token_id
            if bos_id is None:
                raise ValueError(f"{tokenizer_name} has no BOS token")

        ns = sorted(set(ns))
        tokenizer_dir = self.tokenizer_dir(
            corpus, tokenizer_name, source, max_documents
        )
        counts_dir = tokenizer_dir / "counts" / ("bos" if bos else "nobos")
        missing = [n for n in ns if not (counts_dir / counts_filename(n)).exists()]
        for n in set(ns) - set(missing):
            saved = json.loads(
                (counts_dir / counts_filename(n)).with_suffix(".json").read_text()
            )
            if saved["vocab_size"] != vocab_size:
                raise ValueError(
                    f"{counts_dir} was counted with a vocabulary of "
                    f"{saved['vocab_size']}, but the tokenizer has {vocab_size}; "
                    "give another tokenizer_name"
                )
        if missing:
            meta = {
                **_corpus_meta(corpus),
                **self._source_meta(corpus, source),
                "max_documents": max_documents,
                "tokenizer": tokenizer.name_or_path,
                "tokenizer_name": tokenizer_name,
                "vocab_size": vocab_size,
                "transformers_version": version("transformers"),
                "rules": COUNT_RULES,
            }
            stats = {"num_documents": 0, "num_tokens": 0}
            if cache == "tokens":
                tokenized_dir = tokenizer_dir / "tokenized"
                if not TokenizedCorpus.exists(tokenized_dir):
                    with self._open_texts(corpus, source, meta, max_documents) as (
                        texts,
                        _,
                    ):
                        TokenizedCorpus.write(
                            tokenize(texts, tokenizer, batch_size=batch_size),
                            tokenized_dir,
                            vocab_size=vocab_size,
                            meta={**meta, "created": _now()},
                        )
                docs = TokenizedCorpus(tokenized_dir)
                if docs.meta["vocab_size"] != vocab_size:
                    raise ValueError(f"{tokenized_dir} has another vocabulary size")
                # the counts are of the data that was tokenized, not of today's
                meta["dataset_revision"] = docs.meta.get("dataset_revision")
                computed = count_ngrams(
                    _tally(docs, stats),
                    missing,
                    vocab_size,
                    bos_id=bos_id,
                    flush_every=flush_every,
                    total=len(docs),
                )
            else:
                with self._open_texts(
                    corpus, source, meta, max_documents, cache_corpus=cache == "corpus"
                ) as (texts, total):
                    computed = count_ngrams(
                        _tally(
                            tokenize(texts, tokenizer, batch_size=batch_size), stats
                        ),
                        missing,
                        vocab_size,
                        bos_id=bos_id,
                        flush_every=flush_every,
                        total=total,
                    )
            for n, counts in computed.items():
                path = counts_dir / counts_filename(n)
                save_counts(counts, path)
                _write_json(
                    path.with_suffix(".json"),
                    {
                        **meta,
                        "n": n,
                        "bos": bos,
                        "bos_id": bos_id,
                        **stats,
                        "num_ngrams": int(
                            counts.sum()
                            if isinstance(counts, np.ndarray)
                            else counts.data.sum()
                        ),
                        "num_distinct_ngrams": int(
                            np.count_nonzero(counts)
                            if isinstance(counts, np.ndarray)
                            else counts.nnz
                        ),
                        "corpus_tools_version": version("corpus-tools"),
                        "created": _now(),
                    },
                )
        return {n: load_counts(counts_dir / counts_filename(n)) for n in ns}

    # reading

    def _source_meta(self, corpus: Corpus, source: str) -> dict[str, Any]:
        if source == ALL:
            revision = dataset_revision(corpus)
            return {"source": ALL, "dataset_revision": revision}
        meta_path = self.corpus_dir(corpus) / "samples" / f"{source}.meta.json"
        sample_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return {
            "source": source,
            "dataset_revision": sample_meta.get("dataset_revision"),
        }

    @contextmanager
    def _open_rows(
        self, corpus: Corpus, revision: str | None, cache_corpus: bool
    ) -> Iterator[tuple[Iterator[dict[str, Any]], int | None]]:
        with open_rows(
            corpus,
            revision=revision,
            hf_cache_dir=self.hf_cache_dir if cache_corpus else None,
        ) as opened:
            yield opened

    @contextmanager
    def _open_texts(
        self,
        corpus: Corpus,
        source: str,
        meta: dict[str, Any],
        max_documents: int | None,
        *,
        cache_corpus: bool = False,
    ) -> Iterator[tuple[Iterator[str], int | None]]:
        if source == ALL:
            with self._open_rows(corpus, meta["dataset_revision"], cache_corpus) as (
                rows,
                total,
            ):
                texts = (row[corpus.text_field] for row in rows)
                yield _head(texts, total, max_documents)
            return
        path = self.corpus_dir(corpus) / "samples" / f"{source}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; make the sample first")
        with path.open() as sample_file:
            total = sum(1 for _ in sample_file)
            sample_file.seek(0)
            texts = (json.loads(line)[corpus.text_field] for line in sample_file)
            yield _head(texts, total, max_documents)


def _head(
    texts: Iterator[str], total: int | None, max_documents: int | None
) -> tuple[Iterator[str], int | None]:
    if max_documents is None:
        return texts, total
    total = max_documents if total is None else min(total, max_documents)
    return itertools.islice(texts, max_documents), total


def _tally(docs: Iterable[np.ndarray], stats: dict[str, int]) -> Iterator[np.ndarray]:
    for doc in docs:
        stats["num_documents"] += 1
        stats["num_tokens"] += len(doc)
        yield doc
