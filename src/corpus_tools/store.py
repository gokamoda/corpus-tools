"""Where samples, tokenized corpora and counts are saved, and making them on demand.

Results go under the output directory (``Store(output_dir)``, always given),
and what can be made again under the cache directory (``cache_dir``, default
~/.cache/corpus-tools):

    <output_dir>/
        <dataset>/<name>/<split>/           e.g. Skylion007--openwebtext/plain_text/train
            samples/hash_n10000.jsonl       hash sample (+ .meta.json)
            <source>/<tokenizer>/           source = all | hash_n10000 (| ..._head100)
                counts/nobos/1-grams.npy, 2-grams.npz   counts (+ 1-grams.json, ...)
                counts/bos/...              counts with BOS put before each document
    <cache_dir>/
        hf/                                 Hugging Face cache (cache="corpus")
        <dataset>/<name>/<split>/<source>/<tokenizer>/
            tokenized/                      meta.json, shard-00000/ (tokens.bin, offsets.npy, meta.json), ...
            counts/nobos/shards/            counts of each shard, until they are added up

A file that already exists is loaded instead of being made again. Files are
written under a temporary name and renamed last, so a file (or shard
directory) that exists is complete.
"""

import atexit
import itertools
import json
import multiprocessing
import os
import shutil
import sys
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from tqdm import tqdm

from corpus_tools.corpus import (
    Corpus,
    dataset_revision,
    open_rows,
    split_rows,
    stream_shards,
)
from corpus_tools.sample import sample_name, save_hash_sample
from corpus_tools.tokenize import shard_name

DEFAULT_CACHE_DIR = "~/.cache/corpus-tools"
ALL = "all"
COUNT_RULES = (
    "texts tokenized as they are with add_special_tokens=False; "
    "n-grams counted within each document; "
    "with bos, the BOS token id is put before each document"
)
CacheMode = Literal["none", "corpus", "tokens"]
SampleCacheMode = Literal["none", "corpus"]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.rename(path)


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
    file next to it (``.meta.json``). The sample is renamed to
    ``output_path`` last, so it exists only when both are complete.
    """
    revision = revision or dataset_revision(corpus)
    output_path.unlink(missing_ok=True)  # not to be left with the new meta
    partial_path = output_path.with_name(output_path.name + ".partial")
    with open_rows(corpus, revision=revision, hf_cache_dir=hf_cache_dir) as (
        rows,
        total,
    ):
        num_rows = save_hash_sample(
            rows,
            partial_path,
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
    partial_path.rename(output_path)
    return output_path


class Store:
    def __init__(
        self, output_dir: str | Path, cache_dir: str | Path = DEFAULT_CACHE_DIR
    ):
        self.output_dir = Path(output_dir).expanduser()
        self.cache_dir = Path(cache_dir).expanduser()

    @property
    def hf_cache_dir(self) -> Path:
        return self.cache_dir / "hf"

    def corpus_dir(self, corpus: Corpus) -> Path:
        """Directory of the results of this corpus."""
        return self.output_dir / corpus.path

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
        cache: SampleCacheMode = "none",
        revision: str | None = None,
    ) -> Path:
        """Path of the hash sample, made first if it does not exist.

        ``cache`` is how the corpus is read: "none" (streamed) or "corpus"
        (through the Hugging Face cache under the cache directory).
        ``revision`` pins the dataset to a commit (default: the latest). A
        saved sample of another revision is refused, as it has the same path.
        """
        if cache not in ("none", "corpus"):
            raise ValueError(f"Unknown cache mode {cache!r} for a sample")
        path = self.sample_path(corpus, num_samples, max_chars)
        if path.exists():
            meta_path = path.with_suffix(".meta.json")
            saved = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            _check_revision(path, saved, revision)
            return path
        return make_sample(
            corpus,
            num_samples,
            path,
            revision=revision,
            max_chars=max_chars,
            hf_cache_dir=self.hf_cache_dir if cache == "corpus" else None,
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
        """Directory of the counts of this tokenizer (``counts/`` in it)."""
        return self.source_dir(corpus, source, max_documents) / tokenizer_name.replace(
            "/", "--"
        )

    def tokenizer_cache_dir(
        self,
        corpus: Corpus,
        tokenizer_name: str,
        source: str = ALL,
        max_documents: int | None = None,
    ) -> Path:
        """Directory of the token cache and shard counts of this tokenizer."""
        relative = self.tokenizer_dir(
            corpus, tokenizer_name, source, max_documents
        ).relative_to(self.output_dir)
        return self.cache_dir / relative

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
        revision: str | None = None,
        cpus: int | None = None,
        num_workers: int | None = None,
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

        ``revision`` pins the dataset to a commit (default: the latest, or
        the one of the sample). Saved counts, a token cache or a sample of
        another revision are refused, as they have the same paths.

        The corpus is read in shards (see ``_num_shards``). ``cpus`` is how
        many threads may run at once, in all; they are given to processes
        first (one per shard, at most ``num_workers``), and the rest to the
        tokenizer of each process (see ``_workers``). Without either, the
        shards are counted in this process, and the tokenizer uses its own
        number of threads. Each process holds the counts of its shard in
        memory. The counts of each shard are saved in the cache directory
        (``counts/<bos>/shards/``) and removed once added up, so that a stopped
        run is continued from the shards not yet counted.

        Results are saved under ``tokenizer_name`` (default: the canonical Hub
        id of the tokenizer's ``name_or_path``, so "gpt2" and
        "openai-community/gpt2" share their results). Give another name for a tokenizer changed after
        loading, so that its counts are not mixed with the original's.
        """
        try:
            from corpus_tools.count import counts_filename, load_counts, save_counts
        except ImportError as error:
            raise ImportError(
                "Counting needs the extra: corpus-tools[count]"
            ) from error

        if cache not in ("none", "corpus", "tokens"):
            raise ValueError(f"Unknown cache mode {cache!r}")
        if cpus is not None and cpus < 1:
            raise ValueError("cpus must be at least 1")
        if num_workers is not None and num_workers < 1:
            raise ValueError("num_workers must be at least 1")
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
        bos_dir = "bos" if bos else "nobos"
        counts_dir = tokenizer_dir / "counts" / bos_dir
        tokenizer_cache_dir = self.tokenizer_cache_dir(
            corpus, tokenizer_name, source, max_documents
        )
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
            _check_revision(counts_dir / counts_filename(n), saved, revision)
        if missing:
            meta: dict[str, Any] = {
                **_corpus_meta(corpus),
                **self._source_meta(corpus, source, revision),
                "max_documents": max_documents,
                "tokenizer": tokenizer.name_or_path,
                "tokenizer_name": tokenizer_name,
                "vocab_size": vocab_size,
                "transformers_version": version("transformers"),
                "rules": COUNT_RULES,
            }
            texts = _Texts(
                corpus=corpus,
                revision=meta["dataset_revision"],
                hf_cache_dir=self.hf_cache_dir if cache == "corpus" else None,
                sample_path=None
                if source == ALL
                else self.corpus_dir(corpus) / "samples" / f"{source}.jsonl",
                max_documents=max_documents,
                sample_shard_size=SAMPLE_SHARD_SIZE,
            )
            tokenized_dir = None
            if cache == "tokens":
                tokenized_dir = tokenizer_cache_dir / "tokenized"
                texts, num_shards = _prepare_tokenized(
                    tokenized_dir, texts, meta, revision
                )
                # the counts are of the data that was tokenized, not of today's
                meta["dataset_revision"] = texts.revision
            else:
                num_shards = _num_shards(texts)
            shards_dir = tokenizer_cache_dir / "counts" / bos_dir / "shards"
            _prepare_shard_counts(
                shards_dir,
                {
                    "ns": missing,
                    "num_shards": num_shards,
                    "shard_by": _shard_by(texts),
                    "dataset_revision": texts.revision,
                    "vocab_size": vocab_size,
                },
            )
            tasks = [
                _Task(
                    texts=texts,
                    index=index,
                    num_shards=num_shards,
                    ns=tuple(missing),
                    vocab_size=vocab_size,
                    bos_id=bos_id,
                    batch_size=batch_size,
                    flush_every=flush_every,
                    counts_dir=shards_dir / shard_name(index),
                    tokenized_dir=None
                    if tokenized_dir is None
                    else tokenized_dir / shard_name(index),
                    tokenized_meta={**meta, "shard": index, "num_shards": num_shards},
                )
                for index in range(num_shards)
                if not (shards_dir / shard_name(index)).is_dir()
            ]
            processes, threads = _workers(len(tasks), cpus, num_workers)
            _run(tasks, tokenizer, num_shards, processes, threads)
            computed, stats = _add_shard_counts(shards_dir, missing, num_shards)
            for n, counts in computed.items():
                path = counts_dir / counts_filename(n)
                # the meta first: the counts file is the sign that both are complete
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
                save_counts(counts, path)
            shutil.rmtree(shards_dir)
        return {n: load_counts(counts_dir / counts_filename(n)) for n in ns}

    def _source_meta(
        self, corpus: Corpus, source: str, revision: str | None
    ) -> dict[str, Any]:
        if source == ALL:
            revision = revision or dataset_revision(corpus)
            return {"source": ALL, "dataset_revision": revision}
        sample_path = self.corpus_dir(corpus) / "samples" / f"{source}.jsonl"
        meta_path = sample_path.with_suffix(".meta.json")
        sample_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        _check_revision(sample_path, sample_meta, revision)
        return {
            "source": source,
            "dataset_revision": sample_meta.get("dataset_revision"),
        }


def _check_revision(path: Path, meta: dict[str, Any], revision: str | None) -> None:
    """Refuse what was made from another revision than the one asked for."""
    if revision is not None and meta.get("dataset_revision") != revision:
        raise ValueError(
            f"{path} was made from revision {meta.get('dataset_revision')}, "
            f"not {revision}"
        )


# reading in shards

SAMPLE_SHARD_SIZE = 10_000  # documents of a hash sample in a shard
CACHED_SHARD_SIZE = 100_000  # documents in a shard of a split in the HF cache


@dataclass(frozen=True)
class _Texts:
    """Where the texts to count are read from (sent to worker processes)."""

    corpus: Corpus
    revision: str | None
    hf_cache_dir: Path | None = None  # read through the Hugging Face cache
    sample_path: Path | None = None  # a hash sample instead of the whole split
    max_documents: int | None = None
    sample_shard_size: int = SAMPLE_SHARD_SIZE


@dataclass(frozen=True)
class _Task:
    """Count one shard into ``counts_dir``; with ``tokenized_dir``, through
    its token cache."""

    texts: _Texts
    index: int
    num_shards: int
    ns: tuple[int, ...]
    vocab_size: int
    bos_id: int | None
    batch_size: int
    flush_every: int
    counts_dir: Path
    tokenized_dir: Path | None
    tokenized_meta: dict[str, Any]


def _num_shards(texts: _Texts) -> int:
    """How many shards the texts are read in.

    The shards do not depend on the number of processes, so that a run that
    was stopped can be completed with any number of them:

    - the whole split, streamed: its files;
    - the whole split, from the Hugging Face cache: CACHED_SHARD_SIZE
      documents each (made equal);
    - a hash sample: sample_shard_size documents each;
    - with max_documents: one shard, the first documents.
    """
    if texts.max_documents is not None:
        return 1
    if texts.sample_path is not None:
        if not texts.sample_path.exists():
            raise FileNotFoundError(
                f"{texts.sample_path} not found; make the sample first"
            )
        with texts.sample_path.open() as sample_file:
            num_rows = sum(1 for _ in sample_file)
        return max(1, -(-num_rows // texts.sample_shard_size))
    if texts.hf_cache_dir is not None:
        num_rows = split_rows(
            texts.corpus, revision=texts.revision, hf_cache_dir=texts.hf_cache_dir
        )
        return max(1, -(-num_rows // CACHED_SHARD_SIZE))
    return stream_shards(texts.corpus, revision=texts.revision)


def _shard_by(texts: _Texts) -> str:
    """How the texts are split into shards (shards of another split differ)."""
    if texts.max_documents is not None:
        return f"head{texts.max_documents}"
    if texts.sample_path is not None:
        return f"sample/{texts.sample_shard_size}"
    if texts.hf_cache_dir is not None:
        return "cached rows"
    return "files"


@contextmanager
def _open_shard(
    texts: _Texts, index: int, num_shards: int
) -> Iterator[tuple[Iterator[str], int | None]]:
    """Texts of shard ``index`` and their number (None if unknown)."""
    field = texts.corpus.text_field
    if texts.sample_path is not None:
        with texts.sample_path.open() as sample_file:
            start = index * texts.sample_shard_size
            stop = None if index == num_shards - 1 else start + texts.sample_shard_size
            lines = itertools.islice(sample_file, start, stop)
            yield _head(
                (json.loads(line)[field] for line in lines), None, texts.max_documents
            )
        return
    shard = None if texts.max_documents is not None else (num_shards, index)
    with open_rows(
        texts.corpus,
        revision=texts.revision,
        hf_cache_dir=texts.hf_cache_dir,
        shard=shard,
    ) as (rows, total):
        yield _head((row[field] for row in rows), total, texts.max_documents)


def _prepare_tokenized(
    tokenized_dir: Path, texts: _Texts, meta: dict[str, Any], revision: str | None
) -> tuple[_Texts, int]:
    """The texts and number of shards of the token cache, started if new.

    A cache that was started before keeps its shards and its dataset revision,
    so that the shards still to make are read like the ones already made.
    """
    if (tokenized_dir / "tokens.bin").exists():
        raise ValueError(
            f"{tokenized_dir} is a token cache of an older format; "
            "remove it to make it again"
        )
    meta_path = tokenized_dir / "meta.json"
    if meta_path.exists():
        saved = json.loads(meta_path.read_text())
        if saved["vocab_size"] != meta["vocab_size"]:
            raise ValueError(f"{tokenized_dir} has another vocabulary size")
        _check_revision(tokenized_dir, saved, revision)
        texts = replace(texts, revision=saved.get("dataset_revision"))
        return texts, saved["num_shards"]
    num_shards = _num_shards(texts)
    _write_json(meta_path, {**meta, "num_shards": num_shards, "created": _now()})
    return texts, num_shards


# counts of each shard, kept until they are added up


def _prepare_shard_counts(shards_dir: Path, plan: dict[str, Any]) -> None:
    """Keep the shard counts of a stopped run only if made by the same plan.

    The plan (which n, which shards, which data) is in ``plan.json``; shard
    counts of another plan are removed, as they cannot be added up with the
    new ones.
    """
    plan_path = shards_dir / "plan.json"
    plan = json.loads(json.dumps(plan))  # as it reads back
    if shards_dir.exists() and (
        not plan_path.exists() or json.loads(plan_path.read_text()) != plan
    ):
        shutil.rmtree(shards_dir)
    if not shards_dir.exists():
        _write_json(plan_path, plan)


def _save_shard_counts(
    counts: dict[int, Any], stats: dict[str, int], directory: Path
) -> None:
    """Write under ``<shard>.tmp/`` and rename: a shard directory is complete."""
    from corpus_tools.count import counts_filename, save_counts

    tmp = directory.with_name(directory.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for n, value in counts.items():
        save_counts(value, tmp / counts_filename(n))
    _write_json(tmp / "meta.json", stats)
    tmp.rename(directory)


def _add_shard_counts(
    shards_dir: Path, ns: list[int], num_shards: int
) -> tuple[dict[int, Any], dict[str, int]]:
    """The counts and stats of all shards, added up (one shard in memory at a time)."""
    from corpus_tools.count import counts_filename, load_counts

    total: dict[int, Any] = {}
    stats = {"num_documents": 0, "num_tokens": 0}
    for index in tqdm(range(num_shards), desc="Adding up shards", mininterval=10.0):
        directory = shards_dir / shard_name(index)
        for key, value in json.loads((directory / "meta.json").read_text()).items():
            stats[key] += value
        for n in ns:
            value = load_counts(directory / counts_filename(n))
            total[n] = value if n not in total else total[n] + value
    for value in total.values():
        if not isinstance(value, np.ndarray):
            value.sum_duplicates()
            value.sort_indices()
    return total, stats


# counting in processes

_worker: dict[str, Any] = {}  # the tokenizer and progress bar line of a worker


def available_cpus() -> int:
    """CPUs this process may use (those of its job on a cluster)."""
    if hasattr(os, "process_cpu_count"):  # Python 3.13
        return os.process_cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):  # Linux
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _workers(
    num_tasks: int, cpus: int | None, num_workers: int | None
) -> tuple[int, int | None]:
    """Processes, and tokenizer threads in each (None: count in this process,
    with the tokenizer's own number of threads).

    Processes come first, as they run reading and counting in parallel too,
    while threads only tokenize (one shard of OpenWebText with GPT-2: about
    200 seconds of tokenizing, split over the threads, and 53 of the rest).
    """
    if cpus is None and num_workers is None:
        return 1, None
    budget = available_cpus() if cpus is None else cpus
    processes = max(1, min(budget, num_workers or budget, num_tasks))
    return processes, max(1, budget // processes)


def _init_worker(tokenizer: Any, threads: int, positions: Any, lock: Any) -> None:
    # A worker that left a stream before its end (max_documents, an error)
    # would never exit, as any process (see corpus.run_and_exit), and the
    # executor would wait for it forever: the worker ends at once instead.
    atexit.register(_end_worker)
    # read by the tokenizer (Rust's rayon) when it first tokenizes a batch
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    tqdm.set_lock(lock)
    _worker["tokenizer"] = tokenizer
    _worker["position"] = positions.get()


def _end_worker() -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _count_shard_in_worker(task: _Task) -> int:
    return _count_shard(task, _worker["tokenizer"], _worker["position"])


def _count_shard(task: _Task, tokenizer: Any, position: int) -> int:
    """Count one shard and save its counts; returns the shard's index."""
    from corpus_tools.count import count_ngrams
    from corpus_tools.tokenize import TokenizedShard, tokenize

    label = f"shard {task.index + 1}/{task.num_shards}"
    stats = {"num_documents": 0, "num_tokens": 0}
    counting = {
        "ns": list(task.ns),
        "vocab_size": task.vocab_size,
        "bos_id": task.bos_id,
        "flush_every": task.flush_every,
        "desc": f"Counting {label}",
        "position": position,
    }
    if task.tokenized_dir is None:
        with _open_shard(task.texts, task.index, task.num_shards) as (texts, total):
            docs = tokenize(texts, tokenizer, batch_size=task.batch_size)
            counts = count_ngrams(_tally(docs, stats), total=total, **counting)
    else:
        if not task.tokenized_dir.is_dir():
            with _open_shard(task.texts, task.index, task.num_shards) as (
                texts,
                total,
            ):
                TokenizedShard.write(
                    tqdm(
                        tokenize(texts, tokenizer, batch_size=task.batch_size),
                        total=total,
                        desc=f"Tokenizing {label}",
                        position=position,
                        leave=False,
                        mininterval=10.0,
                    ),
                    task.tokenized_dir,
                    vocab_size=task.vocab_size,
                    meta={**task.tokenized_meta, "created": _now()},
                )
        shard = TokenizedShard(task.tokenized_dir)
        counts = count_ngrams(_tally(shard, stats), total=len(shard), **counting)
    _save_shard_counts(counts, stats, task.counts_dir)
    return task.index


def _run(
    tasks: list[_Task],
    tokenizer: Any,
    num_shards: int,
    processes: int,
    threads: int | None,
) -> None:
    """Count the shards of the tasks in ``processes`` worker processes with
    ``threads`` tokenizer threads each, or here if ``threads`` is None.

    Each worker shows its progress bar on its own line (1, 2, ...), under
    the bar of all shards on line 0. On an error, the workers end the shards
    they are on (and save them) before the error is raised.
    """
    progress = tqdm(
        total=num_shards, initial=num_shards - len(tasks), desc="Shards", position=0
    )
    if threads is None:
        for task in tasks:
            _count_shard(task, tokenizer, position=1)
            progress.update()
        progress.close()
        return
    # spawn, not fork: a forked tokenizer or stream reader may hang or turn
    # off its threads
    context = multiprocessing.get_context("spawn")
    positions = context.Queue()
    for position in range(1, processes + 1):
        positions.put(position)
    lock = context.RLock()  # for the bars of all processes to be written in turn
    tqdm.set_lock(lock)
    # an executor, not a Pool: a Pool restarts a worker that dies on starting
    # (a script without `if __name__ == "__main__":`) forever, and hangs
    executor = ProcessPoolExecutor(
        processes,
        mp_context=context,
        initializer=_init_worker,
        initargs=(tokenizer, threads, positions, lock),
    )
    try:
        futures = [executor.submit(_count_shard_in_worker, task) for task in tasks]
        for future in as_completed(futures):
            future.result()
            progress.update()
    except BrokenProcessPool as error:
        raise RuntimeError(
            "A worker process ended. If counting from a script with "
            'cpus or num_workers, run it under `if __name__ == "__main__":`, '
            "since every worker imports the script."
        ) from error
    finally:
        # drop the shards not started, and wait for the ones being counted
        executor.shutdown(wait=True, cancel_futures=True)
        progress.close()


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
