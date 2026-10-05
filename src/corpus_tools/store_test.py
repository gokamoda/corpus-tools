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
from corpus_tools.tokenize import TokenizedCorpus, shard_name
from corpus_tools.tokenize_test import CharTokenizer

CORPUS = Corpus(dataset="someone/letters", name="default", split="train")
TEXTS = ["abcab", "ba", "a", "", "zzza", "hello", "world", "abcab"] * 5
NUM_FILES = 3  # shards of the stream


def contiguous_part(items, num_shards, index):
    size, extra = divmod(len(items), num_shards)
    start = index * size + min(index, extra)
    return items[start : start + size + (index < extra)]


def tokenized_dir_of(tmp_path):
    return tmp_path / "cache/someone--letters/default/train/all/char/tokenized"


def make_store(tmp_path):
    return Store(tmp_path, cache_dir=tmp_path / "cache")


@pytest.fixture
def reads(monkeypatch):
    """Serve TEXTS as the corpus, and record how it was read."""
    calls = []

    @contextmanager
    def fake_open_rows(corpus, *, revision=None, hf_cache_dir=None, shard=None):
        calls.append(
            {"revision": revision, "hf_cache_dir": hf_cache_dir, "shard": shard}
        )
        rows = [{"text": t, "i": i} for i, t in enumerate(TEXTS)]
        if shard is not None:
            rows = contiguous_part(rows, *shard)
        yield iter(rows), len(rows)

    monkeypatch.setattr(store_module, "open_rows", fake_open_rows)
    monkeypatch.setattr(store_module, "stream_shards", lambda corpus, **_: NUM_FILES)
    monkeypatch.setattr(store_module, "split_rows", lambda corpus, **_: len(TEXTS))
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


def test_directories():
    store = Store("~/results")
    assert store.output_dir == Path.home() / "results"
    assert store.cache_dir == Path.home() / ".cache/corpus-tools"
    assert Store("out", cache_dir="~/elsewhere").cache_dir == Path.home() / "elsewhere"


def test_results_and_caches_are_apart(tmp_path, reads):
    store = make_store(tmp_path)
    store.counts(CORPUS, CharTokenizer(), [1], cache="tokens")
    relative = Path("someone--letters/default/train/all/char")
    assert (tmp_path / relative / "counts/nobos/1-grams.npy").exists()
    assert (tmp_path / "cache" / relative / "tokenized/meta.json").exists()
    assert not (tmp_path / relative / "tokenized").exists()
    assert not (tmp_path / "cache" / relative / "counts/nobos/1-grams.npy").exists()


def test_sample_is_made_once(tmp_path, reads):
    store = make_store(tmp_path)
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
    store = make_store(tmp_path)
    counts = store.counts(CORPUS, CharTokenizer(), [2, 1, 3], cache=cache, bos=bos)
    assert_same(counts, expected([1, 2, 3], bos_id=25 if bos else None))
    assert reads[0]["revision"] == "abc123"
    assert (reads[0]["hf_cache_dir"] is not None) == (cache == "corpus")
    num_reads = 1 if cache == "corpus" else NUM_FILES  # with one process
    assert sorted(r["shard"] for r in reads) == [
        (num_reads, i) for i in range(num_reads)
    ]

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
    assert len(reads) == num_reads


def test_token_cache_is_shared_by_bos_and_new_n(tmp_path, reads):
    store = make_store(tmp_path)
    store.counts(CORPUS, CharTokenizer(), [1], cache="tokens")
    counts = store.counts(CORPUS, CharTokenizer(), [2], cache="tokens", bos=True)
    assert_same(counts, expected([2], bos_id=25))
    assert len(reads) == NUM_FILES

    tokenized = TokenizedCorpus(tokenized_dir_of(tmp_path))
    assert [d.tolist() for d in tokenized] == [
        [ord(c) - ord("a") for c in t] for t in TEXTS
    ]
    assert tokenized.meta["num_shards"] == NUM_FILES
    assert tokenized.meta["dataset_revision"] == "abc123"


def test_counts_of_a_sample(tmp_path, reads):
    store = make_store(tmp_path)
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
    store = make_store(tmp_path)
    counts = store.counts(CORPUS, CharTokenizer(), [1], max_documents=3)
    assert_same(counts, expected([1], texts=TEXTS[:3]))
    assert (tmp_path / "someone--letters/default/train/all_head3/char").is_dir()


def test_missing_sample(tmp_path, reads):
    with pytest.raises(FileNotFoundError):
        make_store(tmp_path).counts(CORPUS, CharTokenizer(), [1], source="hash_n4")


def test_bos_needs_a_bos_token(tmp_path, reads):
    tokenizer = CharTokenizer()
    tokenizer.bos_token_id = None
    with pytest.raises(ValueError):
        make_store(tmp_path).counts(CORPUS, tokenizer, [1], bos=True)


def test_tokenizer_name_sets_the_directory(tmp_path, reads):
    store = make_store(tmp_path)
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
    store = make_store(tmp_path)
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
    store = make_store(tmp_path)
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
    assert reads[-1] == {"revision": "r9", "hf_cache_dir": None, "shard": None}


def test_interrupted_token_cache_is_completed(tmp_path, reads):
    store = make_store(tmp_path)
    store.counts(CORPUS, CharTokenizer(), [1], cache="tokens")
    tokenized_dir = tokenized_dir_of(tmp_path)
    counts_dir = tmp_path / "someone--letters/default/train/all/char/counts/nobos"
    # as if stopped while writing shard 1, before the counts were saved
    shard = tokenized_dir / shard_name(1)
    shard.rename(shard.with_name(shard.name + ".tmp"))
    (counts_dir / "1-grams.npy").unlink()
    reads.clear()

    counts = store.counts(CORPUS, CharTokenizer(), [1, 2], cache="tokens")
    assert_same(counts, expected([1, 2]))
    assert [r["shard"] for r in reads] == [(NUM_FILES, 1)]  # only the missing shard
    assert not (tokenized_dir / (shard_name(1) + ".tmp")).exists()


def test_counts_without_their_file_are_made_again(tmp_path, reads):
    store = make_store(tmp_path)
    counts_dir = tmp_path / "someone--letters/default/train/all/char/counts/nobos"
    counts_dir.mkdir(parents=True)
    # as if stopped after the meta was written, before the counts
    (counts_dir / "2-grams.json").write_text("{}")
    assert_same(store.counts(CORPUS, CharTokenizer(), [2]), expected([2]))
    assert json.loads((counts_dir / "2-grams.json").read_text())["n"] == 2


def test_token_cache_of_the_older_format_is_refused(tmp_path, reads):
    tokenized_dir = tokenized_dir_of(tmp_path)
    tokenized_dir.mkdir(parents=True)
    (tokenized_dir / "tokens.bin").write_bytes(b"")
    with pytest.raises(ValueError, match="older format"):
        make_store(tmp_path).counts(CORPUS, CharTokenizer(), [1], cache="tokens")


def test_sample_without_its_file_is_made_again(tmp_path, reads):
    store = make_store(tmp_path)
    path = store.sample_path(CORPUS, 3)
    path.parent.mkdir(parents=True)
    # as if stopped after the meta was written, before the sample was renamed
    path.with_suffix(".meta.json").write_text('{"dataset_revision": "old"}')
    path.with_name(path.name + ".partial").write_text("broken\n")
    assert store.sample(CORPUS, 3, revision="r1") == path
    assert len(path.read_text().splitlines()) == 3
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    assert meta["dataset_revision"] == "r1"
    assert not path.with_name(path.name + ".partial").exists()


def test_sample_is_read_in_shards(tmp_path, reads, monkeypatch):
    monkeypatch.setattr(store_module, "SAMPLE_SHARD_SIZE", 3)
    store = make_store(tmp_path)
    store.sample(CORPUS, 8)
    counts = store.counts(
        CORPUS, CharTokenizer(), [1, 2], source="hash_n8", cache="tokens"
    )
    sample = hash_sample([{"text": t} for t in TEXTS], num_samples=8)
    assert_same(counts, expected([1, 2], texts=[row["text"] for row in sample]))
    tokenized = TokenizedCorpus(
        tmp_path / "cache/someone--letters/default/train/hash_n8/char/tokenized"
    )
    assert [len(shard) for shard in tokenized.shards] == [3, 3, 1]  # 7 distinct


@pytest.fixture
def parquet_corpus(tmp_path, monkeypatch):
    """TEXTS in NUM_FILES local Parquet files, readable by worker processes."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = tmp_path / "letters"
    (directory / "data").mkdir(parents=True)
    for i in range(NUM_FILES):
        pq.write_table(
            pa.table({"text": contiguous_part(TEXTS, NUM_FILES, i)}),
            directory / "data" / f"train-{i:05d}-of-{NUM_FILES:05d}.parquet",
        )
    monkeypatch.setattr(store_module, "dataset_revision", lambda corpus: None)
    monkeypatch.setattr(store_module, "hub_model_id", lambda name: name)
    return Corpus(dataset=str(directory), name="default", split="train")


@pytest.mark.parametrize("cache", ["none", "corpus", "tokens"])
def test_counts_in_processes(tmp_path, parquet_corpus, cache, monkeypatch):
    monkeypatch.setattr(store_module, "CACHED_SHARD_SIZE", 15)  # 3 shards
    store = Store(tmp_path / "out", cache_dir=tmp_path / "cache")
    counts = store.counts(
        parquet_corpus,
        CharTokenizer(),
        [1, 2, 3],
        cache=cache,
        bos=True,
        cpus=2,
    )
    assert_same(counts, expected([1, 2, 3], bos_id=25))
    tokenizer_dir = store.tokenizer_dir(parquet_corpus, "char")
    meta = json.loads((tokenizer_dir / "counts/bos/2-grams.json").read_text())
    assert meta["num_documents"] == len(TEXTS)
    assert meta["num_tokens"] == sum(len(t) for t in TEXTS)
    if cache == "tokens":
        tokenized_dir = store.tokenizer_cache_dir(parquet_corpus, "char") / "tokenized"
        assert TokenizedCorpus(tokenized_dir).meta["num_shards"] == 3


def test_stopped_count_continues_from_its_shard_counts(tmp_path, reads, monkeypatch):
    count_shard = store_module._count_shard

    def fail_on_the_last_shard(task, tokenizer, position):
        if task.index == NUM_FILES - 1:
            raise KeyboardInterrupt
        return count_shard(task, tokenizer, position)

    monkeypatch.setattr(store_module, "_count_shard", fail_on_the_last_shard)
    store = make_store(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        store.counts(CORPUS, CharTokenizer(), [1, 2])
    shards_dir = (
        tmp_path / "cache/someone--letters/default/train/all/char/counts/nobos/shards"
    )
    assert sorted(p.name for p in shards_dir.iterdir()) == [
        "plan.json", "shard-00000", "shard-00001",
    ]  # fmt: skip

    monkeypatch.setattr(store_module, "_count_shard", count_shard)
    reads.clear()
    assert_same(store.counts(CORPUS, CharTokenizer(), [1, 2]), expected([1, 2]))
    assert [r["shard"] for r in reads] == [(NUM_FILES, NUM_FILES - 1)]
    assert not shards_dir.exists()  # removed once added up


def test_shard_counts_of_another_plan_are_not_used(tmp_path, reads, monkeypatch):
    count_shard = store_module._count_shard

    def fail_on_the_last_shard(task, tokenizer, position):
        if task.index == NUM_FILES - 1:
            raise KeyboardInterrupt
        return count_shard(task, tokenizer, position)

    monkeypatch.setattr(store_module, "_count_shard", fail_on_the_last_shard)
    store = make_store(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        store.counts(CORPUS, CharTokenizer(), [1, 2])
    monkeypatch.setattr(store_module, "_count_shard", count_shard)
    reads.clear()
    # other n: the shards are counted again
    assert_same(store.counts(CORPUS, CharTokenizer(), [2, 3]), expected([2, 3]))
    assert len(reads) == NUM_FILES


def test_cached_split_is_read_in_shards_of_a_fixed_size(tmp_path, reads, monkeypatch):
    monkeypatch.setattr(store_module, "CACHED_SHARD_SIZE", 15)
    counts = make_store(tmp_path).counts(
        CORPUS, CharTokenizer(), [1, 2], cache="corpus"
    )
    assert_same(counts, expected([1, 2]))
    assert sorted(r["shard"] for r in reads) == [(3, 0), (3, 1), (3, 2)]  # 40 / 15


def test_error_in_a_worker(tmp_path, parquet_corpus):
    # an upper-case letter is out of the vocabulary of CharTokenizer
    data = Path(parquet_corpus.dataset) / "data"
    import pyarrow as pa
    import pyarrow.parquet as pq

    bad = sorted(data.iterdir())[1]
    pq.write_table(pa.table({"text": ["abc", "ABC"]}), bad)
    store = Store(tmp_path / "out", cache_dir=tmp_path / "cache")
    with pytest.raises(ValueError):
        store.counts(parquet_corpus, CharTokenizer(), [1], num_workers=2)
    shards_dir = (
        store.tokenizer_cache_dir(parquet_corpus, "char") / "counts/nobos/shards"
    )
    # the shard counted next to the bad one was finished and kept
    assert (shards_dir / shard_name(0)).is_dir()
    assert not (shards_dir / shard_name(1)).exists()


def test_script_without_main_guard_ends_with_an_error(tmp_path, parquet_corpus):
    import subprocess
    import sys

    script = tmp_path / "count.py"
    script.write_text(
        "from corpus_tools import Store\n"
        "from corpus_tools.corpus import Corpus\n"
        "from corpus_tools.tokenize_test import CharTokenizer\n"
        f"corpus = Corpus({parquet_corpus.dataset!r}, 'default')\n"
        f"Store({str(tmp_path / 'out')!r}, cache_dir={str(tmp_path / 'cache')!r}).counts(\n"
        "    corpus, CharTokenizer(), [1], num_workers=2, tokenizer_name='char'\n"
        ")\n"
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode != 0
    assert 'if __name__ == "__main__":' in result.stderr


@pytest.mark.parametrize(
    ("num_tasks", "cpus", "num_workers", "expected"),
    [
        (80, None, None, (1, None)),  # in this process, as the tokenizer likes
        (80, 16, None, (16, 1)),  # processes first
        (4, 16, None, (4, 4)),  # few shards: the rest to threads
        (15, 16, None, (15, 1)),
        (80, 16, 4, (4, 4)),  # num_workers caps the processes
        (80, 1, None, (1, 1)),  # one process, one thread
        (2, None, 4, (2, 6)),  # the job's 12 CPUs
    ],
)
def test_workers(monkeypatch, num_tasks, cpus, num_workers, expected):
    monkeypatch.setattr(store_module, "available_cpus", lambda: 12)
    assert store_module._workers(num_tasks, cpus, num_workers) == expected


def test_worker_sets_the_tokenizer_threads(monkeypatch):
    import multiprocessing
    import queue

    monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
    positions = queue.Queue()
    positions.put(3)
    lock = multiprocessing.get_context("spawn").RLock()
    monkeypatch.setattr(store_module.tqdm, "set_lock", lambda lock: None)
    monkeypatch.setattr(store_module, "_worker", {})
    at_exit = []  # not to end this process (pytest) with os._exit
    monkeypatch.setattr(store_module.atexit, "register", at_exit.append)
    store_module._init_worker("tokenizer", 4, positions, lock)
    assert store_module.os.environ["RAYON_NUM_THREADS"] == "4"
    assert store_module._worker == {"tokenizer": "tokenizer", "position": 3}
    assert at_exit == [store_module._end_worker]


def test_cli_needs_an_output_dir(tmp_path, reads, monkeypatch, capsys):
    from corpus_tools import cli

    monkeypatch.setattr(cli, "preset", lambda *args, **kwargs: CORPUS)
    common = ["--corpus", "openwebtext", "--cache-dir", str(tmp_path / "cache")]
    with pytest.raises(SystemExit):
        cli.main(["count", *common, "--tokenizer", "char"])
    assert "--output-dir is required" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(
            ["sample", *common, "--num-samples", "3", "--output", str(tmp_path / "s.jsonl"),
             "--output-dir", str(tmp_path)]
        )  # fmt: skip
    assert "either --output or --output-dir" in capsys.readouterr().err

    cli.main(["sample", *common, "--num-samples", "3", "--output-dir", str(tmp_path)])
    assert make_store(tmp_path).sample_path(CORPUS, 3).exists()


def test_counts_at_a_revision(tmp_path, reads):
    store = make_store(tmp_path)
    store.counts(CORPUS, CharTokenizer(), [1], revision="r1")
    assert {r["revision"] for r in reads} == {"r1"}
    counts_dir = tmp_path / "someone--letters/default/train/all/char/counts/nobos"
    assert (
        json.loads((counts_dir / "1-grams.json").read_text())["dataset_revision"]
        == "r1"
    )
    store.counts(CORPUS, CharTokenizer(), [1])  # the saved counts, of any revision
    with pytest.raises(ValueError, match="r1"):
        store.counts(CORPUS, CharTokenizer(), [1], revision="r2")


def test_token_cache_of_another_revision_is_refused(tmp_path, reads):
    store = make_store(tmp_path)
    store.counts(CORPUS, CharTokenizer(), [1], cache="tokens", revision="r1")
    with pytest.raises(ValueError, match="r1"):
        store.counts(CORPUS, CharTokenizer(), [2], cache="tokens", revision="r2")


def test_sample_of_another_revision_is_not_counted(tmp_path, reads):
    store = make_store(tmp_path)
    store.sample(CORPUS, 3, revision="r1")
    store.counts(CORPUS, CharTokenizer(), [1], source="hash_n3", revision="r1")
    with pytest.raises(ValueError, match="r1"):
        store.counts(CORPUS, CharTokenizer(), [2], source="hash_n3", revision="r2")


def test_cli_cache_and_revision_are_shared(tmp_path, reads, monkeypatch):
    from corpus_tools import cli

    monkeypatch.setattr(cli, "preset", lambda *args, **kwargs: CORPUS)
    monkeypatch.setattr(cli, "hub_model_id", lambda name: name)
    monkeypatch.setattr(store_module, "split_rows", lambda corpus, **_: len(TEXTS))
    common = ["--corpus", "openwebtext", "--output-dir", str(tmp_path),
              "--cache-dir", str(tmp_path / "cache"), "--cache", "corpus",
              "--revision", "r5"]  # fmt: skip
    cli.main(["sample", *common, "--num-samples", "3"])
    assert reads[-1]["hf_cache_dir"] == tmp_path / "cache/hf"
    assert reads[-1]["revision"] == "r5"

    tokenizer = CharTokenizer()
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda name: tokenizer
    )
    reads.clear()
    cli.main(["count", *common, "--tokenizer", "char", "--n", "1"])
    assert reads[-1]["hf_cache_dir"] == tmp_path / "cache/hf"
    assert {r["revision"] for r in reads} == {"r5"}
