"""Runs on real data from the Hugging Face Hub: make test-network."""

from collections import Counter
from typing import Any, cast

import numpy as np
import pytest

from corpus_tools import Store, load_hash_sample, preset

pytestmark = pytest.mark.network


def test_tinystories_with_gpt2(tmp_path):
    from transformers import AutoTokenizer

    tokenizer = cast(Any, AutoTokenizer.from_pretrained("gpt2"))
    corpus = preset("tinystories", split="validation")
    store = Store(tmp_path, cache_dir=tmp_path / "cache")
    sample = load_hash_sample(store.sample(corpus, 50))

    by_cache = {
        cache: Store(tmp_path / cache, cache_dir=tmp_path / cache / "cache").counts(
            corpus, tokenizer, [1, 2], cache=cache, max_documents=300
        )
        for cache in ("none", "tokens")
    }
    assert np.array_equal(by_cache["none"][1], by_cache["tokens"][1])
    assert (by_cache["none"][2] != by_cache["tokens"][2]).nnz == 0

    counts = store.counts(corpus, tokenizer, [2], source="hash_n50", bos=True)
    expected = Counter()
    for row in sample:
        ids = [tokenizer.bos_token_id, *tokenizer.encode(row["text"])]
        expected.update(zip(ids[:-1], ids[1:]))
    coo = counts[2].tocoo()
    assert {
        (int(a), int(b)): int(c) for a, b, c in zip(coo.row, coo.col, coo.data)
    } == dict(expected)


def test_hub_model_id():
    from corpus_tools.store import hub_model_id

    assert hub_model_id("gpt2") == "openai-community/gpt2"
    assert hub_model_id("openai-community/gpt2") == "openai-community/gpt2"
    assert (
        hub_model_id("no-such-user/no-such-model-xyz")
        == "no-such-user/no-such-model-xyz"
    )


def test_run_and_exit_ends_a_process_that_left_a_stream():
    # Without run_and_exit, this process does not exit (see run_and_exit).
    import subprocess
    import sys

    code = (
        "import itertools, time\n"
        "from corpus_tools import preset\n"
        "from corpus_tools.corpus import open_rows, run_and_exit\n"
        "def main():\n"
        "    with open_rows(preset('openwebtext')) as (rows, _):\n"
        "        for i, _ in enumerate(itertools.islice(rows, 50)):\n"
        "            if i == 10:\n"
        "                time.sleep(5)\n"
        "run_and_exit(main)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], timeout=120, check=False)
    assert result.returncode == 0


def test_worker_that_left_a_stream_ends(tmp_path):
    # Without _end_worker, the worker does not exit after leaving the stream,
    # and the count waits for it forever.
    import os
    import signal
    import subprocess
    import sys

    code = (
        "from corpus_tools import Store, preset\n"
        "if __name__ == '__main__':\n"
        f"    Store({str(tmp_path / 'out')!r}, cache_dir={str(tmp_path / 'cache')!r})"
        ".counts(preset('openwebtext'), 'openai-community/gpt2', [1],"
        " max_documents=200, cpus=1)\n"
    )
    # in its own process group, so that a hung worker is killed with it
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        assert process.wait(timeout=300) == 0
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
