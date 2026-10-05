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
    store = Store(tmp_path)
    sample = load_hash_sample(store.sample(corpus, 50))

    by_cache = {
        cache: Store(tmp_path / cache).counts(
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
