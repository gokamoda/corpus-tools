import subprocess
import sys
from pathlib import Path

import pytest

from corpus_tools.corpus import Corpus, preset


def test_presets():
    assert preset("openwebtext") == Corpus("Skylion007/openwebtext", "plain_text")
    assert preset("tinystories", split="validation").split == "validation"
    wikipedia = preset("wikipedia", name="20231101.ja")
    assert wikipedia == Corpus("wikimedia/wikipedia", "20231101.ja")
    assert wikipedia.path == Path("wikimedia--wikipedia/20231101.ja/train")


def test_wikipedia_needs_a_name():
    with pytest.raises(ValueError):
        preset("wikipedia")


def test_unknown_preset():
    with pytest.raises(ValueError):
        preset("wikitext")


@pytest.mark.parametrize(
    ("body", "status", "stderr"),
    [
        ("print('done')", 0, ""),
        ("sys.exit(3)", 3, ""),
        ("sys.exit('bad usage')", 1, "bad usage"),
        ("raise ValueError('boom')", 1, "ValueError: boom"),
        ("raise KeyboardInterrupt", 130, ""),
    ],
)
def test_run_and_exit(body, status, stderr):
    code = (
        "import sys\n"
        "from corpus_tools.corpus import run_and_exit\n"
        "def main():\n"
        f"    {body}\n"
        "run_and_exit(main)\n"
        "print('not reached')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == status
    assert stderr in result.stderr
    assert "not reached" not in result.stdout


def test_run_and_exit_leaves_no_semaphore_of_tqdm():
    code = (
        "from tqdm import tqdm\n"
        "from corpus_tools.corpus import run_and_exit\n"
        "def main():\n"
        "    for _ in tqdm(range(3)):\n"
        "        pass\n"
        "run_and_exit(main)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0
    assert "leaked semaphore" not in result.stderr
