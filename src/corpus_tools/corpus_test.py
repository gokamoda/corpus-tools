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
