"""The ``corpus-tools`` command: ``sample`` and ``count``."""

import argparse
from pathlib import Path

from corpus_tools.corpus import PRESETS, preset, run_and_exit
from corpus_tools.store import ALL, DEFAULT_CACHE_DIR, Store, hub_model_id


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory to save to (default: %(default)s).",
    )
    parser.add_argument("--corpus", choices=sorted(PRESETS), required=True)
    parser.add_argument(
        "--name",
        default=None,
        help="Dataset config; required for wikipedia (e.g. 20231101.ja).",
    )
    parser.add_argument("--split", default="train")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="corpus-tools")
    commands = parser.add_subparsers(dest="command", required=True)

    sample = commands.add_parser(
        "sample",
        help="Save the hash sample of a corpus.",
        description="Read the whole split once and keep the NUM_SAMPLES rows "
        "with the smallest sha256 of their text, in that order.",
    )
    _add_common_args(sample)
    sample.add_argument("--num-samples", type=int, required=True)
    sample.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Store only the first MAX_CHARS characters of each text "
        "(the hash is of the full text).",
    )
    sample.add_argument(
        "--cache-corpus",
        action="store_true",
        help="Read the corpus through the Hugging Face cache under the cache directory "
        "instead of streaming it.",
    )

    count = commands.add_parser(
        "count",
        help="Count token n-grams of a corpus or of its hash sample.",
        description="Texts are tokenized as they are, without special tokens, "
        "and n-grams are counted within each document.",
    )
    _add_common_args(count)
    count.add_argument("--tokenizer", required=True, help="Hugging Face name.")
    count.add_argument("--n", type=int, nargs="+", default=[1, 2])
    count.add_argument(
        "--source",
        default=ALL,
        help=f"{ALL} (the whole split) or a saved sample, e.g. hash_n10000.",
    )
    count.add_argument(
        "--cache",
        choices=["none", "corpus", "tokens"],
        default="none",
        help="none: stream and count. corpus: read through the Hugging Face "
        "cache. tokens: save the token ids once and count from them.",
    )
    count.add_argument(
        "--bos", action="store_true", help="Put the BOS token before each document."
    )
    count.add_argument(
        "--tokenizer-name",
        default=None,
        help="Name to save the counts under (default: the Hub id of --tokenizer, "
        "e.g. openai-community/gpt2 for gpt2).",
    )
    count.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Only the first documents (for a quick test; saved separately).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    store = Store(args.cache_dir)
    corpus = preset(args.corpus, split=args.split, name=args.name)
    if args.command == "sample":
        path = store.sample(
            corpus,
            args.num_samples,
            max_chars=args.max_chars,
            cache_corpus=args.cache_corpus,
        )
        print(f"sample: {path}")
        return

    tokenizer_name = args.tokenizer_name or hub_model_id(args.tokenizer)
    counts = store.counts(
        corpus,
        args.tokenizer,
        args.n,
        source=args.source,
        cache=args.cache,
        bos=args.bos,
        max_documents=args.max_documents,
        tokenizer_name=tokenizer_name,
    )
    counts_dir = (
        store.tokenizer_dir(
            corpus,
            tokenizer_name,
            args.source,
            args.max_documents,
        )
        / "counts"
        / ("bos" if args.bos else "nobos")
    )
    for n, matrix in counts.items():
        total = int(matrix.sum())
        distinct = int((matrix != 0).sum()) if n == 1 else matrix.nnz
        print(f"n={n}: {total:,} n-grams, {distinct:,} distinct ({counts_dir})")


def entry() -> None:
    """The console script: main, ended by run_and_exit (see there for why)."""
    run_and_exit(main)


if __name__ == "__main__":
    entry()
