"""Draw a fixed random sample from a (streaming) dataset by hashing.

Every row gets the key sha256(row[key_field]), and the sample is the
``num_samples`` rows with the smallest keys, written in ascending key order.

- The key does not depend on the content, so the sample is uniformly random.
- The first n rows of a sample are the sample of size n, so a sample can be
  extended (10k -> 100k -> ...) without changing the rows already used.
- The key depends only on the row itself, not on its position, so the sample
  does not change when the dataset is re-sharded or reordered.
- Rows with the same key (identical texts) are kept only once.
- ``hash_fold`` splits a sample into disjoint folds by key, so a row stays in
  the same fold however the sample is extended or later filtered (e.g. by a
  model-specific token length).

The whole dataset is read once, but only ``num_samples`` rows are kept in
memory. The key is computed from the full text; ``max_chars`` only shortens
the text that is stored.

This is the sampling of inseg-attention and lm-detokenization; the saved files
are byte-identical to theirs.
"""

import hashlib
import heapq
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from tqdm import tqdm


def hash_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_fold(key: str, num_folds: int) -> int:
    """Fold (0..num_folds-1) of a row with hash ``key``."""
    if num_folds < 1:
        raise ValueError("num_folds must be at least 1")
    return int(key, 16) % num_folds


def hash_sample(
    rows: Iterable[dict[str, Any]],
    num_samples: int,
    *,
    key_field: str = "text",
    max_chars: int | None = None,
) -> list[dict[str, Any]]:
    """Return the ``num_samples`` rows with the smallest hash keys, sorted by key.

    Each returned row has an extra field "hash" holding its key. With
    ``max_chars``, ``row[key_field]`` is cut to its first ``max_chars``
    characters after hashing.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    if max_chars is not None and max_chars < 1:
        raise ValueError("max_chars must be at least 1")

    # max-heap of the smallest keys seen so far: (negated key, key, row)
    heap: list[tuple[int, str, dict[str, Any]]] = []
    kept_keys: set[str] = set()
    for row in rows:
        key = hash_key(row[key_field])
        if key in kept_keys:
            continue
        neg_key = -int(key, 16)
        if len(heap) == num_samples and neg_key <= heap[0][0]:
            continue
        kept = {**row, "hash": key}
        if max_chars is not None:
            kept[key_field] = row[key_field][:max_chars]
        if len(heap) < num_samples:
            heapq.heappush(heap, (neg_key, key, kept))
        else:
            _, removed_key, _ = heapq.heapreplace(heap, (neg_key, key, kept))
            kept_keys.discard(removed_key)
        kept_keys.add(key)

    return [row for _, _, row in sorted(heap, key=lambda item: -item[0])]


def save_hash_sample(
    rows: Iterable[dict[str, Any]],
    output_path: Path,
    num_samples: int,
    *,
    key_field: str = "text",
    max_chars: int | None = None,
    total: int | None = None,
) -> int:
    """Write ``hash_sample(rows, ...)`` to ``output_path`` as JSON Lines.

    Returns the number of rows written.
    """
    sample = hash_sample(
        tqdm(rows, total=total, desc="Hashing", mininterval=10.0),
        num_samples=num_samples,
        key_field=key_field,
        max_chars=max_chars,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    with tmp_path.open("w") as output_file:
        for row in sample:
            json.dump(row, output_file, ensure_ascii=False)
            output_file.write("\n")
    tmp_path.rename(output_path)
    return len(sample)


def load_hash_sample(path: Path, num_samples: int | None = None) -> list[dict]:
    """Read the first ``num_samples`` rows (all if None) of a saved sample."""
    rows = []
    with path.open() as input_file:
        for line in input_file:
            if num_samples is not None and len(rows) >= num_samples:
                break
            rows.append(json.loads(line))
    if num_samples is not None and len(rows) < num_samples:
        raise ValueError(
            f"{path} has only {len(rows)} rows, but {num_samples} were requested"
        )
    return rows


def iter_hash_fold(
    path: Path, *, fold: int, num_folds: int
) -> Iterator[dict[str, Any]]:
    """Rows of a saved hash sample that fall in ``fold``, in key order."""
    if not 0 <= fold < num_folds:
        raise ValueError(f"fold must be in [0, {num_folds}), got {fold}")
    with path.open() as input_file:
        for line in input_file:
            row = json.loads(line)
            if hash_fold(row["hash"], num_folds) == fold:
                yield row


def sample_name(num_samples: int, max_chars: int | None = None) -> str:
    """Name of a hash sample, used for its file and as a count source."""
    chars = f"_chars{max_chars}" if max_chars is not None else ""
    return f"hash_n{num_samples}{chars}"
