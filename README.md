# corpus-tools

HF データセットの hash サンプルと、token の n-gram 頻度を作るライブラリ。対応しているコーパスは `openwebtext`, `tinystories`, `wikipedia`。

## インストール

```toml
# 使う側の pyproject.toml
dependencies = ["corpus-tools[count]"]  # サンプルだけなら "corpus-tools"

[tool.uv.sources]
corpus-tools = { git = "https://github.com/gokamoda/corpus-tools.git", rev = "<commit>" }
```

開発するときは `make install`。

## 使い方

保存先は `--cache-dir`（Python では `Store(cache_dir)`）で指定する。未指定なら `~/.cache/corpus-tools`。`--name` はデータセットの config で、wikipedia では版（`20231101.ja` など）を必ず指定する。openwebtext と tinystories では省略でき、既定の config（`plain_text`, `default`）を使う。

### sample: hash サンプル

```bash
corpus-tools sample --corpus openwebtext --num-samples 10000
corpus-tools sample --corpus tinystories --split validation --num-samples 1000
corpus-tools sample --corpus wikipedia --name 20231101.ja --num-samples 10000 --max-chars 2000
# 保存先のファイルと、読むデータセットの commit を指定する場合
corpus-tools sample --corpus openwebtext --num-samples 10000 --revision 79d93d786212f7344586290adb811d4ae6a1762c --output data/openwebtext/train_hash_n10000.jsonl
```

```python
from corpus_tools import Store, preset, load_hash_sample

store = Store("~/.cache/corpus-tools")  # 省略時もこの場所
rows = load_hash_sample(store.sample(preset("openwebtext"), 10_000))
```

### count: n-gram 頻度

`--source` は `all`（split 全体）か、保存済みのサンプル（`hash_n10000` など）。

```bash
corpus-tools count --corpus openwebtext --tokenizer openai-community/gpt2 --n 1 2
corpus-tools count --corpus tinystories --tokenizer openai-community/gpt2 --n 1 2 3 --cache tokens
corpus-tools count --corpus openwebtext --tokenizer openai-community/gpt2 --n 1 2 --source hash_n10000 --bos
```

```python
counts = store.counts(preset("openwebtext"), "openai-community/gpt2", [1, 2])
# {1: ndarray[V], 2: csr_matrix[V, V]}
```

数え方の規則、保存先の構成、cache の選び方は `AGENTS.md` にある。
