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

結果（サンプルと頻度）の保存先は `--output-dir`（Python では `Store(output_dir)`）で、毎回指定する。作り直せるもの（HF の cache、tokenize 済みの列、数えている途中の頻度）は `--cache-dir`（`Store(..., cache_dir=)`）に置き、未指定なら `~/.cache/corpus-tools`。`--corpus`, `--name`, `--split`, `--revision`（データセットの commit）, `--cache` は sample と count で共通。`--name` はデータセットの config で、wikipedia では版（`20231101.ja` など）を必ず指定する。openwebtext と tinystories では省略でき、既定の config（`plain_text`, `default`）を使う。

### sample: hash サンプル

```bash
corpus-tools sample --corpus openwebtext --num-samples 10000 --output-dir ~/corpus-tools
corpus-tools sample --corpus tinystories --split validation --num-samples 1000 --output-dir ~/corpus-tools
corpus-tools sample --corpus wikipedia --name 20231101.ja --num-samples 10000 --max-chars 2000 --output-dir ~/corpus-tools
# 3,000 文字未満の文書を除いてからサンプルする場合（hash_n10000_min3000.jsonl）
corpus-tools sample --corpus wikipedia --name 20231101.en --num-samples 10000 --min-chars 3000 --output-dir ~/corpus-tools
# 保存先のファイルと、読むデータセットの commit を指定する場合
corpus-tools sample --corpus openwebtext --num-samples 10000 --revision 79d93d786212f7344586290adb811d4ae6a1762c --output data/openwebtext/train_hash_n10000.jsonl
```

- `--max-chars N`: 各文書の先頭 N 文字だけを保存する（hash は全文から作る）。
- `--min-chars N`: N 文字未満の文書を除いてからサンプルする。結果は、`--min-chars` なしのサンプルから N 文字未満の文書を除いたものと、同じ順番になる。トークン数で絞りたいときは、1 トークンあたりの文字数から余裕を持って N を決め、使う側でトークン数でも絞る。英語の Wikipedia と OpenWebText では、GPT-2・Llama・Qwen・Gemma・SmolLM2 の tokenizer で、1 トークンはおよそ 4〜5 文字（99% 点で 5.5 文字、最大で 6.3 文字）。L トークン以上を残したいなら `--min-chars` を 6L 程度にすると、ほぼすべてが L トークン以上になる。

```python
from corpus_tools import Store, preset, load_hash_sample

store = Store("~/corpus-tools")  # cache_dir は省略すると ~/.cache/corpus-tools
rows = load_hash_sample(store.sample(preset("openwebtext"), 10_000))
```

### count: n-gram 頻度

`--source` は `all`（split 全体）か、保存済みのサンプル（`hash_n10000` など）。

```bash
corpus-tools count --corpus openwebtext --tokenizer openai-community/gpt2 --n 1 2 --output-dir ~/corpus-tools
corpus-tools count --corpus tinystories --tokenizer openai-community/gpt2 --n 1 2 3 --cache tokens --output-dir ~/corpus-tools
corpus-tools count --corpus openwebtext --tokenizer openai-community/gpt2 --n 1 2 --source hash_n10000 --bos --output-dir ~/corpus-tools
# 16 スレッド分を使って並列に数える（openwebtext なら 16 プロセス）
corpus-tools count --corpus openwebtext --tokenizer openai-community/gpt2 --n 1 2 --cpus 16 --output-dir ~/corpus-tools
```

```python
counts = store.counts(preset("openwebtext"), "openai-community/gpt2", [1, 2])
# {1: ndarray[V], 2: csr_matrix[V, V]}
```

数え方の規則、保存先の構成、cache の選び方、並列の仕組みは `AGENTS.md` にある。途中で止めても、次は続きから数える。
