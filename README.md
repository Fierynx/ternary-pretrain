# ternary-pretrain

`ternary-pretrain` is a pure-PyTorch codebase for deterministic decoder-only language-model
pre-training with floating-point and ternary weight-QAT conditions.

## Install

Python 3.12 and `uv` are required.

```console
uv sync --frozen --dev
uv run ternary-pretrain --help
```

Optional integrations are installed explicitly:

```console
uv sync --frozen --dev --extra transformers
uv sync --frozen --dev --extra wandb
```

## CPU smoke workflow

Commands take explicit TOML paths. Paths inside a TOML file are resolved relative to that file.

```console
uv run ternary-pretrain data prepare --config configs/data/dev.toml
uv run ternary-pretrain tokenizer train --config configs/tokenizers/dev.toml
uv run ternary-pretrain data tokenize --config configs/data/dev.toml
uv run ternary-pretrain train --config configs/runs/cpu_smoke.toml
uv run ternary-pretrain evaluate --config configs/runs/cpu_smoke.toml --checkpoint PATH
uv run ternary-pretrain export --config configs/runs/cpu_smoke.toml --checkpoint PATH --format native
uv run ternary-pretrain inspect --config configs/models/tiny.toml
uv run ternary-pretrain cost report --config PATH
```

Configuration loading is strict: unknown keys, invalid ranges, missing inputs, and incompatible
artifact hashes stop before training starts. Output directories are created beneath `artifacts/`
and are never reused implicitly.

## Run artifacts

Each run contains `run.json`, append-only `metrics.jsonl`, `summary.json`, TensorBoard `events/`,
atomic checkpoints in `checkpoints/`, and explicit exports in `exports/`. The run manifest records
software, source, data, tokenizer, and training identities without copying environment variables.

Use `tensorboard --logdir artifacts/runs` to inspect local event files. If a resumed run is rejected,
compare the checkpoint's config, tokenizer, and dataset hashes with the current inputs; compatibility
checks are intentionally strict.

On Windows, PyTorch 2.13's launcher must select its non-libuv TCP store before starting workers. Set
the thread count explicitly to avoid `torchrun` choosing one implicitly, then use the repository shim:

```powershell
$env:OMP_NUM_THREADS = "1"
uv run python -m ternary_pretrain.torchrun --standalone --nproc-per-node=2 `
  -m ternary_pretrain train --config configs/runs/cpu_smoke.toml
```

On Linux, the equivalent command uses `uv run torchrun` directly.

## FineWeb-Edu preparation

The production data config pins a FineWeb-Edu commit, names two Parquet files, and caps the
document count. Data preparation streams documents to disk instead of loading the corpus into
memory.

```console
uv run ternary-pretrain data prepare --config configs/data/fineweb_edu.toml
uv run ternary-pretrain tokenizer train --config configs/tokenizers/fineweb_edu_32k.toml
uv run ternary-pretrain data tokenize --config configs/data/fineweb_edu.toml
```
