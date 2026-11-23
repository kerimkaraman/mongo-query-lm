# mongo-query-lm

A small language model for MongoDB queries, built step by step to learn how model training works: first a character-level model trained **from scratch**, then a pretrained model **fine-tuned with LoRA** to turn questions into queries.

**Status:** Part A, step 0 done (synthetic data generator). See [docs/roadmap.md](docs/roadmap.md) for the steps and what each one teaches.

## Getting started

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.13 by itself).

```bash
uv sync
```

## Generate the dataset

```bash
uv run python -m mongo_query_lm.generator
```

Writes 5,000 synthetic MongoDB `find` filters over an e-commerce schema (`customers`, `orders`, `products`) to `data/queries.jsonl`, one per line, and prints samples and statistics. The output is fully determined by the seed, so the same command always produces the same data.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```
