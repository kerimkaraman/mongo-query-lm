# mongo-query-lm

A small language model for MongoDB queries, built step by step to learn how model training works: first a character-level model trained **from scratch**, then a pretrained model **fine-tuned with LoRA** to turn questions into queries.

**Status:** just started. See [docs/roadmap.md](docs/roadmap.md) for the steps and what each one teaches.

## Getting started

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.13 by itself).

```bash
uv sync
```
