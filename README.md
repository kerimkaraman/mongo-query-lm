# mongo-query-lm

A small language model for MongoDB queries, built step by step to learn how model training works: first a character-level model trained **from scratch**, then a pretrained model **fine-tuned with LoRA** to turn questions into queries.

**Status:** Part A, steps 0-3 done (synthetic data generator, character-level tokenizer, counting bigram baseline, loss). See [docs/roadmap.md](docs/roadmap.md) for the steps and what each one teaches.

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

## Inspect the tokenizer

```bash
uv run python -m mongo_query_lm.tokenizer
```

Builds the character-level vocabulary from the generated data and prints it, one example query encoded into token ids and decoded back, and a round-trip check over the whole dataset.

## Counting bigram baseline

```bash
uv run python -m mongo_query_lm.bigram
```

Counts which character follows which in the generated data, samples 1,000 new queries from those probabilities, and measures their valid JSON rate. Result: 2.8% (the training data is 100%). This is the baseline every later model must beat. See [docs/concepts/bigram.md](docs/concepts/bigram.md) for the idea behind it.

## Loss

```bash
uv run python -m mongo_query_lm.loss
```

Measures how surprised a model is by the real next character (average negative log likelihood over the dataset). Uniform random model: 3.989; counting bigram: 1.626. Also prints the loss of every prediction in one query and where the bigram struggles most.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```
