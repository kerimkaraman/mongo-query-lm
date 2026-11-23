"""Loss: how surprised a model is by the real next token, averaged over the data."""

import itertools
import math
from collections import defaultdict

from mongo_query_lm.bigram import (
    ProbabilityTable,
    build_token_stream,
    convert_counts_to_probabilities,
    count_bigrams,
)
from mongo_query_lm.generator import OUTPUT_PATH
from mongo_query_lm.tokenizer import END_TOKEN, CharTokenizer, load_filter_texts

EXAMPLE_QUERY_INDEX = 4
SHOWN_PROBABILITIES = (1.0, 0.9, 0.5, 0.31, 0.1, 0.01, 0.001)
HARDEST_CONTEXT_COUNT = 6


def compute_token_losses(
    token_ids: list[int], probabilities: ProbabilityTable
) -> list[float]:
    """One loss per prediction: -log of the probability given to the real next token."""
    losses = []
    for current_id, next_id in itertools.pairwise(token_ids):
        probability = probabilities[current_id][next_id]
        if probability == 0:
            raise ValueError("The model gives zero probability to a real token.")
        losses.append(0.0 - math.log(probability))
    return losses


def compute_average_loss(
    token_ids: list[int], probabilities: ProbabilityTable
) -> float:
    losses = compute_token_losses(token_ids, probabilities)
    return sum(losses) / len(losses)


def build_uniform_probabilities(vocabulary_size: int) -> ProbabilityTable:
    """A model that knows nothing: every token is equally likely, always."""
    return [[1 / vocabulary_size] * vocabulary_size for _ in range(vocabulary_size)]


def print_log_table() -> None:
    print("Probability given to the real next token -> loss (-log p):")
    for probability in SHOWN_PROBABILITIES:
        print(f"  {probability:>6.3f} -> {0.0 - math.log(probability):.2f}")


def print_example_losses(
    text: str, tokenizer: CharTokenizer, probabilities: ProbabilityTable
) -> None:
    token_ids = tokenizer.encode(END_TOKEN + text + END_TOKEN)
    losses = compute_token_losses(token_ids, probabilities)
    print(f"\nBigram on one query: {text!r}")
    print("  previous -> real next   probability   loss")
    for current_id, next_id, loss in zip(
        token_ids, token_ids[1:], losses, strict=False
    ):
        previous = tokenizer.characters[current_id]
        real_next = tokenizer.characters[next_id]
        probability = probabilities[current_id][next_id]
        print(
            f"  {previous!r:>8} -> {real_next!r:<8}  {probability:>10.1%}   {loss:.2f}"
        )
    print(f"  {len(losses)} predictions, average loss {sum(losses) / len(losses):.2f}")


def print_model_comparison(
    token_ids: list[int], uniform: ProbabilityTable, bigram: ProbabilityTable
) -> None:
    print(f"\nAverage loss over the whole dataset ({len(token_ids) - 1} predictions):")
    print(f"  random (uniform) model: {compute_average_loss(token_ids, uniform):.3f}")
    print(f"  counting bigram:        {compute_average_loss(token_ids, bigram):.3f}")


def print_hardest_contexts(
    token_ids: list[int], tokenizer: CharTokenizer, bigram: ProbabilityTable
) -> None:
    losses_by_previous: dict[int, list[float]] = defaultdict(list)
    losses = compute_token_losses(token_ids, bigram)
    for current_id, loss in zip(token_ids, losses, strict=False):
        losses_by_previous[current_id].append(loss)
    averages = {
        token_id: sum(group) / len(group)
        for token_id, group in losses_by_previous.items()
    }
    ranked = sorted(averages.items(), key=lambda pair: pair[1], reverse=True)
    print("\nWhere the bigram struggles most (average loss of the next token):")
    for token_id, average in ranked[:HARDEST_CONTEXT_COUNT]:
        character = tokenizer.characters[token_id]
        count = len(losses_by_previous[token_id])
        print(f"  after {character!r:<5} {average:.2f}  ({count} times)")


def main() -> None:
    texts = load_filter_texts(OUTPUT_PATH)
    tokenizer = CharTokenizer(texts)
    token_ids = build_token_stream(texts, tokenizer)
    bigram = convert_counts_to_probabilities(
        count_bigrams(token_ids, tokenizer.vocabulary_size)
    )
    uniform = build_uniform_probabilities(tokenizer.vocabulary_size)
    print_log_table()
    print_example_losses(texts[EXAMPLE_QUERY_INDEX], tokenizer, bigram)
    print_model_comparison(token_ids, uniform, bigram)
    print_hardest_contexts(token_ids, tokenizer, bigram)


if __name__ == "__main__":
    main()
