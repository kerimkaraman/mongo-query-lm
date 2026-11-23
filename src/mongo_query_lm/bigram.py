"""Counting bigram model: which character follows which, turned into probabilities."""

import itertools
import random

from mongo_query_lm.generator import OUTPUT_PATH
from mongo_query_lm.metrics import compute_valid_json_rate, is_valid_json_object
from mongo_query_lm.tokenizer import END_TOKEN, CharTokenizer, load_filter_texts

SAMPLING_SEED = 1337
GENERATED_QUERY_COUNT = 1000
MAX_GENERATED_LENGTH = 300
SHOWN_SAMPLE_COUNT = 10
TOP_NEXT_CHARACTERS = 5
INSPECTED_CHARACTERS = ("{", "$", "}", END_TOKEN)

ProbabilityTable = list[list[float]]


def build_token_stream(texts: list[str], tokenizer: CharTokenizer) -> list[int]:
    """All queries in one sequence, each preceded by the end token of the previous one."""
    stream = END_TOKEN + "".join(text + END_TOKEN for text in texts)
    return tokenizer.encode(stream)


def count_bigrams(token_ids: list[int], vocabulary_size: int) -> list[list[int]]:
    counts = [[0] * vocabulary_size for _ in range(vocabulary_size)]
    for current_id, next_id in itertools.pairwise(token_ids):
        counts[current_id][next_id] += 1
    return counts


def convert_counts_to_probabilities(counts: list[list[int]]) -> ProbabilityTable:
    probabilities = []
    for row in counts:
        row_total = sum(row)
        if row_total == 0:
            raise ValueError("A token is never followed by anything; cannot normalize.")
        probabilities.append([count / row_total for count in row])
    return probabilities


def sample_query(
    probabilities: ProbabilityTable, tokenizer: CharTokenizer, rng: random.Random
) -> str:
    end_id = tokenizer.encode(END_TOKEN)[0]
    token_ids = list(range(tokenizer.vocabulary_size))
    current_id = end_id
    generated_ids = []
    # The length cap stops a query that never produces the end token.
    while len(generated_ids) < MAX_GENERATED_LENGTH:
        current_id = rng.choices(token_ids, weights=probabilities[current_id])[0]
        if current_id == end_id:
            break
        generated_ids.append(current_id)
    return tokenizer.decode(generated_ids)


def generate_queries(
    probabilities: ProbabilityTable, tokenizer: CharTokenizer, count: int, seed: int
) -> list[str]:
    rng = random.Random(seed)
    return [sample_query(probabilities, tokenizer, rng) for _ in range(count)]


def print_table_rows(probabilities: ProbabilityTable, tokenizer: CharTokenizer) -> None:
    print("Most likely next characters:")
    for character in INSPECTED_CHARACTERS:
        row = probabilities[tokenizer.encode(character)[0]]
        ranked = sorted(enumerate(row), key=lambda pair: pair[1], reverse=True)
        cells = [
            f"{tokenizer.characters[token_id]!r} {probability:.0%}"
            for token_id, probability in ranked[:TOP_NEXT_CHARACTERS]
            if probability > 0
        ]
        print(f"  {character!r:<5} -> " + "   ".join(cells))


def print_samples(generated: list[str]) -> None:
    print(f"\nFirst {SHOWN_SAMPLE_COUNT} generated queries (valid JSON object?):")
    for text in generated[:SHOWN_SAMPLE_COUNT]:
        mark = "ok " if is_valid_json_object(text) else "bad"
        print(f"  [{mark}] {text}")


def print_measurements(generated: list[str], training_texts: list[str]) -> None:
    valid_texts = [text for text in generated if is_valid_json_object(text)]
    new_valid_texts = set(valid_texts) - set(training_texts)
    truncated_count = sum(len(text) == MAX_GENERATED_LENGTH for text in generated)
    print(f"\nGenerated {len(generated)} queries (sampling seed {SAMPLING_SEED})")
    print(
        f"Valid JSON rate, training data: {compute_valid_json_rate(training_texts):.1%}"
    )
    print(f"Valid JSON rate, bigram:        {compute_valid_json_rate(generated):.1%}")
    print(f"Valid and not in training data: {len(new_valid_texts)} distinct queries")
    print(f"Hit the {MAX_GENERATED_LENGTH}-character cap:     {truncated_count}")


def main() -> None:
    texts = load_filter_texts(OUTPUT_PATH)
    tokenizer = CharTokenizer(texts)
    counts = count_bigrams(
        build_token_stream(texts, tokenizer), tokenizer.vocabulary_size
    )
    probabilities = convert_counts_to_probabilities(counts)
    print_table_rows(probabilities, tokenizer)
    generated = generate_queries(
        probabilities, tokenizer, GENERATED_QUERY_COUNT, SAMPLING_SEED
    )
    print_samples(generated)
    print_measurements(generated, texts)


if __name__ == "__main__":
    main()
