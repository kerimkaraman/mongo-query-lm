import pytest

from mongo_query_lm.bigram import (
    build_token_stream,
    convert_counts_to_probabilities,
    count_bigrams,
    generate_queries,
)
from mongo_query_lm.tokenizer import END_TOKEN, CharTokenizer

TEXTS = ['{"a": 1}', '{"b": 2}']
TEST_SEED = 3
TEST_QUERY_COUNT = 20


@pytest.fixture
def tokenizer() -> CharTokenizer:
    return CharTokenizer(TEXTS)


def test_stream_starts_and_separates_queries_with_end_token(
    tokenizer: CharTokenizer,
) -> None:
    stream = tokenizer.decode(build_token_stream(TEXTS, tokenizer))
    assert stream == END_TOKEN + TEXTS[0] + END_TOKEN + TEXTS[1] + END_TOKEN


def test_counts_every_consecutive_pair(tokenizer: CharTokenizer) -> None:
    token_ids = tokenizer.encode('{"a"}')
    counts = count_bigrams(token_ids, tokenizer.vocabulary_size)
    quote_id, letter_id = tokenizer.encode('"a')
    assert counts[quote_id][letter_id] == 1
    assert sum(map(sum, counts)) == len(token_ids) - 1


def test_probability_rows_sum_to_one(tokenizer: CharTokenizer) -> None:
    counts = count_bigrams(
        build_token_stream(TEXTS, tokenizer), tokenizer.vocabulary_size
    )
    for row in convert_counts_to_probabilities(counts):
        assert sum(row) == pytest.approx(1.0)


def test_same_seed_gives_same_samples(tokenizer: CharTokenizer) -> None:
    counts = count_bigrams(
        build_token_stream(TEXTS, tokenizer), tokenizer.vocabulary_size
    )
    probabilities = convert_counts_to_probabilities(counts)
    first = generate_queries(probabilities, tokenizer, TEST_QUERY_COUNT, TEST_SEED)
    second = generate_queries(probabilities, tokenizer, TEST_QUERY_COUNT, TEST_SEED)
    assert first == second
    assert all(END_TOKEN not in text for text in first)
