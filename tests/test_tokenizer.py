import pytest

from mongo_query_lm.generator import format_filter, generate_queries
from mongo_query_lm.tokenizer import END_TOKEN, CharTokenizer

TEST_SEED = 7
TEST_QUERY_COUNT = 200


@pytest.fixture
def texts() -> list[str]:
    queries = generate_queries(TEST_QUERY_COUNT, TEST_SEED)
    return [format_filter(query["filter"]) for query in queries]


def test_encode_then_decode_restores_every_text(texts: list[str]) -> None:
    tokenizer = CharTokenizer(texts)
    for text in texts:
        assert tokenizer.decode(tokenizer.encode(text + END_TOKEN)) == text + END_TOKEN


def test_vocabulary_is_every_character_plus_end_token(texts: list[str]) -> None:
    tokenizer = CharTokenizer(texts)
    assert set(tokenizer.characters) == set("".join(texts)) | {END_TOKEN}


def test_same_texts_give_same_ids(texts: list[str]) -> None:
    assert (
        CharTokenizer(texts).characters
        == CharTokenizer(list(reversed(texts))).characters
    )


def test_encode_rejects_unknown_character() -> None:
    tokenizer = CharTokenizer(['{"age": 1}'])
    with pytest.raises(ValueError):
        tokenizer.encode("€")


def test_training_text_with_end_token_is_rejected() -> None:
    with pytest.raises(ValueError):
        CharTokenizer(['{"age": 1}' + END_TOKEN])
