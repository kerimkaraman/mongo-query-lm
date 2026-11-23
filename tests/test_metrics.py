import pytest

from mongo_query_lm.metrics import compute_valid_json_rate, is_valid_json_object


def test_object_is_valid() -> None:
    assert is_valid_json_object('{"price": {"$gt": 50}}')


def test_broken_json_is_invalid() -> None:
    assert not is_valid_json_object('{"price": {"$gt": 50}')


def test_valid_json_that_is_not_an_object_is_invalid() -> None:
    assert not is_valid_json_object("5")


def test_rate_counts_valid_share() -> None:
    assert compute_valid_json_rate(['{"a": 1}', "{", '{"b": 2}', "x"]) == 0.5


def test_rate_over_no_texts_is_rejected() -> None:
    with pytest.raises(ValueError):
        compute_valid_json_rate([])
