import json

from mongo_query_lm.generator import format_filter, generate_queries
from mongo_query_lm.schema import SCHEMA

TEST_SEED = 7
TEST_QUERY_COUNT = 500


def collect_field_names(query_filter: dict) -> set[str]:
    if "$or" in query_filter:
        return {name for branch in query_filter["$or"] for name in branch}
    return set(query_filter)


def test_same_seed_gives_same_queries() -> None:
    assert generate_queries(TEST_QUERY_COUNT, TEST_SEED) == generate_queries(
        TEST_QUERY_COUNT, TEST_SEED
    )


def test_different_seed_gives_different_queries() -> None:
    assert generate_queries(TEST_QUERY_COUNT, TEST_SEED) != generate_queries(
        TEST_QUERY_COUNT, TEST_SEED + 1
    )


def test_generates_requested_count() -> None:
    assert len(generate_queries(TEST_QUERY_COUNT, TEST_SEED)) == TEST_QUERY_COUNT


def test_every_filter_round_trips_through_json() -> None:
    for query in generate_queries(TEST_QUERY_COUNT, TEST_SEED):
        assert json.loads(format_filter(query["filter"])) == query["filter"]


def test_every_field_belongs_to_its_collection() -> None:
    for query in generate_queries(TEST_QUERY_COUNT, TEST_SEED):
        allowed_names = {field.name for field in SCHEMA[query["collection"]]}
        assert collect_field_names(query["filter"]) <= allowed_names


def test_ranges_have_low_below_high() -> None:
    for query in generate_queries(TEST_QUERY_COUNT, TEST_SEED):
        for condition in query["filter"].values():
            if (
                isinstance(condition, dict)
                and "$gte" in condition
                and "$lte" in condition
            ):
                assert condition["$gte"] <= condition["$lte"]
