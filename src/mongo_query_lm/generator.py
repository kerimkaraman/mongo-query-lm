"""Generate synthetic MongoDB find filters from the schema."""

import json
import random
import statistics
from collections import Counter
from pathlib import Path

from mongo_query_lm.schema import SCHEMA, Field, FieldKind

SEED = 42
QUERY_COUNT = 5000
MAX_CONDITIONS = 3
OR_PROBABILITY = 0.15
IN_LIST_MAX_SIZE = 3
DECIMAL_PLACES = 2
SAMPLE_COUNT = 8
OUTPUT_PATH = Path("data/queries.jsonl")

COMPARISON_OPERATORS = ("$gt", "$gte", "$lt", "$lte")

# Which condition shapes make sense for each kind of field.
# Exact equality on a decimal (price == 412.37) is unrealistic, so decimals only get ranges.
SHAPES_BY_KIND = {
    FieldKind.STRING: ("equals", "not_equals", "in_list"),
    FieldKind.INTEGER: ("equals", "comparison", "range"),
    FieldKind.DECIMAL: ("comparison", "range"),
    FieldKind.BOOLEAN: ("equals",),
}

Scalar = str | int | float | bool


def pick_value(field: Field, rng: random.Random) -> Scalar:
    if field.kind is FieldKind.STRING:
        return rng.choice(field.choices)
    if field.kind is FieldKind.INTEGER:
        return rng.randint(int(field.minimum), int(field.maximum))
    if field.kind is FieldKind.DECIMAL:
        return round(rng.uniform(field.minimum, field.maximum), DECIMAL_PLACES)
    return rng.choice((True, False))


def build_condition(field: Field, rng: random.Random) -> Scalar | dict:
    """Build what goes after the field name, e.g. 50 or {"$gt": 50}."""
    shape = rng.choice(SHAPES_BY_KIND[field.kind])
    if shape == "equals":
        return pick_value(field, rng)
    if shape == "not_equals":
        return {"$ne": pick_value(field, rng)}
    if shape == "in_list":
        size = rng.randint(2, IN_LIST_MAX_SIZE)
        return {"$in": rng.sample(field.choices, size)}
    if shape == "comparison":
        return {rng.choice(COMPARISON_OPERATORS): pick_value(field, rng)}
    low, high = sorted((pick_value(field, rng), pick_value(field, rng)))
    return {"$gte": low, "$lte": high}


def build_and_filter(fields: tuple[Field, ...], rng: random.Random) -> dict:
    """Several fields in one document: MongoDB treats them as AND."""
    condition_count = rng.randint(1, MAX_CONDITIONS)
    chosen_fields = rng.sample(fields, condition_count)
    return {field.name: build_condition(field, rng) for field in chosen_fields}


def build_or_filter(fields: tuple[Field, ...], rng: random.Random) -> dict:
    first_field, second_field = rng.sample(fields, 2)
    return {
        "$or": [
            {first_field.name: build_condition(first_field, rng)},
            {second_field.name: build_condition(second_field, rng)},
        ]
    }


def generate_query(rng: random.Random) -> dict:
    collection = rng.choice(sorted(SCHEMA))
    fields = SCHEMA[collection]
    if rng.random() < OR_PROBABILITY:
        query_filter = build_or_filter(fields, rng)
    else:
        query_filter = build_and_filter(fields, rng)
    return {"collection": collection, "filter": query_filter}


def generate_queries(count: int, seed: int) -> list[dict]:
    # One seeded generator for the whole run: same seed, same queries, every time.
    rng = random.Random(seed)
    return [generate_query(rng) for _ in range(count)]


def format_filter(query_filter: dict) -> str:
    """The exact text the model will learn to write."""
    return json.dumps(query_filter)


def save_queries(queries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(query) for query in queries]
    path.write_text("\n".join(lines) + "\n")


def count_operators(query_filter: dict, counter: Counter) -> None:
    for key, condition in query_filter.items():
        if key.startswith("$"):
            counter[key] += 1
        elif not isinstance(condition, dict):
            counter["(equals)"] += 1
        if isinstance(condition, dict):
            count_operators(condition, counter)
        if isinstance(condition, list):
            for item in condition:
                if isinstance(item, dict):
                    count_operators(item, counter)


def print_samples(queries: list[dict]) -> None:
    print(f"\nFirst {SAMPLE_COUNT} queries:")
    for query in queries[:SAMPLE_COUNT]:
        print(f"  {query['collection']:<10} {format_filter(query['filter'])}")


def print_distributions(queries: list[dict]) -> None:
    collections = Counter(query["collection"] for query in queries)
    print(f"\nCollections: {dict(collections.most_common())}")
    operators: Counter = Counter()
    for query in queries:
        count_operators(query["filter"], operators)
    print(f"Operators:   {dict(operators.most_common())}")


def print_text_stats(queries: list[dict]) -> None:
    texts = [format_filter(query["filter"]) for query in queries]
    lengths = [len(text) for text in texts]
    print(f"\nUnique filters: {len(set(texts))} of {len(texts)}")
    print(
        f"Length in characters: min {min(lengths)}, "
        f"mean {statistics.mean(lengths):.1f}, max {max(lengths)}"
    )


def print_reproducibility_check(queries: list[dict]) -> None:
    is_same_with_same_seed = generate_queries(QUERY_COUNT, SEED) == queries
    is_same_with_other_seed = generate_queries(QUERY_COUNT, SEED + 1) == queries
    print(f"\nSeed {SEED} again gives the same queries:  {is_same_with_same_seed}")
    print(f"Seed {SEED + 1} gives the same queries:       {is_same_with_other_seed}")


def main() -> None:
    queries = generate_queries(QUERY_COUNT, SEED)
    save_queries(queries, OUTPUT_PATH)
    print(f"Wrote {len(queries)} queries to {OUTPUT_PATH} (seed {SEED})")
    print_samples(queries)
    print_distributions(queries)
    print_text_stats(queries)
    print_reproducibility_check(queries)


if __name__ == "__main__":
    main()
