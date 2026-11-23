"""Measurements shared by every model: how good are the generated queries?"""

import json


def is_valid_json_object(text: str) -> bool:
    # A bare number like "5" is valid JSON, but only an object can be a find filter.
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict)


def compute_valid_json_rate(texts: list[str]) -> float:
    if not texts:
        raise ValueError("Cannot compute a rate over zero texts.")
    valid_count = sum(is_valid_json_object(text) for text in texts)
    return valid_count / len(texts)
