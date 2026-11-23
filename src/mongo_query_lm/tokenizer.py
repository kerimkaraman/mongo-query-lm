"""Character-level tokenizer: query text to token ids and back."""

import json
from collections import Counter
from pathlib import Path

from mongo_query_lm.generator import OUTPUT_PATH, format_filter

# Never appears inside a JSON filter, so it can safely mark where a query ends.
END_TOKEN = "\n"
VOCABULARY_COLUMNS = 6


class CharTokenizer:
    """Maps each character seen in the training texts to an integer id."""

    def __init__(self, texts: list[str]) -> None:
        if any(END_TOKEN in text for text in texts):
            raise ValueError("Training texts must not contain the end token.")
        # Sorted, so the same texts always give the same ids.
        self.characters = sorted(set("".join(texts)) | {END_TOKEN})
        self.id_by_character = {
            character: token_id for token_id, character in enumerate(self.characters)
        }

    @property
    def vocabulary_size(self) -> int:
        return len(self.characters)

    def encode(self, text: str) -> list[int]:
        unknown_characters = set(text) - self.id_by_character.keys()
        if unknown_characters:
            raise ValueError(f"Characters not in vocabulary: {unknown_characters}")
        return [self.id_by_character[character] for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.characters[token_id] for token_id in token_ids)


def load_filter_texts(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: uv run python -m mongo_query_lm.generator"
        )
    lines = path.read_text().splitlines()
    return [format_filter(json.loads(line)["filter"]) for line in lines]


def print_vocabulary(tokenizer: CharTokenizer, texts: list[str]) -> None:
    counts = Counter("".join(texts))
    counts[END_TOKEN] = len(texts)
    print(f"Vocabulary size: {tokenizer.vocabulary_size}  (id: character x count)")
    cells = [
        f"{token_id:>2}: {character!r:<5} x{counts[character]:<6}"
        for token_id, character in enumerate(tokenizer.characters)
    ]
    for start in range(0, len(cells), VOCABULARY_COLUMNS):
        print("  " + "  ".join(cells[start : start + VOCABULARY_COLUMNS]))


def print_example(tokenizer: CharTokenizer, text: str) -> None:
    token_ids = tokenizer.encode(text + END_TOKEN)
    print(f"\nText:      {text!r}")
    print(f"Tokens:    {list(text + END_TOKEN)}")
    print(f"Token ids: {token_ids}")
    print(f"Decoded:   {tokenizer.decode(token_ids)!r}")
    print(f"Length:    {len(text)} characters + 1 end token = {len(token_ids)} tokens")


def print_round_trip_check(tokenizer: CharTokenizer, texts: list[str]) -> None:
    is_every_text_restored = all(
        tokenizer.decode(tokenizer.encode(text)) == text for text in texts
    )
    total_tokens = sum(len(text) + 1 for text in texts)
    print(
        f"\nAll {len(texts)} queries survive encode -> decode: {is_every_text_restored}"
    )
    print(f"Total tokens in the dataset (with end tokens): {total_tokens}")


def main() -> None:
    texts = load_filter_texts(OUTPUT_PATH)
    tokenizer = CharTokenizer(texts)
    print_vocabulary(tokenizer, texts)
    print_example(tokenizer, texts[4])
    print_round_trip_check(tokenizer, texts)


if __name__ == "__main__":
    main()
