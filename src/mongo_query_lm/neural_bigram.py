"""Neural bigram: learn the next-character table by gradient descent instead of counting."""

import time

import torch
import torch.nn.functional as F

from mongo_query_lm.bigram import (
    GENERATED_QUERY_COUNT,
    SAMPLING_SEED,
    build_token_stream,
    convert_counts_to_probabilities,
    count_bigrams,
    generate_queries,
)
from mongo_query_lm.generator import OUTPUT_PATH
from mongo_query_lm.metrics import compute_valid_json_rate
from mongo_query_lm.tokenizer import CharTokenizer, load_filter_texts

WEIGHT_SEED = 42
LEARNING_RATE = 50.0
ADAM_LEARNING_RATE = 0.3
TRAINING_STEPS = 300
LOG_EVERY_STEPS = 50
TRACED_STEPS = (0, 1, 2, 3, 10, 50, 100, 299)
INSPECTED_CHARACTER = "$"
TRACED_NEXT_CHARACTER = "g"
SHOWN_NEXT_CHARACTERS = 5


def build_training_pairs(token_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Inputs are every token; targets are the token that really came next."""
    stream = torch.tensor(token_ids)
    return stream[:-1], stream[1:]


def initialize_weights(vocabulary_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    # requires_grad tells autograd to track how the loss depends on these numbers.
    return torch.randn(
        (vocabulary_size, vocabulary_size), generator=generator, requires_grad=True
    )


def compute_logits(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    # logits: raw scores per next token, before they are turned into probabilities.
    one_hot = F.one_hot(inputs, num_classes=weights.shape[0]).float()
    return one_hot @ weights


def compute_loss_by_hand(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Softmax, then -log of the real next token's probability: the step 3 loss."""
    probabilities = logits.exp() / logits.exp().sum(dim=1, keepdim=True)
    real_token_probabilities = probabilities[torch.arange(len(targets)), targets]
    return -real_token_probabilities.log().mean()


def print_traced_weight(
    step: int, loss: torch.Tensor, weights: torch.Tensor, traced: tuple[int, int]
) -> None:
    row_id, column_id = traced
    probability = F.softmax(weights[row_id].detach(), dim=0)[column_id].item()
    print(
        f"  step {step:>3}  loss {loss.item():.4f}  "
        f"weight {weights[row_id, column_id].item():+.3f}  "
        f"gradient {weights.grad[row_id, column_id].item():+.5f}  "
        f"probability {probability:.1%}"
    )


def train_with_gradient_descent(
    weights: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    traced: tuple[int, int],
) -> list[float]:
    losses = []
    for step in range(TRAINING_STEPS):
        loss = F.cross_entropy(compute_logits(inputs, weights), targets)
        weights.grad = None
        loss.backward()
        if step in TRACED_STEPS:
            print_traced_weight(step, loss, weights, traced)
        # Gradient descent: move every weight a little against its gradient.
        with torch.no_grad():
            weights -= LEARNING_RATE * weights.grad
        losses.append(loss.item())
    return losses


def train_with_adam(
    weights: torch.Tensor, inputs: torch.Tensor, targets: torch.Tensor
) -> list[float]:
    # Adam adapts the step size of every weight, so rarely seen rows still learn fast.
    optimizer = torch.optim.Adam([weights], lr=ADAM_LEARNING_RATE)
    losses = []
    for _ in range(TRAINING_STEPS):
        loss = F.cross_entropy(compute_logits(inputs, weights), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def print_loss_curves(descent_losses: list[float], adam_losses: list[float]) -> None:
    print("\n  step   gradient descent   Adam")
    shown_steps = [*range(0, TRAINING_STEPS, LOG_EVERY_STEPS), TRAINING_STEPS - 1]
    for step in shown_steps:
        print(f"  {step:>4}   {descent_losses[step]:>16.4f}   {adam_losses[step]:.4f}")


def print_shapes(
    inputs: torch.Tensor, weights: torch.Tensor, targets: torch.Tensor
) -> None:
    logits = compute_logits(inputs, weights)
    print("Shapes:")
    print(f"  inputs  {list(inputs.shape)}  (one character id per prediction)")
    print(f"  one-hot {list(F.one_hot(inputs).shape)}  (each id as a 0/1 vector)")
    print(f"  weights {list(weights.shape)}  (the table the network learns)")
    print(f"  logits  {list(logits.shape)}  (a score for every possible next char)")
    print(f"  targets {list(targets.shape)}  (the character that really came next)")


def print_loss_check(
    inputs: torch.Tensor, weights: torch.Tensor, targets: torch.Tensor
) -> None:
    with torch.no_grad():
        logits = compute_logits(inputs, weights)
        by_hand = compute_loss_by_hand(logits, targets).item()
        built_in = F.cross_entropy(logits, targets).item()
    print("\nLoss with random weights, before training:")
    print(f"  by hand (softmax + -log): {by_hand:.4f}")
    print(f"  F.cross_entropy:          {built_in:.4f}")


def print_row_comparison(
    learned: torch.Tensor, counted: torch.Tensor, tokenizer: CharTokenizer
) -> None:
    row_id = tokenizer.encode(INSPECTED_CHARACTER)[0]
    ranked_ids = counted[row_id].argsort(descending=True)[:SHOWN_NEXT_CHARACTERS]
    print(f"\nAfter {INSPECTED_CHARACTER!r}:   counted   learned")
    for token_id in ranked_ids.tolist():
        character = tokenizer.characters[token_id]
        print(
            f"  {character!r:<6}  {counted[row_id, token_id]:>7.1%}"
            f"   {learned[row_id, token_id]:>7.1%}"
        )
    largest_gap = (learned - counted).abs().max().item()
    print(f"Largest gap anywhere in the 54 x 54 table: {largest_gap:.1%}")


def run_both_trainings(
    inputs: torch.Tensor, targets: torch.Tensor, tokenizer: CharTokenizer
) -> torch.Tensor:
    """Train the same starting weights twice and return the Adam-trained ones."""
    traced = tuple(tokenizer.encode(INSPECTED_CHARACTER + TRACED_NEXT_CHARACTER))
    print(f"\nGradient descent, learning rate {LEARNING_RATE}, tracing one weight:")
    print(f"  ({INSPECTED_CHARACTER!r} -> {TRACED_NEXT_CHARACTER!r}, counted: 31.2%)")
    start = time.perf_counter()
    descent_weights = initialize_weights(tokenizer.vocabulary_size, WEIGHT_SEED)
    descent_losses = train_with_gradient_descent(
        descent_weights, inputs, targets, traced
    )
    adam_weights = initialize_weights(tokenizer.vocabulary_size, WEIGHT_SEED)
    adam_losses = train_with_adam(adam_weights, inputs, targets)
    print_loss_curves(descent_losses, adam_losses)
    print(f"  both trainings took {time.perf_counter() - start:.1f} s")
    return adam_weights


def main() -> None:
    texts = load_filter_texts(OUTPUT_PATH)
    tokenizer = CharTokenizer(texts)
    token_ids = build_token_stream(texts, tokenizer)
    inputs, targets = build_training_pairs(token_ids)
    weights = initialize_weights(tokenizer.vocabulary_size, WEIGHT_SEED)
    print_shapes(inputs, weights, targets)
    print_loss_check(inputs, weights, targets)

    adam_weights = run_both_trainings(inputs, targets, tokenizer)
    learned = F.softmax(adam_weights.detach(), dim=1)
    counted = torch.tensor(
        convert_counts_to_probabilities(
            count_bigrams(token_ids, tokenizer.vocabulary_size)
        )
    )
    print_row_comparison(learned, counted, tokenizer)
    generated = generate_queries(
        learned.tolist(), tokenizer, GENERATED_QUERY_COUNT, SAMPLING_SEED
    )
    print(f"\nValid JSON rate, neural bigram: {compute_valid_json_rate(generated):.1%}")


if __name__ == "__main__":
    main()
