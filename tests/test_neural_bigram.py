import torch
import torch.nn.functional as F

from mongo_query_lm.neural_bigram import (
    build_training_pairs,
    compute_logits,
    compute_loss_by_hand,
    initialize_weights,
)

VOCABULARY_SIZE = 5
TEST_SEED = 0


def test_targets_are_inputs_shifted_by_one() -> None:
    inputs, targets = build_training_pairs([3, 1, 4, 1])
    assert inputs.tolist() == [3, 1, 4]
    assert targets.tolist() == [1, 4, 1]


def test_one_hot_times_weights_selects_a_row() -> None:
    weights = initialize_weights(VOCABULARY_SIZE, TEST_SEED)
    logits = compute_logits(torch.tensor([2, 0]), weights)
    assert torch.equal(logits[0], weights[2])
    assert torch.equal(logits[1], weights[0])


def test_loss_by_hand_matches_cross_entropy() -> None:
    weights = initialize_weights(VOCABULARY_SIZE, TEST_SEED)
    inputs, targets = torch.tensor([0, 3, 4, 1]), torch.tensor([3, 4, 1, 2])
    logits = compute_logits(inputs, weights)
    assert torch.allclose(
        compute_loss_by_hand(logits, targets), F.cross_entropy(logits, targets)
    )


def test_same_seed_gives_same_weights() -> None:
    assert torch.equal(
        initialize_weights(VOCABULARY_SIZE, TEST_SEED),
        initialize_weights(VOCABULARY_SIZE, TEST_SEED),
    )
