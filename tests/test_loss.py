import math

import pytest

from mongo_query_lm.loss import (
    build_uniform_probabilities,
    compute_average_loss,
    compute_token_losses,
)

VOCABULARY_SIZE = 4


def test_uniform_model_loss_is_log_of_vocabulary_size() -> None:
    uniform = build_uniform_probabilities(VOCABULARY_SIZE)
    assert compute_average_loss([0, 1, 2, 3], uniform) == pytest.approx(
        math.log(VOCABULARY_SIZE)
    )


def test_certain_correct_prediction_has_zero_loss() -> None:
    certain = [[0.0, 1.0], [1.0, 0.0]]
    assert compute_token_losses([0, 1, 0], certain) == [0.0, 0.0]


def test_each_loss_is_negative_log_of_real_token_probability() -> None:
    probabilities = [[0.5, 0.5], [0.25, 0.75]]
    losses = compute_token_losses([0, 1, 1], probabilities)
    assert losses == pytest.approx([-math.log(0.5), -math.log(0.75)])


def test_zero_probability_for_real_token_is_rejected() -> None:
    probabilities = [[1.0, 0.0], [0.5, 0.5]]
    with pytest.raises(ValueError):
        compute_token_losses([0, 1], probabilities)
