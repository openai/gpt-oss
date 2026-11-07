import numpy as np

from gpt_oss.evals.healthbench_eval import _compute_clipped_stats


def test_bootstrap_std_is_independent_of_global_numpy_seed() -> None:
    values = [0.0, 0.25, 0.5, 1.0]

    np.random.seed(1)
    first = _compute_clipped_stats(values, "bootstrap_std")
    np.random.seed(999)
    second = _compute_clipped_stats(values, "bootstrap_std")

    assert first == second


def test_bootstrap_std_does_not_advance_global_numpy_rng() -> None:
    values = [0.0, 0.25, 0.5, 1.0]

    np.random.seed(123)
    expected_next = np.random.random()

    np.random.seed(123)
    _compute_clipped_stats(values, "bootstrap_std")
    actual_next = np.random.random()

    assert actual_next == expected_next


def test_bootstrap_std_is_independent_of_input_order() -> None:
    values = [0.0, 0.25, 0.5, 1.0, 0.75]

    first = _compute_clipped_stats(values, "bootstrap_std")
    second = _compute_clipped_stats([1.0, 0.5, 0.0, 0.75, 0.25], "bootstrap_std")

    assert first == second
