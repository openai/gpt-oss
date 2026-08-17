from gpt_oss.evals.cli_utils import (
    resolve_gpqa_debug_mode,
    resolve_n_repeats,
    resolve_num_examples,
)


def test_explicit_examples_override_debug_default() -> None:
    assert resolve_num_examples(2, debug_mode=True, debug_default=10) == 2


def test_debug_default_is_used_without_explicit_examples() -> None:
    assert resolve_num_examples(None, debug_mode=True, debug_default=10) == 10


def test_full_eval_keeps_unlimited_examples() -> None:
    assert resolve_num_examples(None, debug_mode=False, debug_default=10) is None


def test_explicit_subset_uses_single_repeat() -> None:
    assert resolve_n_repeats(2, debug_mode=False) == 1


def test_debug_run_uses_single_repeat() -> None:
    assert resolve_n_repeats(None, debug_mode=True) == 1


def test_full_run_keeps_eight_repeats() -> None:
    assert resolve_n_repeats(None, debug_mode=False) == 8


def test_zero_examples_keeps_full_run_repeats() -> None:
    assert resolve_n_repeats(0, debug_mode=False) == 8


def test_gpqa_fixed_debug_example_is_default_only() -> None:
    assert resolve_gpqa_debug_mode(None, debug_mode=True) is True
    assert resolve_gpqa_debug_mode(2, debug_mode=True) is False
    assert resolve_gpqa_debug_mode(None, debug_mode=False) is False
