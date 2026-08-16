from gpt_oss.evals.cli_utils import resolve_num_examples


def test_explicit_examples_override_debug_default() -> None:
    assert resolve_num_examples(2, debug_mode=True, debug_default=10) == 2


def test_debug_default_is_used_without_explicit_examples() -> None:
    assert resolve_num_examples(None, debug_mode=True, debug_default=10) == 10


def test_full_eval_keeps_unlimited_examples() -> None:
    assert resolve_num_examples(None, debug_mode=False, debug_default=10) is None
