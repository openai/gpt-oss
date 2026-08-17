def resolve_num_examples(
    explicit_examples: int | None, debug_mode: bool, debug_default: int
) -> int | None:
    if explicit_examples is not None:
        return explicit_examples
    return debug_default if debug_mode else None


def resolve_n_repeats(explicit_examples: int | None, debug_mode: bool) -> int:
    explicit_subset = explicit_examples is not None and explicit_examples != 0
    return 1 if explicit_subset or debug_mode else 8


def resolve_gpqa_debug_mode(explicit_examples: int | None, debug_mode: bool) -> bool:
    return debug_mode and explicit_examples is None
