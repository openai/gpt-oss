def resolve_num_examples(
    explicit_examples: int | None, debug_mode: bool, debug_default: int
) -> int | None:
    if explicit_examples is not None:
        return explicit_examples
    return debug_default if debug_mode else None


def resolve_n_repeats(explicit_examples: int | None, debug_mode: bool) -> int:
    return 1 if explicit_examples is not None or debug_mode else 8
