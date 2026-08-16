def resolve_num_examples(
    explicit_examples: int | None, debug_mode: bool, debug_default: int
) -> int | None:
    if explicit_examples is not None:
        return explicit_examples
    return debug_default if debug_mode else None
