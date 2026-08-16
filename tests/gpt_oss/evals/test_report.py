from gpt_oss.evals.report import map_with_progress


def test_empty_input_returns_without_calling_mapper() -> None:
    called = False

    def fail_if_called(_):
        nonlocal called
        called = True
        raise AssertionError("mapper must not run for an empty sample set")

    assert map_with_progress(fail_if_called, [], pbar=True) == []
    assert called is False


def test_empty_input_without_progress_returns_empty() -> None:
    assert map_with_progress(lambda value: value, [], pbar=False) == []
