"""
Utility functions for the simple browser tool.
"""

import logging
from typing import Callable, ParamSpec, TypeVar

from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def with_retries(
    func: Callable[P, R],
    num_retries: int,
    max_wait_time: float,
) -> Callable[P, R]:
    if num_retries > 0:
        retry_decorator = retry(
            stop=stop_after_attempt(num_retries),
            wait=wait_exponential(
                multiplier=1,
                min=2,
                max=max_wait_time,
            ),
            before_sleep=before_sleep_log(logger, logging.INFO),
            after=after_log(logger, logging.INFO),
            retry=retry_if_exception_type(Exception),
        )
        return retry_decorator(func)
    else:
        return func


def maybe_truncate(text: str, num_chars: int = 1024) -> str:
    if len(text) > num_chars:
        text = text[: (num_chars - 3)] + "..."
    return text
