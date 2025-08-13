from __future__ import annotations

import time
from typing import Any

import requests
from requests import RequestException, Response


def request_with_retry(
    method: str,
    url: str,
    *,
    max_retries: int = 3,
    backoff_factor: float = 0.5,
    **kwargs: Any,
) -> Response:
    for attempt in range(max_retries):
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except RequestException:
            if attempt == max_retries - 1:
                raise
            sleep_seconds = backoff_factor * (2 ** attempt)
            time.sleep(sleep_seconds)

    raise RuntimeError("request_with_retry failed unexpectedly")
