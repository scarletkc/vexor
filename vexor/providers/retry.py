"""Retry classification and backoff shared by remote embedding adapters."""

RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
MAX_RETRIES = 2
RETRY_BASE_DELAY = 0.5
RETRY_MAX_DELAY = 4.0


def backoff_delay(attempt: int) -> float:
    return min(RETRY_MAX_DELAY, RETRY_BASE_DELAY * (2**attempt))


def extract_status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "status", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int):
            return value
    return None


def should_retry_error(exc: Exception) -> bool:
    status = extract_status_code(exc)
    if status in RETRYABLE_STATUS_CODES:
        return True
    name = exc.__class__.__name__.lower()
    if "ratelimit" in name or "timeout" in name or "temporarily" in name:
        return True
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "rate limit",
            "timeout",
            "temporar",
            "overload",
            "try again",
            "too many requests",
            "service unavailable",
        )
    )
