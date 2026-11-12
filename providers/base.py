"""What every provider connector implements, plus helpers they share.

The neutral history is a list of plain dicts, the only shape the rest of the app sees:

    {"role": "user", "text": "..."}
    {"role": "assistant", "text": "...",
     "tool_calls": [{"id": "call_ab12", "name": "read_invoice", "args": {...}}],
     "meta": {"<provider>": {...}}}          # optional, opaque to everyone else
    {"role": "tool", "call_id": "call_ab12", "name": "read_invoice", "result": {...}}

A connector turns that history into its SDK's request, calls the model, and turns the
reply back into one neutral assistant message. `meta` lets a connector keep data only it
understands (e.g. Gemini's thought signatures) without leaking it into other providers.
"""

from __future__ import annotations

import threading
import uuid
from typing import Callable, Protocol, Sequence, TypeVar


class Cancelled(Exception):
    """The person stopped the agent."""


class ProviderError(Exception):
    """The provider failed in a way retrying won't fix."""


class ToolSpec(Protocol):
    name: str
    description: str
    parameters: dict


class Provider(Protocol):
    name: str  # registry key, e.g. "gemini"
    model: str

    def complete(
        self,
        history: list[dict],
        system: str,
        tools: Sequence[ToolSpec],
        cancel: threading.Event,
        notify: Callable[[str], None],
    ) -> dict:
        """Return the model's next turn as a neutral assistant message.

        `notify` tells the person about delays (retries, rate limits) as they happen.
        """
        ...


# Per-request limit for hosted APIs, so a stalled request becomes a visible retry.
REQUEST_TIMEOUT_S = 45


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"


T = TypeVar("T")


def with_retries(
    call: Callable[[], T],
    retry_after: Callable[[Exception], float | None],
    cancel: threading.Event,
    notify: Callable[[str], None],
    label: str,
    attempts: int = 5,
) -> T:
    """Run `call`, retrying when `retry_after(exc)` returns a base wait in seconds.

    The wait doubles on each attempt (capped at 30s), and request timeouts are always
    retried. Waits are cut short by `cancel`, so a stop during a backoff is immediate.
    """
    for attempt in range(1, attempts + 1):
        if cancel.is_set():
            raise Cancelled()
        try:
            return call()
        except Exception as exc:
            if cancel.is_set():
                raise Cancelled() from None
            timed_out = _is_timeout(exc)
            base = 1.0 if timed_out else retry_after(exc)
            reason = "timed out" if timed_out else _describe(exc)
            if base is None:
                raise
            wait = min(base * 2 ** (attempt - 1), 30)
            if attempt == attempts:
                raise ProviderError(f"{label} {reason}; gave up after {attempts} tries") from exc
            notify(f"{label} {reason}; retrying in {wait:g}s (attempt {attempt + 1} of {attempts})")
            if cancel.wait(wait):
                raise Cancelled() from None
    raise AssertionError("unreachable")


def _is_timeout(exc: Exception) -> bool:
    # The SDKs bundle different HTTP clients (httpx, google-genai's httpx2), so match by name.
    return isinstance(exc, TimeoutError) or any(
        cls.__name__ in ("TimeoutException", "APITimeoutError") for cls in type(exc).__mro__
    )


def _describe(exc: Exception) -> str:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return f"error {code}" if code else type(exc).__name__
