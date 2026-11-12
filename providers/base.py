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
    ) -> dict:
        """Return the model's next turn as a neutral assistant message."""
        ...


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"


T = TypeVar("T")


def with_retries(
    call: Callable[[], T],
    retry_after: Callable[[Exception], float | None],
    cancel: threading.Event,
    attempts: int = 4,
) -> T:
    """Run `call`, retrying when `retry_after(exc)` returns a wait in seconds.

    Waits are cut short by `cancel`, so a stop during a rate-limit backoff is immediate.
    """
    for attempt in range(1, attempts + 1):
        if cancel.is_set():
            raise Cancelled()
        try:
            return call()
        except Exception as exc:
            wait = retry_after(exc)
            if wait is None or attempt == attempts:
                raise
            if cancel.wait(min(wait, 30)):
                raise Cancelled() from None
    raise AssertionError("unreachable")
