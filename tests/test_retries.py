from __future__ import annotations

import threading

import pytest

from providers.base import Cancelled, ProviderError, with_retries


class Busy(Exception):
    code = 503


class ReadTimeout(Exception):
    pass


ReadTimeout.__name__ = "TimeoutException"


def flaky(failures: list[Exception]):
    def call():
        if failures:
            raise failures.pop(0)
        return "ok"

    return call


def test_retries_with_backoff_and_tells_the_person(monkeypatch):
    waits, notes = [], []
    cancel = threading.Event()
    monkeypatch.setattr(cancel, "wait", lambda t: waits.append(t) or False)
    result = with_retries(flaky([Busy(), Busy(), ReadTimeout()]), lambda e: 3.0, cancel, notes.append, "gemini")
    assert result == "ok"
    assert waits == [3.0, 6.0, 4.0]
    assert notes[0] == "gemini error 503; retrying in 3s (attempt 2 of 5)"
    assert "timed out" in notes[2]


def test_gives_up_with_a_clear_error(monkeypatch):
    cancel = threading.Event()
    monkeypatch.setattr(cancel, "wait", lambda t: False)
    with pytest.raises(ProviderError, match="gave up after 5 tries"):
        with_retries(flaky([Busy()] * 5), lambda e: 3.0, cancel, lambda _: None, "gemini")


def test_stop_during_backoff_is_immediate():
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        with_retries(flaky([Busy()]), lambda e: 3.0, cancel, lambda _: None, "gemini")


def test_non_retryable_errors_pass_through():
    with pytest.raises(ValueError):
        with_retries(flaky([ValueError("bad")]), lambda e: None, threading.Event(), lambda _: None, "x")
