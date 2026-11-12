from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ledger  # noqa: E402
from core import Agent  # noqa: E402
from providers import ProviderError  # noqa: E402
from store import Store  # noqa: E402


@pytest.fixture(autouse=True)
def temp_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "DB_PATH", tmp_path / "ledger.db")
    monkeypatch.setattr(ledger.time, "sleep", lambda _s: None)
    return ledger


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "conversations.db")


class FakeProvider:
    """Replies come from a script: each item is a neutral assistant message or a callable."""

    def __init__(self, name: str, script: list):
        self.name = name
        self.model = f"fake-{name}"
        self.script = list(script)
        self.seen: list[list[dict]] = []

    def complete(self, history, system, tools, cancel, notify):
        self.seen.append(history)
        step = self.script.pop(0)
        return step(history, cancel) if callable(step) else step


def say(text: str) -> dict:
    return {"role": "assistant", "text": text, "tool_calls": []}


def call(name: str, call_id: str = "c1", **args) -> dict:
    return {"role": "assistant", "text": "", "tool_calls": [{"id": call_id, "name": name, "args": args}]}


@pytest.fixture
def make_agent(store):
    def make(*scripts: list, events: list | None = None, conversation: str = "t") -> tuple[Agent, dict]:
        fakes = {name: FakeProvider(name, script) for name, script in zip(("a", "b"), scripts)}
        store.open_conversation(conversation, "a")
        log = events if events is not None else []
        def factory(name: str) -> FakeProvider:
            if name not in fakes:
                raise ProviderError(f"unknown provider {name!r}")
            return fakes[name]

        agent = Agent(store, conversation, "a", lambda kind, payload: log.append((kind, payload)), factory)
        return agent, fakes

    return make


@pytest.fixture
def go():
    return threading.Event()
