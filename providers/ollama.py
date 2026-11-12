"""Ollama connector (SDK: ollama). Runs a local model; no API key."""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Callable, Sequence

import ollama

from providers.base import ProviderError, ToolSpec, new_call_id, with_retries

_THINK = re.compile(r"^.*?</think>|<think>.*?</think>", re.DOTALL)


class OllamaProvider:
    name = "ollama"
    env_key = None
    default_model = "qwen3:4b"

    def __init__(self, api_key: str | None, model: str):
        self.model = model
        self._client = ollama.Client(host=os.environ.get("OLLAMA_HOST"))
        # Reasoning models put their thinking in a separate field when asked to think;
        # with think=False some (qwen3) leak it into the reply instead.
        self._think: bool | None = True

    def complete(
        self,
        history: list[dict],
        system: str,
        tools: Sequence[ToolSpec],
        cancel: threading.Event,
        notify: Callable[[str], None],
    ) -> dict:
        def chat():
            try:
                return self._client.chat(
                    model=self.model,
                    messages=to_ollama(history, system),
                    tools=[_tool(t) for t in tools],
                    think=self._think,
                )
            except ollama.ResponseError as exc:
                if self._think and "think" in str(exc.error).lower():
                    self._think = None
                    return chat()
                raise

        return from_ollama(with_retries(chat, _retry_after, cancel, notify, self.name).message)


def _tool(tool: ToolSpec) -> dict:
    return {
        "type": "function",
        "function": {"name": tool.name, "description": tool.description, "parameters": tool.parameters},
    }


def to_ollama(history: list[dict], system: str) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": system}]
    for message in history:
        role = message["role"]
        if role == "user":
            messages.append({"role": "user", "content": message["text"]})
        elif role == "assistant":
            out: dict = {"role": "assistant", "content": message.get("text") or ""}
            if message.get("tool_calls"):
                out["tool_calls"] = [
                    {"function": {"name": c["name"], "arguments": c["args"]}} for c in message["tool_calls"]
                ]
            messages.append(out)
        elif role == "tool":
            messages.append({"role": "tool", "tool_name": message["name"], "content": json.dumps(message["result"])})
        else:
            raise ValueError(f"unknown role {role!r}")
    return messages


def from_ollama(message) -> dict:
    calls = [
        {"id": new_call_id(), "name": c.function.name, "args": dict(c.function.arguments or {})}
        for c in message.tool_calls or []
    ]
    # Some local models (qwen3) still inline their reasoning even with think=False.
    text = _THINK.sub("", message.content or "").strip()
    return {"role": "assistant", "text": text, "tool_calls": calls}


def _retry_after(exc: Exception) -> float | None:
    if isinstance(exc, ollama.ResponseError):
        if exc.status_code in (429, 503):
            return 3.0
        raise ProviderError(f"Ollama error {exc.status_code}: {exc.error}") from exc
    if isinstance(exc, ConnectionError):
        raise ProviderError(f"can't reach Ollama: {exc}") from exc
    return None
