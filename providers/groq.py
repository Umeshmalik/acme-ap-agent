"""Groq connector (SDK: groq). Groq's API follows the OpenAI chat-completions shape."""

from __future__ import annotations

import json
import threading
from typing import Callable, Sequence

import groq

from providers.base import (
    REQUEST_TIMEOUT_S,
    ProviderError,
    ToolSpec,
    new_call_id,
    with_retries,
)


class GroqProvider:
    name = "groq"
    env_key = "GROQ_API_KEY"
    default_model = "openai/gpt-oss-120b"

    def __init__(self, api_key: str, model: str):
        self.model = model
        self._client = groq.Groq(api_key=api_key, max_retries=0, timeout=REQUEST_TIMEOUT_S)

    def complete(
        self,
        history: list[dict],
        system: str,
        tools: Sequence[ToolSpec],
        cancel: threading.Event,
        notify: Callable[[str], None],
    ) -> dict:
        response = with_retries(
            lambda: self._client.chat.completions.create(
                model=self.model,
                messages=to_groq(history, system),
                tools=[_tool(t) for t in tools],
                tool_choice="auto",
            ),
            _retry_after,
            cancel,
            notify,
            self.name,
        )
        return from_groq(response.choices[0].message)


def _tool(tool: ToolSpec) -> dict:
    return {
        "type": "function",
        "function": {"name": tool.name, "description": tool.description, "parameters": tool.parameters},
    }


def to_groq(history: list[dict], system: str) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": system}]
    for message in history:
        role = message["role"]
        if role == "user":
            messages.append({"role": "user", "content": message["text"]})
        elif role == "assistant":
            out: dict = {"role": "assistant", "content": message.get("text") or ""}
            if message.get("tool_calls"):
                out["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["name"], "arguments": json.dumps(call["args"])},
                    }
                    for call in message["tool_calls"]
                ]
            messages.append(out)
        elif role == "tool":
            messages.append(
                {"role": "tool", "tool_call_id": message["call_id"], "content": json.dumps(message["result"])}
            )
        else:
            raise ValueError(f"unknown role {role!r}")
    return messages


def from_groq(message) -> dict:
    calls = []
    for call in message.tool_calls or []:
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {"_unparsed_arguments": call.function.arguments}
        if not isinstance(args, dict):
            args = {"_unparsed_arguments": call.function.arguments}
        calls.append({"id": call.id or new_call_id(), "name": call.function.name, "args": args})
    return {"role": "assistant", "text": message.content or "", "tool_calls": calls}


def _retry_after(exc: Exception) -> float | None:
    if isinstance(exc, groq.RateLimitError):
        header = exc.response.headers.get("retry-after")
        try:
            return float(header) if header else 10.0
        except ValueError:
            return 10.0
    if isinstance(exc, (groq.APIConnectionError, groq.InternalServerError)):
        return 3.0
    if isinstance(exc, groq.BadRequestError) and "tool_use_failed" in str(exc):
        return 1.0  # the model produced a malformed tool call; asking again usually works
    if isinstance(exc, groq.APIStatusError):
        raise ProviderError(f"Groq error {exc.status_code}: {exc.message}") from exc
    return None
