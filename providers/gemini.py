"""Google Gemini connector (SDK: google-genai)."""

from __future__ import annotations

import base64
import re
import threading
from typing import Callable, Sequence

from google import genai
from google.genai import errors, types

from providers.base import (
    REQUEST_TIMEOUT_S,
    ProviderError,
    ToolSpec,
    new_call_id,
    with_retries,
)

# Gemini 3 rejects replayed function calls that lack a thought signature. Calls made by
# another provider never had one, and Google documents this value for that case.
_SKIP_SIGNATURE = b"skip_thought_signature_validator"


class GeminiProvider:
    name = "gemini"
    env_key = "GEMINI_API_KEY"
    default_model = "gemini-3.8-flash"

    def __init__(self, api_key: str, model: str):
        self.model = model
        self._client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_S * 1000)
        )

    def complete(
        self,
        history: list[dict],
        system: str,
        tools: Sequence[ToolSpec],
        cancel: threading.Event,
        notify: Callable[[str], None],
    ) -> dict:
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=[_declaration(t) for t in tools])],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        response = with_retries(
            lambda: self._client.models.generate_content(
                model=self.model, contents=to_gemini(history, self.model), config=config
            ),
            _retry_after,
            cancel,
            notify,
            self.name,
        )
        return from_gemini(response)


def _declaration(tool: ToolSpec) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=tool.name, description=tool.description, parameters_json_schema=tool.parameters
    )


def to_gemini(history: list[dict], model: str = "") -> list[types.Content]:
    contents: list[types.Content] = []
    for message in history:
        role = message["role"]
        if role == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=message["text"])]))
        elif role == "assistant":
            signatures = message.get("meta", {}).get("gemini", {}).get("signatures", {})
            parts = [types.Part(text=message["text"])] if message.get("text") else []
            for i, call in enumerate(message.get("tool_calls", [])):
                signature = signatures.get(call["id"])
                if signature:
                    signature = base64.b64decode(signature)
                elif i == 0 and model.startswith("gemini-3"):
                    signature = _SKIP_SIGNATURE
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(name=call["name"], args=call["args"]),
                        thought_signature=signature or None,
                    )
                )
            contents.append(types.Content(role="model", parts=parts or [types.Part(text="")]))
        elif role == "tool":
            part = types.Part(
                function_response=types.FunctionResponse(name=message["name"], response=message["result"])
            )
            last = contents[-1] if contents else None
            if last and last.role == "user" and last.parts and last.parts[0].function_response:
                last.parts.append(part)
            else:
                contents.append(types.Content(role="user", parts=[part]))
        else:
            raise ValueError(f"unknown role {role!r}")
    return contents


def from_gemini(response: types.GenerateContentResponse) -> dict:
    candidate = response.candidates[0] if response.candidates else None
    if candidate is None or candidate.content is None or not candidate.content.parts:
        reason = candidate.finish_reason if candidate else "no candidates"
        return {"role": "assistant", "text": f"(the model returned nothing: {reason})", "tool_calls": []}
    texts, calls, signatures = [], [], {}
    for part in candidate.content.parts:
        if part.thought:
            continue
        if part.function_call:
            call_id = part.function_call.id or new_call_id()
            calls.append({"id": call_id, "name": part.function_call.name, "args": dict(part.function_call.args or {})})
            if part.thought_signature:
                signatures[call_id] = base64.b64encode(part.thought_signature).decode()
        elif part.text:
            texts.append(part.text)
    message = {"role": "assistant", "text": "".join(texts), "tool_calls": calls}
    if signatures:
        message["meta"] = {"gemini": {"signatures": signatures}}
    return message


def _retry_after(exc: Exception) -> float | None:
    if isinstance(exc, errors.APIError):
        if exc.code == 429:
            delay = quota_delay(exc.message or "")
            if delay is not None and delay > 60:
                raise ProviderError(
                    f"Gemini quota used up ({_quota_kind(exc.message or '')}); it resets in about"
                    f" {_human(delay)}. Switch provider with /model."
                ) from exc
            return delay or 10.0
        if exc.code in (500, 502, 503, 504):
            return 3.0
        raise ProviderError(f"Gemini error {exc.code}: {exc.message}") from exc
    return None


_RETRY_IN = re.compile(r"retry in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", re.IGNORECASE)


def quota_delay(message: str) -> float | None:
    """Seconds until Gemini says to retry ("Please retry in 17h29m6.7s"), if it says."""
    match = _RETRY_IN.search(message)
    if not match or not any(match.groups()):
        return None
    hours, minutes, seconds = (float(g) if g else 0.0 for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def _quota_kind(message: str) -> str:
    return "daily free-tier limit" if "per day" in message.lower() else "rate limit"


def _human(seconds: float) -> str:
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}h {rest // 60}m" if hours else f"{rest // 60}m"
