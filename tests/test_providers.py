from __future__ import annotations

import base64
import json

from google.genai import types

from providers.gemini import from_gemini, quota_delay, to_gemini
from providers.groq import to_groq

HISTORY = [
    {"role": "user", "text": "what's on invoice-06?"},
    {"role": "assistant", "text": "", "tool_calls": [
        {"id": "call_g1", "name": "read_invoice", "args": {"invoice_id": "invoice-06"}},
        {"id": "call_g2", "name": "lookup_vendor", "args": {"name_or_id": "V-001"}}]},
    {"role": "tool", "call_id": "call_g1", "name": "read_invoice", "result": {"total": "21948.00"}},
    {"role": "tool", "call_id": "call_g2", "name": "lookup_vendor", "result": {"found": True}},
    {"role": "assistant", "text": "One line, INR 21948.00.", "tool_calls": []},
]


def test_quota_delay_parsing():
    assert quota_delay("Generate content requests per day limit exceeded. Please retry in 17h29m6.78s.") == 17 * 3600 + 29 * 60 + 6.78
    assert quota_delay("Please retry in 42.5s.") == 42.5
    assert quota_delay("Resource exhausted") is None


def test_to_groq():
    messages = to_groq(HISTORY, "sys")
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "tool", "assistant"]
    assert messages[2]["tool_calls"][0]["id"] == "call_g1"
    assert json.loads(messages[2]["tool_calls"][0]["function"]["arguments"]) == {"invoice_id": "invoice-06"}
    assert messages[3]["tool_call_id"] == "call_g1"


def test_to_gemini_merges_tool_results_into_one_turn():
    contents = to_gemini(HISTORY, "gemini-2.5-flash")
    assert [c.role for c in contents] == ["user", "model", "user", "model"]
    assert [p.function_call.name for p in contents[1].parts] == ["read_invoice", "lookup_vendor"]
    assert [p.function_response.name for p in contents[2].parts] == ["read_invoice", "lookup_vendor"]
    assert contents[1].parts[0].thought_signature is None


def test_gemini_3_gets_a_placeholder_signature_for_foreign_calls():
    contents = to_gemini(HISTORY, "gemini-3-flash")
    assert contents[1].parts[0].thought_signature == b"skip_thought_signature_validator"


def test_gemini_signature_round_trip():
    response = types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=[
        types.Part(text="thinking", thought=True),
        types.Part(function_call=types.FunctionCall(name="read_invoice", args={"invoice_id": "invoice-01"}),
                   thought_signature=b"sig"),
    ]))])
    message = from_gemini(response)
    call = message["tool_calls"][0]
    assert message["text"] == ""
    assert message["meta"]["gemini"]["signatures"][call["id"]] == base64.b64encode(b"sig").decode()
    replayed = to_gemini([{"role": "user", "text": "x"}, message], "gemini-2.5-flash")
    assert replayed[1].parts[0].thought_signature == b"sig"
