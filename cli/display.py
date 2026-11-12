"""How things are shown to the clerk. Nothing here changes any state."""

from __future__ import annotations

import json

from core import Agent


def show_event(kind: str, payload: object) -> None:
    """The Agent's `on_event` callback: print one event from a turn."""
    if kind == "text":
        lines = str(payload).strip().splitlines() or [""]
        print("agent> " + "\n       ".join(lines))
    elif kind == "tool":
        args = ", ".join(f"{k}={v!r}" for k, v in payload["args"].items())
        print(f"  . {payload['name']}({args})")
    elif kind == "approval":
        show_approval(payload)
    elif kind == "refused":
        print(f"  Not posting: {payload}")
    elif kind == "posted":
        print(
            f"  Posted to the ledger as {payload['entry_id']}: {payload['vendor_id']}"
            f" {payload['invoice_number']}, {payload['currency']} {payload['amount']}"
        )
    elif kind == "notice":
        print(f"  {payload}")


def show_approval(details: dict) -> None:
    print(
        "\n  Approval needed: post to the ledger\n"
        f"    vendor    {details['vendor_id']}  {details['vendor_name']}\n"
        f"    invoice   {details['invoice_number']}  dated {details['invoice_date']}\n"
        f"    amount    {details['currency']} {details['amount']}\n"
        "  Type /approve or /decline <reason>\n"
    )


def show_history(agent: Agent) -> None:
    for message in agent.history:
        if message["role"] == "user":
            print(f"  you: {message['text']}")
        elif message["role"] == "assistant":
            if message.get("text"):
                print(f"  agent: {message['text']}")
            for call in message.get("tool_calls") or []:
                print(f"  agent calls {call['name']}({json.dumps(call['args'])})")
        else:
            print(f"  result of {message['name']}: {json.dumps(message['result'])[:120]}")


def status_line(agent: Agent) -> str:
    """The bottom bar: conversation, model and how many approvals are waiting."""
    waiting = len(agent.pending())
    approvals = f"  |  {waiting} approval{'s' if waiting != 1 else ''} waiting" if waiting else ""
    return f" {agent.conversation}  |  model: {agent.provider_name}{approvals}  |  /help"
