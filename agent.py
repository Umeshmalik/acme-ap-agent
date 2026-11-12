"""Command-line front end for the accounts payable agent.

    python agent.py --conversation demo [--model gemini|groq]

Each turn runs on a worker thread so Ctrl-C can stop it: the main thread marks the turn
cancelled, waits briefly (or until an approved post finishes), and tidies the history.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from typing import Callable

from dotenv import load_dotenv

from core import Agent
from providers import PROVIDERS, Cancelled, ProviderError
from store import Store

DEFAULT_PROVIDER = next(iter(PROVIDERS))


def show_event(kind: str, payload: object) -> None:
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


def run_turn(agent: Agent, work: Callable[[threading.Event], None]) -> None:
    cancel = threading.Event()
    failure: list[BaseException] = []

    def target() -> None:
        try:
            work(cancel)
        except Cancelled:
            pass
        except Exception as exc:  # shown to the clerk; the history is repaired below
            failure.append(exc)

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    try:
        while worker.is_alive():
            worker.join(0.2)
    except KeyboardInterrupt:
        cancel.set()
        print("\n  Stopping...")
        while worker.is_alive():
            try:
                # An approved post is never cut off midway; anything else gets a second.
                worker.join(None if agent.posting.is_set() else 1.0)
                if not agent.posting.is_set():
                    break
            except KeyboardInterrupt:
                print("  Finishing the post that was already under way...")
        agent.stopped()
        print("  Stopped. Nothing further will run for that request.")
        return
    if failure:
        exc = failure[0]
        label = "Provider error" if isinstance(exc, ProviderError) else f"Error ({type(exc).__name__})"
        print(f"  {label}: {exc}")
        agent.stopped(f"(this turn failed: {exc})")


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


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Accounts payable agent for Acme Foods")
    parser.add_argument("--conversation", default="default", help="name of the conversation to start or resume")
    parser.add_argument("--model", choices=list(PROVIDERS), help="provider to use (default: the saved one)")
    args = parser.parse_args(argv)

    store = Store()
    is_new, saved = store.open_conversation(args.conversation, args.model or DEFAULT_PROVIDER)
    agent = Agent(store, args.conversation, saved, on_event=show_event)
    if args.model and args.model != saved:
        try:
            agent.switch(args.model)
        except ProviderError as exc:
            print(f"Can't use {args.model}: {exc}")
    agent.recover()

    state = "new" if is_new else f"resumed, {len(agent.history)} messages"
    print(
        f'Conversation "{args.conversation}" ({state}). Model: {agent.provider_name}.'
        f" Commands: /model {'|'.join(PROVIDERS)}, /approve, /decline <reason>, /history, /quit."
        "\nPress Ctrl-C while the agent is working to stop it."
    )
    if agent.pending():
        show_approval(agent.pending()[0]["details"])

    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print("\n  (type /quit to leave)")
            continue
        if not line:
            continue
        command, _, rest = line.partition(" ")
        rest = rest.strip()
        if command == "/quit":
            return 0
        if command == "/history":
            show_history(agent)
        elif command == "/model":
            if not rest:
                print(f"  Model: {agent.provider_name}. Choose from: {', '.join(PROVIDERS)}")
                continue
            try:
                agent.switch(rest)
                print(f"  Switched to {rest} ({agent.model}).")
            except ProviderError as exc:
                print(f"  Can't switch: {exc}")
        elif command in ("/approve", "/decline"):
            if not agent.pending():
                print("  Nothing is waiting for approval.")
            elif command == "/approve":
                run_turn(agent, agent.approve)
            else:
                run_turn(agent, lambda cancel: agent.decline(rest, cancel))
        elif command.startswith("/"):
            print("  Commands: /model <name>, /approve, /decline <reason>, /history, /quit")
        elif agent.pending():
            print("  A post is waiting for approval: type /approve or /decline <reason> first.")
        else:
            run_turn(agent, lambda cancel: agent.send(line, cancel))


if __name__ == "__main__":
    sys.exit(main())
