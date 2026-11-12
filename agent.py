"""Command-line front end for the accounts payable agent.

    uv run agent.py --conversation demo [--model gemini|groq|ollama]

Each turn runs on a worker thread so Ctrl-C can stop it: the main thread marks the turn
cancelled, waits briefly (or until an approved post finishes), and tidies the history.
Input suggestions and history live in cli_input.py.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Callable, TextIO

from dotenv import load_dotenv

from cli_input import make_reader
from core import Agent
from providers import PROVIDERS, Cancelled, ProviderError
from store import Store

try:
    import fcntl
except ImportError:  # Windows: no conversation lock
    fcntl = None

HERE = Path(__file__).parent
HISTORY_FILE = HERE / ".agent_history"
LOCK_DIR = HERE / ".locks"
DEFAULT_PROVIDER = next(iter(PROVIDERS))
STILL_WORKING_EVERY_S = 15
_CONVERSATION_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")

# command -> (argument hint, help text)
COMMANDS = {
    "/approve": ("", "post the invoice that is waiting for approval"),
    "/decline": ("<reason>", "don't post it; the reason is passed to the agent"),
    "/pending": ("", "show the approval that is waiting, if any"),
    "/model": ("<name>", f"switch provider ({', '.join(PROVIDERS)}); history carries over"),
    "/history": ("", "show this conversation so far"),
    "/help": ("", "show these commands"),
    "/quit": ("", "leave (everything is already saved)"),
}


# output


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


def show_help() -> None:
    for command, (hint, text) in COMMANDS.items():
        print(f"  {(command + ' ' + hint).strip():<20} {text}")
    print("  Anything else is sent to the agent. Ctrl-C stops it while it's working.")


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
    waiting = len(agent.pending())
    approvals = f"  |  {waiting} approval{'s' if waiting != 1 else ''} waiting" if waiting else ""
    return f" {agent.conversation}  |  model: {agent.provider_name}{approvals}  |  /help"


# running a turn


def run_turn(agent: Agent, work: Callable[[threading.Event], None]) -> None:
    cancel = threading.Event()
    failure: list[BaseException] = []

    # Wait on an Event, not Thread.join: a Ctrl-C that interrupts join() makes CPython
    # mark the still-running thread as finished (bpo-45274 handling).
    done = threading.Event()

    def target() -> None:
        try:
            work(cancel)
        except Cancelled:
            pass
        except Exception as exc:  # shown to the clerk; the history is repaired below
            failure.append(exc)
        finally:
            done.set()

    threading.Thread(target=target, daemon=True).start()
    started = time.monotonic()
    next_hint = started + STILL_WORKING_EVERY_S
    try:
        while not done.wait(0.2):
            if time.monotonic() >= next_hint:
                print(f"  (still working on {agent.provider_name}: {time.monotonic() - started:.0f}s; Ctrl-C to stop)")
                next_hint += STILL_WORKING_EVERY_S
    except KeyboardInterrupt:
        cancel.set()
        print("\n  Stopping...")
        # An approved post is never cut off midway; anything else gets about a second.
        deadline = time.monotonic() + 1.0
        while not done.is_set():
            try:
                if not agent.posting.is_set() and time.monotonic() >= deadline:
                    break
                done.wait(0.1)
            except KeyboardInterrupt:
                if not agent.posting.is_set():
                    break
                print("  Finishing the post that was already under way...")
        agent.stopped()
        print("  Stopped. Nothing further will run for that request.")
        return
    if failure:
        exc = failure[0]
        label = "Provider error" if isinstance(exc, ProviderError) else f"Error ({type(exc).__name__})"
        print(f"  {label}: {exc}")
        agent.stopped(f"(this turn failed: {exc})")


def lock_conversation(name: str) -> TextIO | None:
    """Hold an exclusive lock for this conversation; the OS drops it if we die."""
    if fcntl is None:
        return None
    LOCK_DIR.mkdir(exist_ok=True)
    handle = open(LOCK_DIR / f"{name}.lock", "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise SystemExit(f'Conversation "{name}" is already open in another terminal.') from None
    return handle


# main loop


def handle(agent: Agent, line: str) -> bool:
    """Act on one line of input. Returns False to quit."""
    command, _, rest = line.partition(" ")
    command = command.lower()
    rest = rest.strip()
    if not command.startswith("/"):
        if agent.pending():
            print("  A post is waiting for approval: type /approve or /decline <reason> first.")
        else:
            run_turn(agent, lambda cancel: agent.send(line, cancel))
        return True
    if command == "/quit":
        return False
    if command in ("/help", "/"):
        show_help()
    elif command == "/history":
        show_history(agent)
    elif command == "/pending":
        pending = agent.pending()
        if pending:
            show_approval(pending[0]["details"])
            if len(pending) > 1:
                print(f"  ({len(pending) - 1} more waiting after this one)")
        else:
            print("  Nothing is waiting for approval.")
    elif command == "/model":
        if not rest:
            print(f"  Model: {agent.provider_name}. Choose from: {', '.join(PROVIDERS)}")
        else:
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
    else:
        close = [c for c in COMMANDS if c.startswith(command)] or difflib.get_close_matches(command, COMMANDS, n=2)
        hint = f" Did you mean {' or '.join(close)}?" if close else ""
        print(f"  Unknown command {command}.{hint} Type /help for the list.")
    return True


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Accounts payable agent for Acme Foods")
    parser.add_argument("--conversation", default="default", help="name of the conversation to start or resume")
    parser.add_argument("--model", choices=list(PROVIDERS), help="provider to use (default: the saved one)")
    args = parser.parse_args(argv)
    if not _CONVERSATION_NAME.fullmatch(args.conversation):
        parser.error("conversation names use letters, digits, '.', '_' or '-' (up to 64 characters)")
    lock = lock_conversation(args.conversation)  # noqa: F841 - held until exit

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
        " Type / for commands (suggestions appear as you type; right arrow accepts).\nPress Ctrl-C while the agent is working to stop it."
    )
    if agent.pending():
        show_approval(agent.pending()[0]["details"])

    read = make_reader(
        HISTORY_FILE,
        {command: text for command, (_, text) in COMMANDS.items()},
        PROVIDERS,
        status=lambda: status_line(agent),
    )
    while True:
        try:
            line = read("you> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print("  (type /quit to leave)")
            continue
        if line and not handle(agent, line):
            return 0


if __name__ == "__main__":
    sys.exit(main())
