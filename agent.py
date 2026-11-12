"""Start the accounts payable agent in the terminal.

    uv run agent.py --conversation demo [--model gemini|groq|ollama]

This file only wires things together: open the conversation, settle anything a crash left
behind, then read lines until /quit. The pieces live in cli/ (terminal), app/ (the agent,
its tools and posting rules, saving) and providers/ (the LLM connectors).
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv

from app.core import Agent
from app.store import Store
from cli.commands import COMMANDS, handle
from cli.display import show_approval, show_event, status_line
from cli.lock import lock_conversation
from cli.prompt import make_reader
from providers import PROVIDERS, ProviderError

HISTORY_FILE = Path(__file__).parent / ".agent_history"
DEFAULT_PROVIDER = next(iter(PROVIDERS))
_CONVERSATION_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _parse_args(argv)
    lock = lock_conversation(args.conversation)  # noqa: F841 - held until exit
    agent, is_new = _open(args)

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
        try:
            if line and not handle(agent, line):
                return 0
        except sqlite3.Error as exc:  # e.g. disk full or a locked database: say so and carry on
            print(f"  Storage error: {exc}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Accounts payable agent for Acme Foods")
    parser.add_argument("--conversation", default="default", help="name of the conversation to start or resume")
    parser.add_argument("--model", choices=list(PROVIDERS), help="provider to use (default: the saved one)")
    args = parser.parse_args(argv)
    if not _CONVERSATION_NAME.fullmatch(args.conversation):
        parser.error("conversation names use letters, digits, '.', '_' or '-' (up to 64 characters)")
    return args


def _open(args: argparse.Namespace) -> tuple[Agent, bool]:
    """Load or create the conversation, then repair anything a crash left behind."""
    store = Store()
    is_new, saved = store.open_conversation(args.conversation, args.model or DEFAULT_PROVIDER)
    agent = Agent(store, args.conversation, saved, on_event=show_event)
    if args.model and args.model != saved:
        try:
            agent.switch(args.model)
        except ProviderError as exc:
            print(f"Can't use {args.model}: {exc}")
    agent.recover()
    return agent, is_new


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:  # Ctrl-C outside a turn, e.g. during startup: nothing to tidy
        print("\n  Interrupted.")
        sys.exit(130)
    except sqlite3.DatabaseError as exc:
        sys.exit(f"Can't use the database ({exc}). If it is corrupt, move it aside and start again.")
