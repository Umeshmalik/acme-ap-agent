"""What each line the clerk types does: a slash command, or a message to the agent."""

from __future__ import annotations

import difflib

from app.core import Agent
from cli.display import show_approval, show_history
from cli.turn import run_turn
from providers import PROVIDERS, ProviderError

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
        show_pending(agent)
    elif command == "/model":
        switch_model(agent, rest)
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


def show_help() -> None:
    for command, (hint, text) in COMMANDS.items():
        print(f"  {(command + ' ' + hint).strip():<20} {text}")
    print("  Anything else is sent to the agent. Ctrl-C stops it while it's working.")


def show_pending(agent: Agent) -> None:
    pending = agent.pending()
    if not pending:
        print("  Nothing is waiting for approval.")
        return
    show_approval(pending[0]["details"])
    if len(pending) > 1:
        print(f"  ({len(pending) - 1} more waiting after this one)")


def switch_model(agent: Agent, name: str) -> None:
    if not name:
        print(f"  Model: {agent.provider_name}. Choose from: {', '.join(PROVIDERS)}")
        return
    try:
        agent.switch(name)
        print(f"  Switched to {name} ({agent.model}).")
    except ProviderError as exc:
        print(f"  Can't switch: {exc}")
