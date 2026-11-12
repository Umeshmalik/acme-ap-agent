"""Run one agent turn on a worker thread so Ctrl-C can stop it.

The main thread only waits. On Ctrl-C it sets the turn's cancel event (after which the
turn can't save anything), waits about a second, or until an approved post has finished,
then closes the turn's dangling tool calls so the history stays valid.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable

from app.core import Agent
from providers import Cancelled, ProviderError

STILL_WORKING_EVERY_S = 15


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
        _wait_for_post_in_flight(agent, done)
        tidy(agent, "(stopped by the clerk)")
        print("  Stopped. Nothing further will run for that request.")
        return
    if failure:
        exc = failure[0]
        label = "Provider error" if isinstance(exc, ProviderError) else f"Error ({type(exc).__name__})"
        print(f"  {label}: {exc}")
        tidy(agent, f"(this turn failed: {exc})")


def _wait_for_post_in_flight(agent: Agent, done: threading.Event) -> None:
    """An approved post is never cut off midway; anything else gets about a second."""
    deadline = time.monotonic() + 1.0
    warned = False
    while not done.is_set():
        try:
            if not agent.posting.is_set() and time.monotonic() >= deadline:
                return
            done.wait(0.1)
        except KeyboardInterrupt:
            if not agent.posting.is_set():
                return
            if warned:
                # Same as a crash mid-post: the next start checks the ledger for it.
                print("  Quitting now. On the next start the ledger is checked for this post.", flush=True)
                os._exit(130)
            warned = True
            print("  Finishing the post that was already under way... (Ctrl-C again to quit now)")


def tidy(agent: Agent, note: str) -> None:
    """Close the abandoned turn's tool calls. If even that can't be saved, the next start does it."""
    try:
        agent.stopped(note)
    except Exception as exc:
        print(f"  Couldn't save the end of that turn ({type(exc).__name__}: {exc}); it is repaired on the next start.")
