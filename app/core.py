"""The agent: the model/tool loop, approvals, stopping and crash repair.

No terminal I/O here: the agent reports through `on_event(kind, payload)` and the front
end (agent.py) decides how to show it. It only talks to providers through the registry,
to tools through `tools.TOOLS` and to disk through `Store`.

Every public turn method (`send`, `approve`, `decline`) takes a `cancel` event. Once it is
set, the turn saves nothing more (except the outcome of a post already under way), so a
caller can abandon the thread and call `stopped()` to tidy the history.
"""

from __future__ import annotations

import threading
from typing import Callable

import providers
from app.store import Store
from app.tools import TOOLS, Tool, run_tool
from providers import Cancelled, Provider

MAX_STEPS = 15
# ponytail: older turns are simply not sent once a conversation is this long; summarise
# them instead if clerks need the agent to remember that far back.
MAX_HISTORY = 200

SYSTEM_PROMPT = """You are the accounts payable assistant for Acme Foods. You help the clerk
check vendor invoices against the vendor list and post them to the ledger.

- Use the tools to answer; never guess invoice contents, vendor status or ledger state.
- To post, call post_invoice with the invoice id (e.g. "invoice-01"). The program checks
  the vendor and the ledger and asks the clerk to approve; you don't need to ask first.
- Read post_invoice's result carefully: only say an invoice was posted if the result has
  "posted": true and an entry_id. If it was declined or refused, say so and give the reason.
- Money is exact text like "16862.20"; quote it as written, with its currency.
- Earlier turns may have come from a different model; treat them as your own.
- Be brief."""

EventHandler = Callable[[str, object], None]


class AgentBusy(Exception):
    """A post is waiting for approval; it must be approved or declined first."""

    def __init__(self) -> None:
        super().__init__("a post is waiting for approval: type /approve or /decline <reason> first")


class Agent:
    def __init__(
        self,
        store: Store,
        conversation: str,
        provider_name: str,
        on_event: EventHandler,
        provider_factory: Callable[[str], Provider] = providers.create,
        tools: dict[str, Tool] = TOOLS,
    ):
        self.store = store
        self.conversation = conversation
        self.provider_name = provider_name
        self.on_event = on_event
        self.tools = tools
        self.history = store.messages(conversation)
        self.posting = threading.Event()  # set while an approved action is running
        self._factory = provider_factory
        self._providers: dict[str, Provider] = {}
        self._lock = threading.Lock()

    # providers

    def _provider(self, name: str) -> Provider:
        if name not in self._providers:
            self._providers[name] = self._factory(name)
        return self._providers[name]

    @property
    def model(self) -> str:
        return self._provider(self.provider_name).model

    def switch(self, name: str) -> None:
        """Change provider; the history carries over untouched. Raises ProviderError."""
        self._provider(name)
        self.provider_name = name
        self.store.set_model(self.conversation, name)

    # turns

    def pending(self) -> list[dict]:
        return self.store.posts("pending_approval", self.conversation)

    def send(self, text: str, cancel: threading.Event) -> None:
        self._settle_interrupted()
        if self.pending():
            raise AgentBusy()
        self._save({"role": "user", "text": text}, cancel)
        self._loop(cancel)

    def approve(self, cancel: threading.Event) -> None:
        post = self._first_pending()
        call, tool = self._call_and_tool(post["call_id"])
        check = _prepare(tool, call["args"])
        if not check["ok"]:
            self._emit(cancel, "refused", check["reason"])
            self._resolve(post, "refused", {"posted": False, "reason": check["reason"]}, cancel)
        elif check["details"] != post["details"]:
            reason = "the invoice changed after the approval was shown; nothing was posted"
            self._emit(cancel, "refused", reason)
            self._resolve(post, "refused", {"posted": False, "reason": reason}, cancel)
        else:
            self._run_approved(post, tool, cancel)
        self._continue(cancel)

    def decline(self, reason: str, cancel: threading.Event) -> None:
        post = self._first_pending()
        result = {"posted": False, "declined_by_clerk": True, "reason": reason or "no reason given"}
        self._resolve(post, "declined", result, cancel, reason=reason)
        self._continue(cancel)

    def _first_pending(self) -> dict:
        pending = self.pending()
        if not pending:
            raise LookupError("nothing is waiting for approval")
        return pending[0]

    def _continue(self, cancel: threading.Event) -> None:
        """After an approval is settled: show the next one, or let the model carry on."""
        pending = self.pending()
        if pending:
            self.on_event("approval", pending[0]["details"])
        else:
            self._loop(cancel)

    def _loop(self, cancel: threading.Event) -> None:
        provider = self._provider(self.provider_name)
        for _ in range(MAX_STEPS):
            if cancel.is_set():
                raise Cancelled()
            with self._lock:
                history = _recent(self.history)
            reply = provider.complete(
                history,
                SYSTEM_PROMPT,
                list(self.tools.values()),
                cancel,
                notify=lambda text: None if cancel.is_set() else self.on_event("notice", text),
            )
            self._save(reply, cancel)
            if reply["text"]:
                self.on_event("text", reply["text"])
            if not reply["tool_calls"]:
                return
            for call in reply["tool_calls"]:
                if cancel.is_set():
                    raise Cancelled()
                tool = self.tools.get(call["name"])
                self._emit(cancel, "tool", call)
                if tool and tool.needs_approval:
                    self._request_approval(tool, call, cancel)
                else:
                    result = run_tool(call["name"], call["args"])
                    self._save(_tool_message(call, result), cancel)
            pending = self.pending()
            if pending:
                self._emit(cancel, "approval", pending[0]["details"])
                return
        note = f"(stopped after {MAX_STEPS} steps without a final answer)"
        self._save({"role": "assistant", "text": note, "tool_calls": []}, cancel)
        self.on_event("text", note)

    # approvals

    def _request_approval(self, tool: Tool, call: dict, cancel: threading.Event) -> None:
        check = _prepare(tool, call["args"])
        if check["ok"]:
            details = check["details"]
            reason = self._already_open(details)
            if reason is None:
                with self._lock:
                    if cancel.is_set():
                        raise Cancelled()
                    invoice_id = call["args"].get("invoice_id", "")
                    if self.store.add_pending_post(self.conversation, call["id"], invoice_id, details):
                        return
                # Another conversation opened it between our check and our insert.
                reason = self._already_open(details) or "already waiting for approval in another conversation"
            check = {"ok": False, "reason": reason}
        self._emit(cancel, "refused", check["reason"])
        self._save(_tool_message(call, {"posted": False, "refused": True, "reason": check["reason"]}), cancel)

    def _already_open(self, details: dict) -> str | None:
        waiting = self.store.open_posts_for(details["vendor_id"], details["invoice_number"])
        if not waiting:
            return None
        where = waiting[0]["conversation"]
        if waiting[0]["status"] == "posting":
            return (
                f"already being posted in conversation {where!r}; if that program was closed"
                " mid-post, reopen that conversation to finish it"
            )
        return f"already waiting for approval (conversation {where!r})"

    def _run_approved(self, post: dict, tool: Tool, cancel: threading.Event) -> None:
        with self._lock:
            if cancel.is_set():
                raise Cancelled()
            if not self.store.claim_post(post["call_id"]):
                raise LookupError("that approval was already settled elsewhere")
            self.posting.set()
        try:
            # Once started, the action is never abandoned: its outcome is always recorded.
            # If anything below raises, the post stays at `posting` and _settle_interrupted
            # asks the ledger what happened before the next turn (or on restart).
            try:
                result = tool.run(post["details"], reference=self._reference(post))
            except Exception as exc:
                result = tool.recover(post["details"], self._reference(post)) if tool.recover else None
                if result is None:
                    self.on_event("refused", f"the ledger call failed, nothing was posted: {exc}")
                    self._resolve(post, "refused", {"posted": False, "error": str(exc)}, cancel, force=True)
                    return
            self._resolve(post, "posted", result, cancel, entry_id=result.get("entry_id"), force=True)
            self.on_event("posted", result)  # shown even after a stop: the clerk must know
        finally:
            self.posting.clear()

    def _reference(self, post: dict) -> str:
        return f"{self.conversation}/{post['call_id']}"

    def _call_and_tool(self, call_id: str) -> tuple[dict, Tool]:
        for message in reversed(self.history):
            for call in message.get("tool_calls") or []:
                if call["id"] == call_id:
                    return call, self.tools[call["name"]]
        raise LookupError(f"tool call {call_id} not in history")

    # saving

    def _emit(self, cancel: threading.Event, kind: str, payload: object) -> None:
        """Report an event unless the turn was stopped (an abandoned turn stays quiet)."""
        if not cancel.is_set():
            self.on_event(kind, payload)

    def _save(self, message: dict, cancel: threading.Event) -> None:
        with self._lock:
            if cancel.is_set():
                raise Cancelled()
            self.store.append(self.conversation, message)
            self.history.append(message)

    def _resolve(
        self,
        post: dict,
        status: str,
        result: dict,
        cancel: threading.Event,
        entry_id: str | None = None,
        reason: str | None = None,
        force: bool = False,
    ) -> None:
        call, _ = self._call_and_tool(post["call_id"])
        message = _tool_message(call, result)
        with self._lock:
            if cancel.is_set() and not force:
                raise Cancelled()
            self.store.resolve_post(self.conversation, post["call_id"], status, message, entry_id, reason)
            self.history.append(message)

    # stop and crash repair

    def stopped(self, note: str = "(stopped by the clerk)") -> None:
        """Call after abandoning a turn: close any tool calls left without a result."""
        with self._lock:
            self._repair(note)

    def recover(self) -> None:
        """Call on startup: settle posts cut off mid-way, then close dangling tool calls."""
        self._settle_interrupted()
        with self._lock:
            self._repair("(the program stopped before this turn finished)")

    def _settle_interrupted(self) -> None:
        """Settle posts left at `posting` (a crash, or a failure after the ledger call):
        the ledger, searched by our reference, says whether the post happened."""
        if self.posting.is_set():  # a post is running right now; it records its own outcome
            return
        for post in self.store.posts("posting", self.conversation):
            call, tool = self._call_and_tool(post["call_id"])
            result = tool.recover(post["details"], self._reference(post)) if tool.recover else None
            if result is not None:
                self._resolve(post, "posted", result, threading.Event(), entry_id=result.get("entry_id"))
                self.on_event("notice", f"An interrupted post did reach the ledger as {result.get('entry_id')}.")
            else:
                self.store.set_post_status(post["call_id"], "pending_approval")
                self.on_event("notice", "An interrupted post never reached the ledger; it needs approval again.")

    def _repair(self, note: str) -> None:
        if not self.history:
            return
        answered = {m["call_id"] for m in self.history if m["role"] == "tool"}
        waiting = {p["call_id"] for p in self.pending()} | {
            p["call_id"] for p in self.store.posts("posting", self.conversation)
        }
        fixes = [
            _tool_message(call, {"error": "stopped before this tool ran; no action was taken"})
            for message in self.history
            for call in message.get("tool_calls") or []
            if call["id"] not in answered and call["id"] not in waiting
        ]
        last = (fixes or self.history)[-1]
        if not waiting and (last["role"] != "assistant" or last.get("tool_calls")):
            fixes.append({"role": "assistant", "text": note, "tool_calls": []})
        if fixes:
            self.store.append_many(self.conversation, fixes)
            self.history.extend(fixes)


def _recent(history: list[dict]) -> list[dict]:
    """The latest messages, at most MAX_HISTORY, starting at a user message so every tool
    call keeps its result (a turn is never cut in half)."""
    if len(history) <= MAX_HISTORY:
        return list(history)
    starts = [i for i, m in enumerate(history) if m["role"] == "user"]
    fits = [i for i in starts if len(history) - i <= MAX_HISTORY]
    start = fits[0] if fits else (starts[-1] if starts else 0)
    return history[start:]


def _prepare(tool: Tool, args: dict) -> dict:
    try:
        return tool.prepare(**args)
    except TypeError as exc:
        return {"ok": False, "reason": f"bad arguments for {tool.name}: {exc}"}
    except Exception as exc:  # a check that can't run must refuse, never allow
        return {"ok": False, "reason": f"checks failed to run: {type(exc).__name__}: {exc}"}


def _tool_message(call: dict, result: dict) -> dict:
    return {"role": "tool", "call_id": call["id"], "name": call["name"], "result": result}
