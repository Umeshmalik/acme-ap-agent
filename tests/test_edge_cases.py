from __future__ import annotations

import dataclasses
import json
import threading

import ledger
import tools
from conftest import call, say
from core import MAX_HISTORY, _recent
from providers.base import with_retries


def unanswered(history: list[dict]) -> set[str]:
    calls = {c["id"] for m in history for c in m.get("tool_calls") or []}
    return calls - {m["call_id"] for m in history if m["role"] == "tool"}


def test_two_conversations_racing_for_one_invoice_get_one_approval(make_agent, store, go):
    a, _ = make_agent([call("post_invoice", "a1", invoice_id="invoice-01")], conversation="a")
    b, _ = make_agent([call("post_invoice", "b1", invoice_id="invoice-01"), say("refused")], conversation="b")
    a.send("post invoice-01", go)
    # b checks before a's row is visible (the race); the unique index must still stop it.
    real = store.open_posts_for
    store.open_posts_for = lambda *args: []
    b.send("post invoice-01", go)
    store.open_posts_for = real
    assert len(a.pending()) == 1
    assert b.pending() == []
    assert b.history[-2]["result"]["refused"] is True


def test_dropped_connection_is_retried(monkeypatch):
    cancel, notes = threading.Event(), []
    monkeypatch.setattr(cancel, "wait", lambda t: False)
    attempts = iter([ConnectionResetError(), None])

    def flaky():
        exc = next(attempts)
        if exc:
            raise exc
        return "ok"

    assert with_retries(flaky, lambda e: None, cancel, notes.append, "gemini") == "ok"
    assert "connection failed" in notes[0]


def test_stuck_post_is_settled_from_the_ledger_next_turn(make_agent, store, go):
    agent, fakes = make_agent([call("post_invoice", invoice_id="invoice-01"), say("Carrying on.")])
    agent.send("post invoice-01", go)
    tool = agent.tools["post_invoice"]
    broken = {"recover": True}

    def run_then_fail(details, reference):
        tool.run(details, reference=reference)  # the entry lands...
        raise RuntimeError("connection reset")  # ...but we never hear back

    def recover(details, reference):
        if broken.pop("recover", False):
            raise RuntimeError("ledger unreachable")
        return tool.recover(details, reference)

    agent.tools = {**agent.tools, "post_invoice": dataclasses.replace(tool, run=run_then_fail, recover=recover)}
    try:
        agent.approve(go)
    except RuntimeError:
        pass
    assert store.posts("posting", "t")  # outcome unknown, left for the ledger to decide

    agent.send("anything else?", go)  # the next turn asks the ledger first
    assert store.posts("posted", "t")[0]["entry_id"] == ledger.entries(invoice_number="SGP/26-27/0412")[0]["entry_id"]
    assert unanswered(fakes["a"].seen[-1]) == set()
    assert len(ledger.entries(invoice_number="SGP/26-27/0412")) == 1


def test_program_announces_the_posted_entry(make_agent, go):
    events = []
    agent, _ = make_agent([call("post_invoice", invoice_id="invoice-02"), say("Done.")], events=events)
    agent.send("post invoice-02", go)
    agent.approve(go)
    posted = [p for kind, p in events if kind == "posted"]
    assert posted and posted[0]["entry_id"].startswith("JE-")


def test_long_history_is_cut_at_a_turn_boundary():
    turn = [
        {"role": "user", "text": "q"},
        {"role": "assistant", "text": "", "tool_calls": [{"id": "x", "name": "list_invoices", "args": {}}]},
        {"role": "tool", "call_id": "x", "name": "list_invoices", "result": {}},
        {"role": "assistant", "text": "a", "tool_calls": []},
    ]
    history = turn * (MAX_HISTORY // 2)
    kept = _recent(history)
    assert len(kept) <= MAX_HISTORY
    assert kept[0]["role"] == "user"
    assert _recent(turn) == turn


def test_invoice_that_does_not_add_up_is_refused(tmp_path, monkeypatch):
    invoice = json.loads((tools.INVOICE_DIR / "invoice-01.json").read_text())
    invoice["total"] = str(float(invoice["total"]) + 100)
    (tmp_path / "invoice-01.json").write_text(json.dumps(invoice))
    monkeypatch.setattr(tools, "INVOICE_DIR", tmp_path)
    assert "doesn't add up" in tools.check_postable("invoice-01")["reason"]
