from __future__ import annotations

import threading

import ledger
import posting_rules
from conftest import call, say
from core import Agent
from providers import Cancelled


def unanswered(history: list[dict]) -> set[str]:
    calls = {c["id"] for m in history for c in m.get("tool_calls") or []}
    return calls - {m["call_id"] for m in history if m["role"] == "tool"}


def test_guards():
    assert posting_rules.check_postable("invoice-01")["ok"]
    assert "blocked" in posting_rules.check_postable("invoice-04")["reason"]
    assert "JE-000001" in posting_rules.check_postable("invoice-05")["reason"]
    assert not posting_rules.check_postable("../vendors")["ok"]


def test_blocked_and_duplicate_are_refused_without_approval(make_agent, go):
    events = []
    agent, _ = make_agent(
        [call("post_invoice", "c1", invoice_id="invoice-04"), say("blocked"),
         call("post_invoice", "c2", invoice_id="invoice-05"), say("duplicate")],
        events=events,
    )
    agent.send("post invoice-04", go)
    agent.send("post invoice-05", go)
    assert agent.pending() == []
    assert not any(kind == "approval" for kind, _ in events)
    results = [m["result"] for m in agent.history if m["role"] == "tool"]
    assert all(r["posted"] is False for r in results)
    assert len(ledger.entries()) == 1


def test_approve_posts_once_and_resumes_after_restart(make_agent, store, go):
    agent, fakes = make_agent([call("post_invoice", invoice_id="invoice-02"), say("Posted.")])
    agent.send("post invoice-02", go)
    assert agent.pending()[0]["details"]["amount"] == "70800.00"
    assert ledger.entries(invoice_number="BL-88213") == []

    # "Restart": a new Agent over the same store still sees the waiting approval.
    fresh = Agent(store, "t", "a", lambda *_: None, fakes.__getitem__)
    fresh.recover()
    assert len(fresh.pending()) == 1
    fresh.approve(go)
    posted = ledger.entries(invoice_number="BL-88213")
    assert len(posted) == 1
    assert fresh.history[-1]["text"] == "Posted."
    assert fresh.history[-2]["result"]["entry_id"] == posted[0]["entry_id"]
    assert posting_rules.check_postable("invoice-02")["ok"] is False


def test_decline_posts_nothing_and_tells_the_model(make_agent, go):
    agent, fakes = make_agent([call("post_invoice", invoice_id="invoice-03"), say("Not posted.")])
    agent.send("post invoice-03", go)
    agent.decline("PO not received yet", go)
    assert ledger.entries(invoice_number="KOS-2026-1187") == []
    last_seen = fakes["a"].seen[-1]
    assert last_seen[-1]["result"] == {"posted": False, "declined_by_clerk": True, "reason": "PO not received yet"}


def test_interrupted_post_is_reconciled_from_the_ledger(make_agent, store, go):
    agent, fakes = make_agent([call("post_invoice", invoice_id="invoice-01")])
    agent.send("post invoice-01", go)
    store.set_post_status("c1", "posting")  # as if killed during ledger.post
    ledger.post(vendor_id="V-001", invoice_number="SGP/26-27/0412", invoice_date="2026-10-03",
                currency="INR", amount="16862.20", reference="t/c1")

    fresh = Agent(store, "t", "a", lambda *_: None, fakes.__getitem__)
    fresh.recover()
    assert fresh.pending() == []
    assert store.posts("posted", "t")[0]["entry_id"] == ledger.entries(invoice_number="SGP/26-27/0412")[0]["entry_id"]
    assert unanswered(fresh.history) == set()


def test_interrupted_post_that_never_landed_needs_approval_again(make_agent, store, go):
    agent, fakes = make_agent([call("post_invoice", invoice_id="invoice-01")])
    agent.send("post invoice-01", go)
    store.set_post_status("c1", "posting")

    fresh = Agent(store, "t", "a", lambda *_: None, fakes.__getitem__)
    fresh.recover()
    assert len(fresh.pending()) == 1
    assert ledger.entries(invoice_number="SGP/26-27/0412") == []


def test_switching_provider_keeps_the_history(make_agent, go):
    agent, fakes = make_agent(
        [call("read_invoice", invoice_id="invoice-06"), say("One line.")],
        [say("Shree Ganesh, INR 21948.00")],
    )
    agent.send("what's on invoice-06?", go)
    agent.switch("b")
    agent.send("which vendor?", go)
    seen = fakes["b"].seen[0]
    assert [m["role"] for m in seen] == ["user", "assistant", "tool", "assistant", "user"]


def test_stop_mid_turn_leaves_a_valid_history(make_agent, go):
    events = []
    agent, fakes = make_agent(
        [{"role": "assistant", "text": "", "tool_calls": [
            {"id": "r1", "name": "read_invoice", "args": {"invoice_id": "invoice-01"}},
            {"id": "r2", "name": "read_invoice", "args": {"invoice_id": "invoice-02"}}]},
         say("I read invoice-01.")],
        events=events,
    )
    cancel = threading.Event()
    original = agent.on_event

    def stop_on_second_tool(kind, payload):
        original(kind, payload)
        if kind == "tool" and payload["id"] == "r2":
            cancel.set()

    agent.on_event = stop_on_second_tool
    try:
        agent.send("read every invoice", cancel)
    except Cancelled:
        pass
    agent.stopped()
    assert unanswered(agent.history) == set()
    assert agent.history[-1]["role"] == "assistant"
    assert agent.history[-2]["result"]["error"].startswith("stopped")

    agent.send("which did you get through?", go)
    assert agent.history[-1]["text"] == "I read invoice-01."


def test_stop_while_waiting_on_the_model(make_agent, go):
    started = threading.Event()

    def slow(history, cancel):
        started.set()
        cancel.wait(5)
        return say("too late")

    agent, _ = make_agent([slow])
    cancel = threading.Event()
    worker = threading.Thread(target=lambda: _swallow(agent.send, "hi", cancel))
    worker.start()
    started.wait(2)
    cancel.set()
    worker.join(2)
    agent.stopped()
    assert [m["role"] for m in agent.history] == ["user", "assistant"]
    assert "too late" not in agent.history[-1]["text"]


def _swallow(fn, *args):
    try:
        fn(*args)
    except Cancelled:
        pass
