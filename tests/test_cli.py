from __future__ import annotations

import os
import signal
import threading

from conftest import call, say

import ledger
from cli.commands import COMMANDS, handle
from cli.prompt import suggestions
from cli.turn import run_turn

CMDS = {c: text for c, (_, text) in COMMANDS.items()}
PROVIDERS = ["gemini", "groq", "ollama"]


def texts(line: str) -> list[str]:
    return [text for text, _ in suggestions(line, CMDS, PROVIDERS)]


def test_suggestions():
    assert texts("/") == list(CMDS)
    assert texts("/ap") == ["/approve"]
    assert texts("/model ") == PROVIDERS
    assert texts("/model g") == ["gemini", "groq"]
    assert texts("/model groq extra") == []
    assert texts("post in")[:2] == ["invoice-01", "invoice-02"]
    assert texts("post invoice-04") == []          # already complete
    assert texts("hello") == []                    # plain chat: no noise
    assert texts("") == []
    meta = dict(suggestions("post invoice-0", CMDS, PROVIDERS))
    assert "Orbit Facility Services" in meta["invoice-04"]


def test_ctrl_c_during_an_approved_post_waits_for_it(make_agent, monkeypatch):
    monkeypatch.setattr(ledger.time, "sleep", _real_sleep)  # the real 2s post, so Ctrl-C lands mid-post
    agent, _ = make_agent([call("post_invoice", invoice_id="invoice-06"), say("Posted.")])
    agent.send("post invoice-06", threading.Event())
    for delay in (0.5, 0.9):  # Ctrl-C, then an impatient second one
        threading.Timer(delay, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    run_turn(agent, agent.approve)
    rows = ledger.entries(invoice_number="SGP/26-27/0431")
    assert len(rows) == 1
    assert agent.store.posts("posted", "t")[0]["entry_id"] == rows[0]["entry_id"]


def _real_sleep(seconds: float) -> None:
    threading.Event().wait(seconds)


def test_absurd_lines_never_crash(make_agent, capsys):
    agent, _ = make_agent([call("post_invoice", invoice_id="invoice-03"), say("waiting")])
    for line in ["/", "//", "/APPROVE", "/aprove", "/model", "/model banana", "/model  ", "/decline",
                 "/approve now please", "/pending", "/history", "/help"]:
        assert handle(agent, line) is True
    out = capsys.readouterr().out
    assert "Did you mean /approve?" in out
    assert "Can't switch" in out
    assert handle(agent, "/quit") is False
