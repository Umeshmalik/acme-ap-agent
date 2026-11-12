"""The agent's tools: the registry the model sees, and how a tool call is run.

Each tool is a `Tool` in `TOOLS`. The agent core only sees this registry: it runs ordinary
tools directly and sends any tool with `needs_approval` through the approval path
(`prepare` -> clerk approves -> `run`). The posting rules themselves are in posting_rules.py.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable

import ledger
from app.data import ToolError, find_vendor, invoice_ids, load_invoice
from app.posting_rules import check_postable, find_interrupted_post, post_to_ledger


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict  # JSON schema for the arguments
    run: Callable[..., dict]
    # Approval tools: `prepare(**args)` runs the checks and returns either
    # {"ok": True, "details": {...}} or {"ok": False, "reason": "..."};
    # `run(details, reference)` acts; `recover(details, reference)` returns the result if an
    # interrupted `run` did take effect, else None.
    needs_approval: bool = False
    prepare: Callable[..., dict] | None = None
    recover: Callable[[dict, str], dict | None] | None = None


# read-only tools


def list_invoices() -> dict:
    invoices = []
    for invoice_id in invoice_ids():
        invoice = load_invoice(invoice_id)
        invoices.append(
            {
                "invoice_id": invoice_id,
                "invoice_number": invoice["invoice_number"],
                "vendor_name": invoice["vendor_name"],
                "invoice_date": invoice["invoice_date"],
                "currency": invoice["currency"],
                "total": invoice["total"],
            }
        )
    return {"invoices": invoices}


def read_invoice(invoice_id: str) -> dict:
    delay = float(os.environ.get("READ_DELAY") or 0)  # to make the stop test easy
    if delay:
        time.sleep(delay)
    return {"invoice_id": invoice_id, "invoice": load_invoice(invoice_id)}


def lookup_vendor(name_or_id: str) -> dict:
    vendor = find_vendor(name_or_id)
    if vendor is None:
        return {"found": False, "query": name_or_id}
    return {"found": True, "vendor": vendor}


def find_ledger_entries(vendor_id: str | None = None, invoice_number: str | None = None) -> dict:
    filters = {k: v for k, v in (("vendor_id", vendor_id), ("invoice_number", invoice_number)) if v}
    return {"entries": ledger.entries(**filters)}


# the registry

_INVOICE_ID_PARAM = {
    "type": "object",
    "properties": {"invoice_id": {"type": "string", "description": "File id, e.g. 'invoice-01'"}},
    "required": ["invoice_id"],
}

TOOLS: dict[str, Tool] = {
    tool.name: tool
    for tool in [
        Tool(
            name="list_invoices",
            description="List every invoice file with its id, vendor, number, date and total.",
            parameters={"type": "object", "properties": {}},
            run=list_invoices,
        ),
        Tool(
            name="read_invoice",
            description="Read one invoice in full: lines, subtotal, taxes and total.",
            parameters=_INVOICE_ID_PARAM,
            run=read_invoice,
        ),
        Tool(
            name="lookup_vendor",
            description="Look up a vendor by vendor_id (e.g. 'V-001') or exact name. Returns its status.",
            parameters={
                "type": "object",
                "properties": {"name_or_id": {"type": "string"}},
                "required": ["name_or_id"],
            },
            run=lookup_vendor,
        ),
        Tool(
            name="find_ledger_entries",
            description="Find entries already posted to the ledger, filtered by vendor_id and/or invoice_number.",
            parameters={
                "type": "object",
                "properties": {"vendor_id": {"type": "string"}, "invoice_number": {"type": "string"}},
            },
            run=find_ledger_entries,
        ),
        Tool(
            name="post_invoice",
            description=(
                "Post an invoice to the ledger so it gets paid. The program checks the vendor and"
                " the ledger, then asks the clerk to approve. The result says whether it was"
                " posted, declined (with the clerk's reason) or refused (with the reason)."
            ),
            parameters=_INVOICE_ID_PARAM,
            run=post_to_ledger,
            needs_approval=True,
            prepare=check_postable,
            recover=find_interrupted_post,
        ),
    ]
}


def run_tool(name: str, args: dict) -> dict:
    """Run an ordinary tool. Errors come back as a result the model can read."""
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": f"unknown tool {name!r}"}
    if tool.needs_approval:
        return {"error": f"{name} needs approval and can't be run directly"}
    try:
        return tool.run(**args)
    except ToolError as exc:
        return {"error": str(exc)}
    except TypeError as exc:
        return {"error": f"bad arguments for {name}: {exc}"}
    except Exception as exc:  # e.g. a corrupt invoice file: report it, don't end the turn
        return {"error": f"{name} failed: {type(exc).__name__}: {exc}"}
