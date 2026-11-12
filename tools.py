"""The agent's tools for the accounts payable job, and the checks that guard posting.

Each tool is a `Tool` in the `TOOLS` registry. The agent core only sees this registry: it
runs ordinary tools directly and sends any tool with `needs_approval` through the
approval path (`prepare` -> person approves -> `run`).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable

import ledger

HERE = Path(__file__).parent
INVOICE_DIR = HERE / "invoices"
VENDORS_PATH = HERE / "vendors.json"
_INVOICE_ID = re.compile(r"invoice-\d{2}")


class ToolError(Exception):
    """A tool failed in a way the model should hear about (bad input, missing file)."""


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


# data access


def _vendors() -> list[dict]:
    return json.loads(VENDORS_PATH.read_text())


def _load_invoice(invoice_id: str) -> dict:
    if not isinstance(invoice_id, str) or not _INVOICE_ID.fullmatch(invoice_id):
        raise ToolError(f"invalid invoice_id {invoice_id!r}: expected e.g. 'invoice-01'")
    path = INVOICE_DIR / f"{invoice_id}.json"
    if not path.exists():
        raise ToolError(f"no such invoice: {invoice_id}")
    return json.loads(path.read_text())


def _find_vendor(name_or_id: str) -> dict | None:
    key = name_or_id.strip().casefold()
    for vendor in _vendors():
        if key in (vendor["vendor_id"].casefold(), vendor["name"].casefold()):
            return vendor
    return None


# tools


def list_invoices() -> dict:
    invoices = []
    for path in sorted(INVOICE_DIR.glob("invoice-*.json")):
        invoice = json.loads(path.read_text())
        invoices.append(
            {
                "invoice_id": path.stem,
                "invoice_number": invoice["invoice_number"],
                "vendor_name": invoice["vendor_name"],
                "invoice_date": invoice["invoice_date"],
                "currency": invoice["currency"],
                "total": invoice["total"],
            }
        )
    return {"invoices": invoices}


def read_invoice(invoice_id: str) -> dict:
    delay = float(os.environ.get("READ_DELAY") or 0)
    if delay:
        time.sleep(delay)
    return {"invoice_id": invoice_id, "invoice": _load_invoice(invoice_id)}


def lookup_vendor(name_or_id: str) -> dict:
    vendor = _find_vendor(name_or_id)
    if vendor is None:
        return {"found": False, "query": name_or_id}
    return {"found": True, "vendor": vendor}


def find_ledger_entries(vendor_id: str | None = None, invoice_number: str | None = None) -> dict:
    filters = {k: v for k, v in (("vendor_id", vendor_id), ("invoice_number", invoice_number)) if v}
    return {"entries": ledger.entries(**filters)}


def check_postable(invoice_id: str) -> dict:
    """The checks that decide whether an invoice may be posted, enforced in code.

    Returns {"ok": True, "details": {...}} with exactly what would be posted, or
    {"ok": False, "reason": "..."}. Everything posted comes from the invoice file and
    vendors.json, never from the model.
    """
    try:
        invoice = _load_invoice(invoice_id)
    except ToolError as exc:
        return {"ok": False, "reason": str(exc)}
    vendor = _find_vendor(invoice["vendor_name"])
    if vendor is None:
        return {"ok": False, "reason": f"vendor {invoice['vendor_name']!r} is not in the vendor list"}
    if vendor["status"] != "active":
        reason = vendor.get("blocked_reason") or vendor["status"]
        return {
            "ok": False,
            "reason": f"vendor {vendor['vendor_id']} ({vendor['name']}) is blocked: {reason}",
        }
    try:
        total = Decimal(invoice["total"])
    except (InvalidOperation, TypeError):
        return {"ok": False, "reason": f"invoice total is not a number: {invoice['total']!r}"}
    if not total.is_finite() or total <= 0:
        return {"ok": False, "reason": f"invoice total must be positive: {invoice['total']}"}
    existing = ledger.entries(vendor_id=vendor["vendor_id"], invoice_number=invoice["invoice_number"])
    if existing:
        return {
            "ok": False,
            "reason": f"invoice {invoice['invoice_number']} from {vendor['vendor_id']} is already"
            f" in the ledger as {existing[0]['entry_id']}",
        }
    return {
        "ok": True,
        "details": {
            "invoice_id": invoice_id,
            "vendor_id": vendor["vendor_id"],
            "vendor_name": vendor["name"],
            "invoice_number": invoice["invoice_number"],
            "invoice_date": invoice["invoice_date"],
            "currency": invoice["currency"],
            "amount": invoice["total"],
        },
    }


def post_to_ledger(details: dict, reference: str | None = None) -> dict:
    entry_id = ledger.post(
        vendor_id=details["vendor_id"],
        invoice_number=details["invoice_number"],
        invoice_date=details["invoice_date"],
        currency=details["currency"],
        amount=details["amount"],
        reference=reference,
    )
    return {"posted": True, "entry_id": entry_id, **details}


def find_interrupted_post(details: dict, reference: str) -> dict | None:
    """`ledger.post` saves the entry before returning, so a crash can leave an entry we never
    heard about. The reference we passed identifies it."""
    for entry in ledger.entries(vendor_id=details["vendor_id"], invoice_number=details["invoice_number"]):
        if entry["reference"] == reference:
            return {"posted": True, "entry_id": entry["entry_id"], **details}
    return None


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
