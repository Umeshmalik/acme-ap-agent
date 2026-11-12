"""The rules for posting an invoice to the ledger, enforced in code rather than by the model.

    check_postable(invoice_id)        may this invoice be posted? and exactly what would be
    post_to_ledger(details, ref)      post it (only after the clerk approves)
    find_interrupted_post(details, ref)  did a post that was cut off reach the ledger?

Everything posted comes from the invoice file and vendors.json, never from the model: the
model only names an invoice_id.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

import ledger
from data import ToolError, find_vendor, load_invoice


def check_postable(invoice_id: str) -> dict:
    """{"ok": True, "details": {...what would be posted...}} or {"ok": False, "reason": "..."}."""
    try:
        invoice = load_invoice(invoice_id)
    except ToolError as exc:
        return {"ok": False, "reason": str(exc)}
    vendor = find_vendor(invoice["vendor_name"])
    reason = _vendor_problem(invoice, vendor) or _amount_problem(invoice) or _already_posted(invoice, vendor)
    if reason:
        return {"ok": False, "reason": reason}
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


def _vendor_problem(invoice: dict, vendor: dict | None) -> str | None:
    if vendor is None:
        return f"vendor {invoice['vendor_name']!r} is not in the vendor list"
    if vendor["status"] != "active":
        reason = vendor.get("blocked_reason") or vendor["status"]
        return f"vendor {vendor['vendor_id']} ({vendor['name']}) is blocked: {reason}"
    return None


def _amount_problem(invoice: dict) -> str | None:
    """The total must be a positive decimal that equals the lines plus the taxes."""
    try:
        total = Decimal(invoice["total"])
    except (InvalidOperation, TypeError):
        return f"invoice total is not a number: {invoice['total']!r}"
    if not total.is_finite() or total <= 0:
        return f"invoice total must be positive: {invoice['total']}"
    lines = sum((Decimal(line["amount"]) for line in invoice.get("lines", [])), Decimal(0))
    taxes = sum((Decimal(tax["amount"]) for tax in invoice.get("taxes", [])), Decimal(0))
    subtotal = Decimal(invoice.get("subtotal", lines))
    if lines != subtotal or subtotal + taxes != total:
        return f"invoice doesn't add up: lines {lines}, subtotal {subtotal}, taxes {taxes}, total {total}"
    return None


def _already_posted(invoice: dict, vendor: dict) -> str | None:
    existing = ledger.entries(vendor_id=vendor["vendor_id"], invoice_number=invoice["invoice_number"])
    if existing:
        return (
            f"invoice {invoice['invoice_number']} from {vendor['vendor_id']} is already"
            f" in the ledger as {existing[0]['entry_id']}"
        )
    return None


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
