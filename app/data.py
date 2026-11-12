"""Reading the business data: invoice files and the vendor list. Read-only."""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # the project folder
INVOICE_DIR = HERE / "invoices"
VENDORS_PATH = HERE / "vendors.json"
_INVOICE_ID = re.compile(r"invoice-\d{2}")


class ToolError(Exception):
    """A tool failed in a way the model should hear about (bad input, missing file)."""


def invoice_ids() -> list[str]:
    return [path.stem for path in sorted(INVOICE_DIR.glob("invoice-*.json"))]


def load_invoice(invoice_id: str) -> dict:
    """The invoice file as a dict. The id is checked first, so it can't name another file."""
    if not isinstance(invoice_id, str) or not _INVOICE_ID.fullmatch(invoice_id):
        raise ToolError(f"invalid invoice_id {invoice_id!r}: expected e.g. 'invoice-01'")
    path = INVOICE_DIR / f"{invoice_id}.json"
    if not path.exists():
        raise ToolError(f"no such invoice: {invoice_id}")
    return json.loads(path.read_text())


def vendors() -> list[dict]:
    return json.loads(VENDORS_PATH.read_text())


def find_vendor(name_or_id: str) -> dict | None:
    """Match a vendor by id or exact name, ignoring case."""
    key = name_or_id.strip().casefold()
    for vendor in vendors():
        if key in (vendor["vendor_id"].casefold(), vendor["name"].casefold()):
            return vendor
    return None
