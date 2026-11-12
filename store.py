"""Conversation storage in SQLite.

Messages are stored in a provider-neutral shape (see providers.py) and committed one by
one as they happen. The `posts` table tracks every post the model asked for, so a waiting
approval or a half-finished post survives a restart.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).with_name("conversations.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    name TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    conversation TEXT NOT NULL REFERENCES conversations(name),
    seq INTEGER NOT NULL,
    body TEXT NOT NULL,
    PRIMARY KEY (conversation, seq)
);
CREATE TABLE IF NOT EXISTS posts (
    call_id TEXT PRIMARY KEY,
    conversation TEXT NOT NULL REFERENCES conversations(name),
    invoice_id TEXT NOT NULL,
    vendor_id TEXT NOT NULL,
    invoice_number TEXT NOT NULL,
    details TEXT NOT NULL,
    status TEXT NOT NULL,
    entry_id TEXT,
    reason TEXT,
    updated_at TEXT NOT NULL
);
"""

# pending_approval -> posting -> posted
# pending_approval -> declined | refused
POST_STATUSES = ("pending_approval", "posting", "posted", "declined", "refused")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path = DB_PATH):
        self.path = path
        with closing(self._connect()) as conn, conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    # conversations

    def open_conversation(self, name: str, default_model: str) -> tuple[bool, str]:
        """Return (is_new, model), creating the conversation if needed."""
        with closing(self._connect()) as conn, conn:
            row = conn.execute("SELECT model FROM conversations WHERE name = ?", (name,)).fetchone()
            if row:
                return False, row["model"]
            conn.execute(
                "INSERT INTO conversations (name, model, created_at) VALUES (?, ?, ?)",
                (name, default_model, _now()),
            )
            return True, default_model

    def set_model(self, name: str, model: str) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("UPDATE conversations SET model = ? WHERE name = ?", (model, name))

    # messages

    def messages(self, name: str) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT body FROM messages WHERE conversation = ? ORDER BY seq", (name,)
            )
            return [json.loads(row["body"]) for row in rows]

    def append(self, name: str, message: dict) -> None:
        with closing(self._connect()) as conn, conn:
            self._append(conn, name, message)

    def append_many(self, name: str, messages: list[dict]) -> None:
        with closing(self._connect()) as conn, conn:
            for message in messages:
                self._append(conn, name, message)

    @staticmethod
    def _append(conn: sqlite3.Connection, name: str, message: dict) -> None:
        conn.execute(
            "INSERT INTO messages (conversation, seq, body) VALUES (?,"
            " (SELECT COALESCE(MAX(seq), 0) + 1 FROM messages WHERE conversation = ?), ?)",
            (name, name, json.dumps(message)),
        )

    # posts

    def add_pending_post(self, name: str, call_id: str, invoice_id: str, details: dict) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO posts (call_id, conversation, invoice_id, vendor_id, invoice_number,"
                " details, status, updated_at) VALUES (?, ?, ?, ?, ?, ?, 'pending_approval', ?)",
                (
                    call_id,
                    name,
                    invoice_id,
                    details["vendor_id"],
                    details["invoice_number"],
                    json.dumps(details),
                    _now(),
                ),
            )

    def posts(self, status: str, name: str | None = None) -> list[dict]:
        """Posts with a status, oldest first; across all conversations unless `name` is given."""
        sql = "SELECT * FROM posts WHERE status = ?"
        params: tuple = (status,)
        if name is not None:
            sql += " AND conversation = ?"
            params += (name,)
        with closing(self._connect()) as conn:
            rows = conn.execute(sql + " ORDER BY rowid", params).fetchall()
        return [{**dict(row), "details": json.loads(row["details"])} for row in rows]

    def open_posts_for(self, vendor_id: str, invoice_number: str) -> list[dict]:
        """Posts of this invoice that are waiting for approval or in flight, in any conversation."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT call_id, conversation, status FROM posts WHERE vendor_id = ?"
                " AND invoice_number = ? AND status IN ('pending_approval', 'posting')",
                (vendor_id, invoice_number),
            ).fetchall()
        return [dict(row) for row in rows]

    def set_post_status(self, call_id: str, status: str) -> None:
        assert status in POST_STATUSES
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "UPDATE posts SET status = ?, updated_at = ? WHERE call_id = ?",
                (status, _now(), call_id),
            )

    def resolve_post(
        self,
        name: str,
        call_id: str,
        status: str,
        tool_message: dict,
        entry_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Record the final outcome of a post and its tool result in one transaction."""
        assert status in ("posted", "declined", "refused")
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "UPDATE posts SET status = ?, entry_id = ?, reason = ?, updated_at = ?"
                " WHERE call_id = ?",
                (status, entry_id, reason, _now(), call_id),
            )
            self._append(conn, name, tool_message)
