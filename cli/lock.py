"""One terminal per conversation: an OS file lock, released automatically if we die."""

from __future__ import annotations

from pathlib import Path
from typing import TextIO

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None
    import msvcrt

LOCK_DIR = Path(__file__).resolve().parent.parent / ".locks"


def lock_conversation(name: str) -> TextIO:
    """Hold an exclusive lock for this conversation until the returned file is closed."""
    LOCK_DIR.mkdir(exist_ok=True)
    handle = open(LOCK_DIR / f"{name}.lock", "w")
    try:
        if fcntl:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:  # BlockingIOError (POSIX) or PermissionError (Windows)
        handle.close()
        raise SystemExit(f'Conversation "{name}" is already open in another terminal.') from None
    return handle
