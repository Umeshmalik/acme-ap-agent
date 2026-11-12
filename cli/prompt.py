"""Line input for the CLI: suggestions while typing, ghost-text completion, saved history.

As you type, a menu suggests commands (with what they do), provider names after /model
and invoice ids (with vendor and total). Grey ghost text offers the rest of the line from
your history or the first suggestion; press the right arrow to accept it. When input is
not a terminal (piped, scripted) this falls back to plain `input()`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Iterable

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggest, AutoSuggestFromHistory, Suggestion
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory

from app.tools import list_invoices

Option = tuple[str, str]  # (text to insert, short description)


def suggestions(line: str, commands: dict[str, str], providers: Iterable[str]) -> list[Option]:
    """Options for the word being typed at the end of `line`."""
    stripped = line.lstrip()
    word = "" if line.endswith(" ") else (stripped.split(" ")[-1] if stripped else "")
    if stripped.startswith("/model ") and stripped.count(" ") == 1:
        options = [(name, "provider") for name in providers]
    elif stripped.startswith("/") and " " not in stripped:
        options = list(commands.items())
    elif len(word) >= 2:
        options = [
            (i["invoice_id"], f"{i['vendor_name']}  {i['currency']} {i['total']}")
            for i in list_invoices()["invoices"]
        ]
    else:
        return []
    return [(text, meta) for text, meta in options if text.startswith(word) and text != word]


class _Completer(Completer):
    def __init__(self, options: Callable[[str], list[Option]]):
        self._options = options

    def get_completions(self, document, complete_event):
        line = document.text_before_cursor
        word = "" if line.endswith(" ") else line.split(" ")[-1]
        for text, meta in self._options(line):
            yield Completion(text, start_position=-len(word), display_meta=meta)


class _Suggest(AutoSuggest):
    """History first (like a shell); otherwise the first suggestion for the current word."""

    def __init__(self, options: Callable[[str], list[Option]]):
        self._options = options
        self._history = AutoSuggestFromHistory()

    def get_suggestion(self, buffer, document):
        from_history = self._history.get_suggestion(buffer, document)
        if from_history:
            return from_history
        line = document.text
        word = "" if line.endswith(" ") else line.split(" ")[-1]
        matches = self._options(line)
        if matches and (word or line.strip().startswith("/model")):
            return Suggestion(matches[0][0][len(word):])
        return None


def make_reader(
    history_file: Path,
    commands: dict[str, str],
    providers: Iterable[str],
    status: Callable[[], str],
) -> Callable[[str], str]:
    """Return `read(prompt) -> line`. Raises EOFError on Ctrl-D, KeyboardInterrupt on Ctrl-C."""
    if not sys.stdin.isatty():
        return input
    providers = list(providers)

    def options(line: str) -> list[Option]:
        return suggestions(line, commands, providers)

    session = PromptSession(
        history=FileHistory(str(history_file)),
        completer=_Completer(options),
        auto_suggest=_Suggest(options),
        complete_while_typing=True,
        bottom_toolbar=status,
    )
    return lambda prompt: session.prompt(prompt)
