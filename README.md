# Acme Foods AP agent: a small agent harness

A command-line agent for an accounts payable clerk. It reads invoices, checks vendors and the
ledger, and posts to `ledger.py` only after the clerk approves. Brief: [ASSIGNMENT.md](ASSIGNMENT.md). How it works inside: [ARCHITECTURE.md](ARCHITECTURE.md).

## Run it

```bash
uv sync                                    # Python 3.12, locked deps
cp .env.example .env                       # GEMINI_API_KEY, GROQ_API_KEY
uv run ledger.py reset
uv run agent.py --conversation demo        # --model groq to start on Groq
uv run pytest                              # offline tests, fake provider
```

Commands: `/approve`, `/decline <reason>`, `/pending`, `/model <name>`, `/history`, `/help`, `/quit`.
Ctrl-C stops the agent mid-turn. Typing `/` shows a command menu (`prompt_toolkit`).

**Providers:** Gemini `gemini-3.8-flash` (`google-genai`) and Groq `openai/gpt-oss-120b` (`groq`),
overridable with `GEMINI_MODEL` / `GROQ_MODEL`. Gemini's free tier allows only ~20 requests a day.
`ollama` (local `qwen3:4b`) is for offline testing. `READ_DELAY=3` slows `read_invoice` for the stop test.
A provider is one file in `providers/` plus one line in `PROVIDERS`; nothing else names a provider.

## What is saved, where, when

All in `conversations.db` (SQLite): `conversations` (name, current provider, updated on `/model`);
`messages` (one row per message, committed as soon as it exists: the user's message before the model
is called, each reply, each tool result); `posts` (one row per requested post:
`pending_approval` → `posting` → `posted`, or `declined` / `refused`; final status and tool result in
one transaction). Restarting with the same `--conversation` reloads history and any waiting approval.

## Conversation format and switching providers

History is one neutral shape: `user`/`assistant` text, assistant `tool_calls` with our own ids, and
`tool` results pointing at a call id. Each connector converts the full history on every call (Groq:
OpenAI-style `tool_calls` + `role: tool`; Gemini: `function_call` parts, consecutive results merged
into one turn). Provider-only data lives in `meta.<provider>` (e.g. Gemini thought signatures; Groq
calls get Google's documented placeholder). Switching only changes which connector converts next.

## Approval and the posting checks

The model passes only an `invoice_id`. `check_postable` (`app/posting_rules.py`) runs in code before any approval:
file exists, vendor is in `vendors.json` and `active`, total is a positive decimal that equals lines +
taxes, no ledger entry with the same vendor + invoice number, not already open in any conversation
(a partial unique index in SQLite enforces this, so two conversations racing can't both get an approval). Blocked vendors and
duplicates are refused back to the model with no approval shown. Otherwise the box shows exactly what
would be posted. `/approve` re-runs the checks, then calls `ledger.post`; `/decline` returns the reason
to the model. The program itself prints the entry id once posted, so the clerk knows even if the
model call after it fails. Chat is refused while an approval is waiting.

## Stop

Each turn runs on a worker thread; Ctrl-C sets its cancel event. The prompt returns within ~1 s. The
abandoned thread can't save or print, because every save checks the event under a lock. Tool calls
left without a result get `{"error": "stopped ..."}` plus a short assistant note, so history stays
valid for both providers. Backoffs wake on the event. The one thing not cut off is an approved
`ledger.post`: stop waits for it (~2 s) and records the result; a further Ctrl-C quits at once and
the post is settled from the ledger on the next start, exactly like a crash. The CLI waits on a `done` event, not
`Thread.join()`: a Ctrl-C inside `join()` makes CPython mark the thread finished, which once let a post
complete unattended.

## Crashes at awkward moments

- **Killed while waiting for approval:** row is still `pending_approval`; the approval reappears.
- **Killed during `ledger.post`:** row stays `posting`. Every post carries
  `reference="<conversation>/<call_id>"`; on restart the ledger is searched for it. Found → `posted`
  with the real entry id. Not found → back to `pending_approval`. Never retried blindly.
- **Killed mid-turn:** dangling tool calls are closed as after a stop.
- **Ledger call fails mid-session, outcome unknown:** the row stays `posting` and is settled from the
  ledger (same reference lookup) before the next turn, so the conversation doesn't break.
- **Storage errors** (disk full, locked or corrupt DB) are reported instead of a traceback.

## Not handled / next steps

- Providers: 45 s timeout; timeouts, dropped connections, 5xx and short 429s retried with backoff
  (5 tries, shown on screen). A quota resetting in over a minute (Gemini daily tier) fails fast.
- One terminal per conversation via `flock` / `msvcrt.locking` (released if the process dies).
  `/approve` claims the post atomically, so two processes can't both post it.
- Only the last ~200 messages are sent to the model (cut at a turn boundary); older turns are
  forgotten, not summarised. No streaming. One approval at a time; others queue (`/pending`).
- A post left `posting` by a crashed conversation blocks that invoice elsewhere until that
  conversation is reopened (the refusal says which one).
- Duplicates match only vendor + invoice number, as specified; fuzzy duplicates aren't caught.
- With more time: tests against the real SDKs with recorded responses.
