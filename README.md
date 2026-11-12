# Acme Foods AP agent: a small agent harness

A command-line agent for an accounts payable clerk. It reads invoices, checks vendors and the
ledger, and posts to `ledger.py` only after the clerk approves. Brief: [ASSIGNMENT.md](ASSIGNMENT.md).

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

The model passes only an `invoice_id`. `check_postable` (`tools.py`) runs in code before any approval:
file exists, vendor is in `vendors.json` and `active`, total is a positive decimal, no ledger entry
with the same vendor + invoice number, not already waiting in any conversation. Blocked vendors and
duplicates are refused back to the model with no approval shown. Otherwise the box shows exactly what
would be posted. `/approve` re-runs the checks, then calls `ledger.post`; `/decline` returns the reason
to the model. Chat is refused while an approval is waiting.

## Stop

Each turn runs on a worker thread; Ctrl-C sets its cancel event. The prompt returns within ~1 s. The
abandoned thread can't save or print, because every save checks the event under a lock. Tool calls
left without a result get `{"error": "stopped ..."}` plus a short assistant note, so history stays
valid for both providers. Backoffs wake on the event. The one thing not cut off is an approved
`ledger.post`: stop waits for it (~2 s) and records the result. The CLI waits on a `done` event, not
`Thread.join()`: a Ctrl-C inside `join()` makes CPython mark the thread finished, which once let a post
complete unattended.

## Crashes at awkward moments

- **Killed while waiting for approval:** row is still `pending_approval`; the approval reappears.
- **Killed during `ledger.post`:** row stays `posting`. Every post carries
  `reference="<conversation>/<call_id>"`; on restart the ledger is searched for it. Found → `posted`
  with the real entry id. Not found → back to `pending_approval`. Never retried blindly.
- **Killed mid-turn:** dangling tool calls are closed as after a stop.

## Not handled / next steps

- Providers: 45 s timeout; timeouts, 5xx and short 429s retried with backoff (5 tries, shown on
  screen). A quota resetting in over a minute (Gemini daily tier) fails fast with the reset time.
- One terminal per conversation via `flock` (released if the process dies). **Windows has no lock.**
  `/approve` claims the post atomically, so two processes can't both post it.
- **Two conversations requesting the same invoice at the same instant** can both pass the "already
  waiting" check. The ledger re-check at approval narrows but doesn't close this; a unique index on
  open posts per vendor + invoice number would.
- No streaming, no history trimming. One approval shown at a time; others queue (`/pending`).
- Duplicates match only vendor + invoice number, as specified; fuzzy duplicates aren't caught.
- With more time: tests against the real SDKs with recorded responses.
