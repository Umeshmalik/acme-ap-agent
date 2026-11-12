# Acme Foods AP agent: a small agent harness

A command-line agent for an accounts payable clerk. It reads invoices, checks vendors and
the ledger, and posts invoices to `ledger.py` only after the clerk approves. The original
brief is in [ASSIGNMENT.md](ASSIGNMENT.md).

## Run it

Uses [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned in `.python-version`, and
dependencies are locked in `uv.lock`.

```bash
uv sync                          # creates .venv with the locked dependencies
cp .env.example .env             # fill in GEMINI_API_KEY and GROQ_API_KEY
uv run ledger.py reset
uv run agent.py --conversation demo        # add --model groq to start on Groq
uv run pytest                              # offline tests, with a fake provider
```

Commands: `/approve`, `/decline <reason>`, `/pending`, `/model <name>`, `/history`, `/help`,
`/quit`. Press Ctrl-C while the agent is working to stop it.

**Input help** (via `prompt_toolkit`):
- Typing `/` opens a menu of commands with what each one does. After `/model`, the menu
  lists providers. Invoice ids are suggested with their vendor and total.
- Grey ghost text suggests the rest of the line from your history. Press → to accept it.
- History is kept across runs in `.agent_history`.
- A bottom bar shows the conversation, the model and how many approvals are waiting.
- Commands are case-insensitive, and typos get a "did you mean".

**Providers:** Gemini `gemini-2.5-flash` (SDK `google-genai`) and Groq `openai/gpt-oss-120b`
(SDK `groq`). You can override them with `GEMINI_MODEL` and `GROQ_MODEL`. A third
connector, `ollama` (local `qwen3:4b`, no key), is there for offline testing, and it shows
how a provider plugs in. To make the stop test easy, `READ_DELAY=3` adds a sleep to
`read_invoice`.

## Layout

| File | Role |
|---|---|
| `agent.py` | CLI only: commands, Ctrl-C handling, and a one-terminal-per-conversation lock |
| `cli_input.py` | Input suggestions, ghost text and history. It falls back to `input()` when piped |
| `core.py` | `Agent`: the loop, approvals, stop, crash repair. It does no I/O; it reports through `on_event` |
| `tools.py` | The `TOOLS` registry and the posting checks (`check_postable`) |
| `providers/` | `base.py` holds the interface and retry. Each provider is one connector file, and `__init__.py` holds the registry |
| `store.py` | SQLite storage in `conversations.db` |

Adding a provider takes one connector file with `name`, `env_key`, `default_model` and
`complete()`, plus one line in `PROVIDERS`. Removing one means deleting both. Nothing else
names a provider. Tools work the same way: a `Tool` with `needs_approval=True` goes
through the approval path automatically.

## What is saved, where, when

Everything is in `conversations.db` (SQLite, git-ignored):

- `conversations` holds the name and the current provider. It is updated on `/model`.
- `messages` holds the history, one row per message. Each row is committed as soon as the
  message exists: the user's message before the model is called, each model reply, and
  each tool result.
- `posts` has one row per post the model asks for. Its status moves through
  `pending_approval` to `posting` to `posted`, or ends at `declined` or `refused`. The
  final status and the tool result are written in one transaction.

Restarting with the same `--conversation` reloads the history, and the approval box comes
back if a post is still waiting.

## Conversation format and switching providers

History is stored in one neutral shape: `user` and `assistant` text, assistant
`tool_calls` with our own ids, and `tool` results that point to a call id. Each connector
converts the full history on every call:

- **Groq** uses the OpenAI style: `tool_calls` with JSON-string arguments, then `role: tool`
  messages.
- **Gemini** uses `function_call` parts. Consecutive tool results are merged into one
  user turn of `function_response` parts.

Data that only one provider understands goes in `meta.<provider>`. For example, Gemini's
thought signatures are replayed to Gemini and ignored by Groq. Calls made by Groq get
Google's documented placeholder signature when sent to Gemini 3 models. Switching
providers only changes which connector converts the history next time, so earlier tool
calls carry over and you can switch back.

## Approval and the posting checks

The model only passes an `invoice_id`. When it asks to post, `check_postable` runs in code
before any approval is shown:

- The invoice file exists, and its id matches `invoice-NN`.
- The vendor is in `vendors.json` and is `active`.
- The total is a positive decimal.
- The ledger has no entry with the same vendor and invoice number.
- The same invoice isn't already waiting for approval in any conversation.

A blocked vendor or a duplicate is refused and returned to the model. No approval is
shown in that case.

Otherwise a pending row is saved, and the box shows exactly what would be posted, taken
from the invoice file and the vendor list. On `/approve` the checks run again, and the
post is refused if anything changed. Then `ledger.post` runs and the entry id goes back to
the model. On `/decline` the reason goes back to the model as the tool result. While an
approval is waiting, chat is refused until the clerk approves or declines.

## Stop

Each turn runs on a worker thread. Ctrl-C sets that turn's cancel event:

- The main thread waits at most about 1 second, then returns to the prompt.
- From then on the abandoned thread can't save or print anything, because every save
  checks the event under a lock.
- Tool calls left without a result get `{"error": "stopped ..."}`, and a short assistant
  note is added. The history stays valid for both providers, and completed reads stay
  visible for the next question.
- Rate-limit backoffs wake up as soon as the event is set.
- An approved `ledger.post` is the one thing that is not cut off. Stop waits for it (about
  2 seconds) and records the result.

## Crashes at awkward moments

- **Killed while waiting for approval:** the row is still `pending_approval`, so the
  approval is shown again on restart.
- **Killed during `ledger.post`:** the row is left at `posting`. Every post passes
  `reference="<conversation>/<call_id>"`. On restart the agent looks for that reference in
  the ledger:
  - If it is found, the row becomes `posted` with the real entry id.
  - If it isn't found, the row goes back to `pending_approval`.

  A post is never retried blindly.
- **Killed mid-turn:** on restart, dangling tool calls are closed the same way as after a
  stop.

## Not handled / next steps

- Live runs are slow on free tiers. 429 errors are retried a few times with a wait, then
  shown as an error.
- A conversation can only be open in one terminal at a time. This uses an `flock` lock,
  which the OS releases if the process dies; Windows has no lock. `/approve` claims the
  post atomically (`pending_approval` to `posting`), so two processes can't both post it.
- Two different conversations asking for the same invoice in the same instant could both
  pass the "already waiting" check. The ledger duplicate check, run again at approval,
  narrows that window but doesn't close it. Closing it fully would need a unique index on
  the open post for each vendor and invoice number.
- There is no streaming and no history trimming for long conversations. Only one
  approval is shown at a time; the others wait in a queue (see `/pending`).
- Duplicates are matched only on vendor plus invoice number, as specified. Fuzzy
  duplicates (same amount and date under a new number) aren't caught.
- With more time: tests against the real SDKs with recorded responses.
