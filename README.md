# jet

A coding agent built on **TypeSafe** judgments and **Recursive LLM Context
Decomposition (RLCD)**.

> Author: **alex (arczhi)** · License: **CC BY-NC 4.0** — free to share and adapt
> with attribution; **commercial use is not permitted**. See [LICENSE](LICENSE).

jet does not carry one ever-growing transcript. State is explicit, decomposed
into a tree of typed chunks, and a fast System One model (`Jev` / any
OpenAI-compatible judgment endpoint such as `laya-mlx`) continuously scores what
should be in the next context window — hide it, summarize it, or include it in
full. The large LLM (default: `deepseek-v4.1-flash`) is only used for generation
and verification.

Named for speed: most decision traffic goes to the small judgment model, the
expensive model sees a small, query-shaped context.

## Install

```bash
uv sync
cp .env.example .env   # fill in keys
uv run jet doctor      # verify config and provider reachability
uv run jet app         # native macOS client
uv run jet             # terminal client
uv run jet run "explain src/jet/agent/loop.py"
```

## The client

`jet app` starts a local HTTP/SSE service and opens a real window (WKWebView via
pywebview) on macOS; `--browser` uses your default browser instead.

The window is a thin view over the same engine events the terminal client uses:

- transcript with streamed answers, tool cards, and verification chips
- **Plan** pane — the RLCD subgoal tree with live status
- **Context** pane — every chunk the meta-attention pass selected for the current
  step, with its level (`full` / `summary` / `note`) and token cost
- **Session** pane — models in use, chunks, tokens, cost, trace path
- approval dialogs with Allow/Deny (⌘↵ send, ⌘. stop, y/n decide)
- approval mode is switchable at runtime (Ask / Auto / Deny); agents never
  auto-approve silently in Ask mode

Everything the client shows is persisted: chunks in
`~/.jet/sessions/<id>/chunks.db`, decisions in `trace.jsonl`, judgments in
`~/.jet/cache/judgments.db`.

## Configuration

Config resolves in this order (highest priority first):

1. CLI flags
2. environment variables (`JET_*`)
3. `.env` in the project
4. `jet.toml` in the project, then `~/.config/jet/config.toml`
5. defaults

Key variables (see `.env.example`):

| Variable | Meaning |
| --- | --- |
| `JET_JUDGE_PROVIDER` | `typesafe`, `openai_compat`, or `mock` |
| `JET_TYPESAFE_BASE_URL` / `JET_TYPESAFE_API_KEY` / `JET_TYPESAFE_MODEL` | Jev via TypeSafe API |
| `JET_JUDGE_OPENAI_BASE_URL` / `JET_JUDGE_OPENAI_API_KEY` / `JET_JUDGE_OPENAI_MODEL` | local judgment model (laya-mlx) |
| `JET_LLM_PROFILE` | active LLM profile name |
| `JET_LLM_PROFILES` | JSON map of named LLM endpoints (`base_url`, `api_key`, `model`, `extra_headers`, prices) |
| `JET_VERIFIER_LLM_PROFILE` | optional different LLM for verification (cross-model review) |

Example: generate with OpenCode Go, verify with the official DeepSeek API.

```bash
JET_LLM_PROFILE=zen
JET_VERIFIER_LLM_PROFILE=official
JET_LLM_PROFILES={"zen":{"base_url":"https://opencode.ai/zen/go/v1","api_key":"...","model":"deepseek-v4.1-flash","extra_headers":{"x-opencode-session":"{session_id}"}},"official":{"base_url":"https://api.deepseek.com","api_key":"...","model":"deepseek-flash"}}
```

Secrets are never written to the repo. Judgment results are cached on disk by
content hash, so identical decisions are not paid for twice.

## Architecture

```text
input -> context assembly (RLCD) -> routing -> generation -> tool policy -> execution
              ^                                              |
              +------------- chunks / judgments <-------------+
```

- `providers/` — plugin contract for judgment models and LLMs; both are
  swappable by config.
- `context/` — chunk store, recursive decomposition of goals, meta-attention
  scoring, budget-aware context builder.
- `tools/` — tool catalog exposed as compact snippets; full schemas are injected
  only for the tools a judgment pass selects.
- `policy/` — deterministic rules first; judgment and human approval for
  ambiguous operations. Fails closed.
- `agent/` — the loop, session persistence, verification.
- `server/` — FastAPI + SSE service and the single-page client for the desktop window.
- `tracing.py` — every judgment, LLM turn, tool call, and policy decision is
  written to a JSONL trace per session for audit and cost accounting.

## Development

```bash
make ci      # lint + typecheck + tests (unit + HTTP integration)
make smoke   # end-to-end run against a local fake provider
make app     # native macOS client
make run     # terminal client
```

## License

[CC BY-NC 4.0](LICENSE) © alex (arczhi). You may share and modify this project
freely for non-commercial purposes, provided you credit the original author and
indicate changes.
