# How jet works — the ideas behind it

<p align="center">
  <img src="docs/logo.svg" alt="jet logo" width="64" />
</p>

> **Credits.** jet is an implementation of ideas from
> [*Thoughts on a TypeSafe coding agent*](https://docs.google.com/document/d/1G61uUB0FifUnmmrPzFQojZ3KpczYKmXGpgEXDJ2l_Zg/edit?tab=t.0)
> — thank you to the author for sharing the thinking that shaped this design:
> explicit state instead of a growing transcript, a fast judgment model for
> what an LLM would otherwise burn tokens on, and context as something you
> re-derive rather than compact.

This document explains jet's core and novel designs in plain language. It is
written for someone who has used coding agents before and wants to understand
what is different here — no paper-reading required.

---

## The one-sentence version

**Traditional agents carry one ever-growing conversation; jet keeps state as a
queryable tree of facts, and every step, a cheap "judge" model decides what the
expensive model should see.**

Three roles, split by cost:

| Role | Who plays it | Does |
| --- | --- | --- |
| **Judge** | a small fast model (Jev / laya-mlx) | scores, routes, gates, verifies — thousands of times per session |
| **Worker** | the big LLM (DeepSeek Flash) | reads/writes code — only ever sees a freshly-tailored window |
| **Code** | jet itself | owns thresholds, budgets, policies, recursion — never trusts a model with policy |

---

## 1. RLCD — state is a tree, context is a choice

The core abstraction is the **chunk**: one typed unit of session state (a user
message, an assistant reply, a tool result, a subgoal, a project memory file).
Everything the agent has ever seen lives in a SQLite tree, forever, in full.

The novel part is what is *not* there: **no compaction step**. Classic agents
summarize old turns when the transcript grows — lossy, and it happens at the
worst time. jet instead re-derives the window on every step:

1. collect candidate chunks (pre-turn state + pinned memory + subgoals)
2. **meta-attention**: ask the judge "how much of *this chunk* does the agent
   need right now?" for every candidate in one batched call, on one shared
   0-3 scale
3. map scores to levels with code-owned thresholds:
   **hide · note · summary · full**
4. if everything fits uncompressed — keep everything (compression is a budget
   necessity, not a habit)
5. if the budget overflows: shrink the *lowest*-scored chunks first, keep the
   most important in full

Two consequences fall out of this design:

- **Nothing is ever lost.** Old detail stays addressable forever; a session
  restart just re-opens the store and rebuilds a window. There is no "lost the
  thread" failure mode.
- **The context is honest**. The Context pane in the client shows exactly what
  was selected, at which level, and what was hidden — the user sees the same
  thing the model sees.

The current turn always travels **verbatim** (the chat protocol requires
tool-call pairing), and it is never duplicated into memory. Memory = the past,
verbatim = now.

---

## 2. The judge is a different *kind* of model

The judgment model is not a smaller chat model. It answers structured
questions — yes/no probabilities, choices, ordered scores — instead of writing
text:

- *"Is this subgoal already covered by one we have?"* (noul)
- *"Which of these tools will this step actually use?"* (score over a catalog)
- *"Is this shell command safe in this workspace?"* (permission advice)
- *"Does the answer satisfy the goal, given the evidence?"* (verification)

Code owns the thresholds and the policy; the model only supplies calibrated
probabilities. Two details make this cheap and reliable:

- **Fan-out**: independent questions about the same state are asked in one
  request. A 10-tool routing decision plus a plan gate is *one* network call.
- **Disk caching** keyed by (provider, model, state, questions): identical
  questions cost nothing the second time. Real runs show 17/22 cache hits.

When the judge is unavailable, every decision site degrades instead of dying:
context goes all-full, tool routing sends everything, planning skips, the
verifier reports "inconclusive" and the answer stands — the trace records the
degradation.

---

## 3. Plans are async, gated, and never blocking

Decomposition ("recursive" in RLCD) is deliberately in the way of nothing:

1. a gate judgment asks *"does this goal even need a plan?"* — single actions
   skip planning entirely (a fast judgment call replaces an expensive planning
   round trip)
2. the planning LLM runs **concurrently with step 1**; the loop adopts the
   plan whenever it has landed and never waits for it
3. recursion is capped (depth 1 by default); dedup and "is this subgoal
   directly actionable?" judgments fan out in parallel for the whole batch

A task like "fix this test" therefore pays near-zero planning latency, while a
genuinely multi-part goal still gets a visible, resumable plan tree — each
subgoal is a chunk with live status (pending → running → done), shown in the
client's Plan pane.

---

## 4. Tools: snippets always, schemas on demand, writes with receipts

- The system prompt always carries **compact one-line snippets** of every tool.
  Full JSON schemas are injected **only for the tools the judge selected for
  this step** — the model knows what exists without paying for every schema.
- Every write/run goes through **policy**: deterministic rules first (delete
  patterns, sudo, pipe-to-shell), then a judge safety opinion that may only
  *tighten* a decision, then — in Ask mode — a human dialog. Denials are final
  and recorded; the loop fails closed when no approver exists.
- Tool calls stream their arguments: while the model writes a 25 KB file you
  see a live card (`write_file · writing… 21.7k chars`) instead of a silent
  caret. Repeating an identical call is refused with a pointer to the recorded
  result.

---

## 5. Verification without theater

Every turn ends through one of two gates:

- **Tool turns** (work was executed): cross-model verification — the judge and
  a *different* LLM check goal vs. answer vs. recorded evidence in parallel.
  Both must agree.
- **Zero-tool turns** (conversational answers): the fast judge only, so a
  "hello" costs no verification wait.

Failures feed back as explicit notes ("address this finding") with a bounded
retry count; three failed verifications return the best current answer
honestly marked unverified — an unverified answer beats an endless retry loop.
Malformed or out-of-budget verifier output is *inconclusive*, never silently
passing or failing.

---

## 6. Guardrails that fail loud, not silent

- **finite `max_tokens`** per completion — a reasoning model can otherwise
  stream degenerate output for minutes while every timeout keeps resetting
- **stream idle watchdog** — 90s of SSE silence is a hung connection, said
  out loud
- **duplicate-call refusal** — loops are broken with an instruction to use
  the recorded result
- **step-pressure note** — partway through, the agent is told to conclude
  with its best answer instead of exploring forever
- **anti-harness rule** in the system prompt — verify once, then stop; no
  invented verification scripts
- **first-run setup dialog** — both API keys probed on save, per-provider
  errors; bad credentials never touch disk

---

## 7. Tracing: every decision is on the record

Each session writes one JSONL trace: every judgment (purpose, provider,
latency, cache hit), every LLM turn (tokens, cost), every tool call, every
policy verdict. The client's Context/Session panes read from the same
structures — what you see is what the agent used, token costs included.

---

## One final analogy

A traditional agent is a court stenographer reading the entire case file to a
judge before every question. jet is a clerk with an index: the case file lives
on a shelf (the chunk tree), and before each question the clerk hands over
precisely the pages that matter — having first checked, for pennies, whether
the question even needs the full file.
