> 感谢作者提供的思路：[Thoughts on a TypeSafe coding agent](https://docs.google.com/document/d/1G61uUB0FifUnmmrPzFQojZ3KpczYKmXGpgEXDJ2l_Zg/edit?tab=t.0)。本文将其中值得借鉴的设计思想，结合 jet 当前代码，整理成一份通俗易懂的技术说明。

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

> **Reading note.** This document describes the implementation in this
> repository, not an abstract promise about future features. Names such as
> RLCD and judge are shorthand for the modules listed in the final section.

> Prefer a visual walkthrough? Open the [interactive HTML guide](design.html).

## What problem is jet solving?

A conventional coding agent usually puts the entire conversation into the next
LLM request. That approach is easy to implement, but three costs grow together:

1. old, irrelevant conversation consumes input tokens;
2. every tool schema is sent even when the tool will not be used;
3. the same model is asked to plan, choose tools, execute code, and judge its
   own result.

jet separates these jobs. The expensive model remains responsible for useful
code and explanations, while a cheap structured-judgment model answers the
many small questions around it. The application, not either model, owns the
budget, permissions, recursion limits, duplicate-call detection, and final
stop conditions.

## The complete lifecycle of one request

```text
user request
    │
    ├─ persist as a chunk and load project memory
    ├─ start planning in the background
    │
    └─ repeat until done or bounded:
         ├─ rebuild context from stored chunks
         ├─ select likely tools and inject only their full schemas
         ├─ stream the worker LLM response
         ├─ apply policy, ask for approval, and execute tools
         ├─ persist assistant/tool evidence
         └─ verify the current answer
```

The important property is that the next prompt is not simply “the previous
prompt plus one more turn”. It is a fresh, budgeted view of durable state.

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

### Why this is more than ordinary summarization

Summarization replaces old information and can permanently lose a filename,
error message, or exact tool result. jet keeps the original `Chunk` in SQLite.
The short note or summary is only a rendering for this one context window.
If a later step needs the detail, the same chunk can be selected at `full`
again. A pinned chunk, plan node, subgoal, or memory item is also kept in full.

The builder then packs the newest protocol-safe turns first. An assistant
tool-call message and its tool results are treated as one block, so trimming
history cannot create an invalid OpenAI-style message sequence.

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

The judge is therefore an accelerator and a quality signal, not the ultimate
authority. For example, a judge can rank `run_command` as relevant, but it
cannot grant permission to run a dangerous command. That distinction keeps
model uncertainty from becoming an implicit security policy.

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

Planning also has two less visible safeguards:

- proposed subtasks are deduplicated against existing subgoals;
- a directly actionable subtask is kept as-is, while a bundled subtask can
  recurse only within the configured depth and total-node limits.

This makes “recursive” mean bounded recursive decomposition, not an open-ended
chain of LLM calls.

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

### Permission decision order

For a mutating call, the order is deliberately one-way:

```text
deterministic rule → optional judge safety check → human approval (ask mode)
```

Rules can allow, ask, or deny. A deny is final. The judge may tighten an
uncertain decision, but it cannot turn a denied command into an allowed one.
In non-interactive mode, an `ask` decision becomes a denial when no approver is
available. This is a fail-closed boundary rather than a fallback that silently
changes the user's safety preference.

The built-in rules explicitly cover recursive deletion of root/home paths,
`sudo`, remote-content piped into a shell, fork bombs, workspace writes, and
shell execution. Projects can provide additional rules through configuration.

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

The trace is useful for both debugging and product decisions. It can answer:

- which chunks were hidden, summarized, or kept in full;
- which tool scores caused a schema to be injected;
- whether a permission result came from a rule, the judge, or a human;
- how many provider calls were cache hits and what they cost;
- why the turn stopped: done, budget, max steps, error, or cancellation.

## 8. What is genuinely innovative in this implementation?

The individual ingredients are familiar. The innovation is their combination
into one control loop with explicit ownership boundaries:

| Design | The usual shortcut | jet's design | Practical benefit |
| --- | --- | --- | --- |
| Context | keep or summarize a transcript | durable chunk tree + per-step attention | detail is recoverable and prompts stay small |
| Model roles | one model does everything | judge routes; worker implements; code enforces | cheaper decisions and clearer failure modes |
| Planning | block before the first action | gate and plan concurrently | simple tasks do not pay planning latency |
| Tools | send every full schema | snippets always, schemas on demand | less prompt noise without hiding capabilities |
| Safety | let the model decide freely | deterministic rules first, judge can tighten | model mistakes do not grant permission |
| Completion | trust the final prose | verify against recorded evidence | unsupported claims are exposed |
| Reliability | retry until something passes | bounded retries + explicit inconclusive state | no silent success and no endless loops |

The common theme is **cheap probabilistic advice surrounded by deterministic
boundaries**. Models are good at ranking and interpretation; code is better at
limits, state transitions, and security invariants.

## 9. Boundaries and current trade-offs

These choices are intentional and should be understood before deploying jet:

- The judge adds network calls. Caching and batched questions reduce the cost,
  but a fully local, single-model agent may still have lower absolute latency
  for tiny tasks.
- “Everything is retained” means the SQLite session store needs lifecycle and
  privacy management. Retention, export, and deletion policy are deployment
  responsibilities.
- The context score is a relevance estimate, not a proof of correctness. A
  bad score can hide useful material, so the UI and trace expose the decision.
- The default safety posture is conservative. Automatic execution is a
  configured mode, not something inferred from a model's confidence.
- Verification checks available evidence; it cannot prove that an unseen
  external system has no side effects.

## 10. Where to find each design in the code

| Concern | Main implementation |
| --- | --- |
| durable chunks and parent links | `src/jet/context/store.py`, `src/jet/core/types.py` |
| attention scoring and levels | `src/jet/context/attention.py` |
| token-budgeted message assembly | `src/jet/context/builder.py` |
| bounded recursive planning | `src/jet/context/decompose.py` |
| per-step agent loop | `src/jet/agent/loop.py` |
| tool relevance routing | `src/jet/tools/selection.py` |
| permission and approval policy | `src/jet/policy/permissions.py` |
| evidence-based verification | `src/jet/agent/verify.py` |
| provider/cache contracts | `src/jet/providers/`, `src/jet/cache.py` |
| JSONL audit trail | `src/jet/tracing.py` |

These modules are intentionally separated: replacing a judge provider should
not require rewriting the policy engine, and changing the UI should not change
how context is selected.

---

## One final analogy

A traditional agent is a court stenographer reading the entire case file to a
judge before every question. jet is a clerk with an index: the case file lives
on a shelf (the chunk tree), and before each question the clerk hands over
precisely the pages that matter — having first checked, for pennies, whether
the question even needs the full file.
