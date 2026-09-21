"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  eventSource: null,
  turnId: null,
  running: false,
  lastSeq: -1,
  assistantBody: null,
  thinking: null,
  planItems: new Map(),
  approvalId: null,
  pinnedToBottom: true,
  usage: { input_tokens: 0, output_tokens: 0 },
  cost: 0,
};

/* ---------- dom helpers ---------- */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

const transcript = () => $("transcript");

function clearEmpty() {
  const empty = $("empty");
  if (empty) empty.remove();
}

function isPinned() {
  const el = transcript();
  return el.scrollHeight - el.scrollTop - el.clientHeight < 80;
}

function scrollToEnd(force) {
  if (force || state.pinnedToBottom) {
    transcript().scrollTop = transcript().scrollHeight;
  }
}

function refreshJump() {
  $("jump").hidden = state.pinnedToBottom;
}

/* ---------- transcript rendering ---------- */

function addUserMessage(text) {
  clearEmpty();
  const wrap = el("div", "msg user");
  wrap.appendChild(el("div", "who", "you"));
  wrap.appendChild(el("div", "body", text));
  transcript().appendChild(wrap);
  scrollToEnd();
}

function startAssistantMessage() {
  const wrap = el("div", "msg assistant streaming");
  wrap.appendChild(el("div", "who", "jet"));
  const body = el("div", "body");
  wrap.appendChild(body);
  transcript().appendChild(wrap);
  state.assistantBody = body;
  state.thinking = null;
  scrollToEnd();
}

function appendAssistantDelta(text) {
  if (!state.assistantBody) startAssistantMessage();
  state.assistantBody.textContent += text;
  scrollToEnd();
}

function appendThinking(text) {
  if (!state.assistantBody) startAssistantMessage();
  if (!state.thinking) {
    const details = el("details", "thinking");
    details.appendChild(el("summary", null, "thinking"));
    const body = el("div");
    details.appendChild(body);
    transcript().insertBefore(details, state.assistantBody.closest(".msg"));
    state.thinking = body;
  }
  state.thinking.textContent += text;
}

function endAssistantMessage() {
  if (state.assistantBody) {
    const msg = state.assistantBody.closest(".msg");
    if (!state.assistantBody.textContent.trim() && msg) {
      msg.remove();
    } else if (msg) {
      msg.classList.remove("streaming");
    }
  }
  state.assistantBody = null;
  state.thinking = null;
}

function addTool(event) {
  clearEmpty();
  const card = el("div", "tool " + (event.ok ? "ok" : "bad"));
  const header = el("header");
  header.appendChild(el("span", "status", event.ok ? "✓" : "✗"));
  header.appendChild(el("span", "name", event.name));
  header.appendChild(el("span", "args", argumentHint(event.arguments)));
  header.appendChild(el("span", "time", `${event.duration_ms}ms`));
  card.appendChild(header);
  if (event.output) card.appendChild(el("pre", null, event.output));
  header.addEventListener("click", () => card.classList.toggle("open"));
  transcript().appendChild(card);
  scrollToEnd();
}

function argumentHint(args) {
  if (!args) return "";
  for (const key of ["command", "path", "pattern", "query"]) {
    if (typeof args[key] === "string") return args[key];
  }
  return JSON.stringify(args).slice(0, 90);
}

function addChip(text, kind) {
  transcript().appendChild(el("div", "chip " + (kind || ""), text));
  scrollToEnd();
}

function addNotice(text, level) {
  transcript().appendChild(el("div", "notice " + (level || ""), text));
  scrollToEnd();
}

function addTurnSummary(event) {
  const total = event.usage.input_tokens + event.usage.output_tokens;
  const bar = el("div", "turn-summary");
  const cell = (label, value, cls) => {
    const c = el("span", "cell " + (cls || ""));
    c.appendChild(el("b", null, value));
    c.appendChild(el("span", null, label));
    bar.appendChild(c);
  };
  cell(event.stopped_reason, "", "status");
  cell("steps", String(event.steps));
  cell("tools", String(event.tool_calls));
  cell("tokens", total.toLocaleString());
  if (event.cost_usd) cell("$" + event.cost_usd.toFixed(4), "", "cost");
  transcript().appendChild(bar);
  scrollToEnd();
}

/* ---------- header / panels ---------- */

function setModels(payload) {
  const verify = payload.verifier_model && payload.verifier_model !== payload.llm_model
    ? ` → verify ${payload.verifier_model}` : "";
  $("models").textContent = `gen ${payload.llm_model}${verify} · judge ${payload.judge}`;
  $("session-id").textContent = payload.session_id || "—";
  $("session-id").title = payload.session_id || "";
  $("s-session").textContent = payload.session_id || "—";
  $("s-gen").textContent = `${payload.llm_profile} · ${payload.llm_model}`;
  $("s-ver").textContent = `${payload.verifier_profile || payload.llm_profile} · ${payload.verifier_model}`;
  $("s-judge").textContent = payload.judge;
  $("s-workspace").textContent = payload.workspace;
  $("s-chunks").textContent = String(payload.chunks ?? "—");
  state.usage = payload.usage || { input_tokens: 0, output_tokens: 0 };
  $("s-usage").textContent = `${state.usage.input_tokens.toLocaleString()} in / ${state.usage.output_tokens.toLocaleString()} out`;
  $("s-cost").textContent = state.cost ? `$${state.cost.toFixed(4)}` : "$0";
  $("s-trace").textContent = payload.trace_path || "—";
  setApprovalMode(payload.approval_mode);
  setConnState(state.running ? "busy" : "on");
}

function setApprovalMode(mode) {
  for (const button of document.querySelectorAll("#approval-mode button")) {
    button.classList.toggle("active", button.dataset.mode === mode);
  }
}

function setConnState(kind) {
  const conn = $("conn");
  conn.classList.toggle("off", kind === "off");
  conn.classList.toggle("busy", kind === "busy");
  conn.title = kind === "off" ? "service disconnected" : kind === "busy" ? "working" : "connected";
}

function setRunning(running) {
  state.running = running;
  $("send").disabled = running;
  $("stop").hidden = !running;
  $("input").placeholder = running
    ? "jet is working… (⌘. to stop)"
    : "Ask jet to do something in this workspace…";
  setConnState(running ? "busy" : "on");
}

function renderPlan(plan) {
  const list = $("plan");
  list.innerHTML = "";
  state.planItems.clear();
  $("plan-empty").hidden = plan.length > 0;
  plan.forEach((item) => {
    const li = el("li");
    li.dataset.status = item.status || "pending";
    li.appendChild(el("span", "dot", statusDot(item.status)));
    li.appendChild(el("span", null, item.text));
    list.appendChild(li);
    state.planItems.set(item.text, li);
  });
}

function statusDot(status) {
  if (status === "done") return "●";
  if (status === "running") return "◐";
  if (status === "failed") return "✗";
  if (status === "skipped") return "◌";
  return "○";
}

function updatePlanForChunk(chunk) {
  if (chunk.kind !== "subgoal") return;
  const li = state.planItems.get(chunk.content);
  if (li && chunk.status) {
    li.dataset.status = chunk.status;
    li.querySelector(".dot").textContent = statusDot(chunk.status);
  }
}

function renderContext(event) {
  $("context-empty").hidden = true;
  const counts = {};
  for (const view of event.views) counts[view.level] = (counts[view.level] || 0) + 1;
  const parts = Object.entries(counts).map(([level, count]) => `${level} ${count}`).join(" · ");
  $("context-stats").textContent =
    `${event.messages} msgs · ${(event.tokens / 1000).toFixed(1)}k tok · ${parts || "—"} · ${event.hidden_chunks} hidden`;
  const tbody = $("views").querySelector("tbody");
  tbody.innerHTML = "";
  for (const view of event.views) {
    const row = document.createElement("tr");
    row.appendChild(el("td", "level-" + view.level, view.level));
    row.appendChild(el("td", null, view.kind));
    row.appendChild(el("td", null, view.source || "—"));
    const tokens = el("td", "num", String(view.tokens));
    row.appendChild(tokens);
    tbody.appendChild(row);
  }
  switchToTab("context");
}

function switchToTab(name) {
  const tab = document.querySelector(`#tabs button[data-tab="${name}"]`);
  if (!tab) return;
  tab.click();
}

function renderRecent(chunks) {
  const list = $("s-recent");
  list.innerHTML = "";
  for (const chunk of chunks.slice(-14).reverse()) {
    const li = el("li");
    li.appendChild(el("span", "k", chunk.kind.replace(/_/g, " ")));
    li.appendChild(el("span", "src", chunk.source || firstLine(chunk.content)));
    list.appendChild(li);
  }
}

function firstLine(text) {
  const line = (text || "").split("\n")[0].trim();
  return line.length > 44 ? line.slice(0, 43) + "…" : line;
}

/* ---------- data ---------- */

async function refreshState() {
  try {
    const response = await fetch("/api/state");
    if (!response.ok) return;
    setModels(await response.json());
    setConnState("on");
  } catch {
    setConnState("off");
  }
}

async function refreshChunks() {
  try {
    const response = await fetch("/api/chunks?limit=200");
    if (!response.ok) return;
    const payload = await response.json();
    $("s-chunks").textContent = String(payload.total ?? payload.chunks.length);
    renderRecent(payload.chunks);
    return payload.chunks;
  } catch {
    return [];
  }
}

/* ---------- event handling ---------- */

function handleEvent(event) {
  if (event.seq !== undefined) {
    if (event.seq <= state.lastSeq) return;
    state.lastSeq = event.seq;
  }
  switch (event.type) {
    case "stream_open":
    case "heartbeat":
    case "step_started":
      return;
    case "turn_started":
      state.lastSeq = -1;
      clearEmpty();
      addUserMessage(event.task);
      startAssistantMessage();
      return;
    case "plan_ready":
      renderPlan(event.subgoals.map((text) => ({ text, status: "pending" })));
      return;
    case "context_built":
      renderContext(event);
      return;
    case "assistant_delta":
      appendAssistantDelta(event.text);
      return;
    case "assistant_thinking":
      appendThinking(event.text);
      return;
    case "assistant_message":
      endAssistantMessage();
      return;
    case "tool_finished":
      addTool(event);
      return;
    case "chunk_added":
      updatePlanForChunk(event.chunk);
      return;
    case "verified":
      addChip(
        event.satisfied
          ? `verified · ${event.verifier} · ${event.reason}`
          : `unverified · ${event.reason}`,
        event.satisfied ? "good" : "bad"
      );
      return;
    case "notice":
      addNotice(event.text, event.level);
      return;
    case "approval_requested":
      showApproval(event);
      return;
    case "approval_resolved":
      hideApproval();
      return;
    case "turn_canceled":
      endAssistantMessage();
      addNotice(event.reason, "warning");
      finishTurn();
      return;
    case "turn_finished": {
      endAssistantMessage();
      addTurnSummary(event);
      if (event.verification && !event.verification.satisfied) {
        addChip(`unverified · ${event.verification.reason}`, "bad");
      }
      finishTurn(event);
      return;
    }
    default:
      return;
  }
}

function finishTurn(event) {
  if (event) {
    state.usage = event.usage || state.usage;
    state.cost = event.cost_usd || 0;
  }
  setRunning(false);
  state.turnId = null;
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
  refreshState();
  refreshChunks();
}

/* ---------- actions ---------- */

async function sendTask() {
  const input = $("input");
  const task = input.value.trim();
  if (!task || state.running) return;
  input.value = "";
  input.style.height = "auto";
  setRunning(true);
  state.lastSeq = -1;
  state.assistantBody = null;
  state.pinnedToBottom = true;
  let response;
  try {
    response = await fetch("/api/turns", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ task }),
    });
  } catch {
    addNotice("could not reach the jet service", "error");
    setRunning(false);
    return;
  }
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    addNotice(`could not start turn: ${detail.detail || response.status}`, "error");
    setRunning(false);
    return;
  }
  const { turn_id } = await response.json();
  state.turnId = turn_id;
  const source = new EventSource(`/api/turns/${turn_id}/events`);
  state.eventSource = source;
  source.onmessage = (message) => handleEvent(JSON.parse(message.data));
  source.onerror = () => {
    if (state.running) setConnState("off");
  };
}

async function stopTurn() {
  if (!state.turnId) return;
  try {
    await fetch(`/api/turns/${state.turnId}/cancel`, { method: "POST" });
  } catch {
    /* the turn_canceled event or connection loss will surface the failure */
  }
}

function showApproval(event) {
  state.approvalId = event.approval_id;
  $("approval-tool").textContent = event.tool;
  $("approval-args").textContent = JSON.stringify(event.arguments, null, 2);
  $("approval-reason").textContent = event.reason;
  $("approval-backdrop").hidden = false;
  $("approval-allow").focus();
}

function hideApproval() {
  $("approval-backdrop").hidden = true;
  state.approvalId = null;
}

async function decideApproval(allow) {
  if (!state.approvalId || !state.turnId) return;
  const approvalId = state.approvalId;
  hideApproval();
  await fetch(`/api/turns/${state.turnId}/approvals/${approvalId}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ allow }),
  }).catch(() => {});
}

async function newSession() {
  let response;
  try {
    response = await fetch("/api/session/new", { method: "POST" });
  } catch {
    addNotice("could not reach the jet service", "error");
    return;
  }
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    addNotice(`cannot start a new session: ${detail.detail || response.status}`, "error");
    return;
  }
  transcript().innerHTML = "";
  buildEmptyState();
  state.assistantBody = null;
  state.usage = { input_tokens: 0, output_tokens: 0 };
  state.cost = 0;
  $("views").querySelector("tbody").innerHTML = "";
  $("context-stats").textContent = "";
  $("context-empty").hidden = false;
  renderPlan([]);
  await refreshState();
  await refreshChunks();
}

function buildEmptyState() {
  const empty = el("div", "empty");
  empty.id = "empty";
  empty.appendChild(el("div", "empty-mark", "›_"));
  empty.appendChild(el("h1", null, "jet"));
  empty.appendChild(
    el("p", "empty-lede", "New session. Prior state stays on disk but out of context.")
  );
  empty.appendChild(el("p", "hint", "⌘↵ to send · ⌘. to stop"));
  transcript().appendChild(empty);
}

/* ---------- wiring ---------- */

function wire() {
  $("send").addEventListener("click", sendTask);
  $("stop").addEventListener("click", stopTurn);
  $("new-session").addEventListener("click", newSession);
  $("refresh-state").addEventListener("click", refreshState);
  $("approval-allow").addEventListener("click", () => decideApproval(true));
  $("approval-deny").addEventListener("click", () => decideApproval(false));
  $("jump").addEventListener("click", () => {
    state.pinnedToBottom = true;
    scrollToEnd(true);
  });
  transcript().addEventListener("scroll", () => {
    state.pinnedToBottom = isPinned();
    refreshJump();
  });

  const input = $("input");
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(200, input.scrollHeight) + "px";
  });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      sendTask();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.metaKey && event.key === ".") {
      event.preventDefault();
      stopTurn();
      return;
    }
    const modalOpen = !$("approval-backdrop").hidden;
    if (modalOpen) {
      const inTextField = event.target === input;
      if (event.key === "y" && !inTextField) decideApproval(true);
      if ((event.key === "n" || event.key === "Escape") && !inTextField) decideApproval(false);
      return;
    }
  });

  for (const button of document.querySelectorAll("#approval-mode button")) {
    button.addEventListener("click", async () => {
      const response = await fetch("/api/approval-mode", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ mode: button.dataset.mode }),
      });
      if (response.ok) setApprovalMode(button.dataset.mode);
    });
  }

  for (const button of document.querySelectorAll("#tabs button")) {
    button.addEventListener("click", () => {
      for (const tab of document.querySelectorAll("#tabs button")) {
        tab.classList.toggle("active", tab === button);
      }
      for (const panel of document.querySelectorAll(".tab-panel")) {
        panel.classList.toggle("active", panel.id === "panel-" + button.dataset.tab);
      }
    });
  }

  for (const starter of document.querySelectorAll(".starter")) {
    starter.addEventListener("click", () => {
      $("input").value = starter.dataset.task;
      $("input").focus();
      $("input").dispatchEvent(new Event("input"));
    });
  }

  $("session-id").addEventListener("click", () => {
    navigator.clipboard?.writeText($("session-id").textContent).catch(() => {});
  });
}

wire();
refreshState();
refreshChunks();
