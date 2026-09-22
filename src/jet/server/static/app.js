"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  eventSource: null,
  turnId: null,
  running: false,
  lastSeq: -1,
  assistantBody: null,
  thinking: null,
  thinkingCollapsed: false,
  userMessageShown: false,
  planItems: new Map(),
  approvalId: null,
  pinnedToBottom: true,
  startedAt: null,
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

/* ---------- minimal safe markdown (DOM-built, no innerHTML of data) ---------- */

function renderMarkdown(container, raw) {
  const text = raw || "";
  try {
    container.innerHTML = "";
    const segments = text.split(/```/);
    segments.forEach((segment, index) => {
      if (index % 2 === 1) {
        let code = segment;
        const newline = code.indexOf("\n");
        if (newline !== -1 && !code.slice(0, newline).includes(" ")) code = code.slice(newline + 1);
        const pre = el("pre", "md-code");
        pre.appendChild(el("code", null, code.replace(/\n$/, "")));
        container.appendChild(pre);
        return;
      }
      const prose = el("div", "md-prose");
      let list = null;
      const closeList = () => {
        if (list) {
          prose.appendChild(list);
          list = null;
        }
      };
      for (const line of segment.split("\n")) {
        const heading = line.match(/^(#{1,4})\s+(.*)$/);
        const bullet = line.match(/^\s*[-*]\s+(.+)$/);
        if (heading) {
          closeList();
          prose.appendChild(el("div", "md-h", heading[2]));
        } else if (bullet) {
          if (!list) list = el("ul", "md-list");
          const item = el("li");
          inlineMd(item, bullet[1]);
          list.appendChild(item);
        } else if (!line.trim()) {
          closeList();
        } else {
          closeList();
          const paragraph = el("p");
          inlineMd(paragraph, line);
          prose.appendChild(paragraph);
        }
      }
      closeList();
      if (prose.childNodes.length) container.appendChild(prose);
    });
  } catch {
    container.textContent = text;
  }
}

function inlineMd(parent, text) {
  const pattern = /(`[^`]+`|\*\*[^*]+\*\*)/g;
  let last = 0;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last) parent.appendChild(document.createTextNode(text.slice(last, match.index)));
    const token = match[0];
    if (token.startsWith("`")) parent.appendChild(el("code", "md-inline", token.slice(1, -1)));
    else parent.appendChild(el("strong", null, token.slice(2, -2)));
    last = match.index + token.length;
  }
  if (last < text.length) parent.appendChild(document.createTextNode(text.slice(last)));
}

/* ---------- transcript rendering ---------- */

async function copyText(text, button) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const helper = el("textarea");
    helper.value = text;
    document.body.appendChild(helper);
    helper.select();
    document.execCommand("copy");
    helper.remove();
  }
  const original = button.textContent;
  button.textContent = t("copied");
  setTimeout(() => (button.textContent = t("copy")), 1200);
}

function attachCopyButton(messageEl, getText) {
  const button = el("button", "copy-btn", t("copy"));
  button.addEventListener("click", () => copyText(getText(), button));
  messageEl.appendChild(button);
}

function addUserMessage(text) {
  clearEmpty();
  const wrap = el("div", "msg user");
  wrap.appendChild(el("div", "who", t("you")));
  wrap.appendChild(el("div", "body", text));
  attachCopyButton(wrap, () => text);
  transcript().appendChild(wrap);
  scrollToEnd();
}

/* ---------- streaming text pump ---------- */

const pump = { queue: "", raf: null };

function collapseThinking() {
  if (state.thinking && !state.thinkingCollapsed) {
    const details = state.thinking.closest("details");
    if (details) details.open = false;
    state.thinkingCollapsed = true;
  }
}

function appendAssistantDelta(text) {
  if (!state.assistantBody) startAssistantMessage();
  collapseThinking();
  pump.queue += text;
  startPump();
}

function startPump() {
  if (pump.raf) return;
  const tick = () => {
    const body = state.assistantBody;
    if (!body || !pump.queue.length) {
      pump.raf = null;
      return;
    }
    // Drain proportionally to the backlog: when the model bursts ahead the
    // rate rises to catch up, and it never falls visibly behind, so output
    // reads as one fast, fluid stream instead of chunky jumps.
    const step = Math.max(2, Math.ceil(pump.queue.length / 6));
    body.textContent += pump.queue.slice(0, step);
    pump.queue = pump.queue.slice(step);
    scrollToEnd();
    pump.raf = requestAnimationFrame(tick);
  };
  pump.raf = requestAnimationFrame(tick);
}

function flushStream() {
  if (state.assistantBody && pump.queue) {
    state.assistantBody.textContent += pump.queue;
    pump.queue = "";
  }
  if (pump.raf) {
    cancelAnimationFrame(pump.raf);
    pump.raf = null;
  }
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
    const details = el("details", "thinking open");
    details.open = true;
    details.appendChild(el("summary", null, t("thinking")));
    const body = el("div");
    details.appendChild(body);
    transcript().insertBefore(details, state.assistantBody.closest(".msg"));
    state.thinking = body;
    state.thinkingCollapsed = false;
  }
  state.thinking.textContent += text;
  const details = state.thinking.closest("details");
  if (details && details.open) {
    state.thinking.scrollTop = state.thinking.scrollHeight;
  }
  scrollToEnd();
}

function endAssistantMessage() {
  flushStream();
  collapseThinking();
  if (state.assistantBody) {
    const msg = state.assistantBody.closest(".msg");
    if (!state.assistantBody.textContent.trim() && msg) {
      msg.remove();
    } else if (msg) {
      msg.classList.remove("streaming");
      const raw = state.assistantBody.textContent;
      renderMarkdown(state.assistantBody, raw);
      state.assistantBody.classList.add("md");
      attachCopyButton(msg, () => raw);
    }
  }
  state.assistantBody = null;
  state.thinking = null;
}

function addTool(event) {
  clearEmpty();
  // Replace any pending "writing" card for this tool with the real result.
  const pending = document.querySelector(`.tool[data-pending="${event.name}"]`);
  const card = el("div", "tool " + (event.ok ? "ok" : "bad"));
  const header = el("header");
  header.appendChild(el("span", "status", event.ok ? "✓" : "✗"));
  header.appendChild(el("span", "name", event.name));
  header.appendChild(el("span", "args", argumentHint(event.arguments)));
  header.appendChild(el("span", "time", `${event.duration_ms}ms`));
  card.appendChild(header);
  if (event.output) card.appendChild(el("pre", null, event.output));
  header.addEventListener("click", () => card.classList.toggle("open"));
  if (pending) pending.replaceWith(card);
  else transcript().appendChild(card);
  scrollToEnd();
}

function updateToolWriting(event) {
  let card = document.querySelector(`.tool.pending[data-pending="${event.name}"]`);
  if (!card) {
    card = el("div", "tool pending");
    card.dataset.pending = event.name;
    const header = el("header");
    header.appendChild(el("span", "status spin", "◐"));
    header.appendChild(el("span", "name", event.name));
    const args = el("span", "args", "writing…");
    card.appendChild(header);
    card.appendChild(args);
    transcript().appendChild(card);
    scrollToEnd();
  }
  card.querySelector(".args").textContent = `writing… ${(event.chars / 1000).toFixed(1)}k chars`;
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
  const elapsed = state.startedAt ? (performance.now() - state.startedAt) / 1000 : null;
  const bar = el("div", "turn-summary");
  const cell = (label, value, cls) => {
    const c = el("span", "cell " + (cls || ""));
    c.appendChild(el("b", null, value));
    c.appendChild(el("span", null, label));
    bar.appendChild(c);
  };
  cell("", event.stopped_reason, "status " + event.stopped_reason);
  cell("steps", String(event.steps));
  cell("tools", String(event.tool_calls));
  cell("tokens", total.toLocaleString());
  if (elapsed !== null) cell(elapsed.toFixed(1) + "s", "", "elapsed");
  if (event.cost_usd) cell("$" + event.cost_usd.toFixed(4), "", "cost");
  transcript().appendChild(bar);
  scrollToEnd();
}

/* ---------- header / panels ---------- */

function setModels(payload) {
  if (!localStorage.getItem("jet-theme") && payload.theme) {
    applyTheme(payload.theme);
  }
  if (!localStorage.getItem("jet-lang") && payload.lang) {
    applyLang(payload.lang);
  }
  if (payload.setup_required) {
    showSetup(payload);
  } else {
    $("setup-backdrop").hidden = true;
  }
  $("session-id").textContent = payload.session_id || "—";
  $("session-id").title = payload.session_id || "";
  const ws = payload.workspace || "—";
  const parts = ws.split("/").filter(Boolean);
  $("workspace-name").textContent = parts.length > 1 ? parts[parts.length - 1] : ws;
  $("workspace-chip").title = `${ws} · ${t("chipOpen")}`;
  $("s-workspace").textContent = ws;
  $("s-gen").textContent = `${payload.llm_profile} · ${payload.llm_model}`;
  $("s-ver").textContent = `${payload.verifier_profile || payload.llm_profile} · ${payload.verifier_model}`;
  $("s-judge").textContent = payload.judge;
  $("s-chunks").textContent = String(payload.chunks ?? "—");
  $("s-usage").textContent = `${(payload.usage?.input_tokens ?? 0).toLocaleString()} in / ${(payload.usage?.output_tokens ?? 0).toLocaleString()} out`;
  $("s-cost").textContent = `$${(payload.cost_usd ?? 0).toFixed(4)}`;
  setApprovalMode(payload.approval_mode);
  setConnState(state.running ? "busy" : "on");
}

function setApprovalMode(mode) {
  for (const button of document.querySelectorAll(".mode-switch button")) {
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
  $("working").hidden = !running;
  $("input").placeholder = running
    ? "jet is working… (⌘. to stop)"
    : t("placeholderIdle");
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
  // The live conversation of this turn travels verbatim; always show its size.
  const verbatim = event.verbatim_messages || 0;
  if (verbatim) {
    const row = document.createElement("tr");
    row.appendChild(el("td", "level-full", "full"));
    row.appendChild(el("td", null, "conversation"));
    row.appendChild(el("td", null, `${verbatim} msgs · this turn`));
    row.appendChild(el("td", "num", String(event.verbatim_tokens || 0)));
    tbody.appendChild(row);
  }
  for (const view of event.views) {
    const row = document.createElement("tr");
    row.appendChild(el("td", "level-" + view.level, view.level));
    row.appendChild(el("td", null, view.kind));
    row.appendChild(el("td", null, view.source || "—"));
    row.appendChild(el("td", "num", String(view.tokens)));
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
  for (const chunk of chunks.slice(-12).reverse()) {
    const li = el("li");
    li.appendChild(el("span", "k", chunk.kind.replace(/_/g, " ")));
    li.appendChild(el("span", "src", chunk.source || firstLine(chunk.content)));
    list.appendChild(li);
  }
}

function firstLine(text) {
  const line = (text || "").split("\n")[0].trim();
  return line.length > 40 ? line.slice(0, 43) + "…" : line;
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
      return;
    case "turn_started":
      state.lastSeq = -1;
      clearEmpty();
      if (!state.userMessageShown) {
        // Optimistic render already happened in sendTask; only late
        // subscribers (no optimistic message) add it here.
        addUserMessage(event.task);
        state.userMessageShown = true;
      }
      return;
    case "step_started":
      $("step-num").textContent = String(event.step);
      $("working").hidden = false;
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
      collapseThinking();
      addTool(event);
      return;
    case "tool_writing":
      updateToolWriting(event);
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
    state.cost = event.cost_usd || 0;
  }
  setRunning(false);
  $("working").hidden = true;
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
  state.startedAt = performance.now();
  state.lastSeq = -1;
  state.userMessageShown = false;
  state.assistantBody = null;
  state.pinnedToBottom = true;
  // Your message appears immediately, before jet's reply box ever exists.
  addUserMessage(task);
  state.userMessageShown = true;
  let response;
  try {
    response = await fetch("/api/turns", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ task }),
    });
  } catch {
    addNotice(t("serviceUnreachable"), "error");
    setRunning(false);
    return;
  }
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    addNotice(t("startFailed") + (detail.detail || response.status), "error");
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
    addNotice(t("serviceUnreachable"), "error");
    return;
  }
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    addNotice(t("cannotNewSession") + (detail.detail || response.status), "error");
    return;
  }
  transcript().innerHTML = "";
  buildEmptyState();
  state.assistantBody = null;
  state.cost = 0;
  $("views").querySelector("tbody").innerHTML = "";
  $("context-stats").textContent = "";
  $("context-empty").hidden = false;
  renderPlan([]);
  await refreshState();
  await refreshChunks();
}

const LOGO_SVG =
  '<svg viewBox="0 0 32 32"><path d="M25.6 6.4 6.4 17.2l6.8 1.2 1.2 6.8z" fill="#fff"/><path d="M15.6 17.2l10-10.8" stroke="#bfe0f7" stroke-width="1.4" fill="none" stroke-linecap="round"/></svg>';

function buildEmptyState() {
  const empty = el("div", "empty");
  empty.id = "empty";
  const mark = el("div", "empty-mark");
  mark.innerHTML = LOGO_SVG;
  empty.appendChild(mark);
  empty.appendChild(el("h1", null, "jet"));
  empty.appendChild(
    el("p", "empty-lede", t("emptyLede"))
  );
  empty.appendChild(el("p", "hint", t("emptyHint")));
  transcript().appendChild(empty);
}

/* ---------- theme ---------- */

const MOON_SVG = '<svg viewBox="0 0 24 24" width="16" height="16"><path d="M20.6 14.2A8.6 8.6 0 0 1 9.8 3.4 8.6 8.6 0 1 0 20.6 14.2z" fill="currentColor"/></svg>';
const SUN_SVG = '<svg viewBox="0 0 24 24" width="16" height="16"><circle cx="12" cy="12" r="4.4" fill="currentColor"/><g stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M12 2.6v2.2M12 19.2v2.2M2.6 12h2.2M19.2 12h2.2M5.2 5.2l1.6 1.6M17.2 17.2l1.6 1.6M18.8 5.2l-1.6 1.6M6.8 17.2l-1.6 1.6"/></g></svg>';

function applyTheme(theme) {
  const dark = theme === "dark";
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  try {
    localStorage.setItem("jet-theme", dark ? "dark" : "light");
  } catch {
    /* private mode: theme stays for this session only */
  }
  const button = $("theme-toggle");
  button.innerHTML = dark ? SUN_SVG : MOON_SVG;
  button.title = dark ? t("themeToLight") : t("themeToDark");
}

/* ---------- first-run setup ---------- */

function showSetup(payload) {
  if (payload.typesafe_base_url) $("setup-jev-url").value = payload.typesafe_base_url;
  if (payload.llm_base_url) $("setup-llm-url").value = payload.llm_base_url;
  if (payload.llm_model) $("setup-llm-model").value = payload.llm_model;
  $("setup-error").hidden = true;
  $("setup-backdrop").hidden = false;
}

async function submitSetup() {
  const button = $("setup-save");
  button.disabled = true;
  const body = {
    typesafe_api_key: $("setup-jev-key").value.trim(),
    typesafe_base_url: $("setup-jev-url").value.trim(),
    llm_base_url: $("setup-llm-url").value.trim(),
    llm_api_key: $("setup-llm-key").value.trim(),
    llm_model: $("setup-llm-model").value.trim(),
  };
  const response = await fetch("/api/setup", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  }).catch(() => null);
  button.disabled = false;
  if (!response || !response.ok) {
    const payload = response ? await response.json().catch(() => ({})) : {};
    const error = $("setup-error");
    const detail = payload.detail;
    if (detail && typeof detail === "object" && !detail.detail) {
      // Structured per-provider probe report: name the broken key directly.
      const lines = [];
      if (detail.deepseek && !detail.deepseek.ok) {
        lines.push(`DeepSeek: ${detail.deepseek.message}`);
      }
      if (detail.jev && !detail.jev.ok) {
        lines.push(`Jev · TypeSafe: ${detail.jev.message}`);
      }
      error.textContent = lines.join(" · ") || t("setupSaveFailed");
    } else {
      error.textContent = payload.detail || t("setupSaveFailed");
    }
    error.hidden = false;
    return;
  }
  $("setup-backdrop").hidden = true;
  $("setup-llm-key").value = "";
  $("setup-jev-key").value = "";
  await refreshState();
  await refreshChunks();
}

/* ---------- i18n ---------- */

const I18N = {
  en: {
    launch: "Launch",
    stop: "Stop",
    copy: "Copy",
    copied: "Copied",
    copyFailed: "Copy failed",
    new: "New",
    newTitle: "Start a new session",
    placeholderIdle: "Give jet a task and it takes off…",
    placeholderWorking: "jet is working… (⌘. to stop)",
    emptyLede: "Jet-fast. A small model judges in milliseconds; the big model only sees key context.",
    emptyHint: "⌘↵ send · ⌘. stop · approvals appear as dialogs",
    emptyNewLede: "New session. Prior state stays on disk but out of context.",
    chipOpen: "Open folder…",
    themeToDark: "Switch to night mode",
    themeToLight: "Switch to day mode",
    you: "you",
    thinking: "thinking",
    setupTitle: "Welcome to jet — connect your models",
    setupLede: "Enter the two API keys. Endpoints are prefilled with the official addresses and can be edited. Credentials stay on this machine in ~/.jet/credentials.json (mode 600), never in any repository.",
    deepseekName: "DeepSeek",
    deepseekRole: "generation + verification · required",
    jevName: "Jev · TypeSafe",
    jevRole: "fast judgment (routing / filtering / gating) · recommended",
    labelEndpoint: "Endpoint",
    labelKey: "API Key",
    labelModel: "Model",
    jevHint: "Leave empty to use the offline mock judge (usable, but without RLCD's smart context filtering)",
    setupSave: "Save and connect",
    setupSaveFailed: "Could not save, please retry",
    approvalTitle: "Approval needed",
    deny: "Deny",
    allow: "Allow",
    approvalHint: "y allow · n/esc deny · this decision is final for this action",
    modeLabel: "Approval mode",
    modeInline: "Applies to later actions; this one still needs Allow / Deny",
    dirTitle: "Choose the workspace directory",
    dirUp: "↑ Up",
    dirChoose: "Use this directory",
    cancel: "Cancel",
    dirHint: "The last opened directory is restored on launch",
    dirEmpty: "No subdirectories here",
    serviceUnreachable: "could not reach the jet service",
    startFailed: "could not start turn: ",
    cannotNewSession: "cannot start a new session: ",
    cannotSwitch: "could not switch directory: ",
    workspaceSwitched: "workspace switched to ",
    setupEmptyLede: "New session. Prior state stays on disk but out of context.",
  },
  zh: {
    launch: "启动",
    stop: "停止",
    copy: "复制",
    copied: "已复制",
    copyFailed: "复制失败",
    new: "新会话",
    newTitle: "开始一个新会话",
    placeholderIdle: "给 jet 一个任务，即刻起飞…",
    placeholderWorking: "jet 工作中…（⌘. 停止）",
    emptyLede: "起飞一般的速度。小模型秒判，大模型只看关键上下文。",
    emptyHint: "⌘↵ 发送 · ⌘. 停止 · 审批以弹窗出现",
    emptyNewLede: "新会话。历史状态保留在磁盘上，但不在上下文里。",
    chipOpen: "打开目录…",
    themeToDark: "切换到夜间模式",
    themeToLight: "切换到日间模式",
    you: "你",
    thinking: "思考中",
    setupTitle: "欢迎使用 jet —— 连接你的模型",
    setupLede: "输入两组 API key。端点地址已预填官方地址，可自行修改。凭据只保存在本机 ~/.jet/credentials.json（权限 600），不会进入任何仓库。",
    deepseekName: "DeepSeek",
    deepseekRole: "生成 + 验证 · 必填",
    jevName: "Jev · TypeSafe",
    jevRole: "快速判断（路由 / 筛选 / 门控）· 推荐填写",
    labelEndpoint: "端点地址",
    labelKey: "API Key",
    labelModel: "模型",
    jevHint: "留空则使用离线 mock 判断模型（功能可用，但失去 RLCD 的智能上下文筛选）",
    setupSave: "保存并连接",
    approvalTitle: "需要审批",
    deny: "拒绝",
    allow: "允许",
    approvalHint: "y 允许 · n/esc 拒绝 · 此决定对该操作最终生效",
    modeLabel: "审批模式",
    modeInline: "对之后的操作生效；本次仍需 Allow / Deny",
    dirTitle: "选择工作目录",
    dirUp: "↑ 上级",
    dirChoose: "使用此目录",
    cancel: "取消",
    dirHint: "启动时会自动恢复上次打开的目录",
    dirEmpty: "此目录下没有子目录",
    serviceUnreachable: "无法连接 jet 服务",
    startFailed: "无法开始回合：",
    cannotNewSession: "无法开始新会话：",
    cannotSwitch: "无法切换目录：",
    workspaceSwitched: "工作区已切换到 ",
    setupEmptyLede: "新会话。历史状态保留在磁盘上，但不在上下文里。",
  },
};

let LANG = "en";

function t(key) {
  return (I18N[LANG] && I18N[LANG][key]) || I18N.en[key] || key;
}

function applyLang(lang) {
  LANG = I18N[lang] ? lang : "en";
  document.documentElement.lang = LANG === "zh" ? "zh-CN" : "en";
  try {
    localStorage.setItem("jet-lang", LANG);
  } catch {
    /* session-only fallback */
  }
  for (const node of document.querySelectorAll("[data-i18n]")) {
    node.textContent = t(node.dataset.i18n);
  }
  for (const node of document.querySelectorAll("[data-i18n-placeholder]")) {
    node.placeholder = t(node.dataset.i18nPlaceholder);
  }
  for (const node of document.querySelectorAll("[data-i18n-title]")) {
    node.title = t(node.dataset.i18nTitle);
  }
  const button = $("theme-toggle");
  button.title = document.documentElement.dataset.theme === "dark" ? t("themeToLight") : t("themeToDark");
  for (const copyButton of document.querySelectorAll(".copy-btn")) copyButton.textContent = t("copy");
  if ($("empty")) {
    transcript().innerHTML = "";
    buildEmptyState();
  }
  setRunning(state.running);
  const saved = document.querySelectorAll("#lang-switch button");
  for (const node of saved) node.classList.toggle("active", node.dataset.lang === LANG);
}

/* ---------- workspace selection ---------- */

const dirState = { path: null };

async function openWorkspacePicker() {
  try {
    const bridge = window.pywebview && window.pywebview.api;
    if (bridge && typeof bridge.select_folder === "function") {
      const path = await bridge.select_folder();
      if (path) await chooseWorkspace(String(path));
      return;
    }
  } catch {
    /* native dialog unavailable or dismissed — fall through to the browser */
  }
  await navigateDir("");
  $("dir-backdrop").hidden = false;
}

async function navigateDir(path) {
  const response = await fetch(`/api/fs/list?path=${encodeURIComponent(path || "")}`);
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    addNotice(`cannot open ${path}: ${detail.detail || response.status}`, "error");
    return;
  }
  const payload = await response.json();
  dirState.path = payload.path;
  $("dir-current").textContent = payload.path;
  $("dir-current").title = payload.path;
  $("dir-up").hidden = !payload.parent;
  $("dir-up").dataset.path = payload.parent || "";
  const list = $("dir-list");
  list.innerHTML = "";
  if (!payload.entries.length) {
    list.appendChild(el("li", "empty-dir", t("dirEmpty")));
    return;
  }
  for (const entry of payload.entries) {
    const item = el("li");
    const button = el("button", null, entry.name + "/");
    button.addEventListener("click", () => navigateDir(entry.path));
    item.appendChild(button);
    list.appendChild(item);
  }
}

async function chooseWorkspace(path) {
  $("dir-backdrop").hidden = true;
  const response = await fetch("/api/workspace", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ path }),
  }).catch(() => null);
  if (!response || !response.ok) {
    const detail = response ? await response.json().catch(() => ({})) : {};
    addNotice(t("cannotSwitch") + (detail.detail || t("serviceUnreachable")), "error");
    return;
  }
  transcript().innerHTML = "";
  buildEmptyState();
  state.assistantBody = null;
  $("views").querySelector("tbody").innerHTML = "";
  $("context-stats").textContent = "";
  $("context-empty").hidden = false;
  renderPlan([]);
  await refreshState();
  await refreshChunks();
  addNotice(t("workspaceSwitched") + path, "info");
}

/* ---------- wiring ---------- */

function wire() {
  $("send").addEventListener("click", sendTask);
  $("stop").addEventListener("click", stopTurn);
  $("new-session").addEventListener("click", newSession);
  $("theme-toggle").addEventListener("click", async () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    applyTheme(next);
    await fetch("/api/theme", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ theme: next }),
    }).catch(() => {});
  });
  $("workspace-chip").addEventListener("click", openWorkspacePicker);
  $("dir-up").addEventListener("click", () => navigateDir($("dir-up").dataset.path || ""));
  $("dir-cancel").addEventListener("click", () => ($("dir-backdrop").hidden = true));
  $("dir-choose").addEventListener("click", () => chooseWorkspace(dirState.path || ""));
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

  for (const button of document.querySelectorAll(".mode-switch button")) {
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

  $("session-id").addEventListener("click", () => {
    navigator.clipboard?.writeText($("session-id").textContent).catch(() => {});
  });
  $("setup-save").addEventListener("click", submitSetup);

  for (const button of document.querySelectorAll("#lang-switch button")) {
    button.addEventListener("click", async () => {
      applyLang(button.dataset.lang);
      await fetch("/api/lang", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ lang: button.dataset.lang }),
      }).catch(() => {});
    });
  }
}

try {
  const savedLang = localStorage.getItem("jet-lang");
  if (savedLang) applyLang(savedLang);
} catch {
  /* defaults apply */
}
wire();
refreshState();
refreshChunks();
