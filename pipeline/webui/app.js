"use strict";
/* voicepipe web UI — no innerHTML; all DOM built with createElement/textContent */

const BUILD = "v12";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const clear = n => { while (n && n.firstChild) n.removeChild(n.firstChild); return n; };
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v === true) e.setAttribute(k, "");
    else if (v !== false && v != null) e.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null) e.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return e;
};
const fmtNum = n => (n == null ? "—" : Number(n).toLocaleString());
const STAGES = ["categorize", "synthesize", "dedup", "triage", "assemble", "train", "deploy"];

// Personal-tool app: the token lives in localStorage so the user isn't re-prompted across tabs.
// That's an accepted trade-off here. We do NOT put it in the URL (the SSE exception is handled
// below via a single-use ticket fetched from /v1/sse-ticket).
const State = {
  token: localStorage.getItem("vp_token") || "",
  engineBase: (localStorage.getItem("vp_engine") || "").replace(/\/+$/, ""),  // "" = this machine
  authRequired: null,            // learned from /v1/health; the auth modal only ever shows when this is true
  apiKeySet: null,               // learned from /v1/health (ollama_api_key_set)
  roots: [], llamaDir: "",
  projects: [], current: null, tab: "pipeline",
  jobs: [], currentJobId: null, es: null, jobsPollTimer: null,
};
const apiURL = path => (State.engineBase || "") + path;

/* ---------------- api ---------------- */
async function api(path, opts = {}) {
  const headers = Object.assign({ "Content-Type": "application/json" }, opts.headers || {});
  if (State.token) headers["Authorization"] = "Bearer " + State.token;
  const r = await fetch(apiURL(path), { ...opts, headers });
  if (r.status === 401) {
    if (State.authRequired) { promptAuth(); throw new Error("unauthorized"); }
    throw new Error(`server returned 401 for ${path} but reports auth is not required — try restarting the app`);
  }
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = j.detail || JSON.stringify(j); } catch {}
    throw new Error(msg);
  }
  const ct = r.headers.get("content-type") || "";
  return ct.includes("application/json") ? r.json() : r.text();
}

/* ---------------- toasts ---------------- */
function toast(msg, kind = "") {
  const t = el("div", { class: "toast " + kind }, msg);
  $("#toasts").append(t);
  setTimeout(() => { t.style.opacity = "0"; t.style.transition = "opacity .3s"; setTimeout(() => t.remove(), 300); }, kind === "err" ? 6000 : 3500);
}

/* ---------------- auth ---------------- */
let _authChecking = false;
async function promptAuth() {
  // Re-verify against /v1/health (always auth-exempt) before ever showing the password modal —
  // so a stray 401 can't pop a confusing prompt on a server that doesn't actually require auth.
  if ($("#authModal").hidden === false || _authChecking) return;
  _authChecking = true;
  let required = false;
  try {
    const h = await fetch(apiURL("/v1/health")).then(r => (r.ok ? r.json() : null));
    if (h) { State.authRequired = !!h.auth_required; required = State.authRequired; }
  } catch {}
  _authChecking = false;
  if (!required) {
    toast("A request was rejected, but this server doesn't require a password. A stale voicepipe server may be running — fully quit the app (Cmd-Q) and relaunch (or run `pkill -f voicepipe` first).", "err");
    return;
  }
  $("#authModal").hidden = false; $("#authInput").focus();
}
$("#authSubmit").addEventListener("click", () => {
  const v = $("#authInput").value.trim(); if (!v) return;
  State.token = v; localStorage.setItem("vp_token", v);
  $("#authModal").hidden = true; $("#authError").hidden = true; boot();
});
$("#authInput").addEventListener("keydown", e => { if (e.key === "Enter") $("#authSubmit").click(); });

/* ---------------- health ---------------- */
async function loadHealth() {
  try {
    const h = await api("/v1/health");
    State.authRequired = !!h.auth_required;
    State.apiKeySet = !!h.ollama_api_key_set;
    State.roots = h.roots || [];
    const box = clear($("#health"));
    const dot = (on, label) => el("span", {}, el("span", { class: "dot " + (on ? "on" : "off") }), " " + label);
    box.append(dot(h.has_torch, "torch"), dot(h.has_ollama, "ollama"), dot(h.ollama_api_key_set, "key"));
    if (State.engineBase) box.append(el("span", { title: "connected to a remote engine: " + State.engineBase }, " ⇄ remote"));
    if (h.auth_required) box.append(el("span", {}, " 🔒"));
  } catch (e) { State.apiKeySet = null; /* /v1/health is auth-exempt; if it failed the engine is unreachable */ }
  renderRootsHint();
}
function renderRootsHint() {
  const n = State.projects.length;
  const foot = clear($("#rootsHint"));
  foot.append(el("span", {}, `${n} project${n === 1 ? "" : "s"}`),
              el("span", {}, "  ·  "),
              el("a", { href: "#", title: "settings (engine · API key · llama.cpp · folders)", onclick: e => { e.preventDefault(); openSettings(); } }, "settings"));
}

/* ---------------- projects ---------------- */
async function loadProjects() { State.projects = await api("/v1/projects"); renderProjectList(); renderRootsHint(); }
function renderProjectList() {
  const ul = clear($("#projectList"));
  if (!State.projects.length) { ul.append(el("li", { class: "muted", style: "cursor:default" }, "No projects yet.")); return; }
  for (const p of State.projects) {
    ul.append(el("li", { class: State.current && State.current.id === p.id ? "active" : "", onclick: () => selectProject(p.id) },
      el("div", { class: "pn" }, p.name || p.id),
      p.error ? el("div", { class: "pe" }, "⚠ " + p.error) : el("div", { class: "ps" }, p.description || p.path),
      el("button", { class: "px", title: "remove from list (files are kept)",
                     onclick: ev => { ev.stopPropagation(); removeProjectFromList(p); } }, "✕")));
  }
}
async function removeProjectFromList(p) {
  const label = p.name || p.id;
  if (!window.confirm(`Remove "${label}" from the list?\n\nFiles are NOT deleted — re-add it anytime with "+ New project" → "register an existing project directory".`)) return;
  try {
    await api("/v1/projects/" + encodeURIComponent(p.id), { method: "DELETE" });
    if (State.current && State.current.id === p.id) {
      State.current = null; State.currentJobId = null; closeES();
      $("#projectView").hidden = true; $("#emptyState").hidden = false;
    }
    await loadProjects();
    toast(`Removed "${label}" from the list`, "ok");
  } catch (e) { toast("Couldn't remove: " + e.message, "err"); }
}
async function selectProject(id) {
  localStorage.setItem("vp_last_project", id);
  try { State.current = await api("/v1/projects/" + encodeURIComponent(id)); }
  catch (e) { toast("Couldn't open project: " + e.message, "err"); return; }
  State.currentJobId = null; closeES();
  $("#emptyState").hidden = true; $("#projectView").hidden = false;
  renderProjectList();
  $("#pvName").textContent = State.current.name;
  $("#pvDesc").textContent = State.current.description || "";
  await refreshJobs();
  switchTab(State.tab);
}
function switchTab(tab) {
  State.tab = tab;
  $$(".tab").forEach(b => b.classList.toggle("active", b.dataset.tab === tab));
  for (const t of ["pipeline", "configure", "jobs"]) $("#tab-" + t).hidden = (t !== tab);
  if (tab === "pipeline") renderPipeline();
  if (tab === "configure") renderConfigure();
  if (tab === "jobs") renderJobs();
}
$$(".tab").forEach(b => b.addEventListener("click", () => switchTab(b.dataset.tab)));

/* ---------------- pipeline tab ---------------- */
function stageStatusLine(stage, st) {
  const s = st || {};
  switch (stage) {
    case "categorize": return "propose prompt categories from the corpus (writes a proposal you can adopt)";
    case "synthesize": return s.raw && s.raw.pairs ? `${fmtNum(s.raw.pairs)} pairs in ${fmtNum(s.raw.batches)} batches` : "no dataset yet — generate (user, response) pairs";
    case "dedup":      return s.dedup && s.dedup.pairs ? `${fmtNum(s.dedup.pairs)} pairs after dedup` : "drop near-duplicates + cap over-saturated phrases";
    case "triage":     return s.triage && s.triage.kept ? `${fmtNum(s.triage.kept)} kept / ${fmtNum(s.triage.scored)} scored` : "LLM-judge each pair 1–5; keep the good ones";
    case "assemble":   return s.final && s.final.train ? `${fmtNum(s.final.train)} train / ${fmtNum(s.final.val)} val` : "combine kept synth + seeds + salvage → train/val";
    case "train":      return s.adapter && s.adapter.has_adapter ? "adapter trained ✓ (dataset/adapter/final)" : "QLoRA fine-tune the base model";
    case "deploy":     return s.adapter && s.adapter.has_gguf ? "GGUF built ✓ — Modelfile / ollama create / push" : "LoRA → GGUF → Modelfile → Ollama model";
  }
  return "";
}
function stageIsDone(stage, st) {
  const s = st || {};
  return ({ synthesize: !!(s.raw && s.raw.pairs > 0), dedup: !!(s.dedup && s.dedup.pairs > 0),
            triage: !!(s.triage && s.triage.kept > 0), assemble: !!(s.final && s.final.train > 0),
            train: !!(s.adapter && s.adapter.has_adapter), deploy: !!(s.adapter && s.adapter.has_gguf) })[stage] || false;
}
function renderPipeline() {
  const root = clear($("#tab-pipeline"));
  const st = State.current.dataset_state;
  const running = {}; State.jobs.forEach(j => { if (j.status === "running") running[j.stage] = j.job_id; });
  const flow = el("div", { class: "pipeline-flow" });
  STAGES.forEach((stage, i) => {
    if (i) flow.append(el("div", { class: "stage-connector" }));
    const isRun = !!running[stage], isDone = !isRun && stageIsDone(stage, st);
    flow.append(el("div", { class: "stage-card" + (isRun ? " running" : isDone ? " done" : "") },
      el("div", { class: "stage-num" }, isRun ? "" : isDone ? "✓" : String(i + 1)),
      el("div", { class: "stage-meta" },
        el("div", { class: "sn" }, stage, el("span", { class: "stage-cmd" }, `voicepipe ${stage} --project .`)),
        el("div", { class: "ss" }, stageStatusLine(stage, st))),
      el("div", { class: "stage-actions" }, ...stageActions(stage, isRun, running[stage]))));
  });
  root.append(flow);
}
function stageActions(stage, isRun, jobId) {
  if (isRun) return [el("span", { class: "spinner" }), el("button", { class: "btn btn-sm", onclick: () => { switchTab("jobs"); selectJob(jobId); } }, "View")];
  const acts = [];
  let smokeCb = null;
  if (stage === "train") { smokeCb = el("input", { type: "checkbox" }); acts.push(el("label", { class: "stage-opt" }, smokeCb, " smoke (10 steps)")); }
  acts.push(el("button", { class: "btn btn-sm btn-primary", onclick: () => runStage(stage, { smoke: smokeCb && smokeCb.checked }) }, "Run"));
  return acts;
}
async function runStage(stage, opts = {}) {
  if (["synthesize", "triage", "categorize"].includes(stage) && State.apiKeySet === false) {
    toast(`${stage} needs an Ollama Cloud API key — opening Settings.`, "err");
    openSettings("key"); return;
  }
  if (stage === "deploy" && State.current && !((State.current.config.deploy || {}).llama_cpp_dir) && !State.llamaDir) {
    toast("deploy needs a llama.cpp checkout — set one in this project's Configure tab, or a default in Settings.", "err");
    openSettings("llama"); return;
  }
  const overrides = {};
  if (stage === "deploy" && !((State.current.config.deploy || {}).llama_cpp_dir) && State.llamaDir) overrides.llama_cpp_dir = State.llamaDir;
  try {
    const meta = await api(`/v1/projects/${encodeURIComponent(State.current.id)}/stages/${stage}/run`,
      { method: "POST", body: JSON.stringify({ smoke: !!opts.smoke, overrides }) });
    toast(`Started ${stage} — ${meta.job_id}`, "ok");
    await refreshJobs(); switchTab("jobs"); selectJob(meta.job_id);
  } catch (e) { toast(`Couldn't start ${stage}: ${e.message}`, "err"); }
}

/* ---------------- configure tab ---------------- */
const SCALAR_SECTIONS = [
  ["synthesis", "Synthesis", ["model", "alt_model", "temperature", "top_p", "pairs_per_batch", "concurrency", "think", "target", "balance_modes"]],
  ["dedup", "Dedup", ["cosine_threshold", "embed_model", "embed_base_url", "skip_embed"]],
  ["triage", "Triage", ["model", "batch_size", "min_keep", "concurrency"]],
  ["assemble", "Assemble", ["val_fraction", "seed"]],
  ["train", "Train (QLoRA)", ["base_model", "lora_r", "lora_alpha", "lora_dropout", "max_seq_len", "batch_size", "grad_accum", "epochs", "lr", "optim", "seed"]],
  ["deploy", "Deploy", ["ollama_from", "ollama_tag", "gguf_outtype", "base_model_id_override", "llama_cpp_dir"]],
];
const PROSE_FIELDS = [["synth_preamble", "Synthesis preamble"], ["variety_menus", "Variety menus"], ["content_rules", "Content rules (the hard prohibitions)"]];

function inputForValue(val, onChange) {
  if (typeof val === "boolean") { const i = el("input", { type: "checkbox" }); i.checked = val; i.addEventListener("change", () => onChange(i.checked)); return i; }
  if (typeof val === "number") { const i = el("input", { type: "number", step: "any", value: String(val) }); i.addEventListener("change", () => onChange(i.value === "" ? null : Number(i.value))); return i; }
  const i = el("input", { type: "text", value: val == null ? "" : String(val) }); i.addEventListener("change", () => onChange(i.value === "" ? null : i.value)); return i;
}
function field(label, inputEl) { return el("div", { class: "field" }, el("label", {}, label), inputEl); }
function rawJsonField(label, get, set) {
  const ta = el("textarea", { style: "min-height:50px; font-size:12px" }, JSON.stringify(get()));
  ta.addEventListener("change", () => { try { set(JSON.parse(ta.value)); ta.style.borderColor = ""; } catch { ta.style.borderColor = "var(--bad)"; } });
  return el("div", { class: "field" }, el("label", {}, label, el("span", { class: "hint" }, " (JSON)")), ta);
}
function rowListSection(title, cfg, key, cols, blank) {
  cfg[key] = cfg[key] || [];
  const sec = el("details", { class: "cfg-section" }), summary = el("summary", {}, `${title} (${cfg[key].length})`);
  sec.append(summary);
  const body = el("div", { class: "cfg-body" }), list = el("div", { class: "row-list" });
  const setCount = () => summary.textContent = `${title} (${cfg[key].length})`;
  function redraw() {
    clear(list);
    cfg[key].forEach((item, idx) => {
      const row = el("div", { class: "row-item" });
      for (const [c, kind] of cols) {
        const inp = el("input", { class: kind === "num" ? "w" : "", type: kind === "num" ? "number" : "text", step: "any", value: item[c] == null ? "" : String(item[c]), placeholder: c });
        inp.addEventListener("change", () => { item[c] = kind === "num" ? Number(inp.value) : inp.value; });
        row.append(inp);
      }
      row.append(el("button", { class: "btn btn-sm btn-ghost", onclick: () => { cfg[key].splice(idx, 1); redraw(); setCount(); } }, "✕"));
      list.append(row);
    });
  }
  redraw();
  body.append(list, el("button", { class: "btn btn-sm", style: "margin-top:8px", onclick: () => { cfg[key].push(structuredClone(blank)); redraw(); setCount(); } }, "+ Add"));
  sec.append(body); return sec;
}
function renderConfigure() {
  const root = clear($("#tab-configure"));
  const cfg = structuredClone(State.current.config);

  for (const [key, title, fields] of SCALAR_SECTIONS) {
    const sec = el("details", { class: "cfg-section", open: key === "synthesis" });
    sec.append(el("summary", {}, title));
    const body = el("div", { class: "cfg-body" }), grid = el("div", { class: "cfg-grid" });
    for (const f of fields) {
      cfg[key] = cfg[key] || {};
      grid.append(field(f, inputForValue(cfg[key][f], v => cfg[key][f] = v)));
    }
    body.append(grid);
    if (key === "train") body.append(rawJsonField("target_modules", () => cfg.train.target_modules || [], v => cfg.train.target_modules = v));
    if (key === "deploy") {
      body.append(rawJsonField("stop", () => cfg.deploy.stop || [], v => cfg.deploy.stop = v));
      body.append(rawJsonField("parameters", () => cfg.deploy.parameters || {}, v => cfg.deploy.parameters = v));
    }
    sec.append(body); root.append(sec);
  }

  // modes
  const modesSec = el("details", { class: "cfg-section", open: true }), modesSummary = el("summary", {});
  modesSec.append(modesSummary);
  const modesBody = el("div", { class: "cfg-body" });
  const setModeCount = () => modesSummary.textContent = `Modes (${(cfg.modes || []).length})`;
  function redrawModes() {
    clear(modesBody); setModeCount();
    (cfg.modes || []).forEach((m, idx) => {
      const box = el("details", { class: "cfg-section", open: true, style: "margin:8px 0" });
      box.append(el("summary", {}, m.name || "(unnamed mode)"));
      const b = el("div", { class: "cfg-body" }), row = el("div", { class: "cfg-grid" });
      row.append(field("name", inputForValue(m.name, v => m.name = v)));
      row.append(field("weight", inputForValue(typeof m.weight === "number" ? m.weight : 1.0, v => m.weight = v)));
      row.append(field("corpus_anchor", inputForValue(m.corpus_anchor, v => m.corpus_anchor = v)));
      b.append(row);
      b.append(rawJsonField("styles", () => m.styles || [], v => m.styles = v));
      b.append(rawJsonField("context_files", () => m.context_files || [], v => m.context_files = v));
      const ta = el("textarea", { placeholder: "Per-mode instruction block…" }, m.description || "");
      ta.addEventListener("input", () => m.description = ta.value);
      b.append(el("div", { class: "field" }, el("label", {}, "description"), ta));
      b.append(el("button", { class: "btn btn-sm btn-danger", onclick: () => { cfg.modes.splice(idx, 1); redrawModes(); } }, "Remove mode"));
      box.append(b); modesBody.append(box);
    });
    modesBody.append(el("button", { class: "btn btn-sm", onclick: () => { cfg.modes = cfg.modes || []; cfg.modes.push({ name: "NEW_MODE", weight: 1.0, styles: [], context_files: [], description: "" }); redrawModes(); } }, "+ Add mode"));
  }
  redrawModes(); modesSec.append(modesBody); root.append(modesSec);

  root.append(rowListSection("Prompt categories", cfg, "categories", [["name", "text"], ["weight", "num"]], { name: "new_category", weight: 0.1 }));
  root.append(rowListSection("Length profiles", cfg, "length_profiles", [["name", "text"], ["description", "text"], ["weight", "num"]], { name: "medium", description: "100-250 words per response", weight: 0.5 }));

  // prose blocks
  const proseSec = el("details", { class: "cfg-section" });
  proseSec.append(el("summary", {}, "Prose blocks"));
  const proseBody = el("div", { class: "cfg-body" });
  for (const [f, label] of PROSE_FIELDS) {
    const ta = el("textarea", { style: "min-height:180px" }, cfg[f] || "");
    ta.addEventListener("input", () => cfg[f] = ta.value);
    proseBody.append(el("div", { class: "field" }, el("label", {}, label, el("span", { class: "hint" }, " — written to prompts/" + f + ".md")), ta));
  }
  const rub = el("textarea", { style: "min-height:160px" }, (cfg.triage || {}).rubric || "");
  rub.addEventListener("input", () => { cfg.triage = cfg.triage || {}; cfg.triage.rubric = rub.value; });
  proseBody.append(el("div", { class: "field" }, el("label", {}, "Triage rubric", el("span", { class: "hint" }, " — 1–5 scoring instructions + flag definitions")), rub));
  const sysm = el("textarea", { style: "min-height:160px" }, (cfg.deploy || {}).system_message || "");
  sysm.addEventListener("input", () => { cfg.deploy = cfg.deploy || {}; cfg.deploy.system_message = sysm.value; });
  proseBody.append(el("div", { class: "field" }, el("label", {}, "Deploy system prompt", el("span", { class: "hint" }, " — the deployed model's system message")), sysm));
  proseSec.append(proseBody); root.append(proseSec);

  // identity
  const idSec = el("details", { class: "cfg-section" });
  idSec.append(el("summary", {}, "Identity & paths"));
  const idBody = el("div", { class: "cfg-body" });
  idBody.append(field("name", inputForValue(cfg.name, v => cfg.name = v)));
  idBody.append(field("description", inputForValue(cfg.description, v => cfg.description = v)));
  idBody.append(field("corpus_dir", inputForValue(cfg.corpus_dir, v => cfg.corpus_dir = v)));
  idBody.append(field("seeds_file", inputForValue(cfg.seeds_file, v => cfg.seeds_file = v)));
  idBody.append(field("glossary_file", inputForValue(cfg.glossary_file, v => cfg.glossary_file = v)));
  idSec.append(idBody); root.append(idSec);

  const status = el("span", { class: "muted" });
  root.append(el("div", { class: "cfg-save-bar" },
    el("button", { class: "btn btn-primary", onclick: async (e) => {
      e.target.disabled = true; status.textContent = "saving…";
      try {
        State.current = await api(`/v1/projects/${encodeURIComponent(State.current.id)}/config`, { method: "PUT", body: JSON.stringify({ config: cfg }) });
        toast("Configuration saved", "ok"); status.textContent = "saved.";
        $("#pvName").textContent = State.current.name; $("#pvDesc").textContent = State.current.description || "";
        await loadProjects();
      } catch (err) { toast("Save failed: " + err.message, "err"); status.textContent = ""; }
      e.target.disabled = false;
    } }, "Save configuration"),
    el("button", { class: "btn", onclick: () => switchTab("configure") }, "Revert"),
    status, el("span", { class: "hint", style: "margin-left:auto" }, "Long prose → prompts/*.md ; everything else inline in project.toml")));
}

/* ---------------- jobs tab ---------------- */
async function refreshJobs() {
  if (!State.current) return;
  try { State.jobs = await api("/v1/jobs?project=" + encodeURIComponent(State.current.id)); } catch { State.jobs = []; }
}
function renderJobs() {
  const root = clear($("#tab-jobs"));
  const layout = el("div", { class: "jobs-layout" });
  const listCol = el("div", { class: "job-list" });
  if (!State.jobs.length) listCol.append(el("div", { class: "muted" }, "No runs yet. Start a stage from the Pipeline tab."));
  for (const j of State.jobs) {
    listCol.append(el("div", { class: "job-row" + (j.job_id === State.currentJobId ? " active" : ""), "data-jid": j.job_id, onclick: () => selectJob(j.job_id) },
      el("div", { class: "jr-top" }, el("span", { class: "jr-stage" }, j.stage), el("span", { class: "badge " + j.status }, j.status)),
      el("div", { class: "jr-time" }, (j.started_at || "").replace("T", " ").replace("Z", ""))));
  }
  const detailCol = el("div", { id: "jobDetail", class: "job-detail" }, el("div", { class: "muted" }, "Select a run to watch it."));
  layout.append(listCol, detailCol);
  root.append(layout);
  if (State.currentJobId) selectJob(State.currentJobId);
  clearInterval(State.jobsPollTimer);
  State.jobsPollTimer = setInterval(async () => {
    if (State.tab !== "jobs" || !State.current) return;
    const before = JSON.stringify(State.jobs.map(j => [j.job_id, j.status]));
    await refreshJobs();
    if (JSON.stringify(State.jobs.map(j => [j.job_id, j.status])) !== before) {
      const keep = State.currentJobId; renderJobs(); State.currentJobId = keep;
      $$(".job-row").forEach(r => r.classList.toggle("active", r.getAttribute("data-jid") === keep));
    }
  }, 4000);
}
function closeES() { if (State.es) { try { State.es.close(); } catch {} State.es = null; } }

const JobUI = {};
async function selectJob(jobId) {
  State.currentJobId = jobId;
  $$(".job-row").forEach(r => r.classList.toggle("active", r.getAttribute("data-jid") === jobId));
  closeES();
  let meta;
  try { meta = await api("/v1/jobs/" + encodeURIComponent(jobId)); } catch (e) { toast("Couldn't load job: " + e.message, "err"); return; }
  const root = clear($("#jobDetail")); if (!root) return;
  const isTrain = meta.stage === "train";
  const cancelBtn = meta.status === "running"
    ? el("button", { class: "btn btn-sm btn-danger", onclick: async () => { try { await api("/v1/jobs/" + jobId + "/cancel", { method: "POST" }); toast("Cancel requested"); } catch (e) { toast(e.message, "err"); } } }, "Cancel")
    : null;
  const badge = el("span", { class: "badge " + meta.status }, meta.status);
  root.append(el("div", { class: "jd-head" },
    el("div", {}, el("div", { class: "jd-title" }, meta.stage + "  ·  " + jobId), el("div", { class: "jd-sub" }, (meta.command || []).join(" "))),
    el("div", {}, badge, " ", cancelBtn || "")));
  const progBar = el("div", { class: "progress-bar" });
  const progLabel = el("div", { class: "hint" });
  root.append(el("div", { class: "progress-wrap" }, progBar), progLabel);
  const metricStrip = el("div", { class: "metric-strip" });
  root.append(metricStrip);
  let canvas = null;
  if (isTrain) { canvas = el("canvas", { class: "losschart" }); root.append(canvas); }
  const feed = el("div", { class: "event-feed" });
  root.append(feed);
  const consolePre = el("pre", {}, meta.console_tail || "");
  root.append(el("details", { class: "console-box" }, el("summary", {}, "console output"), consolePre));

  Object.assign(JobUI, { feed, progBar, progLabel, metrics: metricStrip, badge, console: consolePre,
                         metricsData: {}, losses: [], canvas, total: null, jobId });

  (meta.events || []).forEach(handleEvent);
  if (meta.status === "running") {
    const since = (meta.events || []).slice(-1)[0]?.seq || 0;
    // EventSource can't set headers, so we exchange the bearer token for a single-use,
    // short-lived ticket (30s; consumed on first use) and put THAT in the query string.
    // Documented exception — see /v1/sse-ticket in server.py.
    let ticketParam = "";
    if (State.authRequired) {
      try {
        const tk = await api("/v1/sse-ticket");
        if (tk && tk.ticket) ticketParam = "&ticket=" + encodeURIComponent(tk.ticket);
      } catch {}
    }
    const url = apiURL("/v1/jobs/" + encodeURIComponent(jobId) + "/events?since=" + since + ticketParam);
    const es = new EventSource(url); State.es = es;
    es.onmessage = ev => { try { handleEvent(JSON.parse(ev.data)); } catch {} };
    es.addEventListener("end", async () => {
      closeES(); await refreshJobs();
      const b = State.jobs.find(j => j.job_id === jobId);
      if (b) { JobUI.badge.textContent = b.status; JobUI.badge.className = "badge " + b.status; }
      refreshConsole(jobId);
    });
  }
}
async function refreshConsole(jobId) { try { JobUI.console.textContent = await api("/v1/jobs/" + jobId + "/log?tail=400"); } catch {} }

function handleEvent(e) {
  if (!JobUI.feed || JobUI.jobId == null) return;
  const cls = ["ev", "type-" + e.type];
  if (e.type === "log") cls.push("lv-" + (e.level || "info"));
  if (e.type === "stage_end") cls.push("st-" + (e.status || "ok"));
  JobUI.feed.append(el("div", { class: cls.join(" ") }, el("span", { class: "t" }, (e.ts || "").slice(11, 23) + "  "), eventText(e)));
  JobUI.feed.scrollTop = JobUI.feed.scrollHeight;
  if (e.type === "progress") {
    if (e.total) { JobUI.total = e.total; JobUI.progBar.style.width = Math.min(100, 100 * (e.current || 0) / e.total) + "%"; }
    JobUI.progLabel.textContent = `${fmtNum(e.current)}${e.total ? " / " + fmtNum(e.total) : ""} ${e.unit || ""}${e.detail ? "  ·  " + e.detail : ""}`;
  }
  if (e.type === "phase") JobUI.progLabel.textContent = "phase: " + e.name;
  if (e.type === "metric") {
    for (const [k, v] of Object.entries(e)) if (!["ts", "stage", "type", "seq"].includes(k)) JobUI.metricsData[k] = v;
    renderMetricStrip();
    if (typeof e.loss === "number") { JobUI.losses.push({ step: Number(e.step) || JobUI.losses.length, loss: e.loss }); drawLossChart(); }
  }
  if (e.type === "stage_end") {
    JobUI.badge.textContent = e.status; JobUI.badge.className = "badge " + e.status;
    if (e.summary) { Object.assign(JobUI.metricsData, flattenSummary(e.summary)); renderMetricStrip(); }
  }
}
function flattenSummary(s, pfx = "") { const o = {}; for (const [k, v] of Object.entries(s || {})) { if (v && typeof v === "object" && !Array.isArray(v)) Object.assign(o, flattenSummary(v, pfx + k + ".")); else o[pfx + k] = v; } return o; }
function renderMetricStrip() {
  const box = clear(JobUI.metrics);
  const order = ["step", "epoch", "loss", "eval_loss", "learning_rate", "grad_norm"];
  const keys = [...new Set([...order.filter(k => k in JobUI.metricsData), ...Object.keys(JobUI.metricsData)])];
  for (const k of keys) {
    let v = JobUI.metricsData[k];
    if (typeof v === "number") v = (v !== 0 && Math.abs(v) < 1e-3) ? v.toExponential(2) : (Number.isInteger(v) ? v : Number(v).toFixed(4));
    box.append(el("span", { class: "m" }, k + " ", el("b", {}, String(v))));
  }
}
function eventText(e) {
  switch (e.type) {
    case "stage_start": return "▶ " + (e.params ? compact(e.params) : "started");
    case "phase": return "‹phase› " + e.name;
    case "progress": return "… " + fmtNum(e.current) + (e.total ? "/" + fmtNum(e.total) : "") + " " + (e.unit || "") + (e.detail ? "  " + e.detail : "");
    case "metric": return "metric " + compact(Object.fromEntries(Object.entries(e).filter(([k]) => !["ts", "stage", "type", "seq"].includes(k))));
    case "artifact": return "📦 " + (e.kind ? e.kind + ": " : "") + e.path + (e.bytes ? "  (" + humanBytes(e.bytes) + ")" : "");
    case "log": return (e.level === "error" ? "✖ " : e.level === "warn" ? "⚠ " : "· ") + e.message;
    case "stage_end": return "■ " + (e.status || "ok") + (e.duration_sec != null ? "  in " + Math.round(e.duration_sec) + "s" : "") + (e.summary ? "  " + compact(e.summary) : "") + (e.error ? "\n" + e.error : "");
    default: return JSON.stringify(e);
  }
}
function compact(o) { try { return JSON.stringify(o).replace(/^[{\[]|[}\]]$/g, "").replace(/"/g, "").slice(0, 400); } catch { return String(o); } }
function humanBytes(n) { const u = ["B", "KB", "MB", "GB"]; let i = 0; while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; } return n.toFixed(i ? 1 : 0) + " " + u[i]; }
function drawLossChart() {
  const c = JobUI.canvas; if (!c) return;
  const dpr = window.devicePixelRatio || 1, W = c.clientWidth || 600, H = c.clientHeight || 140;
  c.width = W * dpr; c.height = H * dpr;
  const ctx = c.getContext("2d"); ctx.scale(dpr, dpr); ctx.clearRect(0, 0, W, H);
  const data = JobUI.losses; if (data.length < 2) return;
  const pad = { l: 42, r: 10, t: 10, b: 18 };
  const xs = data.map(d => d.step), ys = data.map(d => d.loss);
  const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
  const X = x => pad.l + (xmax === xmin ? 0 : (x - xmin) / (xmax - xmin)) * (W - pad.l - pad.r);
  const Y = y => pad.t + (ymax === ymin ? 0.5 : 1 - (y - ymin) / (ymax - ymin)) * (H - pad.t - pad.b);
  ctx.strokeStyle = "#2b313b"; ctx.fillStyle = "#6b7480"; ctx.font = "10px monospace"; ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) { const y = pad.t + i / 3 * (H - pad.t - pad.b); ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke(); ctx.fillText((ymax - i / 3 * (ymax - ymin)).toFixed(2), 4, y + 3); }
  const grad = ctx.createLinearGradient(0, 0, W, 0); grad.addColorStop(0, "#5b86f5"); grad.addColorStop(1, "#7aa2ff");
  ctx.strokeStyle = grad; ctx.lineWidth = 2; ctx.beginPath();
  data.forEach((d, i) => { const px = X(d.step), py = Y(d.loss); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }); ctx.stroke();
  const last = data[data.length - 1]; ctx.fillStyle = "#7aa2ff"; ctx.beginPath(); ctx.arc(X(last.step), Y(last.loss), 3, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#9aa3b0"; ctx.textAlign = "right"; ctx.fillText("step " + last.step + "  loss " + last.loss.toFixed(3), W - pad.r, H - 4); ctx.textAlign = "left";
}

/* ---------------- new project ---------------- */
async function openNewProject() {
  $("#npError").hidden = true; ["npName", "npDesc", "npParent", "npExisting"].forEach(id => $("#" + id).value = "");
  const sel = clear($("#npTemplate"));
  let tmpls = []; try { tmpls = await api("/v1/templates"); } catch {}
  tmpls.forEach(t => sel.append(el("option", { value: t.id }, t.id)));
  const setDesc = () => { const t = tmpls.find(x => x.id === sel.value); $("#npTemplateDesc").textContent = t ? t.description : ""; };
  sel.onchange = setDesc; setDesc();
  $("#newProjectModal").hidden = false; $("#npName").focus();
}
$("#newProjectBtn").addEventListener("click", openNewProject);
$("#newProjectLink").addEventListener("click", e => { e.preventDefault(); openNewProject(); });
$("#npCancel").addEventListener("click", () => $("#newProjectModal").hidden = true);
$("#npCreate").addEventListener("click", async () => {
  const existing = $("#npExisting").value.trim();
  let body;
  if (existing) body = { path: existing };
  else {
    const name = $("#npName").value.trim();
    if (!name) { $("#npError").textContent = "Name is required."; $("#npError").hidden = false; return; }
    body = { name, description: $("#npDesc").value.trim(), template: $("#npTemplate").value };
    if ($("#npParent").value.trim()) body.parent_dir = $("#npParent").value.trim();
  }
  try {
    const detail = await api("/v1/projects", { method: "POST", body: JSON.stringify(body) });
    $("#newProjectModal").hidden = true;
    await loadProjects(); await selectProject(detail.id);
    toast("Project created: " + detail.name, "ok");
  } catch (e) { $("#npError").textContent = e.message; $("#npError").hidden = false; }
});

/* ---------------- settings modal ---------------- */
let _folderRows = [];
function renderFolderRows() {
  const list = clear($("#foldersList"));
  _folderRows.forEach((val, idx) => {
    const inp = el("input", { type: "text", value: val, placeholder: "/path/to/folder", spellcheck: "false" });
    inp.addEventListener("input", () => _folderRows[idx] = inp.value);
    list.append(el("div", { class: "row-item" }, inp,
      el("button", { class: "btn btn-sm btn-ghost", title: "remove", onclick: () => { _folderRows.splice(idx, 1); renderFolderRows(); $("#setFoldersCount").textContent = `(${_folderRows.length})`; } }, "✕")));
  });
}
async function openSettings(focus) {
  $("#setError").hidden = true;
  let cfg = { project_roots: State.roots, roots_are_default: true, ollama_api_key_set: State.apiKeySet, llama_cpp_dir: State.llamaDir, config_file: "", env_file: "" };
  try { cfg = await api("/v1/config"); } catch {}
  State.llamaDir = cfg.llama_cpp_dir || "";
  $("#setApiKey").value = "";
  $("#setApiKeyState").textContent = cfg.ollama_api_key_set ? "✓ a key is set (in " + (cfg.env_file || "~/.config/voicepipe/env") + ")" : "✗ not set — synthesize / triage / categorize need one";
  $("#setLlamaDir").value = cfg.llama_cpp_dir || "";
  $("#setEngineUrl").value = State.engineBase || "";
  $("#setEngineToken").value = State.engineBase ? State.token : "";
  _folderRows = (cfg.project_roots || []).slice(); renderFolderRows();
  $("#setFoldersCount").textContent = `(${_folderRows.length})`;
  $("#foldersHint").textContent = (cfg.roots_are_default ? "Using built-in defaults. " : "") + (cfg.config_file ? "Saved to " + cfg.config_file : "");
  $("#settingsModal").hidden = false;
  if (focus === "key") $("#setApiKey").focus();
  else if (focus === "llama") $("#setLlamaDir").focus();
}
$("#foldersAdd").addEventListener("click", () => { _folderRows.push(""); renderFolderRows(); $("#setFoldersCount").textContent = `(${_folderRows.length})`; $("#foldersList").lastElementChild?.querySelector("input")?.focus(); });
$("#setCancel").addEventListener("click", () => $("#settingsModal").hidden = true);
$("#setSave").addEventListener("click", async () => {
  $("#setError").hidden = true;
  // 1. engine connection (client-side only)
  const newEngine = $("#setEngineUrl").value.trim().replace(/\/+$/, "");
  const newToken = $("#setEngineToken").value.trim();
  const engineChanged = (newEngine !== State.engineBase) || (newEngine && newToken !== State.token);
  if (newEngine) { State.engineBase = newEngine; State.token = newToken; }
  else { State.engineBase = ""; /* keep token for the local 'auth on' case */ }
  localStorage.setItem("vp_engine", State.engineBase);
  if (State.engineBase || State.token) localStorage.setItem("vp_token", State.token);
  // 2. server-side settings (sent to whichever engine we're now pointed at)
  const body = { project_roots: _folderRows.map(s => s.trim()).filter(Boolean), llama_cpp_dir: $("#setLlamaDir").value.trim() };
  const apiKey = $("#setApiKey").value.trim();
  if (apiKey) body.ollama_api_key = apiKey;
  try {
    const cfg = await api("/v1/config", { method: "PUT", body: JSON.stringify(body) });
    State.roots = cfg.project_roots || body.project_roots;
    State.llamaDir = cfg.llama_cpp_dir || "";
    $("#settingsModal").hidden = true;
    toast(engineChanged ? "Connected — reloading…" : "Settings saved", "ok");
    if (engineChanged) { State.current = null; State.currentJobId = null; closeES(); $("#projectView").hidden = true; $("#emptyState").hidden = false; boot(); return; }
    await loadHealth(); await loadProjects();
    if (State.current && !State.projects.some(p => p.id === State.current.id)) {
      State.current = null; $("#projectView").hidden = true; $("#emptyState").hidden = false; renderProjectList();
    }
  } catch (e) { $("#setError").textContent = e.message; $("#setError").hidden = false; }
});

/* surface unexpected JS errors as toasts (the release build has no devtools) */
window.addEventListener("error", e => { try { toast(`error: ${e.message || e.error || e}`, "err"); } catch {} });
window.addEventListener("unhandledrejection", e => { try { toast(`error: ${(e.reason && (e.reason.message || e.reason.stack)) || e.reason || e}`, "err"); } catch {} });

/* ---------------- frameless window controls (Windows/Linux Tauri app — macOS uses overlay traffic lights; a browser gets nothing) ---------------- */
(function setupWindowControls() {
  const T = window.__TAURI__;
  if (!T || !T.window || document.documentElement.classList.contains("is-mac")) return;
  let win = null;
  try { win = (T.window.getCurrentWindow && T.window.getCurrentWindow()) || T.window.appWindow || null; } catch {}
  if (!win) return;
  const mk = (label, title, cls, fn) => el("button", { class: cls || "", title, onclick: () => { try { fn(); } catch {} } }, label);
  $("#winControls").append(
    mk("–", "Minimize", "", () => win.minimize()),
    mk("□", "Maximize / Restore", "", () => (win.toggleMaximize ? win.toggleMaximize() : win.maximize())),
    mk("✕", "Close", "wc-close", () => win.close()),
  );
})();

/* ---------------- boot ---------------- */
let _firstRunHintShown = false;
async function boot() {
  await loadHealth();
  try { await loadProjects(); }
  catch (e) { if (e.message !== "unauthorized") toast("Couldn't reach the engine: " + e.message, "err"); return; }
  if (State.apiKeySet === false && !_firstRunHintShown) {
    _firstRunHintShown = true;
    toast("No Ollama API key set — synthesize / triage / categorize need one. Open Settings (bottom-left) to add it.", "err");
  }
  const last = localStorage.getItem("vp_last_project");
  if (last && State.projects.some(p => p.id === last)) selectProject(last);
  else if (State.projects.length === 1) selectProject(State.projects[0].id);
}
window.addEventListener("resize", () => { if (JobUI.canvas) drawLossChart(); });
boot();
