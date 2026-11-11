/* TruthLens UI. No framework, no build step. Text is always set with textContent / text nodes, never innerHTML. */
(() => {
  "use strict";
  const E = window.TLEngine, DATA = window.TL_DATA;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const reduceMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else if (k === "style") for (const [p, x] of Object.entries(v)) el.style.setProperty(p, x);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat(3)) {
      if (kid == null || kid === false) continue;
      el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return el;
  }
  const ICONS = {
    check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>', flag: '<path d="M6 21V4M6 4h11l-2.5 4L17 12H6"/>', x: '<path d="M6 6l12 12M18 6L6 18"/>',
    dash: '<path d="M7 12h10"/>', sliders: '<path d="M4 7h9M17 7h3M4 17h3M11 17h9"/><circle cx="15" cy="7" r="2"/><circle cx="9" cy="17" r="2"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6l1.4 1.4M17 17l1.4 1.4M18.4 5.6L17 7M7 17l-1.4 1.4"/>',
    moon: '<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/>', copy: '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/>',
    trash: '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>', plus: '<path d="M12 5v14M5 12h14"/>',
    chat: '<path d="M5 5h14a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H10l-4 3.5V16H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1z"/><path d="M8.5 9.5h7M8.5 12.5h4"/>',
    filecheck: '<path d="M6 3h8l4 4v13a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z"/><path d="M14 3v4h4M9 14.5l2.2 2.2L15.5 12"/>',
    folder: '<path d="M3.5 7a1.5 1.5 0 0 1 1.5-1.5h4l2 2.5h8A1.5 1.5 0 0 1 20.5 9.5v8A1.5 1.5 0 0 1 19 19H5a1.5 1.5 0 0 1-1.5-1.5z"/>',
    pulse: '<path d="M3 12h4l2.5-6 4 12 2.5-6H21"/>', clock: '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>',
    info: '<circle cx="12" cy="12" r="8.5"/><path d="M12 11v5M12 8v.01"/>', menu: '<path d="M4 7h16M4 12h16M4 17h16"/>',
    search: '<circle cx="11" cy="11" r="6.5"/><path d="M16 16l4.5 4.5"/>', help: '<rect x="3" y="6" width="18" height="12" rx="2.5"/><path d="M7 10h.01M10.5 10h.01M14 10h.01M17 10h.01M8 14h8"/>',
    panel: '<rect x="3.5" y="4.5" width="17" height="15" rx="2.5"/><path d="M9.5 4.5v15"/>',
    upload: '<path d="M12 16V4M7.5 8.5L12 4l4.5 4.5M5 20h14"/>', open: '<path d="M5 12h14M13 6l6 6-6 6"/>', share: '<path d="M12 15V4M7.5 8.5L12 4l4.5 4.5M5 14v6h14v-6"/>',
  };
  const ico = (n) => { const s = h("span", { class: "ico", "aria-hidden": "true" }); s.innerHTML = `<svg viewBox="0 0 24 24" focusable="false">${ICONS[n]}</svg>`; return s; };

  const STATUS = {
    supported: { title: "Backed by the sources", short: "Backed", icon: "check" },
    unsupported: { title: "Not in the sources", short: "Not in sources", icon: "flag" },
    contradicted: { title: "Contradicted by the sources", short: "Contradicted", icon: "x" },
    none: { title: "Not a factual claim", short: "Not a claim", icon: "dash" },
  };
  const SCORE_LABEL = { lexical: "Word match", embedding: "Meaning match", nli: "Entailment", llm_judge: "Judge confidence" };
  const VERIFIER_NAME = { lexical: "Lexical match", embedding: "Embedding similarity", nli: "NLI model", llm_judge: "LLM judge" };

  // ------------------------------------------------------------------ storage
  const store = {
    get(k, d) { try { const v = localStorage.getItem("truthlens." + k); return v == null ? d : JSON.parse(v); } catch (_) { return d; } },
    set(k, v) { try { localStorage.setItem("truthlens." + k, JSON.stringify(v)); return true; } catch (_) { return false; } },
  };
  function toast(msg, undo) {
    const t = $("#toast"); t.replaceChildren(h("span", {}, msg)); t.hidden = false;
    if (undo) t.append(h("button", { type: "button", onclick: () => { t.hidden = true; undo(); } }, "Undo"));
    clearTimeout(toast.t); toast.t = setTimeout(() => { t.hidden = true; }, undo ? 7000 : 2200);
  }
  async function copyText(text, ok = "Copied") {
    try { await navigator.clipboard.writeText(text); toast(ok); return; } catch (_) { /* fall through */ }
    const ta = h("textarea", { "aria-hidden": "true" }); ta.value = text; ta.style.setProperty("position", "fixed"); ta.style.setProperty("opacity", "0");
    document.body.append(ta); ta.select();
    let done = false; try { done = document.execCommand("copy"); } catch (_) { /* ignore */ }
    ta.remove(); toast(done ? ok : "Copy isn't available here. Select the text and copy it yourself.");
  }
  const embedded = (() => { try { return window.self !== window.top; } catch (_) { return true; } })();

  // ------------------------------------------------------------------ state
  const DEFAULTS = { engine: "browser", threshold: 0.7, k: 5, llm: "auto", verifiers: ["lexical"], act: "lexical", action: "remove", style: "strict", claim_mode: "rule", density: "comfortable" };
  const S = {
    settings: { ...DEFAULTS, ...store.get("settings", {}) },
    docs: null, chunks: [], retriever: null,
    meta: null, server: false,
    history: store.get("history", []),
  };
  S.docs = store.get("docs", null) || DATA.docs.map((d) => ({ ...d, sample: true }));
  function rebuildIndex() {
    S.chunks = E.loadCorpus(S.docs);
    S.retriever = new E.Retriever(S.chunks);
  }
  rebuildIndex();
  const saveSettings = () => store.set("settings", S.settings);
  const useServer = () => S.server && S.settings.engine === "server";

  async function api(path, body) {
    const ctl = new AbortController(); const to = setTimeout(() => ctl.abort(), 120000);
    try {
      const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: ctl.signal });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.error || `The server answered with status ${r.status}.`);
      return data;
    } catch (e) {
      if (e.name === "AbortError") throw new Error("The server took too long to answer. Try again, or use fewer verifiers.");
      throw e;
    } finally { clearTimeout(to); }
  }
  const serverSettings = () => {
    const s = S.settings;
    return { llm: s.llm, verifiers: s.verifiers, act: s.act, action: s.action, style: s.style, claim_mode: s.claim_mode, k: s.k };
  };
  async function runAsk(question) {
    if (useServer()) {
      const rec = await api("/api/ask", { question, settings: serverSettings() });
      rec.ui = { ...rec.ui, engine: "server", mode: "ask", at: Date.now() };
      return rec;
    }
    await sleep(0);
    return E.ask(question, S.retriever, { k: S.settings.k, threshold: S.settings.threshold });
  }
  async function runAudit(text) {
    if (useServer()) {
      const rec = await api("/api/verify", { answer: text, settings: serverSettings() });
      rec.ui = { ...rec.ui, engine: "server", mode: "audit", at: Date.now() };
      return rec;
    }
    await sleep(0);
    return E.audit(text, S.retriever, { threshold: S.settings.threshold });
  }

  // ------------------------------------------------------------------ record helpers
  const normText = (t) => t.replace(/\s+/g, " ").trim().toLowerCase();
  function sentenceStatus(claims, vmap) {
    if (!claims.length) return "none";
    const labels = claims.map((c) => (vmap[c.id] ? vmap[c.id].label : "unsupported"));
    if (labels.includes("contradicted")) return "contradicted";
    return labels.some((l) => l !== "supported") ? "unsupported" : "supported";
  }
  function buildItems(snap, act, source) {
    const by = {};
    for (const [n, list] of Object.entries(snap.verdicts)) by[n] = Object.fromEntries(list.map((v) => [v.claim_id, v]));
    const bySent = {};
    for (const c of snap.claims) (bySent[c.sentence_idx] = bySent[c.sentence_idx] || []).push(c);
    return snap.sentences.map((s, pos) => {
      const claims = bySent[s.idx] || [];
      return { key: `${source}:${s.idx}`, source, pos, total: snap.sentences.length, sentence: s, claims, by, status: sentenceStatus(claims, by[act] || {}) };
    });
  }
  function summarize(rec) {
    const act = rec.ui.act;
    const items = buildItems(rec.baseline, act, "base");
    const vs = items.flatMap((it) => it.claims.map((c) => (it.by[act] || {})[c.id] || { label: "unsupported" }));
    const n = vs.length;
    const c = { supported: 0, unsupported: 0, contradicted: 0 };
    vs.forEach((v) => { c[v.label] = (c[v.label] || 0) + 1; });
    return { items, n, counts: c, flagged: n - c.supported, vs };
  }
  function locate(passage, win) {
    const needle = (win || "").replace(/\s+/g, " ").trim().toLowerCase();
    if (needle.length < 8) return null;
    let norm = ""; const map = []; let space = false;
    for (let i = 0; i < passage.length; i++) {
      if (/\s/.test(passage[i])) { if (!space && norm) { norm += " "; map.push(i); } space = true; }
      else { norm += passage[i].toLowerCase(); map.push(i); space = false; }
    }
    const at = norm.indexOf(needle);
    if (at < 0) return null;
    return [map[at], map[at + needle.length - 1] + 1];
  }
  const windowOf = (v) => v.window || ((/window:\s*([\s\S]+)$/.exec(v.rationale || "") || [])[1] || "");

  /** The claim's words, with those the evidence does not contain marked. */
  function claimWithGaps(text, win) {
    const have = new Set(E.tokenize(win));
    const out = [];
    for (const part of text.split(/([A-Za-z0-9][A-Za-z0-9.,'’-]*)/)) {
      if (!part) continue;
      const toks = /^[A-Za-z0-9]/.test(part) ? E.tokenize(part) : [];
      if (toks.length && toks.some((t) => !have.has(t))) out.push(h("span", { class: "gap" }, part));
      else out.push(part);
    }
    return out;
  }

  // ------------------------------------------------------------------ report export
  function markdownReport(rec) {
    const sum = summarize(rec); const act = rec.ui.act;
    const lines = ["# TruthLens report", ""];
    if (rec.question) lines.push(`**Question:** ${rec.question}`, "");
    lines.push(`**Result:** ${sum.counts.supported} of ${sum.n} claims backed by the sources (${(rec.ui.thresholds || {})[act] ?? "n/a"} strictness, ${VERIFIER_NAME[act] || act}).`, "");
    lines.push("| Sentence | Verdict | Score | Evidence |", "|---|---|---|---|");
    for (const it of sum.items) {
      for (const c of it.claims) {
        const v = (it.by[act] || {})[c.id]; if (!v) continue;
        lines.push(`| ${c.text.replace(/\|/g, "\\|")} | ${STATUS[v.label].short} | ${v.support_score.toFixed(2)} | ${v.evidence_id || "none"} |`);
      }
    }
    lines.push("", "_Checked with TruthLens._");
    return lines.join("\n");
  }

  // ------------------------------------------------------------------ Result component
  class Result {
    constructor(mount, kind) { this.mount = mount; this.kind = kind; this.rec = null; this.view = "generated"; this.sel = null; this.root = null; this.filter = "all"; }
    clear() { this.rec = null; this.mount.replaceChildren(); }
    fail(message) {
      this.rec = null;
      this.mount.replaceChildren(h("section", { class: "failure", role: "alert" }, h("h2", {}, "Couldn't check this"), h("p", {}, message)));
    }
    async show(rec, { animate = true, scroll = false } = {}) {
      this.rec = rec; this.view = "generated"; this.sel = null; this.filter = "all";
      this.build();
      this.refresh(animate);
      if (scroll) this.mount.scrollIntoView({ behavior: reduceMotion() ? "auto" : "smooth", block: "start" });
      if (animate) await this.playPipeline();
    }
    /** Fixed skeleton, so controls that live in it (the strictness slider) are never replaced while in use. */
    build() {
      this.pipeEl = h("ol", { class: "pipeline", "aria-label": "Steps this check went through" });
      this.headEl = h("div", { class: "verdict-head" });
      this.docEl = h("div", { class: "doc", id: `${this.kind}-doc` });
      this.ev = h("div", { class: "ev" });
      this.rail = h("aside", { class: "rail", "aria-label": "Evidence", id: `${this.kind}-rail` }, this.ev);
      if (this.rec.ui.engine === "browser") this.rail.append(this.strictness());
      this.ledgerEl = h("div"); this.sourcesEl = h("div");
      this.mount.replaceChildren(h("section", { "aria-labelledby": `${this.kind}-ttl` },
        h("h2", { class: "sr", id: `${this.kind}-ttl` }, "Result"), this.pipeEl, this.headEl,
        h("div", { class: "workspace" }, this.docEl, this.rail), this.ledgerEl, this.sourcesEl));
    }
    stages() {
      const rec = this.rec, sum = summarize(rec), t = rec.ui.timings || {}, mode = rec.ui.mode;
      const fmt = (ms) => (ms == null ? "" : ms < 1 ? "<1 ms" : `${Math.round(ms)} ms`);
      const list = [];
      if (mode === "ask") {
        list.push(["Retrieve", `${rec.retrieved.length} passages`, fmt(t.retrieve)]);
        list.push(["Write answer", `${rec.baseline.sentences.length} ${rec.baseline.sentences.length === 1 ? "sentence" : "sentences"}`, fmt(t.generate)]);
      }
      list.push(["Split into claims", `${sum.n} ${sum.n === 1 ? "claim" : "claims"}`, fmt(t.split)]);
      list.push(["Verify", `${sum.flagged} flagged`, fmt(t.verify)]);
      const removed = rec.baseline.sentences.length - rec.final.sentences.length;
      list.push(["Fix answer", rec.ui.action === "flag" ? "flag only" : `${removed} removed`, rec.ui.timings ? "" : fmt(rec.ui.elapsed_ms)]);
      return list;
    }
    async playPipeline() {
      const els = $$(".stage", this.mount);
      const marks = $$(".row-btn .mark[data-final]", this.mount);
      const reveal = (m) => {
        m.dataset.s = m.dataset.final; const l = m.closest(".row-btn")?.querySelector(".line"); if (l) l.dataset.s = m.dataset.final;
        $$(`.seg[data-sent="${m.dataset.sent}"]`, this.mount).forEach((g) => { g.dataset.s = g.dataset.final; });
      };
      if (reduceMotion()) { els.forEach((e) => (e.dataset.on = "1")); marks.forEach(reveal); return; }
      for (let i = 0; i < els.length; i++) { await sleep(130); if (!els[i].isConnected) return; els[i].dataset.on = "1"; }
      for (const m of marks) { await sleep(170); if (!m.isConnected) return; reveal(m); }
    }

    draw(pending = false) { this.refresh(pending); }
    refresh(pending = false) {
      const rec = this.rec; if (!rec) return;
      const sum = summarize(rec), act = rec.ui.act;
      this.sum = sum;
      const flagged = sum.flagged, n = sum.n;
      const title = n === 0 ? "Nothing to check here" : flagged === 0 ? (n === 1 ? "The claim is backed by the sources" : `All ${n} claims are backed by the sources`)
        : `${flagged} of ${n} ${n === 1 ? "claim isn't" : "claims aren't"} backed by the sources`;

      const view = this.view;
      const baseItems = sum.items;
      const finalItems = buildItems(rec.final, act, "final");
      const keepIdx = new Set(rec.final.sentences.map((s) => s.idx));
      const removedItems = baseItems.filter((it) => !keepIdx.has(it.sentence.idx) && rec.ui.action !== "flag").map((it) => ({ ...it, key: `removed:${it.sentence.idx}`, source: "removed" }));
      const main = view === "verified" ? finalItems : baseItems;
      const navList = view === "verified" ? [...finalItems, ...removedItems] : baseItems;
      const all = view === "compare" ? [...baseItems, ...finalItems] : navList;
      this.navList = navList;
      if (!this.sel || !all.some((i) => i.key === this.sel)) {
        const firstFlag = all.find((i) => i.status === "unsupported" || i.status === "contradicted");
        this.sel = (firstFlag || all.find((i) => i.status !== "none") || all[0] || {}).key || null;
      }
      this.all = all;

      // header
      const strip = h("div", { class: "strip", role: "list", "aria-label": "Verdict for each claim" });
      sum.items.forEach((it) => it.claims.forEach((c) => {
        const v = (it.by[act] || {})[c.id]; const label = v ? v.label : "unsupported";
        strip.append(h("button", { class: "seg", type: "button", role: "listitem", "data-s": pending ? "pending" : label, "data-final": label, "data-sent": it.sentence.idx, "data-key": it.key,
          title: c.text, "aria-label": `Claim: ${STATUS[label].short}. ${c.text}`, "aria-current": it.key === this.sel ? "true" : null,
          onclick: () => { if (this.view !== "generated") { this.view = "generated"; this.sel = it.key; this.refresh(); } else this.pick(it.key, true); } }));
      }));
      const legend = h("ul", { class: "legend" },
        ["supported", "unsupported", "contradicted"].filter((k) => sum.counts[k] || k === "supported").map((k) =>
          h("li", {}, h("span", { class: "pill", "data-s": k }, ico(STATUS[k].icon), STATUS[k].short), h("b", {}, sum.counts[k] || 0))));
      const tabs = h("div", { class: "seg-ctl", role: "tablist", "aria-label": "Answer version" },
        ["generated", "verified", "compare"].map((k) => h("button", { type: "button", role: "tab", "aria-selected": String(view === k), onclick: () => { this.view = k; this.sel = null; this.refresh(); } },
          { generated: "As written", verified: "After fixing", compare: "Side by side" }[k])));
      this.headEl.replaceChildren(
        h("div", {}, h("h2", { class: `verdict-title ${n && !flagged ? "good" : "bad"}` }, title), strip, legend),
        h("div", { class: "toolbar" }, tabs, this.exportMenu()));

      // pipeline
      this.pipeEl.replaceChildren(...this.stages().map(([a, b, c]) => h("li", { class: "stage", "data-on": pending ? "0" : "1" }, h("b", {}, a), h("span", {}, b), c ? h("i", {}, c) : null)));

      // document
      const doc = this.docEl; doc.replaceChildren();
      if (rec.question) doc.append(h("p", { class: "doc-q" }, "You asked", h("b", {}, rec.question)));
      if (view === "compare") {
        doc.append(h("div", { class: "compare" },
          h("div", {}, h("h3", {}, "As written"), h("ol", { class: "sentences" }, baseItems.map((it) => this.row(it, false)))),
          h("div", {}, h("h3", {}, "After fixing"), finalItems.length ? h("ol", { class: "sentences" }, finalItems.map((it) => this.row(it, false))) : h("p", { class: "abstain" }, "Nothing is left once flagged sentences are removed."))));
      } else if (!main.length) doc.append(h("p", { class: "abstain" }, "The sources do not contain enough information to answer this question."));
      else doc.append(h("ol", { class: "sentences" }, main.map((it) => this.row(it, pending && view === "generated"))));
      if (view === "verified" && removedItems.length) {
        doc.append(h("div", { class: "removed" }, h("h3", {}, rec.rounds > 0 ? "Replaced or removed" : "Removed from the answer"),
          h("ol", { class: "sentences" }, removedItems.map((it) => this.row(it, false, true)))));
      }
      doc.append(h("div", { class: "doc-foot" },
        h("span", {}, "Select a sentence to see its evidence. ", h("kbd", {}, "↑"), " ", h("kbd", {}, "↓"), " moves between sentences."),
        h("button", { class: "btn quiet small", type: "button", onclick: () => this.copyAnswer() }, ico("copy"), view === "verified" ? "Copy fixed answer" : "Copy answer")));

      this.drawRail();
      this.ledgerEl.replaceChildren(this.ledger(sum, act));
      this.sourcesEl.replaceChildren(this.sources());
      this.markSources();
    }

    row(item, pending, removed = false) {
      const label = STATUS[item.status].short;
      const s = item.status;
      const cites = (item.sentence.cited_ids || []).map((id) => h("span", { class: "cite" }, id));
      const mark = h("span", { class: "mark", "aria-hidden": "true", "data-s": pending ? "pending" : s, "data-final": s, "data-sent": item.sentence.idx }, ico(STATUS[s].icon));
      const line = h("span", { class: "line", "data-s": removed ? "removed" : pending ? "none" : s }, item.sentence.text, cites);
      return h("li", {}, h("button", { class: "row-btn", type: "button", "data-key": item.key, "aria-pressed": String(item.key === this.sel),
        onclick: () => this.pick(item.key, true) }, mark, h("span", {}, h("span", { class: "sr" }, `${label}. `), line)));
    }

    pick(key, user) {
      this.sel = key;
      if (this.view === "generated" && key.startsWith("removed")) this.view = "verified";
      $$(".row-btn", this.mount).forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.key === key)));
      $$(".seg", this.mount).forEach((b) => { if (b.dataset.key === key) b.setAttribute("aria-current", "true"); else b.removeAttribute("aria-current"); });
      $$(".ledger tbody tr", this.mount).forEach((r) => r.setAttribute("aria-selected", String(r.dataset.key === key)));
      this.drawRail();
      this.markSources();
      if (user && window.matchMedia("(max-width: 980px)").matches) this.rail.scrollIntoView({ behavior: reduceMotion() ? "auto" : "smooth", block: "nearest" });
    }

    move(delta) {
      const list = this.navList || this.all; if (!list.length) return;
      const i = Math.max(0, Math.min(list.length - 1, list.findIndex((x) => x.key === this.sel) + delta));
      this.pick(list[i].key, false);
      const b = $(`.row-btn[data-key="${CSS.escape(list[i].key)}"]`, this.mount); if (b) b.focus({ preventScroll: false });
    }

    currentItem() { return (this.all || []).find((i) => i.key === this.sel) || null; }

    drawRail() {
      const rail = this.ev; const item = this.currentItem();
      rail.replaceChildren();
      if (!item) { rail.append(h("div", { class: "card" }, h("p", { class: "muted" }, "Select a sentence to see the evidence behind it."))); return; }
      const st = STATUS[item.status];
      const where = item.source === "removed" ? "Removed from the final answer." : `Sentence ${item.pos + 1} of ${item.total}.`;
      if (item.status === "none") {
        rail.append(h("div", { class: "card" }, h("div", { class: "rail-head" }, h("span", { class: "mark", "data-s": "none" }, ico("dash")), h("div", {}, h("h3", {}, st.title), h("p", {}, where))),
          h("p", { class: "note" }, "This sentence doesn't state a fact, so there's nothing to check. Refusals like “the sources do not say” are left alone.")));
      } else item.claims.forEach((c) => rail.append(this.claimCard(item, c, st, where)));
    }

    claimCard(item, claim, st, where) {
      const rec = this.rec, act = rec.ui.act;
      const v = (item.by[act] || {})[claim.id];
      const label = v ? v.label : "unsupported";
      const card = h("div", { class: "card" });
      card.append(h("div", { class: "rail-head" }, h("span", { class: "mark", "data-s": label }, ico(STATUS[label].icon)), h("div", {}, h("h3", {}, STATUS[label].title), h("p", {}, where))));
      if (!v) return card;
      const win = windowOf(v);
      const showGaps = label !== "supported" && win;
      if (item.claims.length > 1 || normText(claim.text) !== normText(item.sentence.text) || showGaps)
        card.append(h("p", { class: "claim-text" }, showGaps ? claimWithGaps(claim.text, win) : `“${claim.text}”`));
      if (showGaps) card.append(h("p", { class: "gap-note" }, label === "contradicted" ? "Underlined words don't appear in the passage below." : "Underlined words don't appear in the closest passage."));

      const t = (rec.ui.thresholds || {})[act];
      const fill = Math.max(0, Math.min(1, v.support_score));
      const gauge = h("div", { class: "gauge" },
        h("div", { class: "gauge-top" }, h("span", {}, SCORE_LABEL[v.verifier] || "Score"), h("b", {}, v.support_score.toFixed(2))),
        h("div", { class: "gauge-bar", style: { "--v": fill, "--t": t ?? 0 } }, h("div", { class: "gauge-fill", "data-s": label }), t != null ? h("div", { class: "gauge-tick", title: `Needs ${t.toFixed(2)}` }) : null),
        h("div", { class: "gauge-cap" }, h("span", {}, "0"), t != null ? h("span", {}, `Backed at ${t.toFixed(2)} or higher`) : h("span", {}, `${VERIFIER_NAME[v.verifier] || v.verifier}`), h("span", {}, "1")));
      card.append(gauge);

      const chunk = v.evidence_id ? rec.retrieved.find((c) => c.id === v.evidence_id) : null;
      if (chunk) {
        const heading = chunk.text.includes("\n") ? chunk.text.split("\n")[0] : "";
        const body = heading ? chunk.text.split("\n").slice(1).join(" ") : chunk.text;
        const span = locate(body, win);
        const para = h("p", { class: "passage" });
        if (heading) para.append(h("span", { class: "h" }, heading));
        if (span) para.append(body.slice(0, span[0]), h("mark", {}, body.slice(span[0], span[1])), body.slice(span[1]));
        else para.append(body);
        card.append(h("div", { class: "src-head" }, label === "contradicted" ? "Passage that says otherwise" : label === "supported" ? "Evidence" : "Closest passage, which doesn't back it up",
          h("span", { class: "doc-name" }, chunk.doc), h("span", { class: "chip-id" }, chunk.id)), para);
      } else card.append(h("p", { class: "note" }, "No passage came close to this claim."));

      if (claim.cited_ids && claim.cited_ids.length && v.cited_ok != null && chunk) {
        card.append(v.cited_ok ? h("p", { class: "note" }, `The answer cited ${claim.cited_ids.join(", ")}, and that's where the support is.`)
          : h("p", { class: "note warn" }, `The answer cited ${claim.cited_ids.join(", ")}, but the closest passage is ${chunk.id}. Citation error.`));
      }
      const others = Object.keys(item.by).filter((n) => n !== act && item.by[n][claim.id]);
      if (others.length) {
        card.append(h("table", { class: "vtable" }, h("tbody", {}, [act, ...others].map((n) => { const x = item.by[n][claim.id];
          return h("tr", {}, h("th", {}, VERIFIER_NAME[n] || n), h("td", {}, h("span", { class: "pill", "data-s": x.label }, ico(STATUS[x.label].icon), STATUS[x.label].short)), h("td", {}, x.support_score.toFixed(2))); }))));
      }
      return card;
    }

    strictness() {
      const rec = this.rec; const t = rec.ui.thresholds.lexical;
      const out = h("output", {}, t.toFixed(2));
      const input = h("input", { type: "range", id: `${this.kind}-strict`, min: "0.3", max: "0.95", step: "0.05", value: String(t),
        "aria-describedby": `${this.kind}-strict-d`,
        oninput: (e) => { const val = parseFloat(e.target.value); out.textContent = val.toFixed(2); cancelAnimationFrame(this._raf); this._raf = requestAnimationFrame(() => this.rescore(val)); } });
      return h("div", { class: "card strict" }, h("label", { for: `${this.kind}-strict` }, "Strictness", out), input,
        h("small", { id: `${this.kind}-strict-d` }, "How much of a claim's wording and numbers must appear in one passage. Raise it to flag more, lower it to flag less. Re-scores instantly."));
    }
    rescore(val) {
      const keep = this.sel; this.rec = E.reverify(this.rec, val);
      S.settings.threshold = val; saveSettings();
      this.sel = keep; this.refresh(false);
    }

    copyAnswer() {
      const snap = this.view === "verified" ? this.rec.final : this.rec.baseline;
      const text = snap.sentences.map((x) => x.text).join(" ");
      copyText(text || "The sources do not contain enough information to answer this question.", this.view === "verified" ? "Fixed answer copied" : "Answer copied");
    }

    ledger(sum, act) {
      const rows = [];
      sum.items.forEach((it) => it.claims.forEach((c) => {
        const v = (it.by[act] || {})[c.id]; if (!v) return;
        if (this.filter !== "all" && v.label !== this.filter) return;
        rows.push(h("tr", { "data-key": it.key, "aria-selected": String(it.key === this.sel), onclick: () => { this.view = "generated"; this.sel = it.key; this.refresh(); } },
          h("td", {}, c.text), h("td", {}, h("span", { class: "pill", "data-s": v.label }, ico(STATUS[v.label].icon), STATUS[v.label].short)),
          h("td", { class: "num" }, v.support_score.toFixed(2)), h("td", {}, v.evidence_id ? h("span", { class: "chip-id" }, v.evidence_id) : "none"),
          h("td", {}, v.cited_ok == null ? "No citation" : v.cited_ok ? "Correct" : "Wrong source")));
      }));
      if (!sum.n) return h("div");
      const opts = [["all", "All", sum.n], ["supported", "Backed", sum.counts.supported], ["unsupported", "Not in sources", sum.counts.unsupported], ["contradicted", "Contradicted", sum.counts.contradicted]].filter((o) => o[0] === "all" || o[2]);
      const filters = h("div", { class: "filters", role: "group", "aria-label": "Filter claims" }, opts.map(([k, l, c]) =>
        h("button", { type: "button", "aria-pressed": String(this.filter === k), onclick: () => { this.filter = k; this.ledgerEl.replaceChildren(this.ledger(this.sum, act)); } }, `${l} ${c}`)));
      return h("div", {}, h("h3", { class: "section-title" }, "Claim by claim", opts.length > 2 ? filters : null),
        h("div", { class: "table-wrap" }, h("table", { class: "ledger" },
          h("thead", {}, h("tr", {}, h("th", {}, "Claim"), h("th", {}, "Verdict"), h("th", { class: "num" }, SCORE_LABEL[act] || "Score"), h("th", {}, "Evidence"), h("th", {}, "Citation"))),
          h("tbody", {}, rows.length ? rows : h("tr", {}, h("td", { colspan: "5" }, "No claims with this verdict."))))));
    }

    sources() {
      const rec = this.rec; const act = rec.ui.act; const sum = this.sum;
      const cited = new Set(); const evidence = new Set();
      sum.items.forEach((it) => { (it.sentence.cited_ids || []).forEach((i) => cited.add(i)); it.claims.forEach((c) => { const v = (it.by[act] || {})[c.id]; if (v && v.evidence_id) evidence.add(v.evidence_id); }); });
      let list = rec.retrieved;
      const audit = rec.ui.mode === "audit";
      if (audit) list = list.filter((c) => evidence.has(c.id) || cited.has(c.id));
      if (!list.length) return h("div");
      const max = Math.max(...list.map((c) => c.score || 0), 0.0001);
      const cards = list.map((c) => {
        const heading = c.text.includes("\n") ? c.text.split("\n")[0] : "";
        const body = heading ? c.text.split("\n").slice(1).join(" ") : c.text;
        const card = h("div", { class: "source", "data-id": c.id },
          h("div", { class: "source-top" }, h("span", { class: "chip-id" }, c.id), h("span", { class: "doc-name" }, heading ? `${c.doc} · ${heading}` : c.doc),
            cited.has(c.id) ? h("span", { class: "tag" }, "cited") : null, h("span", { class: "tag on", "data-evid": "1", hidden: true }, "evidence")),
          c.score != null ? h("div", { class: "rankbar", title: `Retrieval score ${c.score}` }, h("i", { style: { "--v": (c.score || 0) / max } })) : null,
          h("p", {}, body));
        const b = h("button", { class: "more", type: "button", onclick: () => { card.classList.toggle("open"); b.textContent = card.classList.contains("open") ? "Show less" : "Read all"; } }, "Read all");
        if (body.length > 220) card.append(b);
        return card;
      });
      this.srcCards = cards;
      const title = audit ? "Passages used" : `Passages retrieved (${list.length})`;
      const wrap = h("div", {}, h("h3", { class: "section-title" }, title), h("div", { class: "sources" }, cards));
      setTimeout(() => this.markSources(), 0);
      return wrap;
    }
    markSources() {
      const item = this.currentItem(); const act = this.rec.ui.act;
      const ids = new Set();
      if (item) item.claims.forEach((c) => { const v = (item.by[act] || {})[c.id]; if (v && v.evidence_id) ids.add(v.evidence_id); });
      $$(".source", this.mount).forEach((el) => { const on = ids.has(el.dataset.id); el.classList.toggle("hot", on); const t = $("[data-evid]", el); if (t) t.hidden = !on; });
    }

    exportMenu() {
      const wrapEl = h("div", { class: "menu" });
      const btn = h("button", { class: "btn small", type: "button", "aria-haspopup": "menu", "aria-expanded": "false" }, ico("share"), "Share");
      let pop = null;
      const close = () => { if (pop) { pop.remove(); pop = null; btn.setAttribute("aria-expanded", "false"); document.removeEventListener("click", away, true); } };
      const away = (e) => { if (pop && !wrapEl.contains(e.target)) close(); };
      const item = (label, fn) => h("button", { type: "button", role: "menuitem", onclick: () => { close(); fn(); } }, label);
      btn.addEventListener("click", () => {
        if (pop) return close();
        pop = h("div", { class: "menu-pop", role: "menu" },
          item("Copy report as Markdown", () => copyText(markdownReport(this.rec), "Report copied")),
          item("Copy as JSON", () => copyText(JSON.stringify(this.rec, null, 2), "JSON copied")),
          embedded ? null : item("Download JSON", () => download("truthlens-result.json", JSON.stringify(this.rec, null, 2))));
        wrapEl.append(pop); btn.setAttribute("aria-expanded", "true"); document.addEventListener("click", away, true);
      });
      wrapEl.append(btn); return wrapEl;
    }
  }
    function download(name, text) {
    try { const a = h("a", { href: URL.createObjectURL(new Blob([text], { type: "application/json" })), download: name }); document.body.append(a); a.click(); a.remove(); } catch (_) { copyText(text, "JSON copied"); }
  }

  // ------------------------------------------------------------------ ASK view
  const askResult = new Result($("#ask-result"), "ask");
  const auditResult = new Result($("#audit-result"), "audit");
  const QUESTIONS = DATA.questions;
  const FEATURED = ["q005", "q013", "q003", "q021", "q017", "q009"];
  const KIND_NAME = { answerable: "Answerable", partial: "Only partly answerable", unanswerable: "Not in the documents", false_premise: "False premise" };

  function renderChips() {
    const box = $("#chips"); box.replaceChildren();
    FEATURED.map((id) => QUESTIONS.find((q) => q.id === id)).filter(Boolean).forEach((q) =>
      box.append(h("button", { class: "chip", type: "button", onclick: () => { $("#question").value = q.question; submitAsk(); } }, q.question)));
    box.append(h("button", { class: "chip more", type: "button", onclick: openQuestions }, `All ${QUESTIONS.length} questions`));
    box.append(h("button", { class: "chip more", type: "button", onclick: () => { const q = QUESTIONS[Math.floor(Math.random() * QUESTIONS.length)]; $("#question").value = q.question; submitAsk(); } }, "Surprise me"));
  }
  function openQuestions() {
    const body = $("#qd-body"); body.replaceChildren();
    for (const kind of Object.keys(KIND_NAME)) {
      const qs = QUESTIONS.filter((q) => q.kind === kind); if (!qs.length) continue;
      body.append(h("div", { class: "qgroup" }, h("h3", {}, KIND_NAME[kind]), qs.map((q) =>
        h("button", { class: "chip", type: "button", onclick: () => { $("#qdialog").close(); setView("ask"); $("#question").value = q.question; submitAsk(); } }, q.question))));
    }
    $("#qdialog").showModal();
  }

  let busy = false;
  function setBusy(btn, on, idle) { btn.disabled = on; btn.textContent = on ? "Checking…" : idle; }
  async function submitAsk() {
    const q = $("#question").value.trim(); if (!q || busy) return;
    busy = true; setBusy($("#ask-btn"), true, "Check answer");
    if (useServer()) askResult.mount.replaceChildren(h("div", { class: "skeleton", "aria-label": "Checking" }, h("i", {}), h("i", {}), h("i", {})));
    try {
      const rec = await runAsk(q); saveHistory(rec); await askResult.show(rec, { animate: true, scroll: true });
    } catch (e) { askResult.fail(friendly(e)); } finally { busy = false; setBusy($("#ask-btn"), false, "Check answer"); }
  }
  const friendly = (e) => (e instanceof TypeError ? "Couldn't reach the Python server. Switch to the browser engine in Settings, or start server.py again." : e.message || String(e));
  $("#ask-form").addEventListener("submit", (e) => { e.preventDefault(); submitAsk(); });

  // ------------------------------------------------------------------ AUDIT view
  const AUDIT_EXAMPLES = [
    ["Marketing blurb", "The Helio H10 has 10 kWh of usable capacity and weighs 98 kg. It comes with a 15-year standard warranty. The enclosure is rated IP55. Helio ships to 40 countries."],
    ["Support reply", "Yes, you can install it in a bedroom as long as there is a smoke detector within 3 m. The unit needs a dedicated 32 A circuit breaker. Phone support is open 24 hours a day."],
    ["Warranty summary", "The standard warranty lasts 10 years and starts on the day you buy the battery. Extended coverage to 15 years costs 4 percent of the purchase price. The battery supports 5G for remote monitoring."],
  ];
  function renderAuditUi() {
    const ex = $("#audit-examples"); ex.replaceChildren(h("span", { class: "muted" }, "Try an example:"));
    AUDIT_EXAMPLES.forEach(([name, text]) => ex.append(h("button", { class: "chip", type: "button", onclick: () => { $("#audit-text").value = text; submitAudit(); } }, name)));
    const list = $("#against-list"); list.replaceChildren();
    S.docs.forEach((d) => list.append(h("li", {}, h("b", {}, d.name), h("span", {}, `${S.chunks.filter((c) => c.doc === d.name).length} passages`))));
    if (!S.docs.length) list.append(h("li", {}, "No documents yet."));
  }
  async function submitAudit() {
    const text = $("#audit-text").value.trim(); if (!text || busy) return;
    busy = true; setBusy($("#audit-btn"), true, "Check text");
    try {
      if (!S.chunks.length && !useServer()) throw new Error("There are no documents to check against. Add one on the Documents page.");
      const rec = await runAudit(text); saveHistory(rec); await auditResult.show(rec, { animate: true, scroll: true });
    } catch (e) { auditResult.fail(friendly(e)); } finally { busy = false; setBusy($("#audit-btn"), false, "Check text"); }
  }
  $("#audit-form").addEventListener("submit", (e) => { e.preventDefault(); submitAudit(); });

  // ------------------------------------------------------------------ LIBRARY view
  let libSel = null;
  function persistDocs() { S.docs.sort((a, b) => (a.name < b.name ? -1 : 1)); store.set("docs", S.docs); rebuildIndex(); renderAuditUi(); renderLibrary(); renderNav(); }
  function renderLibrary() {
    const root = $("#lib"); root.replaceChildren();
    if (!S.docs.find((d) => d.name === libSel)) libSel = S.docs[0] ? S.docs[0].name : null;
    const left = h("div", {});
    const list = h("ul", { class: "doc-list" }, S.docs.map((d) => {
      const n = S.chunks.filter((c) => c.doc === d.name).length;
      return h("li", {}, h("button", { class: "doc-item", type: "button", "aria-pressed": String(d.name === libSel), onclick: () => { libSel = d.name; renderLibrary(); } },
        h("b", {}, d.name), h("span", {}, `${n} passages, ${d.text.split(/\s+/).filter(Boolean).length} words${d.sample ? "" : ", yours"}`)));
    }));
    if (!S.docs.length) list.append(h("li", {}, h("p", { class: "muted" }, "No documents yet. Add one below.")));
    const nameIn = h("input", { type: "text", id: "new-name", placeholder: "pricing-faq.md", autocomplete: "off" });
    const textIn = h("textarea", { id: "new-text", placeholder: "Paste text. Lines starting with # become passage headings." });
    const addDoc = (name, text) => {
      name = (name || "").trim().replace(/[^\w.\- ]+/g, "") || "document.md"; if (!/\.(md|txt)$/i.test(name)) name += ".md";
      if (!text.trim()) { toast("Add some text first."); return; }
      S.docs = S.docs.filter((d) => d.name !== name); S.docs.push({ name, text: text.trim(), sample: false }); libSel = name; persistDocs(); toast(`Added ${name}`);
    };
    const fileIn = h("input", { type: "file", id: "file-in", accept: ".md,.txt,text/plain,text/markdown", multiple: true, class: "sr" });
    const readFiles = (files) => Array.from(files).forEach((f) => { if (f.size > 400000) { toast(`${f.name} is too large (400 KB max).`); return; } const r = new FileReader(); r.onload = () => addDoc(f.name, String(r.result)); r.readAsText(f); });
    fileIn.addEventListener("change", () => readFiles(fileIn.files));
    const drop = h("label", { class: "drop", for: "file-in" }, ico("upload"), h("span", {}, "Drop .md or .txt files here, or choose files"));
    ["dragover", "dragenter"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); if (ev === "drop") readFiles(e.dataTransfer.files); }));
    left.append(h("h2", { class: "h-sm" }, "Files"), list, h("div", { class: "card" }, h("h2", { class: "h-sm" }, "Add a document"),
      drop, fileIn, h("div", { class: "field", style: { "margin-top": "12px" } }, h("label", { for: "new-name" }, "Name"), nameIn),
      h("div", { class: "field" }, h("label", { for: "new-text" }, "Text"), textIn),
      h("button", { class: "btn primary small", type: "button", onclick: () => { addDoc(nameIn.value, textIn.value); } }, ico("plus"), "Add document")),
      h("p", {}, h("button", { class: "btn quiet small", type: "button", onclick: () => { S.docs = DATA.docs.map((d) => ({ ...d, sample: true })); libSel = null; persistDocs(); toast("Sample documents restored"); } }, "Restore the samples")));

    const right = h("div", {});
    const doc = S.docs.find((d) => d.name === libSel);
    if (doc) {
      right.append(h("div", { class: "row", style: { margin: "0 0 12px", "justify-content": "space-between" } }, h("h2", { class: "h-sm", style: { margin: "0" } }, `${doc.name}, as the checker sees it`),
        S.docs.length ? h("button", { class: "btn quiet small", type: "button", onclick: () => { const gone = doc; S.docs = S.docs.filter((d) => d.name !== gone.name); persistDocs(); toast(`Removed ${gone.name}`, () => { S.docs.push(gone); libSel = gone.name; persistDocs(); }); } }, ico("trash"), "Remove") : null));
      right.append(h("div", { class: "chunk-list" }, S.chunks.filter((c) => c.doc === doc.name).map((c) => {
        const heading = c.text.includes("\n") ? c.text.split("\n")[0] : ""; const body = heading ? c.text.split("\n").slice(1).join(" ") : c.text;
        return h("div", { class: "chunk" }, h("h4", {}, h("span", { class: "chip-id" }, c.id), heading || "No heading"), h("p", {}, body));
      })));
    } else right.append(h("div", { class: "empty" }, h("h3", {}, "No documents"), h("p", {}, "Add a document or restore the samples to start asking questions.")));

    const q = h("input", { type: "text", id: "tester-q", placeholder: "Type a question to see which passages rank first", autocomplete: "off" });
    const out = h("div", {});
    const run = () => {
      out.replaceChildren(); const text = q.value.trim(); if (!text || !S.chunks.length) return;
      const hits = S.retriever.retrieve(text, 5);
      hits.forEach((c) => { const heading = c.text.includes("\n") ? c.text.split("\n")[0] : ""; out.append(h("div", { class: "hit" }, h("span", { class: "sc" }, c.score.toFixed(2)),
        h("p", {}, h("span", { class: "chip-id" }, c.id), " ", h("b", {}, `${c.doc}${heading ? ` · ${heading}` : ""}`), " ", (c.text.includes("\n") ? c.text.split("\n").slice(1).join(" ") : c.text).slice(0, 180)))); });
    };
    q.addEventListener("input", run);
    right.append(h("div", { class: "card tester" }, h("h2", { class: "h-sm" }, "Test retrieval"), h("div", { class: "field" }, h("label", { class: "sr", for: "tester-q" }, "Question"), q), out));
    root.append(left, right);
  }

  // ------------------------------------------------------------------ STRESS TEST view
  let benchData = null;
  async function runBench() {
    const btn = $("#bench-run"); btn.disabled = true; btn.textContent = "Running…"; await sleep(30);
    try {
      const items = E.benchBuild(QUESTIONS, S.retriever, S.settings.k);
      const grid = []; for (let t = 0.3; t <= 0.951; t += 0.05) grid.push(Math.round(t * 100) / 100);
      const scores = E.benchScore(items, grid);
      benchData = { items, grid, scores, at: Date.now() };
      renderBench();
    } finally { btn.disabled = false; btn.textContent = "Run again"; }
  }
  function prf(items, flags) {
    let tp = 0, fp = 0, fn = 0, tn = 0;
    items.forEach((it, i) => { if (it.label && flags[i]) tp++; else if (!it.label && flags[i]) fp++; else if (it.label) fn++; else tn++; });
    const p = tp + fp ? tp / (tp + fp) : 0, r = tp + fn ? tp / (tp + fn) : 0;
    return { tp, fp, fn, tn, p, r, f1: p + r ? (2 * p * r) / (p + r) : 0 };
  }
  const MUT_NAME = { fabricate: "Invented fact", number: "Changed number", entity: "Swapped model (H10 and H15)", negate: "Flipped negation" };
  function renderBench() {
    const out = $("#bench-out"); out.replaceChildren(); if (!benchData) return;
    const { items, grid, scores } = benchData; const t0 = S.settings.threshold;
    const near = grid.reduce((a, b) => (Math.abs(b - t0) < Math.abs(a - t0) ? b : a));
    const m = prf(items, scores[near]);
    const pct = (x) => `${Math.round(x * 100)}%`;
    out.append(h("div", { class: "grid4" },
      h("div", { class: "metric" }, h("b", {}, pct(m.p)), h("span", {}, `Precision: of ${m.tp + m.fp} sentences flagged, ${m.tp} were really false`)),
      h("div", { class: "metric" }, h("b", {}, pct(m.r)), h("span", {}, `Recall: caught ${m.tp} of ${m.tp + m.fn} planted errors`)),
      h("div", { class: "metric" }, h("b", {}, m.f1.toFixed(2)), h("span", {}, `F1 at strictness ${near.toFixed(2)}`)),
      h("div", { class: "metric" }, h("b", {}, String(items.length)), h("span", {}, `Sentences tested, ${items.filter((i) => i.label).length} of them false`))));

    // recall by kind
    const kinds = E.MUTATIONS.map((k) => {
      const idx = items.map((it, i) => (it.mutation === k ? i : -1)).filter((i) => i >= 0);
      const hit = idx.filter((i) => scores[near][i]).length;
      return { k, n: idx.length, hit };
    });
    const bars = h("div", { class: "bars" }, kinds.map(({ k, n, hit }) => {
      const v = n ? hit / n : 0;
      return h("div", { class: "bar-row" }, h("span", {}, MUT_NAME[k]), h("div", { class: "track" }, h("i", { style: { "--v": v }, "data-lo": v < 0.5 ? "1" : "0" })), h("span", { class: "val" }, `${hit} of ${n} caught`));
    }));
    const WEAK = { fabricate: "invented facts", number: "changed numbers", entity: "a swapped model name", negate: "flipped negations" };
    const weak = kinds.filter((x) => x.n && x.hit / x.n < 0.5).map((x) => WEAK[x.k]);
    if (weak.length) out.append(h("div", { class: "callout" }, h("b", {}, "Blind spot."), h("p", {}, `The lexical checker misses ${weak.join(" and ")}. It compares words and numbers, so a sentence that reuses the right words with a wrong subject still passes. The NLI and LLM-judge verifiers in the Python server are built for this case.`)));
    out.append(h("h3", { class: "section-title", style: { "margin-top": "0" } }, "What gets caught"), h("div", { class: "chart-card" }, bars));
    out.append(h("h3", { class: "section-title" }, "Strictness trade-off"), h("div", { class: "chart-card" }, sweepChart(items, grid, scores, near)));
    out.append(h("p", { class: "muted", style: { maxWidth: "78ch" } }, "The truth is known here because we planted the errors ourselves, so this checks that the pipeline works. It doesn't estimate accuracy on real model answers. For that, label your own answers with the steps in the README."));
  }
  function sweepChart(items, grid, scores, cur) {
    const W = 760, Ht = 280, L = 44, R = 70, T = 14, B = 40;
    const x = (t) => L + ((t - grid[0]) / (grid[grid.length - 1] - grid[0])) * (W - L - R), y = (v) => T + (1 - v) * (Ht - T - B);
    const NS = "http://www.w3.org/2000/svg";
    const el = (n, a, ...k) => { const e = document.createElementNS(NS, n); for (const [kk, v] of Object.entries(a || {})) e.setAttribute(kk, v); k.forEach((c) => e.append(c)); return e; };
    const svg = el("svg", { viewBox: `0 0 ${W} ${Ht}`, role: "img", "aria-label": "Precision, recall and F1 as strictness changes from 0.3 to 0.95" });
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      svg.append(el("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: "var(--line-soft)", "stroke-width": 1 }));
      const t = el("text", { x: L - 8, y: y(v) + 4, "text-anchor": "end" }); t.textContent = v.toFixed(2).replace(/0$/, ""); svg.append(t);
    }
    grid.filter((_, i) => i % 2 === 0).forEach((tt) => { const t = el("text", { x: x(tt), y: Ht - B + 20, "text-anchor": "middle" }); t.textContent = tt.toFixed(2); svg.append(t); });
    const ax = el("text", { x: (L + W - R) / 2, y: Ht - 4, "text-anchor": "middle" }); ax.textContent = "Strictness"; svg.append(ax);
    const series = [["Precision", "p", "var(--ink-2)"], ["Recall", "r", "var(--accent)"], ["F1", "f1", "var(--ok)"]];
    const pts = grid.map((t) => prf(items, scores[t]));
    const ends = [];
    series.forEach(([name, key, col]) => {
      const d = pts.map((p, i) => `${i ? "L" : "M"}${x(grid[i]).toFixed(1)},${y(p[key]).toFixed(1)}`).join(" ");
      svg.append(el("path", { d, fill: "none", stroke: col, "stroke-width": key === "f1" ? 3 : 2, "stroke-linejoin": "round", "stroke-dasharray": key === "p" ? "6 4" : "none" }));
      ends.push({ name, col, y: y(pts[pts.length - 1][key]) + 4 });
    });
    ends.sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 17) ends[i].y = ends[i - 1].y + 17;
    ends.forEach((e) => { const lab = el("text", { x: W - R + 8, y: e.y, class: "lab", style: `fill:${e.col}` }); lab.textContent = e.name; svg.append(lab); });
    svg.append(el("line", { x1: x(cur), x2: x(cur), y1: T, y2: Ht - B, stroke: "var(--ink-3)", "stroke-width": 1, "stroke-dasharray": "2 3" }));
    const cl = el("text", { x: x(cur), y: Ht - B - 8, "text-anchor": "middle" }); cl.textContent = "current strictness"; svg.append(cl);
    return svg;
  }
  $("#bench-run").addEventListener("click", runBench);

  // ------------------------------------------------------------------ HISTORY view
  function saveHistory(rec) {
    S.history.unshift({ id: Date.now() + Math.random().toString(36).slice(2, 6), at: Date.now(), rec });
    S.history = S.history.slice(0, 25);
    if (!store.set("history", S.history)) { S.history = S.history.slice(0, 8); store.set("history", S.history); }
    if (typeof renderNav === "function" && current) { renderNav(); renderStats(); renderRecent(); }
  }
  const ago = (t) => { const s = Math.round((Date.now() - t) / 1000); if (s < 60) return "just now"; if (s < 3600) return `${Math.round(s / 60)} min ago`; if (s < 86400) return `${Math.round(s / 3600)} h ago`; return new Date(t).toLocaleDateString(); };
  let histQuery = "", histMode = "all";
  function commitHistory() { store.set("history", S.history); renderHistory(); renderNav(); renderStats(); renderRecent(); }
  function renderHistory() {
    const out = $("#history-out"), tools = $("#history-tools"); out.replaceChildren(); tools.replaceChildren();
    if (!S.history.length) { out.append(h("div", { class: "empty" }, h("h3", {}, "Nothing checked yet"), h("p", {}, "Ask a question or check some text, and it will show up here."), h("a", { class: "btn primary", href: "#ask" }, "Ask a question"))); return; }
    const q = h("input", { class: "search", type: "search", id: "hist-q", placeholder: "Search your checks", value: histQuery, "aria-label": "Search history", oninput: (e) => { histQuery = e.target.value; drawList(); } });
    const modes = h("div", { class: "seg-ctl", role: "group", "aria-label": "Filter by type" }, [["all", "All"], ["ask", "Questions"], ["audit", "Pasted text"]].map(([k, l]) =>
      h("button", { type: "button", "aria-selected": String(histMode === k), onclick: () => { histMode = k; renderHistory(); } }, l)));
    tools.append(q, modes,
      h("button", { class: "btn small", type: "button", onclick: () => copyText(JSON.stringify(S.history.map((e) => e.rec), null, 2), "History copied as JSON") }, ico("copy"), "Copy all as JSON"),
      h("button", { class: "btn quiet small", type: "button", onclick: clearHistory }, ico("trash"), "Clear"));
    const list = h("div", { class: "hist" }); out.append(list);
    function drawList() {
      list.replaceChildren();
      const items = S.history.filter((e) => (histMode === "all" || e.rec.ui.mode === histMode) &&
        (!histQuery || (e.rec.question + " " + e.rec.baseline.sentences.map((x) => x.text).join(" ")).toLowerCase().includes(histQuery.toLowerCase())));
      if (!items.length) { list.append(h("div", { class: "empty" }, h("h3", {}, "No matches"), h("p", {}, "Try a different search or filter."))); return; }
      items.forEach((e) => {
        const sum = summarize(e.rec); const rec = e.rec;
        const title = rec.question || rec.baseline.sentences.map((x) => x.text).join(" ").slice(0, 110);
        list.append(h("div", { class: "hist-item" },
          h("div", {}, h("h3", {}, title), h("p", {}, `${rec.ui.mode === "audit" ? "Pasted text" : "Question"}, ${sum.n} claims, ${sum.flagged} flagged, ${ago(e.at)}`),
            h("div", { class: "mini", "aria-hidden": "true" }, sum.vs.map((v) => h("i", { "data-s": v.label })))),
          h("div", { class: "hist-actions" },
            h("button", { class: "btn small", type: "button", onclick: () => reopen(e) }, ico("open"), "Open"),
            h("button", { class: "icon-btn", type: "button", "aria-label": "Delete this entry", onclick: () => { const at = S.history.indexOf(e); S.history = S.history.filter((x) => x.id !== e.id); commitHistory();
              toast("Entry deleted", () => { S.history.splice(Math.min(at, S.history.length), 0, e); commitHistory(); }); } }, ico("trash")))));
      });
    }
    drawList();
  }
  function clearHistory() {
    if (!S.history.length) return;
    const old = S.history; S.history = []; commitHistory();
    toast("History cleared", () => { S.history = old; commitHistory(); });
  }
  function renderRecent() {
    const box = $("#recent"); if (!box) return;
    const list = S.history.slice(0, 4); box.hidden = !list.length; box.replaceChildren(); if (!list.length) return;
    box.append(h("h2", { class: "h-sm" }, "Recent checks"), h("div", { class: "recent-list" }, list.map((e) => {
      const sm = summarize(e.rec); const st = !sm.flagged ? "supported" : sm.counts.contradicted ? "contradicted" : "unsupported";
      return h("button", { class: "recent-item", type: "button", onclick: () => reopen(e) },
        h("span", {}, e.rec.question || "Pasted text: " + e.rec.baseline.sentences.map((x) => x.text).join(" ").slice(0, 80)),
        h("span", { class: "pill", "data-s": st }, ico(STATUS[st].icon), sm.flagged ? `${sm.flagged} flagged` : "All backed"), h("small", {}, ago(e.at)));
    })));
  }
  function reopen(e) {
    const rec = e.rec;
    if (rec.ui.mode === "audit") { $("#audit-text").value = rec.baseline.sentences.map((s) => s.raw).join(" "); setView("audit"); auditResult.show(rec, { animate: false }); }
    else { $("#question").value = rec.question; setView("ask"); askResult.show(rec, { animate: false }); }
  }

  // ------------------------------------------------------------------ settings
  const dlg = $("#settings");
  function renderSettings() {
    const body = $("#settings-body"); body.replaceChildren();
    const s = S.settings;
    const radio = (name, value, label, hint, checked, off) => h("label", { class: "opt", "data-off": off ? "1" : null },
      h("input", { type: "radio", name, value, checked: checked || null, disabled: off || null }), h("span", {}, label, hint ? h("small", {}, hint) : null));
    const group = (title, hint, ...kids) => h("div", { class: "group" }, h("h3", {}, title), hint ? h("p", { class: "hint" }, hint) : null, kids);
    body.append(group("Engine", S.server ? "A Python server answered, so you can use the full pipeline." : "No Python server found, so everything runs in this page.",
      radio("engine", "browser", "In this browser", "Demo answer writer and the lexical checker. Works offline.", s.engine === "browser" || !S.server),
      radio("engine", "server", "Python server", S.server ? (S.meta.live_ready ? `Real model (${S.meta.label || S.meta.provider}) and every verifier` : "Demo model; add an API key to use a real one") : "Run python server.py to enable this.", s.engine === "server" && S.server, !S.server)));
    body.append(group("Appearance", "Compact tightens the page margins.",
      radio("density", "comfortable", "Comfortable", null, (s.density || "comfortable") === "comfortable"), radio("density", "compact", "Compact", null, s.density === "compact")));
    body.append(group("Passages per question", "How many source passages the answer can draw on.",
      h("div", { class: "strict" }, h("label", { for: "set-k" }, "Passages", h("output", { id: "set-k-out" }, String(s.k))), h("input", { type: "range", id: "set-k", min: "2", max: "10", step: "1", value: String(s.k) }))));
    body.append(group("Default strictness", "Applies to new checks in the browser engine. You can also change it on any result.",
      h("div", { class: "strict" }, h("label", { for: "set-t" }, "Strictness", h("output", { id: "set-t-out" }, s.threshold.toFixed(2))), h("input", { type: "range", id: "set-t", min: "0.3", max: "0.95", step: "0.05", value: String(s.threshold) }))));
    if (S.server) {
      const m = S.meta;
      body.append(group("Python server options", "Used only when the engine is Python server.",
        h("div", { class: "group" }, h("h3", {}, "Verifiers"), m.verifiers.map((v) => h("label", { class: "opt", "data-off": v.available ? null : "1" },
          h("input", { type: "checkbox", name: "verifiers", value: v.name, checked: s.verifiers.includes(v.name) || null, disabled: !v.available || null }), h("span", {}, v.label, h("small", {}, v.available ? v.note : `Unavailable. ${v.note}`))))),
        h("div", { class: "field" }, h("label", { for: "set-act" }, "Verifier that fixes the answer"), h("select", { id: "set-act" }, s.verifiers.map((n) => h("option", { value: n, selected: n === s.act || null }, VERIFIER_NAME[n] || n)))),
        h("div", { class: "field" }, h("label", { for: "set-action" }, "When claims are flagged"), h("select", { id: "set-action" }, [["flag", "Only mark them"], ["remove", "Remove those sentences"], ["regenerate", "Ask the model to rewrite, then remove what's left"]].map(([v, l]) => h("option", { value: v, selected: v === s.action || null }, l)))),
        h("div", { class: "field" }, h("label", { for: "set-style" }, "Answer style"), h("select", { id: "set-style" }, [["strict", "Strict: sources only"], ["permissive", "Permissive: may add background knowledge"]].map(([v, l]) => h("option", { value: v, selected: v === s.style || null }, l)))),
        h("div", { class: "field" }, h("label", { for: "set-claims" }, "Claim splitting"), h("select", { id: "set-claims" }, [["rule", "Rules"], ["llm", "LLM"]].map(([v, l]) => h("option", { value: v, selected: v === s.claim_mode || null }, l))))));
    }
  }
  function readSettings() {
    const s = S.settings; const val = (id) => { const e = $(id, dlg); return e ? e.value : null; };
    const eng = $('input[name="engine"]:checked', dlg); if (eng) s.engine = eng.value;
    const den = $('input[name="density"]:checked', dlg); if (den) { s.density = den.value; document.body.dataset.density = den.value; }
    s.k = parseInt(val("#set-k") || s.k, 10); s.threshold = parseFloat(val("#set-t") || s.threshold);
    if (S.server) {
      const vs = $$('input[name="verifiers"]:checked', dlg).map((i) => i.value); s.verifiers = vs.length ? vs : ["lexical"];
      s.act = s.verifiers.includes(val("#set-act")) ? val("#set-act") : s.verifiers[0]; s.action = val("#set-action") || s.action; s.style = val("#set-style") || s.style; s.claim_mode = val("#set-claims") || s.claim_mode;
    }
    saveSettings(); updateEngineChip();
  }
  dlg.addEventListener("input", (e) => {
    if (e.target.id === "set-k") $("#set-k-out", dlg).textContent = e.target.value;
    if (e.target.id === "set-t") $("#set-t-out", dlg).textContent = parseFloat(e.target.value).toFixed(2);
    if (e.target.name === "verifiers") { readSettings(); const keepScroll = $("#settings-body").scrollTop; renderSettings(); $("#settings-body").scrollTop = keepScroll; }
  });
  dlg.addEventListener("change", readSettings);
  dlg.addEventListener("close", () => { readSettings(); renderBench(); });
  $("#settings-reset").addEventListener("click", () => { S.settings = { ...DEFAULTS, engine: S.server && S.meta.live_ready ? "server" : "browser" }; saveSettings(); renderSettings(); updateEngineChip(); });
  const openSettings = () => { renderSettings(); dlg.showModal(); };
  $("#settings-btn").addEventListener("click", openSettings);
  $("#engine-chip").addEventListener("click", openSettings);
  function updateEngineChip() {
    const chip = $("#engine-chip"), lab = $("#engine-label"); const srv = useServer();
    chip.classList.toggle("server", srv);
    lab.textContent = srv ? "Python server" : "Browser engine";
    $("#engine-sub").textContent = srv ? (S.meta && S.meta.live_ready ? (S.meta.label || S.meta.provider || "Live model") : "Demo model") : "Runs on this page";
    $("#ask-btn").title = srv ? "Runs the Python pipeline" : "Runs in this page";
  }

  // ------------------------------------------------------------------ theme
  const themeBtn = $("#theme-btn"), settingsBtn = $("#settings-btn");
  settingsBtn.append(ico("sliders"), h("span", {}, "Settings"));
  function applyTheme(mode) {
    const root = document.documentElement;
    if (mode === "light" || mode === "dark") root.setAttribute("data-theme", mode); else root.removeAttribute("data-theme");
    const dark = mode === "dark" || (mode !== "light" && window.matchMedia("(prefers-color-scheme: dark)").matches);
    themeBtn.replaceChildren(ico(dark ? "sun" : "moon"), h("span", {}, dark ? "Light" : "Dark"));
  }
  let theme = store.get("theme", "system");
  applyTheme(theme);
  function toggleTheme() {
    const dark = document.documentElement.getAttribute("data-theme") === "dark" || (!document.documentElement.getAttribute("data-theme") && window.matchMedia("(prefers-color-scheme: dark)").matches);
    theme = dark ? "light" : "dark"; store.set("theme", theme); applyTheme(theme);
  }
  themeBtn.addEventListener("click", toggleTheme);

  // ------------------------------------------------------------------ sidebar collapse, palette, shortcuts
  const collapseBtn = $("#collapse-btn"); collapseBtn.append(ico("panel"), h("span", {}, "Collapse"));
  function applyCollapse(on, save = true) {
    document.body.classList.toggle("side-collapsed", on); if (save) store.set("collapsed", on);
    collapseBtn.setAttribute("aria-label", on ? "Expand sidebar" : "Collapse sidebar");
  }
  collapseBtn.addEventListener("click", () => applyCollapse(!document.body.classList.contains("side-collapsed")));
  applyCollapse(!!store.get("collapsed", false), false);
  $(".ico-slot").replaceWith(ico("search"));
  $("#help-btn").append(ico("help"));
  document.body.dataset.density = S.settings.density || "comfortable";

  const pal = $("#palette"), palIn = $("#pal-input"), palList = $("#pal-list");
  let palItems = [], palSel = 0;
  const KIND_SHORT = { answerable: "Answerable", partial: "Partial", unanswerable: "Not in documents", false_premise: "False premise" };
  function commands() {
    const c = [];
    NAV.filter(Boolean).forEach(([v, l, i]) => c.push({ g: "Go to", label: l, icon: i, run: () => setView(v) }));
    c.push({ g: "Actions", label: "Run the stress test", icon: "pulse", run: () => { setView("benchmark"); runBench(); } });
    c.push({ g: "Actions", label: "Switch theme", icon: "sun", run: toggleTheme });
    c.push({ g: "Actions", label: "Open settings", icon: "sliders", run: () => openSettings() });
    c.push({ g: "Actions", label: document.body.classList.contains("side-collapsed") ? "Expand sidebar" : "Collapse sidebar", icon: "panel", run: () => collapseBtn.click() });
    c.push({ g: "Actions", label: "Show keyboard shortcuts", icon: "help", run: () => openHelp() });
    if (S.history.length) c.push({ g: "Actions", label: "Clear history", icon: "trash", run: clearHistory });
    QUESTIONS.forEach((q) => c.push({ g: "Ask a sample question", label: q.question, icon: "chat", hint: KIND_SHORT[q.kind], run: () => { setView("ask"); $("#question").value = q.question; submitAsk(); } }));
    return c;
  }
  function drawPalette() {
    const terms = palIn.value.toLowerCase().split(/\s+/).filter(Boolean);
    palItems = commands().filter((c) => terms.every((t) => (c.label + " " + c.g).toLowerCase().includes(t))).slice(0, 40);
    palSel = Math.min(palSel, Math.max(0, palItems.length - 1));
    palList.replaceChildren(); let last = null;
    if (!palItems.length) { palList.append(h("li", { class: "pal-empty" }, "Nothing matches. Try “warranty” or “stress”.")); return; }
    palItems.forEach((c, i) => {
      if (c.g !== last) { palList.append(h("li", { class: "pal-group", role: "presentation" }, c.g)); last = c.g; }
      palList.append(h("li", { class: "pal-item", role: "option", id: `pal-${i}`, "aria-selected": String(i === palSel), onclick: () => runPalette(i) }, ico(c.icon), h("span", {}, c.label), c.hint ? h("small", {}, c.hint) : null));
    });
    palIn.setAttribute("aria-activedescendant", `pal-${palSel}`);
  }
  function runPalette(i) { const c = palItems[i]; if (!c) return; pal.close(); c.run(); }
  function openPalette() { closeNav(); palIn.value = ""; palSel = 0; drawPalette(); if (!pal.open) pal.showModal(); palIn.focus(); }
  palIn.addEventListener("input", () => { palSel = 0; drawPalette(); });
  palIn.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); palSel = (palSel + (e.key === "ArrowDown" ? 1 : -1) + palItems.length) % (palItems.length || 1); drawPalette(); const el = $(`#pal-${palSel}`, palList); if (el) el.scrollIntoView({ block: "nearest" }); }
    if (e.key === "Enter") { e.preventDefault(); runPalette(palSel); }
  });
  pal.addEventListener("click", (e) => { if (e.target === pal) pal.close(); });
  $("#palette-btn").addEventListener("click", openPalette);
  const helpDlg = $("#helpdialog");
  function openHelp() {
    const rows = [["Ctrl K or ⌘ K", "Open the command palette"], ["/", "Jump to the question or text box"], ["↑ and ↓", "Move between sentences in a result"], ["?", "Show this list"], ["Esc", "Close a dialog or the menu"]];
    $("#help-body").replaceChildren(h("table", { class: "keys" }, h("tbody", {}, rows.map(([k, d]) => h("tr", {}, h("td", {}, h("kbd", {}, k)), h("td", {}, d))))));
    helpDlg.showModal();
  }
  $("#help-btn").addEventListener("click", openHelp);
  document.addEventListener("keydown", (e) => {
    const tag = (e.target.tagName || "").toLowerCase(); const typing = tag === "input" || tag === "textarea" || tag === "select";
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); if (pal.open) pal.close(); else openPalette(); }
    else if (e.key === "?" && !typing && !e.ctrlKey && !e.metaKey) { e.preventDefault(); openHelp(); }
  });


  // ------------------------------------------------------------------ overview stats
  function renderStats() {
    const box = $("#stats"); if (!box) return;
    const runs = S.history.length;
    let n = 0, flagged = 0;
    S.history.forEach((e) => { const sm = summarize(e.rec); n += sm.n; flagged += sm.flagged; });
    const cells = [["Documents", S.docs.length], ["Searchable passages", S.chunks.length], ["Checks run", runs], ["Claims flagged", n ? `${Math.round((flagged / n) * 100)}%` : "None yet"]];
    box.replaceChildren(...cells.map(([l, v]) => h("div", { class: "stat" }, h("span", {}, l), h("b", {}, String(v)))));
  }

  // ------------------------------------------------------------------ about
  function renderAbout() {
    const out = $("#about-out"); if (out.childElementCount) return;
    const sec = (title, ...kids) => h("section", {}, h("h2", {}, title), ...kids);
    out.append(
      sec("The pipeline", h("ol", { class: "flow" },
        [["Retrieve", "BM25 ranks passages from your documents. Documents are split on headings and sentence boundaries."],
         ["Write", "A model drafts an answer and cites passage ids like [c3]. A strict style allows sources only."],
         ["Split", "Each sentence is cut into atomic claims, so a half-true sentence can't hide behind its true half."],
         ["Verify", "Every claim is scored against short windows of evidence by one or more pluggable verifiers."],
         ["Fix", "Flag the claim, drop its sentence, or ask the model to rewrite and check again."]]
          .map(([a, b]) => h("li", {}, h("b", {}, a), h("span", {}, b))))),
      sec("Verifiers", h("div", { class: "table-wrap" }, h("table", { class: "data" },
        h("thead", {}, h("tr", {}, ["Verifier", "How it decides", "Needs", "Weak spot"].map((x) => h("th", {}, x)))),
        h("tbody", {}, [
          ["Lexical match", "Word and number overlap with the best evidence window, plus a negation check.", "Nothing. Runs in this page.", "Paraphrase and swapped names."],
          ["Embedding similarity", "Sentence-embedding similarity between claim and evidence window.", "sentence-transformers", "Similar wording with a different fact."],
          ["NLI model", "Does the evidence window entail the claim? Also detects contradiction.", "torch and transformers", "Evidence spread over distant sentences."],
          ["LLM judge", "A language model rules on the claim given the retrieved evidence.", "An API key (free tiers work)", "Cost, latency and judge bias."],
        ].map((r) => h("tr", {}, r.map((x, i) => h(i ? "td" : "th", {}, x)))))))),
      h("div", { class: "cols" },
        h("section", { class: "card" }, h("h2", {}, "Design decisions"), h("ul", {},
          h("li", {}, "One record format for the browser and Python engines, so a single UI renders both."),
          h("li", {}, "Claims are verified in windows of up to three sentences, so evidence must be local, not somewhere in a long passage."),
          h("li", {}, "Citations are audited too: a supported claim that cites the wrong passage is reported."),
          h("li", {}, "The stress test plants known errors, which gives ground truth without hand labelling."))),
        h("section", { class: "card" }, h("h2", {}, "Known limits"), h("ul", {},
          h("li", {}, "The lexical checker misses a sentence that reuses the right words with a wrong subject. The stress test shows this."),
          h("li", {}, "Number handling is token based. “Ten” and “10” differ, and units aren't converted."),
          h("li", {}, "Removal works per sentence, so one bad claim removes its whole sentence."),
          h("li", {}, "The browser demo writer isn't a language model. Use the Python server for real answers."))),
        h("section", { class: "card" }, h("h2", {}, "Built with"), h("ul", {},
          h("li", {}, "Python 3 standard-library server and pipeline, no framework"),
          h("li", {}, "Vanilla JavaScript front end, no build step, one self-contained HTML file"),
          h("li", {}, "pytest suite covering the pipeline, verifiers, API and static serving"),
          h("li", {}, "Optional models: Claude, Groq, Gemini, OpenRouter, sentence-transformers, NLI")))),
      sec("Run it", h("pre", { class: "code" }, "python server.py              # the app at http://127.0.0.1:8000\npython scripts/build_web.py  # rebuild the single-file site in dist/\npython -m pytest             # run the tests")));
  }

  // ------------------------------------------------------------------ router + keys
  const NAV = [
    ["ask", "Ask", "chat"], ["audit", "Check text", "filecheck"], ["library", "Documents", "folder"], ["benchmark", "Stress test", "pulse"],
    ["history", "History", "clock"], null, ["about", "How it works", "info"],
  ];
  const VIEWS = NAV.filter(Boolean).map((n) => n[0]);
  const TITLES = Object.fromEntries(NAV.filter(Boolean).map((n) => [n[0], n[1]]));
  function renderNav() {
    const nav = $("#nav"); nav.replaceChildren();
    NAV.forEach((n) => {
      if (!n) { nav.append(h("div", { class: "nav-sep", role: "separator" })); return; }
      const a = h("a", { href: "#" + n[0], "data-view": n[0], title: n[1], onclick: () => closeNav() }, ico(n[2]), h("span", {}, n[1]));
      if (n[0] === "history" && S.history.length) a.append(h("span", { class: "count" }, String(S.history.length)));
      if (n[0] === "library") a.append(h("span", { class: "count" }, String(S.docs.length)));
      nav.append(a);
    });
    $$("#nav a").forEach((a) => { if (a.dataset.view === current) a.setAttribute("aria-current", "page"); });
  }
  let current = null;
  function openNav() { document.body.classList.add("nav-open"); $("#scrim").hidden = false; $("#menu-btn").setAttribute("aria-expanded", "true"); }
  function closeNav() { document.body.classList.remove("nav-open"); $("#scrim").hidden = true; $("#menu-btn").setAttribute("aria-expanded", "false"); }
  $("#menu-btn").append(ico("menu"));
  $("#menu-btn").addEventListener("click", () => (document.body.classList.contains("nav-open") ? closeNav() : openNav()));
  $("#scrim").addEventListener("click", closeNav);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeNav(); });
  function setView(v) {
    if (!VIEWS.includes(v)) v = "ask";
    if (location.hash !== "#" + v) { history.replaceState(null, "", "#" + v); }
    show(v);
  }
  function show(v) {
    current = v;
    $$(".view").forEach((s) => { s.hidden = s.dataset.view !== v; });
    $("#crumb").textContent = TITLES[v];
    document.title = v === "ask" ? "TruthLens" : `${TITLES[v]} · TruthLens`;
    if (v === "history") renderHistory();
    if (v === "library") renderLibrary();
    if (v === "audit") renderAuditUi();
    if (v === "benchmark") renderBench();
    if (v === "about") renderAbout();
    if (v === "ask") { renderStats(); renderRecent(); }
    renderNav(); closeNav();
    window.scrollTo({ top: 0 });
  }
  window.addEventListener("hashchange", () => show(VIEWS.includes(location.hash.slice(1)) ? location.hash.slice(1) : "ask"));
  document.addEventListener("keydown", (e) => {
    const tag = (e.target.tagName || "").toLowerCase();
    if (e.key === "/" && tag !== "input" && tag !== "textarea" && !e.metaKey && !e.ctrlKey) { const i = current === "audit" ? $("#audit-text") : $("#question"); if (i) { e.preventDefault(); i.focus(); } }
    if ((e.key === "ArrowDown" || e.key === "ArrowUp") && (current === "ask" || current === "audit") && tag !== "input" && tag !== "textarea" && tag !== "select" && !dlg.open) {
      const r = current === "ask" ? askResult : auditResult;
      if (r.rec && r.all) { e.preventDefault(); r.move(e.key === "ArrowDown" ? 1 : -1); }
    }
  });

  // ------------------------------------------------------------------ boot
  async function detectServer() {
    if (location.protocol === "file:") return;
    try {
      const ctl = new AbortController(); const to = setTimeout(() => ctl.abort(), 1500);
      const r = await fetch("/api/meta", { signal: ctl.signal }); clearTimeout(to);
      if (!r.ok || !(r.headers.get("content-type") || "").includes("json")) return;
      S.meta = await r.json(); S.server = true;
      if (store.get("settings", null) == null) S.settings.engine = S.meta.live_ready ? "server" : "browser";
      if (!S.meta.verifiers.find((v) => v.name === S.settings.act && v.available)) { S.settings.verifiers = ["lexical"]; S.settings.act = "lexical"; }
    } catch (_) { /* no server: the browser engine is enough */ }
  }

  renderChips(); renderAuditUi(); renderLibrary();
  const start = VIEWS.includes(location.hash.slice(1)) ? location.hash.slice(1) : "ask";
  show(start);
  $("#question").value = "What do the H10 and the installation cost?";
  updateEngineChip();
  detectServer().then(updateEngineChip);
  // open with a finished example so the page shows what it does
  if (start === "ask") { runAsk($("#question").value).then((rec) => askResult.show(rec, { animate: true })).catch(() => {}); }
  else if (start === "audit") { $("#audit-text").value = AUDIT_EXAMPLES[2][1]; runAudit($("#audit-text").value).then((r) => auditResult.show(r, { animate: false })).catch(() => {}); }
  window.__truthlens = { S, askResult, auditResult, E };
})();
