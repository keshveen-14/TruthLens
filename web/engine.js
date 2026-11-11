/* TruthLens engine, in the browser. A port of the Python pipeline's offline path:
   corpus chunking, BM25 retrieval, sentence + claim splitting, the lexical verifier, and a demo generator.
   Records have the same shape as the Python server's, so the UI renders either one. */
(function (root) {
  "use strict";

  // ------------------------------------------------------------------ text utils
  const STOP = new Set(("a an the of to in on at for from by with and or but if then than that this these those is are " +
    "was were be been being it its as into onto over under about can could will would should may might " +
    "must do does did has have had not no also both each any all such per via up out which who whom whose " +
    "what when where how there their they them he she we you your our i me my his her").split(/\s+/));
  const TOKEN_RE = /[a-z]+\d+|\d+(?:[.,]\d+)*|[a-z]+/g;

  function stem(t) {
    if (/\d/.test(t)) return t;
    if (t.length > 4 && t.endsWith("ies")) t = t.slice(0, -3) + "y";
    else if (t.length > 3 && t.endsWith("s") && !t.endsWith("ss") && !t.endsWith("us")) t = t.slice(0, -1);
    for (const suf of ["ing", "ed"]) {
      if (t.length > 5 && t.endsWith(suf)) { t = t.slice(0, -suf.length); break; }
    }
    if (t.length > 4 && t.endsWith("e")) t = t.slice(0, -1);
    return t;
  }
  function tokenize(text) {
    const out = [];
    for (const m of text.toLowerCase().match(TOKEN_RE) || []) {
      if (/^\d/.test(m)) { out.push(m.replace(/,/g, "")); continue; }
      if (STOP.has(m)) continue;
      out.push(stem(m));
    }
    return out;
  }
  const isNumber = (t) => /^\d/.test(t);
  const words = (s) => s.split(/\s+/).filter(Boolean).length;

  // ------------------------------------------------------------------ sentences
  const CITE_RE = /\[\s*(c\d+(?:\s*[,;]\s*c\d+)*)\s*\]/gi;
  const TRAIL_CITE_RE = /([.!?])\s*((?:\[\s*c\d+(?:\s*[,;]\s*c\d+)*\s*\]\s*)+)/gi;
  const ABBR_RE = /\b(?:e\.g|i\.e)\.|\b(?:vs|approx|etc|dr|mr|mrs|ms|st|inc|ltd|fig)\./gi;
  const SPLIT_RE = /(?<=[.!?])\s+(?=["'“(]?[A-Z0-9])/;
  const BULLET_RE = /^\s*(?:[-*•]|\d+[.)])\s+/;
  const REFUSAL_RE = new RegExp(
    "(?:\\b(?:sources?|documents?|context|provided\\s+(?:text|information))\\s+" +
    "(?:do(?:es)?\\s+not|don't|doesn't|did\\s+not)\\s+" +
    "(?:say|state|contain|mention|specify|provide|cover|include|address|have)" +
    "|\\b(?:i|we)\\s+(?:could\\s+not|couldn't|cannot|can't|am\\s+unable\\s+to)\\s+find" +
    "|\\bnot\\s+enough\\s+information\\b" +
    "|\\bno\\s+information\\s+(?:is\\s+)?(?:available|provided|given)\\b)", "i");

  function splitSentences(text) {
    const out = [];
    for (let line of text.replace(/\r/g, "").split("\n")) {
      line = line.trim().replace(BULLET_RE, "");
      if (!line) continue;
      line = line.replace(TRAIL_CITE_RE, (_, p, cites) => " " + cites.trim() + p + " ");
      line = line.replace(/\s{2,}/g, " ").trim();
      const masked = line.replace(ABBR_RE, (m) => m.replace(/\./g, "․"));
      for (let part of masked.split(SPLIT_RE)) {
        part = part.replace(/․/g, ".").trim();
        if (part) out.push(part);
      }
    }
    return out;
  }

  function parseSentence(idx, raw) {
    const ids = [];
    for (const m of raw.matchAll(CITE_RE)) {
      for (let x of m[1].split(/[,;]/)) {
        x = x.trim().toLowerCase();
        if (x && !ids.includes(x)) ids.push(x);
      }
    }
    let clean = raw.replace(CITE_RE, "").replace(/\s+([.!?,;:])/g, "$1").replace(/\s{2,}/g, " ").trim();
    return { idx, raw, text: clean, cited_ids: ids, is_refusal: REFUSAL_RE.test(clean) };
  }
  const parseAnswer = (raw) =>
    splitSentences(raw.replace(/\*\*/g, "").replace(/__/g, "")).map((s, i) => parseSentence(i, s));
  const renderAnswer = (sents) => sents.map((s) => s.raw).join(" ");

  // ------------------------------------------------------------------ atomic claims
  const CLAUSE_SPLIT = /;\s+|,\s+(?:and|but|while|whereas|although)\s+/i;
  const VERB_START = new Set(("is are was were has have had can could will would may might must does do did also supports " +
    "includes requires uses offers provides comes covers takes works allows needs weighs ships lasts begins starts " +
    "operates stores delivers measures charges discharges connects").split(" "));
  const INDEP_START = new Set("it they the this these its their a an both each all there we he she".split(" "));
  const strip = (w) => w.toLowerCase().replace(/^[,.;:]+|[,.;:]+$/g, "");
  function tidy(p) {
    p = p.trim().replace(/[ .;,]+$/, "");
    return p ? p[0].toUpperCase() + p.slice(1) + "." : p;
  }
  function atomicClaims(text) {
    text = text.trim();
    const parts = text.split(CLAUSE_SPLIT).map((p) => p && p.trim()).filter(Boolean);
    if (parts.length < 2 || parts.some((p) => words(p) < 3)) return [text];
    const first = parts[0].split(/\s+/);
    let subject = null;
    for (let i = 0; i < first.length; i++) {
      if (VERB_START.has(strip(first[i]))) { subject = first.slice(0, i).join(" ") || null; break; }
    }
    const out = [tidy(parts[0])];
    for (const p of parts.slice(1)) {
      const toks = p.split(/\s+/);
      const w0 = strip(toks[0]);
      let next;
      if ((w0 === "it" || w0 === "they") && subject) next = subject + " " + toks.slice(1).join(" ");
      else if (VERB_START.has(w0)) { if (!subject) return [text]; next = subject + " " + p; }
      else if (INDEP_START.has(w0) || /^[A-Z]/.test(toks[0])) next = p;
      else return [text];
      out.push(tidy(next));
    }
    return out;
  }
  function extractClaims(sentences, prefix = "") {
    const claims = [];
    for (const s of sentences) {
      if (s.is_refusal || s.text.endsWith("?") || words(s.text) < 3) continue;
      atomicClaims(s.text).forEach((t, j) =>
        claims.push({ id: `${prefix}s${s.idx}.${j}`, text: t, sentence_idx: s.idx, cited_ids: [...s.cited_ids] }));
    }
    return claims;
  }

  // ------------------------------------------------------------------ corpus
  function pieces(para, maxWords) {
    if (words(para) <= maxWords) return [para];
    const out = []; let cur = [];
    for (const s of splitSentences(para)) {
      if (cur.length && words([...cur, s].join(" ")) > maxWords) { out.push(cur.join(" ")); cur = []; }
      cur.push(s);
    }
    if (cur.length) out.push(cur.join(" "));
    return out;
  }
  function chunkDocument(doc, text, maxWords = 110) {
    const paras = text.split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean);
    const blocks = []; let heading = "";
    for (const p of paras) {
      if (p.startsWith("#") && !p.includes("\n")) { heading = p.replace(/^#+\s*/, "").trim(); continue; }
      for (const piece of pieces(p, maxWords)) blocks.push([heading, piece]);
    }
    const chunks = []; let curH = null; let cur = [];
    for (const [h, piece] of blocks) {
      if (cur.length && (h !== curH || words([...cur, piece].join(" ")) > maxWords)) { chunks.push([curH, cur]); cur = []; }
      curH = h; cur.push(piece);
    }
    if (cur.length) chunks.push([curH, cur]);
    return chunks.map(([h, parts]) => ({ doc, heading: h || "", text: h ? `${h}\n${parts.join(" ")}` : parts.join(" ") }));
  }
  /** docs: [{name, text}] -> chunks with stable ids c0, c1, ... (documents sorted by name, like the Python loader). */
  function loadCorpus(docs) {
    const chunks = [];
    [...docs].sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0)).forEach((d) => {
      for (const c of chunkDocument(d.name, d.text)) chunks.push({ id: `c${chunks.length}`, ...c });
    });
    return chunks;
  }

  // ------------------------------------------------------------------ retrieval (BM25)
  class BM25 {
    constructor(chunks, k1 = 1.5, b = 0.75) {
      this.k1 = k1; this.b = b;
      this.docs = chunks.map((c) => { const m = new Map(); tokenize(c.text).forEach((t) => m.set(t, (m.get(t) || 0) + 1)); return m; });
      this.lens = this.docs.map((d) => { let n = 0; d.forEach((v) => (n += v)); return n; });
      this.avg = this.lens.length ? this.lens.reduce((a, b2) => a + b2, 0) / this.lens.length : 0;
      const df = new Map();
      this.docs.forEach((d) => d.forEach((_, t) => df.set(t, (df.get(t) || 0) + 1)));
      const n = this.docs.length;
      this.n = n;
      this.idf = new Map();
      df.forEach((f, t) => this.idf.set(t, Math.log(1 + (n - f + 0.5) / (f + 0.5))));
      this.maxIdf = Math.log(1 + (n + 0.5) / 0.5);
    }
    scores(query) {
      const q = tokenize(query);
      return this.docs.map((d, i) => {
        let s = 0;
        for (const t of q) {
          const f = d.get(t);
          if (f) s += this.idf.get(t) * f * (this.k1 + 1) / (f + this.k1 * (1 - this.b + this.b * this.lens[i] / (this.avg || 1)));
        }
        return s;
      });
    }
  }
  class Retriever {
    constructor(chunks) { this.chunks = chunks; this.bm25 = new BM25(chunks); }
    /** Top-k chunks with their BM25 score and rank. */
    retrieve(query, k = 5) {
      const sc = this.bm25.scores(query);
      const order = sc.map((_, i) => i).sort((a, b) => sc[b] - sc[a] || a - b);
      return order.slice(0, k).map((i, r) => ({ ...this.chunks[i], score: Math.round(sc[i] * 1000) / 1000, rank: r + 1 }));
    }
  }

  // ------------------------------------------------------------------ lexical verifier
  const NEG_RE = /\b(?:not|no|never|cannot|without|neither|nor|none|unable)\b|n't/i;
  const bodyOf = (text) => (text.includes("\n") ? text.split("\n").slice(1).join(" ") : text);

  function windowsOf(chunk) {
    if (chunk._wins) return chunk._wins;
    const sents = splitSentences(chunk.text);
    const list = sents.length ? sents : [chunk.text];
    const wins = [];
    for (let n = 1; n <= 3; n++) {
      for (let i = 0; i + n <= list.length; i++) {
        const text = list.slice(i, i + n).join(" ");
        wins.push({ text, set: new Set(tokenize(text)), neg: NEG_RE.test(text) });
      }
    }
    Object.defineProperty(chunk, "_wins", { value: wins, enumerable: false });
    return wins;
  }

  function verifyClaim(claim, chunks, threshold) {
    const ctoks = tokenize(claim.text);
    if (!ctoks.length) return { claim_id: claim.id, verifier: "lexical", label: "supported", support_score: 1, evidence_id: null, cited_ok: null, rationale: "nothing checkable", window: "" };
    const cset = [...new Set(ctoks)];
    const cnums = cset.filter(isNumber);
    const claimNeg = NEG_RE.test(claim.text);
    let ok = { s: -1, ch: null, w: "" };
    let con = { s: -1, ch: null, w: "" };
    const cited = chunks.filter((c) => claim.cited_ids.includes(c.id));
    const rest = chunks.filter((c) => !claim.cited_ids.includes(c.id));
    for (const ch of [...cited, ...rest]) {
      for (const w of windowsOf(ch)) {
        let hit = 0;
        for (const t of cset) if (w.set.has(t)) hit++;
        const cov = hit / cset.length;
        const score = cnums.every((n) => w.set.has(n)) ? cov : cov * 0.5;
        if (score >= threshold && claimNeg !== w.neg) { if (score > con.s) con = { s: score, ch, w: w.text }; }
        else if (score > ok.s) ok = { s: score, ch, w: w.text };
      }
    }
    let label, score, ch, win;
    if (ok.s >= threshold) { label = "supported"; score = ok.s; ch = ok.ch; win = ok.w; }
    else if (con.ch) { label = "contradicted"; score = Math.min(Math.max(ok.s, 0), 0.3); ch = con.ch; win = con.w; }
    else { label = "unsupported"; score = ok.s; ch = ok.ch; win = ok.w; }
    return {
      claim_id: claim.id, verifier: "lexical", label, support_score: Math.round(Math.max(score, 0) * 10000) / 10000,
      evidence_id: ch ? ch.id : null, cited_ok: ch && claim.cited_ids.length ? claim.cited_ids.includes(ch.id) : null,
      rationale: "best window: " + win.slice(0, 160), window: win,
    };
  }
  const verifyAll = (claims, chunks, threshold = 0.7) => claims.map((c) => verifyClaim(c, chunks, threshold));

  // ------------------------------------------------------------------ demo generator
  const INJECTED_BANK = [
    "The unit also supports 5G cellular connectivity for remote monitoring.",
    "It is certified for use in marine and offshore environments.",
    "Helio ships to more than 40 countries worldwide.",
    "The battery can be upgraded to double its capacity with a firmware licence.",
    "A free 20-year warranty applies to every unit sold.",
    "Helio batteries can be installed in bedrooms and bathrooms.",
  ];
  function fnv(str) {
    let h = 0x811c9dc5;
    for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
    return h >>> 0;
  }
  /** Change the first number in a sentence so it stays fluent but is no longer true. */
  function tamper(sentence) {
    let done = false;
    const out = sentence.replace(/\d[\d,]*(?:\.\d+)?/, (m) => {
      done = true;
      const n = parseFloat(m.replace(/,/g, ""));
      const next = n >= 10 ? Math.round(n * 1.5) : n + 5;
      return m.includes(",") ? next.toLocaleString("en-US") : String(next);
    });
    return done ? out : null;
  }

  /** Extractive "model": picks the sentences that best match the question and cites their chunks.
      On about half the questions it also plants one false sentence, so there is something to catch. */
  function demoGenerate(question, chunks, index, opts = {}) {
    const q = [...new Set(tokenize(question))];
    const weight = (t) => index.idf.get(t) ?? index.maxIdf;
    const total = q.reduce((a, t) => a + weight(t), 0) || 1;
    const cands = [];
    chunks.forEach((ch, r) => {
      splitSentences(bodyOf(ch.text)).forEach((s, pos) => {
        if (!/[.!?]$/.test(s)) return;
        const st = tokenize(s);
        const near = (t) => st.some((x) => x === t || (t.length >= 4 && x.length >= 4 && x.slice(0, 4) === t.slice(0, 4) && !/\d/.test(t)));
        const hit = q.reduce((a, t) => a + (near(t) ? weight(t) : 0), 0) / total;
        cands.push({ s, ch, r, pos, score: hit + 0.04 * (chunks.length - r) / chunks.length });
      });
    });
    cands.sort((a, b) => b.score - a.score);
    const best = cands[0];
    const sentences = []; // {text, cite, injected}
    if (best && best.score >= 0.3) {
      const picked = cands.filter((c) => c.score >= Math.max(0.3, best.score * 0.6)).slice(0, 3)
        .sort((a, b) => a.r - b.r || a.pos - b.pos);
      for (const c of picked) sentences.push({ text: c.s, cite: c.ch.id, injected: false });
    }
    if (!sentences.length) {
      return { raw: "The sources do not say.", injected: [], note: "abstained" };
    }
    const h = fnv(question.trim().toLowerCase());
    if (opts.inject !== false && h % 2 === 0) {
      const mode = (h >>> 1) % 2;
      const victim = sentences[sentences.length - 1];
      const t = mode === 1 ? tamper(victim.text) : null;
      if (t) sentences.push({ text: t, cite: victim.cite, injected: true, replaces: sentences.length - 1 });
      else sentences.push({ text: INJECTED_BANK[(h >>> 2) % INJECTED_BANK.length], cite: sentences[0].cite, injected: true });
    }
    // a tampered sentence replaces its original instead of repeating it
    const final = [];
    sentences.forEach((s, i) => { if (!sentences.some((o) => o.replaces === i)) final.push(s); });
    const cited = (s) => s.text.replace(/([.!?])$/, ` [${s.cite}]$1`);
    return { raw: final.map(cited).join(" "), injected: final.map((s) => s.injected) };
  }

  // ------------------------------------------------------------------ pipeline
  const snap = (sentences, claims, verdicts) => ({ answer: renderAnswer(sentences), sentences, claims, verdicts: { lexical: verdicts } });

  /** Ask a question: retrieve, generate (demo), split, verify. */
  function ask(question, retriever, opts = {}) {
    const t0 = performance.now();
    const k = opts.k || 5, threshold = opts.threshold ?? 0.7;
    const chunks = retriever.retrieve(question, k);
    const t1 = performance.now();
    const gen = demoGenerate(question, chunks, retriever.bm25, opts);
    const t2 = performance.now();
    const sentences = parseAnswer(gen.raw);
    sentences.forEach((s, i) => { s.injected = !!(gen.injected && gen.injected[i]); });
    const claims = extractClaims(sentences);
    const t3 = performance.now();
    const verdicts = verifyAll(claims, chunks, threshold);
    const t4 = performance.now();
    return buildRecord({
      mode: "ask", question, chunks, sentences, claims, verdicts, threshold, k,
      timings: { retrieve: t1 - t0, generate: t2 - t1, split: t3 - t2, verify: t4 - t3 },
      model: "Demo generator", started: t0,
    });
  }

  /** Audit: check any pasted answer against the whole library. */
  function audit(answer, retriever, opts = {}) {
    const t0 = performance.now();
    const threshold = opts.threshold ?? 0.7;
    const sentences = parseAnswer(answer);
    const claims = extractClaims(sentences);
    const t1 = performance.now();
    const chunks = retriever.chunks;
    const verdicts = verifyAll(claims, chunks, threshold);
    const t2 = performance.now();
    return buildRecord({
      mode: "audit", question: "", chunks, sentences, claims, verdicts, threshold, k: chunks.length,
      timings: { split: t1 - t0, verify: t2 - t1 }, model: "Pasted text", started: t0,
    });
  }

  function buildRecord(o) {
    const baseline = snap(o.sentences, o.claims, o.verdicts);
    return {
      answer_id: "q", question: o.question, split: "test",
      config: { k: o.k, action: "remove", act_verifier: "lexical", verifiers: ["lexical"], llm: o.model, claim_mode: "rule", style: "strict" },
      retrieved: o.chunks.map((c) => ({ id: c.id, doc: c.doc, text: c.text, score: c.score, rank: c.rank })),
      baseline, final: removeFlagged(baseline, "lexical"), rounds: 0, abstained: false,
      ui: {
        engine: "browser", mode: o.mode, mock: o.mode === "ask", model: o.model, provider: o.mode === "ask" ? "Demo" : "Pasted",
        action: "remove", act: "lexical", thresholds: { lexical: o.threshold }, timings: o.timings,
        elapsed_ms: Math.round(Object.values(o.timings).reduce((a, b) => a + b, 0)), at: Date.now(),
      },
    };
  }

  /** The answer without sentences that contain a flagged claim. */
  function removeFlagged(snapshot, act) {
    const bad = new Set(snapshot.verdicts[act].filter((v) => v.label !== "supported").map((v) => v.claim_id));
    const badS = new Set(snapshot.claims.filter((c) => bad.has(c.id)).map((c) => c.sentence_idx));
    const sentences = snapshot.sentences.filter((s) => !badS.has(s.idx));
    const claims = snapshot.claims.filter((c) => !badS.has(c.sentence_idx));
    const ids = new Set(claims.map((c) => c.id));
    const verdicts = {};
    for (const [n, vs] of Object.entries(snapshot.verdicts)) verdicts[n] = vs.filter((v) => ids.has(v.claim_id));
    return { answer: renderAnswer(sentences), sentences, claims, verdicts };
  }

  /** Re-score an existing record at a new threshold, without regenerating anything. */
  function reverify(rec, threshold) {
    const chunks = rec.retrieved.map((c) => ({ ...c }));
    const verdicts = verifyAll(rec.baseline.claims, chunks, threshold);
    const baseline = { ...rec.baseline, verdicts: { lexical: verdicts } };
    return { ...rec, baseline, final: removeFlagged(baseline, "lexical"), ui: { ...rec.ui, thresholds: { lexical: threshold } } };
  }


  // ------------------------------------------------------------------ benchmark (planted errors)
  /** One false variant of a true sentence. Returns null when the mutation does not apply. */
  function mutate(kind, text, h) {
    if (kind === "fabricate") return INJECTED_BANK[h % INJECTED_BANK.length];
    if (kind === "number") return tamper(text);
    if (kind === "entity") {
      if (/\bH10\b/.test(text)) return text.replace(/\bH10\b/g, "H15");
      if (/\bH15\b/.test(text)) return text.replace(/\bH15\b/g, "H10");
      return null;
    }
    if (kind === "negate") {
      let t = text.replace(/\b(must|does|do|is|are|can|will) not\b/i, "$1");
      if (t !== text) return t;
      t = text.replace(/\bhas no\b/i, "has a");
      return t !== text ? t : null;
    }
    return null;
  }
  const MUTATIONS = ["fabricate", "number", "entity", "negate"];

  /** For every question: the true sentences the demo generator picks (label 0) plus planted false variants (label 1). */
  function benchBuild(questions, retriever, k = 5) {
    const items = [];
    for (const q of questions) {
      const chunks = retriever.retrieve(q.question, k);
      const gen = demoGenerate(q.question, chunks, retriever.bm25, { inject: false });
      const sents = parseAnswer(gen.raw).filter((s) => !s.is_refusal && s.cited_ids.length);
      if (!sents.length) continue;
      const base = { qid: q.id, qkind: q.kind, chunks };
      sents.forEach((s) => items.push({ ...base, text: s.text, cite: s.cited_ids, label: 0, mutation: null }));
      const h = fnv(q.id);
      for (const kind of MUTATIONS) {
        for (const s of [...sents].reverse()) {
          const t = mutate(kind, s.text, h);
          if (t) { items.push({ ...base, text: t, cite: s.cited_ids, label: 1, mutation: kind }); break; }
        }
      }
    }
    return items;
  }
  /** Flag decision for every item at each threshold. */
  function benchScore(items, thresholds) {
    const out = {};
    for (const t of thresholds) {
      out[t] = items.map((it) => {
        const claims = extractClaims([{ idx: 0, text: it.text, cited_ids: it.cite, is_refusal: false }]);
        return verifyAll(claims, it.chunks, t).some((v) => v.label !== "supported");
      });
    }
    return out;
  }

  root.TLEngine = {
    tokenize, splitSentences, parseAnswer, parseSentence, extractClaims, atomicClaims, loadCorpus, chunkDocument,
    Retriever, verifyAll, demoGenerate, ask, audit, reverify, removeFlagged, fnv, INJECTED_BANK,
    benchBuild, benchScore, MUTATIONS,
  };
})(typeof window !== "undefined" ? window : globalThis);
