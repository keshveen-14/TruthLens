"""Claim verifiers. Every verifier returns a Verdict with a `support_score` in [0, 1]
(higher = better supported). A claim is flagged when its label is not "supported"; the
evaluation code can also re-threshold `support_score` to trace precision/recall curves.

  lexical    token-overlap + number + negation check (cheap baseline, no dependencies)
  embedding  cosine similarity to evidence sentence windows (baseline; needs sentence-transformers)
  nli        DeBERTa-v3 NLI model over evidence windows (needs transformers + torch)
  llm_judge  strict LLM fact-checker with a supported / unsupported / contradicted rubric
"""

from __future__ import annotations

import re

from .claims import split_sentences
from .generate import format_sources
from .schema import CONTRADICTED, SUPPORTED, UNSUPPORTED, Chunk, Claim, Verdict
from .util import extract_json, is_number, tokenize

NEG_RE = re.compile(
    r"\b(?:not|no|never|cannot|without|neither|nor|none|unable)\b|n't", re.I
)


def sentence_windows(text: str, max_len: int = 3) -> list:
    """Sliding windows of 1..max_len consecutive sentences: evidence must be local, not 'somewhere in the chunk'."""
    sents = split_sentences(text) or [text]
    wins = []
    for n in range(1, max_len + 1):
        for i in range(0, len(sents) - n + 1):
            wins.append(" ".join(sents[i : i + n]))
    return wins


def _ordered(claim: Claim, chunks: list) -> list:
    """Cited chunks first, then every other retrieved chunk (fallback for mis-citations)."""
    cited = [c for c in chunks if c.id in claim.cited_ids]
    rest = [c for c in chunks if c.id not in claim.cited_ids]
    return cited + rest


class Verifier:
    name = "base"

    def verify(self, claim: Claim, chunks: list) -> Verdict:
        raise NotImplementedError

    def verify_all(self, claims: list, chunks: list) -> list:
        return [self.verify(c, chunks) for c in claims]


class LexicalVerifier(Verifier):
    name = "lexical"

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def verify(self, claim, chunks):
        ctoks = tokenize(claim.text)
        if not ctoks:
            return Verdict(claim.id, self.name, SUPPORTED, 1.0, rationale="nothing checkable")
        cset = set(ctoks)
        cnums = [t for t in cset if is_number(t)]
        claim_neg = bool(NEG_RE.search(claim.text))

        # best window that agrees with the claim's polarity, and best window that matches the
        # claim's words but flips its negation ("has a modem" vs "has no modem")
        best_ok = (-1.0, None, "")
        best_con = (-1.0, None, "")
        for ch in _ordered(claim, chunks):
            for w in sentence_windows(ch.text):
                wt = set(tokenize(w))
                cov = len(cset & wt) / len(cset)
                score = cov if all(n in wt for n in cnums) else cov * 0.5
                if score >= self.threshold and claim_neg != bool(NEG_RE.search(w)):
                    if score > best_con[0]:
                        best_con = (score, ch, w)
                elif score > best_ok[0]:
                    best_ok = (score, ch, w)

        if best_ok[0] >= self.threshold:
            label, (score, ch, win) = SUPPORTED, best_ok
        elif best_con[1] is not None:
            label, (_, ch, win) = CONTRADICTED, best_con
            score = min(max(best_ok[0], 0.0), 0.3)  # below any sensible threshold
        else:
            label, (score, ch, win) = UNSUPPORTED, best_ok
        cited_ok = None
        if ch is not None and claim.cited_ids:
            cited_ok = ch.id in claim.cited_ids
        return Verdict(
            claim.id, self.name, label, round(max(score, 0.0), 4),
            evidence_id=ch.id if ch is not None else None, cited_ok=cited_ok,
            rationale=f"best window: {win[:160]}",
        )


class EmbeddingVerifier(Verifier):
    name = "embedding"

    def __init__(self, threshold: float = 0.6, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy

        self.threshold = threshold
        self.model = SentenceTransformer(model)

    def verify(self, claim, chunks):
        cands = [(ch, w) for ch in _ordered(claim, chunks) for w in sentence_windows(ch.text, 2)]
        embs = self.model.encode([claim.text] + [w for _, w in cands], normalize_embeddings=True)
        sims = embs[1:] @ embs[0]
        i = int(sims.argmax())
        ch = cands[i][0]
        score = float(sims[i])
        return Verdict(
            claim.id, self.name, SUPPORTED if score >= self.threshold else UNSUPPORTED, round(score, 4),
            evidence_id=ch.id, cited_ok=(ch.id in claim.cited_ids) if claim.cited_ids else None,
            rationale=f"best window: {cands[i][1][:160]}",
        )


class NLIVerifier(Verifier):
    """Entailment of the claim by sentence windows of the evidence (premise = window, hypothesis = claim)."""

    name = "nli"

    def __init__(
        self,
        model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        threshold: float = 0.5,
        contra_threshold: float = 0.5,
        device: str | None = None,
        max_windows: int = 48,
    ):
        import torch  # lazy
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.threshold, self.contra_threshold, self.max_windows = threshold, contra_threshold, max_windows
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        id2label = {int(i): l.lower() for i, l in self.model.config.id2label.items()}
        self.ent = next(i for i, l in id2label.items() if l.startswith("entail"))
        self.con = next(i for i, l in id2label.items() if l.startswith("contra"))

    def verify(self, claim, chunks):
        cands = [(ch, w) for ch in _ordered(claim, chunks) for w in sentence_windows(ch.text, 3)]
        cands = cands[: self.max_windows]
        ent, con = [], []
        with self.torch.no_grad():
            for i in range(0, len(cands), 16):
                batch = cands[i : i + 16]
                enc = self.tok(
                    [w for _, w in batch], [claim.text] * len(batch),
                    truncation=True, max_length=512, padding=True, return_tensors="pt",
                ).to(self.device)
                probs = self.torch.softmax(self.model(**enc).logits, dim=-1).cpu()
                ent += probs[:, self.ent].tolist()
                con += probs[:, self.con].tolist()
        if not ent:
            return Verdict(claim.id, self.name, UNSUPPORTED, 0.0, rationale="no evidence")
        bi = max(range(len(ent)), key=lambda j: ent[j])
        bc = max(range(len(con)), key=lambda j: con[j])
        score = ent[bi]
        if score >= self.threshold:
            label, ch, win = SUPPORTED, cands[bi][0], cands[bi][1]
        elif con[bc] >= self.contra_threshold:
            label, ch, win = CONTRADICTED, cands[bc][0], cands[bc][1]
        else:
            label, ch, win = UNSUPPORTED, cands[bi][0], cands[bi][1]
        return Verdict(
            claim.id, self.name, label, round(score, 4), evidence_id=ch.id,
            cited_ok=(ch.id in claim.cited_ids) if claim.cited_ids else None,
            rationale=f"entail={ent[bi]:.2f} contradict={con[bc]:.2f}; window: {win[:140]}",
        )


JUDGE_SYSTEM = (
    "You are a strict fact-checker. You are given SOURCES and one CLAIM. Decide whether the sources "
    "alone support the claim. Ignore your own knowledge of the world: a claim that is true in reality "
    "but not stated in the sources is NOT supported.\n"
    "Labels:\n"
    "- supported: the sources state, or directly entail, every part of the claim (numbers, units, "
    "conditions and qualifiers included).\n"
    "- contradicted: the sources state something incompatible with the claim.\n"
    "- unsupported: anything else, including claims only partly covered, or that add detail, causes, "
    "comparisons or generalisations the sources do not give.\n"
    'Reply with JSON only: {"label": "supported|unsupported|contradicted", '
    '"evidence_ids": ["c1"], "reason": "<one sentence>"}'
)
_JUDGE_SCORE = {SUPPORTED: 1.0, UNSUPPORTED: 0.0, CONTRADICTED: 0.0}
_JUDGE_ALIAS = {"partial": UNSUPPORTED, "partially_supported": UNSUPPORTED, "not_supported": UNSUPPORTED,
                "not supported": UNSUPPORTED, "entailed": SUPPORTED, "contradiction": CONTRADICTED}


class LLMJudgeVerifier(Verifier):
    name = "llm_judge"

    def __init__(self, llm, max_chunks: int = 8):
        self.llm, self.max_chunks = llm, max_chunks
        self.parse_errors = 0

    def verify(self, claim, chunks):
        ordered = _ordered(claim, chunks)[: self.max_chunks]
        user = f"SOURCES:\n{format_sources(ordered)}\n\nCLAIM: {claim.text}"
        data = extract_json(self.llm.complete(JUDGE_SYSTEM, user, task="judge", max_tokens=300))
        label = None
        if isinstance(data, dict):
            raw = str(data.get("label", "")).strip().lower()
            label = raw if raw in _JUDGE_SCORE else _JUDGE_ALIAS.get(raw)
        if label is None:  # unparseable reply: flag it rather than silently trusting the claim
            self.parse_errors += 1
            return Verdict(claim.id, self.name, UNSUPPORTED, 0.0, rationale="judge reply could not be parsed")
        ids = [str(x).lower() for x in (data.get("evidence_ids") or [])] if isinstance(data.get("evidence_ids"), list) else []
        ev = ids[0] if ids else None
        return Verdict(
            claim.id, self.name, label, _JUDGE_SCORE[label], evidence_id=ev,
            cited_ok=(any(i in claim.cited_ids for i in ids) if ids and claim.cited_ids else None),
            rationale=str(data.get("reason", ""))[:300],
        )


def build_verifier(name: str, llm=None, **kw) -> Verifier:
    if name == "lexical":
        return LexicalVerifier(**kw)
    if name == "embedding":
        return EmbeddingVerifier(**kw)
    if name == "nli":
        return NLIVerifier(**kw)
    if name in ("llm", "llm_judge"):
        if llm is None or getattr(llm, "model", "") == "mock":
            raise RuntimeError("llm_judge needs a real LLM backend: set GROQ_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY or ANTHROPIC_API_KEY")
        return LLMJudgeVerifier(llm, **kw)
    raise ValueError(f"unknown verifier {name!r} (use lexical, embedding, nli, llm_judge)")
