from __future__ import annotations

import json
import re

from .schema import Claim, Sentence
from .util import extract_json

CITE_RE = re.compile(r"\[\s*(c\d+(?:\s*[,;]\s*c\d+)*)\s*\]", re.I)
_TRAIL_CITE_RE = re.compile(r"([.!?])\s*((?:\[\s*c\d+(?:\s*[,;]\s*c\d+)*\s*\]\s*)+)", re.I)
_ABBR_RE = re.compile(
    r"\b(?:e\.g|i\.e)\.|\b(?:vs|approx|etc|dr|mr|mrs|ms|st|inc|ltd|fig)\.", re.I
)
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'“(]?[A-Z0-9])")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")

# "The sources do not say..." is an abstention, not a claim to verify.
REFUSAL_RE = re.compile(
    r"(?:\b(?:sources?|documents?|context|provided\s+(?:text|information))\s+"
    r"(?:do(?:es)?\s+not|don't|doesn't|did\s+not)\s+"
    r"(?:say|state|contain|mention|specify|provide|cover|include|address|have)"
    r"|\b(?:i|we)\s+(?:could\s+not|couldn't|cannot|can't|am\s+unable\s+to)\s+find"
    r"|\bnot\s+enough\s+information\b"
    r"|\bno\s+information\s+(?:is\s+)?(?:available|provided|given)\b)",
    re.I,
)


# ---------------------------------------------------------------- sentences

def split_sentences(text: str) -> list:
    """Split text into sentences; keeps decimals ("3.5 kWh") and abbreviations ("e.g.") intact.

    Citations are normalised to sit before the final punctuation ("... kWh [c3].").
    """
    out = []
    for line in text.replace("\r", "").split("\n"):
        line = _BULLET_RE.sub("", line.strip())
        if not line:
            continue
        line = _TRAIL_CITE_RE.sub(lambda m: " " + m.group(2).strip() + m.group(1) + " ", line)
        line = re.sub(r"\s{2,}", " ", line).strip()
        masked = _ABBR_RE.sub(lambda m: m.group(0).replace(".", "․"), line)
        for part in _SPLIT_RE.split(masked):
            part = part.replace("․", ".").strip()
            if part:
                out.append(part)
    return out


def parse_sentence(idx: int, raw: str) -> Sentence:
    ids = []
    for m in CITE_RE.finditer(raw):
        for x in re.split(r"[,;]", m.group(1)):
            x = x.strip().lower()
            if x and x not in ids:
                ids.append(x)
    clean = CITE_RE.sub("", raw)
    clean = re.sub(r"\s+([.!?,;:])", r"\1", clean)
    clean = re.sub(r"\s{2,}", " ", clean).strip()
    return Sentence(idx=idx, raw=raw, text=clean, cited_ids=ids, is_refusal=bool(REFUSAL_RE.search(clean)))


def parse_answer(raw: str) -> list:
    raw = raw.replace("**", "").replace("__", "")
    return [parse_sentence(i, s) for i, s in enumerate(split_sentences(raw))]


def render_answer(sentences: list) -> str:
    return " ".join(s.raw for s in sentences)


# ------------------------------------------------------------ atomic claims

_CLAUSE_SPLIT = re.compile(r";\s+|,\s+(?:and|but|while|whereas|although)\s+", re.I)
_VERB_START = {
    "is", "are", "was", "were", "has", "have", "had", "can", "could", "will", "would", "may", "might",
    "must", "does", "do", "did", "also", "supports", "includes", "requires", "uses", "offers",
    "provides", "comes", "covers", "takes", "works", "allows", "needs", "weighs", "ships", "lasts",
    "begins", "starts", "operates", "stores", "delivers", "measures", "charges", "discharges",
    "connects",
}
_INDEP_START = {
    "it", "they", "the", "this", "these", "its", "their", "a", "an", "both", "each", "all", "there",
    "we", "he", "she",
}


def _tidy(p: str) -> str:
    p = p.strip().rstrip(" .;,")
    return (p[:1].upper() + p[1:] + ".") if p else p


def atomic_claims_rule(text: str) -> list:
    """Conservative rule-based split of compound sentences.

    Splits only on ';' and ', and/but/while/whereas', and only when each part is a clause
    (has its own subject, or a verb whose subject can be carried over). If unsure it returns
    the sentence unchanged. Use `--claims llm` for better decomposition.
    """
    text = text.strip()
    parts = [p.strip() for p in _CLAUSE_SPLIT.split(text) if p and p.strip()]
    if len(parts) < 2 or any(len(p.split()) < 3 for p in parts):
        return [text]

    first_toks = parts[0].split()
    subject = None
    for i, t in enumerate(first_toks):
        if t.lower().strip(",.;:") in _VERB_START:
            subject = " ".join(first_toks[:i]) or None
            break

    out = [_tidy(parts[0])]
    for p in parts[1:]:
        toks = p.split()
        w0 = toks[0].lower().strip(",.;:")
        if w0 in ("it", "they") and subject:
            new = subject + " " + " ".join(toks[1:])
        elif w0 in _VERB_START:
            if not subject:
                return [text]
            new = subject + " " + p
        elif w0 in _INDEP_START or toks[0][:1].isupper():
            new = p
        else:
            return [text]
        out.append(_tidy(new))
    return out


DECOMP_SYSTEM = (
    "Split the sentence into the smallest set of standalone factual claims. Each claim must be "
    "understandable on its own (replace pronouns with the noun they refer to), must not add any "
    "information that is not in the sentence, and must keep numbers and units exactly. "
    'Reply with JSON only: a list of strings, e.g. ["claim one.", "claim two."]. '
    "If the sentence is already a single claim, return it as a one-item list."
)


def atomic_claims_llm(text: str, llm) -> list:
    reply = llm.complete(DECOMP_SYSTEM, f"Sentence: {text}", task="decompose", max_tokens=400)
    data = extract_json(reply)
    if isinstance(data, list):
        items = [str(x).strip() for x in data if isinstance(x, (str, int, float)) and str(x).strip()]
        if items:
            return items
    return atomic_claims_rule(text)


def extract_claims(sentences: list, mode: str = "rule", llm=None, prefix: str = "") -> list:
    """Sentences -> atomic claims, each mapped back to its sentence and that sentence's citations."""
    claims = []
    for s in sentences:
        if s.is_refusal or s.text.endswith("?") or len(s.text.split()) < 3:
            continue
        texts = atomic_claims_llm(s.text, llm) if (mode == "llm" and llm is not None) else atomic_claims_rule(s.text)
        for j, t in enumerate(texts):
            claims.append(
                Claim(id=f"{prefix}s{s.idx}.{j}", text=t, sentence_idx=s.idx, cited_ids=list(s.cited_ids))
            )
    return claims


__all__ = [
    "split_sentences", "parse_sentence", "parse_answer", "render_answer", "atomic_claims_rule",
    "atomic_claims_llm", "extract_claims", "CITE_RE", "REFUSAL_RE", "json",
]
