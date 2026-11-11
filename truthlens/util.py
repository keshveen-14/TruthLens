from __future__ import annotations

import hashlib
import json
import re

STOPWORDS = frozenset(
    """a an the of to in on at for from by with and or but if then than that this these those is are
    was were be been being it its as into onto over under about can could will would should may might
    must do does did has have had not no also both each any all such per via up out which who whom whose
    what when where how there their they them he she we you your our i me my his her""".split()
)

# product codes ("h10", "ip55") stay one token; plain numbers keep decimals / thousands separators
_TOKEN_RE = re.compile(r"[a-z]+\d+|\d+(?:[.,]\d+)*|[a-z]+")


def stem(t: str) -> str:
    """Very light stemmer; good enough for an overlap baseline."""
    if any(ch.isdigit() for ch in t):
        return t
    if len(t) > 4 and t.endswith("ies"):
        t = t[:-3] + "y"
    elif len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us")):
        t = t[:-1]
    for suf in ("ing", "ed"):
        if len(t) > 5 and t.endswith(suf):
            t = t[: -len(suf)]
            break
    if len(t) > 4 and t.endswith("e"):
        t = t[:-1]
    return t


def tokenize(text: str, drop_stop: bool = True, do_stem: bool = True) -> list:
    toks = []
    for m in _TOKEN_RE.findall(text.lower()):
        if m[0].isdigit():
            toks.append(m.replace(",", ""))  # 6,000 -> 6000 ; 6,40,000 -> 640000
            continue
        if drop_stop and m in STOPWORDS:
            continue
        toks.append(stem(m) if do_stem else m)
    return toks


def is_number(tok: str) -> bool:
    return tok[:1].isdigit()


def assign_split(item_id: str, dev_frac: float = 0.3) -> str:
    """Deterministic dev/test split from an id, so it never changes between runs."""
    h = int(hashlib.md5(item_id.encode()).hexdigest(), 16) % 1000
    return "dev" if h < dev_frac * 1000 else "test"


def extract_json(text: str):
    """Parse a JSON object/array out of an LLM reply (tolerates prose and code fences)."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    spans = []
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = text.find(opener), text.rfind(closer)
        if i != -1 and j > i:
            spans.append((i, j))
    for i, j in sorted(spans):
        try:
            return json.loads(text[i : j + 1])
        except Exception:
            continue
    return None
