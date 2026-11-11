from __future__ import annotations

import os
import re

from .claims import split_sentences
from .schema import Chunk


def _words(s: str) -> int:
    return len(s.split())


def _pieces(para: str, max_words: int) -> list:
    """Split an oversized paragraph on sentence boundaries."""
    if _words(para) <= max_words:
        return [para]
    out, cur = [], []
    for s in split_sentences(para):
        if cur and _words(" ".join(cur + [s])) > max_words:
            out.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        out.append(" ".join(cur))
    return out


def chunk_document(doc: str, text: str, max_words: int = 110) -> list:
    """Return [(doc, chunk_text)] with headings kept as context for the paragraphs under them."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    blocks, heading = [], ""
    for p in paras:
        if p.startswith("#") and "\n" not in p:
            heading = p.lstrip("# ").strip()
            continue
        for piece in _pieces(p, max_words):
            blocks.append((heading, piece))

    chunks, cur_h, cur = [], None, []
    for h, piece in blocks:
        if cur and (h != cur_h or _words(" ".join(cur + [piece])) > max_words):
            chunks.append((cur_h, cur))
            cur = []
        cur_h = h
        cur.append(piece)
    if cur:
        chunks.append((cur_h, cur))

    out = []
    for h, parts in chunks:
        body = " ".join(parts)
        out.append((doc, f"{h}\n{body}" if h else body))
    return out


def load_corpus(path: str, max_words: int = 110) -> list:
    """Load every .md/.txt under `path` into chunks with stable ids (c0, c1, ...).

    Ids depend on file order (sorted) and content, so keep the corpus fixed while you label.
    """
    files = []
    for root, _, names in os.walk(path):
        for n in names:
            if n.lower().endswith((".md", ".txt")):
                files.append(os.path.join(root, n))
    files.sort()
    chunks = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            text = fh.read()
        for doc, body in chunk_document(os.path.relpath(f, path), text, max_words):
            chunks.append(Chunk(id=f"c{len(chunks)}", doc=doc, text=body))
    if not chunks:
        raise FileNotFoundError(f"no .md/.txt files found under {path!r}")
    return chunks
