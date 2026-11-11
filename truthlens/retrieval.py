from __future__ import annotations

import math
from collections import Counter

from .schema import Chunk
from .util import tokenize


class BM25Index:
    """Plain-Python BM25 (no dependencies)."""

    def __init__(self, chunks: list, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(c.text)) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list:
        q = tokenize(query)
        out = []
        for d, ln in zip(self.docs, self.lens):
            s = 0.0
            for t in q:
                f = d.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / (self.avg or 1)))
            out.append(s)
        return out


def _rank(scores: list) -> list:
    """Indices ordered best-first (stable on ties)."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


class HybridRetriever:
    """BM25, optionally fused with dense embeddings (RRF) and re-ranked by a cross-encoder.

    Dense and rerank need `sentence-transformers` (see requirements-ml.txt); BM25 alone runs anywhere.
    """

    def __init__(
        self,
        chunks: list,
        dense: bool = False,
        rerank: bool = False,
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        pool: int = 20,
        rrf_k: int = 60,
    ):
        self.chunks = chunks
        self.bm25 = BM25Index(chunks)
        self.pool, self.rrf_k = pool, rrf_k
        self._dense = self._rerank = None
        if dense:
            from sentence_transformers import SentenceTransformer  # lazy

            self._dense = SentenceTransformer(dense_model)
            self._emb = self._dense.encode([c.text for c in chunks], normalize_embeddings=True)
        if rerank:
            from sentence_transformers import CrossEncoder  # lazy

            self._rerank = CrossEncoder(rerank_model)

    def retrieve(self, query: str, k: int = 5) -> list:
        lex = self.bm25.scores(query)
        fused = {i: 1.0 / (self.rrf_k + r + 1) for r, i in enumerate(_rank(lex))}
        if self._dense is not None:
            q = self._dense.encode([query], normalize_embeddings=True)[0]
            sims = (self._emb @ q).tolist()
            for r, i in enumerate(_rank(sims)):
                fused[i] += 1.0 / (self.rrf_k + r + 1)
        order = sorted(fused, key=lambda i: (-fused[i], i))
        if self._rerank is not None:
            cand = order[: max(self.pool, k)]
            ce = self._rerank.predict([(query, self.chunks[i].text) for i in cand])
            order = [i for _, i in sorted(zip(ce, cand), key=lambda x: -x[0])]
        return [self.chunks[i] for i in order[:k]]
