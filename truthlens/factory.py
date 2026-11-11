from __future__ import annotations

from .corpus import load_corpus
from .llm import get_llm
from .pipeline import PipelineConfig, TruthLens
from .retrieval import HybridRetriever
from .verify import build_verifier


def build_pipeline(
    corpus_dir: str,
    *,
    llm_kind: str = "auto",
    verifier_names=("lexical",),
    act: str | None = None,
    cfg: PipelineConfig | None = None,
    dense: bool = False,
    rerank: bool = False,
    model: str | None = None,
    verifier_kwargs: dict | None = None,
) -> TruthLens:
    chunks = load_corpus(corpus_dir)
    retriever = HybridRetriever(chunks, dense=dense, rerank=rerank)
    llm = get_llm(llm_kind, model=model)
    if hasattr(llm, "attach_chunks"):  # only the offline demo model uses this
        llm.attach_chunks(chunks)
    kw = verifier_kwargs or {}
    verifiers = {}
    for n in verifier_names:
        v = build_verifier(n, llm=llm, **kw.get(n, {}))
        verifiers[v.name] = v
    return TruthLens(retriever, llm, verifiers, act or list(verifiers)[0], cfg)
