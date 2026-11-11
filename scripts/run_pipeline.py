"""Run TruthLens over a question file and write one JSON record per answer.

    python scripts/run_pipeline.py --corpus data/corpus --questions data/questions.jsonl \
        --out runs/run1.jsonl --verifiers lexical,nli,llm_judge --act llm_judge --action remove

Re-running with the same --out resumes: answers already in the file are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from truthlens.factory import build_pipeline  # noqa: E402
from truthlens.pipeline import PipelineConfig  # noqa: E402
from truthlens.util import assign_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--questions", default="data/questions.jsonl")
    ap.add_argument("--out", default="runs/run1.jsonl")
    ap.add_argument("--llm", default="auto", choices=["auto", "live", "anthropic", "groq", "gemini", "openrouter", "openai", "mock"])
    ap.add_argument("--model", default=None, help="override TRUTHLENS_MODEL")
    ap.add_argument("--style", default="strict", choices=["strict", "permissive"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--claims", default="rule", choices=["rule", "llm"])
    ap.add_argument("--action", default="flag", choices=["flag", "remove", "regenerate"])
    ap.add_argument("--max-rounds", type=int, default=2)
    ap.add_argument("--verifiers", default="lexical", help="comma list: lexical,embedding,nli,llm_judge")
    ap.add_argument("--act", default=None, help="verifier that decides removal/regeneration (default: first)")
    ap.add_argument("--dense", action="store_true", help="add dense retrieval (needs sentence-transformers)")
    ap.add_argument("--rerank", action="store_true", help="add cross-encoder reranking")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    cfg = PipelineConfig(k=a.k, style=a.style, claim_mode=a.claims, action=a.action, max_rounds=a.max_rounds)
    tl = build_pipeline(
        a.corpus, llm_kind=a.llm, verifier_names=[v.strip() for v in a.verifiers.split(",")],
        act=a.act, cfg=cfg, dense=a.dense, rerank=a.rerank, model=a.model,
    )
    if getattr(tl.llm, "model", "") == "mock":
        print("NOTE: using the MOCK llm (no API key set). Outputs are for smoke tests only.", file=sys.stderr)

    with open(a.questions, encoding="utf-8") as f:
        qs = [json.loads(l) for l in f if l.strip()]
    if a.limit:
        qs = qs[: a.limit]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            done = {json.loads(l)["answer_id"] for l in f if l.strip()}

    with open(a.out, "a", encoding="utf-8") as out:
        for i, q in enumerate(qs, 1):
            aid = q["id"]
            if aid in done:
                continue
            rec = tl.run(q["question"], answer_id=aid, split=q.get("split") or assign_split(aid))
            rec["kind"] = q.get("kind")
            out.write(json.dumps(rec) + "\n")
            out.flush()
            nb, nf = len(rec["baseline"]["claims"]), len(rec["final"]["claims"])
            flagged = sum(1 for v in rec["baseline"]["verdicts"][tl.act] if v["label"] != "supported")
            print(f"[{i}/{len(qs)}] {aid}: {nb} claims, {flagged} flagged by {tl.act}; final has {nf}", file=sys.stderr)

    for v in tl.verifiers.values():
        if getattr(v, "parse_errors", 0):
            print(f"WARNING: {v.name} had {v.parse_errors} unparseable replies (flagged as unsupported)", file=sys.stderr)
    print(f"wrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
