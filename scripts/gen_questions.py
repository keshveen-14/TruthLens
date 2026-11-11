"""Draft an evaluation question set from your corpus with an LLM (needs a real model: set GROQ_API_KEY, GEMINI_API_KEY or ANTHROPIC_API_KEY).

    python scripts/gen_questions.py --corpus data/corpus --out data/questions.jsonl --n 100

A good set mixes four kinds so the system actually produces hallucinations to catch:
answerable, partial (related but the corpus only covers part), unanswerable (plausible, not in the
corpus), false_premise (assumes something untrue). Read and edit the output before using it.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from truthlens.corpus import load_corpus  # noqa: E402
from truthlens.llm import get_llm  # noqa: E402
from truthlens.util import assign_split, extract_json  # noqa: E402

SYSTEM = (
    "You write evaluation questions for a document-QA system. Given a passage, write 4 questions a "
    "real user might ask about the topic it covers:\n"
    "1. answerable: fully answered by the passage\n"
    "2. partial: related, but the passage answers only part of it\n"
    "3. unanswerable: plausible and on-topic but NOT answered by the passage\n"
    "4. false_premise: contains an assumption the passage contradicts or does not support\n"
    'Reply with JSON only: [{"kind": "answerable", "question": "..."}, ...]'
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--out", default="data/questions.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    llm = get_llm("auto")
    if llm.model == "mock":
        sys.exit("gen_questions needs a real LLM: set GROQ_API_KEY, GEMINI_API_KEY or ANTHROPIC_API_KEY")
    chunks = load_corpus(a.corpus)
    random.Random(a.seed).shuffle(chunks)

    items, seen = [], set()
    for ch in chunks:
        if len(items) >= a.n:
            break
        data = extract_json(llm.complete(SYSTEM, f"PASSAGE:\n{ch.text}", task="questions", max_tokens=600))
        for d in data if isinstance(data, list) else []:
            q = str(d.get("question", "")).strip()
            if q and q.lower() not in seen and len(items) < a.n:
                seen.add(q.lower())
                items.append({"question": q, "kind": d.get("kind", "?"), "source_chunk": ch.id})
    random.Random(a.seed).shuffle(items)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for i, it in enumerate(items, 1):
            it["id"] = f"q{i:03d}"
            it["split"] = assign_split(it["id"])
            f.write(json.dumps(it) + "\n")
    print(f"wrote {len(items)} questions to {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
