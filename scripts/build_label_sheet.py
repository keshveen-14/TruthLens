"""Turn a run file into a CSV for human annotation.

    python scripts/build_label_sheet.py --run runs/run1.jsonl --out labels/annotator_a.csv
    python scripts/build_label_sheet.py --run runs/run1.jsonl --out labels/annotator_b.csv --double 25 --seed 7
    python scripts/build_label_sheet.py --run runs/run1.jsonl --out labels/final_claims.csv \
        --stage final --skip-labeled labels/annotator_a.csv

The sheet deliberately hides every verifier's output so annotators are not anchored by it.
Fill the `label` column with S (supported), U (unsupported), C (contradicted) or X (not a checkable
claim / bad split). See LABELING_GUIDE.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.evaluate import read_labels  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="baseline", choices=["baseline", "final"],
                    help="baseline = claims of the unverified answer; final = claims of the post-action answer")
    ap.add_argument("--double", type=int, default=None, help="only N randomly chosen answers (for the 2nd annotator)")
    ap.add_argument("--skip-labeled", default=None, help="skip claims already labelled in this CSV")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    with open(a.run, encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    if a.double:
        random.Random(a.seed).shuffle(recs)
        recs = recs[: a.double]
        recs.sort(key=lambda r: r["answer_id"])
    skip = set(read_labels(a.skip_labeled)) if a.skip_labeled else set()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    n = 0
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["answer_id", "claim_id", "stage", "question", "answer", "claim",
                    "cited_evidence", "other_retrieved_evidence", "label", "notes"])
        for r in recs:
            snap = r[a.stage]
            by_id = {c["id"]: c["text"] for c in r["retrieved"]}
            first = True
            for c in snap["claims"]:
                if (r["answer_id"], c["id"]) in skip:
                    continue
                cited = " || ".join(f"[{i}] {by_id[i]}" for i in c["cited_ids"] if i in by_id)
                other = " || ".join(f"[{i}] {t}" for i, t in by_id.items() if i not in c["cited_ids"])
                w.writerow([r["answer_id"], c["id"], a.stage, r["question"] if first else "",
                            snap["answer"] if first else "", c["text"], cited, other, "", ""])
                first = False
                n += 1
    print(f"wrote {n} claims from {len(recs)} answers to {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
