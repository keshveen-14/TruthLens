"""Evaluate a TruthLens run against human claim labels.

    python -m evaluation.evaluate --run runs/run1.jsonl --labels labels/annotator_a.csv \
        [--labels2 labels/annotator_b.csv] [--primary nli] [--baseline lexical] \
        [--audit-verifier llm_judge] --out reports/run1

Headline numbers (all computed on the TEST split, thresholds tuned on DEV):
  * precision / recall / F1 of "flag the unsupported claim", per verifier, with answer-level
    bootstrap 95% CIs
  * paired bootstrap of F1(verifier) - F1(baseline verifier)
  * inter-annotator agreement (Cohen's kappa) when a second label file is given
  * unsupported-claim rate before vs after the action (remove / regenerate)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

from .metrics import (
    bootstrap_ci, cluster_bootstrap, cohen_kappa, confusion, group_rows, percentile_ci, prf, prf_rows,
)

LABEL_MAP = {
    "s": "S", "supported": "S", "u": "U", "unsupported": "U",
    "c": "C", "contradicted": "C", "x": "X", "skip": "X",
}
NEG_RE = re.compile(r"\b(?:not|no|never|cannot|without|neither|nor|none|unable)\b|n't", re.I)


# ------------------------------------------------------------------ loading

def read_labels(path: str) -> dict:
    """{(answer_id, claim_id): 'S'|'U'|'C'|'X'}; rows with an empty label are ignored."""
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            lab = LABEL_MAP.get((row.get("label") or "").strip().lower())
            if lab:
                out[(row["answer_id"], row["claim_id"])] = lab
    return out


def read_run(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def claim_category(claim: dict) -> str:
    """Coarse error-analysis bucket (first match wins)."""
    t = claim["text"]
    if any(ch.isdigit() for ch in t):
        return "numeric"
    if NEG_RE.search(t):
        return "negation"
    if len(claim.get("cited_ids") or []) >= 2:
        return "multi_source"
    return "other"


# ------------------------------------------------------------------ rows

def build_rows(records: list, labels: dict) -> list:
    rows = []
    for rec in records:
        by_v = {n: {v["claim_id"]: v for v in vs} for n, vs in rec["baseline"]["verdicts"].items()}
        for c in rec["baseline"]["claims"]:
            lab = labels.get((rec["answer_id"], c["id"]))
            if lab is None or lab == "X":
                continue
            rows.append({
                "answer_id": rec["answer_id"], "claim_id": c["id"], "split": rec.get("split", "test"),
                "y": lab in ("U", "C"), "lab": lab, "cat": claim_category(c),
                "v": {n: (d[c["id"]]["support_score"], d[c["id"]]["label"] != "supported")
                      for n, d in by_v.items() if c["id"] in d},
            })
    return rows


def _preds(rows, vname, t=None):
    if t is None:
        return [r["v"][vname][1] for r in rows]
    return [r["v"][vname][0] < t for r in rows]


def sweep(rows: list, vname: str) -> list:
    """Precision/recall/F1 at every distinct threshold (flag iff support_score < t)."""
    rows = [r for r in rows if vname in r["v"]]
    pts = sorted({r["v"][vname][0] for r in rows})
    if not pts:
        return []
    cands = [pts[0] - 1e-6] + pts + [(a + b) / 2 for a, b in zip(pts, pts[1:])] + [pts[-1] + 1e-6]
    out = []
    for t in sorted(set(cands)):
        tp, fp, fn, tn = confusion([r["y"] for r in rows], _preds(rows, vname, t))
        p, r_, f = prf(tp, fp, fn)
        out.append({"verifier": vname, "threshold": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "precision": p, "recall": r_, "f1": f})
    return out


def tune_threshold(dev_rows: list, vname: str):
    best, best_t = -1.0, None
    for s in sweep(dev_rows, vname):  # ascending t; strict '>' keeps the lowest threshold on ties
        if s["f1"] > best:
            best, best_t = s["f1"], s["threshold"]
    return best_t


def metric_block(rows: list, vname: str, t, n_boot: int, seed: int) -> dict:
    rows = [r for r in rows if vname in r["v"]]
    y = [r["y"] for r in rows]
    pr = _preds(rows, vname, t)
    tp, fp, fn, tn = confusion(y, pr)
    p, r_, f = prf(tp, fp, fn)
    groups = group_rows([(r["answer_id"], r["y"], q) for r, q in zip(rows, pr)], lambda x: x[0])
    groups = {k: [(a[1], a[2]) for a in v] for k, v in groups.items()}
    ci = bootstrap_ci(groups, prf_rows, n_boot, seed) if groups else [(None, None)] * 3
    return {"n": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "threshold": t,
            "precision": p, "recall": r_, "f1": f,
            "precision_ci": ci[0], "recall_ci": ci[1], "f1_ci": ci[2]}


def paired_f1_diff(rows: list, a: str, b: str, ta, tb, n_boot: int, seed: int) -> dict:
    rows = [r for r in rows if a in r["v"] and b in r["v"]]
    pa, pb = _preds(rows, a, ta), _preds(rows, b, tb)
    triples = [(r["answer_id"], r["y"], x, z) for r, x, z in zip(rows, pa, pb)]
    groups = group_rows(triples, lambda x: x[0])

    def stat(rs):
        return prf_rows([(r[1], r[2]) for r in rs])[2] - prf_rows([(r[1], r[3]) for r in rs])[2]

    point = stat(triples)
    boots = cluster_bootstrap(groups, stat, n_boot, seed) if groups else []
    lo, hi = percentile_ci(boots)
    return {"a": a, "b": b, "diff": point, "ci": (lo, hi),
            "p_le_0": (sum(1 for x in boots if x <= 0) / len(boots)) if boots else None}


# ------------------------------------------------------------------ agreement

def agreement(l1: dict, l2: dict) -> dict:
    keys = [k for k in l1 if k in l2 and l1[k] != "X" and l2[k] != "X"]
    a, b = [l1[k] for k in keys], [l2[k] for k in keys]
    ab = [x in ("U", "C") for x in a]
    bb = [x in ("U", "C") for x in b]
    return {"n": len(keys), "kappa_binary": cohen_kappa(ab, bb), "kappa_3way": cohen_kappa(a, b),
            "raw_agreement": (sum(1 for x, y in zip(a, b) if x == y) / len(keys)) if keys else None,
            "disagreements": [list(k) for k, x, y in zip(keys, a, b) if x != y][:50]}


# ------------------------------------------------------------------ reduction

def _stat_reduction(rows):
    ub = sum(r[0] for r in rows); nb = sum(r[1] for r in rows)
    ua = sum(r[2] for r in rows); na = sum(r[3] for r in rows)
    sb = sum(r[4] for r in rows); sa = sum(r[5] for r in rows)
    rb = ub / nb if nb else None
    ra = ua / na if na else 0.0
    red = (1 - ra / rb) if rb else None
    keep = (sa / sb) if sb else None
    anyb = sum(1 for r in rows if r[0] > 0) / len(rows)
    anya = sum(1 for r in rows if r[2] > 0) / len(rows)
    return (rb, ra, red, keep, anyb, anya)


def _reduction_from(per_answer: dict, n_boot: int, seed: int) -> dict:
    rows = [(k, *v) for k, v in per_answer.items()]
    groups = {k: [v] for k, v in per_answer.items()}
    if not groups:
        return {"n_answers": 0}
    point = _stat_reduction([v for v in per_answer.values()])
    ci = bootstrap_ci(groups, _stat_reduction, n_boot, seed)
    names = ["rate_before", "rate_after", "relative_reduction", "supported_kept",
             "answers_with_unsupported_before", "answers_with_unsupported_after"]
    out = {"n_answers": len(per_answer), "claims_before": sum(v[1] for v in per_answer.values()),
           "claims_after": sum(v[3] for v in per_answer.values())}
    for n, p, c in zip(names, point, ci):
        out[n] = p
        out[n + "_ci"] = c
    return out


def reduction_human(records: list, labels: dict, n_boot: int, seed: int) -> dict:
    """Exact: needs a human label for every claim before AND after (sheet --stage final adds the new ones)."""
    per, skipped = {}, 0
    for rec in records:
        aid = rec["answer_id"]
        lb = [labels.get((aid, c["id"])) for c in rec["baseline"]["claims"]]
        lf = [labels.get((aid, c["id"])) for c in rec["final"]["claims"]]
        if any(x is None for x in lb + lf):
            skipped += 1
            continue
        yb = [x in ("U", "C") for x in lb if x != "X"]
        yf = [x in ("U", "C") for x in lf if x != "X"]
        per[aid] = (sum(yb), len(yb), sum(yf), len(yf), len(yb) - sum(yb), len(yf) - sum(yf))
    out = _reduction_from(per, n_boot, seed)
    out["answers_skipped_missing_labels"] = skipped
    return out


def reduction_audit(records: list, audit: str, n_boot: int, seed: int) -> dict:
    """Estimated: an automatic verifier (ideally not the one that acted) scores both versions."""
    per = {}
    for rec in records:
        vb = rec["baseline"]["verdicts"].get(audit)
        vf = rec["final"]["verdicts"].get(audit)
        if vb is None or vf is None:
            continue
        fb = [v["label"] != "supported" for v in vb]
        ff = [v["label"] != "supported" for v in vf]
        per[rec["answer_id"]] = (sum(fb), len(fb), sum(ff), len(ff), len(fb) - sum(fb), len(ff) - sum(ff))
    return _reduction_from(per, n_boot, seed)


# ------------------------------------------------------------------ main

def evaluate(records, labels, labels2=None, primary=None, baseline="lexical",
             audit=None, n_boot=2000, seed=0) -> dict:
    rows = build_rows(records, labels)
    verifiers = sorted({n for r in rows for n in r["v"]})
    if not rows:
        raise SystemExit("no labelled claims matched the run file: check answer_id / claim_id columns")
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    eval_rows, eval_name = (test, "test") if test else (rows, "all (no test split: thresholds not held out)")
    warnings = []
    if not test:
        warnings.append("No labelled test-split claims: results use all claims, so tuned thresholds are optimistic.")
    n_dev_pos = sum(1 for r in dev if r["y"])
    if n_dev_pos < 15:
        warnings.append(f"Only {n_dev_pos} unsupported claims in the labelled dev split: tuned thresholds are unreliable; "
                        "label more dev answers or rely on the default-threshold rows.")

    thresholds = {v: (tune_threshold(dev, v) if dev else None) for v in verifiers}
    res = {"meta": {"eval_split": eval_name, "n_answers": len({r["answer_id"] for r in eval_rows}),
                    "n_bootstrap": n_boot, "seed": seed, "warnings": warnings},
           "data": {"labelled_claims": len(rows), "unsupported_rate_human": sum(r["y"] for r in rows) / len(rows),
                    "eval_claims": len(eval_rows), "eval_unsupported": sum(r["y"] for r in eval_rows),
                    "label_counts": {k: sum(1 for r in rows if r["lab"] == k) for k in ("S", "U", "C")}},
           "verifiers": {}}

    for v in verifiers:
        res["verifiers"][v] = {"default": metric_block(eval_rows, v, None, n_boot, seed)}
        if thresholds[v] is not None:
            res["verifiers"][v]["tuned"] = metric_block(eval_rows, v, thresholds[v], n_boot, seed)

    mode = "tuned" if all("tuned" in res["verifiers"][v] for v in verifiers) else "default"
    t_of = (lambda v: thresholds[v]) if mode == "tuned" else (lambda v: None)
    res["meta"]["headline_mode"] = mode
    res["paired"] = [paired_f1_diff(eval_rows, v, baseline, t_of(v), t_of(baseline), n_boot, seed)
                     for v in verifiers if v != baseline and baseline in verifiers]

    primary = primary or ("llm_judge" if "llm_judge" in verifiers else "nli" if "nli" in verifiers else verifiers[0])
    res["meta"]["primary"] = primary
    res["categories"] = {}
    for cat in sorted({r["cat"] for r in eval_rows}):
        sub = [r for r in eval_rows if r["cat"] == cat and primary in r["v"]]
        if not sub:
            continue
        tp, fp, fn, tn = confusion([r["y"] for r in sub], _preds(sub, primary, t_of(primary)))
        p, r_, f = prf(tp, fp, fn)
        res["categories"][cat] = {"n": len(sub), "positives": sum(r["y"] for r in sub),
                                  "precision": p, "recall": r_, "f1": f, "fp": fp, "fn": fn}

    res["sweep"] = [s for v in verifiers for s in sweep(dev or rows, v)]
    if labels2:
        res["agreement"] = agreement(labels, labels2)

    res["reduction_human"] = reduction_human(records, labels, n_boot, seed)
    if audit:
        res["reduction_audit"] = reduction_audit(records, audit, n_boot, seed)
        acts = {r["config"].get("act_verifier") for r in records}
        res["reduction_audit"]["audit_verifier"] = audit
        res["reduction_audit"]["act_verifier"] = sorted(a for a in acts if a)
        if audit in acts:
            warnings.append(f"Audit verifier {audit!r} is the same verifier that removed/regenerated claims; "
                            "that estimate is biased low. Use a human label or a different verifier.")
    res["actions"] = {"action": sorted({r["config"]["action"] for r in records}),
                      "abstained": sum(1 for r in records if r.get("abstained")), "answers": len(records)}
    return res


# ------------------------------------------------------------------ rendering

def _f(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"


def _ci(x, ci, d=3):
    if x is None:
        return "n/a"
    if not ci or ci[0] is None:
        return _f(x, d)
    return f"{x:.{d}f} [{ci[0]:.{d}f}, {ci[1]:.{d}f}]"


def render_markdown(res: dict) -> str:
    m, d = res["meta"], res["data"]
    L = ["# TruthLens evaluation", ""]
    L += [f"- Evaluated on: **{m['eval_split']}** ({m['n_answers']} answers, {d['eval_claims']} claims, "
          f"{d['eval_unsupported']} unsupported)",
          f"- Labelled claims overall: {d['labelled_claims']} "
          f"(S={d['label_counts']['S']}, U={d['label_counts']['U']}, C={d['label_counts']['C']}; "
          f"human unsupported rate {_f(d['unsupported_rate_human'])})",
          f"- 95% CIs: percentile bootstrap over answers, {m['n_bootstrap']} resamples, seed {m['seed']}",
          f"- Headline thresholds: **{m['headline_mode']}** (tuned on dev, applied to the held-out set)", ""]
    for w in m["warnings"]:
        L.append(f"> WARNING: {w}")
    L.append("")
    L += ["## Detecting unsupported claims (positive class = unsupported or contradicted)", "",
          "| verifier | mode | threshold | precision | recall | F1 | TP | FP | FN | TN |", "|---|---|---|---|---|---|---|---|---|---|"]
    for v, blocks in res["verifiers"].items():
        for mode, b in blocks.items():
            L.append(f"| {v} | {mode} | {_f(b['threshold'])} | {_ci(b['precision'], b['precision_ci'])} | "
                     f"{_ci(b['recall'], b['recall_ci'])} | {_ci(b['f1'], b['f1_ci'])} | {b['tp']} | {b['fp']} | {b['fn']} | {b['tn']} |")
    if res["paired"]:
        L += ["", "## F1 difference vs baseline (paired bootstrap)", "", "| verifier | baseline | ΔF1 | 95% CI | P(Δ≤0) |", "|---|---|---|---|---|"]
        for p in res["paired"]:
            L.append(f"| {p['a']} | {p['b']} | {_ci(p['diff'], p['ci'])} | | {_f(p['p_le_0'])} |")
    L += ["", f"## Error analysis by claim type (verifier: {m['primary']})", "", "| type | n | unsupported | precision | recall | F1 | FP | FN |", "|---|---|---|---|---|---|---|---|"]
    for c, b in res["categories"].items():
        L.append(f"| {c} | {b['n']} | {b['positives']} | {_f(b['precision'])} | {_f(b['recall'])} | {_f(b['f1'])} | {b['fp']} | {b['fn']} |")
    if "agreement" in res:
        a = res["agreement"]
        L += ["", "## Label agreement (annotator A vs B)", "",
              f"- Claims labelled by both: {a['n']}", f"- Raw agreement: {_f(a['raw_agreement'])}",
              f"- Cohen's kappa (supported vs not): **{_f(a['kappa_binary'])}**", f"- Cohen's kappa (S/U/C): {_f(a['kappa_3way'])}"]
    for key, title in (("reduction_human", "Unsupported-claim rate before vs after (human labels, exact)"),
                       ("reduction_audit", "Unsupported-claim rate before vs after (automatic audit, estimated)")):
        r = res.get(key)
        if not r:
            continue
        L += ["", f"## {title}", ""]
        if r.get("n_answers", 0) == 0:
            L.append("No answers had complete labels for both versions. "
                     "For `regenerate`, run `scripts/build_label_sheet.py --stage final` and label the new claims."
                     if key == "reduction_human" else "No audit verdicts found in the run.")
            continue
        if key == "reduction_audit":
            L.append(f"Audit verifier: `{r['audit_verifier']}`; acting verifier: `{', '.join(r['act_verifier'])}`.")
            L.append("")
        L += [f"- Answers: {r['n_answers']}; claims before/after: {r['claims_before']} / {r['claims_after']}",
              f"- Unsupported-claim rate: {_ci(r['rate_before'], r['rate_before_ci'])} → {_ci(r['rate_after'], r['rate_after_ci'])}",
              f"- **Relative reduction: {_ci(r['relative_reduction'], r['relative_reduction_ci'])}**",
              f"- Answers containing ≥1 unsupported claim: {_f(r['answers_with_unsupported_before'])} → {_f(r['answers_with_unsupported_after'])}",
              f"- Supported claims kept (after / before): {_ci(r['supported_kept'], r['supported_kept_ci'])}"]
        if key == "reduction_human":
            L.append(f"- Answers skipped (missing labels): {r.get('answers_skipped_missing_labels', 0)}")
    L += ["", f"Actions in run: {', '.join(res['actions']['action'])}; abstained answers: {res['actions']['abstained']}/{res['actions']['answers']}", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--labels", required=True, nargs="+", help="one or more label CSVs (merged)")
    ap.add_argument("--labels2", help="second annotator's CSV, for Cohen's kappa")
    ap.add_argument("--primary")
    ap.add_argument("--baseline", default="lexical")
    ap.add_argument("--audit-verifier")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/report")
    a = ap.parse_args(argv)

    labels = {}
    for p in a.labels:
        labels.update(read_labels(p))
    res = evaluate(read_run(a.run), labels, read_labels(a.labels2) if a.labels2 else None,
                   a.primary, a.baseline, a.audit_verifier, a.bootstrap, a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    with open(a.out + ".md", "w") as f:
        f.write(render_markdown(res))
    with open(a.out + "_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["verifier", "threshold", "tp", "fp", "fn", "tn", "precision", "recall", "f1"])
        w.writeheader()
        w.writerows(res["sweep"])
    print(render_markdown(res))
    print(f"\nwrote {a.out}.md, {a.out}.json, {a.out}_sweep.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
