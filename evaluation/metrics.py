"""Metrics with uncertainty. Pure Python, no dependencies.

The unit of resampling is the ANSWER, not the claim: claims from one answer share a question,
retrieved sources and a generation, so they are not independent. Resampling claims would give
confidence intervals that are too narrow.
"""

from __future__ import annotations

import random


def confusion(y_true: list, y_pred: list) -> tuple:
    """Positive class = 'unsupported / contradicted' (the thing we want to catch)."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    return tp, fp, fn, tn


def prf(tp: int, fp: int, fn: int) -> tuple:
    """(precision, recall, f1); a zero denominator gives 0.0."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def prf_rows(rows: list) -> tuple:
    """rows of (y_true, y_pred) -> (precision, recall, f1)."""
    tp, fp, fn, _ = confusion([r[0] for r in rows], [r[1] for r in rows])
    return prf(tp, fp, fn)


def cohen_kappa(a: list, b: list):
    """Cohen's kappa for two raters over the same items (any hashable labels)."""
    n = len(a)
    if n == 0 or n != len(b):
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    cats = set(a) | set(b)
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def percentile_ci(vals: list, alpha: float = 0.05):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return (None, None)
    lo = vals[int((alpha / 2) * len(vals))]
    hi = vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]
    return (lo, hi)


def cluster_bootstrap(groups: dict, stat_fn, n: int = 2000, seed: int = 0) -> list:
    """Resample whole groups (answers) with replacement; `stat_fn(rows)` -> scalar or tuple.

    Returns the list of bootstrap statistics.
    """
    rng = random.Random(seed)
    keys = list(groups)
    out = []
    for _ in range(n):
        rows = []
        for k in (rng.choice(keys) for _ in keys):
            rows.extend(groups[k])
        out.append(stat_fn(rows))
    return out


def bootstrap_ci(groups: dict, stat_fn, n: int = 2000, seed: int = 0, alpha: float = 0.05):
    """CI per component when stat_fn returns a tuple, a single CI when it returns a scalar."""
    stats = cluster_bootstrap(groups, stat_fn, n, seed)
    if stats and isinstance(stats[0], tuple):
        return [percentile_ci([s[i] for s in stats], alpha) for i in range(len(stats[0]))]
    return percentile_ci(stats, alpha)


def group_rows(rows: list, key_fn) -> dict:
    g = {}
    for r in rows:
        g.setdefault(key_fn(r), []).append(r)
    return g
