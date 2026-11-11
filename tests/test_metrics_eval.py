"""Checks the metric code and the evaluation harness.

The end-to-end test builds labels from MockLLM's known-injected sentences ("oracle" labels). That is
ONLY a plumbing test of the harness; it says nothing about how well TruthLens works on real answers.
"""

import os

import pytest

from evaluation.evaluate import evaluate, reduction_human
from evaluation.metrics import (
    bootstrap_ci, cohen_kappa, confusion, group_rows, percentile_ci, prf, prf_rows,
)
from truthlens.factory import build_pipeline
from truthlens.llm import INJECTED_BANK
from truthlens.pipeline import PipelineConfig

CORPUS = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")


def test_prf_known_values():
    y = [1, 1, 1, 0, 0, 0, 0, 1]
    p = [1, 1, 0, 1, 0, 0, 0, 0]
    tp, fp, fn, tn = confusion(y, p)
    assert (tp, fp, fn, tn) == (2, 1, 2, 3)
    pr, rc, f1 = prf(tp, fp, fn)
    assert pr == pytest.approx(2 / 3) and rc == 0.5 and f1 == pytest.approx(4 / 7)
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)


def test_cohen_kappa_known_values():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)
    a = ["S"] * 20 + ["U"] * 10 + ["S"] * 15 + ["U"] * 5
    b = ["S"] * 20 + ["U"] * 10 + ["U"] * 15 + ["S"] * 5
    # po = 30/50, pa(S)=35/50, pb(S)=25/50 -> pe = .7*.5+.3*.5 = .5 -> kappa = .2
    assert cohen_kappa(a, b) == pytest.approx(0.2)


def test_bootstrap_ci_brackets_the_point_estimate_and_resamples_answers():
    rows = [(f"a{i}", i % 3 == 0, i % 3 == 0 or i % 7 == 0) for i in range(60)]
    groups = {k: [(y, p)] for k, y, p in rows}
    point = prf_rows([(y, p) for _, y, p in rows])
    ci = bootstrap_ci(groups, prf_rows, n=500, seed=1)
    for (lo, hi), x in zip(ci, point):
        assert lo <= x <= hi
    assert percentile_ci(list(range(1, 101)), 0.1) == (6, 96)  # 5th and 95th percentile by index
    assert percentile_ci([]) == (None, None)


def _oracle_labels(records):
    labels = {}
    for r in records:
        for snap in (r["baseline"], r["final"]):
            for c in snap["claims"]:
                bad = any(c["text"] == b for b in INJECTED_BANK)
                labels[(r["answer_id"], c["id"])] = "U" if bad else "S"
    return labels


def test_end_to_end_harness_on_mock_pipeline():
    tl = build_pipeline(CORPUS, llm_kind="mock", verifier_names=["lexical"], cfg=PipelineConfig(action="remove"))
    qs = [(f"q{i:03d}", q) for i, q in enumerate(
        ["How long is the warranty?", "What is the price of the H10?", "Can it be installed in a bedroom?",
         "How noisy is it?", "What does support cost?", "Is it safe?", "What are the specs of the H15?",
         "How do I return it?", "Is there an app?", "What is the efficiency?"] * 4)]
    records = [tl.run(q, answer_id=i, split="dev" if n % 3 == 0 else "test") for n, (i, q) in enumerate(qs)]
    labels = _oracle_labels(records)
    assert any(v == "U" for v in labels.values())

    res = evaluate(records, labels, labels, n_boot=200)
    assert res["meta"]["eval_split"] == "test"
    b = res["verifiers"]["lexical"]["default"]
    assert b["tp"] + b["fn"] == res["data"]["eval_unsupported"]
    assert 0.0 <= b["precision"] <= 1.0 and b["recall"] == pytest.approx(1.0)  # injected sentences are far from the sources
    assert res["agreement"]["kappa_binary"] == 1.0

    red = res["reduction_human"]
    assert red["rate_before"] > 0 and red["rate_after"] == 0 and red["relative_reduction"] == pytest.approx(1.0)
    assert red["supported_kept"] == pytest.approx(1.0)  # remove mode must not delete true claims here


def test_reduction_skips_answers_with_missing_labels():
    tl = build_pipeline(CORPUS, llm_kind="mock", verifier_names=["lexical"], cfg=PipelineConfig(action="remove"))
    recs = [tl.run("How long is the warranty?", answer_id="a1")]
    out = reduction_human(recs, {}, 50, 0)
    assert out["n_answers"] == 0 and out["answers_skipped_missing_labels"] == 1


def test_group_rows():
    assert group_rows([(1, "a"), (2, "b"), (1, "c")], lambda r: r[0]) == {1: [(1, "a"), (1, "c")], 2: [(2, "b")]}
