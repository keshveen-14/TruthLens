import os

from truthlens.factory import build_pipeline
from truthlens.llm import INJECTED_BANK
from truthlens.pipeline import ABSTAIN, PipelineConfig

CORPUS = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")


def _find_injected_question(tl):
    for i in range(60):
        q = f"What is feature number {i} of the H10 battery?"
        r = tl.run(q)
        if any(c["text"] in INJECTED_BANK for c in r["baseline"]["claims"]):
            return q
    raise AssertionError("mock never injected a sentence")


def test_flag_mode_keeps_answer_but_flags_injected_claim():
    tl = build_pipeline(CORPUS, llm_kind="mock", cfg=PipelineConfig(action="flag"))
    q = _find_injected_question(tl)
    r = tl.run(q)
    assert r["final"] is r["baseline"]
    flagged = [v for v in r["baseline"]["verdicts"]["lexical"] if v["label"] != "supported"]
    assert flagged


def test_remove_mode_drops_flagged_sentence_and_keeps_supported_ones():
    tl = build_pipeline(CORPUS, llm_kind="mock", cfg=PipelineConfig(action="remove"))
    q = _find_injected_question(tl)
    r = tl.run(q)
    assert not any(c["text"] in INJECTED_BANK for c in r["final"]["claims"])
    assert 0 < len(r["final"]["claims"]) < len(r["baseline"]["claims"])
    # claim ids survive removal, so human labels on the baseline still apply
    base_ids = {c["id"] for c in r["baseline"]["claims"]}
    assert {c["id"] for c in r["final"]["claims"]} <= base_ids
    # verdicts follow the surviving claims
    assert {v["claim_id"] for v in r["final"]["verdicts"]["lexical"]} == {c["id"] for c in r["final"]["claims"]}


def test_regenerate_mode_removes_flagged_claim_and_uses_new_ids():
    tl = build_pipeline(CORPUS, llm_kind="mock", cfg=PipelineConfig(action="regenerate"))
    q = _find_injected_question(tl)
    r = tl.run(q)
    assert r["rounds"] == 1
    assert all(c["id"].startswith("g1.") for c in r["final"]["claims"])
    assert not any(c["text"] in INJECTED_BANK for c in r["final"]["claims"])


def test_everything_removed_becomes_an_abstention():
    tl = build_pipeline(CORPUS, llm_kind="mock", cfg=PipelineConfig(action="remove"))
    # a verifier threshold of 2.0 flags every claim
    tl.verifiers["lexical"].threshold = 2.0
    r = tl.run("How long is the warranty?")
    assert r["abstained"] and r["final"]["answer"] == ABSTAIN and r["final"]["claims"] == []


def test_record_is_json_serialisable():
    import json

    tl = build_pipeline(CORPUS, llm_kind="mock")
    json.dumps(tl.run("How noisy is the battery?"))
