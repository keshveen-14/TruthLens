"""Streamlit UI: ask a question, see the answer with every claim highlighted green / red.

    streamlit run app.py
"""

from __future__ import annotations

import html

import streamlit as st

from truthlens.factory import build_pipeline
from truthlens.pipeline import PipelineConfig

st.set_page_config(page_title="TruthLens", layout="wide")
st.title("TruthLens")
st.caption("Every claim in the answer is checked against the retrieved sources. Green = supported, red = flagged.")

with st.sidebar:
    st.header("Settings")
    corpus = st.text_input("Corpus folder", "data/corpus")
    llm_kind = st.selectbox("LLM backend", ["auto", "live", "anthropic", "groq", "gemini", "openrouter", "mock"],
                            help="auto uses a real model when an API key is set (Groq, Gemini, OpenRouter, Anthropic), otherwise a mock for demos.")
    verifiers = st.multiselect("Verifiers", ["lexical", "nli", "llm_judge"], default=["lexical"],
                               help="nli needs transformers+torch; llm_judge needs an API key.")
    act = st.selectbox("Verifier that decides what to drop", verifiers or ["lexical"])
    action = st.selectbox("When claims are flagged", ["flag", "remove", "regenerate"],
                          help="flag = annotate only; remove = drop flagged sentences; regenerate = ask the model to fix, then drop what's left.")
    style = st.selectbox("Generation prompt", ["strict", "permissive"])
    claim_mode = st.selectbox("Claim splitting", ["rule", "llm"])
    k = st.slider("Chunks retrieved", 2, 10, 5)


@st.cache_resource(show_spinner="Loading corpus and models...")
def get_pipeline(corpus, llm_kind, verifiers, act, action, style, claim_mode, k):
    cfg = PipelineConfig(k=k, style=style, claim_mode=claim_mode, action=action)
    return build_pipeline(corpus, llm_kind=llm_kind, verifier_names=list(verifiers), act=act, cfg=cfg)


def render(snap: dict, act: str, chunks: dict) -> str:
    verdicts = {v["claim_id"]: v for v in snap["verdicts"][act]}
    claims_by_sentence = {}
    for c in snap["claims"]:
        claims_by_sentence.setdefault(c["sentence_idx"], []).append(c)
    parts = []
    for s in snap["sentences"]:
        cs = claims_by_sentence.get(s["idx"], [])
        if not cs:
            parts.append(f"<span>{html.escape(s['raw'])}</span>")
            continue
        vs = [verdicts[c["id"]] for c in cs if c["id"] in verdicts]
        if any(v["label"] == "contradicted" for v in vs):
            bg = "#fca5a5"
        elif any(v["label"] != "supported" for v in vs):
            bg = "#fecaca"
        else:
            bg = "#bbf7d0"
        tip = "&#10;".join(
            html.escape(f"{v['label']} ({v['support_score']:.2f}) - evidence {v['evidence_id']}: {v['rationale'][:120]}")
            for v in vs
        )
        parts.append(f"<span title='{tip}' style='background:{bg};color:#111;padding:2px 4px;border-radius:4px;margin-right:2px'>"
                     f"{html.escape(s['raw'])}</span>")
    return " ".join(parts)


question = st.text_input("Ask a question about the documents", "How long is the standard warranty and when does it start?")
if st.button("Ask", type="primary") and question.strip():
    if not verifiers:
        st.error("Pick at least one verifier.")
        st.stop()
    try:
        tl = get_pipeline(corpus, llm_kind, tuple(verifiers), act, action, style, claim_mode, k)
        rec = tl.run(question)
    except Exception as e:  # surface config problems (missing key, missing deps) in the UI
        st.error(f"{type(e).__name__}: {e}")
        st.stop()

    if tl.llm.model == "mock":
        st.warning("Mock backend: answers are canned and include deliberately injected false sentences. For demos only.")

    chunks = {c["id"]: c for c in rec["retrieved"]}
    base_v = rec["baseline"]["verdicts"][tl.act]
    n, nf = len(base_v), sum(1 for v in base_v if v["label"] != "supported")
    c1, c2, c3 = st.columns(3)
    c1.metric("Claims checked", n)
    c2.metric("Flagged", nf)
    c3.metric("Unsupported rate", f"{(nf / n):.0%}" if n else "n/a")

    st.subheader("Answer, as generated")
    st.markdown(render(rec["baseline"], tl.act, chunks), unsafe_allow_html=True)
    if action != "flag":
        st.subheader("Answer after verification" + (" (abstained)" if rec["abstained"] else ""))
        st.markdown(render(rec["final"], tl.act, chunks), unsafe_allow_html=True)
        if rec["rounds"]:
            st.caption(f"{rec['rounds']} regeneration round(s) used.")

    with st.expander("Claims and verdicts"):
        for name, vs in rec["baseline"]["verdicts"].items():
            st.markdown(f"**{name}**")
            by_id = {c["id"]: c for c in rec["baseline"]["claims"]}
            st.dataframe(
                [{"claim": by_id[v["claim_id"]]["text"], "label": v["label"], "score": v["support_score"],
                  "evidence": v["evidence_id"], "cited correctly": v["cited_ok"]} for v in vs],
                width="stretch",
            )
    with st.expander("Retrieved sources"):
        for c in rec["retrieved"]:
            st.markdown(f"**[{c['id']}]** `{c['doc']}`")
            st.write(c["text"])
