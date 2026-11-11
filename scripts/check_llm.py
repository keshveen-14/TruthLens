"""Live smoke test for the real-model path. Run once with your key before trusting any results:

    export GROQ_API_KEY=...             (free: console.groq.com/keys)
    # or GEMINI_API_KEY (free: aistudio.google.com/apikey), OPENROUTER_API_KEY, ANTHROPIC_API_KEY
    # Windows PowerShell:  $env:GROQ_API_KEY="..."
    python scripts/check_llm.py [--nli]

Steps: (1) one tiny API call, (2) a full pipeline run with the LLM judge and LLM claim splitting,
(3) optionally the NLI verifier (downloads ~700 MB the first time). Prints what failed and why.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from truthlens.factory import build_pipeline  # noqa: E402
from truthlens.llm import LLMError, get_llm, provider_info  # noqa: E402

Q = "How long is the standard warranty and when does it start?"


def step(name, fn):
    try:
        out = fn()
        print(f"[ ok ] {name}")
        return out
    except LLMError as e:
        print(f"[FAIL] {name}: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli", action="store_true", help="also load and run the NLI verifier")
    a = ap.parse_args()
    info = provider_info()
    print(f"provider: {info['provider_label']}  model: {info['model']}  (override with TRUTHLENS_MODEL / TRUTHLENS_PROVIDER)")

    llm = step("API call", lambda: get_llm("live", cache=False).complete("Reply with one word.", "Say OK.", max_tokens=20))
    print("       reply:", llm.strip()[:60])

    names = ["lexical", "llm_judge"] + (["nli"] if a.nli else [])
    tl = step("build pipeline", lambda: build_pipeline("data/corpus", llm_kind="live", verifier_names=names, act="llm_judge"))
    tl.cfg.claim_mode, tl.cfg.action = "llm", "remove"
    rec = step("run pipeline", lambda: tl.run(Q))

    print("\nANSWER :", rec["baseline"]["answer"])
    for name, vs in rec["baseline"]["verdicts"].items():
        print(f"{name:10s}:", ", ".join(f"{v['claim_id']}={v['label']}" for v in vs))
    parse_errors = getattr(tl.verifiers["llm_judge"], "parse_errors", 0)
    print("judge parse errors:", parse_errors)
    print("AFTER  :", rec["final"]["answer"])
    print("\nAll steps passed." if not parse_errors else "\nPassed, but the judge reply could not be parsed some times.")


if __name__ == "__main__":
    main()
