"""Free-tier providers (Groq, Gemini, OpenRouter, ...) through the OpenAI-style client, against a local fake."""
import http.client
import json
import threading

import pytest

import server
from tests.fake_anthropic import Fake, start
from truthlens import llm as L
from truthlens.factory import build_pipeline
from truthlens.llm import LLMError, OpenAICompatLLM, detect_provider, get_llm

Q = "How long is the standard warranty and when does it start?"
KEYS = ["ANTHROPIC_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY",
        "OPENAI_API_KEY", "TRUTHLENS_PROVIDER", "TRUTHLENS_MODEL", "TRUTHLENS_BASE_URL", "TRUTHLENS_RPM"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    OpenAICompatLLM._clock.clear()


@pytest.fixture()
def fake(monkeypatch, tmp_path):
    f = Fake()
    srv = start(f)
    monkeypatch.setenv("TRUTHLENS_BASE_URL", f"http://127.0.0.1:{srv.server_address[1]}/openai/v1")
    monkeypatch.setattr("truthlens.llm.CachedLLM.__init__.__defaults__", (str(tmp_path / "cache"),))
    monkeypatch.setattr("time.sleep", lambda s: None)  # retries must not slow the tests
    yield f
    srv.shutdown()
    srv.server_close()


def test_detection_order_and_explicit_choice(monkeypatch):
    assert detect_provider() is None
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    assert detect_provider() == "gemini"
    monkeypatch.setenv("GROQ_API_KEY", "q")
    assert detect_provider() == "groq"  # earlier in the list wins
    monkeypatch.setenv("TRUTHLENS_PROVIDER", "gemini")
    assert detect_provider() == "gemini"
    monkeypatch.setenv("TRUTHLENS_PROVIDER", "nope")
    with pytest.raises(LLMError, match="unknown"):
        detect_provider()
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setenv("TRUTHLENS_PROVIDER", "gemini")
    assert detect_provider() == "gemini"  # Google's own variable name works too


def test_no_key_is_a_readable_error_and_auto_falls_back_to_demo():
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        get_llm("live")
    assert get_llm("auto").model == "mock"


def test_request_shape_and_defaults(fake, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    llm = get_llm("live", cache=False)
    assert (llm.provider, llm.label, llm.model) == ("groq", "Groq", "llama-3.3-70b-versatile")
    assert llm.complete("SYS", "USER", task="generate") != ""
    req, hdr = fake.calls[-1], fake.headers[-1]
    assert fake.paths[-1] == "/openai/v1/chat/completions"
    assert req["messages"][0] == {"role": "system", "content": "SYS"} and req["max_tokens"] == 1024
    assert hdr["Authorization"] == "Bearer gsk_test" and "urllib" not in hdr["User-Agent"].lower()


def test_model_override(monkeypatch, fake):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("TRUTHLENS_MODEL", "gemini-9-flash")
    llm = get_llm("live", cache=False)
    llm.complete("s", "u")
    assert llm.provider == "gemini" and fake.calls[-1]["model"] == "gemini-9-flash"


def test_reasoning_text_is_stripped(fake, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    fake.mode = "think"
    out = get_llm("live", cache=False).complete("s", "u")
    assert "<think>" not in out and "reason about" not in out


@pytest.mark.parametrize("mode,status,text", [
    ("401", 401, "rejected the API key"), ("404", 400, "TRUTHLENS_MODEL"), ("500", 502, "API error"),
])
def test_failures_become_readable_errors(fake, monkeypatch, mode, status, text):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    fake.mode = mode
    with pytest.raises(LLMError, match=text) as ei:
        OpenAICompatLLM("groq", max_retries=0).complete("s", "u")
    assert ei.value.status == status


def test_short_rate_limit_is_retried_then_succeeds(fake, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    fake.mode = "429_once"
    assert OpenAICompatLLM("groq").complete("s", "u") != ""
    assert len(fake.calls) == 2


def test_daily_quota_fails_fast_with_the_providers_message(fake, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    fake.mode = "429_daily"
    with pytest.raises(LLMError, match="quota") as ei:
        OpenAICompatLLM("groq").complete("s", "u")
    assert ei.value.status == 429 and len(fake.calls) == 1  # no pointless retries


def test_newer_openai_models_get_max_completion_tokens(fake, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    fake.mode = "maxcompl"
    assert OpenAICompatLLM("openai").complete("s", "u") != ""
    assert "max_completion_tokens" in fake.calls[-1] and "max_tokens" not in fake.calls[-1]


def test_requests_are_paced_under_the_free_tier_limit(fake, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    slept = []
    monkeypatch.setattr("time.sleep", slept.append)
    a, b = OpenAICompatLLM("groq", rpm=30), OpenAICompatLLM("groq", rpm=30)  # two pipelines, one endpoint
    a.complete("s", "u"); b.complete("s", "u"); a.complete("s", "u")
    # 30 per minute = one call per 2 s, across both instances: calls 2 and 3 are scheduled 2 s and 4 s out
    assert len(slept) == 2 and 1.9 < slept[0] <= 2.0 and 3.9 < slept[1] <= 4.0


def test_full_pipeline_on_a_free_provider_key(fake, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    tl = build_pipeline("data/corpus", llm_kind="auto", verifier_names=["lexical", "llm_judge"], act="llm_judge")
    tl.cfg.action, tl.cfg.claim_mode = "regenerate", "llm"
    rec = tl.run(Q)
    assert rec["config"]["llm"] != "mock" and rec["rounds"] == 1
    assert "5G" not in rec["final"]["answer"] and rec["final"]["answer"].startswith("The standard warranty")


def test_server_reports_and_uses_the_free_provider(fake, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    srv = server.create_server("127.0.0.1", 0, "data/corpus")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]

    def call(method, path, body=None):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        c.request(method, path, json.dumps(body) if body else None, {"Content-Type": "application/json"})
        r = c.getresponse()
        return r.status, json.loads(r.read())

    try:
        _, meta = call("GET", "/api/meta")
        assert meta["live_ready"] and meta["provider_label"] == "Gemini" and meta["model"] == "gemini-2.5-flash"
        assert next(v for v in meta["verifiers"] if v["name"] == "llm_judge")["available"]
        # an old saved UI setting ("anthropic") still means "whichever real model is configured"
        st, rec = call("POST", "/api/ask", {"question": Q, "settings": {"llm": "anthropic", "verifiers": ["lexical", "llm_judge"], "act": "llm_judge"}})
        assert st == 200 and rec["ui"]["mock"] is False and rec["ui"]["provider"] == "Gemini"
    finally:
        srv.shutdown()
        srv.server_close()


def test_server_without_any_key_stays_in_demo_mode():
    app = server.App("data/corpus")
    m = app.meta()
    assert m["live_ready"] is False and m["provider"] is None
    assert not next(v for v in m["verifiers"] if v["name"] == "llm_judge")["available"]
