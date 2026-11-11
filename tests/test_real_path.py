"""The Claude code path, exercised with the real `anthropic` SDK against a local fake API."""
import http.client
import json
import threading

import pytest

pytest.importorskip("anthropic")

import server
from tests.fake_anthropic import Fake, start
from truthlens.factory import build_pipeline
from truthlens.llm import AnthropicLLM, LLMError, get_llm

Q = "How long is the standard warranty and when does it start?"


@pytest.fixture()
def fake(monkeypatch, tmp_path):
    f = Fake()
    srv = start(f)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{srv.server_address[1]}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr("truthlens.llm.CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr("truthlens.llm.CachedLLM.__init__.__defaults__", (str(tmp_path / "cache"),))
    yield f
    srv.shutdown()
    srv.server_close()


def test_missing_key_is_a_readable_error_not_a_typeerror(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        AnthropicLLM()


@pytest.mark.parametrize("mode,status,text", [
    ("401", 401, "rejected the API key"), ("404", 400, "not found"), ("429", 429, "rate limit"),
    ("500", 502, "API error"), ("529", 502, "API error"),
])
def test_api_failures_map_to_readable_errors(fake, mode, status, text):
    fake.mode = mode
    with pytest.raises(LLMError, match=text) as ei:
        AnthropicLLM(max_retries=0).complete("s", "u")
    assert ei.value.status == status


def test_no_sampling_parameters_are_sent(fake):
    AnthropicLLM(max_retries=0).complete("s", "u", task="judge")
    assert "temperature" not in fake.calls[-1] and fake.calls[-1]["model"] == "claude-sonnet-5-5"


def test_full_pipeline_with_claude_backend_and_llm_judge(fake):
    tl = build_pipeline("data/corpus", llm_kind="anthropic", verifier_names=["lexical", "llm_judge"], act="llm_judge")
    tl.cfg.action, tl.cfg.claim_mode = "remove", "llm"
    rec = tl.run(Q)
    assert rec["config"]["llm"] != "mock"
    flagged = [v for v in rec["baseline"]["verdicts"]["llm_judge"] if v["label"] != "supported"]
    assert len(flagged) == 1  # the injected 5G sentence
    assert all("5G" not in s["text"] for s in rec["final"]["sentences"])
    assert any(s["text"].startswith("The standard warranty") for s in rec["final"]["sentences"])


def test_regenerate_with_claude_backend(fake):
    tl = build_pipeline("data/corpus", llm_kind="anthropic", verifier_names=["llm_judge"], act="llm_judge")
    tl.cfg.action = "regenerate"
    rec = tl.run(Q)
    assert rec["rounds"] == 1 and "5G" not in rec["final"]["answer"]


def test_empty_model_reply_becomes_an_abstention_not_a_crash(fake):
    fake.mode = "empty"
    tl = build_pipeline("data/corpus", llm_kind="anthropic", verifier_names=["lexical"])
    rec = tl.run(Q)
    assert rec["baseline"]["claims"] == [] and "do not say" in rec["baseline"]["answer"]


def test_empty_replies_are_not_cached(fake, tmp_path):
    fake.mode = "empty"
    llm = get_llm("anthropic")
    llm.complete("s", "u")
    fake.mode = "ok"
    n = len(fake.calls)
    llm.complete("s", "u")
    assert len(fake.calls) == n + 1


def test_server_returns_claude_answer_and_clean_errors(fake):
    srv = server.create_server("127.0.0.1", 0, "data/corpus")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]

    def post(settings):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        c.request("POST", "/api/ask", json.dumps({"question": Q, "settings": settings}),
                  {"Content-Type": "application/json"})
        r = c.getresponse()
        return r.status, json.loads(r.read())

    try:
        st, rec = post({"llm": "auto", "verifiers": ["lexical", "llm_judge"], "act": "llm_judge"})
        assert st == 200 and rec["ui"]["mock"] is False and rec["ui"]["model"] == "claude-sonnet-5-5"
        fake.mode = "401"
        st, err = post({"llm": "anthropic", "verifiers": ["lexical"], "style": "permissive"})
        assert st == 401 and "API key" in err["error"]
    finally:
        srv.shutdown()
        srv.server_close()


def test_source_headings_are_not_glued_to_the_first_sentence():
    from truthlens.generate import format_sources
    from truthlens.schema import Chunk

    out = format_sources([Chunk("c1", "w.md", "Standard warranty\nThe standard warranty lasts 10 years."),
                          Chunk("c2", "w.md", "No heading here.")])
    assert "[c1] (w.md) Standard warranty: The standard warranty lasts 10 years." in out
    assert "[c2] (w.md) No heading here." in out


def test_public_mode_limits_claude_questions_but_not_the_demo(fake, monkeypatch):
    monkeypatch.setattr(server, "PUBLIC", True)
    monkeypatch.setattr(server, "RATE_PER_HOUR", 1)
    srv = server.create_server("127.0.0.1", 0, "data/corpus")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]

    def post(llm):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        c.request("POST", "/api/ask", json.dumps({"question": Q, "settings": {"llm": llm}}),
                  {"Content-Type": "application/json"})
        return c.getresponse().status

    try:
        assert post("anthropic") == 200
        assert post("anthropic") == 429
        assert post("mock") == 200  # the offline demo costs nothing, so it is never limited
    finally:
        srv.shutdown()
        srv.server_close()
