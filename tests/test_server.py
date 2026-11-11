"""The web UI's server: API contract, input validation and static-file safety. Uses the offline demo model."""

import http.client
import json
import threading

import pytest

import server


@pytest.fixture(scope="module")
def conn_factory():
    srv = server.create_server("127.0.0.1", 0, "data/corpus")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    yield lambda: http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    srv.shutdown()
    srv.server_close()


def call(conn_factory, method, path, body=None, headers=None):
    c = conn_factory()
    h = dict(headers or {})
    data = None
    if body is not None:
        data = body if isinstance(body, (bytes, str)) else json.dumps(body)
        h.setdefault("Content-Type", "application/json")
    c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    raw = r.read()
    c.close()
    return r.status, r.getheader("Content-Type", ""), raw


def ask(conn_factory, question="How long is the standard warranty and when does it start?", **settings):
    return call(conn_factory, "POST", "/api/ask", {"question": question, "settings": {"llm": "mock", **settings}})


def test_meta_lists_verifiers_questions_and_documents(conn_factory):
    status, _, raw = call(conn_factory, "GET", "/api/meta")
    meta = json.loads(raw)
    assert status == 200
    assert [v["name"] for v in meta["verifiers"]] == ["lexical", "embedding", "nli", "llm_judge"]
    assert meta["verifiers"][0]["available"] is True
    assert len(meta["questions"]) == 30 and len(meta["documents"]) == 6


def test_ask_returns_the_pipeline_record_plus_ui_fields(conn_factory):
    status, _, raw = ask(conn_factory)
    rec = json.loads(raw)
    assert status == 200
    assert {"question", "retrieved", "baseline", "final", "abstained", "ui"} <= set(rec)
    assert rec["ui"]["mock"] is True and rec["ui"]["act"] == "lexical" and rec["ui"]["thresholds"]["lexical"] == 0.7
    verdicts = rec["baseline"]["verdicts"]["lexical"]
    assert len(verdicts) == len(rec["baseline"]["claims"]) >= 2
    assert any(v["label"] != "supported" for v in verdicts)  # the demo model injects a false sentence here
    # the demo answer must not start with a glued-on section heading
    assert rec["baseline"]["sentences"][0]["text"] == "The standard warranty lasts 10 years."


def test_remove_action_drops_the_flagged_sentence_from_the_final_answer(conn_factory):
    rec = json.loads(ask(conn_factory, action="remove")[2])
    before = [s["text"] for s in rec["baseline"]["sentences"]]
    after = [s["text"] for s in rec["final"]["sentences"]]
    flagged = {c["sentence_idx"] for c, v in zip(rec["baseline"]["claims"], rec["baseline"]["verdicts"]["lexical"])
               if v["label"] != "supported"}
    assert len(after) == len(before) - len(flagged) and flagged


@pytest.mark.parametrize("payload,fragment", [
    ({"question": "   "}, "Enter a question"),
    ({"question": "x" * 501}, "limited to 500"),
    ({"question": "ok?", "settings": {"llm": "mock", "verifiers": ["lexical", "llm_judge"]}}, "ANTHROPIC_API_KEY"),
    ({"question": "ok?", "settings": "nope"}, "settings must be an object"),
    ({"question": "ok?", "settings": {"verifiers": ["bogus"]}}, "at least one verifier"),
])
def test_bad_requests_get_a_plain_message(conn_factory, payload, fragment):
    status, ctype, raw = call(conn_factory, "POST", "/api/ask", payload)
    assert status == 400 and "application/json" in ctype
    assert fragment in json.loads(raw)["error"]


def test_unknown_settings_values_fall_back_to_defaults(conn_factory):
    rec = json.loads(ask(conn_factory, action="explode", style="wild", k=999)[2])
    assert rec["ui"]["action"] == "remove" and len(rec["retrieved"]) <= 10


def test_post_needs_json_and_valid_body(conn_factory):
    assert call(conn_factory, "POST", "/api/ask", "question=hi", {"Content-Type": "text/plain"})[0] == 415
    assert call(conn_factory, "POST", "/api/ask", b"{not json")[0] == 400
    assert call(conn_factory, "POST", "/api/ask", "[1, 2]")[0] == 400
    assert call(conn_factory, "POST", "/api/ask", b"{" + b" " * 40_000 + b"}")[0] == 413
    assert call(conn_factory, "POST", "/api/nope", {})[0] == 404


def test_static_files_are_served_with_security_headers(conn_factory):
    status, ctype, raw = call(conn_factory, "GET", "/")
    assert status == 200 and "text/html" in ctype and b"TruthLens" in raw
    assert call(conn_factory, "GET", "/app.js")[1].startswith("text/javascript")
    assert call(conn_factory, "GET", "/styles.css")[1].startswith("text/css")
    c = conn_factory()
    c.request("GET", "/")
    r = c.getresponse()
    r.read()
    assert r.getheader("X-Content-Type-Options") == "nosniff" and "default-src 'self'" in r.getheader("Content-Security-Policy")


@pytest.mark.parametrize("path", ["/../server.py", "/%2e%2e/server.py", "/..%2fserver.py", "/web/../../README.md", "/nope.txt"])
def test_static_handler_cannot_escape_the_web_folder(conn_factory, path):
    status, _, raw = call(conn_factory, "GET", path)
    assert status == 404 and b"TruthLens: citation" not in raw and b"def create_server" not in raw


def test_rate_limiter_blocks_after_the_hourly_limit_and_daily_cap():
    rl = server.RateLimiter(per_hour=2, daily=3)
    rl.check("a"); rl.check("a")
    with pytest.raises(server.ApiError) as e:
        rl.check("a")
    assert e.value.status == 429 and "per hour" in e.value.message
    rl.check("b")  # another visitor is unaffected, but takes the 3rd of the daily 3
    with pytest.raises(server.ApiError) as e:
        rl.check("c")
    assert "daily limit" in e.value.message


def test_verify_checks_pasted_text_against_the_whole_corpus(conn_factory):
    status, _, raw = call(conn_factory, "POST", "/api/verify", {
        "answer": "The standard warranty lasts 10 years. Helio ships to more than 40 countries worldwide.",
        "settings": {"llm": "mock", "verifiers": ["lexical"]}})
    rec = json.loads(raw)
    assert status == 200 and rec["ui"]["mode"] == "audit"
    labels = {c["text"]: v["label"] for c, v in zip(rec["baseline"]["claims"], rec["baseline"]["verdicts"]["lexical"])}
    assert labels["The standard warranty lasts 10 years."] == "supported"
    assert labels["Helio ships to more than 40 countries worldwide."] != "supported"
    assert len(rec["retrieved"]) > 5  # every chunk was available as evidence, not just the top few


@pytest.mark.parametrize("payload", [{}, {"answer": "   "}, {"answer": "x" * 6001}])
def test_verify_rejects_empty_or_oversized_text(conn_factory, payload):
    assert call(conn_factory, "POST", "/api/verify", payload)[0] == 400


def test_new_front_end_files_are_served(conn_factory):
    for name in ("/engine.js", "/data.js", "/app.js"):
        status, ctype, raw = call(conn_factory, "GET", name)
        assert status == 200 and ctype.startswith("text/javascript") and len(raw) > 1000
