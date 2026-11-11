"""TruthLens web UI: a small local server, standard library only.

    python server.py                 # http://127.0.0.1:8000
    python server.py --port 9000 --corpus data/corpus --no-browser

Without an API key (GROQ_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY or ANTHROPIC_API_KEY) the demo backend is used. It returns canned answers and injects
false sentences on purpose, so the detector has something to catch.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import mimetypes
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from truthlens.factory import build_pipeline  # noqa: E402
from truthlens.llm import LLMError, detect_provider, provider_info  # noqa: E402
from truthlens.pipeline import PipelineConfig  # noqa: E402

WEB = ROOT / "web"
MAX_BODY = 32_768
MAX_QUESTION = 500
VERIFIERS = ("lexical", "embedding", "nli", "llm_judge")
DEFAULT_QUESTION = "How long is the standard warranty and when does it start?"


PUBLIC = os.environ.get("TRUTHLENS_PUBLIC", "").lower() in {"1", "true", "yes"}
RATE_PER_HOUR = int(os.environ.get("TRUTHLENS_RATE_PER_HOUR", "12"))  # real-model questions per visitor
DAILY_CAP = int(os.environ.get("TRUTHLENS_DAILY_CAP", "300"))  # real-model questions per day, all visitors


class RateLimiter:
    """Protects your API key on a public deployment: per-visitor hourly limit plus a global daily cap."""

    def __init__(self, per_hour: int, daily: int):
        self.per_hour, self.daily = per_hour, daily
        self.hits: dict = {}
        self.day, self.day_count = time.strftime("%Y-%m-%d"), 0
        self.lock = threading.Lock()

    def check(self, ip: str) -> None:
        now = time.time()
        with self.lock:
            today = time.strftime("%Y-%m-%d")
            if today != self.day:
                self.day, self.day_count = today, 0
            if self.day_count >= self.daily:
                raise ApiError(429, "The demo has reached its daily limit for real-model questions. Try again tomorrow, or switch to the Demo model in Settings.")
            recent = [t for t in self.hits.get(ip, []) if now - t < 3600]
            if len(recent) >= self.per_hour:
                raise ApiError(429, f"Limit of {self.per_hour} real-model questions per hour reached. Try again later, or switch to the Demo model in Settings.")
            recent.append(now)
            self.hits[ip] = recent
            self.day_count += 1
            if len(self.hits) > 5000:  # forget visitors who have gone quiet
                self.hits = {k: v for k, v in self.hits.items() if v and now - v[-1] < 3600}


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def TruthLensSnap(sentences, claims, verdicts, render):
    from dataclasses import asdict  # noqa: PLC0415

    return {
        "answer": render(sentences),
        "sentences": [asdict(x) for x in sentences],
        "claims": [asdict(c) for c in claims],
        "verdicts": {n: [asdict(v) for v in vs] for n, vs in verdicts.items()},
    }


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _choice(value, allowed, default):
    return value if value in allowed else default


class App:
    """Holds built pipelines (building loads the corpus and any models, so it is cached)."""

    def __init__(self, corpus: str):
        self.corpus = corpus
        self._pipelines: dict = {}
        self._lock = threading.Lock()
        self.limiter = RateLimiter(RATE_PER_HOUR, DAILY_CAP) if PUBLIC else None

    # -- GET /api/meta
    def meta(self) -> dict:
        info = provider_info()
        live_ready = bool(info["provider"]) and (info["provider"] != "anthropic" or _has("anthropic"))
        return {
            "live_ready": live_ready,
            "claude_ready": live_ready,  # old name, kept for older front ends
            **info,
            "verifiers": [
                {"name": "lexical", "label": "Lexical match", "available": True,
                 "note": "Fast baseline. Compares words and numbers, no model needed."},
                {"name": "embedding", "label": "Embedding similarity", "available": _has("sentence_transformers"),
                 "note": "Needs sentence-transformers."},
                {"name": "nli", "label": "NLI model", "available": _has("transformers") and _has("torch"),
                 "note": "Needs torch and transformers."},
                {"name": "llm_judge", "label": "LLM judge", "available": live_ready,
                 "note": "Needs an API key (Groq and Gemini have free ones)."},
            ],
            "questions": self._questions(),
            "default_question": DEFAULT_QUESTION,
            "documents": sorted(p.name for p in Path(self.corpus).glob("*") if p.suffix in {".md", ".txt"}),
        }

    def _questions(self) -> list:
        path = ROOT / "data" / "questions.jsonl"
        if not path.exists():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
        return out

    # -- POST /api/verify
    def verify(self, payload: dict, ip: str = "local") -> dict:
        """Check an answer someone else wrote (pasted text) against every chunk of the corpus."""
        from truthlens.claims import extract_claims, parse_answer, render_answer  # noqa: PLC0415
        from dataclasses import asdict  # noqa: PLC0415

        answer = str(payload.get("answer", "")).strip()
        if not answer:
            raise ApiError(400, "Paste an answer to check first.")
        if len(answer) > 6000:
            raise ApiError(400, "Answers are limited to 6000 characters.")
        s = payload.get("settings") or {}
        names = [n for n in (s.get("verifiers") or ["lexical"]) if n in VERIFIERS] or ["lexical"]
        llm_kind = _choice(s.get("llm"), ("auto", "live", "anthropic", "mock"), "auto")
        if llm_kind == "anthropic":
            llm_kind = "live"
        act = _choice(s.get("act"), names, names[0])
        uses_live = "llm_judge" in names
        if self.limiter and uses_live:
            self.limiter.check(ip)
        key = ("verify", llm_kind, tuple(names), act)
        started = time.perf_counter()
        with self._lock:
            try:
                tl = self._pipelines.get(key)
                if tl is None:
                    tl = build_pipeline(self.corpus, llm_kind=llm_kind, verifier_names=list(names), act=act,
                                        cfg=PipelineConfig(action="flag"))
                    self._pipelines[key] = tl
                chunks = tl.retriever.chunks
                sentences = parse_answer(answer)
                claims = extract_claims(sentences, "rule", tl.llm)
                verdicts = tl._verify(claims, chunks)
            except LLMError as e:
                raise ApiError(e.status, str(e)) from e
            except (ValueError, RuntimeError, ImportError, OSError) as e:
                raise ApiError(400, str(e)) from e
        snap = TruthLensSnap(sentences, claims, verdicts, render_answer)
        rec = {
            "answer_id": "audit", "question": "", "split": "test",
            "config": {"act_verifier": act, "verifiers": names, "llm": getattr(tl.llm, "model", "?")},
            "retrieved": [asdict(c) for c in chunks], "baseline": snap, "final": snap, "rounds": 0, "abstained": False,
            "ui": {"elapsed_ms": round((time.perf_counter() - started) * 1000), "mock": False, "mode": "audit",
                   "model": getattr(tl.llm, "model", "?"), "provider": "Pasted", "action": "flag", "act": act,
                   "thresholds": {n: getattr(v, "threshold", None) for n, v in tl.verifiers.items()}},
        }
        return rec

    # -- POST /api/ask
    def ask(self, payload: dict, ip: str = "local") -> dict:
        question = str(payload.get("question", "")).strip()
        if not question:
            raise ApiError(400, "Enter a question first.")
        if len(question) > MAX_QUESTION:
            raise ApiError(400, f"Questions are limited to {MAX_QUESTION} characters.")

        s = payload.get("settings") or {}
        if not isinstance(s, dict):
            raise ApiError(400, "settings must be an object.")
        names = []
        for n in s.get("verifiers") or ["lexical"]:
            if n in VERIFIERS and n not in names:
                names.append(n)
        if not names:
            raise ApiError(400, "Pick at least one verifier in Settings.")
        llm_kind = _choice(s.get("llm"), ("auto", "live", "anthropic", "mock"), "auto")
        if llm_kind == "anthropic":  # the old UI value; it now means "whichever real model is configured"
            llm_kind = "live"
        act = _choice(s.get("act"), names, names[0])
        action = _choice(s.get("action"), ("flag", "remove", "regenerate"), "remove")
        style = _choice(s.get("style"), ("strict", "permissive"), "strict")
        claim_mode = _choice(s.get("claim_mode"), ("rule", "llm"), "rule")
        try:
            k = max(2, min(10, int(s.get("k", 5))))
        except (TypeError, ValueError):
            k = 5

        uses_live = llm_kind == "live" or (llm_kind == "auto" and detect_provider() is not None)
        if self.limiter and uses_live:
            self.limiter.check(ip)

        key = (llm_kind, tuple(names), act, action, style, claim_mode, k)
        started = time.perf_counter()
        with self._lock:  # one request at a time: pipelines and local models are not built for concurrency
            try:
                tl = self._pipelines.get(key)
                if tl is None:
                    cfg = PipelineConfig(k=k, style=style, claim_mode=claim_mode, action=action)
                    tl = build_pipeline(self.corpus, llm_kind=llm_kind, verifier_names=list(names), act=act, cfg=cfg)
                    self._pipelines[key] = tl
                rec = tl.run(question)
            except LLMError as e:
                raise ApiError(e.status, str(e)) from e
            except FileNotFoundError as e:
                raise ApiError(400, str(e)) from e
            except OSError as e:  # e.g. an NLI / embedding model that cannot be downloaded (offline)
                raise ApiError(
                    400, f"A model file could not be loaded ({type(e).__name__}). If you are offline, "
                         f"pick a verifier that needs no download. Details: {str(e)[:200]}") from e
            except (ValueError, RuntimeError, ImportError) as e:
                raise ApiError(400, str(e)) from e
        rec["ui"] = {
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
            "mock": getattr(tl.llm, "model", "") == "mock",
            "model": getattr(tl.llm, "model", "?"),
            "provider": getattr(tl.llm, "label", ""),
            "action": action,
            "act": act,
            "thresholds": {n: getattr(v, "threshold", None) for n, v in tl.verifiers.items()},
        }
        return rec


SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data:; style-src 'self' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; frame-ancestors 'none'"
    ),
}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "TruthLens"

        def log_message(self, fmt, *args):  # quiet: one short line per API call
            if self.path.startswith("/api/"):
                sys.stderr.write(f"{self.command} {self.path} -> {args[1] if len(args) > 1 else ''}\n")

        def _send(self, status: int, body: bytes, content_type: str, extra: dict | None = None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in {**SECURITY_HEADERS, **(extra or {})}.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, obj):
            self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _static(self, url_path: str):
            rel = unquote(url_path).lstrip("/") or "index.html"
            target = (WEB / rel).resolve()
            if WEB.resolve() not in target.parents or not target.is_file():
                return self._json(404, {"error": "Not found."})
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype in ("application/javascript", "image/svg+xml"):
                ctype += "; charset=utf-8"
            self._send(200, target.read_bytes(), ctype)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/api/meta":
                return self._json(200, app.meta())
            if path.startswith("/api/"):
                return self._json(404, {"error": "Not found."})
            self._static(path)

        def do_POST(self):
            route = urlparse(self.path).path
            if route not in ("/api/ask", "/api/verify"):
                return self._json(404, {"error": "Not found."})
            try:
                if "application/json" not in (self.headers.get("Content-Type") or ""):
                    raise ApiError(415, "Send the request as application/json.")
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    raise ApiError(400, "Invalid Content-Length.")
                if length > MAX_BODY:
                    raise ApiError(413, "Request is too large.")
                try:
                    payload = json.loads(self.rfile.read(length) or b"{}")
                except ValueError:
                    raise ApiError(400, "Request body is not valid JSON.")
                if not isinstance(payload, dict):
                    raise ApiError(400, "Request body must be a JSON object.")
                ip = self.client_address[0]
                if PUBLIC:  # behind the host's proxy the real visitor is the first X-Forwarded-For entry
                    ip = (self.headers.get("X-Forwarded-For") or ip).split(",")[0].strip()
                self._json(200, app.verify(payload, ip) if route == "/api/verify" else app.ask(payload, ip))
            except ApiError as e:
                self._json(e.status, {"error": e.message})
            except Exception as e:  # keep the server alive; show the cause in the UI
                self._json(500, {"error": f"Unexpected error: {type(e).__name__}: {e}"})

    return Handler


def create_server(host: str = "127.0.0.1", port: int = 8000, corpus: str = "data/corpus") -> ThreadingHTTPServer:
    corpus_path = Path(corpus)
    if not corpus_path.is_absolute():
        corpus_path = ROOT / corpus_path
    return ThreadingHTTPServer((host, port), make_handler(App(str(corpus_path))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    on_host = bool(os.environ.get("PORT"))  # hosting platforms (Render, Railway, Fly, Spaces) set PORT
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0" if on_host else "127.0.0.1"),
                    help="use 0.0.0.0 only on a network you trust")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    if on_host:
        a.no_browser = True

    server = create_server(a.host, a.port, a.corpus)
    url = f"http://{a.host if a.host != '0.0.0.0' else '127.0.0.1'}:{server.server_address[1]}"
    print(f"TruthLens running at {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
