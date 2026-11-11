"""A tiny local stand-in for the Anthropic Messages API, so the REAL `anthropic` SDK code path
(AnthropicLLM, LLM judge, claim splitting, regeneration, error mapping) can be tested offline.

It is NOT a language model: it answers from the sources with simple rules and can be told to fail.
Point the SDK at it with ANTHROPIC_BASE_URL.
"""
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BAD = "The unit also supports 5G cellular connectivity for remote monitoring."
WORD = re.compile(r"[a-z0-9]+")
STOP = set("a an the of to in on at for from by with and or is are was were be it its as that this".split())


def _toks(t):
    return {w for w in WORD.findall(t.lower()) if w not in STOP}


class Fake:
    def __init__(self):
        self.mode = "ok"  # ok | 401 | 404 | 429 | 500 | 529 | empty | temp_reject | 429_once | 429_daily | think | maxcompl
        self.paths = []
        self.headers = []
        self.calls = []

    # -- behaviours
    def reply(self, system, user):
        if "strict fact-checker" in system:
            claim = user.split("CLAIM:")[1].strip()
            sources = user.split("CLAIM:")[0]
            ok = _toks(claim) <= _toks(sources)
            label = "supported" if ok else "unsupported"
            cid = re.findall(r"\[(c\d+)\]", sources)
            return "```json\n" + json.dumps({"label": label, "evidence_ids": cid[:1], "reason": "stub"}) + "\n```"
        if system.startswith("Split the sentence"):
            return json.dumps([user.replace("Sentence:", "").strip()])
        if "revising an earlier answer" in system:
            prev = user.split("PREVIOUS ANSWER:")[1].split("UNSUPPORTED CLAIMS:")[0].strip()
            return " ".join(s for s in re.split(r"(?<=[.!?])\s+", prev) if "5G" not in s) or "The sources do not say."
        # generation
        m = re.search(r"^\[(c\d+)\] \([^)]*\) (.*)$", user, re.M)
        if not m:
            return "The sources do not say."
        cid, body = m.group(1), m.group(2)
        body = body.split(": ", 1)[1] if ": " in body.split(".")[0] else body  # drop the "Heading: " label
        sents = [x for x in re.split(r"(?<=[.!?])\s+", body) if x][:2]
        out = [x[:-1] + f" [{cid}]" + x[-1] for x in sents]
        if "warranty" in user.lower().split("question:")[-1]:
            out.append(BAD[:-1] + f" [{cid}]" + BAD[-1])
        return " ".join(out)


def start(fake: Fake):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, status, obj):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _openai(self, req):
            def err(st, msg, headers=None):
                body = json.dumps({"error": {"message": msg, "type": "x"}}).encode()
                self.send_response(st)
                self.send_header("Content-Type", "application/json")
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            if fake.mode == "401":
                return err(401, "Invalid API Key")
            if fake.mode == "404":
                return err(404, "The model `nope` does not exist")
            if fake.mode == "500":
                return err(500, "boom")
            if fake.mode == "429":
                return err(429, "slow down", {"Retry-After": "0"})
            if fake.mode == "429_daily":
                return err(429, "Rate limit reached on tokens per day (TPD). Please try again in 21m3.5s.")
            if fake.mode == "429_once" and len(fake.calls) == 1:
                return err(429, "Rate limit reached. Please try again in 0.01s.")
            if fake.mode == "maxcompl" and "max_tokens" in req:
                return err(400, "Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens' instead.")
            msgs = req["messages"]
            text = "" if fake.mode == "empty" else fake.reply(msgs[0]["content"], msgs[1]["content"])
            if fake.mode == "think":
                text = "<think>let me reason about this first</think>\n" + text
            self._send(200, {"id": "c1", "object": "chat.completion", "model": req["model"],
                             "choices": [{"index": 0, "finish_reason": "stop",
                                          "message": {"role": "assistant", "content": text}}]})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            fake.calls.append(req)
            fake.paths.append(self.path)
            fake.headers.append(dict(self.headers))
            if self.path.endswith("/chat/completions"):
                return self._openai(req)
            errs = {"401": (401, "authentication_error", "invalid x-api-key"),
                    "404": (404, "not_found_error", "model: nope"),
                    "429": (429, "rate_limit_error", "slow down"),
                    "500": (500, "api_error", "boom"),
                    "529": (529, "overloaded_error", "Overloaded")}
            if fake.mode in errs:
                st, t, msg = errs[fake.mode]
                return self._send(st, {"type": "error", "error": {"type": t, "message": msg}})
            if fake.mode == "temp_reject" and "temperature" in req:
                return self._send(400, {"type": "error", "error": {
                    "type": "invalid_request_error", "message": "`temperature` is not supported for this model"}})
            text = "" if fake.mode == "empty" else fake.reply(req.get("system", ""), req["messages"][0]["content"])
            content = [{"type": "text", "text": text}] if text else []
            self._send(200, {"id": "msg_1", "type": "message", "role": "assistant", "model": req["model"],
                             "content": content, "stop_reason": "end_turn", "stop_sequence": None,
                             "usage": {"input_tokens": 1, "output_tokens": 1}})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
