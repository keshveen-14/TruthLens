from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.request

from .config import CACHE_DIR, DEFAULT_MODEL, PROVIDERS


class LLM:
    model = "base"

    def complete(self, system: str, user: str, *, task: str = "", max_tokens: int = 1024) -> str:
        raise NotImplementedError


class LLMError(RuntimeError):
    """A readable failure from the real model backend (bad key, rate limit, outage, ...).

    Subclasses RuntimeError so existing handlers keep working; `status` is the HTTP status the web
    server should answer with.
    """

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


def provider_key(name: str) -> str | None:
    for k in PROVIDERS[name]["keys"]:
        if os.environ.get(k):
            return os.environ[k]
    return None


def detect_provider() -> str | None:
    """The provider to use for real answers, or None (then the offline demo is used).

    TRUTHLENS_PROVIDER picks one explicitly; otherwise the first key found wins, in the order of PROVIDERS.
    """
    want = os.environ.get("TRUTHLENS_PROVIDER", "").strip().lower()
    if want:
        if want not in PROVIDERS:
            raise LLMError(f"TRUTHLENS_PROVIDER={want!r} is unknown. Choose one of: {', '.join(PROVIDERS)}.", 400)
        return want if provider_key(want) else None
    return next((n for n in PROVIDERS if provider_key(n)), None)


def provider_info() -> dict:
    """What the UI shows: whether a real model is configured, and which one."""
    try:
        name = detect_provider()
    except LLMError:
        name = None
    if not name:
        return {"provider": None, "provider_label": None, "model": None}
    return {"provider": name, "provider_label": PROVIDERS[name]["label"],
            "model": os.environ.get("TRUTHLENS_MODEL") or PROVIDERS[name]["model"]}


def _retry_hint(headers, body: str) -> float | None:
    """Seconds the provider asks us to wait: Retry-After header, or 'try again in 6.5s' / '21m3s' in the message."""
    ra = headers.get("Retry-After") if headers else None
    if ra:
        try:
            return max(0.0, float(ra))
        except ValueError:
            pass
    m = re.search(r"try again in\s+(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", body, re.I)
    if m and any(m.groups()):
        h, mi, sec = (float(g) if g else 0.0 for g in m.groups())
        return h * 3600 + mi * 60 + sec
    return None


def _error_text(body: str) -> str:
    try:
        d = json.loads(body)
        d = d[0] if isinstance(d, list) and d else d
        e = d.get("error", d) if isinstance(d, dict) else d
        return str(e.get("message", e) if isinstance(e, dict) else e)[:300]
    except (ValueError, AttributeError):
        return body.strip()[:300]


class OpenAICompatLLM(LLM):
    """Any OpenAI-style /chat/completions endpoint: Groq, Gemini, OpenRouter, OpenAI, Ollama, ... (no SDK).

    Free tiers are tight, so calls are paced under the provider's requests-per-minute limit, short 429 waits
    are retried (honouring the provider's own hint), and a daily-quota 429 fails fast with a clear message.
    """

    def __init__(self, provider: str = "openai", model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, timeout: float = 90.0, max_retries: int = 3,
                 max_wait: float = 30.0, rpm: float | None = None):
        spec = PROVIDERS[provider]
        self.provider, self.label = provider, spec["label"]
        self.key = api_key or provider_key(provider)
        if not self.key:
            raise LLMError(f"{' or '.join(spec['keys'])} is not set. Set it, or choose the Demo model.", 400)
        self.base = (base_url or os.environ.get("TRUTHLENS_BASE_URL") or spec["base"]).rstrip("/")
        self.model = model or os.environ.get("TRUTHLENS_MODEL") or spec["model"]
        self.timeout, self.max_retries, self.max_wait = timeout, max_retries, max_wait
        if rpm is None:
            rpm = float(os.environ.get("TRUTHLENS_RPM", spec.get("rpm") or 0))
        self._gap = 60.0 / rpm if rpm else 0.0
        self._token_param = "max_tokens"

    # One clock per endpoint, shared by every instance: the server builds several pipelines (one per
    # settings combination) and together they must stay under the provider's per-minute limit.
    _clock_lock = threading.Lock()
    _clock: dict = {}

    def _pace(self) -> None:
        if not self._gap:
            return
        with OpenAICompatLLM._clock_lock:
            now = time.monotonic()
            nxt = OpenAICompatLLM._clock.get(self.base, 0.0)
            wait = max(0.0, nxt - now)
            OpenAICompatLLM._clock[self.base] = max(now, nxt) + self._gap
        if wait:
            time.sleep(wait)

    def _fail(self, status: int, detail: str) -> LLMError:
        who = self.label
        if status in (401, 403):
            return LLMError(f"{who} rejected the API key ({status}). Check the key. {detail}", status)
        if status == 404:
            return LLMError(f"{who}: model {self.model!r} was not found. Set TRUTHLENS_MODEL to a model your key can use. {detail}", 400)
        if status == 429:
            return LLMError(f"{who} rate limit or free-tier quota reached. {detail}", 429)
        if status == 400:
            return LLMError(f"{who} rejected the request (400): {detail}", 400)
        return LLMError(f"{who} API error ({status}). Try again shortly. {detail}", 502)

    def complete(self, system, user, *, task="", max_tokens=1024):
        swapped = False
        attempt = 0
        while True:
            payload = {"model": self.model, self._token_param: max_tokens,
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
            req = urllib.request.Request(
                f"{self.base}/chat/completions", data=json.dumps(payload).encode(), method="POST",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}",
                         "Accept": "application/json",
                         # some providers sit behind a firewall that blocks Python's default user agent
                         "User-Agent": "truthlens/1.0"})
            self._pace()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")
                detail = _error_text(body)
                if e.code == 400 and not swapped and "max_completion_tokens" in body and self._token_param == "max_tokens":
                    self._token_param, swapped = "max_completion_tokens", True  # newer OpenAI models
                    continue
                if e.code in (429, 500, 502, 503, 504, 529) and attempt < self.max_retries:
                    hint = _retry_hint(e.headers, body)
                    delay = hint if hint is not None else min(2.0 * 2 ** attempt, self.max_wait)
                    if delay <= self.max_wait:
                        attempt += 1
                        time.sleep(delay)
                        continue
                    detail += f" (the provider asks to wait about {int(delay)} s; a daily quota may be used up)"
                raise self._fail(e.code, detail) from e
            except (TimeoutError, socket.timeout) as e:
                if attempt < self.max_retries:
                    attempt += 1
                    continue
                raise LLMError(f"The request to {self.label} timed out. Try again.", 504) from e
            except urllib.error.URLError as e:
                if isinstance(e.reason, (TimeoutError, socket.timeout)):
                    if attempt < self.max_retries:
                        attempt += 1
                        continue
                    raise LLMError(f"The request to {self.label} timed out. Try again.", 504) from e
                if attempt < self.max_retries:
                    attempt += 1
                    time.sleep(min(2.0 * attempt, self.max_wait))
                    continue
                raise LLMError(f"Could not reach {self.label}. Check your network connection.", 502) from e
            except ValueError as e:  # not JSON
                raise LLMError(f"{self.label} returned something that is not JSON. Check TRUTHLENS_BASE_URL.", 502) from e

        choices = data.get("choices") or []
        msg = (choices[0].get("message") or {}) if choices else {}
        text = msg.get("content") or ""
        if isinstance(text, list):
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        # reasoning models (DeepSeek-R1, Qwen3, ...) put their thinking in the text; the answer follows it
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        return re.sub(r"<think>.*\Z", "", text, flags=re.S).strip()


class AnthropicLLM(LLM):
    def __init__(self, model: str | None = None, api_key: str | None = None, timeout: float = 90.0, max_retries: int = 3):
        try:
            import anthropic  # lazy: only needed with a real backend
        except ImportError as e:
            raise LLMError("The anthropic package is not installed. Run: pip install anthropic", 400) from e

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set. Set it, or choose the Demo model.", 400)
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=key, timeout=timeout, max_retries=max_retries)
        self.model = model or DEFAULT_MODEL
        self.provider, self.label = "anthropic", "Claude"

    def complete(self, system, user, *, task="", max_tokens=1024):
        # No sampling parameters: current models and SDK versions do not accept them. Rerun
        # reproducibility comes from CachedLLM (the on-disk cache), not from temperature.
        a = self._anthropic
        try:
            resp = self.client.messages.create(
                model=self.model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user}],
            )
        except a.AuthenticationError as e:
            raise LLMError("Anthropic rejected the API key (401). Check ANTHROPIC_API_KEY.", 401) from e
        except a.PermissionDeniedError as e:
            raise LLMError("This API key is not allowed to use that model (403).", 403) from e
        except a.NotFoundError as e:
            raise LLMError(f"Model {self.model!r} was not found (404). Set TRUTHLENS_MODEL to a valid model id.", 400) from e
        except a.RateLimitError as e:
            raise LLMError("Anthropic rate limit reached (429). Wait a moment and try again.", 429) from e
        except a.BadRequestError as e:
            raise LLMError(f"Anthropic rejected the request (400): {getattr(e, 'message', e)}", 400) from e
        except a.APITimeoutError as e:
            raise LLMError("The request to Anthropic timed out. Try again.", 504) from e
        except a.APIConnectionError as e:
            raise LLMError("Could not reach the Anthropic API. Check your network connection.", 502) from e
        except a.APIStatusError as e:  # 5xx, 529 overloaded, ...
            raise LLMError(f"Anthropic API error ({e.status_code}). Try again shortly.", 502) from e
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


class CachedLLM(LLM):
    """Disk cache keyed on (model, system, user, max_tokens): reruns are free and reproducible."""

    def __init__(self, inner: LLM, cache_dir: str = CACHE_DIR):
        self.inner, self.dir = inner, cache_dir
        self.model = inner.model
        self.provider, self.label = getattr(inner, "provider", ""), getattr(inner, "label", "")
        os.makedirs(cache_dir, exist_ok=True)

    def complete(self, system, user, *, task="", max_tokens=1024):
        key = hashlib.sha256(json.dumps([self.model, system, user, max_tokens]).encode()).hexdigest()
        path = os.path.join(self.dir, key + ".json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)["text"]
        text = self.inner.complete(system, user, task=task, max_tokens=max_tokens)
        if not text.strip():  # an empty reply (refusal, cut-off) should be retried next time, not replayed
            return text
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"text": text}, f)
        return text


# Sentences the mock "hallucinates". None of them appears in the sample corpus.
INJECTED_BANK = [
    "The unit also supports 5G cellular connectivity for remote monitoring.",
    "It is certified for use in marine and offshore environments.",
    "Helio ships to more than 40 countries worldwide.",
    "The battery can be upgraded to double its capacity with a firmware licence.",
]


class MockLLM(LLM):
    """Offline stand-in for smoke tests and the demo UI. NOT a real model.

    generate    -> first two sentences of the top-ranked source, cited; on about half of the
                   questions it appends a sentence from INJECTED_BANK with a fake citation
    regenerate  -> previous answer minus sentences that contain a flagged claim
    decompose   -> the sentence itself
    anything else (e.g. LLM-judge verification) -> error, needs a real backend
    """

    model = "mock"
    provider, label = "demo", "Demo"

    def __init__(self, inject_unsupported: bool = True):
        self.inject = inject_unsupported
        self.bodies: dict = {}

    def attach_chunks(self, chunks) -> None:
        """Remember each chunk's body without its heading line. The prompt flattens "Heading\\nBody"
        into one line, which would make the canned answer start with the heading glued to a sentence."""
        self.bodies = {c.id: c.text.split("\n", 1)[1] if "\n" in c.text else c.text for c in chunks}

    def complete(self, system, user, *, task="", max_tokens=1024):
        from .claims import split_sentences  # local import avoids a cycle at module load

        if task == "generate":
            q = re.search(r"QUESTION:\s*(.*?)\s*\n\s*ANSWER:", user, re.S)
            question = q.group(1) if q else ""
            src = re.search(r"^\[(c\d+)\] \([^)]*\) (.*)$", user, re.M)
            if not src:
                return "The sources do not say."
            cid, text = src.group(1), src.group(2)
            text = self.bodies.get(cid, text).replace("\n", " ")
            sents = [s for s in split_sentences(text) if s[-1:] in ".!?"][:2]
            out = [s[:-1] + f" [{cid}]" + s[-1] for s in sents]
            h = int(hashlib.md5(question.encode()).hexdigest(), 16)
            if self.inject and h % 2 == 0:
                bad = INJECTED_BANK[h % len(INJECTED_BANK)]
                out.append(bad[:-1] + f" [{cid}]" + bad[-1])
            return " ".join(out) or "The sources do not say."

        if task == "regenerate":
            prev = user.split("PREVIOUS ANSWER:")[1].split("UNSUPPORTED CLAIMS:")[0].strip()
            flagged = [
                l[2:].strip()
                for l in user.split("UNSUPPORTED CLAIMS:")[1].split("REWRITTEN ANSWER:")[0].splitlines()
                if l.startswith("- ")
            ]
            keep = []
            for s in split_sentences(prev):
                clean = re.sub(r"\s*\[c\d+(?:\s*,\s*c\d+)*\]", "", s)
                if not any(f and (f.rstrip(".") in clean or clean.rstrip(".") in f) for f in flagged):
                    keep.append(s)
            return " ".join(keep) or "The sources do not say."

        if task == "decompose":
            m = re.search(r"Sentence:\s*(.*)", user, re.S)
            return json.dumps([m.group(1).strip() if m else user])

        raise RuntimeError(
            f"MockLLM cannot do task {task!r}. Set an API key (GROQ_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY or ANTHROPIC_API_KEY) to use a real model."
        )


def get_llm(kind: str = "auto", model: str | None = None, cache: bool = True) -> LLM:
    """kind: auto (a real model if any API key is set, else the mock) | live (same, but fail if no key) |
    mock | a provider name (anthropic, groq, gemini, openrouter, openai)."""
    if kind == "auto":
        kind = "live" if detect_provider() else "mock"
    if kind == "mock":
        return MockLLM()
    if kind == "live":
        kind = detect_provider()
        if not kind:
            raise LLMError("No API key found. Set GROQ_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY or "
                           "ANTHROPIC_API_KEY (Groq and Gemini have free keys), or choose the Demo model.", 400)
    if kind == "anthropic":
        llm: LLM = AnthropicLLM(model=model)
    elif kind in PROVIDERS:
        llm = OpenAICompatLLM(kind, model=model)
    else:
        raise ValueError(f"unknown llm kind {kind!r}")
    return CachedLLM(llm) if cache else llm
