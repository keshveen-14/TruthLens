import os

# API model string; override with TRUTHLENS_MODEL.
DEFAULT_MODEL = os.environ.get("TRUTHLENS_MODEL", "claude-sonnet-5-5")

# On-disk cache of LLM calls, so evaluation reruns are free and reproducible.
CACHE_DIR = os.environ.get("TRUTHLENS_CACHE", ".cache/llm")


# Real-model providers. Anthropic uses its own SDK; the rest speak the OpenAI chat-completions protocol
# (plain HTTPS, no extra package). Free keys: Groq (console.groq.com/keys), Gemini (aistudio.google.com/apikey),
# OpenRouter (openrouter.ai/keys; models ending in ":free"). "rpm" paces requests under the free-tier limit.
# TRUTHLENS_MODEL overrides the model, TRUTHLENS_BASE_URL the endpoint, TRUTHLENS_PROVIDER picks one when
# several keys are set, TRUTHLENS_RPM overrides the pacing (0 turns it off).
PROVIDERS = {
    "anthropic": {"label": "Claude", "keys": ("ANTHROPIC_API_KEY",), "model": "claude-sonnet-5-5"},
    "groq": {"label": "Groq", "keys": ("GROQ_API_KEY",), "base": "https://api.groq.com/openai/v1",
             "model": "llama-3.3-70b-versatile", "rpm": 25},
    "gemini": {"label": "Gemini", "keys": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
               "base": "https://generativelanguage.googleapis.com/v1beta/openai",
               "model": "gemini-2.5-flash", "rpm": 9},
    "openrouter": {"label": "OpenRouter", "keys": ("OPENROUTER_API_KEY",), "base": "https://openrouter.ai/api/v1",
                   "model": "meta-llama/llama-3.3-70b-instruct:free", "rpm": 15},
    "openai": {"label": "OpenAI-compatible", "keys": ("OPENAI_API_KEY",),
               "base": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"), "model": "gpt-4o-mini"},
}
