from __future__ import annotations

STRICT_SYSTEM = (
    "You answer questions using ONLY the numbered sources provided. Rules:\n"
    "1. Write plain prose, 1-6 short sentences, no markdown, no bullet points.\n"
    "2. End every sentence with the id(s) of the supporting source(s) in square brackets, placed before "
    "the final punctuation, e.g. 'The warranty lasts 10 years [c3].' or 'It weighs 98 kg [c1, c4].' "
    "Use only ids that appear in the sources.\n"
    "3. Prefer one fact per sentence.\n"
    "4. Do not use outside knowledge. If the sources do not contain the answer, or only part of it, say "
    "exactly what is missing in a sentence like 'The sources do not say how X works.' and do not guess."
)

# Deliberately lets the model add background knowledge. Use it to elicit more hallucinations when you
# need enough positive examples for the evaluation (and report that you did).
PERMISSIVE_SYSTEM = (
    "You are a helpful assistant answering questions about a product. Use the numbered sources where "
    "they are relevant and add useful background or general knowledge where it helps. Write plain "
    "prose, 2-6 short sentences, no markdown. Cite the supporting source id in square brackets before "
    "the final punctuation (e.g. 'It weighs 98 kg [c1].') for statements that come from the sources."
)

REGEN_SUFFIX = (
    "\nYou are revising an earlier answer. Some of its claims were NOT supported by the sources. "
    "Rewrite the answer so it contains only claims the sources support: fix a claim if the sources "
    "give the correct version, otherwise drop it. Keep the citation format. If nothing supported "
    "remains, say 'The sources do not say.'"
)

GEN_USER = "SOURCES:\n{sources}\n\nQUESTION: {question}\n\nANSWER:"
REGEN_USER = (
    "SOURCES:\n{sources}\n\nQUESTION: {question}\n\nPREVIOUS ANSWER:\n{previous}\n\n"
    "UNSUPPORTED CLAIMS:\n{flagged}\n\nREWRITTEN ANSWER:"
)


def _flatten(text: str) -> str:
    """One line per source. A chunk is "Heading\\nBody"; joining them with a plain space glues the
    heading onto the first sentence ("Standard warranty The standard warranty ...") and models copy
    that into their answers, so the heading is set off with a colon instead."""
    head, sep, body = text.partition("\n")
    return (f"{head.strip()}: {body}" if sep else head).replace("\n", " ")


def format_sources(chunks: list) -> str:
    return "\n".join(f"[{c.id}] ({c.doc}) {_flatten(c.text)}" for c in chunks)


def generate_answer(llm, question: str, chunks: list, style: str = "strict") -> str:
    system = STRICT_SYSTEM if style == "strict" else PERMISSIVE_SYSTEM
    user = GEN_USER.format(sources=format_sources(chunks), question=question)
    out = llm.complete(system, user, task="generate", max_tokens=600).strip()
    return out or "The sources do not say."


def regenerate_answer(llm, question: str, chunks: list, previous: str, flagged_claims: list) -> str:
    user = REGEN_USER.format(
        sources=format_sources(chunks),
        question=question,
        previous=previous,
        flagged="\n".join(f"- {c}" for c in flagged_claims),
    )
    out = llm.complete(STRICT_SYSTEM + REGEN_SUFFIX, user, task="regenerate", max_tokens=600).strip()
    return out or "The sources do not say."
