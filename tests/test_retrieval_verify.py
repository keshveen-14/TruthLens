import os

import pytest

from truthlens.corpus import load_corpus
from truthlens.retrieval import HybridRetriever
from truthlens.schema import CONTRADICTED, SUPPORTED, UNSUPPORTED, Chunk, Claim
from truthlens.util import tokenize
from truthlens.verify import LexicalVerifier

CORPUS = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")
CH = [
    Chunk("c0", "specs.md", "Specs\nThe H10 weighs 98 kg. The H15 weighs 138 kg. Noise is below 30 dB at one metre."),
    Chunk("c1", "warranty.md", "Warranty\nThe standard warranty lasts 10 years. The warranty cannot be extended beyond 15 years."),
    Chunk("c2", "connect.md", "Connectivity\nThe battery connects through Wi-Fi or Ethernet. It has no cellular modem."),
]


def claim(text, cited=("c0",)):
    return Claim("s0.0", text, 0, list(cited))


def test_tokenizer_normalises_numbers_and_codes():
    assert "6000" in tokenize("6,000 cycles") and "640000" in tokenize("6,40,000 rupees")
    assert "h10" in tokenize("The H10 model")
    assert "3.5" in tokenize("3.5 kWh")


def test_corpus_chunk_ids_are_stable_and_unique():
    a, b = load_corpus(CORPUS), load_corpus(CORPUS)
    assert [c.id for c in a] == [c.id for c in b]
    assert len({c.id for c in a}) == len(a) > 5


def test_bm25_finds_the_warranty_chunk():
    r = HybridRetriever(load_corpus(CORPUS))
    top = r.retrieve("How long is the standard warranty?", k=3)
    assert any("warranty" in c.doc for c in top[:2])


def test_lexical_supports_a_paraphrase():
    v = LexicalVerifier().verify(claim("The H10 battery weighs 98 kg."), CH)
    assert v.label == SUPPORTED and v.evidence_id == "c0" and v.cited_ok


def test_lexical_flags_wrong_number():
    v = LexicalVerifier().verify(claim("The H10 weighs 120 kg."), CH)
    assert v.label != SUPPORTED


def test_lexical_flags_unrelated_claim():
    v = LexicalVerifier().verify(claim("The battery ships to over 40 countries."), CH)
    assert v.label == UNSUPPORTED


def test_lexical_detects_negation_flip():
    v = LexicalVerifier().verify(claim("The battery has a cellular modem.", ("c2",)), CH)
    assert v.label == CONTRADICTED


def test_wrong_citation_is_supported_but_marked_not_cited_ok():
    v = LexicalVerifier().verify(claim("The standard warranty lasts 10 years.", cited=("c0",)), CH)
    assert v.label == SUPPORTED and v.evidence_id == "c1" and v.cited_ok is False


def test_llm_judge_needs_real_llm():
    from truthlens.llm import MockLLM
    from truthlens.verify import build_verifier

    with pytest.raises(RuntimeError):
        build_verifier("llm_judge", llm=MockLLM())
