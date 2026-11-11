from truthlens.claims import (
    atomic_claims_rule, extract_claims, parse_answer, parse_sentence, render_answer, split_sentences,
)


def test_split_keeps_decimals_and_abbreviations():
    s = split_sentences("The H15 outputs 7.5 kW [c1]. It works with e.g. rooftop solar [c2]. Warranty is long [c3].")
    assert len(s) == 3
    assert s[0].startswith("The H15 outputs 7.5 kW")
    assert "e.g." in s[1]


def test_citation_after_period_is_normalised():
    s = split_sentences("The warranty lasts 10 years. [c3] It starts at commissioning. [c4]")
    assert s == ["The warranty lasts 10 years [c3].", "It starts at commissioning [c4]."]


def test_parse_sentence_extracts_ids_and_cleans_text():
    s = parse_sentence(0, "The battery weighs 98 kg [c1, c4].")
    assert s.cited_ids == ["c1", "c4"]
    assert s.text == "The battery weighs 98 kg."


def test_bullets_and_bold_are_stripped():
    sents = parse_answer("- **Warranty** lasts 10 years [c1].\n- It is transferable [c2].")
    assert len(sents) == 2 and sents[0].text.startswith("Warranty lasts")


def test_render_roundtrip_keeps_citations():
    raw = "The battery weighs 98 kg [c1]. It has IP55 protection [c2]."
    assert render_answer(parse_answer(raw)) == raw


def test_refusal_is_not_a_claim():
    sents = parse_answer("The warranty lasts 10 years [c1]. The sources do not say how to claim it.")
    claims = extract_claims(sents)
    assert [c.sentence_idx for c in claims] == [0]
    assert sents[1].is_refusal


def test_atomic_split_carries_subject_over():
    out = atomic_claims_rule("The H10 weighs 98 kg, and it supports Wi-Fi and Ethernet.")
    assert out == ["The H10 weighs 98 kg.", "The H10 supports Wi-Fi and Ethernet."]
    out = atomic_claims_rule("The warranty lasts 10 years; it starts at commissioning.")
    assert len(out) == 2 and out[1].startswith("The warranty starts")


def test_atomic_split_is_conservative():
    # noun-phrase "and" must not be split; unclear subjects keep the sentence whole
    assert atomic_claims_rule("The H10 and the H15 use LFP cells.") == ["The H10 and the H15 use LFP cells."]
    s = "It has a 10 year warranty, and 6 months grace."
    assert atomic_claims_rule(s) == [s]


def test_claims_map_back_to_sentences_and_citations():
    sents = parse_answer("The H10 weighs 98 kg [c1], and it has IP55 protection [c2].")
    claims = extract_claims(sents)
    assert [c.id for c in claims] == ["s0.0", "s0.1"]
    assert all(c.sentence_idx == 0 and c.cited_ids == ["c1", "c2"] for c in claims)
