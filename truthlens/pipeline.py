from __future__ import annotations

from dataclasses import asdict, dataclass

from .claims import extract_claims, parse_answer, render_answer
from .generate import generate_answer, regenerate_answer
from .schema import Sentence

ABSTAIN = "The sources do not contain enough information to answer this question."


@dataclass
class PipelineConfig:
    k: int = 5  # chunks retrieved per question
    style: str = "strict"  # strict | permissive generation prompt
    claim_mode: str = "rule"  # rule | llm claim decomposition
    action: str = "flag"  # flag (only annotate) | remove (drop flagged sentences) | regenerate
    max_rounds: int = 2  # regeneration attempts before falling back to removal


class TruthLens:
    """retrieve -> generate (cited) -> split into claims -> verify -> flag / remove / regenerate.

    Several verifiers can score the same claims (for the baseline comparison); only the
    `act` verifier decides what gets removed or regenerated. Judge the result with a human
    label or a *different* verifier, never with the one that acted (see README).
    """

    def __init__(self, retriever, llm, verifiers: dict, act: str, cfg: PipelineConfig | None = None):
        if act not in verifiers:
            raise ValueError(f"act verifier {act!r} must be one of {list(verifiers)}")
        self.retriever, self.llm, self.verifiers, self.act = retriever, llm, verifiers, act
        self.cfg = cfg or PipelineConfig()

    # -- helpers
    def _verify(self, claims, chunks) -> dict:
        return {name: v.verify_all(claims, chunks) for name, v in self.verifiers.items()}

    def _flagged_ids(self, verdicts) -> set:
        return {v.claim_id for v in verdicts[self.act] if v.flagged}

    @staticmethod
    def _snap(sentences, claims, verdicts) -> dict:
        return {
            "answer": render_answer(sentences),
            "sentences": [asdict(s) for s in sentences],
            "claims": [asdict(c) for c in claims],
            "verdicts": {n: [asdict(v) for v in vs] for n, vs in verdicts.items()},
        }

    # -- main entry
    def run(self, question: str, answer_id: str = "q", split: str = "test") -> dict:
        cfg = self.cfg
        chunks = self.retriever.retrieve(question, cfg.k)
        raw = generate_answer(self.llm, question, chunks, cfg.style)
        sentences = parse_answer(raw)
        claims = extract_claims(sentences, cfg.claim_mode, self.llm)
        verdicts = self._verify(claims, chunks)
        baseline = self._snap(sentences, claims, verdicts)

        final, rounds, abstained = baseline, 0, False
        if cfg.action in ("remove", "regenerate"):
            cur_s, cur_c, cur_v = sentences, claims, verdicts
            if cfg.action == "regenerate":
                while rounds < cfg.max_rounds:
                    bad = self._flagged_ids(cur_v)
                    if not bad:
                        break
                    rounds += 1
                    new_raw = regenerate_answer(
                        self.llm, question, chunks, render_answer(cur_s),
                        [c.text for c in cur_c if c.id in bad],
                    )
                    cur_s = parse_answer(new_raw)
                    cur_c = extract_claims(cur_s, cfg.claim_mode, self.llm, prefix=f"g{rounds}.")
                    cur_v = self._verify(cur_c, chunks)

            # whatever is still flagged is dropped, sentence by sentence
            bad = self._flagged_ids(cur_v)
            bad_sents = {c.sentence_idx for c in cur_c if c.id in bad}
            keep_s = [s for s in cur_s if s.idx not in bad_sents]
            keep_c = [c for c in cur_c if c.sentence_idx not in bad_sents]
            keep_ids = {c.id for c in keep_c}
            keep_v = {n: [v for v in vs if v.claim_id in keep_ids] for n, vs in cur_v.items()}
            abstained = not keep_c
            if not keep_s:
                keep_s = [Sentence(idx=0, raw=ABSTAIN, text=ABSTAIN, cited_ids=[], is_refusal=True)]
            final = self._snap(keep_s, keep_c, keep_v)

        return {
            "answer_id": answer_id,
            "question": question,
            "split": split,
            "config": {
                **asdict(cfg), "act_verifier": self.act, "verifiers": list(self.verifiers),
                "llm": getattr(self.llm, "model", "?"),
            },
            "retrieved": [asdict(c) for c in chunks],
            "baseline": baseline,
            "final": final,
            "rounds": rounds,
            "abstained": abstained,
        }
