from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

SUPPORTED = "supported"
UNSUPPORTED = "unsupported"
CONTRADICTED = "contradicted"


@dataclass
class Chunk:
    id: str  # stable id such as "c12"
    doc: str  # source file name
    text: str


@dataclass
class Sentence:
    idx: int
    raw: str  # as generated, with [cN] markers before the final punctuation
    text: str  # citation-free text
    cited_ids: list = field(default_factory=list)
    is_refusal: bool = False  # "the sources do not say ..." is not a checkable claim


@dataclass
class Claim:
    id: str  # "s2.1" = sentence 2, claim 1 (regenerated answers: "g1.s2.1")
    text: str
    sentence_idx: int
    cited_ids: list = field(default_factory=list)


@dataclass
class Verdict:
    claim_id: str
    verifier: str
    label: str  # supported | unsupported | contradicted
    support_score: float  # higher = better supported; flagged iff below the verifier's threshold
    evidence_id: Optional[str] = None  # chunk that best supports the claim
    cited_ok: Optional[bool] = None  # was that chunk among the ones the answer cited?
    rationale: str = ""

    @property
    def flagged(self) -> bool:
        return self.label != SUPPORTED


def to_dict(obj) -> dict:
    return asdict(obj)
