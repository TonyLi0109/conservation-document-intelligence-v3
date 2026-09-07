"""Optional semantic-judge extension; default evaluation never invokes a model."""

from typing import Literal, Protocol, Sequence

from data_models import KnowledgeArtifact

Support = Literal["supported", "partially_supported", "unsupported"]


class SemanticJudge(Protocol):
    def __call__(self, claim: str, evidence: Sequence[KnowledgeArtifact]) -> Support: ...


def assess_semantic_support(payload, artifacts, *, judge: SemanticJudge | None = None):
    """Call only an explicitly supplied judge, independently of deterministic rates.

    No provider is chosen or imported here. External implementations own their
    request budget, consent and reproducibility metadata; normal tests use a fake.
    """
    if judge is None:
        return {"status": "not_assessed", "claims": []}
    results = []
    for claim in payload.get("claims", []):
        handles = claim.get("evidence_ids", [])
        if not handles or any(handle not in artifacts for handle in handles):
            results.append("unsupported")
            continue
        outcome = judge(claim["text"], [artifacts[handle] for handle in handles])
        if outcome not in {"supported", "partially_supported", "unsupported"}:
            raise ValueError("Judge returned an unknown support assessment")
        results.append(outcome)
    return {"status": "assessed", "claims": results}
