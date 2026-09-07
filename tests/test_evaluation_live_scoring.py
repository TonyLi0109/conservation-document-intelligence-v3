"""Optional live-run scoring tested with scripted calls; no provider requests."""

import json

import pytest

from data_models import KnowledgeArtifact
from evaluation.run import run_live_answers
from validator import validate_render_and_collect_sources


@pytest.fixture
def evidence():
    return KnowledgeArtifact("DOC901", "Carp harvest study", "2",
                             "Targeted harvest reduced carp biomass by 60%.",
                             source_url="https://example.org/carp.pdf")


@pytest.mark.parametrize("percentage,passes", [("90%", False), ("60%", True)])
def test_live_raw_numeric_errors_affect_report(monkeypatch, evidence, percentage, passes):
    import main
    artifacts = {"K1": evidence}
    payload = {"status": "answered", "claims": [{
        "text": f"Targeted harvest reduced carp biomass by {percentage}.",
        "evidence_ids": ["K1"], "supporting_spans": [evidence.original_text_chunk],
    }], "unsupported_facets": []}
    monkeypatch.setattr(main, "call_llm", lambda *args, **kwargs: json.dumps(payload))

    def answer(*args, **kwargs):
        raw = main.call_llm("system", "prompt", artifacts)
        text, sources = validate_render_and_collect_sources(raw, artifacts)
        return text, None, sources

    monkeypatch.setattr(main, "ask_chatbot_with_context", answer)
    result = run_live_answers(None, [{"case_id": "numeric", "question": "effectiveness"}], None, 5)[0]
    assert result["passed"] is passes
    assert result["metrics"]["citation_validity_rate"] == 1.0
    assert result["metrics"]["raw_numeric_support_rate"] == float(passes)
    assert bool(result["failure_reasons"]) is not passes


@pytest.mark.parametrize("expect_claims,passes", [(False, True), (True, False)])
def test_live_discovery_respects_declarative_answer_expectation(monkeypatch, evidence, expect_claims, passes):
    import main
    answer = "- [DOC901 — Carp harvest study, PDF p. 2]"
    monkeypatch.setattr(main, "ask_chatbot_with_context", lambda *args, **kwargs: (answer, None, [evidence]))
    case = {"case_id": "lookup", "question": "Find the carp report", "expect_claims": expect_claims}
    result = run_live_answers(None, [case], None, 5)[0]
    assert result["passed"] is passes
    assert result["details"]["citation_checks"]["answer_kind"] == "sources_only"
    assert result["metrics"]["citation_coverage"] is None
