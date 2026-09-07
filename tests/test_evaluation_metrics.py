"""Independent, adversarial metric fixtures; no model or corpus access."""

import json
import math
from dataclasses import replace

import pytest

from data_models import KnowledgeArtifact
from evaluation.metrics import evaluate_answer, evaluate_claims, retrieval_metrics
from validator import INSUFFICIENT_MESSAGE, validate_render_and_collect_sources


@pytest.fixture
def evidence():
    return KnowledgeArtifact(
        "DOC001", "Wetland [monitoring] report", "12",
        "The restored area increased by 20% to 1,200 hectares. Sampling used 3 sites.",
        source_url="https://example.org/report.pdf", printed_page_label="7",
    )


def payload(evidence, **claim_changes):
    claim = {
        "text": "The restored area increased by 20% to 1200 hectares.",
        "evidence_ids": ["K1"],
        "supporting_spans": [evidence.original_text_chunk],
    }
    claim.update(claim_changes)
    return {"status": "answered", "claims": [claim], "unsupported_facets": []}


def reasons(result):
    return {
        reason for failure in result["failures"]
        for reason in failure.get("reasons", [failure.get("reason")])
    }


def test_retrieval_known_values_dedupe_and_graded_ndcg():
    result = retrieval_metrics(["D0", "D1", "D1", "D2", "D3"], {"D1": 3, "D2": 1, "D3": 1}, 3)
    expected_dcg = 7 / math.log2(3) + 1 / math.log2(4)
    expected_ideal = 7 + 1 / math.log2(3) + 1 / math.log2(4)
    assert result["precision_at_k"] == pytest.approx(2 / 3)
    assert result["recall_at_k"] == pytest.approx(2 / 3)
    assert result["mrr"] == 0.5
    assert result["ndcg_at_k"] == pytest.approx(expected_dcg / expected_ideal)
    assert result["hit_at_k"] == 1.0
    assert result["duplicate_count"] == 1
    assert result["unjudged_at_k"] == 1


def test_short_rankings_do_not_inflate_precision():
    result = retrieval_metrics(["A"], {"A": 1}, 5)
    assert result["precision_at_k"] == 0.2
    assert result["recall_at_k"] == 1.0


def test_mrr_full_ranking_and_cutoff_metrics_are_distinct():
    result = retrieval_metrics(["B", "C", "A"], {"A": 1}, 2)
    assert result["mrr"] == pytest.approx(1 / 3)
    assert result["hit_at_k"] == result["ndcg_at_k"] == 0.0


@pytest.mark.parametrize("labels", [{}, {"A": 0}])
def test_no_positive_labels_are_unscored_not_success(labels):
    result = retrieval_metrics(["A"], labels, 5)
    assert all(result[key] is None for key in ("precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "hit_at_k"))


def test_no_retrieval_with_positive_labels_scores_zero():
    result = retrieval_metrics([], {"A": 1}, 5)
    assert all(result[key] == 0 for key in ("precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "hit_at_k"))


@pytest.mark.parametrize("k,labels", [(0, {"A": 1}), (True, {"A": 1}), (3, {"A": -1}), (3, {"A": True})])
def test_retrieval_rejects_invalid_metric_inputs(k, labels):
    with pytest.raises(ValueError):
        retrieval_metrics(["A"], labels, k)


def test_exact_claim_with_normalized_numeric_tokens_passes(evidence):
    result = evaluate_claims(payload(evidence), {"K1": evidence}, [evidence])
    assert result["passed"]
    assert result["canonical_membership_assessed"]
    assert result["provenance_validity_rate"] == result["numeric_support_rate"] == 1.0
    assert result["semantic_support"] == "not_assessed"


@pytest.mark.parametrize("changes,reason", [
    ({"evidence_ids": ["K404"]}, "unknown_or_invalid_evidence_handle"),
    ({"supporting_spans": ["invented evidence"]}, "span_not_verbatim_in_cited_evidence"),
    ({"evidence_ids": ["K1", "K1"]}, "invalid_claim_schema"),
    ({"evidence_ids": ["DOC001"]}, "invalid_claim_schema"),
])
def test_claim_rejects_fabricated_handles_spans_and_schema(evidence, changes, reason):
    result = evaluate_claims(payload(evidence, **changes), {"K1": evidence}, [evidence])
    assert not result["passed"]
    assert reason in reasons(result)


def test_every_cited_artifact_must_contribute_span(evidence):
    other = KnowledgeArtifact("DOC002", "Unrelated", "1", "Fish occupy rivers.")
    result = evaluate_claims(payload(evidence, evidence_ids=["K1", "K2"]), {"K1": evidence, "K2": other})
    assert "cited_artifact_has_no_supporting_span" in reasons(result)


def test_span_in_retrieved_but_uncited_artifact_does_not_ground_claim(evidence):
    other = KnowledgeArtifact("DOC002", "Unrelated", "1", "Fish occupy rivers.")
    result = evaluate_claims(payload(evidence, supporting_spans=[other.original_text_chunk]), {"K1": evidence, "K2": other})
    assert "span_not_verbatim_in_cited_evidence" in reasons(result)


@pytest.mark.parametrize("field,value", [
    ("title", "Forged title"), ("page_number", "99"),
    ("original_text_chunk", "Fabricated source says restoration increased by 20%."),
    ("source_url", "https://example.org/forged.pdf"),
])
def test_claim_independent_canonical_membership_rejects_forgery(evidence, field, value):
    forged = replace(evidence, **{field: value})
    result = evaluate_claims(payload(forged), {"K1": forged}, [evidence])
    assert "artifact_not_in_canonical_evidence" in reasons(result)
    assert result["provenance_validity_rate"] == 0


def test_fabricated_number_fails_even_when_span_is_exact(evidence):
    result = evaluate_claims(payload(evidence, text="The restored area increased by 90%."), {"K1": evidence}, [evidence])
    assert result["provenance_validity_rate"] == 1.0
    assert result["numeric_support_rate"] == result["grounded_proxy_rate"] == 0
    assert result["failures"][0]["unsupported_numeric_tokens"] == ["90%"]


def test_number_elsewhere_in_chunk_does_not_support_cited_span(evidence):
    result = evaluate_claims(payload(evidence, text="Sampling used 3 sites.", supporting_spans=["The restored area increased by 20%"]), {"K1": evidence})
    assert result["numeric_support_rate"] == 0


def test_terminal_integer_is_not_missed_by_numeric_proxy(evidence):
    result = evaluate_claims(payload(evidence, text="The number of sites was 4."), {"K1": evidence})
    assert result["numeric_claim_count"] == 1
    assert result["numeric_support_rate"] == 0


def test_numeric_overlap_is_explicitly_not_semantic_truth(evidence):
    result = evaluate_claims(payload(evidence, text="Restored area decreased by 20%."), {"K1": evidence})
    assert result["grounded_proxy_rate"] == 1
    assert result["semantic_support"] == "not_assessed"
    assert not result["canonical_membership_assessed"]


def test_empty_claims_have_no_vacuous_grounding_score():
    result = evaluate_claims({"claims": []}, {})
    assert result["grounded_proxy_rate"] is None
    assert result["provenance_validity_rate"] is None
    assert result["numeric_support_rate"] is None
    assert not result["passed"]


def test_malformed_claim_collection_is_exposed():
    result = evaluate_claims({"claims": "not a list"}, {})
    assert not result["schema_valid"]
    assert "claims_must_be_an_array" in reasons(result)


def rendered(evidence):
    return validate_render_and_collect_sources(json.dumps(payload(evidence)), {"K1": evidence})


def test_real_validator_output_with_escaped_title_and_printed_page_passes(evidence):
    markdown, sources = rendered(evidence)
    result = evaluate_answer(markdown, [evidence], sources)
    assert result["passed"]
    assert result["citation_validity_rate"] == result["citation_coverage"] == 1.0
    assert result["answer_kind"] == "answer"


@pytest.mark.parametrize("page", ["Web", "12-14"])
def test_web_and_physical_page_range_citations(evidence, page):
    evidence = replace(evidence, page_number=page, printed_page_label=None)
    markdown, sources = rendered(evidence)
    assert evaluate_answer(markdown, [evidence], sources)["passed"]


@pytest.mark.parametrize("old,new,reason", [
    ("DOC001", "DOC999", "nonexistent_document_citation"),
    ("PDF p. 12", "PDF p. 99", "forged_or_mismatched_citation_metadata"),
    ("Wetland", "Invented", "forged_or_mismatched_citation_metadata"),
    ("DOC001", "K1", "malformed_citation"),
])
def test_rendered_citation_metadata_tampering_is_detected(evidence, old, new, reason):
    markdown, sources = rendered(evidence)
    result = evaluate_answer(markdown.replace(old, new), [evidence], sources)
    assert not result["passed"]
    assert reason in reasons(result)
    assert result["citation_validity_rate"] == 0


def test_unclosed_citation_is_not_silently_ignored(evidence):
    markdown, sources = rendered(evidence)
    result = evaluate_answer(markdown[:-1], [evidence], sources)
    assert result["malformed_citation_count"] == 1
    assert result["citation_count"] == 1


def test_returned_forged_source_cannot_launder_an_authentic_citation(evidence):
    markdown, _ = rendered(evidence)
    forged = replace(evidence, original_text_chunk="A fabricated chunk.")
    result = evaluate_answer(markdown, [evidence], [forged])
    assert result["invalid_source_count"] == 1
    assert result["citation_validity_rate"] == 0


def test_two_chunks_with_identical_display_metadata_do_not_false_fail(evidence):
    other_chunk = replace(evidence, original_text_chunk="Another paragraph on the same page.")
    markdown, sources = rendered(evidence)
    assert evaluate_answer(markdown, [evidence, other_chunk], sources)["passed"]


def test_uncited_bullet_reduces_coverage(evidence):
    markdown, sources = rendered(evidence)
    result = evaluate_answer(markdown + "\n- An invented claim with no citation.", [evidence], sources)
    assert result["citation_coverage"] == 0.5
    assert result["uncited_claim_count"] == 1
    assert not result["passed"]


def test_unsupported_facets_are_not_mistaken_for_factual_claims(evidence):
    markdown, sources = rendered(evidence)
    result = evaluate_answer(markdown + "\n\n**Unsupported facets**\n\n- Cost evidence is absent.", [evidence], sources)
    assert result["claim_count"] == 1
    assert result["passed"]


def test_sources_only_fallback_does_not_pass_an_answer_requirement(evidence):
    markdown, sources = rendered(evidence)
    citation = markdown[markdown.index("[DOC"):]
    result = evaluate_answer("- " + citation, [evidence], sources)
    assert result["answer_kind"] == "sources_only"
    assert result["citation_validity_rate"] == 1.0
    assert result["citation_coverage"] is None
    assert not result["passed"]
    assert evaluate_answer("- " + citation, [evidence], sources, expect_claims=False)["passed"]


def test_abstention_has_no_citation_success_score():
    result = evaluate_answer(INSUFFICIENT_MESSAGE + "\n\n**Unsupported facets**\n\n- No evidence.", [], [], expect_claims=False)
    assert result["answer_kind"] == "abstention"
    assert result["citation_validity_rate"] is None
    assert result["citation_coverage"] is None
    assert result["passed"]


def test_empty_output_does_not_pass_even_when_abstention_allowed():
    result = evaluate_answer("", [], [], expect_claims=False)
    assert result["answer_kind"] == "empty"
    assert not result["passed"]


def test_headings_alone_are_not_a_successful_abstention():
    result = evaluate_answer("**All wetlands are destroyed.**", [], [], expect_claims=False)
    assert result["answer_kind"] == "unclassified"
    assert not result["passed"]


def test_duplicate_and_unreferenced_sources_are_visible(evidence):
    markdown, sources = rendered(evidence)
    other = replace(evidence, document_id="DOC002", title="Other report")
    result = evaluate_answer(markdown + "\n" + markdown, [evidence, other], sources + sources + [other])
    assert result["duplicate_citation_count"] == 1
    assert result["duplicate_source_count"] == 1
    assert result["unreferenced_source_count"] == 1
    assert not result["passed"]
