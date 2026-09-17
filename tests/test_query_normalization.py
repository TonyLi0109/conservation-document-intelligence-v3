"""Whole-query quote wrappers are presentation, not retrieval intent."""

from chat_context import resolve_query
from data_models import KnowledgeArtifact
from database import KnowledgeStore
from query_normalization import strip_wrapping_query_quotes
from retrieval import retrieve_evidence


QUESTION = (
    "Does the corpus provide evidence that climate change causes invasive species "
    "problems in Missouri? Distinguish direct evidence from general associations or risks."
)


def test_only_balanced_whole_query_double_quotes_are_removed():
    assert strip_wrapping_query_quotes('"' + QUESTION + '"') == QUESTION
    assert strip_wrapping_query_quotes("\u201c" + QUESTION + "\u201d") == QUESTION
    assert strip_wrapping_query_quotes("\uff02" + QUESTION + "\uff02") == QUESTION
    internal = 'Distinguish evidence about "climate change" from general risks.'
    assert strip_wrapping_query_quotes(internal) == internal
    assert strip_wrapping_query_quotes('"unbalanced') == '"unbalanced'


def test_independent_query_resolution_is_quote_invariant():
    plain = resolve_query(QUESTION, [])
    quoted = resolve_query('"' + QUESTION + '"', [])
    assert quoted.standalone_query == plain.standalone_query == QUESTION
    assert quoted.active_subject == plain.active_subject


def test_hybrid_retrieval_is_quote_invariant_for_wrapped_question():
    with KnowledgeStore(":memory:") as store:
        store.ingest_chunk(KnowledgeArtifact(
            "DOC901", "Missouri Climate Assessment", "1",
            "Climate change can increase invasive species risks in Missouri ecosystems.",
        ), [1.0, 0.0])
        store.ingest_chunk(KnowledgeArtifact(
            "DOC902", "Missouri Invasive Species Review", "2",
            "The report describes general associations between climate risks and invasive species.",
        ), [0.0, 1.0])
        plain_trace, quoted_trace = {}, {}
        plain = retrieve_evidence(
            store, QUESTION, top_k=2, mode="hybrid_rerank", diagnostics=plain_trace,
        )
        quoted = retrieve_evidence(
            store, '"' + QUESTION + '"', top_k=2, mode="hybrid_rerank",
            diagnostics=quoted_trace,
        )

    identity = lambda items: [(item.document_id, item.page_number) for item in items]
    assert identity(quoted) == identity(plain)
    assert quoted_trace["query"] == plain_trace["query"] == QUESTION
    assert quoted_trace["query_type"] == plain_trace["query_type"] == "semantic_question"
    assert quoted_trace["expanded_query"] == plain_trace["expanded_query"]
    assert quoted_trace["lexical_candidates"] == plain_trace["lexical_candidates"]
    assert quoted_trace["final_artifact_ids"] == plain_trace["final_artifact_ids"]

def test_search_corpus_normalizes_before_embedding(monkeypatch):
    import main

    captured = []
    with KnowledgeStore(":memory:") as store:
        store.ingest_chunk(KnowledgeArtifact(
            "DOC901", "Missouri Climate Assessment", "1",
            "Climate change can increase invasive species risks in Missouri ecosystems.",
        ), [1.0, 0.0])
        monkeypatch.setattr(main, "generate_embedding", lambda query: captured.append(query) or [1.0, 0.0])
        result = main.search_corpus('"' + QUESTION + '"', store, top_k=1, mode="dense")

    assert captured == [QUESTION]
    assert [item.document_id for item in result] == ["DOC901"]
