"""Respect context scope when using prior source IDs as document-family anchors."""

from chat_context import ResolvedQuery
from data_models import DocumentSource, KnowledgeArtifact
from database import KnowledgeStore
import main


def test_temporal_partial_scope_does_not_restrict_search_to_prior_carp_family(monkeypatch):
    with KnowledgeStore(":memory:") as store:
        sources = []
        for doc_id, title, text in (
            ("DOC951", "Invasive Carp Guidance 2020", "Targeted harvest controls invasive carp."),
            ("DOC953", "Other Invasive Fish Guidance 2024", "Targeted harvest is recommended for other invasive fish."),
        ):
            store.upsert_document_sources([DocumentSource(doc_id, title, "https://example.org/fixture.pdf", None,
                                                          doc_id + ".pdf", "pdf", agency="Synthetic agency")])
            artifact = KnowledgeArtifact(doc_id, title, "1", "Status: Final. " + text)
            store.ingest_chunk(artifact, [1., 0.])
            sources.append(artifact)
        question = "What current guidance covers those methods for other invasive fish?"
        resolved = ResolvedQuery(question, "What is the current targeted harvest guidance for other invasive fish?",
                                 "other invasive fish", uses_history=True, relation="PARTIAL_CONTEXT")
        monkeypatch.setattr(main, "resolve_query", lambda *args, **kwargs: resolved)
        history = [{"role": "assistant", "content": "Targeted harvest controls invasive carp.", "sources": [sources[0]]}]
        diagnostics = {}
        answer, _, _ = main.ask_chatbot_with_context(question, store, history=history, diagnostics=diagnostics)
    assert "DOC953" in diagnostics["temporal_selected_document_ids"]
    assert "Other Invasive Fish Guidance" in answer
