"""Synthetic lifecycle policy integration; no provider calls or real-corpus writes."""

from contextlib import contextmanager
import copy

import pytest

from data_models import DocumentSource, KnowledgeArtifact
from database import KnowledgeStore
from document_lifecycle import get_lifecycles
from temporal import detect_temporal_intent, render_temporal_answer, select_temporal_evidence, verified_relationships
from validator import _citation


def document(docid, year, *, title=None, text="", status="final", agency="Synthetic conservation agency"):
    return {"document_id": docid, "year": year, "title": title or f"Carp Guidance {year}",
            "text": f"Published: {year}.\nStatus: {status}.\n{text}\nCarp guidance recommends targeted harvest and monitoring invasive carp.",
            "agency": agency}


@contextmanager
def corpus(*documents):
    """Build source metadata and canonical chunks, then use the actual extractor."""
    with KnowledgeStore(":memory:") as store:
        for doc in documents:
            store.upsert_document_sources([DocumentSource(
                doc["document_id"], doc["title"], "https://example.org/test.pdf", None,
                doc["document_id"] + ".pdf", "pdf", year=doc["year"], agency=doc["agency"],
            )])
            store.ingest_chunk(KnowledgeArtifact(doc["document_id"], doc["title"], "1", doc["text"],
                                                  source_url="https://example.org/test.pdf"), [1., 0.])
        yield store


@pytest.mark.parametrize("relationship", ["amends", "supplements"])
def test_historical_year_does_not_select_later_partial_update(relationship):
    with corpus(document("DOC901", "2020"),
                document("DOC902", "2024", text=f"This guidance {relationship} DOC901.")) as store:
        selected = select_temporal_evidence("What did the 2020 carp guidance recommend?", store,
                                            top_k=1, as_of="2026-01-01")
    assert selected["selected_document_ids"] == ["DOC901"]


def test_bounded_comparison_does_not_select_out_of_range_amendment():
    with corpus(document("DOC901", "2020"), document("DOC902", "2022"),
                document("DOC903", "2024", text="This guidance amends DOC901.")) as store:
        selected = select_temporal_evidence("How did carp guidance change from 2020 to 2022?", store,
                                            top_k=5, as_of="2026-01-01")
    assert selected["selected_document_ids"] == ["DOC901", "DOC902"]


def test_current_amendment_keeps_its_base_available():
    with corpus(document("DOC901", "2020"),
                document("DOC902", "2024", text="This guidance amends DOC901.")) as store:
        selected = select_temporal_evidence("What is the current carp guidance?", store,
                                            top_k=1, as_of="2026-01-01")
        answer, _, sources = render_temporal_answer(selected, store)
    assert set(selected["selected_document_ids"]) == {"DOC901", "DOC902"}
    assert {source.document_id for source in sources} == {"DOC901", "DOC902"}
    assert "Partial update" in answer
    assert not any("replaced/withdrawn" in decision["reason"] for decision in selected["decisions"])


def test_future_revision_is_not_preferred_as_current():
    with corpus(document("DOC901", "2020"),
                document("DOC902", "2023", text="Last revised: 2030-01-01.")) as store:
        assert get_lifecycles(store)["DOC902"]["revision_date"] == "2030-01-01"
        selected = select_temporal_evidence("What is the current carp guidance?", store,
                                            top_k=1, as_of="2026-01-01")
    assert selected["selected_document_ids"] == ["DOC901"]


@pytest.mark.parametrize("effective", ["2026", "2026-09"])
def test_partial_effective_date_does_not_activate_replacement_during_uncertain_period(effective):
    with corpus(document("DOC901", "2020"),
                document("DOC902", "2024", text=f"Effective: {effective}.\nThis guidance supersedes DOC901.")) as store:
        selected = select_temporal_evidence("What is the current carp guidance?", store,
                                            top_k=1, as_of="2026-09-07")
        answer, _, sources = render_temporal_answer(selected, store)
    assert selected["selected_document_ids"] == ["DOC901"]
    assert selected["relationships"] == []
    assert {source.document_id for source in sources} == {"DOC901"}
    assert any(effective in warning and "uncertain" in warning for warning in selected["uncertainties"])
    assert "explicitly supersedes" not in answer


def test_partial_effective_month_activates_after_entire_month_has_elapsed():
    with corpus(document("DOC901", "2020"),
                document("DOC902", "2024", text="Effective: 2026-08.\nThis guidance supersedes DOC901.")) as store:
        selected = select_temporal_evidence("What is the current carp guidance?", store,
                                            top_k=1, as_of="2026-09-07")
    assert selected["selected_document_ids"] == ["DOC902"]
    assert any(edge["document_id"] == "DOC902" and edge["target_id"] == "DOC901"
               for edge in selected["relationships"])
    assert not any("applicability within that period is uncertain" in warning for warning in selected["uncertainties"])


def test_replacement_cycle_does_not_establish_a_unique_current_successor():
    with corpus(document("DOC901", "2020", text="This guidance supersedes DOC902."),
                document("DOC902", "2024", text="This guidance supersedes DOC901.")) as store:
        selected = select_temporal_evidence("What is the current carp guidance?", store,
                                            top_k=1, as_of="2026-09-07")
        answer, _, _ = render_temporal_answer(selected, store)
    assert any("cycle" in warning for warning in selected["uncertainties"])
    assert all("Conflicting replacement relationships" in decision["reason"] for decision in selected["decisions"])
    assert "Preferred within this family" not in answer


@pytest.mark.parametrize("question,mode", [
    ("What electrical current deters invasive carp?", "none"),
    ("How does current velocity affect fish movement?", "none"),
    ("How strong is the river current in wetlands?", "none"),
    ("What is the current guidance for electrical current barriers?", "current"),
    ("Does the latest report quantify electrical current in carp barriers?", "latest"),
])
def test_physical_current_is_distinguished_from_temporal_intent(question, mode):
    assert detect_temporal_intent(question).mode == mode


def test_supersession_annotation_cites_successor_evidence_not_only_old_source():
    with corpus(document("DOC901", "2020"),
                document("DOC902", "2024", text="This guidance supersedes DOC901.")) as store:
        selected = select_temporal_evidence("What did the 2020 carp guidance recommend?", store,
                                            top_k=1, as_of="2026-01-01")
        answer, _, sources = render_temporal_answer(selected, store)
    assert selected["selected_document_ids"] == ["DOC901"]
    assertions = [line for line in answer.splitlines() if "replaced/withdrawn" in line or "explicitly supersedes" in line]
    assert assertions
    successor = next(source for source in sources if source.document_id == "DOC902")
    assert "This guidance supersedes DOC901." in successor.original_text_chunk
    assert all(_citation(successor) in line for line in assertions)


def test_comparison_does_not_attach_post_period_supersession_without_its_evidence():
    with corpus(document("DOC901", "2020"), document("DOC902", "2022"),
                document("DOC903", "2024", text="This guidance supersedes DOC902.")) as store:
        selected = select_temporal_evidence("How did carp guidance change from 2020 to 2022?", store,
                                            top_k=5, as_of="2026-01-01")
        answer, _, sources = render_temporal_answer(selected, store)
    assert selected["selected_document_ids"] == ["DOC901", "DOC902"]
    assert "Explicitly replaced/withdrawn by DOC903" not in answer
    assert "DOC903" not in {source.document_id for source in sources}


def test_replacement_reason_cites_replacement_not_another_incoming_revision():
    with corpus(document("DOC901", "2020"),
                document("DOC902", "2022", text="This guidance is a revision of DOC901."),
                document("DOC903", "2024", text="This guidance supersedes DOC901.")) as store:
        selected = select_temporal_evidence("What did the 2020 carp guidance recommend?", store,
                                            top_k=1, as_of="2026-01-01")
        answer, _, sources = render_temporal_answer(selected, store)
    replacement = next(source for source in sources if source.document_id == "DOC903")
    reason = next(line for line in answer.splitlines() if "Explicitly replaced/withdrawn by DOC903" in line)
    assert _citation(replacement) in reason
    assert selected["decisions"][0]["evidence"]["document_id"] == "DOC903"


def test_future_replacement_does_not_produce_a_present_supersession_assertion():
    with corpus(document("DOC901", "2020"),
                document("DOC902", "2024", text="Effective: 2030-01-01.\nThis guidance supersedes DOC901.")) as store:
        selected = select_temporal_evidence("What is the current carp guidance?", store,
                                            top_k=1, as_of="2026-01-01")
        answer, _, sources = render_temporal_answer(selected, store)
    assert selected["selected_document_ids"] == ["DOC901"]
    assert "explicitly supersedes" not in answer
    assert "replaced/withdrawn" not in answer
    assert {source.document_id for source in sources} == {"DOC901"}


def test_future_withdrawal_does_not_make_current_base_inapplicable():
    with corpus(document("DOC901", "2020"),
                document("DOC902", "2030", title="Carp Withdrawal Notice 2030",
                         text="This document withdraws DOC901.")) as store:
        selected = select_temporal_evidence("What is the current carp guidance?", store,
                                            top_k=1, as_of="2026-01-01")
        answer, _, _ = render_temporal_answer(selected, store)
    assert selected["selected_document_ids"] == ["DOC901"]
    assert "Draft, withdrawn or future-dated" not in answer
    assert "replaced/withdrawn" not in answer


def test_as_of_query_prefers_guidance_applicable_at_requested_date():
    with corpus(document("DOC901", "2020"),
                document("DOC902", "2024", text="This guidance supersedes DOC901."),
                document("DOC903", "2026", text="This guidance supersedes DOC902.")) as store:
        selected = select_temporal_evidence("What carp guidance applied as of 2025-06-01?", store, top_k=1)
    assert selected["selected_document_ids"] == ["DOC902"]


def test_current_final_guidance_and_latest_report_are_distinct_choices():
    with corpus(document("DOC901", "2023", title="Final Carp Guidance 2023"),
                document("DOC902", "2025", title="Carp Research Report 2025")) as store:
        current = select_temporal_evidence("What is the current carp guidance?", store, top_k=1, as_of="2026-01-01")
        latest = select_temporal_evidence("What does the latest carp report say?", store, top_k=1, as_of="2026-01-01")
        answer, _, _ = render_temporal_answer(latest, store)
    assert current["selected_document_ids"] == ["DOC901"]
    assert latest["selected_document_ids"] == ["DOC902"]
    assert latest["relationships"] == []
    assert "does not establish replacement" in " ".join(latest["uncertainties"])
    assert "explicitly supersedes" not in answer


def test_current_guidance_prefers_final_over_same_family_draft():
    with corpus(document("DOC901", "2024", title="Draft Carp Guidance 2024", status="draft"),
                document("DOC902", "2024", title="Final Carp Guidance 2024")) as store:
        selected = select_temporal_evidence("What is the current carp guidance?", store, top_k=1, as_of="2026-01-01")
    assert selected["selected_document_ids"] == ["DOC902"]


@pytest.mark.parametrize("statement", [
    "This guidance does not replace DOC901.",
    "This guidance never supersedes DOC901.",
    "This guidance is not a revision of DOC901.",
    "The unrelated river policy supersedes DOC901.",
])
def test_negated_and_other_document_statements_do_not_establish_owner_relationship(statement):
    with corpus(document("DOC901", "2020"), document("DOC902", "2024", text=statement)) as store:
        index = get_lifecycles(store)
        assert index["DOC902"]["relations"] == []
        assert verified_relationships(store, index) == []


def test_metadata_only_dates_and_title_status_render_without_fabricated_evidence():
    item = document("DOC901", "2023", title="Final Carp Guidance 2023")
    item["text"] = "Management recommendations include harvesting carp and monitoring their movement."
    with corpus(item) as store:
        selected = select_temporal_evidence("What is the current carp guidance?", store, top_k=1, as_of="2026-01-01")
        assert selected["documents"]["DOC901"]["dates"]["publication_date"]["source"] != "explicit"
        answer, _, sources = render_temporal_answer(selected, store)
    assert "Published 2023" in answer
    assert "Version/date evidence:" not in answer
    assert [source.document_id for source in sources] == ["DOC901"]


@pytest.mark.parametrize("tamper", ["page", "owner", "span"])
def test_policy_rechecks_persisted_relationship_provenance(tamper):
    # Deliberately alter only an in-memory copy of extracted metadata to simulate
    # stale/corrupt edge records. Canonical evidence remains unchanged.
    with corpus(document("DOC901", "2020"),
                document("DOC902", "2024", text="This guidance supersedes DOC901.")) as store:
        index = copy.deepcopy(get_lifecycles(store))
        assert verified_relationships(store, index)
        proof = index["DOC902"]["relations"][0]["evidence"]
        proof[{"page": "page_number", "owner": "document_id", "span": "exact_span"}[tamper]] = {
            "page": "999", "owner": "DOC901", "span": "An invented replacement claim.",
        }[tamper]
        assert verified_relationships(store, index) == []


def test_unrelated_source_is_not_added_to_anchored_family():
    with corpus(document("DOC901", "2020"),
                document("DOC902", "2024", text="This guidance supersedes DOC901."),
                document("DOC903", "2025", title="Unrelated Hydrilla Research 2025", agency="Different agency")) as store:
        selected = select_temporal_evidence("Is that still current? Topic: carp.", store,
                                            anchor_document_ids=["DOC901"], top_k=5, as_of="2026-01-01")
    assert "DOC903" not in selected["selected_document_ids"]
    assert "DOC903" not in {source.document_id for source in selected["evidence"]}
