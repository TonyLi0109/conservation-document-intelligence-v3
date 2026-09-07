"""Lifecycle dates/relations must remain conservative, traceable and cached."""

import json

import pytest

from data_models import DocumentSource, KnowledgeArtifact
from database import KnowledgeStore
import document_lifecycle as lifecycle


def metadata(docid, title, year="", agency="Synthetic Conservation Agency"):
    return {"document_id": docid, "title": title, "year": year, "agency": agency, "topic": "test"}


def chunk(docid, text, page="1"):
    return {"document_id": docid, "page_number": page, "original_text_chunk": text}


@pytest.mark.parametrize("text,expected", [
    ("2024", {"value": "2024", "precision": "year"}),
    ("2024-02", {"value": "2024-02", "precision": "month"}),
    ("2024-02-29", {"value": "2024-02-29", "precision": "day"}),
    ("January 2025", {"value": "2025-01", "precision": "month"}),
    ("January 9, 2025", {"value": "2025-01-09", "precision": "day"}),
    ("9 January 2025", {"value": "2025-01-09", "precision": "day"}),
    ("2023-02-29", None), ("2024-00-01", None), ("2024-13", None),
    ("2024-01-00", None), ("2024-01-32", None), ("January 40, 2025", None),
    ("2024-02-29 extra", None), ("01/02/2024", None),
])
def test_date_precision_and_invalid_dates(text, expected):
    assert lifecycle.parse_date(text) == expected


def test_distinct_publication_revision_effective_dates_and_version_have_exact_evidence():
    text = "Published: 2020. Revised: January 9, 2024. Effective: 2024-03-01. Status: Final. Version: 2."
    result = lifecycle.extract_lifecycles([metadata("DOC951", "Carp Guidance", "2021")], [chunk("DOC951", text)])["DOC951"]
    assert result["publication_date"] == "2020"
    assert result["revision_date"] == "2024-01-09"
    assert result["effective_date"] == "2024-03-01"
    assert result["version"] == "2"
    assert result["status"] == "final"
    for signal in [*result["dates"].values(), result["status_evidence"], result["version_evidence"]]:
        assert signal["source"] == "explicit"
        assert signal["evidence"]["exact_span"] in text
        assert signal["evidence"]["page_number"] == "1"
    assert any("2021 (metadata_inferred)" in warning for warning in result["warnings"])


def test_catalog_and_title_dates_remain_heuristic_without_fabricated_source_spans():
    result = lifecycle.extract_lifecycles([metadata("DOC951", "Carp Guidance 2024 Version 2", "2023")], [])["DOC951"]
    assert result["publication_date"] == "2023"
    assert result["dates"]["publication_date"]["source"] == "metadata_inferred"
    assert result["dates"]["publication_date"]["evidence"] is None
    assert result["version_evidence"]["source"] == "title_inferred"
    assert result["warnings"]


def test_invalid_labelled_date_does_not_fall_back_to_partial_year():
    result = lifecycle.extract_lifecycles([metadata("DOC951", "Carp Guidance")], [chunk("DOC951", "Revised: 2024-02-30.")])["DOC951"]
    assert result["revision_date"] is None
    assert result["warnings"]


def test_external_document_dates_and_scheduled_revision_do_not_change_own_dates():
    text = ("The external CWD Management Plan was revised in 2022, originally published in 2010. "
            "This report will be revised in 2030. Revised: January 2025.")
    result = lifecycle.extract_lifecycles([metadata("DOC951", "Annual Review 2024", "2024")], [chunk("DOC951", text, "23")])["DOC951"]
    assert result["publication_date"] == "2024"
    assert result["revision_date"] is None


def test_title_normalization_retains_topic_type_and_issuing_agency_boundary():
    assert lifecycle.normalize_title("Invasive Carp Management Strategy - Revised 2021 Version 2") == "invasive carp management strategy"
    assert lifecycle.normalize_title("Invasive Carp Management Research Report 2025") != lifecycle.normalize_title("Invasive Carp Management Guidance 2023")
    docs = [metadata("DOC951", "Carp Guidance 2020"), metadata("DOC952", "Carp Guidance Revised 2024"),
            metadata("DOC953", "Carp Research Report 2025"), metadata("DOC954", "Carp Guidance 2025", agency="Different issuer"),
            metadata("DOC955", "Carp Guidance 2026", agency="")]
    result = lifecycle.extract_lifecycles(docs, [])
    assert result["DOC951"]["family_id"] == result["DOC952"]["family_id"]
    assert result["DOC951"]["family_source"] == "title_inferred"
    assert all(result[docid]["family_id"] != result["DOC951"]["family_id"] for docid in ("DOC953", "DOC954", "DOC955"))
    assert all(not record["relations"] for record in result.values())


def test_draft_final_candidates_share_family_but_no_supersession_is_assumed():
    result = lifecycle.extract_lifecycles([metadata("DOC951", "Draft Carp Guidance 2024"), metadata("DOC952", "Final Carp Guidance 2024")], [])
    assert result["DOC951"]["family_id"] == result["DOC952"]["family_id"]
    assert result["DOC951"]["status"] == "draft" and result["DOC952"]["status"] == "final"
    assert not result["DOC952"]["relations"]


def test_explicit_revision_chain_preserves_direction_and_amendment_scope():
    docs = [metadata("DOC951", "Carp Guidance 2020", "2020"), metadata("DOC952", "Carp Guidance 2023", "2023"),
            metadata("DOC953", "Carp Amendment 2024", "2024"), metadata("DOC954", "Carp Supplement 2025", "2025")]
    chunks = [chunk("DOC952", "This guidance supersedes DOC951."),
              chunk("DOC953", "This amendment amends DOC952."),
              chunk("DOC954", "This supplement is a supplement to DOC953.")]
    result = lifecycle.extract_lifecycles(docs, chunks)
    assert [(relation["type"], relation["target_id"]) for relation in result["DOC952"]["relations"]] == [("supersedes", "DOC951")]
    assert result["DOC953"]["relations"][0]["type"] == "amendment_of"
    assert result["DOC954"]["relations"][0]["type"] == "supplement_to"
    assert len({record["family_id"] for record in result.values()}) == 1
    assert result["DOC952"]["status"] != "withdrawn"
    for record in result.values():
        for relation in record["relations"]:
            original = next(item["original_text_chunk"] for item in chunks if item["document_id"] == record["document_id"])
            assert relation["source"] == "explicit" and relation["evidence"]["exact_span"] in original


@pytest.mark.parametrize("sentence", [
    "This guidance does not supersede DOC951.", "This guidance will supersede DOC951.",
    "This guidance might replace DOC951.", "If approved, this guidance supersedes DOC951.",
    "The Farm Bill replaces DOC951.", "The previous report states that this guidance supersedes DOC951.",
    '"This guidance supersedes DOC951."',
])
def test_negated_hypothetical_external_and_quoted_relationships_are_not_invented(sentence):
    docs = [metadata("DOC951", "Carp Guidance 2020"), metadata("DOC952", "Carp Report 2024")]
    result = lifecycle.extract_lifecycles(docs, [chunk("DOC952", sentence)])
    assert not result["DOC952"]["relations"]


def test_unknown_or_ambiguous_target_title_does_not_force_relationship():
    docs = [metadata("DOC951", "Forest and Wildlife Guidance 2020", "2020"),
            metadata("DOC952", "Forest and Wildlife Guidance 2023", "2023"), metadata("DOC953", "Replacement Guidance 2024", "2024")]
    text = "This guidance supersedes Forest and Wildlife Guidance. This guidance replaces DOC999."
    result = lifecycle.extract_lifecycles(docs, [chunk("DOC953", text)])
    assert not result["DOC953"]["relations"]
    assert result["DOC953"]["warnings"]


def test_unique_full_title_and_year_resolve_without_document_id():
    docs = [metadata("DOC951", "Invasive Carp Guidance 2020", "2020"),
            metadata("DOC952", "Invasive Carp Guidance 2023", "2023")]
    result = lifecycle.extract_lifecycles(docs, [chunk("DOC952", "This guidance replaces Invasive Carp Guidance 2020.")])
    assert result["DOC952"]["relations"][0]["target_id"] == "DOC951"


def test_real_style_flattened_pdf_clause_resolves_acronym_year_revision_without_blanket_replacement():
    docs = [metadata("DOC951", "Missouri State Wildlife Action Plan", "2015", "Missouri Department of Conservation"),
            metadata("DOC952", "2022 Missouri Comprehensive Conservation Strategy", "2022", "Missouri Department of Conservation")]
    statement = "The 2020 CCS serves as the comp rehensive revision of both the 2010 SFAP and the 2015 SWAP."
    text = "Partner Initial and Draft Review Opportunities in March 2020 Timeframe and Revision " + statement
    result = lifecycle.extract_lifecycles(docs, [chunk("DOC952", text, "22")])
    relation = result["DOC952"]["relations"][0]
    assert relation["type"] == "revision_of" and relation["target_id"] == "DOC951"
    assert relation["evidence"]["exact_span"] == statement
    assert result["DOC952"]["publication_date"] == "2022"
    assert result["DOC952"]["dates"]["publication_date"]["source"] == "metadata_inferred"
    assert result["DOC952"]["revision_date"] == "2020"
    assert result["DOC952"]["warnings"]


def test_advance_version_and_explicit_withdrawal_are_separate_supported_status_signals():
    docs = [metadata("DOC951", "Carp Guidance"), metadata("DOC952", "Withdrawal Notice"), metadata("DOC953", "Global Assessment")]
    result = lifecycle.extract_lifecycles(docs, [chunk("DOC952", "This document withdraws DOC951."),
                                                chunk("DOC953", "Unedited advance version.")])
    assert result["DOC951"]["status"] == "unknown"
    assert result["DOC952"]["relations"][0]["type"] == "withdraws"
    assert result["DOC953"]["status"] == "draft" and result["DOC953"]["warnings"]


def test_future_withdrawal_never_marks_target_globally_withdrawn():
    docs = [metadata("DOC951", "Carp Guidance", "2024"), metadata("DOC952", "Withdrawal Notice", "2027")]
    result = lifecycle.extract_lifecycles(docs, [chunk("DOC951", "Status: Final."),
        chunk("DOC952", "Published: 2027-01-01. Effective: 2027-06-01. This document withdraws DOC951.")])
    assert result["DOC951"]["status"] == "final"
    assert result["DOC952"]["effective_date"] == "2027-06-01"
    assert result["DOC952"]["relations"][0]["target_id"] == "DOC951"


@pytest.mark.parametrize("title,agency,year,text,page,expected", [
    ("Aquatic Invasive Species Research Report", "Army", "2023", "Aquatic Invasive Species Research Report Prepared by the Assistant Secretary of the Army for Civil Works June 2020", "1", "2020-06"),
    ("Carp Science Plan", "USGS", "2024", "U.S. Geological Survey, Reston, Virginia: 2023", "1", "2023"),
    ("Annual Review FY2024", "Agency", "2024", "Annual Review FY2024 By Agency | January 1, 2025", "Web", "2025-01-01"),
    ("Invasive Species Accomplishments Report", "Interior", "", "Suggested Citation U.S. Department of the Interior. 2026. Invasive Species Strategic Plan, 2021-2025: Accomplishments Report.", "1", "2026"),
    ("Convention Resources", "Ramsar Convention", "", "The Ramsar Convention Manual, 6th edition. Copyright \u00a9 Ramsar Convention Secretariat 2013", "1", "2013"),
])
def test_frontmatter_publication_signals_override_catalog_coverage_dates(title, agency, year, text, page, expected):
    result = lifecycle.extract_lifecycles([metadata("DOC951", title, year, agency)], [chunk("DOC951", text, page)])["DOC951"]
    assert result["publication_date"] == expected
    assert result["dates"]["publication_date"]["evidence"]["exact_span"] in text
    if "6th edition" in text:
        assert result["version"] == "6"
        assert result["warnings"]


def test_unrelated_frontmatter_copyright_and_suggested_reference_are_not_own_dates():
    text = "Copyright \u00a9 Unrelated Press 2013. Suggested Citation Other Author. 2026. Unrelated Species Study."
    result = lifecycle.extract_lifecycles([metadata("DOC951", "Carp Guidance 2024", "2024", "Carp Agency")], [chunk("DOC951", text)])["DOC951"]
    assert result["publication_date"] == "2024"
    assert result["dates"]["publication_date"]["source"] != "explicit"


def test_may_month_dates_are_not_rejected_as_hypothetical_language():
    text = "Published: May 2025. Revised: May 3, 2026. Effective: May 1, 2027. Status: Final."
    row = lifecycle.extract_lifecycles([metadata("DOC951", "Carp Guidance")], [chunk("DOC951", text)])["DOC951"]
    assert row["publication_date"] == "2025-05"
    assert row["revision_date"] == "2026-05-03"
    assert row["effective_date"] == "2027-05-01"
    assert row["status"] == "final"


def test_modal_may_still_cannot_create_an_explicit_replacement():
    docs = [metadata("DOC951", "Carp Guidance 2020"), metadata("DOC952", "Carp Guidance 2024")]
    row = lifecycle.extract_lifecycles(docs, [chunk("DOC952", "This guidance may supersede DOC951.")])["DOC952"]
    assert not row["relations"]


@pytest.mark.parametrize("title,expected", [
    ("Final report on draft guidance", "final"),
    ("Study of withdrawn conservation guidance", "unknown"),
    ("Guidance on final habitat outcomes", "unknown"),
    ("Carp Guidance 2024 (Draft)", "draft"),
])
def test_title_status_belongs_to_this_document_not_its_subject(title, expected):
    row = lifecycle.extract_lifecycles([metadata("DOC951", title)], [])["DOC951"]
    assert row["status"] == expected


@pytest.mark.parametrize("text,expected", [
    ("This guidance discusses the unedited advance version of an external assessment.", "unknown"),
    ("An external assessment is an unedited advance version.", "unknown"),
    ("This report is an unedited advance version.", "draft"),
    ("Assessment report on biodiversity - unedited advance version Authors Example.", "draft"),
])
def test_advance_status_requires_own_header_or_direct_self_statement(text, expected):
    row = lifecycle.extract_lifecycles([metadata("DOC951", "Carp Guidance")], [chunk("DOC951", text)])["DOC951"]
    assert row["status"] == expected


def test_dated_own_version_is_not_an_explicit_publication_date():
    row = lifecycle.extract_lifecycles([metadata("DOC951", "Comprehensive Conservation Strategy", "2022")],
        [chunk("DOC951", "The vision is accomplished in this version of the CCS (2020).", "8")])["DOC951"]
    assert row["publication_date"] == "2022" and row["revision_date"] is None
    assert row["dates"]["publication_date"]["source"] == "metadata_inferred"
    assert row["version"] == "2020" and row["warnings"]


def _ingest(store, docid, title, text, year="2024"):
    store.upsert_document_sources([DocumentSource(docid, title, "https://example.org/test.pdf", None, docid + ".pdf", "pdf", year, "Synthetic Conservation Agency")])
    store.ingest_chunk(KnowledgeArtifact(docid, title, "1", text), [1., 0.])


def test_index_is_idempotent_persistent_and_does_not_rescan_unchanged_canonical_text(tmp_path, monkeypatch):
    path = tmp_path / "lifecycle.db"
    with KnowledgeStore(path) as store:
        _ingest(store, "DOC951", "Carp Guidance", "Published: 2024. Status: Final.")
        first = lifecycle.get_lifecycles(store)
        vectors_before = [tuple(row) for row in store.connection.execute("SELECT * FROM vector_embeddings")]
        statements = []
        store.connection.set_trace_callback(statements.append)
        monkeypatch.setattr(lifecycle, "extract_lifecycles", lambda *args: pytest.fail("Unchanged index rebuilt"))
        assert lifecycle.get_lifecycles(store) == first
        assert not any("SELECT" in query.upper() and "ORIGINAL_TEXT_CHUNK" in query.upper() for query in statements)
        assert [tuple(row) for row in store.connection.execute("SELECT * FROM vector_embeddings")] == vectors_before
    with KnowledgeStore(path) as reopened:
        assert lifecycle.get_lifecycles(reopened) == first


@pytest.mark.parametrize("mutation", ["metadata", "text", "insert", "delete"])
def test_persisted_dirty_triggers_rebuild_after_canonical_changes(tmp_path, mutation):
    with KnowledgeStore(tmp_path / "lifecycle.db") as store:
        _ingest(store, "DOC951", "Carp Guidance", "Published: 2020.", "2020")
        before = lifecycle.get_lifecycles(store)
        if mutation == "metadata":
            store.connection.execute("UPDATE documents SET agency='Different agency' WHERE document_id='DOC951'")
        elif mutation == "text":
            store.connection.execute("UPDATE knowledge_artifacts SET original_text_chunk='Published: 2023.' WHERE document_id='DOC951'")
        elif mutation == "insert":
            _ingest(store, "DOC952", "Carp Guidance Revised 2024", "This guidance supersedes DOC951.")
        else:
            store.connection.execute("DELETE FROM knowledge_artifacts WHERE document_id='DOC951'")
            store.connection.execute("DELETE FROM documents WHERE document_id='DOC951'")
        store.connection.commit()
        state = store.connection.execute("SELECT revision,built_revision FROM document_lifecycle_state").fetchone()
        assert state[0] != state[1]
        after = lifecycle.get_lifecycles(store)
        assert after != before
        state = store.connection.execute("SELECT revision,built_revision FROM document_lifecycle_state").fetchone()
        assert state[0] == state[1]


def test_failed_rebuild_rolls_back_derived_index_and_keeps_canonical_data(tmp_path, monkeypatch):
    with KnowledgeStore(tmp_path / "lifecycle.db") as store:
        _ingest(store, "DOC951", "Carp Guidance", "Published: 2024.")
        previous = lifecycle.get_lifecycles(store)
        def broken(*args):
            raise ValueError("extractor failure")
        monkeypatch.setattr(lifecycle, "extract_lifecycles", broken)
        with pytest.raises(ValueError, match="extractor failure"):
            lifecycle.rebuild_lifecycles(store)
        persisted = {row[0]: json.loads(row[1]) for row in store.connection.execute("SELECT document_id,payload FROM document_lifecycles")}
        assert persisted == previous
        assert store.artifact_count == 1
