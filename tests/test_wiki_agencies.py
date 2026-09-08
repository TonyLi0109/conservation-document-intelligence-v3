"""Offline checks for substantive agency pages and proven Wiki relationships."""

from contextlib import contextmanager
import sqlite3

import pytest

import wiki_compiler as wiki
from config import SETTINGS
from data_models import DocumentSource, KnowledgeArtifact
from database import KnowledgeStore


USACE = "U.S. Army Corps of Engineers"
USFWS = "U.S. Fish and Wildlife Service"
MDC = "Missouri Department of Conservation"
DOI = "U.S. Department of the Interior"


@pytest.fixture(autouse=True)
def forbid_wiki_provider_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Preloaded Wiki compilation must not call an API")

    monkeypatch.setattr(wiki, "generate_embedding", forbidden)
    monkeypatch.setattr(wiki, "call_structured_llm", forbidden)


@contextmanager
def small_store(tmp_path, *texts):
    with KnowledgeStore(tmp_path / "agency-wiki.db") as store:
        for index, text in enumerate(texts, 1):
            document_id = f"DOC{index:03d}"
            store.upsert_document_sources([DocumentSource(
                document_id, "Agency conservation report",
                f"https://example.org/conservation/{index}.pdf", None,
                f"{index}.pdf", "pdf",
            )])
            store.ingest_chunk(KnowledgeArtifact(
                document_id, "Agency conservation report", str(index), text,
                printed_page_label=f"report-{index}",
            ), [1.0, 0.0])
        yield store


def assert_canonical_provenance(page, store):
    evidence = page["concept"]["supporting_evidence"]
    for item in evidence + page["concept"]["related_entities"]:
        artifact = page["artifacts"][item["evidence_id"]]
        assert item["exact_span"] in artifact.original_text_chunk
        canonical = store.connection.execute(
            "SELECT original_text_chunk, printed_page_label FROM knowledge_artifacts "
            "WHERE document_id=? AND page_number=?",
            (artifact.document_id, artifact.page_number),
        ).fetchall()
        assert any(artifact.original_text_chunk == row[0]
                   and artifact.printed_page_label == row[1] for row in canonical)
        if item in page["concept"]["related_entities"]:
            assert {key: item[key] for key in ("evidence_id", "exact_span")} in evidence


@pytest.mark.parametrize("agency", [USACE, USFWS])
def test_agency_abbreviation_does_not_split_a_verbatim_sentence(agency):
    statement = f"The {agency} manages wetlands in Missouri to conserve native wildlife."
    text = "This introduction describes the scope of the conservation report. " + statement
    assert statement in wiki._allowed_spans(text, agency)


@pytest.mark.parametrize("agency", [USACE, USFWS])
def test_public_agency_page_prioritizes_work_over_reference_and_address(tmp_path, agency):
    statement = f"The {agency} manages wetlands in Missouri to conserve native wildlife."
    noise = (
        f"References cited: {agency}. 2020. Annual report. "
        f"Contact {agency}, 100 Main Street, Washington, DC 20001."
    )
    with small_store(tmp_path, noise, statement) as store:
        page = wiki.generate_extractive_wiki_concept(agency, store)
        assert statement in page["concept"]["summary"]
        assert page["concept"]["important_facts"][0] == statement
        assert {"Wetland", "Missouri"} <= {
            item["entity_name"] for item in page["concept"]["related_entities"]
        }
        assert_canonical_provenance(page, store)


@pytest.mark.parametrize("agency,alias", [(USACE, "USACE"), (USFWS, "USFWS"),
                                           (MDC, "MDC"), (DOI, "DOI")])
def test_public_agency_page_resolves_known_body_aliases(tmp_path, agency, alias):
    statement = f"{alias} manages wetlands in Missouri to protect native fish and wildlife."
    with small_store(tmp_path, statement) as store:
        page = wiki.generate_extractive_wiki_concept(agency, store)
        assert page["concept"]["concept_title"] == agency
        assert statement in page["concept"]["important_facts"]
        assert "Wetland" in {item["entity_name"] for item in page["concept"]["related_entities"]}
        assert_canonical_provenance(page, store)


@pytest.mark.parametrize("topic,related,statement", [
    (USACE, USFWS, "USACE works with USFWS to restore wetlands and conserve native wildlife."),
    (USFWS, USACE, "USACE works with USFWS to restore wetlands and conserve native wildlife."),
    (MDC, USFWS, "MDC, in cooperation with the U.S. Fish and Wildlife Service, has developed a habitat conservation plan."),
    ("Wetland", USFWS, "The U.S. Fish and Wildlife Service manages wetlands to protect native fish and wildlife."),
    ("Wetland", "Silver carp", "Wetlands provide habitat for silver carp in connected river floodplains."),
    ("Silver carp", "Wetland", "Wetlands provide habitat for silver carp in connected river floodplains."),
    ("Wetland", "Climate change", "Climate change threatens wetlands by altering seasonal flooding and water availability."),
    ("Climate change", "Wetland", "Climate change threatens wetlands by altering seasonal flooding and water availability."),
])
def test_explicit_relationships_work_in_both_directions(topic, related, statement):
    page = wiki._extractive_payload(topic, {"K1": [statement]})
    matches = [item for item in page["related_entities"] if item["entity_name"] == related]
    assert matches
    assert all(item["evidence_id"] == "K1" and item["exact_span"] == statement for item in matches)


@pytest.mark.parametrize("statement", [
    "USACE and USFWS appear in the report's list of abbreviations and agency contacts.",
    "USACE does not manage wetlands in Missouri and has no restoration agreement with USFWS.",
    "There is no evidence that USACE works with USFWS to restore wetlands in Missouri.",
    "USACE studies construction materials while USFWS manages wetlands in Missouri.",
])
def test_agency_mentions_do_not_manufacture_a_relationship(statement):
    assert wiki._extractive_payload(USACE, {"K1": [statement]})["related_entities"] == []


@pytest.mark.parametrize("statement", [
    "If USACE manages wetlands in Missouri, the state will request a new agreement.",
    "If USACE works with USFWS, wetland restoration could proceed after approval.",
    "USACE manages construction projects while wetlands are restored by volunteers.",
    "USACE manages construction projects, and wetlands are restored by volunteers.",
    "USACE manages a national database, and USFWS manages wetlands in Missouri.",
    "USACE and USFWS work on separate conservation projects in different regions.",
    "USACE and USFWS conducted separate surveys of their respective project sites.",
    "USACE manages payroll services for the U.S. Forest Service under an accounting agreement.",
])
def test_agency_relationships_respect_conditionals_clauses_and_entity_names(statement):
    assert wiki._extractive_payload(USACE, {"K1": [statement]})["related_entities"] == []


def test_agency_name_does_not_create_a_nested_location_relation():
    statement = "USACE works with Missouri Department of Conservation to restore natural areas."
    relations = wiki._extractive_payload(USACE, {"K1": [statement]})["related_entities"]
    assert {item["entity_name"] for item in relations} == {MDC}


def test_parenthetical_agency_aliases_preserve_the_canonical_partnership():
    statement = (
        "The U.S. Army Corps of Engineers (USACE) works with the "
        "U.S. Fish and Wildlife Service (USFWS) to restore native habitat."
    )
    for topic, related in ((USACE, USFWS), (USFWS, USACE)):
        relations = wiki._extractive_payload(topic, {"K1": [statement]})["related_entities"]
        assert relations == [{"entity_name": related, "relationship_type": "works with",
                              "evidence_id": "K1", "exact_span": statement}]


def test_interacting_threats_retain_the_sources_modal_qualification():
    statement = "Climate change may interact with other threats such as invasive species in freshwater ecosystems."
    for topic, related in (("Climate change", "Invasive species"),
                           ("Invasive species", "Climate change")):
        relations = wiki._extractive_payload(topic, {"K1": [statement]})["related_entities"]
        assert relations == [{"entity_name": related, "relationship_type": "may interact with",
                              "evidence_id": "K1", "exact_span": statement}]


def test_wetland_habitat_subtype_edges_have_opposite_directions():
    statement = "Wetland types include marshes along seasonally flooded shorelines."
    for topic, related, relationship in (("Wetland", "Marsh", "includes habitat type"),
                                          ("Marsh", "Wetland", "type of habitat")):
        relations = wiki._extractive_payload(topic, {"K1": [statement]})["related_entities"]
        assert relations == [{"entity_name": related, "relationship_type": relationship,
                              "evidence_id": "K1", "exact_span": statement}]


def test_containment_evidence_never_claims_carp_occur_in_protected_great_lakes():
    statement = (
        "Management has contained invasive carp within established ranges, "
        "preventing their spread into the Great Lakes."
    )
    for topic, related, relationship in (
        ("Invasive carp", "Great Lakes", "spread prevention protects"),
        ("Great Lakes", "Invasive carp", "protected from spread of"),
    ):
        relations = wiki._extractive_payload(topic, {"K1": [statement]})["related_entities"]
        assert relations == [{"entity_name": related, "relationship_type": relationship,
                              "evidence_id": "K1", "exact_span": statement}]


def test_new_relationship_rules_keep_negation_conditionals_and_clauses_scoped():
    examples = [
        ("Climate change", "There is no evidence that climate change may interact with invasive species in this watershed."),
        ("Wetland", "If wetland types include marshes, the inventory will require a revised habitat map."),
        ("Climate change", "Climate change affects seasonal flooding while pollution may interact with invasive species."),
        ("Invasive carp", "Management has not contained invasive carp within established ranges, preventing their spread into the Great Lakes."),
    ]
    for topic, statement in examples:
        assert wiki._extractive_payload(topic, {"K1": [statement]})["related_entities"] == [], statement


def test_reference_text_does_not_supply_an_agency_partnership():
    statement = "References cited: USACE and USFWS work to restore wetlands. Annual report, 2020."
    assert wiki._extractive_payload(USACE, {"K1": [statement]})["related_entities"] == []


@pytest.fixture
def real_corpus_copy(tmp_path):
    destination = tmp_path / "real-wiki.db"
    # Read-only backup excludes transient journal state and leaves the deployment
    # seed untouched while the compiler creates its derived indexes and caches.
    with sqlite3.connect(SETTINGS.storage.database_path.resolve().as_uri() + "?mode=ro", uri=True) as source:
        with sqlite3.connect(destination) as target:
            source.backup(target)
    with KnowledgeStore(destination) as store:
        yield store


@pytest.mark.integration
def test_real_usace_page_retains_research_role_and_its_document(real_corpus_copy):
    page = wiki.generate_extractive_wiki_concept(USACE, real_corpus_copy, force_refresh=True)
    relevant = [item for item in page["concept"]["supporting_evidence"]
                if page["artifacts"][item["evidence_id"]].document_id == "DOC006"
                and "research activities performed by the U.S. Army Corps of Engineers" in item["exact_span"]
                and "aquatic invasive species" in item["exact_span"]]
    assert relevant, "The agency's own research report must survive retrieval and U.S. sentence handling"
    assert any(item["exact_span"] in page["concept"]["important_facts"] for item in relevant)
    assert page["concept"]["related_entities"]
    assert_canonical_provenance(page, real_corpus_copy)


@pytest.mark.integration
def test_real_usfws_page_retains_wetlands_role_and_supported_relations(real_corpus_copy):
    page = wiki.generate_extractive_wiki_concept(USFWS, real_corpus_copy, force_refresh=True)
    relevant = [item for item in page["concept"]["supporting_evidence"]
                if page["artifacts"][item["evidence_id"]].document_id == "DOC022"
                and "principal federal agency" in item["exact_span"]
                and "wetland and deepwater habitats" in item["exact_span"]]
    assert relevant, "The agency's explicit wetlands information role must survive compilation"
    assert any(item["exact_span"] in page["concept"]["important_facts"] for item in relevant)
    assert {"Wetland", "Missouri", MDC} & {
        item["entity_name"] for item in page["concept"]["related_entities"]
    }
    assert_canonical_provenance(page, real_corpus_copy)
