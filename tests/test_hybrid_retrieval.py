"""Controlled retrieval outcomes; all evidence is synthetic and all calls offline."""

from contextlib import contextmanager
from dataclasses import asdict, replace

import pytest

from data_models import DocumentSource, KnowledgeArtifact
from database import KnowledgeStore
from config import SETTINGS
from retrieval import classify_query, reciprocal_rank_fusion, retrieve_evidence


def chunk(docid, text, *, title="Synthetic conservation report", page="1", vector=(1., 0., 0.)):
    return KnowledgeArtifact(docid, title, page, text, source_url=f"https://example.org/{docid}.pdf"), vector


@contextmanager
def corpus(*records, agencies=None):
    with KnowledgeStore(":memory:") as store:
        for artifact, vector in records:
            store.upsert_document_sources([DocumentSource(
                artifact.document_id, artifact.title, artifact.source_url, None,
                artifact.document_id + ".pdf", "pdf",
                agency=(agencies or {}).get(artifact.document_id, "Synthetic Conservation Agency"),
            )])
            store.ingest_chunk(artifact, vector)
        yield store


@pytest.fixture(autouse=True)
def no_provider(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Retrieval fixtures must not make provider calls")
    monkeypatch.setattr("api_clients._client", forbidden)


def test_rrf_does_not_give_duplicate_entries_extra_votes():
    rankings = [[5, 2, 9], [2, 7, 5]]
    actual = reciprocal_rank_fusion(rankings)
    assert actual[0] == 2
    assert set(actual) == {2, 5, 7, 9}
    assert actual == reciprocal_rank_fusion([[5, 2, 2, 9], [2, 7, 5]])
    assert actual == reciprocal_rank_fusion(rankings)


@pytest.mark.parametrize("query,expected", [
    ('"Invasive Carp Management Strategy"', "exact_lookup"),
    ("Find the report mentioning telemetry near Lock and Dam 19.", "report_lookup"),
    ("What data show how effective invasive carp harvesting is?", "quantitative_question"),
    ("How do connected habitats protect wildlife diversity?", "semantic_question"),
])
def test_query_type_recognizes_distinct_information_needs(query, expected):
    assert classify_query(query) == expected


def test_lexical_exact_title_recovers_report_without_dense_help():
    target = chunk("DOC901", "Monitoring recommendations appear in this management document.",
                   title="Invasive Carp Management Strategy")
    distractor = chunk("DOC902", "Invasive carp management strategy is discussed repeatedly in background notes.",
                       title="General Species Notes")
    with corpus(target, distractor) as store:
        results = retrieve_evidence(store, '"Invasive Carp Management Strategy"', mode="lexical", top_k=1)
    assert results[0].document_id == "DOC901"


def test_dense_paraphrase_survives_without_shared_lexical_terms():
    target = chunk("DOC901", "Reconnect isolated marsh patches to support seasonal avian passage.",
                   title="Habitat Connectivity", vector=(1., 0., 0.))
    distractor = chunk("DOC902", "Connected corridors in office buildings improve pedestrian circulation.",
                       title="Facility Access", vector=(0., 1., 0.))
    with corpus(target, distractor) as store:
        query = "How can connected corridors aid migrating birds?"
        dense = retrieve_evidence(store, query, query_embedding=[1., 0., 0.], top_k=1, mode="dense")
        hybrid = retrieve_evidence(store, query, query_embedding=[1., 0., 0.], top_k=2, mode="hybrid")
    assert dense[0].document_id == "DOC901"
    assert "DOC901" in {artifact.document_id for artifact in hybrid}


def test_quantitative_ranking_requires_outcome_near_requested_species():
    general = chunk("DOC901", "Invasive carp control is discussed. " +
                    "Aquatic management background. " * 15 +
                    "Plant treatment reduced hydrilla cover by 80 percent.")
    measured = chunk("DOC902", "An invasive carp harvest removed 19 tons of fish and decreased carp density by 27 percent.")
    with corpus(general, measured) as store:
        results = retrieve_evidence(store, "Provide data on the effectiveness of invasive carp control.", top_k=1)
    assert results[0].document_id == "DOC902"


def test_reference_numbers_do_not_outrank_observed_outcomes():
    references = chunk("DOC901", "Invasive carp control: Journal, v. 42, no. 4, p. 899. " +
                       "Carp removal study https://doi.org/example. " * 8)
    study = chunk("DOC902", "Monitoring recorded a 27 percent decrease in invasive carp density after harvest.")
    with corpus(references, study) as store:
        results = retrieve_evidence(store, "What data show invasive carp control effectiveness?", top_k=1)
    assert results[0].document_id == "DOC902"


@pytest.mark.parametrize("unit", ["%", " percent"])
def test_near_duplicate_numeric_context_preserves_different_measured_endpoints(unit):
    common = "Invasive carp monitoring compared the same sites and used consistent sampling protocols. " * 12
    first = chunk("DOC901", common + f"Treatment reduced biomass by 60{unit} in the experiment.")
    second = chunk("DOC902", common + f"Treatment reduced egg survival by 60{unit} in the experiment.")
    with corpus(first, second) as store:
        results = retrieve_evidence(store, "What invasive carp treatment outcomes were measured?", top_k=2)
    assert {a.document_id for a in results} == {"DOC901", "DOC902"}


def test_remembered_report_detail_returns_buried_evidence_page():
    title = "Silverwater Fish Movement Report"
    records = [chunk("DOC901", f"Section {number}: general fish movement background and administrative procedures.",
                     title=title, page=str(number)) for number in range(1, 71)]
    records.append(chunk("DOC901", "Telemetry tracking near Lock and Dam 19 documented silver carp movement through the navigation channel.",
                         title=title, page="147"))
    records.append(chunk("DOC902", "Silverwater fish movement is summarized without telemetry locations.",
                         title="Regional Overview"))
    with corpus(*records) as store:
        results = retrieve_evidence(store, "Find the Silverwater Fish Movement Report mentioning telemetry tracking near Lock and Dam 19.",
                                    top_k=2, mode="hybrid_rerank")
    assert any(artifact.document_id == "DOC901" and artifact.page_number == "147" for artifact in results)


def test_actual_effectiveness_outcomes_outrank_year_and_page_numbers():
    dated = chunk("DOC901", "Invasive carp control effectiveness report 2020, 2021, 2022. Table 12, page 35. Program reference 407.",
                  title="Invasive Carp Program Register")
    measured = chunk("DOC902", "Invasive carp harvest reduced monitored biomass by 42% after targeted removal of 3000 fish during the trial.",
                     title="Invasive Carp Harvest Results")
    with corpus(dated, measured) as store:
        results = retrieve_evidence(store, "What data show how effective invasive carp control methods are?", top_k=1)
    assert results[0].document_id == "DOC902"
    assert "42%" in results[0].original_text_chunk


def test_numeric_aquatic_plant_study_does_not_displace_carp_outcomes():
    plant = chunk("DOC901", "Invasive aquatic plant control reduced vegetation by 95% and removed 5000 pounds of weeds.",
                  title="Aquatic Plant Control Trial", vector=(1., 0., 0.))
    carp = chunk("DOC902", "Targeted harvest removed 42000 invasive carp and reduced monitored biomass by 35%.",
                 title="Invasive Carp Removal Trial", vector=(0.8, 0.6, 0.))
    with corpus(plant, carp) as store:
        results = retrieve_evidence(store, "Provide quantitative effectiveness data for invasive carp control.",
                                    query_embedding=[1., 0., 0.], entity="invasive carp", top_k=1)
    assert results[0].document_id == "DOC902"


def test_near_duplicate_overlap_does_not_crowd_out_distinct_evidence():
    paragraph = (
        "Invasive carp harvest requires coordination across connected waters. Managers deploy targeted nets, "
        "record field observations, inspect collection gear, and document local conditions before reviewing "
        "monitoring results. Regional staff share removal schedules and identify habitat constraints. "
        "The report describes consistent monitoring procedures, sampling locations, seasonal access, "
        "equipment maintenance, and follow-up field assessment for the invasive carp harvest program."
    )
    with corpus(chunk("DOC901", paragraph, title="Carp Harvest Field Manual", page="12"),
                chunk("DOC901", paragraph + " Monitoring details continued.", title="Carp Harvest Field Manual", page="13"),
                chunk("DOC902", "Acoustic telemetry monitoring tracks invasive carp movement after harvest across connected river reaches.",
                      title="Carp Movement Monitoring")) as store:
        results = retrieve_evidence(store, "invasive carp harvest monitoring", top_k=3)
    assert sum(artifact.document_id == "DOC901" for artifact in results) == 1
    assert "DOC902" in {artifact.document_id for artifact in results}


def test_similar_findings_with_different_numbers_or_negation_remain_distinct():
    background = (
        "Investigators monitored invasive carp abundance before and after targeted commercial harvest. "
        "The study used comparable sampling locations, standardized equipment, seasonal sampling, "
        "independent observations, and documented field procedures across the treated river reach. "
    )
    records = [chunk(docid, background + finding, title="Carp Harvest Study") for docid, finding in (
        ("DOC901", "Invasive carp abundance declined by 60% following harvest."),
        ("DOC902", "Invasive carp abundance declined by 6% following harvest."),
        ("DOC903", "Invasive carp abundance did not decline following harvest."),
    )]
    with corpus(*records) as store:
        results = retrieve_evidence(store, "What measured outcomes show the effectiveness of invasive carp harvest?", top_k=3)
    assert {artifact.document_id for artifact in results} == {"DOC901", "DOC902", "DOC903"}


def test_two_strong_pages_from_same_report_are_not_artificially_excluded():
    title = "Invasive Carp Harvest and Monitoring Results"
    with corpus(chunk("DOC901", "Invasive carp harvest removed 3000 fish from the sampled river reach during field operations.", title=title, page="10"),
                chunk("DOC901", "Invasive carp monitoring measured 55% survival after harvest and documented movement outcomes.", title=title, page="108"),
                chunk("DOC902", "Invasive carp are discussed in background introductions to river ecology.", title="Species Background", vector=(0., 1., 0.))) as store:
        results = retrieve_evidence(store, "Compare invasive carp harvest removal totals and monitoring survival outcomes.", top_k=2)
    assert {(artifact.document_id, artifact.page_number) for artifact in results} == {("DOC901", "10"), ("DOC901", "108")}


def test_explicit_document_id_and_allowlist_are_respected():
    with corpus(chunk("DOC901", "Invasive carp control uses targeted harvest."),
                chunk("DOC902", "Invasive carp control includes movement monitoring."),
                chunk("DOC903", "Regional habitat monitoring recommendations.")) as store:
        exact = retrieve_evidence(store, "Find DOC903", top_k=1)
        filtered = retrieve_evidence(store, "invasive carp control", document_ids=["DOC902"], top_k=5)
    assert exact[0].document_id == "DOC903"
    assert [artifact.document_id for artifact in filtered] == ["DOC902"]


def test_ranked_evidence_preserves_all_canonical_provenance_fields():
    record = chunk("DOC901", "Harvest removed 3000 invasive carp during field monitoring.",
                   title="Carp [Field] Results", page="14")
    record = replace(record[0], printed_page_label="viii"), record[1]
    before = asdict(record[0])
    with corpus(record) as store:
        results = retrieve_evidence(store, "invasive carp harvest results", query_embedding=[1., 0., 0.])
        canonical = store.retrieve([1., 0., 0.], 1)[0]
    assert asdict(results[0]) == asdict(canonical) == before
    assert not hasattr(results[0], "score") and not hasattr(results[0], "embedding")
    assert asdict(record[0]) == before


def test_new_zebra_mussel_target_does_not_reuse_previous_carp_ranking():
    with corpus(chunk("DOC901", "Invasive carp harvest removes fish from connected rivers.", title="Carp Management"),
                chunk("DOC902", "Zebra mussels foul intake pipes and increase maintenance costs.", title="Zebra Mussel Impacts", vector=(0., 1., 0.))) as store:
        first = retrieve_evidence(store, "invasive carp harvest", entity="invasive carp", top_k=1)
        second = retrieve_evidence(store, "What are the impacts of zebra mussels?", entity="zebra mussels", top_k=1)
    assert first[0].document_id == "DOC901"
    assert second[0].document_id == "DOC902"


def test_known_scientific_alias_recovers_common_name_evidence():
    with corpus(chunk("DOC901", "Zebra mussels attach to submerged structures and obstruct water intake pipes.", title="Zebra Mussel Impacts"),
                chunk("DOC902", "Native freshwater fish use river channels.", title="Fish Habitats")) as store:
        results = retrieve_evidence(store, "Dreissena polymorpha impacts", top_k=1)
    assert results[0].document_id == "DOC901"


def test_exact_issuing_agency_metadata_improves_report_lookup():
    with corpus(chunk("DOC901", "Invasive carp management and monitoring recommendations.", title="Carp Report"),
                chunk("DOC902", "Invasive carp management and monitoring recommendations are reviewed in greater detail.", title="Carp Report"),
                agencies={"DOC901": "Agency Alpha", "DOC902": "Agency Beta"}) as store:
        results = retrieve_evidence(store, "What reports from Agency Alpha discuss invasive carp?", top_k=1)
    assert results[0].document_id == "DOC901"


def test_candidate_pool_and_final_evidence_are_bounded_and_diagnosable():
    records = [chunk(f"DOC{900 + index:03d}", f"Invasive carp harvest monitoring recorded {index} fish in a regional trial.",
                     title=f"Carp Trial {index}") for index in range(1, 81)]
    diagnostics = {}
    with corpus(*records) as store:
        results = retrieve_evidence(store, "invasive carp harvest monitoring", query_embedding=[1., 0., 0.],
                                    top_k=3, diagnostics=diagnostics)
    assert 0 < len(results) <= 3
    assert diagnostics
    assert len({(artifact.document_id, artifact.page_number, artifact.original_text_chunk) for artifact in results}) == len(results)
    limits = SETTINGS.retrieval
    assert len(diagnostics["dense_candidates"]) <= max(3, limits.dense_top_k)
    assert len(diagnostics["lexical_candidates"]) <= max(3, limits.lexical_top_k)
    assert len(diagnostics["document_candidates"]) <= limits.document_top_k
    assert len(diagnostics["document_chunk_candidates"]) <= limits.document_top_k * limits.document_chunk_k
    assert len(diagnostics["fused_ranking"]) <= max(3, limits.fusion_top_k)
    assert len(diagnostics["reranked_ids"]) <= max(3, limits.rerank_top_k)
    assert len(diagnostics["final_artifact_ids"]) == len(results)
    assert diagnostics["final_document_ids"] == [artifact.document_id for artifact in results]


def test_hybrid_without_query_embedding_falls_back_to_lexical_evidence():
    with corpus(chunk("DOC901", "Invasive carp harvest monitoring supplies management evidence.")) as store:
        results = retrieve_evidence(store, "invasive carp harvest", mode="hybrid", top_k=1)
    assert [artifact.document_id for artifact in results] == ["DOC901"]


def test_malformed_query_punctuation_cannot_execute_sql_or_break_literal_search():
    with corpus(chunk("DOC901", "Invasive carp harvest monitoring supplies management evidence.")) as store:
        before = store.artifact_count
        retrieve_evidence(store, 'carp" OR 1=1; DROP TABLE knowledge_artifacts; -- NEAR("harvest")', mode="lexical")
        assert store.artifact_count == before
        results = retrieve_evidence(store, "invasive carp harvest", mode="lexical", top_k=1)
    assert results[0].document_id == "DOC901"


@pytest.mark.parametrize("query,top_k,mode", [
    ("", 5, "hybrid_rerank"),
    ("carp", 0, "hybrid_rerank"),
    ("carp", True, "hybrid_rerank"),
    ("carp", 5, "unknown_mode"),
])
def test_invalid_retrieval_arguments_fail_before_provider_or_store_mutation(query, top_k, mode):
    with corpus(chunk("DOC901", "Invasive carp harvest monitoring.")) as store:
        before = store.artifact_count
        with pytest.raises((TypeError, ValueError)):
            retrieve_evidence(store, query, top_k=top_k, mode=mode)
        assert store.artifact_count == before
