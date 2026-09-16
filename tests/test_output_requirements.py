"""Presentation/status/provenance requirements without retrieval changes."""
import json

from data_models import KnowledgeArtifact
from pipeline_tracer import PipelineTracer
from validation_presentation import validate_format_and_log


def artifact(doc_id="DOC001", title="Wetland Plan", page="9", printed="7", text="Wetlands support migratory birds."):
    return KnowledgeArtifact(doc_id, title, page, text, printed_page_label=printed)


def envelope(claims, facets=(), status=None):
    return json.dumps({
        "status": status or ("partially_answered" if facets else "answered"),
        "claims": claims,
        "unsupported_facets": list(facets),
    })


def claim(text, evidence_id, span):
    return {"text": text, "evidence_ids": [evidence_id], "supporting_spans": [span]}


def test_claim_statuses_and_required_jsonl_fields(tmp_path, monkeypatch):
    log = tmp_path / "claims.jsonl"
    monkeypatch.setenv("V3_PROVENANCE_LOG", str(log))
    source = artifact()
    payload = envelope([
        claim("Wetlands support birds.", "K1", source.original_text_chunk),
        claim("Wetlands eliminate all flooding.", "K1", "This span is absent."),
    ], ["Statewide cost estimate"])
    rendered, sources = validate_format_and_log(payload, {"K1": source}, query="Summarize wetland findings")
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [record["validation_status"] for record in records] == [
        "SUPPORTED", "UNSUPPORTED", "INSUFFICIENT_EVIDENCE"]
    assert set(records[0]) == {"claim_id", "claim_text", "doc_id", "printed_page",
                               "pdf_page", "supporting_evidence_span", "validation_status"}
    assert records[0]["doc_id"] == "DOC001"
    assert records[0]["printed_page"] == "7" and records[0]["pdf_page"] == "9"
    assert records[0]["supporting_evidence_span"] == source.original_text_chunk
    assert "eliminate all flooding" not in rendered
    assert "**Unsupported facets**" in rendered
    assert "INSUFFICIENT_EVIDENCE" in rendered
    assert sources == [source]


def test_partially_supported_claim_is_logged_but_not_rendered(tmp_path, monkeypatch):
    log = tmp_path / "claims.jsonl"
    monkeypatch.setenv("V3_PROVENANCE_LOG", str(log))
    first = artifact()
    second = artifact("DOC002", "Bird Plan", "3", None, "Bird habitat is monitored.")
    payload = envelope([{"text": "Both plans establish findings.",
                         "evidence_ids": ["K1", "K2"],
                         "supporting_spans": [first.original_text_chunk]}])
    rendered, sources = validate_format_and_log(
        payload, {"K1": first, "K2": second}, query="Compare both plans")
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert {record["validation_status"] for record in records} == {"PARTIALLY_SUPPORTED"}
    assert rendered.startswith("The generated answer could not be verified")
    assert sources == []


def test_comparison_and_management_templates_keep_citations_adjacent():
    first = artifact()
    second = artifact("DOC002", "Bird Plan", "3", None, "Monitoring detects habitat change.")
    payload = envelope([
        claim("Wetlands support birds.", "K1", first.original_text_chunk),
        claim("Monitoring detects change.", "K2", second.original_text_chunk),
    ])
    compared, _ = validate_format_and_log(
        payload, {"K1": first, "K2": second}, query="Compare the two plans")
    assert "### DOC001: Wetland Plan" in compared
    assert "### DOC002: Bird Plan" in compared
    assert "DOC001 ?" not in compared and "DOC002 ?" not in compared
    assert "Wetlands support birds. [DOC001" in compared
    managed, _ = validate_format_and_log(
        payload, {"K1": first, "K2": second}, query="Recommend management next steps")
    assert "**Recommendations by category**" in managed
    assert "**Implementation sequence**" in managed
    assert "2. Monitoring detects change. [DOC002" in managed


def test_quantitative_evidence_preempts_management_template():
    source = artifact(
        title="Invasive Carp Results",
        text="Commercial harvest removed 1.2 million pounds of invasive carp.",
    )
    fact = "Commercial harvest removed 1.2 million pounds of invasive carp."
    payload = envelope([claim(fact, "K1", source.original_text_chunk)])
    query = ("What evidence in the corpus shows that invasive carp control efforts "
             "have been effective in Missouri? Give quantitative results where available.")

    rendered, _ = validate_format_and_log(payload, {"K1": source}, query=query)

    assert rendered.count(fact) == 1
    assert rendered.startswith("- " + fact)
    assert "[DOC001" in rendered
    assert "**Recommendations by category**" not in rendered
    assert "**Implementation sequence**" not in rendered

def test_management_template_owns_list_numbering():
    first = artifact(text="1. Map wetland baselines.")
    second = artifact("DOC002", "Runoff Plan", "3", None,
                      "2) Install runoff controls.")
    payload = envelope([
        claim(first.original_text_chunk, "K1", first.original_text_chunk),
        claim(second.original_text_chunk, "K2", second.original_text_chunk),
    ])

    rendered, _ = validate_format_and_log(
        payload, {"K1": first, "K2": second},
        query="What management actions should I implement? Provide next steps.",
    )

    assert "- Map wetland baselines." in rendered
    assert "- Install runoff controls." in rendered
    assert "1. Map wetland baselines." in rendered
    assert "2. Install runoff controls." in rendered
    assert "- 1. Map" not in rendered
    assert "- 2) Install" not in rendered
    assert "1. 1. Map" not in rendered
    assert "2. 2) Install" not in rendered

def test_tracer_receives_exact_machine_records(tmp_path):
    source = artifact()
    payload = envelope([claim("Wetlands support birds.", "K1", source.original_text_chunk)])
    with PipelineTracer(tmp_path) as tracer:
        validate_format_and_log(payload, {"K1": source}, query="What supports birds?")
    data = json.loads(tracer.path.read_text(encoding="utf-8"))
    records = [event["value"] for event in data["events"]
               if event["stage"] == "claim_provenance"]
    assert len(records) == 1 and records[0]["validation_status"] == "SUPPORTED"
