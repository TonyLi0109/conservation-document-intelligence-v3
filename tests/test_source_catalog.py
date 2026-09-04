"""Regression checks for manually curated document provenance."""

from source_catalog import load_source_catalog


def test_documentcloud_resolved_urls_match_ingested_documents() -> None:
    catalog = load_source_catalog()
    expected_urls = {
        "DOC027": "https://web.archive.org/web/20240115041420id_/https://www.ramsar.org/sites/default/files/documents/library/manual6-2013-e.pdf",
        "DOC031": "https://s3.documentcloud.org/documents/24223250/epa-2000-report-on-americas-water-resources.pdf",
        "DOC032": "https://s3.documentcloud.org/documents/5990576/Summary-for-Policymakers-IPBES-Global-Assessment.pdf",
        "DOC033": "https://s3.documentcloud.org/documents/4911870/Cleveland-Climate-Action-Plan-2018.pdf",
        "DOC034": "https://s3.documentcloud.org/documents/6563087/Canada-s-Green-Plan-1990.pdf",
        "DOC035": "https://s3.documentcloud.org/documents/3105657/Wetland-Protection.pdf",
    }

    assert {
        document_id: catalog[document_id].resolved_url
        for document_id in expected_urls
    } == expected_urls
