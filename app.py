"""Streamlit presentation layer for Conservation Document Intelligence V3."""

from __future__ import annotations

import os

import streamlit as st

from config import CHAT_MODEL_OPTIONS, SETTINGS
from database import KnowledgeStore
from evaluation import run_evaluation
from main import ask_chatbot_with_context, search_corpus
from source_catalog import load_source_catalog
from validator import format_artifact_location
from wiki_compiler import generate_extractive_wiki_concept, generate_wiki_concept


APP_TITLE = "Conservation Document Intelligence"
RESEARCH_DISCLAIMER = (
    "Experimental research prototype: AI-generated answers may contain errors. "
    "Verify important conclusions using the cited source documents."
)


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --forest: #173f35;
        --moss: #39755f;
        --sage: #e9f2ed;
        --paper: #fbfcfa;
        --ink: #17231f;
    }
    .stApp { background: linear-gradient(180deg, #f6faf7 0%, var(--paper) 18rem); }
    .block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 8rem; }
    h1, h2, h3 { color: var(--forest); letter-spacing: -0.025em; }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #dce8e1;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        box-shadow: 0 5px 18px rgba(23, 63, 53, 0.05);
    }
    div[data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.82);
        border: 1px solid #dce8e1;
        border-radius: 14px;
    }
    /* Streamlit renders chat_input inline when it lives inside a tab. Pin it
       to the viewport like a conventional chat composer. The inactive tab
       panel remains hidden, so this does not leak into the other four tabs. */
    div[data-testid="stChatInput"] {
        position: fixed;
        left: 50%;
        bottom: 1rem;
        transform: translateX(-50%);
        width: min(1120px, calc(100vw - 3rem));
        z-index: 1000;
        padding: .55rem;
        border: 1px solid rgba(57, 117, 95, .22);
        border-radius: 16px;
        background: rgba(251, 252, 250, .94);
        box-shadow: 0 12px 35px rgba(23, 63, 53, .16);
        backdrop-filter: blur(12px);
    }
    @media (max-width: 640px) {
        div[data-testid="stChatInput"] {
            bottom: .5rem;
            width: calc(100vw - 1rem);
        }
    }
    button[kind="primary"] { background: var(--forest); border-color: var(--forest); }
    .v3-kicker {
        color: var(--moss); font-size: .78rem; font-weight: 700;
        letter-spacing: .13em; text-transform: uppercase; margin-bottom: .35rem;
    }
    .v3-subtitle { color: #53655e; font-size: 1.03rem; margin-top: -.55rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_store() -> KnowledgeStore:
    """Load the persistent SQLite corpus and NumPy vector cache once."""

    store = KnowledgeStore()
    store.upsert_document_sources(list(load_source_catalog().values()))
    return store


def render_corpus_tab(store: KnowledgeStore) -> None:
    """Render canonical document summaries from the V3 storage boundary."""

    st.header("Corpus")
    st.caption(
        "Canonical, page-aware evidence currently available to retrieval and synthesis."
    )
    documents = store.list_documents()
    total_pages = sum(int(item["Indexed pages"]) for item in documents)
    metrics = st.columns(3)
    metrics[0].metric("Documents", len(documents))
    metrics[1].metric("Canonical chunks", store.artifact_count)
    metrics[2].metric("Indexed page labels", total_pages)

    if not documents:
        st.info(
            "The persistent corpus is empty. Build it with "
            "`python main.py --ingest <pdf_directory>`."
        )
        return

    query = st.text_input(
        "Filter corpus",
        placeholder="Filter by document ID or title",
        key="corpus_filter",
    ).strip().casefold()
    visible = [
        item
        for item in documents
        if not query
        or query in str(item["Document ID"]).casefold()
        or query in str(item["Title"]).casefold()
    ]
    st.dataframe(
        visible,
        width="stretch",
        hide_index=True,
        column_config={
            "Document ID": st.column_config.TextColumn(width="small"),
            "Title": st.column_config.TextColumn(width="large"),
            "Indexed pages": st.column_config.NumberColumn(format="%d"),
            "Chunks": st.column_config.NumberColumn(format="%d"),
            "Source URL": st.column_config.LinkColumn(
                "Source", display_text="Open source"
            ),
        },
    )
    st.caption(f"Showing {len(visible)} of {len(documents)} documents.")
    reports = store.list_ingestion_reports()
    if reports:
        with st.expander("Ingestion coverage and source integrity"):
            st.caption(
                "Coverage is based on successful page text extraction. SHA-256 identifies "
                "the exact local source version used to build the index."
            )
            st.dataframe(
                reports,
                width="stretch",
                hide_index=True,
                column_config={
                    "Text coverage": st.column_config.ProgressColumn(
                        min_value=0.0, max_value=1.0, format="percent"
                    ),
                    "Processed coverage": st.column_config.ProgressColumn(
                        min_value=0.0, max_value=1.0, format="percent"
                    ),
                    "SHA-256": st.column_config.TextColumn(width="large"),
                },
            )


def render_chatbot_tab(store: KnowledgeStore, selected_model: str) -> None:
    """Render session chat while delegating all answers to the V3 engine."""

    st.header("Citation-based chatbot")
    st.caption(
        "Answers use retrieved corpus evidence only. Every displayed citation is "
        "resolved and validated locally before rendering."
    )

    if store.artifact_count == 0:
        st.warning("The corpus index is empty. Run the ingestion command before asking questions.")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        # api_clients also loads .env.local; this notice is intentionally advisory.
        st.caption("API credentials will be loaded from the local V3 environment file.")

    if "v3_chat_messages" not in st.session_state:
        st.session_state.v3_chat_messages = []

    def render_sources(sources: list[object], *, key_namespace: str) -> None:
        """Render only backend-validated cited artifacts."""

        if not sources:
            return
        with st.expander(f"Sources ({len(sources)})"):
            for index, source in enumerate(sources, start=1):
                page = format_artifact_location(source)
                st.markdown(
                    f"**{index}. {source.title}**  \n"
                    f"`{source.document_id}` · {page}"
                )
                snippet = source.original_text_chunk[:500].strip()
                if len(source.original_text_chunk) > 500:
                    snippet += "…"
                st.text(snippet)
                if source.source_url:
                    st.link_button(
                        "Open source document",
                        source.source_url,
                        width="content",
                        key=(
                            f"chat_source_{key_namespace}_{index}_"
                            f"{source.document_id}_{source.page_number}"
                        ),
                    )
                if index < len(sources):
                    st.divider()

    for message_index, message in enumerate(st.session_state.v3_chat_messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                if message.get("preamble"):
                    st.info(message["preamble"])
                st.markdown(message["content"])
                render_sources(
                    message.get("sources", []),
                    key_namespace=f"history_{message_index}",
                )
            else:
                st.write(message["content"])

    question = st.chat_input(
        "Ask a question about the conservation corpus",
        max_chars=2_000,
    )
    if not question:
        return

    st.session_state.v3_chat_messages.append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and validating evidence..."):
            try:
                answer, preamble, validated_sources = ask_chatbot_with_context(
                    question,
                    store,
                    model=selected_model,
                )
            except Exception as error:
                print(f"\n[DEBUG] Chatbot failed: {repr(error)}\n")
                st.error(
                    "Neither semantic nor keyword retrieval could produce a response. "
                    f"Error details: {error}"
                )
                return
        # This Markdown has already passed V3's fail-closed validator. HTML stays disabled.
        if preamble:
            st.info(preamble)
        st.markdown(answer, unsafe_allow_html=False)
        render_sources(
            validated_sources,
            key_namespace=f"current_{len(st.session_state.v3_chat_messages)}",
        )
    st.session_state.v3_chat_messages.append(
        {
            "role": "assistant",
            "preamble": preamble,
            "content": answer,
            "sources": validated_sources,
        }
    )


def render_search_tab(store: KnowledgeStore) -> None:
    """Render the legacy-inspired keyword/semantic retrieval controls."""

    st.header("Search")
    st.caption(
        f"Ranking uses `{SETTINGS.models.embedding_model}` embeddings or keyword "
        "matching. The global GPT model affects generative features, not retrieval."
    )
    st.caption(
        "Choose literal keyword ranking or semantic similarity, then inspect "
        "the exact canonical chunks returned to downstream synthesis."
    )
    if store.artifact_count == 0:
        st.warning("The corpus index is empty. Run the ingestion command before searching.")
        return

    with st.form("v3_corpus_search"):
        query = st.text_input(
            "Search the conservation corpus",
            placeholder="e.g. wetland restoration or climate adaptation",
        )
        control_col1, control_col2 = st.columns(2)
        search_method = control_col1.selectbox(
            "Search method",
            ["Keyword Search", "Semantic Search"],
        )
        top_k = control_col2.selectbox(
            "Number of results",
            [3, 5, 10],
            index=1,
        )
        submitted = st.form_submit_button("Search", type="primary")

    if submitted:
        if not query.strip():
            st.warning("Enter a search query.")
        else:
            with st.spinner(f"Running {search_method.lower()}..."):
                try:
                    if search_method == "Semantic Search":
                        results = search_corpus(query, store, top_k=top_k)
                    else:
                        results = store.retrieve(
                            None,
                            top_k,
                            method="keyword",
                            query_text=query,
                        )
                    st.session_state.v3_search_results = results
                    st.session_state.v3_search_query = query
                    st.session_state.v3_search_method = search_method
                except Exception as error:
                    st.session_state.v3_search_results = []
                    print(f"\n[DEBUG] Corpus search failed: {repr(error)}\n")
                    st.error("Search could not be completed safely. Check the server logs.")

    results = st.session_state.get("v3_search_results")
    if results is None:
        st.info("Enter a query to search the conservation corpus.")
        return
    if not results:
        st.info("No matching evidence chunks were found.")
        return

    st.subheader(f"Results for “{st.session_state.get('v3_search_query', '')}”")
    st.caption(f"Ranked with {st.session_state.get('v3_search_method', 'Search')}.")
    for rank, artifact in enumerate(results, start=1):
        with st.container(border=True):
            heading, location = st.columns([4, 1])
            heading.markdown(f"#### {rank}. {artifact.title}")
            location.markdown(f"**{format_artifact_location(artifact)}**")
            st.caption(f"{artifact.document_id} · Canonical source chunk")
            with st.expander("View exact source text", expanded=rank == 1):
                st.text(artifact.original_text_chunk)


def render_wiki_tab(store: KnowledgeStore, selected_model: str) -> None:
    """Compile and render a provenance-backed concept page."""

    st.header("Concept Wiki")
    st.caption(
        "The Wiki will compile recurring species, habitats, threats, places, "
        "agencies, and relationships while retaining artifact-level provenance."
    )
    if store.artifact_count == 0:
        st.warning("The corpus index is empty. Run ingestion before compiling concepts.")
        return

    entity_catalog = store.list_wiki_entities()
    if not entity_catalog:
        st.info("No recognized Wiki entities occur in the indexed corpus.")
        return

    selector_col1, selector_col2 = st.columns(2)
    entity_type = selector_col1.selectbox(
        "Entity type",
        list(entity_catalog),
        key="wiki_entity_type",
    )
    selected_entity = selector_col2.selectbox(
        "Entity",
        entity_catalog[entity_type],
        key="wiki_entity",
    )
    # Entity selection displays persisted content immediately. If a deployment
    # lacks a page, a local extractive version is created without an API call.
    if st.session_state.get("v3_wiki_entity") != selected_entity:
        try:
            st.session_state.v3_wiki_result = generate_extractive_wiki_concept(
                selected_entity, store
            )
            st.session_state.v3_wiki_entity = selected_entity
        except Exception as error:
            st.session_state.pop("v3_wiki_result", None)
            st.error(f"The pre-generated concept is unavailable: {error}")

    regenerate = st.button("Regenerate from evidence", type="secondary")
    if regenerate:
        with st.spinner("Retrieving and validating concept evidence..."):
            try:
                st.session_state.v3_wiki_result = generate_wiki_concept(
                    selected_entity,
                    store,
                    force_refresh=True,
                    model=selected_model,
                )
                st.session_state.v3_wiki_entity = selected_entity
            except Exception as e:
                print(f"\n[DEBUG] Wiki compilation failed: {repr(e)}\n")
                st.warning(f"Refresh failed; the pre-generated page is retained. {e}")

    result = st.session_state.get("v3_wiki_result")
    if not isinstance(result, dict):
        st.info("No pre-generated Wiki page is available for this entity.")
        return
    if result.get("refresh_error"):
        st.warning(
            "The AI refresh timed out or failed, so the existing pre-generated "
            "page is still shown. You can retry later."
        )
    concept = result.get("concept")
    artifacts = result.get("artifacts")
    if not isinstance(concept, dict) or not isinstance(artifacts, dict):
        st.error("The compiled concept is unavailable.")
        return

    st.divider()
    st.subheader(str(concept["concept_title"]))
    if result.get("knowledge_id"):
        st.caption(
            f"Reusable V3 knowledge artifact: {result['knowledge_id']} · "
            f"compiler {result.get('generation_version', 'unknown')} · "
            f"model {result.get('model_name', selected_model)} · "
            f"{'loaded from cache' if result.get('cached') else 'newly compiled'}"
        )
    st.markdown(str(concept["summary"]), unsafe_allow_html=False)

    facts = concept.get("important_facts", [])
    if facts:
        st.markdown("#### Important facts")
        for fact in facts:
            st.markdown(f"- {fact}", unsafe_allow_html=False)

    entities = concept.get("related_entities", [])
    if entities:
        st.markdown("#### Related entities")
        st.dataframe(
            [
                {
                    "Entity": entity["entity_name"],
                    "Relationship": entity["relationship_type"],
                }
                for entity in entities
            ],
            width="stretch",
            hide_index=True,
        )

    st.markdown("#### Supporting evidence")
    for number, evidence in enumerate(concept["supporting_evidence"], start=1):
        evidence_id = evidence["evidence_id"]
        artifact = artifacts.get(evidence_id)
        if artifact is None:
            continue
        page = format_artifact_location(artifact)
        citation = f"[{artifact.document_id}, {page}]"
        with st.expander(f"Evidence {number} · {citation} · {artifact.title}"):
            st.text(evidence["exact_span"])
            if artifact.source_url:
                st.link_button(
                    "Open source document",
                    artifact.source_url,
                    width="content",
                    key=f"wiki_source_{number}_{artifact.document_id}",
                )


def render_evaluation_tab(store: KnowledgeStore, selected_model: str) -> None:
    """Render metric-first scaffolding for the future evaluation harness."""

    st.header("Evaluation")
    st.caption(
        "Track retrieval integrity, grounded-answer quality, citation validity, "
        "latency, and abstention behavior."
    )
    documents = store.list_documents()
    metric_columns = st.columns(4)
    metric_columns[0].metric("Documents", len(documents))
    metric_columns[1].metric("Evidence chunks", store.artifact_count)
    metric_columns[2].metric("Retrieval Top-K", SETTINGS.retrieval.top_k)
    metric_columns[3].metric(
        "Compiled knowledge", len(store.list_compiled_concepts())
    )

    with st.container(border=True):
        st.subheader("Evaluation queries")
        st.info(
            "The fixed V3 suite measures retrieval recall and evidence-aware answer status. "
            "A run calls the embedding and LLM APIs."
        )
        st.markdown(
            """
            Future evaluation checks will cover:

            - **Relevance:** retrieved evidence addresses the question.
            - **Grounding:** every factual claim has verified supporting text.
            - **Citation quality:** document ID, title, and page resolve correctly.
            - **Completeness:** supported facets are not unnecessarily omitted.
            - **Abstention:** unsupported questions produce an explicit refusal.
            """
        )
        if st.button("Run Evaluation Suite", type="primary"):
            with st.spinner("Running the reproducible evaluation suite..."):
                try:
                    st.session_state.v3_evaluation_report = run_evaluation(
                        store, model=selected_model
                    )
                except Exception as error:
                    st.error(f"Evaluation failed: {error}")
    report = st.session_state.get("v3_evaluation_report")
    if isinstance(report, dict):
        st.subheader("Latest result")
        st.caption(f"Generation model: {report.get('model_name', 'unknown')}")
        result_columns = st.columns(3)
        result_columns[0].metric("Cases", report["case_count"])
        result_columns[1].metric(
            "Retrieval Recall@K", f"{float(report['retrieval_recall_at_k']):.1%}"
        )
        result_columns[2].metric(
            "Status accuracy", f"{float(report['status_accuracy']):.1%}"
        )
        st.dataframe(report["cases"], width="stretch", hide_index=True)


st.markdown('<div class="v3-kicker">Evidence-grounded research platform · V3</div>', unsafe_allow_html=True)
st.title(APP_TITLE)
st.markdown(
    '<p class="v3-subtitle">Search, synthesize, and verify public conservation evidence.</p>',
    unsafe_allow_html=True,
)
st.warning(RESEARCH_DISCLAIMER, icon="⚠️")

default_model = SETTINGS.models.llm_model
default_index = (
    CHAT_MODEL_OPTIONS.index(default_model)
    if default_model in CHAT_MODEL_OPTIONS
    else 0
)
model_column, model_note_column = st.columns([1, 2])
selected_model = model_column.selectbox(
    "Generation model",
    CHAT_MODEL_OPTIONS,
    index=default_index,
    key="v3_global_model",
    help="Used globally by Chatbot, Wiki, and generation-based Evaluation runs.",
)
model_note_column.info(
    "This selection applies to Chatbot, Wiki, and Evaluation. Search uses the "
    "configured embedding model or keyword matching and does not call a GPT model."
)

store = get_store()
corpus_tab, search_tab, wiki_tab, chatbot_tab, evaluation_tab = st.tabs(
    ["Corpus", "Search", "Wiki", "Chatbot", "Evaluation"]
)
with corpus_tab:
    render_corpus_tab(store)
with search_tab:
    render_search_tab(store)
with wiki_tab:
    render_wiki_tab(store, selected_model)
with chatbot_tab:
    render_chatbot_tab(store, selected_model)
with evaluation_tab:
    render_evaluation_tab(store, selected_model)
