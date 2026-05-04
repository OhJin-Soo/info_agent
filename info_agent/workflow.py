from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from langsmith import traceable
from langgraph.graph import END, StateGraph

from info_agent.llm import build_queries, validate_keywords
from info_agent.nlp import extract_noun_candidates
from info_agent.research import (
    ResearchResult,
    dedupe_results,
    group_public_results_by_keyword_and_source,
    mark_keyword,
    public_result,
    response_origin,
    search_keywords_for,
    search_link_result,
    select_keyword_channel_results,
)

if TYPE_CHECKING:
    from info_agent.research import ResearchPipeline


class ResearchState(TypedDict, total=False):
    context: str
    noun_candidates: list[str]
    keywords: list[str]
    queries: list[str]
    search_keywords: list[str]
    search_query: str
    raw_results: list[ResearchResult]
    selected_results: list[ResearchResult]
    summarized_results: list[ResearchResult]
    public_results: list[dict[str, Any]]
    llm_provider: str
    payload: dict[str, Any]


@traceable(run_type="chain", name="Research Workflow")
def run_research_workflow(pipeline: ResearchPipeline, context: str) -> dict[str, Any]:
    graph = build_research_graph(pipeline)
    state = graph.invoke({"context": context})
    return state["payload"]


def build_research_graph(pipeline: ResearchPipeline):
    graph = StateGraph(ResearchState)
    graph.add_node("extract_nouns", lambda state: extract_nouns_node(state))
    graph.add_node("select_terms", lambda state: select_terms_node(pipeline, state))
    graph.add_node("search", lambda state: search_node(pipeline, state))
    graph.add_node("summarize", lambda state: summarize_node(pipeline, state))
    graph.add_node("format_payload", lambda state: format_payload_node(state))

    graph.set_entry_point("extract_nouns")
    graph.add_edge("extract_nouns", "select_terms")
    graph.add_edge("select_terms", "search")
    graph.add_edge("search", "summarize")
    graph.add_edge("summarize", "format_payload")
    graph.add_edge("format_payload", END)
    return graph.compile()


@traceable(run_type="tool", name="Extract Nouns")
def extract_nouns_node(state: ResearchState) -> ResearchState:
    return {"noun_candidates": extract_noun_candidates(state["context"])}


@traceable(run_type="llm", name="Select Terms")
def select_terms_node(pipeline: ResearchPipeline, state: ResearchState) -> ResearchState:
    context = state["context"]
    noun_candidates = state.get("noun_candidates", [])
    plan = pipeline.llm.create_research_plan(context, noun_candidates)
    keywords = validate_keywords(plan.keywords)
    queries = plan.queries or build_queries(keywords, context)
    search_keywords = search_keywords_for(keywords, context)
    return {
        "keywords": keywords,
        "queries": queries,
        "search_keywords": search_keywords,
        "search_query": " ".join(search_keywords),
        "llm_provider": plan.provider,
    }


@traceable(run_type="tool", name="Search Sources")
def search_node(pipeline: ResearchPipeline, state: ResearchState) -> ResearchState:
    results: list[ResearchResult] = []
    for keyword in state.get("search_keywords", []):
        results.extend(mark_keyword(pipeline.wikipedia.search(keyword, limit=2), keyword))

        web_results = mark_keyword(pipeline.tavily.search(keyword, limit=2), keyword)
        if web_results:
            results.extend(web_results)
        else:
            results.append(
                search_link_result(
                    "web",
                    f"Web search: {keyword}",
                    "https://duckduckgo.com/?q={query}",
                    keyword,
                )
            )
        results.append(
            search_link_result(
                "youtube",
                f"YouTube search: {keyword}",
                "https://www.youtube.com/results?search_query={query}",
                keyword,
            )
        )

    deduped = dedupe_results(results)
    selected = select_keyword_channel_results(deduped, state.get("search_keywords", []))
    return {"raw_results": results, "selected_results": selected}


@traceable(run_type="llm", name="Summarize Sources")
def summarize_node(pipeline: ResearchPipeline, state: ResearchState) -> ResearchState:
    summarized = pipeline.summarize_results(state["context"], state.get("selected_results", []))
    public_results = [public_result(result) for result in summarized]
    return {"summarized_results": summarized, "public_results": public_results}


@traceable(run_type="chain", name="Format Payload")
def format_payload_node(state: ResearchState) -> ResearchState:
    raw_results = state.get("raw_results", [])
    summarized = state.get("summarized_results", [])
    public_results = state.get("public_results", [])
    search_keywords = state.get("search_keywords", [])
    payload = {
        "keywords": state.get("keywords", []),
        "noun_candidates": state.get("noun_candidates", []),
        "queries": state.get("queries", []),
        "search_keywords": search_keywords,
        "search_query": state.get("search_query", ""),
        "llm_provider": state.get("llm_provider", "unknown"),
        "keyword_origin": response_origin(state.get("llm_provider", "")),
        "search_origin": "search_api" if any(result.result_origin == "search_api" for result in raw_results) else "fallback_link",
        "summary_origin": pipeline_summary_origin(summarized),
        "pipeline": ["text", "spacy_noun_extraction", "llm_term_selection", "search_api", "llm_summary"],
        "workflow_engine": "langgraph",
        "results": public_results,
        "keyword_results": group_public_results_by_keyword_and_source(public_results, search_keywords),
    }
    return {"payload": payload}


def pipeline_summary_origin(results: list[ResearchResult]) -> str:
    origins = {result.summary_origin for result in results}
    if not origins:
        return "none"
    if len(origins) == 1:
        return origins.pop()
    return "mixed"
