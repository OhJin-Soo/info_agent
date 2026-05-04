from info_agent.llm import (
    DeterministicLLM,
    ResearchPlan,
    ResilientLLM,
    build_queries,
    extract_keywords,
    research_plan_prompt,
    summary_prompt,
    validate_candidate_keywords,
    validate_keywords,
)
from info_agent.nlp import extract_noun_candidates
from info_agent.research import (
    ResearchPipeline,
    ResearchResult,
    TavilySearch,
    dedupe_results,
    keyword_search_query,
    select_balanced_results,
    select_keyword_channel_results,
    select_keyword_covered_results,
)


def test_extract_keywords_prioritizes_acronyms_and_terms() -> None:
    keywords = extract_keywords("RAG는 벡터 검색을 활용하여 external knowledge와 embedding을 사용한다.")

    assert "RAG" in keywords
    assert "embedding" in keywords


def test_build_queries_returns_deterministic_variants() -> None:
    queries = build_queries(["RAG", "embedding", "vector"], "")

    assert queries == [
        "RAG embedding vector",
        "RAG embedding vector explained",
        "RAG embedding vector tutorial",
        "RAG embedding case study",
    ]


def test_validate_keywords_rejects_sentence_like_items() -> None:
    keywords = validate_keywords(
        [
            "RAG",
            "벡터 검색",
            "RAG 어떻게 벡터 검색을 활용하나",
            "Amazon Bedrock AgentCore Memory, short-term memory",
            "external memory layer for conversational agents storing summaries",
        ]
    )

    assert keywords == ["RAG", "벡터 검색"]


def test_validate_keywords_dedupes_case_only_differences() -> None:
    keywords = validate_keywords(["Cross-Session Memory", "cross-session memory", "RAG"])

    assert keywords == ["Cross-Session Memory", "RAG"]


def test_openai_prompt_constrains_keywords_to_domain_terms() -> None:
    prompt = research_plan_prompt("TDX remembers preferences and insights.", ["TDX", "preferences"])

    assert "domain-specific terms only" in prompt
    assert "not ordinary verbs" in prompt
    assert "remembers, preferences, insights, creates" in prompt
    assert "Choose keywords only from noun_candidates" in prompt
    assert '"TDX"' in prompt


def test_summary_prompt_forces_korean_only_output() -> None:
    prompt = summary_prompt("draft", "Title", "Source text")

    assert "Korean only" in prompt
    assert "The output must be entirely in Korean" in prompt
    assert "Do not translate the prompt itself" in prompt


def test_validate_keywords_keeps_format_safety_not_semantic_filtering() -> None:
    keywords = validate_keywords(["TDX", "Cross-Session", "preferences"])

    assert keywords == ["TDX", "Cross-Session", "preferences"]


def test_validate_candidate_keywords_limits_llm_to_spacy_candidates() -> None:
    keywords = validate_candidate_keywords(
        ["RAG", "invented term", "벡터 검색"],
        ["RAG", "벡터 검색"],
    )

    assert keywords == ["RAG", "벡터 검색"]


def test_extract_noun_candidates_uses_spacy_pipeline() -> None:
    candidates = extract_noun_candidates("RAG uses vector search and embedding retrieval.")

    assert "RAG" in candidates
    assert any(candidate in candidates for candidate in ["vector", "embedding", "retrieval"])


def test_keyword_search_query_uses_keywords_not_queries() -> None:
    assert keyword_search_query(["RAG", "벡터 검색", "외부 지식"], "fallback") == "RAG 벡터 검색 외부 지식"


def test_dedupe_results_keeps_highest_confidence_first() -> None:
    results = dedupe_results(
        [
            ResearchResult("web", "Web", "low", "https://example.com", "...", 0.3),
            ResearchResult("web", "Web", "high", "https://example.com", "...", 0.9),
            ResearchResult("wikipedia", "Wikipedia", "other", "https://example.com/2", "...", 0.8),
        ]
    )

    assert [result.title for result in results] == ["high", "other"]


def test_select_balanced_results_keeps_each_channel_visible() -> None:
    results = [
        ResearchResult("web", "Web", f"web {index}", f"https://example.com/web/{index}", "...", 0.9)
        for index in range(6)
    ] + [
        ResearchResult("wikipedia", "Wikipedia", "wiki", "https://example.com/wiki", "...", 0.78),
        ResearchResult("youtube", "YouTube", "video", "https://example.com/video", "...", 0.45),
    ]

    selected = select_balanced_results(results, limit=5)

    assert [result.source for result in selected].count("web") == 3
    assert any(result.source == "wikipedia" for result in selected)
    assert any(result.source == "youtube" for result in selected)


def test_select_keyword_covered_results_keeps_each_keyword_visible() -> None:
    results = [
        ResearchResult("web", "Web", "a", "https://example.com/a", "...", 0.9, matched_keyword="alpha"),
        ResearchResult("web", "Web", "b", "https://example.com/b", "...", 0.9, matched_keyword="beta"),
        ResearchResult("web", "Web", "c", "https://example.com/c", "...", 0.9, matched_keyword="gamma"),
        ResearchResult("wikipedia", "Wikipedia", "extra", "https://example.com/x", "...", 0.8, matched_keyword="alpha"),
    ]

    selected = select_keyword_covered_results(results, ["alpha", "beta", "gamma"], limit=3)

    assert {result.matched_keyword for result in selected} == {"alpha", "beta", "gamma"}


def test_select_keyword_channel_results_keeps_each_keyword_channel_pair() -> None:
    results = [
        ResearchResult(source, source.title(), f"{keyword}-{source}", f"https://example.com/{keyword}/{source}", "...", 0.8, matched_keyword=keyword)
        for keyword in ("alpha", "beta")
        for source in ("web", "wikipedia", "youtube")
    ]

    selected = select_keyword_channel_results(results, ["alpha", "beta"])

    assert len(selected) == 6
    assert {(result.matched_keyword, result.source) for result in selected} == {
        ("alpha", "web"),
        ("alpha", "wikipedia"),
        ("alpha", "youtube"),
        ("beta", "web"),
        ("beta", "wikipedia"),
        ("beta", "youtube"),
    }


class FakeLLM:
    provider = "fake"

    def create_research_plan(self, text: str, noun_candidates: list[str] | None = None) -> ResearchPlan:
        assert "RAG" in text
        assert noun_candidates is not None
        return ResearchPlan(
            keywords=["RAG", "vector search", "RAG 어떻게 벡터 검색을 활용하나"],
            queries=["Retrieval augmented generation"],
            provider=self.provider,
        )

    def summarize_result(self, *, user_text: str, title: str, source_text: str) -> str:
        assert user_text
        return f"summary for {title}: {source_text[:20]}"


def test_pipeline_uses_llm_for_plan_and_summary() -> None:
    pipeline = ResearchPipeline(llm=FakeLLM())
    pipeline.wikipedia.search = lambda query, limit=2: [
        ResearchResult(
            source="wikipedia",
            channel="Wikipedia",
            title="Retrieval-augmented generation",
            url="https://example.com/rag",
            summary="raw",
            confidence=0.8,
            raw_content="RAG retrieves external knowledge before generation.",
        )
    ]
    pipeline.tavily.search = lambda query, limit=2: []

    payload = pipeline.research("RAG는 벡터 검색을 활용한다.")

    assert payload["keywords"] == ["RAG", "vector search"]
    assert payload["queries"] == ["Retrieval augmented generation"]
    assert payload["search_query"] == "RAG vector search"
    assert payload["search_keywords"] == ["RAG", "vector search"]
    assert payload["llm_provider"] == "fake"
    assert payload["keyword_origin"] == "llm"
    assert payload["search_origin"] == "search_api"
    assert payload["summary_origin"] == "llm"
    assert payload["workflow_engine"] == "langgraph"
    assert payload["pipeline"] == ["text", "spacy_noun_extraction", "llm_term_selection", "search_api", "llm_summary"]
    result = next(item for item in payload["results"] if item["url"] == "https://example.com/rag")
    assert result["summary"].startswith("summary for Retrieval-augmented")
    assert result["full_summary"].startswith("summary for Retrieval-augmented")
    assert result["is_truncated"] is False
    assert result["result_origin"] == "search_api"
    assert result["channel"] == "Wikipedia"
    assert result["provider_label"] == "Wikipedia API"
    assert result["matched_keyword"] == "RAG"
    assert result["summary_origin"] == "llm"
    assert result["summary_provider"] == "fake"
    assert "raw_content" not in result


def test_deterministic_llm_matches_fallback_contract() -> None:
    plan = DeterministicLLM().create_research_plan("RAG embedding vector search")

    assert plan.provider == "deterministic"
    assert "RAG" in plan.keywords
    assert plan.queries


class BrokenLLM:
    provider = "broken"

    def __init__(self) -> None:
        self.calls = 0

    def create_research_plan(self, text: str, noun_candidates: list[str] | None = None) -> ResearchPlan:
        self.calls += 1
        raise TimeoutError("unavailable")

    def summarize_result(self, *, user_text: str, title: str, source_text: str) -> str:
        self.calls += 1
        raise TimeoutError("unavailable")


def test_resilient_llm_disables_primary_after_failure() -> None:
    primary = BrokenLLM()
    llm = ResilientLLM(primary)

    plan = llm.create_research_plan("RAG embedding vector search")
    summary = llm.summarize_result(user_text="RAG", title="T", source_text="source text")

    assert plan.provider == "broken:fallback"
    assert summary == "source text"
    assert primary.calls == 1


class CountingLLM:
    provider = "counting"

    def __init__(self) -> None:
        self.plan_calls = 0
        self.summary_calls = 0

    def create_research_plan(self, text: str, noun_candidates: list[str] | None = None) -> ResearchPlan:
        self.plan_calls += 1
        return ResearchPlan(["RAG"], ["RAG"], self.provider)

    def summarize_result(self, *, user_text: str, title: str, source_text: str) -> str:
        self.summary_calls += 1
        return f"{title}: {source_text}"


def test_resilient_llm_caches_plan_and_summary() -> None:
    primary = CountingLLM()
    llm = ResilientLLM(primary, cache_ttl_seconds=60)

    assert llm.create_research_plan("RAG text").keywords == ["RAG"]
    assert llm.create_research_plan("RAG   text").keywords == ["RAG"]
    assert primary.plan_calls == 1

    kwargs = {"user_text": "draft", "title": "Title", "source_text": "source"}
    assert llm.summarize_result(**kwargs) == "Title: source"
    assert llm.summarize_result(**kwargs) == "Title: source"
    assert primary.summary_calls == 1


def test_tavily_search_uses_tavily_api_key_env(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")

    search = TavilySearch()

    assert search.api_key == "tvly-test"


def test_public_result_preserves_full_summary() -> None:
    pipeline = ResearchPipeline(llm=FakeLLM())
    pipeline.wikipedia.search = lambda query, limit=2: [
        ResearchResult(
            source="wikipedia",
            channel="Wikipedia",
            title="Long result",
            url="https://example.com/long",
            summary="raw",
            confidence=0.8,
            raw_content="x" * 600,
        )
    ]
    pipeline.tavily.search = lambda query, limit=2: []

    payload = pipeline.research("RAG는 벡터 검색을 활용한다.")
    result = next(item for item in payload["results"] if item["url"] == "https://example.com/long")

    assert result["full_summary"].startswith("summary for Long result")
    assert len(result["summary"]) <= len(result["full_summary"])
    assert "full_summary" in result


def test_pipeline_searches_each_keyword_per_channel() -> None:
    pipeline = ResearchPipeline(llm=FakeLLM())
    wiki_calls: list[str] = []
    tavily_calls: list[str] = []

    def fake_wikipedia_search(query: str, limit: int = 2) -> list[ResearchResult]:
        wiki_calls.append(query)
        return []

    def fake_tavily_search(query: str, limit: int = 2) -> list[ResearchResult]:
        tavily_calls.append(query)
        return []

    pipeline.wikipedia.search = fake_wikipedia_search
    pipeline.tavily.search = fake_tavily_search

    payload = pipeline.research("RAG는 벡터 검색을 활용한다.")

    assert payload["search_keywords"] == ["RAG", "vector search"]
    assert wiki_calls == ["RAG", "vector search"]
    assert tavily_calls == ["RAG", "vector search"]
    assert any(result["channel"] == "Web" for result in payload["results"])
    assert any(result["channel"] == "YouTube" for result in payload["results"])
    assert {result["matched_keyword"] for result in payload["results"]} == {"RAG", "vector search"}
    assert all(set(group.keys()) == {"keyword", "web", "wikipedia", "youtube"} for group in payload["keyword_results"])
