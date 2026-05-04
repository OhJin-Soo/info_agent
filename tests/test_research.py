from info_agent.llm import DeterministicLLM, ResearchPlan, ResilientLLM, build_queries, extract_keywords
from info_agent.research import ResearchPipeline, ResearchResult, TavilySearch, dedupe_results


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


def test_dedupe_results_keeps_highest_confidence_first() -> None:
    results = dedupe_results(
        [
            ResearchResult("web", "low", "https://example.com", "...", 0.3),
            ResearchResult("web", "high", "https://example.com", "...", 0.9),
            ResearchResult("wiki", "other", "https://example.com/2", "...", 0.8),
        ]
    )

    assert [result.title for result in results] == ["high", "other"]


class FakeLLM:
    provider = "fake"

    def create_research_plan(self, text: str) -> ResearchPlan:
        assert "RAG" in text
        return ResearchPlan(
            keywords=["RAG", "vector search"],
            queries=["Retrieval augmented generation"],
            provider=self.provider,
        )

    def summarize_result(self, *, user_text: str, title: str, source_text: str) -> str:
        assert user_text
        return f"summary for {title}: {source_text[:20]}"


def test_pipeline_uses_llm_for_plan_and_summary() -> None:
    pipeline = ResearchPipeline(llm=FakeLLM())
    pipeline.wikipedia.search = lambda query: [
        ResearchResult(
            source="wikipedia",
            title="Retrieval-augmented generation",
            url="https://example.com/rag",
            summary="raw",
            confidence=0.8,
            raw_content="RAG retrieves external knowledge before generation.",
        )
    ]

    payload = pipeline.research("RAG는 벡터 검색을 활용한다.")

    assert payload["keywords"] == ["RAG", "vector search"]
    assert payload["queries"] == ["Retrieval augmented generation"]
    assert payload["llm_provider"] == "fake"
    assert payload["keyword_origin"] == "llm"
    assert payload["search_origin"] == "search_api"
    assert payload["summary_origin"] == "llm"
    assert payload["pipeline"] == ["text", "llm_keyword_query_extraction", "search_api", "llm_summary"]
    assert payload["results"][0]["summary"].startswith("summary for Retrieval-augmented")
    assert payload["results"][0]["result_origin"] == "search_api"
    assert payload["results"][0]["summary_origin"] == "llm"
    assert payload["results"][0]["summary_provider"] == "fake"
    assert "raw_content" not in payload["results"][0]


def test_deterministic_llm_matches_fallback_contract() -> None:
    plan = DeterministicLLM().create_research_plan("RAG embedding vector search")

    assert plan.provider == "deterministic"
    assert "RAG" in plan.keywords
    assert plan.queries


class BrokenLLM:
    provider = "broken"

    def __init__(self) -> None:
        self.calls = 0

    def create_research_plan(self, text: str) -> ResearchPlan:
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


def test_tavily_search_uses_tavily_api_key_env(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")

    search = TavilySearch()

    assert search.api_key == "tvly-test"
