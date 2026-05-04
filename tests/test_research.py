from info_agent.research import build_queries, dedupe_results, extract_keywords, ResearchResult


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
