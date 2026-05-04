from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from info_agent.llm import LLMClient, build_queries, create_llm_from_env, extract_keywords, summarize


@dataclass(frozen=True)
class ResearchResult:
    source: str
    channel: str
    title: str
    url: str
    summary: str
    confidence: float
    raw_content: str = ""
    full_summary: str = ""
    result_origin: str = "search_api"
    summary_origin: str = "llm"
    summary_provider: str = ""


class TTLCache:
    def __init__(self, ttl_seconds: int = 600) -> None:
        self.ttl_seconds = ttl_seconds
        self._items: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        item = self._items.get(key)
        if item is None:
            return None
        created_at, value = item
        if time.time() - created_at > self.ttl_seconds:
            self._items.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._items[key] = (time.time(), value)


def extract_context(text: str, max_chars: int = 1200) -> str:
    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    if not paragraphs:
        return text.strip()[-max_chars:]
    return "\n".join(paragraphs[-3:])[-max_chars:]


class WikipediaSearch:
    endpoint = "https://en.wikipedia.org/w/api.php"

    def search(self, query: str, limit: int = 3, timeout: float = 2.5) -> list[ResearchResult]:
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": str(limit),
                "prop": "extracts|info",
                "exintro": "1",
                "explaintext": "1",
                "inprop": "url",
                "origin": "*",
            }
        )
        request = urllib.request.Request(
            f"{self.endpoint}?{params}",
            headers={"User-Agent": "info-agent-mvp/0.1"},
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return []

        pages = payload.get("query", {}).get("pages", {})
        results: list[ResearchResult] = []
        for page in pages.values():
            title = page.get("title")
            url = page.get("fullurl")
            extract = page.get("extract", "")
            if not title or not url:
                continue
            results.append(
                ResearchResult(
                    source="wikipedia",
                    channel="Wikipedia",
                    title=title,
                    url=url,
                    summary=summarize(extract or f"Wikipedia article related to {query}."),
                    confidence=0.78,
                    raw_content=extract,
                    full_summary=extract or f"Wikipedia article related to {query}.",
                    result_origin="search_api",
                    summary_origin="fallback",
                    summary_provider="deterministic",
                )
            )
        return results


class TavilySearch:
    endpoint = "https://api.tavily.com/search"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or tavily_api_key_from_env()

    def available(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, limit: int = 3, timeout: float = 5.0) -> list[ResearchResult]:
        if not self.api_key:
            return []

        body = {
            "query": query,
            "search_depth": os.getenv("TAVILY_SEARCH_DEPTH", "basic"),
            "max_results": limit,
            "include_answer": False,
            "include_raw_content": False,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return []

        results: list[ResearchResult] = []
        for item in payload.get("results", []):
            title = item.get("title")
            url = item.get("url")
            content = item.get("content", "")
            if not title or not url:
                continue
            score = item.get("score")
            confidence = float(score) if isinstance(score, int | float) else 0.7
            results.append(
                ResearchResult(
                    source="web",
                    channel="Web",
                    title=title,
                    url=url,
                    summary=summarize(content or f"Tavily search result related to {query}."),
                    confidence=max(0.0, min(confidence, 1.0)),
                    raw_content=content,
                    full_summary=content or f"Tavily search result related to {query}.",
                    result_origin="search_api",
                    summary_origin="fallback",
                    summary_provider="deterministic",
                )
            )
        return results


def search_link_result(source: str, title: str, base_url: str, query: str) -> ResearchResult:
    encoded = urllib.parse.quote_plus(query)
    return ResearchResult(
        source=source,
        channel=channel_for_source(source),
        title=title,
        url=base_url.format(query=encoded),
        summary=f"Search results for '{query}'. Open the link to review current source material.",
        confidence=0.45,
        raw_content=f"{title}\nSearch query: {query}",
        full_summary=f"Search results for '{query}'. Open the link to review current source material.",
        result_origin="fallback_link",
        summary_origin="fallback",
        summary_provider="deterministic",
    )


class ResearchPipeline:
    def __init__(self, cache: TTLCache | None = None, llm: LLMClient | None = None) -> None:
        self.cache = cache or TTLCache()
        self.llm = llm or create_llm_from_env()
        self.wikipedia = WikipediaSearch()
        self.tavily = TavilySearch()

    def research(self, text: str) -> dict[str, Any]:
        context = extract_context(text)
        cache_key = re.sub(r"\s+", " ", context).strip().lower()
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        plan = self.llm.create_research_plan(context)
        keywords = plan.keywords
        queries = plan.queries or build_queries(keywords, context)
        results: list[ResearchResult] = []

        wiki_queries = list(queries[:1])
        if any(keyword.lower() == "rag" for keyword in keywords):
            wiki_queries.append("Retrieval augmented generation")

        for query in wiki_queries:
            results.extend(self.wikipedia.search(query))

        primary_query = queries[0] if queries else context
        if primary_query:
            web_results = self.tavily.search(primary_query)
            if web_results:
                results.extend(web_results)
            else:
                results.append(
                    search_link_result(
                        "web",
                        f"Web search: {primary_query}",
                        "https://duckduckgo.com/?q={query}",
                        primary_query,
                    )
                )
            results.append(
                search_link_result(
                    "youtube",
                    f"YouTube search: {primary_query}",
                    "https://www.youtube.com/results?search_query={query}",
                    primary_query,
                )
            )

        summarized_results = self.summarize_results(context, dedupe_results(results)[:8])

        payload = {
            "keywords": keywords,
            "queries": queries,
            "llm_provider": plan.provider,
            "keyword_origin": response_origin(plan.provider),
            "search_origin": "search_api" if any(result.result_origin == "search_api" for result in results) else "fallback_link",
            "summary_origin": combined_summary_origin(summarized_results),
            "pipeline": ["text", "llm_keyword_query_extraction", "search_api", "llm_summary"],
            "results": [public_result(result) for result in summarized_results],
        }
        self.cache.set(cache_key, payload)
        return payload

    def summarize_results(self, context: str, results: list[ResearchResult]) -> list[ResearchResult]:
        summarized: list[ResearchResult] = []
        for result in results:
            source_text = result.raw_content or result.summary
            if result.source in {"wikipedia", "web", "youtube"}:
                summary = self.llm.summarize_result(
                    user_text=context,
                    title=result.title,
                    source_text=source_text,
                )
                summary_provider = getattr(self.llm, "last_summary_provider", getattr(self.llm, "provider", "unknown"))
            else:
                summary = result.full_summary or result.summary
                summary_provider = result.summary_provider or "deterministic"
            summarized.append(
                ResearchResult(
                    source=result.source,
                    channel=result.channel,
                    title=result.title,
                    url=result.url,
                    summary=summarize(summary),
                    confidence=result.confidence,
                    raw_content=result.raw_content,
                    full_summary=summary,
                    result_origin=result.result_origin,
                    summary_origin=response_origin(summary_provider),
                    summary_provider=summary_provider,
                )
            )
        return summarized


def dedupe_results(results: list[ResearchResult]) -> list[ResearchResult]:
    seen: set[str] = set()
    deduped: list[ResearchResult] = []
    for result in sorted(results, key=lambda item: item.confidence, reverse=True):
        key = result.url.lower().rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return deduped


def public_result(result: ResearchResult) -> dict[str, Any]:
    return {
        "source": result.source,
        "channel": result.channel,
        "provider_label": provider_label(result),
        "title": result.title,
        "url": result.url,
        "summary": result.summary,
        "full_summary": result.full_summary or result.summary,
        "is_truncated": (result.full_summary or result.summary) != result.summary,
        "confidence": result.confidence,
        "result_origin": result.result_origin,
        "summary_origin": result.summary_origin,
        "summary_provider": result.summary_provider,
    }


def response_origin(provider: str) -> str:
    if provider in {"", "deterministic"} or provider.endswith(":fallback"):
        return "fallback"
    return "llm"


def combined_summary_origin(results: list[ResearchResult]) -> str:
    origins = {result.summary_origin for result in results}
    if not origins:
        return "none"
    if len(origins) == 1:
        return origins.pop()
    return "mixed"


def tavily_api_key_from_env() -> str | None:
    return os.getenv("TAVILY_API_KEY")


def channel_for_source(source: str) -> str:
    return {
        "web": "Web",
        "wikipedia": "Wikipedia",
        "youtube": "YouTube",
    }.get(source, source.title())


def provider_label(result: ResearchResult) -> str:
    if result.source == "web" and result.result_origin == "search_api":
        return "Tavily API"
    if result.source == "wikipedia" and result.result_origin == "search_api":
        return "Wikipedia API"
    if result.source == "youtube" and result.result_origin == "fallback_link":
        return "YouTube fallback link"
    if result.result_origin == "fallback_link":
        return "Fallback link"
    return "Search API"
