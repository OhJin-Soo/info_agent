from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "because",
    "between",
    "but",
    "can",
    "for",
    "from",
    "has",
    "have",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "use",
    "uses",
    "using",
    "with",
    "사용",
    "활용",
    "그리고",
    "또는",
    "관련",
    "대한",
    "검색",
    "사용",
    "외부",
    "지식",
    "활용",
    "활용하여",
    "한다",
    "한다는",
}

KOREAN_SUFFIXES = ("으로", "에서", "에게", "에는", "을", "를", "은", "는", "이", "가", "와", "과", "도")


@dataclass(frozen=True)
class ResearchResult:
    source: str
    title: str
    url: str
    summary: str
    confidence: float


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


def extract_keywords(text: str, limit: int = 6) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+-]{1,}|[가-힣]{2,}", text)
    scores: dict[str, int] = {}
    first_seen: dict[str, int] = {}

    for index, token in enumerate(tokens):
        normalized = normalize_token(token)
        key = normalized.lower()
        if key in STOPWORDS or len(key) < 2:
            continue

        score = 1
        if normalized.isupper() and len(normalized) > 1:
            score += 3
        if any(char.isdigit() for char in normalized):
            score += 1
        if len(normalized) >= 7:
            score += 1

        scores[normalized] = scores.get(normalized, 0) + score
        first_seen.setdefault(normalized, index)

    ranked = sorted(scores, key=lambda item: (-scores[item], first_seen[item], item.lower()))
    return ranked[:limit]


def normalize_token(token: str) -> str:
    normalized = token.strip()
    if re.fullmatch(r"[가-힣]+", normalized):
        for suffix in KOREAN_SUFFIXES:
            if normalized.endswith(suffix) and len(normalized) > len(suffix) + 1:
                return normalized[: -len(suffix)]
    return normalized


def build_queries(keywords: list[str], text: str, limit: int = 4) -> list[str]:
    if not keywords:
        fallback = " ".join(re.findall(r"\S+", text)[:8])
        return [fallback] if fallback else []

    primary = " ".join(keywords[:3])
    queries = [
        primary,
        f"{primary} explained",
        f"{primary} tutorial",
    ]
    if len(keywords) >= 2:
        queries.append(f"{keywords[0]} {keywords[1]} case study")

    deduped: list[str] = []
    for query in queries:
        if query and query not in deduped:
            deduped.append(query)
    return deduped[:limit]


def summarize(text: str, max_chars: int = 260) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= max_chars:
        return clean
    trimmed = clean[: max_chars - 1].rsplit(" ", 1)[0]
    return f"{trimmed}..."


class WikipediaSearch:
    endpoint = "https://en.wikipedia.org/w/api.php"

    def search(self, query: str, limit: int = 3, timeout: float = 4.0) -> list[ResearchResult]:
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
                    title=title,
                    url=url,
                    summary=summarize(extract or f"Wikipedia article related to {query}."),
                    confidence=0.78,
                )
            )
        return results


def search_link_result(source: str, title: str, base_url: str, query: str) -> ResearchResult:
    encoded = urllib.parse.quote_plus(query)
    return ResearchResult(
        source=source,
        title=title,
        url=base_url.format(query=encoded),
        summary=f"Search results for '{query}'. Open the link to review current source material.",
        confidence=0.45,
    )


class ResearchPipeline:
    def __init__(self, cache: TTLCache | None = None) -> None:
        self.cache = cache or TTLCache()
        self.wikipedia = WikipediaSearch()

    def research(self, text: str) -> dict[str, Any]:
        context = extract_context(text)
        cache_key = re.sub(r"\s+", " ", context).strip().lower()
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        keywords = extract_keywords(context)
        queries = build_queries(keywords, context)
        results: list[ResearchResult] = []

        wiki_queries = list(queries[:2])
        if any(keyword.lower() == "rag" for keyword in keywords):
            wiki_queries.append("Retrieval augmented generation")

        for query in wiki_queries:
            results.extend(self.wikipedia.search(query))

        primary_query = queries[0] if queries else context
        if primary_query:
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

        payload = {
            "keywords": keywords,
            "queries": queries,
            "results": [result.__dict__ for result in dedupe_results(results)[:8]],
        }
        self.cache.set(cache_key, payload)
        return payload


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
