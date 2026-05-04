from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Protocol

from langsmith import traceable


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
    "그리고",
    "또는",
    "관련",
    "대한",
    "검색",
    "외부",
    "지식",
    "활용",
    "활용하여",
    "한다",
    "한다는",
}

KOREAN_SUFFIXES = ("으로", "에서", "에게", "에는", "을", "를", "은", "는", "이", "가", "와", "과", "도")
KEYWORD_MAX_CHARS = 60
KEYWORD_MAX_TERMS = 5


@dataclass(frozen=True)
class ResearchPlan:
    keywords: list[str]
    queries: list[str]
    provider: str


class LLMClient(Protocol):
    provider: str

    def create_research_plan(self, text: str, noun_candidates: list[str] | None = None) -> ResearchPlan:
        raise NotImplementedError

    def summarize_result(self, *, user_text: str, title: str, source_text: str) -> str:
        raise NotImplementedError


class DeterministicLLM:
    provider = "deterministic"

    @traceable(run_type="chain", name="Deterministic Research Plan")
    def create_research_plan(self, text: str, noun_candidates: list[str] | None = None) -> ResearchPlan:
        keywords = validate_keywords(noun_candidates or extract_keywords(text))
        return ResearchPlan(
            keywords=keywords,
            queries=build_queries(keywords, text),
            provider=self.provider,
        )

    def summarize_result(self, *, user_text: str, title: str, source_text: str) -> str:
        del user_text, title
        return summarize(source_text)


class OpenAIResponsesLLM:
    provider = "openai"
    endpoint = "https://api.openai.com/v1/responses"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5-mini",
        timeout: float = 30.0,
        reasoning_effort: str = "minimal",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort

    @traceable(run_type="llm", name="OpenAI Research Plan")
    def create_research_plan(self, text: str, noun_candidates: list[str] | None = None) -> ResearchPlan:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 6,
                },
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 4,
                },
            },
            "required": ["keywords", "queries"],
        }
        candidates = validate_keywords(noun_candidates or [])
        prompt = research_plan_prompt(text, candidates)
        payload = self._request_json(
            instructions=(
                "You are a research planning assistant. Return only schema-valid JSON. "
                "The keywords array is a controlled vocabulary of searchable domain terms, not a list of common words."
            ),
            input_text=prompt,
            schema_name="research_plan",
            schema=schema,
        )
        keywords = validate_candidate_keywords(clean_string_list(payload.get("keywords"), limit=12), candidates)
        queries = clean_string_list(payload.get("queries"), limit=4)
        if not queries:
            queries = build_queries(keywords, text)
        return ResearchPlan(keywords=keywords or candidates or validate_keywords(extract_keywords(text)), queries=queries, provider=self.provider)

    @traceable(run_type="llm", name="OpenAI Summary")
    def summarize_result(self, *, user_text: str, title: str, source_text: str) -> str:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        }
        prompt = summary_prompt(user_text, title, source_text)
        payload = self._request_json(
            instructions="You summarize retrieved research sources for a text editor.",
            input_text=prompt,
            schema_name="source_summary",
            schema=schema,
        )
        return re.sub(r"\s+", " ", str(payload.get("summary", "")) or source_text).strip()

    def _request_json(self, *, instructions: str, input_text: str, schema_name: str, schema: dict[str, Any]) -> dict[str, Any]:
        body = {
            "model": self.model,
            "instructions": instructions,
            "input": input_text,
            "reasoning": {"effort": self.reasoning_effort},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
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
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        text = extract_response_text(payload)
        return json.loads(text)


class OllamaLLM:
    provider = "ollama"

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434", timeout: float = 8.0) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @traceable(run_type="llm", name="Ollama Research Plan")
    def create_research_plan(self, text: str, noun_candidates: list[str] | None = None) -> ResearchPlan:
        candidates = validate_keywords(noun_candidates or [])
        prompt = (
            "Return JSON with keys keywords and queries. keywords is up to 6 strings and must contain "
            "domain-specific terms only: named products, named concepts, acronyms, standards, technical terms, "
            "or multi-word terms. Exclude ordinary verbs and generic nouns such as remembers, preferences, "
            "insights, creates, system, tool, workflow, data, or solution unless part of a named technical term. "
            "Choose keywords only from the noun_candidates list. "
            "queries is up to 4 search queries. No markdown.\n\n"
            f"noun_candidates:\n{json.dumps(candidates, ensure_ascii=False)}\n\nDraft:\n{text}"
        )
        payload = parse_json_object(self._generate(prompt))
        keywords = validate_candidate_keywords(clean_string_list(payload.get("keywords"), limit=12), candidates)
        queries = clean_string_list(payload.get("queries"), limit=4)
        return ResearchPlan(
            keywords=keywords or candidates or validate_keywords(extract_keywords(text)),
            queries=queries or build_queries(keywords, text),
            provider=self.provider,
        )

    @traceable(run_type="llm", name="Ollama Summary")
    def summarize_result(self, *, user_text: str, title: str, source_text: str) -> str:
        prompt = summary_prompt(user_text, title, source_text)
        return summarize(self._generate(prompt), max_chars=420)

    def _generate(self, prompt: str) -> str:
        body = {"model": self.model, "prompt": prompt, "stream": False}
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return str(payload.get("response", "")).strip()


class ResilientLLM:
    def __init__(
        self,
        primary: LLMClient | None,
        fallback: LLMClient | None = None,
        cache_ttl_seconds: int | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback or DeterministicLLM()
        self.cache_ttl_seconds = cache_ttl_seconds or int(os.getenv("LLM_CACHE_TTL_SECONDS", "3600"))
        self._plan_cache: dict[str, tuple[float, ResearchPlan]] = {}
        self._summary_cache: dict[str, tuple[float, str, str]] = {}
        self._primary_failed = False
        self.provider = primary.provider if primary else self.fallback.provider
        self.last_plan_provider = self.fallback.provider if primary is None else primary.provider
        self.last_summary_provider = self.last_plan_provider

    @traceable(run_type="chain", name="Resilient Research Plan")
    def create_research_plan(self, text: str, noun_candidates: list[str] | None = None) -> ResearchPlan:
        candidates = validate_keywords(noun_candidates or [])
        cache_key = make_cache_key("plan", text, json.dumps(candidates, ensure_ascii=False))
        cached = self._get_plan_cache(cache_key)
        if cached is not None:
            self.last_plan_provider = cached.provider
            return cached

        if self.primary is None or self._primary_failed:
            plan = self.fallback.create_research_plan(text, candidates)
            self.last_plan_provider = self.fallback.provider if self.primary is None else f"{self.primary.provider}:fallback"
            cached_plan = ResearchPlan(plan.keywords, plan.queries, self.last_plan_provider)
            self._set_plan_cache(cache_key, cached_plan)
            return cached_plan
        try:
            plan = self.primary.create_research_plan(text, candidates)
            self.last_plan_provider = self.primary.provider
            self._set_plan_cache(cache_key, plan)
            return plan
        except (urllib.error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError, ValueError):
            self._primary_failed = True
            plan = self.fallback.create_research_plan(text, candidates)
            self.last_plan_provider = f"{self.primary.provider}:fallback"
            cached_plan = ResearchPlan(plan.keywords, plan.queries, self.last_plan_provider)
            self._set_plan_cache(cache_key, cached_plan)
            return cached_plan

    @traceable(run_type="chain", name="Resilient Summary")
    def summarize_result(self, *, user_text: str, title: str, source_text: str) -> str:
        cache_key = make_cache_key("summary", user_text[:1200], title, source_text[:2400])
        cached = self._get_summary_cache(cache_key)
        if cached is not None:
            summary, provider = cached
            self.last_summary_provider = provider
            return summary

        if self.primary is None or self._primary_failed:
            self.last_summary_provider = self.fallback.provider if self.primary is None else f"{self.primary.provider}:fallback"
            summary = self.fallback.summarize_result(user_text=user_text, title=title, source_text=source_text)
            self._set_summary_cache(cache_key, summary, self.last_summary_provider)
            return summary
        try:
            summary = self.primary.summarize_result(user_text=user_text, title=title, source_text=source_text)
            self.last_summary_provider = self.primary.provider
            self._set_summary_cache(cache_key, summary, self.last_summary_provider)
            return summary
        except (urllib.error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError, ValueError):
            self._primary_failed = True
            self.last_summary_provider = f"{self.primary.provider}:fallback"
            summary = self.fallback.summarize_result(user_text=user_text, title=title, source_text=source_text)
            self._set_summary_cache(cache_key, summary, self.last_summary_provider)
            return summary

    def _get_plan_cache(self, key: str) -> ResearchPlan | None:
        item = self._plan_cache.get(key)
        if item is None:
            return None
        created_at, plan = item
        if time.time() - created_at > self.cache_ttl_seconds:
            self._plan_cache.pop(key, None)
            return None
        return plan

    def _set_plan_cache(self, key: str, plan: ResearchPlan) -> None:
        self._plan_cache[key] = (time.time(), plan)

    def _get_summary_cache(self, key: str) -> tuple[str, str] | None:
        item = self._summary_cache.get(key)
        if item is None:
            return None
        created_at, summary, provider = item
        if time.time() - created_at > self.cache_ttl_seconds:
            self._summary_cache.pop(key, None)
            return None
        return summary, provider

    def _set_summary_cache(self, key: str, summary: str, provider: str) -> None:
        self._summary_cache[key] = (time.time(), summary, provider)


def create_llm_from_env() -> LLMClient:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider == "ollama" or (not provider and os.getenv("OLLAMA_MODEL")):
        return ResilientLLM(
            OllamaLLM(
                model=os.getenv("OLLAMA_MODEL", "llama3.1"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "8")),
            )
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if provider == "openai" or api_key:
        if not api_key:
            return ResilientLLM(None)
        return ResilientLLM(
            OpenAIResponsesLLM(
                api_key=api_key,
                model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
                timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
                reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", "minimal"),
            )
        )

    return ResilientLLM(None)


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
    return validate_keywords(ranked, limit=limit)


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


def clean_string_list(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = re.sub(r"\s+", " ", str(item)).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned[:limit]


def validate_keywords(keywords: list[str], limit: int = 6) -> list[str]:
    valid: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        normalized = normalize_keyword(keyword)
        if not is_valid_keyword(normalized):
            continue
        dedupe_key = keyword_dedupe_key(normalized)
        if dedupe_key not in seen:
            valid.append(normalized)
            seen.add(dedupe_key)
        if len(valid) >= limit:
            break
    return valid


def normalize_keyword(keyword: str) -> str:
    normalized = re.sub(r"\s+", " ", str(keyword)).strip().strip("\"'`.,;:!?()[]{}")
    parts = [normalize_token(part) if re.fullmatch(r"[가-힣]+", part) else part for part in normalized.split()]
    return " ".join(parts)


def is_valid_keyword(keyword: str) -> bool:
    if len(keyword) < 2 or len(keyword) > KEYWORD_MAX_CHARS:
        return False
    if re.search(r"[,;:!?\"'`“”‘’()\[\]{}<>|/\\]", keyword):
        return False
    parts = keyword.split()
    if len(parts) > KEYWORD_MAX_TERMS:
        return False
    if not all(re.fullmatch(r"[A-Za-z0-9가-힣.+#-]+", part) for part in parts):
        return False
    lowered = keyword.lower()
    if lowered in STOPWORDS:
        return False
    if re.search(r"(어떻게|방법|설명|비교|사례|튜토리얼|what|how|why|compare|tutorial|explained)", lowered):
        return False
    return True


def keyword_dedupe_key(keyword: str) -> str:
    return re.sub(r"\s+", " ", keyword).strip().casefold()


def extract_response_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    for item in payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                return str(content.get("text", ""))
    raise ValueError("OpenAI response did not include output text")


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            return {}
        payload = json.loads(match.group(0))
    return payload if isinstance(payload, dict) else {}


def make_cache_key(*parts: str) -> str:
    normalized = "\n\n".join(normalize_cache_text(part) for part in parts)
    return sha256(normalized.encode("utf-8")).hexdigest()


def normalize_cache_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def research_plan_prompt(text: str, noun_candidates: list[str] | None = None) -> str:
    candidates = validate_keywords(noun_candidates or [])
    return (
        "Select concise research keywords from spaCy-extracted noun candidates, then create search queries. "
        "Keywords must be domain-specific terms only, not ordinary verbs, adjectives, or generic nouns. "
        "Prefer named products, named concepts, acronyms, standards, technical terms, and multi-word terms. "
        "Do not include generic words like remembers, preferences, insights, creates, system, tool, workflow, data, or solution unless they are part of a named technical term. "
        "Do not include full sentences or search-style questions as keywords. "
        "Choose keywords only from noun_candidates; do not invent new keywords outside that list. "
        "Return keywords in the draft language when useful, but make at least one query strong for English web/Wikipedia search.\n\n"
        f"noun_candidates:\n{json.dumps(candidates, ensure_ascii=False)}\n\n"
        f"Draft:\n{text}"
    )


def summary_prompt(user_text: str, title: str, source_text: str) -> str:
    return (
        "Summarize the source for the user's draft in Korean only. "
        "The output must be entirely in Korean; do not include English sentences except for unavoidable proper nouns or product names. "
        "Write 2 concise sentences. Use only the source text. No markdown. "
        "Do not translate the prompt itself, only output the summary in Korean.\n\n"
        f"Draft:\n{user_text[:1200]}\n\n"
        f"Title:\n{title}\n\n"
        f"Source:\n{source_text[:2400]}"
    )


def validate_candidate_keywords(keywords: list[str], candidates: list[str], limit: int = 6) -> list[str]:
    valid_keywords = validate_keywords(keywords, limit=limit)
    if not candidates:
        return valid_keywords
    candidate_map = {keyword_dedupe_key(candidate): candidate for candidate in validate_keywords(candidates, limit=100)}
    selected: list[str] = []
    for keyword in valid_keywords:
        candidate = candidate_map.get(keyword_dedupe_key(keyword))
        if candidate and candidate not in selected:
            selected.append(candidate)
    return selected[:limit]
