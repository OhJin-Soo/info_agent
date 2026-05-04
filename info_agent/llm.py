from __future__ import annotations

import json
import os
import re
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


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


@dataclass(frozen=True)
class ResearchPlan:
    keywords: list[str]
    queries: list[str]
    provider: str


class LLMClient(Protocol):
    provider: str

    def create_research_plan(self, text: str) -> ResearchPlan:
        raise NotImplementedError

    def summarize_result(self, *, user_text: str, title: str, source_text: str) -> str:
        raise NotImplementedError


class DeterministicLLM:
    provider = "deterministic"

    def create_research_plan(self, text: str) -> ResearchPlan:
        keywords = extract_keywords(text)
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

    def create_research_plan(self, text: str) -> ResearchPlan:
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
        prompt = (
            "Extract concise research keywords and search queries from the user's draft. "
            "Return keywords in the draft language when useful, but make at least one query "
            "strong for English web/Wikipedia search.\n\n"
            f"Draft:\n{text}"
        )
        payload = self._request_json(
            instructions="You are a research planning assistant. Return only schema-valid JSON.",
            input_text=prompt,
            schema_name="research_plan",
            schema=schema,
        )
        keywords = clean_string_list(payload.get("keywords"), limit=6)
        queries = clean_string_list(payload.get("queries"), limit=4)
        if not queries:
            queries = build_queries(keywords, text)
        return ResearchPlan(keywords=keywords or extract_keywords(text), queries=queries, provider=self.provider)

    def summarize_result(self, *, user_text: str, title: str, source_text: str) -> str:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"summary": {"type": "string", "maxLength": 420}},
            "required": ["summary"],
        }
        prompt = (
            "Summarize the source for a writer who is drafting the text below. "
            "Keep it factual, cite no claims that are absent from the source, and write in Korean "
            "when the draft is Korean.\n\n"
            f"Draft:\n{user_text[:1200]}\n\n"
            f"Source title:\n{title}\n\n"
            f"Source text:\n{source_text[:2400]}"
        )
        payload = self._request_json(
            instructions="You summarize retrieved research sources for a text editor.",
            input_text=prompt,
            schema_name="source_summary",
            schema=schema,
        )
        return summarize(str(payload.get("summary", "")) or source_text)

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

    def create_research_plan(self, text: str) -> ResearchPlan:
        prompt = (
            "Return JSON with keys keywords and queries. keywords is up to 6 strings. "
            "queries is up to 4 search queries. No markdown.\n\n"
            f"Draft:\n{text}"
        )
        payload = parse_json_object(self._generate(prompt))
        keywords = clean_string_list(payload.get("keywords"), limit=6)
        queries = clean_string_list(payload.get("queries"), limit=4)
        return ResearchPlan(
            keywords=keywords or extract_keywords(text),
            queries=queries or build_queries(keywords, text),
            provider=self.provider,
        )

    def summarize_result(self, *, user_text: str, title: str, source_text: str) -> str:
        prompt = (
            "Summarize this source for the user's draft in 2 concise Korean sentences. "
            "Use only the source text. No markdown.\n\n"
            f"Draft:\n{user_text[:1200]}\n\nTitle:\n{title}\n\nSource:\n{source_text[:2400]}"
        )
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
    def __init__(self, primary: LLMClient | None, fallback: LLMClient | None = None) -> None:
        self.primary = primary
        self.fallback = fallback or DeterministicLLM()
        self._primary_failed = False
        self.provider = primary.provider if primary else self.fallback.provider
        self.last_plan_provider = self.fallback.provider if primary is None else primary.provider
        self.last_summary_provider = self.last_plan_provider

    def create_research_plan(self, text: str) -> ResearchPlan:
        if self.primary is None or self._primary_failed:
            plan = self.fallback.create_research_plan(text)
            self.last_plan_provider = self.fallback.provider if self.primary is None else f"{self.primary.provider}:fallback"
            return ResearchPlan(plan.keywords, plan.queries, self.last_plan_provider)
        try:
            plan = self.primary.create_research_plan(text)
            self.last_plan_provider = self.primary.provider
            return plan
        except (urllib.error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError, ValueError):
            self._primary_failed = True
            plan = self.fallback.create_research_plan(text)
            self.last_plan_provider = f"{self.primary.provider}:fallback"
            return ResearchPlan(plan.keywords, plan.queries, self.last_plan_provider)

    def summarize_result(self, *, user_text: str, title: str, source_text: str) -> str:
        if self.primary is None or self._primary_failed:
            self.last_summary_provider = self.fallback.provider if self.primary is None else f"{self.primary.provider}:fallback"
            return self.fallback.summarize_result(user_text=user_text, title=title, source_text=source_text)
        try:
            summary = self.primary.summarize_result(user_text=user_text, title=title, source_text=source_text)
            self.last_summary_provider = self.primary.provider
            return summary
        except (urllib.error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError, ValueError):
            self._primary_failed = True
            self.last_summary_provider = f"{self.primary.provider}:fallback"
            return self.fallback.summarize_result(user_text=user_text, title=title, source_text=source_text)


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


def clean_string_list(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = re.sub(r"\s+", " ", str(item)).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned[:limit]


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
