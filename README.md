# Research Assistant Text Editor MVP

`design.md`의 MVP 범위에 맞춘 FastAPI 기반 Research Assistant Text Editor입니다.

## 기능

- 브라우저 텍스트 에디터
- 입력 내용 자동 저장
- 최근 문단 기반 자동 리서치
- 선택 문장 기반 리서치
- 키워드/검색 쿼리 생성
- Wikipedia 실제 검색
- LLM 기반 결과 요약
- Web/YouTube 검색 링크 fallback
- LLM/Search API/fallback 응답 출처 표시
- Web/Tavily, Wikipedia/Wikipedia API, YouTube/fallback link 채널 구분
- `/research`, `/research/selection` API
- query context 기반 TTL 캐싱

## LLM 설정

기본값은 API 키 없이 동작하는 deterministic fallback입니다. 실제 LLM을 쓰려면 아래 중 하나를 설정합니다.

OpenAI Responses API:

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-5-mini
export OPENAI_REASONING_EFFORT=minimal
export LLM_TIMEOUT_SECONDS=30
```

Ollama:

```bash
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama3.1
export OLLAMA_BASE_URL=http://127.0.0.1:11434
```

파이프라인은 `텍스트 → LLM 키워드/쿼리 추출 → Search API 호출 → LLM 요약` 순서로 실행됩니다. LLM 호출이 실패하면 deterministic fallback으로 응답을 유지합니다.

API 응답에는 `keyword_origin`, `search_origin`, `summary_origin`과 결과별 `result_origin`, `summary_origin`, `summary_provider`가 포함됩니다.

Web Search API:

```bash
export TAVILY_API_KEY=tvly-...
```

`TAVILY_API_KEY`가 있으면 Tavily Search API를 호출하고, 없거나 호출이 실패하면 DuckDuckGo 검색 링크 fallback을 표시합니다.

## 실행

```bash
uv run uvicorn main:app --reload
```

브라우저에서 `http://127.0.0.1:8000`을 열면 됩니다.

## 테스트

```bash
uv run pytest
```
