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
- LangGraph 기반 research workflow
- `/research`, `/research/selection` API
- query context 기반 TTL 캐싱

## LLM 설정

키워드 후보는 spaCy로 명사 후보를 먼저 추출하고, LLM은 그 후보 중 domain term만 선택합니다. 기본값은 설치된 spaCy 모델을 자동 탐색하고, 없으면 blank tokenizer fallback을 사용합니다.

```bash
export SPACY_MODEL=en_core_web_sm
```

기본값은 API 키 없이 동작하는 deterministic fallback입니다. 실제 LLM을 쓰려면 아래 중 하나를 설정합니다.

OpenAI Responses API:

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-5-mini
export OPENAI_REASONING_EFFORT=minimal
export LLM_TIMEOUT_SECONDS=30
export LLM_CACHE_TTL_SECONDS=3600
```

Ollama:

```bash
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama3.1
export OLLAMA_BASE_URL=http://127.0.0.1:11434
```

파이프라인은 LangGraph workflow로 감싸져 있으며 `텍스트 → spaCy 명사 후보 추출 → LLM 용어 선택 → Search API 호출 → LLM 요약` 순서로 실행됩니다. LLM 호출이 실패하면 deterministic fallback으로 응답을 유지합니다.

LLM plan/summary 결과는 기본 1시간 동안 메모리 캐시에 저장됩니다. 같은 입력의 키워드 추출이나 같은 source 요약은 반복 호출하지 않습니다.

LangSmith tracing을 켜려면 아래 환경변수를 설정합니다.

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=...
export LANGSMITH_PROJECT=info-agent
```

tracing이 켜지면 LangGraph workflow, spaCy 추출, 검색, 요약, LLM plan 단계가 LangSmith에 중첩 span으로 기록됩니다.

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
