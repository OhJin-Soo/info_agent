# 📄 LLM 기반 Research Assistant Text Editor 설계 문서

## 1. 프로젝트 개요

### 1.1 목적

사용자가 글을 작성하는 과정에서
LLM이 자동으로 관련 자료를 검색하고 요약하여 제공하는
**Research Assistant Text Editor**를 개발한다.

### 1.2 핵심 가치

```text
기존:
사용자 → 검색 → 자료 읽기 → 다시 작성

개선:
사용자 → 작성
          ↓
       자동 검색 + 요약
          ↓
       바로 활용
```

즉, **“글쓰기 + 리서치 통합”**

---

## 2. 시스템 아키텍처

```text
[Mobile App / Web Editor]
        ↓
[Backend API (FastAPI)]
        ↓
[LLM Layer]
        ↓
[Search Tools Layer]
 ├─ Wikipedia API
 ├─ Web Search API (Tavily/Brave)
 └─ YouTube API
        ↓
[Post-processing]
 ├─ 랭킹
 ├─ 요약
 └─ 필터링
```

---

## 3. 핵심 기능

### 3.1 텍스트 에디터

```text
기본 기능
- 텍스트 입력
- 커서/선택
- 자동 저장

확장 기능
- 문장 선택 기반 검색
- 인용 삽입
```

---

### 3.2 LLM 기반 검색 에이전트

#### 역할

```text
1. 텍스트 분석
2. 키워드 추출
3. 검색 쿼리 생성
4. 검색 전략 결정
```

#### 예시

입력:

```text
RAG는 벡터 검색을 활용하여 외부 지식을 활용한다.
```

출력:

```json
{
  "queries": [
    "Retrieval Augmented Generation explained",
    "RAG vector database embedding",
    "RAG tutorial blog",
    "RAG YouTube explanation"
  ]
}
```

---

### 3.3 검색 시스템

#### 데이터 소스

* Wikipedia
* 블로그/웹
* YouTube

#### 구성

```text
Search Orchestrator
 ├─ Wikipedia Search
 ├─ Web Search
 └─ YouTube Search
```

---

### 3.4 결과 후처리

```text
1. 중복 제거
2. 관련도 평가
3. 요약 생성
4. 출처 분류
```

출력 예:

```json
[
  {
    "source": "wikipedia",
    "title": "Retrieval-Augmented Generation",
    "summary": "...",
    "url": "..."
  }
]
```

---

## 4. 에이전트 설계

### 4.1 MVP (추천)

```text
Deterministic Pipeline

[텍스트]
   ↓
[LLM: 키워드 추출]
   ↓
[Search API 호출]
   ↓
[LLM: 요약]
```

👉 이유:

* 안정적
* 디버깅 쉬움
* 비용 예측 가능

---

### 4.2 확장 (고급)

```text
ReAct Agent

Thought → Action → Observation → 반복
```

예:

```text
Thought: RAG 설명 필요
Action: Wikipedia 검색
Observation: 개념 부족
Action: YouTube 검색
```

---

## 5. API 설계

### 5.1 Research API

```http
POST /research
```

요청:

```json
{
  "text": "사용자 입력 텍스트"
}
```

응답:

```json
{
  "keywords": ["RAG", "embedding"],
  "results": [
    {
      "source": "youtube",
      "title": "...",
      "url": "...",
      "summary": "..."
    }
  ]
}
```

---

### 5.2 선택 기반 검색

```http
POST /research/selection
```

---

## 6. 데이터 흐름

```text
사용자 입력
   ↓
debounce (3~5초)
   ↓
최근 문단 추출
   ↓
LLM (쿼리 생성)
   ↓
검색 API 호출
   ↓
결과 수집
   ↓
LLM 요약
   ↓
UI 표시
```

---

## 7. 프론트엔드 구조

```text
Editor Screen
 ├─ Text Editor
 └─ Research Panel
      ├─ Wikipedia
      ├─ Blog
      └─ YouTube
```

UX:

```text
- 자동 추천
- 문장 선택 → 검색
- 카드 UI
- 클릭 시 외부 링크
```

---

## 8. 기술 스택

### 8.1 Frontend

```text
React Native / Flutter / SwiftUI / Kotlin Compose
```

### 8.2 Backend

```text
FastAPI
```

### 8.3 LLM

```text
OpenAI / Local LLM (Ollama)
```

### 8.4 Search API

```text
Tavily / Brave / Google CSE
Wikipedia API
YouTube Data API
```

---

## 9. 성능 및 최적화

### 9.1 비용 최적화

```text
- debounce 적용
- 짧은 context만 사용
- 캐싱 (query 기반)
```

---

### 9.2 응답 속도

```text
- 검색 병렬 처리
- 스트리밍 응답
```

---

### 9.3 캐싱 전략

```text
query → 결과 캐싱
TTL 기반 유지
```

---

## 10. 확장 방향

### 10.1 기능 확장

```text
- 자동 인용 삽입
- 글 요약
- 글 개선 제안
- 스타일 변환
```

---

### 10.2 AI 고도화

```text
- 개인화 추천
- 검색 결과 랭킹 학습
- LLM-as-a-Judge 평가
```

---

### 10.3 협업 기능

```text
- 실시간 공동 편집
- CRDT 기반 동기화
```

---

## 11. 리스크 및 고려사항

```text
검색 품질 문제
할루시네이션
API 비용 증가
응답 지연
```

해결:

```text
출처 표시
confidence score
fallback 전략
```

---

## 12. 포트폴리오 포지셔닝

이 프로젝트는 단순 앱이 아니라:

```text
LLM 기반 Research Agent 시스템
```

으로 설명 가능하다.

핵심 포인트:

```text
LLM + Search Tool + Agent Workflow + UX 통합
```