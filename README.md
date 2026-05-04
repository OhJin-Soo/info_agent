# Research Assistant Text Editor MVP

`design.md`의 MVP 범위에 맞춘 FastAPI 기반 Research Assistant Text Editor입니다.

## 기능

- 브라우저 텍스트 에디터
- 입력 내용 자동 저장
- 최근 문단 기반 자동 리서치
- 선택 문장 기반 리서치
- 키워드/검색 쿼리 생성
- Wikipedia 실제 검색
- Web/YouTube 검색 링크 fallback
- `/research`, `/research/selection` API
- query context 기반 TTL 캐싱

## 실행

```bash
uv run uvicorn main:app --reload
```

브라우저에서 `http://127.0.0.1:8000`을 열면 됩니다.

## 테스트

```bash
uv run pytest
```
