from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from info_agent.research import ResearchPipeline


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


class ResearchRequest(BaseModel):
    text: str = Field(min_length=1, max_length=12000)


app = FastAPI(title="Research Assistant Text Editor", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

pipeline = ResearchPipeline()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/research")
def research(request: ResearchRequest) -> dict:
    return pipeline.research(request.text)


@app.post("/research/selection")
def research_selection(request: ResearchRequest) -> dict:
    return pipeline.research(request.text)


def main() -> None:
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
