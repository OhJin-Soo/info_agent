from __future__ import annotations

import os
import re
from functools import lru_cache

from langsmith import traceable

from info_agent.llm import validate_keywords


NOUN_POS = {"NOUN", "PROPN"}


@traceable(run_type="tool", name="spaCy Noun Extraction")
def extract_noun_candidates(text: str, limit: int = 24) -> list[str]:
    """Extract noun-like candidates with spaCy, then keep keyword-safe terms."""
    nlp = load_spacy_pipeline()
    doc = nlp(text)
    candidates: list[str] = []

    try:
        for chunk in doc.noun_chunks:
            candidates.append(chunk.text)
    except NotImplementedError:
        pass

    for token in doc:
        if token.is_space or token.is_punct:
            continue
        if token.pos_ in NOUN_POS or is_noun_like_token(token.text):
            candidates.append(token.text)

    candidates.extend(re.findall(r"[A-Za-z][A-Za-z0-9.+#-]{1,}|[가-힣]{2,}", text))

    return validate_keywords(candidates, limit=limit)


@lru_cache(maxsize=1)
def load_spacy_pipeline():
    import spacy

    model = os.getenv("SPACY_MODEL", "").strip()
    if model:
        try:
            return spacy.load(model)
        except OSError:
            pass

    for candidate in ("en_core_web_sm", "en_core_web_md"):
        try:
            return spacy.load(candidate)
        except OSError:
            continue

    return spacy.blank(os.getenv("SPACY_LANG", "xx"))


def is_noun_like_token(token: str) -> bool:
    if re.fullmatch(r"[A-Z][A-Z0-9]{1,}", token):
        return True
    if re.search(r"[-+#.]", token):
        return True
    if re.fullmatch(r"[가-힣]{2,}", token):
        return True
    return False
