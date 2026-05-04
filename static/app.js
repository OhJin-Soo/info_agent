const editor = document.querySelector("#editor");
const results = document.querySelector("#results");
const keywords = document.querySelector("#keywords");
const saveStatus = document.querySelector("#saveStatus");
const researchStatus = document.querySelector("#researchStatus");
const refreshButton = document.querySelector("#refreshButton");
const selectionButton = document.querySelector("#selectionButton");

const STORAGE_KEY = "info-agent-editor-text";
let debounceTimer = null;
let activeController = null;

editor.value = localStorage.getItem(STORAGE_KEY) || "";

function setSavedState(message) {
  saveStatus.textContent = message;
}

function recentContext() {
  const paragraphs = editor.value
    .split(/\n+/)
    .map((line) => line.trim())
    .filter(Boolean);
  return (paragraphs.slice(-3).join("\n") || editor.value).slice(-1200);
}

function selectedText() {
  return editor.value.slice(editor.selectionStart, editor.selectionEnd).trim();
}

function scheduleResearch() {
  clearTimeout(debounceTimer);
  setSavedState("Saving...");
  localStorage.setItem(STORAGE_KEY, editor.value);
  setSavedState("Saved");

  debounceTimer = setTimeout(() => {
    runResearch(recentContext(), "/research");
  }, 900);
}

async function runResearch(text, endpoint) {
  const cleanText = text.trim();
  if (cleanText.length < 12) {
    renderEmpty("검색할 문장을 조금 더 입력하세요.");
    researchStatus.textContent = "Waiting for text";
    return;
  }

  if (activeController) {
    activeController.abort();
  }
  activeController = new AbortController();
  researchStatus.textContent = "Searching...";

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: cleanText }),
      signal: activeController.signal,
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    renderResearch(payload);
  } catch (error) {
    if (error.name === "AbortError") return;
    renderEmpty("검색 중 오류가 발생했습니다.");
    researchStatus.textContent = "Error";
  }
}

function renderResearch(payload) {
  keywords.innerHTML = "";
  for (const keyword of payload.keywords || []) {
    const tag = document.createElement("span");
    tag.className = "keyword";
    tag.textContent = keyword;
    keywords.appendChild(tag);
  }

  const items = payload.results || [];
  if (!items.length) {
    renderEmpty("관련 결과를 찾지 못했습니다.");
    researchStatus.textContent = "No results";
    return;
  }

  results.className = "results";
  results.innerHTML = "";
  const keywordGroups = payload.keyword_results || legacyKeywordGroups(items, payload.search_keywords || payload.keywords || []);
  for (const group of keywordGroups) {
    const section = document.createElement("section");
    section.className = "keyword-section";
    section.innerHTML = `<h3>${escapeHtml(group.keyword)}</h3>`;
    for (const source of ["web", "wikipedia", "youtube"]) {
      const sourceBlock = document.createElement("div");
      sourceBlock.className = "source-block";
      sourceBlock.innerHTML = `<h4>${escapeHtml(sourceLabel(source))}</h4>`;
      const sourceItems = group[source] || [];
      if (!sourceItems.length) {
        const empty = document.createElement("p");
        empty.className = "keyword-empty";
        empty.textContent = `${sourceLabel(source)} 검색 결과가 없습니다.`;
        sourceBlock.appendChild(empty);
      }
      for (const item of sourceItems) {
        sourceBlock.appendChild(createResultCard(item));
      }
      section.appendChild(sourceBlock);
    }
    results.appendChild(section);
  }
  const provider = payload.llm_provider ? ` · keywords ${originLabel(payload.keyword_origin)} · summary ${originLabel(payload.summary_origin)}` : "";
  researchStatus.textContent = `${items.length} results${provider}`;
}

function createResultCard(item) {
  const card = document.createElement("article");
  card.className = "result-card";
  const resultLabel = originLabel(item.result_origin);
  const summaryLabel = originLabel(item.summary_origin);
  const fullSummary = item.full_summary || item.summary || "";
  const preview = item.summary || fullSummary;
  const needsMore = item.is_truncated && fullSummary !== preview;
  card.innerHTML = `
      <div class="badges">
        <span class="badge badge-channel">${escapeHtml(item.channel || item.source)}</span>
        <span class="badge ${badgeClass(item.result_origin)}">${escapeHtml(item.provider_label || resultLabel)}</span>
        <span class="badge badge-keyword">Keyword: ${escapeHtml(item.matched_keyword || "-")}</span>
        <span class="badge ${badgeClass(item.summary_origin)}">Summary: ${escapeHtml(summaryLabel)}</span>
      </div>
      <a href="${escapeAttr(item.url)}" target="_blank" rel="noreferrer">${escapeHtml(item.title)}</a>
      <p class="summary-text" data-preview="${escapeAttr(preview)}" data-full="${escapeAttr(fullSummary)}">${escapeHtml(preview)}</p>
      ${
        needsMore
          ? '<button class="more-button" type="button" aria-expanded="false">더보기</button>'
          : ""
      }
      <div class="meta">
        <span>${escapeHtml(item.source)}</span>
        <span>${escapeHtml(item.summary_provider || "")} · ${Math.round((item.confidence || 0) * 100)}%</span>
      </div>
    `;
  const moreButton = card.querySelector(".more-button");
  if (moreButton) {
    moreButton.addEventListener("click", () => toggleSummary(card));
  }
  return card;
}

function legacyKeywordGroups(items, searchKeywords) {
  const grouped = new Map();
  for (const item of items) {
    const keyword = item.matched_keyword || "";
    if (!grouped.has(keyword)) grouped.set(keyword, []);
    grouped.get(keyword).push(item);
  }
  return searchKeywords.map((keyword) => {
    const keywordItems = grouped.get(keyword) || [];
    return {
      keyword,
      web: keywordItems.filter((item) => item.source === "web"),
      wikipedia: keywordItems.filter((item) => item.source === "wikipedia"),
      youtube: keywordItems.filter((item) => item.source === "youtube"),
    };
  });
}

function sourceLabel(source) {
  return {
    web: "Web",
    wikipedia: "Wikipedia",
    youtube: "YouTube",
  }[source] || source;
}

function renderEmpty(message) {
  keywords.innerHTML = "";
  results.className = "results empty";
  results.textContent = message;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[char];
  });
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/`/g, "&#096;");
}

function originLabel(origin) {
  return {
    llm: "LLM",
    search_api: "Search API",
    fallback: "Fallback",
    fallback_link: "Fallback link",
    mixed: "Mixed",
    none: "None",
  }[origin] || "Unknown";
}

function badgeClass(origin) {
  return {
    llm: "badge-llm",
    search_api: "badge-search",
    fallback: "badge-fallback",
    fallback_link: "badge-fallback",
    mixed: "badge-mixed",
  }[origin] || "badge-fallback";
}

function toggleSummary(card) {
  const summary = card.querySelector(".summary-text");
  const button = card.querySelector(".more-button");
  const expanded = button.getAttribute("aria-expanded") === "true";
  summary.textContent = expanded ? summary.dataset.preview : summary.dataset.full;
  button.textContent = expanded ? "더보기" : "접기";
  button.setAttribute("aria-expanded", String(!expanded));
}

editor.addEventListener("input", scheduleResearch);
refreshButton.addEventListener("click", () => runResearch(recentContext(), "/research"));
selectionButton.addEventListener("click", () => {
  const text = selectedText();
  runResearch(text || recentContext(), text ? "/research/selection" : "/research");
});

if (editor.value.trim()) {
  runResearch(recentContext(), "/research");
}
