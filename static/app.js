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
  for (const item of items) {
    const card = document.createElement("article");
    card.className = "result-card";
    card.innerHTML = `
      <a href="${escapeAttr(item.url)}" target="_blank" rel="noreferrer">${escapeHtml(item.title)}</a>
      <p>${escapeHtml(item.summary || "")}</p>
      <div class="meta">
        <span>${escapeHtml(item.source)}</span>
        <span>${Math.round((item.confidence || 0) * 100)}%</span>
      </div>
    `;
    results.appendChild(card);
  }
  researchStatus.textContent = `${items.length} results`;
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

editor.addEventListener("input", scheduleResearch);
refreshButton.addEventListener("click", () => runResearch(recentContext(), "/research"));
selectionButton.addEventListener("click", () => {
  const text = selectedText();
  runResearch(text || recentContext(), text ? "/research/selection" : "/research");
});

if (editor.value.trim()) {
  runResearch(recentContext(), "/research");
}
