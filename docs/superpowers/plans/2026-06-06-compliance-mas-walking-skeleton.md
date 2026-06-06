# Compliance MAS — Walking Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first working vertical slice of the Regulatory Compliance MAS — a LangGraph pipeline that extracts НПА references from a regulatory document, validates each one's currency via a SIA→web cascade, and produces a findings report reproducing the reference `check_regulations_report.md`.

**Architecture:** LangGraph `StateGraph` with three nodes — `extract` (LLM structured output → list of НПА), `npa_validator` (a `create_react_agent` with `sia_search` + `web_search` tools, prompted to cascade SIA-first then web), and `synthesize` (findings → markdown report). State carries the document, extracted НПА, and accumulated findings.

**Tech Stack:** Python 3.11, LangGraph, langchain-openai (gpt-4o-mini), langchain-tavily (web), httpx (SIA HTTP client), Pydantic (boundaries), structlog, pytest.

**Spec:** `docs/superpowers/specs/2026-06-06-compliance-mas-design.md`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `pyproject.toml` | Deps + tool config (ruff/black line 88, pytest markers) |
| `.env.example` | `OPENAI_API_KEY`, `TAVILY_API_KEY`, `SIA_API_URL` |
| `agent/__init__.py` | Package marker |
| `agent/state.py` | Pydantic schemas (`NPA`, `Finding`) + `ComplianceState` TypedDict |
| `agent/tools.py` | `sia_search` (@tool, httpx → SIA /query), `web_search` (@tool, Tavily) |
| `agent/extract.py` | `extract_node` — LLM structured output: text → list[NPA] |
| `agent/specialists.py` | `build_npa_validator()` — create_react_agent; `npa_validator_node` |
| `agent/synthesize.py` | `synthesize_node` — findings → markdown report |
| `agent/graph.py` | `build_graph()` — wires StateGraph |
| `main.py` | CLI entry: read document → run graph → print report |
| `tests/fixtures/instrukciya_spz_excerpt.md` | Excerpt of real СПЗ instruction (section 3 НПА) |
| `tests/conftest.py` | Shared fixtures (sample state, mock LLM/tools) |
| `tests/test_*.py` | Per-module tests |

Каждый узел LangGraph — тонкий (read state → call function → write state), логика в отдельных функциях, тестируемых независимо.

---

## Task 0: Project scaffold

**Files:**
- Create: `pyproject.toml`, `.env.example`, `agent/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Create branch**

```bash
cd /home/petr/projects/ai/regulatory-mas-agent
git checkout main
git checkout -b feature/walking-skeleton
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "regulatory-mas-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=0.2",
    "langchain-openai>=0.2",
    "langchain-tavily>=0.1",
    "langchain-core>=0.3",
    "httpx>=0.27",
    "pydantic>=2",
    "structlog>=24",
    "python-dotenv>=1",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.23", "ruff", "black"]

[tool.black]
line-length = 88

[tool.ruff]
line-length = 88

[tool.pytest.ini_options]
markers = [
    "unit: fast isolated tests",
    "integration: tests hitting real LLM/SIA (cost money)",
]
addopts = "-m 'not integration'"
```

- [ ] **Step 3: Create `.env.example`**

```bash
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
SIA_API_URL=http://localhost:8503
```

- [ ] **Step 4: Create package markers**

`agent/__init__.py`:
```python
"""Regulatory Compliance MAS — agent package."""
```

`tests/__init__.py`:
```python
```

- [ ] **Step 5: Create venv and install**

Run:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```
Expected: installs without error. Verify: `python -c "import langgraph, langchain_openai, langchain_tavily, httpx; print('ok')"` → `ok`

> **Note:** if `langchain-tavily` import differs in the installed version, sync the API via context7 (`langchain-tavily`) before Task 2. The class used below is `TavilySearch`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .env.example agent/__init__.py tests/__init__.py
git commit -m "chore: scaffold walking-skeleton project (deps, structure)"
```

---

## Task 1: State and schemas

**Files:**
- Create: `agent/state.py`
- Test: `tests/test_state.py`

- [ ] **Step 1: Write the failing test**

`tests/test_state.py`:
```python
import pytest
from agent.state import NPA, Finding, ComplianceState


@pytest.mark.unit
def test_npa_minimal_fields():
    npa = NPA(raw="ПП РФ № 1479 от 16.09.2020", kind="ПП", number="1479")
    assert npa.number == "1479"
    assert npa.kind == "ПП"


@pytest.mark.unit
def test_finding_requires_source():
    f = Finding(
        npa_raw="ПП РФ № 1479",
        status="⚠️",
        source="SIA",
        note="действует с изменениями",
    )
    assert f.source == "SIA"
    assert f.status in {"✅", "⚠️", "❌", "❓"}


@pytest.mark.unit
def test_compliance_state_is_typeddict_with_keys():
    # ComplianceState is a TypedDict; instances are plain dicts.
    state: ComplianceState = {
        "document": "текст",
        "doc_meta": {},
        "npa_list": [],
        "findings": [],
        "report": None,
    }
    assert state["document"] == "текст"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_state.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.state'`

- [ ] **Step 3: Write minimal implementation**

`agent/state.py`:
```python
"""State schema for the compliance graph.

Pydantic models live at the boundaries (LLM structured output, findings).
The graph state itself is a TypedDict with an additive reducer on findings.
"""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict
from operator import add

from pydantic import BaseModel, Field

Status = Literal["✅", "⚠️", "❌", "❓"]


class NPA(BaseModel):
    """A regulatory reference extracted from the document."""

    raw: str = Field(..., description="Точная строка ссылки из документа")
    kind: str = Field(..., description="Тип: ФЗ / ПП / Приказ / ГОСТ / СП / иное")
    number: str = Field(..., description="Номер документа, напр. '1479' или '59641'")


class Finding(BaseModel):
    """Verdict on a single НПА, always attributed to a source."""

    npa_raw: str
    status: Status
    source: Literal["SIA", "web", "none"]
    note: str = ""


class ComplianceState(TypedDict):
    """LangGraph state. findings accumulate across (future) parallel nodes."""

    document: str
    doc_meta: dict
    npa_list: list[NPA]
    findings: Annotated[list[Finding], add]
    report: str | None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_state.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add agent/state.py tests/test_state.py
git commit -m "feat: add ComplianceState and NPA/Finding schemas"
```

---

## Task 2: Tools (SIA + web)

**Files:**
- Create: `agent/tools.py`
- Test: `tests/test_tools.py`

The two tools are LangChain `@tool` functions so the react agent can call them. `sia_search` posts to SIA `/query`; `web_search` wraps Tavily. Tests mock the network layer — no real calls.

- [ ] **Step 1: Write the failing test**

`tests/test_tools.py`:
```python
import httpx
import pytest
from agent.tools import sia_search, web_search


@pytest.mark.unit
def test_sia_search_returns_answer_text(monkeypatch):
    def fake_post(url, json, timeout):
        assert json["question"]  # question forwarded
        return httpx.Response(
            200,
            json={
                "answer": "ПП 1479 действует, ред. ПП 90 от 03.02.2025",
                "passages": [{"text": "п.50 ...", "source": "1479.pdf", "score": 0.8}],
                "path": "rag_simple",
                "elapsed_sec": 1.2,
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    out = sia_search.invoke({"query": "актуальна ли ПП 1479"})
    assert "1479" in out
    assert "SIA" in out  # tool labels its source


@pytest.mark.unit
def test_sia_search_handles_unavailable(monkeypatch):
    def fake_post(url, json, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", fake_post)
    out = sia_search.invoke({"query": "x"})
    assert "недоступен" in out.lower() or "SIA_UNAVAILABLE" in out


@pytest.mark.unit
def test_web_search_returns_results(monkeypatch):
    class FakeTavily:
        def invoke(self, q):
            return [{"title": "Гарант", "content": "ГОСТ Р 59641 Изм.1 от 01.10.2024", "url": "garant.ru"}]

    monkeypatch.setattr("agent.tools._tavily", FakeTavily())
    out = web_search.invoke({"query": "ГОСТ Р 59641 актуальность"})
    assert "59641" in out
    assert "web" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.tools'`

- [ ] **Step 3: Write minimal implementation**

`agent/tools.py`:
```python
"""Tools the npa_validator agent can call: SIA corpus and web fallback."""
from __future__ import annotations

import os

import httpx
import structlog
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

logger = structlog.get_logger()

SIA_API_URL = os.environ.get("SIA_API_URL", "http://localhost:8503")
_SIA_TIMEOUT = 30.0

# Tavily client is module-level so tests can monkeypatch agent.tools._tavily.
_tavily = TavilySearch(max_results=5)


@tool
def sia_search(query: str) -> str:
    """Search the curated SIA regulatory corpus (priority source).

    Use FIRST for any НПА that may be in the corpus (ПП 1479, 69-ФЗ, ТК РФ,
    ОТ-приказы). Returns the corpus answer with passages, labeled source=SIA.
    If SIA has nothing or is unavailable, fall back to web_search.
    """
    try:
        resp = httpx.post(
            f"{SIA_API_URL}/query",
            json={"question": query},
            timeout=_SIA_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as exc:
        logger.warning("sia_search.unavailable", error=str(exc))
        return "SIA_UNAVAILABLE: сервис недоступен, используй web_search."

    answer = data.get("answer") or ""
    passages = data.get("passages") or []
    if not answer and not passages:
        return "SIA_EMPTY: в корпусе ничего не найдено, используй web_search."

    srcs = ", ".join(p.get("source", "") for p in passages[:3])
    return f"[source=SIA] {answer}\nИсточники корпуса: {srcs}"


@tool
def web_search(query: str) -> str:
    """Search the open web (fallback source, lower reliability).

    Use ONLY when SIA returned SIA_EMPTY or SIA_UNAVAILABLE. Prefer
    pravo.gov.ru and garant.ru in the query. Results labeled source=web.
    """
    try:
        results = _tavily.invoke(query)
    except Exception as exc:  # tavily raises various; degrade gracefully
        logger.warning("web_search.failed", error=str(exc))
        return "WEB_UNAVAILABLE: веб-поиск не сработал."

    if not results:
        return "WEB_EMPTY: ничего не найдено."

    lines = []
    for r in results[:5]:
        content = (r.get("content") or "")[:300]
        url = r.get("url", "")
        lines.append(f"- {content} ({url})")
    body = "\n".join(lines)
    return f"[source=web] надёжность ниже, проверяй:\n{body}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tools.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add agent/tools.py tests/test_tools.py
git commit -m "feat: add sia_search and web_search tools with graceful degradation"
```

---

## Task 3: Extract node

**Files:**
- Create: `agent/extract.py`, `tests/fixtures/instrukciya_spz_excerpt.md`
- Test: `tests/test_extract.py`

Извлечение НПА из текста через `with_structured_output`. В unit-тесте LLM мокается (фиксированный список); fixture — реальный раздел 3 инструкции.

- [ ] **Step 1: Create the fixture**

`tests/fixtures/instrukciya_spz_excerpt.md` (реальный раздел 3 из инструкции СПЗ):
```markdown
## 3. Нормативные документы

1. Постановление Правительства РФ от 16.09.2020 № 1479 «Об утверждении Правил противопожарного режима в Российской Федерации» (в ред. ПП РФ от 03.02.2025 № 90, вступило в силу 01.09.2025).
2. ГОСТ Р 59641-2021 «Средства первичного пожаротушения. Руководство по размещению, техническому обслуживанию и ремонту» (с Изменением № 1 от 01.10.2024).
3. Инструкция о мерах пожарной безопасности в ООО «РТИТС».
4. СП 3.13130.2009 «Системы противопожарной защиты. Система оповещения и управления эвакуацией людей при пожаре. Требования пожарной безопасности».
```

- [ ] **Step 2: Write the failing test**

`tests/test_extract.py`:
```python
import pytest
from pathlib import Path
from agent.state import NPA
from agent.extract import extract_node, _ExtractResult


FIXTURE = Path("tests/fixtures/instrukciya_spz_excerpt.md")


class _FakeStructuredLLM:
    """Stands in for ChatOpenAI.with_structured_output(_ExtractResult)."""

    def invoke(self, _messages):
        return _ExtractResult(
            npa_list=[
                NPA(raw="ПП РФ от 16.09.2020 № 1479", kind="ПП", number="1479"),
                NPA(raw="ГОСТ Р 59641-2021", kind="ГОСТ", number="59641"),
                NPA(raw="СП 3.13130.2009", kind="СП", number="3.13130"),
            ]
        )


@pytest.mark.unit
def test_extract_node_pulls_npa(monkeypatch):
    monkeypatch.setattr("agent.extract._get_structured_llm", lambda: _FakeStructuredLLM())
    state = {"document": FIXTURE.read_text(encoding="utf-8"), "doc_meta": {},
             "npa_list": [], "findings": [], "report": None}

    out = extract_node(state)

    numbers = {n.number for n in out["npa_list"]}
    assert "1479" in numbers
    assert "59641" in numbers
    # Внутренняя инструкция (не НПА) не должна попасть как ФЗ/ПП/ГОСТ/СП
    assert all(n.kind in {"ФЗ", "ПП", "Приказ", "ГОСТ", "СП", "иное"} for n in out["npa_list"])


@pytest.mark.unit
def test_extract_node_returns_only_npa_list_key(monkeypatch):
    monkeypatch.setattr("agent.extract._get_structured_llm", lambda: _FakeStructuredLLM())
    state = {"document": "x", "doc_meta": {}, "npa_list": [], "findings": [], "report": None}
    out = extract_node(state)
    assert set(out.keys()) == {"npa_list"}  # узел пишет только свою часть state
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_extract.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.extract'`

- [ ] **Step 4: Write minimal implementation**

`agent/extract.py`:
```python
"""Extract node: regulatory document text -> list[NPA] via structured output."""
from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from agent.state import NPA, ComplianceState

_SYSTEM = (
    "Ты извлекаешь ссылки на нормативные правовые акты (НПА) из текста "
    "российского нормативного документа. Извлекай только внешние НПА: "
    "Федеральные законы (ФЗ), Постановления Правительства (ПП), Приказы "
    "министерств, ГОСТ, СП, СНиП. НЕ извлекай внутренние документы организации "
    "(инструкции, положения самого общества) — для них kind не определён. "
    "Для каждого НПА верни точную строку (raw), тип (kind), номер (number)."
)


class _ExtractResult(BaseModel):
    npa_list: list[NPA]


def _get_structured_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(
        _ExtractResult
    )


def extract_node(state: ComplianceState) -> dict:
    llm = _get_structured_llm()
    result: _ExtractResult = llm.invoke(
        [SystemMessage(content=_SYSTEM), HumanMessage(content=state["document"])]
    )
    return {"npa_list": result.npa_list}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_extract.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add agent/extract.py tests/test_extract.py tests/fixtures/instrukciya_spz_excerpt.md
git commit -m "feat: add extract node (document text -> NPA list)"
```

---

## Task 4: npa_validator (react agent + cascade)

**Files:**
- Create: `agent/specialists.py`
- Test: `tests/test_specialists.py`

`npa_validator` — это `create_react_agent` с tools `[sia_search, web_search]`, промпт задаёт каскад (SIA → web → ❓) и формат вывода. Узел гоняет агента по каждому НПА и собирает `Finding`. В unit-тесте подменяем сам агент на стаб (детерминированный), проверяя оркестрацию узла; вызовы tools тестируются отдельно (Task 2).

- [ ] **Step 1: Write the failing test**

`tests/test_specialists.py`:
```python
import pytest
from agent.state import NPA, Finding
from agent import specialists


class _FakeAgent:
    """Stub for the compiled react agent; returns a fixed final message."""

    def __init__(self, verdict_text):
        self._verdict = verdict_text

    def invoke(self, _inputs):
        return {"messages": [type("M", (), {"content": self._verdict})()]}


@pytest.mark.unit
def test_npa_validator_builds_finding_per_npa(monkeypatch):
    # Agent claims ПП 1479 is current via SIA.
    monkeypatch.setattr(
        specialists, "_get_npa_agent",
        lambda: _FakeAgent("STATUS: ⚠️\nSOURCE: SIA\nNOTE: действует с изменениями")
    )
    state = {
        "document": "x", "doc_meta": {},
        "npa_list": [NPA(raw="ПП РФ № 1479", kind="ПП", number="1479")],
        "findings": [], "report": None,
    }

    out = specialists.npa_validator_node(state)

    assert len(out["findings"]) == 1
    f = out["findings"][0]
    assert isinstance(f, Finding)
    assert f.status == "⚠️"
    assert f.source == "SIA"
    assert "1479" in f.npa_raw


@pytest.mark.unit
def test_parse_verdict_handles_unparseable():
    f = specialists._parse_verdict("ПП РФ № 1479", "бессвязный текст без меток")
    assert f.status == "❓"
    assert f.source == "none"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_specialists.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.specialists'`

- [ ] **Step 3: Write minimal implementation**

`agent/specialists.py`:
```python
"""npa_validator specialist: react agent with SIA->web cascade per НПА."""
from __future__ import annotations

import re

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agent.state import NPA, Finding, ComplianceState, Status
from agent.tools import sia_search, web_search

_PROMPT = (
    "Ты проверяешь актуальность одного российского НПА. "
    "Каскад источников строго в порядке:\n"
    "1) Сначала вызови sia_search (приоритетный корпус).\n"
    "2) Если вернулось SIA_EMPTY или SIA_UNAVAILABLE — вызови web_search.\n"
    "3) Если и web пусто — статус ❓.\n"
    "Не выдумывай. Верни СТРОГО в формате:\n"
    "STATUS: <✅|⚠️|❌|❓>\n"
    "SOURCE: <SIA|web|none>\n"
    "NOTE: <одна строка: действует / изменения / утратил силу + чем заменён>"
)

_VALID_STATUS = {"✅", "⚠️", "❌", "❓"}


def _get_npa_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return create_react_agent(llm, [sia_search, web_search], prompt=_PROMPT)


def _parse_verdict(npa_raw: str, text: str) -> Finding:
    status_m = re.search(r"STATUS:\s*([✅⚠️❌❓])", text)
    source_m = re.search(r"SOURCE:\s*(SIA|web|none)", text)
    note_m = re.search(r"NOTE:\s*(.+)", text)

    status: Status = status_m.group(1) if status_m else "❓"
    if status not in _VALID_STATUS:
        status = "❓"
    source = source_m.group(1) if source_m else "none"
    note = note_m.group(1).strip() if note_m else ""
    return Finding(npa_raw=npa_raw, status=status, source=source, note=note)


def npa_validator_node(state: ComplianceState) -> dict:
    agent = _get_npa_agent()
    findings: list[Finding] = []
    for npa in state["npa_list"]:
        query = f"Актуальность: {npa.raw}"
        result = agent.invoke({"messages": [("user", query)]})
        final = result["messages"][-1].content
        findings.append(_parse_verdict(npa.raw, final))
    return {"findings": findings}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_specialists.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add agent/specialists.py tests/test_specialists.py
git commit -m "feat: add npa_validator react agent with SIA->web cascade"
```

---

## Task 5: Synthesize node

**Files:**
- Create: `agent/synthesize.py`
- Test: `tests/test_synthesize.py`

Детерминированный (без LLM) рендер findings в markdown-таблицу формата `check_regulations_report.md`.

- [ ] **Step 1: Write the failing test**

`tests/test_synthesize.py`:
```python
import pytest
from agent.state import Finding
from agent.synthesize import synthesize_node


@pytest.mark.unit
def test_synthesize_renders_table_with_source_column():
    state = {
        "document": "x", "doc_meta": {"title": "Инструкция СПЗ"},
        "npa_list": [],
        "findings": [
            Finding(npa_raw="ПП РФ № 1479", status="⚠️", source="SIA",
                    note="действует с изменениями (ПП 90)"),
            Finding(npa_raw="ГОСТ Р 59641-2021", status="⚠️", source="web",
                    note="Изм.1 от 01.10.2024"),
        ],
        "report": None,
    }
    out = synthesize_node(state)
    report = out["report"]
    assert "| НПА |" in report
    assert "Источник" in report  # source column present
    assert "1479" in report and "59641" in report
    assert "SIA" in report and "web" in report


@pytest.mark.unit
def test_synthesize_marks_empty_findings():
    state = {"document": "x", "doc_meta": {}, "npa_list": [],
             "findings": [], "report": None}
    out = synthesize_node(state)
    assert "НПА не обнаружены" in out["report"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthesize.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.synthesize'`

- [ ] **Step 3: Write minimal implementation**

`agent/synthesize.py`:
```python
"""Synthesize node: findings -> markdown report (check_regulations_report format)."""
from __future__ import annotations

from agent.state import ComplianceState


def synthesize_node(state: ComplianceState) -> dict:
    findings = state["findings"]
    title = state.get("doc_meta", {}).get("title", "документ")

    if not findings:
        return {"report": f"## Проверка НПА — {title}\n\nНПА не обнаружены."}

    lines = [
        f"## Проверка НПА — {title}",
        "",
        "| НПА | Статус | Источник | Примечание |",
        "|-----|--------|----------|------------|",
    ]
    for f in findings:
        note = f.note.replace("|", "/")
        lines.append(f"| {f.npa_raw} | {f.status} | {f.source} | {note} |")

    outdated = [f for f in findings if f.status == "❌"]
    if outdated:
        lines += ["", "### Требуют замены:"]
        lines += [f"- {f.npa_raw} — {f.note}" for f in outdated]

    return {"report": "\n".join(lines)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthesize.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add agent/synthesize.py tests/test_synthesize.py
git commit -m "feat: add synthesize node (findings -> markdown report)"
```

---

## Task 6: Graph assembly

**Files:**
- Create: `agent/graph.py`
- Test: `tests/test_graph.py`

Линейный граф: `extract → npa_validator → synthesize`. End-to-end тест мокает три узловые функции на уровне импортов графа, проверяя проводку и поток state.

- [ ] **Step 1: Write the failing test**

`tests/test_graph.py`:
```python
import pytest
from agent.state import NPA, Finding
from agent import graph as graph_mod


@pytest.mark.unit
def test_graph_runs_extract_validate_synthesize(monkeypatch):
    monkeypatch.setattr(graph_mod, "extract_node",
        lambda s: {"npa_list": [NPA(raw="ПП РФ № 1479", kind="ПП", number="1479")]})
    monkeypatch.setattr(graph_mod, "npa_validator_node",
        lambda s: {"findings": [Finding(npa_raw="ПП РФ № 1479", status="⚠️",
                                        source="SIA", note="изменения")]})
    monkeypatch.setattr(graph_mod, "synthesize_node",
        lambda s: {"report": "## Проверка НПА\n| 1479 | ⚠️ | SIA |"})

    app = graph_mod.build_graph()
    result = app.invoke({
        "document": "doc", "doc_meta": {}, "npa_list": [],
        "findings": [], "report": None,
    })

    assert result["report"].startswith("## Проверка НПА")
    assert len(result["findings"]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_graph.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.graph'`

- [ ] **Step 3: Write minimal implementation**

`agent/graph.py`:
```python
"""Assemble the walking-skeleton graph: extract -> npa_validator -> synthesize."""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from agent.state import ComplianceState
from agent.extract import extract_node
from agent.specialists import npa_validator_node
from agent.synthesize import synthesize_node


def build_graph():
    g = StateGraph(ComplianceState)
    g.add_node("extract", extract_node)
    g.add_node("npa_validator", npa_validator_node)
    g.add_node("synthesize", synthesize_node)

    g.add_edge(START, "extract")
    g.add_edge("extract", "npa_validator")
    g.add_edge("npa_validator", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_graph.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run full unit suite**

Run: `pytest -v`
Expected: PASS — all unit tests green (state, tools, extract, specialists, synthesize, graph).

- [ ] **Step 6: Commit**

```bash
git add agent/graph.py tests/test_graph.py
git commit -m "feat: assemble walking-skeleton graph"
```

---

## Task 7: CLI entry + integration smoke

**Files:**
- Create: `main.py`
- Test: `tests/test_main.py` (unit) + manual integration run

- [ ] **Step 1: Write the failing test**

`tests/test_main.py`:
```python
import pytest
from pathlib import Path
import main as main_mod


@pytest.mark.unit
def test_run_reads_file_and_returns_report(monkeypatch, tmp_path):
    doc = tmp_path / "d.md"
    doc.write_text("## 3. Нормативные документы\n1. ПП РФ № 1479", encoding="utf-8")

    class FakeApp:
        def invoke(self, state):
            assert state["document"].startswith("## 3")
            return {**state, "report": "## Проверка НПА — d"}

    monkeypatch.setattr(main_mod, "build_graph", lambda: FakeApp())
    report = main_mod.run(str(doc))
    assert "Проверка НПА" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_main.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'main'`

- [ ] **Step 3: Write minimal implementation**

`main.py`:
```python
"""CLI entry: read a regulatory document, run the compliance graph, print report."""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

from agent.graph import build_graph

load_dotenv()


def run(path: str) -> str:
    document = Path(path).read_text(encoding="utf-8")
    title = Path(path).stem
    app = build_graph()
    state = {
        "document": document,
        "doc_meta": {"title": title},
        "npa_list": [],
        "findings": [],
        "report": None,
    }
    result = app.invoke(state)
    return result["report"] or "(пустой отчёт)"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <path-to-document.md>")
        sys.exit(1)
    print(run(sys.argv[1]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_main.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: add CLI entry point"
```

- [ ] **Step 6: Integration smoke (real LLM + SIA) — reproduce the reference**

First ensure SIA API is up:
```bash
# in regulatory-rag, separate tmux window 'sia'
cd /home/petr/projects/ai/regulatory-rag && source venv/bin/activate
nohup uvicorn api:app --host 0.0.0.0 --port 8503 > /tmp/sia_api.log 2>&1 &
sleep 20 && curl -s localhost:8503/health
```
Expected: health returns ok.

Then run against the real instruction:
```bash
cd /home/petr/projects/ai/regulatory-mas-agent && source venv/bin/activate
cp /home/petr/knowledge/workspace/projects/pb/инструкция_спз_ртитс_новая.md /tmp/instr.md
python main.py /tmp/instr.md
```
Expected (compare to `check_regulations_report.md`):
- ПП 1479 found, status ⚠️, source SIA (corpus has 1479)
- ГОСТ Р 59641-2021 found, status ⚠️, source web
- СП 3.13130.2009 found (bonus; status from web or ❓)

If ПП 1479 does not resolve via SIA, check SIA corpus is loaded (`curl` a known query) before treating it as a code bug.

- [ ] **Step 7: Record the smoke result**

Append the actual CLI output to `docs/superpowers/plans/2026-06-06-compliance-mas-walking-skeleton.md` under a "Smoke run" heading, and note any deltas vs the reference report. Commit:
```bash
git add docs/
git commit -m "docs: record walking-skeleton smoke run vs reference"
```

---

## Self-Review

**Spec coverage (walking-skeleton subset of spec §4, §5, §12 step 1):**
- extract (§4 node 2) → Task 3 ✅
- npa_validator + SIA→web cascade (§4 node 3, §5) → Tasks 2,4 ✅
- synthesize report format (§4 node 4, §6) → Task 5 ✅
- provenance/source on every finding (§4 cascade) → Finding.source enforced in schema (Task 1), rendered (Task 5) ✅
- reproduce reference report (§2 criterion 1) → Task 7 step 6 ✅
- graceful errors: SIA unavailable → web (§9) → Task 2 ✅
- TDD per node (§10) → every task ✅
- *Out of skeleton (later plans):* requirement_validator, sia_crosscheck, critic, input_guard, eval graders, Streamlit. Tracked in spec §12 steps 2-5. Not gaps — deferred by design.

**Placeholder scan:** No TBD/TODO; every code step has complete code. Tavily import carries a context7-verify note (Task 0 step 5) — concrete class named (`TavilySearch`), not a placeholder.

**Type consistency:** `NPA(raw,kind,number)`, `Finding(npa_raw,status,source,note)`, `ComplianceState` keys, node return-dict keys (`npa_list`, `findings`, `report`), `Status` literal — consistent across Tasks 1–7. Node functions referenced in graph (`extract_node`, `npa_validator_node`, `synthesize_node`) match their definitions.

---
