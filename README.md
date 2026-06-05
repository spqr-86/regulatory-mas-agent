# Regulatory MAS Agent

Multi-Agent System for answering questions on Russian regulatory documents (НПА, ГОСТ, ТК РФ).

Built on top of [Regulatory RAG](https://github.com/spqr-86/regulatory-rag) exposed as an MCP server.

## Architecture

```
User query
     │
     ▼
input_guard ──── off-topic ──► "Задайте вопрос по нормативным документам"
     │
     ▼
Coordinator (LangGraph)
  ├── create_plan()  →  CoordinatorPlan (Pydantic)
  │
  ├── RegulationsAgent  →  MCP: search_regulations  →  SIA RAG pipeline
  │                        (HyDE: генерация гипотетического НПА-текста перед поиском)
  │
  ├── WebAgent          →  MCP: web_search           →  Tavily
  │
  └── CalcAgent         →  calculator (eval)
          │
          ▼
     synthesize()
          │
          ▼
     CriticAgent  ──── score < 7 ──► revision
          │
          ▼
     Final answer
```

## Memory

| Layer | Storage | Scope |
|-------|---------|-------|
| Short-term | SQLite (SqliteSaver) | session history |
| Long-term profiles | SQLite `user_profiles` | org, industry |
| Long-term semantic | ChromaDB `user_history` | past queries |

Pluggable: swap SQLite → Postgres / Redis via LangGraph checkpoint adapters.

## MCP Servers

| Server | Repo | Tool |
|--------|------|------|
| SIA RAG | [regulatory-rag](https://github.com/spqr-86/regulatory-rag) | `search_regulations` |
| Web search | built-in | `web_search` (Tavily) |

## Stack

- **Orchestration:** LangGraph (StateGraph, MemorySaver, SqliteSaver)
- **MCP:** FastMCP (server), `langchain-mcp-adapters` (client)
- **LLM:** OpenAI gpt-4o-mini (agents) + gpt-4o (critic, judge)
- **Memory:** SQLite + ChromaDB
- **UI:** Streamlit (TAO step visualization)

## Project Structure

```
regulatory-mas-agent/
├── agent/
│   ├── tools.py          # @tool wrappers (MCP client calls)
│   ├── specialists.py    # RegulationsAgent, WebAgent, CalcAgent
│   ├── coordinator.py    # CoordinatorAgent + Pydantic schemas
│   ├── critic.py         # CriticAgent + revision loop
│   └── memory.py         # SqliteSaver + user profiles + ChromaDB history
├── eval/
│   ├── task_basket.py    # 10 regulatory test scenarios
│   ├── graders.py        # deterministic + LLM-as-Judge
│   └── run_eval.py       # Iron User + benchmark vs baseline RAG
├── app.py                # Streamlit UI
├── main.py               # CLI entry point
└── pyproject.toml
```

## Evaluation

Three complementary graders (from [τ-bench](https://arxiv.org/abs/2406.12045) methodology):

| Grader | Type | Checks |
|--------|------|--------|
| `state_grader` | deterministic | correct source found |
| `policy_grader` | heuristic | no hallucinated facts |
| LLM-as-Judge | LLM (gpt-4o) | usefulness, groundedness, efficiency |

Benchmark: MAS vs baseline SIA RAG on 50-question gold dataset.

## Quick Start

```bash
# 1. Start SIA MCP server (in regulatory-rag repo)
cd ../regulatory-rag
source venv/bin/activate
python mcp_server.py

# 2. Run agent
cd ../regulatory-mas-agent
source venv/bin/activate
streamlit run app.py
```

## Roadmap

- [x] Architecture design
- [ ] Этап 1: Core MAS (tools + specialists + coordinator)
- [ ] Этап 1.5: Memory (SqliteSaver + profiles + ChromaDB history)
- [ ] Этап 2: Quality (HyDE + input_guard + CriticAgent)
- [ ] Этап 3: Eval (task basket + graders + Iron User)
- [ ] Этап 4: Streamlit UI + VPS deploy (port 8504)
