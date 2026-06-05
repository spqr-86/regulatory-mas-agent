# CLAUDE.md — Regulatory MAS Agent

## Описание проекта

Multi-Agent System для ответов на вопросы по российским нормативным документам (НПА, ГОСТ, ТК РФ).

Использует [Regulatory RAG (SIA)](https://github.com/spqr-86/regulatory-rag) как MCP-сервер.

**Стек:** LangGraph, FastMCP, OpenAI, SQLite, ChromaDB, Streamlit
**VPS:** 213.176.64.237, порт 8504 (планируемый)

## Архитектура

```
User query → input_guard → Coordinator
  ├── RegulationsAgent → MCP: SIA RAG (HyDE)
  ├── WebAgent → MCP: web_search (Tavily)
  └── CalcAgent → calculator
        ↓
  synthesize() → CriticAgent (score < 7 → revision) → Final answer
```

**Память:**
- Short-term: SqliteSaver (`data/memory.db`) — история диалога
- Long-term: SQLite таблица `user_profiles` + ChromaDB коллекция `user_history`

## Файловая структура

```
regulatory-mas-agent/
├── CLAUDE.md              ← этот файл
├── README.md
├── PLAN.md                ← полный план этапов
├── pyproject.toml
├── .env.example
├── agent/
│   ├── tools.py           ← 3 @tool: search_regulations, web_search, calculator
│   ├── specialists.py     ← RegulationsAgent, WebAgent, CalcAgent
│   ├── coordinator.py     ← CoordinatorAgent + Pydantic schemas
│   ├── critic.py          ← CriticAgent + revision loop
│   └── memory.py          ← SqliteSaver + user_profiles + ChromaDB history
├── eval/
│   ├── task_basket.py     ← 10-15 нормативных сценариев
│   ├── graders.py         ← deterministic + LLM-as-Judge
│   └── run_eval.py        ← Iron User + benchmark
├── app.py                 ← Streamlit UI
├── main.py                ← CLI entry point
└── data/
    └── memory.db          ← SQLite (gitignored)
```

## Plan

Полный план этапов: **`PLAN.md`**

Краткий roadmap:
- Этап 1: Core MAS (tools + specialists + coordinator)
- Этап 1.5: Memory (SqliteSaver + профили + ChromaDB history)
- Этап 2: Quality (HyDE + input_guard + CriticAgent)
- Этап 3: Eval (task basket + graders + Iron User)
- Этап 4: Streamlit UI + деплой порт 8504

## Git Workflow

**Ветки:**
- `main` — только стабильный код, не пушим напрямую
- `feature/stage-N-name` — каждый этап
- `fix/xxx` — баги
- `chore/xxx` — зависимости, конфиг

**Цикл:**
```bash
git checkout -b feature/stage-1-core-mas
# код
git add .
git commit -m "feat: add tools.py with 3 @tool functions"
git push -u origin feature/stage-1-core-mas
# PR → squash merge → main
```

**Conventional commits:**
- `feat:` — новая функциональность
- `fix:` — баг
- `test:` — тесты
- `docs:` — документация
- `chore:` — настройка, зависимости

**Issues:** каждый этап = issue, закрывается через `Closes #N` в PR.

## Источники из ноутбуков ШАД

| Компонент | Ноутбук | Секция |
|-----------|---------|--------|
| @tool паттерн | lecture_1_2 | 1.3–1.4 |
| create_react_agent() | seminar_3 | Part 1 |
| MemorySaver / SqliteSaver | seminar_2 | Step 1 |
| HyDE промпт | seminar_2 | Step 2 |
| input_guard | seminar_2 | Step 3 |
| CoordinatorAgent + Pydantic | seminar_3 | Part 2 |
| CriticAgent + Reflexion | seminar_3 | Part 3 |
| Task basket + graders | seminar_4 | Part 2–3 |
| Iron User | seminar_4 | Part 5 |
| LLM-as-Judge | seminar_4 | Part 7 |

Ноутбуки: `~/career/shad/materials/`

## Команды

```bash
source venv/bin/activate

# Запустить SIA MCP сервер (в regulatory-rag)
cd ../regulatory-rag && python mcp_server.py

# Запустить агента
python main.py "вопрос"

# Streamlit UI
streamlit run app.py --server.port 8504

# Тесты
pytest -v

# Lint
ruff check . --fix && black .
```

## Code Style

- Type hints везде
- structlog вместо print/logging
- Explicit error handling, no bare except
- Conventional commits, английский язык
- Line length: 88 (Black/Ruff)
- pytest: parametrize, fixtures, markers unit/integration
