# Уроборос

Самомодифицирующийся агент. Работает в Google Colab, общается через Telegram,
хранит код в GitHub, память — на Google Drive.

**Версия:** 2.9.0

---

## Быстрый старт

1. В Colab добавь Secrets:
   - `OPENROUTER_API_KEY` (обязательно)
   - `TELEGRAM_BOT_TOKEN` (обязательно)
   - `TOTAL_BUDGET` (обязательно, в USD)
   - `GITHUB_TOKEN` (обязательно)
   - `OPENAI_API_KEY` (опционально — для web_search)
   - `ANTHROPIC_API_KEY` (опционально — для claude_code_edit)

2. Опционально добавь config-ячейку:
```python
import os
CFG = {
    "GITHUB_USER": "razzant",
    "GITHUB_REPO": "ouroboros",
    "OUROBOROS_MODEL": "openai/gpt-5.2",
    "OUROBOROS_MODEL_CODE": "openai/gpt-5.2-codex",
    "OUROBOROS_MAX_WORKERS": "5",
}
for k, v in CFG.items():
    os.environ[k] = str(v)
```

3. Запусти boot shim (см. `colab_bootstrap_shim.py`).
4. Напиши боту в Telegram. Первый написавший — владелец.

## Архитектура

```
Telegram → colab_launcher.py (thin entry point)
               ↓
           supervisor/           (package)
            ├── state.py         — persistent state + budget + status
            ├── telegram.py      — TG client + formatting + typing
            ├── git_ops.py       — checkout, sync, rescue, safe_restart
            ├── queue.py         — task queue, priority, timeouts, scheduling
            └── workers.py       — worker lifecycle, health, direct chat
               ↓
           ouroboros/             (agent package)
            ├── agent.py         — thin orchestrator (task handling + events)
            ├── loop.py          — LLM tool loop (concurrent tools, retry, cost)
            ├── context.py       — context builder + prompt caching + compaction
            ├── apply_patch.py   — Claude Code CLI apply_patch shim
            ├── tools/           — pluggable tools
            ├── llm.py           — LLM client + cached token tracking
            ├── memory.py        — scratchpad, identity
            └── review.py        — code review utilities
```

`colab_launcher.py` — тонкий entry point: секреты, bootstrap, main loop.
Вся логика супервизора декомпозирована в `supervisor/` пакет.

`agent.py` — тонкий оркестратор. Принимает задачу, собирает контекст,
вызывает LLM loop, эмитит результаты. Не содержит LLM-логики напрямую.

`loop.py` — ядро: LLM-вызов с инструментами в цикле. **Concurrent tool
execution** (ThreadPoolExecutor), retry, effort escalation, per-round cost
logging. Единственное место где происходит взаимодействие LLM ↔ tools.

`context.py` — сборка LLM-контекста из промптов, памяти, логов и состояния.
**Prompt caching** для Anthropic моделей через `cache_control` на статическом
контенте (~10K tokens). `compact_tool_history()` для сжатия старых tool results.

`tools/` — плагинная архитектура инструментов. Каждый модуль экспортирует
`get_tools()`, новые инструменты добавляются как отдельные файлы.
Включает `codebase_digest` — полный обзор кодовой базы за один вызов.

## Структура проекта

```
BIBLE.md                   — Философия и принципы (корень всего)
VERSION                    — Текущая версия (semver)
README.md                  — Это описание
requirements.txt           — Python-зависимости
prompts/
  SYSTEM.md                — Единый системный промпт Уробороса
supervisor/                — Пакет супервизора (декомпозированный launcher):
  __init__.py               — Экспорты
  state.py                  — State: load/save, budget tracking, status text, log rotation
  telegram.py               — TG client, markdown→HTML, send_with_budget, typing
  git_ops.py                — Git: checkout, reset, rescue, deps sync, safe_restart
  queue.py                  — Task queue: priority, enqueue, persist, timeouts, scheduling
  workers.py                — Worker lifecycle: spawn, kill, respawn, health, direct chat
ouroboros/
  __init__.py              — Экспорт make_agent
  utils.py                 — Общие утилиты (нулевой уровень зависимостей)
  apply_patch.py           — Claude Code CLI apply_patch shim
  agent.py                 — Тонкий оркестратор: handle_task, event emission
  loop.py                  — LLM tool loop: concurrent execution, retry, cost tracking
  context.py               — Сборка контекста + prompt caching + compact_tool_history
  tools/                   — Пакет инструментов (плагинная архитектура):
    __init__.py             — Реэкспорт ToolRegistry, ToolContext
    registry.py             — Реестр: schemas, execute, auto-discovery
    core.py                 — Файловые операции + codebase_digest
    git.py                  — Git операции (commit, push, status, diff) + untracked warning
    shell.py                — Shell и Claude Code CLI
    search.py               — Web search
    control.py              — restart, promote, schedule, cancel, review, chat_history
  llm.py                   — LLM-клиент: API вызовы, cached token tracking
  memory.py                — Память: scratchpad, identity, chat_history
  review.py                — Deep review: стратегическая рефлексия
colab_launcher.py          — Тонкий entry point: секреты → init → bootstrap → main loop
colab_bootstrap_shim.py    — Boot shim (вставляется в Colab, не меняется)
```

Структура не фиксирована — Уроборос может менять её по принципу самомодификации.

## Ветки GitHub

| Ветка | Кто | Назначение |
|-------|-----|------------|
| `main` | Владелец (Cursor) | Защищённая. Уроборос не трогает |
| `ouroboros` | Уроборос | Рабочая ветка. Все коммиты сюда |
| `ouroboros-stable` | Уроборос | Fallback при крашах. Обновляется через `promote_to_stable` |

## Команды Telegram

Обрабатываются супервизором (код):
- `/panic` — остановить всё немедленно
- `/restart` — мягкий перезапуск
- `/status` — статус воркеров, очереди, бюджета
- `/review` — запустить deep review
- `/evolve` — включить режим эволюции
- `/evolve stop` — выключить эволюцию

Все остальные сообщения идут в Уробороса (LLM-first, без роутера).

## Режим эволюции

`/evolve` включает непрерывные self-improvement циклы.
Каждый цикл: оценка → стратегический выбор → реализация → smoke test → Bible check → коммит.
Подробности в `prompts/SYSTEM.md`.

## Deep review

`/review` (владелец) или `request_review(reason)` (агент).
Стратегическая рефлексия: тренд сложности, направление эволюции,
соответствие Библии, метрики кода. Scope — на усмотрение Уробороса.

---

## Changelog

### 2.9.0 — Prompt Caching for Anthropic Models

Multipart system message с `cache_control` для кэширования статического контекста.

- Статический контент (SYSTEM.md + BIBLE.md + README.md ~10K tokens) помечен `cache_control: {"type": "ephemeral"}`
- Динамический контент (state, scratchpad, identity, logs) — отдельный блок без кэша
- 11+ system messages → 1 multipart system message (cleaner API contract)
- `apply_message_token_soft_cap` обновлён для multipart content
- Ожидаемая экономия: ~50% prompt costs при multi-round диалогах (Anthropic pricing: cached tokens = 10% cost)
- context.py: 250 → 301 строк (+51)

### 2.8.0 — Concurrent Tool Execution

Параллельное выполнение tool calls через ThreadPoolExecutor.

- 2.8x speedup на параллельных shell/IO вызовах (benchmark: 3×sleep 0.5s)
- Порядок результатов сохраняется (LLM ожидает tool results в порядке запроса)
- Извлечена `_execute_single_tool()` для тестируемости
- loop.py: 203 → 270 строк (+67)

### 2.7.0 — Codebase Digest Tool

Новый инструмент `codebase_digest` — полный обзор кодовой базы за один вызов.

- AST-based extraction: все файлы, классы, функции, размеры
- Заменяет 15+ `repo_read` вызовов в начале каждого evolution цикла
- core.py: 96 → 208 строк (+112, новый функционал)

### 2.6.0 — Agent Loop Decomposition

Извлечение LLM tool loop из agent.py в отдельный модуль `ouroboros/loop.py`.

- `loop.py` (203 строк): core LLM-with-tools loop — retry, effort escalation, tool execution
- `agent.py`: 515 → 358 строк (-157). Теперь чистый оркестратор

### 2.5.0 — Cost Tracking + Restart DRY

Per-round `llm_round` events, `cached_tokens` tracking, `safe_restart()` consolidation.
