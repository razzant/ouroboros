# ПОСТ-РЕФАКТОРИНГОВЫЙ СВОП: Устранение «хвостов» старой архитектуры

## АНАЛИЗ НАЙДЕННЫХ ПРОБЛЕМ

### КРИТИЧЕСКИЕ (блокирующие / ломающие импорт)

| # | Файл | Проблема | Действие |
|---|------|----------|----------|
| C1 | `colab_launcher.py` | `from google.colab import userdata` / `drive` — ImportError при локальном запуске | **УДАЛИТЬ** файл целиком |
| C2 | `colab_bootstrap_shim.py` | `from google.colab import userdata` / `drive` — ImportError при локальном запуске | **УДАЛИТЬ** файл целиком |
| C3 | `ouroboros/llm.py:123-125` | `HTTP-Referer: https://colab.research.google.com/` в default_headers — утечка контекста | Заменить на `https://github.com/joi-lab/ouroboros_zera` |
| C4 | `ouroboros/tools/evolution_stats.py:28` | `_REPO_DIR = Path(os.environ.get("OUROBOROS_REPO_DIR", "/content/ouroboros_repo"))` — ghost path | Заменить на `/home/zera/ouroboros/drive` |
| C5 | `supervisor/git_ops.py:30` | `REPO_DIR: pathlib.Path = pathlib.Path("/content/ouroboros_repo")` — ghost path | Заменить на `/home/zera/ouroboros/drive` |
| C6 | `supervisor/workers.py:32` | `REPO_DIR: pathlib.Path = pathlib.Path("/content/ouroboros_repo")` — ghost path | Заменить на `/home/zera/ouroboros/drive` |

### СРЕДНИЕ (устаревшие инструкции / мусор)

| # | Файл | Проблема | Действие |
|---|------|----------|----------|
| M1 | `prompts/SYSTEM.md:6` | "I operate in Google Colab" — устаревшая инструкция | Заменить на "I operate locally" |
| M2 | `prompts/SYSTEM.md:150-151` | "Google Colab (Python) — execution environment" | Удалить / заменить |
| M3 | `prompts/SYSTEM.md:152` | "Google Drive (`MyDrive/Ouroboros/`)" — ghost path | Заменить на локальный путь |
| M4 | `prompts/SYSTEM.md:188` | `colab_launcher.py — entry point` в диаграмме | Заменить на `local_launcher.py` |
| M5 | `prompts/SYSTEM.md:190-192` | Диаграмма Google Drive с `state/state.json` | Удалить секцию Google Drive |
| M6 | `prompts/SYSTEM.md:313-314` | "My pricing table (`MODEL_PRICING` in loop.py)" — устаревшая ссылка | Удалить упоминание MODEL_PRICING |
| M7 | `prompts/CONSCIOUSNESS.md:22` | "each round costs money" — budget-инструкция | Заменить на "each round consumes resources" |
| M8 | `prompts/CONSCIOUSNESS.md:38` | "schedule a task to update code" для pricing | Удалить инструкцию по обновлению pricing |
| M9 | `prompts/CONSCIOUSNESS.md:67` | "You have a budget cap for background thinking" | Заменить на "You have resource limits" |
| M10 | `BIBLE.md:62-63` | "budget, code version, environment" | Заменить на "model, code version, environment" |
| M11 | `BIBLE.md:171-172` | "The only resource limit is budget" | Заменить на "The only resource limit is compute" |
| M12 | `BIBLE.md:352` | "Unified TOTAL_BUDGET default to $1" в changelog | Удалить строку из changelog |

### НИЗКИЕ (документация / архитектура)

| # | Файл | Проблема | Действие |
|---|------|----------|----------|
| L1 | `README.md:36` | Диаграмма: `colab_launcher.py` | Заменить на `local_launcher.py` |
| L2 | `README.md:78` | "Quick Start (Google Colab)" | Переименовать / удалить секцию |
| L3 | `README.md:93` | Таблица: `TOTAL_BUDGET` как required | Добавить пометку "legacy" или удалить |
| L4 | `README.md:98-140` | "Set Up Google Colab" — полная инструкция | Удалить / переместить в архив |
| L5 | `README.md:146` | "state lives on Google Drive" | Заменить на "state lives on local filesystem" |
| L6 | `README.md:248` | "Fix: double budget accounting" в changelog | Оставить (исторический факт) |
| L7 | `README.md:274-279` | Множество упоминаний budget/pricing в changelog | Оставить (исторические записи) |
| L8 | `supervisor/events.py:5` | "Extracted from colab_launcher.py" в docstring | Заменить на "Extracted from monolithic launcher" |
| L9 | `supervisor/events.py:198-199` | `launcher = os.path.join(os.getcwd(), "colab_launcher.py")` | Заменить на `local_launcher.py` |
| L10 | `supervisor/__init__.py:1` | "decomposed from monolithic colab_launcher.py" | Заменить на "decomposed from monolithic launcher" |
| L11 | `supervisor/workers.py:47` | "On Linux/Colab, 'spawn' re-imports __main__ (colab_launcher.py)" | Заменить на "re-imports __main__ (local_launcher.py)" |
| L12 | `ouroboros/__init__.py:11` | "colab_launcher.py imports ouroboros.apply_patch" в комментарии | Заменить на "launcher.py imports ouroboros.apply_patch" |

---

## ПЛАН ДЕЙСТВИЙ

### ЭТАП 1: Удаление Colab-файлов
- [ ] Удалить `colab_launcher.py`
- [ ] Удалить `colab_bootstrap_shim.py`

### ЭТАП 2: Исправление критических ghost paths
- [ ] C3: `ouroboros/llm.py:123-125` — HTTP-Referer
- [ ] C4: `ouroboros/tools/evolution_stats.py:28` — REPO_DIR
- [ ] C5: `supervisor/git_ops.py:30` — REPO_DIR
- [ ] C6: `supervisor/workers.py:32` — REPO_DIR

### ЭТАП 3: Очистка системных промптов
- [ ] M1: `prompts/SYSTEM.md:6` — "Google Colab"
- [ ] M2: `prompts/SYSTEM.md:150-151` — Colab environment
- [ ] M3: `prompts/SYSTEM.md:152` — Google Drive
- [ ] M4: `prompts/SYSTEM.md:188` — colab_launcher.py в диаграмме
- [ ] M5: `prompts/SYSTEM.md:190-192` — Google Drive секция
- [ ] M6: `prompts/SYSTEM.md:313-314` — MODEL_PRICING

### ЭТАП 4: Очистка consciousness промпта
- [ ] M7: `prompts/CONSCIOUSNESS.md:22` — "costs money"
- [ ] M8: `prompts/CONSCIOUSNESS.md:38` — pricing update
- [ ] M9: `prompts/CONSCIOUSNESS.md:67` — budget cap

### ЭТАП 5: Очистка BIBLE.md
- [ ] M10: `BIBLE.md:62-63` — budget
- [ ] M11: `BIBLE.md:171-172` — budget limit

### ЭТАП 6: Обновление README.md
- [ ] L1: Диаграмма архитектуры
- [ ] L2-L5: Секция Google Colab

### ЭТАП 7: Обновление комментариев и docstring
- [ ] L8: `supervisor/events.py:5`
- [ ] L9: `supervisor/events.py:198-199`
- [ ] L10: `supervisor/__init__.py:1`
- [ ] L11: `supervisor/workers.py:47`
- [ ] L12: `ouroboros/__init__.py:11`

---

## ФИНАЛЬНАЯ ПРОВЕРКА
После всех изменений:
- [ ] `grep -r "colab" --include="*.py" --include="*.md" .` → 0 результатов (кроме changelog)
- [ ] `grep -r "/content/drive" --include="*.py" .` → 0 результатов
- [ ] `grep -r "/content/ouroboros_repo" --include="*.py" .` → 0 результатов
- [ ] `grep -r "google.colab" --include="*.py" .` → 0 результатов
- [ ] `grep -r "userdata.get" --include="*.py" .` → 0 результатов
- [ ] `grep -r "drive.mount" --include="*.py" .` → 0 результатов
- [ ] `grep -r "TOTAL_BUDGET" --include="*.py" .` → только в supervisor/state.py и legacy code (допустимо)
- [ ] `python3 -m pytest tests/ -v` → все тесты проходят
