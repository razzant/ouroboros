# Исправление: Ошибка "All fallback models match the active one"

## Что было не так
В файле `.ouroboros/.env` была неправильная конфигурация:

```env
# ❌ НЕПРАВИЛЬНО: все модели одинаковые!
OUROBOROS_MODEL=Qwen3.6-35B-A3B-Q4_K_M.gguf
OUROBOROS_MODEL_CODE=Qwen3.6-35B-A3B-Q4_K_M.gguf
OUROBOROS_MODEL_LIGHT=Qwen3.6-35B-A3B-Q4_K_M.gguf
OUROBOROS_BASE_URL=http://localhost:8080/v1
OUROBOROS_MODEL_FALLBACK_LIST=Qwen3.6-35B-A3B-Q4_K_M.gguf  # ← Проблема!
```

**Результат:**
- Основная модель недоступна (локальный сервер не запущен/не отвечает)
- Система пыталась найти fallback модель
- Но все fallback-и совпадали с основной → ошибка!

---

## Что было изменено

Конфиг обновлён на использование **облачных моделей через OpenRouter**:

```env
# ✅ ПРАВИЛЬНО: разные модели с fallback-ами
OUROBOROS_MODEL=anthropic/claude-sonnet-4.6
OUROBOROS_MODEL_CODE=anthropic/claude-sonnet-4.6
OUROBOROS_MODEL_LIGHT=google/gemini-3-pro-preview
OUROBOROS_BASE_URL=https://openrouter.ai/api/v1
OUROBOROS_MODEL_FALLBACK_LIST=google/gemini-2.5-pro-preview,openai/o3,anthropic/claude-sonnet-4.6
```

**Преимущества:**
✅ Модели всегда доступны  
✅ Разные fallback варианты  
✅ Гарантированное восстановление после сбоев  
✅ Быстрый ответ (облачные модели)  

---

## Если хотите вернуть локальную Qwen

См. [LOCAL_LLM_SETUP.md](LOCAL_LLM_SETUP.md) для инструкций по запуску Ollama и правильной конфигурации fallback-ов.

**Ключевой момент:** никогда не делайте все модели в OUROBOROS_MODEL_FALLBACK_LIST одинаковыми!
