# Настройка локальной LLM (например, Qwen3.6-35B)

## Проблема ошибки 
Если вы видите ошибку:
```
⚠️ Failed to get a response from model Qwen3.6-35B-A3B-Q4_K_M.gguf after 3 attempts. 
All fallback models match the active one. Try rephrasing your request.
```

**Причина**: локальный LLM сервер недоступен ИЛИ все fallback модели совпадают с основной.

---

## Решение: Запуск локальной Qwen через Ollama

### 1. Установите Ollama
```bash
# macOS / Linux / Windows
curl -fsSL https://ollama.ai/install.sh | sh
# ИЛИ скачайте с https://ollama.ai
```

### 2. Запустите сервер Ollama с поддержкой OpenAI API

```bash
# Запустите Ollama демон
ollama serve
```

В отдельном терминале:
```bash
# Загрузите модель (если ещё не загружена)
ollama pull qwen3.6:35b

# ИЛИ используйте другие модели:
ollama pull llama2
ollama pull mistral
```

Ollama автоматически запустит OpenAI-совместимый API на `http://localhost:11434/v1`

### 3. Обновите `.ouroboros/.env`

```env
# === Model Configuration (local Ollama) ===
OUROBOROS_MODEL=qwen3.6:35b
OUROBOROS_MODEL_CODE=qwen3.6:35b
OUROBOROS_MODEL_LIGHT=qwen3.6:35b
OUROBOROS_BASE_URL=http://localhost:11434/v1

# === КРИТИЧНО: добавьте облачные fallback модели ===
# (на случай, если локальный сервер упадёт)
OUROBOROS_MODEL_FALLBACK_LIST=qwen3.6:35b,anthropic/claude-sonnet-4.6,google/gemini-3-pro-preview
```

**Зачем fallback модели?** 
- Если локальный сервер недоступен → система переключится на облачные модели
- Если основная модель вернула пустой ответ → система попробует резервный вариант

### 4. Проверьте подключение

```bash
# Тест локального сервера
curl http://localhost:11434/v1/models

# Должно вернуться что-то типа:
# {"object":"list","data":[{"id":"qwen3.6:35b","object":"model"}]}
```

### 5. Рестартуйте бота

```bash
python local_launcher.py
```

---

## Альтернативный вариант: LM Studio

Если Ollama не подходит:

1. Скачайте LM Studio: https://lmstudio.ai
2. Загрузите модель в UI
3. Запустите сервер на `http://localhost:1234/v1`
4. Обновите `.env`:
   ```env
   OUROBOROS_BASE_URL=http://localhost:1234/v1
   ```

---

## Проблема: "Connection refused"

**Если видите ошибку подключения:**

1. Проверьте, что сервер запущен:
   ```bash
   curl http://localhost:11434/v1/models
   ```

2. Если не запущен:
   ```bash
   # Запустите Ollama в фоне
   ollama serve &
   ```

3. Если нужен быстрый fallback на облако, просто измените `.env`:
   ```env
   OUROBOROS_BASE_URL=https://openrouter.ai/api/v1
   OUROBOROS_MODEL=anthropic/claude-sonnet-4.6
   ```

---

## Производительность

- **Qwen3.6-35B локально**: медленнее облачных моделей (~5-30 сек на ответ)
- **Облачные модели**: быстрее (0.5-2 сек)
- **Гибридный подход**: используйте локальную модель для экономии, облачную для скорости

Рекомендация: `OUROBOROS_MODEL=qwen3.6:35b` (локальная), `OUROBOROS_MODEL_LIGHT=google/gemini-3-pro-preview` (облачная быстрая).

---

## Тестирование (pytest)

Ouroboros использует `pytest` для pre-push проверок. Убедитесь что pytest установлен:

```bash
pip install pytest>=7.0
# ИЛИ
pip install --break-system-packages pytest>=7.0
```

**Важно:** Pre-push тесты являются advisory (неблокирующими). Если тесты падают — push всё равно выполняется, но предупреждение логируется.

Смотрите также: [`requirements.txt`](requirements.txt)
