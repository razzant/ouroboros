# Watchdog System — Documentation for AI Agents

> **Цель документации:** Эта документация предназначена для нейросетей/AI-агентов, работающих в рамках проекта Ouroboros Zera. Она объясняет систему Watchdog — механизм мониторинга и автоматического восстановления, который обеспечивает непрерывную работу всей системы.

---

## Table of Contents

1. [Что такое Watchdog и зачем он нужен](#1-что-такое-watchdog-и-зачем-он-нужен)
2. [Архитектура Watchdog](#2-архитектура-watchdog)
3. [Как проверить состояние Watchdog](#3-как-проверить-состояние-watchdog)
4. [Как читать логи](#4-как-читать-логи)
5. [Что делать при проблемах](#5-что-делать-при-проблемах)
6. [Взаимосвязь с Supervisor и Llama-Server](#6-взаимосвязь-с-supervisor-и-llama-server)

---

## 1. Что такое Watchdog и зачем он нужен

### Определение

**Watchdog** (Сторожевой пёс) — это фоновый демон, который непрерывно мониторит здоровье ключевых компонентов системы Ouroboros Zera и автоматически восстанавливает их при обнаружении сбоев.

### Зачем нужен

Система Ouroboros Zera работает автономно и выполняет длительные задачи (эволюция кода, ревью, работа с LLM). Без Watchdog:

- **Падающий Supervisor** оставит систему без управления задачами навсегда
- **Неотвечающий Llama-Server** заблокирует все LLM-запросы
- **Проблемы с Telegram API** нарушат уведомления и управление через чат
- **Зависшие процессы** потребуют ручного вмешательства

Watchdog устраняет необходимость ручного вмешательства, автоматически обнаруживая и исправляя проблемы.

### Ключевые функции

| Функция | Описание |
|---------|----------|
| Мониторинг Supervisor | Проверяет, что процесс supervisor (local_launcher.py) жив |
| Мониторинг Llama-Server | Проверяет доступность локального LLM-сервера на порту 8080 |
| Мониторинг Telegram API | Проверяет работоспособность Telegram бота |
| Автоматический рестарт | Перезапускает Supervisor при обнаружении сбоя |
| Rate limiting | Ограничивает количество рестартов (20/24ч) для предотвращения петель |
| Heartbeat проверка | Проверяет свежесть heartbeat Supervisor |

---

## 2. Архитектура Watchdog

### 2.1 Компоненты системы

```
┌─────────────────────────────────────────────────────────────────┐
│                     Systemd (OS Level)                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           watchdog.service (systemd unit)                │  │
│  │  Type=simple, Restart=always, MemoryMax=256M            │  │
│  └─────────────────────┬──────────────────────────────────┘  │
└────────────────────────┼─────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Watchdog (watchdog.sh)                             │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐   │
│  │ Supervisor   │  │ Llama-Server │  │  Telegram API       │   │
│  │ Monitor      │  │ Monitor      │  │  Monitor            │   │
│  │ (PID +       │  │ (HTTP 8080/  │  │  (getMe endpoint)   │   │
│  │  heartbeat)  │  │  health)     │  │                     │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬──────────┘   │
│         │                 │                      │             │
│         ▼                 ▼                      ▼             │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐   │
│  │ Restart if  │  │ Log status   │  │  Log status         │   │
│  │ dead        │  │ (info only)  │  │  (info only)        │   │
│  └─────────────┘  └──────────────┘  └─────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Restart Rate Limiter (max 20 restarts / 24h window)   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Supervisor (local_launcher.py)                     │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐   │
│  │ Worker      │  │ Direct-Mode  │  │  Consciousness      │   │
│  │ Lifecycle   │  │ Watchdog     │  │  Engine             │   │
│  │ (multiproc) │  │ (thread)     │  │  (background)       │   │
│  └─────────────┘  └──────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Llama Server (localhost:8080)                      │
│         Локальный LLM-сервер для inference                     │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Два уровня Watchdog

Система имеет **два уровня** watchdog:

#### Уровень 1: Внешний Watchdog (watchdog.sh)

- **Файл:** [`.ouroboros/watchdog.sh`](.ouroboros/watchdog.sh)
- **Запуск:** systemd через [`.ouroboros/watchdog.service`](.ouroboros/watchdog.service)
- **Язык:** Bash
- **Цикл:** Проверка каждые 30 секунд
- **Ответственность:** Мониторинг процесса Supervisor

```bash
# Основной цикл (упрощённо)
while true; do
    # 1. Проверяем Supervisor
    if ! is_supervisor_alive; then
        restart_supervisor "process_not_found"
    fi
    
    # 2. Проверяем heartbeat
    if heartbeat_stale; then
        restart_supervisor "heartbeat_stale"
    fi
    
    # 3. Health checks (информационные)
    check_llama_health
    check_telegram_health
    
    sleep 30
done
```

#### Уровень 2: Внутренний Watchdog (в local_launcher.py)

- **Файл:** [`local_launcher.py`](local_launcher.py:227)
- **Тип:** Daemon thread (`_watchdog_thread`)
- **Цикл:** Проверка каждые 30 секунд
- **Ответственность:** Мониторинг direct-mode chat agent

```python
# Встроенный watchdog (упрощённо)
def _chat_watchdog_loop():
    while True:
        time.sleep(30)
        agent = _get_chat_agent()
        if agent._busy:
            idle_sec = now - agent._last_progress_ts
            
            # Мягкий таймаут — предупреждение
            if idle_sec >= SOFT_TIMEOUT_SEC:
                send_warning_to_owner()
            
            # Жёсткий таймаут — рестарт
            if idle_sec >= HARD_TIMEOUT_SEC:
                reset_chat_agent()
```

### 2.3 Конфигурационные переменные

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `WATCHDOG_CHECK_INTERVAL_SEC` | 30 | Интервал между проверками (сек) |
| `WATCHDOG_LLM_TIMEOUT_SEC` | 10 | Таймаут проверки Llama-Server (сек) |
| `WATCHDOG_TG_TIMEOUT_SEC` | 10 | Таймаут проверки Telegram API (сек) |
| `WATCHDOG_LOG_FILE` | `.ouroboros/logs/watchdog.jsonl` | Путь к JSONL логу |
| `WATCHDOG_MAX_RESTARTS_24H` | 20 | Макс рестартов за 24 часа |
| `WATCHDOG_GRACE_PERIOD_SEC` | 5 | Время ожидания перед рестартом |

### 2.4 systemd Unit файл

[`.ouroboros/watchdog.service`](.ouroboros/watchdog.service):

```ini
[Unit]
Description=Ouroboros Zera Supervisor Watchdog
After=network.target llama-server.service
Wants=llama-server.service

[Service]
Type=simple
User=zera
WorkingDirectory=/home/zera/ouroboros_zera
ExecStart=/bin/bash /home/zera/ouroboros_zera/.ouroboros/watchdog.sh
Restart=always
RestartSec=10

# Ресурсные лимиты
MemoryMax=256M
CPUQuota=25%

[Install]
WantedBy=multi-user.target
```

---

## 3. Как проверить состояние Watchdog

### 3.1 Проверка статуса systemd

```bash
# Статус watchdog сервиса
systemctl status ouroboros-watchdog

# Или если сервис называется иначе
systemctl list-units | grep watchdog
```

### 3.2 Проверка процесса Watchdog

```bash
# Проверка, что watchdog.sh запущен
pgrep -f "watchdog.sh"

# Проверка процесса Supervisor
pgrep -f "python.*local_launcher.py"
```

### 3.3 Проверка PID Supervisor

```bash
# Прочитать PID из файла
cat .ouroboros/supervisor.pid

# Проверить, что процесс жив
kill -0 $(cat .ouroboros/supervisor.pid) && echo "Alive" || echo "Dead"
```

### 3.4 Быстрая проверка здоровья

```bash
# Проверка Llama-Server
curl -s http://localhost:8080/health | jq

# Проверка Telegram API
curl -s "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getMe" | jq

# Проверка heartbeat
tail -1 .ouroboros/logs/supervisor.jsonl
```

### 3.5 Анализ логов в реальном времени

```bash
# Мониторинг логов watchdog
tail -f .ouroboros/logs/watchdog.jsonl

# Только health check события
grep '"type": "health_check"' .ouroboros/logs/watchdog.jsonl | tail -20

# Только события рестарта
grep '"type": "watchdog_restart"' .ouroboros/logs/watchdog.jsonl
```

---

## 4. Как читать логи

### 4.1 Формат JSONL лога

Основной лог: [`.ouroboros/logs/watchdog.jsonl`](.ouroboros/logs/watchdog.jsonl)

Каждая строка — JSON объект с полями:

```json
{"ts": "2026-04-28T07:34:28+00:00", "type": "health_check", "llama": "healthy"}
```

### 4.2 Типы событий

#### `watchdog_start` — Запуск Watchdog

```json
{
  "ts": "2026-04-28T07:39:28+00:00",
  "type": "watchdog_start",
  "check_interval": 30,
  "llm_timeout": 10,
  "tg_timeout": 10,
  "max_restarts": 20
}
```

| Поле | Описание |
|------|----------|
| `check_interval` | Интервал проверок в секундах |
| `llm_timeout` | Таймаут проверки LLM в секундах |
| `tg_timeout` | Таймаут проверки Telegram в секундах |
| `max_restarts` | Максимальное количество рестартов за 24 часа |

#### `health_check` — Результат проверки здоровья

```json
{"ts": "2026-04-28T07:34:28+00:00", "type": "health_check", "llama": "healthy"}
{"ts": "2026-04-28T07:34:28+00:00", "type": "health_check", "telegram": "unhealthy"}
```

Значения: `"healthy"` или `"unhealthy"`

#### `watchdog_restart` — Событие рестарта Supervisor

```json
{"ts": "...", "type": "watchdog_restart", "pid": null, "reason": "process_not_found"}
```

Возможные причины (`reason`):
- `"process_not_found"` — процесс Supervisor не найден
- `"heartbeat_stale_age=180s"` — heartbeat устарел (180 секунд)

#### `watchdog_restarted` — Успешный рестарт

```json
{"ts": "...", "type": "watchdog_restarted", "pid": 12345}
```

### 4.3 Текстовый лог

[`.ouroboros/logs/watchdog-stdout.log`](.ouroboros/logs/watchdog-stdout.log) — человекочитаемый формат:

```
[2026-04-28T07:39:28+00:00] [WATCHDOG] ==========================================
[2026-04-28T07:39:28+00:00] [WATCHDOG] Watchdog starting...
[2026-04-28T07:39:28+00:00] [WATCHDOG]   Check interval: 30s
[2026-04-28T07:39:28+00:00] [WATCHDOG]   LLM timeout: 10s
[2026-04-28T07:39:28+00:00] [WATCHDOG]   TG timeout: 10s
[2026-04-28T07:39:28+00:00] [WATCHDOG]   Max restarts/24h: 20
[2026-04-28T07:39:28+00:00] [WATCHDOG]   Grace period: 5s
[2026-04-28T07:39:28+00:00] [WATCHDOG] ==========================================
```

### 4.4 Полезные команды для анализа логов

```bash
# Последние 20 событий
tail -20 .ouroboros/logs/watchdog.jsonl | python3 -m json.tool --no-ensure-ascii

# Счётчик unhealthy событий
grep '"telegram": "unhealthy"' .ouroboros/logs/watchdog.jsonl | wc -l

# Все события рестарта
grep '"type": "watchdog_restart"' .ouroboros/logs/watchdog.jsonl

# Проверка, не достигнут ли лимит рестартов
wc -l .ouroboros/logs/restart_history.log
```

---

## 5. Что делать при проблемах

### 5.1 Supervisor не запускается

**Симптомы:**
- Watchdog постоянно перезапускает Supervisor
- В логах: `"ABORTED: Too many restarts"`

**Действия:**

```bash
# 1. Проверить логи Supervisor
tail -100 .ouroboros/logs/supervisor-output.log

# 2. Проверить .env файл
cat .ouroboros/.env | grep -v TOKEN  # скрыть токены

# 3. Проверить зависимости
python3 -c "import supervisor; print('OK')"

# 4. Проверить порт 8080 (занят?)
lsof -i :8080

# 5. Ручной запуск для отладки
cd /home/zera/ouroboros_zera && python3 local_launcher.py
```

### 5.2 Llama-Server не отвечает

**Симптомы:**
- В логах: `"WARNING: llama-server not responding on port 8080"`
- Задачи зависают на LLM-запросах

**Действия:**

```bash
# 1. Проверить, запущен ли llama-server
pgrep -f "llama-server" || echo "Not running"

# 2. Проверить порт
curl -v http://localhost:8080/health

# 3. Перезапустить llama-server
systemctl restart llama-server

# 4. Проверить логи llama-server
journalctl -u llama-server --since "10 minutes ago"
```

### 5.3 Telegram API не отвечает

**Симптомы:**
- В логах: `"WARNING: Telegram API not responding"`
- Нет уведомлений в Telegram

**Действия:**

```bash
# 1. Проверить токен
echo "${TELEGRAM_BOT_TOKEN:0:10}..."  # первые 10 символов

# 2. Проверить доступность API
curl -s "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getMe" | jq

# 3. Проверить сеть
ping api.telegram.org

# 4. Проверить .env
grep TELEGRAM_BOT_TOKEN .ouroboros/.env
```

### 5.4 Watchdog достиг лимита рестартов

**Симптомы:**
- В логах: `"ERROR: Max restarts (20/24h) reached! Stopping watchdog."`
- Watchdog останавливается

**Действия:**

```bash
# 1. Очистить историю рестартов (сбросить счётчик)
> .ouroboros/logs/restart_history.log

# 2. Перезапустить watchdog
systemctl restart ouroboros-watchdog

# 3. Или запустить вручную
cd /home/zera/ouroboros_zera && bash .ouroboros/watchdog.sh
```

### 5.5 Heartbeat устарел

**Симптомы:**
- В логах: `"WARNING: Heartbeat stale (180s old). Restarting..."`
- Supervisor жив, но не отвечает

**Действия:**

```bash
# 1. Проверить heartbeat
tail -5 .ouroboros/logs/supervisor.jsonl

# 2. Проверить, что Supervisor действительно работает
ps aux | grep local_launcher

# 3. Если Supervisor завис — watchdog его перезапустит автоматически
# 4. Если проблема повторяется — проверить ресурсы системы
free -h
df -h
top -bn1 | head -20
```

### 5.6 Watchdog не запущен

**Действия:**

```bash
# 1. Включить и запустить сервис
sudo systemctl enable ouroboros-watchdog
sudo systemctl start ouroboros-watchdog

# 2. Проверить статус
sudo systemctl status ouroboros-watchdog

# 3. Проверить journal
journalctl -u ouroboros-watchdog --since "5 minutes ago"
```

---

## 6. Взаимосвязь с Supervisor и Llama-Server

### 6.1 Иерархия зависимостей

```
┌─────────────────────────────────────────────────────┐
│                    Systemd                          │
│  llama-server.service (запускается первым)          │
│         │                                           │
│         ▼                                           │
│  ouroboros-watchdog.service (Wants=llama-server)    │
│         │                                           │
│         ▼                                           │
│  Watchdog (watchdog.sh)                             │
│         │                                           │
│         ▼                                           │
│  Supervisor (local_launcher.py)                     │
│         │                                           │
│         ▼                                           │
│  Llama-Server (localhost:8080)                      │
└─────────────────────────────────────────────────────┘
```

### 6.2 Цикл зависимостей

```
┌──────────┐     health check      ┌──────────────┐
│ Watchdog │ ─────────────────────► │ Llama-Server │
│          │ ◄───────────────────── │              │
│          │   /health endpoint    └──────────────┘
│          │
│          │  restart              ┌──────────────┐
│          ─────────────────────► │  Supervisor  │
│                                 │              │
│                                 │  worker      │
│                                 ──────────────► │ Llama-Server
│                                                 │              │
└─────────────────────────────────────────────────┴──────────────┘
```

**Ключевые моменты:**

1. **Watchdog мониторит Supervisor** — проверяет PID и heartbeat
2. **Supervisor использует Llama-Server** — workers отправляют запросы на `localhost:8080`
3. **Watchdog мониторит Llama-Server** — но только логирует (не перезапускает)
4. **Llama-Server управляется systemd** — отдельный сервис `llama-server.service`

### 6.3 Supervisor (local_launcher.py)

Supervisor — это основное приложение, которое:

- Управляет жизненным циклом worker-процессов (multiprocessing)
- Обрабатывает задачи из очереди (PENDING → RUNNING)
- Обеспечивает direct-mode чат с пользователем
- Запускает фоновое сознание (BackgroundConsciousness)

**Встроенный watchdog Supervisor:**

```python
# local_launcher.py:227-268
def _chat_watchdog_loop():
    """Monitor direct-mode chat agent for hangs."""
    while True:
        time.sleep(30)
        agent = _get_chat_agent()
        if agent._busy:
            idle_sec = now - agent._last_progress_ts
            
            # SOFT_TIMEOUT_SEC (600s) — предупреждение
            if idle_sec >= SOFT_TIMEOUT_SEC and not soft_warned:
                send_with_budget(chat_id, f"⏱️ Task running for {total_sec}s...")
                soft_warned = True
            
            # HARD_TIMEOUT_SEC (1800s) — рестарт
            if idle_sec >= HARD_TIMEOUT_SEC:
                reset_chat_agent()
```

### 6.4 Llama-Server

Llama-Server — это локальный LLM-сервер (скорее всего llama.cpp server):

- **Порт:** 8080
- **Health endpoint:** `GET /health`
- **Возвращает:** HTTP 200 если здоров
- **Управляется:** systemd (`llama-server.service`)

**Watchdog проверяет его здоровье, но не перезапускает:**

```bash
# watchdog.sh:66-79
check_llama_health() {
    http_code=$(curl -s -o /dev/null -w "%{http_code}" \
        --connect-timeout 10 --max-time 10 \
        http://localhost:8080/health 2>/dev/null)
    
    if [ "$http_code" = "200" ]; then
        return 0  # healthy
    else
        return 1  # unhealthy
    fi
}
```

### 6.5 Data Flow — как задачи проходят через систему

```
Пользователь (Telegram)
       │
       ▼
Telegram Bot API
       │
       ▼
Supervisor (local_launcher.py)
       │  getUpdates()
       ▼
Event Dispatcher (supervisor/events.py)
       │  maps events to handlers
       ▼
Task Queue (supervisor/queue.py)
       │  PENDING → RUNNING
       ▼
Worker Process (multiprocessing)
       │  fork/spawn
       ▼
Ouroboros Agent (ouroboros/agent.py)
       │  tool calls
       ▼
Llama-Server (localhost:8080)
       │  inference
       ▼
Response → Worker → Supervisor → Telegram
```

### 6.6 Heartbeat механизм

Supervisor пишет heartbeat в лог:

```json
{"ts": "2026-04-28T08:17:13+00:00", "type": "heartbeat"}
```

Watchdog проверяет свежесть heartbeat:

```bash
# Если heartbeat старше 3x check_interval (90s при default)
if [ "$age" -gt $((CHECK_INTERVAL * 3)) ]; then
    restart_supervisor "heartbeat_stale_age=${age}s"
fi
```

---

## Appendix A. Файловая структура

```
.ouroboros/
├── watchdog.sh              # Основной скрипт watchdog
├── watchdog.service         # systemd unit файл
├── run.sh                   # Скрипт запуска Supervisor
├── supervisor.pid           # PID текущего Supervisor
├── logs/
│   ├── watchdog.jsonl       # JSONL лог watchdog
│   ├── watchdog-stdout.log  # Текстовый лог stdout
│   ├── watchdog-stderr.log  # Текстовый лог stderr
│   ├── supervisor-output.log # Вывод Supervisor
│   ├── supervisor.jsonl     # Heartbeat Supervisor
│   └── restart_history.log  # История рестартов (timestamp per line)
└── .env                     # Конфигурация (токены и т.д.)
```

## Appendix B. Полезные команды

```bash
# Полный статус системы
echo "=== Watchdog ===" && systemctl status ouroboros-watchdog --no-pager
echo "=== Supervisor ===" && pgrep -f "local_launcher.py" || echo "Not running"
echo "=== Llama-Server ===" && pgrep -f "llama-server" || echo "Not running"
echo "=== Llama Health ===" && curl -s http://localhost:8080/health | jq

# Очистка логов (если слишком большие)
> .ouroboros/logs/watchdog.jsonl
> .ouroboros/logs/watchdog-stdout.log

# Мониторинг в реальном времени
tail -f .ouroboros/logs/watchdog.jsonl .ouroboros/logs/watchdog-stdout.log

# Проверка лимита рестартов
echo "Restarts in last 24h: $(wc -l < .ouroboros/logs/restart_history.log)"
```

## Appendix C. Troubleshooting Decision Tree

```
Система не работает?
│
├─ Watchdog запущен?
│  ├─ Нет → systemctl start ouroboros-watchdog
│  └─ Да ↓
│
├─ Supervisor запущен?
│  ├─ Нет → Проверить логи: tail .ouroboros/logs/supervisor-output.log
│  │        Проверить .env: cat .ouroboros/.env
│  │        Проверить зависимости: python3 -c "import supervisor"
│  └─ Да ↓
│
├─ Llama-Server отвечает?
│  ├─ Нет → systemctl restart llama-server
│  │        Проверить: curl http://localhost:8080/health
│  └─ Да ↓
│
├─ Telegram API отвечает?
│  ├─ Нет → Проверить токен, сеть
│  │        curl "https://api.telegram.org/bot${TOKEN}/getMe"
│  └─ Да ↓
│
└─ Всё работает → Проблема временная, мониторить логи
```

---

> **Последнее обновление:** 2026-04-28
> **Версия документации:** 1.0
> **Автор:** Сгенерировано для AI-агентов проекта Ouroboros Zera
