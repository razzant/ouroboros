# ============================
# Ouroboros — Runtime launcher (entry point, executed from repository)
# ============================
# Thin orchestrator: secrets, bootstrap, main loop.
# Heavy logic lives in supervisor/ package.

import logging
import os, sys, json, time, uuid, pathlib, subprocess, datetime, threading, queue as _queue_mod
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)

# ----------------------------
# 0) Install launcher deps
# ----------------------------
def install_launcher_deps() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "openai>=1.0.0", "requests"],
        check=True,
    )

install_launcher_deps()

def ensure_claude_code_cli() -> bool:
    """Best-effort install of Claude Code CLI for Anthropic-powered code edits."""
    local_bin = str(pathlib.Path.home() / ".local" / "bin")
    if local_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{local_bin}:{os.environ.get('PATH', '')}"

    has_cli = subprocess.run(["bash", "-lc", "command -v claude >/dev/null 2>&1"], check=False).returncode == 0
    if has_cli:
        return True

    subprocess.run(["bash", "-lc", "curl -fsSL https://claude.ai/install.sh | bash"], check=False)
    has_cli = subprocess.run(["bash", "-lc", "command -v claude >/dev/null 2>&1"], check=False).returncode == 0
    if has_cli:
        return True

    subprocess.run(["bash", "-lc", "command -v npm >/dev/null 2>&1 && npm install -g @anthropic-ai/claude-code"], check=False)
    has_cli = subprocess.run(["bash", "-lc", "command -v claude >/dev/null 2>&1"], check=False).returncode == 0
    return has_cli

# ----------------------------
# 0.1) provide apply_patch shim
# ----------------------------
from ouroboros.apply_patch import install as install_apply_patch
from ouroboros.llm import DEFAULT_LIGHT_MODEL
install_apply_patch()

# ----------------------------
# 1) Secrets + runtime config (ENVIRONMENT-BASED, NO COLAB API)
# ----------------------------
# Required secrets via environment variables
def _required_env(name: str) -> str:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        raise AssertionError(f"Missing required environment variable: {name}")
    return str(v).strip()

OPENROUTER_API_KEY = _required_env("OPENROUTER_API_KEY")
TELEGRAM_BOT_TOKEN = _required_env("TELEGRAM_BOT_TOKEN")
TOTAL_BUDGET_DEFAULT = _required_env("TOTAL_BUDGET")
GITHUB_TOKEN = _required_env("GITHUB_TOKEN")
GITHUB_USER = _required_env("GITHUB_USER")
GITHUB_REPO = _required_env("GITHUB_REPO")

# Robust TOTAL_BUDGET parsing — handles spaces, newlines, etc.
try:
    import re
    _raw_budget = str(TOTAL_BUDGET_DEFAULT or "")
    _clean_budget = re.sub(r'[^0-9.\-]', '', _raw_budget)
    TOTAL_BUDGET_LIMIT = float(_clean_budget) if _clean_budget else 0.0
    if _raw_budget.strip() != _clean_budget:
        log.warning(f"TOTAL_BUDGET cleaned: {_raw_budget!r} → {TOTAL_BUDGET_LIMIT}")
except Exception as e:
    log.warning(f"Failed to parse TOTAL_BUDGET ({TOTAL_BUDGET_DEFAULT!r}): {e}")
    TOTAL_BUDGET_LIMIT = 0.0

# Optional secrets
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Config (with defaults)
MAX_WORKERS = int(os.environ.get("OUROBOROS_MAX_WORKERS", "5"))
MODEL_MAIN = os.environ.get("OUROBOROS_MODEL", "anthropic/claude-sonnet-4.6")
MODEL_CODE = os.environ.get("OUROBOROS_MODEL_CODE", "anthropic/claude-sonnet-4.6")
MODEL_LIGHT = os.environ.get("OUROBOROS_MODEL_LIGHT", DEFAULT_LIGHT_MODEL)

BUDGET_REPORT_EVERY_MESSAGES = 10
SOFT_TIMEOUT_SEC = max(60, int(os.environ.get("OUROBOROS_SOFT_TIMEOUT_SEC", "600")))
HARD_TIMEOUT_SEC = max(120, int(os.environ.get("OUROBOROS_HARD_TIMEOUT_SEC", "1800")))
DIAG_HEARTBEAT_SEC = max(0, int(os.environ.get("OUROBOROS_DIAG_HEARTBEAT_SEC", "30")))
DIAG_SLOW_CYCLE_SEC = max(0, int(os.environ.get("OUROBOROS_DIAG_SLOW_CYCLE_SEC", "20")))

# Push to process env for children
os.environ["OPENROUTER_API_KEY"] = str(OPENROUTER_API_KEY)
os.environ["OPENAI_API_KEY"] = str(OPENAI_API_KEY or "")
os.environ["ANTHROPIC_API_KEY"] = str(ANTHROPIC_API_KEY or "")
os.environ["GITHUB_USER"] = str(GITHUB_USER)
os.environ["GITHUB_REPO"] = str(GITHUB_REPO)
os.environ["OUROBOROS_MODEL"] = str(MODEL_MAIN or "anthropic/claude-sonnet-4.6")
os.environ["OUROBOROS_MODEL_CODE"] = str(MODEL_CODE or "anthropic/claude-sonnet-4.6")
if MODEL_LIGHT:
    os.environ["OUROBOROS_MODEL_LIGHT"] = str(MODEL_LIGHT)
os.environ["OUROBOROS_DIAG_HEARTBEAT_SEC"] = str(DIAG_HEARTBEAT_SEC)
os.environ["OUROBOROS_DIAG_SLOW_CYCLE_SEC"] = str(DIAG_SLOW_CYCLE_SEC)
os.environ["TELEGRAM_BOT_TOKEN"] = str(TELEGRAM_BOT_TOKEN)

if str(ANTHROPIC_API_KEY or "").strip():
    ensure_claude_code_cli()

# ----------------------------
# 2) Define Drive and Repo paths (local filesystem - not Colab)
# ----------------------------
# Use environment variables to override locations for local deployment
DRIVE_ROOT = pathlib.Path(os.environ.get("DRIVE_ROOT", "~/.ouroboros")).expanduser().resolve()
REPO_DIR = pathlib.Path(os.environ.get("REPO_DIR", DRIVE_ROOT / "repo")).resolve()

# Ensure required subdirectories exist
for sub in ["state", "logs", "memory", "index", "locks", "archive"]:
    (DRIVE_ROOT / sub).mkdir(parents=True, exist_ok=True)
REPO_DIR.mkdir(parents=True, exist_ok=True)

# Clear stale owner mailbox files from previous session
try:
    from ouroboros.owner_inject import get_pending_path
    # Clean legacy global file
    _stale_inject = get_pending_path(DRIVE_ROOT)
    if _stale_inject.exists():
        _stale_inject.unlink(missing_ok=True)
    # Clean per-task mailbox dir
    _mailbox_dir = DRIVE_ROOT / "memory" / "owner_mailbox"
    if _mailbox_dir.exists():
        for _f in _mailbox_dir.iterdir():
            _f.unlink(missing_ok=True)
except Exception:
    pass

CHAT_LOG_PATH = DRIVE_ROOT / "logs" / "chat.jsonl"
if not CHAT_LOG_PATH.exists():
    CHAT_LOG_PATH.write_text("", encoding="utf-8")

# ----------------------------
# 3) Git constants
# ----------------------------
BRANCH_DEV = "ouroboros"
BRANCH_STABLE = "ouroboros-stable"
REMOTE_URL = f"https://{GITHUB_TOKEN}:x-oauth-basic@github.com/{GITHUB_USER}/{GITHUB_REPO}.git"

# ----------------------------
# 4) Initialize supervisor modules
# ----------------------------
from supervisor.state import (
    init as state_init, load_state, save_state, append_jsonl,
    update_budget_from_usage, status_text, rotate_chat_log_if_needed,
    init_state,
)
state_init(DRIVE_ROOT, TOTAL_BUDGET_LIMIT)
init_state()

from supervisor.telegram import (
    init as telegram_init, TelegramClient, send_with_budget, log_chat,
)
TG = TelegramClient(str(TELEGRAM_BOT_TOKEN))
telegram_init(
    drive_root=DRIVE_ROOT,
    total_budget_limit=TOTAL_BUDGET_LIMIT,
    budget_report_every=BUDGET_REPORT_EVERY_MESSAGES,
    tg_client=TG,
)

from supervisor.git_ops import (
    init as git_ops_init, ensure_repo_present, checkout_and_reset,
    sync_runtime_dependencies, import_test, safe_restart,
)
git_ops_init(
    repo_dir=REPO_DIR, drive_root=DRIVE_ROOT, remote_url=REMOTE_URL,
    branch_dev=BRANCH_DEV, branch_stable=BRANCH_STABLE,
)

from supervisor.queue import (
    enqueue_task, enforce_task_timeouts, enqueue_evolution_task_if_needed,
    persist_queue_snapshot, restore_pending_from_snapshot,
    cancel_task_by_id, queue_review_task, sort_pending,
)

from supervisor.workers import (
    init as workers_init, get_event_q, WORKERS, PENDING, RUNNING,
    spawn_workers, kill_workers, assign_tasks, ensure_workers_healthy,
    handle_chat_direct, _get_chat_agent, auto_resume_after_restart,
)
workers_init(
    repo_dir=REPO_DIR, drive_root=DRIVE_ROOT, max_workers=MAX_WORKERS,
    soft_timeout=SOFT_TIMEOUT_SEC, hard_timeout=HARD_TIMEOUT_SEC,
    total_budget_limit=TOTAL_BUDGET_LIMIT,
    branch_dev=BRANCH_DEV, branch_stable=BRANCH_STABLE,
)

from supervisor.events import dispatch_event

# ----------------------------
# 5) Bootstrap repo
# ----------------------------
ensure_repo_present()
ok, msg = safe_restart(reason="bootstrap", unsynced_policy="rescue_and_reset")
assert ok, f"Bootstrap failed: {msg}"

# ----------------------------
# 6) Start workers
# ----------------------------
kill_workers()
spawn_workers(MAX_WORKERS)
restored_pending = restore_pending_from_snapshot()
persist_queue_snapshot(reason="startup")
if restored_pending > 0:
    st_boot = load_state()
    if st_boot.get("owner_chat_id"):
        send_with_budget(int(st_boot["owner_chat_id"]),
                         f"♻️ Restored pending queue from snapshot: {restored_pending} tasks.")

append_jsonl(DRIVE_ROOT / "logs" / "supervisor.jsonl", {
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "type": "launcher_start",
    "branch": load_state().get("current_branch"),
    "sha": load_state().get("current_sha"),
    "max_workers": MAX_WORKERS,
    "model_default": MODEL_MAIN, "model_code": MODEL_CODE, "model_light": MODEL_LIGHT,
    "soft_timeout_sec": SOFT_TIMEOUT_SEC, "hard_timeout_sec": HARD_TIMEOUT_SEC,
    "worker_start_method": str(os.environ.get("OUROBOROS_WORKER_START_METHOD") or ""),
    "diag_heartbeat_sec": DIAG_HEARTBEAT_SEC,
    "diag_slow_cycle_sec": DIAG_SLOW_CYCLE_SEC,
})

# ----------------------------
# 6.1) Auto-resume after restart
# ----------------------------
auto_resume_after_restart()

# ----------------------------
# 6.2) Direct-mode watchdog
# ----------------------------
def _chat_watchdog_loop():
    """Monitor direct-mode chat agent for hangs. Runs as daemon thread."""
    soft_warned = False
    while True:
        time.sleep(30)
        try:
            agent = _get_chat_agent()
            if not agent._busy:
                soft_warned = False
                continue

            now = time.time()
            idle_sec = now - agent._last_progress_ts
            total_sec = now - agent._task_started_ts

            if idle_sec >= HARD_TIMEOUT_SEC:
                st = load_state()
                if st.get("owner_chat_id"):
                    send_with_budget(
                        int(st["owner_chat_id"]),
                        f"⚠️ Task stuck ({int(total_sec)}s without progress). "
                        f"Restarting agent.",
                    )
                reset_chat_agent()
                soft_warned = False
                continue

            if idle_sec >= SOFT_TIMEOUT_SEC and not soft_warned:
                soft_warned = True
                st = load_state()
                if st.get("owner_chat_id"):
                    send_with_budget(
                        int(st["owner_chat_id"]),
                        f"⏱️ Task running for {int(total_sec)}s, "
                        f"last progress {int(idle_sec)}s ago. Continuing.",
                    )
        except Exception:
            log.debug("Failed to check/notify chat watchdog", exc_info=True)
            pass

_watchdog_thread = threading.Thread(target=_chat_watchdog_loop, daemon=True)
_watchdog_thread.start()

# ----------------------------
# 6.3) Background consciousness
# ----------------------------
from ouroboros.consciousness import BackgroundConsciousness

def _get_owner_chat_id() -> Optional[int]:
    try:
        st = load_state()
        cid = st.get("owner_chat_id")
        return int(cid) if cid else None
    except Exception:
        return None

_consciousness = BackgroundConsciousness(
    drive_root=DRIVE_ROOT,
    repo_dir=REPO_DIR,
    event_queue=get_event_q(),
    owner_chat_id_fn=_get_owner_chat_id,
)

def reset_chat_agent():
    """Reset the direct-mode chat agent (called by watchdog on hangs)."""
    import supervisor.workers as _w
    _w._chat_agent = None

# ----------------------------
# 7) Main loop
# ----------------------------
import types
_event_ctx = types.SimpleNamespace(
    DRIVE_ROOT=DRIVE_ROOT,
    REPO_DIR=REPO_DIR,
    BRANCH_DEV=BRANCH_DEV,
    BRANCH_STABLE=BRANCH_STABLE,
    TG=TG,
    WORKERS=WORKERS,
    PENDING=PENDING,
    RUNNING=RUNNING,
    MAX_WORKERS=MAX_WORKERS,
    send_with_budget=send_with_budget,
    load_state=load_state,
    save_state=save_state,
    update_budget_from_usage=update_budget_from_usage,
    append_jsonl=append_jsonl,
    enqueue_task=enqueue_task,
    cancel_task_by_id=cancel_task_by_id,
    queue_review_task=queue_review_task,
    persist_queue_snapshot=persist_queue_snapshot,
    safe_restart=safe_restart,
    kill_workers=kill_workers,
    spawn_workers=spawn_workers,
    sort_pending=sort_pending,
    consciousness=_consciousness,
)


def _safe_qsize(q: Any) -> int:
    try:
        return int(q.qsize())
    except Exception:
        return -1


def _handle_supervisor_command(text: str, chat_id: int, tg_offset: int = 0):
    """Handle supervisor slash-commands.

    Returns:
        True  — terminal command fully handled (caller should `continue`)
        str   — dual-path note to prepend (caller falls through to LLM)
        ""    — not a recognized command (falsy, caller falls through)
    """
    lowered = text.strip().lower()

    if lowered.startswith("/panic"):
        send_with_budget(chat_id, "🛑 PANIC: stopping everything now.")
        kill_workers()
        st2 = load_state()
        st2["tg_offset"] = tg_offset
        save_state(st2)
        raise SystemExit("PANIC")

    if lowered.startswith("/restart"):
        send_with_budget(chat_id, "♻️ Restarting...")
        safe_restart(reason="manual /restart", unsynced_policy="rescue_and_reset")
        raise SystemExit("RESTART")

    if lowered == "/status" or lowered.startswith("/status "):
        st = load_state()
        txt = status_text(st, pending=PENDING, running=RUNNING, max_workers=MAX_WORKERS)
        send_with_budget(chat_id, txt)
        return True

    if lowered.startswith("/evolve"):
        parts = lowered.split()
        st = load_state()
        if len(parts) > 1 and parts[1] in ("off", "stop", "pause"):
            if st.get("evolution_mode_enabled"):
                st["evolution_mode_enabled"] = False
                save_state(st)
                send_with_budget(chat_id, "🧬 Evolution paused.")
            else:
                send_with_budget(chat_id, "🧬 Evolution already paused.")
        elif len(parts) > 1 and parts[1] in ("on", "start", "resume"):
            if st.get("evolution_mode_enabled"):
                send_with_budget(chat_id, "🧬 Evolution already running.")
            else:
                st["evolution_mode_enabled"] = True
                save_state(st)
                send_with_budget(chat_id, "🧬 Evolution enabled.")
        else:
            cur = "ON" if st.get("evolution_mode_enabled") else "OFF"
            send_with_budget(chat_id, f"🧬 Evolution: {cur}")
        return True

    if lowered.startswith("/bg"):
        parts = lowered.split()
        st = load_state()
        if len(parts) > 1 and parts[1] in ("off", "stop"):
            if st.get("background_consciousness_enabled"):
                st["background_consciousness_enabled"] = False
                save_state(st)
                _consciousness.stop()
                send_with_budget(chat_id, "🧠 Background consciousness stopped.")
            else:
                send_with_budget(chat_id, "🧠 Background consciousness already stopped.")
        elif len(parts) > 1 and parts[1] in ("on", "start"):
            if st.get("background_consciousness_enabled"):
                send_with_budget(chat_id, "🧠 Background consciousness already running.")
            else:
                st["background_consciousness_enabled"] = True
                save_state(st)
                _consciousness.start()
                send_with_budget(chat_id, "🧠 Background consciousness started.")
        else:
            cur = "ON" if st.get("background_consciousness_enabled", True) else "OFF"
            send_with_budget(chat_id, f"🧠 Background consciousness: {cur}")
        return True

    if lowered.startswith("/review"):
        parts = lowered.split(maxsplit=1)
        reason = parts[1].strip() if len(parts) > 1 else "owner request"
        tid = queue_review_task(reason=reason, force=False)
        if tid:
            send_with_budget(chat_id, f"🔎 Review queued: {tid}")
        else:
            send_with_budget(chat_id, "🔎 Review already queued.")
        return True

    return ""


def _main_loop():
    """Supervisor main loop: poll Telegram, dispatch, heartbeat."""
    import json
    while True:
        try:
            st = load_state()
            owner_id = int(st.get("owner_id") or 0)
            owner_chat_id = int(st.get("owner_chat_id") or 0)
            tg_offset = int(st.get("tg_offset") or 0)

            updates = TG.get_updates(offset=tg_offset, timeout=30)
            if updates:
                for upd in updates:
                    tg_offset = max(tg_offset, upd.get("update_id", 0) + 1)
                    msg = upd.get("message") or upd.get("channel_post") or {}
                    chat_id = msg.get("chat", {}).get("id")
                    text = msg.get("text", "").strip()
                    if not chat_id or not text:
                        continue

                    # Log all incoming creator messages
                    if owner_chat_id and chat_id == owner_chat_id:
                        log_chat(chat_id, text, from_owner=True)
                    else:
                        log_chat(chat_id, text, from_owner=False)

                    if chat_id == owner_chat_id:
                        # First message from creator becomes owner if not set
                        if not owner_id:
                            st["owner_id"] = chat_id
                            st["owner_chat_id"] = chat_id
                            send_with_budget(chat_id, f"✅ Owner registered. Ouroboros online.")
                            save_state(st)

                        # Supervisor commands
                        cmd_res = _handle_supervisor_command(text, chat_id, tg_offset=tg_offset)
                        if cmd_res is True:
                            continue  # command fully handled, skip LLM
                        if cmd_res:
                            text = cmd_res + "\n" + text  # prepend note, still go through LLM

                        # All creator messages are processed through LLM (dual-path)
                        enqueue_task({
                            "id": uuid.uuid4().hex[:8],
                            "type": "task",
                            "chat_id": int(chat_id),
                            "text": text,
                        })
                    else:
                        # Ignore non-owner chats
                        send_with_budget(chat_id, "⛔ Not authorized.")
                # Save new offset
                st["tg_offset"] = tg_offset
                save_state(st)

            enforce_task_timeouts()
            enqueue_evolution_task_if_needed()
            time.sleep(1)

        except KeyboardInterrupt:
            log.info("Shutting down (KeyboardInterrupt)")
            break
        except Exception as e:
            log.error("Main loop error", exc_info=True)
            time.sleep(5)


if __name__ == "__main__":
    _main_loop()
