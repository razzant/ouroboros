#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ouroboros Zera — Local launcher (CLI + Telegram entry point).

Pure local execution — no cloud/Colab dependencies.
Usage:
    cd /home/zera/ouroboros_zera
    python3 local_launcher.py

Architecture:
    - Telegram polling loop for owner chat
    - Multiprocessing worker lifecycle management
    - Event-driven architecture via supervisor package
    - Background consciousness (optional)
"""

import logging
import os
import pathlib
import sys
import uuid
import datetime
import time
import threading
import types
import queue as _queue_mod
from typing import Any, Dict, List, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR
DRIVE_ROOT = SCRIPT_DIR / ".ouroboros"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

def setup_env() -> None:
    """Ensure .ouroboros directory structure exists."""
    for sub in ["state", "logs", "memory", "scratchpad", "locks"]:
        (DRIVE_ROOT / sub).mkdir(parents=True, exist_ok=True)
    log.info("Environment ready: repo=%s, drive=%s", REPO_DIR, DRIVE_ROOT)


# ---------------------------------------------------------------------------
# Secrets + runtime config (local-first, no Colab)
# ---------------------------------------------------------------------------

def _get_secret(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """Get secret from environment variables (local-first)."""
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        v = default
    if required:
        assert v is not None and str(v).strip() != "", f"Missing required secret: {name}"
    return v

def _parse_int_cfg(raw: Optional[str], default: int, minimum: int = 0) -> int:
    """Parse integer config with fallback."""
    try:
        val = int(str(raw))
    except Exception:
        val = default
    return max(minimum, val)

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load variables from .env into os.environ
    log.info(".env file loaded successfully")
except ImportError:
    log.warning("python-dotenv not installed — using system environment variables only")
except Exception as e:
    log.warning(f"Failed to load .env file: {e}")

# Load secrets from environment
OPENROUTER_API_KEY = _get_secret("OPENROUTER_API_KEY", required=True)
TELEGRAM_BOT_TOKEN = _get_secret("TELEGRAM_BOT_TOKEN", required=True)
TOTAL_BUDGET_RAW = _get_secret("TOTAL_BUDGET", default="0")

# Robust TOTAL_BUDGET parsing
try:
    import re
    _raw_budget = str(TOTAL_BUDGET_RAW or "")
    _clean_budget = re.sub(r'[^0-9.\-]', '', _raw_budget)
    TOTAL_BUDGET_LIMIT = float(_clean_budget) if _clean_budget else 0.0
except Exception as e:
    log.warning(f"Failed to parse TOTAL_BUDGET ({TOTAL_BUDGET_DEFAULT!r}): {e}")
    TOTAL_BUDGET_LIMIT = 0.0

OPENAI_API_KEY = _get_secret("OPENROUTER_API_KEY", default="")
ANTHROPIC_API_KEY = _get_secret("ANTHROPIC_API_KEY", default="")
GITHUB_TOKEN = _get_secret("GITHUB_TOKEN", default="")
GITHUB_USER = os.environ.get("GITHUB_USER", "")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "")

# Config with defaults
MAX_WORKERS = _parse_int_cfg(os.environ.get("OUROBOROS_MAX_WORKERS"), 5, minimum=1)
MODEL_MAIN = os.environ.get("OUROBOROS_MODEL", "anthropic/claude-sonnet-4.6")
MODEL_CODE = os.environ.get("OUROBOROS_MODEL_CODE", "anthropic/claude-sonnet-4.6")
MODEL_LIGHT = os.environ.get("OUROBOROS_MODEL_LIGHT", "gpt-4o-mini")

# Timeouts
SOFT_TIMEOUT_SEC = _parse_int_cfg(os.environ.get("OUROBOROS_SOFT_TIMEOUT"), 600, minimum=60)
HARD_TIMEOUT_SEC = _parse_int_cfg(os.environ.get("OUROBOROS_HARD_TIMEOUT"), 1800, minimum=300)
DIAG_HEARTBEAT_SEC = _parse_int_cfg(os.environ.get("OUROBOROS_DIAG_HEARTBEAT"), 30, minimum=10)
DIAG_SLOW_CYCLE_SEC = _parse_int_cfg(os.environ.get("OUROBOROS_DIAG_SLOW_CYCLE"), 300, minimum=0)

# Branches
BRANCH_DEV = os.environ.get("OUROBOROS_BRANCH_DEV", "ouroboros")
BRANCH_STABLE = os.environ.get("OUROBOROS_BRANCH_STABLE", "ouroboros-stable")


# ---------------------------------------------------------------------------
# Agent initialization
# ---------------------------------------------------------------------------

def create_agent():
    """Create and return an OuroborosAgent instance."""
    from ouroboros.agent.agent import Env, OuroborosAgent

    env = Env(
        repo_dir=REPO_DIR,
        drive_root=DRIVE_ROOT,
        branch_dev="ouroboros",
    )
    agent = OuroborosAgent(env=env)
    return agent


# ---------------------------------------------------------------------------
# CLI loop (local interactive mode)
# ---------------------------------------------------------------------------

def cli_loop(agent) -> None:
    """Interactive CLI loop for local execution."""
    print("=" * 60)
    print("  Ouroboros Zera — Local CLI")
    print("  Type your message and press Enter.")
    print("  Commands: 'quit', 'exit', 'help'")
    print("=" * 60)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not user_input:
            continue

        cmd = user_input.lower().strip()
        if cmd in ("quit", "exit"):
            print("Exiting...")
            break
        if cmd == "help":
            print("Commands: quit, exit, help")
            print("Type any message to start a task.")
            continue

        # Build task dict
        task = {
            "id": str(uuid.uuid4())[:8],
            "type": "cli",
            "chat_id": 0,
            "depth": 0,
            "text": user_input,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "_is_direct_chat": True,
        }

        print("\nProcessing...")
        try:
            result = agent.handle_task(task)
            # Print events (should contain the response)
            for event in result:
                if event.get("type") == "send_message":
                    msg = event.get("text", "")
                    if msg:
                        clean = msg.replace("💬 ", "").replace("✅ ", "").replace("⚠️ ", "")
                        print(f"\n{clean}\n")
                elif event.get("type") == "final_response":
                    text = event.get("text", "")
                    if text:
                        print(f"\n{text}\n")
        except Exception as e:
            log.error("Task failed", exc_info=True)
            print(f"\n⚠️ Error: {type(e).__name__}: {e}\n")


# ---------------------------------------------------------------------------
# Supervisor initialization (for Telegram mode)
# ---------------------------------------------------------------------------

def _init_supervisor():
    """Initialize supervisor package modules for Telegram polling mode."""
    from supervisor import state as state_mod
    from supervisor import telegram as telegram_mod
    from supervisor import workers as workers_mod
    from supervisor import queue as queue_mod

    # Initialize state module
    state_mod.init(DRIVE_ROOT, TOTAL_BUDGET_LIMIT)

    # Initialize Telegram client
    TG = telegram_mod.TelegramClient(TELEGRAM_BOT_TOKEN)
    telegram_mod.init(DRIVE_ROOT, TOTAL_BUDGET_LIMIT, 10, TG)

    # Initialize workers module
    workers_mod.init(
        repo_dir=REPO_DIR,
        drive_root=DRIVE_ROOT,
        max_workers=MAX_WORKERS,
        soft_timeout=SOFT_TIMEOUT_SEC,
        hard_timeout=HARD_TIMEOUT_SEC,
        total_budget_limit=TOTAL_BUDGET_LIMIT,
        branch_dev=BRANCH_DEV,
        branch_stable=BRANCH_STABLE,
    )

    return TG


# ---------------------------------------------------------------------------
# Supervisor command handler
# ---------------------------------------------------------------------------

def _handle_supervisor_command(text: str, chat_id: int, tg_offset: int = 0):
    """Handle supervisor slash-commands.

    Returns:
        True  — terminal command fully handled (caller should continue)
        str   — dual-path note to prepend (caller falls through to LLM)
        ""    — not a recognized command (falsy, caller falls through)
    """
    lowered = text.strip().lower()

    # Import needed functions
    from supervisor.state import load_state, save_state
    from supervisor.telegram import send_with_budget
    from supervisor.workers import spawn_workers, kill_workers, get_event_q
    from supervisor.events import dispatch_event
    import supervisor.workers as workers_mod
    import types

    # Build minimal event context
    _event_ctx = types.SimpleNamespace(
        DRIVE_ROOT=DRIVE_ROOT,
        REPO_DIR=REPO_DIR,
        BRANCH_DEV=BRANCH_DEV,
        BRANCH_STABLE=BRANCH_STABLE,
        WORKERS=workers_mod.WORKERS,
        PENDING=workers_mod.PENDING,
        RUNNING=workers_mod.RUNNING,
        MAX_WORKERS=MAX_WORKERS,
        send_with_budget=send_with_budget,
        load_state=load_state,
        save_state=save_state,
        update_budget_from_usage=lambda usage: None,
        append_jsonl=lambda path, evt: None,
        enqueue_task=lambda *a, **k: None,
        cancel_task_by_id=lambda *a, **k: None,
        queue_review_task=lambda *a, **k: None,
        persist_queue_snapshot=lambda *a, **k: None,
        safe_restart=lambda *a, **k: None,
        kill_workers=kill_workers,
        spawn_workers=spawn_workers,
        sort_pending=lambda *a, **k: None,
        consciousness=None,  # Will be set by main loop
    )

    if lowered.startswith("/start"):
        send_with_budget(chat_id,
                         "🐛 Ouroboros Zera online!\n\n"
                         "Available commands:\n"
                         "/status — Worker and queue status\n"
                         "/spawn <n> — Spawn n workers\n"
                         "/kill — Kill all workers\n"
                         "/evolve — Trigger evolution\n"
                         "/consciousness — Toggle consciousness\n"
                         "/help — Show this help")
        return True

    if lowered.startswith("/status"):
        st = load_state()
        pending_count = len(workers_mod.PENDING)
        running_count = len(workers_mod.RUNNING)
        workers_active = sum(1 for w in workers_mod.WORKERS.values() if w.proc.is_alive())
        workers_total = len(workers_mod.WORKERS)
        msg = (f"📊 Status:\n"
               f"Workers: {workers_active}/{workers_total} active\n"
               f"Pending: {pending_count}\n"
               f"Running: {running_count}\n"
               f"Branch: {st.get('current_branch', 'unknown')}\n"
               f"SHA: {st.get('current_sha', 'unknown')[:7]}")
        send_with_budget(chat_id, msg)
        return True

    if lowered.startswith("/spawn"):
        try:
            n = int(text.split()[1]) if len(text.split()) > 1 else 1
        except (IndexError, ValueError):
            n = 1
        spawn_workers(n)
        send_with_budget(chat_id, f"🔧 Spawned {n} workers.")
        return True

    if lowered.startswith("/kill"):
        kill_workers()
        send_with_budget(chat_id, "🔪 All workers killed.")
        return True

    if lowered.startswith("/evolve"):
        event_q = get_event_q()
        event_q.put({
            "type": "schedule_task",
            "task_id": uuid.uuid4().hex[:8],
            "description": "Evolution cycle: analyze current state and propose improvements",
            "priority": 10,
            "source": "owner",
        })
        send_with_budget(chat_id, "🧬 Evolution task scheduled.")
        return True

    if lowered.startswith("/consciousness"):
        # Toggle consciousness via state
        from supervisor.state import load_state, save_state
        st = load_state()
        current = st.get("consciousness_enabled", True)
        st["consciousness_enabled"] = not current
        save_state(st)
        send_with_budget(chat_id,
                         f"🧠 Consciousness {'enabled' if not current else 'disabled'}.")
        return True

    if lowered.startswith("/help"):
        send_with_budget(chat_id,
                         "📚 Available commands:\n"
                         "/start — Welcome message\n"
                         "/status — System status\n"
                         "/spawn [n] — Spawn workers\n"
                         "/kill — Kill all workers\n"
                         "/evolve — Trigger evolution\n"
                         "/consciousness — Toggle consciousness\n"
                         "/help — Show this help")
        return True

    # Not a recognized command — fall through to LLM
    return ""


# ---------------------------------------------------------------------------
# Telegram polling loop
# ---------------------------------------------------------------------------

def telegram_polling_loop():
    """Main loop for Telegram polling with worker management."""
    # Initialize supervisor
    TG = _init_supervisor()

    # Import after initialization
    from supervisor.state import load_state, save_state, append_jsonl
    from supervisor.workers import (
        get_event_q, ensure_workers_healthy, spawn_workers, kill_workers,
        assign_tasks, auto_resume_after_restart, _get_chat_agent,
        handle_chat_direct, _CTX, _LAST_SPAWN_TIME,
    )
    import supervisor.workers as workers_mod  # for direct access to WORKERS/PENDING/RUNNING
    from supervisor.events import dispatch_event
    from supervisor.telegram import send_with_budget, log_chat
    from ouroboros.consciousness import BackgroundConsciousness

    # State
    offset = 0
    _last_message_ts = time.time()
    _ACTIVE_MODE_SEC = 300  # 5 min of activity = active polling mode

    # Load state
    st = load_state()
    saved_offset = st.get("tg_offset", 0)
    if saved_offset:
        offset = saved_offset
        log.info("Restored Telegram offset: %d", offset)

    # Initialize workers
    log.info("Spawning %d workers...", MAX_WORKERS)
    kill_workers()
    spawn_workers(MAX_WORKERS)
    auto_resume_after_restart()

    # Background consciousness
    def _get_owner_chat_id():
        try:
            s = load_state()
            cid = s.get("owner_chat_id")
            return int(cid) if cid else None
        except Exception:
            return None

    _consciousness = BackgroundConsciousness(
        drive_root=DRIVE_ROOT,
        repo_dir=REPO_DIR,
        event_queue=get_event_q(),
        owner_chat_id_fn=_get_owner_chat_id,
    )
    try:
        _consciousness.start()
        log.info("Background consciousness auto-started (default: always on)")
    except Exception as e:
        log.warning("consciousness auto-start failed: %s", e)

    # Event context for dispatch
    _event_ctx = types.SimpleNamespace(
        DRIVE_ROOT=DRIVE_ROOT,
        REPO_DIR=REPO_DIR,
        BRANCH_DEV=BRANCH_DEV,
        BRANCH_STABLE=BRANCH_STABLE,
        TG=TG,
        WORKERS=workers_mod.WORKERS,
        PENDING=workers_mod.PENDING,
        RUNNING=workers_mod.RUNNING,
        MAX_WORKERS=MAX_WORKERS,
        send_with_budget=send_with_budget,
        load_state=load_state,
        save_state=save_state,
        update_budget_from_usage=lambda usage: None,
        append_jsonl=lambda path, evt: append_jsonl(path, evt),
        enqueue_task=lambda *a, **k: None,
        cancel_task_by_id=lambda *a, **k: None,
        queue_review_task=lambda *a, **k: None,
        persist_queue_snapshot=lambda *a, **k: None,
        safe_restart=lambda *a, **k: None,
        kill_workers=kill_workers,
        spawn_workers=spawn_workers,
        sort_pending=lambda *a, **k: None,
        consciousness=_consciousness,
    )

    log.info("Telegram polling started. Waiting for messages...")

    # ------------------------------------------------------------------
    # Main polling loop
    # ------------------------------------------------------------------
    while True:
        loop_started_ts = time.time()

        # Ensure workers healthy
        try:
            ensure_workers_healthy()
        except Exception as e:
            log.warning("ensure_workers_healthy failed: %s", e)

        # Drain worker events
        event_q = get_event_q()
        while True:
            try:
                evt = event_q.get_nowait()
            except _queue_mod.Empty:
                break
            try:
                dispatch_event(evt, _event_ctx)
            except Exception as e:
                log.warning("dispatch_event failed: %s", e, exc_info=True)

        # Assign pending tasks
        try:
            assign_tasks()
        except Exception as e:
            log.warning("assign_tasks failed: %s", e)

        # Persist queue snapshot
        try:
            from supervisor.queue import persist_snapshot
            persist_snapshot(reason="main_loop")
        except Exception as e:
            log.warning("persist_snapshot failed: %s", e)

        # Poll Telegram — adaptive: fast when active, long-poll when idle
        _now = time.time()
        _active = (_now - _last_message_ts) < _ACTIVE_MODE_SEC
        _poll_timeout = 0 if _active else 10
        try:
            updates = TG.get_updates(offset=offset, timeout=_poll_timeout)
        except Exception as e:
            append_jsonl(
                DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "type": "telegram_poll_error",
                    "offset": offset,
                    "error": repr(e),
                },
            )
            time.sleep(1.5)
            continue

        for upd in updates:
            offset = int(upd["update_id"]) + 1
            msg = upd.get("message") or upd.get("edited_message") or {}
            if not msg:
                continue

            chat_id = int(msg["chat"]["id"])
            from_user = msg.get("from") or {}
            user_id = int(from_user.get("id") or 0)
            text = str(msg.get("text") or "")
            caption = str(msg.get("caption") or "")
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

            # Extract image if present
            image_data = None
            if msg.get("photo"):
                best_photo = msg["photo"][-1]
                file_id = best_photo.get("file_id")
                if file_id:
                    b64, mime = TG.download_file_base64(file_id)
                    if b64:
                        image_data = (b64, mime, caption)
            elif msg.get("document"):
                doc = msg["document"]
                mime_type = str(doc.get("mime_type") or "")
                if mime_type.startswith("image/"):
                    file_id = doc.get("file_id")
                    if file_id:
                        b64, mime = TG.download_file_base64(file_id)
                        if b64:
                            image_data = (b64, mime, caption)

            # Load state for owner check
            st = load_state()
            if st.get("owner_id") is None:
                # Register first message sender as owner
                st["owner_id"] = user_id
                st["owner_chat_id"] = chat_id
                st["last_owner_message_at"] = now_iso
                save_state(st)
                log_chat("in", chat_id, user_id, text)
                send_with_budget(chat_id, "✅ Owner registered. Ouroboros online.")
                continue

            # Ignore non-owner messages
            if user_id != int(st.get("owner_id")):
                continue

            log_chat("in", chat_id, user_id, text)
            st["last_owner_message_at"] = now_iso
            _last_message_ts = time.time()
            save_state(st)

            # --- Supervisor commands ---
            if text.strip().lower().startswith("/"):
                try:
                    result = _handle_supervisor_command(text, chat_id, tg_offset=offset)
                    if result is True:
                        continue  # terminal command, fully handled
                    elif result:  # non-empty string = dual-path note
                        text = result + text  # prepend note, fall through to LLM
                except SystemExit:
                    raise
                except Exception:
                    log.warning("Supervisor command handler error", exc_info=True)

            # All other messages (and dual-path commands) → direct chat
            if not text and not image_data:
                continue  # empty message, skip

            # Feed observation to consciousness
            _consciousness.inject_observation(f"Owner message: {text[:100]}")

            agent = _get_chat_agent()

            if agent._busy:
                # BUSY PATH: inject into active conversation
                if image_data:
                    if text:
                        agent.inject_message(text)
                    send_with_budget(chat_id,
                                     "📎 Photo received, but a task is in progress. Send again when I'm free.")
                elif text:
                    agent.inject_message(text)
            else:
                # FREE PATH: batch-collect burst messages, then dispatch
                _BATCH_WINDOW_SEC = 1.5  # collect messages for 1500ms
                _EARLY_EXIT_SEC = 0.15   # if no burst within 150ms → dispatch immediately
                _batch_start = time.time()
                _batch_deadline = _batch_start + _BATCH_WINDOW_SEC
                _batched_texts = [text] if text else []
                _batched_image = image_data  # keep first image

                _batch_state = load_state()
                _batch_state_dirty = False
                while time.time() < _batch_deadline:
                    time.sleep(0.1)
                    try:
                        _extra_updates = TG.get_updates(offset=offset, timeout=0) or []
                    except Exception:
                        _extra_updates = []
                    if not _extra_updates and (time.time() - _batch_start) < _EARLY_EXIT_SEC:
                        # No follow-up messages in first 150ms → single message, dispatch immediately
                        break
                    for _upd in _extra_updates:
                        offset = max(offset, int(_upd.get("update_id", offset - 1)) + 1)
                        _msg2 = _upd.get("message") or _upd.get("edited_message") or {}
                        _uid2 = (_msg2.get("from") or {}).get("id")
                        _cid2 = (_msg2.get("chat") or {}).get("id")
                        _txt2 = _msg2.get("text") or _msg2.get("caption") or ""
                        if _uid2 and _batch_state.get("owner_id") and _uid2 == int(_batch_state["owner_id"]):
                            log_chat("in", _cid2, _uid2, _txt2)
                            _batch_state["last_owner_message_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                            _batch_state_dirty = True
                            # Handle supervisor commands in batch window
                            if _txt2.strip().lower().startswith("/"):
                                try:
                                    _cmd_result = _handle_supervisor_command(_txt2, _cid2, tg_offset=offset)
                                    if _cmd_result is True:
                                        continue  # terminal command, don't batch
                                    elif _cmd_result:
                                        _txt2 = _cmd_result + _txt2  # dual-path: prepend note
                                except SystemExit:
                                    raise
                                except Exception:
                                    log.warning("Supervisor command in batch failed", exc_info=True)
                            if _txt2:
                                _batched_texts.append(_txt2)
                                _batch_deadline = max(_batch_deadline, time.time() + 0.3)
                            if not _batched_image:
                                _doc2 = _msg2.get("document") or {}
                                _photo2 = (_msg2.get("photo") or [None])[-1] or {}
                                _fid2 = _photo2.get("file_id") or _doc2.get("file_id")
                                if _fid2:
                                    _b642, _mime2 = TG.download_file_base64(_fid2)
                                    if _b642:
                                        _batched_image = (_b642, _mime2, _txt2)

                # Save state once if mutated during batch window
                if _batch_state_dirty:
                    save_state(_batch_state)

                # Merge all batched texts into one message
                if len(_batched_texts) > 1:
                    final_text = "\n\n".join(_batched_texts)
                    log.info("Message batch: %d messages merged into one", len(_batched_texts))
                elif _batched_texts:
                    final_text = _batched_texts[0]
                else:
                    final_text = text  # fallback to original

                # Re-check if agent became busy during batch window
                if agent._busy:
                    if final_text:
                        agent.inject_message(final_text)
                    if _batched_image:
                        send_with_budget(chat_id,
                                         "📎 Photo received, but a task is in progress. Send again when I'm free.")
                else:
                    # Dispatch to direct chat handler
                    _consciousness.pause()
                    def _run_task_and_resume(cid, txt, img):
                        try:
                            handle_chat_direct(cid, txt, img)
                        finally:
                            _consciousness.resume()
                    _t = threading.Thread(
                        target=_run_task_and_resume,
                        args=(chat_id, final_text, _batched_image),
                        daemon=True,
                    )
                    try:
                        _t.start()
                    except Exception as _te:
                        log.error("Failed to start chat thread: %s", _te)
                        _consciousness.resume()

        # Save offset
        st = load_state()
        st["tg_offset"] = offset
        save_state(st)

        # Slow cycle diagnostic
        now_epoch = time.time()
        loop_duration_sec = now_epoch - loop_started_ts
        if DIAG_SLOW_CYCLE_SEC > 0 and loop_duration_sec >= float(DIAG_SLOW_CYCLE_SEC):
            log.warning("Slow cycle: %.1fs (pending=%d, running=%d)",
                       loop_duration_sec, len(workers_mod.PENDING), len(workers_mod.RUNNING))

        # Sleep briefly between loop iterations
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point — supports both CLI and Telegram modes."""
    setup_env()

    # Check for Telegram mode
    if TELEGRAM_BOT_TOKEN:
        log.info("Telegram bot token found — starting Telegram polling mode")
        telegram_polling_loop()
    else:
        log.info("No Telegram bot token — starting CLI mode")
        agent = create_agent()
        cli_loop(agent)


if __name__ == "__main__":
    main()
