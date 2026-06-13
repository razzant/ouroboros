"""Worker lifecycle, health, and direct-chat handling for the supervisor."""

from __future__ import annotations
import logging
log = logging.getLogger(__name__)

import json
import multiprocessing as mp
import os
import pathlib
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from supervisor.state import load_state, append_jsonl, reconstruct_task_cost
from supervisor.message_bus import send_with_budget
from ouroboros.outcomes import EXECUTION_INFRA_FAILED, terminal_outcome_axes
from ouroboros.utils import utc_now_iso


REPO_DIR: pathlib.Path = pathlib.Path.home() / "Ouroboros" / "repo"
DRIVE_ROOT: pathlib.Path = pathlib.Path.home() / "Ouroboros" / "data"
MAX_WORKERS: int = 10
SOFT_TIMEOUT_SEC: int = 600
HARD_TIMEOUT_SEC: int = 1800
HEARTBEAT_STALE_SEC: int = 120
QUEUE_MAX_RETRIES: int = 1
TOTAL_BUDGET_LIMIT: float = 0.0
BRANCH_DEV: str = "ouroboros"
BRANCH_STABLE: str = "ouroboros-stable"

_CTX = None
_LAST_SPAWN_TIME: float = 0.0  # grace period: don't count dead workers right after spawn
_SPAWN_GRACE_SEC: float = 90.0  # workers need up to ~60s to init (spawn + pip)

# macOS + Windows default to spawn; Linux keeps fork.
#
# fork() from the long-lived, multi-threaded supervisor is unsafe on macOS: the
# child inherits dead Mach ports, and the first network call that resolves
# system proxies (SCDynamicStoreCopyProxies via _scproxy / httpx / requests)
# SIGSEGVs on the child side of fork pre-exec. macOS therefore uses spawn, like
# Windows. Linux proxy lookup reads env only (no Mach/GCD), so fork stays the
# default there for fast worker startup. ``worker_main`` is a module-level
# target (picklable) and re-derives all state from argv, so spawn is safe; the
# PyInstaller bootloader provides multiprocessing.freeze_support() for frozen
# builds. Override with OUROBOROS_WORKER_START_METHOD when diagnosing.
_DEFAULT_WORKER_START_METHOD = "fork" if sys.platform.startswith("linux") else "spawn"
_WORKER_START_METHOD = str(os.environ.get("OUROBOROS_WORKER_START_METHOD", _DEFAULT_WORKER_START_METHOD) or _DEFAULT_WORKER_START_METHOD).strip().lower()
if _WORKER_START_METHOD not in {"fork", "spawn", "forkserver"}:
    _WORKER_START_METHOD = _DEFAULT_WORKER_START_METHOD


def _get_ctx():
    """Return the multiprocessing context for workers."""
    global _CTX
    if _CTX is None:
        _CTX = mp.get_context(_WORKER_START_METHOD)
    return _CTX


def init(repo_dir: pathlib.Path, drive_root: pathlib.Path, max_workers: int,
         soft_timeout: int, hard_timeout: int, total_budget_limit: float,
         branch_dev: str = "ouroboros", branch_stable: str = "ouroboros-stable") -> None:
    global REPO_DIR, DRIVE_ROOT, MAX_WORKERS, SOFT_TIMEOUT_SEC, HARD_TIMEOUT_SEC
    global TOTAL_BUDGET_LIMIT, BRANCH_DEV, BRANCH_STABLE
    REPO_DIR = repo_dir
    DRIVE_ROOT = drive_root
    MAX_WORKERS = max_workers
    SOFT_TIMEOUT_SEC = soft_timeout
    HARD_TIMEOUT_SEC = hard_timeout
    TOTAL_BUDGET_LIMIT = total_budget_limit
    BRANCH_DEV = branch_dev
    BRANCH_STABLE = branch_stable

    from supervisor import queue
    queue.init(drive_root, soft_timeout, hard_timeout)
    queue.init_queue_refs(PENDING, RUNNING, QUEUE_SEQ_COUNTER_REF)

@dataclass
class Worker:
    wid: int
    proc: mp.Process
    in_q: Any
    busy_task_id: Optional[str] = None


_EVENT_Q = None


def get_event_q():
    """Return EVENT_Q, creating it lazily."""
    global _EVENT_Q
    if _EVENT_Q is None:
        _EVENT_Q = _get_ctx().Queue()
    return _EVENT_Q


WORKERS: Dict[int, Worker] = {}
PENDING: List[Dict[str, Any]] = []
RUNNING: Dict[str, Dict[str, Any]] = {}
CRASH_TS: List[float] = []
QUEUE_SEQ_COUNTER_REF: Dict[str, int] = {"value": 0}

# Shared queue lock; queue.py owns the canonical definition.
from supervisor.queue import _queue_lock


_chat_agent = None
# Serializes every direct-chat caller; _chat_agent has mutable per-call state.
import threading as _threading
_chat_agent_lock = _threading.Lock()


def _get_chat_agent():
    global _chat_agent
    if _chat_agent is None:
        if not getattr(sys, 'frozen', False):
            sys.path.insert(0, str(REPO_DIR))
        from ouroboros.agent import make_agent
        _chat_agent = make_agent(
            repo_dir=str(REPO_DIR),
            drive_root=str(DRIVE_ROOT),
            event_queue=get_event_q(),
        )
    return _chat_agent


def handle_chat_direct(
    chat_id: int,
    text: str,
    image_data: Optional[Union[Tuple[str, str], Tuple[str, str, str]]] = None,
    task_constraint: Optional[dict] = None,
    task_metadata: Optional[dict] = None,
) -> None:
    with _chat_agent_lock:
        _handle_chat_direct_locked(
            chat_id,
            text,
            image_data,
            task_constraint=task_constraint,
            task_metadata=task_metadata,
        )


def _handle_chat_direct_locked(
    chat_id: int,
    text: str,
    image_data: Optional[Union[Tuple[str, str], Tuple[str, str, str]]] = None,
    task_constraint: Optional[dict] = None,
    task_metadata: Optional[dict] = None,
) -> None:
    from supervisor.state import budget_remaining, load_state
    if budget_remaining(load_state()) <= 0:
        try:
            from supervisor.message_bus import get_bridge
            get_bridge().send_message(chat_id, "🚫 Budget exhausted. Task rejected. Please increase TOTAL_BUDGET in settings.")
        except Exception:
            pass
        return
        
    try:
        agent = _get_chat_agent()
        from ouroboros.contracts.task_contract import attach_task_contract

        task = {
            "id": uuid.uuid4().hex[:8],
            "type": "task",
            "chat_id": chat_id,
            "text": text,
            "_is_direct_chat": True,
        }
        if task_constraint:
            task["task_constraint"] = dict(task_constraint)
        if task_metadata:
            task["metadata"] = dict(task_metadata)
        if image_data:
            # image_data is (base64, mime) or (base64, mime, caption).
            task["image_base64"] = image_data[0]
            task["image_mime"] = image_data[1]
            if len(image_data) > 2 and image_data[2]:
                task["image_caption"] = image_data[2]
                if not text:
                    task["text"] = image_data[2]
        if not task["text"]:
            task["text"] = "(image attached)" if image_data else ""
        attach_task_contract(task)
        events = agent.handle_task(task)
        for e in events:
            get_event_q().put(e)
    except Exception as e:
        import traceback
        err_msg = f"⚠️ Error: {type(e).__name__}: {e}"
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "direct_chat_error",
                "error": repr(e),
                "traceback": str(traceback.format_exc())[:2000],
            },
        )
        try:
            from supervisor.message_bus import get_bridge
            get_bridge().send_message(chat_id, err_msg)
        except Exception:
            log.debug("Suppressed exception", exc_info=True)

def _collect_orphaned_owner_messages() -> tuple[list, list]:
    """Read owner-text entries from every mailbox left behind by the process
    that died across the restart.

    Owner messages received while a task was busy are persisted per-task under
    ``memory/owner_mailbox/<task_id>.jsonl``. After an exit-42/99 restart the
    worker comes back with a *new* task_id, so those files are orphaned and the
    round-boundary drain (which keys off the live task_id) never sees them.
    This collects their ``KIND_OWNER_TEXT`` entries (control kinds are skipped —
    stale post-restart), within a 24h recovery horizon, sorted chronologically.
    Read-only: the caller unlinks the files only once it commits to resuming.
    """
    msgs: list = []
    files: list = []
    try:
        from datetime import datetime, timedelta, timezone
        from ouroboros.owner_mailbox import KIND_OWNER_TEXT
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        mbox_dir = DRIVE_ROOT / "memory" / "owner_mailbox"
        if not mbox_dir.exists():
            return [], []
        for path in sorted(mbox_dir.glob("*.jsonl")):
            files.append(path)
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if str(entry.get("kind") or KIND_OWNER_TEXT) != KIND_OWNER_TEXT:
                        continue
                    text = str(entry.get("text") or "").strip()
                    if not text:
                        continue
                    ts = str(entry.get("ts") or "")
                    # Drop only entries we can confidently date as stale; on a
                    # parse failure deliver (losing a real message is worse).
                    try:
                        if datetime.fromisoformat(ts) < cutoff:
                            continue
                    except Exception:
                        pass
                    msgs.append((ts, text))
            except Exception:
                log.debug("Failed reading orphan mailbox %s", path, exc_info=True)
        msgs.sort(key=lambda x: x[0])
        return [t for _, t in msgs], files
    except Exception:
        log.debug("Failed collecting orphaned owner messages", exc_info=True)
        return [], files


def _purge_owner_mailboxes() -> None:
    """Delete every orphaned owner mailbox.

    Used on the deliberate-suppression boots (panic / owner-restart): a message
    the owner suppressed by stopping must NOT be resurrected by a later,
    non-suppressed restart within the 24h window. Only deleting the files
    guarantees the P0 invariant that no orphan owner message is delivered after
    a stop.
    """
    try:
        mbox_dir = DRIVE_ROOT / "memory" / "owner_mailbox"
        if not mbox_dir.exists():
            return
        for path in mbox_dir.glob("*.jsonl"):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                log.debug("Failed to purge owner mailbox %s", path, exc_info=True)
    except Exception:
        log.debug("Failed to purge owner mailboxes", exc_info=True)


def auto_resume_after_restart() -> None:
    """Auto-resume after a recent restart when scratchpad still has work or an
    owner message arrived during the task that the restart interrupted."""
    try:
        owner_restart_flag = DRIVE_ROOT / "state" / "owner_restart_no_resume.flag"
        if owner_restart_flag.exists():
            owner_restart_flag.unlink(missing_ok=True)
            panic_compat_flag = DRIVE_ROOT / "state" / "panic_stop.flag"
            try:
                if panic_compat_flag.read_text(encoding="utf-8").strip() == "owner_restart_no_resume":
                    panic_compat_flag.unlink(missing_ok=True)
            except FileNotFoundError:
                pass
            except Exception:
                log.debug("Failed to consume owner restart compatibility flag", exc_info=True)
            # Owner deliberately restarted without resume → drop any pending
            # owner messages so they cannot resurface on a later restart.
            _purge_owner_mailboxes()
            log.info("Owner restart flag detected — skipping auto-resume.")
            return

        # Panic/owner-restart flags suppress auto-resume and are consumed.
        panic_flag = DRIVE_ROOT / "state" / "panic_stop.flag"
        if panic_flag.exists():
            panic_flag.unlink(missing_ok=True)
            # Emergency Stop → drop any pending owner messages (P0): they must
            # never be resurrected by a subsequent non-suppressed restart.
            _purge_owner_mailboxes()
            log.info("Panic flag detected — skipping auto-resume.")
            return

        st = load_state()
        chat_id = st.get("owner_chat_id")
        if not chat_id:
            return

        restart_verify_path = DRIVE_ROOT / "state" / "pending_restart_verify.json"
        recent_restart = False
        if restart_verify_path.exists():
            recent_restart = True
        else:
            sup_log = DRIVE_ROOT / "logs" / "supervisor.jsonl"
            if sup_log.exists():
                try:
                    lines = sup_log.read_text(encoding="utf-8").strip().split("\n")
                    for line in reversed(lines[-20:]):
                        if not line.strip():
                            continue
                        evt = json.loads(line)
                        if evt.get("type") in ("launcher_start", "restart"):
                            recent_restart = True
                            break
                except Exception:
                    log.debug("Suppressed exception", exc_info=True)

        if not recent_restart:
            return

        # Owner messages orphaned by the restart's new task_id. Read-only here;
        # consumed only on resume.
        missed_msgs, mailbox_files = _collect_orphaned_owner_messages()

        has_scratchpad_work = False
        scratchpad_path = DRIVE_ROOT / "memory" / "scratchpad.md"
        if scratchpad_path.exists():
            scratchpad = scratchpad_path.read_text(encoding="utf-8")
            stripped = scratchpad.strip()
            if stripped and stripped != "# Scratchpad" and "(empty" not in stripped.lower():
                has_scratchpad_work = True
            else:
                content_lines = [
                    ln.strip() for ln in stripped.splitlines()
                    if ln.strip() and not ln.strip().startswith("#") and ln.strip() != "- (empty)"
                ]
                content_lines = [ln for ln in content_lines if not ln.startswith("UpdatedAt:")]
                has_scratchpad_work = bool(content_lines)

        # Resume if there is unfinished scratchpad work OR an owner message was
        # left unanswered when the process restarted.
        if not has_scratchpad_work and not missed_msgs:
            return

        time.sleep(2)  # Let everything initialize

        # Re-check panic: an Emergency Stop raised during the init window above
        # must abort resume (P0); pending owner messages are dropped too.
        if (DRIVE_ROOT / "state" / "panic_stop.flag").exists():
            _purge_owner_mailboxes()
            log.info("Panic flag appeared during init — aborting auto-resume.")
            return

        agent = _get_chat_agent()
        if not agent._busy:
            resume_prompt = (
                "[auto-resume after restart] Continue your work. Read scratchpad "
                "and identity — they contain context of what you were doing."
            )
            if missed_msgs:
                shown = missed_msgs[-20:]
                lines = "\n".join(f"- {m}" for m in shown)
                omitted = len(missed_msgs) - len(shown)
                more = "" if omitted <= 0 else f"\n(+{omitted} earlier message(s) omitted)"
                resume_prompt += (
                    "\n\n[messages from your human arrived while you were working "
                    "and the process restarted before you could read them — they "
                    "have been waiting for a reply, address them first]:\n"
                    + lines + more
                )
            # Consume every orphaned mailbox we scanned now that we are resuming:
            # fresh messages were delivered above, stale/control-only files are
            # discarded so they cannot resurface on a later restart.
            for path in mailbox_files:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    log.debug("Failed to unlink orphan mailbox %s", path, exc_info=True)
            import threading
            threading.Thread(
                target=handle_chat_direct,
                args=(int(chat_id), resume_prompt, None),
                daemon=True,
            ).start()
            append_jsonl(
                DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "auto_resume_triggered",
                    "missed_owner_messages": len(missed_msgs),
                },
            )
    except Exception as e:
        append_jsonl(DRIVE_ROOT / "logs" / "supervisor.jsonl", {
            "ts": utc_now_iso(),
            "type": "auto_resume_error",
            "error": repr(e),
        })

# Log types the worker sink does NOT forward: each already reaches the dashboard
# live via a dedicated EVENT_Q sibling/handler, so forwarding the worker's
# append_jsonl copy too would double-broadcast (and task_checkpoint would also be
# re-persisted to events.jsonl by _handle_log_event, a double file write).
WORKER_LOG_SINK_SUPPRESSED_TYPES = frozenset({
    "tool_call", "llm_round", "task_checkpoint", "task_done", "llm_usage",
})


def _current_custody_session_id() -> str:
    """Server-side custody session id to hand to spawned workers (best-effort)."""
    try:
        from ouroboros.process_custody import current_custody_session_id
        return current_custody_session_id()
    except Exception:
        return ""


def worker_main(wid: int, in_q: Any, out_q: Any, repo_dir: str, drive_root: str,
                custody_session_id: str = "") -> None:
    import os as _os
    # Mark this process as a worker BEFORE importing the agent/LLM stack so the
    # central network-transport policy disables system proxy resolution
    # (trust_env=False) for every HTTP client created here. This is the
    # fork-safety guard (no _scproxy/SCDynamicStoreCopyProxies on the child side
    # of fork) and a clean default for spawned workers too.
    _os.environ["OUROBOROS_IN_WORKER"] = "1"
    # Adopt the server's custody session id. Under the 'spawn' start method this
    # process re-imported process_custody and minted a fresh _SESSION_ID; without
    # adopting the server's id, every service/process this worker records looks
    # foreign to the server's reaper and gets killed at the next reap tick —
    # even a still-running task's services. Passed as an arg (not env) so it
    # cannot survive a server re-exec. See process_custody.adopt_session_id.
    if custody_session_id:
        try:
            from ouroboros.process_custody import adopt_session_id
            adopt_session_id(custody_session_id)
        except Exception:
            pass
    from ouroboros.platform_layer import create_new_session
    create_new_session()
    # Lifeline: if the supervisor dies abruptly, this worker is reparented to
    # init and would keep running LLM rounds invisibly — group-suicide instead.
    try:
        from ouroboros.process_custody import start_parent_lifeline

        start_parent_lifeline(label=f"worker-{wid}")
    except Exception:
        pass
    # Stream this worker's append_jsonl log lines to the dashboard Logs panel.
    # The WS log sink lives only in the main process, so without this every
    # worker-task log line (queued/evolution/review/subagent) is written to file
    # but never broadcast live — the "not all logs arrive" gap. Forward over the
    # existing EVENT_Q -> _handle_log_event -> push_log path. Suppress types that
    # already arrive live via a dedicated sibling event (tool_call/llm_round/
    # task_checkpoint) or are appended in the main process (task_done/llm_usage)
    # to avoid double broadcast and (for task_checkpoint) a double file write.
    try:
        from ouroboros.utils import emit_log_event, set_log_sink

        def _worker_log_sink(obj: Any) -> None:
            if isinstance(obj, dict) and str(obj.get("type") or "") in WORKER_LOG_SINK_SUPPRESSED_TYPES:
                return
            emit_log_event(out_q, obj, log_label="worker log")

        set_log_sink(_worker_log_sink)
    except Exception:
        pass
    import sys as _sys
    import traceback as _tb
    import pathlib as _pathlib
    if not getattr(_sys, 'frozen', False):
        _sys.path.insert(0, repo_dir)
    _drive = _pathlib.Path(drive_root)
    # Spawned workers must pin the runtime-mode baseline from the parent env;
    # forked workers inherit it. This keeps the elevation ratchet consistent.
    try:
        from ouroboros.config import initialize_runtime_mode_baseline
        initialize_runtime_mode_baseline()
    except Exception:
        # Non-fatal: save_settings still has env-var fallback gating.
        try:
            _log_worker_crash(wid, _drive, "init_baseline", None, _tb.format_exc())
        except Exception:
            pass
    try:
        from ouroboros.config import get_skills_repo_path, load_settings as _load_settings
        from ouroboros.extension_loader import reload_all as _reload_extensions

        pytest_default_real_data_dir = (
            "pytest" in _sys.modules
            and not _os.environ.get("OUROBOROS_DATA_DIR")
            and _drive.resolve(strict=False) == (_pathlib.Path.home() / "Ouroboros" / "data").resolve(strict=False)
        )
        if pytest_default_real_data_dir:
            try:
                from ouroboros.utils import append_jsonl, utc_now_iso
                append_jsonl(_drive / "logs" / "supervisor.jsonl", {
                    "ts": utc_now_iso(),
                    "type": "worker_extension_reload_skipped",
                    "worker_id": wid,
                    "reason": "pytest_default_real_data_dir",
                })
            except Exception:
                pass
        else:
            _repo_path = get_skills_repo_path()
            _reload_extensions(_drive, _load_settings, repo_path=_repo_path or None)
    except Exception:
        try:
            _log_worker_crash(wid, _drive, "extension_reload", None, _tb.format_exc())
        except Exception:
            pass
    try:
        from ouroboros.agent import make_agent
        agent = make_agent(repo_dir=repo_dir, drive_root=drive_root, event_queue=out_q)
    except Exception as _e:
        _log_worker_crash(wid, _drive, "make_agent", _e, _tb.format_exc())
        return
    while True:
        try:
            task = in_q.get()
            if task is None or task.get("type") == "shutdown":
                break
            task_drive_root = str(task.get("drive_root") or drive_root)
            if task_drive_root != str(drive_root):
                task_agent = make_agent(repo_dir=repo_dir, drive_root=task_drive_root, event_queue=out_q)
                events = task_agent.handle_task(task)
            else:
                events = agent.handle_task(task)
            for e in events:
                e2 = dict(e)
                e2["worker_id"] = wid
                out_q.put(e2)
        except Exception as _e:
            _log_worker_crash(wid, _drive, "handle_task", _e, _tb.format_exc())


def _write_failure_result(
    task_id: str,
    reason: str = "Worker process crashed (crash storm). Task was not completed.",
    status: str = "",
) -> str:
    """Write failure result for a crashed/orphaned task.

    Returns the FINAL persisted status: if the task already reached a terminal
    state, the monotonic guard preserves it and that existing status is returned
    (so the UI event matches disk); otherwise the written failure status.
    """
    if not task_id:
        return ""
    try:
        from ouroboros.task_results import (
            STATUS_FAILED, STATUS_COMPLETED, STATUS_REJECTED_DUPLICATE,
            STATUS_CANCELLED, load_task_result, write_task_result,
        )
        # STATUS_INTERRUPTED is not final; it is written before requeue.
        _FINAL_STATUSES = {STATUS_COMPLETED, STATUS_FAILED, STATUS_REJECTED_DUPLICATE, STATUS_CANCELLED}
        existing = load_task_result(DRIVE_ROOT, task_id)
        if existing and existing.get("status") in _FINAL_STATUSES:
            return str(existing.get("status") or "")
        final_status = status or STATUS_FAILED
        # Reconstruct from durable llm_usage so an abnormally-finalized task does
        # not record zero cost/rounds (understating per-task + campaign metrics).
        f_cost, f_rounds, f_prompt, f_completion = reconstruct_task_cost(str(task_id))
        write_task_result(
            DRIVE_ROOT,
            task_id,
            final_status,
            result=reason,
            reason_code="worker_terminal_failure" if final_status == STATUS_FAILED else str(final_status or ""),
            outcome_axes=terminal_outcome_axes(
                lifecycle=final_status,
                execution=EXECUTION_INFRA_FAILED if final_status == STATUS_FAILED else str(final_status or ""),
                reason_code="worker_terminal_failure" if final_status == STATUS_FAILED else str(final_status or ""),
                review_trigger="worker_terminal",
            ),
            cost_usd=f_cost,
            total_rounds=f_rounds,
            prompt_tokens=f_prompt,
            completion_tokens=f_completion,
        )
        return final_status
    except Exception:
        log.warning("Failed to write failure result for task %s", task_id, exc_info=True)
        return status or "failed"


def _emit_task_done_terminal(
    task: Optional[Dict[str, Any]],
    task_id: str,
    status: str = "failed",
    *,
    cost_usd: float = 0.0,
    total_rounds: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> None:
    """Emit a task_done event so the UI resolves the live card when a task is
    torn down outside the normal completion path (crash storm, kill, hard
    timeout). Without this the spinner spins forever on these paths.

    Cost fields carry reconstructed totals so an evolution campaign tally fed
    from this terminal event records real spend instead of zeros; callers that
    have no reconstructed cost leave them at 0."""
    if not task_id:
        return
    try:
        chat_id = int((task or {}).get("chat_id") or 0)
    except (TypeError, ValueError):
        chat_id = 0
    if not chat_id:
        return
    status = status or "failed"
    reason_code = "worker_terminal_failure" if status == "failed" else status
    try:
        get_event_q().put({
            "type": "task_done",
            "task_id": str(task_id),
            "task_type": str((task or {}).get("type") or ""),
            "chat_id": chat_id,
            "status": status,
            "outcome_axes": terminal_outcome_axes(
                lifecycle=status,
                execution=EXECUTION_INFRA_FAILED if status == "failed" else status,
                reason_code=reason_code,
                review_trigger="worker_terminal",
            ),
            "reason_code": reason_code,
            "cost_usd": cost_usd,
            "total_rounds": total_rounds,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        })
    except Exception:
        log.debug("Failed to emit terminal task_done for %s", task_id, exc_info=True)


def _log_worker_crash(wid: int, drive_root: pathlib.Path, phase: str, exc: Exception, tb: str) -> None:
    """Best-effort worker-side crash logging."""
    import os as _os
    try:
        path = drive_root / "logs" / "supervisor.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = json.dumps({
            "ts": utc_now_iso(),
            "type": "worker_crash",
            "worker_id": wid,
            "pid": _os.getpid(),
            "phase": phase,
            "error": repr(exc),
            "traceback": str(tb)[:3000],
        }, ensure_ascii=False)
        with path.open("a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except Exception:
        log.debug("Suppressed exception", exc_info=True)


def _first_worker_boot_event_since(offset_bytes: int) -> Optional[Dict[str, Any]]:
    """Read first worker_boot event after a file offset."""
    path = DRIVE_ROOT / "logs" / "events.jsonl"
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            safe_offset = offset_bytes if 0 <= offset_bytes <= size else 0
            f.seek(safe_offset)
            data = f.read().decode("utf-8", errors="replace")
    except Exception:
        log.debug("Suppressed exception", exc_info=True)
        return None

    for line in data.splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            evt = json.loads(raw)
        except Exception:
            log.debug("Suppressed exception in loop", exc_info=True)
            continue
        if isinstance(evt, dict) and str(evt.get("type") or "") == "worker_boot":
            return evt
    return None


def _verify_worker_sha_after_spawn(events_offset: int, timeout_sec: float = 90.0) -> None:
    """Verify newly spawned workers booted at expected current_sha."""
    st = load_state()
    expected_sha = str(st.get("current_sha") or "").strip()
    if not expected_sha:
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "worker_sha_verify_skipped",
                "reason": "missing_current_sha",
            },
        )
        return

    deadline = time.time() + max(float(timeout_sec), 1.0)
    boot_evt = None
    while time.time() < deadline:
        boot_evt = _first_worker_boot_event_since(events_offset)
        if boot_evt is not None:
            break
        time.sleep(0.25)

    if boot_evt is None:
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "worker_sha_verify_timeout",
                "expected_sha": expected_sha,
            },
        )
        return

    observed_sha = str(boot_evt.get("git_sha") or "").strip()
    ok = bool(observed_sha and observed_sha == expected_sha)
    append_jsonl(
        DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {
            "ts": utc_now_iso(),
            "type": "worker_sha_verify",
            "ok": ok,
            "expected_sha": expected_sha,
            "observed_sha": observed_sha,
            "worker_pid": boot_evt.get("pid"),
        },
    )
    if not ok and st.get("owner_chat_id"):
        send_with_budget(
            int(st["owner_chat_id"]),
            f"⚠️ Worker SHA mismatch after spawn: expected {expected_sha[:8]}, got {(observed_sha or 'unknown')[:8]}",
        )


_WORKER_PIDS_FILENAME = "worker_pids.json"


def _worker_pids_path() -> pathlib.Path:
    return DRIVE_ROOT / "state" / _WORKER_PIDS_FILENAME


def _record_worker_pids() -> None:
    """Persist current worker PIDs so a later server instance can reap any that
    survive an abrupt restart. Workers run in their own ``os.setsid`` session, so
    when the parent server dies they are reparented to init and outlive it."""
    try:
        from ouroboros.utils import atomic_write_json
        recs = [{"pid": int(w.proc.pid)} for w in WORKERS.values() if w.proc.pid]
        atomic_write_json(
            _worker_pids_path(),
            {"server_pid": os.getpid(), "ts": utc_now_iso(), "workers": recs},
            trailing_newline=True,
        )
    except Exception:
        log.debug("Failed to record worker pids", exc_info=True)
    # Write-through into the custody ledger (SSOT for the generation reaper);
    # worker_pids.json stays as the legacy session-leader reap path.
    try:
        from ouroboros.process_custody import record_process

        for w in WORKERS.values():
            if w.proc.pid:
                record_process(
                    DRIVE_ROOT,
                    pid=int(w.proc.pid),
                    cmd=f"ouroboros-worker-{w.wid}",
                    purpose=f"worker:{w.wid}",
                    scope="session",
                )
    except Exception:
        log.debug("Failed to ledger worker pids", exc_info=True)


def reap_orphaned_workers() -> int:
    """Kill leftover worker process groups left by a PRIOR server instance.

    ``kill_workers`` only walks the in-memory ``WORKERS`` dict, so workers
    orphaned by an abrupt restart (reparented to init, ~one Python interpreter
    each) were never reaped and accumulated across restarts. On startup we read
    the prior pid record and force-kill any that are still alive AND verifiably
    ours — cmdline matches this interpreter/multiprocessing and the process is
    its own session leader (``pgid == pid``) — which guards against PID reuse and
    bounds the group kill to the worker's own setsid session."""
    try:
        from ouroboros.utils import read_json_dict
        from ouroboros.platform_layer import (
            force_kill_pid,
            kill_process_group_id,
            process_command,
            process_group_id,
        )
    except Exception:
        return 0
    data = read_json_dict(_worker_pids_path()) or {}
    prior = data.get("workers") or []
    if not isinstance(prior, list) or not prior:
        return 0
    current = {w.proc.pid for w in WORKERS.values() if w.proc.pid}
    killed: List[int] = []
    for rec in prior:
        try:
            pid = int((rec or {}).get("pid") or 0)
        except (TypeError, ValueError):
            continue
        if not pid or pid in current or pid == os.getpid():
            continue
        cmd = process_command(pid)
        if not cmd:
            continue  # already dead
        if sys.executable not in cmd and "multiprocessing" not in cmd:
            continue  # PID reused by an unrelated process — do not touch it
        pgid = process_group_id(pid)
        if pgid and pgid == pid:
            kill_process_group_id(pgid)  # the worker's own setsid session
        force_kill_pid(pid)
        killed.append(pid)
    if killed:
        try:
            append_jsonl(
                DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {"ts": utc_now_iso(), "type": "orphaned_workers_reaped", "pids": killed},
            )
        except Exception:
            log.debug("Failed to log orphaned worker reap", exc_info=True)
    return len(killed)


def spawn_workers(n: int = 0) -> None:
    global _CTX, _EVENT_Q
    # Reap any workers left orphaned by a prior/abrupt server exit before we
    # spawn fresh ones, so process groups do not accumulate across restarts.
    reap_orphaned_workers()
    # Fresh context ensures workers use current code.
    _CTX = mp.get_context(_WORKER_START_METHOD)
    _EVENT_Q = _CTX.Queue()
    events_path = DRIVE_ROOT / "logs" / "events.jsonl"
    try:
        events_offset = int(events_path.stat().st_size)
    except Exception:
        events_offset = 0

    count = n or MAX_WORKERS
    append_jsonl(
        DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {
            "ts": utc_now_iso(),
            "type": "worker_spawn_start",
            "start_method": _WORKER_START_METHOD,
            "count": count,
        },
    )
    WORKERS.clear()
    for i in range(count):
        in_q = _CTX.Queue()
        proc = _CTX.Process(target=worker_main,
                           args=(i, in_q, _EVENT_Q, str(REPO_DIR), str(DRIVE_ROOT),
                                 _current_custody_session_id()))
        proc.daemon = True
        proc.start()
        WORKERS[i] = Worker(wid=i, proc=proc, in_q=in_q, busy_task_id=None)
    global _LAST_SPAWN_TIME
    _LAST_SPAWN_TIME = time.time()
    _record_worker_pids()
    # Verify asynchronously so spawn does not block the supervisor loop.
    threading.Thread(target=_verify_worker_sha_after_spawn, args=(events_offset,), daemon=True).start()


def kill_workers(
    force: bool = True,
    *,
    result_reason: str = "Worker process crashed (crash storm). Task was not completed.",
    terminal_status: str = "",
    archive_service_logs: bool = True,
) -> None:
    from supervisor import queue
    with _queue_lock:
        cleared_running = len(RUNNING)
        from ouroboros.platform_layer import kill_pid_tree
        for w in WORKERS.values():
            if w.proc.pid:
                kill_pid_tree(w.proc.pid)
            elif w.proc.is_alive():
                w.proc.terminate()
        for w in WORKERS.values():
            w.proc.join(timeout=3)
        _kill_survivors()
        WORKERS.clear()
        try:
            done_status = terminal_status or "failed"
            orphaned_ids = []
            for task_id in list(RUNNING):
                try:
                    meta = RUNNING.get(task_id) or {}
                    task = meta.get("task") if isinstance(meta, dict) and isinstance(meta.get("task"), dict) else {}
                    persisted = _write_failure_result(task_id, reason=result_reason, status=terminal_status)
                    if archive_service_logs:
                        try:
                            from ouroboros.tools.services import archive_task_service_logs
                            archive_task_service_logs(pathlib.Path(DRIVE_ROOT), str(task_id), task)
                        except Exception:
                            log.debug("Failed to archive service logs for task %s", task_id, exc_info=True)
                    _emit_task_done_terminal(task, str(task_id), persisted or done_status)
                    orphaned_ids.append(task_id)
                except Exception:
                    log.warning("Failed to write failure result for running task %s", task_id, exc_info=True)
            drained = queue.drain_all_pending()
            drained_ids = []
            for task in drained:
                tid = task.get("id")
                if tid:
                    try:
                        persisted = _write_failure_result(tid, reason=result_reason, status=terminal_status)
                        _emit_task_done_terminal(task, str(tid), persisted or done_status)
                        drained_ids.append(tid)
                    except Exception:
                        log.warning("Failed to write failure result for pending task %s", tid, exc_info=True)
            if orphaned_ids or drained_ids:
                append_jsonl(
                    DRIVE_ROOT / "logs" / "supervisor.jsonl",
                    {
                        "ts": utc_now_iso(),
                        "type": "zombie_prevention_cleanup",
                        "orphaned_running": orphaned_ids,
                        "drained_pending": drained_ids,
                    },
                )
        except Exception:
            log.warning("Zombie prevention cleanup failed", exc_info=True)
        RUNNING.clear()
    queue.persist_queue_snapshot(reason="kill_workers")
    if cleared_running:
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "running_cleared_on_kill", "count": cleared_running,
                "force": force,
            },
        )


def _kill_survivors() -> None:
    """Force-kill any workers and their entire descendant trees."""
    from ouroboros.platform_layer import kill_pid_tree
    for w in WORKERS.values():
        pid = w.proc.pid
        if pid is None:
            continue
        if w.proc.is_alive():
            kill_pid_tree(pid)
            w.proc.join(timeout=2)


def respawn_worker(wid: int) -> None:
    ctx = _get_ctx()
    in_q = ctx.Queue()
    proc = ctx.Process(target=worker_main,
                       args=(wid, in_q, get_event_q(), str(REPO_DIR), str(DRIVE_ROOT),
                             _current_custody_session_id()))
    proc.daemon = True
    proc.start()
    # Swap under _queue_lock (an RLock — safe even when the caller already holds
    # it) so a concurrent assign_tasks cannot enqueue into the slot mid-swap.
    with _queue_lock:
        old = WORKERS.get(wid)
        WORKERS[wid] = Worker(wid=wid, proc=proc, in_q=in_q, busy_task_id=None)
    # Close the crashed worker's old queue now that nothing can route to it,
    # otherwise its file descriptors / semaphores leak on every respawn.
    if old is not None and getattr(old, "in_q", None) is not None:
        try:
            old.in_q.close()
            old.in_q.cancel_join_thread()
        except Exception:
            log.debug("Failed to close old worker queue on respawn", exc_info=True)
    _record_worker_pids()
    # Do not reset _LAST_SPAWN_TIME here; respawn grace would hide crash storms.


def _drop_cancelled_pending() -> None:
    """Remove pending tasks cancelled/finished between scheduling and assignment
    so a cancelled subagent never actually starts. Caller holds _queue_lock."""
    if not PENDING:
        return
    try:
        from ouroboros.task_results import (
            STATUS_CANCEL_REQUESTED, STATUS_CANCELLED, _TRULY_TERMINAL_STATUSES,
            load_task_result, write_task_result,
        )
    except Exception:
        return
    survivors: List[Dict[str, Any]] = []
    dropped: List[str] = []
    for t in PENDING:
        tid = str(t.get("id") or "")
        status = ""
        if tid:
            try:
                existing = load_task_result(DRIVE_ROOT, tid)
                status = str((existing or {}).get("status") or "")
            except Exception:
                status = ""
        if status == STATUS_CANCEL_REQUESTED:
            try:
                write_task_result(DRIVE_ROOT, tid, STATUS_CANCELLED, result="Cancelled before start.")
            except Exception:
                log.debug("Failed to finalize cancelled pending task %s", tid, exc_info=True)
            _emit_task_done_terminal(t, tid, "cancelled")
            dropped.append(tid)
            continue
        if status in _TRULY_TERMINAL_STATUSES:
            dropped.append(tid)
            continue
        survivors.append(t)
    if dropped:
        PENDING[:] = survivors
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {"ts": utc_now_iso(), "type": "pending_cancelled_dropped", "task_ids": dropped},
        )


def assign_tasks() -> None:
    from supervisor import queue
    from supervisor.state import budget_remaining, EVOLUTION_BUDGET_RESERVE
    with _queue_lock:
        st = load_state()
        remaining = budget_remaining(st)
        
        if remaining <= 0:
            return  # Stop assigning ALL tasks if budget is completely exhausted

        # Drop tasks cancelled after scheduling but before assignment.
        _drop_cancelled_pending()

        # Evolution is hard-blocked in light runtime mode at the assignment
        # chokepoint too: a task restored from a snapshot or created before the
        # mode switch must never actually run. Cancel them terminally.
        from supervisor.evolution_lifecycle import evolution_block_reason
        evo_block = evolution_block_reason()
        if evo_block and any(str(t.get("type") or "") == "evolution" for t in PENDING):
            blocked_ids = [str(t.get("id") or "") for t in PENDING if str(t.get("type") or "") == "evolution"]
            PENDING[:] = [t for t in PENDING if str(t.get("type") or "") != "evolution"]
            from ouroboros.task_results import STATUS_CANCELLED, write_task_result
            for tid in blocked_ids:
                try:
                    write_task_result(
                        DRIVE_ROOT, tid, STATUS_CANCELLED,
                        result="Evolution is disabled in light runtime mode.",
                    )
                except Exception:
                    log.debug("Failed to cancel light-mode evolution task %s", tid, exc_info=True)
            if st.get("owner_chat_id"):
                send_with_budget(int(st["owner_chat_id"]), evo_block)
            queue.persist_queue_snapshot(reason="evolution_blocked_light")

        for w in WORKERS.values():
            if w.busy_task_id is None and PENDING:
                # Find first suitable task (skip over-budget evolution tasks)
                chosen_idx = None
                for i, candidate in enumerate(PENDING):
                    if str(candidate.get("type") or "") == "evolution" and remaining < EVOLUTION_BUDGET_RESERVE:
                        continue
                    chosen_idx = i
                    break
                if chosen_idx is None:
                    # Only over-budget evolution tasks remain — clean them out
                    PENDING[:] = [t for t in PENDING if str(t.get("type") or "") != "evolution"]
                    queue.persist_queue_snapshot(reason="evolution_dropped_budget")
                    continue
                task = PENDING.pop(chosen_idx)
                if str(task.get("delegation_role") or "") == "subagent" and str(task.get("drive_root") or ""):
                    try:
                        from ouroboros.task_results import STATUS_RUNNING, write_task_result
                        write_task_result(
                            DRIVE_ROOT,
                            str(task.get("id") or ""),
                            STATUS_RUNNING,
                            parent_task_id=task.get("parent_task_id"),
                            root_task_id=task.get("root_task_id"),
                            session_id=task.get("session_id"),
                            actor_id=task.get("actor_id"),
                            delegation_role=task.get("delegation_role"),
                            project_id=task.get("project_id"),
                            role=task.get("role"),
                            description=task.get("description"),
                            objective=task.get("objective") or task.get("description"),
                            expected_output=task.get("expected_output"),
                            constraints=task.get("constraints"),
                            context=task.get("context"),
                            memory_mode=task.get("memory_mode"),
                            drive_root=task.get("drive_root"),
                            child_drive_root=task.get("child_drive_root") or task.get("drive_root"),
                            budget_drive_root=task.get("budget_drive_root"),
                            task_constraint=task.get("task_constraint"),
                            model_lane=task.get("model_lane"),
                            requested_model_lane=task.get("requested_model_lane"),
                            effective_model_lane=task.get("effective_model_lane"),
                            model=task.get("model"),
                            use_local_model=task.get("use_local_model"),
                            task_group_id=task.get("task_group_id"),
                            task_group=task.get("task_group"),
                            subagent_envelope=task.get("subagent_envelope"),
                            metadata=task.get("metadata") if isinstance(task.get("metadata"), dict) else {},
                            result="Subagent assigned to a worker.",
                        )
                    except Exception:
                        log.debug("Failed to mirror running subagent status", exc_info=True)
                w.busy_task_id = task["id"]
                w.in_q.put(task)
                now_ts = time.time()
                RUNNING[task["id"]] = {
                    "task": dict(task), "worker_id": w.wid,
                    "started_at": now_ts, "last_heartbeat_at": now_ts,
                    "soft_sent": False, "attempt": int(task.get("_attempt") or 1),
                }
                task_type = str(task.get("type") or "")
                if task_type in ("evolution", "review"):
                    st = load_state()
                    if st.get("owner_chat_id"):
                        emoji = '🧬' if task_type == 'evolution' else '🔎'
                        send_with_budget(
                            int(st["owner_chat_id"]),
                            f"{emoji} {task_type.capitalize()} task {task['id']} started.",
                        )
                queue.persist_queue_snapshot(reason="assign_task")

def ensure_workers_healthy() -> None:
    """Detect dead workers, finalize/requeue their tasks, respawn.

    Runs under the queue lock: the RUNNING pops and respawn decisions here
    raced with HTTP cancel handlers (double respawn → orphaned worker, and
    "dict changed size" crashes in concurrent iteration). RLock keeps the
    nested enqueue/respawn/persist calls re-entrant.
    """
    from supervisor import queue
    # Workers need init time after spawn.
    if (time.time() - _LAST_SPAWN_TIME) < _SPAWN_GRACE_SEC:
        return
    with _queue_lock:
        _ensure_workers_healthy_locked(queue)


def _ensure_workers_healthy_locked(queue: Any) -> None:
    busy_crashes = 0
    dead_detections = 0
    crashed_tasks = []
    for wid, w in list(WORKERS.items()):
        if not w.proc.is_alive():
            dead_detections += 1
            if w.busy_task_id is not None:
                busy_crashes += 1
            exitcode = w.proc.exitcode
            meta = RUNNING.get(w.busy_task_id, {}) if w.busy_task_id else {}
            task_info = meta.get("task", {}) if isinstance(meta, dict) else {}
            append_jsonl(
                DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "worker_dead_detected",
                    "worker_id": wid,
                    "exitcode": exitcode,
                    "busy_task_id": w.busy_task_id,
                    "task_type": task_info.get("type") if isinstance(task_info, dict) else None,
                    "task_description": (task_info.get("description", "") or "")[:200] if isinstance(task_info, dict) else None,
                    "uptime_sec": round(time.time() - meta["started_at"]) if isinstance(meta, dict) and meta.get("started_at") else None,
                    "attempt": meta.get("attempt") if isinstance(meta, dict) else None,
                    "signal": -exitcode if isinstance(exitcode, int) and exitcode < 0 else None,
                },
            )
            if w.busy_task_id and isinstance(meta, dict) and meta.get("task"):
                crashed_tasks.append({"task_id": w.busy_task_id, "task_type": task_info.get("type") if isinstance(task_info, dict) else None})
                append_jsonl(
                    DRIVE_ROOT / "logs" / "supervisor.jsonl",
                    {
                        "ts": utc_now_iso(),
                        "type": "worker_crash_task_dump",
                        "worker_id": wid,
                        "task": meta["task"],
                        "started_at": meta.get("started_at"),
                        "last_heartbeat_at": meta.get("last_heartbeat_at"),
                        "attempt": meta.get("attempt"),
                    },
                )
            if w.busy_task_id and w.busy_task_id in RUNNING:
                meta = RUNNING.pop(w.busy_task_id) or {}
                try:
                    from ouroboros.tools.services import archive_task_service_logs
                    task_for_roots = meta.get("task") if isinstance(meta, dict) and isinstance(meta.get("task"), dict) else {}
                    archive_task_service_logs(pathlib.Path(DRIVE_ROOT), str(w.busy_task_id), task_for_roots)
                except Exception:
                    log.debug("Failed to archive service logs for task %s", w.busy_task_id, exc_info=True)
                task = meta.get("task") if isinstance(meta, dict) else None
                if isinstance(task, dict):
                    task_type = str(task.get("type") or "")
                    # A negative exitcode means the worker died from a signal
                    # (SIGSEGV/SIGBUS/SIGABRT/SIGKILL). These are deterministic
                    # infrastructure crashes: retrying the same runtime path
                    # reproduces them and only burns budget, so they are terminal
                    # for EVERY task type (not just deep_self_review).
                    is_crash_signal = isinstance(exitcode, int) and exitcode < 0
                    crash_signal = -exitcode if is_crash_signal else None
                    chat_id = int(task.get("chat_id") or 1)
                    attempt = int(task.get("_attempt") or 1)
                    # Reconstruct cost/rounds from durable llm_usage for any
                    # abnormal-termination rollup below (worker died pre-finalize,
                    # so the event would otherwise carry zeros).
                    r_cost, r_rounds, r_prompt, r_completion = reconstruct_task_cost(str(w.busy_task_id))

                    # Already terminal via inline/direct-chat path? Leave it.
                    already_done = False
                    existing_status = ""
                    try:
                        from ouroboros.task_results import load_task_result, _TRULY_TERMINAL_STATUSES
                        existing = load_task_result(DRIVE_ROOT, str(w.busy_task_id))
                        if existing and str(existing.get("status") or "") in _TRULY_TERMINAL_STATUSES:
                            already_done = True
                            existing_status = str(existing.get("status") or "")
                            log.info(
                                "Skipping requeue for task %s — already in terminal state: %s",
                                w.busy_task_id, existing.get("status"),
                            )
                    except Exception:
                        log.debug("Failed to check existing result for %s", w.busy_task_id, exc_info=True)

                    if already_done:
                        # Terminal on disk but the worker died — its normal task_done
                        # event may have been lost with it. Emit an (idempotent)
                        # terminal event so the live card resolves instead of
                        # spinning until reconnect/history reconciliation.
                        _emit_task_done_terminal(task, str(w.busy_task_id), existing_status or "completed")
                    elif is_crash_signal or attempt > QUEUE_MAX_RETRIES:
                        deep = task_type == "deep_self_review"
                        if is_crash_signal:
                            log.warning(
                                "Task %s worker crashed with signal %s — terminal (no retry)",
                                w.busy_task_id, crash_signal,
                            )
                            result_text = (
                                f"❌ {'Deep self-review ' if deep else ''}worker process crashed "
                                f"(signal {crash_signal}). This is an infrastructure/platform crash "
                                "and is not retried automatically. "
                                + (
                                    "Use /restart and then /review to retry after a clean restart."
                                    if deep else
                                    "Use /restart and try again; if it recurs it is a platform-level issue."
                                )
                            )
                            reason_code = "worker_crash_signal"
                        else:
                            log.warning(
                                "Task %s exceeded crash retry limit (%d/%d) — marking failed",
                                w.busy_task_id, attempt, QUEUE_MAX_RETRIES,
                            )
                            result_text = (
                                f"❌ Task failed after {attempt} crash(es) (exit {exitcode}). "
                                "Worker process died repeatedly — likely a platform-level issue. "
                                "Please try again or use a different approach."
                            )
                            reason_code = "worker_crash_retry_exhausted"
                        try:
                            from ouroboros.task_results import STATUS_FAILED, write_task_result
                            write_task_result(
                                DRIVE_ROOT, str(w.busy_task_id), STATUS_FAILED,
                                result=result_text,
                                reason_code=reason_code,
                                outcome_axes=terminal_outcome_axes(lifecycle=STATUS_FAILED, execution=EXECUTION_INFRA_FAILED, reason_code=reason_code, review_trigger="worker_terminal"),
                                crash_signal=crash_signal,
                                crash_exitcode=exitcode if isinstance(exitcode, int) else None,
                                cost_usd=r_cost,
                                total_rounds=r_rounds,
                                prompt_tokens=r_prompt,
                                completion_tokens=r_completion,
                            )
                        except Exception:
                            log.debug("Failed to write failed status for %s", w.busy_task_id, exc_info=True)
                        # Message before task_done: otherwise the UI may close the card first.
                        try:
                            from supervisor.message_bus import get_bridge
                            bridge = get_bridge()
                            if bridge is not None:
                                if is_crash_signal and deep:
                                    user_msg = (
                                        f"❌ Deep self-review failed: worker process crashed (signal {crash_signal}). "
                                        "This is a known platform fork-safety limitation. "
                                        "Please use `/restart` and then `/review` to retry with a fresh process."
                                    )
                                elif is_crash_signal:
                                    user_msg = (
                                        f"❌ Task `{str(w.busy_task_id)[:8]}` failed: worker process crashed "
                                        f"(signal {crash_signal}). This is an infrastructure crash and was not retried."
                                    )
                                else:
                                    user_msg = (
                                        f"❌ Task `{str(w.busy_task_id)[:8]}` failed after {attempt} crash(es). "
                                        "Worker process crashed repeatedly. Please try again."
                                    )
                                bridge.send_message(chat_id, user_msg)
                        except Exception:
                            log.debug("Failed to send failure message for %s", w.busy_task_id, exc_info=True)
                        try:
                            get_event_q().put({
                                "type": "task_done",
                                "task_id": str(w.busy_task_id),
                                "task_type": task_type,
                                "chat_id": chat_id,
                                "status": "failed",
                                "reason_code": reason_code,
                                "outcome_axes": terminal_outcome_axes(lifecycle="failed", execution=EXECUTION_INFRA_FAILED, reason_code=reason_code, review_trigger="worker_terminal"),
                                "cost_usd": r_cost,
                                "total_rounds": r_rounds,
                                "prompt_tokens": r_prompt,
                                "completion_tokens": r_completion,
                            })
                        except Exception:
                            log.debug("Failed to emit terminal event for %s", w.busy_task_id, exc_info=True)
                    elif task_type == "evolution" and not bool(load_state().get("evolution_mode_enabled")):
                        # Evolution was stopped: do not resurrect a dead evolution
                        # worker into another cycle (mirrors the hard-timeout gate
                        # in queue.enforce_task_timeouts).
                        try:
                            from ouroboros.task_results import STATUS_CANCELLED, write_task_result
                            write_task_result(
                                DRIVE_ROOT, str(w.busy_task_id), STATUS_CANCELLED,
                                result="Evolution worker died after the campaign was stopped; not retried.",
                                reason_code="evolution_stopped_no_retry",
                                outcome_axes=terminal_outcome_axes(lifecycle=STATUS_CANCELLED, execution="cancelled", reason_code="evolution_stopped_no_retry", review_trigger="worker_terminal"),
                                cost_usd=r_cost,
                                total_rounds=r_rounds,
                                prompt_tokens=r_prompt,
                                completion_tokens=r_completion,
                            )
                        except Exception:
                            log.debug("Failed to write cancelled status for %s", w.busy_task_id, exc_info=True)
                        _emit_task_done_terminal(
                            task, str(w.busy_task_id), "cancelled",
                            cost_usd=r_cost, total_rounds=r_rounds,
                            prompt_tokens=r_prompt, completion_tokens=r_completion,
                        )
                    else:
                        task = dict(task)
                        task["_attempt"] = attempt + 1
                        try:
                            from ouroboros.task_results import STATUS_INTERRUPTED, write_task_result
                            write_task_result(
                                DRIVE_ROOT, str(w.busy_task_id), STATUS_INTERRUPTED,
                                result=f"Worker process died mid-task (attempt {attempt}). Retrying.",
                                cost_usd=r_cost,
                                total_rounds=r_rounds,
                                prompt_tokens=r_prompt,
                                completion_tokens=r_completion,
                            )
                        except Exception:
                            log.debug("Failed to write interrupted status for %s", w.busy_task_id, exc_info=True)
                        queue.enqueue_task(task, front=True)
            respawn_worker(wid)
            queue.persist_queue_snapshot(reason="worker_respawn_after_crash")

    now = time.time()
    alive_now = sum(1 for w in WORKERS.values() if w.proc.is_alive())
    if dead_detections:
        # Only count busy crashes or all-workers-dead as storm signals.
        if busy_crashes > 0 or alive_now == 0:
            CRASH_TS.extend([now] * max(1, dead_detections))
        else:
            CRASH_TS.clear()

    CRASH_TS[:] = [t for t in CRASH_TS if (now - t) < 60.0]
    if len(CRASH_TS) >= 3:
        # Do not execv on crash storms; keep direct-chat mode alive.
        st = load_state()
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "crash_storm_detected",
                "crash_count": len(CRASH_TS),
                "worker_count": len(WORKERS),
                "crashed_tasks": crashed_tasks,
            },
        )
        if st.get("owner_chat_id"):
            send_with_budget(
                int(st["owner_chat_id"]),
                "⚠️ Frequent worker crashes. Multiprocessing workers disabled, "
                "continuing in direct-chat mode (threading).",
            )
        kill_workers()
        CRASH_TS.clear()
