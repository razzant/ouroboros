"""Telegram bridge: health snapshot + read-only task list panels."""
from __future__ import annotations

import pathlib
import shutil
import time
from typing import Dict

from ouroboros.queue_observation import pool_observation_status

from .telegram_api import _LOCALIZED_TEXTS
from .telegram_i18n import Index
from .telegram_state import _data_dir, _jsonl_tail, _read_json_file


_HEALTH_EN = {
    "queue": "Queue", "idle": "idle", "busy": "working", "workers": "Workers", "not_observed": "not observed",
    "disk": "Disk", "logs": "logs", "incidents": "Incidents (1h)", "clean": "clean",
    "free": "free", "tasks_idle": "🟢 idle — nothing running",
    "tail_none": "none in bounded tail",
    "tail_limited": "earlier coverage may be incomplete",
    "gb": "GB", "mb": "MB",
    # supervisor.jsonl event types worth surfacing as "incidents", by type.
    "incident_worker_event_handler_error": "handler errors",
    "incident_orphaned_workers_reaped": "orphans reaped",
    "incident_zombie_prevention_cleanup": "zombie cleanup",
    "incident_worker_dead_detected": "worker deaths",
}
_HEALTH = Index("health", _HEALTH_EN, "the bridge's health snapshot: queue, workers, incidents, disk")
_HEALTH_INCIDENTS = tuple(key[len("incident_"):] for key in _HEALTH_EN if key.startswith("incident_"))
_SUPERVISOR_TAIL_BYTES = 256 * 1024


def _started_processes(snap: dict) -> int:
    """Every started worker process: the project slots (0 at idle) and the conversation lane's carriers."""
    return int(snap.get("worker_total") or 0) + int(snap.get("lane_carrier_total") or 0)


def _dir_size_mb(path: pathlib.Path) -> float:
    total = 0
    try:
        for f in pathlib.Path(path).glob("*"):
            try:
                total += f.stat().st_size
            except OSError:
                continue
    except Exception:
        return 0.0
    return total / (1024 * 1024)


def _recent_incidents(api, window_sec: float = 3600.0) -> tuple[Dict[str, int], bool]:
    """Count supervisor.jsonl incident events within the window (best-effort, file-only)."""
    counts: Dict[str, int] = {}
    events, omitted_prefix = _jsonl_tail(
        _data_dir(api) / "logs" / "supervisor.jsonl",
        max_entries=800,
        tail_bytes=_SUPERVISOR_TAIL_BYTES,
    )
    now = time.time()
    for e in events:
        typ = str(e.get("type") or "")
        if typ not in _HEALTH_INCIDENTS:
            continue
        ts = str(e.get("ts") or "")
        try:
            from datetime import datetime, timezone
            parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            ev = (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp()
            if now - ev > window_sec:
                continue
        except Exception:
            pass  # unparseable ts → include
        counts[typ] = counts.get(typ, 0) + 1
    return counts, omitted_prefix


def _presence_row(row: dict) -> bool:
    """A correspondent's carried Presence turn: its place's conversation, never the owner's running work.

    Mirrors ``ouroboros.task_carrier.execution_kind`` on the published row (its execution owner's kind,
    else the legacy ``_presence_turn`` flag)."""
    task = row.get("task") if isinstance(row.get("task"), dict) else row
    owner = task.get("_execution_owner") if isinstance(task.get("_execution_owner"), dict) else {}
    kind = owner.get("kind")
    return kind == "presence" if kind in {"direct", "presence", "pooled"} else bool(task.get("_presence_turn"))


def _owner_running(snap: dict) -> list:
    rows = snap.get("running") if isinstance(snap.get("running"), list) else []
    return [row for row in rows if not (isinstance(row, dict) and _presence_row(row))]


def _collect_health(api, lang: str = "en") -> str:
    """Health snapshot from durable files — queue, workers, incidents, disk. Read-only."""
    s = _HEALTH[lang]
    data = _data_dir(api)
    out = []
    snap = _read_json_file(data / "state" / "queue_snapshot.json")
    listed = snap.get("running") if isinstance(snap.get("running"), list) else []
    rc = max(0, int(snap.get("running_count") or 0) - (len(listed) - len(_owner_running(snap))))
    pc = int(snap.get("pending_count") or 0)
    q_icon = "🟢" if (rc + pc) == 0 else "🟡"
    q_state = s["idle"] if (rc + pc) == 0 else s["busy"]
    out.append(f"⚙️ {s['queue']}: {q_icon} {q_state} (RUNNING {rc} · PENDING {pc})")
    if snap and pool_observation_status(snap) == "unknown":  # a failed pool measurement is not 0 processes
        out.append(f"👷 {s['workers']}: {s['not_observed']}")
    elif isinstance(snap.get("worker_total"), int):
        out.append(f"👷 {s['workers']}: {_started_processes(snap)}")
    inc, incidents_limited = _recent_incidents(api)
    if inc:
        parts = [f"{v} {s[f'incident_{k}']}" for k, v in inc.items()]
        if incidents_limited:
            parts.append(s["tail_limited"])
        out.append(f"⚠️ {s['incidents']}: " + " · ".join(parts))
    elif incidents_limited:
        out.append(f"⚠️ {s['incidents']}: {s['tail_none']} · {s['tail_limited']}")
    else:
        out.append(f"✅ {s['incidents']}: {s['clean']}")
    try:
        free_gb = shutil.disk_usage(data).free / (1024 ** 3)
        logs_mb = _dir_size_mb(data / "logs")
        unit = s["gb"]
        munit = s["mb"]
        out.append(f"💾 {s['disk']}: {s['free']} {free_gb:.0f} {unit} · {s['logs']} {logs_mb:.0f} {munit}")
    except Exception:
        pass
    return "\n".join(out)


def _collect_tasks_text(api, lang: str = "en") -> str:
    """Read-only list of running/pending tasks from queue_snapshot.json."""
    s = _HEALTH[lang]
    snap = _read_json_file(_data_dir(api) / "state" / "queue_snapshot.json")
    running = _owner_running(snap)
    pending = snap.get("pending") if isinstance(snap.get("pending"), list) else []
    if not running and not pending:
        return s["tasks_idle"]

    def fmt(task, icon):
        task = task or {}
        tid = str(task.get("id") or task.get("task_id") or "")[:8]
        typ = str(task.get("type") or "task")
        role = str(task.get("delegation_role") or "")
        suffix = f" · {role}" if role and role != "root" else ""
        return f"{icon} {typ}{suffix}" + (f" [{tid}]" if tid else "")

    lines = [fmt(x, "🟡") for x in running[:12]] + [fmt(x, "⏳") for x in pending[:12]]
    total = len(running) + len(pending)
    if total > len(lines):
        lines.append(f"… +{total - len(lines)}")
    return "\n".join(lines)


def _build_menu_tasks(api, command_mode: str, lang: str = "en") -> tuple[str, list[list[dict]]]:
    """Read-only running/pending task list. No inline mutations (owner inject-minimization policy)."""
    t = _LOCALIZED_TEXTS[lang]
    title = t["tasks_title"]
    header = f"{title}\n\n{_collect_tasks_text(api, lang)}"
    keyboard = [
        [{"text": t["btn_refresh"], "callback_data": "nav:tasks"}],
        [{"text": t["btn_back"], "callback_data": "nav:menu"}],
    ]
    return header, keyboard
