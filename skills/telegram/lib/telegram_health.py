"""Telegram bridge: health snapshot + read-only task list panels."""
from __future__ import annotations

import pathlib
import shutil
import time
from typing import Dict

from ouroboros import i18n_memory as _memory
from ouroboros.queue_observation import load_queue_document, pool_observation_status

from .telegram_api import _LOCALIZED_TEXTS
from .telegram_i18n import Index, drive_root, host_sentence
from .telegram_state import _data_dir, _jsonl_tail


_HEALTH_EN = {
    "queue": "Queue", "works": "Works", "idle": "idle", "workers": "Workers", "not_observed": "not observed",
    "disk": "Disk", "logs": "logs", "incidents": "Incidents (1h)", "clean": "clean",
    "free": "free", "tasks_idle": "🟢 idle — nothing running",
    "tasks_unknown": "❔ unknown — the list of works could not be read",
    "unknown": "unknown",
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


def _collect_health(api, lang: str = "en", state=None) -> str:
    """Health snapshot — the works of ``state`` (the server's ``/api/state`` body), the queue's counts, workers,
    incidents, disk. Read-only."""
    s = _HEALTH[lang]
    data = _data_dir(api)
    out = []
    snap = load_queue_document(data)  # the pool's counts as certified on queue.alive, as every queue reader sees them
    listed = snap.get("running") if isinstance(snap.get("running"), list) else []
    rc = max(0, int(snap.get("running_count") or 0) - (len(listed) - len(_owner_running(snap))))
    pc = int(snap.get("pending_count") or 0)
    unread = bool(snap.get("_snapshot_missing") or snap.get("_snapshot_invalid"))  # a lost document is no zero
    rows = s["unknown"] if unread else f"RUNNING {rc} · PENDING {pc}"
    said = _owner_works(state)  # the window's works: an unread projection is unknown, never idle
    q_icon = "❔" if said is None else "🟢" if not said else "🟡"
    q_state = s["unknown"] if said is None else s["idle"] if not said else " · ".join(
        f"{sum(w['state'] == one['state'] for w in said)} {_chip_word(one, lang)}"
        for one in {w["state"]: w for w in said}.values())
    out.append(f"⚙️ {s['works']}: {q_icon} {q_state} · {s['queue']}: {rows}")
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


def _owner_works(state) -> list | None:
    """The window's works without a correspondent's Presence turn (its recorded ``place_turn``), as the RUNNING count
    leaves it out (``_owner_running``, 02.11 ``99fee149788``); ``None`` when the projection was not read."""
    try:  # a kept bridge copy over an older core without the renderer: the panels say unknown, the bridge runs
        from ouroboros.gateway.work_status import said_works
    except ImportError:
        return None
    said = said_works(state)
    return None if said is None else [work for work in said if not work.get("place_turn")]


def _chip_word(work: dict, lang: str) -> str:
    """The window's own word: its chip's catalogue entry (``code:task.chip.*``), English when the memory has none."""
    key = work.get("chip_key")
    return (_memory.tr("code:" + key, lang, work["word"], drive_root=drive_root()) if key else None) or work["word"]


def _reason_words(work: dict, lang: str) -> str:
    """The reason as the window says it: its catalogue entry ``code:work.because.<code>``, else by its text."""
    by_code = _memory.tr("code:work.because." + work["reason_code"], lang, None, drive_root=drive_root()) \
        if work.get("reason_code") else None
    return by_code or host_sentence(work["reason"], lang)


_STATE_ICONS = {"working": "🟡", "finalizing": "🟡", "pausing": "🟡", "queued": "⏳", "waiting": "⏳",
                "question": "❓", "paused": "⏸", "unknown": "❔"}


TELEGRAM_TEXT_LIMIT = 4096  # Telegram's message limit, counted in UTF-16 code units


def _units(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def _cut(text: str, room: int) -> str:
    """``text`` within ``room`` UTF-16 code units, a cut one said with an ellipsis (an astral character is two)."""
    if _units(text) <= room:
        return text
    used, keep = 1, []
    for ch in text:
        used += 2 if ord(ch) > 0xFFFF else 1
        if used > room:
            break
        keep.append(ch)
    return "".join(keep) + "…" if room > 0 else ""


def _collect_tasks_text(state, lang: str = "en", limit: int = TELEGRAM_TEXT_LIMIT) -> str:
    """The works the window shows, from the server's ``/api/state`` body: each work's name, its word and its first
    reason in the server's own sentences. An unread or incomplete projection is unknown. At most twelve lines, and
    with the ``… +N`` line never past ``limit`` UTF-16 code units (Telegram drops a longer message whole)."""
    s = _HEALTH[lang]
    said = _owner_works(state)
    if said is None or not said:
        return s["tasks_unknown" if said is None else "tasks_idle"]
    lines: list = []
    for work in said[:12]:
        word = _chip_word(work, lang)
        reason = _reason_words(work, lang) if work["reason"] else ""
        reason = f": {reason}" if reason and reason.strip().casefold() != word.strip().casefold() else ""
        line = f"{_STATE_ICONS.get(work['state'], '❔')} {work['name'] or '—'} — {word}{reason}"
        more = len(said) - len(lines) - 1
        tail = f"\n… +{more}" if more else ""
        used = _units("\n".join(lines + [line]) + tail)
        if used > limit:
            room = limit - _units("\n".join(lines + [""]) + f"\n… +{len(said) - len(lines)}")
            if lines or room < 2:
                break
            line = _cut(line, room)  # one line longer than the page: cut, said with an ellipsis
        lines.append(line)
    if len(said) > len(lines):
        lines.append(f"… +{len(said) - len(lines)}")
    return "\n".join(lines)


def _build_menu_tasks(state, command_mode: str, lang: str = "en") -> tuple[str, list[list[dict]]]:
    """Read-only list of works (``state``: the ``/api/state`` body). No inline mutations (owner inject-minimization)."""
    t = _LOCALIZED_TEXTS[lang]
    # the header leaves the body at least its shortest form (the ``… +N`` line), else the title is cut;
    # the last cut bites only on a catalogue sentence longer than the page
    title = _cut(t["tasks_title"], TELEGRAM_TEXT_LIMIT - 2 - _units(_collect_tasks_text(state, lang, 0)))
    header = _cut(f"{title}\n\n" + _collect_tasks_text(state, lang, TELEGRAM_TEXT_LIMIT - _units(f"{title}\n\n")),
                  TELEGRAM_TEXT_LIMIT)
    keyboard = [
        [{"text": t["btn_refresh"], "callback_data": "nav:tasks"}],
        [{"text": t["btn_back"], "callback_data": "nav:menu"}],
    ]
    return header, keyboard
