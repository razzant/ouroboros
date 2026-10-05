"""Confirmed pause disclosures, retained on the task until local chat records them.

The existing pause writer registers an episode in the SAME result update as
confirmation. Resume and later pauses retain that duty. The off-loop maintenance
owner uses the saved chat row as its receipt (the upgrade-notice pattern), never
as proof of external delivery. No separate outbox or lifecycle authority exists.
A process-local candidate set avoids rescanning history after bootstrap.
"""
from __future__ import annotations

import logging
import pathlib
import threading
from typing import Any

from ouroboros.utils import iter_jsonl_chain_objects, update_json_locked, utc_now_iso

log = logging.getLogger(__name__)
_LOCK = threading.Lock()
_HINT_LOCK = threading.Lock()
_KNOWN: dict[str, dict[str, object]] = {}
_BOOTSTRAPPED: set[str] = set()
_RECORDED: dict[str, set[tuple[int, str]]] = {}


def notice_fields(current: dict, root: Any, task_id: str, episode_id: str, reason: str) -> dict:
    """Called inside the result writer, only for a newly confirmed pause episode."""
    if not episode_id or reason not in {"budget", "owner"}:
        return {}
    identity = f"pause:{task_id}:{episode_id}"
    pending = dict(current.get("pause_notices") or {})
    pending.setdefault(identity, {"reason": reason, "confirmed_at": utc_now_iso()})
    return {"pause_notices": pending}


def track(root: Any, task_id: str) -> None:
    """Publish a hint AFTER the result commit; concurrent hints cannot be consumed by an older read."""
    with _HINT_LOCK:
        _KNOWN.setdefault(str(pathlib.Path(root).resolve()), {})[task_id] = object()


def _forget(key: str, task_id: str, token: object) -> None:
    with _HINT_LOCK:
        if _KNOWN.get(key, {}).get(task_id) is token:
            _KNOWN[key].pop(task_id)


def _recorded(root: pathlib.Path) -> set[tuple[int, str]]:
    key = str(root)
    if key not in _RECORDED:
        found = set()
        for row in iter_jsonl_chain_objects(root / "logs" / "chat.jsonl"):
            if row.get("direction") == "system" and row.get("type") == "task_pause_notice":
                identity = str(row.get("card_row_id") or "")
                if identity and isinstance(row.get("chat_id"), int):
                    found.add((row["chat_id"], identity))
        _RECORDED[key] = found
    return _RECORDED[key]


def _acknowledge(root: pathlib.Path, task_id: str, identity: str, notice: dict) -> None:
    from ouroboros.task_results import require_writable_task_result_schema, task_result_path

    def update(current: dict):
        require_writable_task_result_schema(current)
        pending = dict(current.get("pause_notices") or {})
        if pending.get(identity) != notice:
            return None
        pending.pop(identity)
        updated = {**current, "pause_notices": pending}
        if not pending:
            updated.pop("pause_notices")
        return updated

    update_json_locked(task_result_path(root, task_id), update, strict_existing_dict=True)


def reconcile_pause_notices(drive_root: Any, *, stop_requested=None) -> None:
    """The existing off-loop maintenance retries local publication, never Resume."""
    from ouroboros.task_result_facts import raw_result_facts
    from ouroboros.task_results import load_task_result, task_results_dir
    from supervisor.log_addressing import address_task_event
    from supervisor import message_bus

    root = pathlib.Path(drive_root).resolve()
    if message_bus.DATA_DIR is None or pathlib.Path(message_bus.DATA_DIR).resolve() != root:
        return  # The shared message writer has not bound this installation yet.
    if not _LOCK.acquire(blocking=False):
        return
    try:
        key = str(root)
        if key not in _BOOTSTRAPPED:
            facts, malformed = raw_result_facts(task_results_dir(root, create=False))
            for name in set(malformed) | {name for name, fact in facts.items() if fact.get("pause_notice_pending")}:
                track(root, pathlib.Path(name).stem)
            _BOOTSTRAPPED.add(key)  # unreadable candidates retry individually, not the whole history
        with _HINT_LOCK:
            candidates = tuple(_KNOWN.get(key, {}).items())
        for task_id, token in candidates:
            if stop_requested is not None and stop_requested():
                return
            try:
                result = load_task_result(root, task_id, strict=True)
                if not result or not result.get("pause_notices"):
                    _forget(key, task_id, token)
                    continue
                event = address_task_event({task_id: {"task": result}}, root, {"task_id": task_id})
                chat_id = message_bus.notification_chat_route(event.get("chat_id"))
                if chat_id is None:
                    continue  # No destination is not permission to default to Main.
                receipts = _recorded(root)
                for identity, notice in dict(result["pause_notices"]).items():
                    if stop_requested is not None and stop_requested():
                        return
                    if not isinstance(notice, dict):
                        continue
                    if (chat_id, identity) not in receipts:
                        cause = "the owner's Pause" if notice.get("reason") == "owner" else "its budget limit"
                        text = (f"The task paused because of {cause}. "
                                "Its work is retained. Use Resume on the task when available.")
                        message_bus.send_with_budget(
                            chat_id, text, task_id=task_id, role="system", system_type="task_pause_notice",
                            narration=False, require_write=True, ensure_record_boundary=True,
                            progress_meta={**{k: event[k] for k in ("root_task_id", "parent_task_id") if k in event},
                                           "card_row": "timeline", "card_row_id": identity},
                        )
                        receipts.add((chat_id, identity))
                    if stop_requested is not None and stop_requested():
                        return
                    _acknowledge(root, task_id, identity, notice)
                if not (load_task_result(root, task_id, strict=True) or {}).get("pause_notices"):
                    _forget(key, task_id, token)
            except Exception:
                # append-before-ack (including a live transport error after append)
                # is recovered from the real saved row, not by trusting a failed send.
                _RECORDED.pop(key, None)
                log.warning("Pause notice for %s remains owed to local history", task_id, exc_info=True)
    finally:
        _LOCK.release()
