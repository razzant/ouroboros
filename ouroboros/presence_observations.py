"""Attributed transport-inbox observations for a continuing Presence author.

The bridge owns its inbox. Initial facts arrive with the admitted event; refreshes
use the existing binding-scoped work route. Full refresh bytes live in this task's
existing artifact store before its latest pointer is replaced under the task lock.
Reentry reads that pointer once: it is a dated observation, not current inbox truth,
canonical chat history, an owner instruction, or delivery evidence. No polling or
queue lifecycle lives here. Task artifact retention keeps previous snapshots too.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any


def _observed_at(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("transport_queue observed_at must be an ISO timestamp with timezone")
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("transport_queue observed_at needs a timezone")
    return stamp


def validate_transport_queue(value: Any, conversation_key: str, source_event_id: str) -> dict:
    """Admit only a full, source-bound snapshot; a clipped preview is not a source."""
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("transport_queue schema_version must be 1")
    if (value.get("conversation_key") != conversation_key
            or value.get("after_source_event_id") != source_event_id):
        raise ValueError("transport_queue source does not match this Presence event")
    _observed_at(value.get("observed_at"))
    events = value.get("events")
    if (not isinstance(events, list) or value.get("complete") is not True
            or type(value.get("pending_count")) is not int or value["pending_count"] != len(events)
            or type(value.get("omitted_count")) is not int or value["omitted_count"] != 0
            or not isinstance(value.get("source"), str) or not value["source"].strip()):
        raise ValueError("transport_queue needs full events, complete=true, matching pending_count and omitted_count=0")
    identities = set()
    for event in events:
        if (not isinstance(event, dict) or not isinstance(event.get("source_event_id"), str)
                or not event["source_event_id"] or not isinstance(event.get("text"), str)
                or event.get("text_truncated", False) is not False
                or ("text_chars" in event and (type(event["text_chars"]) is not int
                                               or event["text_chars"] != len(event["text"])))):
            raise ValueError("transport_queue events need source_event_id and full, untruncated text")
        if event["source_event_id"] in identities:
            raise ValueError("transport_queue has duplicate source_event_id")
        identities.add(event["source_event_id"])
    # A JSON copy prevents callers mutating a snapshot after validation/publication.
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def transport_queue_observation(root: Path, task_id: str, event: Any) -> dict:
    """Read the latest retained observation, falling back only when no refresh exists."""
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.task_results import load_task_result

    try:
        stored = load_task_result(root, task_id, strict=True) or {}
        pointer = stored.get("presence_transport_queue")
        if pointer is not None:
            snapshot = json.loads(read_actor_source_bytes(root, task_id, pointer["source_ref"]))
            receipt = {"received_at": pointer["received_at"], "source_ref": pointer["source_ref"]}
        else:
            snapshot = (getattr(event, "conversation", None) or {}).get("transport_queue")
            receipt = {"source": "initial_event"}
        if snapshot is None:
            return {"status": "unavailable", "reason": "transport_has_not_reported_queue"}
        return {"status": "available", **receipt, "snapshot": validate_transport_queue(
            snapshot, event.conversation_key, event.source_event_id)}
    except (OSError, ValueError, TypeError, KeyError):
        # An unreadable latest snapshot never resurrects an older queue as current.
        return {"status": "unavailable", "reason": "transport_queue_source_unreadable"}


def record_transport_queue(root: Path, task_id: str, binding_id: str, snapshot: Any) -> dict:
    """Write full observation before its pointer, preserving task/output/receipt fields."""
    from ouroboros.artifacts import read_actor_source_bytes, store_actor_source_bytes
    from ouroboros.task_results import load_task_result, require_writable_task_result_schema, task_result_path
    from ouroboros.utils import update_json_locked, utc_now_iso

    stored = load_task_result(root, task_id, strict=True) or {}
    presence = (stored.get("metadata") or {}).get("presence") or {}
    event = presence.get("event") or {}
    if presence.get("binding_id") != binding_id or not event.get("conversation_key"):
        from ouroboros.presence_runner import PresenceTurnError

        raise PresenceTurnError("presence_work_not_found", "work_ref")
    value = validate_transport_queue(snapshot, event["conversation_key"], event.get("source_event_id", ""))
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ref = store_actor_source_bytes(root, task_id, category="context_checkpoints",
                                   source_id="presence-transport-queue", data=raw, extension="json")
    if read_actor_source_bytes(root, task_id, ref) != raw:
        raise ValueError("transport_queue source did not read back")
    pointer = {"source_ref": ref, "observed_at": value["observed_at"], "received_at": utc_now_iso()}
    answer = {"ok": True, "status": "recorded", "observed_at": value["observed_at"]}

    def update(current: dict) -> dict:
        require_writable_task_result_schema(current)
        if ((current.get("metadata") or {}).get("presence") or {}).get("binding_id") != binding_id:
            raise ValueError("transport_queue binding changed")
        old = current.get("presence_transport_queue")
        initial = (event.get("conversation") or {}).get("transport_queue")
        previous = old.get("observed_at") if isinstance(old, dict) else (initial or {}).get("observed_at")
        if previous:
            order = _observed_at(value["observed_at"]) - _observed_at(previous)
            if order.total_seconds() < 0:
                answer.update(status="stale", observed_at=previous)
                return current
            if order.total_seconds() == 0:
                same = (old.get("source_ref", {}).get("sha256") == ref["sha256"] if old else initial == value)
                if not same:
                    raise ValueError("transport_queue conflicting observation at the same timestamp")
                answer["status"] = "duplicate"
                return current
        return {**current, "presence_transport_queue": pointer}

    update_json_locked(task_result_path(root, task_id), update, strict_existing_dict=True)
    if answer["status"] == "recorded":
        readback = (load_task_result(root, task_id, strict=True) or {}).get("presence_transport_queue") or {}
        # Another newer observation may have won after the lock was released.
        if not readback or _observed_at(readback["observed_at"]) < _observed_at(value["observed_at"]):
            raise OSError("transport_queue pointer did not read back")
    return answer
