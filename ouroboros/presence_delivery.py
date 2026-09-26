"""Receipt-backed dialogue writes; provider custody stays with the transport."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from ouroboros.utils import iter_jsonl_objects, jsonl_chain_handles, utc_now_iso

DELIVERY_VERSION = 1
_FIELDS = frozenset({
    "schema_version", "delivery_id", "part_id", "state", "provider", "account_id",
    "conversation_id", "thread_id", "text", "format", "message", "origin",
})
_STATES = frozenset({"delivered", "accepted", "failed", "uncertain"})


class PresenceDeliveryConflict(ValueError):
    """A retained receipt identity already names different provider facts."""


def delivery_reporting_version(value: Any) -> int:
    if type(value) is not int or value not in (0, DELIVERY_VERSION):
        raise ValueError("delivery_reporting_version must be 0 or 1")
    return value


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def validate_delivery(payload: Any) -> dict[str, Any]:
    """Accept facts only, without interpreting provider text or granting power."""
    if not isinstance(payload, dict) or set(payload) != _FIELDS:
        raise ValueError("invalid presence delivery fields")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != DELIVERY_VERSION:
        raise ValueError("presence delivery schema_version must be 1")
    for key in _FIELDS - {"schema_version", "message", "origin"}:
        if not isinstance(payload[key], str):
            raise ValueError(f"presence delivery {key} must be a string")
    for key in ("delivery_id", "provider", "account_id", "conversation_id"):
        if not payload[key].strip():
            raise ValueError(f"presence delivery {key} is required")
    part = payload["part_id"]
    if part != "status" and not (part and part.isascii() and part.isdecimal()):
        raise ValueError("presence delivery part_id must be an index or status")
    if payload["state"] not in _STATES:
        raise ValueError("presence delivery state must be delivered, accepted, failed or uncertain")
    if not isinstance(payload["message"], dict):
        raise ValueError("presence delivery message must be an object")
    origin = payload["origin"]
    if not isinstance(origin, dict) or set(origin) - {"kind", "task_id", "source_event_id"}:
        raise ValueError("invalid presence delivery origin fields")
    if origin.get("kind") not in {"tool", "automatic"}:
        raise ValueError("presence delivery origin.kind must be tool or automatic")
    if any(not isinstance(value, str) for value in origin.values()):
        raise ValueError("presence delivery origin values must be strings")
    # Freeze caller facts before hashing/writing; arrival time is host-owned.
    return json.loads(_canonical(payload))


def _key(skill: str, payload: dict[str, Any]) -> tuple[str, ...]:
    return (skill, *(payload[key] for key in ("account_id", "delivery_id", "part_id", "state")))


def _payload_from_row(row: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    if row.get("type") != "presence_delivery":
        return None
    transport = row.get("transport") or {}
    delivery = transport.get("delivery") or {}
    skill = delivery.get("skill")
    if not isinstance(skill, str) or row.get("source") != f"skill:{skill}":
        raise ValueError("presence delivery history has invalid skill provenance")
    payload = {
        "schema_version": delivery.get("schema_version"),
        **{key: delivery.get(key) for key in ("delivery_id", "part_id", "state")},
        **{key: transport.get(key) for key in (
            "provider", "account_id", "conversation_id", "thread_id", "message", "origin",
        )},
        "text": row.get("text"), "format": row.get("format"),
    }
    return skill, validate_delivery(payload)


def _verified_task(data_dir: Path, skill: str, origin: dict[str, str]) -> tuple[str, dict[str, str]]:
    task_id = origin.get("task_id", "")
    if not task_id:
        return "", {}
    from ouroboros.dialogue_provenance import presence_provenance_from_task
    from ouroboros.task_results import load_task_result

    try:
        task = load_task_result(data_dir, task_id, strict=True) or {}
    except (OSError, ValueError):
        return "", {}
    provenance = presence_provenance_from_task(task)
    if provenance.get("transport_skill") != skill:
        return "", {}
    return task_id, provenance


class PresenceDeliveryRecorder:
    """One Host context's disposable projection over canonical chat history.

    Only this Host receipt writer owns these keys. Other chat writers and
    rotation do not invalidate its incrementally maintained projection. A new
    Host rebuilds once; an ambiguous required write discards the projection.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self._lock = threading.Lock()
        self._index: dict[tuple[str, ...], str] | None = None
        self._history_gapped = False

    def _rebuild(self) -> dict[tuple[str, ...], str]:
        index: dict[tuple[str, ...], str] = {}
        gaps: set[str] = set()
        with jsonl_chain_handles(self.data_dir / "logs" / "chat.jsonl", strict=True) as handles:
            for path, handle in handles:
                for row in iter_jsonl_objects(path, _handle=handle, gap_reasons=gaps):
                    try:
                        parsed = _payload_from_row(row)
                    except (TypeError, AttributeError, ValueError):
                        # A malformed retained receipt is also an unobserved
                        # interval, not proof that its identity never existed.
                        gaps.add("invalid_presence_delivery_row")
                        continue
                    if parsed is None:
                        continue
                    skill, payload = parsed
                    key = _key(skill, payload)
                    digest = hashlib.sha256(_canonical(payload)).hexdigest()
                    if key in index and index[key] != digest:
                        raise OSError("conflicting retained presence delivery receipts")
                    index[key] = digest
        self._history_gapped = bool(gaps)
        return index

    def record(self, skill: str, value: Any) -> dict[str, Any]:
        payload = validate_delivery(value)
        key = _key(skill, payload)
        digest = hashlib.sha256(_canonical(payload)).hexdigest()
        with self._lock:
            if self._index is None:
                self._index = self._rebuild()
            previous = self._index.get(key)
            if previous is not None:
                if previous != digest:
                    raise PresenceDeliveryConflict("presence delivery identity already has different facts")
                return {"ok": True, "recorded": True, "duplicate": True,
                        "history_coverage": "gapped" if self._history_gapped else "indexed"}

            from ouroboros.presence_bindings import conversation_key as presence_conversation_key
            from ouroboros.presence_runner import _stable_numeric_id
            from supervisor.message_bus import log_chat

            task_id, provenance = _verified_task(self.data_dir, skill, payload["origin"])
            conversation_key = presence_conversation_key(*(payload[key] for key in (
                "provider", "account_id", "conversation_id", "thread_id",
            )))
            delivery = {
                "schema_version": DELIVERY_VERSION, "skill": skill,
                **{key: payload[key] for key in ("delivery_id", "part_id", "state")},
                "reported_at": utc_now_iso(),
            }
            if provenance:
                delivery["presence_provenance"] = provenance
            transport = {
                **{key: payload[key] for key in (
                    "provider", "account_id", "conversation_id", "thread_id", "message", "origin",
                )},
                "conversation_key": conversation_key, "delivery": delivery,
            }
            try:
                # log_chat owns the append/rotation lock; never acquire it here.
                log_chat(
                    "out" if payload["state"] in {"delivered", "accepted"} else "system",
                    _stable_numeric_id("presence-conversation", conversation_key), 0,
                    payload["text"], fmt=payload["format"], source=f"skill:{skill}",
                    client_message_id="presence-delivery:" + hashlib.sha256(_canonical({"key": key})).hexdigest(),
                    transport=transport, task_id=task_id, record_type="presence_delivery",
                    drive_root=self.data_dir, require_write=True, ensure_record_boundary=True,
                )
            except Exception:
                # The append may have landed before the failure reached us.
                self._index = None
                self._history_gapped = False
                raise
            self._index[key] = digest
            return {"ok": True, "recorded": True, "duplicate": False,
                    "history_coverage": "gapped" if self._history_gapped else "indexed"}
