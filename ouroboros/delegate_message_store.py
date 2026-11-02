"""Write-ahead live-message frames and the starter's per-run message journal.

The indexed current.sqlite set for one run owns both retry bytes and journal
rows. No event-log replay, mailbox write, wake predicate or engine cursor is
involved. Rows are retained with custody, including ambiguous deliveries.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from ouroboros import store
from ouroboros.delegate_custody import custody_root
from ouroboros.install_identity import require_owned_root
from ouroboros.utils import utc_now_iso


def _set(run_id: str) -> str:
    return "delegate_messages-" + hashlib.sha256(run_id.encode("utf-8")).hexdigest()


def _wire(frame: dict, text: str) -> str:
    # One JSON value: text cannot close the host frame or replace its fields.
    return json.dumps({"kind": "task_message", "frame": frame, "text": text},
                      ensure_ascii=False, sort_keys=True)


def prepare(ctx: Any, entry: Any, relation: str, text: str, message_id: str,
            reply_to: dict | None) -> dict:
    """Commit before POST; an explicit key must replay an existing exact row."""
    root = custody_root(ctx)
    require_owned_root(root)
    sender = str(getattr(ctx, "task_id", "") or "")
    from ouroboros.task_status import load_effective_task_result

    own = load_effective_task_result(root, sender) or {}
    meta = getattr(ctx, "task_metadata", {}) or {}
    binding = meta.get("configured_subagent") or own.get("configured_subagent") or {}
    if not isinstance(binding, dict):
        binding = {}
    with store.open_store(root, write=True) as txn:
        if message_id:
            row = txn.get_row(_set(entry.run_id), message_id)
            if row is None:
                raise ValueError("message_identity_unknown")
            frame = row.get("frame", {})
            if (frame.get("message_id") != message_id or row.get("run_id") != entry.run_id
                    or frame.get("source_task_id") != sender or row.get("text") != text
                    or frame.get("reply_to") != reply_to
                    or row.get("starter_task_id") != entry.task_id
                    or row.get("wire_text") != _wire(frame, text)
                    or hashlib.sha256(row["wire_text"].encode("utf-8")).hexdigest() != row.get("wire_sha256")):
                raise ValueError("message_frame_conflict")
            return row
        mid = uuid.uuid4().hex
        frame = {
            "message_id": mid, "source_task_id": sender,
            "root_task_id": entry.root_task_id or own.get("root_task_id") or meta.get("root_task_id") or entry.task_id,
            "sender_role": str(own.get("delegation_role") or meta.get("delegation_role")
                               or ("subagent" if own.get("parent_task_id") or meta.get("parent_task_id")
                                   else "root" if own else "unknown")),
            "row": str(binding.get("selected_subagent_id") or meta.get("subagent_id")
                       or meta.get("selected_subagent_id") or ""),
            "model": str(getattr(ctx, "active_model_override", "") or getattr(ctx, "active_model", "")
                         or getattr(ctx, "task_model_override", "") or meta.get("model") or ""),
            "ts": utc_now_iso(), "relation": relation,
        }
        if reply_to is not None:
            frame["reply_to"] = reply_to
        wire = _wire(frame, text)
        row = {"schema": 1, "run_id": entry.run_id, "starter_task_id": entry.task_id,
               "frame": frame, "text": text, "wire_text": wire,
               "wire_sha256": hashlib.sha256(wire.encode("utf-8")).hexdigest(),
               "outcome": "prepared", "note": "Prepared before send; delivery is not yet proven."}
        txn.put_row(_set(entry.run_id), mid, row)
    return row


def record_outcome(ctx: Any, run_id: str, message_id: str, outcome: str, reason: str) -> None:
    with store.open_store(custody_root(ctx), write=True) as txn:
        row = txn.get_row(_set(run_id), message_id)
        if row is not None:
            row.update(outcome=outcome, reason=reason)
            row.pop("note", None)
            txn.put_row(_set(run_id), message_id, row)


def journal(ctx: Any, run_id: str) -> list:
    """The starter's own journal only; a message address grants no result access."""
    sender = str(getattr(ctx, "task_id", "") or "")
    with store.open_store(custody_root(ctx), write=False) as txn:
        rows = txn.set_rows(_set(run_id))
    return [{key: value for key, value in row.items() if key not in {"wire_text", "wire_sha256"}}
            for row in rows.values() if row.get("starter_task_id") == sender
            and row.get("frame", {}).get("source_task_id") != sender]
