"""Handoff schema migration and per-holder checkpoint projection."""
from __future__ import annotations

from copy import deepcopy
from typing import Any
import uuid

from ouroboros import delegate_supervision_state as checkpoints
from ouroboros.utils import utc_now_iso

IDENTITY_FIELDS = ("selected_subagent_id", "config_fingerprint", "authority_fingerprint",
                   "work_order_fingerprint", "snapshot_id", "execution_root", "baseline_sha", "target_root")
HOLDER_FIELDS = (*IDENTITY_FIELDS, "run_id", "pending_invocation_id", "actor_binding_source",
                 "settled_terminal", "journal_cursor", "interaction_acknowledged_ids", "pending_wake",
                 "checkpoint", "adopted_at", "deferred_reason", "veto_reason")


def normalize(data: Any) -> dict:
    if not isinstance(data, dict):
        raise ValueError("handoff is not an object")
    if not data:
        return {}
    schema = data.get("schema", 1)
    if schema not in (1, 2):
        raise ValueError("unknown handoff schema")
    row = deepcopy(data)
    if schema == 1:
        holder = {key: row.pop(key) for key in HOLDER_FIELDS if key in row}
        holder["status"] = row.get("status", "reserved")
        row.update(schema=2, holders=[holder])
    holders = row.get("holders")
    if not isinstance(holders, list) or not holders or any(
            not isinstance(h, dict) or not (h.get("run_id") or h.get("pending_invocation_id")) for h in holders):
        raise ValueError("invalid handoff holders")
    identities = [(h.get("run_id"), h.get("pending_invocation_id")) for h in holders]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate handoff holder")
    return row


def read_view(data: Any) -> dict:
    row = normalize(data)
    if row:
        # Legacy Python callers may still inspect one holder at the top level.
        # The persisted document always has a holders list after a mutation.
        primary = min(range(len(row["holders"])), key=lambda i: (
            bool(row["holders"][i].get("settled_terminal")), not bool(row["holders"][i].get("run_id"))))
        aliases = {k: deepcopy(v) for k, v in row["holders"][primary].items() if k in HOLDER_FIELDS}
        row["_primary_holder"] = primary
        row.update(aliases)
        row["_single_base"] = deepcopy(aliases)
    return row


def write_view(row: dict) -> dict:
    row = normalize(row)
    base = row.pop("_single_base", {})
    primary = row.pop("_primary_holder", 0)
    for key in HOLDER_FIELDS:
        if key in row:
            value = row.pop(key)
            if key not in base or base[key] != value:
                row["holders"][primary][key] = value
    row["updated_at"] = utc_now_iso()
    return row


def scoped(row: dict, holder: dict) -> dict:
    return {**{key: value for key, value in row.items() if key not in (*HOLDER_FIELDS, "holders", "_single_base", "_primary_holder")},
            **holder}


def wait_fields(state: dict, *, replay_acknowledged: bool) -> dict:
    pending = state.get("pending_wake") or {}
    if pending.get("acknowledged_at"):
        pending = {}
    if not pending and replay_acknowledged and state.get("status") == "awake":
        acknowledged = state.get("last_acknowledged_wake") or {}
        payload = acknowledged.get("payload") or {}
        if payload:
            wake_id = uuid.uuid4().hex
            pending = {**acknowledged, "wake_id": wake_id, "acknowledged_at": "",
                       "payload": {**payload, "supervision_wake_id": wake_id, "replayed_after_worker_loss": True},
                       "replay_reason": "acknowledged_in_dead_transcript"}
    return {"journal_cursor": int(state.get("journal_cursor") or 0),
            "interaction_acknowledged_ids": list(state.get("interaction_acknowledged_ids") or []),
            "pending_wake": deepcopy(pending), "checkpoint": deepcopy(state.get("checkpoint") or {})}


def checkpoint_view(state: dict, run_id: str) -> dict:
    return checkpoints.run_view(state, run_id) if run_id else checkpoints.task_facts(state)
