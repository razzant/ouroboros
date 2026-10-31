"""Durable identity of the one automatic leaf-first bootstrap per task."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ouroboros import delegate_custody as custody
from ouroboros.utils import atomic_write_json


def _path(drive: Any, task_id: str) -> Path:
    return Path(drive) / "state" / "delegate_bootstrap" / f"{task_id}.json"


def read_identity(drive: Any, task_id: str) -> dict:
    try:
        row = json.loads(_path(drive, task_id).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(row, dict) or row.get("schema") != 1 or not row.get("invocation_id"):
        raise ValueError("initial invocation identity unreadable")
    return row


def claim_initial_invocation(drive: Any, task_id: str, request: dict) -> dict:
    """Called inside the actor decision lock, before START_REQUESTED and transport."""
    prior = read_identity(drive, task_id)
    if prior:
        return {"reason": "initial_start_already_claimed", "invocation_id": prior["invocation_id"],
                "detail": "The initial invocation already has custody; reconcile that identity."}
    row = {"schema": 1, "invocation_id": request["invocation_id"], **{key: request.get(key, "") for key in (
        "selected_subagent_id", "config_fingerprint", "work_order_fingerprint", "authority_fingerprint")}}
    path = _path(drive, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, row)
    return {}


def initial_receipt(ctx: Any, bootstrap: dict) -> dict:
    """A restart observes a claimed bootstrap; it never creates or replays a POST."""
    try:
        drive, task_id = custody.custody_root(ctx), str(ctx.task_id)
    except AttributeError:
        return {}  # a context without a data root has no durable custody to observe
    with custody.actor_decision_lock(drive, task_id):
        if custody.custody_log_unreadable(drive):
            raise ValueError("bootstrap custody unreadable")
        row = read_identity(drive, task_id)
        if not row:
            # Pre-v2 tasks have no marker: exact assignment custody proves that
            # automatic bootstrap already happened. No event history is rewritten.
            from ouroboros.delegate_custody_current import task_snapshot

            view = task_snapshot(drive, task_id)
            candidates = [vars(run) for run in view["runs"].values() if not run.review_owned]
            candidates += [item for item in view["pending"] if not custody.review_owned_source(item.get("source"))]
            row = next((item for item in candidates if item.get("invocation_id") and all(
                str(item.get(key) or "") == str(bootstrap.get(key) or "") for key in (
                    "selected_subagent_id", "config_fingerprint", "work_order_fingerprint"))), {})
        if not row:
            return {}
        identity = str(row["invocation_id"])
        fate = custody.invocation_record(drive, identity) or {}
        bootstrap["initial_invocation_id"] = identity
        bootstrap["exact_start_pending"] = False
        return {"status": "initial_invocation_recorded", "invocation_id": identity,
                "invocation_state": str(fate.get("state") or "unknown"),
                "run_id": str(fate.get("run_id") or ""), "physical_start": "not_repeated",
                **({"retry_of": identity} if fate.get("state") == "pending" else {})}
