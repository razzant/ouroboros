"""Delegated invocation identity and the canonical first-request projection.

The custody facade re-exports these readers; the chain oracle stays independent
of the store reducer so imports and live publication can be compared directly.
"""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional

from ouroboros._usage_rows import REVIEW_ATTRIBUTION_KEYS

INVOCATIONS = "custody_invocations"
OPERATIONS = "custody_operation_invocations"
EVENTS = "custody_event_ordinals"
ORDINAL = "custody_append_ordinal"
COMPLETE = "custody_invocations_complete"
LEGACY_GAP = "legacy_gap"


def operation_key(operation_id: str, slot_id: str) -> str:
    return json.dumps([operation_id, slot_id], ensure_ascii=False)


def event_ordinal(txn, event: dict) -> int:
    """The event's durable place in the serialized custody commits, never its clock.

    ``custody_event_ordinals`` also deduplicates repeated imports and publications.
    """
    prior = txn.get_row(EVENTS, str(event.get("event_id") or "")) if event.get("event_id") else None
    return int(prior["ordinal"]) if prior else max(int(txn.meta(ORDINAL) or 0) + 1,
                                                 int(event.get("custody_ordinal") or 0))


def bound_runs(txn, invocation_id: str) -> dict:
    # Spell out the predicate of the partial index; equality alone cannot use it.
    return txn.custody_rows("invocation_id<>'' AND invocation_id=?", (invocation_id,))


def advance(record: dict, event: dict, ordinal: int) -> dict:
    """Invocation transition; the independent ``rows=`` oracle below defines its meaning."""
    from ouroboros.delegate_custody import START_REQUESTED, STARTED, START_FAILED

    record = dict(record or {"state": "pending", "run_id": ""})
    kind = event.get("type")
    if kind == START_REQUESTED and not record.get("request"):
        record.update(request=dict(event), request_ordinal=ordinal)
    elif kind == STARTED:
        record.update(state="started", run_id=str(event.get("run_id") or ""), started_ordinal=ordinal)
    elif kind == START_FAILED and event.get("definite") is True and record["state"] != "started":
        record["state"] = "failed_definite"
    return record


def legacy_record(txn, invocation_id: str, bound: Optional[dict] = None) -> Optional[dict]:
    """Named legacy path: only this invocation's stored facts, never the event chain.

    One STARTED and a retained original request suffice (including the F1 refusal).
    Multiple starts without ordinals, missing request facts or conflicting requests
    cannot prove the fold's answer and stay explicitly ``legacy_gap`` until repair.
    """
    requested = txn.get_row("custody_open", f"invocation:{invocation_id}")
    refused = txn.get_row("custody_refused", invocation_id)
    bound = bound_runs(txn, invocation_id) if bound is None else bound
    if not (requested or refused or bound):
        return None
    requests = [row["request"] for row in (requested, refused) if row and row.get("request")]
    requests.extend(facts["request_event"] for facts in bound.values() if facts.get("request_event"))
    gap = {"state": LEGACY_GAP, "run_id": "", "reason": "invocation_request_missing"}
    if not requests:
        return gap
    if any(request != requests[0] for request in requests[1:]):
        return {**gap, "reason": "original_request_unknown"}
    request = requests[0]
    started = {rid: facts for rid, facts in bound.items()
               if facts.get("started_ts") or facts.get("_invocation_started_ordinal")}
    if len(started) > 1 and not all(row.get("_invocation_started_ordinal") for row in started.values()):
        return {**gap, "request": request, "reason": "last_started_order_unknown"}
    if bound and not started:
        return {**gap, "request": request, "reason": "start_facts_missing"}
    rid = max(started, key=lambda key: started[key].get("_invocation_started_ordinal", 0)) if started else ""
    return {"request": request, "state": "started" if rid else "failed_definite" if refused else "pending",
            "run_id": rid}


def reduce_event(txn, event: dict, ordinal: int) -> Optional[dict]:
    from ouroboros.delegate_custody import START_REQUESTED, STARTED, START_FAILED

    iid, kind = str(event.get("invocation_id") or ""), event.get("type")
    if kind == START_REQUESTED:
        key = operation_key(str(event.get("operation_id") or ""), str(event.get("slot_id") or ""))
        if txn.get_row(OPERATIONS, key) is None:
            txn.put_row(OPERATIONS, key, {"invocation_id": iid, "request_ordinal": ordinal})
    if not iid or kind not in (START_REQUESTED, STARTED, START_FAILED):
        return None
    record = txn.get_row(INVOCATIONS, iid)
    if record is None:
        record = legacy_record(txn, iid)
        if record and record["state"] == LEGACY_GAP:
            record["legacy_gap"] = record["reason"]
    record = advance(record, event, ordinal)
    txn.put_row(INVOCATIONS, iid, record)
    return record


def _read(txn, invocation_id: str) -> Optional[dict]:
    record = txn.get_row(INVOCATIONS, invocation_id)
    bound = bound_runs(txn, invocation_id)
    # A rollback writer knows neither the invocation row nor the new run fact.
    # Its addressed rows remain usable only when the legacy path can prove the answer.
    if record is None or any(facts.get("started_ts") and not facts.get("_invocation_started_ordinal")
                             for facts in bound.values()):
        return legacy_record(txn, invocation_id, bound)
    if record.get("legacy_gap"):
        return {**record, "state": LEGACY_GAP, "reason": record["legacy_gap"], "run_id": ""}
    if record.get("state") == "pending" and txn.get_row("custody_open", f"invocation:{invocation_id}") is None:
        return legacy_record(txn, invocation_id, bound)
    return record


def _answer(root, record: Optional[dict], request_facts) -> Optional[dict]:
    if record is None:
        return None
    if record["state"] == LEGACY_GAP:
        return {"state": LEGACY_GAP, "run_id": "", "reason": record["reason"]}
    if not record.get("request"):
        return None  # A complete modern fold without REQUESTED has no invocation.
    return {**request_facts(root, record["request"]), "state": record["state"], "run_id": record["run_id"]}


def read(root, invocation_id: str, request_facts) -> Optional[dict]:
    from ouroboros import store

    with store.open_store(root, write=False) as txn:
        record = _read(txn, invocation_id)
    return _answer(root, record, request_facts)


def operation_record(root, operation_id: str, slot_id: str) -> Optional[dict]:
    """The first matching REQUESTED in chain order, addressed by (operation, slot)."""
    from ouroboros import store

    with store.open_store(root, write=False) as txn:
        if not txn.meta(COMPLETE):
            return {"state": LEGACY_GAP, "run_id": "", "reason": "request_order_not_backfilled"}
        first = txn.get_row(OPERATIONS, operation_key(str(operation_id), str(slot_id)))
        iid = str((first or {}).get("invocation_id") or "")
        record = _read(txn, iid) if iid else None
    answer = _answer(root, record, _request_facts)
    return {**answer, "invocation_id": iid} if answer is not None else None


def invocation_record(drive_root: Any, invocation_id: str, *,
                      rows: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """One invocation's durable fate: who requested it, the EXACT body it sent,
    the resources that attempt bound, and how it resolved.
    ``state`` is ``pending`` (requested, never bound, never definitely refused —
    the only state an explicit retry may replay), ``started`` (a run bound to it;
    carried with its ``run_id`` so the caller can wait instead of re-posting), or
    ``failed_definite`` (the daemon definitively refused; the id is dead — replaying
    it against a since-reconfigured route is how a key wedges into a permanent 409).
    ``request`` is the canonical POST body recorded before the wire attempt: a retry
    replays THESE bytes, never a re-derivation, because the engine's replay match
    digests the request and answers a same-key-different-digest POST with 409
    ``idempotency_conflict``.
    ``route``, ``project_id``, ``project_owned`` and ``idempotency_key`` are the
    attempt facts that never ride the wire, read from the FIRST request row — the
    minting — because the invocation stores its facts ONCE and a retry replays them.
    A retry that re-derived any of these from the current route/model/workspace
    context wrote a durable record contradicting the body it actually POSTed.
    First-request lineage, usage attribution (category/source/skill/wave/slot),
    and isolation facts are likewise replayed rather than re-derived.
    ``rows`` folds a caller's chain snapshot; otherwise an addressed store row
    retains the first request and last STARTED. Legacy rows use ``legacy_record``;
    insufficient stored evidence returns ``legacy_gap``, never an absent run.
    """
    from ouroboros.delegate_custody import START_REQUESTED, STARTED, START_FAILED, _unique_rows

    target = str(invocation_id or "").strip()
    if not target:
        return None
    if rows is None:
        from ouroboros.delegate_custody_current import invocation

        return invocation(drive_root, target, _request_facts)
    found: Optional[Dict[str, Any]] = None
    state, run_id = "pending", ""
    for row in _unique_rows(rows):
        if str(row.get("invocation_id") or "") != target:
            continue
        kind = str(row.get("type") or "")
        if kind == START_REQUESTED and found is None:
            found = _request_facts(drive_root, row)
        elif kind == STARTED:
            state, run_id = "started", str(row.get("run_id") or "")
        elif kind == START_FAILED and row.get("definite") is True and state != "started":
            state = "failed_definite"
    if found is None:
        return None
    return {**found, "state": state, "run_id": run_id}


def _request_facts(drive_root: Any, row: Dict[str, Any]) -> Dict[str, Any]:
    """The attempt facts of ONE START_REQUESTED row, the shape ``invocation_record`` returns."""
    from ouroboros.delegate_pending import request_body

    return {
        "task_id": str(row.get("task_id") or ""),
        "surface": str(row.get("surface") or ""),
        "slot_id": str(row.get("slot_id") or ""),
        "operation_id": str(row.get("operation_id") or ""),
        "request": request_body(drive_root, row),
        "route": str(row.get("route") or ""),
        "project_id": str(row.get("project_id") or ""),
        "project_owned": bool(row.get("project_owned")),
        # Absence is a fact: legacy rows fall back to the stored request.
        **({"project_persistent": bool(row["project_persistent"])}
           if "project_persistent" in row else {}),
        "idempotency_key": str(row.get("idempotency_key") or ""),
        "root_task_id": str(row.get("root_task_id") or ""),
        "parent_task_id": str(row.get("parent_task_id") or ""),
        "category": str(row.get("category") or "subagent"),
        "source": str(row.get("source") or "delegated_subagent"),
        **{key: str(row.get(key) or "") for key in REVIEW_ATTRIBUTION_KEYS},
        # The C1 isolation binding: a retry reproduces EXACTLY these — the
        # snapshot, the execution root, the baseline and the authority
        # target the original attempt bound — never a re-derivation.
        "snapshot_id": str(row.get("snapshot_id") or ""),
        "execution_root": str(row.get("execution_root") or ""),
        "execution_binding_fingerprint": str(row.get("execution_binding_fingerprint") or ""),
        "baseline_sha": str(row.get("baseline_sha") or ""),
        "target_root": str(row.get("target_root") or ""),
        "authority_source": str(row.get("authority_source") or ""),
        # Continuation lineage and the same-tree capture identity replay too.
        **{key: str(row.get(key) or "") for key in ("continuation_of", "capture_id", "snapshot_task_id")},
        **{key: copy.deepcopy(row[key]) for key in ("execution_mode", "effective_access", "thread_id", "authority_ceiling", "promotion") if key in row},
        # A copy: the source rows may be the shared, read-only custody memo.
        "resource_ref": copy.deepcopy(row.get("resource_ref")) if isinstance(row.get("resource_ref"), dict) else {},
        "selected_subagent_id": str(row.get("selected_subagent_id") or ""),
        "config_fingerprint": str(row.get("config_fingerprint") or ""),
        "work_order_fingerprint": str(row.get("work_order_fingerprint") or ""),
        "work_order_coverage": str(row.get("work_order_coverage") or ""),
        "authority_fingerprint": str(row.get("authority_fingerprint") or ""),
        "processing": copy.deepcopy(row.get("processing")) if isinstance(row.get("processing"), dict) else {},
        "prompt_cache_ttl": copy.deepcopy(row.get("prompt_cache_ttl")) if isinstance(row.get("prompt_cache_ttl"), dict) else {},
        # #1196: the cap and HOW it was decided replay with the body a
        # retry re-POSTs, so the replayed STARTED row keeps the same basis.
        "max_seconds": int(row.get("max_seconds") or 0) if str(row.get("max_seconds") or "").lstrip("-").isdigit() else 0,
        "requested_at": str(row.get("ts") or ""),  # preserve an unbound start's original window
        "max_seconds_basis": str(row.get("max_seconds_basis") or ""),
        "work_order_source_request": (
            copy.deepcopy(row.get("work_order_source_request"))
            if isinstance(row.get("work_order_source_request"), dict) else {}
        ),
    }
