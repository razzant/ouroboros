"""Translate producer receipts into measurements without interpreting task correctness."""
from __future__ import annotations
import pathlib

from ouroboros.subagent_metrics import publish


def _call_evidence(root, call_id, *, refs=None):
    """Addressed ledger stamps, never prices, inferred execution time or a log scan."""
    from ouroboros import store, usage_store
    from ouroboros.subagent_metrics import CALLS
    from ouroboros.swarm_receipt import instant

    try:
        if refs is None:
            with store.open_store(root, write=False) as txn:
                refs = txn.get_row(CALLS, call_id) or {}
        ids = refs.get("ledger_attempt_ids") or []
        if not ids:
            return {}
        with usage_store.read(root, allow_stale=True) as txn:
            rows = [txn.conn.execute("SELECT state,ts_reserved,ts_dispatched,ts_final,late_receipt FROM attempts WHERE attempt_id=?",
                                     (attempt,)).fetchone() for attempt in ids]
        if any(row is None for row in rows):
            return {}
        spans, queues = [], []
        for row in rows:
            reserve, start, end = (instant(row[key]) for key in ("ts_reserved", "ts_dispatched", "ts_final"))
            if row["state"] == "settled" and not row["late_receipt"] and start is not None and end is not None and end >= start:
                spans.append(end - start)
            if reserve is not None and start is not None and start >= reserve:
                queues.append(start - reserve)
        facts = {"dispatched": True if any(row["ts_dispatched"] for row in rows) else
                False if all(row["state"] == "released" for row in rows) else None,
                "duration_sec": sum(spans) if len(spans) == len(rows) else None,
                "duration_basis": "sum_of_physical_dispatch_to_settlement_seconds",
                "queue": {"basis": "reservation_to_dispatch", "samples": len(queues),
                          "sum_sec": sum(queues), "attempts": len(rows)}}
        if facts["dispatched"] is None:
            facts.pop("dispatched")  # Missing ledger clocks cannot erase a directly received response.
        return facts
    except Exception:
        return {}


def task_execution(root, task, usage):
    from ouroboros.subagent_history import execution_identity

    snapshot = task.get("configured_subagent") or {}
    identity = execution_identity(snapshot)
    if identity["kind"] != "api_model":
        return
    model = identity["target_id"].removesuffix(" (local)").strip()
    for call in usage.get("llm_call_refs") or ():
        if (not isinstance(call, dict) or call.get("model") != model
                or call.get("requested_profile") not in (None, identity["credential_profile_id"])):
            continue
        call_id = call.get("llm_call_id")
        if not call_id:
            continue
        failure = call.get("failure_code")
        received = call.get("model_output_received")
        if not isinstance(received, bool):
            received = True if call.get("response_ref") else None
        usable = call.get("usable_solve_response") is True
        terminal = usable or bool(failure and failure != "provider_outcome_unknown")
        publish(root, {"row_id": snapshot.get("selected_subagent_id"), "row_identity": identity,
            "purpose": "execution", "producer_id": "call:" + call_id, "task_id": str(task.get("id") or task.get("task_id") or ""),
            "ts": call.get("ts"), "dispatched": True if received or call.get("dispatched") is True else None,
            "terminal": terminal, "answer_received": received, "usable_solve_response": usable, "usable_verdict": None,
            "interrupted": failure == "interrupted", "duration_sec": call.get("duration_sec"),
            "queue": call.get("queue"), "quota": {"failure_code": failure, "reset_at": call.get("reset_at")}
                if failure else None, "nanny_billing": task.get("supervisor_route"),
            **_call_evidence(root, call_id),
            "source_refs": [call.get("response_ref") or {"task_id": task.get("id"), "call_id": call_id}],
            "actor": {"id": "call:" + call_id, "task_id": task.get("id"), "kind": "api_call",
                      "observed_route": dict(call.get("observed_route") or {}), "requested_profile": call.get("requested_profile"),
                      "start": call.get("started_at"), "end": call.get("ts") if terminal else None, "terminal": terminal}})


def session_execution(root, custody, summary, identity, observed):
    import hashlib
    from ouroboros.task_results import load_task_result

    try:
        owner = load_task_result(root, custody.task_id, strict=True) or {}
    except Exception:
        owner = {}
    state = summary.get("state")
    terminal = state in {"succeeded", "failed", "cancelled", "interrupted"}
    # A successful empty session is not evidence that it returned an answer.
    received = summary.get("answer_received")
    if received is None and isinstance(observed.get("answer_received"), bool):
        received = observed["answer_received"]
    publish(root, {"row_id": custody.selected_subagent_id,
        "row_identity": {"effort": "", "processing_preference": "", **identity},
        "purpose": "plan" if custody.access == "plan" or custody.mode == "plan" else "execution", "producer_id": "run:" + custody.run_id,
        "task_id": custody.task_id, "ts": summary.get("finishedAt") or custody.started_ts,
        "dispatched": True, "terminal": terminal, "answer_received": received, "usable_verdict": None,
        "interrupted": state in {"cancelled", "interrupted"}, "duration_sec": summary.get("durationSec"),
        "duration_basis": "engine_reported_seconds",
        "queue": summary.get("queue"), "quota": summary.get("failure"),
        "nanny_billing": owner.get("supervisor_route"),
        "source_refs": [{"task_id": custody.task_id, "run_id": custody.run_id, "invocation_id": custody.invocation_id}],
        "actor": {"id": "run:" + custody.run_id, "task_id": custody.task_id, "kind": "delegated_run",
                  "start": summary.get("startedAt") or custody.started_ts, "end": summary.get("finishedAt"),
                  "usage_ref": {"ledger_attempt_ids": ["session-" + hashlib.sha256(custody.run_id.encode()).hexdigest()[:24]]},
                  "terminal": terminal, "account": dict(observed)}})


def review_execution(request, slot, actor, ctx, *, fallback_root=None):
    from ouroboros.review_ledger import ledger_root

    outcome, usage = actor.outcome or {}, actor.usage or {}
    if not outcome.get("outcome_id") or not outcome.get("operation_id"):
        return
    try:
        root = ledger_root(ctx)
    except AttributeError:
        if fallback_root is None:
            raise
        root = pathlib.Path(fallback_root)  # the coordinator's canonical custody root
    session = str(getattr(slot.route, "value", slot.route)) == "agent_session"
    identity = {"kind": "agent_session" if session else "api_model",
                "target_id": slot.session_target if session else slot.model,
                "credential_profile_id": slot.session_profile or "", "effort": slot.effort or "",
                "processing_preference": slot.processing_preference or "",
                **({"access": slot.session_access or "workspace_write"} if session else {})}
    received = outcome.get("transport") == "success" or bool(actor.raw_text)
    pending, unsent = outcome.get("status") == "pending", outcome.get("status") == "not_dispatched"
    publish(root, {"row_id": slot.subagent_id or slot.slot_id, "row_identity": identity,
        "purpose": "plan" if request.surface == "plan_review" else "review",
        "producer_id": "review:" + outcome["outcome_id"], "task_id": request.task_id,
        "ts": usage.get("finished_at") or usage.get("ts"), "terminal": not pending, "pending": pending,
        "dispatched": False if unsent else True if received or outcome.get("usage_ref", {}).get("ledger_attempt_ids")
            or usage.get("delegated_run_id") else None,
        "answer_received": received if not pending else None,
        "usable_verdict": outcome.get("parse") == "valid" and outcome.get("status") == "responded",
        "interrupted": outcome.get("status") == "interrupted",
        "duration_sec": actor.duration_sec if actor.duration_sec > 0 else None,
        "duration_basis": "review_producer_seconds",
        "queue": usage.get("queue"), "quota": {"failure_code": actor.failure_code, "reset_at": actor.reset_at}
            if actor.failure_code else None, "source_refs": [
                {"task_id": request.task_id, "call_id": actor.operation_id + "_prompt"} if pending else outcome.get("source_ref"),
                outcome.get("usage_ref")],
        "actor": {"id": "review:" + outcome["outcome_id"], "task_id": request.task_id, "kind": "review",
                  "start": actor.awaiting_since or usage.get("started_at"), "end": usage.get("finished_at"),
                  "terminal": not pending, "usage_ref": outcome.get("usage_ref")}})
