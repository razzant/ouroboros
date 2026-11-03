"""Current indexed sources of the tree receipt; no forensic history walk."""
from __future__ import annotations

import json

from ouroboros import store
from ouroboros.swarm_receipt import instant
from ouroboros.subagent_metrics import set_name


def tree_inputs(root, root_id, *, task=None):
    from ouroboros.terminal_time import terminal_time_fact
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES

    actors, waits, measurements, gaps = {}, [], [], []
    with store.open_store(root, write=False) as txn:
        ids = [row[0] for row in txn.conn.execute(
            "SELECT task_id FROM tasks WHERE root_task_id=? OR task_id=? ORDER BY task_id", (root_id, root_id))]
        rows = [txn.load(task_id) for task_id in ids]
        owner = next((row for row in rows if row["task_id"] == root_id), {})
        for row in rows:
            tid = row["task_id"]
            actors["task:" + tid] = {"id": "task:" + tid, "task_id": tid,
                "kind": "supervisor" if tid == root_id or row.get("supervisor_route") else "task",
                "start": row.get("started_at"), "end": terminal_time_fact(row)["occurred_at"],
                "terminal": row.get("status") in _TRULY_TERMINAL_STATUSES,
                "parent_task_id": row.get("parent_task_id"), "basis": "host_task_lifetime",
                "supervisor_route": row.get("supervisor_route")}
            measurements.extend(txn.set_rows(set_name("swarm_actors", tid)).values())
            waits.extend(txn.set_rows(set_name("swarm_waits", tid)).values())
            for op_id, op in (row.get("review_operations") or {}).items():
                if op.get("state") in {"preparing", "retained", "dispatched"}:
                    actors["review_operation:" + op_id] = {"id": "review_operation:" + op_id,
                        "task_id": tid, "kind": "review_custody", "terminal": False, "start": None, "end": None}
            for run_id, raw in txn.custody_raw_rows("task_id=?", (tid,)).items():
                run = json.loads(raw)
                if run.get("review_wave_id"):
                    continue  # The canonical review producer owns this actor below.
                actors["run:" + run_id] = {"id": "run:" + run_id, "task_id": tid, "kind": "delegated_run",
                    "start": run.get("started_ts"), "end": None, "terminal": bool(run.get("terminal_state")),
                    "collected": run.get("settled") is True, "basis": "engine_run",
                    "account": {"requested_profile": run.get("profile_id"), "observed_profile": run.get("credential_profile_id")}}
        # Read unresolved requests in the same snapshot as assigned runs: STARTED
        # removes the invocation from custody_open when it binds the run.
        for key, facts in txn.set_rows("custody_open").items():
            request = facts.get("request") or {}
            if not key.startswith("invocation:") or not (
                    request.get("task_id") in ids or request.get("root_task_id") == root_id):
                continue
            invocation_id = request["invocation_id"]
            actors["invocation:" + invocation_id] = {"id": "invocation:" + invocation_id,
                "invocation_id": invocation_id, "task_id": request.get("task_id"),
                "kind": "delegated_start", "state": "pending", "terminal": False,
                "start": None, "end": None, "accounted_upper_bound_usd": None, "basis": "retained_start_request"}
        cursor = txn.meta("subagent_metrics_cursor") or {}
    # Receipt refresh runs on ordinary producer threads. It uses retained facts;
    # only an explicit rollup read catches up legacy history.
    from ouroboros.swarm_rollup import rollup_from_document

    prior = owner.get("swarm_fanout_rollup")
    known_children = prior.get("child_ids") if isinstance(prior, dict) else None
    expected = {child for child in known_children if isinstance(child, str)} if isinstance(known_children, list) else set()
    try:
        facts = rollup_from_document(owner)
        expected = set(facts.get("child_ids") or ())
    except (OSError, ValueError, TypeError, store.StoreError):
        gaps.append("fanout_history_unavailable")
    if expected - set(ids):
        gaps.append("missing_child_results")
    if not owner:
        gaps.append("root_result_absent")
    for event in measurements:
        actor = event.get("actor") or {}
        if actor and actor.get("kind") != "api_call":
            actors[actor["id"]] = {**actors.get(actor["id"], {}), **actor}
    usage, usage_coverage = _usage(root, root_id, measurements)
    edges = _wait_edges(actors, waits)
    # An unreadable or torn instrumented history cannot certify that every actor is known.
    stream = (cursor.get("coverage") or {}).get("stream")
    if stream in {"unavailable", "incomplete"}:
        gaps.append("measurement_stream_" + stream)
    if cursor.get("missing_producers"):
        gaps.append("missing_measurement_publications")
    return {"actors": list(actors.values()), "edges": edges, "waits": waits, "usage": usage,
        "coverage": {"tree": "complete" if not gaps else "unknown", "gaps": gaps,
                     "usage": usage_coverage, "causal_edges": "unknown",
                     "history_before_instrumentation": "unknown"},
        "mind_comparison": owner.get("swarm_comparison", (task or {}).get("swarm_comparison"))}


def _usage(root, root_id, measurements):
    from ouroboros import usage_store
    from ouroboros.runtime_limits import USAGE_DISPLAY_LOCK_TIMEOUT_SEC

    bindings, accounts = {}, {}
    for event in measurements:
        actor = event.get("actor") or {}
        for attempt in (actor.get("usage_ref") or {}).get("ledger_attempt_ids") or ():
            bindings[attempt] = "task:" + str(event.get("task_id")) if actor.get("kind") == "api_call" else actor["id"]
            if attempt == (actor.get("usage_ref") or {}).get("observed_attempt_id"):
                accounts[attempt] = {key: actor[key] for key in ("observed_route", "requested_profile") if key in actor}
    try:
        # A receipt consumes existing money facts. Creating an empty ledger can
        # append an import event to previously absent fan-out history.
        with usage_store.hold(root, write=False, migrate=False, timeout_sec=USAGE_DISPLAY_LOCK_TIMEOUT_SEC) as txn:
            captured = txn.attempt_rows("root_task_id=? OR task_id=?", (root_id, root_id))
        rows = usage_store.decode_attempts(captured)
        return [{**row, **accounts.get(row["attempt_id"], {}),
                 **({"actor_id": bindings[row["attempt_id"]]} if row["attempt_id"] in bindings else {})}
                for row in rows], "unknown" if usage_store.integrity_degraded(root) else "complete"
    except Exception:
        return [], "unknown"


def _wait_edges(actors, waits):
    """Split the waiting supervisor at actual pauses; ancestry never becomes an edge."""
    edges = []
    for wait in sorted(waits, key=lambda row: instant(row.get("started_at")) or 0):
        tid = wait.get("task_id")
        base_id = "task:" + str(tid)
        parent = actors.get(base_id)
        start, wake = instant(wait.get("started_at")), instant(wait.get("woke_at"))
        if not parent or start is None or wake is None or wake < start:
            continue
        parent_start, parent_end = instant(parent.get("start")), instant(parent.get("end"))
        if parent_start is None or parent_end is None or not parent_start <= start <= wake <= parent_end:
            continue
        before, after = base_id + ":before:" + wait["wait_id"], base_id + ":after:" + wait["wait_id"]
        for edge in edges:
            if edge["to"] == base_id:
                edge["to"] = before
        actors[before] = {**parent, "id": before, "end": start}
        actors[base_id] = {**parent, "start": wake}  # the next wait splits this remaining segment
        edges.append({"from": before, "to": base_id, "kind": "observed_resume", "observed": True,
                      "wait_id": wait["wait_id"], "resume_segment": after})
        source = wait.get("waking_actor_id") or wait.get("woke_by", "")
        if source.startswith("task:"):
            source = ":".join(source.split(":")[:2])
        if source in actors and _selected(wait.get("selection") or {}, source):
            edges.append({"from": source, "to": base_id, "kind": "selected_terminal_wake", "observed": True,
                          "wait_id": wait["wait_id"]})
    return edges


def _selected(selection, source):
    kind, _, identity = source.partition(":")
    if kind in {"task", "run"}:
        return identity in selection.get(kind + "s", ())
    if kind == "review":
        return any(identity == f"{pin.get('operation_id')}:{pin.get('slot_id')}" for pin in selection.get("reviewers", ()))
    return False
