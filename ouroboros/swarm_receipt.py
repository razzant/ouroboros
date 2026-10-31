"""Revisable tree-end measurements. An observation is never a team-size decision."""
from __future__ import annotations

import copy
import hashlib
import logging
import pathlib
from datetime import datetime
from decimal import Decimal

from ouroboros.subagent_metrics import number

log = logging.getLogger(__name__)


def instant(value):
    if number(value):
        return float(value)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.timestamp() if parsed.tzinfo is not None else None
    except (ValueError, OverflowError, OSError):
        return None


def _interval(actor):
    start, end = instant(actor.get("start")), instant(actor.get("end"))
    return (start, end) if start is not None and end is not None and end >= start else None


def _overlap(intervals):
    points = {}
    for start, end in intervals:
        points[start] = points.get(start, 0) + 1
        points[end] = points.get(end, 0) - 1
    active = peak = 0
    union = overlap = actor_seconds = 0.0
    prior = None
    for when, delta in sorted(points.items()):
        span = when - prior if prior is not None else 0
        union += span if active else 0
        overlap += span if active > 1 else 0
        actor_seconds += span * active
        active += delta
        peak = max(peak, active)
        prior = when
    return {"observed_actor_seconds": actor_seconds, "observed_union_sec": union,
            "overlap_sec": overlap, "peak_observed_actors": peak,
            "mean_observed_parallelism": actor_seconds / union if union else None,
            "basis": "observed_intervals_not_productivity"}


def _causal_path(actors, edges, complete):
    intervals = {actor["id"]: _interval(actor) for actor in actors}
    lengths = {key: end - start for key, value in intervals.items() if value for start, end in [value]}
    paths = {key: [key] for key in lengths}
    valid, gaps = [], []
    for edge in edges:
        left, right = edge.get("from"), edge.get("to")
        a, b = intervals.get(left), intervals.get(right)
        if (not edge.get("observed") or not a or not b or left == right
                or a[1] > b[0] or (a[0] == a[1] == b[0] == b[1])):
            gaps.append(edge)
        else:
            valid.append(edge)
    incoming = {key: 0 for key in lengths}
    for edge in valid:
        incoming[edge["to"]] += 1
    ready = [key for key in lengths if not incoming[key]]
    roots, visited = len(ready), 0
    while ready:
        source = ready.pop()
        visited += 1
        for edge in valid:
            if edge["from"] != source:
                continue
            actor_id = edge["to"]
            candidate = lengths[source] + intervals[actor_id][1] - intervals[actor_id][0]
            if candidate > lengths[actor_id]:
                lengths[actor_id], paths[actor_id] = candidate, paths[source] + [actor_id]
            incoming[actor_id] -= 1
            if not incoming[actor_id]:
                ready.append(actor_id)
    best = max(lengths, key=lengths.get) if lengths else None
    known = bool(complete and not gaps and visited == len(actors) and roots == 1)
    return {"seconds": lengths[best] if known and best else None,
            "observed_seconds": lengths.get(best), "actor_ids": paths.get(best, []),
            "coverage": "complete" if known else "missing_intervals_or_edges", "rejected_edges": gaps}


def fold_tree(root_task_id, actors, *, edges=(), waits=(), usage=(), coverage=None, mind_comparison=None):
    """Fold immutable actor/attempt identities; ancestry alone supplies no causal edge."""
    actors = list({row["id"]: copy.deepcopy(row) for row in actors}.values())
    intervals = [value for row in actors if (value := _interval(row)) is not None]
    coverage = dict(coverage or {})
    closed = bool(actors and all(row.get("terminal") is True and row.get("collected") is not False for row in actors)
                  and coverage.get("tree") == "complete")
    attempts = {str(row["attempt_id"]): row for row in usage if row.get("attempt_id")}
    supervisors = {root_task_id} | {row.get("task_id") for row in actors if row.get("supervisor_route")}
    costs, pressures = [], []
    known_total = Decimal(0)
    for attempt_id, row in attempts.items():
        cost = row.get("cost_usd")
        known = number(cost) and row.get("cost_final") is True
        known_total += Decimal(str(cost)) if known else 0
        costs.append({"attempt_id": attempt_id, "task_id": row.get("task_id"),
            "actor_id": row.get("actor_id") or ("review:" + str(row["review_slot_id"]) if row.get("review_slot_id")
                                                else "task:" + str(row.get("task_id") or "")),
            "kind": row.get("kind"), "provider": row.get("provider"), "model": row.get("model"),
            "accounted_upper_bound_usd": cost if number(cost) else None, "cost_final": known,
            "account": {key: copy.deepcopy(row[key]) for key in
                        ("credential_profile_id", "requested_profile", "observed_route", "subscription_route") if key in row},
            "account_coverage": "observed" if row.get("credential_profile_id") or row.get("observed_route") else "unknown"})
        if (row.get("task_id") in supervisors and row.get("physical_context")
                and not row.get("review_slot_id") and row.get("kind") != "subscription_session"):
            pressures.append({"attempt_id": attempt_id, "task_id": row.get("task_id"),
                              "prompt_tokens": row.get("prompt_tokens"), "physical_context": copy.deepcopy(row["physical_context"])})
    unknown_costs = sum(not row["cost_final"] for row in costs)
    return {"schema_version": 1, "root_task_id": root_task_id, "state": "closed" if closed else "provisional",
        "coverage": {**coverage, "intervals_observed": len(intervals), "actors_observed": len(actors)},
        "actor_intervals": actors, "selected_waits": list(waits), "causal_edges": list(edges),
        "parallelism": _overlap(intervals),
        "longest_causal_path": _causal_path(actors, edges, coverage.get("causal_edges") == "complete"),
        "cost": {"known_final_usd": float(known_total), "unknown_attempts": unknown_costs,
                 "total_usd": float(known_total) if costs and not unknown_costs and coverage.get("usage") == "complete" else None,
                 "attempts": costs}, "supervisor_context_pressure": pressures,
        "mind_comparison": {"source": "task.swarm_comparison", "author": "mind", "value": mind_comparison}
            if mind_comparison is not None else {"state": "not_recorded", "author": "mind"}}


def publish(root, root_task_id, *, task=None):
    """Serialize revisions after producers release their source transactions."""
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
    from ouroboros.swarm_receipt_sources import tree_inputs
    from ouroboros.task_store import patch_document
    from ouroboros.utils import utc_now_iso

    root = pathlib.Path(root)
    lock_path = root / "state" / ("swarm_receipt." + hashlib.sha256(root_task_id.encode()).hexdigest() + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    outcome = {}
    lock = acquire_exclusive_file_lock(lock_path, timeout_sec=float("inf"), stale_sec=30.0,
                                       owner_aware_stale=True, outcome=outcome)
    if lock is None:
        log.warning("swarm receipt lock unavailable for %s: %s", root_task_id, outcome)
        return None
    try:
        inputs = tree_inputs(root, root_task_id, task=task)
        receipt = fold_tree(root_task_id, **inputs)

        def update(current):
            if not current:
                return None
            prior = current.get("swarm_receipt") or {}
            stable = {key: value for key, value in prior.items() if key not in {"revision", "published_at"}}
            if stable == receipt:
                return None
            return {**current, "swarm_receipt": {**receipt, "revision": prior.get("revision", 0) + 1,
                                                "published_at": utc_now_iso()}}
        row = patch_document(root, root_task_id, update)
        return row.get("swarm_receipt")
    except Exception:
        log.warning("swarm receipt publication unavailable", exc_info=True)
        return None
    finally:
        release_exclusive_file_lock(lock_path, lock)


def refresh_for_task(root, task):
    """Late publications revise an existing receipt; ordinary rounds do no tree scan."""
    from ouroboros.task_results import validate_task_id
    from ouroboros.task_store import receipt_anchor

    if not root or not task:
        return
    if not isinstance(task, dict):
        try:
            validate_task_id(str(task))
        except ValueError:
            return  # System accounting actors have no task-result receipt.
    try:
        if isinstance(task, dict):
            task_id = str(task.get("task_id") or task.get("id") or "")
            root_id = str(task.get("root_task_id") or (task.get("metadata") or {}).get("root_task_id") or task_id)
            anchor = (root_id, bool(task.get("swarm_receipt"))) if task_id == root_id else receipt_anchor(root, root_id)
        else:
            anchor = receipt_anchor(root, str(task))  # two typed fields, never a parsed task result
        if anchor and anchor[1]:
            publish(root, anchor[0])
    except Exception as exc:
        # This hook also runs during provider exception handling; its chain may
        # contain credentials, so retain only the local failure type.
        log.warning("swarm receipt revision unavailable (%s)", type(exc).__name__)


def usage_changed(root, task_ids):
    for task_id in task_ids:
        refresh_for_task(root, task_id)


def retain_comparison(ctx):
    """Retain the mind's selected statement only after completion gates accept it."""
    from ouroboros.task_store import patch_document

    statement = (getattr(ctx, "_completion_selected", None) or {}).get("swarm_comparison")
    root = getattr(ctx, "budget_drive_root", None) or getattr(ctx, "drive_root", None)
    task_id = str(getattr(ctx, "task_id", "") or "")
    if not root or not task_id or not isinstance(statement, str) or not statement.strip():
        return
    try:
        row = patch_document(root, task_id, lambda current:
            {**current, "swarm_comparison": statement} if current else None)
        refresh_for_task(root, row)
    except Exception:
        log.warning("swarm comparison retention unavailable", exc_info=True)
