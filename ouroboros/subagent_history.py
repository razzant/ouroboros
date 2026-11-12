"""Dated execution disclosure in the existing subagent receipt, never routing policy."""

from __future__ import annotations

import json
import logging
import pathlib
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from ouroboros.utils import utc_now_iso, write_text_atomic

log = logging.getLogger(__name__)
LAST_DELEGATION_FILENAME = "subagent_last_delegation.json"
FIRST_CALLS_KEY = "first_call_costs"  # beside the receipts in the projection: row id -> its first-call price samples


def _timestamp_key(value: Any) -> tuple[int, datetime]:
    """Order recorded instants, not their differing ISO spellings (Z/offset/fraction)."""
    text = str(value or "")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return (1, parsed.astimezone(timezone.utc))
    except (ValueError, OverflowError):
        # Unknown/legacy malformed dates cannot outrank a known instant.
        return (0, datetime.min.replace(tzinfo=timezone.utc))


def _last_delegation_path(drive_root=None):
    from ouroboros.config import DATA_DIR

    return pathlib.Path(drive_root or DATA_DIR) / "state" / LAST_DELEGATION_FILENAME


def _price_class(identity: Any) -> list[str]:
    """What a row's price per token depends on among its execution choices: the model and its processing tier."""
    identity = identity if isinstance(identity, Mapping) else {}
    return [str(identity.get("target_id") or ""), str(identity.get("processing_preference") or "")]


def _first_call_cost(drive_root, task_id: str, first: Mapping[str, Any], target: str) -> Optional[dict]:
    """What the task's first usable answer (``first``) cost, as the usage ledger settled it: the ledger
    attempt that carried it (``physical_attempt_id``). A record from before that id existed falls back to the
    task's first attempt on that answer's provider and model that came back with input counted; the ledger
    names a model as its route sends it, so the row's ``target`` and the answer's recorded name meet the
    ledger's spelling in ``normalize_model_identity``. ``usd`` is None when that attempt carries no price; an
    unreadable ledger (the display wait, never longer) or no such attempt is no sample."""
    from ouroboros import usage_store
    from ouroboros.provider_models import normalize_model_identity

    own = {normalize_model_identity(str(name)) for name in (target, first.get("resolved_model")) if name}
    try:
        with usage_store.read(drive_root, allow_stale=True) as txn:
            if first.get("physical_attempt_id"):
                row = txn.conn.execute("SELECT cost_usd FROM attempts WHERE attempt_id=? AND task_id=?",
                                       (str(first["physical_attempt_id"]), task_id)).fetchone()
                return None if row is None else {"usd": None if row["cost_usd"] is None else float(row["cost_usd"])}
            row = next((row for row in txn.conn.execute(
                "SELECT model, cost_usd FROM attempts INDEXED BY attempts_task WHERE task_id=? AND provider=? "
                "AND source='agent.task' AND COALESCE(prompt_tokens, 0) > 0 ORDER BY seq_first",
                (task_id, str(first.get("provider") or "")))
                if normalize_model_identity(str(row["model"] or "")) in own), None)
    except Exception:
        log.debug("first-call cost unavailable from the usage ledger", exc_info=True)
        return None
    return None if row is None else {"usd": None if row["cost_usd"] is None else float(row["cost_usd"])}


def _kept_first_calls(old: Any, rows: Mapping[str, Any], actor: str, sample: Optional[dict], of: list[str]) -> dict:
    """Every kept row's first-call samples, ``sample`` (of price class ``of``) offered to ``actor``'s. A row
    keeps the samples of its recorded price class, one per task (the first recorded), the newest
    ``RECENT_REVIEWS`` by the date of their execution, the bound a review seat's price is measured over. A late
    or repeated observation neither erases nor outranks newer measurements; a row that left the projection
    takes its samples."""
    from ouroboros.review_price import RECENT_REVIEWS

    old, kept = old if isinstance(old, dict) else {}, {}
    for key, row in rows.items():
        own = _price_class(row.get("identity") if isinstance(row, dict) else None)
        ring, first = old.get(key) if isinstance(old.get(key), dict) else {}, {}
        for item in [*(ring.get("calls") if ring.get("of") == own and isinstance(ring.get("calls"), list) else ()),
                     *([sample] if sample and key == actor and of == own else ())]:
            if isinstance(item, dict):
                first.setdefault(item.get("task_id"), item)
        if first:
            kept[key] = {"of": own, "calls": sorted(first.values(), key=lambda item: _timestamp_key(item.get("ts")))[-RECENT_REVIEWS:]}
    return kept


def first_call_facts(drive_root=None) -> dict[str, dict[str, str]]:
    """Each kept row's measured first-call price in the review seats' words (``review_price_text``): the
    median of its kept samples. No sample is no fact; samples without a price are "cost unknown"."""
    import statistics

    from ouroboros.review_price import ReviewPrice, review_price_text
    from ouroboros.subagent_metrics import number

    kept, facts = _projection(drive_root).get(FIRST_CALLS_KEY), {}
    for actor, ring in (kept.items() if isinstance(kept, dict) else ()):
        try:
            calls = [item for item in ring["calls"] if isinstance(item, dict)]
            usd = [float(item["usd"]) for item in calls if number(item.get("usd"))]
            price = ReviewPrice(usd=statistics.median(usd), reviews=len(usd), low=min(usd), high=max(usd)) if usd else None
            if calls:
                facts[actor] = {"first_call": review_price_text(price, "first call")}
        except Exception:  # a damaged ring is no price for its row; the receipts and the other rows still show
            log.debug("first-call samples unreadable for %s", actor, exc_info=True)
    return facts


def _projection(drive_root=None) -> dict[str, Any]:
    """The stored projection whole: the receipts and, beside them, each row's first-call price samples."""
    try:
        data = json.loads(_last_delegation_path(drive_root).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def subagent_last_delegation(drive_root=None) -> dict[str, Any]:
    """Read old single receipts and their additive per-actor history alike. The price samples are not a
    receipt: they stay in the file for its exact reader and never ride in the status snapshot."""
    return {key: value for key, value in _projection(drive_root).items() if key != FIRST_CALLS_KEY}


def execution_identity(snapshot: Mapping[str, Any]) -> dict[str, str]:
    """Only execution-affecting choices, not prose, order or list fingerprints."""
    route = snapshot.get("route") or {}
    return {"kind": str(route.get("kind") or ""),
            "target_id": str(route.get("target_id") or ""),
            "credential_profile_id": str(route.get("credential_profile_id") or ""),
            "effort": str(snapshot.get("effort") or ""),
            "processing_preference": str(snapshot.get("processing_preference") or ""),
            **({"access": str(snapshot.get("access", "workspace_write"))}
               if route.get("kind") == "agent_session" else {})}


def snapshot_handle(snapshot: Mapping[str, Any]) -> str:
    """The handle of the engine a frozen snapshot names, from its own facts."""
    from ouroboros.configured_subagents import engine_handle

    return engine_handle(execution_identity(snapshot))


def recorded_handle(row: Mapping[str, Any]) -> str:
    """Name the engine a history row ran, from the row's OWN recorded facts.

    A typed ``identity`` yields its handle; an older row without one yields its
    recorded route target. Never mapped through the live roster: the row an id
    points at today may be a different engine, and that would relabel the past.
    """
    from ouroboros.configured_subagents import engine_handle

    identity = row.get("identity")
    if isinstance(identity, Mapping) and identity.get("target_id"):
        return engine_handle(identity)
    route, model = str(row.get("route") or ""), str(row.get("requested_model") or "")
    return model if route == "api_model" else route + ("=" + model if route and model else "")


def record_last_delegation(*, route: str, requested_model: str, applied_model: str,
                           run_id: str, selected_subagent_id: str = "",
                           requested_profile: str = "", applied_profile: str = "",
                           drive_root=None, occurred_at: str = "", outcome: str = "unknown",
                           failure_code: str = "", reset_at: str = "",
                           identity: Mapping[str, Any] | None = None, task_id: str = "",
                           invocation_id: str = "", attempt_id: str = "", fallback=None,
                           observed_route: Mapping[str, Any] | None = None,
                           first_call: Mapping[str, Any] | None = None) -> None:
    """Keep the latest fact per actor plus the old top-level receipt interface.

    Occurrence and observation differ when recovery collects an old run. Replays
    neither refresh its date nor displace a newer fact. Task, invocation and
    attempt ids are kept as trace references when present, and ``identity``
    carries the originally requested effort. ``first_call`` is one measured sample of what the
    row's first call cost (``_first_call_cost``), kept beside the receipts (``_kept_first_calls``).
    The existing actor-count bound keeps this a compact projection; the event log retains full history.
    """
    from ouroboros.configured_subagents import MAX_CONFIGURED_SUBAGENTS
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    path = _last_delegation_path(drive_root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(".lock")
        lock = acquire_exclusive_file_lock(lock_path, timeout_sec=2.0, stale_sec=30.0)
        if lock is None:
            return
        try:
            stored = _projection(drive_root)
            old = {key: value for key, value in stored.items() if key != FIRST_CALLS_KEY}
            rows = dict(old.get("latest_by_subagent") or {})
            legacy_actor = str(old.get("selected_subagent_id") or "")
            if legacy_actor and legacy_actor not in rows:
                rows[legacy_actor] = {key: value for key, value in old.items()
                                      if key != "latest_by_subagent"}
            old_actor = rows.get(selected_subagent_id) or {}
            previous = old_actor if old_actor.get("run_id") == run_id else old if old.get("run_id") == run_id else {}
            if run_id and previous and (not occurred_at or (
                    previous.get("outcome") == outcome and previous.get("failure_code", "") == failure_code)):
                return
            observed = str(previous.get("observed_at") or utc_now_iso())
            row = {"ts": occurred_at or observed, "observed_at": observed, "occurred_at": occurred_at,
                   "route": str(route or ""), "requested_model": str(requested_model or ""),
                   "applied_model": str(applied_model or ""),
                   "requested_profile": str(requested_profile or ""),
                   "applied_profile": str(applied_profile or ""),
                   "selected_subagent_id": str(selected_subagent_id or ""),
                   "run_id": str(run_id or ""), "outcome": outcome}
            if identity:
                row["identity"] = dict(identity)
            if fallback:
                row["fallback"] = dict(fallback)
            if observed_route:
                row["observed_route"] = dict(observed_route)
            for key, value in (("failure_code", failure_code), ("reset_at", reset_at),
                               ("task_id", task_id), ("invocation_id", invocation_id), ("attempt_id", attempt_id)):
                if value:
                    row[key] = str(value)
            if selected_subagent_id and (not old_actor or (occurred_at and
                    _timestamp_key(row["ts"]) >= _timestamp_key(old_actor.get("ts")))):
                rows[selected_subagent_id] = row
            rows = dict(sorted(rows.items(), key=lambda item: _timestamp_key(item[1].get("ts")),
                               reverse=True)[:MAX_CONFIGURED_SUBAGENTS])
            latest = row if not old or (occurred_at and _timestamp_key(row["ts"]) >= _timestamp_key(old.get("ts"))) else old
            costs = _kept_first_calls(stored.get(FIRST_CALLS_KEY), rows, selected_subagent_id, first_call and {
                "task_id": task_id, "ts": row["ts"], **first_call}, _price_class(identity))
            write_text_atomic(path, json.dumps({**latest, "latest_by_subagent": rows, **({FIRST_CALLS_KEY: costs} if costs else {})},
                                               ensure_ascii=False, indent=1))
        finally:
            release_exclusive_file_lock(lock_path, lock)
    except Exception:
        log.debug("subagent history projection write failed", exc_info=True)


def record_task_execution(task: Mapping[str, Any], usage: Mapping[str, Any], *, drive_root) -> None:
    """Project configured attempt/start facts, never task correctness.

    A failed API call leaves ``applied_model`` empty; the model that served
    instead is recorded under ``fallback`` and never certifies the failed route.
    """
    from ouroboros.subagent_metric_events import task_execution
    task_execution(drive_root, task, usage)
    snapshot = task.get("configured_subagent") or {}
    identity = execution_identity(snapshot)
    availability = task.get("subagent_availability") or {}
    from ouroboros.owner_wait import governing_continuation

    if (identity["kind"] == "agent_session" and not governing_continuation(task)
            and availability.get("status") == "unavailable" and availability.get("reason")):
        route, _, model = identity["target_id"].partition("=")
        record_last_delegation(route=route, requested_model=model, applied_model="",
            run_id="task:" + str(task.get("id") or ""), task_id=str(task.get("id") or ""),
            selected_subagent_id=str(snapshot.get("selected_subagent_id") or ""),
            requested_profile=identity["credential_profile_id"], identity=identity, drive_root=drive_root,
            occurred_at=str(availability.get("observed_at") or ""), outcome="not_started",
            failure_code=str(availability["reason"]), reset_at=str(availability.get("reset_at") or ""))
        return
    if identity["kind"] != "api_model":
        return
    target = identity["target_id"]
    model = target[:-8].strip() if target.endswith(" (local)") else target
    calls = usage.get("llm_call_refs") or []

    def selected(row):
        return (row.get("model") == model and row.get("requested_profile")
                in (None, identity["credential_profile_id"]))

    call = next((row for row in reversed(calls) if isinstance(row, dict)
                 and selected(row)
                 and (row.get("failure_code") or row.get("usable_solve_response"))), {})
    if call:
        failure = str(call.get("failure_code") or "")
        outcome = "unknown" if failure == "provider_outcome_unknown" else "failed" if failure else "succeeded"
        when = str(call.get("ts") or "")
    elif availability.get("status") not in (None, "", "ready"):
        failure, outcome = str(availability.get("reason") or availability["status"]), "not_started"
        when = str(availability.get("observed_at") or "")
    else:
        return  # No observation is not success or failure.
    other = next((row for row in reversed(calls) if isinstance(row, dict)
                  and row.get("usable_solve_response") and not selected(row)), {}) if failure else {}
    # The row's first-call price is measured when the task's first usable answer was its own model's (never a
    # fallback's) on a route that bills money: the subscription engine and the local lane state no per-call cash.
    from ouroboros.usage_accounting import paid_admission
    first = next((row for row in calls if isinstance(row, dict) and row.get("usable_solve_response")), None)
    first_call = (_first_call_cost(drive_root, str(task.get("id") or ""), first, model)
                  if first and selected(first) and paid_admission(first.get("provider"), None) else None)
    fallback = {key: other[key] for key in ("model", "llm_call_id", "ts") if key in other}
    if other:
        fallback.update(_api_observed_facts(other))
    record_last_delegation(
        route="api_model", requested_model=target,
        run_id=str(call.get("llm_call_id") or task.get("id") or ""),
        requested_profile=identity["credential_profile_id"],
        **_api_observed_facts(call, failed=bool(failure)),
        selected_subagent_id=str(snapshot.get("selected_subagent_id") or ""),
        drive_root=drive_root, occurred_at=when, outcome=outcome,
        failure_code=failure, reset_at=str(call.get("reset_at") or availability.get("reset_at") or ""),
        identity=identity, task_id=str(task.get("id") or ""),
        attempt_id=str(call.get("llm_call_id") or ""), fallback=fallback, first_call=first_call)


def _api_observed_facts(call: Mapping[str, Any], *, failed: bool = False) -> dict:
    route = call.get("observed_route") or {}
    # Provider usage.resolved_model is a host target, not a served-model report.
    return {"applied_model": str(route.get("model") or "") if not failed else "",
            "applied_profile": str(route.get("credentialProfileId") or ""),
            "observed_route": dict(route)}


def session_request_facts(request: Mapping[str, Any], *, selected_subagent_id: str,
                          task_id: str, route: str, processing: Mapping[str, Any],
                          row_effort: Optional[str] = None) -> dict:
    """Compact original intent for existing custody rows, without copying the work order.
    ``row_effort`` is the configured row's pin ('' = Auto): a start failure's receipt
    identity, like a settled run's, never the level the leaf was started at."""
    return {"selected_subagent_id": selected_subagent_id, "task_id": task_id, "route": route,
            "model": str(request.get("model") or ""), "profile_id": str(request.get("credentialProfileId") or ""),
            "access": str(request.get("access") or ""),
            **({"effort": request["effort"]} if isinstance(request.get("effort"), str) else {}),
            **({"row_effort": row_effort} if isinstance(row_effort, str) else {}),
            **({"processing_preference": processing["requested"]}
               if isinstance(processing.get("requested"), str) else {})}


def record_session_execution(drive_root, custody, detail: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    """Common foreground/recovery settlement projection; reviewers keep their own history."""
    if custody.review_owned:
        return
    from ouroboros.delegate_custody import summary_of
    summary = summary_of(detail)
    failure = summary.get("failure") if isinstance(summary.get("failure"), dict) else {}
    # Identity is the CONFIGURED row (its pin, '' for Auto), never the level the leaf ran at —
    # that is the effort fact; a row without the pin keeps the leaf level as it always did.
    effort = custody.row_effort if custody.row_effort is not None else custody.effort
    identity = {"kind": "agent_session",
                "target_id": custody.route_id + ("=" + custody.model if custody.model else ""),
                "credential_profile_id": custody.profile_id,
                "access": custody.access,
                **({"effort": effort} if effort is not None else {}),
                **({"processing_preference": custody.processing_preference}
                   if custody.processing_preference is not None else {})}
    from ouroboros.subagent_metric_events import session_execution
    session_execution(drive_root, custody, summary, identity, observed)
    from ouroboros.swarm_receipt import refresh_for_task
    refresh_for_task(drive_root, custody.task_id)
    record_last_delegation(
        route=custody.route_id, requested_model=custody.model,
        applied_model=str(observed.get("model") or ""), run_id=custody.run_id,
        selected_subagent_id=custody.selected_subagent_id,
        requested_profile=custody.profile_id, applied_profile=str(observed.get("profile_id") or ""),
        drive_root=drive_root, occurred_at=str(summary.get("finishedAt") or ""),
        outcome=str(summary.get("state") or "unknown"),
        failure_code=str(failure.get("code") or ""), reset_at=str(failure.get("resetsAt") or ""), identity=identity,
        task_id=custody.task_id, invocation_id=custody.invocation_id,
        attempt_id=str(observed.get("attempt_id") or ""))


def record_session_start_failure(drive_root, event: Mapping[str, Any]) -> None:
    actor = str(event.get("selected_subagent_id") or "")
    if not actor:
        return
    route, model = str(event.get("route") or ""), str(event.get("model") or "")
    pin = str(event.get("profile_id") or "")
    record_last_delegation(
        route=route, requested_model=model, applied_model="", run_id=str(event.get("invocation_id") or ""),
        selected_subagent_id=actor, requested_profile=pin, drive_root=drive_root,
        task_id=str(event.get("task_id") or ""), invocation_id=str(event.get("invocation_id") or ""),
        occurred_at=str(event.get("ts") or ""), outcome="not_started" if event.get("definite") else "unknown",
        failure_code=str(event.get("reason") or ""), identity={
            "kind": "agent_session", "target_id": route + ("=" + model if model else ""),
            "access": str(event.get("access") or ""), "credential_profile_id": pin,
            **({"effort": event["row_effort"]} if isinstance(event.get("row_effort"), str)
               else {"effort": event["effort"]} if "effort" in event else {}),
            **({"processing_preference": event["processing_preference"]} if "processing_preference" in event else {})})
