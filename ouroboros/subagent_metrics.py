"""Measured configured-row facts. Publication advances history; inspection only reads.

Producer rows and compact totals share the current store transaction with the
archive cursor. Re-publication replaces one producer, including a late answer.
The window covers instrumented observations, never the unobserved installation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import pathlib

from ouroboros import store
from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)
EVENT = "subagent_measurement"
CURSOR = "subagent_metrics_cursor"
PRODUCERS = "subagent_metric_producers"
CALLS = "subagent_metric_call_sources"
COUNTS = ("dispatched", "terminal", "answer_received", "usable_verdict", "no_answer", "pending", "interrupted")


def identity_key(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def set_name(kind, identity):
    """Exportable on every filesystem; public row/task ids remain inside the facts."""
    return kind + "-" + identity_key(str(identity))


def _group(event):
    return set_name("subagent_metrics", event["row_id"]), identity_key(event["row_identity"]) + ":" + event["purpose"]


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _counts(event):
    dispatched, terminal = event.get("dispatched") is True, event.get("terminal") is True
    interrupted = event.get("interrupted") is True
    return {"dispatched": int(dispatched), "terminal": int(terminal),
            "answer_received": int(event.get("answer_received") is True),
            "usable_verdict": int(event.get("usable_verdict") is True),
            "interrupted": int(interrupted), "pending": int(not terminal and (dispatched or event.get("pending") is True)),
            "no_answer": int(dispatched and terminal and not interrupted and event.get("answer_received") is False)}


def fold(prior, event, previous=None):
    """Subtract the prior observation of this immutable producer before adding it."""
    result = copy.deepcopy(prior or {"row_id": event["row_id"], "row_identity": event["row_identity"],
        "purpose": event["purpose"], **dict.fromkeys(COUNTS, 0), "observations": 0,
        "duration_samples": {"count": 0, "sum_sec": 0.0}, "denominators": {"non_interrupted_terminal_dispatches": 0},
        "window": {"from": event.get("ts"), "through": event.get("ts"), "basis": "instrumented_publications"}})
    for row, sign in ((previous, -1), (event, 1)):
        if not row:
            continue
        result["observations"] += sign
        for key, value in _counts(row).items():
            result[key] += sign * value
        result["denominators"]["non_interrupted_terminal_dispatches"] += sign * int(
            row.get("dispatched") is True and row.get("terminal") is True and row.get("interrupted") is not True)
        duration = row.get("duration_sec")
        if number(duration):
            result["duration_samples"]["count"] += sign
            result["duration_samples"]["sum_sec"] += sign * duration
    stamps = [v for v in (result["window"].get("from"), result["window"].get("through"), event.get("ts")) if v]
    result["window"].update({"from": min(stamps) if stamps else None, "through": max(stamps) if stamps else None})
    for key in ("queue", "quota", "nanny_billing"):
        if event.get(key) is not None:
            old = result.get(key) or {}
            if str(event.get("ts") or "") >= str(old.get("observed_at") or ""):
                result[key] = {"observed_at": event.get("ts"), "producer_id": event["producer_id"], "fact": event[key]}
        result.setdefault(key, {"state": "unknown", "reason": "not_observed"})
    result["source_refs"] = {"producer_set": PRODUCERS, "latest_producer_id": event["producer_id"],
                             "latest": event.get("source_refs") or []}
    if event.get("duration_basis"):
        result["duration_basis"] = event["duration_basis"]
    return result


# Physical memory budget for one decoded suffix and its atomic projection. A single
# larger event stays whole. Only the child consumes an unbounded backlog.
BATCH_BYTES = 256 * 1024
JOB = "subagent_metrics"


def _projection(root, events):
    """Capture addressed inputs, then fold with no store transaction held."""
    addresses = set()
    for event in events:
        if event.get("type") == EVENT:
            addresses.add((PRODUCERS, str(event["producer_id"])))
            if event.get("row_id"):
                addresses.add(_group(event))
            if (event.get("actor") or {}).get("kind") == "api_call":
                addresses.add((CALLS, event["producer_id"].removeprefix("call:")))
    with store.open_store(root, write=False) as txn:
        values = {address: txn.get_row(*address) for address in addresses}
    updates, observed, tasks = {}, set(), set()

    def put(address, value):
        values[address] = updates[address] = value

    for event in events:
        if event.get("type") in {"llm_round", "llm_api_error"} and event.get("llm_call_id"):
            put((CALLS, str(event["llm_call_id"])), {key: event[key] for key in
                ("llm_call_id", "task_id", "ledger_attempt_ids", "physical_attempt_id") if key in event})
            continue
        if event.get("type") != EVENT:
            continue
        event = copy.deepcopy(event)
        actor = event.get("actor") or {}
        if actor.get("kind") == "api_call":
            from ouroboros.subagent_metric_events import _call_evidence
            call_id = event["producer_id"].removeprefix("call:")
            call = values.get((CALLS, call_id)) or {}
            if call.get("ledger_attempt_ids"):
                actor["usage_ref"] = {"ledger_attempt_ids": call["ledger_attempt_ids"],
                                      "observed_attempt_id": call.get("physical_attempt_id")}
                event.update(_call_evidence(root, call_id, refs=call))
        producer = str(event["producer_id"])
        observed.add(producer)
        previous = values.get((PRODUCERS, producer))
        if previous == event or previous and previous.get("terminal") is True and event.get("terminal") is False:
            continue
        if previous and _group(previous) != _group(event):
            raise ValueError("producer identity changed")
        if event.get("row_id"):
            address = _group(event)
            put(address, fold(values.get(address), event, previous))
        put((PRODUCERS, producer), event)
        if event.get("task_id"):
            tasks.add(event["task_id"])
            put((set_name("swarm_actors", event["task_id"]), producer), event)
    return updates, observed, tasks


def _defer(root):
    from ouroboros import maintenance_jobs
    from ouroboros.maintenance import child

    source = _source_stamp(root)
    with store.open_store(root, write=True) as txn:
        cursor = txn.meta(CURSOR) or {}
        coverage = {**cursor.get("coverage", {}), "state": "unknown", "stream": "pending",
                    "history_before_window": "unknown"}
        if cursor.get("coverage") != coverage:
            txn.set_meta(CURSOR, {**cursor, "coverage": coverage})
        job = txn.meta("job:" + JOB) or {}
        if job.get("state") != "pending" or job.get("requested_source") != source:
            maintenance_jobs.record_in(txn, JOB, state="pending", requested_source=source)
    child.request()


def _commit_suffix(root, cursor, events, through, upper, source):
    updates, observed, tasks = _projection(root, events)
    missing = sorted(set(cursor.get("missing_producers") or ()) - observed)
    next_cursor = {"through": through, "source_root": str(pathlib.Path(root).resolve()),
        "missing_producers": missing, "source": source if through == upper else None,
        "live": source[0][:2] if through == upper and source[0] else None,
        "coverage": {"state": "partial" if missing else "observed_window", "history_before_window": "unknown",
                     "missing_publications": len(missing), "stream": "complete" if through == upper else "incomplete"}}
    if next_cursor != cursor or updates:
        with store.open_store(root, write=True) as txn:
            if (txn.meta(CURSOR) or {}) != cursor:
                return False  # another fold or publication-gap observation won
            for address, value in updates.items():
                txn.put_row(*address, value)
            txn.set_meta(CURSOR, next_cursor)
    from ouroboros.swarm_receipt import refresh_for_task
    for task_id in tasks:
        refresh_for_task(root, task_id)
    return True


def _publish_suffix(root, before, publication=None):
    """A warm publisher reads only its bounded live suffix, with an inode/EOF proof.

    Missing/stale cursor, rotation, or a large suffix schedules the child before
    any history discovery. Pinning the live handle avoids an archive search even
    if rotation races this read. The absolute cursor includes its consumed prefix.
    """
    with store.open_store(root, write=False) as txn:
        cursor = txn.meta(CURSOR) or {}
    old_live = cursor.get("live")
    if old_live is None and (cursor or any(before)):
        _defer(root)
        return False
    current = _source_stamp(root)
    live = current[0]
    start = old_live[1] if old_live else 0
    if (not live or (old_live and old_live[0] != live[0])
            or not 0 <= live[1] - start <= BATCH_BYTES):
        _defer(root)
        return False
    with (pathlib.Path(root) / "logs/events.jsonl").open("rb") as handle:
        actual = os.fstat(handle.fileno())
        if (actual.st_ino, actual.st_size, actual.st_mtime_ns) != tuple(live):
            _defer(root)
            return False
        handle.seek(start)
        raw = handle.read(live[1] - start)
    if len(raw) != live[1] - start or raw and not raw.endswith(b"\n"):
        _defer(root)
        return False
    try:
        events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except ValueError:
        _defer(root)
        return False
    events = [event for event in events if isinstance(event, dict)]
    through = int(cursor.get("through", 0)) + len(raw)
    if not _commit_suffix(root, cursor, events, through, through, current):
        _defer(root)
        return False
    from ouroboros import maintenance_jobs
    with store.open_store(root, write=True) as txn:
        job = txn.meta("job:" + JOB) or {}
        if job.get("publication_owed") == publication and job.get("state") != "done":
            maintenance_jobs.record_in(txn, JOB, state="done", source=current, publication_owed=None)
    return True


def catch_up(root, *, should_stop=None):
    """Child/explicit repair: fold one stable chain outside writes; CAS bounded batches."""
    from ouroboros.swarm_rollup import fold_event_suffix
    from ouroboros.jsonl_tail import JsonlChainSnapshot

    root, cursor = pathlib.Path(root), None
    try:
        stamp = _source_stamp(root)
        source = JsonlChainSnapshot(root / "logs/events.jsonl")
        live = next((info for _path, info, is_live in source.entries if is_live), None)
        stamp[0] = [live.st_ino, live.st_size, live.st_mtime_ns] if live else None
        while not (should_stop and should_stop()):
            with store.open_store(root, write=False) as txn:
                cursor = txn.meta(CURSOR) or {}
            start = int(cursor.get("through", 0))
            events = []
            through = fold_event_suffix(root, start, events.append, source=source, max_bytes=BATCH_BYTES)
            if not _commit_suffix(root, cursor, events, through, source.upper, stamp):
                break
            if through == source.upper or through == start:
                return through == source.upper
        _defer(root)
        return False
    except Exception:
        with store.open_store(root, write=True) as txn:
            old = txn.meta(CURSOR) or {}
            if cursor is None or old == cursor:
                unknown = {**old, "coverage": {"state": "unknown", "stream": "unavailable",
                                               "history_before_window": "unknown"}}
                if unknown != old:
                    txn.set_meta(CURSOR, unknown)
        raise


def _source_stamp(root):
    stamps = []
    for name in ("logs/events.jsonl", "archive"):
        try:
            info = (pathlib.Path(root) / name).stat()
            stamps.append([info.st_ino, info.st_size, info.st_mtime_ns])
        except FileNotFoundError:
            stamps.append(None)
    return stamps


def job_due(root, row, facts, child):
    if row.get("state") in {"pending", "running"}:
        return True
    if row.get("state") == "failed":
        return row.get("source") != _source_stamp(root)  # changed failed source, including a repaired torn tail
    return bool(row.get("publication_owed")) or not row.get("state") and any(_source_stamp(root))


def run_job(root, should_stop):
    from ouroboros import maintenance_jobs
    from ouroboros.jsonl_tail import JsonlChainSnapshot

    # The recorded job request is fenced too: a publisher racing the last fold
    # keeps the job pending. A torn line rests until a publication requests it.
    source = _source_stamp(root)
    with store.open_store(root, write=False) as txn:
        request = txn.meta("job:" + JOB)
    try:
        complete = catch_up(root, should_stop=should_stop)
    except Exception as exc:
        with store.open_store(root, write=True) as txn:
            if txn.meta("job:" + JOB) == request:
                maintenance_jobs.record_in(txn, JOB, state="failed", error=type(exc).__name__, source=source)
        raise
    upper = JsonlChainSnapshot(root / "logs/events.jsonl").upper
    state = "pending"
    with store.open_store(root, write=True) as txn:
        cursor = txn.meta(CURSOR) or {}
        if txn.meta("job:" + JOB) == request and not should_stop():
            if complete and cursor.get("through") == upper:
                state = "done"
            elif (cursor.get("coverage") or {}).get("stream") == "incomplete":
                state = "failed"  # a torn line waits for its writer, never for a timer
            maintenance_jobs.record_in(txn, JOB, state=state, through=cursor.get("through"),
                                       coverage=cursor.get("coverage"), source=source,
                                       publication_owed=None if state == "done" else (request or {}).get("publication_owed"))
    return {"state": state, "items": 0}


@store.acquisition_budget(0)
def publish(root, event):
    """Keep the observation before the derived index; failure never changes execution."""
    if not root or not event.get("producer_id"):
        return
    try:
        before = _source_stamp(root)
        publication = None
        event = {**copy.deepcopy(event), "type": EVENT}
        previous = {}
        try:
            with store.open_store(root, write=False) as txn:
                previous = txn.get_row(PRODUCERS, str(event["producer_id"])) or {}
        except Exception:
            pass  # Duplicate suppression cannot prevent the canonical append.
        if (event.get("actor") or {}).get("kind") == "api_call" and (previous.get("actor") or {}).get("usage_ref"):
            event["actor"].setdefault("usage_ref", copy.deepcopy(previous["actor"]["usage_ref"]))
        if {key: value for key, value in event.items() if key != "ts"} != {
                key: value for key, value in previous.items() if key != "ts"}:
            event["ts"] = event.get("ts") or previous.get("ts") or utc_now_iso()
            from ouroboros import maintenance_jobs
            publication = identity_key(event)
            try:
                with store.open_store(root, write=True) as txn:
                    maintenance_jobs.record_in(txn, JOB, state="pending", publication_owed=publication)
            except Exception:
                log.debug("Measurement wake awaits the next publisher or explicit catch-up", exc_info=True)
            if not append_jsonl(pathlib.Path(root) / "logs/events.jsonl", event):
                with store.open_store(root, write=True) as txn:
                    cursor = txn.meta(CURSOR) or {}
                    missing = sorted(set(cursor.get("missing_producers") or ()) | {event["producer_id"]})
                    txn.set_meta(CURSOR, {**cursor, "missing_producers": missing,
                        "coverage": {**cursor.get("coverage", {}), "state": "partial", "missing_publications": len(missing)}})
                raise OSError("measurement journal append failed")
        try:
            _publish_suffix(root, before, publication)
        except Exception:
            _defer(root)  # failed projection is unknown even when the old prefix was complete
            raise
        from ouroboros.swarm_receipt import refresh_for_task
        refresh_for_task(root, event.get("task_id"))
    except Exception:
        log.warning("subagent measurement projection unavailable", exc_info=True)


def read(root, row_id, *, row_identity=None, purpose=None):
    """No history read, provider call or write on a model round."""
    unknown = {"state": "unknown", "stream": "not_observed", "history_before_window": "unknown"}
    try:
        with store.open_store(root, write=False) as txn:
            coverage = (txn.meta(CURSOR) or {}).get("coverage", unknown)
            if row_identity is not None:
                purposes = (purpose,) if purpose else ("review", "plan", "execution")
                rows = [txn.get_row(set_name("subagent_metrics", row_id), identity_key(row_identity) + ":" + p) for p in purposes]
            else:
                rows = list(txn.set_rows(set_name("subagent_metrics", row_id)).values())
        return {"coverage": coverage, "rows": [{**row, "coverage": coverage} for row in rows
                                                if row and (not purpose or row["purpose"] == purpose)]}
    except Exception:
        return {"coverage": {**unknown, "stream": "unavailable"}, "rows": []}


def fact_line(task, *, drive_root):
    """A compact dynamic fact for this task's selected row, outside cached catalogs."""
    from ouroboros.subagent_history import execution_identity

    snapshot = task.get("configured_subagent") or {}
    row_id = str(snapshot.get("selected_subagent_id") or "")
    if not row_id:
        return ""
    facts = read(drive_root, row_id, row_identity=execution_identity(snapshot))
    if not facts["rows"]:
        return ("Subagent measurements: " + json.dumps({"coverage": facts["coverage"]}, separators=(",", ":"))
                if facts["coverage"].get("stream") in {"pending", "unavailable"} else "")
    return "Subagent measurements (observed window; earlier history unknown): " + json.dumps(
        [{key: row[key] for key in ("purpose", "dispatched", "terminal", "answer_received", "usable_verdict",
                                    "no_answer", "pending", "interrupted", "coverage")} for row in facts["rows"]],
        separators=(",", ":"))
