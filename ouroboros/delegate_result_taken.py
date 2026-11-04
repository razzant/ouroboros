"""Tell the engine once that Ouroboros has taken a finished delegated result (owner Q2=A, 05.11.2026).

A delegated Git run's patch, output and named evidence are captured into its task record
(``delegate_collection``), and Ouroboros applies or rejects that capture itself, never through the
engine. Unless told, the engine keeps its own copy of every such unapplied patch as actionable
forever (Claudexor ``retention-candidates.ts``). One typed decision fixes that: the ``taken`` action
of the engine's run decision route (``ClaudexorGateway.take_run_result``), after which the engine's
copy follows its existing age and keep-last window; Ouroboros's own copy is untouched.

The decision is sent only when it can no longer cost Ouroboros anything: the run is terminal and
settled, its collection is complete with the patch captured (``collection_pending`` false), and its
line can no longer be continued, because the whole tree is over (``review_thread_retention.tree_hold``,
the same proof that releases a finished reviewer thread). A plan, a review seat, a directory
(``files``) result, an in-place run, a superseded or mid-apply run, or a run whose capture can only
heal with new input never gets one. The runtime-retirement pass asks for every task written terminal
and every task it left waiting (``review_thread_retention.retire_finished_review_threads``), so no
timer, sweep or counter decides it. The outcome is one record per run in the task document,
``delegated_result_taken``: ``taken``; ``refused`` with the engine's code (a run the engine can still
continue, a result that is no patch); ``absent`` (the engine has no such run); or
``route_unavailable`` when the serving engine's decision vocabulary names no ``taken``, so an older
engine gets no request and the record says why. An unknown answer writes nothing and leaves the task
waiting; the next pass repeats the same keyed decision, which the engine answers once.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict

log = logging.getLogger(__name__)

RECORDS = "delegated_result_taken"  # task-document key: run id -> its one decision record


def _candidate(entry: Any) -> bool:
    """A run whose engine copy is a patch Ouroboros captures and decides itself: a Git copy's run."""
    from ouroboros.delegate_directory import is_in_place_run

    return bool(not entry.review_owned and entry.mode != "plan" and entry.snapshot_id and entry.execution_root
                and entry.resource_ref.get("workspace_kind") != "directory" and not is_in_place_run(entry)
                and not entry.superseded_by and not entry.patch_apply_pending)


def owed(entry: Any) -> str:
    """``take`` when the run's result is held whole, ``wait`` while its collection will still be tried,
    else ``""`` (never this decision's, or uncaptured with nothing left to try)."""
    from ouroboros.delegate_capture_attempt import collection_due
    from ouroboros.delegate_custody import TERMINAL_STATES
    from ouroboros.delegate_custody_current import collection_pending

    if not _candidate(entry):
        return ""
    if not entry.settled:
        return "wait"  # custody still open: the tree proof keeps the task waiting anyway
    if entry.terminal_state not in TERMINAL_STATES:
        return ""  # closed without a terminal (proven absent): the engine holds no result
    if collection_pending(entry):
        return "wait" if collection_due(entry) else ""  # a stuck collection heals only with new input
    return "take" if entry.patch_captured else ""


def _decision(engine: Any, run_id: str) -> Dict[str, Any]:
    """One keyed ``taken`` decision; its record, or ``{}`` when the answer proves nothing."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    key = hashlib.sha256(f"taken:{run_id}".encode()).hexdigest()
    try:
        answer = engine.take_run_result(run_id, idempotency_key=key)
    except ClaudexorUnavailable as exc:
        status = int(getattr(exc, "status_code", 0) or 0)
        if status == 404:
            return {"state": "absent", "store_id": str(getattr(engine, "store_identity", "") or "")}
        if 400 <= status < 500 and status not in (408, 429):  # a typed refusal is the engine's settled answer
            return {"state": "refused", "code": str(exc.code or ""), "status": status}
        log.info("Taken decision for %s unanswered: %s", run_id, exc)
        return {}
    if isinstance(answer, dict) and answer.get("accepted") is True and answer.get("status") == "taken":
        return {"state": "taken"}
    return {"state": "refused", "code": str((answer or {}).get("status") or "unaccepted")}


def tell_engine(run: Any, task_id: str) -> str:
    """The runtime-retirement pass's duty for one task's results: ``WAIT`` keeps the task for a later
    pass, ``DONE`` lets it go until its next terminal write. ``run`` is that pass
    (``review_thread_retention._Pass``: its root, liveness probe, census, engine, holds and stop)."""
    from ouroboros.delegate_custody_current import runs
    from ouroboros.owner_keep import is_kept
    from ouroboros.review_thread_retention import DONE, WAIT, tree_hold
    from ouroboros.task_store import patch_document, read_document
    from ouroboros.utils import utc_now_iso

    document = read_document(run.root, task_id) or {}
    recorded = document.get(RECORDS) if isinstance(document.get(RECORDS), dict) else {}
    if not any(owed(entry) for rid, entry in runs(run.root, "task_id=?", (task_id,)).items() if rid not in recorded):
        return DONE
    if is_kept(run.root, task_id):  # the owner keeps this task's materials: its engine copies stay too
        run.hold("result_owner_keep")
        return DONE
    why, held = tree_hold(run.root, task_id, document, live=run.live, census=run.census())
    states = {rid: owed(entry) for rid, entry in held.items() if rid not in recorded}
    if why or "wait" in states.values():
        run.hold(f"result_{why or 'collection_owed'}")
        return WAIT
    take = sorted(rid for rid, state in states.items() if state == "take")
    if not take:
        return DONE
    engine = run.engine()
    try:
        supported = engine is not None and engine.run_result_taken_supported()
    except Exception:  # an unread vocabulary proves no route either way
        engine = None
    if engine is None:
        run.hold("result_engine_unavailable")
        return WAIT
    answers: Dict[str, Dict[str, Any]] = {}
    verdict, done = DONE, 0
    for rid in take:
        if done and run.stop():  # the rest waits for the next pass
            run.stopped, verdict = True, WAIT
            break
        done += 1
        record = _decision(engine, rid) if supported else {
            "state": "route_unavailable", "engine_version": str(getattr(engine, "engine_version", "") or "")}
        if not record:
            run.hold("result_taken_unanswered")
            verdict = WAIT
            continue
        answers[rid] = {**record, "at": utc_now_iso()}
    if answers:
        patch_document(run.root, task_id, lambda current: {**current, RECORDS: {
            **(current.get(RECORDS) if isinstance(current.get(RECORDS), dict) else {}), **answers}})
        for rid, record in answers.items():
            run.report["observed"].append({"task_id": task_id, "run_id": rid, "taken": record["state"]})
    return verdict
