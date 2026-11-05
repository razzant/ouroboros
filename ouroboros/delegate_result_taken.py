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
continue, a result that is no patch); or ``absent`` (the engine has no such run).
A catalog without ``taken`` holds the task without a decision record; capability is read once per pass.
An unknown answer also writes nothing and leaves the task
waiting; the next pass repeats the same key. Interrupted delivery permits one fixed derived-key resend.
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
    """A keyed decision, with one derived-key resend for interrupted delivery; only a settled answer records."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    key = hashlib.sha256(f"taken:{run_id}".encode()).hexdigest()
    retry_key = hashlib.sha256(f"{key}:interrupted".encode()).hexdigest()
    for attempt_key in (key, retry_key):
        try:
            answer = engine.take_run_result(run_id, idempotency_key=attempt_key)
        except ClaudexorUnavailable as exc:
            status = int(getattr(exc, "status_code", 0) or 0)
            if status == 404:
                return {"state": "absent", "store_id": str(getattr(engine, "store_identity", "") or "")}
            if status == 400 or (status == 409 and str(exc.code or "").startswith("run_taken_")):
                return {"state": "refused", "code": str(exc.code or ""), "status": status}
            if status == 409 and exc.code == "delivery_interrupted" and attempt_key == key:
                continue  # the engine answers taken again if the first delivery already landed
            log.info("Taken decision for %s unanswered: %s", run_id, exc)
            return {}
        if isinstance(answer, dict) and answer.get("accepted") is True and answer.get("status") == "taken":
            return {"state": "taken"}
        return {}
    return {}


def _unavailable(run: Any) -> str:
    """This pass's one capability read: an unsupported or unreadable catalog keeps the duty waiting."""
    if not hasattr(run, "_taken_unavailable"):
        engine = run.engine()
        try:
            run._taken_unavailable = ("result_engine_unavailable" if engine is None else
                                      "" if engine.run_result_taken_supported() else "result_taken_unsupported")
        except Exception:
            run._taken_unavailable = "result_engine_unavailable"
    return run._taken_unavailable


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
        return WAIT
    why, held = tree_hold(run.root, task_id, document, live=run.live, census=run.census())
    states = {rid: owed(entry) for rid, entry in held.items() if rid not in recorded}
    if why or "wait" in states.values():
        run.hold(f"result_{why or 'collection_owed'}")
        return WAIT
    take = sorted(rid for rid, state in states.items() if state == "take")
    if not take:
        return DONE
    why = _unavailable(run)
    if why:
        run.hold(why)
        return WAIT
    engine = run.engine()
    answers: Dict[str, Dict[str, Any]] = {}
    verdict, done = DONE, 0
    for rid in take:
        if done and run.stop():  # the rest waits for the next pass
            run.stopped, verdict = True, WAIT
            break
        done += 1
        record = _decision(engine, rid)
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
