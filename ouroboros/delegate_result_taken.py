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

A result whose task settled before the pass could ask (before its first pass, or before this decision
shipped) gets the same duty once from ``walk`` (owner, 05.11.2026): while the ``runtime_retirement`` job
row's ``taken_walk`` record has no ``done_at``, custody holds a run and the pass's capability read finds
``taken``, the pass also asks every other settled task holding a result owed ``taken``, with this duty
alone: never a thread release or a managed document import, which keep their activation rule. A task
still running is the pass's own at its terminal write. Each run gets the same keyed decision and record,
so a recorded run is never asked again. A task the work itself holds (the owner's Keep, a tree not over,
a collection owed) is counted, never kept waiting; an answer or a read that did not come (``_UNREAD``),
a stop, or a queue census that proves nothing leaves the walk open for the step's next pass. The record
keeps plain counts of the walk's own decisions: ``taken``, ``absent`` and ``refused`` by the engine's code
as each task's decisions land (a task its record writes again is the pass's own from then on), then
``held`` by reason and ``done_at`` once one pass asked its whole set without either. After the
maintenance child run that leaves it done, the server tells the owner in Main once
(``deliver_receipt``: a one-time notice whose landed row is its receipt). No timer, count or threshold
decides any of it: the engine's capability and the record do.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, Iterable, Tuple

log = logging.getLogger(__name__)

RECORDS = "delegated_result_taken"  # task-document key: run id -> its one decision record
WALK = "taken_walk"  # the runtime_retirement job row's field: the one walk's counts, then its ``done_at``
NOTICE = "taken_walk_notice"  # the walk's one-time receipt in the owner's chat (``notice_receipts.UPGRADE_TYPES``)
# Why a walk task went undecided THIS pass (an answer or a read that did not come), never why its work holds it.
_UNREAD = ("engine_unavailable", "taken_unsupported", "taken_unanswered", "census_unknown", "tree_unreadable",
           "custody_unreadable")
_HELD_WORDS = {"owner_keep": "kept by you", "continuable": "still continuable",
               "collection_owed": "still being collected"}


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
    return _tell(run, task_id)[0]


def _tell(run: Any, task_id: str) -> Tuple[str, str, Dict[str, Dict[str, Any]]]:
    """``tell_engine``'s verdict, the hold it noted last (``""``: none, or a stop) and the records it wrote."""
    from ouroboros.delegate_custody_current import runs
    from ouroboros.owner_keep import is_kept
    from ouroboros.review_thread_retention import DONE, WAIT, tree_hold
    from ouroboros.task_store import patch_document, read_document
    from ouroboros.utils import utc_now_iso

    def held(why: str) -> Tuple[str, str, Dict[str, Dict[str, Any]]]:
        run.hold(why)
        return WAIT, why, {}

    document = read_document(run.root, task_id) or {}
    recorded = document.get(RECORDS) if isinstance(document.get(RECORDS), dict) else {}
    if not any(owed(entry) for rid, entry in runs(run.root, "task_id=?", (task_id,)).items() if rid not in recorded):
        return DONE, "", {}
    if is_kept(run.root, task_id):  # the owner keeps this task's materials: its engine copies stay too
        return held("result_owner_keep")
    why, tree_runs = tree_hold(run.root, task_id, document, live=run.live, census=run.census())
    states = {rid: owed(entry) for rid, entry in tree_runs.items() if rid not in recorded}
    if why or "wait" in states.values():
        return held(f"result_{why or 'collection_owed'}")
    take = sorted(rid for rid, state in states.items() if state == "take")
    if not take:
        return DONE, "", {}
    why = _unavailable(run)
    if why:
        return held(why)
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
            verdict, why = WAIT, "result_taken_unanswered"
            continue
        answers[rid] = {**record, "at": utc_now_iso()}
    if answers:
        patch_document(run.root, task_id, lambda current: {**current, RECORDS: {
            **(current.get(RECORDS) if isinstance(current.get(RECORDS), dict) else {}), **answers}})
        for rid, record in answers.items():
            run.report["observed"].append({"task_id": task_id, "run_id": rid, "taken": record["state"]})
    return verdict, why, answers


# ---- the one-time walk and its receipt -------------------------------------------------------------------------


def walk(run: Any, visited: Iterable[str], row: Dict[str, Any]) -> None:
    """The one-time walk (module docstring) after a runtime-retirement pass asked its own tasks (``visited``);
    ``row`` is the job row that pass read. Never raises: a failure is the pass's error and leaves the walk open."""
    try:
        _walk(run, set(visited), row)
    except Exception as exc:
        run.report["errors"].append(f"taken walk: {type(exc).__name__}: {exc}"[:300])
        log.warning("The taken walk failed; the step's next pass asks again", exc_info=True)


def _walk(run: Any, visited: set, row: Dict[str, Any]) -> None:
    from ouroboros import maintenance_jobs
    from ouroboros.delegate_custody_current import has_runs, runs
    from ouroboros.maintenance_retirement import JOB
    from ouroboros.review_thread_retention import WAIT
    from ouroboros.task_status import SETTLED_STATUSES
    from ouroboros.task_store import task_versions
    from ouroboros.utils import utc_now_iso

    counts = dict(row[WALK]) if isinstance(row.get(WALK), dict) else {}
    if counts.get("done_at") or run.stopped or not has_runs(run.root) or _unavailable(run):
        return  # over; the pass stopped; nothing delegated yet; or the engine cannot take a result now
    owners = {entry.task_id for entry in runs(run.root).values() if entry.task_id not in visited and owed(entry) == "take"}
    tasks = sorted(task for task, (_seq, status, _owes) in task_versions(run.root, owners).items()
                   if status in SETTLED_STATUSES)
    held: Dict[str, int] = {}
    whole = True
    for task_id in tasks:
        if run.stop():
            run.stopped = True
            return
        if run.census() is None:  # a stale queue snapshot proves no tree over: the rest waits for a later pass
            return
        verdict, why, answers = _tell(run, task_id)
        if answers:  # with the records they count, so an interrupted pass loses none of its count
            counts = _counted(counts, answers.values())
            maintenance_jobs.record(run.root, JOB, **{WALK: counts})
        if run.stopped:
            return
        reason = why.removeprefix("result_")
        if reason in _UNREAD or reason.removeprefix("continued_") in _UNREAD:
            whole = False
        elif verdict == WAIT:
            held[reason] = held.get(reason, 0) + 1
    if whole:
        done = {**_counted(counts, ()), "held": held, "done_at": utc_now_iso()}
        maintenance_jobs.record(run.root, JOB, **{WALK: done})
        run.report["observed"].append({WALK: done})


def _counted(counts: Dict[str, Any], answers: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """COUNTS with ANSWERS added: ``taken``, ``absent`` and ``refused`` by the engine's code, plain integers."""
    out = {"taken": int(counts.get("taken") or 0), "absent": int(counts.get("absent") or 0),
           "refused": dict(counts.get("refused") or {})}
    for answer in answers:
        if answer["state"] == "refused":
            code = str(answer.get("code") or answer.get("status") or "unknown")
            out["refused"][code] = out["refused"].get(code, 0) + 1
        else:
            out[answer["state"]] += 1
    return out


def receipt(record: Dict[str, Any]) -> str:
    """The owner's one line about a finished walk's counts; ``""`` when the walk met nothing to tell."""
    def number(value: Any) -> int:
        return value if type(value) is int and value > 0 else 0

    refused = {str(code): number(n) for code, n in sorted((record.get("refused") or {}).items()) if number(n)}
    held: Dict[str, int] = {}
    for why, n in sorted((record.get("held") or {}).items()):
        if number(n):
            words = _HELD_WORDS.get(str(why), "in work not yet over")
            held[words] = held.get(words, 0) + number(n)
    taken, absent = number(record.get("taken")), number(record.get("absent"))
    clauses = [f"the engine now knows Ouroboros has taken {taken}, so their engine copies follow its ordinary "
               "retention window while Ouroboros keeps its own copy of each"] if taken else []
    if refused:
        clauses.append(f"the engine refused {sum(refused.values())} ("
                       + ", ".join(f"{code} {n}" for code, n in refused.items()) + ")")
    if absent:
        clauses.append(f"the engine no longer had {absent}")
    if held:
        clauses.append(f"{sum(held.values())} stay held (" + ", ".join(f"{n} {words}" for words, n in held.items()) + ")")
    return "Earlier delegated results: " + "; ".join(clauses) + "." if clauses else ""


def deliver_receipt(root: Any) -> None:
    """After a maintenance child run (``MaintenanceChild._run``): the finished walk's receipt in the owner's chat,
    once, never raising. Its type is a one-time notice (``notice_receipts.UPGRADE_TYPES``) recorded when its row
    lands; nothing is sent or marked while no owner chat is bound or the dialogue cannot answer, so a later child
    run delivers it."""
    try:
        from ouroboros import maintenance_jobs
        from ouroboros.maintenance_retirement import JOB
        from ouroboros.notice_receipts import recorded
        from supervisor import message_bus
        from supervisor.state import control_value, load_state

        record = (maintenance_jobs.job(root, JOB) or {}).get(WALK)
        text = receipt(record) if isinstance(record, dict) and record.get("done_at") else ""
        known, owner_chat = control_value(load_state(), "owner_chat_id") if text else (False, None)
        if known and owner_chat and not recorded(root, int(owner_chat), NOTICE):
            message_bus.send_with_budget(int(owner_chat), text, role="system", system_type=NOTICE,
                                         require_write=True, ensure_record_boundary=True)
    except Exception:
        log.debug("The taken walk's receipt waits for a later maintenance child run", exc_info=True)
