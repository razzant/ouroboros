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
shipped) gets the same duty from ``walk`` (owner, 05.11.2026), once: while the ``runtime_retirement``
job row holds no ``taken_walk`` record with ``done_at``, custody holds a run and this pass's capability
read finds ``taken``, the pass also asks every other task owing a result whose row is settled, or
``interrupted`` with a retry whose effective result is (a timeout-retried predecessor). A task still
running is the pass's own at its terminal write. The walk asks this duty alone, under its own per-task
guard; its record is a terminal write, so the next pass treats the task as any terminal write (threads
only for host runs after activation, owed imports under their own release rule). Every task it cannot
decide (a hold of any kind, an error, an unanswered decision, the task in hand at a stop) is left to the
step's ``waiting`` set, where the ordinary duty asks it again, never proving ``taken`` from a read it
could not make. The first answer that is not a settled record, whatever its code, also ends the walk for
that pass, as do a stop and a queue census that goes stale; the next pass moves past the tasks now
waiting, so one failing run costs one pass and a broken engine moves one task per pass. The walk is over
(``done_at``) once one pass asked its whole set. Until then it is open, the step being activated without
that record (``walk_open``): work the step does whenever the maintenance child runs, never by itself a
reason to start one, so an engine without ``taken`` is contacted once per such run while custody holds a
run. The step's first pass walks as soon as custody holds a run, and with none it closes the walk
before it begins (``empty_walk``). Each record the walk writes says so (``walk``), and its counts come
from those records when it is over: ``taken``, ``absent`` and ``refused`` by the engine's code.
``waiting`` counts the runs it left to the ordinary duty, and ``unknown`` the runs of a task whose
document it could not read (never counted as waiting, and never as zero), both written in the
transaction that adds their tasks to ``waiting``. The step's report gets one summary of the walk, never
a line per run. After a maintenance child run that Stop or Panic did not end, once the walk is over, the
server tells the owner in Main once (``deliver_receipt``: a one-time notice whose landed row is its
receipt). No timer, count or threshold decides any of it: the engine's capability and the record do.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, Iterable, List, Set, Tuple

log = logging.getLogger(__name__)

RECORDS = "delegated_result_taken"  # task-document key: run id -> its one decision record
WALK = "taken_walk"  # the runtime_retirement job row's field: the walk's record, ``done_at`` once it is over
NOTICE = "taken_walk_notice"  # the walk's one-time receipt in the owner's chat (``notice_receipts.UPGRADE_TYPES``)
# The engine's typed refusal codes, in the owner's words (``receipt``); any other code is shown as it is.
_REFUSALS = {"run_taken_continuable": "still continuable in the engine", "run_taken_not_patch": "not a patch result"}


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
    """A keyed decision, with one derived-key resend for interrupted delivery; only a settled answer (``state``)
    records, and ``{}`` is any other answer, whatever its code."""
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


def _unavailable(run: Any, entry: Any = None) -> str:
    """This pass's capability read of the engine holding ``entry``'s run (the owned engine when it names none), asked
    once per engine: an unsupported or unreadable catalog keeps that engine's duties waiting, never another's."""
    from ouroboros.execution_domain import engine_key

    known = run.__dict__.setdefault("_taken_unavailable", {})
    key = engine_key(entry) if entry is not None else None
    if key not in known:
        engine = run.work.reach(entry) if key is not None else run.engine()
        try:
            known[key] = ("result_engine_unavailable" if engine is None else
                          "" if engine.run_result_taken_supported() else "result_taken_unsupported")
        except Exception:
            known[key] = "result_engine_unavailable"
    return known[key]


def _work_takes(run: Any) -> bool:
    """Whether this deployment's work engine (``execution_domain``) can take results now, whatever the owned one can."""
    from ouroboros.execution_domain import WORK_ENGINE, current

    domain = current()
    return bool(domain and domain.engine_descriptor) and not _unavailable(
        run, {"kind": WORK_ENGINE, "descriptor": domain.engine_descriptor})


def tell_engine(run: Any, task_id: str) -> str:
    """The runtime-retirement pass's duty for one task's results: ``WAIT`` keeps the task for a later
    pass, ``DONE`` lets it go until its next terminal write. ``run`` is that pass
    (``review_thread_retention._Pass``: its root, liveness probe, census, engine, holds and stop)."""
    return _tell(run, task_id)[0]


def _tell(run: Any, task_id: str, *, walk: bool = False) -> Tuple[str, Dict[str, Dict[str, Any]], bool]:
    """``tell_engine``'s verdict, the task's decision records after it, and whether a decision got an answer that was
    not a settled record. For the walk, each record it writes says so (``walk``), its answers reach the step's report
    only as the walk's summary, and the first unsettled answer ends its decisions."""
    from ouroboros.delegate_custody_current import runs
    from ouroboros.owner_keep import is_kept
    from ouroboros.review_thread_retention import DONE, WAIT, tree_hold
    from ouroboros.task_store import patch_document, read_document
    from ouroboros.utils import utc_now_iso

    document = read_document(run.root, task_id) or {}
    recorded = dict(document[RECORDS]) if isinstance(document.get(RECORDS), dict) else {}
    if not any(owed(entry) for rid, entry in runs(run.root, "task_id=?", (task_id,)).items() if rid not in recorded):
        return DONE, recorded, False
    if is_kept(run.root, task_id):  # the owner keeps this task's materials: its engine copies stay too
        run.hold("result_owner_keep")
        return WAIT, recorded, False
    why, tree_runs = tree_hold(run.root, task_id, document, live=run.live, census=run.census())
    states = {rid: owed(entry) for rid, entry in tree_runs.items() if rid not in recorded}
    if why or "wait" in states.values():
        run.hold(f"result_{why or 'collection_owed'}")
        return WAIT, recorded, False
    take = sorted(rid for rid, state in states.items() if state == "take")
    if not take:
        return DONE, recorded, False
    whys = {rid: _unavailable(run, tree_runs.get(rid)) for rid in take}  # each run's own engine (``execution_domain``)
    if all(whys.values()):  # no engine holding these results can be told now
        run.hold(whys[take[0]])
        return WAIT, recorded, False
    answers: Dict[str, Dict[str, Any]] = {}
    verdict, done, unsettled = DONE, 0, False
    for rid in take:
        if done and run.stop():  # the rest waits for the next pass
            run.stopped, verdict = True, WAIT
            break
        if whys[rid]:  # its engine cannot be told now: this run waits, the others' engines are told
            run.hold(whys[rid])
            verdict = WAIT
            continue
        done += 1
        entry = tree_runs.get(rid)
        record = _decision(run.work.reach(entry) if getattr(entry, "engine_ref", None) else run.engine(), rid)
        if "state" not in record:
            run.hold("result_taken_unanswered")
            verdict, unsettled = WAIT, True
            if walk:
                break
            continue
        answers[rid] = {**record, "at": utc_now_iso(), **({"walk": True} if walk else {})}
    if answers:
        patch_document(run.root, task_id, lambda current: {**current, RECORDS: {
            **(current.get(RECORDS) if isinstance(current.get(RECORDS), dict) else {}), **answers}})
        recorded.update(answers)
        if not walk:  # the walk's answers reach the report as its one summary
            for rid, record in answers.items():
                run.report["observed"].append({"task_id": task_id, "run_id": rid, "taken": record["state"]})
    return verdict, recorded, unsettled


# ---- the one-time walk and its receipt -------------------------------------------------------------------------


def _over(counts: Dict[str, Any], waiting: int = 0, unknown: int = 0) -> Dict[str, Any]:
    """The job-row fields of a walk that is over: its counts, its leftovers and ``done_at``."""
    from ouroboros.utils import utc_now_iso

    return {WALK: {**counts, "waiting": waiting, "unknown": unknown, "done_at": utc_now_iso()}}


def empty_walk(root: Any) -> Dict[str, Any]:
    """The fields the step's first pass writes with its activation when custody holds no run: with nothing delegated,
    the walk is over before it begins (``{}`` otherwise, an unreadable probe included: the guarded walk answers)."""
    from ouroboros.delegate_custody_current import has_runs

    try:
        return {} if has_runs(root) else _over(_counted(()))
    except Exception:
        return {}


def walk_open(row: Dict[str, Any]) -> bool:
    """Whether the job ROW shows the step activated and the walk not over (no record with ``done_at``; a record never
    written is open): work the step does whenever the maintenance child runs, never by itself a reason to start one
    (``maintenance_retirement.due``)."""
    from ouroboros.review_thread_retention import JOB_FIELD

    state = row.get(JOB_FIELD) if isinstance(row.get(JOB_FIELD), dict) else {}
    record = row.get(WALK) if isinstance(row.get(WALK), dict) else {}
    return bool(state.get("activated_at")) and not record.get("done_at")


def walk(run: Any, visited: Iterable[str], row: Dict[str, Any], waiting: List[str]) -> Dict[str, Any]:
    """The walk (module docstring) after a runtime-retirement pass asked its own tasks (``visited``; ``row`` is the
    job row it read): every task it leaves to the ordinary duty is appended to ``waiting``, and the returned job-row
    fields (``{}``: none) are written in the same transaction as that set. Never raises: a failure is the pass's
    error, adds nothing to ``waiting`` and leaves the walk open."""
    record = dict(row[WALK]) if isinstance(row.get(WALK), dict) else {}
    if record.get("done_at"):
        return {}
    try:
        handed, fields, asked = _walk(run, set(visited), record)
    except Exception as exc:
        run.report["errors"].append(f"taken walk: {type(exc).__name__}: {exc}"[:300])
        log.warning("The taken walk failed; it stays open for the step's next pass", exc_info=True)
        return {}
    waiting.extend(handed)
    if asked or (fields.get(WALK) or {}).get("done_at"):
        run.report["observed"].append({WALK: {**record, **(fields.get(WALK) or {}), "tasks": asked}})
    return fields


def _records(root: Any, task_id: str) -> Any:
    """The task's decision records, ``None`` when its document cannot be read."""
    from ouroboros.task_store import read_document

    try:
        document = read_document(root, task_id) or {}
    except Exception:
        log.warning("The taken walk could not read the records of task %s", task_id, exc_info=True)
        return None
    return dict(document[RECORDS]) if isinstance(document.get(RECORDS), dict) else {}


def _walk(run: Any, visited: Set[str], record: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any], int]:
    """``(tasks left to the ordinary duty, job-row fields, tasks asked)`` for one pass of the walk."""
    from ouroboros.delegate_custody_current import has_runs, runs
    from ouroboros.review_thread_retention import WAIT

    handed: List[str] = []
    waiting, unknown = int(record.get("waiting") or 0), int(record.get("unknown") or 0)  # what earlier passes left

    def still_open(left: int, unread: int) -> Dict[str, Any]:  # the open record, written only when it changes
        return {WALK: {"waiting": waiting + left, "unknown": unknown + unread}} if left or unread else {}

    if not has_runs(run.root):  # nothing (left) delegated here: the walk is over
        return handed, _over(_counted(()), waiting, unknown), 0
    if run.stopped or run.stop():
        run.stopped = True
        return handed, {}, 0
    if run.census() is None or (_unavailable(run) and not _work_takes(run)):  # no proof, or nobody to tell: later
        return handed, {}, 0
    owing: Dict[str, List[str]] = {}
    for rid, entry in runs(run.root).items():
        if entry.task_id and owed(entry):
            owing.setdefault(entry.task_id, []).append(rid)
    owners = _settled(run.root, owing)
    seen: Dict[str, Dict[str, Any]] = {}
    left = unread = asked = 0  # the runs this pass leaves waiting, and unread; the tasks it asked
    for task_id in sorted(owners - visited):
        if run.stop():
            run.stopped = True
            return handed, still_open(left, unread), asked
        if run.census() is None:  # the census went stale: no tree can be proven over now, the rest waits
            return handed, still_open(left, unread), asked
        asked += 1
        try:
            verdict, records, unsettled = _tell(run, task_id, walk=True)
        except Exception as exc:  # one failing task never stops the walk: the ordinary duty asks it again
            run.report["errors"].append(f"{task_id}: {type(exc).__name__}: {exc}"[:300])
            log.warning("The taken walk could not ask task %s", task_id, exc_info=True)
            verdict, records, unsettled = WAIT, _records(run.root, task_id), False
        if records is None:  # its records cannot be read: its runs are unknown, never waiting or zero
            handed.append(task_id)
            unread += len(owing[task_id])
        else:
            seen[task_id] = records
            if verdict == WAIT:
                handed.append(task_id)
                left += sum(1 for rid in owing[task_id] if rid not in records)
        if run.stopped or unsettled:  # a stop, or an answer that was not a settled record: the rest waits
            return handed, still_open(left, unread), asked
    for task_id in owners & visited:  # this pass's own tasks: a record the walk wrote may be theirs now
        records = _records(run.root, task_id)
        if records is None:  # unreadable: unknown, and left to the ordinary duty
            handed.append(task_id)
            unread += len(owing[task_id])
        else:
            seen[task_id] = records
    counts = _counted(answer for records in seen.values() for answer in records.values()
                      if isinstance(answer, dict) and answer.get("walk") is True)
    return handed, _over(counts, waiting + left, unknown + unread), asked


def _settled(root: Any, owing: Dict[str, List[str]]) -> Set[str]:
    """The owing tasks the walk asks: a settled row, or an ``interrupted`` one whose retry's effective result is
    settled (the predecessor ``retry_line`` names at that retry's terminal write). An effective result that cannot
    be read is the guarded duty's to answer."""
    from ouroboros.task_results import STATUS_INTERRUPTED
    from ouroboros.task_status import SETTLED_STATUSES, load_effective_task_result
    from ouroboros.task_store import task_versions

    def retried(task_id: str) -> bool:
        try:
            effective = load_effective_task_result(root, task_id, materialize_artifacts=False) or {}
        except Exception:
            return True
        return str(effective.get("status") or "") in SETTLED_STATUSES

    return {task_id for task_id, (_seq, status, _owes) in task_versions(root, owing).items()
            if status in SETTLED_STATUSES or (status == STATUS_INTERRUPTED and retried(task_id))}


def _counted(answers: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """``taken``, ``absent`` and ``refused`` by the engine's code over ANSWERS, plain integers."""
    out: Dict[str, Any] = {"taken": 0, "absent": 0, "refused": {}}
    for answer in answers:
        if answer.get("state") == "refused":
            code = str(answer.get("code") or answer.get("status") or "unknown")
            out["refused"][code] = out["refused"].get(code, 0) + 1
        elif answer.get("state") in ("taken", "absent"):
            out[answer["state"]] += 1
    return out


def _refusal(code: str) -> str:
    """One engine refusal code in the owner's words: a known code by what it means, a bare status (a refusal
    without a code) as an invalid request, any other code as it is."""
    return _REFUSALS.get(code) or ("as an invalid request" if code.isdigit() else f"with code {code}")


def receipt(record: Dict[str, Any]) -> str:
    """The owner's one line about a finished walk, counting runs throughout; ``""`` when it met nothing to tell."""
    def number(value: Any) -> int:
        return value if type(value) is int and value > 0 else 0

    def plural(count: int) -> str:
        return f"{count:,} run" + ("" if count == 1 else "s")

    def left(count: int) -> str:
        return "was" if count == 1 else "were"

    refused = {str(code): number(n) for code, n in sorted((record.get("refused") or {}).items()) if number(n)}
    taken, absent, waiting, unknown = (number(record.get(key)) for key in ("taken", "absent", "waiting", "unknown"))
    clauses = [f"the engine now knows Ouroboros has taken {plural(taken)}, so their engine copies follow its ordinary "
               "retention window while Ouroboros keeps its own copy of each"] if taken else []
    if refused:
        detail = ", ".join(f"{n:,} {_refusal(code)}" for code, n in refused.items())
        clauses.append(f"the engine refused {plural(sum(refused.values()))} ({detail})")
    if absent:
        clauses.append(f"the engine no longer had {plural(absent)}")
    if waiting:
        clauses.append(f"{plural(waiting)} {left(waiting)} left to the ordinary pass")
    if unknown:
        clauses.append(f"{plural(unknown)} could not be read and {left(unknown)} left to the ordinary pass as well")
    if not clauses:
        return ""
    return "One-time catch-up with the engine on earlier delegated results: " + "; ".join(clauses) + "."


def deliver_receipt(root: Any) -> None:
    """After a maintenance child run (``MaintenanceChild._run``, never after Stop or Panic): the finished walk's
    receipt in the owner's chat, once, never raising. Its type is a one-time notice
    (``notice_receipts.UPGRADE_TYPES``) recorded when its row lands; nothing is sent or marked while no owner chat
    is bound or the dialogue cannot answer, so a later child run delivers it."""
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
