"""The one historical walk: old finished reviewer threads archived and released, then host-minted registrations retired.

Owner choice (razzant/ouroboros#1180 with #2821, 08.11.2026): no reviewer-specific cleaner and no schedule. One
visible pass per installation does what the ordinary ``runtime_retirement`` step (``review_thread_retention``) never
does for material older than its activation, and then retires the engine project registrations Ouroboros minted for
its own one-shot folders before minted roots declared ``scope.ephemeral`` (``delegate_registration_policy``).

Threads first. Every settled task holding a review-owned custody run (``delegate_result_taken._settled``: a settled
row, or an ``interrupted`` one whose retry's effective result is) is asked, in task-id order from the record's
``cursor``, by the step's own per-task duty with ``historical`` set (``_Pass.task``): the same tree, custody,
continuation and Keep proofs, the same complete verified archive in the task's ``source_handles/delegated_activity``
before any bound release, the same held records. Only the activation cutoff does not apply. A release this walk made
says so on the release record itself (``historical`` in ``review_thread_releases``), bound before any request, so
counting needs no second store and survives a crash: each ask reads the task's own records, and an attributed settled
operation the walk's record has not counted is counted once; a task still waiting keeps the operations already counted
in ``counted``. A task it leaves waiting stays in the record's ``waiting`` and is asked the same way whenever the step
runs, until its duty is done; a hold any task's last ask named stays in ``task_holds``. Releasing a thread loses the
engine's native reviewer continuity for it (a later Continue of that seat starts a fresh session); the archive stays
readable offline.

Registrations second, once the thread cursor is exhausted, read through the engine's strict project list
(``project_registrations``: a malformed list is unreadable, never "none"). A project is a candidate only when custody
names it (``custody_runs.project_id``). It goes only when every naming run is settled and bound to this very engine
store (``daemon_store_id``; an unbound or other store's id proves nothing), any scope root the runs' recorded requests
name is the listed root, the root is proven host-minted (``minted_shape``, or a private copy whose record reads
``registered``), custody no longer owns it (that is ``retire_settled_registrations``'s), none is persistent, and every
task is proven over, not kept and its line not continuable. A project one of whose tasks still waits in this walk is
``deferred``: it is asked once that task's duty is done (bounded by the walk's own waiting tasks, never new projects).
Under the project's retirement lock the rows are read again and every proof repeated; then each task's Keep gate
(``owner_keep.gate``, the task custody lock, no wait) is held across the intent record and the engine's own
``DELETE /v2/projects/:id``. Only a receipt naming this project with ``registryRemoved`` counts as removed (its
partition archived, artifacts kept; saved to ``logs/events.jsonl`` through ``observed``); a 404 is absent; a typed 409
(``project_has_threads``, ``project_has_active_run``) is the engine keeping it. Any other answer leaves the durable
``pending`` intent: the next pass reads the same store's list, without resending. Gone from the list, the outcome is
``gone_after_unknown`` (no claim that this request removed it); still listed, the request remains unconfirmed and
counts as ``engine_unanswered``. A different root or store is held with its own reason.
Nothing here touches the engine's home, trashes or purges a thread, or deletes a file. A pass that removed a project
reads the list once more and counts each removed project still listed (``listed_after_removal``).

Then ``done_at`` and, after a maintenance child run that Stop or Panic did not end, one Main message
(``deliver_receipt``, a one-time notice whose landed row is its receipt). A project still deferred then is reported
as waiting for its sessions' release, never as cleaned; the walk keeps asking its waiting tasks, then those projects,
until none is left. Its facts are the ``runtime_retirement`` job row's ``historical_walk``; it runs only while the
step runs (``owed``), never by itself a reason to start a child, and never before the step's first activated pass.
"""
from __future__ import annotations

import contextlib
import logging
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

WALK = "historical_walk"  # the runtime_retirement job row's field
NOTICE = "historical_retirement_notice"  # the walk's one-time receipt in the owner's chat (``notice_receipts``)
_COUNTED = ("retained", "accepted", "refused", "abandoned", "unknown")
_BUSY_LIMIT = 2  # attempts to acquire local gates before reporting busy; no removal was sent


def _open_work(record: Dict[str, Any]) -> bool:
    """Work still owed: admitted tasks, deferred projects, an unresolved intent or the removal's readback."""
    return bool(record.get("waiting") or record.get("deferred") or record.get("pending") or record.get("removed_ids"))


def owed(row: Dict[str, Any]) -> bool:
    """Whether the job ROW shows the step activated and this walk open or still owing work it admitted."""
    from ouroboros.review_thread_retention import JOB_FIELD

    state = row.get(JOB_FIELD) if isinstance(row.get(JOB_FIELD), dict) else {}
    record = row.get(WALK) if isinstance(row.get(WALK), dict) else {}
    return bool(state.get("activated_at")) and (not record.get("done_at") or _open_work(record))


def empty_walk(root: Any) -> Dict[str, Any]:
    """The field the step's first pass writes with its activation when custody holds no run: nothing is older than
    this code, so the walk is over before it begins (``{}`` otherwise, an unreadable probe included)."""
    from ouroboros.delegate_custody_current import has_runs
    from ouroboros.utils import utc_now_iso

    try:
        return {} if has_runs(root) else {WALK: {**_copy({}), "done_at": utc_now_iso()}}
    except Exception:
        return {}


def walk(run: Any, row: Dict[str, Any]) -> Dict[str, Any]:
    """One pass of the walk after the step's own tasks (``run`` is ``review_thread_retention._Pass``, ``row`` the job
    row it read): the job-row fields to write in the step's transaction, ``{}`` when nothing changed. Never raises."""
    from ouroboros.review_thread_retention import JOB_FIELD

    state = row.get(JOB_FIELD) if isinstance(row.get(JOB_FIELD), dict) else {}
    prior = row.get(WALK) if isinstance(row.get(WALK), dict) else {}
    if not state.get("activated_at") or (prior.get("done_at") and not _open_work(prior)):
        return {}
    record = _copy(prior)
    try:
        _waiting(run, record)
        if not record.get("done_at") and not record.get("threads_done_at"):
            _threads(run, record)
        if record.get("threads_done_at") and not run.stopped:
            _registrations(run, record)
    except Exception as exc:  # what it already did stays recorded; the rest waits for the step's next pass
        run.report["errors"].append(f"historical walk: {type(exc).__name__}: {exc}"[:300])
        log.warning("The historical walk failed; it stays open for the step's next pass", exc_info=True)
    if record == prior:
        return {}
    if record.get("done_at") and not prior.get("done_at") and receipt(record):
        run.report["observed"].append({WALK: receipt(record)})
    return {WALK: record}


def _copy(record: Dict[str, Any]) -> Dict[str, Any]:
    import copy

    out = copy.deepcopy(record)
    out.setdefault("threads", {key: 0 for key in _COUNTED})
    out.setdefault("waiting", [])
    out.setdefault("task_holds", {})
    return out


# ---- threads -------------------------------------------------------------------------------------------------


def _walk_states(root: Any, task_id: str) -> Dict[str, str]:
    """``{operation id: state}`` of the releases this walk made, as the task's own records say: ``historical`` on the
    record, written before any request. Read before the ask: what an ordinary pass settled counts, and a settled
    record the ask's fresh bind then replaces is still counted."""
    from ouroboros.review_thread_retention import RELEASES
    from ouroboros.task_store import read_document

    releases = (read_document(root, task_id) or {}).get(RELEASES)
    return {str(item["operation_id"]): str(item.get("state") or "")
            for item in (releases.values() if isinstance(releases, dict) else ())
            if isinstance(item, dict) and item.get("operation_id") and item.get("historical")}


def _ask(run: Any, record: Dict[str, Any], task_id: str) -> str:
    """The step's per-task thread duty with the cutoff lifted; its counts and holds land on ``record``. A settled
    outcome is counted from the task's own release records: an operation this walk made (``historical`` on the record,
    written before any request) that no persisted pass counted, whoever settled it and whenever: a pass whose job
    row was lost re-asks its tasks, and a task still waiting keeps its counted operations (``counted``). A bound
    operation is no acceptance, and an ordinary pass's release is not this walk's."""
    from ouroboros.review_thread_retention import SETTLED, WAIT

    before = {key: int(run.report.get(key) or 0) for key in _COUNTED}
    holds = dict(run.report["holds"])
    credited = set((record.get("counted") or {}).get(task_id) or ())
    states: Dict[str, str] = {}
    run.walk_states = {}  # what this ask settles; never another task's or an earlier ask's
    try:
        states = _walk_states(run.root, task_id)
        verdict = run.task(task_id, historical=True)
    except Exception as exc:  # one unreadable task never stops the walk: it waits
        run.report["errors"].append(f"{task_id}: {type(exc).__name__}: {exc}"[:300])
        log.warning("The historical walk could not ask task %s", task_id, exc_info=True)
        verdict = WAIT
    counts = record["threads"]
    for key in ("retained", "unknown"):
        counts[key] = int(counts.get(key) or 0) + int(run.report.get(key) or 0) - before[key]
    for operation, state in {**states, **run.walk_states}.items():
        if state in SETTLED and operation not in credited:
            counts[state] = int(counts.get(state) or 0) + 1
            credited.add(operation)
    named = sorted(reason for reason, count in run.report["holds"].items() if count > holds.get(reason, 0))
    run.report["holds"] = holds  # its holds stand on the walk's record: the step's report counts each task once
    if named:
        record["task_holds"][task_id] = named
    else:
        record["task_holds"].pop(task_id, None)
    waiting = [task for task in record["waiting"] if task != task_id]
    record["waiting"] = sorted(waiting + [task_id]) if verdict == WAIT else waiting
    counted = dict(record.get("counted") or {})  # only a task still waiting can be asked again
    if verdict == WAIT and credited:
        counted[task_id] = sorted(credited)
    else:
        counted.pop(task_id, None)
    if counted:
        record["counted"] = counted
    else:
        record.pop("counted", None)
    return verdict


def _ready(run: Any) -> bool:
    """Whether a tree can be proven over now and the engine can be asked about threads at all."""
    if run.stopped or run.stop():
        run.stopped = True
        return False
    return run.census() is not None and not run.usable()


def _waiting(run: Any, record: Dict[str, Any]) -> None:
    for task_id in list(record["waiting"]):
        if not _ready(run):
            return
        _ask(run, record, task_id)


def _threads(run: Any, record: Dict[str, Any]) -> None:
    from ouroboros.delegate_custody_current import runs
    from ouroboros.delegate_result_taken import _settled
    from ouroboros.utils import utc_now_iso

    owners: Dict[str, List[str]] = {}
    for rid, entry in runs(run.root).items():
        if entry.task_id and entry.review_owned:
            owners.setdefault(entry.task_id, []).append(rid)
    cursor = str(record.get("cursor") or "")
    for task_id in sorted(task for task in (_settled(run.root, owners) if owners else ()) if task > cursor):
        if not _ready(run):
            return
        record["threads"]["asked"] = int(record["threads"].get("asked") or 0) + 1
        _ask(run, record, task_id)
        record["cursor"] = task_id
        if run.stopped:
            return
    record["threads_done_at"] = utc_now_iso()


# ---- registrations -------------------------------------------------------------------------------------------


def _minted(root: str, data_root: Any) -> bool:
    """Proof that Ouroboros minted the folder: its data-root shape, or its own private-copy record, read."""
    from ouroboros.delegate_registration_policy import minted_shape
    from ouroboros.private_copy_read import checkout_registration

    try:
        return minted_shape(root, data_root) or checkout_registration(root, data_root) == "registered"
    except (OSError, ValueError):
        return False


def _same_path(left: str, right: str) -> bool:
    import os

    return os.path.realpath(left) == os.path.realpath(right)


def _recorded_roots(data_root: Any, rows: List[Any]) -> Optional[set]:
    """The project scope roots the runs' recorded request bodies name (a thread turn names none: its thread owns
    the scope); ``None`` when an invocation record cannot be read."""
    from ouroboros.delegate_custody import invocation_record

    roots = set()
    for row in rows:
        if not row.invocation_id:
            continue
        try:
            found = invocation_record(data_root, row.invocation_id)
        except Exception:
            return None
        request = found.get("request") if isinstance(found, dict) and found.get("state") != "legacy_gap" else None
        if not isinstance(request, dict):
            return None  # no record, a legacy gap or an unread body: no evidence of the folder it registered
        scope = request.get("scope")
        if isinstance(scope, dict) and scope.get("kind") == "project" and isinstance(scope.get("root"), str):
            roots.add(scope["root"])
    return roots


def _tasks(rows: List[Any]) -> List[str]:
    return sorted({task for row in rows for task in (row.task_id, row.root_task_id) if task})


def _hold(run: Any, project: Dict[str, Any], rows: List[Any], store_id: str) -> str:
    """Why custody, its owning store or its tasks keep this registration (``""``: nothing does)."""
    from ouroboros.delegate_continuation import still_continuable
    from ouroboros.owner_keep import is_kept
    from ouroboros.subagent_worktrees import _task_proven_over

    root = str(project.get("root") or "")
    if not rows:
        return "custody_changed"
    if any(row.project_owned for row in rows):
        return "custody_owned"  # the ordinary sweep's (``retire_settled_registrations``)
    if any(row.project_persistent for row in rows):
        return "persistent"
    if not store_id or any(str(row.daemon_store_id or "") != store_id for row in rows):
        return "store_unproven"  # an id another store (or an unbound run) named is no proof about this one
    if not _minted(root, run.root):
        return "ownership_unproven"
    recorded = _recorded_roots(run.root, rows)
    if recorded is None:
        return "custody_unreadable"
    if any(not _same_path(named, root) for named in recorded):
        return "root_mismatch"
    if any(not row.settled for row in rows):
        return "run_unsettled"
    tasks = _tasks(rows)
    if not tasks:
        return "ownership_unproven"
    if any(is_kept(run.root, task) for task in tasks):
        return "owner_keep"
    if any(not _task_proven_over(task, run.root, run.live) for task in tasks):
        return "task_not_proven_over"
    census = run.census()
    if census is None:
        return "census_unknown"
    if any(still_continuable(run.root, row, census) for row in rows):
        return "continuable"
    return ""


def _intend(run: Any, record: Dict[str, Any], pending: Dict[str, Any]) -> None:
    """Write the intent before the request leaves; after a lost answer only readback is allowed. The record holds
    it only once written: an intent that failed to land names no request (none is sent), never a later unknown."""
    from ouroboros import maintenance_jobs
    from ouroboros.maintenance_retirement import JOB

    maintenance_jobs.record(run.root, JOB, **{WALK: {**record, "pending": pending}})
    record["pending"] = pending


def _delete(engine: Any, project_id: str) -> Tuple[str, Any]:
    """The engine's answer to one removal: a verified receipt, its typed refusal, or ``unknown``."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    try:
        answer = engine.remove_project(project_id)
    except ClaudexorUnavailable as exc:
        status, code = int(getattr(exc, "status_code", 0) or 0), str(getattr(exc, "code", "") or "")
        if status == 404 and code == "project_not_found":
            return "absent", code
        if status == 409 and code:
            return "kept", code  # a definite refusal: the registry is unchanged
        log.info("Historical removal of project %s unanswered: %s", project_id, exc)
        return "unknown", f"{code or 'unanswered'}:{status}"
    if isinstance(answer, dict) and answer.get("projectId") == project_id and answer.get("registryRemoved") is True:
        return "removed", answer
    return "unknown", "receipt_unverified"


def _retire(run: Any, record: Dict[str, Any], engine: Any, store_id: str, project: Dict[str, Any],
            rows: List[Any]) -> Tuple[str, Any]:
    """One candidate's outcome. Every proof is made again under the project's retirement lock, and each task's Keep
    gate is held across the intent and the engine's removal (no wait: a busy gate sends nothing)."""
    from ouroboros.delegate_custody_current import runs
    from ouroboros.delegate_custody_usage import project_retirement_lock
    from ouroboros.owner_keep import OWNER_KEEP, gate

    project_id = str(project["id"])
    linked = sorted(set(_tasks(rows)) & set(record["waiting"]))
    if linked:
        return "deferred", linked  # this walk's own pending release is why it stays: asked once that settles
    why = _hold(run, project, rows, store_id)
    if why:
        return "held", why
    try:
        with project_retirement_lock(run.root, project_id):
            now = list(runs(run.root, "project_id=?", (project_id,)).values())
            why = ("custody_changed" if {row.run_id for row in now} != {row.run_id for row in rows}
                   else _hold(run, project, now, store_id))
            if why:
                return "held", why
            with contextlib.ExitStack() as gates:
                for task in _tasks(now):
                    why = gates.enter_context(gate(run.root, task, timeout_sec=0))
                    if why:
                        return ("held", why) if why == OWNER_KEEP else ("busy", why)
                _intend(run, record, {"project_id": project_id, "root": str(project["root"]), "store_id": store_id})
                return _delete(engine, project_id)
    except TimeoutError:
        return "busy", "project_lock"


def _apply(run: Any, record: Dict[str, Any], project: Dict[str, Any], outcome: Tuple[str, Any], *,
           main: bool) -> bool:
    """Record one outcome; False ends this pass at the project (its answer is owed to the next one)."""
    project_id = str(project["id"])
    kind, detail = outcome
    retry = record.get("retry") if isinstance(record.get("retry"), dict) else {}
    if kind == "unknown":
        return False  # the intent stays for readback, never for another removal request
    if kind == "busy":
        attempts = (int(retry.get("attempts") or 0) if retry.get("project_id") == project_id else 0) + 1
        record["retry"] = {"project_id": project_id, "attempts": attempts}
        if attempts < _BUSY_LIMIT:
            return False  # nothing was sent
        kind, detail = "held", "busy"
    if retry.get("project_id") == project_id:
        record.pop("retry", None)
    intent = record.get("pending", {})
    if intent.get("project_id") == project_id:
        record.pop("pending", None)
    else:
        intent = {}
    deferred = record.setdefault("deferred", {})
    if kind == "deferred":
        deferred[project_id] = detail
    else:
        deferred.pop(project_id, None)
        counts = record["registrations"]
        counts["candidates"] = int(counts.get("candidates") or 0) + 1
        if kind in ("removed", "absent", "gone_after_unknown"):
            counts[kind] = int(counts.get(kind) or 0) + 1
            if kind == "removed":
                record.setdefault("removed_ids", []).append(project_id)
        else:
            bucket = counts.setdefault("kept" if kind == "kept" else "held", {})
            bucket[detail] = int(bucket.get(detail) or 0) + 1
        run.report["observed"].append({"historical_registration": {
            "project_id": project_id, "root": str(project.get("root") or ""), "outcome": kind,
            **({"intent": intent} if intent else {}),
            **({"receipt": detail} if kind == "removed" else {"code": detail} if detail else {})}})
    if main:
        record["project_cursor"] = project_id
    return True


def _reconcile(run: Any, record: Dict[str, Any], listing: Dict[str, Dict[str, Any]], store_id: str) -> bool:
    """Read back an unanswered request in the same store; a listed project never proves non-execution."""
    pending = record.get("pending") if isinstance(record.get("pending"), dict) else None
    if not pending:
        return True
    project_id = str(pending.get("project_id") or "")
    main = project_id not in (record.get("deferred") or {})
    listed = listing.get(project_id)
    project = listed or {"id": project_id, "root": str(pending.get("root") or "")}
    if not store_id or str(pending.get("store_id") or "") != store_id:
        return _apply(run, record, project, ("held", "store_changed"), main=main)
    if listed is None:
        return _apply(run, record, project, ("gone_after_unknown", ""), main=main)
    if listed["root"] != pending.get("root"):
        return _apply(run, record, project, ("held", "root_changed"), main=main)
    # The read may precede the first request's late effect; it cannot authorize a duplicate DELETE.
    return _apply(run, record, project, ("held", "engine_unanswered"), main=main)


def _registrations(run: Any, record: Dict[str, Any]) -> None:
    from ouroboros.delegate_custody_current import runs
    from ouroboros.utils import utc_now_iso

    named: Dict[str, List[Any]] = {}
    if record.get("done_at") and not (record.get("pending") or record.get("removed_ids")) and all(
            set(tasks) & set(record["waiting"]) for tasks in (record.get("deferred") or {}).values()):
        return  # every deferred project still waits for its sessions: nothing to read here
    for entry in runs(run.root, "project_id<>?", ("",)).values():
        named.setdefault(entry.project_id, []).append(entry)
    if not (named or record.get("pending") or record.get("deferred") or record.get("removed_ids")):
        record.setdefault("done_at", utc_now_iso())
        return
    if run.stopped or run.stop():
        run.stopped = True
        return
    if run.census() is None:
        return  # no task can be proven over now
    engine = run.engine()
    reader = getattr(engine, "project_registrations", None)
    if engine is None or reader is None:
        run.hold("historical_engine_unavailable")
        return
    store_id = str(getattr(engine, "store_identity", "") or "")
    try:
        listing = {str(row["id"]): row for row in reader()}
    except Exception:  # unreadable or malformed: it proves no project's absence
        run.hold("historical_projects_unread")
        log.info("The historical walk could not read the engine's projects", exc_info=True)
        return
    record.setdefault("registrations", {"candidates": 0, "removed": 0, "absent": 0, "kept": {}, "held": {}})
    if not _reconcile(run, record, listing, store_id):
        return
    for project_id, tasks in sorted((record.get("deferred") or {}).items()):
        if set(tasks) & set(record["waiting"]):
            continue  # its sessions' release is still owed
        if run.stop():
            run.stopped = True
            return
        project = listing.get(project_id)
        outcome = ("absent", "not_listed") if project is None else _retire(
            run, record, engine, store_id, project, named.get(project_id, []))
        if not _apply(run, record, project or {"id": project_id}, outcome, main=False):
            return
    if not record.get("done_at"):
        cursor = str(record.get("project_cursor") or "")
        for project_id in sorted(pid for pid in listing if pid > cursor and pid in named):
            if run.stop():
                run.stopped = True
                return
            if not _apply(run, record, listing[project_id],
                          _retire(run, record, engine, store_id, listing[project_id], named[project_id]), main=True):
                return
        record["done_at"] = utc_now_iso()
    _readback(run, record, reader)


def _readback(run: Any, record: Dict[str, Any], reader: Any) -> None:
    """Read the list once more after this pass removed something: a removal still listed is counted, never hidden."""
    removed = record.get("removed_ids") or []
    if not removed:
        return
    try:
        listed = {str(row["id"]) for row in reader()}
    except Exception:
        run.hold("historical_projects_unread")
        return  # the next pass reads it back
    counts = record["registrations"]
    counts["listed_after_removal"] = int(counts.get("listed_after_removal") or 0) + sum(
        1 for project_id in removed if project_id in listed)
    record.pop("removed_ids", None)


# ---- the receipt ---------------------------------------------------------------------------------------------


def _number(value: Any) -> int:
    return value if type(value) is int and value > 0 else 0


def _plural(count: int, word: str) -> str:
    return f"{count:,} {word}" + ("" if count == 1 else "s")


_REASONS = {"ownership_unproven": "Ouroboros could not prove it registered them",
            "store_unproven": "their runs are not bound to this engine store",
            "root_mismatch": "their recorded folder differs from the engine's",
            "root_changed": "their folder changed after a lost answer", "store_changed": "the engine store changed",
            "custody_unreadable": "their run records could not be read",
            "persistent": "they are your own projects", "owner_keep": "you keep their tasks",
            "task_not_proven_over": "their tasks are not proven over", "run_unsettled": "a run of theirs is not settled",
            "continuable": "their work can still be continued", "custody_changed": "a run started meanwhile",
            "census_unknown": "the task queue could not be read", "engine_unanswered": "their removal is unconfirmed",
            "busy": "their tasks stayed busy", "custody_owned": "the ordinary cleanup still owns them",
            "project_has_threads": "they still hold sessions", "project_has_active_run": "a run is active there"}


def _why(buckets: Dict[str, Any]) -> str:
    return ", ".join(f"{n:,} because {_REASONS.get(code, f'of {code}')}" for code, n in sorted(buckets.items()) if n)


def receipt(record: Dict[str, Any]) -> str:
    """The owner's one message about the finished walk; ``""`` when it met nothing to tell."""
    threads = record.get("threads") if isinstance(record.get("threads"), dict) else {}
    regs = record.get("registrations") if isinstance(record.get("registrations"), dict) else {}
    accepted, waiting = _number(threads.get("accepted")), len(record.get("waiting") or [])
    held = sum(1 for task in record.get("task_holds") or {} if task not in (record.get("waiting") or []))
    removed, absent = _number(regs.get("removed")), _number(regs.get("absent"))
    lost, deferred = _number(regs.get("gone_after_unknown")), len(record.get("deferred") or {})
    kept = {code: _number(n) for code, n in (regs.get("kept") or {}).items()}
    stays = {code: _number(n) for code, n in (regs.get("held") or {}).items()}
    clauses = []
    if accepted:
        clauses.append(f"{_plural(accepted, 'finished reviewer session')} {'was' if accepted == 1 else 'were'} archived "
                       "in their tasks, verified and released by the engine; the archive stays readable, but those "
                       "reviewers' native session memory cannot be resumed")
    if waiting:
        clauses.append(f"{_plural(waiting, 'task')} with sessions {'waits' if waiting == 1 else 'wait'} for work that is "
                       f"not over and {'is' if waiting == 1 else 'are'} released when it ends")
    if held:
        clauses.append(f"sessions of {_plural(held, 'task')} stay as they are (kept, incomplete or unprovable; "
                       "their reasons are on each task)")
    if removed:
        clauses.append(f"{_plural(removed, 'engine project')} Ouroboros had registered for its own temporary folders "
                       f"{'was' if removed == 1 else 'were'} removed through the engine, which kept their artifacts")
    if absent:
        clauses.append(f"{_plural(absent, 'such project')} {'was' if absent == 1 else 'were'} already gone")
    if lost:
        clauses.append(f"{_plural(lost, 'project')} disappeared after a request whose answer was lost, so whether "
                       "that request removed it is unknown")
    if deferred:
        clauses.append(f"{_plural(deferred, 'project')} {'waits' if deferred == 1 else 'wait'} until its sessions' "
                       "release settles and is not cleaned up yet")
    if sum(kept.values()):
        clauses.append(f"the engine kept {_plural(sum(kept.values()), 'project')} ({_why(kept)})")
    if sum(stays.values()):
        clauses.append(f"{_plural(sum(stays.values()), 'project')} stay registered ({_why(stays)})")
    if _number(regs.get("listed_after_removal")):
        clauses.append(f"{_plural(regs['listed_after_removal'], 'removed project')} still appeared in the engine's list")
    if not clauses:
        return ""
    return "One-time cleanup of earlier reviewer sessions and engine projects: " + "; ".join(clauses) + "."


def deliver_receipt(root: Any) -> None:
    """After a maintenance child run (``MaintenanceChild._run``, never after Stop or Panic): the finished walk's
    receipt in the owner's chat, once, never raising (``delegate_result_taken.deliver_receipt``'s rule; each send
    site names its own notice type): nothing is sent while no owner chat is bound or the dialogue cannot answer,
    and a walk that met nothing says nothing."""
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
        log.debug("The historical walk's receipt waits for a later maintenance child run", exc_info=True)
