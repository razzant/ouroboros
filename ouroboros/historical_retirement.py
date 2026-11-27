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

The strict project list discovers candidates. Under the project lock, one snapshot supplies runs, pending
requests and retirement duty. Minted folder and registration creation are separate proofs: an adopter's binding
is not ownership. Only original, unreleased ownership in the same store can authorize release, and only at the exact
root and store its creator's receipt names (never the list's current name or a local respelling; no receipt keeps
the registration). A waiting task defers it.
With the tasks’ Keep gates held, an addressed read confirms the binding before the walk records intent (``shared``:
the conditional bound release, with its project, store and root) and requests bound release. Only a matching
receipt can prove release or absence. An unanswered request stays the walk's open intent (its cursor and
``done_at`` wait) until a bound read finds it absent, its binding dead (a confirmed 409), or a repeat of the same
release, after the same checks and from the saved address, is answered (``remove_registration``). A served base
that is not the one that took the request, or an operation the engine does not offer now, answers nothing: the
intent waits for their return. An older walk's intent has no ``shared``: its id-only DELETE is only read (absence
or a dead binding settle it as above), never sent again, and a registration still there ends as held. The final
list read counts removed ids still listed. All changes concern engine registrations and preserve artifacts.

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
# This walk's own release, sent and not yet answered: ``historical`` is its bound release, repeated only from the
# saved ``shared`` intent (never on the mark alone), ``historical_walk`` an older id-only DELETE, only read.
_WALK_LANES = ("historical", "historical_walk")


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


def _recorded_roots(data_root: Any, rows: List[Any], store_id: str) -> Optional[Tuple[set, set]]:
    """Return separately the folders named by uses (compared as recorded keys) and the roots their creators' receipts
    name exactly as the engine recorded them; unreadable evidence is not empty."""
    from ouroboros.delegate_invocation import read as read_invocation
    from ouroboros.delegate_pending import request_body
    from ouroboros.delegate_registration_policy import recorded_workspace_key
    from ouroboros.execution_domain import registration_engine

    roots, created_roots = set(), set()
    for row in rows:
        original = row.request_event
        if not isinstance(original, dict) or not isinstance(row.project_resolution, dict):
            return None
        origin = row.project_resolution or original.get("project_resolution") or {}
        try:
            # STARTED carries a copy of the first request. Older rows can use its addressed invocation instead.
            if original and (original.get("invocation_id") != row.invocation_id or original.get("project_id") != row.project_id):
                return None
            if not original and row.invocation_id and not (origin.get("registration_root")
                    and origin.get("registration_store_id") and (isinstance(origin.get("registration_created"), bool)
                    or origin.get("creator_invocation") == row.invocation_id)):
                original = read_invocation(data_root, row.invocation_id, lambda _root, event: event)
                if (not original or original.get("state") != "started" or original.get("run_id") != row.run_id
                        or original.get("project_id") != row.project_id):
                    return None
                origin = origin or original.get("project_resolution") or {}
            if origin and registration_engine(origin) != registration_engine(row):
                return None
            if origin.get("registration_store_id") and origin["registration_store_id"] != store_id:
                return None
            folder = origin.get("registration_root") if origin.get("registration_store_id") else None
            bound = isinstance(folder, str) and bool(folder)
            if not bound and original:
                body = request_body(data_root, original)
                if not isinstance(body, dict):
                    return None
                scope = body.get("scope")
                if isinstance(scope, dict) and scope.get("kind") == "project" and not scope.get("ephemeral"):
                    folder = scope.get("root")
        except Exception:
            return None
        if not isinstance(folder, str) or not folder:
            continue
        roots.add(recorded_workspace_key(folder))  # a recorded root may spell the same folder in another case
        if row.registration_outcome not in ("", "kept"):
            continue  # an earlier release (or confirmed absence) ended the old authority
        created = origin.get("registration_created")
        creator = created is None and row.invocation_id and origin.get("creator_invocation") == row.invocation_id
        legacy_owner = created is None and original and original.get("project_owned") is True
        # Only the registration's own receipt binds its creator to a store and a root. A handshake or run receipt,
        # the engine's current list or a local respelling cannot.
        if bound and (created is True or creator or legacy_owner):
            created_roots.add(folder)
    return roots, created_roots


def _creator_root(recorded: Optional[Tuple[set, set]]) -> str:
    """The one root the creators' receipts name, exactly: the address of the walk's bound read and release ("" for
    none or more than one, which keeps the registration)."""
    return next(iter(recorded[1])) if recorded is not None and len(recorded[1]) == 1 else ""


def _tasks(rows: List[Any]) -> List[str]:
    return sorted({task for row in rows for task in (row.task_id, row.root_task_id) if task})


def _hold(run: Any, project: Dict[str, Any], address: str, recorded: Optional[Tuple[set, set]], rows: List[Any],
          engine: Any, store_id: str, tasks: List[str], *, pending: bool, owed: Any) -> str:
    """Why custody, its owning store or its tasks keep this registration (``""``: nothing does). ``address`` is the
    creator's receipt root (``_creator_root``); the list's spelling of the candidate can only keep it."""
    from ouroboros.delegate_continuation import still_continuable
    from ouroboros.delegate_registration_policy import recorded_workspace_key
    from ouroboros.execution_domain import same_engine
    from ouroboros.owner_keep import is_kept
    from ouroboros.subagent_worktrees import _task_proven_over

    if any(row.project_owned for row in rows):
        return "custody_owned"  # the ordinary sweep's (``retire_settled_registrations``)
    if any(row.project_persistent for row in rows):
        return "persistent"
    if not store_id or any(str(row.daemon_store_id or "") != store_id or not same_engine(engine, row) for row in rows):
        return "store_unproven"  # an id another store (or an unbound run) named is no proof about this one
    if recorded is None:
        return "custody_unreadable"
    # The uses, the receipt and the list name one folder, compared as recorded keys (a case-variant spelling is the
    # same folder); another root is another registration, kept unasked.
    named = recorded[0] | ({recorded_workspace_key(address)} if address else set())
    if named and named != {recorded_workspace_key(str(project.get("root") or ""))}:
        return "root_mismatch"
    if not address or not _minted(address, run.root):
        return "ownership_unproven"  # no receipt (or two) names the root this host created: kept
    if any(not row.settled for row in rows):
        return "run_unsettled"
    if owed:
        return "custody_owned"
    if pending:
        return "run_unsettled"
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
    """Write the intent before the request leaves; after a lost answer a ``shared`` one repeats that same release
    (``_repeats``), any other is only read back. The record holds it only once written: an intent that failed to
    land names no request (none is sent), never a later unknown."""
    from ouroboros import maintenance_jobs
    from ouroboros.maintenance_retirement import JOB

    maintenance_jobs.record(run.root, JOB, **{WALK: {**record, "pending": pending}})
    record["pending"] = pending


def _release(root: Any, engine: Any, project_id: str, facts: dict, registration_root: str, *,
             repeat: bool = False) -> Tuple[str, Any]:
    from ouroboros.delegate_registration_policy import REMOVAL_CONFLICTS, remove_registration
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    try:
        return remove_registration(root, engine, project_id, facts, registration_root=registration_root, lane="historical")
    except ClaudexorUnavailable as exc:
        if exc.status_code == 409 and exc.code in REMOVAL_CONFLICTS:
            return "kept", exc.code
        if exc.status_code == 409 and exc.code == "store_mismatch":
            # Another store's refusal settles a first request only; a repeat's answer belongs to the store
            # that took it, and its intent waits for that store's return.
            return ("unknown", "store_changed") if repeat else ("held", exc.code)
        if exc.status_code == 409 and exc.code == "project_root_mismatch":
            return "held", exc.code
        return "unknown", exc.code
    except Exception as exc:  # includes transport TimeoutError after the intent, distinct from a busy local lock
        return "unknown", str(exc)


def _conditional(pending: Any) -> bool:
    """Whether a saved intent proves its first request was the conditional bound release: ``_intend`` wrote ``shared``
    with the whole address before anything left. An older walk's id-only DELETE, or an intent without that address,
    proves nothing about what was sent: it is read, never sent again."""
    return (isinstance(pending, dict) and pending.get("shared") is True
            and all(isinstance(pending.get(key), str) and pending[key] for key in ("project_id", "store_id", "root")))


def _repeats(record: Dict[str, Any], project: Dict[str, Any], facts: dict, lane: str) -> bool:
    """Whether the walk's own unanswered release may go again: its ``historical`` mark and a saved intent that proves a
    bound release of exactly this store, id and root. The mark alone proves nothing; anything else is only read."""
    pending = record.get("pending")
    return (lane == "historical" and _conditional(pending) and pending["project_id"] == str(project["id"])
            and pending["root"] == str(project["root"]) and pending["store_id"] == facts.get("daemon_store_id")
            and pending.get("engine_ref") == facts.get("engine_ref"))


def _answered(pending: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """The definite answer a saved intent holds, its bound receipt (``None``: none; the earlier id-only protocol saved
    none), whatever the engine serves now."""
    from ouroboros.gateways.claudexor_maintenance import registration_receipt_matches

    observed = pending.get("outcome")
    states = {"removed": "released", "absent": "absent", "gone_after_unknown": "absent"}
    if (isinstance(observed, list) and len(observed) == 2 and isinstance(observed[0], str) and observed[0] in states
            and registration_receipt_matches(observed[1], str(pending.get("project_id") or ""), pending.get("store_id"),
                                             pending.get("root"), (states[observed[0]],))):
        return observed[0], observed[1]
    return None


def _retire(run: Any, record: Dict[str, Any], engine: Any, facts: dict, project: Dict[str, Any],
            rows: List[Any]) -> Tuple[str, Any]:
    """One candidate's outcome (``_attempt``). While this walk's own release stays unanswered nothing but its answer
    closes it: any other outcome leaves the intent open and ends the pass there. An older release, read only, ends
    with its own outcome."""
    from ouroboros.delegate_custody_current import removal_status

    outcome = _attempt(run, record, engine, facts, project, rows)
    if outcome[0] == "unknown":
        return outcome
    lane, _ = removal_status(run.root, str(project["id"]), facts)
    return ("unknown", outcome[1]) if _repeats(record, project, facts, lane) else outcome


def _attempt(run: Any, record: Dict[str, Any], engine: Any, facts: dict, project: Dict[str, Any],
             rows: List[Any]) -> Tuple[str, Any]:
    """Prove eligibility under the project's retirement lock, and hold each task's Keep gate across the intent and
    the engine's removal (no wait: a busy gate sends nothing). The walk's own unanswered release goes again only
    through these same proofs; a read finding it absent is gone after an unknown request."""
    from ouroboros.delegate_custody_current import registration_snapshot, removal_status
    from ouroboros.delegate_custody_usage import project_preparing, project_retirement_lock
    from ouroboros.delegate_registration_policy import ANSWERED_BY_OWN_STORE, PROJECT_HAS_THREADS, engine_facts, removal_answered
    from ouroboros.delegate_workspace_hold import workspace_holds, workspace_lock
    from ouroboros.execution_domain import may_share_registration
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.gateways.claudexor_maintenance import bound_registration_supported
    from ouroboros.owner_keep import OWNER_KEEP, gate

    project_id = str(project["id"])
    store_id = facts.get("daemon_store_id", "")
    rows = [row for row in rows if may_share_registration(row, facts)]
    if not rows or not store_id or any(row.daemon_store_id != store_id for row in rows):
        return "held", "store_unproven"
    try:
        with project_retirement_lock(run.root, project_id, engine=engine), workspace_lock(run.root, facts):
            if engine_facts(engine) != facts:
                return "held", "store_changed"
            try:
                named, pending, owed = registration_snapshot(run.root, project_id, facts)
                now = [row for row in named.values() if may_share_registration(row, facts)]
                recorded = _recorded_roots(run.root, now, store_id)
                address = _creator_root(recorded)  # exact: never the list's current name or a local respelling
                workspace_live, workspace_tasks = workspace_holds(run.root, engine, project_id, address, facts,
                                                                  live_task_ids=run.census())
                pending_use = (project_preparing(run.root, project_id, facts)
                               or any(may_share_registration(row, facts) for row in pending)
                               or workspace_live)
            except Exception:
                return "held", "custody_unreadable"
            if {row.run_id for row in now} != {row.run_id for row in rows}:
                return "held", "custody_changed"
            lane, _ = removal_status(run.root, project_id, facts)
            if lane == "custody_kept":  # the engine's remembered refusal: kept for its threads, not unanswered
                return "kept", PROJECT_HAS_THREADS
            repeat = lane in _WALK_LANES
            if lane and not repeat:
                return "held", "engine_unanswered"
            tasks = sorted(workspace_tasks.union(_tasks(now)))
            linked = sorted(set(tasks) & set(record["waiting"]))
            if linked:
                return "deferred", linked
            why = _hold(run, project, address, recorded, now, engine, store_id, tasks, pending=pending_use, owed=owed)
            if why:
                return "held", why
            with contextlib.ExitStack() as gates:
                for task in tasks:
                    why = gates.enter_context(gate(run.root, task, timeout_sec=0))
                    if why:
                        return ("held", why) if why == OWNER_KEEP else ("busy", why)
                if engine_facts(engine) != facts:
                    return "held", "store_changed"
                try:
                    if not bound_registration_supported(engine):
                        return "held", "registration_unproven"
                    binding = engine.bound_registration(project_id, store_id, address)
                except ClaudexorUnavailable as exc:
                    if repeat and exc.status_code == 409 and exc.code in ANSWERED_BY_OWN_STORE:
                        removal_answered(run.root, project_id, facts, address, "")  # its own store's answer
                    if repeat and exc.status_code == 409 and exc.code == "store_mismatch":
                        return "unknown", "store_changed"  # another base answers nothing: the intent waits
                    return "held", exc.code
                if binding["state"] == "absent":  # invalidate earlier admissions that saw this id
                    removal_answered(run.root, project_id, facts, address, "")
                    return ("gone_after_unknown" if repeat else "absent"), binding
                if repeat and not _repeats(record, {"id": project_id, "root": address}, facts, lane):
                    return "held", "engine_unanswered"  # an older release, or one no saved intent proves: only read
                _intend(run, record, {"project_id": project_id, "root": address, "store_id": store_id,
                                      **({"engine_ref": facts["engine_ref"]} if "engine_ref" in facts else {}),
                                      "shared": True})
                outcome = _release(run.root, engine, project_id, facts, binding["root"], repeat=repeat)
                if outcome[0] in ("removed", "absent", "gone_after_unknown"):
                    _intend(run, record, {**record["pending"], "outcome": list(outcome)})
                    _finish_removal(run, project_id, facts, binding["root"])
                return outcome
    except TimeoutError:
        return "busy", "project_lock"


def _apply(run: Any, record: Dict[str, Any], project: Dict[str, Any], outcome: Tuple[str, Any], *,
           main: bool) -> bool:
    """Record one outcome; False ends this pass at the project (its answer is owed to the next one)."""
    project_id = str(project["id"])
    kind, detail = outcome
    retry = record.get("retry") if isinstance(record.get("retry"), dict) else {}
    if kind == "unknown":  # the intent stays until a definite answer: a readback, or the same release repeated
        run.hold("historical_release_unanswered")
        return False
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
        intent = {key: value for key, value in intent.items() if key not in ("outcome", "shared")}
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
            **({"receipt": detail} if isinstance(detail, dict) else {"code": detail} if detail else {})}})
    if main:
        record["project_cursor"] = project_id
    return True


def _finish_removal(run, project_id, facts, folder):
    from ouroboros.delegate_custody_current import mark_removal, removal_status
    from ouroboros.delegate_registration_policy import removal_answered

    removal_answered(run.root, project_id, facts, folder)  # the folder this release held first
    lane, _ = removal_status(run.root, project_id, facts)
    if lane in _WALK_LANES:
        mark_removal(run.root, project_id, facts, "")


def _reconcile(run: Any, record: Dict[str, Any], listing: Dict[str, Dict[str, Any]], engine: Any, listed_by: dict,
               named: Dict[str, List[Any]]) -> bool:
    """Read back an unanswered request in the same store; a listed project never proves non-execution. Absent ends
    it as gone after an unknown request and a confirmed 409 (a dead binding) as held. Only a ``shared`` intent goes
    again, through ``_retire`` and from its saved address, and only while the served base is the one that took it
    and the engine offers the bound operations: otherwise, like an unread one, it waits for the next pass, its
    cursor and ``done_at`` with it. An older intent is only read: present or unreadable, it ends as held."""
    pending = record.get("pending") if isinstance(record.get("pending"), dict) else None
    if not pending:
        return True
    project_id = str(pending.get("project_id") or "")
    main = project_id not in (record.get("deferred") or {})
    conditional = _conditional(pending)
    project = ({"id": project_id, "root": str(pending["root"])} if conditional  # the address it saved, not the list's
               else listing.get(project_id) or {"id": project_id, "root": str(pending.get("root") or "")})
    from ouroboros.delegate_custody_current import mark_removal, removal_status
    from ouroboros.delegate_custody_usage import project_retirement_lock
    from ouroboros.delegate_registration_policy import ANSWERED_BY_OWN_STORE, engine_facts, removal_answered
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.gateways.claudexor_maintenance import bound_registration_supported

    store_id, folder = pending.get("store_id"), pending.get("root")
    original = {"daemon_store_id": store_id if isinstance(store_id, str) else "",
                **({"engine_ref": pending["engine_ref"]} if "engine_ref" in pending else {})}
    with project_retirement_lock(run.root, project_id, engine=original):
        same = bool(pending.get("project_id") == project_id and original["daemon_store_id"]
                    and original == listed_by == engine_facts(engine))
        answer = _answered(pending)
        why, present = "store_changed", None  # present: what a bound read answered (None: none could be asked)
        if same and answer is None:
            why = "engine_unanswered"
            try:
                if folder and bound_registration_supported(engine):
                    reply = engine.bound_registration(project_id, original["daemon_store_id"], folder)
                    present = reply["state"] != "absent"
                    if not present:
                        answer = ("gone_after_unknown", reply)
                        _intend(run, record, {**pending, "outcome": list(answer)})
            except ClaudexorUnavailable as exc:
                if exc.status_code == 409 and exc.code == "store_mismatch":
                    return _apply(run, record, project, ("unknown", "store_changed"), main=main)  # another base: no answer
                if not (exc.status_code == 409 and exc.code in ANSWERED_BY_OWN_STORE):  # unread: read again next pass
                    return _apply(run, record, project, ("unknown", exc.code), main=main)
                removal_answered(run.root, project_id, original, folder, "")  # its own store: dead for the first request too
                return _apply(run, record, project, ("held", "root_changed"), main=main)
        if answer is not None:
            _finish_removal(run, project_id, original, folder)
            return _apply(run, record, project, answer, main=main)
        if conditional and present is None:  # no addressed answer: its base or the bound operations are away
            return _apply(run, record, project, ("unknown", why), main=main)
        # Carry the older walk’s unresolved intent beyond its one-shot summary, as the lane that is only read.
        lane, _ = removal_status(run.root, project_id, original)
        if lane == "historical_walk" or (lane == "store_unproven" and not store_id):
            mark_removal(run.root, project_id, original, "historical_walk")
        if not conditional:
            return _apply(run, record, project, ("held", why), main=main)
    # Still present: the same bound release again, only after the same proofs under the same locks.
    return _apply(run, record, project, _retire(run, record, engine, listed_by, project, named.get(project_id, [])),
                  main=main)


def _reconcile_saved(run: Any, record: Dict[str, Any], project_id: str) -> bool:
    """The saved release is rechecked alone, before any pass-wide read: its own project's rows answer for it,
    so an original store that stays away costs one addressed question a pass — not every custody row and the
    whole inventory. A conditional intent carries its whole address (``_conditional``); an older id-only one
    is read through the full pass, as before. False ends the pass with the intent still open; True lets the
    pass below run — whose ``_reconcile`` then has nothing left, and the cursor's discovery resumes with the
    answer."""
    from ouroboros.delegate_custody_current import runs
    from ouroboros.delegate_registration_policy import engine_facts

    if not _conditional(record.get("pending")):
        return True
    if run.stopped or run.stop():
        run.stopped = True
        return False
    if run.census() is None:
        return False  # no task can be proven over now: the read waits for the next pass
    engine = run.engine()
    if engine is None:
        run.hold("historical_engine_unavailable")
        return False
    return _reconcile(run, record, {}, engine, engine_facts(engine),
                      {project_id: list(runs(run.root, "project_id=?", (project_id,)).values())})


def _settled(record: Dict[str, Any]) -> bool:
    """A finished walk with no work this pass can read: nothing unresolved, and every deferred project still
    waits for its sessions (the saved release below may settle the last open work mid-pass)."""
    return bool(record.get("done_at") and not (record.get("pending") or record.get("removed_ids")) and all(
        set(tasks) & set(record["waiting"]) for tasks in (record.get("deferred") or {}).values()))


def _registrations(run: Any, record: Dict[str, Any]) -> None:
    from ouroboros.delegate_custody_current import runs
    from ouroboros.delegate_registration_policy import engine_facts
    from ouroboros.utils import utc_now_iso

    named: Dict[str, List[Any]] = {}
    if _settled(record):
        return  # every deferred project still waits for its sessions: nothing to read here
    if isinstance(record.get("pending"), dict) and not _reconcile_saved(
            run, record, str(record["pending"].get("project_id") or "")):
        return
    if _settled(record):  # the saved release just settled the last open work: this pass reads nothing more
        return
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
    listed_by = engine_facts(engine)
    try:
        listing = {str(row["id"]): row for row in reader()}
    except Exception:  # unreadable or malformed: it proves no project's absence
        run.hold("historical_projects_unread")
        log.info("The historical walk could not read the engine's projects", exc_info=True)
        return
    record.setdefault("registrations", {"candidates": 0, "removed": 0, "absent": 0, "kept": {}, "held": {}})
    if listed_by != engine_facts(engine):
        run.hold("historical_projects_unread")
        return
    if not _reconcile(run, record, listing, engine, listed_by, named):
        return
    for project_id, tasks in sorted((record.get("deferred") or {}).items()):
        if set(tasks) & set(record["waiting"]):
            continue  # its sessions' release is still owed
        if run.stop():
            run.stopped = True
            return
        project = listing.get(project_id)
        outcome = ("held", "not_listed") if project is None else _retire(
            run, record, engine, listed_by, project, named.get(project_id, []))
        if not _apply(run, record, project or {"id": project_id}, outcome, main=False):
            return
    if not record.get("done_at"):
        cursor = str(record.get("project_cursor") or "")
        for project_id in sorted(pid for pid in listing if pid > cursor and pid in named):
            if run.stop():
                run.stopped = True
                return
            if not _apply(run, record, listing[project_id],
                          _retire(run, record, engine, listed_by, listing[project_id], named[project_id]), main=True):
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
            "registration_unproven": "the engine cannot confirm their registration",
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
    unconfirmed = stays.pop("not_listed", 0)  # presence was never confirmed: no retention is claimed for it
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
    if unconfirmed:
        clauses.append(f"the registrations of {_plural(unconfirmed, 'project')} could not be confirmed")
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
