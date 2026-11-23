"""An execution workspace's hold on another project's temporary engine registration of its folder.

ARCHITECTURE §10 invariant 18: workspace use holds, never owns. A start for project P that runs its files in a folder
Q (``execution.workspaceRoot``) holds the release of a temporary registration X of another project while the engine
may name Q as X's folder: from P's preparation, before the engine has the command, through a START whose answer is
unknown, to its live run. A start in another folder R holds nothing. Only the engine tells them apart, in its own
domain: the addressed read of X (``GET /v2/projects/:id/registration`` bound by ``expectedStoreId`` and
``expectedRoot``, with ``candidate`` set to the folder exactly as the start sends it) answers ``candidateMatch``.
``same`` holds; ``different``, or X ``absent``, frees; anything else holds: ``unknown``, no field from an engine that
ignores the query, a refusal or a failed read. A folder equal to X's recorded root holds without a read. The host
never resolves a folder on its own filesystem, lists the engine's projects or registers a root to learn an address;
the answer decides this hold only, is never authority, permission or a route, and never rides a START.

One lock per engine route (``workspace_lock``) orders what meets here, each observation published durably before its
side lets the lock go: a preparation's hold (its folder and start written beside its process-held lock), a START's
claim, the owed duty of a temporary registration being created, and a bound release with the folder mark it sets
before it leaves and keeps until its answer (``FOLDER`` and X's exact recorded root). A release reads the holds and
asks about each folder before it sends; a preparation reads the unanswered releases and asks about its own folder
before it holds, and a start's claim asks again. Whichever comes first, the later one sees it.

A run that ended still holds through its task line's Continue and its task's Keep gate, found by addressed reads only:
the custody set, the live census, the recoverable handles, the registration's exact recorded root, and the uses its
owed duty keeps (``workspace_uses``, and one ``custody_registration_uses`` row per linked START), which last exactly as
long as that duty. Both sides write them under the lock: a START's claim names the duties alive on its engine
(``registration_links``) and its START_REQUESTED adds its invocation to each in the same commit; a duty's creation
records the live uses it observed before its possibly-creating request, under the one hold that writes the duty
(``creation_fence``); unable to observe, it sends nothing. Neither asks the engine; the release does, for each use it
finds, and gates the Keep of every related ended user. An unreadable preparation note is recorded as unknown and holds
that one registration. A use that ended before a registration existed never met it: its next start prepares again.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging

log = logging.getLogger(__name__)

FOLDER = "workspace:"  # a removal mark keyed by a registration's exact recorded root while its release is unanswered
FREE = ("different", "absent")  # the only answers that free a folder; every other one holds it
UNKNOWN = "unknown"  # an owed duty's use no observation could name: it holds that registration
UNCONFIRMED = "The release of this folder's registration is still unconfirmed. Nothing was sent."


def request_workspace(body) -> str:
    """The execution folder a start body names, exactly as sent (``execution.workspaceRoot``); "" for none."""
    execution = body.get("execution") if isinstance(body, dict) else None
    root = execution.get("workspaceRoot") if isinstance(execution, dict) else None
    return root if isinstance(root, str) else ""


def start_workspace(body, execution_root="", thread_folder="") -> str:
    """The one fact of the folder a start's work runs in, as its preparation, its claim (``registration_workspace``)
    and every later reader take it: the body's ``execution.workspaceRoot``, else the recorded execution root (a
    private copy, a direct or in-place folder), else the folder of the thread it is bound to (the plan thread's copy,
    or the engine's ``thread.repoRoot``), which a thread turn's body never carries. The body is never changed."""
    return request_workspace(body) or str(execution_root or "") or str(thread_folder or "")


def workspace_lock(drive, engine):
    """The engine route's one lock over workspace holds, recorded starts' claims and registration releases."""
    from ouroboros.delegate_custody_usage import project_retirement_lock

    return project_retirement_lock(drive, FOLDER, engine=engine)


def acquired(stack, lock, why):
    """Enter ``lock`` on ``stack`` before anything is sent: a busy acquisition refuses typed (a first start
    definitely never ran); a timeout of what runs under it later is that operation's own outcome."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    try:
        return stack.enter_context(lock)
    except TimeoutError as exc:
        refusal = ClaudexorUnavailable("registration_busy", f"{why}: {exc}. Nothing was sent.")
        refusal.registration_claim_refused = True
        raise refusal from exc


def folder_release(drive, registration_root, engine) -> str:
    """The registration whose bound release of this exact recorded root is unanswered on this engine ("": none)."""
    from ouroboros.delegate_custody_current import removal_status

    return removal_status(drive, FOLDER + registration_root, engine)[0] if registration_root else ""


def mark_folder_release(drive, registration_root, engine, project_id) -> None:
    """Set (``project_id``) or end ("") the folder mark of a bound release of this exact recorded root."""
    from ouroboros.delegate_custody_current import mark_removal

    mark_removal(drive, FOLDER + registration_root, engine, project_id)


def candidate_answer(gateway, project_id, store_id, registration_root, folder) -> str:
    """The engine's word on ``folder`` beside registration ``project_id`` bound by store and root: ``same``,
    ``different``, ``absent`` (no such registration now) or ``unknown`` (a failure is never a negation)."""
    from ouroboros.gateways.claudexor_maintenance import bound_registration_supported, candidate_match

    if folder == registration_root:
        return "same"
    try:
        if not (project_id and store_id and registration_root and folder and bound_registration_supported(gateway)):
            return "unknown"
        reply = gateway.bound_registration(project_id, store_id, registration_root, candidate=folder)
    except Exception:
        log.debug("The engine did not compare a workspace with registration %s", project_id, exc_info=True)
        return "unknown"
    return "absent" if reply["state"] == "absent" else candidate_match(reply)


def unanswered_releases(drive, engine) -> list:
    """``(recorded root, project id, store)`` of each bound release on this engine whose answer is still owed."""
    from ouroboros import store
    from ouroboros.delegate_custody_current import _REMOVALS
    from ouroboros.execution_domain import registration_engine

    route, store_id = registration_engine(engine)
    with store.open_store(drive, write=False) as txn:
        rows = txn.set_rows(_REMOVALS)
    return [(mark[len(FOLDER):], project_id, mark_store)
            for (mark, mark_route), marks in ((json.loads(group), marks) for group, marks in rows.items())
            if mark_route == route and mark.startswith(FOLDER)
            for mark_store, project_id in marks.items()
            if project_id and (not store_id or mark_store in (store_id, ""))]


def fence_refusal(drive, gateway, folder, engine) -> str:
    """Why a start in ``folder`` waits: an unanswered release on its engine the engine does not rule out ("": none)."""
    for root, project_id, store_id in unanswered_releases(drive, engine):
        if candidate_answer(gateway, project_id, store_id, root, folder) not in FREE:
            return UNCONFIRMED
    return ""


def prepare_workspace_use(drive, gateway, folder, engine, invocation_id, preparation) -> None:
    """Hold ``folder`` for a start being prepared, once no unanswered release on its engine may concern it; the hold
    ends with ``preparation``, after the START row took it over. Nothing is looked up, registered or claimed."""
    if preparation is None or not folder:
        return
    with contextlib.ExitStack() as fence:
        acquired(fence, workspace_lock(drive, engine), "The workspace fence is busy")
        _hold_folder(drive, gateway, folder, engine, invocation_id, preparation)


def _hold_folder(drive, gateway, folder, engine, invocation_id, preparation) -> None:
    """Under ``workspace_lock``: ``prepare_workspace_use`` without taking the lock again."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    if preparation is None or not folder:
        return
    if why := fence_refusal(drive, gateway, folder, engine):
        raise ClaudexorUnavailable("registration_unconfirmed", why)
    acquired(preparation, _preparing(drive, folder, engine, invocation_id), "The workspace preparation is busy")


@contextlib.contextmanager
def creation_fence(drive, gateway, folder, engine, invocation_id, preparation):
    """One hold of ``workspace_lock`` around a start that may create a temporary registration: its own folder's hold,
    then (``observe``, right before a request that may create it) the live uses on its engine route, of any store,
    not yet knowing the id. The caller records them in that registration's owed duty before the lock ends, so no
    claim lands between the observation and the duty. A busy lock or an unreadable observation sends nothing
    (typed); a registration the lookup already found is adopted (``resolve_registration``). Yields
    ``(observe, uses)``: ``uses`` holds what ``observe`` read, the start's own preparation left out."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    uses, busy = [], []

    def observe():
        if busy:  # no observation without the lock: only a registration the lookup found can serve
            raise busy[0]
        try:
            route = {**engine, "store_identity_unproven": True}  # every store of the route: the new one is not known
            uses[:] = [use for use in _live_uses(drive, None, route) if use != {"invocation_id": invocation_id}]
        except Exception as exc:  # unknown is never none: the request that may create waits
            refusal = ClaudexorUnavailable("registration_unconfirmed", f"The live uses of this folder could not be "
                                           f"read: {type(exc).__name__}. Nothing was sent.")
            refusal.registration_claim_refused = True
            raise refusal from exc

    with contextlib.ExitStack() as fence:
        try:
            acquired(fence, workspace_lock(drive, engine), "The workspace fence is busy")
        except ClaudexorUnavailable as refusal:
            if preparation is not None and folder:
                raise  # its own folder cannot be held either
            busy.append(refusal)
        else:
            _hold_folder(drive, gateway, folder, engine, invocation_id, preparation)
        yield observe, uses


def _bucket(drive, engine):
    from ouroboros.delegate_custody_usage import _project_lock_path

    return _project_lock_path(drive, FOLDER, engine).with_suffix(".preparing")


@contextlib.contextmanager
def _preparing(drive, folder, engine, invocation_id):
    """A process-held preparation (taken under ``workspace_lock``): its lock and a note naming its folder and start.
    Each preparation owns its own pair (a nonce beside the start's digest): two preparations of one start, a retry
    racing its original, hold and release only their own."""
    import uuid

    from ouroboros.delegate_custody_usage import _preparation_store, _take_project_lock
    from ouroboros.platform_layer import release_exclusive_file_lock

    bucket = _bucket(drive, engine)
    token = hashlib.sha256(invocation_id.encode('utf-8')).hexdigest()[:24]
    name = f"{_preparation_store(engine)}-{token}-{uuid.uuid4().hex[:16]}"
    lock, note = bucket / f"{name}.lock", bucket / f"{name}.folder"
    bucket.mkdir(parents=True, exist_ok=True)
    note.write_text(json.dumps({"folder": folder, "invocation_id": invocation_id}), encoding="utf-8")  # before the hold
    fd = _take_project_lock(lock, 0)
    if fd is None:
        note.unlink(missing_ok=True)
        raise TimeoutError("the workspace preparation could not be held")
    try:
        yield
    finally:
        release_exclusive_file_lock(lock, fd)
        note.unlink(missing_ok=True)


def _prepared(drive, engine) -> list:
    """Under ``workspace_lock``: ``(folder, start)`` of this engine's live preparations (``(None, {})``: one whose note
    is unreadable). A preparation that ended or whose process died leaves nothing behind it here."""
    from ouroboros.delegate_custody_usage import _preparation_store, _take_project_lock
    from ouroboros.platform_layer import release_exclusive_file_lock

    bucket, store = _bucket(drive, engine), _preparation_store(engine)
    uses = []
    for lock in sorted(bucket.glob("*.lock")):
        if store != "unbound" and lock.name.split("-", 1)[0] not in (store, "unbound"):
            continue
        fd = _take_project_lock(lock, 0)  # taken only once its preparer released it or was proven dead
        if fd is None:
            try:
                note = json.loads(lock.with_suffix(".folder").read_text(encoding="utf-8"))
                uses.append((str(note["folder"]), {"invocation_id": str(note.get("invocation_id") or "")}))
            except (OSError, ValueError, KeyError, TypeError):
                uses.append((None, {}))
            continue
        release_exclusive_file_lock(lock, fd)
        lock.with_suffix(".folder").unlink(missing_ok=True)
    for note in bucket.glob("*.folder"):  # a preparer that died between its note and its lock
        if not note.with_suffix(".lock").exists():
            note.unlink(missing_ok=True)
    return uses


def _live_rows(txn, project_id) -> tuple:
    """Inside one read: the pending STARTs' request facts and the rows custody holds, of every other project
    (``None``: of every project, before the registration has an id)."""
    from ouroboros.delegate_custody_current import HELD, HELD_WHERE

    if project_id is None:
        pending = txn.conn.execute("SELECT facts FROM obligations WHERE set_name='custody_open' AND id LIKE 'invocation:%'")
        return [json.loads(row["facts"])["request"] for row in pending], txn.custody_raw_rows(HELD_WHERE, (HELD,))
    pending = [json.loads(row["facts"])["request"] for row in txn.conn.execute(
        "SELECT facts FROM obligations WHERE set_name='custody_open' AND id LIKE 'invocation:%' AND CASE "
        "WHEN json_valid(facts) THEN json_extract(facts, '$.request.project_id') IS NOT ? ELSE 1 END", (project_id,))]
    return pending, txn.custody_raw_rows(f"{HELD_WHERE} AND project_id IS NOT ?", (HELD, project_id))


def _alive_duties(drive, engine) -> list:
    """The owed keys of the temporary registrations alive on this engine (the host's own duties): a few rows."""
    from ouroboros import store
    from ouroboros.delegate_custody_current import OWED
    from ouroboros.execution_domain import may_share_registration

    with store.open_store(drive, write=False) as txn:
        duties = txn.set_rows(OWED)
    return sorted(key for key, duty in duties.items() if may_share_registration(duty, engine))


@contextlib.contextmanager
def start_fence(drive, payload):
    """Around the durable claim of a START with an explicit folder: ``workspace_lock``, under which the claim names
    the temporary registrations alive on its engine (``registration_links``). Its START_REQUESTED adds its invocation
    to each one's owed uses in the same commit (``delegate_custody_current._owed_updates``). A busy lock refuses
    the claim typed: nothing is recorded or sent. Any claim does so, one without a registration of its own included."""
    if not payload.get("registration_workspace"):
        yield
        return
    with contextlib.ExitStack() as fence:
        acquired(fence, workspace_lock(drive, payload), "The workspace fence is busy")
        payload["registration_links"] = _alive_duties(drive, payload)
        yield


def _live_uses(drive, project_id, engine) -> list:
    from ouroboros import store
    from ouroboros.delegate_custody_current import _decode
    from ouroboros.execution_domain import may_share_registration

    uses = [start if folder is not None else {UNKNOWN: "preparation_unreadable"} for folder, start in _prepared(drive, engine)]
    with store.open_store(drive, write=False) as txn:
        pending, rows = _live_rows(txn, project_id)
    uses += [{"invocation_id": str(request.get("invocation_id") or "")} for request in pending
             if may_share_registration(request, engine) and _recorded_folders(drive, request, request)]
    uses += [{"run_id": run_id} for run_id, row in ((run_id, json.loads(raw)) for run_id, raw in rows.items())
             if may_share_registration(row, engine) and not _decode(row).settled
             and _recorded_folders(drive, row, row.get("request_event") or {})]
    return [use for index, use in enumerate(uses) if use not in uses[:index]]


def workspace_holds(drive, gateway, project_id, registration_root, engine, *, live_task_ids=None,
                    recoverable_handles=()) -> tuple:
    """Under ``workspace_lock``: ``(True, set())`` when a use of another project may run in this registration's
    folder, else ``(False, tasks)``: the ended users there whose Keep gates the release also holds.

    A preparation, a pending or unknown START and a live run hold unless the engine rules their folder out. A run that
    ended holds through its Continue (and its task's Keep gate) unless the engine rules its folder out, once one of the
    addressed reads in the module's description finds it. An unknown use in the owed duty holds."""
    from ouroboros import store
    from ouroboros.delegate_continuation import still_continuable
    from ouroboros.delegate_custody_current import _decode, linked_uses, owed_key
    from ouroboros.execution_domain import may_share_registration, registration_engine
    from ouroboros.store_hot_facts import CUSTODY_WORKSPACE, CUSTODY_WORKSPACE_UNDERIVED, WORKSPACE_VERSION

    if not registration_root:
        return False, set()
    uses = [(folder, None) for folder, _start in _prepared(drive, engine)]
    with store.open_store(drive, write=False) as txn:  # a pending row and its STARTED row share one read
        pending, rows = _live_rows(txn, project_id)
        met = linked_uses(txn, owed_key(project_id, engine))
        if any(not isinstance(use, dict) or UNKNOWN in use for use in met):
            return True, set()
        named = [(use.get("run_id"), use.get("invocation_id")) for use in met]
        named += [(run, invocation) for _task, run, invocation in recoverable_handles]
        for run_id, invocation_id in named:  # addressed: the uses its duty met, the recoverable handles
            for column, value in (("run_id", run_id), ("invocation_id", invocation_id)):
                if isinstance(value, str) and value:
                    rows.update(txn.custody_raw_rows(f"{column}=? AND project_id IS NOT ?", (value, project_id)))
        if live_task_ids:
            rows.update(txn.custody_raw_rows("task_id IN (SELECT value FROM json_each(?)) AND project_id IS NOT ?",
                                             (json.dumps(sorted(live_task_ids)), project_id)))
        if int(txn.meta("schema_version") or 1) >= WORKSPACE_VERSION:  # the projected column and its partial indexes
            at_root = {row[0]: row[1] for row in txn.conn.execute(
                "SELECT run_id, facts FROM custody_runs INDEXED BY custody_workspace "
                "WHERE registration_workspace<>'' AND registration_workspace=? AND project_id IS NOT ? "
                "UNION ALL SELECT run_id, facts FROM custody_runs INDEXED BY custody_workspace_underived "
                f"WHERE {CUSTODY_WORKSPACE_UNDERIVED} AND ({CUSTODY_WORKSPACE})=? AND project_id IS NOT ?",
                (registration_root, project_id) * 2)}
        else:  # a drive this boot has not migrated yet: a passive read runs no DDL, so one raw scan of that drive
            at_root = txn.custody_raw_rows(f"project_id IS NOT ? AND ({CUSTODY_WORKSPACE})=?",
                                           (project_id, registration_root))
    for request in pending:
        if may_share_registration(request, engine):
            uses.extend((folder, None) for folder in _recorded_folders(drive, request, request))
    for run_id, raw in {**rows, **at_root}.items():
        row = json.loads(raw)
        if may_share_registration(row, engine):
            entry = _decode(row)
            folders = [registration_root] if run_id in at_root else _recorded_folders(drive, row, row.get("request_event") or {})
            uses.extend((folder, entry if entry.settled else None) for folder in folders)
    store_id, ruled_out, tasks = registration_engine(engine)[1], {}, set()
    for folder, ended in sorted(uses, key=lambda use: use[1] is not None):  # what holds outright is asked first
        if folder is not None and folder not in ruled_out:  # one question per folder; only the engine relates it
            ruled_out[folder] = candidate_answer(gateway, project_id, store_id, registration_root, folder) in FREE
        if folder is not None and ruled_out[folder]:
            continue
        if ended is None or still_continuable(drive, ended, live_task_ids, recoverable_handles=recoverable_handles):
            return True, set()
        if ended.task_id:  # every related ended user: its gate decides Keep at the release, a late one included
            tasks.add(ended.task_id)
    return False, tasks


def _recorded_folders(drive, row, request) -> list:
    """The execution folders a use's records name: the one its START recorded, else an older row's geometry and
    body (``[None]``: a body that cannot be read, which no answer can rule out)."""
    from ouroboros.delegate_pending import request_body

    if (captured := request.get("registration_workspace")) is not None:
        return [captured] if captured else []
    folders = [root for root in (row.get("execution_root"), request.get("execution_root")) if root]
    body = request.get("request")
    if request.get("request_ref") and (request.get("request_shape") or {}).get("execution") is not False:
        body = request_body(drive, request)
        if body is None:
            return [None]
    return folders + [folder for folder in (request_workspace(body),) if folder]
