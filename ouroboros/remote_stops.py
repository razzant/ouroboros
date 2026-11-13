"""Emergency Stop's owed stops on remote engines: the durable intent, the stop a later start delivers on its own
engine, its discharge before any re-adoption, and an unknown start looked for again on its engine's store, never sent
again.

A remote engine (``remote_engines``) outlives this installation's Panic: Panic stops the owned engine and every local
process, its own SSH forwards included (they are ordinary members of its census), but a remote engine keeps working
until each of our runs there is stopped by its own control request. So the intent is durable before anything else is
tried, on the custody row every later reader already consults: the ``CONTAINMENT_FAULT`` row with reason
``PANIC_STOP_UNCONFIRMED``, keyed by the run, or, for a start whose answer was lost before a run id came back, by its
invocation (the store keeps it on the open invocation row, ``delegate_custody_current``). Both Panic doors produce it:
the server's own Panic (``panic_stop_work``, best effort, never awaited and never ahead of a local kill) and a
launcher's Panic flag, which the next boot translates before any work is restored (``supervisor.queue_snapshot``) and
again before its receipt is acknowledged (``supervisor.worker_chat_lane``). An installation without a remote engine
connection reads and writes nothing here.

Killing our own forward cuts a stop Panic sends even when the engine is reachable, so the stop is expected at the next
start, not during Panic: until then remote work may continue (the owner is told where a remote engine is chosen and
where its owed stops are shown). The maintenance sweep (``delegate_custody_reconcile``) delivers every owed stop on the
run's recorded engine as soon as it answers, with no task asking: an owed run is cancelled whoever owns it and stays
owed until a terminal read-back; an owed start is looked for without a POST (``taskGroupId``, a UUID derived from its
invocation and saved before the POST, read on the store its invocation recorded, and, for a staged copy, its run
folder). Each run or queued job found gets its own owed stop and is cancelled the same way. The engine's group lookup
answers for its currently served generations only, so finding nothing proves nothing: the start stays owed and
visible, and it is looked for again at a bounded pace (``lookup_due``). Every path that gives a task its run again
(``execution_domain.run_gateway``, a retried or continued start) passes through ``discharged`` /
``refuse_owed_start`` first. A sent-unknown start that owes no stop is found the same way by every accepted-start
reader (``locate_started``: a sweep at that pace, an explicit retry, a review seat's rejoin) and bound once, with
nothing cancelled, replayed or sent.

A raw model operation is not owed here: nothing local executes its tool calls after Panic, and its unsettled attempt
row is already the durable fact recovery reads on its own engine. Panic and the boot after a Panic send it one
best-effort cancel; a create whose answer was lost has no handle any route can find, and stays unresolved.
"""

from __future__ import annotations

import logging
import pathlib
import threading
import time
import uuid
from typing import Any, Dict, List, Mapping, Optional, Tuple

log = logging.getLogger(__name__)

# The containment-fault reason of remote work Emergency Stop asked to stop: open until a terminal read-back (a run or a
# queued job) proves it ended; a start without a run id keeps it on its invocation until a lookup finds its run.
PANIC_STOP_UNCONFIRMED = "panic_remote_stop_unconfirmed"
TASK_GROUPS_ROUTE = ("GET", "/v2/task-groups/:id/runs")
_TASK_GROUP_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://ouroboros.local/remote-start")
# One best-effort stop sent while Panic runs: the transport bound of one request through a forward Panic's census is
# ending anyway. Nothing waits for it.
_PANIC_STOP_SEC = 3.0
_RUN_SCAN_PAGES = 5  # newest-first pages of 200 run summaries read to find a staged start by its folder
# A sent-unknown start is looked for on the first pass of each process (the next start's), then again at most this
# often, through the sweep's own attach, whether the last pass reached its engine or not: unchanged history is not
# re-read every tick, and an engine that answers again is asked again without any task or tunnel asking first.
_RELOOK_SEC = 900.0
_LOOKED_UP: Dict[str, float] = {}
# Every remote tunnel is published, and every remote run, start and model operation reaches its engine, through this
# folder of the data root (``remote_engines._tunnel_dir``'s parent, kept equal by test); it is never removed, so its
# absence proves this installation never reached a remote engine there.
TUNNEL_HOME = ("state", "remote_engines")


def task_group_id(invocation_id: str) -> str:
    """The ``taskGroupId`` a remote start carries: derived from its invocation id, so it is known before the POST."""
    return str(uuid.uuid5(_TASK_GROUP_NAMESPACE, str(invocation_id)))


def remote_configured(root: Any = None) -> bool:
    """Whether this installation has remote engine work at ``root`` (``delegate_recovery.remote_engines_named``)."""
    from ouroboros.delegate_recovery import remote_engines_named

    return remote_engines_named(root)


def remote_holdings(root: Any) -> bool:
    """Whether current custody at ``root`` still holds remote work Emergency Stop may owe a stop to: an open run or a
    start without a run id recorded on a remote engine, or an open model operation there; asked when no connection is
    named, so a removed row's recorded work keeps its stop. An installation that never reached a remote engine
    (no ``TUNNEL_HOME`` folder) holds none, and nothing is read. Current custody only (the held index, the open set,
    open attempts), never history; a read that fails raises."""
    root = pathlib.Path(root)
    if not root.joinpath(*TUNNEL_HOME).is_dir():
        return False
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_custody_current import open_held_runs
    from ouroboros.execution_domain import is_remote_ref, recorded_ref

    return (any(row.run_id and not row.settled and is_remote_ref(row.engine_ref) for row in open_held_runs(root).values())
            or any(is_remote_ref(recorded_ref(record)) for record in custody.pending_invocations(root))
            or bool(_open_remote_operations(root)))


def _fault(record: Any) -> Dict[str, Any]:
    fault = getattr(record, "containment_fault", None)
    if fault is None and isinstance(record, Mapping):
        fault = record.get("containment_fault")
    return fault if isinstance(fault, dict) else {}


def owed(record: Any) -> bool:
    """Whether ``record`` (a run's custody or a pending invocation) carries Emergency Stop's owed remote stop."""
    return _fault(record).get("reason") == PANIC_STOP_UNCONFIRMED


def incident_after(current: Mapping[str, Any], row: Mapping[str, Any]) -> Dict[str, Any]:
    """The one incident a run, or a start without a run id, holds after a CONTAINMENT_FAULT row (both reducers apply
    it: ``delegate_custody._apply``, ``delegate_pending._invocation_fault``): that row, except that the owed stop is
    discharged only by proof (a resolution, a settlement or a closure). A later incident of another reason about the
    same work (a failed read, an unverified cancel, another writer's) never replaces it in that slot, so no reader
    loses the stop; the chain keeps that row as its record."""
    if (current or {}).get("reason") == PANIC_STOP_UNCONFIRMED != row.get("reason"):
        return dict(current)
    return dict(row)


def _detail(ref: Mapping[str, Any]) -> str:
    return (f"Emergency Stop asked the remote engine {ref.get('connection')!r} to stop this work; until its own "
            "engine reads it back stopped, it may still be working there. The stop is delivered when that engine "
            "answers this installation again.")


# --- the intent -----------------------------------------------------------------------------------------------

def record_owed_stops(root: Any) -> Dict[str, int]:
    """Durably mark every open remote run and every remote start without a run id as owing a stop (idempotent:
    what already owes one is left as it is). Raises when custody cannot be read, so a boot keeps its Panic input."""
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_custody_current import open_held_runs
    from ouroboros.execution_domain import is_remote_ref, recorded_ref

    root = pathlib.Path(root)
    runs = [row for row in open_held_runs(root).values()
            if row.run_id and not row.settled and is_remote_ref(row.engine_ref)]
    for row in runs:
        if not owed(row):
            custody.record_containment_fault(root, row, PANIC_STOP_UNCONFIRMED, _detail(row.engine_ref),
                                             engine_ref=dict(row.engine_ref))
    starts = [record for record in custody.pending_invocations(root) if is_remote_ref(recorded_ref(record))]
    for record in starts:
        if not owed(record):
            custody.emit(root, custody.CONTAINMENT_FAULT, {
                "run_id": "", "invocation_id": str(record["invocation_id"]), "task_id": str(record.get("task_id") or ""),
                "reason": PANIC_STOP_UNCONFIRMED, "detail": _detail(recorded_ref(record))})
    return {"runs": len(runs), "starts": len(starts)}


def owed_stop_counts(root: Any) -> Dict[str, Dict[str, int]]:
    """``{connection: {"runs", "starts"}}`` of the stops Emergency Stop still owes on each remote engine (what the
    Remote engines rows show until each is read back stopped); a custody read failure raises."""
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_custody_current import open_held_runs
    from ouroboros.execution_domain import recorded_ref

    counts: Dict[str, Dict[str, int]] = {}
    for row in open_held_runs(pathlib.Path(root)).values():
        if owed(row) and not row.settled:
            name = str((row.engine_ref or {}).get("connection") or "")
            counts.setdefault(name, {"runs": 0, "starts": 0})["runs"] += 1
    for record in custody.pending_invocations(pathlib.Path(root)):
        if owed(record):
            name = str(recorded_ref(record).get("connection") or "")
            counts.setdefault(name, {"runs": 0, "starts": 0})["starts"] += 1
    return counts


# --- Panic's own request: best effort, unawaited -----------------------------------------------------------

def panic_stop_work(data_dir: Any) -> Dict[str, Any]:
    """Panic's remote step, on Panic's own request thread: the durable intent, then one best-effort stop per open
    remote run, remote start without a run id and remote model operation, each through its recorded engine's forward
    if one is still open (none is opened, nothing is awaited). The census ends those forwards with every local
    process at the same time, so a stop may be cut on the way: it stays owed and the next start delivers it."""
    if not remote_configured(data_dir):
        return {"requested": True, "remote_engines": 0}
    root, targets, recorded = pathlib.Path(data_dir), [], {}
    try:
        recorded = record_owed_stops(root)
    except Exception:
        log.warning("Panic could not record remote stop intents; the next boot records them", exc_info=True)
    try:
        from ouroboros import delegate_custody as custody
        from ouroboros.delegate_custody_current import open_held_runs
        from ouroboros.execution_domain import is_remote_ref, recorded_ref

        targets += [("run", row.run_id, dict(row.engine_ref)) for row in open_held_runs(root).values()
                    if row.run_id and not row.settled and is_remote_ref(row.engine_ref)]
        targets += [("start", str(record["invocation_id"]), dict(recorded_ref(record)))
                    for record in custody.pending_invocations(root) if is_remote_ref(recorded_ref(record))]
    except Exception:
        log.warning("Panic could not read delegated custody for remote stops", exc_info=True)
    try:
        targets += _open_remote_operations(root)
    except Exception:
        log.warning("Panic could not read model operations for remote stops", exc_info=True)
    for target in targets:
        threading.Thread(target=_best_effort_stop, args=target, name=f"panic-remote-{target[0]}", daemon=True).start()
    return {"requested": True, "owed": recorded, "remote_stops_sent": len(targets)}


def _best_effort_stop(kind: str, identity: str, ref: dict, *, attach: bool = False,
                      timeout_sec: float = _PANIC_STOP_SEC) -> str:
    """One stop request on ``ref``'s engine; what it proves is read later, never here. Returns its outcome word."""
    from ouroboros.remote_engines import bound_gateway

    try:
        client = bound_gateway(ref, timeout_sec=timeout_sec, attach=attach)
        try:
            if kind == "run":
                client.cancel_run(identity, reason="owner_panic")
            elif kind == "start":
                for member in client.task_group_members(task_group_id(identity), expected_store=ref.get("store"))[0]:
                    client.cancel_run(member, reason="owner_panic")
            else:
                client.cancel_model_operation(identity, reason_code="owner_panic")
        finally:
            client.close()
        return "requested"
    except Exception as exc:
        return str(getattr(exc, "code", "") or type(exc).__name__)


def _open_remote_operations(root: Any) -> List[Tuple[str, str, dict]]:
    """``("operation", id, engine_ref)`` of each unsettled model attempt whose request manifest names a remote engine
    and an operation id (a create whose answer was lost names none: no route finds it, so it stays unresolved).

    A stop path's read (Panic's gate among them), so it never imports or creates the money store: an install with
    neither the store nor its journal never had money (``store_switch``'s rule) and holds no operation; a journal
    still waiting for its import (this read passes ``migrate=False``), contention or a damaged store raise: unknown,
    never none."""
    from ouroboros import usage_store
    from ouroboros.execution_domain import is_remote_ref
    from ouroboros.observability import call_manifest_path
    from ouroboros.usage_ledger import LEDGER_REL
    from ouroboros.utils import read_json_dict

    root = pathlib.Path(root)
    if not (root / usage_store.STORE_REL).exists() and not (root / LEDGER_REL).exists():
        return []
    with usage_store.hold(root, write=False, migrate=False) as txn:
        rows = usage_store.decode_attempts(txn.attempt_rows(usage_store._OPEN_PREDICATE))
    out = []
    for row in rows:
        if row.get("provider") != "claudexor":
            continue
        request = read_json_dict(call_manifest_path(root, str(row.get("task_id") or ""),
                                                    f"{row.get('attempt_id')}_model_request")) or {}
        ref, operation = request.get("engine_ref"), str(request.get("operation_id") or "")
        if is_remote_ref(ref) and operation:
            out.append(("operation", operation, dict(ref)))
    return out


def cancel_open_model_operations(root: Any) -> Dict[str, str]:
    """The boot after a Panic (a launcher's Panic sent none): one best-effort cancel per open remote model operation
    on its recorded engine, attaching when needed; the attempt row stays the durable fact its recovery settles."""
    if not remote_configured(root):
        return {}
    return {identity: _best_effort_stop(kind, identity, ref, attach=True, timeout_sec=30.0)
            for kind, identity, ref in _open_remote_operations(root)}


# --- the gates before a task has its work again ----------------------------------------------------------------

def _root(record: Any) -> pathlib.Path:
    from ouroboros import config

    held = getattr(record, "ledger_root", "") or (record.get("ledger_root") if isinstance(record, Mapping) else "")
    return pathlib.Path(str(held or config.DATA_DIR))


def discharged(record: Any, gateway: Any) -> Any:
    """``gateway`` once the run ``record`` names owes no Emergency Stop: an owed stop is carried out and proven
    terminal first (``cancel_and_verify``), so the caller reads a stopped run, never a live one it would adopt. An
    unproven stop closes ``gateway`` and refuses (``remote_stop_owed``); the intent stays for the sweep."""
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_custody_current import run as current_run
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    def field(name: str) -> str:
        return str(getattr(record, name, "") or (record.get(name) if isinstance(record, Mapping) else "") or "")

    run_id, root = field("run_id"), _root(record)
    if not run_id:
        try:  # a pending start (a review panel's rejoin, a retry) is never re-sent while a stop is owed
            refuse_owed_start(root, field("invocation_id"))
        except Exception:
            gateway.close()
            raise
        return gateway
    try:
        durable = current_run(root, run_id)
    except Exception as exc:
        gateway.close()
        raise ClaudexorUnavailable("remote_stop_unreadable", "Whether Emergency Stop still owes this remote run a "
                                   f"stop could not be read ({type(exc).__name__}); it is not resumed.") from exc
    if durable is None or not owed(durable):
        return gateway
    try:
        outcome = custody.cancel_and_verify(root, gateway, durable, "owner_panic")
    except Exception as exc:
        gateway.close()
        raise ClaudexorUnavailable("remote_stop_owed", "Emergency Stop's stop of this remote run could not be "
                                   f"carried out ({type(exc).__name__}); it is not resumed.") from exc
    if outcome["outcome"] != custody.CANCEL_CONFIRMED:
        custody.record_containment_fault(root, durable, PANIC_STOP_UNCONFIRMED,
                                         f"Emergency Stop's stop is still unconfirmed: {outcome['outcome']}",
                                         engine_ref=dict(durable.engine_ref or {}))
        gateway.close()
        raise ClaudexorUnavailable("remote_stop_owed", "Emergency Stop asked this remote run to stop and its engine "
                                   f"has not confirmed it ({outcome['outcome']}); it is not resumed.")
    return gateway


def owed_start(root: Any, invocation_id: str) -> Optional[Dict[str, Any]]:
    """The pending invocation ``invocation_id`` when it owes Emergency Stop a stop, else ``None``."""
    from ouroboros import delegate_custody as custody

    for record in custody.pending_invocations(root):
        if str(record.get("invocation_id") or "") == str(invocation_id) and owed(record):
            return record
    return None


def refuse_owed_start(root: Any, invocation_id: str) -> None:
    """A retry may replay a pending start only when Emergency Stop owes it nothing: an owed start is looked for by
    the sweep on its own engine's store and is never sent again."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    if invocation_id and owed_start(root, invocation_id) is not None:
        raise ClaudexorUnavailable("remote_stop_owed", "Emergency Stop owes this remote start a stop: it is looked up "
                                   "on its own engine and stopped there, never sent again; nothing was started.")


# --- an unknown start, looked for again without a POST -----------------------------------------------------------

def unsent(record: Mapping[str, Any]) -> bool:
    """Whether a remote start's own facts prove its POST never left: it recorded ``staging`` and not ``posting``
    (``REMOTE_START_PHASE``). Without a phase, or past ``staging``, nothing is proven."""
    return str(record.get("remote_start_phase") or "") == "staging"


def resolve_unsent_start(drive_root: Any, record: Mapping[str, Any]) -> Dict[str, Any]:
    """An owed remote start whose POST never left this computer (``unsent``: Emergency Stop came during staging or
    the steps before the POST): definitely not started, so nothing is owed on any engine. Closed without contacting
    one; a copy staging may have left on the host is named, not assumed absent."""
    from ouroboros import delegate_custody as custody
    from ouroboros.execution_domain import recorded_ref
    from ouroboros.remote_workspace import run_folder

    invocation, task = str(record["invocation_id"]), str(record.get("task_id") or "")
    ref = recorded_ref(record)
    try:
        folder = run_folder(str(ref.get("remote_root") or ""), invocation) if not ref.get("workspace") else ""
    except Exception:
        folder = ""
    result = {"invocation_id": invocation, "task_id": task, "action": "invocation_closed_unsent",
              "reason": "remote_start_not_sent"}
    custody.emit(drive_root, custody.START_FAILED, {
        "run_id": "", "task_id": task, "project_id": str(record.get("project_id") or ""), "invocation_id": invocation,
        "definite": True, "reason": "remote_start_not_sent",
        "basis": "no record that its POST was about to leave this computer; Emergency Stop came first",
        **({"remote_folder": folder} if folder else {})})
    custody.emit(drive_root, custody.RECONCILED, result)
    return result


def lookup_due(record: Mapping[str, Any]) -> bool:
    """Whether this pass looks for the sent-unknown start ``record``: the first pass of each process, then once
    ``_RELOOK_SEC`` has passed since the last pass that asked. Asking marks the time whatever the answer (found,
    empty, or an engine the sweep could not reach), so the pace is bounded and never stops: no open tunnel is
    required, the sweep attaches to the recorded engine as it does for an owed run."""
    key, now = str(record.get("invocation_id") or ""), time.monotonic()
    last = _LOOKED_UP.get(key)
    if last is not None and now - last < _RELOOK_SEC:
        return False
    _LOOKED_UP[key] = now
    return True


def find_started_runs(gateway: Any, record: Mapping[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    """``(identities, facts)``: the runs and queued jobs the start ``record`` produced that ``gateway``'s store (the
    one its invocation recorded) lists now: its task group's members and, for a staged copy, the runs anchored at its
    run folder. ``facts`` says what the lookup could not see (members named by neither a run nor a job id, a scan cut
    at its page bound): the engine lists only its currently served generations, so an empty answer is never proof
    that nothing started. Raises the engine's typed refusal."""
    from ouroboros.execution_domain import recorded_ref
    from ouroboros.remote_workspace import run_folder

    ref, invocation = recorded_ref(record), str(record.get("invocation_id") or "")
    found: List[str] = []
    facts: Dict[str, Any] = {"task_group": "unsupported"}
    if gateway.serves(*TASK_GROUPS_ROUTE):
        members, group = gateway.task_group_members(task_group_id(invocation), expected_store=ref.get("store"))
        found += members
        facts = {"task_group": "read", **group}
    if not ref.get("workspace") and ref.get("remote_root"):
        folder = run_folder(str(ref["remote_root"]), invocation).rstrip("/") + "/work"
        summaries = gateway.list_runs(pages=_RUN_SCAN_PAGES)
        for summary in summaries:
            roots = {str((summary.get("project") or {}).get("root") or ""), str(summary.get("executionRoot") or "")}
            if folder in roots and (summary.get("runId") or summary.get("jobId")):
                found.append(str(summary.get("runId") or summary.get("jobId")))
        facts["run_folder_scan"] = "bounded" if len(summaries) >= _RUN_SCAN_PAGES * 200 else "read"
    return list(dict.fromkeys(found)), facts


def locate_started(gateway: Any, record: Mapping[str, Any]) -> Any:
    """The accepted-start read of a remote start (``delegate_accepted.locate``, for every reader: a sweep, an explicit
    retry, a review seat's rejoin): its runs on the store its invocation recorded (``find_started_runs``), never the
    body this computer recorded. The first found is the run its reader binds once (``delegate_accepted.recover``:
    nothing is cancelled, replayed or sent; a start Emergency Stop owes a stop is ``resolve_owed_start``'s); any other
    is named. Nothing found is ``not_found``, no proof that nothing started; another store answers for nothing."""
    import json

    from ouroboros.delegate_accepted import Lookup
    from ouroboros.execution_domain import recorded_ref
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    ref = recorded_ref(record) or dict(getattr(gateway, "engine_ref", None) or {})
    store = str(getattr(gateway, "store_identity", "") or "")
    belongs = str(record.get("daemon_store_id") or "") or str(ref.get("store") or "")
    if belongs and store != belongs:
        return Lookup("store_mismatch", store_id=store, detail=f"store {store or 'unproven'} answered, but this start "
                      f"belongs to store {belongs}")
    try:
        runs, facts = find_started_runs(gateway, {**record, "engine_ref": ref})
    except ClaudexorUnavailable as exc:
        return Lookup("store_mismatch" if exc.code == "remote_engine_changed" else "unavailable",
                      detail=f"{exc.code}: {exc}"[:500])
    if not runs:
        return Lookup("not_found", store_id=store, detail=f"its task group and run folder list nothing now: "
                      f"{json.dumps(facts, sort_keys=True)}")
    return Lookup("found", store_id=store, run_id=runs[0],
                  detail=f"also listed for it: {', '.join(runs[1:])}" if runs[1:] else "")


def resolve_owed_start(drive_root: Any, gateway: Any, record: Dict[str, Any], *,
                       deliberate_terminal: str = "") -> List[Dict[str, Any]]:
    """The sweep's duty for a remote start Emergency Stop owes a stop: never a POST, never a closure from absence. A
    start whose own phase facts prove its POST never left is closed as not sent (``resolve_unsent_start``). Each run
    or queued job found on its store gets its own durable owed stop first; the first is bound to the invocation (its
    row then carries the stop, bound where every STARTED writer of an invocation converges:
    ``delegate_accepted.bind_started``) and every one is cancelled and read back by ``_reconcile_one`` now and on every
    later pass until terminal (one outcome each). Finding nothing keeps the start owed and visible, to be looked for
    again when due (``lookup_due``)."""
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_accepted import bind_started
    from ouroboros.delegate_custody_reconcile import _mark_root_cost, _reconcile_one, recovered_custody
    from ouroboros.delegate_pending import request_body
    from ouroboros.execution_domain import recorded_ref
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    invocation, task = str(record["invocation_id"]), str(record.get("task_id") or "")
    if unsent(record):
        return [resolve_unsent_start(drive_root, record)]
    ref = recorded_ref(record)
    base = {"invocation_id": invocation, "task_id": task}
    if ref.get("store") and str(getattr(gateway, "store_identity", "") or "") != str(ref["store"]):
        return [{**base, "action": "invocation_retained", "reason": "remote_engine_changed"}]
    try:
        runs, facts = find_started_runs(gateway, record)
    except ClaudexorUnavailable as exc:
        return [{**base, "action": "invocation_retained", "reason": f"remote_lookup_{exc.code}"}]
    if not runs:
        return [{**base, "action": "invocation_retained", "reason": "remote_start_not_found", "lookup": facts}]
    for run_id in runs:  # the intent rides each run's own row before anything binds or cancels it
        custody.emit(drive_root, custody.CONTAINMENT_FAULT, {
            "run_id": run_id, "invocation_id": invocation, "task_id": task, "reason": PANIC_STOP_UNCONFIRMED,
            "detail": _detail(ref), "engine_ref": dict(ref)})
    body = record.get("request") if isinstance(record.get("request"), dict) and record.get("request") else (
        request_body(drive_root, record))
    held = None
    if isinstance(body, dict) and body:
        entry, shape = recovered_custody(drive_root, gateway, {**record, "request": body}, runs[0],
                                         str(getattr(gateway, "store_identity", "") or ""))
        held = bind_started(drive_root, entry, shape={**shape, "recovered_after_emergency_stop": True}, exclusive=True)
    if held is None or held.custody is None or not held.durable:
        # The stop rides each found run's own row above; the sweep stops those rows whoever owns them.
        return [{**base, "action": "invocation_retained", "lookup": facts,
                 "reason": "accepted_start_unbound" if held is not None else "invocation_request_unrecorded"}]
    if not held.adopted:
        _mark_root_cost(drive_root, held.custody)
    bound = held.custody
    bound.containment_fault = {"reason": PANIC_STOP_UNCONFIRMED}
    outcomes = [{**_reconcile_one(drive_root, gateway, bound, deliberate_terminal=deliberate_terminal),
                 "invocation_id": invocation, "lookup": facts}]
    from ouroboros.delegate_custody_current import run as current_run

    for extra in runs[1:]:  # ours too (one group, one invocation): each is stopped on its own row, never adopted
        row = current_run(drive_root, extra)
        if row is not None:
            outcomes.append({**_reconcile_one(drive_root, gateway, row, deliberate_terminal=deliberate_terminal),
                             "invocation_id": invocation, "stray_of_invocation": True})
    return outcomes


__all__ = ["PANIC_STOP_UNCONFIRMED", "TASK_GROUPS_ROUTE", "TUNNEL_HOME", "cancel_open_model_operations", "discharged",
           "find_started_runs", "incident_after", "locate_started", "lookup_due", "owed", "owed_start",
           "owed_stop_counts", "panic_stop_work", "record_owed_stops", "refuse_owed_start", "remote_configured",
           "resolve_owed_start", "resolve_unsent_start", "task_group_id", "unsent"]
