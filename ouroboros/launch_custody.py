"""One selection of what service launches exist and what each still owes, for every custody consumer.

A launch's debt lives in three stores that lag each other by design: its drive's ledger, the installation's ownership
document, and the pending copies a stop writes when it cannot take the ledger lock. No store is the whole, so every
consumer combines them by LAUNCH before it filters by owner: a stale copy cannot hide what the ledger names, and a
compacted ledger cannot hide what only a copy names. A store that cannot be read is reported, never read as empty. No
new store: this reads the two that exist, and it neither signals nor reads a process.

Kept services (``start_service(keep_alive=True)``, a task's ``service_teardown=keep``) are one such selection: the
session-scope service launches a task left running, read here for every reader (Activity › Background, the Runtime
section, ``stop_service(service_id=...)``), whatever worker started them, and stopped through the one owned stop.
"""

from __future__ import annotations

import logging
import pathlib
import re
import time
from typing import Any, Dict, List, NamedTuple, Optional

try:
    from typing import TypedDict
except ImportError:  # pragma: no cover - Python 3.10 compatibility.
    from typing_extensions import TypedDict

from ouroboros import process_custody as pc

log = logging.getLogger(__name__)


class LaunchCustody(NamedTuple):
    readable: bool  # the ledger of the drive asked about
    rows: List[Dict[str, Any]]  # its latest row per key, then every launch only another store or drive names
    copies: Dict[int, List[Dict[str, Any]]]  # {id(row): what the launch's ownership copy names}
    unknown: set  # id(row) of each launch whose stores could not be read whole
    blind: str  # why a launch no row here names may exist (the set unreadable, a damaged copy nothing attributes)
    previous: bytes  # the ledger bytes its rows came from, for that ledger's compaction
    records: List[Dict[str, Any]]  # the ownership set as read


def read_custody(drive_root: Any, *, strict: bool = True, every_drive: bool = False) -> LaunchCustody:
    """A launch's row is its own drive's latest (one read per drive) or, once that is gone, its ownership copy's; what
    the copy names beside it comes with it (``process_custody._launch_target``). Rows are as stored, so a sweep's
    survivors still match its ledger. Without ``every_drive`` a launch of another drive stays out: a sweep compacts
    only this drive's ledger. What cannot be read makes its launch ``unknown``; what no row can attribute (a damaged
    copy of a launch nothing else names, an unreadable set) makes the selection ``blind`` for every owner."""
    from ouroboros.owned_shutdown import _ledgered_launch, owned_records

    here = pathlib.Path(drive_root).resolve(strict=False)
    readable, stored, previous = pc._read_ledger_records(pathlib.Path(drive_root), strict=strict)
    rows = list(stored)
    ledgers = {here: (readable, {pc.custody_key(row): row for row in rows})}
    copies: Dict[int, List[Dict[str, Any]]] = {}
    unknown: set = set()
    blind = ""
    records: List[Dict[str, Any]] = []
    damaged: set = set()
    try:
        records = owned_records(drive_root, strict=True, only=set(), damaged=damaged)
    except Exception as exc:
        log.warning("ownership set of %s could not be read: %s", drive_root, exc)
        blind = "the ownership set cannot be read: it may name more launches"
        unknown.update(id(row) for row in rows if pc.custody_key(row)[0] == "launch")
    for record in records:
        if not _ledgered_launch(record):
            continue
        copy = record["ledger_entry"]
        key = pc.custody_key(copy)
        local = key in ledgers[here][1]
        home = here if local else pathlib.Path(record.get("drive_root") or here).resolve(strict=False)
        if home != here and not every_drive:
            continue
        if home not in ledgers:
            found, other, _bytes = pc._read_ledger_records(home, strict=strict)
            ledgers[home] = (found, {pc.custody_key(row): row for row in other})
        found, by_key = ledgers[home]
        row = by_key.get(key, copy)
        if not local:
            rows.append(row)
        copies[id(row)] = pc._copy_names(copy)
        if not found:
            unknown.add(id(row))
    launches = {pc.custody_record_id(row): row for row in rows if pc.custody_key(row)[0] == "launch"}
    for record_id in damaged:
        if record_id in launches:
            unknown.add(id(launches[record_id]))
        elif record_id.startswith("launch-"):
            blind = f"a pending ownership copy ({record_id}) cannot be read, and no other store says whose launch it is"
    return LaunchCustody(readable, rows, copies, unknown, blind, previous, records)


def unsettled_rows(drive_root: Any, owners: Any) -> List[tuple]:
    """What ``owners``' launches and older rows may still run: ``(row, "live" | "unknown")``. Read only, never a stop.
    Raises when the drive's ledger cannot be read, or when nothing else answers and a launch may exist that no row
    attributes: unavailable is never "none"."""
    custody = read_custody(drive_root, every_drive=True)
    if not custody.readable:
        raise OSError("process_custody_unreadable")
    found = unsettled_selection(drive_root, [row for row in custody.rows if str(row.get("owner_task") or "") in owners],
                                custody)
    if custody.blind and not found:
        raise OSError(f"process_custody_unknown: {custody.blind}")
    return found


def unsettled_selection(drive_root: Any, entries: List[Dict[str, Any]], custody: LaunchCustody) -> List[tuple]:
    """The rows of a selection whose processes may still run. A launch row is judged by its whole recorded reach in ONE
    census for all of them (``process_scope.observe_owned_processes``): an owed or marker-only member keeps it live
    after its leader died, and a reused leader PID is not the launch. What could not be read leaves no launch settled.
    An older row keeps its leader PID or surviving group."""
    from ouroboros.platform_layer import pid_is_alive as alive  # the readers' own call-time lookup, as before
    from ouroboros.process_scope import observe_owned_processes

    launches = [entry for entry in entries if pc._owned_service_row(entry)]
    states: Dict[Any, str] = {}
    if launches:
        states = observe_owned_processes([pc._launch_target(entry, custody.copies.get(id(entry))) for entry in launches],
                                         spared=pc.spared_daemon_roots(drive_root))
        states = {key: "unknown" if state == "settled" and key in custody.unknown else state
                  for key, state in states.items()}
    found = []
    for entry in entries:
        if id(entry) in states:
            if states[id(entry)] != "settled":
                found.append((entry, states[id(entry)]))
        elif alive(int(entry.get("pid") or 0)) or pc._service_group_survives_leader(entry):
            found.append((entry, "live"))
    return found


def name_ledger_debt(root: Any, timeout_sec: float) -> Dict[str, Any]:
    """Debt only the canonical ledger names (its ownership writes failed, or its row landed after its record retired)
    joins the set, so an exit's one stop ends it and its report counts it. One read; normally nothing to add. Returns
    what the exit cannot count, empty otherwise: ``records``, launches whose copy no store took either (their ledger
    rows keep them for the next start's sweep), and ``error``, why the selection could not be read whole."""
    from ouroboros.owned_shutdown import record_ledgered_process

    unwritten: List[str] = []
    try:
        custody = read_custody(root)
        named = {record.get("record_id") for record in custody.records}
        for row in [] if custody.blind else custody.rows:
            if row.get("known_members") and pc._owned_service_row(row) and pc.custody_record_id(row) not in named:
                if not record_ledgered_process(root, row, timeout_sec=timeout_sec):
                    unwritten.append(pc.custody_record_id(row))
                timeout_sec = 0.0  # one wait for the lock per exit: the rest go straight to their pending copy
        error = custody.blind or ("" if custody.readable else "the canonical process ledger cannot be read")
    except Exception as exc:
        log.warning("Debt only the ledger names was not read at this exit; the next start's sweep ends it", exc_info=True)
        error = f"{type(exc).__name__}: {exc}"[:200]
    return {key: value for key, value in (("records", unwritten), ("error", error)) if value}



# --------------------------------------------------------------------------- kept services
_KEPT_PURPOSES = ("service:", "workspace_service:")
_LAUNCH_ID = re.compile(r"^(launch-[0-9a-f]{32}|pid-[0-9]{1,10}|service-[0-9a-f]{32})$")
_ADDRESS = re.compile(rb"https?://(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1?\]):[0-9]{2,5}")
_ADDRESS_SCAN_BYTES = 64 * 1024
_STOP_SEC = 10.0  # one owner/agent stop's bound; what is not confirmed by then stays stamped for the next start
STOP_OUTCOMES = ("stopped", "not_confirmed", "unknown", "not_found", "refused")
REASON_MAX_CHARS = 1000  # a stop's reason is recorded whole in its event row; a longer one is refused, never cut


class KeptService(TypedDict, total=False):
    """One service a task left running on purpose, as custody knows it now (not the task's ``services_kept`` history).
    ``launch_id``: its durable launch (the ownership set's id), the one a stop names. ``state``: ``running`` (its
    recorded reach still has a live member), ``unknown`` (it could not be read whole) or ``capture_owed`` (it ended,
    but its declared outputs still owe a capture: a stop of it signals nothing and captures them); an ended launch
    that owes nothing is not listed.
    ``since``: when its launch was recorded. ``address``: the first loopback URL its own log printed (scheme, host and
    port only), ``""`` when none or the log is not on this host. ``log_path``: its live log, kept by a stop."""

    launch_id: str
    name: str
    task_id: str
    task_title: str
    state: str
    since: str
    address: str
    log_path: str
    backend: str
    pid: int


class KeptServicesResponse(TypedDict):
    """GET /api/services/kept. ``complete`` is False when a custody store could not be read whole: the list may then
    miss a service, and ``note`` says why (an empty incomplete list is unknown, never "none")."""

    services: List[KeptService]
    complete: bool
    note: str


class KeptServiceStopRequest(TypedDict):
    """POST /api/services/kept/{launch_id}/stop: why it is stopped; recorded with the outcome."""

    reason: str


class KeptServiceStopResponse(TypedDict, total=False):
    """The receipt of one stop of one kept launch. ``outcome``: ``stopped`` (a fresh read after the stop finds none of
    its processes), ``not_confirmed`` (some are still alive or unverified; the stop stays recorded and the next start
    retries it), ``unknown`` (custody could not be read after the stop), ``not_found`` (no kept service with that id
    runs, custody read whole; nothing was signalled), ``unknown`` too when it is not listed but custody could not be
    read whole (nothing was signalled either) or ``refused`` (no reason, a reason over ``REASON_MAX_CHARS``, an
    invalid id, or not this caller's to stop: a root task stops only a finished task's launch). Its log stays where it
    is (``log_path``) until retention archives it under the task that started it. ``outputs``
    (``tools.kept_service_outputs.capture_kept_service_outputs``): once its stop is confirmed, its declared outputs are
    captured into the starting task's artifact store, never the stopper's: ``state`` ``captured``, ``incomplete``
    (its declaration stays: a later stop of the same id retries the capture), ``none_declared``, ``unavailable`` (no
    declaration was recorded: nothing captured) or ``not_attempted`` (its stop is not confirmed; its declaration
    stays). A ``not_found`` launch whose declaration is still unconsumed (an earlier capture was incomplete or not
    attempted, or it ended by itself) has it captured then. ``ok`` is True only when the launch is confirmed ended and
    its outputs are ``captured`` or ``none_declared``."""

    ok: bool
    outcome: str
    launch_id: str
    name: str
    task_id: str
    actor: str
    reason: str
    detail: str
    log_path: str
    stopped_at: str
    outputs: Dict[str, Any]


def _kept_row(row: Dict[str, Any]) -> bool:
    return str(row.get("purpose") or "").startswith(_KEPT_PURPOSES) and row.get("scope") == "session"


def _log_address(path: str) -> str:
    """The first loopback URL a dev server printed at its start, or ''."""
    try:
        with open(path, "rb") as handle:
            found = _ADDRESS.search(handle.read(_ADDRESS_SCAN_BYTES))
        return found.group(0).decode("ascii") if found else ""
    except OSError:
        return ""


def _host_log(drive: pathlib.Path, task_id: str, name: str, executor: bool = False) -> str:
    if not task_id or pathlib.Path(task_id).name != task_id or not name or pathlib.Path(name).name != name:
        return ""
    return str(drive / "services" / task_id / f"{name}{'.executor' if executor else ''}.log")


def _durable_launch(stored: Dict[str, Any]) -> bool:
    """A local executor service record naming its launch by a durable scope marker, on every platform (the POSIX
    measured stop's ``workspace_executor._recorded_launch`` is a different question): that launch's id joins its ledger row."""
    from ouroboros.process_census import is_launch_marker

    return stored.get("executor_type") == "local" and is_launch_marker(str(stored.get("scope_marker") or ""))


def _kept_selection(drive_root: Any) -> tuple:
    """``([(item, stop entries)], complete, note)``: every kept launch whose processes may still run, each with the
    ownership-set records a stop of it acts on (a ledger row's record, a local executor's record of the same launch,
    a Docker executor record). One custody read, one census for every launch row; a Docker backend is probed."""
    from ouroboros import workspace_executor as executor
    from ouroboros.owned_shutdown import installation_root
    from ouroboros.process_custody import custody_key, custody_record_id, ledger_path

    root = installation_root(drive_root)
    custody = read_custody(root, every_drive=True)
    notes = [note for note in (custody.blind, "" if custody.readable else "the process ledger cannot be read") if note]
    drives = {custody_key(record["ledger_entry"]): pathlib.Path(record.get("drive_root") or root)
              for record in custody.records if isinstance(record.get("ledger_entry"), dict)}
    by_id = {str(record.get("record_id") or ""): record for record in custody.records}
    rows = [row for row in custody.rows if _kept_row(row)]
    row_keys = {custody_key(row) for row in rows}
    attached: Dict[tuple, Dict[str, Any]] = {}
    docker: List[tuple] = []
    for record in custody.records:
        if record.get("kind") != "service":
            continue
        path = pathlib.Path(str(record.get("record_path") or ""))
        stored = executor._load_process_record(path)
        if stored is None or not stored.get("keep_alive"):
            continue
        if _durable_launch(stored):  # its ledger row and this record are one launch, on every platform
            launch = executor._launch_row(stored)
            if custody_key(launch) not in row_keys:  # the ledger lost its row: the record's own copy stands in
                rows.append(launch)
                row_keys.add(custody_key(launch))
                drives[custody_key(launch)] = path.parents[2]
            attached[custody_key(launch)] = {**record, "_stored": stored}
        else:
            docker.append((record, stored))
    states = dict((id(row), state) for row, state in unsettled_selection(root, rows, custody)) if rows else {}
    found: List[tuple] = []
    for row in rows:
        if id(row) not in states:
            continue  # settled: nothing it reaches is alive
        key = custody_key(row)
        drive = drives.get(key, root)
        rid = custody_record_id(row)
        purpose = str(row.get("purpose") or "")
        name = purpose.split(":", 1)[1]
        task_id = str(row.get("owner_task") or "")
        record = by_id.get(rid) or {"record_id": rid, "kind": "supervised", "drive_root": str(drive),
                                    "record_path": str(ledger_path(drive)), "ledger_entry": row,
                                    "owner_task": task_id, "scope": "session", "purpose": purpose}
        entries = [record]
        local = attached.get(key)
        if local is not None:
            entries.append({key_: value for key_, value in local.items() if key_ != "_stored"})
        log_path = (str((local or {}).get("_stored", {}).get("backend_log_path") or "") if local
                    else _host_log(drive, task_id, name))
        found.append(({"launch_id": rid, "name": name, "task_id": task_id,
                       "state": "running" if states[id(row)] == "live" else "unknown",
                       "since": str(row.get("ts") or (local or {}).get("birth") or ""),
                       "address": _log_address(log_path) if log_path else "", "log_path": log_path,
                       "backend": "local_executor" if local or purpose.startswith("workspace_service:") else "host",
                       "pid": int(row.get("pid") or 0)}, entries))
    for record, stored in docker:
        if stored.get("executor_type") == "local":
            alive = executor._host_pid_matches_record(stored)
            state = "running" if alive else ("" if _exit_proven(int(stored.get("host_pid") or 0)) else "unknown")
        else:
            probed = executor._docker_pid_state(str(stored.get("container_name") or ""), str(stored.get("backend_pid") or ""))
            state = {"running": "running", "exited": ""}.get(probed, "unknown")
        if not state:
            continue
        found.append(({"launch_id": str(record.get("record_id") or ""), "name": str(stored.get("name") or ""),
                       "task_id": str(stored.get("task_id") or ""), "state": state,
                       "since": str(stored.get("created_at") or ""), "address": "", "log_path": "",
                       "backend": "docker" if stored.get("executor_type") == "docker_exec" else "local_executor",
                       "pid": int(stored.get("host_pid") or 0)}, [record]))
    return found, not notes, "; ".join(notes)


def _exit_proven(pid: int) -> bool:
    from ouroboros.owned_shutdown import _exit_proven as proven

    return proven(pid)


def _owed_ended(drive_root: Any, running: set, complete: bool) -> List[KeptService]:
    """Ended kept launches whose declared outputs still owe a capture (``kept_service_outputs.outstanding_declarations``):
    ``capture_owed``, or ``unknown`` while custody could not be read whole (it may still run)."""
    from ouroboros.owned_shutdown import installation_root
    from ouroboros.tools.kept_service_outputs import KEPT_DECLARATION, outstanding_declarations

    owed: List[KeptService] = []
    for path, declaration in outstanding_declarations(installation_root(drive_root)):
        facts = declaration if isinstance(declaration, dict) else {}
        launch_id = str(facts.get("launch_id") or path.name[:-len(KEPT_DECLARATION)])
        if launch_id in running:
            continue
        owed.append({"launch_id": launch_id, "name": str(facts.get("name") or ""),
                     "task_id": str(facts.get("task_id") or path.parent.name),
                     "state": "capture_owed" if complete else "unknown", "since": str(facts.get("declared_at") or ""),
                     "address": "", "log_path": "", "backend": "", "pid": 0})
    return owed


def kept_services(drive_root: Any) -> KeptServicesResponse:
    """What kept services run now, and the ended ones whose declared outputs still owe a capture, for every reader;
    never a stop. An unreadable store is said, never an empty list."""
    try:
        found, complete, note = _kept_selection(drive_root)
        services = [item for item, _entries in found]
        services += _owed_ended(drive_root, {item["launch_id"] for item in services}, complete)
    except Exception as exc:
        log.warning("kept services could not be read", exc_info=True)
        return {"services": [], "complete": False, "note": f"custody could not be read: {type(exc).__name__}"}
    return {"services": services, "complete": complete, "note": note}


def kept_services_fact(drive_root: Any) -> Optional[Dict[str, Any]]:
    """The Runtime section's ``kept_services``: present only when one runs or the selection is incomplete."""
    view = kept_services(drive_root)
    if not view["services"] and view["complete"]:
        return None
    fact: Dict[str, Any] = {"services": [{key: item[key] for key in ("launch_id", "name", "task_id", "state", "since",
                                                                     "address") if item.get(key)}
                                         for item in view["services"]],
                            "stop": "stop_service(service_id=<launch_id>, reason=...) from a root task once the task "
                                    "that started it has finished (a capture_owed one: captures its outputs); the owner: "
                                    "Activity › Background"}
    if not view["complete"]:
        fact["incomplete"] = view["note"]
    return fact


def _producer_unfinished(root: pathlib.Path, task_id: str) -> str:
    """Why the task that started a kept launch is not proven finished, or ``""``. Finished: its effective result
    (retries included) is terminal AND a fresh queue view shows neither its worker nor a helper of its tree still
    running (its result is written before its post-task work ends). Unreadable, stale or absent is not finished."""
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES
    from ouroboros.task_status import (
        load_effective_task_result,
        task_has_live_queue_ownership,
        task_subtree_has_live_queue_rows,
    )

    if not task_id:
        return "the task that started it is not recorded"
    try:
        status = str(load_effective_task_result(root, task_id, materialize_artifacts=False).get("status") or "")
    except Exception as exc:  # noqa: BLE001 — unreadable is not finished
        return f"its task {task_id} cannot be read ({type(exc).__name__})"
    if status not in _TRULY_TERMINAL_STATUSES:
        return f"its task {task_id} has not finished ({status or 'no result recorded'})"
    if task_has_live_queue_ownership(root, task_id) or task_subtree_has_live_queue_rows(root, task_id):
        return (f"its task {task_id} or a helper of it may still run (the queue lists it, or no fresh queue view "
                "proves otherwise)")
    return ""


def stop_kept_service(drive_root: Any, launch_id: str, *, reason: str, actor: str, finished_producer: bool = False,
                      timeout_sec: float = _STOP_SEC) -> KeptServiceStopResponse:
    """Stop ONE kept launch by its durable id: the owner's Activity Stop and a root task's ``stop_service(service_id=)``
    land here. ``finished_producer`` (the root task's door; the owner keeps the broader authority): only a launch whose
    task is proven finished (``_producer_unfinished``) is stopped, else ``refused`` with nothing signalled. The stop is
    the exit's own (``owned_shutdown._stop_records``: measured launch identity, owed members, daemon roots spared),
    stamped first so a crash leaves it for the next start; the outcome is a fresh read after it, never the stop's own
    claim. Its log stays where it is and its declared outputs go to the task that started it; an ended launch whose
    declaration no capture consumed has it captured now. Nothing else is started, restarted or signalled."""
    from ouroboros.owned_shutdown import _stamp, _stop_records, installation_root
    from ouroboros.tools.kept_service_outputs import capture_kept_service_outputs, unconsumed_declaration
    from ouroboros.utils import append_jsonl, utc_now_iso

    reason, launch_id = str(reason or "").strip(), str(launch_id or "").strip()
    receipt: KeptServiceStopResponse = {"ok": False, "launch_id": launch_id, "actor": actor,
                                        "reason": reason if len(reason) <= REASON_MAX_CHARS else "",
                                        "stopped_at": utc_now_iso()}
    if not reason or len(reason) > REASON_MAX_CHARS or not _LAUNCH_ID.fullmatch(launch_id):
        return {**receipt, "outcome": "refused", "detail": (
            "a reason is required" if not reason else f"the reason is longer than {REASON_MAX_CHARS} characters"
            if len(reason) > REASON_MAX_CHARS else "not a kept service launch id")}
    root = installation_root(drive_root)
    try:
        found, complete, note = _kept_selection(root)
        match = next(((item, entries) for item, entries in found if item["launch_id"] == launch_id), None)
        pending = unconsumed_declaration(root, launch_id) if match is None and complete else None
    except Exception as exc:
        return {**receipt, "outcome": "unknown", "detail": f"custody could not be read: {type(exc).__name__}"}
    producer = match[0]["task_id"] if match is not None else pending[1] if pending else ""
    unfinished = _producer_unfinished(root, producer) if finished_producer and (match or pending) else ""
    if unfinished:
        receipt.update(task_id=producer, outcome="refused", detail=(
            f"{unfinished}; a root task stops only a service a finished task left running (its own services: "
            "stop_service(name=...)); nothing was signalled"))
    elif match is None and not complete:  # absent from a partial read is unknown, never "not running"
        receipt.update(outcome="unknown", detail=f"not listed, but custody could not be read whole ({note}); "
                                                 "nothing was signalled")
    elif match is None and pending:  # ended without a complete capture: an earlier stop's, or by itself
        receipt.update(task_id=pending[1], outcome="not_found",
                       outputs=capture_kept_service_outputs(pending[0], pending[1], launch_id, actor=actor,
                                                            reason=reason),
                       detail="no process of this launch is running; nothing was signalled; no capture had consumed "
                              "its output declaration, so its declared outputs were captured now (see outputs)")
        receipt["ok"] = receipt["outputs"]["state"] in ("captured", "none_declared")
    elif match is None:
        receipt.update(outcome="not_found", detail="no kept service with this launch id is running; nothing was signalled")
    else:
        item, entries = match
        receipt.update(name=item["name"], task_id=item["task_id"], log_path=item["log_path"])
        drive = pathlib.Path(entries[0].get("drive_root") or root)
        ids = {entry["record_id"] for entry in entries}
        _stamp(root, "stop_requested_at", lambda entry: entry.get("record_id") in ids)
        _stop_records(root, entries, time.monotonic() + max(0.0, float(timeout_sec)))
        try:
            after, complete, note = _kept_selection(root)
            still = next((left for left, _entries in after if left["launch_id"] == launch_id), None)
        except Exception as exc:
            still, complete, note = None, False, f"custody could not be read: {type(exc).__name__}"
        receipt["outputs"] = {"task_id": item["task_id"], "state": "not_attempted",
                              "detail": "declared outputs are captured once its stop is confirmed; its declaration "
                                        "stays, and a later stop of this launch id captures them once it has ended"}
        if still is not None:
            receipt.update(outcome="not_confirmed", detail=(
                f"its processes are not confirmed ended ({still['state']}); the stop stays recorded and the next "
                "start retries it"))
        elif not complete:
            receipt.update(outcome="unknown", detail=f"after the stop {note}")
        else:
            receipt["outputs"] = capture_kept_service_outputs(drive, item["task_id"], launch_id, actor=actor,
                                                              reason=reason)
            receipt.update(ok=receipt["outputs"]["state"] in ("captured", "none_declared"), outcome="stopped",
                           detail="no process of this launch is running; its log is kept for retention")
    try:
        append_jsonl(root / "logs" / "events.jsonl", {"ts": receipt["stopped_at"], "type": "kept_service_stop",
                                                       **{key: value for key, value in receipt.items() if key != "ok"}})
    except Exception:
        log.warning("kept service stop of %s is not recorded in events", launch_id, exc_info=True)
    return receipt
