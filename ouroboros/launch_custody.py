"""One selection of what service launches exist and what each still owes, for every custody consumer.

A launch's debt lives in three stores that lag each other by design: its drive's ledger, the installation's ownership
document, and the pending copies a stop writes when it cannot take the ledger lock. No store is the whole, so every
consumer combines them by LAUNCH before it filters by owner: a stale copy cannot hide what the ledger names, and a
compacted ledger cannot hide what only a copy names. An unreadable store leaves custody unknown. Local service
diagnostics are born outside disposable child drives, so their retention needs no post-stop file transfer.

Kept services (``start_service(keep_alive=True)``, a task's ``service_teardown=keep``) are one such selection: the
session-scope service launches a task left running, read here for every reader (Activity › Background, the Runtime
section, ``stop_service(service_id=...)``), whatever worker started them, and stopped through the one owned stop.
"""

from __future__ import annotations

import errno
import json
import logging
import pathlib
import re
import time
from os import stat_result
from typing import Any, BinaryIO, Collection, Dict, List, NamedTuple, Optional

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


def read_custody(drive_root: Any, *, strict: bool = True, every_drive: bool = False, launch_id: str = "") -> LaunchCustody:
    """A launch's row is its own drive's latest (one read per drive) or, once that is gone, its ownership copy's; what
    the copy names beside it comes with it (``process_custody._launch_target``). Rows are as stored, so a sweep's
    survivors still match its ledger. Without ``every_drive`` a launch of another drive stays out: a sweep compacts
    only this drive's ledger. What cannot be read makes its launch ``unknown``; what no row can attribute (a damaged
    copy of a launch nothing else names, an unreadable set) makes the selection ``blind`` for every owner."""
    from ouroboros.owned_shutdown import _ledgered_launch, owned_records

    here = pathlib.Path(drive_root).resolve(strict=False)
    readable, stored, previous = pc._read_ledger_records(pathlib.Path(drive_root), strict=strict)
    rows = [row for row in stored if not launch_id or pc.custody_record_id(row) == launch_id]
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
        if launch_id and pc.custody_record_id(copy) != launch_id:
            continue
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
        if launch_id and record_id != launch_id:
            continue
        if record_id in launches:
            unknown.add(id(launches[record_id]))
        elif record_id.startswith("launch-") or (launch_id and record_id.startswith("service-")):
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
    retries it), ``unknown`` (custody could not be read after the stop, or the capture's own read of the launch did
    not confirm that end), ``not_found`` (no kept service with that id
    runs, custody read whole; nothing was signalled), ``unknown`` too when it is not listed but custody could not be
    read whole (nothing was signalled either) or ``refused`` (no reason, a reason over ``REASON_MAX_CHARS``, an
    invalid id, or not this caller's to stop: a root task stops only a finished task's launch). Its log stays where it
    is (``log_path``) until retention archives it under the task that started it. ``outputs``
    (``tools.kept_service_outputs.capture_kept_service_outputs``): once its stop is confirmed, its declared outputs are
    captured into the starting task's artifact store, never the stopper's: ``state`` ``captured``, ``incomplete``
    (its declaration stays: a later stop retries), ``missing`` (failed but proven absent, settled), ``none_declared``, ``unavailable`` (no
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


_STANDARD_LOG_SUFFIX = "@standard.log"
_LAUNCH_LOG_NAME = re.compile(r"([A-Za-z0-9_.-]+)@([0-9a-f]{32})(@standard)?\.log")


def open_service_log(drive: pathlib.Path, task: str, name: str, marker: str, *,
                     needs_value_masking: bool) -> tuple[pathlib.Path, BinaryIO]:
    """Create an installation-owned, launch-specific file before giving stdout to the producer.

    A child drive's disposal never moves this file. The start must not run if private placement fails.
    An archive is masked by the producer and bound to its launch in the installation's store.
    """
    from ouroboros.confined_files import confined_directory, native_name
    from ouroboros.install_identity import require_owned_root
    from ouroboros.maintenance_retirement import artifact_changed
    from ouroboros.process_census import LAUNCH_SCOPE_PREFIX
    from ouroboros.task_custody import canonical_owner_root
    from ouroboros.tool_access_paths import PRIVATE_SERVICE_LOGS

    suffix = ".log" if needs_value_masking else _STANDARD_LOG_SUFFIX
    filename = f"{name}@{marker.removeprefix(LAUNCH_SCOPE_PREFIX)}{suffix}"
    if not native_name(task) or not native_name(filename) or not service_log_launch_id(filename):
        raise OSError(errno.EINVAL, "invalid service log address")
    data_root = canonical_owner_root(require_owned_root(drive))
    task_log_dir = data_root / PRIVATE_SERVICE_LOGS / task
    data_root.mkdir(parents=True, exist_ok=True)
    with confined_directory(data_root) as root, root.private_child(PRIVATE_SERVICE_LOGS, create=True) as store:
        with store.private_child(task, create=True) as task_logs:
            handle = task_logs.create_private_file(filename)
    artifact_changed(data_root)
    return task_log_dir / filename, handle


def open_service_log_source(
    drive: pathlib.Path, path: pathlib.Path, *, closed: bool = False,
) -> tuple[BinaryIO, stat_result]:
    """Open a launch's own log (or a legacy drive-local log); never an arbitrary install file."""
    from ouroboros.confined_files import confined_directory, open_confined_file
    from ouroboros.task_custody import canonical_owner_root
    from ouroboros.tool_access_paths import PRIVATE_SERVICE_LOGS

    data_root = canonical_owner_root(drive)
    if path.parent.parent == data_root / PRIVATE_SERVICE_LOGS:
        with confined_directory(data_root) as root, root.private_child(PRIVATE_SERVICE_LOGS) as logs:
            with logs.private_child(path.parent.name) as task_logs:
                return task_logs.open(path.name, private=True, quiescent=closed)
    parts = path.relative_to(drive).parts
    if len(parts) != 3 or parts[0] != "services" or not path.name.endswith(".log"):
        raise OSError(errno.EINVAL, "not a service log address")
    return open_confined_file(drive, parts, quiescent=closed)


def service_log_allows_standard_masking(filename: str) -> bool:
    """Only the opted-in launch filename. Neither legacy form carries a masking policy."""
    match = _LAUNCH_LOG_NAME.fullmatch(filename)
    return bool(match and match[3])


def service_log_launch_id(filename: str) -> str:
    match = _LAUNCH_LOG_NAME.fullmatch(filename)
    return "launch-" + match[2] if match else ""


def service_log_archive_root(drive_root: pathlib.Path, task_id: str) -> pathlib.Path | None:
    """A receipt outlives a child drive. Only the installation that owns this placement may take it."""
    from ouroboros.install_identity import require_owned_root
    from ouroboros.source_retention import retained_child_root
    from ouroboros.task_custody import canonical_owner_root

    drive = require_owned_root(drive_root)
    canonical = canonical_owner_root(drive)
    return canonical if drive == canonical or retained_child_root(canonical, drive, task_id) else None


def _service_entries(drive: pathlib.Path, *parts: str, retiring: bool = False) -> list[tuple[str, bool]] | None:
    from ouroboros.confined_files import confined_directory

    try:
        with confined_directory(drive, parts) as directory:
            return directory.entries(include_links=retiring)
    except OSError as exc:
        if exc.errno == errno.ENOENT or (not retiring and exc.errno in (errno.ENOTDIR, errno.ELOOP)):
            return None
        raise


def service_log_files(drive: pathlib.Path, *parts: str, retiring: bool = False) -> list[str] | None:
    entries = _service_entries(drive, *parts, retiring=retiring)
    if entries is None:
        return None
    return sorted(name for name, directory in entries if (retiring or not directory) and name.lower().endswith(".log"))


class ServiceLogUncustodied(Exception):
    """The drive must stay: its logs have no confirmed retained disposition."""

    def __init__(self) -> None:
        super().__init__("service_log_uncustodied")


def require_no_legacy_service_logs(drive: pathlib.Path, owners: list[str]) -> None:
    """Old drive-local files are not evidence of a safely retained or closed launch."""
    try:
        if any(service_log_files(drive, "services", task, retiring=True)
               for task, directory in _service_entries(drive, "services", retiring=True) or ()
               if directory or task in owners):
            raise ServiceLogUncustodied
    except OSError as exc:
        raise ServiceLogUncustodied from exc


def _host_log(drive: pathlib.Path, task_id: str, filename: str) -> str:
    if (not task_id or pathlib.Path(task_id).name != task_id or task_id == ".." or not filename.endswith(".log")
            or pathlib.Path(filename).name != filename):
        return ""
    from ouroboros.task_custody import canonical_owner_root
    from ouroboros.tool_access_paths import PRIVATE_SERVICE_LOGS

    if service_log_launch_id(filename):
        canonical = canonical_owner_root(drive) / PRIVATE_SERVICE_LOGS / task_id / filename
        if canonical.is_file():
            return str(canonical)
    return str(drive / "services" / task_id / filename)


def _durable_launch(stored: Dict[str, Any]) -> bool:
    """A local executor service record naming its launch by a durable scope marker, on every platform (the POSIX
    measured stop's ``workspace_executor._recorded_launch`` is a different question): that launch's id joins its ledger row."""
    from ouroboros.process_census import is_launch_marker

    return stored.get("executor_type") == "local" and is_launch_marker(str(stored.get("scope_marker") or ""))


def drive_service_writers_unsettled(drive: pathlib.Path, owners: Collection[str]) -> bool:
    """Retirement probe for one drive, outside the queue lock. Logs are not writer evidence.

    Read this drive's ledger and its copies in the current ownership set, never another drive's ledger.
    Cover an executor's own marker even when its registration in that set failed. Probe only named services.
    """
    from ouroboros import workspace_executor as executor
    from ouroboros.confined_files import confined_directory, native_name
    from ouroboros.platform_layer import pid_provably_gone

    custody = read_custody(drive)
    if not custody.readable or custody.blind:
        return True
    rows = {pc.custody_key(row): row for row in custody.rows
            if str(row.get("purpose") or "").startswith(_KEPT_PURPOSES)}
    executor_parts = ("state", executor._PROCESS_STATE_DIR)
    names = {name for name, _directory in _service_entries(drive, *executor_parts, retiring=True) or ()
             if name.startswith("service-") and name.endswith(".json")}
    for copy in custody.records:
        if pathlib.Path(str(copy.get("drive_root") or "")) != drive:
            continue
        if copy.get("kind") == "service":
            if not native_name(copy.get("record_id")) or copy.get("unconfirmed_since"):
                return True
            names.add(f"{copy['record_id']}.json")
        elif isinstance(row := copy.get("ledger_entry"), dict) and str(row.get("purpose") or "").startswith(_KEPT_PURPOSES):
            if copy.get("unconfirmed_since"):
                return True
            rows.setdefault(pc.custody_key(row), row)
    if names:
        with confined_directory(drive, executor_parts) as folder:
            for name in names:
                with folder.open(name)[0] as source:
                    stored = json.load(source)
                path = drive.joinpath(*executor_parts, name)
                if (not isinstance(stored, dict) or not executor._valid_process_record(path, stored, check_identity=False)
                        or stored.get("record_type") != "service" or stored.get("task_id") not in owners):
                    return True
                if _durable_launch(stored):
                    row = executor._launch_row(stored)
                    rows.setdefault(pc.custody_key(row), row)
                elif stored["executor_type"] == "docker_exec":
                    if executor._docker_pid_state(stored["container_name"], stored["backend_pid"]) != "exited":
                        return True
                elif (pid := int(stored.get("host_pid") or 0)) <= 0 or not pid_provably_gone(pid):
                    return True
    if any(row.get("owner_task") not in owners or id(row) in custody.unknown for row in rows.values()):
        return True
    launches = []
    for row in rows.values():
        if pc._owned_service_row(row):
            launches.append(row)
        elif (pid := int(row.get("pid") or 0)) <= 0 or not pid_provably_gone(pid) or pc._service_group_survives_leader(row):
            return True
    return bool(unsettled_selection(drive, launches, custody))


def _kept_selection(drive_root: Any, *, launch_id: str = "", task_id: str = "") -> tuple:
    """``([(item, stop entries)], complete, note)``: every kept launch whose processes may still run, each with the
    ownership-set records a stop of it acts on (a ledger row's record, a local executor's record of the same launch,
    a Docker executor record). One custody read, one census for every launch row; a Docker backend is probed."""
    from ouroboros import workspace_executor as executor
    from ouroboros.owned_shutdown import installation_root
    from ouroboros.process_custody import custody_key, custody_record_id, ledger_path

    root = installation_root(drive_root)
    custody = read_custody(root, every_drive=True, launch_id=launch_id)
    notes = [note for note in (custody.blind, "" if custody.readable else "the process ledger cannot be read") if note]
    drives = {custody_key(record["ledger_entry"]): pathlib.Path(record.get("drive_root") or root)
              for record in custody.records if isinstance(record.get("ledger_entry"), dict)}
    by_id = {str(record.get("record_id") or ""): record for record in custody.records}
    rows = [row for row in custody.rows if _kept_row(row)]
    row_keys = {custody_key(row) for row in rows}
    leaders = {row.get("pid") for row in rows}
    attached: Dict[tuple, Dict[str, Any]] = {}
    docker: List[tuple] = []
    for record in custody.records:
        if record.get("kind") != "service" or (task_id and record.get("task_id") != task_id):
            continue
        if launch_id and record.get("record_id") != launch_id and (
                record.get("executor_type") != "local" or not launch_id.startswith("launch-")
                or leaders and record.get("host_pid") not in leaders):
            continue
        path = pathlib.Path(str(record.get("record_path") or ""))
        stored = executor._load_process_record(path)
        if stored is None or not executor._valid_process_record(path, stored, check_identity=False):
            notes.append(f"service record {record.get('record_id')} cannot be read")
            continue
        if not stored.get("keep_alive"):
            continue
        if _durable_launch(stored):  # its ledger row and this record are one launch, on every platform
            launch = executor._launch_row(stored)
            if launch_id and custody_record_id(launch) != launch_id:
                continue
            if custody_key(launch) not in row_keys:  # the ledger lost its row: the record's own copy stands in
                rows.append(launch)
                row_keys.add(custody_key(launch))
                drives[custody_key(launch)] = path.parents[2]
            attached[custody_key(launch)] = {**record, "_stored": stored}
        elif not launch_id or record.get("record_id") == launch_id:
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
        legacy_name = f"{name}.executor.log" if purpose.startswith("workspace_service:") else f"{name}.log"
        log_path = (str(local["_stored"].get("backend_log_path") or "") if local else
                    _host_log(drive, task_id, str(row.get("service_log_name") or legacy_name)))
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
    from ouroboros.tools.kept_service_outputs import KEPT_DECLARATION, _confirmed_stop, outstanding_declarations

    owed: List[KeptService] = []
    for path, declaration in outstanding_declarations(installation_root(drive_root)):
        facts = declaration if isinstance(declaration, dict) else {}
        launch_id = str(facts.get("launch_id") or path.name[:-len(KEPT_DECLARATION)])
        if launch_id in running:
            continue
        ended = _confirmed_stop(facts) or (complete and facts.get("keep_alive", True))
        owed.append({"launch_id": launch_id, "name": str(facts.get("name") or ""),
                     "task_id": str(facts.get("task_id") or path.parent.name),
                     "state": "capture_owed" if ended else "unknown", "since": str(facts.get("declared_at") or ""),
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
    from contextlib import ExitStack
    from ouroboros.owned_shutdown import installation_root
    from ouroboros.service_outcomes import public_retained_receipt
    from ouroboros.tools.kept_service_outputs import confirmed_output_stop, service_name_hold, unconsumed_declaration
    from ouroboros.utils import append_jsonl, utc_now_iso

    reason, launch_id = str(reason or "").strip(), str(launch_id or "").strip()
    receipt: KeptServiceStopResponse = {"ok": False, "launch_id": launch_id, "actor": actor,
                                        "reason": reason if len(reason) <= REASON_MAX_CHARS else "",
                                        "stopped_at": utc_now_iso()}
    if not reason or len(reason) > REASON_MAX_CHARS or not _LAUNCH_ID.fullmatch(launch_id):
        return {**receipt, "outcome": "refused", "refusal": ("reason_required" if not reason else
                "reason_too_long" if len(reason) > REASON_MAX_CHARS else "invalid_launch"), "detail": (
            "a reason is required" if not reason else f"the reason is longer than {REASON_MAX_CHARS} characters"
            if len(reason) > REASON_MAX_CHARS else "not a kept service launch id")}
    root = installation_root(drive_root)
    try:
        with ExitStack() as held:
            found, complete, note = _kept_selection(root, launch_id=launch_id)
            if found:
                selected = found[0][0]
                held.enter_context(service_name_hold(root, selected["task_id"], selected["name"]))
                # The name may have changed hands before we acquired it. Select the launch again before any signal.
                found, complete, note = _kept_selection(root, launch_id=launch_id, task_id=selected["task_id"])
            match = found[0] if found else None
            pending = (root, match[0]["task_id"]) if match else unconsumed_declaration(root, launch_id)
            if pending and confirmed_output_stop(*pending, launch_id):
                match, complete = None, True  # this declaration attests the selected launch already ended
            elif match or not complete:
                pending = None
            _stop_selected(root, receipt, match, pending, complete, note, finished_producer, timeout_sec)
    except Exception as exc:
        receipt.update(outcome="unknown", detail=f"custody could not be read: {type(exc).__name__}")
    try:
        append_jsonl(root / "logs" / "events.jsonl", {"ts": receipt["stopped_at"], "type": "kept_service_stop",
                     **public_retained_receipt(receipt), "actor": actor, "reason": receipt["reason"]})
    except Exception:
        log.warning("kept service stop of %s is not recorded in events", launch_id, exc_info=True)
    return receipt


def _stop_selected(root, receipt, match, pending, complete, note, finished_producer, timeout_sec):
    from ouroboros.owned_shutdown import _stamp, _stop_records
    from ouroboros.tools.kept_service_outputs import capture_kept_service_outputs

    launch_id, actor, reason = receipt["launch_id"], receipt["actor"], receipt["reason"]
    producer = match[0]["task_id"] if match is not None else pending[1] if pending else ""
    unfinished = _producer_unfinished(root, producer) if finished_producer and (match or pending) else ""
    if unfinished:
        receipt.update(task_id=producer, outcome="refused", refusal="producer_unfinished", detail=(
            f"{unfinished}; a root task stops only a service a finished task left running (its own services: "
            "stop_service(name=...)); nothing was signalled"))
    elif match and match[0]["state"] == "unknown":
        receipt.update(outcome="unknown", detail="the state of this launch is unknown; nothing was signalled")
    elif match is None and not complete:  # absent from a partial read is unknown, never "not running"
        receipt.update(outcome="unknown", detail=f"not listed, but custody could not be read whole ({note}); "
                                                 "nothing was signalled")
    elif match is None and pending:  # ended without a complete capture: an earlier stop's, or by itself
        receipt.update(task_id=pending[1], outcome="not_found",
                       outputs=capture_kept_service_outputs(pending[0], pending[1], launch_id, actor=actor,
                                                            reason=reason, verify_exit=True),
                       detail="no process of this launch is running; nothing was signalled; see outputs for its outstanding declaration")
        if receipt["outputs"]["state"] == "not_attempted":
            receipt.update(outcome="unknown", detail=receipt["outputs"]["detail"])
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
            after, complete, note = _kept_selection(root, launch_id=launch_id, task_id=item["task_id"])
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
            # verify_exit: the capture's own fresh census is the identity-checked end proof its first
            # stop row needs; this receipt's earlier read is not visible inside the capture.
            receipt["outputs"] = capture_kept_service_outputs(drive, item["task_id"], launch_id, actor=actor,
                                                              reason=reason, verify_exit=True,
                                                              legacy_target=str(entries[0].get("container_name") or ""))
            receipt.update(ok=receipt["outputs"]["state"] in ("captured", "none_declared"), outcome="stopped",
                           detail="no process of this launch is running; its log is kept for retention")
            if receipt["outputs"]["state"] == "not_attempted":  # the capture's own read did not confirm that end
                receipt.update(outcome="unknown", detail=receipt["outputs"]["detail"])
