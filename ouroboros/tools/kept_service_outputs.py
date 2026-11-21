"""A launch's declared outputs and the row its stop still owes.

All starts with outputs (and kept starts without them) publish their capture context and baseline before spawn
(``declare_kept``, ``start: unconfirmed`` until confirmed). A proven no-op retracts it; an unknown start is not an end.
Output declarations live under ``services/kept_outputs/<task>/<launch>.kept.json`` in the installation, so the hot
readers enumerate debts, not history. A no-output declaration lives in the producer's ``services/<task>``;
an ordinary launch with no outputs first writes it at its confirmed stop (``finish_own_outputs``).
A stop of that launch by the owner or another task (``launch_custody.stop_kept_service``) replays the ordinary capture
(``shell._register_process_outputs``) in the starting task's context, into ITS artifact store, never the
stopper's (``capture_kept_service_outputs``); a Docker launch's outputs are read only through the container it ran in.
It is OWED while it names outputs that may still hold bytes no capture took: its own task's stop by name settles it as
well (``finish_own_outputs``), and so does a capture that finds what it lacks proven absent (``missing``: a failed
output, not a debt). After an incomplete or unattempted capture, or once the launch ended by itself or with its
server, a stop of the same launch id finds it (``unconsumed_declaration``) and captures again, as does a same-name
start of its task before it writes anything (``capture_before_reuse``); reading a declaration, capturing and consuming
it hold the same producer/name lock wherever they run. A capture that speaks a launch's first confirmed end (no stop
snapshot in the declaration; its own complete custody read observed the end and a Docker group answered its probe:
asking to verify, or an inspection that could not be made, proves none) publishes it as a stop row
(``services_stopped``) into the producer's room; every later capture of that launch publishes only changed file
results (``service_outputs_updated``), never a second stop — the declaration's ``stop`` snapshot is the durable memory
of what was already spoken, and counts it (``row_revision``: the launch's one card row, each result after the last).
A confirmed end writes its row (``unspoken``) before telling it. ``say_unspoken`` keeps the same stamp and revision
until both recipients acknowledge it, with no second stop. The producer's completion pass adds the stop's source
(``tell_own_stop``); while that task may run, no other reader publishes the pass's row.
No timer ends a debt: until its files are settled (``files_settled``), retirement keeps the
declaration and the drive, copy or session scratch holding its working directory or outputs (``owed_output_hold``,
read by the retirement gates, each operation's candidates sharing one read of ``owed_declarations``); a record
whose files are settled holds none of them and is no listed debt. A declaration
that cannot say what it holds, or a directory of them that cannot be listed, stops the retirement operation reading it
at that operation's own error boundary, deleting nothing: it could name anything. The inventory lists the launch as
``capture_owed``. A declaration that names no outputs owes nothing and is retired with its logs into an event row
(``retire_kept_declaration``). Never the command or its environment.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pathlib
import re
import stat
import threading
import weakref
from dataclasses import dataclass
from typing import Any, Dict

from ouroboros.owned_shutdown import installation_root
from ouroboros.service_outcomes import ROW_OWED, row_teller
from ouroboros.tool_access import active_tool_profile, canonical_data_root
from ouroboros.tools.registry import ToolContext
from ouroboros.utils import append_jsonl, utc_now_iso


KEPT_DECLARATION = ".kept.json"
KEPT_DIRECTORY = "kept_outputs"
_CAPTURE_LOCK_SEC = 10.0  # how long one capture waits for another process's capture of the same task's launches


class KeptDeclarationUnreadable(OSError):
    """An outstanding declaration cannot say what it holds (unreadable, or not what ``declare_kept`` wrote): the debts
    are not known whole, so a retirement reading them stops at its own error boundary, deleting nothing."""


class KeptCaptureBusy(OSError):
    """The producer's capture lock could not be taken: nothing was read or consumed, and a retry captures."""


class KeptNamesake(RuntimeError):
    """A kept launch of the name a start would reuse still holds it (``capture_before_reuse``): ``state`` ``running``,
    ``unknown`` (it may run, or what says so cannot be read: nothing was signalled) or ``owed`` (it ended and its
    declared outputs may still hold uncaptured bytes: a retried start captures them first); ``launch``: its kept row."""

    def __init__(self, state: str, launch: Dict[str, Any], cause: str):
        super().__init__(cause)
        self.state, self.launch = state, launch


@dataclass
class _UnwrittenDeclaration:
    context: ToolContext
    release: weakref.finalize


def output_producer(ctx, record):
    """The context a launch's outputs are captured in: its producer's roots as its start froze them, while a
    reservation holds them and after its declaration took them over (``finish_own_outputs``); else the caller's."""
    reservation = getattr(record, "unwritten_outputs", None)
    return reservation.context if reservation is not None else getattr(record, "producer_context", None) or ctx


def pending_output_stop(record):
    return getattr(record, "stopped_outputs", None)


_name_holds = threading.local()


def _name_lock_path(root, task_id, name):
    key = hashlib.sha256(f"{task_id}\0{name}".encode()).hexdigest()
    return installation_root(root) / "state" / "service_name_locks" / key


@contextlib.contextmanager
def writer_reservation(root, task_id, name, needs_declaration):
    """Under the name lock, exclude a new writer if a live producer alone holds the old capture inputs.

    Only a writer with outputs needs a reservation of its own. Reserve before spawn so even a full
    disk after launch cannot erase it. Activity Stop takes the ordinary name lock independently.
    """
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    with contextlib.ExitStack() as reservation:
        if root is not None:
            path = _name_lock_path(root, task_id, name).with_suffix(".writer")
            if not needs_declaration:
                try:
                    path.stat()
                except FileNotFoundError:
                    yield reservation
                    return
            fd = acquire_exclusive_file_lock(path, timeout_sec=0, owner_aware_stale=True)
            if fd is None:
                raise KeptCaptureBusy("the producer may still hold unwritten output declarations for this name")
            pid = os.getpid()

            def release():
                if os.getpid() == pid:  # a fork inherited the descriptor, not the producer's name
                    release_exclusive_file_lock(path, fd)

            reservation.callback(release)
        yield reservation


@contextlib.contextmanager
def service_name_hold(root, task_id: str, name: str):
    """Serialize one producer/name across workers. Its replacement can itself finish the old capture."""
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    if root is None:
        yield
        return
    path = _name_lock_path(root, task_id, name)
    held = getattr(_name_holds, "paths", ())
    owner = (os.getpid(), path)
    if owner in held:
        yield
        return
    fd = acquire_exclusive_file_lock(path, timeout_sec=_CAPTURE_LOCK_SEC, owner_aware_stale=True)
    if fd is None:
        raise KeptCaptureBusy("could not acquire the service name for capture or replacement")
    _name_holds.paths = (*held, owner)
    try:
        yield
    finally:
        _name_holds.paths = held
        release_exclusive_file_lock(path, fd)


def _owed_directory(drive_root: Any) -> pathlib.Path:
    return installation_root(drive_root) / "services" / KEPT_DIRECTORY


def declaration_path(drive_root: Any, task_id: str, launch_id: str, *, owed: bool = True) -> pathlib.Path:
    """Where this launch's declaration lives: ``owed`` (it names outputs) in the installation's one directory of
    outstanding declarations, whatever drive started it (its producer's capture context travels inside, so placement
    never changes whose artifact store it lands in); else beside its log, where only retention visits it."""
    name = f"{launch_id}{KEPT_DECLARATION}"
    if not owed:
        return pathlib.Path(drive_root) / "services" / str(task_id) / name
    return _owed_directory(drive_root) / str(task_id) / name


def _owes(path: pathlib.Path, declaration: Any) -> bool:
    """Owed: it names outputs, cannot be read (unknown, so owed), or lies in the outstanding directory, which only a
    start naming outputs writes: whatever else is found there is damage, never "owes nothing"."""
    return path.parent.parent.name == KEPT_DIRECTORY or not isinstance(declaration, dict) or bool(declaration.get("outputs"))


def _names_places(declaration: Any) -> bool:
    if not isinstance(declaration, dict):
        return False
    cwd, outputs = declaration.get("cwd"), declaration.get("outputs")
    return bool(cwd) and isinstance(cwd, str) and isinstance(outputs, list) and bool(outputs) and all(
        isinstance(item, str) for item in outputs)


def _read(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except Exception:  # noqa: BLE001 — unreadable: unknown, never "owes nothing"
        return None


def _entries(directory: pathlib.Path) -> list:
    """``directory``'s entries in name order; ``[]`` when it does not exist. Any other failure to list it raises:
    ``Path.glob`` would drop a ``PermissionError`` and read the debts as none."""
    try:
        with os.scandir(directory) as entries:
            return sorted(entries, key=lambda entry: entry.name)
    except FileNotFoundError:
        return []


def _outstanding_files(root: Any) -> list:
    """Every declaration file of the one outstanding directory, each producer's folder listed strictly (``_entries``)."""
    return [pathlib.Path(file.path) for producer in _entries(_owed_directory(root))
            if producer.is_dir() for file in _entries(pathlib.Path(producer.path)) if file.name.endswith(KEPT_DECLARATION)]


def outstanding_declarations(root: Any) -> list:
    """``[(path, declaration or None)]``, for a reader that says what it can (the inventory): a file that cannot be
    read is ``None``, a directory that cannot be listed raises."""
    found = []
    for path in _outstanding_files(root):
        try:
            found.append((path, _read(path)))
        except FileNotFoundError:  # consumed since it was listed
            continue
    return found


def owed_declarations(root: Any) -> list:
    """``[(path, declaration)]``, the debts a retirement operation decides on: only the one outstanding directory is
    read, so the cost is what is owed, never task history, and all of it is owed (``_owes``). Each must say what it
    holds: one that cannot (unreadable or malformed, ``_names_places``) raises ``KeptDeclarationUnreadable``, as a
    directory that cannot be listed raises its ``OSError``: the operation stops at its own error boundary, deleting
    nothing. A pass over many candidates reads it once and passes it on."""
    found = outstanding_declarations(root)
    for path, declaration in found:
        if not _names_places(declaration):
            raise KeptDeclarationUnreadable(f"kept service declaration cannot say what it holds: {path}")
    return found


def _inside(place: pathlib.Path, location: pathlib.Path) -> bool:
    return place == location or location in place.parents


def _held_places(path: pathlib.Path, declaration: Any) -> list:
    """What one outstanding declaration holds: its own file, its working directory, each declared output and the
    drive its capture writes into. Once every file is settled (``files_settled``) only itself: a confirmed capture
    keeps no drive for the row its room is still owed."""
    if files_settled(declaration):
        return [path]
    cwd = pathlib.Path(declaration["cwd"])
    places: list = [path, cwd, *(cwd / output for output in declaration["outputs"])]
    context = declaration.get("context") if isinstance(declaration.get("context"), dict) else {}
    if context.get("drive_root"):
        places.append(pathlib.Path(str(context["drive_root"])))
    return places


def owed_output_hold(root: Any, location: Any, *, owed: Any = None) -> str:
    """Why ``location`` (a drive, copy or session scratch a retirer would delete) must stay:
    ``"kept_service_capture_owed"`` when an owed declaration's working directory, declared output or capture drive lies
    inside it, else ``""``. ``owed``: the selection this operation already read (``owed_declarations``), so its
    candidates share one read; an effect-time recheck passes none and reads its own (raising, like the operation's first
    read, when the debts cannot be known). Specific by construction: no other drive or copy is held."""
    target = pathlib.Path(location).resolve(strict=False)
    for path, declaration in owed_declarations(root) if owed is None else owed:
        if any(_inside(pathlib.Path(place).resolve(strict=False), target) for place in _held_places(path, declaration)):
            return "kept_service_capture_owed"
    return ""


def _declaration_address(ctx: ToolContext, record: Any) -> tuple:
    """The selected launch's declaration address, independent of the current holder of its name."""
    from ouroboros.service_outcomes import launch_facts

    launch_id = launch_facts(record).get("launch_id")
    if not launch_id:  # an older record with no launch of its own: the ownership set's id of its unmarked row
        proc = getattr(record, "proc", None) or getattr(record, "local_proc", None)
        launch_id = f"pid-{int(getattr(proc, 'pid', None) or 0)}"
    return getattr(record, "drive_root", None) or ctx.drive_root, record.task_id, launch_id


def finish_own_outputs(ctx, record, payload, finish):
    """A confirmed stop: capture, then leave the launch record what this end still owes. Outputs a later capture may
    take stay with the producer's frozen inputs. The row a completion pass will tell (``row_teller``) is written with
    the confirmed end before anything is said, for every launch, with or without declared outputs, and the stop names
    it (``ROW_OWED``): the record is consumed only once that row is found stored (``tell_own_stop``). A stop no pass
    will tell settles as it always did: it consumes what it captured."""
    from ouroboros.service_outcomes import after_stop_error

    producer, told = output_producer(ctx, record), row_teller(ctx)
    registered, prior, held = False, None, None
    try:
        drive, task, launch = _declaration_address(ctx, record)
        path = declaration_path(drive, task, launch, owed=bool(record.outputs))
        with service_name_hold(drive, task, record.name):
            if getattr(record, "kept_declared", False):
                try:
                    held = _read(path)
                except FileNotFoundError:  # an outside capture consumed it; do not read a future writer's files
                    payload["output_settled_elsewhere"] = True
                    return False
                prior = _confirmed_stop(held) if isinstance(held, dict) else None
                if prior and "row_revision" not in payload:
                    # An outside capture confirmed this end first: its settled files are not read again, and this
                    # stop's row follows the rows that record already told (the retry path sets both itself).
                    payload.update(output_outcomes=payload.get("output_outcomes") or prior.get("output_outcomes"),
                                   row_revision=int(prior.get("row_revision") or 1) + 1)
            try:
                registered = finish(producer, record, payload)
            except Exception as exc:
                after_stop_error(payload, record.outputs, exc)
            problem = _remember_end(producer, record, payload, drive, launch, told, prior)
            if problem:
                payload.update(problem)
                return registered  # the previous declaration still holds the producer's bytes
            if (not told and not payload.get("artifact_output_owed")
                    and isinstance(held, dict) and _UNSPOKEN not in held):
                _consume(path)  # no outstanding files, unwritten stop row, or earlier row debt
            if getattr(record, "unwritten_outputs", None) is not None:
                # The declaration holds the capture inputs now, so the writer reservation goes. A later retry of
                # this record still resolves its outputs in the producer's frozen roots, never a successor's.
                record.producer_context = record.unwritten_outputs.context
                record.unwritten_outputs.release()
                record.unwritten_outputs = None
    except (OSError, ValueError) as exc:
        after_stop_error(payload, record.outputs, exc)
        if getattr(record, "kept_declared", False):
            payload["artifact_output_owed"] = bool(record.outputs)  # the outstanding declaration was not retired
    return registered


def _remember_end(producer, record, payload, drive, launch, told, prior=None) -> Dict[str, str]:
    """Write this stop's confirmed end into the launch record when an output is still owed or a completion pass will
    tell its row (``told``, and the stop's own observation names this launch, so the pass finds the record): that row
    rides with the end, and the stop names it (``ROW_OWED``). Returns the writer's problem, if it had one."""
    from ouroboros import utils
    from ouroboros.service_outcomes import output_stop_snapshot

    told = told and payload.get("launch_id") == launch
    if not told and not payload.get("artifact_output_owed"):
        return {}
    stop = {**output_stop_snapshot(payload, record), "launch_id": launch}
    if prior and payload.get("output_retry") and _file_facts(stop["output_outcomes"]) == _file_facts(prior.get("output_outcomes")):
        return {}  # the file answers its record already holds are no news: it stays as it is, and the pass tells no row
    mark = {"ts": utils.utc_now_iso(), "pass": True, "observation": {
        **stop, "lifecycle": "outputs_updated" if payload.get("output_retry") else "stopped"}} if told else None
    if mark:
        payload[ROW_OWED] = {"ts": mark["ts"], "drive": str(drive)}
    return declare_kept(producer, record, stop=stop, unspoken=mark)


def release_after_stop(record, payload, records, lock):
    """Leave a by-name handle for a retry; the same declaration lets another worker recover the stopped launch."""
    with lock:
        if records.get(record.service_id) is record:
            if payload is not None and (payload.get("artifact_output_owed") or getattr(record, "retirement_pending", False)):
                record.stopped_outputs = payload.copy()
            else:
                records.pop(record.service_id)


def retry_own_outputs(ctx, record, finish, records, lock):
    with service_name_hold(record.drive_root or ctx.drive_root, record.task_id, record.name):
        with lock:
            stopped = pending_output_stop(record)
            if records.get(record.service_id) is not record or stopped is None:
                return None, False
            payload = stopped.copy()
        saved = confirmed_output_stop(*_declaration_address(ctx, record)) if getattr(record, "kept_declared", False) else None
        if saved:
            payload["output_outcomes"] = saved.get("output_outcomes")
        payload["output_retry"] = True
        # Whatever this retry finds follows everything this launch's record already told its room.
        payload["row_revision"] = max(int(payload.get("row_revision") or 1), int((saved or {}).get("row_revision") or 1)) + 1
        for field in ("artifact_output_owed", "artifact_output_failed", "artifact_output_unknown",
                      "artifact_outputs", "kept_declaration_problem", ROW_OWED):  # this retry's own, or none
            payload.pop(field, None)
        registered = finish_own_outputs(ctx, record, payload, finish)
        release_after_stop(record, payload, records, lock)
        return (None, False) if payload.get("output_settled_elsewhere") else (payload, registered)


def _consume(path: pathlib.Path) -> None:
    from ouroboros.task_custody import fence_publication
    fence_publication()
    path.unlink(missing_ok=True)
    with contextlib.suppress(OSError):  # another declaration or a concurrent write keeps this directory
        fence_publication()
        path.parent.rmdir()  # an empty producer directory must not become another historical census


def unconsumed_declaration(root: Any, launch_id: str) -> tuple | None:
    """``(drive, task_id)`` of the owed declaration of ``launch_id`` (a validated launch id) no complete capture
    consumed, or None. Read in the one outstanding directory, never a walk; a directory that cannot be listed raises
    (absence read from a partial listing would be a guess). One naming no outputs owes nothing."""
    name = f"{launch_id}{KEPT_DECLARATION}"
    for producer in _entries(_owed_directory(root)):
        if producer.is_dir():
            try:
                (pathlib.Path(producer.path) / name).stat()
            except FileNotFoundError:
                continue
            return installation_root(root), producer.name
    return None


def capture_before_reuse(drive_root: Any, task_id: str, name: str) -> dict | None:
    """Settle outstanding files before letting a namesake write them. Return its gaps (possibly empty),
    or None when this call did not finalize any previous declaration."""
    if drive_root is None:
        return
    from ouroboros.service_outcomes import restart_gaps

    gap = None
    try:
        for entry in _entries(_owed_directory(drive_root) / task_id):
            if not entry.name.endswith(KEPT_DECLARATION):
                continue
            path = pathlib.Path(entry.path)
            try:
                declaration = _read(path)
            except FileNotFoundError:
                continue
            if not isinstance(declaration, dict) or not declaration.get("name"):
                raise KeptDeclarationUnreadable("an outstanding declaration cannot identify its service")
            if declaration["name"] != name:
                continue
            launch_id = entry.name[:-len(KEPT_DECLARATION)]
            answer = capture_kept_service_outputs(drive_root, task_id, launch_id, actor=f"agent:{task_id}",
                                                  reason="capture before same-name restart", verify_exit=True)
            if answer["state"] == "not_attempted":
                raise KeptNamesake(answer.get("launch", {}).get("state", "unknown"), answer.get("launch", {}), answer["detail"])
            if answer.get("only_told"):
                continue  # settled before this call: only its room was owed, and this start has nothing to disclose
            try:
                left = _read(path)
            except FileNotFoundError:
                left = None
            if left is not None and not files_settled(left):  # a record kept only for its room owes the writer nothing
                raise KeptNamesake("owed", {}, f"{launch_id} still owes declared outputs ({answer['state']})")
            if answer["state"] == "missing":
                gap = restart_gaps({"artifact_output_failed": True})
            elif answer["state"] in ("captured", "none_declared") and gap is None:
                gap = {}
    except OSError as exc:
        raise KeptNamesake("unknown", {}, f"previous output declarations could not be checked ({type(exc).__name__})") from exc
    return gap


def retry_named_outputs(ctx: ToolContext, name: str) -> dict | None:
    """Recover a confirmed exit by name. An unreadable declaration neither identifies that name
    nor rules out a readable match later in the same producer's directory."""
    drive, task = ctx.drive_root, str(ctx.task_id or "manual")
    with service_name_hold(drive, task, name):
        unreadable = None
        for entry in _entries(_owed_directory(drive) / task):
            if not entry.name.endswith(KEPT_DECLARATION):
                continue
            launch = entry.name[:-len(KEPT_DECLARATION)]
            try:
                held, declaration = _load_declaration(drive, task, launch)
            except FileNotFoundError:
                continue
            except (OSError, ValueError) as exc:
                unreadable = exc
                continue
            saved = _confirmed_stop(declaration)
            if declaration["name"] != name or not saved and declaration.get("keep_alive", True):
                continue
            if files_settled(declaration):  # nothing is left to capture: at most its room is still owed its row
                say_unspoken(held, declaration)
                continue
            # A caller's completion pass tells what this finds: the row waits for it in the record (``ROW_OWED``).
            result = capture_kept_service_outputs(drive, task, launch, actor=f"agent:{task}",
                reason="capture after confirmed exit", verify_exit=True, checkpoint=False, owed=row_teller(ctx))
            if not saved and not result.get("observation"):
                raise KeptDeclarationUnreadable("the ordinary launch's end could not be confirmed for capture")
            return {**(saved or {}), **(result.get("observation") or {"artifact_output_failed": True,
                    "artifact_output_unknown": True, "artifact_output_owed": True, "output_retry": True}),
                    **({ROW_OWED: {"ts": result["row_ts"], "drive": str(drive)}} if result.get("row_ts") else {})}
        if unreadable is not None:
            raise unreadable
    return None


def _executor_facts(record):
    executor = getattr(record, "executor", None)
    if executor is None:
        return {}
    return {"type": executor.kind, "id": executor.executor_id, "network": executor.network,
            "container_name": getattr(executor, "container_name", ""), "path_mappings": [
                {"host_path": str(mapping.host_path), "backend_path": mapping.backend_path}
                for mapping in getattr(executor, "mappings", ())]}


def declare_kept(ctx: ToolContext, record: Any, reservation=None, *, stop=None, unspoken=None,
                 prepared: bool = False) -> Dict[str, str]:
    """Publish the frozen producer inputs before spawn, then the confirmed start or a stop's new facts."""
    if not (record.keep_alive or record.outputs or stop):
        return {}
    from ouroboros.utils import atomic_write_json

    context = None
    try:
        drive, task_id, launch_id = _declaration_address(ctx, record)
        metadata = getattr(ctx, "task_metadata", None) if isinstance(getattr(ctx, "task_metadata", None), dict) else {}
        contract = getattr(ctx, "task_contract", None) or metadata.get("task_contract")
        ceiling = contract.get("capability_ceiling") if isinstance(contract, dict) else None
        context = {"repo_dir": str(ctx.repo_dir), "system_repo_dir": str(getattr(ctx, "system_repo_dir", "") or ""),
                   "drive_root": str(ctx.drive_root), "budget_drive_root": str(canonical_data_root(ctx)),
                   "workspace_root": str(getattr(ctx, "workspace_root", "") or ""),
                   "workspace_mode": str(getattr(ctx, "workspace_mode", "") or ""),
                   "tool_profile": str(active_tool_profile(ctx)),
                   "executor_ref": _executor_facts(record),
                   **({"capability_ceiling": ceiling} if ceiling is not None else {})}
        if prepared and record.outputs:
            record.producer_context = _producer_context({"context": context, "task_id": task_id})
        path = declaration_path(drive, task_id, launch_id, owed=bool(record.outputs))
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, {"schema": 2, "launch_id": launch_id, "task_id": task_id, "declared_at": utc_now_iso(),
                                 "name": record.name, "cwd": str(getattr(record, "host_cwd", None) or record.cwd),
                                 "cwd_root": record.cwd_root, "cwd_base": record.cwd_base, "cwd_source": record.cwd_source,
                                 "skill_name": record.skill_name, "outputs": record.outputs,
                                 "before_outputs": {key: list(value) for key, value in record.before_outputs.items()},
                                 "keep_alive": bool(record.keep_alive), **({"stop": stop} if stop else {}),
                                 **({_UNSPOKEN: unspoken} if unspoken else {}), **({"start": _UNCONFIRMED} if prepared else {}),
                                 **({"backend_pid": record.backend_pid, "custody_drive": str(drive)}
                                    if context["executor_ref"].get("type") == "docker_exec" else {}),
                                 "context": context}, trailing_newline=True)
        record.kept_declared = True
        return {}
    except Exception as exc:  # noqa: BLE001 — the producer still owns the unwritten declaration, if held
        if record.outputs and reservation is not None:
            # A shutdown can also discard this record. Its writer descriptor lives at most as long as the owner.
            producer = _producer_context({"context": context, "task_id": record.task_id}) if context else ctx
            record.unwritten_outputs = _UnwrittenDeclaration(producer, weakref.finalize(record, reservation.pop_all().close))
        return {"kept_declaration_problem": str(type(exc).__name__)}


def _producer_context(declaration: Dict[str, Any]) -> ToolContext:
    """The starting task's capture context as its start recorded it: its roots, profile, executor and ceiling."""
    facts = declaration.get("context") if isinstance(declaration.get("context"), dict) else {}
    metadata: Dict[str, Any] = {"tool_profile": str(facts.get("tool_profile") or ""),
                                "budget_drive_root": str(facts.get("budget_drive_root") or ""),
                                "executor_ref": dict(facts.get("executor_ref") or {})}
    if "capability_ceiling" in facts:
        metadata["task_contract"] = {"capability_ceiling": facts["capability_ceiling"]}
    optional = {key: pathlib.Path(str(facts[key])) for key in ("system_repo_dir", "workspace_root") if facts.get(key)}
    return ToolContext(repo_dir=pathlib.Path(str(facts["repo_dir"])), drive_root=pathlib.Path(str(facts["drive_root"])),
                       workspace_mode=str(facts.get("workspace_mode") or ""), task_metadata=metadata,
                       executor_ref=dict(metadata["executor_ref"]), task_id=str(declaration["task_id"]), **optional)


def _capture_executor(path, declaration, legacy_target, verify_exit):
    """``(executor, ended Docker record, probed)`` of the launch's original capture target. ``probed``: nothing
    of this target is left unproven by the caller's custody read: a host or local launch needs no more, and a
    Docker group answered ``exited`` on its recorded identity (anything else raises). False: a Docker target
    this call did not or could not probe, which proves no end. The ended record is the process record a proven end
    left behind: this call's probe proved it, or the declaration's own confirmed stop did."""
    from dataclasses import replace
    from ouroboros import workspace_executor as backend
    from ouroboros.workspace_executor_paths import normalize_executor_ref

    executor = normalize_executor_ref(declaration["context"].get("executor_ref") or {})
    if declaration["launch_id"].startswith(("launch-", "pid-")):
        # A host fallback can carry its task's Docker configuration in a legacy declaration.
        if declaration.get("schema") == 2 and executor and executor.kind == "docker_exec":
            raise ValueError("a host launch cannot identify this Docker output target")
        return (executor if executor and executor.kind == "local" else None), None, True
    if declaration.get("schema") == 2 and executor and executor.kind == "local":
        return executor, None, True
    if not executor or executor.kind != "docker_exec":
        raise ValueError("the original Docker output target is unknown")
    pid = declaration.get("backend_pid")
    drive = pathlib.Path(declaration.get("custody_drive") or declaration["context"]["drive_root"])
    if declaration_path(drive, declaration["task_id"], declaration["launch_id"]) != path:
        raise ValueError("the process record belongs to another installation")
    record_path = drive / "state" / backend._PROCESS_STATE_DIR / f"{declaration['launch_id']}.json"
    try:
        stored = backend._load_process_record(record_path, strict=True)
    except FileNotFoundError:
        stored = None
    else:
        if (not stored or not backend._valid_process_record(record_path, stored, check_identity=False)
                or bool(stored.get("keep_alive")) != declaration.get("keep_alive", True) or stored.get("record_type") != "service"
                or stored.get("executor_type") != "docker_exec" or stored.get("task_id") != declaration["task_id"]
                or stored.get("name") != declaration["name"] or (pid and stored.get("backend_pid") != pid)):
            raise ValueError("the durable record cannot identify this Docker service")
        pid = stored["backend_pid"]
    if declaration.get("schema") != 2:
        target = stored["container_name"] if stored else legacy_target
        if not re.fullmatch(r"[0-9a-f]{64}", target or ""):
            raise ValueError("the original container identity of this legacy service is unknown")
        from ouroboros.utils import atomic_write_json

        executor = replace(executor, container_name=target)
        declaration["context"]["executor_ref"]["container_name"] = target
        declaration.update(schema=2, backend_pid=pid)
        atomic_write_json(path, declaration, trailing_newline=True)
    if stored and stored["container_name"] != executor.container_name:
        raise ValueError("the output declaration and durable Docker target disagree")
    if _confirmed_stop(declaration) or (stored and stored.get("backend_completed") is True):
        return executor, record_path if stored else None, True
    if verify_exit and isinstance(pid, str) and pid.isdigit() and int(pid) > 0:
        # Publishing the process record or its ownership copy may have failed.
        state = backend._docker_pid_state(executor.container_name, pid)
        if state != "exited":
            raise KeptNamesake(state, {"launch_id": declaration["launch_id"], "state": state},
                               "the original Docker group is not confirmed ended")
        if stored:
            with contextlib.suppress(OSError, ValueError):  # the later stop snapshot still carries the measured end
                backend._remember_service_backend_exit(record_path)
        return executor, record_path if stored else None, True
    return executor, None, False


def _load_declaration(drive_root, task_id, launch_id):
    path = declaration_path(drive_root, task_id, launch_id)
    try:
        declaration = _read(path)
    except FileNotFoundError:
        path = declaration_path(drive_root, task_id, launch_id, owed=False)
        declaration = _read(path)
    if (not isinstance(declaration, dict) or declaration.get("task_id") != task_id
            or declaration.get("launch_id") != launch_id or not isinstance(declaration.get("name"), str)
            or not declaration["name"] or not isinstance(declaration.get("context"), dict)
            or not declaration["context"].get("drive_root")
            or (_owes(path, declaration) and not _names_places(declaration))):
        raise KeptDeclarationUnreadable("the declaration cannot identify its service and outputs")
    return path, declaration


def _confirmed_stop(declaration):
    stop = declaration.get("stop")
    if not isinstance(stop, dict) or any(stop.get(key) != declaration.get(key) for key in ("task_id", "launch_id", "name")):
        return None
    cleanup = stop.get("process_cleanup") if isinstance(stop.get("process_cleanup"), dict) else {}
    if (stop.get("state") in ("exited", "stopped") and not stop.get("stop_failed") and (
            cleanup.get("exit_confirmed") is True or stop.get("backend") == "docker" and stop.get("cleanup_dispatched") is True)):
        return stop
    return None


def confirmed_output_stop(drive, task, launch):
    try:
        return _confirmed_stop(_load_declaration(drive, task, launch)[1])
    except FileNotFoundError:
        return None


def capture_kept_service_outputs(drive_root: Any, task_id: str, launch_id: str, *, actor: str,
                                 reason: str, legacy_target: str = "", verify_exit: bool = False,
                                 checkpoint: bool = True, owed: bool = False) -> Dict[str, Any]:
    """A confirmed exit: capture into its producer's store. A competing capture may already have consumed it.
    ``checkpoint``: this call says what it finds through the door. Else ``owed``: its caller's completion pass will,
    and the row waits in the record (``row_ts``), never in memory alone; with neither, the facts are only handed back."""
    try:
        _, selected = _load_declaration(drive_root, task_id, launch_id)
        with service_name_hold(drive_root, task_id, selected["name"]):
            path, declaration = _load_declaration(drive_root, task_id, launch_id)
            if selected["name"] != declaration["name"]:
                raise KeptDeclarationUnreadable("the saved service name changed")
            mark = declaration.get(_UNSPOKEN)
            declaration = say_unspoken(path, declaration)
            if declaration is None or _UNSPOKEN in declaration and files_settled(declaration):
                # Every file was settled before this call and only its room's row was owed: nothing is captured
                # or drawn again. The room is told now (the record is gone) or is still owed that same row.
                last = mark.get("observation") or {}
                state = "missing" if last.get("artifact_output_failed") else "captured" if selected.get("outputs") else "none_declared"
                return {"task_id": task_id, "state": state, "registered": False, "only_told": True, "observation": last,
                        "output_outcomes": last.get("output_outcomes") or [], "detail": "" if declaration is None else _ROOM_OWED}
            confirmed = _confirmed_stop(declaration)
            ordinary = not declaration.get("keep_alive", True)
            ended = False
            if ordinary and not confirmed:
                from ouroboros.launch_custody import ordinary_launch_ended

                # A Docker launch is checked by _capture_executor on its own pinned group, never by a host census.
                drive = pathlib.Path(declaration["context"]["drive_root"])
                ended = verify_exit and (launch_id.startswith("service-") or ordinary_launch_ended(
                    drive, task_id, launch_id, declaration["name"], started=declaration.get("start") != _UNCONFIRMED))
                if not ended:
                    return {"task_id": task_id, "state": "not_attempted", "detail": "the launch's exit is not confirmed"}
            elif verify_exit and not confirmed:
                from ouroboros.launch_custody import _kept_selection

                found, complete, note = _kept_selection(drive_root, launch_id=launch_id, task_id=task_id)
                if found or not complete:
                    launch = found[0][0] if found else {}
                    return {"task_id": task_id, "state": "not_attempted", "launch": launch,
                            "detail": f"the previous service is {launch.get('state', 'unknown')}; {note}"}
                ended = True
            return _capture(path, declaration, actor, reason, legacy_target, ended, checkpoint, owed)
    except FileNotFoundError:
        return {"task_id": task_id, "state": "unavailable", "detail": "this launch recorded no output declaration"}
    except (OSError, ValueError) as exc:
        return {"task_id": task_id, "state": "unavailable", "detail": f"its output declaration cannot be read ({type(exc).__name__}); it is kept"}


def _file_facts(rows):
    """What a capture says of the files; which call registered a file is no news of the file."""
    return [{key: value for key, value in row.items() if key != "registered"} for row in rows or ()]


def _capture(path, declaration, actor, reason, legacy_target, ended, checkpoint, owed=False):
    """``ended``: this call's own complete custody read found no process of the launch (never the request to verify)."""
    from ouroboros.service_outcomes import declared_output_outcomes, display, public_outputs
    from ouroboros.tools.services import _capture_service_outputs
    from ouroboros.utils import atomic_write_json
    from ouroboros.workspace_executor_paths import prepare_docker_outputs

    task_id, launch_id = declaration["task_id"], declaration["launch_id"]
    outputs = [str(item) for item in declaration.get("outputs") or []]
    notes, registered, state, payload = "", False, "none_declared", dict(declaration)
    stop = _confirmed_stop(declaration)
    payload["output_outcomes"] = (stop or {}).get("output_outcomes")
    ended_record = None
    probed = not outputs  # nothing to capture resolves no backend target: the custody read is the whole proof
    if outputs:
        try:
            ctx, cwd = _producer_context(declaration), str(declaration.get("cwd") or "")
            executor, ended_record, probed = _capture_executor(path, declaration, legacy_target, ended)
            if not probed:
                return {"task_id": task_id, "state": "not_attempted", "detail": "the backend group exit is not confirmed"}
            answered = {row["index"] for row in payload["output_outcomes"] or () if row["status"] in _SETTLED}
            context = prepare_docker_outputs(executor, pathlib.Path(cwd),
                [item for i, item in enumerate(outputs, 1) if i not in answered], pinned=True)
            before = {key: tuple(value) for key, value in (declaration.get("before_outputs") or {}).items()}
            registered = _capture_service_outputs(ctx, payload, before, context, executor=executor)
            notes = payload.get("artifact_outputs", "")
            state = ("incomplete" if payload.get("artifact_output_owed") else
                     "missing" if payload.get("artifact_output_failed") else "captured")
        except KeptNamesake as exc:
            return {"task_id": task_id, "state": "not_attempted", "launch": exc.launch, "detail": str(exc)}
        except Exception as exc:  # noqa: BLE001 — said, never a clean capture
            notes, state = f"Declared output capture could not be confirmed ({type(exc).__name__}).", "incomplete"
            if stop is None and not probed:
                return {"task_id": task_id, "state": "not_attempted", "detail": notes}
    # The first confirmed end no stop snapshot attests, and this capture its only speaker: the custody read
    # above observed it and the capture target's own inspection agreed (a Docker group answered its probe).
    # An inspection that was not made, or failed before it answered, creates no exit: only file facts speak. Nor does
    # a start nobody confirmed: no process found is then no proof that one ever ran.
    first_end = stop is None and ended and probed and (not declaration.get("keep_alive", True)
                                                     or declaration.get("start") != _UNCONFIRMED)
    outcomes = public_outputs(declared_output_outcomes(outputs, payload.get("output_outcomes")))
    observation = display({**(stop or {}),
        **({"state": "exited", "process_cleanup": {"exit_confirmed": True}} if first_end else {}),
        "task_id": task_id, "launch_id": launch_id, "name": declaration["name"],
        "lifecycle": "stopped" if first_end else "outputs_updated", "output_retry": not first_end,
        "stop_source": "retained",
        # The launch record counts what its room was told: a later file result orders after the stop it follows.
        "row_revision": int((stop or {}).get("row_revision") or 1) + (stop is not None),
        "log_outcome": (stop or {}).get("log_outcome", "not_attempted"), "output_outcomes": outcomes,
        "artifact_output_failed": state in ("incomplete", "missing"),
        "artifact_output_owed": state == "incomplete", "artifact_output_unknown": state == "incomplete"})
    row = {"ts": utc_now_iso(), "type": "kept_service_outputs", "task_id": task_id, "launch_id": launch_id,
           "name": observation.get("name"), "actor": actor, "reason": reason, "state": state, "output_outcomes": outcomes}
    drive = pathlib.Path(str(declaration["context"]["drive_root"]))
    recorded = append_jsonl(drive / "logs" / "events.jsonl", row)  # the audit receipt: who captured, and why
    settled = recorded and state in ("captured", "none_declared", "missing")  # at most its room's row is still owed
    held_name = bool(declaration.get(_NAME_HELD))
    if settled:
        held_name = False
        if ended_record is not None:
            # A proven Docker end leaves its process record behind, and a successor can reuse that group's PID: while
            # that record cannot be removed the declaration keeps the name (``files_settled``). That is the record's
            # fate alone: the confirmed end and the file answers are remembered and told below all the same.
            from ouroboros.workspace_executor import _forget_process

            if not _forget_process(ended_record):
                held_name = True
                notes += "\nCaptured outputs, but the old Docker service record still holds its name."
    if not recorded:
        state, notes = "incomplete", f"{notes}\nThe capture receipt could not be recorded."
    if state == "incomplete":
        notes = (f"{notes.strip()}\nIts output declaration and the files it names are kept: once the cause is fixed, "
                 f"stop_service(service_id={launch_id}, reason=...) captures them again.")
        observation.update(artifact_output_failed=True, artifact_output_owed=True, artifact_output_unknown=True)
    # Spoken: the first confirmed end; file results that differ from the ones the declaration's stop snapshot
    # already spoke; or, with no end proven and no snapshot to remember a partial result, the files of a settled
    # capture, once. Equal facts are no news; the receipt row above is their record.
    spoken = first_end or (_file_facts(outcomes) != _file_facts(stop.get("output_outcomes")) if stop is not None
                           else bool(outputs) and state != "incomplete")
    speak = checkpoint and spoken  # this call says it through the door
    keep = spoken and (speak or owed)  # the row waits in the record: for the door, or for the caller's completion pass
    remembered, written = declaration, True
    if spoken and (keep or not settled or held_name) or held_name != bool(declaration.get(_NAME_HELD)):
        # The launch record first, whatever became of the receipt above: the confirmed end, the file answers, the row
        # its room is owed and a name its old process record still keeps. A later capture then speaks only new file
        # results, never a second stop; the next reader says an unconfirmed row again (``say_unspoken``).
        remembered = {key: value for key, value in declaration.items() if key != _NAME_HELD}
        if held_name:
            remembered[_NAME_HELD] = True
        if spoken:
            remembered["stop"] = dict(observation) if stop is None else {
                **stop, "output_outcomes": outcomes, "row_revision": observation.get("row_revision", 1),
                **{key: observation[key] for key in ("artifact_output_failed", "artifact_output_owed", "artifact_output_unknown")}}
        if keep:  # the launch's newest row replaces an older one still owed: one row per launch
            remembered[_UNSPOKEN] = {"ts": row["ts"], "observation": dict(observation), **({} if speak else {"pass": True})}
        try:
            atomic_write_json(path, remembered, trailing_newline=True)
        except OSError:
            # Do not publish (or consume for) an unrecorded revision. The earlier record is the retry's source.
            remembered, written, speak = declaration, False, False
            notes += "\nThe updated declaration could not be written; the earlier one remains."
    said = speak and _told(drive, observation, row["ts"])
    # A row is still owed to its room: this one, not confirmed told or left for the caller's pass; or an older one
    # that nothing newer replaces (a capture nobody will tell hands its caller the launch's newest facts).
    owed_row = not said if keep else not spoken and _UNSPOKEN in remembered
    try:
        if settled and written and not owed_row and not held_name:
            _consume(path)  # every file is settled and its room was told: the facts live in those rows
        elif said and written:
            atomic_write_json(path, {key: value for key, value in remembered.items() if key != _UNSPOKEN},
                              trailing_newline=True)
    except OSError:
        if settled:
            state, notes = "incomplete", f"{notes}\nThe declaration could not be retired."
    if settled and owed_row and written and not held_name:
        notes += "\n" + _ROOM_OWED
    return {"task_id": task_id, "state": "incomplete" if held_name else state, "registered": bool(registered),
            "detail": notes.strip(), "output_outcomes": outcomes, "observation": observation,
            **({"row_ts": row["ts"]} if keep and written and not speak else {})}


_UNSPOKEN = "unspoken"  # declaration key: the row its room is not yet confirmed told, ``{ts, observation[, pass]}``
_UNCONFIRMED = "unconfirmed"  # declaration ``start``: written before the spawn, which nothing has confirmed yet
_NAME_HELD = "name_held"  # declaration key: the ended launch's Docker process record could not be removed
_SETTLED = {"saved", "partial", "absent"}
_ROOM_OWED = "Its room is not confirmed told; the launch record stays until it is, holding no files."


def _told(drive: pathlib.Path, observation: dict, ts: str, **writer: Any) -> bool:
    from ouroboros.service_finalization import RETAINED_TOLD, publish_retained_outputs

    return publish_retained_outputs(drive, observation, ts=ts, **writer) in RETAINED_TOLD


def files_settled(declaration: Any) -> bool:
    """Its own record says the launch's end left no file to capture: every declared one reached a final answer, or
    none was declared. No capture can add to it, so it holds no drive, copy or scratch and is no debt an inventory
    lists. It stays only until its room is told (``say_unspoken``) or, where its capture's receipt could not be
    recorded, until a repeated stop or retention takes it. A name its old Docker record still keeps is not settled."""
    stop = declaration.get("stop") if isinstance(declaration, dict) else None
    if not isinstance(stop, dict) or stop.get("artifact_output_owed") or declaration.get(_NAME_HELD):
        return False
    known = {row.get("index") for row in stop.get("output_outcomes") or ()
             if isinstance(row, dict) and row.get("status") in _SETTLED}
    outputs = declaration.get("outputs")
    return isinstance(outputs, list) and known >= set(range(1, len(outputs) + 1))


def say_unspoken(path: pathlib.Path, declaration: Any, *, late: bool = False, event_queue: Any = None) -> Any:
    """Under the name lock, retry the exact row; before the producer ends, only its pass may tell it.

    ``late`` is the producer's own pass or retention's authorization to recover. A confirmed publication drops
    only the mark, or consumes the whole declaration if the files are settled independently."""
    if not isinstance(declaration, dict) or _UNSPOKEN not in declaration:
        return declaration
    owed = declaration[_UNSPOKEN]
    if (not isinstance(owed, dict) or not isinstance(observation := owed.get("observation"), dict)
            or any(observation.get(key) != declaration.get(key) for key in ("task_id", "launch_id", "name"))
            or not isinstance(owed.get("ts"), str) or not owed["ts"]):
        return declaration  # unreadable debt is not settled
    from ouroboros.task_custody import fence_publication
    facts = declaration.get("context") if isinstance(declaration.get("context"), dict) else {}
    try:
        drive = pathlib.Path(str(facts["drive_root"]))
        if owed.get("pass") and not late:
            from ouroboros.launch_custody import _producer_unfinished

            if _producer_unfinished(installation_root(drive), str(declaration.get("task_id") or "")):
                return declaration
        if not _told(drive, observation, owed["ts"],
                     own=bool(owed.get("pass")), event_queue=event_queue):
            return declaration
        if files_settled(declaration):
            _consume(path)
            return None
        from ouroboros.utils import atomic_write_json

        told = {key: value for key, value in declaration.items() if key != _UNSPOKEN}
        fence_publication()
        atomic_write_json(path, told, trailing_newline=True)
        return told
    except (KeyError, OSError, ValueError):
        return declaration


def tell_own_stop(drive, observation: dict, owed: dict | None, event_queue: Any) -> None:
    """Complete or repair the stop's durable handoff; absence of its hint never settles an earlier declaration."""
    from ouroboros import utils
    from ouroboros.service_outcomes import display

    owed = owed or {}
    drive = pathlib.Path(owed.get("drive") or drive)
    ts = owed.get("ts") or utils.utc_now_iso()
    observation = display(observation)
    try:
        task, launch, name = (observation[key] for key in ("task_id", "launch_id", "name"))
        with service_name_hold(drive, task, name):
            try:
                path, declaration = _load_declaration(drive, task, launch)
            except FileNotFoundError:
                if observation.get("output_outcomes") != [] or observation.get("lifecycle") != "stopped":
                    raise  # a display receipt cannot reconstruct missing capture inputs
                path = declaration_path(drive, task, launch, owed=False)
                path.parent.mkdir(parents=True, exist_ok=True)
                declaration = {"schema": 2, "launch_id": launch, "task_id": task, "name": name,
                               "declared_at": ts, "outputs": [], "keep_alive": False,
                               "context": {"drive_root": str(drive)}}
            revision = observation.get("row_revision", 1)
            prior = declaration.get("stop")
            if isinstance(prior, dict) and type(prior.get("row_revision")) is int and prior["row_revision"] > revision:
                return  # a newer fact already replaced this card row
            mark = declaration.get(_UNSPOKEN)
            if isinstance(mark, dict) and (mark.get("observation") or {}).get("row_revision", 1) == revision:
                ts = mark.get("ts") or ts  # never change the timestamp of a row that may have landed
            pending = {**declaration, "stop": observation, _UNSPOKEN: {
                "ts": ts, "pass": True, "observation": observation}}
            utils.atomic_write_json(path, pending, trailing_newline=True)
            say_unspoken(path, pending, late=True, event_queue=event_queue)
            return
    except (KeyError, OSError, ValueError):
        pass  # never remove earlier custody after a failed write; the sinks may still accept this row
    _told(drive, observation, ts, own=True, event_queue=event_queue)


def retract_unstarted(ctx: ToolContext, record: Any) -> None:
    """As its admission ends, a start that spawned nothing (no process of it, no backend submit begun) takes its
    declaration back: it names no launch and owes nothing. An unknown start (``start: unconfirmed``) or a launch that runs keeps it."""
    if getattr(record, "proc", None) or getattr(record, "local_proc", None) or getattr(record, "backend_log_path", ""):
        return
    if (reservation := getattr(record, "unwritten_outputs", None)) is not None:
        reservation.release()
        record.unwritten_outputs = None
    if getattr(record, "kept_declared", False):
        with contextlib.suppress(OSError, ValueError):
            drive, task, launch = _declaration_address(ctx, record)
            _consume(declaration_path(drive, task, launch, owed=bool(record.outputs)))
        record.kept_declared = False


def retire_kept_declaration(drive_root: pathlib.Path, task_id: str, path: pathlib.Path, cutoff: float,
                             report: Dict[str, Any]) -> None:
    """Retention of a kept launch's declaration: an OWED one (``_owes``) is never retired here; it stays, counted in
    ``report["capture_owed"]``, until a complete capture consumes it. One that names no outputs, past retention and
    once nothing of its task may still run, moves into one ``kept_service_declaration_retired`` row and the file goes."""
    from ouroboros import body_candidate

    try:
        st = path.lstat()
        if not stat.S_ISREG(st.st_mode):
            return
        declaration = _read(path)
        if isinstance(declaration, dict) and (_UNSPOKEN in declaration or files_settled(declaration)):
            # This visit already reads the record: it says the row its room is still owed, under the lock every
            # reader of the record takes, and retires a record whose files are settled once its room is told.
            # No stop, no capture, no other scan; an owed file keeps the ordinary answer below.
            gone = False
            try:
                with service_name_hold(drive_root, task_id, str(declaration.get("name") or "")):
                    declaration = _read(path)  # as it is now, under its lock; unreadable is owed, below
                    if isinstance(declaration, dict):
                        declaration = say_unspoken(path, declaration, late=st.st_mtime < cutoff)
                        if declaration is not None and _UNSPOKEN not in declaration and files_settled(declaration):
                            _consume(path)
                            declaration = None
                        gone = declaration is None
            except KeptCaptureBusy:
                pass
            if gone or files_settled(declaration):
                report["deleted_files" if gone else "retained_files"] += 1
                return
        if _owes(path, declaration):
            report["retained_files"] += 1
            report.setdefault("capture_owed", []).append(path.name[:-len(KEPT_DECLARATION)])
            return
        if st.st_mtime >= cutoff:
            return
        if body_candidate.live_lineage_processes(drive_root, task_id):
            report["retained_files"] += 1
            return
        if append_jsonl(drive_root / "logs" / "events.jsonl", {
                "ts": utc_now_iso(), "type": "kept_service_declaration_retired", "task_id": task_id,
                "launch_id": declaration.get("launch_id"), "name": declaration.get("name")}):
            _consume(path)
            report["deleted_files"] += 1
    except FileNotFoundError:
        return
    except Exception as exc:  # noqa: BLE001 — kept for a reader
        report["errors"].append(f"{path}: {type(exc).__name__}: {exc}")


def retire_kept_declarations(drive_root: pathlib.Path, cutoff: float, report: Dict[str, Any]) -> None:
    """The existing service-log pass visits canonical declarations once, with their actual producer ids."""
    root = installation_root(drive_root)
    if root != drive_root.resolve():
        return
    for path in _outstanding_files(root):
        retire_kept_declaration(root, path.parent.name, path, cutoff, report)
