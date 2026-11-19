"""A service launch's declared outputs, while a capture is still owed.

A kept service (``start_service(keep_alive=True)``) outlives the task that started it, and that task's
in-memory service record is the only place its declared outputs, their start baseline and its capture context
lived. So its start also writes them under its own launch id (``declare_kept``): a declaration naming outputs in
the installation's one directory of outstanding declarations (``services/kept_outputs/<task>/<launch>.kept.json``,
which the hot readers enumerate, so they cost what is owed and never task history), one naming none beside its log.
An ordinary launch uses the same directory only when its stop confirmed exit but left an output unsettled.
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
of what was already spoken. No timer ends a debt: until it is settled, retirement keeps the
declaration and the drive, copy or session scratch holding its working directory or outputs (``owed_output_hold``,
read by the retirement gates, each operation's candidates sharing one read of ``owed_declarations``). A declaration
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
    reservation = getattr(record, "unwritten_outputs", None)
    return reservation.context if reservation is not None else ctx


def pending_output_stop(record):
    return getattr(record, "stopped_outputs", None)


_name_holds = threading.local()


def _name_lock_path(root, task_id, name):
    from ouroboros.owned_shutdown import installation_root

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
    from ouroboros.owned_shutdown import installation_root

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
    """Whether a read declaration says what it holds, as ``declare_kept`` writes it: its working directory and outputs."""
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
    drive its capture writes into."""
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
    if not launch_id:
        from ouroboros.process_custody import custody_record_id

        proc = getattr(record, "proc", None) or getattr(record, "local_proc", None)
        launch_id = custody_record_id({"pid": getattr(proc, "pid", None), "scope_marker": getattr(record, "scope_marker", "")})
    return getattr(record, "drive_root", None) or ctx.drive_root, record.task_id, launch_id


def finish_own_outputs(ctx, record, payload, finish):
    """A confirmed stop: capture and settle the same declaration, or leave it with the producer's frozen inputs."""
    from ouroboros.service_outcomes import after_stop_error, output_stop_snapshot

    producer = output_producer(ctx, record)
    if not record.outputs and not record.keep_alive:
        return finish(producer, record, payload)
    registered = False
    try:
        drive, task, launch = _declaration_address(ctx, record)
        path = declaration_path(drive, task, launch, owed=bool(record.outputs))
        with service_name_hold(drive, task, record.name):
            if getattr(record, "kept_declared", False):
                try:
                    path.stat()
                except FileNotFoundError:  # an outside capture consumed it; do not read a future writer's files
                    payload["output_settled_elsewhere"] = True
                    return False
            try:
                registered = finish(producer, record, payload)
            except Exception as exc:
                after_stop_error(payload, record.outputs, exc)
            if payload.get("artifact_output_owed"):
                problem = declare_kept(producer, record, stop={**output_stop_snapshot(payload, record), "launch_id": launch})
                if problem:
                    payload.update(problem)
                    return registered
            elif getattr(record, "kept_declared", False):
                _consume(path)
            if getattr(record, "unwritten_outputs", None) is not None:
                record.unwritten_outputs.release()
                record.unwritten_outputs = None
    except (OSError, ValueError) as exc:
        after_stop_error(payload, record.outputs, exc)
        if getattr(record, "kept_declared", False):
            payload["artifact_output_owed"] = bool(record.outputs)  # the outstanding declaration was not retired
    return registered


def release_after_stop(record, payload, records, lock):
    """Leave a by-name handle for a retry; the same declaration lets another worker recover the stopped launch."""
    with lock:
        if records.get(record.service_id) is record:
            if payload is not None and payload.get("artifact_output_owed"):
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
        if getattr(record, "kept_declared", False):
            saved = confirmed_output_stop(*_declaration_address(ctx, record))
            if saved:
                payload["output_outcomes"] = saved.get("output_outcomes")
        payload["output_retry"] = True
        for field in ("artifact_output_owed", "artifact_output_failed", "artifact_output_unknown",
                      "artifact_outputs", "kept_declaration_problem"):
            payload.pop(field, None)
        registered = finish_own_outputs(ctx, record, payload, finish)
        release_after_stop(record, payload, records, lock)
        return (None, False) if payload.get("output_settled_elsewhere") else (payload, registered)


def _consume(path: pathlib.Path) -> None:
    path.unlink(missing_ok=True)
    try:
        path.parent.rmdir()  # an empty producer directory must not become another historical census
    except OSError:
        pass  # another declaration or an unfinished concurrent write keeps this directory


def unconsumed_declaration(root: Any, launch_id: str) -> tuple | None:
    """``(drive, task_id)`` of the owed declaration of ``launch_id`` (a validated launch id) no complete capture
    consumed, or None. Read in the one outstanding directory, never a walk; a directory that cannot be listed raises
    (absence read from a partial listing would be a guess). One naming no outputs owes nothing."""
    from ouroboros.owned_shutdown import installation_root

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
            try:
                path.stat()
            except FileNotFoundError:
                if answer["state"] == "missing":
                    gap = restart_gaps({"artifact_output_failed": True})
                elif answer["state"] in ("captured", "none_declared") and gap is None:
                    gap = {}
                continue
            raise KeptNamesake("owed", {}, f"{launch_id} still owes declared outputs ({answer['state']})")
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
                saved = confirmed_output_stop(drive, task, launch)
            except (OSError, ValueError) as exc:
                unreadable = exc
                continue
            if saved and saved.get("name") == name:
                result = capture_kept_service_outputs(drive, task, launch, actor=f"agent:{task}",
                    reason="capture after confirmed exit", verify_exit=True, checkpoint=False)
                return {**saved, **(result.get("observation") or {"artifact_output_failed": True,
                        "artifact_output_unknown": True, "artifact_output_owed": True}), "output_retry": True}
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


def declare_kept(ctx: ToolContext, record: Any, reservation=None, *, stop=None) -> Dict[str, str]:
    """Pin the producer's capture inputs. Publish at a kept start, or when a confirmed stop owes outputs."""
    if not (record.keep_alive or record.outputs):
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
        if not record.keep_alive and stop is None:  # no stopped debt while an ordinary process still writes
            if reservation is not None:
                producer = _producer_context({"context": context, "task_id": task_id})
                record.unwritten_outputs = _UnwrittenDeclaration(producer, weakref.finalize(record, reservation.pop_all().close))
            return {}
        path = declaration_path(drive, task_id, launch_id, owed=bool(record.outputs))
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, {"schema": 2, "launch_id": launch_id, "task_id": task_id, "declared_at": utc_now_iso(),
                                 "name": record.name, "cwd": str(getattr(record, "host_cwd", None) or record.cwd),
                                 "cwd_root": record.cwd_root, "cwd_base": record.cwd_base, "cwd_source": record.cwd_source,
                                 "skill_name": record.skill_name, "outputs": record.outputs,
                                 "before_outputs": {key: list(value) for key, value in record.before_outputs.items()},
                                 "keep_alive": bool(record.keep_alive), **({"stop": stop} if stop else {}),
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
    this call did not or could not probe, which proves no end."""
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
    if verify_exit and pid:
        # Publishing the process record or its ownership copy may have failed.
        state = backend._docker_pid_state(executor.container_name, pid)
        if state != "exited":
            raise KeptNamesake(state, {"launch_id": declaration["launch_id"], "state": state},
                               "the original Docker group is not confirmed ended")
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
                                 checkpoint: bool = True) -> Dict[str, Any]:
    """A confirmed exit: capture into its producer's store. A competing capture may already have consumed it."""
    try:
        _, selected = _load_declaration(drive_root, task_id, launch_id)
        with service_name_hold(drive_root, task_id, selected["name"]):
            path, declaration = _load_declaration(drive_root, task_id, launch_id)
            if selected["name"] != declaration["name"]:
                raise KeptDeclarationUnreadable("the saved service name changed")
            confirmed = _confirmed_stop(declaration)
            if not confirmed and not declaration.get("keep_alive", True):
                return {"task_id": task_id, "state": "not_attempted", "detail": "the launch's exit is not confirmed"}
            ended = False  # asking to verify observes nothing: only the read below, made whole, does
            if verify_exit and not confirmed:
                from ouroboros.launch_custody import _kept_selection

                found, complete, note = _kept_selection(drive_root, launch_id=launch_id, task_id=task_id)
                if found or not complete:
                    launch = found[0][0] if found else {}
                    return {"task_id": task_id, "state": "not_attempted", "launch": launch,
                            "detail": f"the previous service is {launch.get('state', 'unknown')}; {note}"}
                ended = True
            return _capture(path, declaration, actor, legacy_target, ended, checkpoint)
    except FileNotFoundError:
        return {"task_id": task_id, "state": "unavailable", "detail": "this launch recorded no output declaration"}
    except (OSError, ValueError) as exc:
        return {"task_id": task_id, "state": "unavailable", "detail": f"its output declaration cannot be read ({type(exc).__name__}); it is kept"}


def _file_facts(rows):
    """What a capture says of the files; which call registered a file is no news of the file."""
    return [{key: value for key, value in row.items() if key != "registered"} for row in rows or ()]


def _capture(path, declaration, actor, legacy_target, ended, checkpoint):
    """``ended``: this call's own complete custody read found no process of the launch (never the request to verify)."""
    from ouroboros.service_outcomes import declared_output_outcomes, public, public_outputs
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
            settled = {row["index"] for row in payload["output_outcomes"] or ()
                       if row["status"] in {"saved", "partial", "absent"}}
            context = prepare_docker_outputs(executor, pathlib.Path(cwd),
                [item for i, item in enumerate(outputs, 1) if i not in settled], pinned=True)
            before = {key: tuple(value) for key, value in (declaration.get("before_outputs") or {}).items()}
            registered = _capture_service_outputs(ctx, payload, before, context, executor=executor)
            notes = payload.get("artifact_outputs", "")
            state = ("incomplete" if payload.get("artifact_output_owed") else
                     "missing" if payload.get("artifact_output_failed") else "captured")
        except KeptNamesake as exc:
            return {"task_id": task_id, "state": "not_attempted", "launch": exc.launch, "detail": str(exc)}
        except Exception as exc:  # noqa: BLE001 — said, never a clean capture
            notes, state = f"Declared output capture could not be confirmed ({type(exc).__name__}).", "incomplete"
    # The first confirmed end no stop snapshot attests, and this capture its only speaker: the custody read
    # above observed it and the capture target's own inspection agreed (a Docker group answered its probe).
    # An inspection that was not made, or failed before it answered, creates no exit: only file facts speak.
    first_end = stop is None and ended and probed
    outcomes = public_outputs(declared_output_outcomes(outputs, payload.get("output_outcomes")))
    observation = public({**(stop or {}),
        **({"state": "exited", "process_cleanup": {"exit_confirmed": True}} if first_end else {}),
        "task_id": task_id, "launch_id": launch_id, "name": declaration["name"],
        "lifecycle": "stopped" if first_end else "outputs_updated", "output_retry": not first_end,
        "stop_source": "retained",
        "log_outcome": (stop or {}).get("log_outcome", "not_attempted"), "output_outcomes": outcomes,
        "artifact_output_failed": state in ("incomplete", "missing"),
        "artifact_output_owed": state == "incomplete", "artifact_output_unknown": state == "incomplete"})
    row = {"ts": utc_now_iso(), "type": "kept_service_outputs", "task_id": task_id, "launch_id": launch_id,
           "name": observation.get("name"), "actor": actor, "state": state, "output_outcomes": outcomes}
    facts = declaration.get("context") if isinstance(declaration.get("context"), dict) else {}
    recorded = append_jsonl(pathlib.Path(str(facts["drive_root"])) / "logs" / "events.jsonl", row)
    if recorded and state in ("captured", "none_declared", "missing"):
        if ended_record is not None:
            # Natural Docker exit leaves its record behind. A successor can reuse that group's PID.
            from ouroboros.workspace_executor import _forget_process

            _forget_process(ended_record)
            try:
                ended_record.stat()
            except FileNotFoundError:
                pass
            else:
                return {"task_id": task_id, "state": "incomplete", "registered": registered,
                        "detail": "Captured outputs, but the old Docker service record still holds its name.",
                        "output_outcomes": outcomes}
        try:
            _consume(path)  # consumed: its facts now live in that row
        except OSError:
            state, notes = "incomplete", f"{notes}\nThe declaration could not be retired."
    elif not recorded:
        state, notes = "incomplete", f"{notes}\nThe capture receipt could not be recorded."
    if state == "incomplete":
        notes = (f"{notes.strip()}\nIts output declaration and the files it names are kept: once the cause is fixed, "
                 f"stop_service(service_id={launch_id}, reason=...) captures them again.")
        observation.update(artifact_output_failed=True, artifact_output_owed=True, artifact_output_unknown=True)
    # Spoken: the first confirmed end; file results that differ from the ones the declaration's stop snapshot
    # already spoke; or, with no end proven and no snapshot to remember a partial result, the files of a settled
    # capture, once (its declaration is consumed). Equal facts are no news; the receipt row above is their record.
    spoken = first_end or (_file_facts(outcomes) != _file_facts(stop.get("output_outcomes")) if stop is not None
                           else bool(outputs) and state != "incomplete")
    if recorded and spoken and state == "incomplete":
        # The declaration survives: it must remember the end and the file facts already spoken, so
        # a later capture speaks only new file results and never a second stop.
        try:
            atomic_write_json(path, {**declaration, "stop": dict(observation) if first_end
                                     else {**stop, "output_outcomes": outcomes}}, trailing_newline=True)
        except OSError:
            spoken = False  # what is not remembered is not spoken: a later capture says it, once
            notes += "\nThe updated declaration could not be written; the earlier one remains."
    if recorded and checkpoint and spoken:
        from ouroboros.service_finalization import publish_retained_outputs

        publish_retained_outputs(pathlib.Path(str(facts["drive_root"])), observation)
    return {"task_id": task_id, "state": state, "registered": bool(registered), "detail": notes.strip(),
            "output_outcomes": outcomes, "observation": observation}


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
    from ouroboros.owned_shutdown import installation_root

    root = installation_root(drive_root)
    if root != drive_root.resolve():
        return
    for path in _outstanding_files(root):
        retire_kept_declaration(root, path.parent.name, path, cutoff, report)
