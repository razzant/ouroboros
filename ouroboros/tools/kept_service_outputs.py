"""A kept service's declared outputs, for a stop that is not its own task's.

A kept service (``start_service(keep_alive=True)``) outlives the task that started it, and that task's
in-memory service record is the only place its declared outputs, their start baseline and its capture context
lived. So its start also writes them under its own launch id (``declare_kept``): a declaration naming outputs in
the installation's one directory of outstanding declarations (``services/kept_outputs/<task>/<launch>.kept.json``,
which the hot readers enumerate, so they cost what is owed and never task history), one naming none beside its log.
A stop of that launch by the owner or another task (``launch_custody.stop_kept_service``) replays the ordinary capture
(``shell._register_process_outputs``) in the starting task's context, into ITS artifact store, never the
stopper's (``capture_kept_service_outputs``). It is OWED while it names outputs no complete capture consumed:
its own task's successful stop by name consumes it as well (``consume_own_declaration``). After an incomplete or
unattempted capture, or once the launch ended by itself or with its server, a stop of the same launch id finds it
(``unconsumed_declaration``) and captures again. No timer ends that: until a complete capture, retirement keeps the
declaration and the drive, copy or session scratch holding its working directory or outputs (``owed_output_hold``,
read by the retirement gates, each operation's candidates sharing one read of ``owed_declarations``). A declaration
that cannot say what it holds, or a directory of them that cannot be listed, stops the retirement operation reading it
at that operation's own error boundary, deleting nothing: it could name anything. The inventory lists the launch as
``capture_owed``. A declaration that names no outputs owes nothing and is retired with its logs into an event row
(``retire_kept_declaration``). Never the command or its environment.
"""

from __future__ import annotations

import json
import os
import pathlib
import stat
from typing import Any, Dict

from ouroboros.tool_access import active_tool_profile, canonical_data_root
from ouroboros.tools.registry import ToolContext
from ouroboros.utils import append_jsonl, utc_now_iso


KEPT_DECLARATION = ".kept.json"
KEPT_DIRECTORY = "kept_outputs"


class KeptDeclarationUnreadable(OSError):
    """An outstanding declaration cannot say what it holds (unreadable, or not what ``declare_kept`` wrote): the debts
    are not known whole, so a retirement reading them stops at its own error boundary, deleting nothing."""


def declaration_path(drive_root: Any, task_id: str, launch_id: str, *, owed: bool = True) -> pathlib.Path:
    """Where this launch's declaration lives: ``owed`` (it names outputs) in the installation's one directory of
    outstanding declarations, whatever drive started it (its producer's capture context travels inside, so placement
    never changes whose artifact store it lands in); else beside its log, where only retention visits it."""
    from ouroboros.owned_shutdown import installation_root

    name = f"{launch_id}{KEPT_DECLARATION}"
    if not owed:
        return pathlib.Path(drive_root) / "services" / str(task_id) / name
    return installation_root(drive_root) / "services" / KEPT_DIRECTORY / str(task_id) / name


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
    from ouroboros.owned_shutdown import installation_root

    return [pathlib.Path(file.path) for producer in _entries(installation_root(root) / "services" / KEPT_DIRECTORY)
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


def executor_launch(executor_key: str) -> tuple:
    """``(drive, task_id, launch_id)`` of the executor service ``executor_key`` as its start declared it."""
    from ouroboros.process_custody import custody_record_id
    from ouroboros.workspace_executor import _services_snapshot

    record = next(item for item in _services_snapshot() if item.service_id == executor_key)
    return record.drive_root, record.task_id, (
        custody_record_id({"pid": int(record.backend_pid), "scope_marker": record.scope_marker})
        if record.scope_marker else pathlib.Path(record.durable_record_path).stem)


def consume_own_declaration(launch: tuple) -> None:
    """``launch`` = ``(drive, task_id, launch_id)``: its own task's stop by name captured this kept launch's outputs
    completely, so its declaration owes nothing. A failed unlink leaves it owed: an outside stop of this launch id
    captures again, and an unchanged output is not duplicated."""
    for owed in (True, False):  # it is in exactly one of the two places
        try:
            _consume(declaration_path(*launch, owed=owed))
        except OSError:
            pass


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
    for path in _outstanding_files(root):
        if path.name == name:
            return installation_root(root), path.parent.name
    return None


def declare_kept(ctx: ToolContext, kept: Dict[str, Any] | None, *, launch: tuple = (), executor_key: str = "") -> Dict[str, str]:
    """A kept service's start writes, beside its log under its own launch id, what a stop by the owner or another
    task needs to capture its declared outputs as this task's own stop would (``capture_kept_service_outputs``):
    the declarations, their start baseline and this task's capture context. Never the command or its environment.
    ``{}``, or the problem, disclosed in the start result: the service runs either way."""
    if kept is None:
        return {}
    from ouroboros.process_custody import custody_record_id
    from ouroboros.utils import atomic_write_json

    try:
        if executor_key:
            drive, task_id, launch_id = executor_launch(executor_key)
        else:
            drive, task_id, pid, marker = launch
            launch_id = custody_record_id({"pid": pid, "scope_marker": marker})
        metadata = getattr(ctx, "task_metadata", None) if isinstance(getattr(ctx, "task_metadata", None), dict) else {}
        contract = getattr(ctx, "task_contract", None) or metadata.get("task_contract")
        ceiling = contract.get("capability_ceiling") if isinstance(contract, dict) else None
        executor_ref = getattr(ctx, "workspace_executor_ref", None)
        context = {"repo_dir": str(ctx.repo_dir), "system_repo_dir": str(getattr(ctx, "system_repo_dir", "") or ""),
                   "drive_root": str(ctx.drive_root), "budget_drive_root": str(canonical_data_root(ctx)),
                   "workspace_root": str(getattr(ctx, "workspace_root", "") or ""),
                   "workspace_mode": str(getattr(ctx, "workspace_mode", "") or ""),
                   "tool_profile": str(active_tool_profile(ctx)),
                   "executor_ref": executor_ref() if callable(executor_ref) else {},
                   **({"capability_ceiling": ceiling} if ceiling is not None else {})}
        path = declaration_path(drive, task_id, launch_id, owed=bool(kept["outputs"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, {"schema": 1, "launch_id": launch_id, "task_id": task_id, "declared_at": utc_now_iso(),
                                 **{**kept, "before_outputs": {key: list(value) for key, value in
                                                               kept["before_outputs"].items()}},
                                 "context": context}, trailing_newline=True)
        return {}
    except Exception as exc:  # noqa: BLE001 — disclosed: a later outside stop then says it captured nothing
        return {"kept_declaration_problem": (f"{type(exc).__name__}: {exc}"[:300] + "; a stop by the owner or "
                                             "another task cannot capture this service's declared outputs")}


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


def capture_kept_service_outputs(drive_root: Any, task_id: str, launch_id: str, *, actor: str,
                                 reason: str) -> Dict[str, Any]:
    """After a CONFIRMED stop of a kept launch by the owner or another task: its declared outputs, captured as the
    task that started it would have (``_register_process_outputs`` in ITS context, into ITS artifact store), never
    as the stopper's. One ``kept_service_outputs`` row on that drive names the launch, who stopped it and why, and
    then holds the declaration's facts. ``state``: ``captured``, ``incomplete`` (its notes name what was not; the
    declaration stays, so a later stop of this launch id captures again: an unchanged capture is not duplicated),
    ``none_declared``, or ``unavailable``: no declaration was recorded (a start before kept services recorded one,
    or a failed write), so nothing is captured and its working directory and log stay as they are."""
    from ouroboros.tools.services import _baseline_gap_note, _service_output_binding, _working_tree_changes
    from ouroboros.tools.shell import _register_process_outputs
    from ouroboros.tools.shell_outputs import _prepare_declared_output_context

    places = [declaration_path(drive_root, task_id, launch_id, owed=owed) for owed in (True, False)]
    path = next((place for place in places if place.exists()), places[0])
    try:
        declaration = json.loads(path.read_text(encoding="utf-8"))
        if declaration.get("launch_id") != launch_id or declaration.get("task_id") != task_id:
            raise ValueError("it names another launch")
        if _owes(path, declaration) and not _names_places(declaration):
            raise ValueError("it does not name its working directory and outputs")
    except FileNotFoundError:
        return {"task_id": task_id, "state": "unavailable", "detail": (
            "this launch recorded no output declaration (it started before kept services recorded one, or its "
            "record was not written); its working directory and log stay as they are")}
    except Exception as exc:  # noqa: BLE001 — kept for a reader; nothing is guessed from it
        return {"task_id": task_id, "state": "unavailable",
                "detail": f"its output declaration cannot be read ({type(exc).__name__}: {exc}); it is kept"}
    outputs = [str(item) for item in declaration.get("outputs") or []]
    notes, registered, state = "", False, "none_declared"
    if outputs:
        try:
            ctx, cwd = _producer_context(declaration), str(declaration.get("cwd") or "")
            binding = _service_output_binding(
                ctx, cwd_root=str(declaration.get("cwd_root") or ""), cwd_base=str(declaration.get("cwd_base") or ""),
                cwd=cwd, cwd_source=str(declaration.get("cwd_source") or ""),
                skill_name=str(declaration.get("skill_name") or ""))
            output_context, refused = _prepare_declared_output_context(ctx, outputs, pathlib.Path(cwd))
            if refused is not None:
                raise RuntimeError(refused.text)
            before = {key: tuple(value) for key, value in (declaration.get("before_outputs") or {}).items()}
            notes, failed, registered = _register_process_outputs(
                ctx, outputs, pathlib.Path(cwd), cwd_root=str(declaration.get("cwd_root") or ""),
                changed_paths=_working_tree_changes(cwd), before_outputs=before, binding=binding,
                output_context=output_context)
            notes += _baseline_gap_note(outputs, before)
            state = "incomplete" if failed else "captured"
        except Exception as exc:  # noqa: BLE001 — said, never a clean capture
            notes, state = f"output capture failed: {type(exc).__name__}: {exc}", "incomplete"
    row = {"ts": utc_now_iso(), "type": "kept_service_outputs", "task_id": task_id, "launch_id": launch_id,
           "name": declaration.get("name"), "actor": actor, "reason": reason, "state": state,
           "notes": notes.strip(), "declaration": declaration}
    facts = declaration.get("context") if isinstance(declaration.get("context"), dict) else {}
    recorded = append_jsonl(pathlib.Path(str(facts.get("drive_root") or drive_root)) / "logs" / "events.jsonl", row)
    if recorded and state in ("captured", "none_declared"):
        _consume(path)  # consumed: its facts now live in that row
    elif state == "incomplete":
        notes = (f"{notes.strip()}\nIts output declaration and the files it names are kept: once the cause is fixed, "
                 f"stop_service(service_id={launch_id}, reason=...) captures them again.")
    return {"task_id": task_id, "state": state, "registered": bool(registered), "detail": notes.strip()}


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
                "launch_id": declaration.get("launch_id"), "name": declaration.get("name"), "declaration": declaration}):
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
