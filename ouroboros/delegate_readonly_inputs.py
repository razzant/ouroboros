"""Materialize selected or automatic native inputs for a folderless session request.

The engine accepts one readonly project scope, not additional read roots. Give
each invocation its own scratch with verified input copies (``_lineage_copies``:
bytes only, cloned where the volume can, streamed where compression may later
want the source); never pass parent scratch or external runtime data as the
engine's project. A plan-thread turn stages nothing: its thread keeps its first
folder. These copies confer no workspace/write authority. Native and session
inputs use one access owner.

``input_files`` (#2593) is the caller's explicit selection of ordinary files it
may already read, each a ``{root, path}`` pair resolved and checked as ``read_file``
resolves and checks it. Before the request is sent or the child queued, each file's
exact bytes are retained in the selecting task's canonical artifact store
(``source_handles/delegated_readonly_inputs/<invocation|scheduled-<child>>/``, named by
content, ``inputs/selected/<sha256[:16]>/<name>``): the existing snapshot home that is no
deliverable, outlives the task drive, and whose name the lineage walker prunes.
A nonempty selection replaces the automatic lineage copies; a folderless start restages
it once into its scope; a start in a repository or folder
keeps it as scope and cwd and names the retained copies by absolute path:
the engine has no readable-root field, so reaching them is the harness's native read
reach, not a host guarantee. A scheduled API child's first input names its copies for
``read_file``, its parent's store being lineage; a session child's first start verifies
them. A prose path is never a selection; ``attachment_paths`` remains the separate
engine-document import. Preparation is not a read.
"""
from __future__ import annotations

import copy
import json
import os
import pathlib
import stat
from typing import Any, Dict, List, NamedTuple, Optional

CONTAINER = "delegated_readonly_inputs"
SELECTED_DIR = "inputs/selected"
ADDRESS_HEX = 16  # the content folder is the sha256's first 16 hex (receipts keep all 64): ~48 fewer path characters
# ``read_file``'s ordinary-file roots: ``skill_payload`` needs a bucket and skill name this selection has no field for.
INPUT_FILE_ROOTS = ("active_workspace", "system_repo", "runtime_data", "task_drive", "artifact_store",
                    "user_files", "subagent_projects", "deliverables")
INPUT_FILES_CONTRACT = (
    "Files you can already read that this read-only run needs, each {\"path\": ..., \"root\": ...}. root is one of "
    + ", ".join(INPUT_FILE_ROOTS) + " (skill_payload and others refuse); omitted, it is inferred from an absolute path, "
    "else active_workspace. Every entry is mandatory: each is checked as read_file checks it and its exact bytes are "
    "retained in your task's artifact store before the run starts, and one that cannot be prepared refuses the whole "
    "start typed, unrun. The receipt covers exactly these entries (not the automatic inputs); prepared is not read. A "
    "nonempty selection replaces automatic folderless task/artifact lineage copies; omitted, null or [] keeps them. A "
    "folderless run also finds the copies inside its folder; a run in a repository or folder keeps it and gets the "
    "retained copies' absolute paths outside it, reached by its harness's own native read reach. A plan-thread "
    "turn keeps its original folder and names new copies outside it. A path named only "
    "in the prompt is never copied. Omit (or use null or []) for the existing automatic inputs. "
    "Not engine documents: those are attachment_paths.")


class InputSelectionError(ValueError):
    """A typed refusal before anything is sent; ``results`` is one row per selected entry."""

    def __init__(self, code: str, detail: str, results: Optional[List[Dict[str, Any]]] = None) -> None:
        super().__init__(detail)
        self.code, self.detail, self.results = code, detail, list(results or [])


class ReadonlyInputs(NamedTuple):
    scope_root: str       # the folderless scope ("" when the run keeps its own root)
    instruction: str      # appended to the run's instructions
    selected: Optional[List[Dict[str, Any]]]  # the receipt: None when nothing was asked, else one row per entry
    selection_note: str   # the selection's own text, restated in a continuation's prompt

    @property
    def custody(self) -> Dict[str, Any]:
        """The durable source map on the start's START_REQUESTED row."""
        return {"input_files": self.selected} if self.selected is not None else {}  # an explicit [] is a fact too


NO_INPUTS = ReadonlyInputs("", "", None, "")


def input_files_parameter(scope: str) -> Dict[str, Any]:
    """The ``input_files`` schema delegate_start and schedule_subagent share. The item schema is typed
    for both: a compact custom-tool projection drops the description but keeps this structure, which
    ``selection_entries`` and ``resolve_selection`` enforce again before anything happens."""
    items = {"type": "object", "required": ["path"], "additionalProperties": False, "properties": {
        "path": {"type": "string"}, "root": {"type": "string", "enum": list(INPUT_FILE_ROOTS)}}}
    return {"input_files": {"type": "array", "items": items, "description": f"{INPUT_FILES_CONTRACT} {scope}"}}


def selection_entries(value: Any) -> Optional[List[Dict[str, str]]]:
    """The selection's shape: ``None`` when omitted, else exact ``{root, path}`` entries."""
    if value is None:
        return None
    if not isinstance(value, list) or any(
            not isinstance(item, dict) or set(item) - {"root", "path"}
            or not isinstance(item.get("path"), str) or not item["path"].strip() or "\x00" in item["path"]
            or not isinstance(item.get("root", ""), str) for item in value):
        raise InputSelectionError("input_files_invalid",
                                  "input_files is a list of {root, path} objects with a nonempty path.")
    return [{"root": str(item.get("root") or "").strip(), "path": item["path"]} for item in value]


def receiving_reader(ctx: Any, resource_policy: Any = None) -> Any:
    """The read-only recipient's limits ON TOP of the caller's contract (never instead of its constraint:
    every entry must pass both); a scheduled child adds its own ``resource_policy``. Readers differ in
    which contract carrier they prefer (``protected_artifacts._task_contract`` takes a nonempty
    ``task_metadata['task_contract']``), so the policy replaces each carrier's, never one left stale."""
    reader = copy.copy(ctx)
    reader.task_constraint = {"mode": "local_readonly_subagent"}
    if resource_policy:
        contract = getattr(ctx, "task_contract", None)
        reader.task_contract = {**(contract if isinstance(contract, dict) else {}), "resource_policy": resource_policy}
        metadata = getattr(ctx, "task_metadata", None)
        if isinstance(metadata, dict) and isinstance(metadata.get("task_contract"), dict) and metadata["task_contract"]:
            reader.task_metadata = {**metadata, "task_contract": {**metadata["task_contract"],
                                                                  "resource_policy": resource_policy}}
    return reader


def prepared_copy_read(ctx: Any, binding: Any) -> bool:
    """A read of exactly one retained copy this task was handed (``prepared_input_files``, host-set at
    admission), unchanged. Its source passed the same inherited Presence ceiling when selected, so the
    copy needs no artifact or runtime grant; any other path in that store stays outside the ceiling."""
    from ouroboros.artifacts import stream_artifact_file

    metadata = getattr(ctx, "task_metadata", None)
    rows = metadata.get("prepared_input_files") if isinstance(metadata, dict) else None
    if str(getattr(binding, "operation", "")) != "read" or not isinstance(rows, list):
        return False
    try:
        target = pathlib.Path(binding.target_path).resolve(strict=True)
        row = next(row for row in rows if isinstance(row, dict) and row.get("status") == "prepared"
                   and row.get("local") and pathlib.Path(row["local"]).resolve(strict=True) == target)
        return stream_artifact_file(target, expected={"sha256": row["sha256"], "size": row["size"]}) is not None
    except (StopIteration, OSError, ValueError, KeyError, TypeError, RuntimeError):
        return False


def resolve_selection(ctx: Any, entries: List[Dict[str, str]], receiving: Any = None) -> List[Dict[str, Any]]:
    """Admit every entry by ``read_file``'s own ``read_admission`` (Presence ceiling included) for the
    caller AND the receiving reader, before any copy."""
    from ouroboros.tools.core_file_tools import read_admission
    from ouroboros.tools.tool_resolution import _root_containing_absolute_path

    readers = (ctx, receiving if receiving is not None else receiving_reader(ctx))
    resolved, results = [], []
    for entry in entries:
        row = {"root": entry["root"], "path": entry["path"]}
        try:
            label = row["root"] = (entry["root"] or _root_containing_absolute_path(ctx, "read_file", entry["path"])
                                   or "active_workspace")
            if label not in INPUT_FILE_ROOTS:
                results.append({**row, "status": "unsupported_root",
                                "detail": f"input_files reads ordinary files from {', '.join(INPUT_FILE_ROOTS)}"})
                continue
            binding = None  # resolved for the caller; the receiver is judged on that same target
            for reader in readers:
                _label, binding, refusal = read_admission(reader, label, entry["path"], resolved_binding=binding)
                if refusal is not None:
                    raise PermissionError(getattr(refusal, "text", refusal))
            target = binding.target_path
        except Exception as exc:  # an escape, policy or resolver refusal: all are denials
            results.append({**row, "status": "denied", "detail": f"{type(exc).__name__}: {exc}"})
            continue
        try:
            if not target.exists():
                from ouroboros.evidence_packs import materialize_path

                materialize_path(target)  # a packed checkpoint handle reads back as read_file reads it
            mode = target.stat().st_mode
        except OSError as exc:
            results.append({**row, "status": "missing", "source": str(target), "detail": str(exc)})
            continue
        if not stat.S_ISREG(mode):
            results.append({**row, "status": "not_a_file", "source": str(target),
                            "detail": "only regular files are copied; select each file, or bind one directory with workspace_root"})
            continue
        resolved.append({**row, "source": str(target)})
        results.append({**row, "status": "checked", "source": str(target)})
    if len(resolved) != len(entries):
        raise InputSelectionError("input_files_unavailable", (
            "Some selected input_files cannot be prepared, so nothing was started. Fix or drop the entries "
            "whose status is not 'checked' and start again."), results)
    return resolved


def copy_selection(rows: List[Dict[str, Any]], root: Optional[pathlib.Path]) -> List[Dict[str, Any]]:
    """Copy each checked (or already retained) row to ``root/inputs/selected/<sha256[:16]>/<name>``.

    A retained row is verified against its recorded identity; ``root=None`` verifies it in
    place (no second copy). An occupied address keeps its bytes only when they are these exact
    bytes; anything else refuses, never overwritten. Any failure refuses the whole selection."""
    from ouroboros.artifacts import ArtifactIdentityError, copy_artifact_file, stream_artifact_file
    from ouroboros.artifact_storage import temporary_sibling

    receipt: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        expected = {"sha256": row["sha256"], "size": row["size"]} if row.get("local") else None
        base = {key: row[key] for key in ("root", "path", "source")}
        name = pathlib.PurePath(row["source"]).name
        try:
            if root is None:
                measured, local = stream_artifact_file(row["local"], expected=expected), pathlib.Path(row["local"])
            else:
                staging = temporary_sibling(root / SELECTED_DIR / name)
                origin = row.get("local") or row["source"]
                from ouroboros import work_domain

                if work_domain.is_work(origin):  # a work file's bytes: the streamed confined intake, never a service open
                    staging.parent.mkdir(parents=True, exist_ok=True)
                    with staging.open("xb") as sink:
                        measured = work_domain.measure(origin, sink)
                    if expected and any(expected[key] != measured[key] for key in ("sha256", "size")):
                        staging.unlink(missing_ok=True)
                        raise ArtifactIdentityError(f"selected input changed: {origin}")
                else:
                    measured = copy_artifact_file(origin, staging, expected=expected)
                local = root / SELECTED_DIR / measured["sha256"][:ADDRESS_HEX] / name  # hash-stable, never positional
                local.parent.mkdir(parents=True, exist_ok=True)
                if local.exists() or local.is_symlink():  # a repeated selection's same bytes stay; others refuse
                    staging.unlink()
                    try:
                        if local.is_symlink():
                            raise OSError("a link")
                        stream_artifact_file(local, expected={"sha256": measured["sha256"], "size": measured["size"]})
                    except OSError:
                        raise OSError(f"{local} already holds other bytes; nothing was overwritten") from None
                else:
                    staging.replace(local)
        except OSError as exc:
            status = "changed" if isinstance(exc, ArtifactIdentityError) else "copy_failed"
            raise InputSelectionError("input_files_unavailable", (
                "A selected input could not be prepared unchanged, so nothing was started; earlier rows were copied "
                "but are not delivered."), [*receipt, {**base, "status": status, "detail": str(exc)}, *(
                    {**{key: later[key] for key in ("root", "path", "source")}, "status": "not_attempted"}
                    for later in rows[index + 1:])]) from exc
        receipt.append({**base, "status": "prepared", "local": str(local),
                        "retained": str(row.get("retained") or local), **measured})
    return receipt


def schedule_selected_inputs(ctx: Any, entries: List[Dict[str, str]], child_task_id: str,
                             resource_policy: Any = None) -> List[Dict[str, Any]]:
    """Check and retain a scheduled child's selection before it is queued.

    The caller's read authority AND the child's read-only limits (with its own resource
    policy) must both admit each file. The bytes stay in the scheduling task's artifact store,
    which its direct child reads as lineage. Later refusals still retain these bytes with the task."""
    rows = resolve_selection(ctx, entries, receiving_reader(ctx, resource_policy))
    root = _retained_root(ctx, f"scheduled-{child_task_id}")
    root.mkdir(parents=True, exist_ok=False)
    receipt = copy_selection(rows, root)
    _write_manifest(root, [], receipt)
    return receipt


def start_inputs(ctx: Any, authority: Any, invocation_id: str, input_files: Any, prepared: Any,
                 thread_folder: Optional[str] = None):
    """``(inputs, refusal)`` for one new start: the explicit selection, else the default lineage staging.

    Only a read-only run (readonly access or an evidence-only investigation) stages inputs; a
    nonempty selection for a writable run refuses typed, while omission and [] never gate it.
    ``thread_folder``: a plan-thread turn's, the folder its engine thread keeps (``prepare_readonly_inputs``).
    Every refusal precedes the request row."""
    from ouroboros.delegate_shared import _fail

    try:
        entries = selection_entries(input_files)
        asked = entries is not None or bool(prepared)
        if not (authority.access == "readonly" or authority.investigation):
            if entries or prepared:
                return NO_INPUTS, _fail("delegate_start", "input_files_readonly_only", "input_files prepares inputs for a "
                                        "read-only run; this start can write. Omit input_files, or start it read-only.",
                                        definitely_unrun=True)
            return NO_INPUTS._replace(selected=[] if asked else None), None
        inputs = prepare_readonly_inputs(ctx, invocation_id, entries, prepared, thread_folder=thread_folder)
    except InputSelectionError as exc:
        return NO_INPUTS, _fail("delegate_start", exc.code, exc.detail, input_files=exc.results, definitely_unrun=True)
    except (OSError, ValueError) as exc:
        return NO_INPUTS, _fail("delegate_start", "readonly_inputs_unavailable", str(exc), definitely_unrun=True)
    return inputs._replace(selected=inputs.selected if asked else None), None


def prepare_readonly_inputs(ctx: Any, invocation_id: str, entries: Optional[List[Dict[str, str]]] = None,
                            prepared: Optional[List[Dict[str, Any]]] = None, *,
                            thread_folder: Optional[str] = None) -> ReadonlyInputs:
    """This invocation's inputs: the explicit selection, which replaces folderless lineage copies, else those copies.

    ``entries`` are this call's ``input_files``; ``prepared`` a scheduled child's
    schedule-time receipt for its first start; either one, nonempty, is the whole set (no lineage
    file is enumerated or copied). Without either the result equals the pre-selection behavior
    exactly. A plan-thread turn (``thread_folder`` given) runs in the folder
    its thread was created on, whose copies that start staged: the engine drops a turn's scope and
    no other field reaches a new folder, so the turn stages nothing, keeps that folder (as the
    engine's thread record names it) as its scope and registration identity, and names its
    selection by retained path, outside it. A folderless turn whose record names no folder ("")
    is refused before anything is retained, staged or registered: no other root stands in for it."""
    from ouroboros.tool_access import folderless_scratch_dir

    rows = resolve_selection(ctx, entries) if entries else list(prepared or [])
    scratch = folderless_scratch_dir(ctx)
    if scratch is None and not rows:
        return ReadonlyInputs("", "", [], "")
    if not invocation_id or pathlib.Path(invocation_id).name != invocation_id:
        raise ValueError("invalid readonly invocation identity")
    container = None if scratch is None else _owned_container(scratch)
    if container is not None and thread_folder == "":
        raise ValueError("The engine's record of this plan thread names no folder (repoRoot), so this read-only "
                         "turn's scope is unknown; nothing was retained, staged, registered or sent. Start a "
                         "fresh plan, or continue once the engine names the thread's folder.")
    if entries:
        retained_root = _retained_root(ctx, invocation_id)
        retained_root.mkdir(parents=True, exist_ok=False)
        rows = copy_selection(rows, retained_root)
        _write_manifest(retained_root, [], rows)
    turn = thread_folder is not None
    outside = [{**row, "carrier": "outside_scope_path"} for row in (rows if entries else copy_selection(rows, None))
               ] if container is None or turn else []  # the run keeps its folder and names the retained copies
    manifest = pathlib.Path(outside[0]["retained"]).parents[3] / "inputs.json" if outside else None
    beside = thread_folder or getattr(ctx, "workspace_root", "") or (manifest.parent if manifest else "")
    if outside and (delivered := _work_delivery(ctx, invocation_id, None, outside, beside)) is not None:
        outside = [{**row, "local": str(delivered / _selected_rel(row))} for row in outside]  # what the work reads
        manifest = delivered / "inputs.json"
    note = _selection_note(manifest, outside, inside_scope=False) if outside else ""
    if container is None:
        return ReadonlyInputs("", "\n" + note, outside, note)
    if turn:
        return ReadonlyInputs(thread_folder, "\n" + note if note else "", outside, note)
    root = container / invocation_id
    root.mkdir(parents=True, exist_ok=False)
    lineage = [] if rows else _lineage_copies(ctx, root)  # a nonempty selection is the whole set
    receipt = [{**row, "carrier": "scope"} for row in copy_selection(rows, root)]
    _write_manifest(root, lineage, receipt)
    if (delivered := _work_delivery(ctx, invocation_id, root, receipt, root)) is not None:  # the scope the engine reads
        receipt = [{**row, "local": str(delivered / pathlib.Path(row["local"]).relative_to(root))} for row in receipt]
        root = delivered
    note = note or (_selection_note(root / "inputs.json", receipt, inside_scope=True, scope=root) if receipt else "")
    return ReadonlyInputs(str(root), (
        "\nREADONLY INPUTS: This invocation has its own scratch scope, not a workspace grant. "
        "inputs.json maps the selected files, or default native lineage sources, to verified local copies. "
        "Read those local paths when the work order names the original sources. "
        "The manifest describes staged copies, not an additional filesystem-read boundary.\n"
    ) + note, receipt, note)


def _selected_rel(row: Dict[str, Any]) -> str:
    return f"{SELECTED_DIR}/{row['sha256'][:ADDRESS_HEX]}/{pathlib.Path(row['local']).name}"


def _work_delivery(ctx: Any, invocation_id: str, root: Optional[pathlib.Path], rows: List[Dict[str, Any]],
                   execution: Any) -> Optional[pathlib.Path]:
    """Managed (``work_domain``): this invocation's inputs materialized in the work domain, so the work engine reads
    their bytes there and nothing of the service. The folder is a session scratch its task's drive owns
    (``session_scratch_owners/inputs-<invocation>``, made by ``workspace_preflight._session_scratch`` apart from
    ``execution``): the custody pass retires it once that task is over or its drive settles, a producer's retention
    marker or an unlistable subtree holding it as any scratch's, and a plan-thread turn keeps naming it meanwhile.
    ``root``: the folderless scope the service staged, mirrored whole (``inputs.json`` included); else the selected
    rows' verified copies under ``inputs/selected`` with their own ``inputs.json``. One tar stream the container
    unpacks as the work domain. None unmanaged: the service's own copies stay what the run reads."""
    from ouroboros import work_domain

    if not work_domain.managed():
        return None
    from ouroboros.task_results import validate_task_id
    from ouroboros.tool_access_paths import canonical_data_root
    from ouroboros.workspace_preflight import (
        SESSION_SCRATCH_DIRNAME,
        _note_open_scratch,
        _owned_directory,
        _session_scratch,
    )

    task_id, key, execution = validate_task_id(str(getattr(ctx, "task_id", "") or "")), f"inputs-{invocation_id}", \
        work_domain.settle(execution)
    owner = _owned_directory(getattr(ctx, "drive_root", ""), ("task_drives", task_id, SESSION_SCRATCH_DIRNAME, key), execution)
    target = _session_scratch(owner, execution)
    _note_open_scratch(canonical_data_root(ctx), task_id, key, owner)
    members = []
    if root is not None:
        for folder, _dirs, files in os.walk(root, followlinks=False):
            for name in sorted(files):
                path = pathlib.Path(folder) / name
                if not path.is_symlink():
                    members.append((path.relative_to(root).as_posix(), path,
                                    stat.S_IMODE(path.stat().st_mode)))
    else:
        selected = []
        for row in rows:
            rel = _selected_rel(row)
            members.append((rel, pathlib.Path(row["local"]), 0o644))  # the service's own verified copy
            selected.append({"root": row["root"], "source": row["source"], "local": rel, "selected_path": row["path"],
                             "source_sha256": row["sha256"], "source_size": row["size"],
                             "sha256": row["sha256"], "size": row["size"]})
        members.append(("inputs.json", json.dumps(selected, ensure_ascii=False, indent=2).encode("utf-8"), 0o644))
    work_domain.import_tree(target, members)
    return target


def _write_manifest(root: pathlib.Path, lineage: List[Dict[str, Any]], receipt: List[Dict[str, Any]]) -> None:
    from ouroboros.utils import write_bytes_atomic

    selected = [{"root": row["root"], "source": row["source"], "local": pathlib.Path(row["local"]).relative_to(root).as_posix(),
                 "selected_path": row["path"], "source_sha256": row["sha256"], "source_size": row["size"],
                 "sha256": row["sha256"], "size": row["size"]} for row in receipt]
    write_bytes_atomic(root / "inputs.json", json.dumps(lineage + selected, ensure_ascii=False, indent=2).encode("utf-8"))


def _selection_rows(rows: List[Dict[str, Any]], scope: Optional[pathlib.Path] = None) -> str:
    def inside(row: Dict[str, Any]) -> str:
        return f"{pathlib.Path(row['local']).relative_to(scope).as_posix()} in this folder, host path " if scope else ""

    return "\n".join(f"- {row['root']}:{row['path']} -> {inside(row)}{row['local']} "
                     f"(sha256 {row['sha256']}, {row['size']} bytes)" for row in rows)


def _selection_note(manifest: Any, receipt: List[Dict[str, Any]], *, inside_scope: bool,
                    scope: Optional[pathlib.Path] = None) -> str:
    where = ("inside this run's folder" if inside_scope else
             "outside this run's folder, which this selection does not change; the host prepared them but cannot "
             "confirm this harness may read outside its folder; report an unreadable copy rather than guess its contents")
    return (f"SELECTED INPUT FILES: the parent selected these files; the host copied each byte-for-byte before this "
            f"run started, {where}. Manifest: {manifest}. Preparation is not a read: open the copies "
            f"you need.\n{_selection_rows(receipt, scope)}\n")


def native_selection_block(rows: Any) -> str:
    """A scheduled child's own reference to its retained copies, frozen into its first input; "" without any."""
    if not rows:
        return ""
    return ("[SELECTED INPUT FILES]\nThe parent selected these files (input_files); the host retained each one's exact "
            "bytes before queueing this task, so a copy still holds what was selected even if the original changed. Read "
            "a copy with read_file(path=<its absolute path below>); read_file reads UTF-8 text and reports a binary file "
            "as an error. Preparation is not a read.\n" + _selection_rows(rows))


def _retained_root(ctx: Any, folder: str) -> pathlib.Path:
    """This task's canonical retained home for one selection (``review_source_closure``'s snapshot pattern)."""
    from ouroboros.artifacts import task_artifact_dir_path, task_id_for_artifacts
    from ouroboros.tool_access import canonical_data_root

    return (task_artifact_dir_path(canonical_data_root(ctx), task_id_for_artifacts(ctx), create=True)
            / "source_handles" / CONTAINER / folder)


def _owned_container(base: pathlib.Path) -> pathlib.Path:
    container = pathlib.Path(base) / CONTAINER
    if container.is_symlink() or container.resolve(strict=False).parent != pathlib.Path(base).resolve():
        raise ValueError("readonly scratch container is not owned by this task")
    return container


def _lineage_copies(ctx: Any, root: pathlib.Path) -> List[Dict[str, Any]]:
    """Every permitted lineage file's exact bytes under ROOT, each its own inode (``artifacts.copy_artifact_file``):
    a clone where the volume makes one, captured whole at one instant, else a bounded stream that refuses a source
    changing while read. Bytes only: a copy gets ROOT's new-file mode, never the source's mode or times. A file a
    supported ``record_compression`` pass may compress (``in_scope``) streams until already compressed, since an APFS clone
    keeps its source shared; an already compressed record keeps its compression in the clone."""
    from ouroboros.artifact_storage import compression_supported, is_compressed
    from ouroboros.artifacts import copy_artifact_file
    from ouroboros.protected_artifacts import block_reason_for_path
    from ouroboros.record_compression import in_scope
    from ouroboros.tool_access import (
        active_tool_profile,
        build_resolved_resource_binding,
        canonical_data_root,
        decide_tool_access,
        lineage_read_roots,
    )
    from ouroboros.tools.core_file_tools import _runtime_data_read_check

    reader = copy.copy(ctx)
    reader.task_constraint = {"mode": "local_readonly_subagent"}
    manifest, data = [], canonical_data_root(ctx)
    preserve_compression = compression_supported()
    mode = stat.S_IMODE(root.stat().st_mode) & 0o666  # ROOT is this start's new folder: what a new file gets here
    runtime_check = _runtime_data_read_check(reader)
    for label in ("task_drive", "artifact_store"):
        if not decide_tool_access(profile=active_tool_profile(reader), root=label, operation="read").allow:
            continue
        for index, base in enumerate(lineage_read_roots(ctx, label)):
            try:
                if not stat.S_ISDIR(base.stat().st_mode):
                    raise OSError(f"lineage input container is not a directory: {base}")
            except FileNotFoundError:
                continue
            for source in _input_files(base):
                relative = source.relative_to(base)
                if source.is_symlink():
                    continue
                binding = build_resolved_resource_binding(reader, root=label, operation="read", path=str(source))
                if runtime_check(binding.target_path):
                    continue
                if block_reason_for_path(reader, binding.target_path, "read_bytes", binding):
                    continue
                local = pathlib.Path("inputs") / label / str(index) / relative
                clone = (not preserve_compression or not in_scope(data, binding.target_path)
                         or is_compressed(os.lstat(binding.target_path)))
                measured = copy_artifact_file(binding.target_path, root / local, mode=mode, clone=clone)  # fenced, verified
                manifest.append({"root": label, "source": str(source), "local": local.as_posix(),
                                 "source_sha256": measured["sha256"], "source_size": measured["size"],
                                 "sha256": measured["sha256"], "size": measured["size"]})
    return manifest


def _input_files(base):
    def failed(exc):
        raise exc
    for folder, dirs, files in os.walk(base, followlinks=False, onerror=failed):
        dirs[:] = sorted(name for name in dirs if name != CONTAINER
                         and not (pathlib.Path(folder) / name).is_symlink())
        for name in sorted(files):
            yield pathlib.Path(folder) / name
