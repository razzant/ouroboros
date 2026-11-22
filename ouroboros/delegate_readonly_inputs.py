"""Materialize selected or automatic native inputs for a folderless session request.

The engine accepts one readonly project scope, not additional read roots. Give
each invocation its own scratch with verified input copies (``_lineage_copies``:
bytes only, cloned where the volume can, streamed where compression may later
want the source); never pass parent scratch or external runtime data as the
engine's project. A plan-thread turn stages nothing: its thread keeps its first
folder. These copies confer no workspace/write authority. Native and session
inputs use one access owner.

``input_files`` selects ordinary files or local Git commit ranges. Files pass
read_file admission; delegate_git_range checks and renders the comparison. Before send or queue, each file's
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
import tempfile
from contextlib import ExitStack
from dataclasses import replace
from typing import Any, Dict, List, NamedTuple, Optional

CONTAINER = "delegated_readonly_inputs"
SELECTED_DIR = "inputs/selected"
ADDRESS_HEX = 16  # the content folder is the sha256's first 16 hex (receipts keep all 64): ~48 fewer path characters
# ``read_file``'s ordinary-file roots: ``skill_payload`` needs a bucket and skill name this selection has no field for.
INPUT_FILE_ROOTS = ("active_workspace", "system_repo", "runtime_data", "task_drive", "artifact_store",
                    "user_files", "subagent_projects", "deliverables")
INPUT_FILES_CONTRACT = (
    "Read-only inputs, every entry mandatory. Select a file {path, root?} from "
    + ", ".join(INPUT_FILE_ROOTS) + " (skill_payload and others refuse); omitted root is inferred from an absolute "
    "path, else active_workspace. Or select {git_range:{base,head,root?,repository?,path?}}: two local commits "
    "in your repository; empty repository/path use the defaults. Complete patch and manifest, never a preview. "
    "Nonempty input_files replaces folderless lineage; omit, null or [] keeps it. A bound folder or plan thread "
    "keeps its scope and receives absolute paths subject to native read reach. Preparation is not a read. "
    "Engine documents use attachment_paths.")


class InputSelectionError(ValueError):
    """A typed refusal before send, with selection or file-copy results."""

    def __init__(self, code: str, detail: str, results: Optional[List[Dict[str, Any]]] = None) -> None:
        super().__init__(detail)
        self.code, self.detail, self.results = code, detail, list(results or [])


class ReadonlyInputs(NamedTuple):
    scope_root: str       # the folderless scope ("" when the run keeps its own root)
    instruction: str      # appended to the run's instructions
    selected: Optional[List[Dict[str, Any]]]  # the receipt: None when nothing was asked, else one row per retained file
    selection_note: str   # the selection's own text, restated in a continuation's prompt
    lineage: Optional[tuple] = None  # (count, bytes) of the automatic lineage copies this start made, else None

    @property
    def custody(self) -> Dict[str, Any]:
        """The durable source map on the start's START_REQUESTED row."""
        return {"input_files": self.selected} if self.selected is not None else {}  # an explicit [] is a fact too


NO_INPUTS = ReadonlyInputs("", "", None, "")


def input_files_parameter(scope: str) -> Dict[str, Any]:
    """The ``input_files`` schema delegate_start and schedule_subagent share. The item schema is typed
    for both: a compact custom-tool projection drops the description but keeps this structure, which
    ``selection_entries`` and ``resolve_selection`` enforce again before anything happens."""
    from ouroboros.delegate_git_range import FIELDS, ROOTS

    ordinary = {"type": "object", "required": ["path"], "additionalProperties": False, "properties": {
        "path": {"type": "string"}, "root": {"type": "string", "enum": list(INPUT_FILE_ROOTS)}}}
    git_range = {"type": "object", "required": ["base", "head"], "additionalProperties": False,
                 "properties": {**{key: {"type": "string", **({"minLength": 1} if key in {"base", "head"} else {})}
                                    for key in FIELDS},
                                "root": {"type": "string", "enum": list(ROOTS)}}}
    # The two branches are disjoint.
    items = {"anyOf": [ordinary, {"type": "object", "required": ["git_range"], "additionalProperties": False,
                                  "properties": {"git_range": git_range}}]}
    return {"input_files": {"type": "array", "items": items, "description": f"{INPUT_FILES_CONTRACT} {scope}"}}


def selection_entries(value: Any) -> Optional[List[Dict[str, Any]]]:
    """Validate the two exclusive selection shapes before either consumer proceeds."""
    from ouroboros.delegate_git_range import validated_range

    if value is None:
        return None
    entries = []
    if not isinstance(value, list):
        raise InputSelectionError("input_files_invalid", "input_files must be an array")
    for item in value:
        if isinstance(item, dict) and set(item) == {"git_range"}:
            try:
                entries.append({"git_range": validated_range(item["git_range"])})
            except ValueError as exc:
                raise InputSelectionError("input_files_invalid", str(exc)) from exc
        elif isinstance(item, dict) and not set(item) - {"root", "path"}:
            path = item.get("path")
            if (not isinstance(path, str) or not path.strip() or "\x00" in path
                    or not isinstance(item.get("root", ""), str)):
                raise InputSelectionError("input_files_invalid",
                                          "input_files is a list of {root, path} objects with a nonempty path.")
            entries.append({"root": item.get("root", "").strip(), "path": path})
        else:
            raise InputSelectionError("input_files_invalid", "Select either {path, root?} or {git_range: {...}}.")
    return entries


_INERT_EMPTY = ("bucket", "skill_name")
_CHILD_METADATA_KEYS = ("resource_intent", "root_task_id", "session_id", "project_id", "allowed_resources",
                        "required_capabilities", "budget_drive_root", "child_drive_root", "workspace_root",
                        "workspace_mode", "depth_provenance")


def receiving_reader(ctx: Any, resource_policy: Any = None, *, workspace_root: str = "", child_task_id: str = "",
                     child_drive: Optional[pathlib.Path] = None, resource_intent: Optional[dict] = None) -> Any:
    """The read-only recipient's limits ON TOP of the caller's contract (never instead of its constraint:
    every entry must pass both). A direct start's run reads in the caller's own roots.

    A scheduled child is the reader its worker will build (``agent.py``): the serving body as its system
    root, since ``body_candidate.restore`` binds no subagent while the parent's ``bind`` moved its own;
    the scheduled workspace and resource intent; and the parent's contract under the child's ``resource_policy``.
    Its project folder may differ from either body root: a source passes only where both readers name it."""
    reader = copy.copy(ctx)
    reader.task_constraint = {"mode": "local_readonly_subagent"}
    if child_task_id:
        from ouroboros.artifacts import task_id_for_artifacts
        from ouroboros.body_candidate import serving_repo_dir_for
        from ouroboros.consciousness_authority import consciousness_origin_metadata
        from ouroboros.dialogue_provenance import presence_binding_authority_metadata
        from ouroboros.tool_access import canonical_data_root
        from ouroboros.tool_access_reads import capture_parent_workspace

        reader.repo_dir = reader.system_repo_dir = serving_repo_dir_for(ctx)
        reader.serving_repo_dir = None
        parent = getattr(ctx, "task_metadata", None) or {}
        # Each check keeps the carrier it reads on the parent (Presence the attribute, protected artifacts a
        # nonempty metadata contract); an empty carrier takes the other's, and both take the child's policy.
        own, held = (value if isinstance(value, dict) else {}
                     for value in (getattr(ctx, "task_contract", None), parent.get("task_contract")))
        own, held = ({**contract, **({"resource_policy": resource_policy} if resource_policy else {})}
                     for contract in (own or held, held or own))
        reader.task_contract = own
        reader.task_id, reader.is_direct_chat = child_task_id, False
        # The child's metadata is what dispatch gives it (supervisor/task_dispatch.py): the lineage and
        # structural keys it names, the tree origin and only the Presence binding, never the speaker's
        # event or a parent's handoff.
        reader.task_metadata = {
            **{key: copy.deepcopy(parent[key]) for key in _CHILD_METADATA_KEYS if key in parent},
            **({"task_contract": held} if held else {}),
            **({"resource_intent": copy.deepcopy(resource_intent)} if resource_intent is not None else {}),
            **consciousness_origin_metadata({**parent, "task_id": task_id_for_artifacts(ctx)}),
            **presence_binding_authority_metadata(parent, task_contract=getattr(ctx, "task_contract", None)),
            "parent_task_id": task_id_for_artifacts(ctx), "parent_workspace": capture_parent_workspace(ctx),
        }
        if child_drive is not None:
            reader.drive_root = child_drive
            reader.task_metadata.update(drive_root=str(child_drive), child_drive_root=str(child_drive),
                                        budget_drive_root=str(canonical_data_root(ctx)))
    if workspace_root:
        from ouroboros.work_domain import settle

        reader.workspace_root, reader.workspace_mode = settle(workspace_root), "read_only"
    return reader


def require_selected_tool(ctx: Any, tool: str, args: dict) -> dict:
    """Reuse the dispatch capability checks without executing a read."""
    from ouroboros.tools.registry_core import _presence_bound_args
    from ouroboros.tools.registry_guards import _capability_resource_guard_result

    bound, refusal = _presence_bound_args(ctx, tool, args)
    if refusal:
        raise PermissionError(refusal)
    if result := _capability_resource_guard_result(ctx, tool, bound):
        raise PermissionError(result.text)
    return bound


def _dispatch_read_binding(reader: Any, args: dict) -> Any:
    from ouroboros.tools.tool_resolution import _build_builtin_target_binding, _normalize_dispatch_path_args_result

    args = dict(args)
    _normalize_dispatch_path_args_result(reader, "read_file", args)
    if (args.get("root") or "active_workspace") not in INPUT_FILE_ROOTS:
        raise ValueError("input_files needs an ordinary read_file root")
    return _build_builtin_target_binding(reader, "read_file", args)


def selected_read_binding(reader: Any, args: dict, *, exact: Optional[pathlib.Path] = None,
                          historic: bool = False) -> Any:
    """Resolve the reader's own dispatch spelling. Fixed args may address, never replace, the source."""
    from ouroboros.tools.registry_core import PRESENCE_RESOURCE_BLOCKED

    def operands(bound: dict) -> dict:
        # An empty skill selector is the ordinary roots' default, so a static binding of it adds nothing; every other
        # operand (a content bound, a non-empty selector, an unknown name) would change what is read and refuses.
        return {key: value for key, value in bound.items() if key not in _INERT_EMPTY or value != ""}

    if historic:
        from ouroboros.delegate_git_range import ROOTS
        from ouroboros.tool_access import build_resolved_resource_binding

        # Historical names stay literal on Git roots unless the grant fixes a path.
        # Other roots use normal file dispatch; every grant must still reach the same target.
        fixed = operands(require_selected_tool(reader, "read_file", {}))
        bound = {**args, **fixed}
    else:
        bound = operands(require_selected_tool(reader, "read_file", args))
    if bound.keys() - {"root", "path"} or not all(isinstance(value, str) for value in bound.values()):
        raise PermissionError(PRESENCE_RESOURCE_BLOCKED)
    if historic and "path" not in fixed and bound.get("root") in ROOTS:
        binding = build_resolved_resource_binding(reader, root=bound["root"], operation="read", path=bound["path"],
                                                  literal=True)
    else:
        binding = _dispatch_read_binding(reader, bound)
    if exact is None and bound != args:
        exact = _dispatch_read_binding(reader, args).target_path
    if exact is not None and binding.target_path != exact:
        raise PermissionError(PRESENCE_RESOURCE_BLOCKED)
    return binding


def _admit_read(reader: Any, binding: Any, *, prospective_copy: Optional[pathlib.Path] = None) -> None:
    from ouroboros.tools.core_file_tools import read_admission

    _, _, refusal = read_admission(reader, binding.root, str(binding.target_path), resolved_binding=binding,
                                  prospective_copy=prospective_copy)
    if refusal is not None:
        raise PermissionError(getattr(refusal, "text", refusal))


def admit_selected_source(reader: Any, binding: Any, *, entry: Optional[pathlib.Path] = None) -> None:
    """Read and derive only from the proved authority coordinates; historic entries keep their lexical identity.
    Each reader answers for its own working folder beside the address it asked: a relative protected path
    anchors at both, for the caller as for the receiver, so packaging a selection into another reader's
    hands never widens the grounds a participant's own contract already set."""
    from ouroboros.protected_artifacts import block_reason_for_path

    if entry is not None:
        binding = replace(binding, requested_path=entry)  # deleted directories still have historical descendants
    _admit_read(reader, binding)
    operations = ("copy", "hash") if entry is None else ("copy", "hash", "static_introspection")
    targets = ((binding.target_path, False),) if entry is None else ((binding.target_path, False), (entry, True))
    policies = [binding]
    # Rebase only relative policy paths, never read grants or the historical entry: each reader's own
    # effective working folder (``active_repo_dir``'s rule, so a folderless task answers for its task
    # drive) anchors its contract's relative paths exactly as the asked address does.
    from ouroboros.tools.tool_resolution import active_repo_dir_for
    from ouroboros.work_domain import settle

    if (workspace := settle(active_repo_dir_for(reader))) != settle(binding.base_path):
        policies.append(replace(binding, base_path=workspace))
    for policy in policies:
        for target, is_entry in targets:
            for operation in ("read_bytes", *operations) if is_entry or policy is not binding else operations:
                if reason := block_reason_for_path(reader, target, operation, policy, entry=is_entry):
                    raise PermissionError(reason)


def _admit_copy_address(reader: Any, path: pathlib.Path) -> None:
    """Resolve the exact content address as the receiving child will read it, before publishing it."""
    try:
        binding = selected_read_binding(reader, {"path": str(path)}, exact=path)
    except (ValueError, RuntimeError) as exc:
        raise PermissionError(str(exc)) from exc  # the staging owner discards its unpublished temporary
    _admit_read(reader, binding, prospective_copy=path)


def _admit_file_selection(readers: tuple, row: dict, *, scheduled: bool) -> Any:
    caller, reader = readers
    args = {key: value for key, value in row.items() if value}
    source = selected_read_binding(caller, args)
    admit_selected_source(caller, source)
    if scheduled:
        # A relative selector belongs to the caller. Pin its (possibly
        # re-anchored) original address before the recipient resolves its read.
        args = {**({"root": source.root} if "root" in args else {}),
                "path": str(source.requested_path or source.target_path)}
    received = selected_read_binding(reader, args, exact=source.target_path)
    admit_selected_source(reader, received)
    return source


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


def resolve_selection(ctx: Any, entries: List[Dict[str, Any]], receiving: Any = None, *,
                      range_dir: Optional[pathlib.Path] = None) -> List[Dict[str, Any]]:
    """Admit every entry by ``read_file``'s own ``read_admission`` (Presence ceiling included) for the
    caller AND the receiving reader, before any copy."""
    readers = (ctx, receiving if receiving is not None else receiving_reader(ctx))
    resolved, results = [], []
    for index, entry in enumerate(entries):
        if "git_range" in entry:
            if range_dir is None:  # direct-start preflight validates only the selector shape
                results.append({**entry, "status": "deferred", "detail": "Git comparison happens during retention."})
                continue
            from ouroboros.delegate_git_range import RangeAdmission, materialize_range, range_binding

            admission = RangeAdmission(require_selected_tool, selected_read_binding, admit_selected_source)
            try:
                binding = range_binding(readers, entry["git_range"], admission)
                resolved.extend(materialize_range(readers, binding, entry["git_range"], range_dir / str(index),
                                                  admission))
                results.append({**entry, "status": "checked"})
            except Exception as exc:
                results.append({**entry, "status": "denied", "detail": f"{type(exc).__name__}: {exc}"})
            continue
        row = {"root": entry["root"], "path": entry["path"]}
        try:
            if row["root"] and row["root"] not in INPUT_FILE_ROOTS:
                results.append({**row, "status": "unsupported_root",
                                "detail": f"input_files reads ordinary files from {', '.join(INPUT_FILE_ROOTS)}"})
                continue
            binding = _admit_file_selection(readers, row, scheduled=receiving is not None)
            row["root"] = binding.root
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
    if any(row["status"] not in {"checked", "deferred"} for row in results):
        raise InputSelectionError("input_files_unavailable", (
            "Some selected input_files cannot be prepared, so nothing was started. Fix or drop failed entries "
            "and start again. A deferred range has not been compared."), results)
    return resolved


def copy_selection(rows: List[Dict[str, Any]], root: Optional[pathlib.Path], *, receiving: Any = None) -> List[Dict[str, Any]]:
    """Copy each checked (or already retained) row to ``root/inputs/selected/<sha256[:16]>/<name>``.

    A retained row is verified against its recorded identity; ``root=None`` verifies it in
    place (no second copy). An occupied address keeps its bytes only when they are these exact
    bytes; anything else refuses, never overwritten. A scheduled receiver must also admit the actual
    address before it is published. Any failure refuses the whole selection."""
    from ouroboros.artifact_storage import temporary_sibling
    from ouroboros.artifacts import ArtifactIdentityError, copy_artifact_file, stream_artifact_file

    def identity(row: Dict[str, Any]) -> Dict[str, Any]:
        """What a row selected: a range product's immutable range included, whether copied or not."""
        return {key: row[key] for key in ("root", "path", "source", "git_range") if key in row}

    receipt: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        expected = {"sha256": row["sha256"], "size": row["size"]} if row.get("local") else None
        base = identity(row)
        name = row["path"] if "git_range" in row else pathlib.PurePath(row["source"]).name
        try:
            if root is None:
                measured, local = stream_artifact_file(row["local"], expected=expected), pathlib.Path(row["local"])
            else:
                staging = temporary_sibling(root / SELECTED_DIR / name)
                origin = row.get("local") or row.get("_origin") or row["source"]
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
                if receiving is not None:
                    try:
                        _admit_copy_address(receiving, local)
                    except PermissionError:
                        staging.unlink(missing_ok=True)
                        raise
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
            status = ("changed" if isinstance(exc, ArtifactIdentityError) else
                      "denied" if isinstance(exc, PermissionError) else "copy_failed")
            raise InputSelectionError("input_files_unavailable", (
                "A selected input could not be prepared unchanged, so nothing was started; earlier rows were copied "
                "but are not delivered."), [*receipt, {**base, "status": status, "detail": str(exc)}, *(
                    {**identity(later), "status": "not_attempted"} for later in rows[index + 1:])]) from exc
        receipt.append({**base, "status": "prepared", "local": str(local),
                        "retained": str(row.get("retained") or local), **measured})
    return receipt


def schedule_selected_inputs(ctx: Any, entries: List[Dict[str, Any]], child_task_id: str,
                             resource_policy: Any = None, *, workspace_root: str = "",
                             child_drive: Optional[pathlib.Path] = None,
                             resource_intent: Optional[dict] = None) -> List[Dict[str, Any]]:
    """Check and retain a scheduled child's selection before it is queued.

    The caller's read authority AND the child's read-only limits (with its own resource
    policy) must both admit each file. The bytes stay in the scheduling task's artifact store,
    which its direct child reads as lineage. Later refusals still retain these bytes with the task."""
    folder = f"scheduled-{child_task_id}"
    reader = receiving_reader(ctx, resource_policy, workspace_root=workspace_root, child_task_id=child_task_id,
                              child_drive=child_drive, resource_intent=resource_intent)
    return _retain_selection(ctx, entries, folder, reader)


def _retain_selection(ctx, entries, folder, receiving=None):
    with ExitStack() as stack:
        staging = (pathlib.Path(stack.enter_context(tempfile.TemporaryDirectory()))
                   if any("git_range" in entry for entry in entries) else None)
        rows = resolve_selection(ctx, entries, receiving, range_dir=staging)
        root = _retained_root(ctx, folder)
        root.mkdir(parents=True, exist_ok=False)
        receipt = copy_selection(rows, root, receiving=receiving)
        _write_manifest(root, [], receipt)
        return receipt


def start_inputs(ctx: Any, authority: Any, invocation_id: str, input_files: Any, prepared: Any,
                 thread_folder: Optional[str] = None, *, configured_order: bool = False):
    """``(inputs, refusal)`` for one new start: the explicit selection, else the default lineage staging.

    Only a read-only run (readonly access or an evidence-only investigation) stages inputs; a
    nonempty selection for a writable run refuses typed, while omission and [] never gate it.
    ``thread_folder``: a plan-thread turn's, the folder its engine thread keeps (``prepare_readonly_inputs``).
    Every refusal follows engine discovery and precedes the request row."""
    from ouroboros.delegate_evidence import record_start_blocked
    from ouroboros.delegate_shared import _fail

    def refuse(code: str, detail: str, **facts):
        record_start_blocked(ctx, str(getattr(ctx, "task_id", "") or ""), code)
        return NO_INPUTS, _fail("delegate_start", code, detail, definitely_unrun=True,
                                engine_discovery_completed=True, **facts)

    try:
        entries = selection_entries(input_files)
        asked = entries is not None or bool(prepared)
        if not (authority.access == "readonly" or authority.investigation):
            if entries or (entries is None and prepared):
                return refuse("input_files_readonly_only", "input_files prepares inputs for a read-only run; "
                              "this start can write. Omit input_files, or start it read-only.")
            return NO_INPUTS._replace(selected=[] if asked else None), None
        inputs = prepare_readonly_inputs(ctx, invocation_id, entries, prepared, thread_folder=thread_folder)
        bootstrap = getattr(ctx, "_configured_actor_bootstrap", None) or {}
        if configured_order and entries is not None and bootstrap.get("prepared_input_files"):
            note = ("\nThis start's explicit input_files replaces the original schedule-time selection. "
                    + ("Use the selected copies above." if entries else "The effective explicit selection is empty ([])."))
            inputs = inputs._replace(instruction=inputs.instruction + note, selection_note=inputs.selection_note + note)
    except InputSelectionError as exc:
        return refuse(exc.code, exc.detail, input_files=exc.results)
    except (OSError, ValueError) as exc:
        return refuse("readonly_inputs_unavailable", str(exc))
    return inputs._replace(selected=inputs.selected if asked else None), None


def replayed_lineage(request: Any, invocation_id: str, investigation: bool, selected: Any) -> Optional[str]:
    """What a replayed start may say of automatic lineage copies it holds no count of, from what THAT start's own
    request recorded and nothing else (no folder is read or made before the engine's lookup): ``NO_RECORD`` only
    where it was prepared for them (a read-only or investigation start without a nonempty selection whose recorded
    ``scope.root`` is this invocation's own local preparation address, ``<task drive>/CONTAINER/<invocation_id>``),
    else None: a managed copy's address is kept by its scratch owner, so the record alone does not decide it."""
    from ouroboros.helper_receipt import NO_RECORD

    body = request if isinstance(request, dict) else {}
    root = pathlib.PurePath(str((body.get("scope") if isinstance(body.get("scope"), dict) else {}).get("root") or ""))
    prepared = bool(invocation_id) and root.name == invocation_id and root.parent.name == CONTAINER
    possible = (body.get("access") == "readonly" or investigation) and not selected and prepared
    return NO_RECORD if possible else None


def prepare_readonly_inputs(ctx: Any, invocation_id: str, entries: Optional[List[Dict[str, Any]]] = None,
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

    if entries is not None:
        prepared = None  # an explicit [] replaces the schedule-time selection too
    scratch = folderless_scratch_dir(ctx)
    if scratch is None and not (entries or prepared):
        return ReadonlyInputs("", "", [], "")
    if not invocation_id or pathlib.Path(invocation_id).name != invocation_id:
        raise ValueError("invalid readonly invocation identity")
    container = None if scratch is None else _owned_container(scratch)
    if container is not None and thread_folder == "":
        raise ValueError("The engine's record of this plan thread names no folder (repoRoot), so this read-only "
                         "turn's scope is unknown; nothing was retained, staged, registered or sent. Start a "
                         "fresh plan, or continue once the engine names the thread's folder.")
    rows = _retain_selection(ctx, entries, invocation_id) if entries else list(prepared or [])
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
    ) + note, receipt, note, (len(lineage), sum(int(row["size"]) for row in lineage)) if lineage else None)


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
            selected.append(_manifest_row(row, rel))
        members.append(("inputs.json", json.dumps(selected, ensure_ascii=False, indent=2).encode("utf-8"), 0o644))
    work_domain.import_tree(target, members)
    return target


def _manifest_row(row, local):
    return {**{key: row[key] for key in ("root", "source", "git_range", "sha256", "size") if key in row},
            "local": local, "selected_path": row["path"], "source_sha256": row["sha256"], "source_size": row["size"]}


def _write_manifest(root: pathlib.Path, lineage: List[Dict[str, Any]], receipt: List[Dict[str, Any]]) -> None:
    from ouroboros.utils import write_bytes_atomic

    selected = [_manifest_row(row, pathlib.Path(row["local"]).relative_to(root).as_posix()) for row in receipt]
    write_bytes_atomic(root / "inputs.json", json.dumps(lineage + selected, ensure_ascii=False, indent=2).encode("utf-8"))


def selection_label(row):
    if spec := row.get("git_range"):
        return (f"{row['root']}:{row['source']} {spec['base']}..{spec['head']} "
                f"{json.dumps(spec['path'])} -> {row['path']}")
    return f"{row['root']}:{row['path']}"


def _selection_rows(rows: List[Dict[str, Any]], scope: Optional[pathlib.Path] = None) -> str:
    def inside(row: Dict[str, Any]) -> str:
        return f"{pathlib.Path(row['local']).relative_to(scope).as_posix()} in this folder, host path " if scope else ""

    rendered = "\n".join(f"- {selection_label(row)} -> {inside(row)}{row['local']} "
                         f"(sha256 {row['sha256']}, {row['size']} bytes)" for row in rows)
    if any(row.get("git_range") for row in rows):
        rendered += ("\nFor each range, range.json gives the original range.patch hash and names the "
                     "lossless text companion and its inverse, if needed. For UTF-8-only readers use the companion when listed. "
                     "With read_file, follow the returned start_char if cut within a window, or advance start_line past a complete page.")
    return rendered


def _selection_note(manifest: Any, receipt: List[Dict[str, Any]], *, inside_scope: bool,
                    scope: Optional[pathlib.Path] = None) -> str:
    where = ("inside this run's folder" if inside_scope else
             "outside this run's folder, which this selection does not change; the host prepared them but cannot "
             "confirm this harness may read outside its folder; report an unreadable copy rather than guess its contents")
    return (f"SELECTED INPUT FILES: this start uses the following copies; the host retained their exact bytes before this "
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
    keeps its source shared while it lives (before macOS 14, for good); an already compressed record keeps its
    compression in the clone."""
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
