"""Shared runtime and inherited workspace bindings for read operations.

Ordinary foci can address canonical runtime data independently of their execution
drive. Only children inherit ancestor workspaces, through exact parent links.
Dispatch still enforces task and Presence restrictions; writes never consult
this projection.
"""
from __future__ import annotations

import pathlib
from typing import Any
from ouroboros.store import StoreError


def _access():
    from ouroboros import tool_access
    return tool_access


def canonical_runtime_reader(ctx: Any, canonical_root: pathlib.Path, rel: str) -> str:
    """Address canonical bytes through existing READ aliases; dispatch owns admission.

    Relative runtime paths keep execution-drive bytes. This recipe predicts no
    permission, existence or freshness and reads no file to construct its address.
    """
    source = (canonical_root / rel).resolve(strict=False)
    if ctx is None:
        return f"{source} (no tool context here, so whether this actor can open it is unknown)"
    path = rel if pathlib.Path(ctx.drive_root).resolve(strict=False) == canonical_root.resolve(strict=False) else str(source)
    return (f"read_file(root='runtime_data', path={path!r})"
            " (reader address, subject to this task's tool/resource restrictions and Presence grants; see Runtime context)")


def capture_parent_workspace(ctx: Any) -> dict[str, str]:
    """Capture the parent's actual focus before choosing the child's folder."""
    from ouroboros.tools.tool_resolution import active_repo_dir_for

    access = _access()
    meta = getattr(ctx, "task_metadata", None) or {}
    if (getattr(ctx, "is_direct_chat", False) and not getattr(ctx, "workspace_root", None)
            and isinstance(meta, dict) and meta.get("_project_room_note")):
        return {"root": str(meta.get("_project_room_dir") or ""), "mode": "", "source": "project_room",
                "availability": "unavailable", "detail": str(meta["_project_room_note"])}
    if getattr(ctx, "workspace_root", None):
        source = "active_workspace"
    elif access.project_room_lens_dir(ctx) is not None:
        source = "project_room"
    elif access.folderless_scratch_dir(ctx) is not None:
        source = "task_drive"
    else:
        source = "system_repo"
    return {"root": str(active_repo_dir_for(ctx).resolve(strict=False)),
            "mode": str(getattr(ctx, "workspace_mode", "") or ""), "source": source}


def _value(record: dict, key: str):
    meta = record.get("metadata")
    return record.get(key) or (meta.get(key) if isinstance(meta, dict) else None)


def _parent_records(ctx: Any):
    """Read exact ancestry only; never enumerate tasks or a mutable project map."""
    from ouroboros.task_results import load_task_result, validate_task_id

    meta = getattr(ctx, "task_metadata", None)
    current = meta if isinstance(meta, dict) else {}
    yield current
    seen = {str(getattr(ctx, "task_id", "") or "")}
    canonical = _access().canonical_data_root(ctx)
    while parent := _value(current, "parent_task_id"):
        try:
            parent = validate_task_id(parent)
            if parent in seen:
                return
            seen.add(parent)
            current = load_task_result(canonical, parent, strict=True) or {}
        except (OSError, ValueError, TypeError, StoreError):
            return  # Missing historical provenance does not invent a different folder.
        if not current:
            return
        yield current


def inherited_read_roots(ctx: Any) -> list[tuple[str, pathlib.Path]]:
    """Physical aliases of existing logical roots, for READ operations alone."""
    access = _access()
    roots = [("runtime_data", access.canonical_data_root(ctx))]
    if access.active_tool_profile(ctx) not in {"local_readonly_subagent", "acting_subagent"}:
        from ouroboros.presence_authority import presence_ceiling_from_context

        # A Presence grant addresses its existing physical roots; ordinary
        # read parity must not expand a selected runtime_data='.' ceiling.
        return roots if presence_ceiling_from_context(ctx) is None else []
    for record in _parent_records(ctx):
        edge = _value(record, "parent_workspace")
        candidates = [("active_workspace", edge.get("root"))] if isinstance(edge, dict) else []
        # The current context already contributes its own default. Ancestor rows
        # also retain the selected folder and any nonstandard execution drive.
        candidates += [("active_workspace", _value(record, "workspace_root")),
                       ("runtime_data", _value(record, "drive_root")),
                       ("runtime_data", _value(record, "child_drive_root"))]
        for label, raw in candidates:
            if not str(raw or "").strip():
                continue
            try:
                roots.append((label, pathlib.Path(raw).expanduser().resolve(strict=False)))
            except (OSError, ValueError, RuntimeError):
                continue
    return list(dict.fromkeys(roots))


def inherited_read_base(ctx: Any, root: str, target: pathlib.Path) -> pathlib.Path | None:
    access = _access()
    bases = [base for label, base in inherited_read_roots(ctx)
             if label == root and access.path_is_relative_to(target, base)]
    return max(bases, key=lambda base: len(base.parts), default=None)


def read_allows_outside_home(ctx: Any) -> bool:
    """Inherit the parent's READ path class without changing action authority."""
    access = _access()
    if access.active_tool_profile(ctx) not in {"local_readonly_subagent", "acting_subagent"}:
        return access.is_external_workspace(ctx)
    found_edge = False
    for record in _parent_records(ctx):
        edge = _value(record, "parent_workspace")
        if isinstance(edge, dict):
            found_edge = True
            if edge.get("mode") == "external":
                return True
    # Legacy children have no captured source edge; keep their former path class.
    return not found_edge and access.is_external_workspace(ctx)


def readonly_start_folder(value: Any) -> str:
    """A read starting place is a directory, not a Git mutation boundary."""
    path = pathlib.Path(str(value or "").strip()).expanduser().resolve(strict=False)
    if not path.is_dir():
        raise ValueError(f"workspace_root is not a directory: {value}")
    return str(path)


def admit_child_start_folder(ctx: Any, value: Any, params: dict) -> str:
    """Select an existing parent READ binding; the folder grants no authority."""
    if str(params.get("write_surface") or "").strip().lower() == "genesis":
        raise ValueError("workspace_root cannot select a starting folder for genesis, which creates its own empty project")
    folder = readonly_start_folder(value)
    write_root = str(params.get("write_root") or "").strip()
    if (str(params.get("write_surface") or "").strip().lower() == "external_workspace" and write_root
            and pathlib.Path(write_root).expanduser().resolve(strict=False) != pathlib.Path(folder)):
        raise ValueError("workspace_root and write_root name different folders; select one acting workspace")
    from ouroboros.tools.tool_resolution import _root_containing_absolute_path

    root = _root_containing_absolute_path(ctx, "read_file", folder)
    # External/Cyber parents can already READ off-home addresses through
    # user_files even when no finite root contains the address. Its existing
    # resolver decides that reach, including refusals for ordinary parents.
    binding = _access().build_resolved_resource_binding(
        ctx, root=root or "user_files", operation="read", path=folder)
    return str(binding.target_path)


def scheduled_origin(ctx: Any) -> str:
    """The task a scheduled follow-up was registered from, for READS only (#1488, owner 2026-10-13 A).

    Related and independent schedules alike: the origin's materials, not its money,
    deadline, controls, candidate or Git paths. Named only by the HOST-written
    occurrence receipt (``schedule_admission``) of this actor's root, never by task
    metadata an ordinary admission may carry, so its retry and its children read
    that root's origin and nothing further back. Absent or malformed grants nothing.
    An unreadable receipt grants nothing NOW and is read again next time: only what a
    completed read showed is remembered for this actor, never a transient fault.
    Disclosed cost: a folderless readonly helper copies every permitted lineage file
    (``delegate_readonly_inputs``), so each such start copies the origin's whole
    ``task_drive``/``artifact_store``, and one unreadable origin file refuses it.
    """
    meta = getattr(ctx, "task_metadata", None)
    root_id = str((meta if isinstance(meta, dict) else {}).get("root_task_id") or getattr(ctx, "task_id", "") or "")
    cached = getattr(ctx, "_scheduled_origin", None)  # (root it was read for, origin)
    if isinstance(cached, tuple) and cached[0] == root_id:
        return cached[1]
    try:
        from ouroboros.task_results import load_task_result, validate_task_id

        row = load_task_result(_access().canonical_data_root(ctx), validate_task_id(root_id), strict=True) or {}
    except Exception:
        return ""  # unknown is not absence: deny this read, ask the receipt again on the next one
    origin = ""
    try:
        receipt = row.get("schedule_admission") if isinstance(row.get("schedule_admission"), dict) else {}
        frozen = receipt.get("task") if isinstance(receipt.get("task"), dict) else {}
        if receipt.get("status") == "accepted" and receipt.get("token") and str(frozen.get("id") or "") == root_id:
            origin = str(((frozen.get("metadata") or {}).get("followup_origin") or {}).get("task_id") or "")
    except Exception:
        origin = ""  # a malformed receipt proves no origin
    try:
        setattr(ctx, "_scheduled_origin", (root_id, origin))
    except Exception:
        pass
    return origin
