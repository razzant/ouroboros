"""Move one work's room now, and its execution resources at readmission.

The immutable binding owns placement. Queue rows own writer leases; changing a
running row cannot acquire a lease and must never change its folder underneath it.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def placement_basis(drive_root: Any, task_id: str, project_id: str) -> dict:
    """Capture the destination before transport or asynchronous project naming."""
    from ouroboros.projects_registry import project_id_for_task, project_admission_view

    bound = project_id_for_task(drive_root, task_id, strict=True)
    return project_admission_view(drive_root, bound or project_id, allow_unregistered=not bound)


def move_task_to_project(drive_root: Any, task_id: str, project_id: str, *, name: str = "",
                         origin: dict, basis: dict, running: Any, pending: Any,
                         initiated_by: str = "mind") -> dict:
    """The mind and owner button share this binding-first placement transaction."""
    from ouroboros.projects_registry import (
        bind_task_to_project, create_project, get_project, origin_claim_lock,
        project_admission_basis, project_admission_guard, project_id_for_task, update_project,
    )
    from ouroboros.project_lease import mark_task_project
    from supervisor.queue import _queue_lock, persist_queue_snapshot

    with origin_claim_lock(), _queue_lock:
        bound = project_id_for_task(drive_root, task_id, strict=True)
        pid = bound or project_id
        if basis.get("project_id") != pid:
            basis = placement_basis(drive_root, task_id, pid)
        if bound:
            with project_admission_guard(drive_root, basis):
                project = get_project(drive_root, bound, strict=True)
                renamed = bool(name and name != project.get("name"))
                if renamed:
                    project = update_project(drive_root, bound, name=name)
        else:
            project = create_project(drive_root, pid, name=name, origin=initiated_by, admission_basis=basis)
            renamed = False
            if basis["project"] is None:
                basis = project_admission_basis(pid, project)
        binding = bind_task_to_project(
            drive_root, task_id, pid, project.get("chat_id"), origin=origin,
            admission_basis=basis, execution_readmission=True,
        )
        # Only PENDING rows can acquire a new lease on their next assignment.
        marked = mark_task_project(running, pending, task_id, pid, authority="binding", project=project)
        if marked:
            try:
                persist_queue_snapshot(reason="project_placement")
            except Exception:
                log.warning("Placement committed; queue snapshot write failed", exc_info=True)
    from ouroboros.project_handoff import enqueue_project_handoff, handoff_identity

    receipt = enqueue_project_handoff(drive_root, task_id, source_ref=origin.get("ref"))
    if project.get("created") and initiated_by == "mind":
        from ouroboros.project_dialogue import announce_project_started

        announce_project_started(drive_root, project, task_id)
    try:
        from supervisor.message_bus import get_bridge

        get_bridge().broadcast({"type": "projects_changed", "project_id": pid, "chat_id": project.get("chat_id")})
    except Exception:
        log.debug("Placement committed; live project notification unavailable", exc_info=True)
    return {"status": "delivered", "project_id": pid, "chat_id": project.get("chat_id"),
            "reason": "renamed" if renamed else ("created" if project.get("created") else "attached"),
            "project": project, "binding": binding, "adopted": bool(bound),
            "handoff_receipt": receipt, "handoff_id": handoff_identity(pid, task_id, origin.get("ref"))}


def pending_placement(drive_root: Any, task: dict, *, bindings_snapshot: dict | None = None) -> dict:
    """Read the move this pending attempt has not consumed; never consult siblings."""
    from ouroboros.projects_registry import project_binding_for_task

    if task.get("delegation_role") == "subagent":
        return {}
    binding = project_binding_for_task(drive_root, str(task.get("id") or ""), strict=True,
                                       bindings_snapshot=bindings_snapshot) or {}
    return binding if (binding.get("placement_admission")
                       and (task.get("placement_applied_at") or (task.get("metadata") or {}).get("placement_applied_at"))
                       != binding.get("bound_at")) else {}


def refresh_pending_placement(drive_root: Any, task: dict, *, binding: dict | None = None) -> bool:
    """Refresh pending lease identity before selection, keeping folder access unchanged."""
    from ouroboros.sleep_custody import carrier_receipt
    from ouroboros.project_admission import project_admission_basis, project_admission_guard

    try:
        binding = pending_placement(drive_root, task) if binding is None else binding
        if not binding:
            return True
        if carrier_receipt(task).get("retains_project_lane"):
            return False
        with project_admission_guard(drive_root, {**binding["placement_admission"], "frozen": True}) as project:
            task["project_id"] = binding["project_id"]
            task["_project_admission"] = project_admission_basis(binding["project_id"], project, frozen=True)
            task.pop("_project_scope_none", None)
        return True
    except Exception:
        log.warning("Pending placement cannot be readmitted for %s", task.get("id"), exc_info=True)
        return False


def readmit_placement(drive_root: Any, task: dict, *, repo_dir: Any, running: Any, pending: Any) -> bool:
    """Under Q, acquire the selected room's lease before changing execution resources."""
    from ouroboros.project_admission import project_admission_basis, project_admission_guard
    from ouroboros.project_lease import candidate_is_leasable, cold_project_holders, running_project_ids
    from ouroboros.workspace_admission import validate_workspace_root

    binding = pending_placement(drive_root, task)
    if not binding:
        return True
    if not refresh_pending_placement(drive_root, task, binding=binding):
        return False
    if not candidate_is_leasable(task, running_project_ids(running.values()),
                                cold_holders=cold_project_holders(pending)):
        return False
    # Placement changes the room's default, never a separately selected resource: an explicit
    # folder or opt-out, or the candidate of body work. The registry holds that candidate when a
    # restart dropped a direct actor's binding, or before a Continue inherits it (prepare).
    from ouroboros.body_candidate import continuation_predecessor, find

    metadata = task.setdefault("metadata", {})
    kind = (metadata.get("resource_intent") or {}).get("kind")
    owner = str(metadata.get("root_task_id") or task.get("root_task_id") or task.get("id") or "")
    body = kind in (None, "", "system_repo") and any(
        find(lineage, drive_root, strict=True) for lineage in (owner, continuation_predecessor(task)) if lineage)
    pinned = body or kind in {"explicit_resource", "explicit_none"}
    # Placement pins room identity; readmission consumes its current generation.
    with project_admission_guard(drive_root, {**binding["placement_admission"], "frozen": True}) as project:
        workspace = {"root": str(task.get("workspace_root") or ""), "mode": str(task.get("workspace_mode") or "")}
        if not pinned:
            folder = validate_workspace_root(project.get("working_dir"), system_repo_dir=repo_dir, drive_root=drive_root)
            workspace = {"root": str(folder or ""), "mode": "external" if folder else ""}
        task.update(project_id=binding["project_id"], chat_id=project.get("chat_id"),
                    workspace_root=workspace["root"], workspace_mode=workspace["mode"],
                    _project_admission=project_admission_basis(binding["project_id"], project, frozen=True),
                    placement_applied_at=binding["bound_at"])
        task.pop("_project_scope_none", None)
        contract = task.get("task_contract") or metadata.get("task_contract")
        if isinstance(contract, dict):
            task["task_contract"] = {**contract, "workspace": workspace}
        metadata["placement_applied_at"] = binding["bound_at"]
        if isinstance(metadata.get("task_contract"), dict):
            metadata["task_contract"] = task["task_contract"]
        if body and not kind and not workspace["root"]:
            # Recorded where every default resolver reads it: no room lens or scratch replaces the body.
            # A kept folder is never relabelled system_repo; children and follow-ups inherit the label.
            metadata["resource_intent"] = {"kind": "system_repo"}
        metadata.pop("_project_room_dir", None)
        metadata.pop("_project_room_note", None)
    return True


def append_placement_notice(ctx: Any, messages: list) -> bool:
    """One host fact per durable move, including owner-button moves of a live actor."""
    from ouroboros.tool_access import canonical_data_root
    from ouroboros.projects_registry import project_binding_for_task, get_project

    try:
        binding = project_binding_for_task(canonical_data_root(ctx), str(ctx.task_id or ""), strict=True) or {}
        if not binding.get("placement_admission"):
            return False
        applied = (getattr(ctx, "task_metadata", None) or {}).get("placement_applied_at") == binding["bound_at"]
        marker = f"[PROJECT_{'EXECUTION' if applied else 'PLACEMENT'} {binding['bound_at']}]"
        if getattr(ctx, "_seen_project_placement", "") == marker:
            return False
        if not any(marker in str(message.get("content") or "") for message in messages):
            project = get_project(canonical_data_root(ctx), binding["project_id"], strict=True) or {}
            if applied:
                active = getattr(ctx, "active_repo_dir", None)
                folder = active() if callable(active) else getattr(ctx, "workspace_root", None) or getattr(ctx, "repo_dir", None)
                resources = f"Readmission has completed. This execution's working folder is {folder}. "
            else:
                resources = ("The current execution keeps its folder and writer lease; readmission at the next "
                             "execution boundary selects the room's default unless an explicit resource or code candidate is retained. ")
            messages.append({"role": "user", "content": (
                f"{marker} This work ({ctx.task_id}) now lives in {project.get('name') or binding['project_id']} "
                f"(project_id={binding['project_id']}). Routing, journal and knowledge use this room now. "
                f"{resources}Knowledge is not automatically reloaded.")})
        ctx._seen_project_placement = marker
        return True
    except Exception:
        log.debug("Placement notice unavailable at this boundary", exc_info=True)
        return False
