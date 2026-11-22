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
        _validated_origin, bind_task_to_project, create_project, get_project, origin_claim_lock, project_chat_id,
        project_admission_basis, project_admission_guard, project_id_for_task, update_project,
    )
    from ouroboros.project_lease import mark_task_project
    from supervisor.queue import _queue_lock, persist_queue_snapshot

    with origin_claim_lock(), _queue_lock:
        bound = project_id_for_task(drive_root, task_id, strict=True)
        pid = bound or project_id
        if basis.get("project_id") != pid:
            basis = placement_basis(drive_root, task_id, pid)
        _validated_origin(origin, (basis.get("project") or {}).get("chat_id", project_chat_id(pid)))
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


def placement_folder(drive_root: Any, task: dict, project: dict | None, *, own: bool | None = None) -> str:
    """The folder a moved work's readmission selects, as its lane reads it until then (``project_lease.lane_folder``):
    its own recorded one when it keeps its resource (``own``, else ``keeps_own_resource``: an explicit folder or
    opt-out, or body work), else the room's recorded one (``project``: that room's registry row)."""
    if keeps_own_resource(drive_root, task)[0] if own is None else own:
        return str(task.get("workspace_root") or "") if task.get("workspace_mode") == "external" else ""
    return str((project or {}).get("working_dir") or "").strip()


_OWN = "_placement_own"  # memory only: {bound_at, kind[, own, body]}, a moved row's ``keeps_own_resource`` for its move


def _resource_basis(task: dict, binding: dict) -> dict:
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    return {"bound_at": binding.get("bound_at"), "kind": (metadata.get("resource_intent") or {}).get("kind")}


def known_resource(task: dict, binding: dict) -> tuple[bool, bool] | None:
    """``keeps_own_resource`` for this row's move as memory knows it, reading nothing; None when only the copy
    records can say. An intent decides it unless body work may hold it (none, ``system_repo``); then the answer
    read for this move and intent rides on the row (``settle_resource``, readmission). A queued root adds no
    candidate: only the root prepares its own (``body_candidate._refusal_before_prepare`` refuses its helpers), and
    readmission reads the records again at the launch, so one retired meanwhile moves the lane there."""
    basis = _resource_basis(task, binding)
    if basis["kind"] not in (None, "", "system_repo"):
        return basis["kind"] in {"explicit_resource", "explicit_none"}, False
    carried = task.get(_OWN)
    if isinstance(carried, dict) and "own" in carried and {key: carried.get(key) for key in basis} == basis:
        return bool(carried["own"]), bool(carried.get("body"))
    return None


def pending_resource(task: dict) -> dict | None:
    """The move and intent whose ``keeps_own_resource`` the chooser reads off Q (``dispatch_choice._screen``) before
    it decides the row, else None; until then the row's folder lane is provisionally none."""
    carried = task.get(_OWN)
    if not isinstance(carried, dict) or "own" in carried:
        return None
    return {key: carried.get(key) for key in ("bound_at", "kind")}


def settle_resource(task: dict, basis: dict, own: bool, body: bool) -> bool:
    """Under Q: the answer read off Q for ``basis``, if the row still awaits that one; its folder lane then follows
    from the room's row its refresh admitted (``_project_admission``), reading nothing. Whether it settled."""
    if pending_resource(task) != basis:
        return False
    task[_OWN] = {**basis, "own": own, "body": body}
    task["_placement_folder"] = placement_folder(None, task, (task.get("_project_admission") or {}).get("project"),
                                                 own=own)
    return True


def queued_lane_folder(drive_root: Any, task: dict, *, bindings_snapshot: dict | None = None) -> str:
    """Off Q, for a queued row as a queue publication shows it: the folder lane it holds or waits for, as the
    chooser reads it (a pending move's ``placement_folder`` from its durable binding and that room's row, else
    its recorded folder). A readmitted or RUNNING row's lane is its recorded folder (``project_lease.lane_folder``)."""
    from ouroboros.project_lease import lane_folder
    from ouroboros.projects_registry import get_project

    task = {key: value for key, value in task.items() if key != "_placement_folder"}
    binding = pending_placement(drive_root, task, bindings_snapshot=bindings_snapshot)
    if not binding:
        return lane_folder(task)
    moved = {**task, "project_id": binding["project_id"]}
    room = get_project(drive_root, binding["project_id"], strict=True)
    return lane_folder({**moved, "_placement_folder": placement_folder(drive_root, moved, room)})


def refresh_pending_placement(drive_root: Any, task: dict, *, binding: dict | None = None,
                              lane: bool = False) -> bool:
    """Refresh pending lease identity before selection, keeping folder access unchanged.

    ``lane`` (the chooser's stat read, ``worker_assignment._candidate_ineligible``): the folder lane meanwhile reads
    ``_placement_folder``, the folder this work's readmission will select, so the chooser waits for the folder it
    will work in. Memory decides it (``known_resource``) or it is none until the chooser read the copy records off
    Q (``pending_resource``): nothing is read for it under Q. The queue snapshot carries it to the publication's
    readers. Enqueue, restore, admission and readmission refresh identity only."""
    from ouroboros.sleep_custody import carrier_receipt
    from ouroboros.project_admission import project_admission_basis, project_admission_guard

    try:
        binding = pending_placement(drive_root, task) if binding is None else binding
        if not binding:
            task.pop("_placement_folder", None)
            task.pop(_OWN, None)
            return True
        if carrier_receipt(task).get("retains_project_lane"):
            return False
        with project_admission_guard(drive_root, {**binding["placement_admission"], "frozen": True}) as project:
            task["project_id"] = binding["project_id"]
            task["_project_admission"] = project_admission_basis(binding["project_id"], project, frozen=True)
            task.pop("_project_scope_none", None)
            if lane:
                answer = known_resource(task, binding)
                if answer is None:
                    task[_OWN] = _resource_basis(task, binding)  # the chooser reads it off Q (``pending_resource``)
                task["_placement_folder"] = "" if answer is None else placement_folder(drive_root, task, project,
                                                                                       own=answer[0])
        return True
    except Exception:
        log.warning("Pending placement cannot be readmitted for %s", task.get("id"), exc_info=True)
        return False


def keeps_own_resource(drive_root: Any, task: dict) -> tuple[bool, bool]:
    """``(pinned, body)``: whether readmission leaves this moved work's resource alone, and whether as body work.

    Placement changes the room's default, never a separately selected resource: an explicit
    folder or opt-out, or the candidate of body work. The registry holds that candidate when a
    restart dropped a direct actor's binding, or before a Continue inherits it (prepare).
    Unpinned work takes the room's folder as it stands at that boundary (``project_folder`` names it)."""
    from ouroboros.body_candidate import continuation_predecessor, find

    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    kind = (metadata.get("resource_intent") or {}).get("kind")
    owner = str(metadata.get("root_task_id") or task.get("root_task_id") or task.get("id") or "")
    body = kind in (None, "", "system_repo") and any(
        find(lineage, drive_root, strict=True) for lineage in (owner, continuation_predecessor(task)) if lineage)
    return body or kind in {"explicit_resource", "explicit_none"}, body


def readmit_placement(drive_root: Any, task: dict, *, repo_dir: Any, running: Any, pending: Any) -> bool:
    """Under Q, acquire the selected room's lease and its folder's before changing execution resources.

    ``running``: what holds lanes now, RUNNING with the launches claimed beside it
    (``worker_assignment._started_projection``); a mapping is read by its values."""
    from ouroboros.project_admission import project_admission_basis, project_admission_guard
    from ouroboros.project_lease import candidate_is_leasable, lane_holders, running_project_ids
    from ouroboros.workspace_admission import validate_workspace_root
    from supervisor.conversation_lane import resumes_on_lane

    binding = pending_placement(drive_root, task)
    if not binding:
        return True
    if not refresh_pending_placement(drive_root, task, binding=binding, lane=False):
        return False
    running = list(running.values() if hasattr(running, "values") else running or ())
    leased, holders = running_project_ids(running), lane_holders(running, pending)
    # My human's returned turn holds no room lease, as on its lane (``resumes_on_lane``) and as a fresh turn: neither
    # the room's running writer nor a tree that keeps its lane holds it; pooled work still serializes on both. The
    # room's lane is read first, as before; the folder's once the resource below is selected.
    pooled = not resumes_on_lane(task)
    if pooled and not candidate_is_leasable({**task, "_placement_folder": ""}, leased, holders=holders):
        return False
    metadata = task.setdefault("metadata", {})
    kind = (metadata.get("resource_intent") or {}).get("kind")
    pinned, body = keeps_own_resource(drive_root, task)  # the launch reads it afresh, as it always did
    task[_OWN] = {**_resource_basis(task, binding), "own": pinned, "body": body}  # the chooser's next pass reads it
    # Placement pins room identity; readmission consumes its current generation.
    with project_admission_guard(drive_root, {**binding["placement_admission"], "frozen": True}) as project:
        workspace = {"root": str(task.get("workspace_root") or ""), "mode": str(task.get("workspace_mode") or "")}
        if not pinned:
            folder = validate_workspace_root(project.get("working_dir"), system_repo_dir=repo_dir, drive_root=drive_root)
            workspace = {"root": str(folder or ""), "mode": "external" if folder else ""}
        selected = workspace["root"] if workspace["mode"] == "external" else ""
        if pooled and not candidate_is_leasable({**task, "_placement_folder": selected}, leased, holders=holders):
            # The folder it would select is held: nothing of this work changed; the chooser waits for that folder.
            task["_placement_folder"] = placement_folder(drive_root, task, project, own=pinned)
            return False
        task.pop("_placement_folder", None)
        task.pop(_OWN, None)
        # A Presence turn keeps its source conversation's chat id: every room reader resolves its binding first.
        task.update(project_id=binding["project_id"], chat_id=task.get("chat_id") if metadata.get(
                        "presence_event_identity") else project.get("chat_id"),
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
