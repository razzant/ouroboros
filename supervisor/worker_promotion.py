"""Turning a chat turn — or a project scope — into a queued task.

Resolves where the work came from, binds it to a Project when one is named,
refuses a duplicate of something already live, admits an external workspace only
after proving the tree is a real checkout, and fails the promoted task LOUDLY
rather than leaving a half-admitted row in the queue.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional
from supervisor.state import append_jsonl
from ouroboros.utils import utc_now_iso
from supervisor.queue import _queue_lock




log = logging.getLogger(__name__)


def _pool():
    """The parent module, read at call time.

    The parent owns the rebindable module state and the members tests
    monkeypatch there; reading them through the module at each call keeps
    one binding, where a from-import would freeze the value this leaf saw
    at import time (the owner-approved D18/D33 mechanical exception).
    """
    from supervisor import workers

    return workers


def _origin_from_mapping(mapping: Any, *, absent: str) -> dict:
    """Typed binding origin from an event/metadata mapping (ref passed BY VALUE
    from chat ingress; ``absent`` is the closed-enum reason when none rode along)."""
    source = mapping if isinstance(mapping, dict) else {}
    ref = source.get("origin_message_ref") or source.get("source_ref")
    if isinstance(ref, dict) and ref:
        text = source.get("origin_message_text") or source.get("source_text")
        origin = {"ref": dict(ref)}
        if isinstance(text, str) and text:
            origin["text"] = text
        return origin
    return {"absent": absent}


def _origin_from_task_record(task_id: str) -> Optional[dict]:
    """Ingress-captured origin from the persisted task record.

    A QUEUED task's ctx.task_metadata does not carry the origin (only the task
    dict/record does), so the mid-run ensure_project_scope bind falls back to
    the durable record — mirroring the UI convert path's _owner_task_origin."""
    try:
        # Child-merging reader: a forked/workspace root persists its RUNNING
        # record on its CHILD drive; the effective-status SSOT merges it (same
        # reason gateway/projects.py::_owner_task_origin uses it).
        from ouroboros.task_status import load_effective_task_result

        record = load_effective_task_result(_pool().DRIVE_ROOT, task_id) or {}
        ref = record.get("origin_message_ref")
        text = record.get("origin_message_text")
        if isinstance(ref, dict) and ref and isinstance(text, str) and text.strip():
            return {"ref": dict(ref), "text": text}
    except Exception:
        log.debug("origin task-record lookup failed for %s", task_id, exc_info=True)
    return None


def _report_binding_failure(
    task_id: str, project_id: str, exc: Exception, *, path: str, reason: str = "",
    drive_root: Any = None,
) -> None:
    """A failed durable bind is LOUD (BIBLE P1: silent linkage loss is memory
    loss): warning log + typed events.jsonl row; the task itself keeps running.

    ``reason`` names a REFUSAL that never reached the bind (today
    ``project_scope_conflict``: the task is already bound elsewhere, so no second
    project is created; ``project_binding_unreadable``: the store could not be
    read at all); the row is otherwise the same shape a raising bind writes.
    ``drive_root`` names the store the failure belongs to for a caller that does
    not own the pool root — the reaper passes the queue's drive.
    """
    # A refusal carries no live traceback, so only a real bind failure logs one.
    log.warning("%s for %s/%s (%s)", reason or "bind_task_to_project failed",
                task_id, project_id, path, exc_info=not reason)
    try:
        root = _pool().DRIVE_ROOT if drive_root is None else drive_root
        append_jsonl(root / "logs" / "events.jsonl", {
            "ts": utc_now_iso(),
            "type": "project_binding_failed",
            "task_id": str(task_id or ""),
            "project_id": str(project_id or ""),
            "bind_path": path,
            "error": f"{type(exc).__name__}: {exc}",
            **({"reason": reason} if reason else {}),
        })
    except Exception:
        log.debug("project_binding_failed event write failed", exc_info=True)


def _canonical_promoted_repair_constraint(value: Any) -> tuple[Optional[dict], str]:
    """Pin and validate the authority envelope for a promoted skill repair."""
    from ouroboros.contracts.skill_payload_policy import resolve_constrained_payload_path
    from ouroboros.contracts.task_constraint import TaskConstraint, normalize_task_constraint

    constraint = normalize_task_constraint(value)
    if constraint is None or not (
        constraint.mode == "skill_repair"
        or (constraint.mode == "normal" and (constraint.skill_name or constraint.payload_root))
    ):
        return None, ""
    canonical = TaskConstraint(
        mode="normal",
        skill_name=constraint.skill_name,
        payload_root=constraint.payload_root,
        # Selection is not enable authority. The task resolves its real owner
        # source when it chooses to enable the repaired installation.
        allow_enable=False,
        allow_review=True,
    )
    try:
        payload_dir = resolve_constrained_payload_path(_pool().DRIVE_ROOT, canonical, ".")
    except (TypeError, ValueError):
        return None, "invalid_skill_repair_constraint"
    if not payload_dir.is_dir():
        return None, "skill_repair_payload_missing"
    # X3 (owner 11=B): the repair is admitted against ONE exact payload state.
    # An unreadable payload cannot anchor a hash chain — fail closed here, not
    # after the task has already spent rounds.
    try:
        from ouroboros.skill_loader import compute_content_hash

        base_content_hash = compute_content_hash(payload_dir)
    except Exception:
        return None, "skill_repair_payload_unreadable"
    return {
        "mode": canonical.mode,
        "skill_name": canonical.skill_name,
        "payload_root": canonical.payload_root,
        "allow_enable": canonical.allow_enable,
        "allow_review": True,
        "_base_content_hash": base_content_hash,
    }, ""


def _promote_duplicate_reason(task_id: str, ctx: Any, *, admission_token: str) -> str:
    """Fail closed if a promoted id is already live, durable, or uncheckable.

    A row that is only THIS admission's emitted stub (#1160) is its own
    pre-receipt, not a second owner of the id, so it is read around here.
    """
    pending = getattr(ctx, "PENDING", _pool().PENDING)
    running = getattr(ctx, "RUNNING", _pool().RUNNING)
    with _queue_lock:
        live_duplicate = any(
            isinstance(row, dict) and str(row.get("id") or "") == task_id
            for row in list(pending or [])
        ) or task_id in (running or {})
    try:
        from ouroboros.routing_wait import is_own_admission_stub
        from ouroboros.task_results import load_task_result

        stored = load_task_result(
            getattr(ctx, "DRIVE_ROOT", _pool().DRIVE_ROOT), task_id, strict=True,
        )
        stored_duplicate = bool(stored) and not is_own_admission_stub(
            stored, admission_token,
        )
    except Exception:
        log.warning("promote: duplicate-id lookup failed for %s", task_id, exc_info=True)
        return "task_id_lookup_failed"
    return "duplicate_task_id" if live_duplicate or stored_duplicate else ""


def _promoted_force_plan_metadata(evt: dict) -> dict:
    if evt.get("force_plan") is not True:
        return {}
    source = str(evt.get("force_plan_source") or "operator").strip() or "operator"
    return {"metadata": {"force_plan": True, "force_plan_source": source}}


def _presence_promotion(evt: dict) -> bool:
    """A promote carrying Presence authority: a speaker's, or a delegated descendant's binding."""
    from ouroboros.dialogue_provenance import presence_root_carrier

    return bool(presence_root_carrier(evt, task_contract=evt.get("task_contract")))


def _promote_project_scope(evt: dict) -> str:
    """The project an admitted promote lands in: the explicit one the event carries,
    else the project the OWNER MESSAGE it came from already has.

    The tool already inherits that scope when it emits the event, but the emit →
    admission window is real: a sibling card of the same message can be turned into a
    Project in between, and the root would then arrive in Main as a second convertible
    unit for one piece of work. An IMPLICIT event therefore calls this TWICE: once
    when the handler starts, so the early admission decisions have a scope, and again
    under the claim lock that also holds the durable bind — only the second answer can
    still be true when the bind lands. An event whose scope is already settled returns
    it unchanged, so the second call re-reads nothing. Presence promotion keeps its
    empty scope — the tool-side override already decided that a public conversation
    cannot choose a Project. Fail-open: an unreadable store leaves the scope exactly
    as the event stated it."""
    explicit = str(evt.get("project_id") or "")
    if explicit or _presence_promotion(evt) or not isinstance(evt.get("source_ref"), dict):
        return explicit
    try:
        from ouroboros.projects_registry import origin_claim_lock, project_id_for_origin

        with origin_claim_lock():
            return str(project_id_for_origin(_pool().DRIVE_ROOT, evt.get("source_ref")) or "")
    except Exception:
        log.debug("promote: origin project lookup failed", exc_info=True)
        return ""


def _admit_project_scope(
    evt: dict, task: dict, tid: str, pid: str, attachment_manifest: list,
) -> Optional[dict]:
    """Admit the promoted task into project ``pid``: lifecycle fence, scope on the
    task record, project row, durable bind and project-thread routing; the project
    to announce is stored on the task record and ``promote_chat_to_task`` announces
    it only after ``enqueue_task`` succeeds. Returns the caller's rejection dict, or
    ``None`` when the task is admitted (and when there is no project to admit it into).

    Held by the caller under ``origin_claim_lock``: for an implicit promote the
    origin re-read and this bind are ONE transaction against the UI conversion's
    claim, so a sibling card cannot bind the owner message in between and leave this
    root in Main as a second convertible unit for one piece of work."""
    if not pid:
        return None
    # Deletion closes admission before cancellation/quiescence begins. Check
    # the durable lifecycle before creating projects or child drives;
    # enqueue_task repeats this check atomically under the queue lock.
    try:
        from ouroboros.projects_registry import get_reserved_project

        existing_project = get_reserved_project(_pool().DRIVE_ROOT, pid)
        existing_lifecycle = str((existing_project or {}).get("lifecycle") or "active")
        if existing_project is not None and existing_lifecycle != "active":
            return _pool()._reject_promoted_after_attachment_stage({
                "status": "needs_manual_target",
                "reason": "project_routing_fence",
                "project_lifecycle": existing_lifecycle,
                "detail": f"project lifecycle: {existing_lifecycle}",
                "task_id": tid,
            }, attachment_manifest)
    except Exception:
        log.warning("promote: project admission lookup failed for %s", pid, exc_info=True)
        return _pool()._reject_promoted_after_attachment_stage({
            "status": "needs_manual_target",
            "reason": "project_routing_fence_lookup_failed",
            "task_id": tid,
        }, attachment_manifest)
    task["project_id"] = pid
    # When the model is CREATING a named project (project_name set), pass the
    # human display name so the project isn't named after its bare id (v6.33.0).
    project_display_name = str(evt.get("project_name") or "").strip()
    try:
        from ouroboros.projects_registry import bind_task_to_project, create_project, touch_project

        project = create_project(
            _pool().DRIVE_ROOT, pid, name=project_display_name, origin="promote_chat_to_task",
        )
        touch_project(_pool().DRIVE_ROOT, pid)
        # Bind the task to its project (durable task->project map). Without this
        # the task is project-scoped only in its own metadata; the frontend (via
        # all_task_bindings in /api/state) and the mailbox follow-up router
        # (project_chat_for_task) can't recognise it as a project task, so it
        # surfaces in the main chat with a stray "turn into project" button (P2).
        try:
            # Absence semantics by PROVENANCE (structural, never keyword):
            # a chat-born event carries client_message_id, so a missing ref
            # there is a producer BUG (grep-able producer_missing_ref); an
            # event from a context with no owner message (headless/scheduled/
            # consciousness promote) is a DESIGNED absence.
            absent_reason = (
                "producer_missing_ref"
                if str(evt.get("client_message_id") or "").strip()
                and not evt.get("origin_suppressed")
                else "mid_task_no_origin"
            )
            bind_task_to_project(
                _pool().DRIVE_ROOT,
                tid,
                pid,
                (project or {}).get("chat_id"),
                origin=_origin_from_mapping(evt, absent=absent_reason),
            )
        except Exception as exc:
            _report_binding_failure(tid, pid, exc, path="promote_chat_to_task")
            return _pool()._reject_promoted_after_attachment_stage({
                "status": "needs_manual_target",
                "reason": "project_binding_failed",
                "task_id": tid,
            }, attachment_manifest)
        # The promoted task runs in the PROJECT thread: route its live card +
        # owner mailbox to the project's chat_id (not the main chat it was
        # promoted from) so follow-ups steer to it via
        # _route_project_chat_to_running_task and its progress is visible in
        # the project panel.
        try:
            proj_chat = int((project or {}).get("chat_id") or 0)
        except (TypeError, ValueError):
            proj_chat = 0
        if proj_chat:
            task["chat_id"] = proj_chat
            # The agent just created/bound this project server-side (no client
            # round-trip, unlike the UI "Turn into project" flow). Tell the
            # frontend so it refreshes projectChatIds NOW — otherwise this new
            # project's live frames render in the main chat until the periodic
            # /api/state poll catches up (≤20s) and isMyThread misclassifies them.
            try:
                from supervisor.message_bus import get_bridge

                get_bridge().broadcast({"type": "projects_changed", "project_id": pid, "chat_id": proj_chat})
            except Exception:
                log.debug("promote: projects_changed broadcast failed for %s", pid, exc_info=True)
        if evt.get("_source_created") and not (project or {}).get("created"):
            # The source-resolution half of THIS promote registered the
            # project off-loop (_prepare_promote_source_off_loop) — same
            # agent-initiated creation, so the announce gate honors it.
            project = {**(project or {}), "created": True}
        # The "Project · Started" row is owed only once the task is REALLY in the
        # queue: promote_chat_to_task announces it after enqueue_task succeeded,
        # so a workspace refusal after project creation announces nothing.
        task["_announce_project"] = project
    except Exception:
        log.warning("promote: project registration failed for %s", pid, exc_info=True)
        return _pool()._reject_promoted_after_attachment_stage({
            "status": "needs_manual_target",
            "reason": "project_registration_failed",
            "task_id": tid,
        }, attachment_manifest)
    return None


def bind_retry_to_origin_project(
    drive_root: Any, task: dict, task_id: str, retry_task_id: str,
) -> str:
    """Carry a timed-out root's Project onto the retry that replaces it.

    A retry is the SAME work under a new physical id, so it belongs to the room
    the work already has (DEVELOPMENT "the captured reference is the identity of
    the work"). Before this, the new id copied the origin ref but carried no
    binding, and the retried work painted a Main card offering "Turn into
    project" until some later implicit act adopted it. The predecessor's own
    binding answers first — it is the SSOT for that task's project — and the
    origin-keyed lookup covers a root that was never bound itself while another
    task id of the same owner message was. The new row reuses the predecessor's
    stored origin BY VALUE, so the retry joins that message's one convertible
    unit instead of starting a second.

    The reaper calls this INSIDE its retry admission transaction, under the same
    ``origin_claim_lock`` every implicit claim holds, and only once cancellation
    can no longer win the boundary: ``bind_task_to_project`` is immutable, so a
    bound-but-never-admitted retry id would answer ``project_id_for_task``
    forever. Creates no project and never raises — a refused bind (a project that
    stopped accepting them) or an unreadable store leaves the retry unbound and
    is disclosed as ``project_binding_failed``. Returns the project the retry was
    bound to, "" when there was nothing to inherit.
    """
    tid = str(retry_task_id or "").strip()
    origin_id = str(task_id or "").strip()
    if not tid or not origin_id or tid == origin_id:
        return ""
    from ouroboros.projects_registry import (
        bind_task_to_project,
        project_binding_for_task,
        project_id_for_origin,
    )

    origin = _origin_from_mapping(task, absent="mid_task_no_origin")
    try:
        predecessor = project_binding_for_task(drive_root, origin_id) or {}
        pid = str(predecessor.get("project_id") or "") or str(
            project_id_for_origin(drive_root, origin.get("ref"), strict=True) or ""
        )
    except Exception as exc:
        _report_binding_failure(tid, "", exc, path="timeout_retry_admission",
                                reason="project_binding_unreadable", drive_root=drive_root)
        return ""
    if not pid:
        return ""
    if isinstance(predecessor.get("source_ref"), dict):
        origin = {"ref": dict(predecessor["source_ref"])}
        if isinstance(predecessor.get("source_text"), str):
            origin["text"] = predecessor["source_text"]
    elif predecessor.get("origin_absent"):
        origin = {"absent": str(predecessor["origin_absent"])}
    try:
        bind_task_to_project(drive_root, tid, pid, origin=origin)
    except Exception as exc:
        _report_binding_failure(tid, pid, exc, path="timeout_retry_admission",
                                drive_root=drive_root)
        return ""
    return pid


def promote_chat_to_task(evt: dict, ctx: Any) -> dict:
    """Enqueue a first-class pooled owner task from a conversation-lane promote.
    The task carries the originating ``chat_id`` (its live card and replies
    land in that thread) and the optional ``project_id`` scope; it competes for
    the project writer lease like any other top-level project task.
    """
    from ouroboros.consciousness_authority import apply_consciousness_authority, consciousness_origin_metadata
    from ouroboros.contracts.task_contract import attach_task_contract
    from ouroboros.project_naming import admission_names

    tid = str(evt.get("task_id") or uuid.uuid4().hex[:16])
    admission_token = str(evt.get("routing_token") or "").strip()
    objective = str(evt.get("objective") or "")
    if not objective.strip():
        return {"status": "needs_manual_target", "reason": "empty_objective", "task_id": tid}
    # Reject before project/source/workspace side effects. enqueue_task repeats
    # the check atomically for the tiny race before queue insertion.
    duplicate_reason = _promote_duplicate_reason(tid, ctx, admission_token=admission_token)
    if duplicate_reason:
        return {
            "status": "needs_manual_target",
            "reason": duplicate_reason,
            "task_id": tid,
        }

    evt = dict(evt)
    source_note = str(evt.get("_source_note") or "")
    # IMPLICIT scope: the event named no project, so the answer comes from the owner
    # message's own Project and can still change before the bind (a sibling card
    # converting in the emit -> admission window). Remembered HERE, before the
    # assignment below turns that answer into the event's stated scope.
    implicit_scope = (
        not str(evt.get("project_id") or "")
        and not _presence_promotion(evt)
        and isinstance(evt.get("source_ref"), dict)
    )
    effective_pid = evt["project_id"] = _promote_project_scope(evt)
    repair_constraint, constraint_error = _canonical_promoted_repair_constraint(
        evt.get("task_constraint")
    )
    if constraint_error:
        return {
            "status": "needs_manual_target",
            "reason": constraint_error,
            "task_id": tid,
        }
    try:
        chat_id = int(evt.get("chat_id") or 0)
    except (TypeError, ValueError):
        chat_id = 0
    if not chat_id:
        st = ctx.load_state()
        try:
            chat_id = int(st.get("owner_chat_id") or 0)
        except (TypeError, ValueError):
            chat_id = 0
    expected_output = str(evt.get("expected_output") or "").strip()
    text = objective if not expected_output else f"{objective}\n\nExpected output: {expected_output}"
    title, suggested_name = admission_names(evt, objective)
    task = {
        "id": tid,
        "root_task_id": tid,
        "delegation_role": "root",
        "type": "task",
        "chat_id": chat_id,
        "text": text,
        "description": objective,
        "objective": objective,
        "expected_output": expected_output,
        "title": title,
        "suggested_name": suggested_name,
        "source": "promote_chat_to_task",
        "objective_author": dict(evt.get("objective_author") or {}),
        "owner_corpus": list(evt.get("owner_corpus") or []),
        "_require_unique_task_id": True,
        "_require_worker_pool": True,
        "_admission_token": admission_token,
        "promotion_admission_token": admission_token,
        **_promoted_force_plan_metadata(evt),
    }
    origin = consciousness_origin_metadata(evt)
    if origin:
        # A root consciousness started keeps its origin/category/level by value (В9':
        # ordinary Main flow, no presence-style project/workspace/source stripping).
        task["actor_id"] = "consciousness"
        task.setdefault("metadata", {}).update(origin)
        apply_consciousness_authority(task)
    inherited_attachment_manifest = _pool()._apply_presence_promotion_authority(
        evt, task, objective=objective, expected_output=expected_output,
    )
    attachment_manifest, attachment_rejection = _pool()._stage_promoted_initial_attachments(
        evt, task, tid, inherited_manifest=inherited_attachment_manifest,
    )
    if attachment_rejection is not None:
        return attachment_rejection
    if repair_constraint is not None:
        # X3: bind the admission hash to the REAL task id, durably, before the
        # task exists anywhere else — every payload write CAS-checks this chain.
        # FAIL CLOSED, like the unreadable-payload branch above: a repair admitted
        # without its binding CAS-checks nothing (every later check no-ops), which
        # is precisely the drift-blind repair this mechanism replaces.
        _base_content_hash = str(repair_constraint.pop("_base_content_hash", "") or "")
        try:
            from ouroboros.skill_repair_admission import record_repair_admission

            record_repair_admission(
                _pool().DRIVE_ROOT, str(repair_constraint.get("skill_name") or ""),
                task_id=tid, base_content_hash=_base_content_hash)
        except Exception:
            log.warning("Failed to record skill repair admission for %s", tid, exc_info=True)
            return _pool()._reject_promoted_after_attachment_stage({
                "status": "needs_manual_target",
                "reason": "skill_repair_admission_unwritable",
                "task_id": tid,
            }, attachment_manifest)
        # Bind the selected resource before attach_task_contract. This remains
        # an ordinary managed task; the selection is not a reduced tool profile.
        task["task_constraint"] = repair_constraint
    # Ingress-captured origin identity rides the task record (post-hoc UI convert
    # reads it from the persisted result — never re-derived from content).
    if isinstance(evt.get("source_ref"), dict) and evt.get("source_ref"):
        task["origin_message_ref"] = dict(evt["source_ref"])
        if isinstance(evt.get("source_text"), str) and evt.get("source_text"):
            task["origin_message_text"] = evt["source_text"]
    elif evt.get("origin_suppressed") is True:
        # The door's other stamp (an owner message it never logged) rides the root
        # in METADATA, where run_origin reads it, the way a ref rides by value.
        task.setdefault("metadata", {})["origin_suppressed"] = True
    if task.get("objective_author"):
        task.setdefault("metadata", {})["objective_author"] = dict(task["objective_author"])
    if isinstance(evt.get("predecessor_authority_source"), dict):
        task["predecessor_authority_source"] = dict(evt["predecessor_authority_source"])
    # Owner Surface Fact: the promoting turn's sending-surface fact lands in
    # METADATA (the renderer reads task["metadata"]["client_surface"]), never a
    # top-level key — and metadata may not exist yet (only force_plan creates it).
    if isinstance(evt.get("client_surface"), dict) and evt.get("client_surface"):
        task.setdefault("metadata", {})["client_surface"] = dict(evt["client_surface"])
    pid = str(evt.get("project_id") or "").strip()
    from ouroboros.projects_registry import origin_claim_lock

    with origin_claim_lock():
        if implicit_scope:
            # Re-resolved under the SAME lock that holds the bind below (an RLock, so
            # the helper's own acquire is free). A sibling card of this owner message
            # may have created its Project since this handler started; reading the
            # origin once, a hundred lines before the bind, is what let the promoted
            # root arrive in Main as a second convertible unit for one message.
            pid = evt["project_id"] = _promote_project_scope(evt)
        rejection = _admit_project_scope(evt, task, tid, pid, attachment_manifest)
    if rejection is not None:
        return rejection
    # The answer the task was actually admitted with, so the promote's own receipt
    # names the room the binding names.
    effective_pid = pid
    # Workspace admission (v6.58.0 SSOT + the Q10=A auto-provision) lives in one
    # helper so this entry point stays readable and under the method gate.
    workspace_outcome = _admit_promoted_workspace(evt, ctx, task, pid=pid, tid=tid)
    if workspace_outcome is not None:
        return _pool()._reject_promoted_after_attachment_stage(
            workspace_outcome, attachment_manifest,
        )
    if not _pool()._relocate_promoted_attachments(task, tid, attachment_manifest):
        return _pool()._reject_promoted_after_attachment_stage({
            "status": "needs_manual_target",
            "reason": "attachment_admission_rejected",
            "detail": "Attachment staging could not be finalized (reason=staging_unavailable).",
            "attachment_manifest": [
                {
                    "ordinal": row.get("ordinal", index),
                    "status": "rejected",
                    "reason": "staging_unavailable",
                    "label": str(row.get("label") or f"attachment {index + 1}"),
                }
                for index, row in enumerate(attachment_manifest)
            ],
            "task_id": tid,
        }, attachment_manifest)
    if attachment_manifest:
        from ouroboros.gateway.tasks import _render_attachment_lines

        from ouroboros.artifacts import attachment_manifest_projection
        authority = attachment_manifest_projection(task.get("drive_root") or _pool().DRIVE_ROOT, tid, attachment_manifest)
        task.update(authority)
        task["text"] = f"{task['text']}\n\n[ATTACHMENTS]\n{_render_attachment_lines(authority)}\n[END_ATTACHMENTS]"
        task["attachments"] = authority["attachment_manifest"]
        task["attachment_images"] = [row for row in attachment_manifest if row.get("is_image")]
        if isinstance(task.get("task_contract"), dict):
            task["task_contract"].update(authority)
    # Popped BEFORE the row is built/queued so the announce fact never rides it.
    announce_project = task.pop("_announce_project", None)
    attach_task_contract(task)
    admitted = ctx.enqueue_task(task)
    if isinstance(admitted, dict) and admitted.get("_admission_blocked"):
        return _pool()._reject_promoted_after_attachment_stage({
            "status": "needs_manual_target",
            "reason": str(admitted.get("_admission_blocked") or "admission_fence"),
            "detail": str(admitted.get("_admission_detail") or ""),
            "project_lifecycle": str(admitted.get("_project_lifecycle") or ""),
            "task_id": tid,
        }, attachment_manifest)
    if announce_project is not None:
        _pool()._announce_created_project(announce_project, tid, task=task)
    # Owner 3=A: the promoter's unmet planning obligation now belongs to this root
    # (stamped by _promoted_force_plan_metadata above); release it on the promoter's
    # live row BEFORE the snapshot persist below, so one persist shows both facts.
    from supervisor.plan_obligation import transfer_promoter_obligation

    obligation_transfer = transfer_promoter_obligation(ctx, evt, tid)
    # A positive promote confirmation is allowed only after the durable queue
    # projection exists.  The event handler writes the scheduled task result
    # after the routing receipt; keeping that last step outside this function
    # makes the result itself the cross-process admission receipt.
    persist_snapshot = getattr(ctx, "persist_queue_snapshot", None)
    if not callable(persist_snapshot):
        return _pool()._reject_promoted_after_attachment_stage({
            "status": "needs_manual_target",
            "reason": "queue_snapshot_persist_unavailable",
            "task_id": tid,
            "admission_started": True,
        }, attachment_manifest)
    try:
        persisted = persist_snapshot(reason="promote_chat_to_task") is not False
    except Exception:
        log.warning("promote: queue snapshot persist failed for %s", tid, exc_info=True)
        persisted = False
    if not persisted:
        return _pool()._reject_promoted_after_attachment_stage({
            "status": "needs_manual_target",
            "reason": "queue_snapshot_persist_failed",
            "task_id": tid,
            "admission_started": True,
        }, attachment_manifest)
    # v6.82 (P5) disclosed residual: a PROMOTED root carries the host-attested
    # `cancelable` marker from its first RUNNING relay, not from enqueue — the
    # promote path emits no owner-facing progress frame of its own, and minting a
    # marker-only bubble would either add chat noise or bypass the canonical
    # message seam (tests/test_heartbeat_presentation.py). While it is still
    # PENDING the Dashboard Activity row cancels it; the card action appears once
    # it starts.
    # A project root may execute from a forked child drive.  Its budget-root
    # result therefore receives this admitted contract before worker startup.
    outcome = _pool()._promoted_scheduled_outcome(task, admitted, tid)
    outcome["_admitted_suggested_name"] = task["suggested_name"]
    if attachment_manifest:
        outcome.update(authority)
    if effective_pid:
        outcome["project_id"] = effective_pid
    if source_note:
        outcome["source_note"] = source_note
    if obligation_transfer:
        outcome["force_plan_transfer"] = obligation_transfer
    return outcome


def _admit_promoted_workspace(evt: dict, ctx: Any, task: dict, *, pid: str, tid: str) -> Optional[dict]:
    """Bind the promoted task's active workspace, or return a failure outcome.

    Extracted verbatim from ``promote_chat_to_task`` (v6.90.x submarine unwind) to
    keep that function under the hard method gate; the admission SEQUENCE is
    unchanged. Returns ``None`` when the task was bound (or legitimately has no
    workspace) and mutates ``task`` in place; returns a ``needs_manual_target``
    outcome dict when admission must fail LOUDLY.
    """
    # v6.58.0 (slice 1) — the promote path admits a workspace through the SAME SSOT
    # as /api/tasks. A task born in a project ROOM defaults to the room's registered
    # working_dir (workspace="none" on the event opts out); a SET-but-broken
    # working_dir fails LOUDLY here — never a silent workspace-less task that would
    # resolve to the self_modification profile over the system repo.
    from ouroboros.workspace_admission import (
        WORKSPACE_NONE,
        bounded_workspace_preflight,
        compose_workspace_block,
        resolve_room_workspace,
        workspace_repair_hint,
    )

    if _presence_promotion(evt):
        # Keep the admitted folder from the inherited contract, never a public
        # event's replacement. Presence retains its canonical shared memory.
        workspace = task["task_contract"].get("workspace") or {}
        resolved_ws, ws_error = resolve_room_workspace(
            drive_root=_pool().DRIVE_ROOT, system_repo_dir=_pool().REPO_DIR,
            project_id="", explicit_workspace=str(workspace.get("root") or ""),
        )
        if ws_error:
            return {
                "status": "needs_manual_target", "reason": "workspace_unusable", "task_id": tid,
                "detail": workspace_repair_hint(ws_error=ws_error, presence=True),
            }
        if resolved_ws:
            task.update(workspace_root=resolved_ws, workspace_mode="external", memory_mode="shared")
        return None

    # Q10=A (owner, 2026-08-08): a project promoted with NO working folder gets one
    # AUTO-PROVISIONED via the existing ensure_project_workspace seam (an idempotent
    # standalone git repo under the durable subagent_projects root — passes the same
    # validate_workspace_root SSOT below). This binds the task's real tree as its
    # active workspace, fixing path/cwd confinement, the external tool profile and
    # the one-writer lease for the file-less project class (the submarine shape).
    # STRICTLY empty-only: a NON-EMPTY working_dir — valid or broken — is never
    # blind-ensured over (a broken one must LOUD-FAIL through resolve_room_workspace,
    # the v6.58.0 invariant, not be papered over with a fresh empty repo). The
    # workspace="none" sentinel still opts out entirely. Docs are NOT part of this
    # decision: since D-ARCH (2026-08-08) the doc matrix keys on project membership
    # and the owner mode, so binding a workspace here never drags ARCHITECTURE.md
    # out of a max context.
    if (
        pid
        and not str(evt.get("workspace_root") or "").strip()
        and str(evt.get("workspace") or "").strip().lower() != WORKSPACE_NONE
    ):
        provisioned_now = ""
        try:
            from ouroboros.projects_registry import get_project as _get_project_entry

            _existing_wd = str((_get_project_entry(_pool().DRIVE_ROOT, pid) or {}).get("working_dir") or "").strip()
        except Exception:
            # Registry read failure: do NOT provision (a blind ensure here could
            # mint a fresh empty repo over a project whose working_dir merely
            # failed to load). resolve_room_workspace re-reads and decides.
            _existing_wd = "unreadable"
            log.warning("promote: project working_dir lookup failed for %s", pid, exc_info=True)
        if not _existing_wd:
            try:
                from ouroboros.projects_registry import ensure_project_workspace

                provisioned_now = str(ensure_project_workspace(_pool().DRIVE_ROOT, pid, _pool().REPO_DIR) or "")
            except Exception:
                provisioned_now = ""
                log.warning("promote: workspace auto-provisioning raised for %s", pid, exc_info=True)
            if not provisioned_now:
                # Bind-or-fail (v6.58.0): falling through to a workspace-less
                # self_modification-profile task over the system repo is exactly
                # the silent degradation the admission SSOT exists to kill.
                return {
                    "status": "needs_manual_target",
                    "reason": "workspace_provisioning_failed",
                    "task_id": tid,
                    "detail": workspace_repair_hint(
                        ws_error=f"project {pid!r} has no working folder and auto-provisioning "
                        "one failed; see the supervisor log (ensure_project_workspace)",
                    ),
                }
            task.setdefault("metadata", {})["workspace_autoprovisioned"] = True

    resolved_ws, ws_error = resolve_room_workspace(
        drive_root=_pool().DRIVE_ROOT,
        system_repo_dir=_pool().REPO_DIR,
        project_id=pid,
        explicit_workspace=str(evt.get("workspace_root") or "").strip(),
        workspace_sentinel=str(evt.get("workspace") or ""),
    )
    if ws_error:
        return {
            "status": "needs_manual_target", "reason": "workspace_unusable", "task_id": tid,
            "detail": workspace_repair_hint(
                ws_error=ws_error, explicit_workspace=str(evt.get("workspace_root") or "").strip(),
                project_id=pid, drive_root=_pool().DRIVE_ROOT, system_repo_dir=_pool().REPO_DIR,
            ),
        }
    if resolved_ws:
        task["workspace_root"] = resolved_ws
        task["workspace_mode"] = "external"
        task["memory_mode"] = "forked"
        if pid and str(evt.get("workspace_root") or "").strip():
            # An explicit folder for a room that has none yet becomes the room's
            # folder: the same validated canonical path this task runs in, written
            # only while `working_dir` is still empty (compare-and-set under the
            # registry lock), so a set value is never overwritten and the room's
            # later direct turns are not blind to where the work went.
            try:
                from ouroboros.projects_registry import update_project

                update_project(_pool().DRIVE_ROOT, pid, working_dir=resolved_ws, only_if_empty=("working_dir",))
            except Exception:
                log.warning("promote: could not record working_dir for project %s", pid, exc_info=True)
        # The lease lane keys off task["project_id"]: for a project room it is already
        # set; for a bare workspace promote, resolve it (registry-first → derived hash)
        # so one folder is one serialized lane on EVERY entry path (slice 0 invariant).
        if not str(task.get("project_id") or "").strip():
            try:
                from ouroboros.project_facts import resolve_project_id as _resolve_pid

                derived_pid = _resolve_pid({"workspace_root": resolved_ws})
                if derived_pid:
                    task["project_id"] = derived_pid
            except Exception:
                log.debug("promote: project_id derivation failed for %s", tid, exc_info=True)
        # Memory-fork parity with /api/tasks: the room task runs on an ISOLATED child
        # drive (forked seed), with the canonical root kept for budget/status.
        try:
            from ouroboros.headless import prepare_task_drive

            child_drive = prepare_task_drive(
                _pool().DRIVE_ROOT, tid, "forked", project_id=str(task.get("project_id") or "")
            )
            if child_drive is not None:
                task["drive_root"] = str(child_drive)
                task["budget_drive_root"] = str(_pool().DRIVE_ROOT)
        except Exception:
            log.warning("promote: child drive fork failed for %s", tid, exc_info=True)
        # Preflight parity, HARD-CAPPED: this runs on the supervisor event-drain
        # thread, so the git/toolchain snapshot gets a bounded window and degrades
        # to a disclosed skip note instead of stalling event delivery.
        preflight_summary = bounded_workspace_preflight(resolved_ws)
        metadata = task.setdefault("metadata", {})
        metadata["workspace_root"] = resolved_ws
        metadata["workspace_preflight"] = preflight_summary
        task["text"] = (
            f"{task['text']}\n\n[HEADLESS_WORKSPACE]\n"
            + compose_workspace_block(
                workspace_root=resolved_ws,
                workspace_mode="external",
                memory_mode="forked",
                workspace_preflight=preflight_summary,
            )
            + "[END_HEADLESS_WORKSPACE]"
        )
    return None


def ensure_project_scope(evt: dict, ctx: Any) -> dict:
    """Create/attach the registry project for an in-task ensure_project_scope call
    and bind the CURRENT (already-running) task to it, then broadcast so the UI moves
    the card into the project thread. Mirrors the project-registration half of
    promote_chat_to_task, but for a task that already exists (the worker has already
    set ctx.project_id locally; this makes it durable + visible). Returns the typed
    outcome the receipt rail reports: ``delivered`` with the bound project, or
    ``rejected`` with the reason (bound elsewhere, a refused bind, a registration
    failure) -- the tool never says "OK" for a bind that did not land."""
    tid = str(evt.get("task_id") or "").strip()
    pid = str(evt.get("project_id") or "").strip()
    if not tid or not pid:
        return {"status": "rejected", "reason": "missing_task_or_project"}
    name = str(evt.get("project_name") or "").strip()
    try:
        from ouroboros.projects_registry import (
            bind_task_to_project,
            create_project,
            origin_claim_lock,
            project_id_for_origin,
            project_id_for_task,
            touch_project,
            update_project,
        )

        # The task's ingress-captured origin, resolved BEFORE the authority read (it
        # answers both questions): event metadata, then the live RUNNING task dict
        # (queued tasks carry no origin in ctx.task_metadata, and this also covers
        # forked/workspace roots whose running record lives on a CHILD drive), then
        # the durable task record on the canonical drive — the mid-run "make this a
        # project named X" path must keep the start message.
        origin = _origin_from_mapping(evt, absent="mid_task_no_origin")
        if "absent" in origin:
            running = getattr(ctx, "RUNNING", None)
            running_row = running.get(tid) if isinstance(running, dict) else None
            task_row = running_row.get("task") if isinstance(running_row, dict) else None
            candidate = _origin_from_mapping(task_row, absent="mid_task_no_origin")
            if "ref" in candidate and "text" in candidate:
                origin = candidate
        if "absent" in origin:
            origin = _origin_from_task_record(tid) or origin
        # The claim lock is the one the UI conversion and the promote handler hold,
        # so a card of the SAME owner message cannot claim a project between this
        # read and this bind.
        with origin_claim_lock():
            # This task's EXACT binding is the AUTHORITY, read BEFORE any side effect
            # (owner decision B4=A). create_project runs before bind_task_to_project here,
            # so a task already bound elsewhere used to mint a second project, mark the
            # lease, broadcast it and announce it in chat before the immutable bind
            # refused. An UNREADABLE store is disclosed once and treated as "no binding":
            # the work continues as it did before this read existed.
            try:
                bound = str(project_id_for_task(_pool().DRIVE_ROOT, tid, strict=True) or "")
                # The project the owner MESSAGE already has answers for a SIBLING task id
                # of the same message — but only when this act does not name a different
                # room: an explicit name/id is the model's own choice (P13) and forks.
                adopted = "" if bound else str(
                    project_id_for_origin(_pool().DRIVE_ROOT, origin.get("ref"), strict=True) or ""
                )
            except Exception:
                bound, adopted = "", ""
                log.warning("project_binding_unreadable: ensure_project_scope for task %s continues "
                            "as unbound", tid, exc_info=True)
            if bound and bound != pid:
                # Bound elsewhere: the request is a RENAME of the project this task already
                # belongs to, never a second project. Nothing else happens - no create, no
                # lease mark, no broadcast, no announcement.
                rename = "name_unchanged"
                if name:
                    try:
                        from ouroboros.projects_registry import get_project

                        row = get_project(_pool().DRIVE_ROOT, bound) or {}
                        if str(row.get("name") or "") != name:
                            update_project(_pool().DRIVE_ROOT, bound, name=name)
                            rename = "renamed"
                    except Exception:
                        rename = "rename_failed"
                        log.warning("ensure_project_scope: rename of %s to %r failed", bound, name, exc_info=True)
                _report_binding_failure(
                    tid, pid,
                    ValueError(f"task is already bound to project {bound!r}; it stays there"),
                    path="ensure_project_scope", reason="project_scope_conflict",
                )
                return {"status": "rejected", "reason": "project_scope_conflict",
                        "project_id": bound, "detail": rename}

            project = create_project(_pool().DRIVE_ROOT, pid, name=name, origin="ensure_project_scope")
            touch_project(_pool().DRIVE_ROOT, pid)
            try:
                proj_chat = int((project or {}).get("chat_id") or 0)
            except (TypeError, ValueError):
                proj_chat = 0
            try:
                bind_task_to_project(_pool().DRIVE_ROOT, tid, pid, proj_chat or None, origin=origin)
            except Exception as exc:
                # A refused bind leaves the task where it was: stop before the lease
                # mark, the broadcast and the announcement (the promote path already
                # rejects this way), instead of publishing a project the task is not in.
                _report_binding_failure(tid, pid, exc, path="ensure_project_scope")
                return {"status": "rejected", "reason": "project_binding_failed",
                        "project_id": pid, "detail": f"{type(exc).__name__}: {exc}"}
        # Make the one-writer-per-project lease recognize THIS already-running task
        # as a lane occupant: project_lease reads task["project_id"] from the
        # supervisor RUNNING map, which (unlike the promote path that sets it at
        # build time) is NOT set for a mid-flight self-scope. Without this, a task
        # that self-scopes to project X would not hold X's lane and a concurrent
        # X task could be assigned and write the same project. SSOT helper shared
        # with the UI api_project_from_task convert path so the two cannot drift.
        # A task ADOPTING the project its owner message already has owns the binding
        # it just wrote, so its lane FOLLOWS that truth (authority="binding") even
        # when the row still carries a derived id from a bare-workspace promote.
        try:
            from ouroboros.project_lease import mark_task_project

            running = getattr(ctx, "RUNNING", None)
            pending = getattr(ctx, "PENDING", None)
            if isinstance(running, dict):
                with _queue_lock:
                    mark_task_project(running, pending, tid, pid,
                                      authority="binding" if adopted == pid else "")
        except Exception:
            log.debug("ensure_project_scope: RUNNING project_id update failed for %s", tid, exc_info=True)
        if proj_chat:
            try:
                from supervisor.message_bus import get_bridge

                get_bridge().broadcast({"type": "projects_changed", "project_id": pid, "chat_id": proj_chat})
            except Exception:
                log.debug("ensure_project_scope: projects_changed broadcast failed for %s", pid, exc_info=True)
        running = getattr(ctx, "RUNNING", None)
        row = running.get(tid) if isinstance(running, dict) else None
        _pool()._announce_created_project(
            project, tid, task=row.get("task") if isinstance(row, dict) else None,
        )
        return {"status": "delivered", "project_id": pid, "chat_id": proj_chat,
                "reason": "created" if (project or {}).get("created") else ("adopted" if adopted == pid else "attached")}
    except Exception as exc:
        log.warning("ensure_project_scope: project registration failed for %s", pid, exc_info=True)
        return {"status": "rejected", "reason": "project_registration_failed",
                "project_id": pid, "detail": f"{type(exc).__name__}: {exc}"}
