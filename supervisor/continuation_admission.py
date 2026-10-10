"""The owner's Continue, queue side: one action identity, admitted once (Batch4 1A/3A).

``admit_continuation`` is the one transaction behind ``POST
/api/tasks/{id}/continue``; its order is the protocol (data half:
``ouroboros/owner_continue.py``):

1. REPLAY FIRST. The successor id is derived from ``(predecessor, nonce)``;
   an existing successor row carrying the SAME full binding is returned as
   the recorded admission at any status (still queued, held, running, ended).
   A row under that id with another binding refuses. A queue row that exists
   without its result row (a stop between the snapshot and the result write)
   is recovered as the same admission, never admitted twice.
2. CLAIM on the predecessor before any effect (``claim_on_predecessor``):
   eligibility, the complete owner sources and the binding (its room read from
   the predecessor's canonical Project binding, ``continuation_room``) are
   settled first; a write that fails refuses with nothing done. The same nonce
   resumes its own claim and frozen room; another nonce is answered with the
   accepted successor.
3. ADMISSION through the existing reservation and durable enqueue (queue row,
   persisted snapshot, then the result row under the queue lock). A
   predecessor whose writers are not proven settled — a RUNNING or
   dispatchable member of its tree, a delegated run its custody still holds
   or cannot read — admits the SAME task under a typed non-dispatch hold
   (``continuation_writer_unsettled``): no diagnostic successor, no second
   writer, no timeout release. The owner's existing conversation and controls
   diagnose; the hold releases only when a fresh check finds the writers
   settled (the owner's Resume, or a reconciliation at a tree member's
   terminal) and no newer Stop/Pause/Restart hold has replaced it.

A request that times out on the client is never "not admitted": the client
retries with the same nonce and step 1 answers it.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Optional

from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)

from supervisor.events_budget import HOLD_CONTINUATION_WRITER  # noqa: E402 -- the one hold vocabulary


def _recorded_admission(result: Dict[str, Any], predecessor: str, nonce: str) -> Optional[Dict[str, Any]]:
    from ouroboros.owner_continue import binding_sha

    admission = result.get("continuation_admission") if isinstance(result.get("continuation_admission"), dict) else {}
    binding = admission.get("binding") if isinstance(admission.get("binding"), dict) else {}
    if not binding:
        return None
    if (str(binding.get("predecessor_task_id") or "") != predecessor
            or str(binding.get("action_nonce") or "") != nonce
            or str(admission.get("binding_sha256") or "") != binding_sha(binding)):
        return {"mismatch": True}
    return admission


def _replay(q: Any, predecessor: str, nonce: str, successor: str) -> Optional[Dict[str, Any]]:
    from ouroboros.owner_continue import binding_sha, mark_claim_admitted
    from ouroboros.task_results import load_task_result, write_task_result

    try:
        with q._queue_lock:
            stored = load_task_result(q.DRIVE_ROOT, successor, strict=True) or {}
            row = next((item for item in q.PENDING if str(item.get("id") or "") == successor), None)
            if row is None:
                row = (q.RUNNING.get(successor) or {}).get("task")
            if not stored and row is None:
                return None
            claim = (load_task_result(q.DRIVE_ROOT, predecessor, strict=True) or {}).get("continued_by") or {}
            admission = _recorded_admission(stored, predecessor, nonce)
            recovered = admission is None
            if admission is None and row is not None:
                admission = ((row.get("metadata") or {}).get("continuation") or {}).get("admission")
            if admission is None:
                admission = ((stored.get("metadata") or {}).get("continuation") or {}).get("admission")
            binding = (admission or {}).get("binding") or {}
            if (not binding or _recorded_admission({"continuation_admission": admission}, predecessor, nonce)
                    in (None, {"mismatch": True}) or binding != claim.get("binding")
                    or claim.get("binding_sha256") != binding_sha(binding)
                    or binding.get("successor_task_id") != successor):
                return {"ok": False, "error": "continuation_identity_conflict", "successor_task_id": successor}
            if recovered:
                write_task_result(q.DRIVE_ROOT, successor, stored.get("status") or "scheduled",
                                  continuation_admission=admission, metadata=(row or stored).get("metadata"),
                                  root_task_id=successor, chat_id=binding.get("chat_id"),
                                  project_id=binding.get("project_id") or "",
                                  **{k: v for k, v in (row or {}).items()
                                     if k in {"deadline_at", "reasoning_effort", "workspace_root", "workspace_mode"}
                                     and v})
            if row is not None and row.get("_continuation_prepared"):
                prepared = row.pop("_continuation_prepared")
                if q.persist_queue_snapshot(reason="owner_continue_binding_recovered") is not True:
                    row["_continuation_prepared"] = prepared
                    return {"ok": False, "error": "queue_snapshot_persist_failed",
                            "successor_task_id": successor, "unconfirmed": True}
            mark_claim_admitted(q.DRIVE_ROOT, predecessor, nonce)
            return {"ok": True, "replay": True, "recovered": recovered, "task_id": successor,
                    "successor_task_id": successor, "predecessor_task_id": predecessor,
                    "status": str(stored.get("status") or "scheduled"),
                    "held": bool(admission.get("held"))}
    except Exception as exc:
        return {"ok": False, "error": "continuation_publication_unconfirmed",
                "successor_task_id": successor, "unconfirmed": True, "detail": str(exc)[:200]}


def conflicting_writers(q: Any, predecessor: str, *, drive_root: Any = None,
                        owner_pause_fence_id: str = "") -> List[Dict[str, Any]]:
    """Observe all retained tree custody, regardless of the member's lifecycle.

    Queue authority is copied briefly; custody observations never cancel runs.
    Unknown files or unbound starts keep the same Continue action held.
    ``owner_pause_fence_id`` is the owner Pause's own census: a queued admitted
    dispatch that only that exact accepted Pause's latch holds is not running
    work (its earlier effects are still observed below). Continue never passes
    it: that row would dispatch again once the Pause is resumed. The answered
    predecessor's own late phase (D10) is custody too: running or open, or
    paused (Resume would write again) — the Pause's own census alone reads its
    saved pause as settled.
    """
    from ouroboros.budget_pause import observe_task_runs
    from ouroboros.owner_pause import tree_member_results
    from supervisor.queue_transitions import budget_pause_fact, queued_admitted_dispatch

    custody_root = pathlib.Path(drive_root or q.DRIVE_ROOT)
    blockers: List[Dict[str, Any]] = []
    members = {predecessor}
    member_results = {}
    from supervisor.owner_pause_control import warm_paused_member

    with q._queue_lock:
        running = [(str(tid), dict(meta.get("task") or {}),
                    warm_paused_member(meta) if owner_pause_fence_id else False)
                   for tid, meta in q.RUNNING.items() if isinstance(meta, dict)]
        pending = [dict(task) for task in q.PENDING if isinstance(task, dict)]
        latch = dict(q.BUDGET_ROOT_FENCES.get(predecessor) or {})
    owner_latched = bool(owner_pause_fence_id and latch.get("cause") == "owner_pause"
                         and str(latch.get("status") or "") in {"active", "paused"}
                         and str(latch.get("fence_id") or "") == str(owner_pause_fence_id))
    for task_id, task, warm in running:
        if str(task.get("root_task_id") or task_id) == predecessor:
            members.add(task_id)
            if warm:
                continue  # saved warm under the owner's Pause: its stack waits, it writes nothing
            blockers.append({"kind": "running_member", "task_id": task_id})
    from ouroboros.post_task_checkpoint import late_phase_state

    late = late_phase_state(custody_root, predecessor)
    if late and not (late == "paused" and owner_pause_fence_id):
        blockers.append({"kind": f"late_phase_{late}", "task_id": predecessor})
    for task in pending:
        tid = str(task.get("id") or "")
        if str(task.get("root_task_id") or tid) == predecessor:
            members.add(tid)
            if owner_latched and queued_admitted_dispatch(task):
                continue
            if budget_pause_fact(task) is None or (task.get("admitted_dispatch") == "possible"
                                                   and not task.get("_budget_pause")):
                blockers.append({"kind": "dispatchable_member", "task_id": tid})
    try:
        member_results = tree_member_results(custody_root, predecessor)
        for task_id, row in member_results.items():
            members.add(task_id)
            from ouroboros.tool_custody import retained_tool_custody
            blockers.extend(retained_tool_custody(custody_root, task_id, row))
        # A member whose result has been collected can still own a remote start.
        from ouroboros.delegate_custody_memo import custody_rows_with_integrity
        rows, malformed = custody_rows_with_integrity(custody_root, predecessor)
        if malformed is None or malformed:
            raise OSError("tree_custody_unreadable")
        for row in rows:
            if str(row.get("root_task_id") or row.get("task_id") or "") == predecessor:
                members.add(str(row.get("task_id") or predecessor))
    except Exception as exc:
        blockers.append({"kind": "tree_census_unreadable", "detail": str(exc)[:200]})
    for member in sorted(members):
        observed = observe_task_runs(custody_root, member, reason="continuation_writer_check", request_stop=False)
        if observed.get("custody_read") != "ok":
            blockers.append({"kind": "custody_unreadable", "task_id": member})
        else:
            blockers.extend({"kind": "delegated_run", "task_id": member,
                             "run_id": str(run.get("run_id") or ""),
                             "invocation_id": str(run.get("invocation_id") or ""),
                             # The owner Pause's consumers read it; Continue's hold does not.
                             "review_owned": bool(run.get("review_owned"))}
                            for run in observed.get("runs") or [] if isinstance(run, dict))
    try:
        from ouroboros import process_custody as pc
        from ouroboros.platform_layer import pid_is_alive
        from ouroboros.tool_custody import task_process_blockers
        blockers.extend(task_process_blockers(custody_root, members))
        complete, records = pc._read_ledger_strict(custody_root)
        if not complete:
            raise OSError("process_custody_unreadable")
        for row in records:
            if str(row.get("owner_task") or "") in members:
                pid = int(row.get("pid") or 0)
                if pid and (pid_is_alive(pid) or pc._service_group_survives_leader(row)):
                    blockers.append({"kind": "owned_process", "task_id": row.get("owner_task"), "pid": pid})
    except Exception as exc:
        blockers.append({"kind": "process_custody_unreadable", "detail": str(exc)[:200]})
    try:
        from ouroboros import usage_store
        # The root's open attempts only (the store's open-set index).
        with usage_store.read(custody_root) as txn:
            attempts = txn.open_attempts(root_task_id=predecessor)
        for attempt in attempts:
            if attempt.get("state") in {"dispatched", "unresolved"}:
                consumer = attempt.get("local_answer_consumer_id")
                retired = (member_results.get(str(attempt.get("task_id") or ""), {})
                           .get("retired_model_consumers") or {}).get(consumer) if consumer else None
                if (isinstance(retired, dict) and retired.get("retired_at")
                        and type(attempt.get("local_answer_task_attempt")) is int
                        and retired.get("task_attempt") == attempt["local_answer_task_attempt"]):
                    continue
                # A reviewer's own send (its slot/skill attribution) is review
                # custody: the owner Pause's readers list it as still finishing.
                blockers.append({"kind": "model_handoff", "attempt_id": attempt.get("attempt_id"),
                                 "review_owned": bool(attempt.get("review_slot_id") or attempt.get("review_skill"))})
    except Exception as exc:
        blockers.append({"kind": "attempt_custody_unreadable", "detail": str(exc)[:200]})
    return blockers


def action_writers(q: Any, predecessor: str, *, drive_root: Any = None,
                   owner_pause_fence_id: str = "") -> List[Dict[str, Any]]:
    """An ADDRESSED action's census (Continue, Resume, held selection) of ``predecessor``.

    It first retires the positively ended local owners of this tree only
    (``local_custody_repair``: a retained witness or platform-qualified
    absence), then observes exactly as ``conflicting_writers`` — which stays
    the passive, write-free reader every census and GET uses (#1554).
    """
    from ouroboros.local_custody_repair import repair_ended_local_custody
    from ouroboros.owner_pause import tree_member_results

    root = pathlib.Path(drive_root or q.DRIVE_ROOT)
    try:
        members = set(tree_member_results(root, predecessor)) | {predecessor}
        repaired = repair_ended_local_custody(root, members, root_task_id=predecessor)
        if repaired:
            q.append_jsonl(pathlib.Path(q.DRIVE_ROOT) / "logs" / "events.jsonl",
                           {"type": "local_custody_retired", "root_task_id": predecessor, "retired": repaired})
    except Exception:
        log.warning("Ended local custody of %s was not repaired; it stays held", predecessor, exc_info=True)
    return conflicting_writers(q, predecessor, drive_root=drive_root, owner_pause_fence_id=owner_pause_fence_id)


def _successor_task(q: Any, predecessor: str, result: Dict[str, Any], binding: Dict[str, Any],
                    verdict: Dict[str, Any], sources: Dict[str, Any], admission: Dict[str, Any]) -> Dict[str, Any]:
    from ouroboros.owner_continue import owner_corpus_rows, work_order_text

    deadline_at = str(binding.get("deadline_at") or "")
    text = work_order_text(predecessor, verdict["cause"], sources, deadline_at=deadline_at)
    title = str(result.get("title") or result.get("suggested_name") or result.get("objective") or predecessor)[:80]
    original = sources.get("original") or {}
    origin_ref = original.get("origin_message_ref")
    from ouroboros.settings_scales import EFFORT_SCALE

    # The same work keeps the effort it was explicitly started on. A root's stored value is
    # only that explicit choice (dispatch stamps children alone; a switch_model is not
    # stored), read from the result row its admission wrote even if no worker ever ran.
    effort = str(result.get("reasoning_effort") or "")
    task: Dict[str, Any] = {
        **({"reasoning_effort": effort} if effort in EFFORT_SCALE else {}),
        "id": binding["successor_task_id"], "type": "task", "chat_id": binding.get("chat_id"),
        "project_id": str(binding.get("project_id") or ""), "text": text, "objective": text,
        "title": f"Continue: {title}", "suggested_name": f"Continue: {title}",
        "root_task_id": binding["successor_task_id"], "depth": 0, "delegation_role": "root",
        **({"deadline_at": deadline_at} if deadline_at else {}),
        **{key: binding[key] for key in ("workspace_root", "workspace_mode") if binding.get(key)},
        "metadata": {
            **({"project_id": binding["project_id"]} if binding.get("project_id") else {}),
            # Keep verified owner-door provenance on its canonical carrier;
            # the new objective below is still the host's continuation text.
            **({"origin_message_ref": dict(origin_ref), "origin_message_text": original["content"]}
               if isinstance(origin_ref, dict) and origin_ref else {}),
            # The objective is host-composed facts; the owner corpus is seeded
            # with the owner's exact words only (loop_messages seeding).
            "objective_author": {"kind": "continuation", "predecessor_task_id": predecessor},
            "owner_corpus": owner_corpus_rows(sources),
            "continuation": {
                "predecessor_task_id": predecessor, "cause": verdict["cause"],
                "owner_sources": {"original": sources.get("original"), "later": sources.get("later") or []},
                "peer_context": sources.get("peer_context") or [],
                "billing_group_id": binding.get("billing_group_id"),
                "billing_group_limit_usd": binding.get("billing_group_limit_usd"),
                "billing_group_limit_source": binding.get("billing_group_limit_source"),
                "billing_group_limit_revision": binding.get("billing_group_limit_revision"),
                "admission": admission,
            },
        },
    }
    try:
        from ouroboros.agent_startup_checks import valid_task_result_authority_source
        from ouroboros.server_routing_context import _task_result_ground_truth

        source = _task_result_ground_truth({**result, "task_id": predecessor}).get("authority_source")
        if valid_task_result_authority_source(source, predecessor):
            task["predecessor_task_id"] = predecessor
            task["predecessor_authority_source"] = dict(source)
    except Exception:
        log.debug("Predecessor authority pointer unavailable for %s", predecessor, exc_info=True)
    return task


def _billing_group(q: Any, predecessor: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """The whole-work group the successor spends from, and the cap it started under.

    A chain keeps the FIRST root's group and cap; otherwise the predecessor's
    own earliest live ledger row names the group it spent in and the cap it
    started under (an older block's own literal included, ``legacy_live``).
    With neither a durable initial binding nor a ledger cap, the predecessor
    stays its own group, its spend preserved, under the configured cap — the
    choice is pinned on the predecessor's result (``legacy_default``) so its
    own later work and every successor share one ceiling.
    """
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    carried = result.get("billing_group") or metadata.get("billing_group") or metadata.get("continuation") or {}
    if carried.get("billing_group_id") and "billing_group_limit_usd" in carried:
        return {key: carried.get(key) for key in (
            "billing_group_id", "billing_group_limit_usd", "billing_group_limit_source", "billing_group_limit_revision")}
    from ouroboros.usage_admission import ledger_billing_binding, original_group_limit

    group = str(result.get("root_task_id") or predecessor)
    recorded = ledger_billing_binding(q.DRIVE_ROOT, group)  # the root's own first row: its group, not only itself
    if recorded:
        return {key: recorded.get(key) for key in ("billing_group_id", "billing_group_limit_usd", "billing_group_limit_source")}
    found = original_group_limit(q.DRIVE_ROOT, group)
    if found["source"] != "no_attempt_recorded":
        return {"billing_group_id": group, "billing_group_limit_usd": found["limit_usd"],
                "billing_group_limit_source": found["source"]}
    from ouroboros.config import runtime_setting
    from ouroboros.task_results import stamp_task_result_schema, task_result_path
    from ouroboros.utils import update_json_locked

    limit = float(runtime_setting("OUROBOROS_PER_TASK_COST_USD", "0") or 0)
    binding = {"billing_group_id": group, "billing_group_limit_usd": limit if limit > 0 else None,
               "billing_group_limit_source": "legacy_default", "billing_group_limit_revision": utc_now_iso()}

    def pin(current):
        nonlocal binding
        if current.get("billing_group"):  # already chosen (by an earlier Continue or admission): share it
            binding = current["billing_group"]
            return None
        return stamp_task_result_schema({**current, "billing_group": binding})

    if result.get("status"):  # a recorded predecessor: the choice becomes its durable binding, or nothing is admitted
        try:
            update_json_locked(task_result_path(q.DRIVE_ROOT, group), pin, strict_existing_dict=True)
        except (OSError, ValueError, TimeoutError) as exc:
            # An unpinned choice would let the predecessor's own later work follow a changed
            # setting instead of the successor's ceiling: refuse now; the same nonce retries.
            log.warning("Could not pin the default billing group on %s", group, exc_info=True)
            raise ValueError("billing_authority_unavailable") from exc
    return {key: binding.get(key) for key in ("billing_group_id", "billing_group_limit_usd", "billing_group_limit_source")}


def _prepare_project_room(q: Any, predecessor: str, task: Dict[str, Any]) -> Dict[str, Any]:
    """The successor's prepared Project basis; bound work also binds its successor.

    A fresh admission carries the basis (the producer seam schedule_task uses); a
    basis-less row would be held as legacy forever. When the predecessor is bound
    to the claim's room, the successor joins it through the same writer a timeout
    retry uses, so late frames, helpers and history resolve the room by lineage,
    with the original owner message as its origin. The successor id belongs to
    this claim alone: the bind is idempotent, fenced by the basis and made before
    the row exists. Refusals raise; the queue lock is not held here.
    """
    from ouroboros.projects_registry import bind_task_to_project, project_admission_view, project_binding_for_task

    bound = project_binding_for_task(q.DRIVE_ROOT, predecessor, strict=True) or {}
    basis = project_admission_view(q.DRIVE_ROOT, task["project_id"], frozen=True,
                                   allow_unregistered=bound.get("project_id") != task["project_id"])
    if (bound.get("project_id"), bound.get("project_chat_id")) == (task["project_id"], task["chat_id"]):
        ref = bound.get("source_ref")
        origin = ({"ref": ref, **({"text": bound["source_text"]} if "source_text" in bound else {})}
                  if isinstance(ref, dict) else {"absent": bound.get("origin_absent") or "post_hoc_unresolved"})
        bind_task_to_project(q.DRIVE_ROOT, task["id"], task["project_id"], task["chat_id"],
                             origin=origin, admission_basis=basis)
    return basis


def admit_continuation(predecessor_task_id: str, *, action_nonce: str) -> Dict[str, Any]:
    """Admit (or replay) the owner's Continue of one interrupted root (module docstring)."""
    from ouroboros.owner_continue import (
        VERB, BINDING_VERSION, binding_sha, claim_on_predecessor, continuation_eligibility,
        continuation_room, owner_sources, successor_id, valid_nonce,
    )
    from ouroboros.task_results import STATUS_SCHEDULED, load_task_result, validate_task_id, write_task_result

    from supervisor import queue as q
    predecessor = validate_task_id(predecessor_task_id)
    nonce = valid_nonce(action_nonce)
    successor = successor_id(predecessor, nonce)
    replay = _replay(q, predecessor, nonce, successor)
    if replay is not None:
        return replay
    try:
        result = load_task_result(q.DRIVE_ROOT, predecessor, strict=True) or {}
    except Exception:
        return {"ok": False, "error": "predecessor_record_unreadable"}
    claim = result.get("continued_by") if isinstance(result.get("continued_by"), dict) else {}
    if claim and str(claim.get("action_nonce") or "") != nonce:
        return {"ok": False, "error": "already_continued",
                "successor_task_id": str(claim.get("successor_task_id") or ""),
                **({"state": "bound", "action_nonce": claim.get("action_nonce")}
                   if claim.get("state") == "bound" else {})}
    verdict = continuation_eligibility(result, predecessor) if not claim else {
        "eligible": True, "cause": str((claim.get("binding") or {}).get("cause") or ""), "refusal": ""}
    if not verdict["eligible"]:
        return {"ok": False, "error": verdict["refusal"], "cause": verdict["cause"]}
    sources = owner_sources(q.DRIVE_ROOT, result, predecessor)
    if sources["gaps"]:
        return {"ok": False, "error": "owner_source_missing", "gaps": list(sources["gaps"])}
    try:
        billing = {} if claim else _billing_group(q, predecessor, result)
    except Exception:
        return {"ok": False, "error": "billing_authority_unavailable"}
    try:
        # A recorded claim keeps the room it froze, even if the work was converted since.
        room = {} if claim else continuation_room(q.DRIVE_ROOT, predecessor, result)
    except (OSError, ValueError):
        return {"ok": False, "error": "project_routing_fence_lookup_failed"}
    binding = dict(claim.get("binding") or {}) if claim else {
        "verb": VERB, "binding_version": BINDING_VERSION, "predecessor_task_id": predecessor,
        "action_nonce": nonce, "successor_task_id": successor, "cause": verdict["cause"], **room,
        "workspace_root": str(result.get("workspace_root") or ""),
        "workspace_mode": str(result.get("workspace_mode") or ""),
        "deadline_at": str(result.get("deadline_at") or (result.get("metadata") or {}).get("deadline_at")
                           or (result.get("task_contract") or {}).get("deadline_at") or ""), **billing,
    }
    binding["admission_token"] = f"continue:{binding_sha({k: v for k, v in binding.items() if k != 'admission_token'})[:40]}"
    if not claim:
        try:
            claim, _created = claim_on_predecessor(q.DRIVE_ROOT, predecessor, nonce, binding)
        except ValueError as exc:
            text = str(exc)
            if text.startswith("already_continued:"):
                try:
                    other = (load_task_result(q.DRIVE_ROOT, predecessor, strict=True) or {}).get("continued_by") or {}
                    return {"ok": False, "error": "already_continued", "successor_task_id": other["successor_task_id"],
                            **({"state": "bound", "action_nonce": other["action_nonce"]}
                               if other.get("state") == "bound" else {})}
                except Exception:
                    return {"ok": False, "error": "continuation_unconfirmed"}
            return {"ok": False, "error": text or "continuation_refused"}
        except Exception as exc:
            return {"ok": False, "error": "continuation_claim_unwritable", "detail": str(exc)[:200]}
        binding = dict(claim["binding"])
    blockers = action_writers(q, predecessor)
    admission = {"binding": binding, "binding_sha256": binding_sha(binding), "admitted_at": utc_now_iso(),
                 "held": bool(blockers)}
    task = _successor_task(q, predecessor, result, binding, verdict, sources, admission)
    if blockers:
        from supervisor.events_budget import hold_budget_row

        hold_budget_row(task, reason=HOLD_CONTINUATION_WRITER,
                        detail="the interrupted task's own work is not proven settled; this Continue waits",
                        extra={"predecessor_task_id": predecessor, "blockers": blockers[:20],
                               "root_task_id": successor})
    project_basis = None
    if task.get("project_id"):
        try:
            project_basis = _prepare_project_room(q, predecessor, task)
        except (OSError, ValueError, RuntimeError) as exc:
            # The claim stays: the same nonce retries once the authority is readable.
            return {"ok": False, "error": getattr(exc, "reason", "project_routing_fence_lookup_failed"),
                    "successor_task_id": successor}
    token = binding["admission_token"]
    reservation = q.reserve_task_admission(successor, token)
    if reservation.get("status") == "existing_same_token":
        return _replay(q, predecessor, nonce, successor) or {"ok": False, "error": "continuation_unconfirmed"}
    if reservation.get("status") not in {"reserved", "already_reserved"}:
        return {"ok": False, "error": str(reservation.get("reason") or "admission_blocked"),
                "successor_task_id": successor, "unconfirmed": True}
    task["_admission_token"] = token
    task["_continuation_prepared"] = admission["binding_sha256"]
    with q.prepared_root_billing(task), q._queue_lock:  # the ledger read happens before the lock
        admitted = q.enqueue_task(task, project_admission=project_basis)
        if isinstance(admitted, dict) and admitted.get("_admission_blocked"):
            q.release_task_admission(successor, token)
            return {"ok": False, "error": str(admitted["_admission_blocked"]), "successor_task_id": successor}
        if q.persist_queue_snapshot(reason="owner_continue_admitted") is not True:
            q.PENDING[:] = [row for row in q.PENDING if str(row.get("id") or "") != successor]
            q.persist_queue_snapshot(reason="owner_continue_rollback")
            q.release_task_admission(successor, token)
            # The claim stays: the same nonce retries this admission, a new one is refused.
            return {"ok": False, "error": "queue_snapshot_persist_failed", "successor_task_id": successor}
        try:
            write_task_result(
                q.DRIVE_ROOT, successor, STATUS_SCHEDULED, continuation_admission=admission,
            chat_id=task.get("chat_id"), project_id=task.get("project_id") or "", title=task["title"],
            suggested_name=task["suggested_name"], root_task_id=successor, metadata=task["metadata"],
            # The binding a successor that never starts must still hand on to its own Continue.
            **{key: task[key] for key in ("deadline_at", "reasoning_effort", "workspace_root", "workspace_mode")
               if task.get(key)},
            **({"reason_code": HOLD_CONTINUATION_WRITER,
                "resource_limit": {"status": "budget_hold", "auto_resume": False, "exact_continuation": False,
                                   **task["_budget_pause_hold"]}} if blockers else {}),
            result=("Continue accepted; held until the interrupted task's own work is settled."
                    if blockers else "Continue accepted and durably scheduled."))
        except Exception as exc:
            return {"ok": False, "error": "continuation_publication_unconfirmed",
                    "successor_task_id": successor, "unconfirmed": True, "detail": str(exc)[:200]}
        prepared = admitted.pop("_continuation_prepared")
        if q.persist_queue_snapshot(reason="owner_continue_committed") is not True:
            admitted["_continuation_prepared"] = prepared
            return {"ok": False, "error": "queue_snapshot_persist_failed",
                    "successor_task_id": successor, "unconfirmed": True}
        q.release_task_admission(successor, token)
    try:
        from ouroboros.owner_continue import mark_claim_admitted

        mark_claim_admitted(q.DRIVE_ROOT, predecessor, nonce)
    except Exception:
        # Only the predecessor's mailbox release waits on this mark; a replay rewrites it.
        log.warning("Continue of %s admitted but its claim mark was not written", predecessor, exc_info=True)
    append_jsonl(q.DRIVE_ROOT / "logs" / "events.jsonl",
                 {"ts": utc_now_iso(), "type": "owner_continue_admitted", "task_id": successor,
                  "predecessor_task_id": predecessor, "cause": verdict["cause"], "held": bool(blockers),
                  "blockers": blockers[:20], "owner_visible": True,
                  "toast_once": f"{successor}:continue-admitted"})
    return {"ok": True, "replay": False, "task_id": successor, "successor_task_id": successor,
            "predecessor_task_id": predecessor, "status": STATUS_SCHEDULED, "held": bool(blockers),
            **({"blockers": blockers[:20]} if blockers else {})}


def release_settled_continuations(tree_root: str) -> List[str]:
    """Release Continues whose predecessor ``tree_root`` has no unsettled writer left.

    The reconciliation half of the writer hold, called when a member of that
    tree reaches its terminal. Only a row still under ITS OWN continuation hold
    is released (a newer Restart/Pause/Stop hold is never overridden), and the
    release is the ordinary selection, which re-checks the writers itself.
    """
    from supervisor.events_budget import budget_hold_fact, observe_held_budget_selection, select_held_budget_row

    from supervisor import queue as q
    released: List[str] = []
    with q._queue_lock:
        task_ids = [str(task.get("id") or "") for task in q.PENDING if isinstance(task, dict)
                and (budget_hold_fact(task) or {}).get("reason") == HOLD_CONTINUATION_WRITER
                and ((task.get("metadata") or {}).get("continuation") or {}).get("predecessor_task_id") == tree_root]
    for task_id in task_ids:
        observation = observe_held_budget_selection(q, task_id)
        with q._queue_lock:
            task = next((row for row in q.PENDING if str(row.get("id") or "") == task_id), None)
            hold = budget_hold_fact(task)
            if task is None or (hold or {}).get("reason") != HOLD_CONTINUATION_WRITER:
                continue
            outcome = select_held_budget_row(q, task, hold, selected_by="reconciliation", observation=observation)
            if outcome.get("ok"):
                released.append(task_id)
    return released
