"""Unknown silence: dated facts and one related mind, never process custody.

The timeout pass owns observations under Q. The existing admission lane persists
and delivers them outside Q; the task document and ordinary queue own recovery.
An episode is answered only by the source's own progress, not by its diagnostic.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from functools import partial
from types import SimpleNamespace

from ouroboros.config import BODY_MAINTENANCE_INTERVAL_SEC
from ouroboros.work_diagnostics import diagnostic_ref

log = logging.getLogger(__name__)
_publishing: set[tuple[str, str]] = set()


def observe(queue, task_id: str, meta: dict, now: float, threshold: float) -> None:
    """Q caller. A clock can request attention, never finalization."""
    progress = float(meta.get("last_progress_at") or meta["started_at"])
    attempt = int(meta.get("attempt") or (meta.get("task") or {}).get("_attempt") or 1)
    identity = hashlib.sha256(json.dumps([task_id, attempt, meta["started_at"], progress]).encode()).hexdigest()[:24]
    fact = meta.get("silence_attention") or {}
    if fact.get("episode_id") != identity:
        fact = {"episode_id": identity, "task_id": task_id, "task_attempt": attempt,
                "last_progress_at": progress, "observed_at": now, "threshold_sec": threshold,
                "last_heartbeat_at": meta.get("last_heartbeat_at"), "state": "pending_attention"}
        meta["silence_attention"] = fact
    _submit(queue, task_id, meta)


def _submit(queue, task_id, meta):
    from supervisor.workspace_preparation import ADMISSION_LANE, current_generation

    fact = meta["silence_attention"]
    now = time.time()
    parent = _waiting_parent(queue, meta["task"], now)
    responsible = _parent_responsible(queue, fact, parent, now)
    new_parent = bool(parent and not fact.get("parent_delivered")
                      and fact.get("recipient_task_id") in (None, "", parent[0]))
    if meta.get("_silence_published") == fact and fact.get("self_delivered"):
        if responsible or (not new_parent and fact.get("attention_wait") in {
                "diagnostic_queued", "diagnostic_recorded", "diagnostic_not_recursed"}):
            return
    # Failed/held publication retries on the existing maintenance cadence. Source
    # progress, a new waiter or loss of the responsible parent is checked in memory.
    retry = meta.get("_silence_retry")
    if retry and retry[:2] == (fact, responsible) and not new_parent and now < retry[2]:
        return
    identity = fact["episode_id"]
    key = (str(queue.DRIVE_ROOT), identity)
    generation = current_generation()
    if key in _publishing or generation is None or generation.launches_closed():
        return
    _publishing.add(key)
    meta["_silence_retry"] = (copy.deepcopy(fact), responsible, now + BODY_MAINTENANCE_INTERVAL_SEC)
    if not generation.submit(partial(_publish, queue, task_id, meta, key), ADMISSION_LANE):
        _publishing.discard(key)
        meta.pop("_silence_retry", None)


def recovered(queue, task_id: str, meta: dict, now: float) -> None:
    """Only real source progress ends an episode; borrowed activity does not."""
    fact = meta.get("silence_attention") or {}
    if fact and fact.get("state") != "progress_observed" and float(meta.get("last_progress_at") or 0) > fact["last_progress_at"]:
        fact.update(state="progress_observed", recovered_at=now)
    if fact and fact.get("state") == "progress_observed" and meta.get("_silence_published") != fact:
        _submit(queue, task_id, meta)


def _waiting_parent(queue, task: dict, now: float) -> tuple[str, dict, str] | None:
    parent = str(task.get("parent_task_id") or "")
    seen = set()
    responsibility = {str(task.get("id") or "")}
    while parent and parent not in seen:
        seen.add(parent)
        meta = _parent_meta(queue, parent)
        if not meta:
            return None
        selected = next((tid for row in (meta.get("active_operation_leases") or {}).values()
                      if isinstance(row, dict) and row.get("kind") == "in_slot_wait"
                      and float(row.get("until_ts") or 0) > now
                      for tid in row.get("wait_for_tasks") or () if tid in responsibility), "")
        selected = next((tid for tid in _parent_sleep(meta).get("tasks") or () if tid in responsibility), selected)
        if selected and _available_parent(meta):
            return parent, meta, selected
        responsibility.add(parent)
        parent = str((meta.get("task") or {}).get("parent_task_id") or "")
    return None


def _parent_meta(queue, task_id):
    live = queue.RUNNING.get(task_id)
    if isinstance(live, dict):
        return live
    parked = next((row for row in queue.PENDING if row.get("id") == task_id), None)
    return {"task": parked, "attempt": parked.get("_attempt", 1)} if parked else {}


def _parent_sleep(meta):
    """Only this attempt's model-selected sleep, never an owner-answer park."""
    from supervisor.worker_owner_wait import waiting_on_owner

    if waiting_on_owner(meta, int(meta.get("attempt") or 1)):
        pending = meta.get("owner_wait_park_pending") or {}
        wait = ((pending.get("event") or {}).get("checkpoint") if pending else meta.get("owner_wait")) or {}
    else:
        wait = (meta.get("task") or {}).get("_budget_pause") or {}
    chosen = wait.get("sleep")
    return chosen if wait.get("reason") == "sleep" and isinstance(chosen, dict) else {}


def _available_parent(meta):
    from supervisor.worker_owner_wait import waiting_on_owner
    from ouroboros.work_stop import current_stop

    return bool(meta and not diagnostic_ref(meta.get("task") or {})
                and (not waiting_on_owner(meta, int(meta.get("attempt") or 1)) or _parent_sleep(meta))
                and not current_stop(meta) and not current_stop(meta.get("task")))


def _parent_responsible(queue, fact, parent, now):
    if not fact.get("parent_delivered"):
        return False
    recipient = _parent_meta(queue, fact.get("recipient_task_id"))
    return bool(_available_parent(recipient) and (
        recipient.get("active_llm_call")
        or float(recipient.get("last_progress_at") or 0) > now - fact["threshold_sec"]
        or parent and parent[0] == fact.get("recipient_task_id")))


def _save(queue, task_id, fact):
    from ouroboros.task_store import patch_document

    def update(current):
        if not current or current.get("silence_attention") == fact:
            return None
        return {**current, "silence_attention": copy.deepcopy(fact)}

    patch_document(queue.DRIVE_ROOT, task_id, update)


def _publish(queue, task_id, meta, key) -> None:
    from ouroboros.owner_mailbox import write_task_message
    from ouroboros.store import StoreError
    from ouroboros.task_results import load_task_result
    from ouroboros.usage_ledger import BillingAuthorityUnavailable

    try:
        with queue._queue_lock:
            if queue.RUNNING.get(task_id) is not meta or meta["silence_attention"]["episode_id"] != key[1]:
                return
            fact = copy.deepcopy(meta["silence_attention"])
            task = copy.deepcopy(meta["task"])
        stored = (load_task_result(queue.DRIVE_ROOT, task_id, strict=True) or {}).get("silence_attention") or {}
        if stored.get("episode_id") == fact["episode_id"]:
            fact = {**stored, **fact, **{k: stored[k] for k in
                    ("self_delivered", "recipient_task_id", "parent_delivered", "diagnostic_id") if k in stored}}
        if fact != stored:
            _save(queue, task_id, fact)  # the observation precedes every delivery/admission
        saved = copy.deepcopy(fact)
        text = ("Silence observation (not a stop request): " + json.dumps(fact, sort_keys=True) +
                ". No recognized progress or active wait was observed. Heartbeat is not proof of progress; "
                "absence of progress is not proof of a dead worker. Inspect the current work and choose "
                "waiting or an existing lawful control. No kill, retry or successor was requested.")
        if not fact.get("self_delivered"):
            fact["self_delivered"] = write_task_message(queue._task_drive_for_task(task, task_id), text,
                task_id, source_task_id="", provenance="system", msg_id="silence:" + fact["episode_id"])
        with queue._queue_lock:
            current = meta.get("silence_attention") or {}
            if queue.RUNNING.get(task_id) is not meta or current.get("episode_id") != fact["episode_id"]:
                return
            if current.get("state") == "progress_observed":
                fact.update(state="progress_observed", recovered_at=current["recovered_at"])
            parent = _waiting_parent(queue, task, time.time())
            responsible = _parent_responsible(queue, fact, parent, time.time())
        if fact["state"] != "progress_observed" and not diagnostic_ref(task):
            if parent and not fact.get("recipient_task_id"):
                fact["recipient_task_id"] = parent[0]
                _save(queue, task_id, fact)  # fix the recipient before a possibly ambiguous delivery
            if parent and parent[0] == fact.get("recipient_task_id") and not fact.get("parent_delivered"):
                parent_id, parent_meta, waited_task = parent
                if write_task_message(queue._task_drive_for_task(parent_meta["task"], parent_id), text,
                        parent_id, source_task_id=task_id, provenance="system",
                        work_attention={"kind": "silence", "task_id": task_id,
                                        "waiting_for_task_id": waited_task, "episode_id": fact["episode_id"]},
                        msg_id="silence-parent:" + fact["episode_id"]):
                    fact["parent_delivered"] = True
                    responsible = True
            if not responsible:
                _admit(queue, task_id, task, meta, fact)
        elif diagnostic_ref(task):
            fact["attention_wait"] = "diagnostic_not_recursed"
        if fact != saved:
            _save(queue, task_id, fact)
        if fact["state"] == "progress_observed":
            _withdraw_unsent(queue, task_id, fact)
        published = copy.deepcopy(fact)
        with queue._queue_lock:
            if queue.RUNNING.get(task_id) is meta and (meta.get("silence_attention") or {}).get("episode_id") == fact["episode_id"]:
                # Progress that raced persistence wins in memory and is published next pass.
                fact.update({k: v for k, v in meta["silence_attention"].items()
                             if k in {"state", "recovered_at"}})
                meta["silence_attention"] = fact
                meta["_silence_published"] = published
                meta["_silence_retry"] = (published, responsible, time.time() + BODY_MAINTENANCE_INTERVAL_SEC)
    except (OSError, ValueError, StoreError, BillingAuthorityUnavailable) as exc:
        log.warning("Silence attention remains pending for %s: %s", task_id, exc)
    finally:
        with queue._queue_lock:
            _publishing.discard(key)


def _withdraw_unsent(queue, task_id, fact):
    """Withdraw only a proven-unsubmitted diagnostic through cancellation's owner.

    No live process is touched. The ordinary queue cancellation pass settles the
    pending row; a claimed or possibly dispatched diagnostic retains custody.
    """
    from ouroboros.cancel_intents import cancel_pending, request_cancel

    with queue._queue_lock:
        for task in queue.PENDING:
            ref = diagnostic_ref(task)
            if (ref.get("task_id") == task_id and ref.get("episode_id") == fact["episode_id"]
                    and task.get("admitted_dispatch") == "none" and not task.get("_launch_claim")
                    and not cancel_pending(queue.DRIVE_ROOT, task["id"], strict=True)):
                request_cancel(queue.DRIVE_ROOT, task["id"], reason="Source progress observed before diagnostic dispatch",
                               source="silence_resolved", requested_by="supervisor", scope="single")


def _source_label(task) -> str:
    """The source's human name for the diagnostic's card; its id stays in the details."""
    for key in ("title", "suggested_name", "description", "objective"):
        if name := " ".join(str(task.get(key) or "").split()):
            return name[:80]
    first = next((line.strip() for line in str(task.get("text") or "").splitlines() if line.strip()), "")
    return first[:80] or "untitled work"


def _deadline(queue, task, meta):
    """Bound this one turn by the accepted work's remaining clocks, including queue wait."""
    from datetime import datetime, timezone
    from ouroboros.deadline_utils import parse_deadline_ts
    from ouroboros.model_wait import execution_elapsed_seconds
    from ouroboros.runtime_limits import resolve_task_execution_timeout_sec

    bounds = []
    with queue._queue_lock:
        clocks = [meta]
        root = queue.RUNNING.get(task.get("root_task_id"))
        if root is not None and root is not meta:
            clocks.append(root)
        for clock in clocks:
            source = clock["task"]
            for carrier in (source, source.get("metadata") or {}, source.get("task_contract") or {}):
                deadline = parse_deadline_ts(carrier.get("deadline_at"))
                if deadline is not None:
                    bounds.append(deadline.timestamp())
            ceiling = resolve_task_execution_timeout_sec(source)
            if ceiling is not None:
                bounds.append(time.time() + max(0, ceiling - execution_elapsed_seconds(clock, time.time())))
    return datetime.fromtimestamp(min(bounds), timezone.utc).isoformat() if bounds else ""


def _admit(queue, task_id, task, meta, fact) -> None:
    from ouroboros.contracts.task_contract import attach_task_contract, build_task_contract
    from ouroboros.dialogue_provenance import presence_binding_authority_metadata
    from ouroboros.model_wait import execution_elapsed_seconds
    from ouroboros.runtime_limits import resolve_task_execution_timeout_sec
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.tools.control_delegation import (
        admitted_depth_cap,
        check_delegation_admission,
        child_budget_for_schedule,
        stamp_depth_provenance,
    )
    from ouroboros.store import StoreError
    from ouroboros.tools.control_scheduling import _build_child_subagent_contract
    from ouroboros.usage_admission import task_billing_fields
    from ouroboros.work_diagnostics import _current_source_reason, control_reason
    from supervisor import events_schedule_task as delegation

    diagnostic_id = "silence-" + fact["episode_id"]
    fact["diagnostic_id"] = diagnostic_id
    recorded = load_task_result(queue.DRIVE_ROOT, diagnostic_id, strict=True)
    if recorded and (recorded.get("status") != "requested" or recorded.get("workspace_preparation")):
        fact["diagnostic_status"] = recorded.get("status")
        if recorded.get("reason_code"):
            fact["diagnostic_reason"] = recorded["reason_code"]
        fact["attention_wait"] = "diagnostic_recorded"
        return
    root_id = str(task.get("root_task_id") or task_id)
    billing = task_billing_fields({"id": task_id}, root_id, None, queue.DRIVE_ROOT)
    if not billing.get("billing_group_limit_source"):
        fact["attention_wait"] = "billing_authority_unavailable"
        return
    reference = {"kind": "silence", "task_id": task_id, "root_task_id": root_id,
                 "task_attempt": fact["task_attempt"], "episode_id": fact["episode_id"]}
    contract = build_task_contract(task)
    # The schedule owner prefers the captured live contract, then its durable
    # parent row. Read that fallback off Q; keep this admission's rights frozen.
    # An unreadable row is not the omitted legacy budget: nothing is frozen, and
    # this unsent episode rereads it on the maintenance cadence.
    try:
        parent_budget = delegation._events()._parent_delegation_budget(
            SimpleNamespace(RUNNING={task_id: {"task": task}}, DRIVE_ROOT=queue.DRIVE_ROOT),
            task_id, task.get("budget_drive_root") or queue.DRIVE_ROOT, strict=True)
    except (OSError, ValueError, StoreError):
        fact["attention_wait"] = "delegation_authority_unavailable"
        return
    if parent_budget:
        contract["delegation_budget"] = parent_budget
    max_depth = admitted_depth_cap(contract, delegation._events().get_max_subagent_depth())
    metadata = {"work_diagnostic": reference, "billing_group": {k: v for k, v in billing.items() if k.startswith("billing_group_")},
                **presence_binding_authority_metadata(task.get("metadata"), task_contract=contract),
                **{k: v for k, v in (task.get("metadata") or {}).items()
                   if k in {"work_kind", "initiator", "consciousness_autonomy", "runtime_mode_cap"}}}
    origin = task.get("metadata") or {}
    if task.get("type") == "evolution" or origin.get("evolution_root_task_id") or origin.get("evolution_transaction"):
        metadata["evolution_root_task_id"] = origin.get("evolution_root_task_id") or root_id
    label = _source_label(task)
    diagnostic = {"id": diagnostic_id, "type": "task", "parent_task_id": task_id,
        "root_task_id": root_id, "delegation_role": "subagent", "depth": int(task.get("depth") or 0) + 1,
        "chat_id": task.get("chat_id"), "project_id": task.get("project_id"),
        "budget_drive_root": str(queue.DRIVE_ROOT), "metadata": metadata,
        "root_cost_ceiling_usd": task.get("root_cost_ceiling_usd") or (task.get("metadata") or {}).get("root_cost_ceiling_usd"),
        "task_constraint": {"mode": "local_readonly_subagent"},
        "description": f"Silence check: {label}",
        "text": (f"One task-related diagnostic of work {task_id} ({label}). Observation: {json.dumps(fact)}. "
                 "Inspect its current facts and available evidence. Decide whether to wait or use the existing "
                 "lawful task controls. You are not a product writer. Do not repeat the work or start a successor "
                 "just because it was silent. A terminal result does not prove its writers exited. "
                 "Report your evidence and decision to the original work using forward_to_worker; finish this diagnostic "
                 "when the decision is made. No recovery is guaranteed by this observation."),
        "deadline_at": _deadline(queue, task, meta),
        "execution_timeout_sec": resolve_task_execution_timeout_sec(task),
        "_execution_consumed_sec": execution_elapsed_seconds(meta, time.time())}
    # The ordinary child builder retains every inherited restriction, including
    # supported Presence ceilings and declared sources. The diagnostic profile
    # intersects that authority at schema/dispatch; it never replaces it.
    workspace = contract.get("workspace") or {}
    diagnostic.update(workspace_root=workspace.get("root", ""), workspace_mode=workspace.get("mode", ""))
    diagnostic["task_contract"] = _build_child_subagent_contract({
        "tid": diagnostic_id, "parent_contract": contract, "parent_task_id": task_id, "root_task_id": root_id,
        "objective": diagnostic["text"], "constraints": contract.get("constraints", ""),
        "context": contract.get("context", ""), "deadline_at": diagnostic["deadline_at"],
        "workspace_root": diagnostic["workspace_root"], "workspace_mode": diagnostic["workspace_mode"],
        "attachment_manifest": contract.get("attachment_manifest"),
        "attachment_manifest_ref": contract.get("attachment_manifest_ref"),
        "child_delegation_budget": {**child_budget_for_schedule(
            contract, current_depth=diagnostic["depth"] - 1, new_depth=diagnostic["depth"], max_depth=max_depth,
            may_mutate=False, may_fan_out=False, max_children=0, intent_note=""), "may_delegate": False}})
    diagnostic["task_contract"], depth_provenance = stamp_depth_provenance(
        diagnostic["task_contract"], attempted_depth=diagnostic["depth"], max_depth=max_depth)
    count_roots = [task.get("budget_drive_root") or queue.DRIVE_ROOT]
    if str(queue.DRIVE_ROOT) != str(count_roots[0]):
        count_roots.append(queue.DRIVE_ROOT)

    room_mode, room_refusal = delegation._subagent_room_mode(queue, task)

    def delegation_refusal(count, *, early=False):
        # Frozen source rights; the schedule owner's current registered limits.
        if room_refusal:
            return room_refusal
        rights = check_delegation_admission(contract["delegation_budget"], direct_child_count=count, room_mode=room_mode)
        if not rights.ok and not (early and rights.reason_code == "delegation_rights_child_count_unknown"):
            return rights.reason_code, rights.detail
        cap = admitted_depth_cap(contract, delegation._events().get_max_subagent_depth())
        if diagnostic["depth"] > cap:
            return "subtask_depth_limit", f"Subtask depth limit ({cap}) exceeded by attempted depth {diagnostic['depth']}."
        _blocked, detail = delegation._active_cap_state(
            root_id, task_id, queue.PENDING, queue.RUNNING, delegation.get_max_active_subagents_per_root())
        return ("active_subagent_hard_cap" if detail else ""), detail

    def current_source(source_id):
        with queue._queue_lock:
            return copy.deepcopy(queue.RUNNING.get(source_id))

    reason = control_reason(diagnostic, queue.DRIVE_ROOT, before_handoff=True, current_source=current_source)
    if reason:
        fact["attention_wait"] = reason
        return
    def recheck(count, *, early=False):
        if queue.RUNNING.get(task_id) is not meta or _current_source_reason(reference, fact, meta):
            return "source_progressed", "The source no longer needs this diagnostic."
        rows = [*queue.PENDING, *(m.get("task") or {} for m in queue.RUNNING.values())]
        if any(diagnostic_ref(row).get("task_id") == task_id for row in rows):
            return "diagnostic_already_owned", "One diagnostic already belongs to this source."
        if queue.RETURNS_NAMED:
            return "admission_closed", "The returning stop closed admission."
        if root_id in queue.BUDGET_ROOT_FENCES:
            return "root_budget_fence", "The accepted work's budget or Pause fence holds admission."
        if (queue.ACCEPTANCE_FENCES.get(root_id) or {}).get("status") in {"active", "sealed"}:
            return "task_acceptance_fence", "The accepted work's completion fence holds admission."
        return delegation_refusal(count, early=early)

    from supervisor.task_admission import enqueue_subagent_with_scheduled_result, scheduled_admission_rejection
    from supervisor.workspace_preparation import accept

    attach_task_contract(diagnostic)
    direct_child_count = delegation._direct_child_count(count_roots, task_id, diagnostic_id)
    with queue._queue_lock:
        refusal, detail = delegation_refusal(direct_child_count, early=True)
    fields = {key: diagnostic[key] for key in (
        "metadata", "parent_task_id", "root_task_id", "delegation_role", "task_constraint",
        "root_cost_ceiling_usd", "task_contract", "chat_id", "description")}
    if refusal:
        fact["attention_wait"] = refusal
        delegation._reject_schedule_task(
            queue, tid=diagnostic_id, chat_id=None, delegation_role="subagent", parent_id=task_id,
            root_task_id=root_id, role="", result_fields=fields, detail=detail, reason_code=refusal,
            extra_fields={"admission_outcome": "never_admitted"})
        return
    with queue._queue_lock:
        refusal, _detail = recheck(direct_child_count, early=True)
    if refusal:
        fact["attention_wait"] = refusal
        return
    write_task_result(queue.DRIVE_ROOT, diagnostic_id, "requested", create_only=True, **fields)

    def durable_child_count():
        # Read the room again outside Q before the interlocked admission.
        nonlocal room_mode, room_refusal
        room_mode, room_refusal = delegation._subagent_room_mode(queue, task)
        if room_refusal:
            return direct_child_count
        return delegation._direct_child_count(count_roots, task_id, diagnostic_id, propagate_contention=True)

    def admit(preparation):
        admitted, refusal, detail, reject = enqueue_subagent_with_scheduled_result(
            queue, diagnostic, result_fields=fields, preparation=preparation,
            admitted_task_contract=diagnostic["task_contract"], admitted_depth_provenance=depth_provenance,
            direct_child_count=direct_child_count, pending_ref=queue.PENDING, recheck_child_admission=recheck,
            durable_child_count=durable_child_count,
            transition_id=diagnostic_id)
        fact["attention_wait"] = refusal or admitted.get("_admission_blocked") or "diagnostic_queued"
        if admitted.get("_admission_blocked"):
            preparation.refuse(**scheduled_admission_rejection(admitted, project_id=task.get("project_id"),
                                                               root_task_id=root_id))
        elif refusal == "scheduled_event_replay":
            preparation.abandon()
        elif refusal:
            preparation.refuse(detail, refusal, persist_result=reject)

    fact["attention_wait"] = "diagnostic_preparing"
    accept({"task_id": diagnostic_id, "task_constraint": diagnostic["task_constraint"]}, queue,
           {"chat_id": task.get("chat_id"), "parent_id": task_id, "root_task_id": root_id,
            "delegation_role": "subagent"}, admission=admit)
