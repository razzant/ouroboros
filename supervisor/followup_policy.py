"""Relationship admission over the existing schedule/result/control owners.

Provenance is host-authored; relationship is an explicit model decision. A row
written before relationships were recorded (no ``followup_relation`` at all)
keeps its published launch and own-root expense rules, relationship still
``unknown``: only an actual Stop or owner Pause of its origin
holds it, and no shared wallet is inferred. A published owner-door row (no
source, origin or relationship) is independent. A recorded but unreadable
relationship never dispatches or acquires a wallet. A Stop's identity survives in
the original result; Restart's episode is history in the schedule document.
Neither enabled nor a timestamp releases a control. The lifecycle writer alone
records an audited release of the exact observed hold.

Final starts use queue -> schedule -> cancellation -> origin launch locks. Pause
waits outside queue, then rechecks launch authority without waiting under queue;
cancellation never acquires schedule. No new occurrence or receipt protocol lives here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import uuid
import threading
from contextlib import ExitStack, contextmanager

STOP_SOURCES = frozenset({"http_single", "http_cascade", "http_graceful",
                          "owner", "owner_stop", "agent_tool", "owner_restart"})
HOST_FIELDS = ("followup_origin", "followup_relation", "followup_hold",
               "followup_released", "followup_wait")
_CONTROL_DEPTH = threading.local()


class StopActionConflict(RuntimeError):
    """A known action identity was reused with different control arguments."""


def new_stop_fields(source, *, previous=None, action_id="", action_binding=()):
    """Keep exact retries on the current control, even after a later Stop.

    Receipts travel in the existing cancel/result carrier, never a second store.
    Legacy calls without identity are distinct accepted actions, not deduplicated
    by arguments or time. A reused identity cannot change the action's meaning.
    """
    from ouroboros.utils import utc_now_iso
    if source not in STOP_SOURCES:
        return {}, False
    prior = previous or {}
    receipts = dict(prior.get("action_receipts") or {})
    if action_id:
        key = hashlib.sha256(action_id.encode()).hexdigest()
        binding = hashlib.sha256(json.dumps(action_binding).encode()).hexdigest()
        if key in receipts:
            if receipts[key] != binding:
                raise StopActionConflict("stop_action_id reused with a different action")
            return {"followup_stop": copy.deepcopy(prior)}, True
        receipts[key] = binding
    return {"followup_stop": {"control_id": uuid.uuid4().hex, "source": str(source),
            "requested_at": utc_now_iso(), **({"action_receipts": receipts} if receipts else {})}}, False


def legacy_unrecorded(record):
    """No relationship decision was ever recorded: the published row shape."""
    return "followup_relation" not in record


def relation_kind(record):
    relation = record.get("followup_relation") or {}
    if isinstance(relation, dict) and relation.get("kind") in {"related", "independent"}:
        return relation["kind"]
    # Only the established skill producer or an explicit owner creation has
    # independent semantics. An arbitrary historical source is not proof. The
    # published owner door stamped no source, origin or relationship: that
    # structural shape is an owner schedule, never a task's continuation.
    if record.get("source") == "owner" or (record.get("source") == "skill_manifest" and record.get("skill")):
        return "independent"
    if not record.get("source") and not record.get("followup_origin") and legacy_unrecorded(record):
        return "independent"
    return "unknown"


def origin_of(record):
    origin = record.get("followup_origin")
    if isinstance(origin, dict) and origin.get("task_id"):
        return {key: str(origin.get(key) or "") for key in ("task_id", "root_task_id")}
    if record.get("source") == "task_followup":
        meta = (record.get("task") or {}).get("metadata") or {}
        task_id = str(meta.get("origin_task_id") or "")
        return {"task_id": task_id, "root_task_id": str(meta.get("origin_root_task_id") or task_id)}
    return {}


def resolve_relation(root, origin, kind, *, declared_by):
    """Resolve money off the schedule lock from actual task/ledger authority."""
    from ouroboros.deadline_utils import parse_deadline_ts
    from ouroboros.task_results import load_task_result
    from ouroboros.usage_admission import task_billing_fields
    from ouroboros.usage_ledger import BillingAuthorityUnavailable

    if kind not in {"related", "independent"}:
        raise ValueError("followup_relation_required")
    relation = {"kind": kind, "declared_by": declared_by, "revision": uuid.uuid4().hex}
    if kind == "independent":
        return relation
    task_id, root_id = str(origin.get("task_id") or ""), str(origin.get("root_task_id") or "")
    if not task_id or not root_id:
        raise ValueError("followup_origin_unavailable")
    rows = [load_task_result(pathlib.Path(root), tid, strict=True) for tid in dict.fromkeys((task_id, root_id))]
    if any(not row for row in rows):
        raise ValueError("followup_origin_unavailable")
    actual_root = str(rows[0].get("root_task_id") or (rows[0].get("metadata") or {}).get("root_task_id") or task_id)
    if actual_root != root_id:
        raise ValueError("followup_origin_unavailable")
    try:
        billing = task_billing_fields({"id": task_id}, root_id, None, root)
    except BillingAuthorityUnavailable as exc:
        raise ValueError("followup_billing_authority_unavailable") from exc
    if not billing.get("billing_group_limit_source") or not billing.get("billing_group_limit_revision"):
        raise ValueError("followup_billing_authority_unavailable")
    relation["billing_group"] = {k: v for k, v in billing.items() if k.startswith("billing_group_")}
    deadlines = []
    for row in rows:
        meta = row.get("metadata") or {}
        for source in (row, meta, row.get("task_contract") or {}, meta.get("task_contract") or {}):
            raw = source.get("deadline_at")
            if raw:
                parsed = parse_deadline_ts(str(raw))
                if parsed is None:
                    raise ValueError("followup_deadline_unavailable")
                deadlines.append(parsed)
    relation["deadline_at"] = min(deadlines).isoformat() if deadlines else ""
    return relation


def policy_view(root, data, record):
    """Pure current projection; callers that authorize starts hold control locks.

    A read that met confirmed store contention also says ``busy``: the same
    unavailable wait, which ``refresh_policy`` neither persists nor audits."""
    from ouroboros.cancel_intents import active_intent, _validated_single_cancel_target
    from ouroboros.owner_pause import read_fence, fence_closed
    from ouroboros.task_results import load_task_result, _TRULY_TERMINAL_STATUSES
    from ouroboros.deadline_utils import parse_deadline_ts
    from supervisor.schedule_occurrence import contention
    import datetime

    kind = relation_kind(record)
    if kind == "independent":
        return {"relation": kind, "controls": {}, "wait": ""}
    controls, wait, busy = {}, "", False
    released = record.get("followup_released") or {}
    try:
        from ouroboros.delegate_recovery import owner_restart_in_progress, pending_owner_restart
        if owner_restart_in_progress(root) or pending_owner_restart(root):
            wait = "owner_restart_in_progress"
        origin = origin_of(record)
        origin_ids = set(origin.values()) - {""}
        control_ids = origin_ids | {_validated_single_cancel_target(root, tid) for tid in origin_ids}
        for tid in sorted(control_ids):
            row = load_task_result(pathlib.Path(root), tid, strict=True)
            # A published row outlives its collected origin (no recorded Stop
            # there); an unreadable origin still raises as unknown authority.
            if not row and not (kind == "unknown" and legacy_unrecorded(record)):
                raise ValueError("followup_origin_unavailable")
            row = row or {}
            intent = active_intent(root, tid, strict=True) or {}
            stop = intent.get("followup_stop") or row.get("followup_stop") or {}
            if not stop:
                old = intent or row.get("cancel_origin") or {}
                if old.get("source") in STOP_SOURCES and old.get("request_id"):
                    stop = {"control_id": old["request_id"]}
            if stop:
                controls["stop:" + tid] = stop["control_id"]
        root_id = str(origin.get("root_task_id") or "")
        fence = read_fence(root, root_id) if root_id else {}
        if fence_closed(fence):
            root_row = load_task_result(pathlib.Path(root), root_id, strict=True) or {}
            pause_key = "pause:" + root_id
            if root_row.get("status") in _TRULY_TERMINAL_STATUSES:
                # The old tree stays closed. Restore selects this schedule's
                # exact hold through the existing release carrier only.
                controls[pause_key] = fence["fence_id"]
                if released.get(pause_key) != fence["fence_id"]:
                    wait = "origin_owner_paused"
            else:
                wait = "origin_owner_paused"
        if kind == "related":
            relation = record["followup_relation"]
            binding = relation.get("billing_group") or {}
            if (not binding.get("billing_group_limit_source") or not binding.get("billing_group_limit_revision")
                    or "billing_group_limit_usd" not in binding):
                raise ValueError("followup_billing_authority_unavailable")
            deadline = relation.get("deadline_at")
            if deadline and (parse_deadline_ts(deadline) is None or
                            parse_deadline_ts(deadline) <= datetime.datetime.now(datetime.timezone.utc)):
                wait = "work_deadline_passed"
            fired_id = record.get("last_task_id")
            if fired_id:
                fired = load_task_result(pathlib.Path(root), fired_id, strict=True) or {}
                if fired.get("status") not in {"completed", "cancelled", "failed"} and not bound_task(fired, record):
                    wait = "pending_binding_unavailable"
    except Exception as exc:
        wait, busy = "followup_authority_unavailable", contention(exc)
    controls = {key: value for key, value in controls.items() if released.get(key) != value}
    # An unrecorded (published) relationship is held only by an actual control;
    # its release then needs the relationship decision like any unknown one.
    unknown_hold = kind == "unknown" and (controls or not legacy_unrecorded(record))
    reason = ("relationship_unknown" if unknown_hold else "" if kind == "unknown" else
              "origin_stopped" if any(not key.startswith("pause:") for key in controls) else
              "origin_owner_paused" if controls else "")
    hold = None
    if reason:
        hold = {"hold_id": _hold_id(record, controls),
                "reason": reason, "controls": controls}
    return {"relation": kind, "controls": controls, "wait": wait, "hold": hold, **({"busy": True} if busy else {})}


def _hold_id(record, controls):
    identity = {"schedule_id": record.get("id"), "controls": controls, "relation": relation_kind(record),
                "revision": (record.get("followup_relation") or {}).get("revision")}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def current_hold(record):
    """Classify old Restart fan-out as history, retaining every other control.

    Occurrence claims and firing history are untouched. The scheduler includes
    legacy Restart holds even on disabled settled rows, then audits retirement.
    A mixed hold gets the identity of its remaining controls: a Restore naming
    the old combined hold cannot release the reclassified one.
    """
    hold = record.get("followup_hold")
    if not isinstance(hold, dict) or "restart" not in (hold.get("controls") or {}):
        return hold
    controls = {k: v for k, v in hold["controls"].items() if k != "restart"}
    return {**hold, "controls": controls, "hold_id": _hold_id(record, controls)} if controls else None


def observed_store(root, data):
    """Read-only views disclose new controls even before the next producer pass."""
    out = copy.deepcopy(data)
    for row in out.get("tasks", []):
        view = policy_view(root, out, row)
        durable = row.get("followup_hold")
        observed = view.get("hold") or current_hold(row)
        row.pop("followup_hold", None)
        if observed:
            row["followup_hold"] = observed
        row["hold_persisted"] = bool(durable and durable == observed)
        row["followup_wait"] = view["wait"]
        row["relation"] = view["relation"]
    return out


def scan_policy(root, snapshot):
    """Off-Q observations of rows that can act or still bind an admitted task.

    Pre-occurrence one-shots can retain an unstarted last_task_id: that live
    binding still owes Stop/Restore convergence even though the row is disabled.
    Settled historical ids need only their typed status, never their origins,
    except rows with an unretired Restart control. Those already in this finite
    schedule snapshot join the ordinary audited refresh once, including disabled
    rows; no task-history scan or GET-side migration is needed.
    """
    from ouroboros.store import StoreError, acquisition_budget
    from ouroboros.task_store import task_versions
    from ouroboros.task_status import SETTLED_STATUSES

    rows = [row for row in snapshot.get("tasks") or [] if isinstance(row, dict)]
    try:
        with acquisition_budget(0):  # optional legacy inventory must not stall the active rows
            versions = task_versions(root, [row["last_task_id"] for row in rows
                                           if row.get("last_task_id") and not row.get("enabled", True)
                                           and not isinstance(row.get("occurrence"), dict)])
    except StoreError:
        versions = {}  # unknown legacy bindings wait for the next scan
    live = {tid for tid, (_seq, status, _owes) in versions.items() if status not in SETTLED_STATUSES}
    return {str(row.get("id") or ""): (row, policy_view(root, snapshot, row)) for row in rows
            if row.get("enabled", True) or isinstance(row.get("occurrence"), dict)
            or row.get("delete_requested_at") or row.get("last_task_id") in live
            or "restart" in ((row.get("followup_hold") or {}).get("controls") or {})}


def refresh_policy(root, data, record, *, observed=None):
    """Persist restrictions only through the table producer's audited write seam."""
    from supervisor import queue_schedules as store

    view = policy_view(root, data, record) if observed is None else observed
    hold = view.get("hold") or current_hold(record)
    if view.get("busy") or record.get("followup_hold") == hold and record.get("followup_wait", "") == view["wait"]:
        return view  # contention is not a restriction to persist: no write, no audit flap
    before = copy.deepcopy(record)
    after = {**record, "followup_wait": view["wait"]}
    after.pop("followup_hold", None)
    if hold:
        after["followup_hold"] = hold
    audit = dict(drive_root=pathlib.Path(root), operation_id=uuid.uuid4().hex, actor="host",
                 task_id="", action="hold", schedule_id=str(record["id"]),
                 reason=hold["reason"] if hold else view["wait"] or (
                     "restart_history" if "restart" in ((before.get("followup_hold") or {}).get("controls") or {}) else ""))
    if not store._audit_schedule_mutation(phase="intent", result="intended", before=before, **audit):
        raise store.ScheduleRefused("audit_unavailable", "follow-up restriction audit unavailable; dispatch remains blocked")
    record.clear()
    record.update(after)
    try:
        store._write_scheduled_tasks(data, root)
    except Exception:
        record.clear()
        record.update(before)
        raise
    result = "held" if hold else ("waiting" if view["wait"] else "eligibility_restored")
    if not store._audit_schedule_mutation(phase="outcome", result=result, before=before, after=after, **audit):
        raise store.ScheduleRefused("changed_audit_incomplete", "restriction is durable; audit outcome unavailable")
    return view


@contextmanager
def control_guard(root, record):
    """Share cancellation and actual owner Pause's final local start boundary.

    An automatic caller (``queue_schedules.automatic_acquisitions``) tries each
    lock once; contention raises its typed refusal and nothing is changed."""
    from ouroboros.cancel_intents import cancellation_projection_lock
    from ouroboros.owner_pause import launch_lock
    from supervisor.queue_schedules import automatic

    if relation_kind(record) == "independent":
        yield
        return
    key = (str(pathlib.Path(root).resolve()), origin_of(record).get("root_task_id", ""))
    depths = getattr(_CONTROL_DEPTH, "keys", set())
    if key in depths:
        yield
        return
    wait = not automatic()
    with ExitStack() as stack:
        stack.enter_context(cancellation_projection_lock(root, wait=wait))
        origin = origin_of(record)
        if relation_kind(record) != "independent" and origin.get("root_task_id"):
            stack.enter_context(launch_lock(root, origin["root_task_id"], wait=wait))
        _CONTROL_DEPTH.keys = depths | {key}
        try:
            yield
        finally:
            _CONTROL_DEPTH.keys = depths


@contextmanager
def scheduled_start(root, task):
    """Final published worker handoff; no occurrence identity is manufactured.

Every pending task keeps its ID and existing dispatch evidence. Missing rows or
unknown relationships wait; restoring the schedule does not release task holds.
The occurrence owner persists dispatch before the physical handoff inside this fence.
"""
    from supervisor import queue_schedules as store

    meta = task.get("metadata") or {}
    schedule_id = str(meta.get("schedule_id") or "")
    if not schedule_id:
        yield True
        return
    with store.schedule_transaction(root):
        data = store.load_schedule_store(root)
        row = next((r for r in data["tasks"] if r.get("id") == schedule_id), None)
        if row is None:
            from ouroboros.task_results import load_task_result
            actual = load_task_result(pathlib.Path(root), str(task.get("id") or ""), strict=True) or {}
            meta = actual.get("metadata") if isinstance(actual.get("metadata"), dict) else {}
            # Only related work needs its row; a published task recorded none.
            kind = (meta.get("followup_relation") or {}).get("kind") if "followup_relation" in meta else ""
            yield bool(actual) and kind in {"", "independent", "unknown"}
            return
        with control_guard(root, row):
            view = refresh_policy(root, data, row)
            if view.get("busy"):  # contention defers the start; it never reads as a held row
                from ouroboros.store import StoreBusy
                error = StoreBusy("follow-up authority busy", reason="contention")
                error.rollback_confirmed = True
                raise error
            yield not (view.get("hold") or view["wait"] or row.get("followup_hold")) and bound_task(task, row)


def bound_task(task, record):
    """Prepared work must carry the current host relationship and original money.

    Independent and unrecorded (published) rows carry neither money nor a
    deadline: a task frozen before relationships existed keeps its identity.
    """
    from ouroboros.deadline_utils import parse_deadline_ts
    carried = (task.get("metadata") or {}).get("followup_relation") or {}
    kind = relation_kind(record)
    if kind == "independent":
        return carried.get("kind") in {None, "unknown", "independent"}
    if kind != "related":
        return legacy_unrecorded(record) and carried.get("kind") in {None, "unknown"}
    if carried.get("kind") != kind:
        return False
    relation = record.get("followup_relation") or {}
    if (task.get("metadata") or {}).get("billing_group") != relation.get("billing_group"):
        return False
    deadline = parse_deadline_ts(relation.get("deadline_at"))
    if deadline:
        carried = parse_deadline_ts(task.get("deadline_at"))
        contract = parse_deadline_ts((task.get("task_contract") or {}).get("deadline_at"))
        if carried is None or carried > deadline or (contract is not None and contract > deadline):
            return False
    return True


def bind_task(root, task, record):
    """Bind creation or a proven unrun occurrence to the row's host-owned facts.

    This changes neither task identity nor execution/resource intent. The
    occurrence owner alone may apply it to already admitted work.
    """
    from ouroboros.deadline_utils import parse_deadline_ts

    meta = task.setdefault("metadata", {})
    origin = origin_of(record)
    if origin:
        meta.update(origin_task_id=origin["task_id"], origin_root_task_id=origin["root_task_id"])
        meta["objective_author"] = {"kind": "task", "task_id": origin["task_id"]}
    meta["followup_origin"] = origin
    meta["followup_relation"] = copy.deepcopy(record.get("followup_relation") or {"kind": relation_kind(record)})
    if relation_kind(record) != "related":
        return
    from ouroboros.usage_ledger import BillingAuthorityUnavailable
    try:
        meta["billing_group"] = task_binding(root, record)
    except BillingAuthorityUnavailable as exc:
        raise ValueError("followup_billing_authority_unavailable") from exc
    relation = record["followup_relation"]
    dates = [parse_deadline_ts(raw) for raw in (relation.get("deadline_at"), task.get("deadline_at"),
             (task.get("task_contract") or {}).get("deadline_at")) if raw]
    if any(date is None for date in dates):
        raise ValueError("followup_deadline_unavailable")
    if dates:
        task["deadline_at"] = meta["deadline_at"] = min(dates).isoformat()
        if task.get("task_contract"):
            task["task_contract"]["deadline_at"] = task["deadline_at"]
            meta["task_contract"] = copy.deepcopy(task["task_contract"])


def normalize_template(record):
    """Discard host-shaped template input, including historical poisoned rows."""
    from ouroboros.schedule_contract import RESERVED_TEMPLATE_FIELDS
    template = copy.deepcopy(record.get("task") or {})
    for key in RESERVED_TEMPLATE_FIELDS:
        template.pop(key, None)
    metadata = template.get("metadata") or {}
    template["metadata"] = {k: v for k, v in metadata.items() if k not in RESERVED_TEMPLATE_FIELDS}
    return template


def task_binding(root, record):
    """The host row is the only authority for scheduled whole-work money.

    ``None`` means the task's own root economics: independent work, and an
    unrecorded (published) relationship, which never borrows its origin's cap.
    """
    from ouroboros.usage_admission import authority_unavailable
    kind = relation_kind(record)
    if kind == "independent" or (kind == "unknown" and legacy_unrecorded(record)):
        return None
    binding = (record.get("followup_relation") or {}).get("billing_group") or {}
    if kind != "related" or not binding.get("billing_group_limit_source"):
        raise authority_unavailable("refused", str(record.get("id")), "related_followup_unbound")  # read cleanly
    return copy.deepcopy(binding)


def restore_followup(root, data, current, *, expected_hold_id, resolved, actor, task_id, reason,
                     writer_kind="consciousness", writer_metadata=None):
    """Release precisely a durable, observed generation; never enable other holds."""
    from supervisor import queue_schedules as store
    with control_guard(root, current):
        before = copy.deepcopy(current)
        prior_hold = before.get("followup_hold") or {}
        view = refresh_policy(root, data, current)
        hold = current.get("followup_hold") or {}
        status = ""
        if not prior_hold:
            status = "hold_not_observed"
        elif not expected_hold_id or expected_hold_id != hold.get("hold_id"):
            status = "stale_hold"
        elif relation_kind(current) == "unknown" and not resolved:
            status = "relationship_required"
        elif resolved and resolved[0] != origin_of(current):
            status = "origin_changed"
        elif view["wait"] in {"followup_authority_unavailable", "work_deadline_passed", "owner_restart_in_progress"}:
            status = view["wait"]
        audit = dict(drive_root=pathlib.Path(root), operation_id=uuid.uuid4().hex, actor=actor,
                     task_id=task_id, action="restore", schedule_id=str(current["id"]), reason=reason)
        if not store._audit_schedule_mutation(phase="intent", result="intended", before=before, **audit):
            raise store.ScheduleRefused("audit_unavailable", "restore not applied")
        if status:
            recorded = store._audit_schedule_mutation(phase="outcome", result=status, before=current, **audit)
            return {"ok": False, "changed": current != before, "status": status,
                    "schedule_id": current["id"], "schedule": store._audit_row(current),
                    "running_or_queued": store._schedule_running_or_queued(current["id"], root),
                    "audit": "recorded" if recorded else "incomplete"}
        after = copy.deepcopy(current)
        if resolved:
            if relation_kind(current) != "unknown":
                raise store.ScheduleRefused("relationship_immutable", "only an unknown relationship can be resolved")
            after["followup_origin"], after["followup_relation"] = resolved
        from ouroboros.consciousness_authority import stamp_schedule_writer
        stamp_schedule_writer(after, writer_kind, writer_metadata,
                              started_root=relation_kind(after) == "independent")
        after["followup_released"] = {**(after.get("followup_released") or {}), **hold.get("controls", {})}
        after.pop("followup_hold", None)
        after["followup_wait"] = policy_view(root, data, after)["wait"]
        current.clear()
        current.update(after)
        store._write_scheduled_tasks(data, root)
        recorded = store._audit_schedule_mutation(phase="outcome", result="hold_released", before=before, after=after, **audit)
        return {"ok": recorded, "changed": True,
                "status": "hold_released" if recorded else "changed_audit_incomplete",
                "schedule_id": current["id"], "schedule": store._audit_row(current),
                "running_or_queued": store._schedule_running_or_queued(current["id"], root),
                "audit": "recorded" if recorded else "incomplete",
                "detail": "Exact future hold released; enabled, fired task, money, deadline and other task holds preserved."}


def record_restart(root, *, new=False):
    """Retain the manual Restart episode before consuming its existing marker.

    The marker closes admission during restart. Crossing this episode fences an
    off-lock preparation once; healthy future schedules acquire no durable hold.
    """
    from supervisor import queue_schedules as store

    with store.schedule_transaction(root):
        data = store.load_schedule_store(root)
        # Each boot retry while the marker exists may tighten the generation.
        control = {"control_id": uuid.uuid4().hex}
        audit = dict(drive_root=pathlib.Path(root), operation_id=control["control_id"], actor="owner:restart",
                     task_id="", action="record_restart", schedule_id="*", reason="manual_restart")
        if not store._audit_schedule_mutation(phase="intent", result="intended", **audit):
            raise store.ScheduleRefused("audit_unavailable", "Restart episode not recorded")
        data["followup_restart"] = control
        store._write_scheduled_tasks(data, root)
        if not store._audit_schedule_mutation(phase="outcome", result="recorded", **audit):
            raise store.ScheduleRefused("changed_audit_incomplete", "Restart episode recorded; outcome audit unavailable")
