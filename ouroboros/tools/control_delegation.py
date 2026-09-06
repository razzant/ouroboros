"""Delegation-budget + in-task project-scoping affordances (v6.37.0).

Extracted from ``ouroboros/tools/control.py`` to keep that dispatcher module under
the module-size hard gate. These are the cyber-racing postmortem additions: the
typed child delegation-budget narrowing (C3.1) and the ``ensure_project_scope``
tool handler (C4.1). ``control.py`` imports both; ``_ensure_project_scope`` reaches
back into ``control._emit_control_event`` lazily (call-time) so there is no import
cycle.
"""

from __future__ import annotations

import json
import pathlib
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable

from ouroboros.contracts.task_contract import (
    _normalized_intent_note,
    normalize_bool,
    normalize_depth_provenance,
)
from ouroboros.config import MAX_SUBAGENT_DEPTH_HARD_CAP
from ouroboros.depth_evidence import parse_task_depth
from ouroboros.tools.registry import ToolContext
from ouroboros.utils import utc_now_iso

# FR2: ONE shared from-scratch git tree per task-tree for cooperative acting-subagent
# builds, keyed by root_task_id. Process-local cache so multiple fan-out waves of the
# SAME parent route children into the SAME tree (the parent worker schedules them all).
_COOP_SHARED_ROOTS: Dict[str, str] = {}
_COOP_LOCK = threading.Lock()


@dataclass(frozen=True)
class DelegationBudgetDecision:
    ok: bool
    budget: Dict[str, Any]
    reason_code: str = ""
    detail: str = ""
    # The lane an applicable, non-advisory `require_lane` constraint verified
    # this admission against (F9/sol #1). Empty when no such constraint applied.
    # Carried out so the scheduler can stamp it onto the child record: the
    # dispatch-time policy default (auto+harness ⇒ light) must not override a
    # lane the admission gate just enforced.
    required_lane: str = ""


@dataclass(frozen=True)
class DelegationAdmissionDecision:
    """Typed parent-rights admission outcome for one direct-child request."""

    ok: bool
    reason_code: str = ""
    detail: str = ""
    direct_child_count: int | None = 0


def check_delegation_admission(
    parent_budget: Dict[str, Any], *, direct_child_count: int | None = 0,
) -> DelegationAdmissionDecision:
    """Enforce explicit recursion rights while keeping omitted legacy permissive."""
    budget = parent_budget if isinstance(parent_budget, dict) else {}
    if direct_child_count is None:
        count = None
    else:
        try:
            count = max(0, int(direct_child_count))
        except (TypeError, ValueError):
            count = None
    if not normalize_bool(budget.get("may_delegate", True)):
        return DelegationAdmissionDecision(
            False,
            reason_code="delegation_rights_may_delegate",
            detail="The parent delegation budget explicitly forbids descendants (may_delegate=false).",
            direct_child_count=count,
        )
    depth_remaining = budget.get("depth_remaining")
    try:
        if depth_remaining is not None and int(depth_remaining) <= 0:
            return DelegationAdmissionDecision(
                False,
                reason_code="delegation_rights_depth_exhausted",
                detail="The parent delegation budget has no remaining descendant depth.",
                direct_child_count=count,
            )
    except (TypeError, ValueError):
        # Malformed depth is treated as unknown here; the existing configured
        # depth gate remains authoritative and emits its own typed refusal.
        pass
    max_children = budget.get("max_children")
    max_children_limit = None
    try:
        if max_children is not None and int(max_children) > 0:
            max_children_limit = int(max_children)
    except (TypeError, ValueError):
        pass
    fan_out_allowed = normalize_bool(budget.get("may_fan_out", True))
    if count is None and (max_children_limit is not None or not fan_out_allowed):
        return DelegationAdmissionDecision(
            False,
            reason_code="delegation_rights_child_count_unknown",
            detail=(
                "The host could not prove the parent's current direct-child count, "
                "so it cannot safely apply the explicit fan-out/child cap."
            ),
            direct_child_count=None,
        )
    if (
        max_children_limit is not None
        and count is not None
        and count >= max_children_limit
    ):
        return DelegationAdmissionDecision(
            False,
            reason_code="delegation_rights_max_children",
            detail=(
                "The parent delegation budget has reached its explicit direct-child "
                f"cap ({max_children_limit})."
            ),
            direct_child_count=count,
        )
    if not fan_out_allowed and count is not None and count >= 1:
        return DelegationAdmissionDecision(
            False,
            reason_code="delegation_rights_may_fan_out",
            detail=(
                "The parent delegation budget permits one direct child but forbids "
                "additional direct children (may_fan_out=false)."
            ),
            direct_child_count=count,
        )
    return DelegationAdmissionDecision(ok=True, direct_child_count=count)


def durable_direct_child_count(
    drive_root: Any, parent_task_id: Any, *, exclude_task_id: Any = "",
) -> int | None:
    """Count the parent's already admitted direct children from task-result SSOT."""
    parent = str(parent_task_id or "").strip()
    excluded = str(exclude_task_id or "").strip()
    if not parent or not str(drive_root or "").strip():
        return None
    try:
        from ouroboros.task_results import STATUS_REJECTED_DUPLICATE, STATUS_REQUESTED, list_task_results

        rows = list_task_results(drive_root, strict=True)
    except Exception:
        return None
    return sum(
        1 for row in rows
        if isinstance(row, dict)
        and str(row.get("parent_task_id") or "").strip() == parent
        and str(row.get("delegation_role") or "") == "subagent"
        and str(row.get("status") or "") not in {STATUS_REQUESTED, STATUS_REJECTED_DUPLICATE}
        and not (
            str(row.get("status") or "") == "failed"
            and (
                not isinstance(row.get("delegation_admission"), dict)
                or str(row["delegation_admission"].get("status") or "") != "accepted"
            )
        )
        and str(row.get("id") or row.get("task_id") or "").strip() != excluded
    )


def _bounded_permitted_depth(value: Any) -> int | None:
    """Keep persisted admission authority within the immutable host ceiling."""
    try:
        return min(MAX_SUBAGENT_DEPTH_HARD_CAP, max(0, int(value)))
    except (TypeError, ValueError):
        return None


def admitted_depth_cap(parent_contract: Any, live_max_depth: Any) -> int:
    """Return the admitted lineage cap, bounded by current global controls.

    A persisted permitted depth is an admission fact and survives ordinary
    Settings changes. The explicit global zero remains an owner kill-switch,
    and the immutable host ceiling always applies.
    """
    budget = (
        parent_contract.get("delegation_budget", parent_contract)
        if isinstance(parent_contract, dict)
        else parent_contract
    )
    budget = budget if isinstance(budget, dict) else {}
    try:
        live_cap = min(MAX_SUBAGENT_DEPTH_HARD_CAP, max(0, int(live_max_depth)))
    except (TypeError, ValueError):
        live_cap = 0
    # The explicit global zero is an owner kill-switch and applies even to a
    # lineage admitted under an earlier non-zero setting.
    if live_cap == 0:
        return 0
    provenance = normalize_depth_provenance(budget.get("depth_provenance"))
    permitted = _bounded_permitted_depth(provenance.get("permitted_depth"))
    if permitted is not None:
        return permitted
    # An incomplete/malformed projection is not an admission fact. Preserve the
    # legacy live-limit behavior until a typed contract can supply a valid cap.
    return live_cap


def depth_provenance_for_schedule(
    parent_budget: Dict[str, Any], *, new_depth: int, max_depth: int,
    achieved_depth: Any = None, use_remaining_envelope: bool = False,
    requested_depth: Any = None,
) -> Dict[str, Any]:
    """Carry requested/permitted/attempted/achieved depth as additive facts.

    A request is telemetry, never a cap: only the configured cap and the legacy
    remaining envelope narrow what a branch may do, so asking for less than the
    cap records the intent without silently binding the descendants to it.
    """
    budget = parent_budget if isinstance(parent_budget, dict) else {}
    inherited = normalize_depth_provenance(budget.get("depth_provenance"))
    requested = inherited.get("requested_depth")
    # Only the root scheduler can treat an explicitly supplied legacy envelope
    # as request provenance. A descendant's remaining value is already narrowed
    # and cannot reconstruct what the root originally requested.
    if (
        requested is None
        and use_remaining_envelope
        and new_depth == 1
        and "depth_remaining" in budget
        and not inherited
    ):
        try:
            requested = max(0, int(budget.get("depth_remaining")))
        except (TypeError, ValueError):
            requested = None
    if requested is None:
        # An explicit request of record on THIS call stays unchanged. A
        # non-integer fails soft to "no request recorded": the
        # field is telemetry, and refusing the whole schedule over it would
        # trade a real capability for a bookkeeping nicety.
        try:
            explicit = max(0, int(requested_depth or 0))
        except (TypeError, ValueError):
            explicit = 0
        if explicit > 0:
            requested = explicit
    try:
        cap = min(MAX_SUBAGENT_DEPTH_HARD_CAP, max(0, int(max_depth)))
    except (TypeError, ValueError):
        cap = 0
    current_permitted = cap
    remaining = budget.get("depth_remaining")
    if (
        use_remaining_envelope
        and not inherited
        and isinstance(remaining, int)
        and not isinstance(remaining, bool)
    ):
        # A legacy descendant may lack provenance, but its already-narrowed
        # remaining envelope still bounds authority: parent depth is new_depth-1.
        current_permitted = min(
            current_permitted,
            max(0, int(new_depth) - 1) + max(0, remaining),
        )
    inherited_permitted = _bounded_permitted_depth(inherited.get("permitted_depth"))
    if inherited_permitted is None:
        permitted = current_permitted
    else:
        # A persisted admission fact survives ordinary Settings changes, but
        # never exceeds the immutable host ceiling.
        permitted = inherited_permitted
    try:
        attempted = max(0, int(new_depth))
    except (TypeError, ValueError):
        attempted = None
    try:
        achieved = None if achieved_depth is None else max(0, int(achieved_depth))
    except (TypeError, ValueError):
        achieved = None
    return {
        "requested_depth": requested,
        "permitted_depth": permitted,
        "attempted_depth": attempted,
        "achieved_depth": achieved,
    }


def stamp_depth_provenance(
    task_contract: Dict[str, Any], *, attempted_depth: int, max_depth: int,
    achieved_depth: Any = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Stamp admitted depth, deriving intent only from an explicit root envelope."""
    contract = dict(task_contract) if isinstance(task_contract, dict) else {}
    budget = dict(contract.get("delegation_budget") or {})
    inherited = normalize_depth_provenance(budget.get("depth_provenance"))
    if inherited:
        provenance = depth_provenance_for_schedule(
            budget, new_depth=attempted_depth, max_depth=max_depth,
            achieved_depth=achieved_depth,
        )
    else:
        remaining = budget.get("depth_remaining")
        requested = None
        permitted = None
        if isinstance(remaining, int) and not isinstance(remaining, bool):
            if max(0, int(attempted_depth)) == 0:
                requested = max(0, remaining)
            permitted = _bounded_permitted_depth(
                min(
                    min(MAX_SUBAGENT_DEPTH_HARD_CAP, max(0, int(max_depth))),
                    max(0, int(attempted_depth)) + max(0, remaining),
                )
            )
        provenance = {
            "requested_depth": requested,
            "permitted_depth": permitted,
            "attempted_depth": max(0, int(attempted_depth)),
            "achieved_depth": (
                None if achieved_depth is None else max(0, int(achieved_depth))
            ),
        }
    budget["depth_provenance"] = provenance
    contract["delegation_budget"] = budget
    return contract, provenance


def schedule_delegation_refusal(
    parent_contract: Dict[str, Any], status_root: Any, parent_task_id: Any,
) -> str:
    """Return the typed direct-child rights refusal, or an empty string."""

    budget = parent_contract.get("delegation_budget") if isinstance(parent_contract, dict) else {}
    decision = check_delegation_admission(
        budget if isinstance(budget, dict) else {},
        direct_child_count=durable_direct_child_count(status_root, parent_task_id),
    )
    if decision.ok:
        return ""
    return f"⚠️ TOOL_ERROR (schedule_subagent): {decision.reason_code}: {decision.detail}"


def stamp_task_assignment_depth(
    task: Dict[str, Any], *, max_depth: int,
) -> Dict[str, Any]:
    """Stamp the first host-visible achieved-depth fact onto the worker payload."""

    contract = task.get("task_contract") if isinstance(task.get("task_contract"), dict) else {}
    if not contract:
        return {}
    depth = parse_task_depth(task.get("depth"), default=0)
    task["depth"] = depth
    budget = dict(contract.get("delegation_budget") or {})
    admitted = normalize_depth_provenance(budget.get("depth_provenance"))
    if admitted:
        requested = admitted.get("requested_depth")
        permitted = _bounded_permitted_depth(admitted.get("permitted_depth"))
        if permitted is None:
            permitted = min(MAX_SUBAGENT_DEPTH_HARD_CAP, max(0, int(max_depth)))
        provenance = {
            "requested_depth": requested,
            "permitted_depth": permitted,
            "attempted_depth": (
                admitted.get("attempted_depth")
                if admitted.get("attempted_depth") is not None
                else depth
            ),
            "achieved_depth": depth,
        }
        budget["depth_provenance"] = provenance
        achieved_contract = {**contract, "delegation_budget": budget}
    else:
        # A no-provenance assignment is a legacy/historical row. Its depth is
        # observable, but mutable live Settings cannot reconstruct the request
        # or permission that admitted it.
        provenance = {
            "requested_depth": None,
            "permitted_depth": None,
            "attempted_depth": depth,
            "achieved_depth": depth,
        }
        budget["depth_provenance"] = provenance
        achieved_contract = {**contract, "delegation_budget": budget}
    task["task_contract"] = achieved_contract
    task["depth_provenance"] = provenance
    metadata = dict(task.get("metadata")) if isinstance(task.get("metadata"), dict) else {}
    metadata["task_contract"] = achieved_contract
    metadata["depth_provenance"] = provenance
    task["metadata"] = metadata
    return {"task_contract": achieved_contract, "depth_provenance": provenance}


def record_depth_limit_refusal(
    ctx: Any,
    fields: Dict[str, Any],
    params: Dict[str, Any],
    configured_subagent: Dict[str, Any],
    *,
    current_depth: int,
    new_depth: int,
    max_depth: int,
) -> str:
    """Persist one normal over-cap spawn attempt without starting a child."""

    metadata = (
        getattr(ctx, "task_metadata", {})
        if isinstance(getattr(ctx, "task_metadata", {}), dict)
        else {}
    )
    parent_contract = (
        metadata.get("task_contract")
        if isinstance(metadata.get("task_contract"), dict)
        else {}
    )
    if not parent_contract and isinstance(getattr(ctx, "task_contract", None), dict):
        parent_contract = getattr(ctx, "task_contract")
    parent_id = str(getattr(ctx, "task_id", "") or metadata.get("parent_task_id") or "")
    task_id = uuid.uuid4().hex[:8]
    root_id = str(metadata.get("root_task_id") or parent_id or task_id)
    session_id = str(metadata.get("session_id") or "")
    status_root = pathlib.Path(str(
        metadata.get("budget_drive_root")
        or getattr(ctx, "budget_drive_root", "")
        or getattr(ctx, "drive_root", ".")
    ))
    child_budget = child_budget_for_schedule(
        parent_contract,
        current_depth=current_depth,
        new_depth=new_depth,
        max_depth=max_depth,
        may_mutate=fields.get("may_mutate", False),
        may_fan_out=params.get("may_fan_out", True),
        max_children=params.get("max_children", 0),
        intent_note=params.get("delegation_intent", ""),
        requested_depth=params.get("requested_depth", 0),
    )
    from ouroboros.contracts.task_contract import build_task_contract

    child_contract = build_task_contract({
        "objective": fields.get("objective"),
        "expected_output": fields.get("expected_output"),
        "constraints": fields.get("constraints"),
        "context": fields.get("context"),
        "parent_task_id": parent_id,
        "root_task_id": root_id,
        "session_id": session_id,
        "delegation_role": "subagent",
        "allowed_resources": parent_contract.get("allowed_resources", {}),
        "delegation_budget": child_budget,
    })
    child_contract, provenance = stamp_depth_provenance(
        child_contract,
        attempted_depth=new_depth,
        max_depth=max_depth,
        achieved_depth=None,
    )
    reason_code = "subtask_depth_limit"
    detail = f"Subtask depth limit ({max_depth}) exceeded by attempted depth {new_depth}."
    try:
        from ouroboros.task_results import STATUS_FAILED, write_task_result

        saved = write_task_result(
            status_root,
            task_id,
            STATUS_FAILED,
            parent_task_id=parent_id or None,
            root_task_id=root_id,
            session_id=session_id,
            actor_id=f"subagent:{fields.get('role') or 'researcher'}",
            delegation_role="subagent",
            role=str(fields.get("role") or "researcher"),
            description=str(fields.get("objective") or ""),
            objective=str(fields.get("objective") or ""),
            expected_output=str(fields.get("expected_output") or ""),
            constraints=str(fields.get("constraints") or ""),
            context=str(fields.get("context") or ""),
            depth=new_depth,
            task_contract=child_contract,
            depth_provenance=provenance,
            configured_subagent=dict(configured_subagent or {}),
            budget_drive_root=str(status_root),
            memory_mode=str(fields.get("memory_mode") or ""),
            reason_code=reason_code,
            delegation_admission={
                "status": "rejected",
                "reason_code": reason_code,
                "attempted_depth": new_depth,
                "permitted_depth": provenance.get("permitted_depth"),
            },
            result=f"Subagent rejected: {reason_code}: {detail}",
        )
        if not saved:
            raise RuntimeError("task result transition was not persisted")
    except Exception:
        return (
            "⚠️ TOOL_ERROR (schedule_subagent): depth_refusal_evidence_unavailable: "
            f"{detail} The child was not started, but durable attempt evidence could "
            "not be written; treat attempted depth as unknown."
        )
    return (
        f"⚠️ TOOL_ERROR (schedule_subagent): {reason_code}: {detail} "
        f"task_id={task_id}; depth_provenance="
        + json.dumps(provenance, ensure_ascii=False, sort_keys=True)
    )


def _constraint_payload(row: Any) -> Dict[str, Any]:
    if isinstance(row, dict):
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else row
        return payload if isinstance(payload, dict) else {}
    return {}


def _constraint_applies(payload: Dict[str, Any], *, role: str = "", write_surface: str = "") -> bool:
    scope = payload.get("scope")
    if not isinstance(scope, dict):
        return True
    role_req = str(scope.get("role") or "").strip()
    if role_req and role_req != str(role or "").strip():
        return False
    surface_req = str(scope.get("surface") or "").strip()
    if surface_req and surface_req != str(write_surface or "").strip():
        return False
    return True


def effective_delegation_budget(
    declared_budget: Dict[str, Any],
    *,
    missing_capabilities: Iterable[str] = (),
    unresolved_constraints: Iterable[Dict[str, Any]] = (),
    write_surface: str = "",
    role: str = "",
    requested_lane: str = "",
    # What the request MEANS (`subagents.intended_lane`), not what the child ends up
    # running. This gate runs at ADMISSION, before the child is dispatched, so the
    # effective lane does not exist yet — it used to be handed a schedule-time
    # resolution, which is a live answer given before the queue wait.
    intended_lane: str = "",
    active_child_count: int | None = None,
) -> DelegationBudgetDecision:
    """Reconcile schedule-time delegation budget with needs and back-constraints.

    Pure admission reducer: no IO, no queue mutation, no tool dispatch. It narrows
    existing delegation-budget vocabulary or returns a typed rejection reason.
    """

    budget = dict(declared_budget if isinstance(declared_budget, dict) else {})
    required_lane_applied = ""
    missing = [str(cap or "").strip() for cap in missing_capabilities or [] if str(cap or "").strip()]
    if missing:
        return DelegationBudgetDecision(
            False,
            budget,
            reason_code="capability_profile_mismatch",
            detail=(
                "Declared required_capabilities are not available to the selected subagent profile: "
                + ", ".join(missing)
            ),
        )
    effective_max = budget.get("max_children")
    for row in unresolved_constraints or []:
        payload = _constraint_payload(row)
        if bool(payload.get("advisory")):
            continue
        if not _constraint_applies(
            payload,
            role=role,
            write_surface=write_surface,
        ):
            continue
        directive = str(payload.get("directive") or "").strip().lower()
        if directive == "halt_fanout":
            return DelegationBudgetDecision(
                False,
                budget,
                reason_code="delegation_constraint_halt_fanout",
                detail=str(payload.get("rationale") or "Unresolved delegation constraint halted fan-out."),
            )
        if directive == "block_surface":
            scope = payload.get("scope")
            if isinstance(scope, dict):
                blocked_surface = str(scope.get("surface") or "").strip()
            else:
                blocked_surface = str(scope or "").strip()
            if blocked_surface and blocked_surface == str(write_surface or "").strip():
                return DelegationBudgetDecision(
                    False,
                    budget,
                    reason_code="delegation_constraint_block_surface",
                    detail=str(payload.get("rationale") or f"Surface {blocked_surface!r} is blocked by an unresolved delegation constraint."),
                )
        if directive == "require_lane":
            scope = payload.get("scope")
            required_lane = str(scope.get("lane") if isinstance(scope, dict) else scope or "").strip()
            if required_lane:
                required_lane_applied = required_lane
            if required_lane and required_lane != str(intended_lane or "").strip():
                return DelegationBudgetDecision(
                    False,
                    budget,
                    reason_code="delegation_constraint_require_lane",
                    # The refusal states the FACTS this reducer holds — required,
                    # requested, intended — and nothing about what an omitted lane
                    # means. It used to restate that default, and the default is
                    # owned by `subagents.intended_lane`, three modules away: the
                    # sentence went stale in v6.87.7, was corrected in v6.87.14, and
                    # went stale AGAIN in v6.87.26, each time telling the model the
                    # opposite of the truth at the exact moment it is deciding how to
                    # fix a rejected spawn. A copy of a rule you do not own drifts;
                    # the remedy below holds whatever the default is.
                    detail=str(
                        payload.get("rationale")
                        or (
                            f"Unresolved delegation constraint requires lane {required_lane!r} "
                            f"(requested {str(requested_lane or '').strip() or 'auto'!r}, "
                            f"intended {str(intended_lane or '').strip()!r}). Name "
                            f"model_lane={required_lane!r} explicitly, or "
                            "override_delegation_constraint("
                            f"{str(payload.get('constraint_id') or '').strip()!r}) instead. If this "
                            "install has no such slot the child will run Main and disclose the "
                            "reduction in capability_delta."
                        )
                    ),
                )
        if directive == "cap_children":
            scope = payload.get("scope")
            cap_value: int | None = None
            if isinstance(scope, dict):
                raw_cap = scope.get("max_children")
            else:
                raw_cap = scope
            try:
                cap_value = int(raw_cap)
            except (TypeError, ValueError):
                cap_value = None
            if cap_value is not None and cap_value >= 0:
                if isinstance(effective_max, int) and effective_max > 0:
                    effective_max = min(effective_max, cap_value)
                else:
                    effective_max = cap_value
                if active_child_count is not None and cap_value >= 0 and int(active_child_count) >= cap_value:
                    return DelegationBudgetDecision(
                        False,
                        {**budget, "max_children": effective_max},
                        reason_code="delegation_constraint_child_cap",
                        detail=str(payload.get("rationale") or f"Unresolved delegation constraint caps children at {cap_value}."),
                    )
    if effective_max is not None:
        budget["max_children"] = effective_max
    return DelegationBudgetDecision(True, budget, required_lane=required_lane_applied)


def normalize_required_capabilities(value: Any) -> tuple[list[str], str]:
    from ouroboros.tool_access import SUBAGENT_CAPABILITIES

    if value in (None, "", ()):
        return [], ""
    if not isinstance(value, (list, tuple)):
        return [], "required_capabilities must be a list of strings."
    caps = [str(item or "").strip().lower() for item in value if str(item or "").strip()]
    invalid = [cap for cap in caps if cap not in SUBAGENT_CAPABILITIES]
    if invalid:
        return [], (
            "required_capabilities contains unsupported value(s): "
            + ", ".join(invalid)
            + f". Expected one of {', '.join(SUBAGENT_CAPABILITIES)}."
        )
    return caps, ""


def profile_from_task_constraint(task_constraint: Dict[str, Any]) -> str:
    from ouroboros.tool_capabilities import ACTING_SUBAGENT_MODE

    return (
        "acting_subagent"
        if task_constraint.get("mode") == ACTING_SUBAGENT_MODE and task_constraint.get("surface")
        else "local_readonly_subagent"
    )


def ensure_cooperative_shared_root(ctx: Any, root_task_id: str) -> str:
    """Mint (once per task-tree) ONE shared from-scratch git tree for a cooperative
    acting-subagent build, reusing ``subagent_worktrees.provision_genesis_project``
    (a durable standalone repo under the projects root, outside repo/ and data/).
    Cached by ``root_task_id`` so multiple fan-out waves share ONE tree. Children
    write into it via ``write_surface=external_workspace``; deeper descendants inherit
    it automatically through ``parent_workspace_root``. Returns the tree path, or a
    ``⚠️`` error string for the LLM."""
    key = str(root_task_id or "").strip() or str(getattr(ctx, "task_id", "") or "").strip()
    # Hold the lock across BOTH the cache check AND the mint so two concurrent fan-out
    # waves of the same root cannot each provision a tree (no double-mint / orphan tree).
    # The mint is a one-time `git init` on an empty dir, so serializing it is cheap.
    with _COOP_LOCK:
        cached = _COOP_SHARED_ROOTS.get(key)
        if cached and pathlib.Path(cached).is_dir():
            return cached
        try:
            from ouroboros.subagent_worktrees import provision_genesis_project

            handle = provision_genesis_project(
                repo_dir=ctx.repo_dir,
                task_id=key or uuid.uuid4().hex[:8],
                parent_task_id=str(getattr(ctx, "task_id", "") or ""),
                dir_name=f"coop_{key[:12]}" if key else "coop",
            )
        except Exception as exc:
            return (
                "⚠️ COOP_WORKSPACE_ERROR: could not provision a shared cooperative workspace "
                f"({type(exc).__name__}: {exc}); pass write_root explicitly or schedule read-only children."
            )
        path = str(handle.path)
        _COOP_SHARED_ROOTS[key] = path
        return path


def resolve_cooperative_write_root(
    ctx: Any, requested_surface: str, write_root: str, workspace_root: str, metadata: Dict[str, Any]
) -> tuple[str, str, str]:
    """Resolve the effective acting write_root and the caller profile for a scheduled
    wave. A flat parent (no inherited workspace) requesting external_workspace with no
    write_root builds cooperatively from scratch, so the host mints ONE shared tree.
    Returns ``(effective_write_root, caller_profile, error_or_empty)``. Gated on the
    mutative toggle so a disabled setup falls through to the disabled message rather
    than minting an unused tree."""
    from ouroboros.tool_access import active_tool_profile

    caller_profile = active_tool_profile(ctx)
    effective = write_root
    if (
        requested_surface == "external_workspace"
        and not str(write_root or "").strip()
        and not workspace_root
        and caller_profile != "local_readonly_subagent"
    ):
        from ouroboros.config import get_allow_mutative_subagents

        if get_allow_mutative_subagents("external_workspace"):
            key = str((metadata or {}).get("root_task_id") or getattr(ctx, "task_id", "") or "").strip()
            shared = ensure_cooperative_shared_root(ctx, key)
            if isinstance(shared, str) and shared.startswith("⚠️"):
                return "", caller_profile, shared
            effective = shared
    return effective, caller_profile, ""


def _narrow_child_delegation_budget(
    parent_budget: Dict[str, Any],
    *,
    child_depth_remaining: int,
    may_mutate: bool,
    may_fan_out: bool,
    max_children: int,
    intent_note: str,
    parent_is_subagent: bool = True,
) -> Dict[str, Any]:
    """Build a child's delegation_budget that only ever NARROWS within the parent's
    (C3.1): recursion authority (delegate/fan-out) is AND-ed with the parent's and
    max_children is capped to the parent's positive cap, so a parent that disabled
    delegation/fan-out can never hand a child MORE recursion authority than it holds.

    ``may_mutate`` is special: a ROOT task's default budget is may_mutate=False
    ("mutation is opt-in"), which is NOT an explicit read-only denial — so a root
    HONORS the per-call may_mutate grant (the agent explicitly asking for a mutative
    child). Only a SUBAGENT parent's may_mutate gates the child, so a read-only
    subagent cannot escalate by spawning a mutative descendant. (``parent_is_subagent``
    defaults True — the conservative choice for an unspecified caller.)

    Legacy contracts carry no delegation_budget, so a missing parent authority defaults
    to True (unrestricted — pre-C3.1 behavior)."""
    parent_budget = parent_budget if isinstance(parent_budget, dict) else {}
    parent_may_delegate = normalize_bool(parent_budget.get("may_delegate", True))
    parent_may_mutate = normalize_bool(parent_budget.get("may_mutate", True))
    parent_may_fan_out = normalize_bool(parent_budget.get("may_fan_out", True))
    parent_max_children = parent_budget.get("max_children")
    if isinstance(max_children, int) and max_children > 0:
        child_max_children = max_children
        if isinstance(parent_max_children, int) and parent_max_children > 0:
            child_max_children = min(child_max_children, parent_max_children)
    else:
        child_max_children = parent_max_children
    # STRICT boolean parse of the per-call grants (live-subagent contract): a tool
    # call may pass the STRING "false"/"0" — bool("false") is truthy and would
    # silently grant mutation/fan-out, so route through the same normalize_bool the
    # contract uses. The parent_* flags come from a normalized contract (real bools).
    child_may_mutate = normalize_bool(may_mutate)
    if parent_is_subagent:
        child_may_mutate = child_may_mutate and parent_may_mutate
    return {
        "may_delegate": (child_depth_remaining > 0) and parent_may_delegate,
        "may_mutate": child_may_mutate,
        "may_fan_out": normalize_bool(may_fan_out) and parent_may_fan_out,
        "depth_remaining": child_depth_remaining,
        "max_children": child_max_children,
        "intent_note": _normalized_intent_note(
            str(intent_note or "").strip() or str(parent_budget.get("intent_note") or "")
        ),
    }


def child_budget_for_schedule(
    parent_contract: Any,
    *,
    current_depth: int,
    new_depth: int,
    max_depth: int,
    may_mutate: bool,
    may_fan_out: bool,
    max_children: int,
    intent_note: str,
    requested_depth: Any = None,
) -> Dict[str, Any]:
    """Resolve a child's delegation_budget at schedule time (C3.1): decrement
    depth_remaining one generation (falling back to the configured max_depth/new_depth
    gap for legacy contracts), then NARROW within the parent. A ROOT scheduler
    (current_depth 0) honors its explicit may_mutate grant; a SUBAGENT scheduler's
    may_mutate gates the child (no read-only escalation)."""
    parent_budget = parent_contract.get("delegation_budget") if isinstance(parent_contract, dict) else {}
    parent_budget = parent_budget if isinstance(parent_budget, dict) else {}
    provenance = depth_provenance_for_schedule(
        parent_budget, new_depth=new_depth, max_depth=max_depth,
        use_remaining_envelope=True, requested_depth=requested_depth,
    )
    permitted_remaining = max(
        0, int(provenance.get("permitted_depth") or 0) - int(new_depth),
    )
    parent_depth_remaining = parent_budget.get("depth_remaining")
    child_depth_remaining = permitted_remaining
    if isinstance(parent_depth_remaining, int):
        child_depth_remaining = min(
            child_depth_remaining, max(0, parent_depth_remaining - 1),
        )
    child_budget = _narrow_child_delegation_budget(
        parent_budget,
        child_depth_remaining=child_depth_remaining,
        may_mutate=may_mutate,
        may_fan_out=may_fan_out,
        max_children=max_children,
        intent_note=intent_note,
        parent_is_subagent=current_depth > 0,
    )
    child_budget["depth_provenance"] = provenance
    return child_budget


def _ensure_project_scope(ctx: ToolContext, project_name: str = "", project_id: str = "") -> str:
    """Create (or attach to) a named Ouroboros PROJECT and scope THE CURRENT task to
    it — the in-task structural affordance for "create a project named X" once work
    is already running. promote_chat_to_task only creates a NEW task in a project;
    this binds the task you are ALREADY in (so you don't fall back to a bare mkdir).
    Idempotent for the same project; refuses to re-scope to a different one.
    Subagents inherit the parent's scope and cannot change it.
    """
    # delegation_role lives on the task metadata / contract lineage, NOT as a
    # ToolContext attribute — read it the canonical way.
    _meta = getattr(ctx, "task_metadata", {})
    _meta = _meta if isinstance(_meta, dict) else {}
    _contract = getattr(ctx, "task_contract", {})
    _contract = _contract if isinstance(_contract, dict) else {}
    _lineage = _contract.get("lineage", {}) if isinstance(_contract.get("lineage", {}), dict) else {}
    if str(_meta.get("delegation_role") or _lineage.get("delegation_role") or "").strip() == "subagent":
        return "⚠️ TOOL_ERROR (ensure_project_scope): subagents inherit the parent's project scope and cannot change it."
    from ouroboros.project_facts import (
        explicit_project_id_ok,
        project_id_from_display_name,
        sanitize_project_id,
    )
    from ouroboros.project_naming import clean_model_title

    # Run the agent-supplied name through the SAME lexical cleaner the proactive namer and
    # turn-into-project conversion use (project_naming SSOT) so every project-naming path
    # produces consistent titles (quote/emoji strip, length cap); fall back to the raw value.
    display_name = clean_model_title(project_name) or str(project_name or "").strip()
    explicit = str(project_id or "").strip()
    if explicit:
        if not explicit_project_id_ok(explicit):
            return (
                f"⚠️ TOOL_ARG_ERROR (ensure_project_scope): project_id {explicit!r} is not "
                "filesystem-clean; use lowercase alphanumeric/_/-/. (<=64 chars)"
            )
        pid = sanitize_project_id(explicit)
    elif display_name:
        pid = project_id_from_display_name(display_name)
    else:
        return "⚠️ TOOL_ARG_ERROR (ensure_project_scope): provide project_name (to create/name a project) or project_id (an existing one)."
    if not pid:
        return "⚠️ TOOL_ARG_ERROR (ensure_project_scope): could not derive a project id from the given name."

    current = sanitize_project_id(getattr(ctx, "project_id", "") or "")
    if current:
        if current == pid:
            return f"OK: this task is already scoped to project '{pid}' (no change)."
        return (
            f"⚠️ TOOL_ERROR (ensure_project_scope): this task is already scoped to project "
            f"'{current}'; it cannot be re-scoped to '{pid}'."
        )

    tid = str(getattr(ctx, "task_id", "") or "")
    # Scope the REST of this task immediately so journal_write and per-project
    # knowledge target the project now; the emitted event makes the supervisor
    # create the registry project, bind THIS task durably, and broadcast.
    ctx.project_id = pid
    evt = {
        "type": "ensure_project_scope",
        "task_id": tid,
        "project_id": pid,
        "project_name": display_name,
        "ts": utc_now_iso(),
    }
    # Lazy import avoids a control.py <-> control_delegation.py cycle (control is
    # fully loaded by the time any tool handler runs).
    from ouroboros.tools.control import _attach_origin_from_metadata, _emit_control_event

    # A direct-chat task carries its ingress-captured origin in task_metadata;
    # for a QUEUED task the supervisor-side handler falls back to the persisted
    # task record (workers._origin_from_task_record), so the durable bind keeps
    # the project's start-message identity on this mid-run scoping path too.
    _attach_origin_from_metadata(ctx, evt)

    mode = _emit_control_event(ctx, evt)
    return (
        f"OK: created/attached project '{display_name or pid}' (id={pid}) and scoped this "
        f"task into it ({mode}). journal_write and project knowledge now target this "
        "project; its live progress now routes to the project thread."
    )
