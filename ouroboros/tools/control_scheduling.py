"""Scheduling one live subagent: what the parent asked for, and nothing else.

The request is assembled here — constraint selection for a read-only or acting
child, the child's drive, its narrowed contract and delegation budget, the
requested-status envelope persisted before emission — and then emitted. The
lane, the model, the effort, the route and the effective executor are resolved
once at dispatch, not here, so a queued child's record never names a resolution
nobody has made yet. What comes back to the parent is the request plus the
schedule-time facts it cannot see for itself: tree slot occupancy, the child's
predicted authority, and any host-minted shared cooperative tree.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from ouroboros.config import get_max_subagent_depth
from ouroboros.depth_evidence import parse_task_depth
from ouroboros.contracts.task_contract import (
    build_task_contract,
    effective_acceptance_claims,
    normalize_allowed_resources,
)
from ouroboros.headless import prepare_task_drive, task_state_dir
from ouroboros.subagent_runtime import (
    SubagentSelectionError,
    effective_runtime_subagent_settings,
    select_subagent_snapshot,
)
from ouroboros.subagents import (
    LEGACY_SUBAGENT_FIELDS,
    build_subagent_envelope,
)
from ouroboros.task_results import STATUS_REQUESTED, write_task_result
from ouroboros.tool_capabilities import ACTING_SUBAGENT_MODE, LOCAL_READONLY_SUBAGENT_MODE
from ouroboros.tools.control_delegation import (
    admitted_depth_cap,
    child_budget_for_schedule,
    normalize_required_capabilities,
    profile_from_task_constraint,
    record_depth_limit_refusal,
    resolve_cooperative_write_root,
    schedule_delegation_refusal,
)
from ouroboros.tools.control_events import _SCHEDULE_EMIT_LOCK, _emit_control_event
from ouroboros.tools.control_subagent_spec import (
    RETIRED_SCHEDULE_PARAMS,
    _INTERNAL_SCHEDULE_OPTIONS,
    _validated_schedule_fields,
    schedule_subagent_param_names,
)
from ouroboros.tools.registry import ToolContext, active_repo_dir_for, system_repo_dir_for
from ouroboros.utils import append_jsonl, utc_now_iso
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result


def _publish_scheduling_refusal(ctx: Any, status: str, code: str, text: str) -> str:
    """Publish one scheduling refusal at the branch that composed it (D02).

    The validator helpers stay pure functions with no invocation to publish
    into; the single caller that turns a refusal into the result of a call
    publishes it here, with the code the one adapter already assigns.
    """
    return _publish_tool_result(ctx, ToolResult(status=status, code=code, text=text))

log = logging.getLogger(__name__)


def _ctl():
    """The parent module, read at call time.

    The parent owns the rebindable module state and the members tests
    monkeypatch there; reading them through the module at each call keeps
    one binding, where a from-import would freeze the value this leaf saw
    at import time (the owner-approved D18/D33 mechanical exception).
    """
    from ouroboros.tools import control

    return control


def _record_scheduled_subagent(ctx: ToolContext, record: Dict[str, Any]) -> None:
    """Append a scheduled-subagent record to ctx under the emit lock.

    The read-copy-append-setattr of ``_last_scheduled_subagents`` is a lost-update
    race when a burst of schedule_subagent calls is emitted in parallel; the lock
    serializes it. (list.append is atomic under the GIL, but the surrounding RMW
    is not.)
    """
    with _SCHEDULE_EMIT_LOCK:
        scheduled_records = list(getattr(ctx, "_last_scheduled_subagents", []) or [])
        scheduled_records.append(record)
        setattr(ctx, "_last_scheduled_subagents", scheduled_records)


def _emit_swarm_fanout(
    ctx: ToolContext,
    *,
    parent_task_id: str,
    root_task_id: str,
    depth: int,
    task_group_id: str,
    task_ids: List[str],
    role: str,
    requested_model_lane: str,
    objective: str,
    emitted_live: bool,
) -> None:
    """Emit one durable swarm_fanout telemetry event per spawn wave (WS8).

    The name avoids task_/llm_/tool_ prefixes and the event sets no
    delegation_role/subagent_task_id, so the Logs UI never renders a phantom
    child card or folds it into a grouped-task lane (web/modules/log_events.js).
    inter_wave_latency_sec reuses ``_last_wave_ts`` under the emit lock (no new
    persistent state).
    """
    now = time.time()
    with _SCHEDULE_EMIT_LOCK:
        prev = float(getattr(ctx, "_last_wave_ts", 0.0) or 0.0)
        inter_wave = round(now - prev, 3) if prev > 0 else None
        setattr(ctx, "_last_wave_ts", now)
    evt = {
        "ts": utc_now_iso(),
        "type": "swarm_fanout",
        "task_id": parent_task_id,
        "parent_task_id": parent_task_id,
        "root_task_id": root_task_id,
        "depth": depth,
        "task_group_id": task_group_id,
        "requested_count": len(task_ids),
        "task_ids": task_ids,
        "role": role,
        # The REQUEST. What the children actually ran on is a per-child DISPATCH
        # fact and lives on each child's own record — a wave event written before
        # any child started cannot know it, and `effective_model_lanes` used to
        # claim it anyway.
        "requested_model_lane": requested_model_lane,
        "slot_count": len(task_ids),
        "objective_preview": objective[:200],
        "emitted_live": bool(emitted_live),
        "inter_wave_latency_sec": inter_wave,
    }
    try:
        append_jsonl(ctx.drive_logs() / "events.jsonl", evt)
    except Exception:
        log.debug("Failed to emit swarm_fanout telemetry", exc_info=True)


def _subagent_slot_note(ctx: ToolContext, root_task_id: str) -> str:
    """Compact slot-occupancy transparency for the schedule_subagent result (v6.54.3, 1.6).

    Read-only queue-snapshot facts — the LLM decides what to do with them (P5);
    nothing here gates admission (the supervisor stays authoritative). Counts are
    from the last persisted snapshot, i.e. BEFORE this wave lands."""
    try:
        status_root = Path(str(getattr(ctx, "budget_drive_root", "") or ctx.drive_root))
        snap = json.loads((status_root / "state" / "queue_snapshot.json").read_text(encoding="utf-8"))
    except Exception:
        return ""

    def _is_tree_subagent(row: Any) -> bool:
        if not isinstance(row, dict):
            return False
        task = row.get("task") if isinstance(row.get("task"), dict) else row
        return (
            str(task.get("delegation_role") or "") == "subagent"
            and str(task.get("root_task_id") or "") == str(root_task_id or "")
        )

    active = sum(1 for r in (snap.get("running") or []) if _is_tree_subagent(r))
    queued = sum(1 for r in (snap.get("pending") or []) if _is_tree_subagent(r))
    try:
        from ouroboros.config import get_max_active_subagents_per_root
        cap = int(get_max_active_subagents_per_root())
    except Exception:
        return ""
    tail = "; children beyond the active cap WAIT for a free slot" if active >= cap else ""
    return f" [tree slots before this wave: {active}/{cap} active, {queued} queued{tail}]"


def _capability_mismatch_message(selected_profile: str, missing_caps: Any) -> str:
    """v6.57.0 (1.6): name the CORRECT spawn so the parent fixes a capability mismatch
    in one move instead of burning a round guessing (the prober-without-shell incidents).
    shell/write/edit/service/vcs need an ACTING child (a write_surface); a read-only child
    has no shell and no writable roots."""
    if set(missing_caps) & {"shell", "write", "edit", "service", "vcs"}:
        hint = (
            "These need an ACTING child: pass write_surface (self_worktree for a throwaway "
            "checkout to run shell/build in; external_workspace for the shared project tree; "
            "genesis for a from-scratch project). A read-only child has no shell/writable roots."
        )
    else:
        hint = (
            "Adjust the child's profile/lane so the declared capabilities are available, "
            "or drop capabilities the child does not actually need."
        )
    return (
        "⚠️ SUBAGENT_CAPABILITY_MISMATCH: selected child profile "
        f"{selected_profile!r} cannot satisfy required_capabilities={missing_caps}. " + hint
    )


def _finalize_schedule_emission(ctx: ToolContext, emission: Dict[str, Any]) -> str:
    """Record the scheduled wave, emit swarm_fanout telemetry, and build the
    tool-result string. Extracted from _schedule_task to keep that function
    within the per-function size budget (P7). The emission facts ride ONE spec
    dict — the same idiom as ``_validated_schedule_fields`` — keeping the
    signature inside the <8-parameter contract. Keys: ``task_ids``,
    ``requested_model_lane``, ``objective``, ``role``, ``depth``,
    ``parent_task_id``, ``root_task_id``, ``emitted_modes``, plus optional
    ``write_surface`` and ``coop_shared_tree``.

    It reports the REQUEST and nothing else. Until v6.87.28 it also printed an
    `effective_lane=` and a `CAPABILITY_DELTA` line, both produced by resolving the
    child inside the scheduling call — an answer about live availability, given
    before the child was queued, let alone started. The reduction now reaches the
    parent where it can act on it: in `[SUBTASK_OUTCOME]`, when it reads the answer
    and decides how far to trust it. ``coop_shared_tree`` is the ONE exception by
    design: the host-minted shared coop tree is a SCHEDULE-TIME fact (the parent's
    effective write_root input, not a dispatch-time resolution), and withholding it
    forced every wave to rediscover its own tree by trial and error (the submarine
    waves' 'user_files path blocked' loop)."""
    task_ids = list(emission.get("task_ids") or [])
    requested_model_lane = str(emission.get("requested_model_lane") or "")
    objective = str(emission.get("objective") or "")
    role = str(emission.get("role") or "")
    depth = int(emission.get("depth") or 0)
    parent_task_id = str(emission.get("parent_task_id") or "")
    root_task_id = str(emission.get("root_task_id") or "")
    emitted_modes = list(emission.get("emitted_modes") or [])
    write_surface = str(emission.get("write_surface") or "")
    coop_shared_tree = str(emission.get("coop_shared_tree") or "")
    configured = emission.get("configured_subagent") if isinstance(
        emission.get("configured_subagent"), dict
    ) else {}
    selected_id = str(configured.get("selected_subagent_id") or "")
    selected_route = configured.get("route") if isinstance(configured.get("route"), dict) else {}
    route_kind = str(selected_route.get("kind") or "")
    legacy_selection = bool(emission.get("legacy_selection"))
    worker_note = " (live queue emission requested)" if any(m == "live" for m in emitted_modes) else ""
    try:
        _record_scheduled_subagent(ctx, {
            "task_ids": task_ids,
            "requested_model_lane": requested_model_lane,
            "objective": objective,
            "role": role,
        })
    except Exception:
        pass
    try:
        _emit_swarm_fanout(
            ctx,
            parent_task_id=parent_task_id,
            root_task_id=root_task_id,
            depth=depth,
            task_group_id="",
            task_ids=task_ids,
            role=role,
            requested_model_lane=requested_model_lane,
            objective=objective,
            emitted_live=any(m == "live" for m in emitted_modes),
        )
    except Exception:
        pass
    slot_note = _subagent_slot_note(ctx, root_task_id)
    # v6.57.0 (1.6): preview the child's EFFECTIVE tool profile (shell/writable) so
    # the parent knows up front whether the child can run shell / write — the wasted
    # rounds where a prober child hit workspace_blocked came from neither side
    # knowing. AUTHORITY only: it carries no lane, because the lane is not resolved
    # yet and a preview that guesses one is the claim this release removed.
    profile_note = ""
    try:
        from ouroboros.tool_access import predicted_subagent_profile, summarize_subagent_profile

        profile_note = "\n" + summarize_subagent_profile(
            predicted_subagent_profile(write_surface=write_surface))
    except Exception:
        pass
    coop_note = ""
    if str(coop_shared_tree or "").strip():
        try:
            import pathlib as _pl

            _tree = _pl.Path(coop_shared_tree)
            coop_note = (
                f"\nshared coop tree: {_tree} — children write there "
                "(write_surface=external_workspace); you read it via "
                f"root=subagent_projects, path={_tree.name!r}/…"
            )
        except Exception:
            coop_note = f"\nshared coop tree: {coop_shared_tree}"
    commitment = (
        "ordinary recursive API actor"
        if route_kind == "api_model"
        else "recursive nanny; exact route/work-order authority is frozen before its first model call"
    )
    legacy_note = (
        "\nDEPRECATED_LEGACY_SELECTOR: deterministically mapped to this exact configured row; "
        "future calls must pass subagent_id."
        if legacy_selection else ""
    )
    return (
        f"Subagent request queued {task_ids[0]}: {objective} "
        f"(subagent_id={selected_id}, route={route_kind}, {commitment})"
        f"{worker_note}{slot_note}{profile_note}{coop_note}{legacy_note}"
    )


def _build_acting_constraint(
    *,
    write_surface: str,
    write_root: str,
    protected_paths_grant: bool,
    external_tool_grants: Any,
    parent_workspace_root: str,
    ctx: Any = None,
):
    """Validate a mutative-subagent request; return its constraint dict, or an
    error string for the LLM (which can then fall back to a read-only subagent).

    The toggle/surface checks here give the caller immediate feedback. The
    supervisor is the authoritative gate and provisions the self_worktree
    (filling write_root/base_sha) before the child runs.

    ``ctx`` is the invocation this refusal belongs to, so the denial is published
    by the branch that made it rather than re-read from the bytes it printed. It
    is optional because the selector is also called directly, outside a tool
    invocation, where there is nothing to publish to.
    """
    from ouroboros.config import get_allow_mutative_subagents
    from ouroboros.contracts.task_constraint import VALID_WRITE_SURFACES

    if write_surface not in VALID_WRITE_SURFACES:
        allowed = ", ".join(sorted(VALID_WRITE_SURFACES))
        return (
            "⚠️ TOOL_ARG_ERROR (schedule_subagent): write_surface must be one of "
            f"{allowed} (or omit it for a read-only subagent)."
        )
    if not get_allow_mutative_subagents(write_surface):
        return _publish_tool_result(ctx, ToolResult(
            status="blocked", code="ACCESS_BLOCKED",
            text=(
            "⚠️ MUTATIVE_SUBAGENTS_DISABLED: acting children with "
            f"write_surface={write_surface!r} are disabled here. "
            "OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS is the master gate: an explicit owner "
            "true/false applies to every surface; when it is empty the runtime mode "
            "decides — advanced/pro allow every surface, light allows the external "
            "build surfaces (external_workspace, genesis — they write outside the "
            "Ouroboros runtime) and keeps self_worktree (a checkout of the live body) "
            "off. Schedule a read-only subagent (omit write_surface), use an external "
            "surface, or have the owner enable the toggle."
            ),
        ))
    grants: List[str] = []
    if isinstance(external_tool_grants, (list, tuple)):
        grants = [str(g).strip() for g in external_tool_grants if str(g).strip()]
    resolved_write_root = str(write_root or "").strip()
    if write_surface == "external_workspace" and not resolved_write_root:
        resolved_write_root = str(parent_workspace_root or "").strip()
    if write_surface == "external_workspace" and not resolved_write_root:
        return (
            "⚠️ TOOL_ARG_ERROR (schedule_subagent): write_surface=external_workspace "
            "requires write_root (the external project directory) or a parent workspace."
        )
    return {
        "mode": ACTING_SUBAGENT_MODE,
        "surface": write_surface,
        "write_root": resolved_write_root,
        "protected_paths_grant": protected_paths_grant,
        "external_tool_grants": grants,
        "parent_only_commit": True,
        "return_kind": "workspace_patch",
        "allow_enable": False,
        "allow_review": False,
    }


def _select_subagent_constraint(write_surface, write_root, protected_paths_grant, external_tool_grants, parent_workspace_root, caller_readonly=False, ctx=None):
    """Read-only default (no surface), a validated acting constraint, or an error string."""
    if not write_surface or str(write_surface).strip().lower() == "read_only":
        # `read_only` is the explicit, provider-safe alias for the omit-surface
        # read-only path (the handler also normalizes it; this guard keeps the selector
        # correct for any direct caller and matches the schema enum) — never acting.
        return {"mode": LOCAL_READONLY_SUBAGENT_MODE, "allow_enable": False, "allow_review": False}
    if caller_readonly:
        # A read-only subagent may delegate read-only children only — never spawn an acting one.
        return _publish_tool_result(ctx, ToolResult(
            status="blocked", code="ACCESS_BLOCKED",
            text=(
            "⚠️ MUTATIVE_SUBAGENTS_DISABLED: a read-only subagent cannot spawn a mutative (acting) "
            "child. Only the root agent, workspace tasks, or acting subagents may pass write_surface; "
            "schedule a read-only child instead."
            ),
        ))
    return _build_acting_constraint(
        write_surface=write_surface,
        write_root=write_root,
        protected_paths_grant=protected_paths_grant,
        external_tool_grants=external_tool_grants,
        parent_workspace_root=parent_workspace_root,
        ctx=ctx,
    )


def _schedule_parent_chat(ctx: Any) -> Any:
    """The scheduling parent's own chat, or None when it genuinely has none.

    Coercing a missing chat to 0 here would hand the consumer an explicit
    hidden-partition address the parent never had, and the consumer's fallback
    (owner chat) is the honest answer for a task with no address at all.
    """
    from ouroboros.contracts.chat_id_policy import HIDDEN_CHAT_ID
    from supervisor.message_bus import coerce_chat_identity

    value = getattr(ctx, "current_chat_id", None)
    return None if value is None else coerce_chat_identity(value, HIDDEN_CHAT_ID)


def _populate_subagent_event_extras(
    evt: Dict[str, Any], *, current_chat_id: Any, child_drive: Any, workspace_root: str,
    workspace_mode: str, executor_ref: Any, context: str, parent_task_id: str,
) -> None:
    """Add the optional fields of a schedule_subagent event in place (extracted from
    _schedule_task to keep it under the method gate; pure field assignment)."""
    # Membership, not truthiness (C4): 0 is the hidden partition — a real
    # address the consumer must not re-route — so only absence stays absent.
    if current_chat_id is not None:
        evt["chat_id"] = current_chat_id
    if child_drive is not None:
        evt["drive_root"] = str(child_drive)
        evt["child_drive_root"] = str(child_drive)
    if workspace_root:
        evt["workspace_root"] = workspace_root
    if workspace_mode:
        evt["workspace_mode"] = workspace_mode
    if executor_ref:
        evt["executor_ref"] = executor_ref
        evt["metadata"] = {**(evt.get("metadata") if isinstance(evt.get("metadata"), dict) else {}), "executor_ref": executor_ref}
    if context:
        evt["context"] = context
    if parent_task_id:
        evt["parent_task_id"] = parent_task_id


def _prepare_child_drive(tid, status_drive_root, memory_mode, parent_project_id):
    """Prepare the forked/empty child drive. On failure clean up the drive + the
    task-state dir and return ``(None, error_string)``; otherwise ``(drive, "")``.
    (Extracted from _schedule_task to keep it under the method gate.)"""
    if memory_mode not in {"forked", "empty"}:
        return None, ""
    try:
        return prepare_task_drive(status_drive_root, tid, memory_mode, project_id=parent_project_id), ""
    except Exception as exc:
        shutil.rmtree(task_state_dir(status_drive_root, tid), ignore_errors=True)
        log.warning("Failed to prepare child drive for subtask %s", tid, exc_info=True)
        return None, f"⚠️ SUBTASK_DRIVE_ERROR: failed to prepare {memory_mode} child drive: {exc}"


def _earliest_deadline_at(requested: str, inherited: str) -> str:
    """The tighter of two ISO deadlines (either may be empty/unparseable)."""
    from ouroboros.deadline_utils import parse_deadline_ts

    stamps = {text: parse_deadline_ts(text) for text in (requested, inherited) if text}
    usable = {text: ts for text, ts in stamps.items() if ts is not None}
    if not usable:
        return requested or inherited
    return min(usable, key=lambda text: usable[text])


def _build_child_subagent_contract(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Build a delegated child's task contract from a single spec mapping (extracted
    from _schedule_task to keep it under the method size gate; one dict param to stay
    within the parameter-count discipline; pure construction)."""
    parent_contract = spec.get("parent_contract")
    objective = spec.get("objective", "")
    expected_output = spec.get("expected_output", "")
    constraints = spec.get("constraints", "")
    delegation_budget = spec.get("child_delegation_budget")
    narrowed_deadline_at = _earliest_deadline_at(
        str(spec.get("deadline_at") or ""),
        str(parent_contract.get("deadline_at") or "") if isinstance(parent_contract, dict) else "",
    )
    # The child's claims come from the parent's EXPLICIT acceptance_claims param —
    # its ingress — through the one effective-claims seam; there is no plan wave at
    # dispatch. Re-stated below even when EMPTY: omitted means the child has none,
    # never "inherit the parent's" (the deadline_at spread lesson).
    child_claims, _claims_source = effective_acceptance_claims(
        {"acceptance_claims": spec.get("acceptance_claims")}
    )
    return build_task_contract({
        "id": spec.get("tid"),
        "type": "task",
        "description": objective,
        "objective": objective,
        "expected_output": expected_output,
        "constraints": constraints,
        "workspace_root": spec.get("workspace_root", ""),
        "workspace_mode": spec.get("workspace_mode", ""),
        "project_id": spec.get("parent_project_id", ""),
        "allowed_resources": spec.get("allowed_resources"),
        # A caller may bind the child to an EARLIER deadline than the parent's when the
        # parent can only consume the child's handoff inside a narrower window (planning
        # scouts). Never LATER: the earliest of the two wins, so a requested deadline can
        # only tighten the inherited one.
        "deadline_at": narrowed_deadline_at,
        "parent_task_id": spec.get("parent_task_id", ""),
        "root_task_id": spec.get("root_task_id"),
        "session_id": spec.get("session_id", ""),
        "delegation_role": "subagent",
        "metadata": {
            "task_contract": {
                **parent_contract,
                "source": "parent_delegation",
                "objective": objective,
                "expected_output": expected_output,
                "constraints": constraints,
                # The spread above hands the child EVERY parent field, and this merged
                # mapping outranks the task-level keys in build_task_contract. Any field we
                # deliberately narrow must therefore be re-stated after it, or the parent's
                # value silently wins back — which is exactly what used to happen to a
                # requested child deadline whenever the parent carried one of its own.
                "deadline_at": narrowed_deadline_at,
                "delegation_budget": delegation_budget,
                "resource_policy": spec.get("resource_policy", parent_contract.get("resource_policy", {})),
                "attachment_manifest": spec.get("attachment_manifest") or [],
                # Same lesson for the criteria carriers, re-stated even when EMPTY:
                # without these, the parent's claims/criteria leak into every child and
                # child verify receipts would "support" claims the child never owned.
                "acceptance_claims": child_claims,
                "success_criteria": [],
            } if isinstance(parent_contract, dict) else {
                "delegation_budget": delegation_budget,
                "resource_policy": spec.get("resource_policy", {}),
                "acceptance_claims": child_claims,
                "attachment_manifest": spec.get("attachment_manifest") or [],
            },
        },
    })


def _resolve_executor_ref(ctx: Any) -> dict:
    """The child's workspace executor reference (docker/host), or {} when unavailable."""
    accessor = getattr(ctx, "workspace_executor_ref", None)
    if callable(accessor):
        try:
            candidate = accessor()
            if isinstance(candidate, dict) and candidate:
                return dict(candidate)
        except Exception:
            return {}
    return {}


def _inherited_workspace_from_active_repo(
    ctx: ToolContext, workspace_root: str, workspace_mode: str
) -> tuple[str, str]:
    """Inherit an external active workspace for readonly children when metadata is absent."""
    if workspace_root:
        return workspace_root, workspace_mode
    try:
        active = active_repo_dir_for(ctx).resolve(strict=False)
        system = system_repo_dir_for(ctx).resolve(strict=False)
        if active != system:
            return str(active), workspace_mode or "external"
    except Exception:
        pass
    return workspace_root, workspace_mode


def _schedule_task(ctx: ToolContext, internal: Dict[str, Any] | None = None, /, **params: Any) -> str:
    allowed_params = schedule_subagent_param_names() | HIDDEN_LEGACY_SCHEDULE_PARAMS
    retired = sorted(str(key) for key in params if key in RETIRED_SCHEDULE_PARAMS)
    if retired:
        return _publish_scheduling_refusal(
            ctx, "error", "TOOL_ARG_ERROR", "⚠️ TOOL_ARG_ERROR (schedule_subagent): " + " ".join(
                f"{name} was withdrawn: {LEGACY_SUBAGENT_FIELDS[RETIRED_SCHEDULE_PARAMS[name]]}. "
                "Drop it — the owner's configured effort applies, exactly as it did when "
                f"{name} was omitted." for name in retired))
    unsupported = sorted(str(key) for key in params if key not in allowed_params)
    if unsupported:
        bad = ", ".join(unsupported)
        return _publish_scheduling_refusal(
            ctx, "error", "TOOL_ARG_ERROR", "⚠️ TOOL_ARG_ERROR (schedule_subagent): unsupported argument(s): "
            f"{bad}. Use the strict schema: subagent_id, objective, expected_output, "
            "optional role/context/constraints/memory_mode and (for mutative children) "
            "write_surface/write_root/protected_paths_grant/external_tool_grants.")
    internal = dict(internal or {})
    if set(internal) - _INTERNAL_SCHEDULE_OPTIONS:
        raise TypeError(f"_schedule_task: unknown internal scheduling option(s): "
                        f"{sorted(set(internal) - _INTERNAL_SCHEDULE_OPTIONS)}")
    fields, arg_error = _validated_schedule_fields(params, ctx=ctx)
    if arg_error:
        return _publish_scheduling_refusal(ctx, "error", "TOOL_ARG_ERROR", arg_error)
    deadline_at = fields["deadline_at"]
    objective = fields["objective"]
    expected_output = fields["expected_output"]
    role = fields["role"]
    context = fields["context"]
    constraints = fields["constraints"]
    memory_mode = fields["memory_mode"]
    may_mutate = fields["may_mutate"]
    try:
        configured_subagent, legacy_selection = select_subagent_snapshot(
            effective_runtime_subagent_settings(_ctl().load_settings()),
            subagent_id=str(params.get("subagent_id") or ""),
            legacy_model_lane=params.get("model_lane"),
            legacy_executor=params.get("executor"),
            legacy_model_lane_supplied="model_lane" in params,
            legacy_executor_supplied="executor" in params,
        )
    except SubagentSelectionError as exc:
        return f"⚠️ {exc.code}: {exc.detail}"
    route = configured_subagent.get("route") if isinstance(configured_subagent.get("route"), dict) else {}
    requested_model_lane = "auto"  # bounded historical projection only
    requested_executor = "harness" if route.get("kind") == "agent_session" else "native"

    current_depth, depth_error = _context_task_depth(ctx)
    if depth_error:
        return (
            "⚠️ TOOL_ERROR (schedule_subagent): invalid_task_depth: "
            f"{depth_error}"
        )
    new_depth = current_depth + 1
    metadata = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
    parent_contract = fields["parent_contract"]
    max_depth = admitted_depth_cap(parent_contract, get_max_subagent_depth())
    if new_depth > max_depth:
        return record_depth_limit_refusal(
            ctx, fields, params, configured_subagent,
            current_depth=current_depth, new_depth=new_depth, max_depth=max_depth,
        )

    if getattr(ctx, 'is_direct_chat', False):
        from ouroboros.utils import append_jsonl
        try:
            append_jsonl(ctx.drive_logs() / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "schedule_task_from_direct_chat",
                "description": objective[:200],
                "warning": "schedule_subagent called from direct chat context — potential duplicate work",
            })
        except Exception:
            pass
    current_task_id = str(getattr(ctx, "task_id", "") or "")
    parent_task_id = str(current_task_id or metadata.get("parent_task_id") or "").strip()
    root_task_id_seed = str(metadata.get("root_task_id") or current_task_id or "").strip()
    session_id = str(metadata.get("session_id") or "")
    # Absence stays absence: this stamps the child's address, and 0 is a real
    # destination now rather than a synonym for "unset".
    current_chat_id = _schedule_parent_chat(ctx)
    budget_drive_root = str(metadata.get("budget_drive_root") or getattr(ctx, "budget_drive_root", "") or ctx.drive_root)
    status_drive_root, root_cost_ceiling_usd = Path(budget_drive_root), getattr(getattr(ctx, "_cost_ceiling", None), "ceiling_usd", None)
    if refusal := schedule_delegation_refusal(parent_contract, status_drive_root, parent_task_id):
        return refusal
    workspace_root = str(getattr(ctx, "workspace_root", "") or metadata.get("workspace_root") or "").strip()
    workspace_mode = str(getattr(ctx, "workspace_mode", "") or metadata.get("workspace_mode") or "").strip()
    workspace_root, workspace_mode = _inherited_workspace_from_active_repo(ctx, workspace_root, workspace_mode)
    parent_project_id = str(getattr(ctx, "project_id", "") or "").strip()
    requested_surface = str(params.get("write_surface") or "").strip().lower()
    # `read_only` is a first-class, provider-safe alias for "omit write_surface" (NOT a
    # VALID_WRITE_SURFACES acting surface) — normalize it to the read-only path so
    # constraint selection, mutating detection, and the event all treat it as read-only (P5).
    if requested_surface == "read_only":
        requested_surface = ""
    if requested_surface:
        from ouroboros.presence_authority import presence_ceiling_allows_delegated_surface

        if not presence_ceiling_allows_delegated_surface(ctx, requested_surface):
            return (
                "⚠️ PRESENCE_DELEGATION_BLOCKED: mutative delegation requires an exact "
                "selected write root for this surface in the inherited capability ceiling."
            )
    # FR2: a flat parent requesting external_workspace with no write_root builds
    # cooperatively in ONE host-minted shared tree (helper extracted to keep this
    # method under the size gate).
    effective_write_root, caller_profile, coop_err = resolve_cooperative_write_root(
        ctx, requested_surface, params.get("write_root", ""), workspace_root, metadata)
    if coop_err:
        return coop_err
    task_constraint = _select_subagent_constraint(
        requested_surface, effective_write_root, params.get("protected_paths_grant", False),
        params.get("external_tool_grants"), workspace_root,
        caller_readonly=(caller_profile == "local_readonly_subagent"), ctx=ctx)
    if isinstance(task_constraint, str):
        return task_constraint
    from ouroboros.tool_access import subagent_profile_satisfies

    required_caps, cap_error = normalize_required_capabilities(params.get("required_capabilities"))
    if cap_error:
        return _publish_scheduling_refusal(ctx, "error", "TOOL_ARG_ERROR", f"⚠️ TOOL_ARG_ERROR (schedule_subagent): {cap_error}")
    selected_profile = profile_from_task_constraint(task_constraint)
    ok, missing_caps = subagent_profile_satisfies(selected_profile, required_caps)
    if not ok:
        # An argument error: both inputs and the remedy are THIS call's arguments.
        return _publish_scheduling_refusal(ctx, "error", "TOOL_ARG_ERROR", _capability_mismatch_message(selected_profile, missing_caps))
    allowed_resources = normalize_allowed_resources(
        (parent_contract.get("allowed_resources") if isinstance(parent_contract, dict) else {})
        or metadata.get("allowed_resources")
        or {}
    )
    executor_ref = _resolve_executor_ref(ctx)
    # SCHEDULING STATES INTENT AND NOTHING ELSE. The lane, the model, the effort, the
    # route, the profile and the effective executor are all resolved ONCE, at
    # dispatch, by `subagents.resolve_subagent_dispatch` — see it for why. What is
    # recorded here is what the parent ASKED for, plus the parent's own lane, which
    # is the fact an omitted lane inherits and which only the parent knows.
    tid = uuid.uuid4().hex[:8]
    task_ids: List[str] = [tid]
    root_task_id = root_task_id_seed or tid
    parent_model_lane = str(metadata.get("effective_model_lane") or "")
    parent_cognitive_route = {
        "model": str(getattr(ctx, "active_model", "") or metadata.get("model") or ""),
        "effort": str(getattr(ctx, "active_effort", "") or metadata.get("reasoning_effort") or ""),
        "use_local_model": bool(getattr(ctx, "active_use_local", metadata.get("use_local_model", False))),
    }
    child_drive, _drive_err = _prepare_child_drive(
        tid, status_drive_root, memory_mode, parent_project_id)
    if _drive_err:
        return _publish_scheduling_refusal(ctx, "error", "TOOL_ERROR", _drive_err)
    child_attachment_manifest, attachment_error = _materialize_child_attachment_manifest(
        parent_contract, child_drive or status_drive_root, tid,
        owner_drive=Path(ctx.drive_root), owner_task_id=parent_task_id,
    )
    if attachment_error:
        shutil.rmtree(task_state_dir(status_drive_root, tid), ignore_errors=True)
        return f"⚠️ SUBTASK_ATTACHMENT_ERROR: {attachment_error}"

    # C3.1: propagate and narrow the parent's typed delegation intent.
    child_delegation_budget = child_budget_for_schedule(
        parent_contract,
        current_depth=current_depth, new_depth=new_depth, max_depth=max_depth,
        may_mutate=may_mutate, may_fan_out=params.get("may_fan_out", True),
        max_children=params.get("max_children", 0),
        intent_note=params.get("delegation_intent", ""),
        requested_depth=params.get("requested_depth", 0),
    )

    child_contract = _build_child_subagent_contract({
        "tid": tid, "objective": objective, "expected_output": expected_output, "constraints": constraints,
        "workspace_root": workspace_root, "workspace_mode": workspace_mode, "parent_project_id": parent_project_id,
        "allowed_resources": allowed_resources, "parent_contract": parent_contract,
        "parent_task_id": parent_task_id, "root_task_id": root_task_id, "session_id": session_id,
        "child_delegation_budget": child_delegation_budget, "deadline_at": str(deadline_at or ""),
        "acceptance_claims": fields["acceptance_claims"], "resource_policy": fields["resource_policy"],
        "attachment_manifest": child_attachment_manifest,
    })
    # The requested-status envelope carries the REQUEST. Its derived half stays
    # empty until dispatch fills it, so a queued child's public description never
    # names a lane, a model or an effort that no resolution has produced.
    envelope = build_subagent_envelope(
        task_id=tid,
        parent_task_id=parent_task_id,
        root_task_id=root_task_id,
        depth=new_depth,
        role=role,
        requested_lane=requested_model_lane,
        executor=requested_executor,
        status=STATUS_REQUESTED,
    )
    intent_fields = {
        "model_lane": requested_model_lane,
        "requested_model_lane": requested_model_lane,
        "parent_model_lane": parent_model_lane,
        "requested_executor": requested_executor,
        "configured_subagent": configured_subagent,
        "parent_cognitive_route": parent_cognitive_route,
    }
    evt = {
        "type": "schedule_subagent",
        "description": objective,
        "objective": objective,
        "expected_output": expected_output,
        "constraints": constraints,
        "role": role,
        "task_id": tid,
        "depth": new_depth,
        "ts": utc_now_iso(),
        "root_task_id": root_task_id,
        "session_id": session_id,
        "actor_id": f"subagent:{role}",
        "delegation_role": "subagent",
        "memory_mode": memory_mode,
        "project_id": parent_project_id,
        "budget_drive_root": budget_drive_root,
        "root_cost_ceiling_usd": root_cost_ceiling_usd,
        "task_constraint": task_constraint,
        "write_surface": requested_surface,
        "task_contract": child_contract,
        "allowed_resources": allowed_resources,
        "required_capabilities": required_caps,
        **intent_fields,
        "subagent_envelope": envelope,
    }
    _populate_subagent_event_extras(
        evt, current_chat_id=current_chat_id, child_drive=child_drive,
        workspace_root=workspace_root, workspace_mode=workspace_mode,
        executor_ref=executor_ref, context=context, parent_task_id=parent_task_id,
    )
    try:
        write_task_result(
            status_drive_root,
            tid,
            STATUS_REQUESTED,
            parent_task_id=parent_task_id or None,
            root_task_id=root_task_id,
            session_id=session_id,
            actor_id=f"subagent:{role}",
            delegation_role="subagent",
            project_id=parent_project_id,
            role=role,
            description=objective,
            objective=objective,
            expected_output=expected_output,
            constraints=constraints,
            context=context,
            workspace_root=workspace_root,
            workspace_mode=workspace_mode,
            executor_ref=executor_ref,
            allowed_resources=allowed_resources,
            task_contract=child_contract,
            required_capabilities=required_caps,
            chat_id=current_chat_id,
            memory_mode=memory_mode,
            drive_root=str(child_drive) if child_drive is not None else "",
            child_drive_root=str(child_drive) if child_drive is not None else "",
            budget_drive_root=budget_drive_root,
            root_cost_ceiling_usd=root_cost_ceiling_usd,
            task_constraint=task_constraint,
            **intent_fields,
            subagent_envelope=envelope,
            result="Subagent request queued. Awaiting supervisor acceptance.",
        )
    except Exception:
        log.warning("Failed to persist requested task status for %s", tid, exc_info=True)
        try:
            (status_drive_root / "task_results" / f"{tid}.json").unlink(missing_ok=True)
        except Exception:
            pass
        if child_drive is not None:
            shutil.rmtree(child_drive, ignore_errors=True)
        return _publish_scheduling_refusal(ctx, "error", "TOOL_ERROR", f"⚠️ SUBTASK_STATUS_ERROR: failed to persist requested status for {tid}; subagent was not scheduled.")

    emitted_modes: List[str] = [_emit_control_event(ctx, evt)]
    return _finalize_schedule_emission(ctx, {
        "task_ids": task_ids,
        "requested_model_lane": requested_model_lane,
        "objective": objective,
        "role": role,
        "depth": new_depth,
        "parent_task_id": parent_task_id,
        "root_task_id": root_task_id_seed or current_task_id,
        "emitted_modes": emitted_modes,
        "write_surface": requested_surface,
        "configured_subagent": configured_subagent,
        "legacy_selection": legacy_selection,
        # Host-minted shared coop tree only (a caller-supplied write_root is the
        # parent's own knowledge already).
        "coop_shared_tree": (
            effective_write_root
            if effective_write_root and effective_write_root != str(params.get("write_root", "") or "")
            else ""
        ),
    })


def _context_task_depth(ctx: ToolContext) -> tuple[int, str]:
    try:
        return parse_task_depth(getattr(ctx, "task_depth", 0), default=0), ""
    except (TypeError, ValueError) as exc:
        return 0, str(exc)


def _materialize_child_attachment_manifest(
    parent_contract: Dict[str, Any], target_root: Path, task_id: str,
    *, owner_drive: Optional[Path] = None, owner_task_id: str = "",
) -> tuple[list[dict], str]:
    """Copy inherited task inputs into the child's own artifact store."""

    initial = parent_contract.get("attachment_manifest") if parent_contract else []
    inherited = list(initial) if isinstance(initial, list) else []
    if owner_drive is not None and owner_task_id:
        from ouroboros.owner_mailbox import owner_attachment_manifest

        inherited.extend(owner_attachment_manifest(owner_drive, owner_task_id))
    if not inherited:
        return [], ""
    from ouroboros.artifacts import materialize_inherited_attachment_manifest

    return materialize_inherited_attachment_manifest(inherited, target_root, task_id)


def maybe_emit_delegated_run_fanout(ctx: ToolContext, *, run_id: str, route_id: str,
                                    objective: str, durable: bool) -> None:
    """swarm_fanout for a delegated harness run, only under host-attested Swarm intent.

    A task admitted through the Swarm button carries the typed metadata fact
    ``force_plan_source == "swarm"`` — the gate reads exactly that admission fact,
    never keywords or prompt text (P5). Only such a hosting task folds its
    delegate_start into swarm telemetry; an ordinary delegated run on a task that
    never asked for a swarm emits nothing. The event reuses the exact existing
    ``swarm_fanout`` wave shape (``requested_count=1``, ``role="delegated_run"``,
    requested lane = the selected session route) and means STARTED/REQUESTED, not
    completed. An uncustodied start (``durable=False``) is not attested and emits
    nothing. Telemetry must never break a start that already succeeded, so
    failures stay logged, never raised.
    """
    metadata = getattr(ctx, "task_metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    if not durable or metadata.get("force_plan_source") != "swarm":
        return
    task_id = str(getattr(ctx, "task_id", "") or "")
    try:
        depth = int(getattr(ctx, "task_depth", 0) or 0) + 1
    except (TypeError, ValueError):
        depth = 1
    try:
        _emit_swarm_fanout(
            ctx,
            parent_task_id=task_id,
            root_task_id=str(metadata.get("root_task_id") or task_id),
            depth=depth,
            task_group_id="",
            task_ids=[str(run_id or "")],
            role="delegated_run",
            requested_model_lane=str(route_id or ""),
            objective=str(objective or ""),
            emitted_live=True,
        )
    except Exception:
        log.debug("Failed to emit delegated-run swarm_fanout telemetry", exc_info=True)


HIDDEN_LEGACY_SCHEDULE_PARAMS: frozenset[str] = frozenset({"model_lane", "executor"})
