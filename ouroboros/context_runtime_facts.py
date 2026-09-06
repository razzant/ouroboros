"""The runtime section's FACT builders: what the host can honestly say it knows.

Extracted whole from ``context.py`` at its module ceiling (v7 leaf) so the four
facts the runtime section renders keep one home: the project room a task sits in,
the budget rails it runs under, the toolset a promoted task materialized, and the
configured delegation route with its honestly-labeled historical observations.
Each returns a plain projection and reads no context state, so nothing here can
change what the section MEANS — only what it reports. ``context`` re-exports every
name, so historical imports and monkeypatch targets keep working unchanged.
"""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Any, Dict, List, Optional

from ouroboros.task_pacing import in_task_cost_ceiling_disclosure as _in_task_cost_ceiling

log = logging.getLogger(__name__)


def _project_room_fact(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The project-room working-folder FACT for a room turn, or None.

    Extracted verbatim from ``build_runtime_section`` (v6.90.x submarine unwind)
    to keep that builder under the hard method gate; the resolution and the
    stated rule are unchanged.
    """
    # v6.58.0 (2.2): a conversation/decision turn in a project ROOM sees the room's
    # working folder as a structural FACT — it can promote work into that folder
    # without ITSELF becoming a workspace task (decision turns deliberately keep the
    # promote/steer/route toolset, which workspace profiles exclude). The default
    # transport: promote_chat_to_task from this room inherits working_dir unless
    # workspace='none'. Registry read is anchored at the canonical DATA_DIR.
    # v6.61.3 room lens: the rule now states the REAL chat-lane affordances (reads +
    # default shell cwd resolve to the folder; writes go through promoted tasks) —
    # the robot-room incident was exactly a fact/affordance split. A set-but-broken
    # working_dir is disclosed loudly instead of a silent system-repo fallback.
    try:
        _room_pid = str(task.get("project_id") or "").strip()
        if _room_pid and not str(task.get("workspace_root") or "").strip():
            from ouroboros.config import DATA_DIR as _DATA_DIR
            from ouroboros.projects_registry import get_project as _get_project
            from ouroboros.workspace_admission import room_chat_lens_dir as _room_lens

            _room = _get_project(_DATA_DIR, _room_pid) or {}
            _room_wd = str(_room.get("working_dir") or "").strip()
            if _room_wd:
                # Same resolver the agent uses for the tool lens, so the stated rule
                # and the actual tool surface cannot diverge (the robot incident).
                _lens_dir, _room_note = _room_lens(_DATA_DIR, _room_pid)
                _lens_active = bool(task.get("_is_direct_chat")) and bool(_lens_dir)
                fact = {
                    "project_id": _room_pid,
                    "working_dir": _room_wd,
                    "rule": (
                        (
                            "This room's chat lane LOOKS AT the project folder: read_file/"
                            "list_files/search_code/query_code with root=active_workspace and "
                            "the DEFAULT shell cwd resolve to working_dir. The Ouroboros "
                            "system repo needs explicit root=\"system_repo\" (reads) or an "
                            "explicit cwd (shell). File WRITES here go through "
                            "promote_chat_to_task — the promoted task inherits this folder as "
                            "its workspace (workspace='none' opts out)."
                        )
                        if _lens_active
                        else (
                            "This project has a working folder. Tasks promoted from this room "
                            "run with it as their active workspace by default; pass "
                            "workspace='none' to promote a folder-less task."
                        )
                    ),
                }
                if _room_note:
                    fact["working_dir_warning"] = _room_note
                return fact
    except Exception:
        log.debug("Failed to inject project_room working_dir fact", exc_info=True)
    return None


def _runtime_budget_info(env: Any, task: Dict[str, Any], ctx: Any = None) -> Dict[str, Any]:
    """Start-of-task budget block: global projection + the STATIC per-task tree cap,
    written once at task start so the cached prefix stays byte-stable (DEVELOPMENT
    cache_friendliness item 22); live tree spend rides only the cache-breaking
    surfaces (checkpoint/pacing/milestones)."""
    try:
        from ouroboros.usage_accounting import usage_projection
        from ouroboros.settings_setup_contract import resolve_total_budget_usd

        total_usd = resolve_total_budget_usd()
        budget_root = pathlib.Path(task.get("budget_drive_root") or env.drive_root)
        projection = usage_projection(budget_root, global_limit_usd=total_usd)
        spent_usd = float(projection.get("accounted_usd") or 0.0)
        budget_info = {
            "status": "available" if total_usd is not None else "no_global_limit", "total_usd": total_usd,
            "spent_usd": spent_usd, "remaining_usd": None if total_usd is None else total_usd - spent_usd,
            "reserved_usd": float(projection.get("reserved_usd") or 0.0),
            "unresolved_upper_bound_usd": float(projection.get("unresolved_upper_bound_usd") or 0.0),
            "unknown_unmetered": int(projection.get("unknown_unmetered") or 0),
        }
    except Exception:
        log.error("Budget authority unavailable for runtime context", exc_info=True)
        budget_info = {"status": "unavailable"}
    try:
        root_cap = float(os.environ.get("OUROBOROS_PER_TASK_COST_USD", "0") or 0)
    except (TypeError, ValueError):
        root_cap = 0.0
    if root_cap > 0:
        budget_info["per_task_tree_cap_usd"] = root_cap
        budget_info["per_task_tree_cap_rule"] = (
            "Hard cap for THIS task's WHOLE tree (own model calls + all subagents), enforced "
            "by the physical-attempt ledger: dispatches are refused once the tree's accounted "
            "spend reaches it and the task is force-stopped. Budget checkpoints during the task report the live tree number."
        )
    if ctx is not None:
        budget_info["in_task_cost_ceiling"] = _in_task_cost_ceiling(ctx, budget_info.get("remaining_usd"))
    return budget_info


def _promoted_task_toolset(env: Any) -> Dict[str, Any]:
    """The LIVE built-in toolset available to an ordinary promoted task.

    Workspace focus changes the default target, not the top-level principal's
    tool names. The projection therefore asks the real registry once and keeps
    credential omissions typed instead of maintaining a second static catalog.
    Dynamic extension/MCP availability remains task-time state.
    """
    from types import SimpleNamespace

    from ouroboros.tools.registry import ToolRegistry, _builtin_tool_availability

    registry = ToolRegistry(pathlib.Path(env.repo_dir), pathlib.Path(getattr(env, "drive_root", ".")))

    probe = SimpleNamespace(
        task_id="promote_toolset_probe",
        task_metadata={},
        task_contract={},
        task_constraint=None,
        is_workspace_mode=lambda: False,
        is_ephemeral_turn=False,
    )
    registry.set_context(probe)
    top_level_tools = set(registry.available_tools())
    # Typed omissions: registered built-ins that live availability removes right
    # now (credential gates). Named with their reason so the router can tell
    # "does not exist" from "exists but currently unavailable".
    unavailable = {}
    for name in registry._entries:
        available, reason, detail = _builtin_tool_availability(name, probe)
        if not available:
            unavailable[name] = f"{reason}: {detail}" if detail else reason
    return {
        "top_level_tools": sorted(top_level_tools),
        **({"unavailable_builtin_tools": dict(sorted(unavailable.items()))} if unavailable else {}),
        "rule": (
            "LIVE built-in tool availability, evaluated by the real tool "
            "registry at promote time. Project focus changes the default root, "
            "not this ordinary top-level toolset. unavailable_builtin_tools "
            "exist but are currently unusable (e.g. missing credentials) — do "
            "not demand them. Dynamic extension/MCP tools are NOT listed (their "
            "availability is unknowable at promote time). If an objective/"
            "expected_output demands specific BUILT-IN tools, demand only names "
            "listed here."
        ),
    }


def _delegation_capability_fact() -> Optional[Dict[str, Any]]:
    """B4-lite: honestly-labeled HISTORICAL delegation observations.

    Deliberately NOT live health — receipts prove what the last execution did,
    not what a lane can do now; live lane facts arrive from plan-review wave
    rows and typed delegate refusals. Pure bounded file reads over the existing
    receipt projections: no daemon probes, no new health authority. Absent
    receipt files mean absent observations, never "healthy". Fail-soft on its
    own (None on any failure) so a problem here never drops the surrounding
    capabilities digest.
    """
    try:
        from ouroboros.reviewer_slot_config import reviewer_slot_last_executions
        from ouroboros.subagents import subagent_last_delegation

        def _observed_label(ts: Any) -> str:
            # Timestamp only: the verbatim "historical, not live health" disclaimer
            # lives ONCE in the note below, never repeated per row.
            return f"last observed at {str(ts or '').strip() or 'unknown time'}"

        delegation: Dict[str, Any] = {
            "note": (
                "Every row here is historical, not live health (the last "
                "recorded execution per reviewer slot / delegated run): "
                "live lane facts arrive from plan-review wave rows and typed "
                "delegate refusals. A missing row means no observation on "
                "record — never healthy."
            ),
        }
        slot_rows: List[Dict[str, Any]] = []
        for slot_id, row in sorted(reviewer_slot_last_executions().items()):
            if not isinstance(row, dict):
                continue
            status = str(row.get("status") or "").strip()
            fact: Dict[str, Any] = {
                "slot": str(slot_id),
                "outcome": (("ok" if status == "ok" else "failed") if status
                            else "unknown"),
                "observed": _observed_label(row.get("ts")),
            }
            requested = row.get("requested") if isinstance(row.get("requested"), dict) else {}
            effective = row.get("effective") if isinstance(row.get("effective"), dict) else {}
            if requested.get("profile_id"):
                fact["requested_profile"] = str(requested["profile_id"])
            if effective.get("profile_id"):
                fact["applied_profile"] = str(effective["profile_id"])
            # B1's typed failure facts, forwarded only when recorded (a dated
            # window carries reset_at without a code and an undated one the
            # code without a reset — read both independently).
            for key in ("failure_code", "reset_at"):
                if row.get(key):
                    fact[key] = row[key]
            slot_rows.append(fact)
        if slot_rows:
            delegation["reviewer_slots_last"] = slot_rows
        last = subagent_last_delegation()
        if isinstance(last, dict) and last:
            last_fact = {
                "route": str(last.get("route") or ""),
                "requested_model": str(last.get("requested_model") or ""),
                "applied_model": str(last.get("applied_model") or ""),
                "observed": _observed_label(last.get("ts")),
            }
            if last.get("requested_profile"):
                last_fact["requested_profile"] = str(last["requested_profile"])
            if last.get("applied_profile"):
                last_fact["applied_profile"] = str(last["applied_profile"])
            if last.get("selected_subagent_id"):
                last_fact["selected_subagent_id"] = str(last["selected_subagent_id"])
            delegation["subagent_last_delegation"] = last_fact
        if len(delegation) == 1:
            return None
        return delegation
    except Exception:
        log.debug("Failed to build delegation capability fact", exc_info=True)
        return None
