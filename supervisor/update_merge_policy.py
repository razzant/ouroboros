"""Presentation-only labels and text for managed-update conflicts.

Git decides whether a merge is clean. Every conflict, regardless of pathname,
goes through the same reviewed assisted resolver. The doc/code/hot split only
helps the resolver and UI describe the plan; it grants or blocks nothing.
``assisted_objective`` renders the resolver task's objective text — presentation
only as well; the authority lives in the tx marker and its fingerprint.
"""

from __future__ import annotations

import posixpath
from typing import Any, Dict, List

DOCUMENT_EXACT = frozenset({"README.md"})
DOCUMENT_PREFIXES = ("docs/",)
HOT_CODE_PATHS = frozenset({
    "ouroboros/loop.py",
    "ouroboros/size_ratchet_manifest.py",
    "ouroboros/tools/control.py",
    "ouroboros/tools/registry.py",
    # v7 D04 split leaves: code that merely moved out of the hot registry
    # keeps the label (parity rule pinned by tests/test_lc2_owner_facades.py
    # for the inverse direction).
    "ouroboros/tools/registry_guard_process.py",
    "ouroboros/tools/registry_guards.py",
    # F3.1 typed-organ leaves: the registry class body and the typed result
    # vocabulary keep the hot-code label at their new homes (same parity).
    # extension_dispatch received the extension/MCP dispatch bodies that lived
    # on the hot ToolRegistry class — the label follows the moved body.
    "ouroboros/tools/extension_dispatch.py",
    "ouroboros/tools/registry_core.py",
    "ouroboros/tools/tool_catalog.py",
    "ouroboros/tools/tool_context.py",
    "ouroboros/tools/tool_resolution.py",
    "ouroboros/tools/tool_result.py",
    "ouroboros/config.py",
    "supervisor/queue.py",
    "supervisor/events.py",
    # v7 D08 split leaves: code that moved out of the hot control/queue/events
    # monoliths keeps the label (same parity rule as the D04 block above).
    "ouroboros/tools/control_events.py",
    "ouroboros/tools/control_routing.py",
    "ouroboros/tools/control_runtime.py",
    # v7 D07 split leaves: the rest of the hot control monolith moved out with
    # the D07 lane and keeps the label (same parity rule as the D04 block).
    "ouroboros/tools/control_scheduling.py",
    "ouroboros/tools/control_subagent_spec.py",
    "ouroboros/tools/control_task_results.py",
    "supervisor/queue_schedules.py",
    "supervisor/events_budget.py",
    "supervisor/events_chat_delivery.py",
    "supervisor/events_coop_checkpoint.py",
    "supervisor/events_project_routing.py",
    "supervisor/events_runtime_controls.py",
    "supervisor/events_schedule_task.py",
    "supervisor/events_subagent_admission.py",
    "supervisor/events_worker_reports.py",
    # F2.2 cancel/custody organ leaves (same parity rule), plus the
    # queue_transitions home of the owner-stop closure that moved out of the
    # hot events monolith.
    "supervisor/events_evolution_done.py",
    "supervisor/events_task_done.py",
    "supervisor/queue_snapshot.py",
    "supervisor/queue_timeouts.py",
    "supervisor/queue_transitions.py",
    "supervisor/worker_assignment.py",
    "supervisor/worker_health.py",
    # v7 D01 split leaves: code that moved out of the hot loop monolith keeps
    # the label (same parity rule as the D04 block above).
    "ouroboros/loop_acceptance.py",
    "ouroboros/loop_acceptance_review.py",
    "ouroboros/loop_budget.py",
    "ouroboros/loop_delivery.py",
    "ouroboros/loop_forced_finalization.py",
    "ouroboros/loop_messages.py",
    "ouroboros/loop_model_call.py",
    "ouroboros/loop_nudges.py",
    "ouroboros/loop_round_limits.py",
})


def _norm(path: str) -> str:
    normalized = posixpath.normpath(str(path or "").replace("\\", "/"))
    return normalized[2:] if normalized.startswith("./") else normalized.lstrip("/")


def is_document_path(path: str) -> bool:
    p = _norm(path)
    if p in DOCUMENT_EXACT or posixpath.basename(p).upper().startswith("CHANGELOG"):
        return True
    return p.endswith(".md") and any(p.startswith(prefix) for prefix in DOCUMENT_PREFIXES)


def is_hot_code(path: str) -> bool:
    return _norm(path) in HOT_CODE_PATHS


def classify_conflicts(conflict_paths: List[str]) -> Dict[str, object]:
    """Return one route plus presentation labels; filenames never set policy."""
    paths = [str(path).strip() for path in (conflict_paths or []) if str(path).strip()]
    docs = [path for path in paths if is_document_path(path)]
    code = [path for path in paths if path not in docs]
    return {
        "kind": "conflicting" if paths else "clean",
        "doc_conflict_paths": docs,
        "code_conflict_paths": code,
        "hot_code_paths": [path for path in code if is_hot_code(path)],
    }


def rescue_pointer_note(tx: Dict[str, Any]) -> str:
    """One plain sentence pointing the resolver at rescued uncommitted work.

    Reads the latest rescue pointer (``progress_rescue``, falling back to
    ``rollback_rescue``); when several rescues were taken, only the latest is
    named plus a count — no history rendering. Returns "" when there is nothing
    to point at."""
    pointer = tx.get("progress_rescue") or tx.get("rollback_rescue")
    if not isinstance(pointer, dict) or not pointer.get("path"):
        return ""
    count = int(pointer.get("count") or 1)
    tally = f" ({count} rescues were taken; this is the latest)" if count > 1 else ""
    return (
        f" A previous attempt's uncommitted work was rescued to {pointer['path']}{tally}; "
        "changes.diff there is a plain diff against the reviewed base. Read the rescued "
        "files to re-apply prior resolutions — do not run git commands."
    )


def carrier_guidance(conflicts: List[str]) -> str:
    """Version-carrier guidance for the resolver (owner decisions Q8/Q24): the landed
    update carries the TARGET's version; prose and history stay the fork's own.

    The carrier inventory is the span SSOT (``release_sync.CARRIER_SPAN_PATHS``,
    imported at call time — presentation only, no policy). Conflicts confined to
    declared spans are resolved mechanically before the resolver task is built
    (supervisor/update_carriers.py), so a carrier still in *conflicts* DEGRADED
    to manual resolution and the guidance names what that means."""
    from ouroboros.tools.release_sync import CARRIER_SPAN_PATHS

    if not any(_norm(path) in CARRIER_SPAN_PATHS for path in conflicts):
        return ""
    return (
        " Version carriers: the update lands under the official target's version — VERSION is "
        "already projected, every NON-conflicted carrier token (pyproject.toml, "
        "web/package.json, the README badge, the docs/ARCHITECTURE.md header, install pages) is "
        "already synced mechanically, and carrier conflicts confined to declared version spans "
        "were already resolved to the target's side. A carrier still in your list degraded to "
        "manual resolution (a broken or duplicate span anchor, or a conflict outside the spans): "
        "make its version tokens match VERSION exactly. In the README Version History table keep "
        "BOTH sides' rows (never delete this fork's local history rows); resolve prose conflicts "
        "on their merits."
    )


def assisted_objective(tx: Dict[str, Any]) -> str:
    """Objective text for the single authorized assisted-resolution task."""
    target = str(tx.get("target_sha") or "")[:12]
    conflicts = list(tx.get("conflict_paths") or [])
    if conflicts:
        work = (
            f"Resolve each conflicting file ({', '.join(conflicts)}), preserve both intents "
            "where possible, and remove every conflict marker (<<<<<<<, =======, >>>>>>>)."
        )
    else:
        work = (
            "The merge itself is clean, but it combines local and official history and therefore "
            "requires review. Inspect the staged combination and correct it if needed."
        )
    retry_note = ""
    if str(tx.get("failed_update_ref") or ""):
        retry_note = (
            f" A previous attempt at this same update is preserved on branch {tx['failed_update_ref']}; "
            "you may read files from it for reference, but resolve the staged merge in front of you."
        )
    return (
        f"A managed Ouroboros update (target {target}) has been merged into your working tree by the "
        "supervisor: MERGE_HEAD is set and the combined tree is staged for review. Do NOT run any git "
        "command (fetch/merge/commit/checkout are blocked) — the merge is already staged for you. "
        f"{work} Do not discard either side merely because a file is normally restricted. When ready, "
        "run `preflight_review` with the commit message, then `commit_reviewed` (it will create the reviewed "
        "2-parent merge commit), then `request_restart` to finish landing the update. "
        f"Terminal contract: if this task ends WITHOUT that reviewed merge commit landing (given up, "
        f"cancelled, or review not passed), the supervisor rolls the repository back to the pre-update "
        f"state; your resolution work is normally preserved (best-effort) on branch failed-update-{target} "
        "plus a rescue snapshot, and the owner can simply retry the update."
        f"{carrier_guidance(conflicts)}{retry_note}{rescue_pointer_note(tx)}"
    )
