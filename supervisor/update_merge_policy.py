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
    "ouroboros/loop_forced_finalization.py",
    "ouroboros/loop_delivery.py",
    "ouroboros/loop_budget.py",
    "ouroboros/loop_model_call.py",
    "ouroboros/loop_nudges.py",
    "ouroboros/loop_round_limits.py",
    "ouroboros/loop_acceptance_review.py",
    "ouroboros/loop_acceptance.py",
    "ouroboros/loop_messages.py",
    "ouroboros/size_ratchet_manifest.py",
    "ouroboros/tool_module_inventory.py",
    "ouroboros/tools/control.py",
    "ouroboros/tools/control_events.py",
    "ouroboros/tools/control_routing.py",
    "ouroboros/tools/control_runtime.py",
    "ouroboros/tools/control_scheduling.py",
    "ouroboros/tools/control_subagent_spec.py",
    "ouroboros/tools/control_task_results.py",
    "ouroboros/tools/registry.py",
    "ouroboros/tools/registry_core.py",
    "ouroboros/tools/registry_guard_process.py",
    "ouroboros/tools/registry_guards.py",
    "ouroboros/tools/tool_resolution.py",
    "ouroboros/tools/extension_dispatch.py",
    "ouroboros/tools/tool_catalog.py",
    "ouroboros/tools/tool_context.py",
    "ouroboros/tools/tool_result.py",
    "ouroboros/config.py",
    "ouroboros/model_slots.py",
    "ouroboros/review_model_routes.py",
    "ouroboros/runtime_limits.py",
    "ouroboros/settings_defaults.py",
    "ouroboros/settings_scales.py",
    "supervisor/queue.py",
    "supervisor/queue_evolution.py",
    "supervisor/queue_schedules.py",
    "supervisor/queue_snapshot.py",
    "supervisor/queue_timeouts.py",
    "supervisor/event_taxonomy.py",
    "supervisor/events.py",
    "supervisor/events_budget.py",
    "supervisor/events_chat_delivery.py",
    "supervisor/events_coop_checkpoint.py",
    "supervisor/events_evolution_done.py",
    "supervisor/events_project_routing.py",
    "supervisor/events_runtime_controls.py",
    "supervisor/events_schedule_task.py",
    "supervisor/events_subagent_admission.py",
    "supervisor/events_task_done.py",
    "supervisor/events_worker_reports.py",
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
    return (
        f"A managed Ouroboros update (target {target}) has been merged into your working tree by the "
        "supervisor: MERGE_HEAD is set and the combined tree is staged for review. Do NOT run any git "
        "command (fetch/merge/commit/checkout are blocked) — the merge is already staged for you. "
        f"{work} Do not discard either side merely because a file is normally restricted. When ready, "
        "run `advisory_review` with the commit message, then `commit_reviewed` (it will create the reviewed "
        "2-parent merge commit), then `request_restart` to finish landing the update."
        f"{rescue_pointer_note(tx)}"
    )
