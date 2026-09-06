"""Task-tree coordination tools: tree_note / tree_read (the swarm blackboard + typed
child->parent beacons). Extracted from control.py for module size; storage lives in
``ouroboros.task_tree_ledger`` and is scoped by ``root_task_id`` (the whole task tree)."""

from __future__ import annotations

from typing import Any, Dict, List

from ouroboros.tools.registry import ToolContext, ToolEntry


def tree_root_id(ctx: ToolContext) -> str:
    """Resolve the task-tree root (root_task_id), falling back to this task's own id (the
    root has no parent). Scopes the coordination ledger to the WHOLE swarm/tree."""
    md = getattr(ctx, "task_metadata", {})
    rid = str(md.get("root_task_id") or "").strip() if isinstance(md, dict) else ""
    return rid or str(getattr(ctx, "task_id", "") or "").strip()


def _tree_note(
    ctx: ToolContext,
    kind: str,
    text: str,
    needs_parent_attention: bool = False,
    payload: Dict[str, Any] | None = None,
) -> str:
    if isinstance(payload, dict) and str(payload.get("type") or "") == "child_result_disposition":
        if str(kind or "").strip().lower() != "decision":
            return (
                "⚠️ CHILD_RESULT_DISPOSITION_INVALID: child_result_disposition payloads "
                "require kind='decision'."
            )
        # The join ledger is the sole lineage/hash/write authority. Invalid tagged
        # payloads return here and are never downgraded into ordinary text notes.
        # A `children` array is the batch form: one call, one shared rationale,
        # expanded into the same per-child authoritative rows as the single form.
        if "children" in payload:
            from ouroboros.tools.join_ledger import _record_child_result_disposition_batch

            return _record_child_result_disposition_batch(ctx, payload, text)
        from ouroboros.tools.join_ledger import _record_child_result_disposition

        return _record_child_result_disposition(ctx, payload, text)

    from ouroboros.task_tree_ledger import tree_ledger_append

    md = getattr(ctx, "task_metadata", {})
    role = str(md.get("role") or md.get("subagent_role") or "") if isinstance(md, dict) else ""
    return tree_ledger_append(
        tree_root_id(ctx),
        kind,
        text,
        task_id=str(getattr(ctx, "task_id", "") or ""),
        role=role,
        needs_parent_attention=bool(needs_parent_attention),
        payload=payload if isinstance(payload, dict) else None,
    )


def _tree_read(
    ctx: ToolContext,
    limit: int = 40,
    offset: int = 0,
    snapshot: str = "",
) -> str:
    from ouroboros.task_tree_ledger import tree_ledger_page
    from ouroboros.tool_access import canonical_data_root

    rid = tree_root_id(ctx)
    if not rid:
        return "⚠️ TOOL_ARG_ERROR (tree_read): no task-tree scope."
    page = tree_ledger_page(
        rid,
        limit=limit,
        offset=offset,
        snapshot=snapshot,
        data_root=canonical_data_root(ctx),
    )
    if not page:
        return f"(task-tree coordination ledger [{rid}] is empty)"
    if page.startswith("TREE_READ_SNAPSHOT_CHANGED"):
        return page
    return f"## Task-tree coordination ledger ({rid})\n\n{page}"


def get_tools() -> List[ToolEntry]:
    from ouroboros.task_tree_ledger import (
        CHILD_RESULT_DISPOSITIONS, DELEGATION_CONSTRAINT_DIRECTIVES, LEDGER_KINDS,
    )

    # The schema is where the model learns WHEN to choose each value, so the
    # per-value consequence belongs here rather than in prose elsewhere. Read
    # from the validator's own frozenset, sorted so the enum order is stable
    # across worker processes (a frozenset's iteration order is not).
    disposition_schema = {
        "type": "string",
        "enum": sorted(CHILD_RESULT_DISPOSITIONS),
        "description": (
            "integrated = consumed by this answer (item closes); irrelevant = not needed "
            "here (the child result stays as evidence; item closes); deferred = STILL OWED: "
            "your terminal answer reads degraded/best_effort until you re-disposition this "
            "exact child_result_sha256."
        ),
    }

    return [
        ToolEntry("tree_note", {
            "name": "tree_note",
            "description": (
                "Append a coordination entry to the SHARED task-tree ledger (the swarm "
                "blackboard, scoped to root_task_id — visible to the parent and all "
                "siblings/descendants of THIS task tree). Use it to publish the shared "
                "frame BEFORE fanning out interdependent children and to coordinate while "
                "they run. kind: contract|decision|fact|note (coordination) or "
                "milestone|partial_finding|blocker|question|interface_contract|review_requested|"
                "delegation_constraint (child->parent beacon). blocker/question/interface_contract/"
                "review_requested/delegation_constraint (or "
                "needs_parent_attention=true) surface an early return "
                "in the parent's wait. Domain-agnostic: 'contract' = code APIs OR "
                "presentation section-ownership OR a research claim schema — the seam for "
                "THIS task. Keep entries short; bulk detail belongs in artifacts."
            ),
            "parameters": {"type": "object", "required": ["kind", "text"], "properties": {
                "kind": {"type": "string", "enum": list(LEDGER_KINDS)},
                "text": {"type": "string", "description": "Short coordination text (<=4000 chars)."},
                "needs_parent_attention": {"type": "boolean", "default": False, "description": "Force a parent early-wait return (implied by blocker/question/interface_contract/review_requested)."},
                "payload": {
                    "type": "object",
                    "description": (
                        "Structured payload. For review_requested, provide exactly evidence_ref "
                        "and evidence_sha256; this records a non-blocking exact-evidence request "
                        "and never starts a reviewer. If the parent launches a check, it carries "
                        "that hash into the existing accounting/dedup path. Required for "
                        "delegation_constraint. For a parent "
                        "decision about a direct child result, set type=child_result_disposition, "
                        "child_task_id, disposition, and the exact SHA-256 shown by child evidence; "
                        "tree_note text is the rationale. To disposition MANY children in one call, "
                        "set type=child_result_disposition and a children array of "
                        "{child_task_id, disposition, child_result_sha256} entries (one shared "
                        "rationale; each entry is validated exactly like the single form)."
                    ),
                    "properties": {
                        "type": {"type": "string", "enum": ["child_result_disposition"]},
                        "child_task_id": {"type": "string"},
                        "disposition": disposition_schema,
                        "child_result_sha256": {"type": "string"},
                        "children": {
                            "type": "array",
                            "description": (
                                "Batch child dispositions: entries of {child_task_id, "
                                "disposition, child_result_sha256}."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "child_task_id": {"type": "string"},
                                    "disposition": disposition_schema,
                                    "child_result_sha256": {"type": "string"},
                                },
                                "required": [
                                    "child_task_id", "disposition", "child_result_sha256",
                                ],
                            },
                        },
                        "constraint_id": {"type": "string"},
                        "evidence_ref": {"type": "string"},
                        "evidence_sha256": {"type": "string"},
                        "directive": {"type": "string", "enum": list(DELEGATION_CONSTRAINT_DIRECTIVES)},
                        "scope": {},
                        "rationale": {"type": "string"},
                        "created_by": {"type": "string"},
                    },
                },
            }},
        }, lambda ctx, kind, text, needs_parent_attention=False, payload=None: _tree_note(ctx, kind, text, needs_parent_attention, payload), timeout_sec=15),
        ToolEntry("tree_read", {
            "name": "tree_read",
            "description": (
                "Read the tail of the shared task-tree coordination ledger (newest last) — "
                "the shared frame, decisions, facts, and sibling beacons for THIS task tree."
            ),
            "parameters": {"type": "object", "properties": {
                "limit": {"type": "integer", "default": 40, "description": "Max entries (<=200)."},
                "offset": {
                    "type": "integer",
                    "default": 0,
                    "description": "Number of newer entries already consumed.",
                },
                "snapshot": {
                    "type": "string",
                    "default": "",
                    "description": "Stable cursor returned by the preceding page.",
                },
            }},
        }, lambda ctx, limit=40, offset=0, snapshot="": _tree_read(
            ctx, limit, offset, snapshot,
        ), timeout_sec=15),
    ]


__all__ = ["get_tools", "tree_root_id"]
