"""A helper that started without knowledge meets the knowledge index with its first delegation.

A delegated child or nanny starts without knowledge (``memory_view.ViewSpec.knowledge``; a
declared-input child carries no memory view at all). When it first delegates in its task (an
accepted ``schedule_subagent``, whose record ``ctx._last_scheduled_subagents`` holds, or a
``delegate_start`` of new work, not ``retry_of``, that was not refused) the host adds, once per
task, the global shelf's knowledge index and its room's to that call's result, rendered by
``context.knowledge_navigation_sections`` exactly as integrating contexts see them. A focus whose
start already held knowledge (Main, a root, a wake, Presence) gets nothing.

The index is part of that result's body, never a delivery note: the batch's measured projection
(``tool_result_delivery``) allots it with the other results and, when the frame cannot hold it,
shows head and tail with the exact retained source; its opening line names ``knowledge_list``,
which reads it again. The producer's own text stays the result's ``producer_text``, the clean
payload host readers parse. Nothing already sent is rewritten. The flag lives on the task context:
a new worker after a restart may add it once more, and compaction may later fold it like any result.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List

log = logging.getLogger(__name__)

_DELIVERED_ATTR = "_knowledge_index_delivered"
_OPENING = "Knowledge index, added once with your first delegation (you started without it); knowledge_list reads it again."


def _accepted_delegation(ctx: Any, row: Dict[str, Any]) -> bool:
    """This call delegated, by typed facts: a refusal never counts, a replay is not new work."""
    from ouroboros.nanny_pacing import BASELINE_RESET_TOOLS  # the acts of delegation; D03->D07 is lazy-only

    name = row.get("fn_name")
    if name not in BASELINE_RESET_TOOLS or row.get("is_error") is not False:
        return False
    if name == "schedule_subagent":  # an accepted request leaves its record; a refused one leaves none
        return bool(getattr(ctx, "_last_scheduled_subagents", None))
    return not str((row.get("tool_args") or {}).get("retry_of") or "").strip()


def _start_held_knowledge(ctx: Any) -> bool:
    """Whether this task's first request held knowledge; without a captured start, nothing is known missing."""
    core = getattr(getattr(ctx, "context_fit_plan", None), "core", None)
    if core is None:
        return True
    if not core.memory_view_json:  # a declared-input child: no memory view, no knowledge
        return False
    from ouroboros.memory_view import snapshot_from_json

    return bool(snapshot_from_json(core.memory_view_json).spec.knowledge)


def _with_index(row: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """The row with the index trailing its result; the producer's own text stays its retained payload."""
    from ouroboros.context import knowledge_navigation_sections
    from ouroboros.project_facts import placed_project_id
    from ouroboros.tool_access import canonical_data_root
    from ouroboros.tools.tool_result import ToolResult

    sections = knowledge_navigation_sections(canonical_data_root(ctx), str(placed_project_id(ctx) or ""))
    text = "\n\n".join([str(row["result"]), _OPENING, *sections])
    typed = row.get("tool_result")
    if not isinstance(typed, ToolResult):
        return {**row, "result": text}
    clean = str(row["result"]) if typed.producer_text is None else typed.producer_text
    return {**row, "result": text, "tool_result": dataclasses.replace(typed, text=text, producer_text=clean)}


def with_knowledge_index(ctx: Any, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """``results`` with the index on the batch's first accepted delegation, once per task."""
    if ctx is None or getattr(ctx, _DELIVERED_ATTR, False):
        return results
    try:
        first = next((index for index, row in enumerate(results) if _accepted_delegation(ctx, row)), None)
        if first is None:
            return results
        setattr(ctx, _DELIVERED_ATTR, True)  # the first delegation decides, with an index or without one
        if _start_held_knowledge(ctx):
            return results
        rows = list(results)
        rows[first] = _with_index(rows[first], ctx)
        return rows
    except Exception:
        log.debug("knowledge index: none added", exc_info=True)
        return results
