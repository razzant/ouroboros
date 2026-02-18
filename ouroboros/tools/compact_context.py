"""
compact_context tool — selective LLM-driven context summarization.

When the LLM decides (via self-check or its own judgment) that old tool
results are bloating context, it calls this tool to selectively compress
message history. The LLM specifies WHICH parts to compress and which to keep.

This is LLM-first (Bible P3): the agent decides what's important,
not a hardcoded rule.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from ouroboros.tools.registry import ToolEntry

log = logging.getLogger(__name__)


def _compact_context(keep_last_n: int = 6, **kwargs) -> str:
    """
    Trigger selective compaction of tool history in the current conversation.

    The agent calls this when it notices context is getting large.
    The actual compaction happens in the loop via compact_tool_history()
    with the specified keep_last_n parameter.

    Args:
        keep_last_n: Number of recent tool rounds to keep intact (default 6).
                     Older rounds get their results summarized to 1-line summaries.
                     Set higher to preserve more recent context, lower to compress more.

    Returns:
        Confirmation message with compaction settings applied.
    """
    ctx = kwargs.get("_ctx")
    if ctx is None:
        return "⚠️ No context available"

    # Validate range
    keep_last_n = max(2, min(keep_last_n, 20))

    # Store the compaction request on context for the loop to pick up
    ctx._pending_compaction = keep_last_n

    return (
        f"✅ Context compaction scheduled: keeping last {keep_last_n} tool rounds intact, "
        f"older rounds will be summarized to 1-line summaries. "
        f"This will take effect on the next round."
    )


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            name="compact_context",
            schema={
                "name": "compact_context",
                "description": (
                    "Selectively compress old tool results in conversation history to save context tokens. "
                    "Call this when you notice context is getting large (e.g., after self-check reminder). "
                    "Keeps recent N tool rounds intact; older rounds get summarized to 1-line summaries. "
                    "You decide what to keep (via keep_last_n) — no information is lost, just compressed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keep_last_n": {
                            "type": "integer",
                            "description": "Number of recent tool rounds to keep fully intact (default 6, range 2-20). Lower = more compression.",
                            "default": 6,
                        },
                    },
                    "required": [],
                },
            },
            handler=_compact_context,
            timeout_sec=5,
        ),
    ]
