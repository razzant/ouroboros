"""Dispatch-time executor notes for delegated children.

The child-facing halves of the executor axis, moved WHOLE from ``agent.py`` at
its module-size ceiling (B1/F7): the substrate note a dispatched child reads
(``dispatch_executor_note``) and the typed terminal of a pin no route can honor
(``executor_blocked_outcome``) — one pair, one vocabulary, both speaking about
the same ``SubagentExecutorResolution``. ``agent`` re-exports both under their
historical names, so every existing import and monkeypatch target keeps working
(the byte-pinned transport suite imports them from ``ouroboros.agent``).
"""

from __future__ import annotations

import json  # noqa: F401 -- historical import surface; the binding rows are D07 ledger territory (MIGRATION rows 3937-3938)
from typing import Any, Dict, Optional, Tuple  # noqa: F401 -- historical import surface; the binding rows are D07 ledger territory (MIGRATION rows 3937-3938)

from ouroboros.subagents import SubagentExecutorResolution, SubagentLaneResolution  # noqa: F401 -- historical import surface; the binding rows are D07 ledger territory (MIGRATION rows 3937-3938)


def _nanny_route_dispatched_for(task: Dict[str, Any], dispatch: Any) -> bool:
    """The loop's dispatched-onto-the-substrate fact, one place (charter D4).

    A configured agent_session row counts as dispatched even when the executor
    resolution reads "blocked": for a blocked start that is moot (the task
    terminals unrun), but a mid-run failure must keep the reminders/nudges/chip
    alive on the wake loops (owner 2026-08-28)."""
    snapshot = task.get("configured_subagent") if isinstance(
        task.get("configured_subagent"), dict) else {}
    route = snapshot.get("route") if isinstance(snapshot.get("route"), dict) else {}
    return bool(
        str(route.get("kind") or "") == "agent_session"
        or (
            dispatch is not None
            and dispatch.executor_resolution is not None
            and dispatch.executor_resolution.executor == "harness"
        )
    )


def _fill_executor_blocked_caps(ctx: Any, cap_info: Dict[str, Any], dispatch: Any) -> None:
    """Project the $0-unrun terminal facts onto cap_info (charter D2).

    Two producers, one projection: a dispatch-time blocked pin, or the host's
    pre-start of a configured session leaf DEFINITELY refused before the first
    model round (typed refusal, no custody handle). Callers gate on an empty
    startup wake — a non-empty wake means a fence/receipt episode owns the
    facts instead."""
    if dispatch is not None and dispatch.blocked:
        res = dispatch.executor_resolution
        cap_info["executor_blocked_reason"] = str(
            (res.reason if res is not None else "")
            or dispatch.delta.reason or "harness_not_configured"
        )
        cap_info["executor_blocked_requested"] = str(res.requested if res is not None else "harness")
        cap_info["executor_blocked_reset_at"] = str(res.reset_at if res is not None else "")
        return
    refusal = getattr(ctx, "_configured_startup_refusal", None)
    if isinstance(refusal, dict):
        cap_info["executor_blocked_reason"] = str(
            refusal.get("reason") or "configured_session_unavailable"
        )
        cap_info["executor_blocked_requested"] = str(refusal.get("requested") or "harness")
        cap_info["executor_blocked_reset_at"] = str(refusal.get("reset_at") or "")

# The v7 adoption keeps ONE home for the dispatch-note pair (MIGRATION rows
# 3935-3936): the bodies live in ouroboros/agent_dispatch.py; this module keeps
# the historical bindings. The rest of this module is D07-owned and untouched.
from ouroboros.agent_dispatch import (  # noqa: E402, F401 -- intentional public re-exports
    dispatch_executor_note,
    executor_blocked_outcome,
)
