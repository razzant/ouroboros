"""Pure session task/request preparation shared by first-send sizing and dispatch.

Route capabilities are negotiated by the caller. This module never starts,
waits, collects, recovers or cancels an operation.
"""
from __future__ import annotations

import json
from typing import Any, Dict

from ouroboros.deadline_utils import bounded_seconds
from ouroboros.triad_review import default_output_contract, review_output_shape


def render_review_session_prompt(request: Any, slot: Any, task: str) -> str:
    """The session wrapper, shared by source preparation's exact first-send measurement."""
    contract = str((request.policy or {}).get("output_contract") or "")
    contract = contract or default_output_contract(review_output_shape(request.surface))
    return "\n".join((
        "You are an independent Ouroboros reviewer slot running as a read-only agent session.",
        f"Surface: {request.surface}", f"Role hint: {slot.role_hint or 'general reviewer'}", "", task, "",
        "OUTPUT CONTRACT (your host parses this structurally):",
        contract + "\nThis contract governs the unwrapped substantive deliverable; emit any host-required transport metadata outside it exactly as separately instructed.",
        f"Slot: {slot.slot_id}",
    ))


def prepare_review_session_request(invocation: Any, route: Any, *,
                                   prompt: str, root: str, thread_id: str,
                                   schema_asked: bool, ephemeral: bool = False) -> Dict[str, Any]:
    """Prepare and measure the exact wire request; no start, wait or recovery effects."""
    from ouroboros.review_execution import ReviewRouteUnavailable, _CLAUDEXOR_MAX_SECONDS
    from ouroboros.subagents import delegated_run_shape
    shape = delegated_run_shape(False)
    seconds = bounded_seconds(invocation.timeout_sec, default=300, maximum=_CLAUDEXOR_MAX_SECONDS)
    instructions, use_thread, output_schema = invocation.instructions, invocation.use_thread, invocation.output_schema
    run_request = {
        "prompt": prompt,
        "instructions": instructions,
        "authPreference": "subscription",
        "mode": shape.mode,
        "access": shape.access,
        "scope": {"kind": "project", "root": root, **({"ephemeral": True} if ephemeral else {})},
        # A one-element explicit pool is the pin; primaryHarness is only preference.
        "harnesses": [route.route_id],
        "primaryHarness": route.route_id,
        "maxSeconds": seconds,
    }
    if use_thread:
        run_request["_use_thread"] = True
        run_request["_thread_id"] = thread_id
    if route.model:
        run_request["model"] = route.model
    if route.effort:
        run_request["effort"] = route.effort
    if use_thread or getattr(route, "profile_id", ""):
        run_request["credentialProfileId"] = getattr(route, "profile_id", "") or None
    if schema_asked:
        run_request["outputSchema"] = output_schema
    if invocation.source_delivery:
        from ouroboros.tools.review_brief_coupling import SESSION_INLINE_DIFF_CEILING_CHARS
        serialized = json.dumps(run_request, ensure_ascii=False)
        invocation.source_delivery.update(first_send_chars=len(serialized),
                                          first_send_bytes=len(serialized.encode("utf-8")),
                                          first_send_ceiling=SESSION_INLINE_DIFF_CEILING_CHARS)
        if len(serialized) >= SESSION_INLINE_DIFF_CEILING_CHARS:
            raise ReviewRouteUnavailable("Complete source-first request exceeds session send bound",
                                         code="degraded_source_unreachable")
    return run_request
