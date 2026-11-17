"""A server's refusal that states its window becomes the route's window before the round retries.

The refused round's plan was sized without that window (unknown, or a larger one). The refusal's typed fact
(``size_refusal``, structured fields first) is written as the route's refusal observation in the one window store
(``context_window.record_observation``; its order of sources is unchanged), and the plan is rebound on the same
route, which re-reads that store and narrows the shown view the way a start with that window would (Nano for an
80k window). The round's size recovery then retries that view first (``REBOUND_VIEW_STEP``), in the same turn.

The observation is scoped like every other on this route (``context_window.observation_scope``): an
OpenAI-compatible route's scope is its provider, address and model, so every role and helper that sends to
that address with that model reads it, and no other route does. Claudexor rebinds on its own receipts and the
built-in local model refuses before sending with its own counts; neither comes here. A refusal naming no number,
or a number without units (Alibaba's "Range of input length should be [1, N]"), teaches no window here: the
existing passes size those.
"""
from __future__ import annotations

from typing import Any


def _nano_tools(ctx: Any) -> list:
    """The schema list a request rebound into Nano sends: this actor's base (standing work set or live-call selectors),
    plus every tool this task's history already called and every tool it loaded (``enable_tools``), where current
    policy permits it, so each call in history keeps its schema and a loaded tool stays loaded (``continuation_tools``
    keeps them the same way). Nothing else of the old envelope stays."""
    from ouroboros.continuation_tools import history_tool_names
    from ouroboros.tool_policy import initial_tool_schemas, nano_standing

    loaded = set(getattr(ctx.tools._ctx, "_actor_loaded_tool_names", ()) or ())
    names = sorted(history_tool_names(ctx.messages) | loaded)
    return initial_tool_schemas(ctx.tools, context_mode="nano", schema_names=names,
                                standing=nano_standing(ctx.tools._ctx))


def learn_window_from_refusal(ctx: Any, loop: Any) -> bool:
    """Record the refusal's stated window on the route and rebind the plan; True when the shown view changed.

    ``ctx`` is the round's model-call context, ``loop`` the loop facade (``_rebind_context_fit_plan``)."""
    from ouroboros.context_window import observation_scope, record_refusal
    from ouroboros.size_refusal import for_usage, stamped

    usage, plan = ctx.accumulated_usage, ctx.context_fit_plan
    if stamped(usage) is None and not (usage.get("_last_llm_provider_answer") or {}).get("size"):
        return False  # legacy words size the ordinary passes, not this attempt's early rebind
    refusal = for_usage(usage)
    if (refusal is None or not refusal.token_window or plan is None or ctx.active_use_local
            or getattr(plan, "provider", "") == "claudexor" or not getattr(plan, "window_witness_bound", True)):
        return False
    record = getattr(plan, "window_record", None)
    boundary = getattr(record, "boundary", None) if record is not None else None
    if boundary is not None and boundary <= refusal.token_window:
        return False  # the plan already stands at or under this window: the passes size what is left
    scope = observation_scope(plan)
    if not scope:
        return False
    record_refusal(usage, plan)
    shown, mode = (ctx.messages[0] if ctx.messages else None), ctx.active_context_mode
    ctx.context_fit_plan, ctx.active_context_mode = loop._rebind_context_fit_plan(
        plan, ctx.tools, ctx.messages, model=ctx.active_model, use_local=ctx.active_use_local, start_mode=mode,
        tool_schemas=ctx.tool_schemas, model_role=ctx.model_role or getattr(plan, "model_role", "") or "main")
    if ctx.active_context_mode == "nano" and mode != "nano":
        # At first contact the same set a start in Nano sends; mid-task also the schemas its history called.
        from ouroboros.transcript_prefix import rewrite_view

        ctx.tool_schemas[:] = _nano_tools(ctx)
        rewrite_view(ctx.tools._ctx, ctx.task_id, "size_refusal_nano_tools", scope="tools")
    return bool(ctx.messages) and ctx.messages[0] != shown
