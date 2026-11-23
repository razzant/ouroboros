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
Route schema trials also price a destination's permitted tool list and ceiling notice before selecting its view.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any


@dataclass
class RouteSchemas:
    """One possible send list and its own ceiling notice; no trial mutates the task slot."""

    schemas: list
    notice: list
    left_out: tuple = ()

    def price(self):
        return self.schemas, self.notice

    def adopt(self, slot, resident, messages, *, by):
        from ouroboros.transcript_prefix import rewrite_view

        if resident != self.schemas:
            resident[:] = self.schemas
            rewrite_view(slot, getattr(slot, "task_id", ""), by, scope="tools")
        if self.left_out:
            slot._route_left_out_tool_names = _earlier(slot) | set(self.left_out)
        messages.extend(self.notice)


def _earlier(slot):
    return set(getattr(slot, "_route_left_out_tool_names", ()) or ())


def fit_route_schemas(tools, messages, schemas, model, limit, *, loaded, history):
    from ouroboros.context_fit import extract_plain_text_from_content
    from ouroboros.tool_policy import fit_tool_schemas_to_limit, route_tool_limit_notice, usable_selectors

    fitted, left_out = fit_tool_schemas_to_limit(schemas, limit, keep=loaded, protect=history)
    if not left_out:
        return RouteSchemas(fitted, [])
    text = route_tool_limit_notice(model, limit, len(schemas), left_out,
                                  selectors=usable_selectors(tools, fitted), schemas=schemas)
    notice = [] if any(text in extract_plain_text_from_content(row.get("content"))
                       for row in messages if row.get("role") == "user") else [{"role": "user", "content": text}]
    return RouteSchemas(fitted, notice, left_out)


def route_schema_trials(tools, messages, resident, mode, model, use_local, *, reselect=False):
    """Resolve each tried mode from current authority and the destination's schema cap, once per selection.

    A full return rebuilds what Nano did not carry; both frames keep still-permitted history and explicit loads.
    A same-route memory pass keeps its shown list except when trying Nano. An absent envelope stays absent.
    """
    if not resident:
        return None
    from ouroboros.continuation_tools import history_tool_names
    from ouroboros.provider_models import tool_schema_limit
    from ouroboros.tool_policy import initial_tool_schemas, nano_standing

    slot = tools._ctx
    history = history_tool_names(messages)
    loaded = set(getattr(slot, "_actor_loaded_tool_names", ()) or ())
    earlier, limit = _earlier(slot), tool_schema_limit(model, use_local=use_local)
    resident_names = {s["function"]["name"] for s in resident}

    @cache
    def selected(nano):
        schemas = resident
        if (reselect or nano and mode != "nano") and callable(getattr(tools, "schemas", None)):
            names = history | loaded | (resident_names if mode == "nano" else set())
            if not nano:
                names |= resident_names | earlier  # includes the permitted on-demand schemas already resident
            schemas = initial_tool_schemas(tools, context_mode="nano" if nano else "max",
                                           schema_names=sorted(names), standing=nano_standing(slot))
        return fit_route_schemas(tools, messages, schemas, model, limit, loaded=loaded, history=history)

    return lambda target: selected(target == "nano")


def fit_route_tool_ceiling(ctx):
    """An ordinary round may receive an enlarged resident catalog from the preceding tool batch."""
    from ouroboros.provider_models import tool_schema_limit

    schemas, limit = ctx.tool_schemas, tool_schema_limit(ctx.active_model, use_local=ctx.active_use_local)
    if limit is None or schemas is None or len(schemas) <= limit:
        return False
    trial = route_schema_trials(ctx.tools, ctx.messages, schemas, ctx.active_context_mode,
                                ctx.active_model, ctx.active_use_local)(ctx.active_context_mode)
    trial.adopt(ctx.tools._ctx, schemas, ctx.messages, by="route_tool_ceiling")
    return True


def learn_window_from_refusal(ctx: Any, loop: Any, candidate=None, round_id: str = "") -> bool:
    """Record the refusal's window and rebind; a mode change may change schemas without changing the system row.

    ``ctx`` is the round's model-call context, ``loop`` the loop facade (``_rebind_context_fit_plan``)."""
    from ouroboros.context_window import observation_scope, observed_plan, planned_chat, record_refusal
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
    if observed_plan(plan, usage, candidate) is None:
        return False  # the current round still reads the refusal, but the planned window cannot learn from this send
    recorded = (record_refusal(usage, plan, candidate=candidate, round_id=round_id) if candidate is not None
                else record_refusal(usage, plan))
    if planned_chat(candidate) and not recorded:
        return False
    shown, mode = (ctx.messages[0] if ctx.messages else None), ctx.active_context_mode
    ctx.context_fit_plan, ctx.active_context_mode = loop._rebind_context_fit_plan(
        plan, ctx.tools, ctx.messages, model=ctx.active_model, use_local=ctx.active_use_local, start_mode=mode,
        tool_schemas=ctx.tool_schemas, model_role=ctx.model_role or getattr(plan, "model_role", "") or "main")
    return ctx.active_context_mode != mode or (bool(ctx.messages) and ctx.messages[0] != shown)
