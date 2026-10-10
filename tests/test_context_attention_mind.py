"""Context attention, Lane B: dialogue-scope units with typed owner protection (3A), the
per-round facts line (5A), and the model-free rungs of the refusal ladder (7A).

Shares the serial Main fit fixtures of ``tests.test_loop_compaction``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import context_compaction as cc
from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
from ouroboros.context_source_view import _EMERGENCY_LABELS
from ouroboros.loop_messages import CONTEXT_FACTS_NAME
from tests.test_context_reclaim_materializer import _request
from tests.test_loop_compaction import _ctx, _failed_capture, _fit, _measure_cycle

OWNER = ("Owner: keep the old API alive until the migration is verified. " * 20).strip()
HOST = "[Context view receipt] status=applied " + "detail " * 400
ASSISTANT = "I concluded the migration is safe because both code paths are covered. " * 30
# _transcript() rows: 0 system, 1 assignment, 2-3 tool "one", 4 own prose, 5 owner words, 6 host prose, 7-8 tool "two"


def _tool(call_id="read", text="exact evidence " * 400):
    return [{"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "type": "function", "function": {
        "name": "read_file", "arguments": json.dumps({"path": f"{call_id}.py"})}}]},
        {"role": "tool", "tool_call_id": call_id, "content": text}]


def _transcript():
    return [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "Assignment: migrate the API."},
            *_tool("one"), {"role": "assistant", "content": ASSISTANT}, {"role": "user", "content": OWNER},
            {"role": "user", "content": HOST}, *_tool("two")]


def _kinds(messages, units):
    return [cc.unit_kind(messages, unit) for unit in units]


def test_dialogue_scope_adds_prose_units_under_the_same_positional_ids():
    messages = _transcript()
    tool_scope, dialogue = cc.context_units(messages), cc.context_units(messages, scope="dialogue")
    assert _kinds(messages, tool_scope) == ["tool", "tool"]
    assert _kinds(messages, dialogue) == ["tool", "assistant", "user", "user", "tool"]
    assert [unit.unit_id for unit in tool_scope] == [unit.unit_id for unit in dialogue if unit.unit_id in
                                                     {u.unit_id for u in tool_scope}]
    assert {unit.start for unit in dialogue}.isdisjoint({0, 1})  # the system view and the assignment are never units
    assert [unit.start for unit in dialogue if cc.unit_kind(messages, unit) != "tool"] == [4, 5, 6]


def test_unfinished_protocol_native_and_non_text_rows_stay_atomic():
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "go"},
                {"role": "assistant", "content": [{"type": "thinking", "thinking": "opaque"}]},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]},
                {"role": "user", "content": "typed", "acceptance_observation": {"kind": "x"}},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "open", "type": "function", "function": {
                    "name": "read_file", "arguments": "{}"}}]}]  # the active continuation: no result yet
    assert cc.context_units(messages, scope="dialogue") == ()


def test_typed_owner_words_protect_their_rows_and_role_user_alone_does_not():
    from ouroboros.tools.compact_context import owner_protected_texts

    messages = _transcript()
    units = cc.context_units(messages, scope="dialogue")
    directives = [{"role": "user", "content": OWNER}, {"role": "user", "content": [{"type": "text", "text": "Answer: B"}]},
                  {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:,"}}]}]
    texts = owner_protected_texts(SimpleNamespace(_owner_directives=directives))
    assert texts == (OWNER, "Answer: B") and owner_protected_texts(SimpleNamespace()) == ()
    protected = cc.owner_protected_unit_ids(messages, units, texts)
    assert [unit.start for unit in units if unit.unit_id in protected] == [5]  # not the host receipt at 6
    assert cc.owner_protected_unit_ids(messages, units, ()) == frozenset()
    mixed = _transcript() + [{"role": "user", "content": "[Quiz answered]\nAnswer: B\n(submitted from the web UI)"}]
    mixed_units = cc.context_units(mixed, scope="dialogue")
    protected = cc.owner_protected_unit_ids(mixed, mixed_units, ("Answer: B",))
    assert [unit.start for unit in mixed_units if unit.unit_id in protected] == [len(mixed) - 1]  # whole row


def test_authored_view_folds_own_prose_keeps_owner_words_and_restores_exactly(tmp_path):
    messages = _transcript()
    request = replace(_request(messages, 0), working_note="Migration verified; old API kept.",
                      expected_view_revision=cc.context_reclaim_transcript_sha256(messages), keep_unit_ids=())
    assistant_unit = next(u for u in cc.context_units(messages, scope="dialogue") if cc.unit_kind(messages, u) == "assistant")
    candidate, receipt, usage = cc.compact_tool_history_llm(
        messages, request=request, observed_messages=messages, tool_schemas=[], drive_root=tmp_path, task_id="mind",
        fit_candidate=lambda _m, _t: {"accepted": True}, protected_texts=(OWNER,))
    assert receipt.status == "applied" and usage is None
    assert {"role": "user", "content": OWNER} in candidate  # the owner's words, verbatim and whole
    assert not any(m.get("role") == "assistant" and m.get("content") == ASSISTANT for m in candidate)
    assert not any(m.get("role") == "tool" for m in candidate) and candidate[:2] == messages[:2]
    restored, _ = cc._restored_source_views([{"checkpoint_ref": receipt.checkpoint_ref, "unit_id": assistant_unit.unit_id,
                                              "raw_sha256": assistant_unit.raw_sha256}],
                                            drive_root=tmp_path, task_id="mind", request=request)
    assert ASSISTANT in restored[0]["content"][0]["text"]
    assert cc._capsule_metadata(restored[0])[1]["retention"] == "source_view"


def test_actual_refusal_runs_the_host_copies_rung_before_any_helper(tmp_path, monkeypatch):
    """7A end to end on the host side: the refusal opens rung (a) through the real emergency
    pass (no model), the transcript is republished, and ONE strictly smaller retry follows;
    the helper is never consulted while a model-free rung can still shrink the request."""
    from ouroboros import loop

    context = _ctx(tmp_path)
    old_facts = {"role": "user", HOST_CONTEXT_KIND_KEY: CONTEXT_FACTS_NAME, "content": HOST * 5}
    latest_facts = {"role": "user", HOST_CONTEXT_KIND_KEY: CONTEXT_FACTS_NAME, "content": "Current measured facts"}
    context.messages.extend([*_tool("one"), {"role": "user", "content": OWNER}, old_facts, latest_facts])
    inner = context.tools._ctx
    assert not getattr(inner, "_last_context_observation", None)  # first refused request: no usable exposure
    inner._owner_directives = [{"role": "user", "content": OWNER}]
    events, sends = [], []
    def measure(trial, **_kwargs):
        from ouroboros.loop_model_call import _remember_main_fit
        measured = _fit(estimated_input=cc._context_tokens_for_messages(trial.messages, 1.0), goal=1)
        _remember_main_fit(trial, measured)
        return measured
    monkeypatch.setattr(loop, "_measure_round_main_fit", measure)
    monkeypatch.setattr(loop, "_run_main_reclaim", lambda *_a, **_kw: pytest.fail("the helper must not run before rung a"))

    def dispatch(ctx, disposition, *, candidate_predicate=None, max_tokens=None, **_kwargs):
        sends.append(list(ctx.messages))
        if len(sends) == 1:
            ctx.accumulated_usage["_last_llm_error_kind"] = "context_overflow"
            return None, 0.0
        assert candidate_predicate(_candidate(disposition, 800)) and max_tokens == 65_536
        return {"role": "assistant", "content": "fits", "tool_calls": []}, 0.0

    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: _failed_capture())
    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda *_a, **kw: events.append(kw or _a[-1]))
    msg, _cost, mode = loop._call_round_model(context)
    assert msg["content"] == "fits" and mode == "max" and len(sends) == 2
    assert old_facts in sends[0] and old_facts not in sends[1]
    assert latest_facts in sends[1]
    assert {"role": "user", "content": OWNER} in sends[1] and sends[1][2:4] == _tool("one")  # bodies: a later rung
    assert any(_EMERGENCY_LABELS["host_copies"] in json.dumps(m.get("content")) for m in sends[1])
    assert inner.messages is context.messages and inner._transcript_rewrite_sanctioned == "compaction"
    emergency = [e for e in events if e.get("checkpoint_kind") == "context_reclaim_emergency"]
    assert [(e["rung"], e["status"]) for e in emergency] == [("host_copies", "applied")]
    assert inner._context_overflow_retries == {("route-a", "exec:round:1", "host_copies")}


def _candidate(disposition, size):
    from tests.test_loop_compaction import _candidate_request

    return _candidate_request(disposition, size=size)


def test_a_model_free_wake_delivery_earns_the_first_retry_before_any_rung(tmp_path, monkeypatch):
    """A wake's stored-source delivery applied on the refusal is already a smaller candidate:
    it is retried first; no rung is spent (or latched) unless that retry is refused too."""
    from ouroboros import loop, loop_model_call

    context = _ctx(tmp_path)
    sends = []
    monkeypatch.setattr(loop, "_measure_round_main_fit", _measure_cycle([_fit()]))
    monkeypatch.setattr(loop_model_call, "_project_wake_input", lambda ctx, *, overflowed=False: overflowed)
    monkeypatch.setattr(loop, "_run_main_reclaim", lambda *_a, **_kw: pytest.fail("no rung before the delivery retry"))

    def dispatch(ctx, disposition, *, candidate_predicate=None, **_kwargs):
        sends.append(candidate_predicate)
        if len(sends) == 1:
            ctx.accumulated_usage["_last_llm_error_kind"] = "context_overflow"
            return None, None  # a refused attempt may price as None
        assert candidate_predicate(_candidate(disposition, 800))
        return {"role": "assistant", "content": "fits", "tool_calls": []}, 0.25

    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: _failed_capture())
    msg, cost, mode = loop._call_round_model(context)
    assert msg["content"] == "fits" and cost == 0.25 and mode == "max" and len(sends) == 2
    assert context.tools._ctx._context_overflow_retries == set()


def test_memory_rung_passes_the_actual_window_and_current_transcript(tmp_path, monkeypatch):
    from ouroboros import loop
    from ouroboros.loop_round_limits import _coarsen_memory_view

    seen = []
    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda *_a, **_kw: None)

    def plan(system):
        p = SimpleNamespace(model="same-model", route_fp="route-a", preferred_mode="max", core_sha256="a" * 64,
                            core=object(), provider="openai", output_reserve_tokens=65_536, window_tokens=500_000)
        p.projection = lambda mode: SimpleNamespace(system_content_json=system, calibration_ratio=1.0)
        p.reproject_transcript = lambda messages, mode: [{"role": "system", "content": system}, *messages[1:]]

        def reproject_for_route(*, window_tokens, known_window, ratio, output_reserve,
                                tool_schemas, start_mode, current_messages):
            seen.append({"window": window_tokens, "known": known_window, "mode": start_mode,
                         "messages": list(current_messages), "schemas": list(tool_schemas),
                         "reserve": output_reserve})
            return plan("COARSE")

        p.reproject_for_route = reproject_for_route
        return p

    context = _ctx(tmp_path)
    context.messages.extend(_tool("tail"))
    before = list(context.messages)
    context.context_fit_plan = context.tools._ctx.context_fit_plan = plan("FULL")
    # A large transcript changes memory granularity, never the route's physical window.
    fit = _fit(estimated_input=520_000)
    assert _coarsen_memory_view(context, fit) is True
    assert seen == [{"window": 500_000, "known": True, "mode": "max", "messages": before,
                     "schemas": [], "reserve": 65_536}]
    assert context.context_fit_plan.window_tokens == 500_000
    assert context.messages[0]["content"] == "COARSE" and context.messages[1:] == before[1:]
    assert context.tools._ctx.context_fit_plan is context.context_fit_plan
    assert _coarsen_memory_view(context, fit) is False  # already at this granularity
    unknown = replace(fit, measurement=replace(fit.measurement, capacity_total_tokens=None))
    count = len(seen)
    assert _coarsen_memory_view(context, unknown) is False and len(seen) == count


def test_fallback_chain_admits_an_overflow_after_the_ladder_and_keeps_its_other_gates():
    from ouroboros import loop_transport
    from ouroboros.loop_llm_call import TRANSPORT_DEATHS_KEY

    ctx = SimpleNamespace(task_id="t", exact_model_route=False, route_wait_on_primary=False)
    assert loop_transport.fallback_chain_allowed(ctx, "context_overflow", None, {}) is True
    assert loop_transport.fallback_chain_allowed(ctx, "deadline_exhausted", None, {}) is False
    assert loop_transport.fallback_chain_allowed(ctx, "llm_output_exhausted", None, {}) is False
    assert loop_transport.fallback_chain_allowed(ctx, "context_overflow", None, {TRANSPORT_DEATHS_KEY: {}}) is False
    assert loop_transport.fallback_chain_allowed(replace_ns(ctx, exact_model_route=True), "context_overflow", None, {}) is False


def replace_ns(ns, **changes):
    return SimpleNamespace(**{**vars(ns), **changes})


def test_facts_line_names_room_money_and_tariff_as_known_or_unknown(tmp_path, monkeypatch):
    from ouroboros import loop, pricing
    from ouroboros.loop_messages import CONTEXT_FACTS_HEADER, append_context_facts, context_facts_line

    context = _ctx(tmp_path, preferred="low", mode="low")
    context.messages.extend(_tool("one"))
    before = [dict(m) for m in context.messages]
    monkeypatch.setattr(loop, "_measure_round_main_fit", _measure_cycle([_fit(profile="owner_low", mode="low")]))
    assert append_context_facts(context, money={"budget_remaining_usd": 12.5, "quota": "3 of 10 this hour"}) is True
    assert context.messages[:-1] == before  # appended as a new tail row, nothing before it rewritten
    line = context.messages[-1]["content"]
    assert context.messages[-1]["role"] == "user" and context.messages[-1][HOST_CONTEXT_KIND_KEY] == CONTEXT_FACTS_NAME
    assert line.startswith(f"{CONTEXT_FACTS_HEADER} round 1 |")
    assert "model same-model" in line and "mode low" in line
    usage = context.accumulated_usage
    free = 250_000 - usage["_context_prompt_estimate"] - usage["_context_reply_allowance_tokens"]
    for part in ("window 500,000", "target 250,000", "input ~120,000 before this line (cold_estimate)",
                 f"free {free:,}", "largest unit:2:3:", "cost unknown", "budget left $12.50",
                 "quota 3 of 10 this hour", "tariff unknown"):
        assert part in line, part
    assert "should" not in line and "compact" not in line.lower()  # facts, no advice
    assert append_context_facts(context) is False and len(context.messages) == len(before) + 1  # identical round/route/body/schema
    context.round_idx, context.accumulated_usage["cost"] = 2, 1.2345
    context.accumulated_usage["cost_final"] = False
    context.context_fit_plan.provider = "openai"
    monkeypatch.setattr(pricing, "get_pricing",
                        lambda *, provider, allow_live_fetch, model="": {"same-model": (2.0, 0, 0, 10.0)})
    assert append_context_facts(context, money={"ceiling_usd": 40, "tree_line": "tree $3.00 of $40.00"}) is True
    second = context.messages[-1]["content"]
    assert "round 2" in second and "recorded cost $1.23 (total not final)" in second
    assert "ceiling $40.00" in second and "tree $3.00 of $40.00" in second
    assert "tariff $2.00/M in, $10.00/M out" in second and "quota unknown" in second
    unbound = _ctx(tmp_path)
    unbound.context_fit_plan = None
    assert "input unmeasured" in context_facts_line(unbound, measured=False) and "window unknown" in context_facts_line(unbound, measured=False)
    assert append_context_facts(unbound) is True and "input unmeasured" in unbound.messages[-1]["content"]


def test_facts_refresh_for_changed_send_but_preserve_all_prior_rows(tmp_path, monkeypatch):
    from copy import deepcopy
    from ouroboros import loop
    from ouroboros.loop_messages import append_context_facts

    context = _ctx(tmp_path)
    monkeypatch.setattr(loop, "_measure_round_main_fit", _measure_cycle([_fit()]))
    assert append_context_facts(context, money={})
    assert not append_context_facts(context, money={})

    for change in ("body", "schemas", "route", "effort", "stale"):
        if change == "body":
            context.messages.append({"role": "user", "content": "A newly arrived owner correction."})
        elif change == "schemas":
            context.tool_schemas.append({"type": "function", "function": {"name": "new_capability"}})
        elif change == "route":
            context.context_fit_plan.route_fp = "different-account-route"
        elif change == "effort":
            context.active_effort = "max"
        else:
            context.context_fit_plan.stale = True
        before = deepcopy(context.messages)
        assert append_context_facts(context, money={}), change
        assert context.messages[:-1] == before  # append outside the sent prefix
        assert context.messages[-1][HOST_CONTEXT_KIND_KEY] == CONTEXT_FACTS_NAME
        assert not append_context_facts(context, money={}), change


def test_facts_largest_components_include_core_and_schemas(tmp_path, monkeypatch):
    from ouroboros import loop
    from ouroboros.loop_messages import append_context_facts

    context = _ctx(tmp_path)
    context.messages[0]["content"] = "Complete required core. " * 6000
    context.messages.extend(_tool("evidence"))
    context.tool_schemas = [{"type": "function", "function": {
        "name": "large_schema", "description": "Complete schema documentation. " * 3000}}]
    monkeypatch.setattr(loop, "_measure_round_main_fit", _measure_cycle([_fit()]))
    assert append_context_facts(context, money={})
    line = context.messages[-1]["content"]
    assert "system[0:0]" in line and "tool schemas" in line and "unit:2:3:" in line
