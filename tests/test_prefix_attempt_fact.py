"""What a request does to its route's cached start is recorded on the attempt that sends it.

A provider keeps one cache per model and reads it in the wire order tools, system blocks,
messages, so the record compares a candidate with the last answered request on its own route
and names the first place they differ. The observer's single baseline (``observe_send``)
compares with the previous send on any model; these cases pin what that baseline cannot say:
a return to a model after rounds on another, which system block changed, and that the fact
reaches the attempt's ledger row.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros import loop, transcript_prefix as prefix
from ouroboros.physical_attempt_context import PhysicalAttemptContext
from tests._usage_store_testing import dispatched_attempts
from tests.test_llm_claudexor import setup as gateway_fixture
from tests.test_model_wait import live_wait as wait_fixture
from tests.test_subscription_main_wait import main_call as main_fixture

setup, live_wait, main_call = gateway_fixture, wait_fixture, main_fixture

OPUS, CODEX = "anthropic/claude-opus-5.5", "claudexor::codex=gpt-6-astra"
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {}}}]


def system(rules="rules", memory="memory", context="context"):
    marked = {"cache_control": {"type": "ephemeral"}}
    return {"role": "system", "content": [{"type": "text", "text": rules, **marked},
                                          {"type": "text", "text": memory, **marked},
                                          {"type": "text", "text": context}]}


def turn(n):
    return [{"role": "assistant", "content": f"step {n}"}, {"role": "tool", "tool_call_id": f"c{n}", "content": f"result {n}"}]


def send(slot, messages, route, *, tools=TOOLS, now=0.0, round_idx=1):
    """One answered request: the candidate's fact at dispatch, then the answer's observation."""
    fact = prefix.attempt_prefix_fact(slot, messages, tools, route, now=now)
    prefix.observe_send(slot, messages, round_idx=round_idx, tools=tools)
    return fact


def test_the_first_request_and_a_pure_extension_of_it():
    slot = SimpleNamespace()
    start = [system(), {"role": "user", "content": "task"}]
    assert send(slot, start, OPUS, now=100.0) == {"departs": "first"}
    assert send(slot, start + turn(1), OPUS, now=130.5) == {"departs": "append", "base_at": 100.0}


def test_a_return_to_a_model_is_compared_with_that_models_last_request():
    slot = SimpleNamespace()
    start = [system(), {"role": "user", "content": "task"}]
    send(slot, start, OPUS, now=0.0)
    on_codex = start + turn(1)
    assert send(slot, on_codex, CODEX, now=10.0) == {"departs": "route", "from_route": OPUS}
    send(slot, on_codex + turn(2), CODEX, now=20.0)
    # Back on the first model, three rounds later: its own last request is the baseline, and only rows were added.
    back = prefix.attempt_prefix_fact(slot, on_codex + turn(2) + turn(3), [*TOOLS], OPUS, now=50.0)
    assert back == {"departs": "append", "from_route": CODEX, "base_at": 0.0}
    # The single baseline alone would have compared with the Codex request and said nothing of the route.
    assert prefix.departure(slot, on_codex + turn(2) + turn(3), TOOLS) is None


def test_the_changed_system_block_is_named_by_its_position():
    slot = SimpleNamespace()
    rows = [{"role": "user", "content": "task"}, *turn(1)]
    send(slot, [system(), *rows], OPUS)
    memory = prefix.attempt_prefix_fact(slot, [system(memory="memory, one page later"), *rows], TOOLS, OPUS, now=1.0)
    assert (memory["departs"], memory["index"], memory["of"]) == ("system", 1, 3)
    context = prefix.attempt_prefix_fact(slot, [system(context="another capture"), *rows], TOOLS, OPUS, now=1.0)
    assert (context["departs"], context["index"], context["of"]) == ("system", 2, 3)
    # A marker that moves is not content: the same blocks with other cache markers still only extend the request.
    unmarked = system()
    for block in unmarked["content"]:
        block.pop("cache_control", None)
    assert prefix.attempt_prefix_fact(slot, [unmarked, *rows, *turn(2)], TOOLS, OPUS, now=1.0)["departs"] == "append"


def test_the_earliest_difference_in_wire_order_wins():
    slot = SimpleNamespace()
    rows = [{"role": "user", "content": "task"}, *turn(1), *turn(2)]
    send(slot, [system(), *rows], OPUS)
    other_tools = [{"type": "function", "function": {"name": "read_file", "description": "Reads one file.", "parameters": {}}}]
    changed_everything = [system(memory="new"), {"role": "user", "content": "task"}, *turn(9)]
    assert prefix.attempt_prefix_fact(slot, changed_everything, other_tools, OPUS, now=1.0)["departs"] == "tools"
    folded = [system(), {"role": "user", "content": "task"}, {"role": "user", "content": "[fold of steps 1-2]"}]
    fact = prefix.attempt_prefix_fact(slot, folded, TOOLS, OPUS, now=1.0)
    assert (fact["departs"], fact["index"], fact["of"]) == ("messages", 2, 6)  # the system row is message 0
    shorter = [system(), *rows[:3]]
    fact = prefix.attempt_prefix_fact(slot, shorter, TOOLS, OPUS, now=1.0)
    assert (fact["departs"], fact["index"], fact["of"]) == ("messages", 4, 6)


def test_the_seams_that_rewrote_the_view_are_named_and_not_consumed():
    slot = SimpleNamespace()
    rows = [{"role": "user", "content": "task"}, *turn(1)]
    send(slot, [system(), *rows], OPUS)
    prefix.sanction_rewrite(slot, "compaction")
    prefix.sanction_rewrite(slot, "continuation_restored", scope="system")
    prefix.sanction_rewrite(slot, "handover", scope="system")  # a second seam of one scope: the stamp keeps the last, the fact both
    fact = prefix.attempt_prefix_fact(slot, [system(context="again"), rows[0]], TOOLS, OPUS, now=5.0)
    assert fact["origins"] == ["compaction", "continuation_restored", "handover"]
    # The answer's observer still receives both stamps: the record only read them.
    seen = prefix.observe_send(slot, [system(context="again"), rows[0]], round_idx=2, tools=TOOLS)
    assert (seen["kind"], seen["sanctioned_by"]) == ("system_rewritten", "handover")
    assert "origins" not in prefix.attempt_prefix_fact(slot, [system(context="again"), rows[0], *turn(2)], TOOLS, OPUS, now=6.0)


def test_only_an_answered_request_becomes_the_routes_record():
    slot = SimpleNamespace()
    start = [system(), {"role": "user", "content": "task"}]
    send(slot, start, OPUS, now=0.0)
    # A candidate that was never answered (a failed send) leaves the record of the last answered request.
    prefix.attempt_prefix_fact(slot, [system(memory="other"), start[1]], TOOLS, OPUS, now=5.0)
    again = prefix.attempt_prefix_fact(slot, start + turn(1), TOOLS, OPUS, now=9.0)
    assert again == {"departs": "append", "base_at": 0.0}
    # An observation that named no route (a helper's send) changes no record either.
    setattr(slot, prefix.PENDING_ROUTE_ATTR, None)
    prefix.observe_send(slot, [system(memory="helper view")], round_idx=3, tools=TOOLS)
    assert prefix.attempt_prefix_fact(slot, start + turn(1), TOOLS, OPUS, now=12.0)["departs"] == "append"


def test_a_slot_restored_from_before_the_record_says_so_and_a_checkpoint_carries_the_routes():
    old = SimpleNamespace()
    prefix.observe_send(old, [system(), {"role": "user", "content": "task"}], round_idx=1, tools=TOOLS)
    assert prefix.attempt_prefix_fact(old, [system()], TOOLS, OPUS, now=1.0) == {"departs": "unrecorded"}

    slot = SimpleNamespace()
    start = [system(), {"role": "user", "content": "task"}]
    send(slot, start, OPUS, now=0.0)
    saved = json.loads(json.dumps({key: getattr(slot, key) for key in prefix.OBSERVER_STATE_ATTRS if hasattr(slot, key)}))
    restored = SimpleNamespace(**saved)
    assert prefix.attempt_prefix_fact(restored, start + turn(1), TOOLS, OPUS, now=4.0) == {"departs": "append", "base_at": 0.0}
    assert prefix.attempt_prefix_fact(restored, [system(rules="new rules"), start[1]], TOOLS, OPUS, now=4.0)["index"] == 0


def test_a_task_keeps_a_bounded_number_of_routes():
    slot = SimpleNamespace()
    start = [system(), {"role": "user", "content": "task"}]
    for n in range(prefix._ROUTES_KEPT + 1):
        send(slot, start, f"model-{n}", now=float(n))
    routes = getattr(slot, prefix.ROUTE_SENDS_ATTR)
    assert [name for name, held in routes.items() if "messages" in held] == [f"model-{n}" for n in range(1, prefix._ROUTES_KEPT + 1)]
    # The route whose record was let go keeps its name: its next request has no base and says so, never "first" or "entered".
    assert routes["model-0"] == {"dropped": True}
    dropped = prefix.attempt_prefix_fact(slot, start, TOOLS, "model-0", now=9.0)
    assert dropped["departs"] == "unrecorded" and "base_at" not in dropped and prefix.prefix_class(dropped, "1h") == ("not_recorded", "")
    assert prefix.attempt_prefix_fact(slot, start, TOOLS, "model-new", now=9.0)["departs"] == "route"


def _physical(**extra):
    return PhysicalAttemptContext(
        profile="owner_max", rendered_mode="max", measurement_basis="fresh_route_usage", route_fp="fp",
        round_id="exec:round:1", target_total_tokens=None, capacity_total_tokens=1000, context_target_miss=False,
        automatic_pass_used=False, **extra)


def test_the_fact_rides_the_physical_context_into_the_ledger_row():
    from dataclasses import asdict

    from ouroboros.usage_ledger import UsageLedgerCorrupt, _validate_candidate_facts

    ctx = SimpleNamespace(tools=SimpleNamespace(_ctx=SimpleNamespace()), accumulated_usage={}, tool_schemas=TOOLS,
                          messages=[system(), {"role": "user", "content": "task"}], active_model=OPUS, active_use_local=False)
    prefix.remember_attempt_prefix(ctx)
    bound = prefix.with_attempt_prefix(_physical(), ctx.accumulated_usage)
    assert bound.prefix == {"departs": "first"} and asdict(bound)["prefix"] == {"departs": "first"}
    assert bound == _physical()  # the record is no part of the context's identity
    assert PhysicalAttemptContext(**asdict(bound)).prefix == {"departs": "first"}
    # Without a fact the context passes through untouched; without a context there is nothing to carry.
    assert prefix.with_attempt_prefix(_physical(), {}) == _physical() and prefix.with_attempt_prefix(_physical(), {}).prefix is None
    assert prefix.with_attempt_prefix(None, ctx.accumulated_usage) is None

    row = {"candidate_measurement_kind": "canonical_json_v1", "candidate_raw_sha256": "0" * 64,
           "candidate_context_sha256": "1" * 64, "candidate_raw_size_bytes": 2, "candidate_context_size_bytes": 1,
           "physical_context": asdict(bound)}
    _validate_candidate_facts(row, 1)  # the row the ledger writes for this send is one it accepts back
    row["physical_context"].pop("prefix")
    _validate_candidate_facts(row, 1)  # and a row written before the record existed still is
    row["physical_context"]["prefix"] = "append"
    with pytest.raises(UsageLedgerCorrupt, match="prefix"):
        _validate_candidate_facts(row, 1)


def test_real_sends_leave_the_fact_on_their_ledger_rows(main_call):
    """Through the loop's own dispatch and the real usage store: first request, an extension, a rewritten start."""
    from tests.test_llm_claudexor import result

    ctx, gateway, *_ = main_call
    slot = ctx.tools._ctx

    def send(round_idx):
        ctx.round_idx = round_idx
        answer, _cost = loop._dispatch_round_model(ctx, loop._measure_round_main_fit(ctx, automatic_pass_used=False), attempt_cap=1)
        assert answer
        prefix.observe_send(slot, ctx.messages, round_idx=round_idx, tools=ctx.tool_schemas)  # as the loop does after an answer
        gateway.results.append(result())
        gateway.dispatch.append("response_received")

    send(1)
    ctx.messages.extend(turn(2))
    send(2)
    ctx.messages[0] = {**ctx.messages[0], "content": "another capture of the system text"}
    send(3)
    facts = [(row.get("physical_context") or {}).get("prefix") for row in dispatched_attempts(ctx.drive_root)]
    assert [fact["departs"] for fact in facts] == ["first", "append", "system"]
    assert (facts[2]["index"], facts[2]["of"]) == (0, 1) and "base_at" in facts[1] and "from_route" not in facts[2]


def test_one_reader_names_the_class_in_the_providers_own_order():
    name = prefix.prefix_class
    assert name(None) == name({}) == name({"departs": "unrecorded"}) == ("not_recorded", "")
    assert name({"departs": "first"}) == ("first_request", "")
    assert name({"departs": "route", "from_route": CODEX}) == ("entered_model", CODEX)
    assert name({"departs": "tools", "origins": ["continuation_restored"], "from_route": CODEX, "since_s": 9e9}, "1h") == \
        ("tools_changed", "continuation_restored")  # an earlier departure already accounts for the row
    assert name({"departs": "system", "index": 1, "of": 3, "block": "my story and identity"}) == ("system_changed", "my story and identity")
    assert name({"departs": "messages", "index": 4, "of": 9, "origins": ["compaction", "handover"]}) == ("history_rewritten", "compaction, handover")
    # A pure extension: the pause is arithmetic over recorded facts (the base's answer, the row's own dispatch) and a
    # named lifetime, never a stored cause; without the dispatch time or a named lifetime it is not derived.
    assert name({"departs": "append", "base_at": 0.0}, "1h", 3601.0) == ("paused", "")
    assert name({"departs": "append", "base_at": 0.0}, "default", 3601.0) == name({"departs": "append", "base_at": 0.0}, "1h") == ("append", "")
    assert name({"departs": "append", "base_at": 0.0, "from_route": CODEX}, "5m", 301.0) == ("paused", "")
    assert name({"departs": "append", "base_at": 0.0, "from_route": CODEX}, "1h", 12.0) == ("returned_to_model", CODEX)
    assert name({"departs": "append", "base_at": 0.0}, "1h", 12.0) == ("append", "")


def test_another_sends_answer_never_records_a_candidate_that_was_not_answered():
    slot = SimpleNamespace()
    start = [system(), {"role": "user", "content": "task"}]
    send(slot, start, OPUS, now=0.0)
    prefix.attempt_prefix_fact(slot, start + turn(1), TOOLS, OPUS, now=5.0)  # measured, and its send failed
    helper = [{"role": "system", "content": "a helper's own view"}, {"role": "user", "content": "summarise this"}]
    prefix.observe_send(slot, helper, round_idx=2, tools=TOOLS)  # a helper's answer is observed on the same slot
    held = getattr(slot, prefix.ROUTE_SENDS_ATTR)[OPUS]
    assert (held["at"], len(held["messages"])) == (0.0, 1)  # still the request that was answered
    assert prefix.attempt_prefix_fact(slot, start + turn(1), TOOLS, OPUS, now=9.0) == {"departs": "append", "base_at": 0.0}


def test_the_forced_final_takes_its_fact_from_the_input_it_measured():
    slot = SimpleNamespace()
    start = [system(), {"role": "user", "content": "task"}]
    send(slot, start, OPUS, now=0.0)
    working = [system(rules="a working view the forced send does not carry"), start[1]]
    ctx = SimpleNamespace(tools=SimpleNamespace(_ctx=slot), accumulated_usage={}, tool_schemas=TOOLS, messages=working,
                          active_model=OPUS, active_use_local=False)
    forced = [*start, *turn(1), {"role": "user", "content": "finish with what you have"}]
    bound = prefix.with_attempt_prefix(_physical(), ctx.accumulated_usage, ctx, forced)
    assert bound.prefix["departs"] == "append"  # the measured input extends the route's last request; ``ctx.messages`` would say "system"
    assert prefix.with_attempt_prefix(None, ctx.accumulated_usage, ctx, working) is None  # no context, nothing taken
    prefix.observe_send(slot, forced, round_idx=2, tools=TOOLS)  # and its answer is the route's record
    assert len(getattr(slot, prefix.ROUTE_SENDS_ATTR)[OPUS]["messages"]) == len(forced) - 1


def test_an_unmeasured_candidate_ends_the_wait_of_the_measured_one():
    """A route with no fit plan is sent unmeasured: its answer must never become the measured route's record."""
    slot = SimpleNamespace()
    start = [system(), {"role": "user", "content": "task"}]
    send(slot, start, OPUS, now=0.0)
    ctx = SimpleNamespace(tools=SimpleNamespace(_ctx=slot), accumulated_usage={}, tool_schemas=TOOLS, messages=start + turn(1),
                          active_model=OPUS, active_use_local=False)
    prefix.remember_attempt_prefix(ctx)  # measured for this route; its send fails
    assert ctx.accumulated_usage[prefix.ATTEMPT_PREFIX_KEY]["departs"] == "append"
    assert prefix.forget_attempt_prefix(ctx) is None  # the next route is not measured
    assert prefix.ATTEMPT_PREFIX_KEY not in ctx.accumulated_usage
    assert prefix.with_attempt_prefix(_physical(), ctx.accumulated_usage).prefix is None
    prefix.observe_send(slot, start + turn(1), round_idx=2, tools=TOOLS)  # the very same rows, answered by that other route
    held = getattr(slot, prefix.ROUTE_SENDS_ATTR)[OPUS]
    assert (held["at"], len(held["messages"])) == (0.0, 1)


def _answered(usage, route=OPUS, read=True):
    """What ``loop_llm_call`` notes where a response arrives: an answer that read its input, or a body error."""
    model, local = (route[:-len(" (local)")], True) if route.endswith(" (local)") else (route, False)
    reply = {"prompt_tokens": 1_000} if read else {"prompt_tokens": 0, "provider_error": {"code": "context_length_exceeded"}}
    prefix.note_prefix_answer(usage, reply, model, local)


def _main_ctx(slot, usage, messages):
    return SimpleNamespace(tools=SimpleNamespace(_ctx=slot), accumulated_usage=usage, tool_schemas=TOOLS,
                           messages=messages, active_model=OPUS, active_use_local=False)


def test_an_answer_the_loop_did_not_accept_still_moves_its_routes_base(monkeypatch):
    """An empty or output-limited reply was written and charged: the next measurement sees its route answered."""
    import time

    monkeypatch.setattr(time, "time", lambda: 5_000.0)
    slot, usage = SimpleNamespace(), {}
    ctx = _main_ctx(slot, usage, [system(), {"role": "user", "content": "task"}])
    prefix.remember_attempt_prefix(ctx)
    assert usage[prefix.ATTEMPT_PREFIX_KEY] == {"departs": "first"}
    _answered(usage)  # an empty answer: the loop never observes it
    ctx.messages = ctx.messages + turn(1)
    prefix.remember_attempt_prefix(ctx)
    assert usage[prefix.ATTEMPT_PREFIX_KEY] == {"departs": "append", "base_at": 5_000.0}  # dated by the answer
    # An answer of another route ends the wait without moving this route's base.
    _answered(usage, route=CODEX)
    prefix.forget_attempt_prefix(ctx)
    assert len(getattr(slot, prefix.ROUTE_SENDS_ATTR)[OPUS]["messages"]) == 1  # still the first request's record
    assert getattr(slot, prefix.PENDING_ROUTE_ATTR) is None


def test_a_body_error_read_nothing_and_moves_no_base():
    slot, usage = SimpleNamespace(), {}
    ctx = _main_ctx(slot, usage, [system(), {"role": "user", "content": "task"}])
    prefix.remember_attempt_prefix(ctx)
    _answered(usage, read=False)  # a structured error in an HTTP 200 body
    prefix.note_prefix_answer(usage, {"completion_tokens": 0}, OPUS, False)  # neither input nor output: no evidence
    assert prefix.ANSWER_KEY not in usage
    ctx.messages = ctx.messages + turn(1)
    prefix.remember_attempt_prefix(ctx)
    assert usage[prefix.ATTEMPT_PREFIX_KEY] == {"departs": "first"}  # nothing was answered yet


def test_a_retry_after_its_own_answer_extends_that_request(monkeypatch):
    import time

    clock = {"now": 100.0}
    monkeypatch.setattr(time, "time", lambda: clock["now"])
    slot, usage = SimpleNamespace(), {}
    start = [system(), {"role": "user", "content": "task"}]
    send(slot, start, OPUS, now=0.0)
    ctx = _main_ctx(slot, usage, start + turn(1))
    prefix.remember_attempt_prefix(ctx)
    bound = prefix.with_attempt_prefix(_physical(), usage)
    assert bound.prefix["departs"] == "append" and bound.prefix["base_at"] == 0.0
    # The first attempt is answered (empty) at 399; the retry starts at 401 and extends that answered request.
    clock["now"] = 399.0
    _answered(usage)
    retry = prefix.refresh_attempt_prefix(bound, usage, OPUS, False)
    assert retry.prefix == {"departs": "append", "base_at": 399.0}
    assert prefix.refresh_attempt_prefix(retry, usage, OPUS, False) is retry  # no new answer: nothing to refresh
    # A failed send left no answer, and another route's answer is not this candidate's.
    assert prefix.refresh_attempt_prefix(bound, {**usage, prefix.ANSWER_SEEN_KEY: 1}, OPUS, False) is bound
    _answered(usage, route=CODEX)
    assert prefix.refresh_attempt_prefix(retry, usage, OPUS, False) is retry
    assert prefix.refresh_attempt_prefix(_physical(), usage, OPUS, False).prefix is None  # no fact, nothing made up


def test_a_checkpoint_carries_the_waiting_candidate_and_its_answer():
    slot, usage = SimpleNamespace(), {}
    start = [system(), {"role": "user", "content": "task"}]
    send(slot, start, OPUS, now=0.0)
    ctx = _main_ctx(slot, usage, start + turn(1))
    prefix.remember_attempt_prefix(ctx)
    _answered(usage)  # an output-limited reply; the task parks before another measurement
    saved = json.loads(json.dumps({key: getattr(slot, key) for key in prefix.OBSERVER_STATE_ATTRS if hasattr(slot, key)}))
    restored, restored_usage = SimpleNamespace(**saved), json.loads(json.dumps(usage))
    ctx = _main_ctx(restored, restored_usage, start + turn(1) + turn(2))
    prefix.remember_attempt_prefix(ctx)
    fact = restored_usage[prefix.ATTEMPT_PREFIX_KEY]
    assert fact["departs"] == "append" and fact["base_at"] == usage[prefix.ANSWER_KEY]["at"]  # the answered reply, not the older one


def test_a_deferred_promotion_keeps_the_seams_named_after_the_measurement():
    slot, usage = SimpleNamespace(), {}
    start = [system(), {"role": "user", "content": "task"}]
    send(slot, start, OPUS, now=0.0)
    prefix.sanction_rewrite(slot, "compaction")  # carried by the next candidate
    ctx = _main_ctx(slot, usage, [system(), {"role": "user", "content": "task"}, {"role": "user", "content": "[fold]"}])
    prefix.remember_attempt_prefix(ctx)
    _answered(usage)  # unaccepted
    prefix.sanction_rewrite(slot, "route_rebind", scope="system")  # after the answer, before the next measurement
    ctx.messages = [system(context="rebound"), *ctx.messages[1:]]
    prefix.remember_attempt_prefix(ctx)
    assert usage[prefix.ATTEMPT_PREFIX_KEY]["origins"] == ["route_rebind"]  # the fold was carried; the rebind was not


def test_the_accepted_answer_dates_its_routes_base(monkeypatch):
    import time

    monkeypatch.setattr(time, "time", lambda: 700.0)
    slot, usage = SimpleNamespace(), {}
    start = [system(), {"role": "user", "content": "task"}]
    prefix.attempt_prefix_fact(slot, start, TOOLS, OPUS, now=650.0, usage=usage)  # measured at 650, answered at 700
    _answered(usage)
    prefix.observe_send(slot, start, round_idx=1, tools=TOOLS, usage=usage)
    assert getattr(slot, prefix.ROUTE_SENDS_ATTR)[OPUS]["at"] == 700.0
    prefix.attempt_prefix_fact(slot, start + turn(1), TOOLS, OPUS, now=800.0, usage=usage)
    prefix.observe_send(slot, start + turn(1), round_idx=2, tools=TOOLS, usage=usage)  # no answer noted since: its measurement
    assert getattr(slot, prefix.ROUTE_SENDS_ATTR)[OPUS]["at"] == 800.0


def test_an_unmeasured_route_ends_the_wait_at_its_real_call_site():
    """``loop_model_call._measure_round_main_fit`` with no plan for the acting model forgets the waiting candidate."""
    from ouroboros import loop_model_call

    slot = SimpleNamespace()
    start = [system(), {"role": "user", "content": "task"}]
    send(slot, start, OPUS, now=0.0)
    usage = {}
    prefix.attempt_prefix_fact(slot, start + turn(1), TOOLS, OPUS, now=1.0, usage=usage)
    usage[prefix.ATTEMPT_PREFIX_KEY] = {"departs": "append"}
    ctx = SimpleNamespace(tools=SimpleNamespace(_ctx=slot), accumulated_usage=usage, messages=start + turn(1),
                          context_fit_plan=SimpleNamespace(model=OPUS), active_model=CODEX, active_use_local=False)
    assert loop_model_call._measure_round_main_fit(ctx, automatic_pass_used=False) is None
    assert getattr(slot, prefix.PENDING_ROUTE_ATTR) is None and prefix.ATTEMPT_PREFIX_KEY not in usage


def test_a_generated_reply_without_an_input_count_still_read_its_input(monkeypatch):
    """Output proves the provider read the input, its count reported or not: a retry extends that answered request."""
    import time

    clock = {"now": 100.0}
    monkeypatch.setattr(time, "time", lambda: clock["now"])
    slot, usage = SimpleNamespace(), {}
    start = [system(), {"role": "user", "content": "task"}]
    send(slot, start, OPUS, now=0.0)
    ctx = _main_ctx(slot, usage, start + turn(1))
    prefix.remember_attempt_prefix(ctx)
    bound = prefix.with_attempt_prefix(_physical(), usage)
    clock["now"] = 399.0
    prefix.note_prefix_answer(usage, {"completion_tokens": 3}, OPUS, False)
    assert prefix.refresh_attempt_prefix(bound, usage, OPUS, False).prefix == {"departs": "append", "base_at": 399.0}


def test_a_malformed_answer_record_never_blocks_a_send_or_consumes_a_reply():
    usage = {prefix.ANSWER_KEY: {"n": "invalid-persisted-count", "route": OPUS, "at": "x"},
             prefix.ATTEMPT_PREFIX_KEY: {"departs": "append"}}
    assert prefix.with_attempt_prefix(_physical(), usage).prefix == {"departs": "append"}
    prefix.note_prefix_answer(usage, {"prompt_tokens": 10}, OPUS, False)
    assert usage[prefix.ANSWER_KEY]["n"] == 1 and usage[prefix.ANSWER_KEY]["route"] == OPUS


def test_a_candidate_without_a_system_row_names_the_removed_block():
    slot = SimpleNamespace()
    send(slot, [system(), {"role": "user", "content": "task"}], OPUS, now=0.0)
    fact = prefix.attempt_prefix_fact(slot, [{"role": "user", "content": [{"type": "text", "text": "## Identity\nhi"}]}],
                                      TOOLS, OPUS, now=1.0)
    assert (fact["departs"], fact["of"], fact["block"]) == ("system", 0, "removed block")
