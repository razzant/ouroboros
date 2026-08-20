"""Dispatch: what the child really got, and how loudly the difference is told.

Split verbatim out of ``tests/test_model_slot_role_model.py`` by theme. This module owns the
lane inherited through the whole dispatch path, the reduction that reaches the record, the
child and the parent's readback, the explicit harness pin that is a typed blocker rather than
a paid reroute, the route effort ceiling disclosed at dispatch, and the harness policy an
explicit or required lane always wins over.
"""

from __future__ import annotations


import pytest

from ouroboros import subagents

from tests._model_slot_role_shared import (
    _enqueue_through_supervisor,
    _scheduling_ctx,
)
from tests._model_slot_role_shared import _owned_gateway_uses_each_test_transport as __owned_gateway_uses_each_test_transport

# Fixtures are requested by name as test parameters, so they are re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
_owned_gateway_uses_each_test_transport = __owned_gateway_uses_each_test_transport


def _light_lane_ctx(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "provider::cheap")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "provider::strong")
    return _scheduling_ctx(tmp_path, **kwargs)

def _dispatched(tmp_path, monkeypatch, **schedule_kwargs):
    """Drive the WHOLE path: tool call -> event -> supervisor -> the worker's dispatch.

    Everything a child GETS is decided in the last step, so a test that stops at the
    event asserts on intent and calls it a resolution."""
    from ouroboros.agent import resolve_dispatch_axes

    task = _enqueue_through_supervisor(tmp_path, monkeypatch, **schedule_kwargs)
    return task, resolve_dispatch_axes(task)

def test_an_omitted_lane_inherits_through_the_whole_dispatch_path(tmp_path, monkeypatch):
    """The default the owner chose, asserted on the task a WORKER actually runs.

    A Heavy parent that hands a child a slice of its own job used to get a Light
    child at every surface — event, envelope, durable record — with nothing saying
    the demotion had happened. Inheritance is not a resolver-local nicety: the
    parent's lane has to survive the event, the supervisor AND the queue, because
    the child that inherits it is resolved after all three."""
    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "provider::strong")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "provider::cheap")

    task, _ = _dispatched(tmp_path / "inherit", monkeypatch, parent_lane="heavy")
    assert task["effective_model_lane"] == "heavy"
    assert task["model"] == "provider::strong"
    assert task["requested_model_lane"] == "auto"
    # Inheriting what the parent runs takes nothing away, so nothing shouts.
    assert task["capability_delta"]["reduced"] is False

    named, _ = _dispatched(
        tmp_path / "named", monkeypatch, parent_lane="heavy", model_lane="light")
    assert named["effective_model_lane"] == "light"
    assert named["model"] == "provider::cheap"

def test_a_reduction_reaches_the_record_the_child_and_the_parents_readback(tmp_path, monkeypatch):
    """The invariant: a child landing below what was asked for is LOUD in all THREE
    places named by the owner — the durable record/envelope, the child's own prompt,
    and the TERMINAL parent-facing result.

    The executor pin is the reduction that had no reporting at all — `harness` was
    recorded on the event, the task and the envelope (under the key `executor`, which
    reads as who RAN it) and then no code ever resolved it, so a child that ran
    natively left a durable record claiming a harness had run it.

    Since the D28 correction the unhonored EXPLICIT pin resolves to `blocked` rather
    than to paid `native` (see test_an_explicit_harness_pin_is_a_typed_blocker), so
    what the three surfaces must carry is the BLOCK. The disclosure duty is the same;
    only the honest answer changed.

    None of the three can be reached at SCHEDULE time any more, which is the point:
    all three read a fact that does not exist until the child starts."""
    from ouroboros.agent import capability_delta_prompt_block
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.control import _get_task_result

    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    task, dispatch = _dispatched(tmp_path / "three", monkeypatch, executor="harness")

    # 1) the durable record + its envelope
    assert task["effective_executor"] == "blocked"
    delta = task["capability_delta"]
    assert delta["reduced"] is True and delta["reason"] == "harness_not_configured"
    assert task["subagent_envelope"]["capability_delta"] == delta
    assert task["subagent_envelope"]["executor"] == "harness"

    # 2) the child's own prompt
    block = capability_delta_prompt_block(dispatch)
    assert "[CAPABILITY DELTA]" in block
    assert "executor harness->blocked" in block

    # 3) the parent, when it READS the answer
    ctx = _scheduling_ctx(tmp_path / "readback")
    write_task_result(tmp_path / "readback", "child1", "completed",
                      result="done", capability_delta=delta)
    out = _get_task_result(ctx, "child1")
    assert "capability_delta" in out and "harness_not_configured" in out

    # ...and the scheduling result no longer pretends to know: it states the request.
    from ouroboros.tools.control import _schedule_task

    sched_ctx = _scheduling_ctx(tmp_path / "sched")
    scheduled = _schedule_task(sched_ctx, objective="o", expected_output="e",
                               executor="harness")
    assert "CAPABILITY_DELTA" not in scheduled
    assert "requested_lane=auto" in scheduled

def test_a_child_that_got_what_was_asked_stays_quiet(tmp_path, monkeypatch):
    """A warning that always fires is not a warning. Nothing was taken away here, so
    no block reaches the child and no delta reaches the parent's readback — `auto`
    resolving to a concrete executor is the absence of a preference, not a loss."""
    from ouroboros.agent import capability_delta_prompt_block

    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "provider::cheap")
    task, dispatch = _dispatched(tmp_path / "quiet", monkeypatch, parent_lane="light")
    assert task["capability_delta"]["reduced"] is False
    assert task["capability_delta"]["effective_executor"] == "native"
    assert capability_delta_prompt_block(dispatch) == ""

def test_an_explicit_harness_pin_is_a_typed_blocker_not_a_paid_reroute(tmp_path, monkeypatch):
    """D28, the owner's words: at an EXPLICIT `executor: harness` an unavailable route
    stays a TYPED BLOCKER — «деньги API не тратятся без явного выбора». The §8 ban #12
    exception (fall back to another route) is AUTO-ONLY.

    This resolved to `native` with a loud `capability_delta`, which discloses the wrong
    thing: however loudly it is announced, re-routing the pin to native execution
    spends exactly the metered money the parent refused. The reason string matches
    `cxi/p34-converged`'s rule table so synthesis adopts that table without a
    behavioural diff (synthesis hazard H1)."""
    from ouroboros import subagents as sub

    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")

    # EXPLICIT harness, no route: blocked, and the block is one predicate.
    task, dispatch = _dispatched(tmp_path / "pinned", monkeypatch, executor="harness")
    assert dispatch.executor == "blocked" and dispatch.blocked is True
    assert dispatch.route == ""
    assert task["effective_executor"] == "blocked"
    assert dispatch.delta.reason == "harness_not_configured"
    assert dispatch.delta.reduced is True

    # AUTO with no route: native, and quiet — nothing was asked for.
    auto_task, auto_dispatch = _dispatched(tmp_path / "auto", monkeypatch, executor="auto")
    assert auto_dispatch.executor == "native" and auto_dispatch.blocked is False
    assert auto_task["capability_delta"]["reduced"] is False

    # The whole rule table, at the SURVIVING resolver (p34's typed table — H1):
    # `route` is a DelegationRoute or None, and the outcome is a typed record.
    route_a = sub.DelegationRoute(route_id="route-a")

    def row(requested, route):
        res = sub.resolve_subagent_executor(requested, route=route)
        return res.executor, res.reason

    assert row("harness", None) == ("blocked", "harness_not_configured")
    assert row("harness", route_a) == ("harness", "harness_ready")
    assert row("auto", None) == ("native", "harness_not_configured")
    assert row("auto", route_a) == ("harness", "harness_ready")
    assert row("native", None) == ("native", "requested_native")
    # An exhausted subscription window blocks a PIN and falls auto back, loudly.
    spent = sub.resolve_subagent_executor("harness", route=route_a, reset_at="2030-01-01T00:00:00Z")
    assert (spent.executor, spent.reason) == ("blocked", "subscription_window_exhausted")
    assert sub.resolve_subagent_executor("auto", route=route_a, reset_at="X").executor == "native"
    # `blocked` is a resolution OUTCOME, never a request a parent may make.
    assert "blocked" not in sub.SUBAGENT_EXECUTORS

def test_a_blocked_pin_ends_the_task_unrun_and_spends_nothing(tmp_path, monkeypatch):
    """The typed blocker has to be ENFORCED, not merely recorded: a record saying
    `blocked` while the child ran natively would be the claim-vs-record defect this
    branch exists to remove — and it would spend the money D28 refuses.

    Drives the REAL agent path (`_handle_task_scoped`) with the tool loop stubbed, and
    asserts the loop was never entered, the result is typed, and the child that only
    asked for `auto` still runs."""
    from ouroboros import agent as agent_module
    from ouroboros.agent import Env, OuroborosAgent

    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    monkeypatch.setattr("ouroboros.agent.build_llm_messages", lambda **kwargs: ([], {}))

    calls: list = []

    def _never(**kwargs):
        calls.append(kwargs)
        return "the model was called", {}, {"reasoning_notes": [], "tool_calls": []}

    monkeypatch.setattr(agent_module, "run_llm_loop", _never)

    repo = tmp_path / "repo"
    repo.mkdir()
    drive = tmp_path / "drive"
    drive.mkdir()

    pinned = _enqueue_through_supervisor(tmp_path / "sched", monkeypatch, executor="harness")
    pinned.update({"id": "pinned1", "chat_id": 1, "drive_root": str(drive)})

    agent = OuroborosAgent(Env(repo_dir=repo, drive_root=drive))
    events = agent._handle_task_scoped(dict(pinned))

    assert calls == [], "a pinned child must never reach the model"

    # The terminal event stream: typed, and zero spend.
    done = [evt for evt in events if str(evt.get("type") or "") == "task_done"]
    assert done, events
    assert done[-1].get("reason_code") == "subagent_executor_unavailable", done[-1]
    assert float(done[-1].get("cost_usd") or 0.0) == 0.0

    # The durable record is the authority, and it states the block plus WHY.
    import json as _json

    result_path = drive / "task_results" / "pinned1.json"
    # The record carries the "⚠️ EXECUTOR_UNAVAILABLE" prose, whose U+FE0F tail is
    # undefined in cp1252 — so a locale-bound read dies on a Windows runner while the
    # production reader (`utils.read_json_dict`) has always named utf-8. The encoding
    # is stated here for the same reason, and the hostility is pinned below so a
    # future edit cannot quietly drop back to an ASCII fixture and hide the class.
    with pytest.raises(UnicodeDecodeError):
        result_path.read_text(encoding="cp1252")
    record = _json.loads(result_path.read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["reason_code"] == "subagent_executor_unavailable"
    assert record["effective_executor"] == "blocked"
    assert float(record.get("cost_usd") or 0.0) == 0.0
    assert "EXECUTOR_UNAVAILABLE" in str(record.get("result") or "")
    # The parent is told in prose too, naming the alternative that DOES spend.
    told = [evt for evt in events if str(evt.get("type") or "") == "send_message"]
    assert told and "executor='auto'" in str(told[-1].get("text") or "")

    # The same child asking only for `auto` DOES run: the blocker is scoped to the
    # explicit pin, not to "no route configured".
    auto = _enqueue_through_supervisor(tmp_path / "sched2", monkeypatch, executor="auto")
    auto.update({"id": "auto1", "chat_id": 1, "drive_root": str(drive)})
    agent._handle_task_scoped(dict(auto))
    assert len(calls) == 1, "an auto child must still run natively"

def test_a_lane_with_no_configured_slot_reports_the_model_it_really_got(tmp_path, monkeypatch):
    """Asking for Heavy on an install with no Heavy slot runs the Main model. The
    resolution used to keep calling that `effective_lane="heavy"` — the record
    claimed a strength nobody configured, while `_use_local_for_lane` had known
    the truth all along and kept it to itself."""
    from ouroboros.agent import capability_delta_prompt_block

    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "")
    task, dispatch = _dispatched(tmp_path / "noheavy", monkeypatch, model_lane="heavy")
    assert task["effective_model_lane"] == "main"
    assert task["capability_delta"]["reason"] == "lane_slot_unavailable=heavy"
    assert "model_lane heavy->main" in capability_delta_prompt_block(dispatch)

    # ...and a configured Heavy slot is honored silently.
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "provider::strong")
    task, dispatch = _dispatched(tmp_path / "heavy", monkeypatch, model_lane="heavy")
    assert task["effective_model_lane"] == "heavy"
    assert capability_delta_prompt_block(dispatch) == ""

def test_a_route_effort_ceiling_is_disclosed_at_dispatch(tmp_path, monkeypatch):
    """The learned per-route effort ceiling reached `llm_usage` and nothing else, so a
    child ran below the effort the owner configured and nobody was told. Effort is no
    longer requestable, so what the ceiling is measured against is the DERIVED effort
    — the owner's setting for this task type, which is still the owner's business.

    The STORED effort stays that derived value on purpose: the dispatcher re-clamps
    per model, and a fallback route with a wider band must not inherit this route's
    ceiling."""
    from ouroboros.agent import capability_delta_prompt_block
    from ouroboros.llm import LLMClient

    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "provider::cheap")
    monkeypatch.setenv("OUROBOROS_EFFORT_TASK", "max")
    monkeypatch.setitem(LLMClient._EFFORT_CEILING_CACHE, "provider::cheap", "low")

    task, dispatch = _dispatched(tmp_path / "ceiling", monkeypatch, model_lane="light")
    delta = task["capability_delta"]
    assert (delta["derived_effort"], delta["effective_effort"]) == ("max", "low")
    assert delta["reason"] == "route_effort_ceiling=low"
    assert "effort max->low" in capability_delta_prompt_block(dispatch)
    assert task["reasoning_effort"] == "max"

    # An effort inside the band is not a delta.
    monkeypatch.setenv("OUROBOROS_EFFORT_TASK", "low")
    _task, dispatch = _dispatched(tmp_path / "inband", monkeypatch, model_lane="light")
    assert capability_delta_prompt_block(dispatch) == ""

def test_a_legacy_parent_lane_does_not_break_scheduling_or_dispatch(tmp_path, monkeypatch):
    """Inheritance reads the PARENT's stored lane, and durable data outlives the schema
    that wrote it. A pre-v6.39 `code` on the parent's record must not turn every child
    it spawns into an uncaught ValueError — the public schema stays strict about what
    a CALLER may ask for, which is a different question."""
    from ouroboros.tools.control import _schedule_task

    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    task, _ = _dispatched(tmp_path / "legacy", monkeypatch, parent_lane="code")
    assert task["effective_model_lane"] == "main"

    # A caller asking for it directly is still refused.
    ctx = _light_lane_ctx(tmp_path / "asked", monkeypatch)
    assert "model_lane must be one of" in _schedule_task(
        ctx, objective="o", expected_output="e", model_lane="code")

def test_the_completion_envelope_is_built_from_the_same_mapping_as_the_scheduler():
    """The envelope a RAN child publishes and the one the scheduler wrote are twins,
    and they were two field-by-field mappings in two modules. They had already
    drifted: the completion side re-derived the effective-lane fallback as a
    hardcoded `light`, so a record missing that field came back describing a lane
    the resolver would never produce — and the delta axes had to be added twice."""
    from ouroboros.subagents import envelope_from_task

    delta = {"requested_executor": "harness", "effective_executor": "native", "reduced": True}
    env = envelope_from_task(
        {"id": "c1", "model_lane": "", "requested_executor": "harness",
         "effective_model_lane": "heavy", "effective_executor": "native",
         "executor_route": "", "tool_profile": "acting_subagent",
         "capability_delta": delta},
        status="completed", usage={"rounds": 3},
    )
    assert env["effective_lane"] == "heavy"
    assert env["executor"] == "harness"
    assert env["effective_executor"] == "native"
    assert env["tool_profile"] == "acting_subagent"
    assert env["capability_delta"]["reduced"] is True
    assert env["usage"]["rounds"] == 3

    # A record with NO resolution on it describes no lane, rather than substituting
    # one: "not dispatched" and "ran on the lane of record" are different facts.
    assert envelope_from_task({"id": "c2"}, status="requested")["effective_lane"] == ""

def test_lane_rank_is_the_only_lane_ordering(tmp_path):
    """One comparison decides "weaker than what was asked" for every axis. Effort
    already had `config.effort_rank`; the lane had nothing, so the question was
    simply never asked. `auto` has no rank — it is a request to inherit, not a
    strength, so the thing an effective lane is measured against is the lane the
    request RESOLVED FROM, never the literal `auto`."""
    from ouroboros.subagents import LANE_STRENGTH, lane_is_weaker, lane_rank

    assert LANE_STRENGTH == ("light", "main", "heavy")
    assert lane_rank("light") < lane_rank("main") < lane_rank("heavy")
    assert lane_rank("auto") == -1
    assert lane_rank("code") == -1
    assert lane_is_weaker("main", "heavy") is True
    assert lane_is_weaker("heavy", "main") is False
    # Nothing can rank below `auto`, which is why comparing against it was a
    # disclosure that could never fire.
    assert lane_is_weaker("light", "auto") is False

def test_intended_lane_is_the_one_owner_of_what_a_request_means(tmp_path):
    """`auto` means "the parent's lane". Two places need that answer and neither may
    own it: the resolution measures the effective lane against it, and the ADMISSION
    gate for a `require_lane` constraint runs before the child is dispatched, so it
    cannot ask what lane the child ended up on. One predicate, two readers."""
    from ouroboros.subagents import LANE_OF_RECORD, intended_lane, resolve_subagent_lane

    assert intended_lane("auto", "heavy") == "heavy"
    assert intended_lane("light", "heavy") == "light"
    assert intended_lane("auto", "") == LANE_OF_RECORD
    # Stored garbage on either side must not make a child unschedulable.
    assert intended_lane("auto", "code") == LANE_OF_RECORD
    assert intended_lane("code", "heavy") == "heavy"
    # The resolution asks this predicate rather than re-deriving it.
    assert resolve_subagent_lane("auto", parent_lane="heavy").resolved_from == "heavy"

def test_an_inherited_lane_that_lands_on_main_is_as_loud_as_an_explicit_one(tmp_path, monkeypatch):
    """The headline DEFAULT was the one case the headline INVARIANT could not see.

    The delta compared the effective lane against the literal request. On the
    inheritance path the request is `auto`, whose rank is -1, so no effective lane
    could ever rank below it: a child that inherited Heavy and really ran Main was
    silent, while the identical situation reached through an EXPLICIT `heavy` was
    loud. The comparison runs against the lane the request RESOLVED FROM."""
    from ouroboros.agent import capability_delta_prompt_block

    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "")

    task, dispatch = _dispatched(tmp_path / "inherited-noheavy", monkeypatch, parent_lane="heavy")
    delta = task["capability_delta"]
    assert (delta["requested_lane"], delta["resolved_lane"], delta["effective_lane"]) == (
        "auto", "heavy", "main")
    assert delta["reduced"] is True
    assert delta["reason"] == "lane_slot_unavailable=heavy"
    assert task["effective_model_lane"] == "main"
    # The block names the lane that was INHERITED, not the bare `auto` request —
    # "auto->main" would read as a parent that asked for nothing.
    assert "model_lane auto(inherited heavy)->main" in capability_delta_prompt_block(dispatch)

    # An inherited lane the install CAN provide still takes nothing away.
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "provider::strong")
    _ok, quiet = _dispatched(tmp_path / "inherited-ok", monkeypatch, parent_lane="heavy")
    assert capability_delta_prompt_block(quiet) == ""

def test_the_effort_the_delta_reports_is_the_effort_the_dispatcher_will_run(tmp_path, monkeypatch):
    """`effective_effort` claims to be "the effort this route will actually run", and
    it was derived from the route's CEILING alone — half of the `[floor, ceiling]`
    band `_clamp_effort_for_model` clamps to. A route with a learned floor (v6.73.2,
    endpoints where reasoning is mandatory) therefore had the delta report the
    derived effort verbatim while the call ran something else. Both go through one
    body, and a floor that RAISES the effort is reported honestly without being
    called a reduction."""
    from ouroboros.agent import capability_delta_prompt_block
    from ouroboros.llm import LLMClient

    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "provider::cheap")
    monkeypatch.setenv("OUROBOROS_EFFORT_TASK", "none")
    monkeypatch.setitem(LLMClient._EFFORT_FLOOR_CACHE, "provider::cheap", "low")
    monkeypatch.setitem(LLMClient._EFFORT_FLOOR_LOADED, "provider::cheap", float("inf"))

    task, dispatch = _dispatched(tmp_path / "floor", monkeypatch, model_lane="light")
    delta = task["capability_delta"]
    dispatcher = LLMClient.clamp_effort_for_route("provider::cheap", "none")
    assert dispatcher == "low"
    assert delta["effective_effort"] == dispatcher
    # Being given MORE than was derived is not a reduction, so nothing shouts.
    assert delta["reduced"] is False
    assert capability_delta_prompt_block(dispatch) == ""

    # ...and it must not become a false alarm when ANOTHER axis opens the block: a
    # raised effort inside a real reduction still is not something taken away.
    _both, both_dispatch = _dispatched(
        tmp_path / "floor-and-pin", monkeypatch, model_lane="light", executor="harness")
    block = capability_delta_prompt_block(both_dispatch)
    assert "executor harness->blocked" in block
    assert "effort none->low" not in block

def test_a_require_lane_refusal_states_the_facts_not_the_lane_default(tmp_path):
    """The refusal is read by the model at the exact moment it is deciding how to fix
    a rejected spawn, and it restated a default owned three modules away in
    `subagents`. That copy went stale in v6.87.7, was corrected in v6.87.14 and went
    stale AGAIN in v6.87.26 — it told the model an omitted lane resolves to `light`
    while the code inherits the parent's. It now states only what the reducer holds,
    and it is measured against the INTENDED lane, because admission runs before the
    child is dispatched and the effective lane does not exist yet."""
    from ouroboros.tools.control_delegation import effective_delegation_budget

    row = {"payload": {"constraint_id": "c1", "directive": "require_lane",
                       "scope": {"lane": "heavy"}}}
    refusal = effective_delegation_budget(
        {}, unresolved_constraints=[row], role="critic",
        requested_lane="auto", intended_lane="main")
    assert refusal.ok is False
    assert refusal.reason_code == "delegation_constraint_require_lane"
    # No claim about what an omitted lane means — that rule is not this module's.
    assert "v6.87" not in refusal.detail
    # The facts it does hold, and a REACHABLE remedy: "ask for the lane explicitly"
    # is not one when the install has no such slot, so the constraint has to give.
    assert "'heavy'" in refusal.detail and "'auto'" in refusal.detail and "'main'" in refusal.detail
    assert "override_delegation_constraint('c1')" in refusal.detail

    # An omitted lane that INHERITS the required one is admitted: the gate reads the
    # same predicate the resolution does, so it cannot disagree with it about `auto`.
    from ouroboros.subagents import intended_lane

    ok = effective_delegation_budget(
        {}, unresolved_constraints=[row], role="critic", requested_lane="auto",
        intended_lane=intended_lane("auto", "heavy"))
    assert ok.ok is True

def test_the_admission_gate_asks_the_predicate_rather_than_the_raw_request(tmp_path, monkeypatch):
    """The WIRING, not the reducer. The reducer above is pure and can be handed
    anything; what decides whether a real spawn is admitted is what the SUPERVISOR
    passes it. Handing it the raw request means `auto` is compared verbatim against a
    required lane, so a Heavy parent whose omitted-lane child INHERITS Heavy — the
    v6.87.26 default, and the common case — is rejected for asking for the very lane
    the constraint demands."""
    import ouroboros.task_tree_ledger as ledger

    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "provider::strong")
    monkeypatch.setattr(
        ledger, "open_delegation_constraints",
        lambda _root: [{"payload": {"constraint_id": "c1", "directive": "require_lane",
                                    "scope": {"lane": "heavy"}}}])

    task = _enqueue_through_supervisor(tmp_path, monkeypatch, parent_lane="heavy")
    assert task["parent_model_lane"] == "heavy"
    assert task["requested_model_lane"] == "auto"

def test_the_parent_sees_the_reduction_when_it_reads_the_childs_answer(tmp_path):
    """The TERMINAL parent-facing disclosure, and since v6.87.28 the only one: the
    reduction is not known until the child is dispatched, so a scheduling result
    cannot carry it. This is also the moment the parent cares most — it is reading
    the ANSWER that decides whether to trust a weaker result."""
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.control import _get_task_result

    ctx = _scheduling_ctx(tmp_path / "readback")
    reduced = {"requested_lane": "heavy", "resolved_lane": "heavy", "effective_lane": "main",
               "requested_executor": "harness", "effective_executor": "native",
               "reason": "lane_slot_unavailable=heavy", "reduced": True}
    write_task_result(tmp_path / "readback", "child1", "completed",
                      result="done", capability_delta=reduced)
    out = _get_task_result(ctx, "child1")
    assert "capability_delta" in out
    assert "lane_slot_unavailable=heavy" in out

    # A delta that took nothing away and ignored nothing is noise in every payload.
    write_task_result(tmp_path / "readback", "child2", "completed",
                      result="done", capability_delta={**reduced, "reduced": False})
    assert "capability_delta" not in _get_task_result(ctx, "child2")

    # ...but an IGNORED legacy field is something to say, even without a reduction.
    write_task_result(tmp_path / "readback", "child3", "completed", result="done",
                      capability_delta={"reduced": False, "legacy_note": "reasoning_effort='max' ignored"})
    assert "legacy_note" in _get_task_result(ctx, "child3")

def test_the_batch_absorb_discloses_the_reduction_too(tmp_path):
    """The TWIN of the single-child read, and the one a fan-out parent actually uses.

    A parent absorbs children through two surfaces: `get_task_result`/`wait_task` read
    one child in full, and `wait_tasks` projects a batch compactly — which is the
    right tool for "five independent children scheduled in one burst" by its own tool
    description. The delta reached the first and not the second, so the parent most
    likely to have several weakened children was the one told about none of them. The
    compact projection is a DISCLOSED omission of forensics; a capability reduction is
    not forensics, it is what decides how far to trust the answer."""
    import json as _json

    from ouroboros.task_results import write_task_result
    from ouroboros.tools.control import _get_task_result, _wait_for_tasks

    ctx = _scheduling_ctx(tmp_path / "batch")
    reduced = {"requested_lane": "heavy", "resolved_lane": "heavy", "effective_lane": "main",
               "requested_executor": "harness", "effective_executor": "native",
               "reason": "lane_slot_unavailable=heavy", "reduced": True}
    root = tmp_path / "batch"
    write_task_result(root, "c1", "completed", result="done", capability_delta=reduced)
    write_task_result(root, "c2", "completed", result="done",
                      capability_delta={**reduced, "reduced": False, "legacy_note": ""})

    batch = _json.loads(_wait_for_tasks(ctx, ["c1", "c2"], timeout_sec=1))["tasks"]
    assert batch["c1"]["capability_delta"]["reason"] == "lane_slot_unavailable=heavy"
    # The same predicate decides both surfaces, so they cannot disagree about which
    # deltas are worth saying.
    assert "capability_delta" not in batch["c2"]
    assert ("capability_delta" in _get_task_result(ctx, "c1")) is True
    assert ("capability_delta" in _get_task_result(ctx, "c2")) is False

def test_one_resolution_writes_every_derived_field(tmp_path, monkeypatch):
    """"Not two resolvers, not two records." Every derived field on a child's record
    comes from `SubagentDispatch.record_fields()`, so an added axis is one edit rather
    than a field-by-field mapping repeated in four modules that drift apart a release
    later — and no OTHER surface may mint one."""
    from ouroboros.agent import resolve_dispatch_axes
    from ouroboros.subagents import SUBAGENT_INTENT_FIELDS, resolve_subagent_dispatch

    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    task = {"id": "c1", "type": "task", "delegation_role": "subagent",
            "requested_model_lane": "auto", "parent_model_lane": "heavy",
            "requested_executor": "auto"}
    before = dict(task)
    dispatch = resolve_dispatch_axes(task)

    derived = dispatch.record_fields()
    assert set(derived) == {
        "effective_model_lane", "model", "use_local_model", "reasoning_effort",
        "effective_executor", "executor_route", "tool_profile", "capability_delta"}
    # Nothing derived leaks into the intent half, and nothing intended is rewritten.
    assert not set(derived) & set(SUBAGENT_INTENT_FIELDS)
    for key in SUBAGENT_INTENT_FIELDS:
        assert task.get(key) == before.get(key)

    # The resolution is a pure function of the record: asking twice answers twice.
    assert resolve_subagent_dispatch(before, task_type="task").record_fields() == derived

    # A task that is not a delegated child is not resolved at all.
    root = {"id": "r1", "type": "task"}
    assert resolve_dispatch_axes(root) is None
    assert "capability_delta" not in root

def test_queue_snapshot_projects_every_scheduling_intent_field(monkeypatch, tmp_path):
    """R2-3 (F9 delta): a PENDING child's queue-snapshot row is all a restarted
    supervisor has, so an intent field missing from the projection is silently
    dropped across restart — `required_model_lane` was, re-opening the
    auto+harness⇒light default over a gate-verified lane. Walk
    SUBAGENT_INTENT_FIELDS against the REAL projection so no future intent
    field can be dropped the same way."""
    from supervisor import state as state_mod
    import json as _json

    from ouroboros.subagents import SUBAGENT_INTENT_FIELDS
    from supervisor import queue as queue_mod

    pending: list = []
    running: dict = {}
    queue_mod.init_queue_refs(pending, running, {"value": 0})
    monkeypatch.setattr(state_mod, "QUEUE_SNAPSHOT_PATH",
                        tmp_path / "queue_snapshot.json")
    task = {"id": "t-intent-pin", "type": "task"}
    sentinels = {name: f"sentinel-{i}" for i, name in enumerate(SUBAGENT_INTENT_FIELDS)}
    task.update(sentinels)
    pending.append(task)
    assert queue_mod.persist_queue_snapshot(reason="intent-field-pin") is True
    snapshot = _json.loads((tmp_path / "queue_snapshot.json").read_text(encoding="utf-8"))
    row = snapshot["pending"][0]["task"]
    for name, value in sentinels.items():
        assert row.get(name) == value, (
            f"scheduling intent field {name!r} is missing from the pending "
            "queue-snapshot projection (supervisor/queue.py) — a restart would "
            "silently drop it")

def test_a_stored_auto_parent_lane_is_the_lane_of_record_not_the_cheapest(monkeypatch):
    """A task record can legitimately carry the literal `auto` as its effective lane —
    the supervisor falls that field back to the REQUESTED lane, which is `auto`
    whenever a task was queued without a resolved one. Its children read it as
    `parent_lane`, and `auto` is not a strength: unhandled it reached `_lane_model`
    as an unknown lane, whose fall-through was the LIGHT model. The child dropped to
    the cheapest route on this install, silently, and called the lane `auto`."""
    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "provider::cheap")

    res = subagents.resolve_subagent_lane("auto", parent_lane="auto")
    assert res.effective_lane == subagents.LANE_OF_RECORD == "main"
    assert res.model == "provider::main"
    # The fall-through itself: an unknown lane is "no lane on record", not Light.
    assert subagents._lane_model("code") == "provider::main"

def test_prompt_block_omits_the_broken_below_phrase_on_an_executor_only_delta():
    """reduced=True with NO disclosable axis is the auto-fallback case (the axis
    renderer deliberately keeps a non-pinned executor out of the list): the block
    used to render "You are running BELOW what your parent asked for: " over an
    empty list — a broken sentence duplicating dispatch_executor_note's job."""
    from types import SimpleNamespace

    from ouroboros.agent import capability_delta_prompt_block

    class _Delta:
        def as_dict(self):
            return {
                "requested_lane": "auto", "resolved_lane": "main",
                "effective_lane": "main", "derived_effort": "",
                "effective_effort": "", "requested_executor": "auto",
                "effective_executor": "native",
                "reason": "subscription_window_exhausted",
                "reduced": True, "legacy_note": "",
            }

    block = capability_delta_prompt_block(
        SimpleNamespace(delta=_Delta(), executor_resolution=None))
    assert "BELOW what your parent asked" not in block
    assert block == ""  # nothing else to say either: the executor note owns it

def _harness_ready_dispatch(monkeypatch):
    """Force the executor axis to a healthy harness route without a live daemon."""
    route = subagents.DelegationRoute(route_id="codex")
    monkeypatch.setattr(
        subagents, "dispatch_executor_resolution",
        lambda task: subagents.resolve_subagent_executor("auto", route=route),
    )

def test_auto_lane_on_harness_executor_defaults_to_light_by_policy(monkeypatch):
    """B2 (poltergeist phase B): a harness-dispatched child whose request said
    `auto` is a NANNY — its own rounds are custody chores around a $0 delegated
    run, so the dispatch policy resolves it to the LIGHT lane instead of the
    parent's expensive lane, and the provenance says the POLICY answered."""
    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "provider::cheap")
    _harness_ready_dispatch(monkeypatch)

    dispatch = subagents.resolve_subagent_dispatch(
        {"id": "c1", "type": "task", "requested_model_lane": "auto",
         "parent_model_lane": "main"},
        task_type="task",
    )
    assert dispatch.executor == "harness"
    assert dispatch.lane.effective_lane == "light"
    assert dispatch.lane.model == "provider::cheap"
    assert dispatch.lane.provenance == "policy"
    assert dispatch.delta.as_dict()["lane_provenance"] == "policy"
    # Not a reduction relative to itself: the policy IS the resolved baseline.
    assert dispatch.lane.reduced is False

def test_explicit_lane_always_wins_over_the_harness_policy(monkeypatch):
    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "provider::strong")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "provider::cheap")
    _harness_ready_dispatch(monkeypatch)

    dispatch = subagents.resolve_subagent_dispatch(
        {"id": "c2", "type": "task", "requested_model_lane": "heavy"},
        task_type="task",
    )
    assert dispatch.executor == "harness"
    assert dispatch.lane.effective_lane == "heavy"
    assert dispatch.lane.model == "provider::strong"
    assert dispatch.lane.provenance == "requested"

def test_a_required_lane_wins_over_the_harness_policy_default(monkeypatch):
    """F9 (sol #1) admission→dispatch consistency: a child ADMITTED under a
    satisfied `require_lane` constraint (auto request, parent on the required
    lane) carries `required_model_lane` on its record — and the dispatch policy
    default (auto+harness ⇒ light) must NOT apply over it. With the policy
    suppressed, `auto` inherits the parent's lane, which is exactly the lane the
    gate verified; the provenance honestly says "inherited"."""
    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "provider::strong")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "provider::cheap")
    _harness_ready_dispatch(monkeypatch)

    dispatch = subagents.resolve_subagent_dispatch(
        {"id": "c-req", "type": "task", "requested_model_lane": "auto",
         "parent_model_lane": "heavy", "required_model_lane": "heavy"},
        task_type="task",
    )
    assert dispatch.executor == "harness"
    assert dispatch.lane.effective_lane == "heavy"
    assert dispatch.lane.model == "provider::strong"
    assert dispatch.lane.provenance == "inherited"

    # Stored garbage in the field is ignored — the policy applies as usual.
    garbage = subagents.resolve_subagent_dispatch(
        {"id": "c-junk", "type": "task", "requested_model_lane": "auto",
         "parent_model_lane": "heavy", "required_model_lane": "warp-lane"},
        task_type="task",
    )
    assert garbage.lane.effective_lane == "light"
    assert garbage.lane.provenance == "policy"

def test_preflight_native_fallback_reresolves_without_the_harness_policy(monkeypatch):
    """F10 (sol #2, probe `native light policy`): a harness dispatch falsified at
    the toolset preflight falls back to NATIVE — and must not stay on the
    policy-light lane/cheap model the harness resolution chose. The fallback
    re-resolves lane/model/effort as a native dispatch would (parent
    inheritance), and the record, delta and envelope all describe it."""
    from types import SimpleNamespace

    from ouroboros.agent import preflight_delegate_visibility, resolve_dispatch_axes

    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "provider::strong")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "provider::cheap")
    _harness_ready_dispatch(monkeypatch)

    task = {"id": "c-fb", "type": "task", "delegation_role": "subagent",
            "requested_model_lane": "auto", "parent_model_lane": "heavy",
            "requested_executor": "auto"}
    dispatch = resolve_dispatch_axes(task)
    assert dispatch.lane.effective_lane == "light"  # the harness policy, pre-preflight
    assert task["model"] == "provider::cheap"

    tools = SimpleNamespace(available_tools=lambda: ["read_file", "web_search"])
    amended, changed = preflight_delegate_visibility(tools, task, dispatch)
    assert changed is True
    assert amended.executor == "native"
    # Lane and model re-resolved WITHOUT the harness policy: parent inheritance.
    assert amended.lane.effective_lane == "heavy"
    assert amended.lane.model == "provider::strong"
    assert amended.lane.provenance == "inherited"
    # Every stamped surface tells the re-resolved story.
    assert task["effective_model_lane"] == "heavy"
    assert task["model"] == "provider::strong"
    assert task["effective_executor"] == "native"
    assert task["capability_delta"]["effective_lane"] == "heavy"
    assert task["capability_delta"]["lane_provenance"] == "inherited"
    assert "delegate_tools_invisible" in task["capability_delta"]["reason"]
    assert task["capability_delta"]["reduced"] is True
    assert task["subagent_envelope"]["effective_lane"] == "heavy"
    assert task["subagent_envelope"]["model"] == "provider::strong"
    assert task["subagent_envelope"]["effective_executor"] == "native"

def test_native_child_keeps_plain_inheritance(monkeypatch):
    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "provider::strong")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "")

    dispatch = subagents.resolve_subagent_dispatch(
        {"id": "c3", "type": "task", "requested_model_lane": "auto",
         "parent_model_lane": "heavy"},
        task_type="task",
    )
    assert dispatch.executor == "native"
    assert dispatch.lane.effective_lane == "heavy"
    assert dispatch.lane.provenance == "inherited"

def test_policy_light_with_an_empty_light_slot_lands_main_and_says_so(monkeypatch):
    """The provenance names the DECISION source even when the slot outcome moves
    the effective lane: policy said light, no light slot exists, the model is
    Main — and the record must carry both facts, not blend them."""
    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.delenv("OUROBOROS_MODEL_LIGHT", raising=False)
    _harness_ready_dispatch(monkeypatch)

    dispatch = subagents.resolve_subagent_dispatch(
        {"id": "c4", "type": "task", "requested_model_lane": "auto"},
        task_type="task",
    )
    assert dispatch.lane.provenance == "policy"
    assert dispatch.lane.resolved_from == "light"
    assert dispatch.lane.effective_lane == "main"
    assert dispatch.lane.model == "provider::main"

def test_switch_model_never_rewrites_the_dispatch_lane_record(monkeypatch, tmp_path):
    """B2 acceptance-model provenance: the nanny raising itself for an acceptance
    round is a ToolContext override (visible per-round in llm_usage rows), never a
    rewrite of the durable dispatch resolution — the record keeps saying which
    lane the child was DISPATCHED on."""
    from ouroboros.tools.control import _switch_model
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "provider::cheap")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    record = {"effective_model_lane": "light", "model": "provider::cheap",
              "capability_delta": {"lane_provenance": "policy"}}
    ctx.task_metadata = dict(record)

    out = _switch_model(ctx, model="provider::main")
    assert "OK: switching" in out
    assert ctx.active_model_override == "provider::main"
    # The durable dispatch record is untouched — acceptance-round provenance is
    # read from llm_usage (each round carries the REAL model), not from here.
    assert {k: ctx.task_metadata[k] for k in record} == record
