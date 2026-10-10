"""The admitted budget wrap-up against the send the main loop REALLY makes.

Every earlier parity pin rebuilt the "actual" candidate by calling the prospective
builder's own builder with the prospective builder's own arguments, so a wire key
only the real send carries (``stream``) could never fail it — and in production it
cost three tasks their final answer in one night. This module drives
``call_llm_with_retry`` itself and meets the physical request at the executor seam,
the only place where the two payloads can honestly be compared."""
from __future__ import annotations

import copy
import queue
from types import SimpleNamespace

import pytest

from ouroboros import llm as llm_module
from ouroboros import task_pacing, usage_accounting
from ouroboros.contracts.task_contract import normalize_budget_profile
from ouroboros.llm import LLMClient
from ouroboros.llm_claudexor import cache_key_for_model
from ouroboros.llm_messages import HOST_CONTEXT_NOTICE_BEFORE_TASK, STABLE_PREFIX_BLOCKS_KEY
from ouroboros.loop import _check_budget_limits
from ouroboros.loop_llm_call import call_llm_with_retry
from ouroboros.task_pacing import main_loop_wire_options

from tests.test_tree_cost_ceiling import _ctx, _patch_execute_candidate

_IDENTITY = ("model", "provider", "candidate_raw_sha256", "candidate_raw_size_bytes")
_MESSAGES = [{"role": "system", "content": "policy"}, {"role": "user", "content": "wrap up"}]
# Main Max with a handbook: every stable item precedes the changing evidence.
_DECLARED_MESSAGES = [
    {"role": "system", "content": [
        {"type": "text", "text": "policy", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "handbook", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "memory", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "evidence"},
    ], STABLE_PREFIX_BLOCKS_KEY: 3},
    {"role": "user", "content": "wrap up"},
]
_TOOLS = [{"type": "function", "function": {
    "name": "probe", "description": "probe", "parameters": {"type": "object", "properties": {}},
}}]
_ROUTES = [
    ("openai::gpt-test", {"OPENAI_API_KEY": "unused"}),
    ("openai/gpt-test", {"OPENROUTER_API_KEY": "unused"}),
    ("anthropic/claude-test", {"OPENROUTER_API_KEY": "unused"}),
    ("anthropic::claude-test", {"ANTHROPIC_API_KEY": "unused"}),
]
_OPENAI_FAMILY_ROUTES = _ROUTES[:2]


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """pytest is not a worker: keep capability discovery and pricing off the network."""
    monkeypatch.setattr(LLMClient, "_SUPPORTED_PARAMS_FETCHED", True, raising=False)
    monkeypatch.setattr("ouroboros.pricing._fetch_live_rows", lambda *_a, **_kw: {})


class _Captured(Exception):
    """Stops the real send once the physical request exists."""


@pytest.mark.parametrize("model,env", _ROUTES)
def test_prospective_wrapup_candidate_is_the_candidate_the_main_loop_sends(monkeypatch, tmp_path, model, env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    captured = {}

    def execute(request, send, before_dispatch):
        captured["request"] = request
        raise _Captured()

    _patch_execute_candidate(monkeypatch, llm_module, execute)
    client = LLMClient(api_key="unused")
    logs = tmp_path / "logs"
    logs.mkdir()
    with usage_accounting.usage_scope(usage_accounting.UsageScope(
        drive_root=tmp_path, task_id="parity", root_task_id="parity",
    )):
        prospective = task_pacing.prospective_wrapup_attempt_request(
            llm=client, messages=_MESSAGES, model=model, reasoning_effort="high",
            tools=_TOOLS, cache_affinity=cache_key_for_model(model),
        )
        try:
            call_llm_with_retry(client, _MESSAGES, model, _TOOLS, "high", 1, logs, "parity", 1,
                                queue.Queue(), {}, initial_messages=_MESSAGES)
        except _Captured:
            pass
    assert "request" in captured, "the real send never reached the physical executor"
    assert {key: getattr(captured["request"], key) for key in _IDENTITY} == {
        key: getattr(prospective, key) for key in _IDENTITY}


@pytest.mark.parametrize("model,env", _OPENAI_FAMILY_ROUTES)
def test_declared_system_prefix_split_is_projected_once_inside_the_candidate_builder(monkeypatch, tmp_path, model, env):
    """The prospective wrap-up candidate and the real send agree on the SPLIT copy, and
    the split happens INSIDE ``_build_remote_kwargs`` (llm_openai_compatible.py, the
    ``openai_family_route`` block before the direct/OpenRouter branch split): the spy sees
    the canonical declared system ENTER the builder and the split copy LEAVE it,
    on the priced build and on the sent build alike. A split that ran earlier (in the
    canonical transcript or ``chat()``) would show an already-split system entering; one
    that ran later (``_finalized_physical_candidate``) would show a whole system leaving;
    removing the block keeps the identity parity but fails every wire-shape assertion."""
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    builds, captured = [], {}
    real_build = LLMClient._build_remote_kwargs

    def spy(self, target, messages, *args, **kwargs):
        entering = copy.deepcopy(messages)
        payload = real_build(self, target, messages, *args, **kwargs)
        builds.append((entering, copy.deepcopy(payload["messages"]), copy.deepcopy(target.get("wire_layout"))))
        return payload

    monkeypatch.setattr(LLMClient, "_build_remote_kwargs", spy)

    def execute(request, send, before_dispatch):
        captured["request"] = request
        raise _Captured()

    _patch_execute_candidate(monkeypatch, llm_module, execute)
    client = LLMClient(api_key="unused")
    logs = tmp_path / "logs"
    logs.mkdir()
    canonical = copy.deepcopy(_DECLARED_MESSAGES)
    with usage_accounting.usage_scope(usage_accounting.UsageScope(
        drive_root=tmp_path, task_id="parity", root_task_id="parity",
    )):
        prospective = task_pacing.prospective_wrapup_attempt_request(
            llm=client, messages=_DECLARED_MESSAGES, model=model, reasoning_effort="high",
            tools=_TOOLS, cache_affinity=cache_key_for_model(model),
        )
        try:
            call_llm_with_retry(client, _DECLARED_MESSAGES, model, _TOOLS, "high", 1, logs, "parity", 1,
                                queue.Queue(), {}, initial_messages=_DECLARED_MESSAGES)
        except _Captured:
            pass
    assert _DECLARED_MESSAGES == canonical, "the canonical transcript is never mutated"
    assert "request" in captured, "the real send never reached the physical executor"
    assert {key: getattr(captured["request"], key) for key in _IDENTITY} == {
        key: getattr(prospective, key) for key in _IDENTITY}
    assert len(builds) == 2, "exactly one priced build and one sent build"
    for entering, wire, layout in builds:
        assert entering == canonical, "the canonical declaration enters the builder unsplit"
        stable = entering[0]["content"][:-1]
        assert entering[0][STABLE_PREFIX_BLOCKS_KEY] == len(stable)
        assert [message["role"] for message in wire] == ["system"] * len(stable) + ["user", "user"]
        assert [message["content"][0]["text"] for message in wire[:len(stable)]] == [block["text"] for block in stable]
        assert all(("cache_control" in message["content"][0]) == model.startswith("openai/") for message in wire[:len(stable)])
        assert wire[-2]["content"] == "[SYSTEM NOTICE]\n" + HOST_CONTEXT_NOTICE_BEFORE_TASK + "\n\nevidence"
        assert wire[-1] == {"role": "user", "content": "wrap up"}
        assert all(STABLE_PREFIX_BLOCKS_KEY not in message for message in wire)
        assert layout == {"system_prefix_split": True, "moved_blocks": 1}
    assert builds[0][1] == builds[1][1], "the priced copy and the sent copy are one wire"


def test_every_wire_option_of_the_send_reaches_the_priced_copy():
    """The one owner: what the send declares is what the prospective builder is handed."""
    options = main_loop_wire_options("openai/gpt-test", allow_server_web_search=True)
    assert options["stream"] is True
    assert set(options) == {"stream", "cache_affinity", "allow_server_web_search", "bypass_response_cache"}


def _completion(text):
    body = {"id": "c1", "object": "chat.completion", "model": "gpt-test", "choices": [{
        "index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    return SimpleNamespace(model_dump=lambda: body)


def _last_fit_rail(monkeypatch, tmp_path, execute):
    """Drive the whole rail: last-fit decision -> admitted candidate -> the real send."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "unused")
    monkeypatch.setattr("ouroboros.loop._loop_tree_accounting", lambda **_k: {"settled_usd": 20.0, "accounted_usd": 20.0})
    # proxy, exact probe, prepared: one fits, two do not; then the FRESH send's own admission fits.
    answers = iter((True, False, True, False, True, False, True))
    monkeypatch.setattr(task_pacing, "wrapup_reservation_fits", lambda **_kwargs: next(answers))
    _patch_execute_candidate(monkeypatch, llm_module, execute)
    logs = tmp_path / "logs"
    logs.mkdir()
    ctx = _ctx(drive_logs=logs, llm=LLMClient(api_key="unused"), active_model="openai/gpt-test")
    ctx.messages = [dict(message) for message in _MESSAGES]
    ceiling = task_pacing.resolve_cost_ceiling(None, normalize_budget_profile({"cost_hard_stop_pct": 50}), root_cap_usd=50.0)
    with usage_accounting.usage_scope(usage_accounting.UsageScope(
        drive_root=tmp_path, task_id="task1", root_task_id="task1",
    )):
        return ctx, _check_budget_limits(ctx, None, ceiling)


def test_the_budget_rail_delivers_the_models_wrapup_as_a_best_effort_answer(monkeypatch, tmp_path):
    from ouroboros.outcomes import EXECUTION_BEST_EFFORT, derive_loop_outcome
    from ouroboros.project_dialogue import OUTCOME_PHASE_HEADLINE, outcome_phase

    attempts, sends = [], []

    def execute(request, send, before_dispatch):
        attempts.append(request)
        before_dispatch(SimpleNamespace(attempt_id=f"a{len(attempts)}", drive_root=tmp_path))   # the identity predicate runs here
        sends.append(request)
        return _completion("Done so far: parts 1-3 verified. Not finished: part 4.")

    ctx, result = _last_fit_rail(monkeypatch, tmp_path, execute)

    assert result is not None
    assert (len(attempts), len(sends)) == (1, 1), "admitted on the first physical attempt: no refusal, no second send"
    text, usage, trace = result
    assert text.startswith("Done so far")
    assert usage["reason_code"] == "budget_exhausted" and usage["_best_effort_extracted"] is True
    axes = derive_loop_outcome(text, usage, trace)["outcome_axes"]
    assert axes["execution"]["status"] == EXECUTION_BEST_EFFORT
    phase = outcome_phase({"status": "completed", "outcome_axes": axes, "reason_code": "budget_exhausted"}, {})
    assert OUTCOME_PHASE_HEADLINE[phase] == "Done with warnings"   # not "Failed": the answer was delivered


def test_a_drifted_admitted_candidate_is_sent_once_more_instead_of_losing_the_answer(monkeypatch, tmp_path):
    """The predicate refuses before dispatch; the refusal must not cost the owner the final answer."""
    import ouroboros.loop as loop_module

    calls, events = [], []
    real_once = loop_module._call_forced_model_once

    def once(ctx, *, initial_messages=None, admitted_request=None, admission=None):
        calls.append(admitted_request is not None)
        if admitted_request is not None:
            raise usage_accounting.PhysicalAttemptPreconditionFailed(
                "physical candidate precondition rejected dispatch", attempt_id="refused-1")
        return real_once(ctx)

    monkeypatch.setattr(loop_module, "_call_forced_model_once", once)
    monkeypatch.setattr(loop_module, "_emit_checkpoint_event",
                        lambda _queue, _task, _logs, data: events.append(data))
    ctx, result = _last_fit_rail(
        monkeypatch, tmp_path, lambda request, send, before_dispatch: _completion("Recovered wrap-up."))

    assert calls == [True, False], "one admitted send, then exactly one ordinary send"
    assert result[0].startswith("Recovered wrap-up") and result[1]["_best_effort_extracted"] is True
    drift = [event for event in events if event.get("checkpoint_kind") == "forced_candidate_drift"]
    assert len(drift) == 1 and drift[0]["refused_attempt_id"] == "refused-1"
    assert drift[0]["admitted"]["model"] and drift[0]["admitted"]["candidate_raw_sha256"]


def test_a_closed_dispatch_window_is_a_deadline_not_drift(monkeypatch, tmp_path):
    import ouroboros.loop as loop_module
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted

    calls = []

    def once(ctx, *, initial_messages=None, admitted_request=None, admission=None):
        calls.append(admitted_request is not None)
        raise PhysicalDispatchInterrupted("dispatch window closed")

    monkeypatch.setattr(loop_module, "_call_forced_model_once", once)
    ctx, result = _last_fit_rail(
        monkeypatch, tmp_path, lambda request, send, before_dispatch: _completion("never reached"))

    assert calls == [True], "a deadline refusal is never retried"
    assert "never reached" not in result[0]


@pytest.mark.parametrize("mode", ["nano", "max"])
def test_the_forced_lookahead_and_send_share_one_measurement_allowance_and_identity(monkeypatch, tmp_path, mode):
    """A10: the priced wrap-up copy and the admitted forced send are measured ONCE under one bound
    Main context, so a rendered Nano gets the window's reply (not the whole ceiling) on both, their
    clock-free identity still matches, and a direct or drift send measures the same bytes to the same
    allowance. Max keeps the ceiling on both (the quiet side)."""
    from dataclasses import replace

    from ouroboros import loop_forced_finalization as forced
    from tests.test_context_fit_v664 import _plan

    monkeypatch.setenv("OPENROUTER_API_KEY", "unused")
    plan = replace(_plan(preferred=mode, window=128_000, known=True), initial_mode=mode)
    owner_ctx = SimpleNamespace(context_fit_plan=plan, active_context_mode=mode, task_metadata={}, model_turn_state=None)
    captured = []

    def execute(request, send, before_dispatch):
        captured.append(request)
        raise _Captured()

    _patch_execute_candidate(monkeypatch, llm_module, execute)
    logs = tmp_path / "logs"
    logs.mkdir()
    messages = [{"role": "system", "content": "policy " * 2_000}, {"role": "user", "content": "wrap up " * 40_000}]
    ctx = _ctx(drive_logs=logs, llm=LLMClient(api_key="unused"), active_model=plan.model, task_type="task",
               tools=SimpleNamespace(_ctx=owner_ctx), messages=[dict(message) for message in messages], tool_schemas=_TOOLS)
    with usage_accounting.usage_scope(usage_accounting.UsageScope(
        drive_root=tmp_path, task_id="task1", root_task_id="task1",
    )):
        request, prepared = task_pacing.prepared_wrapup_candidate(ctx, copy.deepcopy(messages), allow_server_web_search=False)
        assert (owner_ctx._forced_physical_context is not None) and owner_ctx._forced_physical_context.rendered_mode == mode
        forced._call_forced_model_once(ctx, initial_messages=prepared, admitted_request=request)
        assert owner_ctx._forced_physical_context is None  # consumed by the admitted send
        ctx.messages = prepared
        forced._call_forced_model_once(ctx)  # a direct / drift send: its own fresh measurement of the same bytes
    assert len(captured) == 2, "both forced sends reached the physical executor"
    admitted, direct = captured
    assert admitted.max_completion_tokens == direct.max_completion_tokens == request.max_completion_tokens
    assert admitted.candidate_clock_free_sha256 == request.candidate_clock_free_sha256 is not None
    if mode == "nano":
        assert 8_192 < request.max_completion_tokens < 65_536  # the 128K window's room, not the ceiling
    else:
        assert request.max_completion_tokens == 65_536
    # Main's PLANNED allowance is measured on the canonical transcript, the sent one on the sealed
    # candidate (the wire projection differs by a few hundred tokens at most, well inside the slack).
    assert abs(ctx.accumulated_usage["_context_reply_allowance_tokens"] - request.max_completion_tokens) < 1_000


@pytest.mark.parametrize("mode", ["nano", "max"])
def test_the_budget_probe_prices_the_reply_the_window_leaves(monkeypatch, tmp_path, mode):
    """The last-fit probe (a priced copy of the transcript, never sent) is built under its own Main
    measurement: a rendered Nano reserves the reply its 128K window leaves, Max the whole ceiling."""
    from dataclasses import replace

    from tests.test_context_fit_v664 import _plan

    monkeypatch.setenv("OPENROUTER_API_KEY", "unused")
    monkeypatch.setattr("ouroboros.loop._loop_tree_accounting", lambda **_k: {"settled_usd": 20.0, "accounted_usd": 20.0})
    plan = replace(_plan(preferred=mode, window=128_000, known=True), initial_mode=mode)
    # The last-fit probe is an explicit profile's authored rail: the task carries that profile.
    owner_ctx = SimpleNamespace(context_fit_plan=plan, active_context_mode=mode, task_metadata={}, model_turn_state=None,
                                task_contract={"budget_profile": {"cost_hard_stop_pct": 50}})
    priced = []

    def fits(**kwargs):
        if kwargs.get("request") is not None:  # the exact probe: record its price basis and stop here
            priced.append(kwargs["request"])
            raise _Captured()
        return kwargs.get("reservation_count", 1) == 1  # the proxy: one fits, two do not, so the probe runs

    monkeypatch.setattr(task_pacing, "wrapup_reservation_fits", fits)
    logs = tmp_path / "logs"
    logs.mkdir()
    messages = [{"role": "system", "content": "policy " * 2_000}, {"role": "user", "content": "wrap up " * 40_000}]
    ctx = _ctx(drive_logs=logs, llm=LLMClient(api_key="unused"), active_model=plan.model, task_type="task",
               tools=SimpleNamespace(_ctx=owner_ctx), messages=[dict(message) for message in messages], tool_schemas=_TOOLS)
    ceiling = task_pacing.resolve_cost_ceiling(None, normalize_budget_profile({"cost_hard_stop_pct": 50}), root_cap_usd=50.0)
    with usage_accounting.usage_scope(usage_accounting.UsageScope(
        drive_root=tmp_path, task_id="task1", root_task_id="task1",
    )), pytest.raises(_Captured):
        _check_budget_limits(ctx, None, ceiling)
    [probe] = priced
    if mode == "nano":
        assert 8_192 < probe.max_completion_tokens < 65_536  # the window's room, not the ceiling
    else:
        assert probe.max_completion_tokens == 65_536
