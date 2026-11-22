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
from tests._cases import case_folders

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


def test_prospective_wrapup_candidate_is_the_candidate_the_main_loop_sends(subtests, monkeypatch, tmp_path):
    import httpx

    from ouroboros import net_transport
    from ouroboros.opencode_go import GO_CHAT_BASE_URL
    from tests._openai_sse import streamed_reply

    go_route = ("openai-compatible::example-model", {
        "OPENAI_COMPATIBLE_API_KEY": "unused", "OPENAI_COMPATIBLE_BASE_URL": GO_CHAT_BASE_URL,
    })
    real_execute = llm_module._execute_candidate
    case_paths = case_folders(tmp_path)
    for model, env in (*_ROUTES, go_route):
        tmp_path = next(case_paths)
        with subtests.test(model=model, env=env), monkeypatch.context() as monkeypatch:
            for key, value in env.items():
                monkeypatch.setenv(key, value)
            captured = {}

            def execute(request, send, before_dispatch):
                captured["request"] = request
                if model == go_route[0]:
                    return real_execute(request, send, before_dispatch)
                raise _Captured()

            if model == go_route[0]:
                def peer(request):
                    assert str(request.url) == GO_CHAT_BASE_URL + "/chat/completions"
                    assert request.headers["x-opencode-session"]
                    return streamed_reply()

                monkeypatch.setattr(net_transport, "remote_httpx_transport", lambda *_a, **_k: httpx.MockTransport(peer))
                monkeypatch.setattr(net_transport, "env_proxies_configured", lambda: False)
            _patch_execute_candidate(monkeypatch, llm_module, execute)
            client = LLMClient(api_key="unused")
            logs = tmp_path / "logs"
            logs.mkdir()
            with usage_accounting.usage_scope(usage_accounting.UsageScope(
                drive_root=tmp_path, task_id="parity", root_task_id="parity",
            )):
                prospective = task_pacing.prospective_wrapup_attempt_request(
                    llm=client, messages=_MESSAGES, model=model, reasoning_effort="high",
                    tools=_TOOLS, cache_affinity=cache_key_for_model(model), task_id="parity",
                )
                try:
                    reply, _ = call_llm_with_retry(client, _MESSAGES, model, _TOOLS, "high", 1, logs, "parity", 1,
                                                   queue.Queue(), {}, initial_messages=_MESSAGES)
                    if model == go_route[0]:
                        assert reply["content"] == "4" and usage_accounting.last_physical_attempt_capture().state == "settled"
                except _Captured:
                    assert model != go_route[0]
            assert "request" in captured, "the real send never reached the physical executor"
            assert {key: getattr(captured["request"], key) for key in _IDENTITY} == {
                key: getattr(prospective, key) for key in _IDENTITY}


def test_the_sealed_boundary_is_one_marker_in_the_priced_copy_and_the_send_alike(subtests, monkeypatch, tmp_path):
    """A re-seal after this round's facts row puts the boundary on it: the forced final's priced
    copy and the real send render the same marker, and their candidate identity includes it."""
    case_paths = case_folders(tmp_path)
    for model, env in _ROUTES:
        tmp_path = next(case_paths)
        with subtests.test(model=model, env=env), monkeypatch.context() as monkeypatch:
            from ouroboros.context_budget import bound_host_row
            from ouroboros.context_fit import seal_task_transcript
            from ouroboros.loop_messages import CONTEXT_FACTS_NAME
            from tests.test_message_cache_boundary import _marked_texts

            for key, value in env.items():
                monkeypatch.setenv(key, value)
            messages = [*copy.deepcopy(_MESSAGES), {"role": "assistant", "content": "", "tool_calls": [
                {"id": "probe-1", "type": "function", "function": {"name": "probe", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "probe-1", "content": "probe output"},
                bound_host_row(CONTEXT_FACTS_NAME, "[facts] this round")]
            seal_task_transcript(messages, min_prefix_tokens=0)
            assert "cache_control" in messages[-1]
            finalized, captured = [], {}
            real_finalize = LLMClient._normalize_payload_cache_ttl

            def finalize(self, target, payload):
                applied = real_finalize(self, target, payload)
                finalized.append(copy.deepcopy(payload))
                return applied

            monkeypatch.setattr(LLMClient, "_normalize_payload_cache_ttl", finalize)

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
                    llm=client, messages=messages, model=model, reasoning_effort="high",
                    tools=_TOOLS, cache_affinity=cache_key_for_model(model),
                )
                try:
                    call_llm_with_retry(client, messages, model, _TOOLS, "high", 1, logs, "parity", 1,
                                        queue.Queue(), {}, initial_messages=messages)
                except _Captured:
                    pass
            assert "request" in captured, "the real send never reached the physical executor"
            assert {key: getattr(captured["request"], key) for key in _IDENTITY} == {
                key: getattr(prospective, key) for key in _IDENTITY}
            marked = [] if model.startswith("openai::") else ["[facts] this round"]
            assert len(finalized) == 2 and [_marked_texts(payload) for payload in finalized] == [marked, marked]


def test_declared_system_prefix_split_is_projected_once_inside_the_candidate_builder(subtests, monkeypatch, tmp_path):
    """The prospective wrap-up candidate and the real send agree on the SPLIT copy, and
    the split happens INSIDE ``_build_remote_kwargs`` (llm_openai_compatible.py, the
    ``openai_family_route`` block before the direct/OpenRouter branch split): the spy sees
    the canonical declared system ENTER the builder and the split copy LEAVE it,
    on the priced build and on the sent build alike. A split that ran earlier (in the
    canonical transcript or ``chat()``) would show an already-split system entering; one
    that ran later (``_finalized_physical_candidate``) would show a whole system leaving;
    removing the block keeps the identity parity but fails every wire-shape assertion."""
    case_paths = case_folders(tmp_path)
    for model, env in _OPENAI_FAMILY_ROUTES:
        tmp_path = next(case_paths)
        with subtests.test(model=model, env=env), monkeypatch.context() as monkeypatch:
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


@pytest.mark.parametrize("api_surface", ["responses", "chat_completions"])
@pytest.mark.parametrize("learned", [False, True])
def test_wrapup_parity_after_learned_responses_effort_repair(monkeypatch, tmp_path, api_surface, learned):
    import ouroboros.llm_fallback as fallback
    import ouroboros.request_wire_contract as wire
    from ouroboros.loop_llm_call import MAIN_LOOP_MAX_TOKENS
    from ouroboros.request_wire_recovery import (
        finalize_wire_response, note_wire_send_succeeded, plan_wire_retry_from_exception,
        prepare_wire_payload_for_send, request_wire_call_scope,
    )
    from tests.test_request_wire_recovery_phase2b import _Rejected, _capture

    monkeypatch.setenv("OPENAI_API_KEY", "unused")
    monkeypatch.setenv("OPENAI_API_SURFACE", "responses")
    monkeypatch.setattr(wire, "canonical_wire_evidence_root", lambda: tmp_path)
    client, model = LLMClient(api_key="unused"), "openai::gpt-test"
    if learned:
        target = client._resolve_remote_target(model)
        source = client._build_remote_candidate(
            target, _MESSAGES, "high", MAIN_LOOP_MAX_TOKENS, "auto", None, _TOOLS,
            skip_capability_fetch=True, **main_loop_wire_options(model, allow_server_web_search=False))
        # Teach the real durable store using a synthetic refusal and settled success.
        with request_wire_call_scope():
            prepare_wire_payload_for_send(target, source, api_surface="responses")
            repaired = plan_wire_retry_from_exception(_Rejected(
                "Unsupported value: 'reasoning.effort' does not support 'high' with this model. "
                "Supported values are: 'low' and 'medium'."))
            assert repaired["reasoning"] == {"effort": "medium"}
            note_wire_send_succeeded(_capture(repaired, target))
            finalize_wire_response({"role": "assistant", "content": "ok"}, {})
        assert (tmp_path / "state" / wire.REQUEST_WIRE_STATE_FILE).is_file()

    # A Responses repair must not leak onto the same model's Chat surface.
    monkeypatch.setenv("OPENAI_API_SURFACE", api_surface)
    finalized, captured = [], {}
    real_finalize = llm_module._finalized_physical_candidate

    def finalize(*args, **kwargs):
        payload = real_finalize(*args, **kwargs)
        finalized.append(copy.deepcopy(payload))
        return payload

    monkeypatch.setattr(llm_module, "_finalized_physical_candidate", finalize)
    monkeypatch.setattr(fallback, "_finalized_physical_candidate", finalize)

    def execute(request, send, before_dispatch):
        captured["request"] = request
        raise _Captured()

    _patch_execute_candidate(monkeypatch, llm_module, execute)
    logs = tmp_path / "logs"
    logs.mkdir()
    with usage_accounting.usage_scope(usage_accounting.UsageScope(
        drive_root=tmp_path, task_id="parity", root_task_id="parity",
    )):
        prospective = task_pacing.prospective_wrapup_attempt_request(
            llm=client, messages=_MESSAGES, model=model, reasoning_effort="high", tools=_TOOLS)
        try:
            call_llm_with_retry(client, _MESSAGES, model, _TOOLS, "high", 1, logs, "parity", 1,
                                queue.Queue(), {}, initial_messages=_MESSAGES)
        except _Captured:
            pass
    assert "request" in captured, "the real send never reached the physical executor"
    assert len(finalized) == 2, "one preview and one send, with no drift rebuild"
    assert {key: getattr(captured["request"], key) for key in _IDENTITY} == {
        key: getattr(prospective, key) for key in _IDENTITY}
    assert finalized[0] == finalized[1], "full physical payload parity, not only its size"
    assert wire.payload_effort(finalized[1]) == ("medium" if learned and api_surface == "responses" else "high")


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


@pytest.mark.parametrize("carrier,tier", [
    ({"source": "owner", "tier": "fast"}, "fast"),
    ({"source": "room", "tier": "economy"}, "economy"),
    ({"source": "owner", "tier": "standard"}, "standard"),
])
def test_a_tiered_works_admitted_forced_send_is_the_priced_candidate(monkeypatch, tmp_path, carrier, tier):
    """The forced final asks the tier the work's rounds and its priced copy ask (task_pacing): its identity predicate
    admits it on the first attempt, never refusing it as drift to resend it without the tier."""
    from ouroboros import loop_forced_finalization as forced

    monkeypatch.setenv("OPENROUTER_API_KEY", "unused")
    owner_ctx = SimpleNamespace(task_metadata={"processing": carrier}, model_turn_state=None)
    sent = []

    def execute(request, send, before_dispatch):
        before_dispatch(SimpleNamespace(attempt_id=f"a{len(sent) + 1}", drive_root=tmp_path))  # the predicate runs here
        sent.append(request)
        return _completion("Done so far.")

    _patch_execute_candidate(monkeypatch, llm_module, execute)
    logs = tmp_path / "logs"
    logs.mkdir()
    ctx = _ctx(drive_logs=logs, llm=LLMClient(api_key="unused"), active_model="openai/gpt-test",
               tools=SimpleNamespace(_ctx=owner_ctx), messages=[dict(message) for message in _MESSAGES])
    with usage_accounting.usage_scope(usage_accounting.UsageScope(
        drive_root=tmp_path, task_id="task1", root_task_id="task1",
    )):
        request, prepared = task_pacing.prepared_wrapup_candidate(ctx, copy.deepcopy(_MESSAGES), allow_server_web_search=False)
        assert request.processing_preference == tier
        forced._call_forced_model_once(ctx, initial_messages=prepared, admitted_request=request)
    [admitted] = sent
    assert admitted.processing_preference == tier
    assert admitted.candidate_clock_free_sha256 == request.candidate_clock_free_sha256 is not None


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


def _forced_plan(mode="max", window=128_000):
    """A minimal Main plan for the forced candidate's physical measurement."""
    import json
    from ouroboros.context_fit import ContextFitPlan, ContextFitProjection

    projections = [ContextFitProjection(mode=selected, system_content_json=json.dumps(selected + " policy"),
        estimated_tokens=10, calibrated_tokens=10, calibration_ratio=1.0) for selected in ("max", "low")]
    return ContextFitPlan(core_sha256="a" * 64, preferred_mode=mode, initial_mode=mode,
        model="openai/test-model", provider="openrouter", route_fp="route-a", status="confirmed", stale=False,
        window_tokens=window, output_reserve_tokens=65_536, user_content_json=json.dumps("go"),
        max_projection=projections[0], low_projection=projections[1])


@pytest.mark.parametrize("mode", ["nano", "max"])
def test_the_forced_lookahead_and_send_share_one_measurement_allowance_and_identity(monkeypatch, tmp_path, mode):
    """A10: the priced wrap-up copy and the admitted forced send are measured ONCE under one bound
    Main context, so a rendered Nano gets the window's reply (not the whole ceiling) on both, their
    clock-free identity still matches, and a direct or drift send measures the same bytes to the same
    allowance. Max keeps the ceiling on both (the quiet side)."""
    from dataclasses import replace

    from ouroboros import loop_forced_finalization as forced
    from ouroboros.context_fit import ContextCore

    monkeypatch.setenv("OPENROUTER_API_KEY", "unused")
    plan = replace(_forced_plan(mode, 128_000), initial_mode=mode)
    source = ('prompts/SYSTEM.md', '1' * 64, 128, '2' * 64)
    plan = replace(plan, core=ContextCore('', '', '', '', '', '', plan.user_content_json, False,
                                         base_prompt_source=source))
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
    assert admitted.physical_context.base_prompt_source == direct.physical_context.base_prompt_source == source
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


    monkeypatch.setenv("OPENROUTER_API_KEY", "unused")
    monkeypatch.setattr("ouroboros.loop._loop_tree_accounting", lambda **_k: {"settled_usd": 20.0, "accounted_usd": 20.0})
    plan = replace(_forced_plan(mode, 128_000), initial_mode=mode)
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


@pytest.mark.parametrize("pending", ["model", "locality"])
def test_a_restored_plan_for_the_pending_route_never_prices_the_acting_tail(monkeypatch, tmp_path, pending):
    """A cold tail's plan may be the next ordinary round's: another model, or this model's local server. The
    last-fit probe of the acting wrap-up never prices on that plan. Unmeasured, it reserves a rendered Nano's
    whole reply; before any service is finalized the acting route's own fit prices the stop again, on a copy,
    and the switch stays pending for the round that consumes it."""
    from dataclasses import replace

    from ouroboros import capability_evidence as ce, context, context_fit as cf, loop, loop_forced_finalization as forced
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setenv("OPENROUTER_API_KEY", "unused")
    monkeypatch.setattr("ouroboros.loop._loop_tree_accounting", lambda **_k: {"settled_usd": 20.0, "accounted_usd": 20.0})
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path / "evidence")
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    monkeypatch.setattr(context, "_context_fit_route", lambda task, **_: (
        {"model": task["model"], "provider": "local" if task["use_local_model"] else "openrouter",
         "use_local": task["use_local_model"]},
        ce.CapabilityEvidence(128_000, "confirmed", "metadata", "route-acting")))
    acting = _forced_plan().model
    restored = replace(_forced_plan("nano", 500_000), initial_mode="nano",
                       **({"model": "openai/pending-model"} if pending == "model" else {"provider": "local"}))
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    slot = registry._ctx
    slot.task_id, slot.task_metadata, slot.context_fit_plan = "task1", {}, restored
    slot.task_contract = {"budget_profile": {"cost_hard_stop_pct": 50}}  # an explicit profile's authored margin
    slot.active_model, slot.active_use_local, slot.active_context_mode = acting, False, "nano"  # it ran in Nano
    slot.active_model_override = restored.model if pending == "model" else acting
    slot.active_use_local_override = pending == "locality"
    measured, priced, finalized, sent = [], [], [], []
    measure = forced._forced_physical_context

    def physical(ctx, messages):
        plan = slot.context_fit_plan
        context = measure(ctx, messages)
        measured.append(None if context is None else (plan.model, plan.provider))
        return context

    def fits(**kwargs):
        if kwargs.get("request") is None:  # the proxy: one fits, two do not, so the exact probe runs
            return kwargs.get("reservation_count", 1) == 1
        whole = kwargs["request"].max_completion_tokens == 65_536
        priced.append(kwargs["request"].max_completion_tokens)
        return not whole or kwargs.get("reservation_count", 1) == 1  # the window's room fits twice, the ceiling once

    def execute(request, send, before_dispatch):
        sent.append(request)
        raise _Captured()  # a forced send stops at the executor, before any provider

    _patch_execute_candidate(monkeypatch, llm_module, execute)
    monkeypatch.setattr(forced, "_forced_physical_context", physical)
    monkeypatch.setattr(task_pacing, "wrapup_reservation_fits", fits)
    monkeypatch.setattr(loop, "_finalize_forced_services", lambda *_a, **_kw: finalized.append(True))
    logs = tmp_path / "logs"
    logs.mkdir()
    messages = [{"role": "system", "content": "policy " * 2_000}, {"role": "user", "content": "wrap up " * 40_000}]
    ctx = _ctx(drive_logs=logs, llm=LLMClient(api_key="unused"), active_model=acting, task_type="task",
               tools=registry, messages=[dict(message) for message in messages], tool_schemas=_TOOLS)
    ceiling = task_pacing.resolve_cost_ceiling(None, normalize_budget_profile({"cost_hard_stop_pct": 50}), root_cap_usd=50.0)
    with usage_accounting.usage_scope(usage_accounting.UsageScope(
        drive_root=tmp_path, task_id="task1", root_task_id="task1",
    )):
        assert _check_budget_limits(ctx, None, ceiling) is None, "the acting fit admits ordinary work"
    assert measured == [None, (acting, "openrouter")] and finalized == sent == []
    assert priced[0] == 65_536 and 8_192 < priced[-1] < 65_536  # the whole ceiling, then the window's room
    assert (slot.active_model_override, slot.active_use_local_override) == (
        (restored.model, False) if pending == "model" else (acting, True))


@pytest.mark.parametrize('decision', ['control', 'pricing', 'legacy', 'fallthrough', 'held_low', 'owner_low', 'smaller_resume', 'unaffordable'])
def test_pricing_view_does_not_survive_exact_budget_resume(tmp_path, monkeypatch, decision):
    """A cold pending route is priced on the acting route, parked without a send, and granted through the real queue.
    Only the budget continuation keeps the pre-price view; a legacy forced send keeps the acting fit. An already
    held Low, a newer owner Low or a smaller restored window still governs. The transport stops at the executor.
    """
    import importlib
    import json
    import time
    from ouroboros import budget_pause as bp, capability_evidence as ce, context, context_fit as cf
    from ouroboros import loop, memory_view as mv, owner_wait, usage_accounting as ua
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.loop_routes import bind_round_route
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.control_runtime import _switch_model
    from tests import _memory_view_synthetic as syn
    from tests._budget_pause_exact_helpers import _install_queue

    class Captured(BaseException):
        pass  # even a forced-call fallback cannot swallow this executor boundary

    resumed = False
    small, wide = 'openai::gpt-6-sol', 'openai::gpt-6-astra'
    profile = normalize_budget_profile({'cost_hard_stop_pct': 15})
    monkeypatch.setenv('OUROBOROS_MODEL', small)
    monkeypatch.setenv('OUROBOROS_CONTEXT_MODE', 'max')
    monkeypatch.setenv('OUROBOROS_TASK_REVIEW_MODE', 'off')
    monkeypatch.setenv('OUROBOROS_TASK_ABS_CEILING_SEC', '21600')
    monkeypatch.setenv('OPENAI_API_KEY', 'unused')
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'cyber_pro')
    monkeypatch.setattr(LLMClient, 'available_models', lambda _: [small, wide])
    monkeypatch.setattr(LLMClient, '_SUPPORTED_PARAMS_FETCHED', True, raising=False)
    monkeypatch.setattr('ouroboros.pricing._fetch_live_rows', lambda *a, **kw: {})
    monkeypatch.setattr(ce, 'canonical_evidence_root', lambda: tmp_path / 'evidence')
    monkeypatch.setattr(cf, '_route_calibration_ratio', lambda *a, **kw: 1.0)
    monetary, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, 'budget_remaining', lambda *_a, **_kw: 200.0)
    queries, prices, sends = [], [], []
    def route(task, **kwargs):
        chosen, _ = cf._context_route(task)
        window = 100_000 if chosen['model'] == small or (resumed and decision == 'smaller_resume') else 500_000
        queries.append((chosen['model'], chosen['model_role'], chosen.get('account_pin'), window))
        return chosen, ce.CapabilityEvidence(window, 'confirmed', 'metadata', chosen['model'])
    monkeypatch.setattr(context, '_context_fit_route', route)
    snapshot = syn.snapshot(room_facts=syn.room('1'), story=syn.pages(1, 300),
        conversations=[syn.conversation(i, words=10, word_chars=1000) for i in range(10)])
    core = cf.ContextCore('Core', 'BIBLE', '# Architecture\n' + 'Whole source book. ' * 2000,
        '', 'Identity', 'Facts', '"Assignment"', False, memory_view_json=mv.snapshot_json(snapshot))
    task = {'id': 'budget-view', 'type': 'task', 'chat_id': 1, '_attempt': 1, 'text': 'continue',
            'reasoning_effort': 'none', 'budget_profile': profile}
    def fit(model):
        return cf.build_context_fit_plan(None, core, {**task, 'model': model}, preferred_mode='max',
            tool_schemas=[], reasoning_effort='none', route_resolver=route)
    def registry(plan):
        tools = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
        ctx = tools._ctx
        ctx.task_id, ctx.task_attempt, ctx.task_started_at = task['id'], 1, time.time()
        ctx.current_chat_id, ctx.current_task_type = 1, 'task'
        ctx.task_contract = {'budget_profile': profile}
        ctx.context_fit_plan, ctx.active_context_mode = plan, plan.initial_mode
        ctx.owner_wait_callback = lambda *_: None
        ctx.active_model, ctx.active_use_local, ctx.active_effort = small, False, 'none'
        ctx.primary_route = {'model': small, 'use_local': False, 'role': 'main'}
        ctx.route_choice = {'kind': 'following', 'role': 'main'}
        ctx._cost_ceiling = task_pacing.resolve_cost_ceiling(200, profile, root_cap_usd=60)
        return tools
    old = fit(small)
    before = registry(old)
    _switch_model(before._ctx, model=wide, effort='high')
    write_task_result(tmp_path, task['id'], 'running', billing_group={
        'billing_group_id': task['id'], 'billing_group_limit_usd': 60.0,
        'billing_group_limit_source': 'initial_task_admission', 'billing_group_limit_revision': 'fixture'})
    usage = {'cost': 31.0 if decision != 'fallthrough' else 25.0,
             '_context_prompt_estimate': 0 if decision == 'control' else 400_000}
    if decision == 'held_low':
        before._ctx.active_context_mode = 'low'
    messages = old.messages_for(before._ctx.active_context_mode)
    messages.extend([{'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'evidence', 'type': 'function',
        'function': {'name': 'read_file', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'evidence', 'content': 'Retained tool evidence. ' * 14000}])
    wait = owner_wait.checkpoint_owner_wait(before._ctx, messages, {'tool_calls': []}, usage, 3, [], set())
    saved = json.loads(read_actor_source_bytes(tmp_path, task['id'], wait['source_ref']))
    tools = registry(fit(wide))
    ctx, trace, accumulated, seen = tools._ctx, {}, {}, set()
    owner_wait.restore_continuation_state(tools, saved, messages, trace, accumulated, seen)
    plan, mode = owner_wait.rebind_restored_route(tools, saved, messages)
    initial_system = copy.deepcopy(messages[0])
    initial = {'mode': mode, 'model': plan.model, 'steps': plan.projection(mode).memory_facts['floor']['steps']}
    if decision == 'legacy':
        ctx.owner_wait_callback = None
    # Frozen synthetic tariff; production reservation arithmetic and rail decisions remain real.
    monkeypatch.setattr('ouroboros.pricing.get_pricing', lambda **_: {
        'openai/gpt-6-sol': (100.0, 100.0, 100.0, 100.0),
        'openai/gpt-6-astra': (100.0, 100.0, 100.0, 100.0)})
    if decision == 'unaffordable':
        monkeypatch.setattr('ouroboros.pricing.get_pricing', lambda **_: {
            'openai/gpt-6-sol': (300.0, 300.0, 300.0, 300.0)})
    real_cost = ua._reservation_cost
    def cost(request):
        prices.append({'model': request.model, 'mode': ctx.active_context_mode,
                       'prompt': request.prompt_tokens_estimate, 'reply': request.max_completion_tokens,
                       'reservation': real_cost(request)})
        return real_cost(request)
    monkeypatch.setattr(ua, '_reservation_cost', cost)
    monkeypatch.setattr(loop, '_loop_tree_accounting', lambda **_: {'settled_usd': usage['cost'], 'accounted_usd': usage['cost']})
    monkeypatch.setattr('ouroboros.loop_budget._wrapup_global_remaining', lambda: 200.0)
    def execute(request, send, before_dispatch):
        sends.append({'model': request.model, 'mode': ctx.active_context_mode,
                      'plan': ctx.context_fit_plan.model, 'effort': ctx.active_effort,
                      'role': ctx.context_fit_plan.model_role,
                      'local': ctx.active_use_local, 'core': ctx.context_fit_plan.core_sha256,
                      'pending': ctx.active_model_override, 'pending_effort': ctx.active_effort_override,
                      'steps': ctx.context_fit_plan.projection(ctx.active_context_mode).memory_facts['floor']['steps'],
                      'tool_evidence_preserved': any(m.get('tool_call_id') == 'evidence' and
                          m.get('content') == 'Retained tool evidence. ' * 14000 for m in messages)})
        raise Captured()
    for name in ('ouroboros.llm_attempt', 'ouroboros.llm', 'ouroboros.llm_anthropic',
                 'ouroboros.llm_fallback', 'ouroboros.llm_gigachat', 'ouroboros.llm_local'):
        module = importlib.import_module(name)
        if hasattr(module, '_execute_candidate'):
            monkeypatch.setattr(module, '_execute_candidate', execute)
    logs = tmp_path / 'logs'
    logs.mkdir(exist_ok=True)
    limit = loop._RoundLimitContext(messages=messages, llm=LLMClient(api_key='unused'),
        active_model=small, active_use_local=False, active_effort='none', max_retries=1,
        drive_logs=logs, task_id=task['id'], round_idx=3, event_queue=None, accumulated_usage=accumulated,
        task_type='task', max_rounds=100, tools=tools, llm_trace=trace, owner_msg_seen=seen, tool_schemas=[])
    source_mode, pause_rail = None, None
    try:
        with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id=task['id'], root_task_id=task['id'],
                                         global_limit_usd=200, root_limit_usd=60)):
            try:
                assert loop._check_budget_limits(limit, 200.0, ctx._cost_ceiling) is None
            except bp.BudgetPauseRequested as stopped:
                assert sends == []
                row = stopped.pause
                source = json.loads(read_actor_source_bytes(tmp_path, task['id'], row['source_ref']))
                source_mode, pause_rail = source['route']['active_context_mode'], row['rail']
                assert source['messages'][0] == initial_system
                assert source_mode == initial['mode']
                assert source['route']['active_model_override'] == wide
                assert source['route']['active_effort_override'] == 'high'
                assert source['route']['active_effort'] == 'none'
                assert source['route']['active_use_local_override'] is False
                assert source['context_model_role'] == 'main'
                marker = bp.exact_pause_marker(row, default_root=task['id'])
                task['_budget_pause'] = marker
                workers.PENDING.append(task)
                bp.set_budget_pause(tmp_path, task['id'], {**row, 'state': bp.STATE_PAUSED})
                bp.end_dispatch_fence(task['id'])
                assert monetary.resume_budget_paused_task(task['id'])['ok'] is True
                resumed = True
                if decision == 'owner_low':
                    monkeypatch.setenv('OUROBOROS_CONTEXT_MODE', 'low')
                tools = registry(fit(wide))
                ctx = tools._ctx
                ctx.budget_pause_resume = task['_budget_pause_resume']
                restored = bp.load_budget_pause(ctx, ctx.budget_pause_resume)
                messages, trace, accumulated, seen = [], {}, {}, set()
                model, effort, local, mode, _, plan = bp.resume_paused_loop(
                    tools, restored, messages, trace, accumulated, seen, budget_remaining_usd=200)
            model, local, effort, plan, mode = bind_round_route(tools, messages,
                (small, False, 'none'), plan, mode, 'low' if decision == 'owner_low' else 'max', [], llm=limit.llm)
            from ouroboros.loop_llm_call import call_llm_with_retry
            call_llm_with_retry(limit.llm, messages, model, [], effort, 1, logs, task['id'], 4,
                queue.Queue(), accumulated, use_local=local, model_role=plan.model_role)
    except Captured:
        pass
    finally:
        bp.end_dispatch_fence(task['id'])
        print('BUDGET_VIEW_WITNESS', json.dumps({'decision': decision, 'old_mode': old.initial_mode,
            'initial': initial, 'source_mode': source_mode, 'pause_rail': pause_rail, 'prices': prices,
            'sends': sends, 'queries': queries}, sort_keys=True))
    assert len(sends) == 1
    assert sends[0]['plan'] == (small if decision == 'legacy' else wide)
    [sent] = sends
    assert sent['mode'] == ('low' if decision in {'legacy', 'held_low', 'owner_low', 'smaller_resume'} else 'max')
    assert (sent['effort'], sent['role'], sent['local'], sent['core']) == (
        'none' if decision == 'legacy' else 'high', 'main', False, old.core_sha256)
    assert sent['tool_evidence_preserved']
    assert (sent['pending'], sent['pending_effort']) == ((wide, 'high') if decision == 'legacy' else (None, None))
    assert pause_rail == (None if decision in {'legacy', 'fallthrough'} else
                          'graceful_ceiling' if decision == 'control' else 'wrapup_last_fit')
    if decision != 'control':
        assert any(row['mode'] == 'low' for row in prices), 'the acting route must actually narrow during pricing'
