"""Natural-round consumer regressions: metadata never becomes a recovery send."""
from types import SimpleNamespace

import pytest

from ouroboros import primary_route_observation as observation
from tests import test_unknown_fallback_first as fallback_fixtures
from tests.test_unknown_fallback_first import PRIMARY, _RouteLLM, _death, _run

data_root = fallback_fixtures.data_root


def test_shared_loop_observes_only_at_natural_boundaries_and_keeps_failed_refresh_historical(data_root, tmp_path, monkeypatch):
    from ouroboros import llm_probe
    from tests.test_completion_selection import finish
    from tests._usage_store_testing import ledger_rows

    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "fb/one")
    clock, probes = [0], []
    monkeypatch.setattr(observation, "time", SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 1))
    answers = iter([{"kind": "upstream_catalog", "source": "codex", "credential_profile_id": "new-account",
                     "account_fingerprint": "new-fingerprint", "observed_at": "2026-10-09T10:00:00Z"},
                    {"kind": "upstream_catalog", "source": "codex", "credential_profile_id": "new-account",
                     "account_fingerprint": "new-fingerprint", "observed_at": "2026-10-09T10:03:00Z"}, {}])
    def metadata(*args, **kwargs):
        probes.append((args, kwargs))
        return next(answers)
    monkeypatch.setattr(llm_probe, "upstream_transport_reachable", metadata)
    llm = _RouteLLM(data_root, **{PRIMARY: [_death]})
    chat, rounds = llm.chat, []
    def scripted(**kwargs):
        message, usage = chat(**kwargs)
        assert kwargs["model"] == "fb/one"
        rounds.append(kwargs)
        clock[0] = [0, 30, 150, 300, 301][len(rounds) - 1]
        if len(rounds) == 5:
            return finish("completed on fallback"), usage
        return {"role": "assistant", "content": "", "tool_calls": [{"id": f"read-{len(rounds)}", "type": "function",
                 "function": {"name": "chat_history", "arguments": "{}"}}]}, usage
    llm.chat = scripted
    text, usage, trace, registry = _run(tmp_path, llm)
    assert text == "completed on fallback"
    assert [model for model, _ in llm.sent] == [PRIMARY] + ["fb/one"] * 5
    assert len(ledger_rows(data_root)) == 6  # observations create no attempts or reservation
    assert len(probes) == 3
    assert all(args[1] == PRIMARY and kw["account_override"] == "" and kw["expected_route"] is None
               and 0 < kw["timeout"] <= 3 for args, kw in probes)
    notes = trace["primary_route_observations"]
    assert [note["facts"]["kind"] for note in notes] == ["upstream_catalog", "unconfirmed"]
    assert all(note["generation_tested"] is False for note in notes)
    assert registry._ctx._primary_route_observation["facts"] == {"kind": "unconfirmed"}
    final_prompt = str(rounds[-1]["messages"])
    assert "Earlier observations are historical" in final_prompt and 'switch_model(primary="return")' in final_prompt
    assert not usage.get("primary_return_results")


@pytest.mark.parametrize("role,pin,local,exact,deferred", [
    ("consciousness", "", False, False, None), ("main", "chosen-account", False, False, None),
    ("main", "", True, False, None), ("main", "", False, True, None), ("main", "", False, False, True),
])
def test_round_consumer_preserves_role_pin_locality_and_exact_isolation(monkeypatch, role, pin, local, exact, deferred):
    import json
    from ouroboros import llm_probe, loop, loop_model_call, local_model
    from ouroboros.model_slots import MODEL_ACCOUNTS_KEY

    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({role: pin}))
    probes = []
    monkeypatch.setattr(llm_probe, "upstream_transport_reachable", lambda *a, **k: probes.append(k) or {"kind": "upstream_http"})
    monkeypatch.setattr(local_model, "get_manager", lambda: SimpleNamespace(serving_context_evidence=lambda: {"confirmed": True, "source": "owned_server_arguments"}))
    owner = SimpleNamespace(primary_route={"model": "claudexor::codex=future", "use_local": local, "role": role},
                            exact_model_route=exact, task_metadata={}, _execution_trace={})
    monkeypatch.setattr(loop, "_dispatch_round_model", lambda *_a, **_k: ({"content": "accepted fallback"}, 0))
    monkeypatch.setattr(loop, "_measure_round_main_fit", lambda *_a, **_k: None)
    monkeypatch.setattr(loop_model_call, "_append_routing_receipts", lambda _ctx: False)
    monkeypatch.setattr(loop_model_call, "_project_wake_input", lambda _ctx: False)
    ctx = SimpleNamespace(tools=SimpleNamespace(_ctx=owner), llm=object(), context_fit_plan=None,
        messages=[{"role": "user", "content": "go"}], defer_resource_wait=deferred, attempt_cap=None,
        accumulated_usage={}, active_context_mode="max", active_model="fallback/model", active_use_local=False, tool_schemas=[])
    loop_model_call._call_round_model(ctx)
    assert bool(probes) is (not local and not exact and deferred is None)
    if probes:
        assert probes[0]["model_role"] == role and probes[0]["account_override"] == pin
    assert bool(owner._execution_trace) is (not exact and deferred is None)


def test_a_requested_return_is_not_an_accepted_response(monkeypatch):
    owner = SimpleNamespace(_execution_trace={})
    ctx = SimpleNamespace(tools=SimpleNamespace(_ctx=owner), active_model=PRIMARY, active_use_local=False, accumulated_usage={})
    observation.record_primary_return_request(owner, PRIMARY, False, "main")
    observation.record_round_route_result(ctx, "main", accepted=False)
    ctx.active_model = "fallback/model"
    observation.record_round_route_result(ctx, "fallback:0", accepted=True)
    assert owner._execution_trace["primary_return_requests"][0]["status"] == "requested"
    assert ctx.accumulated_usage["primary_return_results"][0]["status"] == "no_accepted_response"
    assert [row["to_binding"][0] for row in ctx.accumulated_usage["accepted_routes"]] == ["fallback/model"]


def test_a_return_answered_by_another_route_is_not_the_primarys_accepted_response():
    owner = SimpleNamespace(_execution_trace={})
    ctx = SimpleNamespace(tools=SimpleNamespace(_ctx=owner), active_model=PRIMARY, active_use_local=False, accumulated_usage={})
    observation.record_primary_return_request(owner, PRIMARY, False, "main")
    ctx.active_model = "fallback/model"  # the round's wait reroute answered on the fallback
    observation.record_round_route_result(ctx, "main", accepted=True)
    result = ctx.accumulated_usage["primary_return_results"][0]
    assert result["status"] == "answered_by_other_route"
    assert result["requested_binding"][0] == PRIMARY and result["actual_binding"][0] == "fallback/model"
    assert [row["to_binding"][0] for row in ctx.accumulated_usage["accepted_routes"]] == ["fallback/model"]


@pytest.mark.parametrize('accepted', [True, False])
def test_shared_loop_records_requested_return_separately_from_actual_acceptance(data_root, tmp_path, monkeypatch, accepted):
    from tests.test_completion_selection import finish
    from ouroboros import llm_probe
    monkeypatch.setenv('OUROBOROS_MODEL_FALLBACKS', 'fb/one')
    monkeypatch.setattr(llm_probe, 'upstream_transport_reachable', lambda *a, **k: {})
    scripts = [_death] + ([] if accepted else [lambda: RuntimeError('HTTP 400 bad request')])
    llm = _RouteLLM(data_root, **{PRIMARY: scripts})
    chat, requested = llm.chat, []
    def scripted(**kwargs):
        message, usage = chat(**kwargs)
        if not requested:
            requested.append(True)
            return {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'return', 'type': 'function',
                'function': {'name': 'switch_model', 'arguments': '{"primary":"return"}'}}]}, usage
        return finish('finished'), usage
    llm.chat = scripted
    _, usage, trace, _ = _run(tmp_path, llm)
    assert len(trace['primary_return_requests']) == 1
    returned = usage['primary_return_results']
    assert len(returned) == 1 and returned[0]['requested_binding'][0] == PRIMARY
    assert returned[0]['status'] == ('accepted_response' if accepted else 'no_accepted_response')
    assert [model for model, _ in llm.sent][:3] == [PRIMARY, 'fb/one', PRIMARY]
    assert llm.sent[-1][0] == (PRIMARY if accepted else 'fb/one')


@pytest.mark.parametrize('elapsed', [1, 4])
def test_owned_metadata_handshake_and_catalog_share_one_non_generating_budget(monkeypatch, elapsed):
    from ouroboros import llm_capability_policy, llm_claudexor
    calls = []
    clock = iter([10, 10 + elapsed])
    monkeypatch.setattr('time.monotonic', lambda: next(clock, 10 + elapsed))
    gateway = SimpleNamespace(operations=lambda **kw: calls.append(('operations', kw)) or [],
                              list_source_models=lambda *a, **kw: calls.append(('catalog', kw)) or {},
                              close=lambda: calls.append(('closed', {})))
    def read(**kwargs):
        calls.append(('handshake', kwargs))
        return gateway
    monkeypatch.setattr(llm_capability_policy, 'read_owned_gateway', read)
    if elapsed > 3:
        with pytest.raises(TimeoutError):
            llm_claudexor.model_catalog('codex', timeout_sec=3)
        assert [name for name, _ in calls] == ['handshake', 'closed']
    else:
        llm_claudexor.model_catalog('codex', requested_model='future', timeout_sec=3)
        assert calls[0] == ('handshake', {'timeout_sec': 3})
        assert calls[1:3] == [('operations', {'timeout_sec': 2}),
                              ('catalog', {'requested_model': 'future', 'timeout_sec': 2})]
        assert calls[-1][0] == 'closed'
