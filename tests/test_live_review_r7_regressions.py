"""Regressions from review round R7 (astra, 22.10): synthetic transports and a disposable filesystem only."""
import asyncio
import errno
import json
import ssl
from types import SimpleNamespace

import pytest

from tests.test_live_conversation import host_world as host_world
from tests.test_live_conversation_config import config_api as config_api, request
from tests.test_live_conversation_driver import Host, started, until


@pytest.mark.parametrize("fail_timing", [False, True])
def test_timing_write_failure_cannot_skip_response_finalization(host_world, monkeypatch, fail_timing):
    from ouroboros import live_conversation, usage_accounting as ua
    from ouroboros.contracts.live_conversation import LiveOptions

    host = host_world.host
    original = host.observe
    def observe(direction, raw):
        if fail_timing and json.loads(raw).get("type") == "call.timing":
            raise OSError(errno.ENOSPC, "synthetic full wire-log volume")
        return original(direction, raw)

    async def consume(_session):
        host.response_started("unfinished")

    async def scenario():
        await host.session(consume, LiveOptions())
        assert ua.read_usage_records(host.ctx.drive_root)[0]["state"] == "dispatched"
        with monkeypatch.context() as fault:
            fault.setattr(host, "observe", observe)
            try:
                await host.finish()
            except OSError as exc:
                assert fail_timing and exc.errno == errno.ENOSPC
        [row] = ua.read_usage_records(host.ctx.drive_root)
        # A logging error may surface, but lifetime/accounting cleanup is still mandatory.
        assert row["state"] == "unresolved", (row["state"], sorted(host.responses))
        assert row["reason"] == "call_ended_without_receipt"
        assert not host.responses
        assert host not in live_conversation._active_hosts
    asyncio.run(scenario())


@pytest.mark.parametrize("file_state", ["readable", "malformed", "permission"])
def test_owner_can_disable_when_old_settings_cannot_be_read(config_api, tmp_path, monkeypatch, file_state):
    from skills.live_conversation import plugin

    api, _, _, _ = config_api
    path = tmp_path / "feature/conversation.json"
    fields = {"provider": "fixture-engine:voice", "model": "fixture-model"}
    assert request(api, {**fields, "enabled": True})[0] == 200
    ended = []
    monkeypatch.setattr(plugin, "end_feature_calls", ended.append)
    if file_state == "malformed":
        path.write_text("{incomplete", encoding="utf-8")
    elif file_state == "permission":
        path.chmod(0o200)  # Parent remains writable: atomic replacement is permitted.
        with pytest.raises(PermissionError):
            path.read_text(encoding="utf-8")
    try:
        status, result = request(api, {**fields, "enabled": False})
    finally:
        path.chmod(0o600)
    assert status == 200 and result["enabled"] is False
    assert ended == ["live_conversation:call"]


@pytest.mark.parametrize("rotation", ["closed", "go_away"])
def test_answers_between_rotations_keep_a_healthy_call_alive(rotation):
    from skills.live_conversation.driver import UNPROVEN_RESUMES

    async def scenario():
        host = Host()
        task, session = await started(host)
        try:
            for n in range(UNPROVEN_RESUMES + 3):
                session.emit("ready", input_rate=24000, output_rate=24000)
                session.emit("resumable", handle=f"handle-{n}")
                session.emit("response_started", response_id=f"answer-{n}")
                session.emit("response_done", response_id=f"answer-{n}", usage={}, raw_usage={},
                             status="completed", continues=False)
                await until(lambda: len(host.usage) == 2 * (n + 1))
                if rotation == "closed":
                    session.emit("closed", resumable=True, reason="transport_lost")
                else:
                    session.emit("renew", resume=True, within_s=2, reason="go_away")
                await until(lambda: len(host.sessions) == n + 2 or task.done())
                assert not task.done()
                session = host.sessions[-1]
                assert host.options[-1].resume_handle == f"handle-{n}"
        finally:
            host.hub.ended_by_owner = True
            await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


@pytest.mark.parametrize("shape", ["legacy", "response"])
@pytest.mark.parametrize("status,code", [(401, "authentication"), (403, "authentication"),
    (429, "rate_limit"), (500, "engine_unavailable"), (503, "engine_unavailable"), (599, "engine_unavailable")])
def test_actual_websocket_http_exceptions_are_typed_without_echoing_headers(shape, status, code):
    from websockets.datastructures import Headers
    from websockets.exceptions import InvalidStatus, InvalidStatusCode
    from websockets.http11 import Response
    from ouroboros.contracts.realtime import RealtimeSetupError
    from tests.test_realtime_session_handshake import _session

    headers = Headers({"Authorization": "synthetic-secret-canary"})
    error = (InvalidStatusCode(status, headers) if shape == "legacy"
             else InvalidStatus(Response(status, "synthetic-secret-canary", headers)))
    with pytest.raises(RealtimeSetupError) as caught:
        asyncio.run(_session(error).connect_wire())
    assert caught.value.code == code
    assert "synthetic-secret-canary" not in str(caught.value)


@pytest.mark.parametrize("kind", ["os", "timeout", "tls", "close", "credential", "cancelled", "http400"])
def test_non_http_failures_and_unmapped_status_preserve_identity(kind):
    from websockets.exceptions import ConnectionClosedError, InvalidStatusCode
    from websockets.frames import Close
    from websockets.datastructures import Headers
    from ouroboros.contracts.realtime import RealtimeSetupError
    from tests.test_realtime_session_handshake import _session

    errors = {"os": OSError(503, "not an HTTP status"), "timeout": TimeoutError("timed out"),
        "tls": ssl.SSLError("TLS handshake failed"), "close": ConnectionClosedError(Close(1008, "policy"), None),
        "credential": RealtimeSetupError("credential", "missing synthetic key"),
        "cancelled": asyncio.CancelledError(), "http400": InvalidStatusCode(400, Headers())}
    error = errors[kind]
    async def scenario():
        # asyncio.run may re-raise cancellation with a different object. Check the port itself.
        with pytest.raises(type(error)) as caught:
            await _session(error).connect_wire()
        assert caught.value is error
    asyncio.run(scenario())


def test_timing_is_plain_json_and_first_response_is_stable(host_world):
    from ouroboros.contracts.live_conversation import LiveOptions
    host = host_world.host
    async def consume(_session):
        host.response_started("first")
        first = host.timing["first_response_at"]
        host.response_started("first")
        assert host.timing["first_response_at"] == first
    async def scenario():
        await host.session(consume, LiveOptions())
        await host.finish()
    asyncio.run(scenario())
    trace = json.loads(json.dumps(host.trace, allow_nan=False))
    rows = [json.loads(row) for row in host.wire_path.read_text(encoding="utf-8").splitlines()]
    timings = [json.loads(row["raw"]) for row in rows if row["direction"] == "host"
               and json.loads(row["raw"]).get("type") == "call.timing"]
    assert timings == [{"type": "call.timing", **trace["timing"]}]
    # R7-4: admission is stamped on the turn (before queueing and preparation); the host start is its own fact.
    assert set(trace["timing"]) == {"admitted_at", "host_started_at", "context_ready_at", "session_open_at",
                                    "first_response_at", "connections"}
    assert trace["timing"]["admitted_at"] <= trace["timing"]["host_started_at"]


def test_real_overview_producer_stays_whole_in_each_ordinary_mode(tmp_path, monkeypatch):
    from ouroboros.context_sections import build_knowledge_sections
    from ouroboros import context_fit as fit
    from tests.test_light_overview_reserved import _mind_writes_the_overview
    from tests.test_section_floor import RESERVE, _core, _plan, _text, _host_facts

    _mind_writes_the_overview(tmp_path)
    env = SimpleNamespace(drive_path=lambda path: tmp_path / path)
    overview = next(s for s in build_knowledge_sections(env) if getattr(s, "section_id", "") == "shared_understanding")
    assert overview.operative_view is None
    monkeypatch.setattr(fit, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    monkeypatch.setattr(fit, "main_output_reserve_tokens", lambda **_kw: RESERVE)
    for mode in ("max", "low", "nano"):
        plan = _plan(_core(tmp_path, head=[overview]), 2_000_000, mode).reproject_for_route(
            window_tokens=RESERVE + 1, known_window=True, ratio=1, output_reserve=RESERVE,
            tool_schemas=[], start_mode=mode)
        assert overview.full_text in _text(plan, mode)
        assert "shared_understanding" not in {r["section_id"] for r in _host_facts(plan, mode)}


def test_openai_measured_window_reaches_the_ordinary_presend_check(tmp_path):
    from dataclasses import replace
    from ouroboros.capability_evidence import record_route_facts
    from ouroboros.llm_attempt import main_context_fit_refusal
    from ouroboros.usage_accounting import AttemptRequest, PhysicalAttemptContext
    from skills.openai_realtime.plugin import CATALOG, ROUTE

    model = CATALOG.model("")
    evidence = record_route_facts(tmp_path, provider=ROUTE.accounting_provider,
        model=ROUTE.model_prefix + model.id, base_url=ROUTE.endpoint, facts=model.facts)
    physical = PhysicalAttemptContext("owner_nano", "nano", "cold_estimate", evidence.route_fp, "round",
        None, evidence.window_tokens, False, False, 1.0)
    request = AttemptRequest(ROUTE.model_prefix + model.id, ROUTE.accounting_provider,
        max_completion_tokens=32000, prompt_tokens_bounded_estimate=120000,
        physical_context=physical, working_reserve_tokens=model.facts.working_reserve_tokens)
    assert main_context_fit_refusal(request) is None
    assert main_context_fit_refusal(replace(request,
        prompt_tokens_bounded_estimate=evidence.window_tokens - model.facts.working_reserve_tokens + 1)) is not None
