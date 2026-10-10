"""Owner-approved managed continuation after upstream recovery, with old custody."""
from datetime import datetime, timedelta, timezone
import itertools
import json
import time
from types import SimpleNamespace

import httpx
import pytest

from ouroboros import loop as loop_mod, loop_transport as transport
from ouroboros.loop import run_llm_loop
from ouroboros.tools.registry import ToolRegistry
from tests.test_loop_transport_wait import _loop_kwargs, _read_network_wait_events


def test_managed_unknown_waits_for_upstream_then_adds_new_input(tmp_path, monkeypatch):
    sends, probes, sleeps, notes = [], [], [], []
    previous = {"physical_attempt_id": "old-paid-attempt", "outcome": "unknown", "model": "test-model"}
    def send(_llm, messages, *args, **kwargs):
        usage = args[8]  # model, tools, effort, retries, logs, task, round, event, usage
        sends.append([dict(row) for row in messages])
        if len(sends) == 1:
            usage.update(_last_llm_error_kind="provider_outcome_unknown", _pending_transport_outcome=dict(previous))
            return None, 0.0
        usage.pop("_last_llm_error_kind", None)
        return {"role": "assistant", "content": "continued answer"}, 0.0
    def reachable(*args, **kwargs):
        probes.append(kwargs)
        assert len(sends) == 1
        return {"kind": "upstream_http", "status_code": 200} if len(probes) == 3 else {}
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", send)
    monkeypatch.setattr(transport, "upstream_transport_reachable", reachable)
    monkeypatch.setattr(transport, "interruptible_wait_sleep", lambda seconds, wake: sleeps.append(seconds) or False)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "1")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    result, usage, trace = run_llm_loop(**_loop_kwargs(tmp_path, registry, notes))
    assert result == "continued answer" and len(sends) == 2 and len(probes) == 3
    assert len(sleeps) == 3  # No cognition or compaction was sent between observations.
    assert sends[0] != sends[1]
    assert any("NEW physical model attempt" in str(row.get("content")) and "old-paid-attempt" in str(row.get("content")) for row in sends[1])
    assert usage["transport_recovery"]["previous_attempt"] == previous
    assert usage["transport_recovery"]["old_outcome"] == "unknown"
    assert any("another charge is possible" in text for text in notes)
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    recovered = [row for row in events if row.get("detail") == "new_attempt_after_unknown_outcome"]
    assert recovered[0]["outcome_custody"] == previous


def test_repeated_unknown_does_not_restart_backoff(tmp_path, monkeypatch):
    """Work-order §B9 "repeated unknown creates no burst": a granted continuation
    that dies unknown AGAIN returns to the SAME wait episode — the backoff keeps
    growing, the redial counter keeps counting, and exactly one [SYSTEM NOTICE]
    is appended per real continuation."""
    sends, probes, sleeps, notes = [], [], [], []
    def send(_llm, messages, *args, **kwargs):
        usage = args[8]  # model, tools, effort, retries, logs, task, round, event, usage
        sends.append([dict(row) for row in messages])
        if len(sends) <= 2:  # two unknown outcomes in a row -> two continuations
            usage.update(_last_llm_error_kind="provider_outcome_unknown",
                         _pending_transport_outcome={"physical_attempt_id": f"paid-attempt-{len(sends)}",
                                                     "outcome": "unknown"})
            return None, 0.0
        usage.pop("_last_llm_error_kind", None)
        return {"role": "assistant", "content": "continued answer"}, 0.0
    def reachable(*args, **kwargs):
        probes.append(kwargs)
        return {"kind": "upstream_http", "status_code": 200}
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", send)
    monkeypatch.setattr(transport, "upstream_transport_reachable", reachable)
    monkeypatch.setattr(transport, "interruptible_wait_sleep", lambda seconds, wake: sleeps.append(seconds) or False)
    # A wall clock that advances one second per read: the re-arm below must not depend on the
    # platform tick (windows-latest time.time() advances in ~15.6 ms steps under mocked sleeps).
    ticks = itertools.count(time.time() + 1000.0, 1.0)
    monkeypatch.setattr(transport, "time", SimpleNamespace(monotonic=time.monotonic, time=lambda: next(ticks)))
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "1")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    result, usage, _trace = run_llm_loop(**_loop_kwargs(tmp_path, registry, notes))

    assert result == "continued answer" and len(sends) == 3 and len(probes) == 2
    # The freshness bound re-arms at the latest unknown outcome: the second probe
    # may not accept evidence that only proves recovery from the first failure.
    assert probes[1]["observed_after"] > probes[0]["observed_after"]
    # The second wait starts where the first left off: no fresh 4s episode.
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0]
    notices = [row for row in sends[-1] if "NEW physical model attempt" in str(row.get("content"))]
    assert len(notices) == 2  # one per real continuation — no burst, no extra
    assert usage["transport_recovery"]["previous_attempt"]["physical_attempt_id"] == "paid-attempt-2"
    rows = _read_network_wait_events(tmp_path)
    assert [row["phase"] for row in rows].count("entered") == 1  # one episode for the whole sequence
    repeats = [row for row in rows if row.get("detail") == "continuation_outcome_unknown"]
    assert len(repeats) == 1 and repeats[0]["phase"] == "continued"
    assert repeats[0]["redials"] == 1  # the wait iteration that granted the attempt already counted its redial
    assert repeats[0]["outcome_custody"]["physical_attempt_id"] == "paid-attempt-2"
    waits = [row for row in rows if row["phase"] == "waiting"]
    assert [row["redials"] for row in waits] == [0, 1]  # one redial per wait iteration, never reset
    assert [row["next_sleep_sec"] for row in waits] == [4.0, 8.0] == sleeps


def test_alternating_unknown_and_transport_failures_keep_one_episode(tmp_path, monkeypatch):
    """A granted continuation released before dispatch ($0, ``transport_unavailable``) and a free
    redial that then crosses dispatch and dies unknown both stay in the SAME episode: one ``entered``
    row, a backoff that keeps growing across the flap, one redial per wait iteration, and custody
    that follows the latest unknown attempt."""
    sends, probes, sleeps, notes = [], [], [], []
    kinds = ["provider_outcome_unknown", "transport_unavailable", "provider_outcome_unknown"]
    def send(_llm, messages, *args, **kwargs):
        usage = args[8]  # model, tools, effort, retries, logs, task, round, event, usage
        sends.append([dict(row) for row in messages])
        if len(sends) <= len(kinds):
            kind = usage["_last_llm_error_kind"] = kinds[len(sends) - 1]
            if kind == "provider_outcome_unknown":
                usage["_pending_transport_outcome"] = {"physical_attempt_id": f"paid-attempt-{len(sends)}", "outcome": "unknown"}
            return None, 0.0
        usage.pop("_last_llm_error_kind", None)
        return {"role": "assistant", "content": "continued answer"}, 0.0
    def reachable(*args, **kwargs):
        probes.append(kwargs)
        return {"kind": "upstream_http", "status_code": 200}
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", send)
    monkeypatch.setattr(transport, "upstream_transport_reachable", reachable)
    monkeypatch.setattr(transport, "interruptible_wait_sleep", lambda seconds, wake: sleeps.append(seconds) or False)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "1")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    result, usage, _trace = run_llm_loop(**_loop_kwargs(tmp_path, registry, notes))

    assert result == "continued answer" and len(sends) == 4 and len(probes) == 2  # no probe for the free redial
    assert sleeps == [4.0, 8.0, 16.0]  # one episode: the backoff never restarts across the flap
    rows = _read_network_wait_events(tmp_path)
    assert [row["phase"] for row in rows].count("entered") == 1 and not [row for row in rows if row["phase"] == "ended"]
    assert [row["detail"] for row in rows if row["phase"] == "continued"] == [
        "continuation_transport_unavailable", "redial_outcome_unknown"]
    assert [row["redials"] for row in rows if row["phase"] == "waiting"] == [0, 1, 2]
    assert rows[-1]["detail"] == "new_attempt_after_unknown_outcome"
    assert rows[-1]["outcome_custody"]["physical_attempt_id"] == "paid-attempt-3"  # custody follows the latest unknown
    assert usage["transport_recovery"]["previous_attempt"]["physical_attempt_id"] == "paid-attempt-3"
    notices = [row for row in sends[-1] if "NEW physical model attempt" in str(row.get("content"))]
    assert len(notices) == 2 and "paid-attempt-3" in str(notices[-1]["content"])
    # The owner is told that money became unknown when the free redial crossed dispatch (the
    # entry note said "$0"), not only at the next grant; and that the granted attempt never
    # reached the provider (the grant note said "continuing").
    assert sum("another charge is possible" in text for text in notes) >= 2
    assert any("could not reach the provider" in text for text in notes)


def test_grant_without_a_new_attempt_is_not_a_phantom_repeat(tmp_path):
    """After a grant, a round that made NO new physical attempt (a failed probe, a call that yielded
    to control) still carries the sticky unknown kind: the episode keeps its grant and its custody
    untouched and writes no row. Only a fresh unknown attempt — new pending custody — repeats."""
    ctx = SimpleNamespace(task_id="t", _accumulated_usage={})
    episode = transport.TransportWaitEpisode(
        wait_cause="provider_outcome_unknown", started_monotonic=0.0, continuation_granted=True,
        outcome_custody={"physical_attempt_id": "paid-attempt-1"}, wait_iterations=1, redials=1)
    kwargs = dict(msg_present=False, error_kind="provider_outcome_unknown", drive_logs=tmp_path, task_id="t",
                  model="m", emit_progress=lambda *a, **kw: None)
    assert transport.reconcile_transport_wait(episode, ctx, **kwargs) is episode
    assert episode.continuation_granted and episode.outcome_custody == {"physical_attempt_id": "paid-attempt-1"}
    assert _read_network_wait_events(tmp_path) == []
    ctx._accumulated_usage["_pending_transport_outcome"] = {"physical_attempt_id": "paid-attempt-2", "outcome": "unknown"}
    assert transport.reconcile_transport_wait(episode, ctx, **kwargs) is episode
    assert not episode.continuation_granted and episode.redials == 1
    assert episode.outcome_custody["physical_attempt_id"] == "paid-attempt-2"
    assert [(row["phase"], row["detail"], row["redials"]) for row in _read_network_wait_events(tmp_path)] == [
        ("continued", "continuation_outcome_unknown", 1)]


def test_unknown_policy_admits_direct_turns_but_not_presence_or_a_readable_operation(tmp_path):
    """Owner decision 1A: a direct turn's eligible unknown outcome enters the same wait and
    continuation as a queued one, inside its interactive bound. Inline Presence keeps its own
    no-resend terminal, and an accepted operation whose read failed is only rejoined."""
    notes = []
    direct = SimpleNamespace(task_id="t", is_direct_chat=True, _accumulated_usage={})
    episode = transport.reconcile_transport_wait(None, direct, msg_present=False,
        error_kind="provider_outcome_unknown", drive_logs=tmp_path, task_id="t", model="m",
        emit_progress=lambda text, **_kw: notes.append(text))
    assert episode is not None and episode.interactive and episode.wait_cause == "provider_outcome_unknown"
    assert episode.wait_bound_sec is not None and notes and "Stop cancels" not in notes[0]
    readable = {"_pending_transport_outcome": {"physical_attempt_id": "a", "same_operation_recoverable": True}}
    for ctx in (SimpleNamespace(task_id="t", is_direct_chat=True, current_task_type="presence", _accumulated_usage={}),
                SimpleNamespace(task_id="t", _accumulated_usage=readable),
                SimpleNamespace(task_id="t", task_metadata={"presence": {"binding": "b"}}, _accumulated_usage={})):
        assert transport.reconcile_transport_wait(None, ctx, msg_present=False,
            error_kind="provider_outcome_unknown", drive_logs=tmp_path, task_id="t", model="m",
            emit_progress=lambda *a, **kw: pytest.fail("unexpected automatic continuation")) is None


def test_deadline_closes_unknown_wait_without_a_probe_or_send(tmp_path, monkeypatch):
    ctx = SimpleNamespace(task_id="t", task_metadata={"deadline_at": (datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()})
    episode = transport.TransportWaitEpisode(wait_cause="provider_outcome_unknown")
    monkeypatch.setattr(transport, "upstream_transport_reachable", lambda *a, **k: pytest.fail("probe after deadline"))
    assert not transport.continue_unknown_transport(episode, llm=None, tools=SimpleNamespace(_ctx=ctx),
        messages=[], accumulated_usage={}, drive_logs=tmp_path, task_id="t", model="m", emit_progress=lambda *a, **k: None)


@pytest.mark.parametrize("status,ready", [(200, True), (401, True), (405, True), (503, False)])
def test_upstream_probe_is_non_generating_and_uses_resolved_route(monkeypatch, status, ready):
    from ouroboros.llm import LLMClient

    seen = []
    original = httpx.Client
    def client(**kwargs):
        assert kwargs["trust_env"] is False
        def respond(request):
            seen.append(request)
            return httpx.Response(status)
        return original(**kwargs, transport=httpx.MockTransport(respond))
    monkeypatch.setattr(httpx, "Client", client)
    llm = LLMClient()
    monkeypatch.setattr(llm, "_resolve_remote_target",
                        lambda model: {"base_url": "https://exact-provider.invalid/api"})
    observed = transport.upstream_transport_reachable(llm, "vendor/model", timeout=3)
    assert bool(observed) is ready
    assert len(seen) == 1 and seen[0].method == "HEAD"
    assert str(seen[0].url) == "https://exact-provider.invalid/api"
    assert not seen[0].content and "authorization" not in seen[0].headers


def test_loopback_response_cannot_prove_upstream_recovery(monkeypatch):
    monkeypatch.setattr(httpx, "Client", lambda **kw: pytest.fail("loopback is not upstream"))
    llm = SimpleNamespace(_resolve_remote_target=lambda model: {"base_url": "http://127.0.0.1:1234/v1"})
    assert not transport.upstream_transport_reachable(llm, "vendor/model", timeout=3)


@pytest.mark.parametrize("fact", ["absent", "stale", "local", "upstream"])
def test_subscription_requires_fresh_typed_upstream_observation(monkeypatch, fact):
    from ouroboros import llm_claudexor
    now = datetime.now(timezone.utc)
    observed = (now + timedelta(seconds=1)).isoformat() if fact != "stale" else "2020-01-01T00:00:00Z"
    catalog = {"source": "codex", "observedAt": observed, "credentialProfileId": "profile-a",
               "models": [{"id": "test"}], "provenance": "fixture"}
    if fact != "absent":
        catalog["provenance"] = "local_cache" if fact == "local" else "provider_http"
    monkeypatch.setattr(llm_claudexor, "model_catalog", lambda *a, **kw: catalog)
    monkeypatch.setattr("ouroboros.model_slots.model_role_option", lambda *a: "profile-a")
    assert bool(transport.upstream_transport_reachable(None, "claudexor::codex=test", timeout=3)) is (fact == "upstream")


@pytest.mark.parametrize("route_kind", ["", "agent_session"])
def test_claudexor_control_loss_keeps_same_operation_past_read_window(tmp_path, monkeypatch, route_kind):
    from ouroboros import llm_claudexor
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    inv = llm_claudexor._ModelInvocation({"usage_model": "claudexor::codex=test"}, {}, {"timeout": 1})
    inv.operation_id, inv.invocation_id, inv.task_id, inv.root = "same-op", "same-attempt", "t", tmp_path
    inv.create_attempted = True
    now, reads = [0.0], []
    ctx = SimpleNamespace(task_id="t", _configured_subagent_route_kind=route_kind)
    waiter = SimpleNamespace(tool_context=ctx, control_reason=lambda: None)
    monkeypatch.setattr(llm_claudexor, "current_model_wait", lambda: waiter)
    monkeypatch.setattr(llm_claudexor, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=lambda t: now.__setitem__(0, now[0]+t)))
    class Gateway:
        def get_model_operation(self, operation, **kwargs):
            reads.append(operation)
            if len(reads) <= 3: raise ClaudexorUnavailable("daemon_unreachable", "offline")
            return {"id": operation, "state": "succeeded", "dispatch": {"state": "response_received"},
                    "response": {"state": "ready", "ref": {"sha256": "exact"}}}
        def get_model_result(self, operation, **kwargs):
            assert operation == "same-op"
            return b'{"outcome":"completed","message":{"content":"same paid answer"}}'
        def create_model_operation(self, *a, **kw): pytest.fail("second model operation")
        def close(self): pass
    gateway = inv.gateway = Gateway()
    monkeypatch.setattr(llm_claudexor, "read_owned_gateway", lambda: gateway)
    monkeypatch.setattr(inv, "retain", lambda raw: None)
    answer = inv.receive()
    assert answer["message"]["content"] == "same paid answer"
    assert reads == ["same-op"] * 4 and now[0] > inv.timeout
    assert inv.operation_id == "same-op" and inv.invocation_id == "same-attempt"
    events = [json.loads(line) for line in (tmp_path / "logs/events.jsonl").read_text().splitlines()]
    assert events[-1]["detail"] == "same_model_operation_rejoined"


def test_managed_continuation_keeps_old_money_and_mints_one_new_attempt(tmp_path, monkeypatch):
    from ouroboros import usage_accounting as ua
    from tests.test_transport_death_retry import _LedgerLLM, _ledger, _loop_kwargs as ledger_kwargs
    llm = _LedgerLLM(tmp_path, lambda: httpx.ReadError("lost after dispatch"))
    observations = []
    def reachable(*args, **kwargs):
        observations.append(llm.calls)
        return {"kind": "upstream_http", "status_code": 200}
    monkeypatch.setattr(transport, "upstream_transport_reachable", reachable)
    monkeypatch.setattr(transport, "interruptible_wait_sleep", lambda *args: False)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="t-death", root_task_id="t-death", global_limit_usd=100)):
        text, usage, _ = run_llm_loop(**ledger_kwargs(tmp_path, llm, []))
    assert text == "done" and llm.calls == 2 and observations == [1]
    rows = _ledger(tmp_path)
    assert [(row["state"], row["revision"]) for row in rows] == [("unresolved", 4), ("settled", 3)]
    assert rows[0]["physical_failure"]
    old, new = rows[0]["attempt_id"], rows[1]["attempt_id"]
    assert old != new
    assert usage["transport_recovery"]["previous_attempt"]["physical_attempt_id"] == old
    assert ua.usage_projection(tmp_path)["unresolved_upper_bound_usd"] == 1.0


@pytest.mark.parametrize("reported_model", ["absent", None, "test"])
@pytest.mark.parametrize("axis", ["matching", "empty", "before_wait", "source", "profile", "fingerprint", "model", "local"])
@pytest.mark.parametrize("advisory", [False, True])
def test_catalog_reachability_binds_effective_account_and_wait_start(monkeypatch, axis, reported_model, advisory):
    import time
    from ouroboros import llm_claudexor
    started = time.time() - 10
    catalog = dict(source="codex", credentialProfileId="effective-profile", accountFingerprint="account-a",
                   observedAt=datetime.fromtimestamp(started + 1, timezone.utc).isoformat(),
                   provenance="provider_http", models=[{"id": "test"}])
    if advisory:
        catalog.update(models=[], admission={"requestedModel": "test", "inventoryAbsence": "advisory"})
    if axis == "before_wait": catalog["observedAt"] = datetime.fromtimestamp(started - 1, timezone.utc).isoformat()
    if axis == "source": catalog["source"] = "foreign"
    if axis == "profile": catalog["credentialProfileId"] = "foreign"
    if axis == "fingerprint": catalog["accountFingerprint"] = "foreign"
    if axis == "model":
        catalog["models"] = [{"id": "foreign"}]
        if advisory: catalog["admission"]["requestedModel"] = "foreign"
    if axis == "local": catalog["provenance"] = "local_cache"
    if axis == "empty": catalog = {}
    route = {"source": "codex", "credentialProfileId": "effective-profile", "accountFingerprint": "account-a"}
    if reported_model != "absent": route["model"] = reported_model
    inv = llm_claudexor._ModelInvocation({}, {}, {})
    inv.operation_id = "old-unknown-operation"
    error = inv.error({"code": "transport_unknown", "retryable": False},
                      {"dispatch": {"state": "unknown", "route": route}}, unknown=True)
    reads = []
    def read(source, account, **kwargs):
        assert (source, account, kwargs["requested_model"]) == ("codex", "effective-profile", "test")
        reads.append(kwargs)
        return catalog
    monkeypatch.setattr(llm_claudexor, "model_catalog", read)
    result = transport.upstream_transport_reachable(None, "claudexor::codex=test", timeout=3,
        account_override="", observed_after=started,
        expected_route=error.route)
    assert bool(result) is (axis == "matching")
    assert len(reads) == 1
    assert error.route == route and error.code == "model_outcome_unknown" and not error.retryable
    assert error.operation_id == "old-unknown-operation"


@pytest.mark.parametrize("axis", ["source", "model", "profile"])
def test_explicit_unknown_operation_route_mismatch_refuses_before_catalog(monkeypatch, axis):
    from ouroboros import llm_claudexor
    route = {"source": "codex", "model": None, "credentialProfileId": "effective-profile"}
    route[{"source": "source", "model": "model", "profile": "credentialProfileId"}[axis]] = "foreign"
    monkeypatch.setattr(llm_claudexor, "model_catalog", lambda *a, **kw: pytest.fail("mismatched route was probed"))
    assert not transport.upstream_transport_reachable(None, "claudexor::codex=test", timeout=3,
        account_override="effective-profile", expected_route=route)


def test_null_reported_model_recovers_without_rewriting_unknown_custody(tmp_path, monkeypatch):
    from ouroboros import llm_claudexor
    previous = {"physical_attempt_id": "old-paid-attempt", "operation_id": "old-operation", "outcome": "unknown",
                "route": {"source": "codex", "model": None, "credentialProfileId": "profile-a",
                          "accountFingerprint": "account-a"}}
    episode = transport.TransportWaitEpisode(wait_cause="provider_outcome_unknown", outcome_custody=previous)
    def catalog(source, profile, **kwargs):
        assert (source, profile, kwargs["requested_model"]) == ("codex", "profile-a", "test")
        return {"source": source, "credentialProfileId": profile, "accountFingerprint": "account-a",
                "provenance": "provider_http",
                "observedAt": datetime.fromtimestamp(episode.started_at + 1, timezone.utc).isoformat(),
                "models": [{"id": "test"}]}
    monkeypatch.setattr(llm_claudexor, "model_catalog", catalog)
    monkeypatch.setattr("ouroboros.model_slots.task_model_binding", lambda *a, **kw: ("main", "profile-a"))
    monkeypatch.setattr(llm_claudexor, "chat_claudexor", lambda *a, **kw: pytest.fail("metadata must not generate"))
    ctx = SimpleNamespace(task_id="t", task_metadata={})
    usage, messages = {}, []
    assert transport.continue_unknown_transport(episode, llm=None, tools=SimpleNamespace(_ctx=ctx),
        messages=messages, accumulated_usage=usage, drive_logs=tmp_path, task_id="t",
        model="claudexor::codex=test", emit_progress=lambda *a, **kw: None)
    assert messages[0]["role"] == "user" and "NEW physical model attempt" in messages[0]["content"]
    assert usage["transport_recovery"]["previous_attempt"] == previous
    assert usage["transport_recovery"]["old_outcome"] == "unknown"
    assert previous["route"]["model"] is None and episode.outcome_custody == previous


def test_stop_before_unknown_probe_preserves_no_send_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(transport, "transport_repeat_stop_requested", lambda *_: True)
    monkeypatch.setattr(transport, "upstream_transport_reachable", lambda *a, **kw: pytest.fail("probe after Stop"))
    assert not transport.continue_unknown_transport(transport.TransportWaitEpisode(), llm=None,
        tools=SimpleNamespace(_ctx=SimpleNamespace(task_id="t")), messages=[], accumulated_usage={},
        drive_logs=tmp_path, task_id="t", model="m", emit_progress=lambda *a, **kw: None)


@pytest.mark.parametrize("remaining", [2700.0, 3.0])
def test_upstream_head_uses_connection_window_in_every_socket_phase(monkeypatch, remaining):
    """A metadata HEAD cannot hold an unleased task for the cognition read window."""
    from ouroboros.llm import LLMClient

    requests = []
    original = httpx.Client

    def client(**kwargs):
        def respond(request):
            requests.append(request)
            return httpx.Response(200)
        return original(**kwargs, transport=httpx.MockTransport(respond))

    monkeypatch.setattr(httpx, "Client", client)
    llm = LLMClient()
    monkeypatch.setattr(llm, "_resolve_remote_target",
                        lambda model: {"base_url": "https://exact-provider.invalid/api"})
    observed = transport.upstream_transport_reachable(llm, "vendor/model", timeout=remaining)
    assert observed["kind"] == "upstream_http" and len(requests) == 1
    assert requests[0].method == "HEAD" and not requests[0].content
    # Compare the actual HTTPX request with the established ordinary connection
    # allowance, retaining a shorter owner remainder on every socket phase.
    bound = min(remaining, llm._no_proxy_timeout(remaining).connect)
    assert requests[0].extensions["timeout"] == dict(connect=bound, read=bound, write=bound, pool=bound)


def test_unknown_policy_admits_configured_session_model_to_managed_continuation(tmp_path):
    ctx = SimpleNamespace(task_id="t", exact_model_route=True, _configured_subagent_route_kind="agent_session")
    episode = transport.reconcile_transport_wait(None, ctx, msg_present=False,
        error_kind="provider_outcome_unknown", drive_logs=tmp_path, task_id="t", model="m",
        emit_progress=lambda *a, **kw: None)
    assert episode is not None and episode.wait_cause == "provider_outcome_unknown"
    assert not episode.interactive
