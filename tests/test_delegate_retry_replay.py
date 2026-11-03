"""#1614: a lost start is recovered by its ticket, without repeating the prompt.

``delegate_start(retry_of=<pending_invocation_id>)`` replays the retained canonical
request under its original key. An omitted or empty prompt reuses that request; other
text is an exact assertion on both direct and configured calls. The
single validator keeps its order, the recorded fingerprints survive the replay, the
present target/snapshot checks still run, and a request argument the replay cannot
honor is refused before anything is sent. Every engine call goes to a stub transport.
"""

from __future__ import annotations

import hashlib
import json

import httpx
import pytest

from ouroboros import delegate_custody as custody
from ouroboros import subagents
from ouroboros.config import CLAUDEXOR_MIN_VERSION, CLAUDEXOR_PROTOCOL_MAJOR
from ouroboros.gateways import claudexor as cx
from tests._delegated_transport_shared import (  # noqa: F401  (autouse fixture applies on import)
    _delegating_ctx,
    _owned_gateway_uses_each_test_transport,
    _select_transport_access,
    _transport_snapshot,
)
from tests.test_builtin_refusal_results import _delegate_registry

PROMPT = "Summarize the repository layout."
ADVICE = "Also list the test directories."


class _Engine:
    """The stub engine: records every start POST and loses the first ``lose`` of them."""

    def __init__(self, tmp_path, monkeypatch, *, lose=1):
        self.keys, self.bodies, self.lose = [], [], lose
        real_gateway = cx.ClaudexorGateway

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/v2/handshake":
                return httpx.Response(200, json={"protocolMajor": CLAUDEXOR_PROTOCOL_MAJOR, "compatible": True,
                                                 "engine": {"version": CLAUDEXOR_MIN_VERSION}})
            if path == "/v2/agent-capabilities":
                return httpx.Response(200, json={"harnesses": [{"id": "some-route", "enabled": True,
                                                                "status": "ok", "accessProfilesSupported": ["readonly"]}]})
            if path == "/v2/quota":
                return httpx.Response(200, json={"snapshots": []})
            if path == "/v2/projects":
                return httpx.Response(200, json={"projects": [{"id": "prj-existing", "root": str(tmp_path)}]})
            assert path == "/v2/runs", path
            self.keys.append(request.headers.get("Idempotency-Key"))
            self.bodies.append(json.loads(request.read()))
            if len(self.keys) <= self.lose:
                raise httpx.ConnectError("the reply was lost after the POST")
            return httpx.Response(200, json={"runId": f"run-{len(self.keys)}"})

        def connect(*_a, **_k):
            gateway = real_gateway(cx.DaemonEndpoint("127.0.0.1", 1, "secret-token"))
            gateway._client = httpx.Client(base_url="http://127.0.0.1:1", transport=httpx.MockTransport(handler),
                                           headers=dict(gateway._client.headers))
            return gateway

        _select_transport_access(monkeypatch, "readonly")
        monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
        monkeypatch.setattr(cx, "ClaudexorGateway", connect)


def _lost_start(tmp_path, monkeypatch, *, configured=False):
    """A real readonly start whose reply is lost: (registry, engine, pending token)."""
    from ouroboros.tools import delegate

    registry, engine = _delegate_registry(tmp_path, monkeypatch), _Engine(tmp_path, monkeypatch)
    lost = json.loads(delegate._delegate_start(registry._ctx, PROMPT, max_seconds=120).text)
    assert lost["reason"] == "daemon_unreachable" and lost["pending_invocation_id"] == engine.keys[0], lost
    if configured:
        registry._ctx._configured_actor_bootstrap = {
            "selected_subagent_id": "transport-fixture",
            "snapshot": _transport_snapshot(subagents.get_subagent_harness(), "readonly")}
    return registry, engine, lost["pending_invocation_id"]


def _call(registry, **args):
    return json.loads(registry.execute_result("delegate_start", args).text)


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("prompt", [None, "", PROMPT])
def test_a_public_retry_replays_the_recorded_request_under_its_key(tmp_path, monkeypatch, configured, prompt):
    registry, engine, token = _lost_start(tmp_path, monkeypatch, configured=configured)
    payload = _call(registry, retry_of=token, **({} if prompt is None else {"prompt": prompt}))
    assert payload["status"] == "started" and payload["idempotent_recovery"] is True, payload
    assert payload["invocation_id"] == token and payload["max_seconds"] == 120
    assert engine.keys == [token, token] and engine.bodies[1] == engine.bodies[0]
    assert custody.pending_invocations(tmp_path) == []
    assert "coordination_text" not in payload


def test_a_configured_retry_refuses_changed_text(tmp_path, monkeypatch):
    registry, engine, token = _lost_start(tmp_path, monkeypatch, configured=True)
    payload = _call(registry, retry_of=token, prompt=ADVICE)
    assert payload["reason"] == "retry_prompt_mismatch", payload
    assert engine.keys == [token]
    assert ADVICE not in json.dumps(engine.bodies)


@pytest.mark.parametrize("prompt", [" ", PROMPT + " ", "Describe something else."])
def test_direct_retry_text_is_an_exact_assertion(tmp_path, monkeypatch, prompt):
    registry, engine, token = _lost_start(tmp_path, monkeypatch)
    payload = _call(registry, retry_of=token, prompt=prompt)
    assert payload["reason"] == "retry_prompt_mismatch" and "Omit prompt" in payload["detail"], payload
    assert engine.keys == [token]
    assert [row["invocation_id"] for row in custody.pending_invocations(tmp_path)] == [token]


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("extra,reason", [
    ({"max_seconds": 300}, "retry_selector_conflict"),
    ({"effort": "high"}, "retry_selector_conflict"),
    ({"directory_strategy": "direct"}, "retry_selector_conflict"),
    ({"scope_paths": ["."]}, "retry_selector_conflict"),
    ({"continue_from": "run-elsewhere"}, "continuation_selector_conflict"),
])
def test_a_retry_refuses_a_request_argument_it_cannot_replay(tmp_path, monkeypatch, configured, extra, reason):
    registry, engine, token = _lost_start(tmp_path, monkeypatch, configured=configured)
    payload = _call(registry, retry_of=token, **extra)
    assert payload["status"] == "refused" and payload["reason"] == reason, payload
    assert engine.keys == [token]
    if "max_seconds" in extra:
        assert payload["recorded_max_seconds"] == 120 and "max_seconds=120" in payload["detail"]


@pytest.mark.parametrize("configured", [False, True])
def test_a_retry_accepts_resent_values_equal_to_the_recording(tmp_path, monkeypatch, configured):
    registry, engine, token = _lost_start(tmp_path, monkeypatch, configured=configured)
    payload = _call(registry, retry_of=token, max_seconds=120, effort="low")
    assert payload["status"] == "started", payload
    assert engine.keys == [token, token] and engine.bodies[1] == engine.bodies[0]


def _record(state, *, task_id="mine", body=None, **extra):
    return {"task_id": task_id, "state": state, "run_id": "run-x" if state == "started" else "",
            "request": body, **extra}


@pytest.mark.parametrize("record,text,reason", [
    (None, "", "unknown_invocation"),
    # Each record also fails every later check: the earliest one answers.
    (_record("started", task_id="theirs"), "other", "invocation_not_owned"),
    (_record("started"), "other", "invocation_already_started"),
    (_record("failed_definite"), "other", "invocation_definitely_refused"),
    (_record("pending", body={"model": "m"}), "other", "invocation_request_unrecorded"),
    ({"state": "legacy_gap", "run_id": "", "reason": "invocation_request_missing"}, "", "invocation_request_unrecorded"),
    (_record("pending", body={"prompt": "P"}), " ", "retry_prompt_mismatch"),
    (_record("pending", body={"prompt": "P"}), "P ", "retry_prompt_mismatch"),
])
def test_the_single_validator_answers_in_its_order(monkeypatch, record, text, reason):
    from ouroboros.tools.delegate_integration import _validated_invocation

    monkeypatch.setattr(custody, "invocation_record", lambda *_a, **_k: record)
    found, refusal = _validated_invocation("drive", "inv", "mine", text)
    payload = json.loads(refusal.text)
    assert found is None and payload["reason"] == reason
    if reason == "invocation_already_started":
        assert payload["run_id"] == "run-x"


@pytest.mark.parametrize("text", ["", "P", "note"])
def test_the_validator_reuses_the_record_or_its_continuation_note(monkeypatch, text):
    from ouroboros.tools.delegate_integration import _validated_invocation

    record = _record("pending", body={"prompt": "HOST FACTS\n\nnote" if text == "note" else "P"},
                     continuation_of="run-a" if text == "note" else "",
                     work_order_fingerprint=hashlib.sha256(b"note").hexdigest())
    monkeypatch.setattr(custody, "invocation_record", lambda *_a, **_k: record)
    assert _validated_invocation("drive", "inv", "mine", text) == (record, None)


@pytest.mark.parametrize("recorded", [
    {},  # a legacy row without digests
    {"work_order_fingerprint": "f" * 64, "authority_fingerprint": "a" * 64, "work_order_coverage": "partial",
     "work_order_source_request": {"coverage": "partial", "complete_sha256": "e" * 64}},
])
def test_an_empty_retry_keeps_recorded_fingerprints_and_derives_only_a_missing_digest(tmp_path, monkeypatch, recorded):
    registry, engine, _token = _lost_start(tmp_path, monkeypatch)
    body = engine.bodies[0]
    assert custody.record_start_requested(
        tmp_path, run_id="", task_id=registry._ctx.task_id, invocation_id="inv-kept", idempotency_key="inv-kept",
        max_seconds=120, max_seconds_basis="requested", request=body, project_id="prj-existing",
        project_owned=False, route="some-route", **recorded)
    payload = _call(registry, retry_of="inv-kept")
    assert payload["status"] == "started", payload
    assert engine.keys[-1] == "inv-kept" and engine.bodies[-1] == body
    custody._CUSTODY.clear()
    run = custody.replay(tmp_path)[payload["run_id"]]
    if not recorded:  # the retained prompt's digest, never the empty caller text's
        assert run.work_order_fingerprint == hashlib.sha256(body["prompt"].encode("utf-8")).hexdigest()
        assert run.work_order_fingerprint != hashlib.sha256(b"").hexdigest()
    else:
        assert (run.work_order_fingerprint, run.authority_fingerprint, run.work_order_coverage) == (
            "f" * 64, "a" * 64, "partial")
        assert run.work_order_source_request == recorded["work_order_source_request"]


@pytest.mark.parametrize("drift", ["root_moved", "snapshot_gone", "workspace_inactive"])
def test_an_empty_retry_still_proves_present_target_and_snapshot(tmp_path, monkeypatch, drift):
    from ouroboros.tools import delegate

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(cx, "ClaudexorGateway", lambda *_a, **_k: pytest.fail("a refused retry reached the engine"))
    ctx = _delegating_ctx(tmp_path, acting=True)
    target = tmp_path / "moved" if drift == "root_moved" else ctx.workspace_root
    assert custody.record_start_requested(
        tmp_path, run_id="", task_id=ctx.task_id, idempotency_key="inv-w", invocation_id="inv-w", request={
            "prompt": "edit the docs", "access": "workspace_write", "mode": "agent",
            "scope": {"kind": "project", "root": str(target)},
            "execution": {"isolation": "live", "delegated": True}, "primaryHarness": "some-route"},
        project_id="p", project_owned=False, route="some-route", snapshot_id="snap-gone",
        execution_root=str(tmp_path / "exec"), baseline_sha="b" * 40, target_root=str(target),
        authority_source="acting_constraint")
    if drift == "workspace_inactive":
        ctx.workspace_mode = ""
    reason = json.loads(delegate._delegate_start(ctx, "", retry_of="inv-w").text)["reason"]
    assert reason == {"root_moved": "retry_root_divergence", "snapshot_gone": "execution_snapshot_missing",
                      "workspace_inactive": "workspace_not_active"}[drift]


def test_an_omitted_prompt_keeps_fresh_start_semantics(tmp_path, monkeypatch):
    """Omission is the empty prompt: a direct fresh start still refuses it, and a
    configured fresh start sends its canonical work order, as before (#1614)."""
    registry, engine = _delegate_registry(tmp_path, monkeypatch), _Engine(tmp_path, monkeypatch, lose=0)
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", json.dumps({"enabled": True, "items": [{
        "subagent_id": "reader", "name": "Reader", "recommended_use": "Read and answer.",
        "route": {"kind": "agent_session", "target_id": "some-route=weak-model", "credential_profile_id": ""},
        "effort": "low", "access": "readonly"}]}))
    assert _call(registry, subagent_id="reader")["reason"] == "empty_prompt"
    assert engine.keys == []
    registry._ctx._configured_actor_bootstrap = {
        "selected_subagent_id": "transport-fixture", "canonical_work_order": "THE CANONICAL WORK ORDER",
        "work_order_fingerprint": "w" * 64,
        "snapshot": _transport_snapshot(subagents.get_subagent_harness(), "readonly")}
    payload = _call(registry)
    assert payload["status"] == "started", payload
    assert [body["prompt"] for body in engine.bodies] == ["THE CANONICAL WORK ORDER"]
