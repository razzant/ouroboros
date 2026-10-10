"""A live message into a running delegated run: ``delegate_message``.

The fifth nanny verb places one message into the run's RUNNING turn through the
engine's capability-declared channel (the route row's ``liveInput``, the
``POST /v2/runs/:id/messages`` operation). What these pin: the gateway returns a
typed body at any HTTP status and types every untyped refusal with its code;
the verb refuses malformed calls and foreign runs before any wire call, never
POSTs a FRESH message to a settled run or an incapable route, mirrors the
engine's typed outcomes 1:1 (plus the host's ``not_found``), and keeps custody
of ``message_id`` (the wire Idempotency-Key): returned in every result, replayed
only when handed back, and the replay skips the host short-circuits so the
engine can answer with the stored receipt.
"""

import json
import queue as stdqueue

import pytest


@pytest.fixture(autouse=True)
def _fresh_custody_memo():
    from ouroboros import delegate_custody as custody

    custody._CUSTODY.clear()
    yield
    custody._CUSTODY.clear()


def _ctx(tmp_path, task_id="t-nanny"):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolContext

    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path,
                      task_constraint=TaskConstraint(mode="local_readonly_subagent"))
    ctx.task_id = task_id
    ctx.event_queue = stdqueue.Queue()
    return ctx


def _own_run(run_id="run-1", *, task_id="t-nanny", route_id="some-route", settled=False):
    from ouroboros import delegate_custody as custody

    custody._CUSTODY[run_id] = custody.RunCustody(
        run_id=run_id, task_id=task_id, route_id=route_id, model="m",
        project_id="prj", project_owned=False, settled=settled,
    )


def _row(route_id="some-route", live_input="mid_turn"):
    row = {"id": route_id, "enabled": True, "accessProfilesSupported": ["readonly"]}
    if live_input is not None:
        row["liveInput"] = live_input
    return row


_OPS_WITH_ROUTE = [{"id": "run.message", "method": "POST", "path": "/v2/runs/:id/messages"}]


class _Stub:
    """The gateway as the verb sees it; every wire call is RECORDED."""

    engine_version = "3.16.0"

    def __init__(self, *, result=None, error=None, operations=None, harnesses=None,
                 capability_error=None):
        self.result = result
        self.error = error
        self.operations_rows = _OPS_WITH_ROUTE if operations is None else operations
        self.harnesses = [_row()] if harnesses is None else harnesses
        self.capability_error = capability_error
        self.posts = []
        self.reads = []

    def handshake(self, **_kw):
        return {}

    def operations(self, **_kw):
        self.reads.append("operations")
        if self.capability_error is not None:
            raise self.capability_error
        return list(self.operations_rows)

    def agent_capabilities(self, **_kw):
        self.reads.append("agent_capabilities")
        return {"harnesses": list(self.harnesses)}

    def send_run_message(self, rid, text, *, idempotency_key, expected_attempt_id="",
                         timeout_sec=None):
        self.posts.append({"run_id": rid, "text": text, "key": idempotency_key,
                           "timeout_sec": timeout_sec})
        if self.error is not None:
            raise self.error
        return dict(self.result)

    def close(self):
        pass


def _install(monkeypatch, stub):
    from ouroboros.gateways import claudexor as gw

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: stub)
    return stub


def _events(tmp_path):
    path = tmp_path / "logs" / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _receipts(tmp_path):
    return [e for e in _events(tmp_path) if e["type"] == "delegate_message_outcome"]


# -- the gateway: typed bodies at any status, typed problems otherwise -----------


def test_send_run_message_returns_typed_bodies_and_types_every_refusal(monkeypatch):
    import httpx

    from ouroboros.gateways import claudexor as cx

    replies = {}

    class _Recorder:
        def request(self, method, path, **kwargs):
            replies.update(method=method, path=path, json=kwargs.get("json"),
                           headers=kwargs.get("headers"), timeout=kwargs.get("timeout"))
            return httpx.Response(replies["code"], json=replies["body"])

    gateway = cx.ClaudexorGateway(cx.DaemonEndpoint("127.0.0.1", 1, "secret"))
    gateway.close()
    gateway._client = _Recorder()

    replies.update(code=200, body={"accepted": True, "outcome": "accepted", "runId": "run 1",
                                   "messageId": "m-1", "attemptId": "a01"})
    body = gateway.send_run_message("run 1", "steer left", idempotency_key="m-1",
                                    timeout_sec=12.0)
    assert body["outcome"] == "accepted"
    assert replies["method"] == "POST"
    assert replies["path"] == "/v2/runs/run%201/messages"
    assert replies["json"] == {"text": "steer left"}
    # The message identity IS the wire Idempotency-Key, sent verbatim.
    assert replies["headers"] == {"Idempotency-Key": "m-1"}
    assert replies["timeout"] is not None

    # expectedAttemptId rides only when a caller holds one.
    gateway.send_run_message("run 1", "x", idempotency_key="m-2", expected_attempt_id="a01")
    assert replies["json"] == {"text": "x", "expectedAttemptId": "a01"}

    # A typed outcome is the ANSWER whatever the status (defensive: the engine
    # answers every typed outcome at 200 by contract).
    replies.update(code=200, body={"accepted": False, "outcome": "not_active",
                                   "reason": "run_terminal"})
    assert gateway.send_run_message("run 1", "x", idempotency_key="m-3")["reason"] == "run_terminal"

    # The 409 idempotency problems carry their code AND status.
    replies.update(code=409, body={"code": "idempotency_conflict", "message": "different digest"})
    with pytest.raises(cx.ClaudexorUnavailable) as exc:
        gateway.send_run_message("run 1", "y", idempotency_key="m-3")
    assert (exc.value.status_code, exc.value.code) == (409, "idempotency_conflict")

    # The daemon's own 404 (every daemon 404 has a body) stays the typed refusal it is.
    replies.update(code=404, body={"error": "no such run"})
    with pytest.raises(cx.ClaudexorUnavailable) as exc:
        gateway.send_run_message("run-gone", "x", idempotency_key="m-4")
    assert exc.value.status_code == 404

    # 501: this engine build has no live-message service.
    replies.update(code=501, body={"error": "not supported"})
    with pytest.raises(cx.ClaudexorUnavailable) as exc:
        gateway.send_run_message("run 1", "x", idempotency_key="m-5")
    assert exc.value.status_code == 501

    # A 2xx without a typed outcome is malformed, never an accepted delivery.
    replies.update(code=200, body={"ok": True})
    with pytest.raises(cx.ClaudexorUnavailable) as exc:
        gateway.send_run_message("run 1", "x", idempotency_key="m-6")
    assert exc.value.code == "malformed_response"


def test_send_run_message_transport_death_is_daemon_unreachable():
    import httpx

    from ouroboros.gateways import claudexor as cx

    class _Dead:
        def request(self, *a, **k):
            raise httpx.ConnectError("refused")

    gateway = cx.ClaudexorGateway(cx.DaemonEndpoint("127.0.0.1", 1, "secret"))
    gateway.close()
    gateway._client = _Dead()
    with pytest.raises(cx.ClaudexorUnavailable) as exc:
        gateway.send_run_message("run-1", "x", idempotency_key="m-1")
    assert exc.value.code == "daemon_unreachable" and exc.value.status_code == 0


def test_run_message_supported_reads_the_engine_route_catalog():
    from ouroboros.gateways import claudexor as cx

    assert cx.run_message_supported(_OPS_WITH_ROUTE)
    assert not cx.run_message_supported([])
    assert not cx.run_message_supported([{"method": "GET", "path": "/v2/runs/:id/messages"}])
    assert not cx.run_message_supported([{"method": "POST", "path": "/v2/runs/:id/control"}])
    assert not cx.run_message_supported(["junk", None])


# -- argument and custody refusals (no wire call) ---------------------------------


def test_empty_text_is_an_agent_fault_before_any_wire_call(tmp_path, monkeypatch):
    from ouroboros.delegate_interactions import _delegate_message
    from ouroboros.delegate_shared import AGENT_FAULT_CODE

    stub = _install(monkeypatch, _Stub())
    ctx = _ctx(tmp_path)
    _own_run()
    for text in ("", "   ", None, 7):
        out = _delegate_message(ctx, "run-1", text)
        payload = json.loads(out.text)
        assert payload["reason"] == "message_text_required"
        assert out.code == AGENT_FAULT_CODE and payload["ok"] is False
    out = json.loads(_delegate_message(ctx, "", "hello").text)
    assert out["reason"] == "missing_run_id"
    assert stub.posts == [] and stub.reads == []
    assert _receipts(tmp_path) == []


def test_foreign_and_unknown_runs_are_refused_without_a_wire_call(tmp_path, monkeypatch):
    from ouroboros.delegate_interactions import _delegate_message

    stub = _install(monkeypatch, _Stub())
    ctx = _ctx(tmp_path)
    _own_run(task_id="t-other")
    foreign = json.loads(_delegate_message(ctx, "run-1", "steer").text)
    assert foreign["status"] == "refused" and foreign["reason"] == "run_not_owned"
    assert foreign["owner_task_id"] == "t-other"
    unknown = json.loads(_delegate_message(ctx, "run-nobody", "steer").text)
    assert unknown["reason"] == "run_ownership_unknown"
    assert stub.posts == [] and stub.reads == []


# -- host short-circuits for a FRESH message -----------------------------------------


def test_a_settled_run_is_not_active_without_a_post(tmp_path, monkeypatch):
    from ouroboros.delegate_interactions import _delegate_message
    from ouroboros.delegate_shared import SUBSTRATE_REFUSAL_CODE

    stub = _install(monkeypatch, _Stub())
    ctx = _ctx(tmp_path)
    _own_run(settled=True)
    out = _delegate_message(ctx, "run-1", "steer")
    payload = json.loads(out.text)
    assert payload["status"] == "not_active" and payload["reason"] == "run_settled"
    assert payload["accepted"] is False and out.code == SUBSTRATE_REFUSAL_CODE
    assert payload["message_id"]  # minted and returned even for a host verdict
    assert stub.posts == [] and stub.reads == []
    receipt = _receipts(tmp_path)[0]
    assert receipt["outcome"] == "not_active" and receipt["message_id"] == payload["message_id"]


@pytest.mark.parametrize("stub_kwargs,reason,live_input", [
    ({"operations": []}, "engine_lacks_operation", None),
    ({"harnesses": [_row(live_input="none")]}, "route_live_input_none", "none"),
    ({"harnesses": [_row(live_input=None)]}, "route_live_input_none", "none"),
    ({"harnesses": [_row(route_id="another-route")]}, "route_not_in_capability_catalog", None),
    ({"capability_error": RuntimeError("catalog down")}, "capability_read_failed", "unknown"),
])
def test_an_incapable_route_is_unsupported_without_a_post(tmp_path, monkeypatch,
                                                          stub_kwargs, reason, live_input):
    """A18: the operation must be listed AND the route row must declare a
    liveInput other than none; a failed read is unsupported too, never a guess
    and never a refusal that spends a round."""
    from ouroboros.delegate_interactions import _delegate_message
    from ouroboros.delegate_shared import SUBSTRATE_REFUSAL_CODE

    stub = _install(monkeypatch, _Stub(**stub_kwargs))
    ctx = _ctx(tmp_path)
    _own_run()
    out = _delegate_message(ctx, "run-1", "steer")
    payload = json.loads(out.text)
    assert payload["status"] == "unsupported" and payload["reason"] == reason
    assert payload["live_input"] == live_input
    assert out.code == SUBSTRATE_REFUSAL_CODE and payload["ok"] is False
    assert stub.posts == []
    if reason == "capability_read_failed":
        # A failed READ is not a missing channel: the note keeps the run and never
        # prescribes cancel + restart.
        assert "unknown, not absent" in payload["note"] and "cancel" not in payload["note"]
    else:
        assert "delegate_start" in payload["note"]
        assert "continue_from=<run_id>" in payload["note"]
        assert "prompt=<corrections>" in payload["note"]
        assert "verify settlement" in payload["note"]
        assert "some work may need repeating" in payload["note"]


def test_a_capable_route_reads_both_facts_then_posts_once(tmp_path, monkeypatch):
    from ouroboros.delegate_interactions import _delegate_message

    stub = _install(monkeypatch, _Stub(result={"accepted": True, "outcome": "accepted",
                                                "attemptId": "a01", "harnessId": "codex",
                                                "liveInput": "mid_turn"}))
    ctx = _ctx(tmp_path)
    _own_run()
    _delegate_message(ctx, "run-1", "steer")
    assert stub.reads == ["operations", "agent_capabilities"]
    assert len(stub.posts) == 1


# -- typed outcomes -------------------------------------------------------------------


def test_accepted_relays_typed_facts_and_records_a_digest_never_the_text(tmp_path, monkeypatch):
    import hashlib

    from ouroboros.delegate_interactions import _delegate_message

    text = "MANGO: stop after step 3 and report"
    stub = _install(monkeypatch, _Stub(result={
        "accepted": True, "outcome": "accepted", "runId": "run-1", "messageId": "ignored",
        "attemptId": "a01", "harnessId": "codex", "liveInput": "mid_turn",
        "nativeTurnId": "turn-7"}))
    ctx = _ctx(tmp_path)
    _own_run()
    out = _delegate_message(ctx, "run-1", text)
    payload = json.loads(out.text)
    assert out.status == "ok" and "ok" not in payload
    assert payload["status"] == "accepted" and payload["accepted"] is True
    assert (payload["attempt_id"], payload["harness_id"]) == ("a01", "codex")
    assert (payload["live_input"], payload["native_turn_id"]) == ("mid_turn", "turn-7")
    # The host-minted identity is what was sent AND what comes back.
    assert payload["message_id"] == stub.posts[0]["key"]
    assert len(payload["message_id"]) == 32
    assert "delegate_wait" in payload["note"] and "NEW" in payload["note"]
    receipt = _receipts(tmp_path)[0]
    assert receipt["text_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert receipt["text_chars"] == len(text)
    assert receipt["http_status"] == 200 and receipt["attempt_id"] == "a01"
    assert text not in json.dumps(receipt)


def test_delivered_is_an_ok_observation_with_its_own_note(tmp_path, monkeypatch):
    from ouroboros.delegate_interactions import _delegate_message

    _install(monkeypatch, _Stub(result={"accepted": True, "outcome": "delivered",
                                        "nativeTurnId": "t1"}))
    ctx = _ctx(tmp_path)
    _own_run()
    out = _delegate_message(ctx, "run-1", "steer")
    payload = json.loads(out.text)
    assert out.status == "ok" and payload["status"] == "delivered"
    assert "obedience" in payload["note"]


def test_engine_rejected_classifies_by_reason(tmp_path, monkeypatch):
    """A vendor refusal on an active turn is the substrate saying no; a replayed
    message_id with different text is the caller's own defect."""
    from ouroboros.delegate_interactions import _delegate_message
    from ouroboros.delegate_shared import AGENT_FAULT_CODE, SUBSTRATE_REFUSAL_CODE

    ctx = _ctx(tmp_path)
    _own_run()
    _install(monkeypatch, _Stub(result={"accepted": False, "outcome": "rejected",
                                        "reason": "rpc_refused", "message": "turn/steer refused"}))
    out = _delegate_message(ctx, "run-1", "steer")
    payload = json.loads(out.text)
    assert payload["status"] == "rejected" and payload["reason"] == "rpc_refused"
    assert out.code == SUBSTRATE_REFUSAL_CODE and payload["detail"] == "turn/steer refused"
    assert "NEW message_id" in payload["note"]

    _install(monkeypatch, _Stub(result={"accepted": False, "outcome": "rejected",
                                        "reason": "multi_attempt"}))
    assert _delegate_message(ctx, "run-1", "steer").code == SUBSTRATE_REFUSAL_CODE

    from ouroboros.gateways import claudexor as cx

    _install(monkeypatch, _Stub(error=cx.ClaudexorUnavailable(
        "idempotency_conflict", "different digest", status_code=409)))
    out = _delegate_message(ctx, "run-1", "steer", message_id="m-old")
    payload = json.loads(out.text)
    assert payload["status"] == "rejected" and payload["reason"] == "idempotency_conflict"
    assert out.code == AGENT_FAULT_CODE


def test_engine_not_active_and_unsupported_are_substrate_refusals(tmp_path, monkeypatch):
    from ouroboros.delegate_interactions import _delegate_message
    from ouroboros.delegate_shared import SUBSTRATE_REFUSAL_CODE

    ctx = _ctx(tmp_path)
    _own_run()
    _install(monkeypatch, _Stub(result={"accepted": False, "outcome": "not_active",
                                        "reason": "interaction_pending", "attemptId": "a01"}))
    out = _delegate_message(ctx, "run-1", "steer")
    payload = json.loads(out.text)
    assert (payload["status"], payload["reason"]) == ("not_active", "interaction_pending")
    assert out.code == SUBSTRATE_REFUSAL_CODE and "delegate_answer" in payload["note"]

    _install(monkeypatch, _Stub(result={"accepted": False, "outcome": "unsupported",
                                        "reason": "thread_bound"}))
    out = _delegate_message(ctx, "run-1", "steer")
    payload = json.loads(out.text)
    assert (payload["status"], payload["reason"]) == ("unsupported", "thread_bound")
    assert out.code == SUBSTRATE_REFUSAL_CODE


def test_payload_verdict_4xx_is_rejected_as_an_agent_fault(tmp_path, monkeypatch):
    from ouroboros.delegate_interactions import _delegate_message
    from ouroboros.delegate_shared import AGENT_FAULT_CODE
    from ouroboros.gateways import claudexor as cx

    ctx = _ctx(tmp_path)
    _own_run()
    _install(monkeypatch, _Stub(error=cx.ClaudexorUnavailable(
        "invalid_request", "text too long", status_code=400)))
    out = _delegate_message(ctx, "run-1", "steer")
    payload = json.loads(out.text)
    assert payload["status"] == "rejected" and payload["reason"] == "invalid_request"
    assert out.code == AGENT_FAULT_CODE
    assert _receipts(tmp_path)[0]["http_status"] == 400


@pytest.mark.parametrize("code,status", [
    ("daemon_unreachable", 0),          # transport death
    ("http_503", 503),                  # any 5xx
    ("internal_error", 500),            # the receipt-save failure path
    ("delivery_in_progress", 409),      # the idempotency store: fate unknown
    ("delivery_interrupted", 409),
    ("http_429", 429),                  # says nothing about delivery
])
def test_ambiguous_failures_are_delivery_unknown_naming_the_same_id_retry(
        tmp_path, monkeypatch, code, status):
    from ouroboros.delegate_interactions import _delegate_message
    from ouroboros.gateways import claudexor as cx

    ctx = _ctx(tmp_path)
    _own_run()
    _install(monkeypatch, _Stub(error=cx.ClaudexorUnavailable(code, "boom", status_code=status)))
    out = _delegate_message(ctx, "run-1", "steer")
    payload = json.loads(out.text)
    # Recorded as an OK observation (the fate is unknown, not refused), with
    # the reason relayed and the recovery named: the SAME message_id.
    assert out.status == "ok" and "ok" not in payload
    assert payload["status"] == "delivery_unknown" and payload["reason"] == code
    assert payload["accepted"] is False
    assert "SAME message_id" in payload["note"] and "different" in payload["note"]
    assert _receipts(tmp_path)[0]["http_status"] == status


def test_any_404_after_positive_reads_is_not_found_with_custody_untouched(tmp_path, monkeypatch):
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_interactions import _delegate_message
    from ouroboros.delegate_shared import SUBSTRATE_REFUSAL_CODE
    from ouroboros.gateways import claudexor as cx

    ctx = _ctx(tmp_path)
    _own_run()
    stub = _install(monkeypatch, _Stub(error=cx.ClaudexorUnavailable(
        "http_404", "no such run", status_code=404)))
    out = _delegate_message(ctx, "run-1", "steer")
    payload = json.loads(out.text)
    assert payload["status"] == "not_found" and out.code == SUBSTRATE_REFUSAL_CODE
    assert stub.reads == ["operations", "agent_capabilities"] and len(stub.posts) == 1
    # Custody is not closed by this path: the run row stays open and unsettled.
    assert custody._CUSTODY["run-1"].settled is False
    assert not [e for e in _events(tmp_path) if e["type"].startswith("delegate_run_")]


def test_an_untyped_host_failure_is_delivery_unknown_never_a_traceback(tmp_path, monkeypatch):
    from ouroboros.delegate_interactions import _delegate_message

    class _Broken(_Stub):
        def send_run_message(self, *a, **k):
            raise KeyError("wire shape")

    ctx = _ctx(tmp_path)
    _own_run()
    _install(monkeypatch, _Broken())
    payload = json.loads(_delegate_message(ctx, "run-1", "steer").text)
    assert payload["status"] == "delivery_unknown" and payload["reason"] == "host_exception"
    assert "KeyError" in payload["detail"]


def test_a_dead_daemon_at_handshake_is_a_typed_refusal(tmp_path, monkeypatch):
    from ouroboros.delegate_interactions import _delegate_message
    from ouroboros.gateways import claudexor as cx

    class _Down(_Stub):
        def handshake(self, **_kw):
            raise cx.ClaudexorUnavailable("daemon_unreachable", "no socket")

    ctx = _ctx(tmp_path)
    _own_run()
    stub = _install(monkeypatch, _Down())
    payload = json.loads(_delegate_message(ctx, "run-1", "steer").text)
    assert payload["status"] == "refused" and payload["reason"] == "daemon_unreachable"
    assert payload["message_id"] and stub.posts == []


def test_a_spent_budget_before_the_post_sends_nothing(tmp_path, monkeypatch):
    """The internal deadline sits strictly below the 120 s ToolEntry timeout;
    spent before the POST, the answer is typed and nothing was sent — the
    same-id retry the note prescribes is exactly right (the key is unused)."""
    import ouroboros.delegate_interactions as interactions
    from ouroboros.delegate_interactions import _delegate_message

    assert interactions._MESSAGE_DEADLINE_SEC < 120
    monkeypatch.setattr(interactions, "_MESSAGE_DEADLINE_SEC", -1.0)
    ctx = _ctx(tmp_path)
    _own_run()
    stub = _install(monkeypatch, _Stub())
    payload = json.loads(_delegate_message(ctx, "run-1", "steer").text)
    assert payload["status"] == "delivery_unknown" and payload["reason"] == "deadline_exhausted"
    assert "nothing was sent" in payload["detail"]
    assert stub.posts == []


# -- message_id custody (A16/A27) -------------------------------------------------------


def test_a_returned_message_id_replays_under_the_same_key_skipping_the_short_circuits(
        tmp_path, monkeypatch):
    from ouroboros.delegate_interactions import _delegate_message

    ctx = _ctx(tmp_path)
    # The replay must reach the engine even though the host would refuse a FRESH
    # message here (settled custody, an engine catalog without the operation):
    # the daemon holds the stored receipt and is the only party that can answer.
    _own_run(settled=True)
    stub = _install(monkeypatch, _Stub(operations=[], result={
        "accepted": True, "outcome": "accepted", "attemptId": "a01"}))
    payload = json.loads(_delegate_message(ctx, "run-1", "steer", message_id="m-earlier").text)
    assert payload["status"] == "accepted" and payload["message_id"] == "m-earlier"
    assert stub.reads == []
    assert stub.posts == [{"run_id": "run-1", "text": "steer", "key": "m-earlier",
                           "timeout_sec": stub.posts[0]["timeout_sec"]}]


def test_a_fresh_call_mints_a_new_id_each_time(tmp_path, monkeypatch):
    from ouroboros.delegate_interactions import _delegate_message

    ctx = _ctx(tmp_path)
    _own_run()
    stub = _install(monkeypatch, _Stub(result={"accepted": True, "outcome": "accepted"}))
    first = json.loads(_delegate_message(ctx, "run-1", "same text").text)["message_id"]
    second = json.loads(_delegate_message(ctx, "run-1", "same text").text)["message_id"]
    # Never a content-stable key: the same text twice is two deliveries.
    assert first != second and [p["key"] for p in stub.posts] == [first, second]


# -- the registration surfaces --------------------------------------------------------


def test_the_message_verb_is_registered_on_every_contract_surface():
    import ouroboros.delegate_interactions as interactions
    from ouroboros.delegate_shared import _AGENT_FAULT_REASONS
    from ouroboros.nanny_pacing import BASELINE_RESET_TOOLS, DELEGATE_ACTIVITY_TOOLS
    from ouroboros.tool_capabilities import (
        ACTING_SUBAGENT_TOOL_NAMES,
        LOCAL_READONLY_SUBAGENT_TOOL_NAMES,
    )
    from ouroboros.tools import delegate

    entries = {entry.name: entry for entry in delegate.get_tools()}
    assert "delegate_message" in entries
    entry = entries["delegate_message"]
    assert entry.timeout_sec == 120 and entry.timeout_sec > interactions._MESSAGE_DEADLINE_SEC
    schema = entry.schema["parameters"]
    assert schema["required"] == ["run_id", "text"]
    assert set(schema["properties"]) == {"run_id", "text", "message_id"}
    assert "delegate_message" in LOCAL_READONLY_SUBAGENT_TOOL_NAMES
    assert "delegate_message" in ACTING_SUBAGENT_TOOL_NAMES
    # Supervision activity, never a burn-baseline reset (only start/schedule are).
    assert "delegate_message" in DELEGATE_ACTIVITY_TOOLS
    assert "delegate_message" not in BASELINE_RESET_TOOLS
    assert {"message_text_required", "idempotency_conflict"} <= _AGENT_FAULT_REASONS
    assert delegate._delegate_message is interactions._delegate_message


def test_descriptions_are_capability_based_not_harness_named():
    from ouroboros.tools import delegate

    text = json.dumps([entry.schema for entry in delegate.get_tools()])
    assert "Codex-lane runs have no mid-run questions" not in text
    assert "liveInput" in text
    assert "message.* rows" in text  # delegate_wait names the timeline as the reconciler
    cancel = next(entry.schema["description"] for entry in delegate.get_tools()
                  if entry.name == "delegate_cancel")
    assert "continue_from=<run_id>" in cancel and "prompt=<corrections>" in cancel
    assert "verified settlement" in cancel and "Only confirmed" in cancel
    assert "retained" in cancel and "some work may need repeating" in cancel
    assert "loses all" not in cancel and "start over" not in cancel


def test_the_wait_timeline_keeps_message_receipts(tmp_path):
    from ouroboros.delegate_progress import timeline_tail

    rows = timeline_tail({"timeline": [
        {"type": "message.accepted", "title": "Live message accepted (12 bytes)",
         "severity": "info", "attemptId": "a01", "messageId": "m-1", "outcome": "accepted"},
        {"type": "message.delivered", "title": "Live message delivered (12 bytes)",
         "severity": "info", "messageId": "m-1", "outcome": "delivered"},
        {"type": "harness.started", "title": "started", "severity": "info", "outcome": 7},
    ]})
    assert (rows[0]["messageId"], rows[0]["outcome"], rows[0]["attemptId"]) == ("m-1", "accepted", "a01")
    assert (rows[1]["messageId"], rows[1]["outcome"]) == ("m-1", "delivered")
    assert "outcome" not in rows[2] and "messageId" not in rows[2]


# -- the fake daemon's contract, pinned against the REAL gateway --------------------------


def test_fake_daemon_run_message_contract(tmp_path):
    from ouroboros.gateways.claudexor import (
        ClaudexorGateway,
        ClaudexorUnavailable,
        discover_daemon_at,
        run_message_supported,
    )
    from tests.system_e2e.interfaces import FAKE_ASK_MARKER, FAKE_HANG_MARKER, FakeClaudexorDaemon

    with FakeClaudexorDaemon(runs_dir=tmp_path / "runs") as daemon:
        daemon.install(tmp_path / "cx")
        with ClaudexorGateway(discover_daemon_at(tmp_path / "cx")) as gateway:
            gateway.handshake()
            # Both discovery facts the verb negotiates are served.
            assert run_message_supported(gateway.operations())
            row = gateway.agent_capabilities()["harnesses"][0]
            assert row["id"] == daemon.harness_id and row["liveInput"] == "mid_turn"
            request = {
                "prompt": FAKE_HANG_MARKER + " keep running", "instructions": "i",
                "authPreference": "subscription", "mode": "ask",
                "scope": {"kind": "project", "root": str(tmp_path)},
                "harnesses": [daemon.harness_id], "primaryHarness": daemon.harness_id,
                "access": "readonly", "maxSeconds": 60,
            }
            run_id = str(gateway.start_run(request, idempotency_key="inv-msg-1")["runId"])
            first = gateway.send_run_message(run_id, "steer left", idempotency_key="m-1")
            assert first["outcome"] == "accepted" and first["accepted"] is True
            assert first["messageId"] == "m-1" and first["runId"] == run_id
            assert first["liveInput"] == "mid_turn" and first["attemptId"] == "a01"
            # A replay under the same key is the STORED receipt, never a second delivery.
            assert gateway.send_run_message(run_id, "steer left", idempotency_key="m-1") == first
            assert [m["text"] for m in daemon.runs[run_id]["messages"]] == ["steer left"]
            # The same key with different text is the typed 409.
            with pytest.raises(ClaudexorUnavailable) as exc:
                gateway.send_run_message(run_id, "steer RIGHT", idempotency_key="m-1")
            assert (exc.value.status_code, exc.value.code) == (409, "idempotency_conflict")
            # The key is REQUIRED on the wire.
            with pytest.raises(ClaudexorUnavailable) as exc:
                gateway.send_run_message(run_id, "x", idempotency_key="")
            assert exc.value.code == "missing_idempotency_key"
            # A run parked on a question is never steered beside it (INV-048).
            asking = str(gateway.start_run({**request, "prompt": FAKE_ASK_MARKER + " q"},
                                           idempotency_key="inv-msg-2")["runId"])
            parked = gateway.send_run_message(asking, "steer", idempotency_key="m-2")
            assert (parked["outcome"], parked["reason"]) == ("not_active", "interaction_pending")
            # An unknown run is the daemon's own 404, with a body.
            with pytest.raises(ClaudexorUnavailable) as exc:
                gateway.send_run_message("run-nobody", "x", idempotency_key="m-3")
            assert exc.value.status_code == 404
            posts = daemon.calls("POST", f"/v2/runs/{run_id}/messages")
            assert [p["idempotency_key"] for p in posts] == ["m-1", "m-1", "m-1", ""]
            assert posts[0]["body"] == {"text": "steer left"}


def test_fake_daemon_route_without_live_input_answers_unsupported(tmp_path):
    from ouroboros.gateways.claudexor import ClaudexorGateway, discover_daemon_at
    from tests.system_e2e.interfaces import FAKE_HANG_MARKER, FakeClaudexorDaemon

    with FakeClaudexorDaemon(runs_dir=tmp_path / "runs", live_input="none") as daemon:
        daemon.install(tmp_path / "cx")
        with ClaudexorGateway(discover_daemon_at(tmp_path / "cx")) as gateway:
            gateway.handshake()
            assert gateway.agent_capabilities()["harnesses"][0]["liveInput"] == "none"
            run_id = str(gateway.start_run({
                "prompt": FAKE_HANG_MARKER, "instructions": "i", "authPreference": "subscription",
                "mode": "ask", "scope": {"kind": "project", "root": str(tmp_path)},
                "harnesses": [daemon.harness_id], "primaryHarness": daemon.harness_id,
                "access": "readonly", "maxSeconds": 60,
            }, idempotency_key="inv-none-1")["runId"])
            body = gateway.send_run_message(run_id, "steer", idempotency_key="m-1")
            assert (body["outcome"], body["reason"]) == ("unsupported", "no_live_session")
            assert "messages" not in daemon.runs[run_id]


def test_the_verb_end_to_end_against_the_fake_daemon(tmp_path, monkeypatch):
    """The whole verb over the loopback fake: discovery, POST, typed receipt."""
    from ouroboros.delegate_interactions import _delegate_message
    from ouroboros.gateways import claudexor as cx
    from tests.system_e2e.interfaces import FAKE_HANG_MARKER, FakeClaudexorDaemon

    with FakeClaudexorDaemon(runs_dir=tmp_path / "runs") as daemon:
        daemon.install(tmp_path / "cx")
        endpoint = cx.discover_daemon_at(tmp_path / "cx")
        # The verb builds ``ClaudexorGateway()`` bare; point its discovery at the fake.
        monkeypatch.setattr(cx, "discover_daemon", lambda home=None: endpoint)
        with cx.ClaudexorGateway(endpoint) as setup:
            setup.handshake()
            run_id = str(setup.start_run({
                "prompt": FAKE_HANG_MARKER, "instructions": "i", "authPreference": "subscription",
                "mode": "ask", "scope": {"kind": "project", "root": str(tmp_path)},
                "harnesses": [daemon.harness_id], "primaryHarness": daemon.harness_id,
                "access": "readonly", "maxSeconds": 60,
            }, idempotency_key="inv-e2e-1")["runId"])
        ctx = _ctx(tmp_path)
        _own_run(run_id, route_id=daemon.harness_id)
        payload = json.loads(_delegate_message(ctx, run_id, "steer left").text)
        assert payload["status"] == "accepted" and payload["accepted"] is True
        assert payload["harness_id"] == daemon.harness_id and payload["live_input"] == "mid_turn"
        posts = daemon.calls("POST", f"/v2/runs/{run_id}/messages")
        assert len(posts) == 1 and posts[0]["idempotency_key"] == payload["message_id"]
        # The replay reaches the daemon and comes back as the stored receipt.
        again = json.loads(_delegate_message(ctx, run_id, "steer left",
                                             message_id=payload["message_id"]).text)
        assert again["status"] == "accepted" and again["message_id"] == payload["message_id"]
        assert len(daemon.runs[run_id]["messages"]) == 1


# -- current questions accompanying an interaction_pending refusal ----------------


def _message_question(iid="int-current", question="Which port should we use?"):
    return {
        "interactionId": iid, "sourceTool": "AskUserQuestion",
        "requestedAt": "2026-10-08T18:00:00Z", "timeoutAt": None,
        "questions": [{"id": "q-port", "question": question, "header": "Port",
                       "options": [{"label": "8080", "description": "default port"},
                                   {"label": "9090", "description": "alternate"}],
                       "multi_select": False}],
    }


class _QuestionStub(_Stub):
    def __init__(self, *, detail=None, detail_error=None, **kwargs):
        kwargs.setdefault("result", {
            "accepted": False, "outcome": "not_active", "reason": "interaction_pending",
            "message": "Answer the pending question", "attemptId": "a01",
        })
        super().__init__(**kwargs)
        self.detail = {"pendingInteractions": [_message_question()]} if detail is None else detail
        self.detail_error = detail_error
        self.detail_reads = []
        self.answers = []
        self.closed = 0

    def get_run(self, rid, *, timeout_sec=None):
        self.detail_reads.append((rid, timeout_sec))
        if self.detail_error:
            raise self.detail_error
        return self.detail

    def answer_interaction(self, *args, **kwargs):
        self.answers.append((args, kwargs))
        return {"status": "delivered"}

    def close(self):
        self.closed += 1


@pytest.mark.serial
@pytest.mark.parametrize("message_id", ["", "m-replayed-refusal"])
def test_pending_refusal_includes_current_answerable_questions_without_consuming_wait_news(
        tmp_path, monkeypatch, message_id):
    from ouroboros import delegate_interactions as interactions
    from ouroboros.delegate_shared import SUBSTRATE_REFUSAL_CODE
    from ouroboros.gateways.claudexor import pending_interactions

    ctx = _ctx(tmp_path)
    _own_run(settled=bool(message_id))  # a replay bypasses fresh-message prechecks
    stub = _install(monkeypatch, _QuestionStub())
    memo = {"other-run": frozenset({"other-question"})}
    monkeypatch.setattr(interactions, "_REPORTED_INTERACTIONS", memo)
    out = interactions._delegate_message(ctx, "run-1", "steer", message_id=message_id)
    payload = json.loads(out.text)
    assert (payload["status"], payload["reason"], payload["accepted"]) == (
        "not_active", "interaction_pending", False)
    assert out.code == SUBSTRATE_REFUSAL_CODE and payload["ok"] is False
    assert payload["attempt_id"] == "a01" and payload["detail"] == "Answer the pending question"
    assert payload["message_id"] == (message_id or stub.posts[0]["key"])
    assert payload["pending_interactions"] == pending_interactions(stub.detail)
    assert payload["interaction_observation"]["status"] == "observed"
    assert "continuation" not in payload and "continuation_note" not in payload
    assert "state" not in payload and "waiting_on_user" not in payload
    assert "stored receipt" in payload["interaction_observation"]["note"]
    assert "delegate_answer" in payload["interaction_observation"]["note"]
    assert len(stub.detail_reads) == 1 and 0 < stub.detail_reads[0][1] <= 60
    assert stub.reads == ([] if message_id else ["operations", "agent_capabilities"])
    assert stub.answers == [] and stub.closed == 1
    assert memo == {"other-run": frozenset({"other-question"})}
    receipts = _receipts(tmp_path)
    assert len(receipts) == 1 and receipts[0]["outcome"] == "not_active"
    assert receipts[0]["reason"] == "interaction_pending"
    assert receipts[0]["message_id"] == payload["message_id"]
    assert "pending_interactions" not in receipts[0]


@pytest.mark.serial
@pytest.mark.parametrize("outcome,reason", [
    ("accepted", ""), ("delivered", ""), ("rejected", "rpc_refused"),
    ("unsupported", "interaction_pending"), ("not_active", "run_terminal"),
    ("delivery_unknown", "interaction_pending"),
])
def test_unrelated_message_outcomes_do_not_read_run_detail(tmp_path, monkeypatch, outcome, reason):
    from ouroboros.delegate_interactions import _delegate_message

    ctx = _ctx(tmp_path)
    _own_run()
    stub = _install(monkeypatch, _QuestionStub(result={"outcome": outcome, "reason": reason}))
    payload = json.loads(_delegate_message(ctx, "run-1", "steer").text)
    assert payload["status"] == outcome
    assert "interaction_observation" not in payload and "pending_interactions" not in payload
    assert stub.detail_reads == [] and stub.answers == [] and stub.closed == 1


@pytest.mark.serial
@pytest.mark.parametrize("detail,error", [
    ({}, None), ({"pendingInteractions": None}, None),
    ({"pendingInteractions": "unreadable"}, None),
    ({"pendingInteractions": ["invalid row"]}, None),
    ({"pendingInteractions": [{}]}, None),
    ({"pendingInteractions": [{"interactionId": "i", "questions": []}]}, None),
    ({"pendingInteractions": [{"interactionId": "i", "questions": [None]}]}, None),
    ({"pendingInteractions": [_message_question(), {"interactionId": "bad"}]}, None),
    ({"pendingInteractions": [{**_message_question(), "attemptId": {"invalid": "identity"}}]}, None),
    ({"pendingInteractions": [{"interactionId": "i", "questions": [
        {"id": "q", "question": "Q?", "options": ["invalid option"]}]}]}, None),
    ({"pendingInteractions": []}, RuntimeError("retained read failure")),
])
def test_failed_question_observation_preserves_the_refusal_as_unknown(
        tmp_path, monkeypatch, detail, error):
    from ouroboros.delegate_interactions import _delegate_message
    from ouroboros.delegate_shared import SUBSTRATE_REFUSAL_CODE

    ctx = _ctx(tmp_path)
    _own_run()
    stub = _install(monkeypatch, _QuestionStub(detail=detail, detail_error=error))
    out = _delegate_message(ctx, "run-1", "steer", message_id="m-refused")
    payload = json.loads(out.text)
    assert (payload["status"], payload["reason"], payload["message_id"]) == (
        "not_active", "interaction_pending", "m-refused")
    assert out.code == SUBSTRATE_REFUSAL_CODE and payload["detail"] == "Answer the pending question"
    assert "pending_interactions" not in payload
    observation = payload["interaction_observation"]
    assert (observation["status"], observation["reason"]) == ("unavailable", "detail_read_failed")
    assert "unknown" in observation["note"] and "delegate_wait" in observation["note"]
    assert ("retained read failure" if error else "pendingInteractions") in observation["detail"]
    assert len(stub.detail_reads) == 1 and stub.closed == 1 and stub.answers == []
    assert [r["outcome"] for r in _receipts(tmp_path)] == ["not_active"]


@pytest.mark.serial
@pytest.mark.parametrize("message_id", ["", "m-stored"])
def test_question_resolved_between_refusal_and_read_is_an_empty_current_observation(
        tmp_path, monkeypatch, message_id):
    from ouroboros.delegate_interactions import _delegate_message

    ctx = _ctx(tmp_path)
    _own_run()
    stub = _install(monkeypatch, _QuestionStub(detail={"pendingInteractions": []}))
    payload = json.loads(_delegate_message(ctx, "run-1", "steer", message_id=message_id).text)
    assert (payload["status"], payload["reason"]) == ("not_active", "interaction_pending")
    assert payload["pending_interactions"] == []
    assert payload["interaction_observation"]["status"] == "empty"
    assert "may have resolved" in payload["interaction_observation"]["note"]
    assert stub.answers == [] and len(stub.detail_reads) == 1 and stub.closed == 1


@pytest.mark.serial
@pytest.mark.parametrize("remaining", [0.0, 0.25])
def test_question_observation_uses_only_the_remaining_message_budget(
        tmp_path, monkeypatch, remaining):
    from types import SimpleNamespace
    from ouroboros import delegate_interactions as interactions

    clock = [0.0]
    monkeypatch.setattr(interactions, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    class _SlowPost(_QuestionStub):
        def send_run_message(self, *args, **kwargs):
            result = super().send_run_message(*args, **kwargs)
            clock[0] = interactions._MESSAGE_DEADLINE_SEC - remaining
            return result

    ctx = _ctx(tmp_path)
    _own_run()
    stub = _install(monkeypatch, _SlowPost())
    payload = json.loads(interactions._delegate_message(ctx, "run-1", "steer").text)
    assert (payload["status"], payload["reason"]) == ("not_active", "interaction_pending")
    if remaining:
        assert stub.detail_reads == [("run-1", remaining)]
        assert payload["interaction_observation"]["status"] == "observed"
    else:
        assert stub.detail_reads == [] and "pending_interactions" not in payload
        assert payload["interaction_observation"]["status"] == "not_read"
        assert payload["interaction_observation"]["reason"] == "deadline_exhausted"
    assert stub.closed == 1 and stub.answers == []


@pytest.mark.serial
@pytest.mark.parametrize("shape", ["many_questions", "long_answer_keys", "long_label"])
def test_large_pending_questions_fit_and_are_retrievable_with_complete_refusal(
        tmp_path, monkeypatch, shape):
    import hashlib
    from pathlib import Path
    from ouroboros.delegate_interactions import _delegate_message
    from ouroboros.gateways.claudexor import pending_interactions
    from ouroboros.loop_tool_execution import _truncate_tool_result
    from ouroboros.tool_capabilities import tool_result_limit
    from ouroboros.tools.core import _read_file

    if shape == "many_questions":
        rows = [_message_question(f"int-{i}", f"Question {i}: " + "Q" * 1700) for i in range(18)]
    elif shape == "long_answer_keys":
        rows = [_message_question("I" * 16_000)]
        rows[0]["questions"][0]["id"] = "K" * 16_000
    else:
        rows = [_message_question()]
        rows[0]["questions"][0]["options"][0]["label"] = "Exact label: " + "L" * 30_000
    ctx = _ctx(tmp_path)
    _own_run()
    stub = _install(monkeypatch, _QuestionStub(detail={"pendingInteractions": rows}))
    out = _delegate_message(ctx, "run-1", "steer", message_id="m-large")
    assert len(out.text) <= tool_result_limit("delegate_message")
    assert _truncate_tool_result(out.text, "delegate_message", {}) == out.text
    payload = json.loads(out.text)
    assert (payload["status"], payload["reason"], payload["message_id"]) == (
        "not_active", "interaction_pending", "m-large")
    assert payload["interactions_omitted"] == len(rows) - len(payload["pending_interactions"])
    if shape == "long_answer_keys":
        assert payload["pending_interactions"] == []  # keys are omitted whole, never altered
    delivery = payload["interactions_delivery"]
    assert delivery["complete"] is False
    assert "not exact selectable values" in delivery["note"]
    artifact = delivery["artifact"]
    raw = Path(artifact["abs_path"]).read_bytes()
    assert artifact["bytes"] == len(raw) and artifact["sha256"] == hashlib.sha256(raw).hexdigest()
    complete = json.loads(raw)
    assert complete["pending_interactions"] == pending_interactions(stub.detail)
    for key in ("status", "reason", "message_id", "accepted", "ok", "host_code", "detail"):
        assert complete[key] == payload[key]
    read_args = {key: value for key, value in delivery["read_next"].items() if key != "tool"}
    view = _read_file(ctx, **read_args)
    delivered = _truncate_tool_result(view, "read_file", read_args)
    assert raw.decode("utf-8") in delivered  # the readonly nanny can read the entire source
    assert stub.closed == 1 and stub.answers == []


@pytest.mark.serial
@pytest.mark.parametrize("failure", ["render", "stage", "observe"])
def test_question_delivery_failure_never_replaces_the_message_receipt(
        tmp_path, monkeypatch, failure):
    from ouroboros import delegate_interactions as interactions
    from ouroboros.delegate_shared import SUBSTRATE_REFUSAL_CODE

    def _broken(*_args, **_kwargs):
        raise RuntimeError("retained rendering failure")

    if failure == "render":
        monkeypatch.setattr(interactions, "_interactions_payload", _broken)
    elif failure == "observe":
        monkeypatch.setattr(interactions, "_message_pending_observation", _broken)
    else:
        monkeypatch.setattr(interactions, "_stage_full_output", lambda *_args, **_kwargs: None)
    ctx = _ctx(tmp_path)
    _own_run()
    stub = _install(monkeypatch, _QuestionStub(detail={
        "pendingInteractions": [_message_question(question="Q" * 30_000)],
    }))
    out = interactions._delegate_message(ctx, "run-1", "steer", message_id="m-original")
    payload = json.loads(out.text)
    assert (payload["status"], payload["reason"], payload["message_id"]) == (
        "not_active", "interaction_pending", "m-original")
    assert out.code == SUBSTRATE_REFUSAL_CODE and payload["detail"] == "Answer the pending question"
    if failure in ("render", "observe"):
        assert "pending_interactions" not in payload
        assert payload["interaction_observation"]["reason"] == (
            "detail_render_failed" if failure == "render" else "detail_read_failed")
        assert "retained rendering failure" in payload["interaction_observation"]["detail"]
    else:
        delivery = payload["interactions_delivery"]
        assert delivery["complete"] is False and delivery["artifact"] is None
        assert "UNRECOVERABLE" in delivery["note"]
    assert [r["outcome"] for r in _receipts(tmp_path)] == ["not_active"]
    assert stub.closed == 1 and stub.answers == []


def test_pending_normalizer_retains_only_provided_observation_identities():
    from ouroboros.gateways.claudexor import pending_interactions

    row = _message_question()
    supplied = {**row, "runId": "run-observed", "attemptId": "a-current", "harnessId": "h-current"}
    normalized = pending_interactions({"pendingInteractions": [row, supplied, {
        **row, "runId": None, "attemptId": None, "harnessId": None,
    }]}, strict=True)
    assert not {"run_id", "attempt_id", "harness_id"}.intersection(normalized[0])
    assert {key: normalized[1][key] for key in ("run_id", "attempt_id", "harness_id")} == {
        "run_id": "run-observed", "attempt_id": "a-current", "harness_id": "h-current",
    }
    assert {key: normalized[2][key] for key in ("run_id", "attempt_id", "harness_id")} == {
        "run_id": None, "attempt_id": None, "harness_id": None,
    }


@pytest.mark.parametrize("refusal_attempt", [None, "a-original"])
def test_replayed_refusal_observes_changed_then_vanished_questions_without_borrowing_attempt(
        tmp_path, monkeypatch, refusal_attempt):
    from ouroboros import delegate_interactions as interactions

    ctx = _ctx(tmp_path)
    _own_run(settled=True)
    stub = _install(monkeypatch, _QuestionStub(result={
        "outcome": "not_active", "reason": "interaction_pending", "attemptId": refusal_attempt,
    }))
    memo = {"run-1": frozenset({"int-old"})}
    monkeypatch.setattr(interactions, "_REPORTED_INTERACTIONS", memo)
    for iid in ("int-current", "int-replacement", None):
        stub.detail = {"pendingInteractions": [{
            **_message_question(iid), "runId": "run-1", "attemptId": "a-current", "harnessId": "h-current",
        }] if iid else []}
        payload = json.loads(interactions._delegate_message(
            ctx, "run-1", "original steer", message_id="old-refusal").text)
        assert payload["attempt_id"] == refusal_attempt
        assert payload["status"] == "not_active" and payload["accepted"] is False
        assert payload["message_id"] == "old-refusal" and payload["ok"] is False
        if iid:
            current = payload["pending_interactions"][0]
            assert current["interaction_id"] == iid and current["attempt_id"] == "a-current"
            assert current["harness_id"] == "h-current" and current["run_id"] == "run-1"
        else:
            assert payload["pending_interactions"] == []
            assert "does not prove an answer was delivered" in payload["interaction_observation"]["note"]
        assert "continuation" not in payload and "state" not in payload
    assert len(stub.detail_reads) == len(stub.posts) == stub.closed == 3
    assert {p["key"] for p in stub.posts} == {"old-refusal"} and not stub.answers
    assert memo == {"run-1": frozenset({"int-old"})}
    # The next real wait still recognizes the newly observed question as news.
    assert interactions._interactions_are_news("run-1", [{"interaction_id": "int-replacement"}])


@pytest.mark.parametrize("source_fails", [False, True])
@pytest.mark.parametrize("detail_shape", ["questions", "empty", "missing"])
def test_complete_published_refusal_envelope_fits_with_huge_receipt_and_observation_fields(
        tmp_path, monkeypatch, source_fails, detail_shape):
    import hashlib
    from pathlib import Path
    from ouroboros import delegate_interactions as interactions
    from ouroboros.delegate_shared import SUBSTRATE_REFUSAL_CODE
    from ouroboros.loop_tool_execution import _truncate_tool_result
    from ouroboros.tool_capabilities import tool_result_limit
    from ouroboros.tools.delegate import get_tools
    from ouroboros.tools.tool_result import (
        _install_tool_result_sidecar, _published_tool_result, _restore_tool_result_sidecar,
    )

    rid, mid, aid = "run-" + "R" * 20_000, "message-" + "M" * 20_000, "attempt-" + "A" * 20_000
    question = {**_message_question("int-" + "I" * 20_000),
                "runId": rid, "attemptId": "current-" + "C" * 20_000}
    question["questions"][0]["id"] = "question-" + "Q" * 20_000
    question["questions"][0]["options"][0]["label"] = "label-" + "L" * 20_000
    detail = {"pendingInteractions": [question] if detail_shape == "questions" else []}
    if detail_shape == "missing":
        detail = {}
    body = {"outcome": "not_active", "reason": "interaction_pending", "attemptId": aid,
            "harnessId": "harness-" + "H" * 20_000, "message": "refusal-" + "D" * 30_000}
    ctx = _ctx(tmp_path)
    _own_run(rid)
    stub = _install(monkeypatch, _QuestionStub(result=body, detail=detail))
    original_write = Path.write_bytes
    failed_writes = []

    def write_bytes(path, data):
        if source_fails and ".interactions.json." in path.name:
            failed_writes.append(path)
            raise OSError("retained source-write failure")
        return original_write(path, data)

    monkeypatch.setattr(Path, "write_bytes", write_bytes)
    memo = {"other": frozenset({"old"})}
    monkeypatch.setattr(interactions, "_REPORTED_INTERACTIONS", memo)
    entry = next(entry for entry in get_tools() if entry.name == "delegate_message")
    sentinel = object()
    token = _install_tool_result_sidecar(ctx, sentinel)
    try:
        text = entry.handler(ctx, rid, "steer", message_id=mid)
        published = _published_tool_result(ctx, sentinel)
    finally:
        _restore_tool_result_sidecar(token)
    assert published is not sentinel and published.text == text
    assert published.code == SUBSTRATE_REFUSAL_CODE
    assert len(text) <= tool_result_limit("delegate_message")
    assert _truncate_tool_result(text, "delegate_message", {}) == text
    payload = json.loads(text)
    assert payload["status"] == "not_active" and payload["reason"] == "interaction_pending"
    assert payload["accepted"] is False and payload["ok"] is False
    assert payload["host_code"] == SUBSTRATE_REFUSAL_CODE
    assert {"run_id", "message_id", "attempt_id", "harness_id", "detail"} <= set(payload["fields_omitted"])
    assert payload["fields_omitted_count"] == len(payload["fields_omitted"])
    assert "continuation" not in payload and "state" not in payload
    assert memo == {"other": frozenset({"old"})}
    delivery = payload["interactions_delivery"]
    if source_fails:
        assert failed_writes and all(not path.exists() for path in failed_writes)
        assert delivery["artifact"] is None and delivery["read_next"] is None
        assert "UNRECOVERABLE" in delivery["note"]
        assert "unavailable here" in payload["fields_omitted_note"]
        assert "read the staged artifact" not in payload.get("interactions_note", "")
    else:
        artifact = delivery["artifact"]
        raw = Path(artifact["abs_path"]).read_bytes()
        assert artifact["bytes"] == len(raw) and artifact["sha256"] == hashlib.sha256(raw).hexdigest()
        full = json.loads(raw)
        assert (full["run_id"], full["message_id"], full["attempt_id"]) == (rid, mid, aid)
        assert full["detail"] == body["message"] and full["harness_id"] == body["harnessId"]
        if detail_shape == "questions":
            assert full["pending_interactions"][0]["interaction_id"] == question["interactionId"]
            assert full["pending_interactions"][0]["questions"][0]["question_id"] == question["questions"][0]["id"]
        elif detail_shape == "missing":
            assert "pending_interactions" not in full
    assert len(stub.posts) == len(stub.detail_reads) == stub.closed == 1 and stub.answers == []
    assert [r["outcome"] for r in _receipts(tmp_path)] == ["not_active"]


def test_registered_message_handler_observes_current_questions_over_real_gateway(
        tmp_path, monkeypatch):
    from ouroboros.gateways import claudexor as cx
    from ouroboros.tools.delegate import get_tools
    from tests.system_e2e.interfaces import FAKE_ASK_MARKER, FakeClaudexorDaemon

    with FakeClaudexorDaemon(runs_dir=tmp_path / "runs") as daemon:
        daemon.install(tmp_path / "cx")
        endpoint = cx.discover_daemon_at(tmp_path / "cx")
        monkeypatch.setattr(cx, "discover_daemon", lambda home=None: endpoint)
        with cx.ClaudexorGateway(endpoint) as setup:
            setup.handshake()
            rid = str(setup.start_run({
                "prompt": FAKE_ASK_MARKER, "instructions": "i", "authPreference": "subscription",
                "mode": "ask", "scope": {"kind": "project", "root": str(tmp_path)},
                "harnesses": [daemon.harness_id], "primaryHarness": daemon.harness_id,
                "access": "readonly", "maxSeconds": 60,
            }, idempotency_key="question-observation")["runId"])
        ctx = _ctx(tmp_path)
        _own_run(rid, route_id=daemon.harness_id)
        entry = next(entry for entry in get_tools() if entry.name == "delegate_message")
        original_send = cx.ClaudexorGateway.send_run_message
        next_question = ["int-after-refusal"]

        def send_then_change_question(gateway, *args, **kwargs):
            receipt = original_send(gateway, *args, **kwargs)
            # The read is later than the POST: these are deliberately different facts.
            iid = next_question[0]
            daemon.runs[rid]["pending"] = [{
                **_message_question(iid), "runId": rid,
                "attemptId": "a-observed-after-post", "harnessId": daemon.harness_id,
            }] if iid else []
            return receipt

        monkeypatch.setattr(cx.ClaudexorGateway, "send_run_message", send_then_change_question)
        for iid in ("int-after-refusal", "int-new-current", None):
            next_question[0] = iid
            payload = json.loads(entry.handler(ctx, rid, "steer", message_id="stored-refusal"))
            assert (payload["status"], payload["reason"], payload["attempt_id"]) == (
                "not_active", "interaction_pending", "a01")
            assert payload["accepted"] is False and payload["ok"] is False
            assert payload["message_id"] == "stored-refusal" and "continuation" not in payload
            if iid:
                row = payload["pending_interactions"][0]
                assert row["interaction_id"] == iid and row["attempt_id"] == "a-observed-after-post"
                assert row["questions"][0]["question_id"] == "q-port"
                assert row["questions"][0]["options"][0]["label"] == "8080"
            else:
                assert payload["pending_interactions"] == []
        assert len(daemon.calls("POST", f"/v2/runs/{rid}/messages")) == 3
        assert len(daemon.calls("GET", f"/v2/runs/{rid}")) == 3
        assert not daemon.runs[rid].get("messages")
        assert not [call for call in daemon.calls("POST")
                    if call.get("method") == "POST" and "/answer" in call.get("path", "")]


def test_message_question_observation_does_not_consume_the_next_real_wait(tmp_path, monkeypatch):
    from ouroboros import delegate_interactions as interactions
    from ouroboros.tools.delegate import _delegate_wait

    ctx = _ctx(tmp_path)
    _own_run()
    memo = {}
    monkeypatch.setattr(interactions, "_REPORTED_INTERACTIONS", memo)
    stub = _install(monkeypatch, _QuestionStub(detail={
        "lastSeq": 5, "pendingInteractions": [_message_question()],
        "summary": {"state": "running", "effectiveAccess": "readonly", "waitingOnUser": True},
    }))
    refused = json.loads(interactions._delegate_message(ctx, "run-1", "steer").text)
    assert refused["interaction_observation"]["status"] == "observed" and not memo
    waited = json.loads(_delegate_wait(ctx, "run-1", wait_sec=1, since_seq=5))
    assert waited["status"] == "waiting_on_user"
    assert waited["pending_interactions"][0]["interaction_id"] == "int-current"
    assert waited["pending_interactions"][0]["questions"][0]["question_id"] == "q-port"
    assert memo == {"run-1": frozenset({"int-current"})}
    assert len(stub.posts) == 1 and len(stub.detail_reads) == 2 and not stub.answers


def test_pending_detail_wall_timeout_keeps_known_refusal(tmp_path, monkeypatch):
    import threading
    from ouroboros import delegate_interactions as interactions

    release, finished = threading.Event(), threading.Event()

    class BlockingDetail(_QuestionStub):
        def get_run(self, rid, *, timeout_sec=None):
            self.detail_reads.append((rid, timeout_sec))
            try:
                release.wait(5)
                return self.detail
            finally:
                finished.set()

    ctx = _ctx(tmp_path)
    _own_run()
    stub = _install(monkeypatch, BlockingDetail())
    monkeypatch.setattr(interactions, "_MESSAGE_DEADLINE_SEC", 0.1)
    try:
        payload = json.loads(interactions._delegate_message(ctx, "run-1", "steer").text)
        assert not release.is_set()
        assert payload["status"] == "not_active"
        assert payload["reason"] == "interaction_pending"
        assert payload["ok"] is False
        assert "poll_wall_timeout" in payload["interaction_observation"]["detail"]
        assert "pending_interactions" not in payload
        assert len(stub.posts) == 1 and stub.answers == []
    finally:
        release.set()
        assert finished.wait(5)
