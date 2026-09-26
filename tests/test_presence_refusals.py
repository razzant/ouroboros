"""Recovery disposition is a producer fact, not a guess from an HTTP status."""
import asyncio
import json
import queue
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest

import ouroboros.loop_llm_call as call_mod
from ouroboros import loop, usage_accounting as ua
from ouroboros.gateway import host_service
from ouroboros.loop_llm_call import TRANSPORT_DEATHS_KEY
from ouroboros.presence_admission import PresenceAdmissionError
from ouroboros.presence_runner import PresenceTurnError
from ouroboros.presence_runner import PresenceTurnGate, presence_turn_task_id, presence_retry_proof, run_presence_turn
from ouroboros.presence_runner import _notify_unresolved_turn, presence_result_from_stored, presence_unknown_outcome
from ouroboros.task_results import load_task_result, task_result_path, write_task_result
from ouroboros.tools.registry import ToolRegistry
from tests.test_host_service_responsiveness import _answer, _event, _presence_app, _request, _turn
from tests.test_presence_delivery import _payload
from tests.test_transport_death_retry import _LedgerLLM, _events, _ledger, _no_chain


def test_auth_block_and_origin_rejection_share_status_not_disposition(tmp_path):
    app, binding, _ctx = _presence_app(tmp_path, _answer)
    request = _request(app, {"binding_id": binding, "event": _event("event")})
    request.headers = {"x-skill-token": "invalid"}
    auth = asyncio.run(host_service._api_presence_turn(request))
    event = _event("event")
    event["provider"] = "wrong-provider"
    origin = asyncio.run(host_service._api_presence_turn(
        _request(app, {"binding_id": binding, "event": event})))
    assert auth.status_code == origin.status_code == 403
    assert json.loads(auth.body)["disposition"] == "blocked"
    assert json.loads(auth.body)["code"] == "presence_auth_blocked"
    assert json.loads(origin.body)["disposition"] == "rejected"
    assert json.loads(origin.body)["code"] == "presence_origin_mismatch"


@pytest.mark.parametrize("code,disposition", [
    ("chat_log_unwritable", "retry"),
    ("presence_result_missing", "retry"),
    ("presence_attachment_admission_rejected", "rejected"),
    ("presence_admission_conversation_mismatch", "rejected"),
])
def test_turn_refusals_keep_legacy_facts_and_semantic_disposition(tmp_path, code, disposition):
    manifest = [{"status": "rejected", "label": "input"}]

    def refuse(**kwargs):
        raise PresenceTurnError(code, "staged_files", attachment_manifest=manifest)

    app, binding, _ctx = _presence_app(tmp_path, refuse)
    response = asyncio.run(_turn(app, binding, "event"))
    assert response.status_code == 409
    body = json.loads(response.body)
    assert body == {"ok": False, "error": f"{code}: staged_files", "code": code,
                    "field": "staged_files", "attachment_manifest": manifest,
                    "disposition": disposition}


@pytest.mark.parametrize("code,disposition", [
    ("presence_behavior_skill_disabled", "blocked"),
    ("presence_required_capability_unavailable", "blocked"),
    ("presence_bindings_unreadable", "retry"),
    ("presence_binding_wrong_transport", "rejected"),
])
def test_admission_dispositions(tmp_path, monkeypatch, code, disposition):
    app, binding, _ctx = _presence_app(tmp_path, _answer)

    def refuse(*args):
        raise PresenceAdmissionError(code, "binding_id")

    monkeypatch.setattr(host_service, "_admit_presence", refuse)
    response = asyncio.run(_turn(app, binding, "event"))
    body = json.loads(response.body)
    assert response.status_code == 409
    assert body["code"] == code and body["field"] == "binding_id"
    assert body["disposition"] == disposition


def test_receipt_write_retry_and_conflict_do_not_share_disposition(tmp_path, monkeypatch):
    app, _binding, ctx = _presence_app(tmp_path, _answer)
    payload = _payload()
    first = asyncio.run(host_service._api_presence_delivery(_request(app, payload)))
    assert first.status_code == 200
    changed = dict(payload, text="different facts")
    conflict = asyncio.run(host_service._api_presence_delivery(_request(app, changed)))
    assert conflict.status_code == 409
    assert json.loads(conflict.body)["disposition"] == "rejected"

    def unwritable(*args):
        raise OSError("unreadable archive")

    monkeypatch.setattr(ctx.presence_deliveries, "record", unwritable)
    unavailable = asyncio.run(host_service._api_presence_delivery(_request(app, payload)))
    assert unavailable.status_code == 503
    assert json.loads(unavailable.body)["disposition"] == "retry"
    assert json.loads(unavailable.body)["code"] == "presence_receipt_unwritable"


def test_missing_work_is_rejected_without_losing_original_error(tmp_path):
    app, binding, _ctx = _presence_app(tmp_path, _answer)
    request = _request(app)
    request.path_params = {"work_ref": "missing-work"}
    request.query_params = {"binding_id": binding}
    response = asyncio.run(host_service._api_presence_work(request))
    assert response.status_code == 404
    assert json.loads(response.body) == {
        "ok": False, "error": "presence work reference not found",
        "code": "presence_work_not_found", "disposition": "rejected",
    }


@pytest.mark.parametrize("status,code,disposition", [
    ("running", "presence_attempt_outcome_unknown", "retry"),
    ("interrupted", "presence_attempt_outcome_unknown", "retry"),
    ("cancelled", "presence_turn_cancelled", "blocked"),
])
def test_lost_or_cancelled_turn_never_regenerates_and_releases_host_capacity(tmp_path, status, code, disposition):
    attempted = []

    def runner(**kwargs):
        return run_presence_turn(
            repo_dir=tmp_path, drive_root=tmp_path, gate=PresenceTurnGate(1),
            agent_factory=lambda **_kw: attempted.append("agent") or None, **kwargs,
        )

    app, binding, ctx = _presence_app(tmp_path, runner)
    turn_id = presence_turn_task_id(binding, "event")
    write_task_result(tmp_path, turn_id, status, result="prior attempt unconfirmed",
                      metadata={"source": "presence", "presence": {"binding_id": binding}})
    original = load_task_result(tmp_path, turn_id)
    for _ in range(2):
        response = asyncio.run(_turn(app, binding, "event"))
        body = json.loads(response.body)
        assert response.status_code == 409
        assert body["ok"] is False and body["code"] == code
        assert body["disposition"] == disposition and body["turn_ref"] == turn_id
        assert not body.get("text") and attempted == []
        assert ctx.presence_turns.live() == [] and not any(ctx._inflight.values())
        assert load_task_result(tmp_path, turn_id) == original


def test_unresolved_turn_asks_owner_once_and_never_addresses_correspondent(tmp_path, monkeypatch):
    from supervisor import message_bus

    task_id = "presence-unresolved"
    write_task_result(tmp_path, task_id, "running", metadata={"source": "presence"})
    deliveries = []
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "try_get_bridge", lambda: object())
    monkeypatch.setattr(message_bus, "send_with_budget", lambda *args, **kwargs: deliveries.append((args, kwargs)))
    _notify_unresolved_turn(tmp_path, task_id)
    _notify_unresolved_turn(tmp_path, task_id)
    assert len(deliveries) == 1
    args, kwargs = deliveries[0]
    assert args[0] == 1 and task_id in args[1]
    assert kwargs == {"role": "system", "system_type": "presence_recovery_required", "require_write": True,
                      "ensure_record_boundary": True}
    assert load_task_result(tmp_path, task_id)["presence_recovery_owner_notified"]


def test_concurrent_replay_sends_one_owner_recovery_notice(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from supervisor import message_bus

    task_id = "presence-concurrent-refusal"
    write_task_result(tmp_path, task_id, "failed", reason_code="resource_refusal_no_resend",
                      metadata={"source": "presence"})
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "try_get_bridge", lambda: object())
    entered, release, second_started = Event(), Event(), Event()
    deliveries = []

    def record(*args, **kwargs):
        deliveries.append((args, kwargs))
        if len(deliveries) == 1:
            entered.set()
            assert release.wait(5)

    monkeypatch.setattr(message_bus, "send_with_budget", record)

    def second_reader():
        second_started.set()
        _notify_unresolved_turn(tmp_path, task_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(_notify_unresolved_turn, tmp_path, task_id)
        assert entered.wait(5)
        second = pool.submit(second_reader)
        assert second_started.wait(5)
        try:
            # The second replay cannot race through read/check/write while the
            # first still owes its durable stamp, even though no turn is admitted.
            assert not second.done()
        finally:
            release.set()
        first.result(timeout=5)
        second.result(timeout=5)
    assert len(deliveries) == 1
    assert load_task_result(tmp_path, task_id)["presence_recovery_owner_notified"]


def test_unrecorded_owner_question_never_gets_a_notified_stamp(tmp_path, monkeypatch):
    from supervisor import message_bus

    task_id = "presence-unrecorded"
    write_task_result(tmp_path, task_id, "running", metadata={"source": "presence"})
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "try_get_bridge", lambda: object())
    monkeypatch.setattr(message_bus, "get_bridge", lambda: pytest.fail("unrecorded message was sent"))
    monkeypatch.setattr(message_bus, "log_chat", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk")))
    _notify_unresolved_turn(tmp_path, task_id)
    assert not load_task_result(tmp_path, task_id).get("presence_recovery_owner_notified")
    assert not (tmp_path / "logs" / "chat.jsonl").exists()


@pytest.mark.parametrize("quarantined", [False, True])
def test_unreadable_or_quarantined_result_never_becomes_a_new_turn(tmp_path, quarantined):
    invoked = []

    def runner(**kwargs):
        return run_presence_turn(
            repo_dir=tmp_path, drive_root=tmp_path, gate=PresenceTurnGate(1),
            agent_factory=lambda **_kw: invoked.append("agent") or None, **kwargs,
        )

    app, binding, ctx = _presence_app(tmp_path, runner)
    task_id = presence_turn_task_id(binding, "event")
    result_path = task_result_path(tmp_path, task_id)
    if quarantined:
        result_path = result_path.parent / "quarantine" / result_path.name
        result_path.parent.mkdir(parents=True)
    result_path.write_text("{broken", encoding="utf-8")
    for _ in range(2):
        response = asyncio.run(_turn(app, binding, "event"))
        body = json.loads(response.body)
        assert response.status_code == 409 and body["code"] == "presence_result_unreadable"
        assert body["disposition"] == "retry" and body["turn_ref"] == task_id
        assert invoked == [] and ctx.presence_turns.live() == [] and not any(ctx._inflight.values())
        assert result_path.read_text(encoding="utf-8") == "{broken"


def test_failed_durable_start_never_calls_agent_then_same_event_may_retry(tmp_path, monkeypatch):
    from ouroboros import task_results
    from ouroboros.presence_runner import PresenceTurnEvent, presence_event_identity
    from ouroboros.presence_bindings import conversation_key

    invoked = []

    class Agent:
        def handle_task(self, task):
            invoked.append(task["id"])
            task_results.write_task_result(tmp_path, task["id"], "completed",
                                           metadata=task["metadata"], terminal_origin="model_final",
                                           result="real reply")
            return [{"type": "presence_result", "outcome": "message", "text": "real reply"}]

    def runner(**kwargs):
        return run_presence_turn(repo_dir=tmp_path, drive_root=tmp_path,
                                 agent_factory=lambda **_kw: Agent(),
                                 gate=PresenceTurnGate(1), **kwargs)

    app, binding, _ctx = _presence_app(tmp_path, runner)
    original = task_results.write_task_result

    def refuse_start(*args, **kwargs):
        if kwargs.get("create_only"):
            raise OSError("disk refused durable start")
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(task_results, "write_task_result", refuse_start)
        failed = asyncio.run(_turn(app, binding, "event"))
    assert failed.status_code == 409
    assert json.loads(failed.body)["code"] == "presence_start_unwritable"
    assert invoked == [] and load_task_result(tmp_path, presence_turn_task_id(binding, "event")) is None
    succeeded = asyncio.run(_turn(app, binding, "event"))
    assert succeeded.status_code == 200 and invoked == [presence_turn_task_id(binding, "event")]
    assert json.loads(succeeded.body)["text"] == "real reply"
    stored = load_task_result(tmp_path, invoked[0])
    event = _event("event")
    event["conversation_key"] = conversation_key(event["provider"], event["account_id"],
                                                  event["conversation_id"], event["thread_id"])
    assert stored["metadata"]["presence_event_identity"] == presence_event_identity(binding, PresenceTurnEvent(**event))


def test_resource_refusal_never_returns_a_prepared_reply_as_completed(tmp_path):
    invoked = []

    class Agent:
        def handle_task(self, task):
            invoked.append(task["id"])
            write_task_result(tmp_path, task["id"], "failed", metadata=task["metadata"],
                              reason_code="resource_refusal_no_resend", result="draft external speech")
            return [{"type": "presence_result", "outcome": "message", "text": "draft external speech"}]

    app, binding, ctx = _presence_app(tmp_path, lambda **kwargs: run_presence_turn(
        repo_dir=tmp_path, drive_root=tmp_path, agent_factory=lambda **_kw: Agent(),
        gate=PresenceTurnGate(1), **kwargs))
    for _ in range(2):
        refused = asyncio.run(_turn(app, binding, "event"))
        body = json.loads(refused.body)
        assert refused.status_code == 409 and body["code"] == "presence_resources_unavailable"
        assert body["disposition"] == "retry" and not body.get("text")
        assert not ctx.presence_turns.live() and not any(ctx._inflight.values())
    assert len(invoked) == 1  # retry cannot regenerate work behind an already terminal row


def test_resource_refusal_preserves_scheduled_child_work_ref_on_first_call_and_replay(tmp_path):
    child_id = "scheduled-presence-child"
    invoked = []

    class Agent:
        def handle_task(self, task):
            invoked.append(task["id"])
            metadata = {**task["metadata"], "presence_work_ref": child_id}
            write_task_result(tmp_path, task["id"], "failed", metadata=metadata,
                              reason_code="resource_refusal_no_resend", result="private diagnostic")
            return [{"type": "presence_result", "outcome": "deferred", "text": "", "work_ref": child_id}]

    app, binding, _ctx = _presence_app(tmp_path, lambda **kwargs: run_presence_turn(
        repo_dir=tmp_path, drive_root=tmp_path, agent_factory=lambda **_kw: Agent(),
        gate=PresenceTurnGate(1), **kwargs))
    for _ in range(2):
        response = asyncio.run(_turn(app, binding, "event"))
        body = json.loads(response.body)
        assert response.status_code == 409 and body["code"] == "presence_resources_unavailable"
        assert body["turn_ref"] == presence_turn_task_id(binding, "event")
        assert body["work_ref"] == child_id and not body.get("text")
    assert invoked == [presence_turn_task_id(binding, "event")]


def test_retained_resource_refusal_outranks_later_context_overflow(tmp_path):
    from ouroboros import loop

    usage = {"_last_llm_error_kind": "context_overflow", "execution_status": "infra_failed",
             "resource_refusal": {"reason": "quota", "temporary": True,
                                  "reset_at": "2099-01-01T00:00:00Z"}}
    ctx = loop._RoundLimitContext(
        messages=[{"role": "user", "content": "go"}], llm=SimpleNamespace(), active_model="test-model",
        active_effort="low", max_retries=1, drive_logs=tmp_path, task_id="presence-quota",
        round_idx=1, event_queue=None, accumulated_usage=usage, task_type="presence",
        active_use_local=False, max_rounds=200, drive_root=tmp_path)
    _text, terminal, trace = loop._handle_provider_unavailable(ctx, error_kind="context_overflow")
    assert terminal["reason_code"] == "resource_refusal_no_resend"
    assert terminal["execution_status"] == "infra_failed"
    assert trace["forced_finalization"]["source"] == "resource_refusal_no_resend"


def test_only_positive_first_round_not_started_proves_retry_eligibility():
    task = {"id": "presence-a", "metadata": {"presence_event_identity": "event-identity"}}
    usage = {"_presence_pre_dispatch_only": True, "resource_refusal": {
        "temporary": True, "reset_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()}}
    ctx = SimpleNamespace(_swarm_handoff_attempt=None)
    assert presence_retry_proof(task, usage, {"tool_calls": []}, ctx)["task_id"] == "presence-a"
    for changed_usage, trace, changed_ctx in (
        ({**usage, "_presence_pre_dispatch_only": False}, {"tool_calls": []}, ctx),
        ({**usage, "rounds": 1}, {"tool_calls": []}, ctx),
        (usage, {"tool_calls": [{"name": "send"}]}, ctx),
        (usage, {"tool_calls": []}, SimpleNamespace(_swarm_handoff_attempt={"status": "scheduled"})),
        ({**usage, "resource_refusal": {"temporary": True}}, {"tool_calls": []}, ctx),
    ):
        assert not presence_retry_proof(task, changed_usage, trace, changed_ctx)


def test_terminal_pipeline_stamps_no_effect_certificate_only_on_typed_proof(tmp_path):
    from ouroboros import agent_task_pipeline
    from tests.test_presence_runner import _admission as admitted, _event as incoming
    from ouroboros.presence_runner import _build_task

    event = incoming()
    admission = admitted()
    task = _build_task(admission, event, drive_root=tmp_path, staged_files=())
    reset = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    usage = {"execution_status": "infra_failed", "reason_code": "resource_refusal_no_resend",
             "resource_refusal": {"temporary": True, "reset_at": reset},
             "_presence_pre_dispatch_only": True}
    ctx = SimpleNamespace(_swarm_handoff_attempt=None, task_contract=task["task_contract"],
                          task_metadata=task["metadata"])
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    agent_task_pipeline.emit_task_results(env, None, None, [], task, "refused", usage,
                                          {"tool_calls": []}, 0.0, tmp_path / "logs", ctx=ctx)
    row = load_task_result(tmp_path, task["id"])
    assert row["reason_code"] == "resource_refusal_no_resend"
    assert row["metadata"]["presence_retry_proof"]["kind"] == "first_round_engine_not_started"


def test_producer_records_only_terminal_engine_not_started_as_no_effect(tmp_path):
    from ouroboros.llm_claudexor import ClaudexorModelNotDispatched
    from ouroboros.loop_llm_call import _LlmErrorContext, _record_llm_call_error
    from ouroboros.usage_accounting import PhysicalAttemptCapture
    from dataclasses import replace

    usage = {}
    context = _LlmErrorContext(task_id="presence-test", task_type="presence", execution_id="e",
                               round_id="r", llm_call_id="c", round_idx=1, attempt=0,
                               model="claudexor::codex=model", request_ref=None,
                               drive_logs=tmp_path / "logs", event_queue=None,
                               accumulated_usage=usage)
    refusal = ClaudexorModelNotDispatched({
        "code": "subscription_window_exhausted", "message": "quota",
        "context": {"resetsAt": "2099-01-01T00:00:00+00:00"}})
    refusal.physical_attempt_capture = PhysicalAttemptCapture(
        attempt_id="a", model="m", provider="claudexor", state="released",
        candidate_measurement_kind="canonical_json_v1")
    refusal.presence_all_operations_not_started = True
    _record_llm_call_error(refusal, context)
    assert usage["_presence_pre_dispatch_only"] is True
    # One questionable physical outcome poisons the entire attempt, even if a
    # subsequent route has a clean not-started refusal.
    refusal.physical_attempt_capture = replace(refusal.physical_attempt_capture, state="unresolved")
    _record_llm_call_error(refusal, context)
    refusal.physical_attempt_capture = PhysicalAttemptCapture(
        attempt_id="a", model="m", provider="claudexor", state="released",
        candidate_measurement_kind="canonical_json_v1")
    _record_llm_call_error(refusal, context)
    assert usage["_presence_pre_dispatch_only"] is False


def test_host_retries_only_attested_no_effect_after_reset_with_new_physical_identity(tmp_path, monkeypatch):
    from ouroboros import presence_runner

    invoked = []
    reset = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()

    class Agent:
        def handle_task(self, task):
            invoked.append(task["id"])
            if len(invoked) == 1:
                proof = presence_retry_proof(task, {
                    "_presence_pre_dispatch_only": True,
                    "resource_refusal": {"temporary": True, "reset_at": reset},
                }, {"tool_calls": []}, SimpleNamespace(_swarm_handoff_attempt=None))
                assert proof
                write_task_result(tmp_path, task["id"], "failed", metadata={
                    **task["metadata"], "presence_retry_proof": proof},
                    reason_code="resource_refusal_no_resend", result="private diagnostic")
                return [{"type": "presence_result", "outcome": "message", "text": "draft must not speak"}]
            write_task_result(tmp_path, task["id"], "completed", metadata=task["metadata"],
                              terminal_origin="model_final", result="Recovered reply")
            return [{"type": "presence_result", "outcome": "message", "text": "Recovered reply"}]

    app, binding, ctx = _presence_app(tmp_path, lambda **kwargs: run_presence_turn(
        repo_dir=tmp_path, drive_root=tmp_path, gate=PresenceTurnGate(1),
        agent_factory=lambda **_kw: Agent(), **kwargs))
    first_id = presence_turn_task_id(binding, "event")
    first = asyncio.run(_turn(app, binding, "event"))
    assert first.status_code == 409 and not json.loads(first.body).get("text")
    waiting = asyncio.run(_turn(app, binding, "event"))
    assert waiting.status_code == 409 and invoked == [first_id]

    actual_datetime = datetime

    class LaterDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return actual_datetime.now(tz) + timedelta(minutes=10)

    monkeypatch.setattr(presence_runner, "datetime", LaterDatetime)
    recovered = asyncio.run(_turn(app, binding, "event"))
    assert recovered.status_code == 200 and json.loads(recovered.body)["text"] == "Recovered reply"
    second_id = json.loads(recovered.body)["turn_ref"]
    assert second_id != first_id and invoked == [first_id, second_id]
    assert load_task_result(tmp_path, first_id)["presence_retry_next"] == second_id
    assert load_task_result(tmp_path, first_id)["status"] == "failed"
    assert load_task_result(tmp_path, second_id)["status"] == "completed"
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.presence_runner import _previous_turn_path

    key = conversation_key("telegram", "bot-1", "room-1", "topic-1")
    # A crash after the successor terminal but before its pointer write must
    # repair the pointer to the PHYSICAL successor, not the old failed ID.
    pointer = _previous_turn_path(tmp_path, key)
    assert pointer.exists()
    pointer.unlink()
    assert asyncio.run(_turn(app, binding, "event")).body == recovered.body
    assert json.loads(pointer.read_text(encoding="utf-8"))["task_id"] == second_id
    assert invoked == [first_id, second_id] and ctx.presence_turns.live() == []
    rows = [json.loads(line) for line in (tmp_path / "logs" / "chat.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len([row for row in rows if row.get("direction") == "in" and
                row.get("client_message_id") == "event"]) == 1


def test_late_source_bound_terminal_outweighs_old_quarantine(tmp_path):
    from ouroboros.presence_runner import PresenceTurnEvent, presence_event_identity
    from ouroboros.presence_bindings import conversation_key

    app, binding, _ctx = _presence_app(tmp_path, lambda **kw: run_presence_turn(
        repo_dir=tmp_path, drive_root=tmp_path, gate=PresenceTurnGate(1),
        agent_factory=lambda **_unused: pytest.fail("no model re-execution"), **kw))
    task_id = presence_turn_task_id(binding, "event")
    bad = task_result_path(tmp_path, task_id).parent / "quarantine" / f"{task_id}.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{old invalid record", encoding="utf-8")
    event = _event("event")
    event["conversation_key"] = conversation_key(
        event["provider"], event["account_id"], event["conversation_id"], event["thread_id"])
    write_task_result(tmp_path, task_id, "completed", terminal_origin="model_final", result="late answer",
                      metadata={"source": "presence", "presence": {"binding_id": binding},
                                "presence_event_identity": presence_event_identity(binding, PresenceTurnEvent(**event))})
    response = asyncio.run(_turn(app, binding, "event"))
    assert response.status_code == 200 and json.loads(response.body)["text"] == "late answer"
    assert bad.read_text(encoding="utf-8") == "{old invalid record"


def test_required_jsonl_boundary_read_error_refuses_append(tmp_path, monkeypatch):
    from pathlib import Path
    from ouroboros.utils import append_jsonl

    path = tmp_path / "logs" / "chat.jsonl"
    path.parent.mkdir()
    original_bytes = b'{"torn": 1}'
    path.write_bytes(original_bytes)
    opener = Path.open

    def fail_probe(self, mode="r", *args, **kwargs):
        if self == path and mode == "rb":
            raise OSError("tail is unreadable")
        return opener(self, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", fail_probe)
        assert append_jsonl(path, {"new": 2}, ensure_record_boundary=True, require_lock=True) is False
    assert path.read_bytes() == original_bytes
    assert append_jsonl(path, {"new": 2}, ensure_record_boundary=True, require_lock=True) is True
    assert path.read_bytes().splitlines()[1] == b'{"new": 2}'


def test_ambiguous_start_write_never_regenerates_after_error(tmp_path, monkeypatch):
    from ouroboros import task_results

    invoked = []

    class Agent:
        def handle_task(self, task):
            invoked.append(task["id"])
            return []

    app, binding, ctx = _presence_app(tmp_path, lambda **kwargs: run_presence_turn(
        repo_dir=tmp_path, drive_root=tmp_path, gate=PresenceTurnGate(1),
        agent_factory=lambda **_kw: Agent(), **kwargs))
    original = task_results.write_task_result

    def wrote_then_lost_ack(*args, **kwargs):
        saved = original(*args, **kwargs)
        if kwargs.get("create_only"):
            raise OSError("write acknowledged nowhere")
        return saved

    with monkeypatch.context() as patch:
        patch.setattr(task_results, "write_task_result", wrote_then_lost_ack)
        first = asyncio.run(_turn(app, binding, "event"))
    assert first.status_code == 409 and json.loads(first.body)["code"] == "presence_start_unwritable"
    assert load_task_result(tmp_path, presence_turn_task_id(binding, "event"))["status"] == "running"
    again = asyncio.run(_turn(app, binding, "event"))
    assert again.status_code == 409 and json.loads(again.body)["code"] == "presence_attempt_outcome_unknown"
    assert invoked == [] and ctx.presence_turns.live() == []


_UNKNOWN_FENCE = pytest.mark.parametrize("usage", [
    # The forced rail's own fallback text under the fence, worded provider_unavailable.
    {"execution_status": "infra_failed", "reason_code": "provider_unavailable", "terminal_origin": "host_notice",
     "_last_llm_error_kind": "provider_outcome_unknown",
     "_pending_transport_outcome": {"operation_id": "op-1", "outcome": "unknown"}},
    # A round-one draft the rail salvaged as model_final behind the same fence is not the answer.
    {"execution_status": "infra_failed", "reason_code": "provider_unavailable", "terminal_origin": "model_final",
     "_best_effort_extracted": True, "_last_llm_error_kind": "provider_outcome_unknown"},
    # A round still holding a transport-death record forbids the resend whatever the later sticky kind.
    {"execution_status": "infra_failed", "reason_code": "provider_unavailable", "terminal_origin": "model_final",
     "_best_effort_extracted": True, "_last_llm_error_kind": "server_error",
     TRANSPORT_DEATHS_KEY: {"round_id": "round-1", "count": 1, "backoff_sec": 2.0}},
], ids=["host_fallback", "salvaged_draft", "death_record"])


@_UNKNOWN_FENCE
def test_unknown_outcome_fallback_terminal_never_acknowledges_the_event(tmp_path, monkeypatch, usage):
    """A quota-refused primary whose fallback died with an unknown outcome is not a silent answer.

    The forced rail words that terminal ``provider_unavailable`` (the unknown fence outranks the
    refusal source) and may salvage a round-one draft as ``model_final``, so the guard can key
    neither on the reason word nor on authorship. The real pipeline stamps the loop's own
    no-resend predicate on the row; that marker keeps the event with the transport on the first
    call and on replay, preserves already scheduled work, and asks the owner once — never a
    retry certificate.
    """
    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros.tools.control_routing import _finish_swarm_handoff

    child_id = "scheduled-after-quota"
    invoked, notices = [], []
    monkeypatch.setattr("ouroboros.presence_runner._write_unresolved_notice",
                        lambda _root, task_id: notices.append(task_id))

    class Agent:
        def handle_task(self, task):
            invoked.append(task["id"])
            task["_skip_post_task_synthesis"] = True
            agent_ctx = SimpleNamespace(task_contract=task["task_contract"], task_metadata=task["metadata"])
            _finish_swarm_handoff(agent_ctx, {"task_id": child_id}, "Admission receipt", status="scheduled")
            pending = []
            pipeline.emit_task_results(
                SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path), None, None, pending, task,
                "half-written draft", dict(usage), {"tool_calls": [], "reasoning_notes": []}, 0.0,
                tmp_path / "logs", ctx=agent_ctx,
            )
            return pending

    app, binding, ctx = _presence_app(tmp_path, lambda **kwargs: run_presence_turn(
        repo_dir=tmp_path, drive_root=tmp_path, agent_factory=lambda **_kw: Agent(),
        gate=PresenceTurnGate(1), **kwargs))
    task_id = presence_turn_task_id(binding, "event")
    for _ in range(2):
        response = asyncio.run(_turn(app, binding, "event"))
        body = json.loads(response.body)
        assert response.status_code == 409 and body["code"] == "presence_attempt_outcome_unknown"
        assert body["disposition"] == "retry" and not body.get("text")
        assert body["work_ref"] == child_id
        assert not ctx.presence_turns.live() and not any(ctx._inflight.values())
    assert invoked == [task_id] and notices == [task_id] * 2
    stored = load_task_result(tmp_path, task_id)
    assert stored["status"] == "failed" and stored["reason_code"] == "provider_unavailable"
    assert stored["outcome_axes"]["execution"]["status"] == "infra_failed"
    marker = stored["metadata"]["presence_unknown_outcome"]
    assert marker["source"] == "provider_outcome_unknown_no_resend"
    assert marker["error_kind"] == usage["_last_llm_error_kind"]
    assert marker["operation_id"] == (usage.get("_pending_transport_outcome") or {}).get("operation_id", "")
    assert stored["metadata"]["presence_work_ref"] == child_id and stored["metadata"]["presence_outcome"] == "deferred"
    assert "presence_retry_proof" not in stored["metadata"]  # unknown is not not_started
    # The deferred-work view of the same row hands out no draft either.
    view = presence_result_from_stored(stored, task_id)
    assert (view.outcome, view.text, view.work_ref) == ("silent", "", child_id)


def test_lost_terminal_write_after_start_barrier_never_acknowledges_the_event(tmp_path, monkeypatch):
    """The in-memory envelope is not authority: a RUNNING row after handle_task is an unproven effect.

    The pipeline logs and swallows a failed terminal write; the Host must then refuse with the
    scheduled work preserved from the execution's own handoff fact, and a retry must not regenerate.
    """
    child_id = "scheduled-before-terminal-loss"
    invoked = []

    class Agent:
        def handle_task(self, task):
            invoked.append(task["id"])  # the terminal write failed after the durable start
            return [{"type": "presence_result", "outcome": "message", "text": "answer", "work_ref": child_id}]

    app, binding, ctx = _presence_app(tmp_path, lambda **kwargs: run_presence_turn(
        repo_dir=tmp_path, drive_root=tmp_path, agent_factory=lambda **_kw: Agent(),
        gate=PresenceTurnGate(1), **kwargs))
    first = asyncio.run(_turn(app, binding, "event"))
    body = json.loads(first.body)
    assert first.status_code == 409 and body["code"] == "presence_attempt_outcome_unknown"
    assert body["disposition"] == "retry" and not body.get("text") and body["work_ref"] == child_id
    assert load_task_result(tmp_path, presence_turn_task_id(binding, "event"))["status"] == "running"
    again = asyncio.run(_turn(app, binding, "event"))
    assert again.status_code == 409 and json.loads(again.body)["code"] == "presence_attempt_outcome_unknown"
    assert invoked == [presence_turn_task_id(binding, "event")] and ctx.presence_turns.live() == []


def test_inline_turn_repeats_a_dispatched_transport_death_as_a_no_effect_attempt(tmp_path, monkeypatch):
    """The bounded same-round transport-death repeat is not the forbidden event retry.

    An inline Presence turn is a direct-chat task, so its PRIMARY round keeps the paid repeat
    rail (``loop_llm_call._TRANSPORT_DEATH_RETRIES``): a DISPATCHED request whose socket died
    with a typed transport death is sent once more in the same round as a NEW physical attempt
    with its own ledger row. That repeat is no-effect BY CONSTRUCTION in this narrow sense: no
    Host-executed tool and no correspondent action can stem from the failed attempt, because the
    host executes tools and the transport speaks only after a model response has landed, and
    none landed. Provider-owned retrieval (e.g. the server-side web search a provider runs
    before the response lands) may execute again on the repeat: read-only, non-mutating, priced.
    So the owner rule (the same event is retried only with positive proof of no prior effect and
    a new physical identity) is met, and the unknown-outcome refusal (empty 409 plus owner
    notice) applies when the round ends unresolved without a further permitted repeat: the
    deadline or a typed finalize control can refuse even the first repeat, and a repeat failing
    with any other class ends the round at once. Real Host endpoint, runner, loop, round
    dispatcher, terminal pipeline and attempt ledger: the socket is scripted, the backoff sleep is
    recorded instead of slept, and the fallback chain is a tripwire.
    """
    from ouroboros import agent_task_pipeline as pipeline

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "other/model")  # a configured chain the rail must never walk
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(loop, "_run_cross_model_fallback_chain", _no_chain)
    sleeps, notices, executed, sends = [], [], [], []
    monkeypatch.setattr(call_mod, "_sleep_within_deadline", lambda sec, _deadline, **_kw: (sleeps.append(sec), True)[1])
    monkeypatch.setattr("ouroboros.presence_runner._write_unresolved_notice",
                        lambda _root, task_id: notices.append(task_id))

    def spoken():
        path = tmp_path / "logs" / "chat.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
        return [row["text"] for row in rows if row.get("direction") == "out"]

    class ObservedLLM(_LedgerLLM):
        """Every send records the host state it left from: what ran and what was spoken before it."""

        def chat(self, **kwargs):
            sends.append({"messages": [dict(row) for row in kwargs["messages"]],
                          "tools_executed": list(executed), "spoken": spoken()})
            return super().chat(**kwargs)

    llm = ObservedLLM(tmp_path, lambda: httpx.ReadError("socket died after dispatch"), "ok")
    captured = {}

    class Agent:
        def handle_task(self, task):
            task["_skip_post_task_synthesis"] = True
            registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
            agent_ctx = registry._ctx
            agent_ctx.is_direct_chat = bool(task["_is_direct_chat"])  # the runner's own flag → ToolContext.is_direct_chat
            agent_ctx.task_metadata = dict(task["metadata"])
            agent_ctx.task_contract = dict(task["task_contract"])
            real_execute = registry.execute_result
            monkeypatch.setattr(registry, "execute_result",
                                lambda name, args: (executed.append(name), real_execute(name, args))[1])
            text, usage, trace = loop.run_llm_loop(
                [{"role": "user", "content": task["text"]}], registry, llm, tmp_path / "logs",
                lambda *_a, **_kw: None, queue.Queue(), task_type=task["type"], task_id=task["id"],
                drive_root=tmp_path)
            events = []
            pipeline.emit_task_results(SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path), None, None, events,
                                       task, text, usage, trace, 0.0, tmp_path / "logs", ctx=agent_ctx)
            captured.update(text=text, usage=usage, trace=trace)
            return events

    app, binding, host = _presence_app(tmp_path, lambda **kwargs: run_presence_turn(
        repo_dir=tmp_path, drive_root=tmp_path, gate=PresenceTurnGate(1),
        agent_factory=lambda **_kw: Agent(), **kwargs))
    task_id = presence_turn_task_id(binding, "event")
    response = asyncio.run(_turn(app, binding, "event"))
    body = json.loads(response.body)
    assert response.status_code == 200 and body["ok"] is True and body["turn_ref"] == task_id
    assert body["outcome"] == "message" and body["text"] == "done"

    # Exactly two physical attempts: the dead send stays unresolved at its bound, the repeat settles.
    assert llm.calls == 2
    by_attempt = {}
    for row in _ledger(tmp_path):
        by_attempt.setdefault(row["attempt_id"], []).append(row["state"])
    assert list(by_attempt.values()) == [["reserved", "dispatched", "unresolved"], ["reserved", "dispatched", "settled"]]
    assert ua.usage_projection(tmp_path)["unresolved_upper_bound_usd"] == 1.0
    # No effect between them: the repeat is the same logical request, and before either send
    # no tool had run and nothing had been spoken; the transport heard the answer once, after it landed.
    assert sends[1]["messages"] == sends[0]["messages"]
    assert [send["tools_executed"] for send in sends] == [[], []] and executed == []
    assert [send["spoken"] for send in sends] == [[], []] and spoken() == ["done"]
    assert sleeps == [4.0]  # the first-death backoff; no wait episode, no chain, no forced final
    api_errors = _events(tmp_path / "logs", "llm_api_error")
    assert [(row["error_kind"], row["retry_same_request"], row["transport_cause_type"]) for row in api_errors] == [
        ("provider_outcome_unknown", True, "ReadError")]
    assert api_errors[0]["physical_attempt_id"] == next(iter(by_attempt))
    assert _events(tmp_path / "logs", "llm_non_retryable_same_request") == []
    assert _events(tmp_path / "logs", "llm_retry_deadline_exhausted") == []
    usage, trace = captured["usage"], captured["trace"]
    assert captured["text"] == "done" and trace.get("forced_finalization") is None
    assert TRANSPORT_DEATHS_KEY not in usage and "_last_llm_error_kind" not in usage  # the usable response cleared both
    assert usage.get("reason_code") is None and usage.get("execution_status") != "infra_failed"
    assert presence_unknown_outcome(usage) == {}
    # The durable row the Host reads back is a completed turn, not an unknown outcome: no marker,
    # no owner notice, no retry certificate, and a replay serves the answer without a third send.
    stored = load_task_result(tmp_path, task_id)
    assert stored["status"] == "completed" and stored["reason_code"] == "final_message"
    assert "presence_unknown_outcome" not in stored["metadata"] and "presence_retry_proof" not in stored["metadata"]
    assert notices == [] and host.presence_turns.live() == []
    again = asyncio.run(_turn(app, binding, "event"))
    assert again.status_code == 200 and json.loads(again.body)["text"] == "done" and llm.calls == 2


_INFRA_OUTAGE = {"execution": {"status": "infra_failed", "reason_code": "provider_unavailable"}}


@pytest.mark.parametrize("row,metadata", [
    ({"reason_code": "resource_refusal_no_resend"}, {}),
    ({"reason_code": "provider_unavailable", "outcome_axes": _INFRA_OUTAGE},
     {"presence_unknown_outcome": {"source": "provider_outcome_unknown_no_resend",
                                   "error_kind": "provider_outcome_unknown", "operation_id": ""}}),
])
def test_deferred_work_view_never_delivers_a_salvaged_draft_of_a_refused_attempt(row, metadata):
    """A refused or unproven attempt has no reply: the forced rail may still stamp
    ``model_final`` over the round-one draft it salvaged before the refusal or behind the
    unknown fence, and the ``/presence/work`` projection must not hand that draft to the
    correspondent."""
    stored = {"status": "failed", "terminal_origin": "model_final", "result": "half-written draft",
              "metadata": {"presence_outcome": "message", "presence_work_ref": "child-1", **metadata}, **row}
    projected = presence_result_from_stored(stored, "work-1")
    assert (projected.outcome, projected.text, projected.work_ref) == ("silent", "", "child-1")
    # A model's own failed terminal (round limit, no infrastructure fault) still replays its answer.
    own = presence_result_from_stored({"status": "failed", "terminal_origin": "model_final", "result": "final words",
                                       "reason_code": "round_limit", "metadata": {"presence_outcome": "message"},
                                       "outcome_axes": {"execution": {"status": "best_effort"}}}, "work-2")
    assert (own.outcome, own.text) == ("message", "final words")
    # A CONFIRMED outage keeps the model's own salvaged partial and the admitted child (TZ2).
    confirmed = presence_result_from_stored({
        "status": "failed", "terminal_origin": "model_final", "result": "salvaged partial",
        "reason_code": "provider_unavailable", "outcome_axes": _INFRA_OUTAGE,
        "metadata": {"presence_outcome": "deferred", "presence_work_ref": "child-1"}}, "work-3")
    assert (confirmed.outcome, confirmed.text, confirmed.work_ref) == ("deferred", "salvaged partial", "child-1")


@pytest.mark.parametrize("row,expected", [
    ({"status": "failed", "reason_code": "resource_refusal_no_resend"}, "presence_resources_unavailable"),
    ({"status": "running"}, "presence_attempt_outcome_unknown"),
    ({"status": "interrupted", "terminal_origin": "model_final"}, "presence_attempt_outcome_unknown"),
    # The host reconciled a stale RUNNING row from its own already-failed axes: no terminal of its own.
    ({"status": "failed", "status_reconciled_from": "running", "reason_code": "provider_unavailable",
      "outcome_axes": _INFRA_OUTAGE}, "presence_attempt_outcome_unknown"),
    ({"status": "failed", "terminal_origin": "model_final", "reason_code": "provider_unavailable",
      "outcome_axes": _INFRA_OUTAGE, "metadata": {"presence_unknown_outcome": {
          "source": "provider_outcome_unknown_no_resend"}}}, "presence_attempt_outcome_unknown"),
    # Confirmed infrastructure terminals keep the deferred projection instead of a refusal.
    ({"status": "failed", "terminal_origin": "host_notice", "reason_code": "provider_unavailable",
      "outcome_axes": _INFRA_OUTAGE}, ""),
    ({"status": "failed", "terminal_origin": "model_final", "reason_code": "provider_unavailable",
      "outcome_axes": _INFRA_OUTAGE}, ""),
    ({"status": "failed", "terminal_origin": "host_notice", "reason_code": "llm_api_error",
      "outcome_axes": {"execution": {"status": "infra_failed", "reason_code": "llm_api_error",
                                     "failure": {"kind": "provider", "error_kind": "context_overflow"}}}}, ""),
    # A model final the axis reducer relabelled from a legacy diagnostic prefix is still the model's reply.
    ({"status": "failed", "terminal_origin": "model_final", "reason_code": "provider_failure",
      "result": "⚠️ Failed to get a response is what I would tell them.",
      "outcome_axes": {"execution": {"status": "infra_failed", "reason_code": "provider_failure"}}}, ""),
])
def test_terminal_refusal_keys_on_provenance_not_on_the_infra_failed_axis(row, expected):
    from ouroboros.presence_runner import _terminal_refusal

    assert _terminal_refusal(row) == expected
