"""External speech follows terminal authorship through execution and replay."""

import json
import queue
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from ouroboros import agent as agent_module, agent_task_pipeline as pipeline, loop
from ouroboros.gateway.host_service import create_host_service_app
from ouroboros.presence_authority import presence_ceiling_payload
from ouroboros.presence_runner import PresenceTurnError, _cached_result, build_presence_result_event
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.task_finalization import provider_terminal_body
from ouroboros.tools.registry import ToolRegistry
from tests.test_host_service_api import _seed_presence_behavior, _seed_token
from tests.test_presence_completion import _call
from tests.test_presence_failed_handoff import _failed_parent
from tests.test_presence_runner import _admission


def _run_loop(root, monkeypatch, responses, *, held=False, presence=None):
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "1")
    registry = ToolRegistry(repo_dir=root, drive_root=root)
    registry._ctx.is_direct_chat = True
    registry._ctx.task_metadata = {"inline_max_rounds": 1, **({"presence": presence} if presence else {})}
    registry._ctx.task_contract = {"capability_ceiling": presence_ceiling_payload(_admission().capability_ceiling)}
    registry.override_handler("chat_history", lambda *_a, **_kw: "Synthetic history")
    calls, held_origins = [], []
    responses = iter(responses)

    def respond(*_a, **_kw):
        calls.append(1)
        return next(responses), 0.0

    def review(**_kw):
        if held:
            held_origins.append(registry._ctx._accumulated_usage.get("terminal_origin"))
            registry._ctx._owner_directives = [{"content": "A new material constraint"}]
        return held

    monkeypatch.setattr(loop, "call_llm_with_retry", respond)
    monkeypatch.setattr(loop, "_run_task_acceptance_review_once", review)
    text, usage, trace = loop.run_llm_loop(
        [{"role": "user", "content": "Please help"}], registry,
        SimpleNamespace(default_model=lambda: "test-model"), root / "logs",
        lambda *_a, **_kw: None, queue.Queue(), task_id="presence-loop", drive_root=root,
    )
    task = {"id": "presence-loop", "type": "presence", "_presence_turn": True,
            "chat_id": 7, "text": "Please help", "_skip_post_task_synthesis": True}
    events = []
    pipeline.emit_task_results(SimpleNamespace(drive_root=root, repo_dir=root), None, None,
        events, task, text, usage, trace, 0.0, root / "logs", ctx=registry._ctx)
    result = next(row for row in events if row["type"] == "presence_result")
    return result, load_task_result(root, task["id"]), calls, held_origins


def _read_response():
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": "read", "type": "function", "function": {"name": "chat_history", "arguments": "{}"},
    }]}


@pytest.mark.parametrize("authored", [False, True])
def test_real_round_limit_delivers_only_the_current_authored_final(tmp_path, monkeypatch, authored):
    reply = "I found the record; the remaining check is incomplete." if authored else ""
    forced = json.dumps({"delivery_control": "replace", "full_answer": reply,
                         "presence_finish": {"outcome": "message", "message": reply}}) if authored else ""
    # A real turn's context carries its Presence metadata; that, with the ceiling, arms the forced call.
    presence = {"binding_id": "1" * 32, "event": {"conversation_key": "telegram:bot-1:room-1:topic-1"}}
    result, stored, calls, _held = _run_loop(tmp_path, monkeypatch, [_read_response(), {"content": forced}],
                                             presence=presence)
    assert len(calls) == 2 and stored["reason_code"] == "round_limit"
    assert stored["terminal_origin"] == ("model_final" if authored else "host_notice")
    assert result["outcome"] == ("message" if authored else "silent")
    assert result["text"] == reply
    assert _cached_result(tmp_path, result["task_id"]).text == reply
    if not authored:
        assert "MAX_ROUNDS" in stored["result"]
    else:
        assert stored["result"] == reply


def test_exact_host_diagnostic_is_deliverable_when_the_model_authors_it(tmp_path, monkeypatch):
    """Owner Q4 keeps host bytes internal by default without forbidding model speech.

    Host-authored terminal bytes never speak, and a forced final speaks only its typed
    declaration; an ordinary final is the model's own text in one channel, so a model
    that chooses to restate a diagnostic there is heard. The host adds no text filter.
    """
    _result, host, _calls, _held = _run_loop(tmp_path / "host", monkeypatch, [_read_response(), {"content": ""}])
    result, authored, calls, _held = _run_loop(tmp_path / "author", monkeypatch, [{"content": host["result"]}])
    assert len(calls) == 1  # ordinary implicit final, no presence_finish required
    assert authored["terminal_origin"] == "model_final"
    assert result["outcome"] == "message" and result["text"] == host["result"]


def test_held_model_candidate_cannot_author_a_later_host_fallback(tmp_path, monkeypatch):
    result, stored, _calls, origins = _run_loop(tmp_path, monkeypatch,
        [{"content": "Not yet a final answer"}, {"content": ""}], held=True)
    assert origins == [None]
    assert stored["terminal_origin"] == "host_notice"
    assert result["outcome"] == "silent" and result["text"] == ""


@pytest.mark.parametrize("origin", ["host_notice", "host_salvage", ""])
@pytest.mark.parametrize("admission", ["scheduled", "unconfirmed", "rejected"])
def test_answerless_failed_parent_keeps_only_admitted_child_custody(tmp_path, origin, admission):
    result, stored, raw = _failed_parent(tmp_path, admission=admission, accepted=True, usage={
        "execution_status": "infra_failed", "reason_code": "provider_unavailable", "terminal_origin": origin,
    })
    assert result.outcome == ("deferred" if admission == "scheduled" else "silent")
    assert result.work_ref == ("managed-work" if admission == "scheduled" else "")
    assert result.text == "" and stored["result"] == raw
    assert stored["status"] == "failed"


@pytest.mark.parametrize("origin", ["host_notice", "host_salvage", "model_final"])
@pytest.mark.parametrize("admission", ["scheduled", "rejected"])
def test_unresolved_attempt_is_refused_whatever_authored_its_terminal(tmp_path, origin, admission):
    """Under the unknown-outcome fence authorship decides nothing: the rail's notice, its salvage
    and a round-one draft it stamped ``model_final`` are all unanswered events, refused back to
    the transport on the first call and on replay with only an admitted child's custody."""
    refused, stored, raw = _failed_parent(tmp_path, admission=admission, accepted=True, usage={
        "execution_status": "infra_failed", "reason_code": "provider_unavailable", "terminal_origin": origin,
        "_best_effort_extracted": origin == "model_final", "_last_llm_error_kind": "provider_outcome_unknown",
    }, refusal="presence_attempt_outcome_unknown")
    assert refused.work_ref == ("managed-work" if admission == "scheduled" else "")
    assert stored["status"] == "failed" and stored["terminal_origin"] == origin and stored["result"] == raw
    assert stored["metadata"]["presence_unknown_outcome"]["error_kind"] == "provider_outcome_unknown"
    with pytest.raises(PresenceTurnError) as replay:
        _cached_result(tmp_path, refused.turn_ref)
    assert replay.value.code == "presence_attempt_outcome_unknown" and replay.value.work_ref == refused.work_ref


@pytest.mark.parametrize("origin,frozen,outcome,expected", [
    ("host_notice", "Old frozen diagnostic", "message", ""),
    ("host_salvage", "Old frozen partial", "deferred", ""),
    ("host_notice", None, "message", ""),
    ("model_final", "", "deferred", ""),
    ("", "", "deferred", ""),
    ("model_final", "Exact authored reply", "message", "Exact authored reply"),
    ("model_final", None, "message", "Raw retained text"),
    ("", None, "message", "Raw retained text"),  # explicit legacy unknown-origin compatibility
    ("", "Legacy frozen reply", "message", "Legacy frozen reply"),
    ("model_final", provider_terminal_body("Raw retained text", "Recorded host detail"), "message", "Raw retained text"),
    ("model_final", provider_terminal_body("Older explicit reply", "Recorded host detail"), "message",
     provider_terminal_body("Older explicit reply", "Recorded host detail")),
    ("", provider_terminal_body("Raw retained text", "Recorded host detail"), "message",
     provider_terminal_body("Raw retained text", "Recorded host detail")),
    ("model_final", "Unused text", "silent", ""),
    ("model_final", "Unused text", "tool_delivered", ""),
])
def test_cached_and_deferred_api_share_authorship_and_keyed_empty_body(tmp_path, origin, frozen, outcome, expected):
    _seed_token(tmp_path, skill="telegram-bot", token="presence-token",
                permissions=["presence"], manifest_permissions=["presence"])
    binding_id = _seed_presence_behavior(tmp_path)
    metadata = {"presence": {"binding_id": binding_id, "delivery_reporting_version": 1},
                "presence_outcome": outcome, "presence_work_ref": "later-child"}
    if frozen is not None:
        metadata["presence_result_text"] = frozen
    write_task_result(tmp_path, "completed-child", "completed", result="Raw retained text",
                      terminal_origin=origin, terminal_host_notice="Recorded host detail", metadata=metadata)
    cached = _cached_result(tmp_path, "completed-child")
    with TestClient(create_host_service_app(tmp_path)) as client:
        response = client.get("/presence/work/completed-child", params={"binding_id": binding_id},
                              headers={"X-Skill-Token": "presence-token"})
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == cached.text == expected
    assert body["outcome"] == cached.outcome
    assert body["status"] == "completed"  # producer lifecycle is independent of speech
    assert body["work_ref"] == "completed-child" and cached.work_ref == "later-child"
    assert body["delivery_reporting_version"] == cached.delivery_reporting_version == 1
    if origin in {"host_notice", "host_salvage"}:
        assert body["outcome"] == ("deferred" if outcome == "deferred" else "silent")


def test_fresh_missing_origin_never_borrows_legacy_authorship():
    task = {"id": "new"}
    result = build_presence_result_event(task, "Unknown producer", SimpleNamespace())
    assert result["outcome"] == "silent" and result["text"] == ""
    assert task["metadata"]["presence_result_text"] == ""
    assert "terminal_origin" not in task  # absence is not relabelled as host


@pytest.fixture
def native_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_module.OuroborosAgent, "_log_worker_boot_once", lambda *_a: None)
    monkeypatch.setattr(agent_module, "validate_task_authority_sources", lambda *_a: None)
    monkeypatch.setattr(agent_module.OuroborosAgent, "_start_task_heartbeat_loop", lambda *_a: None)
    agent = agent_module.OuroborosAgent(agent_module.Env(repo_dir=tmp_path, drive_root=tmp_path))
    ctx = agent.tools._ctx
    monkeypatch.setattr(agent, "_prepare_task_context", lambda *_a: (ctx, [], {}))
    return agent, ctx


@pytest.mark.parametrize("case", ["accepted_empty", "nonstring", "exception", "budget"])
def test_native_host_replacement_resets_authorship_and_keeps_diagnostic(tmp_path, monkeypatch, native_agent, case):
    from ouroboros.usage_accounting import BudgetExceeded

    agent, ctx = native_agent
    ctx._presence_completion = {"outcome": "message"}
    ctx._presence_completion_accepted = True

    accepted = []
    real_loop = agent_module.run_llm_loop
    if case == "accepted_empty":
        monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
        ctx.task_contract = {"capability_ceiling": presence_ceiling_payload(_admission().capability_ceiling)}
        ctx.is_direct_chat = True
        agent.llm = SimpleNamespace(default_model=lambda: "test-model")
        replies = iter([_call("message", ""), {"content": ""}])
        monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_kw: (next(replies), 0.0))

    def run(**kwargs):
        if case == "accepted_empty":
            result = real_loop(**kwargs)
            accepted.append((ctx._presence_completion_accepted, dict(result[1])))
            return result
        if case == "exception":
            error = RuntimeError("Synthetic processing failure")
            error._ouroboros_loop_usage = {"terminal_origin": "model_final"}
            raise error
        if case == "budget":
            raise BudgetExceeded("Synthetic exhausted budget")
        return ("" if case == "accepted_empty" else None), {
            "terminal_origin": "model_final", "presence_completion_outcome": "message",
        }, {"tool_calls": [], "reasoning_notes": []}

    monkeypatch.setattr(agent_module, "run_llm_loop", run)
    events = agent._handle_task_scoped({"id": "replacement", "chat_id": 7, "type": "presence",
        "_presence_turn": True, "_is_direct_chat": True, "_skip_post_task_synthesis": True, "text": "Go"})
    result = next(row for row in events if row["type"] == "presence_result")
    stored = load_task_result(tmp_path, "replacement")
    assert stored["terminal_origin"] == "host_notice" and stored["result"]
    assert result["outcome"] == "silent" and result["text"] == ""
    assert _cached_result(tmp_path, "replacement").text == ""
    if case in {"accepted_empty", "nonstring"}:
        assert ctx._presence_completion_accepted is False
        assert "empty response" in stored["result"]
        assert "presence_completion_outcome" not in stored["loop_outcome"].get("usage", {})
        if case == "accepted_empty":
            assert accepted[0][0] is True
            assert accepted[0][1]["terminal_origin"] == "model_final"
            assert accepted[0][1]["presence_completion_outcome"] == "message"
    else:
        assert stored["status"] == "failed"
        assert stored["reason_code"] == ("task_exception" if case == "exception" else "budget_exhausted")


@pytest.mark.parametrize("admission", ["scheduled", "unconfirmed", "rejected"])
def test_empty_deferred_final_keeps_only_confirmed_child_polling(tmp_path, monkeypatch, native_agent, admission):
    from ouroboros.tools.control_routing import _finish_swarm_handoff

    agent, ctx = native_agent
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    ctx.task_contract = {"capability_ceiling": presence_ceiling_payload(_admission().capability_ceiling)}
    ctx.task_metadata = {"presence": {"binding_id": "test-binding"}}
    ctx.is_direct_chat = True
    _finish_swarm_handoff(ctx, {"task_id": "managed-child"}, "Admission receipt", status=admission)
    agent.llm = SimpleNamespace(default_model=lambda: "test-model")
    replies, calls = iter([_call("deferred", ""), {"content": ""}]), []

    def respond(*_a, **_kw):
        calls.append(1)
        return next(replies), 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", respond)
    events = agent._handle_task_scoped({"id": "empty-deferred", "chat_id": 7, "type": "presence",
        "_presence_turn": True, "_is_direct_chat": True, "_skip_post_task_synthesis": True, "text": "Go",
        "metadata": {"presence": {"binding_id": "test-binding"}}})
    result = next(row for row in events if row["type"] == "presence_result")
    stored, cached = load_task_result(tmp_path, "empty-deferred"), _cached_result(tmp_path, "empty-deferred")
    assert len(calls) == 2 and ctx._presence_completion_accepted is False
    assert stored["terminal_origin"] == "host_notice" and "empty response" in stored["result"]
    assert stored["status"] == "completed"  # the existing lifecycle does not release child custody
    assert result["outcome"] == cached.outcome == ("deferred" if admission == "scheduled" else "silent")
    assert result["work_ref"] == cached.work_ref == ("managed-child" if admission == "scheduled" else "")
    assert result["text"] == cached.text == ""
