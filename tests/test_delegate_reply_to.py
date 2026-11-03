"""Exact question round trips without transferring the starter's powers."""
import json
import queue
from types import SimpleNamespace

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.delegate_interactions import _delegate_answer, _waiting_on_user_payload
from ouroboros.gateways.claudexor import pending_interactions
from ouroboros.owner_mailbox import (acknowledged_task_message_ids, drain_owner_entries,
                                     write_task_message)
from ouroboros.task_results import write_task_result
from ouroboros.tools.core import _forward_to_worker
from ouroboros.tools.core_artifacts import _escalate
from tests.test_delegate_message import _ctx, _install, _QuestionStub, _message_question


@pytest.fixture
def setup(tmp_path, monkeypatch):
    custody._CUSTODY.clear()
    for tid, parent in (("root", ""), ("nanny", "root"), ("peer", "root")):
        write_task_result(tmp_path, tid, "running", parent_task_id=parent, root_task_id="root")
    assert custody.emit(tmp_path, custody.STARTED, {
        "run_id": "run-1", "task_id": "nanny", "root_task_id": "root", "route": "some-route"})
    stub = _install(monkeypatch, _QuestionStub())
    ctx = _ctx(tmp_path, "nanny")
    ctx.task_metadata = {"parent_task_id": "root", "root_task_id": "root", "delegation_role": "subagent"}
    waiting = json.loads(_waiting_on_user_payload(ctx, "run-1", "running", 1, pending_interactions(stub.detail)))
    address = waiting["pending_interactions"][0]["reply_to"]
    yield ctx, stub, address
    custody._CUSTODY.clear()


def test_escalation_round_trip_keeps_question_and_starter_submits(setup, tmp_path):
    ctx, stub, address = setup
    assert address == {"starter_task_id": "nanny", "run_id": "run-1",
                       "interaction_id": "int-current", "question_ids": ["q-port"]}
    result = _escalate(ctx, question="Which port?", assumption="Wait for a decision", reply_to=address)
    assert result.startswith("OK: escalated")
    mail = drain_owner_entries(tmp_path, "root")
    assert mail[0]["reply_to"] == address
    assert mail[0]["provenance"] == "descendant_task"
    parent = _ctx(tmp_path, "root")
    assert "Message forwarded" in _forward_to_worker(parent, "nanny", "Use 9090", reply_to=address)
    answer = drain_owner_entries(tmp_path, "nanny")[0]
    assert answer["reply_to"] == address
    assert not stub.answers  # Neither mail delivery nor the address submits anything.
    result = _delegate_answer(ctx, "run-1", "int-current", [{"question_id": "q-port", "free_text": "Use 9090"}], reply_to=address)
    assert json.loads(result.text)["status"] == "delivered"
    assert stub.answers[0][0][0:2] == ("run-1", "int-current")


@pytest.mark.parametrize("change", ["starter", "run", "interaction", "questions", "resolved", "different_pending", "wrong_run_detail", "unreadable", "plan"])
def test_wrong_or_stale_address_never_selects_another_question(setup, change):
    ctx, stub, address = setup
    address = dict(address)
    if change == "starter":
        address["starter_task_id"] = "root"
    elif change == "run":
        address["run_id"] = "different-run"
    elif change == "interaction":
        address["interaction_id"] = "newer"
    elif change == "questions":
        address["question_ids"] = ["different-question"]
    elif change == "resolved":
        stub.detail = {"pendingInteractions": []}
    elif change == "different_pending":
        row = _message_question()
        row["interactionId"] = "newer"
        stub.detail = {"pendingInteractions": [row]}
    elif change == "wrong_run_detail":
        stub.detail["pendingInteractions"][0]["runId"] = "different-run"
    elif change == "unreadable":
        stub.detail = {}
    else:
        address["plan_run_id"] = address.pop("interaction_id")
    result = json.loads(_delegate_answer(ctx, "run-1", "int-current", [{"question_id": "q-port", "free_text": "yes"}], reply_to=address).text)
    assert result["reason"] in {"reply_to_mismatch", "reply_to_unresolved"}
    assert not stub.answers


def test_legacy_absent_address_keeps_explicit_ids_and_never_infers_latest(setup):
    ctx, stub, _ = setup
    answers = [{"question_id": "q-port", "free_text": "yes"}]
    assert json.loads(_delegate_answer(ctx, "run-1", "", answers).text)["reason"] == "missing_interaction_id"
    assert not stub.answers
    assert json.loads(_delegate_answer(ctx, "run-1", "int-current", answers).text)["status"] == "delivered"
    assert stub.answers[0][0][1] == "int-current"


def test_reply_address_cannot_redirect_mail_or_authorize_peer_answer(setup, tmp_path):
    ctx, stub, address = setup
    parent = _ctx(tmp_path, "root")
    assert "TOOL_ARG_ERROR" in _forward_to_worker(parent, "peer", "yes", reply_to=address)
    assert not drain_owner_entries(tmp_path, "peer")
    peer = _ctx(tmp_path, "peer")
    result = _delegate_answer(peer, "run-1", "int-current", [{"question_id": "q-port", "free_text": "approved"}], reply_to=address)
    assert json.loads(result.text)["reason"] == "run_not_owned" and not stub.answers
    assert "Message forwarded" in _forward_to_worker(peer, "nanny", "My recommendation", reply_to=address)
    assert drain_owner_entries(tmp_path, "nanny")[0]["provenance"] == "peer_task"


def test_address_survives_crash_before_insertion_and_ack_follows_insert(setup, tmp_path, monkeypatch):
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros import loop

    _, _, address = setup
    assert write_task_message(tmp_path, "yes", "nanny", source_task_id="peer", provenance="peer_task",
                              relation="sibling", msg_id="answer", reply_to=address)
    seen = set()
    assert drain_owner_entries(tmp_path, "nanny", seen)[0]["reply_to"] == address
    assert not acknowledged_task_message_ids(tmp_path, "nanny")
    assert drain_owner_entries(tmp_path, "nanny", set())[0]["reply_to"] == address
    messages = []
    ctx = SimpleNamespace(task_id="nanny", task_metadata={}, drive_root=tmp_path, task_attempt=1)
    append = loop._append_or_merge_user_message

    def observed(messages, text, **kwargs):
        assert not acknowledged_task_message_ids(tmp_path, "nanny")
        assert "reply_to=" in text and "int-current" in text and "starter submits" in text
        return append(messages, text, **kwargs)

    monkeypatch.setattr(loop, "_append_or_merge_user_message", observed)
    _drain_incoming_messages(messages, queue.Queue(), tmp_path, "nanny", queue.Queue(), set(), owner_ctx=ctx)
    assert "int-current" in json.dumps(messages)
    assert "answer" in acknowledged_task_message_ids(tmp_path, "nanny")
    assert not drain_owner_entries(tmp_path, "nanny", set())


def test_root_owner_question_and_late_answer_keep_address(setup, tmp_path, monkeypatch):
    from ouroboros.owner_quiz import quiz_answer_frame, recorded_answer_frame, quiz_states
    from ouroboros.tools import owner_delivery

    _, _, address = setup
    delivered = []
    monkeypatch.setattr(owner_delivery, "deliver_owner_event", lambda ctx, event: delivered.append(event) or "queued")
    assert _escalate(_ctx(tmp_path, "root"), question="Which port?", assumption="Keep waiting", reply_to=address).startswith("OK:")
    assert delivered[0]["reply_to"] == address
    block = quiz_states(tmp_path, "root")[delivered[0]["quiz_id"]]
    assert block["reply_to"] == address
    live = quiz_answer_frame(block, None, "9090")
    block.update(comment="9090", answered_index=None)
    assert recorded_answer_frame(block) == live
    assert "int-current" in live and "starter submits" in live


def test_owner_question_id_does_not_rejoin_a_different_reply_target(setup, tmp_path, monkeypatch):
    from ouroboros.owner_quiz import quiz_states
    from ouroboros.tools import owner_delivery

    _, _, address = setup
    delivered = []
    monkeypatch.setattr(owner_delivery, "deliver_owner_event", lambda ctx, event: delivered.append(event) or "queued")
    ctx = _ctx(tmp_path, "root")
    fields = {"question": "Which port?", "assumption": "Keep waiting", "request_id": "port-choice"}
    assert _escalate(ctx, **fields, reply_to=address).startswith("OK:")
    assert "QUIZ_REQUEST_CONFLICT" in _escalate(ctx, **fields)
    assert len(delivered) == 1
    block, = quiz_states(tmp_path, "root").values()
    assert block["reply_to"] == address


def test_multi_question_address_allows_an_explicit_partial_answer(setup):
    ctx, stub, address = setup
    stub.detail["pendingInteractions"][0]["questions"].append({"id": "other", "question": "Other?"})
    address = {**address, "question_ids": ["q-port", "other"]}
    result = _delegate_answer(ctx, "run-1", "int-current", [{"question_id": "q-port", "free_text": "9090"}], reply_to=address)
    assert json.loads(result.text)["status"] == "delivered"
    assert len(stub.answers) == 1


@pytest.mark.parametrize("read_seconds", [99, 101])
def test_address_validation_preserves_the_answer_deadline(setup, monkeypatch, read_seconds):
    from ouroboros import delegate_interactions as interactions

    ctx, stub, address = setup
    clock = [0.0]
    monkeypatch.setattr(interactions, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    get_run = stub.get_run

    def read(*args, **kwargs):
        clock[0] += read_seconds
        return get_run(*args, **kwargs)

    monkeypatch.setattr(stub, "get_run", read)
    result = json.loads(_delegate_answer(ctx, "run-1", "int-current",
        [{"question_id": "q-port", "free_text": "9090"}], reply_to=address).text)
    if read_seconds > interactions._ANSWER_DEADLINE_SEC:
        assert result["reason"] == "reply_to_unresolved" and not stub.answers
    else:
        assert result["status"] == "delivered"
        assert stub.answers[0][1]["timeout_sec"] == interactions._ANSWER_DEADLINE_SEC - read_seconds


def test_answer_transport_passes_only_an_explicit_remaining_timeout():
    import httpx
    from ouroboros.gateways.claudexor import ClaudexorGateway, DaemonEndpoint

    calls = []
    gateway = ClaudexorGateway(DaemonEndpoint("127.0.0.1", 1, "test"))
    gateway.close()
    gateway._client = SimpleNamespace(request=lambda *a, **kw: calls.append(kw) or
                                      httpx.Response(200, json={"status": "delivered"}))
    gateway.answer_interaction("run", "question", [])
    gateway.answer_interaction("run", "question", [], timeout_sec=12)
    assert "timeout" not in calls[0]
    assert calls[1]["timeout"].read == 12
    assert calls[0]["json"] == calls[1]["json"] == {"answers": []}


def test_schema_is_unconditional_and_plan_address_has_one_selector(setup, tmp_path):
    from ouroboros.tools.core import get_tools
    from ouroboros.tools.delegate import get_tools as delegated_tools

    for tool in get_tools() + delegated_tools():
        if tool.name in {"escalate", "forward_to_worker", "delegate_message", "delegate_answer"}:
            assert "reply_to" in tool.schema["parameters"]["properties"]
    _, _, address = setup
    with pytest.raises(ValueError, match="exactly one"):
        write_task_message(tmp_path, "yes", "nanny", source_task_id="root",
                           reply_to={**address, "plan_run_id": "plan"})
    plan_address = {**address, "plan_run_id": address["run_id"]}
    del plan_address["interaction_id"]
    assert write_task_message(tmp_path, "plan answer", "nanny", source_task_id="root", reply_to=plan_address)
    assert write_task_message(tmp_path, "legacy context", "nanny", source_task_id="root")
