"""Mind controls share owner fences, grants and the predecessor CAS."""

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests.test_owner_continue import _interrupted
from tests.test_swarm_host_admission import host as _api_host

api_host = _api_host


def _event(action, target, nonce="mind-call-1"):
    return {"action": action, "task_id": target, "actor_task_id": "mind-root", "action_nonce": nonce,
            "reason": "finish the earlier request"}


def test_self_pause_refused_and_nonce_reuses_the_provider_call(tmp_path):
    from ouroboros.tools.control_mind import _continue_task, _pause_task
    from ouroboros.tool_call_log import invocation_scope

    emitted = []
    ctx = SimpleNamespace(task_id="mind-root", drive_root=tmp_path, pending_events=emitted, event_queue=None)
    assert json.loads(_pause_task(ctx, "mind-root"))["error"] == "self_pause_refused"
    assert json.loads(_continue_task(ctx, "other"))["error"] == "control_call_identity_missing"
    with invocation_scope({"tool_call_id": "provider-call"}):
        first = json.loads(_continue_task(ctx, "other", "interrupted"))
        again = json.loads(_continue_task(ctx, "other", "interrupted"))
    assert first["status"] == "requested" and first["action_nonce"] == again["action_nonce"]
    with invocation_scope({"tool_call_id": "next-call"}):
        assert json.loads(_continue_task(ctx, "other"))["action_nonce"] != first["action_nonce"]


def test_mind_pause_other_root_records_actor_and_resume_keeps_attribution(tmp_path, monkeypatch):
    from ouroboros.control_actor import pause_author
    from ouroboros.owner_pause import read_fence
    from ouroboros.task_results import write_task_result
    from supervisor.events_mind_controls import _apply

    q, _state, workers = _install_queue(tmp_path, monkeypatch)
    task = {"id": "other", "type": "task", "root_task_id": "other", "chat_id": 7, "_attempt": 1, "admitted_dispatch": "none"}
    workers.PENDING.append(task)
    write_task_result(tmp_path, "other", "scheduled", root_task_id="other")
    result = _apply(_event("pause_task", "other"))
    assert result["ok"]
    fence = read_fence(tmp_path, "other")
    assert fence["by"]["actor"] == "ouroboros" and fence["by"]["task_id"] == "mind-root"
    # The owner reads who decided and why: Ouroboros, in its own sentence.
    assert fence["by"]["actor"] == "ouroboros" and fence["by"]["text"] == "finish the earlier request"
    assert fence["by"]["at"] and pause_author(fence)["text"] == "finish the earlier request"
    assert _apply(_event("pause_task", "other"))["fence_id"] == fence["fence_id"]
    resumed = _apply(_event("resume_task", "other", "resume-call"))
    assert resumed["ok"], resumed
    assert read_fence(tmp_path, "other")["resume_actor"]["task_id"] == "mind-root"


def test_panic_hold_survives_mind_resume_but_owner_can_select_it(tmp_path, monkeypatch):
    from ouroboros.task_results import write_task_result
    from supervisor.events_budget import HOLD_PANIC, hold_budget_row
    from supervisor.events_mind_controls import _apply
    from supervisor.queue_transitions import resume_budget_paused_task

    q, _state, workers = _install_queue(tmp_path, monkeypatch)
    task = {"id": "other", "root_task_id": "other", "_attempt": 1, "type": "task", "admitted_dispatch": "none"}
    hold_budget_row(task, reason=HOLD_PANIC)
    workers.PENDING.append(task)
    write_task_result(tmp_path, "other", "scheduled", root_task_id="other")
    assert _apply(_event("resume_task", "other"))["error"] == "panic_requires_owner_resume"
    assert task["_budget_pause_hold"]["reason"] == HOLD_PANIC
    assert resume_budget_paused_task("other")["ok"]


@pytest.mark.parametrize("declared", ["owner", "ouroboros"])
def test_a_declared_pause_author_grants_a_mind_no_panic_release(tmp_path, monkeypatch, declared):
    """Provenance, never power: whoever a Pause's fence names, a mind's Resume of a Panic hold is refused."""
    from ouroboros.owner_pause import read_fence
    from ouroboros.task_results import write_task_result
    from supervisor.events_budget import HOLD_PANIC, hold_budget_row
    from supervisor.events_mind_controls import _apply
    from supervisor.owner_pause_control import request_owner_pause

    q, _state, workers = _install_queue(tmp_path, monkeypatch)
    task = {"id": "other", "root_task_id": "other", "_attempt": 1, "type": "task", "admitted_dispatch": "none"}
    workers.PENDING.append(task)
    write_task_result(tmp_path, "other", "scheduled", root_task_id="other")
    by = {"actor": declared, "text": "", "at": "2026-10-20T00:43:00+00:00"}
    assert request_owner_pause("other", request_id="press", by=by)["ok"]
    assert read_fence(tmp_path, "other")["by"] == by
    hold_budget_row(task, reason=HOLD_PANIC)
    assert _apply(_event("resume_task", "other"))["error"] == "panic_requires_owner_resume"
    assert task["_budget_pause_hold"]["reason"] == HOLD_PANIC
    # Nor does a declaration take power away: a declared Pause without Panic resumes as before.
    workers.PENDING.append({"id": "plain", "root_task_id": "plain", "_attempt": 1, "type": "task",
                            "admitted_dispatch": "none"})
    write_task_result(tmp_path, "plain", "scheduled", root_task_id="plain")
    assert request_owner_pause("plain", request_id="press-plain", by=by)["ok"]
    assert _apply(_event("resume_task", "plain", "resume-plain"))["ok"]


def test_mind_continue_and_owner_button_race_share_one_successor(tmp_path, monkeypatch):
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation
    from supervisor.events_mind_controls import _apply

    _q, _state, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        mind = executor.submit(_apply, _event("continue_task", "pred-1"))
        owner = executor.submit(admit_continuation, "pred-1", action_nonce="owner-button-call")
        answers = [mind.result(), owner.result()]
    accepted = [answer for answer in answers if answer.get("ok")]
    assert len(accepted) == 1, answers
    successor = accepted[0]["successor_task_id"]
    assert len(workers.PENDING) == 1 and workers.PENDING[0]["id"] == successor
    claim = load_task_result(tmp_path, "pred-1")["continued_by"]
    if answers[0].get("ok"):
        assert claim["by"]["actor"] == "ouroboros" and claim["by"]["task_id"] == "mind-root"
        assert load_task_result(tmp_path, successor)["continuation_admission"]["by"]["actor"] == "ouroboros"
    else:
        assert "actor" not in claim


def test_mind_continue_is_attributed_and_idempotent(tmp_path, monkeypatch):
    from ouroboros.task_results import load_task_result
    from supervisor.events_mind_controls import _apply

    _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path)
    event = _event("continue_task", "pred-1")
    first, again = _apply(event), _apply(event)
    assert first["ok"] and again["replay"]
    assert first["successor_task_id"] == again["successor_task_id"]
    assert load_task_result(tmp_path, "pred-1")["continued_by"]["by"]["task_id"] == "mind-root"


@pytest.mark.parametrize("source", ["owner", "panic"])
def test_continue_after_owner_stop_is_not_authorized_by_a_mind(tmp_path, monkeypatch, source):
    from supervisor.events_mind_controls import _apply

    _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, origin={"source": source, "reason": "owner_stop"})
    assert not _apply(_event("continue_task", "pred-1"))["ok"]


def test_control_refusal_arrives_as_system_task_mail_not_owner_words(tmp_path, monkeypatch):
    from ouroboros.owner_mailbox import _mailbox_path
    from supervisor.events_mind_controls import handle_mind_task_control

    _install_queue(tmp_path, monkeypatch)
    pushed = []
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, bridge=SimpleNamespace(push_log=pushed.append))
    handle_mind_task_control(_event("pause_task", "unknown"), ctx)
    record = json.loads(_mailbox_path(tmp_path, "mind-root").read_text(encoding="utf-8").splitlines()[0])
    assert record["kind"] == "task_message" and record["provenance"] == "system"
    assert json.loads(record["text"])["outcome"]["error"] == "task_not_live"
    assert pushed[0]["actor_task_id"] == "mind-root"


@pytest.mark.serial
@pytest.mark.parametrize("actor", ["ordinary-analysis", "silence-analysis"])
def test_api_selected_task_id_retains_pause_resume_continue(api_host, actor):
    from starlette.requests import Request

    from ouroboros.gateway.tasks import _create_task_from_body
    from ouroboros.owner_pause import read_fence
    from ouroboros.task_results import load_task_result
    from ouroboros.tool_call_log import invocation_scope
    from ouroboros.tools.control_mind import _request_control
    from supervisor import workers
    from supervisor.events_mind_controls import _apply

    request = Request({"type": "http", "app": SimpleNamespace(state=SimpleNamespace(
        drive_root=api_host.root, repo_dir=workers.REPO_DIR))})
    for task_id in (actor, "other"):
        response = _create_task_from_body(request, {"task_id": task_id, "description": "Ordinary work"})
        assert response.status_code == 200, response.body
    stored = load_task_result(api_host.root, actor)
    assert "work_diagnostic" not in stored["metadata"]
    ctx = SimpleNamespace(task_id=actor, drive_root=api_host.root, task_metadata=stored["metadata"],
                          pending_events=[], event_queue=None)
    _interrupted(api_host.root)
    for action, target in (("pause_task", "other"), ("resume_task", "other"), ("continue_task", "pred-1")):
        with invocation_scope({"tool_call_id": action}):
            receipt = json.loads(_request_control(ctx, action, target, "Complete the request"))
        assert receipt["ok"], receipt
        event = ctx.pending_events.pop()
        outcome = _apply(event)
        assert outcome["ok"], outcome
        if action == "pause_task":
            assert read_fence(api_host.root, target)["by"]["task_id"] == actor
        elif action == "resume_task":
            assert read_fence(api_host.root, target)["resume_actor"]["task_id"] == actor
        else:
            assert load_task_result(api_host.root, target)["continued_by"]["by"]["task_id"] == actor
            assert _apply(event)["successor_task_id"] == outcome["successor_task_id"]
    assert not api_host.attempts
