"""Calls share ordinary post-task synthesis; lost live authority still refuses."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.test_call_task_lifecycle import direct_call as direct_call


@pytest.mark.parametrize("call", [False, True])
def test_admitted_calls_and_text_dispatch_the_same_post_task_work(direct_call, tmp_path, monkeypatch, call):
    from ouroboros import agent_task_pipeline as pipeline
    from supervisor import worker_chat_lane

    calls = []
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *args, **kwargs: calls.append(kwargs))
    metadata = {"execution_transport": direct_call.turn.request} if call else {}
    admitted = worker_chat_lane._admit_chat_task(SimpleNamespace(), 1, "Conversation", task_metadata=metadata,
                                                realtime_turn=direct_call.turn if call else None)
    assert admitted is not None
    try:
        pipeline.emit_task_results(
            SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path), None, None, [], admitted["task"],
            "Conversation ended.", {"rounds": 1}, {"tool_calls": [], "reasoning_notes": []}, 0.0, tmp_path / "logs")
        assert len(calls) == 1
        assert not admitted["task"].get("_skip_post_task_synthesis")
    finally:
        admitted["registry"].unregister(admitted["task"]["id"])


@pytest.mark.parametrize("seam", ["recovery", "preparation"])
@pytest.mark.parametrize("case", ["text", "live", "missing", "mismatched", "closed", "owner_stop"])
def test_only_restart_refusals_add_a_synthesis_skip(direct_call, seam, case):
    from ouroboros.agent import OuroborosAgent
    from ouroboros.realtime_turn import recovery_refusal

    turn = direct_call.turn
    task = {"id": "synthetic-task", "metadata": {}}
    if case != "text":
        task["metadata"]["execution_transport"] = dict(turn.request)
    if case == "mismatched":
        task["metadata"]["execution_transport"]["call_id"] = "another-call"
    if case == "closed":
        turn._closed = True
    if case == "owner_stop":
        task["_skip_post_task_synthesis"] = True
    active_turn = None if case in {"text", "missing"} else turn
    refused = case in {"missing", "mismatched", "closed"}
    if seam == "recovery":
        assert recovery_refusal(task, active_turn) == ("realtime_restart_required" if refused else "")
    else:
        class OrdinaryPreparation(Exception):
            pass
        def drive_path(name):
            raise OrdinaryPreparation(name)  # stop at the next ordinary step, before context/provider work
        agent = SimpleNamespace(env=SimpleNamespace(drive_path=drive_path), _realtime_turn=active_turn)
        if refused:
            assert OuroborosAgent._prepare_task_context(agent, task, {}) == (
                None, [], {"realtime_refusal": "realtime_restart_required"})
        else:
            with pytest.raises(OrdinaryPreparation):
                OuroborosAgent._prepare_task_context(agent, task, {})
    assert bool(task.get("_skip_post_task_synthesis")) is (refused or case == "owner_stop")


@pytest.mark.parametrize("call", [False, True])
def test_presence_calls_keep_the_text_turn_post_task_policy(direct_call, tmp_path, call):
    from ouroboros.presence_runner import PresenceTurnGate, run_presence_turn
    from tests.test_presence_runner import _admission, _event, _terminal

    repo, data = tmp_path / "repo", tmp_path / "presence-data"
    repo.mkdir()
    data.mkdir()
    captured = []
    class Agent:
        def handle_task(self, task):
            captured.append(task)
            _terminal(data, task, "Reply")
            return [{"type": "presence_result", "outcome": "message", "text": "Reply", "work_ref": ""}]
    turn = direct_call.turn
    turn.spec = replace(turn.spec, presence_binding_id=_admission().binding_id)
    result = run_presence_turn(admission=_admission(), event=_event(), repo_dir=repo, drive_root=data,
                               agent_factory=lambda **kwargs: Agent(), gate=PresenceTurnGate(2),
                               live_turn=turn if call else None)
    assert result.outcome == "message"
    assert not captured[0].get("_skip_post_task_synthesis")
