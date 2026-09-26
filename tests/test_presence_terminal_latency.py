"""Native Presence returns durable replies while existing synthesis retains custody."""

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import agent as agent_module, agent_task_pipeline as pipeline, loop
from ouroboros.model_wait import current_model_wait
from ouroboros.presence_runner import PresenceTurnGate, run_presence_turn
from ouroboros.task_results import load_task_result
from tests.test_presence_completion import _call
from tests.test_presence_runner import _admission, _event


@pytest.fixture
def agent_bootstrap(tmp_path, monkeypatch):
    # These tests exercise task completion, not installation boot. A cold worker
    # otherwise scans host processes/Git and profiles the machine inside the
    # synthesis barrier's timeout, depending on earlier tests' boot-log state.
    monkeypatch.setattr(agent_module.OuroborosAgent, "_log_worker_boot_once", lambda *_a: None)
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "WORLD.md").write_text("# Test environment\n", encoding="utf-8")


@pytest.mark.parametrize("outcome", ["message", "silent", "tool_delivered"])
def test_native_agent_returns_durable_result_before_synthesis_and_retry_reuses_it(tmp_path, monkeypatch, outcome, agent_bootstrap):
    monkeypatch.delenv("OUROBOROS_IN_WORKER", raising=False)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setattr(agent_module, "validate_task_authority_sources", lambda *_a: None)
    monkeypatch.setattr(agent_module.OuroborosAgent, "_start_task_heartbeat_loop", lambda *_a: None)
    monkeypatch.setattr(agent_module.subagent_runtime, "apply_task_start_settings_or_disclose", lambda *_a: None)
    started, release = threading.Event(), threading.Event()
    stages, owners, agents = [], [], []
    threads_before = set(threading.enumerate())

    def consolidate(*_a, **_kw):
        owners.append(current_model_wait())
        started.set()
        assert release.wait(5), "test releases the existing synthesis worker"
        stages.append("consolidation")

    monkeypatch.setattr(pipeline, "_run_chat_consolidation", consolidate)
    monkeypatch.setattr(pipeline, "_run_scratchpad_consolidation", lambda *_a: stages.append("scratchpad"))
    monkeypatch.setattr(pipeline, "_record_task_facts", lambda *_a, **_k: stages.append("facts"))
    monkeypatch.setattr(pipeline, "_run_reflection", lambda *_a, **_k: stages.append("reflection"))
    monkeypatch.setattr(pipeline, "_apply_reflection_memory_actions", lambda *_a, **_k: stages.append("memory"))
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_k: (_call(outcome, "Reply" if outcome == "message" else ""), 0.0))

    # Cold tool-catalog construction is setup, outside the completion-order barrier.
    ready_agents = [agent_module.OuroborosAgent(agent_module.Env(
        repo_dir=tmp_path, drive_root=tmp_path)) for _ in range(2)]

    def factory(**_kwargs):
        actual = ready_agents.pop(0)
        actual.llm = SimpleNamespace(default_model=lambda: "test-model")

        def prepare(task, _refusal):
            ctx = actual.tools._ctx
            ctx.task_contract = task["task_contract"]
            ctx.task_metadata = task["metadata"]
            ctx.is_direct_chat = True
            ctx.owner_message_admission_lock = actual._owner_message_admission_lock
            ctx.owner_message_admission_agent = actual
            actual._persist_running_record(task)
            return ctx, [{"role": "user", "content": task["text"]}], {}

        actual._prepare_task_context = prepare
        agents.append(actual)
        return actual

    kwargs = dict(admission=_admission(), event=_event(), repo_dir=tmp_path, drive_root=tmp_path,
                  agent_factory=factory, gate=PresenceTurnGate(1, state_root=tmp_path))
    try:
        with ThreadPoolExecutor(max_workers=1) as requests:
            first_future = requests.submit(run_presence_turn, **kwargs)
            assert started.wait(3)
            first = first_future.result(timeout=2)
            assert first.outcome == outcome
            assert first.text == ("Reply" if outcome == "message" else "")
            row = load_task_result(tmp_path, first.task_id)
            assert row["status"] == "completed"
            assert row["result"] == first.text
            assert row["root_phase_checkpoint"]["post_task_synthesis"] == "running"
            assert row["cost_final"] is False
            assert owners[0] is not None and owners[0].worker_slot_held is False
            assert run_presence_turn(**kwargs) == first
            assert len(agents) == 1
            # The previous task's model/tools ended; the same conversation can
            # accept its next event even while prior memory work is in flight.
            second = requests.submit(run_presence_turn, **{**kwargs, "event": replace(
                _event(), source_event_id="telegram:bot-1:43", text="Next",
            )}).result(timeout=2)
            assert second.task_id != first.task_id and len(agents) == 2
            assert not release.is_set()
    finally:
        release.set()
        for thread in set(threading.enumerate()) - threads_before:
            thread.join(timeout=5)
    assert stages.count("facts") == stages.count("reflection") == 2
    for task_id in (first.task_id, second.task_id):
        row = load_task_result(tmp_path, task_id)
        assert row["root_phase_checkpoint"]["post_task_synthesis"] == "completed"
        assert row["cost_final"] is True
        assert (str(tmp_path.resolve()), task_id) not in pipeline._POST_TASK_SYNTHESIS_INFLIGHT


@pytest.mark.parametrize("worker,split,presence,expected", [
    (False, False, True, False), (True, False, True, True),
    (False, True, True, True), (False, False, False, True),
])
def test_only_native_unsplit_presence_changes_post_task_wait(tmp_path, monkeypatch, worker, split, presence, expected):
    monkeypatch.setattr(pipeline, "in_worker_process", lambda: worker)
    calls = []
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *_a, **kw: calls.append(kw["blocking"]))
    task = {"id": "test", "type": "presence" if presence else "task", "_is_direct_chat": True,
            "_presence_turn": presence}
    env = SimpleNamespace(drive_root=tmp_path)
    pipeline._dispatch_root_post_task(env, task, "Done", None, [], {}, {}, {}, tmp_path / "logs",
        budget_drive_root="", split_drive=split, project_scoped=False, project_task=False,
        parent_env=env if split else None, parent_task=task if split else None)
    assert calls == [expected]


def test_pending_finish_cannot_hide_a_failed_empty_agent_result(tmp_path, monkeypatch, agent_bootstrap):
    monkeypatch.setattr(agent_module, "validate_task_authority_sources", lambda *_a: None)
    monkeypatch.setattr(agent_module.OuroborosAgent, "_start_task_heartbeat_loop", lambda *_a: None)
    agent = agent_module.OuroborosAgent(agent_module.Env(repo_dir=tmp_path, drive_root=tmp_path))
    ctx = agent.tools._ctx
    ctx._presence_completion = {"outcome": "silent"}
    ctx._presence_completion_accepted = True
    monkeypatch.setattr(agent, "_prepare_task_context", lambda *_a: (ctx, [], {}))
    monkeypatch.setattr(agent_module, "run_llm_loop", lambda **_kw: (
        "", {"execution_status": "failed"}, {"tool_calls": [], "reasoning_notes": []},
    ))
    events = agent._handle_task_scoped({"id": "failed", "chat_id": 7, "type": "presence", "_presence_turn": True,
        "_is_direct_chat": True, "_skip_post_task_synthesis": True, "text": "Go"})
    result = next(row for row in events if row["type"] == "presence_result")
    assert result["outcome"] == "silent" and result["text"] == ""
    stored = load_task_result(tmp_path, "failed")
    assert stored["status"] == "failed" and "empty response" in stored["result"]
    assert stored["terminal_origin"] == "host_notice"
