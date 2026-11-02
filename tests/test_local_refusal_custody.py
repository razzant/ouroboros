"""Producer-proven pre-effect refusals through registry and owner controls."""
import copy
import json
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests.test_owner_pause import _owner_park

pytestmark = pytest.mark.serial


def _world(tmp_path, monkeypatch):
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.registry import ToolRegistry

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_kw: 5.0)
    repo = tmp_path / "repo"
    repo.mkdir()
    for target in (repo / "notes.txt", tmp_path / "notes.txt"):
        target.write_text("alpha\nbeta\nalpha\n", encoding="utf-8")
    write_task_result(tmp_path, "root", "running", root_task_id="root", chat_id=0)
    workers.RUNNING["root"] = {"task": {"id": "root", "root_task_id": "root",
        "type": "task", "chat_id": 0}, "worker_id": 0, "attempt": 1}
    registry = ToolRegistry(repo_dir=repo, drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = "root"
    return registry, queue, workers, repo


def _pause_and_resume(root, monkeypatch, queue, workers):
    from ouroboros import budget_pause, owner_pause
    from supervisor.events_budget import install_exact_budget_pause
    from supervisor.owner_pause_control import request_owner_pause

    assert request_owner_pause("root", request_id="P")["ok"]
    pause = _owner_park(root, monkeypatch, "root", [], root="root")
    ctx = SimpleNamespace(DRIVE_ROOT=root, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
        WORKERS=workers.WORKERS, sort_pending=lambda: None,
        persist_queue_snapshot=queue.persist_queue_snapshot, bridge=None)
    install_exact_budget_pause(ctx, "root", budget_pause.exact_pause_marker(pause)["checkpoint"])
    assert owner_pause.read_fence(root, "root")["state"] == owner_pause.FENCE_PAUSED
    assert queue.resume_budget_paused_task("root")["ok"]


@pytest.mark.parametrize("root", ["active_workspace", "runtime_data"])
@pytest.mark.parametrize("case", ["missing_match", "duplicate_match", "missing_file"])
def test_editor_validation_finishes_without_a_writer_and_owner_can_resume(tmp_path, monkeypatch, root, case):
    from ouroboros.task_results import load_task_result

    registry, queue, workers, repo = _world(tmp_path, monkeypatch)
    target = (repo if root == "active_workspace" else tmp_path) / "notes.txt"
    before = target.read_bytes()
    result = registry.execute_result("edit_text", {"root": root,
        "path": "missing.txt" if case == "missing_file" else "notes.txt",
        "old_str": "not present" if case == "missing_match" else "alpha", "new_str": "gamma"})
    assert result.status != "ok", result
    assert result.meta["operation_outcome"] == "completed_no_effect", result
    assert target.read_bytes() == before
    assert load_task_result(tmp_path, "root").get("launch_handoffs", {}) == {}
    _pause_and_resume(tmp_path, monkeypatch, queue, workers)


@pytest.mark.parametrize("tool,args", [
    ("edit_text", {"path": "notes.txt", "old_str": "", "new_str": "gamma"}),
    ("write_file", {"path": "bad.py", "content": "def broken(:"}),
    ("write_file", {}),
    ("schedule_followup", {"objective": "check", "relation": "independent"}),
    ("schedule_followup", {"objective": "check", "relation": "independent", "run_at": "invalid"}),
    ("schedule_followup", {"objective": "", "relation": "independent", "run_at": "2099-01-01T00:00:00Z"}),
    ("manage_schedules", {"action": "invalid"}),
    ("delegate_start", {"prompt": ""}),
    ("delegate_start", {"prompt": "check", "continue_from": "run-old", "retry_of": "inv-old"}),
])
def test_pre_effect_argument_refusals_do_not_hold_continue(tmp_path, monkeypatch, tool, args):
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation
    from tests.test_owner_continue import _interrupted, _owner_mail, NONCE

    registry, queue, workers, repo = _world(tmp_path, monkeypatch)
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    effects = []
    monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway",
                        lambda *_a, **_kw: effects.append("daemon"))
    monkeypatch.setattr(queue, "upsert_scheduled_task", lambda *_a, **_kw: effects.append("schedule"))
    before = (repo / "notes.txt").read_bytes()
    result = registry.execute_result(tool, args)
    assert result.status != "ok", result
    assert result.meta.get("operation_outcome") == "completed_no_effect", result
    assert not effects and not (repo / "bad.py").exists()
    assert (repo / "notes.txt").read_bytes() == before
    assert load_task_result(tmp_path, "root").get("launch_handoffs", {}) == {}
    workers.RUNNING.clear()
    _interrupted(tmp_path, task_id="root")
    _owner_mail(tmp_path, task_id="root")
    admitted = admit_continuation("root", action_nonce=NONCE)
    assert admitted["ok"] and not admitted["held"], admitted


@pytest.mark.parametrize("root", ["active_workspace", "runtime_data"])
def test_actual_editor_write_still_completes_then_pauses(tmp_path, monkeypatch, root):
    registry, queue, workers, repo = _world(tmp_path, monkeypatch)
    result = registry.execute_result("edit_text", {"root": root, "path": "notes.txt",
        "old_str": "beta", "new_str": "gamma"})
    assert result.status == "ok", result
    assert result.meta.get("operation_outcome") != "completed_no_effect"
    assert ((repo if root == "active_workspace" else tmp_path) / "notes.txt").read_text(
        encoding="utf-8") == "alpha\ngamma\nalpha\n"
    _pause_and_resume(tmp_path, monkeypatch, queue, workers)


def test_a_write_then_error_is_not_mislabelled_a_pre_effect_refusal(tmp_path, monkeypatch):
    from ouroboros.task_results import load_task_result
    from ouroboros.tools import git

    registry, _queue, _workers, repo = _world(tmp_path, monkeypatch)
    def write_then_error(path, text):
        path.write_text(text, encoding="utf-8")
        raise OSError("injected failure after writing")
    monkeypatch.setattr(git, "write_text", write_then_error)
    result = registry.execute_result("edit_text", {"path": "notes.txt", "old_str": "beta", "new_str": "gamma"})
    assert result.status == "error"
    assert result.meta.get("operation_outcome") != "completed_no_effect"
    assert (repo / "notes.txt").read_text(encoding="utf-8") == "alpha\ngamma\nalpha\n"
    assert not load_task_result(tmp_path, "root").get("launch_handoffs", {}), "the write completed before its exception"


def _completion_refusal(registry, case):
    """A refused second selection keeps the first answer and its observation."""
    ctx = registry._ctx
    ctx._completion_observation = {"tool_count": 3}
    ctx._completion_conflict = False
    first = registry.execute_result("finish_task", {"action": "finish", "answer": "First complete answer."})
    assert first.status == "ok" and first.meta.get("operation_outcome") != "completed_no_effect"
    selected = copy.deepcopy(ctx._completion_request)
    args = ({"action": "stop", "answer": "Partial answer."} if case == "argument" else
            {"action": "finish", "answer": "A contradictory answer."})
    result = registry.execute_result("finish_task", args)
    text = ("ERROR: COMPLETION_ARGUMENT: stop requires a rationale naming unfinished work"
            if case == "argument" else "ERROR: COMPLETION_CONFLICT: contradictory completion requests "
            "in one response; select again after seeing all results.")
    assert (result.status, result.code, result.text) == ("error", "TOOL_ARG_ERROR", text)
    assert ctx._completion_request == selected
    assert ctx._completion_conflict is (case == "conflict")
    return result


@pytest.mark.parametrize("case", ["argument", "conflict"])
@pytest.mark.parametrize("actor", ["root", "child"])
@pytest.mark.parametrize("next_action", ["pause", "continue"])
def test_completion_refusal_keeps_owner_controls_available(tmp_path, monkeypatch, case, actor, next_action):
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.continuation_admission import admit_continuation
    from tests.test_owner_continue import NONCE, _interrupted, _owner_mail

    registry, task_queue, workers, _repo = _world(tmp_path, monkeypatch)
    if actor == "child":
        write_task_result(tmp_path, actor, "running", root_task_id="root", parent_task_id="root")
        registry._ctx.task_id = actor
        registry._ctx.parent_task_id = "root"
        registry._ctx.task_metadata = {"root_task_id": "root", "parent_task_id": "root",
                                       "delegation_role": "subagent"}
    result = _completion_refusal(registry, case)
    if actor == "child":
        # The parent's controls include the settled child's remaining custody.
        write_task_result(tmp_path, actor, "completed", result="Child's partial work")
    if next_action == "pause":
        _pause_and_resume(tmp_path, monkeypatch, task_queue, workers)
    else:
        workers.RUNNING.clear()
        _interrupted(tmp_path, task_id="root")
        _owner_mail(tmp_path, task_id="root")
        continued = admit_continuation("root", action_nonce=NONCE)
        assert continued["ok"] and not continued["held"], continued
    assert result.meta["operation_outcome"] == "completed_no_effect"
    assert not load_task_result(tmp_path, actor).get("launch_handoffs")


@pytest.mark.parametrize("action", ["finish", "stop"])
def test_valid_completion_still_selects_the_answer_and_settles(tmp_path, monkeypatch, action):
    from ouroboros.task_results import load_task_result

    registry, task_queue, workers, _repo = _world(tmp_path, monkeypatch)
    registry._ctx._completion_observation = {"tool_count": 3}
    registry._ctx._completion_conflict = False
    args = {"action": action, "answer": "Exact selected answer."}
    if action == "stop":
        args["rationale"] = "The remaining export is unfinished."
    result = registry.execute_result("finish_task", args)
    assert (result.status, result.code) == ("ok", "OK")
    assert json.loads(result.text) == {"status": "completion_requested", "completion_control": True,
                                      "action": action}
    selected = registry._ctx._completion_request
    assert selected["action"] == action and selected["answer"] == args["answer"]
    assert selected["rationale"] == args.get("rationale", "")
    assert selected["observation"] == {"tool_count": 3} and selected["source"] == "finish_task"
    assert registry._ctx._completion_conflict is False
    assert result.meta.get("operation_outcome") != "completed_no_effect"
    assert not load_task_result(tmp_path, "root").get("launch_handoffs")
    _pause_and_resume(tmp_path, monkeypatch, task_queue, workers)


@pytest.mark.parametrize("case", ["argument", "conflict"])
def test_completion_refusal_preserves_an_unrelated_unknown_call(tmp_path, monkeypatch, case):
    from ouroboros.owner_pause import operation_start, tool_handoff
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import conflicting_writers

    registry, task_queue, _workers, _repo = _world(tmp_path, monkeypatch)
    with pytest.raises(TimeoutError):
        with tool_handoff(registry._ctx, "unknown-external-effect"):
            with operation_start(registry._ctx):
                raise TimeoutError("a submitted effect has no terminal receipt")
    before = load_task_result(tmp_path, "root").get("launch_handoffs", {})
    assert len(before) == 1
    result = _completion_refusal(registry, case)
    assert result.meta["operation_outcome"] == "completed_no_effect"
    assert load_task_result(tmp_path, "root").get("launch_handoffs", {}) == before
    assert any(row["kind"] == "tool_handoff" and row.get("task_id") == "root"
               for row in conflicting_writers(task_queue, "root"))
