"""Batch4 exact-range review F1-F4, driven through their real consumers.

F1: a fresh owner Pause supersedes a Resume grant already handed to the worker.
F2: a definite pre-effect refusal settles its own registry launch claim.
F3: a stale settlement census never re-pauses a root that Resume released.
F4: MCP transport entry (a stdio server start) waits for launch admission.
"""

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests.test_batch4_full_repair import _park
from tests.test_batch4_producer_custody import _registry



def test_a_stale_settlement_census_cannot_repause_a_resumed_queued_root(tmp_path, monkeypatch):
    """Resume of a queued root lands between refresh's census and its publication."""
    from ouroboros import owner_pause
    from ouroboros.task_results import STATUS_SCHEDULED, write_task_result
    from supervisor import continuation_admission
    from supervisor.owner_pause_control import refresh_owner_pause_tree, request_owner_pause
    from supervisor.queue_transitions import budget_pause_fact

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "queued-root", STATUS_SCHEDULED, chat_id=0)
    workers.PENDING.append({"id": "queued-root", "type": "task", "chat_id": 0, "_attempt": 1,
                            "root_task_id": "queued-root", "admitted_dispatch": "none"})
    real = continuation_admission.conflicting_writers
    armed, resumed = [True], []

    def census_then_resume(*args, **kwargs):
        writers = real(*args, **kwargs)
        if armed and armed.pop():
            resumed.append(queue.resume_budget_paused_task("queued-root"))
        return writers

    monkeypatch.setattr(continuation_admission, "conflicting_writers", census_then_resume)
    ack = request_owner_pause("queued-root", request_id="p")

    assert ack["ok"] is True and resumed and resumed[0]["owner_pause_released"] is True, resumed
    fence = owner_pause.read_fence(tmp_path, "queued-root")
    assert fence["state"] == owner_pause.FENCE_RELEASED, "a stale census revived released authority"
    assert fence["release_reason"] == "owner_resume_unstarted_root"
    assert ack["state"] == owner_pause.FENCE_RELEASED
    assert "queued-root" not in queue.BUDGET_ROOT_FENCES and budget_pause_fact(workers.PENDING[0]) is None
    assert refresh_owner_pause_tree("queued-root") == owner_pause.FENCE_RELEASED
    events = (tmp_path / "logs" / "events.jsonl").read_text()
    assert '"owner_pause_settled"' not in events


@pytest.mark.parametrize("tool,args", [
    ("github_read", {"method": "get", "kind": "issue", "target": 0}),
    ("github_write", {"method": "comment", "target": 3, "body": " "}),
    ("github_write", {"method": "update", "target": -1, "state": "closed"}),
    ("github_read", {"method": "get", "kind": "pr", "target": 0}),
    ("github_write", {"method": "review", "target": 0, "body": "x"}),
    ("github_write", {"method": "create_issue", "title": " "}),
    ("start_service", {"cmd": ["sleep", "1"], "name": "bad name!"}),
    ("start_service", {"cmd": ["sleep", "1"], "readiness": {"timeout_sec": -1}}),
])
def test_definite_pre_effect_refusal_releases_its_launch_claim(tmp_path, monkeypatch, tool, args):
    from ouroboros.task_results import load_task_result
    from ouroboros.tools import github
    from supervisor import state

    registry, queue, workers = _registry(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_kw: 5.0)
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")  # the credential gate precedes the handler
    launched = []
    monkeypatch.setattr(github, "_gh_run", lambda *a, **kw: launched.append(a))
    result = registry.execute_result(tool, args)
    assert not launched and result.status != "ok", result
    assert result.meta.get("operation_outcome") == "completed_no_effect", result
    assert not load_task_result(tmp_path, "root").get("launch_handoffs")
    _park(tmp_path, monkeypatch, queue, workers)
    assert queue.resume_budget_paused_task("root")["ok"] is True


def test_returned_github_parse_error_does_not_keep_local_execution_alive(tmp_path, monkeypatch):
    from ouroboros.task_results import load_task_result
    from ouroboros.tools import github
    from supervisor import state

    registry, queue, workers = _registry(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_kw: 5.0)
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    def gh_result(args, *_a, **_kw):
        if args[:3] == ["repo", "view", "--json"]:
            return github.GhResult(True, '{"nameWithOwner":"owner/selected","url":"https://github.example/owner/selected"}', 0, None, "")
        return github.GhResult(True, "{truncated", 0, None, "")
    monkeypatch.setattr(github, "_gh_run", gh_result)
    result = registry.execute_result("github_read", {"method": "get", "kind": "issue", "target": 5})
    assert "failed to parse issue JSON" in result.text
    assert result.meta.get("operation_outcome") != "completed_no_effect"
    assert not load_task_result(tmp_path, "root").get("launch_handoffs", {})
    _park(tmp_path, monkeypatch, queue, workers)
    assert queue.resume_budget_paused_task("root")["ok"]
