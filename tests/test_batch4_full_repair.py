"""Full-qualification regressions through real admission and Pause consumers."""
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests.test_batch4_repair_compositions import _running
from tests.test_owner_pause import _owner_park
from tests.worker_pool_support import affirm_ready



def _park(root, monkeypatch, queue, workers):
    from ouroboros import budget_pause
    from supervisor.events_budget import install_exact_budget_pause
    from supervisor.owner_pause_control import request_owner_pause

    assert request_owner_pause("root", request_id="qualified-pause")["ok"]
    row = _owner_park(root, monkeypatch, "root", [], root="root")
    ctx = SimpleNamespace(DRIVE_ROOT=root, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
                          WORKERS=workers.WORKERS, sort_pending=lambda: None,
                          persist_queue_snapshot=queue.persist_queue_snapshot, bridge=None)
    install_exact_budget_pause(ctx, "root", budget_pause.exact_pause_marker(row)["checkpoint"])
    return row


def test_custody_becoming_unreadable_after_paused_state_refuses_resume(tmp_path, monkeypatch):
    from ouroboros import process_custody, owner_pause

    queue, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    _park(tmp_path, monkeypatch, queue, workers)
    assert owner_pause.read_fence(tmp_path, "root")["state"] == "paused"
    path = process_custody.ledger_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"corrupt":\n', encoding="utf-8")
    refused = queue.resume_budget_paused_task("root")
    assert refused["error"] == "owner_pause_effects_unsettled"
    assert any(row["kind"] == "process_custody_unreadable" for row in refused["blockers"])
    assert "root" in queue.BUDGET_ROOT_FENCES


@pytest.mark.parametrize("stage", ["observation", "assignment", "consumption"])
def test_exact_resume_never_opens_a_replacement_owner_fence(tmp_path, monkeypatch, stage):
    from ouroboros import budget_pause, owner_pause
    from ouroboros.model_wait import ModelWaitInterrupted
    from tests._budget_pause_exact_helpers import _loop_ctx
    from supervisor import continuation_admission

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_kw: 5.0)
    _running(tmp_path, workers)
    _park(tmp_path, monkeypatch, queue, workers)
    replacement = []
    def replace_fence():
        with owner_pause.launch_lock(tmp_path, "root"):
            owner_pause.release_fence(tmp_path, "root", reason="prior generation released")
        fence, created = owner_pause.install_fence(tmp_path, "root", request_id="replacement")
        assert created
        replacement.append(fence)
    if stage == "observation":
        observe = continuation_admission.conflicting_writers
        def observe_then_replace(*args, **kwargs):
            result = observe(*args, **kwargs)
            replace_fence()
            return result
        monkeypatch.setattr(continuation_admission, "conflicting_writers", observe_then_replace)
        assert queue.resume_budget_paused_task("root")["error"] == "selection_authority_changed"
    else:
        assert queue.resume_budget_paused_task("root")["ok"]
        sent = []
        workers.WORKERS[0] = affirm_ready(SimpleNamespace(wid=0, busy_task_id=None, reaping=False,
                                            in_q=SimpleNamespace(put=lambda task: sent.append(dict(task)))))
        if stage == "assignment":
            replace_fence()
        workers.assign_tasks()
        if stage == "assignment":
            assert not sent
        else:
            assert [task["id"] for task in sent] == ["root"]
            ctx, limit = _loop_ctx(tmp_path, "root")
            ctx.budget_pause_resume = sent[0]["_budget_pause_resume"]
            saved = budget_pause.load_budget_pause(ctx)
            replace_fence()
            monkeypatch.setattr("ouroboros.owner_wait.restore_continuation_state", lambda *_a, **_kw: None)
            monkeypatch.setattr("ouroboros.owner_wait.rebind_restored_route", lambda *_a, **_kw: (None, "max"))
            monkeypatch.setattr(budget_pause, "_hold_control_reason", lambda _ctx: "stop")
            usage = {}
            with pytest.raises(ModelWaitInterrupted):
                budget_pause.resume_paused_loop(limit.tools, saved, list(limit.messages), {}, usage, set(),
                                                budget_remaining_usd=5.0)
            assert usage["budget_pause_hold"]["error"] == "ValueError: owner_pause_fence_changed"
    assert owner_pause.read_fence(tmp_path, "root") == replacement[0]
