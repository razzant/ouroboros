"""Restart retention carries consumed beacon identities into cold readiness."""

import pytest

from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("fresh", [False, True])
def test_retained_warm_sleep_keeps_consumed_cursor_and_new_beacons(tmp_path, monkeypatch, fresh):
    from ouroboros import model_sleep, owner_wait
    from ouroboros import task_tree_ledger as ledger
    from ouroboros.budget_pause import exact_pause_marker
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.control_task_results import _wait_attention_poll
    from supervisor.restart_retention import retain_sleep_checkpoint
    from supervisor.sleep_wake import wake_ready_sleepers

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_kw: 5.0)
    write_task_result(tmp_path, "sleeper", "running")
    write_task_result(tmp_path, "child", "interrupted", parent_task_id="sleeper",
                      root_task_id="sleeper", delegation_role="subagent")
    ctx, limit = _loop_ctx(tmp_path, "sleeper")
    limit.messages.append({"role": "tool", "tool_call_id": "call_b", "content": "done b"})
    assert ledger.tree_ledger_append("sleeper", "question", "already read", task_id="child",
                                     data_root=tmp_path).startswith("OK")
    assert _wait_attention_poll(ctx, "", ["child"])({}, {})["beacons"][0]["text"] == "already read"
    chosen = model_sleep.selectors(ctx, tasks=["child"])
    assert model_sleep.request_sleep(ctx, chosen, "warm")["reason"] == "sleep_armed"
    checkpoint = owner_wait.checkpoint_owner_wait(ctx, limit.messages, {}, {}, 4, [], set())
    owner_wait.set_owner_wait(tmp_path, "sleeper", {**checkpoint, "state": "waiting"})
    retained = retain_sleep_checkpoint(tmp_path, "sleeper", 1)
    workers.PENDING.append({"id": "sleeper", "type": "task", "chat_id": 0, "root_task_id": "sleeper",
                            "_attempt": 1, "_budget_pause": exact_pause_marker(retained)})
    assert wake_ready_sleepers(queue) == []
    assert retained["sleep_attention_cursors"]["sleeper:child"]["seen_ids"]
    if fresh:
        assert ledger.tree_ledger_append("sleeper", "question", "new", task_id="child",
                                         data_root=tmp_path).startswith("OK")
        wake = wake_ready_sleepers(queue)
        assert wake[0]["ok"] and wake[0]["ready"]["reason"] == "child_attention_beacon"
        from ouroboros.budget_pause import budget_pause_row

        beacons = budget_pause_row(tmp_path, "sleeper")["sleep"]["wake_beacons"]["beacons"]
        assert [row["text"] for row in beacons] == ["new"]
