"""A granted cold wake discloses beacons once, without consuming hidden backlog."""

import json
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _fast_hold, _install_queue, _loop_ctx, _quiet_external

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("handoff", ["inline", "preview", "source", "source_unavailable"])
def test_real_cold_wake_adopts_only_inline_beacons(tmp_path, monkeypatch, handoff):
    from ouroboros import budget_pause, model_sleep, owner_wait
    from ouroboros import task_tree_ledger as ledger
    from ouroboros.owner_mailbox import acknowledged_task_message_ids, write_task_message
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.control_task_results import _wait_attention_poll
    from supervisor.events_budget import install_exact_budget_pause
    from supervisor.sleep_wake import wake_ready_sleepers

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_kw: 5.0)
    _fast_hold(monkeypatch, budget_pause)
    _quiet_external(monkeypatch, budget_pause)
    write_task_result(tmp_path, "sleeper", "running")
    write_task_result(tmp_path, "child", "interrupted", parent_task_id="sleeper",
                      root_task_id="sleeper", delegation_role="subagent",
                      **({"result": "r" * 20000} if handoff == "preview" else {}),
                      **({"verification_ledger": {"summary": {"large": "x" * 20000}}}
                         if handoff in {"source", "source_unavailable"} else {}))
    ctx, limit_ctx = _loop_ctx(tmp_path, "sleeper")
    limit_ctx.messages.append({"role": "tool", "tool_call_id": "call_b", "content": "done b"})
    chosen = model_sleep.selectors(ctx, tasks=["child"])
    assert model_sleep.request_sleep(ctx, chosen, "cold")["reason"] == "sleep_armed"
    with pytest.raises(budget_pause.BudgetPauseRequested) as parked:
        budget_pause.enter_cold_sleep(limit_ctx)
    budget_pause.end_dispatch_fence("sleeper")
    workers.RUNNING["sleeper"] = {"task": {"id": "sleeper", "type": "task", "chat_id": 0,
                                          "root_task_id": "sleeper"}, "worker_id": 0, "attempt": 1}
    supervisor = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING,
        PENDING=workers.PENDING, WORKERS=workers.WORKERS, sort_pending=lambda: None,
        persist_queue_snapshot=lambda reason="": True, bridge=None)
    install_exact_budget_pause(supervisor, "sleeper",
                               budget_pause.exact_pause_marker(parked.value.pause)["checkpoint"])
    assert wake_ready_sleepers(queue) == []
    assert write_task_message(tmp_path, "quiet progress", "sleeper", source_task_id="peer",
                              provenance="peer_task", msg_id="quiet")
    assert wake_ready_sleepers(queue) == []
    monkeypatch.setattr(ledger, "utc_now_iso", lambda: "2026-10-10T00:00:00Z")
    for index in range(7):
        assert ledger.tree_ledger_append("sleeper", "question", f"q-{index}", task_id="child",
                                         data_root=tmp_path).startswith("OK")
    wake = wake_ready_sleepers(queue)
    assert wake[0]["ok"] is True and wake[0]["ready"]["reason"] == "child_attention_beacon"
    assert [row["text"] for row in budget_pause.budget_pause_row(tmp_path, "sleeper")["sleep"]["wake_beacons"]["beacons"]] == [f"q-{i}" for i in range(5)]

    resumed, _limit = _loop_ctx(tmp_path, "sleeper")
    resumed.budget_pause_resume = workers.PENDING[0]["_budget_pause_resume"]
    state_blob = budget_pause.load_budget_pause(resumed)
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_args, **_kw: (None, "max"))
    if handoff == "source_unavailable":
        monkeypatch.setattr("ouroboros.artifacts.store_actor_source_bytes",
                            lambda *_args, **_kw: (_ for _ in ()).throw(OSError("spill unavailable")))
    messages = []
    budget_pause.resume_paused_loop(SimpleNamespace(_ctx=resumed), state_blob, messages,
                                    {}, {}, set(), budget_remaining_usd=5.0)
    disclosed = json.loads(messages[-1]["content"].rsplit("\n", 1)[-1])
    assert acknowledged_task_message_ids(tmp_path, "sleeper") == set()
    if handoff in {"inline", "preview"}:
        assert [row["text"] for row in disclosed["wake_beacons"]["beacons"]] == [f"q-{i}" for i in range(5)]
        assert bool(disclosed.get("preview_only")) == (handoff == "preview")
        expected = ["q-5", "q-6"]
    else:
        assert "wake_beacons" not in disclosed and disclosed["preview_only"]
        assert disclosed["complete_source"]
        if handoff == "source_unavailable":
            assert disclosed["complete_source"]["status"] == "unavailable"
        expected = [f"q-{i}" for i in range(5)]
    snapshot = _wait_attention_poll(resumed, "", ["child"], consume=False)({}, {})
    assert [row["text"] for row in snapshot["beacons"]] == expected
    next_poll = _wait_attention_poll(resumed, "", ["child"])({}, {})
    assert [row["text"] for row in next_poll["beacons"]] == expected
    if handoff in {"inline", "preview"}:
        assert next_poll["beacons_remaining"] == 0
    else:
        assert next_poll["beacons_remaining"] == 2
        tail = _wait_attention_poll(resumed, "", ["child"])({}, {})
        assert [row["text"] for row in tail["beacons"]] == ["q-5", "q-6"]
    assert _wait_attention_poll(resumed, "", ["child"])({}, {}) is None
    assert ledger.tree_ledger_append("sleeper", "question", "fresh", task_id="child",
                                     data_root=tmp_path).startswith("OK")
    fresh = _wait_attention_poll(resumed, "", ["child"])({}, {})
    assert [row["text"] for row in fresh["beacons"]] == ["fresh"]


def test_beacon_consumed_before_cold_park_does_not_wake_it_again(tmp_path, monkeypatch):
    from ouroboros import budget_pause, model_sleep
    from ouroboros import task_tree_ledger as ledger
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.control_task_results import _wait_attention_poll
    from supervisor.events_budget import install_exact_budget_pause
    from supervisor.sleep_wake import wake_ready_sleepers

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_kw: 5.0)
    _fast_hold(monkeypatch, budget_pause)
    _quiet_external(monkeypatch, budget_pause)
    write_task_result(tmp_path, "sleeper", "running")
    write_task_result(tmp_path, "child", "interrupted", parent_task_id="sleeper",
                      root_task_id="sleeper", delegation_role="subagent")
    ctx, limit_ctx = _loop_ctx(tmp_path, "sleeper")
    limit_ctx.messages.append({"role": "tool", "tool_call_id": "call_b", "content": "done b"})
    monkeypatch.setattr(ledger, "utc_now_iso", lambda: "2026-10-10T00:00:00Z")
    assert ledger.tree_ledger_append("sleeper", "question", "already read", task_id="child",
                                     data_root=tmp_path).startswith("OK")
    observed = _wait_attention_poll(ctx, "", ["child"])({}, {})
    assert [row["text"] for row in observed["beacons"]] == ["already read"]
    chosen = model_sleep.selectors(ctx, tasks=["child"])
    assert model_sleep.request_sleep(ctx, chosen, "cold")["reason"] == "sleep_armed"
    with pytest.raises(budget_pause.BudgetPauseRequested) as parked:
        budget_pause.enter_cold_sleep(limit_ctx)
    budget_pause.end_dispatch_fence("sleeper")
    workers.RUNNING["sleeper"] = {"task": {"id": "sleeper", "type": "task", "chat_id": 0,
                                          "root_task_id": "sleeper"}, "worker_id": 0, "attempt": 1}
    supervisor = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING,
        PENDING=workers.PENDING, WORKERS=workers.WORKERS, sort_pending=lambda: None,
        persist_queue_snapshot=lambda reason="": True, bridge=None)
    install_exact_budget_pause(supervisor, "sleeper",
                               budget_pause.exact_pause_marker(parked.value.pause)["checkpoint"])
    assert wake_ready_sleepers(queue) == []
    saved_cursor = budget_pause.budget_pause_row(tmp_path, "sleeper")["sleep_attention_cursors"]
    assert saved_cursor["sleeper:child"]["seen_ids"]
    assert ledger.tree_ledger_append("sleeper", "question", "new", task_id="child",
                                     data_root=tmp_path).startswith("OK")
    ready = wake_ready_sleepers(queue)
    assert ready[0]["ok"] is True and ready[0]["ready"]["reason"] == "child_attention_beacon"
    beacons = budget_pause.budget_pause_row(tmp_path, "sleeper")["sleep"]["wake_beacons"]["beacons"]
    assert [row["text"] for row in beacons] == ["new"]


def test_completion_between_first_readiness_and_cold_admission_does_not_replay_beacon(tmp_path, monkeypatch):
    from ouroboros import budget_pause, model_sleep, owner_wait
    from ouroboros import task_tree_ledger as ledger
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.control_task_results import _wait_attention_poll
    from supervisor.events_budget import install_exact_budget_pause
    from supervisor.sleep_wake import wake_ready_sleepers

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_kw: 5.0)
    _fast_hold(monkeypatch, budget_pause)
    _quiet_external(monkeypatch, budget_pause)
    write_task_result(tmp_path, "sleeper", "running")
    write_task_result(tmp_path, "child", "running", parent_task_id="sleeper",
                      root_task_id="sleeper", delegation_role="subagent")
    ctx, limit_ctx = _loop_ctx(tmp_path, "sleeper")
    limit_ctx.messages.append({"role": "tool", "tool_call_id": "call_b", "content": "done b"})
    assert {row["kind"] for row in model_sleep.cold_blockers(ctx)} >= {"running_member"}

    first_readiness = model_sleep.wake_reason
    observations = []

    def complete_after_initial_readiness(*args, **kwargs):
        reason = first_readiness(*args, **kwargs)
        observations.append(reason)
        if len(observations) == 1:
            assert reason == ""
            assert ledger.tree_ledger_append("sleeper", "question", "completion question",
                                             task_id="child", data_root=tmp_path).startswith("OK")
            write_task_result(tmp_path, "child", "completed", result="done")
        return reason

    monkeypatch.setattr(model_sleep, "wake_reason", complete_after_initial_readiness)
    chosen = model_sleep.selectors(ctx, tasks=["child"])
    assert model_sleep.request_sleep(ctx, chosen, "cold")["reason"] == "sleep_armed"
    assert observations == [""]
    assert model_sleep.cold_blockers(ctx) == []
    with pytest.raises(budget_pause.BudgetPauseRequested) as parked:
        budget_pause.enter_cold_sleep(limit_ctx)
    budget_pause.end_dispatch_fence("sleeper")
    workers.RUNNING["sleeper"] = {"task": {"id": "sleeper", "type": "task", "chat_id": 0,
                                          "root_task_id": "sleeper"}, "worker_id": 0, "attempt": 1}
    supervisor = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING,
        PENDING=workers.PENDING, WORKERS=workers.WORKERS, sort_pending=lambda: None,
        persist_queue_snapshot=lambda reason="": True, bridge=None)
    install_exact_budget_pause(supervisor, "sleeper",
                               budget_pause.exact_pause_marker(parked.value.pause)["checkpoint"])
    wake = wake_ready_sleepers(queue)
    assert wake[0]["ok"] is True and wake[0]["ready"]["reason"] == "child_attention_beacon"

    resumed, _limit = _loop_ctx(tmp_path, "sleeper")
    resumed.budget_pause_resume = workers.PENDING[0]["_budget_pause_resume"]
    state_blob = budget_pause.load_budget_pause(resumed)
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_args, **_kw: (None, "max"))
    messages = []
    budget_pause.resume_paused_loop(SimpleNamespace(_ctx=resumed), state_blob, messages,
                                    {}, {}, set(), budget_remaining_usd=5.0)
    disclosed = json.loads(messages[-1]["content"].rsplit("\n", 1)[-1])
    assert [row["text"] for row in disclosed["wake_beacons"]["beacons"]] == ["completion question"]
    assert _wait_attention_poll(resumed, "", ["child"])({}, {}) is None
