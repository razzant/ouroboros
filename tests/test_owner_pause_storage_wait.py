"""Pure control reads retain the executing stack, never replaying a task effect."""
from __future__ import annotations

import contextlib
import copy
import json
import time
from types import SimpleNamespace

import pytest

from ouroboros import budget_pause, owner_pause, store
from ouroboros.model_wait import ModelWaitInterrupted, task_model_wait_scope
from ouroboros.task_results import write_task_result
from tests._budget_pause_exact_helpers import _loop_ctx
from tests.test_admission_contention import database_reader

pytestmark = pytest.mark.serial


@pytest.fixture
def boundary(tmp_path):
    # The fixture's independent SQLite connection must never reach a live root.
    root = (tmp_path / "owned").resolve()
    assert root.is_relative_to(tmp_path.resolve())
    root.mkdir()
    write_task_result(root, "same", "running")
    ctx, limit = _loop_ctx(root, "same", attempt=3)
    return root, ctx, limit


def _sleep(monkeypatch, callback):
    monkeypatch.setattr(owner_pause, "time", SimpleNamespace(
        monotonic=time.monotonic, time=time.time, sleep=callback))


@pytest.mark.parametrize("read_site", ["initial", "warm_hold"])
def test_real_boundary_waits_at_both_reads_without_replaying_effects(boundary, monkeypatch, read_site):
    root, ctx, limit = boundary
    before = copy.deepcopy((limit.messages, limit.llm_trace, limit.accumulated_usage))
    paused, waits = [], []
    if read_site == "warm_hold":
        owner_pause.install_fence(root, "same", request_id="already-paused")
    with contextlib.ExitStack() as held:
        release = []
        def request(*_args, **_kwargs):
            paused.append(True)
            release.append(held.enter_context(database_reader(root, exclusive=True)))
        monkeypatch.setattr(budget_pause, "request_pause", request)
        if read_site == "initial":
            release.append(held.enter_context(database_reader(root, exclusive=True)))

        def available(_seconds):
            waits.append(True)
            assert len(paused) == (read_site == "warm_hold")
            assert (limit.messages, limit.llm_trace, limit.accumulated_usage) == before
            with owner_pause.launch_lock(root, "same", wait=False):
                pass
            release[0]()
            if read_site == "warm_hold":
                owner_pause.release_fence(root, "same", reason="resume")
        _sleep(monkeypatch, available)
        budget_pause.enter_owner_pause(limit)
    assert waits == [True]
    assert ctx.task_id == "same" and ctx.task_attempt == 3 and limit.round_idx == 4
    assert (limit.messages, limit.llm_trace, limit.accumulated_usage) == before
    rows = [json.loads(line) for line in (root / "logs/events.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row.get("checkpoint_kind") == "control_store_wait"]
    assert [row["phase"] for row in rows] == ["entered", "ended"]
    assert rows[0]["state"] == "waiting" and rows[0]["episode_id"] == rows[1]["episode_id"]
    assert all("rail" not in row and row["state"] not in {"paused", "pausing"} for row in rows)


def test_pause_arriving_during_storage_wait_is_observed_before_any_effect(boundary, monkeypatch):
    root, ctx, limit = boundary
    pauses = []
    with database_reader(root, exclusive=True) as release:
        def available(_seconds):
            release()
            owner_pause.install_fence(root, "same", request_id="pause-during-read")
        _sleep(monkeypatch, available)
        def park(*_args, **_kwargs):
            pauses.append(ctx._owner_pause_fence_id)
            raise budget_pause.BudgetPauseRequested({"pause_id": "exact"})
        monkeypatch.setattr(budget_pause, "request_pause", park)
        with pytest.raises(budget_pause.BudgetPauseRequested):
            budget_pause.enter_owner_pause(limit)
    assert pauses == [owner_pause.read_fence(root, "same")["fence_id"]]
    assert limit.round_idx == 4 and ctx.task_attempt == 3


@pytest.mark.parametrize("control", ["panic", "deadline", "cancelled", "finalize_requested"])
def test_existing_controls_remain_distinct_while_the_database_is_busy(boundary, monkeypatch, control):
    root, ctx, limit = boundary
    task = {"id": "same"}
    hooks = {}
    if control == "deadline":
        task["deadline_at"] = "2000-01-01T00:00:00+00:00"
    else:
        hooks["owner_control"] = lambda: control
    with task_model_wait_scope(task=task, drive_root=root, event_queue=None,
                               worker_slot_held=False, **hooks):
        with database_reader(root, exclusive=True) as release:
            if control == "finalize_requested":
                _sleep(monkeypatch, lambda _: release())
                budget_pause.enter_owner_pause(limit)
            else:
                _sleep(monkeypatch, lambda _: pytest.fail("ending control must not sleep"))
                with pytest.raises(ModelWaitInterrupted) as raised:
                    budget_pause.enter_owner_pause(limit)
                assert raised.value.control_reason == control


@pytest.mark.parametrize("kind", ["uninspectable", "unknown_rollback", "lost", "malformed", "permission"])
def test_non_contention_keeps_existing_authority_semantics(boundary, monkeypatch, kind):
    from ouroboros import store_hot_facts

    _root, ctx, _limit = boundary
    error = {"uninspectable": store.StoreBusy("denied", reason="uninspectable"),
             "unknown_rollback": store.StoreBusy("uncertain", reason="contention"),
             "lost": store.StoreLost("missing"), "malformed": ValueError("invalid"),
             "permission": PermissionError("denied")}[kind]
    if kind == "unknown_rollback":
        error.rollback_confirmed = False
    def fail(*_args, **_kwargs):
        raise error
    monkeypatch.setattr(store_hot_facts, "controls", fail)  # the one addressed control read
    _sleep(monkeypatch, lambda _: pytest.fail("not positive contention"))
    if isinstance(error, store.StoreBusy):
        with pytest.raises(store.StoreBusy) as raised:
            owner_pause.wait_member_fence(ctx)
        assert raised.value is error
    else:
        assert owner_pause.wait_member_fence(ctx)["state"] == "unknown"


def test_supervisor_reader_still_returns_contention_without_owning_a_wait(boundary, monkeypatch):
    root, ctx, _limit = boundary
    _sleep(monkeypatch, lambda _: pytest.fail("ordinary reader acquired a wait"))
    with database_reader(root, exclusive=True), store.acquisition_budget(0):
        with pytest.raises(store.StoreBusy) as raised:
            owner_pause.member_fence(ctx)
    assert raised.value.reason == "contention"
    assert owner_pause.member_fence(ctx) == {}
