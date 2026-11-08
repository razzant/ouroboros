"""Batch4 group consumers composed with the published strict writer and waits."""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses

import pytest

from ouroboros import usage_accounting as ua
from tests.test_billing_group import _scope
from tests.test_physical_candidate_capture import data_root as data_root
from tests.test_usage_lock_continuity import held_lock, owner
from tests.test_usage_lock_continuity import short_acquisitions as short_acquisitions
from tests._usage_store_testing import request
from tests._usage_store_testing import root as root



def group_scope(root, task="child", root_id="successor", *, cap=10, root_cap=None):
    return _scope(root, task, root_id, group="original", group_limit=cap, root_limit=root_cap,
                  parent_task_id=root_id if task != root_id else "",
                  billing_group_limit_source="initial_task_admission", billing_group_limit_revision="original-rev")


@pytest.mark.parametrize("axis", ["root", "group", "global"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_pre_send_wait_revalidates_late_charge_on_each_axis(root, monkeypatch, axis, asynchronous):
    from ouroboros import _usage_wait
    from ouroboros.task_results import write_task_result
    write_task_result(root, 'successor', 'running', root_task_id='successor')
    write_task_result(root, 'child', 'running', root_task_id='successor', parent_task_id='successor')
    scope = group_scope(root, cap=2 if axis == "group" else 20, root_cap=2 if axis == "root" else 20)
    other_root = "successor" if axis == "root" else "original" if axis == "group" else "foreign"
    other_scope = group_scope(root, task=other_root, root_id=other_root, cap=scope.billing_group_limit_usd)
    if axis == "global":
        other_scope = dataclasses.replace(other_scope, billing_group_id="foreign")
    with ua.usage_scope(other_scope):
        late = ua.reserve_attempt(request(root, task_id=other_root, root_task_id=other_root, reservation_usd=0))
        ua.mark_dispatched(late)
        ua.mark_unresolved(late, "late charge")
    sent, prepared, waits = [], [], []
    lock_owner = contextlib.ExitStack()
    original_hold = _usage_wait._hold
    def observed_wait(wait_owner, phase, *args):
        original_hold(wait_owner, phase, *args)
        waits.append(phase)
        if phase == "entered":
            # Release only after actual contention is observed. The real lock
            # keeps its production acquisition slice; setup/cleanup are not timed faults.
            lock_owner.close()
    monkeypatch.setattr(_usage_wait, "_hold", observed_wait)
    with contextlib.ExitStack() as stack:
        stack.callback(lock_owner.close)
        stack.enter_context(owner(root, task={"id": "child", "root_task_id": "successor"}))
        stack.enter_context(ua.usage_scope(scope))
        stack.enter_context(ua.physical_attempt_limit(1))
        def before(reservation):
            prepared.append(reservation)
            # The same writer receives the original/sibling's delayed receipt after reservation.
            with ua.usage_scope(None):
                # Its known price reaches the $2 cap exactly (#1487: equality refuses).
                ua.settle_attempt(late, cost_usd=2.0, cost_final=True)
            lock_owner.enter_context(held_lock(root))
        req = request(root, root_task_id="successor", global_limit_usd=2 if axis == "global" else 100)
        async def send():
            sent.append(1)
            return {"usage": {}}
        with pytest.raises(ua.BudgetExceeded) as error:
            if asynchronous:
                asyncio.run(ua.execute_physical_attempt_async(req, send, before_dispatch=before))
            else:
                ua.execute_physical_attempt(req, lambda: sent.append(1), before_dispatch=before)
        assert error.value.limit_scope == ("global" if axis == "global" else "root")
        assert error.value.root_task_id == "successor"
        assert error.value.physical_attempt_capture.state == "released"
        assert ua._PHYSICAL_LIMIT.get().used == 0
    assert not sent and len(prepared) == 1
    assert waits == ["entered", "ended"]
    assert ua.read_usage_records(root, final_only=True)[-1]["state"] == "released"
