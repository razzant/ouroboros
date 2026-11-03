"""Resume identity is one observation, including transient SQLite contention.

The recovered Astra contention and Opus double-read probes exposed a successful
root Resume with a surviving queue latch. This consumer contract instead refuses
the faulted press without changing its saved pause, then proves healthy recovery,
child admission and explicit (never automatic) selection of earlier children.
"""
import contextlib
import copy
import sys
import threading

import pytest

from ouroboros import budget_pause, cancel_intents, owner_pause
from supervisor import queue
from supervisor.owner_pause_control import request_owner_pause
from tests._store_journal_shared import exclusive_hold
from tests.test_execution_clock_resume_handover import clock  # noqa: F401
from tests.test_retry_pause_generations import _park, _running_actor
from tests.test_retry_pause_identity import _assign, _fresh_plan, _loop_once, reaper  # noqa: F401
from tests.test_swarm_host_admission import host  # noqa: F401

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("latch_pending", [False, True])
@pytest.mark.parametrize("fault", ["readable", "io", "lineage", "sqlite"])
def test_one_identity_read_refuses_unknown_then_resumes_the_tree(
        host, monkeypatch, clock, reaper, latch_pending, fault):  # noqa: F811
    running = _running_actor(host, monkeypatch, clock, reaper, "retry")
    plan = _fresh_plan(monkeypatch)
    child = queue.enqueue_task({"id": "unselected", "type": "task", "text": "child work", "chat_id": 0,
                                "root_task_id": "held", "parent_task_id": running["id"], "depth": 1,
                                "delegation_role": "subagent"})
    assert not child.get("_admission_blocked")
    assert request_owner_pause(running["id"], request_id="pause")["ok"]
    _park(host, running, plan)
    if latch_pending:
        queue.BUDGET_ROOT_FENCES.pop("held")
        child.pop("_budget_pause", None)  # the durable fence alone prevents launch
    before = copy.deepcopy(budget_pause.budget_pause_row(host.root, running["id"]))
    latch_before = copy.deepcopy(queue.BUDGET_ROOT_FENCES)
    real, reads, threads = cancel_intents._validated_single_cancel_target, [], []

    def observe(*args, **kwargs):
        if (sys._getframe(1).f_code.co_name == "root_actor"
                and sys._getframe(2).f_code.co_name == "_grant_exact_resume"):
            reads.append(args[1])
            if len(reads) == 1 and fault != "readable":
                if fault == "io":
                    raise OSError("one transient lineage read failure")
                if fault == "lineage":
                    raise cancel_intents.CancelIntentLineageIndeterminate("lineage temporarily unreadable")
                entered, release = threading.Event(), threading.Event()

                def writer():  # Original Astra probe: real second SQLite writer, no mocked StoreBusy.
                    with contextlib.closing(exclusive_hold(host.root / "state/current.sqlite")) as conn:
                        entered.set()  # the lineage read is busy, under WAL as under the rollback journal
                        release.wait(30)
                        conn.execute("ROLLBACK")

                thread = threading.Thread(target=writer)
                threads.append(thread)
                thread.start()
                try:
                    assert entered.wait(3)
                    return real(*args, **kwargs)
                finally:
                    release.set()
                    thread.join(3)
                    assert not thread.is_alive()
        return real(*args, **kwargs)

    with monkeypatch.context() as transient:
        transient.setattr(cancel_intents, "_validated_single_cancel_target", observe)
        first = queue.resume_budget_paused_task(running["id"])
    assert reads == ["held"], "one root classification per grant, never a second fence classification"
    if fault != "readable":
        assert first == {"ok": False, "error": "owner_pause_custody_unreadable",
                         "message": "Resume refused: owner pause custody unreadable"}, first
        assert budget_pause.budget_pause_row(host.root, running["id"]) == before
        assert queue.BUDGET_ROOT_FENCES == latch_before
        assert not _assign(host, monkeypatch), "neither the root nor unselected child dispatches"
        second = queue.resume_budget_paused_task(running["id"])
        assert second["ok"], second
    else:
        assert first["ok"], first
    [sent] = _assign(host, monkeypatch)
    assert sent["id"] == running["id"] and child in host.pending
    assert _loop_once(host, monkeypatch, sent, plan=plan)["kind"] == "pause"
    assert owner_pause.read_fence(host.root, "held")["state"] == "released"
    assert "held" not in queue.BUDGET_ROOT_FENCES
    assert not _assign(host, monkeypatch), "root Resume never auto-selects children"
    late = queue.enqueue_task({"id": "late-child", "type": "task", "text": "new child", "chat_id": 0,
                               "root_task_id": "held", "parent_task_id": running["id"], "depth": 1,
                               "delegation_role": "subagent"})
    assert not late.get("_admission_blocked"), late
    selected = queue.resume_budget_paused_task("unselected", selected_by=running["id"])
    assert selected["ok"], selected
    print("identity recovery", {"fault": fault, "latch_pending": latch_pending, "first": first,
                               "child_selected": selected["ok"], "sqlite_threads_ended": all(not t.is_alive() for t in threads)})
