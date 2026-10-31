"""Count gaps retain the same unsent scheduler owner without weakening parent caps."""
from __future__ import annotations

import queue as stdqueue

import pytest

from ouroboros import store, task_store
from ouroboros._usage_wait import ACQUISITION_SLICE_SEC
from ouroboros.task_results import write_task_result
from ouroboros.tools.control_delegation import durable_direct_child_count, schedule_delegation_refusal
from supervisor import queue, workspace_preparation as prep
from tests.test_admission_contention import database_reader
from tests.test_workspace_preparation import env as env, spy_provisions, _WAIT_SEC

pytestmark = pytest.mark.serial
_PARENT_ROWS = "parent_task_id=? AND delegation_role='subagent'"
_LIMITS = {"cap": {"max_children": 1}, "fanout": {"may_fan_out": False}}


def _parent(env, limit="cap"):
    budget = {"may_delegate": True, "depth_remaining": 2, **_LIMITS[limit]}
    write_task_result(env.data, "parent", "running", root_task_id="root",
                      task_contract={"delegation_budget": budget})
    return {"delegation_budget": budget}


@pytest.mark.parametrize("acting", [True, False])
@pytest.mark.parametrize("limit", ["cap", "fanout"])
@pytest.mark.parametrize("contended", [False, True])
def test_final_count_contention_retains_the_eligible_first_child(env, monkeypatch, acting, limit, contended):
    _parent(env, limit)
    event = env.request("count-child") if acting else env.readonly("count-child", root="root")
    # readonly() uses the root as parent; this case shares the exact parent budget.
    event["parent_task_id"] = "parent"
    bus = env.ctx.event_queue = stdqueue.Queue()
    copies = spy_provisions(monkeypatch)
    original, reads, gaps, claims = task_store.task_facts, [], [], []

    def facts(root, *args, **kwargs):
        if kwargs.get("where") != _PARENT_ROWS:
            return original(root, *args, **kwargs)
        under_q, budget = queue._queue_lock._is_owned(), store._WAIT_BUDGET.get()
        reads.append((under_q, budget))
        # The interlocked recheck reads on the admission lane under the gate's slice budget, off Q.
        if contended and budget is not None and not gaps:
            claim = next(row for row in prep._LIVE.values() if row.task_id == "count-child")
            claims.append((claim.token, dict(claim.resource)))
            with database_reader(env.data, exclusive=True):
                try:
                    return original(root, *args, **kwargs)
                except store.StoreBusy as exc:
                    gaps.append(exc.reason)
                    raise
        return original(root, *args, **kwargs)

    monkeypatch.setattr(task_store, "task_facts", facts)
    assert env.dispatch_returns(event) and env.gen.wait_idle(_WAIT_SEC)
    assert bus.empty()  # retried at its own lane, never re-put
    row = env.row("count-child")
    assert row["status"] == "scheduled" and env.pending_ids() == ["count-child"]
    assert copies == (["count-child"] if acting else [])
    assert reads[0] == (False, None), "initial count lost its ordinary off-Q acquisition budget"
    assert not any(under_q for under_q, _budget in reads), "a count was read while Q was owned"
    assert (False, ACQUISITION_SLICE_SEC) in reads and gaps == (["contention"] if contended else [])
    if claims:  # the retry reused the same claim and, for a copied surface, the same copy
        assert row[prep.MARKER]["token"] == claims[0][0]
        assert row[prep.MARKER].get("resource", {}) == claims[0][1]
    assert env.dispatch_returns(event)  # an already admitted duplicate remains harmless
    assert env.pending_ids() == ["count-child"]


def test_contended_failed_child_body_count_rejoins_the_same_preparation(env, monkeypatch):
    _parent(env)
    write_task_result(env.data, "failed-prior", "failed", parent_task_id="parent", root_task_id="root",
                      delegation_role="subagent", delegation_admission={"status": "rejected"})
    event = env.request("body-count")
    copies, gaps = spy_provisions(monkeypatch), []
    original = task_store.documents_by_ids

    def bodies(root, ids):
        if store._WAIT_BUDGET.get() is not None and "failed-prior" in ids and not gaps:
            with database_reader(env.data, exclusive=True):
                try:
                    return original(root, ids)
                except store.StoreBusy as exc:
                    gaps.append(exc.reason)
                    raise
        return original(root, ids)

    monkeypatch.setattr(task_store, "documents_by_ids", bodies)
    assert env.dispatch_returns(event) and env.gen.wait_idle(_WAIT_SEC)
    assert gaps == ["contention"] and copies == ["body-count"]
    assert env.pending_ids() == ["body-count"] and env.row("body-count")["status"] == "scheduled"


@pytest.mark.parametrize("limit, expected", [
    ("cap", "delegation_rights_max_children"), ("fanout", "delegation_rights_may_fan_out"),
])
def test_retry_reads_the_real_count_and_preserves_the_parent_limit(env, monkeypatch, limit, expected):
    _parent(env, limit)
    event = env.request("cap-child")
    copies, gaps = spy_provisions(monkeypatch), []
    original = task_store.task_facts

    def facts(root, *args, **kwargs):
        if store._WAIT_BUDGET.get() is not None and kwargs.get("where") == _PARENT_ROWS and not gaps:
            caught = None
            with database_reader(env.data, exclusive=True):
                try:
                    return original(root, *args, **kwargs)
                except store.StoreBusy as exc:
                    caught = exc
                    gaps.append(exc.reason)
            # The next read must see newly committed authority, never the old
            # pre-preparation count or a guessed zero substituted for Busy.
            write_task_result(env.data, "other-admitted", "scheduled", parent_task_id="parent",
                              root_task_id="root", delegation_role="subagent")
            raise caught
        return original(root, *args, **kwargs)

    monkeypatch.setattr(task_store, "task_facts", facts)
    assert env.dispatch_returns(event) and env.gen.wait_idle(_WAIT_SEC)
    row = env.row("cap-child")
    assert gaps == ["contention"] and copies == ["cap-child"]
    assert row["status"] == "failed" and row["reason_code"] == expected
    assert row["delegation_admission"]["direct_child_count"] == 1 and env.pending_ids() == []


@pytest.mark.parametrize("failure", ["permission", "corrupt", "uninspectable", "unknown", "rollback_unknown"])
def test_genuine_unknown_count_is_not_temporary_contention(env, monkeypatch, failure):
    _parent(env)
    event = env.request("unknown-count")
    original, reads = task_store.task_facts, []
    error = (PermissionError("denied") if failure == "permission" else
             store.StoreLost("corrupt") if failure == "corrupt" else
             store.StoreBusy("unavailable", reason="contention" if failure == "rollback_unknown" else failure))
    if failure == "rollback_unknown":
        error.rollback_confirmed = False

    def facts(root, *args, **kwargs):
        if store._WAIT_BUDGET.get() is not None and kwargs.get("where") == _PARENT_ROWS:
            reads.append(True)
            raise error
        return original(root, *args, **kwargs)

    monkeypatch.setattr(task_store, "task_facts", facts)
    assert env.dispatch_returns(event) and env.gen.wait_idle(_WAIT_SEC)
    row = env.row("unknown-count")
    assert reads == [True] and row["status"] == "failed"
    assert row["reason_code"] == "delegation_rights_child_count_unknown"
    assert row["delegation_admission"]["direct_child_count"] is None and env.pending_ids() == []


def test_default_reader_and_producer_keep_unknown_without_a_retry_owner(env):
    contract = _parent(env)
    with database_reader(env.data, exclusive=True), store.acquisition_budget(0):
        assert durable_direct_child_count(env.data, "parent") is None
        # Unknown is not a refusal at the worker's pre-check: the host's interlock decides.
        assert schedule_delegation_refusal(contract, env.data, "parent") == ""
        with pytest.raises(store.StoreBusy) as caught:
            durable_direct_child_count(env.data, "parent", propagate_contention=True)
        assert caught.value.reason == "contention"
    assert durable_direct_child_count(env.data, "parent") == 0


# ---------------------------------------------------------------------------
# Owner promise (14 Oct): brief SQLite contention never refuses a subagent launch.
# Both pre-checks (the worker tool, the supervisor's early read) leave an unknown
# count to the interlocked re-check, which defers contention and refuses a hard unknown.
# ---------------------------------------------------------------------------

def _contended_parent_rows(monkeypatch, *, when):
    """Raise real-shaped store contention on the parent-rows count read while ``when()`` holds."""
    original, gaps = task_store.task_facts, []

    def facts(root, *args, **kwargs):
        if kwargs.get("where") == _PARENT_ROWS and when():
            gaps.append(queue._queue_lock._is_owned())
            raise store.StoreBusy("current store busy: database is locked", reason="contention")
        return original(root, *args, **kwargs)

    monkeypatch.setattr(task_store, "task_facts", facts)
    return gaps


def test_worker_tool_under_count_contention_schedules_instead_of_refusing(tmp_path, monkeypatch):
    from ouroboros.tools.control_scheduling import _schedule_task
    from ouroboros.tools.registry import ToolContext
    from tests._shared import configure_test_subagent

    configure_test_subagent(monkeypatch)
    budget = {"may_delegate": True, "depth_remaining": 2, "max_children": 3}
    write_task_result(tmp_path, "parent", "running", root_task_id="parent", task_contract={"delegation_budget": budget})
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, current_task_type="task")
    ctx.task_id, ctx.event_queue = "parent", stdqueue.Queue()
    ctx.task_metadata, ctx.task_contract = {"root_task_id": "parent"}, {"delegation_budget": budget}
    gaps = _contended_parent_rows(monkeypatch, when=lambda: True)

    result = _schedule_task(ctx, subagent_id="api-scout", objective="Collect the figures",
                            expected_output="figures", memory_mode="empty")

    assert not result.startswith("⚠️"), result
    assert gaps == [False], "the worker's own count read hit the contention"
    event = ctx.event_queue.get_nowait()
    assert event["type"] == "schedule_subagent" and event["parent_task_id"] == "parent"


def test_supervisor_early_count_contention_is_no_rejection_and_admits_after_release(env, monkeypatch):
    _parent(env, "cap")
    event = env.request("early-busy")
    gaps = _contended_parent_rows(monkeypatch, when=lambda: not queue._queue_lock._is_owned() and not gaps)

    assert env.dispatch_returns(event) and env.gen.wait_idle(_WAIT_SEC)

    assert gaps == [False], "only the early (off-Q) count was contended"
    row = env.row("early-busy")
    assert row["status"] == "scheduled" and env.pending_ids() == ["early-busy"], row
    assert row.get("delegation_admission", {}).get("status") != "rejected"

