"""A damaged task record leaves only its own row unknown, through the real tree consumers.

The census keeps every member it can read. A damaged row a surviving link ties to a tree
holds that tree alone, named; a row no link ties is a named gap that holds no tree and is
never read as absent. Pause, Resume, Continue, cold sleep and both cascade modes are driven
directly; the hot-facts job never publishes a root another task's body names.
"""
from __future__ import annotations

import contextlib
import json
from types import SimpleNamespace

import pytest

from ouroboros import maintenance_jobs, store, store_hot_facts as hot, task_store
from tests._budget_pause_exact_helpers import _install_queue
from tests.test_hot_facts import _old_task, _task

# What a row's stored body can be left as, and the census's reason (None: no body row at all).
DAMAGE = {
    "torn": ("{", "body_unreadable"),
    "array": ("[]", "body_not_object"),
    "null": ("null", "body_not_object"),
    "unnamed": ("{}", "identity_missing"),
    # Another task's whole document, naming the very tree under test as its root.
    "foreign": (json.dumps({"task_id": "other-task", "status": "running", "root_task_id": "root-1"}),
                "identity_mismatch"),
    "missing": (None, "body_missing"),
}


def _damage(root, tid, kind, **typed):
    """An underived row TID (typed columns as an older writer left them) whose body is DAMAGE[kind]."""
    with store.open_store(root, write=True) as txn:
        _old_task(txn, tid, **typed)
        body = DAMAGE[kind][0]
        if body is None:
            txn.conn.execute("DELETE FROM task_bodies WHERE task_id=?", (tid,))
        else:
            txn.conn.execute("UPDATE task_bodies SET body=? WHERE task_id=?", (body, tid))


def _tree(root):
    """root-1 with a derived child and a metadata-only legacy member, beside an unrelated root."""
    root.mkdir(exist_ok=True)
    with store.open_store(root, write=True) as txn:
        _task(txn, "root-1", status="running")
        _task(txn, "child", parent_task_id="root-1", root_task_id="root-1", status="running")
        _old_task(txn, "legacy", parent_task_id="root-1", metadata={"root_task_id": "root-1"}, status="completed")
        _task(txn, "other", status="running")


# --- the census itself ---------------------------------------------------------------------------

@pytest.mark.parametrize("descendants", [False, True], ids=["root", "descendants"])
@pytest.mark.parametrize("kind", list(DAMAGE))
def test_a_damaged_row_is_named_and_every_readable_member_kept(tmp_path, kind, descendants):
    _tree(tmp_path)
    _damage(tmp_path, "stray", kind)
    census = task_store.tree_facts(tmp_path, "root-1", descendants=descendants)
    assert set(census.members) == {"root-1", "child", "legacy"}
    assert census.members["legacy"]["root_task_id"] == "root-1"
    assert census.unresolved == {"stray": {"reason": DAMAGE[kind][1], "linked": False}}
    # The neighbouring tree's census names the same row and nothing of root-1.
    assert task_store.tree_facts(tmp_path, "other").unresolved == census.unresolved


@pytest.mark.parametrize("tie", ["parent", "member_parent", "retry_of_root", "superseded", "caller_id", "handoff",
                                 "own_root"])
def test_a_damaged_row_tied_to_one_tree_is_linked_there_only(tmp_path, tie):
    _tree(tmp_path)
    typed = {"parent": {"parent_task_id": "child"}, "superseded": {"superseded_by": "child"}}.get(tie, {})
    _damage(tmp_path, "stray" if tie != "own_root" else "root-2", "torn", **typed)
    with store.open_store(tmp_path, write=True) as txn:
        if tie == "member_parent":
            _task(txn, "grandchild", parent_task_id="stray", root_task_id="root-1", status="running")
        if tie == "retry_of_root":  # a parentless retry: only the root's forward link names it
            _task(txn, "root-1", status="failed", retry_task_id="stray")
        if tie == "handoff":
            txn.put_handoff("stray", "launch_handoffs", "op-1",
                            {"tool": "shell", "task_id": "stray", "root_task_id": "root-1", "state": "claimed"})
    tree = "root-2" if tie == "own_root" else "root-1"
    linked = {"stray"} if tie == "caller_id" else ()
    damaged = "root-2" if tie == "own_root" else "stray"
    census = task_store.tree_facts(tmp_path, tree, linked=linked)
    assert census.unresolved == {damaged: {"reason": "body_unreadable", "linked": True}}
    # The same row is only a named, unlinked gap for every other tree.
    assert task_store.tree_facts(tmp_path, "other").unresolved == {damaged: {"reason": "body_unreadable",
                                                                            "linked": False}}


def test_a_malformed_handoff_ties_nothing_and_refuses_nothing(tmp_path):
    _tree(tmp_path)
    _damage(tmp_path, "stray", "torn")
    with store.open_store(tmp_path, write=True) as txn:
        txn.conn.execute("INSERT INTO task_handoffs (task_id, op_id, kind, facts) "
                         "VALUES ('stray', 'op-1', 'launch_handoffs', '{')")
    assert task_store.tree_facts(tmp_path, "root-1").unresolved == {"stray": {"reason": "body_unreadable",
                                                                             "linked": False}}


def test_a_flagged_row_is_a_member_only_when_its_own_body_names_it(tmp_path):
    """control_valid=0 also means a bad status: identity decides, then the typed root is trusted."""
    _tree(tmp_path)
    with store.open_store(tmp_path, write=True) as txn:
        _task(txn, "odd", parent_task_id="root-1", root_task_id="root-1", status="")  # bad status, own body
        _task(txn, "copied", root_task_id="root-1", status="")
        document = txn.load("copied")
        txn.conn.execute("UPDATE task_bodies SET body=? WHERE task_id='copied'",
                         (store.encode_document({**document, "task_id": "someone-else"}),))
        assert txn.row("odd", "control_valid") == txn.row("copied", "control_valid") == 0
    census = task_store.tree_facts(tmp_path, "root-1")
    assert "odd" in census.members and "copied" not in census.members
    assert census.unresolved == {"copied": {"reason": "identity_mismatch", "linked": False}}
    # The member read refuses odd's invalid status: named and linked, while the others still answer.
    from ouroboros.owner_pause import tree_member_results

    read = tree_member_results(tmp_path, "root-1")
    assert set(read.members) == {"root-1", "child", "legacy"}
    assert read.unresolved == {"copied": {"reason": "identity_mismatch", "linked": False},
                               "odd": {"reason": "record_unreadable", "linked": True}}


def test_a_store_failure_still_refuses_the_whole_census(tmp_path, monkeypatch):
    from ouroboros.owner_pause import tree_member_results
    from supervisor.terminal_cascade import _cascade_descendant_rows

    _tree(tmp_path)
    _damage(tmp_path, "stray", "torn")

    @contextlib.contextmanager
    def lost(*_args, **_kwargs):
        raise store.StoreLost("store file unreadable")
        yield

    monkeypatch.setattr(task_store, "open_store", lost)
    with pytest.raises(store.StoreLost):
        task_store.tree_facts(tmp_path, "root-1")
    with pytest.raises(ValueError, match="tree_membership_unreadable"):
        tree_member_results(tmp_path, "root-1")
    with pytest.raises(store.StoreLost):
        _cascade_descendant_rows(tmp_path, "root-1", strict=True)
    gaps = []
    assert _cascade_descendant_rows(tmp_path, "root-1", gaps=gaps) == {}
    assert gaps == ["descendant_enumeration_unavailable"], "a queue-only fallback is never a complete census"


# --- Pause, Resume, Continue, cold sleep ----------------------------------------------------------

def _live_root(tmp_path, workers, task_id, chat_id=0):
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    write_task_result(tmp_path, task_id, STATUS_RUNNING, chat_id=chat_id)
    workers.RUNNING[task_id] = {"task": {"id": task_id, "type": "task", "chat_id": chat_id, "root_task_id": task_id},
                                "worker_id": 0, "attempt": 1}


def _pause_and_park(tmp_path, monkeypatch, queue, workers, task_id):
    from ouroboros import budget_pause
    from supervisor.events_budget import install_exact_budget_pause
    from supervisor.owner_pause_control import request_owner_pause
    from tests.test_owner_pause import _owner_park

    assert request_owner_pause(task_id, request_id=f"press-{task_id}")["ok"]
    row = _owner_park(tmp_path, monkeypatch, task_id, [], root=task_id)
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
                          WORKERS=workers.WORKERS, sort_pending=lambda: None,
                          persist_queue_snapshot=queue.persist_queue_snapshot, bridge=None)
    install_exact_budget_pause(ctx, task_id, budget_pause.exact_pause_marker(row)["checkpoint"])


@pytest.mark.parametrize("kind", list(DAMAGE))
def test_an_unattributed_damaged_row_lets_pause_settle_and_resume_proceed(tmp_path, monkeypatch, kind):
    from ouroboros import owner_pause

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    _live_root(tmp_path, workers, "solo")
    _damage(tmp_path, "stray", kind)
    _pause_and_park(tmp_path, monkeypatch, queue, workers, "solo")
    fence = owner_pause.read_fence(tmp_path, "solo")
    assert fence["state"] == owner_pause.FENCE_PAUSED
    assert fence["membership_gaps"] == {"rows": [{"task_id": "stray", "reason": DAMAGE[kind][1]}], "count": 1}
    assert queue.resume_budget_paused_task("solo")["ok"] is True


GAP_WORDS = (" The Pause did not wait for 1 saved task record that could not be read and that nothing ties to this "
             "task (stray); whether it belongs here is unknown.")


@pytest.mark.serial  # binds the shared chat writer, as tests/test_pause_notices.py does
def test_the_owner_reads_the_gap_in_the_pause_row_and_keeps_it_after_resume(tmp_path, monkeypatch):
    from ouroboros import pause_notices
    from ouroboros.startup_migrations import prepare_startup_state
    from supervisor import message_bus
    from tests.test_pause_notices import _rows

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 1})
    monkeypatch.setattr(message_bus, "get_bridge", lambda: SimpleNamespace(send_message=lambda *_a, **_k: None))
    prepare_startup_state(tmp_path)
    _live_root(tmp_path, workers, "solo", chat_id=1)  # a notice needs its task's own chat route
    _damage(tmp_path, "stray", "torn")
    _pause_and_park(tmp_path, monkeypatch, queue, workers, "solo")
    pause_notices.reconcile_pause_notices(tmp_path)
    def notices():
        return [row["text"] for row in _rows(tmp_path) if str(row.get("card_row_id") or "").startswith("pause:solo:")]

    assert notices() == ["This task was paused. Its work is retained; Resume continues it." + GAP_WORDS]
    assert queue.resume_budget_paused_task("solo")["ok"] is True
    assert notices() == ["This task was paused. Its work is retained; Resume continues it." + GAP_WORDS]
    # Without a gap the owner's row keeps its exact words.
    assert pause_notices.notice_text({"reason": "owner", "by": {"actor": "owner"}}) == (
        "You paused this task. Its work is retained; Resume continues it.")
    two = {"count": 23, "rows": [{"task_id": "a", "reason": "body_unreadable"}, {"task_id": "b", "reason": "x"}]}
    assert pause_notices.notice_text({"reason": "owner", "membership_gaps": two}).endswith(
        "23 saved task records that could not be read and that nothing ties to this task (a, b and 21 more); "
        "whether they belong here is unknown.")


def test_only_a_still_closed_pause_lends_its_gap_to_a_resume(tmp_path):
    from ouroboros import owner_pause
    from ouroboros.task_results import write_task_result

    write_task_result(tmp_path, "solo", "running")
    assert owner_pause.paused_gap(tmp_path, "solo") == {} == owner_pause.paused_gap(tmp_path, "absent")
    fence, _created = owner_pause.install_fence(tmp_path, "solo", request_id="press")
    gap = {"rows": [{"task_id": "stray", "reason": "body_unreadable"}], "count": 1}
    owner_pause.set_fence_state(tmp_path, "solo", fence_id=fence["fence_id"], state=owner_pause.FENCE_PAUSED,
                                membership_gaps=gap)
    assert owner_pause.paused_gap(tmp_path, "solo") == {"membership_gaps": gap}
    owner_pause.set_fence_state(tmp_path, "solo", fence_id=fence["fence_id"], state=owner_pause.FENCE_RELEASED)
    assert owner_pause.read_fence(tmp_path, "solo")["membership_gaps"] == gap  # kept on the record
    assert owner_pause.paused_gap(tmp_path, "solo") == {}  # never lent to a later, different Resume


@pytest.mark.serial  # binds the shared chat writer, as tests/test_pause_notices.py does
def test_a_gap_found_after_the_pause_is_named_by_the_resume_not_rewritten_into_its_notice(tmp_path, monkeypatch):
    from ouroboros import owner_pause, pause_notices
    from ouroboros.startup_migrations import prepare_startup_state
    from supervisor import message_bus
    from supervisor.owner_pause_control import refresh_owner_pause_tree
    from tests.test_pause_notices import _rows

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 1})
    monkeypatch.setattr(message_bus, "get_bridge", lambda: SimpleNamespace(send_message=lambda *_a, **_k: None))
    prepare_startup_state(tmp_path)
    _live_root(tmp_path, workers, "solo", chat_id=1)
    _damage(tmp_path, "stray", "torn")
    _pause_and_park(tmp_path, monkeypatch, queue, workers, "solo")
    pause_notices.reconcile_pause_notices(tmp_path)
    chat = _rows(tmp_path)
    _damage(tmp_path, "stray-2", "array")  # found only after the Pause settled
    assert refresh_owner_pause_tree("solo") == owner_pause.FENCE_PAUSED
    assert owner_pause.read_fence(tmp_path, "solo")["membership_gaps"]["count"] == 2
    assert refresh_owner_pause_tree("solo") == owner_pause.FENCE_PAUSED  # an unchanged census writes nothing
    pause_notices.reconcile_pause_notices(tmp_path)
    assert _rows(tmp_path) == chat, "the older notice keeps its event-time words and nothing new is sent"
    ack = queue.resume_budget_paused_task("solo")
    assert ack["ok"] is True
    assert ack["membership_gaps"] == {"rows": [{"task_id": "stray", "reason": "body_unreadable"},
                                               {"task_id": "stray-2", "reason": "body_not_object"}], "count": 2}
    assert ack["message"] == ("Resume accepted. The Pause did not wait for 2 saved task records that could not be "
                              "read and that nothing ties to this task (stray, stray-2); whether they belong here "
                              "is unknown.")


def _counted_censuses(monkeypatch):
    from supervisor import continuation_admission

    calls, real = [], continuation_admission.conflicting_writers
    monkeypatch.setattr(continuation_admission, "conflicting_writers",
                        lambda *a, **k: calls.append(a[1]) or real(*a, **k))
    return calls


AFTER = {"task_id": "stray-2", "reason": "body_not_object"}
TIED = {"kind": "member_record_unreadable", "task_id": "stray-2", "detail": "body_not_object"}


@pytest.mark.parametrize("case", ["clean", "after", "beside_the_pause_gap", "tied"])
def test_the_resume_census_names_a_row_damaged_after_the_pause_settled(tmp_path, monkeypatch, case):
    """Exact Resume of a parked root: no settle tick re-read the tree after the damage, only Resume's own census."""
    from ouroboros import owner_pause
    from ouroboros.task_results import load_task_result

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    _live_root(tmp_path, workers, "solo")
    if case == "beside_the_pause_gap":
        _damage(tmp_path, "stray", "torn")
    _pause_and_park(tmp_path, monkeypatch, queue, workers, "solo")
    fence, notices = owner_pause.read_fence(tmp_path, "solo"), load_task_result(tmp_path, "solo").get("pause_notices")
    assert fence["state"] == owner_pause.FENCE_PAUSED
    if case != "clean":
        _damage(tmp_path, "stray-2", "array", **({"parent_task_id": "solo"} if case == "tied" else {}))
    censuses = _counted_censuses(monkeypatch)
    ack = queue.resume_budget_paused_task("solo")
    assert censuses == ["solo"], "Resume's own census only: none added for the gap"
    if case == "tied":  # a row a link ties to the tree still holds it, named
        assert ack["ok"] is False and ack["error"] == "owner_pause_effects_unsettled"
        assert TIED in ack["blockers"]
        assert "membership_gaps" not in ack
    elif case == "clean":
        assert ack["ok"] is True and "membership_gaps" not in ack and "message" not in ack
    else:
        rows = [{"task_id": "stray", "reason": "body_unreadable"}, AFTER] if case != "after" else [AFTER]
        assert ack["ok"] is True
        assert ack["membership_gaps"] == {"rows": rows, "count": len(rows)}
        assert ack["message"].startswith(f"Resume accepted. The Pause did not wait for {len(rows)} saved task record")
        assert ack["message"].endswith(f"({', '.join(row['task_id'] for row in rows)}); whether "
                                       f"{'it belongs' if len(rows) == 1 else 'they belong'} here is unknown.")
    # History is not rewritten: the fence and the Pause row keep what the Pause's own census saw.
    assert owner_pause.read_fence(tmp_path, "solo").get("membership_gaps") == fence.get("membership_gaps")
    assert load_task_result(tmp_path, "solo").get("pause_notices") == notices


@pytest.mark.parametrize("case", ["clean", "after", "tied"])
def test_an_admitted_unstarted_root_resume_names_the_gap_its_census_saw(tmp_path, monkeypatch, case):
    """Pause A of a queued root whose dispatch was admitted: its Resume's fresh tree census decides."""
    from ouroboros import owner_pause
    from ouroboros.task_results import STATUS_SCHEDULED, write_task_result
    from supervisor.owner_pause_control import request_owner_pause

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "queued-root", STATUS_SCHEDULED, chat_id=0)
    workers.PENDING.append({"id": "queued-root", "type": "task", "chat_id": 0, "_attempt": 1,
                            "root_task_id": "queued-root", "admitted_dispatch": "possible"})
    assert request_owner_pause("queued-root", request_id="p")["ok"]
    gap_before = owner_pause.read_fence(tmp_path, "queued-root").get("membership_gaps")
    if case != "clean":
        _damage(tmp_path, "stray-2", "array", **({"parent_task_id": "queued-root"} if case == "tied" else {}))
    censuses = _counted_censuses(monkeypatch)
    ack = queue.resume_budget_paused_task("queued-root")
    assert censuses == ["queued-root"], "Resume's own census only: none added for the gap"
    if case == "tied":
        assert ack["ok"] is False and ack["error"] == "owner_pause_effects_unsettled"
        assert TIED in ack["blockers"]
        assert "membership_gaps" not in ack
        return
    assert ack["ok"] is True and ack["owner_pause_released"] is True and ack["never_started"] is False
    if case == "clean":
        assert "membership_gaps" not in ack and "message" not in ack
    else:
        assert ack["membership_gaps"] == {"rows": [AFTER], "count": 1}
        assert ack["message"].endswith("(stray-2); whether it belongs here is unknown.")
    assert owner_pause.read_fence(tmp_path, "queued-root").get("membership_gaps") == gap_before


def _warm_root(tmp_path, _monkeypatch, workers):
    """A live root that parks WARM under the Pause: its worker keeps the stack, nothing is re-queued."""
    _live_root(tmp_path, workers, "solo")
    return "solo", lambda: workers.RUNNING["solo"].update(owner_wait={"state": "waiting", "reason": "owner_pause"})


def _answered_tree(tmp_path, _monkeypatch, workers):
    """An answered root whose helpers the Pause holds: one parked warm, one never started."""
    from ouroboros.task_results import write_task_result
    from tests.test_owner_pause import _live_tree

    _live_tree(tmp_path, workers)
    workers.RUNNING.pop("root-1")
    write_task_result(tmp_path, "root-1", "completed")
    return "root-1", lambda: workers.RUNNING["child-run"].update(owner_wait={"state": "waiting", "reason": "owner_pause"})


def _late_review(tmp_path, monkeypatch, _workers):
    """An answered root whose prepared review is not sent yet (that fact alone is given): Resume sends it."""
    from ouroboros import review_operation
    from ouroboros.task_results import write_task_result

    write_task_result(tmp_path, "solo", "completed")
    monkeypatch.setattr(review_operation, "task_has_live_review_operation",
                        lambda _root, task_id, *, sent_only=False, **_k: task_id == "solo" and not sent_only)
    return "solo", lambda: None


# door: (its tree, the field its accepted Resume answers with, the censuses it reads once tied to nothing)
DOORS = {"warm_root": (_warm_root, "warm", 1), "answered_tree": (_answered_tree, "tree_only", 1),
         "late_review": (_late_review, "late_phase", 3)}


@pytest.mark.parametrize("case", ["clean", "after", "tied"])
@pytest.mark.parametrize("door", list(DOORS))
def test_every_resume_door_names_a_row_damaged_after_the_pause_settled(tmp_path, monkeypatch, door, case):
    """Before the exact door: a warm root, an answered tree and its unsent review each answer with their own census."""
    from ouroboros import owner_pause
    from supervisor.owner_pause_control import refresh_owner_pause_tree, request_owner_pause

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    build, answer, reads = DOORS[door]
    root, park = build(tmp_path, monkeypatch, workers)
    assert request_owner_pause(root, request_id=f"press-{door}")["ok"]
    park()
    assert refresh_owner_pause_tree(root) == owner_pause.FENCE_PAUSED
    fence = owner_pause.read_fence(tmp_path, root)
    if case != "clean":
        _damage(tmp_path, "stray-2", "array", **({"parent_task_id": root} if case == "tied" else {}))
    censuses = _counted_censuses(monkeypatch)
    ack = queue.resume_budget_paused_task(root)
    if case == "tied":  # a row a link ties to the tree still holds it, named, and the Pause stays closed
        assert ack["ok"] is False and ack["error"] == "owner_pause_effects_unsettled"
        assert TIED in ack["blockers"] and "membership_gaps" not in ack
        assert owner_pause.fence_closed(owner_pause.read_fence(tmp_path, root))
        return
    assert censuses == [root] * reads, "the door's own censuses only: none added for the gap"
    assert ack["ok"] is True and ack[answer], ack
    if case == "clean":
        assert "membership_gaps" not in ack and "message" not in ack
    else:
        assert ack["membership_gaps"] == {"rows": [AFTER], "count": 1}
        assert ack["message"] == "Resume accepted." + GAP_WORDS.replace("(stray)", "(stray-2)")
    assert owner_pause.read_fence(tmp_path, root).get("membership_gaps") == fence.get("membership_gaps")


def test_a_resume_joins_its_census_to_the_pause_gap_each_row_once_within_the_bound():
    from ouroboros.pause_notices import GAP_ROWS, gap_fact

    rows = [{"task_id": f"r{n}", "reason": "body_unreadable"} for n in range(GAP_ROWS + 4)]
    pause = gap_fact(rows[:GAP_ROWS + 3])  # names twenty, counts twenty-three
    assert pause == {"rows": rows[:GAP_ROWS], "count": GAP_ROWS + 3}
    # Resume's census reads them all again and one more: each is counted once, the Pause's names first.
    assert gap_fact(rows, pause) == {"rows": rows[:GAP_ROWS], "count": GAP_ROWS + 4}
    # Whatever the census no longer reads, the count never drops below the Pause's own.
    assert gap_fact([rows[0]], pause) == pause
    small = gap_fact(rows[:2])
    assert gap_fact([rows[1], rows[5]], small) == {"rows": [rows[0], rows[1], rows[5]], "count": 3}
    assert gap_fact([], small) == small and gap_fact([]) == {}


def test_a_damaged_member_holds_only_its_own_tree_named(tmp_path, monkeypatch):
    from ouroboros import owner_pause
    from supervisor.continuation_admission import conflicting_writers

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    _live_root(tmp_path, workers, "solo")
    _live_root(tmp_path, workers, "neighbour")
    _damage(tmp_path, "kid", "torn", parent_task_id="solo")
    for root_id in ("solo", "neighbour"):
        _pause_and_park(tmp_path, monkeypatch, queue, workers, root_id)
    held = owner_pause.read_fence(tmp_path, "solo")
    assert held["state"] == owner_pause.FENCE_REQUESTED
    blockers = conflicting_writers(queue, "solo", drive_root=tmp_path, owner_pause_fence_id=held["fence_id"])
    assert {"kind": "member_record_unreadable", "task_id": "kid", "detail": "body_unreadable"} in blockers
    assert not [b for b in blockers if b["kind"] == "tree_census_unreadable"]
    settled = owner_pause.read_fence(tmp_path, "neighbour")
    assert settled["state"] == owner_pause.FENCE_PAUSED
    assert settled["membership_gaps"] == {"rows": [{"task_id": "kid", "reason": "body_unreadable"}], "count": 1}


def _reserve_model_request(root, task_id, root_task_id):
    from ouroboros import usage_accounting as ua

    return ua.reserve_attempt(ua.AttemptRequest(model="m", provider="test", reservation_usd=.2, drive_root=root,
                                                task_id=task_id, root_task_id=root_task_id))


@pytest.mark.parametrize("tie", ["unattributed", "parent", "custody", "model_request"])
def test_continue_is_held_only_by_a_damaged_row_tied_to_its_tree(tmp_path, monkeypatch, tie):
    from supervisor.continuation_admission import admit_continuation
    from tests.test_owner_continue import NONCE, _interrupted

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, "pred-1")
    _damage(tmp_path, "kid", "torn", **({"parent_task_id": "pred-1"} if tie == "parent" else {}))
    if tie == "custody":
        with store.open_store(tmp_path, write=True) as txn:
            txn.put_custody("run-k", {"run_id": "run-k", "task_id": "kid", "root_task_id": "pred-1"}, state="held")
    if tie == "model_request":
        _reserve_model_request(tmp_path, "kid", "pred-1")
    ack = admit_continuation("pred-1", action_nonce=NONCE)
    assert ack["ok"] is True, ack
    hold = next((task.get("_budget_pause_hold") for task in workers.PENDING
                 if task["id"] == ack["successor_task_id"]), None) or {}
    named = [b for b in hold.get("blockers") or () if b.get("kind") == "member_record_unreadable"]
    assert not [b for b in hold.get("blockers") or () if b.get("kind") == "tree_census_unreadable"]
    if tie == "unattributed":
        from ouroboros.task_results import load_task_result

        assert ack["held"] is False and not named
        # Known work continues; the record it did not wait for is named in its answer and its admission.
        gap = {"rows": [{"task_id": "kid", "reason": "body_unreadable"}], "count": 1}
        assert ack["membership_gaps"] == gap
        assert load_task_result(tmp_path, ack["successor_task_id"])["continuation_admission"]["membership_gaps"] == gap
        assert admit_continuation("pred-1", action_nonce=NONCE)["membership_gaps"] == gap  # the replay too
    else:
        assert ack["held"] is True and "membership_gaps" not in ack
        assert named == [{"kind": "member_record_unreadable", "task_id": "kid", "detail": "body_unreadable"}]


def _dispatched_model_request(root, task_id, root_task_id):
    """An open model attempt of TASK_ID under ROOT_TASK_ID, already sent: its only surviving tie."""
    from ouroboros import usage_accounting as ua

    ua.mark_dispatched(_reserve_model_request(root, task_id, root_task_id))


def _cold_tie(root, tie):
    """``kid`` with a torn legacy body, tied to ``sleeper``'s tree by TIE (or by nothing). Its model
    request was sent while its record could still be read; only then was the record damaged."""
    if tie in ("model_request", "other_root_request"):
        _dispatched_model_request(root, "kid", "sleeper" if tie == "model_request" else "elsewhere")
    _damage(root, "kid", "torn", **({"parent_task_id": "sleeper"} if tie == "child" else {}))


@pytest.mark.parametrize("tie", ["unattributed", "child", "model_request", "other_root_request"])
def test_a_cold_sleep_is_refused_only_by_its_own_damaged_member(tmp_path, tie):
    from ouroboros import model_sleep
    from tests.test_model_sleep import _ctx, _result

    _result(tmp_path, "sleeper")
    _cold_tie(tmp_path, tie)
    ctx = _ctx(tmp_path)
    chosen = model_sleep.selectors(ctx, wake_after_sec=60)
    if tie in ("child", "model_request"):  # a sent model request of the root ties it as a typed parent does
        with pytest.raises(model_sleep.ColdSleepBlocked, match="member_record_unreadable kid"):
            model_sleep.request_sleep(ctx, chosen, model_sleep.MODE_COLD)
    else:  # another root's open request ties nothing here: the read is the root's own open set
        armed = model_sleep.request_sleep(ctx, chosen, model_sleep.MODE_COLD)
        assert armed["reason"] == "sleep_armed"
        assert armed["membership_gaps"] == {"rows": [{"task_id": "kid", "reason": "body_unreadable"}], "count": 1}
        assert armed["note"].endswith(" This cold sleep does not watch 1 saved task record that could not be read "
                                      "and that nothing ties to this task (kid); whether it belongs here is unknown.")


def test_an_unreadable_model_request_set_holds_a_cold_sleep_beside_damage_it_could_tie(tmp_path, monkeypatch):
    from ouroboros import model_sleep, usage_store
    from ouroboros.usage_store import UsageLedgerCorrupt
    from tests.test_model_sleep import _ctx, _result

    _result(tmp_path, "sleeper")
    _cold_tie(tmp_path, "model_request")
    ctx = _ctx(tmp_path)
    chosen = model_sleep.selectors(ctx, wake_after_sec=60)

    def unreadable(*_a, **_k):
        raise UsageLedgerCorrupt("usage store unreadable: torn page")

    monkeypatch.setattr(usage_store, "read", unreadable)
    with pytest.raises(model_sleep.ColdSleepBlocked) as refused:
        model_sleep.request_sleep(ctx, chosen, model_sleep.MODE_COLD)
    kinds = {blocker["kind"] for blocker in refused.value.blockers}
    assert "attempt_custody_unreadable" in kinds, "unknown is never an empty request set"
    assert {"kind": "member_record_unreadable", "detail": "kid: body_unreadable"} in refused.value.blockers
    assert getattr(ctx, "_model_sleep", None) is None


@pytest.mark.parametrize("source", ["model_requests", "custody"])
def test_a_census_names_no_gap_while_a_link_source_is_unreadable(tmp_path, monkeypatch, source):
    """Unknown is never "nothing ties it": the row holds, named, beside the unreadable source."""
    from ouroboros import delegate_custody_current, owner_pause
    from supervisor.continuation_admission import conflicting_writers

    queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
    _tree(tmp_path)
    _damage(tmp_path, "stray", "torn")
    gaps = []
    conflicting_writers(queue, "root-1", drive_root=tmp_path, gaps=gaps)
    assert gaps == [{"task_id": "stray", "reason": "body_unreadable"}], "every source answered: a named gap"

    def unreadable(*_a, **_k):
        raise OSError("link source unreadable")

    if source == "model_requests":
        monkeypatch.setattr(owner_pause, "tree_model_requests", unreadable)
    else:
        monkeypatch.setattr(delegate_custody_current, "tree_task_ids", unreadable)
    gaps = []
    blockers = conflicting_writers(queue, "root-1", drive_root=tmp_path, gaps=gaps)
    assert gaps == []
    assert {"kind": "member_record_unreadable", "task_id": "stray", "detail": "body_unreadable"} in blockers
    unread = "attempt_custody_unreadable" if source == "model_requests" else "tree_census_unreadable"
    assert any(blocker["kind"] == unread for blocker in blockers)


@pytest.mark.parametrize("tie", ["model_request", "unattributed"])
def test_the_round_boundary_refuses_a_cold_sleep_its_tree_now_ties(tmp_path, monkeypatch, tie):
    """``enter_cold_sleep`` re-reads the same ties after arming: the model is told, or the sleep parks."""
    from ouroboros import budget_pause, model_sleep
    from tests._budget_pause_exact_helpers import _fast_hold, _loop_ctx, _quiet_external
    from tests.test_model_sleep import _result

    _result(tmp_path, "sleeper")
    _cold_tie(tmp_path, tie)
    ctx, limit_ctx = _loop_ctx(tmp_path, "sleeper")
    _fast_hold(monkeypatch, budget_pause)
    _quiet_external(monkeypatch, budget_pause)
    ctx._model_sleep = {"sleep_id": "s1", "mode": "cold", **model_sleep.selectors(ctx, wake_after_sec=60)}
    if tie == "model_request":
        budget_pause.enter_cold_sleep(limit_ctx)
        assert ctx._model_sleep is None
        assert "Your cold sleep was not taken" in limit_ctx.messages[-1]["content"]
        assert "member_record_unreadable kid: body_unreadable" in limit_ctx.messages[-1]["content"]
    else:  # nothing ties it: the final checkpoint passes too and the sleep parks
        try:
            with pytest.raises(budget_pause.BudgetPauseRequested) as parked:
                budget_pause.enter_cold_sleep(limit_ctx)
        finally:
            budget_pause.end_dispatch_fence("sleeper")
        assert parked.value.pause["reason"] == "sleep"


def test_the_final_checkpoint_holds_on_a_tie_sent_after_the_round_boundary_until_it_settles(tmp_path, monkeypatch):
    from ouroboros import budget_pause, model_sleep, usage_accounting as ua
    from tests._budget_pause_exact_helpers import _loop_ctx, _quiet_external
    from tests.test_model_sleep import _result

    _result(tmp_path, "sleeper")
    ctx, limit_ctx = _loop_ctx(tmp_path, "sleeper")
    _quiet_external(monkeypatch, budget_pause)
    ctx._model_sleep = {"sleep_id": "s1", "mode": "cold", **model_sleep.selectors(ctx, wake_after_sec=60)}
    original, census, holds, sent = model_sleep.cold_blockers, [], [], []

    def census_then_tie(source, **kwargs):
        census.append(original(source, **kwargs))
        if len(census) == 1:  # after the round boundary: kid's request is sent, then its record is torn
            sent.append(_reserve_model_request(tmp_path, "kid", "sleeper"))
            ua.mark_dispatched(sent[0])
            _damage(tmp_path, "kid", "torn")
        return census[-1]

    def settle(_seconds):  # the request ends; the torn record is then tied to nothing
        holds.append(dict(limit_ctx.accumulated_usage["budget_pause_hold"]))
        assert len(holds) == 1, "a settled request lets the existing retry park"
        ua.settle_attempt(sent[0], {"prompt_tokens": 1, "completion_tokens": 1}, cost_usd=0.0, cost_final=True)

    monkeypatch.setattr(model_sleep, "cold_blockers", census_then_tie)
    monkeypatch.setattr(budget_pause.time, "sleep", settle)
    monkeypatch.setattr(budget_pause, "_hold_control_reason", lambda _ctx: "")
    try:
        with pytest.raises(budget_pause.BudgetPauseRequested):
            budget_pause.enter_cold_sleep(limit_ctx)
    finally:
        budget_pause.end_dispatch_fence("sleeper")
    assert census[0] == []
    assert holds[0]["hold_reason"] == budget_pause.HOLD_COLD_WRITERS_UNSETTLED
    assert "member_record_unreadable" in holds[0]["error"] and "kid: body_unreadable" in holds[0]["error"]
    assert census[-1] == [], "once nothing ties it, the torn record is no writer"


# --- the cascade, both modes --------------------------------------------------------------------

@pytest.mark.parametrize("strict", [False, True], ids=["summary", "wrap_up"])
def test_the_cascade_keeps_durable_children_beside_an_unattributed_damaged_row(tmp_path, strict):
    from supervisor.terminal_cascade import _cascade_descendant_rows

    _tree(tmp_path)
    _damage(tmp_path, "stray", "torn")
    # No queue snapshot holds the children: only the durable census names them.
    assert _cascade_descendant_rows(tmp_path, "root-1", strict=strict) == {"child": "running", "legacy": "completed"}
    gaps = []
    _cascade_descendant_rows(tmp_path, "root-1", strict=strict, gaps=gaps)
    assert gaps == ["unresolved_task_rows: stray (body_unreadable)"]


def test_a_mid_tree_cascade_reaches_a_damaged_descendant_by_ancestry(tmp_path):
    from supervisor.terminal_cascade import _cascade_descendant_rows

    _tree(tmp_path)
    _damage(tmp_path, "grand", "torn", parent_task_id="child", status="failed")
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "state/queue_snapshot.json").write_text(json.dumps({
        "pending": [{"id": "queued-kid", "parent_task_id": "grand"}],
        "running": [{"id": "legacy", "task": {"id": "legacy", "parent_task_id": "child"}}]}), encoding="utf-8")
    gaps = []
    rows = _cascade_descendant_rows(tmp_path, "child", strict=True, gaps=gaps)
    # The damaged row is reached through its typed parent; the queue bridge beyond it too; a
    # durable row keeps its own facts over the queue's. Its membership is proven, its own record
    # is not: the Stop receipt still names it.
    assert rows == {"grand": "failed", "queued-kid": "scheduled"}
    assert gaps == ["unresolved_task_rows: grand (body_unreadable)"]
    census = task_store.tree_facts(tmp_path, "child", descendants=True)
    assert "grand" in census.members
    assert census.unresolved == {"grand": {"reason": "body_unreadable", "linked": True}}


def test_the_stop_targets_own_damaged_record_is_no_descendant_gap(tmp_path):
    from supervisor.terminal_cascade import _cascade_descendant_rows

    _tree(tmp_path)
    _damage(tmp_path, "root-1", "torn")
    gaps = []
    assert _cascade_descendant_rows(tmp_path, "root-1", gaps=gaps) == {"child": "running", "legacy": "completed"}
    assert gaps == []


def test_a_known_members_missing_body_holds_its_tree_by_name_not_the_whole_census(tmp_path, monkeypatch):
    from ouroboros.owner_pause import tree_member_results
    from supervisor.continuation_admission import conflicting_writers

    queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
    _tree(tmp_path)
    with store.open_store(tmp_path, write=True) as txn:  # a derived member whose body row is gone
        _task(txn, "kid", parent_task_id="root-1", root_task_id="root-1", status="completed")
        txn.conn.execute("DELETE FROM task_bodies WHERE task_id='kid'")
    read = tree_member_results(tmp_path, "root-1")
    assert set(read.members) == {"root-1", "child", "legacy"}
    assert read.unresolved == {"kid": {"reason": "body_missing", "linked": True}}
    blockers = conflicting_writers(queue, "root-1", drive_root=tmp_path)
    assert {"kind": "member_record_unreadable", "task_id": "kid", "detail": "body_missing"} in blockers
    assert not [b for b in blockers if b["kind"] == "tree_census_unreadable"]


@pytest.mark.parametrize("damage", ["unattributed", "descendant", "none"])
def test_the_normal_stop_summary_says_its_gap_and_records_it(tmp_path, monkeypatch, damage):
    from ouroboros.task_results import STATUS_CANCELLED, load_task_result, write_task_result
    from supervisor import terminal_delivery as delivery, workers

    sent = []
    monkeypatch.setattr(workers, "get_event_q", lambda *, create=True: SimpleNamespace(put=sent.append),
                        raising=False)
    write_task_result(tmp_path, "root-1", STATUS_CANCELLED, chat_id=9)
    write_task_result(tmp_path, "child", STATUS_CANCELLED, parent_task_id="root-1", root_task_id="root-1")
    if damage == "unattributed":
        _damage(tmp_path, "stray", "torn")
    if damage == "descendant":  # reached by its typed parent: listed, and its unreadable record named
        _damage(tmp_path, "grand", "torn", parent_task_id="child", status="cancelled")
    assert delivery.deliver_cascade_summary(tmp_path, "root-1", {"id": "root-1", "chat_id": 9}, {})
    (event,) = [e for e in sent if e.get("type") == "send_message"]
    named = {"unattributed": "stray", "descendant": "grand"}.get(damage)
    said = "Some saved task records could not be read; this summary may be incomplete."
    assert (said in event["text"]) is bool(named)
    receipt = load_task_result(tmp_path, "root-1")["cancel_receipt"]
    assert receipt.get("membership_gap") == (f"unresolved_task_rows: {named} (body_unreadable)" if named else None)
    assert {row["task_id"] for row in receipt["children"]} == {"child", *(["grand"] if damage == "descendant" else [])}
    # Both builds carried the gap: the owed row is the very text that was sent.
    assert delivery._read_registry(tmp_path)["pending"][event["delivery_id"]]["text"] == event["text"]


def test_wrap_up_replay_keeps_its_durable_children_and_names_the_gap(tmp_path, monkeypatch):
    from supervisor import terminal_delivery as delivery
    from tests.test_owner_stop_retained_work import EVENT_FIELD, _child, _collect, _wrap_up
    from ouroboros.task_results import write_task_result

    _wrap_up(tmp_path)
    write_task_result(tmp_path, "parent1", "cancelled", chat_id=17)
    _child(tmp_path, result="kept through the replay")
    _collect(tmp_path, monkeypatch)
    _damage(tmp_path, "stray", "foreign")
    queued = []
    monkeypatch.setattr(delivery, "deliver_unreviewed_salvage", lambda *_a, **_k: None)
    monkeypatch.setattr(delivery, "register_pending_delivery", lambda _root, event: queued.append(event) or True)
    # A watchdog replay after the children settled: no sweep outcomes, only the durable tree.
    assert delivery.deliver_cascade_summary(tmp_path, "parent1", {"chat_id": 17}, {})
    owner = queued[0][EVENT_FIELD]["text"]
    assert "Subtask kid (Stopped)" in owner
    assert "could not be read; this list may be incomplete" in owner
    assert "this summary may be incomplete" not in queued[0]["text"], "Wrap up's own row names it, once"
    from ouroboros.task_results import load_task_result

    receipt = load_task_result(tmp_path, "parent1")["delegate_terminal_reconciliation"]["owner_stop_children"]
    assert receipt["gap"] == "unresolved_task_rows: stray (identity_mismatch)"


def test_the_wrap_up_projection_keeps_children_and_names_the_gap(tmp_path):
    from supervisor.owner_stop import _child_result_projection

    _tree(tmp_path)
    _damage(tmp_path, "stray", "array")
    text = _child_result_projection(SimpleNamespace(DRIVE_ROOT=tmp_path), "root-1")
    assert "child (running)" in text and "legacy (completed)" in text
    assert "Descendant membership unknown: unresolved_task_rows: stray (body_not_object)" in text


# --- the hot-facts job ---------------------------------------------------------------------------

def test_the_job_waits_on_another_tasks_body_and_never_publishes_its_root(tmp_path):
    _tree(tmp_path)
    _damage(tmp_path, "stray", "foreign")  # its body names root-1
    with store.open_store(tmp_path, write=True) as txn:
        _old_task(txn, "odd", parent_task_id="root-1", root_task_id="root-1", status="")
    before = task_store.tree_facts(tmp_path, "root-1")
    with pytest.raises(ValueError, match="identity_mismatch"):
        hot.run_hot_facts(tmp_path)
    job = maintenance_jobs.job(tmp_path, hot.JOB)
    assert job["state"] == "failed" and job["waiting_on"]["id"] == "stray"
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.row("stray", "root_task_id") is None, "another task's root was published"
        assert txn.row("legacy", "root_task_id", "control_valid") == ("root-1", 1)
        assert txn.row("odd", "root_task_id", "control_valid") == ("root-1", 0)
    after = task_store.tree_facts(tmp_path, "root-1")
    assert before == after and set(after.members) == {"root-1", "child", "legacy", "odd"}
    # An older writer re-NULLs a derived row: its own body still places it.
    with store.open_store(tmp_path, write=True) as txn:
        _old_task(txn, "legacy", parent_task_id="root-1", metadata={"root_task_id": "root-1"}, status="completed")
        assert txn.row("legacy", "root_task_id") is None
    assert task_store.tree_facts(tmp_path, "root-1") == after
    # The damaged row's own writer repairs it: the job resumes and the gap closes.
    with store.open_store(tmp_path, write=True) as txn:
        _task(txn, "stray", status="completed")
    assert hot.run_hot_facts(tmp_path)["state"] == "done"
    repaired = task_store.tree_facts(tmp_path, "root-1")
    assert repaired.unresolved == {} and set(repaired.members) == {"root-1", "child", "legacy", "odd"}


# --- cost ---------------------------------------------------------------------------------------

def _measure(root, monkeypatch, call):
    """Body bytes decoded, unrelated ones, and legacy ones decoded while a store read was open."""
    counts = {"decoded": 0, "unrelated": 0, "under_store": 0}
    real_loads, real_open, held = json.loads, task_store.open_store, []

    @contextlib.contextmanager
    def tracked(*args, **kwargs):
        with real_open(*args, **kwargs) as txn:
            held.append(True)
            try:
                yield txn
            finally:
                held.pop()

    def loads(raw, *args, **kwargs):
        blob = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
        counts["decoded"] += len(blob)
        counts["unrelated"] += len(blob) if b"UNRELATED_BODY" in blob else 0
        counts["under_store"] += len(blob) if held and b"LEGACY_BODY" in blob else 0
        return real_loads(raw, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(task_store, "open_store", tracked)
        patch.setattr(json, "loads", loads)
        result = call(root)
    return result, counts


def test_the_census_cost_does_not_grow_with_unrelated_history(tmp_path, monkeypatch):
    from ouroboros.owner_pause import tree_member_results

    observed = []
    for size in (20, 2000):
        root = tmp_path / f"n{size}"
        _tree(root)
        _damage(root, "stray", "torn")
        with store.open_store(root, write=True) as txn:
            _old_task(txn, "legacy", parent_task_id="root-1", metadata={"root_task_id": "root-1"},
                      status="completed", result="LEGACY_BODY")
            for index in range(size):
                _task(txn, f"foreign-{index:05}", status="completed", result="UNRELATED_BODY" * 100)
        census, counts = _measure(root, monkeypatch, lambda r: tree_member_results(r, "root-1"))
        assert set(census.members) == {"root-1", "child", "legacy"} and set(census.unresolved) == {"stray"}
        observed.append(counts)
    assert observed[0] == observed[1]
    assert observed[0]["unrelated"] == 0 and observed[0]["under_store"] == 0
