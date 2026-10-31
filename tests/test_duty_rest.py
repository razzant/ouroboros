"""A duty's member that cannot progress rests (``obligations.DutyRest``): its typed outcome holds
while every recorded input holds, the pass selects it without reading a body or writing, and the
first change of an input (a new fact on the task, an external stamp, its own retry instant) puts
it back exactly once. A rest is a kept fact, never debt and never a retirement of the debt."""
import json

import pytest

from ouroboros import obligations as o, store, task_store
from ouroboros.task_results import write_task_result

SET = "nonterminal"


def revision(root):
    with store.open_store(root, write=False) as txn:
        return txn.revision()


def marks(root):
    with store.open_store(root, write=False) as txn:
        return txn.set_rows(o.REST_SET)


@pytest.fixture
def root(tmp_path, monkeypatch):
    for tid in ("a", "b"):
        write_task_result(tmp_path, tid, "running")
    # A pass reads members, signatures and rests; it never decodes a task body.
    monkeypatch.setattr(task_store, "read_document", lambda *a, **k: pytest.fail("a rest pass read a body"))
    return tmp_path


def test_a_member_rests_until_a_new_fact_puts_it_back_exactly_once(root, monkeypatch):
    first = o.DutyRest(root, "probe", name=SET)
    assert first.examine == ["a", "b"] and not first.resting  # no rest: every member is examined
    first.wait("a", "owner_resume")
    first.commit()
    assert set(marks(root)) == {"probe:a"} and marks(root)["probe:a"]["waits_on"] == ["owner_resume"]

    before = revision(root)
    second = o.DutyRest(root, "probe", name=SET)
    assert second.examine == ["b"] and set(second.resting) == {"a"}
    second.commit()
    assert revision(root) == before  # a quiet pass writes nothing

    monkeypatch.undo()
    write_task_result(root, "a", "running", note="a new fact")  # any write of the task moves its signature
    third = o.DutyRest(root, "probe", name=SET)
    assert third.examine == ["a", "b"]  # back exactly once ...
    third.wait("a", "owner_resume")
    third.commit()
    assert o.DutyRest(root, "probe", name=SET).examine == ["b"]  # ... and at rest again


def test_an_external_stamp_or_the_retry_instant_wakes_a_rest(root):
    rest = o.DutyRest(root, "probe", name=SET, inputs={"registry": [1, 2, 3]})
    rest.wait("a", "terminal_delivery", inputs=("registry",))
    rest.wait("b", "delivery_retry", due_at="2999-01-01T00:00:00+00:00")
    rest.commit()
    assert o.DutyRest(root, "probe", name=SET, inputs={"registry": [1, 2, 3]}).examine == []
    assert o.DutyRest(root, "probe", name=SET, inputs={"registry": [1, 2, 4]}).examine == ["a"]
    assert o.DutyRest(root, "probe", name=SET, inputs={}).examine == ["a"]  # an unknown stamp proves nothing
    later = o.DutyRest(root, "probe", name=SET, inputs={"registry": [1, 2, 3]}, now="2999-01-01T00:00:01+00:00")
    assert later.examine == ["b"]  # its own persisted retry instant arrived


def test_a_members_own_input_wakes_only_that_member(root):
    rest = o.DutyRest(root, "probe", {"a": {"usage_revision": 1}, "b": {"usage_revision": 7}})
    rest.wait("a", "post_task_synthesis_open")
    rest.wait("b", "post_task_synthesis_open")
    rest.commit()
    assert marks(root)["probe:a"]["inputs"] == {"usage_revision": 1}
    assert o.DutyRest(root, "probe", {"a": {"usage_revision": 2}, "b": {"usage_revision": 7}}).examine == ["a"]


def test_an_unreadable_store_rests_nothing_and_examines_every_candidate(root, monkeypatch):
    rest = o.DutyRest(root, "probe", {"a": {}})
    rest.wait("a", "owner_resume")
    rest.commit()

    def busy(*_a, **_k):
        raise store.StoreBusy("current store busy", reason="contention")

    monkeypatch.setattr(store, "open_store", busy)
    blind = o.DutyRest(root, "probe", {"a": {}, "b": {}})
    assert blind.examine == ["a", "b"] and not blind.resting
    blind.wait("a", "owner_resume")
    blind.commit()  # nothing to write: no signature was read
    with pytest.raises(store.StoreBusy):
        o.DutyRest(root, "probe", name=SET)  # a set whose members are unknown cannot be passed over


def test_wake_examines_a_rest_without_dropping_it(root):
    rest = o.DutyRest(root, "probe", name=SET)
    rest.wait("a", "owner_resume")
    rest.commit()
    woken = o.DutyRest(root, "probe", name=SET, wake=("owner_resume",))
    assert woken.examine == ["a", "b"]
    woken.wait("a", "owner_resume")
    before = revision(root)
    woken.commit()
    assert revision(root) == before and set(marks(root)) == {"probe:a"}


def test_a_departed_member_drops_its_rest_once(root, monkeypatch):
    rest = o.DutyRest(root, "probe", name=SET)
    rest.wait("a", "owner_resume")
    rest.wait("b", "owner_resume")
    rest.commit()
    monkeypatch.undo()
    write_task_result(root, "a", "completed", result="done")  # leaves the set
    o.DutyRest(root, "probe", name=SET).commit()
    assert set(marks(root)) == {"probe:b"}
    before = revision(root)
    o.DutyRest(root, "probe", name=SET).commit()
    assert revision(root) == before


def test_a_write_racing_the_pass_is_never_rested_over(root, monkeypatch):
    rest = o.DutyRest(root, "probe", name=SET)
    monkeypatch.undo()
    write_task_result(root, "a", "running", note="landed after the selection")
    rest.wait("a", "owner_resume")
    rest.commit()
    assert "probe:a" not in marks(root)
    assert o.DutyRest(root, "probe", name=SET).examine == ["a", "b"]


def test_a_rest_is_a_kept_fact_never_debt(root):
    rest = o.DutyRest(root, "probe", candidates={"a": {}})
    rest.wait("a", "owner_resume")
    rest.commit()
    with store.open_store(root, write=False) as txn:
        assert "probe:a" in txn.set_rows(o.REST_SET)  # a row of its own kept set ...
    assert o.REST_SET not in o.BOOT_SETS  # ... that no rebuild derives or erases


# ---------------------------------------------------------------- question delivery (20 s duty)

import asyncio  # noqa: E402
import datetime  # noqa: E402

from ouroboros import owner_quiz, task_results  # noqa: E402
from ouroboros.gateway.task_decision import answer_decision, recover_question_deliveries  # noqa: E402
from supervisor import message_bus  # noqa: E402
from tests.test_quiz_history_evidence import runtime as runtime  # noqa: E402,F401

# A finished task's unanswered card as the live install holds them (no typed delivery debt): a
# current question for Main (answerable late, owner decision В17a=A), nothing for a retry to deliver.
EXPIRED = {"quiz_id": "q", "question": "Old choice?", "options": ["A", "B"], "state": "expired_terminal",
           "asked_at": "2026-09-24T00:00:00+00:00", "reconciled_at": "2026-09-24T01:00:00+00:00",
           "chat_id": 1, "publication_chat_id": 1, "publication_evidence": "history"}


def recover(runtime):
    return asyncio.run(recover_question_deliveries(runtime.root))


def counted_loads(monkeypatch):
    seen, real = [], task_results.load_task_result
    monkeypatch.setattr(task_results, "load_task_result",
                        lambda root, tid, *a, **k: seen.append(tid) or real(root, tid, *a, **k))
    return seen


def test_question_duty_rests_waiting_questions_until_a_new_fact(runtime, monkeypatch):
    for tid in ("old-a", "old-b"):
        task_results.write_task_result(runtime.root, tid, "completed", chat_id=1, owner_quiz={"q": dict(EXPIRED)})
    loads = counted_loads(monkeypatch)
    assert recover(runtime)["attempted"] == 0 and sorted(loads) == ["old-a", "old-b"]
    assert marks(runtime.root)["question_delivery:old-a"]["waits_on"] == ["owner_answer"]
    before, loads[:] = revision(runtime.root), []
    assert recover(runtime)["attempted"] == 0 and loads == [] and revision(runtime.root) == before
    assert {"old-a", "old-b"} <= set(o.members(runtime.root, "owner_questions"))  # Main still sees both
    task_results.write_task_result(runtime.root, "old-a", "completed", note="a new fact")
    recover(runtime)
    assert loads == ["old-a"]  # back exactly once
    loads[:] = []
    recover(runtime)
    assert loads == []


def test_a_late_answer_wakes_its_question_and_the_retry_instant_wakes_the_owed_delivery(runtime, monkeypatch):
    task_results.write_task_result(runtime.root, "old-a", "completed", chat_id=1, owner_quiz={"q": dict(EXPIRED)})
    recover(runtime)
    real_log = message_bus.log_chat

    def unavailable(*args, **kwargs):
        if kwargs.get("record_type") == "quiz_answer":
            raise OSError("evidence unavailable")
        return real_log(*args, **kwargs)

    monkeypatch.setattr(message_bus, "log_chat", unavailable)
    status, reply = asyncio.run(answer_decision(runtime.root, {
        "request_id": "late-owner", "decision_id": "quiz:old-a:q", "option_index": 1}))
    assert status == 503 and reply["question_delivery"]["state"] == "owed"
    loads = counted_loads(monkeypatch)
    assert recover(runtime)["attempted"] == 0 and loads == ["old-a"]  # the answer is a new fact; its retry waits
    rest = marks(runtime.root)["question_delivery:old-a"]
    assert rest["waits_on"] == ["quiz_history_write_failed"] and rest["due_at"]
    loads[:] = []
    assert recover(runtime)["attempted"] == 0 and loads == []
    monkeypatch.setattr(message_bus, "log_chat", real_log)
    later = (datetime.datetime.fromisoformat(rest["due_at"]) + datetime.timedelta(seconds=1)).isoformat()
    monkeypatch.setattr(owner_quiz, "utc_now_iso", lambda: later)
    assert recover(runtime)["attempted"] == 1 and loads[0] == "old-a"
    assert owner_quiz.quiz_states(runtime.root, "old-a")["q"]["delivery"]["state"] == "delivered"


# ---------------------------------------------------------------- synthesis / terminal projection

from ouroboros import agent_task_pipeline as pipeline, terminal_projection  # noqa: E402


def counted_bodies(monkeypatch):
    """Every task-body decode goes through ``Txn.body_text`` (reads and merge-writes alike)."""
    seen, real = [], store.Txn.body_text
    monkeypatch.setattr(store.Txn, "body_text", lambda self, tid: seen.append(tid) or real(self, tid))
    return seen


def stuck_roots(root):
    """A late phase paused by its owner, and a finished pooled root whose child result was
    never adopted (its drive is named, no copy-back recorded): the 16.10 fixture's two classes."""
    write_task_result(root, "paused-1", "completed", root_task_id="paused-1", result="r",
                      root_phase_checkpoint={"post_task_synthesis": "paused"})
    child = root / "state" / "headless_tasks" / "unadopted-1" / "data"
    write_task_result(root, "unadopted-1", "failed", root_task_id="unadopted-1", result="r", drive_root=str(child))
    assert {"paused-1", "unadopted-1"} <= set(o.members(root, "synthesis"))


def steady(root):
    return pipeline.recover_pending_root_post_task_synthesis(
        root, startup=False, owner_liveness=lambda _tid: False, claim_guard=None)


def test_synthesis_duty_ends_a_saved_pause_and_rests_unadopted_projections(tmp_path, monkeypatch):
    stuck_roots(tmp_path)
    bodies = counted_bodies(monkeypatch)
    steady(tmp_path)  # ends the saved Pause once; the first settle mints readiness (a write): its rest lands next pass
    steady(tmp_path)
    rests = marks(tmp_path)
    assert "synthesis_recovery:paused-1" not in rests  # nothing waits on a Resume: the Pause ended that upkeep
    assert synthesis_phase(tmp_path, "paused-1") == "degraded"
    assert rests["synthesis_recovery:unadopted-1"]["waits_on"] == ["child_files"]
    before, bodies[:] = revision(tmp_path), []
    steady(tmp_path)
    assert bodies == [] and revision(tmp_path) == before
    assert "unadopted-1" in o.members(tmp_path, "synthesis")  # the debt stays
    write_task_result(tmp_path, "unadopted-1", "failed", note="a new fact")
    bodies[:] = []
    steady(tmp_path)
    assert set(bodies) == {"unadopted-1"}  # back exactly once
    bodies[:] = []
    steady(tmp_path)
    assert bodies == []


def test_a_projection_whose_card_becomes_producible_comes_back_once_and_settles(tmp_path, monkeypatch):
    stuck_roots(tmp_path)
    steady(tmp_path)
    steady(tmp_path)
    assert marks(tmp_path)["synthesis_recovery:unadopted-1"]["waits_on"] == ["child_files"]
    child = str(tmp_path / "state" / "headless_tasks" / "unadopted-1" / "data")
    write_task_result(tmp_path, "unadopted-1", "failed", headless_child_drive_root=child,
                      child_ref_promotion={"schema_version": 1, "status": "complete"})  # the copy-back landed
    bodies = counted_bodies(monkeypatch)
    steady(tmp_path)
    assert "unadopted-1" in bodies
    settled = task_results.load_task_result(tmp_path, "unadopted-1")
    assert settled["canonical_terminal_projection"]["summary_id"] == "task-terminal:unadopted-1"
    assert "unadopted-1" not in o.members(tmp_path, "synthesis")  # the debt was paid, not forgotten
    steady(tmp_path)
    assert "synthesis_recovery:unadopted-1" not in marks(tmp_path)  # the departed member's rest is dropped
    bodies[:] = []
    steady(tmp_path)
    assert bodies == []


def test_a_rest_an_earlier_version_left_on_a_saved_pause_does_not_hide_it(tmp_path, monkeypatch):
    """A member an earlier version rested on its owner's Resume is examined anyway: the steady pass
    still ends that saved Pause once, and a projection rest beside it holds across a restart."""
    stuck_roots(tmp_path)
    legacy = o.DutyRest(tmp_path, "synthesis_recovery", name="synthesis")
    legacy.wait("paused-1", "owner_resume")
    legacy.commit()
    assert marks(tmp_path)["synthesis_recovery:paused-1"]["waits_on"] == ["owner_resume"]
    steady(tmp_path)
    assert synthesis_phase(tmp_path, "paused-1") == "degraded"
    steady(tmp_path)
    bodies = counted_bodies(monkeypatch)
    pipeline.recover_pending_root_post_task_synthesis(tmp_path)
    assert "unadopted-1" not in bodies  # a projection rest holds across a restart


def synthesis_phase(root, task_id):
    return (task_results.load_task_result(root, task_id).get("root_phase_checkpoint") or {}).get("post_task_synthesis")


def test_projection_reconcile_rests_on_its_cause_and_the_registry_wakes_only_delivery_waits(tmp_path, monkeypatch):
    stuck_roots(tmp_path)
    child = tmp_path / "state" / "headless_tasks" / "adopted-1" / "data"
    write_task_result(tmp_path, "adopted-1", "completed", root_task_id="adopted-1", result="r", drive_root=str(child),
                      headless_child_drive_root=str(child),
                      child_ref_promotion={"schema_version": 1, "status": "complete"})
    registry = tmp_path / "state" / "terminal_deliveries.json"
    registry.write_text(json.dumps({"schema_version": 1, "delivered": [], "pending": {
        "final:adopted-1:0123456789abcdef": {"task_id": "adopted-1", "chat_id": 1, "text": "answer"}}}),
        encoding="utf-8")
    terminal_projection.reconcile_terminal_projections(tmp_path)
    terminal_projection.reconcile_terminal_projections(tmp_path)
    rests = marks(tmp_path)
    assert rests["terminal_projection:unadopted-1"]["waits_on"] == ["child_files"]
    assert rests["terminal_projection:adopted-1"]["waits_on"] == ["terminal_delivery"]
    bodies = counted_bodies(monkeypatch)
    terminal_projection.reconcile_terminal_projections(tmp_path)
    assert bodies == []
    registry.write_text(json.dumps({"schema_version": 1, "delivered": [], "pending": {}}) + " ", encoding="utf-8")
    terminal_projection.reconcile_terminal_projections(tmp_path)
    assert "adopted-1" in bodies and "unadopted-1" not in bodies  # only the delivery wait woke


def test_a_projection_deferred_on_its_child_files_never_reads_the_delivery_registry(tmp_path, monkeypatch):
    stuck_roots(tmp_path)
    monkeypatch.setattr(terminal_projection, "_delivery_disposition",
                        lambda *a: pytest.fail("registry read before the row's own facts"))
    waits = {}
    terminal_projection.settle_terminal_projection(tmp_path, "unadopted-1", waits=waits)
    assert waits == {"on": "child_files"}


# ---------------------------------------------------------------- abandoned usage: cost projection (20 s duty)

from ouroboros.terminal_cost_reconciliation import reconcile_abandoned_usage  # noqa: E402
from tests.test_terminal_cost_reconciliation import attempt, dirty  # noqa: E402
from tests.test_terminal_cost_reconciliation import env as env  # noqa: E402,F401


def test_usage_duty_reads_no_unsettled_owner_and_rests_an_open_late_phase(env, monkeypatch):
    for tid in ("paused", "live"):
        write_task_result(env.root, tid, "running", result="r")
        attempt(env, tid)
    write_task_result(env.root, "paused", "completed", result="r", root_phase_checkpoint={"post_task_synthesis": "paused"})
    bodies = counted_bodies(monkeypatch)
    reconcile_abandoned_usage(env.root)
    assert "live" not in bodies and "paused" in bodies  # a running owner's projection waits unread
    assert marks(env.root)["cost_projection:paused"]["waits_on"] == ["post_task_synthesis_open"]
    assert {"paused", "live"} <= set(dirty(env))  # the debt stays
    bodies[:] = []
    before = revision(env.root)
    reconcile_abandoned_usage(env.root)
    assert bodies == [] and revision(env.root) == before
    write_task_result(env.root, "paused", "completed", note="a new fact")
    reconcile_abandoned_usage(env.root)
    assert set(bodies) == {"paused"}  # back exactly once
    bodies[:] = []
    reconcile_abandoned_usage(env.root)
    assert bodies == []
    attempt(env, "paused-helper", logical="paused")  # a new receipt on its root moves its usage revision
    reconcile_abandoned_usage(env.root)
    assert "paused" in bodies
    bodies[:] = []
    reconcile_abandoned_usage(env.root)
    assert "paused" not in bodies


# ---------------------------------------------------------------- pending drives at boot

from ouroboros.startup_task_files import _drive_stamps, recover_terminal_task_files  # noqa: E402


@pytest.mark.parametrize("wal", [False, True])
def test_boot_rests_settled_drives_that_wait_on_their_child_and_reports_them_as_found(tmp_path, monkeypatch, wal):
    if wal:  # the installer's engine on a local disk: a child's commit stays in its WAL log until a checkpoint
        import sqlite3

        monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 53, 1))
        monkeypatch.setattr(store, "_filesystem_type", lambda _path: "apfs")
    gone = tmp_path / "state" / "headless_tasks" / "gone" / "data"  # the child drive was settled away
    write_task_result(tmp_path, "gone", "failed", result="r", child_drive_root=str(gone))
    dead = tmp_path / "state" / "headless_tasks" / "dead" / "data"
    dead.mkdir(parents=True)
    write_task_result(dead, "dead", "running", result="unfinished")  # a child that never settled
    write_task_result(tmp_path, "dead", "completed", result="r", child_drive_root=str(dead))
    assert {"gone", "dead"} <= set(o.members(tmp_path, "pending_drives"))
    first = recover_terminal_task_files(tmp_path, set())
    assert first["unresolved"] == ["dead"] and first["errors"] == []
    rests = marks(tmp_path)
    assert rests["drive_recovery:gone"]["waits_on"] == ["child_drive_absent"]
    assert rests["drive_recovery:dead"]["waits_on"] == ["child_result_unsettled"]
    bodies, before = counted_bodies(monkeypatch), revision(tmp_path)
    again = recover_terminal_task_files(tmp_path, set())
    assert again["unresolved"] == ["dead"]  # reported as found, still excluded from healing
    assert bodies == [] and revision(tmp_path) == before
    stamps = _drive_stamps(tmp_path, "dead")
    write_task_result(dead, "dead", "running", note="the child store moved")
    assert _drive_stamps(tmp_path, "dead") != stamps, "a commit still in the WAL log moves the stamp too"
    bodies[:] = []  # the write's own merge read is not the recovery's
    recover_terminal_task_files(tmp_path, set())
    assert set(bodies) == {"dead"}  # back exactly once
    bodies[:] = []
    assert recover_terminal_task_files(tmp_path, set())["unresolved"] == ["dead"] and bodies == []


def test_a_retained_settlements_basis_moves_with_a_commit_still_in_the_wal_log(tmp_path, monkeypatch):
    import sqlite3

    from ouroboros.task_custody import settlement_basis

    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 53, 1))
    monkeypatch.setattr(store, "_filesystem_type", lambda _path: "apfs")
    drive = tmp_path / "task_drives" / "kept"
    drive.mkdir(parents=True)
    write_task_result(drive, "kept", "running")
    write_task_result(tmp_path, "kept", "completed", result="r")
    basis = settlement_basis(tmp_path, "kept", drive, drive)
    write_task_result(drive, "kept", "running", note="a commit the WAL log holds")
    assert settlement_basis(tmp_path, "kept", drive, drive) != basis


def test_a_wal_log_that_comes_and_goes_without_a_commit_wakes_no_rest(tmp_path, monkeypatch):
    """The log is opened empty by a reader and deleted at the last close: neither is a commit, so a
    retained verdict stays rested across sweeps; a commit still moves the basis."""
    import sqlite3

    from ouroboros.task_custody import settlement_basis
    from ouroboros.task_results import load_task_result

    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 53, 1))
    monkeypatch.setattr(store, "_filesystem_type", lambda _path: "apfs")
    drive = tmp_path / "task_drives" / "kept"
    drive.mkdir(parents=True)
    write_task_result(drive, "kept", "running")
    write_task_result(tmp_path, "kept", "completed", result="r")
    store.close_connections()  # the last close checkpoints the log into the database and deletes it
    basis = settlement_basis(tmp_path, "kept", drive, drive)
    for _sweep in range(2):
        assert load_task_result(drive, "kept")["status"] == "running"  # a reader opens an empty log
        assert settlement_basis(tmp_path, "kept", drive, drive) == basis
        store.close_connections()
        assert settlement_basis(tmp_path, "kept", drive, drive) == basis
    write_task_result(drive, "kept", "running", note="a commit")
    assert settlement_basis(tmp_path, "kept", drive, drive) != basis
