"""A confirmed pause owes one local System row, independently of future Resume.

The duty reads the ``pause_notices`` obligation set the pause writer publishes;
it never enumerates results and never reads chat history (invariant 10).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros import obligations, pause_notices
from ouroboros.budget_pause import set_budget_pause
from ouroboros.startup_migrations import prepare_startup_state
from ouroboros.task_results import load_task_result, write_task_result
from supervisor import message_bus

pytestmark = pytest.mark.serial


@pytest.fixture
def notices(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    sent = []
    monkeypatch.setattr(message_bus, "DATA_DIR", root)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 1})
    bridge = SimpleNamespace(send_message=lambda chat, text, **kw: sent.append((chat, text, kw)))
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(message_bus, "_advance_project_visible_revision", lambda *_: None)
    prepare_startup_state(root)  # the lifecycle import creates the sets, as boot does
    write_task_result(root, "task", "running", chat_id=1)
    return root, sent


def _pause(root, episode="one", reason="budget", state="paused"):
    return set_budget_pause(root, "task", {"pause_id": episode, "state": state, "reason": reason})


def _rows(root):
    path = root / "logs" / "chat.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _owing(root):
    return set(obligations.members(root, "pause_notices"))


def test_confirmed_pause_persists_system_row_and_repeated_confirmation_does_not_duplicate(notices):
    root, sent = notices
    _pause(root, state="pausing")
    assert not _owing(root)  # nothing confirmed, nothing owed
    pause_notices.reconcile_pause_notices(root)
    assert _rows(root) == []
    _pause(root)
    assert _owing(root) == {"task"}
    pause_notices.reconcile_pause_notices(root)
    _pause(root)
    pause_notices.reconcile_pause_notices(root)
    assert len(sent) == len(_rows(root)) == 1
    row = _rows(root)[0]
    assert row["direction"] == "system" and row["type"] == "task_pause_notice"
    assert row["task_id"] == "task" and row["chat_id"] == 1
    assert row["card_row_id"] == "pause:task:one"
    assert "pause_notices" not in load_task_result(root, "task")
    assert load_task_result(root, "task")["status"] == "running"
    # Discharged: the membership and the transient receipt are both gone.
    assert not _owing(root) and not obligations.members(root, "pause_notice_receipts")


def test_resume_and_second_pause_preserve_both_owed_episodes(notices):
    root, sent = notices
    _pause(root)
    _pause(root, state="resumed")
    _pause(root, episode="two")
    pending = load_task_result(root, "task")["pause_notices"]
    assert len(pending) == 2
    # The duty holds no process memory: the durable membership alone finds the task.
    assert _owing(root) == {"task"}
    pause_notices.reconcile_pause_notices(root)
    assert len(sent) == 2
    assert {r["card_row_id"] for r in _rows(root)} == {"pause:task:one", "pause:task:two"}
    assert load_task_result(root, "task")["budget_pause"]["pause_id"] == "two"
    assert not _owing(root)


def test_chat_append_before_ack_is_recovered_without_a_second_send(notices, monkeypatch):
    root, sent = notices
    _pause(root)
    original = pause_notices._acknowledge
    monkeypatch.setattr(pause_notices, "_acknowledge", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    pause_notices.reconcile_pause_notices(root)
    assert len(_rows(root)) == 1 and load_task_result(root, "task")["pause_notices"]
    assert obligations.members(root, "pause_notice_receipts") and _owing(root) == {"task"}
    monkeypatch.setattr(pause_notices, "_acknowledge", original)
    pause_notices.reconcile_pause_notices(root)
    assert len(_rows(root)) == len(sent) == 1
    assert "pause_notices" not in load_task_result(root, "task")
    assert not obligations.members(root, "pause_notice_receipts")


def test_a_send_that_fails_after_its_row_landed_is_not_repeated(notices, monkeypatch):
    """A live transport error after the append: the receipt, not the failed call, decides."""
    root, sent = notices
    _pause(root)
    real_send = message_bus.send_with_budget

    def lands_then_raises(*args, **kwargs):
        real_send(*args, **kwargs)
        raise OSError("transport down")

    monkeypatch.setattr(message_bus, "send_with_budget", lands_then_raises)
    pause_notices.reconcile_pause_notices(root)
    assert len(_rows(root)) == 1 and load_task_result(root, "task")["pause_notices"]
    pause_notices.reconcile_pause_notices(root)  # still failing: no second row, and now acknowledged
    assert len(_rows(root)) == 1 and "pause_notices" not in load_task_result(root, "task")


def test_failed_required_append_remains_owed_and_retries(notices, monkeypatch):
    root, sent = notices
    _pause(root)
    original = message_bus.log_chat
    monkeypatch.setattr(message_bus, "log_chat", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    pause_notices.reconcile_pause_notices(root)
    assert not sent and not _rows(root) and load_task_result(root, "task")["pause_notices"]
    assert _owing(root) == {"task"}
    monkeypatch.setattr(message_bus, "log_chat", original)
    pause_notices.reconcile_pause_notices(root)
    assert len(sent) == 1 and "pause_notices" not in load_task_result(root, "task")


def test_owner_members_do_not_announce_tree_confirmation(notices):
    from ouroboros.owner_pause import install_fence, set_fence_state
    root, sent = notices
    _pause(root, reason="owner")
    pause_notices.reconcile_pause_notices(root)
    assert not sent
    fence, _ = install_fence(root, "task", request_id="request")
    set_fence_state(root, "task", fence_id=fence["fence_id"], state="paused", expected_state="requested")
    assert _owing(root) == {"task"}
    pause_notices.reconcile_pause_notices(root)
    assert len(sent) == 1 and "owner's Pause" in sent[0][1]


def test_first_dispatch_budget_pause_publishes_its_membership(notices):
    """The third pause writer goes through ``write_task_result`` with a projector."""
    root, sent = notices
    write_task_result(root, "task", "scheduled",
                      _field_projector=lambda current, incoming: {
                          **incoming, **pause_notices.notice_fields(current, root, "task", "episode", "budget")})
    assert _owing(root) == {"task"}
    pause_notices.reconcile_pause_notices(root)
    assert len(sent) == 1 and not _owing(root)


def test_closed_generation_does_not_publish_or_acknowledge(notices):
    root, sent = notices
    _pause(root)
    pause_notices.reconcile_pause_notices(root, stop_requested=lambda: True)
    assert not sent and load_task_result(root, "task")["pause_notices"]
    pause_notices.reconcile_pause_notices(root, stop_requested=lambda: False)
    assert len(sent) == 1


def test_missing_address_never_falls_back_to_owner_main(notices):
    root, sent = notices
    write_task_result(root, "task", "running", chat_id=None)
    _pause(root)
    pause_notices.reconcile_pause_notices(root)
    assert not sent and load_task_result(root, "task")["pause_notices"]
    assert _owing(root) == {"task"}  # no destination keeps the debt


def test_worker_replica_cannot_resurrect_an_acknowledged_notice():
    from ouroboros.post_task_checkpoint import project_replica_task_result_fields
    overlay = project_replica_task_result_fields(
        {"task_id": "task", "status": "running"},
        {"pause_notices": {"pause:task:old": {"reason": "budget"}}},
    )
    assert "pause_notices" not in overlay


def test_a_pass_reads_only_owing_tasks_among_completed_history(notices, monkeypatch):
    """Two-sided: 300 completed results and a long chat history cost nothing; the owing task is served."""
    import ouroboros.task_result_facts as facts
    import ouroboros.task_results as results
    import ouroboros.utils as utils
    root, sent = notices
    for index in range(300):
        write_task_result(root, f"done{index}", "completed", chat_id=1, result="ok")
    (root / "logs").mkdir(exist_ok=True)
    with (root / "logs" / "chat.jsonl").open("a", encoding="utf-8") as handle:
        for index in range(500):
            handle.write(json.dumps({"direction": "in", "chat_id": 1, "text": f"old {index}"}) + "\n")
    monkeypatch.setattr(facts, "raw_result_facts", lambda *a, **k: pytest.fail("results enumerated"))
    monkeypatch.setattr(utils, "iter_jsonl_chain_objects", lambda *a, **k: pytest.fail("chat history read"))
    monkeypatch.setattr(results, "list_task_results", lambda *a, **k: pytest.fail("results listed"))
    loaded = []
    real_load = results.load_task_result
    monkeypatch.setattr(results, "load_task_result", lambda r, tid, **k: loaded.append(tid) or real_load(r, tid, **k))
    pause_notices.reconcile_pause_notices(root)  # nothing owed: no result is opened at all
    assert not loaded and not sent
    _pause(root)
    pause_notices.reconcile_pause_notices(root)
    assert len(sent) == 1 and set(loaded) == {"task"}


def test_a_crash_before_the_result_write_leaves_a_candidate_that_is_retired(notices):
    """Membership is published first: a crash before the result write is an extra candidate, never a loss."""
    root, sent = notices
    obligations.add(root, "pause_notices", "task", {"task_id": "task"})
    pause_notices.reconcile_pause_notices(root)
    assert not sent and not _owing(root)
    obligations.add(root, "pause_notices", "gone", {"task_id": "gone"})  # its result never landed
    pause_notices.reconcile_pause_notices(root)
    assert not _owing(root)


def test_retiring_a_stale_candidate_cannot_erase_a_pause_confirmed_meanwhile(notices, monkeypatch):
    import ouroboros.task_results as results
    root, sent = notices
    obligations.add(root, "pause_notices", "task", {"task_id": "task"})  # stale: the result owes nothing yet
    real_load = results.load_task_result
    first = True

    def read_then_new_pause(*args, **kwargs):
        nonlocal first
        old = real_load(*args, **kwargs)
        if first:
            first = False
            _pause(root)  # commits after the maintenance read, before the stale-candidate retirement
        return old

    monkeypatch.setattr(results, "load_task_result", read_then_new_pause)
    pause_notices.reconcile_pause_notices(root)
    assert not sent and real_load(root, "task")["pause_notices"]
    assert _owing(root) == {"task"}  # retirement re-read the result under its lock
    pause_notices.reconcile_pause_notices(root)
    assert len(sent) == 1 and not real_load(root, "task").get("pause_notices")


def test_unavailable_sets_send_nothing_and_scan_nothing(notices, monkeypatch):
    import ouroboros.task_results as results
    root, sent = notices
    _pause(root)
    obligations.path(root, "pause_notices").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(results, "load_task_result", lambda *a, **k: pytest.fail("a result was read"))
    pause_notices.reconcile_pause_notices(root)
    assert not sent


def test_the_lifecycle_import_seeds_pauses_owed_by_an_older_body(tmp_path, monkeypatch):
    """An install upgraded with a pause already owed: the one-time import publishes it."""
    root = tmp_path / "data"
    (root / "task_results").mkdir(parents=True)
    monkeypatch.setattr(message_bus, "DATA_DIR", root)
    write_task_result(root, "old", "running", chat_id=1)
    path = root / "task_results" / "old.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["pause_notices"] = {"pause:old:one": {"reason": "budget", "confirmed_at": "2026-10-01T00:00:00+00:00"}}
    path.write_text(json.dumps(row), encoding="utf-8")  # as a body without the sets wrote it
    for name in obligations.BOOT_SETS:  # no set exists yet on that install
        obligations.path(root, name).unlink(missing_ok=True)
    prepare_startup_state(root, rebuild=True)
    assert set(obligations.members(root, "pause_notices")) == {"old"}
