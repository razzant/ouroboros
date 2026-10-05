"""A confirmed pause owes one local System row, independently of future Resume."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros import pause_notices
from ouroboros.budget_pause import set_budget_pause
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
    monkeypatch.setattr(pause_notices, "_KNOWN", {})
    monkeypatch.setattr(pause_notices, "_BOOTSTRAPPED", set())
    monkeypatch.setattr(pause_notices, "_RECORDED", {})
    write_task_result(root, "task", "running", chat_id=1)
    return root, sent


def _pause(root, episode="one", reason="budget", state="paused"):
    return set_budget_pause(root, "task", {"pause_id": episode, "state": state, "reason": reason})


def _rows(root):
    path = root / "logs" / "chat.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def test_confirmed_pause_persists_system_row_and_repeated_confirmation_does_not_duplicate(notices):
    root, sent = notices
    _pause(root, state="pausing")
    pause_notices.reconcile_pause_notices(root)
    assert _rows(root) == []
    _pause(root)
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


def test_resume_and_second_pause_preserve_both_owed_episodes(notices):
    root, sent = notices
    _pause(root)
    _pause(root, state="resumed")
    _pause(root, episode="two")
    pending = load_task_result(root, "task")["pause_notices"]
    assert len(pending) == 2
    # Lost process hints are rebuilt once from the actual retained task field.
    pause_notices._KNOWN.clear()
    pause_notices.reconcile_pause_notices(root)
    assert len(sent) == 2
    assert {r["card_row_id"] for r in _rows(root)} == {"pause:task:one", "pause:task:two"}
    assert load_task_result(root, "task")["budget_pause"]["pause_id"] == "two"


def test_chat_append_before_ack_is_recovered_without_a_second_send(notices, monkeypatch):
    root, sent = notices
    _pause(root)
    original = pause_notices._acknowledge
    monkeypatch.setattr(pause_notices, "_acknowledge", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    pause_notices.reconcile_pause_notices(root)
    assert len(_rows(root)) == 1 and load_task_result(root, "task")["pause_notices"]
    monkeypatch.setattr(pause_notices, "_acknowledge", original)
    pause_notices.reconcile_pause_notices(root)
    assert len(_rows(root)) == len(sent) == 1
    assert "pause_notices" not in load_task_result(root, "task")


def test_failed_required_append_remains_owed_and_retries(notices, monkeypatch):
    root, sent = notices
    _pause(root)
    original = message_bus.log_chat
    monkeypatch.setattr(message_bus, "log_chat", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    pause_notices.reconcile_pause_notices(root)
    assert not sent and not _rows(root) and load_task_result(root, "task")["pause_notices"]
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
    pause_notices.reconcile_pause_notices(root)
    assert len(sent) == 1 and "owner's Pause" in sent[0][1]


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


def test_worker_replica_cannot_resurrect_an_acknowledged_notice():
    from ouroboros.post_task_checkpoint import project_replica_task_result_fields
    overlay = project_replica_task_result_fields(
        {"task_id": "task", "status": "running"},
        {"pause_notices": {"pause:task:old": {"reason": "budget"}}},
    )
    assert "pause_notices" not in overlay


def test_pauses_after_bootstrap_are_discovered_without_another_history_scan(notices, monkeypatch):
    import ouroboros.task_result_facts as facts
    root, sent = notices
    pause_notices.reconcile_pause_notices(root)
    monkeypatch.setattr(facts, "raw_result_facts", lambda *a, **k: pytest.fail("history rescanned"))
    _pause(root)
    pause_notices.reconcile_pause_notices(root)
    assert len(sent) == 1


def test_a_committed_episode_hint_cannot_be_erased_by_an_older_scan(notices, monkeypatch):
    import ouroboros.task_results as results
    root, sent = notices
    pause_notices.reconcile_pause_notices(root)  # bootstrap already completed
    pause_notices.track(root, "task")
    real_load = results.load_task_result
    first = True

    def read_then_new_pause(*args, **kwargs):
        nonlocal first
        old = real_load(*args, **kwargs)
        if first:
            first = False
            _pause(root)  # commits after the maintenance read, before its empty-result cleanup
        return old

    monkeypatch.setattr(results, "load_task_result", read_then_new_pause)
    pause_notices.reconcile_pause_notices(root)
    assert not sent and real_load(root, "task")["pause_notices"]
    pause_notices.reconcile_pause_notices(root)
    assert len(sent) == 1 and not real_load(root, "task").get("pause_notices")
