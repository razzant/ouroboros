"""Selected readiness is a peek; delivery and restart preserve exact cursors."""
import threading
from types import SimpleNamespace

import pytest

from ouroboros import review_ledger as ledger, review_waiting as waiting
from ouroboros.task_results import write_task_result
from tests.test_review_ledger import _raw, BOTH
from tests.test_review_ledger_settlement import _record


def test_helper_selection_peeks_only_and_pins_ownership(tmp_path, monkeypatch):
    from ouroboros import model_sleep, owner_wait

    ctx = SimpleNamespace(task_id="parent", drive_root=tmp_path, budget_drive_root=None)
    write_task_result(tmp_path, "child", "running", parent_task_id="parent")
    write_task_result(tmp_path, "peer", "running", parent_task_id="other")
    runs = {name: SimpleNamespace(run_id=name, review_owned=review, terminal_state="", settled=False)
            for name, review in (("own-run", False), ("review-run", True))}
    monkeypatch.setattr(model_sleep, "_owned_runs", lambda *_args, **_kw: runs)
    monkeypatch.setattr(owner_wait, "_wait_entries", lambda *_a: pytest.fail("peek materialized mail"))
    default = waiting.select_helpers(ctx)
    assert default["selection"]["tasks"] == default["selection"]["runs"] == []
    state = waiting.select_helpers(ctx, {"tasks": ["child"], "runs": ["own-run"]})
    assert state["selection"]["tasks"] == ["child"] and state["selection"]["runs"] == ["own-run"]
    write_task_result(tmp_path, "peer", "completed", parent_task_id="other")
    write_task_result(tmp_path, "new-child", "completed", parent_task_id="parent")
    assert not waiting.ready_helpers(ctx, state)
    write_task_result(tmp_path, "child", "completed", parent_task_id="parent")
    assert [row["id"] for row in waiting.ready_helpers(ctx, state)] == ["task:child"]
    assert state["seen_helpers"] == []
    assert waiting.ready_helpers(ctx, state)  # readiness never acknowledges
    state["seen_helpers"] = ["task:child"]
    assert not waiting.ready_helpers(ctx, state)
    runs["own-run"].settled = True
    assert waiting.ready_helpers(ctx, state)[0]["id"] == "run:own-run"


def test_delivery_cursor_waits_for_transcript_and_checkpoint(tmp_path, monkeypatch):
    from ouroboros import working_checkpoint
    from ouroboros.late_review_marks import owe_after_end, OWED_SET
    from ouroboros.store import open_store
    from ouroboros.tools.tool_result import _install_tool_result_sidecar, _restore_tool_result_sidecar, _published_tool_result

    record = _record(tmp_path)
    ctx = SimpleNamespace(task_id="task-1", drive_root=tmp_path, budget_drive_root=None,
                          _review_wait_selection={"selection": {}, "seen_helpers": [], "seen_outcomes": [], "duty_id": "owed"})
    owe_after_end(tmp_path, "owed", {"left_review": {"task_id": "task-1", "record_id": record["record_id"]}})
    waiting.save_wait(ctx, record["record_id"], ctx._review_wait_selection)
    record = ledger.settle_from_collected(tmp_path, record["record_id"], [_raw("a", "a", parts=BOTH)])
    token = _install_tool_result_sidecar(ctx, object())
    try:
        waiting.publish_delivery(ctx, "answer", record)
        receipt = _published_tool_result(ctx, None).meta["review_delivery"]
    finally:
        _restore_tool_result_sidecar(token)
    assert ledger.load_record(tmp_path, record["record_id"])["waiting"]["seen_outcomes"] == []
    commits = []
    monkeypatch.setattr(working_checkpoint, "defer_content_ack", lambda _ctx, callback: commits.append(callback))
    ctx._left_review_wait = {"record_id": record["record_id"], "done": True, "noticed": False, "notice": "settled"}
    waiting.acknowledge_delivery(ctx, receipt)
    assert ctx._review_wait_selection["seen_outcomes"]
    assert ctx._left_review_wait["done"] and ctx._left_review_wait["notice"] == "settled"
    assert ledger.load_record(tmp_path, record["record_id"])["waiting"]["seen_outcomes"] == []
    commits[0]()
    saved = ledger.load_record(tmp_path, record["record_id"])["waiting"]
    assert saved["seen_outcomes"] == receipt["outcomes"]
    with open_store(tmp_path, write=True) as txn:
        assert txn.get_row(OWED_SET, "owed")["left_review"]["seen_outcomes"] == receipt["outcomes"]
        txn.delete_row(OWED_SET, "owed")
    commits[0]()
    with open_store(tmp_path, write=False) as txn:
        assert txn.get_row(OWED_SET, "owed") is None, "an ACK cannot recreate a discharged room duty"
    restored = SimpleNamespace(**vars(ctx))
    restored._left_review_wait = saved
    assert waiting.select_helpers(restored)["seen_outcomes"] == receipt["outcomes"]
    restored._left_review_wait = {"record_id": record["record_id"]}
    assert waiting.select_helpers(restored)["selection"] == {}, "legacy waits mean the whole wave"


def test_unreadable_record_cannot_acknowledge_delivery(tmp_path):
    record = _record(tmp_path)
    ctx = SimpleNamespace(task_id="task-1", drive_root=tmp_path, budget_drive_root=None)
    ledger.record_path(tmp_path, record["record_id"]).write_text("{broken")
    waiting.acknowledge_delivery(ctx, {"record_id": record["record_id"], "outcomes": ["wave:operation:outcome"]})
    assert not getattr(ctx, "_pending_content_acks", None)
    assert not getattr(ctx, "_review_wait_selection", None)


@pytest.mark.parametrize("author", ["task-1", "foreign-author"])
def test_delivery_cursor_accepts_only_its_author(tmp_path, author):
    from ouroboros.working_checkpoint import flush_content_acks

    record = _record(tmp_path)
    ctx = SimpleNamespace(task_id=author, drive_root=tmp_path, budget_drive_root=None)
    waiting.acknowledge_delivery(ctx, {"record_id": record["record_id"], "outcomes": ["wave:operation:outcome"]})
    assert bool(getattr(ctx, "_review_wait_selection", None)) is (author == "task-1")
    assert bool(getattr(ctx, "_pending_content_acks", None)) is (author == "task-1")
    flush_content_acks(ctx)
    saved = ledger.load_record(tmp_path, record["record_id"])
    assert bool(saved.get("waiting")) is (author == "task-1")


@pytest.mark.parametrize("paused", [False, True])
def test_pause_precedes_a_ready_helper(tmp_path, monkeypatch, paused):
    import contextlib
    from ouroboros import review_pause, owner_pause

    entry = SimpleNamespace(author_left_for="", event=threading.Event(), episode={},
                            started_at=1, pause_before_launch=False, retry_state={})
    monkeypatch.setattr(review_pause, "author_paused", lambda _op: paused)
    monkeypatch.setattr(review_pause, "author_hears_mail", lambda: False)
    monkeypatch.setattr(review_pause, "_author_coordinates", lambda _op: (str(tmp_path), "parent", "parent"))
    monkeypatch.setattr(owner_pause, "launch_lock", lambda *_a: contextlib.nullcontext())
    monkeypatch.setattr(waiting, "wake_reason", lambda *_a, **_kw: "task:child:completed")
    with review_pause.receptive_author({"drive_root": tmp_path, "task_id": "parent"}):
        assert review_pause.author_detaches(SimpleNamespace(wait=None), [entry])
    assert entry.pause_before_launch is paused
    assert entry.author_left_for == ("" if paused else "helper_result")


@pytest.mark.parametrize("lost", [False, True])
def test_ended_author_does_not_turn_remote_pending_into_api_loss(tmp_path, lost):
    from ouroboros.review_pause import _settle_left_record

    record = _record(tmp_path)
    seats = [{"slot_id": name, "parts": list(BOTH), "operation_id": name, "status": "error",
              "operation_state": "custody_lost" if lost else "in_flight", "late_result_pending": True}
             for name in ("a", "b")]
    settled = _settle_left_record({"record_id": record["record_id"], "drive_root": str(tmp_path)}, seats, final=True)
    assert settled["state"] == ("settled" if lost else "pending")
    assert settled["verdict"]["aggregate"] != "PASS"


def test_restored_notice_retains_exact_reference_selection_and_seen(tmp_path):
    from ouroboros.review_pause import wake_restored_left_review, PROCESS_TOKEN

    reference = {"record_id": "rl-original", "task_id": "parent", "surface": "change"}
    state = {"owner": "ended-process", "review_reference": reference, "selection": {"tasks": ["child"]},
             "seen_helpers": ["task:child"], "seen_outcomes": ["wave:operation:outcome"], "msg_id": "original"}
    ctx = SimpleNamespace(task_id="parent", drive_root=tmp_path, _left_review_wait=state)
    assert wake_restored_left_review(ctx)
    assert ctx._left_review_wait["review_reference"] == reference
    assert ctx._left_review_wait["seen_helpers"] == ["task:child"]
    assert ctx._left_review_wait["seen_outcomes"] == ["wave:operation:outcome"]
    assert ctx._left_review_wait["owner"] == PROCESS_TOKEN
    assert not wake_restored_left_review(ctx), "same restoration must not notify twice"
