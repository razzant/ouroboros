"""The durable pending outbox for root answers that could not be sent.

Split out of ``tests/test_cancel_intents_phase_a.py`` by theme: every nonblocking root
answer enters the outbox, replays are exactly-once with backoff, give-up is loud, and
capacity eviction is disclosed.
"""

from __future__ import annotations
import json
import pathlib
import types

from tests._cancel_intents_shared import _CaptureQueue


def _age_pending_rows(drive) -> None:
    """Backdate every owed row past the (backoff-spaced) replay min-age (test-only)."""
    from datetime import datetime, timedelta, timezone

    store = pathlib.Path(drive) / "state" / "terminal_deliveries.json"
    data = json.loads(store.read_text(encoding="utf-8"))
    # Past the LARGEST backoff step (60 * 2**5 = 1920s), so every attempt is due.
    old = (datetime.now(timezone.utc) - timedelta(seconds=7200)).isoformat()
    for row in (data.get("pending") or {}).values():
        row["registered_at"] = old
        if "last_replay_at" in row:
            row["last_replay_at"] = old
    store.write_text(json.dumps(data), encoding="utf-8")

def test_pending_outbox_replays_an_unsent_answer_exactly_once(tmp_path):
    """A-F7: crash between settle and send used to lose the answer forever."""
    from supervisor import terminal_delivery as td

    event = {
        "type": "send_message", "chat_id": 9, "task_id": "outbox1",
        "text": "the salvaged answer", "format": "markdown",
        "delivery_id": td.delivery_id_for("outbox1", "the salvaged answer"),
    }
    assert td.register_pending_delivery(tmp_path, event) is True
    owed = td.pending_deliveries(tmp_path)
    assert [row["delivery_id"] for row in owed] == [event["delivery_id"]]

    queue = _CaptureQueue()
    # A row younger than the min age is presumed still in flight, not lost.
    assert td.replay_pending_deliveries(tmp_path, event_queue=queue) == []
    assert queue.events == []
    _age_pending_rows(tmp_path)

    replayed = td.replay_pending_deliveries(tmp_path, event_queue=queue)
    assert replayed == [event["delivery_id"]]
    (sent,) = queue.events
    assert sent["type"] == "send_message" and sent["chat_id"] == 9
    assert sent["text"] == "the salvaged answer" and sent["format"] == "markdown"

    # A confirmed send clears the row in the SAME write as the delivered mark.
    td.register_delivery(tmp_path, event["delivery_id"])
    assert td.pending_deliveries(tmp_path) == []
    queue.events.clear()
    assert td.replay_pending_deliveries(tmp_path, event_queue=queue) == []
    assert queue.events == []
    # An already-delivered id is never registered as owed again — but the
    # answer IS durably tracked, so the GR3-4 contract answers True (False is
    # reserved for a real durability gap that must keep a cancel intent open).
    assert td.register_pending_delivery(tmp_path, event) is True
    assert td.pending_deliveries(tmp_path) == []

def test_pending_outbox_gives_up_loudly_instead_of_retrying_forever(tmp_path, monkeypatch):
    """A-F7 bound + AR2-7: an unreachable chat must not become a tick-rate retry
    storm — and exhaustion is a DISCLOSED outcome, never a silent drop: the full
    text is preserved on disk, a typed ``terminal_delivery_exhausted`` event
    lands in events.jsonl, and the owner gets a chat notice naming both."""
    from supervisor import terminal_delivery as td

    notices: list = []
    monkeypatch.setattr(
        "supervisor.message_bus.send_with_budget",
        lambda chat_id, text, **kw: notices.append((chat_id, text, kw)),
    )
    event = {
        "type": "send_message", "chat_id": 9, "task_id": "outbox2",
        "text": "never lands", "delivery_id": td.delivery_id_for("outbox2", "never lands"),
    }
    td.register_pending_delivery(tmp_path, event)
    queue = _CaptureQueue()
    for _ in range(td._PENDING_MAX_REPLAYS):
        _age_pending_rows(tmp_path)
        assert td.replay_pending_deliveries(tmp_path, event_queue=queue) == [
            event["delivery_id"],
        ]
        assert td.record_replay_dispatch(tmp_path, event["delivery_id"])
        queue.release_all()  # the handler returned without a receipt: the copy is no longer outstanding
    assert notices == [], "no give-up notice while attempts remain"
    _age_pending_rows(tmp_path)
    assert td.replay_pending_deliveries(tmp_path, event_queue=queue) == []
    assert td.pending_deliveries(tmp_path) == []
    assert len(queue.events) == td._PENDING_MAX_REPLAYS
    # The disclosure: durable typed event + preserved full copy + chat notice.
    rows = [
        json.loads(line)
        for line in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    (exhausted,) = [r for r in rows if r.get("type") == "terminal_delivery_exhausted"]
    assert exhausted["task_id"] == "outbox2"
    assert exhausted["delivery_id"] == event["delivery_id"]
    assert exhausted["chat_id"] == 9
    preserved = pathlib.Path(str(exhausted["preserved_path"]))
    assert preserved.is_file() and preserved.read_text(encoding="utf-8") == "never lands"
    (notice,) = notices
    assert notice[0] == 9
    # A missing ACK is not non-delivery, and the copy is of THIS message.
    assert "Sending a message for task outbox2 was not confirmed" in notice[1]
    assert "could not be delivered" not in notice[1] and "full text" not in notice[1]
    assert f"A copy of this message is saved at {preserved}." in notice[1]
    assert notice[2] == {"role": "system", "system_type": "terminal_incident"}

def test_pending_outbox_spaces_replays_with_backoff(tmp_path):
    """Queue admission dates backoff; dispatch entries increase its interval."""
    from supervisor import terminal_delivery as td

    event = {
        "type": "send_message", "chat_id": 9, "task_id": "outbox3",
        "text": "spaced", "delivery_id": td.delivery_id_for("outbox3", "spaced"),
    }
    td.register_pending_delivery(tmp_path, event)
    queue = _CaptureQueue()
    _age_pending_rows(tmp_path)
    assert td.replay_pending_deliveries(tmp_path, event_queue=queue) == [
        event["delivery_id"],
    ]
    # Queue admission only dates the base interval. Dispatch doubles it.
    assert td.replay_pending_deliveries(tmp_path, event_queue=queue) == []
    assert td.pending_deliveries(tmp_path)[0]["dispatch_attempts"] == 0
    assert td.record_replay_dispatch(tmp_path, event["delivery_id"])
    (row,) = td.pending_deliveries(tmp_path)
    assert row["dispatch_attempts"] == 1
    assert row.get("last_replay_at"), "dispatch also stamps the replay time"
    assert td._replay_due(row) is False
    # The doubled min-age: attempt 2 is due only after 2 * base seconds.
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).timestamp()
    assert td._replay_due(row, now=now + td._REPLAY_MIN_AGE_SEC + 1) is False
    assert td._replay_due(row, now=now + 2 * td._REPLAY_MIN_AGE_SEC + 1) is True

def _emit_root_results(tmp_path, monkeypatch, *, text="the final answer"):
    """Drive the REAL emit_task_results for an ordinary nonblocking root."""
    import time

    import ouroboros.agent_task_pipeline as atp

    drive_root = tmp_path / "data"
    logs = drive_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    class _FakeEnv:
        def __init__(self, root):
            self.drive_root = root

        def drive_path(self, sub):
            p = self.drive_root / sub
            p.mkdir(parents=True, exist_ok=True)
            return p

    class _FakeMemory:
        def load_identity(self):
            return "id"

    monkeypatch.setattr(atp, "_run_post_task_processing_async", lambda *a, **kw: None)
    pending_events: list = []
    task = {"id": "nb-root", "type": "task", "chat_id": 3, "text": "hello"}
    atp.emit_task_results(
        env=_FakeEnv(drive_root), memory=_FakeMemory(), llm=None,
        pending_events=pending_events,
        task=task, text=text,
        usage={"cost": 0.0, "rounds": 1, "prompt_tokens": 1, "completion_tokens": 1},
        llm_trace={"tool_calls": [], "reasoning_notes": []},
        start_time=time.time() - 1.0,
        drive_logs=logs,
        ctx=types.SimpleNamespace(pending_restart_reason=None),
    )
    return drive_root, pending_events

def test_every_nonblocking_root_answer_enters_the_durable_outbox(tmp_path, monkeypatch):
    """GR2-5 crash shape: an ordinary (nonblocking) root's final answer is owed
    in the durable outbox right after result persistence — a worker crash before
    the buffered drain no longer loses the answer; boot replay delivers once."""
    from supervisor import terminal_delivery as td

    drive_root, pending_events = _emit_root_results(tmp_path, monkeypatch)

    send = next(e for e in pending_events if e.get("type") == "send_message")
    assert str(send.get("delivery_id") or "").startswith("final:nb-root:")
    owed = td.pending_deliveries(drive_root)
    assert [row["delivery_id"] for row in owed] == [send["delivery_id"]]

    # Crash shape: the buffered send never went out. Boot replay delivers ONCE.
    _age_pending_rows(drive_root)
    queue = _CaptureQueue()
    assert td.replay_pending_deliveries(drive_root, event_queue=queue) == [send["delivery_id"]]
    (replayed,) = queue.events
    assert replayed["text"] == "the final answer" and replayed["chat_id"] == 3

    # The confirmed send clears the owed row; nothing double-delivers after.
    td.register_delivery(drive_root, send["delivery_id"])
    queue.events.clear()
    assert td.replay_pending_deliveries(drive_root, event_queue=queue) == []
    assert queue.events == []

def test_normal_path_stays_single_send_with_the_owed_registration(tmp_path, monkeypatch):
    """GR2-5 no-double half: the pipeline registration and the blocking path's
    deliver_final_message_live mint the SAME delivery id, so registration is
    idempotent and the send handler's dedupe keeps one delivery."""
    from ouroboros.task_finalization import deliver_final_message_live
    from supervisor import terminal_delivery as td

    drive_root, pending_events = _emit_root_results(tmp_path, monkeypatch)
    send = next(e for e in pending_events if e.get("type") == "send_message")
    did = send["delivery_id"]

    # The live-delivery seam re-registers the same event: still ONE owed row.
    queue = _CaptureQueue()
    assert deliver_final_message_live(queue, pending_events, "nb-root", drive_root=drive_root)
    assert send["delivery_id"] == did, "the id is stable across both seams"
    owed = td.pending_deliveries(drive_root)
    assert [row["delivery_id"] for row in owed] == [did], "no second owed row"

def test_outbox_capacity_eviction_is_disclosed(tmp_path, monkeypatch):
    """GR2-6 (reproduced): the 65th registration used to silently pop the oldest
    owed answer. The eviction now preserves the full text, emits the typed
    durable event with the distinct outbox_capacity reason, and notifies chat."""
    from supervisor import terminal_delivery as td

    notices: list = []
    monkeypatch.setattr(
        "supervisor.message_bus.send_with_budget",
        lambda chat_id, text, **kw: notices.append((chat_id, text)),
    )
    for i in range(td._PENDING_CAP):
        td.register_pending_delivery(tmp_path, {
            "type": "send_message", "chat_id": 5, "task_id": f"cap{i}",
            "text": f"answer {i}", "delivery_id": td.delivery_id_for(f"cap{i}", f"answer {i}"),
        })
    assert len(td.pending_deliveries(tmp_path)) == td._PENDING_CAP

    td.register_pending_delivery(tmp_path, {
        "type": "send_message", "chat_id": 5, "task_id": "cap-new",
        "text": "the newest answer", "delivery_id": td.delivery_id_for("cap-new", "the newest answer"),
    })

    ids = {row["task_id"] for row in td.pending_deliveries(tmp_path)}
    assert "cap-new" in ids and "cap0" not in ids, "oldest evicted, newest kept"
    rows = [
        json.loads(line)
        for line in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    (evicted,) = [r for r in rows if r.get("type") == "terminal_delivery_exhausted"]
    assert evicted["task_id"] == "cap0"
    assert evicted["reason"] == "outbox_capacity"
    preserved = pathlib.Path(str(evicted["preserved_path"]))
    assert preserved.is_file() and "answer 0" in preserved.read_text(encoding="utf-8")
    assert notices and "cap0" in notices[0][1], "owner-visible notice"


def _exhaust(td, drive, task_id, chat_id, text="copy me"):
    row = {"task_id": task_id, "chat_id": chat_id, "text": text,
           "delivery_id": td.delivery_id_for(task_id, text)}
    td._disclose_exhausted_delivery(drive, row)
    rows = [json.loads(line) for line in (pathlib.Path(drive) / "logs" / "events.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    return row, [r for r in rows if r.get("type") == "terminal_delivery_exhausted"][-1]


def test_only_the_same_delivery_recorded_at_the_same_chat_retires_the_notice(tmp_path, monkeypatch):
    """#1317: the send succeeded but the outbox row was never cleared. The cancel
    receipt's ``delivered_chat_id`` for THIS delivery id is positive evidence and
    retires the owner notice; the typed event stays. Another chat, another
    delivery, or an id merely present in the delivered registry is not proof."""
    from ouroboros.task_results import write_task_result
    from supervisor import terminal_delivery as td

    notices: list = []
    monkeypatch.setattr("supervisor.message_bus.send_with_budget",
                        lambda chat_id, text, **kw: notices.append((chat_id, text)))
    did = td.delivery_id_for("t-same", "copy me")
    write_task_result(tmp_path, "t-same", "cancelled", result="stopped",
                      cancel_receipt={"delivery_id": did, "delivered_chat_id": 9})
    _row, event = _exhaust(td, tmp_path, "t-same", 9)
    assert notices == [] and event["owner_notice"] == "not_sent_same_delivery_recorded_at_this_chat"

    write_task_result(tmp_path, "t-other-chat", "cancelled", result="stopped",
                      cancel_receipt={"delivery_id": td.delivery_id_for("t-other-chat", "copy me"),
                                      "delivered_chat_id": 4})
    _row, event = _exhaust(td, tmp_path, "t-other-chat", 9)
    assert "owner_notice" not in event and notices[-1][0] == 9

    write_task_result(tmp_path, "t-other-id", "cancelled", result="stopped",
                      cancel_receipt={"delivery_id": "cancel:t-other-id:ci_1", "delivered_chat_id": 9})
    _row, event = _exhaust(td, tmp_path, "t-other-id", 9)
    assert "owner_notice" not in event and len(notices) == 2

    # A bare id in the delivered registry says nothing about the destination.
    td.register_delivery(tmp_path, td.delivery_id_for("t-registry", "copy me"))
    _row, event = _exhaust(td, tmp_path, "t-registry", 9)
    assert "owner_notice" not in event and len(notices) == 3
    assert all("was not confirmed" in text for _chat, text in notices)


def test_capacity_eviction_and_a_failed_copy_are_worded_as_unconfirmed(tmp_path, monkeypatch):
    from supervisor import terminal_delivery as td

    notices: list = []
    monkeypatch.setattr("supervisor.message_bus.send_with_budget",
                        lambda chat_id, text, **kw: notices.append(text))

    def refuse(*_args, **_kwargs):
        raise OSError("Too many open files")

    monkeypatch.setattr("ouroboros.observability.preserve_salvaged_output", refuse)
    row = {"task_id": "t-evicted", "chat_id": 9, "text": "owed", "delivery_id": "d-evicted"}
    td._disclose_exhausted_delivery(tmp_path, row, reason="outbox_capacity")
    (text,) = notices
    assert text.startswith("⚠️ Sending a message for task t-evicted was not confirmed: its pending-outbox slot")
    assert text.endswith("No copy of this message could be saved.")
