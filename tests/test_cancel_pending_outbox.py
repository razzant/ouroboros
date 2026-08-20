"""The durable outbox: an unsent answer is replayed exactly once, or given up loudly.

Split out of ``tests/test_cancel_intents_phase_a.py`` by theme: the replay that fires
once, the backoff between attempts, the loud give-up instead of an endless retry, the
coverage that every nonblocking root answer enters the outbox, and the disclosed
capacity eviction.
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
    assert "could not be delivered" in notice[1]
    assert str(preserved) in notice[1]


def test_pending_outbox_spaces_replays_with_backoff(tmp_path):
    """AR2-7: ``registered_at`` alone let all five attempts burn on consecutive
    ticks. Each bump stamps ``last_replay_at`` and the next attempt waits an
    exponentially longer min-age, so the cap covers a realistic outage window."""
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
    # Immediately after a replay the row is NOT due again (fresh last_replay_at,
    # and the min-age has doubled) — the next tick must not burn attempt 2.
    assert td.replay_pending_deliveries(tmp_path, event_queue=queue) == []
    (row,) = td.pending_deliveries(tmp_path)
    assert row["replay_attempts"] == 1
    assert row.get("last_replay_at"), "each bump stamps the replay time"
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
