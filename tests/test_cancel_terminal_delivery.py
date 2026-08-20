"""The durable terminal delivery seam: the answer the owner is owed, and its receipt.

Split out of ``tests/test_cancel_intents_phase_a.py`` by theme: the durable send-ordered
registry, the honest unreviewed-salvage message and the real block that heals its
placeholder, the completed outcome that must not read as salvage, the receipt identity
that survives the settle, and the owed registration that precedes every enqueue.
"""

from __future__ import annotations

import hashlib
import json
import types

from ouroboros import cancel_intents as ci
from ouroboros.task_results import STATUS_COMPLETED, STATUS_RUNNING, load_task_result, write_task_result

from tests._cancel_intents_shared import _CaptureQueue
from tests._cancel_intents_shared import qenv as _qenv

# The fixture is requested by name as a test parameter, so it is re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
qenv = _qenv


def test_delivery_registry_is_durable_and_send_ordered(tmp_path):
    from supervisor import terminal_delivery as td

    did = td.delivery_id_for("t1", "answer text")
    assert not td.already_delivered(tmp_path, did)
    assert td.register_delivery(tmp_path, did) is True
    assert td.already_delivered(tmp_path, did)          # survives on disk
    assert td.register_delivery(tmp_path, did) is False  # duplicate registration


def test_deliver_unreviewed_salvage_builds_honest_message(tmp_path):
    from supervisor import terminal_delivery as td

    preserved = tmp_path / "full.txt"
    long_text = "line of salvage\n" * 600
    preserved.write_text(long_text, encoding="utf-8")
    write_task_result(tmp_path, "task-a", "cancelled", result="stopped")
    queue = _CaptureQueue()
    delivered = td.deliver_unreviewed_salvage(
        tmp_path,
        {"chat_id": 7},
        "task-a",
        outcome="cancelled",
        salvaged_text=long_text,
        preserved_path=str(preserved),
        children=[{"task_id": "c1", "outcome": "cancelled", "salvaged": True}],
        event_queue=queue,
    )
    assert delivered is True
    (event,) = queue.events
    assert event["chat_id"] == 7 and event["task_id"] == "task-a"
    assert event["delivery_id"].startswith("final:task-a:")
    # Q4 non-mimicry: the receipt is typed SYSTEM end to end.
    assert event["role"] == "system" and event["system_type"] == "cancel_receipt"
    text = event["text"]
    assert "WITHOUT review" in text
    assert "last persisted intermediate model message" in text
    assert "NOT a final answer" in text
    omitted = len(long_text.strip()) - td.SALVAGE_PREVIEW_CHARS
    assert f"{omitted} chars omitted" in text           # exact disclosed count
    assert "1 descendant task(s) were settled with it" in text
    # Q5=A: the technical facts stay OUT of chat and live in the durable
    # cancel_receipt block the details panel renders.
    assert str(preserved) not in text
    assert "sha256" not in text
    assert "task's details panel" in text
    stored = load_task_result(tmp_path, "task-a")
    receipt = stored["cancel_receipt"]
    full_digest = hashlib.sha256(preserved.read_bytes()).hexdigest()
    assert receipt["salvage"]["path"] == str(preserved)
    assert receipt["salvage"]["sha256"] == full_digest
    assert receipt["salvage"]["size_bytes"] == preserved.stat().st_size
    assert receipt["preview_omitted_chars"] == omitted
    assert receipt["children"] == [
        {"task_id": "c1", "outcome": "cancelled", "salvaged": True}
    ]
    assert receipt["delivery_id"] == event["delivery_id"]

    # Second delivery of the same content is suppressed only AFTER registration.
    td.register_delivery(tmp_path, event["delivery_id"])
    queue.events.clear()
    assert td.deliver_unreviewed_salvage(
        tmp_path, {"chat_id": 7}, "task-a",
        outcome="cancelled", salvaged_text=long_text,
        preserved_path=str(preserved),
        children=[{"task_id": "c1", "outcome": "cancelled", "salvaged": True}],
        event_queue=queue,
    ) is False
    assert queue.events == []


def test_real_salvage_block_heals_placeholder_and_survives_replay(tmp_path):
    """m6-preserved-key: a REAL salvage receipt carries preserved=True, so a
    late real block heals an early placeholder, while a placeholder replay
    still never clobbers a persisted real block (the original minor-6 pin)."""
    from supervisor import terminal_delivery as td

    write_task_result(tmp_path, "task-m6", "cancelled", result="stopped")
    # An early placeholder persisted first (no durable copy existed yet).
    td._persist_cancel_receipt(
        tmp_path, "task-m6",
        settled_status="cancelled", outcome="cancelled",
        delivery_id="d-m6", preserved_path="", preview_omitted=0,
    )
    stored = load_task_result(tmp_path, "task-m6")
    assert stored["cancel_receipt"]["salvage"] == {"path": "", "preserved": False}

    # A late REAL salvage block replayed over it -> the real block WINS.
    preserved = tmp_path / "m6-full.txt"
    preserved.write_text("the whole salvaged text", encoding="utf-8")
    td._persist_cancel_receipt(
        tmp_path, "task-m6",
        settled_status="cancelled", outcome="cancelled",
        delivery_id="d-m6", preserved_path=str(preserved), preview_omitted=0,
    )
    stored = load_task_result(tmp_path, "task-m6")
    salvage = stored["cancel_receipt"]["salvage"]
    assert salvage["path"] == str(preserved)
    assert salvage["preserved"] is True
    assert salvage["sha256"] == hashlib.sha256(preserved.read_bytes()).hexdigest()
    assert salvage["size_bytes"] == preserved.stat().st_size

    # A placeholder replay after the real block -> the real block SURVIVES.
    td._persist_cancel_receipt(
        tmp_path, "task-m6",
        settled_status="cancelled", outcome="cancelled",
        delivery_id="d-m6", preserved_path="", preview_omitted=0,
    )
    stored = load_task_result(tmp_path, "task-m6")
    assert stored["cancel_receipt"]["salvage"] == salvage


def test_completed_outcome_reads_as_result_not_salvage(tmp_path):
    """GR2-12: the completed-vs-salvage branch keys on the TYPED stored status,
    never on the presentation prose in ``outcome``."""
    from supervisor import terminal_delivery as td

    queue = _CaptureQueue()
    td.deliver_unreviewed_salvage(
        tmp_path, {"chat_id": 3}, "task-b",
        outcome="completed before the cancellation (result preserved)",
        salvaged_text="the finished answer", settled_status="completed",
        event_queue=queue,
    )
    (event,) = queue.events
    assert event["text"].startswith("✅ Task task-b completed before the cancellation")
    assert "WITHOUT review" not in event["text"]

    # Prose that merely STARTS with "completed" no longer forges the ✅ frame:
    # without the typed status the message stays an honest unreviewed salvage.
    queue.events.clear()
    td.deliver_unreviewed_salvage(
        tmp_path, {"chat_id": 3}, "task-c",
        outcome="completed-looking prose without a typed status",
        salvaged_text="salvaged text", event_queue=queue,
    )
    (event,) = queue.events
    assert event["text"].startswith("⚠️ Task task-c")
    assert "WITHOUT review" in event["text"]


def test_receipt_identity_is_the_stop_episode_and_survives_the_settle(tmp_path):
    """CF-04: the receipt delivery id is ``cancel:<tid>:<request_id>`` — bound
    to the stop episode, stable across wording changes AND across the settle
    (the publish half rebuilds after the intent row is gone and must re-derive
    the SAME id from the owed row the pre-settle half registered)."""
    from supervisor import terminal_delivery as td

    write_task_result(tmp_path, "ep-1", STATUS_RUNNING, result="working")
    intent = ci.request_cancel(tmp_path, "ep-1")
    rid = intent["request_id"]

    # Pre-settle half (owed registration): id comes from the ACTIVE intent.
    event = td.build_unreviewed_salvage_event(
        tmp_path, {"chat_id": 4}, "ep-1", outcome="cancelled",
        salvaged_text="partial work", settled_status="cancelled",
    )
    assert event["delivery_id"] == f"cancel:ep-1:{rid}"
    assert event["role"] == "system" and event["system_type"] == "cancel_receipt"
    assert td.register_pending_delivery(tmp_path, event) is True

    # Settle removes the active intent; the publish half re-derives the id
    # from the pending owed row instead of falling back to a content digest.
    ci.settle_intent(tmp_path, "ep-1", outcome="cancelled", request_id=rid)
    rebuilt = td.build_unreviewed_salvage_event(
        tmp_path, {"chat_id": 4}, "ep-1", outcome="cancelled",
        salvaged_text="partial work", settled_status="cancelled",
    )
    assert rebuilt["delivery_id"] == event["delivery_id"]

    # No episode at all (e.g. a reap without an intent): content-derived
    # fallback keeps the pre-S3 vocabulary.
    other = td.build_unreviewed_salvage_event(
        tmp_path, {"chat_id": 4}, "no-episode", outcome="cancelled",
        salvaged_text="text", settled_status="cancelled",
    )
    assert other["delivery_id"].startswith("final:no-episode:")


def test_salvage_receipt_is_complete_for_a_short_answer_too(tmp_path):
    """A-F14 under Q5=A: every salvage still gets its verification receipt —
    the exact-completeness half in chat, the path/sha half in the durable
    ``cancel_receipt`` block the details panel renders."""
    from supervisor import terminal_delivery as td

    preserved = tmp_path / "short.txt"
    preserved.write_text("a short but whole answer", encoding="utf-8")
    write_task_result(tmp_path, "short-task", "cancelled", result="stopped")
    queue = _CaptureQueue()
    td.deliver_unreviewed_salvage(
        tmp_path, {"chat_id": 5}, "short-task", outcome="cancelled",
        salvaged_text="a short but whole answer", preserved_path=str(preserved),
        event_queue=queue,
    )
    (event,) = queue.events
    digest = hashlib.sha256(preserved.read_bytes()).hexdigest()
    assert "nothing omitted" in event["text"]
    assert "task's details panel" in event["text"]
    receipt = load_task_result(tmp_path, "short-task")["cancel_receipt"]
    assert receipt["salvage"]["sha256"] == digest
    assert receipt["salvage"]["path"] == str(preserved)

    # An unreadable preservation is stamped UNVERIFIED in the durable block
    # instead of silently claiming a verified copy.
    queue.events.clear()
    write_task_result(tmp_path, "short-task-2", "cancelled", result="stopped")
    td.deliver_unreviewed_salvage(
        tmp_path, {"chat_id": 5}, "short-task-2", outcome="cancelled",
        salvaged_text="another whole answer", preserved_path=str(tmp_path / "gone.txt"),
        event_queue=queue,
    )
    (event,) = queue.events
    receipt = load_task_result(tmp_path, "short-task-2")["cancel_receipt"]
    assert receipt["salvage"].get("unreadable") is True

    # No preserved copy at all is disclosed in CHAT (the owner must know the
    # preview is the only copy).
    queue.events.clear()
    td.deliver_unreviewed_salvage(
        tmp_path, {"chat_id": 5}, "short-task-3", outcome="cancelled",
        salvaged_text="third whole answer", preserved_path="", event_queue=queue,
    )
    (event,) = queue.events
    assert "NO durable full copy" in event["text"]


def test_deliver_final_message_live_registers_owed_before_enqueue(tmp_path):
    """AR2-4 (§8-A2): the NORMAL terminal path enters the durable outbox — the
    answer is owed BEFORE the enqueue, so a crash between put and processing
    replays it; the shared delivery id keeps it single-delivery."""
    from ouroboros.task_finalization import deliver_final_message_live
    from supervisor import terminal_delivery as td

    events = [{"type": "send_message", "chat_id": 3, "task_id": "fin1", "text": "the answer"}]

    class _BoomQueue:
        def put(self, evt):
            raise RuntimeError("queue died")

    # Even when the put dies, the answer is already OWED — the crash window the
    # incident lived in is closed for this seam.
    assert deliver_final_message_live(_BoomQueue(), events, "fin1", drive_root=tmp_path) is False
    owed = td.pending_deliveries(tmp_path)
    assert [row["task_id"] for row in owed] == ["fin1"]
    did = str(events[0]["delivery_id"])
    assert owed[0]["delivery_id"] == did

    # The normal path enqueues the same id; a confirmed send clears the row.
    queue = _CaptureQueue()
    assert deliver_final_message_live(queue, events, "fin1", drive_root=tmp_path) is True
    (sent,) = queue.events
    assert sent["delivery_id"] == did
    td.register_delivery(tmp_path, did)
    assert td.pending_deliveries(tmp_path) == []

    # A final without a chat id is never registered: replay could not send it.
    events2 = [{"type": "send_message", "chat_id": 0, "task_id": "fin2", "text": "x"}]
    assert deliver_final_message_live(_CaptureQueue(), events2, "fin2", drive_root=tmp_path) is True
    assert td.pending_deliveries(tmp_path) == []


def test_reaper_registers_the_salvage_before_task_done(qenv, monkeypatch):
    """AR2-5a crash order: the owed salvage delivery precedes the task_done
    enqueue, so a crash between them can no longer resolve the card while
    losing the owner's answer."""
    from supervisor import task_reaper as tr
    from supervisor import workers as workers_mod

    calls: list = []
    monkeypatch.setattr(tr, "_kill_and_confirm_worker_dead", lambda *_a, **_kw: True)
    monkeypatch.setattr(tr, "_deliver_reap_salvage",
                        lambda _q, task, tid, reason, unreconciled_runs=None:
                        calls.append(("salvage", tid)))
    monkeypatch.setattr(
        workers_mod, "get_event_q",
        lambda: types.SimpleNamespace(
            put=lambda evt: calls.append((str(evt.get("type")), str(evt.get("task_id")))),
        ),
    )
    monkeypatch.setattr(workers_mod, "respawn_worker", lambda wid: None)
    monkeypatch.setattr(
        qenv.q, "reconstruct_task_cost",
        lambda tid, fields=True, **_kw: {"cost_accounting_status": "available",
                                         "cost_final": True, "cost_usd": 0.0},
    )

    tr.reap_timed_out_task({
        "worker_id": 0, "proc": None, "task_id": "reap1",
        "task": {"id": "reap1", "chat_id": 4}, "task_type": "chat",
        "terminal_reason": "idle_timeout", "attempt": 3, "owner_chat_id": 0,
        "runtime_sec": 10.0, "will_retry": False,
    })

    assert ("salvage", "reap1") in calls
    assert ("task_done", "reap1") in calls
    assert calls.index(("salvage", "reap1")) < calls.index(("task_done", "reap1"))


def test_finalize_on_miss_delivers_the_unreviewed_salvage(qenv, monkeypatch):
    """AR2-5b (owner 5=A): the miss lane used to emit NO delivery at all — a
    cancelled outcome now ships the unreviewed salvage through the shared seam."""
    delivered: list = []
    monkeypatch.setattr(
        "supervisor.terminal_delivery.deliver_unreviewed_salvage",
        lambda drive, task, tid, **kw: delivered.append({"task_id": tid, **kw}),
    )
    monkeypatch.setattr(qenv.q, "_emit_cancel_task_done", lambda *_a, **_kw: None)
    write_task_result(qenv.drive, "miss-del", STATUS_RUNNING, result="was working",
                      chat_id=6)
    ci.request_cancel(qenv.drive, "miss-del", reason="stop")

    assert qenv.tl.cancel_task_custody("miss-del") == qenv.tl.CANCEL_CANCELLED
    (row,) = delivered
    assert row["task_id"] == "miss-del"
    assert row["outcome"] == "cancelled"


def test_finalize_on_miss_completion_wins_delivers_the_completed_result(qenv, monkeypatch):
    """AR2-5b: the completion-wins branch of the miss lane delivers the KEPT
    answer through the normal deduped seam — owed BEFORE enqueued."""
    from supervisor import terminal_delivery as td
    from supervisor import workers as workers_mod

    queue = _CaptureQueue()
    monkeypatch.setattr(workers_mod, "get_event_q", lambda: queue)
    child_drive = qenv.drive / "child-of-misswin"
    write_task_result(child_drive, "miss-win", STATUS_COMPLETED,
                      result="the finished answer", chat_id=6)
    write_task_result(qenv.drive, "miss-win", STATUS_RUNNING, result="mirror",
                      chat_id=6, child_drive_root=str(child_drive))
    ci.request_cancel(qenv.drive, "miss-win", reason="late cancel")

    assert qenv.tl.cancel_task_custody("miss-win") == qenv.tl.CANCEL_ALREADY_SETTLED
    (sent,) = [e for e in queue.events if e.get("type") == "send_message"]
    assert sent["text"] == "the finished answer"
    assert sent["chat_id"] == 6
    owed = td.pending_deliveries(qenv.drive)
    assert [r["delivery_id"] for r in owed] == [sent["delivery_id"]], "owed before enqueued"


def test_fast_settled_reentry_delivers_idempotently_and_settles_with_the_claim(
    qenv, monkeypatch,
):
    """GR2-4 (fast already-settled re-entry): delivery runs BEFORE the settle
    and the settle is fenced by the claimed generation — never an unfenced
    removal of an intent another owner may hold."""
    order: list = []
    monkeypatch.setattr(
        "supervisor.terminal_delivery.deliver_miss_lane_outcome",
        lambda *a, **kw: order.append(("deliver", str(a[3]))),
    )
    real_settle = ci.settle_intent
    monkeypatch.setattr(
        "ouroboros.cancel_intents.settle_intent",
        lambda root, tid, **kw: order.append(("settle", tid)) or real_settle(root, tid, **kw),
    )
    write_task_result(qenv.drive, "fast1", STATUS_RUNNING, result="working", chat_id=6)
    ci.request_cancel(qenv.drive, "fast1", reason="stop")
    # Natural completion wins the race before custody arrives.
    write_task_result(qenv.drive, "fast1", STATUS_COMPLETED, result="the answer", chat_id=6)

    assert qenv.tl.cancel_task_custody("fast1") == qenv.tl.CANCEL_ALREADY_SETTLED

    assert order.index(("deliver", "fast1")) < order.index(("settle", "fast1"))
    assert ci.active_intent(qenv.drive, "fast1") is None
    settled_rows = [
        json.loads(line)
        for line in (qenv.drive / "logs" / "supervisor.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    settle_row = next(
        r for r in settled_rows
        if r.get("type") == "cancel_intent" and r.get("event") == "settled"
        and r.get("task_id") == "fast1"
    )
    assert int(settle_row.get("generation") or 0) >= 1, (
        "the settle must ride the claimed generation, not an unfenced removal"
    )
