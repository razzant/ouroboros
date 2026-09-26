"""The owed terminal answer: salvage receipts, delivery ordering, idempotent sends.

Split out of ``tests/test_cancel_intents_phase_a.py`` by theme: the durable delivery
registry, the honest unreviewed-salvage message, owed-before-enqueue ordering on the
live and finalize-on-miss paths, and the settled-reentry that delivers exactly once.
"""

from __future__ import annotations
import hashlib
import json
import types
from collections import deque

import pytest
from ouroboros import cancel_intents as ci
from ouroboros.task_results import (
    STATUS_COMPLETED,
    STATUS_RUNNING,
    load_task_result,
    write_task_result,
)

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


@pytest.mark.serial
def test_one_cancel_leaves_exactly_one_salvaged_paragraph_in_the_chat(tmp_path, monkeypatch):
    """Owner item spam L: the stop receipt OWNS the preserved paragraph.

    A cancel used to put the same salvaged text in the chat three times: the
    receipt quoted it, the terminal task row quoted it again, and the project
    summary quoted it a third time. The receipt keeps its bounded preview and
    its durable full-copy facts; the rows that follow name what the bytes are
    and point at the untruncated copy instead of repeating it.
    """
    from ouroboros.project_dialogue import (
        SALVAGE_EXCERPT_LABEL, append_terminal_task_projection,
    )
    from supervisor import terminal_delivery as td

    salvage = "Rewrote the atlas builder and reran the suite."
    preserved = tmp_path / "full.txt"
    preserved.write_text(salvage, encoding="utf-8")
    write_task_result(
        tmp_path, "stopped-one", "cancelled", result=salvage, chat_id=7,
        terminal_origin="host_salvage", reason_code="owner_requested_cancel",
    )
    queue = _CaptureQueue()
    assert td.deliver_unreviewed_salvage(
        tmp_path, {"chat_id": 7}, "stopped-one", outcome="cancelled",
        salvaged_text=salvage, preserved_path=str(preserved), event_queue=queue,
    ) is True
    (receipt,) = queue.events
    assert salvage in receipt["text"]

    from supervisor import events_chat_delivery as delivery
    from ouroboros.utils import append_jsonl

    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    sent = []
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path, RUNNING={}, append_jsonl=append_jsonl,
        send_with_budget=lambda chat, text, **_kw: sent.append((chat, text)),
    )
    delivery._handle_send_message(receipt, ctx)
    assert sent == [(7, receipt["text"])]
    stored = load_task_result(tmp_path, "stopped-one")
    assert stored["cancel_receipt"]["salvage"]["preserved"] is True
    assert append_terminal_task_projection(
        tmp_path, "stopped-one", {"id": "stopped-one", "chat_id": 7}, stored,
        {"status": "cancelled", "chat_id": 7},
    )
    row = next(
        json.loads(line)
        for line in (tmp_path / "logs" / "chat.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    assert f"{SALVAGE_EXCERPT_LABEL}." in row["text"]
    assert salvage not in row["text"]
    assert "get_task_result" not in row["text"]
    assert row["text"].startswith("Cancelled. Root task stopped-one.")


@pytest.mark.serial
def test_cascade_receipt_dedups_the_actual_destination_and_preserves_main(qenv, monkeypatch):
    """A settled root borrows child lineage, then send-time binding wins.

    The cascade does not re-emit a settled root's task_done or rewrite older
    chat rows. A later terminal projection consumes the confirmed receipt.
    """
    from ouroboros.observability import preserve_salvaged_output
    from ouroboros.project_dialogue import (
        SALVAGE_EXCERPT_LABEL, _completion_excerpt, append_terminal_task_projection,
        enqueue_project_completion_summary,
    )
    from ouroboros.projects_registry import create_project, bind_task_to_project
    from ouroboros.utils import append_jsonl
    from supervisor import events_chat_delivery as delivery

    queue = _CaptureQueue()
    monkeypatch.setattr(qenv.workers, "get_event_q", lambda: queue)
    monkeypatch.setattr(qenv.q, "_emit_cancel_task_done", lambda *_a, **_k: None)
    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    text = "Rewrote the atlas builder and reran the suite."
    write_task_result(
        qenv.drive, "settled-root", "failed", result=text,
        terminal_origin="host_salvage", reason_code="budget_exhausted",
    )
    preserve_salvaged_output(qenv.drive, "settled-root", text)
    qenv.q.PENDING[:] = [{
        "id": "live-kid", "chat_id": 77,
        "parent_task_id": "settled-root", "root_task_id": "settled-root",
    }]
    write_task_result(
        qenv.drive, "live-kid", "scheduled",
        parent_task_id="settled-root", root_task_id="settled-root",
    )
    assert qenv.tl.cancel_task_by_id("settled-root", cascade=True)
    event = next(e for e in queue.events if e.get("system_type") == "cancel_receipt")
    assert event["chat_id"] == 77 and text in event["text"]
    before = load_task_result(qenv.drive, "settled-root")
    assert "chat_id" not in before
    assert text in _completion_excerpt(before, chat_id=77)

    project = create_project(qenv.drive, "receipt-destination", name="Receipt destination")
    bind_task_to_project(
        qenv.drive, "settled-root", project["id"], project["chat_id"],
        origin={"absent": "system"},
    )
    sent = []
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=qenv.drive, RUNNING={}, append_jsonl=append_jsonl,
        send_with_budget=lambda chat, body, **_kw: sent.append((chat, body)),
    )
    delivery._handle_send_message(event, ctx)
    assert sent == [(project["chat_id"], event["text"])]
    stored = load_task_result(qenv.drive, "settled-root")
    assert stored["cancel_receipt"]["delivered_chat_id"] == project["chat_id"]
    assert _completion_excerpt(stored, chat_id=project["chat_id"]) == f"{SALVAGE_EXCERPT_LABEL}."
    assert text in _completion_excerpt(stored, chat_id=77)
    assert text in _completion_excerpt(stored, chat_id=1)
    task = {"id": "settled-root", "project_id": project["id"], "chat_id": project["chat_id"]}
    assert append_terminal_task_projection(
        qenv.drive, "settled-root", task, stored,
        {"status": "failed", "chat_id": project["chat_id"]},
    )
    terminal = next(
        row for row in map(json.loads, (qenv.drive / "logs/chat.jsonl").read_text().splitlines())
        if row.get("type") == "task_summary"
    )
    assert text not in terminal["text"] and SALVAGE_EXCERPT_LABEL in terminal["text"]
    assert "get_task_result" not in terminal["text"]
    queue.events.clear()
    assert enqueue_project_completion_summary(
        qenv.drive, {}, "settled-root", task, stored, {"status": "failed"},
    )
    summary = next(e for e in queue.events if e.get("system_type") == "project_completion_summary")
    assert summary["chat_id"] == 1 and text in summary["text"]
    assert summary["text"].endswith("Open the Project for details.")


@pytest.mark.serial
def test_unsent_receipts_and_new_stop_episodes_do_not_inherit_delivery(tmp_path, monkeypatch):
    """An origin address, failed send and duplicate skip prove no new delivery."""
    from ouroboros.project_dialogue import _completion_excerpt, SALVAGE_EXCERPT_LABEL
    from ouroboros.utils import append_jsonl
    from supervisor import terminal_delivery as td, events_chat_delivery as delivery

    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    bound = {"chat": 9}
    monkeypatch.setattr(delivery, "_bound_project_chat_id", lambda *_a: bound["chat"])
    text = "Preserved applied output."
    write_task_result(
        tmp_path, "origin-seven", "cancelled", result=text,
        chat_id=7, terminal_origin="host_salvage",
    )

    def build(did):
        return td.build_unreviewed_salvage_event(
            tmp_path, {"chat_id": 7}, "origin-seven", outcome="cancelled",
            salvaged_text=text, delivery_id=did,
        )

    def stored():
        return load_task_result(tmp_path, "origin-seven")

    def fail(*_a, **_k):
        raise RuntimeError("transport rejected")

    event = build("cancel:origin-seven:one")
    assert text in _completion_excerpt(stored(), chat_id=7)
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path, RUNNING={}, send_with_budget=fail, append_jsonl=append_jsonl,
    )
    delivery._handle_send_message(event, ctx)
    assert "delivered_chat_id" not in stored()["cancel_receipt"]
    assert text in _completion_excerpt(stored(), chat_id=9)
    sent = []
    ctx.send_with_budget = lambda chat, body, **_k: sent.append((chat, body))
    delivery._handle_send_message(event, ctx)
    assert len(sent) == 1 and sent[0][0] == 9
    assert _completion_excerpt(stored(), chat_id=9) == f"{SALVAGE_EXCERPT_LABEL}."
    assert text in _completion_excerpt(stored(), chat_id=7)
    bound["chat"] = 1
    delivery._handle_send_message(event, ctx)
    assert len(sent) == 1
    assert stored()["cancel_receipt"]["delivered_chat_id"] == 9
    assert text in _completion_excerpt(stored(), chat_id=1)
    build("cancel:origin-seven:one")
    assert stored()["cancel_receipt"]["delivered_chat_id"] == 9
    new_event = build("cancel:origin-seven:two")
    assert "delivered_chat_id" not in stored()["cancel_receipt"]
    assert text in _completion_excerpt(stored(), chat_id=9)
    # A late actual send for the older episode cannot attest the newer receipt.
    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    monkeypatch.setattr(td, "already_delivered", lambda *_a: False)
    delivery._handle_send_message(event, ctx)
    assert "delivered_chat_id" not in stored()["cancel_receipt"]
    delivery._handle_send_message(new_event, ctx)
    assert stored()["cancel_receipt"]["delivered_chat_id"] == 1
    assert _completion_excerpt(stored(), chat_id=1) == f"{SALVAGE_EXCERPT_LABEL}."

def test_receipt_names_the_stop_cause_before_and_after_the_settle(tmp_path):
    """The technical stop facts in the DETAILS panel name the cause the owner
    actually made. While the intent is live the receipt reads it there; once
    custody has settled, the intent row is gone and the same scalars live on the
    stored ``cancel_origin``, so a receipt built then is not suddenly causeless."""
    from supervisor import terminal_delivery as td

    write_task_result(tmp_path, "task-live-cause", STATUS_RUNNING, chat_id=1)
    intent = ci.request_cancel(tmp_path, "task-live-cause", reason="server_shutdown",
                               source="snapshot_restore")
    td._persist_cancel_receipt(
        tmp_path, "task-live-cause",
        settled_status="cancelled", outcome="cancelled",
        delivery_id="d-live", preserved_path="", preview_omitted=0,
    )
    live = load_task_result(tmp_path, "task-live-cause")["cancel_receipt"]
    assert live["stop_reason"] == "server_shutdown"
    assert live["stop_requested_at"] == intent["requested_at"]

    # A task whose custody already settled: no intent row is left to read.
    origin = {"reason": "server_shutdown", "source": "snapshot_restore",
              "requested_at": "2026-09-19T23:41:07+00:00"}
    write_task_result(tmp_path, "task-settled-cause", "cancelled", chat_id=1,
                      result="stopped", cancel_origin=origin)
    assert ci.active_intent(tmp_path, "task-settled-cause") is None

    td._persist_cancel_receipt(
        tmp_path, "task-settled-cause",
        settled_status="cancelled", outcome="cancelled",
        delivery_id="d-settled", preserved_path="", preview_omitted=0,
    )
    settled = load_task_result(tmp_path, "task-settled-cause")["cancel_receipt"]
    assert settled["stop_reason"] == "server_shutdown"
    assert settled["stop_requested_at"] == origin["requested_at"]

    # Quiet direction: a task that was never cancelled gets no stop cause at all.
    write_task_result(tmp_path, "task-no-cause", STATUS_COMPLETED, chat_id=1, result="done")
    td._persist_cancel_receipt(
        tmp_path, "task-no-cause",
        settled_status="completed", outcome="completed",
        delivery_id="d-none", preserved_path="", preview_omitted=0,
    )
    plain = load_task_result(tmp_path, "task-no-cause")["cancel_receipt"]
    assert "stop_reason" not in plain and "stop_requested_at" not in plain


def test_salvage_receipt_states_files_rescued_even_without_salvageable_text(tmp_path):
    """TZ-2 C2: "(no salvageable agent output ...)" must not read as "no files". The
    receipt states the stat-only artifact-store count — positive, zero or unknown — and
    that no hashes were computed; the typed fact rides ``cancel_receipt``. The count is a
    mutable disclosure, never part of the content-derived delivery identity."""
    from ouroboros.headless import task_artifacts_dir
    from supervisor import terminal_delivery as td

    def build(tid, task=None):
        return td.build_unreviewed_salvage_event(
            tmp_path, task or {"chat_id": 4}, tid, outcome="cancelled", settled_status="cancelled")

    write_task_result(tmp_path, "files-1", STATUS_RUNNING, result="working")
    store = task_artifacts_dir(tmp_path, "files-1")
    (store / "draft.docx").write_bytes(b"x")
    event = build("files-1")
    assert "(no salvageable agent output was found for this task)" in event["text"]
    assert "Files rescued: 1 " in event["text"] and "hashes not computed" in event["text"], event["text"]
    receipt = load_task_result(tmp_path, "files-1")["cancel_receipt"]
    assert receipt["files_rescued"] == {"count": 1, "state": "positive", "hash_computed": False,
                                        "stores": [{"store": str(store), "count": 1, "readable": True}]}
    (store / "more.txt").write_bytes(b"y")
    rebuilt = build("files-1")
    assert rebuilt["delivery_id"] == event["delivery_id"] and "Files rescued: 2 " in rebuilt["text"]
    assert load_task_result(tmp_path, "files-1")["cancel_receipt"]["files_rescued"]["count"] == 2

    write_task_result(tmp_path, "files-0", STATUS_RUNNING, result="working")
    task_artifacts_dir(tmp_path, "files-0")
    zero = build("files-0")
    assert "Files rescued: none" in zero["text"] and "hashes not computed" in zero["text"], zero["text"]
    assert load_task_result(tmp_path, "files-0")["cancel_receipt"]["files_rescued"]["state"] == "zero"

    write_task_result(tmp_path, "files-x", STATUS_RUNNING, result="working")
    task_artifacts_dir(tmp_path, "files-x", create=False).write_text("not a directory", encoding="utf-8")
    unknown = build("files-x")
    assert "Files rescued: unknown" in unknown["text"], unknown["text"]
    assert load_task_result(tmp_path, "files-x")["cancel_receipt"]["files_rescued"]["state"] == "unknown"

    # A split root: the child drive named on the task row is walked beside the canonical store.
    child = tmp_path / "child-drive"
    write_task_result(tmp_path, "files-s", STATUS_RUNNING, result="working")
    (task_artifacts_dir(child, "files-s") / "out.txt").write_bytes(b"o")
    split = build("files-s", {"chat_id": 4, "child_drive_root": str(child)})
    assert "Files rescued: 1 " in split["text"], split["text"]
    stores = load_task_result(tmp_path, "files-s")["cancel_receipt"]["files_rescued"]["stores"]
    assert [row["store"] for row in stores] == [str(task_artifacts_dir(tmp_path, "files-s", create=False)),
                                                str(task_artifacts_dir(child, "files-s", create=False))]
