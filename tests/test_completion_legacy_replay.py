"""Upgrade/replay reaches the existing sender, preserving content and event identity."""
import asyncio
import copy
from collections import deque
from types import SimpleNamespace

import pytest

from ouroboros import terminal_projection
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.utils import append_jsonl
from supervisor import events_chat_delivery as delivery, message_bus
from tests.test_completion_notices import _record
from tests.test_project_completion_mirror import _enqueue_row
from tests.test_telegram_subagent_cards import _setup, _Rec


@pytest.mark.parametrize("notify", ["off", "on"])
@pytest.mark.parametrize("salvage", [False, True])
@pytest.mark.parametrize("lane", ["canonical", "outbox"])
def test_legacy_pending_event_through_sender(notify, salvage, lane, tmp_path, monkeypatch):
    plugin, api = _setup(tmp_path / "telegram", monkeypatch, TELEGRAM_NOTIFY_TASKS=notify)
    handle = plugin._make_outbound(api)
    root = tmp_path / "host"
    record = _record("receipt")
    record.update(terminal_origin="host_salvage" if salvage else "model_final",
                  result="Preserved intermediate output (not a final answer): useful draft" if salvage else "Final answer")
    event = _enqueue_row(root, monkeypatch, record)
    # This additive field is the only difference from the actual pre-upgrade producer.
    event["progress_meta"].pop("completion_notice")
    original = copy.deepcopy(event)
    ready = {"token": "publication-1", "main_delivery": {"event": event, "attempts": 0}}
    fields = {key: value for key, value in record.items() if key not in {"task_id", "status"}}
    fields.update(canonical_terminal_projection_ready=ready)
    write_task_result(root, "root-project", "completed", **fields)
    if lane == "canonical":
        event = terminal_projection.retain_main_delivery(root, "root-project", event)
    # Outbox replay goes directly to the same sender, bypassing the producer.
    frames = []
    bridge = message_bus.LocalChatBridge({})
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", root)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7})
    monkeypatch.setattr(message_bus, "publish_event", lambda _topic, row: asyncio.run(handle(row)))
    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    monkeypatch.setattr(delivery, "_handoff_delivered_review", lambda *_a, **_kw: None)
    ctx = SimpleNamespace(DRIVE_ROOT=root, RUNNING={}, append_jsonl=append_jsonl,
                          send_with_budget=message_bus.send_with_budget)
    delivery._handle_send_message(event, ctx)
    assert frames[-1]["content"] == original["text"]
    assert frames[-1]["target_label"] == original["progress_meta"]["target_label"]
    assert frames[-1]["delivery_id"] == original["delivery_id"]
    sent = "\n".join(row[1] for row in _Rec.sent)
    assert ("useful draft" in sent) is salvage
    if notify == "off":
        assert len(_Rec.sent) == int(salvage)
        assert "done with warnings" not in sent.lower()
    else:
        assert len(_Rec.sent) == 1 and original["progress_meta"]["target_label"] in sent
        assert "earlier version" not in sent
    stored = load_task_result(root, "root-project", strict=True)
    assert stored["canonical_terminal_projection_ready"]["main_delivery"]["event"] == original
    delivery._handle_send_message(event, ctx)
    assert len(_Rec.sent) == (1 if notify == "on" or salvage else 0)


def test_unknown_legacy_source_is_not_assumed_status_only(tmp_path, monkeypatch):
    plugin, api = _setup(tmp_path, monkeypatch, TELEGRAM_NOTIFY_TASKS="off")
    event = {"text": "Kept content whose result source is unavailable.", "role": "system",
             "system_type": "project_completion_summary", "task_id": "old"}
    assert terminal_projection.completion_event_view(tmp_path, event, {}) is event
    asyncio.run(plugin._make_outbound(api)(event))
    assert _Rec.sent[0][1] == event["text"]


def _pending_event(root, monkeypatch, *, salvage=False, failed_answer=None):
    """Use the real producer, retaining its pre-upgrade (no metadata) wire shape."""
    from ouroboros.projects_registry import create_project, bind_task_to_project
    from supervisor import terminal_delivery

    project = create_project(root, "launch", name="Launch")
    bind_task_to_project(root, "root-project", project["id"], project["chat_id"], origin={"absent": "system"})
    row = _record("receipt")
    row.update(task_id="root-project", project_id="launch", title="Ship release", chat_id=project["chat_id"],
               terminal_origin="host_salvage" if salvage else "model_final",
               result="useful draft" if salvage else "Final answer")
    write_task_result(root, "root-project", "completed", **{k: v for k, v in row.items() if k not in {"task_id", "status"}})
    if failed_answer is not None:
        final = {"type": "send_message", "task_id": "root-project", "chat_id": project["chat_id"],
                 "text": failed_answer, "delivery_id": terminal_delivery.delivery_id_for("root-project", failed_answer)}
        assert terminal_delivery.register_pending_delivery(root, final)
        with monkeypatch.context() as stage:
            stage.setattr("ouroboros.utils.in_worker_process", lambda: True)
            terminal_delivery._retire_exhausted(root, [final])
    queued = []
    register = terminal_delivery.register_pending_delivery

    def register_legacy(drive, event):
        legacy = copy.deepcopy(event)
        if legacy.get("system_type") == "project_completion_summary":
            legacy["progress_meta"].pop("completion_notice", None)
        return register(drive, legacy)

    with monkeypatch.context() as stage:
        # Model the old wire shape BEFORE durable registration, not just in the
        # queued copy: otherwise the outbox would bypass legacy enrichment.
        stage.setattr(terminal_delivery, "register_pending_delivery", register_legacy)
        stage.setattr(terminal_delivery, "enqueue_terminal_delivery", lambda _root, event, **_kw: queued.append(copy.deepcopy(event)) or True)
        from ouroboros.project_dialogue import project_completion_delivery_outcome
        assert project_completion_delivery_outcome(root, {}, "root-project", row, row, {})[1]
    event = queued[0]
    event["progress_meta"].pop("completion_notice")
    return event


def _legacy_sender(root, monkeypatch, notify="on"):
    plugin, api = _setup(root / "telegram", monkeypatch, TELEGRAM_NOTIFY_TASKS=notify)
    handle = plugin._make_outbound(api)
    frames = []
    bridge = message_bus.LocalChatBridge({})
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", root)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7})
    monkeypatch.setattr(message_bus, "publish_event", lambda _topic, event: asyncio.run(handle(event)))
    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    monkeypatch.setattr(delivery, "_handoff_delivered_review", lambda *_a, **_kw: None)
    ctx = SimpleNamespace(DRIVE_ROOT=root, RUNNING={}, append_jsonl=append_jsonl, send_with_budget=message_bus.send_with_budget)
    return lambda event: delivery._handle_send_message(event, ctx), frames


@pytest.mark.parametrize("notify", ["off", "on"])
def test_quarantined_result_does_not_block_retained_message(notify, tmp_path, monkeypatch):
    from ouroboros.store_import import ensure_current_store
    from supervisor import terminal_delivery

    event = _pending_event(tmp_path / "producer", monkeypatch, salvage=True)
    root = tmp_path / "host"
    raw = root / "task_results" / "root-project.json"
    raw.parent.mkdir(parents=True)
    raw.write_text("{truncated result", encoding="utf-8")
    ensure_current_store(root)
    with pytest.raises(ValueError, match="unreadable or invalid"):
        load_task_result(root, "root-project", strict=True)
    assert terminal_delivery.register_pending_delivery(root, event)
    [retained] = terminal_delivery.pending_deliveries(root)
    send, frames = _legacy_sender(root, monkeypatch, notify)
    send(retained)
    assert frames[-1]["content"] == event["text"]
    assert "useful draft" in _Rec.sent[0][1]
    assert not terminal_delivery.pending_deliveries(root)


@pytest.mark.parametrize("lane", ["canonical", "outbox"])
@pytest.mark.parametrize("failed_answer", ["Final answer", "Earlier answer"])
def test_legacy_replay_recovers_source_delivery_registry_warning(lane, failed_answer, tmp_path, monkeypatch):
    from supervisor import terminal_delivery

    root = tmp_path / "host"
    event = _pending_event(root, monkeypatch, failed_answer=failed_answer)
    [failure] = terminal_delivery.terminal_answer_receipts(root, "root-project")["exhausted"]
    path = failure["preserved_path"]
    applies = failed_answer == event["progress_meta"]["completion_answer"]
    assert path and (path in event["text"]) is applies
    assert ("Sending the final answer was not confirmed" in event["text"]) is applies
    if lane == "canonical":
        write_task_result(root, "root-project", "completed", canonical_terminal_projection_ready={
            "token": "publication-1", "main_delivery": {"event": copy.deepcopy(event), "attempts": 0}})
        retained = terminal_projection.retain_main_delivery(root, "root-project", event)
    else:
        [retained] = [e for e in terminal_delivery.pending_deliveries(root)
                      if e["delivery_id"] == event["delivery_id"]]
        assert "completion_notice" not in retained["progress_meta"]
    send, frames = _legacy_sender(root, monkeypatch)
    send(retained)
    assert frames[-1]["content"] == event["text"]
    assert "Launch › Ship release" in _Rec.sent[0][1]
    assert (path in _Rec.sent[0][1]) is applies
    assert ("Sending the final answer was not confirmed" in _Rec.sent[0][1]) is applies
    if applies:
        assert "Review and verification details" not in _Rec.sent[0][1]


@pytest.mark.parametrize("frozen", ["different-answer", "different-attempt", "pointer"])
def test_legacy_mirror_cannot_borrow_a_later_answers_file(tmp_path, monkeypatch, frozen):
    from supervisor import terminal_delivery

    event = _pending_event(tmp_path, monkeypatch)
    row = load_task_result(tmp_path, "root-project", strict=True)
    assert event["progress_meta"]["completion_answer"] == row["result"]
    final = terminal_delivery.build_completed_result_event(tmp_path, row, "root-project", row)
    assert terminal_delivery.register_pending_delivery(tmp_path, final)
    with monkeypatch.context() as worker:
        worker.setattr("ouroboros.utils.in_worker_process", lambda: True)
        terminal_delivery._retire_exhausted(tmp_path, [final])
    if frozen == "different-answer":
        event["progress_meta"]["completion_answer"] = "Earlier answer"
    elif frozen == "different-attempt":
        event["progress_meta"]["task_attempt"] = 1
        row["task_attempt"] = 2
    else:
        event["progress_meta"].pop("completion_answer")
    view = terminal_projection.completion_event_view(tmp_path, event, row)
    [failure] = terminal_delivery.terminal_answer_receipts(tmp_path, "root-project")["exhausted"]
    notice = view["progress_meta"].get("completion_notice", {})
    assert failure["preserved_path"] and failure["preserved_path"] not in notice.get("detail", "")
    assert view["text"] == event["text"] and view["delivery_id"] == event["delivery_id"]
    if frozen == "pointer":
        assert "affected answer is unknown" in notice["detail"]
    else:
        assert view is event
