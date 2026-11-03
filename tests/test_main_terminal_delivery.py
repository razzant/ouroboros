"""Main terminal obligations outlive the transport cache and failed chat writes."""
from __future__ import annotations

import json
from collections import deque
from types import SimpleNamespace

import pytest

from ouroboros import dialogue_store, store
from ouroboros.task_results import load_task_result, write_task_result
from tests._dialogue import chat_rows
from ouroboros.terminal_projection import SETTLEMENT_DEFERRED, settle_terminal_projection
from supervisor import terminal_delivery as delivery


def lost_bus(queued):
    """Each admitted copy is observed then lost before its handler, leaving no membership."""
    return SimpleNamespace(put=queued.append, delivery_outstanding=lambda _did: False,
                           put_unless_outstanding=lambda event: queued.append(event) is None)


@pytest.fixture
def owed(tmp_path, monkeypatch):
    from ouroboros.projects_registry import bind_task_to_project, create_project

    project = create_project(tmp_path, "launch", name="Launch")
    bind_task_to_project(tmp_path, "root", project["id"], project["chat_id"], origin={"absent": "system"})
    write_task_result(tmp_path, "root", "completed", root_task_id="root", project_id=project["id"],
                      chat_id=project["chat_id"], result="**The answer**", terminal_origin="model_final")
    queued = []
    monkeypatch.setattr("supervisor.workers.get_event_q", lambda *, create=True: lost_bus(queued))
    assert settle_terminal_projection(tmp_path, "root") == SETTLEMENT_DEFERRED
    assert len(queued) == 1
    return SimpleNamespace(root=tmp_path, queued=queued, event=queued[0], project=project)


def _due(root):
    row = load_task_result(root, "root")
    ready = row["canonical_terminal_projection_ready"]
    ready["main_delivery"]["last_attempt_at"] = "2000-01-01T00:00:00+00:00"
    write_task_result(root, "root", "completed", canonical_terminal_projection_ready=ready)


@pytest.mark.parametrize("inflight", [False, True])
def test_main_reoffer_preserves_clock_while_bus_copy_is_outstanding(owed, monkeypatch, inflight):
    from supervisor.event_observation import ObservedEventQueue
    bus = ObservedEventQueue()
    monkeypatch.setattr("supervisor.workers.get_event_q", lambda *, create=True: bus)
    bus.put(owed.event)
    if inflight:
        assert bus.get_nowait() == owed.event
    _due(owed.root)
    before = load_task_result(owed.root, 'root')['canonical_terminal_projection_ready']['main_delivery']
    settle_terminal_projection(owed.root, 'root')
    assert bus.qsize() == (0 if inflight else 1), 'no second queued copy'
    assert load_task_result(owed.root, 'root')['canonical_terminal_projection_ready']['main_delivery'] == before
    if not inflight:
        bus.get_nowait()
    bus.release_delivery(owed.event['delivery_id'])  # handler lost the send, no durable ACK
    settle_terminal_projection(owed.root, 'root')
    assert bus.get_nowait() == owed.event
    assert bus.empty()
    assert load_task_result(owed.root, 'root')['canonical_terminal_projection_ready']['main_delivery']['attempts'] == before['attempts'] + 1


@pytest.mark.parametrize('unknown', [True, False])
def test_main_unknown_membership_spends_nothing_but_actual_put_failure_is_throttled(owed, monkeypatch, unknown):
    from supervisor.event_observation import ObservedEventQueue

    class FailingBus(ObservedEventQueue):
        puts = 0

        def put_unless_outstanding(self, event):
            self.puts += 1
            raise OSError('real queue put failed')

    bus = SimpleNamespace() if unknown else FailingBus()
    monkeypatch.setattr('supervisor.workers.get_event_q', lambda *, create=True: bus)
    _due(owed.root)
    before = load_task_result(owed.root, 'root')['canonical_terminal_projection_ready']['main_delivery']
    for _ in range(3):
        assert settle_terminal_projection(owed.root, 'root') == SETTLEMENT_DEFERRED
    after = load_task_result(owed.root, 'root')['canonical_terminal_projection_ready']['main_delivery']
    if unknown:
        assert after == before
    else:
        assert bus.puts == 1
        assert after['attempts'] == before['attempts'] + 1
        assert after['last_attempt_at'] != before['last_attempt_at']
        assert after['event'] == before['event']


def test_canonical_debt_survives_capacity_and_retry_limits_without_reset_or_scan(owed, monkeypatch):
    from ouroboros.obligations import members

    # Its cache entry is oldest. Eviction does not erase the canonical envelope.
    for index in range(delivery._PENDING_CAP):
        delivery.register_pending_delivery(owed.root, {"delivery_id": f"other:{index}", "chat_id": 0,
                                                      "task_id": f"other-{index}", "text": "x"})
    assert owed.event["delivery_id"] not in {r["delivery_id"] for r in delivery.pending_deliveries(owed.root)}
    assert "root" in members(owed.root, "terminal_projection")
    original = owed.event.copy()
    monkeypatch.setattr("ouroboros.terminal_projection._already_in_chat",
                        lambda *_a, **_kw: pytest.fail("settled Project receipt must not rescan history"))
    for attempt in range(1, delivery._PENDING_MAX_REPLAYS + 3):
        # Normal maintenance cannot enqueue immediately after the previous attempt.
        assert settle_terminal_projection(owed.root, "root") == SETTLEMENT_DEFERRED
        assert len(owed.queued) == attempt
        _due(owed.root)
        assert settle_terminal_projection(owed.root, "root") == SETTLEMENT_DEFERRED
        assert len(owed.queued) == attempt + 1
        saved = load_task_result(owed.root, "root")["canonical_terminal_projection_ready"]["main_delivery"]
        assert saved["attempts"] == attempt + 1
        assert saved["event"] == original
    # Generic outbox replay never gives this debt a second retry clock.
    cache = json.loads((owed.root / "state/terminal_deliveries.json").read_text())
    cache["pending"] = {owed.event["delivery_id"]: {**owed.event, "registered_at": "2000-01-01T00:00:00Z"}}
    (owed.root / "state/terminal_deliveries.json").write_text(json.dumps(cache))
    from tests._cancel_intents_shared import DeliveryBus
    bus = DeliveryBus()
    before = load_task_result(owed.root, "root")["canonical_terminal_projection_ready"]["main_delivery"]
    assert delivery.replay_pending_deliveries(owed.root, event_queue=bus) == []
    assert bus.empty(), "the observable membership bus must receive no second-clock copy"
    assert load_task_result(owed.root, "root")["canonical_terminal_projection_ready"]["main_delivery"] == before
    # Positive control: this same bus really admits ordinary due outbox replay.
    ordinary = {"type": "send_message", "delivery_id": "final:control:0123456789abcdef",
                "task_id": "control", "text": "Ordinary final", "chat_id": 1,
                "registered_at": "2000-01-01T00:00:00Z"}
    cache["pending"][ordinary["delivery_id"]] = ordinary
    (owed.root / "state/terminal_deliveries.json").write_text(json.dumps(cache))
    assert delivery.replay_pending_deliveries(owed.root, event_queue=bus) == [ordinary["delivery_id"]]
    assert bus.get_nowait()["delivery_id"] == ordinary["delivery_id"]
    assert bus.empty() and bus.delivery_outstanding(ordinary["delivery_id"])


def test_strict_chat_failure_cannot_acknowledge_then_repair_survives_registry_aging(owed, monkeypatch):
    from supervisor import events_chat_delivery as handler, message_bus as bus
    from ouroboros.obligations import members
    from ouroboros.utils import append_jsonl

    frames = []
    bridge = bus.LocalChatBridge({})
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(bus, "DATA_DIR", owed.root)
    monkeypatch.setattr(bus, "load_state", lambda: {"owner_id": 1})
    monkeypatch.setattr(bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(bus, "publish_event", lambda *_a, **_kw: None)
    monkeypatch.setattr(handler, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    context = SimpleNamespace(DRIVE_ROOT=owed.root, RUNNING={}, send_with_budget=bus.send_with_budget,
                              append_jsonl=append_jsonl)
    real_append = dialogue_store.append
    monkeypatch.setattr(dialogue_store, "append", lambda *_a, **_kw: (_ for _ in ()).throw(store.StoreBusy("refused")))
    handler._handle_send_message(owed.event, context)
    assert not frames
    assert not delivery.already_delivered(owed.root, owed.event["delivery_id"])
    assert load_task_result(owed.root, "root")["canonical_terminal_projection_ready"]
    assert owed.event["delivery_id"] in {r["delivery_id"] for r in delivery.pending_deliveries(owed.root)}
    monkeypatch.setattr(dialogue_store, "append", real_append)
    handler._handle_send_message(owed.event, context)
    assert frames[-1]["delivery_id"] == owed.event["delivery_id"]
    row = load_task_result(owed.root, "root")
    assert row["canonical_terminal_projection_ready"] is None
    assert row["canonical_terminal_projection"]["main_disposition"] == "delivered"
    assert "root" not in members(owed.root, "terminal_projection")
    for index in range(delivery._REGISTRY_CAP + 1):
        delivery.register_delivery(owed.root, f"other:{index}")
    registry = json.loads((owed.root / "state/terminal_deliveries.json").read_text())
    assert owed.event["delivery_id"] not in registry["delivered"]
    assert delivery.already_delivered(owed.root, owed.event["delivery_id"], task_id="root",
                                      publication_token=owed.event["terminal_projection_token"])
    handler._DELIVERED_MESSAGE_IDS.clear()
    before = len(frames)
    handler._handle_send_message(owed.event, context)
    assert len(frames) == before


def test_tick_replay_reoffers_cached_main_debt_only_when_its_own_clock_is_due(owed):
    # A lost live send (no sender ran) stays cached: the boot/tick replay names it, but
    # the canonical claim decides, so it neither retries early nor needs the 600 s sweep.
    assert delivery.replay_pending_deliveries(owed.root) == []
    assert owed.queued == [owed.event]
    _due(owed.root)
    assert delivery.replay_pending_deliveries(owed.root) == []
    assert owed.queued == [owed.event, owed.event]
    saved = load_task_result(owed.root, "root")["canonical_terminal_projection_ready"]["main_delivery"]
    assert saved["attempts"] == 2 and saved["event"] == owed.event
    delivery.replay_pending_deliveries(owed.root)
    assert len(owed.queued) == 2, "a just-claimed attempt is not due again"


def test_bound_main_event_survives_later_project_registry_failure(owed, monkeypatch):
    monkeypatch.setattr("ouroboros.projects_registry.task_presentation_snapshot",
                        lambda *_a, **_kw: pytest.fail("retained event must not be rerouted"))
    _due(owed.root)
    assert settle_terminal_projection(owed.root, "root") == SETTLEMENT_DEFERRED
    assert owed.queued == [owed.event, owed.event]


@pytest.mark.parametrize("fmt", ["", "markdown"])
@pytest.mark.parametrize("receipt", [None, (False, "transport refused")])
def test_unbound_notice_requires_its_durable_row_without_a_delivery_receipt(tmp_path, monkeypatch, fmt, receipt):
    from supervisor import message_bus as bus

    monkeypatch.setattr(bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(bus, "load_state", lambda: {"owner_id": 1})
    sent = []
    monkeypatch.setattr(bus, "get_bridge", lambda: SimpleNamespace(
        send_message=lambda *args, **kwargs: sent.append((args, kwargs)) or receipt))
    real_append = dialogue_store.append
    monkeypatch.setattr(dialogue_store, "append", lambda *_a, **_kw: (_ for _ in ()).throw(store.StoreBusy("refused")))
    with pytest.raises(RuntimeError, match="could not be persisted"):
        bus.send_with_budget(1, "Startup notice", role="system", fmt=fmt, require_write=True)
    assert not sent
    monkeypatch.setattr(dialogue_store, "append", real_append)
    bus.send_with_budget(1, "Startup notice", role="system", fmt=fmt, require_write=True)
    assert len(sent) == 1
    [row] = chat_rows(tmp_path)
    assert row["text"] == "Startup notice"


@pytest.mark.parametrize("fmt", ["", "markdown"])
@pytest.mark.parametrize("receipt", [None, (False, "transport refused")])
def test_bound_delivery_keeps_debt_without_a_positive_transport_receipt(owed, monkeypatch, fmt, receipt):
    from supervisor import events_chat_delivery as handler, message_bus as bus
    from ouroboros.utils import append_jsonl

    frames = []
    bridge = bus.LocalChatBridge({})
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(bus, "DATA_DIR", owed.root)
    monkeypatch.setattr(bus, "load_state", lambda: {"owner_id": 1})
    monkeypatch.setattr(bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(bus, "publish_event", lambda *_a, **_kw: None)
    monkeypatch.setattr(handler, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    context = SimpleNamespace(DRIVE_ROOT=owed.root, RUNNING={}, send_with_budget=bus.send_with_budget,
                              append_jsonl=append_jsonl)
    event = {**owed.event, "format": fmt}
    send = bridge.send_message
    monkeypatch.setattr(bridge, "send_message", lambda *_a, **_kw: receipt)
    handler._handle_send_message(event, context)
    assert not frames
    assert not delivery.already_delivered(owed.root, event["delivery_id"])
    assert load_task_result(owed.root, "root")["canonical_terminal_projection_ready"]
    assert event["delivery_id"] in {r["delivery_id"] for r in delivery.pending_deliveries(owed.root)}
    assert any(row.get("delivery_id") == event["delivery_id"] for row in chat_rows(owed.root))

    monkeypatch.setattr(bridge, "send_message", send)
    handler._handle_send_message(event, context)
    assert len(frames) == 1
    assert frames[0]["markdown"] is bool(fmt)
    assert delivery.already_delivered(owed.root, event["delivery_id"])
    assert load_task_result(owed.root, "root")["canonical_terminal_projection_ready"] is None


@pytest.mark.parametrize("result, expected", [
    ({"outcome_axes": {"execution": {"status": "ok"}}}, False),
    ({"terminal_plan_review_open": True}, True),
    ({"outcome_axes": {"execution": {"status": "best_effort"}}, "reason_code": "best_effort"}, True),
])
def test_main_limitation_is_one_recorded_fact_beside_the_answer(tmp_path, monkeypatch, result, expected):
    from tests.test_project_completion_mirror import _enqueue_row, _result

    event = _enqueue_row(tmp_path, monkeypatch, _result(**result))
    assert bool(event["progress_meta"].get("completion_reason")) is expected
    assert event["progress_meta"]["completion_answer"].startswith("**Done:")


def test_obsolete_attempt_cannot_acknowledge_or_suppress_new_main_publication(owed, monkeypatch):
    from supervisor import events_chat_delivery as handler
    from ouroboros.utils import append_jsonl

    row = load_task_result(owed.root, "root")
    ready = row["canonical_terminal_projection_ready"]
    fresh = {**owed.event, "terminal_projection_token": "new-publication"}
    ready.update(token="new-publication", main_delivery={"event": fresh, "attempts": 1})
    marker = {**row["canonical_terminal_projection"], "token": "new-publication"}
    write_task_result(owed.root, "root", "completed", canonical_terminal_projection_ready=ready,
                      canonical_terminal_projection=marker)
    # An old sender returned after the newer publication had become ready.
    delivery.register_delivery(owed.root, owed.event["delivery_id"], emitted=owed.event)
    assert load_task_result(owed.root, "root")["canonical_terminal_projection_ready"] == ready
    assert not delivery.already_delivered(owed.root, fresh["delivery_id"], task_id="root",
                                          publication_token="new-publication")
    sent = []
    context = SimpleNamespace(DRIVE_ROOT=owed.root, RUNNING={}, append_jsonl=append_jsonl,
                              send_with_budget=lambda *args, **kw: sent.append((args, kw)))
    monkeypatch.setattr(handler, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    handler._handle_send_message(owed.event, context)
    assert sent == []
    handler._handle_send_message(fresh, context)
    assert len(sent) == 1 and sent[0][1]["require_write"] is True
    assert load_task_result(owed.root, "root")["canonical_terminal_projection_ready"] is None
    assert delivery.already_delivered(owed.root, fresh["delivery_id"], task_id="root",
                                      publication_token="new-publication")
    assert not delivery.already_delivered(owed.root, fresh["delivery_id"], task_id="other",
                                          publication_token="new-publication")  # the typed task, not the id, names it


@pytest.mark.parametrize("branch", ["final", "cancel", "cascade"])
def test_every_terminal_delivery_id_requires_durable_chat_before_ack(tmp_path, monkeypatch, branch):
    """Exercise real producer events and the shared handler, including generic outbox callers."""
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.utils import append_jsonl
    from supervisor import events_chat_delivery as handler, message_bus as bus

    task = {"id": "task", "chat_id": 1, "root_task_id": "task"}
    write_task_result(tmp_path, "task", "running", chat_id=1, root_task_id="task")
    queued, frames = [], []
    monkeypatch.setattr("supervisor.workers.get_event_q", lambda *, create=True: SimpleNamespace(put=queued.append))
    if branch == "final":
        stored = write_task_result(tmp_path, "task", "completed", result="Model's final answer",
                                   terminal_origin="model_final")
        assert delivery.deliver_completed_result(tmp_path, task, "task", stored)
    else:
        request_cancel(tmp_path, "task", scope="tree" if branch == "cascade" else "task")
        write_task_result(tmp_path, "task", "cancelled", result="Stopped")
        if branch == "cascade":
            assert delivery.deliver_cascade_summary(tmp_path, "task", task, {"task": "cancelled"})
        else:
            assert delivery.deliver_unreviewed_salvage(tmp_path, task, "task", outcome="cancelled",
                                                       salvaged_text="Preserved intermediate work")
    [event] = queued
    did = event["delivery_id"]
    assert did.startswith(f"{branch}:")
    bridge = bus.LocalChatBridge({})
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(bus, "load_state", lambda: {"owner_id": 1})
    monkeypatch.setattr(bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(bus, "publish_event", lambda *_a, **_kw: None)
    monkeypatch.setattr(handler, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    context = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, append_jsonl=append_jsonl,
                              send_with_budget=bus.send_with_budget)
    real_append = dialogue_store.append
    monkeypatch.setattr(dialogue_store, "append", lambda *_a, **_kw: (_ for _ in ()).throw(store.StoreBusy("refused")))
    handler._handle_send_message(event, context)
    assert frames == []
    assert not delivery.already_delivered(tmp_path, did)
    assert did in {row["delivery_id"] for row in delivery.pending_deliveries(tmp_path)}
    stored = load_task_result(tmp_path, "task")
    assert not (stored.get("cancel_receipt") or {}).get("delivered_chat_id")
    assert not stored.get("canonical_terminal_projection_ready")

    monkeypatch.setattr(dialogue_store, "append", real_append)
    handler._handle_send_message(event, context)
    assert len(frames) == 1 and frames[0]["delivery_id"] == did
    assert delivery.already_delivered(tmp_path, did)
    assert did not in {row["delivery_id"] for row in delivery.pending_deliveries(tmp_path)}
    if branch != "final":
        assert load_task_result(tmp_path, "task")["cancel_receipt"]["delivered_chat_id"] == 1
    handler._handle_send_message(event, context)
    assert len(frames) == 1


@pytest.mark.parametrize("state,path", [("unknown", ""), ("exhausted", ""), ("exhausted", "saved/answer.md")])
def test_main_mirror_qualifies_source_delivery_without_mutating_result_axes(tmp_path, monkeypatch, state, path):
    from ouroboros import terminal_projection
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from tests._cancel_intents_shared import DeliveryBus

    project = create_project(tmp_path, "source", name="Source")
    bind_task_to_project(tmp_path, "root", project["id"], origin={"absent": "system"})
    write_task_result(tmp_path, "root", "completed", root_task_id="root", project_id=project["id"],
                      chat_id=project["chat_id"], result="Full authored answer", terminal_origin="model_final")
    axes = load_task_result(tmp_path, "root").get("outcome_axes")
    monkeypatch.setattr(terminal_projection, "_delivery_disposition",
                        lambda *a: {"state": state, "preserved_path": path})
    bus = DeliveryBus()
    assert settle_terminal_projection(tmp_path, "root", event_queue=bus) == SETTLEMENT_DEFERRED
    mirror = bus.get_nowait()
    assert mirror["system_type"] == "project_completion_summary"
    assert mirror["progress_meta"]["completion_answer"] == "Full authored answer"
    reason = mirror["progress_meta"]["completion_reason"]
    assert reason.count("Source Project:") == 1
    assert "could not be confirmed" in reason if state == "unknown" else "attempts ran out" in reason
    if path:
        assert path in reason
    elif state == "exhausted":
        assert "no copy" in reason
    assert load_task_result(tmp_path, "root").get("outcome_axes") == axes


def test_capacity_eviction_of_main_cache_does_not_announce_final_abandonment(owed, monkeypatch):
    from ouroboros.terminal_projection import _delivery_disposition
    from supervisor import message_bus

    notices = []
    monkeypatch.setattr(message_bus, "send_with_budget", lambda *a, **kw: notices.append((a, kw)))
    monkeypatch.setattr(delivery, "_PENDING_CAP", 1)
    delivery.register_pending_delivery(owed.root, {"type": "send_message", "delivery_id": "other",
        "task_id": "other", "chat_id": 1, "text": "Another message"})
    assert not notices
    assert _delivery_disposition(owed.root, "root")["state"] == ""
    assert load_task_result(owed.root, "root")["canonical_terminal_projection_ready"]["main_delivery"]["event"] == owed.event
    # Genuine ordinary final loss still produces its notice.
    delivery._disclose_exhausted_delivery(owed.root, {"type": "send_message", "task_id": "control",
        "delivery_id": "final:control:0123456789abcdef", "chat_id": 1, "text": "Ordinary answer"}, reason="outbox_capacity")
    assert notices and "not confirmed" in notices[-1][0][1]
