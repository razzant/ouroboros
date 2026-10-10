"""The #198 routing picker: durable refusal row -> owner click -> the SAME
supervisor handlers the LLM routing tools use, confirmed by the SAME durable
receipts. Everything here runs against real files under tmp_path; only the
worker-event queue is faked (its handlers live in the supervisor process)."""

import json

import pytest

from ouroboros.gateway.routing_decision import (
    _derived_identity,
    handle_routing_decision,
    parse_routing_decision_id,
)
from ouroboros.project_dialogue import (
    append_chat_annotation,
    build_owner_message_ref,
    chat_annotation_receipt,
)

OPTIONS = [
    {"action": "steer_task", "task_id": "t-live", "label": "Fix CI"},
    {"action": "new_task_in_project", "project_id": "p1", "label": "New task in Web"},
]


def _seed_refusal(root, cmid="cm-1", token="tok-1", options=OPTIONS, **extra):
    assert append_chat_annotation(
        root, cmid, action="route_decision", status="needs_manual_target",
        routing_token=token, options=options, **extra,
    )


def _seed_origin(root, cmid="cm-1", text="original owner words", chat_id=0):
    path = root / "logs" / "chat.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "direction": "in", "client_message_id": cmid,
            "text": text, "chat_id": chat_id,
            "ts": "2026-10-06T12:00:00+00:00",
        }) + "\n")


class _Queue:
    def __init__(self, on_put=None):
        self.events = []
        self._on_put = on_put

    def put_nowait(self, evt):
        self.events.append(evt)
        if self._on_put:
            self._on_put(evt)


def _wire_queue(monkeypatch, queue):
    import supervisor.workers as workers

    monkeypatch.setattr(workers, "get_event_q", lambda: queue)


def test_decision_id_parse_keeps_colons_inside_the_message_id():
    assert parse_routing_decision_id("routing:cm:with:colons:tok") == ("cm:with:colons", "tok")
    assert parse_routing_decision_id("routing:cm-1:tok-1") == ("cm-1", "tok-1")
    for bad in ("", "routing:", "routing:cm", "quiz:cm:tok", "routing::tok"):
        assert parse_routing_decision_id(bad) == ("", "")


def test_derived_identity_is_deterministic_and_task_shaped():
    token_a, task_a = _derived_identity("cm-1", "tok-1", 0)
    token_b, task_b = _derived_identity("cm-1", "tok-1", 0)
    assert (token_a, task_a) == (token_b, task_b)
    assert len(task_a) == 16 and int(task_a, 16) >= 0
    assert _derived_identity("cm-1", "tok-1", 1) != (token_a, task_a)


def test_malformed_and_unknown_rows_refuse_honestly(tmp_path):
    status, body = handle_routing_decision(
        tmp_path, request_id="r1", decision_id="routing::", option_index=0)
    assert (status, body["error"]) == (400, "malformed_decision_id")
    # No refusal row at all -> the card settles as superseded, never retries.
    status, body = handle_routing_decision(
        tmp_path, request_id="r1", decision_id="routing:cm-1:tok-1", option_index=0)
    assert (status, body["state"]) == (409, "superseded")


def test_option_bounds_and_undispatchable_rows(tmp_path):
    _seed_refusal(tmp_path, options=OPTIONS + [{"action": "answer_inline"}])
    for bad_index in (-1, 3, "0"):
        status, body = handle_routing_decision(
            tmp_path, request_id="r1", decision_id="routing:cm-1:tok-1",
            option_index=bad_index)
        assert (status, body["error"]) == (400, "option_out_of_range")
    status, body = handle_routing_decision(
        tmp_path, request_id="r1", decision_id="routing:cm-1:tok-1", option_index=2)
    assert (status, body["error"]) == (400, "option_not_dispatchable")


def test_missing_origin_text_settles_instead_of_forging_a_message(tmp_path, monkeypatch):
    _seed_refusal(tmp_path)
    _wire_queue(monkeypatch, _Queue())
    status, body = handle_routing_decision(
        tmp_path, request_id="r1", decision_id="routing:cm-1:tok-1", option_index=0)
    assert (status, body["error"]) == (409, "origin_text_unavailable")


def test_steer_click_dispatches_the_verbatim_message_and_settles(tmp_path, monkeypatch):
    _seed_refusal(tmp_path, attachment_manifest=[{"path": "/up/a.png", "label": "a.png"}])
    _seed_origin(tmp_path, text="please fix the CI flake")
    dispatch_token, _ = _derived_identity("cm-1", "tok-1", 0)

    def _supervisor_delivers(evt):
        # The real steer handler appends the delivered receipt under the
        # DISPATCH token; the wait below reads exactly that seam.
        append_chat_annotation(
            tmp_path, "cm-1", action="steer_task", target="t-live",
            status="delivered", routing_token=evt["routing_token"],
        )

    queue = _Queue(on_put=_supervisor_delivers)
    _wire_queue(monkeypatch, queue)
    status, body = handle_routing_decision(
        tmp_path, request_id="r1", decision_id="routing:cm-1:tok-1",
        option_index=0, comment="prefer a revert")
    assert status == 200 and body["dispatched"] == "delivered"
    assert body["answered_index"] == 0
    (evt,) = queue.events
    assert evt["type"] == "steer_task" and evt["target_task_id"] == "t-live"
    assert evt["routing_token"] == dispatch_token
    assert evt["message"].startswith("please fix the CI flake")
    assert "[Owner picker comment] prefer a revert" in evt["message"]
    assert evt["attachment_uploads"] == [{"path": "/up/a.png", "label": "a.png"}]
    # Origin provenance rides BY VALUE, same rail as the LLM promote path.
    assert evt["source_ref"]["client_message_id"] == "cm-1"
    assert evt["source_text"] == "please fix the CI flake"
    # The closing row under the ORIGINAL token is the replay's confirmation.
    closing = chat_annotation_receipt(tmp_path, "cm-1", "tok-1")
    assert closing["status"] == "delivered" and closing["detail"] == "request:r1"
    # Same request replays as its own confirmation; a different click loses.
    status, body = handle_routing_decision(
        tmp_path, request_id="r1", decision_id="routing:cm-1:tok-1", option_index=1)
    assert status == 200 and body["duplicate"] is True
    status, body = handle_routing_decision(
        tmp_path, request_id="r2", decision_id="routing:cm-1:tok-1", option_index=1)
    assert (status, body["error"]) == (409, "decision_closed")
    assert len(queue.events) == 1  # neither replay nor loser re-dispatched


def _manual_picker(tmp_path, monkeypatch, **effort):
    """The real tool/annotation/WS producer, consumed later via the HTTP door."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway import task_decision
    from ouroboros.gateway.history import _user_annotation
    from ouroboros.project_dialogue import latest_chat_annotations
    from ouroboros.projects_registry import create_project
    from ouroboros.tools.control import _route_to_project
    from supervisor import events, message_bus
    from tests.test_root_effort_ingress import (
        _install_queue, _pool_ready, _supervisor_ctx, _tool_ctx,
    )

    if "reasoning_effort" in effort:  # a root Ouroboros creates itself takes an explicit effort in Cyber Pro only
        monkeypatch.setattr("ouroboros.config._BOOT_RUNTIME_MODE", "cyber_pro")
    q, _state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_ready(monkeypatch, workers)
    workspace = tmp_path.with_name(tmp_path.name + "-project")
    workspace.mkdir()
    create_project(tmp_path, "p1", name="Web", working_dir=str(workspace))
    sup = _supervisor_ctx(tmp_path, workers)
    sup.persist_queue_snapshot = q.persist_queue_snapshot
    frames = []
    sup.bridge = message_bus.LocalChatBridge.__new__(message_bus.LocalChatBridge)
    sup.bridge._broadcast_fn = frames.append
    sup.bridge._chat_transports = {}
    monkeypatch.setattr(message_bus, "publish_event", lambda *a, **k: None)
    monkeypatch.setattr(message_bus, "log_chat", lambda *a, **k: None)
    handlers = {
        "routing_manual_target": events._handle_routing_manual_target,
        "promote_chat_to_task": events._handle_promote_chat_to_task,
        "steer_task": events._handle_steer_task,
    }
    queue = _Queue(on_put=lambda evt: handlers[evt["type"]](evt, sup))
    _wire_queue(monkeypatch, queue)
    ctx = _tool_ctx(tmp_path, sup, client_message_id="cm-1",
                    routing_contract={"manual_options": OPTIONS})
    ctx.is_direct_chat, ctx.event_queue = True, queue
    _seed_origin(tmp_path, chat_id=7)
    result = _route_to_project(ctx, message="original owner words", predecessor_task_id="", **effort)
    assert "NEEDS_MANUAL_TARGET" in result
    [manual] = queue.events
    card = _user_annotation("user", "cm-1", latest_chat_annotations(tmp_path))
    assert card["routing_token"] == manual["routing_token"]
    [frame] = frames
    assert frame["type"] == "message_annotation"
    assert frame.get("reasoning_effort") == card.get("reasoning_effort") == manual.get("reasoning_effort")
    monkeypatch.setattr(task_decision, "request_drive_root", lambda request: tmp_path)
    app = Starlette(routes=[Route("/api/decisions", task_decision.api_decision_answer, methods=["POST"])])
    click = {"request_id": "r1", "decision_id": f"routing:cm-1:{card['routing_token']}"}
    return TestClient(app), click, card, queue, workers, frames


@pytest.mark.parametrize("requested,expected", [("XHigh ", "xhigh"), (None, None)])
def test_promote_click_confirms_from_the_admission_record(tmp_path, monkeypatch, requested, expected):
    from ouroboros.gateway.history import _user_annotation
    from ouroboros.task_results import load_task_result

    client, click, card, queue, workers, _frames = _manual_picker(
        tmp_path, monkeypatch, **({"reasoning_effort": requested} if requested is not None else {}))
    assert card.get("reasoning_effort") == expected
    click["option_index"] = 1
    dispatch_token, task_id = _derived_identity("cm-1", card["routing_token"], 1)
    # A delayed supervisor leaves only the durable pending claim. Reload and
    # replay it through the same API, then let the real handler consume it.
    from ouroboros import routing_wait

    dispatch = queue._on_put
    with monkeypatch.context() as delayed:
        delayed.setattr(queue, "_on_put", None)
        delayed.setattr(routing_wait, "wait_for_promotion_admission",
                        lambda *a, **k: {"status": "unconfirmed"})
        pending = client.post("/api/decisions", json=click)
    assert pending.status_code == 503 and pending.json()["error"] == "dispatch_unconfirmed"
    assert queue._on_put is dispatch and workers.PENDING == []
    pending_row = chat_annotation_receipt(tmp_path, "cm-1", card["routing_token"])
    assert pending_row["status"] == "dispatch_pending"
    assert _user_annotation("user", "cm-1", {"cm-1": pending_row}).get("reasoning_effort") == expected
    response = client.post("/api/decisions", json=click)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["dispatched"] == "scheduled" and body["task_id"] == task_id
    assert body.get("reasoning_effort") == expected
    [task] = workers.PENDING
    assert task["id"] == task_id and task["project_id"] == "p1"
    row = load_task_result(tmp_path, task_id)
    assert row["status"] == "scheduled"
    assert row["promotion_admission"]["routing_token"] == dispatch_token
    assert row["promotion_admission"]["queue_snapshot_persisted"] is True
    assert row.get("reasoning_effort") == task.get("reasoning_effort") == expected
    for value in (card, pending_row, body, row, task):
        assert ("reasoning_effort" in value) is (expected is not None)
    from supervisor import queue as supervisor_queue

    snapshot = json.loads(supervisor_queue.QUEUE_SNAPSHOT_PATH.read_text())
    assert snapshot["pending"][0]["id"] == task_id
    assert snapshot["pending"][0]["task"].get("reasoning_effort") == expected
    closing = chat_annotation_receipt(tmp_path, "cm-1", card["routing_token"])
    hydrated = _user_annotation("user", "cm-1", {"cm-1": closing})
    assert hydrated.get("reasoning_effort") == expected
    assert queue.events[-1]["host_initiated"] is True
    replay = client.post("/api/decisions", json=click)
    assert replay.status_code == 200 and replay.json()["duplicate"] is True
    assert queue.events[1]["routing_token"] == queue.events[2]["routing_token"]
    assert queue.events[1]["task_id"] == queue.events[2]["task_id"]
    assert len(queue.events) == 3 and len(workers.PENDING) == 1
    assert load_task_result(tmp_path, task_id) == row


def test_existing_task_picker_click_does_not_apply_or_report_new_root_effort(tmp_path, monkeypatch):
    from ouroboros.gateway.history import _user_annotation
    from ouroboros.owner_mailbox import _mailbox_path
    from ouroboros.task_results import load_task_result, write_task_result

    client, click, card, queue, workers, frames = _manual_picker(tmp_path, monkeypatch, reasoning_effort="max")
    workers.PENDING.append({"id": "t-live", "type": "task", "chat_id": 7,
                            "root_task_id": "t-live", "reasoning_effort": "low"})
    write_task_result(tmp_path, "t-live", "scheduled", reasoning_effort="low", root_task_id="t-live")
    before = load_task_result(tmp_path, "t-live")
    click["option_index"] = 0
    response = client.post("/api/decisions", json=click)
    assert response.status_code == 200, response.text
    assert response.json()["dispatched"] == "delivered"
    assert "reasoning_effort" not in response.json()
    assert queue.events[-1]["type"] == "steer_task" and "reasoning_effort" not in queue.events[-1]
    assert len(workers.PENDING) == 1 and workers.PENDING[0]["reasoning_effort"] == "low"
    assert load_task_result(tmp_path, "t-live") == before
    mailbox = _mailbox_path(tmp_path, "t-live")
    [mail] = [json.loads(line) for line in mailbox.read_text().splitlines()]
    assert mail["text"] == "original owner words"
    assert mail["kind"] == "owner_text"
    closing = chat_annotation_receipt(tmp_path, "cm-1", card["routing_token"])
    assert "reasoning_effort" not in closing
    assert frames[-1]["status"] == "delivered" and "reasoning_effort" not in frames[-1]

    hydrated = _user_annotation("user", "cm-1", {"cm-1": closing})
    assert "reasoning_effort" not in hydrated
    replay = client.post("/api/decisions", json=click)
    assert replay.status_code == 200 and replay.json()["duplicate"] is True
    assert "reasoning_effort" not in replay.json()
    assert len(queue.events) == 2 and len(mailbox.read_text().splitlines()) == 1


def test_dead_queue_returns_a_retriable_503(tmp_path, monkeypatch):
    import supervisor.workers as workers

    _seed_refusal(tmp_path)
    _seed_origin(tmp_path)

    def _broken():
        raise RuntimeError("supervisor down")

    monkeypatch.setattr(workers, "get_event_q", _broken)
    status, body = handle_routing_decision(
        tmp_path, request_id="r1", decision_id="routing:cm-1:tok-1", option_index=0)
    assert (status, body["error"]) == (503, "dispatch_unavailable")
    # The refusal row is untouched: the card stays open for a real retry.
    assert chat_annotation_receipt(tmp_path, "cm-1", "tok-1")["status"] == "needs_manual_target"


def test_ingress_routes_the_routing_family(tmp_path, monkeypatch):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway import task_decision as td

    monkeypatch.setattr(td, "request_drive_root", lambda request: tmp_path)
    app = Starlette(routes=[Route("/api/decisions", endpoint=td.api_decision_answer,
                                  methods=["POST"])])
    res = TestClient(app).post("/api/decisions", json={
        "request_id": "r1", "decision_id": "routing:cm-1:tok-1", "option_index": 0,
    })
    # No refusal row exists in this drive: the routing handler answered (409
    # superseded), which proves the family is served end-to-end, not 501.
    assert res.status_code == 409
    assert res.json()["state"] == "superseded"


def test_manual_target_refusal_persists_the_attachment_manifest(tmp_path):
    """The producer half: _handle_routing_manual_target must store the routing
    turn's staged-attachment specs on the durable refusal row (#198)."""
    from supervisor.events import _handle_routing_manual_target

    class _Ctx:
        DRIVE_ROOT = tmp_path

        @staticmethod
        def append_jsonl(path, row):
            pass

    evt = {
        "type": "routing_manual_target", "routing_token": "tok-9",
        "chat_id": 0, "client_message_id": "cm-9",
        "reason": "target_unspecified", "options": OPTIONS,
        "reasoning_effort": "high",
        "attachment_uploads": [{"path": "/up/b.pdf", "label": "b.pdf"}],
        "ts": "2026-08-31T00:00:00Z",
    }
    _handle_routing_manual_target(evt, _Ctx)
    receipt = chat_annotation_receipt(tmp_path, "cm-9", "tok-9")
    assert receipt["status"] == "needs_manual_target"
    assert receipt["reasoning_effort"] == "high"
    assert receipt["attachment_manifest"] == [{"path": "/up/b.pdf", "label": "b.pdf"}]
    assert [row["action"] for row in receipt["options"]] == [
        "steer_task", "new_task_in_project",
    ]


def test_route_to_project_candidates_reorder_is_host_validated(tmp_path, monkeypatch):
    """Owner decision 2=B: `candidates` reorders the host-built option list —
    named ids come first, unknown ids vanish, nothing new is invented."""
    import types

    from ouroboros.tools import control

    captured = {}

    def _capture(ctx, evt):
        captured.update(evt)
        return "wired", {"status": "needs_manual_target", "options": evt["options"]}

    # Campaign owner: _route_to_project lives in control_routing, which froze
    # its _emit_and_wait_for_routing binding at import time.
    from ouroboros.tools import control_routing

    monkeypatch.setattr(control_routing, "_emit_and_wait_for_routing", _capture)
    manual = [
        {"action": "steer_task", "task_id": "t-a", "label": "A"},
        {"action": "steer_task", "task_id": "t-b", "label": "B"},
        {"action": "new_task_in_project", "project_id": "p1", "label": "New in P1"},
    ]
    ctx = types.SimpleNamespace(
        current_chat_id=1, drive_root=tmp_path, is_direct_chat=True,
        task_metadata={"client_message_id": "cm-1",
                       "origin_message_ref": build_owner_message_ref(
                           chat_id=1, client_message_id="cm-1", ts="2026-09-24T00:00:00+00:00", text="route me"),
                       "routing_contract": {"manual_options": manual}},
    )
    text = control._route_to_project(
        ctx, "", "route me", predecessor_task_id="",
        candidates=["p1", "ghost-id", "t-a"],
    )
    assert "NEEDS_MANUAL_TARGET" in text
    ordered = [row.get("task_id") or row.get("project_id") for row in captured["options"]]
    assert ordered == ["p1", "t-a", "t-b"]  # candidates first, rest kept, ghost ignored


def test_click_identity_reaches_both_presentation_paths(monkeypatch):
    """C1 wiring pin: the refusal's routing_token must survive BOTH surfaces
    the picker card is built from — the live WS ack and the history replay
    projection. Without it the card cannot compose its decision_id."""
    from ouroboros.gateway.history import _user_annotation
    from supervisor.message_bus import LocalChatBridge

    projected = _user_annotation("user", "cm-1", {"cm-1": {
        "status": "needs_manual_target", "routing_token": "tok-1",
        "options": [{"action": "steer_task", "task_id": "t1"}], "action": "route_decision",
    }})
    assert projected["routing_token"] == "tok-1"

    published = []
    import supervisor.message_bus as mb

    monkeypatch.setattr(mb, "publish_event", lambda topic, evt: published.append(evt))
    bus = LocalChatBridge.__new__(LocalChatBridge)
    ws_frames = []
    bus._broadcast_fn = ws_frames.append
    bus._chat_transports = {}
    bus.send_routing_ack(
        0, client_message_id="cm-1", action="route_decision",
        status="needs_manual_target", options=[{"action": "steer_task", "task_id": "t1"}],
        routing_token="tok-1", reasoning_effort="max",
    )
    (frame,) = ws_frames
    assert frame["routing_token"] == "tok-1"
    assert frame["reasoning_effort"] == "max"
    assert frame["type"] == "message_annotation"
    (bus_evt,) = published
    assert bus_evt["routing_token"] == "tok-1"
    assert bus_evt["reasoning_effort"] == "max"


def test_rejected_dispatch_reopens_the_original_card(tmp_path, monkeypatch):
    """C2: the handler's rejection receipt lands under the DISPATCH token; the
    gateway re-asserts the refusal under the ORIGINAL token so 'pick another'
    is a real invitation — the next click still validates and dispatches."""
    _seed_refusal(tmp_path, attachment_manifest=[{"path": "/up/a.png", "label": "a"}])
    _seed_origin(tmp_path)

    def _supervisor_rejects(evt):
        append_chat_annotation(
            tmp_path, "cm-1", action="steer_task", target="t-live",
            status="needs_manual_target", routing_token=evt["routing_token"],
            reason="target_closed",
        )

    queue = _Queue(on_put=_supervisor_rejects)
    _wire_queue(monkeypatch, queue)
    status, body = handle_routing_decision(
        tmp_path, request_id="r1", decision_id="routing:cm-1:tok-1", option_index=0)
    assert (status, body["state"]) == (409, "open")
    # R5/R16: the toast shows the host's sentence for the refused act, not the code.
    assert body["reason"] == "target_closed"
    assert body["cause"] == "Not delivered: that task has already finished"
    reopened = chat_annotation_receipt(tmp_path, "cm-1", "tok-1")
    assert reopened["status"] == "needs_manual_target"
    assert [row["action"] for row in reopened["options"]] == [
        "steer_task", "new_task_in_project"]
    assert reopened["attachment_manifest"] == [{"path": "/up/a.png", "label": "a"}]


def test_competing_click_refused_while_dispatch_pending(tmp_path, monkeypatch):
    """M3: first-wins BEFORE the side effect — while r1's dispatch is
    unconfirmed, a different request cannot dispatch a second event; r1's
    replay re-enters and settles."""
    import ouroboros.routing_wait as rw

    _seed_refusal(tmp_path)
    _seed_origin(tmp_path)
    queue = _Queue()
    _wire_queue(monkeypatch, queue)
    monkeypatch.setattr(rw, "wait_for_routing_annotation",
                        lambda *a, **k: {"status": "unconfirmed"})
    status, body = handle_routing_decision(
        tmp_path, request_id="r1", decision_id="routing:cm-1:tok-1", option_index=0)
    assert (status, body["error"]) == (503, "dispatch_unconfirmed")
    assert len(queue.events) == 1
    # A competing click is refused without dispatching anything.
    status, body = handle_routing_decision(
        tmp_path, request_id="r2", decision_id="routing:cm-1:tok-1", option_index=1)
    assert (status, body["error"], body["state"]) == (409, "dispatch_in_flight", "pending")
    assert len(queue.events) == 1
    # The winner's replay re-dispatches the SAME identity and settles.
    monkeypatch.setattr(rw, "wait_for_routing_annotation",
                        lambda *a, **k: {"status": "delivered"})
    status, body = handle_routing_decision(
        tmp_path, request_id="r1", decision_id="routing:cm-1:tok-1", option_index=0)
    assert status == 200 and body["dispatched"] == "delivered"
    assert len(queue.events) == 2
    assert queue.events[0]["routing_token"] == queue.events[1]["routing_token"]


def test_actionable_refusal_persists_the_numbered_list_for_the_router(monkeypatch, tmp_path):
    """Owner decision 4=A: a plain '2' reply must ground against EXACTLY the
    list the owner was shown — the bus persists it as a durable outbound chat
    row the router's Recent chat renders (web history skips the typed row)."""
    import supervisor.message_bus as mb
    from supervisor.message_bus import LocalChatBridge

    logged = []
    monkeypatch.setattr(mb, "publish_event", lambda topic, evt: None)
    monkeypatch.setattr(mb, "log_chat", lambda *a, **k: logged.append((a, k)))
    bus = LocalChatBridge.__new__(LocalChatBridge)
    bus._broadcast_fn = None
    bus._chat_transports = {}
    bus.send_routing_ack(
        0, client_message_id="cm-1", action="route_decision",
        status="needs_manual_target", routing_token="tok-1",
        options=[{"action": "steer_task", "task_id": "t1", "label": "Fix CI"},
                 {"action": "new_task_in_project", "project_id": "p1",
                  "project_name": "Web"}],
    )
    ((args, kwargs),) = logged
    assert args[0] == "out"
    assert "1. Fix CI" in args[3] and "2. New task in Web" in args[3]
    assert kwargs["record_type"] == "routing_options"
    # A settled ack persists nothing.
    bus.send_routing_ack(0, client_message_id="cm-1", action="steer_task",
                         status="delivered", routing_token="tok-2")
    assert len(logged) == 1


def test_same_request_id_cannot_switch_options_and_stale_tokens_cannot_claim(
    tmp_path, monkeypatch,
):
    """Scope findings: the claim binds (request_id AND option); the CAS binds
    the token, so neither a same-id different-option replay nor a click on a
    superseded card can dispatch a second identity."""
    import ouroboros.routing_wait as rw

    _seed_refusal(tmp_path)
    _seed_origin(tmp_path)
    queue = _Queue()
    _wire_queue(monkeypatch, queue)
    monkeypatch.setattr(rw, "wait_for_routing_annotation",
                        lambda *a, **k: {"status": "unconfirmed"})
    status, body = handle_routing_decision(
        tmp_path, request_id="r1", decision_id="routing:cm-1:tok-1", option_index=0)
    assert (status, body["error"]) == (503, "dispatch_unconfirmed")
    # Same id, DIFFERENT option: refused, nothing new dispatched.
    status, body = handle_routing_decision(
        tmp_path, request_id="r1", decision_id="routing:cm-1:tok-1", option_index=1)
    assert (status, body["error"]) == (409, "request_option_mismatch")
    assert len(queue.events) == 1
    # A NEWER routing attempt re-mints the card under tok-2: the stale tok-1
    # card can no longer claim (its receipt is gone -> superseded), and the
    # fresh card claims fine.
    _seed_refusal(tmp_path, token="tok-2")
    status, body = handle_routing_decision(
        tmp_path, request_id="r9", decision_id="routing:cm-1:tok-1", option_index=0)
    assert (status, body["state"]) == (409, "superseded")
    monkeypatch.setattr(rw, "wait_for_routing_annotation",
                        lambda *a, **k: {"status": "delivered"})
    status, body = handle_routing_decision(
        tmp_path, request_id="r2", decision_id="routing:cm-1:tok-2", option_index=0)
    assert status == 200 and len(queue.events) == 2


def test_grounding_row_never_consumes_the_history_quota():
    """Scope finding: the hidden routing_options row is skipped by the render
    loop, so it must not count toward the human-row quota either."""
    from ouroboros.gateway.history import _chat_quota_predicate

    counts = _chat_quota_predicate(lambda chat_id, entry: True)
    assert counts({"direction": "out", "chat_id": 1, "text": "hi"})
    assert not counts({"direction": "out", "chat_id": 1,
                       "type": "routing_options", "text": "1. A"})


def test_compare_and_append_refuses_under_the_lock(tmp_path):
    """Delta-review follow-up: the CAS branches inside append_chat_annotation
    — losing the status race and hitting a foreign token — refuse WITHOUT
    writing, and an unconditional append still works."""
    _seed_refusal(tmp_path, token="tok-1")
    # Status mismatch: latest is needs_manual_target, caller requires closed.
    assert not append_chat_annotation(
        tmp_path, "cm-1", action="route_decision", status="dispatch_pending",
        routing_token="tok-1", require_latest_status={"delivered"},
    )
    # Token mismatch: a newer attempt owns the card.
    assert not append_chat_annotation(
        tmp_path, "cm-1", action="route_decision", status="dispatch_pending",
        routing_token="tok-0", require_latest_status={"needs_manual_target"},
        require_latest_token={"tok-0"},
    )
    untouched = chat_annotation_receipt(tmp_path, "cm-1", "tok-1")
    assert untouched["status"] == "needs_manual_target"
    # Matching guards write; unconditional append never checks.
    assert append_chat_annotation(
        tmp_path, "cm-1", action="route_decision", status="dispatch_pending",
        routing_token="tok-1", require_latest_status={"needs_manual_target"},
        require_latest_token={"tok-1"},
    )
    assert append_chat_annotation(
        tmp_path, "cm-1", action="route_decision", status="needs_manual_target",
        routing_token="tok-1",
    )


# --- the deciding turn is SHOWN an existing receipt (disclosure, never a gate) ---

def _decision_ctx(tmp_path):
    import types

    return types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        PENDING=[],
        RUNNING={},
        load_state=lambda: {"owner_id": 1, "owner_chat_id": 1},
        update_state=lambda fn: fn({"owner_id": 1, "owner_chat_id": 1}),
    )


def test_a_turn_nobody_typed_is_given_the_same_main_lane_facts(tmp_path):
    """P3c: a consciousness wake-up has no owner message, so nothing used to build its
    Main manifest and every predecessor it named was refused as not addressable. The
    same facts now come through ONE seam over the owner path, never a second copy that
    could drift; only what an owner MESSAGE carries is absent."""
    from ouroboros.projects_registry import create_project
    from ouroboros.server_routing_context import _decision_turn_metadata, main_lane_routing_metadata

    create_project(tmp_path, "racer", name="Racer")
    ctx = _decision_ctx(tmp_path)

    owner = _decision_turn_metadata(ctx, 1, "cm-owner", {})
    wake = main_lane_routing_metadata(ctx, 1)

    assert wake["main_routing_manifest"] == owner["main_routing_manifest"]
    assert [row["project_id"] for row in wake["main_routing_manifest"]["projects"]] == ["racer"]
    assert wake["routing_contract"]["source_lane"] == "main"
    assert "client_message_id" not in wake


def test_decision_turn_is_shown_the_existing_receipt_for_the_same_message(tmp_path):
    """I7 (owner decision B5=A): one owner message became task c405c824 and was then
    steered into three more live roots, each paying a review wave, because the
    deciding turn was never told a receipt already existed. It is a FACT on the
    contract the turn already receives; the choice stays with the model."""
    from ouroboros.server_routing_context import _decision_turn_metadata

    append_chat_annotation(
        tmp_path, "cm-dup", action="promote_chat_to_task", target="c405c824",
        target_label="MLConf deck", status="dispatched", routing_token="tok-dup",
    )

    metadata = _decision_turn_metadata(_decision_ctx(tmp_path), 1, "cm-dup", {})
    receipt = metadata["routing_contract"]["message_routing_receipt"]

    assert receipt["action"] == "promote_chat_to_task"
    assert receipt["target"] == "c405c824"
    assert receipt["target_label"] == "MLConf deck"
    assert receipt["status"] == "dispatched"
    assert receipt["ts"]
    # Disclosure only: the model keeps every action it had.
    assert "promote_chat_to_task" in metadata["routing_contract"]["valid_actions"]


def test_decision_turn_without_a_receipt_carries_no_such_key(tmp_path):
    from ouroboros.server_routing_context import _decision_turn_metadata

    metadata = _decision_turn_metadata(_decision_ctx(tmp_path), 1, "cm-fresh", {})

    assert "message_routing_receipt" not in metadata["routing_contract"]


def test_unreadable_annotations_do_not_break_the_decision_turn(tmp_path):
    from ouroboros.server_routing_context import _decision_turn_metadata

    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs" / "chat_annotations.jsonl").write_text("{ torn", encoding="utf-8")

    metadata = _decision_turn_metadata(_decision_ctx(tmp_path), 1, "cm-dup", {})

    assert "message_routing_receipt" not in metadata["routing_contract"]
    assert metadata["routing_contract"]["llm_first"] is True


def test_decision_turn_reads_every_recorded_act_on_the_same_message_as_facts(tmp_path):
    """The latest receipt alone hid an earlier act on the same owner message (a promote,
    then a steer relaying it): each act keeps its own receipt, listed oldest first, and
    the contract says these are facts, not a ban."""
    from ouroboros.server_routing_context import _decision_turn_metadata

    append_chat_annotation(tmp_path, "cm-fan", action="promote_chat_to_task", target="root-a",
                           target_label="Deck", status="dispatched", routing_token="tok-1")
    append_chat_annotation(tmp_path, "cm-fan", action="steer_task", target="root-b",
                           target_label="Build", status="delivered", routing_token="tok-2")
    append_chat_annotation(tmp_path, "cm-other", action="steer_task", target="root-c",
                           target_label="Else", status="delivered", routing_token="tok-3")

    contract = _decision_turn_metadata(_decision_ctx(tmp_path), 1, "cm-fan", {})["routing_contract"]

    assert contract["message_routing_receipt"]["target"] == "root-b"  # the latest act, as before
    assert [(act["action"], act["target"], act["status"]) for act in contract["message_routing_acts"]] == [
        ("promote_chat_to_task", "root-a", "dispatched"), ("steer_task", "root-b", "delivered")]
    assert "not a ban" in contract["message_routing_acts_note"]
    assert set(contract["valid_actions"]) >= {"promote_chat_to_task", "steer_task"}
    single = _decision_turn_metadata(_decision_ctx(tmp_path), 1, "cm-other", {})["routing_contract"]
    assert "message_routing_acts" not in single and single["message_routing_receipt"]["target"] == "root-c"


def _boundary(tmp_path, metadata, messages, *, delivery=None):
    from types import SimpleNamespace

    return SimpleNamespace(tools=SimpleNamespace(_ctx=SimpleNamespace(
        task_metadata=metadata, last_owner_delivery=delivery)), messages=messages, drive_root=tmp_path)


def test_routing_acts_taken_during_the_turn_reach_its_next_model_boundary_as_append_only_facts(tmp_path):
    """The decision metadata is captured once, at start; an act recorded since — the turn's own
    steer, a status the rail later wrote — is read from the same receipts at the next boundary
    and appended once, never rewritten, never phrased as a ban. Acts with no link to the
    message are named as not listed, not inferred."""
    from ouroboros.loop_model_call import ROUTING_RECEIPTS_HEADER, _append_routing_receipts
    from ouroboros.server_routing_context import _decision_turn_metadata

    append_chat_annotation(tmp_path, "cm-1", action="promote_chat_to_task", target="root-a",
                           target_label="Deck", status="dispatched", routing_token="tok-1")
    metadata = _decision_turn_metadata(_decision_ctx(tmp_path), 1, "cm-1", {})
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "owner text"}]
    ctx = _boundary(tmp_path, metadata, messages)
    assert _append_routing_receipts(ctx) is False  # the startup receipt already said exactly this
    append_chat_annotation(tmp_path, "cm-1", action="steer_task", target="root-b", target_label="Build",
                           status="delivered", routing_token="tok-2")
    append_chat_annotation(tmp_path, "agent-steer:tok-9", action="steer_task", target="root-c",
                           status="delivered", routing_token="tok-9")
    before = [dict(message) for message in messages]
    assert _append_routing_receipts(ctx) is True
    assert messages[:2] == before and len(messages) == 3
    note = messages[-1]["content"]
    assert note.startswith(ROUTING_RECEIPTS_HEADER) and "facts, not a ban" in note
    acts = [line for line in note.splitlines() if line.startswith("- ")]
    assert [line.split(":", 1)[0] for line in acts] == [
        "- promote_chat_to_task → Deck (root-a)", "- steer_task → Build (root-b)"]
    assert "root-c" not in note and "Not listed: an act recorded under its own agent-steer id" in note
    assert _append_routing_receipts(ctx) is False  # unchanged receipts: no second row
    # The rail later records the promote's outcome: a new row; the sent one stays as it was.
    append_chat_annotation(tmp_path, "cm-1", action="promote_chat_to_task", target="root-a",
                           target_label="Deck", status="scheduled", routing_token="tok-1")
    assert _append_routing_receipts(ctx) is True and messages[2]["content"] == note
    assert "scheduled" in messages[3]["content"] and "dispatched" not in messages[3]["content"]
    # An owner message relayed into the turn mid-run brings its own receipts.
    append_chat_annotation(tmp_path, "cm-2", action="steer_task", target="root-d", status="delivered",
                           routing_token="tok-4")
    relayed = _boundary(tmp_path, metadata, messages, delivery={"client_message_id": "cm-2"})
    assert _append_routing_receipts(relayed) is True and "root-d" in messages[-1]["content"]
    child = _boundary(tmp_path, {**metadata, "delegation_role": "subagent"}, [])
    assert _append_routing_receipts(child) is False
