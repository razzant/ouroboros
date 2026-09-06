"""The #198 routing picker: durable refusal row -> owner click -> the SAME
supervisor handlers the LLM routing tools use, confirmed by the SAME durable
receipts. Everything here runs against real files under tmp_path; only the
worker-event queue is faked (its handlers live in the supervisor process)."""

import json


from ouroboros.gateway.routing_decision import (
    _derived_identity,
    handle_routing_decision,
    parse_routing_decision_id,
)
from ouroboros.project_dialogue import append_chat_annotation, chat_annotation_receipt

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


def test_promote_click_confirms_from_the_admission_record(tmp_path, monkeypatch):
    from ouroboros.task_results import task_result_path
    from ouroboros.utils import update_json_locked

    _seed_refusal(tmp_path)
    _seed_origin(tmp_path)
    dispatch_token, derived_task_id = _derived_identity("cm-1", "tok-1", 1)

    def _supervisor_schedules(evt):
        assert evt["task_id"] == derived_task_id

        def _mut(current):
            from ouroboros.contracts.schema_versions import SCHEMA_VERSION_KEY
            from ouroboros.task_result_schema import TASK_RESULT_SCHEMA_VERSION

            updated = dict(current)
            updated["promotion_admission"] = {
                "routing_token": evt["routing_token"], "status": "scheduled",
            }
            updated["status"] = "scheduled"
            # Campaign ABI 7.0: readers QUARANTINE an unstamped row — the fake
            # supervisor must write what a real writer writes.
            updated[SCHEMA_VERSION_KEY] = TASK_RESULT_SCHEMA_VERSION
            return updated

        update_json_locked(task_result_path(tmp_path, evt["task_id"], create=True), _mut)

    _wire_queue(monkeypatch, _Queue(on_put=_supervisor_schedules))
    status, body = handle_routing_decision(
        tmp_path, request_id="r1", decision_id="routing:cm-1:tok-1", option_index=1)
    assert status == 200 and body["dispatched"] == "scheduled"
    assert body["task_id"] == derived_task_id


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
        "attachment_uploads": [{"path": "/up/b.pdf", "label": "b.pdf"}],
        "ts": "2026-08-31T00:00:00Z",
    }
    _handle_routing_manual_target(evt, _Ctx)
    receipt = chat_annotation_receipt(tmp_path, "cm-9", "tok-9")
    assert receipt["status"] == "needs_manual_target"
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
        current_chat_id=1, drive_root=tmp_path,
        task_metadata={"client_message_id": "cm-1",
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
        routing_token="tok-1",
    )
    (frame,) = ws_frames
    assert frame["routing_token"] == "tok-1"
    assert frame["type"] == "message_annotation"
    (bus_evt,) = published
    assert bus_evt["routing_token"] == "tok-1"


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
