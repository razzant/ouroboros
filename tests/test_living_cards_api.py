"""Living cards over HTTP: the read API, the app bytes, pins on Widgets, and owner presses.

A press is the owner's act: recorded first (idempotent per ``request_id``; ``once``
keeps the first), then delivered into the authoring task's mailbox as ``card_input``
(a paused, held or queued author keeps it until it reads it) or, once that task ended,
as the owner's own message in the card's chat.
"""

from __future__ import annotations

import json

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.card_store import load_card, store_app_bytes, write_card
from ouroboros.gateway.cards import PINNED_CARDS_SKILL, card_routes

ACTION = {"type": "action", "label": "Book it", "submit": {"id": "book"}}
FORM = {"type": "form", "title": "Dates", "fields": [{"name": "from"}, {"name": "to"}],
        "submit": {"id": "dates", "once": True}}


def _component_card(root, *, chat_id=1, task_id="task-1"):
    return write_card(root, chat_id=chat_id, task_id=task_id, title="Flights", fallback_text="A or B",
                      kind="components", body={"components": [ACTION, FORM], "data": {"rows": []}})


def _app_card(root, page=b"<!doctype html><script>alert(1)</script>"):
    return write_card(root, chat_id=1, task_id="task-1", title="Dice", fallback_text="A dice roller",
                      kind="app", body={"app": store_app_bytes(root, page)})


@pytest.fixture
def client(tmp_path, monkeypatch):
    import ouroboros.gateway.ws as ws
    from ouroboros.gateway.widgets import api_widgets

    broadcasts = []

    async def _capture(message):
        broadcasts.append(message)

    monkeypatch.setattr(ws, "broadcast_ws", _capture)
    app = Starlette(routes=[*card_routes(), Route("/api/widgets", endpoint=api_widgets, methods=["GET"])])
    app.state.drive_root = tmp_path
    test_client = TestClient(app)
    test_client.broadcasts = broadcasts
    return test_client


# --- read API ---------------------------------------------------------------------------------

def test_a_card_reads_back_with_its_body_and_press_summary(tmp_path, client):
    card = _component_card(tmp_path)
    response = client.get(f"/api/cards/{card['card_id']}")
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["components"] == [ACTION, FORM] and body["data"] == {"rows": []}
    assert (body["kind"], body["title"], body["revision"], body["pinned"]) == ("components", "Flights", 1, False)
    assert body["submissions_summary"] == [] and body["submissions_dropped"] == 0 and "app" not in body
    missing = client.get("/api/cards/card_0000000000000000")
    assert missing.status_code == 404 and missing.json()["reason_code"] == "card_not_found"
    assert client.get("/api/cards/..%2Fstate").status_code == 404


def test_an_app_is_served_as_inert_text_never_as_a_page(tmp_path, client):
    page = b"<!doctype html><script>alert(1)</script>"
    card = _app_card(tmp_path, page)
    view = client.get(f"/api/cards/{card['card_id']}").json()
    assert view["app"]["url"] == f"/api/cards/{card['card_id']}/app" and view["app"]["size"] == len(page)
    served = client.get(view["app"]["url"])
    assert served.status_code == 200 and served.content == page
    assert served.headers["content-type"] == "text/plain; charset=utf-8"
    assert served.headers["x-content-type-options"] == "nosniff" and served.headers["cache-control"] == "no-store"
    component = _component_card(tmp_path)
    assert client.get(f"/api/cards/{component['card_id']}/app").status_code == 404


# --- pins -------------------------------------------------------------------------------------

def test_a_kept_component_card_joins_the_widgets_page_and_leaves_it(tmp_path, client):
    from ouroboros.contracts.skill_manifest import canonical_skill_name

    assert canonical_skill_name(PINNED_CARDS_SKILL) != PINNED_CARDS_SKILL  # no installed skill can own it
    card = _component_card(tmp_path)
    pinned = client.post(f"/api/cards/{card['card_id']}/pin")
    assert pinned.status_code == 200 and pinned.json()["pinned"] is True and pinned.json()["pinned_at"]
    [tab] = client.get("/api/widgets").json()["ui_tabs"]
    assert tab["key"] == f"_cards:{card['card_id']}" and tab["skill"] == "_cards" and tab["pinned_card"] is True
    assert tab["render"] == {"kind": "declarative", "schema_version": 1, "start": "auto",
                             "components": [ACTION, FORM], "data": {"rows": []}}
    assert tab["revision"] == "1" and client.get(f"/api/cards/{card['card_id']}").json()["pinned"] is True
    assert client.broadcasts[-1] == {"type": "extension_lifecycle", "skill": "_cards", "action": "card_pinned",
                                     "data": {"card_id": card["card_id"]}}
    removed = client.delete(f"/api/cards/{card['card_id']}/pin")
    assert removed.status_code == 200 and removed.json()["pinned"] is False
    assert client.get("/api/widgets").json()["ui_tabs"] == []
    assert client.broadcasts[-1]["action"] == "card_unpinned"
    count = len(client.broadcasts)
    assert client.delete(f"/api/cards/{card['card_id']}/pin").json()["pinned"] is False
    assert len(client.broadcasts) == count  # nothing changed, nothing announced


def test_only_component_cards_can_be_kept_and_pins_are_bounded(tmp_path, client, monkeypatch):
    import ouroboros.card_store as store

    app = _app_card(tmp_path)
    refused = client.post(f"/api/cards/{app['card_id']}/pin")
    assert refused.status_code == 409 and refused.json()["reason_code"] == "card_not_pinnable"
    assert "make an app into a skill" in refused.json()["error"]
    assert client.post("/api/cards/card_0000000000000000/pin").status_code == 404
    monkeypatch.setattr(store, "MAX_PINS", 1)
    first, second = _component_card(tmp_path), _component_card(tmp_path)
    assert client.post(f"/api/cards/{first['card_id']}/pin").status_code == 200
    full = client.post(f"/api/cards/{second['card_id']}/pin")
    assert full.status_code == 409 and full.json()["reason_code"] == "pins_full"
    assert client.post(f"/api/cards/{first['card_id']}/pin").status_code == 200  # re-keeping refreshes, never counts twice


# --- presses (POST /api/decisions, card: family) -----------------------------------------------

@pytest.fixture
def decide(tmp_path, monkeypatch):
    """The decision app with liveness and the bridge controlled; ``live`` toggles the author."""
    import supervisor.message_bus as mb
    import supervisor.queue as q
    from ouroboros.gateway import task_decision as td

    state = {"live": True}
    frames = []
    bridge = mb.LocalChatBridge({})
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(mb, "DATA_DIR", tmp_path)
    monkeypatch.setattr(mb, "_BRIDGE", bridge)
    monkeypatch.setattr(td, "request_drive_root", lambda request: tmp_path)
    monkeypatch.setattr(td, "_live_root_task",
                        lambda task_id, **_: ({"id": task_id}, "") if state["live"] else (None, "task_not_live"))
    monkeypatch.setattr(q, "_task_drive_for_task", lambda task, task_id: tmp_path, raising=False)
    client = TestClient(Starlette(routes=[Route("/api/decisions", endpoint=td.api_decision_answer, methods=["POST"])]))

    def post(body):
        return client.post("/api/decisions", json=body)

    post.state, post.frames, post.bridge = state, frames, bridge
    return post


def _mail(tmp_path, task_id="task-1"):
    from ouroboros.owner_mailbox import drain_owner_entries

    return drain_owner_entries(tmp_path, task_id, set())


def test_a_press_reaches_the_live_author_as_card_input_once(tmp_path, decide):
    from ouroboros.owner_mailbox import KIND_CARD_INPUT

    card = _component_card(tmp_path)
    decision_id = f"card:{card['card_id']}:dates"
    response = decide({"request_id": "r1", "revision": 1, "decision_id": decision_id, "values": {"from": "Mon", "to": "Fri"}})
    assert response.status_code == 200, response.text
    assert response.json() == {"ok": True, "decision_id": decision_id, "request_id": "r1", "duplicate": False,
                               "delivery": "delivered", "card_id": card["card_id"], "revision": 1}
    [entry] = _mail(tmp_path)
    assert entry["kind"] == KIND_CARD_INPUT and entry["msg_id"] == f"card_input:{card['card_id']}:r1"
    assert entry["text"].startswith(f"[Card input] The owner pressed «Submit» on your card «Flights» (card {card['card_id']}).")
    assert json.loads(entry["text"].split("Values:\n", 1)[1]) == {"from": "Mon", "to": "Fri"}
    [stored] = load_card(tmp_path, card["card_id"])["submissions"]
    assert stored["delivered"] == "delivered" and stored["label"] == "Submit" and stored["form_title"] == "Dates"
    state = [f for f in decide.frames if f.get("type") == "card_state"][-1]
    assert state["submissions_summary"][0]["delivery"] == "delivered"

    again = decide({"request_id": "r1", "revision": 1, "decision_id": decision_id, "values": {"from": "Sat"}})
    assert again.status_code == 200 and again.json()["duplicate"] is True and again.json()["delivery"] == "delivered"
    assert len(_mail(tmp_path)) == 1
    later = decide({"request_id": "r2", "revision": 1, "decision_id": decision_id, "values": {"from": "Sun"}})
    assert later.status_code == 409 and later.json()["error"] == "card_action_once"
    assert later.json()["winner"] == {"request_id": "r1", "at": later.json()["winner"]["at"], "delivery": "delivered"}
    assert len(load_card(tmp_path, card["card_id"])["submissions"]) == 1


def test_each_press_of_an_ordinary_action_is_its_own_submission(tmp_path, decide):
    card = _component_card(tmp_path)
    for request_id in ("r1", "r2"):
        response = decide({"request_id": request_id, "revision": 1, "decision_id": f"card:{card['card_id']}:book"})
        assert response.status_code == 200 and response.json()["duplicate"] is False
    assert [e["text"].count("«Book it»") for e in _mail(tmp_path)] == [1, 1]
    assert "Values:" not in _mail(tmp_path)[0]["text"]


@pytest.mark.parametrize(("body", "status", "reason"), [
    ({"decision_id": "card:{id}:book", "values": {"x": "1"}}, 400, "values_invalid"),
    ({"decision_id": "card:{id}:dates", "values": {"nope": "1"}}, 400, "values_invalid"),
    ({"decision_id": "card:{id}:dates", "values": {"from": "x" * 4001}}, 400, "values_invalid"),
    ({"decision_id": "card:{id}:dates", "values": {"from": {"nested": 1}}}, 400, "values_invalid"),
    ({"decision_id": "card:{id}:dates", "values": ["Mon"]}, 400, "values_invalid"),
    ({"decision_id": "card:{id}:dates", "option_index": 0}, 400, "unexpected_fields"),
    ({"decision_id": "card:card_0000000000000000:book"}, 404, "card_not_found"),
    ({"decision_id": "card:{id}"}, 400, "malformed_decision_id"),
    ({"decision_id": "card:{id}:book", "revision": "1"}, 400, "revision_required"),
])
def test_a_refused_press_records_nothing(tmp_path, decide, body, status, reason):
    card = _component_card(tmp_path)
    press = {"request_id": "r1", "revision": 1, **body, "decision_id": body["decision_id"].format(id=card["card_id"])}
    response = decide({key: value for key, value in press.items() if value is not None})
    assert response.status_code == status and response.json()["reason_code"] == reason
    assert load_card(tmp_path, card["card_id"])["submissions"] == [] and _mail(tmp_path) == []


def test_form_values_accept_json_scalars(tmp_path, decide):
    card = _component_card(tmp_path)
    response = decide({"request_id": "r1", "revision": 1, "decision_id": f"card:{card['card_id']}:dates",
                       "values": {"from": 3, "to": None}})
    assert response.status_code == 200
    assert load_card(tmp_path, card["card_id"])["submissions"][0]["values"] == {"from": 3, "to": None}


def test_widget_and_app_cards_have_no_host_drawn_controls(tmp_path, decide):
    app = _app_card(tmp_path)
    response = decide({"request_id": "r1", "revision": 1, "decision_id": f"card:{app['card_id']}:book"})
    assert response.status_code == 404 and response.json()["reason_code"] == "action_not_found"


def test_a_press_after_the_author_ended_is_the_owners_message_in_the_cards_chat(tmp_path, decide):
    """The owner's bubble holds their own words (the pressed label, each value), as a Reply to the card."""
    from supervisor.card_delivery import send_card

    card = _component_card(tmp_path)
    assert send_card(decide.bridge, card, 1, task_id="task-1", interactive=True) == (True, "ok")
    [shown] = [f for f in decide.frames if f.get("type") == "card"]
    assert load_card(tmp_path, card["card_id"])["publication_address"] == shown["row_address"]
    decide.state["live"] = False
    decision_id = f"card:{card['card_id']}:book"
    response = decide({"request_id": "r1", "revision": 1, "decision_id": decision_id})
    assert response.status_code == 200, response.text
    assert response.json()["delivery"] == "forwarded" and _mail(tmp_path) == []
    queued = decide.bridge._inbox.get_nowait()
    assert queued["chat_id"] == 1 and queued["client_message_id"] == f"card_input:{card['card_id']}:r1"
    assert queued["text"] == "Book it", "never the host's [Card input] frame in the owner's voice"
    echo = [f for f in decide.frames if f.get("type") == "chat" and f.get("role") == "user"]
    assert len(echo) == 1 and echo[0]["content"] == "Book it"
    assert echo[0]["reply_to"]["address"] == shown["row_address"] and echo[0]["reply_to"]["role"] == "assistant"
    form = decide({"request_id": "r3", "revision": 1, "decision_id": f"card:{card['card_id']}:dates",
                   "values": {"from": "Mon", "to": 3}})
    assert form.json()["delivery"] == "forwarded"
    assert decide.bridge._inbox.get_nowait()["text"] == "Dates\nfrom: Mon\nto: 3"
    again = decide({"request_id": "r1", "revision": 1, "decision_id": decision_id})
    assert again.json()["duplicate"] is True and again.json()["delivery"] == "forwarded"
    assert decide.bridge._inbox.empty()
    assert load_card(tmp_path, card["card_id"])["submissions"][0]["delivered"] == "forwarded"


def test_a_kept_copy_cannot_press_a_replaced_card_but_a_recorded_press_still_retries(tmp_path, decide):
    card = _component_card(tmp_path)
    decide.state["live"] = False
    owed = decide({"request_id": "r1", "revision": 1, "decision_id": f"card:{card['card_id']}:book"})
    assert owed.status_code == 200
    write_card(tmp_path, chat_id=1, task_id="task-2", title="Flights (new)", fallback_text="C or D", kind="components",
               body={"components": [{"type": "action", "label": "Book C", "submit": {"id": "book"}}]},
               replace=card["card_id"])
    stale = decide({"request_id": "r2", "revision": 1, "decision_id": f"card:{card['card_id']}:book"})
    assert stale.status_code == 409 and stale.json()["reason_code"] == "card_stale" and stale.json()["revision"] == 2
    assert [row["request_id"] for row in load_card(tmp_path, card["card_id"])["submissions"]] == ["r1"]
    fresh = decide({"request_id": "r3", "revision": 2, "decision_id": f"card:{card['card_id']}:book"})
    assert fresh.status_code == 200 and load_card(tmp_path, card["card_id"])["submissions"][-1]["label"] == "Book C"
    again = decide({"request_id": "r1", "revision": 1, "decision_id": f"card:{card['card_id']}:book"})
    assert again.status_code == 200 and again.json()["duplicate"] is True, "a recorded press answers whatever the card became"


def test_an_owed_press_retried_after_a_replace_reaches_the_author_it_was_pressed_for(tmp_path, decide):
    card = _component_card(tmp_path, chat_id=-7)  # a machine chat: the press stays owed while its author is gone
    decide.state["live"] = False
    owed = decide({"request_id": "r1", "revision": 1, "decision_id": f"card:{card['card_id']}:dates", "values": {"from": "Mon"}})
    assert owed.status_code == 202 and owed.json()["delivery"] == "owed"
    write_card(tmp_path, chat_id=-7, task_id="task-2", title="New task card", fallback_text="C", kind="components",
               body={"components": [{"type": "action", "label": "Other", "submit": {"id": "other"}}]},
               replace=card["card_id"])
    again = decide({"request_id": "r2", "revision": 2, "decision_id": f"card:{card['card_id']}:other"})
    assert again.status_code == 202, "a press of the new card is the new author's"
    decide.state["live"] = True
    retried = decide({"request_id": "r1", "revision": 1, "decision_id": f"card:{card['card_id']}:dates"})
    assert retried.status_code == 200 and retried.json()["duplicate"] is True and retried.json()["delivery"] == "delivered"
    [entry] = _mail(tmp_path, "task-1")
    assert "«Submit» on your card «Flights»" in entry["text"] and "Form: Dates." in entry["text"]
    assert json.loads(entry["text"].split("Values:\n", 1)[1]) == {"from": "Mon"}
    assert _mail(tmp_path, "task-2") == []


def test_a_once_press_that_lost_to_an_owed_winner_says_so(tmp_path, decide):
    card = _component_card(tmp_path, chat_id=-7)
    decide.state["live"] = False
    assert decide({"request_id": "r1", "revision": 1, "decision_id": f"card:{card['card_id']}:dates"}).status_code == 202
    later = decide({"request_id": "r2", "revision": 1, "decision_id": f"card:{card['card_id']}:dates"})
    assert later.status_code == 409 and later.json()["winner"]["delivery"] == "owed"
    assert later.json()["winner"]["request_id"] == "r1"


def test_the_summary_names_each_press_by_its_request(tmp_path, decide):
    from ouroboros.card_store import submissions_summary

    card = _component_card(tmp_path, chat_id=-7)
    decide.state["live"] = False
    assert decide({"request_id": "r1", "revision": 1, "decision_id": f"card:{card['card_id']}:dates",
                   "values": {"from": "Mon"}}).status_code == 202
    [entry] = submissions_summary(load_card(tmp_path, card["card_id"]))
    assert entry == {"action_id": "dates", "label": "Submit", "at": entry["at"], "delivery": "owed", "request_id": "r1"}


def test_an_undeliverable_press_stays_owed_and_a_retry_delivers_it(tmp_path, decide):
    card = _component_card(tmp_path, chat_id=-7)  # a machine chat: no owner turn to start there
    decide.state["live"] = False
    decision_id = f"card:{card['card_id']}:book"
    owed = decide({"request_id": "r1", "revision": 1, "decision_id": decision_id})
    assert owed.status_code == 202 and owed.json()["delivery"] == "owed"
    assert owed.json()["reason_code"] == "card_chat_unanswerable"
    decide.state["live"] = True
    retried = decide({"request_id": "r1", "revision": 1, "decision_id": decision_id})
    assert retried.status_code == 200 and retried.json() | {"duplicate": True} == retried.json()
    assert retried.json()["delivery"] == "delivered" and len(_mail(tmp_path)) == 1


@pytest.mark.parametrize("pause", ["owner_hold", "queued"])
def test_a_paused_authors_press_waits_in_its_mailbox_until_it_reads_it(tmp_path, monkeypatch, pause):
    """Like an answer: a living author that reads no mail now keeps the press (no turn of its own, the hold
    stays), the card says the task reads it when it continues, and its read ends that wait live."""
    import queue
    from types import SimpleNamespace

    from ouroboros.card_store import submissions_summary
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from supervisor.events_runtime_controls import _handle_owner_message_injected
    from tests.test_question_answer_routing import runtime
    from tests.test_quiz_answer import _post
    from tests.test_quiz_late_answer import _inbox

    app, bridge, q, task = runtime(tmp_path, monkeypatch)
    card = _component_card(tmp_path, task_id="asker")
    press = {"request_id": "r1", "revision": 1, "decision_id": f"card:{card['card_id']}:book"}
    if pause == "queued":
        q.PENDING.append(q.RUNNING.pop("asker")["task"])
    else:
        task["_owner_hold"] = True
    response = _post(app, press)
    assert response.status_code == 200, response.text
    assert (response.json()["delivery"], response.json()["reason_code"]) == ("delivered", "awaiting_recipient_read")
    assert _inbox(bridge) == [] and task.get("_owner_hold") is (True if pause == "owner_hold" else None)
    assert submissions_summary(load_card(tmp_path, card["card_id"]))[0]["reason_code"] == "awaiting_recipient_read"
    assert _post(app, press).json()["reason_code"] == "awaiting_recipient_read", "a retry still says it waits"

    task.pop("_owner_hold", None)
    if pause == "queued":
        q.RUNNING["asker"] = {"task": q.PENDING.pop()}
    events = queue.Queue()
    ctx = SimpleNamespace(task_id="asker", task_metadata={}, drive_root=tmp_path, budget_drive_root=tmp_path)
    _drain_incoming_messages([], queue.Queue(), tmp_path, "asker", events, set(), owner_ctx=ctx)
    [pressed] = ctx._owner_directives
    assert (pressed["source"], pressed["content"]) == ("owner_press", "Book it")
    assert pressed["reply_to"]["card_id"] == card["card_id"]
    [read] = [event for event in list(events.queue) if event.get("card_ack")]
    assert "reason_code" not in submissions_summary(load_card(tmp_path, card["card_id"]))[0]
    frames = []
    bridge._broadcast_fn = frames.append
    _handle_owner_message_injected(read, SimpleNamespace(DRIVE_ROOT=tmp_path, bridge=bridge))
    [state] = [frame for frame in frames if frame.get("type") == "card_state"]
    assert state["submissions_summary"][0]["delivery"] == "delivered" and "reason_code" not in state["submissions_summary"][0]
    assert "reason_code" not in _post(app, press).json()
    # The other direction: a press while its author reads mail never waits, and its read needs no receipt.
    reading = _post(app, {**press, "request_id": "r2"})
    assert reading.status_code == 200 and "reason_code" not in reading.json()
    assert "card_delivery" not in _mail(tmp_path, "asker")[-1]


@pytest.mark.parametrize("hold", ["", "restart_hold", "owner_pause"])
def test_a_sleeping_authors_press_wakes_it_only_when_unheld(tmp_path, monkeypatch, hold):
    from supervisor.sleep_wake import wake_ready_sleepers
    from tests.test_question_answer_routing import sleeping_asker
    from tests.test_quiz_answer import _post
    from tests.test_quiz_late_answer import _inbox

    app, bridge, q = sleeping_asker(tmp_path, monkeypatch, hold)
    card = _component_card(tmp_path, task_id="asker")
    response = _post(app, {"request_id": "r1", "revision": 1, "decision_id": f"card:{card['card_id']}:book"})
    assert response.status_code == 200 and response.json()["delivery"] == "delivered", response.text
    assert [entry["msg_id"] for entry in _mail(tmp_path, "asker")] == [f"card_input:{card['card_id']}:r1"]
    assert _inbox(bridge) == [], "never a turn of its own"
    woke = [outcome for outcome in wake_ready_sleepers(q) if outcome.get("ok")]
    # An unheld sleeper reads its mail and wakes for it; a held one keeps the press and says it waits.
    assert bool(woke) is (hold == "")
    assert response.json().get("reason_code", "") == ("" if hold == "" else "awaiting_recipient_read")


def test_a_read_before_the_wait_is_recorded_never_brings_the_wait_back(tmp_path, monkeypatch):
    """It continued and read the press before the press's delivery was recorded: answer and record say Sent."""
    from ouroboros.card_store import acknowledge_press, submissions_summary
    from ouroboros.gateway import card_decision
    from tests.test_question_answer_routing import runtime
    from tests.test_quiz_answer import _post

    app, _bridge, _q, task = runtime(tmp_path, monkeypatch)
    task["_owner_hold"] = True
    card = _component_card(tmp_path, task_id="asker")
    deliver = card_decision._deliver

    def read_at_once(drive_root, record, submission, source):
        outcome = deliver(drive_root, record, submission, source)
        acknowledge_press(drive_root, record["card_id"], submission["request_id"])
        return outcome

    monkeypatch.setattr(card_decision, "_deliver", read_at_once)
    response = _post(app, {"request_id": "r1", "revision": 1, "decision_id": f"card:{card['card_id']}:book"})
    assert response.status_code == 200 and response.json()["delivery"] == "delivered", response.text
    assert "reason_code" not in response.json()
    assert "reason_code" not in submissions_summary(load_card(tmp_path, card["card_id"]))[0]


# --- the card_input mailbox kind ---------------------------------------------------------------

def test_card_input_is_owner_input_for_the_drain_the_wake_and_custody(tmp_path):
    import queue

    from types import SimpleNamespace

    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import KIND_CARD_INPUT, write_owner_message
    from ouroboros.owner_wait import classify_wake
    from ouroboros.task_custody import owner_mail_rows, unread_mail_rows

    frame = "[Card input] The owner pressed «Book it» on your card «Flights» (card card_0123456789abcdef)."
    assert write_owner_message(tmp_path, frame, "task-1", msg_id="card_input:c:r1", kind=KIND_CARD_INPUT)
    assert classify_wake([{"kind": KIND_CARD_INPUT}], "") == "owner_text"
    assert classify_wake([{"kind": "hurry"}], "") == "hurry"  # the other direction: a control is not owner text
    assert len(owner_mail_rows(tmp_path, "task-1")[0]) == 1 and len(unread_mail_rows(tmp_path, "task-1")[0]) == 1
    messages = []
    ctx = SimpleNamespace()
    _drain_incoming_messages(messages, queue.Queue(), tmp_path, "task-1", None, set(), owner_ctx=ctx)
    assert any(frame in str(message.get("content")) for message in messages)
    assert ctx._owner_directives[-1]["source"] == "owner_card_input"  # a legacy input with no recoverable submission
    from ouroboros.loop_messages import owner_input_revision
    from ouroboros.owner_words import owner_words_contain
    assert owner_input_revision(ctx) == 1 and not owner_words_contain(ctx, "Book it")


@pytest.mark.parametrize("action, values", [("book", None), ("dates", {"from": "  moonrise  "})])
def test_the_living_card_drain_records_the_choice_or_supplied_values(tmp_path, decide, action, values):
    import queue
    from types import SimpleNamespace
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.loop_messages import owner_input_revision, owner_reply_owed
    from ouroboros.owner_words import cited_owner_basis, owner_words_contain
    from ouroboros.owner_continue import owner_corpus_rows, owner_sources

    card = _component_card(tmp_path)
    response = decide({"request_id": "moon-press", "revision": 1, "decision_id": f"card:{card['card_id']}:{action}",
                       **({"values": values} if values is not None else {})})
    assert response.json()["delivery"] == "delivered"
    ctx = SimpleNamespace(task_id="task-1", drive_root=tmp_path)
    messages = []
    # A reviewer or a child can read this mailbox before the serving loop acknowledges the entry.
    from ouroboros.review_evidence_sections import _accept_owner_directives
    from ouroboros.owner_words import task_owner_words
    [entry] = _mail(tmp_path, ctx.task_id)
    [unread] = _accept_owner_directives(SimpleNamespace(), tmp_path, ctx.task_id)
    field, _ = task_owner_words(tmp_path, ctx.task_id, task={})
    _drain_incoming_messages(messages, queue.Queue(), tmp_path, ctx.task_id, None, set(), owner_ctx=ctx)
    [row] = ctx._owner_directives
    label = "Book it" if values is None else "Dates"
    assert label in messages[0]["content"] and owner_input_revision(ctx) == 1 and owner_reply_owed(ctx)
    assert not owner_words_contain(ctx, label) and not owner_words_contain(ctx, "Card input")
    assert row["source"] == ("owner_press" if values is None else "owner_mailbox")
    if values:
        assert row["content"] == values["from"] and owner_words_contain(ctx, values["from"])
    else:
        assert cited_owner_basis(ctx, label)[0]["source"] == "owner_press"
    sources = owner_sources(tmp_path, {"origin_message_text": "Open the chart", "origin_message_ref": {}}, ctx.task_id)
    recovered = owner_corpus_rows(sources)[-1]
    assert (recovered["content"], recovered["source"]) == (row["content"], row["source"])

    # A pre-snapshot mailbox can resolve the stored submission. A pre-fix capture of its exact frame
    # is the same addressed input. Different bytes at that identity still refuse Continue.
    from ouroboros.answer_facts import living_card_input
    legacy = dict(entry)
    legacy.pop("card_press")
    assert living_card_input(legacy, tmp_path)["content"] == row["content"]
    saved = {"source": "owner_mailbox", "content": entry["text"], "msg_id": entry["msg_id"]}
    prior = {"origin_message_text": "Open the chart", "origin_message_ref": {},
             "review_evidence": {"task_inputs": {"owner_requirements_and_decisions": [saved]}}}
    assert not owner_sources(tmp_path, prior, ctx.task_id)["gaps"]
    saved["content"] += " Different frame."
    assert "owner_message_identity_conflict" in owner_sources(tmp_path, prior, ctx.task_id)["gaps"]
    assert (unread["source"], unread["content"], unread["reply_to"]) == (row["source"], row["content"], row["reply_to"])
    assert field[0]["source"] == row["source"] and label in field[1]["text"]
    assert field[0]["text"] == values["from"] if values else "pressed" in field[0]["text"]


@pytest.mark.parametrize("action, values", [("book", None), ("dates", {"from": "  after moonrise  "})])
def test_forwarded_card_keeps_its_bubble_and_opens_a_typed_head(tmp_path, decide, action, values):
    from types import SimpleNamespace
    from ouroboros.answer_facts import row_source
    from ouroboros.dialogue_provenance import quote_words
    from ouroboros.loop_messages import _initialize_owner_directives, owner_input_revision, owner_reply_owed
    from ouroboros.owner_words import carried_owner_corpus, owner_words_contain
    from supervisor.message_ingress import accepted_chat_message

    card = _component_card(tmp_path)
    decide.state["live"] = False
    response = decide({"request_id": "eclipse", "revision": 1, "decision_id": f"card:{card['card_id']}:{action}",
                       **({"values": values} if values else {})})
    assert response.json()["delivery"] == "forwarded"
    queued = decide.bridge._inbox.get_nowait()
    bubble = "Dates\nfrom: " + values["from"] if values else "Book it"
    assert queued["text"] == bubble
    row = accepted_chat_message(tmp_path, 1, queued["client_message_id"])
    assert row["text"] == bubble
    source = "owner_mailbox" if values else "owner_press"
    assert row_source(row, "owner_mailbox") == source
    assert quote_words(row) == (values["from"] if values else "")
    metadata = {**queued["task_metadata"], "origin_message_ref": {"chat_id": 1, "client_message_id": queued["client_message_id"]},
                "origin_message_text": bubble}
    ctx = SimpleNamespace(task_metadata=metadata, drive_root=tmp_path)
    _initialize_owner_directives(ctx, [{"role": "user", "content": "[Card input]"}])
    assert owner_input_revision(ctx) == 1 and owner_reply_owed(ctx)
    [opening] = carried_owner_corpus(ctx, bubble)
    assert (opening["source"], opening["content"]) == (source, values["from"] if values else "Book it")
    assert not owner_words_contain(ctx, "Dates" if values else "Book it")

    from ouroboros.owner_words import task_owner_words
    from ouroboros.project_dialogue import build_owner_message_ref
    reference = build_owner_message_ref(chat_id=1, client_message_id=row["client_message_id"], ts=row["ts"], text=bubble)
    field, _ = task_owner_words(tmp_path, "observatory", task={"origin_message_ref": reference, "origin_message_text": bubble})
    assert field[0]["source"] == source
    assert field[0]["text"] == values["from"] if values else "pressed" in field[0]["text"]
    assert field[1]["ref"] == opening["reply_to"]["address"]

    # The original card reference on older rows has no submitted values by value. The source store still has them.
    from ouroboros.owner_continue import owner_corpus_rows, owner_sources
    from ouroboros.owner_relay import inject_relay, owner_quote, write_relay
    old = {**row, "card_press": {k: v for k, v in row["card_press"].items() if k != "submission"}}
    quote = owner_quote(tmp_path, old)
    assert (quote["input_source"], quote["text"]) == (source, opening["content"])
    frames = []
    reader = SimpleNamespace(task_id="lunar-reader", drive_root=tmp_path)
    relay, created = write_relay(tmp_path, reader.task_id, relay_id="lunar-mail", chat_id=1, quotes=[quote],
                                 teller_task_id="lunar-courier", note="The submission")
    assert created
    inject_relay(reader, relay, frames)
    assert reader._owner_directives[0]["content"] == opening["content"]
    assert opening["reply_to"]["text"] in frames[0]["content"] and opening["content"] in frames[0]["content"]
    if not values:
        assert "Owner card press" in frames[0]["content"]
    recovered = owner_sources(tmp_path, {"origin_message_text": "Read the chart", "origin_message_ref": {}}, reader.task_id)
    later = owner_corpus_rows(recovered)[-1]
    assert (later["source"], later["content"], later["reply_to"]) == (source, opening["content"], opening["reply_to"])


def test_continue_normalizes_only_the_exact_forwarded_form_bubble(tmp_path, decide, monkeypatch):
    import sys

    # This API witness runs on its own, with the Continue suite unavailable.
    monkeypatch.setitem(sys.modules, "tests.test_owner_continue", None)
    from ouroboros.owner_mailbox import write_owner_message
    from supervisor.continuation_admission import admit_continuation
    from tests._budget_pause_exact_helpers import _install_queue
    from tests._owner_continue_helpers import NONCE, _interrupted

    card = _component_card(tmp_path)
    decide.state["live"] = False
    values = {"from": "  first date  ", "to": "  last date  "}
    assert decide({"request_id": "fern-form", "revision": 1, "decision_id": f"card:{card['card_id']}:dates",
                   "values": values}).json()["delivery"] == "forwarded"
    queued = decide.bridge._inbox.get_nowait()
    ref = dict(queued["task_metadata"]["card_press"])
    ref.pop("submission")
    bubble = queued["text"].strip()
    assert write_owner_message(tmp_path, bubble, "fern-reader", msg_id="fern-forward", card_press=ref)
    _, _, workers = _install_queue(tmp_path, monkeypatch)
    def interrupt(capture):
        _interrupted(tmp_path, task_id="fern-reader", review_evidence={"task_inputs": {
            "owner_requirements_and_decisions": [{"source": "owner_mailbox", "content": capture, "msg_id": "fern-forward"}]}})
    interrupt(bubble + " changed")
    refused = admit_continuation("fern-reader", action_nonce=NONCE)
    assert refused["gaps"] == ["owner_message_identity_conflict"] and not workers.PENDING
    interrupt(bubble)
    accepted = admit_continuation("fern-reader", action_nonce=NONCE)
    assert accepted.get("ok"), accepted
    [successor] = workers.PENDING
    [form] = [r for r in successor["metadata"]["owner_corpus"] if r.get("msg_id") == "fern-forward"]
    assert form["source"] == "owner_mailbox" and form["content"] == "\n".join(values.values())
    assert form["reply_to"]["card_id"] == card["card_id"]


@pytest.mark.parametrize("component, caption, heading, bubble", [
    ({"type": "form", "title": "Cedar dates", "fields": [{"name": "from"}]}, "Submit", "Cedar dates", "Cedar dates"),
    ({"type": "form", "label": "Cedar dates", "submit_label": "  File dates  ", "fields": [{"name": "from"}]},
     "  File dates  ", "Cedar dates", "  File dates  "),
    ({"type": "action", "label": "  Choose the cedar route  "}, "  Choose the cedar route  ", "", "  Choose the cedar route  "),
])
def test_only_the_complete_drawn_caption_proves_a_wordless_press(tmp_path, decide, component, caption, heading, bubble):
    from types import SimpleNamespace

    from ouroboros.card_store import load_card, write_card
    from ouroboros.loop_messages import _initialize_owner_directives, owner_input_revision
    from ouroboros.owner_words import cited_owner_basis, owner_words_contain

    card = write_card(tmp_path, chat_id=1, task_id="cedar-desk", title="Cedar options", fallback_text="Choose",
                      kind="components", body={"components": [{**component, "submit": {"id": "cedar"}}]})
    decide.state["live"] = False
    assert decide({"request_id": "cedar-choice", "revision": 1, "decision_id": f"card:{card['card_id']}:cedar"}).json()["delivery"] == "forwarded"
    event = decide.bridge._inbox.get_nowait()
    assert event["text"] == bubble
    [stored] = load_card(tmp_path, card["card_id"])["submissions"]
    assert stored["label"] == caption
    meta = {**event["task_metadata"], "origin_message_ref": {"chat_id": 1, "client_message_id": event["client_message_id"]},
            "origin_message_text": bubble}
    ctx = SimpleNamespace(task_metadata=meta, drive_root=tmp_path)
    _initialize_owner_directives(ctx, [{"role": "user", "content": bubble}])
    [act] = ctx._owner_directives
    assert owner_input_revision(ctx) == 1 and act["reply_to"]["pressed_label"] == caption
    if heading:
        assert heading in act["reply_to"]["text"] and not cited_owner_basis(ctx, heading)[0]
    assert not owner_words_contain(ctx, caption)
    assert cited_owner_basis(ctx, caption)[0]["source"] == "owner_press"
    assert cited_owner_basis(ctx, caption.strip()[1:])[0] is None


def test_form_fields_keep_their_word_boundaries_in_the_corpus_and_its_copies(tmp_path, decide):
    import queue
    from types import SimpleNamespace

    from ouroboros.loop_messages import _initialize_owner_directives, owner_input_revision
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_continue import owner_corpus_rows, owner_sources
    from ouroboros.owner_words import carried_owner_corpus, cited_owner_basis, owner_words_contain

    card = _component_card(tmp_path)
    values = {"from": "  at\nmoonrise", "to": "sunset  "}
    assert decide({"request_id": "elm-fields", "revision": 1, "decision_id": f"card:{card['card_id']}:dates",
                   "values": values}).json()["delivery"] == "delivered"
    ctx = SimpleNamespace(task_id="task-1", drive_root=tmp_path)
    _drain_incoming_messages([], queue.Queue(), tmp_path, ctx.task_id, None, set(), owner_ctx=ctx)
    assert owner_input_revision(ctx) == 1
    [original] = ctx._owner_directives
    assert original["content"] == "\n".join(values.values())
    def check(focus):
        assert not owner_words_contain(focus, "moonrise\nsunset")
        assert cited_owner_basis(focus, "moonrise\nsunset")[0] is None
        for value in values.values():
            assert owner_words_contain(focus, value) and cited_owner_basis(focus, value)[0]
    check(ctx)
    from ouroboros.owner_relay import _rows_to_carry, inject_relay, write_relay
    [(_, _, quote, _, _)] = _rows_to_carry(ctx, tmp_path)
    entry, created = write_relay(tmp_path, "elm-reader", relay_id="elm-fields", note="Read the dates",
                                 quotes=[quote], teller_task_id=ctx.task_id, chat_id=1)
    assert created
    receiver = SimpleNamespace(task_id="elm-reader", drive_root=tmp_path)
    inject_relay(receiver, entry, [])
    check(receiver)
    recovered = owner_corpus_rows(owner_sources(tmp_path, {"origin_message_text": "Read the dates"}, ctx.task_id))
    for rows in (recovered, carried_owner_corpus(ctx), [cited_owner_basis(ctx, "moonrise")[0]]):
        seed = SimpleNamespace(task_metadata={"objective_author": {"kind": "task"}, "owner_corpus": rows})
        _initialize_owner_directives(seed, [{"role": "user", "content": "Read the calendar"}])
        check(seed)


def test_start_task_seeds_one_cited_form_submission(tmp_path, decide, monkeypatch):
    import queue
    from pathlib import Path
    from types import SimpleNamespace

    from ouroboros.loop_messages import _initialize_owner_directives
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.tools import control_routing
    from ouroboros.tools.registry import ToolContext

    card = _component_card(tmp_path)
    values = {"from": "at moonrise", "to": "sunset"}
    assert decide({"request_id": "elm-start", "revision": 1, "decision_id": f"card:{card['card_id']}:dates",
                   "values": values}).json()["delivery"] == "delivered"
    ctx = ToolContext(repo_dir=Path.cwd(), drive_root=tmp_path, task_metadata={})
    ctx.task_id, ctx.current_chat_id = "task-1", 1
    _drain_incoming_messages([], queue.Queue(), tmp_path, ctx.task_id, None, set(), owner_ctx=ctx)
    events = []
    monkeypatch.setattr(control_routing, "_emit_and_wait_for_routing",
                        lambda _, evt: (events.append(evt) or "live", {"status": "scheduled"}))
    monkeypatch.setattr(control_routing, "_promotion_pool_disabled_from_snapshot", lambda _: "")
    control_routing._start_task(ctx, "Read the elm dates", title="Elm dates", owner_words="moonrise\nsunset")
    assert not events
    control_routing._start_task(ctx, "Read the elm dates", title="Elm dates", owner_words="moonrise")
    [event] = events
    successor = SimpleNamespace(task_metadata=event)
    _initialize_owner_directives(successor, [{"role": "user", "content": "Read the elm dates"}])
    for rows in (event["owner_corpus"], successor._owner_directives):
        [row] = rows
        assert (row["msg_id"], row["basis"], row["cited"]) == (ctx._owner_directives[0]["msg_id"], "cited", "moonrise")
        assert row["reply_to"]["submitted_values"] == values
