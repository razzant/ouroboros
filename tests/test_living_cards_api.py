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
    assert entry["text"].startswith(f"[Card input] The owner pressed «Dates» on your card «Flights» (card {card['card_id']}).")
    assert json.loads(entry["text"].split("Values:\n", 1)[1]) == {"from": "Mon", "to": "Fri"}
    [stored] = load_card(tmp_path, card["card_id"])["submissions"]
    assert stored["delivered"] == "delivered" and stored["label"] == "Dates"
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
    ({"decision_id": "card:{id}:missing"}, 404, "action_not_found"),
    ({"decision_id": "card:card_0000000000000000:book"}, 404, "card_not_found"),
    ({"decision_id": "card:{id}"}, 400, "malformed_decision_id"),
    ({"decision_id": "card:{id}:book", "revision": None}, 400, "revision_required"),
    ({"decision_id": "card:{id}:book", "revision": "1"}, 400, "revision_required"),
    ({"decision_id": "card:{id}:book", "revision": 2}, 409, "card_stale"),
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
    assert "«Dates» on your card «Flights»" in entry["text"] and json.loads(entry["text"].split("Values:\n", 1)[1]) == {"from": "Mon"}
    assert all("«Dates»" not in row["text"] for row in _mail(tmp_path, "task-2")), "never the replacing task's"


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
    assert entry == {"action_id": "dates", "label": "Dates", "at": entry["at"], "delivery": "owed", "request_id": "r1"}


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
    assert [row["content"] for row in ctx._owner_directives][-1].startswith("[Card input] The owner pressed «Book it»")
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
    assert ctx._owner_directives[-1]["source"] == "owner_mailbox"
