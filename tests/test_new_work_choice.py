"""A message sent as new work on the model chosen for its room.

The composer's pick is saved for the room (or the Main chat) and answered with a token of that choice. The next
message names the token; its first acceptance assigns the model and tier from the saved choice itself, refuses the
whole message when another window changed the choice first, and keeps what it assigned on the accepted row. The
routing then starts that message's own turn on what was assigned and hands it to no work already running.
"""
from __future__ import annotations

import json

import pytest

from ouroboros import config, route_choice, server_owner_routing
from ouroboros import projects_registry as projects
from ouroboros import server_routing_context as routing
from ouroboros.composer_cognition import model_choice_token, saved_choice, update_project_model_choice
from supervisor import message_ingress
from supervisor.message_ingress import ModelChoiceChanged, new_work_choice
from tests._task_settings_helpers import isolated_runtime, save_owner  # noqa: F401 -- the shared isolated root
from tests.test_chat_attachments import _history, _web_bridge
from tests.test_project_routing_v664 import _ctx

MAIN = "claudexor::codex=fictional-main"
BETA = "claudexor::codex=fictional-beta"
GAMMA = "openrouter-fiction/gamma"
CHOSEN = {"kind": "model", "model": BETA, "use_local": False}
OTHER = {"kind": "model", "model": GAMMA, "use_local": False}
ROLE = {"OUROBOROS_MODEL": MAIN, "OUROBOROS_MODEL_ACCOUNTS": json.dumps({"main": "acc-main"}),
        "OUROBOROS_MODEL_CONTEXT_WINDOWS": json.dumps({"main": 123456}),
        "OUROBOROS_MODEL_PROCESSING_PREFERENCES": json.dumps({"main": "fast"})}


@pytest.fixture
def rooms(isolated_runtime):  # noqa: F811 -- the imported fixture, used by name
    save_owner(ROLE)
    room = projects.create_project(isolated_runtime, "fable-room", origin="owner_ui")
    return isolated_runtime, room


def _choose(root, project_id, choice, tier=""):
    """The owner's pick as its writer saves it; returns the token the page was answered with."""
    if project_id:
        update_project_model_choice(root, project_id, choice)
        if tier:
            from ouroboros.composer_cognition import update_project_processing_choice

            update_project_processing_choice(root, project_id, tier)
    else:
        from ouroboros.composer_cognition import main_chat_route_setting

        save_owner({**json.loads(config.SETTINGS_PATH.read_text(encoding="utf-8")),
                    "OUROBOROS_MAIN_CHAT_MODEL": main_chat_route_setting(choice),
                    "OUROBOROS_MAIN_CHAT_PROCESSING": tier})
    return model_choice_token(project_id, choice)


@pytest.mark.parametrize("scope", ["room", "main"])
def test_the_acceptance_assigns_the_saved_choice_and_refuses_one_that_changed(rooms, scope):
    root, room = rooms
    project_id, chat_id = (room["id"], room["chat_id"]) if scope == "room" else ("", 1)
    token = _choose(root, project_id, CHOSEN, tier="economy")
    assert saved_choice(root, project_id)["choice_token"] == token
    # The frame names only the choice; what the work starts on is read from what is saved.
    assert new_work_choice(root, chat_id, "Compare the two drafts", None, token) == {
        "model_choice": CHOSEN, "processing_choice": "economy"}
    # Nothing asked, a slash command and a Reply addressed to a work take no answer: none of them spends the choice.
    for text, reply, named in (("Plain words", None, ""), ("Plain words", None, None), ("/status", None, token),
                               ("Go on", {"address": "task:a", "work_task_id": "a"}, token),
                               ("Go on", {"address": "task:a", "work_task_id": ""}, token)):
        assert new_work_choice(root, chat_id, text, reply, named) is None
    # A quote of an ordinary row addresses no work: the message is still its own new work.
    assert new_work_choice(root, chat_id, "About this", {"address": "row:1", "ordinary": True}, token)["model_choice"] == CHOSEN
    # Another window changed the choice before this message was accepted: refused, with the typed cause.
    _choose(root, project_id, OTHER)
    with pytest.raises(ModelChoiceChanged) as refused:
        new_work_choice(root, chat_id, "Compare the two drafts", None, token)
    assert refused.value.reason_code == "model_choice_changed" and "Nothing was sent" in str(refused.value)
    # A Reply addressed to a living work by its id is that work's whatever became of the pick: the work keeps its own
    # model, the Reply is never refused for a choice that changed, and it spends nothing.
    assert new_work_choice(root, chat_id, "Go on", {"address": "task:a", "work_task_id": "a"}, token) is None
    # A token never names another room's choice, nor one the frame made up.
    for forged in (model_choice_token("elsewhere", OTHER), "not-a-token"):
        with pytest.raises(ModelChoiceChanged):
            new_work_choice(root, chat_id, "Compare the two drafts", None, forged)
    # A tier that moved alone is no change of the choice: the acceptance takes the tier saved now.
    current = _choose(root, project_id, OTHER, tier="fast")
    assert new_work_choice(root, chat_id, "Again", None, current) == {"model_choice": OTHER, "processing_choice": "fast"}


def test_a_room_that_cannot_be_read_refuses_the_message(rooms):
    root, room = rooms
    token = _choose(root, room["id"], CHOSEN)
    # A chat that is neither Main nor a room has no choice of its own, and never takes the Main chat's.
    main = _choose(root, "", OTHER)
    for named in (token, main):
        with pytest.raises(ModelChoiceChanged, match="could not be confirmed"):
            new_work_choice(root, room["chat_id"] + 1, "Compare the two drafts", None, named)
    assert new_work_choice(root, 1, "Compare the two drafts", None, main)["model_choice"] == OTHER
    projects.begin_project_deletion(root, room["id"])
    with pytest.raises(ModelChoiceChanged, match="could not be confirmed"):
        new_work_choice(root, room["chat_id"], "Compare the two drafts", None, token)


def test_a_rooms_acceptance_reads_the_registry_once(rooms, monkeypatch):
    """The strict routing read already holds the room's row: its saved choice is taken from that row."""
    from ouroboros import project_admission

    root, room = rooms
    token = _choose(root, room["id"], CHOSEN, tier="fast")
    reads: list = []
    snapshot = project_admission._registry_snapshot
    monkeypatch.setattr(project_admission, "_registry_snapshot",
                        lambda *args, **kwargs: reads.append(1) or snapshot(*args, **kwargs))
    assert new_work_choice(root, room["chat_id"], "Compare the two drafts", None, token) == {
        "model_choice": CHOSEN, "processing_choice": "fast"}
    assert len(reads) == 1, reads


def test_the_accepted_row_keeps_the_first_answer_whatever_a_repeat_or_another_window_says(rooms, monkeypatch):
    root, room = rooms
    token = _choose(root, room["id"], CHOSEN, tier="economy")
    bridge, echoes = _web_bridge(root, monkeypatch)
    sent = dict(broadcast=True, sender_session_id="window-1", client_message_id="msg-1",
                chat_id=room["chat_id"], project_id=room["id"])
    bridge.ui_send("Compare the two drafts", task_metadata={"new_work_token": token}, **sent)
    row = message_ingress.accepted_chat_message(root, room["chat_id"], "msg-1")
    assigned = {"model_choice": CHOSEN, "processing_choice": "economy"}
    assert row["new_work"] == assigned and echoes[-1]["new_work"] is True
    [update] = bridge.get_updates(0, timeout=0)
    assert update["message"]["task_metadata"]["new_work"] == assigned
    assert "new_work_token" not in update["message"]["task_metadata"]
    # Window 2 changes the room afterwards; the same id sent again (Send again, a reload's kept copy, a forged
    # answer of its own) rejoins the accepted row and takes what that first acceptance assigned.
    stale = _choose(root, room["id"], OTHER, tier="fast")
    for repeat in ({"new_work_token": token}, {"new_work_token": stale}, {},
                   {"new_work": {"model_choice": OTHER, "processing_choice": "ultrafast"}}):
        bridge.ui_send("Compare the two drafts", task_metadata=repeat, **sent)
        assert message_ingress.accepted_chat_message(root, room["chat_id"], "msg-1") == row
        assert echoes[-1]["new_work"] is True
    assert bridge.get_updates(0, timeout=0) == []  # the dispatch it owed was taken once
    # History replays the row as its echo said it, so a window that lost the echo still learns the answer from the
    # saved row; an ordinary message's row carries no such mark, in its echo or in history.
    bridge.ui_send("Plain words", **{**sent, "client_message_id": "msg-9"})
    replayed = {row.get("client_message_id"): row for row in _history(root, room["chat_id"])}
    assert replayed["msg-1"]["new_work"] is True and replayed["msg-1"]["ingress_accepted"] is True
    assert "new_work" not in replayed["msg-9"] and "new_work" not in echoes[-1]


def test_a_refused_message_saves_no_row_and_reaches_nobody(rooms, monkeypatch):
    root, room = rooms
    token = _choose(root, room["id"], CHOSEN)
    _choose(root, room["id"], OTHER)  # another window, before this message's first acceptance
    bridge, echoes = _web_bridge(root, monkeypatch)
    with pytest.raises(ModelChoiceChanged):
        bridge.ui_send("Compare the two drafts", broadcast=True, sender_session_id="window-1",
                       client_message_id="msg-2", chat_id=room["chat_id"], project_id=room["id"],
                       task_metadata={"new_work_token": token})
    assert message_ingress.accepted_chat_message(root, room["chat_id"], "msg-2") is None
    assert echoes == [] and bridge.get_updates(0, timeout=0) == []
    # A caller's own ``new_work`` is never an answer: an ordinary message carries none.
    bridge.ui_send("Plain words", broadcast=True, sender_session_id="window-1", client_message_id="msg-3",
                   chat_id=room["chat_id"], project_id=room["id"],
                   task_metadata={"new_work": {"model_choice": OTHER, "processing_choice": ""}})
    assert "new_work" not in message_ingress.accepted_chat_message(root, room["chat_id"], "msg-3")
    assert "new_work" not in echoes[-1]
    assert "new_work" not in bridge.get_updates(0, timeout=0)[0]["message"]["task_metadata"]


def test_the_socket_names_the_refusal_so_the_page_can_tell_it_apart(rooms, monkeypatch):
    import asyncio

    from ouroboros.gateway import ws

    root, room = rooms
    token = _choose(root, room["id"], CHOSEN)
    _choose(root, room["id"], OTHER)
    bridge, _echoes = _web_bridge(root, monkeypatch)
    sent: list = []

    class Socket:
        async def send_text(self, text):
            sent.append(json.loads(text))

    def accept():
        bridge.ui_send("Compare the two drafts", broadcast=True, sender_session_id="window-1",
                       client_message_id="msg-4", chat_id=room["chat_id"], project_id=room["id"],
                       task_metadata={"new_work_token": token})

    asyncio.run(ws._accept_chat_after(Socket(), None, accept, "msg-4", room["chat_id"]))
    [notice] = sent
    assert notice["system_type"] == "initialization_notice" and notice["client_message_id"] == "msg-4"
    assert notice["reason_code"] == "model_choice_changed" and "changed elsewhere" in notice["content"]


ASSIGNED = {"model_choice": CHOSEN, "processing_choice": "economy"}


class _Roots:
    """One live reader in every room asked: the work an ordinary message would reach."""
    gap = ""

    def rows(self, chat_id=None):
        return [{"task_id": "running-a", "room_reader": True, "status": "running", "chat_id": chat_id}]


@pytest.fixture
def busy(rooms, monkeypatch):
    """A busy room and a busy Main: the reader's delivery is recorded, never performed."""
    root, room = rooms
    delivered = []
    ctx = _ctx(root, running={"running-a": {"task": {"id": "running-a"}, "attempt": 1}})
    monkeypatch.setattr(server_owner_routing, "_deliver_to_reader",
                        lambda _ctx, candidate, *a, **kw: delivered.append(candidate["task_id"]) or candidate["task_id"])
    monkeypatch.setattr(server_owner_routing, "_ordinary_room_reader", lambda *_a: True)
    monkeypatch.setattr(server_owner_routing, "_owner_turn_key", lambda *_a: "owner:1")
    monkeypatch.setattr(server_owner_routing, "_owner_turn_closed", lambda *_a: False)
    return ctx, room, delivered


def test_words_sent_as_new_work_reach_no_work_already_running_and_a_clarification_still_does(busy):
    ctx, room, delivered = busy
    project = lambda metadata, **kw: server_owner_routing._route_project_chat_to_running_task(  # noqa: E731
        ctx, room["chat_id"], "Compare the two drafts", "msg-1", task_metadata=metadata, roots=_Roots(), **kw)
    main = lambda metadata, **kw: server_owner_routing._route_main_followup(  # noqa: E731
        ctx, 1, "Compare the two drafts", "msg-1", task_metadata=metadata, roots=_Roots(), **kw)
    # A clarification without a pick keeps its path: the one work reading the room takes it.
    assert project({}) == main({}) == "running-a" and delivered == ["running-a", "running-a"]
    delivered.clear()
    # Accepted as new work: neither reader is asked, so the words start their own turn.
    assert project({"new_work": ASSIGNED}) == main({"new_work": ASSIGNED}) == "" and delivered == []
    # Anything but that acceptance's whole answer is no such word.
    for broken in ("beta", {"model_choice": {"kind": "model"}}, None):
        delivered.clear()
        assert project({"new_work": broken}) == main({"new_work": broken}) == "running-a"
    # A late answer's own thread and an agent's note keep their readers whatever the metadata carries.
    delivered.clear()
    rows = _Roots()
    rows.rows = lambda chat_id=None: [{"task_id": "running-a", "room_reader": True, "late_input_of": "author", "chat_id": chat_id}]
    assert server_owner_routing._route_project_chat_to_running_task(
        ctx, room["chat_id"], "An answer", "msg-2", task_metadata={"new_work": ASSIGNED}, thread_of="author", roots=rows) == "running-a"
    assert main({"new_work": ASSIGNED}, note={"author": {"kind": "agent", "label": "fiction"}}) == "running-a"


def test_the_turn_starts_on_what_was_assigned_and_is_told_it_is_new_work(rooms, monkeypatch):
    from supervisor import worker_chat_lane

    root, room = rooms
    turns = []
    ctx = _ctx(root, direct=lambda *_a, **kwargs: turns.append(kwargs["task_metadata"]))
    monkeypatch.setattr(routing, "_recent_root_results", lambda *_a: ([], {}))
    monkeypatch.setattr(server_owner_routing, "_carry_project_room_context", lambda *a, **kw: None)
    for chat_id, project_id in ((room["chat_id"], room["id"]), (1, "")):
        turns.clear()
        server_owner_routing._route_owner_message(object(), ctx, {
            "chat_id": chat_id, "text": "Compare the two drafts", "client_message_id": "msg-1", "source": "web",
            "log_text": "Compare the two drafts", "task_metadata": {"new_work": ASSIGNED}})
        [metadata] = turns
        assert metadata["routing_contract"]["new_work"] == {"rule": routing.NEW_WORK_RULE}
        # The seed turns the acceptance's answer into the turn's own carriers; its admission then adds nothing.
        task = worker_chat_lane._seed_task(chat_id, "Compare the two drafts", None, None, metadata)
        assert task["metadata"]["lead"] == {"source": "room", "model": BETA, "use_local": False}
        assert task["metadata"]["processing"] == {"source": "room", "tier": "economy"}
        assert "new_work" not in task["metadata"]
        stamped = (dict(task["metadata"]["lead"]), dict(task["metadata"]["processing"]))
        update_project_model_choice(root, room["id"], OTHER)  # a later change of the room reaches only later work
        route_choice.capture_lead(task, root)
        assert (task["metadata"]["lead"], task["metadata"]["processing"]) == stamped
        update_project_model_choice(root, room["id"], {"kind": "main"})
    # An ordinary turn is told nothing of the kind and takes its room's choice at its own admission.
    turns.clear()
    server_owner_routing._route_owner_message(object(), ctx, {
        "chat_id": room["chat_id"], "text": "Plain words", "client_message_id": "msg-2", "source": "web",
        "log_text": "Plain words", "task_metadata": {}})
    assert "new_work" not in turns[0]["routing_contract"]
    plain = worker_chat_lane._seed_task(room["chat_id"], "Plain words", None, None, turns[0])
    assert "lead" not in plain["metadata"] and "processing" not in plain["metadata"]


def test_only_the_web_acceptances_answer_rides_the_bus(rooms, monkeypatch):
    root, _room = rooms
    bridge, _echoes = _web_bridge(root, monkeypatch)
    bridge.enqueue_local_message("From elsewhere", source="skill:fiction", chat_id=77, user_id=77,
                                 task_metadata={"new_work": ASSIGNED, "kept": True})
    bridge.enqueue_local_message("From the web acceptance", source="web", task_metadata={"new_work": ASSIGNED})
    elsewhere, web = (update["message"]["task_metadata"] for update in bridge.get_updates(0, timeout=0))
    assert elsewhere == {"kept": True} and web == {"new_work": ASSIGNED}


def test_the_carriers_are_built_only_from_a_whole_answer():
    assert route_choice.new_work_carriers({"model_choice": CHOSEN, "processing_choice": "fast"}) == {
        "lead": {"source": "room", "model": BETA, "use_local": False}, "processing": {"source": "room", "tier": "fast"}}
    assert route_choice.new_work_carriers({"model_choice": {"kind": "main"}}) == {
        "lead": {"source": "main"}, "processing": {"source": "default"}}
    for broken in (None, "beta", {}, {"model_choice": "beta"}, {"model_choice": {"kind": "model", "model": ""}},
                   {"model_choice": {"kind": "model", "model": BETA}}):
        assert route_choice.new_work_carriers(broken) == {}
