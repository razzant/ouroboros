"""A press on a finished author's card follows that author's thread, the way a late answer does.

While the author's admitted Continue successor lives, the press reaches it. Otherwise it is the
owner's own message typed ``card_press``: it joins the live head that an earlier late answer or
press of the same author started, in Main as in a Project, and never an unrelated reader; a
fresh head starts only when no such head lives. Every head reads the card frame rebuilt from the
stored press, while the owner's row keeps their words.
"""

from __future__ import annotations

import pytest

from ouroboros.card_store import record_submission, write_card
from tests.test_quiz_answer import _decision_app, _post
from tests.test_quiz_late_answer import _inbox, _intake, _late_answer_head, _late_answered, _late_bridge

ACTION = {"type": "action", "label": "Book it", "submit": {"id": "book"}}


def _card(root, chat_id, task_id="task-1", bridge=None):
    """``task_id`` showed an interactive card in ``chat_id`` (published when a bridge is given)."""
    card = write_card(root, chat_id=chat_id, task_id=task_id, title="Flights", fallback_text="A or B",
                      kind="components", body={"components": [ACTION], "data": {}})
    if bridge is not None:
        from supervisor.card_delivery import send_card

        assert send_card(bridge, card, chat_id, task_id=task_id, interactive=True) == (True, "ok")
    return card


def _ended_author(tmp_path, monkeypatch, chat_id):
    """task-1's card in ``chat_id``; task-1 ended, so a press is the owner's message."""
    import supervisor.message_bus as mb

    bridge, _frames = _late_bridge(tmp_path, monkeypatch)
    monkeypatch.setattr(mb, "load_state", lambda: {"session_id": "s-1"})
    return bridge, _decision_app(tmp_path, monkeypatch), _card(tmp_path, chat_id, bridge=bridge)


def _press(app, card, request_id):
    response = _post(app, {"request_id": request_id, "revision": 1, "decision_id": f"card:{card['card_id']}:book"})
    assert response.status_code == 200, response.text
    return response.json()


def _press_head(head_id, author, chat_id, card, request_id, *, project_id=""):
    """A live direct head that a press on ``author``'s card started (typed card_press provenance)."""
    head = _late_answer_head(head_id, author, chat_id, project_id=project_id)
    head._current_task_metadata = {**({"project_id": project_id} if project_id else {}),
                                   "card_press": {"task_id": author, "card_id": card["card_id"], "request_id": request_id}}
    return head


def _intake_once(tmp_path, monkeypatch, bridge, **kwargs):
    """One pass of the real bridge intake; real threads afterwards, so a later press can use its executor."""
    import threading

    thread = threading.Thread
    try:
        return _intake(tmp_path, monkeypatch, bridge, **kwargs)
    finally:
        monkeypatch.setattr(threading, "Thread", thread)


def _read(tmp_path, task_id):
    """One drain of ``task_id``'s mailbox: what the model reads and the owner directives it records."""
    import queue
    from types import SimpleNamespace

    from ouroboros.loop_round_limits import _drain_incoming_messages

    messages = []
    ctx = SimpleNamespace(task_id=task_id, task_metadata={}, drive_root=tmp_path, budget_drive_root=tmp_path)
    _drain_incoming_messages(messages, queue.Queue(), tmp_path, task_id, None, set(), owner_ctx=ctx)
    return str(messages), [row["content"] for row in getattr(ctx, "_owner_directives", [])]


def test_presses_on_a_finished_authors_card_are_one_thread_in_main(tmp_path, monkeypatch):
    """Main starts a head per owner message; a press is typed, so the next one joins the first one's head."""
    from ouroboros.owner_mailbox import drain_owner_entries

    bridge, app, card = _ended_author(tmp_path, monkeypatch, 1)
    assert _press(app, card, "r1")["delivery"] == "forwarded"
    [(_args, first)] = _intake_once(tmp_path, monkeypatch, bridge)
    assert first["task_metadata"]["card_press"] == {"task_id": "task-1", "card_id": card["card_id"], "request_id": "r1"}

    _press_head("head-1", "task-1", 1, card, "r1")
    assert _press(app, card, "r2")["delivery"] == "forwarded"
    assert _intake_once(tmp_path, monkeypatch, bridge) == [], "the next press joins the head the first one started"
    [entry] = drain_owner_entries(tmp_path, "head-1", set())
    assert (entry["text"], entry["card_press"]["request_id"]) == ("Book it", "r2")
    model, directives = _read(tmp_path, "head-1")
    assert f"[Card input] The owner pressed «Book it» on the card «Flights» that task task-1 showed (card {card['card_id']})." in model
    assert directives == ["Book it"], "the owner's own words stay their words"


@pytest.mark.parametrize("head_author", ["task-1", "task-other"])
def test_a_press_joins_the_head_a_late_answer_of_its_author_started(tmp_path, monkeypatch, head_author):
    """One author's late input is one thread: a late answer's head takes that author's press; another author's does not."""
    from ouroboros.owner_mailbox import drain_owner_entries

    bridge, app, card = _ended_author(tmp_path, monkeypatch, 1)
    _late_answer_head("head-1", head_author, 1)
    assert _press(app, card, "r1")["delivery"] == "forwarded"
    started = _intake_once(tmp_path, monkeypatch, bridge)
    if head_author == "task-1":
        assert started == [] and [e["card_press"]["request_id"] for e in drain_owner_entries(tmp_path, "head-1", set())] == ["r1"]
    else:
        assert len(started) == 1 and drain_owner_entries(tmp_path, "head-1", set()) == []


def test_a_late_answer_joins_the_head_a_press_of_its_asker_started(tmp_path, monkeypatch):
    from ouroboros.owner_mailbox import drain_owner_entries

    bridge = _late_answered(tmp_path, monkeypatch, 1)
    _press_head("head-1", "task-1", 1, _card(tmp_path, 1), "r0")
    assert _intake_once(tmp_path, monkeypatch, bridge) == []
    assert [e["late_answer"]["quiz_id"] for e in drain_owner_entries(tmp_path, "head-1", set())] == ["q1"]


def test_a_press_in_a_room_never_reaches_an_unrelated_reader(tmp_path, monkeypatch):
    """Room input goes to its sole reader, but a typed press is addressed: an unrelated reader never gets it."""
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.projects_registry import create_project

    chat_id = int(create_project(tmp_path, "racer")["chat_id"])
    bridge, app, card = _ended_author(tmp_path, monkeypatch, chat_id)
    unrelated = {"id": "busy-root", "chat_id": chat_id, "root_task_id": "busy-root",
                 "delegation_role": "root", "drive_root": str(tmp_path)}
    assert _press(app, card, "r1")["delivery"] == "forwarded"
    started = _intake_once(tmp_path, monkeypatch, bridge, running={"busy-root": {"task": unrelated}})
    assert drain_owner_entries(tmp_path, "busy-root", set()) == [], "the room's reader is not the card's author"
    [(_args, kwargs)] = started
    assert kwargs["task_metadata"]["card_press"]["task_id"] == "task-1"


def test_a_press_on_a_continued_tasks_card_reaches_its_successor(tmp_path, monkeypatch):
    """The author ended and was continued: its admitted successor holds the press, as it would an answer."""
    from ouroboros.owner_mailbox import KIND_CARD_INPUT, drain_owner_entries
    from tests.test_question_answer_routing import continue_to, runtime

    app, bridge, q, task = runtime(tmp_path, monkeypatch, running=False)
    continue_to(tmp_path)
    q.RUNNING["successor"] = {"task": {**task, "id": "successor"}}
    card = _card(tmp_path, 1, task_id="asker")
    answer = _press(app, card, "r1")
    assert answer["delivery"] == "delivered" and "reason_code" not in answer
    assert _inbox(bridge) == [], "no turn of its own"
    [entry] = drain_owner_entries(tmp_path, "successor", set())
    assert entry["kind"] == KIND_CARD_INPUT
    assert entry["text"] == f"[Card input] The owner pressed «Book it» on the card «Flights» that task asker showed (card {card['card_id']})."


def test_a_head_a_press_started_reads_the_card_frame(tmp_path, monkeypatch):
    import queue as queue_mod

    import supervisor.workers as workers
    from ouroboros.gateway.card_decision import card_press_model_text

    card = _card(tmp_path, 1)
    record_submission(tmp_path, card["card_id"], request_id="r1", action_id="book", label="Book it", values=None,
                      once=False, revision=1)
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "get_event_q", lambda: queue_mod.Queue())

    class _Agent:
        task = None

        def handle_task(self, task):
            self.task = task
            return []

    agent = _Agent()
    press = {"task_id": "task-1", "card_id": card["card_id"], "request_id": "r1"}
    workers._run_chat_task(agent, 1, "Book it", None,
                           task_metadata={"client_message_id": f"card_input:{card['card_id']}:r1", "card_press": press})
    assert agent.task["text"] == (f"[Card input] The owner pressed «Book it» on the card «Flights» that task task-1 showed "
                                  f"(card {card['card_id']}).")
    lost = card_press_model_text(tmp_path, {**press, "request_id": "gone"}, "Book it")
    assert lost == "[Card input] On a card of task task-1 that can no longer be read, the owner pressed:\nBook it"
    plain = _Agent()
    workers._run_chat_task(plain, 1, "Book it", None, task_metadata={"client_message_id": "m-2"})
    assert plain.task["text"] == "Book it"


def test_a_continue_admitted_between_choice_and_admission_still_gets_the_press(tmp_path, monkeypatch):
    """The author was chosen, then ended and was continued before its admission lock: the press follows
    the admitted successor under that one's own lock instead of falling through to the owner's message."""
    from ouroboros.gateway import task_decision as td
    from ouroboros.owner_mailbox import KIND_CARD_INPUT, drain_owner_entries
    from tests.test_question_answer_routing import continue_to, runtime

    app, bridge, q, task = runtime(tmp_path, monkeypatch)
    card = _card(tmp_path, 1, task_id="asker")
    holder, calls = td._mail_holder, []

    def ends_after_choice(task_id):
        calls.append(task_id)
        if calls.count("asker") == 2:  # the re-check under the lock: it ended, its Continue is admitted
            q.RUNNING.pop("asker", None)
            continue_to(tmp_path)
            q.RUNNING["successor"] = {"task": {**task, "id": "successor"}}
        return holder(task_id)

    monkeypatch.setattr(td, "_mail_holder", ends_after_choice)
    assert _press(app, card, "r1")["delivery"] == "delivered"
    assert _inbox(bridge) == [] and drain_owner_entries(tmp_path, "asker", set()) == []
    [entry] = drain_owner_entries(tmp_path, "successor", set())
    assert entry["kind"] == KIND_CARD_INPUT and "that task asker showed" in entry["text"]


def test_a_retry_of_an_accepted_press_converges_instead_of_reaching_a_later_continue(tmp_path, monkeypatch):
    """The owner's ingress accepted the press, then its delivery mark was lost; a Continue admitted since
    never gets a second copy: the retry converges on the accepted message."""
    import ouroboros.card_store as card_store
    import supervisor.message_bus as mb
    from ouroboros.owner_mailbox import drain_owner_entries
    from tests.test_question_answer_routing import continue_to, runtime

    app, bridge, q, task = runtime(tmp_path, monkeypatch, running=False)
    monkeypatch.setattr(mb, "load_state", lambda: {"session_id": "s-1"})
    card = _card(tmp_path, 1, task_id="asker", bridge=bridge)
    mark = card_store.mark_delivery
    monkeypatch.setattr(card_store, "mark_delivery", lambda *a, **k: None)  # the mark is lost
    assert _press(app, card, "r1")["delivery"] == "forwarded"
    monkeypatch.setattr(card_store, "mark_delivery", mark)
    assert card_store.load_card(tmp_path, card["card_id"])["submissions"][0]["delivered"] == "owed"

    continue_to(tmp_path)
    q.RUNNING["successor"] = {"task": {**task, "id": "successor"}}
    again = _press(app, card, "r1")
    assert again["duplicate"] is True and again["delivery"] == "forwarded"
    assert drain_owner_entries(tmp_path, "successor", set()) == []
    assert len(_inbox(bridge)) == 1, "one owner message for one press"
