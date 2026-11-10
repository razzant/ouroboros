"""#2107: a head that a late answer, or a press on a finished author's card, starts is named by
what it answers: the accepted question, or the card's title as the press recorded it.

The real decision ingress, bridge intake and conversation-lane door (the admission's RUNNING row and the
queue document it publishes), a lane carrier's real body (``worker_process.run_worker_task``: the first
step, the namer at first work, agent.py's own RUNNING record), the supervisor's ``task_named`` handler,
routing receipts and their replay, the works projection, the real park and Resume, and the real Restart
run; only the model body is inert and Light is a double that records what it was asked to name. The
model still reads the host frame and the owner's row keeps the owner's words. The admission name is the
namer's fallback: in Main the one namer at first work reads what the head answers and the owner's words
(never the frame) and may replace it, and the queue row follows; a failed Light call leaves it standing.
Inputs that join the head never rename it, an authored title wins, a missing snapshot falls back to the
owner's own words, and a refused admission leaves no row.
"""
from __future__ import annotations

import logging
import queue as queue_mod
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros.owner_quiz import reconcile_terminal, record_asked
from ouroboros.task_results import list_task_results, load_task_result
from supervisor import workers
from supervisor.active_activity import get_direct_activity_registry
from tests.test_agent_restart_intent import runtime  # noqa: F401  # the shared real-queue fixture

QUESTION = "Which **database** should the new service use?\nIt writes a lot and reads little."
NAME = "Which database should the new service use?"
QUESTIONS = (("q1", QUESTION), ("q2", "Should migrations run on deploy?"), ("q3", "Keep the old schema?"))
COINED = "Postgres for the new service"


_LIGHTS: list = []  # this test's Light doubles, for _namers_settle_before_the_patches_undo


class _Light:
    """Light as the namer meets it: the client the real ``project_naming.llm_project_name`` is handed
    (``llm_client``). Records each request it is asked to name, then coins ``coin`` or fails. ``hold`` keeps
    the call open until it is set (a namer that settles late)."""

    def __init__(self, coin="", hold=None):
        self.coin, self.hold, self.requests = coin, hold, []
        _LIGHTS.append(self)

    def chat(self, messages, **_kwargs):
        self.requests.append(str(messages[0]["content"]).split("Owner request:\n", 1)[1])
        if self.hold is not None:
            assert self.hold.wait(10)
        if not self.coin:
            raise RuntimeError("Light is unavailable")
        return {"role": "assistant", "content": self.coin}, {}


class _Supervisor:
    """The supervisor's consumer of a carrier's frames, as far as names go: every frame is kept, and a
    ``task_named`` frame reaches the server's own handler (``events.EVENT_HANDLERS``) over the live pool
    table and the bridge. While ``holding``, a ``task_named`` frame is still in flight and kept back; ``release``
    hands the kept ones on, ``crossed`` the last sent first (two carrier processes' frames cross on the way to
    the server); a frame never released was lost with its carrier."""

    def __init__(self, bridge):
        self.bridge, self.frames, self.held, self.holding = bridge, [], [], False

    def put(self, item, *_args, **_kwargs):
        self.frames.append(item)
        if isinstance(item, dict) and item.get("type") == "task_named":
            (self.held.append if self.holding else self._handle)(item)

    def release(self, *, crossed=False):
        held, self.held, self.holding = self.held, [], False
        for item in reversed(held) if crossed else held:
            self._handle(item)

    def _handle(self, item):
        from supervisor.events import EVENT_HANDLERS

        EVENT_HANDLERS["task_named"](item, SimpleNamespace(RUNNING=workers.RUNNING, PENDING=workers.PENDING,
                                                           bridge=self.bridge))


class _Head:
    """The actor a lane carrier runs, with no model: agent.py's own RUNNING record, then the first work
    frame that starts the carrier's namer. ``during`` runs once the head is live, before that record; the
    namer's result settles (its frame handled as the supervisor handles it) while the head still runs;
    ``after`` runs then, as later owner input would. ``returns`` builds the frames its body returns."""

    def __init__(self, root, during=None, after=None, returns=None, settle=True):
        self.env = SimpleNamespace(drive_root=root, budget_drive_root=root)
        self._task_started_ts = time.time()
        self.root, self.during, self.after, self.returns, self.settle = root, during, after, returns, settle
        self.task, self.live, self.named = None, [], []

    def handle_task(self, task):
        from ouroboros.agent import OuroborosAgent

        self.task = task
        if self.during is not None:
            self.during(task)
        OuroborosAgent._persist_running_record(self, task)
        self.live = _names(self.root, task["id"])  # at its start, before its first work
        # The first non-addressing tool call: the frame that starts a Main turn's namer.
        self._event_queue.put({"type": "log_event", "data": {
            "task_id": task["id"], "type": "tool_call_started", "tool": "repo_read"}})
        if self.settle:
            _join_namers(task["id"])
        self.named = _names(self.root, task["id"])
        if self.after is not None:
            self.after(task)
        return self.returns(task) if self.returns is not None else []


def _names(root, task_id):
    """The head's name as its queue consumers read it: its queue row (what a park, a detach or a Resume
    takes), the published queue document (what a Restart restores) and the owner's works list."""
    from ouroboros.queue_observation import load_queue_document
    from tests._work_projection_shared import state_fields

    live = (workers.RUNNING.get(task_id) or {}).get("task") or next(
        (row for row in workers.PENDING if row.get("id") == task_id), {})
    document = load_queue_document(root)
    rows = [*(document.get("running") or []), *(document.get("pending") or [])]
    published = next((row.get("task") or {} for row in rows
                      if str((row.get("task") or {}).get("id") or "") == task_id), {})
    works = [work.get("name") for work in state_fields(root)["works"] if task_id in work["members"]["ids"]]
    return [live.get("suggested_name"), published.get("suggested_name"), works]


def _lane(root, monkeypatch, coin="", *, hold=None, carrier=True):
    """The real decision ingress and bridge over ``root``, Light ``_Light(coin, hold)``; with ``carrier``, an
    initialized queue (its document is published) and one READY lane spare."""
    import supervisor.message_bus as mb
    from tests.test_quiz_answer import _decision_app
    from tests.test_quiz_late_answer import _late_bridge

    spare = None
    if carrier:
        from supervisor import conversation_lane
        from tests._budget_pause_exact_helpers import _install_queue
        from tests.conversation_carrier_support import install_lane_spare

        _install_queue(root, monkeypatch)
        monkeypatch.setattr(conversation_lane, "_ARMED_TABLE", None)  # the one spare carries the head
        monkeypatch.setattr(conversation_lane, "_SPAWN_REFUSED", [""])
        monkeypatch.setattr(workers, "_get_chat_agent", lambda: pytest.fail("the server constructed an agent"))
        spare = install_lane_spare(workers)
        assert workers.open_repo_writer_admission()
    bridge, frames = _late_bridge(root, monkeypatch)
    monkeypatch.setattr(mb, "load_state", lambda: {"session_id": "s-1"})
    monkeypatch.setattr(mb, "publish_event", lambda *_a, **_k: None)
    monkeypatch.setattr(workers, "get_event_q", lambda: queue_mod.Queue())
    # Light reaches the namer as its client, never as ``ouroboros.llm.LLMClient``: a module the lane imports first
    # (the carrier's ``agent`` and its loop) would keep that name bound to Light for the rest of the process (#2853).
    from ouroboros import project_naming

    light, namer = _Light(coin, hold), project_naming.llm_project_name
    monkeypatch.setattr(project_naming, "llm_project_name",
                        lambda *args, **kwargs: namer(*args, **{**kwargs, "llm_client": light}))
    monkeypatch.setattr(project_naming, "_light_naming_model", lambda: "light-test")
    for send in ("chat", "chat_async"):  # any other model client of the lane sends nothing
        monkeypatch.setattr(f"ouroboros.llm.LLMClient.{send}", lambda *_a, **_k: pytest.fail("no provider call"))
    get_direct_activity_registry().clear()
    return SimpleNamespace(bridge=bridge, frames=frames, app=_decision_app(root, monkeypatch), light=light,
                           spare=spare, supervisor=_Supervisor(bridge))


def _admit(lane, args, kwargs, head):
    """The owner's message through the real door to the lane spare; its RUNNING row's name; then the
    carrier's real body with ``head`` (``conversation_carrier_support.carry``)."""
    from supervisor.worker_chat_lane import handle_chat_direct
    from tests.conversation_carrier_support import carry

    before = set(workers.RUNNING)
    handle_chat_direct(*args, **kwargs)
    [task_id] = [tid for tid, meta in workers.RUNNING.items()
                 if meta.get("lane") == "conversation" and tid not in before]
    admitted = workers.RUNNING[task_id]["task"].get("suggested_name")
    [ran] = carry(workers, lane.spare, head, out_q=lane.supervisor)
    assert ran["id"] == task_id
    return admitted


def _intake(root, monkeypatch, bridge, running=None):
    """One pass of the real bridge intake, its threads inline; returns the fresh heads it started."""
    import server
    from tests.test_v6730_origin_invariant import _ImmediateThread

    started: list = []
    ctx = SimpleNamespace(
        DRIVE_ROOT=root, PENDING=[], RUNNING=dict(running or {}),
        load_state=lambda: {"owner_id": 1, "owner_chat_id": 1, "session_id": "s-1"},
        update_state=lambda fn: fn({"owner_id": 1, "owner_chat_id": 1}),
        consciousness=SimpleNamespace(inject_observation=lambda _t: None, pause=lambda: None, resume=lambda: None),
        get_chat_agent=lambda: SimpleNamespace(_busy=False),
        handle_chat_direct=lambda *a, **k: started.append((a, k)),
        send_with_budget=lambda *a, **k: None,
    )
    with monkeypatch.context() as inline:
        inline.setattr(threading, "Thread", _ImmediateThread)
        server._process_bridge_updates(bridge, 0, ctx)
    return started


def _room(root, room):
    """Main, or the Project room ``racer``: its chat id and how a receipt labels a work there."""
    if room == "main":
        return 1, lambda name: name
    from ouroboros.project_facts import project_display_name
    from ouroboros.projects_registry import create_project

    project = create_project(root, "racer")
    shown = project_display_name(project.get("name"), project["id"])
    return int(project["chat_id"]), lambda name: f"{shown} › {name}"


def _asker(root, chat_id):
    """task-1 asked its questions in ``chat_id`` and finished."""
    for quiz_id, question in QUESTIONS:
        record_asked(root, "task-1", quiz_id=quiz_id, question=question, options=["sqlite", "postgres"],
                     assumption="sqlite meanwhile", chat_id=chat_id)
    reconcile_terminal(root, "task-1")


def _answer(lane, quiz_id, option_index, request_id):
    from tests.test_quiz_answer import _post

    response = _post(lane.app, {"request_id": request_id, "decision_id": f"quiz:task-1:{quiz_id}",
                                "option_index": option_index})
    assert response.status_code == 200 and response.json()["forwarded"] is True, response.text


def _join_namers(task_id):
    for thread in threading.enumerate():
        if thread.name in {f"namer-{task_id}", f"namer-call-{task_id}"}:
            thread.join(10)
            assert not thread.is_alive()


@pytest.fixture(autouse=True)
def _namers_settle_before_the_patches_undo(monkeypatch):
    """A namer a test leaves waiting in Light (a ``hold`` never set: the test failed or ended first) is released and
    joined while this test's patches still stand, never finishing later in the next test's queue and Light. Its
    ``monkeypatch`` argument orders this teardown before monkeypatch's own undo."""
    previous_threads = set(threading.enumerate())
    yield
    lights, _LIGHTS[:] = list(_LIGHTS), []
    for light in lights:
        if light.hold is not None:
            light.hold.set()
    for thread in threading.enumerate():
        if thread not in previous_threads and thread.name.startswith("namer-"):
            thread.join(10)
            assert not thread.is_alive(), f"{thread.name} did not settle"


def _receipt(lane, root, client_message_id, chat_id):
    """The joined input's receipt as the live frame says it, and whether a reload replays the same."""
    from tests.test_routing_receipt_facts import _annotation_frames, _replay, _shown

    frames = [frame for frame in _annotation_frames(lane.frames) if frame.get("client_message_id") == client_message_id]
    frame = frames[0] if len(frames) == 1 else {}
    return [frame.get("action"), frame.get("status"), frame.get("target_label"), frame.get("target_titles"),
            bool(frame) and _shown(_replay(root, client_message_id, chat_id)) == _shown(frame)]


def _observe(root, lane, head, admitted, joined, chat_id):
    """What every consumer shows for the head once it ended: its admitted name, its queue row, published
    document and works name at its start and after the namer, Light's requests, its RUNNING row, the live
    name frames, the receipts of the inputs that joined it before and after the namer (live and replayed),
    and those inputs waiting in its mailbox."""
    from ouroboros.owner_mailbox import drain_owner_entries

    tid = head.task["id"]
    return {
        "admitted": admitted, "at_start": head.live, "after_namer": head.named,
        "light_requests": list(lane.light.requests),
        "running_row": (load_task_result(root, tid) or {}).get("suggested_name"),
        "task_named": [frame["suggested_name"] for frame in lane.frames
                       if frame.get("type") == "task_named" and frame.get("task_id") == tid],
        "receipts": [_receipt(lane, root, client_message_id, chat_id) for client_message_id in joined],
        "joined_unread": len(drain_owner_entries(root, tid, set())),
    }


def _expected(seed, label, *, light_request="", coined=""):
    """The head named ``seed`` at admission; in Main Light was asked ``light_request`` and coined ``coined``."""
    final = coined or seed
    return {"admitted": seed, "at_start": [seed, seed, [seed]], "after_namer": [final, final, [final]],
            "light_requests": [light_request] if light_request else [], "running_row": final,
            "task_named": [seed, coined] if coined else [seed],
            "receipts": [["mailbox_delivery", "delivered", label(seed), [seed], True],
                         ["mailbox_delivery", "delivered", label(final), [final], True]],
            "joined_unread": 2}


@pytest.mark.parametrize("room,coin", [("main", COINED), ("main", ""), ("project", "")],
                         ids=["main-light-coins", "main-light-fails", "project"])
def test_a_late_answer_head_is_named_by_its_question_on_every_surface(tmp_path, monkeypatch, room, coin):
    from ouroboros.owner_quiz import late_answer_model_text
    from tests.test_quiz_late_answer import LATE_HEAD, _accepted_rows

    chat_id, label = _room(tmp_path, room)
    _asker(tmp_path, chat_id)
    lane = _lane(tmp_path, monkeypatch, coin)
    _answer(lane, "q1", 1, "r1")
    [(args, kwargs)] = _intake(tmp_path, monkeypatch, lane.bridge)

    def joined(quiz_id, request_id):  # the asker's next late answer arrives while the head runs
        def answer(_task):
            _answer(lane, quiz_id, 0, request_id)
            assert _intake(tmp_path, monkeypatch, lane.bridge, workers.RUNNING) == [], "it joins its live head"
        return answer

    head = _Head(tmp_path, during=joined("q2", "r2"), after=joined("q3", "r3"))
    admitted = _admit(lane, args, kwargs, head)

    # The model's frame and the owner's words are untouched.
    assert head.task["text"] == late_answer_model_text(tmp_path, kwargs["task_metadata"]["late_answer"], args[1])
    assert head.task["text"].startswith(LATE_HEAD.format("task-1") + "\n[Owner quiz answer] quiz q1")
    assert args[1] == "2. postgres" == _accepted_rows(tmp_path, "quiz_late_answer:task-1:q1")[0]["text"]
    assert head.task["origin_message_text"] == "2. postgres"
    light = " ".join(f"{QUESTION}\n2. postgres".split()) if room == "main" else ""
    assert _observe(tmp_path, lane, head, admitted, ["quiz_late_answer:task-1:q2", "quiz_late_answer:task-1:q3"],
                    chat_id) == _expected(NAME, label, light_request=light, coined=coin)


@pytest.mark.parametrize("room,coin", [("main", COINED), ("main", ""), ("project", "")],
                         ids=["main-light-coins", "main-light-fails", "project"])
def test_a_card_press_head_is_named_by_the_cards_title_on_every_surface(tmp_path, monkeypatch, room, coin):
    from ouroboros.gateway.card_decision import card_press_model_text
    from tests.test_card_press_thread import _card, _press
    from tests.test_quiz_late_answer import _accepted_rows

    chat_id, label = _room(tmp_path, room)
    lane = _lane(tmp_path, monkeypatch, coin)
    card = _card(tmp_path, chat_id, bridge=lane.bridge)
    assert _press(lane.app, card, "r1")["delivery"] == "forwarded"
    [(args, kwargs)] = _intake(tmp_path, monkeypatch, lane.bridge)

    def joined(request_id):
        def press(_task):
            assert _press(lane.app, card, request_id)["delivery"] == "forwarded"
            assert _intake(tmp_path, monkeypatch, lane.bridge, workers.RUNNING) == [], "it joins its live head"
        return press

    head = _Head(tmp_path, during=joined("r2"), after=joined("r3"))
    admitted = _admit(lane, args, kwargs, head)

    assert head.task["text"] == card_press_model_text(tmp_path, kwargs["task_metadata"]["card_press"], "Book it")
    assert head.task["text"].startswith("[Card input] The owner pressed «Book it» on the card «Flights»")
    assert args[1] == "Book it" == _accepted_rows(tmp_path, f"card_input:{card['card_id']}:r1")[0]["text"]
    joined_ids = [f"card_input:{card['card_id']}:{request_id}" for request_id in ("r2", "r3")]
    assert _observe(tmp_path, lane, head, admitted, joined_ids, chat_id) == _expected(
        "Flights", label, light_request="Flights Book it" if room == "main" else "", coined=coin)


def _snapshot_answer(root):
    """task-1's q1 answered after it finished, with the host's accepted snapshot the ingress carries."""
    from tests.test_quiz_late_answer import _answered_late

    block = _answered_late(root, option_index=1)
    return {"task_id": "task-1", "quiz_id": "q1", "answer_snapshot": {"task_id": "task-1", "quiz": block}}


def _turn(lane, root, words, metadata, head=None):
    """One owner message with ``metadata`` admitted and carried: the head that ran it."""
    head = head or _Head(root)
    _admit(lane, (1, words, None), {"task_metadata": metadata}, head)
    return head


def test_an_ordinary_turn_keeps_its_namer_and_a_typed_one_without_provenance_is_ordinary(tmp_path, monkeypatch):
    """The control: an owner turn without valid late-input provenance gets no admission name and is not born
    named, and the first work still asks Light to name its own text; a failure falls back to that text. Its
    queue row stays nameless after the namer's frame: such a row keeps reading its durable result."""
    lane = _lane(tmp_path, monkeypatch)
    heads = [_turn(lane, tmp_path, words, metadata)
             for words, metadata in (("what else is open?", {"client_message_id": "m-1"}),
                                     ("2. postgres", {"client_message_id": "m-2",
                                                      "late_answer": {"task_id": "task-1"}}))]
    for head, words in zip(heads, ("what else is open?", "2. postgres")):
        assert (head.task["text"], head.task.get("suggested_name"), head.live[0], head.named[0]) == (
            words, None, None, None)
        assert load_task_result(tmp_path, head.task["id"])["suggested_name"] == words
    assert lane.light.requests == ["what else is open?", "2. postgres"]
    assert [frame["suggested_name"] for frame in lane.frames if frame.get("type") == "task_named"] == [
        "what else is open?", "2. postgres"], "only the namer's own fallback, as before"


@pytest.mark.parametrize("kind", ["late_answer", "card_press"])
def test_a_head_without_its_snapshot_is_named_by_the_owners_own_words(tmp_path, monkeypatch, kind):
    """A message accepted without the question snapshot (an older intake), or a press whose stored press can no
    longer be read, keeps its frame's disclosure; the owner's words name it and are all Light is asked to name,
    never a question re-read or invented for it, nor the frame."""
    from tests.test_card_press_thread import _card

    lane = _lane(tmp_path, monkeypatch)
    if kind == "late_answer":
        _snapshot_answer(tmp_path)  # the canonical block exists; the message just does not carry it
        words, metadata = "prod parity", {"late_answer": {"task_id": "task-1", "quiz_id": "q1"}}
    else:
        card = _card(tmp_path, 1)
        words, metadata = "Book it", {"card_press": {"task_id": "task-1", "card_id": card["card_id"],
                                                     "request_id": "gone"}}
    head = _turn(lane, tmp_path, words, {"client_message_id": "late-1", **metadata})
    assert head.task["text"] != words, "the model still reads its host frame"
    assert (head.task.get("suggested_name"), load_task_result(tmp_path, head.task["id"]).get("suggested_name"),
            lane.light.requests) == (words, words, [words])


def test_an_authored_title_wins_and_needs_no_namer(tmp_path, monkeypatch):
    lane = _lane(tmp_path, monkeypatch, COINED)
    head = _turn(lane, tmp_path, "2. postgres", {
        "client_message_id": "quiz_late_answer:task-1:q1", "late_answer": _snapshot_answer(tmp_path),
        "title": "Release plan"})
    assert (head.task.get("suggested_name"), head.live, head.named,
            load_task_result(tmp_path, head.task["id"]).get("suggested_name"), lane.light.requests) == (
        "Release plan", ["Release plan", "Release plan", ["Release plan"]],
        ["Release plan", "Release plan", ["Release plan"]], "Release plan", [])


@pytest.mark.parametrize("change", ["retitled", "removed"])
def test_a_press_admitted_after_its_card_changed_reads_the_press_it_accepted(tmp_path, monkeypatch, change):
    """The frame and the name both come from the stored press, which keeps the title it was made on: a card
    another author retitled since names the head by the title the owner pressed; a card whose file is gone is
    disclosed in the frame and the owner's words name the head."""
    from ouroboros.card_store import _card_path, record_submission, write_card
    from tests.test_card_press_thread import ACTION, _card

    lane = _lane(tmp_path, monkeypatch)
    card = _card(tmp_path, 1)
    record_submission(tmp_path, card["card_id"], request_id="r1", action_id="book", label="Book it", values=None,
                      once=False, revision=1)
    if change == "retitled":
        write_card(tmp_path, chat_id=1, task_id="task-2", title="Trains", fallback_text="C or D", kind="components",
                   body={"components": [ACTION], "data": {}}, replace=card["card_id"])
    else:
        _card_path(tmp_path, card["card_id"]).unlink()
    press = {"task_id": "task-1", "card_id": card["card_id"], "request_id": "r1"}
    head = _turn(lane, tmp_path, "Book it", {"client_message_id": f"card_input:{card['card_id']}:r1",
                                             "card_press": press})
    if change == "retitled":
        assert head.task["text"] == (f"[Card input] The owner pressed «Book it» on the card «Flights» that task "
                                     f"task-1 showed (card {card['card_id']}).")
        assert (head.task["suggested_name"], lane.light.requests) == ("Flights", ["Flights Book it"])
    else:
        assert head.task["text"] == ("[Card input] On a card of task task-1 that can no longer be read, "
                                     "the owner pressed:\nBook it")
        assert (head.task["suggested_name"], lane.light.requests) == ("Book it", ["Book it"])


@pytest.mark.parametrize("settles", ["before_the_park", "after_the_resume",
                                     "lost_in_a_restart"])
def test_a_parked_head_keeps_the_coined_name_into_its_resumed_attempt(tmp_path, monkeypatch, settles):
    """A Main head whose namer coined a name is paused and parked by the supervisor (RUNNING -> PENDING, the
    one path a park and a detach take), then Resumed on its conversation carrier (never the idle project worker beside
    it), whose resumed attempt runs no namer. The namer writes
    the record before its ``task_named`` frame, and the server handles that frame before the park, after it,
    after the Resume handed the worker the row's older copy, crossed with the resumed attempt's own first frame
    (two carrier processes' frames), or never: it was lost with its carrier when the application restarted
    (the real stop, snapshot and restore). Each time the resumed head's queue row, published document, works
    list, start record, live card and the receipt of the next answer joining it end on the coined name."""
    from ouroboros import budget_pause
    from supervisor import queue
    from supervisor.worker_process import run_worker_task
    from tests._budget_pause_exact_helpers import _fast_hold, _loop_ctx, _quiet_external
    from tests.conversation_carrier_support import dispatched_tasks, install_lane_spare, park_lane_pause
    from tests.test_restart_retention import _pool_events
    from tests.test_restart_saved_work import _stop_and_boot
    from tests.worker_pool_support import affirm_ready

    _asker(tmp_path, 1)
    hold = threading.Event()
    lost, crossing = settles == "lost_in_a_restart", settles == "crossing_the_resumed_start"
    late = settles in {"after_the_park", "after_the_resume", "crossing_the_resumed_start"}  # Light answers late
    early = settles in {"before_the_park", "after_the_park", "lost_in_a_restart"}  # coined before the Resume
    lane = _lane(tmp_path, monkeypatch, COINED, hold=hold if late else None)
    _fast_hold(monkeypatch, budget_pause)
    _quiet_external(monkeypatch, budget_pause)
    _answer(lane, "q1", 1, "r1")
    [(args, kwargs)] = _intake(tmp_path, monkeypatch, lane.bridge)

    def pause(task):  # the turn's exact pause, as its loop requests it and its body returns it
        ctx, limit = _loop_ctx(tmp_path, task["id"], direct=True)
        ctx.current_chat_id = 1
        with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
            budget_pause.request_pause(limit, rail=budget_pause.RAIL_GLOBAL_EXHAUSTED, scope="global",
                                       reason_text="budget")
        return [budget_pause.pause_event(task, raised.value.pause)]

    def settle(_task=None):  # the first attempt's namer settles: its record, then its frame
        hold.set()
        _join_namers(head.task["id"])

    in_flight = (lambda _task: setattr(lane.supervisor, "holding", True)) if lost else None  # its namer's frame
    head = _Head(tmp_path, during=in_flight, returns=pause, settle=not late)
    _admit(lane, args, kwargs, head)
    tid = head.task["id"]
    [event] = [frame for frame in lane.supervisor.frames if frame.get("type") == "budget_pause"]
    park_lane_pause(event, worker_id=lane.spare.wid)
    budget_pause.end_dispatch_fence(tid)
    assert tid not in workers.RUNNING and [row["id"] for row in workers.PENDING] == [tid]
    if settles == "after_the_park":
        settle()
    if lost:  # the frame dies with its carrier: the application stops and the next generation restores the queue
        lane.supervisor.held, lane.supervisor.holding = [], False
        monkeypatch.setattr(workers, "WORKERS", {})
        _pool_events(workers, monkeypatch)
        _stop_and_boot(queue, workers)
    parked = _names(tmp_path, tid)

    assert queue.resume_budget_paused_task(tid)["ok"]
    if lost:  # the restart's pool: its lane spare again
        lane.spare = install_lane_spare(workers)
    sent: list = []  # one READY idle project worker beside the lane's spare
    workers.WORKERS[0] = affirm_ready(SimpleNamespace(wid=0, busy_task_id=None, in_q=SimpleNamespace(put=sent.append)))
    workers.assign_tasks()
    [resumed] = dispatched_tasks(lane.spare)
    assert not [row for row in sent if row.get("type") != "worker_quiescence"], "my human's head returns on its lane"
    if settles == "after_the_resume":
        settle()
    lane.supervisor.holding = crossing
    again = _Head(tmp_path, during=settle if crossing else None)
    run_worker_task(again, dict(resumed), wid=lane.spare.wid, out_q=lane.supervisor, repo_dir=str(tmp_path),
                    drive_root=str(tmp_path))
    if crossing:  # the first attempt's frame reaches the server first, the resumed attempt's own first frame last
        lane.supervisor.release(crossed=True)
    _answer(lane, "q2", 0, "r2")
    assert _intake(tmp_path, monkeypatch, lane.bridge, running=workers.RUNNING) == []

    coined, admitted = [COINED, COINED, [COINED]], [NAME, NAME, [NAME]]
    assert {
        "first_attempt": head.named[0], "parked": parked, "dispatched": resumed["suggested_name"],
        "resumed_start": again.live, "after": _names(tmp_path, tid),
        "record": load_task_result(tmp_path, tid).get("suggested_name"),
        "live_card": [frame["suggested_name"] for frame in lane.frames
                      if frame.get("type") == "task_named" and frame.get("task_id") == tid][-1],
        "receipt": _receipt(lane, tmp_path, "quiz_late_answer:task-1:q2", 1),
    } == {
        "first_attempt": COINED if settles == "before_the_park" else NAME, "parked": coined if early else admitted,
        "dispatched": COINED if early else NAME, "resumed_start": admitted if crossing else coined, "after": coined,
        "record": COINED, "live_card": COINED, "receipt": ["mailbox_delivery", "delivered", COINED, [COINED], True],
    }
    assert lane.light.requests[0] == " ".join(f"{QUESTION}\n2. postgres".split())


@pytest.mark.parametrize("failure", ["busy", "unreadable"])
def test_a_result_the_handler_cannot_read_leaves_the_rows_name_standing(tmp_path, monkeypatch, caplog, failure):
    """The server's ``task_named`` handler reads a named row's result off Q and never waits on a busy store. When
    that read fails (the zero-wait probe meets contention, or the record cannot be read) the row's own name stands
    and is what the bridge broadcasts, logged: an older birth frame never renames a head whose row already carries
    its coined name. A newer name the namer wrote meanwhile waits for the next frame that reads it."""
    import ouroboros.task_results as task_results
    from ouroboros.store import StoreBusy, bounded_wait
    from ouroboros.task_results import write_task_result
    from supervisor.queue import _queue_lock

    _asker(tmp_path, 1)
    lane = _lane(tmp_path, monkeypatch, COINED)
    _answer(lane, "q1", 1, "r1")
    [(args, kwargs)] = _intake(tmp_path, monkeypatch, lane.bridge)
    head = _Head(tmp_path)
    _admit(lane, args, kwargs, head)
    tid, read, reads = head.task["id"], task_results.load_task_result, []

    def unread(root, task_id, **options):
        if task_id != tid:
            return read(root, task_id, **options)
        reads.append((_queue_lock._is_owned(), bounded_wait(10.0)))  # under Q? how long may it wait?
        if failure == "busy":
            raise StoreBusy("database is locked", reason="contention")
        raise ValueError(f"task result authority is unreadable or invalid: task:{task_id}")

    def frame(name, *, readable):
        with monkeypatch.context() as patched, caplog.at_level(logging.WARNING, logger="supervisor.conversation_lane"):
            if not readable:
                patched.setattr(task_results, "load_task_result", unread)
            lane.supervisor.put({"type": "task_named", "task_id": tid, "suggested_name": name})
        shown = [item["suggested_name"] for item in lane.frames if item.get("type") == "task_named"
                 and item.get("task_id") == tid]
        return [_names(tmp_path, tid), shown[-1]]

    coined, final = [COINED, COINED, [COINED]], "Postgres, with a write-ahead log"
    assert frame(NAME, readable=False) == [coined, COINED]  # an older birth frame
    write_task_result(tmp_path, tid, load_task_result(tmp_path, tid)["status"], suggested_name=final)
    # the namer's newer name, its result unread: the row waits; the works list reads a lane turn's record itself
    assert frame(final, readable=False) == [[COINED, COINED, [final]], COINED]
    assert frame(final, readable=True) == [[final, final, [final]], final]
    assert (head.named, reads) == (coined, [(False, 0.0), (False, 0.0)])
    assert [record.levelno for record in caplog.records
            if record.name == "supervisor.conversation_lane" and tid in record.getMessage()] == [logging.WARNING] * 2


@pytest.mark.parametrize("returns_named", [True, False])
def test_a_refused_head_leaves_no_row_and_a_deferred_one_is_saved_with_its_name(runtime, monkeypatch,  # noqa: F811
                                                                                 returns_named):
    """Restart intake closed (a named return) refuses the late answer before any row exists; otherwise
    it is saved for after Restart, its admission receipt carrying the name by value. Neither one runs,
    publishes a name early, or leaves a RUNNING row."""
    from ouroboros import server_restart
    from supervisor.worker_chat_lane import handle_chat_direct

    root = runtime.root
    lane = _lane(root, monkeypatch, carrier=False)
    late = _snapshot_answer(root)
    notices: list = []
    monkeypatch.setattr(runtime.workers, "send_with_budget", lambda *a, **k: notices.append(a[1]))
    monkeypatch.setattr(runtime.workers, "_get_chat_agent", lambda: pytest.fail("the head must not run"))
    monkeypatch.setattr(server_restart, "agent_restart_pending", lambda: True)
    monkeypatch.setattr(server_restart, "agent_restart_draining", lambda: True)  # only the drain saves a turn
    if returns_named:
        monkeypatch.setitem(runtime.queue.RETURNS_NAMED, "door", "managed_update")
    before = {row["task_id"] for row in list_task_results(root)}
    running = set(runtime.workers.RUNNING)
    handle_chat_direct(1, "2. postgres", None,
                       task_metadata={"client_message_id": "quiz_late_answer:task-1:q1", "late_answer": late})

    assert lane.light.requests == [] and set(runtime.workers.RUNNING) == running
    assert not [frame for frame in lane.frames if frame.get("type") == "task_named"]
    rows = [row for row in list_task_results(root) if row["task_id"] not in before]
    if returns_named:
        assert notices == ["Restart intake is closed. This turn was not accepted; send it after Restart."]
        assert rows == [] and runtime.queue.PENDING == []
        return
    assert notices == ["Saved for execution after Restart."]
    [saved] = runtime.queue.PENDING
    [row] = rows
    assert (saved["suggested_name"], saved["text"]) == ("Which db for the pilot?", "2. postgres")
    assert (row["task_id"], row["status"], row["host_admission"]["status"], row.get("suggested_name")) == (
        saved["id"], "scheduled", "accepted", "Which db for the pilot?")


def test_a_head_saved_for_restart_keeps_its_name_through_the_real_return(runtime, monkeypatch):  # noqa: F811
    """A late answer arriving while an accepted Restart drains is saved for after it; the real Restart writes the queue
    snapshot, the next generation restores it and returns the head to a worker, whose real start records it. Its
    admission name rides every hop: the works list names the running head, its RUNNING record keeps it, and the
    asker's next answer, joining it, is received under that name live and after a reload. A row saved without a
    name stays nameless: none is invented."""
    from ouroboros import agent_task_pipeline as pipeline
    from supervisor.worker_chat_lane import handle_chat_direct
    from supervisor.worker_process import run_worker_task
    from tests._work_projection_shared import state_fields
    from tests.test_agent_restart_intent import accept, enqueue, worker
    from tests.test_restart_retention import _pool_events
    from tests.test_restart_saved_work import _ack, _stop_and_boot

    root = runtime.root
    _asker(root, 1)
    lane = _lane(root, monkeypatch, carrier=False)
    accept(runtime)
    monkeypatch.setattr(runtime.workers, "_get_chat_agent", lambda: pytest.fail("the head must wait for Restart"))
    monkeypatch.setattr(runtime.workers, "send_with_budget", lambda *a, **k: None)
    _answer(lane, "q1", 1, "r1")
    [(args, kwargs)] = _intake(root, monkeypatch, lane.bridge)
    with monkeypatch.context() as drain:
        drain.setattr("ouroboros.server_restart.agent_restart_draining", lambda: True)  # it arrives in the drain
        handle_chat_direct(*args, **kwargs)
    saved = runtime.queue.PENDING[-1]
    enqueue(runtime, "older-1", chat_id=1)

    # The requester finishes, the accepted Restart runs, and the next generation boots from its snapshot.
    _, _delivered = worker(runtime, monkeypatch)
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **k: None)
    final_events: list = []
    pipeline.emit_task_results(SimpleNamespace(drive_root=root, repo_dir=root), None, None, final_events,
        {**runtime.workers.RUNNING["requester"]["task"], "chat_id": 0, "text": "A"}, "Finished A", {"rounds": 1},
        {"tool_calls": [], "reasoning_notes": []}, 0, root / "logs", ctx=runtime.ctx)
    runtime.workers.RUNNING.pop("requester")
    runtime.server._check_pending_restart_drain(runtime.host)
    for name, stand_in in (("_safe_restart_serialized", lambda *a, **k: (True, "")),
                           ("_begin_task_teardown", lambda **k: ("cancelled", "restart")),
                           ("_managed_update_pending_kwargs", lambda: {}), ("_request_restart_exit", lambda: None)):
        monkeypatch.setattr(runtime.server, name, stand_in)
    runtime.host.safe_restart, runtime.host.update_state = (lambda **k: (True, "")), (lambda f: f({}))
    runtime.host.kill_workers = lambda **k: None
    runtime.server._handle_restart_in_supervisor(final_events[-1], runtime.host)
    _ack(root, runtime.server._planned_delegate_restart_transaction_id)
    monkeypatch.setattr(runtime.workers, "WORKERS", {})
    _pool_events(runtime.workers, monkeypatch)
    runtime.restart._AGENT_RESTART.clear()
    runtime.queue.RETURNS_NAMED.clear()
    _stop_and_boot(runtime.queue, runtime.workers)
    restored = {row["id"]: dict(row) for row in runtime.queue.PENDING}
    _, returned = worker(runtime, monkeypatch)
    runtime.workers.assign_tasks()
    names = {work["work_id"]: work.get("name") for work in state_fields(root)["works"]}
    [handed] = [task for task in returned if task["id"] == saved["id"]]
    head = _Head(root)
    run_worker_task(head, dict(handed), wid=0, out_q=lane.supervisor, repo_dir=str(root), drive_root=str(root))

    _answer(lane, "q2", 0, "r2")
    assert _intake(root, monkeypatch, lane.bridge, running=runtime.workers.RUNNING) == []
    assert ([row["id"] for row in returned], restored[saved["id"]].get("suggested_name"),
            restored["older-1"].get("suggested_name"), names.get(saved["id"]), names.get("older-1"),
            load_task_result(root, saved["id"]).get("suggested_name"),
            [frame["suggested_name"] for frame in lane.frames
             if frame.get("type") == "task_named" and frame.get("task_id") == saved["id"]],
            _receipt(lane, root, "quiz_late_answer:task-1:q2", 1)) == (
        [saved["id"]], NAME, None, NAME, "", NAME, [NAME], ["mailbox_delivery", "delivered", NAME, [NAME], True])
