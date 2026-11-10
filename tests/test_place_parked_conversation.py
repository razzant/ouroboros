"""A yielded Presence author hears its conversation once and retakes the lease before reasoning.

These real Host and worker consumers share the promotion/transport fixture with conversation-work tests;
review/private-question lending and reentry have their own lifecycle and failure cases here.
"""
from __future__ import annotations

import json
import os
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import places
from ouroboros.presence_admission import admit_presence_turn
from ouroboros.presence_bindings import conversation_key
from ouroboros.presence_runner import PresenceTurnEvent, capture_presence_event
from ouroboros.task_results import load_task_result
from tests.test_place_conversation_work import (
    PLACE_EVENT, SILENT, _recipient, attention, call, event_of, finish, place_rows,
    post, publish, reading, until,
)
from tests.test_place_conversation_work import host as host

pytestmark = pytest.mark.usefixtures("provider_catalog_offline")



QUESTION = call(("escalate", {"question": "May I start?", "options": [{"label": "Yes"}], "wait_for_answer": True,
                              "max_wait_minutes": 1, "request_id": "q1"}))
THREAD_KEY = "slack:T1:C1:1.0"


def _park_at_question(world, then, *, thread="1.0"):
    """A real focus through the Host parks at a real private question with its conversation lent."""
    from ouroboros.conversation_lease import live_author

    world.focus_steps[:] = [QUESTION, then]
    _source, reply = post(world, 8000, "Please start the migration.", thread=thread, facts=("any_message", "mention"))
    assert reply.status_code == 200, reply.text
    author = until(lambda: world.h.agents and world.h.calls >= 1 and world.h.agents[-1].tools._ctx.task_id, "the author")
    until(lambda: (load_task_result(world.data, author).get("owner_wait") or {}).get("state") == "waiting", "the park")
    key = conversation_key(world.provider, world.account, "C1", thread)
    until(lambda: not live_author(key, drive_root=world.data), "its conversation to be lent")
    return author


def test_a_real_question_park_wakes_for_an_untagged_message_and_the_same_author_reads_it_holding_its_lease(host):
    from ouroboros.conversation_lease import live_author
    from ouroboros.owner_quiz import quiz_states

    world = host()
    seen = {}

    def read(messages):
        seen.update(words=json.dumps(messages, ensure_ascii=False), holder=live_author(THREAD_KEY, drive_root=world.data),
                    calls=world.h.calls, directives=json.dumps(getattr(world.h.agents[0].tools._ctx, "_owner_directives", []) or [],
                                                               ensure_ascii=False))
        return finish("Noted; still waiting on the question.")

    author = _park_at_question(world, read)
    before = world.h.calls
    source, reply = post(world, text="Also migrate the archive.")
    assert reply.status_code == 200 and reply.json()["outcome"] == "silent", reply.text
    assert attention(world, source)["reason"] == "current_author" and attention(world, source)["recipient"] == author
    assert [row["place_arrival"]["wake"] for row in place_rows(world.data, author)] == [True]
    until(lambda: seen.get("words"), "the woken author's next model call")
    assert seen["calls"] == before + 1 and len(world.h.agents) == 1, "the SAME author: no second focus"
    assert seen["holder"] == author, "it took its conversation back before the model call"
    assert seen["words"].count("Also migrate the archive.") == 1, "the model reads it once, in its mail"
    assert PLACE_EVENT in seen["words"] and "Also migrate the archive." not in seen["directives"]
    assert "nothing new was recorded" not in seen["words"] and "they follow as place events" in seen["words"]
    until(lambda: (load_task_result(world.data, author) or {}).get("status") == "completed", "its terminal")
    states = [block.get("state") for block in quiz_states(world.data, author).values()]
    assert "answered" not in states, "place words never answer the owner's question"


def test_a_real_review_park_wakes_for_an_untagged_message_and_the_same_author_reads_it_holding_its_lease(host, monkeypatch):
    from ouroboros.conversation_lease import live_author
    from tests.test_presence_continuation import ANSWER, review, waits

    world = host(review_mode="auto")
    seen = {}

    def read(messages):
        seen.update(words=json.dumps(messages, ensure_ascii=False), holder=live_author(THREAD_KEY, drive_root=world.data),
                    agents=len(world.h.agents))
        return waits()

    world.focus_steps[:] = [review(answer=ANSWER), waits, read, finish(ANSWER)]
    _source, reply = post(world, 8100, "Please prepare the report.", facts=("any_message", "mention"))
    assert reply.status_code == 200, reply.text
    assert world.h.entered.wait(60), "the held reviewer never started"
    author = world.h.agents[-1].tools._ctx.task_id
    until(lambda: (load_task_result(world.data, author).get("owner_wait") or {}).get("reason") == "review"
          and (load_task_result(world.data, author).get("owner_wait") or {}).get("state") == "waiting", "the review park", 90)
    until(lambda: not live_author(THREAD_KEY, drive_root=world.data), "its conversation to be lent")
    source, reply = post(world, text="Include the archive numbers.")
    try:
        assert reply.status_code == 200 and attention(world, source)["recipient"] == author, reply.text
        until(lambda: seen.get("words"), "the woken author's next model call", 90)
        assert seen["agents"] == 1 and seen["holder"] == author and seen["words"].count("Include the archive numbers.") == 1
    finally:
        world.h.release.set()
    until(lambda: (load_task_result(world.data, author) or {}).get("status") == "completed", "its terminal", 120)
    assert world.h.settled.wait(60)


@pytest.mark.parametrize("passive", ["reaction", "place_paused", "off"])
def test_nothing_passive_or_controlled_wakes_a_real_question_park(host, monkeypatch, passive):
    from ouroboros import owner_pause

    world = host(wakes=("mention",))
    thread = "" if passive == "threadless" else "1.0"
    author = _park_at_question(world, finish("Done."), thread=thread)
    model_woke = threading.Event()
    scripted_model = world.h.script

    def observe_wake(messages):
        model_woke.set()
        return scripted_model(messages)

    world.h.script = observe_wake
    calls = world.h.calls
    arguments = {"text": "Words beside the park.", "thread": thread}
    if passive == "reaction":
        arguments.update(text="", facts=(), actor={"platform_actor_id": "U2", "kind": "user"}, message={
            "message_id": "9", "reaction": {"schema_version": 1, "change": "add", "old": [], "new": [{"type": "emoji", "value": "+1"}],
                                            "target": {"message_id": "1", "resolution": "resolved"}}})
    elif passive == "bot":
        arguments.update(facts=("any_message", "from_bot"), actor={"platform_actor_id": "B1", "is_bot": True})
    elif passive == "owner_pause":
        owner_pause.install_fence(world.data, author, request_id="p")
    elif passive == "budget_fence":
        world.queue.BUDGET_ROOT_FENCES[author] = {"status": "active", "root_task_id": author, "fence_id": "f"}
        publish(world)
    elif passive == "place_paused":
        places.edit_place(world.data, world.place.place_id, "set_state", author="owner", state="paused")
    elif passive == "off":
        monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: False)
    source, reply = post(world, **arguments)
    assert reply.status_code == 200 and reply.json()["outcome"] == "silent", (passive, reply.text)
    assert attention(world, source)["decision"] == "heard" and "recipient" not in attention(world, source), passive
    assert place_rows(world.data, author) == [], passive
    assert not model_woke.wait(1.5), (passive, "no round and no stand-in focus")
    assert world.h.calls == calls and len(world.h.agents) == 1, (passive, "no round and no stand-in focus")
    assert (load_task_result(world.data, author).get("owner_wait") or {}).get("state") == "waiting", passive
    # The control lifted, the same live park still wakes for the next real message.
    if passive == "owner_pause":
        owner_pause.release_fence(world.data, author, reason="test")
    elif passive == "budget_fence":
        world.queue.BUDGET_ROOT_FENCES.pop(author)
        publish(world)
    elif passive == "place_paused":
        places.edit_place(world.data, world.place.place_id, "set_state", author="owner", state="active")
    elif passive == "off":
        monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: True)
    facts = ("any_message", "mention") if passive == "threadless" else ("any_message",)
    woke, reply = post(world, text="Now a real message.", thread=thread, facts=facts)
    assert reply.status_code == 200 and attention(world, woke)["recipient"] == author, (passive, reply.text)
    until(lambda: (load_task_result(world.data, author) or {}).get("status") == "completed", "its terminal")


def test_a_parked_author_and_a_reading_root_get_one_routing_focus_listing_both_and_exact_tell_wakes_only_that_author(host):
    from ouroboros.conversation_lease import live_author

    world = host()
    seen, routed = {}, []

    def model(messages):
        words = json.dumps(messages, ensure_ascii=False)
        if "## Work open in this conversation" in words:  # the routing focus: its own context lists them
            routed.append(words)
            if len(routed) == 1:
                return call(("tell_task", {"task_id": seen["author"], "arrival": True}))
            return SILENT
        if "Which of you?" in words:  # the woken original author
            seen.update(words=words, holder=live_author(THREAD_KEY, drive_root=world.data))
            return finish("Noted.")
        return QUESTION

    world.h.script = model
    assert post(world, 8200, "Please start the migration.", facts=("any_message", "mention"))[1].status_code == 200
    seen["author"] = until(lambda: world.h.agents and world.h.calls >= 1 and world.h.agents[-1].tools._ctx.task_id, "the author")
    until(lambda: (load_task_result(world.data, seen["author"]).get("owner_wait") or {}).get("state") == "waiting", "the park")
    until(lambda: not live_author(THREAD_KEY, drive_root=world.data), "its conversation to be lent")
    root = reading(world, objective="The other work")
    calls = world.allowance.calls
    source, reply = post(world, text="Which of you?")
    assert reply.status_code == 200, reply.text
    assert world.allowance.calls > calls, "an ordinary focus under the allowance"
    until(lambda: seen.get("words"), "the chosen author's next model call", 90)
    assert seen["author"] in routed[0] and root["id"] in routed[0], "the whole set, the parked author included"
    assert place_rows(world.data, root["id"]) == [] and len(place_rows(world.data, seen["author"])) == 1, "never a broadcast"
    assert _recipient(world, source)["task_id"] == seen["author"]
    until(lambda: (load_task_result(world.data, seen["author"]) or {}).get("status") == "completed", "its terminal")


def test_a_message_left_to_the_mind_may_reach_a_later_current_author_but_a_held_one_never_moves(host):
    import threading as _threading

    from ouroboros.conversation_lease import PresenceTurnGate
    from ouroboros.place_arrivals import pin_recipient
    from ouroboros.place_intake import offer_arrival

    world = host()
    event = event_of(world, "left-to-the-mind", text="To whoever holds this now.", facts=("any_message",))
    admission = admit_presence_turn(drive_root=world.data, authenticated_transport_skill=world.skill, provider=world.provider,
                                    account_id=world.account, conversation_id="C1", thread_id="1.0")
    captured = capture_presence_event(admission, event, drive_root=world.data)
    mind = {"to": "mind", "why": "not_reading"}
    assert pin_recipient(world.data, captured["id"], event.conversation_key, mind, expect={}) == mind
    lease = PresenceTurnGate(state_root=world.data / "state").acquire(event.conversation_key)
    lease.bind_author("later-author")
    lease.intake_agent = SimpleNamespace(_owner_message_admission_lock=_threading.RLock(), _accepting_owner_messages=True,
                                         _busy=True, _current_task_id="later-author")
    try:
        assert offer_arrival(world.data, captured, event, wake=True) == ("current_author", "later-author")
        assert len(place_rows(world.data, "later-author")) == 1
        # It holds it now: a different current author later never takes it, whatever becomes of the first.
        lease.bind_author("yet-another-author")
        lease.intake_agent._current_task_id = "yet-another-author"
        assert offer_arrival(world.data, captured, event, wake=True) == ("current_author", "later-author")
        assert place_rows(world.data, "yet-another-author") == []
    finally:
        lease.release()


def test_a_real_question_park_in_a_spawned_carrier_wakes_for_an_untagged_message_with_nothing_local_to_this_host(host, monkeypatch, tmp_path):
    """The real turn in a real ``worker_main``: its real wait (``worker_owner_wait``) parks and lends through the real
    supervisor handlers, this Host's door reaches it through durable facts alone, and the wait's own mail read wakes it."""
    import multiprocessing

    from ouroboros.conversation_lease import _AUTHORS, live_author
    from ouroboros.owner_quiz import quiz_states
    from ouroboros.presence_carrier import presence_seed_task
    from supervisor.worker_owner_wait import _grant_resume, handle_owner_wait
    from tests.place_work_worker_child import run_carrier
    from tests.worker_pool_support import affirm_ready

    world = host()
    key = conversation_key(world.provider, world.account, "C1", "1.0")
    source_id = f"{world.provider}:{world.account}:8300"
    payload = {"source_event_id": source_id, "provider": world.provider, "account_id": world.account, "conversation_id": "C1",
               "thread_id": "1.0", "conversation_key": "ignored", "actor": {"platform_actor_id": "U2", "username": "person"},
               "conversation": {}, "message": {"message_id": "8300", "source_order": 8300},
               "text": "Please start the migration.", "facts": ["any_message", "mention"]}
    event = PresenceTurnEvent(source_id, world.provider, world.account, "C1", "1.0", key, payload["actor"], {}, payload["message"],
                              payload["text"], facts=tuple(payload["facts"]), continuation_version=2)
    admission = admit_presence_turn(drive_root=world.data, authenticated_transport_skill=world.skill, provider=world.provider,
                                    account_id=world.account, conversation_id="C1", thread_id="1.0")
    capture_presence_event(admission, event, drive_root=world.data)
    task = presence_seed_task(admission=admission, event=event, skill_name=world.skill, event_payload=payload,
                              staged_files=[], gate_wait=False)
    author = task["id"]
    task["_attempt"] = 1
    facts = tmp_path / "carrier-facts"
    facts.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(world.data))
    context = multiprocessing.get_context("spawn")
    in_q, out_q = context.Queue(), context.Queue()
    proc = context.Process(target=run_carrier, args=(in_q, out_q, str(world.h.repo), str(world.data), str(facts), key), name="place-carrier")
    proc.start()
    slot = affirm_ready(world.workers.Worker(7, proc, in_q, busy_task_id=author))
    world.workers.WORKERS[7] = slot
    meta = world.queue.RUNNING[author] = {"task": task, "worker_id": 7, "attempt": 1, "started_at": time.time()}
    publish(world)
    frames, stop = [], threading.Event()

    def pump():  # the supervisor's two real handlers for this wait, nothing else of its tick
        while not stop.is_set():
            try:
                frame = out_q.get(timeout=0.2)
            except Exception:
                continue
            frames.append(frame)
            if isinstance(frame, dict) and frame.get("type") == "owner_wait":
                handle_owner_wait(frame, world.workers)
                if frame.get("phase") == "resume":
                    _grant_resume(author, meta, slot)

    pumper = threading.Thread(target=pump, daemon=True)
    pumper.start()
    events = world.data / "logs" / "events.jsonl"
    try:
        until(lambda: events.exists() and any('"worker_ready"' in line and f'"pid": {proc.pid}' in line
                                              for line in events.read_text(encoding="utf-8").splitlines()),
              "the carrier's worker_ready row", 120)
        in_q.put(task)
        until(lambda: (load_task_result(world.data, author) or {}).get("owner_wait", {}).get("state") == "waiting", "the real park", 120)
        until(lambda: not live_author(key, drive_root=world.data), "its conversation to be lent")
        first = json.loads((facts / "carrier-1.json").read_text())
        assert first["pid"] == proc.pid != os.getpid() and first["holder"] == author
        assert not [item for item in _AUTHORS if item[1] == author], "nothing of this author is local to the Host"
        calls = world.allowance.calls
        source, reply = post(world, text="Also migrate the archive.")
        assert reply.status_code == 200 and attention(world, source)["recipient"] == author, reply.text
        assert world.allowance.calls == calls and world.focus_inputs == [], "no allowance and no second focus"
        until(lambda: (facts / "carrier-2.json").exists(), "the same author's next model call", 90)
        second = json.loads((facts / "carrier-2.json").read_text())
        assert second["pid"] == proc.pid and second["holder"] == author, "it retook its conversation before the model call"
        assert second["messages"].count("Also migrate the archive.") == 1
        until(lambda: (load_task_result(world.data, author) or {}).get("status") == "completed", "its terminal", 120)
        assert "answered" not in [block.get("state") for block in quiz_states(world.data, author).values()]
    finally:
        if proc.is_alive():
            in_q.put({"type": "shutdown"})
            proc.join(60)
        if proc.is_alive():
            proc.kill()
            proc.join(10)
        stop.set()
        pumper.join(10)
        for bus in (in_q, out_q):
            bus.close()
            bus.join_thread()
        world.queue.RUNNING.pop(author, None)


def test_the_reentry_note_leaves_only_mailed_messages_to_their_mail_and_its_source_stays_whole(host, monkeypatch, tmp_path):
    from ouroboros import presence_continuation
    from ouroboros.place_intake import offer_arrival

    world = host()
    seen, retained = {}, []
    real_retain = presence_continuation._retain_reentry

    def retain(binding, ctx, rows, gaps, queue=None):
        retained.extend(rows)
        return real_retain(binding, ctx, rows, gaps, queue)

    monkeypatch.setattr(presence_continuation, "_retain_reentry", retain)

    def read(messages):
        seen["words"] = json.dumps(messages, ensure_ascii=False)
        return finish("Noted.")

    author = _park_at_question(world, read)
    bot, _ = post(world, text="Chatter from a bot.", facts=("any_message", "from_bot"), actor={"platform_actor_id": "B1", "is_bot": True})
    reaction, _ = post(world, text="", facts=(), actor={"platform_actor_id": "U2", "kind": "user"}, message={
        "message_id": "9", "reaction": {"schema_version": 1, "change": "add", "old": [], "new": [{"type": "emoji", "value": "+1"}],
                                        "target": {"message_id": "1", "resolution": "resolved"}}})
    assert attention(world, bot)["decision"] == attention(world, reaction)["decision"] == "heard"
    upload = tmp_path / "numbers.txt"
    upload.write_text("exact bytes", encoding="utf-8")
    event = event_of(world, "with-a-file", text="See the numbers.", facts=("any_message",))
    admission = admit_presence_turn(drive_root=world.data, authenticated_transport_skill=world.skill, provider=world.provider,
                                    account_id=world.account, conversation_id="C1", thread_id="1.0")
    captured = capture_presence_event(admission, event, drive_root=world.data, staged_files=(upload,))
    assert offer_arrival(world.data, captured, event, wake=True) == ("current_author", author)
    until(lambda: seen.get("words"), "the woken author's next model call")
    assert seen["words"].count("See the numbers.") == 1 and "numbers.txt" in seen["words"], "once, in its mail, with its file"
    assert "Chatter from a bot." in seen["words"] and "+1" in seen["words"], "what no mail carries stays in the note"
    assert {"with-a-file", bot, reaction} <= {row.get("client_message_id") for row in retained}, "the retained source stays whole"
    until(lambda: (load_task_result(world.data, author) or {}).get("status") == "completed", "its terminal")


@pytest.mark.parametrize("fault", ["unreadable", "torn"])
def test_a_mailbox_read_that_proves_nothing_keeps_every_row_in_the_reentry_note(tmp_path, monkeypatch, fault):
    from ouroboros import owner_mailbox
    from ouroboros.presence_continuation import _arrivals_in_mail

    binding = SimpleNamespace(drive_root=tmp_path, task_id="author")
    ctx = SimpleNamespace(task_attempt=1, _loop_mailbox_seen_ids=set())
    assert owner_mailbox.write_task_message(tmp_path, "words", "author", source_task_id="s", provenance="place_event", msg_id="place-event:e",
                                            place_arrival={"source_event_id": "e", "message": {}, "wake": True})
    assert _arrivals_in_mail(binding, ctx) == {"place-event:e"}
    assert ctx._loop_mailbox_seen_ids == set(), "a private read: nothing seen"
    assert owner_mailbox.acknowledged_task_message_ids(tmp_path, "author", attempt_key=1) == set(), "and nothing acknowledged"
    if fault == "torn":
        with owner_mailbox._mailbox_path(tmp_path, "author").open("ab") as handle:
            handle.write(b'{"msg_id": "place-ev')
    else:
        monkeypatch.setattr(owner_mailbox, "drain_owner_entries", lambda *_a, **_k: (_ for _ in ()).throw(OSError("unreadable")))
    assert _arrivals_in_mail(binding, ctx) is None


def test_a_torn_mailbox_at_a_real_park_keeps_the_message_in_the_note_says_why_and_still_delivers_it(host):
    from ouroboros.owner_mailbox import _mailbox_path

    world = host()
    seen = {}

    def read(messages):
        seen["words"] = json.dumps(messages, ensure_ascii=False)
        return finish("Noted.")

    author = _park_at_question(world, read)
    path = _mailbox_path(world.data, author)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(b'{"msg_id": "torn-earlier", "te')
    source, reply = post(world, text="Words beside a torn row.")
    assert reply.status_code == 200 and attention(world, source)["recipient"] == author, reply.text
    until(lambda: seen.get("words"), "the woken author's next model call")
    assert seen["words"].count("Words beside a torn row.") == 2, "unproven: kept rather than lost, the duplicate explained"
    assert "your mailbox could not be read whole" in seen["words"]
    assert "they follow as place events" not in seen["words"] and PLACE_EVENT in seen["words"]
    until(lambda: (load_task_result(world.data, author) or {}).get("status") == "completed", "its terminal")
