"""A place handoff joins the author's serialized intake, including its final drain."""
from dataclasses import replace
from types import SimpleNamespace
import copy
import json
import queue
import threading
import time

import pytest

from ouroboros import conversation_lease as leases
from ouroboros.place_intake import offer_to_author, settle_heard, decide_attention
from ouroboros.presence_runner import capture_presence_event, presence_turn_replay
from tests.test_place_focus import focus as focus, task_for


def intake(author):
    return SimpleNamespace(_owner_message_admission_lock=threading.RLock(), _accepting_owner_messages=True,
                           _busy=True, _current_task_id=author)


@pytest.mark.parametrize("ending", ["released", "intake_closed"])
def test_handoff_cannot_settle_after_the_author_closes(focus, monkeypatch, ending):
    from ouroboros.owner_mailbox import _mailbox_path
    root, _, admission, event = focus
    task = capture_presence_event(admission, event, drive_root=root)
    lease = leases.PresenceTurnGate(state_root=root / "state").acquire(event.conversation_key)
    lease.bind_author("finishing-author")
    agent = lease.intake_agent = intake("finishing-author")
    read_author = leases.live_author
    def ends_after_census(*args, **kwargs):
        author = read_author(*args, **kwargs)
        if ending == "released":
            lease.release()
        else:
            with agent._owner_message_admission_lock:
                agent._accepting_owner_messages = False
        return author
    monkeypatch.setattr(leases, "live_author", ends_after_census)
    try:
        assert not offer_to_author(root, task, event)
        assert not _mailbox_path(root, "finishing-author").exists()
        assert presence_turn_replay(root, task["id"], event.conversation_key, task["metadata"]["presence_event_identity"]) is None
        assert decide_attention(root, admission, event)[0] == "focus"
    finally:
        lease.release()


def test_intake_close_and_mail_acceptance_share_one_lock(focus, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from ouroboros import owner_mailbox
    root, _, admission, event = focus
    task = capture_presence_event(admission, event, drive_root=root)
    lease = leases.PresenceTurnGate(state_root=root / "state").acquire(event.conversation_key)
    lease.bind_author("author")
    agent = lease.intake_agent = intake("author")
    append_entered, append_release, close_entered, closed = (threading.Event() for _ in range(4))
    append = owner_mailbox.write_task_message
    def paused_append(*args, **kwargs):
        append_entered.set()
        assert append_release.wait(10)
        assert not closed.is_set()
        return append(*args, **kwargs)
    def close():
        close_entered.set()
        with agent._owner_message_admission_lock:
            agent._accepting_owner_messages = False
            closed.set()
    monkeypatch.setattr(owner_mailbox, "write_task_message", paused_append)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            offered = pool.submit(offer_to_author, root, task, event)
            try:
                assert append_entered.wait(10)
                closing = pool.submit(close)
                assert close_entered.wait(10)
                assert not closed.is_set()
            finally:
                append_release.set()
            assert offered.result(timeout=10)
            closing.result(timeout=10)
        # Its retry (a lost settlement) finds the arrival the author holds; a new arrival is refused.
        assert offer_to_author(root, task, event)
        later = replace(event, source_event_id="after-close")
        assert not offer_to_author(root, capture_presence_event(admission, later, drive_root=root), later)
        assert len(owner_mailbox._mailbox_path(root, "author").read_text().splitlines()) == 1
    finally:
        append_release.set()
        lease.release()


@pytest.mark.parametrize("arrival", [False, True])
def test_final_drain_returns_accepted_place_context_to_the_author(focus, monkeypatch, arrival):
    from ouroboros import loop
    from ouroboros.tools.registry import ToolRegistry
    root, _, admission, event = focus
    task = task_for(focus)
    incoming = replace(event, source_event_id="event2", text="A newly accepted place event")
    captured = capture_presence_event(admission, incoming, drive_root=root)
    lease = leases.PresenceTurnGate(state_root=root / "state").acquire(event.conversation_key)
    lease.bind_author(task["id"])
    agent = lease.intake_agent = intake(task["id"])
    registry = ToolRegistry(repo_dir=root, drive_root=root)
    ctx = registry._ctx
    ctx.task_metadata, ctx.task_contract, ctx.is_direct_chat = task["metadata"], task["task_contract"], True
    ctx.owner_message_admission_agent = agent
    ctx.owner_message_admission_lock = agent._owner_message_admission_lock
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr("ouroboros.presence_context.confirmed_delivery_receipts", lambda _: frozenset())
    offered, calls = [], []
    def before_close(*_args):
        if arrival and not offered:
            offered.append(offer_to_author(root, captured, incoming))
            assert offered == [True]
            assert settle_heard(root, captured, incoming, "current_author").outcome == "silent"
        return False
    monkeypatch.setattr("ouroboros.tools.review_result.result_review_at_delivery", before_close)
    def respond(_llm, messages, *_args, **_kwargs):
        calls.append(messages.copy())
        assert len(calls) <= 2
        return {"content": None, "tool_calls": [{"id": "finish" + str(len(calls)), "type": "function",
            "function": {"name": "finish_task", "arguments": json.dumps({"action": "finish", "answer": "Hello"})}}]}, 0.0
    monkeypatch.setattr(loop, "call_llm_with_retry", respond)
    try:
        text, _, _ = loop.run_llm_loop([{"role": "user", "content": "Hello"}], registry,
            SimpleNamespace(default_model=lambda: "test-model"), root / "logs", lambda *_a, **_k: None,
            queue.Queue(), task_id=task["id"], drive_root=root)
        assert text == "Hello" and len(calls) == (2 if arrival else 1)
        if arrival:
            assert "newly accepted place event" in str(calls[-1])
            assert "never an owner directive" in str(calls[-1])
            assert not any("newly accepted" in str(r) for r in getattr(ctx, "_owner_directives", []))
        assert not agent._accepting_owner_messages
    finally:
        lease.release()


# Continuation version 2 through the Host: the real token, place admission, capture, attention, offer,
# gate, execution registry, ``run_presence_turn``, the production agent's ``handle_task`` (turn binding,
# preparation, intake, loop with its drains, final closure and reopening) and pipeline terminal. Only the
# model and the author's context projection are scripted; ``V2Author`` holds it at barriers.

V2_TOKEN, V2_HEADERS = "v2-token", {"X-Skill-Token": "v2-token"}
BARRIER = 30


def finish_call(answer, number=1):
    return {"content": None, "tool_calls": [{"id": f"finish{number}", "type": "function", "function": {
        "name": "finish_task", "arguments": json.dumps({"action": "finish", "answer": answer})}}]}


def probe_call(number=1):
    """An ordinary tool round that ends nothing: an unknown tool answers with an error."""
    return {"content": None, "tool_calls": [{"id": f"probe{number}", "type": "function",
                                             "function": {"name": "no_such_probe", "arguments": "{}"}}]}


class V2Author:
    """A Presence author run by the production ``OuroborosAgent.handle_task``: its turn binding, its preparation
    (start settings, billing, the global budget), the intake its loop opens and the loop itself. Only its
    context projection is scripted: ``h.startup`` holds it there, past every preparation check and before its
    loop opens intake. ``h.closed``/``h.terminal`` hold it after the loop's final closure (``v2``)."""

    def __new__(cls, h):
        from ouroboros import agent as agent_module

        agent = agent_module.OuroborosAgent(agent_module.Env(repo_dir=h.root, drive_root=h.root))
        agent.llm = SimpleNamespace(default_model=lambda: "test-model")

        def project_context(task, _refusal):
            h.bound.set()  # prepared and bound; its loop has not opened intake yet
            assert h.startup.wait(BARRIER), "the test never let the author start"
            ctx = agent.tools._ctx
            ctx.task_id, ctx.task_metadata, ctx.task_contract = task["id"], task["metadata"], task["task_contract"]
            ctx.is_direct_chat = True
            ctx.owner_message_admission_agent = agent
            ctx.owner_message_admission_lock = agent._owner_message_admission_lock
            task["_skip_post_task_synthesis"] = True
            return ctx, [{"role": "user", "content": task["text"]}], {}

        agent._prepare_task_context = project_context
        h.agents.append(agent)
        return agent


@pytest.fixture
def v2(tmp_path, monkeypatch):
    import os
    from starlette.testclient import TestClient
    from ouroboros import agent as agent_module, loop, places
    from ouroboros.gateway import host_service
    from ouroboros.presence_runner import run_presence_turn
    from ouroboros.settings_integrity import task_settings_snapshot
    from tests.test_host_service_api import _seed_token

    root = tmp_path / "data"
    root.mkdir()
    # The production agent runs. Installation boot, its heartbeat thread, the authority-source scan and the
    # settings hot reload (which would replace this environment) are setup, not what these tests decide.
    monkeypatch.setattr(agent_module.OuroborosAgent, "_log_worker_boot_once", lambda *_a: None)
    monkeypatch.setattr(agent_module.OuroborosAgent, "_start_task_heartbeat_loop", lambda *_a: None)
    monkeypatch.setattr(agent_module, "validate_task_authority_sources", lambda *_a: None)
    monkeypatch.setattr(agent_module.subagent_runtime, "apply_task_start_settings_or_disclose",
                        lambda *_a: task_settings_snapshot({}, os.environ))
    for name, value in {"OUROBOROS_PRESENCE_MAX_ACTIVE": "1", "OUROBOROS_TASK_REVIEW_MODE": "off",
                        "OUROBOROS_SAFETY_MODE": "off", "MCP_ENABLED": "false"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: True)
    monkeypatch.setattr("ouroboros.presence_admission.focus_allowance",
                        lambda _root: {"status": "available", "remaining_usd": 10})
    monkeypatch.setattr("ouroboros.presence_context._own_work_section", lambda *_args: "")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr("ouroboros.presence_context.confirmed_delivery_receipts", lambda _: frozenset())
    monkeypatch.setattr("ouroboros.tools.review_result.result_review_at_delivery", lambda *_a: False)
    monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", lambda *_a: False)
    _seed_token(root, skill="bridge", token=V2_TOKEN, permissions=["presence"], manifest_permissions=["presence"])
    place = places.connect_place(root, connector="bridge", provider="chat", account_id="a", conversation_id="r",
                                 wakes=["mention"])
    other = places.connect_place(root, connector="bridge", provider="chat", account_id="a", conversation_id="r2",
                                 wakes=["mention"])
    h = SimpleNamespace(root=root, place=place, other=other, agents=[], script=[], calls=[], closed=threading.Event(),
                        terminal=threading.Event(), startup=threading.Event(), bound=threading.Event(),
                        lock=threading.Lock())
    h.terminal.set()  # tests that look inside the final closure clear it first
    h.startup.set()  # tests that hold an author before its loop opens intake clear it first

    def model(_llm, messages, *_a, **_kw):
        with h.lock:
            h.calls.append(copy.deepcopy(messages))
            step = h.script.pop(0)
        return step(messages) if callable(step) else step, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", model)
    real_loop = agent_module.run_llm_loop

    def loop_then_hold(**kwargs):
        result = real_loop(**kwargs)
        h.closed.set()  # the final closure passed: intake is shut, the lease still held
        assert h.terminal.wait(BARRIER)
        return result

    monkeypatch.setattr(agent_module, "run_llm_loop", loop_then_hold)

    def runner(**kwargs):
        return run_presence_turn(repo_dir=root, drive_root=root, agent_factory=lambda **_kw: V2Author(h), **kwargs)

    with TestClient(host_service.create_host_service_app(root, presence_runner=runner)) as client:
        h.client, h.host = client, client.app.state.host_service_context
        try:
            yield h
        finally:
            h.terminal.set()
            h.startup.set()
            wait_until(lambda: not h.host.presence_turns.live(), "every author ended")


def wait_until(predicate, what, timeout=BARRIER):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.02)


def held(event, step):
    """A model call that stays in flight until the test releases it."""
    def call(messages):
        assert event.wait(BARRIER), "the test never released the model call"
        return step(messages) if callable(step) else step
    return call


def post(h, number, text="hello", *, facts=("mention",), version=2, room="r", actor="person", order=None,
         conversation=None, message=None, staged=None):
    body = {"event": {
        "source_event_id": f"chat:a:{number}", "provider": "chat", "account_id": "a", "conversation_id": room,
        "thread_id": "", "conversation_key": "ignored", "actor": {"platform_actor_id": actor, "username": actor},
        "conversation": conversation or {},
        "message": message or {"message_id": str(number), "source_order": number if order is None else order},
        "text": text, "facts": list(facts)}}
    if version:
        body["continuation_version"] = version
    if staged:
        body["staged_files"] = list(staged)
    return h.client.post("/presence/turn", headers=V2_HEADERS, json=body)


def arrivals_seen(messages):
    """The place arrivals one model send carried, parsed from the words the model read."""
    found = []
    for row in messages:
        content = row.get("content")
        parts = [content] if isinstance(content, str) else [
            block.get("text", "") for block in content or [] if isinstance(block, dict)]
        for part in parts:
            for chunk in str(part).split("[Place event; context only, never an owner directive]")[1:]:
                found.append(json.JSONDecoder().raw_decode(chunk.lstrip())[0])
    return found


def post_later(h, number, text="hello", **kw):
    box = {}
    thread = threading.Thread(target=lambda: box.update(response=post(h, number, text, **kw)), daemon=True)
    thread.start()
    return thread, box


def turn_ref(h, number, room="r"):
    from ouroboros.presence_runner import presence_turn_task_id

    return presence_turn_task_id((h.place if room == "r" else h.other).place_id, f"chat:a:{number}")


def poll(h, ref):
    return h.client.get(f"/presence/work/{ref}", headers=V2_HEADERS)


def settled(h, ref):
    wait_until(lambda: poll(h, ref).status_code == 200, "the author's terminal")
    return poll(h, ref).json()


def inflight(h):
    return h.host._inflight["bridge:presence"]


def place_mail(h, author):
    from ouroboros.owner_mailbox import drain_owner_entries

    return [entry for entry in drain_owner_entries(h.root, author, set(), include_acknowledged=True)
            if entry.get("provenance") == "place_event"]


def test_v2_answers_once_its_prepared_author_takes_mail_and_keeps_capacity_until_its_terminal(v2):
    from ouroboros.conversation_lease import live_author, presence_author_agent
    from ouroboros.task_results import load_task_result

    h, go, at_send = v2, threading.Event(), []
    ref = turn_ref(h, 1)

    def first_send(_messages):
        at_send.append(load_task_result(h.root, ref).get("presence_continuation"))  # what this send follows
        assert go.wait(BARRIER), "the test never released the first model call"
        return finish_call("Done.")

    h.script = [first_send, finish_call("Done, with the totals.", 2)]
    h.startup.clear()  # hold the author past its preparation, before its loop opens intake
    thread, box = post_later(h, 1)
    wait_until(h.bound.is_set, "the author's preparation")
    thread.join(0.5)
    assert "response" not in box, "the author answered before its loop could read what it takes"
    h.startup.set()
    thread.join(BARRIER)
    first = box["response"]
    assert first.status_code == 200, first.text
    assert first.json() == {"ok": True, "status": "continuing", "outcome": "deferred", "text": "", "turn_ref": ref,
                            "work_ref": "", "delivery_reporting_version": 0, "continuation_version": 2,
                            "continuation_ref": ref, "output_ref": ""}
    # Answered after the real RUNNING row, the real preparation and the loop's open intake, before the model's
    # first send (whose answer is still held); nothing lent or returned.
    wait_until(lambda: at_send, "the author's first model send")
    assert at_send[0]["initial"]["status"] == "continuing"
    row = load_task_result(h.root, ref)
    assert row["status"] == "running"
    assert presence_author_agent(h.root, ref)._accepting_owner_messages
    record = row["presence_continuation"]
    assert record["continuation_version"] == 2 and record["initial"]["status"] == "continuing"
    assert "lent_at" not in record and "first_lent_at" not in record
    assert live_author(row["metadata"]["presence"]["event"]["conversation_key"], drive_root=h.root) == ref
    assert inflight(h) == 1 and h.host.presence_turns.live() == [ref]
    assert post(h, 1).json() == first.json()  # a retry: the identical stored envelope, nothing rerun
    pending = poll(h, ref)
    assert pending.status_code == 202 and pending.json()["continuation_version"] == 2
    # The transport FIFO's next event of this conversation meets this same author while its first send is held.
    second = post(h, 2, "and the totals, please").json()
    assert (second["status"], second["outcome"]) == ("completed", "silent")
    assert [entry["place_arrival"]["source_event_id"] for entry in place_mail(h, ref)] == ["chat:a:2"]
    assert h.host.presence_turns.live() == [ref] and inflight(h) == 1 and len(h.calls) == 1
    go.set()
    late = settled(h, ref)
    # The first answer was selected before the model saw the arrival: that unseen wake reopens it once.
    assert [row["source_event_id"] for row in arrivals_seen(h.calls[1])] == ["chat:a:2"]
    assert (late["status"], late["outcome"], late["text"]) == ("completed", "message", "Done, with the totals.")
    assert late["continuation_version"] == 2 and late["output_ref"].startswith("presence-output-")
    assert late["reply_anchor"] == {"source_event_id": "chat:a:2", "message": {"message_id": "2", "source_order": 2}}
    wait_until(lambda: inflight(h) == 0, "the reservation's single return at the terminal")
    assert len(h.calls) == 2 and len(h.agents) == 1


def test_a_v2_author_answers_only_after_its_real_preparation_passed(v2, monkeypatch):
    """The production ``handle_task`` binds its turn for Stop before preparing, but it answers early only once
    its loop is prepared. A global budget refusal in that preparation ends the author before any loop: its
    event gets that refusal, never ``continuing``, so a sequential transport hands no follower to a consumer
    that will never read it, and its next event meets ordinary admission. Nothing reaches a model."""
    import supervisor.state

    h, entered, refuse = v2, threading.Event(), threading.Event()

    def exhausted(_state, **_kwargs):  # the production check's own refusal: no room left in the global budget
        entered.set()
        assert refuse.wait(BARRIER), "the test never let the budget read answer"
        return 0.0

    monkeypatch.setattr(supervisor.state, "budget_remaining", exhausted)
    thread, box = post_later(h, 1)
    ref = turn_ref(h, 1)
    assert entered.wait(BARRIER), "the author never reached its budget check"
    agent, = h.agents
    assert agent._busy and agent._current_task_id == ref  # Stop, Pause and steering address it already
    thread.join(0.5)
    assert "response" not in box, f"the author answered before its preparation passed: {box['response'].text}"
    refuse.set()
    thread.join(BARRIER)
    first = box["response"]
    assert first.status_code == 500 and first.json()["disposition"] == "retry", first.text
    assert not h.bound.is_set() and place_mail(h, ref) == []
    # The transport's next event: no ended author takes it as context, and it is not acknowledged as heard.
    follower = post(h, 2, "and the totals, please")
    assert follower.status_code == 500 and follower.json()["disposition"] == "retry", follower.text
    assert place_mail(h, ref) == [] and place_mail(h, turn_ref(h, 2)) == []
    assert h.calls == [] and not h.host.presence_turns.live()


def test_a_flood_joins_one_author_and_the_reply_anchors_its_last_covered_message(v2):
    from ouroboros.task_results import load_task_result

    h, go = v2, threading.Event()
    seen = {}

    def answer(messages):
        seen["second"] = str(messages)
        return finish_call("Corrected answer.", 2)

    h.script = [held(go, finish_call("First answer.")), answer]
    assert post(h, 1, "please summarize").json()["status"] == "continuing"
    ref = turn_ref(h, 1)
    wait_until(lambda: len(h.calls) == 1, "the author's first model send")
    # Background chatter from many people, a direct correction, then more chatter: each is
    # acknowledged at once as context of the live author, without a turn, slot or model call.
    texts = {}
    for number in range(2, 42):
        texts[number] = f"chatter {number}"
        body = post(h, number, texts[number], facts=(), actor=f"person-{number}").json()
        assert (body["status"], body["outcome"]) == ("completed", "silent"), body
    texts[42] = "correction: summarize only the last week"
    assert post(h, 42, texts[42]).json()["outcome"] == "silent"
    texts[43] = "late chatter"
    assert post(h, 43, texts[43], facts=(), actor="person-43").json()["outcome"] == "silent"
    mail = place_mail(h, ref)
    assert len(mail) == 42 and [entry["place_arrival"]["wake"] for entry in mail].count(True) == 1
    assert not any("context_source_ref" in entry["text"] for entry in mail)  # nothing left out, nothing retained
    assert inflight(h) == 1 and h.host.presence_turns.live() == [ref] and len(h.calls) == 1
    # The first answer was selected before the model saw any arrival; the correction is a wake it has
    # not seen, so the ready final reopens for one more round that sees everything at once.
    go.set()
    late = settled(h, ref)
    assert late["text"] == "Corrected answer." and len(h.calls) == 2 and len(h.agents) == 1
    assert all(text in seen["second"] for text in texts.values())
    assert late["reply_anchor"]["source_event_id"] == "chat:a:43"  # the latest covered, by provider order
    assert late["reply_anchor"]["message"] == {"message_id": "43", "source_order": 43}
    # After the author's end, chatter is heard only: no turn, no model call.
    assert post(h, 44, "after", facts=(), actor="person-44").json()["outcome"] == "silent"
    assert load_task_result(h.root, turn_ref(h, 44)) is None and len(h.calls) == 2


@pytest.mark.parametrize("round_kind", ["ordinary", "after_final"])
def test_background_alone_never_reopens_and_is_covered_only_once_the_model_saw_it(v2, round_kind):
    h, go = v2, threading.Event()
    ordinary = round_kind == "ordinary"
    first = probe_call() if ordinary else finish_call("Answer.")
    h.script = [held(go, first)] + ([finish_call("Answer.", 2)] if ordinary else [])
    assert post(h, 1).json()["status"] == "continuing"
    ref = turn_ref(h, 1)
    wait_until(lambda: len(h.calls) == 1, "the author's first model send")  # the arrival comes after it
    assert post(h, 2, "by the way", facts=(), actor="other").json()["outcome"] == "silent"
    go.set()
    late = settled(h, ref)
    # Delivered before an ordinary call, the background message is covered; drained after the
    # final answer it reopens nothing and the anchor stays on the author's own event.
    assert late["text"] == "Answer." and len(h.calls) == (2 if ordinary else 1)
    assert late["reply_anchor"]["source_event_id"] == ("chat:a:2" if ordinary else "chat:a:1")


@pytest.mark.parametrize("version", [1, 2])
def test_a_wake_drained_after_the_finishing_call_reopens_the_ready_final(v2, version):
    h, go = v2, threading.Event()
    seen = {}

    def answer(messages):
        seen["second"] = str(messages)
        return finish_call("Revised.", 2)

    h.script = [held(go, finish_call("Draft.")), answer]
    thread, box = post_later(h, 1, version=version)
    ref = turn_ref(h, 1)
    wait_until(lambda: len(h.calls) == 1, "the author's first model call")
    assert post(h, 2, "please add the totals", version=version).json()["outcome"] == "silent"
    go.set()
    thread.join(BARRIER)
    assert settled(h, ref)["text"] == "Revised."  # version 2 answered early: its request is no clock here
    assert len(h.calls) == 2 and "please add the totals" in seen["second"]
    assert box["response"].status_code == 200


@pytest.mark.parametrize("version", [1, 2])
def test_a_retry_after_a_lost_settlement_stays_with_the_author_that_holds_it(v2, monkeypatch, version):
    from ouroboros import place_intake

    h, go = v2, threading.Event()
    h.script = [held(go, finish_call("Answer.")), finish_call("Answer with the arrival.", 2)]
    h.terminal.clear()
    thread, _box = post_later(h, 1, version=version)
    ref = turn_ref(h, 1)
    wait_until(lambda: len(h.calls) == 1, "the author's first model call")
    real = place_intake.settle_heard
    lost = []

    def lose_once(root, task, event, reason="heard", **kw):
        if event.source_event_id == "chat:a:2" and not lost:
            lost.append(reason)
            raise OSError("connection lost after the author took the arrival")
        return real(root, task, event, reason, **kw)

    monkeypatch.setattr(place_intake, "settle_heard", lose_once)
    failed = post(h, 2, "one more thing", version=version)
    assert failed.status_code >= 400 and lost == ["current_author"] and len(place_mail(h, ref)) == 1
    go.set()
    assert h.closed.wait(BARRIER)  # the author's final closure shut its intake; it still holds the conversation
    retry = post_later(h, 2, "one more thing", version=version)
    retry[0].join(5)
    assert not retry[0].is_alive(), "the retry waited for the author instead of settling"
    assert retry[1]["response"].json()["outcome"] == "silent"
    assert len(place_mail(h, ref)) == 1 and len(h.agents) == 1  # one arrival, held by one author
    h.terminal.set()
    thread.join(BARRIER)
    assert settled(h, ref)["text"] == "Answer with the arrival." and len(h.agents) == 1


def test_v2_never_waits_in_the_gate_and_leaves_nothing_behind(v2, monkeypatch):
    from ouroboros.conversation_lease import PresenceTurnGate
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.task_results import load_task_result

    h = v2
    gate = PresenceTurnGate(1, state_root=h.root / "state")  # another process's view of the same files
    key, other = conversation_key("chat", "a", "r", ""), conversation_key("chat", "a", "r2", "")

    def leaves_nothing(number):
        assert h.host.presence_turns.live() == [] and inflight(h) == 0
        assert load_task_result(h.root, turn_ref(h, number)) is None

    # A conversation held by an author that takes no mail here (preparing, closing or remote).
    holder = gate.acquire(key)
    holder.bind_author("remote-author")
    try:
        busy = post(h, 1)
        assert busy.status_code == 429 and busy.json() == {
            "ok": False, "error": "presence_capacity_full: conversation", "code": "presence_capacity_full",
            "disposition": "retry", "field": "conversation", "retry_after": 1, "attachment_manifest": []}
        leaves_nothing(1)
    finally:
        holder.release()
    # Every active slot is held by another conversation's author.
    holder = gate.acquire(other)
    try:
        busy = post(h, 2)
        assert busy.status_code == 429 and busy.json()["field"] == "capacity" and busy.json()["retry_after"] == 5
        leaves_nothing(2)
        lease, refused = gate.try_admit(key)  # the refused pass released the conversation it had taken
        assert lease is None and refused == "capacity"
        h.script = [finish_call("Legacy done.")]
        legacy, box = post_later(h, 3, version=1)  # versions 0 and 1 still wait in the gate
        legacy.join(0.5)
        assert legacy.is_alive() and "response" not in box
    finally:
        holder.release()
    legacy.join(BARRIER)
    assert box["response"].json()["status"] == "completed"
    h.script = [finish_call("Second.")]
    assert post(h, 2).json()["status"] == "continuing"  # the same event, retried once a slot is free
    assert settled(h, turn_ref(h, 2))["text"] == "Second."


def test_distinct_conversations_run_their_own_authors_side_by_side(v2, monkeypatch):
    monkeypatch.setenv("OUROBOROS_PRESENCE_MAX_ACTIVE", "2")
    h, go = v2, threading.Event()
    h.script = [held(go, finish_call("Room one.")), held(go, finish_call("Room two."))]
    assert post(h, 1, room="r").json()["status"] == "continuing"
    assert post(h, 2, room="r2").json()["status"] == "continuing"
    assert sorted(h.host.presence_turns.live()) == sorted([turn_ref(h, 1), turn_ref(h, 2, room="r2")])
    assert inflight(h) == 2
    go.set()
    assert {settled(h, turn_ref(h, 1))["text"], settled(h, turn_ref(h, 2, room="r2"))["text"]} == {"Room one.", "Room two."}


@pytest.mark.parametrize("version", [1, 2])
def test_an_arrival_reaches_the_model_with_its_message_conversation_and_attachments(v2, version):
    from ouroboros import dialogue_store
    from ouroboros.presence_bindings import conversation_key

    h, go = v2, threading.Event()
    h.script = [held(go, probe_call()), finish_call("Answer.", 2)]
    thread, _box = post_later(h, 1, version=version)
    ref = turn_ref(h, 1)
    wait_until(lambda: len(h.calls) == 1, "the author's first model call")
    staged = h.root / "state" / "skills" / "bridge" / "minutes.txt"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_text("Minutes of the planning call", encoding="utf-8")
    inbox = {"schema_version": 1, "source": "bridge inbox", "observed_at": "2026-10-29T00:00:00+00:00",
             "conversation_key": conversation_key("chat", "a", "r", ""), "after_source_event_id": "chat:a:2",
             "complete": True, "pending_count": 1, "omitted_count": 0,
             "events": [{"source_event_id": "chat:a:3", "text": "a queued follow-up"}]}
    message = {"message_id": "2", "source_order": 2, "reply_to_message_id": "1", "url": "https://chat.example/r/2"}
    conversation = {"title": "Planning room", "type": "group", "transport_queue": inbox}
    joined = post(h, 2, "see the minutes", version=version, message=message, conversation=conversation,
                  staged=[str(staged)])
    assert joined.json()["outcome"] == "silent", joined.text
    go.set()
    thread.join(BARRIER)
    assert settled(h, ref)["text"] == "Answer."
    # The model reads the arrival's structure, not only its words. The queue keeps every fact but its events,
    # completeness included, and names the source that holds them.
    arrival, = [row for row in arrivals_seen(h.calls[1]) if row["source_event_id"] == "chat:a:2"]
    assert arrival["event"] == "see the minutes" and arrival["message"] == message
    assert arrival["facts"] == ["mention"] and "minutes.txt" in json.dumps(arrival["attachment_manifest"])
    assert {key: arrival["conversation"][key] for key in ("title", "type")} == {"title": "Planning room", "type": "group"}
    queue = arrival["conversation"]["transport_queue"]
    assert {key: value for key, value in queue.items() if key != "events"} == {
        key: value for key, value in inbox.items() if key != "events"}
    source = arrival["context_source_ref"]  # a locator of the arrival's own retained source, as row readers print it
    assert "memory_read(source_ref=" in queue["events"] and source["task_id"] == turn_ref(h, 2)
    assert source["kind"] == "task_source" and not {"read", "canonical_root"} & set(source)
    room_row, = [entry.row for entry in dialogue_store.iter_rows(h.root, "chat")
                 if entry.row.get("client_message_id") == "chat:a:2" and entry.row.get("direction") == "in"]
    assert room_row["transport"]["conversation"]["transport_queue"] == inbox  # the exact snapshot, kept
    assert room_row["transport"]["message"] == message
    # The author's own registered readers resolve that reference: one retained source, and the room row's
    # reference in its chat history and memory rows. No history scan, no other store.
    readers = author_readers(h, ref)
    assert {"memory_read", "chat_history"} <= {schema["function"]["name"] for schema in readers.schemas()}
    exact = readers.execute("memory_read", {"source_ref": source})
    observed = json.JSONDecoder().raw_decode(exact[exact.index("{"):])[0]
    assert observed["transport"]["conversation"]["transport_queue"] == inbox
    assert observed["transport"]["message"] == message and observed["source_event_id"] == "chat:a:2"
    history = readers.execute("chat_history", {"provider": "chat", "account_id": "a", "conversation_id": "r"})
    rows = readers.execute("memory_read", {"rows": True, "room_id": str(room_row["chat_id"])})
    assert source in context_refs(history) and source in context_refs(rows)


def test_an_arrival_whose_queue_source_cannot_be_retained_is_not_handed_over(v2, monkeypatch):
    """No omission without a source this author can resolve: when the arrival's exact facts cannot be retained,
    the handoff is refused before any mail, and its transport retries the same event."""
    from ouroboros import chat_chain
    from ouroboros.presence_bindings import conversation_key

    h, go = v2, threading.Event()
    h.script = [held(go, probe_call()), finish_call("Answer.", 2)]
    assert post(h, 1).json()["status"] == "continuing"
    ref = turn_ref(h, 1)
    wait_until(lambda: len(h.calls) == 1, "the author's first model send")
    inbox = {"schema_version": 1, "source": "bridge inbox", "observed_at": "2026-10-29T00:00:00+00:00",
             "conversation_key": conversation_key("chat", "a", "r", ""), "after_source_event_id": "chat:a:2",
             "complete": True, "pending_count": 1, "omitted_count": 0,
             "events": [{"source_event_id": "chat:a:3", "text": "a queued follow-up"}]}
    real = chat_chain.retain_memory_source

    def unavailable(*_args, **_kwargs):
        raise OSError("source storage unavailable")

    monkeypatch.setattr(chat_chain, "retain_memory_source", unavailable)
    refused = post(h, 2, "see the queue", conversation={"transport_queue": inbox})
    assert refused.status_code == 400 and refused.json()["disposition"] == "retry", refused.text
    assert place_mail(h, ref) == []
    monkeypatch.setattr(chat_chain, "retain_memory_source", real)
    assert post(h, 2, "see the queue", conversation={"transport_queue": inbox}).json()["outcome"] == "silent"
    entry, = place_mail(h, ref)
    assert json.loads(entry["text"])["context_source_ref"]["task_id"] == turn_ref(h, 2)
    go.set()
    assert settled(h, ref)["reply_anchor"]["source_event_id"] == "chat:a:2"


def author_readers(h, ref):
    """A registry bound as this author's context, the way a later actor reads its sources."""
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    stored = load_task_result(h.root, ref)
    registry = ToolRegistry(repo_dir=h.root, drive_root=h.root)
    registry.set_context(ToolContext(repo_dir=h.root, drive_root=h.root, task_id=ref, current_chat_id=stored["chat_id"],
                                     task_contract=stored.get("task_contract") or {}, task_metadata=stored["metadata"]))
    return registry


def context_refs(text):
    """Every ``memory_read(source_ref=...)`` locator a reader rendered."""
    marker = "memory_read(source_ref="
    return [json.JSONDecoder().raw_decode(part)[0] for part in text.split(marker)[1:]]


def test_a_send_refused_before_dispatch_covers_no_arrival(monkeypatch):
    from ouroboros import loop, loop_model_call
    from ouroboros.loop_llm_call import TIMEOUT_REPEAT_KEY
    from ouroboros.presence_continuation import reply_anchor, uncovered_wake

    own = {"source_event_id": "chat:a:1", "message": {"message_id": "1", "source_order": 1}}
    arrival = {"source_event_id": "chat:a:2", "message": {"message_id": "2", "source_order": 2}, "wake": True}
    author = SimpleNamespace(_presence_delivered=[arrival], _presence_covered=0)
    metadata = {"presence_continuation_version": 2, "presence": {"event": own}}
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_k: pytest.fail("a refused send dispatched"))
    # A restored timeout repeat refuses this round's send before dispatch (``timeout_repeat_blocked``).
    refused = SimpleNamespace(tools=SimpleNamespace(_ctx=author), active_context_mode="full",
                              accumulated_usage={TIMEOUT_REPEAT_KEY: {"state": "spent", "repeat": {}}})
    assert loop_model_call._call_round_model(refused)[0] is None
    assert author._presence_covered == 0 and uncovered_wake(author)  # no model read it: a wake still reopens
    assert reply_anchor(author, metadata) == own
    answered = SimpleNamespace(**{**vars(refused), "accumulated_usage": {}})
    monkeypatch.setattr(loop_model_call, "_call_round_model_once", lambda _send: ({"content": "read"}, 0.0, "full"))
    assert loop_model_call._call_round_model(answered)[0] == {"content": "read"}
    assert author._presence_covered == 1 and not uncovered_wake(author)
    assert reply_anchor(author, metadata) == {key: arrival[key] for key in ("source_event_id", "message")}


def test_a_resumed_author_covers_only_the_arrivals_its_reentry_note_carries_whole(tmp_path, monkeypatch):
    """The reentry note is model input like a drain: each incoming room row it carries whole joins coverage with
    its exact message, as context and never a wake. A row it names only by address, imported earlier history,
    replies and delivery facts move nothing; a model send answered after the note covers what it carried."""
    from ouroboros import presence_continuation as pc, presence_observations
    from ouroboros.presence_continuation import covered_send, reentry_note, reply_anchor, uncovered_wake

    key = "chat:a:r:0"

    def row(number, direction="in", text="", **extra):
        return {"ts": f"2026-10-29T00:00:0{number}+00:00", "chat_id": 7, "direction": direction, "task_id": f"turn-{number}",
                "text": text or f"words {number}", "client_message_id": f"chat:a:{number}",
                "transport": {"conversation_key": key, "message": {"message_id": str(number), "source_order": number}},
                **extra}

    rows = [row(3, text="an older correction " + "x" * 400),  # named by address only: it no longer fits
            row(1, historical_import={"authorship": "person"}), row(4, direction="out"),
            row(5, type="presence_delivery"), row(6), row(8)]
    monkeypatch.setattr(pc, "_conversation_rows_since", lambda *_a: (rows, []))
    monkeypatch.setattr(pc, "_retain_reentry", lambda *_a: "get_task_result(...)")  # a retained source bounds the note
    monkeypatch.setattr(pc, "_REENTRY_NOTE_CHARS", 700)
    monkeypatch.setattr(presence_observations, "transport_queue_observation",
                        lambda *_a: {"status": "unavailable", "reason": "transport_has_not_reported_queue"})
    event = SimpleNamespace(conversation_key=key, provider="chat", account_id="a", conversation_id="r", thread_id="")
    binding = pc.ReviewWaitBinding(None, tmp_path, "author", "identity", event, cursor={"at": "2026-10-29T00:00:00+00:00"})
    author = SimpleNamespace(task_metadata={})
    note = reentry_note(binding, author)
    assert "words 6" in note and "words 8" in note and "an older correction" not in note
    assert author._presence_delivered == [
        {"source_event_id": f"chat:a:{number}", "message": {"message_id": str(number), "source_order": number},
         "wake": False} for number in (6, 8)]
    assert not uncovered_wake(author)  # context, never a reason to reopen a ready final
    metadata = {"presence_continuation_version": 2,
                "presence": {"event": {"source_event_id": "chat:a:2", "message": {"message_id": "2", "source_order": 2}}}}
    assert reply_anchor(author, metadata)["source_event_id"] == "chat:a:2"  # carried, not yet answered
    covered_send(author, lambda: ({"content": "read"}, 0.0, "full"))
    assert reply_anchor(author, metadata) == {"source_event_id": "chat:a:8",
                                              "message": {"message_id": "8", "source_order": 8}}
