"""Real connector reactions through the real Host and a production author: context, never a wake, anchor or final.

``fixtures/reaction_connector_wire.json`` holds the exact bodies the frozen Slack and Telegram connectors posted to
``/presence/turn`` and ``/presence/delivery`` (its ``provenance`` names them). The real token, routes, place admission,
capture, attention, offer, gate, agent bootstrap, loop, waits, final selection, pipeline and readers run; only the model
is scripted (``install_bootstrap_harness``). A receipt's ``origin`` gains the live task its worker would have stamped.
"""
from __future__ import annotations

import copy
import datetime
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from ouroboros import dialogue_store, places
from ouroboros.gateway.host_service import create_host_service_app
from ouroboros.presence_runner import _read_previous_turn, run_presence_turn
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.tools.registry import ToolEntry

pytestmark = pytest.mark.usefixtures("provider_catalog_offline")
WIRE = json.loads((Path(__file__).parent / "fixtures" / "reaction_connector_wire.json").read_text(encoding="utf-8"))
TOKEN = "reaction-connector-token"
HEADERS = {"X-Skill-Token": TOKEN}
FINAL = "All three services are healthy."
CHAINS = {
    "slack": SimpleNamespace(skill="slack-bridge", account="T1", first="Ev-message", later="Ev-later", key="slack:T1:C1:1.0",
                             reactions=("Ev-add", "Ev-remove", "Ev-reply", "Ev-unknown"), heard="Ev-add",
                             receipts={"applied": "g-applied", "no-op": "g-noop", "failed": "g-failed"}),
    "telegram": SimpleNamespace(skill="telegram-bot", account="9", first="telegram:9:1", later="telegram:9:7", key="telegram:9:-100:5",
                                reactions=tuple(f"telegram:9:{n}" for n in range(2, 7)), heard="telegram:9:2",
                                receipts={"applied": "telegram-operation:g-applied", "failed": "telegram-operation:g-failed",
                                          "uncertain": "telegram-operation:g-uncertain"}),
}


def call(*calls):
    return {"content": None, "tool_calls": [{"id": name, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments)}} for name, arguments in calls]}


def finish(answer=FINAL, **arguments):
    return call(("finish_task", {"action": "finish", "answer": answer, **arguments}))


def until(predicate, what, seconds=60):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if value := predicate():
            return value
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.fixture
def chain(tmp_path, monkeypatch):
    from tests.test_host_service_api import _seed_token
    from tests.test_presence_continuation_bootstrap import install_bootstrap_harness

    monkeypatch.setattr("ouroboros.pricing._fetch_live_rows", lambda *_a, **_kw: {})  # a settling reviewer's price stays off the wire
    clients = []

    def start(provider, *, review_mode="off", tools=None):
        spec, data = CHAINS[provider], tmp_path / "data"
        h = install_bootstrap_harness(monkeypatch, data, review_mode=review_mode)
        h.addressed = ""
        _seed_token(data, skill=spec.skill, token=TOKEN, permissions=["presence"], manifest_permissions=["presence"])
        place = places.connect_place(data, connector=spec.skill, provider=provider, account_id=spec.account,
                                     conversation_id="", wakes=["any_message"])

        def factory(**kwargs):
            agent = h.factory(**kwargs)
            for name, handler in (tools or {}).items():
                agent.tools.register(ToolEntry(name=name, handler=handler, timeout_sec=30, safety_policy="skip", schema={
                    "name": name, "description": name, "parameters": {"type": "object", "properties": {}}}))
            return agent

        client = TestClient(create_host_service_app(data, presence_runner=lambda **kw: run_presence_turn(
            repo_dir=h.repo, drive_root=data, agent_factory=factory, **kw)))
        client.__enter__()
        clients.append(client)
        return SimpleNamespace(h=h, data=data, place=place, client=client, spec=spec, wire=WIRE[provider],
                               author=lambda: h.agents[-1].tools._ctx)

    yield start
    for client in clients:
        client.__exit__(None, None, None)


def post(world, source):
    reply = world.client.post("/presence/turn", headers=HEADERS, json=world.wire["events"][source])
    assert reply.status_code == 200, reply.text
    return reply.json()


def gesture_tool(holder, name="applied"):
    """The connector worker's report of one gesture, attributed to the turn whose tool queued it."""
    def gesture(ctx, **_arguments):
        world = holder["world"]
        body = copy.deepcopy(world.wire["receipts"][world.spec.receipts[name]])
        body["origin"] = {"kind": "tool", "task_id": ctx.task_id, "source_event_id": world.spec.first}
        reply = world.client.post("/presence/delivery", headers=HEADERS, json=body)
        assert reply.status_code == 200 and reply.json()["recorded"] is True, reply.text
        return "{}"
    return gesture


def attention(world):
    with dialogue_store.transaction(world.data, write=False) as conn:
        rows = conn.execute("SELECT client_message_id,dispatch_input FROM dialogue WHERE stream='chat' AND direction='in'").fetchall()
    return {row["client_message_id"]: (json.loads(row["dispatch_input"] or "{}").get("place_attention") or {}).get("reason")
            for row in rows}


def settled(world):
    ref = until(lambda: world.h.agents and world.author().task_id, "the author")
    until(lambda: (load_task_result(world.data, ref) or {}).get("status") == "completed", "the terminal")
    return world.client.get(f"/presence/work/{ref}", headers=HEADERS).json()


def seen_reaction(messages):
    return any("Reaction, not speech" in str(row.get("content")) for row in messages)


def tool_text(messages, identifier):
    return next(str(row["content"]) for row in reversed(messages) if row.get("role") == "tool" and row.get("tool_call_id") == identifier)


@pytest.mark.parametrize("provider", CHAINS)
def test_real_reactions_alone_start_nothing(chain, provider):
    world = chain(provider)
    for source in world.spec.reactions:
        assert post(world, source)["outcome"] == "silent"
    assert world.h.calls == 0 and not world.h.agents
    assert {source: attention(world)[source] for source in world.spec.reactions} == dict.fromkeys(world.spec.reactions, "reaction_context_only")
    assert all(entry.row["text"] == "" for entry in dialogue_store.iter_rows(world.data, "chat"))


# --- Each wait a live author can be in: a real reaction ends none, the next round reads it ----------------


@pytest.mark.parametrize("provider", CHAINS)
@pytest.mark.parametrize("wait,arguments,expected", [
    ("await_messages", {"timeout_sec": 4}, '"reason": "timeout"'),
    ("wait_tasks", {"task_ids": ["child"], "timeout_sec": 4}, '"timed_out": true'),
    ("wait_task", {"task_id": "child", "timeout_sec": 4}, "child"),
])
def test_a_tool_wait_runs_to_its_bound_over_a_real_reaction(chain, provider, wait, arguments, expected):
    world = chain(provider)
    seen = {}

    def start(_messages):
        write_task_result(world.data, "child", "running", root_task_id=world.author().task_id, parent_task_id=world.author().task_id)
        return call((wait, arguments))

    def after(messages):
        seen.update(result=tool_text(messages, wait), reaction=seen_reaction(messages))
        return finish()

    world.h.script = [start, after]
    post(world, world.spec.first)
    until(lambda: world.h.calls == 1, "the wait")
    time.sleep(0.2)
    assert post(world, world.spec.heard)["outcome"] == "silent"
    assert attention(world)[world.spec.heard] == "current_author" and world.h.calls == 1
    assert settled(world)["text"] == FINAL
    assert expected in seen["result"] and "owner_mailbox_pending" not in seen["result"]
    assert seen["reaction"] and world.h.calls == 2 and len(world.h.agents) == 1


@pytest.mark.parametrize("provider", CHAINS)
def test_control_a_real_message_ends_that_wait(chain, provider):
    world = chain(provider)
    seen = {}

    def after(messages):
        seen["result"] = tool_text(messages, "await_messages")
        return finish()

    world.h.script = [call(("await_messages", {"timeout_sec": 30})), after]
    post(world, world.spec.first)
    until(lambda: world.h.calls == 1, "the wait")
    time.sleep(0.2)
    post(world, world.spec.later)
    settled(world)
    assert '"reason": "owner_mailbox_pending"' in seen["result"]


@pytest.mark.parametrize("provider", CHAINS)
@pytest.mark.parametrize("lent,reason", [(True, "reaction_context_only"), (False, "current_author")])
def test_an_owner_question_park_runs_over_a_real_reaction(chain, provider, monkeypatch, lent, reason):
    """Its conversation is lent, or held by a custody blocker: room context, or unread mail. Neither ends the park."""
    def short_bound(ctx, **_arguments):
        ctx._owner_wait_deadline_at = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=6)).isoformat()
        return "{}"

    if not lent:
        monkeypatch.setattr("ouroboros.presence_continuation._custody_blockers", lambda *_a: ["held by this test"])
    monkeypatch.setattr("supervisor.workers.get_event_q", lambda *, create=True: SimpleNamespace(put=lambda _event: None))
    world = chain(provider, tools={"short_bound": short_bound})
    seen = {}

    def after(messages):
        seen["reaction"] = seen_reaction(messages)
        return finish()

    world.h.script = [call(("escalate", {"question": "May I start?", "options": [{"label": "Yes"}], "wait_for_answer": True,
                                         "max_wait_minutes": 1, "request_id": "q1"}), ("short_bound", {})), after]
    post(world, world.spec.first)
    until(lambda: (load_task_result(world.data, world.author().task_id).get("owner_wait") or {}).get("state") == "waiting", "the park")
    post(world, world.spec.heard)
    assert attention(world)[world.spec.heard] == reason
    time.sleep(1.5)  # a wake lands within the park's one-second poll
    assert world.h.calls == 1 and len(world.h.agents) == 1
    settled(world)
    assert load_task_result(world.data, world.author().task_id)["owner_wait"]["resume_reason"] == "timeout" and seen["reaction"]


@pytest.mark.parametrize("provider", CHAINS)
def test_a_review_wave_keeps_its_author_over_a_real_reaction(chain, provider):
    from tests.test_presence_continuation import LEFT, review

    world = chain(provider, review_mode="auto")
    seen = {}

    def after(messages):
        seen.update(result=tool_text(messages, "review"), reaction=seen_reaction(messages))
        return finish()

    world.h.script = [review(answer=FINAL), after]
    post(world, world.spec.first)
    assert world.h.entered.wait(60), "the held reviewer never started"
    post(world, world.spec.heard)
    assert attention(world)[world.spec.heard] == "current_author"
    time.sleep(1.0)
    assert world.h.calls == 1
    world.h.release.set()
    settled(world)
    assert world.h.settled.wait(60)
    assert LEFT not in seen["result"] and seen["reaction"]


@pytest.mark.parametrize("provider", CHAINS)
def test_a_delegate_wait_and_a_transport_backoff_run_over_a_real_reaction(chain, provider):
    import ouroboros.delegate_supervision as supervision
    from ouroboros import loop_transport

    seen = {}

    def delegate(ctx, **_arguments):
        reads = []

        def wait_once(_ctx, run_id, *_args, **_kwargs):  # the physical observer, as in test_place_arrival_wake
            reads.append(run_id)
            if len(reads) == 1:
                post(world, world.spec.heard)
            return json.dumps({"status": "no_progress" if len(reads) == 1 else "terminal", "run_id": run_id,
                               "last_seq": len(reads), **({"state": "succeeded"} if len(reads) > 1 else {})})

        seen["wake"] = json.loads(supervision.supervised_wait(ctx, "run-1", wait_once=wait_once).text)
        seen["reads"] = len(reads)
        return "{}"

    def backoff(ctx, **_arguments):
        started = time.monotonic()
        loop_transport.transport_wait_step(
            loop_transport.TransportWaitEpisode(started_monotonic=started), tools=SimpleNamespace(_ctx=ctx),
            error_kind="transport_unavailable", drive_root=ctx.drive_root, drive_logs=ctx.drive_root / "logs", task_id=ctx.task_id,
            model="test-model", emit_progress=lambda *_a, **_k: None, incoming_messages=None, owner_msg_seen=set(ctx._loop_mailbox_seen_ids))
        seen["slept"] = time.monotonic() - started
        return "{}"

    world = chain(provider, tools={"delegate": delegate, "backoff": backoff})

    def after(messages):
        seen["reaction"] = seen_reaction(messages)
        return finish()

    world.h.script = [call(("delegate", {}), ("backoff", {})), after]
    post(world, world.spec.first)
    settled(world)
    assert seen["reads"] == 2 and seen["wake"]["status"] == "terminal"
    assert not [row for row in seen["wake"].get("wake_events") or () if row.get("type") == "addressed_message"]
    assert seen["slept"] >= loop_transport.NETWORK_WAIT_BACKOFF_START_SEC - 0.2, "unread context-only mail cut the backoff short"
    assert seen["reaction"]


# --- The owed final, the reply it answers, and a gesture beside them --------------------------------------


@pytest.mark.parametrize("provider,outcome", [(provider, name) for provider, spec in CHAINS.items() for name in spec.receipts])
def test_a_real_gesture_receipt_is_an_action_beside_the_owed_final(chain, provider, outcome):
    from ouroboros.presence_context import _previous_turn_line

    holder = {}
    world = holder["world"] = chain(provider, tools={"gesture": gesture_tool(holder, outcome)})
    world.h.script = [call(("gesture", {})), finish()]
    post(world, world.spec.first)
    view = settled(world)
    assert (view["outcome"], view["text"]) == ("message", FINAL), "a gesture replaced the owed final"
    assert world.h.calls == 2
    pointer = until(lambda: (_read_previous_turn(world.data, world.spec.key) or {}).get("task_id") == world.author().task_id
                    and _read_previous_turn(world.data, world.spec.key), "the pointer")
    assert pointer["message"] == FINAL and pointer["transport_sends"] == []
    line = _previous_turn_line(pointer)
    assert "action, not speech" in line and FINAL in line
    reaction = world.wire["receipts"][world.spec.receipts[outcome]]["message"]["reaction"]
    assert reaction["schema_version"] == 1 and [action.get("effect") for action in pointer["transport_actions"]] == [reaction.get("effect")]


@pytest.mark.parametrize("provider", CHAINS)
def test_a_real_gesture_with_explicit_silence_is_the_whole_answer(chain, provider):
    holder = {}
    world = holder["world"] = chain(provider, tools={"gesture": gesture_tool(holder)})
    world.h.script = [call(("gesture", {})), finish("A reaction was enough.", silent=True)]
    post(world, world.spec.first)
    view = settled(world)
    assert (view["outcome"], view["text"]) == ("silent", "")
    pointer = until(lambda: (_read_previous_turn(world.data, world.spec.key) or {}).get("task_id") == world.author().task_id
                    and _read_previous_turn(world.data, world.spec.key), "the pointer")
    assert pointer["delivery"] == "unknown" and pointer["transport_sends"] == [], "a gesture read as confirmed speech"
    assert [action["operation"] for action in pointer["transport_actions"]] != []


def test_a_telegram_edit_report_stays_ordinary_beside_reactions(chain):
    world = chain("telegram")
    reply = world.client.post("/presence/delivery", headers=HEADERS, json=world.wire["receipts"]["telegram-operation:edit"])
    assert reply.status_code == 200 and reply.json()["recorded"] is True
    row = next(entry.row for entry in dialogue_store.iter_rows(world.data, "chat") if entry.row.get("type") == "presence_delivery")
    assert row["transport"]["message"]["replacement_text"] == "edited words" and "reaction" not in row["transport"]["message"]


@pytest.mark.parametrize("provider", CHAINS)
@pytest.mark.parametrize("message", [False, True])
def test_a_reaction_never_moves_the_reply_a_covered_message_does(chain, provider, message):
    world = chain(provider)
    go = threading.Event()

    def held(_messages):
        assert go.wait(60)
        return call(("no_such_probe", {}))

    world.h.script = [call(("no_such_probe", {})), held, finish()]
    post(world, world.spec.first)
    until(lambda: world.h.calls == 2, "the author's held send")
    if message:
        post(world, world.spec.later)
    post(world, world.spec.heard)
    assert attention(world)[world.spec.heard] == "current_author"
    go.set()
    assert settled(world)["reply_anchor"]["source_event_id"] == (world.spec.later if message else world.spec.first)


@pytest.mark.parametrize("provider", CHAINS)
def test_a_reaction_during_the_finishing_call_keeps_the_ready_final(chain, provider):
    world = chain(provider)

    def finishing(_messages):
        post(world, world.spec.heard)
        assert attention(world)[world.spec.heard] == "current_author"
        return finish()

    world.h.script = [finishing]
    post(world, world.spec.first)
    assert settled(world)["text"] == FINAL and world.h.calls == 1


# --- Every reader, on rows real connectors produced -----------------------------------------------------------


@pytest.mark.parametrize("provider", CHAINS)
def test_every_reader_keeps_real_reactions_as_labelled_context(chain, provider):
    from starlette.applications import Starlette
    from starlette.routing import Route

    from ouroboros import chat_chain, memory_view as mv
    from ouroboros.consciousness_wake import observe_wake
    from ouroboros.dialogue_evidence import read_room_source
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.memory import Memory
    from ouroboros.place_arrivals import account_candidates, activity
    from ouroboros.tools.chronicle import check_quotes
    from ouroboros.tools.plan_dialogue import dialogue_view

    holder = {}
    world = holder["world"] = chain(provider, tools={"gesture": gesture_tool(holder)})
    world.h.script = [call(("gesture", {})), finish()]
    post(world, world.spec.first)
    settled(world)
    until(lambda: not world.h.agents[-1]._busy, "the author's end")
    for source in world.spec.reactions:
        post(world, source)
    rows = [entry.row for entry in dialogue_store.iter_rows(world.data, "chat")]
    reaction = next(row for row in rows if row.get("client_message_id") == world.spec.heard)
    room = reaction["chat_id"]
    with TestClient(Starlette(routes=[Route("/api/chat/history", make_chat_history_endpoint(world.data))])) as client:
        shown = client.get("/api/chat/history", params={"chat_id": room}).json()["messages"]
    labelled = [row for row in shown if (row.get("place_history") or {}).get("reaction")]
    assert labelled and all(row["text"] == "" for row in labelled)
    history = Memory(drive_root=world.data).chat_history(count=50)
    assert "Reaction, not speech" in history and "Reaction action, not speech" in history
    task = {"id": "focus-next", "chat_id": 1, "metadata": {"usage_category": "consciousness", "wake_cause": "focus", "focus": {
        "room_chat_id": room, "place_id": world.place.place_id, "conversation_key": reaction["transport"]["conversation_key"]}}}
    view = mv.capture_memory_view(world.data, task, mv.view_spec_for_task(task, world.data))
    assert "[Reaction, not speech:" in mv.render_room(view)
    assert not any(item["owner_side"] for item in view.room["lane1"] if item["kind"] == "place_fact")
    source = read_room_source(world.data, room)
    projected = [row for row in source["rows"] if row.get("reaction")]
    retained = [row for row in rows if row.get("chat_id") == room and (row.get("transport") or {}).get("message", {}).get("reaction")]
    assert [row["reaction"]["fact"] for row in projected] == [row["transport"]["message"]["reaction"] for row in retained]
    conversation, _snapshot, _facts = dialogue_view({**source, "lines": source["text"].count("\n")})
    assert "[Reaction, not speech:" in conversation and "[Reaction action, not speech:" in conversation
    observed = observe_wake(world.data, boundary=None, since=0.0, now=time.time(), reason="timer").full_text()
    assert "Presence reaction in" in observed and "Reaction, not speech" in observed
    assert activity(world.data, world.place)["unanswered"] == 0
    assert "-77" not in [row["platform_actor_id"] for row in account_candidates(world.data, world.place.place_id)]
    address = chat_chain.format_address(chat_chain.row_address(reaction))
    for speaker in ("human", "ouroboros", "host", "unattributed"):
        assert check_quotes(world.data, [{"address": address, "speaker": speaker, "text": "added"}]) == (False, 0), speaker
