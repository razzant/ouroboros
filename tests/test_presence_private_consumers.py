"""#2780 at its real consumers: what a private start is to the connector, the owner's room and the scheduler.

The real token, Host routes, capture, attention, question, HTTP answer, turn drain, ``start_task``, supervisor promotion,
queue publication, durable intake, worker body and scheduler run; only the model and Safety's verdict are scripted. An
outward root of the same turn is each test's positive control: the same harness reaches it, so what the private root
does not get is the door's own answer. Synthetic data; no connector runs and nothing is sent anywhere.
"""
from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import pytest

from ouroboros.conversation_lease import open_root_intake, root_conversation_key
from ouroboros.dialogue_provenance import presence_caller_binding, presence_related_work, run_origin
from ouroboros.presence_admission import admit_presence_turn
from ouroboros.presence_runner import build_presence_result_event
from ouroboros.projects_registry import create_project
from ouroboros.task_results import load_task_result
from tests.test_place_conversation_work import (
    HEADERS,
    attention,
    call,
    dispatched,
    event_of,
    finish,
    place_rows,
    post,
    run_root,
    until,
)
from tests.test_place_conversation_work import host as host  # fixture
from tests.test_presence_owner_contact import _run, contact_door
from tests.test_presence_private_start import FOCUS_KEYS, SPEAKER_KEYS, _answered

pytestmark = pytest.mark.usefixtures("provider_catalog_offline")

PRIVATE = {"objective": "Investigate the source", "title": "Source research", "destination": "owner"}
OUTWARD = {"objective": "Answer them", "title": "Answer", "destination": "place"}
WORDS = "Read the public context only; do not promise implementation."  # the owner's answer (``_answer``)
ARRIVAL = "Is anyone looking into this?"
FINAL, OUTWARD_FINAL = "Private report for the owner alone.", "The answer for the conversation."
DUE = "2000-01-01T00:00:00+00:00"


@pytest.fixture
def asked(host, monkeypatch):
    """The real Host world and, in its conversation, a turn able to ask the owner and hold their HTTP answer."""
    world = host()
    event = event_of(world, "the-asking-turn")
    turn = contact_door(world.data, monkeypatch, world.workers.get_event_q())
    turn.event = event
    turn.admission = admit_presence_turn(
        drive_root=world.data, authenticated_transport_skill=world.skill, provider=event.provider,
        account_id=event.account_id, conversation_id=event.conversation_id, thread_id=event.thread_id)
    return world, turn


def _admission_lane(world, turn_events):
    """The real promote handler over the world's published pool; its lane queues what it pops (``promote``'s shape)."""
    from supervisor.events import _handle_promote_chat_to_task

    def enqueued(task):
        for internal in ("_admission_token", "_require_unique_task_id", "_require_worker_pool"):
            task.pop(internal, None)
        world.queue.PENDING.append(task)
        return task

    lane = SimpleNamespace(DRIVE_ROOT=world.data, WORKERS=world.workers.WORKERS, PENDING=world.queue.PENDING, bridge=None,
                           append_jsonl=lambda *_a, **_kw: None, persist_queue_snapshot=lambda **_kw: True,
                           enqueue_task=enqueued, load_state=lambda: {"owner_chat_id": 1})

    def put_nowait(event):
        if event.get("type") == "promote_chat_to_task":
            return _handle_promote_chat_to_task(event, lane)
        return turn_events.put_nowait(event)

    return SimpleNamespace(put_nowait=put_nowait)


def _started(world, turn, *starts, before=None):
    """One real turn: it asks, holds the owner's answer and calls ``start_task`` as given; it publishes its handoff as a
    turn's final does. Returns what admission queued, in order. ``before(ctx, handle)`` runs first."""

    def during(ctx, task, _agent):
        handle = _answered(turn, ctx)
        ctx.event_queue = _admission_lane(world, ctx.event_queue)
        if before:
            before(ctx, handle)
        for arguments in starts:
            quote = {"owner_quote": handle} if arguments["destination"] == "owner" else {}
            out = ctx._consumer_tools.execute("start_task", {**arguments, **quote})
            assert out.startswith("OK: task"), out
        build_presence_result_event(task, "On it.", ctx, terminal_origin="model_final")

    assert _run(turn, during).outcome == "silent"
    return list(world.queue.PENDING)


def _ended(box):
    box["thread"].join(120)
    assert not box["thread"].is_alive() and "error" not in box, box.get("error")
    frames = []
    while not box["out"].empty():
        frames.append(box["out"].get_nowait())
    return frames


def _work(world, work_ref):
    return world.client.get(f"/presence/work/{work_ref}", headers=HEADERS, params={"binding_id": world.place.place_id})


def _room(root, chat_id):
    from ouroboros.gateway.history import _assemble_history_response

    return json.dumps(json.loads(_assemble_history_response(root, chat_id, 50, 0))["messages"], ensure_ascii=False)


@pytest.mark.parametrize("facts", [("any_message",), ("mention",)], ids=["untagged-in-its-thread", "a-wake"])
def test_the_next_message_of_the_conversation_never_reaches_a_private_root_alone_in_it(asked, facts):
    """The private root runs in the real worker body while the connector posts: an untagged follow-up is heard by the
    room and held by no work; a wake is an ordinary new focus. Neither is that root's mail."""
    world, turn = asked
    [root] = _started(world, turn, PRIVATE)
    seen = {}

    def model(_messages):
        seen["arrival"] = post(world, text=ARRIVAL, facts=facts)
        return finish(FINAL)

    _ended(run_root(world, dispatched(world, root), model))
    source, reply = seen["arrival"]
    assert reply.status_code == 200, reply.text
    assert root_conversation_key(root) == "" and place_rows(world.data, root["id"]) == []
    assert attention(world, source).get("recipient", "") == ""
    if "mention" in facts:
        until(lambda: any(ARRIVAL in words for words in world.focus_inputs), "the ordinary focus")
    else:
        assert reply.json()["outcome"] == "silent" and attention(world, source)["reason"] == "no_wake_match"
        assert not any(ARRIVAL in words for words in world.focus_inputs)


def test_beside_an_outward_root_the_next_message_goes_to_the_outward_root(asked):
    """Two live roots of one conversation, one its reader: a private root that read it too would make this a choice."""
    world, turn = asked
    private, outward = _started(world, turn, PRIVATE, OUTWARD)
    dispatched(world, private), dispatched(world, outward)
    assert open_root_intake(world.data, private) is False and open_root_intake(world.data, outward) is True
    spent = world.allowance.calls

    source, reply = post(world, text=ARRIVAL)

    assert reply.status_code == 200 and reply.json()["outcome"] == "silent", reply.text
    details = attention(world, source)
    assert (details["reason"], details["recipient"]) == ("conversation_work", outward["id"])
    assert [row["place_arrival"]["source_event_id"] for row in place_rows(world.data, outward["id"])] == [source]
    assert place_rows(world.data, private["id"]) == []
    assert not world.focus_inputs and world.allowance.calls == spent  # no second focus and no allowance


@pytest.mark.parametrize("room", ["main", "project"])
def test_a_private_final_lands_in_its_owners_room_and_only_the_outward_final_is_a_connectors(asked, room):
    """The same worker body ends both roots. The owner's root writes its own room and no connector can read it by its
    reference; the outward root's final is published to the connector and to no room."""
    world, turn = asked
    project = create_project(world.data, "owner-research", name="Owner research") if room == "project" else None
    where = {"project_id": project["id"], "workspace": "none"} if project else {}
    private, outward = _started(world, turn, {**PRIVATE, **where}, OUTWARD)
    expected = project["chat_id"] if project else 1
    assert private["chat_id"] == expected != outward["chat_id"]

    frames = _ended(run_root(world, dispatched(world, private), finish(FINAL)))
    assert not [frame for frame in frames if frame.get("type") == "presence_result"]
    delivered = [frame for frame in frames if frame.get("type") == "send_message" and FINAL in str(frame.get("text"))]
    assert delivered and {frame["chat_id"] for frame in delivered} == {expected}
    assert load_task_result(world.data, private["id"])["status"] == "completed"
    for frame in delivered:
        turn.bus.put(frame)
    turn.pump()
    assert FINAL in _room(world.data, expected) and FINAL not in _room(world.data, outward["chat_id"])

    refused = _work(world, private["id"])
    assert refused.status_code == 404 and refused.json()["code"] == "presence_work_not_found"
    assert FINAL not in refused.text
    assert world.client.get(f"/presence/work/{private['id']}", headers=HEADERS).status_code == 404  # no binding narrows it

    frames = _ended(run_root(world, dispatched(world, outward), finish(OUTWARD_FINAL)))
    assert not [frame for frame in frames if frame.get("type") == "send_message" and OUTWARD_FINAL in str(frame.get("text"))]
    read = _work(world, outward["id"])
    assert read.status_code == 200 and read.json()["text"] == OUTWARD_FINAL
    assert read.json()["status"] == "completed" and FINAL not in read.text
    assert OUTWARD_FINAL not in _room(world.data, expected)


@pytest.mark.parametrize("room", ["main", "project"])
def test_the_two_step_followup_fires_as_the_owners_scheduled_work_and_reads_their_words_one_hop(asked, room):
    """The owner's route of 5 November 2026. The turn cannot schedule the owner's work; it starts it, and that root
    schedules. The row is saved and read back, fires through the real scheduler, admission and resource resolution, and
    the occurrence is no new owner ingress: their words reach it only through its direct origin's records."""
    from ouroboros.owner_relay import held_owner_messages
    from ouroboros.owner_words import objective_author_text
    from ouroboros.tool_access import lineage_task_ids
    from supervisor.queue import list_scheduled_tasks

    world, turn = asked
    world.monkeypatch.setattr("ouroboros.agent.make_agent", world.h.factory)  # a forked Project root's own agent
    project = create_project(world.data, "owner-research", name="Owner research") if room == "project" else None
    where = {"project_id": project["id"]} if project else {}
    refused = {}

    def schedule_owner_work(ctx, handle):
        arguments = {"objective": "Recheck the source", "run_at": DUE, "relation": "independent"}
        for name, choice in (("missing", {}), ("invalid", {"destination": "nowhere"}), ("owner", {"destination": "owner"}),
                             ("quoted", {"destination": "owner", "owner_quote": handle})):
            refused[name] = ctx._consumer_tools.execute("schedule_followup", {**arguments, **choice})

    [root] = _started(world, turn, {**PRIVATE, **where}, before=schedule_owner_work)
    assert "FOLLOWUP_DESTINATION_REQUIRED" in refused["missing"] and "destination" in refused["invalid"]
    assert "FOLLOWUP_DESTINATION_OWNER" in refused["owner"] and "start_task(destination='owner'" in refused["owner"]
    assert "unsupported argument(s): owner_quote" in refused["quoted"]  # no quote buys this tool the owner's route
    assert "FOLLOWUP_SCHEDULED" not in str(refused) and list_scheduled_tasks(world.data)["tasks"] == []

    note = str(pathlib.Path(root.get("drive_root") or world.data) / "task_drives" / root["id"] / "owner-words.md")
    _ended(run_root(world, dispatched(world, root),
                    call(("write_file", {"root": "task_drive", "path": note, "content": WORDS})),
                    call(("schedule_followup", {"objective": "Recheck the source", "run_at": DUE, "relation": "independent"})),
                    finish(FINAL)))
    assert world.queue.RUNNING.pop(root["id"], None)

    saved = json.loads((world.data / "state" / "scheduled_tasks.json").read_text(encoding="utf-8"))
    [row] = saved["tasks"] if isinstance(saved, dict) else saved
    assert row["followup_origin"]["task_id"] == root["id"] and row["followup_relation"]["kind"] == "independent"
    assert row["task"]["metadata"]["resource_intent"]["kind"] == ("room_default" if project else "system_repo")
    assert not {"destination", "owner_quote", "origin_message_ref", "origin_message_text"} & set(row["task"]["metadata"])
    assert not {"task_contract", "presence"} & set(row["task"])
    assert row["writer_kind"] == "owner"

    world.queue.check_scheduled_tasks()
    world.queue.check_scheduled_tasks()
    fired = next(task for task in world.queue.PENDING if (task.get("metadata") or {}).get("schedule_occurrence"))
    stored = next(item for item in list_scheduled_tasks(world.data)["tasks"] if item["id"] == row["id"])
    assert not stored.get("followup_hold") and not stored.get("last_error"), stored
    assert load_task_result(world.data, fired["id"])["schedule_admission"]["status"] == "accepted"
    metadata = fired["metadata"]
    assert metadata["resource_intent"]["kind"] == row["task"]["metadata"]["resource_intent"]["kind"]
    assert (fired.get("project_id") or "") == (project["id"] if project else "")
    assert fired["chat_id"] == (project["chat_id"] if project else 1)
    assert metadata["work_kind"] == "owner" and metadata["presence_binding_relation"] == {"binding_id": world.place.place_id}
    assert not (SPEAKER_KEYS | FOCUS_KEYS) & set(metadata) and presence_related_work(world.place.place_id, fired)

    # No new owner ingress: the follow-up carries the root's already-admitted owner corpus by value,
    # but it gets no new owner stamp or source handle of its own.
    assert run_origin(fired)["owner_ingress"] is False and not fired.get("origin_message_ref")
    assert metadata.get("owner_corpus")
    assert not {"origin_message_ref", "origin_message_text"} & set(metadata)
    assert metadata["objective_author"] == {"kind": "task", "task_id": root["id"]}

    seen = {}

    def read_origin(_messages):
        return call(("get_task_result", {"task_id": root["id"], "include_authority": True}),
                    ("read_file", {"root": "task_drive", "path": note}))

    def judged(messages):
        seen["read"] = {row["tool_call_id"]: str(row["content"]) for row in messages if row.get("role") == "tool"}
        seen["ctx"] = world.h.agents[-1].tools._ctx
        return finish("Compared.")

    _ended(run_root(world, dispatched(world, fired), read_origin, judged))
    assert WORDS in seen["read"]["read_file-1"], seen["read"]  # what the origin left in its drive
    assert WORDS in seen["read"]["get_task_result-0"], seen["read"]  # and its record's owner origin
    assert lineage_task_ids(seen["ctx"]) == (fired["id"], root["id"])
    assert presence_caller_binding(seen["ctx"]) is None and held_owner_messages(seen["ctx"]) == {}
    assert WORDS in json.dumps(getattr(seen["ctx"], "_owner_directives", []), ensure_ascii=False)
    assert objective_author_text(seen["ctx"]) == f"drafted by task {root['id']}, not spoken by my human"
