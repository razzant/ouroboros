"""A typed reaction at the readers beyond the room's own: the scheduled wake's observation, and a row the call
door retained before it checked a reaction's shape."""
import time
from dataclasses import replace

from ouroboros.consciousness_wake import observe_wake
from ouroboros.memory import Memory
from ouroboros.presence_admission import admit_presence_turn
from ouroboros.presence_runner import capture_presence_event
from tests.test_place_reactions import ANON, CHAT, USER, fact, history, stored, wire
from tests.test_place_reactions import room as room


def post(world, source, actor, reaction=None, **body):
    response = world.client.post("/presence/turn", json=wire(replace(world.event, source_event_id=source), actor,
                                                             reaction, **body))
    assert response.status_code == 200, response.text


def test_a_wake_lists_each_reaction_as_its_context_fact_never_as_a_correspondent_message(room):
    world = room(["mention"])
    target = fact()
    target["target"] |= {"author": {"platform_actor_id": "U2", "display_name": "Bo"},
                         "excerpt": {"text": "the plan", "truncated": True}}
    post(world, "e1", USER, target)
    post(world, "e2", CHAT, fact())
    post(world, "e3", ANON, fact("count"))
    post(world, "e4", {"platform_actor_id": "person"}, message={}, text="Real words.", facts=[])
    assert not world.calls  # a reaction is context; the unaddressed message matched no wake
    observation = observe_wake(world.root, boundary=None, since=0, now=time.time())
    assert [kind for kind, _offset, _line in observation.events] == ["correspondent_reaction"] * 3 + ["correspondent_message"]
    lines = [line for _kind, _offset, line in observation.events]
    assert lines[0].endswith(', by Ann (user): [Reaction, not speech: added thumbsup on message 1712.0001 from Bo: '
                             '"the plan" (excerpt, truncated)]')
    assert lines[1].endswith(", by News (chat): [Reaction, not speech: added thumbsup on message 1712.0001]")
    assert lines[2].endswith(", by Anonymous reactions (anonymous): "
                             "[Reaction, not speech: counts observed: thumbsup 3 on message 1712.0001]")
    assert lines[3].startswith("- Presence correspondent message in Place chat r") and lines[3].endswith(': "Real words."')
    text = observation.full_text()
    assert "3 Presence reactions (context, not speech), 1 Presence correspondent message;" in text
    assert ': ""' not in text and "correspondent messages" not in text  # no reaction reads as an empty message
    assert b'"composition": {"correspondent_message": 1, "correspondent_reaction": 3}' in observation.source_bytes()


def test_a_reaction_kept_before_its_door_checked_shape_still_reads_everywhere(room):
    world = room(["any_message"])
    admission = admit_presence_turn(drive_root=world.root, authenticated_transport_skill="bridge", provider="chat",
                                    account_id="a", conversation_id="r")
    kept = replace(world.event, actor=dict(ANON), message={"reaction": {**fact("count"), "counts": 3}})
    capture_presence_event(admission, kept, drive_root=world.root)  # what the call door retained before its check
    row, = stored(world.root)
    assert row["transport"]["message"]["reaction"]["counts"] == 3
    observation = observe_wake(world.root, boundary=None, since=0, now=time.time())
    (kind, _offset, line), = observation.events
    assert kind == "correspondent_reaction"
    assert line.endswith("[Reaction, not speech: no counts reported on message 1712.0001]")
    found = Memory(drive_root=world.root).chat_history(count=20, search="1712.0001")
    assert "error reading history" not in found and "no counts reported on message 1712.0001" in found
    shown, = history(world)
    assert shown["place_history"]["reaction"]["summary"] == "no counts reported on message 1712.0001"
