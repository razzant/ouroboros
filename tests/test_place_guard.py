"""Every dispatch guard refuses its target case and permits its working neighbor."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import place_guard as guard, tool_effects as _tool_effects
from ouroboros.places import Place, NoteEntry
from ouroboros.tools.registry_guards import _capability_resource_guard_result
from tests._safety_fixtures import supervisor as _supervisor
from tests.test_tool_effects import published_tool as _published_tool

published_tool, supervisor = _published_tool, _supervisor
REAL_EFFECT_OF, REAL_CALL_PLACE_FACTS = _tool_effects.effect_of, guard.call_place_facts  # bound before stubs


def _only(facts):
    """The sole destination row: call facts carry rows, never a top-level copy of one."""
    assert len(facts["places"]) == 1 and "place_id" not in facts and "sends_as" not in facts
    return facts["places"][0]


@pytest.fixture
def places(monkeypatch, tmp_path):
    rows = [Place("quiet", "connector", "service", "account", "C1", "Quiet", "silent",
                  notes=(NoteEntry("Owner instruction", "owner"), NoteEntry("My note", "ouroboros"))),
            Place("active", "connector", "service", "account", "C2", "Active")]
    monkeypatch.setattr(guard.places, "places_of_connector", lambda root, connector: rows if connector == "connector" else [])
    monkeypatch.setattr(guard.places, "load_places", lambda root: rows)
    monkeypatch.setattr(guard.places, "place_records", lambda root: guard.places.PlaceRecords(
        0, {row.place_id: row for row in rows}, (), {}))

    def resolve(root, *, connector, targets, conversation="", thread=""):
        values = [v for values in targets.values() for v in values]
        match = next((p for p in rows if p.conversation_id in values), None)
        return match, match is not None

    monkeypatch.setattr(guard.places, "place_for_target", resolve)
    monkeypatch.setattr(guard.conversation_lease, "live_author", lambda *a, **k: "")
    ctx = SimpleNamespace(drive_root=tmp_path, task_id="caller", task_metadata={})
    return rows, ctx


def test_advises_extension_and_mcp_both_directions(published_tool, places, monkeypatch):
    from ouroboros import tool_effects
    _, ctx = places
    ctx.task_metadata = {"initiator": "consciousness", "consciousness_autonomy": "observe"}
    for effect in ("write", "message", None):
        name, tool = published_tool(effect)
        assert "Advises" in guard.dispatch_refusal(ctx, name, {}, ext_tool=tool)
    name, tool = published_tool("read")
    assert guard.dispatch_refusal(ctx, name, {}, ext_tool=tool) == ""
    assert "Advises" in guard.dispatch_refusal(ctx, "mcp_demo__read", {}, is_mcp=True)
    # The server's own readOnlyHint is a declared read, so Advises may call it (disclosed widening).
    monkeypatch.setattr(tool_effects, "_MCP_ANNOTATIONS", {})
    tool_effects.record_mcp_annotations("demo", [{"name": "read", "annotations": {"readOnlyHint": True}}])
    assert guard.dispatch_refusal(ctx, "mcp_demo__read", {}, is_mcp=True) == ""
    ctx.task_metadata["consciousness_autonomy"] = "act"
    assert guard.dispatch_refusal(ctx, "mcp_demo__read", {}, is_mcp=True) == ""
    assert guard.dispatch_refusal(ctx, "read_file", {}) == ""


@pytest.mark.parametrize(("state", "metadata"), [
    pytest.param("silent", {}, id='silent-metadata0'),
    pytest.param("silent", {"schedule_id": "s"}, id='silent-metadata2'),
    pytest.param("paused", {"task_type": "task"}, id='paused-metadata1'),
    pytest.param("paused", {"initiator": "consciousness", "consciousness_autonomy": "full", "wake_cause": "focus"}, id='paused-metadata3'),
])
def test_place_state_from_every_origin_and_active_sibling(published_tool, places, metadata, state):
    rows, ctx = places
    rows[0] = replace(rows[0], state=state)
    ctx.task_metadata = metadata
    for effect in ("message", "write"):
        name, tool = published_tool(effect, target="room", conversation="room")
        refusal = guard.dispatch_refusal(ctx, name, {"room": "C1"}, ext_tool=tool)
        assert refusal.startswith("⚠️ RESOURCE_CONSTRAINT_BLOCKED:")
        assert state.title() in refusal and "Circles" in refusal
        assert "configure_presence" in refusal
        assert guard.dispatch_refusal(ctx, name, {"room": "C2"}, ext_tool=tool) == ""
    name, tool = published_tool("read", target="room")
    assert guard.dispatch_refusal(ctx, name, {"room": "C1"}, ext_tool=tool) == ""


@pytest.mark.parametrize("args", [{}, {"room": "unknown"}])
def test_unattributable_write_explains_connector_wide_hold(published_tool, places, args):
    rows, ctx = places
    name, tool = published_tool("write", target="room")
    assert "cannot be attributed" in guard.dispatch_refusal(ctx, name, args, ext_tool=tool)
    rows[0] = replace(rows[0], state="active")
    assert guard.dispatch_refusal(ctx, name, args, ext_tool=tool) == ""


def test_multi_recipient_send_cannot_hide_silent_place(published_tool, places):
    _, ctx = places
    name, tool = published_tool("message", target=["to", "cc", "bcc"])
    for args in ({"to": ["C2", "C1"]}, {"to": "C2", "bcc": "C1"}, {"to": "C2", "cc": "C1"}):
        assert "Quiet is Silent" in guard.dispatch_refusal(ctx, name, args, ext_tool=tool)
    assert guard.dispatch_refusal(ctx, name, {"to": ["C2"]}, ext_tool=tool) == ""


def test_one_author_names_task_and_allows_current_or_idle(published_tool, places, monkeypatch):
    _, ctx = places
    name, tool = published_tool("message", target="room", conversation="room", thread="thread")
    authors = {"service:account:C2:T": "other"}
    monkeypatch.setattr(guard.conversation_lease, "live_author", lambda key, **kw: authors.get(key, ""))
    assert "tell_task(task_id='other'" in guard.dispatch_refusal(ctx, name, {"room": "C2", "thread": "T"}, ext_tool=tool)
    assert guard.dispatch_refusal(ctx, name, {"room": "C2", "thread": "else"}, ext_tool=tool) == ""
    ctx.task_id = "other"
    assert guard.dispatch_refusal(ctx, name, {"room": "C2", "thread": "T"}, ext_tool=tool) == ""


def test_a_declared_generic_send_meets_the_place_guards_as_a_message(published_tool, places, monkeypatch):
    """A generic tool's call its connector declares as words is a message here: a Silent place refuses it through
    its nested target and another turn's conversation names its author; the tool's other calls keep its write."""
    rows, ctx = places
    name, tool = published_tool("write", target=[], sends={"when": {"path": "chat.postMessage"}, "text": "body.text",
                                                              "target": "body.channel", "conversation": "body.channel"})
    def send(room):
        return {"path": "chat.postMessage", "body": {"channel": room, "text": "hi"}}
    assert "Quiet is Silent" in guard.dispatch_refusal(ctx, name, send("C1"), ext_tool=tool)
    authors = {"service:account:C2:0": "other"}
    monkeypatch.setattr(guard.conversation_lease, "live_author", lambda key, **kw: authors.get(key, ""))
    assert "tell_task(task_id='other'" in guard.dispatch_refusal(ctx, name, send("C2"), ext_tool=tool)
    rows[0] = replace(rows[0], state="active")
    assert guard.dispatch_refusal(ctx, name, {"path": "pins.add", "body": {"channel": "C2"}}, ext_tool=tool) == ""
    authors.clear()  # the send now passes and the turn's record keeps the declared words
    assert guard.dispatch_refusal(ctx, name, send("C2"), ext_tool=tool) == ""
    assert ctx._place_message_calls[-1]["text"] == "hi"


def test_a_verdict_cannot_make_a_conversation_write_words(supervisor, published_tool, places, monkeypatch):
    """A write into a conversation (a reaction, a pin) is not words by itself: the dispatch guard gives it no author
    lease, and a verdict cannot make it words (a ``speech`` key in its answer is dropped). Another turn authoring
    that conversation refuses neither its SAFE reading nor, in Cyber Pro, its later identical issue after a flag."""
    h = supervisor
    monkeypatch.setattr(_tool_effects, "effect_of", REAL_EFFECT_OF)
    monkeypatch.setattr(guard, "call_place_facts", REAL_CALL_PLACE_FACTS)
    monkeypatch.setattr(guard.conversation_lease, "live_author", lambda key, **kw: "other" if key == "service:account:C2:0" else "")
    name, tool = published_tool("write", target="room", conversation="room")
    assert guard.dispatch_refusal(h.ctx, name, {"room": "C2", "emoji": "eyes"}, ext_tool=tool) == ""
    h.replies.append({"status": "SAFE", "reason": "words to the room", "speech": True})
    assert h.module.check_safety(name, {"room": "C2", "text": "We will ship Friday."}, ctx=h.ctx) == (True, "")
    monkeypatch.setattr(h.module, "get_runtime_mode", lambda: "cyber_pro")
    words = {"room": "C2", "text": "Ship it."}
    h.replies.append({"status": "DANGEROUS", "reason": "the note asks for a draft first", "speech": True})
    assert not h.module.check_safety(name, words, ctx=h.ctx)[0]
    h.later()
    ok, message = h.module.check_safety(name, words, ctx=h.ctx)
    assert ok and message.startswith("⚠️ SAFETY_NOTE: let through on your later identical call")


@pytest.mark.parametrize(("account_source", "broad_account"), [("focus", "*"), ("call", "")])
def test_broad_place_uses_resolved_account_for_author_lease(published_tool, tmp_path, broad_account, account_source):
    from ouroboros.conversation_lease import PresenceTurnGate
    from ouroboros.places import connect_place

    place = connect_place(tmp_path, connector="connector", provider="service", account_id=broad_account)
    name, tool = published_tool("message", target=["room", "account_id"], conversation="room", thread="thread")
    ctx = SimpleNamespace(drive_root=tmp_path, task_id="caller", task_metadata={})
    args = {"room": "C2", "thread": "T"}
    if account_source == "focus":
        ctx.task_metadata = {"presence": {"transport_skill": "connector", "place_id": place.place_id,
                                         "event": {"account_id": "resolved-account"}}}
    else:
        args["account_id"] = "resolved-account"
    lease = PresenceTurnGate(state_root=tmp_path / "state").acquire("service:resolved-account:C2:T")
    lease.bind_author("author")
    try:
        assert "tell_task(task_id='author'" in guard.dispatch_refusal(ctx, name, args, ext_tool=tool)
        assert _only(guard.call_place_facts(ctx, name, args, ext_tool=tool))["conversation_key"] == (
            "service:resolved-account:C2:T")
        ctx.task_id = "author"
        assert guard.dispatch_refusal(ctx, name, args, ext_tool=tool) == ""
        ctx.task_id = "caller"
        assert guard.dispatch_refusal(ctx, name, {**args, "account_id": "other-account"}, ext_tool=tool) == ""
        assert guard.dispatch_refusal(ctx, name, {**args, "thread": "other-thread"}, ext_tool=tool) == ""
        if account_source == "focus":
            ctx.task_metadata["presence"]["transport_skill"] = "unrelated"
            assert guard.dispatch_refusal(ctx, name, args, ext_tool=tool) == ""
    finally:
        lease.release()


def test_supervisor_facts_keep_every_place_once(published_tool, places):
    _, ctx = places
    name, tool = published_tool("message", target=["to", "cc"])
    facts = guard.call_place_facts(ctx, name, {"to": "C2", "cc": "C1"}, ext_tool=tool)
    assert facts["declared"] and facts["effect"] == "message"
    assert {row["place_id"] for row in facts["places"]} == {"active", "quiet"}
    reply = guard.call_place_facts(ctx, "place_reply", {"place_id": "quiet", "text": "hi"})
    assert _only(reply)["place_state"] == "silent" and reply["effect"] == "message"
    # Notes are standing delegation: the Supervisor's bundle carries them once (test_safety_place_evidence).
    assert "notes" not in _only(reply) and "notes" not in reply
    assert _only(reply)["conversation_key"] == "service:account:C1:0"


def test_a_send_into_another_place_carries_its_notes_beside_the_result(published_tool, places, monkeypatch, tmp_path):
    """Through the real tool-execution seam, in every mode: a send into a place other than the turn's focus carries
    that place's label and my human's notes beside its result; a send into the focus place and a write carry none."""
    import json
    from ouroboros import loop_tool_execution as execution
    from ouroboros.tools.tool_result import ToolResult
    _, ctx = places
    registry = SimpleNamespace(_ctx=ctx, CODE_TOOLS=frozenset(),
                               execute_result=lambda *a: ToolResult(status="ok", code="OK", text="queued"))
    monkeypatch.setattr(execution, "persist_call", lambda *a, **k: {})

    def run(name, args):
        call = {"id": "c1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        return execution._execute_single_tool(registry, call, tmp_path / "logs", "caller")["result"]
    name, _tool = published_tool("message", target="room", conversation="room")
    assert run(name, {"room": "C1", "text": "hi"}) == (
        'queued\n\n[DESTINATION] Quiet: your human\'s notes for this place: "Owner instruction"')
    ctx.task_metadata = {"focus": {"place_id": "quiet"}}  # the turn's own place: its notes are already in its frame
    assert run(name, {"room": "C1", "text": "hi"}) == "queued"
    ctx.task_metadata = {}
    name, _tool = published_tool("write", target="room", conversation="room")
    assert run(name, {"room": "C1"}) == "queued"


def test_actual_registry_callsite_blocks_and_allows(published_tool, places):
    _, ctx = places
    name, tool = published_tool("message", target="room")
    result = _capability_resource_guard_result(ctx, name, {"room": "C1"}, ext_tool=tool)
    assert result.code == "RESOURCE_CONSTRAINT_BLOCKED" and result.status == "blocked"
    assert _capability_resource_guard_result(ctx, name, {"room": "C2"}, ext_tool=tool) is None


@pytest.mark.parametrize("declares_place", [True, False])
def test_unreadable_place_store_holds_only_connector_writes(published_tool, places, monkeypatch, declares_place):
    """A store that needs repair never blocks a skill that connects no place."""
    from ouroboros import extension_registry_state as registry
    from ouroboros.places import PlaceError

    _, ctx = places

    def broken(root, connector):
        raise PlaceError("places_unreadable", "places.json", "Place records need repair")

    monkeypatch.setattr(guard.places, "places_of_connector", broken)
    rows = [{"provides": "place", "skill": "connector", "id": "place"}] if declares_place else []
    monkeypatch.setattr(registry, "live_contribution_projection", lambda: rows)
    name, tool = published_tool("write", target="channel")
    refusal = guard.dispatch_refusal(ctx, name, {"channel": "C1"}, ext_tool=tool)
    if declares_place:
        assert "need repair" in refusal and "Circles" in refusal
    else:
        assert refusal == ""
    name, tool = published_tool("read", target="channel")
    assert guard.dispatch_refusal(ctx, name, {"channel": "C1"}, ext_tool=tool) == ""
    assert guard.call_place_facts(ctx, name, {"channel": "C1"}, ext_tool=tool).get("places_unreadable", "") == (
        "Place records need repair" if declares_place else "")



def test_unreadable_place_store_judges_a_bound_call_by_its_own_publication(published_tool, places, monkeypatch):
    """A replacement that no longer provides a place cannot lift the hold from a call bound to the connector."""
    from ouroboros import extension_registry_state as registry
    from ouroboros.places import PlaceError

    _, ctx = places

    def broken(root, connector):
        raise PlaceError("places_unreadable", "places.json", "Place records need repair")

    monkeypatch.setattr(guard.places, "places_of_connector", broken)
    published_tool.bundle.contributions = [{"provides": "place", "id": "place"}]
    name, _raw = published_tool("write", target="channel")
    bound = registry.get_tool_stamped(name)
    monkeypatch.setattr(registry, "live_contribution_projection", lambda: [])
    assert "need repair" in guard.dispatch_refusal(ctx, name, {"channel": "C1"}, ext_tool=bound)

@pytest.mark.parametrize("kind,account,owner", [("user", "account", True), ("bot", "account", False), (None, "account", False)])
def test_call_facts_use_captured_identity_and_user_owner_match(published_tool, places, kind, account, owner):
    from ouroboros.places import OwnerAccount
    from tests.test_place_arrivals import seed_identity_arrival
    rows, ctx = places
    rows[1] = replace(rows[1], owner_accounts=(OwnerAccount("writer", account_id=account),))
    identity = {"sends_as": {"id": "writer", "label": "My mailbox", "kind": kind}}
    if kind:
        seed_identity_arrival(ctx.drive_root, rows[1], identity)
    name, tool = published_tool("message", target="room", conversation="room")
    facts = guard.call_place_facts(ctx, name, {"room": "C2"}, ext_tool=tool)
    if kind:
        assert _only(facts)["sends_as"] == identity["sends_as"] and _only(facts)["sends_as_owner"] is owner
    else:
        assert "sends_as" not in _only(facts) and "sends_as_owner" not in _only(facts)
    assert guard.call_place_facts(ctx, "read_file", {}) == {}


def test_call_facts_pick_the_named_accounts_identity_out_of_the_places_window(published_tool, places):
    """The call site hands its account to the captured read: a connector-wide place answers for the account named."""
    from tests.test_place_arrivals import seed_identity_arrival
    rows, ctx = places
    mine = {"sends_as": {"id": "mine", "label": "Mine", "kind": "user"}}
    other = {"sends_as": {"id": "other", "label": "Other", "kind": "user"}}
    seed_identity_arrival(ctx.drive_root, rows[1], mine, ts="2026-10-26T12:00:00Z", account_id="A")
    seed_identity_arrival(ctx.drive_root, rows[1], other, ts="2026-10-26T12:00:01Z", account_id="B")
    name, tool = published_tool("message", target=["room", "account_id"], conversation="room")
    for account, expected in (("A", mine), ("B", other)):
        facts = guard.call_place_facts(ctx, name, {"room": "C2", "account_id": account}, ext_tool=tool)
        assert _only(facts)["sends_as"] == expected["sends_as"], account


@pytest.mark.parametrize("root_source", ["drive_root", "budget_drive_root", "metadata"])
def test_call_identity_uses_context_root_not_configured_root(published_tool, places, monkeypatch, root_source):
    from tests.test_place_arrivals import seed_identity_arrival

    rows, ctx = places
    root, other_root = ctx.drive_root, ctx.drive_root / "other-installation"
    for captured_root, label in [(root, "Call mailbox"), (other_root, "Other mailbox")]:
        seed_identity_arrival(captured_root, rows[1], {"sends_as": {"id": "writer", "label": label, "kind": "user"}})
    monkeypatch.setattr("ouroboros.config.DATA_DIR", other_root)
    if root_source != "drive_root":
        ctx.drive_root = other_root
        if root_source == "metadata":
            ctx.task_metadata["budget_drive_root"] = root
        else:
            ctx.budget_drive_root = root
    name, tool = published_tool("message", target="room", conversation="room")
    facts = guard.call_place_facts(ctx, name, {"room": "C2"}, ext_tool=tool)
    assert _only(facts)["sends_as"]["label"] == "Call mailbox"


@pytest.mark.parametrize("kind", ["user", "bot", None])
def test_safety_call_uses_focus_snapshot_only_for_its_target(published_tool, places, kind):
    from tests.test_place_arrivals import seed_identity_arrival
    rows, ctx = places
    event = {"sends_as": {"id": "event", "label": "At the event", "kind": kind}} if kind else None
    captured = {"sends_as": {"id": "current", "label": "Newer arrival", "kind": "user"}}
    ctx.task_metadata = {"focus": {"place_id": "active", "identity": event, "identity_owner": True}}
    for place in rows:
        seed_identity_arrival(ctx.drive_root, place, captured)
    name, tool = published_tool("message", target="room", conversation="room")
    facts = _only(guard.call_place_facts(ctx, name, {"room": "C2"}, ext_tool=tool))
    if kind:
        assert facts["sends_as"] == event["sends_as"]
        assert facts["sends_as_owner"] is (kind == "user")
    else:
        assert "sends_as" not in facts and "sends_as_owner" not in facts
        del ctx.task_metadata["focus"]["identity"]  # A legacy focus has no declaration either.
        assert "sends_as" not in _only(guard.call_place_facts(ctx, name, {"room": "C2"}, ext_tool=tool))
    facts = _only(guard.call_place_facts(ctx, name, {"room": "C1"}, ext_tool=tool))
    assert facts["sends_as"] == captured["sends_as"] and facts["sends_as_owner"] is False


def test_supervisor_facts_carry_this_turns_own_sends_in_its_conversation_both_ways(published_tool, places, tmp_path):
    """Speech into the turn's own conversation is judged with what the turn already said there: the connector's
    receipts and the run's own dispatched message calls without one; elsewhere, or without a carrier, nothing."""
    from ouroboros.utils import append_jsonl
    _, ctx = places
    own = "service:account:C2:0"
    ctx.task_metadata = {"focus": {"place_id": "active"}, "presence": {
        "delivery_reporting_version": 1, "place_id": "active", "transport_skill": "connector",
        "event": {"conversation_key": own, "provider": "service", "account_id": "account", "conversation_id": "C2", "thread_id": ""}}}
    for row in ({"task_id": "caller", "direction": "in", "text": "hello"},
                {"task_id": "caller", "type": "presence_delivery", "text": "said earlier",
                 "transport": {"conversation_key": own, "delivery": {"state": "delivered", "delivery_id": "d1", "part_id": "0"}}}):
        append_jsonl(tmp_path / "logs" / "chat.jsonl", row)
    name, tool = published_tool("message", target=["room"], conversation="room")
    reply = guard.call_place_facts(ctx, "place_reply", {"place_id": "active", "text": "final"})
    assert reply["this_turn_sends"] == {"confirmed": '"said earlier"', "calls_without_receipt": []}
    assert guard.dispatch_refusal(ctx, name, {"room": "C2", "text": "queued words"}, ext_tool=tool) == ""
    assert guard.dispatch_refusal(ctx, name, {"room": "C2", "text": "said earlier"}, ext_tool=tool) == ""  # a receipt confirms it
    assert "Silent" in guard.dispatch_refusal(ctx, name, {"room": "C1", "text": "never left"}, ext_tool=tool)  # refused: not noted
    facts = guard.call_place_facts(ctx, "place_reply", {"place_id": "active", "text": "final"})
    pending = facts["this_turn_sends"]["calls_without_receipt"]
    assert [row["text"] for row in pending] == ["queued words"] and pending[0]["tool"] == name and pending[0]["at"]
    assert facts["this_turn_sends"]["confirmed"] == '"said earlier"'
    assert "this_turn_sends" in guard.call_place_facts(ctx, name, {"room": "C2", "text": "more"}, ext_tool=tool)
    assert "this_turn_sends" not in guard.call_place_facts(ctx, name, {"room": "C3", "text": "elsewhere"}, ext_tool=tool)
    assert "this_turn_sends" not in guard.call_place_facts(ctx, published_tool("read", target=["room"], conversation="room")[0],
                                                           {"room": "C2"}, ext_tool=tool)
    bare = SimpleNamespace(drive_root=tmp_path, task_id="other", task_metadata={})
    assert "this_turn_sends" not in guard.call_place_facts(bare, "place_reply", {"place_id": "active", "text": "final"})


def test_a_destinations_circle_words_trail_its_result_after_the_act(monkeypatch, tmp_path):
    """After a send into another place has run, its result carries that place's circle words with both origins, from
    one read of the store: a fact beside the act, never guidance the act had before it ran."""
    from ouroboros import places as real_places
    from ouroboros.tools.tool_result import ToolResult
    target = real_places.connect_place(tmp_path, connector="connector", provider="service", account_id="account",
                                       conversation_id="C9", label="Team room")
    circle = real_places.create_circle(tmp_path, name="Team", text="Circle words", author="owner")
    real_places.edit_place(tmp_path, target.place_id, "set_circle", author="ouroboros", circle_id=circle["circle_id"])
    monkeypatch.setattr(guard, "effect_of", lambda *a, **k: _tool_effects.ToolEffect(_tool_effects.EFFECT_MESSAGE))
    monkeypatch.setattr(guard, "_call_destinations", lambda *a: (None, {}, [target], [], True))
    ctx = SimpleNamespace(drive_root=tmp_path, task_id="caller", task_metadata={})
    result = guard.with_destination_facts(ctx, "ext_send", {"room": "C9"}, ToolResult(status="ok", code="OK", text="queued"))
    assert result.producer_text == "queued" and result.text.startswith("queued\n\n[DESTINATION] Team room: ")
    assert ('its circle "Team"\'s words, by your human, placed there by you (they still restrict here; consent needs '
            'your human\'s placement): "Circle words"') in result.text and result.host_annotations[-1] in result.text


@pytest.mark.parametrize(("author", "placed_by", "said"), [
    ("owner", "owner", ": "),  # their letter: restriction and consent, nothing to qualify
    ("owner", "ouroboros", " (they still restrict here; consent needs your human's placement): "),
    ("ouroboros", "owner", " (context, not your human's choice for this place): "),  # my words never become theirs
    ("ouroboros", "ouroboros", " (context, not your human's choice for this place): "),
], ids=["their-letter", "their-words-my-placement", "my-words-their-placement", "all-mine"])
def test_a_destinations_circle_words_say_what_binds_by_who_wrote_and_who_placed(tmp_path, author, placed_by, said):
    from ouroboros import places as real_places
    place = real_places.connect_place(tmp_path, connector="connector", provider="service", conversation_id="C9", label="Room")
    circle = real_places.create_circle(tmp_path, name="Team", text="Circle words", author=author)
    real_places.edit_place(tmp_path, place.place_id, "set_circle", author=placed_by, circle_id=circle["circle_id"])
    [entry] = guard.authored_notes(real_places.place_records(tmp_path), real_places.get_place(tmp_path, place.place_id))
    assert entry["standing_delegation"] is (author == placed_by == "owner")
    assert f'placed there by {"your human" if placed_by == "owner" else "you"}{said}"Circle words"' in guard.destination_line("Room", [entry])
