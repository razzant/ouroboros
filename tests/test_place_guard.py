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


def test_a_started_send_refused_a_further_step_carries_its_destination_facts(published_tool, places):
    """The target of a dispatched send stays visible beside its receipt; the host does not attest delivery."""
    from ouroboros.tools.tool_result import ToolResult
    _, ctx = places
    name, tool = published_tool("message", target="room", conversation="room")
    args = {"room": "C1", "text": "hi"}
    note = '[DESTINATION] Quiet: your human\'s notes for this place: "Owner instruction"'
    refusal = "⚠️ OWNER_OPERATION_REFUSED: the host refused this call a further action: stop (cancelled)."
    started = ToolResult(status="blocked", code="OWNER_OPERATION_REFUSED",
                         text=f"part one queued\n\n{refusal}",
                         meta={"dynamic_provider": True, "physical_dispatch": True,
                               "refused_step": True, "control_reason": "cancelled"},
                         producer_text="part one queued", host_annotations=(refusal,))
    annotated = guard.with_destination_facts(ctx, name, args, started, ext_tool=tool)
    assert annotated.text == f"part one queued\n\n{refusal}\n\n{note}"
    assert annotated.host_annotations == (refusal, note)
    assert annotated.producer_text == "part one queued" and annotated.status == "blocked"
    no_report = replace(started, status="error", code="EXTENSION_ERROR", text="handler error\n\n" + refusal, producer_text=None)
    reported = guard.with_destination_facts(ctx, name, args, no_report, ext_tool=tool)
    assert reported.host_annotations == (refusal, note) and reported.producer_text is None
    not_started = replace(started, meta={"dynamic_provider": True, "refused_step": True,
                                         "control_reason": "cancelled"})
    assert guard.with_destination_facts(ctx, name, args, not_started, ext_tool=tool) is not_started
    ordinary = ToolResult(status="blocked", code="RESOURCE_CONSTRAINT_BLOCKED",
                          text="⚠️ RESOURCE_CONSTRAINT_BLOCKED: Quiet is Silent")
    assert guard.with_destination_facts(ctx, name, args, ordinary, ext_tool=tool) is ordinary
    ctx.task_metadata = {"focus": {"place_id": "quiet"}}  # the turn's own place: its notes are already in its frame
    assert guard.with_destination_facts(ctx, name, args, started, ext_tool=tool) is started


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


@pytest.mark.parametrize("broad_account", ["", "*"])
@pytest.mark.parametrize("focus_account", [None, "A", "B"])
def test_grouped_accounts_keep_their_own_identity_and_focus(published_tool, tmp_path, broad_account, focus_account):
    from ouroboros import places
    from tests.test_place_arrivals import seed_identity_arrival

    place = places.connect_place(tmp_path, connector="connector", provider="service", account_id=broad_account,
                                 conversation_id="", label="All accounts")
    places.edit_place(tmp_path, place.place_id, "confirm_owner_account", author="owner",
                      platform_actor_id="owner-B", account_id="B")
    identities = {"A": {"sends_as": {"id": "bot-A", "label": "Bot mailbox", "kind": "bot"}},
                  "B": {"sends_as": {"id": "owner-B", "label": "Owner mailbox", "kind": "user"}}}
    for account, identity in identities.items():
        seed_identity_arrival(tmp_path, place, identity, account_id=account)
    metadata = {}
    if focus_account:
        metadata = {"focus": {"place_id": place.place_id, "identity": identities[focus_account],
                              "identity_owner": focus_account == "B"},
                    "presence": {"transport_skill": "connector", "event": {"account_id": focus_account}}}
    ctx = SimpleNamespace(drive_root=tmp_path, task_id="caller", task_metadata=metadata)
    name, tool = _grouped(published_tool, target=["account_id", "room"], conversation="room", touches=[
        {"conversation": "other_room", "account": "other_account", "effect": "message"},
        {"conversation": "copy", "account": "other_account", "effect": "write"}])
    facts = guard.call_place_facts(ctx, name, {"account_id": "A", "room": "room-A", "other_account": "B",
                                              "other_room": "room-B", "copy": "draft-B"}, ext_tool=tool)
    assert len(facts["places"]) == 2
    for row in facts["places"]:
        account = row["account_id"]
        assert row["sends_as"] == identities[account]["sends_as"]
        assert row["sends_as_owner"] is (account == "B")
        assert row["conversation_keys"] == [f"service:{account}:room-{account}:0"]
        assert row["effects"] == (["message"] if account == "A" else ["message", "write"])


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

    def reached(*_a):
        call = guard._CallReach(_tool_effects.ToolEffect(_tool_effects.EFFECT_MESSAGE), {})
        call.reach(target, _tool_effects.EFFECT_MESSAGE, "target", "account", [])
        return call
    monkeypatch.setattr(guard, "_call_destinations", reached)
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


def _grouped(published_tool, effect="message", **row):
    """A connector tool whose publication declares PluginAPI 2.5 and these fields."""
    published_tool.bundle.loaded_skill.manifest.plugin_api = {"version": "2.5", "capabilities": []}
    return published_tool(effect, **row)


def _store(tmp_path, *rows):
    """Real places of ``connector`` in a real store: ``(conversation, label, state[, thread])``."""
    from dataclasses import replace as _replace
    from ouroboros import places

    made = {}
    for conversation, label, state, *thread in rows:
        place = places.connect_place(tmp_path, connector="connector", provider="service", account_id="A",
                                     conversation_id=conversation, label=label, state=state,
                                     place_id=f"{label}-{conversation}")
        if thread:
            place = places._write(tmp_path, place.place_id, lambda old, t=thread[0]: _replace(old, thread_id=t))
        made[label] = place
    return made, SimpleNamespace(drive_root=tmp_path, task_id="caller", task_metadata={})


def test_a_group_address_meets_its_place_and_only_words_take_an_author(published_tool, tmp_path):
    """Recipients declared as a write group: a Paused bare address refuses the send, an Active one is written to
    without an author lease; a message group's conversation is authored like the call's own."""
    from ouroboros.conversation_lease import PresenceTurnGate
    from ouroboros.presence_bindings import conversation_key

    made, ctx = _store(tmp_path, ("C1", "Room", "active"), ("p@x", "Peer", "paused"), ("E1", "Emitted", "active"))
    name, tool = _grouped(published_tool, target="room", conversation="room", touches=[
        {"conversation": "to", "effect": "write"}, {"conversation": "emitted", "thread": "emitted", "effect": "message"}])
    args = {"account_id": "A", "room": "C1", "to": ["p@x"], "text": "hi"}
    assert "Peer is Paused" in guard.dispatch_refusal(ctx, name, args, ext_tool=tool)
    guard.places.edit_place(tmp_path, made["Peer"].place_id, "set_state", author="owner", state="active")
    for key, author, call, refused in (
            (conversation_key("service", "A", "p@x"), "writer", args, False),  # the recipient's write takes no author
            (conversation_key("service", "A", "E1", "E1"), "speaker", {**args, "emitted": "E1"}, True)):
        lease = PresenceTurnGate(state_root=tmp_path / "state").acquire(key)
        lease.bind_author(author)
        try:
            refusal = guard.dispatch_refusal(ctx, name, call, ext_tool=tool)
        finally:
            lease.release()
        assert (f"tell_task(task_id={author!r}" in refusal) is refused and (refusal == "") is not refused, refusal
    facts = guard.call_place_facts(ctx, name, {**args, "emitted": "E1"}, ext_tool=tool)
    rows = {row["place_id"]: row for row in facts["places"]}
    assert (rows[made["Peer"].place_id]["effects"], rows[made["Peer"].place_id]["conversation_keys"]) == (["write"], [])
    assert rows[made["Emitted"].place_id]["conversation_keys"] == [conversation_key("service", "A", "E1", "E1")]
    assert ctx._place_message_calls[-1]["conversation_keys"] == [conversation_key("service", "A", "C1")]


def test_the_declared_targets_threads_pair_without_a_product(published_tool, tmp_path):
    """One conversation with several threads, or several with one, pairs exactly; several of each is refused for words
    and changes, named for the repair, while a read is never refused."""
    from ouroboros.presence_bindings import conversation_key

    _made, ctx = _store(tmp_path, ("C1", "Room", "active"), ("C2", "Other", "active"))
    name, tool = published_tool("message", target="room", conversation="room", thread="topic")
    facts = guard.call_place_facts(ctx, name, {"room": "C1", "topic": ["t1", "t2"]}, ext_tool=tool)
    assert facts["places"][0]["conversation_keys"] == [conversation_key("service", "A", "C1", "t1"),
                                                       conversation_key("service", "A", "C1", "t2")]
    both = {"room": ["C1", "C2"], "topic": ["t1", "t2"]}
    refusal = guard.dispatch_refusal(ctx, name, both, ext_tool=tool)
    assert "cannot be paired" in refusal and "Nothing ran" in refusal
    name, tool = published_tool("read", target="room", conversation="room", thread="topic")
    assert guard.dispatch_refusal(ctx, name, both, ext_tool=tool) == ""


def test_one_argument_naming_conversation_and_thread_pairs_each_id_with_itself(published_tool, tmp_path):
    """A Paused place of C1 with thread C2 is no pair a call naming C1 and C2 (each its own thread) reaches."""
    _made, ctx = _store(tmp_path, ("C1", "Room", "active"), ("C1", "Cross", "paused", "C2"), ("C2", "Other", "active"))
    name, tool = published_tool("message", target=["account_id", "room"], conversation="room", thread="room")
    assert guard.dispatch_refusal(ctx, name, {"account_id": "A", "room": ["C1", "C2"]}, ext_tool=tool) == ""
    # Negative control: a call that does name C1 with thread C2 reaches the Paused place.
    name, tool = published_tool("message", target="room", conversation="room", thread="topic")
    assert "Cross is Paused" in guard.dispatch_refusal(ctx, name, {"room": "C1", "topic": "C2"}, ext_tool=tool)


@pytest.mark.parametrize("rooms,threads,pairs,thread_selector", [
    ("C1", ["t1", "t2"], [("C1", "t1"), ("C1", "t2")], "topic"),
    (["C1", "C2"], "t1", [("C1", "t1"), ("C2", "t1")], "topic"),
    (["C1", "C2"], ["C1", "C2"], [("C1", "C1"), ("C2", "C2")], "room"),
])
def test_optional_groups_preserve_each_primary_pair_and_its_keys(
        published_tool, tmp_path, rooms, threads, pairs, thread_selector):
    from ouroboros.presence_bindings import conversation_key

    made, ctx = _store(tmp_path, *((room, str(i), "active", thread) for i, (room, thread) in enumerate(pairs)))
    args = {"account_id": "A", "room": rooms, "topic": threads}
    declaration = dict(target=["account_id", "room", "topic"], conversation="room", thread=thread_selector)
    name, tool = published_tool("message", **declaration)
    assert {row["place_id"] for row in guard.call_place_facts(ctx, name, args, ext_tool=tool)["places"]} == {
        place.place_id for place in made.values()}
    name, tool = _grouped(published_tool, **declaration, touches=[{"conversation": "copies", "effect": "read"}])
    facts = guard.call_place_facts(ctx, name, args, ext_tool=tool)
    assert {row["place_id"]: row["conversation_keys"] for row in facts["places"]} == {
        made[str(i)].place_id: [conversation_key("service", "A", room, thread)]
        for i, (room, thread) in enumerate(pairs)}


_ADDRESS_GROUPS = [{"conversation": "to", "effect": "write"}, {"conversation": "folder", "effect": "write"},
                   {"conversation": "draft", "thread": "draft", "effect": "write"},
                   {"conversation": "copy_to", "account": "copy_account", "effect": "write"}]


@pytest.mark.parametrize("address,unknown", [
    ({}, True),  # the account alone names no place of it
    ({"to": ["new@x"]}, True), ({"folder": "Archive"}, True), ({"draft": "d-old"}, True),  # none of A's places covers
    ({"account_id": None, "to": ["new@x"]}, True),  # nor without an account
    # Naming no conversation, the call acts in its account, which no place of it covers: its addresses cannot vouch
    # for it (the primary never disappears beside groups) ...
    ({"to": ["C1"]}, True), ({"copy_to": "x@b", "copy_account": "B"}, True),
    # ... while naming its conversation they are what they are: a place's own conversation, an account no place covers.
    ({"room": "C1", "to": ["C1"]}, False), ({"room": "C1", "copy_to": "x@b", "copy_account": "B"}, False)])
@pytest.mark.parametrize("effect", ["message", "write"])
def test_inside_narrow_places_an_address_none_covers_stays_unknown(published_tool, tmp_path, effect, address, unknown):
    """An account with only conversation places: a new recipient, an unknown folder or an old draft none of them
    covers stays unknown whether or not its account is named, as a declared target does, and so does the call's own
    destination when it names no conversation (its account), so a Silent or Paused place of the connector holds the
    call; an address in an account no place covers is no place's. With no such place an unknown address refuses
    nothing."""
    made, ctx = _store(tmp_path, ("C1", "Room", "active"), ("dm-1", "Private", "paused"))
    name, tool = _grouped(published_tool, effect=effect, target=["account_id", "room"], conversation="room",
                          touches=_ADDRESS_GROUPS)
    args = {"account_id": "A", "text": "hi", **address}
    refusal = guard.dispatch_refusal(ctx, name, args, ext_tool=tool)
    assert ("cannot be attributed while Private is Silent or Paused" in refusal) is unknown, refusal
    assert unknown or refusal == "", refusal
    guard.places.edit_place(tmp_path, made["Private"].place_id, "set_state", author="owner", state="active")
    assert guard.dispatch_refusal(ctx, name, args, ext_tool=tool) == ""


@pytest.mark.parametrize("address,paused", [
    ({"to": ["new@x"]}, ""), ({"folder": "Archive"}, ""), ({"draft": "d-old"}, ""),  # the account's own place
    ({"to": ["dm-1"]}, "Private"),  # an exact Paused place outranks the account-wide Active one
    ({"copy_to": "x@b", "copy_account": "B"}, "")])
def test_an_account_wide_place_covers_unknown_addresses_and_an_exact_place_outranks_it(published_tool, tmp_path,
                                                                                       address, paused):
    """An account-wide Active place covers every address of its account no narrower place names: new recipients,
    unknown folders and old drafts are its own; an exact Paused place still holds its own address."""
    _made, ctx = _store(tmp_path, ("", "Account", "active"), ("dm-1", "Private", "paused"))
    name, tool = _grouped(published_tool, effect="write", target=["account_id", "room"], conversation="room",
                          touches=_ADDRESS_GROUPS)
    refusal = guard.dispatch_refusal(ctx, name, {"account_id": "A", **address}, ext_tool=tool)
    assert (f"{paused} is Paused" in refusal) if paused else refusal == "", refusal


def test_a_read_group_neither_refuses_nor_speaks(published_tool, tmp_path):
    made, ctx = _store(tmp_path, ("C1", "Room", "active"), ("S1", "Source", "paused"))
    name, tool = _grouped(published_tool, target="room", conversation="room",
                          touches=[{"conversation": "source", "thread": "source", "effect": "read"}])
    args = {"account_id": "A", "room": "C1", "source": "S1", "text": "hi"}
    assert guard.dispatch_refusal(ctx, name, args, ext_tool=tool) == ""
    rows = {row["place_id"]: row for row in guard.call_place_facts(ctx, name, args, ext_tool=tool)["places"]}
    assert (rows[made["Source"].place_id]["effects"], rows[made["Source"].place_id]["conversation_keys"]) == (["read"], [])


@pytest.mark.parametrize("row,args,refusal", [
    ({"touches": [{"conversation": "to", "effect": "write"}]}, {"to": "p@x"}, "connector's declaration needs a fix"),
    ({"touches": [{"conversation": "to", "thread": "topic", "effect": "write"}]}, {"to": "p@x", "topic": ["t"]},
     "one account and one thread"),
])
def test_a_call_whose_addresses_cannot_be_read_is_refused_whole(published_tool, tmp_path, row, args, refusal):
    _made, ctx = _store(tmp_path, ("C1", "Room", "active"))
    if "topic" in args:
        published_tool.bundle.loaded_skill.manifest.plugin_api = {"version": "2.5", "capabilities": []}
    name, tool = published_tool("write", target="room", conversation="room", **row)
    message = guard.dispatch_refusal(ctx, name, args, ext_tool=tool)
    assert message.startswith("⚠️ RESOURCE_CONSTRAINT_BLOCKED: ") and "Nothing ran" in message and refusal in message
    assert guard.call_place_facts(ctx, name, args, ext_tool=tool)["places"] == []


def test_a_group_write_into_the_turns_own_conversation_is_not_speech(published_tool, tmp_path):
    """Before a call starts, only where its words land can make it speak in the turn's conversation."""
    import json
    from ouroboros.place_reply import may_speak
    from ouroboros.presence_bindings import conversation_key

    made, ctx = _store(tmp_path, ("C1", "Own", "active"), ("C2", "Elsewhere", "active"))
    ctx.task_metadata = {"presence": {"place_id": made["Own"].place_id,
                                      "event": {"conversation_key": conversation_key("service", "A", "C1")}}}
    name, tool = _grouped(published_tool, target="room", conversation="room", touches=[
        {"conversation": "copy_to", "effect": "write"}, {"conversation": "also", "effect": "message"}])

    def speaks(**args):
        return may_speak(ctx, {"function": {"name": name, "arguments": json.dumps({"account_id": "A", "room": "C2", **args})}},
                         ext_tool=tool)
    assert speaks(copy_to="C1") is False
    assert speaks(also="C1") is True


def _scope(place_id, conversation, state="active", *, account="A", thread="", destination=None):
    return Place(place_id, "connector", "service", account, conversation, place_id, state, thread_id=thread,
                 destination=destination)


def _match(rows, conversation, match, *, account="A", **extra):
    return guard.places.target_place(rows, targets={"conversation_id": [conversation], "account_id": [account], **extra},
                                     conversation="conversation_id", thread="thread_id", match=match)


def test_an_exact_constraint_meets_only_a_place_naming_its_conversation():
    """exact: account-wide and broad places are no candidates; a place naming the conversation by its primary or
    destination scope is, with that scope's own account and thread; none is a known absence, never unknown."""
    rows = [_scope("account", ""), _scope("broad", "", account=""), _scope("dm", "dm-1", "paused"),
            _scope("by-destination", "C9", destination={"account_id": "", "conversation_id": "p@x"}),
            _scope("other-account", "q@x", "paused", account="B"), _scope("topic", "t@x", "paused", thread="T1"),
            _scope("rfc", "<AbC@host>", "paused")]
    assert _match(rows, "new@x", "exact") == (None, True)
    assert _match(rows, "new@x", "scope")[0].place_id == "account"  # negative control: scope answers the account's place
    assert _match(rows, "dm-1", "exact")[0].place_id == "dm"
    assert _match(rows, "p@x", "exact")[0].place_id == "by-destination"
    assert _match(rows, "q@x", "exact") == (None, True)  # another account's place
    assert _match(rows, "t@x", "exact") == (None, True)  # a thread place is not the conversation's other threads
    assert _match(rows, "t@x", "exact", thread_id=["T1"])[0].place_id == "topic"
    assert _match(rows, "<abc@host>", "exact") == (None, True)  # an opaque id is never folded
    with pytest.raises(ValueError, match="names its conversation"):
        guard.places.target_place(rows, targets={"account_id": ["A"]}, conversation="conversation_id", match="exact")
    with pytest.raises(ValueError, match="match is"):
        _match(rows, "dm-1", "fuzzy")


def test_exact_candidates_rank_by_the_scope_that_matched_never_another_field():
    """An Active place with a detailed primary scope whose destination names R does not hide an equally specific Paused
    place naming R; a more specific matched scope still wins. The scope matcher keeps its previous order."""
    rows = [_scope("active-detailed", "C9", thread="T9", destination={"account_id": "", "conversation_id": "R"}),
            _scope("paused-r", "R", "paused", account="")]
    assert _match(rows, "R", "exact") == (rows[1], True)
    assert _match(rows, "R", "scope")[0].place_id == "active-detailed"  # unchanged: ranked by the place's own fields
    rows.append(_scope("active-a-r", "R"))
    assert _match(rows, "R", "exact")[0].place_id == "active-a-r"


def test_exact_casefold_folds_only_the_conversation():
    """A stored Place spelled ``Bob@Example.invalid`` meets ``bob@example.invalid`` only under the connector's chosen
    ``exact_casefold``; account and thread still compare exactly."""
    rows = [_scope("bob", "Bob@Example.invalid", "paused", account=""), _scope("topic", "Room@X", "paused", thread="T1"),
            _scope("box", "Carol@X", "paused", account="Box")]
    assert _match(rows, "bob@example.invalid", "exact") == (None, True)
    for spelled in ("bob@example.invalid", "BOB@EXAMPLE.INVALID"):
        assert _match(rows, spelled, "exact_casefold")[0].place_id == "bob"
    assert _match(rows, "room@x", "exact_casefold", thread_id=["t1"]) == (None, True)
    assert _match(rows, "room@x", "exact_casefold", thread_id=["T1"])[0].place_id == "topic"
    assert _match(rows, "carol@x", "exact_casefold", account="box") == (None, True)
    assert _match(rows, "carol@x", "exact_casefold", account="Box")[0].place_id == "box"


_RECIPIENTS = [{"conversation": "to", "effect": "write", "match": "exact_casefold"},
               {"conversation": "cc", "effect": "write", "match": "exact_casefold"}]


@pytest.mark.parametrize("account_wide", [True, False], ids=["account-wide", "narrow-only"])
def test_exact_recipient_constraints_hold_only_their_own_address(published_tool, tmp_path, account_wide):
    """A grouped reply whose recipients are its audience and exact constraints: beside a Paused private conversation it
    passes with or without an account-wide place (an unknown recipient is a known absence); a Paused bare address
    among its recipients, stored in another case, holds it in To or Cc. Its own conversation keeps the scope matcher."""
    made, ctx = _store(tmp_path, ("GROUP", "Group", "active"), ("dm-1", "Private", "paused"), ("Peer@X", "Peer", "active"),
                       *([("", "Account", "active")] if account_wide else []))
    name, tool = _grouped(published_tool, target=["account_id", "to", "cc"], conversation="room", touches=_RECIPIENTS)
    args = {"account_id": "A", "room": "GROUP", "to": ["peer@x"], "cc": ["other@x"], "text": "hi"}
    assert guard.dispatch_refusal(ctx, name, args, ext_tool=tool) == ""
    rows = {row["place_label"]: row["via"] for row in guard.call_place_facts(ctx, name, args, ext_tool=tool)["places"]}
    assert rows == {"Group": ["target"], "Peer": ["touches"]}  # the audience is no destination of its own
    guard.places.edit_place(tmp_path, made["Peer"].place_id, "set_state", author="owner", state="paused")
    for recipients in ({}, {"to": ["x@x"], "cc": ["PEER@x"]}):
        assert "Peer is Paused" in guard.dispatch_refusal(ctx, name, {**args, **recipients}, ext_tool=tool)
    refusal = guard.dispatch_refusal(ctx, name, {**args, "room": "elsewhere", "to": ["x@x"], "cc": []}, ext_tool=tool)
    assert ("cannot be attributed while" in refusal) is (not account_wide), refusal


@pytest.mark.parametrize("account_wide", [True, False], ids=["account-wide", "narrow-only"])
def test_a_grouped_call_naming_no_conversation_acts_in_its_account(published_tool, tmp_path, account_wide):
    """A new send naming no conversation is an account-level operation beside its groups: an Active account-wide place
    covers it; inside an account with only narrow places it is unknown, so a Paused place of the connector holds it.
    Its audience is never matched as a destination, and the account it acts in is no place it speaks in."""
    _made, ctx = _store(tmp_path, ("dm-1", "Private", "paused"), ("peer@x", "Peer", "active"),
                        *([("", "Account", "active")] if account_wide else []))
    name, tool = _grouped(published_tool, target=["account_id", "to"], conversation="room", touches=_RECIPIENTS[:1])
    args = {"account_id": "A", "to": ["peer@x"], "text": "hi"}
    refusal = guard.dispatch_refusal(ctx, name, args, ext_tool=tool)
    assert refusal == "" if account_wide else "cannot be attributed while Private is Silent or Paused" in refusal, refusal
    facts = guard.call_place_facts(ctx, name, args, ext_tool=tool)
    assert {row["place_label"]: row["via"] for row in facts["places"]} == (
        {"Account": ["account"], "Peer": ["touches"]} if account_wide else {"Peer": ["touches"]})
    assert facts["touches"] == [{"effect": "write", "argument": "to", "conversation": "peer@x", "account": "A",
                                 "match": "exact_casefold", "place_id": _made["Peer"].place_id}]


def _reply_metadata(made, message, *, source="e1"):
    return {"presence": {"place_id": made["Group"].place_id, "transport_skill": "connector", "event": {
        "source_event_id": source, "conversation_key": "service:A:GROUP", "message": message,
        "destination": {"account_id": "A", "conversation_id": "GROUP"}}}}


def test_the_automatic_replys_addresses_meet_their_places_now(tmp_path, monkeypatch):
    """An automatic reply's selected message names further literal addresses: a Paused place one meets by its match
    holds the reply now; a scope address no place attributes holds it while a place of the connector is Paused;
    without the field (a legacy connector) nothing new applies, and a malformed set never reads as absent."""
    from ouroboros import place_reply

    notices = []
    monkeypatch.setattr(place_reply, "record_notice", lambda root, **row: notices.append(row))
    made, _ctx = _store(tmp_path, ("GROUP", "Group", "active"), ("Peer@X", "Peer", "paused"), ("dm-1", "Private", "active"))
    exact = [{"conversation": "peer@x", "effect": "write", "match": "exact_casefold"},
             {"conversation": "other@x", "effect": "write", "match": "exact"}]
    hold = place_reply.reply_hold(tmp_path, _reply_metadata(made, {"reply_targets": exact}), "t1", "Answer")
    assert hold == "Reply held: Peer is Paused" and notices[-1]["held_text"] == "Answer"
    assert place_reply.reply_hold(tmp_path, _reply_metadata(made, {}), "t1", "Answer") == ""  # legacy: no field
    guard.places.edit_place(tmp_path, made["Peer"].place_id, "set_state", author="owner", state="active")
    assert place_reply.reply_hold(tmp_path, _reply_metadata(made, {"reply_targets": exact}), "t1", "Answer") == ""
    archive = [{"conversation": "Archive", "effect": "write", "match": "scope"}]
    assert place_reply.reply_hold(tmp_path, _reply_metadata(made, {"reply_targets": archive}), "t1", "Answer") == ""
    guard.places.edit_place(tmp_path, made["Private"].place_id, "set_state", author="owner", state="paused")
    assert "cannot be attributed while Private is Silent or Paused" in place_reply.reply_hold(
        tmp_path, _reply_metadata(made, {"reply_targets": archive}), "t1", "Answer")
    assert "cannot be read" in place_reply.reply_hold(
        tmp_path, _reply_metadata(made, {"reply_targets": [{"conversation": "x", "effect": "message", "match": "exact"}]}),
        "t1", "Answer")


def test_the_selected_anchor_names_the_replys_addresses_and_its_digest(tmp_path):
    """The kept anchor, not the waking event, answers after completion; an anchor whose message names none is the
    legacy boundary for that reply. The digest and Safety's subject carry the anchor and full set only when named, so
    a legacy reply's digest keeps its bytes; the Supervisor meets each address's place as a write beside the words."""
    import hashlib
    import json
    from ouroboros import place_reply

    made, ctx = _store(tmp_path, ("GROUP", "Group", "active"), ("peer@x", "Peer", "active"))
    named = [{"conversation": "peer@x", "effect": "write", "match": "exact"}]
    metadata = _reply_metadata(made, {"reply_targets": [{"conversation": "first@x", "effect": "write", "match": "exact"}]})
    metadata["presence_reply_anchor"] = {"source_event_id": "e2", "message": {"reply_targets": named}}
    assert place_reply.reply_addresses(metadata) == ({"source_event_id": "e2", "reply_targets": named}, "")
    assert place_reply.reply_addresses(metadata, {"source_event_id": "e3", "message": {}}) == (None, "")
    carrier = place_reply.reply_carrier(metadata)
    legacy = {"text": "Answer", "destination": carrier["destination"], "place_id": carrier["place_id"]}
    assert place_reply._reply_digest(carrier, "Answer") == hashlib.sha256(
        json.dumps(legacy, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    addresses, _problem = place_reply.reply_addresses(metadata)
    assert place_reply._reply_digest(carrier, "Answer", addresses) != place_reply._reply_digest(carrier, "Answer")
    facts = guard.call_place_facts(ctx, "place_reply", place_reply._reply_subject(carrier, "Answer", addresses))
    assert {row["place_label"]: (row["effects"], row["via"]) for row in facts["places"]} == {
        "Group": (["message"], ["target"]), "Peer": (["write"], ["touches"])}


@pytest.mark.parametrize("targets,problem", [
    (None, ""), ([], ""),
    ([{"conversation": "p@x", "effect": "write", "match": "exact_casefold", "thread": ""}], ""),
    ("p@x", "must be a list"),
    ([{"conversation": "p@x", "effect": "write"}], "holds conversation, effect and match"),
    ([{"conversation": "p@x", "effect": "write", "match": "exact", "account": "B"}], "holds conversation, effect and match"),
    ([{"conversation": "", "effect": "write", "match": "exact"}], "non-empty string id"),
    ([{"conversation": "p@x", "effect": "message", "match": "exact"}], "effect is 'write'"),
    ([{"conversation": "p@x", "effect": "write", "match": "fuzzy"}], "match is scope"),
])
def test_an_events_reply_targets_are_absent_or_whole_at_ingress(targets, problem):
    """``message.reply_targets``: absent is the legacy boundary, ``[]`` names none, anything malformed is refused at the
    one event parser, never read as absent."""
    from ouroboros.presence_events import PresenceEventInvalid, parse_presence_event

    message = {} if targets is None else {"reply_targets": targets}
    payload = {"source_event_id": "e1", "provider": "service", "account_id": "A", "conversation_id": "GROUP",
               "thread_id": "", "conversation_key": "", "actor": {"id": "u1"}, "conversation": {}, "message": message,
               "text": "hi"}
    if not problem:
        assert parse_presence_event(payload).message == message
        return
    with pytest.raises(PresenceEventInvalid) as refused:
        parse_presence_event(payload)
    assert (refused.value.code, refused.value.field) == ("presence_payload_invalid", "event.message.reply_targets")
    assert problem in refused.value.message


def test_two_equally_specific_spellings_hold_when_either_is_paused():
    """Two exact_casefold places of equal precision naming Bob and bob, one Paused: the pause holds whichever comes first
    (change the existing place's state rather than connecting a second spelling). An Active place naming the address in
    one account stays more specific than a global pause: that is the stated cost."""
    rows = [_scope("bob-active", "bob@x"), _scope("Bob-paused", "Bob@x", "paused")]
    for order in (rows, rows[::-1]):
        assert _match(order, "BOB@x", "exact_casefold")[0].place_id == "Bob-paused"
    specific = [_scope("global-paused", "bob@x", "paused", account=""), _scope("account-active", "Bob@X")]
    assert _match(specific, "bob@x", "exact_casefold")[0].place_id == "account-active"


@pytest.mark.parametrize("declared", [True, False], ids=["declared-empty-value", "undeclared-selector"])
def test_an_empty_declared_conversation_is_not_an_absent_selector(published_tool, tmp_path, declared):
    """A row declaring its conversation and given an empty value names no conversation: it acts in its account and its
    other targets are its audience, so a Paused recipient place is no destination of it. A row declaring no
    conversation reads its target values as destinations, as before, and meets that Paused place."""
    _made, ctx = _store(tmp_path, ("", "Account", "active"), ("peer@x", "Peer", "paused"))
    name, tool = _grouped(published_tool, target=["account_id", "to"], **({"conversation": "room"} if declared else {}),
                          touches=[{"conversation": "folder", "effect": "write"}])
    refusal = guard.dispatch_refusal(ctx, name, {"account_id": "A", "room": "", "to": ["peer@x"], "folder": "Drafts",
                                                 "text": "hi"}, ext_tool=tool)
    assert refusal == "" if declared else "Peer is Paused" in refusal, refusal


def test_an_absent_anchor_is_the_waking_event_and_an_unusable_one_holds(tmp_path, monkeypatch):
    """No kept anchor (a promoted or legacy turn's form) reads the waking event's addresses; a kept anchor without its
    source_event_id and message cannot be read and holds the reply, never falling back to another message."""
    from ouroboros import place_reply

    monkeypatch.setattr(place_reply, "record_notice", lambda root, **row: None)
    made, _ctx = _store(tmp_path, ("GROUP", "Group", "active"), ("peer@x", "Peer", "paused"))
    metadata = _reply_metadata(made, {"reply_targets": [{"conversation": "peer@x", "effect": "write", "match": "exact"}]})
    for anchor in (None, {}):
        assert place_reply.reply_addresses(metadata, anchor)[0]["source_event_id"] == "e1"
    assert place_reply.reply_hold(tmp_path, metadata, "t1", "Answer") == "Reply held: Peer is Paused"
    for unusable in ({"source_event_id": "e2"}, {"source_event_id": "e2", "message": "x"}, {"message": {}}, ["e2"]):
        metadata["presence_reply_anchor"] = unusable
        addresses, problem = place_reply.reply_addresses(metadata)
        assert addresses is None and "anchor" in problem, unusable
        assert "cannot be read" in place_reply.reply_hold(tmp_path, metadata, "t1", "Answer"), unusable


def test_reply_addresses_live_in_the_replys_own_route_never_the_source_account(tmp_path, monkeypatch):
    """The reply's further addresses take its verified route, the retained destination it goes to (here the place's
    destination in account D), never the source event's account S: a Paused address place in S holds nothing, one in D
    holds the reply."""
    from ouroboros import place_reply

    monkeypatch.setattr(place_reply, "record_notice", lambda root, **row: None)
    route = {"transport": "service", "account_id": "D", "conversation_id": "OUT", "thread_id": ""}
    guard.places.connect_place(tmp_path, connector="connector", provider="service", account_id="S",
                               conversation_id="SRC", label="Source", place_id="source")
    guard.places._write(tmp_path, "source", lambda old: replace(old, destination=dict(route)))
    guard.places.connect_place(tmp_path, connector="connector", provider="service", account_id="S",
                               conversation_id="peer@x", label="Peer in S", state="paused", place_id="peer-s")
    metadata = {"presence": {"place_id": "source", "transport_skill": "connector", "event": {
        "source_event_id": "e1", "account_id": "S", "conversation_key": "service:S:SRC", "destination": route,
        "message": {"reply_targets": [{"conversation": "peer@x", "effect": "write", "match": "exact"}]}}}}
    assert place_reply.reply_hold(tmp_path, metadata, "t1", "Answer") == ""
    guard.places.connect_place(tmp_path, connector="connector", provider="service", account_id="D",
                               conversation_id="peer@x", label="Peer in D", state="paused", place_id="peer-d")
    assert place_reply.reply_hold(tmp_path, metadata, "t1", "Answer") == "Reply held: Peer in D is Paused"


@pytest.mark.parametrize("match,paused_spelling", [("exact", "bob@x"), ("exact_casefold", "Bob@X")])
@pytest.mark.parametrize("wildcard", ["", "*"])
@pytest.mark.parametrize("via", ["own-scope", "destination"])
def test_a_wildcard_account_names_no_account_in_the_exact_matcher(match, paused_spelling, wildcard, via):
    """``*``, like ``""``, names no account: a global Paused place naming the address, by its own scope or by its
    destination, stays less specific than an Active place naming it in account A (the stated cost)."""
    paused = (_scope("global-paused", paused_spelling, "paused", account=wildcard) if via == "own-scope" else
              _scope("global-paused", "C9", "paused", account="B",
                     destination={"account_id": wildcard, "conversation_id": paused_spelling}))
    rows = [paused, _scope("a-active", "bob@x")]
    for order in (rows, rows[::-1]):
        assert _match(order, "bob@x", match)[0].place_id == "a-active"


def test_the_scope_matcher_still_ranks_a_wildcard_account_as_before():
    """Negative control: only the exact matcher reads ``*`` as unnamed; the scope matcher's order is untouched."""
    rows = [_scope("global-paused", "bob@x", "paused", account="*"), _scope("a-active", "bob@x")]
    assert _match(rows, "bob@x", "scope")[0].place_id == "global-paused"


@pytest.mark.parametrize("blank", ["", "*"])
def test_a_blank_reply_route_keeps_its_further_addresses_account_blank(tmp_path, blank):
    """The reply's retained destination names no account: a further address's global place reaches the Supervisor with
    its notes and no account, never the source event's account S that ordinary dispatch would infer."""
    from ouroboros import place_reply

    route = {"transport": "service", "account_id": blank, "conversation_id": "OUT", "thread_id": ""}
    guard.places.connect_place(tmp_path, connector="connector", provider="service", account_id="S",
                               conversation_id="SRC", label="Source", place_id="source")
    guard.places._write(tmp_path, "source", lambda old: replace(old, destination=dict(route)))
    guard.places.connect_place(tmp_path, connector="connector", provider="service", account_id="", conversation_id="peer@x",
                               label="Peer everywhere", place_id="peer", first={"note": "Peer prefers short notes."})
    metadata = {"presence": {"place_id": "source", "transport_skill": "connector", "event": {
        "source_event_id": "e1", "account_id": "S", "conversation_key": "service:S:SRC", "destination": route,
        "message": {"reply_targets": [{"conversation": "peer@x", "effect": "write", "match": "exact"}]}}}}
    ctx = SimpleNamespace(drive_root=tmp_path, task_id="caller", task_metadata=metadata)
    addresses, _problem = place_reply.reply_addresses(metadata)
    subject = place_reply._reply_subject(place_reply.reply_carrier(metadata), "Answer", addresses)
    facts = guard.call_place_facts(ctx, "place_reply", subject)
    peer = next(row for row in facts["places"] if row["place_id"] == "peer")
    assert "account_id" not in peer and peer["effects"] == ["write"], peer
    assert [touch.get("account") for touch in facts["touches"]] == [None]
    notes = guard.place_evidence(ctx, {}, {row["place_id"] for row in facts["places"]}, "place_reply", subject)
    assert any(note["text"] == "Peer prefers short notes." for note in notes), notes


@pytest.mark.parametrize("account", ["", "*", "S", "D"])
@pytest.mark.parametrize("provider", ["service", "other-service"])
def test_reply_main_facts_use_the_retained_route_not_the_source(tmp_path, monkeypatch, account, provider):
    """The actual key and sender follow the retained route; source scope/notes remain source facts.
    An unknown sender must not pick the last arrival from an arbitrary account."""
    from ouroboros import place_reply, place_arrivals
    from ouroboros.presence_bindings import conversation_key

    source = guard.places.connect_place(tmp_path, connector="connector", provider="service", account_id="S",
                                        conversation_id="SRC", label="Source", place_id="source",
                                        first={"note": "Source guidance remains."})
    identity = {"sends_as": {"kind": "user", "id": "source-user"}}
    destination_identity = {"sends_as": {"kind": "bot", "id": "destination-bot"}}
    reads = []
    monkeypatch.setattr(place_arrivals, "recent_identity",
                        lambda root, place_id, selected: reads.append(selected) or destination_identity)
    route = {"transport": provider, "account_id": account, "conversation_id": "OUT", "thread_id": ""}
    guard.places._write(tmp_path, source.place_id, lambda old: replace(old, thread_id="source-thread", destination=route))
    metadata = {"focus": {"place_id": source.place_id, "identity": identity, "identity_owner": True},
                "presence": {"place_id": source.place_id, "transport_skill": "connector", "event": {
                    "source_event_id": "e1", "account_id": "S", "conversation_key": "service:S:SRC:source-thread",
                    "destination": route, "message": {}}}}
    ctx = SimpleNamespace(drive_root=tmp_path, task_id="caller", task_metadata=metadata)
    subject = place_reply._reply_subject(place_reply.reply_carrier(metadata), "Answer")
    facts = guard.call_place_facts(ctx, "place_reply", subject)
    row = _only(facts)
    actual_account = "" if account == "*" else account
    assert row.get("account_id", "") == actual_account
    assert row["conversation_keys"] == [conversation_key(provider, actual_account, "OUT", "")]
    assert "this_turn_sends" not in facts
    if actual_account and provider == "service":
        assert row["sends_as"] == (identity if account == "S" else destination_identity)["sends_as"]
        assert row["sends_as_owner"] is (account == "S")
        assert reads == ([] if account == "S" else ["D"])
    else:
        assert "sends_as" not in row and "sends_as_owner" not in row and not reads
    assert (row["conversation_id"], row["thread_id"]) == ("SRC", "source-thread")  # the Place's scope
    notes = guard.place_evidence(ctx, {}, {source.place_id}, "place_reply", subject)
    assert any(note["text"] == "Source guidance remains." for note in notes)


def test_reply_without_a_route_keeps_legacy_scope_but_an_empty_route_infers_nothing(places):
    _, ctx = places
    legacy = {"place_id": "active", "text": "Answer"}
    assert _only(guard.call_place_facts(ctx, "place_reply", legacy))["conversation_key"] == "service:account:C2:0"
    empty = _only(guard.call_place_facts(ctx, "place_reply", {**legacy, "destination": {}}))
    assert empty["conversation_keys"] == [] and empty["conversation_key"] == ""
    assert "account_id" not in empty and "sends_as" not in empty
