"""Every dispatch guard refuses its target case and permits its working neighbor."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import place_guard as guard
from ouroboros.places import Place, NoteEntry
from ouroboros.tools.registry_guards import _capability_resource_guard_result
from tests.test_tool_effects import published_tool as _published_tool

published_tool = _published_tool


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

    def resolve(root, *, connector, targets, conversation="", thread=""):
        values = [v for values in targets.values() for v in values]
        match = next((p for p in rows if p.conversation_id in values), None)
        return match, match is not None

    monkeypatch.setattr(guard.places, "place_for_target", resolve)
    monkeypatch.setattr(guard.conversation_lease, "live_author", lambda *a, **k: "")
    ctx = SimpleNamespace(drive_root=tmp_path, task_id="caller", task_metadata={})
    return rows, ctx


def test_advises_extension_and_mcp_both_directions(published_tool, places):
    _, ctx = places
    ctx.task_metadata = {"initiator": "consciousness", "consciousness_autonomy": "observe"}
    for effect in ("write", "message", None):
        name, tool = published_tool(effect)
        assert "Advises" in guard.dispatch_refusal(ctx, name, {}, ext_tool=tool)
    name, tool = published_tool("read")
    assert guard.dispatch_refusal(ctx, name, {}, ext_tool=tool) == ""
    assert "Advises" in guard.dispatch_refusal(ctx, "mcp_demo__read", {}, is_mcp=True)
    ctx.task_metadata["consciousness_autonomy"] = "act"
    assert guard.dispatch_refusal(ctx, "mcp_demo__read", {}, is_mcp=True) == ""
    assert guard.dispatch_refusal(ctx, "read_file", {}) == ""


@pytest.mark.parametrize("metadata", [{}, {"task_type": "task"}, {"schedule_id": "s"},
    {"initiator": "consciousness", "consciousness_autonomy": "full", "wake_cause": "focus"}])
@pytest.mark.parametrize("state", ["silent", "paused"])
def test_place_state_from_every_origin_and_active_sibling(published_tool, places, metadata, state):
    rows, ctx = places
    rows[0] = replace(rows[0], state=state)
    ctx.task_metadata = metadata
    for effect in ("message", "write"):
        name, tool = published_tool(effect, target="room", conversation="room")
        refusal = guard.dispatch_refusal(ctx, name, {"room": "C1"}, ext_tool=tool)
        assert refusal.startswith("⚠️ RESOURCE_CONSTRAINT_BLOCKED:")
        assert state.title() in refusal and "Consciousness › Places" in refusal
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


@pytest.mark.parametrize("broad_account", ["", "*"])
@pytest.mark.parametrize("account_source", ["focus", "call"])
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
        assert "need repair" in refusal and "Consciousness › Places" in refusal
    else:
        assert refusal == ""
    name, tool = published_tool("read", target="channel")
    assert guard.dispatch_refusal(ctx, name, {"channel": "C1"}, ext_tool=tool) == ""
    assert guard.call_place_facts(ctx, name, {"channel": "C1"}, ext_tool=tool).get("places_unreadable", "") == (
        "Place records need repair" if declares_place else "")


@pytest.mark.parametrize("kind,account,owner", [("user", "account", True), ("user", "other", False),
    ("bot", "account", False), (None, "account", False)])
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
