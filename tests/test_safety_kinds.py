"""Host provenance, policy coverage and attributed fixed evidence for both levels."""
import pytest

from ouroboros import tool_effects as _tool_effects
from tests._safety_fixtures import supervisor as supervisor_fixture
supervisor = supervisor_fixture
from ouroboros.consciousness_authority import work_kind, consciousness_origin_metadata
REAL_EFFECT_OF = _tool_effects.effect_of  # bound before the fixture stubs it


@pytest.mark.parametrize("metadata,kind", [
    ({}, "consciousness"), ({"work_kind": "owner", "initiator": "consciousness"}, "owner"),
    ({"project_id": "project-room"}, "consciousness"),
])
def test_fallback_order_and_child_copy(metadata, kind):
    assert work_kind(metadata) == kind
    assert consciousness_origin_metadata(metadata)["work_kind"] == kind


@pytest.mark.parametrize("basis,kind", [
    ({"source": "place_owner", "content": "Look into it", "basis": "cited", "carried": True}, "owner"),
    (None, "consciousness"),
    ({"source": "place_owner", "content": "Look into it"}, "consciousness"),  # no verified basis: not a cite
])
def test_only_a_verified_cite_mints_an_independent_owner_root(basis, kind):
    """No host flag mints owner kind: a confirmed account's event, an old ``addressed`` key or a ``work_kind``
    stamp on the focus leave a started root the mind's own work; the host-verified cite alone makes it the owner's."""
    metadata = {"initiator": "consciousness", "wake_cause": "focus", "task_id": "focus-turn",
                "focus": {"addressed": True, "actor": {"owner_confirmed": True}}}
    assert work_kind(metadata) == "consciousness"
    assert consciousness_origin_metadata(metadata)["work_kind"] == "consciousness"
    child = consciousness_origin_metadata(metadata, started_root=True, owner_basis=basis)
    assert child["work_kind"] == kind and child["origin_wake_task_id"] == "focus-turn"
    if kind == "consciousness":
        assert child["consciousness_cause"] == "started"
    else:
        assert "consciousness_cause" not in child and "initiator" not in child
    # the focus itself never changes kind, and a child copy cannot write back into it
    assert consciousness_origin_metadata(metadata, owner_basis=basis)["work_kind"] == "consciousness"
    child["focus"]["actor"]["owner_confirmed"] = "changed"
    assert metadata["focus"]["actor"]["owner_confirmed"] is True


def test_caller_kind_selects_level_for_steering_and_root_minting(supervisor):
    h = supervisor
    h.modes.update(owner="off", consciousness="full")
    checked = ["resume_task", "continue_task", "pause_task", "start_task",
               "schedule_followup", "toggle_consciousness", "configure_presence"]
    for name in checked:
        h.module.check_safety(name, {}, ctx=h.ctx)
    assert not h.calls
    h.ctx.task_metadata["work_kind"] = "consciousness"
    for name in checked:
        h.module.check_safety(name, {}, ctx=h.ctx)
    assert len(h.calls) == len(checked)
    # Relays between my own works leave the check: the send they lead to is checked where it happens.
    for name in ["tell_task", "forward_to_worker", "schedule_subagent", "delegate_start", "delegate_message"]:
        h.module.check_safety(name, {}, ctx=h.ctx)
    h.module.check_safety("manage_schedules", {"action": "list"}, ctx=h.ctx)
    assert len(h.calls) == len(checked)
    h.module.check_safety("manage_schedules", {"action": "restore"}, ctx=h.ctx)
    assert len(h.calls) == len(checked) + 1


@pytest.mark.parametrize("retired", ["presence_finish", "initiate_presence"])
def test_retired_presence_names_have_no_policy_bypass(supervisor, retired):
    h = supervisor
    assert retired not in h.module.TOOL_POLICY
    h.module.check_safety("presence_cancel_work", {}, ctx=h.ctx)
    assert not h.calls  # the surviving own-work control still skips
    h.module.check_safety(retired, {}, ctx=h.ctx)
    assert len(h.calls) == 1  # unknown/reused names get the ordinary default check


def test_only_declared_reads_skip(supervisor, monkeypatch):
    h = supervisor
    from ouroboros import tool_effects
    for declared in [False, True]:
        monkeypatch.setattr(tool_effects, "effect_of", lambda *a, **kw: tool_effects.ToolEffect("read", declared=declared))
        h.module.check_safety("extension_read", {}, ctx=h.ctx)
    assert len(h.calls) == 1
    assert h.events[-1]["type"] == "safety_declared_read_skip"


def test_an_mcp_read_only_hint_skips_and_any_other_mcp_call_is_checked(supervisor, monkeypatch):
    """A configured server's ``readOnlyHint: true`` skips the check like a manifest read (disclosed widening)."""
    h = supervisor
    from ouroboros import tool_effects
    monkeypatch.setattr(h.module, "get_runtime_mode", lambda: "cyber_pro")  # holds and flags would be advice anyway
    monkeypatch.setattr(tool_effects, "effect_of", REAL_EFFECT_OF)
    monkeypatch.setattr(tool_effects, "_MCP_ANNOTATIONS", {})
    tool_effects.record_mcp_annotations("slack", [{"name": "history", "annotations": {"readOnlyHint": True}},
                                                  {"name": "post", "annotations": {"readOnlyHint": False}}])
    assert h.module.check_safety("mcp_slack__history", {"channel": "C1"}, ctx=h.ctx) == (True, "")
    assert not h.calls and h.events[-1] == {"type": "safety_declared_read_skip", "tool": "mcp_slack__history",
                                            "source": "mcp", "reason": "server annotation readOnlyHint"}
    assert h.module.check_safety("mcp_slack__post", {"channel": "C1"}, ctx=h.ctx)[0]
    assert len(h.calls) == 1


def test_bundle_authorship_scope_and_no_reasoning_dump(supervisor, monkeypatch):
    h = supervisor
    from ouroboros import places, place_guard
    monkeypatch.setattr(place_guard, "call_place_facts", lambda *a, **kw: {"places": [{
        "place_id": "room", "place_label": "Friends", "place_state": "active"}], "targets": {"to": ["audience"]}})
    room = places.Place("room", "connector", "chat", label="Friends",
                        notes=(places.NoteEntry("owner permission", "owner"), places.NoteEntry("old permission", "migration"),
                               places.NoteEntry("mind idea", "ouroboros")))
    monkeypatch.setattr(places, "place_records", lambda root: places.PlaceRecords(0, {"room": room}, (), {}))
    h.values["OUROBOROS_SAFETY_OWNER_RULES"] = "owner standing rule"
    h.module.check_safety("remote_write", {"text": "payload"}, messages=[{"role": "assistant", "content": "private reasoning"},
        {"role": "tool", "content": "huge tool output"}], ctx=h.ctx)
    prompt = h.calls[0]["user_prompt"]
    for text in ["owner standing rule", "owner permission", "old permission", "mind idea", "Friends", "audience", "your work"]:
        assert text in prompt
    assert '"author": "ouroboros",\n        "standing_delegation": false' in prompt
    for text in ["private reasoning", "huge tool output", "MUST NOT BE INCLUDED"]:
        assert text not in prompt
