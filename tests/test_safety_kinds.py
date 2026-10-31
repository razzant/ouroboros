"""Host provenance, policy coverage and attributed fixed evidence for both levels."""
import pytest

from tests._safety_fixtures import supervisor as supervisor_fixture
supervisor = supervisor_fixture
from ouroboros.consciousness_authority import work_kind, consciousness_origin_metadata


@pytest.mark.parametrize("metadata,kind", [
    ({}, "consciousness"), ({"work_kind": "owner", "initiator": "consciousness"}, "owner"),
    ({"initiator": "consciousness", "origin_message_ref": "x"}, "consciousness"),
    ({"presence_binding_authority": {"id": "old"}}, "consciousness"),
    ({"presence_binding_authority": {}, "origin_message_ref": "owner"}, "consciousness"),
    ({"presence_root_carrier": True}, "consciousness"), ({"presence": True}, "consciousness"),
    ({"origin_message_ref": "owner-message"}, "owner"), ({"origin_suppressed": True}, "owner"),
    ({"owner_ingress": True, "project_id": "project-room"}, "owner"),
    ({"project_id": "project-room"}, "consciousness"),
])
def test_fallback_order_and_child_copy(metadata, kind):
    assert work_kind(metadata) == kind
    assert consciousness_origin_metadata(metadata)["work_kind"] == kind


@pytest.mark.parametrize("addressed,confirmed,kind", [(True, True, "owner"), (True, False, "consciousness"),
                                                      (False, True, "consciousness"), (False, False, "consciousness")])
def test_confirmed_addressed_request_mints_only_an_independent_owner_root(addressed, confirmed, kind):
    metadata = {"initiator": "consciousness", "wake_cause": "focus", "task_id": "focus-turn",
                "focus": {"addressed": addressed, "actor": {"owner_confirmed": confirmed}}}
    assert work_kind(metadata) == "consciousness"
    assert consciousness_origin_metadata(metadata)["work_kind"] == "consciousness"
    child = consciousness_origin_metadata(metadata, started_root=True)
    assert child["work_kind"] == kind and child["origin_wake_task_id"] == "focus-turn"
    if kind == "consciousness":
        assert child["consciousness_cause"] == "started"
    child["focus"]["actor"]["owner_confirmed"] = "changed"
    assert metadata["focus"]["actor"]["owner_confirmed"] is confirmed


def test_caller_kind_selects_level_for_steering_and_root_minting(supervisor):
    h = supervisor
    h.modes.update(owner="off", consciousness="full")
    checked = ["forward_to_worker", "tell_task", "resume_task", "continue_task", "pause_task", "start_task",
               "schedule_followup", "toggle_consciousness", "configure_presence"]
    for name in checked:
        h.module.check_safety(name, {}, ctx=h.ctx)
    assert not h.calls
    h.ctx.task_metadata["work_kind"] = "consciousness"
    for name in checked:
        h.module.check_safety(name, {}, ctx=h.ctx)
    assert len(h.calls) == len(checked)
    for name in ["schedule_subagent", "delegate_start", "delegate_message"]:
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


def test_bundle_authorship_scope_and_no_reasoning_dump(supervisor, monkeypatch):
    h = supervisor
    from ouroboros import places, place_guard
    monkeypatch.setattr(place_guard, "call_place_facts", lambda *a, **kw: {"places": [{
        "place_id": "room", "place_label": "Friends", "place_state": "active"}], "targets": {"to": ["audience"]}})
    monkeypatch.setattr(places, "load_places", lambda root: [places.Place("room", "connector", "chat", label="Friends",
        notes=(places.NoteEntry("owner permission", "owner"), places.NoteEntry("old permission", "migration"),
               places.NoteEntry("mind idea", "ouroboros")))])
    h.values["OUROBOROS_SAFETY_OWNER_RULES"] = "owner standing rule"
    h.module.check_safety("remote_write", {"text": "payload"}, messages=[{"role": "assistant", "content": "private reasoning"},
        {"role": "tool", "content": "huge tool output"}], ctx=h.ctx)
    prompt = h.calls[0]["user_prompt"]
    for text in ["owner standing rule", "owner permission", "old permission", "mind idea", "Friends", "audience", "your work"]:
        assert text in prompt
    assert '"author": "ouroboros",\n        "standing_delegation": false' in prompt
    for text in ["private reasoning", "huge tool output", "MUST NOT BE INCLUDED"]:
        assert text not in prompt
