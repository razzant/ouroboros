"""Incident 062f199a replay: a wake's start beside my story, its own room and my other conversations."""
import dataclasses
import json

from ouroboros import context_fit as cf, memory_view as mv, capability_evidence as ce
from ouroboros.context_view_facts import low_start_line
from ouroboros.tool_result_delivery import project_tool_result_batch
from tests import _memory_view_synthetic as syn


# The incident's start carried the whole Architecture book (1.44M characters) beside its maps. Since 26.11.2026 the
# start carries one resident Body Map of the real map's size (about 14K characters) with its parts-and-addresses form.
BODY_MAP, BODY_MAP_PARTS = "## Body map\n" + "b" * 14_000, "## Body map (parts and addresses)\n" + "p" * 3_000


def _incident(tmp_path, monkeypatch, *, replies=100):
    task = {"id": "incident01", "usage_category": "consciousness", "chat_id": 1}
    spec = mv.view_spec_for_task(task, tmp_path)
    rooms = [syn.conversation(i, words=1) for i in range(164)]
    snap = dataclasses.replace(syn.snapshot(role="consciousness", story=syn.pages(103, 200),
        room_facts=syn.room(lane1=[syn.spoken(i, "ouroboros", 8_000) for i in range(replies)], lane2=359),
        conversations=rooms), spec=spec)
    core = cf.ContextCore("Required core", "Constitution", BODY_MAP, "", "Identity",
        "f" * 500_000, '"Assignment"', body_map_parts=BODY_MAP_PARTS, memory_view_json=mv.snapshot_json(snap),
        section_root=str(tmp_path), section_task_id=task["id"])
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: 1.0581)
    plan = cf.build_context_fit_plan(None, core, task, preferred_mode="max", route_resolver=lambda *_a, **_kw: (
        {"provider": "test", "model": "test"}, ce.CapabilityEvidence(872_000, "confirmed", "metadata", "incident-route")))
    return plan


def test_incident_start_holds_max_and_keeps_memory_whole_beside_the_body_map(tmp_path, monkeypatch):
    plan = _incident(tmp_path, monkeypatch)
    shown = plan.projection(plan.initial_mode)
    steps = shown.memory_facts["floor"]["steps"]
    print("INCIDENT_START", json.dumps({"floor": steps, "input": shown.calibrated_tokens, "mode": plan.initial_mode}))
    # These old replies have no recorded gist. The incident's start was Low only because the whole books did not
    # fit beside them; with the resident Body Map it is an honest Max that holds the story and all replies whole.
    assert plan.initial_mode == "max" and steps == {}
    assert shown.memory_facts["max_room"]["cause"] == "held" and shown.memory_facts["max_room"]["need"] <= 872_000
    assert not shown.memory_facts["floor"].get("mode_switch") and not low_start_line(plan, plan.initial_mode, "incident01")
    # The start leaves the reply reserve and the wake's own work room: one eighth since the owner's 29.10 decision
    # (a wake is one turn, as a direct chat turn is), read from the view so the replay runs on any revision.
    work = -(-872_000 // 8) * shown.memory_facts["spec"]["work_room_levels"]
    assert shown.calibrated_tokens + 65_536 + work <= 872_000


def test_an_overfull_incident_start_runs_in_the_named_temporary_low_with_the_map_at_its_parts(tmp_path, monkeypatch):
    # The same wake with more of my own ungisted replies in its room than the window holds beside the Map and
    # the work's room: the genuine pressure the books once stood in for.
    plan = _incident(tmp_path, monkeypatch, replies=300)
    shown = plan.projection(plan.initial_mode)
    facts = shown.memory_facts
    note = shown.system_content_json
    print("INCIDENT_OVERFULL", json.dumps({"floor": facts["floor"]["steps"], "mode": plan.initial_mode,
                                           "switch": facts.get("mode_switch"), "addressed": facts.get("host_sections_addressed")}))
    assert plan.initial_mode == "low" and facts["floor"]["mode_switch"] == {"from": "max", "to": "low"}
    assert facts["max_room"]["cause"] == "crowded" and facts["max_room"]["telling_need"] > 872_000
    assert "Low, temporarily, for the window; my human's mode is Max. The body map, my memory at the gate's level" in note
    assert "Only the window cuts my memory here; Low's own target does not bind. The host's measure of this start:" in note
    assert low_start_line(plan, "low", "incident01") == (
        "Task incident01 runs in Low for now, not in your Max: my memory did not fit the window beside the work's room "
        "even in its shortest tellings. Your setting stays Max.")
    # The Map stands at its parts and addresses before any of my memory goes below a telling.
    assert any(row.get("section_id") == "body_map" for row in facts["host_sections_addressed"])
    sent = json.dumps(plan.messages_for("low"))
    assert BODY_MAP_PARTS.split("\n", 1)[1] in sent and BODY_MAP.split("\n", 1)[1] not in sent
    assert shown.calibrated_tokens + 65_536 <= 872_000
    # Only the window cuts: Low's own target does not bind under the owner's Max, so what goes below a telling here is
    # what my tellings beside the Map's parts could not hold (telling_need above the window, asserted above).
    assert facts["floor"]["target_tokens"] is None and facts["floor"]["budget_allowance_tokens"] is None


def test_incident_chosen_range_survives_zero_result_room(tmp_path, monkeypatch):
    plan = _incident(tmp_path, monkeypatch)
    body = "Requested range\n" + "r" * 2_567 + "\nDecisive end"
    messages = plan.messages_for(plan.initial_mode)
    messages.append({"role": "assistant", "content": "Working material " * 150_000})
    def measure(rows, schemas):
        return dataclasses.asdict(cf.measure_main_fit(plan, rows, schemas, profile="owner_max",
            rendered_mode=plan.initial_mode, round_id="1").measurement)
    results, receipt = project_tool_result_batch([{"tool_call_id": "read", "fn_name": "read_file",
        "tool_args": {"start_char": 1000, "max_chars": 2_567}, "result": body}], messages, [], drive_root=tmp_path,
        task_id="incident01", fit_candidate=measure, policy="measured_frame")
    print("INCIDENT_READ", json.dumps({"partial": bool(results[0].get("result_partial")),
        "ranges": results[0].get("result_source_view", {}).get("shown_ranges"),
        "recovery_required": receipt.get("recovery_required")}))
    assert results[0]["result"] == body and not results[0].get("result_partial")
    assert receipt["recovery_required"]
