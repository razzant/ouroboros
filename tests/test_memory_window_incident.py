"""Incident 062f199a replay, also runnable unchanged against the base revision."""
import dataclasses
import json

from ouroboros import context_fit as cf, memory_view as mv, capability_evidence as ce
from ouroboros.tool_result_delivery import project_tool_result_batch
from tests import _memory_view_synthetic as syn


def _incident(tmp_path, monkeypatch):
    task = {"id": "incident01", "usage_category": "consciousness", "chat_id": 1}
    spec = mv.view_spec_for_task(task, tmp_path)
    rooms = [syn.live_room(i, words=1) for i in range(164)]
    if spec.live_rooms == "directory":
        rooms = [{**room, "directory": True, "words": [], "notes": []} for room in rooms]
    snap = dataclasses.replace(syn.snapshot(role="consciousness", story=syn.pages(103, 200),
        room_facts=syn.room(lane1=[syn.spoken(i, "ouroboros", 8_000) for i in range(100)], lane2=359),
        live=rooms), spec=spec)
    core = cf.ContextCore("Required core", "Constitution", "b" * 1_440_000, "", "Identity",
        "f" * 500_000, '"Assignment"', False, memory_view_json=mv.snapshot_json(snap))
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: 1.0581)
    plan = cf.build_context_fit_plan(None, core, task, preferred_mode="max", route_resolver=lambda *_a, **_kw: (
        {"provider": "test", "model": "test"}, ce.CapabilityEvidence(872_000, "confirmed", "metadata", "incident-route")))
    return plan


def test_incident_start_keeps_story_before_old_replies(tmp_path, monkeypatch):
    plan = _incident(tmp_path, monkeypatch)
    shown = plan.projection(plan.initial_mode)
    steps = shown.memory_facts["floor"]["steps"]
    print("INCIDENT_START", json.dumps({"floor": steps, "input": shown.calibrated_tokens, "mode": plan.initial_mode}))
    assert steps.get("F2", 0) > 0 and steps.get("F5", 0) < 103
    assert steps != {"F1": 359, "F5": 103}
    # The start leaves the reply reserve and the wake's own work room: one eighth since the owner's 29.10 decision
    # (a wake is one turn, as a direct chat turn is), read from the view so the replay runs on any revision.
    work = -(-872_000 // 8) * shown.memory_facts["spec"]["work_room_levels"]
    assert shown.calibrated_tokens + 65_536 + work <= 872_000


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
