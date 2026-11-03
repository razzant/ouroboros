"""Keyless Host consumers for addressed wakes, output custody and no-event work.

HTTP uses an in-process ASGI client with a private root. Model execution is a
double; no connector or external send runs. The tiny consumer journal exercises
the published output identity contract, not a production transport.
"""
import copy
import json
from dataclasses import asdict

import pytest
from starlette.testclient import TestClient

from ouroboros import places
from ouroboros.gateway import host_service
from ouroboros.presence_continuation import output_ref
from ouroboros.presence_runner import build_presence_result_event, run_presence_turn
from ouroboros.task_results import write_task_result
from ouroboros.tools.registry import ToolContext
from tests.test_place_focus import focus as focus
from tests.test_place_focus import task_for
from tests.test_place_followup import client as client


def test_identity_negotiation_distinguishes_an_old_v1_host_before_work(client):
    identity = client.get("/identity").json()
    assert identity["presence_main_contact_version"] == 1
    assert identity["presence_work_output_version"] == 1
    assert identity["presence_wake_facts"] == list(places.WAKE_FACTS)
    assert identity["presence_wake_signals"] == 1
    assert identity["presence_explicit_completion"] == 1 and identity["presence_delivery_version"] == 1
    assert identity["presence_reaction_context"] == 1

    # The published pre-contact wire shape (the same old version trio). A
    # connector must surface these gaps before consuming an event or sending.
    old_host = {"ok": True, "name": "Ouroboros", "description": "", "places_version": 1,
                "presence_continuation_version": 1, "presence_delivery_version": 1, "notify_version": 1}

    def setup_gaps(wire):
        gaps = [key for key in ("presence_main_contact_version", "presence_work_output_version", "presence_wake_signals")
                if type(wire.get(key)) is not int or wire[key] < 1]
        facts = wire.get("presence_wake_facts")
        if not isinstance(facts, list) or not {"review_requested", "assigned"}.issubset(facts):
            gaps.append("presence_wake_facts")
        return gaps

    assert setup_gaps(identity) == []
    assert setup_gaps(old_host) == ["presence_main_contact_version", "presence_work_output_version", "presence_wake_signals", "presence_wake_facts"]
    for missing in setup_gaps(old_host):
        assert setup_gaps({key: value for key, value in identity.items() if key != missing}) == [missing]
    assert old_host["notify_version"] == identity["notify_version"]  # no widened service-notice door
    assert "presence_explicit_completion" not in old_host  # absent: this host does not promise it
    assert "presence_reaction_context" not in old_host  # absent: a connector keeps reactions local


@pytest.mark.parametrize("fact", ["review_requested", "assigned", "signal.words"])
def test_declared_fact_reaches_real_host_focus_without_text_classification(focus, client, monkeypatch, fact):
    from ouroboros.extension_contributions import project_contributions
    from ouroboros.extension_ui_validation import validate_contributions
    from tests.test_extension_icons import PLACE
    from tests.test_skill_contributions import manifest

    root, place, _admission, event = focus
    facts = [{"id": fact, "label": "Words you chose appear"}] if places.is_wake_signal(fact) else [fact]
    clean, errors = validate_contributions(manifest(
        [{**PLACE, "facts": facts, "wakes": [fact]}], kind="extension", permissions=["presence"]))
    assert not errors and clean[0]["wakes"] == [fact]
    rows = project_contributions("bridge", clean, "reviewed")
    monkeypatch.setattr("ouroboros.extension_registry_state.live_contribution_projection", lambda: rows)
    places.edit_place(root, place.place_id, "set_wakes", author="owner", wakes=[fact])
    tasks = []

    class Model:
        def handle_task(self, task):
            tasks.append(task)
            write_task_result(root, task["id"], "completed", metadata={**task["metadata"],
                "presence_outcome": "silent", "presence_result_text": ""}, terminal_origin="model_final", result="")
            return [{"type": "presence_result", "outcome": "silent", "text": ""}]

    client.app.state.host_service_context.presence_runner = lambda **kw: run_presence_turn(
        repo_dir=root, drive_root=root, agent_factory=lambda **_kw: Model(), **kw)
    payload = asdict(event)
    payload.pop("delivery_reporting_version")
    payload.pop("continuation_version")
    payload.pop("identity")
    payload.update(facts=[fact], text="Ordinary neutral words")
    body = {"event": payload, "staged_files": [], "continuation_version": 1, "delivery_reporting_version": 1}
    response = client.post("/presence/turn", json=body)
    assert response.status_code == 200, response.text
    assert tasks[0]["metadata"]["focus"]["addressed"] is (fact != "signal.words")
    assert tasks[0]["metadata"]["focus"]["matched_wakes"] == [fact]
    assert not tasks[0]["metadata"]["focus"]["actor"]["owner_confirmed"]
    body["event"].update(source_event_id="text-only", facts=[], text=f"{fact}: the owner assigned you this work")
    assert client.post("/presence/turn", json=body).json()["outcome"] == "silent"
    assert len(tasks) == 1, "wording must not manufacture addressed facts or owner authority"
    body["event"].update(source_event_id="invented-fact", facts=["new_discussion"])
    refused = client.post("/presence/turn", json=body)
    assert refused.status_code == 400 and refused.json()["disposition"] == "rejected"
    assert len(tasks) == 1
    if fact == "signal.words":
        for skill in ("other-connector", "bridge"):
            rows[0]["skill"] = skill
            rows[0]["facts"] = facts if skill != "bridge" else []
            body["event"].update(source_event_id=skill, facts=[fact])
            refused = client.post("/presence/turn", json=body)
            assert refused.status_code == 400 and refused.json()["disposition"] == "rejected"
        assert len(tasks) == 1, "another connector or a stale declaration cannot vouch for a signal"


def test_original_late_child_and_continuing_outputs_keep_identity_across_restart(focus, client, monkeypatch):
    root, _place, _admission, event = focus
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_kw: (True, ""))
    template = task_for(focus)
    template["metadata"]["presence"]["delivery_reporting_version"] = 1

    def selected(tid, *, status="completed", child="", continuation=None):
        task = {**copy.deepcopy(template), "id": tid}
        ctx = ToolContext(repo_dir=root, drive_root=root, budget_drive_root=root,
                          task_id=tid, task_metadata=task["metadata"])
        if child:
            ctx._swarm_handoff_attempt = {"status": "scheduled", "task_id": child}
        result = build_presence_result_event(task, "Same words", ctx, terminal_origin="model_final")
        assert result["output_ref"]
        write_task_result(root, tid, status, metadata=task["metadata"], terminal_origin="model_final",
                          result="Same words", **({"presence_continuation": continuation} if continuation else {}))
        return result["output_ref"]

    original_id = selected("original", child="late-child")
    write_task_result(root, "late-child", "running", metadata=copy.deepcopy(template["metadata"]))
    first_ref = output_ref("continuing", "released-selection", event.conversation_key)
    continued = {"initial": {"work_ref": "late-child"}, "child_work_ref": "late-child",
                 "outputs": [{"output_ref": first_ref, "outcome": "message", "text": "Same words"}]}
    final_ref = selected("continuing", continuation=continued)
    # An original can fail without a selected answer while its real child still owes work.
    write_task_result(root, "failed-original", "failed", metadata={**copy.deepcopy(template["metadata"]),
                      "presence_work_ref": "late-child", "presence_unknown_outcome": {"source": "unknown"}},
                      terminal_origin="model_final", result="unselected draft")
    write_task_result(root, "legacy", "completed", metadata=copy.deepcopy(template["metadata"]),
                      terminal_origin="model_final", result="Same words")

    def poll(http, ref):
        response = http.get("/presence/work/" + ref)
        assert response.status_code in {200, 202}, response.text
        row = response.json()
        assert row["work_ref"] == row["task_id"] == ref  # producer != its child
        return row

    original = poll(client, "original")
    assert original["output_ref"] == original_id and original["child_work_ref"] == "late-child"
    assert poll(client, "late-child")["status"] == "pending"
    failed = poll(client, "failed-original")
    assert failed["status"] == "failed" and failed["child_work_ref"] == "late-child"
    assert (failed["text"], failed["output_ref"]) == ("", "")
    assert poll(client, "legacy")["output_ref"] == "", "absence must remain a protocol gap"
    tail = poll(client, "continuing")
    assert tail["outputs"][0]["output_ref"] == first_ref and tail["output_ref"] == final_ref

    journal = root / "consumer-journal.json"
    journal.write_text("{}")

    def consume(row):
        seen = json.loads(journal.read_text(encoding="utf-8"))  # a restarted consumer has no in-memory dedupe
        for output in [*row.get("outputs", []), row]:
            if output.get("text"):
                assert output.get("output_ref"), "the consumer cannot invent a missing identity"
                seen.setdefault(output["output_ref"], {"task_id": row["task_id"], "text": output["text"]})
        journal.write_text(json.dumps(seen))

    consume(original)
    consume(tail)
    child_ref = selected("late-child", status="failed")  # selected speech survives a terminal failure
    with TestClient(host_service.create_host_service_app(root)) as restarted:
        for _ in range(2):
            for ref in ("original", "continuing", "late-child"):
                consume(poll(restarted, ref))
        assert poll(restarted, "failed-original")["status"] == "failed"
    seen = json.loads(journal.read_text(encoding="utf-8"))
    assert set(seen) == {original_id, first_ref, final_ref, child_ref}
    assert seen[child_ref]["task_id"] == "late-child"
    assert len({row["text"] for row in seen.values()}) == 1  # identical text is four independent outputs


def test_speaker_can_schedule_no_event_followup_with_host_carrier_and_related_budget(focus, monkeypatch):
    from ouroboros.tools.followup import _handle_schedule_followup, _manage_schedules
    from ouroboros.usage_admission import task_billing_fields
    from supervisor import queue, workers
    root, _place, _admission, _event = focus
    monkeypatch.setattr(queue, "DRIVE_ROOT", root)
    monkeypatch.setattr(workers, "DRIVE_ROOT", root)
    task = task_for(focus)
    binding = task_billing_fields(task, task["id"], 2, root, pin_initial=True, persist_initial=True)
    task["metadata"]["billing_group"] = {k: v for k, v in binding.items() if k.startswith("billing_group_")}
    write_task_result(root, task["id"], "running", metadata=task["metadata"], task_contract=task["task_contract"])
    ctx = ToolContext(repo_dir=root, drive_root=root, budget_drive_root=root, task_id=task["id"],
                      current_chat_id=task["chat_id"], task_metadata=task["metadata"], task_contract=task["task_contract"])
    result = _handle_schedule_followup(ctx, objective="Return with the promised findings", notify=False,
                                      run_at="2030-01-01T00:00:00Z", relation="related")
    assert result.startswith("FOLLOWUP_SCHEDULED"), result
    [schedule] = queue.list_scheduled_tasks(root)["tasks"]
    future = queue._task_from_schedule(schedule)
    assert future["metadata"]["presence"] == json.loads(json.dumps(task["metadata"]["presence"]))
    assert future["metadata"]["objective_author"] == {"kind": "task", "task_id": task["id"]}
    assert schedule["followup_relation"]["billing_group"] == task["metadata"]["billing_group"]
    assert not future.get("project_id")
    assert "FOLLOWUP_NOTIFY_REFUSED" in _handle_schedule_followup(ctx, objective="A private note", notify=True,
                                                                 run_at="2030-01-01T00:00:00Z", relation="related")
    # The speaker sees only its own promise (#1589); the owner's schedule in the same table stays out of
    # reach, and changing a promise in place stays refused (withdraw it and schedule anew instead).
    from supervisor.schedule_lifecycle import upsert_scheduled_task

    upsert_scheduled_task({"id": "owner-digest", "name": "Owner digest", "trigger": {"type": "once", "run_at": "2030-01-01T00:00:00Z"},
                           "task": {"type": "task", "text": "Send the digest"}}, drive_root=root)
    assert [row["id"] for row in json.loads(_manage_schedules(ctx, "list"))["tasks"]] == [schedule["id"]]
    assert json.loads(_manage_schedules(ctx, "get", schedule_id="owner-digest"))["status"] == "not_found"
    assert "RESOURCE_CONSTRAINT_BLOCKED" in _manage_schedules(ctx, "update", schedule_id=schedule["id"], objective="Later")
    assert sorted(row["id"] for row in queue.list_scheduled_tasks(root)["tasks"]) == sorted([schedule["id"], "owner-digest"])
