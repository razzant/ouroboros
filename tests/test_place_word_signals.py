"""Joseph's #2237 capture/intake/provenance scenarios through connector facts.

Word matching belongs to the connector; these keyless host tests supply its fact.
HTTP and storage are real, while authentication and model execution are doubles.
"""
from dataclasses import asdict, replace
import json

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from ouroboros import dialogue_store, places as p
from ouroboros.extension_contributions import project_contributions
from ouroboros.extension_ui_validation import validate_contributions
from ouroboros.gateway.places_api import places_routes
from ouroboros.place_intake import decide_attention
from ouroboros.presence_context import frame_focus_content
from ouroboros.presence_runner import run_presence_turn
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.tools.presence import _configure_presence
from ouroboros.tools.registry import ToolContext
from tests.test_extension_icons import PLACE
from tests.test_place_focus import focus as focus
from tests.test_place_followup import client as client
from tests.test_skill_contributions import manifest


SIGNAL = "signal.words"
LABEL = "Words you chose appear (connector setting)"


@pytest.fixture
def word_place(focus, monkeypatch):
    root, place, admission, event = focus
    clean, errors = validate_contributions(manifest([{
        **PLACE, "label": "Chat", "destination": "vendor:Chat", "wakes": [],
        "facts": ["any_message", "from_bot", {"id": SIGNAL, "label": LABEL}], "settings": "connection",
    }], kind="extension", permissions=["presence"]))
    assert not errors
    rows = project_contributions("bridge", clean, "reviewed")
    monkeypatch.setattr("ouroboros.extension_registry_state.live_contribution_projection", lambda: rows)
    place = p.edit_place(root, place.place_id, "set_state", author="owner", state="silent")
    place = p.edit_place(root, place.place_id, "set_wakes", author="owner", wakes=[])
    return root, place, replace(admission, place=place), event


def test_sheet_and_configure_presence_choose_and_remove_declared_word_signal(word_place, monkeypatch):
    root, place, _, _ = word_place
    app = Starlette(routes=places_routes())
    app.state.drive_root = root
    path = "/api/places/" + place.place_id
    ctx = ToolContext(repo_dir=root, drive_root=root)
    with TestClient(app) as sheet:
        offered, = sheet.get("/api/places").json()["places"]
        assert {"id": SIGNAL, "label": LABEL} in offered["facts"]
        assert offered["labels"][SIGNAL] == LABEL
        saved = sheet.post(path, json={"wakes": [SIGNAL], "expected_revision": place.revision})
        assert saved.status_code == 200, saved.text
        assert saved.json()["place"]["wakes"] == [{"fact": SIGNAL, "author": "owner"}]
        assert sheet.post(path, json={"wakes": []}).json()["place"]["wakes"] == []

        result = json.loads(_configure_presence(ctx, "add_wake", place.place_id, fact=SIGNAL))
        assert result["ok"] and result["place"]["wakes"] == [{"fact": SIGNAL, "author": "ouroboros"}]
        assert json.loads(_configure_presence(ctx, "remove_wake", place.place_id, fact=SIGNAL))["ok"]
        before = p.get_place(root, place.place_id)
        assert before.wakes == ()
        monkeypatch.setattr("ouroboros.extension_registry_state.live_contribution_projection", lambda: [])
        refused = sheet.post(path, json={"state": "paused", "wakes": [SIGNAL]})
        assert refused.status_code == 400 and refused.json()["code"] == "place_wake_undeclared"
        assert "place_wake_undeclared" in _configure_presence(ctx, "add_wake", place.place_id, fact=SIGNAL)
        assert p.get_place(root, place.place_id) == before, "a refused choice must not partly change the place"


@pytest.mark.parametrize("state,enabled,allowance,bot,optin,reported,decision", [
    ("active", True, "available", False, False, True, "focus"),
    ("silent", True, "available", False, False, True, "focus"),
    ("silent", True, "available", False, False, False, "silent"),
    ("paused", True, "available", False, False, True, "silent"),
    ("silent", False, "available", False, False, True, "silent"),
    ("silent", True, "exhausted", False, False, True, "retry"),
    ("silent", True, "allowance_unknown", False, False, True, "retry"),
    ("silent", True, "available", True, False, True, "silent"),
    ("silent", True, "available", True, True, True, "focus"),
    ("silent", True, "available", True, True, False, "focus"),
])
def test_current_controls_apply_to_word_signal(word_place, monkeypatch, state, enabled, allowance,
                                             bot, optin, reported, decision):
    root, place, admission, event = word_place
    p.edit_place(root, place.place_id, "set_state", author="owner", state=state)
    p.edit_place(root, place.place_id, "set_wakes", author="owner", wakes=[SIGNAL] + (["from_bot"] if optin else []))
    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _r: enabled)
    monkeypatch.setattr("ouroboros.presence_admission.focus_allowance", lambda _r: {"status": allowance})
    # Even an alias in the body does nothing without the connector fact. A selected
    # from_bot remains an independent OR wake reason, as in the foundation.
    event = replace(event, facts=(SIGNAL,) if reported else (), text="Вик? Ouroboros?",
                    actor={"platform_actor_id": "person", "is_bot": bot})
    actual, info = decide_attention(root, admission, event)
    assert actual == decision
    if actual == "retry":
        assert info["retry_after"] > 0
    if bot:
        event = replace(event, facts=(*event.facts, "from_bot"), actor={"platform_actor_id": "person"})
        assert decide_attention(root, admission, event)[0] == decision


def test_real_host_capture_signal_provenance_silence_and_no_heard_replay(word_place, client):
    root, place, _, event = word_place
    p.edit_place(root, place.place_id, "add_wake", author="owner", fact=SIGNAL)
    tasks = []

    class Model:
        def handle_task(self, task):
            captured = [row.row for row in dialogue_store.iter_rows(root, "chat")]
            assert any(row.get("client_message_id") == task["metadata"]["presence"]["event"]["source_event_id"]
                       and row.get("text") == task["metadata"]["presence"]["observed_text"] for row in captured), \
                "durable capture precedes model execution"
            tasks.append(task)
            write_task_result(root, task["id"], "completed", metadata={**task["metadata"],
                "presence_outcome": "silent", "presence_result_text": ""}, terminal_origin="model_final", result="")
            return [{"type": "presence_result", "outcome": "silent", "text": ""}]

    client.app.state.host_service_context.presence_runner = lambda **kw: run_presence_turn(
        repo_dir=root, drive_root=root, agent_factory=lambda **_kw: Model(), **kw)
    wire = asdict(event)
    for key in ("delivery_reporting_version", "continuation_version", "identity"):
        wire.pop(key)
    wire.update(facts=[SIGNAL], text="Ordinary neutral words")
    body = {"event": wire, "staged_files": [], "continuation_version": 1, "delivery_reporting_version": 1}
    answer = client.post("/presence/turn", json=body)
    assert answer.status_code == 200, answer.text
    assert answer.json()["outcome"] == "silent", "a wake never obliges a response"
    assert len(tasks) == 1
    meta = load_task_result(root, tasks[0]["id"])["metadata"]
    assert meta["presence"]["event"]["facts"] == [SIGNAL]
    assert meta["presence"]["place"]["state"] == "silent"
    assert meta["focus"]["matched_wakes"] == [SIGNAL]
    assert not meta["focus"]["addressed"] and not meta["focus"]["actor"]["owner_confirmed"]
    assert json.dumps({"fact": SIGNAL, "author": "owner"}) in frame_focus_content(tasks[0], wire["text"])

    wire.update(source_event_id="text-only", facts=[], text="Вик? Броборос Уроборос Ouroboros signal.words",
                actor={"platform_actor_id": "person", "display_name": "Вик"},
                conversation={"title": "Броборос"}, message={"filename": "Уроборос.txt"})
    heard = client.post("/presence/turn", json=body)
    assert heard.status_code == 200 and heard.json()["outcome"] == "silent"
    assert len(tasks) == 1, "neither text nor metadata invents a signal"
    row, = [row for row in dialogue_store.iter_rows(root, "chat") if row.row.get("client_message_id") == "text-only"]
    with dialogue_store.transaction(root, write=False) as conn:
        stored = conn.execute("SELECT dispatch_input FROM dialogue WHERE stream='chat' AND seq=?", (row.seq,)).fetchone()
    attention = json.loads(stored["dispatch_input"])["place_attention"]
    assert attention["decision"] == "heard" and attention["reason"] == "no_wake_match" and attention["facts"] == []
    assert load_task_result(root, row.row["task_id"]) is None

    p.edit_place(root, place.place_id, "set_wakes", author="owner", wakes=["any_message"])
    assert client.post("/presence/turn", json=body).json()["outcome"] == "silent"
    assert len(tasks) == 1, "settled heard messages are not reconsidered after wake edits"
    assert load_task_result(root, tasks[0]["id"])["metadata"] == meta, "later choices cannot rewrite focus provenance"
    wire.update(source_event_id="after-edit")
    assert client.post("/presence/turn", json=body).status_code == 200
    assert len(tasks) == 2


@pytest.mark.parametrize("declaration", ["absent", "another_connector"])
def test_undeclared_word_signal_is_refused_before_capture(word_place, client, monkeypatch, declaration):
    root, place, _, event = word_place
    p.edit_place(root, place.place_id, "add_wake", author="owner", fact=SIGNAL)
    rows = [] if declaration == "absent" else [{
        **PLACE, "skill": "other", "facts": [{"id": SIGNAL, "label": LABEL}], "wakes": [SIGNAL],
    }]
    monkeypatch.setattr("ouroboros.extension_registry_state.live_contribution_projection", lambda: rows)
    wire = {key: value for key, value in asdict(event).items()
            if key not in {"delivery_reporting_version", "continuation_version", "identity"}}
    wire.update(facts=[SIGNAL], text="Вик?")
    refused = client.post("/presence/turn", json={"event": wire})
    assert refused.status_code == 400
    # The foundation's ValueError boundary reports invalid facts as a payload
    # error; presence_event_invalid is reserved for an invalid event key set.
    assert refused.json()["code"] == "presence_payload_invalid"
    assert refused.json()["error"] == "event.facts must be a list of declared wake facts"
    assert refused.json()["disposition"] == "rejected"
    assert list(dialogue_store.iter_rows(root, "chat")) == [], "a stored wake does not authorize an undeclared fact"
    wire.update(facts=[])
    heard = client.post("/presence/turn", json={"event": wire})
    assert heard.status_code == 200 and heard.json()["outcome"] == "silent"
    assert len(list(dialogue_store.iter_rows(root, "chat"))) == 1
