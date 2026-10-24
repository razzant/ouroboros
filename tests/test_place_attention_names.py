"""Host name attention is independent of provider mentions and owner authority."""
from dataclasses import asdict, replace
import json

import pytest

from ouroboros import places as p
from ouroboros.gateway.places_api import update_place
from ouroboros.place_intake import decide_attention
from ouroboros.presence_runner import run_presence_turn
from ouroboros.task_results import write_task_result
from tests.test_place_focus import focus as focus
from tests.test_place_followup import client as client


NAMES = ["Вик", "Броборос", "Уроборос", "Ouroboros"]


def named(root, place, names=NAMES):
    return p.edit_place(root, place.place_id, "set_attention_names", author="owner", attention_names=names)


def test_roundtrip_clear_old_records_and_atomic_gateway(focus, monkeypatch):
    root, place, _, _ = focus
    monkeypatch.setattr("ouroboros.place_projection.place_declarations", lambda _r: [])
    assert place.attention_names == ()
    assert p._decode({k: v for k, v in asdict(place).items() if not k.startswith("attention_names")}) == place
    saved = update_place(root, place.place_id, {"attention_names": [" Вик ", "Ouroboros", "ВИК"],
                                                "expected_revision": place.revision})["place"]
    assert saved["attention_names"] == ("Вик", "Ouroboros")
    assert saved["attention_names_author"] == "owner"
    before = p.get_place(root, place.place_id)
    for bad in ("Вик", None, [1], [""], ["a\nb"]):
        with pytest.raises(p.PlaceError):
            update_place(root, place.place_id, {"state": "paused", "attention_names": bad})
        assert p.get_place(root, place.place_id) == before
    with pytest.raises(p.PlaceError, match="Current revision"):
        update_place(root, place.place_id, {"attention_names": [], "expected_revision": place.revision})
    assert not update_place(root, place.place_id, {"attention_names": []})["place"]["attention_names"]
    assert p.get_place(root, place.place_id).wakes == place.wakes


@pytest.mark.parametrize("text,matched", [
    ("Привет, ВИК!", ("Вик",)), ("(Броборос): уроборос? OUROBOROS.", ("Броборос", "Уроборос", "Ouroboros")),
    ("Вик-то, 'Вик', «Вик»", ("Вик",)), ("Виктор Броборосик Уробороса antiOuroboros Ouroboros123 _Вик_", ()),
    ("Вик\u0301", ()), ("\u0301Вик", ()), ("Вик\u203fnext", ()), ("Вик٢", ()),
    ("", ()), ("ordinary neutral body", ()),
])
def test_literal_unicode_words(focus, text, matched):
    root, place, _, _ = focus
    place = named(root, place)
    assert p.attention_name_matches(place, text, ()) == matched


def test_casefold_and_literal_not_regex(focus):
    root, place, _, _ = focus
    place = named(root, place, ["Straße", "A+B", "[Vic]"])
    assert p.attention_name_matches(place, "STRASSE A+B [Vic]", ()) == place.attention_names
    assert not p.attention_name_matches(place, "Strassen AAAB Vic", ())


@pytest.mark.parametrize("state,enabled,allowance,bot,optin,decision", [
    ("active", True, "available", False, False, "focus"),
    ("silent", True, "available", False, False, "focus"),
    ("paused", True, "available", False, False, "silent"),
    ("active", False, "available", False, False, "silent"),
    ("active", True, "exhausted", False, False, "retry"),
    ("active", True, "allowance_unknown", False, False, "retry"),
    ("active", True, "available", True, False, "silent"),
    ("active", True, "available", True, True, "focus"),
])
def test_current_controls_apply_to_name_only(focus, monkeypatch, state, enabled, allowance, bot, optin, decision):
    root, place, admission, event = focus
    named(root, place)
    p.edit_place(root, place.place_id, "set_state", author="owner", state=state)
    if optin:
        p.edit_place(root, place.place_id, "add_wake", author="owner", fact="from_bot")
    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _r: enabled)
    monkeypatch.setattr("ouroboros.presence_admission.focus_allowance", lambda _r: {"status": allowance})
    event = replace(event, facts=("from_bot",) if bot else (), text="Вик?", actor={"platform_actor_id": "someone"})
    assert decide_attention(root, admission, event)[0] == decision
    if bot and not optin:
        assert decide_attention(root, admission, replace(event, facts=(), actor={"is_bot": True}))[0] == "silent"


def test_tool_writer_same_rule_and_authorship(focus, monkeypatch):
    from types import SimpleNamespace
    from ouroboros.tools.presence import _configure_presence, get_tools
    root, place, _, _ = focus
    monkeypatch.setattr("ouroboros.tool_access.canonical_data_root", lambda _ctx: root)
    monkeypatch.setattr("ouroboros.place_projection.place_declarations", lambda _r: [])
    ctx = SimpleNamespace(task_metadata={}, is_direct_chat=False)
    result = json.loads(_configure_presence(ctx, "set_attention_names", place.place_id, attention_names=["Вик"]))
    assert result["ok"] and result["place"]["attention_names_author"] == "ouroboros"
    assert "set_attention_names" in get_tools()[0].schema["parameters"]["properties"]["action"]["enum"]
    assert json.loads(_configure_presence(ctx, "set_attention_names", place.place_id, attention_names=[]))["ok"]
    assert not p.get_place(root, place.place_id).attention_names


def test_real_host_capture_focus_provenance_silence_metadata_exclusion_and_no_replay(focus, client):
    from ouroboros import dialogue_store
    from ouroboros.presence_context import frame_focus_content
    root, place, _, event = focus
    named(root, place)
    tasks = []

    class Model:
        def handle_task(self, task):
            rows = list(dialogue_store.iter_rows(root, "chat"))
            assert any(row.row.get("text") == "Вик?" for row in rows), "durable capture precedes the model"
            tasks.append(task)
            write_task_result(root, task["id"], "completed", metadata={**task["metadata"],
                "presence_outcome": "silent", "presence_result_text": ""}, terminal_origin="model_final", result="")
            return [{"type": "presence_result", "outcome": "silent", "text": ""}]

    client.app.state.host_service_context.presence_runner = lambda **kw: run_presence_turn(
        repo_dir=root, drive_root=root, agent_factory=lambda **_kw: Model(), **kw)
    wire = asdict(event)
    wire.pop("delivery_reporting_version")
    wire.pop("continuation_version")
    wire.update(facts=[], text="Вик?")
    body = {"event": wire, "staged_files": [], "continuation_version": 1, "delivery_reporting_version": 1}
    answer = client.post("/presence/turn", json=body)
    assert answer.status_code == 200, answer.text
    assert answer.json()["outcome"] == "silent", "attention never obliges a response"
    meta = tasks[0]["metadata"]
    assert meta["presence"]["event"]["facts"] == []
    assert meta["focus"]["matched_wakes"] == []
    assert meta["focus"]["matched_attention_names"] == ["Вик"]
    assert not meta["focus"]["addressed"] and not meta["focus"]["actor"]["owner_confirmed"]
    assert '"host_name_rule": {"names": ["Вик"], "author": "owner"}' in frame_focus_content(tasks[0], "Вик?")
    wire.update(source_event_id="metadata-only", text="neutral body", actor={"platform_actor_id": "person", "display_name": "Вик"},
                conversation={"title": "Броборос"}, message={"sender_name": "Ouroboros", "filename": "Уроборос.txt"})
    assert client.post("/presence/turn", json=body).json()["outcome"] == "silent"
    named(root, place, ["neutral"])
    assert client.post("/presence/turn", json=body).json()["outcome"] == "silent"
    assert len(tasks) == 1, "settled heard messages are not reconsidered after editing aliases"
    wire.update(source_event_id="new-name-event", text="neutral")
    assert client.post("/presence/turn", json=body).status_code == 200
    assert len(tasks) == 2


def test_queued_admission_uses_deciding_name_snapshot_for_provenance(focus, monkeypatch):
    from ouroboros import place_intake
    from ouroboros.presence_context import frame_focus_content
    root, place, admission, event = focus
    event = replace(event, text="Newname?", facts=())
    named(root, place, ["Newname"])  # admission still holds the earlier empty rule
    original = place_intake.attention_snapshot

    def decide_then_edit(*args):
        result = original(*args)
        named(root, place, ["Latername"])  # no second read may rewrite the decision's provenance
        return result

    monkeypatch.setattr(place_intake, "attention_snapshot", decide_then_edit)
    seen = []

    class Model:
        def handle_task(self, task):
            seen.append(task)
            write_task_result(root, task["id"], "completed", metadata={**task["metadata"],
                "presence_outcome": "silent", "presence_result_text": ""}, terminal_origin="model_final", result="")
            return [{"type": "presence_result", "outcome": "silent", "text": ""}]

    result = run_presence_turn(admission=admission, event=event, repo_dir=root, drive_root=root,
                               agent_factory=lambda **_kw: Model())
    assert result.outcome == "silent" and len(seen) == 1
    meta = seen[0]["metadata"]
    assert meta["focus"]["matched_attention_names"] == ["Newname"]
    assert meta["presence"]["place"]["attention_names"] == ["Newname"]
    assert '"host_name_rule": {"names": ["Newname"], "author": "owner"}' in frame_focus_content(seen[0], event.text)
    assert p.get_place(root, place.place_id).attention_names == ("Latername",)
