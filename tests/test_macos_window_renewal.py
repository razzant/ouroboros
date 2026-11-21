"""The macOS window-renewal check judges honestly without opening a window here.

The native run is operator- or dispatch-invoked on macOS (devtools/macos_wkwebview/window_renewal.md). These cases
pin what its verdict may conclude from recorded checks: a candidate passes only beside a baseline that kept its web
view through the same operations, a check not established or not calibrated is never a pass, an unsupported host
never reads as one, the Project Send ledger keeps missing and false apart across transitions, a held request is bound
to a generation only by its measured parent chain, one native AppKit Return drives an ordinary Send with no
substitution or retry, a restored quote is compared in its complete stored shape, and the UI anchors and the dispatch-only
workflow it relies on still exist. The arm's source and managed mirror are built here with real Git, so a wrong seed
or offer cannot first appear on the one hosted Mac.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import shutil
import sys
from types import SimpleNamespace

import pytest
import yaml

from devtools.macos_wkwebview import arm_source, native_window, typing_evidence
from devtools.macos_wkwebview import journey_model as journeys
from devtools.macos_wkwebview import window_renewal as renewal

REPO = pathlib.Path(__file__).resolve().parents[1]


def _arm(**values):
    checks = {phase: dict.fromkeys(names, True) for phase, names in renewal.PHASES.items()}
    for key, value in values.items():
        phase, name = key.split(".")
        checks[phase][name] = value
    return {"checks": checks, "gaps": []}


def _baseline(**values):
    kept = dict.fromkeys(renewal.CONTROL, False)
    return _arm(**{**kept, "fixture.sample_document_opened": False, "restart.diagnostic_lookup_recorded": None,
                   "update.diagnostic_lookup_recorded": None, **values})


def _judge(baseline=None, candidate=None):
    arms = {name: arm for name, arm in (("baseline", baseline), ("candidate", candidate)) if arm is not None}
    return renewal.judge({"arms": arms})


def test_the_candidate_passes_only_beside_a_baseline_that_kept_its_view_through_the_same_operations():
    assert _judge(_baseline(), _arm())["verdict"] == "PASS"
    for key in renewal.CONTROL:
        shown = _judge(_baseline(**{key: True}), _arm())
        assert shown["verdict"] == "INCONCLUSIVE" and any(key in gap for gap in shown["gaps"]), key
    for key in renewal.SAME_OPERATION:
        assert _judge(_baseline(**{key: False}), _arm())["verdict"] == "INCONCLUSIVE", key
    assert _judge(None, _arm())["verdict"] == "INCONCLUSIVE"


def test_a_kept_view_a_lost_draft_or_a_second_copy_fails_the_candidate():
    for key in ("restart.view_new", "restart.main_active_after", "fixture.view_new", "update.project_active_after", "handoff.project_active_after",
                "handoff.project_editable", "update.project_draft", "handoff.project_reply_restored", "update.project_reattach_notice",
                "update.project_window_typed_pixels", "handoff.project_sent_once", "update.draft_restored", "restart.reply_restored",
                "restart.no_replay", "update.pending_owner", "session.draft_gone", "fixture.sample_document_opened",
                "restart.project_send_kept", "update.project_send_kept", "session.project_send_kept",
                "update.project_send_native_input"):
        judged = _judge(_baseline(), _arm(**{key: False}))
        assert judged["verdict"] == "FAIL" and any(key in line for line in judged["failures"]), key


def test_a_check_not_established_or_not_calibrated_is_never_a_pass_or_a_regression():
    unknown = _judge(_baseline(), _arm(**{"update.reply_restored": None}))
    assert unknown["verdict"] == "INCONCLUSIVE" and not unknown["failures"]
    unread = _judge(_baseline(), _arm(**{"restart.diagnostic_lookup_recorded": None}))
    assert unread["verdict"] == "INCONCLUSIVE" and any("restart.diagnostic_lookup_recorded" in gap for gap in unread["gaps"])
    for key in renewal.CALIBRATED:
        blind = _judge(_baseline(**{key: False}), _arm(**{key: False}))
        assert blind["verdict"] == "INCONCLUSIVE" and not blind["failures"], key
        assert _judge(_baseline(), _arm(**{key: False}))["verdict"] == "FAIL", key
    stopped = _judge(_baseline(), {**_arm(), "gaps": ["stopped at TimeoutError: no window"]})
    assert stopped["verdict"] == "INCONCLUSIVE"


def test_an_unsupported_host_never_reads_as_a_pass():
    assert renewal.judge({"unsupported": "needs macOS; this is linux", "arms": {}})["verdict"] == "UNSUPPORTED"
    assert renewal.EXIT_CODES == {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 3, "UNSUPPORTED": 4}


def test_readability_needs_a_capture():
    assert typing_evidence.pixels_readable(None) is None
    assert typing_evidence.pixels_readable({"ok": False}) is None
    assert typing_evidence.pixels_readable({"ok": True, "distinct_colors": 3, "dark_fraction": 1.0}) is False
    assert typing_evidence.pixels_readable({"ok": True, "distinct_colors": 400, "dark_fraction": 0.9}) is True


def test_an_older_arm_that_opens_main_still_calibrates_the_same_updates():
    differences = {f"{phase}.{key}": None for phase in renewal.PROJECT_UPDATES for key in renewal.PROJECT_CHECKS
                   if key not in ("project_active_before", "document_ready")}
    differences.update({f"{phase}.project_active_after": False for phase in renewal.PROJECT_UPDATES})
    assert _judge(_baseline(**differences), _arm())["verdict"] == "PASS"
    for key in ("handoff.project_sent_once", "update.project_typed_pixels", "update.project_window_typed_pixels",
                "update.project_send_native_input"):
        assert _judge(_baseline(**differences), _arm(**{key: None}))["verdict"] == "INCONCLUSIVE"


@pytest.mark.parametrize("change", ["letters", "background", "caret", "misplaced"])
def test_window_typing_requires_the_appended_letters(tmp_path, change):
    from PIL import Image, ImageDraw, ImageFont

    word, font = native_window.TYPING_WORD, ImageFont.load_default(size=18)
    width, advance = font.getlength(word), font.getlength(word + " ")
    window = {"frame": [0, 0, 244, 64], "viewport": [0, 12, 244, 52]}
    box = [10, 5, 234, 41, 244, 52, 4, width, advance]
    paths = []
    for i in range(4):
        picture = Image.new("RGB", (244, 64), "white")
        draw = ImageDraw.Draw(picture)
        draw.rectangle((10, 17, 233, 52), fill="silver" if i == 3 and change == "background" else "white")
        text = word + " " + word if i == 3 and change == "letters" else word
        draw.text((14, 21), text, font=font, fill="black")
        if i == 3 and change == "misplaced":
            draw.text((14 + 2 * advance, 21), word, font=font, fill="black")
        if i == 1 or (i == 3 and change == "caret"):
            x = round(14 + width + (advance if i == 3 else 0))
            draw.line((x, 21, x, 39), fill="black")
        paths.append(tmp_path / f"window-{i}.png")
        picture.save(paths[-1])
    metrics, _ = typing_evidence.typing_delta(paths, box, window)
    if change == "background":
        assert metrics["change"] > 3 * metrics["noise"] + 40  # the old area check accepted this repaint
    assert metrics["visible_word"] is (change == "letters")


def test_receipts_distinguish_missing_reads_replayed_text_and_original_identity():
    wire = {"type": "chat", "content": "synthetic", "client_message_id": "original", "sender_session_id": "one"}
    accepted = {"role": "user", "text": "synthetic", "client_message_id": "original", "sender_session_id": "one",
                "ingress_accepted": True, "row_address": "row:1"}
    def read(*rows, complete=True):
        return renewal.receipt({"messages": list(rows), "window": {"complete": complete}}, wire)
    assert renewal.receipt(None, wire) is None
    assert read(complete=False) is None
    assert read() == {"count": 0, "addresses": []}
    assert read(accepted) == {"count": 1, "addresses": ["row:1"]}
    assert read(dict(accepted, client_message_id="other")) == {"count": 1, "addresses": []}
    assert read(dict(accepted, sender_session_id="other")) == {"count": 1, "addresses": []}
    assert read(dict(accepted, ingress_accepted=False))["addresses"] == []
    assert read(accepted, dict(accepted, client_message_id="other"))["count"] == 2
    assert read(accepted, accepted)["count"] == 2


def test_pending_custody_requires_the_same_original_sender_room_reply_content_and_id():
    wire = {"type": "chat", "content": "hello", "client_message_id": "id-1", "sender_session_id": "sender",
            "chat_id": 23, "project_id": "p", "reply_to": "row:23#original"}
    def kept(*frames):
        return renewal.same_frame(wire, [{"frame": f, "views": []} for f in frames])
    assert kept(wire)
    for key in wire:
        assert not kept({**wire, key: "other"}), key
    assert not kept()
    assert not kept(wire, wire)
    assert renewal.same_frame({}, []) is None


@pytest.mark.parametrize("case,verdict", [
    ("one", True), ("wrong project", False), ("wrong wire room", False), ("wrong wire reply", False),
    ("wrong saved room", False), ("wrong saved reply", False), ("unresolved reply", False), ("wrong reply room", False),
    ("wrong sender", False), ("wrong saved id", False), ("wrong saved text", False), ("unaccepted", False),
    ("duplicate id", False), ("duplicate text", False), ("also in Main", False), ("two writes", False),
    ("partial Project", None), ("partial Main", None), ("missing frame", None), ("missing id", None),
    ("missing saved id", None), ("missing address", None), ("missing acceptance", None),
    ("no row yet", None), ("counter unavailable", None), ("other draft", None), ("live file", None),
])
def test_one_return_requires_the_original_frame_one_accepted_project_row_and_complete_main_history(monkeypatch, case, verdict):
    arm = object.__new__(renewal.Arm)
    arm.tag, arm.project_reply, arm.facts, arm.generation = "sample", "row:17@seed#0123456789ab", {}, {"port": 5001}
    arm.project = {"id": "room-sample", "chat_id": 17}
    text = "Project-update-sample"
    wire = {"type": "chat", "content": text, "client_message_id": "new-id", "sender_session_id": "sender",
            "project_id": arm.project["id"], "chat_id": 17, "reply_to": arm.project_reply}
    row = {"role": "user", "text": text, "ingress_accepted": True, "client_message_id": "new-id",
           "sender_session_id": "sender", "row_address": "row:17@now#0123456789ab",
           "reply_to": {"address": arm.project_reply, "chat_id": 17}}
    wire.update({"wrong project": {"project_id": "other"}, "wrong wire room": {"chat_id": 1},
                 "wrong wire reply": {"reply_to": "row:1@other"}, "missing id": {"client_message_id": ""}}.get(case, {}))
    row.update({"wrong saved room": {"row_address": "row:1@now#0123456789ab"}, "wrong saved reply": {"reply_to": {}},
                "unresolved reply": {"reply_to": {"address": arm.project_reply, "unresolved": True}},
                "wrong reply room": {"reply_to": {"address": arm.project_reply, "chat_id": 1}},
                "wrong sender": {"sender_session_id": "other"}, "wrong saved id": {"client_message_id": "other"},
                "wrong saved text": {"text": "Main draft"}, "unaccepted": {"ingress_accepted": False}}.get(case, {}))
    for scenario, key in (("missing saved id", "client_message_id"), ("missing address", "row_address"),
                          ("missing acceptance", "ingress_accepted")):
        if case == scenario:
            del row[key]
    project_rows = [] if case == "no row yet" else [row]
    if case in ("duplicate id", "duplicate text"):
        project_rows.append({**row, "client_message_id": "new-id" if case == "duplicate id" else "second-id"})
    main_rows = [{**row, "client_message_id": "other-id"}] if case == "also in Main" else []
    arm.history = lambda chat_id=1: {"messages": main_rows if chat_id == 1 else project_rows,
                                     "window": {"complete": case != ("partial Main" if chat_id == 1 else "partial Project")}}
    arm.wire_counts = lambda content: {"two writes": 2, "counter unavailable": None}.get(case, 1) if content == text else None
    arm.project_sends = []
    view, drives = _view(port=5001), []
    def drive(content, prefix, target, *, from_draft):
        drives.append((content, prefix, target, from_draft))
        return {"wire": None if case == "missing frame" else wire, "drive": "appkit", "disclosure": None}
    draft = {"text": "Main text" if case == "other draft" else text, "names": [], "input_files": int(case == "live file")}
    arm.win = SimpleNamespace(key_window=lambda target: target, send_native=drive,
                              draft=lambda field: draft if field == "#pchat-room-sample-input" else None,
                              bubble=lambda msg_id, prefix, *, address: {"count": 1, "visible_accepted": address == row.get("row_address")})
    monkeypatch.setattr(renewal, "wait_for", lambda read, *_: read() or read())
    arm.project_send("update", view)
    assert arm.project_delivery("update") is verdict
    assert drives == ([] if case in ("other draft", "live file") else [(text, "#pchat-room-sample", view, True)])
    assert len(arm.project_sends) == len(drives)  # every attempted Send joins the identity-bound ledger
    if verdict is True:
        facts = arm.facts["update_project_send"]
        assert facts["wire"] == wire and facts["saved"] == {"count": 1, "addresses": [row["row_address"]]}
        assert facts["main"]["count"] == 0 and facts["writes"] == 1
        assert facts["accepted_address"] == row["row_address"] and facts["drive"] == "appkit"
        assert arm.native_send_proof("update") is True


def test_project_draft_uses_its_own_field_quote_target_and_empty_picker():
    prefix, reply = "#pchat-sample", "row:17@seed#0123456789ab"
    quote = {"address": reply, "chat_id": 17, "snippet": "Project-seed-sample"}
    staged = {"prefix": prefix, "reply": reply, "text": "Project text", "filename": "project.txt",
              "quote_text": "Project-seed-sample", "room_id": 17, "saved_quote": dict(quote),
              "shown_quote": {"author": "You", "when": "· 10:41", "text": "Project-seed-sample",
                              "label": "Replying to You, 10:41: Project-seed-sample"},
              "typed": True, "quoted": True, "file": True}
    doc = {"text": "Project text", "reply": True, "names": [], "input_files": 0, "notice": "Attach your file again"}
    quotes, cell = [], {"quote": dict(quote)}
    room = SimpleNamespace(until=lambda read, _: read(), observe=lambda name, read: read(),
                           draft=lambda field: doc if field == prefix + "-input" else None,
                           stored_quote=lambda marker, room_id: cell["quote"],
                           composer_quote=lambda _: staged["shown_quote"],
                           follows_quote=lambda addr, p, text: quotes.append((addr, p, text)) or addr == reply)
    room.restored_quote = lambda *args: native_window.RoomWindow.restored_quote(room, *args)
    checks = {}
    native_window.RoomWindow.inspect_draft(room, staged, checks, "marker")
    assert checks == {"project_draft": True, "project_reply_restored": True, "project_reattach_notice": True}
    assert quotes == [(reply, prefix, "Project-seed-sample")]
    cell["quote"] = dict(quote, snippet="changed")  # any single stored-shape difference is demonstrated
    checks = {}
    native_window.RoomWindow.inspect_draft(room, staged, checks, "marker")
    assert checks["project_reply_restored"] is False and quotes == [(reply, prefix, "Project-seed-sample")]
    cell["quote"] = None  # an unreadable slot stays missing, not false
    checks = {}
    native_window.RoomWindow.inspect_draft(room, staged, checks, "marker")
    assert checks["project_reply_restored"] is None
    cell["quote"] = dict(quote)
    doc.update(text="Main text", input_files=1)
    room.follows_quote = lambda *_: False
    checks = {}
    native_window.RoomWindow.inspect_draft(room, staged, checks, "marker")
    assert not any(checks.values())
    room.draft = lambda *_: None
    checks = {}
    native_window.RoomWindow.inspect_draft(room, staged, checks, "marker")
    assert not checks


def test_selecting_the_already_open_project_keeps_its_composer():
    seen = []
    room = SimpleNamespace(visible=lambda field: seen.append(field) or True)
    assert native_window.RoomWindow.open_project(room, {"id": "sample"}) is True
    assert seen == ["#project-panel.open #pchat-sample-input"]


def test_native_typing_and_both_images_use_the_focused_project_field(tmp_path, monkeypatch):
    field, view, label = "#pchat-sample-input", _view(port=5001), "handoff-project"
    keys, crops, images = [], [], []
    box = [8, 12, 200, 50, 230, 180, 4, 42, 48]
    def type_keys(sel, text, *, target, replace=True):
        keys.append((sel, text, target, replace))
        return True
    def capture(op, *, kind, name, target):
        assert op == "capture" and target is view
        images.append(name)
        return {"file": name + ".png"}
    def delta(paths, geometry, window):
        assert geometry == box and len(paths) == 4
        crops.append((paths[-1].name, window))
        return {"visible_word": True}, object()
    win = SimpleNamespace(out=tmp_path, key_window=lambda target: target, type=type_keys, call=capture,
                          value=lambda sel: "Window Window" if sel == field else "",
                          box=lambda sel: box if sel == field else None, page=lambda script: None,
                          observe=lambda name, read: read())
    monkeypatch.setattr(native_window.time, "sleep", lambda _: None)
    monkeypatch.setattr(typing_evidence, "window_colors", lambda _: {"ok": True, "distinct_colors": 80, "dark_fraction": 0.4})
    monkeypatch.setattr(typing_evidence, "typing_delta", delta)
    monkeypatch.setattr(typing_evidence, "echo_overlap", lambda *_: 1.0)
    checks, evidence = {}, {}
    native_window.RoomWindow.typing(win, view, field, label, checks, evidence)
    assert keys == [(field, "Window", view, True), (field, " Window", view, False)]
    assert evidence["field"] == field and evidence["css_field"] == box
    assert images == [f"{label}-{kind}-{sample}" for sample in ("quiet-0", "quiet-1", "quiet-2", "typed")
                      for kind in ("window", "renderer")]
    assert crops == [(f"{label}-window-typed.png", view), (f"{label}-renderer-typed.png", None)]
    assert checks == dict.fromkeys(("project_typed", "project_window_pixels", "project_typed_pixels", "project_window_typed_pixels"), True)


def test_a_deduplicated_wire_write_still_fails_and_a_lost_counter_is_unknown():
    assert renewal.send_writes([1, 0], [1, 0]) is True
    assert renewal.send_writes([1, 0], [2, 0]) is False
    assert renewal.send_writes([1, 0], [1, 1]) is False
    assert renewal.send_writes([1, 0], [0, 0]) is None
    assert renewal.send_writes(None, [1, 0]) is None


def test_once_needs_one_addressed_room_row_a_quiet_main_and_a_single_write():
    saved = {"count": 1, "addresses": ["row:17@x#0123456789ab"]}
    quiet = {"count": 0, "addresses": []}
    kept = renewal.history_verdict(saved, quiet)
    assert kept is True  # one addressed room row and a quiet Main: the server's own reading
    assert renewal.once_verdict(kept, 1) is True
    assert renewal.once_verdict(kept, 1, addressed=False) is False
    assert renewal.once_verdict(renewal.history_verdict(saved, {"count": 1, "addresses": ["row:1@x#0123456789ab"]}), 1) is False
    assert renewal.once_verdict(renewal.history_verdict({"count": 1, "addresses": []}, quiet), 1) is False
    assert renewal.once_verdict(kept, 2) is False
    assert renewal.once_verdict(kept, None) is None  # a lost counter proves no silence
    assert renewal.once_verdict(renewal.history_verdict(None, quiet), 1) is None
    # The history verdict is the server's own reading: a leak to Main or a lost accepted address is its
    # fact, and neither the captured frame nor the page's counter enters it.
    assert renewal.history_verdict(saved, {"count": 1, "addresses": ["row:1@x#0123456789ab"]}) is False
    assert renewal.history_verdict(saved, quiet, accepted_address="row:17@x#999999999999") is False
    assert renewal.history_verdict(saved, quiet, accepted_address=saved["addresses"][0]) is True
    wire = {"type": "chat", "content": "t", "project_id": "p", "chat_id": 3, "reply_to": "r"}
    assert renewal.frame_addressed(wire, "p", 3, "t", "r")
    assert not renewal.frame_addressed(dict(wire, chat_id=1), "p", 3, "t", "r")
    assert not renewal.frame_addressed(None, "p", 3, "t", "r")


def _project_ledger_arm(case):
    """One arm's Project Send ledger with the named history and wire-counter readings: no window, no page."""
    arm = object.__new__(renewal.Arm)
    arm.facts, arm.gaps = {}, []
    wire = {"type": "chat", "content": "Project-seed-tg", "client_message_id": "id-1",
            "sender_session_id": "s", "project_id": "room-tg", "chat_id": 17, "reply_to": None}
    entry = {"phase": "seed", "project": "room-tg", "room": 17, "text": "Project-seed-tg",
             "reply": None, "drive": "appkit", "wire": dict(wire)}
    if case.startswith("attempt only"):
        entry["wire"] = None  # an attempted Send whose frame was never observed
    arm.project_sends = [] if case == "no send ever" else [entry]
    if case == "kept beside attempt":
        arm.project_sends.append({**entry, "phase": "update", "text": "Project-update-tg", "wire": None})
    if case == "misaddressed frame":
        entry["wire"] = dict(wire, chat_id=18)  # the captured frame names another room than the Project's
    row = {"role": "user", "text": "Project-seed-tg", "client_message_id": "id-1", "sender_session_id": "s",
           "ingress_accepted": True, "row_address": "row:17@x#0123456789ab", "reply_to": {}}
    if case.startswith("accepted "):
        entry["accepted_address"] = row["row_address"]
    def history(chat_id=1):
        if chat_id == 1:
            rows = [dict(row, row_address="row:1@x#0123456789ab")] if case == "leaked to Main" else []
            return {"messages": rows, "window": {"complete": True}}
        if case == "duplicate row":
            rows = [row, dict(row, client_message_id="id-2")]
        elif case == "wrong room":
            rows = [dict(row, row_address="row:1@x#0123456789ab")]
        elif case == "accepted movable":
            rows = [dict(row, row_address="row:17@moved#0123456789ab")]
        elif case == "accepted unreadable":
            return None
        elif case in ("accepted missing", "accepted incomplete", "not yet accepted", "misaddressed frame"):
            rows = []
        else:
            rows = [row]
        return {"messages": rows, "window": {"complete": case not in ("room read incomplete", "accepted incomplete")}}
    arm.history = history
    writes = {"two writes": 2, "counter lost": None, "attempt only, duplicated": 2}.get(case, 1)
    arm.wire_counts = lambda text: writes if text == "Project-seed-tg" else None
    return arm


# ``proved``: whether the arm's own complete server history reads contradicted the Send, as apart from a page.
@pytest.mark.parametrize("case,verdict,proved", [
    ("kept", True, False), ("duplicate row", False, True), ("leaked to Main", False, True),
    ("two writes", False, False), ("wrong room", False, True), ("room read incomplete", None, False),
    ("counter lost", None, False), ("attempt only, quiet", None, False), ("attempt only, duplicated", False, False),
    ("no send ever", None, False), ("kept beside attempt", None, False), ("accepted missing", False, True),
    ("accepted movable", False, True), ("accepted incomplete", None, False), ("accepted unreadable", None, False),
    ("not yet accepted", None, False), ("misaddressed frame", False, False),
])
def test_the_project_send_ledger_keeps_missing_and_false_apart(case, verdict, proved):
    arm = _project_ledger_arm(case)
    assert arm.project_sends_kept("restart") is verdict
    rows = arm.facts["restart_project_ledger"]
    assert len(rows) == len(arm.project_sends)
    assert all(set(found) <= {"phase", "drive", "kept", "server_kept", "writes", "count", "main"} for found in rows)
    assert any(row.get("server_kept") is False for row in rows) is proved  # only a history read can prove the loss itself


@pytest.mark.parametrize("case,kept_after", [
    ("leaked to Main", False), ("accepted movable", False),  # the server's own complete read contradicts the Send
    ("two writes", None), ("attempt only, duplicated", None),  # only a page read contradicted it
    ("misaddressed frame", None),  # only the captured frame named another room
])
def test_a_late_view_change_erases_page_measurements_but_not_what_the_server_proved(case, kept_after):
    """A Project Send the server itself contradicted is not a measurement of a document: a view that changed
    afterwards still demotes every page-bound check of the phase, and the gap stays recorded."""
    arm = _project_ledger_arm(case)
    assert arm.project_sends_kept("restart") is False  # each case is a demonstrated negative where it was read
    checks = {"generation_new": True, "accepted_once": True, "view_new": None, "project_send_kept": False,
              "project_sent_once": True, "project_editable": True, "draft_restored": True}
    old = _view(view="a", loopback=True, port=5001)
    arm.reconcile_view("restart", old, old, _view(view="b", loopback=True, port=5001), checks, tuple(checks))
    assert checks["project_send_kept"] is kept_after
    assert checks["view_new"] is None and checks["project_editable"] is None
    assert checks["project_sent_once"] is None and checks["draft_restored"] is None
    assert checks["generation_new"] is True and checks["accepted_once"] is True
    assert len(arm.gaps) == 1


def test_the_ledger_and_native_drive_checks_gate_the_candidate_only():
    for key in ("restart.project_send_kept", "update.project_send_kept", "handoff.project_send_kept",
                "session.project_send_kept", "update.project_send_native_input", "handoff.project_send_native_input"):
        assert _judge(_baseline(), _arm(**{key: False}))["verdict"] == "FAIL", key
        assert _judge(_baseline(), _arm(**{key: None}))["verdict"] == "INCONCLUSIVE", key
    # A baseline that never kept its own ledger does not excuse the candidate: these are direct
    # custody checks, not calibrated probes.
    assert _judge(_baseline(**{"restart.project_send_kept": False, "update.project_send_native_input": None}),
                  _arm())["verdict"] == "PASS"


def test_every_transition_and_the_cold_relaunch_reread_the_project_ledger():
    for phase in (*renewal.TRANSITIONS, "session"):
        assert "project_send_kept" in renewal.PHASES[phase], phase
    for phase in renewal.PROJECT_UPDATES:
        assert "project_send_native_input" in renewal.PHASES[phase], phase
    assert "project_send_native_input" not in renewal.PHASES["restart"]


@pytest.mark.parametrize("resumed,live,reads,expect", [
    (True, {"task_id": "t1", "status": "running", "read_requests": 2}, 2, True),
    (True, {"task_id": "t2", "status": "running", "read_requests": 2}, 2, False),
    (False, {"task_id": "t1", "status": "failed", "read_requests": 2}, 2, False),
    (False, {"task_id": "t1", "status": "running", "read_requests": 3}, 2, False),
    (True, {"task_id": "t1", "status": "running"}, 2, None),
    (False, {"task_id": "t1", "status": "running", "read_requests": 2}, 2, None),
    (False, {}, 2, None),
    (True, {}, 2, None),
    (True, {"task_id": "t1", "status": "running", "read_requests": 2}, None, None),
])
def test_continuation_needs_a_tied_request_and_a_quiet_timeout_is_not_evidence_of_failure(resumed, live, reads, expect):
    assert renewal.continuation(resumed, live, "t1", reads) is expect


@pytest.mark.parametrize("sample, expected", [
    ({}, True), ({"read_requests": 2}, False), ({"held_requests": 2}, False),
    ({"requests_in_generation": 1}, False), ({"read_requests": None}, None),
    ({"held_requests": 0}, None), ({"pause_cause": "quota"}, False),
    ({"phase": None, "activities_complete": False}, None),
    ({"phase": None, "activities_complete": False, "read_requests": 2}, False),
    # the gateway's own typed "unknown" (fence or reason unreadable) is missing evidence while the counters stand still
    ({"phase": "unknown", "activities_complete": False}, None),
    ({"phase": "budget_paused", "pause_cause": "unknown"}, None),
    ({"phase": "unknown", "activities_complete": False, "read_requests": 2}, False),
    ({"phase": "unknown", "activities_complete": False, "held_requests": 2}, False),
    ({"phase": "unknown", "activities_complete": False, "requests_in_generation": 1}, False),
    # a known contradiction is still a contradiction, and "unknown" never swallows an ordinary negative
    ({"phase": "budget_paused", "pause_cause": "restart"}, False),
    ({"phase": "running", "pause_cause": None}, False),
    ({"phase": "budget_pausing"}, False),
    ({"activity_id": "other"}, False),
])
def test_owner_pause_requires_unchanged_read_and_held_counters_in_the_same_sample(sample, expected):
    owner_pause = {"activity_id": "p", "phase": "budget_paused", "pause_cause": "owner", "read_requests": 1,
                   "held_requests": 1, "requests_in_generation": 0, "activities_complete": True}
    paused = {**owner_pause, **sample}
    assert journeys.pause_kept(paused, "p", owner_pause) is expected


def test_task_contradictions_stay_failures_when_the_baseline_proof_is_unavailable():
    for key in ("restart.pause_kept", "update.live_continued"):
        assert _judge(_baseline(**{key: None}), _arm(**{key: False}))["verdict"] == "FAIL"
        assert _judge(_baseline(**{key: None}), _arm(**{key: None}))["verdict"] == "INCONCLUSIVE"


def test_cold_relaunch_keeps_the_pause_sample_that_showed_probe_activity(monkeypatch):
    arm = object.__new__(renewal.Arm)
    before = {"activity_id": "p", "phase": "budget_paused", "pause_cause": "owner", "read_requests": 1,
              "held_requests": 1, "requests_in_generation": 0, "activities_complete": True}
    after = dict(before, read_requests=2)
    arm.task_ids, arm.checks, arm.facts = {"paused": "p"}, {"session": {}}, {"tasks_before": {"paused": before}}
    arm.task_facts = lambda: {"paused": after}
    monkeypatch.setattr(renewal, "wait_for", lambda read, seconds: read())
    arm.pause_after_quit()
    assert arm.facts["paused_after_quit"] == after and arm.checks["session"]["pause_kept_after_quit"] is False


def test_cold_relaunch_polls_only_unknown_and_keeps_the_first_conclusive_sample(monkeypatch):
    owner = {"activity_id": "p", "phase": "budget_paused", "pause_cause": "owner", "read_requests": 1,
             "held_requests": 1, "requests_in_generation": 0, "activities_complete": True}
    unreadable = dict(owner, phase="unknown", activities_complete=False)
    advanced = dict(owner, read_requests=2)

    def run(samples):
        arm = object.__new__(renewal.Arm)
        arm.task_ids, arm.checks, arm.facts = {"paused": "p"}, {"session": {}}, {"tasks_before": {"paused": owner}}
        reads = iter(samples)
        arm.task_facts = lambda: {"paused": next(reads)}
        clock = [0.0]
        monkeypatch.setattr(renewal, "time", SimpleNamespace(
            monotonic=lambda: clock[0], sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds)))
        arm.pause_after_quit()  # the real polling seam: renewal.wait_for with its own deadline
        return arm, reads

    # unknown is retried; a definite false is final even though a later poll would read true
    arm, reads = run([unreadable, advanced, owner, owner])
    assert arm.checks["session"]["pause_kept_after_quit"] is False and arm.facts["paused_after_quit"] == advanced
    assert next(reads) == owner  # polling stopped at the conclusive sample
    # a first true is final too, with its own sample, and a later advance is never read
    arm, _ = run([unreadable, owner, advanced])
    assert arm.checks["session"]["pause_kept_after_quit"] is True and arm.facts["paused_after_quit"] == owner
    # nothing conclusive by the deadline: the last fresh sample stays missing evidence, never a pass or a failure
    arm, _ = run([unreadable] * 200)
    assert arm.checks["session"]["pause_kept_after_quit"] is None and arm.facts["paused_after_quit"] == unreadable


def test_quit_settles_only_on_observed_server_generations_that_ended(monkeypatch):
    arm = object.__new__(renewal.Arm)
    arm.servers = []
    monkeypatch.setattr(renewal, "alive", lambda generation: False)
    assert arm.servers_gone() is False  # no identity to prove release from: never a vacuous settle
    arm.servers = [{"pid": 1, "birth": "a"}]
    assert arm.servers_gone() is True
    monkeypatch.setattr(renewal, "alive", lambda generation: None)
    assert arm.servers_gone() is False  # an unreadable probe is not an exit


def test_the_untied_window_of_each_transition_opens_immediately_before_its_request():
    arm = object.__new__(renewal.Arm)
    arm.model = SimpleNamespace(attribution=lambda: {"basis": {}, "unattributed": {"live": 2, "paused": 0}})
    arm.generation, arm.checks, arm.facts = {"pid": 1, "birth": "a"}, {"fixture": {}}, {}
    arm.api = lambda method, route, body, timeout: (0, None)
    arm.wait_gone = lambda gen: True
    arm.wait_generation = lambda gen: {"pid": 2, "birth": "b"}
    arm.generation_step("fixture")
    assert arm.unattributed_before == {"live": 2, "paused": 0}  # copied from the counters before the request


def test_an_early_untied_arrival_during_the_transition_keeps_the_check_missing():
    arm = object.__new__(renewal.Arm)
    arm.task_ids = {"live": "t1"}
    arm.generation = {"pid": 2, "birth": "b"}
    arm.unattributed_before = {"live": 1, "paused": 0}  # the boundary snapshot, taken before the request
    arm.model = SimpleNamespace(attribution=lambda: {"basis": {}, "unattributed": {"live": 2, "paused": 0}},
                                activity=lambda *_: {"held_requests": 2, "read_requests": 2, "requests_in_generation": 0})
    arm.api = lambda method, route, timeout=15.0: (200, {"active_chat_activities": [],
                                                        "active_chat_activities_complete": True})
    untied = arm.task_facts()["live"]["unattributed_in_generation"]
    assert untied == 1  # the arrival during the transition counts, not only one after confirmation
    assert renewal.continuation(False, {"task_id": "t1", "status": "running", "read_requests": 2},
                                "t1", 2) is None  # missing attribution, never a demonstrated stop
    arm.unattributed_before = {"live": 2, "paused": 0}  # the next transition re-snapshots the grown counts
    assert arm.task_facts()["live"]["unattributed_in_generation"] == 0


def test_the_native_send_proof_needs_the_appkit_drive_and_the_visible_accepted_room_row(monkeypatch):
    arm = object.__new__(renewal.Arm)
    wire = {"client_message_id": "id-1", "sender_session_id": "s"}
    address = "row:17@x#0123456789ab"
    send = {"drive": "appkit", "wire": wire, "project": "p", "accepted_address": address}
    arm.facts = {"update_project_send": send}
    b = {"count": 1, "visible_accepted": True}
    def read(msg_id, prefix, *, address):
        assert (msg_id, prefix, address) == ("id-1", "#pchat-p", send["accepted_address"])
        return b
    arm.win = SimpleNamespace(bubble=read)
    monkeypatch.setattr(renewal, "wait_for", lambda read, *_: read())
    assert arm.native_send_proof("update") is True
    b = {"count": 2, "visible_accepted": False}
    assert arm.native_send_proof("update") is False
    b = {"count": 1, "visible_accepted": False}
    assert arm.native_send_proof("update") is False  # a bubble bound to another address
    b = {"count": 0, "visible_accepted": False}
    assert arm.native_send_proof("update") is False
    b = {"count": 1, "visible_accepted": None, "visibility": "transparent"}
    assert arm.native_send_proof("update") is None  # present but not shown: unestablished, never counted
    assert send["bubble"] == b
    b = None
    assert arm.native_send_proof("update") is None
    send.update(drive=None)
    assert arm.native_send_proof("update") is None  # an uncertified drive never passes as native
    send.update(drive="appkit", wire=wire, accepted_address=None)
    assert arm.native_send_proof("update") is None
    send.update(drive="appkit", wire=None, accepted_address=address)
    assert arm.native_send_proof("update") is None
    assert arm.native_send_proof("handoff") is None  # a phase that never sent stays missing


BIRTH = "Tue Nov 18 10:00:00 2026"
PROBE_BODY = {"messages": [{"role": "tool", "content": "probe result LIVE-tg"}]}


def _journey(tmp_path, monkeypatch, *, owner, chain, births, binding=None, known=()):
    monkeypatch.setattr(journeys, "connection_owner", lambda port: owner)
    monkeypatch.setattr(journeys, "process_ancestry", lambda pid, limit=12: chain)
    monkeypatch.setattr(journeys, "ps", lambda pid, column: births.get(pid, ""))
    workspaces = {}
    for key in journeys.TASKS:
        workspaces[key] = tmp_path / key
        workspaces[key].mkdir()
    path = tmp_path / "bindings.json"
    if binding is not None:
        path.write_text(json.dumps(binding), encoding="utf-8")
    model = journeys.journey_model(workspaces, "tg", path, lambda: list(known))
    model.release["live"].set()
    model._peer.value = ("127.0.0.1", 51234)  # what the handler thread records before the gate runs
    return model


def _binding(pid=4242, birth=BIRTH):
    return {"main": {"pid": pid, "port": 49999, "fingerprint": {"start_time": birth}}}


def test_the_measured_parent_chain_binds_a_held_request_to_its_generation(tmp_path, monkeypatch):
    with _journey(tmp_path, monkeypatch, owner=5150, chain=[5150, 4242, 999], births={4242: BIRTH},
                  binding=_binding()) as model:
        model._gate(PROBE_BODY)
        assert model.count("live", {"pid": 4242, "birth": BIRTH}) == 1
        assert model.attribution() == {"basis": {"parent_chain": 1}, "unattributed": {"live": 0, "paused": 0}}


def test_the_arrival_view_alone_never_attributes_a_request(tmp_path, monkeypatch):
    # The binding names a live generation, but the requester is unresolved: missing, never the arrival view.
    with _journey(tmp_path, monkeypatch, owner=None, chain=[], births={}, binding=_binding()) as model:
        model._gate(PROBE_BODY)
        assert model.count("live", {"pid": 4242, "birth": BIRTH}) == 0
        assert model.count("live") == 1
        assert model.attribution()["unattributed"]["live"] == 1
        assert model.attribution()["basis"] == {"requester_unresolved": 1}


def test_an_orphan_of_a_dead_generation_stays_unattributed(tmp_path, monkeypatch):
    with _journey(tmp_path, monkeypatch, owner=5150, chain=[5150, 1], births={},
                  binding=_binding(pid=4343, birth="Tue Nov 18 11:00:00 2026")) as model:
        model._gate(PROBE_BODY)
        assert model.count("live") == 1 and model.attribution()["unattributed"]["live"] == 1
        assert model.attribution()["basis"] == {"generation_not_in_chain": 1}


def test_a_reused_pid_without_its_birth_never_counts(tmp_path, monkeypatch):
    with _journey(tmp_path, monkeypatch, owner=5150, chain=[5150, 4242], births={4242: "another birth"},
                  binding=_binding()) as model:
        model._gate(PROBE_BODY)
        assert model.count("live", {"pid": 4242, "birth": BIRTH}) == 0
        assert model.attribution()["unattributed"]["live"] == 1


def test_a_driver_confirmed_generation_binds_even_when_the_binding_read_fails(tmp_path, monkeypatch):
    with _journey(tmp_path, monkeypatch, owner=5150, chain=[5150, 4242, 999], births={4242: BIRTH},
                  binding=None, known=[(4242, BIRTH)]) as model:
        model._gate(PROBE_BODY)
        assert model.count("live", {"pid": 4242, "birth": BIRTH}) == 1
        assert model.attribution()["unattributed"]["live"] == 0


def test_the_request_thread_carries_its_peer_to_the_gate(tmp_path, monkeypatch):
    with _journey(tmp_path, monkeypatch, owner=None, chain=[], births={}) as model:
        seen = []
        model._serve_post = lambda request: seen.append(model._peer.value)
        request = SimpleNamespace(connection=SimpleNamespace(getpeername=lambda: ("127.0.0.1", 51234)))
        model._server.RequestHandlerClass.do_POST(request)
        assert seen == [("127.0.0.1", 51234)]
        assert model._peer.value is None  # the peer never leaks into the next request on this thread


def _roster(*keys, exclude=""):
    """The host's own roster note, rendered by the real producer, over roots titled by their submitted description."""
    from ouroboros.context_budget import bound_host_row
    from ouroboros.peer_roster import render_roster_note

    roots = [{"task_id": f"task-{key}", "title": journeys.task_description(key), "chat_id": 0, "project_id": "",
              "status": "running"} for key in keys]
    return bound_host_row("ouroboros_roster_snapshot", render_roster_note({"roots": roots}, exclude=exclude))


def _request(*rows):
    return {"model": "stub", "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
            "messages": [{"role": "system", "content": "You are Ouroboros."}, *rows]}


def _own(key):
    return {"role": "user", "content": journeys.task_description(key)}


def test_a_task_is_bound_by_its_originating_message_not_by_the_host_roster_tail(tmp_path, monkeypatch):
    with _journey(tmp_path, monkeypatch, owner=None, chain=[], births={}) as model:
        # The second task's first request ends with a roster naming the first: the last marker is NOT its own.
        assert journeys.TASKS == ("live", "paused")
        paused_first = _request(_own("paused"), _roster("live", "paused", exclude="task-paused"))
        assert model._key(paused_first) == "paused"
        assert model._key(_request(_own("live"), _roster("live", "paused", exclude="task-live"))) == "live"
        assert model._key(_request(_own("live"))) == "live"
        # After its own probe read, the tool lineage binds even when the roster tail names the other task.
        read = {"role": "tool", "tool_call_id": "c1", "content": model.tokens["paused"]}
        assert model._key(_request(_own("paused"), read, _roster("live", exclude="task-paused"))) == "paused"
        # A text-block user row and a restored transcript keep their own opening marker.
        blocks = {"role": "user", "content": [{"type": "text", "text": journeys.task_description("live")}]}
        assert model._key(_request(blocks, _roster("paused"))) == "live"


def test_an_unrelated_request_is_never_given_a_test_tasks_name(tmp_path, monkeypatch):
    with _journey(tmp_path, monkeypatch, owner=None, chain=[], births={}) as model:
        roster = _roster("live", "paused")
        main = _request({"role": "user", "content": "How are my tasks going?"}, roster)
        project = _request({"role": "user", "content": "In this room, quote [E2E-LINEAGE:live] as plain text."}, roster)
        quoted = _request({"role": "user", "content": "Ask me again."}, {"role": "assistant", "content": "ok"},
                          {"role": "tool", "tool_call_id": "c1", "content": journeys.task_description("paused")})
        assert [model._key(body) for body in (main, project, quoted)] == ["", "", ""]
        model.release["paused"].clear()
        model._gate(main)  # an unrelated turn is answered at once, never held and never counted
        assert model.count("paused") == 0 and model.count("live") == 0 and model.attribution()["basis"] == {}


def _send_room(armed=True, saw=None, wrote=None, key_error=None, key_result=True):
    calls = []
    def page(script, **data):
        if "__ouroWs" in script:
            return True
        if "field.focus()" in script:
            calls.append("arm")
            return armed
        if "__renewalSend?.seen" in script:
            return saw
        if "__renewalWireCounts" in script:
            return wrote
        if "restore?.()" in script:
            calls.append("disarm")
            return True
        raise AssertionError(script)
    def call(op, **args):
        if op != "key":
            raise AssertionError(op)
        calls.append(("key", args["text"]))
        if key_error is not None:
            raise key_error
        return key_result
    sent = []
    def send_once(text, hold, prefix, target, *, from_draft):
        sent.append((text, hold, from_draft))  # no substitution exists: any call here fails a test below
        return {"client_message_id": "substituted", "sender_session_id": "never"}
    win = SimpleNamespace(page=page, call=call, until=lambda read, seconds=20: read(),
                          send_once=send_once, type=lambda *a, **k: calls.append("type") or True)
    win.disarm_send_capture = lambda: native_window.RoomWindow.disarm_send_capture(win)
    return win, calls, sent


WIRE = {"type": "chat", "content": "Project-seed-tg", "client_message_id": "id-1", "sender_session_id": "s"}


def test_one_native_return_drives_the_ordinary_send_and_keeps_its_frame():
    win, calls, sent = _send_room(saw=dict(WIRE))
    result = native_window.RoomWindow.send_native(win, "Project-seed-tg", "#pchat-p", {"port": 1}, from_draft=True)
    assert result == {"wire": WIRE, "drive": "appkit", "disclosure": None}
    assert calls == ["arm", ("key", "\r"), "disarm"] and sent == []


def test_a_seed_send_types_natively_before_the_return_key():
    win, calls, sent = _send_room(saw=dict(WIRE))
    result = native_window.RoomWindow.send_native(win, "Project-seed-tg", "#pchat-p", {"port": 1}, from_draft=False)
    assert result["drive"] == "appkit" and calls[0] == "type" and ("key", "\r") in calls
    assert sent == []


def test_a_lost_key_reply_with_a_captured_frame_keeps_the_wire_without_certifying_the_drive():
    win, calls, sent = _send_room(saw=dict(WIRE), key_error=RuntimeError("no answer for key; outcome unknown"))
    result = native_window.RoomWindow.send_native(win, "Project-seed-tg", "#pchat-p", {}, from_draft=True)
    assert result == {"wire": WIRE, "drive": None, "disclosure": "RuntimeError"}
    assert sent == [] and "disarm" in calls


def test_a_key_answer_without_its_typed_effect_never_certifies_native_input():
    win, calls, sent = _send_room(saw=dict(WIRE), key_result=False)
    result = native_window.RoomWindow.send_native(win, "Project-seed-tg", "#pchat-p", {}, from_draft=True)
    assert result == {"wire": WIRE, "drive": None, "disclosure": "key_outcome_uncertain"}
    assert sent == []


def test_a_write_without_a_captured_frame_is_never_repeated():
    win, calls, sent = _send_room(saw=None, wrote=1)
    result = native_window.RoomWindow.send_native(win, "Project-seed-tg", "#pchat-p", {}, from_draft=True)
    assert result == {"wire": None, "drive": "appkit", "disclosure": "frame_not_captured", "writes": 1}
    assert sent == [] and "disarm" in calls


def test_a_proven_zero_write_stays_missing_evidence_never_a_substitution():
    win, calls, sent = _send_room(saw=None, wrote=0)
    result = native_window.RoomWindow.send_native(win, "Project-seed-tg", "#pchat-p", {}, from_draft=True)
    assert result == {"wire": None, "drive": "appkit", "disclosure": "native_key_did_not_submit", "writes": 0}
    assert sent == [] and "disarm" in calls


def test_an_unreadable_write_counter_stays_unknown_never_a_zero():
    win, calls, sent = _send_room(saw=None, wrote=None)
    result = native_window.RoomWindow.send_native(win, "Project-seed-tg", "#pchat-p", {}, from_draft=True)
    assert result == {"wire": None, "drive": "appkit", "disclosure": "write_counter_unavailable"}
    assert sent == [] and "disarm" in calls


def test_an_unavailable_native_action_is_never_substituted_or_retried():
    win, calls, sent = _send_room(saw=None, wrote=None, key_error=RuntimeError("observer gone"))
    result = native_window.RoomWindow.send_native(win, "Project-seed-tg", "#pchat-p", {}, from_draft=True)
    assert result == {"wire": None, "drive": None, "disclosure": "RuntimeError"}
    assert sent == [] and "disarm" in calls


def test_an_ineligible_field_or_a_stale_capture_never_arms_a_send():
    win, calls, sent = _send_room(armed=False)
    result = native_window.RoomWindow.send_native(win, "Project-seed-tg", "#pchat-p", {}, from_draft=True)
    assert result == {"wire": None, "drive": None, "disclosure": "field_ineligible"}
    assert ("key", "\r") not in calls and sent == []
    win, calls, sent = _send_room(armed=None)
    result = native_window.RoomWindow.send_native(win, "Project-seed-tg", "#pchat-p", {}, from_draft=True)
    assert result["disclosure"] == "capture_unavailable" and ("key", "\r") not in calls and sent == []


def test_the_native_return_key_carries_its_key_code():
    assert native_window.KEY_CODES["\r"] == 36


PROBE_RUNNER = """
const spec = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const box = r => ({left: r[0], top: r[1], right: r[2], bottom: r[3], width: r[2] - r[0], height: r[3] - r[1]});
const inside = (r, x, y) => x >= r[0] && x < r[2] && y >= r[1] && y < r[3];
const style = s => ({display: 'block', visibility: 'visible', opacity: '1', ...s});
const feed = {parentElement: null, style: style({}), getBoundingClientRect: () => box(spec.feed)};
const wrapper = {parentElement: feed, style: style(spec.wrapper || {})};
const text = {};
const rows = spec.rows.map(r => ({parentElement: wrapper, style: style(r.style), dataset: {clientMessageId: r.id, rowAddress: r.address},
    getBoundingClientRect: () => box(r.rect), getClientRects: () => (r.rects === false ? [] : [box(r.rect)]),
    closest: () => feed, contains: node => node === text || node === rows[0], querySelectorAll: () => [], rect: r.rect}));
globalThis.innerWidth = spec.viewport[0]; globalThis.innerHeight = spec.viewport[1];
globalThis.getComputedStyle = node => node.style;
globalThis.document = {querySelectorAll: () => rows, elementFromPoint: (x, y) =>
    (spec.covers || []).some(c => inside(c, x, y)) ? {overlay: true} : rows[0] && inside(rows[0].rect, x, y) ? text : {body: true}};
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
new AsyncFunction('a', process.argv[1])(spec.args).then(r => console.log(JSON.stringify(r)));
"""


def _probe(tmp_path, *, rows=None, address="a1", **spec):
    import shutil as _shutil
    import subprocess

    node = _shutil.which("node")
    assert node, "Node is required to run the page-side bubble probe over its synthetic DOM"
    row = {"id": "m1", "address": "a1", "rect": [20, 100, 220, 160]}
    spec = {"feed": [0, 0, 400, 300], "viewport": [400, 300], **spec,
            "rows": [row] if rows is None else rows,
            "args": {"id": "m1", "feed": "#pchat-p-messages", "address": address,
                     "sel": "#pchat-p-messages .chat-bubble.user[data-client-message-id]"}}
    done = subprocess.run([node, "-e", PROBE_RUNNER, native_window.BUBBLE_PROBE], input=json.dumps(spec),
                          capture_output=True, text=True, timeout=30, cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@pytest.mark.parametrize("row, spec, expected", [
    ({}, {}, (True, "exposed")),
    ({"style": {"opacity": "0.4"}}, {}, (None, "transparent")),  # the send animation's own fade-in
    ({"style": {"opacity": "1"}}, {"wrapper": {"opacity": "0"}}, (None, "transparent")),
    ({}, {"wrapper": {"opacity": "0.97"}}, (True, "exposed")),
    ({"style": {"visibility": "hidden"}}, {}, (None, "hidden")),
    ({}, {"wrapper": {"display": "none"}}, (None, "hidden")),
    ({"rects": False}, {}, (None, "hidden")),
    ({"rect": [20, 400, 220, 460]}, {}, (None, "clipped")),  # below the feed's own clip, though inside no viewport
    ({"rect": [20, -40, 220, 20]}, {}, (True, "exposed")),  # partly scrolled out: the shown part counts
    ({}, {"covers": [[0, 0, 400, 300]]}, (None, "covered")),  # intersects the viewport, but no point is exposed
    ({}, {"covers": [[0, 130, 400, 300]]}, (True, "exposed")),  # a composer over the lower half leaves exposed points
    ({}, {"feed": [0, 0, 400, 90]}, (None, "clipped")),
])
def test_the_acknowledged_bubble_must_actually_be_shown_not_only_intersect_the_viewport(tmp_path, row, spec, expected):
    seen = _probe(tmp_path, rows=[{"id": "m1", "address": "a1", "rect": [20, 100, 220, 160], **row}], **spec)
    assert (seen["visible_accepted"], seen["visibility"]) == expected and seen["count"] == 1


def test_the_bubble_probe_keeps_duplicates_wrong_addresses_and_the_retry_shape_apart(tmp_path):
    row = {"id": "m1", "address": "a1", "rect": [20, 100, 220, 160]}
    assert _probe(tmp_path, rows=[row, dict(row)]) == {"count": 2, "visible_accepted": False}
    assert _probe(tmp_path, rows=[dict(row, address="other")]) == {"count": 1, "visible_accepted": False}
    assert _probe(tmp_path, rows=[]) == {"count": 0, "visible_accepted": False}
    assert _probe(tmp_path, rows=[row], address=None) == {"count": 1, "retry": False}


def test_the_echo_probe_reads_the_room_it_is_asked_about():
    seen = {}
    win = SimpleNamespace(page=lambda script, **data: seen.update(data) or {"count": 1, "retry": False})
    native_window.RoomWindow.bubble(win, "id-1", "#pchat-p")
    assert seen["sel"] == "#pchat-p-messages .chat-bubble.user[data-client-message-id]"
    native_window.RoomWindow.bubble(win, "id-1")
    assert seen["sel"] == "#chat-messages .chat-bubble.user[data-client-message-id]"


def test_the_stored_quote_travels_complete_for_an_exact_in_memory_comparison():
    quote = {"address": "row:17@x#0123456789ab", "chat_id": 17, "role": "user", "author_label": "You",
             "kind": "message", "snippet": "Project-seed-tg", "unresolved": False, "ts": "2026-11-21T00:00:00Z",
             "task_id": "", "media": {"kind": "image", "available": True, "url": "/uploads/photo.png"}}
    slots = {"reply": dict(quote, address="row:1@x#0123456789ab", chat_id=1), "project_reply": dict(quote)}
    win = SimpleNamespace(custody=lambda marker, room: slots)
    assert native_window.RoomWindow.stored_quote(win, "m", 17) == quote  # every persisted field travels
    assert native_window.RoomWindow.stored_quote(win, "m", None) == slots["reply"]
    win.custody = lambda marker, room: None
    assert native_window.RoomWindow.stored_quote(win, "m", None) is None
    win.custody = lambda marker, room: {"reply": "corrupt"}
    assert native_window.RoomWindow.stored_quote(win, "m", None) is None


def test_a_restored_quote_needs_its_complete_shape_and_its_visible_jump():
    quote = {"address": "row:17@x#0123456789ab", "chat_id": 17, "snippet": "Project-seed-tg"}
    shown = {"author": "You", "when": "· 09:36", "text": "Project-seed-tg",
             "label": "Replying to You, 09:36: Project-seed-tg"}
    jumps, current = [], [dict(quote)]
    win = SimpleNamespace(follows_quote=lambda addr, prefix, text: jumps.append((addr, prefix, text)) or True,
                          stored_quote=lambda marker, room: current.pop() if current else None,
                          composer_quote=lambda _: dict(shown))
    staged = {"saved_quote": dict(quote), "room_id": 17, "quote_text": "Project-seed-tg", "shown_quote": dict(shown)}
    assert native_window.RoomWindow.restored_quote(win, staged, "m", quote["address"], "#pchat-p") is True
    assert jumps == [(quote["address"], "#pchat-p", "Project-seed-tg")]
    current.append(dict(quote, snippet="changed"))  # any single-field difference is demonstrated
    assert native_window.RoomWindow.restored_quote(win, staged, "m", quote["address"], "#pchat-p") is False
    assert native_window.RoomWindow.restored_quote(win, staged, "m", quote["address"], "#pchat-p") is None
    other = {"address": "row:1@x#0123456789ab", "chat_id": 17}
    current.append(dict(other))  # a complete quote of a different row is still false
    staged_other = {"saved_quote": dict(other), "room_id": 17}
    assert native_window.RoomWindow.restored_quote(win, staged_other, "m", quote["address"], "#pchat-p") is False
    assert native_window.RoomWindow.restored_quote(win, {}, "m", quote["address"], "#pchat-p") is None
    for display, expect in ((dict(shown, author="Other"), False), (dict(shown, when="10:51"), False),
                            (dict(shown, text="Wrong"), False), (False, False), (None, None)):
        current.append(dict(quote))
        win.composer_quote = lambda _: display
        assert native_window.RoomWindow.restored_quote(win, staged, "m", quote["address"], "#pchat-p") is expect
    assert jumps == [(quote["address"], "#pchat-p", "Project-seed-tg")]  # storage alone cannot certify the strip


@pytest.mark.parametrize("day,clock,label_clock,verdict", [
    ("Yesterday, ", "23:58", "23:58", True),
    ("Nov 20, ", "23:58", "23:58", True),
    ("Yesterday, ", "23:59", "23:59", False),
    ("Yesterday, ", "23:58", "23:59", False),
    ("Yesterday, ", None, "23:58", False),
])
def test_restored_quote_compares_the_visible_clock_across_days(day, clock, label_clock, verdict):
    address = "row:17@x#0123456789ab"
    quote = {"address": address, "ts": "2026-11-20T23:58:00Z", "snippet": "Seed: 23:58"}
    shown = {"author": "You", "when": "· 23:58", "text": "Seed: 23:58", "thumb": False,
             "label": "Replying to You, 23:58: Seed: 23:58"}
    staged = {"saved_quote": dict(quote), "room_id": 17, "shown_quote": shown}
    restored = dict(shown, when=f"· {day}{clock}" if clock else None,
                    label=f"Replying to You, {day}{label_clock}: Seed: 23:58")
    jumps = []
    win = SimpleNamespace(stored_quote=lambda *_: dict(quote), composer_quote=lambda _: restored,
                          follows_quote=lambda *args: jumps.append(args) or True)
    assert native_window.RoomWindow.restored_quote(win, staged, "m", address, "#pchat-p") is verdict
    assert bool(jumps) is verdict


def test_a_restored_quote_with_a_changed_attachment_reference_is_demonstrated_different():
    """No native fixture stages an attached quote — the scenario replies to text rows — so this
    portable shape pins the attachment-bearing comparison: a restored quote that swapped its
    media reference, or lost it, is a demonstrated difference, never equal under a stripped
    comparison."""
    quote = {"address": "row:17@x#0123456789ab", "chat_id": 17, "role": "user", "author_label": "You",
             "kind": "message", "snippet": "Project-seed-tg", "unresolved": False,
             "ts": "2026-11-21T00:00:00Z", "task_id": "",
             "media": {"kind": "image", "available": True, "url": "/uploads/photo-17.png"}}
    shown = {"author": "You", "when": "· 00:00", "text": "Project-seed-tg", "thumb": True,
             "label": "Replying to You, 00:00: Project-seed-tg"}
    staged = {"saved_quote": dict(quote), "room_id": 17, "quote_text": "Project-seed-tg", "shown_quote": shown}
    current = []
    win = SimpleNamespace(follows_quote=lambda *_: True,
                          stored_quote=lambda marker, room: current.pop() if current else None,
                          composer_quote=lambda _: dict(shown))
    current.append(dict(quote))
    assert native_window.RoomWindow.restored_quote(win, staged, "m", quote["address"], "#pchat-p") is True
    current.append(dict(quote, media={"kind": "image", "available": True, "url": "/uploads/photo-18.png"}))
    assert native_window.RoomWindow.restored_quote(win, staged, "m", quote["address"], "#pchat-p") is False
    current.append(dict(quote, media=None))  # a lost attachment reference is a difference too
    assert native_window.RoomWindow.restored_quote(win, staged, "m", quote["address"], "#pchat-p") is False


def test_an_unavailable_capture_preserves_independent_facts_and_blocks_pass():
    missing_window = {"update.window_pixels": None, "update.window_typed_pixels": None}
    result = _judge(_baseline(), _arm(**missing_window))
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["comparison"]["update.typed"]["candidate"] is True
    assert result["comparison"]["update.typed_pixels"]["candidate"] is True
    assert result["comparison"]["update.live_continued"]["candidate"] is True
    assert _judge(_baseline(), _arm(**missing_window, **{"update.no_replay": False}))["verdict"] == "FAIL"


def test_a_lost_restart_reply_advances_to_observation_without_another_request():
    arm = object.__new__(renewal.Arm)
    old = {"pid": 1, "birth": "a"}
    arm.generation, arm.checks, arm.facts, arm.model = old, {"fixture": {}}, {}, None
    seen = []
    def api(method, route, body, timeout):
        seen.append((method, route, body["cmd"]))
        return 0, None
    arm.api = api
    def exited(gen):
        seen.append(("exit", gen))
        return True
    arm.wait_gone = exited
    arm.wait_generation = lambda gen: {"pid": 2, "birth": "b"}
    arm.generation_step("fixture")
    assert seen == [("POST", "/api/command", "/restart"), ("exit", old)]
    assert arm.facts["fixture_request"]["http"] == 0 and arm.checks["fixture"]["generation_new"]
    assert arm.unattributed_before == {}  # an unprepared arm keeps an empty untied window


def _view(**changes):
    return {"shell": "main", "pid": 1, "window": "one", "number": 9, "view": "a", "installed": True,
            "visible": True, "key": True, "on_screen": True, **changes}


def _settling(monkeypatch, renews_at):
    """An arm whose window shows view "a" until ``renews_at`` seconds, then view "b"; time is the test's own."""
    arm = object.__new__(renewal.Arm)
    clock = [0.0]
    monkeypatch.setattr(renewal, "time", SimpleNamespace(
        monotonic=lambda: clock[0], sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds)))
    arm.generation, arm.gaps, arm.facts = {"port": 5001}, [], {}
    arm.win = SimpleNamespace(state=lambda: _view(view="a" if clock[0] < renews_at else "b", loopback=True, port=5001))
    return arm, clock


def _page_checks():
    return {"generation_new": True, "live_continued": True, "pause_kept": True, "accepted_once": True,
            "view_new": None, "installed": True, "draft_restored": True, "typed": True, "wire_once": False}


@pytest.mark.parametrize("renews_at, final, verdict", [
    (10_000, "a", "confirmed old view"),    # never renews: the baseline's own control, or a candidate that did not renew
    (30, "b", "late view"),                 # renews after the settle window: earlier steps measured the old document
    (10_000, None, "unreadable final"),     # the window cannot be read at the end: no confirmation
    (5, "b", "prompt new view"),            # renewed inside the window: definite
])
def test_a_settle_timeout_on_the_old_view_is_confirmed_only_by_a_final_observation(monkeypatch, renews_at, final, verdict):
    arm, clock = _settling(monkeypatch, renews_at)
    old = _view(view="a", loopback=True, port=5001)
    sampled = arm.settle(old)
    assert sampled["view"] == ("b" if renews_at < 25 else "a")  # an early sample may still be the old view
    checks = _page_checks()
    checks["view_new"] = True if sampled["view"] != old["view"] else None  # exactly what transition() records
    clock[0] = 60
    last = None if final is None else arm.win.state() if final == "b" else _view(view="a")
    arm.reconcile_view("restart", old, sampled, last, checks, tuple(checks))
    if verdict in ("confirmed old view", "prompt new view"):
        assert checks["view_new"] is (False if verdict == "confirmed old view" else True)
        assert checks["draft_restored"] is True and checks["wire_once"] is False and arm.gaps == []
    else:
        assert checks["view_new"] is None and len(arm.gaps) == 1
        # nothing measured on the old document turns green or red; server-owned facts stand
        assert {k: checks[k] for k in ("installed", "draft_restored", "typed", "wire_once")} == dict.fromkeys(
            ("installed", "draft_restored", "typed", "wire_once"))
        assert all(checks[k] is True for k in ("generation_new", "live_continued", "pause_kept", "accepted_once"))


def test_the_settle_outcomes_reach_the_verdict_as_a_control_a_gap_or_a_failure():
    confirmed_old = _baseline(**{"restart.view_new": False})
    assert _judge(confirmed_old, _arm())["verdict"] == "PASS"  # the baseline's confirmed old view is a real control
    # a candidate whose renewal arrived after the timer is unestablished, not a failure
    assert _judge(confirmed_old, _arm(**{"restart.view_new": None}))["verdict"] == "INCONCLUSIVE"
    # a candidate that stayed on the old view to the final observation is a demonstrated failure
    assert _judge(confirmed_old, _arm(**{"restart.view_new": False}))["verdict"] == "FAIL"
    # a baseline whose view later changed no longer calibrates anything
    assert _judge(_baseline(**{"restart.view_new": None}), _arm())["verdict"] == "INCONCLUSIVE"


@pytest.mark.parametrize("phase,ready,mounted,field_ready", [
    ("restart", True, True, False), ("restart", True, False, False), ("restart", True, None, False),
    ("restart", False, None, False), ("update", True, True, True), ("handoff", True, True, True),
    ("update", True, False, False), ("handoff", True, None, False), ("update", True, True, False),
    ("handoff", False, False, False),
])
def test_room_selection_and_native_actions_around_the_single_request(tmp_path, phase, ready, mounted, field_ready):
    arm = object.__new__(renewal.Arm)
    arm.root = tmp_path
    old, new = _view(), _view(view="b", shell="successor" if phase == "handoff" else "main")
    checks = dict.fromkeys(renewal.PHASES[phase])
    arm.checks, arm.generation, arm.facts, arm.gaps = {phase: checks}, {"port": 5001}, {}, []
    arm.project = {"id": "room-sample", "name": "room-sample", "chat_id": 17}
    arm.tag, arm.answer, arm.project_reply, arm.marker = "sample", "row:1@main", "row:17@seed", "fixture"
    arm.task_ids, arm.tree = {}, {"handoff": "revision"}
    arm.slots = lambda: {}
    seen = []
    project_field = "#pchat-room-sample-input"
    def prepare(draft, marker, project):
        assert marker == arm.marker
        if project is None:
            assert draft == {"prefix": "#chat", "reply": arm.answer, "filename": f"Note-{phase}-sample.txt",
                             "text": f"Draft-{phase}-sample"}
        else:
            assert project is arm.project and draft["prefix"] + "-input" == project_field
            assert draft["text"] == f"Project-{phase}-sample"
            assert draft["reply"] == (None if phase == "restart" else arm.project_reply)
            assert bool(draft["filename"]) == (phase != "restart")
        seen.append("prepare Project" if project else "prepare Main")
        draft.update(typed=True, quoted=True, file=True)
    def shown(project, target, *, origin=None):
        if phase == "restart":
            assert project is None and target is new and origin == 42
            assert seen[-1] == "reopen" and "select Main" not in seen
            seen.append("observe landed Main")
            return mounted
        assert project is arm.project
        if target is old:
            seen.append("original room")
            return True
        assert target is new and origin == 42
        seen.append("observe restored room")
        return mounted
    def visible(field, *, editable):
        assert field == project_field and editable
        return "request" not in seen or field_ready
    def send(name, target):
        assert name == phase and target is new
        seen.append("Project Send")
        return True
    arm.win = SimpleNamespace(prepare_draft=prepare, shows_room=shown, visible=visible,
        document=lambda port: {"origin": 42} if ready and port == 5001 else None, state=lambda: new,
        open_project=lambda p: seen.append("select Project") or True, main=lambda: seen.append("select Main") or True,
        until=lambda read: read(), page=lambda script: "revision", confirm_cancel=lambda prompt: True,
        inspect_draft=lambda draft, *_: seen.append("inspect " + draft["prefix"]),
        typing=lambda target, field, label, *_: seen.append("type " + field))
    arm.model = SimpleNamespace(count=lambda *_: 0)
    arm.presentation = lambda _: seen.append("presentation") or ({}, old)
    arm.observe = lambda name, read: read()
    arm.checkpoint = lambda: None
    arm.generation_step = lambda _: seen.append("request")
    arm.settle = lambda before: new
    def reopen(label, observed, target):
        if label == "restart":
            seen.append("reopen")
            checks["reopened"] = True
    arm.after_presentation = reopen
    arm.peer_project = arm.inspect_sends = lambda *_: None
    arm.project_send = send
    arm.native_send_proof = lambda phase: seen.append("native proof") or True
    arm.project_delivery = lambda phase: seen.append("Project receipt") or True
    arm.transition(phase)
    if phase == "restart":
        order = ["prepare Project", "prepare Main", "presentation", "request", "reopen"]
        if ready:
            order.append("observe landed Main")
        order += ["select Main", "inspect #chat", "select Project", "inspect #pchat-room-sample", "select Main"]
        assert checks["main_active_after"] is (mounted if ready else None)
        assert _judge(_baseline(), _arm(**{"restart.main_active_after": checks["main_active_after"]}))["verdict"] == (
            "PASS" if mounted is True else "FAIL" if mounted is False else "INCONCLUSIVE")
    else:
        order = ["prepare Main", "prepare Project", "presentation", "original room", "request"]
        assert checks["project_active_before"] is True
        assert checks["document_ready"] is (True if ready else None)
        assert checks["project_active_after"] is (mounted if ready else None)
        assert checks["project_editable"] is (field_ready if ready and mounted else None)
        if ready:
            order.append("observe restored room")
        if ready and mounted and field_ready:
            order += ["inspect #pchat-room-sample", "Project Send", "Project receipt", "native proof", "type " + project_field]
            assert checks["project_sent_once"] is True and checks["project_send_native_input"] is True
        else:
            assert checks["project_sent_once"] is None and checks["project_send_native_input"] is None
        order += ["select Main", "inspect #chat"]
    assert seen == [*order, "type #chat-input"]
    assert checks["old_draft"] is True


@pytest.mark.parametrize("project_id, chat_id, received", [
    ("peer-update-demo", 17, True), ("other", 17, False), ("peer-update-demo", 18, False), (None, None, False),
], ids=["match", "wrong-project", "wrong-room", "poll-only"])
def test_a_visible_peer_needs_its_own_websocket_frame(project_id, chat_id, received):
    arm = object.__new__(renewal.Arm)
    arm.tag, arm.checks, arm.facts, arm.gaps = "demo", {"update": {}}, {}, []
    seen = []
    def create(name):
        seen.append("create " + name)
        return {"id": name, "name": name, "chat_id": 17}
    def page(script):
        if script.startswith("return"):
            return {"type": "projects_changed", "project_id": project_id, "chat_id": chat_id} if project_id else None
        seen.append("dispose")
    def visible(peer, target, *, in_list):
        return peer["id"] == "peer-update-demo" and in_list
    arm.create_project = create
    arm.win = SimpleNamespace(watch_project=lambda name: seen.append("listen " + name) or True,
                              until=lambda read, _: read(), page=page, shows_room=visible)
    arm.peer_project("update", _view())
    assert seen == ["listen peer-update-demo", "create peer-update-demo", "dispose"]
    assert arm.checks["update"] == {"live_connected": received, "peer_visible": True}


def test_a_failed_reopen_stays_unknown():
    arm = object.__new__(renewal.Arm)
    arm.gaps = []
    def refused(kind, **args):
        raise RuntimeError("/private/observer/path")
    arm.win = SimpleNamespace(action=refused)
    assert arm.reopened("restart reopen", _view()) is None and arm.gaps == ["restart reopen (RuntimeError)"]


def test_update_waits_for_did_exit_before_close_and_the_same_visible_key_view_after_reopen(monkeypatch):
    arm = object.__new__(renewal.Arm)
    arm.checks, arm.gaps = {"update": {}}, []
    current = _view(fullscreen=True, fullscreen_phase="full")
    windowed = _view(fullscreen=False, fullscreen_phase="windowed")
    pending = iter([dict(windowed, fullscreen_phase="exiting"), windowed, dict(windowed, visible=False),
                    dict(windowed, view="other"), dict(windowed, key=False), dict(windowed, on_screen=False),
                    windowed, windowed])  # the final presentation read follows the successful reopen
    state, actions = current, []
    def observe():
        nonlocal state
        state = next(pending)
        return state
    arm.win = SimpleNamespace(state=observe, action=lambda kind: actions.append((kind, state["fullscreen_phase"])))
    wait = renewal.wait_for
    monkeypatch.setattr(renewal, "wait_for", lambda read, seconds, _: wait(read, seconds, 0))
    arm.after_presentation("update", {"fullscreen": current}, current)
    assert actions == [("fullscreen", "full"), ("hide", "windowed"), ("reopen", "windowed")]
    assert arm.checks["update"] == {"fullscreen_kept": True, "reopened": True}
    assert next(pending, None) is None


def test_a_public_reason_keeps_only_the_words_the_fixture_or_observer_wrote():
    def reason(fail):
        try:
            fail()
        except Exception as exc:  # noqa: BLE001 - the reason of whatever was raised
            return native_window.public_reason(exc)
    assert reason(lambda: native_window.AttachedWindow.capture(None, "window", "Not a name")) == \
        "ValueError: invalid capture name"
    assert reason(lambda: json.loads("{")) == "JSONDecodeError"
    def foreign():
        raise RuntimeError("/Users/someone/private/log line")
    assert reason(foreign) == "RuntimeError"


@pytest.fixture
def fixture_source(tmp_path):
    """Keep the real package directory: its name also names the branch bundled by each arm."""
    repo = tmp_path / "repo"
    for name, text in {"VERSION": "9.9.9\n", "server.py": "SERVER = 1\n", "launcher.py": "LAUNCHER = 1\n",
                       "ouroboros/__init__.py": "", "web/app.js": "export {};\n"}.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text, encoding="utf-8")
    for args in (("init", "-q", "-b", "main"), ("add", "-A"), ("commit", "-q", "-m", "frozen")):
        arm_source.git(*args, cwd=repo)
    return repo, arm_source.git("rev-parse", "HEAD", cwd=repo)


@pytest.mark.skipif(os.name == "nt" or shutil.which("git") is None, reason="the macOS instrument's POSIX git and tar")
def test_an_arm_launches_its_exact_tree_and_is_offered_two_comment_only_updates(tmp_path, fixture_source):
    repo, sha = fixture_source
    root = tmp_path / "arm"
    root.mkdir()
    source = arm_source.prepare(repo, sha, root)
    def src(*args):
        return arm_source.git(*args, cwd=source["src"])
    def heads():
        lines = arm_source.git("ls-remote", "--heads", source["mirror"]).splitlines()
        return {ref: commit for commit, ref in (line.split() for line in lines)}
    assert (source["sha"], source["version"]) == (sha, "9.9.9")
    assert src("rev-parse", "HEAD^{tree}") == source["tree"] == arm_source.git("rev-parse", "HEAD^{tree}", cwd=repo)
    assert src("rev-list", "--count", "HEAD") == "1"
    comment = arm_source.UPDATE_COMMENT.decode().strip()
    assert src("diff", "--name-only", source["base"], source["target"]) == "server.py"
    assert src("show", f"{source['target']}:server.py") == "SERVER = 1\n\n" + comment
    assert src("diff", "--name-only", source["target"], source["handoff"]) == "launcher.py"
    assert src("show", f"{source['handoff']}:launcher.py") == "LAUNCHER = 1\n\n" + comment
    assert heads() == {"refs/heads/ouroboros": source["target"], "refs/heads/ouroboros-stable": source["base"],
                       "refs/heads/main": source["base"]}
    manifest = json.loads((source["seed"] / "repo_bundle_manifest.json").read_text(encoding="utf-8"))
    assert (manifest["source_sha"], manifest["release_tag"], manifest["managed_remote_url"]) == (
        source["base"], "", str(source["mirror"]))
    assert manifest["bundle_sha256"] == arm_source.sha256(source["seed"] / "repo.bundle")
    assert set(src("bundle", "list-heads", source["seed"] / "repo.bundle").splitlines()) == {
        f"{source['base']} HEAD", f"{source['base']} refs/heads/ouroboros"}
    arm_source.offer(source, source["handoff"])
    assert heads()["refs/heads/ouroboros"] == source["handoff"]


def test_the_fixture_navigates_once_to_a_marked_address():
    page = renewal.fixture_page(49152, "ab12")
    assert "location.assign('http://127.0.0.1:49152/p-ab12?k-ab12=q-ab12#f-ab12')" in page
    assert renewal.FIXTURE_BUTTON in page and page.count("location.assign") == 1


@pytest.mark.parametrize("outcome", [*native_window.OSLOG_RESULTS, "unknown_status", None])
def test_the_owned_launcher_log_projects_only_the_exact_transition(tmp_path, outcome):
    log = tmp_path / "data/logs/launcher.log"
    log.parent.mkdir(parents=True)
    prefix = "2026-11-19T00:00:00.000Z [INFO] launcher.webkit: WebKit "
    lookup = "process_log_lookup scope=current_process since=10.000 until="
    rows = [prefix + "window_replacement_requested requested_at=15.000", prefix + lookup + "15.000 result=matched",
            prefix + "window_replacement_requested requested_at=20.005", prefix + lookup + "21.000 result=matched",
            "Full raw log message /never/public", prefix + "window_replacement_requested requested_at=45.000"]
    if outcome:
        rows.append(prefix + lookup + "20.005 result=" + outcome)
    log.write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert native_window.launcher_diagnostics(tmp_path, 20, 30) == {
        "window_replacement_requested": True,
        "process_log_lookup": outcome if outcome in native_window.OSLOG_RESULTS else "lookup_unobserved",
    }
    assert native_window.launcher_diagnostics(tmp_path, 30, 40) == {
        "window_replacement_requested": False, "process_log_lookup": "request_unobserved",
    }


def test_an_unreadable_or_oversized_launcher_log_has_no_positive_observation(tmp_path):
    assert native_window.launcher_diagnostics(tmp_path, 20, 30) == {
        "window_replacement_requested": False, "process_log_lookup": "launcher_log_unavailable",
    }
    log = tmp_path / "data/logs/launcher.log"
    log.parent.mkdir(parents=True)
    log.write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    assert native_window.launcher_diagnostics(tmp_path, 20, 30) == {
        "window_replacement_requested": False, "process_log_lookup": "launcher_log_limit",
    }


def _import_staged(path, name):
    """Import one produced file under its own name, as the launcher's exec does: its module-level code reads
    resources beside that copy, never the source tree, and nothing here attaches, captures or starts an app."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_staged_observer_imports_with_the_page_probe_it_reads_at_import(tmp_path):
    arm = tmp_path / "arm"
    arm.mkdir()  # Arm.__init__ creates its own root before the launcher's channel is built
    hooks = native_window.stage_observer(arm)
    staged = hooks / "native_window.py"
    module = _import_staged(staged, "staged_observer")
    assert pathlib.Path(module.__file__).resolve() == staged.resolve()  # the arm's own copy, not the source tree
    assert module.BUBBLE_PROBE == native_window.BUBBLE_PROBE  # the probe every page-side bubble read sends
    assert (hooks / "composer_clearance.py").is_file() and (hooks / "macos_capture.py").is_file()
    (hooks / "bubble_probe.js").unlink()  # the former copy set left that resource in the source tree
    with pytest.raises(FileNotFoundError):
        _import_staged(staged, "staged_observer_without_the_probe")


def test_the_macos_job_runs_only_on_request_and_the_windows_job_keeps_its_triggers():
    workflow = yaml.safe_load((REPO / ".github/workflows/desktop-window.yml").read_text(encoding="utf-8"))
    triggers = workflow.get("on") or workflow.get(True)
    assert triggers["push"] == {"tags": ["v*"]}
    assert triggers["workflow_dispatch"]["inputs"]["macos_window_renewal"]["default"] is False
    assert workflow["jobs"]["native-window"]["if"] == "${{ !inputs.macos_window_renewal }}"
    job = workflow["jobs"]["macos-window-renewal"]
    assert job["if"] == "${{ github.event_name == 'workflow_dispatch' && inputs.macos_window_renewal }}"
    text = yaml.safe_dump(job)
    assert job["runs-on"] == "macos-latest" and "window_renewal.py" in text and "safe_test.py" in text
    assert "secrets." not in text and job["steps"][0]["with"]["persist-credentials"] is False
    assert job["env"] == {"BASELINE_SHA": "${{ inputs.baseline_sha }}", "CANDIDATE_SHA": "${{ inputs.candidate_sha }}"}
    inputs = triggers["workflow_dispatch"]["inputs"]
    assert "default" not in inputs["baseline_sha"] and "default" not in inputs["candidate_sha"]  # chosen per run
    assert set(workflow["jobs"]) == {"native-window", "macos-window-renewal"} and "matrix" not in text
    for name in ("test_desktop_room_session.py", "test_launcher_webkit.py", "test_launcher_webkit_diagnostics.py",
                 "test_macos_source_entry.py", "test_macos_window_renewal.py", "desktop_room_session.test.js"):
        assert name in text
    assert "OURO_SIZE_RATCHET_BASE_REF" in text and "scripts/check_domains.py" in text
    paths = job["steps"][-1]["with"]["path"].splitlines()
    assert [path.split("/report/")[1] for path in paths] == ["report.json", "baseline/*.png", "candidate/*.png"]


def _driver_environment(monkeypatch, tmp_path):
    """The real entry point off-macOS: its own disposable arms root and a darwin view."""
    monkeypatch.setenv("OUROBOROS_TEST_TEMP_ROOT", str(tmp_path / "arms"))
    (tmp_path / "arms").mkdir()
    monkeypatch.setattr(sys, "platform", "darwin")
    return tmp_path / "report"


def test_every_module_and_name_the_native_driver_imports_lazily_resolves():
    """Check the imports used only by the opt-in driver before trying the real window."""
    harness = importlib.import_module("tests.system_e2e.harness")
    for name in ("ScriptedStubModel", "classify_call", "default_lineage_binder", "message_text", "scripted_completion",
                 "PROXY_ENV_KEYS", "STRIPPED_PROVIDER_ENV_KEYS", "assert_settings_keyless", "keyless_settings",
                 "write_settings_file", "ArtifactOracle", "wait_durable_result"):
        assert getattr(harness, name, None) is not None, name
    environment = importlib.import_module("ouroboros.test_environment")
    for name in ("isolated_environment", "retain_tree"):
        assert callable(getattr(environment, name, None)), name
    for module in ("devtools.macos_wkwebview.host_processes", "devtools.macos_wkwebview.journey_model",
                   "devtools.macos_wkwebview.typing_evidence"):
        assert importlib.import_module(module) is not None, module


def test_a_harness_import_failure_lands_in_the_report_instead_of_past_it(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "tests.system_e2e.harness", None)
    out = _driver_environment(monkeypatch, tmp_path)
    assert renewal.main(["--baseline", "a" * 40, "--candidate", "b" * 40, "--out", str(out)]) == \
        renewal.EXIT_CODES["INCONCLUSIVE"]
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["finished_at"] and report["judgement"]["verdict"] == "INCONCLUSIVE"
    for arm in ("baseline", "candidate"):
        assert report["arms"][arm]["phase"] == "prepare"
        assert "prepare stopped (ModuleNotFoundError)" in report["arms"][arm]["gaps"]
    assert "halted" not in json.dumps(report)  # no raw import-machinery message reaches the artifact


def test_an_escape_past_the_arm_is_typed_redacted_and_the_report_is_finished(monkeypatch, tmp_path):
    def escape(self):
        raise RuntimeError("/Users/someone/private/runner log")
    monkeypatch.setattr(renewal.Arm, "run", escape)
    out = _driver_environment(monkeypatch, tmp_path)
    assert renewal.main(["--baseline", "a" * 40, "--candidate", "b" * 40, "--out", str(out)]) == \
        renewal.EXIT_CODES["INCONCLUSIVE"]
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert set(report["arms"]) == {"baseline"}  # its uncertain exit prevents starting the next arm
    assert report["finished_at"] and report["judgement"]["verdict"] == "INCONCLUSIVE"
    assert report["judgement"]["gaps"][0] == "baseline stopped before its final observation (RuntimeError)"
    assert "/Users/someone" not in json.dumps(report)  # the private path never reaches the public artifact


def test_a_cleanup_failure_keeps_the_recorded_cause_and_the_final_report(monkeypatch, tmp_path):
    def refuse(self):
        raise RuntimeError("git /private/runner/fetch failed")
    monkeypatch.setattr(renewal.Arm, "prepare", refuse)
    def no_space(root, **_):
        raise OSError("no space left on /private/device")
    monkeypatch.setattr(renewal.shutil, "rmtree", no_space)
    out = _driver_environment(monkeypatch, tmp_path)
    assert renewal.main(["--baseline", "a" * 40, "--candidate", "b" * 40, "--out", str(out)]) == \
        renewal.EXIT_CODES["INCONCLUSIVE"]
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["finished_at"] and report["judgement"]["verdict"] == "INCONCLUSIVE"
    gaps = report["arms"]["baseline"]["gaps"]
    assert "prepare stopped (RuntimeError)" in gaps and "cleanup (OSError)" in gaps
    assert "fixture exit not confirmed; directory retained by safe_test" not in gaps  # the arm was settled
    assert "/private" not in json.dumps(report)  # neither failure's message reaches the artifact


@pytest.mark.skipif(os.name == "nt" or shutil.which("git") is None, reason="the macOS instrument's POSIX git and tar")
def test_prepare_builds_the_arms_source_an_isolated_environment_and_a_keyless_loopback_model(tmp_path, monkeypatch, fixture_source):
    """Exercise the same seed, private environment, loopback model and settings as the native run, without a GUI."""
    from tests.system_e2e.harness import PROXY_ENV_KEYS, STRIPPED_PROVIDER_ENV_KEYS, assert_settings_keyless
    repo, sha = fixture_source
    arms = tmp_path / "arms"
    arms.mkdir()
    (tmp_path / "out").mkdir()  # main() creates the report root before Arm makes its own subfolder
    monkeypatch.setattr(renewal, "REPO", repo)
    case = renewal.Arm("baseline", sha, arms, tmp_path / "out", lambda: None)
    try:
        case.prepare()
        assert case.tree["version"] == "9.9.9"
        assert case.tree["base"] != case.tree["target"] != case.tree["handoff"]
        assert case.model.base_url.startswith("http://127.0.0.1:")
        assert not (set(STRIPPED_PROVIDER_ENV_KEYS) | set(PROXY_ENV_KEYS)) & set(case.env)
        assert case.env["PIP_NO_INDEX"] == "1" and case.env["OUROBOROS_SERVER_HOST"] == "127.0.0.1"
        settings = json.loads((case.data / "settings.json").read_text(encoding="utf-8"))
        assert_settings_keyless(settings)
        assert settings["OUROBOROS_MAX_WORKERS"] == 3 and settings["OUROBOROS_UPDATE_CHANNEL"] == "development"
        assert settings["OPENAI_COMPATIBLE_BASE_URL"] == case.model.base_url
        for key in renewal.TASKS:
            assert (case.workspaces[key] / f"probe-{key}.txt").read_text(encoding="utf-8").strip() == \
                f"{key.upper()}-{case.tag}"
    finally:
        if case.model is not None:
            case.model.__exit__(None, None, None)
