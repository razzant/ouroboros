"""The macOS window-renewal check judges honestly without opening a window here.

The native run is operator- or dispatch-invoked on macOS (devtools/macos_wkwebview/window_renewal.md). These cases
pin what its verdict may conclude from recorded checks: a candidate passes only beside a baseline that kept its web
view through the same operations, a check not established or not calibrated is never a pass, an unsupported host
never reads as one, and the UI anchors and the dispatch-only workflow it relies on still exist. The arm's source and
managed mirror are built here with real Git, so a wrong seed or offer cannot first appear on the one hosted Mac.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
from types import SimpleNamespace

import pytest
import yaml

from devtools.macos_wkwebview import arm_source, native_window
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
    for key in ("restart.view_new", "fixture.view_new", "update.project_active_after", "handoff.project_active_after",
                "handoff.project_editable", "update.project_draft", "handoff.project_reply_restored", "update.project_reattach_notice",
                "update.project_window_typed_pixels", "handoff.project_sent_once", "update.draft_restored", "restart.reply_restored",
                "restart.no_replay", "update.pending_owner", "session.draft_gone", "fixture.sample_document_opened"):
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
    assert native_window.pixels_readable(None) is None
    assert native_window.pixels_readable({"ok": False}) is None
    assert native_window.pixels_readable({"ok": True, "distinct_colors": 3, "dark_fraction": 1.0}) is False
    assert native_window.pixels_readable({"ok": True, "distinct_colors": 400, "dark_fraction": 0.9}) is True


def test_an_older_arm_that_opens_main_still_calibrates_the_same_updates():
    differences = {f"{phase}.{key}": None for phase in renewal.PROJECT_UPDATES for key in renewal.PROJECT_CHECKS
                   if key not in ("project_active_before", "document_ready")}
    differences.update({f"{phase}.project_active_after": False for phase in renewal.PROJECT_UPDATES})
    assert _judge(_baseline(**differences), _arm())["verdict"] == "PASS"
    for key in ("handoff.project_sent_once", "update.project_typed_pixels", "update.project_window_typed_pixels"):
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
    metrics, _ = native_window.typing_delta(paths, box, window)
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
def test_one_click_requires_the_original_frame_one_accepted_project_row_and_complete_main_history(monkeypatch, case, verdict):
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
    view, clicks = _view(port=5001), []
    def click(content, hold, prefix, target, *, from_draft):
        clicks.append((content, hold, prefix, target, from_draft))
        return None if case == "missing frame" else wire
    draft = {"text": "Main text" if case == "other draft" else text, "names": [], "input_files": int(case == "live file")}
    arm.win = SimpleNamespace(key_window=lambda target: target, send_once=click,
                              draft=lambda field: draft if field == "#pchat-room-sample-input" else None)
    monkeypatch.setattr(renewal, "wait_for", lambda read, _: read() or read())
    arm.project_send("update", view)
    assert arm.project_delivery("update") is verdict
    assert clicks == ([] if case in ("other draft", "live file") else [(text, False, "#pchat-room-sample", view, True)])
    if verdict is True:
        facts = arm.facts["update_project_send"]
        assert facts["wire"] == wire and facts["saved"] == {"count": 1, "addresses": [row["row_address"]]}
        assert facts["main"]["count"] == 0 and facts["writes"] == 1


def test_project_draft_uses_its_own_field_quote_target_and_empty_picker():
    prefix, reply = "#pchat-sample", "row:17@seed#0123456789ab"
    staged = {"prefix": prefix, "reply": reply, "text": "Project text", "filename": "project.txt",
              "quote_text": "Project-seed-sample", "typed": True, "quoted": True, "file": True}
    doc = {"text": "Project text", "reply": True, "names": [], "input_files": 0, "notice": "Attach your file again"}
    quotes = []
    room = SimpleNamespace(until=lambda read, _: read(), observe=lambda name, read: read(),
                           draft=lambda field: doc if field == prefix + "-input" else None,
                           follows_quote=lambda addr, p, text: quotes.append((addr, p, text)) or addr == reply)
    checks = {}
    native_window.RoomWindow.inspect_draft(room, staged, checks, "marker")
    assert checks == {"project_draft": True, "project_reply_restored": True, "project_reattach_notice": True}
    assert quotes == [(reply, prefix, "Project-seed-sample")]
    doc.update(text="Main text", input_files=1)
    room.follows_quote = lambda *_: False
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
    monkeypatch.setattr(native_window, "window_colors", lambda _: {"ok": True, "distinct_colors": 80, "dark_fraction": 0.4})
    monkeypatch.setattr(native_window, "typing_delta", delta)
    monkeypatch.setattr(native_window, "echo_overlap", lambda *_: 1.0)
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
    arm.generation, arm.checks, arm.facts = old, {"fixture": {}}, {}
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


def _view(**changes):
    return {"shell": "main", "pid": 1, "window": "one", "number": 9, "view": "a", "installed": True,
            "visible": True, "key": True, "on_screen": True, **changes}


@pytest.mark.parametrize("phase,ready,mounted,field_ready", [
    ("restart", True, False, False), ("update", True, True, True), ("handoff", True, True, True),
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
    arm.win = SimpleNamespace(prepare_draft=prepare, shows_project=shown, visible=visible,
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
    arm.after_presentation = arm.peer_project = arm.inspect_sends = lambda *_: None
    arm.project_send = send
    arm.project_delivery = lambda phase: seen.append("Project receipt") or True
    arm.transition(phase)
    if phase == "restart":
        order = ["prepare Project", "prepare Main", "presentation", "request", "select Main", "inspect #chat",
                 "select Project", "inspect #pchat-room-sample", "select Main"]
    else:
        order = ["prepare Main", "prepare Project", "presentation", "original room", "request"]
        assert checks["project_active_before"] is True
        assert checks["document_ready"] is (True if ready else None)
        assert checks["project_active_after"] is (mounted if ready else None)
        assert checks["project_editable"] is (field_ready if ready and mounted else None)
        if ready:
            order.append("observe restored room")
        if ready and mounted and field_ready:
            order += ["inspect #pchat-room-sample", "Project Send", "type " + project_field, "Project receipt"]
            assert checks["project_sent_once"] is True
        else:
            assert checks["project_sent_once"] is None
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
                              until=lambda read, _: read(), page=page, shows_project=visible)
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


@pytest.mark.skipif(os.name == "nt" or shutil.which("git") is None, reason="the macOS instrument's POSIX git and tar")
def test_an_arm_launches_its_exact_tree_and_is_offered_two_comment_only_updates(tmp_path):
    repo, root = tmp_path / "repo", tmp_path / "arm"
    for name, text in {"VERSION": "9.9.9\n", "server.py": "SERVER = 1\n", "launcher.py": "LAUNCHER = 1\n",
                       "web/app.js": "export {};\n"}.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text, encoding="utf-8")
    root.mkdir()
    for args in (("init", "-q", "-b", "main"), ("add", "-A"), ("commit", "-q", "-m", "frozen")):
        arm_source.git(*args, cwd=repo)
    sha = arm_source.git("rev-parse", "HEAD", cwd=repo)
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
    assert source["base"] in src("bundle", "list-heads", source["seed"] / "repo.bundle")
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
