"""Consumer regressions: UIA fakes, real registered plugin and child-result writer.

No Windows desktop is accessed. The COM oracle is the existing 1.4.17-shaped
provider; only native calls are substituted, not matching, receipts or delivery.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import threading
import types
from pathlib import Path

import pytest

from tests.test_windows_computer_use_uia import (
    Automation, Calls, COMError, dialog, make, window,
)
from tests.test_windows_computer_use_uia import fake_comtypes  # noqa: F401 (shared autouse fixture)
from windows_computer_use import core, plugin, tree, win_uia


@pytest.fixture
def registered(monkeypatch):
    calls = Calls()
    current = {"root": dialog(calls), "windows": []}
    fake_win32 = types.ModuleType("windows_computer_use.win32fb")
    fake_win32.resolve_window = lambda hwnd: window()
    fake_win32.window_pid = lambda hwnd: 42
    fake_win32.list_windows = lambda limit: current["windows"][:limit]
    monkeypatch.setitem(sys.modules, "windows_computer_use.win32fb", fake_win32)
    monkeypatch.setattr(sys.modules["windows_computer_use"], "win32fb", fake_win32, raising=False)
    monkeypatch.setattr(plugin, "_ensure_windows", lambda: None)
    monkeypatch.setattr(win_uia, "client_module", lambda: None)
    monkeypatch.setattr(win_uia, "thread_automation", lambda: Automation(current["root"]))

    class API:
        def __init__(self):
            self.tools = {}

        def register_tool(self, name, handler, **metadata):
            self.tools[name] = {"handler": handler, **metadata}

    api = API()
    plugin.register(api)
    from ouroboros import extension_loader
    monkeypatch.setattr(extension_loader, "get_tool", lambda name: api.tools.get(name))
    return calls, current


def delivered(tmp_path, name, **args):
    from ouroboros.extension_process_runner import _RESULT_CAP, _call_tool, _write_child_result

    reply = asyncio.run(_call_tool(name, args, tmp_path, Path(__file__).resolve().parents[1]))
    path = tmp_path / "child.json"
    _write_child_result({"result_path": str(path)}, {"ok": True, "result": reply})
    assert path.stat().st_size <= _RESULT_CAP
    frame = json.loads(path.read_text(encoding="utf-8"))
    assert frame["ok"] is True, frame
    print(f"WCU_WIRE {name}: {path.stat().st_size} bytes")
    return json.loads(frame["result"])


def test_find_schema_and_indexed_action_agree_with_real_handlers(tmp_path, registered):
    from ouroboros import extension_loader
    from jsonschema import Draft202012Validator

    calls, current = registered
    current["root"] = dialog(calls, *(make(calls, f"b{i}", 50000, name="Save", patterns={10000: None})
                                      for i in range(2)))
    find_schema = extension_loader.get_tool("element_find")["schema"]
    act_schema = extension_loader.get_tool("element_act")["schema"]
    plain = {"hwnd": 1, "select": {"name": "Save"}, "timeout_ms": 0}
    indexed = {"hwnd": 1, "select": {"name": "Save", "nth": 1}, "timeout_ms": 0}
    Draft202012Validator(find_schema).validate(plain)
    assert not Draft202012Validator(find_schema).is_valid(indexed)
    Draft202012Validator(act_schema).validate(dict(indexed, do="invoke"))
    assert delivered(tmp_path, "element_find", **plain)["matched"] == 2
    assert delivered(tmp_path, "element_act", **dict(indexed, do="invoke"))["ok"]
    assert calls == [("Invoke", "b1")]


def test_read_host_budget_contains_uia_and_win32_phases(registered):
    from ouroboros import extension_loader

    tool = extension_loader.get_tool("window_read")
    phase_ms = tool["schema"]["properties"]["timeout_ms"]["maximum"] + plugin.WORKER_GRACE_MS
    assert tool["timeout_sec"] * 1000 > 2 * phase_ms


@pytest.mark.parametrize("field", ["automation_id", "role", "name", "name_contains", "value_contains"])
@pytest.mark.parametrize("bad", [123, False, None, [], {}])
@pytest.mark.parametrize("tool", ["element_act", "element_find"])
def test_malformed_selector_never_widens_at_registered_host_call(tmp_path, registered, field, bad, tool):
    calls, current = registered
    target = make(calls, "save", 50000, name="Save", patterns={10000: None})
    current["root"] = dialog(calls, target)
    select = {"role": "button", "automation_id": "save", field: bad}
    with pytest.raises(ValueError, match=field):
        delivered(tmp_path, tool, hwnd=1, select=select, timeout_ms=0,
                  **({"do": "invoke"} if tool == "element_act" else {}))
    assert not calls


def test_valid_long_selector_is_matched_before_display_is_fitted(tmp_path, registered):
    calls, current = registered
    exact = " A " * 60000
    first = make(calls, "first", 50000, name=exact[:-1] + "x", patterns={10000: None})
    target = make(calls, "target", 50000, name=exact, patterns={10000: None})
    current["root"] = dialog(calls, first, target)
    out = delivered(tmp_path, "element_act", hwnd=1, select={"name": exact}, do="invoke", timeout_ms=0)
    assert out["ok"] and calls == [("Invoke", "target")]
    assert out["select"]["clipped_fields"]["name"]["utf16_units"] == len(exact)


def test_successful_close_preserves_action_and_reports_unavailable_after_state(tmp_path, registered):
    calls, current = registered
    target = make(calls, "close", 50000, name="Close", patterns={10000: None})

    def unavailable(*args):
        raise COMError(-2147220991, "element not available")

    def close():
        calls.append(("Invoke", "close"))
        target.GetCurrentPropertyValue = unavailable
        target.GetCurrentPattern = unavailable
        target.fail_live.add("role")

    target.patterns[10000].Invoke = close
    current["root"] = dialog(calls, target)
    out = delivered(tmp_path, "element_act", hwnd=1, select={"automation_id": "close"}, do="invoke", timeout_ms=0)
    assert out["ok"] is True and out["did"] == "invoke" and out["mutating"] is True
    assert out["action_outcome"] == "completed" and out["after_read"] == "unavailable"
    assert {"role", "name", "automation_id", "class", "value", "enabled", "focused", "offscreen"} <= set(out["element_after"]["unread"])
    assert "re-read afterwards" not in out["note"]
    assert calls == [("Invoke", "close")]


@pytest.mark.parametrize("verb,pattern_id,method", [
    ("invoke", 10000, "Invoke"), ("set_value", 10002, "SetValue"), ("toggle", 10015, "Toggle"),
    ("expand", 10005, "Expand"), ("collapse", 10005, "Collapse"), ("select", 10010, "Select"),
    ("scroll_into_view", 10017, "ScrollIntoView"), ("focus", None, "SetFocus"),
])
def test_exception_inside_action_is_not_a_preguard_refusal(tmp_path, registered, verb, pattern_id, method):
    calls, current = registered
    target = make(calls, "go", 50000, patterns={pattern_id: None} if pattern_id else {})

    def entered(*args):
        calls.append((method, "go"))
        raise COMError(-2147220991, "element not available")

    setattr(target.patterns[pattern_id] if pattern_id else target, method, entered)
    current["root"] = dialog(calls, target)
    out = delivered(tmp_path, "element_act", hwnd=1, select={"automation_id": "go"}, do=verb,
                    timeout_ms=0, **({"text": ""} if verb == "set_value" else {}))
    assert out["ok"] is False and out["action_call_made"] is True
    assert out["action_outcome"] == "unknown" and out["mutating"] is core.act_mutating(verb)
    assert out["element"]["automation_id"] == "go"
    assert "retry blindly" in out["hint"] and "did" not in out
    assert calls == [(method, "go")]


def test_preguard_refusal_never_claims_an_action_call(tmp_path, registered):
    calls, current = registered
    target = make(calls, "go", 50000, name="Go", patterns={10000: None}, fail_live={"name"})
    current["root"] = dialog(calls, target)
    out = delivered(tmp_path, "element_act", hwnd=1, select={"name": "Go"}, do="invoke", timeout_ms=0)
    assert out["ok"] is False and "re-reading" in out["error"]
    assert out.get("action_call_made", False) is False and not calls


@pytest.mark.parametrize("count,label,limit", [
    (20, '"\\\u0001界' * 10000, None), (200, "x" * 3000, 200),
], ids=["default-large-escaped", "ceiling-many-values"])
def test_find_keeps_count_and_discloses_display_cuts(tmp_path, registered, count, label, limit):
    calls, current = registered
    current["root"] = dialog(calls, *(make(calls, f"e{i}", 50004, name=label,
                                          extra={30043: True, 30045: label}) for i in range(count)))
    args = {"limit": limit} if limit else {}
    out = delivered(tmp_path, "element_find", hwnd=1, select={"role": "edit"}, **args)
    assert out["ok"] is True and out["matched"] == count and out["searched_whole_window"] is True
    assert out["shown"] == len(out["elements"]) > 0
    assert out["omitted"] == count - out["shown"]
    assert "result_size" in out["truncated_by"]
    assert out["omitted"] or out["elements"][0]["clipped_fields"]


def test_ambiguity_survives_wire_with_original_match_count(tmp_path, registered):
    calls, current = registered
    current["root"] = dialog(calls, *(make(calls, f"b{i}", 50000, name="label" * 30) for i in range(3000)))
    # timeout_ms=0 bounds the one sweep by the 750 ms worker grace; this fake tree's 3000
    # siblings outlast that on a loaded runner ("timeout"). An ambiguous match returns at once
    # whatever the bound, so a real wait budget changes only how long the sweep may take.
    out = delivered(tmp_path, "element_act", hwnd=1, select={"role": "button"}, do="invoke", timeout_ms=60000)
    assert out["ok"] is False and out["reason"] == "ambiguous" and out["matched"] == 3000
    assert out["shown"] == len(out["candidates"]) > 0
    assert out["omitted"] == 3000 - out["shown"] > 0
    assert "result_size" in out["truncated_by"] and "nth" in out["hint"]
    assert not calls


def test_large_set_value_receipt_cannot_turn_completed_mutation_into_wire_failure(tmp_path, registered):
    calls, current = registered
    old = "x" * 630000
    target = make(calls, "edit", 50004, name="Edit", extra={30043: True, 30045: old}, patterns={10002: {"value": old}})

    def clear(text):
        calls.append(("SetValue", "edit", text))
        target.patterns[10002]._value = text

    target.patterns[10002].SetValue = clear
    current["root"] = dialog(calls, target)
    out = delivered(tmp_path, "element_act", hwnd=1, select={"automation_id": "edit"}, do="set_value", text="", timeout_ms=0)
    assert out["ok"] is True and out["did"] == "set_value" and out["mutating"] is True
    assert out["action_outcome"] == "completed" and calls == [("SetValue", "edit", "")]
    assert out["element"]["clipped_fields"]["value"]["utf16_units"] == len(old)
    assert "result_size" in out["truncated_by"]


def test_large_window_list_survives_wire_and_names_omitted_windows(tmp_path, registered):
    _, current = registered
    current["windows"] = [{"hwnd": hex(i), "title": '"' * 20000, "class": "Window"} for i in range(500)]
    out = delivered(tmp_path, "window_list", limit=500)
    assert out["count"] == 500 and out["shown"] == len(out["windows"]) > 0
    assert out["omitted"] == 500 - out["shown"] > 0
    assert "title" in out["windows"][0]["clipped_fields"]


def test_unreadable_candidates_keep_count_and_failed_fields_with_display_cuts(tmp_path, registered):
    calls, current = registered
    current["root"] = dialog(calls, *(make(calls, f"b{i}", 50000, name="x", aid="a" * 40000,
                                          fail_cached={30005, 30010}) for i in range(30)))
    out = delivered(tmp_path, "element_act", hwnd=1, select={"role": "button", "name": "x"}, do="invoke", timeout_ms=0)
    assert out["reason"] == "unreadable_candidates" and out["unreadable"] == 30
    assert out["unreadable_shown"] == len(out["unreadable_candidates"]) > 0
    assert out["unreadable_omitted"] == 30 - out["unreadable_shown"]
    assert {"name", "enabled"} <= set(out["unreadable_candidates"][0]["unread"])
    assert "automation_id" in out["unreadable_candidates"][0]["clipped_fields"]
    assert not calls


def test_huge_single_root_and_window_metadata_still_fit_read_delivery(tmp_path, registered):
    calls, current = registered
    text = '"' * 630000
    current["root"] = make(calls, "root", 50032, name=text)
    sys.modules["windows_computer_use.win32fb"].resolve_window = lambda hwnd: tree.WindowInfo(
        1, text, "Window", 42, None, True, "argument", 96)
    out = delivered(tmp_path, "window_read", hwnd=1)
    assert out["ok"] and out["nodes_returned"] == 1
    assert "result_size" in out["truncated_by"]
    assert out["window"]["clipped_fields"]["title"]["utf16_units"] == len(text)


def test_unsupported_after_fields_are_not_reported_as_provider_failures(tmp_path, registered):
    calls, current = registered
    target = make(calls, "go", 50000, patterns={10000: None})
    target.props = {30003: 50000, 30011: "go"}  # other properties return the not-supported sentinel
    current["root"] = dialog(calls, target)
    out = delivered(tmp_path, "element_act", hwnd=1, select={"automation_id": "go"}, do="invoke", timeout_ms=0)
    assert out["ok"] and out["after_read"] == "complete"
    assert "unread" not in out["element_after"] and "state" not in out["element_after"]


@pytest.mark.parametrize("tool", ["window_list", "window_read", "element_find", "element_act"])
def test_lone_surrogates_are_valid_json_through_both_encoders(tmp_path, registered, monkeypatch, tool):
    calls, current = registered
    text = "Ж界🙂\ud800"
    current["root"] = dialog(calls, make(calls, "go", 50000, name=text, patterns={10000: None}))
    current["windows"] = [{"hwnd": "0x1", "title": text}]
    fb = sys.modules["windows_computer_use.win32fb"]
    fb.resolve_window = lambda hwnd: tree.WindowInfo(1, text, "Window", 42, None, True, "argument", 96)
    args = {} if tool == "window_list" else {"hwnd": 1}
    if tool in {"element_find", "element_act"}:
        args["select"] = {"name": text}
    if tool == "element_act":
        args.update(do="invoke", timeout_ms=0)
    out = delivered(tmp_path, tool, **args)
    assert out["ok"] is True
    rendered = json.dumps(out, ensure_ascii=True)
    assert "\\ud800" in rendered and "\\u0416" in rendered
    wire = (tmp_path / "child.json").read_text(encoding="utf-8")
    assert "Ж界🙂" in wire  # ordinary non-ASCII stays readable


def test_win32_timeout_marks_name_unread_without_an_extra_window_scan(tmp_path, monkeypatch):
    import ctypes

    class Fn:
        def __init__(self, impl=lambda *args: 0):
            self.impl = impl

        def __call__(self, *args):
            return self.impl(*args)

    class DLL:
        def __getattr__(self, name):
            fn = Fn()
            setattr(self, name, fn)
            return fn

    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: DLL(), raising=False)
    monkeypatch.setattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE, raising=False)
    name = "windows_computer_use._win32_timeout_test"
    spec = importlib.util.spec_from_file_location(name, Path(win_uia.__file__).with_name("win32fb.py"))
    fb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fb)
    fb._user32.IsWindowVisible = Fn(lambda hwnd: True)
    fb._user32.IsWindowEnabled = Fn(lambda hwnd: True)
    seen = []
    fb._user32.SendMessageTimeoutW = Fn(lambda *args: seen.append(args[1]) or 0)
    walk = fb.win32_walk(window(), core.ReadLimits())
    assert walk.nodes[0].unread == ("name",) and seen == [fb.WM_GETTEXTLENGTH]
    reply = plugin._json_tool(lambda: tree.render(core.read_plan({"hwnd": 1}), window(), walk, "win32", "", None, 1))()
    from ouroboros.extension_process_runner import _write_child_result
    path = tmp_path / "fallback.json"
    _write_child_result({"result_path": str(path)}, {"ok": True, "result": reply})
    out = json.loads(json.loads(path.read_text())["result"])
    assert out["root"]["unread"] == ["name"] and out["nodes_with_unread_fields"] == 1


# ── a document's value: matching sees the whole text, displays name their cut ──

LONG = "x" * 5000


def text_reads(calls, label):
    return [call[2] for call in calls if call[:2] == ("GetText", label)]


def test_document_text_past_the_display_cut_is_found_and_the_cut_is_named(tmp_path, registered):
    calls, current = registered
    current["root"] = dialog(calls, make(calls, "doc", 50030, patterns={10014: {"text": LONG + " Total 42"}}))
    out = delivered(tmp_path, "element_find", hwnd=1, select={"role": "document", "value_contains": "total 42"})
    assert out["ok"] is True and out["matched"] == 1 and out["searched_whole_window"] is True
    assert out["elements"][0]["value"] == LONG[:4000] and out["elements"][0]["text_truncated"] is True
    # Selection read the whole document; the description asked for one unit more than it shows.
    assert text_reads(calls, "doc") == [-1, 4001]


def test_a_competitor_matching_past_the_display_cut_makes_act_refuse(tmp_path, registered):
    calls, current = registered
    current["root"] = dialog(calls,
                             make(calls, "first", 50030, patterns={10014: {"text": "Invoice 42 draft"}}),
                             make(calls, "second", 50030, patterns={10014: {"text": LONG + "Invoice 42 final"}}))
    out = delivered(tmp_path, "element_act", hwnd=1, select={"value_contains": "invoice 42"}, do="focus", timeout_ms=0)
    assert out["ok"] is False and out["reason"] == "ambiguous" and out["matched"] == 2
    candidates = {item["automation_id"]: item for item in out["candidates"]}
    assert candidates["second"]["text_truncated"] is True and "text_truncated" not in candidates["first"]
    assert not [call for call in calls if call[0] == "SetFocus"]


def test_act_on_a_document_matched_past_the_display_cut(tmp_path, registered):
    calls, current = registered
    current["root"] = dialog(calls, make(calls, "doc", 50030, patterns={10014: {"text": LONG + "Invoice 42"}}))
    out = delivered(tmp_path, "element_act", hwnd=1, select={"role": "document", "value_contains": "invoice 42"},
                    do="focus", timeout_ms=0)
    assert out["ok"] is True and ("SetFocus", "doc") in calls
    assert out["element"]["text_truncated"] is True and out["element_after"]["text_truncated"] is True
    # The sweep and the pre-action re-read match the whole text; before/after show a bounded prefix.
    assert text_reads(calls, "doc") == [-1, -1, 4001, 4001]


@pytest.mark.parametrize("text,shown,cut", [
    ("d" * 239, "d" * 239, False), ("d" * 240, "d" * 240, False), ("d" * 241, "d" * 240, True),
    ("d" * 5000, "d" * 240, True), ("🙂" * 120, "🙂" * 120, False), ("🙂" * 121, "🙂" * 120, True),
], ids=["under", "exact", "one-over", "long", "pairs-exact", "pairs-over"])
def test_window_read_marks_a_document_cut_at_max_text_chars(tmp_path, registered, text, shown, cut):
    calls, current = registered
    current["root"] = dialog(calls, make(calls, "doc", 50030, patterns={10014: {"text": text}}))
    out = delivered(tmp_path, "window_read", hwnd=1, shape="flat")
    node = next(item for item in out["items"] if item.get("automation_id") == "doc")
    assert node["value"] == shown and node.get("text_truncated", False) is cut
    assert text_reads(calls, "doc") == [241]  # UTF-16 units, like the provider's maxLength


# ── set_value: only a string is text; "" is the explicit clear ────────────────

def value_edit(calls):
    return make(calls, "edit", 50004, extra={30043: True, 30045: "keep me"}, patterns={10002: {"value": "keep me"}})


@pytest.mark.parametrize("text", [None, 0, False, [], {}])
def test_set_value_without_a_string_is_refused_before_any_effect(tmp_path, registered, text):
    calls, current = registered
    current["root"] = dialog(calls, value_edit(calls))
    with pytest.raises(ValueError, match="set_value needs text as a string"):
        delivered(tmp_path, "element_act", hwnd=1, select={"automation_id": "edit"}, do="set_value", text=text,
                  timeout_ms=0)
    assert not calls


@pytest.mark.parametrize("text", ["", "Привет, 世界 🙂", "  spaced  "], ids=["clear", "unicode", "spaces"])
def test_set_value_hands_an_explicit_string_over_unchanged(tmp_path, registered, text):
    calls, current = registered
    current["root"] = dialog(calls, value_edit(calls))
    out = delivered(tmp_path, "element_act", hwnd=1, select={"automation_id": "edit"}, do="set_value", text=text,
                    timeout_ms=0)
    assert out["ok"] is True and out["action_outcome"] == "completed"
    assert calls == [("SetValue", "edit", text)]


# ── abandoned workers and capped listings say what is unknown ─────────────────

def join_workers():
    for thread in threading.enumerate():
        if thread.name == "wcu-worker":
            thread.join(10)


def test_an_abandoned_act_worker_claims_neither_a_call_nor_its_absence(tmp_path, registered, monkeypatch):
    calls, current = registered
    release = threading.Event()
    target = make(calls, "go", 50000, patterns={10000: None})
    target.patterns[10000].Invoke = lambda: calls.append(("Invoke", "go")) or release.wait(10)
    current["root"] = dialog(calls, target)
    monkeypatch.setattr(plugin, "WORKER_GRACE_MS", 25)
    try:
        out = delivered(tmp_path, "element_act", hwnd=1, select={"automation_id": "go"}, do="invoke", timeout_ms=0)
    finally:
        release.set()
        join_workers()
    assert out["ok"] is False and out["action_call_made"] == "unknown" and out["action_outcome"] == "unknown"
    assert out["requested"] == "invoke" and out["mutating"] is True and out["select"] == {"automation_id": "go"}
    assert "did" not in out and "retry blindly" in out["hint"]


def test_a_fallback_after_an_abandoned_read_counts_its_live_reader(tmp_path, registered, monkeypatch):
    _, current = registered
    release = threading.Event()
    fb = sys.modules["windows_computer_use.win32fb"]
    fb.win32_walk = lambda win, limits, visible_only=False: tree.Walk(nodes=[tree.Node(role="window", name="Dialog")])
    monkeypatch.setattr(win_uia, "thread_automation", lambda: release.wait(10) and Automation(current["root"]))
    monkeypatch.setattr(plugin, "WORKER_GRACE_MS", 25)
    try:
        hung = delivered(tmp_path, "window_read", hwnd=1, timeout_ms=100)
    finally:
        release.set()
        join_workers()
    assert hung["backend"] == "win32" and hung["uia_workers_live"] == 1
    assert "left running" in hung["fallback_reason"]

    def broken():
        raise OSError("provider gone")

    monkeypatch.setattr(win_uia, "thread_automation", broken)
    failed = delivered(tmp_path, "window_read", hwnd=1)
    assert failed["backend"] == "win32" and "provider gone" in failed["fallback_reason"]
    assert failed["uia_workers_live"] == 0  # a worker that returned is not counted as live


def test_window_list_echoes_its_limit_and_never_passes_a_cut_listing_as_whole(tmp_path, registered):
    _, current = registered
    current["windows"] = [{"hwnd": hex(i), "title": f"w{i}"} for i in range(1, 8)]
    cut = delivered(tmp_path, "window_list", limit=5)
    assert (cut["count"], cut["shown"], cut["omitted"], cut["limit"]) == (5, 5, 0, 5)
    assert [row["title"] for row in cut["windows"]] == ["w1", "w2", "w3", "w4", "w5"]
    assert cut["truncated"] is True and cut["truncated_by"] == ["limit"] and cut["total_known"] is False
    assert any("stopped at limit=5" in note and "display cuts only" in note for note in cut["notes"])
    assert "untitled windows are left out" in cut["selection"]
    whole = delivered(tmp_path, "window_list", limit=7)
    assert whole["count"] == 7 and whole["total_known"] is True and whole["truncated_by"] == [] and not whole["notes"]
    clamped = delivered(tmp_path, "window_list", limit=1000)
    assert clamped["limit"] == 500 and clamped["count"] == 7 and clamped["total_known"] is True
    assert clamped["notes"] == ["limit=1000 is outside [1, 500]; this listing used 500"]
