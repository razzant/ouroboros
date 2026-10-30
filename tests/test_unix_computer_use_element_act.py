"""unix_computer_use ``element_act``: one element named by identity, one of its own AX actions, an honest receipt.

The helper's logic (``macos_native.act_in_window`` / ``op_act``) runs for real against a fake ``_AX`` surface: a
dict-shaped element graph with retain/release accounting, per-element actions, settable flags and action outcomes.
Only the ctypes boundary is faked, so the choice rules (several matches, an unfinished walk, a redrawn view), the
preconditions, the three outcomes (completed, unknown after the app's timeout, failed) and the readback are the
production code paths. The plugin tests fake the process boundary as the targeting tests do and pin that the tool
never activates, never falls back to the pointer, and reports a helper that gave no answer as unknown.
"""
from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from skills.unix_computer_use import macos_native as native
from tests.test_unix_computer_use_effects import marked, ssh_mac
from tests.test_unix_computer_use_targeting import KEYNOTE, SAFARI, app_dict, inputs, mac, receipt

# --- a fake accessibility surface --------------------------------------------------------------------------------

TIMEOUT = -25204


class FakeAX:
    """The ``_AX`` surface over a dict graph. Elements are string ids; ``("ref", id)`` is a retained reference as
    ``attr`` answers one. ``retained`` counts outstanding retains per element (the app element included)."""

    def __init__(self, nodes: dict, *, windows: list[str], focused: str | None = None, trusted: bool = True,
                 perform=None, on_set=None):
        self.nodes, self.windows, self.focused, self.trusted = nodes, list(windows), focused, trusted
        self.perform_rule = perform or (lambda el, action: 0)
        self.on_set = on_set or (lambda el, value: self.nodes[el].__setitem__("value", value))
        self.retained: dict[str, int] = {}
        self.performed: list[tuple[str, str]] = []
        self.sets: list[tuple[str, str]] = []
        self.timeouts: list[float] = []
        self.AXUIElementGetWindow = object()  # the private symbol is present unless a test removes it

    # process-level
    def AXIsProcessTrusted(self):
        return self.trusted

    def AXUIElementCreateApplication(self, pid):
        self.retained["app"] = self.retained.get("app", 0) + 1
        return "app"

    def set_timeout(self, elem, seconds):
        self.timeouts.append(seconds)

    def CFRetain(self, ref):
        self.retained[ref] = self.retained.get(ref, 0) + 1
        return ref

    def CFRelease(self, ref):
        assert self.retained.get(ref, 0) > 0, f"release of {ref!r} without a retain"
        self.retained[ref] -= 1

    # element reads
    def _vals(self, el, names):
        node = self.nodes.get(el)
        if node is None:
            return -25202, []
        if node.get("busy"):
            return TIMEOUT, []
        out = []
        for name in names:
            key = {"AXRole": "role", "AXSubrole": "subrole", "AXTitle": "title", "AXDescription": "description",
                   "AXValue": "value", "AXPosition": "pos", "AXSize": "size", "AXEnabled": "enabled",
                   "AXFocused": "focused", "AXIdentifier": "identifier"}[name]
            if key in node.get("unread", ()):
                out.append(("axerror", -25200))
            else:
                out.append(node.get(key))
        return 0, out

    def multi(self, el, names):
        return self._vals(el, names)

    def attr(self, el, name):
        if el == "app":
            if name in ("AXFocusedWindow", "AXMainWindow"):
                if self.focused and name == "AXFocusedWindow":
                    return ("ref", self.CFRetain(self.focused)), 0
                return None, -25212
            raise AssertionError(f"unexpected app attribute {name}")
        err, vals = self._vals(el, [name])
        if err == 0 and isinstance(vals[0], tuple) and vals[0][0] == "axerror":
            return None, vals[0][1]  # as the real ``attr``: a failed attribute is (None, its AXError)
        return (vals[0] if err == 0 else None), err

    def children(self, el, limit):
        kids = list(self.nodes[el].get("children", []))
        return 0, [self.CFRetain(k) for k in kids[:limit]], len(kids)

    def window_refs(self, app):
        return [self.CFRetain(w) for w in self.windows], 0

    def window_id(self, win):
        return self.nodes[win].get("window_id") if self.AXUIElementGetWindow is not None else None

    # actions
    def action_names(self, el):
        return list(self.nodes[el].get("actions", [])), 0

    def settable(self, el, name):
        return self.nodes[el].get("settable", False), 0

    def perform(self, el, action):
        self.performed.append((el, action))
        code = self.perform_rule(el, action)
        if code == 0:
            after = self.nodes[el].get("after_press")
            if after == "gone":
                del self.nodes[el]
            elif isinstance(after, dict):
                self.nodes[el].update(after)
        return code

    def set_string(self, el, name, value):
        self.sets.append((el, value))
        code = self.perform_rule(el, "set_value")
        if code in (0, TIMEOUT):
            self.on_set(el, value)
        return code

    def balanced(self) -> bool:
        return all(count == 0 for count in self.retained.values())


def deck() -> dict:
    """A Keynote-like window: a toolbar with two Play buttons, a text field, a disabled button, a checkbox."""
    return {
        "win": {"role": "AXWindow", "title": "Deck.key", "window_id": 39, "children": ["bar", "field", "web"], "pos": (0, 0), "size": (800, 600)},
        "bar": {"role": "AXToolbar", "children": ["play", "play2", "stop", "share"]},
        "play": {"role": "AXButton", "title": "Play", "identifier": "toolbar.play", "actions": ["AXPress", "AXShowMenu"], "enabled": True, "pos": (10, 10), "size": (40, 20)},
        "play2": {"role": "AXButton", "description": "Play", "identifier": "toolbar.play.b", "actions": ["AXPress"], "enabled": True, "pos": (60, 10), "size": (40, 20)},
        "stop": {"role": "AXButton", "title": "Stop", "actions": ["AXPress"], "enabled": False},
        "share": {"role": "AXButton", "title": "Share", "actions": ["AXShowMenu"], "enabled": True},
        "field": {"role": "AXTextField", "title": "Presenter notes", "value": "old", "settable": True, "enabled": True, "focused": False},
        "web": {"role": "AXWebArea", "children": ["weblink"]},
        "weblink": {"role": "AXLink", "title": "Play", "actions": ["AXPress"]},
    }


def act(ax: FakeAX, do: str, select: dict, *, pid=200, max_nodes=600, timeout_ms=3000, fronts=None) -> dict:
    reads = iter(fronts or [app_dict(SAFARI), app_dict(SAFARI)])
    return native.op_act(pid, do, json.dumps(select), max_nodes, timeout_ms, ax=ax, front=lambda: next(reads, None), settle=lambda _s: None)


# --- the helper's own logic ------------------------------------------------------------------------------------

def test_a_unique_press_is_found_acted_on_and_read_back_without_moving_the_front():
    ax = FakeAX(deck(), windows=["win"], focused="win")
    out = act(ax, "press", {"role": "Button", "identifier": "toolbar.play"})
    assert out["ok"] is True and out["observed_effect"] is True and out["dispatched"] is True and out["action_outcome"] == "completed"
    assert ax.performed == [("play", "AXPress")], "the element's own action, once"
    assert out["element"]["label"] == "Play" and out["element"]["identifier"] == "toolbar.play" and out["element_after"]["label"] == "Play"
    assert out["window"] == {"from": "AXFocusedWindow", "title": "Deck.key", "window_id": 39}
    assert out["searched_whole_window"] is True and out["matched"] == 1 and out["truncated_by"] == ["web_content_skipped"]
    assert out["front_before"] == app_dict(SAFARI) and out["front_after"] == app_dict(SAFARI) and out["target_in_front"] is False
    assert out["select"] == {"role": "Button", "identifier": "toolbar.play"} and out["pid"] == 200 and out["do"] == "press"
    assert "no other UI change" in out["note"]
    assert ax.balanced(), f"every retain is released once: {ax.retained}"


def test_several_matches_refuse_without_nth_and_nth_picks_in_walk_order():
    ax = FakeAX(deck(), windows=["win"], focused="win")
    out = act(ax, "press", {"role": "Button", "label": "Play"})
    assert out["ok"] is False and out["reason"] == "element_ambiguous" and out["dispatched"] is False and not marked(out)
    assert [c["index"] for c in out["candidates"]] == [1, 2] and out["candidates"][1]["identifier"] == "toolbar.play.b"
    assert ax.performed == [] and out["note"] == "no action was made"
    second = act(ax, "press", {"role": "Button", "label": "Play", "nth": 2})
    assert second["ok"] is True and ax.performed == [("play2", "AXPress")], "nth is 1-based in walk order; an exact description also matches"
    third = act(ax, "press", {"role": "Button", "label": "Play", "nth": 3})
    assert third["ok"] is False and third["reason"] == "element_not_found" and "nth=3" in third["error"]
    assert ax.balanced()


@pytest.mark.parametrize("attribute", ["title", "description"])
@pytest.mark.parametrize("label", ["Play", "P" * 220 + "LAY"])
def test_labels_match_the_full_exact_title_or_description(attribute, label):
    nodes = deck()
    for el, text in (("play", label), ("play2", label.lower())):
        nodes[el].pop("title", None)
        nodes[el].pop("description", None)
        nodes[el][attribute] = text
    ax = FakeAX(nodes, windows=["win"], focused="win")
    for text, el in ((label, "play"), (label.lower(), "play2")):
        out = act(ax, "press", {"role": "button", "label": text})
        assert out["ok"] is True and out["matched"] == 1
        assert out["element"]["label"] == text and out["element_after"]["label"] == text and ax.performed[-1] == (el, "AXPress")
    assert ax.balanced()


def test_a_window_that_was_not_fully_read_refuses_even_with_nth():
    ax = FakeAX(deck(), windows=["win"], focused="win")
    # breadth-first: win, bar, field, web, play | play2, stop, share are never read within a budget of 5
    out = act(ax, "press", {"identifier": "toolbar.play"}, max_nodes=5)
    assert out["ok"] is False and out["reason"] == "window_not_fully_read" and out["searched_whole_window"] is False
    assert out["truncated_by"] == ["max_nodes", "web_content_skipped"] and out["matched"] == 1 and out["dispatched"] is False and ax.performed == []
    assert "nth=1" not in out["error"] and "max_nodes" in out["error"] and out["visited"] == 5
    first = act(ax, "press", {"identifier": "toolbar.play", "nth": 1}, max_nodes=5)
    assert first["reason"] == "window_not_fully_read" and first["dispatched"] is False and ax.performed == []
    missing = act(ax, "press", {"label": "Share"}, max_nodes=5)
    assert missing["reason"] == "window_not_fully_read" and "did not finish" in missing["error"], "absence is unproved in an unfinished walk"
    assert ax.balanced()


@pytest.mark.parametrize("unread", ["node", "children", "child_count"])
def test_nth_never_skips_an_unread_subtree_that_hides_an_earlier_match(unread):
    nodes = {
        "win": {"role": "AXWindow", "title": "Test", "children": ["left", "right"]},
        "left": {"role": "AXGroup", "children": ["first"], "busy": unread == "node"},
        "right": {"role": "AXGroup", "children": ["second"]},
        "first": {"role": "AXButton", "title": "Play", "actions": ["AXPress"]},
        "second": {"role": "AXButton", "title": "Play", "actions": ["AXPress"]},
    }
    ax = FakeAX(nodes, windows=["win"], focused="win")
    children = ax.children

    def read_children(el, limit):
        if el == "left" and unread != "node":
            return (TIMEOUT if unread == "children" else 0), [], 1
        return children(el, limit)

    ax.children = read_children
    out = act(ax, "press", {"role": "Button", "label": "Play", "nth": 1})
    assert out["matched"] == 1 and out["truncated_by"] == ["read_errors"]
    assert out["reason"] == "window_not_fully_read" and out["dispatched"] is False
    assert ax.performed == [] and ax.balanced()


def test_web_content_is_searched_only_on_request_and_a_busy_subtree_makes_the_search_incomplete():
    ax = FakeAX(deck(), windows=["win"], focused="win")
    out = act(ax, "press", {"role": "Link", "label": "Play"})
    assert out["reason"] == "element_not_found" and out["searched_whole_window"] is True, "web content skipped by choice is not an unfinished walk"
    out = act(ax, "press", {"role": "Link", "label": "Play", "include_web": True})
    assert out["ok"] is True and ax.performed == [("weblink", "AXPress")]
    nodes = deck()
    nodes["bar"]["busy"] = True
    busy = FakeAX(nodes, windows=["win"], focused="win")
    out = act(busy, "set_value", {"role": "TextField", "label": "Presenter notes", "text": "x"})
    assert out["reason"] == "window_not_fully_read" and "read_errors" in out["truncated_by"], "a subtree the app did not answer may hide a match"
    assert busy.performed == [] and busy.sets == [] and ax.balanced() and busy.balanced()


def test_an_element_that_did_not_answer_a_named_attribute_forbids_the_choice():
    nodes = deck()
    nodes["play2"]["unread"] = ["description"]  # a Play button whose label the app failed to answer
    ax = FakeAX(nodes, windows=["win"], focused="win")
    out = act(ax, "press", {"role": "Button", "label": "Play", "nth": 1})
    assert out["ok"] is False and out["reason"] == "candidates_unread" and out["dispatched"] is False and out["unreadable"] == 1
    assert out["matched"] == 1 and out["unreadable_candidates"][0]["unread"] == {"AXDescription": "Failure"} and ax.performed == []
    assert "one of them may be the target" in out["error"]
    by_identifier = act(ax, "press", {"identifier": "toolbar.play"})
    assert by_identifier["ok"] is True, "an attribute the selector does not name may stay unread"
    nodes = deck()
    nodes["play"]["unread"] = ["identifier"]
    ax = FakeAX(nodes, windows=["win"], focused="win")
    out = act(ax, "press", {"role": "Button", "identifier": "toolbar.play.b"})
    assert out["reason"] == "candidates_unread" and out["unreadable"] == 1 and ax.performed == []
    # the fresh read before acting applies the same rule
    ax = FakeAX(deck(), windows=["win"], focused="win")
    real_multi = ax.multi

    def multi(el, names):
        err, vals = real_multi(el, names)
        if el == "play" and names == native._ELEMENT_ATTRS:
            vals[2] = ("axerror", -25204)
        return err, vals

    ax.multi = multi
    out = act(ax, "press", {"role": "Button", "label": "Play", "nth": 1})
    assert out["reason"] == "element_unread" and "AXTitle" in out["error"] and ax.performed == [] and ax.balanced()


def test_preconditions_refuse_a_disabled_element_a_missing_action_and_an_unsettable_value():
    ax = FakeAX(deck(), windows=["win"], focused="win")
    disabled = act(ax, "press", {"label": "Stop"})
    assert disabled["reason"] == "element_disabled" and disabled["dispatched"] is False and disabled["element"]["enabled"] is False
    menu = act(ax, "press", {"label": "Share"})
    assert menu["reason"] == "action_unsupported" and menu["actions"] == ["AXShowMenu"] and "never through the pointer" in menu["error"]
    assert ax.performed == []
    nodes = deck()
    nodes["field"]["settable"] = False
    out = act(FakeAX(nodes, windows=["win"], focused="win"), "set_value", {"role": "TextField", "text": "x"})
    assert out["reason"] == "value_not_settable" and out["dispatched"] is False
    nodes = deck()
    nodes["field"].update({"value": True})
    out = act(FakeAX(nodes, windows=["win"], focused="win"), "set_value", {"role": "TextField", "text": "x"})
    assert out["reason"] == "unsupported_value_type" and "checkbox" in out["error"]
    assert ax.balanced()


def test_the_outcome_after_the_apps_timeout_is_unknown_and_a_refused_call_is_failed_neither_carries_the_mark():
    ax = FakeAX(deck(), windows=["win"], focused="win", perform=lambda el, a: TIMEOUT)
    out = act(ax, "press", {"identifier": "toolbar.play"})
    assert out["ok"] is False and out["dispatched"] is True and out["action_outcome"] == "unknown" and out["reason"] == "outcome_unknown"
    assert out["ax_error"] == "CannotComplete" and not marked(out) and "may still act" in out["note"] and "do not retry blindly" in out["note"]
    assert "element_after" in out, "the element is still read back: the app may have acted"
    failed = act(FakeAX(deck(), windows=["win"], focused="win", perform=lambda el, a: -25200), "press", {"identifier": "toolbar.play"})
    assert failed["ok"] is False and failed["dispatched"] is True and failed["action_outcome"] == "failed" and failed["ax_error"] == "Failure"
    assert not marked(failed) and "nothing is known to have changed" in failed["note"]
    assert ax.balanced()


def test_set_value_is_verified_by_readback_and_a_completed_action_survives_a_failed_readback():
    ax = FakeAX(deck(), windows=["win"], focused="win")
    out = act(ax, "set_value", {"role": "TextField", "label": "Presenter notes", "text": "new notes"})
    assert out["ok"] is True and marked(out) and out["verified"] is True and ax.sets == [("field", "new notes")]
    assert out["element"]["value"] == "old" and out["element_after"]["value"] == "new notes" and "reads it back" in out["note"]
    cleared = act(ax, "set_value", {"role": "TextField", "text": ""})
    assert cleared["ok"] is True and cleared["verified"] is True and ax.nodes["field"]["value"] == "", "an empty string clears"
    normalizing = FakeAX(deck(), windows=["win"], focused="win", on_set=lambda el, v: None)
    out = act(normalizing, "set_value", {"role": "TextField", "text": "new"})
    assert out["ok"] is True and marked(out) and out["verified"] is False and "different one" in out["note"]
    gone = FakeAX(deck(), windows=["win"], focused="win", on_set=lambda el, v: normalizing_gone(gone, el, v))
    out = act(gone, "set_value", {"role": "TextField", "text": "bye"})
    assert out["ok"] is True and marked(out) and out["verified"] is None
    assert out["element_after"] == {"unread": {"element": "InvalidUIElement"}} and "may be gone" in out["note"]
    assert ax.balanced() and normalizing.balanced() and gone.balanced()


def normalizing_gone(ax: FakeAX, el: str, _value: str) -> None:
    del ax.nodes[el]


# Past 200 characters, with combining marks and an astral character so that no cut lands on a clean boundary.
LONG_LABEL = "N" + "e\u0301\u0302\U0001d11e" * 80 + " [label-end]"
LONG_BEFORE = "\u2192 " + "ab\u0301\U0001d11e" * 70 + " [before-end]"
LONG_TEXT = "set: " + "x\u0302\U0001d11e" * 100 + " [text-end]"
NORMALIZED = " [app-normalized-suffix]"


def long_field() -> dict:
    nodes = deck()
    nodes["field"].update(title=LONG_LABEL, value=LONG_BEFORE)
    return nodes


def test_a_set_value_receipt_keeps_the_complete_label_before_value_and_normalized_readback(monkeypatch, capsys):
    nodes = long_field()
    ax = FakeAX(nodes, windows=["win"], focused="win", on_set=lambda el, v: nodes[el].__setitem__("value", v + NORMALIZED))
    real = native.op_act
    monkeypatch.setattr(native, "op_act", lambda pid, do, sel, nodes_, ms: real(pid, do, sel, nodes_, ms, ax=ax, front=lambda: None, settle=lambda _s: None))
    selector = {"role": "TextField", "label": LONG_LABEL, "text": LONG_TEXT}
    assert native.main(["act", "200", "set_value", json.dumps(selector)]) == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line.isascii(), "the helper's one line is ASCII; the parent decodes the same strings under any locale"
    out = json.loads(line)
    assert len(LONG_LABEL) > 200 and len(LONG_BEFORE) > 200 and len(LONG_TEXT + NORMALIZED) > 200
    assert out["ok"] is True and marked(out) and out["action_outcome"] == "completed" and ax.sets == [("field", LONG_TEXT)]
    assert {"label": out["element"]["label"], "label_after": out["element_after"]["label"], "before": out["element"]["value"],
            "after": out["element_after"]["value"]} == {"label": LONG_LABEL, "label_after": LONG_LABEL, "before": LONG_BEFORE,
                                                       "after": LONG_TEXT + NORMALIZED}, "whole: the value replaced and the one kept"
    assert out["verified"] is False and "different one" in out["note"]
    assert not any("value_len" in out[key] for key in ("element", "element_after")), "no cut, so no length of a cut"
    assert ax.balanced()


def test_refusal_and_candidate_rows_keep_the_complete_label_and_value():
    nodes = long_field()
    nodes["play"].update(title=LONG_LABEL, value=LONG_BEFORE)
    nodes["play2"].update(description=LONG_LABEL, value=LONG_BEFORE)
    ambiguous = act(FakeAX(nodes, windows=["win"], focused="win"), "press", {"role": "Button", "label": LONG_LABEL})
    assert ambiguous["reason"] == "element_ambiguous" and len(ambiguous["candidates"]) == 2
    nodes = long_field()
    nodes["play2"].update(title=LONG_LABEL, value=LONG_BEFORE, unread=["description"])
    unreadable = act(FakeAX(nodes, windows=["win"], focused="win"), "press", {"role": "Button", "label": LONG_LABEL})
    assert unreadable["reason"] == "candidates_unread" and len(unreadable["unreadable_candidates"]) == 1
    nodes = long_field()
    nodes["field"]["enabled"] = False
    disabled = act(FakeAX(nodes, windows=["win"], focused="win"), "set_value", {"role": "TextField", "label": LONG_LABEL, "text": "x"})
    assert disabled["reason"] == "element_disabled" and disabled["dispatched"] is False
    rows = [*ambiguous["candidates"], *unreadable["unreadable_candidates"], disabled["element"]]
    assert [(row["label"], row["value"]) for row in rows] == [(LONG_LABEL, LONG_BEFORE)] * 4
    assert not any("value_len" in row for row in rows)


@pytest.mark.parametrize("value", [None, ("axerror", -25212), ("axerror", -25205)])
@pytest.mark.parametrize("text", ["", "new"])
@pytest.mark.parametrize("code", [0, TIMEOUT])
def test_set_value_with_absent_readback_keeps_verification_unknown(value, text, code):
    nodes = deck()
    ax = FakeAX(nodes, windows=["win"], focused="win", perform=lambda el, action: code,
                on_set=lambda el, text: nodes[el].__setitem__("value", value))
    out = act(ax, "set_value", {"role": "TextField", "text": text})
    assert out["dispatched"] is True and ax.sets == [("field", text)]
    assert out["action_outcome"] == ("completed" if code == 0 else "unknown")
    assert out["ok"] is (code == 0) and marked(out) is (code == 0)
    assert "value" not in out["element_after"] and out["verified"] is None
    assert ax.balanced()


def test_a_press_that_removes_its_element_is_completed_and_a_raising_readback_keeps_the_action():
    nodes = deck()
    nodes["play"]["after_press"] = "gone"
    ax = FakeAX(nodes, windows=["win"], focused="win")
    out = act(ax, "press", {"identifier": "toolbar.play"})
    assert out["ok"] is True and marked(out) and out["element_after"] == {"unread": {"element": "InvalidUIElement"}}
    raising = FakeAX(deck(), windows=["win"], focused="win")
    real_multi = raising.multi
    calls = {"n": 0}

    def multi(el, names):
        if el == "play" and names == native._ELEMENT_ATTRS:
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("provider went away")
        return real_multi(el, names)

    raising.multi = multi
    out = act(raising, "press", {"identifier": "toolbar.play"})
    assert out["ok"] is True and marked(out) and out["action_outcome"] == "completed"
    assert out["element_after"] == {"unread": {"element": "RuntimeError: provider went away"}}
    assert ax.balanced() and raising.balanced()


@pytest.mark.parametrize("new_title", ["Pause", "play"])
def test_a_redrawn_view_is_caught_by_the_fresh_read_before_acting(new_title):
    ax = FakeAX(deck(), windows=["win"], focused="win")
    real_multi = ax.multi

    def multi(el, names):
        err, vals = real_multi(el, names)
        if el == "play" and names == native._ELEMENT_ATTRS:
            vals[2] = new_title  # a case-only change also changes the exact label
        return err, vals

    ax.multi = multi
    out = act(ax, "press", {"role": "Button", "label": "Play", "nth": 1})
    assert out["ok"] is False and out["reason"] == "element_changed" and out["dispatched"] is False and ax.performed == []
    assert ax.balanced()


def test_ax_tree_exposes_an_identifier_usable_by_an_action_and_preserves_node_fields(monkeypatch):
    nodes = deck()
    nodes["play"]["identifier"] = "toolbar." + "p" * 220
    ax = FakeAX(nodes, windows=["win"], focused="win")
    monkeypatch.setattr(native, "_AX", lambda: ax)
    tree = native.op_tree(200)
    play = next(row for row in tree["nodes"] if row.get("identifier") == nodes["play"]["identifier"])
    assert {key: play[key] for key in ("role", "label", "x", "y", "w", "h")} == {
        "role": "Button", "label": "Play", "x": 10, "y": 10, "w": 40, "h": 20,
    }
    out = act(ax, "press", {"role": play["role"], "identifier": play["identifier"]})
    assert out["ok"] is True and out["element"]["identifier"] == play["identifier"]
    assert out["element"]["enabled"] is True and ax.performed == [("play", "AXPress")]
    assert ax.balanced()


def test_windows_are_named_by_id_or_exact_title_and_never_picked_among_equals():
    nodes = deck()
    nodes["win2"] = {"role": "AXWindow", "title": "Deck.key", "window_id": 40, "children": ["ok2"]}
    nodes["ok2"] = {"role": "AXButton", "title": "OK", "actions": ["AXPress"], "enabled": True}
    nodes["inspector"] = {"role": "AXWindow", "title": "Inspector", "window_id": 41, "children": ["ok3"]}
    nodes["ok3"] = {"role": "AXButton", "title": "OK", "actions": ["AXPress"], "enabled": True}
    ax = FakeAX(nodes, windows=["win", "win2", "inspector"], focused="win")
    by_id = act(ax, "press", {"label": "OK", "window": 40})
    assert by_id["ok"] is True and ax.performed == [("ok2", "AXPress")] and by_id["window"] == {"from": "AXWindows", "title": "Deck.key", "window_id": 40}
    by_title = act(ax, "press", {"label": "OK", "window": "Inspector"})
    assert by_title["ok"] is True and ax.performed[-1] == ("ok3", "AXPress")
    same = act(ax, "press", {"label": "OK", "window": "Deck.key"})
    assert same["ok"] is False and same["reason"] == "window_ambiguous" and [w["window_id"] for w in same["windows"]] == [39, 40, 41]
    none = act(ax, "press", {"label": "OK", "window": 77})
    assert none["reason"] == "window_not_found" and none["dispatched"] is False and "window" not in none
    focused = act(ax, "press", {"label": "OK"})
    assert focused["reason"] == "element_not_found" and focused["window"]["from"] == "AXFocusedWindow", "no window named: the focused one, said so"
    ax.AXUIElementGetWindow = None  # the private symbol is optional: titles still name windows, ids say they are unavailable
    by_title = act(ax, "press", {"label": "OK", "window": "Inspector"})
    assert by_title["ok"] is True and "window_id" not in by_title["window"]
    none = act(ax, "press", {"label": "OK", "window": 40})
    assert none["reason"] == "windows_unread" and "exact title" in none["error"]
    assert ax.balanced()
    unfocused = FakeAX(deck(), windows=["win"], focused=None)
    out = act(unfocused, "press", {"identifier": "toolbar.play"})
    assert out["ok"] is True and out["window"]["from"] == "AXWindows", "no focused or main window: the first one, said so"
    assert unfocused.balanced()


@pytest.mark.parametrize("window", ["Deck.key", 39])
def test_named_window_refuses_when_another_window_cannot_be_excluded(window):
    nodes = deck()
    nodes["other"] = {"role": "AXWindow", "title": "Other", "window_id": 40}
    if isinstance(window, str):
        nodes["other"]["unread"] = ["title"]
    else:
        del nodes["other"]["window_id"]
    ax = FakeAX(nodes, windows=["win", "other"], focused="win")
    out = act(ax, "press", {"identifier": "toolbar.play", "window": window})
    assert out["reason"] == "windows_unread" and out["dispatched"] is False
    assert out["windows"][1]["unread"] and ax.performed == [] and ax.balanced()
    # The other naming method is fully readable and does not depend on the failed attribute.
    alternate = 39 if isinstance(window, str) else "Deck.key"
    out = act(ax, "press", {"identifier": "toolbar.play", "window": alternate})
    assert out["ok"] is True and ax.performed == [("play", "AXPress")] and ax.balanced()


def test_focused_window_retains_title_read_errors_without_preventing_identity_selection():
    nodes = deck()
    nodes["win"]["unread"] = ["title"]
    ax = FakeAX(nodes, windows=["win"], focused="win")
    out = act(ax, "press", {"identifier": "toolbar.play"})
    assert out["ok"] is True and out["window"]["unread"] == {"AXTitle": "Failure"}
    assert "title" not in out["window"] and ax.performed == [("play", "AXPress")] and ax.balanced()


@pytest.mark.parametrize("focused_reply", [(None, TIMEOUT), (None, -25200), (None, 0), (("wrong", "win"), 0)])
def test_default_window_refuses_failed_or_malformed_focused_read_without_fallback(focused_reply):
    ax = FakeAX(deck(), windows=["win"], focused=None)
    original_attr = ax.attr
    reads = []

    def attr(el, name):
        if el == "app":
            reads.append(name)
            assert name == "AXFocusedWindow", "a failed default-window read must not try main or AXWindows"
            return focused_reply
        return original_attr(el, name)

    ax.attr = attr
    ax.window_refs = lambda _app: pytest.fail("AXWindows fallback after a failed focused-window read")
    out = act(ax, "press", {"identifier": "toolbar.play"})
    assert out["ok"] is False and out["reason"] == "windows_unread" and out["dispatched"] is False
    assert ax.performed == [] and reads == ["AXFocusedWindow"] and ax.balanced()


def test_default_window_refuses_busy_main_read_after_ordinary_focused_absence():
    ax = FakeAX(deck(), windows=["win"], focused=None)
    ax.attr = lambda _el, name: (None, -25212) if name == "AXFocusedWindow" else (None, TIMEOUT)
    ax.window_refs = lambda _app: pytest.fail("AXWindows fallback after busy main-window read")
    out = act(ax, "press", {"identifier": "toolbar.play"})
    assert out["ok"] is False and out["reason"] == "windows_unread" and out["dispatched"] is False
    assert "AXMainWindow" in out["error"] and ax.performed == [] and ax.balanced()


def test_default_window_uses_windows_after_unsupported_focused_and_main_attributes():
    ax = FakeAX(deck(), windows=["win"], focused=None)
    original_attr = ax.attr
    ax.attr = lambda el, name: (None, -25205) if el == "app" else original_attr(el, name)
    out = act(ax, "press", {"identifier": "toolbar.play"})
    assert out["ok"] is True and out["window"]["from"] == "AXWindows"
    assert ax.performed == [("play", "AXPress")] and ax.balanced()


def test_the_selector_admits_identity_only_and_the_untrusted_process_refuses_first():
    parse = native.parse_selector
    assert parse('{"role": "AXButton", "label": "OK", "nth": 2, "window": "39"}')[0] == {"role": "AXButton", "label": "OK", "nth": 2, "window": {"window_id": 39}, "include_web": False}
    assert parse({"label": "Deck", "window": "Deck.key", "text": ""})[0]["window"] == {"title": "Deck.key"}
    for bad, fragment in ((" {}", "empty"), ({"id": 3}, "unknown field"), ({"role": "Button", "nth": 0}, "1-based"), ({"role": "Button", "nth": True}, "1-based"),
                          ({"role": 3}, "non-empty string"), ({"role": "Button", "text": 5}, "must be a string"), ({"role": "Button", "window": 1.5}, "window_id or an exact title"),
                          ("not json", "not JSON"), ("[1]", "JSON object")):
        sel, err = parse(bad)
        assert sel == {} and fragment in err, (bad, err)
    assert native.choose(1, None, True) == "one" and native.choose(1, None, False) == "incomplete"
    assert native.choose(2, None, True) == "ambiguous" and native.choose(0, None, False) == "incomplete" and native.choose(2, 2, False) == "incomplete"
    assert native.matches({"role": "button"}, "AXButton", None, None, None) is True
    assert native.matches({"identifier": "a.b"}, "AXButton", None, None, "a.c") is False
    untrusted = FakeAX(deck(), windows=["win"], focused="win", trusted=False)
    out = act(untrusted, "press", {"identifier": "toolbar.play"})
    assert out == {"pid": 200, "do": "press", "select": {"identifier": "toolbar.play"}, "ok": False, "reason": "accessibility_untrusted", "dispatched": False}
    bad = act(FakeAX(deck(), windows=["win"], focused="win"), "set_value", {"role": "TextField"})
    assert bad["reason"] == "bad_selector" and "empty string to clear" in bad["error"]
    bad = act(FakeAX(deck(), windows=["win"], focused="win"), "press", {"role": "Button", "text": "x"})
    assert bad["reason"] == "bad_selector" and "press ignores it" in bad["error"]
    bad = act(FakeAX(deck(), windows=["win"], focused="win"), "toggle", {"role": "Button"})
    assert bad["reason"] == "unknown_action"


def test_the_helper_cli_dispatches_act_and_answers_one_json_line(monkeypatch, capsys):
    seen = {}

    def fake_act(pid, action, selector, max_nodes, timeout_ms):
        seen.update(pid=pid, action=action, selector=selector, max_nodes=max_nodes, timeout_ms=timeout_ms)
        return {"ok": False, "reason": "element_not_found", "dispatched": False}

    monkeypatch.setattr(native, "op_act", fake_act)
    assert native.main(["act", "200", "press", '{"role": "Button"}', "50", "700"]) == 1
    assert seen == {"pid": 200, "action": "press", "selector": '{"role": "Button"}', "max_nodes": 50, "timeout_ms": 700}
    assert json.loads(capsys.readouterr().out.strip())["reason"] == "element_not_found"
    assert native.main(["act", "200", "press"]) == 1 and seen["max_nodes"] == 600 and seen["timeout_ms"] == 3000


# --- the plugin: resolve, one helper call, pass the receipt through -------------------------------------------------

def helper_with_act(outcome):
    """A native-helper fake answering ``resolve`` as the shared fixture does and ``act`` with ``outcome(args)``."""
    def helper(args, state):
        if args[0] == "resolve":
            if args[1] in ("Keynote", "pid:200"):
                return {"ok": True, "candidates": [app_dict(KEYNOTE)]}
            return {"ok": False, "reason": "not_running", "candidates": [], "running": ["Safari", "Keynote"]}
        if args[0] == "act":
            return outcome(args)
        raise AssertionError(f"unexpected helper op {args}")
    return helper


COMPLETED = {"ok": True, "pid": 200, "do": "press", "dispatched": True, "action_outcome": "completed", "observed_effect": True,
             "element": {"role": "Button", "label": "Play"}, "element_after": {"role": "Button", "label": "Play"},
             "front_before": app_dict(SAFARI), "front_after": app_dict(SAFARI), "target_in_front": False, "note": "the element performed AXPress"}


def test_element_act_resolves_then_acts_in_one_helper_call_and_never_switches_or_clicks(tmp_path, monkeypatch):
    module, api, state = mac(tmp_path, monkeypatch, helper=helper_with_act(lambda args: {**COMPLETED, "do": args[2]}))
    out = receipt(api.tools["element_act"](app="Keynote", do="press", select={"role": "Button", "label": "Play"}, window=39))
    assert out["ok"] is True and marked(out) and out["action_outcome"] == "completed" and out["app"] == app_dict(KEYNOTE)
    assert [args[0] for args in state["helper"]] == ["resolve", "act"], "resolve, then one act call; never activate"
    op, pid, do, selector, max_nodes, timeout_ms = state["helper"][1]
    assert (pid, do, max_nodes, timeout_ms) == ("200", "press", "600", "3000")
    assert json.loads(selector) == {"role": "Button", "label": "Play", "include_web": False, "window": 39}
    assert state["front"] == SAFARI and inputs(state) == [], "no switch, no pointer: the front is reported, not changed"
    assert out["front_after"] == app_dict(SAFARI) and out["target_in_front"] is False and out["platform"] == "macos"
    text = receipt(api.tools["element_act"](app="pid:200", do="set_value", select={"role": "TextField"}, text="", max_nodes=5000, timeout_ms=50))
    assert json.loads(state["helper"][-1][3]) == {"role": "TextField", "include_web": False, "text": ""} and state["helper"][-1][4:] == ["2000", "100"]
    assert text["do"] == "set_value"


def test_element_act_refusals_send_nothing_and_say_why(tmp_path, monkeypatch):
    refusal = {"ok": False, "pid": 200, "do": "press", "reason": "element_ambiguous", "dispatched": False, "candidates": [{"index": 1}, {"index": 2}],
               "note": "no action was made", "front_before": app_dict(SAFARI), "front_after": app_dict(SAFARI)}
    module, api, state = mac(tmp_path, monkeypatch, helper=helper_with_act(lambda args: refusal))
    out = receipt(api.tools["element_act"](app="Keynote", do="press", select={"role": "Button", "label": "Play"}))
    assert out["ok"] is False and out["reason"] == "element_ambiguous" and out["dispatched"] is False and not marked(out)
    missing = receipt(api.tools["element_act"](app="Nope", do="press", select={"role": "Button"}))
    assert missing["ok"] is False and missing["reason"] == "not_running" and missing["dispatched"] is False and "nothing was sent" in missing["error"]
    assert [args[0] for args in state["helper"]] == ["resolve", "act", "resolve"], "an unresolved app never reaches act"
    for kwargs in ({"app": "", "do": "press", "select": {"role": "Button"}}, {"app": "Keynote", "do": "toggle", "select": {"role": "Button"}}):
        early = receipt(api.tools["element_act"](**kwargs))
        assert early["ok"] is False and early["dispatched"] is False and early["reason"] in ("app_required", "unknown_action"), early
    assert len(state["helper"]) == 3 and inputs(state) == []


def test_a_helper_that_gave_no_answer_or_stopped_mid_way_is_an_unknown_outcome_never_not_sent(tmp_path, monkeypatch):
    module, api, state = mac(tmp_path, monkeypatch, helper=helper_with_act(lambda args: {"ok": False, "error": "TypeError: boom", "op": "act"}))
    crashed = receipt(api.tools["element_act"](app="Keynote", do="press", select={"role": "Button"}))
    assert crashed["ok"] is False and crashed["reason"] == "helper_failed" and crashed["dispatched"] is None and crashed["action_outcome"] == "unknown"
    assert not marked(crashed) and "do not assume it did not" in crashed["note"] and crashed["select"] == {"role": "Button"}
    real_run = module.subprocess.run

    def silent_act(cmd, *a, **k):
        argv = [str(p) for p in cmd]
        if any(p.endswith("macos_native.py") for p in argv) and "act" in argv:
            state["calls"].append(argv)
            return subprocess.CompletedProcess(argv, 1, "", "TimeoutExpired: the helper was cut off")
        return real_run(cmd, *a, **k)

    monkeypatch.setattr(module.subprocess, "run", silent_act)
    lost = receipt(api.tools["element_act"](app="Keynote", do="press", select={"role": "Button"}))
    assert lost["ok"] is False and lost["reason"] == "helper_no_answer" and lost["dispatched"] is None and lost["action_outcome"] == "unknown"
    assert not marked(lost) and "cut off" in lost["error"]


def test_element_act_is_local_macos_only(tmp_path, monkeypatch):
    tools, transport = ssh_mac(tmp_path, monkeypatch)
    out = receipt(tools["element_act"](app="Keynote", do="press", select={"role": "Button"}))
    assert out["ok"] is False and out["reason"] == "unsupported" and out["backend"] == "ssh_macos" and out["dispatched"] is False and not marked(out)
    assert not [argv for argv in transport.calls if argv[:1] == ["ssh"]], "nothing goes over the wire"
    module, api, state = mac(tmp_path / "local", monkeypatch)
    monkeypatch.setattr(module, "_platform", lambda: "linux")
    monkeypatch.setattr(module, "_session_type", lambda: "x11")
    out = receipt(api.tools["element_act"](app="Firefox", do="press", select={"role": "Button"}))
    assert out["ok"] is False and out["reason"] == "unsupported" and out["platform"] == "linux" and state["helper"] == []


def test_element_act_is_registered_as_an_identity_addressed_action(tmp_path, monkeypatch):
    module, api, state = mac(tmp_path, monkeypatch)
    meta = api.meta["element_act"]
    assert meta["schema"]["required"] == ["app", "do", "select"] and meta["schema"]["properties"]["do"]["enum"] == ["press", "set_value"]
    assert set(meta["schema"]["properties"]["select"]["properties"]) == {"role", "label", "identifier", "nth"}
    assert meta["schema"]["properties"]["select"]["additionalProperties"] is False
    assert "app" not in meta["schema"]["properties"]["select"]["properties"] and meta["timeout_sec"] >= 45
    lead = meta["description"]
    assert "never by a mark or node id" in lead and "NOT brought forward" in lead and "do not retry blindly" in lead
    assert len(api.tools) == 25


def test_native_qa_interface_accepts_the_registered_schema(tmp_path, monkeypatch):
    from tests.test_macos_computer_use_native import _interface

    _module, api, _state = mac(tmp_path, monkeypatch)
    _interface(lambda name: api.meta[name]["schema"])


def test_native_qa_requests_follow_the_schema_through_real_selector_logic(tmp_path, monkeypatch):
    import jsonschema

    from tests.test_macos_computer_use_native import _act

    ax = FakeAX(deck(), windows=["win"], focused="win")
    _module, api, state = mac(tmp_path, monkeypatch, helper=helper_with_act(
        lambda args: act(ax, args[2], json.loads(args[3]), max_nodes=int(args[4]), timeout_ms=int(args[5]))))

    def call(name, **args):
        jsonschema.validate(args, api.meta[name]["schema"])
        return receipt(api.tools[name](**args))

    target = SimpleNamespace(pid=200, title="Deck.key")
    pressed = _act(call, target, "press", role="Button", identifier="toolbar.play")
    assert pressed["dispatched"] is True and pressed["action_outcome"] == "completed"
    assert pressed["front_before"] == pressed["front_after"] == app_dict(SAFARI)
    changed = _act(call, target, "set_value", role="TextField", label="Presenter notes", text="new")
    assert changed["verified"] is True and ax.sets == [("field", "new")]
    ambiguous = _act(call, target, "press", role="Button", label="Play")
    assert ambiguous["reason"] == "element_ambiguous" and ambiguous["dispatched"] is False
    second = _act(call, target, "press", role="Button", label="Play", nth=2)
    assert second["action_outcome"] == "completed" and ax.performed == [("play", "AXPress"), ("play2", "AXPress")]
    assert inputs(state) == [] and ax.balanced()
