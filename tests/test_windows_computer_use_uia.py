"""UI Automation reads and actions of ``skills/windows_computer_use`` over a comtypes-faithful provider.

No Windows, COM or desktop is involved. The fakes reproduce what comtypes 1.4.17
actually hands the skill, because the defects these tests pin live exactly there:

- an interface pointer is a ``c_void_p`` subclass whose ``.value`` is the pointer
  itself; a successful NULL out-param is a falsy pointer, never ``None``
  (``comtypes/_post_coinit/unknwn.py::_compointer_base``, ctypes out-params);
- ``GetCurrentPattern`` returns ``IUnknown``: pattern members exist only after
  ``QueryInterface``; ``GetCachedPatternAs`` requires the interface id;
- ``GetCachedPropertyValue`` returns VARIANT values as Python values
  (``BoundingRectangle`` is a left/top/width/height tuple of floats) and the
  not-supported sentinel as a non-NULL ``IUnknown`` pointer;
- ``TextRange.GetText(maxLength)`` cuts at maxLength UTF-16 units without
  saying so; -1 returns the whole text;
- a provider failure is a raised ``COMError`` carrying ``hresult``.

Property and pattern ids are those of UIAutomationClient.h (cross-checked against
the independent ``uiautomation`` 2.0.29 tables).
"""

from __future__ import annotations

import ctypes
import json
import sys
import threading
import types
from itertools import count

import pytest

import tests._windows_computer_use  # noqa: F401  (registers the payload package)
from windows_computer_use import core, plugin, tree, win_uia  # noqa: E402

E_POINTER = -2147467261
E_INVALIDARG = -2147024809
UIA_E_ELEMENTNOTAVAILABLE = -2147220991
_ids = count(1)


class COMError(Exception):
    def __init__(self, hresult: int, text: str = "provider failure"):
        super().__init__(text)
        self.hresult = hresult


class ComPtr(ctypes.c_void_p):
    """comtypes ``_compointer_base``: ``.value`` is the pointer itself."""

    value = property(lambda self: self)


NOT_SUPPORTED = ComPtr(0xDEAD)  # ReservedNotSupportedValue arrives as an IUnknown pointer


def null(kind=ComPtr):
    return kind()


class Iface:
    """Marker classes standing for the generated comtypes interfaces."""


IFACES = {name: type(name, (Iface,), {}) for name in (
    "IUIAutomationValuePattern", "IUIAutomationTextPattern", "IUIAutomationInvokePattern",
    "IUIAutomationTogglePattern", "IUIAutomationExpandCollapsePattern",
    "IUIAutomationSelectionItemPattern", "IUIAutomationScrollItemPattern",
)}
PATTERN_IFACE = {
    10002: "IUIAutomationValuePattern", 10014: "IUIAutomationTextPattern", 10000: "IUIAutomationInvokePattern",
    10015: "IUIAutomationTogglePattern", 10005: "IUIAutomationExpandCollapsePattern",
    10010: "IUIAutomationSelectionItemPattern", 10017: "IUIAutomationScrollItemPattern",
}


@pytest.fixture(autouse=True)
def fake_comtypes(monkeypatch):
    gen = types.ModuleType("comtypes.gen.UIAutomationClient")
    gen.__dict__.update(IFACES)
    gen.AutomationElementMode_Full = 1
    gen.TreeScope_Element = 1
    comtypes = types.ModuleType("comtypes")
    comtypes.gen = types.ModuleType("comtypes.gen")
    comtypes.gen.UIAutomationClient = gen
    monkeypatch.setitem(sys.modules, "comtypes", comtypes)
    monkeypatch.setitem(sys.modules, "comtypes.gen", comtypes.gen)
    monkeypatch.setitem(sys.modules, "comtypes.gen.UIAutomationClient", gen)


class Calls(list):
    pass


class Pattern(ComPtr):
    """A concrete pattern interface (after QueryInterface)."""

    def __init__(self, calls, element, *, value=None, text=None):
        super().__init__(next(_ids))
        self.calls, self.element, self._value, self._text = calls, element, value, text

    @property
    def CurrentValue(self):
        return self._value

    @property
    def DocumentRange(self):
        pattern = self

        class Range:
            def GetText(self, max_length):
                # ITextRangeProvider::GetText: the text "possibly truncated at the specified
                # maximum length" - BSTR characters, i.e. UTF-16 units - or all of it for -1.
                pattern.calls.append(("GetText", pattern.element.label, max_length))
                if max_length < -1:
                    raise COMError(E_INVALIDARG, "E_INVALIDARG")
                text = pattern._text or ""
                if max_length == -1:
                    return text
                return text.encode("utf-16-le", "surrogatepass")[: 2 * max_length].decode("utf-16-le", "surrogatepass")
        return Range()

    def SetValue(self, text):
        self.calls.append(("SetValue", self.element.label, text))

    def Invoke(self):
        self.calls.append(("Invoke", self.element.label))

    def Toggle(self):
        self.calls.append(("Toggle", self.element.label))

    def Expand(self):
        self.calls.append(("Expand", self.element.label))

    def Collapse(self):
        self.calls.append(("Collapse", self.element.label))

    def Select(self):
        self.calls.append(("Select", self.element.label))

    def ScrollIntoView(self):
        self.calls.append(("ScrollIntoView", self.element.label))


class Unknown(ComPtr):
    """What GetCurrentPattern returns: IUnknown, members only after QueryInterface."""

    def __init__(self, iface_name, concrete):
        super().__init__(next(_ids))
        self.iface_name, self.concrete = iface_name, concrete

    def QueryInterface(self, iface):
        if iface is not IFACES[self.iface_name]:
            raise COMError(-2147467262, "E_NOINTERFACE")
        return self.concrete


class Element(ComPtr):
    """An IUIAutomationElement; NULL instances fail like comtypes NULL pointers."""

    def setup(self, calls, label, props, *, patterns=None, fail_cached=(), fail_live=(), fail_patterns=(),
              refuse_children=False, children=()):
        self.calls, self.label, self.props = calls, label, dict(props)
        self.patterns = dict(patterns or {})
        self.fail_cached, self.fail_live, self.fail_patterns = set(fail_cached), set(fail_live), set(fail_patterns)
        self.refuse_children, self.children, self.parent = refuse_children, list(children), None
        for child in self.children:
            child.parent = self
        return self

    def _alive(self):
        if not self:
            raise ValueError("NULL COM pointer access")

    def GetCachedPropertyValue(self, prop_id):
        self._alive()
        if prop_id in self.fail_cached:
            raise COMError(UIA_E_ELEMENTNOTAVAILABLE)
        return self.props.get(prop_id, NOT_SUPPORTED)

    def GetCurrentPropertyValue(self, prop_id):
        self._alive()
        return self.props.get(prop_id, NOT_SUPPORTED)

    def GetCachedPatternAs(self, pattern_id, riid):  # the generated signature: riid is required
        self._alive()
        raise AssertionError("cached pattern objects are not part of the cache request")

    def GetCurrentPattern(self, pattern_id):
        self._alive()
        if pattern_id in self.fail_patterns:
            raise COMError(UIA_E_ELEMENTNOTAVAILABLE)
        if pattern_id not in self.patterns:
            return null()
        return Unknown(PATTERN_IFACE[pattern_id], self.patterns[pattern_id])

    def _live(self, field, prop_id):
        self._alive()
        if field in self.fail_live:
            raise COMError(UIA_E_ELEMENTNOTAVAILABLE)
        return self.props.get(prop_id, "")

    CurrentName = property(lambda self: self._live("name", 30005))
    CurrentAutomationId = property(lambda self: self._live("automation_id", 30011))
    CurrentControlType = property(lambda self: self._live("role", 30003))

    def SetFocus(self):
        self.calls.append(("SetFocus", self.label))


def make(calls, label, control_type, *, name=None, aid=None, extra=None, **kwargs):
    props = {30003: control_type, 30005: name or "", 30011: aid or label, 30012: "", 30004: "",
             30010: True, 30008: False, 30022: False, 30020: 0, 30043: False, 30041: False,
             30036: False, 30028: False}
    props.update(extra or {})
    element = Element(next(_ids)).setup(calls, label, props, **kwargs)
    for pattern_id, spec in list(element.patterns.items()):
        element.patterns[pattern_id] = Pattern(calls, element, **(spec or {}))
    return element


class Walker:
    def __init__(self, calls):
        self.calls = calls

    def GetFirstChildElementBuildCache(self, element, cache):
        if not element:
            raise COMError(E_POINTER, "E_POINTER")
        if element.refuse_children:
            raise COMError(E_POINTER, "E_POINTER")
        return element.children[0] if element.children else null(Element)

    def GetNextSiblingElementBuildCache(self, element, cache):
        if not element:
            raise COMError(E_POINTER, "E_POINTER")
        siblings = element.parent.children if element.parent else [element]
        index = siblings.index(element)
        return siblings[index + 1] if index + 1 < len(siblings) else null(Element)


class CacheRequest:
    def __init__(self):
        self.properties, self.patterns = [], []

    def AddProperty(self, prop_id):
        self.properties.append(prop_id)

    def AddPattern(self, pattern_id):
        self.patterns.append(pattern_id)


class Automation:
    def __init__(self, root):
        self.root = root
        self.ControlViewWalker = Walker(root.calls)
        self.ControlViewCondition = object()
        self.caches = []

    def CreateCacheRequest(self):
        self.caches.append(CacheRequest())
        return self.caches[-1]

    def ElementFromHandleBuildCache(self, hwnd, cache):
        return self.root


def window(hwnd=1):
    return tree.WindowInfo(hwnd, "dialog", "Dialog", 42, None, True, "argument", 96)


def act(root, args):
    plan = core.act_plan({"hwnd": 1, "timeout_ms": 0, **args})
    return win_uia.act_in_session(Automation(root), plan, window(), lambda hwnd: 42)


def find(root, args):
    plan = core.find_plan({"hwnd": 1, **args})
    return win_uia.find_in_session(Automation(root), plan, 1)


def read(root, **args):
    plan = core.read_plan({"hwnd": 1, **args})
    return win_uia.read_in_session(Automation(root), plan, window())


def dialog(calls, *children):
    return make(calls, "dialog", 50032, name="Dialog", children=children)


# ── selection: a failed read of a named field is never absence ────────────────

def test_an_unreadable_twin_makes_act_refuse_instead_of_invoking_the_other():
    calls = Calls()
    first = make(calls, "ok-1", 50000, name="OK", patterns={10000: None})
    second = make(calls, "ok-2", 50000, name="OK", patterns={10000: None}, fail_cached={30005})
    root = dialog(calls, first, second)

    result = act(root, {"select": {"role": "button", "name": "OK"}, "do": "invoke"})

    assert result["ok"] is False and result["reason"] == "unreadable_candidates"
    assert result["matched"] == 1 and result["unreadable"] == 1
    assert result["unreadable_candidates"][0]["unread"] == ["name"]
    assert result["searched_whole_window"] is False and result["polls"] == 1
    assert "stopped_before_timeout_ms" not in result and "last_walk_ms" not in result
    assert not [call for call in calls if call[0] == "Invoke"]


def test_find_reports_the_unreadable_twin_and_does_not_claim_the_whole_window():
    calls = Calls()
    root = dialog(calls,
                  make(calls, "ok-1", 50000, name="OK"),
                  make(calls, "ok-2", 50000, name="OK", fail_cached={30005}))

    result = find(root, {"select": {"role": "button", "name": "OK"}})

    assert result["ok"] is True and result["matched"] == 1
    assert result["searched_whole_window"] is False and result["unreadable"] == 1

    lonely = find(dialog(calls, make(calls, "ok-2", 50000, name="OK", fail_cached={30005})),
                  {"select": {"name": "OK"}})
    assert lonely["ok"] is False and lonely["reason"] == "unreadable_candidates"


def test_a_readable_mismatch_or_an_unasked_field_does_not_block_the_choice():
    calls = Calls()
    target = make(calls, "save", 50000, name="Save", patterns={10000: None})
    # A text element whose name read fails cannot be the button the selector names.
    other_role = make(calls, "label", 50020, fail_cached={30005})
    by_role = act(dialog(calls, target, other_role), {"select": {"role": "button", "name": "Save"}, "do": "invoke"})
    # A button whose name fails could be "Save" for a role+name selector, but not
    # for one that names only the automation id it answers.
    unasked = make(calls, "cancel", 50000, fail_cached={30005})
    both = dialog(calls, target, unasked)
    by_id = act(both, {"select": {"automation_id": "save"}, "do": "invoke"})
    competing = act(both, {"select": {"role": "button", "name": "Save"}, "do": "invoke"})

    assert by_role["ok"] is True and by_id["ok"] is True
    assert competing["reason"] == "unreadable_candidates"
    assert calls.count(("Invoke", "save")) == 2


def test_an_unreadable_role_counts_when_the_selector_names_a_role():
    calls = Calls()
    root = dialog(calls,
                  make(calls, "ok-1", 50000, name="OK", patterns={10000: None}),
                  make(calls, "ok-2", 50000, name="OK", fail_cached={30003}))

    result = act(root, {"select": {"role": "button", "name": "OK"}, "do": "invoke"})

    assert result["reason"] == "unreadable_candidates"
    # Without the control type, the description cannot know whether a live
    # proxy value read applies either. Preserve that failed field, not just
    # the selector field that made this a competing candidate.
    assert set(result["unreadable_candidates"][0]["unread"]) == {"role", "value"}


# ── traversal: NULL ends the list, a raised call refuses ──────────────────────

def test_null_children_end_the_list_without_phantom_nodes():
    calls = Calls()
    root = dialog(calls, make(calls, "a", 50000, name="A"), make(calls, "b", 50020, name="B"))

    reading = read(root)

    assert reading["nodes_read"] == 3
    assert [child["role"] for child in reading["root"]["children"]] == ["button", "text"]
    assert all("children" not in child for child in reading["root"]["children"])
    assert reading["truncated_by"] == [] and reading["total_known"] is True


def test_a_walker_error_is_a_refusal_never_the_end_of_the_children():
    calls = Calls()
    broken = make(calls, "pane", 50033, name="Pane", refuse_children=True)
    root = dialog(calls, broken)

    reading = read(root)
    pane = reading["root"]["children"][0]
    assert pane["children_unread"] == "provider_refused"
    assert "provider_refused" in reading["truncated_by"] and reading["total_known"] is False

    with pytest.raises(win_uia.UIAError, match="traversal incomplete"):
        act(root, {"select": {"name": "Pane"}, "do": "focus"})


# ── waiting: the last whole walk answers; a walk the deadline cut stays a timeout

class Clock:
    """win_uia's clock: time passes between polls, and where a test stalls a read."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def stall_second_walk(clock, element, *, rename=None):
    """The second walk that reads ``element`` stalls past any deadline on it."""
    read = element.GetCachedPropertyValue
    walks = []

    def stalling(prop_id):
        if prop_id == 30003:
            walks.append(clock.now)
            if len(walks) == 2:
                clock.now += 30
                if rename:
                    element.props[30005] = rename
        return read(prop_id)

    element.GetCachedPropertyValue = stalling


def test_act_answers_not_found_from_the_last_whole_walk_instead_of_starting_a_cut_re_poll(monkeypatch):
    # The real-desktop sequence (run 39058404823): every walk of a small window
    # completed, then a poll begun after the 3000 ms wait ran out was cut before
    # its first node and the answer read 'timeout, 0 nodes'. A whole walk that
    # ends inside the last pause is now the final look.
    monkeypatch.setattr(win_uia, "time", Clock())
    calls = Calls()
    root = dialog(calls, make(calls, "a", 50000, name="A"), make(calls, "b", 50020, name="B"))
    plan = core.act_plan({"hwnd": 1, "select": {"name": "no such element"}, "do": "focus"})

    result = win_uia.act_in_session(Automation(root), plan, window(), lambda hwnd: 42)

    assert result["reason"] == "not_found" and result["searched_whole_window"] is True
    assert result["nodes_scanned"] == 3 and result["polls"] > 1
    assert 3000 - 1000 * win_uia.POLL_PAUSE_S - 1 <= result["waited_ms"] <= 3000


def test_act_keeps_a_timeout_when_the_target_appears_in_the_part_of_a_walk_the_deadline_cut(monkeypatch):
    # The target appears after a whole walk; the next walk stalls past the
    # deadline before reaching it. That walk began in time, so its cut is the
    # answer - a 'not_found' from the older walk would deny what is now there.
    clock = Clock()
    monkeypatch.setattr(win_uia, "time", clock)
    calls = Calls()
    first = make(calls, "a", 50000, name="A")
    target = make(calls, "late", 50000, name="Later", patterns={10000: None})
    root = dialog(calls, first, target)
    read = first.GetCachedPropertyValue
    seen = []

    def stalling(prop_id):
        if prop_id == 30003:
            seen.append(clock.now)
            if len(seen) == 2:
                target.props[30005] = "Go"
                clock.now += 30
        return read(prop_id)

    first.GetCachedPropertyValue = stalling
    plan = core.act_plan({"hwnd": 1, "select": {"name": "Go"}, "do": "invoke"})

    result = win_uia.act_in_session(Automation(root), plan, window(), lambda hwnd: 42)

    assert result["reason"] == "timeout" and result["searched_whole_window"] is False
    assert result["polls"] == 2 and ("Invoke", "late") not in calls


def test_a_cut_walk_that_met_a_candidate_stays_a_timeout(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(win_uia, "time", clock)
    calls = Calls()
    late = make(calls, "late", 50000, name="Later", patterns={10000: None})
    root = dialog(calls, late, make(calls, "b", 50020, name="B"))
    stall_second_walk(clock, late, rename="Go")
    plan = core.act_plan({"hwnd": 1, "select": {"name": "Go"}, "do": "invoke"})

    result = win_uia.act_in_session(Automation(root), plan, window(), lambda hwnd: 42)

    assert result["reason"] == "timeout" and result["searched_whole_window"] is False
    assert result["matched"] == 1 and result["polls"] == 2
    assert ("Invoke", "late") not in calls


def test_find_answers_not_found_from_the_last_whole_walk_of_its_wait(monkeypatch):
    monkeypatch.setattr(win_uia, "time", Clock())
    calls = Calls()
    root = dialog(calls, make(calls, "a", 50000, name="A"), make(calls, "b", 50020, name="B"))

    result = find(root, {"select": {"name": "no such element"}, "timeout_ms": 1000})

    assert result["reason"] == "not_found" and result["searched_whole_window"] is True
    assert result["nodes_scanned"] == 3 and result["polls"] > 1


def test_find_keeps_a_timeout_when_a_re_poll_begun_in_time_is_cut(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(win_uia, "time", clock)
    calls = Calls()
    slow = make(calls, "slow", 50000, name="Slow")
    root = dialog(calls, slow, make(calls, "b", 50020, name="B"))
    stall_second_walk(clock, slow)

    result = find(root, {"select": {"name": "no such element"}, "timeout_ms": 1000})

    assert result["reason"] == "timeout" and result["searched_whole_window"] is False
    assert result["polls"] == 2


def slow_walks(clock, element, seconds):
    """Every walk that reads ``element`` takes ``seconds`` of the clock."""
    read = element.GetCachedPropertyValue

    def slow(prop_id):
        if prop_id == 30003:
            clock.now += seconds
        return read(prop_id)

    element.GetCachedPropertyValue = slow


def test_act_does_not_start_a_re_poll_that_cannot_finish_at_the_last_walks_pace(monkeypatch):
    # Run 39063120643: walks of a small window took real time, the last re-poll
    # began inside the wait and was cut after 8 nodes -> 'timeout' although every
    # earlier walk had read the window whole. Act now stops while the last whole
    # walk is still the newest look.
    clock = Clock()
    monkeypatch.setattr(win_uia, "time", clock)
    calls = Calls()
    a = make(calls, "a", 50000, name="A")
    root = dialog(calls, a, make(calls, "b", 50020, name="B"))
    slow_walks(clock, a, 0.25)
    plan = core.act_plan({"hwnd": 1, "select": {"name": "no such element"}, "do": "focus"})

    result = win_uia.act_in_session(Automation(root), plan, window(), lambda hwnd: 42)

    assert result["reason"] == "not_found" and result["searched_whole_window"] is True
    assert result["nodes_scanned"] == 3 and result["waited_ms"] <= 3000


@pytest.mark.parametrize("unreadable", [False, True])
def test_act_discloses_an_early_stop_after_a_slow_first_walk_then_fast_walks(monkeypatch, unreadable):
    clock = Clock()
    monkeypatch.setattr(win_uia, "time", clock)
    monkeypatch.setattr(win_uia, "POLL_PAUSE_S", 0.125)  # exact binary fractions
    calls = Calls()
    root = dialog(calls, make(calls, "a", 50000, name="A", fail_cached={30005} if unreadable else ()))
    automation = Automation(root)
    walks = []

    def fetch(hwnd, cache):
        walks.append(clock.now)
        clock.now += 1.25 if len(walks) == 1 else 0.125
        return root

    automation.ElementFromHandleBuildCache = fetch
    plan = core.act_plan({"hwnd": 1, "select": {"name": "no such element"}, "do": "focus"})

    result = win_uia.act_in_session(automation, plan, window(), lambda hwnd: 42)

    assert result["reason"] == ("unreadable_candidates" if unreadable else "not_found")
    assert result["polls"] == len(walks) == 7 and result["matched"] == 0
    assert result["waited_ms"] == 2750 and result["stopped_before_timeout_ms"] == 250
    assert result["last_walk_ms"] == 125
    assert "another walk at the last pace" in result["hint"] and "call element_act again" in result["hint"]


def test_act_measures_an_early_stop_against_timeout_ms_not_the_grace_floor(monkeypatch):
    # timeout_ms=500 already waits past itself to the 750 ms grace floor; that is
    # no early stop and must not tell the agent to look again.
    clock = Clock()
    monkeypatch.setattr(win_uia, "time", clock)
    monkeypatch.setattr(win_uia, "POLL_PAUSE_S", 0.125)
    calls = Calls()
    root = dialog(calls, make(calls, "a", 50000, name="A"))
    automation = Automation(root)

    def fetch(hwnd, cache):
        clock.now += 0.25
        return root

    automation.ElementFromHandleBuildCache = fetch
    plan = core.act_plan({"hwnd": 1, "select": {"name": "no such element"}, "do": "focus", "timeout_ms": 500})

    result = win_uia.act_in_session(automation, plan, window(), lambda hwnd: 42)

    assert result["reason"] == "not_found" and result["polls"] == 2 and result["waited_ms"] == 625
    assert "stopped_before_timeout_ms" not in result and "last_walk_ms" not in result
    assert "call element_act again" not in result["hint"]


@pytest.mark.parametrize("wake_at", [3.0, 3.2])
def test_act_answers_from_the_last_walk_when_a_pause_reaches_or_oversleeps_the_deadline(monkeypatch, wake_at):
    # A scheduler can wake the pause late; a sweep started past the deadline would
    # be cut before its first node.
    class Late(Clock):
        def sleep(self, seconds):
            self.now = wake_at if self.now > 2.7 else self.now + seconds

    monkeypatch.setattr(win_uia, "time", Late())
    calls = Calls()
    root = dialog(calls, make(calls, "a", 50000, name="A"), make(calls, "b", 50020, name="B"))
    plan = core.act_plan({"hwnd": 1, "select": {"name": "no such element"}, "do": "focus"})

    result = win_uia.act_in_session(Automation(root), plan, window(), lambda hwnd: 42)

    assert result["reason"] == "not_found" and result["searched_whole_window"] is True
    assert result["nodes_scanned"] == 3
    assert "stopped_before_timeout_ms" not in result and "last_walk_ms" not in result


def test_find_takes_its_last_look_at_the_end_of_the_wait(monkeypatch):
    # This small window fits in SWEEP_MS, so a look begun at the deadline is
    # whole: an element that appears in the final pause is still found.
    clock = Clock()
    monkeypatch.setattr(win_uia, "time", clock)
    calls = Calls()
    late = make(calls, "late", 50000, name="Later")
    root = dialog(calls, late, make(calls, "b", 50020, name="B"))
    read = late.GetCachedPropertyValue

    def appearing(prop_id):
        if prop_id == 30003 and clock.now >= 0.98:
            late.props[30005] = "Go"
        return read(prop_id)

    late.GetCachedPropertyValue = appearing

    result = find(root, {"select": {"name": "Go"}, "timeout_ms": 1000})

    assert result["ok"] is True and result["matched"] == 1


# ── window_read: values, states, rectangles and visibility from the cache ─────

def test_window_read_shows_values_states_and_rectangles():
    calls = Calls()
    edit = make(calls, "amount", 50004, name="Amount", extra={
        30043: True, 30045: "1500", 30001: (10.0, 20.0, 100.0, 30.0)})
    check = make(calls, "agree", 50002, name="Agree", extra={30041: True, 30086: 1})
    item = make(calls, "row", 50007, name="Row", extra={30036: True, 30079: True})
    combo = make(calls, "kind", 50003, name="Kind", extra={30028: True, 30070: 1})
    helped = make(calls, "hint", 50020, name="Hint", extra={30013: "a help text", 30022: False})
    hidden = make(calls, "far", 50020, name="Far", extra={30022: True})
    root = dialog(calls, edit, check, item, combo, helped, hidden)

    nodes = {node["automation_id"]: node for node in read(root, shape="flat")["items"]}

    assert nodes["amount"]["value"] == "1500"
    assert nodes["amount"]["rect"]["left"] == 10 and nodes["amount"]["rect"]["right"] == 110
    assert nodes["amount"]["rect"]["center"] == {"x": 60, "y": 35}
    assert nodes["agree"]["state"]["checked"] == "on"
    assert nodes["row"]["state"]["selected"] is True
    assert nodes["kind"]["state"]["expanded"] == "expanded"
    assert nodes["hint"]["state"]["offscreen"] is False
    assert nodes["far"]["state"]["offscreen"] is True
    # States an element does not expose stay absent, not false.
    assert "checked" not in nodes["amount"].get("state", {})
    visible = {node["automation_id"] for node in read(root, shape="flat", visible_only=True)["items"]}
    assert "far" not in visible and "hint" in visible


def test_the_cache_request_carries_the_pattern_state_properties():
    calls = Calls()
    automation = Automation(dialog(calls))
    win_uia.read_in_session(automation, core.read_plan({"hwnd": 1}), window())

    cache = automation.caches[0]
    assert {30045, 30043, 30086, 30041, 30079, 30036, 30070, 30028, 30022} <= set(cache.properties)
    assert cache.patterns == []
    # Ids of UIAutomationClient.h the reads depend on.
    assert (win_uia.UIA_IsOffscreenPropertyId, win_uia.UIA_TextPatternId, win_uia.UIA_ValuePatternId,
            win_uia.UIA_TogglePatternId, win_uia.UIA_ScrollItemPatternId) == (30022, 10014, 10002, 10015, 10017)


def test_proxy_edits_and_documents_are_read_live_through_their_interfaces():
    calls = Calls()
    proxy = make(calls, "proxy", 50004, extra={30043: True, 30045: ""},
                 patterns={10002: {"value": "typed into WinForms"}})
    document = make(calls, "doc", 50030, patterns={10014: {"text": "RichEdit body"}})
    root = dialog(calls, proxy, document)

    nodes = {node["automation_id"]: node for node in read(root, shape="flat")["items"]}

    assert nodes["proxy"]["value"] == "typed into WinForms"
    assert nodes["doc"]["value"] == "RichEdit body"


def test_a_failed_read_is_listed_on_the_node_instead_of_looking_absent():
    calls = Calls()
    flaky = make(calls, "flaky", 50004, name="Flaky", fail_cached={30005}, fail_patterns={10002, 10014})
    root = dialog(calls, flaky)

    reading = read(root, shape="flat")
    node = next(item for item in reading["items"] if item["automation_id"] == "flaky")

    assert set(node["unread"]) == {"name", "value"} and "name" not in node
    assert reading["nodes_with_unread_fields"] == 1


# ── actions: unsupported patterns, the pre-action re-read, receipts ───────────

def test_a_missing_pattern_is_refused_as_unsupported_with_the_real_list():
    calls = Calls()
    toggle_only = make(calls, "agree", 50002, name="Agree", patterns={10015: None}, fail_patterns={10005})
    root = dialog(calls, toggle_only)

    with pytest.raises(win_uia.UIAError) as refused:
        act(root, {"select": {"automation_id": "agree"}, "do": "invoke"})

    message = str(refused.value)
    assert "does not support InvokePattern" in message and "cast failed" not in message
    assert "patterns it does support: toggle, could not query expand/collapse" in message


def test_a_failed_pattern_query_is_a_provider_error_not_absence():
    calls = Calls()
    root = dialog(calls, make(calls, "go", 50000, name="Go", fail_patterns={10000}))

    with pytest.raises(win_uia.UIAError, match="provider error"):
        act(root, {"select": {"automation_id": "go"}, "do": "invoke"})


def test_a_document_matched_by_its_text_passes_the_same_reread_and_acts():
    calls = Calls()
    document = make(calls, "doc", 50030, patterns={10014: {"text": "Invoice 42 draft"}})
    root = dialog(calls, document)

    result = act(root, {"select": {"role": "document", "value_contains": "invoice 42"}, "do": "focus"})

    assert result["ok"] is True and ("SetFocus", "doc") in calls
    assert result["element"]["value"] == "Invoice 42 draft"


def test_the_action_aborts_when_the_chosen_element_cannot_be_reread():
    calls = Calls()
    root = dialog(calls, make(calls, "go", 50000, name="Go", patterns={10000: None}, fail_live={"name"}))

    with pytest.raises(win_uia.UIAError, match="re-reading the chosen element"):
        act(root, {"select": {"role": "button", "name": "Go"}, "do": "invoke"})
    assert ("Invoke", "go") not in calls


@pytest.mark.parametrize("verb,pattern_id,call", [
    ("invoke", 10000, "Invoke"), ("set_value", 10002, "SetValue"), ("toggle", 10015, "Toggle"),
    ("expand", 10005, "Expand"), ("collapse", 10005, "Collapse"), ("select", 10010, "Select"),
    ("scroll_into_view", 10017, "ScrollIntoView"), ("focus", None, "SetFocus"),
])
def test_every_verb_goes_through_its_pattern_and_receipts_before_and_after(verb, pattern_id, call):
    calls = Calls()
    patterns = {pattern_id: None} if pattern_id else {}
    root = dialog(calls, make(calls, "target", 50000, name="Target", patterns=patterns))
    args = {"select": {"automation_id": "target"}, "do": verb}
    if verb == "set_value":
        args["text"] = "new"

    result = act(root, args)

    assert result["ok"] is True and result["did"] == verb
    assert result["mutating"] is (verb not in core.NON_MUTATING_ACTS)
    assert [entry for entry in calls if entry[0] == call][0][1] == "target"
    assert result["element"]["automation_id"] == "target" and "element_after" in result


def test_two_readable_matches_are_still_answered_with_the_candidates():
    calls = Calls()
    root = dialog(calls, make(calls, "a", 50000, name="OK"), make(calls, "b", 50000, name="OK"))

    result = act(root, {"select": {"name": "OK"}, "do": "invoke"})
    second = act(root, {"select": {"name": "OK", "nth": 1}, "do": "focus"})

    assert result["reason"] == "ambiguous" and result["matched"] == 2
    assert {item["automation_id"] for item in result["candidates"]} == {"a", "b"}
    assert second["ok"] is True and ("SetFocus", "b") in calls


# ── the plugin's tool contract ────────────────────────────────────────────────

class _API:
    def __init__(self):
        self.tools = {}

    def register_tool(self, name, handler, **metadata):
        self.tools[name] = {"handler": handler, **metadata}


def test_the_four_tools_register_with_all_eight_verbs():
    api = _API()
    plugin.register(api)

    assert set(api.tools) == {"window_list", "window_read", "element_find", "element_act"}
    verbs = api.tools["element_act"]["schema"]["properties"]["do"]["enum"]
    assert verbs == list(core.ACTS) and len(verbs) == 8
    assert "cheaper" not in api.tools["window_read"]["description"]


def test_tool_answers_are_json_strings(monkeypatch):
    calls = Calls()
    root = dialog(calls, make(calls, "go", 50000, name="Go", patterns={10000: None}))
    fake_win32 = types.ModuleType("windows_computer_use.win32fb")
    fake_win32.resolve_window = lambda hwnd: window()
    fake_win32.window_pid = lambda hwnd: 42
    fake_win32.list_windows = lambda limit: [{"hwnd": "0x1", "title": "Dialog"}]
    monkeypatch.setitem(sys.modules, "windows_computer_use.win32fb", fake_win32)
    monkeypatch.setattr(sys.modules["windows_computer_use"], "win32fb", fake_win32, raising=False)
    monkeypatch.setattr(plugin, "_ensure_windows", lambda: None)
    threads = {}

    def client_module():
        threads["wrapper"] = threading.current_thread().name
        return sys.modules["comtypes.gen.UIAutomationClient"]

    def thread_automation():
        threads["automation"] = threading.current_thread().name
        return Automation(root)

    monkeypatch.setattr(win_uia, "client_module", client_module)
    monkeypatch.setattr(win_uia, "thread_automation", thread_automation)
    api = _API()
    plugin.register(api)

    listed = api.tools["window_list"]["handler"](limit=5)
    acted = api.tools["element_act"]["handler"](hwnd=1, select={"automation_id": "go"}, do="invoke")

    assert isinstance(listed, str) and json.loads(listed)["count"] == 1
    assert isinstance(acted, str) and json.loads(acted)["ok"] is True
    assert ("Invoke", "go") in calls
    # The generated wrapper is readied before the deadline, outside the worker
    # that may be abandoned; UIA calls run in the worker.
    assert threads["automation"] == "wcu-worker" and threads["wrapper"] != "wcu-worker"


def test_a_wrapper_that_cannot_be_prepared_falls_back_or_refuses(monkeypatch):
    fake_win32 = types.ModuleType("windows_computer_use.win32fb")
    fake_win32.resolve_window = lambda hwnd: window()
    fake_win32.win32_walk = lambda win, limits, visible_only=False: tree.Walk()
    monkeypatch.setitem(sys.modules, "windows_computer_use.win32fb", fake_win32)
    monkeypatch.setattr(sys.modules["windows_computer_use"], "win32fb", fake_win32, raising=False)
    monkeypatch.setattr(plugin, "_ensure_windows", lambda: None)

    def broken():
        raise OSError("type library not registered")

    monkeypatch.setattr(win_uia, "client_module", broken)
    monkeypatch.setattr(win_uia, "thread_automation", lambda: pytest.fail("no worker may start"))
    api = _API()
    plugin.register(api)

    reading = json.loads(api.tools["window_read"]["handler"](hwnd=1))
    acted = json.loads(api.tools["element_act"]["handler"](hwnd=1, select={"name": "x"}, do="focus"))

    assert reading["backend"] == "win32" and "type library not registered" in reading["fallback_reason"]
    assert acted["ok"] is False and "type library not registered" in acted["error"]


@pytest.mark.skipif(sys.platform == "win32", reason="the refusal is the non-Windows answer")
def test_the_tools_refuse_outside_windows():
    api = _API()
    plugin.register(api)

    with pytest.raises(plugin.ToolError, match="interactive Windows session"):
        api.tools["window_list"]["handler"]()


@pytest.mark.parametrize("shape", ["tree", "flat"])
@pytest.mark.parametrize("label", ['"' * 4000, "\\" * 4000, "\u0001" * 4000, "界🙂" * 2000],
                         ids=["quotes", "backslashes", "control", "unicode"])
def test_large_read_survives_actual_child_result_encoding(tmp_path, shape, label):
    from ouroboros.extension_process_runner import _RESULT_CAP, _write_child_result

    walk = tree.Walk(nodes=[tree.Node(
        parent=0 if index else None, role="text", name=label,
    ) for index in range(100)])
    plan = core.read_plan({"hwnd": 1, "max_nodes": 100, "max_text_chars": 4000, "shape": shape})
    result = tree.render(plan, window(), walk, "uia", "", None, 1)
    # Exercise both production serialization layers, rather than measuring
    # the inner object and assuming how much the relay will escape it.
    reply = plugin._json_tool(lambda: result)()
    path = tmp_path / "child-result.json"
    _write_child_result({"result_path": str(path)}, {"ok": True, "result": reply})
    assert path.stat().st_size <= _RESULT_CAP
    delivered = json.loads(path.read_text(encoding="utf-8"))
    assert delivered["ok"] is True, delivered
    assert json.loads(delivered["result"]) == result
    assert result["nodes_returned"] > 0
    if result["nodes_read"] < 100:
        assert "result_size" in result["truncated_by"]
        assert not result["total_known"]
        assert result["discovered_unread"] == 100 - result["nodes_read"]


def test_a_reading_too_large_for_the_host_is_cut_and_says_so():
    """Through the host an out-of-process answer is relayed only up to 512 KiB; a
    5000-node ceiling reading must arrive as a named cut, not as a relay error."""
    def walk_of(count):
        walk = tree.Walk()
        for index in range(count):
            walk.nodes.append(tree.Node(
                parent=(index - 1) // 4 if index else None, depth=0, role="button", name=f"Element label {index}",
                automation_id=f"auto_{index}", cls="Button", rect=tree.Rect(10, 20, 110, 50), enabled=True))
        return walk

    def rendered(count, shape):
        plan = core.read_plan({"hwnd": 1, "max_nodes": count, "shape": shape})
        return tree.render(plan, window(), walk_of(count), backend="uia", backend_detail="", fallback_reason=None,
                           elapsed_ms=1)

    for shape in ("tree", "flat"):
        big = rendered(5000, shape)
        reply = plugin._json_tool(lambda: big)()
        assert len(json.dumps(reply, ensure_ascii=False).encode("utf-8")) <= tree.RESULT_BYTES
        assert "result_size" in big["truncated_by"] and big["total_known"] is False
        assert 0 < big["nodes_read"] < 5000 and big["discovered_unread"] == 5000 - big["nodes_read"]
        assert any("bytes" in note and "text_contains" in note for note in big["notes"])
    small = rendered(400, "tree")
    assert small["nodes_read"] == 400 and small["truncated_by"] == [] and small["total_known"] is True
