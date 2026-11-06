"""unix_computer_use speed-first slice: richer tree rows, and resolve + read / resolve + ONE action (+ a read) in ONE helper.

The helper's logic runs for real against the ``FakeAX`` surface of the element-action tests (only the ctypes boundary is
faked; the CFURL conversion runs against CoreFoundation itself on macOS). The plugin tests fake the process boundary as
the targeting tests do. What is pinned: tree rows keep every legacy key and add ``parent_id``, separate title and
description, ``url`` and sparse ``enabled``/``focused`` with honest cuts; ``observe`` and ``element`` are one process;
the action receipt is printed before the read after it, so a read that fails, runs out of the shared budget or is cut
off by the helper's timeout never erases or repeats the action; a new option the backend cannot honour is refused.
"""
from __future__ import annotations

import ctypes
import json
import subprocess
import sys

import pytest

from skills.unix_computer_use import macos_native as native
from tests.test_unix_computer_use_effects import marked, ssh_mac
from tests.test_unix_computer_use_element_act import LONG_BEFORE, LONG_LABEL, TIMEOUT, FakeAX, deck
from tests.test_unix_computer_use_targeting import KEYNOTE, SAFARI, app_dict, helper_op, inputs, mac, receipt

URL = "https://chat.example.test/archives/C0SYNTH/p1700000000000100?thread_ts=1700000000.000100&cid=C0SYNTH"


def rich() -> dict:
    """``deck`` plus a link row with a long command text, a URL, a description beside its title and a focused field."""
    nodes = deck()
    nodes["win"]["children"].append("msg")
    nodes["msg"] = {"role": "AXGroup", "title": "Synthetic message", "description": LONG_LABEL, "children": ["cmd", "link"]}
    nodes["cmd"] = {"role": "AXStaticText", "value": LONG_BEFORE}
    nodes["link"] = {"role": "AXLink", "title": "permalink", "url": URL, "actions": ["AXPress"], "enabled": True}
    nodes["field"]["focused"] = True
    return nodes


class Clock:
    """A controllable ``now``: time passes only when a faked step spends it."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def tree(ax, **kw) -> dict:
    return native.op_tree(200, kw.pop("max_nodes", 150), 3000, ax=ax, **kw)


def element(ax, spec, *, emit=None, now=None, fronts=None) -> dict:
    reads = iter(fronts or [app_dict(SAFARI)] * 2)
    extra = {"now": now} if now else {}
    return native.op_element(json.dumps({"app": "Keynote", **spec}), ax=ax, front=lambda: next(reads, None),
                             settle=lambda _s: None, emit=emit, **extra)


@pytest.fixture(autouse=True)
def _resolved(monkeypatch):
    """The launch-free resolution answers from a fixed list (never the real lsappinfo)."""
    def resolve(query):
        if query in ("Keynote", "pid:200"):
            return {"ok": True, "query": query, "candidates": [app_dict(KEYNOTE)]}
        return {"ok": False, "query": query, "reason": "not_running", "candidates": [], "running": ["Safari", "Keynote"]}
    monkeypatch.setattr(native, "op_resolve", resolve)


# --- tree rows --------------------------------------------------------------------------------------------------------

def test_rows_carry_hierarchy_separate_texts_urls_and_sparse_states_with_honest_cuts():
    ax = FakeAX(rich(), windows=["win"], focused="win")
    out = tree(ax)
    rows = {row.get("label") or row.get("value") or row["role"]: row for row in out["nodes"]}
    by_id = {row["id"]: row for row in out["nodes"]}
    assert "parent_id" not in by_id[1] and all(by_id[row["parent_id"]]["depth"] == row["depth"] - 1 for row in out["nodes"][1:])
    msg = rows["Synthetic message"]
    cmd = next(row for row in out["nodes"] if row["role"] == "StaticText")
    link = rows["permalink"]
    assert cmd["parent_id"] == link["parent_id"] == msg["id"], "a message's text and link name the message they belong to"
    assert msg["title"] == "Synthetic message" and msg["description"] == LONG_LABEL[:200] and msg["description_len"] == len(LONG_LABEL)
    assert len(LONG_LABEL.encode("utf-16-le")) // 2 > len(LONG_LABEL), "cuts and lengths count code points, never UTF-16 units"
    assert cmd["value"] == LONG_BEFORE[:200] and cmd["value_len"] == len(LONG_BEFORE)
    assert link["url"] == URL and "url_len" not in link and "enabled" not in link, "enabled appears only as false"
    assert rows["Stop"]["enabled"] is False and rows["Presenter notes"]["focused"] is True
    assert sum("focused" in row for row in out["nodes"]) == 1, "focused appears only as true"
    assert out["limits"]["text_limit"] == 200 and out["window_id"] == 39 and ax.balanced()


def test_text_limit_reads_whole_texts_or_cuts_them_and_the_legacy_label_keeps_its_cut():
    nodes = rich()
    nodes["msg"]["title"] = LONG_LABEL
    whole = tree(FakeAX(nodes, windows=["win"], focused="win"), text_limit=100_000)
    msg = next(row for row in whole["nodes"] if row.get("title") == LONG_LABEL)
    cmd = next(row for row in whole["nodes"] if row["role"] == "StaticText")
    assert msg["description"] == LONG_LABEL and cmd["value"] == LONG_BEFORE and not any(k.endswith("_len") for k in {**msg, **cmd})
    assert msg["label"] == LONG_LABEL[:200], "label is the legacy 200-character field; title holds the whole text"
    short = tree(FakeAX(nodes, windows=["win"], focused="win"), text_limit=12)
    link = next(row for row in short["nodes"] if row["role"] == "Link")
    assert link["url"] == URL[:12] and link["url_len"] == len(URL) and link["label"] == "permalink"[:12]
    clamped = tree(FakeAX(nodes, windows=["win"], focused="win"), text_limit=10**9)
    assert clamped["limits"]["text_limit"] == native.TEXT_LIMIT["ceiling"]


@pytest.mark.skipif(sys.platform != "darwin", reason="CoreFoundation is macOS only")
def test_an_axurl_value_converts_to_its_absolute_string():
    ax = native._AX()
    create = ax._sig(ax.CF, "CFURLCreateWithBytes", ctypes.c_void_p,
                     [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32, ctypes.c_void_p])
    base_bytes, rel_bytes = b"https://chat.example.test/archives/", b"C0SYNTH/p17?thread_ts=1.2#frag"
    base = create(None, base_bytes, len(base_bytes), native._UTF8, None)
    rel = create(None, rel_bytes, len(rel_bytes), native._UTF8, base)
    try:
        assert ax.CFGetTypeID(rel) == ax.T_URL
        assert ax.to_py(rel) == "https://chat.example.test/archives/C0SYNTH/p17?thread_ts=1.2#frag"
        assert ax.to_py(base) == base_bytes.decode()
    finally:
        ax.CFRelease(rel)
        ax.CFRelease(base)


def test_a_named_window_roots_the_tree_exactly_or_is_refused():
    nodes = rich()
    nodes["win2"] = {"role": "AXWindow", "title": "Thread", "window_id": 40, "children": ["ok2"]}
    nodes["ok2"] = {"role": "AXButton", "title": "OK"}
    ax = FakeAX(nodes, windows=["win", "win2"], focused="win")
    for window in (40, "40", "Thread"):
        out = tree(ax, window=window)
        assert out["ok"] is True and out["root"] == "AXWindows" and out["window"] == "Thread" and out["window_id"] == 40
        assert [row["label"] for row in out["nodes"]] == ["Thread", "OK"]
    nodes["win2"]["title"] = "Deck.key"
    same = tree(ax, window="Deck.key")
    assert same["ok"] is False and same["reason"] == "window_ambiguous" and [w["window_id"] for w in same["windows"]] == [39, 40]
    assert tree(ax, window=77)["reason"] == "window_not_found"
    assert tree(ax, window=40, root="focused")["reason"] == "bad_option", "a named window and the focused element are two roots"
    assert ax.balanced()


# --- observe: resolve and read in one process --------------------------------------------------------------------------

def test_observe_resolves_and_reads_in_one_process_and_refuses_before_reading(monkeypatch):
    ax = FakeAX(rich(), windows=["win"], focused="win")
    monkeypatch.setattr(native, "_AX", lambda: ax)
    out = native.op_observe(json.dumps({"app": "Keynote", "max_nodes": 5, "text_limit": 50}))
    assert out["ok"] is True and out["target"] == app_dict(KEYNOTE) and out["pid"] == 200 and out["visited"] == 5
    assert out["limits"]["text_limit"] == 50
    front = native.op_observe("{}", front=lambda: app_dict(SAFARI))
    assert front["target"] == app_dict(SAFARI) and front["pid"] == 100
    monkeypatch.setattr(native, "op_tree", lambda *a, **k: pytest.fail("an unresolved application is never read"))
    missing = native.op_observe(json.dumps({"app": "Nope"}))
    assert missing["ok"] is False and missing["stage"] == "resolve" and missing["reason"] == "not_running"
    unknown = native.op_observe("{}", front=lambda: None)
    assert unknown["stage"] == "resolve" and unknown["reason"] == "front_unknown"
    for bad, fragment in (({"app": "Keynote", "pid": 1}, "unknown option"), ({"max_nodes": "9"}, "must be int"),
                          ({"include_web": 1}, "must be bool"), ({"window": True}, "window_id or an exact title"), ("[]", "JSON object")):
        refused = native.op_observe(bad if isinstance(bad, str) else json.dumps(bad))
        assert refused["reason"] == "bad_option" and fragment in refused["error"], (bad, refused)
    assert ax.balanced()


def test_observe_names_the_resolved_target_also_when_the_read_raises(monkeypatch, capsys):
    """Resolution is known exactly when the answer names a ``target``: a read that raises after the application was
    resolved keeps it (``screenshot(app=)`` says ``selected_by: fallback`` only for such an app), while an answer
    without one leaves the resolution unknown (the screenshot refuses)."""
    monkeypatch.setattr(native, "op_tree", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("provider went away")))
    assert native.main(["observe", json.dumps({"app": "Keynote", "max_nodes": 1})]) == 1
    printed = capsys.readouterr().out.strip().splitlines()
    assert len(printed) == 1 and json.loads(printed[0]) == {"ok": False, "error": "RuntimeError: provider went away",
                                                            "op": "observe", "target": app_dict(KEYNOTE)}
    missing = native.op_observe(json.dumps({"app": "Nope"}))
    assert "target" not in missing and missing["stage"] == "resolve", "nothing was read for an unresolved app"


# --- element: resolve and ONE action, optionally followed by a read of the same window ---------------------------------

def test_element_without_observe_is_exactly_the_act_receipt_plus_its_target():
    emitted = []
    one = element(FakeAX(deck(), windows=["win"], focused="win"), {"do": "press", "select": {"identifier": "toolbar.play"}}, emit=emitted.append)
    reads = iter([app_dict(SAFARI)] * 2)
    legacy = native.op_act(200, "press", {"identifier": "toolbar.play"}, ax=FakeAX(deck(), windows=["win"], focused="win"),
                           front=lambda: next(reads), settle=lambda _s: None)
    assert one.pop("target") == app_dict(KEYNOTE)
    one["elapsed_ms"] = legacy["elapsed_ms"] = 0
    assert one == legacy and "observation" not in one and emitted == [], "no read asked, no checkpoint printed"


def test_observe_prints_the_receipt_first_then_reads_the_same_window_after_one_action():
    nodes = deck()
    ax = FakeAX(nodes, windows=["win"], focused="win")
    emitted = []

    def checkpoint(receipt):
        emitted.append(receipt)
        nodes["field"]["value"] = "changed after the checkpoint"  # only a read made AFTER the checkpoint can see this

    out = element(ax, {"do": "set_value", "select": {"role": "TextField", "text": "typed"}, "observe": True, "text_limit": 9}, emit=checkpoint)
    assert ax.sets == [("field", "typed")] and ax.performed == [], "one action, never repeated"
    assert len(emitted) == 1 and emitted[0]["action_outcome"] == "completed" and emitted[0]["target"] == app_dict(KEYNOTE)
    assert "observation" not in emitted[0] and emitted[0]["element_after"]["value"] == "typed"
    seen = out["observation"]
    assert out["ok"] is True and marked(out) and seen["ok"] is True and seen["limits"]["text_limit"] == 9
    assert seen["window"] == {"from": "action_window", "title": "Deck.key", "window_id": 39} == {**out["window"], "from": "action_window"}
    field = next(row for row in seen["nodes"] if row["role"] == "TextField")
    assert field["value"] == "changed a" and field["value_len"] == len("changed after the checkpoint") and seen["marks"]
    assert ax.balanced()


@pytest.mark.parametrize("failure", ["window_gone", "budget", "raises"])
def test_a_read_after_the_action_that_fails_keeps_the_receipt_and_never_repeats_the_action(failure):
    nodes = deck()
    after = {"done": False}
    clock = Clock()

    def perform(el, action):
        after["done"] = True
        if failure == "window_gone":
            del nodes["win"]  # the press closed the window the read is scoped to
        if failure == "budget":
            clock.t = 99.0  # the action itself outlived timeout_ms
        return 0

    ax = FakeAX(nodes, windows=["win"], focused="win", perform=perform)
    children = ax.children

    def kids(el, limit):
        if failure == "raises" and after["done"]:
            raise RuntimeError("provider went away")
        return children(el, limit)

    ax.children = kids
    out = element(ax, {"do": "press", "select": {"identifier": "toolbar.play"}, "observe": True}, now=clock)
    assert out["ok"] is True and out["dispatched"] is True and out["action_outcome"] == "completed" and marked(out)
    assert ax.performed == [("play", "AXPress")]
    if failure == "budget":
        assert out["element_after"] == {"unread": {"element": "budget_exhausted"}} and "could be read back" in out["note"]
    expected = {"window_gone": "accessibility_read_failed", "budget": "budget_exhausted", "raises": "observation_error"}[failure]
    assert out["observation"]["ok"] is False and out["observation"]["reason"] == expected, out["observation"]
    assert ax.balanced()


@pytest.mark.parametrize("slow", ["resolve", "window", "precondition"])
def test_a_budget_spent_before_the_action_refuses_it_and_nothing_is_done(monkeypatch, slow):
    """``timeout_ms`` counts from the helper call's start: a slow resolution, window pick or precondition read that
    spends it means no action, said as ``budget_exhausted`` with ``dispatched: false`` (no fresh timeout per stage)."""
    clock, ax = Clock(), FakeAX(deck(), windows=["win"], focused="win")
    resolve, attr, names = native.op_resolve, ax.attr, ax.action_names

    def slow_resolve(query):
        clock.t += 3.5 if slow == "resolve" else 0.0
        return resolve(query)

    def slow_attr(el, name):
        clock.t += 3.5 if slow == "window" and el == "app" else 0.0
        return attr(el, name)

    def slow_names(el):
        clock.t += 3.5 if slow == "precondition" else 0.0
        return names(el)

    monkeypatch.setattr(native, "op_resolve", slow_resolve)
    ax.attr, ax.action_names = slow_attr, slow_names
    out = element(ax, {"do": "press", "select": {"identifier": "toolbar.play"}, "timeout_ms": 3000, "observe": True}, now=clock)
    assert out["ok"] is False and out["reason"] == "budget_exhausted" and out["dispatched"] is False and not marked(out)
    assert ax.performed == [] and out["note"] == "no action was made" and out["observation"]["reason"] == "budget_exhausted"
    if slow == "precondition":
        assert out["matched"] == 1 and out["element"]["identifier"] == "toolbar.play", "found and checked, then refused for time"
    assert ax.balanced()


@pytest.mark.parametrize("acted_at, oversleep, settled, read_back, observed", [
    (2.0, 0.0, 0.15, True, 350),     # time left after the settle: the readback and the read run on what remains
    (2.95, 0.0, 0.05, False, None),  # the settle uses the rest: nothing left, so the readback is skipped
    (2.8, 0.5, 0.15, False, None),   # the settle oversleeps past the deadline: the readback is skipped
    (3.5, 0.0, None, False, None),   # the action itself outlived the budget: no settle, no readback
])
def test_the_settle_readback_and_read_after_the_action_spend_only_what_is_left(acted_at, oversleep, settled, read_back, observed):
    clock = Clock()
    acted, settles = {"done": False}, []

    def perform(el, action):
        clock.t, acted["done"] = acted_at, True
        return 0

    def settle(seconds):
        settles.append(seconds)
        clock.t += seconds + oversleep

    ax = FakeAX(deck(), windows=["win"], focused="win", perform=perform)
    attr = ax.attr

    def facts_read(el, name):
        clock.t += 0.5 if acted["done"] and el == "win" else 0.0  # the window's facts after the action spend the same budget
        return attr(el, name)

    ax.attr = facts_read
    spec = {"app": "Keynote", "do": "press", "select": {"identifier": "toolbar.play"}, "observe": True, "timeout_ms": 3000}
    out = native.op_element(json.dumps(spec), ax=ax, front=lambda: None, settle=settle, now=clock)
    assert out["ok"] is True and out["action_outcome"] == "completed" and marked(out) and ax.performed == [("play", "AXPress")]
    assert settles == ([] if settled is None else [pytest.approx(settled)]), "the settle shrinks to what is left"
    assert ("identifier" in out["element_after"]) is read_back
    if not read_back:
        assert out["element_after"] == {"unread": {"element": "budget_exhausted"}} and "could be read back" in out["note"]
    seen = out["observation"]
    if observed is None:
        assert seen["reason"] == "budget_exhausted"
    else:
        assert seen["ok"] is True and abs(seen["limits"]["timeout_ms"] - observed) <= 1, "what was left, never a fresh timeout_ms"
    assert ax.balanced()


def test_an_unknown_outcome_is_read_but_never_repeated_and_refusals_act_on_nothing():
    ax = FakeAX(deck(), windows=["win"], focused="win", perform=lambda el, action: TIMEOUT)
    unknown = element(ax, {"do": "press", "select": {"identifier": "toolbar.play"}, "observe": True})
    assert unknown["action_outcome"] == "unknown" and "do not retry blindly" in unknown["note"] and not marked(unknown)
    assert ax.performed == [("play", "AXPress")] and unknown["observation"]["ok"] is True
    calm = FakeAX(deck(), windows=["win"], focused="win")
    ambiguous = element(calm, {"do": "press", "select": {"role": "Button", "label": "Play"}, "observe": True})
    assert ambiguous["reason"] == "element_ambiguous" and ambiguous["dispatched"] is False and ambiguous["observation"]["ok"] is True
    for select, reason in (({"label": "OK", "window": 77}, "window_not_found"), ({}, "bad_selector")):
        refused = element(calm, {"do": "press", "select": select, "observe": True})
        assert refused["reason"] == reason and refused["dispatched"] is False and refused["observation"]["reason"] == "no_window"
    assert calm.performed == [] and ax.balanced() and calm.balanced()


def test_element_refuses_an_unresolved_app_and_options_it_cannot_honour_before_acting(monkeypatch):
    monkeypatch.setattr(native, "op_act", lambda *a, **k: pytest.fail("nothing is acted on"))
    missing = element(None, {"app": "Nope", "do": "press", "select": {"role": "Button"}, "observe": True})
    assert missing["stage"] == "resolve" and missing["reason"] == "not_running" and missing["dispatched"] is False
    assert element(None, {"app": "pid:999", "do": "press", "select": {"role": "Button"}})["stage"] == "resolve"
    for spec, fragment in (({"do": "press", "select": {"role": "Button"}, "text_limit": 900}, "observe=true"),
                           ({"app": " ", "do": "press", "select": {}}, "app is required"),
                           ({"do": "press", "select": {}, "observe": "yes"}, "observe must be bool")):
        refused = element(None, spec)
        assert refused["reason"] == "bad_option" and refused["dispatched"] is False and fragment in refused["error"], refused


def test_the_helper_prints_the_checkpoint_line_before_the_final_line_and_keeps_it_through_a_crash(monkeypatch, capsys):
    ax = FakeAX(deck(), windows=["win"], focused="win")
    real = native.op_element
    monkeypatch.setattr(native, "op_element", lambda raw, emit=None: real(raw, ax=ax, front=lambda: None, settle=lambda _s: None, emit=emit))
    spec = json.dumps({"app": "Keynote", "do": "press", "select": {"identifier": "toolbar.play"}, "observe": True})
    assert native.main(["element", spec]) == 0
    first, last = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert first["checkpoint"] == "action" and "observation" not in first and first["action_outcome"] == "completed"
    assert "checkpoint" not in last and last["observation"]["ok"] is True and last["element"] == first["element"]
    monkeypatch.setattr(native, "observe_window", lambda *a, **k: (_ for _ in ()).throw(MemoryError("read too big")))
    assert native.main(["element", spec]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    crashed = json.loads(lines[-1])
    assert all(line.isascii() for line in lines) and len(lines) == 2
    assert crashed["action_outcome"] == "completed" and crashed["dispatched"] is True and "checkpoint" not in crashed
    assert crashed["observation"] == {"ok": False, "reason": "observation_error", "error": "MemoryError: read too big"}
    assert ax.performed == [("play", "AXPress")] * 2, "each call acted once"


# --- the plugin: one helper call, the checkpoint kept through a timeout, new options refused, never ignored -----------

RECEIPT = {"ok": True, "pid": 200, "do": "press", "dispatched": True, "action_outcome": "completed", "observed_effect": True,
           "element": {"role": "Button", "label": "Play"}, "front_before": app_dict(SAFARI), "front_after": app_dict(SAFARI),
           "target_in_front": False, "note": "the element performed AXPress", "target": app_dict(KEYNOTE)}


def test_element_act_observe_is_one_helper_call_carrying_the_shared_budget(tmp_path, monkeypatch):
    seen = {"ok": True, "nodes": [{"id": 1, "depth": 0, "role": "Window"}], "window": {"from": "action_window", "window_id": 39}}
    module, api, state = mac(tmp_path, monkeypatch, helper=lambda args, st: {**RECEIPT, "observation": seen})
    out = receipt(api.tools["element_act"](app="Keynote", do="press", select={"label": "Play"}, observe=True, text_limit=5000, timeout_ms=4000))
    assert [args[0] for args in state["helper"]] == ["element"] and inputs(state) == []
    spec = json.loads(state["helper"][0][1])
    assert (spec["observe"], spec["text_limit"], spec["timeout_ms"]) == (True, 5000, 4000)
    assert out["observation"] == seen and out["app"] == app_dict(KEYNOTE) and "target" not in out and marked(out)
    meta = {name: api.meta[name]["schema"]["properties"] for name in ("ax_tree", "element_act")}
    assert meta["element_act"]["observe"] == {"type": "boolean", "default": False} and meta["element_act"]["text_limit"]["default"] == 200
    assert meta["ax_tree"]["text_limit"]["default"] == 200 and meta["ax_tree"]["window"]["type"] == ["integer", "string"]


@pytest.mark.parametrize(("dispatched", "tail"), [
    (True, '{"ok": true, "observation": {"ok": tr'),
    (True, "Traceback (most recent call last):"),
    (False, ""),
])
def test_a_helper_stopped_after_its_receipt_keeps_the_receipt_and_says_the_read_did_not_finish(tmp_path, monkeypatch, tail, dispatched):
    module, api, state = mac(tmp_path, monkeypatch)
    saved = RECEIPT if dispatched else {"ok": False, "dispatched": False, "reason": "element_ambiguous",
                                        "note": "no action was made", "target": app_dict(KEYNOTE)}
    printed = json.dumps({**saved, "checkpoint": "action"}) + "\n" + tail

    def stopped(cmd, *a, **k):
        if any(str(p).endswith("macos_native.py") for p in cmd):
            raise subprocess.TimeoutExpired(cmd, k.get("timeout"), output=printed.encode())
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", stopped)
    out = receipt(api.tools["element_act"](app="Keynote", do="press", select={"label": "Play"}, observe=True))
    assert out["dispatched"] is dispatched and marked(out) is dispatched
    assert all(out[key] == value for key, value in saved.items() if key != "target"), "the saved action or refusal stays intact"
    error = out["observation"]["error"]
    assert out["observation"]["reason"] == "helper_stopped" and "read the window with ax_tree before acting again" in error
    assert "after the action" not in error, "an observation failure must not imply that a refused action happened"
    assert "checkpoint" not in out and out["app"] == app_dict(KEYNOTE)

    def silent(cmd, *a, **k):
        raise subprocess.TimeoutExpired(cmd, k.get("timeout"), output=tail.encode() or None)

    monkeypatch.setattr(module.subprocess, "run", silent)
    lost = receipt(api.tools["element_act"](app="Keynote", do="press", select={"label": "Play"}, observe=True))
    assert lost["dispatched"] is None and lost["action_outcome"] == "unknown" and not marked(lost), "no receipt: unknown, never not-sent"


def test_new_options_are_refused_where_they_cannot_be_honoured(tmp_path, monkeypatch):
    module, api, state = mac(tmp_path, monkeypatch)
    lone = receipt(api.tools["element_act"](app="Keynote", do="press", select={"label": "Play"}, text_limit=900))
    assert lone["reason"] == "bad_option" and lone["dispatched"] is False and state["helper"] == []
    tools, transport = ssh_mac(tmp_path / "ssh", monkeypatch)
    for kwargs in ({"text_limit": 5000}, {"window": 39}):
        refused = receipt(tools["ax_tree"](**kwargs))
        assert refused["ok"] is False and refused["reason"] == "unsupported_option" and refused["backend"] == "ssh_macos"
    assert receipt(tools["ax_tree"]())["reason"] != "unsupported_option", "without new options the remote answer is unchanged"
    remote = receipt(tools["element_act"](app="Keynote", do="press", select={"label": "Play"}, observe=True))
    assert remote["reason"] == "unsupported" and remote["dispatched"] is False
    assert not [argv for argv in transport.calls if argv[:1] == ["ssh"] and "observe" in " ".join(argv)]


def test_ax_tree_passes_the_window_and_renders_a_window_refusal_with_the_rows(tmp_path, monkeypatch):
    rows = [{"from": "AXWindows", "title": "Deck.key", "window_id": 39}, {"from": "AXWindows", "title": "Deck.key", "window_id": 40}]

    def helper(args, st):
        spec = json.loads(args[1])
        if spec["window"] == "Deck.key":
            return {"ok": False, "pid": 200, "reason": "window_ambiguous", "windows": rows, "error": "2 windows", "target": app_dict(KEYNOTE)}
        return {"ok": True, "pid": 200, "window": "Deck.key", "window_id": 39, "nodes": [], "marks": [], "target": app_dict(KEYNOTE)}

    module, api, state = mac(tmp_path, monkeypatch, helper=helper)
    refused = receipt(api.tools["ax_tree"](app="Keynote", window="Deck.key"))
    assert refused["reason"] == "window_ambiguous" and refused["windows"] == rows and refused["app"] == app_dict(KEYNOTE)
    read = receipt(api.tools["ax_tree"](app="Keynote", window=39, text_limit=0))
    assert read["ok"] is True and read["window_id"] == 39 and json.loads(state["helper"][-1][1])["text_limit"] == 200
    assert [args[0] for args in state["helper"]] == ["observe", "observe"] and state["front"] == SAFARI


@pytest.mark.parametrize("op, options, fragment", [
    ("observe", {"bogus": 1}, "unknown option"),
    ("element", {"app": "Keynote", "do": "press", "select": {"role": "Button"}, "text_limit": 900}, "observe=true"),
])
def test_the_real_isolated_helper_process_answers_the_new_ops_with_one_line(op, options, fragment):
    """``python -I -B macos_native.py <op> <json>`` as the plugin starts it: refused options answer before any
    application is resolved or touched, in one ASCII JSON line, and the helper writes no bytecode."""
    script = native.__file__
    done = subprocess.run([sys.executable, "-I", "-B", script, op, json.dumps(options)], capture_output=True, text=True, timeout=60)
    lines = done.stdout.strip().splitlines()
    assert done.returncode == 1 and len(lines) == 1 and lines[0].isascii(), (done.stdout, done.stderr)
    answer = json.loads(lines[0])
    assert answer["ok"] is False and answer["reason"] == "bad_option" and fragment in answer["error"] and "checkpoint" not in answer


# --- screenshot selection after the single helper answers or fails ---

@pytest.mark.parametrize("unread, why", [
    ({"ok": False, "reason": "accessibility_untrusted"}, "accessibility_untrusted"),
    # The read raised after the application was resolved: the helper's answer still names the target.
    ({"ok": False, "error": "RuntimeError: provider went away", "op": "observe"}, "provider went away"),
])
def test_screenshot_with_an_unlocatable_window_says_it_captured_the_main_display(tmp_path, monkeypatch, unread, why):
    def helper(args, state):
        if args[0] == "tree":
            return {**unread, "pid": int(args[1])}
        if args[0] == "resolve":
            return {"ok": True, "candidates": [app_dict(KEYNOTE)]}
        if args[0] == "displays":
            return {"ok": True, "displays": [{"id": 1, "main": True, "x": 0, "y": 0, "width": 1728, "height": 1117, "index": 1}]}
        return {"ok": False, "error": "no fake"}
    module, api, state = mac(tmp_path, monkeypatch, helper=helper)
    monkeypatch.setattr(module, "_png_dimensions", lambda _p: (3456, 2234))
    out = receipt(api.tools["screenshot"](job_id="fallback", app="Keynote"))
    assert out["ok"] is True and "app" not in out, "the main display is not presented as the named app's"
    assert out["display"]["selected_by"] == "fallback" and why in out["display"]["app_window_unread"]
    assert ["screencapture", "-x", out["full_resolution_path"]] in state["calls"], "the resolved app's fallback is the main display"


@pytest.mark.parametrize("failure, says", [
    ("no_output", "native helper gave no answer"),
    ("timeout", "TimeoutExpired"),
    ("not_launched", "PermissionError"),
])
def test_screenshot_of_an_app_the_helper_did_not_resolve_captures_nothing(tmp_path, monkeypatch, failure, says):
    """The main-display fallback is for a RESOLVED application whose window was not located. With the display list read,
    when the one helper call does not establish the application (no output, cut off at its timeout, never started, an
    answer naming no target, or a refusal at resolution), no display was chosen for it: a refusal, and nothing is
    captured. An unreadable display list is the inherited capture-only path, pinned by the next test."""
    module, api, state = mac(tmp_path, monkeypatch)
    fake = module.subprocess.run

    def run(cmd, *a, **k):
        argv = [str(part) for part in cmd]
        if not any(part.endswith("macos_native.py") for part in argv) or helper_op(argv)[0] != "observe":
            return fake(cmd, *a, **k)
        state["calls"].append(argv)
        state["helper"].append(helper_op(argv))
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, k.get("timeout"), output=b"")
        if failure == "not_launched":
            raise PermissionError(13, "Permission denied", argv[0])
        printed = {"no_output": "",
                   "answer_without_target": json.dumps({"ok": False, "error": "MemoryError: lost", "op": "observe"}),
                   "not_running": json.dumps({"ok": False, "stage": "resolve", "reason": "not_running", "candidates": [],
                                              "running": ["Safari"]})}[failure]
        return subprocess.CompletedProcess(argv, 1, printed + "\n" if printed else "",
                                           "Traceback (most recent call last):\nMemoryError\n" if failure == "no_output" else "")

    monkeypatch.setattr(module.subprocess, "run", run)
    out = receipt(api.tools["screenshot"](job_id="unresolved", app="Keynote"))
    assert out["ok"] is False and says in out["error"] and set(out) == {"ok", "error"}, out
    assert not [argv for argv in state["calls"] if argv[0] == "screencapture"], "no display was captured for an unresolved app"
    assert [args[0] for args in state["helper"]] == ["displays", "observe"]
    assert receipt(api.tools["screenshot"](job_id="plain"))["ok"] is True, "a screenshot without app= is unaffected"


def test_screenshot_with_unreadable_displays_has_no_mapped_input(tmp_path, monkeypatch):
    module, api, state = mac(tmp_path, monkeypatch, helper=lambda args, st: {"ok": False, "error": "locked"})
    monkeypatch.setattr(module, "_png_dimensions", lambda _p: (800, 600))
    monkeypatch.setattr(module, "_macos_logical_size", lambda: (0, 0))
    out = receipt(api.tools["screenshot"](job_id="fallback"))
    assert out["ok"] is True and "display" not in out and out["mapped_input_available"] is False
    assert "observation_id" not in out and "coord_transform" not in out
    assert out["input_geometry_evidence"] == "unavailable" and out["input_width"] is None
    assert "locked" in out["coordinate_note"] and "Capture only" in out["coordinate_note"]
    assert receipt(api.tools["click"](x=10, y=10, observation_id=""))["dispatched"] is False
    assert inputs(state) == []
    # A clean-looking Finder ratio still cannot prove which display was captured.
    monkeypatch.setattr(module, "_macos_logical_size", lambda: (400, 300))
    out = receipt(api.tools["screenshot"](job_id="finder"))
    assert out["input_width"] == 400 and out["mapped_input_available"] is False
    assert out["input_geometry_evidence"] == "finder_desktop_union"
    assert "observation_id" not in out and "coord_transform" not in out
    # app= on this inherited path: the list comes first, so the application is not looked up and no selection is
    # established; the default capture stays capture-only and says so.
    state["helper"].clear()
    named = receipt(api.tools["screenshot"](job_id="named", app="Keynote"))
    assert named["ok"] is True and named["display"] == {"selected_by": "default", "app_display_unread": "locked"} and "app" not in named
    assert named["mapped_input_available"] is False and "observation_id" not in named and "coord_transform" not in named
    assert "locked" in named["coordinate_note"] and "Capture only" in named["coordinate_note"]
    assert [args[0] for args in state["helper"]] == ["displays"], "no resolution is attempted without a display list"
    assert [argv for argv in state["calls"] if argv[0] == "screencapture"][-1][1:-1] == ["-x"], "the default capture: no -D"
