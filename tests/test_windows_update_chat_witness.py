"""The Windows qualification's screen witness of its chat messages, judged on synthetic UIA readings.

``devtools/qualification/windows_update/uia_reading.py`` says what one reading of ``Main`` shows; ``scenario.py`` waits
for each message on its own, scrolls once to one that is on the page but off screen, and after an answered, exactly
durable turn without a witness fails that check and goes on, ``unconfirmed`` at best, while a later ``failed`` decides.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "devtools" / "qualification" / "windows_update"))

import fixture_model as fm  # noqa: E402
import scenario  # noqa: E402
from uia_reading import MessageWitness, message_counts  # noqa: E402

TAG = "4242-1"
P1, R1 = fm.prompt_text(1, TAG), fm.reply_text(1, TAG)
M_P1, M_R1 = fm.prompt_token(1, TAG), fm.reply_token(1, TAG)
WANTS = {"prompt": (P1, M_P1), "reply": (R1, M_R1)}
RECT = {"left": 300, "top": 200, "right": 900, "bottom": 222, "width": 600, "height": 22, "center": {"x": 600, "y": 211}}
TURN_1 = {"expected_durable": [("user", P1), ("assistant", R1)], "phase": "turn_1"}


def leaf(name, *, offscreen=False, rect=RECT, **extra):
    """A ``text`` leaf as window_read shapes it: a state key is present only when it was read."""
    node = {"role": "text", "name": name, **extra}
    if offscreen is not None:
        node["state"] = {"offscreen": offscreen}
    if rect is not None:
        node["rect"] = rect
    return node


def bubble(*children, role="user"):
    return {"role": "group", "class": f"chat-bubble {role}", "children": list(children)}


def main_window(*rows):
    """``Main``: the window, WebView2's document (its value is the page's text) and the feed with its rows."""
    feed = {"role": "group", "automation_id": "chat-messages", "children": list(rows)}
    document = {"role": "document", "value": f"{P1} {R1}", "children": [feed]}
    return {"role": "window", "name": "Ouroboros v0", "children": [document]}


def prompt_counts(reading):
    return message_counts(reading, WANTS)["messages"]["prompt"]


def test_a_visible_leaf_with_the_whole_text_witnesses_its_message():
    found = message_counts(main_window(bubble(leaf(P1)), bubble(leaf(R1), role="assistant")), WANTS)["messages"]
    assert [found[key]["exact_visible"] for key in ("prompt", "reply")] == [1, 1]
    assert (found["prompt"]["in_feed"], found["prompt"]["value"]) == (1, 1)


def test_a_split_rendering_witnesses_by_the_run_marker():
    found = prompt_counts(main_window(bubble(leaf("Qualification prompt 1 ("), leaf(M_P1), leaf("): confirm initial desktop setup."))))
    assert (found["exact_visible"], found["marker_visible"]) == (0, 1)


def test_a_visible_marker_outside_the_main_feed_is_not_a_message_witness():
    found = prompt_counts({"role": "window", "children": [{"role": "document", "children": [bubble(leaf(P1))]}]})
    assert found["outside_feed"] == 1
    assert found["exact_visible"] == found["marker_visible"] == found["in_feed"] == 0


@pytest.mark.parametrize(("node", "reason"), [
    (leaf(P1, offscreen=True), "offscreen"),                       # scrolled out of the feed's view
    (leaf(P1, offscreen=None), "unflagged"),                       # IsOffscreen not read: unknown is not visible
    (leaf(P1, rect={**RECT, "width": 0}), "unflagged"),            # no area on screen
    (leaf(P1, rect=None), "unflagged"),                            # no rectangle at all
    ({"role": "edit", "name": "Message", "value": P1}, "input"),   # the composer still holding the words
    ({"role": "combo_box", "children": [leaf(P1)]}, "input"),
    ({"role": "list_item", "name": f"You {P1}", "rect": RECT, "state": {"offscreen": False}}, "container"),
    ({**leaf(P1), "children": [leaf("Copy")]}, "container"),       # a text node with children is no leaf
    (leaf(fm.prompt_text(2, TAG)), None),                          # the other turn's prompt
])
def test_only_a_visible_text_leaf_witnesses(node, reason):
    found = prompt_counts(main_window(bubble(node)))
    assert found["exact_visible"] == found["marker_visible"] == 0
    assert found["value"] == 1  # the page's text alone never witnesses
    if reason:
        assert found[reason] == 1


@pytest.mark.parametrize("node", [leaf(P1, unread=["name"]), leaf(P1, text_truncated=True), leaf(P1, children_unread="max_depth")])
def test_a_leaf_the_reader_failed_or_cut_is_unjudged(node):
    found = message_counts(main_window(bubble(node)), WANTS)
    assert (found["unjudged_text_leaves"], found["messages"]["prompt"]["exact_visible"]) == (1, 0)


def test_each_message_is_witnessed_in_whichever_reading_shows_it():
    """No reading shows both, as in a window too short for the talk: the former matcher wanted prompt and reply whole,
    in order and in one reading, so it never confirmed such a turn."""
    witness = MessageWitness(WANTS)
    witness.observe(main_window(bubble(leaf(P1, offscreen=True)), bubble(leaf(R1), role="assistant")))
    assert witness.seen == {"reply": "exact"} and witness.next_reveal() == ("prompt", M_P1)
    witness.revealed("prompt", {"ok": True, "element": {"name": P1}})
    assert witness.next_reveal() is None  # one scroll per message
    witness.observe(main_window(bubble(leaf(P1)), bubble(leaf(R1, offscreen=True), role="assistant")))
    assert witness.complete() and witness.seen == {"reply": "exact", "prompt": "exact"}
    detail = repr(witness.detail())
    assert not any(text in detail for text in (P1, R1, M_P1, M_R1))  # counts and fixed words only


@pytest.mark.parametrize(("receipt", "word"), [
    ({"ok": True, "select": {"name_contains": M_P1}, "element": {"name": P1}}, "ok"),
    ({"ok": False, "reason": "ambiguous", "candidates": [{"name": P1}]}, "ambiguous"),
    ({"ok": False, "reason": "not_found"}, "not_found"),
    ({"ok": False, "action_outcome": "unknown", "error": P1}, "unknown"),
    ({"ok": False, "error": f"no ScrollItemPattern on {P1}"}, "refused"),
    (None, "invalid"),
    ("window_ownership_lost_before_act", "window_ownership_lost_before_act"),  # a harness gap code
])
def test_a_scroll_receipt_leaves_one_fixed_word(receipt, word):
    witness = MessageWitness(WANTS)
    witness.revealed("prompt", receipt)
    assert witness.reveal == {"prompt": word}


class Driver:
    """The owned window's UIA stand-in: readings in turn (the last one stays) and its scroll receipts."""

    main_hwnd = 0x2A
    root_pid = 42

    def __init__(self, readings, scroll):
        self.readings, self.scroll, self.scrolls = list(readings), scroll, []

    def ensure_alive(self, phase):
        return None

    def set_exact_text(self, hwnd, **kwargs):
        return None

    def wait_unique_node(self, hwnd, **kwargs):
        return {}

    def act_or_click(self, hwnd, **kwargs):
        return None

    def read_tree(self, hwnd, *, phase, allow_truncated=False):
        if not self.readings:
            raise scenario.Unconfirmed(phase, "uia_window_read_failed")
        row = self.readings.pop(0) if len(self.readings) > 1 else self.readings[0]
        if isinstance(row, scenario.Unconfirmed):
            raise row
        return {"root": row}

    def _element_act(self, hwnd, *, phase, **kwargs):
        self.scrolls.append(kwargs)
        return self.scroll


class Model:
    def cursor(self):
        return 0

    def events_since(self, cursor):
        return [{"kind": "completion", "slot": 1, "status": 200, "reason": "ok"}]


class Host:
    @staticmethod
    def window_owned_by(hwnd, pid):
        return hwnd == Driver.main_hwnd and pid == Driver.root_pid

    @staticmethod
    def foreground_window():
        return Driver.main_hwnd


class Run(scenario.QualificationRun):
    """A run whose window, model and durable history are synthetic, with short waits."""

    TURN_SEC = 3.0

    def __init__(self, tmp_path, readings, scroll=None):
        args = argparse.Namespace(candidate="a" * 40, tag=TAG, evidence=str(tmp_path / "evidence"),
                                  mirror_url=str(tmp_path / "qa" / "mirror.git"))
        super().__init__(args, host_mod=Host, app_root_override=tmp_path / "home" / "Ouroboros")
        self.driver, self.model = Driver(readings, scroll or {"ok": True}), Model()

    def durable_history(self):
        return True, True, [("user", P1), ("assistant", R1)]


def test_a_turn_scrolls_to_one_message_at_a_time_and_confirms_each(tmp_path):
    both_off = main_window(bubble(leaf(P1, offscreen=True)), bubble(leaf(R1, offscreen=True), role="assistant"))
    prompt_on = main_window(bubble(leaf(P1)), bubble(leaf(R1, offscreen=True), role="assistant"))
    reply_on = main_window(bubble(leaf(P1, offscreen=True)), bubble(leaf(R1), role="assistant"))
    run = Run(tmp_path, [both_off, prompt_on, reply_on])
    run.send_chat_turn(1, **TURN_1)
    assert [(s["select"], s["do"]) for s in run.driver.scrolls] == [
        ({"role": "text", "name_contains": M_P1}, "scroll_into_view"), ({"role": "text", "name_contains": M_R1}, "scroll_into_view")]
    check = run.result["checks"]["turn_1_visible_leaf_reply_confirmed"]
    assert check["ok"] is True and [check["detail"][key]["reveal"] for key in ("prompt", "reply")] == ["ok", "ok"]
    assert not run.result["reasons"]


def test_a_missing_witness_after_a_durable_answer_goes_on_unconfirmed(tmp_path):
    run = Run(tmp_path, [main_window(bubble(leaf(P1)))])  # the reply never reaches the screen
    run.TURN_SEC = 0.0
    run.send_chat_turn(1, **TURN_1)  # returns: the Update path is next
    check = run.result["checks"]["turn_1_visible_leaf_reply_confirmed"]
    assert check["ok"] is False and check["detail"]["reply"]["seen"] == "" and check["detail"]["reply"]["value"] == 1
    assert (run.result["verdict"], run.result["phase"], run.result["code"]) == ("unconfirmed", "turn_1", "chat_visible_leaf_text_unconfirmed")
    assert run.result["reasons"][0]["fields"]["continued"] is True
    run.settle("unconfirmed", "update_check", "plain_check_of_b_unobserved", {})
    assert run.result["code"] == "chat_visible_leaf_text_unconfirmed"  # a later gap does not replace it
    run.settle("failed", "update_apply", "update_outcome_not_applied", {})
    run.settle("failed", "turn_2", "not_on_b_after_update", {})
    assert (run.result["verdict"], run.result["code"]) == ("failed", "update_outcome_not_applied")
    assert [reason["code"] for reason in run.result["reasons"]] == [
        "chat_visible_leaf_text_unconfirmed", "plain_check_of_b_unobserved", "update_outcome_not_applied", "not_on_b_after_update"]


def test_without_one_reading_of_the_window_the_turn_stops(tmp_path):
    run = Run(tmp_path, [])  # every reading refused: the window is not this run's
    run.TURN_SEC = 0.0
    with pytest.raises(scenario.Unconfirmed) as stop:
        run.send_chat_turn(1, **TURN_1)
    assert (stop.value.code, stop.value.fields["uia_gap"]) == ("chat_visible_leaf_text_unconfirmed", "uia_window_read_failed")
    assert not run.result["reasons"] and not run.driver.scrolls


def test_owned_read_then_identity_loss_does_not_continue(tmp_path, monkeypatch):
    run = Run(tmp_path, [main_window(bubble(leaf(P1))),
                        scenario.Unconfirmed("turn_1", "uia_window_identity_mismatch")])
    monkeypatch.setattr(scenario, "poll_until", lambda _, predicate, **kw: predicate() or predicate())
    with pytest.raises(scenario.Unconfirmed, match="uia_window_identity_mismatch"):
        run.send_chat_turn(1, **TURN_1)
    assert not run.driver.scrolls and not run.result["reasons"]


def test_old_durable_hit_cannot_authorize_continuation(tmp_path):
    run = Run(tmp_path, [main_window(bubble(leaf(P1)))])
    run.TURN_SEC = 0.0
    histories = iter([(True, True, [("user", P1), ("assistant", R1)]), (True, True, [])])
    run.durable_history = lambda: next(histories)
    with pytest.raises(scenario.Failure, match="exchange_missing_from_durable_history"):
        run.send_chat_turn(1, **TURN_1)
    assert not run.result["reasons"]
