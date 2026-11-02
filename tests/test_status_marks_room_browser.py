"""Status marks in a room (docs/DESIGN.md "Room work status") in the real SPA, in Chromium and WebKit, over the
controlled transport of ``tests/test_room_work_browser.py``.

A room's status control draws one shared mark after its word (web/modules/status_mark.js, drawn by the recipe at the
end of web/ui.css): the room's shape in the word's ink (three moving dots, the same dots still, the pause's two bars,
a ring), the owner's amber dot in that place while something waits for the owner, the green dot of ``Online`` at
rest, a ring while the page reconnects and nothing while the host starts. A Shade row's known outcome is a dot in the
outcome's hue beside its word. The transcript's typing bubble is the same moving mark, shown only while the room's
work moves; reduced motion stops every bounce. Colours are compared with the tokens they resolve to, never with RGB
literals.
"""
from __future__ import annotations

import json
import time

import pytest

from tests import test_room_work_browser as rw

room_ui = rw.room_ui
subscription_ui = rw.subscription_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

# One mark as painted: its shape and hue, the size of each part it shows, their motion, its colour beside its word's,
# and the colours the tokens it may take resolve to on this page.
MARK = """(selector) => {
    const mark = document.querySelector(selector);
    const place = mark?.closest('.rw-trigger, .rw-row, .typing-bubble');
    const word = place?.querySelector('.rw-word, .status-badge') || place;
    const color = (el) => getComputedStyle(el).color;
    const resolve = (token) => { const probe = document.body.appendChild(document.createElement('i'));
        probe.style.color = `var(${token})`; const value = color(probe); probe.remove(); return value; };
    const parts = [...(mark?.children || [])].filter((part) => getComputedStyle(part).display !== 'none');
    return {
        shown: Boolean(mark) && !mark.hidden && getComputedStyle(mark).display !== 'none',
        shape: mark?.dataset.shape || '', hue: mark?.dataset.hue || '', count: place?.querySelectorAll('.status-mark').length || 0,
        // The laid-out size: a moving dot's wave scales it for a moment, which is not its size.
        sizes: parts.map((part) => `${part.offsetWidth}x${part.offsetHeight}`),
        width: mark ? mark.getBoundingClientRect().width : 0,
        motion: parts.map((part) => getComputedStyle(part).animationName),
        color: mark ? color(mark) : '', ink: word ? color(word) : '', word: (word?.textContent || '').trim(),
        tokens: Object.fromEntries(['--mark-owner', '--green', '--red', '--text-meta', '--text-secondary', '--accent-light',
            '--status-ok-fg', '--status-warn-fg', '--status-error-fg', '--status-neutral-fg'].map((token) => [token, resolve(token)])),
    };
}"""

BAR = {"main": "#page-chat .rw-trigger .status-mark", "project": "#project-panel .project-panel-bar .rw-trigger .status-mark"}
DOTS, BARS, ONE = ["4x4"] * 3, ["2x8"] * 2, ["7x7"]


class Transport(rw.Transport):
    """The room transport with a host that may not be ready yet."""

    ready = True

    def _state(self, route):
        self.state_reads += 1
        route.fulfill(json=self.body(ready=self.ready))


def mark(page, selector):
    return page.evaluate(MARK, selector)


def wait_mark(page, selector, predicate, what, timeout=20):
    deadline, current = time.monotonic() + timeout, None
    while time.monotonic() < deadline:
        current = mark(page, selector)
        if predicate(current):
            rw.settle_frames(page)
            return mark(page, selector)
        page.wait_for_timeout(100)
    raise AssertionError(f"timed out waiting for {what}: {json.dumps(current, ensure_ascii=False, indent=1)}")


def bar_state(page, transport, activities, word, room="main"):
    transport.activities = activities
    transport.refresh()
    return wait_mark(page, BAR[room], lambda m: m["word"] == word, f"the {room} bar saying {word}")


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_the_room_bar_draws_one_mark_for_every_state(room_ui, theme):
    transport = Transport(room_ui["page"], history={1: rw.reading(4), 42: rw.reading(4, 0, "Project")}, projects=[rw.PROJECT])
    page = rw.open_app(room_ui, transport, theme=theme, viewport={"width": 1280, "height": 800})
    states = dict(rw.header_states("main"))
    seen = {}
    for name in ("idle", "live", "queued", "access", "paused", "pausing", "unknown", "answer", "live+answer"):
        seen[name] = bar_state(page, transport, states[name], rw.HEADER_WORDS[name])
        assert seen[name]["count"] == 1 and seen[name]["shown"], (name, seen[name])
    rw.evidence(f"status-marks-bar-{theme}", seen)
    rw.capture(page, f"status-marks-bar-{theme}")
    tokens = seen["idle"]["tokens"]
    # The room's shape in its word's ink: live work moves in the crimson word's ink; a queue and a wait that ends by
    # itself hold the same dots still; a Pause stands on two bars until Resume; unconfirmed state is a faint ring.
    for name, shape, sizes in (("live", "moving", DOTS), ("pausing", "moving", DOTS), ("queued", "still", DOTS),
                               ("access", "still", DOTS), ("paused", "pause", BARS), ("unknown", "ring", ONE)):
        state = seen[name]
        assert (state["shape"], state["hue"], state["sizes"]) == (shape, "ink", sizes), (name, state)
        assert state["color"] == state["ink"], ("a work's mark takes its word's ink", name, state)
    assert seen["live"]["ink"] == tokens["--accent-light"], seen["live"]
    assert seen["paused"]["ink"] == seen["queued"]["ink"] == tokens["--text-meta"], (seen["paused"], seen["queued"])
    assert seen["live"]["motion"] == ["status-mark-bounce"] * 3 and seen["queued"]["motion"] == ["none"] * 3, seen
    # Whatever waits for the owner is the amber dot in the mark's place, beside the owner's own amber word and beside
    # Working alike; Working keeps its crimson word.
    for name in ("answer", "live+answer"):
        state = seen[name]
        assert (state["shape"], state["hue"], state["sizes"], state["color"]) == ("dot", "owner", ONE, tokens["--mark-owner"]), state
    assert seen["answer"]["ink"] == tokens["--status-warn-fg"], seen["answer"]
    assert seen["live+answer"]["ink"] == tokens["--accent-light"] != tokens["--mark-owner"], seen["live+answer"]
    # The bar's mark keeps the moving dots' width in every shape it draws, so the caret beside it never moves when
    # the room starts or stops working.
    widths = {name: seen[name]["width"] for name in ("idle", "live", "queued", "paused", "unknown", "answer")}
    assert set(widths.values()) == {18}, widths
    # At rest the room is a status sentence: Online beside its own green dot.
    assert (seen["idle"]["shape"], seen["idle"]["hue"], seen["idle"]["sizes"]) == ("dot", "ok", ONE), seen["idle"]
    assert (seen["idle"]["color"], seen["idle"]["ink"]) == (tokens["--green"], tokens["--status-ok-fg"]), seen["idle"]

    # The host not ready yet: Starting… draws no mark. (Only a complete ready body proves the room idle first; an
    # unready one keeps what it omitted, unconfirmed.)
    bar_state(page, transport, [], "Online")
    transport.ready = False
    starting = bar_state(page, transport, [], "Starting…")
    assert not starting["shown"] and starting["shape"] == "none", starting
    transport.ready = True
    bar_state(page, transport, states["live"], "Working")
    # The socket closed: Reconnecting… is a ring in its word's ink, whatever the room did.
    count = transport.disconnect()
    offline = wait_mark(page, BAR["main"], lambda m: m["word"] == "Reconnecting…", "the reconnecting bar")
    assert (offline["shape"], offline["hue"], offline["sizes"]) == ("ring", "ink", ONE), offline
    assert offline["color"] == offline["ink"] == tokens["--text-secondary"], offline
    transport.await_reconnect(count)
    transport.refresh()
    wait_mark(page, BAR["main"], lambda m: m["word"] == "Working", "the bar back to work")

    # A Project's bar draws the same mark: its Pause stands on two bars.
    where = {"chat_id": rw.PROJECT["chat_id"], "project_id": rw.PROJECT["id"]}
    transport.activities = [rw.activity("p-paused", "budget_paused", pause_cause="owner", **where)]
    transport.refresh()
    rw.open_project(page, rw.PROJECT)
    paused = wait_mark(page, BAR["project"], lambda m: m["word"] == "Paused", "the Project bar paused")
    assert (paused["shape"], paused["hue"], paused["sizes"], paused["count"]) == ("pause", "ink", BARS, 1), paused
    rw.capture(page, f"status-marks-project-bar-{theme}")
    assert transport.posts == [] and transport.decision_posts() == []


def test_shade_outcomes_are_dots_in_their_own_hue_beside_their_words(room_ui):
    names = {"o-warn": "Check the cited pages", "o-stop": "Rebuild the index", "o-fail": "Translate the appendix"}
    history = {1: [*rw.reading(6), *(rw.progress(task_id, f"{name}.", 30 + index, suggested_name=name)
                                     for index, (task_id, name) in enumerate(names.items()))]}
    transport = Transport(room_ui["page"], history=history, activities=[rw.activity(task_id) for task_id in names])
    page = rw.open_app(room_ui, transport, theme="light", viewport={"width": 1280, "height": 800})
    rw.wait_facts(page, "main", lambda f: f["tone"] == "live" and not f["disabled"], "a live control")
    rw.trigger(page).click()
    live = rw.wait_facts(page, "main", lambda f: len(f["rows"]) == 3, "three live rows")
    assert [item["mark"] for item in live["rows"]] == ["moving ink"] * 3, live
    # The three works finish while the list is open: each row settles with its card's own outcome.
    transport.details.update({
        "o-warn": {"task_id": "o-warn", "status": "completed", "outcome_axes": {"execution": {"status": "degraded"}},
                   "suggested_name": names["o-warn"]},
        "o-stop": {"task_id": "o-stop", "status": "cancelled", "suggested_name": names["o-stop"]},
        "o-fail": {"task_id": "o-fail", "status": "failed", "suggested_name": names["o-fail"]},
    })
    transport.activities = []
    transport.refresh()
    words = {"o-warn": "Done with warnings", "o-stop": "Stopped", "o-fail": "Failed"}
    rw.wait_facts(page, "main", lambda f: all((rw.row(f, key) or {}).get("word") == word for key, word in words.items()),
                  "three settled outcomes", timeout=30)
    rows = {key: mark(page, f"#page-chat .rw-row[data-id='{key}'] .status-mark") for key in words}
    rw.evidence("status-marks-shade-outcomes", rows)
    rw.capture(page, "status-marks-shade-outcomes")
    tokens = rows["o-warn"]["tokens"]
    # Done with warnings succeeded (its word says the warnings); Stopped was someone's decision; Failed failed.
    for key, hue, dot, word in (("o-warn", "ok", "--green", "--status-ok-fg"), ("o-stop", "neutral", "--text-meta", "--status-neutral-fg"),
                                ("o-fail", "failed", "--red", "--status-error-fg")):
        row = rows[key]
        assert (row["word"], row["shape"], row["hue"], row["sizes"], row["count"]) == (words[key], "dot", hue, ONE, 1), (key, row)
        assert (row["color"], row["ink"]) == (tokens[dot], tokens[word]), (key, row)
    assert tokens["--status-warn-fg"] not in {row["ink"] for row in rows.values()}, "no outcome is amber"


def test_typing_bubble_moves_only_while_the_room_works_and_reduced_motion_stops_every_bounce(room_ui):
    transport = Transport(room_ui["page"], history={1: rw.reading(4)}, activities=[rw.activity("t-queued", "queued")])
    page = rw.open_app(room_ui, transport, theme="light", viewport={"width": 1280, "height": 800})
    typing = page.locator("#typing-indicator")
    # A queued room holds still dots in its bar and no typing bubble in its transcript.
    queued = wait_mark(page, BAR["main"], lambda m: m["word"] == "Queued", "the queued bar")
    assert queued["shape"] == "still" and not typing.is_visible(), queued
    page.wait_for_timeout(300)
    assert not typing.is_visible(), "a queued room never bounces"
    # Work moves: the bubble shows the same moving mark, in the bubble's secondary ink.
    transport.activities = [rw.activity("t-live")]
    transport.refresh()
    typing.wait_for(state="visible", timeout=10_000)
    bubble = mark(page, "#typing-indicator .status-mark")
    assert (bubble["shape"], bubble["hue"], bubble["sizes"], bubble["count"]) == ("moving", "ink", DOTS, 1), bubble
    assert bubble["color"] == bubble["tokens"]["--text-secondary"] and bubble["motion"] == ["status-mark-bounce"] * 3, bubble
    assert page.locator("#typing-indicator .typing-dots").count() == 0
    rw.capture(page, "status-marks-typing-moving")
    # The owner's question elsewhere puts the amber dot in the bar; the work still moves, and so does the bubble.
    transport.activities = [rw.activity("t-live"), rw.activity("t-ask", kind="direct_chat", client_message_id="cm-ask",
                                                                 **rw.waited("q-typing"))]
    transport.refresh()
    owner = wait_mark(page, BAR["main"], lambda m: m["hue"] == "owner", "the owner's dot in the bar")
    assert owner["word"] == "Working" and typing.is_visible(), owner
    # Reduced motion: every moving mark keeps its three dots and stops bouncing.
    trigger = rw.trigger(page)
    transport.activities = [rw.activity("t-live")]
    transport.refresh()
    wait_mark(page, BAR["main"], lambda m: m["shape"] == "moving", "the bar moving again")
    trigger.click()
    rw.wait_facts(page, "main", lambda f: [item["mark"] for item in f["rows"]] == ["moving ink"], "the moving row")
    page.emulate_media(reduced_motion="reduce")
    rw.settle_frames(page)
    for selector in (BAR["main"], "#page-chat .rw-row .status-mark", "#typing-indicator .status-mark"):
        still = mark(page, selector)
        assert still["shape"] == "moving" and still["sizes"] == DOTS and still["motion"] == ["none"] * 3, (selector, still)
    page.emulate_media(reduced_motion="no-preference")
    rw.settle_frames(page)
    assert mark(page, BAR["main"])["motion"] == ["status-mark-bounce"] * 3
    assert transport.posts == [] and transport.decision_posts() == []
