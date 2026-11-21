"""A paused helper of ended work, then its closure, in the real SPA over the server's own work projection (#3047)."""
from __future__ import annotations

import pytest

from tests import test_room_work_browser as rw

room_ui = rw.room_ui
subscription_ui = rw.subscription_ui
pytestmark = [pytest.mark.ui_browser]

MARKER = f".nav-project-row[data-project-id='{rw.PROJECT['id']}'] .status-mark"
PAUSE = {"reason": "budget", "status": "paused_exact", "exact_continuation": True, "scope": "global", "root_task_id": "old-root"}
NEXT = {"title": "Next stage", "suggested_name": "Next stage",
        "continuation": {"predecessor_task_id": "old-root", "work_id": "old-root", "work_name": "A research"}}
OLD = {"task_id": "old-root", "status": "completed", "suggested_name": "A research"}


def _helper():
    return rw.activity("helper", "budget_paused", chat_id=42, parent_task_id="old-root", root_task_id="old-root",
                       pause_cause="budget", can_resume=True)


def _next():
    return rw.activity("next-root", chat_id=42, root_task_id="next-root", suggested_name="Next stage")


def test_the_closed_helper_takes_the_amber_with_it_and_the_next_stage_stays(room_ui):
    history = {1: rw.reading(4), 42: [*rw.reading(6, 0, "Project"), rw.progress("next-root", "Reading the sources.", 30)]}
    transport = rw.Transport(room_ui["page"], history=history, projects=[rw.PROJECT], activities=[_helper(), _next()])
    transport.raw = {"helper": {"_budget_pause": PAUSE, "parent_task_id": "old-root", "root_task_id": "old-root"}, "next-root": NEXT}
    transport.results = {"old-root": OLD}
    page = rw.open_app(room_ui, transport, viewport={"width": 1100, "height": 722})
    rw.open_project(page, rw.PROJECT)
    transport.refresh()

    # The control: before the closure the helper of the ended work holds the owner's turn in the room and the sidebar.
    held = rw.wait_facts(page, "project", lambda f: "owner" in f["mark"] or "amber" in f["mark"], "the helper's amber")
    rw.evidence("ended-helper-before", held)
    rw.capture(page, "ended-helper-before")

    transport.activities = [_next()]
    transport.raw = {"next-root": NEXT}
    transport.results = {"old-root": OLD, "helper": {"task_id": "helper", "status": "cancelled", "root_task_id": "old-root",
                                                     "parent_task_id": "old-root", "cancel_origin": {"source": "work_finished"}}}
    transport.refresh()

    after = rw.wait_facts(page, "project", lambda f: f["badge"] == "Working" and "owner" not in f["mark"] and "amber" not in f["mark"],
                          "the next stage working, no amber")
    assert (after["tone"], after["mark"]) == ("live", "moving ink"), after
    page.wait_for_function("(selector) => document.querySelector(selector)?.dataset.shape === 'moving'", arg=MARKER)
    marker = page.locator(MARKER)
    assert (marker.get_attribute("data-shape"), marker.get_attribute("data-hue")) == ("moving", "ink")
    rw.trigger(page, "project").click()
    shade = rw.wait_facts(page, "project", lambda f: len(f["rows"]) == 1, "the next stage alone")
    rw.evidence("ended-helper-after", shade)
    rw.capture(page, "ended-helper-after")
    [row] = shade["rows"]
    assert (row["word"], row["tone"]) == ("Working", "live") and not row["resume"] and not row["question"], row

    # A reload reads the same projection: the closed helper does not come back as a turn.
    page.reload()
    rw.open_project(page, rw.PROJECT)
    transport.refresh()
    reloaded = rw.wait_facts(page, "project", lambda f: f["badge"] == "Working" and "owner" not in f["mark"] and "amber" not in f["mark"],
                             "the next stage still alone after a reload")
    assert (reloaded["tone"], reloaded["mark"]) == ("live", "moving ink"), reloaded
