"""Reviewer-found lifecycle edges through the real SPA, controlled census, both engines."""
# ruff: noqa: F401, F811 -- pytest discovers the imported, parameter-named fixtures.
import time

import pytest

from tests.test_room_work_browser import (
    PROJECT,
    Transport,
    activity,
    capture,
    evidence,
    open_app,
    open_project,
    room_ui,
    row_main,
    subscription_ui,
    trigger,
    wait_facts,
)

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def test_last_cardless_project_turn_returns_focus_to_its_own_composer(room_ui):
    transport = Transport(room_ui["page"], projects=[PROJECT], history={1: [], 42: []},
                          activities=[activity("last-project-turn", kind="direct_chat", chat_id=42)])
    page = open_app(room_ui, transport)
    open_project(page, PROJECT)
    transport.refresh()
    trigger(page, "project").click()
    wait_facts(page, "project", lambda f: len(f["rows"]) == 1, "the cardless Project turn")
    row_main(page, "last-project-turn", "project").focus()
    transport.activities.clear()
    transport.refresh()
    wait_facts(page, "project", lambda f: f["rows"][0]["settled"], "retained ended row")
    page.keyboard.press("Escape")
    # Closing returns focus immediately; only the inert picture waits for its fold.
    assert page.locator("#project-panel .chat-input").evaluate("el => el === document.activeElement")
    current = wait_facts(page, "project", lambda f: f["surfaceHidden"] and f["disabled"], "the ended list folded away")
    evidence("project-ended-focus", current)
    capture(page, "project-ended-focus")
    assert current["surfaceHidden"] and current["disabled"], current
    assert page.locator("#project-panel .chat-input").evaluate("el => el === document.activeElement"), current
    assert transport.posts == [], transport.posts


@pytest.mark.parametrize("verb,endpoint,phase", [
    ("hurry", "hurry", "working"), ("resume", "resume", "budget_paused"),
    ("stop_now", "cancel", "working"),
])
def test_shared_card_and_header_adapter_dispatches_exact_task(room_ui, verb, endpoint, phase):
    """Use the real menu and direct Resume owners; paused rows offer Resume beside the menu."""
    transport = Transport(room_ui["page"])
    page = open_app(room_ui, transport)
    task_id = "exact-adapter-" + verb
    offered = page.evaluate("""async ([id, phase]) => {
        const {openTaskControls, createTaskResumeButton, requestStop} = await import('/static/modules/task_control_menu.js');
        const anchor = document.createElement('button');
        anchor.textContent = 'Task controls'; anchor.dataset.id = id;
        const page = document.querySelector('#page-chat');
        page.append(anchor);
        if (phase === 'budget_paused') page.append(createTaskResumeButton(id, {canResume: () => true}));
        openTaskControls(anchor, id, {activity: {phase}, stop: action => requestStop(id, action)});
        return [...document.querySelectorAll('.task-control-item')].map(el => el.dataset.taskControl);
    }""", [task_id, phase])
    if verb == 'resume':
        assert offered == ['stop_now'], offered  # no duplicate Resume in the menu
        page.keyboard.press('Escape')
        page.locator('#page-chat button[data-resume-run]').click()
    else:
        assert verb in offered, offered
        page.locator(f".task-control-item[data-task-control='{verb}']").click()
    expected = f"/api/tasks/{task_id}/{endpoint}"
    deadline = time.monotonic() + 5
    while not transport.posts and time.monotonic() < deadline:
        page.wait_for_timeout(20)
    assert [path for path, _ in transport.posts] == [expected], transport.posts
    if endpoint == "cancel":
        assert transport.posts[0][1]["cascade"] is True
    assert transport.decision_posts() == []
