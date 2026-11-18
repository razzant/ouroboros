"""Project and Main Find geometry on a frozen public checkout, with a page-only overlay."""
import pytest
from playwright.sync_api import expect

from tests import test_room_work_browser as rw

from .capture import Frames
from .scene import PROJECT
from .test_preview import setup, subscription_ui as subscription_ui, theme as theme

pytestmark = pytest.mark.ui_browser
BASE = "54f57a5b5baf9c6448bc60a3e7788cf7290dd9be"
MEASURE = """root => {
    const panel = document.querySelector(root), sheet = panel.querySelector('.chat-find');
    const header = root === '#page-chat' ? panel.querySelector('.chat-page-header')
        : document.querySelector('#project-panel .project-panel-bar');
    const p = panel.getBoundingClientRect(), s = sheet.getBoundingClientRect(), h = header.getBoundingClientRect();
    return {height:h.height, flow:getComputedStyle(header).position, gap:s.top-p.top, fromBar:s.top-h.bottom,
        focused:document.activeElement === sheet.querySelector('input'), keyboard:document.body.classList.contains('keyboard-open')};
}"""


@pytest.mark.parametrize("room", ["main", "project"])
def test_find(subscription_ui, theme, room):
    page, scene = setup(subscription_ui, theme, "Now", width=390, baseline=BASE)
    root = "#page-chat"
    if room == "project":
        rw.open_project(page, PROJECT)
        root = f"#panel-pchat-{PROJECT['id']}"
    page.keyboard.press("Control+f")
    expect(page.locator(root + " .chat-find input")).to_be_focused()
    frames = Frames(page, theme, f"find-{room}")
    baseline = page.evaluate(MEASURE, root)
    assert baseline["focused"] and not baseline["keyboard"]
    frames.add("Полная высота экрана, поле поиска в фокусе.")
    # Exercise the frontend's own keyboard detection by changing the viewport only.
    page.set_viewport_size({"width": 390, "height": 544})
    page.wait_for_selector("body.keyboard-open", state="attached")
    rw.settle_frames(page)
    before = page.evaluate(MEASURE, root)
    frames.add("Уменьшенное окно до CSS-наложения.")
    page.add_style_tag(content="body.keyboard-open .project-panel .chat-find { top: var(--space-1); }")
    rw.settle_frames(page)
    after = page.evaluate(MEASURE, root)
    frames.add("То же окно с CSS-наложением, без изменения исходных файлов.")
    assert before["focused"] and before["keyboard"] and before["height"] > 0
    assert after["focused"] and after["keyboard"] and after["height"] > 0
    if room == "project":
        assert before["flow"] == "relative" and before["gap"] >= 0.8 * before["height"], before
        assert before["fromBar"] >= 0.75 * before["height"], before
        assert 0 <= after["gap"] <= 16 and abs(after["fromBar"]) <= 16, after
    else:
        assert abs(after["gap"] - before["gap"]) <= 1 and abs(after["fromBar"] - before["fromBar"]) <= 1
    assert not scene.posts_wire
    frames.finish_find([round(row["fromBar"], 1) for row in (baseline, before, after)])
