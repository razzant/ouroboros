"""#708 in a real browser: the production chat renders what a real cleanup over real Git published.

The rows and the terminal event come from the real lifecycle, outbox, send handler and history endpoint
(``tests/test_evolution_cleanup_disclosure.py``); nothing about the card is staged. Each scene is painted live
from the frames the bridge broadcast, then repainted from the history endpoint, then by a fresh page that knows
only that history (a reload). A campaign Stop's cleanup lands after its cycle's card has settled: the page then
holds the history it read at the terminal, and only the late row's own resync can bring the stored line.
``OUROBOROS_UI_BROWSER_ENGINE`` selects the engine; screenshots and what ``measure.js`` reads off each one go to
``OUROBOROS_UI_EVIDENCE_DIR``, named by the engine that drew them, width and theme.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_evolution_cleanup_disclosure import (  # noqa: F401
    CHAT, TASK, ahead, cycle, dirty, fault_git, git, read_only, run_printed, undeletable,
)
from tests.test_subscription_setup_browser import subscription_ui  # noqa: F401

pytestmark = [pytest.mark.ui_browser]

MEASURE = Path(__file__).resolve().parents[1] / "skills" / "ui_design" / "references" / "measure.js"
PAGE = """<!doctype html><html class="ouro-ui" data-theme="{theme}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="/static/ui.css"><link rel="stylesheet" href="/static/style.css">
<script src="/static/purify.min.js"></script><script src="/static/marked.min.js"></script></head>
<body><main id="content"></main></body></html>"""
START_CHAT = """async () => {
    const { createChatInstance } = await import('/static/modules/chat.js');
    const handlers = new Map();
    window.cleanupChat = createChatInstance({
        ws: { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
              isConnected: () => true, send() {} },
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 }, updateUnreadBadge() {},
        stateSnapshots: { begin: () => ({generation: 1, requestedAt: Date.now()}),
            gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {} },
        chatId: 7, mountEl: document.querySelector('#content'), asPanel: true,
    });
    window.cleanupEmit = (type, frame) => handlers.get(type)(frame);
}"""
SCROLL_END = """id => { const box = document.getElementById(`chat-live-timeline-${id}`);
    if (!box || box.scrollHeight <= box.clientHeight + 1) return false;
    box.scrollTop = box.scrollHeight; return true; }"""
PROGRESS = {"type": "chat", "task_id": TASK, "chat_id": CHAT, "role": "system", "is_progress": True,
            "content": "Reading the backlog", "ts": "2026-11-01T17:59:00+00:00"}


def _terminal(scene, status="completed"):
    """The real evolution terminal seam, then the real durable projection its dispatcher makes."""
    from ouroboros.outcomes import normalize_outcome_axes
    from ouroboros.project_dialogue import append_terminal_task_projection
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor import events_evolution_done

    write_task_result(scene.data, TASK, status, chat_id=CHAT, root_task_id=TASK)
    row = load_task_result(scene.data, TASK)
    event = {"ts": "2026-11-01T18:00:00+00:00", "type": "task_done", "task_id": TASK, "task_type": "evolution",
             "status": status, "chat_id": CHAT, "outcome_axes": normalize_outcome_axes(row)}
    metadata = {"evolution_transaction": scene.tx}
    ctx = SimpleNamespace(DRIVE_ROOT=scene.data, REPO_DIR=scene.repo, append_jsonl=lambda *a: None)
    events_evolution_done._handle_evolution_task_done(
        ctx, evt={"task_id": TASK, "metadata": metadata}, task_id=TASK, task={"id": TASK, "metadata": metadata},
        task_done_event=event, outcome_axes=event["outcome_axes"], cost=0.1, rounds=2)
    assert append_terminal_task_projection(scene.data, TASK, {"id": TASK, "chat_id": CHAT},
                                           load_task_result(scene.data, TASK), event)
    return event


def _history_payload(scene):
    """The history endpoint's whole answer, its paging facts included, as the page reads it. On its own thread:
    a scene that reads it again while its page is open is inside Playwright's event loop."""
    from concurrent.futures import ThreadPoolExecutor

    from ouroboros.gateway.history import make_chat_history_endpoint

    endpoint = make_chat_history_endpoint(scene.data)
    with ThreadPoolExecutor(1) as pool:
        response = pool.submit(asyncio.run, endpoint(SimpleNamespace(query_params={"chat_id": str(CHAT)}))).result()
    assert response.status_code == 200
    return json.loads(response.body)


def _progress(scene):
    """The cycle's own narration, as a worker stored it: the card exists because the cycle ran."""
    from ouroboros import dialogue_store

    dialogue_store.append(scene.data, "progress", {
        "type": "task_progress", "chat_id": CHAT, "task_id": TASK, "text": "Reading the backlog",
        "ts": "2026-11-01T17:59:00+00:00", "narration": True})


def _detail(scene):
    from ouroboros.gateway.tasks import _task_get_response

    response = _task_get_response(SimpleNamespace(path_params={"task_id": TASK}, app=SimpleNamespace(
        state=SimpleNamespace(drive_root=scene.data))))
    assert response.status_code == 200
    return json.loads(response.body)


def _open_page(ui, scene, *, history_payload, width, theme, reads=None):
    """``reads`` collects every history request the page makes, so a scene can tell its reads apart."""
    page = ui["new_page"](viewport={"width": width, "height": 900})
    page.route("**/cleanup-card", lambda route: route.fulfill(content_type="text/html", body=PAGE.format(theme=theme)))

    def history(route):
        (reads if reads is not None else []).append(route.request.url)
        route.fulfill(content_type="application/json", body=json.dumps(history_payload()))

    page.route("**/api/chat/history?*", history)
    detail = _detail(scene)
    page.route(f"**/api/tasks/{TASK}*", lambda route: route.fulfill(content_type="application/json", body=json.dumps(detail)))
    page.goto(ui["url"] + "/cleanup-card")
    page.evaluate(START_CHAT)
    return page


def _shot(page, name, tmp_path, theme):
    """The screen at 1:1 and what ``measure.js`` reads off the cycle's card (or the page) after that paint."""
    output = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or tmp_path)
    output.mkdir(parents=True, exist_ok=True)
    engine = page.context.browser.browser_type.name  # the engine that really drew it, whatever was asked for
    stem = f"cleanup-{engine}-{page.viewport_size['width']}-{theme}-{name}"
    page.screenshot(path=str(output / f"{stem}.png"), full_page=True, animations="disabled")
    subject = f'.chat-live-card[data-task-id="{TASK}"]' if page.locator(".chat-live-card").count() else ".chat-bubble"
    page.locator(subject).first.screenshot(path=str(output / f"{stem}-card.png"), animations="disabled")  # its whole height
    if page.evaluate(SCROLL_END, TASK):  # the card's timeline scrolls on its own: its last row, read to the end
        page.locator(subject).first.screenshot(path=str(output / f"{stem}-card-end.png"), animations="disabled")
    measured = page.evaluate(f"({MEASURE.read_text(encoding='utf-8')})({json.dumps({'subject': subject})})")
    (output / f"{stem}-measure.json").write_text(json.dumps(measured, ensure_ascii=False, indent=1), encoding="utf-8")
    rows = [f for f in measured["findings"] if f["kind"] == "page-overflow" or (
        f["kind"] in {"text-clipped", "text-covered"} and ("chat-live-line" in f["where"] or "chat-bubble" in f["where"]))]
    assert not rows, rows  # the card's own chrome is measured too, and kept as evidence beside the shot
    return measured


def _expect_card(playwright, page, phase, line, details, absent=()):
    """Collapsed: the phase word and one short fact, no ids. Details: the exact addresses, literally."""
    card = page.locator(f'.chat-live-card[data-task-id="{TASK}"]')
    playwright.expect(card.locator("[data-live-phase]").first).to_have_text(phase)
    activity = card.locator("[data-live-activity]").first
    playwright.expect(activity).to_contain_text(line)
    for value in details:
        assert value not in activity.inner_text(), f"{value} belongs in Details"
    toggle = card.locator(":scope > [data-live-summary-button] [data-live-disclosure]")
    if card.get_attribute("data-expanded") != "1":
        toggle.click()
    row = card.locator(".chat-live-line").filter(has_text="Repository:")  # the terminal's own line repeats only the fact
    playwright.expect(row).to_have_count(1)
    text = row.inner_text()
    for value in details:
        assert value in text, value
    for value in absent:
        assert value not in text, value
    assert row.locator("em, strong em").count() == 0, "an id, branch or path must not render as emphasis"
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    return row


def _quiet(page, reads, still_ms=1500, limit_ms=8000):
    """Wait until the page has made no history read for ``still_ms`` (twice the resync debounce)."""
    seen, still = len(reads), 0
    for _ in range(limit_ms // 100):
        page.wait_for_timeout(100)
        still = still + 100 if len(reads) == seen else 0
        seen = len(reads)
        if still >= still_ms:
            return
    raise AssertionError(f"history reads never went quiet: {len(reads)}")


def _close(page):
    page.evaluate("() => cleanupChat.destroy()")
    page.close()


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("width", [1100, 390])
def test_no_op_cleanup_on_the_real_card_live_and_after_reload(request, cycle, tmp_path, width, theme):  # noqa: F811
    playwright = pytest.importorskip("playwright.sync_api")
    dirty(cycle)
    _progress(cycle)
    (cycle.repo / "long_named_directory_with_underscores").mkdir()
    (cycle.repo / "long_named_directory_with_underscores" / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    leftover = ahead(cycle)
    event = _terminal(cycle)
    oid = git(cycle.repo, "rev-parse", "refs/stash")
    tx = cycle.campaign()["history"][-1]["transaction"]
    details = [oid, tx["cleanup_preserved_ref"], leftover, cycle.base, str(cycle.repo)]
    line = "Evolution cleanup moved local changes to a Git stash and kept local commits on a branch"
    cycle.deliver()
    frames = [frame for frame in cycle.frames if frame.get("card_row_id")]
    assert len(frames) == 1
    history = _history_payload(cycle)
    ui = request.getfixturevalue("subscription_ui")
    page = _open_page(ui, cycle, history_payload=lambda: history, width=width, theme=theme)
    try:
        # Live: the card exists, the owed row lands on it, the terminal closes it with the fact it carried.
        page.evaluate("""f => cleanupEmit('chat', f)""", PROGRESS)
        page.evaluate("""f => cleanupEmit('chat', f)""", frames[0])
        page.evaluate("""f => cleanupEmit('log', f)""", {"type": "log", "chat_id": CHAT, "data": event})
        row = _expect_card(playwright, page, "Done", line, details)
        assert "whole dirty checkout" in row.inner_text()
        _shot(page, "no-op-live", tmp_path, theme)

        # The same page repaints from the history endpoint: still one row.
        replay = page.evaluate("() => cleanupChat.refreshHistory({revision: 1})")
        assert replay["painted"]
        _expect_card(playwright, page, "Done", line, details)
    finally:
        _close(page)

    # A reload knows only the history endpoint.
    page = _open_page(ui, cycle, history_payload=lambda: history, width=width, theme=theme)
    try:
        assert page.evaluate("() => cleanupChat.refreshHistory({revision: 1})")["painted"]
        card = page.locator(f'.chat-live-card[data-task-id="{TASK}"]')
        playwright.expect(card.locator("[data-live-activity]").first).to_contain_text(line)
        _shot(page, "no-op-reload-collapsed", tmp_path, theme)
        _expect_card(playwright, page, "Done", line, details)
        _shot(page, "no-op-reload-details", tmp_path, theme)
    finally:
        _close(page)


@pytest.mark.parametrize(("width", "theme"), [(1100, "light"), (390, "dark")])
def test_partial_success_reads_as_what_it_was_after_reload(request, cycle, monkeypatch, tmp_path, width, theme):  # noqa: F811
    playwright = pytest.importorskip("playwright.sync_api")
    dirty(cycle)
    leftover = ahead(cycle)
    _progress(cycle)
    fault_git(monkeypatch, "reset")
    _terminal(cycle)
    oid = git(cycle.repo, "rev-parse", "refs/stash")
    cycle.deliver()
    history = _history_payload(cycle)
    ui = request.getfixturevalue("subscription_ui")
    page = _open_page(ui, cycle, history_payload=lambda: history, width=width, theme=theme)
    try:
        assert page.evaluate("() => cleanupChat.refreshHistory({revision: 1})")["painted"]
        row = _expect_card(playwright, page, "Done", "but the reset to the cycle's starting commit failed",
                           [oid, leftover, "reset refused"])
        assert "The checkout is still at" in row.inner_text()
        _shot(page, "partial-reload", tmp_path, theme)
    finally:
        _close(page)


COPY_CODE = """code => { const range = document.createRange(); range.selectNodeContents(code);
    const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range);
    return selection.toString(); }"""


@pytest.mark.parametrize(("width", "theme"), [(1100, "light"), (390, "dark")])
def test_a_failed_stash_reads_as_what_it_was_and_its_copied_command_restores(  # noqa: F811
        request, cycle, read_only, tmp_path, width, theme):  # noqa: F811
    """Real Git saved the entry and removed a file, then failed. The card says so, and the command as a person
    selects it on the page (repeated spaces in the path included) gives the file back."""
    playwright = pytest.importorskip("playwright.sync_api")
    leftover = ahead(cycle)
    undeletable(cycle, read_only)
    _progress(cycle)
    _terminal(cycle)
    oid = git(cycle.repo, "rev-parse", "refs/stash")
    cycle.deliver()
    assert not (cycle.repo / "untracked.txt").exists()
    history = _history_payload(cycle)
    ui = request.getfixturevalue("subscription_ui")
    page = _open_page(ui, cycle, history_payload=lambda: history, width=width, theme=theme)
    try:
        assert page.evaluate("() => cleanupChat.refreshHistory({revision: 1})")["painted"]
        row = _expect_card(playwright, page, "Done", "but the stash command failed, so the cleanup stopped there",
                           [oid, leftover, "locked/held.txt", str(cycle.repo)])
        assert "The checkout is still at" in row.inner_text() and "may already be gone" in row.inner_text()
        _shot(page, "failed-stash", tmp_path, theme)
        copied = row.locator("code").filter(has_text=" stash apply ").evaluate(COPY_CODE)
        overwrite = row.locator("code").filter(has_text=" restore --source=").evaluate(COPY_CODE)
    finally:
        _close(page)
    assert str(cycle.repo) in copied and copied.endswith(f" stash apply {oid}")
    assert run_printed(copied).returncode != 0, "it reports the files still in place, as the row says"
    assert (cycle.repo / "untracked.txt").read_text(encoding="utf-8") == "somebody's new file\n"
    assert (cycle.repo / "tracked.txt").read_text(encoding="utf-8") == "somebody's edit\n"
    assert str(cycle.repo) in overwrite and overwrite.endswith(f" restore --source={oid} --worktree -- :/")


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("width", [1100, 390])
def test_a_late_stop_cleanup_updates_its_settled_card_live_and_after_reload(request, cycle, tmp_path, width, theme):  # noqa: F811
    """The cycle was stopped and its card already says Stopped from the history the page read then. Minutes later
    the campaign Stop's cleanup moves the work: its owed row lands on that card, and the card's collapsed line takes
    the stored fact live, through the history read that row schedules. Stopped stays Stopped."""
    from ouroboros.outcomes import normalize_outcome_axes
    from ouroboros.project_dialogue import append_terminal_task_projection
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor import evolution_lifecycle

    playwright = pytest.importorskip("playwright.sync_api")
    dirty(cycle)
    _progress(cycle)
    write_task_result(cycle.data, TASK, "cancelled", chat_id=CHAT, root_task_id=TASK, reason_code="cancelled",
                      cancel_origin={"source": "evolution_stop", "reason": "disabled via owner chat"})
    row = load_task_result(cycle.data, TASK)
    terminal = {"ts": "2026-11-01T18:00:00+00:00", "type": "task_done", "task_id": TASK, "task_type": "evolution",
                "status": "cancelled", "chat_id": CHAT, "outcome_axes": normalize_outcome_axes(row)}
    assert append_terminal_task_projection(cycle.data, TASK, {"id": TASK, "chat_id": CHAT}, row, terminal)
    assert "evolution_cleanup" not in terminal["outcome_axes"]["execution"]
    served = {"now": _history_payload(cycle)}  # what the history endpoint answers at this moment
    reads = []
    line = "Evolution cleanup moved local changes to a Git stash"
    ui = request.getfixturevalue("subscription_ui")
    page = _open_page(ui, cycle, history_payload=lambda: served["now"], width=width, theme=theme, reads=reads)
    card = page.locator(f'.chat-live-card[data-task-id="{TASK}"]')
    try:
        page.evaluate("""f => cleanupEmit('chat', f)""", PROGRESS)
        page.evaluate("""f => cleanupEmit('log', f)""", {"type": "log", "chat_id": CHAT, "data": terminal})
        playwright.expect(card.locator("[data-live-phase]").first).to_have_text("Stopped")
        _quiet(page, reads)  # the page's first read and the terminal's own completion read come and go first
        settled_reads = len(reads)
        assert line not in card.locator("[data-live-activity]").first.inner_text()
        _shot(page, "stop-before-cleanup", tmp_path, theme)

        ended = evolution_lifecycle.complete_evolution_campaign("disabled via owner chat", status="stopped")
        assert ended["transaction_history"][-1]["cycle_outcome"] == "owner_stopped"
        oid = git(cycle.repo, "rev-parse", "refs/stash")
        cycle.deliver()
        [frame] = [frame for frame in cycle.frames if frame.get("card_row_id")]
        served["now"] = _history_payload(cycle)
        page.evaluate("""f => cleanupEmit('chat', f)""", frame)

        playwright.expect(card.locator("[data-live-activity]").first).to_contain_text(line)
        assert len(reads) > settled_reads, "the stored line came from the read the late row scheduled"
        _shot(page, "stop-late-live-collapsed", tmp_path, theme)
        _expect_card(playwright, page, "Stopped", line, [oid, str(cycle.repo)])
        _shot(page, "stop-late-live-details", tmp_path, theme)
    finally:
        _close(page)
    page = _open_page(ui, cycle, history_payload=lambda: served["now"], width=width, theme=theme)
    try:
        assert page.evaluate("() => cleanupChat.refreshHistory({revision: 1})")["painted"]
        _expect_card(playwright, page, "Stopped", line, [oid, str(cycle.repo)])
        assert not page.get_by_role("button", name="Retry loading messages").is_visible()
        _shot(page, "stop-reload", tmp_path, theme)
    finally:
        _close(page)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_a_cycle_without_a_card_gets_the_row_alone(request, cycle, tmp_path, theme):  # noqa: F811
    from supervisor import evolution_cleanup

    playwright = pytest.importorskip("playwright.sync_api")
    dirty(cycle)
    tx = {"transaction_id": "orphan-tx", "base_head": cycle.base, "task_id": "never-started"}
    evolution_cleanup._cleanup_worktree_after_cycle(tx, "never-started")
    evolution_cleanup.publish_cleanup(tx, "never-started", cycle.data)
    cycle.deliver()
    history = _history_payload(cycle)
    assert [row["task_id"] for row in history["messages"]] == ["never-started"]
    ui = request.getfixturevalue("subscription_ui")
    page = _open_page(ui, cycle, history_payload=lambda: history, width=1100, theme=theme)
    try:
        assert page.evaluate("() => cleanupChat.refreshHistory({revision: 1})")["painted"]
        assert page.locator(".chat-live-card").count() == 0
        bubble = page.locator(".chat-bubble").filter(has_text="Evolution cleanup moved local changes")
        playwright.expect(bubble).to_have_count(1)
        assert tx["cleanup_stash_oids"][0] in bubble.inner_text()
        assert "`" not in bubble.inner_text(), "the row's literals render as code in a standalone bubble too"
        assert bubble.locator("code").first.inner_text() == str(cycle.repo), "exact: expect() would normalize its spaces"
        _shot(page, "no-card", tmp_path, theme)
    finally:
        _close(page)


def test_a_cleanup_that_moved_nothing_adds_nothing_to_the_card(request, cycle, tmp_path):  # noqa: F811
    playwright = pytest.importorskip("playwright.sync_api")
    _progress(cycle)
    _terminal(cycle)
    cycle.deliver()
    history = _history_payload(cycle)
    ui = request.getfixturevalue("subscription_ui")
    page = _open_page(ui, cycle, history_payload=lambda: history, width=1100, theme="dark")
    try:
        assert page.evaluate("() => cleanupChat.refreshHistory({revision: 1})")["painted"]
        card = page.locator(f'.chat-live-card[data-task-id="{TASK}"]')
        playwright.expect(card.locator("[data-live-phase]").first).to_have_text("Done")
        toggle = card.locator(":scope > [data-live-summary-button] [data-live-disclosure]")
        if card.get_attribute("data-expanded") != "1":
            toggle.click()
        assert card.locator(".chat-live-line").filter(has_text="cleanup").count() == 0
        assert "cleanup" not in card.inner_text().lower()
        _shot(page, "nothing-moved", tmp_path, "dark")
    finally:
        _close(page)
