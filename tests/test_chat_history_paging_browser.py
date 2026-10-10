"""Positive browser acceptance over real retained JSONL history and its gateway.

Only the request transport can be held/failed; every successful page and cursor
comes from the real history endpoint. The shared server uses its local mock LLM.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET, _SETTLE_RESTORE_FRAMES, _emit_ws_frame

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
MAIN = "#chat-messages"
HISTORY_URL = "/api/chat/history"
_FRAMES = "() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done)))"
_EDGE_SCROLL = """(root, direction) => {
    root.scrollTop = direction === 'older' ? 0 : root.scrollHeight;
    root.dispatchEvent(new WheelEvent('wheel', {deltaY: direction === 'older' ? -1 : 1}));
    root.dispatchEvent(new Event('scroll'));
}"""
_OBSERVE_HISTORY = """() => {
    const fetch = window.fetch.bind(window);
    window.__historyReads = [];
    window.__historyFault = '';
    window.fetch = async (input, init) => {
        const url = new URL(typeof input === 'string' ? input : input.url, location.href);
        if (url.pathname !== '/api/chat/history') return fetch(input, init);
        const read = {cursor: url.searchParams.get('cursor'),
            chatId: Number(url.searchParams.get('chat_id') || 1), done: false};
        window.__historyReads.push(read);
        if (read.cursor && window.__historyFault === 'hold') {
            window.__historyFault = '';
            await new Promise(resolve => { window.__releaseHistory = resolve; window.__heldHistory = read; });
        } else if (read.cursor && window.__historyFault === 'fail') {
            window.__historyFault = ''; read.error = 'injected transport failure'; read.done = true;
            throw new TypeError(read.error);
        }
        try {
            const response = await fetch(input, init);
            read.status = response.status;
            read.body = await response.clone().json();
            read.done = true;
            return response;
        } catch (error) { read.error = String(error); read.done = true; throw error; }
    };
}"""


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _result(root, task_id, **fields):
    _write(root / "task_results" / f"{task_id}.json", [{
        "_schema_version": 1, "task_id": task_id, "status": "completed", **fields,
    }])


def _human(index, chat_id=1, **extra):
    return {"ts": "2026-09-12T10:00:00Z", "chat_id": chat_id,
            "direction": "in", "text": f"history-human-{index:04d}", **extra}


def _progress(index, chat_id=1, **extra):
    return {"ts": "2026-09-12T10:00:00Z", "chat_id": chat_id,
            "task_id": f"history-task-{index // 145}",
            "content": f"history-progress-{index:04d}", **extra}


def _bulk_history(root):
    for segment in range(5):
        _write(root / "archive" / f"chat_20260901T00000{segment}.jsonl",
               [_human(segment * 360 + index) for index in range(360)])
        _write(root / "archive" / f"progress_20260901T00000{segment}.jsonl",
               [_progress(segment * 145 + index) for index in range(145)])
    _write(root / "logs" / "chat.jsonl", [_human(1800)])
    _write(root / "logs" / "progress.jsonl", [_progress(725)])
    for index in range(6):
        _result(root, f"history-task-{index}")


def _sparse_history(root):
    from ouroboros.projects_registry import create_project

    project = create_project(root, "history-sparse", name="Sparse archive room")
    foreign = create_project(root, "history-foreign", name="Other archive room")
    _write(root / "archive" / "chat_20260801T000000.jsonl", [
        _human(0, project["chat_id"], text="SPARSE_FIRST_SAVED_MESSAGE"),
    ])
    for segment in range(5):
        _write(root / "archive" / f"chat_20260902T00000{segment}.jsonl", [
            _human(index, foreign["chat_id"], text="OTHER_ROOM_ONLY " + "x" * 2000)
            for index in range(350)
        ])
    with (root / "logs" / "chat.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_human(1, project["chat_id"], text="SPARSE_LATEST_MESSAGE")) + "\n")
    return project


def _open(page, url):
    page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
    page.add_init_script(f"({_OBSERVE_HISTORY})()")
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_function("() => window.__testSockets?.some(socket => socket.readyState === WebSocket.OPEN)")
    _idle(page, MAIN)


def _idle(page, feed):
    page.wait_for_function("""feed => {
        const root = document.querySelector(feed);
        return root?.querySelector('.chat-load-older')
            && !root.querySelector('.chat-load-older button')?.disabled
            && window.__historyReads.every(read => read.done);
    }""", arg=feed, timeout=30_000)
    page.evaluate(_FRAMES)


def _reads(page, chat_id=1):
    return page.evaluate("id => window.__historyReads.filter(read => read.chatId === id)", chat_id)


def _step(page, feed, direction="older", *, automatic=False):
    before = page.evaluate("() => window.__historyReads.length")
    if automatic and direction == "older":
        page.locator(feed).evaluate(_EDGE_SCROLL, direction)
    else:
        # Mixed cards have no physical edge: the common button fills the known gap.
        # This exercises the visible button's handler without changing a reader's
        # selection or forcing an off-screen control into the reading viewport.
        page.locator(f"{feed} .chat-load-older button").evaluate("node => node.click()")
    page.wait_for_function("n => window.__historyReads.length > n", arg=before, timeout=30_000)
    _idle(page, feed)


def _to_beginning(page, feed):
    for _ in range(80):
        _idle(page, feed)
        # The walk's own reads carry a cursor. A refresh's recent read (a card finished,
        # the room's revision moved) can land after the last of them and says nothing
        # about the walk, so the beginning is the last cursor read that has no more.
        latest = page.evaluate("""() => {
            const reads = window.__historyReads.filter(read => read.done && read.body);
            return (reads.filter(read => read.cursor).at(-1) || reads.at(-1))?.body;
        }""")
        if latest and latest.get("has_more") is False:
            note = page.locator(feed).locator('..').locator('.chat-load-older-note').inner_text()
            assert note in {"Beginning of saved history", "Some saved history is not loaded. Shown messages may have gaps."}
            return
        _step(page, feed, automatic=True)
    pytest.fail("archive navigation did not reach its physical beginning")


def _open_project(page, project):
    row = page.locator(f'.nav-project-row[data-project-id="{project["id"]}"]')
    row.wait_for(state="attached", timeout=30_000)
    mobile_toggle = page.locator('#page-chat [data-mobile-nav-toggle]')
    # A translated-offscreen drawer is still CSS-visible to Playwright.
    if mobile_toggle.is_visible() and not page.locator('#primary-sidebar').evaluate("node => node.classList.contains('open')"):
        mobile_toggle.click()
    row.click()
    feed = f'#pchat-{project["id"]}-messages'
    page.locator(feed).wait_for(state="visible", timeout=30_000)
    _idle(page, feed)
    # Settle completed rendering before the first deliberate edge gesture.
    page.evaluate(_SETTLE_RESTORE_FRAMES)
    return feed


def _screenshot(page, tmp_path, name):
    root = Path(os.environ.get("HISTORY_UI_EVIDENCE_DIR") or tmp_path / "history-ui-evidence")
    root.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(root / f"{name}.png"), full_page=False, animations="disabled")


# A room shows its newest message (DESIGN "History edges"): the feed rests at its
# end and its last row is on screen, clear of the composer drawn over the feed.
_AT_NEWEST = """feed => {
    const root = document.querySelector(feed), box = root.getBoundingClientRect();
    const node = [...root.querySelectorAll('[data-history-id]')].at(-1);
    const composer = root.parentElement.querySelector('.chat-input-area, #chat-input-area');
    const floor = Math.min(box.bottom, composer?.getBoundingClientRect().top ?? box.bottom);
    const rect = node?.getBoundingClientRect();
    return {gap: root.scrollHeight - root.scrollTop - root.clientHeight, text: node?.textContent || '',
        top: rect ? rect.top - box.top : null, clearance: rect ? floor - rect.bottom : null};
}"""
_ROW_OFFSET = "n => n.getBoundingClientRect().top - n.closest('.chat-messages').getBoundingClientRect().top"


def _assert_at_newest(page, feed, newest):
    """The room is at its newest message: no distance left to its end, that row on screen."""
    page.evaluate(_FRAMES)
    at = page.evaluate(_AT_NEWEST, feed)
    assert at["gap"] <= 8 and newest in at["text"], at
    assert at["top"] is not None and at["top"] >= 0 and at["clearance"] >= -1, at
    return at


def _capture_selection_failure(page, title, output, engine, before_box, line_height):
    """Save this synthetic drag's facts before its browser context closes."""
    directory = Path(output) / "browser" / f"history-selection-{engine}-{page.viewport_size['width']}"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        facts = title.evaluate("""(node, before) => {
            const rect = node.getBoundingClientRect(), style = getComputedStyle(node);
            const selection = getSelection();
            const point = {x: before.box.x + 7, y: before.box.y + before.lineHeight / 2};
            return {viewport: {width: innerWidth, height: innerHeight, dpr: devicePixelRatio},
                selection: selection.toString(), rangeCount: selection.rangeCount,
                before, after: {x: rect.x, y: rect.y, width: rect.width, height: rect.height},
                connected: node.isConnected, userSelect: style.userSelect, lineHeight: style.lineHeight,
                hit: document.elementFromPoint(point.x, point.y)?.outerHTML,
                title: node.outerHTML, line: node.closest('.chat-live-line')?.outerHTML};
        }""", {"box": before_box, "lineHeight": line_height})
        (directory / "selection.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    except Exception as error:
        print(f"HISTORY_SELECTION_DIAGNOSTIC facts unavailable: {type(error).__name__}", flush=True)
    for name, capture in (
        ("screenshot", lambda: page.screenshot(path=str(directory / "screenshot.png"), timeout=5_000)),
        ("dom", lambda: (directory / "page.html").write_text(page.content(), encoding="utf-8")),
    ):
        try:
            capture()
        except Exception as error:
            print(f"HISTORY_SELECTION_DIAGNOSTIC {name} unavailable: {type(error).__name__}", flush=True)
    print(f"HISTORY_SELECTION_DIAGNOSTIC {directory}", flush=True)


def _assert_main_beginning_visible(page):
    page.locator(MAIN).evaluate("node => { node.scrollTop = 0; node.dispatchEvent(new Event('scroll')); }")
    page.evaluate(_SETTLE_RESTORE_FRAMES)
    bounds = page.evaluate("""() => {
        const root = document.querySelector('#chat-messages');
        const first = [...root.querySelectorAll('.message')].find(node => node.textContent === 'history-human-0000');
        const note = root.parentElement.querySelector('.chat-load-older-note');
        const header = document.querySelector('#page-chat .chat-page-header');
        const box = root.getBoundingClientRect();
        const noteBox = note?.getBoundingClientRect(), headerBox = header?.getBoundingClientRect();
        return {top: first?.getBoundingClientRect().top, noteTop: noteBox?.top, noteBottom: noteBox?.bottom,
            noteInHeader: header?.contains(note), headerTop: headerBox?.top, headerBottom: headerBox?.bottom,
            floor: Math.max(box.top, header?.getBoundingClientRect().bottom || 0),
            bottom: box.bottom, scrollTop: root.scrollTop, note: note?.textContent};
    }""")
    assert bounds["scrollTop"] <= 1, bounds
    assert bounds["floor"] - 2 <= bounds["top"] < bounds["bottom"], bounds
    # Uncertain coverage stays in persistent chrome; a complete beginning note
    # belongs at the feed edge. Both must remain visible in their actual owner.
    note_floor = bounds["headerTop"] if bounds["noteInHeader"] else bounds["floor"]
    note_ceiling = bounds["headerBottom"] if bounds["noteInHeader"] else bounds["bottom"]
    assert note_floor - 2 <= bounds["noteTop"] < bounds["noteBottom"] <= note_ceiling + 2, bounds
    assert bounds["note"] in {"Beginning of saved history", "Some saved history is not loaded. Shown messages may have gaps."}, bounds


@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
def test_history_archive_navigation_rotation_retry_and_sparse_project(
    direct_server_with_data, browser_engine, tmp_path,
):
    from playwright.sync_api import sync_playwright

    root, url = direct_server_with_data["data_dir"], direct_server_with_data["url"]
    _bulk_history(root)
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            for viewport_index, (width, height, mobile) in enumerate([(1280, 850, False), (390, 844, True)]):
                context = browser.new_context(viewport={"width": width, "height": height},
                                              is_mobile=mobile, has_touch=mobile)
                page = context.new_page()
                try:
                    _open(page, url)
                    first = next(read["body"] for read in _reads(page) if read.get("body"))
                    assert first["page_cursor"] and first["has_more"]
                    assert first["window"]["complete"] is False
                    initial_id = next(row["history_id"] for row in first["messages"]
                                      if row.get("text") == "history-human-1800")
                    page.evaluate("() => { window.__historyFault = 'fail'; }")
                    _step(page, MAIN)
                    assert page.locator(f"{MAIN} .chat-load-older button").inner_text() == "Retry loading messages"
                    failed_cursor = _reads(page)[-1]["cursor"]
                    assert page.locator(f'{MAIN} [data-history-id="{initial_id}"]').count() == 1
                    _step(page, MAIN)
                    assert _reads(page)[-1]["cursor"] == failed_cursor

                    page.evaluate("() => { window.__historyFault = 'hold'; }")
                    page.locator(f"{MAIN} .chat-load-older button").evaluate("node => node.click()")
                    page.wait_for_function("() => Boolean(window.__heldHistory)")
                    live_text = f"LIVE_DURING_HISTORY_{browser_engine}_{width}"
                    for source in ("chat", "progress"):
                        source_path = root / "logs" / f"{source}.jsonl"
                        destination = root / "archive" / f"{source}_20260913T{viewport_index + 1:06d}.jsonl"
                        os.replace(source_path, destination)
                        _write(source_path, [])
                    live = _human(1900, text=live_text, direction="out", format="markdown",
                                  ts="2026-09-13T10:00:00Z")
                    _write(root / "logs" / "chat.jsonl", [live])
                    _emit_ws_frame(page, {"type": "chat", "role": "assistant", "chat_id": 1,
                                          "content": live_text, "ts": live["ts"]})
                    page.evaluate("() => window.__releaseHistory()")
                    _idle(page, MAIN)
                    assert page.locator(f"{MAIN} .message").filter(has_text=live_text).count() == 1
                    _to_beginning(page, MAIN)
                    assert page.locator(f"{MAIN} .chat-load-newer").count() == 0
                    page.locator(f"{MAIN} .message").filter(has_text="history-human-0000").wait_for(state="attached")
                    _assert_main_beginning_visible(page)
                    records = [row for read in _reads(page) if read.get("body", {}).get("messages")
                               for row in read["body"]["messages"]]
                    assert len({row["text"] for row in records if row.get("text", "").startswith("history-human-")}) == 1801
                    assert len({row["text"] for row in records if row.get("text", "").startswith("history-progress-")}) == 726
                    mounted = page.locator(f"{MAIN} [data-history-id]").evaluate_all(
                        "nodes => nodes.map(node => node.dataset.historyId)")
                    evidence = Path(os.environ.get("HISTORY_UI_EVIDENCE_DIR") or tmp_path / "history-ui-evidence")
                    evidence.mkdir(parents=True, exist_ok=True)
                    (evidence / f"archive-dom-{browser_engine}-{width}.json").write_text(json.dumps(
                        page.locator(f"{MAIN} [data-history-id]").evaluate_all("""nodes => nodes.map(node => ({
                            id: node.dataset.historyId, tag: node.tagName, className: node.className,
                            task: node.closest('.chat-live-card')?.dataset.taskId, text: node.textContent.slice(0, 180),
                        }))"""), indent=2), encoding="utf-8")
                    _screenshot(page, tmp_path, f"archive-beginning-{browser_engine}-{width}")
                    assert len(mounted) == len(set(mounted)), "physical source rows must not duplicate"
                    assert len(mounted) < 1200, "distant page bodies must leave the rendered window"
                    # At the beginning the control has left; the present is the floating
                    # control's: one read of the newest page, which lands at its end.
                    assert page.locator(f"{MAIN} .chat-load-older button").evaluate("node => node.hidden")
                    seen = len(_reads(page))
                    page.locator("#chat-scroll-bottom").click()
                    page.wait_for_function("n => window.__historyReads.length > n", arg=seen)
                    _idle(page, MAIN)
                    assert [read["cursor"] for read in _reads(page)[seen:]] == [None]
                    _assert_at_newest(page, MAIN, live_text)
                    assert page.locator(f"{MAIN} .chat-load-newer").count() == 0
                    assert page.locator(f"{MAIN} .message").filter(has_text=live_text).count() == 1
                    page.reload(wait_until="domcontentloaded")
                    _idle(page, MAIN)
                    assert page.locator(f"{MAIN} .message").filter(has_text=live_text).count() == 1
                finally:
                    context.close()
            project = _sparse_history(root)
            for width, height, mobile in [(1280, 850, False), (390, 844, True)]:
                context = browser.new_context(viewport={"width": width, "height": height},
                                              is_mobile=mobile, has_touch=mobile)
                page = context.new_page()
                try:
                    _open(page, url)
                    feed = _open_project(page, project)
                    _assert_at_newest(page, feed, "SPARSE_LATEST_MESSAGE")
                    _to_beginning(page, feed)
                    # Every row of this room is mounted, so no edge control may claim
                    # that something newer waits beyond the rendered transcript, and
                    # further edge scrolling must not rescan the foreign-room pages.
                    assert page.locator(f"{feed} .chat-load-newer").count() == 0
                    settled = len(_reads(page, project["chat_id"]))
                    for direction in ("newer", "newer", "older"):
                        page.locator(feed).evaluate(_EDGE_SCROLL, direction)
                        _idle(page, feed)
                    assert len(_reads(page, project["chat_id"])) == settled, "a settled sparse room must not refetch"
                    assert page.locator(f"{feed} .message").filter(has_text="SPARSE_FIRST_SAVED_MESSAGE").count() == 1
                    assert "OTHER_ROOM_ONLY" not in page.locator(feed).locator('..').inner_text()
                    # A page is counted in the room's own rows: the first read holds both of
                    # them, behind five archives of another room, and no page is read empty.
                    assert not any(read.get("cursor") for read in _reads(page, project["chat_id"]))
                    _screenshot(page, tmp_path, f"sparse-beginning-{browser_engine}-{width}")
                finally:
                    context.close()
        finally:
            browser.close()


def _feature_history(root):
    from ouroboros.artifacts import store_task_artifact_bytes
    from ouroboros.project_dialogue import append_chat_annotation
    from ouroboros.projects_registry import create_project

    project = create_project(root, "history-details", name="History detail room")
    destination = create_project(root, "history-destination", name="Routing destination room")
    cid = project["chat_id"]
    quiz = {"quiz_id": "saved-choice", "question": "How should the retained history read?",
            "options": [{"label": "First", "detail": "Keep the first form"},
                        {"label": "Second", "detail": "Keep the second form", "recommended": True}],
            "state": "open", "assumption": "Keep inspecting the archive"}
    comment = "Use my own words\nKeep both lines."
    image = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jWQAAAABJRU5ErkJggg==")
    image_name = f"chat-media-{hashlib.sha256(image).hexdigest()}.png"
    for name, content in [(image_name, image), ("history-note.txt", b"Persisted history document\n")]:
        store_task_artifact_bytes(root, "history-media", name, content)
    old = [
        _human(0, cid, text="FEATURE_FIRST_SAVED_MESSAGE"),
        _human(1, cid, direction="out", type="quiz", task_id="quiz-owner", text=quiz["question"], quiz=quiz),
        _human(2, cid, direction="out", type="photo", task_id="history-media", text="Archived image", caption="Archived image",
               mime="image/png", download_url=f"/api/tasks/history-media/artifacts/{image_name}"),
        _human(3, cid, direction="out", type="document", task_id="history-media", text="Archived document", caption="Archived document",
               filename="history-note.txt", mime="text/plain", size_bytes=27,
               download_url="/api/tasks/history-media/artifacts/history-note.txt"),
        _human(4, cid, text="Routed archive message", client_message_id="routed-archive"),
    ]
    _write(root / "archive" / "chat_20260901T000000.jsonl", old)
    # Only the conversation keeps Load more history: the narration's oldest rows
    # arrive with an older page of the conversation, before its first message.
    _write(root / "logs" / "chat.jsonl", [
        _human(5, cid, text="Routed to another project", client_message_id="routed-other"),
        *[_human(index, cid) for index in range(6, 1655)],
        _human(1656, cid, direction="system", type="quiz_answer", task_id="quiz-owner", text="",
               quiz={**quiz, "state": "answered", "comment": comment}),
        *[_human(1700 + index, cid, text=f"Following dialogue {index}", ts="2026-09-12T10:00:01Z")
          for index in range(20)]])
    append_chat_annotation(root, "routed-archive", action="route_to_project", status="delivered",
                           target="far-parent", target_label="History detail room", project_id=project["id"], project_chat_id=cid)
    append_chat_annotation(root, "routed-other", action="route_to_project", status="delivered",
                           target_label=destination["name"], project_id=destination["id"], project_chat_id=destination["chat_id"])
    child = {"task_id": "linked-child", "delegation_role": "subagent", "subagent_task_id": "linked-child",
             "parent_task_id": "far-parent", "root_task_id": "far-parent", "subagent_role": "Archive reader"}
    _write(root / "archive" / "progress_20260901T000000.jsonl", [
        _progress(0, cid, task_id="far-parent", content="FAR_PARENT_ORIGINAL_NARRATION"),
        _progress(1, cid, **child, subagent_event="progress", content="EARLY_CHILD_NARRATION"),
    ])
    for index in range(5):
        _write(root / "archive" / f"progress_20260902T00000{index}.jsonl", [
            _progress(2 + index * 130 + offset, cid) for offset in range(130)
        ])
    narration = "## Copyable heading\n\nParagraph after heading. " + "Reading preserved source. " * 24
    narration += "\n[Reference link](https://example.com/history-reference)"
    review = {"panels": [{"panel_id": "history-panel", "surface": "task_acceptance", "aggregate_signal": "PASS",
                           "transport_status": "success", "parse_status": "valid", "reason": "REVIEW_DETAIL_SENTINEL", "actors": []}]}
    child_result = "TERMINAL_CHILD_RESULT\n## Result heading\nResult paragraph. UNIQUE_CHILD_TAIL"
    _write(root / "logs" / "progress.jsonl", [
        _progress(700, cid, **child, subagent_event="completed", status="completed", content="Child finished",
                  result=child_result),
        _progress(701, cid, task_id="focus-root", content=narration, suggested_name="Selectable history title"),
    ])
    for task_id in ["far-parent", "linked-child", "focus-root", "history-media", "quiz-owner", *[f"history-task-{n}" for n in range(5)]]:
        _result(root, task_id, chat_id=cid, project_id=project["id"],
                **({"review_projection": review, "suggested_name": "Selectable history title"} if task_id == "focus-root" else {}))
    return project, destination, comment


@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
def test_history_details_selection_replay_and_project_reopen(direct_server_with_data, browser_engine, tmp_path, request):
    from playwright.sync_api import sync_playwright

    root, url = direct_server_with_data["data_dir"], direct_server_with_data["url"]
    project, destination, comment = _feature_history(root)
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            for width, height, mobile in [(1280, 900, False), (390, 844, True)]:
                context = browser.new_context(viewport={"width": width, "height": height},
                                              is_mobile=mobile, has_touch=mobile)
                context.route("https://example.com/history-reference", lambda route: route.fulfill(body="Reference opened"))
                page = context.new_page()
                try:
                    _open(page, url)
                    feed = _open_project(page, project)
                    card = page.locator(f'{feed} .chat-live-card[data-task-id="focus-root"]')
                    card.locator(':scope > [data-live-summary-button]').click()
                    line = card.locator(':scope > [data-live-timeline] .chat-live-line.expandable').filter(has_text="Copyable heading").first
                    toggle = line.locator('[data-live-line-toggle]')
                    title = line.locator('.chat-live-line-title')
                    title.scroll_into_view_if_needed()
                    box = title.bounding_box()
                    line_height = title.evaluate("node => parseFloat(getComputedStyle(node).lineHeight)")
                    page.mouse.move(box["x"] + 7, box["y"] + line_height / 2)
                    page.mouse.down()
                    page.mouse.move(box["x"] + min(box["width"] - 4, 100), box["y"] + line_height / 2, steps=10)
                    page.mouse.up()
                    try:
                        assert page.evaluate("() => getSelection().toString().length > 0")
                    except AssertionError:
                        from tests.ci_evidence import output_dir

                        evidence = output_dir(request.config) or Path(
                            os.environ.get("HISTORY_UI_EVIDENCE_DIR") or tmp_path / "history-ui-evidence")
                        _capture_selection_failure(page, title, evidence, browser_engine, box, line_height)
                        raise
                    assert line.get_attribute("data-expanded") == "0", "drag selection must not activate the title"
                    page.evaluate("() => getSelection().removeAllRanges()")
                    toggle.click()
                    assert line.get_attribute("data-expanded") == "1"
                    toggle.press("Enter")
                    assert line.get_attribute("data-expanded") == "0"
                    toggle.press(" ")
                    assert line.get_attribute("data-expanded") == "1"
                    assert line.locator('.chat-live-line-title br').count() > 0
                    copied = title.evaluate("""node => {
                        const range = document.createRange(); range.selectNodeContents(node);
                        const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range);
                        const copied = selection.toString(); selection.removeAllRanges(); return copied;
                    }""")
                    assert "Copyable heading\n" in copied and "Paragraph after heading" in copied
                    assert copied.count("Reading preserved source.") == 24
                    assert copied.rstrip().endswith("Reference link"), "copy must reach the full narration tail"
                    with page.expect_popup() as popup:
                        line.get_by_role("link", name="Reference link").click()
                    popup.value.wait_for_load_state()
                    popup.value.close()
                    assert line.get_attribute("data-expanded") == "1", "nested link must not toggle its title owner"

                    card.locator('[data-review-section-toggle]').click()
                    card.locator('[data-review-group-toggle]').click()
                    card.locator('[data-review-attempt-toggle]').first.click()
                    detail = card.locator('[data-review-attempt-detail]').first
                    assert "REVIEW_DETAIL_SENTINEL" in detail.inner_text()
                    page.evaluate("""({feed, lineKey}) => {
                        const root = document.querySelector(feed);
                        const line = root.querySelector(`[data-live-line-key="${lineKey}"]`);
                        const card = line.closest('.chat-live-card');
                        const detail = card.querySelector('[data-review-attempt-detail]');
                        const focus = card.querySelector('[data-review-attempt-toggle]');
                        focus.focus({preventScroll:true});
                        root.scrollTop += detail.getBoundingClientRect().top - root.getBoundingClientRect().top - 80;
                        const range = document.createRange(); range.selectNodeContents(line.querySelector('.chat-live-line-title'));
                        const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range);
                        window.__historyKept = {line, detail, focus, selection: selection.toString(),
                            remaining: root.scrollHeight - root.scrollTop - root.clientHeight,
                            top: detail.getBoundingClientRect().top - root.getBoundingClientRect().top};
                    }""", {"feed": feed, "lineKey": line.get_attribute("data-live-line-key")})
                    assert page.evaluate("() => window.__historyKept.remaining > 100"), "fixture must place the review away from bottom-follow"
                    _screenshot(page, tmp_path, f"review-before-prepend-{browser_engine}-{width}")
                    _step(page, feed)
                    kept = page.evaluate("""feed => {
                        const root = document.querySelector(feed), old = window.__historyKept;
                        return {line: root.contains(old.line), detail: root.contains(old.detail),
                            focused: document.activeElement === old.focus,
                            selection: getSelection().toString() === old.selection,
                            drift: Math.abs(old.detail.getBoundingClientRect().top - root.getBoundingClientRect().top - old.top)};
                    }""", feed)
                    _screenshot(page, tmp_path, f"review-after-prepend-{browser_engine}-{width}")
                    assert kept["line"] and kept["detail"] and kept["focused"] and kept["selection"], kept
                    assert kept["drift"] <= 6, kept
                    _screenshot(page, tmp_path, f"selected-review-{browser_engine}-{width}")
                    page.evaluate("() => { getSelection().removeAllRanges(); document.activeElement?.blur(); }")
                    _to_beginning(page, feed)
                    child = page.locator(f'{feed} .chat-live-card[data-task-id="linked-child"]')
                    assert child.get_attribute("data-finished") == "1"
                    if child.get_attribute("data-expanded") != "1":
                        child.locator(':scope > [data-live-summary-button]').click()
                    assert "EARLY_CHILD_NARRATION" in child.inner_text()
                    assert "TERMINAL_CHILD_RESULT" in child.inner_text()
                    assert child.locator(':scope > [data-live-summary-button] [data-live-phase]').inner_text() == "Done"
                    evidence = Path(os.environ.get("HISTORY_UI_EVIDENCE_DIR") or tmp_path / "history-ui-evidence")
                    evidence.mkdir(parents=True, exist_ok=True)
                    (evidence / f"terminal-child-{browser_engine}-{width}.html").write_text(child.evaluate("node => node.outerHTML"), encoding="utf-8")
                    _screenshot(page, tmp_path, f"terminal-child-{browser_engine}-{width}")
                    result_line = child.locator(':scope > [data-live-timeline] .chat-live-line').filter(has_text="TERMINAL_CHILD_RESULT").first
                    assert result_line.count() == 1, "the full terminal result must remain accessible in the child timeline"
                    if result_line.locator('[data-live-line-toggle]').count() and result_line.get_attribute("data-expanded") != "1":
                        result_line.locator('[data-live-line-toggle]').click()
                    result_copy = result_line.locator('.chat-live-line-body').evaluate("""node => {
                        const range = document.createRange(); range.selectNodeContents(node);
                        const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range);
                        const copied = selection.toString(); selection.removeAllRanges(); return copied;
                    }""")
                    assert "[RESULT]" in result_copy and "TERMINAL_CHILD_RESULT" in result_copy
                    assert "Result heading\n" in result_copy and "UNIQUE_CHILD_TAIL" in result_copy
                    quiz = page.locator(f'{feed} [data-quiz-id="saved-choice"]')
                    assert quiz.get_attribute("data-state") == "answered"
                    assert quiz.locator('.chat-quiz-answer').text_content() == f"Owner's answer: {comment}"
                    assert quiz.locator('.chat-quiz-option').count() == 2
                    assert quiz.locator('.chat-quiz-option.chosen').count() == 0
                    assert quiz.locator('.chat-quiz-option-recommended').count() == 1
                    image = page.locator(f'{feed} img')
                    page.wait_for_function("feed => [...document.querySelectorAll(`${feed} img`)].some(img => img.complete && img.naturalWidth > 0)", arg=feed)
                    assert image.count() > 0
                    assert page.locator(feed).get_by_text("history-note.txt", exact=True).count() > 0
                    anchor = page.locator(f'{feed} [data-client-message-id="routed-archive"]')
                    assert anchor.locator('.msg-routing-annotation').text_content() == "Routed to project · History detail room"
                    assert anchor.locator('.msg-routing-actions').count() == 0
                    other = page.locator(f'{feed} [data-client-message-id="routed-other"]')
                    # The routing receipt's button lives in the shared action row between the note and the
                    # timestamp (never inside the nowrap note line): DESIGN "Quiz card", ARCHITECTURE 03.
                    assert other.locator('.msg-routing-actions [data-intent="open-project"]').count() == 1
                    assert other.locator('.msg-routing-annotation').get_by_role("button").count() == 0
                    assert other.evaluate("""node => {
                        const note = node.querySelector('.msg-routing-annotation'), row = node.querySelector('.msg-routing-actions'),
                            time = node.querySelector('.msg-time');
                        const follows = (a, b) => (a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING) !== 0;
                        return Boolean(note && row && time) && follows(note, row) && follows(row, time);
                    }""")
                    # Live receipt updates use the same room address as retained history.
                    for routed_project in (project, destination):
                        _emit_ws_frame(page, {"type": "message_annotation", "annotation_type": "routing_ack",
                            "chat_id": project["chat_id"], "client_message_id": "routed-other",
                            "action": "route_to_project", "status": "delivered", "target_label": routed_project["name"],
                            "project_id": routed_project["id"], "project_chat_id": str(routed_project["chat_id"])})
                        assert other.locator('.msg-routing-annotation').text_content() == f"Routed to project · {routed_project['name']}"
                        assert other.locator('.msg-routing-actions').count() == int(routed_project is destination)
                    anchor.scroll_into_view_if_needed()
                    _screenshot(page, tmp_path, f"archive-features-{browser_engine}-{width}")
                    other.scroll_into_view_if_needed()
                    _screenshot(page, tmp_path, f"routing-other-project-{browser_engine}-{width}")
                    other.locator('.msg-routing-actions [data-intent="open-project"]').click()
                    destination_feed = f'#pchat-{destination["id"]}-messages'
                    page.locator(destination_feed).wait_for(state="visible", timeout=30_000)
                    _idle(page, destination_feed)
                    _screenshot(page, tmp_path, f"routing-destination-opened-{browser_engine}-{width}")
                    # Opened again after reading back to its first message, the room is at its
                    # newest message and reads its newest page only (owner decision 2026-10-05).
                    page.locator('#project-panel-close').click()
                    seen = len(page.evaluate("() => window.__historyReads"))
                    reopened = _open_project(page, project)
                    assert not [read["cursor"] for read in page.evaluate("n => window.__historyReads.slice(n)", seen)
                                if read["cursor"]], "no older page is read again"
                    _assert_at_newest(page, reopened, "Following dialogue 19")
                    _screenshot(page, tmp_path, f"reopened-at-newest-{browser_engine}-{width}")
                finally:
                    context.close()
        finally:
            browser.close()


QUIET_ASK = "QUIET_ORIGIN_REQUEST: plan the archive migration"


def _rooms_behind_other_traffic(root):
    """Two Projects whose rows lie only in OLD archives, behind other rooms' traffic.

    ``history-quiet`` holds the owner's request that started it (a retained origin)
    and 12 messages in the oldest archive, behind eleven archives of other rooms;
    ``history-deep`` holds 400 messages (more than two pages) in two archives of its
    own, five archives of another room between and after them.
    """
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.projects_registry import bind_task_to_project, create_project

    quiet = create_project(root, "history-quiet", name="Quiet archived room")
    deep = create_project(root, "history-deep", name="Deep archived room")
    busy = create_project(root, "history-busy", name="Busy other room")
    asked = "2026-08-01T09:00:00Z"
    bind_task_to_project(root, "quiet-root", quiet["id"], quiet["chat_id"], origin={
        "ref": build_owner_message_ref(chat_id=1, client_message_id="quiet-ask", ts=asked, text=QUIET_ASK),
        "text": QUIET_ASK})
    at = lambda day, index: f"2026-08-{day:02d}T{10 + index // 60:02d}:{index % 60:02d}:00Z"  # noqa: E731

    def busy_archives(stamp):
        for segment in range(5):
            _write(root / "archive" / f"chat_{stamp}{segment}.jsonl", [
                _human(index, busy["chat_id"], text="OTHER_ROOM_ONLY " + "x" * 2000) for index in range(350)])

    _write(root / "archive" / "chat_20260801T000000.jsonl", [
        _human(0, ts=asked, client_message_id="quiet-ask", text=QUIET_ASK),
        *[_human(index, quiet["chat_id"], ts=at(1, index), direction="out" if index % 2 else "in",
                 text=f"QUIET_ROW_{index:02d}") for index in range(12)],
        *[_human(index, deep["chat_id"], ts=at(2, index), text=f"DEEP_ROW_{index:04d}") for index in range(200)],
    ])
    busy_archives("20260803T00000")
    _write(root / "archive" / "chat_20260804T000000.jsonl", [
        _human(index, deep["chat_id"], ts=at(4, index - 200), text=f"DEEP_ROW_{index:04d}")
        for index in range(200, 400)])
    busy_archives("20260805T00000")
    _write(root / "logs" / "chat.jsonl", [
        _human(index, busy["chat_id"], text=f"OTHER_ROOM_LIVE {index}") for index in range(20)])
    return quiet, deep


@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
def test_a_room_behind_other_rooms_archives_opens_at_its_newest_message_and_pages_only_older(
    direct_server_with_data, browser_engine, tmp_path,
):
    """The owner's complaint (2026-10-04): a Project quiet for a while opened with nothing but
    its retained origin, its own messages cut off by other rooms' newer archives. A room's
    pages are its own rows (owner decisions 2026-10-05): it opens at its newest message
    without a press, complete when its whole history fits, and `Load more history` only
    ever lands the next OLDER rows above, by the previous page's cursor, until it leaves."""
    import re

    from playwright.sync_api import sync_playwright

    root, url = direct_server_with_data["data_dir"], direct_server_with_data["url"]
    quiet, deep = _rooms_behind_other_traffic(root)
    texts = "nodes => nodes.map(node => node.textContent)"

    def deep_rows(feed):
        found = (re.search(r"DEEP_ROW_(\d{4})", text) for text in
                 page.locator(f"{feed} [data-history-id]").evaluate_all(texts))
        return [int(match.group(1)) for match in found if match]

    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            for width, height, mobile in [(1280, 850, False), (390, 844, True)]:
                context = browser.new_context(viewport={"width": width, "height": height},
                                              is_mobile=mobile, has_touch=mobile)
                page = context.new_page()
                try:
                    _open(page, url)
                    feed = _open_project(page, quiet)
                    _assert_at_newest(page, feed, "QUIET_ROW_11")
                    reads = _reads(page, quiet["chat_id"])
                    assert reads and not any(read["cursor"] for read in reads), "opened without a press"
                    body = reads[-1]["body"]
                    assert body["window"]["complete"] is True and body["has_more"] is False, body["window"]
                    assert not any(row.get("origin_projected") for row in body["messages"]), \
                        "the request's own saved row adopts the retained origin"
                    shown = [text for text in page.locator(f"{feed} [data-history-id]").evaluate_all(texts)
                             if "QUIET_" in text]
                    assert len(shown) == 13 and QUIET_ASK in shown[0] and "QUIET_ROW_11" in shown[-1], shown
                    assert page.locator(f"{feed} .saved-project-context").count() == 0
                    panel = page.locator(feed).locator("..").inner_text()
                    assert "Some saved history is not loaded" not in panel and "OTHER_ROOM_ONLY" not in panel
                    assert page.locator(f"{feed} .chat-load-older-note").inner_text() == "Beginning of saved history"
                    assert page.locator(f"{feed} .chat-load-older button").evaluate("node => node.hidden")
                    _screenshot(page, tmp_path, f"quiet-room-newest-{browser_engine}-{width}")

                    page.locator("#project-panel-close").click()
                    feed = _open_project(page, deep)
                    _assert_at_newest(page, feed, "DEEP_ROW_0399")
                    reads = _reads(page, deep["chat_id"])
                    assert reads and not any(read["cursor"] for read in reads), "opened without a press"
                    newest = reads[0]["body"]  # the first recent read owns the paging chain
                    shown = deep_rows(feed)
                    assert newest["has_more"] is True and shown == list(range(250, 400)), shown[:3]
                    button = page.locator(f"{feed} .chat-load-older button")
                    cursor, presses = newest["next_cursor"], 0
                    while not button.evaluate("node => node.hidden"):
                        assert presses < 4, "the room's 400 messages are three pages"
                        # The reader scrolls up to the control and presses it.
                        page.locator(feed).evaluate(
                            "n => { n.dispatchEvent(new WheelEvent('wheel', {deltaY: -1})); n.scrollTop = 160; }")
                        page.evaluate(_FRAMES)
                        button.scroll_into_view_if_needed()
                        page.evaluate(_FRAMES)
                        first = page.locator(f"{feed} [data-history-id]").filter(has_text=f"DEEP_ROW_{shown[0]:04d}")
                        place, seen = first.evaluate(_ROW_OFFSET), len(_reads(page, deep["chat_id"]))
                        button.click()
                        page.wait_for_function("([id, n]) => window.__historyReads.filter(r => r.chatId === id).length > n",
                                               arg=[deep["chat_id"], seen])
                        _idle(page, feed)
                        presses += 1
                        pressed = _reads(page, deep["chat_id"])[seen:]
                        assert [read["cursor"] for read in pressed] == [cursor], \
                            "one press reads the next older page, by the cursor the previous page gave"
                        assert pressed[0]["body"]["messages"], "no press lands nothing"
                        now = deep_rows(feed)
                        added = now[:len(now) - len(shown)]
                        assert added and now[len(added):] == shown and added == sorted(added) and added[-1] < shown[0], \
                            ("older rows land above the rows already shown", added[:3], shown[:3])
                        assert abs(first.evaluate(_ROW_OFFSET) - place) <= 8, "the reader keeps their place"
                        shown, cursor = now, pressed[0]["body"].get("next_cursor")
                    assert presses == 2 and shown == list(range(400)), (presses, shown[:3])
                    assert all(read["cursor"] != newest["page_cursor"] for read in _reads(page, deep["chat_id"])), \
                        "the newest page is never read again"
                    assert page.locator(f"{feed} .chat-load-older-note").inner_text() == "Beginning of saved history"
                    _screenshot(page, tmp_path, f"deep-room-beginning-{browser_engine}-{width}")
                finally:
                    context.close()
        finally:
            browser.close()
