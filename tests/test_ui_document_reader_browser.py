"""Real delivery -> chat file card -> document reader, live and after a history reload.

The mock model calls the real ``send_file`` tool, so every file reaches Chat through
the supervisor document handler exactly as the owner receives it: inline bytes in the
live frame, the immutable task-artifact route after a reload. Main and a Project room
are both driven, in Chromium and WebKit at the desktop client size (1473x978) and a
phone (390x844), in light and dark appearance. The 409 and 503 answers are Playwright
route stubs (labelled where used); every other answer is the real server's."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from devtools.benchmarks.common.server_runner import _api
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401 - pytest fixture import
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET

LIMIT = 1024 * 1024
WIDE, NARROW = {"width": 1473, "height": 978}, {"width": 390, "height": 844}
TABLE = ("| " + " | ".join(f"Measured column {index}" for index in range(12)) + " |\n"
         + "|" + "---|" * 12 + "\n" + "| " + " | ".join(f"value-{index}-with-length" for index in range(12)) + " |\n")
BRIEF = (
    "# Delivery brief\n\n"
    "The first line of a paragraph\ncontinues on the next line in the same paragraph.\n"
    "An explicit break ends here  \nand this line follows it.\n\n"
    '<script>window.__readerXss = 1</script><img src="/reader-probe/raw.png" onerror="window.__readerXss = 2">\n\n'
    "![Remote chart](https://example.invalid/reader-probe/remote.png) and ![Local chart](./reader-probe/local.png)\n\n"
    "[Sibling note](./sibling.md) and [Project site](https://example.com/)\n\n"
    '<svg onload="window.__readerXss = 3"><circle r="4"></circle></svg>\n\n'
    + TABLE + "\n```python\nprint('" + "a very long line of code " * 12 + "')\n```\n\n"
    + "".join(f"## Section {index}\n\nParagraph {index} keeps the reader long enough to scroll.\n\n" for index in range(30))
)
NOTES = "Plain notes\nline two\twith a tab\nПривет, мир\n" + "long-line " * 40 + "\n"
# 1 ASCII byte, then two-byte characters: the 1 MiB cap lands inside a character.
BIG = "x" + "й" * 800_000
FILES = {
    "brief.md": BRIEF.encode(), "notes.txt": NOTES.encode(), "big.md": BIG.encode(),
    "broken.txt": b"valid start \xff\xfe invalid middle\n", "binary.txt": b"text\x00\x01\x02 binary",
    "page.html": b"<h1>not read here</h1>", "tampered.md": b"# Tampered\n\nOriginal delivered bytes.\n",
}

READER_STATE = """() => {
    const dialog = document.querySelector('dialog.document-reader[open]');
    if (!dialog) return null;
    const body = dialog.querySelector('.document-reader-body');
    const md = body.querySelector('.document-reader-markdown');
    const rect = dialog.getBoundingClientRect();
    return {
        title: dialog.querySelector('.document-reader-title').textContent,
        meta: dialog.querySelector('.document-reader-meta').textContent,
        notice: dialog.querySelector('.document-reader-notice').hidden ? '' : dialog.querySelector('.document-reader-notice').textContent,
        status: body.querySelector('.document-reader-status')?.textContent || '',
        retry: Boolean(body.querySelector('[data-reader-action="retry"]')),
        views: !dialog.querySelector('.document-reader-views').hidden,
        markdown: md ? md.innerText : null,
        source: body.querySelector('.document-reader-source')?.textContent ?? null,
        paragraphs: md ? [...md.querySelectorAll('p')].map((p) => ({ text: p.textContent, breaks: p.querySelectorAll('br').length })) : [],
        active: document.activeElement === body,
        activeInside: dialog.contains(document.activeElement),
        active_elements: md ? md.querySelectorAll('script, img, iframe, object, embed, svg, video, audio').length : 0,
        links: md ? [...md.querySelectorAll('a')].map((a) => [a.textContent, a.getAttribute('href')]) : [],
        tables: md ? [...md.querySelectorAll('.md-table-wrap')].map((w) => w.scrollWidth > w.clientWidth + 1) : [],
        bodyOverflowX: body.scrollWidth - body.clientWidth,
        scroll: [body.scrollTop, body.scrollHeight, body.clientHeight],
        rect: [rect.left, rect.top, rect.width, rect.height],
        viewport: [innerWidth, innerHeight],
        xss: window.__readerXss ?? null,
    };
}"""


def _send_files_mock(paths, calls):
    state = {"sent": False}

    def handler(request_handler):
        request = json.loads(request_handler.rfile.read(int(request_handler.headers.get("content-length") or 0)))
        names = {tool.get("function", {}).get("name") for tool in request.get("tools") or []}
        message = {"role": "assistant", "content": "The files are delivered."}
        if not state["sent"] and "send_file" in names:
            state["sent"] = True
            message = {"role": "assistant", "content": "", "tool_calls": [{
                "id": f"reader-file-{index}", "type": "function", "function": {
                    "name": "send_file", "arguments": json.dumps({"file_path": str(path), "caption": ""}),
                },
            } for index, path in enumerate(paths)]}
            calls.extend(path.name for path in paths)
        finish = "tool_calls" if message.get("tool_calls") else "stop"
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        common = {"id": "mock-reader", "model": request.get("model") or "mock-model"}
        if request.get("stream"):
            delta = dict(message)
            if delta.get("tool_calls"):
                delta["tool_calls"] = [dict(call, index=index) for index, call in enumerate(delta["tool_calls"])]
            frames = [{**common, "object": "chat.completion.chunk",
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]},
                      {**common, "object": "chat.completion.chunk", "choices": [], "usage": usage}]
            data = ("".join("data: " + json.dumps(frame) + "\n\n" for frame in frames) + "data: [DONE]\n\n").encode()
            content_type = "text/event-stream"
        else:
            data = json.dumps({**common, "object": "chat.completion", "usage": usage,
                               "choices": [{"index": 0, "message": message, "finish_reason": finish}]}).encode()
            content_type = "application/json"
        request_handler.send_response(200)
        request_handler.send_header("Content-Type", content_type)
        request_handler.send_header("Content-Length", str(len(data)))
        request_handler.end_headers()
        request_handler.wfile.write(data)

    return handler


def _reader(page):
    return page.evaluate(READER_STATE)


def _open(page, name, *, ready=".document-reader-markdown, .document-reader-source, .document-reader-status:not(.is-loading)",
          feed="#chat-messages"):
    card = page.locator(f"{feed} .chat-file-card").filter(has_text=name)
    # Playwright may scroll again to avoid fixed chrome before dispatching
    # the click. Measure the actual opening gesture, before the reader handler.
    card.evaluate("""(node, feed) => node.addEventListener('click', () => {
        node.__readerOpeningScroll = document.querySelector(feed).scrollTop;
    }, {capture: true, once: true})""", feed)
    card.click()
    page.locator("dialog.document-reader[open]").wait_for(state="visible")
    page.locator(f"dialog.document-reader[open] :is({ready})").first.wait_for(state="attached")
    state = _reader(page)
    state["opening_scroll"] = card.evaluate("node => node.__readerOpeningScroll")
    return state


def _close_with_escape(page):
    page.keyboard.press("Escape")
    page.locator("dialog.document-reader").wait_for(state="detached")


def _feed_scroll(page, feed="#chat-messages"):
    """The conversation's scroll offset once layout has settled (two equal reads)."""
    read = "feed => { const node = document.querySelector(feed); return [node.scrollTop, node.scrollHeight]; }"
    value = page.evaluate(read, feed)
    for _ in range(20):
        page.wait_for_timeout(150)
        latest = page.evaluate(read, feed)
        if latest == value:
            return value[0]
        value = latest
    return value[0]


def _check_brief(state, title="brief.md"):
    assert state["title"] == title and state["meta"].startswith("MD · ")
    joined = [paragraph for paragraph in state["paragraphs"] if paragraph["text"].startswith("The first line")]
    # Document semantics: one paragraph, soft newlines joined, only the explicit hard break kept.
    assert len(joined) == 1 and joined[0]["breaks"] == 1, state["paragraphs"][:3]
    assert "continues on the next line in the same paragraph" in joined[0]["text"]
    # Raw HTML stays literal text, images stay references, nothing active is mounted.
    assert "<script>window.__readerXss = 1</script>" in state["markdown"]
    assert "Image: Remote chart" in state["markdown"] and "Image: Local chart" in state["markdown"]
    assert state["active_elements"] == 0 and state["xss"] is None
    links = dict(state["links"])
    assert links.get("Sibling note") is None, "a relative link is not resolved against any folder"
    assert links.get("Project site") == "https://example.com/"
    assert state["views"] is True and state["bodyOverflowX"] <= 1


@pytest.mark.ui_browser
@pytest.mark.serial
def test_delivered_documents_open_in_the_reader_live_and_after_reload(direct_server_with_data, monkeypatch, tmp_path):  # noqa: F811
    from playwright.sync_api import sync_playwright
    from ouroboros.artifacts import task_artifact_dir_path
    from tests import fixtures_mock_llm

    root = direct_server_with_data["data_dir"]
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path / "evidence")))
    evidence.mkdir(parents=True, exist_ok=True)
    source = root / "reader-src"
    source.mkdir()
    paths = []
    for name, data in FILES.items():
        (source / name).write_bytes(data)
        paths.append(source / name)
    calls = []
    monkeypatch.setattr(fixtures_mock_llm._Handler, "do_POST", _send_files_mock(paths, calls))
    url = direct_server_with_data["url"]
    with sync_playwright() as playwright:
        chromium = playwright.chromium.launch()
        try:
            page = chromium.new_page(viewport=WIDE, color_scheme="dark")
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
            frames, requests, failed = {}, [], []

            def capture(payload):
                value = json.loads(payload) if isinstance(payload, str) and '"document"' in payload else {}
                if value.get("type") == "document":
                    frames.setdefault(value["filename"], value)

            page.on("websocket", lambda socket: socket.on("framereceived", capture))
            page.on("request", lambda request: requests.append((request.url, request.headers.get("range"))))
            page.on("requestfailed", lambda request: failed.append(request.url))
            page.goto(url, wait_until="domcontentloaded")
            page.locator("#chat-input").wait_for(state="visible")
            page.wait_for_function("() => window.__testSockets?.some(socket => socket.readyState === WebSocket.OPEN)")
            page.locator("#chat-input").fill("Send me the documents.")
            page.locator("#chat-send").click()
            for name in FILES:
                page.locator(".chat-file-card").filter(has_text=name).wait_for(state="visible", timeout=60_000)
            assert sorted(calls) == sorted(FILES) and sorted(frames) == sorted(FILES)
            assert all(frame["file_base64"] and frame["download_url"].startswith("/api/tasks/") for frame in frames.values())
            artifact = {name: url + frame["download_url"] for name, frame in frames.items()}
            readable = {name: page.locator(".chat-file-card").filter(has_text=name).locator(".chat-file-more").inner_text()
                        for name in FILES}
            assert readable == {name: ("•••" if name == "page.html" else "Read") for name in FILES}

            # A delivered frame precedes the model's final answer. Wait for that
            # answer before measuring a fixed scroll offset: while following the
            # live edge, its arrival legitimately moves the underlying feed.
            page.locator("#chat-messages .chat-bubble.assistant").filter(
                has_text="The files are delivered.").wait_for(timeout=60_000)
            page.locator('#chat-messages .chat-live-card [data-phase="done"]').wait_for()
            # Still live: inline delivered bytes, no history reload or file request.
            requests.clear()
            state = _open(page, "brief.md", ready=".document-reader-markdown")
            _check_brief(state)
            assert state["active"], "the reading region takes focus so keys scroll it at once"
            live_markdown = state["markdown"]
            page.wait_for_timeout(300)
            assert not [entry for entry in requests if "/artifacts/" in entry[0] or "/api/files/" in entry[0]
                        or "reader-probe" in entry[0] or "example.com" in entry[0]], requests
            page.screenshot(path=str(evidence / "chromium-wide-dark-brief.png"))
            ring = "() => getComputedStyle(document.querySelector('.document-reader-body')).outlineStyle"
            assert page.evaluate(ring) == "none", "a pointer open shows no ring on the reading region"
            page.keyboard.press("PageDown")
            page.wait_for_function("() => document.querySelector('.document-reader-body').scrollTop > 0")
            assert page.evaluate(ring) == "solid", "the first key shows the one focus ring"
            for _ in range(8):
                page.keyboard.press("Tab")
                assert _reader(page)["activeInside"], "Tab stays inside the reader"
            page.locator('[data-reader-view="source"]').click()
            source_state = _reader(page)
            assert source_state["source"] == BRIEF and source_state["markdown"] is None
            assert page.locator(".document-reader-source").get_attribute("contenteditable") is None
            page.locator('[data-reader-view="formatted"]').click()
            _close_with_escape(page)
            assert page.evaluate("() => document.activeElement?.closest('.chat-file-card')?.textContent.includes('brief.md')")
            assert abs(_feed_scroll(page) - state["opening_scroll"]) <= 1, "closing returns to the same place in the conversation"

            state = _open(page, "notes.txt", ready=".document-reader-source")
            assert state["source"] == NOTES and not state["views"] and state["meta"].startswith("TXT · ")
            page.locator('dialog.document-reader [data-reader-action="close"]').click()
            page.locator("dialog.document-reader").wait_for(state="detached")

            state = _open(page, "big.md", ready=".document-reader-markdown")
            assert "Showing the first 1.0 MB of 1.5 MB" in state["notice"] and "UTF-8" not in state["notice"]
            assert "�" not in state["markdown"]
            _close_with_escape(page)
            state = _open(page, "broken.txt", ready=".document-reader-source")
            assert "not valid UTF-8" in state["notice"] and "�" in state["source"]
            _close_with_escape(page)
            state = _open(page, "binary.txt", ready=".document-reader-status:not(.is-loading)")
            assert "does not look like text" in state["status"] and state["source"] is None
            _close_with_escape(page)
            page.locator(".chat-file-card").filter(has_text="page.html").click()
            page.locator(".chat-file-dialog[open]").wait_for(state="visible")
            assert page.locator("dialog.document-reader").count() == 0, "HTML keeps the file dialog"
            page.screenshot(path=str(evidence / "chromium-wide-dark-file-dialog.png"))
            page.locator('.chat-file-dialog[open] [data-file-action="close"]').click()

            # The stored copy of one delivery is damaged on disk after it was sent.
            tampered = frames["tampered.md"]
            stored = task_artifact_dir_path(root, tampered["task_id"]) / tampered["file_ref"]["path"]
            stored.chmod(0o644)
            stored.write_bytes(b"# Tampered\n\nChanged after delivery!!!\n")

            # Replay: the same documents now come from the immutable artifact route.
            page.reload(wait_until="domcontentloaded")
            page.locator(".chat-file-card").filter(has_text="brief.md").wait_for(state="visible", timeout=30_000)
            requests.clear()
            state = _open(page, "brief.md", ready=".document-reader-markdown")
            _check_brief(state)
            assert state["markdown"] == live_markdown, "replay shows the same delivered document"
            assert (artifact["brief.md"], None) in requests, "a known small file is read whole"
            _close_with_escape(page)
            requests.clear()
            state = _open(page, "big.md", ready=".document-reader-markdown")
            assert (artifact["big.md"], f"bytes=0-{LIMIT - 1}") in requests
            assert "Showing the first 1.0 MB of 1.5 MB" in state["notice"] and "UTF-8" not in state["notice"]
            _close_with_escape(page)
            requests.clear()
            state = _open(page, "tampered.md", ready=".document-reader-status:not(.is-loading)")
            assert "did not pass its integrity check" in state["status"] and not state["retry"]
            assert not [entry for entry in requests if "/api/files/" in entry[0]], "no fallback to a current file path"
            _close_with_escape(page)

            # Platform refusal (#1297) offers Retry; a changed row does not. Both answers are
            # Playwright route stubs: the real 404 is the tampered copy above.
            page.route(artifact["notes.txt"], lambda route: route.fulfill(status=503, json={
                "error": "artifact could not be read", "reason_code": "artifact_unavailable"}))
            state = _open(page, "notes.txt", ready=".document-reader-status:not(.is-loading)")
            assert "cannot read the delivered copy right now" in state["status"] and state["retry"]
            page.unroute(artifact["notes.txt"])
            page.locator('[data-reader-action="retry"]').click()
            page.locator("dialog.document-reader .document-reader-source").wait_for(state="attached")
            assert _reader(page)["source"] == NOTES
            _close_with_escape(page)
            page.route(artifact["notes.txt"], lambda route: route.fulfill(status=409, json={
                "error": "changed", "reason_code": "artifact_identity_changed"}))
            state = _open(page, "notes.txt", ready=".document-reader-status:not(.is-loading)")
            assert "changed after it was delivered" in state["status"] and not state["retry"]
            page.unroute(artifact["notes.txt"])
            _close_with_escape(page)

            # Rapid close: the held read is aborted and its late answer paints nothing.
            held = []
            page.route("**/artifacts/**", lambda route: held.append(route))
            page.locator(".chat-file-card").filter(has_text="brief.md").click()
            page.locator("dialog.document-reader .document-reader-status.is-loading").wait_for(state="attached")
            _close_with_escape(page)
            page.locator(".chat-file-card").filter(has_text="notes.txt").click()
            page.wait_for_function("() => document.querySelector('dialog.document-reader .document-reader-status.is-loading')")
            for _ in range(40):
                if artifact["brief.md"] in failed:
                    break
                page.wait_for_timeout(50)
            assert artifact["brief.md"] in failed, "closing aborted the pending read"
            assert _reader(page)["title"] == "notes.txt"
            for route in held:
                try:
                    route.continue_()
                except Exception:  # the aborted request has no route left to continue
                    pass
            page.unroute("**/artifacts/**")
            page.locator("dialog.document-reader .document-reader-source").wait_for(state="attached")
            state = _reader(page)
            assert state["title"] == "notes.txt" and state["source"] == NOTES and state["markdown"] is None
            _close_with_escape(page)

            # Phone: a full sheet in light appearance; Close returns to the same place.
            page.set_viewport_size(NARROW)
            page.emulate_media(color_scheme="light")
            state = _open(page, "brief.md", ready=".document-reader-markdown")
            assert [round(value) for value in state["rect"]] == [0, 0, NARROW["width"], NARROW["height"]], state["rect"]
            assert any(state["tables"]) and state["bodyOverflowX"] <= 1, "only the table scrolls sideways"
            close = page.locator('dialog.document-reader [data-reader-action="close"]').bounding_box()
            download = page.locator('dialog.document-reader [data-reader-action="download"]').bounding_box()
            assert close and close["x"] + close["width"] <= NARROW["width"] and close["y"] >= 0
            # Close stays beside the title; the other actions take one row below it.
            assert download["x"] + download["width"] <= NARROW["width"] and close["y"] < download["y"]
            page.screenshot(path=str(evidence / "chromium-narrow-light-brief.png"))
            page.locator('dialog.document-reader [data-reader-action="close"]').click()
            page.locator("dialog.document-reader").wait_for(state="detached")
            assert abs(_feed_scroll(page) - state["opening_scroll"]) <= 1
        finally:
            chromium.close()

        webkit = playwright.webkit.launch()
        try:
            page = webkit.new_page(viewport=WIDE, color_scheme="light")
            requests = []
            page.on("request", lambda request: requests.append((request.url, request.headers.get("range"))))
            page.goto(url, wait_until="domcontentloaded")
            page.locator(".chat-file-card").filter(has_text="brief.md").wait_for(state="visible", timeout=30_000)
            state = _open(page, "brief.md", ready=".document-reader-markdown")
            _check_brief(state)
            assert state["active"]
            assert page.evaluate("() => getComputedStyle(document.querySelector('.document-reader-body')).outlineStyle") == "none"
            page.screenshot(path=str(evidence / "webkit-wide-light-brief.png"))
            _close_with_escape(page)
            assert page.evaluate("() => document.activeElement?.closest('.chat-file-card')?.textContent.includes('brief.md')")
            assert abs(_feed_scroll(page) - state["opening_scroll"]) <= 1
            state = _open(page, "big.md", ready=".document-reader-markdown")
            assert "Showing the first 1.0 MB of 1.5 MB" in state["notice"] and "UTF-8" not in state["notice"]
            _close_with_escape(page)
            page.set_viewport_size(NARROW)
            page.emulate_media(color_scheme="dark")
            state = _open(page, "notes.txt", ready=".document-reader-source")
            assert state["source"] == NOTES
            assert [round(value) for value in state["rect"]] == [0, 0, NARROW["width"], NARROW["height"]], state["rect"]
            page.screenshot(path=str(evidence / "webkit-narrow-dark-notes.png"))
            _close_with_escape(page)
            state = _open(page, "brief.md", ready=".document-reader-markdown")
            assert any(state["tables"]) and state["bodyOverflowX"] <= 1
            page.screenshot(path=str(evidence / "webkit-narrow-dark-brief.png"))
            _close_with_escape(page)
        finally:
            webkit.close()


PROJECT_FILES = {"room-brief.md": BRIEF.encode(), "room-notes.txt": NOTES.encode()}

LEFT_BEHIND = """room => ({
    readers: document.querySelectorAll('dialog.document-reader').length,
    open_dialogs: document.querySelectorAll('dialog[open]').length,
    room: Boolean(document.querySelector(room)),
    focus_connected: !document.activeElement || document.activeElement.isConnected,
})"""

OPEN_ROOM = """project => window.dispatchEvent(new CustomEvent('ouro:open-project', {detail: {project}}))"""


def _enter_room(page, project):
    """Open a Project room from the navigation column, as the owner does."""
    row = page.locator(f'.nav-project-row[data-project-id="{project["id"]}"]')
    row.wait_for(state="attached", timeout=30_000)
    toggle = page.locator("#page-chat [data-mobile-nav-toggle]")
    if toggle.is_visible() and not page.locator("#primary-sidebar").evaluate("node => node.classList.contains('open')"):
        toggle.click()
    row.click()
    feed = f'#pchat-{project["id"]}-messages'
    page.locator(feed).wait_for(state="visible", timeout=30_000)
    return feed


def _await_abort(page, failed, url):
    for _ in range(60):
        if url in failed:
            return True
        page.wait_for_timeout(50)
    return False


def _release(page, held):
    for route in held:
        try:
            route.continue_()
        except Exception:  # an aborted request has no route left to continue
            pass
    held.clear()
    page.unroute("**/artifacts/**")


@pytest.mark.ui_browser
@pytest.mark.serial
def test_project_room_reader_survives_reload_and_closes_with_its_room(direct_server_with_data, monkeypatch, tmp_path):  # noqa: F811
    """A Project room's delivered documents read live and after a reload; leaving the room
    destroys its chat, which aborts a pending read and removes an open reader, and the room
    reopens to a working reader. Leaving goes through the app's own `ouro:open-project`
    navigation (what a Project reference does): a modal reader makes the panel's own Close inert."""
    from playwright.sync_api import sync_playwright
    from tests import fixtures_mock_llm

    root = direct_server_with_data["data_dir"]
    url = direct_server_with_data["url"]
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path / "evidence")))
    evidence.mkdir(parents=True, exist_ok=True)
    source = root / "room-src"
    source.mkdir()
    paths = []
    for name, data in PROJECT_FILES.items():
        (source / name).write_bytes(data)
        paths.append(source / name)
    calls = []
    monkeypatch.setattr(fixtures_mock_llm._Handler, "do_POST", _send_files_mock(paths, calls))
    room = _api(url, "POST", "/api/projects", {"name": "Reader room"})["project"]
    other = _api(url, "POST", "/api/projects", {"name": "Other room"})["project"]
    assert int(room["chat_id"]) not in (1, int(other["chat_id"])), (room, other)
    with sync_playwright() as playwright:
        chromium = playwright.chromium.launch()
        try:
            page = chromium.new_page(viewport=WIDE, color_scheme="dark")
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
            frames, requests, failed = {}, [], []

            def capture(payload):
                value = json.loads(payload) if isinstance(payload, str) and '"document"' in payload else {}
                if value.get("type") == "document":
                    frames.setdefault(value["filename"], value)

            page.on("websocket", lambda socket: socket.on("framereceived", capture))
            page.on("request", lambda request: requests.append((request.url, request.headers.get("range"))))
            page.on("requestfailed", lambda request: failed.append(request.url))
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_function("() => window.__testSockets?.some(socket => socket.readyState === WebSocket.OPEN)")
            feed = _enter_room(page, room)
            page.locator(f'[id="pchat-{room["id"]}-input"]').fill("Send me the room documents.")
            page.locator(f'[id="pchat-{room["id"]}-send"]').click()
            for name in PROJECT_FILES:
                page.locator(f"{feed} .chat-file-card").filter(has_text=name).wait_for(state="visible", timeout=60_000)
            assert sorted(calls) == sorted(PROJECT_FILES) and sorted(frames) == sorted(PROJECT_FILES)
            assert {int(frame["chat_id"]) for frame in frames.values()} == {int(room["chat_id"])}, frames
            assert page.locator("#chat-messages .chat-file-card").count() == 0, "the room's files stay in the room"
            artifact = {name: url + frame["download_url"] for name, frame in frames.items()}

            # Like Main, measure the settled turn, not its file-before-final interval.
            page.locator(f"{feed} .chat-bubble.assistant").filter(
                has_text="The files are delivered.").wait_for(timeout=60_000)
            page.locator(f'{feed} .chat-live-card [data-phase="done"]').wait_for()
            # Live, in the room: the inline delivered bytes.
            requests.clear()
            state = _open(page, "room-brief.md", ready=".document-reader-markdown", feed=feed)
            _check_brief(state, "room-brief.md")
            assert state["active"]
            live_markdown = state["markdown"]
            assert not [entry for entry in requests if "/artifacts/" in entry[0] or "/api/files/" in entry[0]], requests
            page.screenshot(path=str(evidence / "chromium-wide-dark-project-brief.png"))
            _close_with_escape(page)
            assert page.evaluate(f"() => document.activeElement?.closest('{feed} .chat-file-card')?.textContent.includes('room-brief.md')")
            assert abs(_feed_scroll(page, feed) - state["opening_scroll"]) <= 1, "closing returns to the same place in the room"

            # Reload: the room rebuilds from history and reads the immutable artifact route.
            page.reload(wait_until="domcontentloaded")
            page.wait_for_function("() => window.__testSockets?.some(socket => socket.readyState === WebSocket.OPEN)")
            feed = _enter_room(page, room)
            page.locator(f"{feed} .chat-file-card").filter(has_text="room-brief.md").wait_for(state="visible", timeout=30_000)
            requests.clear()
            state = _open(page, "room-brief.md", ready=".document-reader-markdown", feed=feed)
            _check_brief(state, "room-brief.md")
            assert state["markdown"] == live_markdown, "replay shows the same delivered document"
            assert (artifact["room-brief.md"], None) in requests, requests
            _close_with_escape(page)

            # Leaving the room while a read is pending aborts it and leaves nothing behind.
            held = []
            page.route("**/artifacts/**", lambda route: held.append(route))
            page.locator(f"{feed} .chat-file-card").filter(has_text="room-notes.txt").click()
            page.locator("dialog.document-reader .document-reader-status.is-loading").wait_for(state="attached")
            failed.clear()
            page.evaluate(OPEN_ROOM, other)
            page.locator(f'#pchat-{other["id"]}-messages').wait_for(state="visible", timeout=30_000)
            assert _await_abort(page, failed, artifact["room-notes.txt"]), "leaving the room aborted the pending read"
            assert page.evaluate(LEFT_BEHIND, feed) == {
                "readers": 0, "open_dialogs": 0, "room": False, "focus_connected": True}
            _release(page, held)
            # Nothing modal is left over the app: the room now shown takes a click at once.
            page.locator(f'[id="pchat-{other["id"]}-input"]').click(timeout=5_000)

            # The room reopens to a working reader (now from history, so the artifact route).
            feed = _enter_room(page, room)
            requests.clear()
            state = _open(page, "room-notes.txt", ready=".document-reader-source", feed=feed)
            assert state["source"] == NOTES and state["title"] == "room-notes.txt"
            assert (artifact["room-notes.txt"], None) in requests, requests

            # Leaving with the document open closes it the same way.
            page.evaluate(OPEN_ROOM, other)
            page.locator(f'#pchat-{other["id"]}-messages').wait_for(state="visible", timeout=30_000)
            assert page.evaluate(LEFT_BEHIND, feed) == {
                "readers": 0, "open_dialogs": 0, "room": False, "focus_connected": True}
            page.locator(f'[id="pchat-{other["id"]}-input"]').click(timeout=5_000)
        finally:
            chromium.close()

        webkit = playwright.webkit.launch()
        try:
            page = webkit.new_page(viewport=NARROW, color_scheme="light")
            failed = []
            page.on("requestfailed", lambda request: failed.append(request.url))
            page.goto(url, wait_until="domcontentloaded")
            feed = _enter_room(page, room)
            page.locator(f"{feed} .chat-file-card").filter(has_text="room-brief.md").wait_for(state="visible", timeout=30_000)
            state = _open(page, "room-brief.md", ready=".document-reader-markdown", feed=feed)
            _check_brief(state, "room-brief.md")
            assert [round(value) for value in state["rect"]] == [0, 0, NARROW["width"], NARROW["height"]], state["rect"]
            page.screenshot(path=str(evidence / "webkit-narrow-light-project-brief.png"))
            _close_with_escape(page)
            assert page.evaluate(f"() => document.activeElement?.closest('{feed} .chat-file-card')?.textContent.includes('room-brief.md')")
            assert abs(_feed_scroll(page, feed) - state["opening_scroll"]) <= 1

            held = []
            page.route("**/artifacts/**", lambda route: held.append(route))
            page.locator(f"{feed} .chat-file-card").filter(has_text="room-notes.txt").click()
            page.locator("dialog.document-reader .document-reader-status.is-loading").wait_for(state="attached")
            page.evaluate(OPEN_ROOM, other)
            page.locator(f'#pchat-{other["id"]}-messages').wait_for(state="visible", timeout=30_000)
            assert _await_abort(page, failed, artifact["room-notes.txt"])
            assert page.evaluate(LEFT_BEHIND, feed) == {
                "readers": 0, "open_dialogs": 0, "room": False, "focus_connected": True}
            _release(page, held)
            # On a phone the other room covers Main's menu: close it, then reopen the room.
            page.locator("#project-panel-close").click()
            feed = _enter_room(page, room)
            state = _open(page, "room-notes.txt", ready=".document-reader-source", feed=feed)
            assert state["source"] == NOTES
            _close_with_escape(page)
        finally:
            webkit.close()


# A notification as the owner meets it: the real notifier decides on a socket frame and raises a
# banner (a recording stand-in for the system's Notification, permission granted); the test then
# clicks that banner, which runs the app's own activation.
NOTIFIER = """() => {
    localStorage.setItem('ouroboros.notifications', JSON.stringify({enabled: true}));
    window.__notes = [];
    class TestNotification {
        constructor(title, options) { this.title = title; this.options = options; window.__notes.push(this); }
        close() {}
    }
    TestNotification.permission = 'granted';
    TestNotification.requestPermission = () => Promise.resolve('granted');
    window.Notification = TestNotification;
}"""

EMIT_NOTICE = """([chatId, ts]) => {
    const socket = window.__testSockets.find((candidate) => candidate.readyState === WebSocket.OPEN);
    if (!socket) throw new Error('test socket is not open');
    socket.dispatchEvent(new MessageEvent('message', {data: JSON.stringify({
        type: 'chat', role: 'assistant', system_type: 'proactive_message', chat_id: chatId,
        content: 'Something in this chat needs you.', ts})}));
}"""

SCREEN = """() => ({
    readers: document.querySelectorAll('dialog.document-reader').length,
    open_dialogs: document.querySelectorAll('dialog[open]').length,
    focus_connected: !document.activeElement || document.activeElement.isConnected,
})"""

KEPT_ROOM = """id => {
    const page = document.getElementById(`panel-pchat-${id}`);
    return page && {hidden: page.hidden, pending: page.dataset.pendingWork || '',
        staged: [...page.querySelectorAll('.attach-name')].map((node) => node.textContent)};
}"""

NOTHING_LEFT = {"readers": 0, "open_dialogs": 0, "focus_connected": True}


def _click_notification(page, chat_id, step):
    count = page.evaluate("() => window.__notes.length")
    page.evaluate(EMIT_NOTICE, [chat_id, f"2026-10-07T12:00:{step:02d}Z"])
    page.wait_for_function("count => window.__notes.length > count", arg=count)
    page.evaluate("() => window.__notes.at(-1).onclick()")


def _deliver(page, monkeypatch, paths, calls, *, composer, send, feed):
    from tests import fixtures_mock_llm

    monkeypatch.setattr(fixtures_mock_llm._Handler, "do_POST", _send_files_mock(paths, calls))
    page.locator(composer).fill("Send me the documents.")
    page.locator(send).click()
    for path in paths:
        page.locator(f"{feed} .chat-file-card").filter(has_text=path.name).wait_for(state="visible", timeout=60_000)
    # The turn's closing reply: its model call is over, so the next delivery's mock cannot be
    # consumed by this turn.
    page.locator(f"{feed} .chat-bubble").filter(has_text="The files are delivered.").first.wait_for(timeout=60_000)


def _hold_artifacts(page):
    held = []
    page.route("**/artifacts/**", lambda route: held.append(route))
    return held


@pytest.mark.ui_browser
@pytest.mark.serial
def test_reader_closes_when_its_chat_leaves_the_screen_and_staged_files_stay(direct_server_with_data, monkeypatch, tmp_path):  # noqa: F811
    """A document open — or still loading — in Main or in a room closes when that chat leaves the
    screen without being destroyed: a notification opening a room over Main, a notification back to
    Main that hides a room kept for its staged file, a switch to another room, a page change. The
    pending read is aborted, no modal is left over the next view, and the kept room still holds its
    staged file when it is shown again."""
    from playwright.sync_api import sync_playwright

    root = direct_server_with_data["data_dir"]
    url = direct_server_with_data["url"]
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path / "evidence")))
    evidence.mkdir(parents=True, exist_ok=True)
    main_paths, room_paths = [], []
    for folder, files, paths in (("main-src", {"nav-brief.md": BRIEF, "nav-notes.txt": NOTES}, main_paths),
                                 ("kept-src", {"kept-brief.md": BRIEF, "kept-notes.txt": NOTES}, room_paths)):
        (root / folder).mkdir()
        for name, text in files.items():
            (root / folder / name).write_text(text, encoding="utf-8")
            paths.append(root / folder / name)
    room = _api(url, "POST", "/api/projects", {"name": "Kept room"})["project"]
    other = _api(url, "POST", "/api/projects", {"name": "Next room"})["project"]
    calls, frames = [], {}
    with sync_playwright() as playwright:
        for engine, viewport, scheme in (("chromium", WIDE, "dark"), ("webkit", NARROW, "light")):
            browser = getattr(playwright, engine).launch()
            try:
                page = browser.new_page(viewport=viewport, color_scheme=scheme)
                page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
                page.add_init_script(f"({NOTIFIER})()")
                failed = []
                page.on("requestfailed", lambda request: failed.append(request.url))

                def capture(payload):
                    value = json.loads(payload) if isinstance(payload, str) and '"document"' in payload else {}
                    if value.get("type") == "document":
                        frames.setdefault(value["filename"], value)

                page.on("websocket", lambda socket: socket.on("framereceived", capture))
                page.goto(url, wait_until="domcontentloaded")
                page.wait_for_function("() => window.__testSockets?.some(socket => socket.readyState === WebSocket.OPEN)")
                if not calls:
                    _deliver(page, monkeypatch, main_paths, calls, composer="#chat-input", send="#chat-send", feed="#chat-messages")
                    feed = _enter_room(page, room)
                    _deliver(page, monkeypatch, room_paths, calls, composer=f'[id="pchat-{room["id"]}-input"]',
                             send=f'[id="pchat-{room["id"]}-send"]', feed=feed)
                    assert sorted(calls) == sorted(path.name for path in main_paths + room_paths)
                    # From here on every document comes from its immutable artifact route.
                    page.reload(wait_until="domcontentloaded")
                    page.wait_for_function("() => window.__testSockets?.some(socket => socket.readyState === WebSocket.OPEN)")
                artifact = {name: url + frame["download_url"] for name, frame in frames.items()}
                page.locator("#chat-messages .chat-file-card").filter(has_text="nav-notes.txt").wait_for(timeout=30_000)

                # 1. Main is reading (the read still pending); a notification opens a room over Main.
                held = _hold_artifacts(page)
                page.locator("#chat-messages .chat-file-card").filter(has_text="nav-notes.txt").click()
                page.locator("dialog.document-reader .document-reader-status.is-loading").wait_for(state="attached")
                _click_notification(page, int(room["chat_id"]), 1)
                feed = f'#pchat-{room["id"]}-messages'
                page.locator(feed).wait_for(state="visible", timeout=30_000)
                assert _await_abort(page, failed, artifact["nav-notes.txt"]), "Main's pending read was aborted"
                assert page.evaluate(SCREEN) == NOTHING_LEFT
                _release(page, held)

                # 2. The room holds a staged file and is reading; a notification goes back to Main.
                page.locator("#project-panel .chat-file-input-hidden").set_input_files(
                    [{"name": "kept.txt", "mimeType": "text/plain", "buffer": b"kept"}])
                page.locator("#project-panel .attach-name").filter(has_text="kept.txt").wait_for()
                page.locator(f"{feed} .chat-file-card").filter(has_text="kept-notes.txt").wait_for(timeout=30_000)
                held = _hold_artifacts(page)
                failed.clear()
                page.locator(f"{feed} .chat-file-card").filter(has_text="kept-notes.txt").click()
                page.locator("dialog.document-reader .document-reader-status.is-loading").wait_for(state="attached")
                _click_notification(page, 1, 2)
                page.wait_for_function("id => document.getElementById(`panel-pchat-${id}`)?.hidden === true", arg=room["id"])
                assert _await_abort(page, failed, artifact["kept-notes.txt"]), "the kept room's pending read was aborted"
                assert page.evaluate(SCREEN) == NOTHING_LEFT
                assert page.evaluate(KEPT_ROOM, room["id"]) == {"hidden": True, "pending": "1", "staged": ["kept.txt"]}
                _release(page, held)
                page.locator("#project-panel").wait_for(state="hidden")
                page.locator("#chat-input").click(timeout=5_000)

                # 3. Shown again it is the same room with its file; reading, it gives way to another room.
                feed = _enter_room(page, room)
                assert page.evaluate(KEPT_ROOM, room["id"]) == {"hidden": False, "pending": "", "staged": ["kept.txt"]}
                state = _open(page, "kept-brief.md", ready=".document-reader-markdown", feed=feed)
                _check_brief(state, "kept-brief.md")
                page.evaluate(OPEN_ROOM, other)
                page.locator(f'#pchat-{other["id"]}-messages').wait_for(state="visible", timeout=30_000)
                assert page.evaluate(SCREEN) == NOTHING_LEFT
                assert page.evaluate(KEPT_ROOM, room["id"]) == {"hidden": True, "pending": "1", "staged": ["kept.txt"]}
                page.locator(f'[id="pchat-{other["id"]}-input"]').click(timeout=5_000)

                # 4. Back in Main, reading; a page change made while the reader is open (the
                # navigation itself is inert under the modal, so the change is made by code).
                _click_notification(page, 1, 3)
                page.locator("#project-panel").wait_for(state="hidden")
                state = _open(page, "nav-brief.md", ready=".document-reader-markdown")
                _check_brief(state, "nav-brief.md")
                page.evaluate("() => document.querySelector('[data-nav-page=\"settings\"]').click()")
                page.locator("#page-settings.active").wait_for(state="attached")
                assert page.evaluate(SCREEN) == NOTHING_LEFT
                page.screenshot(path=str(evidence / f"{engine}-{scheme}-page-change-no-reader.png"))
                if viewport == NARROW:
                    page.locator("#page-settings [data-mobile-nav-toggle]").click()
                page.locator('[data-nav-page="chat"]').click()
                state = _open(page, "nav-notes.txt", ready=".document-reader-source")
                assert state["source"] == NOTES
                _close_with_escape(page)

                # The kept room still holds its staged file.
                feed = _enter_room(page, room)
                assert page.evaluate(KEPT_ROOM, room["id"])["staged"] == ["kept.txt"]
                page.locator("#project-panel .attach-name").filter(has_text="kept.txt").wait_for(state="visible")
                page.wait_for_timeout(400)  # the drawer and panel transitions, for the screenshot only
                page.screenshot(path=str(evidence / f"{engine}-{scheme}-kept-room-staged.png"))
            finally:
                browser.close()


GUIDE = (
    "# Settings\n\n"
    '[Project site](https://example.com/ "Download") and ![chart](https://example.invalid/c.png "Quarterly chart")\n\n'
    "```python\nprint('copy me')\n```\n"
)
# Past the 32 KiB rich-block bound: it stays plain text and still copies whole.
HUGE_CODE = "```text\n" + "a line of plain code\n" * 1800 + "```\n"
TRANSLATED = {"language": "ru", "english": False, "revision": 1, "entries": {
    "guide.md": {"text": "руководство.md", "provenance": "imported"},
    "Settings": {"text": "Настройки", "provenance": "imported"},
    "Download": {"text": "Скачать", "provenance": "imported"},
    "Close": {"text": "Закрыть", "provenance": "imported"},
    "Close document": {"text": "Закрыть документ", "provenance": "imported"},
    "Formatted": {"text": "Оформленный", "provenance": "imported"},
    "Source": {"text": "Исходный", "provenance": "imported"},
    "Document text": {"text": "Текст документа", "provenance": "imported"},
    "code:media.read": {"text": "Читать", "provenance": "imported"},
    "code:code.copy": {"text": "Копировать", "provenance": "imported"},
    "code:code.copied": {"text": "Скопировано", "provenance": "imported"},
    "code:code.copy_code": {"text": "Копировать код", "provenance": "imported"},
}}

NO_ASYNC_CLIPBOARD = """() => {
    Object.defineProperty(Navigator.prototype, 'clipboard', {configurable: true, get: () => undefined});
    window.__copied = [];
    // Capture after the pointer's default focus action, but before the Copy
    // handler moves focus into its temporary textarea. Engine names do not
    // determine whether a clicked button takes focus on this platform.
    document.addEventListener('click', (event) => {
        if (!event.target.closest?.('dialog.document-reader .md-code-copy')) return;
        window.__copyFocus = document.activeElement;
        window.__copyScroll = document.querySelector('dialog.document-reader .document-reader-body').scrollTop;
    }, true);
    document.addEventListener('copy', () => {
        const node = document.activeElement;
        window.__copied.push({
            text: node && 'value' in node ? node.value.slice(node.selectionStart, node.selectionEnd) : String(getSelection()),
            tag: node?.tagName || '', in_dialog: Boolean(node?.closest?.('dialog[open]')),
        });
    }, true);
}"""

TRANSLATED_STATE = """() => {
    const dialog = document.querySelector('dialog.document-reader[open]');
    const md = dialog.querySelector('.document-reader-markdown');
    const button = (selector) => dialog.querySelector(selector);
    return {
        title: dialog.querySelector('.document-reader-title').textContent,
        meta: dialog.querySelector('.document-reader-meta').textContent,
        heading: md.querySelector('h1, .md-h1')?.textContent || '',
        link_title: md.querySelector('a[href="https://example.com/"]')?.getAttribute('title'),
        image_title: md.querySelector('.md-image-ref')?.getAttribute('title'),
        close: [button('[data-reader-action="close"]').textContent, button('[data-reader-action="close"]').getAttribute('aria-label')],
        download: button('[data-reader-action="download"]').textContent,
        views: [...dialog.querySelectorAll('[data-reader-view]')].map((node) => node.textContent),
        region: dialog.querySelector('.document-reader-body').getAttribute('aria-label'),
        copy: md.querySelector('.md-code-copy')?.textContent,
    };
}"""

COPY_STATE = """() => {
    const dialog = document.querySelector('dialog.document-reader[open]');
    const code = dialog.querySelector('.md-code-block pre > code');
    return {
        button: dialog.querySelector('.md-code-copy').textContent,
        // Where focus is: the Copy button, the reading region, or the tag of anything else.
        active: document.activeElement === dialog.querySelector('.md-code-copy') ? 'copy'
            : document.activeElement === dialog.querySelector('.document-reader-body') ? 'region'
            : document.activeElement?.tagName || '',
        focus_restored: Boolean(window.__copyFocus) && document.activeElement === window.__copyFocus,
        scroll_before_copy: window.__copyScroll,
        scroll: dialog.querySelector('.document-reader-body').scrollTop,
        textareas: dialog.querySelectorAll('textarea').length,
        highlighted: code.classList.contains('hljs'),
        code_text: code.textContent,
        copied: window.__copied.at(-1) || null,
    };
}"""


@pytest.mark.ui_browser
@pytest.mark.serial
def test_reader_in_a_translated_ui_and_copy_without_async_clipboard(direct_server_with_data, monkeypatch, tmp_path):  # noqa: F811
    """With a non-English dictionary (the house idiom: the gateway's /api/ui/i18n answer, stubbed),
    the reader's controls are translated while the document — its name, even one the dictionary
    knows, its size line, its text and its authors' link and image titles — stays as written and
    is never reported as a missing translation. Without the async clipboard, Copy inside the modal
    reader copies through a selection made inside the dialog. A code block past the 32 KiB
    rich-block bound stays plain and copies whole."""
    from playwright.sync_api import sync_playwright

    root = direct_server_with_data["data_dir"]
    url = direct_server_with_data["url"]
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path / "evidence")))
    evidence.mkdir(parents=True, exist_ok=True)
    (root / "guide-src").mkdir()
    paths = []
    for name, text in (("guide.md", GUIDE), ("huge-code.md", HUGE_CODE)):
        (root / "guide-src" / name).write_text(text, encoding="utf-8")
        paths.append(root / "guide-src" / name)
    assert len(HUGE_CODE) > 32768
    calls = []
    with sync_playwright() as playwright:
        for engine, viewport, scheme in (("chromium", WIDE, "dark"), ("webkit", NARROW, "light")):
            browser = getattr(playwright, engine).launch()
            try:
                page = browser.new_page(viewport=viewport, color_scheme=scheme)
                page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
                page.add_init_script(f"({NO_ASYNC_CLIPBOARD})()")
                misses = []
                page.route("**/api/ui/i18n", lambda route: route.fulfill(json=TRANSLATED))
                page.route("**/api/ui/i18n/missing", lambda route: (
                    misses.extend(item.get("key", "") for item in (route.request.post_data_json or {}).get("items", [])),
                    route.fulfill(json={"ok": True})))
                page.goto(url, wait_until="domcontentloaded")
                page.wait_for_function("() => window.__testSockets?.some(socket => socket.readyState === WebSocket.OPEN)")
                assert page.evaluate("() => navigator.clipboard === undefined")
                if not calls:
                    _deliver(page, monkeypatch, paths, calls, composer="#chat-input", send="#chat-send", feed="#chat-messages")
                else:
                    page.locator(".chat-file-card").filter(has_text="guide.md").wait_for(timeout=30_000)
                page.wait_for_function("() => document.documentElement.lang === 'ru'")
                card = page.locator(".chat-file-card").filter(has_text="guide.md")
                assert card.locator(".chat-file-more").inner_text() == "Читать"

                _open(page, "guide.md", ready=".document-reader-markdown")
                page.wait_for_function("() => document.querySelector('dialog.document-reader [data-reader-action=\"close\"]').textContent === 'Закрыть'")
                state = page.evaluate(TRANSLATED_STATE)
                # The document stays as written, a name the dictionary knows included.
                assert state["title"] == "guide.md" and state["meta"].startswith("MD · "), state
                assert (state["heading"], state["link_title"], state["image_title"]) == ("Settings", "Download", "Quarterly chart"), state
                # Its chrome is translated.
                assert state["close"] == ["Закрыть", "Закрыть документ"] and state["download"] == "Скачать", state
                assert state["views"] == ["Оформленный", "Исходный"] and state["region"] == "Текст документа", state
                assert state["copy"] == "Копировать", state
                page.screenshot(path=str(evidence / f"{engine}-{viewport['width']}-{scheme}-translated-reader.png"))

                before = page.evaluate(COPY_STATE)
                assert before["highlighted"], "a code block within the bound is highlighted"
                page.locator("dialog.document-reader .md-code-copy").click()
                page.wait_for_function("() => document.querySelector('dialog.document-reader .md-code-copy').textContent === 'Скопировано'")
                after = page.evaluate(COPY_STATE)
                # Exactly what the block holds, selected inside the dialog.
                assert after["code_text"].strip() == "print('copy me')", after
                assert after["copied"] == {"text": after["code_text"], "tag": "TEXTAREA", "in_dialog": True}, after
                # Restore the actual pre-handler focus and viewport, not an
                # engine-name guess or the position before click auto-scrolling.
                assert after["focus_restored"], after
                assert after["textareas"] == 0 and after["scroll"] == after["scroll_before_copy"], after
                _close_with_escape(page)

                state = _open(page, "huge-code.md", ready=".document-reader-markdown")
                big = page.evaluate(COPY_STATE)
                assert not big["highlighted"] and len(big["code_text"]) > 32768, len(big["code_text"])
                # Keyboard activation from a scrolled document exercises both
                # actual focus restoration and a nonzero viewport position.
                page.evaluate("""() => {
                    const dialog = document.querySelector('dialog.document-reader');
                    dialog.querySelector('.document-reader-body').scrollTop = 100;
                    dialog.querySelector('.md-code-copy').focus({preventScroll: true});
                }""")
                page.keyboard.press("Enter")
                page.wait_for_function("() => document.querySelector('dialog.document-reader .md-code-copy').textContent === 'Скопировано'")
                after = page.evaluate(COPY_STATE)
                assert after["focus_restored"] and after["textareas"] == 0, after
                assert after["scroll"] == after["scroll_before_copy"] > 0, after
                copied = after["copied"]
                assert copied["in_dialog"] and copied["text"] == big["code_text"], len(copied["text"])
                assert copied["text"].strip() == HUGE_CODE[len("```text\n"):-len("```\n")].strip()
                _close_with_escape(page)

                page.evaluate("() => import('/static/modules/i18n.js').then((module) => module.flushMisses())")
                page.wait_for_timeout(500)
                authored = ("guide.md", "huge-code.md", "MD · ", "Project site", "Quarterly chart", "Image: chart", "copy me", "plain code")
                assert not [key for key in misses if any(part in key for part in authored)], misses
            finally:
                browser.close()
