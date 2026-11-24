"""Chat Markdown reading and fidelity through the real SPA (#1370, #1367, #1368).

The served page is the real `web/` tree: `index.html`, `chat.js`, both Markdown
renderers and the production CSS. Only the HTTP API and the socket are
answered by the test, with deterministic history rows. Every check is an
invariant a reader can see — a heading never below the text it opens, a table
that scrolls only when it must and is then reachable by keyboard, code that
reads and copies as the author wrote it, an image that stays a visible
reference and never loads — measured at desktop, a narrow Project panel and a
phone, in Chromium and WebKit.
"""
from __future__ import annotations

import json
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import pytest

from tests.ui_console_errors import PageErrors

pytestmark = [pytest.mark.ui_browser]
WEB = Path(__file__).resolve().parents[1] / "web"

LEAD = (
    "The migration keeps every stored answer readable while the renderer changes underneath it, "
    "so the owner can compare the old and the new presentation on the same history without "
    "reloading anything or losing the place they were reading."
)
HEADINGS = "\n\n".join([
    "# Reading ladder", LEAD,
    "## What changes", f"**Bold lead sentence.** {LEAD}",
    "### Sub-question", f"A paragraph with `inline code` and a [link](https://example.com/docs). {LEAD}",
    "#### Fourth level", LEAD, "---", "After the rule.",
])
PROSE_CELLS = [
    ("Keep the grid",
     "Rows and columns stay a real table, so a reader can compare one option against another by reading "
     "down a column, and assistive technology still announces the header of every cell it reads. Cells "
     "wrap like any paragraph, which means a long explanation grows the row downwards instead of pushing "
     "the rest of the table out of the bubble and hiding it behind a sideways scroll that nobody notices.",
     "Rows become taller when a single cell carries a long explanation, so a very wordy table reads as a "
     "sequence of paragraphs placed side by side."),
    ("Stack rows as cards",
     "Every row turns into a small card with the column names repeated above each value. That reads well "
     "one row at a time on a phone, but comparing two options means scrolling between cards and "
     "remembering what the previous one said, and the renderer would need a rule for which tables to "
     "stack, which is a guess about the author's intent that the author never made. Numeric tables "
     "stacked this way lose the column alignment that made them easy to scan in the first place.",
     "The transcript gets longer, and table semantics must be rebuilt by hand once display changes."),
]
PROSE_TABLE = "\n".join(
    ["| Option | What changes | Risk |", "| --- | --- | --- |"]
    + [f"| {a} | {b} | {c} |" for a, b, c in PROSE_CELLS]
)
NUMERIC_TABLE = "\n".join([
    "| Region | Revenue | Change | Status |",
    "| :--- | ---: | :---: | --- |",
    "| North | 1,204,331.50 | +4.2% | on track |",
    "| South | 98,014.07 | -0.8% | watch |",
])
WIDE_TABLE = "\n".join([
    "| Metric | " + " | ".join(f"Month {n}" for n in range(1, 9)) + " |",
    "| --- | " + " | ".join("---:" for _ in range(8)) + " |",
    "| Requests | " + " | ".join(f"{1_000_000 + n * 7_123_457:,}" for n in range(8)) + " |",
])
LONG_URL = "https://example.com/" + "/".join(f"segment-{n:02d}-with-a-long-name" for n in range(8)) + "?q=1&r=2"
URL_TABLE = "\n".join([
    "| Source | Address |", "| --- | --- |",
    f"| Report | [{LONG_URL}]({LONG_URL}) |",
    "| Commit | `3f786850e387550fdab836ed7e6dc881de23001b6f0c1a4b19d2f8e4c1a7d2e90` |",
])
FENCE_BODY = "\n".join([
    "&amp; &lt; &#42; **stars** <div class=\"x\">",
    "- literal dash",
    "# literal hash",
    "[x](https://example.com/y) | a | b |",
])
FIDELITY = "\n\n".join([
    "Fidelity check.",
    f"```text\n{FENCE_BODY}\n```",
    "Inline: `&amp; &lt; &#42; **stars** <b>` stays literal.",
    "Prose entities: Fish &amp; chips &lt;tag&gt; read once.",
    "Raw HTML: <img src=x onerror=\"window.__markdownOnerror = 1\"> <script>window.__markdownScript = 1</script>",
    'Before ![diagram](https://images.example/a.png "Architecture") after.',
    "[![badge](https://images.example/b.png)](https://example.com/docs)",
    "![unsafe](javascript:alert(1))",
    "![](https://images.example/empty.png)",
    "![see [the source](https://example.com/src)](https://images.example/c.png)",
    "- [ ] Compute \\(x_1 + 1\\)\n- [x] Checked $$y_1 *c*$$",
])
# Compact Markdown (a Skill Review report, a task timeline) keeps its density,
# and its tables scroll exactly like a rich answer's.
COMPACT_NARROW = "| Check | Result |\n| --- | :---: |\n| Fence kept | yes |"


def compact_wide_table(runs):
    return "\n".join([
        "| Signal | " + " | ".join(f"Run {n}" for n in range(1, runs + 1)) + " |",
        "| --- | " + " | ".join("---:" for _ in range(runs)) + " |",
        "| Latency ms | " + " | ".join(f"{120_000 + n * 3_457:,}.25" for n in range(runs)) + " |",
    ])


COMPACT_WIDE = compact_wide_table(8)
COMPACT_LINE = "const report = " + "measured_dependency_" * 12 + "END_OF_REPORT"
SKILL_REVIEW = (
    f"# Skill review: demo\n\nReviewers: alpha\n\n## Findings\n\n```c++\n{FENCE_BODY}\n```\n\n"
    f"```text\n{COMPACT_LINE}\n```\n\n"
    f"### Evidence\n\n{COMPACT_NARROW}\n\n#### Timings\n\n{COMPACT_WIDE}\n"
)
ROWS = [
    {"role": "assistant", "text": text, "ts": f"2026-09-28T10:00:0{index}Z", "markdown": True}
    for index, text in enumerate([HEADINGS, PROSE_TABLE + "\n\n" + NUMERIC_TABLE,
                                  WIDE_TABLE + "\n\n" + URL_TABLE, FIDELITY])
] + [{"role": "system", "system_type": "skill_review", "text": SKILL_REVIEW, "ts": "2026-09-28T10:00:09Z"}]
HISTORY_COVERAGE = {"v": 1, "view": "fixture", "upper": {"chat": len(ROWS), "progress": 0}, "spans": {
    "chat": {"from": 0, "to": len(ROWS), "chain": "fixture", "gaps": []},
    "progress": {"from": 0, "to": 0, "chain": "empty", "gaps": []}}}


def capture(page, name):
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if root:
        Path(root).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(Path(root) / f"{name}.png"), animations="disabled")


@pytest.fixture(params=["chromium", "webkit"])
def reading_ui(request):
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("Set OUROBOROS_RUN_UI_SMOKE=1 to run the browser flow")
    playwright = pytest.importorskip("playwright.sync_api")
    engine = request.param
    foreign: list[str] = []

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            self.path = self.path.removeprefix("/static")
            super().do_GET()

    def respond(route):
        path = urlparse(route.request.url).path
        if path == "/api/onboarding":
            route.fulfill(status=204)
            return
        body = {
            # One complete first page under the history page contract: a cursor, nothing older,
            # and coverage of every chat byte up to the horizon (progress known empty).
            "/api/chat/history": {"messages": ROWS, "progress": [], "has_more": False, "page_cursor": "fixture-page",
                                  "coverage": HISTORY_COVERAGE},
            "/api/state": {"supervisor_ready": True, "active_chat_activities": [], "projects": []},
            "/api/projects": {"projects": []},
            "/api/health": {"ok": True, "version": "fixture"},
        }.get(path, {})
        route.fulfill(content_type="application/json", body=json.dumps(body))

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(WEB)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"127.0.0.1:{server.server_port}"

    def local_only(route):
        # A Markdown image or any other third-party load must never leave the page.
        if urlparse(route.request.url).netloc == origin:
            route.fallback()
        else:
            foreign.append(route.request.url)
            route.abort()

    try:
        with playwright.sync_playwright() as pw:
            try:
                browser = getattr(pw, engine).launch(headless=True)
            except playwright.Error as exc:
                if "Executable doesn't exist" in str(exc):
                    pytest.skip(f"{engine} is not installed: {exc}")
                raise
            try:
                page = browser.new_page(viewport={"width": 1280, "height": 900})
                page.add_init_script("""(() => {
                    window.__copied = null;
                    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: {
                        writeText: async (text) => { window.__copied = text; },
                    }});
                })()""")
                errors = PageErrors(page, engine)
                page.route("**/*", local_only)
                page.route("**/api/**", respond)
                page.route_web_socket("**/ws", lambda ws: ws.send(json.dumps({"type": "heartbeat"})))
                yield {"page": page, "url": f"http://{origin}", "engine": engine, "foreign": foreign}
                assert not errors.unexpected(), errors
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def open_chat(ui, width, height=900):
    page = ui["page"]
    page.set_viewport_size({"width": width, "height": height})
    page.goto(ui["url"] + "/")
    page.wait_for_selector("#chat-input", state="attached")
    page.locator("#chat-messages .chat-bubble.assistant", has_text="After the rule.").wait_for()
    # The history fixture honours the page contract, so no screenshot carries a false load failure.
    assert page.get_by_text("Some saved history could not be loaded.").count() == 0
    return page


def mount_project_panel(page):
    """A real secondary chat instance in the Project panel (440px at this viewport)."""
    page.evaluate("""async () => {
        const {createChatInstance} = await import('/static/modules/chat.js');
        const {createStateSnapshotSequencer} = await import('/static/modules/chat_activity.js');
        const mount = document.createElement('aside');
        mount.id = 'reading-panel'; mount.className = 'project-panel open';
        document.body.append(mount);
        const ws = {on: () => () => {}, isConnected: () => true, send: () => {}};
        window.readingPanel = createChatInstance({ws, state: {activePage: 'chat', projectChatIds: new Set()},
            updateUnreadBadge: () => {}, stateSnapshots: createStateSnapshotSequencer(() => {}),
            chatId: 101, projectId: 'reading', idPrefix: 'reading', mountEl: mount, asPanel: true});
        await window.readingPanel.refreshHistory();
    }""")
    page.locator("#reading-panel .chat-bubble.assistant", has_text="After the rule.").wait_for()
    return "#reading-panel"


MEASURE = """(root) => {
    const column = document.querySelector(root + ' .chat-messages') || document.querySelector(root);
    const cs = getComputedStyle(column);
    const content = column.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
    const style = (el) => getComputedStyle(el);
    const bubble = [...column.querySelectorAll('.chat-bubble.assistant')].find((b) => b.textContent.includes('After the rule.'));
    const message = bubble.querySelector('.message');
    const heading = (tag) => {
        const el = message.querySelector(tag), s = style(el);
        return {size: parseFloat(s.fontSize), weight: Number(s.fontWeight), color: s.color,
                above: parseFloat(s.marginTop), below: parseFloat(s.marginBottom), wrap: s.textWrap || s.textWrapStyle || ''};
    };
    const code = message.querySelector('code.inline-code');
    const hr = message.querySelector('hr');
    const tables = [...column.querySelectorAll('.message.ui-rich-content .md-table-wrap')].map((wrap) => ({
        head: wrap.querySelector('th').textContent,
        fadeClass: wrap.classList.contains('scroll-fade-x'),
        size: parseFloat(style(wrap.querySelector('table')).fontSize),
        overflow: wrap.scrollWidth - wrap.clientWidth,
        region: wrap.getAttribute('role'), label: wrap.getAttribute('aria-label'),
        tabindex: wrap.getAttribute('tabindex'),
        edges: [wrap.hasAttribute('data-scroll-start'), wrap.hasAttribute('data-scroll-end')],
        whiteSpace: style(wrap.querySelector('td')).whiteSpace,
        valign: style(wrap.querySelector('td')).verticalAlign,
        numeric: style(wrap.querySelector('table')).fontVariantNumeric,
        aligns: [...wrap.querySelectorAll('tbody tr:first-child td')].map((td) => [td.getAttribute('align'), style(td).textAlign]),
    }));
    return {
        engine: navigator.userAgent, content, bubble: bubble.getBoundingClientRect().width,
        body: parseFloat(style(message.querySelector('p')).fontSize), ink: style(message).color,
        strong: Number(style(message.querySelector('strong')).fontWeight),
        h1: heading('h1'), h2: heading('h2'), h3: heading('h3'), h4: heading('h4'),
        code: {color: style(code).color, parent: style(code.parentElement).color},
        hr: {height: hr.getBoundingClientRect().height, top: style(hr).borderTopWidth, style: style(hr).borderTopStyle,
             above: parseFloat(style(hr).marginTop), below: parseFloat(style(hr).marginBottom)},
        tables,
    };
}"""


def settle_tables(page, root):
    """Wait until every laid-out table's region state follows its overflow.

    The marking rides a ResizeObserver, which reports after layout; a
    measurement taken between a layout change and that report would see the
    previous state. A state that never converges still fails the assertions.
    """
    try:
        page.wait_for_function("""(root) => [...document.querySelectorAll(root + ' .md-table-wrap')]
            .filter((wrap) => wrap.getClientRects().length)
            .every((wrap) => (wrap.scrollWidth - wrap.clientWidth > 1) === (wrap.getAttribute('role') === 'region'))""",
            arg=root, timeout=5000)
    except Exception:
        pass


def measure(page, root):
    settle_tables(page, root)
    return page.evaluate(MEASURE, root)


def assert_reading(metrics, *, narrow):
    body = metrics["body"]
    assert body == 16, metrics
    # The ladder: # = ## > ### > deeper = the reading size, never below it.
    assert metrics["h1"]["size"] == metrics["h2"]["size"] == 20, metrics
    assert metrics["h3"]["size"] == 18 and metrics["h4"]["size"] == body, metrics
    for level in ("h1", "h2", "h3", "h4"):
        heading = metrics[level]
        assert heading["weight"] == 600 and heading["color"] == metrics["ink"], (level, metrics)
    assert metrics["strong"] <= metrics["h2"]["weight"], metrics
    for level in ("h2", "h3", "h4"):
        assert metrics[level]["above"] > metrics[level]["below"], (level, metrics)
    assert metrics["code"]["color"] == metrics["code"]["parent"], "inline code keeps the text ink"
    hr = metrics["hr"]
    assert hr["top"] == "1px" and hr["style"] == "solid" and hr["height"] <= 1 and hr["above"] == hr["below"] == 24, hr
    if narrow:
        # One narrow-column rule: the bubble takes the whole column.
        assert metrics["bubble"] >= metrics["content"] - 2, metrics
    else:
        assert metrics["bubble"] <= metrics["content"] * 0.8 + 1, metrics
    tables = {table["head"]: table for table in metrics["tables"]}
    assert set(tables) >= {"Option", "Region", "Metric", "Source"}, tables
    for table in tables.values():
        # A rich answer's table reads at the answer's own size, never shrunk to fit.
        assert table["fadeClass"] and table["size"] == body, table
        assert table["whiteSpace"] == "normal" and table["valign"] == "top", table
        assert "tabular-nums" in table["numeric"], table
        scrolls = table["overflow"] > 1
        # A table is a keyboard region exactly while it scrolls; unscrolled, only its end edge hides columns.
        assert (table["region"], table["tabindex"], table["edges"]) == (
            ("region", "0", [False, True]) if scrolls else (None, None, [False, False])), table
        if scrolls:
            assert table["label"] == "Scrollable table", table
    assert tables["Metric"]["overflow"] > 1, "the eight-column table must still scroll"
    assert tables["Region"]["aligns"] == [["left", "left"], ["right", "right"], ["center", "center"], [None, "start"]]
    if not narrow:
        for head in ("Option", "Region", "Source"):
            assert tables[head]["overflow"] <= 1, tables[head]


def assert_keyboard_scroll(page, root):
    wrap = page.locator(f"{root} .md-table-wrap").filter(has_text="Month 8").first
    wrap.focus()
    assert wrap.evaluate("el => document.activeElement === el")
    assert wrap.evaluate("el => [el.hasAttribute('data-scroll-start'), el.hasAttribute('data-scroll-end')]") == [False, True]
    # Held, not tapped: WebKit scrolls a focused region while the key is down.
    page.keyboard.press("ArrowRight", delay=300)
    page.wait_for_function("el => el.scrollLeft > 0 && el.hasAttribute('data-scroll-start')", arg=wrap.element_handle())


def test_reading_ladder_and_tables_at_three_widths(reading_ui):
    ui, engine = reading_ui, reading_ui["engine"]
    page = open_chat(ui, 1280)
    assert_reading(measure(page, "#chat-messages"), narrow=False)
    assert_keyboard_scroll(page, "#chat-messages")
    page.locator("#chat-messages .chat-bubble", has_text="Reading ladder").first.scroll_into_view_if_needed()
    capture(page, f"reading-desktop-headings-{engine}")
    page.locator("#chat-messages .md-table-wrap", has_text="Keep the grid").first.scroll_into_view_if_needed()
    capture(page, f"reading-desktop-tables-{engine}")

    panel = mount_project_panel(page)
    assert page.locator(panel).evaluate("el => el.getBoundingClientRect().width") == 440
    assert_reading(measure(page, panel), narrow=True)
    assert_keyboard_scroll(page, panel)
    page.locator(f"{panel} .md-table-wrap", has_text="Keep the grid").first.scroll_into_view_if_needed()
    capture(page, f"reading-panel440-tables-{engine}")
    page.evaluate("window.readingPanel.destroy()")

    page = open_chat(ui, 390, 844)
    assert_reading(measure(page, "#chat-messages"), narrow=True)
    assert_keyboard_scroll(page, "#chat-messages")
    page.locator("#chat-messages .chat-bubble", has_text="Reading ladder").first.scroll_into_view_if_needed()
    capture(page, f"reading-phone390-headings-{engine}")
    page.locator("#chat-messages .md-table-wrap", has_text="Keep the grid").first.scroll_into_view_if_needed()
    capture(page, f"reading-phone390-tables-{engine}")
    assert not ui["foreign"], ui["foreign"]


def test_code_images_and_raw_html_keep_the_author_text(reading_ui):
    ui, engine = reading_ui, reading_ui["engine"]
    page = open_chat(ui, 1280)
    bubble = page.locator("#chat-messages .chat-bubble.assistant", has_text="Fidelity check.").first
    bubble.scroll_into_view_if_needed()
    code = bubble.locator(".md-code-block pre > code")
    assert code.evaluate("el => el.textContent") == FENCE_BODY + "\n"
    bubble.locator("[data-code-copy]").click()
    page.wait_for_function("() => window.__copied !== null")
    assert page.evaluate("() => window.__copied") == FENCE_BODY + "\n"
    assert bubble.locator("code.inline-code").evaluate("el => el.textContent") == "&amp; &lt; &#42; **stars** <b>"
    assert bubble.locator("p", has_text="Prose entities").inner_text() == "Prose entities: Fish & chips <tag> read once."
    assert bubble.locator("script, img, [onerror]").count() == 0
    assert page.evaluate("() => !window.__markdownOnerror && !window.__markdownScript")
    raw = bubble.locator("p", has_text="Raw HTML").inner_text()
    assert '<img src=x onerror="window.__markdownOnerror = 1">' in raw and "<script>" in raw

    diagram = bubble.locator("a.md-image-ref", has_text="Image: diagram")
    assert diagram.evaluate("a => [a.getAttribute('href'), a.title, a.target, a.rel, a.classList.contains('md-link')]") == [
        "https://images.example/a.png", "Architecture", "_blank", "noopener noreferrer", True]
    linked = bubble.locator("p", has_text="Image: badge")
    assert linked.locator("a").count() == 1, "a linked image keeps its one outer link"
    assert linked.locator("a").get_attribute("href") == "https://example.com/docs"
    assert linked.locator("a .md-image-ref").inner_text() == "Image: badge"
    unsafe = bubble.locator(".md-image-ref", has_text="Image: unsafe")
    # A refused address is plain text in its paragraph's ink, never a link-styled anchor without a destination.
    assert unsafe.count() == 1 and unsafe.evaluate("""el => [el.tagName, el.closest('a') === null,
        getComputedStyle(el).color === getComputedStyle(el.parentElement).color,
        getComputedStyle(el).textDecorationLine]""") == ["SPAN", True, True, "none"]
    assert bubble.locator("a.md-image-ref[href='https://images.example/empty.png']").inner_text() == "Image"
    # Words that already carry a link keep that link; anchors never nest.
    worded = bubble.locator(".md-image-ref", has_text="Image: see the source")
    assert worded.evaluate("el => [el.tagName, el.querySelector('a')?.getAttribute('href')]") == [
        "SPAN", "https://example.com/src"]
    assert bubble.locator("a a").count() == 0
    capture(page, f"reading-fidelity-{engine}")
    assert not [url for url in ui["foreign"] if "images.example" in url], ui["foreign"]


def test_task_list_math_renders_beside_its_checkbox(reading_ui):
    page = open_chat(reading_ui, 1280)
    bubble = page.locator("#chat-messages .chat-bubble.assistant", has_text="Fidelity check.").first
    # Code highlighting and KaTeX run in the same enhancement write.
    bubble.locator(".md-code-block code.hljs").wait_for()
    # Each task keeps its box and its words, and KaTeX reads the author's TeX with no Markdown taken out of it.
    assert bubble.locator("li.task-list-item").evaluate_all("""items => items.map((li) => [
        li.querySelector('.md-checkbox')?.textContent,
        [...li.childNodes].filter((node) => node.nodeType === Node.TEXT_NODE).map((node) => node.textContent).join(''),
        li.querySelector('.katex annotation')?.textContent ?? null, li.querySelectorAll('em, input').length])""") == [
        ["", " Compute ", "x_1 + 1", 0], ["✓", " Checked ", "y_1 *c*", 0]]


AUTOLINKS = """async (sources) => {
    const {renderChatMarkdown} = await import('/static/modules/chat_markdown.js');
    return sources.map((source) => {
        const host = document.createElement('div');
        host.innerHTML = renderChatMarkdown(source);
        const seen = {links: [...host.querySelectorAll('a')].map((a) => a.getAttribute('href'))};
        // A code block's language label and Copy button are its chrome, not the author's text: read them apart.
        const block = host.querySelector('.md-code-block');
        if (block) {
            const chrome = [...block.querySelectorAll('.md-code-language, [data-code-copy]')];
            seen.toolbar = chrome.map((el) => el.textContent);
            seen.code = block.querySelector('pre > code').textContent;
            chrome.forEach((el) => el.remove());
        }
        seen.text = host.textContent.trim();
        return seen;
    });
}"""


def test_an_angle_autolink_keeps_its_destination_and_other_angles_stay_text(reading_ui):
    page = open_chat(reading_ui, 1280)
    seen = page.evaluate(AUTOLINKS, [
        "See <https://example.com/docs> now",
        "Write <mailto:team@example.com>",
        "Keep <std::vector> and <b>bold</b> and <javascript:alert(1)>",
        "Code `<https://example.com/x>` stays code",
        # Math-like code is code: tilde and indented blocks keep the address and entity literal.
        "~~~\n$$ <https://example.com/t> &amp; \\(x\\) $$\n~~~",
        "    $$ <https://example.com/i> &amp; $$",
        "Math $$ <https://example.com/m> $$ is text",
        # An escaped `<` reads as `<`; code and parked math keep the author's backslash.
        "Escaped \\<b>bold\\</b>, a\\\\<i>, `\\<x>` and $$a \\< b$$",
        "~~~\n\\<div> \\\\<p>\n~~~",
    ])
    assert seen == [
        {"text": "See https://example.com/docs now", "links": ["https://example.com/docs"]},
        {"text": "Write mailto:team@example.com", "links": ["mailto:team@example.com"]},
        {"text": "Keep <std::vector> and <b>bold</b> and <javascript:alert(1)>", "links": []},
        {"text": "Code <https://example.com/x> stays code", "links": []},
        {"text": "$$ <https://example.com/t> &amp; \\(x\\) $$", "links": [], "toolbar": ["text", "Copy"],
         "code": "$$ <https://example.com/t> &amp; \\(x\\) $$\n"},
        {"text": "$$ <https://example.com/i> &amp; $$", "links": [], "toolbar": ["text", "Copy"],
         "code": "$$ <https://example.com/i> &amp; $$\n"},
        {"text": "Math $$ <https://example.com/m> $$ is text", "links": []},
        {"text": "Escaped <b>bold</b>, a\\<i>, \\<x> and $$a \\< b$$", "links": []},
        {"text": "\\<div> \\\\<p>", "links": [], "toolbar": ["text", "Copy"], "code": "\\<div> \\\\<p>\n"},
    ], seen


MULTILINE_CODE = """async (sources) => {
    const {mountChatMarkdown, enhanceChatMarkdown} = await import('/static/modules/chat_markdown.js');
    return sources.map((source) => {
        const host = document.createElement('div');
        document.body.append(host);
        mountChatMarkdown(host, source);
        const dispose = enhanceChatMarkdown(host);
        const seen = {code: [...host.querySelectorAll('code')].map((el) => el.textContent),
            tex: [...host.querySelectorAll('.katex annotation')].map((el) => el.textContent),
            marks: [...host.querySelectorAll('a, img, em, pre, .md-image-ref')].map((el) => el.tagName)};
        dispose();
        host.remove();
        return seen;
    });
}"""


def test_a_code_span_across_lines_keeps_its_bytes_and_the_math_after_it(reading_ui):
    page = open_chat(reading_ui, 1280)
    # Each span holds an unmatched opening delimiter; KaTeX reads only the prose math after it.
    seen = page.evaluate(MULTILINE_CODE, [
        "Run `echo $$ <https://images.example/c> &amp;\nnext` then $$x_1 *c*$$",
        "> Run ```a $$ `` ![i](https://images.example/q.png)\n> next```` $$ end``` then $$y_1 *c*$$",
        "- Run ``a \\[ `\n  b`` then \\[z_1 *c*\\]",
        # A ``` that cannot open a fence leaves the math after it to KaTeX alone.
        "Mention ``` literally; compute $$x_1 *c* + \\text{`q'}$$",
    ])
    assert seen == [
        {"code": ["echo $$ <https://images.example/c> &amp; next"], "tex": ["x_1 *c*"], "marks": []},
        {"code": ["a $$ `` ![i](https://images.example/q.png) next```` $$ end"], "tex": ["y_1 *c*"], "marks": []},
        {"code": ["a \\[ ` b"], "tex": ["z_1 *c*"], "marks": []},
        {"code": [], "tex": ["x_1 *c* + \\text{`q'}"], "marks": []},
    ], seen
    assert not [url for url in reading_ui["foreign"] if "images.example" in url], reading_ui["foreign"]


def test_compact_consumers_show_the_same_code(reading_ui):
    ui, engine = reading_ui, reading_ui["engine"]
    page = open_chat(ui, 1280)
    review = page.locator("#chat-messages .chat-bubble.system", has_text="Skill review: demo").first
    review.locator("[data-skill-review-toggle]").click()
    full = review.locator(".skill-review-full")
    full.wait_for(state="visible")
    assert full.locator("pre > code").evaluate_all("nodes => nodes.map(el => [el.textContent, el.children.length])") == [
        [FENCE_BODY + "\n", 0], [COMPACT_LINE + "\n", 0],
    ]
    assert full.locator("pre em, pre strong, pre .md-li, pre .md-h1, pre a, pre table").count() == 0
    review.scroll_into_view_if_needed()
    capture(page, f"reading-skill-review-{engine}")
    # The task-card timeline builds its rows with the same exported builder the live card uses.
    timeline = page.evaluate("""async (body) => {
        const {buildTimelineItemHtml} = await import('/static/modules/chat_activity.js');
        const host = document.createElement('div');
        host.className = 'chat-live-card';
        host.innerHTML = buildTimelineItemHtml({phase: 'working', headline: 'Checking the fence', body, lineKey: 'fence'},
            {expandedLineKeys: new Set(), groupId: 'reading'});
        document.querySelector('#chat-messages').append(host);
        const code = host.querySelector('.chat-live-line-body pre > code');
        return [code.textContent, code.children.length];
    }""", f"Output:\n```c#\n{FENCE_BODY}\n```\nDone")
    assert timeline == [FENCE_BODY + "\n", 0]


# Compact Markdown reaches the page with no sanitizer after the renderer, so code
# written inside a link destination must stay code and never become attribute text.
LINK_CODE = (
    '[open](https://example.com/```\n" onclick="window.__linkCode = 1" data-x="\n```)\n'
    '![pic](https://example.com/`x" onmouseover="window.__linkCode = 2`) and [docs](https://example.com/d)'
)


def test_compact_link_destination_keeps_code_out_of_attributes(reading_ui):
    page = open_chat(reading_ui, 1280)
    seen = page.evaluate("""async (body) => {
        const {buildTimelineItemHtml} = await import('/static/modules/chat_activity.js');
        const {renderMarkdown} = await import('/static/modules/utils.js');
        const row = document.createElement('div');
        row.className = 'chat-live-card';
        row.innerHTML = buildTimelineItemHtml({phase: 'working', headline: 'Link code', body, lineKey: 'link-code'},
            {expandedLineKeys: new Set(), groupId: 'link-code'});
        const plain = document.createElement('div');
        plain.innerHTML = renderMarkdown(body);
        document.querySelector('#chat-messages').append(row, plain);
        const hosts = [row.querySelector('.chat-live-line-body'), plain];
        for (const el of hosts.flatMap((host) => [...host.querySelectorAll('*')])) {
            el.dispatchEvent(new MouseEvent('mouseover', {bubbles: true}));
            el.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true}));
        }
        return hosts.map((host) => ({
            injected: [...host.querySelectorAll('*')].flatMap((el) => [...el.attributes].map((a) => a.name))
                .filter((name) => name.startsWith('on') || name === 'data-x'),
            links: [...host.querySelectorAll('a')].map((a) => [a.getAttribute('href'), a.textContent]),
            code: [...host.querySelectorAll('pre > code, code.inline-code')].map((c) => c.textContent),
            fired: window.__linkCode ?? null,
        }));
    }""", LINK_CODE)
    for host in seen:
        assert host == {
            "injected": [],
            "links": [["https://example.com/d", "docs"]],
            "code": ['" onclick="window.__linkCode = 1" data-x="\n', 'x" onmouseover="window.__linkCode = 2'],
            "fired": None,
        }, host


# A compact image or link destination is the address alone: a title is not part
# of it, balanced parentheses are, and a refused address leaves nothing behind.
COMPACT_REFERENCES = (
    'Before ![diagram](https://images.example/a.png "Architecture") after.\n'
    "![chart](https://images.example/Foo_(bar).png) and [wiki](https://example.com/wiki/Foo_(bar)).\n"
    "![unsafe](javascript:alert(1)) end"
)


def test_compact_image_reference_is_its_address_alone(reading_ui):
    ui = reading_ui
    page = open_chat(ui, 1280)
    seen = page.evaluate("""async (body) => {
        const {buildTimelineItemHtml} = await import('/static/modules/chat_activity.js');
        const {renderMarkdown} = await import('/static/modules/utils.js');
        const row = document.createElement('div');
        row.className = 'chat-live-card';
        row.innerHTML = buildTimelineItemHtml({phase: 'working', headline: 'References', body, lineKey: 'refs'},
            {expandedLineKeys: new Set(), groupId: 'refs'});
        const plain = document.createElement('div');
        plain.innerHTML = renderMarkdown(body);
        document.querySelector('#chat-messages').append(row, plain);
        return [row.querySelector('.chat-live-line-body'), plain].map((host) => ({
            text: host.textContent,
            links: [...host.querySelectorAll('a')].map((a) => [a.getAttribute('href'), a.textContent, a.className]),
            media: host.querySelectorAll('img, picture, video, audio, source').length,
        }));
    }""", COMPACT_REFERENCES)
    for host in seen:
        assert host == {
            "text": "Before Image: diagram after.\nImage: chart and wiki.\nImage: unsafe end",
            "links": [
                ["https://images.example/a.png", "Image: diagram", "md-link md-image-ref"],
                ["https://images.example/Foo_(bar).png", "Image: chart", "md-link md-image-ref"],
                ["https://example.com/wiki/Foo_(bar)", "wiki", "md-link"],
                ["#", "Image: unsafe", "md-link md-image-ref"],
            ],
            "media": 0,
        }, host
    assert not [url for url in ui["foreign"] if "images.example" in url], ui["foreign"]


def assert_column_bound(page, column):
    """`createChatInstance` binds its own column's compact Markdown (chat.js); the test adds no binding."""
    assert page.evaluate("(column) => document.querySelector(column).hasAttribute('data-md-compact')", column), column


COMPACT = """(column) => {
    const root = document.querySelector(column);
    const style = (el) => getComputedStyle(el);
    const review = [...root.querySelectorAll('.chat-bubble.system')].find((b) => b.textContent.includes('Skill review: demo'));
    const full = review.querySelector('.skill-review-full');
    const label = (el) => ({size: parseFloat(style(el).fontSize), weight: Number(style(el).fontWeight)});
    const table = (wrap) => ({
        head: wrap.querySelector('th').textContent,
        fadeClass: wrap.classList.contains('scroll-fade-x'),
        size: parseFloat(style(wrap.querySelector('table')).fontSize),
        overflow: wrap.scrollWidth - wrap.clientWidth,
        region: wrap.getAttribute('role'), label: wrap.getAttribute('aria-label'), tabindex: wrap.getAttribute('tabindex'),
        edges: [wrap.hasAttribute('data-scroll-start'), wrap.hasAttribute('data-scroll-end')],
    });
    const row = root.querySelector('.chat-live-card[data-reading-timeline] .chat-live-line');
    return {
        reviewText: parseFloat(style(full).fontSize),
        reviewHeadings: ['.md-h1', '.md-h2', '.md-h3'].map((cls) => [...full.querySelectorAll(cls)].map(label)).flat(),
        reviewTables: [...full.querySelectorAll('.md-table-wrap')].map(table),
        rowText: row && parseFloat(style(row.querySelector('.chat-live-line-body')).fontSize),
        rowHeading: row && label(row.querySelector('.chat-live-line-body .md-h2')),
        rowTables: row ? [...row.querySelectorAll('.md-table-wrap')].map(table) : [],
    };
}"""

# A timeline row is wider than a review body: its table needs more columns to overflow.
TIMELINE_BODY = f"## Measured\nThe run finished.\n{compact_wide_table(14)}\nDone"


def add_timeline_row(page, column):
    """A task card row written after the column was bound, built by the live card's own builder."""
    page.evaluate("""async ([column, body]) => {
        const {buildTimelineItemHtml} = await import('/static/modules/chat_activity.js');
        const host = document.createElement('div');
        host.className = 'chat-live-card';
        host.dataset.readingTimeline = '1';
        host.innerHTML = buildTimelineItemHtml({phase: 'done', headline: 'Timings', body, lineKey: 'timings'},
            {expandedLineKeys: new Set(), groupId: 'reading-timings'});
        document.querySelector(column).append(host);
    }""", [column, TIMELINE_BODY])


def assert_compact(metrics):
    # Density is retained: every compact heading level is a body-size semibold
    # label in a Skill Review report (its text is the bubble's 16px), and the
    # row's own size in a timeline.
    assert metrics["reviewText"] == 16, metrics
    assert len(metrics["reviewHeadings"]) == 4, metrics
    assert all(h == {"size": 14, "weight": 600} for h in metrics["reviewHeadings"]), metrics
    assert metrics["rowHeading"] == {"size": metrics["rowText"], "weight": 600}, metrics
    tables = {t["head"]: t for t in metrics["reviewTables"] + metrics["rowTables"]}
    assert set(tables) == {"Check", "Signal"} and len(metrics["rowTables"]) == 1, metrics
    for table in metrics["reviewTables"] + metrics["rowTables"]:
        # Compact tables keep --type-body; none is shrunk to fit.
        assert table["fadeClass"] and table["size"] == 14, table
        scrolls = table["overflow"] > 1
        assert (table["region"], table["tabindex"], table["edges"]) == (
            ("region", "0", [False, True]) if scrolls else (None, None, [False, False])), table
        if scrolls:
            assert table["label"] == "Scrollable table", table
    assert tables["Check"]["overflow"] <= 1, tables
    assert all(t["overflow"] > 1 for t in metrics["reviewTables"] + metrics["rowTables"] if t["head"] == "Signal"), metrics


PATCHED_TABLE = """(body) => new Promise((resolve) => {
    const frames = (n, then) => (n ? requestAnimationFrame(() => frames(n - 1, then)) : then());
    const wrap = () => document.querySelector('[data-patched-timeline] .md-table-wrap');
    const state = () => ({
        region: wrap().getAttribute('role'), label: wrap().getAttribute('aria-label'),
        tabindex: wrap().getAttribute('tabindex'),
        edges: [wrap().hasAttribute('data-scroll-start'), wrap().hasAttribute('data-scroll-end')],
        focused: document.activeElement === wrap(),
    });
    const {record, renderer} = window.patchedTimeline;
    const kept = wrap();
    kept.focus();
    const before = state();
    // Same glyphs in another order: the cell's text changes, no size does.
    record.items[0].body = body;
    renderer.renderLiveCardTimeline(record);
    frames(3, () => resolve({before, after: state(), same: wrap() === kept}));
})"""


def test_keyed_timeline_patch_keeps_a_scrolling_table_a_keyboard_region(reading_ui):
    page = open_chat(reading_ui, 1280)
    column = "#chat-messages"
    assert_column_bound(page, column)
    body = TIMELINE_BODY
    page.evaluate("""async ([column, body]) => {
        const {buildTimelineItemHtml} = await import('/static/modules/chat_activity.js');
        const {createLiveCardTimelineRenderer} = await import('/static/modules/chat_render_batch.js');
        const card = document.createElement('div');
        card.className = 'chat-live-card';
        card.dataset.patchedTimeline = '1';
        card.dataset.expanded = '1';
        const timelineEl = document.createElement('div');
        timelineEl.className = 'chat-live-timeline';
        card.append(timelineEl);
        document.querySelector(column).append(card);
        const record = {root: card, timelineEl, expandedLineKeys: new Set(), groupId: 'patched', items: [
            {phase: 'done', headline: 'Timings', body, lineKey: 'timings'}]};
        // The live card's own keyed renderer (chat.js builds the same one).
        const renderer = createLiveCardTimelineRenderer({withStableViewport: (fn) => fn(), buildTimelineItemHtml});
        renderer.renderLiveCardTimeline(record);
        window.patchedTimeline = {record, renderer};
    }""", [column, body])
    page.wait_for_function("() => document.querySelector('[data-patched-timeline] .md-table-wrap')"
                           "?.getAttribute('role') === 'region'")
    swapped = body.replace("120,000.25", "120,000.52")
    assert swapped != body
    seen = page.evaluate(PATCHED_TABLE, swapped)
    region = {"region": "region", "label": "Scrollable table", "tabindex": "0", "edges": [False, True], "focused": True}
    assert seen == {"before": region, "after": region, "same": True}, seen
    wrap = page.locator(f"{column} [data-patched-timeline] .md-table-wrap")
    page.keyboard.press("ArrowRight", delay=300)
    page.wait_for_function("el => el.scrollLeft > 0 && el.hasAttribute('data-scroll-start')", arg=wrap.element_handle())


def open_review(page, column):
    review = page.locator(f"{column} .chat-bubble.system", has_text="Skill review: demo").first
    review.locator("[data-skill-review-toggle]").click()
    review.locator(".skill-review-full").wait_for(state="visible")
    return review


def wait_compact_regions(page, column):
    # The wrapper becomes a region once it is laid out (the review body was hidden,
    # the timeline row arrived after binding).
    page.wait_for_function("""(column) => {
        const wide = [...document.querySelectorAll(column + ' .md-table-wrap')]
            .filter((w) => !w.closest('.ui-rich-content') && w.querySelector('th')?.textContent === 'Signal');
        return wide.length === 2 && wide.every((w) => w.getAttribute('role') === 'region');
    }""", arg=column)


def test_compact_surfaces_keep_their_density_and_scroll_tables_by_keyboard(reading_ui):
    ui, engine = reading_ui, reading_ui["engine"]
    page = open_chat(ui, 1280)
    column = "#chat-messages"
    assert_column_bound(page, column)
    review = open_review(page, column)
    add_timeline_row(page, column)
    wait_compact_regions(page, column)
    settle_tables(page, column)
    assert_compact(page.evaluate(COMPACT, column))
    fitted_code, wide_code = review.locator('.skill-review-full pre').all()
    assert fitted_code.evaluate("el => el.scrollWidth - el.clientWidth") <= 1
    assert fitted_code.get_attribute('tabindex') is None
    assert fitted_code.get_attribute('data-scroll-end') is None
    assert wide_code.text_content() == COMPACT_LINE + '\n'
    assert wide_code.evaluate("""el => {
        const mask = node => getComputedStyle(node).maskImage;
        return el.scrollWidth > el.clientWidth && el.hasAttribute('data-scroll-end') && mask(el) !== 'none'
            && mask(el.parentElement) === 'none' && getComputedStyle(el.parentElement).backgroundColor !== 'rgba(0, 0, 0, 0)';
    }""")
    assert wide_code.get_attribute('aria-label') == 'Scrollable code'
    wide_code.scroll_into_view_if_needed()
    capture(page, f'compact-code-desktop-{engine}')
    review.locator('[data-skill-review-toggle]').focus()
    full = review.locator('.skill-review-full')
    page.keyboard.press('Tab')
    assert full.evaluate('el => document.activeElement === el')
    assert full.evaluate("el => el.getAttribute('role') === 'region' && el.scrollHeight > el.clientHeight")
    page.keyboard.press('ArrowDown', delay=300)
    page.wait_for_function("el => el.scrollTop > 0 && el.hasAttribute('data-scroll-above')", arg=full.element_handle())
    page.keyboard.press('Tab')
    assert wide_code.evaluate('el => document.activeElement === el')
    page.keyboard.press('ArrowRight', delay=300)
    page.wait_for_function("el => el.scrollLeft > 0 && el.hasAttribute('data-scroll-start')", arg=wide_code.element_handle())
    # The scroller's actual direction is used, not a direction inferred from the chat.
    wide_code.evaluate("el => { el.dir = 'rtl'; el.scrollLeft = 0; el.dispatchEvent(new Event('scroll')); }")
    page.wait_for_function("el => el.hasAttribute('data-scroll-rtl') && el.hasAttribute('data-scroll-end')", arg=wide_code.element_handle())
    page.keyboard.press('ArrowLeft', delay=300)
    page.wait_for_function("el => el.scrollLeft < 0 && el.hasAttribute('data-scroll-start')", arg=wide_code.element_handle())
    wide = review.locator(".md-table-wrap", has_text="Latency ms")
    wide.focus()
    assert wide.evaluate("el => document.activeElement === el")
    page.keyboard.press("ArrowRight", delay=300)
    page.wait_for_function("el => el.scrollLeft > 0 && el.hasAttribute('data-scroll-start')", arg=wide.element_handle())
    review.scroll_into_view_if_needed()
    capture(page, f"compact-desktop-skill-review-{engine}")
    page.locator(f"{column} .chat-live-card[data-reading-timeline]").scroll_into_view_if_needed()
    capture(page, f"compact-desktop-timeline-{engine}")

    panel = mount_project_panel(page)
    panel_column = f"{panel} .chat-messages"
    assert_column_bound(page, panel_column)
    open_review(page, panel_column)
    add_timeline_row(page, panel_column)
    wait_compact_regions(page, panel_column)
    settle_tables(page, panel_column)
    assert_compact(page.evaluate(COMPACT, panel_column))
    page.locator(f"{panel_column} .skill-review-full .md-table-wrap", has_text="Latency ms").scroll_into_view_if_needed()
    capture(page, f"compact-panel440-skill-review-{engine}")
    page.evaluate("(column) => { window.readingColumn = document.querySelector(column); }", panel_column)
    page.evaluate("window.readingPanel.destroy()")
    # destroy() releases the column's binding with the page (destroyChatMarkdown).
    assert page.evaluate("() => !readingColumn.isConnected && !readingColumn.hasAttribute('data-md-compact')")

    page = open_chat(ui, 390, 844)
    assert_column_bound(page, column)
    review = open_review(page, column)
    add_timeline_row(page, column)
    wait_compact_regions(page, column)
    settle_tables(page, column)
    assert_compact(page.evaluate(COMPACT, column))
    review.locator(".md-table-wrap", has_text="Check").scroll_into_view_if_needed()
    capture(page, f"compact-phone390-skill-review-{engine}")
    assert not ui["foreign"], ui["foreign"]


WIDE_RICH_MD = "\n\n".join([
    "```js\nconst wideCodeLine = '" + ("alpha_beta_gamma_" * 12) + "';\n```",
    "$$" + " + ".join(f"x_{{{i}}}^{{(k)}}" for i in range(1, 26)) + "$$",
    "```mermaid\ngraph LR\n  A[Responsive] --> B[Fits column]\n```",
    '```mermaid\n%%{init: {"flowchart": {"useMaxWidth": false}}}%%\ngraph LR\n  '
    + " --> ".join(f"Stage_{i}[Step {i} verification node]" for i in range(1, 8)) + "\n```",
    "```chart\n{\"broken\": " + ("\"wrapped word \" " * 12) + "\n```",
])
CHART_SOURCE = json.dumps({
    'type': 'bar',
    'data': {'labels': [f'Sample {i}' for i in range(32)], 'datasets': [
        {'label': 'long_metric_id_' * 12, 'data': list(range(32))},
    ]},
}, indent=2)


def test_rich_horizontal_blocks_rtl_folded_and_render_only_edges(reading_ui):
    page = open_chat(reading_ui, 390, 844)
    seen = page.evaluate("""async ([richMd, wideTable, chartSource]) => {
        const {mountChatMarkdown, enhanceChatMarkdown} = await import('/static/modules/chat_markdown.js');
        const {refreshProjectAnswerFold} = await import('/static/modules/project_answer.js');
        const {openSkillDetailSheet} = await import('/static/modules/skill_detail_sheet.js');
        const col = document.querySelector('#chat-messages');
        const waitFrames = (n = 4) => new Promise((r) => {
            const step = (k) => k ? requestAnimationFrame(() => step(k - 1)) : r();
            step(n);
        });
        const edge = (el) => ({
            fadeX: el.classList.contains('scroll-fade-x'),
            overflow: el.scrollWidth - el.clientWidth,
            role: el.getAttribute('role'), label: el.getAttribute('aria-label'),
            tab: el.getAttribute('tabindex'),
            edges: [el.hasAttribute('data-scroll-start'), el.hasAttribute('data-scroll-end'), el.hasAttribute('data-scroll-rtl')],
        });
        const unmasked = (el) => !el.classList.contains('scroll-fade-x')
            && ['none', ''].includes(getComputedStyle(el).webkitMaskImage || getComputedStyle(el).maskImage || '');

        // 1. Enhanced rich blocks + RTL table: inner scrollers carry .scroll-fade-x; trays/Copy/chart-error stay unmasked.
        const host = document.createElement('div');
        host.className = 'chat-bubble assistant';
        const msg = document.createElement('div');
        msg.className = 'message';
        host.append(msg); col.append(host);
        mountChatMarkdown(msg, richMd);
        const rtlHolder = document.createElement('div');
        rtlHolder.dir = 'rtl';
        mountChatMarkdown(rtlHolder, wideTable);
        msg.append(...rtlHolder.childNodes);
        msg.lastElementChild.dir = 'rtl';
        const disposeRich = enhanceChatMarkdown(msg);
        const deadline = performance.now() + 8000;
        while (msg.querySelectorAll('.md-mermaid-scroll').length < 2 && performance.now() < deadline) await waitFrames(2);
        await waitFrames(4);

        const pre = msg.querySelector('.md-code-block > pre');
        const math = msg.querySelector('.katex-display');
        const [fitMermaid, mermaidScroll] = [...msg.querySelectorAll('.md-mermaid-scroll')];
        const rtlWrap = msg.querySelector('.md-table-wrap[dir="rtl"]');
        const chartErr = msg.querySelector('.md-chart-error');
        const rtlBefore = edge(rtlWrap);
        rtlWrap.scrollLeft = -120;
        rtlWrap.dispatchEvent(new Event('scroll'));
        await waitFrames(2);
        const rtlScrolled = edge(rtlWrap);

        // 2. Folded Project mirror: both wide and short blocks keep tabindex="-1" (short geometry never strips fold-stop).
        const foldBubble = document.createElement('div');
        foldBubble.className = 'chat-bubble assistant project-answer';
        const foldMsg = document.createElement('div');
        foldMsg.className = 'message';
        foldBubble.append(foldMsg); col.append(foldBubble);
        mountChatMarkdown(foldMsg, Array.from({length: 14}, (_, i) => `Line ${i}`).join('\\n\\n')
            + '\\n\\n```txt\\nshort\\n```\\n\\n```txt\\n' + 'wide_token_'.repeat(18) + '\\n```');
        refreshProjectAnswerFold(foldBubble);
        await waitFrames(3);
        const disposeFold = enhanceChatMarkdown(foldMsg);
        await waitFrames(4);
        const [shortFoldPre, wideFoldPre] = [...foldMsg.querySelectorAll('.md-code-block > pre')];
        const foldTabs = [foldBubble.classList.contains('is-folded'), shortFoldPre.getAttribute('tabindex'),
            shortFoldPre.hasAttribute('data-fold-stop'), wideFoldPre.getAttribute('tabindex')];

        const chartMd = '\\n\\n```chart\\n' + chartSource + '\\n```';
        // 3. Read the same valid chart through an actual chart mount and the real About sheet.
        // The sheet shares Place History's render-only path: it exposes source and never charts it.
        const graph = document.createElement('div');
        col.append(graph);
        mountChatMarkdown(graph, chartMd);
        const disposeGraph = enhanceChatMarkdown(graph);
        openSkillDetailSheet({item: {name: 'Throughput', source_id: 'builtin'},
            readDetail: async () => ({about: wideTable + '\\n\\n' + richMd + chartMd})});
        await waitFrames(4);
        const ro = document.querySelector('.skills-sheet-about');
        const source = [...ro.querySelectorAll('.md-chart')].at(-1);
        const range = document.createRange();
        range.selectNodeContents(source);
        const text = range.getBoundingClientRect();
        const box = source.getBoundingClientRect();
        const canvas = graph.querySelector('canvas');
        const active = canvas?.parentElement;
        const roState = {
            marked: ro.hasAttribute('data-md-rendered'),
            table: edge(ro.querySelector('.md-table-wrap')),
            code: edge(ro.querySelector('.md-code-block > pre')),
            mermaidSvg: ro.querySelectorAll('.md-mermaid svg, .md-chart canvas').length,
            mermaidOverflow: ro.querySelector('.md-mermaid').scrollWidth - ro.querySelector('.md-mermaid').clientWidth,
            mermaidUnmasked: unmasked(ro.querySelector('.md-mermaid')) && unmasked(ro.querySelector('.md-chart')),
            sourceText: range.toString(), sourceHeight: box.height, sourceScrollHeight: source.scrollHeight,
            sourceClientHeight: source.clientHeight, sourceOverflowX: source.scrollWidth - source.clientWidth,
            sourceFits: text.top >= box.top && text.bottom <= box.bottom && text.right <= box.right + 1,
            chart: active && {processed: active.dataset.processed, height: active.clientHeight,
                overflow: getComputedStyle(active).overflow, canvasHeight: canvas.clientHeight},
        };
        window.readingSkillAbout = ro;
        const result = {
            code: edge(pre), math: edge(math), fitDiagram: edge(fitMermaid), diagram: edge(mermaidScroll),
            traysUnmasked: unmasked(msg.querySelector('.md-code-block'))
                && unmasked(msg.querySelector('[data-code-copy]'))
                && [...msg.querySelectorAll('.md-mermaid')].every(unmasked)
                && unmasked(chartErr),
            chartWrap: getComputedStyle(chartErr).whiteSpace,
            rtlBefore, rtlScrolled, foldTabs, roState,
        };
        disposeRich(); disposeFold(); disposeGraph();
        host.remove(); foldBubble.remove(); graph.remove();
        return result;
    }""", [WIDE_RICH_MD, WIDE_TABLE, CHART_SOURCE])

    for key, label in (("code", "Scrollable code"), ("math", "Scrollable formula"), ("diagram", "Scrollable diagram")):
        block = seen[key]
        assert block["fadeX"] and block["overflow"] > 1, (key, block)
        assert (block["role"], block["label"], block["tab"], block["edges"]) == ("region", label, "0", [False, True, False]), block
    short_diagram = seen["fitDiagram"]
    short_overflows = short_diagram["overflow"] > 1
    assert short_diagram["role"] == ("region" if short_overflows else None), short_diagram
    assert short_diagram["edges"] == [False, short_overflows, False], short_diagram
    assert seen["traysUnmasked"] and seen["chartWrap"] == "pre-wrap", seen
    assert seen["rtlBefore"]["fadeX"] and seen["rtlBefore"]["overflow"] > 1, seen["rtlBefore"]
    assert seen["rtlBefore"]["edges"] == [False, True, True] and seen["rtlScrolled"]["edges"][0] and seen["rtlScrolled"]["edges"][2], seen
    assert seen["foldTabs"] == [True, "-1", True, "-1"], seen["foldTabs"]
    ro = seen["roState"]
    assert ro["marked"] and ro["mermaidSvg"] == 0 and ro["mermaidOverflow"] <= 1 and ro["mermaidUnmasked"], ro
    assert ro["table"]["role"] == "region" and ro["code"]["role"] == "region", ro
    assert ro["sourceText"].rstrip('\n') == CHART_SOURCE
    assert ro["sourceFits"] and ro["sourceOverflowX"] <= 1, ro
    assert ro["sourceHeight"] > 480 and ro["sourceScrollHeight"] <= ro["sourceClientHeight"] + 1, ro
    assert ro["chart"]["processed"] == "true" and ro["chart"]["overflow"] == "hidden", ro
    assert 0 < ro["chart"]["canvasHeight"] <= ro["chart"]["height"] < ro["sourceHeight"], ro
    body = page.locator('.skills-detail-sheet [data-sheet-body]')
    body.evaluate('el => { el.scrollTop = el.scrollHeight; }')
    capture(page, f"skill-about-chart-end-phone390-{reading_ui['engine']}")
    page.locator('.skills-detail-sheet [data-sheet-close]').click()
    assert page.evaluate('() => !readingSkillAbout.isConnected && !readingSkillAbout.hasAttribute("data-md-rendered")')


LINK_TABLE = "\n".join([
    "| Signal | " + " | ".join(f"Month {n}" for n in range(1, 9)) + " | Source |",
    "| --- | " + " | ".join("---:" for _ in range(8)) + " | --- |",
    "| [first](https://example.com/first) | " + " | ".join(f"{120_000 + n * 3_457:,}.25" for n in range(8))
    + " | [report](https://example.com/report) |",
])
LINK_EDGE = """() => {
    const wrap = document.querySelector('[data-link-case] .md-table-wrap');
    const link = wrap.querySelectorAll('a.md-link')[1];
    const w = wrap.getBoundingClientRect(), l = link.getBoundingClientRect();
    const fade = parseFloat(getComputedStyle(wrap).getPropertyValue('--scroll-fade-inline-edge')) || 24;
    return {scrollLeft: wrap.scrollLeft, end: wrap.hasAttribute('data-scroll-end'),
            under: l.right > w.right - fade,
            clear: !wrap.hasAttribute('data-scroll-end') || l.right <= w.right - fade + 1};
}"""


def test_table_link_tab_clears_the_horizontal_fade(reading_ui):
    page = open_chat(reading_ui, 390, 844)
    page.evaluate("""async (md) => {
        const {mountChatMarkdown, enhanceChatMarkdown} = await import('/static/modules/chat_markdown.js');
        const host = document.createElement('div');
        host.className = 'chat-bubble assistant';
        host.dataset.linkCase = '1';
        const msg = document.createElement('div');
        msg.className = 'message';
        host.append(msg);
        document.querySelector('#chat-messages').append(host);
        mountChatMarkdown(msg, md);
        window.__linkDispose = enhanceChatMarkdown(msg);
    }""", LINK_TABLE)
    wrap = page.locator('[data-link-case] .md-table-wrap')
    page.wait_for_function("el => el.getAttribute('role') === 'region'", arg=wrap.element_handle())
    before = page.evaluate(LINK_EDGE)
    assert before["under"] and before["end"], before
    # Explicit tab stops keep the descendant links reachable by keyboard in both engines for this synthetic case.
    page.evaluate("() => document.querySelectorAll('[data-link-case] a.md-link').forEach((a) => { a.tabIndex = 0; })")
    wrap.locator('a.md-link').first.focus()
    page.keyboard.press('Tab')
    page.wait_for_function("() => document.activeElement === document.querySelectorAll('[data-link-case] a.md-link')[1]")
    page.wait_for_function("""() => {
        const wrap = document.querySelector('[data-link-case] .md-table-wrap');
        const link = wrap.querySelectorAll('a.md-link')[1];
        const fade = parseFloat(getComputedStyle(wrap).getPropertyValue('--scroll-fade-inline-edge')) || 24;
        return !wrap.hasAttribute('data-scroll-end')
            || link.getBoundingClientRect().right <= wrap.getBoundingClientRect().right - fade + 1;
    }""")
    after = page.evaluate(LINK_EDGE)
    assert after["scrollLeft"] > before["scrollLeft"], (before, after)
    assert after["clear"], (before, after)
    capture(page, f"markdown-link-tab-{reading_ui['engine']}")
    page.evaluate("() => { window.__linkDispose(); document.querySelector('[data-link-case]').remove(); }")
    assert not reading_ui["foreign"], reading_ui["foreign"]


CODE_BODY = f"```\n{COMPACT_LINE}\n```"
PATCH_CODE_TO_TABLE = """(body) => new Promise((resolve) => {
    const frames = (n, then) => (n ? requestAnimationFrame(() => frames(n - 1, then)) : then());
    const card = document.querySelector('[data-code-to-table]');
    const kept = card.querySelector('.chat-live-line-body');
    const before = {codeRegion: card.querySelector('.md-compact-code')?.getAttribute('role'), kept: Boolean(kept)};
    const {record, renderer} = window.codeToTable;
    record.items[0].body = body;
    renderer.renderLiveCardTimeline(record);
    frames(3, () => {
        const wrap = card.querySelector('.md-table-wrap');
        resolve({before, bodySame: card.querySelector('.chat-live-line-body') === kept, connected: kept?.isConnected,
            region: wrap?.getAttribute('role'), label: wrap?.getAttribute('aria-label'),
            tabindex: wrap?.getAttribute('tabindex'),
            edges: wrap ? [wrap.hasAttribute('data-scroll-start'), wrap.hasAttribute('data-scroll-end')] : null,
            codeGone: !card.querySelector('.md-compact-code')});
    });
})"""


def test_keyed_timeline_code_to_table_reuses_the_outer_div(reading_ui):
    page = open_chat(reading_ui, 1280)
    column = "#chat-messages"
    assert_column_bound(page, column)
    page.evaluate("""async ([column, body]) => {
        const {buildTimelineItemHtml} = await import('/static/modules/chat_activity.js');
        const {createLiveCardTimelineRenderer} = await import('/static/modules/chat_render_batch.js');
        const card = document.createElement('div');
        card.className = 'chat-live-card';
        card.dataset.codeToTable = '1';
        card.dataset.expanded = '1';
        const timelineEl = document.createElement('div');
        timelineEl.className = 'chat-live-timeline';
        card.append(timelineEl);
        document.querySelector(column).append(card);
        const record = {root: card, timelineEl, expandedLineKeys: new Set(), groupId: 'code-to-table', items: [
            {phase: 'done', headline: 'Timings', body, lineKey: 'timings'}]};
        const renderer = createLiveCardTimelineRenderer({withStableViewport: (fn) => fn(), buildTimelineItemHtml});
        renderer.renderLiveCardTimeline(record);
        window.codeToTable = {record, renderer};
    }""", [column, CODE_BODY])
    page.wait_for_function("() => document.querySelector('[data-code-to-table] .md-compact-code')"
                           "?.getAttribute('role') === 'region'")
    seen = page.evaluate(PATCH_CODE_TO_TABLE, TIMELINE_BODY)
    assert seen == {
        "before": {"codeRegion": "region", "kept": True},
        "bodySame": True, "connected": True,
        "region": "region", "label": "Scrollable table", "tabindex": "0",
        "edges": [False, True], "codeGone": True,
    }, seen
    wrap = page.locator(f"{column} [data-code-to-table] .md-table-wrap")
    wrap.focus()
    page.keyboard.press("ArrowRight", delay=300)
    page.wait_for_function("el => el.scrollLeft > 0 && el.hasAttribute('data-scroll-start')", arg=wrap.element_handle())
    capture(page, f"markdown-keyed-code-to-table-{reading_ui['engine']}")