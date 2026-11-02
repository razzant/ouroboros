"""Browser smoke test for rich chat markdown rendering and security."""

from __future__ import annotations

import textwrap

import pytest

from ouroboros import dialogue_store
from ouroboros.test_evidence import diagnostic_directory

from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401 - pytest fixture import
from tests.ui_console_errors import record_console_errors, record_failed_loads, unexpected_console_errors


@pytest.mark.ui_browser
def test_ui_browser_marked_rich_markdown_and_security(direct_server_with_data):  # noqa: F811
    """Rich chat markdown renders every supported format without activating HTML."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import expect, sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    mega_message = textwrap.dedent(
        r"""
        # Rich markdown smoke
        ## Structure
        ### Details
        #### Fourth level

        > A quoted conclusion.

        | Name | Value |
        | --- | ---: |
        | Alpha | 1 |
        | Beta | 2 |

        - [x] Verified
        - [ ] Pending

        1. Ordered parent
           - Nested unordered child
        2. Ordered sibling

        ```javascript
        const answer = 42;
        ```

        $$x^2 + y^2 = z^2$$

        $$
        E = mc^2
        $$

        \[
        F = ma
        \]

        Inline math \(a+b=c\).

        Inline code: `Vec<T> & value`.

        Prices stay literal: $5 now and $10 later.

        ```mermaid
        graph TD
            A[Start $$E=mc^2$$] --> B[Done]
            B --> C[Unsafe editor]
            B --> D[Unsafe file]
            click A "https://example.com"
            click C "vscode://x"
            click D "file:///x"
        ```

        ```chart
        {"type":"bar","data":{"labels":["$$Revenue$$","B"],"datasets":[{"label":"Count","data":[1,2]}]},"options":{"responsive":false,"maintainAspectRatio":true,"animation":false}}
        ```
        """
    ).strip()
    security_message = (
        "Security literal: <script>window.__markdownScriptRan = true</script> "
        "<img src=x onerror=\"window.__markdownOnerrorRan = true\"> "
        "<details>detail</details> <mark>mark</mark> <sub>sub</sub>"
    )
    rows = [
        {"ts": "2026-08-30T10:00:00+00:00", "direction": "out", "chat_id": 1,
         "text": mega_message, "format": "markdown"},
        {"ts": "2026-08-30T10:00:01+00:00", "direction": "out", "chat_id": 1,
         "text": security_message, "format": "markdown"},
        {"ts": "2026-08-30T10:00:02+00:00", "direction": "out", "chat_id": 1,
         "text": "First spacing paragraph.\n\nSecond spacing paragraph.", "format": "markdown"},
        {"ts": "2026-08-30T10:00:03+00:00", "direction": "in", "chat_id": 1,
         "text": "User line one\nUser line two", "format": "markdown"},
        {"ts": "2026-08-30T10:00:04+00:00", "direction": "out", "chat_id": 1,
         "text": "Oversized mermaid test\n\n```mermaid\ngraph TD\nA[" + ("x" * 32769) + "]\n```",
         "format": "markdown"},
    ]
    # The running server reads its own dialogue store (#1982), never a log file written after it started.
    for row in rows:
        dialogue_store.append(data_dir, "chat", row)

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1280, "height": 800})
            page = context.new_page()
            console_errors = record_console_errors(page)
            failed_loads = record_failed_loads(page)
            page.add_init_script(
                """(() => {
                    window.__copiedCode = '';
                    window.__autoRenderKatexInRichBlocks = [];
                    let renderMathImpl;
                    Object.defineProperty(window, 'renderMathInElement', {
                        configurable: true,
                        get: () => renderMathImpl,
                        set: (fn) => {
                            renderMathImpl = (root, options) => {
                                fn(root, options);
                                window.__autoRenderKatexInRichBlocks.push(root.querySelectorAll(
                                    '.md-mermaid .katex, .md-chart .katex'
                                ).length);
                            };
                        },
                    });
                    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: {
                        writeText: async (text) => { window.__copiedCode = text; },
                    }});
                })()"""
            )
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                rich = page.locator(".chat-bubble.assistant", has_text="Rich markdown smoke").first
                rich.wait_for(state="visible", timeout=30_000)
                assert rich.locator("h1.md-h1").count() == 1, rich.inner_html()
                assert rich.locator("h2.md-h2").count() == 1
                assert rich.locator("h3.md-h3").count() == 1
                # The chat reading ladder (DESIGN §5): # = ## above ###, deeper levels at the
                # 16px reading size, all semibold; h4+ carry the smallest label class.
                h4 = rich.locator("h4.md-h3")
                assert h4.count() == 1, rich.inner_html()
                assert h4.evaluate("el => [getComputedStyle(el).fontSize, getComputedStyle(el).fontWeight]") == ["16px", "600"]
                assert rich.locator("h1.md-h1").evaluate("el => getComputedStyle(el).fontSize") == "20px"
                assert rich.locator("h3.md-h3").evaluate("el => getComputedStyle(el).fontSize") == "18px"
                assert rich.locator("blockquote.md-quote").count() == 1
                table = rich.locator(".md-table-wrap > table.md-table")
                assert table.count() == 1
                assert table.locator("tr").count() == 3
                assert table.locator("tr").first.locator("th").count() == 2
                assert rich.locator("input[type=checkbox]").count() == 0
                glyphs = rich.locator(".md-checkbox")
                assert glyphs.count() == 2
                assert glyphs.nth(0).inner_text() == "✓"
                assert glyphs.nth(1).inner_text() == ""
                assert rich.locator("ol ul").count() == 1

                code_block = rich.locator(".md-code-block:has(code.language-javascript)")
                assert code_block.locator(".md-code-language").inner_text() == "javascript"
                code_block.locator("[data-code-copy]").click()
                page.wait_for_function("() => window.__copiedCode.includes('const answer = 42;')")

                copy_button = code_block.locator("[data-code-copy]")
                page.evaluate(
                    "() => { Object.defineProperty(navigator, 'clipboard', { configurable: true, "
                    "value: undefined }); Object.defineProperty(document, 'execCommand', { "
                    "configurable: true, value: () => false }); }"
                )
                copy_button.click()
                page.wait_for_function(
                    "button => button.textContent === 'Copy failed'", arg=copy_button.element_handle(),
                )
                assert page.locator("body > textarea").count() == 0
                copy_button.evaluate("button => { button.textContent = 'Copy'; }")
                page.evaluate(
                    "() => { Object.defineProperty(document, 'execCommand', { configurable: true, "
                    "value: undefined }); }"
                )
                copy_button.click()
                page.wait_for_function(
                    "button => button.textContent === 'Copy failed'", arg=copy_button.element_handle(),
                )
                assert page.locator("body > textarea").count() == 0

                page.wait_for_function(
                    "(root) => root.querySelectorAll('.katex').length >= 2",
                    arg=rich.element_handle(), timeout=10_000,
                )
                assert rich.locator(".katex-display").count() == 3
                inline_code = rich.locator("code.inline-code", has_text="Vec<T> & value")
                assert inline_code.count() == 1
                assert inline_code.inner_text() == "Vec<T> & value"
                assert "&lt;" not in inline_code.inner_text() and "&amp;" not in inline_code.inner_text()
                money = rich.locator("p", has_text="$5 now and $10 later")
                assert money.locator(".katex").count() == 0
                diagram = rich.locator(".md-mermaid")
                diagram.locator("svg").wait_for(state="visible", timeout=30_000)
                assert "Syntax error" not in diagram.inner_text()
                assert "Start" in diagram.inner_text() and "Done" in diagram.inner_text()
                assert all(
                    count == 0
                    for count in page.evaluate("() => window.__autoRenderKatexInRichBlocks")
                )
                diagram_links = diagram.locator("a").evaluate_all(
                    "links => links.map(link => ({ href: link.getAttribute('href'), "
                    "xlink: link.getAttribute('xlink:href'), target: link.getAttribute('target'), "
                    "rel: link.getAttribute('rel') }))"
                )
                safe_links = [
                    link for link in diagram_links
                    if "https://example.com" in (link["href"] or link["xlink"] or "")
                ]
                assert len(safe_links) == 1, diagram_links
                assert safe_links[0]["target"] == "_blank"
                assert safe_links[0]["rel"] == "noopener noreferrer"
                assert all(
                    not (link["href"] or link["xlink"] or "").startswith(("vscode:", "file:"))
                    for link in diagram_links
                ), diagram_links
                assert page.locator("script#chat-mermaid-library").count() == 1
                chart_canvas = rich.locator(".md-chart canvas")
                assert chart_canvas.count() == 1
                assert rich.locator(".md-chart .katex").count() == 0
                assert chart_canvas.evaluate(
                    "canvas => { const chart = Chart.getChart(canvas); return {"
                    " labels: chart.data.labels, data: chart.data.datasets[0].data, "
                    "responsive: chart.options.responsive,"
                    " aspect: chart.options.maintainAspectRatio }; }"
                ) == {
                    "labels": ["$$Revenue$$", "B"], "data": [1, 2],
                    "responsive": True, "aspect": False,
                }

                spacing = page.locator(
                    ".chat-bubble.assistant", has_text="First spacing paragraph.",
                ).first
                spacing.wait_for(state="visible", timeout=30_000)
                paragraphs = spacing.locator(".message > p")
                assert paragraphs.count() == 2
                gap = paragraphs.evaluate_all(
                    "nodes => { const first = nodes[0].getBoundingClientRect(); "
                    "const second = nodes[1].getBoundingClientRect(); return { "
                    "actual: second.top - first.bottom, expected: "
                    "parseFloat(getComputedStyle(nodes[0]).marginBottom) }; }"
                )
                assert abs(gap["actual"] - gap["expected"]) < 1, gap

                user = page.locator(".chat-bubble.user", has_text="User line one").first
                user.wait_for(state="visible", timeout=30_000)
                user_message = user.locator(".message")
                assert user_message.evaluate("node => getComputedStyle(node).whiteSpace") == "pre-wrap"
                user_lines = user_message.evaluate(
                    "node => { const range = document.createRange(); range.selectNodeContents(node); "
                    "return new Set([...range.getClientRects()].map(rect => Math.round(rect.top))).size; }"
                )
                assert user_lines == 2

                oversized = page.locator(
                    ".chat-bubble.assistant", has_text="Oversized mermaid test",
                ).first
                oversized.wait_for(state="visible", timeout=30_000)
                oversized.locator(".md-mermaid-error").wait_for(state="visible", timeout=10_000)
                assert oversized.locator(".md-mermaid").count() == 0
                # Distant rows sleep under content-visibility and wake on the observer's next frame.
                diagnostic = oversized.locator(".md-diagram-error-note")
                diagnostic.scroll_into_view_if_needed()
                expect(diagnostic).to_have_text("Diagram could not be rendered.", use_inner_text=True)

                security = page.locator(".chat-bubble.assistant", has_text="Security literal").first
                security.wait_for(state="visible", timeout=30_000)
                assert security.locator("script, img, details, mark, sub, [onerror]").count() == 0
                assert page.evaluate("() => !window.__markdownScriptRan && !window.__markdownOnerrorRan")
                literal = security.locator(".message").inner_text()
                assert "<details>detail</details>" in literal
                assert "<mark>mark</mark>" in literal
                assert "<sub>sub</sub>" in literal
                rich.locator("h1.md-h1").scroll_into_view_if_needed()
                page.screenshot(path=str(diagnostic_directory(data_dir.parent, "rich-markdown") / "rich-markdown-top.png"), full_page=True)
                rich.locator(".md-chart").scroll_into_view_if_needed()
                page.screenshot(path=str(diagnostic_directory(data_dir.parent, "rich-markdown") / "rich-markdown-bottom.png"), full_page=True)
                assert unexpected_console_errors(console_errors, failed_loads) == []
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_ui_browser_mermaid_load_failure_retries(direct_server_with_data):  # noqa: F811
    """A failed Mermaid script is removed and a later diagram retries the load."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    attempts = 0

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()

            def route_mermaid(route):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    route.abort()
                else:
                    route.continue_()

            page.route("**/static/mermaid.min.js", route_mermaid)
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                page.evaluate(
                    """async () => {
                        const { enhanceChatMarkdown } = await import('/static/modules/chat_markdown.js');
                        const root = document.createElement('div');
                        root.className = 'chat-bubble assistant';
                        root.innerHTML = '<div class="message"><div class="md-mermaid">graph TD; A-->B</div></div>';
                        document.body.appendChild(root);
                        enhanceChatMarkdown(root);
                    }"""
                )
                page.locator(".md-mermaid-error").wait_for(state="visible", timeout=10_000)
                assert page.locator("script#chat-mermaid-library").count() == 0

                page.evaluate(
                    """async () => {
                        const { enhanceChatMarkdown } = await import('/static/modules/chat_markdown.js');
                        const root = document.createElement('div');
                        root.className = 'chat-bubble assistant';
                        root.innerHTML = '<div class="message"><div class="md-mermaid">graph TD; C-->D</div></div>';
                        document.body.appendChild(root);
                        enhanceChatMarkdown(root);
                    }"""
                )
                page.locator(".md-mermaid svg").wait_for(state="visible", timeout=30_000)
                assert attempts == 2
                assert page.locator("script#chat-mermaid-library").count() == 1
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise
