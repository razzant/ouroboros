"""Browser proof for the terminal Project completion projection."""

from __future__ import annotations

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401


def _store(root, stream, row):
    """A row the running server would have stored: its own dialogue store (#1982), never a log file written after
    it started."""
    from ouroboros import dialogue_store

    dialogue_store.append(root, stream, row)


def _wait_status(page, expected, timeout=10_000):
    page.wait_for_function(
        """(expected) => {
            const el = document.querySelector('#chat-status');
            return el && el.textContent.trim() === expected;
        }""",
        arg=expected,
        timeout=timeout,
    )


@pytest.mark.ui_browser
def test_ui_project_completion_pointer_keeps_project_history_scoped(direct_server_with_data):  # noqa: F811
    """Main gets one human pointer; Project keeps its detailed terminal history."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    from ouroboros.project_dialogue import append_chat_annotation
    from ouroboros.projects_registry import create_project

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    project = create_project(data_dir, "tower-defence", name="Tower Defence")
    project_chat = int(project["chat_id"])
    target_label = "Tower Defence › Fix nested delegation"
    task_id = "tower-root-1"
    _store(data_dir, "chat", {
        "ts": "2026-08-22T10:00:00+00:00", "direction": "out", "chat_id": 1,
        "user_id": 1, "text": f"{target_label} · Completed",
        "type": "project_completion_summary", "task_id": task_id,
        "project_id": project["id"], "project_name": project["name"],
        "target_label": target_label, "status": "completed",
    })
    _store(data_dir, "chat", {
        "ts": "2026-08-22T09:58:00+00:00", "direction": "in", "chat_id": 1,
        "user_id": 1, "text": "Continue the Tower Defence task",
        "client_message_id": "route-to-tower",
    })
    append_chat_annotation(data_dir, "route-to-tower", action="route_to_project",
                           target=project["id"], target_label=target_label, status="delivered")
    _store(data_dir, "progress", {
        "ts": "2026-08-22T09:59:00+00:00", "type": "send_message", "direction": "out",
        "chat_id": project_chat, "task_id": task_id, "is_progress": True,
        "content": "Project progress: nested delegation is running.",
    })
    _store(data_dir, "chat", {
        "ts": "2026-08-22T10:00:01+00:00", "direction": "system", "chat_id": project_chat,
        "user_id": 1, "text": "Project task summary", "type": "task_summary",
        "task_id": task_id, "tool_calls": 1, "rounds": 2, "status": "completed",
    })

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                _wait_status(page, "Online", timeout=30_000)
                summary = page.locator('#chat-messages .chat-bubble[data-system-type="project_completion_summary"]')
                summary.wait_for(state="visible", timeout=30_000)
                summary_text = summary.inner_text()
                assert target_label in summary_text
                # The Main row is a pointer: one status line (and a cause sentence when there is one),
                # never the answer excerpt (owner Q5=A) and no invitation: its Project reference is the door.
                assert summary.locator(".message").inner_text().strip() == f"{target_label} · Completed"
                assert "Release shipped." not in summary_text
                # The row points at its Project with the one reference: the Project's name, never a
                # button of its own and never the raw id.
                reference = summary.locator('[data-intent="open-project"]')
                assert reference.count() == 1
                assert reference.get_attribute("aria-label") == f"Open project {project['name']}"
                assert "Open Project" not in summary_text
                assert project["id"] not in summary_text

                annotation = page.locator('#chat-messages .msg-routing-annotation').filter(has_text=target_label)
                annotation.wait_for(state="visible", timeout=10_000)
                assert project["id"] not in annotation.inner_text()
                main_text = page.locator("#chat-messages").inner_text()
                assert "Project progress: nested delegation is running." not in main_text
                assert "Project task summary" not in main_text

                reference.click()
                page.wait_for_selector("#project-panel:not([hidden])", timeout=30_000)
                assert page.locator("#project-panel-title").inner_text() == project["name"]
                panel = page.locator(f"#panel-pchat-{project['id']}")
                panel.wait_for(state="visible", timeout=30_000)
                task_card = panel.locator(f'.chat-live-card[data-task-id="{task_id}"]')
                task_card.wait_for(state="visible", timeout=10_000)
                assert task_card.locator(".chat-live-phase").inner_text().strip() == "Done"
                task_card.locator("[data-live-disclosure]").click()
                task_card.get_by_text(
                    "Project progress: nested delegation is running.",
                    exact=False,
                ).first.wait_for(state="visible", timeout=30_000)
                assert "Project progress: nested delegation is running." in task_card.inner_text()
                assert panel.locator('.chat-bubble[data-system-type="project_completion_summary"]').count() == 0
                assert panel.locator('.system-message-actions [data-intent="open-project"]').count() == 0
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


_LONG_ANSWER = (
    "**Done: release shipped.** PR #7 merged into `main`.\n\n"
    + "\n\n".join(f"Paragraph {i}: " + "detail " * 30 for i in range(1, 9))
)


@pytest.mark.ui_browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
@pytest.mark.parametrize("width", [1280, 390])
def test_ui_project_completion_mirror_is_an_ordinary_folded_message(direct_server_with_data, engine, width):  # noqa: F811
    """A root that ended with its own answer reaches Main as Ouroboros's message; others stay pointers."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    from ouroboros.projects_registry import create_project

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    project = create_project(data_dir, f"mirror-{engine}-{width}", name="A rather long Project name that must yield on a phone")
    base = {"direction": "system", "chat_id": 1, "user_id": 1, "type": "project_completion_summary",
            "project_id": project["id"], "project_name": project["name"], "status": "completed"}
    _store(data_dir, "chat", {
        **base, "ts": "2026-08-22T10:00:00+00:00", "task_id": "root-long",
        "text": "P › Long · Done", "target_label": "P › Long",
        "completion_answer": _LONG_ANSWER})
    _store(data_dir, "chat", {
        **base, "ts": "2026-08-22T10:01:00+00:00", "task_id": "root-short",
        "text": "P › Short · Done", "target_label": "P › Short",
        "completion_answer": "All good, nothing to report."})
    _store(data_dir, "chat", {
        **base, "ts": "2026-08-22T10:02:00+00:00", "task_id": "root-pointer", "status": "failed",
        "text": "P › Broken · Failed", "target_label": "P › Broken"})

    try:
        with sync_playwright() as pw:
            browser = getattr(pw, engine).launch(headless=True)
            page = browser.new_page(viewport={"width": width, "height": 900})
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                _wait_status(page, "Online", timeout=30_000)
                rows = page.locator('#chat-messages .chat-bubble[data-system-type="project_completion_summary"]')
                rows.nth(2).wait_for(state="visible", timeout=30_000)
                long_row, short_row, pointer = rows.nth(0), rows.nth(1), rows.nth(2)

                # The answered endings speak in Ouroboros's voice and say nothing in the host's.
                for row in (long_row, short_row):
                    classes = row.get_attribute("class")
                    assert "assistant" in classes and "project-answer" in classes and "system" not in classes.split()
                    text = row.inner_text()
                    assert "Open the Project for details." not in text and "Open Project" not in text
                    assert row.locator(".sender").inner_text().strip() == "Ouroboros"
                    assert row.locator(".system-message-actions .chat-quiz-project").count() == 1
                assert "release shipped" in long_row.inner_text()
                # The fold: the long answer is clamped and faded, the short one is neither.
                page.wait_for_function(
                    "() => document.querySelectorAll('.chat-bubble.project-answer.is-folded').length === 1",
                    timeout=10_000)
                geometry = long_row.locator(".message").evaluate(
                    "n => ({client: n.clientHeight, scroll: n.scrollHeight})")
                assert geometry["scroll"] > geometry["client"] > 0
                assert "is-folded" not in short_row.get_attribute("class")
                # The whole answer is in the node: the fold is a viewport, never a cut.
                assert "Paragraph 8" in long_row.locator(".message").evaluate("n => n.textContent")
                # No horizontal overflow of the transcript, the chip yields instead.
                assert page.locator("#chat-messages").evaluate("n => n.scrollWidth <= n.clientWidth + 1")

                # The ending without an answer is still the System pointer, and it points at the
                # Project with the SAME control as the answered ones: a row's voice never picks it.
                assert "system" in pointer.get_attribute("class").split()
                assert pointer.locator(".message").inner_text().strip() == "P › Broken · Failed"
                shapes = [row.locator('[data-intent="open-project"]').evaluate(
                    "n => [n.className, n.getAttribute('aria-label'), [...n.children].map(c => c.className).join(' ')]")
                    for row in (long_row, pointer)]
                assert shapes[0] == shapes[1]

                # The chip names the Project and opens it.
                chip = long_row.locator(".chat-quiz-project")
                assert project["name"] in (chip.get_attribute("title") or "")
                chip.click()
                page.wait_for_selector("#project-panel:not([hidden])", timeout=30_000)
                assert page.locator("#project-panel-title").inner_text() == project["name"]
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_ui_project_completion_mirror_folds_after_mounting_on_a_hidden_page(direct_server_with_data, engine):  # noqa: F811
    """A row mounted while another page is open has no layout box; the fade arrives when Chat is shown."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    from ouroboros.projects_registry import create_project

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    project = create_project(data_dir, f"hidden-{engine}", name="Hidden mount")
    _store(data_dir, "chat", {
        "direction": "system", "chat_id": 1, "user_id": 1, "type": "project_completion_summary",
        "project_id": project["id"], "project_name": project["name"], "status": "completed",
        "ts": "2026-08-22T11:00:00+00:00", "task_id": f"root-hidden-{engine}",
        "text": "Hidden mount › Long · Done",
        "target_label": "Hidden mount › Long", "completion_answer": _LONG_ANSWER})

    try:
        with sync_playwright() as pw:
            browser = getattr(pw, engine).launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            try:
                page.goto(url + "#dashboard", wait_until="domcontentloaded", timeout=30_000)
                _wait_status(page, "Online", timeout=30_000)
                row = page.locator("#chat-messages .chat-bubble.project-answer")
                row.wait_for(state="attached", timeout=30_000)
                assert not page.locator("#page-chat").evaluate("n => n.classList.contains('active')")
                # No box, no measurement: the row is left unmarked rather than guessed.
                assert row.locator(".message").evaluate("n => n.clientHeight") == 0
                assert "is-folded" not in row.get_attribute("class")

                page.locator('[data-nav-page="chat"]').first.click()
                page.wait_for_function(
                    "() => document.querySelectorAll('.chat-bubble.project-answer.is-folded').length === 1",
                    timeout=10_000)
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
@pytest.mark.parametrize("width", [1280, 390])
def test_ui_project_completion_mirror_has_one_project_door(direct_server_with_data, engine, width):  # noqa: F811
    """The accepted replacement for Main's in-place expansion: a folded preview,
    Copy, factual caveat/time and one keyboard-operable Project reference."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    import os
    import re
    from pathlib import Path
    from playwright.sync_api import expect, sync_playwright
    from ouroboros.projects_registry import create_project

    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or direct_server_with_data["data_dir"] / "evidence") / f"mirror-{engine}-{width}"
    evidence.mkdir(parents=True, exist_ok=True)
    root = direct_server_with_data["data_dir"]
    project = create_project(root, f"mirror-door-{engine}-{width}", name="Long report")
    answer = _LONG_ANSWER + "\n\nThe [deep link](https://example.com/deep) and `TAIL_MARKER` end it."
    _store(root, "chat", {
        "direction": "system", "chat_id": 1, "user_id": 1, "type": "project_completion_summary",
        "project_id": project["id"], "project_name": project["name"], "status": "completed",
        "ts": "2026-08-22T12:00:00+00:00", "task_id": "mirror-door",
        "text": "Long report · Done with warnings",
        "completion_answer": answer, "completion_reason": "Windows verification was not performed.",
        "terminal_time": {"v": 1, "source": "executor_terminal", "occurred_at": "2026-08-22T11:59:00Z"}})
    tab = "Alt+Tab" if engine == "webkit" else "Tab"
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": 900})
        page.add_init_script("""window.__copied = []; Object.defineProperty(navigator, 'clipboard', {configurable: true,
            value: {writeText: async text => { window.__copied.push(text); }}});""")
        try:
            page.goto(direct_server_with_data["url"], wait_until="domcontentloaded")
            _wait_status(page, "Online", timeout=30_000)
            row = page.locator('#chat-messages .chat-bubble.project-answer[data-task-id="mirror-door"]')
            for stage in ("history", "reload"):
                expect(row).to_have_class(re.compile(r'\bis-folded\b'))
                row.scroll_into_view_if_needed()
                message = row.locator('.message')
                assert message.evaluate('n => n.scrollHeight > n.clientHeight && n.clientHeight > 0')
                assert 'TAIL_MARKER' in message.text_content()
                assert row.locator('[data-answer-fold], .project-answer-files').count() == 0
                assert row.get_by_role('button', name='Files', exact=True).count() == 0
                assert message.locator('[data-fold-stop]').count() == 1
                assert message.locator('a').evaluate_all('ns => ns.every(n => n.tabIndex === -1)')
                expect(row.locator('.project-answer-limitation')).to_have_text('Windows verification was not performed.')
                # The caveat is the one note; the row's own time is its one compact time line (owner 17.10, Q7 A):
                # it ended a minute before it was added, so both facts, the full instants in its title.
                assert row.locator(':scope > .msg-provenance').count() == 1
                assert re.fullmatch(r'Ended .+ · added .+', row.locator('.msg-time [aria-hidden="true"]').inner_text())
                assert (row.locator('.msg-time').get_attribute('title') or '').startswith('Task ended ')
                assert row.locator('.system-message-actions').locator('button').count() == 1
                row.screenshot(path=str(evidence / f'{stage}.png'))
                if stage == 'history':
                    row.locator('.chat-message-copy').click()
                    page.wait_for_function('() => window.__copied.length === 1')
                    assert page.evaluate('() => window.__copied[0]') == answer
                    page.reload(wait_until='domcontentloaded')
                    _wait_status(page, 'Online', timeout=30_000)
            # The hidden tail cannot receive Tab; the existing Project reference can.
            row.locator('.sender').click()
            project_ref = row.locator('[data-intent="open-project"]')
            for _ in range(8):
                page.keyboard.press(tab)
                assert not message.evaluate('n => n.contains(document.activeElement)'), 'focus reached clipped text'
                if project_ref.evaluate('n => n === document.activeElement'):
                    break
            expect(project_ref).to_be_focused()
            page.keyboard.press('Enter')
            expect(page.locator('#project-panel-title')).to_have_text(project['name'])
        finally:
            browser.close()


@pytest.mark.ui_browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_ui_folded_completion_excludes_late_table_focus(direct_server_with_data, engine):  # noqa: F811
    """Table enhancement/resize must not create a keyboard door into clipped Main text.
    The identical ordinary Project table remains a named, keyboard-scrollable region.
    """
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    import json
    import os
    import re
    from pathlib import Path
    from playwright.sync_api import expect, sync_playwright
    from ouroboros.projects_registry import create_project

    root = direct_server_with_data["data_dir"]
    project = create_project(root, f"table-focus-{engine}", name="Table report")
    table = ("| " + " | ".join(f"Column {i}" for i in range(8)) + " |\n"
             + "| " + " | ".join("---" for _ in range(8)) + " |\n"
             + "\n".join("| " + " | ".join("wide-cell-" + "X" * 70 for _ in range(8)) + " |" for _ in range(4)))
    answer = _LONG_ANSWER + "\n\n" + table + "\n\nTABLE_FOCUS_TAIL"
    _store(root, "chat", {
        "direction": "system", "chat_id": 1, "user_id": 1, "type": "project_completion_summary",
        "project_id": project["id"], "project_name": project["name"], "status": "completed",
        "ts": "2026-08-22T12:00:00+00:00", "task_id": "table-focus",
        "text": "Table report · Done", "completion_answer": answer})
    _store(root, "chat", {
        "direction": "out", "chat_id": int(project["chat_id"]), "user_id": 1,
        "ts": "2026-08-22T11:59:00+00:00", "task_id": "table-focus", "text": answer})
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or root / "evidence") / f"table-focus-{engine}"
    evidence.mkdir(parents=True, exist_ok=True)
    observations = []
    tab = "Alt+Tab" if engine == "webkit" else "Tab"
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.add_init_script("""window.__copied = []; Object.defineProperty(navigator, 'clipboard', {configurable: true,
            value: {writeText: async text => { window.__copied.push(text); }}});""")
        try:
            page.goto(direct_server_with_data["url"], wait_until="domcontentloaded")
            _wait_status(page, "Online", timeout=30_000)
            row = page.locator('#chat-messages .project-answer[data-task-id="table-focus"]')
            expect(row).to_have_class(re.compile(r'\bis-folded\b'))
            message, tables = row.locator('.message'), row.locator('.md-table-wrap')
            # Wait for the real late ResizeObserver callback, not merely rendered Markdown.
            expect(tables.first).to_have_attribute('role', 'region')
            row.scroll_into_view_if_needed()
            for stage, width in (("initial", 1280), ("late-write", 1280), ("resize", 390)):
                if stage == "late-write":
                    # A renderer can append a table after the preview was measured. Exercise
                    # the production binding's existing mutation/resize path, not a mock.
                    message.evaluate('''n => {
                        const table = n.querySelector('.md-table-wrap').cloneNode(true);
                        for (const name of ['tabindex', 'role', 'aria-label', 'data-fold-stop', 'data-scroll-start', 'data-scroll-end']) table.removeAttribute(name);
                        n.appendChild(table);
                    }''')
                wrap = tables.last
                expect(wrap).to_have_attribute('role', 'region')
                page.set_viewport_size({"width": width, "height": 900})
                # Two frames settle both fold measurement and the table's own resize callback.
                page.evaluate('() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))')
                expect(row).to_have_class(re.compile(r'\bis-folded\b'))
                row.locator('.sender').click()
                project_ref = row.locator('[data-intent="open-project"]')
                hidden_focus = False
                for _ in range(12):
                    page.keyboard.press(tab)
                    state = message.evaluate('n => ({scroll: n.scrollTop, hiddenFocus: n.contains(document.activeElement), active: document.activeElement?.outerHTML})')
                    observations.append({"stage": stage, "width": width, "table_tabindex": wrap.get_attribute('tabindex'), **state})
                    hidden_focus |= state['hiddenFocus']
                    if project_ref.evaluate('n => n === document.activeElement'):
                        break
                row.screenshot(path=str(evidence / f'main-{stage}-{width}.png'))
                assert not hidden_focus, observations
                assert message.evaluate('n => n.scrollTop') == 0, observations
                assert wrap.get_attribute('tabindex') == '-1'
                expect(project_ref).to_be_focused()
            assert row.locator('[data-answer-fold], .project-answer-files').count() == 0
            row.locator('.chat-message-copy').click()
            page.wait_for_function('() => window.__copied.length === 1')
            assert page.evaluate('() => window.__copied[0]') == answer
            project_ref.click()
            panel = page.locator(f'#panel-pchat-{project["id"]}')
            expect(panel).to_be_visible()
            full_table = panel.locator('.chat-bubble.assistant .md-table-wrap')
            expect(full_table).to_have_attribute('tabindex', '0')
            expect(full_table).to_have_attribute('role', 'region')
            assert full_table.get_attribute('aria-label')
            # Enter by the real sequential-key path, not programmatic focus.
            panel.locator('.chat-bubble.assistant .sender').click()
            for _ in range(12):
                page.keyboard.press(tab)
                if full_table.evaluate('n => n === document.activeElement'):
                    break
            expect(full_table).to_be_focused()
            before = full_table.evaluate('n => n.scrollLeft')
            # WebKit scrolls focused regions while held, not on a zero-duration tap
            # (the same consumer contract as test_markdown_reading_browser.py).
            page.keyboard.press('ArrowRight', delay=300)
            page.wait_for_function('(n) => n.scrollLeft > 0', arg=full_table.element_handle())
            assert full_table.evaluate('n => n.scrollLeft') > before
            panel.screenshot(path=str(evidence / 'project-table.png'))
            page.locator('#project-panel-close').click()
            expect(panel).not_to_be_visible()
            # Change only the fixture's fold height: table width/height stay fixed.
            # This isolates syncFold -> MutationObserver restoration from table resize.
            # Pin the fixture table width: WebKit's page scrollbar otherwise
            # changes its width by six pixels when the message becomes taller.
            tables.first.evaluate('n => { n.style.minWidth = `${n.clientWidth}px`; n.style.maxWidth = "none"; }')
            page.evaluate('() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))')
            dimensions = tables.first.evaluate('n => [n.clientWidth, n.clientHeight]')
            # The fold is measured in lines (--project-answer-lines on the bubble, project_answer.js measureFold):
            # give the bubble enough lines for the whole answer, as a taller fold would.
            row.evaluate("""n => { const m = n.querySelector('.message');
                n.style.setProperty('--project-answer-lines', String(Math.ceil(m.scrollHeight / parseFloat(getComputedStyle(m).lineHeight)) + 1)); }""")
            page.evaluate("window.dispatchEvent(new Event('resize'))")
            expect(row).not_to_have_class(re.compile(r'\bis-folded\b'))
            expect(tables.first).to_have_attribute('tabindex', '0')
            assert tables.first.evaluate('n => [n.clientWidth, n.clientHeight]') == dimensions
            observations.append({'stage': 'unfold-without-table-resize', 'table_tabindex': tables.first.get_attribute('tabindex')})
        finally:
            (evidence / 'focus.json').write_text(json.dumps(observations, indent=2), encoding='utf-8')
            browser.close()


@pytest.mark.ui_browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_ui_folded_completion_excludes_delayed_diagram_fallback(direct_server_with_data, engine):  # noqa: F811
    """A late real Mermaid load failure cannot add a hidden keyboard stop."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    import re
    from playwright.sync_api import expect, sync_playwright
    from ouroboros.projects_registry import create_project

    root = direct_server_with_data["data_dir"]
    project = create_project(root, f"late-diagram-{engine}", name="Diagram report")
    # The fallback is also an implicitly keyboard-focusable overflowing code box.
    diagram_source = 'graph TD; A-->B\n%% ' + 'X' * 250
    answer = _LONG_ANSWER + '\n\n```mermaid\n' + diagram_source + '\n```'
    _store(root, "chat", {
        "direction": "system", "chat_id": 1, "user_id": 1, "type": "project_completion_summary",
        "project_id": project["id"], "project_name": project["name"], "status": "completed",
        "ts": "2026-08-22T12:00:00+00:00", "task_id": "late-diagram",
        "text": "Diagram report · Done", "completion_answer": answer})
    _store(root, "chat", {
        "direction": "out", "chat_id": int(project["chat_id"]), "user_id": 1,
        "ts": "2026-08-22T11:59:00+00:00", "task_id": "late-diagram", "text": answer})
    tab = "Alt+Tab" if engine == "webkit" else "Tab"
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        page = browser.new_page(viewport={"width": 1146, "height": 792})
        page.add_init_script("""window.__copied = []; Object.defineProperty(navigator, 'clipboard', {configurable: true,
            value: {writeText: async text => { window.__copied.push(text); }}});""")
        pending = []
        page.route("**/mermaid.min.js", lambda route: pending.append(route))
        try:
            with page.expect_request("**/mermaid.min.js"):
                page.goto(direct_server_with_data["url"], wait_until="domcontentloaded")
            row = page.locator('#chat-messages .project-answer[data-task-id="late-diagram"]')
            expect(row).to_have_class(re.compile(r'\bis-folded\b'))
            assert row.locator('[data-code-copy]').count() == 0
            pending.pop().abort()
            button = row.locator('[data-code-copy]')
            expect(button).to_have_attribute('tabindex', '-1')
            row.locator('.sender').click()
            reference = row.locator('[data-intent="open-project"]')
            for _ in range(12):
                page.keyboard.press(tab)
                assert not row.locator('.message').evaluate('n => n.contains(document.activeElement)')
                if reference.evaluate('n => n === document.activeElement'):
                    break
            expect(reference).to_be_focused()
            assert row.locator('.message').evaluate('n => n.scrollTop') == 0
            page.unroute("**/mermaid.min.js")
            page.route("**/mermaid.min.js", lambda route: route.abort())
            reference.click()
            panel = page.locator(f'#panel-pchat-{project["id"]}')
            ordinary = panel.locator('[data-code-copy]')
            expect(ordinary).to_be_visible()
            assert ordinary.evaluate('n => n.tabIndex') == 0
            ordinary.click()
            expect(ordinary).to_have_text('Copied')
            copied = page.evaluate('() => window.__copied.at(-1)')
            assert copied == panel.locator('.md-code-block pre > code').inner_text()
            assert copied.strip() == diagram_source
        finally:
            browser.close()


@pytest.mark.ui_browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
@pytest.mark.parametrize("width, theme", [(1100, "dark"), (390, "light")])
def test_ui_no_answer_ending_keeps_codes_copy_and_project_context(
    direct_server_with_data, monkeypatch, engine, width, theme,  # noqa: F811
):
    """Real producer -> isolated dialogue store -> HTTP -> Main; disclosure, Copy and reload keep their meaning."""
    import json
    import os
    from pathlib import Path
    from playwright.sync_api import expect, sync_playwright
    from ouroboros import outcomes
    from tests.test_project_completion_mirror import _enqueue_row
    from tests.ui_console_errors import PageErrors

    root = direct_server_with_data["data_dir"]
    result = {"status": "failed", "reason_code": "example_new_reason", "terminal_origin": "host_salvage",
              "result": "Two update scenarios remain unchecked.",
              "outcome_axes": outcomes.terminal_outcome_axes(
                  lifecycle="failed", execution=outcomes.EXECUTION_INFRA_FAILED,
                  reason_code="example_new_reason", review_trigger="worker_terminal")}
    event = _enqueue_row(root, monkeypatch, result)
    meta = event["progress_meta"]
    row_data = {**meta, "direction": "system", "chat_id": 1, "user_id": 1,
                "type": "project_completion_summary", "task_id": "root-project",
                "text": event["text"], "ts": "2026-10-01T10:00:00Z", "delivery_id": event["delivery_id"]}
    _store(root, "chat", row_data)
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or root / "evidence") / f"ending-{engine}-{width}"
    evidence.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": 750}, reduced_motion="reduce")
        page.add_init_script("""window.__copied = []; Object.defineProperty(navigator, 'clipboard', {configurable:true,
            value: {writeText: async text => window.__copied.push(text)}});""")
        page.add_init_script(f"localStorage.setItem('ouroboros.theme', '{theme}')")
        errors = PageErrors(page, engine)
        try:
            page.goto(direct_server_with_data["url"], wait_until="domcontentloaded")
            row = page.locator('#chat-messages .chat-bubble[data-task-id="root-project"][data-system-type="project_completion_summary"]')
            for stage in ("history", "reload"):
                _wait_status(page, "Online", timeout=30_000)
                expect(row.locator('.project-ending')).to_be_visible()
                row.scroll_into_view_if_needed()
                assert "system" in row.get_attribute('class').split()
                expect(row.locator('.sender')).to_have_text('System')
                expect(row.locator('.project-ending-title')).to_have_text(meta['task_name'])
                expect(row.locator('.ui-status')).to_have_text('Failed')
                expect(row.locator('.project-ending-cause')).to_have_text('The recorded reason has no description yet.')
                expect(row.locator('.project-ending-salvage')).to_have_text(meta['completion_notice']['salvage'])
                assert row.inner_text().count(meta['project_name']) == 1
                assert 'example_new_reason' not in row.inner_text()
                disclosure = row.locator('.project-ending-details')
                summary = disclosure.locator('summary')
                summary.focus()
                page.keyboard.press('Enter')
                expect(disclosure).to_have_attribute('open', '')
                expect(disclosure.locator('p')).to_have_text('Recorded reason: example_new_reason')
                assert page.locator('#chat-messages').evaluate('n => n.scrollWidth <= n.clientWidth + 1')
                row.screenshot(path=str(evidence / f'{stage}.png'))
                if stage == 'history':
                    row.locator('.chat-message-copy').click()
                    page.wait_for_function('window.__copied.length === 1')
                    assert page.evaluate('window.__copied[0]') == event['text']
                    page.reload(wait_until='domcontentloaded')
            reference = row.locator('[data-intent="open-project"]')
            assert reference.count() == 1
            reference.focus()
            page.keyboard.press('Enter')
            expect(page.locator('#project-panel-title')).to_have_text(meta['project_name'])
            (evidence / 'page-errors.json').write_text(json.dumps(list(errors)), encoding='utf-8')
            assert not errors.unexpected(), errors
        finally:
            browser.close()
