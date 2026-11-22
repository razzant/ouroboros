"""A model handover in a task card's Details, in the served app (DESIGN: model metadata; ``settled_card.js``).

``web/tests/model_handover_details.test.js`` owns what each handover row says and how a stale or new attempt
retires it. This checks the page boundary once per engine on one representative card: Details opened while the
work still runs show the stored history even after an early terminal answer, without a second click; retranslation
does not read it again; on reload it reopens, and wraps at phone width. The stored rows are typed fixtures of the
declared shape (the loop that writes them: ``tests/test_authoring_handover_loop.py``). An exact row quote also
opens its card and reads its Details. Waits observe the DOM or
the read the click caused. With ``OUROBOROS_BROWSER_EVIDENCE_OUT`` set, one desktop and one phone screenshot of the
card go there (``handover-<engine>-<size>.png``).
"""
import os
import pathlib

import pytest

from tests.ui_console_errors import PageErrors

pytest_plugins = ("tests.test_ui_smoke_playwright",)
pytestmark = [pytest.mark.ui_browser]

ASTRA, FABLE, OPUS = "codex=gpt-6-astra", "claude-fable-5-1", "claude-opus-5-5"
TID = "card-model-handover"
CHILD = "card-model-handover-child"
ANSWER = "The predecessor's checks passed; the report stands as written."
SWITCH = {"reason": "fallback", "recovery_prompted": True}
HANDOVERS = [
    {"from_model": ASTRA, "to_model": FABLE, "tool_calls_at_handover": 1, "status": "incomplete",
     "incomplete_observed": True, "reply_observed": True, **SWITCH},
    {"from_model": FABLE, "to_model": OPUS, "tool_calls_at_handover": 2, "status": "selected",
     "selection": "answer_sha256", **SWITCH},
    {"from_model": OPUS, "to_model": "mock-unused-route", "status": "pending", **SWITCH},
]
LINES = [f"{ASTRA} → {FABLE} · handover recorded as incomplete", f"{FABLE} → {OPUS} · confirmed a saved answer",
         f"{OPUS} → mock-unused-route · successor reply not recorded"]
OUTCOME = {"execution": {"status": "ok"}, "objective": {"status": "not_evaluated"}}


def _open_details(card):
    card.evaluate("el => el.scrollIntoView({block: 'center'})")  # clear of the floating header and composer
    card.locator(":scope > [data-live-summary-button] [data-live-disclosure]").click()


def _handover_lines(page, card):
    """The row's lines once the page painted it; the head and compact line never carry the switch."""
    from playwright.sync_api import expect

    rows = card.locator(":scope > .chat-live-details > [data-live-facts] > .chat-live-fact").filter(
        has=page.locator("dt", has_text="Model handover"))
    expect(rows).to_have_count(1, timeout=30_000)
    for line in LINES:
        expect(rows.locator("dd")).to_contain_text(line)
    for part in ("[data-live-summary-button]", "[data-live-meta]"):
        words = card.locator(part).first.inner_text()
        assert "→" not in words and "handover" not in words.lower(), (part, words)
    return rows.locator("dd").inner_text().split("\n")


def _witness(card, engine, size):
    out = os.environ.get("OUROBOROS_BROWSER_EVIDENCE_OUT", "").strip()
    if out:
        pathlib.Path(out).mkdir(parents=True, exist_ok=True)
        card.screenshot(path=str(pathlib.Path(out) / f"handover-{engine}-{size}.png"), animations="disabled")


def _watch_task_reads(page):
    page.add_init_script("""(() => {
        const fetch = window.fetch.bind(window);
        window.taskDetailReads = [];
        window.fetch = (input, ...rest) => {
            const path = new URL(input instanceof Request ? input.url : input, location.href).pathname;
            if (/^\\/api\\/tasks\\/[^/]+$/.test(path)) window.taskDetailReads.push(path);
            return fetch(input, ...rest);
        };
    })()""")


def _translate_open_details(page, card):
    from playwright.sync_api import expect

    reads = page.evaluate("window.taskDetailReads.slice()")
    words = card.locator(":scope > .chat-live-details > [data-live-facts] .chat-live-fact[data-i18n-fmt]")
    details = card.locator(":scope > .chat-live-details").element_handle()
    label = "Смена модели"
    for revision, selection in enumerate(("подтвердила сохранённый ответ", "выбрала сохранённый ответ"), 1):
        page.evaluate("""async payload => {
            const {setLanguage} = await import('/static/modules/i18n.js');
            await setLanguage(payload.language, payload);
        }""", {"language": "ru", "revision": revision, "entries": {
            "code:task.fact.model_handover": {"text": label},
            "code:task.handover.selected.answer_sha256": {"text": selection},
        }})
        expect(words.locator("dt")).to_have_text(label)
        expect(words.locator("dd")).to_contain_text(f"{FABLE} → {OPUS} · {selection}")
    page.evaluate("""async () => {
        const {setLanguage} = await import('/static/modules/i18n.js');
        await setLanguage('en', {language: 'en', english: true, entries: {}});
    }""")
    assert _handover_lines(page, card) == LINES
    assert details.evaluate("(el) => el.isConnected")
    expect(card).to_have_attribute("data-expanded", "1")
    assert page.evaluate("window.taskDetailReads") == reads


@pytest.mark.parametrize("engine", ["webkit", "chromium"])
def test_open_details_show_the_settled_model_handover(direct_server_with_data, engine):
    from playwright.sync_api import expect, sync_playwright
    from ouroboros import dialogue_store
    from ouroboros.task_results import write_task_result
    from ouroboros.terminal_projection import _project_row
    from supervisor.message_bus import log_chat
    from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET, _OBSERVE_STATE_READS, _emit_ws_frame, \
        _wait_socket_open_quiescent

    fixture = direct_server_with_data
    root = fixture["data_dir"]
    write_task_result(root, TID, "running", chat_id=1, root_task_id=TID, delegation_role="root")

    def visit(browser, theme, phone=False):
        context = browser.new_context(
            viewport={"width": 390, "height": 844} if phone else {"width": 1100, "height": 750},
            color_scheme=theme, reduced_motion="reduce", **({"device_scale_factor": 2, "has_touch": True} if phone else {}))
        page = context.new_page()
        errors = PageErrors(page, engine)
        page.add_init_script(f"({_CAPTURE_TEST_SOCKET})(); ({_OBSERVE_STATE_READS})();")
        _watch_task_reads(page)
        page.goto(fixture["url"], wait_until="domcontentloaded")
        _wait_socket_open_quiescent(page)
        return context, page, errors, page.locator(f'.chat-live-card[data-task-id="{TID}"]')

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            # Desktop, light: Details opened while the work runs, whose read finds no history yet.
            context, page, errors, card = visit(browser, "light")
            progress = {"task_id": TID, "chat_id": 1, "content": "Reading the predecessor's checks", "narration": True,
                        "ts": "2026-11-18T09:00:00Z"}
            dialogue_store.append(root, "progress", progress)
            _emit_ws_frame(page, {**progress, "type": "chat", "role": "assistant", "is_progress": True})
            expect(card).to_have_attribute("data-finished", "0")
            with page.expect_response(lambda response: response.url.endswith(f"/api/tasks/{TID}")) as opened:
                _open_details(card)
            assert opened.value.json()["status"] == "running"
            # The early answer can close the card before the result write (test_finalization_timing.py).
            log_chat("out", 1, 1, ANSWER, task_id=TID, drive_root=root)
            _emit_ws_frame(page, {"type": "chat", "role": "assistant", "chat_id": 1, "task_id": TID, "content": ANSWER,
                                  "task_terminal_status": "completed", "outcome_axes": OUTCOME})
            expect(card.locator("[data-live-phase]")).to_have_text("Done", timeout=30_000)
            expect(card.locator("[data-live-facts] dt", has_text="Model handover")).to_have_count(0)
            # The stored result's terminal summary wakes the same open disclosure.
            result = write_task_result(root, TID, "completed", chat_id=1, root_task_id=TID, delegation_role="root",
                                       result=ANSWER, terminal_origin="model_final", outcome_axes=OUTCOME,
                                       reason_code="final_message", authoring_handovers=HANDOVERS)
            summary = _project_row(TID, result, {}, {"chat_id": 1})
            dialogue_store.append(root, "chat", summary)
            _emit_ws_frame(page, {**summary, "type": "chat", "role": "system", "system_type": "task_summary",
                                  "content": summary["text"]})
            expect(card.locator("[data-live-phase]")).to_have_text("Done", timeout=30_000)
            assert _handover_lines(page, card) == LINES  # still the one click
            _translate_open_details(page, card)
            # After a reload the card comes from history, closed; opening it reads the same history.
            page.reload(wait_until="domcontentloaded")
            _wait_socket_open_quiescent(page)
            expect(card.locator("[data-live-phase]")).to_have_text("Done", timeout=30_000)
            _open_details(card)
            assert _handover_lines(page, card) == LINES
            _witness(card, engine, "desktop")
            assert not errors.unexpected(), errors
            context.close()
            # A phone, dark: the row wraps inside its card.
            context, page, errors, card = visit(browser, "dark", phone=True)
            expect(card.locator("[data-live-phase]")).to_have_text("Done", timeout=30_000)
            _open_details(card)
            assert _handover_lines(page, card) == LINES
            assert card.locator(":scope > .chat-live-details").evaluate("el => el.scrollWidth - el.clientWidth") <= 1
            _witness(card, engine, "phone")
            assert not errors.unexpected(), errors
            context.close()
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["webkit", "chromium"])
def test_folded_settled_child_reads_detail_only_when_revealed(direct_server_with_data, engine):
    from playwright.sync_api import expect, sync_playwright
    from ouroboros.task_results import write_task_result
    from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET, _OBSERVE_STATE_READS, _emit_ws_frame, \
        _wait_socket_open_quiescent

    fixture = direct_server_with_data
    root = fixture["data_dir"]
    write_task_result(root, TID, "completed", chat_id=1, root_task_id=TID, delegation_role="root", outcome_axes=OUTCOME)
    write_task_result(root, CHILD, "completed", chat_id=1, root_task_id=TID, parent_task_id=TID,
                      delegation_role="subagent", outcome_axes=OUTCOME, authoring_handovers=HANDOVERS)
    child_path = f"/api/tasks/{CHILD}"

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page = browser.new_page(viewport={"width": 1100, "height": 750})
            assert page.request.post(f"{fixture['url'].rstrip('/')}/api/ui/preferences",
                                     data={"nested_subagents_expanded": True}).ok
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})(); ({_OBSERVE_STATE_READS})();")
            _watch_task_reads(page)
            with page.expect_response(lambda r: r.url.endswith("/api/ui/preferences")):
                page.goto(fixture["url"], wait_until="domcontentloaded")
            _wait_socket_open_quiescent(page)
            parent = page.locator(f'.chat-live-card[data-task-id="{TID}"]')
            child = parent.locator(f'.chat-live-card[data-task-id="{CHILD}"]')
            progress = {"type": "chat", "role": "assistant", "chat_id": 1, "is_progress": True,
                        "narration": True, "content": "Checking an item."}
            _emit_ws_frame(page, {**progress, "task_id": TID})
            _emit_ws_frame(page, {"type": "chat", "role": "assistant", "chat_id": 1, "task_id": TID,
                                  "task_terminal_status": "completed", "outcome_axes": OUTCOME, "content": ANSWER})
            expect(parent).to_have_attribute("data-finished", "1")
            _emit_ws_frame(page, {**progress, "task_id": CHILD, "subagent_task_id": CHILD, "parent_task_id": TID,
                                  "delegation_role": "subagent", "subagent_event": "completed",
                                  "outcome_axes": OUTCOME, "result": ANSWER})
            expect(child).to_have_attribute("data-finished", "1")
            expect(child).to_have_attribute("data-expanded", "1")
            expect(child).to_be_hidden()
            # The existing hydrator starts in a microtask, even if the page cannot show the child.
            assert child_path not in page.evaluate("window.taskDetailReads")
            _open_details(parent)
            assert _handover_lines(page, child) == LINES
            assert page.evaluate("window.taskDetailReads").count(child_path) == 1
            _open_details(parent)
            _open_details(parent)
            assert page.evaluate("window.taskDetailReads").count(child_path) == 1
        finally:
            browser.close()


def test_exact_row_reference_reads_open_details(direct_server_with_data):
    from playwright.sync_api import expect, sync_playwright
    from ouroboros import dialogue_store
    from ouroboros.chat_references import reference_from_row
    from ouroboros.task_results import write_task_result
    from ouroboros.terminal_projection import _project_row
    from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET, _OBSERVE_STATE_READS, _wait_socket_open_quiescent

    fixture = direct_server_with_data
    root = fixture["data_dir"]
    source = {"chat_id": 1, "task_id": TID, "direction": "out", "ts": "2026-11-18T09:00:00Z",
              "text": "Reading the predecessor's checks", "narration": True}
    dialogue_store.append(root, "progress", source)
    result = write_task_result(root, TID, "completed", chat_id=1, root_task_id=TID, delegation_role="root",
                               result=ANSWER, terminal_origin="model_final", outcome_axes=OUTCOME,
                               reason_code="final_message", authoring_handovers=HANDOVERS)
    summary = _project_row(TID, result, {}, {"chat_id": 1})
    dialogue_store.append(root, "chat", {**summary, "ts": "2026-11-18T09:00:30Z"})
    ref = reference_from_row(source)
    dialogue_store.append(root, "chat", {"chat_id": 1, "direction": "out", "ts": "2026-11-18T09:01:00Z",
                                         "text": "Here are the checks I quoted.", "reply_to": ref})

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1100, "height": 750}, reduced_motion="reduce")
            errors = PageErrors(page, "chromium")
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})(); ({_OBSERVE_STATE_READS})();")
            _watch_task_reads(page)
            page.goto(fixture["url"], wait_until="domcontentloaded")
            _wait_socket_open_quiescent(page)
            card = page.locator(f'.chat-live-card[data-task-id="{TID}"]')
            expect(card).to_have_attribute("data-finished", "1")
            expect(card).to_have_attribute("data-expanded", "0")
            task_path = f"/api/tasks/{TID}"
            assert task_path not in page.evaluate("window.taskDetailReads")
            address = ref["address"]
            page.locator(f'.chat-quote[data-address="{address}"] .chat-quote-main').click()
            landed = card.locator("[data-live-timeline] .is-quote-target")
            expect(landed).to_contain_text(source["text"])
            expect(card).to_have_attribute("data-expanded", "1")
            assert _handover_lines(page, card) == LINES  # only the quote was pressed
            assert page.evaluate("window.taskDetailReads").count(task_path) == 1
            assert not errors.unexpected(), errors
        finally:
            browser.close()
