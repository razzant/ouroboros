"""#2380/#2411 and the owner's 1 Nov choice in the real browser and server.

The candidate server runs from the candidate bytes (``direct_server_with_data``); the predecessor's interruption in
Main and its delegated custody are real durable rows; Continue, Pause and Resume go through the page's own request
owners and the real endpoints. A Resume refused by a fact no wait settles (an unknown delegated start) stays on the
work's row in Main's work list after its toast is gone and after a reload: dated, last on the line, its words in the
row's details, the Pause and its Resume as they were. Once that start's outcome is known, Resume returns the work to
its writer wait: Waiting, saying what it waits for, with no Resume; when the earlier attempt's run settles, the work
starts by itself. Each engine is attested from the browser it launched; screenshots at 1148x808 and 390 px in both
themes go to ``OUROBOROS_UI_EVIDENCE_DIR``, else ``<data>/ui-evidence/resume-writer-wait``.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from tests.test_continue_independent_review import PRED, _current_pool, _pending, _rows  # noqa: F401 - fixture
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401 - pytest fixture
from tests.ui_chat_viewport_smoke import wait_drawer_closed

pytestmark = [pytest.mark.ui_browser]
LEAF = "delegated run run-leaf is running"
UNKNOWN = "a delegated start has an unknown outcome (no run was recorded)"
SIZES = ((1148, 808), (390, 844))


def _toasts_gone(page):
    page.wait_for_function("() => !document.querySelector('#toast-stack .toast')", timeout=20000)


def _toast_shown(page, text):
    """The toast saying ``text``, once it is fully in: a shot during its fade-in shows a ghost of it."""
    toast = page.locator("#toast-stack .toast").filter(has_text=text)
    toast.wait_for()
    page.wait_for_function("() => [...document.querySelectorAll('#toast-stack .toast')]"
                           ".every((el) => Number(getComputedStyle(el).opacity) > 0.95)")
    return toast


def _close_list(page):
    if page.locator(".rw-trigger").first.get_attribute("aria-expanded") == "true":
        page.keyboard.press("Escape")
        page.wait_for_function("() => document.querySelector('.rw-trigger')?.getAttribute('aria-expanded') !== 'true'")


def _in_main(root):
    """The predecessor interrupted by the owner's Restart, in Main (chat 1): its Continue's work lives there."""
    from ouroboros.task_results import STATUS_CANCELLED, write_task_result
    from tests.test_owner_continue import _interrupted

    _interrupted(root, PRED, origin={"source": "owner_restart", "reason": "owner_restart", "scope": "single"})
    write_task_result(root, PRED, STATUS_CANCELLED, chat_id=1, origin_message_ref={"chat_id": 1, "client_message_id": "m-1"})


def _row_facts(page, work_id):
    return page.evaluate("""(id) => {
        const row = document.querySelector(`.rw-row[data-id="${id}"]`);
        if (!row) return null;
        const text = (sel) => row.querySelector(sel)?.textContent?.trim() || '';
        const detail = row.querySelector('.rw-detail');
        const secondary = row.querySelector('.rw-secondary');
        const box = row.getBoundingClientRect();
        const lines = (el) => { if (!el || el.hidden) return 0; const r = document.createRange(); r.selectNodeContents(el);
            return new Set([...r.getClientRects()].filter((c) => c.width > 0).map((c) => Math.round(c.top))).size; };
        return { word: text('.rw-word'), line: text('.rw-secondary'), lines: lines(secondary),
            resume: Boolean(row.querySelector('[data-resume-run]')),
            details: detail && !detail.hidden ? [...detail.querySelectorAll('.rw-detail-line')].map((l) => l.textContent.trim()) : [],
            overflow: row.scrollWidth > row.clientWidth + 1, right: box.right, viewport: innerWidth,
            theme: document.documentElement.dataset.theme };
    }""", work_id)


def _open_list(page):
    trigger = page.locator(".rw-trigger").first
    trigger.wait_for()
    if trigger.get_attribute("aria-expanded") != "true":
        trigger.click()
    page.wait_for_function("() => document.querySelector('.rw-trigger')?.getAttribute('aria-expanded') === 'true'")


def _shots(page, evidence, engine, stem, work_id, observed):
    """Both themes at the owner's desktop window and at phone width: the row, its open list and the page."""
    for theme in ("light", "dark"):
        page.emulate_media(color_scheme=theme)
        page.wait_for_function("(t) => document.documentElement.dataset.theme === t", arg=theme)
        for width, height in SIZES:
            page.set_viewport_size({"width": width, "height": height})
            if width < 700:
                wait_drawer_closed(page)
            _open_list(page)
            row = page.locator(f'.rw-row[data-id="{work_id}"]')
            row.wait_for()
            row.scroll_into_view_if_needed()
            page.wait_for_timeout(250)  # the list's rise settles before the look
            facts = _row_facts(page, work_id)
            observed[f"{stem}-{theme}-{width}"] = facts
            assert facts["theme"] == theme and not facts["overflow"] and facts["right"] <= facts["viewport"] + 1, facts
            row.screenshot(path=str(evidence / f"{engine}-{stem}-row-{theme}-{width}.png"))
            page.screenshot(path=str(evidence / f"{engine}-{stem}-page-{theme}-{width}.png"))
    page.emulate_media(color_scheme="light")
    page.set_viewport_size({"width": SIZES[0][0], "height": SIZES[0][1]})


# A short name on purpose: the server's manager socket lives under this test's tmp_path, and macOS caps an AF_UNIX
# path at 104 bytes (ouroboros/test_environment.default_temp_parent; a long name overflows it, issue #2517).
@pytest.mark.parametrize("engine", ["webkit", "chromium"])
def test_resume_waits(direct_server_with_data, engine):  # noqa: F811
    """A refused Resume stays dated on its row; a later one waits for the earlier writers, then the work starts."""
    from playwright.sync_api import sync_playwright

    from ouroboros import delegate_custody as dc

    server = direct_server_with_data
    root = server["data_dir"]
    server["stop_server"]()
    _in_main(root)
    server["start_server"]()
    # Seeded while the server runs (as the held-Continue case seeds them): a running leaf and a start whose outcome
    # is unknown, so the first Resume is refused and the second one, once that outcome is known, waits.
    _rows(root, leaf=True, review=False)
    _pending(root, "inv-unbound", source=None)
    dc._CUSTODY.clear()
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or root / "ui-evidence") / "resume-writer-wait"
    evidence.mkdir(parents=True, exist_ok=True)
    observed: dict = {}
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            attested = {"engine": browser.browser_type.name, "version": browser.version}
            assert attested["engine"] == engine, attested
            page = browser.new_page(viewport={"width": SIZES[0][0], "height": SIZES[0][1]}, color_scheme="light")
            page.goto(server["url"], wait_until="domcontentloaded")
            with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith("/continue")) as sent:
                successor = page.evaluate(
                    "async (id) => (await import('/static/modules/task_continue.js')).continueTaskAction(id)", PRED)
            ack = sent.value.json()
            assert ack["ok"] and ack["held"] is True and successor == ack["successor_task_id"], ack
            _toasts_gone(page)
            # The owner's own client pauses the held Continue (its declaration names the owner).
            paused = page.evaluate("async (id) => (await import('/static/modules/api_client.js')).pauseTask(id, 'pause-ui-1')",
                                   successor)
            assert paused.get("ok"), paused

            # 1. Resume refused by a fact no wait settles: the toast, then the row keeps it after the toast is gone.
            with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith("/resume")) as sent:
                page.evaluate("async (id) => (await import('/static/modules/task_control_menu.js')).resumeTaskAction(id)",
                              successor)
            assert sent.value.status == 409, sent.value.text()
            refusal = sent.value.json()
            toast = _toast_shown(page, "Resume refused:")
            assert toast.inner_text().strip() == f"Resume refused: {LEAF}; {UNKNOWN}."
            page.screenshot(path=str(evidence / f"{engine}-1-refused-toast.png"))
            _toasts_gone(page)
            _open_list(page)
            page.wait_for_function("(id) => document.querySelector(`.rw-row[data-id=\"${id}\"] .rw-secondary`)"
                                   "?.textContent.includes('Resume refused at')", arg=PRED, timeout=30000)
            facts = _row_facts(page, PRED)
            assert facts["word"] == "Paused" and facts["line"].startswith("You paused it · "), facts
            assert facts["resume"], "the paused work keeps its own Resume"
            _shots(page, evidence, engine, "2-refused", PRED, observed)
            # Its details give the door's own words, dated: the card's own Details when the work's card is loaded
            # (its row then goes to that card), else the row's details.
            _open_list(page)
            via = page.get_attribute(f'.rw-row[data-id="{PRED}"] .rw-main', "data-target")
            if via == "card":
                _close_list(page)
                card = page.locator(f'.chat-live-card[data-task-id="{successor}"], .chat-live-card[data-task-id="{PRED}"]').first
                card.locator("[data-live-disclosure]").first.click()
                detail = card.locator("[data-live-phase-detail]").first
                detail.wait_for(state="visible", timeout=20000)
                details = [f"Card details: {detail.inner_text().strip()}"]
                last = details[0]
                card.screenshot(path=str(evidence / f"{engine}-3-refused-card-details.png"))
            else:
                page.locator(f'.rw-row[data-id="{PRED}"] .rw-main').click()
                page.wait_for_function("(id) => [...document.querySelectorAll(`.rw-row[data-id=\"${id}\"] .rw-detail-line`)]"
                                       ".some((l) => l.textContent.startsWith('Last Resume'))", arg=PRED, timeout=20000)
                details = _row_facts(page, PRED)["details"]
                [last] = [line for line in details if line.startswith("Last Resume")]
                page.locator(f'.rw-row[data-id="{PRED}"]').screenshot(path=str(evidence / f"{engine}-3-refused-details.png"))
            assert last.endswith(f": {LEAF}; {UNKNOWN}") and "Resume refused at" in last, details
            observed["details_via"] = via

            # 2. A reload reads it from the task's record again.
            page.reload(wait_until="domcontentloaded")
            _open_list(page)
            page.wait_for_function("(id) => document.querySelector(`.rw-row[data-id=\"${id}\"] .rw-secondary`)"
                                   "?.textContent.includes('Resume refused at')", arg=PRED, timeout=30000)
            reloaded = _row_facts(page, PRED)
            assert reloaded["line"] == facts["line"], (reloaded, facts)
            page.screenshot(path=str(evidence / f"{engine}-4-refused-after-reload.png"))

            # 3. The start's outcome is known now; the leaf still runs: Resume returns the work to its writer wait.
            assert dc.emit(root, dc.START_FAILED, {"invocation_id": "inv-unbound", "task_id": PRED, "definite": True,
                                                   "error": "refused"})
            _close_list(page)
            with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith("/resume")) as sent:
                page.evaluate("async (id) => (await import('/static/modules/task_control_menu.js')).resumeTaskAction(id)",
                              successor)
            accepted = sent.value.json()
            assert sent.value.status == 200 and accepted["held"] is True, accepted
            toast = _toast_shown(page, "Resume accepted;")
            assert toast.inner_text().strip() == (
                f"Resume accepted; it starts by itself when the earlier attempt's work finishes: {LEAF}.")
            page.screenshot(path=str(evidence / f"{engine}-5-waits-toast.png"))
            _toasts_gone(page)
            _open_list(page)
            page.wait_for_function("(id) => document.querySelector(`.rw-row[data-id=\"${id}\"] .rw-word`)"
                                   "?.textContent.trim() === 'Waiting'", arg=PRED, timeout=30000)
            waiting = _row_facts(page, PRED)
            assert waiting["line"].startswith("Continue starts when the earlier attempt's work is confirmed finished"), waiting
            assert "Resume refused" not in waiting["line"] and not waiting["resume"], "it waits: no Resume, no old refusal"
            _shots(page, evidence, engine, "6-waiting", PRED, observed)

            # 4. The earlier attempt's run settles: the cadence releases the wait and the work starts by itself.
            assert dc.emit(root, dc.SETTLED, {"run_id": "run-leaf", "task_id": PRED, "route": "codex",
                                              "model": "leaf-model", "state": "completed"})
            started = None
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                detail = page.evaluate("async (id) => (await fetch(`/api/tasks/${id}`)).json()", successor)
                if detail.get("status") not in {"scheduled", "", None}:
                    started = detail.get("status")
                    break
                page.wait_for_timeout(2000)
            assert started == "running", f"the released Continue did not start: {detail}"
            _open_list(page)
            page.wait_for_function("(id) => document.querySelector(`.rw-row[data-id=\"${id}\"] .rw-word`)"
                                   "?.textContent.trim() === 'Working'", arg=PRED, timeout=30000)
            page.screenshot(path=str(evidence / f"{engine}-7-started-{started}.png"))
            observed["started"] = {"status": started, "row": _row_facts(page, PRED)}
            (evidence / f"{engine}-facts.json").write_text(json.dumps({
                "attested": attested, "continue_ack": ack, "pause": paused, "refusal": refusal, "accepted": accepted,
                "refused_row": facts, "refused_details": details, "reloaded_row": reloaded, "waiting_row": waiting,
                "observed": observed, "candidate": server["candidate_identity"]}, indent=2, default=str))
            print("RESUME_WRITER_WAIT_EVIDENCE", engine, attested, evidence)
        finally:
            browser.close()
