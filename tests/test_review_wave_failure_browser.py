"""A review Ouroboros itself stopped, in real WebKit: isolated server → ledger → task detail → card → reader.

The records are written by the ledger's own builder with the operation's own wave failure note
(``review_change.wave_failure_note``, as ``_settle`` attaches it); the card reads them through the real task
detail, hydrator and keyed renderer, and Read opens the saved note in the existing document reader. A record
the earlier operation wrote (no note) sits beside it unchanged. ``EVIDENCE`` is a module seam a QA wrapper may
set; ``OUROBOROS_UI_EVIDENCE_DIR`` names where the frames go otherwise.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import review_ledger as rl
from ouroboros.agent_task_pipeline import _store_task_result
from ouroboros.post_task_synthesis import _record_task_facts
from ouroboros.tools.review_change import wave_failure_note
from tests.test_review_record_card_projection import _gate_record
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET

pytest_plugins = ("tests.test_ui_smoke_playwright",)
pytestmark = [pytest.mark.ui_browser]
STOPPED, EARLIER, ANSWERED = "wave-failure-stopped", "wave-failure-earlier", "wave-failure-answered"
COMBOS = (("light", 1148, 808), ("dark", 1148, 808), ("light", 390, 844), ("dark", 390, 844))
SECRET = "sk-ant-api03-" + "Zq9xW2" * 7
TAIL = "TAIL-OF-THE-WHOLE-NOTE"
WORDS = ("⚠️ REVIEW_BLOCKED: durable review custody could not be reserved before dispatch — OSError: [Errno 28] "
         "No space left on device: the attempt row could not be saved; " + "state write retried and refused; " * 40
         + f"credential seen in the environment dump {SECRET} {TAIL}\n"
         "No reviewer was started; fix the state write and retry.")
LEAD = "Ouroboros could not complete this review; its note says which step failed."
EVIDENCE: Path | None = None


def _store(root, repo, task_id):
    env, task = SimpleNamespace(drive_root=root, repo_dir=repo), {"id": task_id, "type": "task", "chat_id": 1}
    _store_task_result(env, task, "Review check", {"rounds": 1, "cost": 0}, {"tool_calls": []})
    _record_task_facts(env, task, {"rounds": 1}, {"tool_calls": []}, root / "logs")


def _unsent(root, task_id, goal, *, note):
    """The record of a wave the host stopped before any send: three prepared seats, none dispatched."""
    plans = [{"slot_id": slot, "model": model, "route": "api_chat", "parts": parts, "retrieves": len(parts) > 1}
             for slot, model, parts in (("t1", "openai/gpt-5.6-sol", ["change"]),
                                         ("t2", "codex=gpt-5.6-sol", ["change", "coupling"]),
                                         ("s1", "openai/gpt-5.6-sol", ["change", "coupling"]))]
    record = rl.build_wave_record({"task_id": task_id, "root_task_id": task_id, "enforcement": "blocking",
                                   "structured": {"rows": plans, "started_ts": "2026-11-01T17:00:00+00:00"},
                                   "blocked": True, "block_reason": "infra_failure", "goal": goal,
                                   "subject": {"root_kind": "system_repo", "root": "/repo", "kind": "index"}},
                                  surface="change", drive_root=root)
    if note:
        record.verdict["wave_failures"] = [wave_failure_note(root, record, {"block_reason": "infra_failure", "text": WORDS})]
    return rl.write_record(root, record)


def seed(fixture):
    root, repo = fixture["data_dir"], fixture["repo_dir"]
    fixture["stop_server"]()
    _unsent(root, STOPPED, "Readable review causes", note=True)
    _unsent(root, EARLIER, "Readable review causes", note=False)
    answered = _gate_record(ANSWERED)  # one seat passed, one rejected, an added critic errored; then the host stopped
    answered.brief["goal"] = "Answers beside a stopped wave"
    answered.verdict["wave_failures"] = [wave_failure_note(root, answered, {
        "block_reason": "infra_failure", "text": "RuntimeError: coupling history unwritable after the answers arrived"})]
    rl.write_record(root, answered)
    for task_id in (STOPPED, EARLIER, ANSWERED):
        _store(root, repo, task_id)
    fixture["start_server"]()


def _open_record(page, task_id):
    card = page.locator(f'.chat-live-card[data-task-id="{task_id}"]')
    card.wait_for(state="visible", timeout=30000)
    card.locator("[data-live-review-summary]").click()
    group = card.locator("[data-review-group]").first
    group.locator("[data-review-group-toggle]").click()
    group.locator(".review-check-body").wait_for(state="visible")
    card.scroll_into_view_if_needed()
    page.wait_for_timeout(250)  # the chevron's turn ends before the frame
    return card, group


def test_stopped_wave_card(direct_server_with_data, tmp_path):
    """A stopped review reads its cause and opens the whole safe note. The short name keeps ``tmp_path`` (the
    fixture server's TMPDIR) short enough for its multiprocessing socket under macOS's 104-byte limit."""
    from playwright.sync_api import sync_playwright

    seed(direct_server_with_data)
    url = direct_server_with_data["url"]
    evidence = EVIDENCE or Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or tmp_path / "wave-failure-evidence")
    evidence.mkdir(parents=True, exist_ok=True)
    results = []
    with sync_playwright() as pw:
        browser = pw.webkit.launch(headless=True)
        try:
            for theme, width, height in COMBOS:
                tag = f"{theme}-{width}"
                page = browser.new_page(viewport={"width": width, "height": height})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
                page.goto(url, wait_until="domcontentloaded")
                page.evaluate("theme => document.documentElement.dataset.theme = theme", theme)

                card, group = _open_record(page, STOPPED)
                head = group.locator(".review-check-head .review-state").inner_text().strip()
                assert head == "not sent", head
                lead = group.locator("[data-review-wave-failure]")
                assert lead.count() == 1 and LEAD in lead.inner_text()
                seats = group.locator("[data-review-opinion]")
                assert seats.count() == 3 and "No cause was recorded." not in group.inner_text()
                shown = group.locator(".review-seats").inner_text()
                assert not re.search(r"REVIEW_BLOCKED|infra_failure|rejected|Errno", shown), shown
                assert card.evaluate("el => el.scrollWidth <= el.clientWidth + 1")
                assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
                card.screenshot(path=str(evidence / f"stopped-{tag}.png"))

                # Read: the panel's saved note, complete and masked, in the existing reader; Escape returns focus.
                read = lead.locator("[data-review-read-url]")
                source = read.get_attribute("data-review-read-url")
                response = page.request.get(url + source)
                assert response.status == 200, response.status
                exported = response.body().decode("utf-8")
                assert TAIL in exported and "No reviewer was started" in exported and SECRET[:16] not in exported
                read.focus()
                page.keyboard.press("Enter")
                reader = page.locator("dialog[open]:has(.document-reader-panel)")
                reader.wait_for(state="visible")
                page.wait_for_function("tail => document.querySelector('dialog[open] .document-reader-body')?.textContent.includes(tail)",
                                       arg=TAIL)
                assert SECRET[:16] not in reader.inner_text()
                page.screenshot(path=str(evidence / f"reader-{tag}.png"))
                page.keyboard.press("Escape")
                reader.wait_for(state="hidden")
                assert read.evaluate("el => el === document.activeElement")

                # Record facts keep the code and the producer's words; the whole stays one Read away.
                facts = group.locator("[data-review-facts]")
                facts.locator("summary").click()
                body = facts.locator(".review-facts-body")
                text = body.inner_text()
                assert "Wave failure" in text and "infra_failure" in text and "custody could not be reserved" in text
                assert "Saved wave failure note" in text and SECRET[:16] not in text
                body.evaluate("el => { el.scrollTop = 0; }")
                card.screenshot(path=str(evidence / f"stopped-facts-{tag}.png"))
                # The row itself, centred in the viewport (clear of the fixed composer on a phone).
                body.locator("dt", has_text=re.compile(r"^Wave failure$")).evaluate("el => el.scrollIntoView({block: 'center'})")
                page.wait_for_timeout(150)
                page.screenshot(path=str(evidence / f"wave-facts-{tag}.png"))

                # The record the earlier operation wrote: unchanged, its seats still say no cause was recorded.
                earlier, earlier_group = _open_record(page, EARLIER)
                assert earlier_group.locator("[data-review-wave-failure]").count() == 0
                assert "No cause was recorded." in earlier_group.inner_text()
                earlier.screenshot(path=str(evidence / f"earlier-{tag}.png"))

                # Answers beside a stopped wave: every received answer keeps its own line and the review its cause.
                answered, answered_group = _open_record(page, ANSWERED)
                assert answered_group.locator("[data-review-wave-failure]").count() == 1
                assert "rejected it" in answered_group.inner_text() and "passed it" in answered_group.inner_text()
                answered.screenshot(path=str(evidence / f"answered-{tag}.png"))
                assert not errors, errors
                results.append({"theme": theme, "width": width, "height": height, "head": head,
                                "export_sha256": response.headers.get("x-review-export-sha256"), "page_errors": errors})
                page.close()
        finally:
            browser.close()
    (evidence / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
