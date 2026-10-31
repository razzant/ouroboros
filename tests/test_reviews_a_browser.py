"""Reviews A in real WebKit: isolated server → ledger → task detail → closed card → keyed update → reader.

The records are written by the ledger's own builder and a pending one is settled by the real late-settlement
path (``settle_from_collected``); the card reads them through the real task detail, the hydrator and the keyed
renderer, and Read opens the existing document reader. No mock HTML runtime and no paid calls: the fixture
serves this checkout with a local fake provider and reaps its server before releasing the copy. ``BODY`` and
``EVIDENCE`` are module seams a QA wrapper may set (a retained real answer, a directory for the frames).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import review_ledger as rl
from ouroboros.agent_task_pipeline import _store_task_result
from ouroboros.post_task_synthesis import _record_task_facts
from tests.test_review_record_card_projection import CRITICAL, _answer, _gate_record
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET, _emit_ws_frame

pytest_plugins = ("tests.test_ui_smoke_playwright",)
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
TASK, QUIET = "reviews-a-browser", "reviews-a-quiet"
COMBOS = (("dark", 1100, 750), ("light", 1100, 750), ("dark", 390, 844), ("light", 390, 844))
LIVE = {f"{theme}-{width}": f"reviews-a-live-{theme}-{width}" for theme, width, _height in COMBOS}
BODY = "# Saved reviewer answer\n\n" + "".join(
    f"## Finding {n}\n\nThe change keeps record {n} readable; its neighbour stays visible when one record breaks.\n\n"
    f"- evidence: `review_projection.py` line {100 + n}\n- fix: keep each read local\n\n" for n in range(40)) + "Final limits: TAIL\n"
EVIDENCE: Path | None = None
TOKENS = re.compile(r"PASS|FAIL|QUORUM_FAILED|NOT_PERFORMED|NOT_DISPATCHED|DEGRADED|UNKNOWN|\bterminal\b")
STATES = """root => [...root.querySelectorAll('.review-state')].map(node => node.textContent.trim())"""
SETTLE = """() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))"""


def _store(root, repo, task_id, text):
    env, task = SimpleNamespace(drive_root=root, repo_dir=repo), {"id": task_id, "type": "task", "chat_id": 1}
    _store_task_result(env, task, text, {"rounds": 1, "cost": 0}, {"tool_calls": []})
    _record_task_facts(env, task, {"rounds": 1}, {"tool_calls": []}, root / "logs")


def _retain(root, task_id, record, seats):
    for seat in record.rows:
        if seat["seat_id"] in seats:
            seat["source_refs"].append(rl.retain_text_source(
                root, task_id, record_id=record.record_id, seat_id=seat["seat_id"], role="response",
                part=",".join(seat["parts"]), text=BODY))
    return rl.write_record(root, record)


def seed(fixture):
    """Seven saved commit reviews (the newest file lost), a quiet task stored after the index rotated, and one
    task per viewport whose newest review is still waiting for one counted reviewer."""
    root, repo = fixture["data_dir"], fixture["repo_dir"]
    fixture["stop_server"]()
    records = []
    for n in range(7):
        record = _gate_record(TASK, opus_findings=12)
        record.brief["goal"] = f"Saved changes {n}"
        records.append(_retain(root, TASK, record, {"sol", "opus"}))
    rl.record_path(root, records[-1]["record_id"]).unlink()
    _store(root, repo, TASK, "Reviews A check")
    pending = {}
    for key, task_id in LIVE.items():
        record = _gate_record(task_id, opus_pending=True)
        record.brief["goal"] = "Live change"
        pending[key] = _retain(root, task_id, record, {"sol"})["record_id"]
        _store(root, repo, task_id, "Live review check")
    # An archived index segment makes every record-less task's omitted count "unknown" (R1's phantom card).
    (rl.ledger_dir(root) / "index.20261001T000000000000.jsonl").write_text("", encoding="utf-8")
    _store(root, repo, QUIET, "A quiet answer")
    fixture["start_server"]()
    return pending


def _settle(fixture, task_id, record_id):
    """The opus seat's late answer arrives: the real settlement, then the task's own terminal write."""
    root = fixture["data_dir"]
    rl.settle_from_collected(root, record_id, [{
        "slot_id": "opus", "model_id": "anthropic/claude-opus-5", "status": "responded", "parts": ["change"],
        "answers": {"change": _answer("change", "FAIL", [CRITICAL])}, "raw_text": "The retry loop never ends.",
    }])
    _store(root, fixture["repo_dir"], task_id, "Live review check")


def _reference(task_id, revision):
    return {"type": "chat", "role": "system", "system_type": "review_reference", "is_progress": True,
            "task_id": task_id, "chat_id": 1, "surface": "task_acceptance", "state_revision": revision * 64,
            "text": "", "ts": "2026-10-25T19:00:00Z"}


def _focus_probe(page, selector):
    """Focus a control by keyboard and mark the node, so the same node can be recognized after an update."""
    page.locator(selector).focus()
    page.evaluate("""sel => { const node = document.querySelector(sel); node.__reviewsProbe = true; }""", selector)


def _measure(page, selector):
    return page.evaluate("""sel => { const node = document.querySelector(sel); if (!node) return null;
        const s = getComputedStyle(node); return {focusVisible: node.matches(':focus-visible'), fontSize: s.fontSize, fontWeight: s.fontWeight, color: s.color,
        outlineStyle: s.outlineStyle, outlineWidth: s.outlineWidth, outlineColor: s.outlineColor, outlineOffset: s.outlineOffset}; }""",
        selector)


def test_reviews_a_real_webkit_card_reader_and_history(direct_server_with_data, tmp_path):
    from playwright.sync_api import sync_playwright

    pending = seed(direct_server_with_data)
    url = direct_server_with_data["url"]
    evidence = EVIDENCE or Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or tmp_path / "reviews-a-evidence")
    evidence.mkdir(parents=True, exist_ok=True)
    results = []
    with sync_playwright() as pw:
        browser = pw.webkit.launch(headless=True)
        try:
            for (theme, width, height), (key, live_task) in zip(COMBOS, LIVE.items()):
                page = browser.new_page(viewport={"width": width, "height": height}, accept_downloads=True)
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
                page.goto(url, wait_until="domcontentloaded")
                page.evaluate("theme => document.documentElement.dataset.theme = theme", theme)
                card = page.locator(f'.chat-live-card[data-task-id="{TASK}"]')
                card.wait_for(state="visible", timeout=30000)
                assert card.get_attribute("data-expanded") == "0"
                # R1 phantom: a quiet task with no review stays without a card, a door or the word "terminal".
                assert page.locator(f'.chat-live-card[data-task-id="{QUIET}"]:visible').count() == 0
                assert "terminal" not in page.locator("#chat-messages").inner_text()
                door = card.locator("[data-live-review-summary]")
                door.wait_for(state="visible", timeout=10000)
                assert door.inner_text().strip() == "Reviews 5"
                card.screenshot(path=str(evidence / f"closed-{theme}-{width}.png"))
                door.focus()
                page.keyboard.press("Enter")
                section = card.locator("[data-review-section]")
                section.wait_for(state="visible")
                assert door.get_attribute("aria-expanded") == "true"
                assert card.get_attribute("data-expanded") == "0", "Reviews never needs Details"
                page.wait_for_function("() => document.querySelector('[data-review-section]')?.textContent.includes('Saved changes')")
                door_focus = _measure(page, f'.chat-live-card[data-task-id="{TASK}"] [data-live-review-summary]')
                gap = section.locator('[data-review-group]', has_text="Review record unavailable")
                assert "could not be read" in gap.inner_text()
                assert not [word for word in section.evaluate(STATES) if TOKENS.search(word)], section.evaluate(STATES)
                group = section.locator("[data-review-group]", has_text="Saved changes 5")
                group.locator("[data-review-group-toggle]").click()
                seats = group.locator("[data-review-opinion]")
                assert seats.count() == 3
                assert "rejected it" in seats.nth(1).inner_text()
                assert "12 findings" in seats.nth(1).inner_text()
                assert "adviser, not counted" in seats.nth(2).inner_text()
                assert group.locator("[data-review-facts]").count() == 1
                assert not [word for word in group.evaluate(STATES) if TOKENS.search(word)]
                card.locator("[data-live-title]").scroll_into_view_if_needed()
                page.wait_for_timeout(250)
                page.screenshot(path=str(evidence / f"open-{theme}-{width}.png"))
                card.screenshot(path=str(evidence / f"card-{theme}-{width}.png"))
                assert card.evaluate("el => el.scrollWidth <= el.clientWidth + 1")
                assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
                type_scale = {name: _measure(page, f'.chat-live-card[data-task-id="{TASK}"] {sel}') for name, sel in (
                    ("subject", ".review-check-subject"), ("state", ".review-check-head .review-state"),
                    ("seat", ".review-seat-name"), ("line", ".review-line"), ("read", ".review-read"))}

                # Reader: the exact retained source, Download byte-identical, Escape returns focus to Read.
                read = group.locator("[data-review-read-url]").first
                source = read.get_attribute("data-review-read-url")
                response = page.request.get(url + source)
                assert response.status == 200 and response.body() == BODY.encode("utf-8")
                read.focus()
                read_focus = _measure(page, f'.chat-live-card[data-task-id="{TASK}"] [data-review-read-url]')
                page.keyboard.press("Enter")
                reader = page.locator("dialog[open]:has(.document-reader-panel)")
                reader.wait_for(state="visible")
                last = BODY.strip().splitlines()[-1].strip()
                page.wait_for_function("last => document.querySelector('dialog[open] .document-reader-body')?.textContent.includes(last)", arg=last)
                page.screenshot(path=str(evidence / f"reader-{theme}-{width}.png"))
                _emit_ws_frame(page, {"type": "task_done", "task_id": TASK, "status": "completed", "result": "live update",
                                      "review_projection": {"panels": []}})
                assert last in reader.locator(".document-reader-body").inner_text(), "an update never switches the selected source"
                with page.expect_download() as download:
                    reader.locator('[data-reader-action="download"]').click()
                saved = evidence / f"saved-{theme}-{width}.txt"
                download.value.save_as(saved)
                assert saved.read_bytes() == BODY.encode("utf-8")
                page.keyboard.press("Escape")
                reader.wait_for(state="hidden")
                assert read.evaluate("el => el === document.activeElement")

                # Record facts: opened by the owner, scrolled inside, focused; a keyed update keeps all three.
                facts = group.locator("[data-review-facts]")
                facts.locator("summary").click()
                body = facts.locator(".review-facts-body")
                body.evaluate("el => { el.scrollTop = 60; }")
                scrolled = body.evaluate("el => el.scrollTop")
                assert scrolled > 0, "Record facts scroll inside their own box"
                facts.locator("summary").focus()
                page.evaluate("() => { document.activeElement.__reviewsProbe = true; }")
                # A new review of the same task lands while the owner reads: the keyed update adds its row only.
                arrived = _gate_record(TASK, opus_findings=2)
                arrived.brief["goal"] = f"Arrived while reading {key}"
                _retain(direct_server_with_data["data_dir"], TASK, arrived, {"sol"})
                _store(direct_server_with_data["data_dir"], direct_server_with_data["repo_dir"], TASK, "Reviews A check")
                _emit_ws_frame(page, _reference(TASK, "a"))
                page.wait_for_function("task => document.querySelector(`.chat-live-card[data-task-id=\"${task}\"] [data-live-review-summary]`)?.textContent === 'Reviews 6'",
                                       arg=TASK, timeout=15000)
                assert "Arrived while reading" in section.inner_text()
                assert facts.get_attribute("open") is not None
                assert body.evaluate("el => el.scrollTop") == scrolled
                assert page.evaluate("() => document.activeElement?.__reviewsProbe === true")
                facts_text = body.inner_text()
                assert "Verdict" in facts_text and "FAIL" in facts_text, "the record's own tokens stay in Record facts"

                # History: earlier records outside the compact window are read on request.
                section.locator("[data-review-history]").click()
                page.wait_for_function("() => document.querySelector('[data-review-section]')?.textContent.includes('Saved changes 0')")

                # Real late settlement on this viewport's live task: pending → FAIL in place, focus kept.
                live = page.locator(f'.chat-live-card[data-task-id="{live_task}"]')
                live.scroll_into_view_if_needed()
                live.locator("[data-live-review-summary]").click()
                live_group = live.locator("[data-review-group]", has_text="Live change")
                live_group.locator("[data-review-group-toggle]").click()
                head = live_group.locator(".review-check-head .review-state")
                assert head.inner_text().strip() == "no verdict · 1 of 2 answered"
                waiting = live_group.locator('[data-review-opinion$=":opus"] .review-state')
                assert waiting.inner_text().strip() == "awaiting"
                page.wait_for_timeout(250)  # the chevron's turn ends before the frame
                live.screenshot(path=str(evidence / f"pending-{theme}-{width}.png"))
                _focus_probe(page, f'.chat-live-card[data-task-id="{live_task}"] [data-review-opinion$=":sol"] [data-review-read-url]')
                _settle(direct_server_with_data, live_task, pending[key])
                _emit_ws_frame(page, _reference(live_task, "c"))
                page.wait_for_function("""task => document.querySelector(`.chat-live-card[data-task-id="${task}"] .review-check-head .review-state`)
                    ?.textContent.trim() === 'rejected'""", arg=live_task, timeout=15000)
                assert waiting.inner_text().strip() == "rejected it"
                assert live_group.locator(".review-state", has_text="awaiting").count() == 0
                assert "awaiting" not in live_group.inner_text()
                assert page.evaluate("() => document.activeElement?.__reviewsProbe === true && document.activeElement.isConnected")
                assert live_group.locator("[data-review-group-toggle]").get_attribute("aria-expanded") == "true"
                assert live.get_attribute("data-expanded") == "0"
                page.evaluate(SETTLE)
                live.screenshot(path=str(evidence / f"settled-{theme}-{width}.png"))

                # A failed refresh stays visible on the closed Reviews door, and Retry reads the real task again.
                door.click()
                assert section.get_attribute("data-expanded") == "0"
                pattern = re.compile(r".*/api/tasks/" + TASK + r"(?:\?.*)?$")
                page.route(pattern, lambda route: route.fulfill(status=503, content_type="application/json",
                                                                body='{"error":"controlled read refusal"}'))
                _emit_ws_frame(page, _reference(TASK, "d"))
                retry = section.locator("[data-review-hydrate-retry]")
                retry.wait_for(state="visible", timeout=15000)
                assert "shown data may be incomplete" in section.inner_text()
                card.screenshot(path=str(evidence / f"hydration-{theme}-{width}.png"))
                page.unroute(pattern)
                retry.click()
                page.wait_for_function("() => !document.querySelector('[data-review-hydrate-retry]')")
                assert card.get_attribute("data-expanded") == "0"
                assert not errors, errors
                results.append({"theme": theme, "width": width, "height": height, "door": door_focus, "read_focus": read_focus,
                                "type": type_scale, "source_sha256": hashlib.sha256(response.body()).hexdigest(),
                                "download_sha256": hashlib.sha256(saved.read_bytes()).hexdigest(), "page_errors": errors})
                page.close()
        finally:
            browser.close()
    (evidence / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
