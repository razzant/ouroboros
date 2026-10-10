"""Render the real outbox consumer's receipt frames and its history projection."""
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import test_merge_receipt_qualification as qualification
from tests import test_pr_merge_receipts as merge_fixture
from tests import test_subscription_setup_browser as ui_fixture
from tests.test_merge_receipt_qualification import (
    test_concurrent_publication_through_outbox_dedup_and_history as exercise_publication,
)

world = merge_fixture.world
subscription_ui = ui_fixture.subscription_ui

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def assert_review_source(text, source):
    other = "record" if source == "declaration" else "declaration"
    assert f"review source: {source}" in text and f"review source: {other}" not in text


@pytest.mark.parametrize("source", ["declaration", "record"])
def test_merge_card_stays_current_after_outbox_delivery_and_reload(request, world, monkeypatch, tmp_path, source):
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.utils import append_jsonl

    append_jsonl(world.root / "logs/progress.jsonl", {
        "type": "task_progress", "chat_id": 7, "task_id": "merge-task",
        "text": "Reading the PR.", "ts": "2026-09-16T00:00:00Z",
    })
    suffix = "" if source == "declaration" else "-record"
    if source == "record":
        qualification.install_ledger(monkeypatch, {qualification.RECORD_ID: qualification.ledger_record()})
        monkeypatch.setattr(qualification, "registered_merge", lambda world: qualification.record_merge(
            world, review_record_id=qualification.RECORD_ID))
    exercise_publication(world, monkeypatch, "buffered")
    history = json.loads(asyncio.run(make_chat_history_endpoint(world.root)(
        SimpleNamespace(query_params={"chat_id": "7"}))).body)
    fixture = json.loads((world.root / "delivery-projection.json").read_text())
    ui = request.getfixturevalue("subscription_ui")
    page, url = ui["page"], ui["url"]
    page.route("**/receipt-projection", lambda route: route.fulfill(content_type="text/html", body='''
        <!doctype html><html><head><link rel="stylesheet" href="/static/ui.css">
        <link rel="stylesheet" href="/static/style.css"></head><body><main id="content"></main></body></html>'''))
    page.route("**/api/chat/history?*", lambda route: route.fulfill(
        content_type="application/json", body=json.dumps(history)))
    page.goto(url + "/receipt-projection")
    page.evaluate('''async () => {
        const { createChatInstance } = await import('/static/modules/chat.js');
        const handlers = new Map();
        window.receiptChat = createChatInstance({
            ws: { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
                  isConnected: () => true, send() {} },
            state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 }, updateUnreadBadge() {},
            stateSnapshots: { begin: () => ({generation: 1, requestedAt: Date.now()}),
                gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {} },
            chatId: 7, mountEl: document.querySelector('#content'), asPanel: true,
        });
        window.receiptSend = row => handlers.get('chat')({...row, chat_id: 7});
        receiptSend({task_id: 'merge-task', role: 'system', is_progress: true,
                     content: 'Reading the PR.', ts: '2026-09-16T00:00:00Z'});
    }''')
    output = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or tmp_path)
    output.mkdir(parents=True, exist_ok=True)
    try:
        for row in fixture["live"]:
            page.evaluate("row => receiptSend(row)", row)
        card = page.locator('.chat-live-card[data-task-id="merge-task"]')
        if card.get_attribute("data-expanded") != "1":
            card.locator('[data-live-summary-button]').click()
        page.screenshot(path=str(output / f"merge-receipt-live{suffix}.png"), full_page=True)
        line = card.locator('.chat-live-line.result')
        assert line.count() == 1 and "merge: merged" in line.inner_text()
        assert_review_source(line.inner_text(), source)
        replay = page.evaluate("() => receiptChat.refreshHistory({revision: 1})")
        assert replay["painted"], "receipt history must paint successfully"
        card = page.locator('.chat-live-card[data-task-id="merge-task"]')
        if card.get_attribute("data-expanded") != "1":
            card.locator('[data-live-summary-button]').click()
        page.screenshot(path=str(output / f"merge-receipt-replay{suffix}.png"), full_page=True)
        assert not page.get_by_role("button", name="Retry loading messages").is_visible()
        line = card.locator('.chat-live-line.result')
        assert line.count() == 1 and "merge: merged" in line.inner_text()
        assert_review_source(line.inner_text(), source)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    finally:
        page.evaluate("() => receiptChat.destroy()")
    print(f"VISUAL_EVIDENCE {output}")


@pytest.mark.parametrize("placement", ["first", "last", "orphan"])
def test_cold_bounded_receipt_renders_without_a_prior_live_revision(request, world, monkeypatch, tmp_path, placement):
    from tests.test_merge_receipt_qualification import bounded_receipt_history

    payload = bounded_receipt_history(world, monkeypatch)
    placed = [row for row in payload["messages"] if row.get("card_row")]
    ordinary = [row for row in payload["messages"] if not row.get("card_row")]
    assert len(placed) == 1 and len(ordinary) == 59
    if placement == "last":
        payload["messages"] = ordinary + placed
    elif placement == "orphan":
        payload["messages"] = placed
        placed[0]["system_type"] = "future_system_fact"
    ui = request.getfixturevalue("subscription_ui")
    page = ui["page"]
    page.route("**/cold-receipt", lambda route: route.fulfill(content_type="text/html", body='''
        <!doctype html><html><head><link rel="stylesheet" href="/static/ui.css">
        <link rel="stylesheet" href="/static/style.css"></head><body><main id="content"></main></body></html>'''))
    page.route("**/api/chat/history?*", lambda route: route.fulfill(
        content_type="application/json", body=json.dumps(payload)))
    page.goto(ui["url"] + "/cold-receipt")
    page.evaluate('''async () => {
        const { createChatInstance } = await import('/static/modules/chat.js');
        window.receiptChat = createChatInstance({
            ws: { on() { return () => {}; }, isConnected: () => true, send() {} },
            state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 }, updateUnreadBadge() {},
            stateSnapshots: { begin: () => ({generation: 1, requestedAt: Date.now()}),
                gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {} },
            chatId: 7, mountEl: document.querySelector('#content'), asPanel: true,
        });
        await receiptChat.refreshHistory({revision: 1});
    }''')
    output = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or tmp_path)
    output.mkdir(parents=True, exist_ok=True)
    try:
        card = page.locator('.chat-live-card[data-task-id="merge-task"]')
        if placement == "orphan":
            assert card.count() == 0
            assert page.locator('.chat-bubble').filter(has_text="merge: merged").count() == 1
            page.screenshot(path=str(output / "merge-receipt-cold-orphan.png"), full_page=True)
            return
        if card.get_attribute("data-expanded") != "1":
            card.locator('[data-live-summary-button]').click()
        page.screenshot(path=str(output / f"merge-receipt-cold-{placement}.png"), full_page=True)
        assert page.locator('.chat-live-line').filter(has_text="merge: merged").count() == 1
        line = card.locator('.chat-live-line.result')
        assert line.count() == 1 and "merge: merged" in line.inner_text()
        assert "merge: queued" not in line.inner_text()
        # Same page identity can refresh current truth without moving the reader's card.
        card.evaluate("e => { window.readingCard = e; }")
        page.evaluate("() => receiptChat.refreshHistory({revision: 2})")
        assert card.evaluate("e => e === window.readingCard && e.dataset.expanded === '1'")
        assert page.locator('.chat-live-line').filter(has_text="merge: merged").count() == 1
    finally:
        page.evaluate("() => receiptChat.destroy()")
