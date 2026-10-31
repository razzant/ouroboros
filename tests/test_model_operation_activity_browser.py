"""Real Main and Project components display model facts on desktop and touch: a short state
word on the chip with the title primary, and the exact phase and cost reason in the card's details
(owner 2026-10-12, option B)."""
from copy import deepcopy

import pytest

from tests._work_projection_shared import with_work_fields
from tests.test_project_work_card_browser import project_chat as project_chat, _evidence_dir
from tests.ui_chat_viewport_smoke import set_viewport

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


@pytest.mark.parametrize("chat_id", [1, 7], ids=["main", "project"])
def test_model_phase_card_header_reconnect_and_terminal(project_chat, chat_id, tmp_path):
    ui, page, expect = project_chat, project_chat["page"], project_chat["expect"]
    ui["rows"].clear()
    row = {"activity_id": "phase-task", "chat_id": chat_id, "project_id": "project" if chat_id == 7 else "",
           "client_message_id": "message", "kind": "direct_chat", "phase": "preparing", "task_attempt": 1}
    ui["state"]["active_chat_activities"] = [row]
    ui["rows"].append({"task_id": row["activity_id"], "chat_id": chat_id, "role": "assistant",
        "is_progress": True, "narration": True, "text": "Reading the requested sources.",
        "suggested_name": "Новый этап восстановления проектов",
        "cost_presentation": {"scope": "own", "tracked_amount": 0, "tracked_final": True, "accounting_open": False,
                              "has_unpriced": True, "has_rows": True},
        "ts": "2026-10-11T00:00:00Z", "history_id": "progress:phase-task"})
    page.reload()
    page.wait_for_selector("#chat-input")
    badge = page.locator("#chat-status")
    card = page.locator('.chat-live-card[data-task-id="phase-task"]')
    expect(badge).to_have_text("Preparing")
    card.wait_for(state="visible")
    chip = card.locator("[data-live-phase]")
    title = card.locator("[data-live-title]")
    disclosure = card.locator(":scope > [data-live-summary-button] [data-live-disclosure]")
    phase_detail = card.locator(":scope > .chat-live-details > [data-live-phase-detail]")
    cost_detail = card.locator(":scope > .chat-live-details > [data-live-cost-detail]")
    meta = card.locator(":scope > [data-live-foot] [data-live-meta]")  # the meta line leads the card's foot
    suffix = f"{ui['engine']}-{ui['theme']}-{ui['width']}-{chat_id}"
    # The title is primary and the amount keeps its honest word; the reason is not a caption.
    expect(title).to_have_text("Новый этап восстановления проектов")
    expect(meta).to_contain_text("Tracked: $0.00")
    assert "no price" not in meta.inner_text()
    expect(cost_detail).to_be_hidden()

    def details(exact):
        """The card's details say exactly what the chip shortens: opened by pointer, closed by keyboard."""
        expect(phase_detail).to_be_hidden()
        disclosure.click()
        expect(disclosure).to_have_attribute("aria-expanded", "true")
        expect(phase_detail).to_have_text(f"Model call: {exact}")
        expect(cost_detail).to_have_text("Cost: some steps have no price, so Tracked counts only the priced steps.")
        disclosure.focus()
        page.keyboard.press("Enter")
        expect(disclosure).to_have_attribute("aria-expanded", "false")
        expect(phase_detail).to_be_hidden()

    def plain(phase, text):
        row.pop("model_activity", None)
        row["phase"] = phase
        page.evaluate("state => window.chat.hydrateStateSnapshot(state)", with_work_fields(ui["state"]))
        expect(badge).to_have_text(text)
        expect(chip).to_have_text(text)

    plain("working", "Working")
    plain("thinking", "Thinking")
    plain("working", "Working")
    page.screenshot(path=str(_evidence_dir(tmp_path) / f"phase-neutral-{suffix}.png"), animations="disabled")

    def show(phase, text, exact=None, room="Working"):
        """The card's chip says the model operation; the header says the room's word from the server's work
        projection: its exact call (Preparing, Sending, Thinking), else Working."""
        row.update(phase="working", model_activity={"phase": phase, "operation_id": "same-operation",
                   "invocation_id": "same-send", "llm_call_id": "call", "task_attempt": 1})
        page.evaluate("state => window.chat.hydrateStateSnapshot(state)", with_work_fields(ui["state"]))
        expect(badge).to_have_text(room)
        expect(chip).to_have_text(text)
        assert card.locator("[data-wait-id]").count() == 0, "queue admission is not a quota/auth wait"
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
        assert card.evaluate("el => el.scrollWidth <= el.clientWidth + 1")
        if exact:
            details(exact)

    show("queued", "Queued", "Request accepted · waiting for admission")
    # A call waiting for admission goes on by itself: still dots on a neutral chip (status marks).
    expect(chip.locator(":scope > .status-mark")).to_have_attribute("data-shape", "still")
    expect(chip).to_have_attribute("data-chip-tone", "neutral")
    pausing = {"activity_id": "settling-other", "chat_id": chat_id, "kind": "managed_task", "phase": "budget_pausing"}
    ui["state"]["active_chat_activities"].append(pausing)
    page.evaluate("state => window.chat.hydrateStateSnapshot(state)", with_work_fields(ui["state"]))
    # Another work settling a Pause beside one that works: the room works.
    expect(badge).to_have_text("Working")
    expect(chip).to_have_text("Queued")
    page.screenshot(path=str(_evidence_dir(tmp_path) / f"phase-pausing-{suffix}.png"), animations="disabled")
    ui["state"]["active_chat_activities"].remove(pausing)
    page.evaluate("state => window.chat.hydrateStateSnapshot(state)", with_work_fields(ui["state"]))
    expect(badge).to_have_text("Working")
    page.screenshot(path=str(_evidence_dir(tmp_path) / f"phase-queued-{suffix}.png"), animations="disabled")
    # A cold reconnect recovers the current census, without any fresh typing frame.
    page.reload()
    expect(badge).to_have_text("Working")
    expect(chip).to_have_text("Queued")
    row["model_activity"]["blockers"] = [{"kind": "global_limit"}]
    page.evaluate("state => window.chat.hydrateStateSnapshot(state)", with_work_fields(ui["state"]))
    expect(badge).to_have_text("Working")
    expect(chip).to_have_text("Queued")
    details("Request accepted · waiting for admission · shared capacity limit")
    page.evaluate("chatId => window.emit('log', {data: {type: 'owner_hurry', phase: 'applied',"
        "task_id: 'phase-task', chat_id: chatId, is_progress: false, ts: '2026-10-11T00:00:00Z'}})", chat_id)
    hurry = card.locator('[data-live-hurry]')
    expect(hurry).to_be_visible()
    expect(hurry).to_have_attribute('aria-label', 'Owner hurry applied')
    assert hurry.locator('svg').count() == 1
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    assert badge.evaluate("el => el.getBoundingClientRect().width <= innerWidth")
    assert card.evaluate("el => el.scrollWidth <= el.clientWidth + 1")
    page.screenshot(path=str(_evidence_dir(tmp_path) / f"phase-capacity-{suffix}.png"), animations="disabled")
    show("dispatching", "Sending", "Preparing request dispatch", room="Sending")
    show("awaiting_response", "Sending", "Sending request / waiting for response", room="Sending")
    assert "Thinking" not in badge.inner_text()
    page.screenshot(path=str(_evidence_dir(tmp_path) / f"phase-awaiting-{suffix}.png"), animations="disabled")
    # The owner's 2026-10-12 view: title primary beside a short chip, at the native window
    # width, a phone and a narrower phone; details opened show the exact phase and cost reason.
    for width in (1100, 390, 320):
        set_viewport(page, width, 722)
        expect(title).to_be_visible()
        assert title.evaluate("el => el.getBoundingClientRect().width >= 120"), f"title squeezed at {width}px"
        assert chip.evaluate("el => el.getBoundingClientRect().width < 140"), f"chip crowds the title at {width}px"
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
        assert card.evaluate("el => el.scrollWidth <= el.clientWidth + 1")
        page.screenshot(path=str(_evidence_dir(tmp_path) / f"phase-short-{suffix}-{width}.png"), animations="disabled")
        disclosure.click()
        expect(phase_detail).to_have_text("Model call: Sending request / waiting for response")
        expect(cost_detail).to_be_visible()
        assert card.evaluate("el => el.scrollWidth <= el.clientWidth + 1")
        page.screenshot(path=str(_evidence_dir(tmp_path) / f"phase-details-{suffix}-{width}.png"), animations="disabled")
        disclosure.click()
    set_viewport(page, ui["width"], 850)
    show("rejoining", "Rejoining", "Rejoining the same model operation")
    expect(chip.locator(":scope > .status-mark")).to_have_attribute("data-shape", "still")
    show("response_ready", "Working", "Response available")
    show("response_received", "Working", "Response received · processing")
    stale = deepcopy(ui["state"])
    plain("working", "Working")
    page.evaluate("chatId => window.emit('chat', {chat_id: chatId, task_id: 'phase-task', role: 'assistant',"
        "content: 'Finished the requested work.', system_type: 'task_summary', task_terminal_status: 'completed',"
        "ts: '2026-10-11T00:00:01Z'})", chat_id)
    page.evaluate("state => window.chat.hydrateStateSnapshot(state)", with_work_fields(stale))
    expect(chip).to_have_text("Done")
    assert phase_detail.text_content() == "", "a stale model observation never survives the terminal"
    expect(chip.locator(":scope > .status-mark")).to_have_attribute("data-shape", "dot")
    expect(chip).to_have_attribute("data-chip-tone", "ok")
    page.screenshot(path=str(_evidence_dir(tmp_path) / f"phase-terminal-{suffix}.png"), animations="disabled")
