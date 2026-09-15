"""Project task-card facts survive real browser hydration and panel recreation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401


@pytest.mark.serial
@pytest.mark.ui_browser
@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
def test_bound_direct_task_header_and_review_cost_survive_reopen(
    direct_server_with_data, browser_engine, monkeypatch,  # noqa: F811
):
    """Exercise Project open/reopen, reference metadata and durable completion.

    The real server owns Project registration, binding, history and task detail.
    Its activity fragment comes from the production census builder with an
    explicit direct-registry fixture, so no model task or paid call is needed.
    """
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import expect, sync_playwright

    import supervisor.queue as queue_mod
    from ouroboros.gateway.state import _chat_activities_snapshot_safe
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from ouroboros.task_results import STATUS_COMPLETED, STATUS_RUNNING, write_task_result

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    project_id, task_id = "card-recovery", "card-recovery-root"
    project = create_project(data_dir, project_id, name="Context anatomy")
    chat_id = int(project["chat_id"])
    bind_task_to_project(data_dir, task_id, project_id, chat_id, origin={"absent": "system"})
    write_task_result(
        data_dir, task_id, STATUS_RUNNING, chat_id=1, project_id="",
        suggested_name="Analyze greeting context", _is_direct_chat=True,
    )
    progress = data_dir / "logs" / "progress.jsonl"
    progress.parent.mkdir(parents=True, exist_ok=True)
    rows = [{
        "task_id": task_id, "chat_id": 1, "direction": "out", "is_progress": True,
        "content": text, "ts": f"2026-09-15T11:05:0{index}+00:00", "cancelable": True,
    } for index, text in enumerate([
        "Reading the context sources.",
        "Task acceptance review is running; Main can receive and answer messages.",
    ])]
    reference = {
        "type": "review_reference", "task_id": task_id, "chat_id": 1,
        "presentation_owner_task_id": task_id, "surface": "task_acceptance",
        "state_revision": "a" * 64, "is_progress": True,
        "ts": "2026-09-15T11:05:03+00:00",
        "cost_accounting_status": "available", "cost_final": False,
        "accounted_upper_bound_usd_with_children": 2.75,
    }
    progress.write_text("".join(json.dumps(row) + "\n" for row in [*rows, reference]), encoding="utf-8")

    # No live global queue or registry is consulted by the fixture fragment.
    monkeypatch.setattr(queue_mod, "PENDING", [])
    monkeypatch.setattr(queue_mod, "RUNNING", {})
    monkeypatch.setattr(queue_mod, "BUDGET_ROOT_FENCES", {})
    active = {"running": True}
    original = {
        "activity_id": task_id, "chat_id": 1, "project_id": "", "kind": "direct_chat",
        "phase": "thinking", "client_message_id": "fixture-owner-message", "started_at": 1.0,
    }
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(data_dir.parent)))
    evidence.mkdir(parents=True, exist_ok=True)
    card_selector = f'#panel-pchat-{project_id} .chat-live-card[data-task-id="{task_id}"]'

    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        try:
            def project_activity(route):
                response = route.fetch()
                payload = response.json()
                direct = [dict(original)] if active["running"] else []
                availability = {"complete": True}
                payload["active_direct_turns"] = direct
                payload["active_chat_activities"] = _chat_activities_snapshot_safe(
                    data_dir, payload["task_bindings"], direct_turns=direct,
                    availability=availability,
                )
                payload["active_chat_activities_complete"] = availability["complete"]
                route.fulfill(content_type="application/json", body=json.dumps(payload))

            page.route("**/api/state", project_activity)
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            project_row = page.locator(f'.nav-project-row[data-project-id="{project_id}"]')
            project_row.wait_for(state="visible", timeout=30_000)

            def open_project():
                project_row.click()
                page.locator("#project-panel").wait_for(state="visible")
                page.locator(card_selector).wait_for(state="visible", timeout=15_000)

            def close_project():
                page.locator("#project-panel-close").click()
                page.wait_for_function(
                    "() => !document.getElementById('project-panel').classList.contains('open')"
                )

            def screenshot(stage):
                page.screenshot(path=str(evidence / f"task-card-{browser_engine}-{stage}.png"), animations="disabled")

            def assert_running(amount):
                card = page.locator(card_selector)
                # The subject is a DIRECT turn, so the block wears the compact
                # chrome (DESIGN.md "Conversation activity block"): no
                # Working/Done chip, a running indicator while the turn runs.
                # The phase therefore survives as the record's fact the block
                # logic reads, not as a visible chip.
                expect(card).to_have_attribute("data-direct", "1", timeout=10_000)
                expect(card.locator("[data-live-phase]")).to_have_attribute("data-phase", "working")
                expect(card.locator("[data-live-typing]")).to_be_visible()
                expect(card.locator("[data-live-meta]")).not_to_contain_text("Activity unconfirmed")
                expect(card.locator("[data-live-meta]")).to_contain_text(f"up to ${amount}")
                expect(card).to_have_attribute("data-finished", "0")

            open_project()
            assert_running("2.75")
            screenshot("opened")
            count = page.locator(f"{card_selector} [data-live-count]").inner_text()

            # Same review revision: metadata must update even when its detail
            # hydration is deduplicated. A reference does not add an activity note.
            updated = {**reference, "chat_id": chat_id, "ts": "2026-09-15T11:05:04+00:00",
                       "accounted_upper_bound_usd_with_children": 3.50}
            with progress.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(updated) + "\n")
            page.evaluate(
                "row => window.__ouroWs.emit('log', {chat_id: row.chat_id, data: row})", updated,
            )
            assert_running("3.50")
            expect(page.locator(f"{card_selector} [data-live-count]")).to_have_text(count)

            close_project()
            open_project()
            assert_running("3.50")
            screenshot("reopened")

            # A lost terminal event must recover from the durable result when
            # the direct activity leaves the census; absence alone is not Done.
            terminal = write_task_result(
                data_dir, task_id, STATUS_COMPLETED, chat_id=1, project_id="",
                suggested_name="Analyze greeting context", result="Context analysis completed.",
                cost_accounting_status="available", cost_final=True,
                accounted_upper_bound_usd_with_children=3.50,
            )
            active["running"] = False
            page.evaluate("() => window.__ouroWs.emit('projects_changed', {})")
            card = page.locator(card_selector)
            expect(card).to_have_attribute("data-finished", "1", timeout=15_000)
            expect(card.locator("[data-live-phase]")).to_have_attribute("data-phase", "done")
            expect(card.locator("[data-live-typing]")).not_to_be_visible()
            page.evaluate(
                "row => window.__ouroWs.emit('log', {chat_id: row.chat_id, data: row})",
                {**terminal, "type": "task_done", "chat_id": chat_id},
            )
            close_project()
            open_project()
            expect(card).to_have_attribute("data-finished", "1")
            expect(card.locator("[data-live-phase]")).to_have_attribute("data-phase", "done")
            screenshot("completed")
        except Exception:
            page.screenshot(path=str(evidence / f"task-card-{browser_engine}-failed.png"), animations="disabled")
            raise
        finally:
            browser.close()
