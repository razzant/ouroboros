"""Browser witness for two `stop_service` runs of the same service label.

Uses the service test helpers (`start_demo_report`, `stop_demo_report`) on one real
`ToolRegistry` bound to `direct_server_with_data["data_dir"]`:
1. Opens the real UI page and card Details BEFORE running A and B (0 rows).
2. Runs A (`report.txt` saved), relays the real `_emit_checkpoint_event` frame
   onto the live WebSocket -> 1 visible entry matching A (`b"run-a"`).
3. Runs B (`report.txt` saved + `not-created.txt` failed), relays its real frame
   onto the live WebSocket -> 2 visible entries (A and B bound separately, exact
   bytes downloaded by their rendered `download_url` links).
4. Reloads without WS injection -> reopened Details matches A and B from server.
5. Opens Dashboard -> Logs -> expands card and both `details.service-stop-result`,
   verifying A and B entries and byte-exact downloads.

Captures 3 PNGs and 1 short JSON summary per engine (chromium, webkit).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import queue
from urllib.parse import urljoin

import pytest

from ouroboros import dialogue_store
from ouroboros.task_results import write_task_result
from ouroboros.test_evidence import diagnostic_directory
from ouroboros.tools.registry import ToolRegistry
from ouroboros.utils import utc_now_iso
from tests.test_services_tool_v2 import start_demo_report, stop_demo_report
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data
from tests.ui_chat_viewport_smoke import (
    _CAPTURE_TEST_SOCKET,
    _OBSERVE_STATE_READS,
    _emit_ws_frame,
    _wait_socket_open_quiescent,
)

pytestmark = [pytest.mark.ui_browser]


def _evidence_dir(data_dir: Path) -> Path:
    configured = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", "").strip()
    target = Path(configured) if configured else diagnostic_directory(
        data_dir.parent, "service-stop-checkpoint", publish=True,
    )
    target.mkdir(parents=True, exist_ok=True)
    return target


def _next_service_stop_frame(event_q: queue.Queue, expected_stop: dict) -> dict:
    """Read the matching `log_event` checkpoint emitted to `event_queue`."""
    while True:
        item = event_q.get(timeout=5)
        payload = item.get("data") if isinstance(item, dict) and item.get("type") == "log_event" else None
        if (
            isinstance(payload, dict)
            and payload.get("type") == "task_checkpoint"
            and payload.get("checkpoint_kind") == "service_stop"
        ):
            assert payload.get("service_stop") == expected_stop, (payload.get("service_stop"), expected_stop)
            return payload


def _open_live_card(card):
    """Expand a chat live card only if collapsed, then wait for its timeline."""
    from playwright.sync_api import expect

    card.wait_for(state="attached", timeout=30_000)
    if card.get_attribute("data-expanded") != "1":
        card.locator(":scope > [data-live-summary-button] [data-live-disclosure]").click()
    timeline = card.locator(":scope > [data-live-timeline]")
    expect(timeline).to_be_visible(timeout=15_000)
    return timeline


def _ensure_chat_stop_rows_visible(timeline, expected_n: int) -> None:
    """Wait for `expected_n` `.service-stop-result` rows and uncollapse any hidden link."""
    from playwright.sync_api import expect

    rows = timeline.locator(".service-stop-result")
    expect(rows).to_have_count(expected_n, timeout=15_000)
    for idx in range(expected_n):
        row = rows.nth(idx)
        link = row.locator(".service-stop-output a").first
        if link.count() == 0 or link.is_visible():
            continue
        toggle = row.locator("[data-live-line-toggle]").first
        if toggle.count() == 0:
            parent_line = row.locator(
                'xpath=ancestor::*[contains(concat(" ", normalize-space(@class), " "), " chat-live-line ")][1]'
            )
            toggle = parent_line.locator("[data-live-line-toggle]").first
        if (
            toggle.count() > 0
            and toggle.is_visible()
            and toggle.is_enabled()
            and toggle.get_attribute("aria-expanded") == "false"
        ):
            toggle.click()


def _verify_surface_rows(
    page,
    base_url: str,
    container,
    stop_a: dict,
    stop_b: dict,
) -> dict[str, str]:
    """Assert exactly two `.service-stop-result` entries bound individually to A and B."""
    from playwright.sync_api import expect

    rows = container.locator(".service-stop-result")
    expect(rows).to_have_count(2, timeout=15_000)

    url_a = stop_a["outputs"][0]["files"][0]["download_url"]
    url_b = stop_b["outputs"][0]["files"][0]["download_url"]
    assert url_a != url_b
    assert url_a.startswith("/") and not url_a.startswith("//"), url_a
    assert url_b.startswith("/") and not url_b.startswith("//"), url_b

    seen: dict[str, str] = {}
    for idx in range(2):
        row = rows.nth(idx)
        text = row.inner_text()
        assert "Stopped" in text and "Journal: saved" in text and "report.txt: saved" in text, text
        links = row.locator(".service-stop-output a")
        assert links.count() == 1, (idx, links.count(), text)
        link = links.first
        assert link.is_visible(), (idx, text)
        assert link.inner_text().strip() == "report.txt", (idx, link.inner_text(), text)
        href = link.get_attribute("href") or ""

        is_b = "not-created.txt: not saved" in text
        label = "B" if is_b else "A"
        expected_url = url_b if is_b else url_a
        expected_bytes = b"run-b" if is_b else b"run-a"
        assert href == expected_url, (label, href, expected_url)

        resp = page.request.get(urljoin(base_url, href))
        assert resp.status == 200, (label, href, resp.status)
        body = resp.body()
        assert body == expected_bytes, (label, body, expected_bytes)
        assert label not in seen, f"duplicate row for {label}"
        seen[label] = hashlib.sha256(body).hexdigest()

    assert set(seen) == {"A", "B"}, seen
    return seen


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_service_stop_two_launches_preserve_and_render_distinct_outputs(
    direct_server_with_data, engine,
):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import expect, sync_playwright

    url = direct_server_with_data["url"]
    data_dir = Path(direct_server_with_data["data_dir"])
    repo_dir = Path(direct_server_with_data.get("repo_dir") or (data_dir.parent / "repo"))
    repo_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = _evidence_dir(data_dir)

    task_id = f"task-service-stop-{engine}"
    write_task_result(
        data_dir,
        task_id,
        "running",
        description="Verify two service_stop runs for the same service label",
        suggested_name=f"Service stop ({engine})",
        chat_id=1,
    )
    dialogue_store.append(
        data_dir,
        "progress",
        {
            "ts": utc_now_iso(),
            "type": "send_message",
            "is_progress": True,
            "chat_id": 1,
            "task_id": task_id,
            "suggested_name": f"Service stop ({engine})",
            "content": "Service stop witness initialized",
        },
    )

    event_q: queue.Queue = queue.Queue()
    reg = ToolRegistry(repo_dir=repo_dir, drive_root=data_dir)
    reg._ctx.task_id = task_id
    reg._ctx.current_chat_id = 1
    reg._ctx._execution_trace = {}
    reg._ctx.event_queue = event_q

    page_errors: list[str] = []
    console_errors: list[str] = []
    screenshots: dict[str, str] = {}

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        try:
            page = browser.new_context(viewport={"width": 1280, "height": 800}).new_page()
            page.on("pageerror", lambda err: page_errors.append(str(err)))
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})();({_OBSERVE_STATE_READS})()")
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            _wait_socket_open_quiescent(page)

            # 1) Open real card Details BEFORE A and B -> 0 entries.
            card = page.locator(f'.chat-live-card[data-task-id="{task_id}"]')
            timeline = _open_live_card(card)
            expect(timeline.locator(".service-stop-result")).to_have_count(0)

            # 2) Run A -> relay real checkpoint to WS -> 1 entry.
            started_a = start_demo_report(reg, "run-a")
            assert started_a["ready"] is True
            stop_a = stop_demo_report(reg)
            assert stop_a["launch_id"] == started_a["launch_id"]
            assert [o["state"] for o in stop_a["outputs"]] == ["saved"]
            frame_a = _next_service_stop_frame(event_q, stop_a)
            _emit_ws_frame(page, {"type": "log", "chat_id": 1, "data": frame_a})
            _ensure_chat_stop_rows_visible(timeline, 1)

            # 3) Run B -> relay real checkpoint to WS -> 2 distinct entries bound to A and B.
            started_b = start_demo_report(reg, "run-b", missing=True)
            assert started_b["ready"] is True and started_b["launch_id"] != started_a["launch_id"]
            stop_b = stop_demo_report(reg)
            assert stop_b["launch_id"] == started_b["launch_id"]
            assert stop_b["outputs"][0]["state"] == "saved"
            assert stop_b["outputs"][1] == {"name": "not-created.txt", "state": "failed", "files": []}
            frame_b = _next_service_stop_frame(event_q, stop_b)
            _emit_ws_frame(page, {"type": "log", "chat_id": 1, "data": frame_b})
            _ensure_chat_stop_rows_visible(timeline, 2)
            live_hashes = _verify_surface_rows(page, url, timeline, stop_a, stop_b)

            live_png = evidence_dir / f"service-stop-{engine}-live-details.png"
            page.screenshot(path=str(live_png), full_page=True)
            screenshots["live_details"] = str(live_png)

            # 4) Reload WITHOUT WS injection -> reopened Details reads from server history.
            page.reload(wait_until="domcontentloaded", timeout=30_000)
            _wait_socket_open_quiescent(page)
            reloaded_card = page.locator(f'.chat-live-card[data-task-id="{task_id}"]')
            reloaded_timeline = _open_live_card(reloaded_card)
            _ensure_chat_stop_rows_visible(reloaded_timeline, 2)
            reopened_hashes = _verify_surface_rows(page, url, reloaded_timeline, stop_a, stop_b)

            reopened_png = evidence_dir / f"service-stop-{engine}-reopened-details.png"
            page.screenshot(path=str(reopened_png), full_page=True)
            screenshots["reopened_details"] = str(reopened_png)

            # 5) Dashboard -> Logs -> open task card + both `details.service-stop-result`.
            page.locator('[data-nav-page="dashboard"]').click()
            page.locator('[data-dashboard-tab="logs"]').click()
            log_card = page.locator(f'.log-task-card[data-task-group="{task_id}"]')
            log_card.wait_for(state="attached", timeout=30_000)
            if log_card.get_attribute("open") is None:
                log_card.locator(":scope > summary").click()
            log_stops = log_card.locator("details.service-stop-result")
            expect(log_stops).to_have_count(2, timeout=15_000)
            for idx in range(2):
                if log_stops.nth(idx).get_attribute("open") is None:
                    log_stops.nth(idx).locator(":scope > summary").click()
            logs_hashes = _verify_surface_rows(page, url, log_card, stop_a, stop_b)

            log_card.scroll_into_view_if_needed()
            logs_png = evidence_dir / f"service-stop-{engine}-logs.png"
            page.screenshot(path=str(logs_png), full_page=True)
            screenshots["logs"] = str(logs_png)

            assert not page_errors, page_errors
            assert not console_errors, console_errors
        finally:
            browser.close()

    summary_path = evidence_dir / f"service-stop-{engine}-summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "engine": engine,
                "task_id": task_id,
                "launch_a": stop_a["launch_id"],
                "launch_b": stop_b["launch_id"],
                "sha256": {"live": live_hashes, "reopened": reopened_hashes, "logs": logs_hashes},
                "screenshots": screenshots,
                "errors": {"page": page_errors, "console": console_errors},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
