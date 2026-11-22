"""Browser witness for two `service_stop` checkpoints of the same service label.

Uses real host producers (`copy_file_to_task_artifacts`, `output_facts`,
`write_blob`, `publish_stop`) on a `ToolContext` bound to
`direct_server_with_data["data_dir"]` without changing or bypassing runtime safety checks:
1. Opens real card Details BEFORE A and B (0 `.service-stop-result` entries).
2. Publishes A (masked-log `full_log_ref` + `tail` canary -> `Journal: saved`, `report.txt: saved`),
   verifies 0 rows before dispatch, then dispatches a test `MessageEvent` via `_emit_ws_frame`
   to the page's real WS listener -> 1 visible entry matching A (`b"run-a"`).
3. Publishes B (`retained_live_log_path` + `tail` canary, no `full_log_ref` -> `Journal: retained`,
   `report.txt: saved`, `not-created.txt: not saved`), verifies 1 row before dispatch, then dispatches
   via `_emit_ws_frame` -> 2 distinct entries with separate immutable downloads and no journal link.
4. Reloads without WS injection -> reopened Details hydrates A and B from server HTTP history.
5. Opens Dashboard -> Logs -> expands task card and both `details.service-stop-result`, verifying
   A/B distinction, canary non-disclosure, non-journal extra anchors, and byte-exact downloads.

Captures 3 PNGs (basenames in JSON summary) and 1 short JSON summary per engine (chromium, webkit).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import queue
from urllib.parse import parse_qs, urljoin, urlsplit

import pytest

from ouroboros.artifacts import copy_file_to_task_artifacts
from ouroboros import dialogue_store
from ouroboros.observability import write_blob
from ouroboros.task_results import write_task_result
from ouroboros.test_evidence import diagnostic_directory
from ouroboros.tools.service_stop import output_facts, publish_stop
from ouroboros.tools.tool_context import ToolContext
from ouroboros.utils import utc_now_iso
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


def _publish_demo_stop(
    ctx: ToolContext,
    marker: str,
    *,
    canary: str,
    retained: bool = False,
    missing: bool = False,
) -> tuple[dict, str]:
    """Publish one service_stop checkpoint via real artifact and checkpoint producers."""
    task_drive = ctx.task_drive_root()
    task_drive.mkdir(parents=True, exist_ok=True)
    report_path = task_drive / "report.txt"
    report_path.write_bytes(marker.encode("utf-8"))
    if missing:
        assert not (task_drive / "not-created.txt").exists()

    record = copy_file_to_task_artifacts(ctx, report_path, kind="process_output")
    assert record is not None and record["name"] == "report.txt"
    declarations = ["report.txt", "not-created.txt"] if missing else ["report.txt"]
    reports = [{"status": "saved", "records": [record]}] + ([{}] if missing else [])
    outputs = output_facts(ctx, declarations, reports, confirmed=True)

    launch_id = f"launch-{hashlib.sha256(f'{ctx.task_id}:{marker}'.encode('utf-8')).hexdigest()[:12]}"
    if retained:
        retained_dir = Path(ctx.drive_root) / "services" / str(ctx.task_id)
        retained_dir.mkdir(parents=True, exist_ok=True)
        retained_path = retained_dir / f"demo-{launch_id}.log"
        retained_path.write_text(f"selected=***\ncanary={canary}\n", encoding="utf-8")
        log_finalization = {
            "deleted_live_log": False,
            "full_log_ref": {},
            "retained_live_log_path": str(retained_path),
            "tail": f"line=1 canary={canary}\n",
            "errors": [],
        }
        private_marker = retained_path.name
    else:
        log_ref = write_blob(
            Path(ctx.drive_root),
            f"selected=***\nREADY {launch_id}\n",
            kind="txt",
        )
        log_finalization = {
            "deleted_live_log": True,
            "full_log_ref": log_ref,
            "tail": f"line=1 canary={canary}\n",
            "errors": [],
        }
        private_marker = log_ref["sha256"]

    payload = {
        "launch_id": launch_id,
        "name": "demo",
        "returncode": 0,
        "outputs": declarations,
        "_output_facts": outputs,
        "log_finalization": log_finalization,
    }
    publish_stop(ctx, payload, confirmed=True)
    return payload["service_stop"], private_marker


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


def _is_raw_log_href(href: str) -> bool:
    """True only for the exact standard raw-row download route; never decodes signed/opaque params."""
    parsed = urlsplit(href)
    if parsed.scheme or parsed.netloc:
        return False
    if parsed.path not in {"/api/logs/events/raw", "/api/logs/tools/raw", "/api/logs/supervisor/raw"}:
        return False
    params = parse_qs(parsed.query, keep_blank_values=False)
    return params.get("download") == ["1"] and bool(params.get("ref"))


def _verify_surface_rows(
    page,
    base_url: str,
    container,
    stop_a: dict,
    stop_b: dict,
    *,
    allow_raw_links: bool = False,
    canaries: tuple[str, ...] = (),
) -> dict[str, str]:
    """Assert exactly two `.service-stop-result` entries bound individually to A and B."""
    from playwright.sync_api import expect

    rows = container.locator(".service-stop-result")
    expect(rows).to_have_count(2, timeout=15_000)

    url_a = stop_a["outputs"][0]["files"][0]["download_url"]
    url_b = stop_b["outputs"][0]["files"][0]["download_url"]
    assert url_a != url_b
    assert url_a.startswith("/") and not url_a.startswith("//")
    assert url_b.startswith("/") and not url_b.startswith("//")

    seen: dict[str, str] = {}
    for idx in range(2):
        row = rows.nth(idx)
        text = row.inner_text()
        links = row.locator(".service-stop-output a")
        assert links.count() == 1, (idx, links.count())
        link = links.first
        assert link.is_visible(), idx
        assert link.inner_text().strip() == "report.txt", idx
        href = link.get_attribute("href") or ""

        is_b = "not-created.txt: not saved" in text
        label = "B" if is_b else "A"
        expected_journal = "Journal: retained" if is_b else "Journal: saved"
        assert "Stopped" in text and expected_journal in text and "report.txt: saved" in text, (label, idx)
        if is_b:
            assert "Journal: saved" not in text, (label, idx)
        else:
            assert "Journal: retained" not in text and "not-created.txt" not in text, (label, idx)

        all_anchors = row.locator("a")
        anchor_count = all_anchors.count()
        assert anchor_count >= 1, (label, idx)
        if not allow_raw_links:
            assert anchor_count == 1, (label, idx, anchor_count)
        for a_idx in range(anchor_count):
            anchor = all_anchors.nth(a_idx)
            a_href = anchor.get_attribute("href") or ""
            a_text = anchor.inner_text() or ""
            for canary in canaries:
                assert canary and canary not in a_text and canary not in a_href, (label, idx, a_idx)
            assert "observability" not in a_text and "observability" not in a_href, (label, idx, a_idx)
            if a_href != href:
                assert allow_raw_links, (label, idx, a_idx)
                classes = (anchor.get_attribute("class") or "").split()
                assert "log-raw-download" in classes, (label, idx, a_idx)
                if a_href:
                    assert (_is_raw_log_href(a_href) or a_href.startswith("blob:" + base_url.rstrip("/") + "/")), (label, idx, a_idx)

        for canary in canaries:
            assert canary and canary not in text, (label, idx)
        assert "observability" not in text and "retained_live_log_path" not in text, (label, idx)

        expected_url = url_b if is_b else url_a
        expected_bytes = b"run-b" if is_b else b"run-a"
        assert href == expected_url, (label, idx)

        resp = page.request.get(urljoin(base_url, href))
        assert resp.status == 200, (label, resp.status)
        body = resp.body()
        assert body == expected_bytes, (label, len(body))
        assert label not in seen, f"duplicate row for {label}"
        seen[label] = hashlib.sha256(body).hexdigest()

    assert set(seen) == {"A", "B"}, sorted(seen)
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
    ctx = ToolContext(
        repo_dir=repo_dir,
        drive_root=data_dir,
        task_id=task_id,
        current_chat_id=1,
        _execution_trace={},
        event_queue=event_q,
    )
    canary_a = f"canary-a-{engine}"
    canary_b = f"canary-b-{engine}"

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

            # 2) Publish A (saved masked journal + report.txt) -> 0 before test MessageEvent, 1 after.
            stop_a, ref_marker_a = _publish_demo_stop(ctx, "run-a", canary=canary_a)
            assert stop_a["journal"] == {"state": "saved"}
            assert [o["state"] for o in stop_a["outputs"]] == ["saved"]
            expect(timeline.locator(".service-stop-result")).to_have_count(0)
            frame_a = _next_service_stop_frame(event_q, stop_a)
            _emit_ws_frame(page, {"type": "log", "chat_id": 1, "data": frame_a})
            _ensure_chat_stop_rows_visible(timeline, 1)

            # 3) Publish B (retained journal, report.txt saved, not-created.txt missing) -> 1 before, 2 after.
            stop_b, ref_marker_b = _publish_demo_stop(
                ctx, "run-b", canary=canary_b, retained=True, missing=True,
            )
            assert stop_b["launch_id"] != stop_a["launch_id"]
            assert stop_b["journal"] == {"state": "retained"}
            assert stop_b["outputs"][0]["state"] == "saved"
            assert stop_b["outputs"][1] == {"name": "not-created.txt", "state": "failed", "files": []}
            canaries = (canary_a, canary_b, ref_marker_a, ref_marker_b)
            assert ctx._execution_trace["verification_events"] == [
                {"kind": "service_stop", "service_stop": stop_a},
                {"kind": "service_stop", "service_stop": stop_b},
            ]
            expect(timeline.locator(".service-stop-result")).to_have_count(1)
            frame_b = _next_service_stop_frame(event_q, stop_b)
            _emit_ws_frame(page, {"type": "log", "chat_id": 1, "data": frame_b})
            _ensure_chat_stop_rows_visible(timeline, 2)
            live_hashes = _verify_surface_rows(
                page, url, timeline, stop_a, stop_b, canaries=canaries,
            )

            live_png = evidence_dir / f"service-stop-{engine}-live-details.png"
            page.screenshot(path=str(live_png), full_page=True)
            screenshots["live_details"] = live_png.name

            # 4) Reload WITHOUT WS injection -> reopened Details hydrates from server HTTP history.
            page.reload(wait_until="domcontentloaded", timeout=30_000)
            _wait_socket_open_quiescent(page)
            reloaded_card = page.locator(f'.chat-live-card[data-task-id="{task_id}"]')
            reloaded_timeline = _open_live_card(reloaded_card)
            _ensure_chat_stop_rows_visible(reloaded_timeline, 2)
            reopened_hashes = _verify_surface_rows(
                page, url, reloaded_timeline, stop_a, stop_b, canaries=canaries,
            )

            reopened_png = evidence_dir / f"service-stop-{engine}-reopened-details.png"
            page.screenshot(path=str(reopened_png), full_page=True)
            screenshots["reopened_details"] = reopened_png.name

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
            logs_hashes = _verify_surface_rows(
                page, url, log_card, stop_a, stop_b, allow_raw_links=True, canaries=canaries,
            )

            log_card.scroll_into_view_if_needed()
            logs_png = evidence_dir / f"service-stop-{engine}-logs.png"
            page.screenshot(path=str(logs_png), full_page=True)
            screenshots["logs"] = logs_png.name

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
