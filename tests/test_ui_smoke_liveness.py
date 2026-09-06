"""Behavioral liveness scenarios for the chat header pill (stream T).

The class under test: "Working..." must be a claim the system can justify.
L1 drives the owner's exact symptom end to end WITHOUT route stubs — durable
disk state -> real gateway -> real browser — and asserts the pill CONVERGES to
Online (every prior regression test only asserted that Working... appears).
L2 is the capability-preservation gate: a genuinely live managed task keeps
its truthful Working... with no flicker and no durable-detail polling.
"""
from __future__ import annotations

import json
import os
import pathlib

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401 - pytest fixture import
from tests.test_ui_smoke_project_continuity import (
    _goto_main_ready,
    _install_task_detail_gate,
    _launch,
    _wait_status,
)

import tempfile

_SCREENSHOT_DIR = pathlib.Path(
    os.environ.get("OUROBOROS_UI_SMOKE_SHOT_DIR", "")
    or (pathlib.Path(tempfile.gettempdir()) / "ouroboros-ui-shots")
)


def _seed_orphan_swarm(data_dir: pathlib.Path) -> None:
    """Durable state reproducing the stuck pill: a subagent FINAL chat row whose
    swarm parent has no rows in the window, a STALE progress floor (older than
    the child, so the server honestly keeps the lineage — the client reconcile
    is the load-bearing fix), and the parent's completed task result on disk."""
    logs = data_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    child = {
        "ts": "2026-08-14T23:00:00+00:00", "direction": "out", "chat_id": 1,
        "text": "child final answer", "task_id": "swarm-child",
        "delegation_role": "subagent", "parent_task_id": "swarm-root",
        "root_task_id": "swarm-root", "subagent_task_id": "swarm-child",
    }
    owner = {
        "ts": "2026-08-14T22:00:00+00:00", "direction": "in", "chat_id": 1,
        "user_id": 1, "text": "start the swarm", "task_id": "",
    }
    (logs / "chat.jsonl").write_text(
        json.dumps(owner) + "\n" + json.dumps(child) + "\n", encoding="utf-8"
    )
    # Floor carriers only: no task_id, so replay mints no card for them —
    # a task_id here would mint an unfinished card with no durable result,
    # which by owner decision (Q3=A) honestly keeps the pill lit.
    stale_progress = [
        json.dumps({
            "ts": f"2026-08-13T10:00:{i:02d}+00:00", "chat_id": 1,
            "content": f"old telemetry {i}",
        })
        for i in range(3)
    ]
    (logs / "progress.jsonl").write_text("\n".join(stale_progress) + "\n", encoding="utf-8")
    results = data_dir / "task_results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "swarm-root.json").write_text(
        json.dumps({
            "_schema_version": 1,
            "task_id": "swarm-root",
            "status": "completed",
            "summary": "swarm finished",
            "root_phase_checkpoint": {"post_task_synthesis": "completed"},
        }),
        encoding="utf-8",
    )


@pytest.mark.ui_browser
def test_ui_smoke_orphan_swarm_card_converges_to_online(direct_server_with_data):  # noqa: F811
    """L1: reload with an orphaned swarm parent in durable history -> the pill
    ends Online and the minted card is finished by durable truth. Stub-free:
    real /api/chat/history, real /api/state, real /api/tasks/<id>."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    _seed_orphan_swarm(direct_server_with_data["data_dir"])
    card = '.chat-live-card[data-task-id="swarm-root"]'
    with sync_playwright() as pw:
        try:
            browser, page = _launch(pw)
        except PlaywrightError as exc:
            pytest.skip(f"Playwright browser unavailable: {exc}")
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            # The lineage-adopted parent card IS minted from replay (server
            # honestly kept the stale-floor lineage) ...
            page.wait_for_selector(card, timeout=30_000)
            # ... and the card-set reconcile closes it from the durable
            # task result, returning the pill to Online. Before the fix
            # this stayed "Working..." across every reload and restart.
            page.wait_for_function(
                """(sel) => document.querySelector(sel)?.dataset.finished === '1'""",
                arg=card,
                timeout=30_000,
            )
            _wait_status(page, "Online", timeout=15_000)
            _SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(_SCREENSHOT_DIR / "liveness_L1_online.png"))
            # The child's answer text survives as an ordinary bubble.
            assert page.get_by_text("child final answer").count() >= 1
        finally:
            browser.close()


@pytest.mark.ui_browser
def test_ui_smoke_live_managed_task_keeps_working_without_detail_polls(direct_server_with_data):  # noqa: F811
    """L2 (capability gate): a task the snapshot vouches for keeps an unbroken
    truthful Working... — no flicker, and the reconcile never polls durable
    detail for a snapshot-confirmed id."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    with sync_playwright() as pw:
        try:
            browser, page = _launch(pw)
        except PlaywrightError as exc:
            pytest.skip(f"Playwright browser unavailable: {exc}")
        try:
            _install_task_detail_gate(page)

            def _inject(route):
                response = route.fetch()
                payload = response.json()
                payload["active_chat_activities"] = [{
                    "activity_id": "live-root", "chat_id": 1, "project_id": "",
                    "client_message_id": "", "kind": "managed_task",
                    "phase": "working", "started_at": 1.0,
                }]
                route.fulfill(content_type="application/json", body=json.dumps(payload))

            page.route("**/api/state*", _inject)
            _goto_main_ready(page, url, "Working...")
            page.evaluate(
                """() => {
                    window.__statusFlips = 0;
                    const el = document.querySelector('#chat-status');
                    window.__statusObserver = new MutationObserver(() => {
                        if (el.textContent.trim() !== 'Working...') window.__statusFlips += 1;
                    });
                    window.__statusObserver.observe(el, {childList: true, characterData: true, subtree: true});
                }"""
            )
            # Two full header-refresh cycles (3s interval) + margin.
            page.wait_for_timeout(7_000)
            _wait_status(page, "Working...", timeout=1_000)
            assert page.evaluate("() => window.__statusFlips") == 0
            assert page.evaluate("() => window.__taskDetailCalls.length") == 0
        finally:
            browser.close()
