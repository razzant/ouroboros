"""The owner's Pause of an answered root's upkeep in the browser: it ends at once, nothing to resume.

The real SPA (served static) talks to the REAL history and Pause endpoints and the
real activity census; the upkeep is the real post-task coordinator and its fallback-draft
stage (a two-send Light operation since the scratchpad pass is retired, #918) with the
deterministic Light transport of ``test_late_phase_pause_cut``.
Screenshots: ``OUROBOROS_UI_EVIDENCE_DIR``, else ``<tmp>/screenshots`` (retained by
``scripts/safe_test.py``).
"""

from __future__ import annotations

import json
import os
import pathlib
import threading

import pytest
from starlette.applications import Starlette

from ouroboros import dialogue_store
from starlette.routing import Route
from starlette.testclient import TestClient

from tests.test_late_phase_pause_cut import ROOT, _join, _phase, _start, fit  # noqa: F401 — fixture
from tests.test_late_phase_pause_cut import late as late  # noqa: F401 — fixture
from tests.test_subscription_setup_browser import subscription_ui as subscription_ui  # noqa: F401 — fixture
from tests._work_projection_shared import state_fields

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def _capture(page, root: pathlib.Path, name: str) -> None:
    target = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or root / "screenshots")
    target.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(target / f"{name}.png"), animations="disabled")


def _history(root: pathlib.Path) -> None:
    """The delivered answer and the host-attested control marker of this root's card;
    the dialogue the upkeep consolidates lives in another room."""
    dialogue_store.append(root, "progress", {
        "ts": "2026-08-17T10:00:00+00:00", "type": "send_message", "task_id": ROOT, "is_progress": True,
        "direction": "out", "chat_id": 1, "user_id": 1, "text": "step one", "content": "step one",
        "cancelable": True})
    rows = [{"ts": f"2026-08-17T08:{index // 60:02d}:{index % 60:02d}+00:00", "direction": "in", "chat_id": 0,
             "text": f"hidden room line {index}"} for index in range(100)]
    rows += [{"ts": "2026-08-17T10:00:05+00:00", "direction": "out", "chat_id": 1, "user_id": 1,
              "text": "Already delivered answer", "task_id": ROOT, "format": "markdown"}]
    for row in rows:
        dialogue_store.append(root, "chat", row)


def test_browser_pause_ends_the_upkeep_at_once_with_the_delivered_answer_intact(subscription_ui, late):
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.gateway.task_pause import api_task_pause
    from ouroboros.owner_pause import read_fence
    from ouroboros.task_results import load_task_result

    f, ui = late, subscription_ui
    page = ui["page"]
    _history(f.root)
    app = Starlette(routes=[Route("/api/chat/history", make_chat_history_endpoint(f.root)),
                           Route("/api/tasks/{task_id}/pause", api_task_pause, methods=["POST"])])
    app.state.drive_root = f.root
    entered, release, posts = threading.Event(), threading.Event(), []
    f.light.hooks["draft"] = [lambda: (entered.set(), release.wait(30))]
    threads = _start(f)
    try:
        assert entered.wait(10)
        with TestClient(app) as client:
            def forward(route):
                request = route.request
                path = request.url.split(ui["url"], 1)[-1]
                if request.method == "POST":
                    response = client.post(path, json=request.post_data_json)
                    posts.append((path, response.status_code, response.json()))
                else:
                    response = client.get(path)
                route.fulfill(status=response.status_code, content_type="application/json", body=response.content)

            page.route("**/api/chat/history*", forward)
            page.route(f"**/api/tasks/{ROOT}/pause", forward)
            # The real census and the server's own work projection of it, which the page reads.
            page.route("**/api/state", lambda route: route.fulfill(content_type="application/json", body=json.dumps({
                "supervisor_ready": True, "projects": [], "active_chat_activities_complete": True,
                **state_fields(f.root, {}, direct_turns=[])})))
            page.route(f"**/api/tasks/{ROOT}", lambda route: route.fulfill(
                content_type="application/json", body=json.dumps(load_task_result(f.root, ROOT))))
            page.goto(ui["url"] + "/")
            card = page.locator(f'.chat-live-card[data-task-id="{ROOT}"]')
            page.get_by_text("Already delivered answer").first.wait_for(timeout=10000)
            card.wait_for(timeout=10000)
            chip = card.locator(".chat-live-phase").first
            page.wait_for_function("(el) => /Finalizing/.test(el.closest('.chat-live-card').innerText)",
                                   arg=chip.element_handle(), timeout=10000)
            _capture(page, f.root, "upkeep-1-finalizing-answer-delivered")

            card.locator("[data-cancel-run]").click()
            page.locator('[data-task-control="pause"]').click()
            page.wait_for_function("(el) => /Pausing/.test(el.closest('.chat-live-card').innerText)",
                                   arg=chip.element_handle(), timeout=15000)
            assert posts and posts[-1][1] in (200, 202) and posts[-1][2]["ok"], posts
            _capture(page, f.root, "upkeep-2-pausing-sent-read-finishing")

            release.set()
            _join(threads)
            checkpoint = load_task_result(f.root, ROOT)["root_phase_checkpoint"]
            assert _phase(f) == "degraded" and f.light.kinds() == ["draft"]
            assert checkpoint["post_task_stop_reason"] == "owner_pause:skipped=memory_fallback_draft,promotion"
            assert read_fence(f.root, ROOT)["state"] == "released"
            # Nothing is saved to resume: the card settles on its answer, never Paused, no Resume.
            page.wait_for_function(
                "(el) => !/Paused|Pausing|Finalizing/.test(el.closest('.chat-live-card').innerText)",
                arg=chip.element_handle(), timeout=20000)
            assert card.locator("[data-resume-run]").count() == 0
            assert page.get_by_text("Already delivered answer").count() == 1
            assert [path for path, _status, _body in posts] == [f"/api/tasks/{ROOT}/pause"]
            _capture(page, f.root, "upkeep-3-ended-answer-intact")
    finally:
        release.set()
        _join(threads)
