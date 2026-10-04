"""Owner D10 in the browser: Pause and Resume an answered root's late work from its card.

The real SPA (served static) talks to the REAL history, Pause and Resume
endpoints and the real activity census; the late phase is the real post-task
coordinator and consolidation with the deterministic Light transport of
``test_late_phase_pause_resume``. Screenshots: ``OUROBOROS_UI_EVIDENCE_DIR``,
else ``<tmp>/screenshots`` (retained by ``scripts/safe_test.py``).
"""

from __future__ import annotations

import json
import os
import pathlib
import threading

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from tests.test_late_phase_pause_resume import ROOT, _join, _phase, _start, fit  # noqa: F401 — fixture
from tests.test_late_phase_pause_resume import late as late  # noqa: F401 — fixture
from tests.test_subscription_setup_browser import subscription_ui as subscription_ui  # noqa: F401 — fixture

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def _capture(page, root: pathlib.Path, name: str) -> None:
    target = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or root / "screenshots")
    target.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(target / f"{name}.png"), animations="disabled")


def _history(root: pathlib.Path) -> None:
    """The delivered answer and the host-attested control marker of this root's card;
    the dialogue the late phase consolidates lives in another room."""
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "progress.jsonl").write_text(json.dumps({
        "ts": "2026-08-17T10:00:00+00:00", "type": "send_message", "task_id": ROOT, "is_progress": True,
        "direction": "out", "chat_id": 1, "user_id": 1, "text": "step one", "content": "step one",
        "cancelable": True}) + "\n", encoding="utf-8")
    rows = [{"ts": f"2026-08-17T08:{index // 60:02d}:{index % 60:02d}+00:00", "direction": "in", "chat_id": 0,
             "text": f"hidden room line {index}"} for index in range(100)]
    rows += [{"ts": "2026-08-17T10:00:05+00:00", "direction": "out", "chat_id": 1, "user_id": 1,
              "text": "Already delivered answer", "task_id": ROOT, "format": "markdown"}]
    (logs / "chat.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_browser_pauses_and_resumes_late_work_with_the_delivered_answer_intact(subscription_ui, late):
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.gateway.state import _chat_activities_snapshot_safe
    from ouroboros.gateway.task_pause import api_task_pause
    from ouroboros.gateway.tasks import api_task_resume
    from ouroboros.task_results import load_task_result

    f, ui = late, subscription_ui
    page = ui["page"]
    _history(f.root)
    app = Starlette(routes=[Route("/api/chat/history", make_chat_history_endpoint(f.root)),
                           Route("/api/tasks/{task_id}/pause", api_task_pause, methods=["POST"]),
                           Route("/api/tasks/{task_id}/resume", api_task_resume, methods=["POST"])])
    app.state.drive_root = f.root
    entered, release, posts = threading.Event(), threading.Event(), []
    f.light.hooks["scratchpad"] = [lambda: (entered.set(), release.wait(30))]
    spawned: list = []
    threads = _start(f)
    try:
        assert entered.wait(10)
        with TestClient(app) as client:
            def forward(route):
                request = route.request
                path = request.url.split(ui["url"], 1)[-1]
                if request.method == "POST":
                    before = set(threading.enumerate())
                    response = client.post(path, json=request.post_data_json)
                    spawned.extend(thread for thread in threading.enumerate() if thread not in before)
                    posts.append((path, response.status_code, response.json()))
                else:
                    response = client.get(path)
                route.fulfill(status=response.status_code, content_type="application/json", body=response.content)

            page.route("**/api/chat/history*", forward)
            page.route(f"**/api/tasks/{ROOT}/pause", forward)
            page.route(f"**/api/tasks/{ROOT}/resume", forward)
            page.route("**/api/state", lambda route: route.fulfill(content_type="application/json", body=json.dumps({
                "supervisor_ready": True, "projects": [], "active_chat_activities_complete": True,
                "active_chat_activities": _chat_activities_snapshot_safe(f.root, {}, direct_turns=[])})))
            page.route(f"**/api/tasks/{ROOT}", lambda route: route.fulfill(
                content_type="application/json", body=json.dumps(load_task_result(f.root, ROOT))))
            page.goto(ui["url"] + "/")
            card = page.locator(f'.chat-live-card[data-task-id="{ROOT}"]')
            page.get_by_text("Already delivered answer").first.wait_for(timeout=10000)
            card.wait_for(timeout=10000)
            chip = card.locator(".chat-live-phase").first
            page.wait_for_function("(el) => /Finalizing/.test(el.closest('.chat-live-card').innerText)",
                                   arg=chip.element_handle(), timeout=10000)
            _capture(page, f.root, "late-1-finalizing-answer-delivered")

            card.locator("[data-cancel-run]").click()
            page.locator('[data-task-control="pause"]').click()
            page.wait_for_function("(el) => /Pausing/.test(el.closest('.chat-live-card').innerText)",
                                   arg=chip.element_handle(), timeout=15000)
            assert posts and posts[-1][1] in (200, 202) and posts[-1][2]["ok"], posts
            _capture(page, f.root, "late-2-pausing-sent-draft-finishing")

            release.set()
            _join(threads)
            assert _phase(f) == "paused" and f.light.kinds() == ["scratchpad"]
            page.wait_for_function("(el) => /Paused/.test(el.closest('.chat-live-card').innerText)",
                                   arg=chip.element_handle(), timeout=15000)
            assert page.get_by_text("Already delivered answer").count() == 1
            _capture(page, f.root, "late-3-paused-answer-intact")

            card.locator("[data-cancel-run]").click()
            page.locator('[data-task-control="resume"]').click()
            page.wait_for_function("() => !document.querySelector('.task-control-menu')")
            _join(spawned)
            assert posts[-1][0].endswith("/resume") and posts[-1][2]["ok"], posts
            assert _phase(f) == "completed" and f.light.kinds() == ["scratchpad"] * 3
            page.get_by_text("Resuming: the work left after the delivered answer continues.").wait_for(timeout=10000)
            page.wait_for_function(
                "(el) => !/Paused|Pausing|Finalizing/.test(el.closest('.chat-live-card').innerText)",
                arg=chip.element_handle(), timeout=20000)
            assert page.get_by_text("Already delivered answer").count() == 1
            _capture(page, f.root, "late-4-resumed-finished-answer-intact")
    finally:
        release.set()
        _join(threads)
        _join(spawned)
