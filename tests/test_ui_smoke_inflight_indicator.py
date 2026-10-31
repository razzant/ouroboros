"""Browser smoke test for the in-flight direct/ephemeral turn indicator.

Lives in its own module (not test_ui_smoke_playwright.py) so the giant smoke
module stays under the size-ratchet byte gate. Reuses its server fixture.
"""

from __future__ import annotations

import copy
import json
from urllib.parse import urlsplit

import pytest

from tests._work_projection_shared import with_work_fields
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401 - pytest fixture import
from tests.ui_failure_evidence import FailureEvidence, exception_fact


def _observe_census(route, state, evidence):
    """Journal the fixture's two row samples without changing when it replies."""
    state["reads"] += 1
    request_id = state["reads"]
    requests = evidence.details.setdefault("requests", {})

    def record(phase, **facts):
        if evidence.output_dir is not None:
            requests[request_id] = phase
            evidence.details["pending_request_ids"] = [
                key for key, value in requests.items() if value not in {"fulfilled", "aborted"}]
            evidence.checkpoint(phase, request_id=request_id, reads=state["reads"],
                                fixture_generation=state["fixture_generation"], **facts)

    if evidence.output_dir is not None:
        url = route.request.url
        parsed = urlsplit(url)
        record("intercepted", url=url, path=parsed.path, query=parsed.query, rows=state["rows"])
    phase = "fetch"
    try:
        response = route.fetch()
        payload = response.json()
        record("fetch_returned", status=response.status)
        # This sample stays AFTER fetch/json, as in the original fixture.
        payload["active_chat_activities"] = list(state["rows"])
        original_scope = {key: copy.deepcopy(payload.get(key)) for key in (
            "active_chat_activities_complete", "supervisor_ready", "chat_id", "project_id")}
        payload["active_chat_activities_complete"] = True
        payload = with_work_fields(payload)  # and the server's work projection of those rows, which the page reads
        record("response_built", rows=payload["active_chat_activities"],
               original_scope=original_scope, active_chat_activities_complete=True)
        phase = "fulfill"
        record("fulfill_started")
        route.fulfill(content_type="application/json", body=json.dumps(payload))
        record("fulfilled")
    except BaseException as exc:
        record("aborted", operation=phase, exception=exception_fact(exc))
        raise


@pytest.mark.ui_browser
def test_ui_smoke_chat_inflight_indicator_lifecycle(direct_server_with_data, request):  # noqa: F811
    """The header follows the /api/state census; a typing frame is only a receipt."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright
    from tests.ci_evidence import output_dir

    # NB: the server runs as a subprocess; its DirectActivityRegistry is not
    # reachable from this process. The census the page reads is shaped here by
    # intercepting /api/state (the server's own payload plus the activity rows),
    # and the typing receipt is driven through the window.__ouroWs debug hook,
    # which is exactly the surface the browser exercises for real WS frames.
    url = direct_server_with_data["url"]
    state = {"rows": [], "reads": 0, "fixture_generation": 0}

    def _census(route):
        _observe_census(route, state, evidence)

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            evidence = FailureEvidence(page, browser, output_dir(request.config),
                                       request.node.nodeid, "chromium")
            evidence.details.update(production_generation="unavailable",
                                    request_initiator="unavailable; typing and project poll are possible readers")
            with evidence:
                page.route("**/api/state*", _census)
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                status_badge = page.locator("#chat-status")
                status_badge.wait_for(state="attached", timeout=30_000)
                # Idle state
                page.wait_for_function(
                    """() => {
                        const el = document.querySelector('#chat-status');
                        return el && (el.textContent.trim() === 'Online' || el.textContent.trim() === 'Ready');
                    }""",
                    timeout=10_000,
                )

                # Off the chat page the 3 s header poll is paused, so the census
                # read that follows the receipt is the receipt's own (the 20 s
                # projects poll is the only other reader).
                page.click('[data-nav-page="settings"]')
                evidence.checkpoint("left_chat")
                page.wait_for_timeout(300)
                reads_before = state["reads"]
                evidence.checkpoint("reads_before", reads_before=reads_before)

                # The census now lists the direct turn; the typing receipt (it
                # carries the submission's client_message_id) pulls that census
                # at once instead of waiting for the next poll.
                state["rows"] = [{
                    "activity_id": "act-smoke-1", "chat_id": 1, "project_id": "",
                    "client_message_id": "msg-smoke-1", "kind": "direct_chat",
                    "phase": "thinking", "started_at": 1.0,
                }]
                state["fixture_generation"] += 1
                evidence.checkpoint("fixture_rows_changed", fixture_generation=state["fixture_generation"],
                                    rows=state["rows"])
                evidence.checkpoint("typing_emit_start")
                page.evaluate("""() => {
                    if (window.__ouroWs) {
                        window.__ouroWs.emit('typing', {
                            type: 'typing',
                            chat_id: 1,
                            activity_id: 'act-smoke-1',
                            client_message_id: 'msg-smoke-1',
                            phase: 'thinking',
                            kind: 'direct_chat'
                        });
                    }
                }""")
                evidence.checkpoint("typing_emit_returned")

                # In-flight state: reached through the receipt's own census read. The header
                # names the census phase in the task-phase words (#1740: 'Thinking', no dots).
                page.wait_for_function(
                    """() => {
                        const el = document.querySelector('#chat-status');
                        return el && el.textContent.trim() === 'Thinking';
                    }""",
                    timeout=5_000,
                )
                evidence.checkpoint("thinking_visible")
                evidence.checkpoint("assert_census_read", reads=state["reads"], reads_before=reads_before)
                assert state["reads"] > reads_before, "the receipt did not pull the census"
                page.click('[data-nav-page="chat"]')

                # Dots indicator visible (main chat instance id is #typing-indicator)
                typing_el = page.locator("#typing-indicator")
                typing_el.wait_for(state="visible", timeout=5_000)

                # A typing frame alone never lights the header: with the census
                # empty again, another receipt settles the page at Online.
                state["rows"] = []
                state["fixture_generation"] += 1
                evidence.checkpoint("fixture_rows_changed", fixture_generation=state["fixture_generation"],
                                    rows=state["rows"])
                evidence.checkpoint("completion_and_typing_emit_start")
                page.evaluate("""() => {
                    if (window.__ouroWs) {
                        window.__ouroWs.emit('chat', {
                            type: 'chat',
                            role: 'assistant',
                            content: 'Done!',
                            task_id: 'act-smoke-1',
                            chat_id: 1
                        });
                        window.__ouroWs.emit('typing', {
                            type: 'typing',
                            chat_id: 1,
                            activity_id: 'act-smoke-2',
                            client_message_id: 'msg-smoke-2',
                            phase: 'thinking',
                            kind: 'direct_chat'
                        });
                    }
                }""")
                evidence.checkpoint("completion_and_typing_emit_returned")

                page.wait_for_function(
                    """() => {
                        const el = document.querySelector('#chat-status');
                        return el && el.textContent.trim() === 'Online';
                    }""",
                    timeout=5_000,
                )
                typing_el.wait_for(state="hidden", timeout=5_000)
                evidence.checkpoint("idle_and_dots_hidden")
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise
