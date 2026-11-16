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


# Who asked for each census read, recorded as it starts. A gated read keeps its reader's stack
# (state_snapshots.js readStateSnapshot): a room's own read names chat.js refreshHeaderControlState, the page's
# poll app.js refreshProjectsNav.
_RECORD_STATE_READERS = """(() => {
    const fetch = window.fetch;
    window.__stateReaders = [];
    window.fetch = (url, init) => {
        if (String(url).split('?')[0].endsWith('/api/state')) window.__stateReaders.push(String(new Error().stack));
        return fetch(url, init);
    };
})();"""


def _reader(stack):
    """'room', 'page poll', or the recorded stack itself when it names neither (or both)."""
    stack = stack or "no reader recorded"
    room, page_poll = "refreshHeaderControlState" in stack, "refreshProjectsNav" in stack
    return "room" if room and not page_poll else "page poll" if page_poll and not room else stack


_STATUS = "() => document.querySelector('#chat-status')?.textContent.trim()"


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
    state = {"rows": [], "reads": 0, "fixture_generation": 0, "hold": False}
    held = []

    def _census(route):
        if state["hold"]:
            evidence.checkpoint("read_held", url=route.request.url)
            held.append(route)
            return
        _observe_census(route, state, evidence)

    def _held_read():
        """The next census read, unanswered, and who asked for it (the page reads one census at a time)."""
        for _ in range(300):
            if held:
                reader = _reader(page.evaluate("() => window.__stateReaders.at(-1)"))
                evidence.checkpoint("reader", reader=reader)
                return held.pop(0), reader
            page.wait_for_timeout(50)
        raise AssertionError("no census read started within 15 s")

    def _turn(n):
        return [{"activity_id": f"act-smoke-{n}", "chat_id": 1, "project_id": "",
                 "client_message_id": f"msg-smoke-{n}", "kind": "direct_chat",
                 "phase": "thinking", "started_at": float(n)}]

    def _receipt(n):
        evidence.checkpoint("typing_emit_start", turn=n)
        page.evaluate("""n => window.__ouroWs.emit('typing', {
            type: 'typing', chat_id: 1, activity_id: `act-smoke-${n}`, client_message_id: `msg-smoke-${n}`,
            phase: 'thinking', kind: 'direct_chat'})""", n)
        evidence.checkpoint("typing_emit_returned", turn=n)

    def _rows(rows):
        state["rows"] = rows
        state["fixture_generation"] += 1
        evidence.checkpoint("fixture_rows_changed", fixture_generation=state["fixture_generation"], rows=rows)

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            evidence = FailureEvidence(page, browser, output_dir(request.config),
                                       request.node.nodeid, "chromium")
            evidence.details.update(production_generation="unavailable",
                                    request_initiator="each read's stack, window.__stateReaders")
            with evidence:
                page.add_init_script(_RECORD_STATE_READERS)
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

                # Main on screen reads the census through the page's 3 s poll (app.js), one read at a time
                # (state_snapshots.js): a forced read asked while one is in flight starts the moment it settles,
                # a poll tick is never queued behind it. Held unanswered, each read shows who asked for it.
                page.wait_for_function("() => window.__ouroWs?.isConnected?.() === true", timeout=30_000)
                state["hold"] = True
                read, reader = _held_read()
                for _ in range(5):  # what the boot or the socket's open still had in flight is answered as asked
                    if reader == "page poll":
                        break
                    _observe_census(read, state, evidence)
                    read, reader = _held_read()
                assert reader == "page poll", reader
                _observe_census(read, state, evidence)
                read, reader = _held_read()
                assert reader == "page poll", f"with no receipt the room read the census itself: {reader}"

                # The typing receipt (it carries the submission's client_message_id) arrives while the page's read
                # is in flight; that read, asked before the turn began, answers the census without it. The receipt
                # pulls the census at once: the next read is the room's own, not the page's next poll.
                _receipt(1)
                _observe_census(read, state, evidence)
                _rows(_turn(1))
                read, reader = _held_read()
                assert reader == "room", f"the receipt pulled no census read of its own: {reader}"
                assert page.evaluate(_STATUS) in {"Online", "Ready"}, "the receipt alone moved the header"
                state["hold"] = False
                _observe_census(read, state, evidence)
                # In-flight state, from the receipt's own read: the header names the census phase in the
                # task-phase words (#1740: 'Thinking', no dots).
                page.wait_for_function(
                    """() => {
                        const el = document.querySelector('#chat-status');
                        return el && el.textContent.trim() === 'Thinking';
                    }""",
                    timeout=5_000,
                )
                evidence.checkpoint("thinking_visible")

                # Dots indicator visible (main chat instance id is #typing-indicator)
                typing_el = page.locator("#typing-indicator")
                typing_el.wait_for(state="visible", timeout=5_000)

                # A typing frame alone never lights the header: with the census
                # empty again, another receipt settles the page at Online.
                _rows([])
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

                # Main behind another page is away and quiet (PR #2951, room_retire.js): a receipt draws nothing
                # there, and Main catches up when it shows again.
                page.click('[data-nav-page="settings"]')
                evidence.checkpoint("left_chat")
                page.wait_for_timeout(300)
                reads_before = state["reads"]
                evidence.checkpoint("reads_before", reads_before=reads_before)
                _rows(_turn(3))
                _receipt(3)
                page.wait_for_timeout(1_000)
                assert page.evaluate(_STATUS) in {"Online", "Ready"}, "an away Main drew the receipt's census"
                page.click('[data-nav-page="chat"]')
                # Back, Main catches up as after a reconnect: its census read names the census phase.
                page.wait_for_function(
                    """() => {
                        const el = document.querySelector('#chat-status');
                        return el && el.textContent.trim() === 'Thinking';
                    }""",
                    timeout=5_000,
                )
                evidence.checkpoint("thinking_visible_after_return")
                evidence.checkpoint("assert_census_read", reads=state["reads"], reads_before=reads_before)
                assert state["reads"] > reads_before, "the return did not read the census"
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise
