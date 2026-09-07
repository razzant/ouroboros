"""Browser behavior of the Logs panel and its task groups."""
import pytest

from tests.test_ui_smoke_playwright import (
    direct_server as direct_server,
    direct_server_with_data as direct_server_with_data,
)


@pytest.mark.ui_browser
def test_ui_logs_error_group_survives_later_heartbeat(direct_server):
    """The Errors filter keeps a task whose visible timeline contains a failure."""
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page()
            page.route("**/api/logs/*", lambda route: route.fulfill(json={"entries": []}))
            page.goto(direct_server, wait_until="domcontentloaded")
            result = page.evaluate("""async () => {
                const { initLogs } = await import('/static/modules/logs.js');
                const { LOG_CATEGORIES } = await import('/static/modules/log_events.js');
                const mount = document.createElement('div');
                document.body.appendChild(mount);
                const handlers = new Map();
                initLogs({
                    mount, ws: { on: (name, callback) => handlers.set(name, callback) },
                    state: { activeFilters: Object.fromEntries(Object.keys(LOG_CATEGORIES)
                        .map((name) => [name, name === 'errors'])) },
                });
                const emit = (data) => handlers.get('log')({ data });
                const failure = { type: 'tool_call_finished', task_id: 'error-task',
                    tool: 'run_command', is_error: true, ts: '2026-09-06T00:00:00Z' };
                const heartbeat = { type: 'task_heartbeat', task_id: 'error-task',
                    ts: '2026-09-06T00:00:01Z' };
                emit(failure);
                emit(heartbeat);
                emit(heartbeat); // Replayed overlap must not create a second row.
                emit({ ...heartbeat, task_id: 'quiet-task' });
                const failed = mount.querySelector('[data-task-group="error-task"]');
                const quiet = mount.querySelector('[data-task-group="quiet-task"]');
                return {
                    category: failed.dataset.category, visible: !failed.hidden,
                    timeline: failed.querySelectorAll('.log-task-event').length,
                    failureVisible: failed.querySelector('.log-phase.error') !== null,
                    quietHidden: quiet.hidden, cards: mount.querySelectorAll('.log-task-card').length,
                };
            }""")
            assert result == {
                "category": "errors", "visible": True, "timeline": 2,
                "failureVisible": True, "quietHidden": True, "cards": 2,
            }
        finally:
            browser.close()


