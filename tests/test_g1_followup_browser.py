"""Actual Activity consumer and Restore request, served from the candidate bytes."""
from __future__ import annotations

from pathlib import Path
import os
import time

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401 - pytest fixture
from tests.ui_chat_viewport_smoke import wait_drawer_closed

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def test_activity_relationship_hold_and_stale_restore(direct_server_with_data, monkeypatch):  # noqa: F811 - pytest fixture
    from playwright.sync_api import sync_playwright
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.followup import _handle_schedule_followup
    from ouroboros.task_results import write_task_result
    from ouroboros.cancel_intents import request_cancel
    from supervisor.queue_schedules import load_schedule_store
    from tests.test_g1_followup_policy import BINDING, DEADLINE, ORIGIN
    from tests._budget_pause_exact_helpers import _install_queue
    import json

    server = direct_server_with_data
    root = server['data_dir']
    server['stop_server']()
    queue, _, workers = _install_queue(root, monkeypatch)
    write_task_result(root, ORIGIN, 'completed', root_task_id=ORIGIN,
                      billing_group=BINDING, deadline_at=DEADLINE)
    ctx = ToolContext(repo_dir=server['repo_dir'], drive_root=root, task_id=ORIGIN,
                      current_chat_id=1,
                      task_metadata={'root_task_id': ORIGIN, 'delegation_role': 'root',
                                     'resource_intent': {'kind': 'system_repo'}},
                      task_contract={'objective': 'x', 'delegation_role': 'root'})
    result = _handle_schedule_followup(ctx, relation='related', run_at='2000-01-01T00:00:00+00:00',
                                      objective='Continue original work after dependency clears')
    assert result.startswith('FOLLOWUP_SCHEDULED'), result
    queue.check_scheduled_tasks()
    [fired] = workers.PENDING
    fired_id = fired['id']
    request_cancel(root, ORIGIN, reason='Stop after follow-up admission', source='http_single',
                   requested_by='owner', allow_settled_target=True)
    queue.check_scheduled_tasks()
    assert queue.persist_queue_snapshot(reason='g1_browser_held_pending')
    server['start_server']()
    [record] = load_schedule_store(root)['tasks']
    old_id = record['followup_hold']['hold_id']
    evidence = Path(os.environ.get('OUROBOROS_UI_EVIDENCE_DIR') or root / 'ui-evidence') / 'g1-repair'
    evidence.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page(viewport={'width': 1440, 'height': 960})
            page.goto(server['url'], wait_until='domcontentloaded')
            page.click('[data-nav-page="dashboard"]')
            page.click('[data-dashboard-tab="activity"]')
            section = page.locator('[data-activity-section="schedules"]')
            response = page.request.get(server['url'] + '/api/schedules').json()
            (evidence / 'before-restore.json').write_text(json.dumps(response, indent=2))
            print('G1_BROWSER_SCHEDULES', json.dumps(response))
            history = section.locator('[data-activity-history] > summary')
            history.wait_for(state='visible')
            history.click()
            button = section.get_by_role('button', name='Restore hold')
            button.wait_for(state='visible')
            assert button.get_attribute('data-hold-id') == old_id
            visible = section.inner_text()
            assert 'related' in visible and 'cap $10' in visible and 'deadline' in visible
            assert 'Original work stopped or restarted' in visible and 'root-1' in visible and '2099' in visible
            for width in (1440, 390):
                page.set_viewport_size({'width': width, 'height': 960})
                if width < 700:
                    wait_drawer_closed(page)
                section.scroll_into_view_if_needed()
                assert section.locator('.activity-sub').evaluate_all(
                    '(els) => els.length > 0 && els.every((el) => el.scrollWidth <= el.clientWidth + 1)')
                page.screenshot(path=str(evidence / f'g1-activity-{width}.png'), full_page=True)
            page.set_viewport_size({'width': 1440, 'height': 960})
            # A new accepted Stop while the old DOM still offers its old identity.
            request_cancel(root, ORIGIN, reason='New Stop', source='http_single', requested_by='owner',
                           allow_settled_target=True)
            # The live server's schedule tick persists the new hold on its own (~0.5 s); wait for it so
            # the stale click meets an already-refreshed row and changes nothing (#1462).
            deadline = time.monotonic() + 30
            while load_schedule_store(root)['tasks'][0]['followup_hold']['hold_id'] == old_id:
                assert time.monotonic() < deadline, 'the schedule tick never persisted the new hold'
                time.sleep(0.1)
            with page.expect_response(lambda r: r.request.method == 'POST' and r.url.endswith('/action')) as receipt:
                button.click()
            result = receipt.value.json()
            assert result['status'] == 'stale_hold', result
            assert result['ok'] is False and result['changed'] is False, result
            assert result['running_or_queued'] is True, result
            assert receipt.value.request.post_data_json['expected_hold_id'] == old_id
            [current] = load_schedule_store(root)['tasks']
            assert current['followup_hold']['hold_id'] != old_id
            assert current['last_task_id'] == fired_id and current['enabled'] is False
            page.wait_for_function("() => document.querySelector('#toast-stack .toast')")
            toasts = page.locator('#toast-stack').inner_text()
            (evidence / 'stale-restore.json').write_text(json.dumps({
                'response': result, 'request': receipt.value.request.post_data_json,
                'toasts': toasts, 'schedule': current, 'candidate': server['candidate_identity'],
            }, indent=2, default=str))
            page.wait_for_function("() => [...document.querySelectorAll('#toast-stack .toast')].every(el => Number(getComputedStyle(el).opacity) > 0.95)")
            page.screenshot(path=str(evidence / 'g1-activity-stale-restore.png'), full_page=True)
            # Refused without changing anything (activity.js scheduleOutcomeToast; the policy suite covers
            # the request that itself persisted the new hold, which reads "refused").
            assert page.locator('#toast-stack .toast-danger').filter(
                has_text='restore did not change anything: stale_hold').count() == 1, toasts
            assert 'restored' not in toasts.lower() and 'applied' not in toasts.lower(), toasts
            # The backend's real audit-failure path is covered in the policy
            # suite. Here adapt a REAL applied release response to that wire
            # outcome to distinguish the two consumer messages in one browser.
            applied = []
            def incomplete_audit(route):
                response = route.fetch()
                ack = response.json()
                assert ack['status'] == 'hold_released', ack
                applied.append(ack)
                route.fulfill(response=response, json={**ack, 'ok': False,
                    'status': 'changed_audit_incomplete', 'audit': 'incomplete'})
            page.route('**/api/schedules/*/action', incomplete_audit)
            page.locator('#toast-stack').evaluate('(el) => el.replaceChildren()')
            # The refusal re-renders the section with the persisted new hold; press that control.
            section.locator(f'[data-hold-id="{current["followup_hold"]["hold_id"]}"]').wait_for()
            button.click()
            page.locator('#toast-stack .toast-warn').filter(has_text='applied, but its audit record is incomplete').wait_for()
            assert len(applied) == 1
            assert not load_schedule_store(root)['tasks'][0].get('followup_hold')
            page.wait_for_function("() => Number(getComputedStyle(document.querySelector('#toast-stack .toast-warn')).opacity) > 0.95")
            page.screenshot(path=str(evidence / 'g1-activity-applied-audit-incomplete.png'), full_page=True)
            print('G1_ACTIVITY_SCREENSHOTS', evidence)
        finally:
            browser.close()
