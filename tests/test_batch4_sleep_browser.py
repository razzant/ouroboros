"""Real warm/cold consumer probes, test-only MockLLM and disposable fixture roots.

Each selected sleep frees pooled capacity for BRAVO without replacing ALPHA's
identity. Warm sleep receives owner Pause; cold sleep crosses owner Restart.
A typed Hurry makes the sleep ready, but owner holds still require explicit
Resume. No short timer wakes the model. The second mock call is held while
we inspect original started_at and accumulated sleep exclusion.
"""
from __future__ import annotations

import json
import threading

import pytest

from tests import test_batch4_owner_controls_browser as b4
from tests import test_ui_smoke_playwright as smoke

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
direct_server_with_data = smoke.direct_server_with_data


@pytest.fixture(autouse=True)
def _keep_candidate(monkeypatch):
    original = smoke.isolated_environment
    monkeypatch.setattr(smoke, 'isolated_environment',
                        lambda *a, **k: {**original(*a, **k), 'OUROBOROS_DISABLE_MANAGED_UPDATES': '1'})


class SleepModel(b4._ScriptedModel):
    def __init__(self, monkeypatch, mode):
        super().__init__(monkeypatch)
        self.mode = mode
        self.resumed, self.finish = threading.Event(), threading.Event()

    def handle(self, handler):
        payload = json.loads(handler.rfile.read(int(handler.headers.get('Content-Length', 0))) or b'{}')
        marker, main = self.marker(payload), bool(payload.get('tools'))
        with self.lock:
            self.calls.append((marker, main))
            ordinal = self.calls.count((marker, True))
        message, reason = {'role': 'assistant', 'content': 'OK'}, 'stop'
        if main and marker == 'ALPHA' and ordinal == 1:
            message, reason = {'role': 'assistant', 'content': '', 'tool_calls': [{
                'id': 'call_sleep', 'type': 'function', 'function': {
                    'name': 'await_messages', 'arguments': json.dumps({'mode': self.mode})}}]}, 'tool_calls'
        elif main and marker == 'ALPHA' and ordinal == 2:
            self.resumed.set()
            self.finish.wait(90)
        b4._reply(handler, payload, message, reason)


def snapshot(root):
    return json.loads((root / 'state/queue_snapshot.json').read_text(encoding='utf-8'))


def open_activity_fresh(page):
    # Re-enter a real consumer surface; clicking the already active tab does not refresh.
    page.locator('[data-nav-page="dashboard"]').click()
    page.locator('[data-dashboard-tab="logs"]').click()
    page.locator('[data-dashboard-tab="activity"]').click()
    return page.locator('#dashboard-panel-activity')


def member(snapshot, task_id, kind):
    return next((r for r in snapshot.get(kind, []) if r.get('id') == task_id), {})


@pytest.mark.parametrize('mode', ['warm', 'cold'])
def test_sleep_lends_capacity_preserves_identity_and_owner_hold(direct_server_with_data, monkeypatch, mode):
    from playwright.sync_api import expect, sync_playwright
    from ouroboros.budget_pause import budget_pause_row
    from ouroboros.task_results import load_task_result

    server = direct_server_with_data
    root, url = server['data_dir'], server['url']
    evidence = b4._evidence_dir(root, 'sleep-' + mode)
    model = SleepModel(monkeypatch, mode)
    server['stop_server']()
    b4._seed_roots(root, (b4.ALPHA, b4.BRAVO))
    server['start_server']()
    b4._resume_after_app_stop(url, b4.ALPHA, b4.BRAVO)  # the seeded Quit holds both (owner S1)
    record = {'mode': mode}
    try:
        with sync_playwright() as pw:
            browser, page, errors = b4._launch(pw)
            record['engine_version'] = browser.version
            try:
                b4._open_chat(page, url)
                # BRAVO actually finishes while ALPHA sleeps: lending capacity is
                # distinguished from merely renaming the active task's phase.
                b4._wait(lambda: b4._task(page, url, b4.BRAVO).get('status') == 'completed',
                         90, 'independent work completed while ALPHA sleeps')
                assert model.main_calls('ALPHA') == 1
                if mode == 'warm':
                    parked = b4._wait(lambda: member(snapshot(root), b4.ALPHA, 'running').get('owner_wait'),
                                      30, 'warm same-stack wait')
                    before = member(snapshot(root), b4.ALPHA, 'running')
                    assert parked['reason'] == 'sleep' and parked['sleep']['mode'] == 'warm'
                    assert not member(snapshot(root), b4.ALPHA, 'pending')
                    original_started = before['started_at']
                else:
                    parked = b4._wait(lambda: budget_pause_row(root, b4.ALPHA), 30, 'cold exact checkpoint')
                    before = member(snapshot(root), b4.ALPHA, 'pending')
                    assert parked['reason'] == 'sleep' and parked['sleep']['mode'] == 'cold'
                    assert before and not member(snapshot(root), b4.ALPHA, 'running')
                    original_started = parked['started_at']
                record['parked'] = parked
                record['original_started_at'] = original_started
                record['sleep_snapshot'] = snapshot(root)
                record['sleep_queue_api'] = b4._get(page, url, '/api/tasks?queue_only=1')
                open_activity_fresh(page)
                sleep_label = page.locator('.activity-row', has_text='Batch4 Alpha').inner_text()
                record['sleep_activity_label'] = sleep_label
                page.screenshot(path=str(evidence / '01-sleeping-capacity-released.png'))

                if mode == 'warm':
                    ack = page.request.post(url + f'/api/tasks/{b4.ALPHA}/pause',
                                            data={'request_id': 'sleep-warm-owner-pause'})
                    assert ack.ok, ack.text()
                    record['hold_ack'] = ack.json()
                    b4._wait(lambda: b4._fence(root, b4.ALPHA).get('state') == 'paused', 60, 'saved owner Pause')
                else:
                    page.locator('[data-nav-page="chat"]').click()
                    b4._restart_dialog(page, page.locator('[data-chat-command="restart"]'))
                    b4._owner_restart(page, url, root)
                    b4._wait(lambda: b4._task(page, url, b4.ALPHA).get('reason_code') == 'owner_restart_hold',
                             30, 'cold sleep retained by owner Restart')
                held = b4._task(page, url, b4.ALPHA)
                record['held_task'] = held
                assert held['status'] == 'scheduled'
                checkpoint_started = held['budget_pause']['started_at']
                record['checkpoint_started_at'] = checkpoint_started
                # Pool dispatch and loop start are distinct timestamps; both
                # must precede this sleep, and Resume retains the saved loop clock.
                assert checkpoint_started < held['budget_pause']['paused_at']
                assert model.main_calls('ALPHA') == 1

                # Existing HTTP typed-control route addresses the real mailbox.
                wake = page.request.post(url + f'/api/tasks/{b4.ALPHA}/hurry',
                                         data={'request_id': 'sleep-' + mode + '-hurry'})
                assert wake.ok, wake.text()
                record['wake_ack'] = wake.json()
                # Negative bounded observation is paired with real positive
                # Resume below; the sleep itself has no elapsed-time wake.
                assert not model.resumed.wait(3), 'readiness bypassed owner hold'
                assert model.main_calls('ALPHA') == 1
                record['held_queue_api'] = b4._get(page, url, '/api/tasks?queue_only=1')
                open_activity_fresh(page)
                hold_label = 'paused' if mode == 'warm' else 'held after Restart'
                expect(page.locator('.activity-row', has_text='Batch4 Alpha')).to_contain_text(hold_label, timeout=30_000)
                page.screenshot(path=str(evidence / '02-ready-but-owner-held.png'))
                trigger = page.locator(f'[data-act="task-control"][data-id="{b4.ALPHA}"]')
                actions = b4._menu_action(page, trigger, 'resume')
                assert actions == ['resume', 'stop_now']
                assert model.resumed.wait(60), 'explicit owner Resume did not continue the same sleeper'
                running = b4._wait(lambda: member(snapshot(root), b4.ALPHA, 'running'), 15, 'resumed task snapshot')
                record['resumed_snapshot'] = running
                record['resumed_queue_api'] = b4._get(page, url, '/api/tasks?queue_only=1')
                open_activity_fresh(page)
                active = page.locator('.activity-row', has_text='Batch4 Alpha')
                expect(active).to_contain_text('running', timeout=30_000)
                expect(active).not_to_contain_text('sleep')
                page.screenshot(path=str(evidence / '03-resumed-running-not-sleeping.png'))
                assert running['id'] == b4.ALPHA and running['started_at'] == checkpoint_started
                assert running['budget_paused_sec'] > 0, 'sleep interval was discarded'
                model.finish.set()
                done = b4._wait(lambda: (lambda r: r if r.get('status') == 'completed' else None)(
                    b4._task(page, url, b4.ALPHA)), 90, 'same sleeper completed')
                record['done'] = done
                assert done['root_task_id'] == b4.ALPHA
                assert not load_task_result(root, b4.ALPHA).get('continued_by')
                page.locator('[data-nav-page="chat"]').click()
                expect(b4._chip(page, b4.ALPHA)).to_have_text('Done', timeout=30_000)
                page.screenshot(path=str(evidence / '04-same-task-finished.png'))
                assert not errors
                assert 'sleep' in sleep_label.lower(), 'Sleeping task displayed as: ' + sleep_label
            except Exception:
                page.screenshot(path=str(evidence / 'failure.png'))
                raise
            finally:
                record['page_errors'] = errors
                browser.close()
    finally:
        model.finish.set()
        model.teardown.set()
        record['model_calls'] = model.calls
        (evidence / 'record.json').write_text(json.dumps(record, indent=2, default=str), encoding='utf-8')
