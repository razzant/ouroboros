import assert from 'node:assert/strict';
import test from 'node:test';

// An answered root under the owner's Pause is a census-only Activity row whose census phase only guesses at a park
// (#2669). The confirmed work whose live episode it is says what is true (the server's work projection, shaped as
// `gateway/work_projection.works_snapshot` sends it): its word and line, the row's own outcome still leading, and only
// the controls that work lists. The browser suite tests/test_owner_pause_ended_members_browser.py renders the same rule
// over a backend-generated projection.
globalThis.document = globalThis.document || { createElement: () => ({}) };
const { initActivity, liveActivityRowHtml } = await import('../modules/activity.js');

const census = (phase, extra = {}) => ({ activity_id: 'root-1', kind: 'managed_task', phase, pause_cause: 'owner',
    status: 'completed', cancelable: true, can_resume: false, started_at: 0, ...extra });
const paused = { code: 'paused_by_owner', text: 'You paused it', since: '2026-11-04T01:00:00+00:00',
    by: { actor: 'owner', text: '', at: '2026-11-04T01:00:00+00:00' } };
const work = (activity, transition, tail, controls) => ({ work_id: 'root-1', room: 1,
    episode: { task_id: 'root-1', live: true }, activity, transition, live: activity === 'working', confirmed: true,
    because: [paused], tails: [tail], controls, release: activity === 'unknown' ? 'none' : 'mind' });
const unconfirmed = (id) => ({ code: 'external_runs_unconfirmed', text: '1 external run may still be running', count: 1,
    ids: [id], kind: 'run', state: 'unknown' });
const running = { code: 'delegated_running', text: '1 external run still working', count: 1, ids: ['run-leaf'],
    kind: 'run', state: 'live' };
const STOP = [{ action: 'stop', target: 'root-1', scope: 'cascade' }];
// The caller's outcome slot, made legible: the chip's word and its quieter second fact.
const slot = (view) => `<chip>${view.text}|${view.secondary || ''}</chip>`;

test('an ended work requires no census row and keeps only its exact projected control', () => {
    const snapshot = { ...work('unknown', 'none', unconfirmed('inv-1'), STOP), because: [], outcome: 'completed',
        name: 'Delivered report', work_id: 'series', episode: { task_id: 'root-1', live: false } };
    const row = liveActivityRowHtml(null, 0, slot, false, snapshot);
    assert.match(row, /Delivered report/);
    assert.match(row, /<chip>Done\|Activity unconfirmed<\/chip>.*1 external run may still be running/);
    assert.match(row, /data-id="root-1" data-root="1" data-tail-work="series"/);
    assert.doesNotMatch(row, /data-resume|data-budget-paused/);
    assert.doesNotMatch(liveActivityRowHtml(null, 0, slot, false, { ...snapshot, controls: [] }), /data-act=/);
});

test('a parked census row says its confirmed work, the outcome leading, never the census guess', () => {
    const row = liveActivityRowHtml(census('budget_paused'), 0, slot, false,
        work('unknown', 'none', unconfirmed('run-leaf'), STOP));
    assert.match(row, /<chip>Done\|Activity unconfirmed<\/chip> · <span data-i18n-skip>You paused it · [^·<]+ · 1 external run may still be running<\/span>/);
    assert.doesNotMatch(row, /Paused|Pausing/);
    assert.match(row, /data-act="task-control" data-id="root-1"/, 'its work lists Stop: the row offers it');
    // The run seen running: the work says the Pause is still settling.
    const live = liveActivityRowHtml(census('budget_pausing', { live: true }), 0, slot, false,
        work('working', 'pausing', running, [{ action: 'pause', target: 'root-1' }, ...STOP]));
    assert.match(live, /<chip>Done\|Pausing…<\/chip> · <span data-i18n-skip>You paused it · [^·<]+ · 1 external run still working<\/span>/);
});

test('a work that lists no control offers none: no trigger, so no generic Stop or Resume', () => {
    const row = liveActivityRowHtml(census('budget_paused'), 0, slot, false,
        work('unknown', 'none', unconfirmed('inv-1'), []));
    assert.match(row, /Activity unconfirmed/);
    assert.doesNotMatch(row, /data-act="task-control"/);
    // Positive control: without a confirmed work the row keeps its census word and the shared control.
    const unread = liveActivityRowHtml(census('budget_paused', { status: undefined }), 0, null);
    assert.match(unread, /<span data-i18n-skip>Paused<\/span>/);
    assert.match(unread, /data-act="task-control" data-id="root-1" data-root="1" data-budget-paused="1"/);
});

test('a late row takes its state and line from the work; rows that are not parked keep their census words', () => {
    const late = { title: 'Audit the release', answered_at: '' };
    const unknown = liveActivityRowHtml(census('budget_paused', { late_work: late }), 0, slot, false,
        work('unknown', 'none', unconfirmed('run-leaf'), STOP));
    assert.match(unknown, /Remaining work status unknown · answer delivered · You paused it · [^·<]+ · 1 external run may still be running/);
    assert.doesNotMatch(unknown, /Remaining work paused/);
    const live = liveActivityRowHtml(census('budget_pausing', { late_work: late }), 0, slot, false,
        work('working', 'pausing', running, STOP));
    assert.match(live, /Remaining work pausing · answer delivered · You paused it · [^·<]+ · 1 external run still working/);
    // A working direct turn is no parked guess: its census words stand, its work beside it unchanged.
    const turn = liveActivityRowHtml({ activity_id: 'turn', kind: 'direct_chat', phase: 'working', started_at: 0 }, 0,
        null, false, { ...work('working', 'none', running, STOP), episode: { task_id: 'turn', live: true } });
    assert.match(turn, /data-act="task-control" data-id="turn" data-root="1"/);
    assert.doesNotMatch(turn, /data-tail-work/);
    assert.match(turn, /<span class="ui-row-meta activity-sub"><span class="status-mark activity-mark"[^>]*>(?:<i><\/i>){3}<\/span><span data-i18n-skip>Working<\/span><\/span>/);
});

test('a work the server marks with an unbound start arms its Stop trigger with the truthful confirmation (#3428)', () => {
    const ended = { ...work('unknown', 'none', unconfirmed('inv-1'), STOP), because: [], outcome: 'completed',
        name: 'Delivered report', episode: { task_id: 'root-1-c', live: false } };
    assert.doesNotMatch(liveActivityRowHtml(null, 0, slot, false, ended), /data-unbound-start/,
        'without the server fact the ordinary bound-run confirmation stays');
    const row = liveActivityRowHtml(null, 0, slot, false, { ...ended, unbound_start: true });
    assert.match(row, /data-act="task-control" data-id="root-1-c"[^>]*data-tail-work="root-1" data-unbound-start="1"/);
    assert.match(row, /data-task-control-direct="1">Stop now<\/button>/, 'one projected Stop opens confirmation directly');
});

test('an incomplete work projection never proves the Activity list empty (#3428)', async (t) => {
    const { setup } = await import('./dashboard_dom_fixture.js');
    const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
    const { mount, routes, ws } = setup(t);
    routes.set('/api/tasks?queue_only=1', response({ queue: { running: [], pending: [] } }));
    routes.set('/api/schedules', response({ tasks: [] }));
    routes.set('/api/services/kept', response({ services: [], complete: true }));
    const state = (extra) => response({ bg_consciousness_enabled: false, active_chat_activities: [],
        active_chat_activities_complete: true, works: [], ...extra });
    const queue = () => mount.querySelector('[data-activity-section="queue"]');
    const activity = initActivity({ mount, ws });

    routes.set('/api/state', state({ works_complete: true }));
    await activity.refresh();
    assert.match(queue().textContent, /Nothing running or queued/, 'both sources complete: the list proves empty');

    // The census is complete while the work projection is not (an unreadable delegated custody): an ended
    // task's unconfirmed helper may be missing from the empty works list, so "nothing" is not proven.
    routes.set('/api/state', state({ works_complete: false, works_gaps: [{ reason: 'delegated_custody_unavailable' }] }));
    await activity.refresh();
    assert.doesNotMatch(queue().textContent, /Nothing running or queued/);
    assert.match(queue().textContent, /Queue empty; work status unknown/);

    // An incomplete census keeps its own words, whatever the projection says.
    routes.set('/api/state', state({ active_chat_activities_complete: false, works_complete: true }));
    await activity.refresh();
    assert.match(queue().textContent, /Queue empty; live turns unknown/);
    activity.destroy();
});

test('Activity rechecks each work control against accepted state before queue repaint', async (t) => {
    const { setup } = await import('./dashboard_dom_fixture.js');
    const { closeTaskControlMenu } = await import('../modules/task_control_menu.js');
    const previousObserver = globalThis.IntersectionObserver;
    let activity;
    t.after(() => { closeTaskControlMenu(); activity?.destroy(); globalThis.IntersectionObserver = previousObserver; });
    const { mount, routes, calls, ws, document, window } = setup(t);
    const create = document.createElement;
    const rect = () => ({ top: 40, left: 40, right: 160, bottom: 70, width: 120, height: 30 });
    document.createElement = (tag) => Object.assign(create(tag), { getBoundingClientRect: rect, style: { setProperty() {} } });
    Object.assign(window, { innerWidth: 1024, innerHeight: 768, clearTimeout: globalThis.clearTimeout });
    globalThis.IntersectionObserver = class { observe() {} takeRecords() { return []; } disconnect() {} };

    const response = (data) => ({ ok: true, status: 200, json: async () => data });
    const queueUrl = '/api/tasks?queue_only=1';
    const queue = response({ queue: { running: [], pending: [] } });
    const stops = ['prior-two', 'prior-one'].map((target) => ({ action: 'stop', target, scope: 'cascade' }));
    const state = (pauseTarget, position) => response({ supervisor_ready: true, works_complete: true,
        bg_consciousness_enabled: false, bg_consciousness_state: { position },
        active_chat_activities: [], active_chat_activities_complete: true,
        works: [{ ...work('working', 'none', running, [{ action: 'pause', target: pauseTarget, scope: 'cascade' }, ...stops]),
            name: 'Report follow-up', work_id: 'report-work', outcome: 'completed', because: [],
            episode: { task_id: 'report-latest', live: false } }],
    });
    routes.set(queueUrl, queue);
    routes.set('/api/schedules', response({ tasks: [] }));
    routes.set('/api/services/kept', response({ services: [], complete: true }));
    routes.set('/api/tasks/report-latest', response({ status: 'completed' }));
    for (const { target } of stops) routes.set(`/api/tasks/${target}/cancel`, response({ ok: true }));
    activity = initActivity({ mount, ws });
    const anchor = () => mount.querySelector('[data-tail-work="report-work"]');
    const menu = () => document.body.querySelector('.task-control-menu');
    const settle = () => new Promise((resolve) => setImmediate(resolve));

    const pressDuringRefresh = async (action, expected) => {
        routes.set('/api/state', state('prior-one', 'Initial state'));
        await activity.refresh();
        const button = anchor();
        button.getBoundingClientRect = rect;
        await mount.fire('click', { target: button });
        const portal = menu();
        const item = portal?.querySelectorAll('[role="menuitem"]').find((row) => row.dataset.taskControl === action);
        assert.ok(item, `${action} is offered in the original menu`);

        let releaseRows;
        routes.set(queueUrl, new Promise((resolve) => { releaseRows = resolve; }));
        routes.set('/api/state', state('prior-two', 'Updated state'));
        const repaint = activity.refresh();
        try {
            await settle();
            assert.match(mount.querySelector('[data-activity-section="background"]').textContent, /Updated state/);
            assert.equal(anchor(), button, 'the queue row has not been repainted yet');
            assert.equal(menu(), portal, 'the original menu is still on screen');
            routes.set(queueUrl, queue); // release future reads, leaving this refresh blocked
            const since = calls.length;
            await item.fire('click', { stopPropagation() {} });
            await settle();
            assert.deepEqual(calls.slice(since).filter((url) => /\/(pause|cancel|resume)$/.test(url)), expected);
        } finally {
            releaseRows(queue);
            await repaint;
        }
    };
    await pressDuringRefresh('pause', []);
    await pressDuringRefresh('stop_now', ['/api/tasks/prior-two/cancel', '/api/tasks/prior-one/cancel']);
});
