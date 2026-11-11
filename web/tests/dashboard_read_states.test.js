import assert from 'node:assert/strict';
import test from 'node:test';
import { initActivity } from '../modules/activity.js';
import { initLogs } from '../modules/logs.js';
import { initCosts } from '../modules/costs.js';
import { initDashboard } from '../modules/dashboard.js';
import { initStorage } from '../modules/storage.js';
import { NodeStub, setup } from './dashboard_dom_fixture.js';

const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
const queueUrl = '/api/tasks?queue_only=1';
// Explicit gateway projection fixtures; Activity no longer re-derives stops from queue reason names.
const projectedWork = (id, activity = 'working', extra = {}) => ({ work_id: id, confirmed: true,
    activity, transition: 'none', because: [], tails: [], episode: { task_id: id, live: true },
    controls: activity === 'paused' ? [{ action: 'resume', target: id }] : [], ...extra });
const backgroundUrl = '/api/state';
const schedulesUrl = '/api/schedules';
const keptUrl = '/api/services/kept';
const logUrl = (name) => `/api/logs/${name}?view=compact&limit=150`;
const settle = () => new Promise((resolve) => setImmediate(resolve));
const section = (mount, name) => mount.querySelector(`[data-activity-section="${name}"]`);
const message = (node) => node.querySelector('.ui-status').textContent;

function emptyActivity(routes) {
    routes.set(queueUrl, response({ queue: { running: [], pending: [] } }));
    routes.set(backgroundUrl, response({ bg_consciousness_enabled: false, active_chat_activities: [], active_chat_activities_complete: true }));
    routes.set(schedulesUrl, response({ tasks: [] }));
    routes.set(keptUrl, response({ services: [], complete: true, note: '' }));
}

test('Activity task list waits for migration and refreshes itself after publication', async t => {
    const { mount, routes, ws, calls, toasts } = setup(t);
    emptyActivity(routes);
    routes.set(queueUrl, response({ reason_code: 'migrating' }, 503));
    const activity = initActivity({ mount, ws });
    await activity.refresh();
    assert.equal(message(section(mount, 'queue')), 'Saved tasks are being upgraded…');
    assert.deepEqual(toasts(), []);
    await new Promise(resolve => setTimeout(resolve, 750));
    assert.equal(calls.filter(url => url === queueUrl).length, 2);
    emptyActivity(routes);
    await new Promise(resolve => setTimeout(resolve, 750));
    assert.equal(calls.filter(url => url === queueUrl).length, 3);
    assert.equal(message(section(mount, 'queue')), '');
    assert.match(section(mount, 'queue').textContent, /Nothing running or queued/);
    await new Promise(resolve => setTimeout(resolve, 750));
    assert.equal(calls.filter(url => url === queueUrl).length, 3);
    routes.set(queueUrl, response({ reason_code: 'migrating' }, 503));
    await activity.refresh();
    activity.destroy();
    await new Promise(resolve => setTimeout(resolve, 750));
    assert.equal(calls.filter(url => url === queueUrl).length, 4, 'destroy cancels migration retries');
});

test('Activity names known non-Project scope waits without implying a Project', async (t) => {
    const { mount, routes, ws } = setup(t);
    emptyActivity(routes);
    routes.set(queueUrl, response({ queue: { running: [], pending: [{ id: 'main', task: {
        title: 'Original work', _project_scope_none: true,
        project_admission_hold: { label: 'Waiting for task scope verification', detail: '<bindings unavailable>' },
    } }] } }));
    const activity = initActivity({ mount, ws });
    await activity.refresh();
    assert.match(mount.textContent, /Waiting for task scope verification/);
    assert.doesNotMatch(mount.textContent, /Waiting for Project verification/);
    assert.match(mount.textContent, /&lt;bindings unavailable&gt;/);
});

test('Activity names a run as its card does, keeps the whole order behind Details, and refreshes without a word', async (t) => {
    const { mount, routes, ws } = setup(t);
    emptyActivity(routes);
    const order = 'Complete two confirmed recovery defects\nof the existing candidate <exact tree>.';
    routes.set(queueUrl, response({ queue: { running: [
        { id: 'named', type: 'task', runtime_sec: 60, task: { id: 'named', suggested_name: 'Complete two confirmed recovery defects', text: order } },
    ], pending: [{ task: { id: 'short', suggested_name: 'Continue the project', text: 'Continue the project.' } }] } }));
    const activity = initActivity({ mount, ws });
    await activity.refresh();
    const rows = section(mount, 'queue').querySelectorAll('.activity-row');
    assert.equal(rows[0].querySelector('.activity-name').textContent, 'Complete two confirmed recovery defects');
    const details = rows[0].querySelector('details');
    assert.equal(details.querySelector('summary').textContent, 'Details');
    assert.match(details.querySelector('.activity-order').textContent, /of the existing candidate &lt;exact tree&gt;\./);
    assert.doesNotMatch(rows[0].querySelector('.activity-name').textContent, /exact tree/, 'never the whole order as its title');
    assert.equal(rows[1].querySelector('details'), null, 'an order its name already says keeps no copy behind Details');
    // A refresh reads silently (aria-busy only): no line comes and goes under the heading and moves the rows.
    details.open = true;
    const again = activity.refresh();
    assert.equal(message(section(mount, 'queue')), '');
    await again;
    assert.equal(section(mount, 'queue').querySelector('.activity-row').querySelector('details').hasAttribute('open'), true, 'an open Details stays open');
});

test('Activity failed reads stay unknown; independent successful empty state stays empty', async (t) => {
    const { mount, routes, ws } = setup(t);
    emptyActivity(routes);
    routes.set(queueUrl, response({ error: 'offline' }, 503));
    routes.set(backgroundUrl, new Error('offline'));
    const activity = initActivity({ mount, ws });
    await activity.refresh();
    assert.match(message(section(mount, 'queue')), /Could not load.*unknown/);
    assert.doesNotMatch(section(mount, 'queue').textContent, /Nothing running/);
    assert.match(message(section(mount, 'background')), /Could not load/);
    assert.equal(section(mount, 'background').querySelectorAll('button').length, 0, 'unknown must not invent Start');
    assert.match(section(mount, 'schedules').textContent, /No scheduled tasks/);
    assert.equal(message(section(mount, 'schedules')), '');
    emptyActivity(routes);
    await activity.refresh();
    assert.match(section(mount, 'queue').textContent, /Nothing running or queued/);
    assert.equal(message(section(mount, 'queue')), '');
    assert.equal(section(mount, 'background').querySelector('button').textContent, 'Start');
});

test('Activity keeps last-known data and actions on refresh failure, then reconciles recovery', async (t) => {
    const { mount, routes, ws } = setup(t);
    emptyActivity(routes);
    routes.set(schedulesUrl, response({ tasks: [{ id: 'scheduled', name: 'Daily report', enabled: true }] }));
    const activity = initActivity({ mount, ws });
    await activity.refresh();
    const scheduled = section(mount, 'schedules');
    routes.set(schedulesUrl, response({}, 503));
    await activity.refresh();
    assert.match(scheduled.textContent, /Daily report/);
    assert.match(message(scheduled), /Previously loaded.*unknown/);
    assert.ok(scheduled.querySelectorAll('button').every((button) => !button.disabled), 'failed refresh does not remove existing capability');
    assert.equal(section(mount, 'background').querySelector('button').disabled, false);
    routes.set(schedulesUrl, response({ tasks: [{ id: 'scheduled', name: 'Daily report', enabled: false }] }));
    await activity.refresh();
    assert.equal(message(scheduled), '');
    assert.equal(scheduled.querySelector('button').textContent, 'Enable');
    assert.equal(scheduled.querySelector('button').disabled, false);
});

const lifecycleSchedules = [
    { id: 'active', name: 'Active schedule', enabled: true, status: 'active', task: { text: 'Inspect the results\nKeep the full instruction readable.' }, trigger: { type: 'cron', expr: '* * * * *' } },
    { id: 'disabled', name: 'Disabled schedule', enabled: false, status: 'disabled', trigger: { type: 'cron', expr: '* * * * *' } },
    { id: 'withdrawn', name: 'Withdrawn promise', enabled: false, status: 'delete_pending', withdrawal: { restorable: true, work_in_flight: false }, trigger: { type: 'cron', expr: '* * * * *' } },
    { id: 'consumed', name: 'Consumed once', enabled: false, status: 'consumed', consumed: true, retained: true, completed_at: '2026-09-21T00:00:00Z', trigger: { type: 'once', run_at: '2026-09-20T00:00:00Z' } },
];

test('Activity separates standing schedules from retained history and keeps Restore', async (t) => {
    const { mount, routes, ws } = setup(t);
    emptyActivity(routes);
    routes.set(schedulesUrl, response({ tasks: lifecycleSchedules }));
    await initActivity({ mount, ws }).refresh();
    const schedules = section(mount, 'schedules');
    const history = schedules.querySelector('[data-activity-history]');

    // Standing schedules are the list; history is one collapsed disclosure.
    const standing = [...schedules.querySelectorAll('.activity-row')].filter((row) => !history.contains(row));
    assert.equal(standing.length, 2);
    assert.match(standing[0].textContent, /Active schedule[\s\S]*active/);
    assert.equal(standing[0].querySelector('button').textContent, 'Disable');
    assert.equal(standing[0].querySelector('button').dataset.action, 'disable');
    assert.match(standing[1].textContent, /Disabled schedule[\s\S]*disabled/);
    assert.equal(standing[1].querySelector('button').textContent, 'Enable');
    assert.equal(standing[1].querySelector('button').dataset.action, 'restore');
    assert.equal(standing[1].querySelector('button').dataset.done, 'enabled', 'its outcome says the button\'s own word');

    assert.equal(history.hasAttribute('open'), false);
    assert.match(history.querySelector('summary').textContent, /withdrawn \(2\)/);
    const retained = history.querySelectorAll('.activity-row');
    assert.equal(retained.length, 2);
    assert.match(retained[0].textContent, /Withdrawn promise[\s\S]*withdrawn · restorable/);
    assert.equal(retained[0].querySelector('button').textContent, 'Restore');
    // A fired one-shot cannot be re-armed, so it offers no Enable — only Delete.
    assert.match(retained[1].textContent, /Consumed once[\s\S]*consumed/);
    assert.match(retained[1].textContent, /consumed once · history/);
    const consumedButtons = [...retained[1].querySelectorAll('button')].map((b) => b.dataset.act);
    assert.deepEqual(consumedButtons, ['schedule-delete']);
});

test('Activity with only retained rows says so instead of showing an empty list', async (t) => {
    const { mount, routes, ws } = setup(t);
    emptyActivity(routes);
    routes.set(schedulesUrl, response({ tasks: lifecycleSchedules.slice(2) }));
    await initActivity({ mount, ws }).refresh();
    const schedules = section(mount, 'schedules');
    assert.match(schedules.textContent, /No active or disabled schedules\./);
    assert.equal(schedules.querySelector('[data-activity-history]').querySelectorAll('.activity-row').length, 2);
});

test('Activity lifecycle buttons name their action to the one audited endpoint', async (t) => {
    const { mount, routes, ws } = setup(t);
    emptyActivity(routes);
    routes.set(schedulesUrl, response({ tasks: lifecycleSchedules }));
    const sent = [];
    for (const id of ['active', 'withdrawn']) {
        routes.set(`/api/schedules/${id}/action`, (url, init) => {
            sent.push({ url, body: JSON.parse(init.body) });
            return response({ ok: true, changed: true, status: 'updated', audit: 'recorded', running_or_queued: false });
        });
    }
    await initActivity({ mount, ws }).refresh();
    const schedules = section(mount, 'schedules');
    const history = schedules.querySelector('[data-activity-history]');
    await mount.fire('click', { target: schedules.querySelector('[data-act="schedule-toggle"][data-id="active"]') });
    await mount.fire('click', { target: history.querySelector('[data-act="schedule-toggle"][data-id="withdrawn"]') });
    assert.deepEqual(sent.map((entry) => [entry.url, entry.body.action]), [
        ['/api/schedules/active/action', 'disable'],
        ['/api/schedules/withdrawn/action', 'restore'],
    ]);
    // Every change carries a reason; nothing infers the command from that text.
    assert.ok(sent.every((entry) => typeof entry.body.reason === 'string' && entry.body.reason.length > 0));
    // No full-record round trip: a stale read can never overwrite runtime fields.
    assert.equal(sent.every((entry) => !('task' in entry.body) && !('enabled' in entry.body)), true);
});

test('Activity reports what a lifecycle action actually did, delete included', async (t) => {
    const { mount, routes, ws, toasts } = setup(t);
    emptyActivity(routes);
    routes.set(schedulesUrl, response({ tasks: lifecycleSchedules }));
    // Both buttons of the same row read the SAME outcome contract; a refused
    // delete must not read as a silent no-op just because it is a delete.
    const outcomes = [
        { ok: false, changed: false, status: 'not_found', audit: 'not_written' },
        { ok: false, changed: true, status: 'changed_audit_incomplete', audit: 'incomplete',
          detail: 'the change is durable; its audit outcome record could not be written' },
        { ok: true, changed: true, status: 'updated', audit: 'recorded', running_or_queued: true },
        { ok: true, changed: true, status: 'updated', audit: 'recorded', running_or_queued: null },
        { ok: true, changed: true, status: 'deleted', audit: 'recorded', running_or_queued: false },
    ];
    let next = 0;
    for (const id of ['active', 'disabled']) {
        routes.set(`/api/schedules/${id}/action`, () => response(outcomes[next++]));
    }
    const activity = initActivity({ mount, ws });
    await activity.refresh();
    const click = async (act, id) => {
        const target = section(mount, 'schedules').querySelector(`[data-act="${act}"][data-id="${id}"]`);
        await mount.fire('click', { target });
    };
    await click('schedule-toggle', 'active');
    assert.match(toasts().at(-1), /did not change anything: not_found/);
    await click('schedule-toggle', 'active');
    assert.match(toasts().at(-1), /audit record is incomplete: the change is durable/);
    await click('schedule-toggle', 'active');
    assert.match(toasts().at(-1), /queued or running and was not cancelled/);
    await click('schedule-toggle', 'active');
    assert.match(toasts().at(-1), /is unknown/);
    // A clean change with nothing in flight has nothing to disclose.
    routes.set('/api/schedules/disabled/action', () => response(outcomes[4]));
    await click('schedule-toggle', 'disabled');
    assert.equal(toasts().length, 4);
});

test('Activity does not call a schedule action applied without a positive changed receipt', async (t) => {
    const { mount, routes, ws, toasts } = setup(t);
    emptyActivity(routes);
    routes.set(schedulesUrl, response({ tasks: lifecycleSchedules }));
    const outcomes = [
        {},
        { ok: false, status: 'not_found' },
    ];
    let next = 0;
    routes.set('/api/schedules/active/action', () => response(outcomes[next++]));
    const activity = initActivity({ mount, ws });
    await activity.refresh();
    const click = async () => {
        const target = section(mount, 'schedules').querySelector(
            '[data-act="schedule-toggle"][data-id="active"]');
        await mount.fire('click', { target });
    };
    await click();
    assert.match(toasts().at(-1), /server did not say what happened/);
    assert.doesNotMatch(toasts().at(-1), /did not change anything/);
    assert.doesNotMatch(toasts().at(-1), /applied/,
        'an empty success-shaped payload must not disclose an applied mutation');
    await click();
    assert.match(toasts().at(-1), /server did not say what happened/);
    assert.doesNotMatch(toasts().at(-1), /applied/,
        'ok=false without changed=true must not disclose an applied mutation');
});

test('Activity delete requires confirmation and reports the withdrawal receipt', async (t) => {
    const { mount, routes, ws, document, toasts } = setup(t);
    emptyActivity(routes);
    routes.set(schedulesUrl, response({ tasks: [{ ...lifecycleSchedules[0], source: 'skill_manifest', skill: 'demo' }] }));
    const sent = [];
    routes.set('/api/schedules/active/action', (_, init) => {
        sent.push(JSON.parse(init.body));
        return response({ ok: true, changed: true, status: 'withdrawn', audit: 'recorded', running_or_queued: false });
    });
    await initActivity({ mount, ws }).refresh();
    for (const confirmed of [false, true]) {
        const click = mount.fire('click', { target: mount.querySelector('[data-act="schedule-delete"]') });
        const dialog = document.body.querySelector('[role="dialog"]');
        assert.match(dialog.textContent, /stays restorable for a while/);
        assert.equal(sent.length, 0, 'nothing is sent before confirmation');
        await dialog.parentElement.fire('click', { target: dialog.querySelector(confirmed ? '[data-confirm-ok]' : '[data-confirm-cancel]') });
        await click;
        assert.equal(sent.length, confirmed ? 1 : 0);
    }
    assert.deepEqual(sent, [{ action: 'delete', reason: 'owner deleted the schedule from Activity', expected_hold_id: '' }]);
    assert.deepEqual(toasts(), ['Schedule withdrawn.']);
});

test('Activity keeps the retained-history disclosure and focus the owner left open', async (t) => {
    const { mount, routes, ws, calls } = setup(t);
    emptyActivity(routes);
    // A held withdrawal offers only its exact release; the unheld one offers Restore.
    const tasks = [...lifecycleSchedules, { id: 'held', name: 'Held promise', enabled: false, status: 'delete_pending',
        withdrawal: { restorable: false, work_in_flight: false }, trigger: { type: 'cron', expr: '* * * * *' },
        relation: 'related', followup_hold: { hold_id: 'hold-1', reason: 'origin_stopped' }, hold_persisted: true }];
    routes.set(schedulesUrl, response({ tasks }));
    const activity = initActivity({ mount, ws });
    await activity.refresh();
    const schedules = section(mount, 'schedules');
    assert.equal(schedules.querySelector('[data-activity-history]').hasAttribute('open'), false);

    // The owner opens the history and focuses a control inside it.
    const details = schedules.querySelector('[data-activity-history]');
    details.open = true;
    const restore = details.querySelectorAll('button').find((button) => button.textContent === 'Restore');
    restore.focus();

    // A poll rebuilds the markup; it must not re-collapse or drop the focus.
    await activity.refresh();
    const rebuilt = section(mount, 'schedules').querySelector('[data-activity-history]');
    assert.equal(rebuilt.hasAttribute('open'), true, 'a refresh must not close what the owner opened');
    assert.equal(globalThis.document.activeElement.dataset.id, 'withdrawn');
    assert.equal(globalThis.document.activeElement.dataset.act, 'schedule-toggle');
    assert.equal(globalThis.document.activeElement.textContent, 'Restore', 'withdrawal Restore must not become Restore hold');
    rebuilt.querySelector('[data-hold-id]').focus();
    await activity.refresh();
    assert.equal(globalThis.document.activeElement.dataset.holdId, 'hold-1');

    // Collapsing it again is equally durable across the next refresh.
    schedules.querySelector('[data-activity-history]').open = false;
    await activity.refresh();
    assert.equal(section(mount, 'schedules').querySelector('[data-activity-history]').hasAttribute('open'), false);
    const objective = schedules.querySelector('[data-schedule-text]');
    objective.open = true;
    objective.querySelector('summary').focus();
    const reads = calls.length;
    await mount.fire('click', { target: objective.querySelector('summary') });
    assert.equal(calls.length, reads, 'expanding authored text does not trigger a lifecycle action or refresh');
    await activity.refresh();
    assert.equal(schedules.querySelector('[data-schedule-text]').hasAttribute('open'), true);
    assert.equal(globalThis.document.activeElement.dataset.act, 'schedule-text');
    schedules.querySelector('[data-schedule-text]').open = false;
    await activity.refresh();
    assert.equal(schedules.querySelector('[data-schedule-text]').hasAttribute('open'), false);
});

test('Activity preserves disclosure and current focus as of render, not request start', async (t) => {
    const { mount, routes, ws } = setup(t);
    emptyActivity(routes);
    routes.set(schedulesUrl, response({ tasks: lifecycleSchedules }));
    const activity = initActivity({ mount, ws });
    await activity.refresh();

    // A refresh is in flight. The owner changes both disclosure and focus while
    // waiting; a late render must preserve those current choices rather than
    // restoring the state captured when the request began.
    const details = section(mount, 'schedules').querySelector('[data-activity-history]');
    details.open = false;
    const oldControl = details.querySelector('[data-act="schedule-toggle"][data-id="withdrawn"]');
    oldControl.focus();
    let finishSchedules;
    routes.set(schedulesUrl, () => new Promise((resolve) => { finishSchedules = resolve; }));
    const pending = activity.refresh();
    const moved = new NodeStub('button');
    globalThis.document.body.appendChild(moved);
    moved.focus();
    details.open = true;
    section(mount, 'schedules').querySelector('[data-schedule-text]').open = true;
    finishSchedules(response({ tasks: lifecycleSchedules }));
    await pending;

    const rebuilt = section(mount, 'schedules').querySelector('[data-activity-history]');
    assert.equal(rebuilt.hasAttribute('open'), true,
        'a render must retain disclosure state changed while its request was pending');
    assert.equal(section(mount, 'schedules').querySelector('[data-schedule-text]').hasAttribute('open'), true);
    assert.equal(globalThis.document.activeElement, moved,
        'a render must not steal focus the owner moved while its request was pending');
});

test('Activity invalid success payload is unavailable and stale requests cannot replace newer state', async (t) => {
    const { mount, routes, ws } = setup(t);
    emptyActivity(routes);
    const deferred = [];
    routes.set(queueUrl, () => new Promise((resolve) => deferred.push(resolve)));
    const activity = initActivity({ mount, ws });
    const first = activity.refresh();
    const second = activity.refresh();
    deferred[1](response({ queue: { running: [{ id: 'current', task: { title: 'Current work' } }], pending: [] } }));
    await second;
    deferred[0](response({}, 503));
    await first;
    assert.equal(message(section(mount, 'queue')), '');
    assert.match(section(mount, 'queue').textContent, /Current work/);
    routes.set(queueUrl, response({ error: 'broken payload' }));
    await activity.refresh();
    assert.match(message(section(mount, 'queue')), /Could not refresh/);
    assert.match(section(mount, 'queue').textContent, /Current work/);
});

test('Activity unites live direct turns with queue identities and preserves queue controls', async (t) => {
    const { mount, routes, ws, calls } = setup(t);
    emptyActivity(routes);
    const now = Date.now;
    Date.now = () => 100000;
    t.after(() => { Date.now = now; });
    routes.set(backgroundUrl, response({ bg_consciousness_enabled: true, works: [projectedWork('queued-1'), projectedWork('paused-1', 'paused')],
        active_chat_activities_complete: false, active_chat_activities: [
            { activity_id: 'direct-1', kind: 'direct_chat', phase: 'thinking', started_at: 88 },
            { activity_id: 'queued-1', kind: 'direct_chat', phase: 'thinking', started_at: 80 },
            { activity_id: 'paused-1', kind: 'managed_task', phase: 'running', started_at: 80 },
        ] }));
    routes.set(queueUrl, response({ queue: {
        running: [{ id: 'queued-1', type: 'task', runtime_sec: 9, task: { title: 'Queue title' } }],
        pending: [{ task: { id: 'paused-1', title: 'Paused work', _budget_pause: true } }],
    } }));
    const activity = initActivity({ mount, ws });
    await activity.refresh();
    const queue = section(mount, 'queue');
    const rows = queue.querySelectorAll('.activity-row');
    assert.equal(rows.length, 3, 'partial positive census adds exactly the missing direct identity');
    // The list reads by state: what works now first, then paused work collapsed under its count.
    assert.match(rows[0].textContent, /Queue title[\s\S]*Working · task · 9s/);
    const paused = queue.querySelector('details[data-open-key="paused"]');
    assert.equal(paused.querySelector('summary').textContent, 'Paused (1)');
    assert.equal(paused.hasAttribute('open'), true, 'open until the owner folds it: Resume stays one press away');
    assert.ok(paused.contains(rows[2]) && !paused.contains(rows[0]) && !paused.contains(rows[1]));
    assert.match(rows[2].textContent, /Paused work[\s\S]*Paused/);
    assert.equal(rows[2].querySelector('button').dataset.budgetPaused, '1');
    // A live census row says its state in the chip's own word (translated at its producer), never the census code.
    assert.equal(rows[1].querySelector('[data-i18n-skip]').textContent, 'Thinking');
    assert.match(rows[1].textContent, /Direct turn[\s\S]*· 12s/);
    assert.doesNotMatch(rows[1].textContent, /thinking/);
    assert.equal(rows[1].querySelector('button').dataset.id, 'direct-1');
    assert.equal(rows[1].querySelector('button').dataset.act, 'task-control');
    // Stop… is the neutral shared trigger on every row, never a danger button.
    assert.deepEqual(rows.map((row) => row.querySelector('button').className), Array(3).fill('btn btn-sm btn-default'));
    assert.doesNotMatch(queue.textContent, /Nothing running/);
    // Kept services are their own read (custody, one census): the state read is still not repeated.
    assert.deepEqual(calls, [queueUrl, backgroundUrl, schedulesUrl, keptUrl], 'reuse the existing state read');
    // A group the owner folded stays folded through the next refresh.
    paused.open = false;
    await activity.refresh();
    assert.equal(section(mount, 'queue').querySelector('details[data-open-key="paused"]').hasAttribute('open'), false);
});

test('Activity names work held by an application stop as held, never as a money pause', async (t) => {
    const { mount, routes, ws } = setup(t);
    emptyActivity(routes);
    routes.set(queueUrl, response({ queue: { running: [], pending: [
        { task: { id: 'saved-1', title: 'Saved before Quit', _budget_pause_hold: { reason: 'saved_work_hold' } } },
        { task: { id: 'restart-1', title: 'Held by Restart', _budget_pause_hold: { reason: 'owner_restart_hold' } } },
        { task: { id: 'released-1', title: 'Resumed already',
            _resume_selection: { selected_by: 'owner' } } },
    ] } }));
    routes.set(backgroundUrl, response({ bg_consciousness_enabled: false, works: [
        projectedWork('saved-1', 'paused', { because: [{ code: 'saved_on_stop', text: 'Saved when the app stopped' }] }),
        projectedWork('restart-1', 'paused', { because: [{ code: 'after_restart', text: 'Held after a restart' }] }),
        projectedWork('released-1', 'queued')], active_chat_activities: [] }));
    await initActivity({ mount, ws }).refresh();
    const rows = section(mount, 'queue').querySelectorAll('.activity-row');
    assert.equal(rows.length, 3);
    // Queued work leads; the two held rows collapse under "Paused (2)", their order kept.
    assert.equal(section(mount, 'queue').querySelector('details[data-open-key="paused"]').querySelector('summary').textContent, 'Paused (2)');
    for (const row of rows.slice(1)) {
        assert.match(row.textContent, /Paused/);
        assert.doesNotMatch(row.textContent, /budget/);
        assert.equal(row.querySelector('button').dataset.budgetPaused, '1', 'the same Resume is offered');
    }
    assert.match(rows[1].textContent, /Saved before Quit/);
    // The owner's Resume released this hold: it is neither held nor paused any more.
    assert.doesNotMatch(rows[0].textContent, /Held after a restart|Paused/);
    assert.equal(rows[0].querySelector('button').dataset.budgetPaused, undefined);
});

test('Activity complete empty census differs from unknown and failed state preserves queue facts', async (t) => {
    const { mount, routes, ws } = setup(t);
    emptyActivity(routes);
    const activity = initActivity({ mount, ws });
    const queue = section(mount, 'queue');
    await activity.refresh();
    assert.match(queue.textContent, /Nothing running or queued/);
    routes.set(backgroundUrl, response({ bg_consciousness_enabled: false,
        active_chat_activities: [], active_chat_activities_complete: false }));
    await activity.refresh();
    assert.match(queue.textContent, /Queue empty; live turns unknown/);
    routes.set(backgroundUrl, response({}, 503));
    await activity.refresh();
    assert.match(queue.textContent, /Queue empty; live turns unknown/);
    assert.equal(message(queue), '');
    routes.set(queueUrl, response({ queue: { running: [{ id: 'running-1', task: { title: 'Still working' } }], pending: [] } }));
    await activity.refresh();
    assert.match(queue.textContent, /Still working/);
    assert.equal(message(queue), '');
    routes.set(backgroundUrl, response({ bg_consciousness_enabled: false,
        active_chat_activities: [{ activity_id: 'direct-2', kind: 'direct_chat', phase: 'thinking' }],
        active_chat_activities_complete: true }));
    routes.set(queueUrl, response({}, 503));
    await activity.refresh();
    assert.match(message(queue), /Could not refresh.*unknown/);
    assert.match(queue.textContent, /Still working/);
    assert.doesNotMatch(queue.textContent, /Nothing running/);
});

test('Logs reports partial history without losing live rows or deduplication, then clears the gap on reconnect', async (t) => {
    const { mount, routes, ws, calls } = setup(t);
    const live = { type: 'test_live_event', ts: '2026-09-09T12:00:00Z', message: 'live' };
    const earlier = { type: 'test_earlier_event', ts: '2026-09-09T11:59:00Z', message: 'earlier' };
    for (const name of ['events', 'tools', 'progress', 'supervisor']) routes.set(logUrl(name), response({ entries: [] }));
    let finishEvents;
    routes.set(logUrl('events'), () => new Promise((resolve) => { finishEvents = resolve; }));
    routes.set(logUrl('tools'), response({}, 503));
    initLogs({ mount, ws, state: { activePage: 'dashboard', dashboardActiveSubtab: 'logs' } });
    ws.emit('log', { data: live });
    const entries = mount.querySelector('#log-entries');
    assert.equal(entries.children.length, 1, 'live delivery does not wait for backfill');
    finishEvents(response({ entries: [earlier, live] }));
    await settle();
    const status = mount.querySelector('.logs-history-status');
    assert.equal(entries.children.length, 2, 'backfill/live twin appears once');
    assert.match(status.textContent, /incomplete: tools could not be loaded/);
    assert.equal(status.dataset.tone, 'danger');
    assert.equal(status.hidden, false);
    await mount.querySelector('#btn-clear-logs').fire('click');
    assert.equal(entries.children.length, 0);
    assert.equal(status.hidden, false, 'Clear cannot hide missing-history evidence');
    routes.set(logUrl('events'), response({ entries: [earlier, live] }));
    routes.set(logUrl('tools'), response({ entries: [] }));
    ws.emit('open');
    await settle();
    assert.equal(status.hidden, true);
    assert.equal(status.textContent, '');
    assert.equal(entries.children.length, 0, 'existing exact dedupe survives Clear and reconnect');
    assert.equal(calls.length, 8, 'only existing init/reconnect backfill runs');
});

test('Logs reads oldest to newest, says why it is empty, and keeps one quiet tone until a filter narrows it', async (t) => {
    const { mount, routes, ws } = setup(t);
    const live = { type: 'test_live_event', ts: '2026-09-09T12:00:00Z', message: 'live' };
    const earlier = { type: 'test_earlier_event', ts: '2026-09-09T11:59:00Z', message: 'earlier' };
    for (const name of ['events', 'tools', 'progress', 'supervisor']) routes.set(logUrl(name), response({ entries: [] }));
    let finishEvents;
    routes.set(logUrl('events'), () => new Promise((resolve) => { finishEvents = resolve; }));
    const state = { activePage: 'dashboard', dashboardActiveSubtab: 'logs' };
    initLogs({ mount, ws, state });
    const empty = mount.querySelector('[data-log-empty]');
    assert.equal(empty.hidden, true, 'nothing is said about an empty list while its history loads');
    ws.emit('log', { data: live });
    finishEvents(response({ entries: [earlier] }));
    await settle();
    const entries = mount.querySelector('#log-entries');
    // History that arrives after a live event takes its place by its own time (U035).
    assert.deepEqual(entries.children.map((entry) => entry.dataset.logTime), [String(Date.parse(earlier.ts)), String(Date.parse(live.ts))]);
    // The event's one line: Raw at its end as the quiet small capsule, never a row of its own (U027, U032).
    const main = entries.children[0].querySelector('.log-main');
    const raw = main.querySelector('.log-raw-toggle');
    assert.equal(raw.className, 'btn btn-ghost btn-sm log-raw-toggle');
    assert.equal(entries.children[0].querySelector('.log-actions'), null);
    // Raw is one delegated component (logs_raw.js): the browser's click reaches the list, which opens the record.
    entries.dispatchEvent({ type: 'click', target: raw });
    assert.equal(entries.children[0].querySelector('.log-raw').hidden, false);
    assert.equal(raw.getAttribute('aria-expanded'), 'true');
    // Every filter on narrows nothing: no chip takes the chosen tone; one off, the ones still on do (U031).
    const chips = mount.querySelectorAll('.filter-chip');
    assert.ok(chips.length > 1);
    assert.ok(chips.every((chip) => !chip.classList.contains('active') && chip.getAttribute('aria-pressed') === 'true'));
    await chips[0].fire('click');
    assert.equal(chips[0].classList.contains('active'), false);
    assert.equal(chips[0].getAttribute('aria-pressed'), 'false');
    assert.ok(chips.slice(1).every((chip) => chip.classList.contains('active')));
    for (const chip of chips.slice(1)) await chip.fire('click');
    assert.equal(empty.hidden, false);
    assert.match(empty.textContent, /The filters hide every event\./);
    await empty.fire('click', { target: empty.querySelector('[data-log-show-all]') });
    assert.ok(entries.children.every((entry) => !entry.hidden), 'Show all brings every event back');
    assert.equal(empty.hidden, true);
    assert.ok(chips.every((chip) => !chip.classList.contains('active')));
    // Clear says when, and that new events appear (U033).
    await mount.querySelector('#btn-clear-logs').fire('click');
    assert.match(empty.textContent, /^Cleared at \d\d:\d\d\. New events appear here\.$/);
    ws.emit('log', { data: { ...live, ts: '2026-09-09T12:01:00Z' } });
    assert.equal(empty.hidden, true);
});

test('Logs prints an event\'s name once and names a task card by its short id without repeating its lineage', async (t) => {
    const { mount, routes, ws } = setup(t);
    for (const name of ['events', 'tools', 'progress', 'supervisor']) routes.set(logUrl(name), response({ entries: [] }));
    initLogs({ mount, ws, state: { activePage: 'dashboard', dashboardActiveSubtab: 'logs' } });
    await settle();
    ws.emit('log', { data: { type: 'worker_ready', ts: '2026-09-16T12:00:00Z' } });
    const entries = mount.querySelector('#log-entries');
    const plain = entries.children.find((node) => !node.dataset.taskGroup);
    const headline = plain.querySelector('.log-headline').textContent;
    const type = plain.querySelector('.log-type');
    assert.ok(!type || type.textContent !== headline, 'the type is not printed again beside the same headline (U028)');
    ws.emit('log', { data: { type: 'task_heartbeat', task_id: '55d180783850458f', phase: 'running', ts: '2026-09-16T12:00:02Z',
        runtime_sec: 61 } });
    const card = entries.children.find((node) => node.dataset.taskGroup === '55d180783850458f');
    assert.equal(card.querySelector('[data-task-kind]').textContent, 'task 55d18078', 'eight characters, as every row');
    const pills = card.querySelector('[data-task-summary]').querySelectorAll('.log-pill').map((pill) => pill.textContent);
    assert.ok(!pills.some((pill) => pill.startsWith('task=') || pill === 'task'), `no repeated id: ${pills}`);
    // A Timeline's rows exist while it is open (logs.js): opened the way the browser does, its `toggle` reaching the list.
    const details = card.querySelector('details');
    details.open = true;
    entries.dispatchEvent({ type: 'toggle', target: details });
    const timeline = card.querySelector('[data-task-timeline]').querySelectorAll('.log-pill').map((pill) => pill.textContent);
    assert.ok(timeline.includes('task=55d180783850458f'), 'the timeline keeps the full id');
    // An older frame delivered later (history after the live stream) neither dates the card back nor moves it.
    const shownAt = card.querySelector('[data-task-ts]').textContent;
    ws.emit('log', { data: { type: 'task_heartbeat', task_id: '55d180783850458f', phase: 'running', ts: '2026-09-16T11:59:00Z' } });
    assert.equal(card.querySelector('[data-task-ts]').textContent, shownAt);
    assert.equal(card.dataset.logTime, String(Date.parse('2026-09-16T12:00:02Z')));
});

test('Logs catches unavailable and malformed sources while retaining later live events', async (t) => {
    const { mount, routes, ws } = setup(t);
    routes.set(logUrl('events'), response(null));
    routes.set(logUrl('tools'), new Error('offline'));
    routes.set(logUrl('progress'), response({}, 503));
    routes.set(logUrl('supervisor'), { ok: true, json: async () => { throw new Error('not JSON'); } });
    initLogs({ mount, ws, state: { activePage: 'dashboard', dashboardActiveSubtab: 'logs' } });
    await settle();
    assert.match(mount.querySelector('.logs-history-status').textContent, /events, tools, progress, supervisor/);
    ws.emit('log', { data: { type: 'test_after_failure', ts: '2026-09-09T13:00:00Z' } });
    assert.equal(mount.querySelector('#log-entries').children.length, 1);
});

test('an older failed backfill cannot overwrite a newer complete reconnect result', async (t) => {
    const { mount, routes, ws } = setup(t);
    const pending = [];
    routes.set(logUrl('events'), () => new Promise((resolve) => pending.push(resolve)));
    for (const name of ['tools', 'progress', 'supervisor']) routes.set(logUrl(name), response({ entries: [] }));
    initLogs({ mount, ws, state: { activePage: 'dashboard', dashboardActiveSubtab: 'logs' } });
    ws.emit('open');
    pending[1](response({ entries: [{ type: 'new_backfill', ts: '2026-09-09T14:00:00Z' }] }));
    await settle();
    const status = mount.querySelector('.logs-history-status');
    assert.equal(status.hidden, true);
    pending[0](response({}, 503));
    await settle();
    assert.equal(status.hidden, true);
    assert.equal(status.textContent, '');
    assert.equal(mount.querySelector('#log-entries').children.length, 1);
});

// The limits card's rows (names, help, saving, drafts across reads) are checked in costs_limits.test.js.

test('Costs reads the allowance from the consciousness snapshot, /api/state when that fails, and opens its doors', async (t) => {
    const { mount, routes, window } = setup(t);
    const tabs = [];
    initCosts({ mount, state: { activePage: 'dashboard' }, openDashboardTab: async (tab) => { tabs.push(tab); } });
    routes.set('/api/cost-breakdown', response({}, 503));
    routes.set('/api/settings', response({ TOTAL_BUDGET: 200, OUROBOROS_PER_TASK_COST_USD: 50, OUROBOROS_CONSCIOUSNESS_DAILY_USD: 390 }));
    routes.set('/api/consciousness', response({ error: 'busy' }, 503));
    routes.set('/api/state', response({ bg_consciousness_enabled: true, bg_consciousness_state: { status: 'wake_paused',
        live_wake_task_id: 'wake-1', live_wake_held: true, spent_24h_usd: 301.2, daily_usd: 390, allowance_status: 'available',
        live_wake_hold: { cause: 'allowance', ceiling_usd: 2.99 } } }));
    window.dispatchEvent(new CustomEvent('ouro:dashboard-subtab-shown', { detail: { tab: 'costs' } }));
    await settle(); await settle(); await settle();
    assert.equal(mount.querySelector('#setting-OUROBOROS_CONSCIOUSNESS_DAILY_USD').value, '390');
    assert.match(mount.querySelector('#costs-allowance-spend').textContent, /^\$301\.20 of \$390\.00 in the last 24 hours\u00a0· Consciousness ›$/,
        'the separator keeps to the word before it');
    const state = mount.querySelector('#costs-allowance-state');
    // The fixture reads a node's own text before its children's: the order in the page is the browser lane's.
    assert.match(state.textContent, /The consciousness wake-up is paused at the allowance it had left when it started \(\$2\.99\)\./);
    assert.match(state.querySelector('.costs-allowance-note').textContent, /^Resume keeps that limit, so it pauses again/,
        'what follows from the fact is quieter than the fact');
    const door = state.querySelector('[data-open-task]');
    assert.equal(door.dataset.openTask, 'wake-1', 'the held wake\'s card in Main');
    const opened = [];
    window.addEventListener('ouro:open-task', (event) => opened.push(event.detail));
    const card = mount.querySelector('.costs-budget');
    await card.fire('click', { target: door });
    assert.deepEqual(opened, [{ task_id: 'wake-1', chat_id: 1 }]);
    await card.fire('click', { target: mount.querySelector('[data-costs-consciousness]') });
    assert.deepEqual(tabs, ['consciousness']);
});

test('Costs reads the window once more when the room time it gave passes, and only then', async (t) => {
    const { mount, routes, window } = setup(t);
    initCosts({ mount, state: { activePage: 'dashboard' } });
    let reads = 0;
    routes.set('/api/cost-breakdown', response({}, 503));
    routes.set('/api/settings', response({ TOTAL_BUDGET: 200, OUROBOROS_PER_TASK_COST_USD: 50, OUROBOROS_CONSCIOUSNESS_DAILY_USD: 10 }));
    routes.set('/api/consciousness', () => {
        reads += 1;
        return reads === 1
            ? response({ enabled: true, spent_24h_usd: 12.4, daily_usd: 10, allowance_status: 'exhausted',
                allowance_room_after: new Date(Date.now() + 50).toISOString() })
            : response({ enabled: true, spent_24h_usd: 4, daily_usd: 10, allowance_status: 'available' });
    });
    window.dispatchEvent(new CustomEvent('ouro:dashboard-subtab-shown', { detail: { tab: 'costs' } }));
    await settle(); await settle(); await settle();
    assert.equal(reads, 1);
    assert.match(mount.querySelector('#costs-allowance-state').textContent, /Consciousness limit reached/);
    await new Promise((resolve) => setTimeout(resolve, 1300));
    await settle(); await settle(); await settle();
    assert.equal(reads, 2, 'one read when the time it gave passed');
    assert.doesNotMatch(mount.querySelector('#costs-allowance-state').textContent, /limit reached/, 'the old time does not stand');
    await new Promise((resolve) => setTimeout(resolve, 300));
    assert.equal(reads, 2, 'no polling after it');
});

test('Dashboard binds stored, keyboard and programmatic selection to the same named panels', async (t) => {
    const { mount, window } = setup(t);
    const content = new NodeStub(); content.id = 'content'; mount.appendChild(content);
    const changes = [];
    window.addEventListener('ouro:dashboard-subtab-shown', (event) => changes.push(event.detail.tab));
    const state = { dashboardActiveSubtab: 'costs' };
    const dashboard = initDashboard({ state });
    const tabs = dashboard.page.querySelectorAll('[role="tab"]');
    const strip = dashboard.page.querySelector('[role="tablist"]');
    const assertSelected = (name) => {
        assert.equal(state.dashboardActiveSubtab, name);
        assert.deepEqual(tabs.filter((tab) => tab.getAttribute('aria-selected') === 'true').map((tab) => tab.dataset.dashboardTab), [name]);
        assert.deepEqual(tabs.filter((tab) => tab.tabIndex === 0).map((tab) => tab.dataset.dashboardTab), [name]);
        for (const tab of tabs) {
            const panel = dashboard.page.querySelector(`#${tab.getAttribute('aria-controls')}`);
            assert.equal(panel.getAttribute('aria-labelledby'), tab.id);
            assert.equal(panel.hidden, tab.dataset.dashboardTab !== name);
            assert.equal(panel.classList.contains('active'), tab.dataset.dashboardTab === name);
        }
    };
    assertSelected('costs');
    assert.deepEqual(changes, ['costs']);
    const costs = dashboard.page.querySelector('#dashboard-tab-costs');
    await strip.fire('keydown', { target: costs, key: 'ArrowRight', preventDefault() {} });
    assertSelected('updates');
    assert.equal(document.activeElement.id, 'dashboard-tab-updates');
    assert.deepEqual(changes, ['costs', 'updates'], 'one key gesture causes one domain load');
    dashboard.activateTab('activity');
    assertSelected('activity');
    const activity = dashboard.page.querySelector('#dashboard-tab-activity');
    await strip.fire('keydown', { target: activity, key: 'Home', preventDefault() {} });
    assertSelected('logs');
    const count = changes.length;
    dashboard.activateTab('not-a-tab');
    assertSelected('logs');
    assert.equal(changes.length, count);
    dashboard.destroy();
    await strip.fire('click', { target: costs });
    assertSelected('logs');
});

test('Logs files a wake-up group under Consciousness with its label, from the origin fact on its frames', async (t) => {
    const { mount, routes, ws } = setup(t);
    for (const name of ['events', 'tools', 'progress', 'supervisor']) routes.set(logUrl(name), response({ entries: [] }));
    initLogs({ mount, ws, state: { activePage: 'dashboard', dashboardActiveSubtab: 'logs' } });
    ws.emit('log', { data: { type: 'tool_call_finished', tool: 'read_file', is_error: false, task_id: 'wake-1',
        initiator: 'consciousness', ts: '2026-09-16T12:00:00Z' } });
    ws.emit('log', { data: { type: 'tool_call_finished', tool: 'read_file', is_error: false, task_id: 'turn-2',
        ts: '2026-09-16T12:00:01Z' } });
    // A later frame of the same wake without the stamp (a supervisor-rebuilt
    // row) keeps the group's label: the fact is sticky once seen.
    ws.emit('log', { data: { type: 'task_heartbeat', task_id: 'wake-1', phase: 'running', ts: '2026-09-16T12:00:02Z' } });
    await settle();
    const entries = mount.querySelector('#log-entries');
    const card = (id) => entries.children.find((node) => node.dataset.taskGroup === id);
    assert.equal(card('wake-1').dataset.category, 'consciousness');
    assert.equal(card('wake-1').querySelector('[data-task-kind]').textContent, 'Consciousness');
    assert.equal(card('turn-2').dataset.category, 'tasks');
    assert.equal(card('turn-2').querySelector('[data-task-kind]').textContent, 'task turn-2');
});

test('Logs keeps the executor chip quiet and prints its detail sentence as text (#360 A)', async (t) => {
    const { mount, routes, ws } = setup(t);
    for (const name of ['events', 'tools', 'progress', 'supervisor']) routes.set(logUrl(name), response({ entries: [] }));
    initLogs({ mount, ws, state: { activePage: 'dashboard', dashboardActiveSubtab: 'logs' } });
    const evidence = { delegated_runs_started: 2, delegated_runs_settled: 2, delegated_runs_succeeded: 1,
        delegated_runs_failed: 1, subscription_cost_usd: null, harness_models: [] };
    ws.emit('log', { data: { type: 'send_message', is_progress: true, task_id: 'child-x', ts: '2026-09-16T12:00:00Z',
        content: 'done', executor_route: 'codex', execution_evidence: evidence } });
    // An ungrouped row keeps the same pair: a quiet chip and its sentence beside it.
    ws.emit('log', { data: { type: 'executor_note', executor_route: 'claude', ts: '2026-09-16T12:00:01Z' } });
    await settle();
    const entries = mount.querySelector('#log-entries');
    const card = entries.children.find((node) => node.dataset.taskGroup === 'child-x');
    const chip = card.querySelector('[data-task-executor]');
    assert.equal(chip.hidden, false);
    assert.match(chip.textContent, /Codex · 1 of 2 runs failed$/);
    const detail = card.querySelector('[data-task-executor-detail]');
    assert.equal(detail.hidden, false, 'the detail is visible text, not a hover-only title');
    assert.equal(detail.textContent, chip.title);
    assert.match(detail.textContent, /2 runs settled \(1 ok, 1 failed\)/);
    const standalone = entries.children.find((node) => !node.dataset.taskGroup);
    assert.match(standalone.querySelector('.log-executor-chip').textContent, /Claude Code$/);
    assert.match(standalone.querySelector('.log-executor-detail').textContent, /^Dispatched to Claude Code/);
});

test('Activity treats skill provenance as an ordinary paused schedule', async (t) => {
    const { mount, routes, ws } = setup(t);
    emptyActivity(routes);
    routes.set(schedulesUrl, response({ tasks: [
        { id: 'held', name: 'Held skill', enabled: false, status: 'disabled', source: 'skill_manifest',
          skill: 'demo', trigger: { type: 'cron', expr: '* * * * *' } },
    ] }));
    await initActivity({ mount, ws }).refresh();
    const row = section(mount, 'schedules').querySelector('.activity-row');
    assert.equal(row.querySelector('[data-act="schedule-toggle"]').textContent, 'Enable');
    assert.match(row.textContent, /from skill demo/);
    assert.equal(row.querySelector('[data-act="schedule-delete"]').dataset.managed, undefined);
});

test('saved Pause with unreadable tree authority remains unknown in Activity and shared Restart', async (t) => {
    const { confirmAndSendRestart } = await import('../modules/chat_activity.js');
    const { mount, routes, ws } = setup(t);
    emptyActivity(routes);
    routes.set(queueUrl, response({ queue: { running: [], pending: [{ id: 'held', task: {
        id: 'held', root_task_id: 'held', type: 'task', title: 'Saved work', _budget_pause: { reason: 'owner' },
    } }] } }));
    routes.set(backgroundUrl, response({ bg_consciousness_enabled: false,
        active_chat_activities: [{ activity_id: 'held', phase: 'unknown' }], active_chat_activities_complete: false }));
    await initActivity({ mount, ws }).refresh();
    assert.match(section(mount, 'queue').textContent, /Activity unconfirmed/);
    assert.doesNotMatch(section(mount, 'queue').textContent, /paused/);
    assert.equal(section(mount, 'queue').querySelector('[data-act="task-control"]').dataset.budgetPaused, undefined);
    let body = '';
    const result = await confirmAndSendRestart({ ws,
        openConfirmDialog: async (options) => { body = options.body; return false; } });
    assert.equal(result, 'cancelled');
    assert.match(body, /Pause status could not be read/);
});

test('Restart warns about a pausing child of a parked root, once, until it settles', async (t) => {
    const { confirmAndSendRestart } = await import('../modules/chat_activity.js');
    const { routes, ws } = setup(t);
    const root = { activity_id: 'root', phase: 'budget_paused' };
    const child = { activity: 'working', transition: 'pausing', confirmed: true };
    const snapshot = (members, extra = {}) => ({ active_chat_activities: [root],
        active_chat_activities_complete: true, works_complete: true,
        works: [projectedWork('root', 'paused', { members: { states: members } })], ...extra });
    const read = async (body) => {
        routes.set(backgroundUrl, response(body));
        let words;
        assert.equal(await confirmAndSendRestart({ ws,
            openConfirmDialog: async (options) => { words = options.body; return false; } }), 'cancelled');
        return words;
    };
    assert.match(await read(snapshot({ child })), /1 task is still pausing/);
    assert.match(await read(snapshot({ child }, { active_chat_activities: [root,
        { activity_id: 'child', phase: 'budget_pausing' }] })), /1 task is still pausing/);
    assert.match(await read(snapshot({ child, sibling: child })), /2 tasks are still pausing/);
    assert.doesNotMatch(await read(snapshot({ child: { ...child, activity: 'paused', transition: 'none' } })), /still pausing/);
    assert.match(await read(snapshot({ child: { ...child, confirmed: false } })), /Pause status could not be read/);
    assert.match(await read(snapshot({}, { works_complete: false })), /Pause status could not be read/);
});

test('Activity Resume follows the root census while owner Pause is settling', async (t) => {
    const { mount, routes, ws } = setup(t);
    emptyActivity(routes);
    routes.set(queueUrl, response({ queue: { running: [], pending: ['root', 'child'].map((id) => ({ id, task: {
        id, root_task_id: 'root', type: 'task', title: id, _budget_pause: { reason: 'owner' },
    } })) } }));
    const activity = initActivity({ mount, ws });
    for (const phase of ['budget_pausing', 'unknown', undefined, 'budget_paused']) {
        routes.set(backgroundUrl, response({ bg_consciousness_enabled: false,
            active_chat_activities: phase ? [{ activity_id: 'root', phase }] : [],
            active_chat_activities_complete: phase !== undefined,
            works: phase?.startsWith('budget_') ? [projectedWork('root', 'paused', {
                members: { states: { child: { activity: 'paused', confirmed: true } } } })] : [] }));
        await activity.refresh();
        for (const id of ['root', 'child']) {
            const button = section(mount, 'queue').querySelector(`[data-id="${id}"]`);
            assert.equal(button.dataset.budgetPaused, phase?.startsWith('budget_') ? '1' : undefined, `${id}: ${phase}`);
        }
    }
});

test('Activity names saved sleep without claiming a budget pause and preserves owner Pause precedence', async (t) => {
    const { mount, routes, ws } = setup(t);
    emptyActivity(routes);
    const queue = { running: [], pending: [{ id: 'sleeping-root', task: {
        id: 'sleeping-root', root_task_id: 'sleeping-root', type: 'task', title: 'Saved sleep',
        _budget_pause: { reason: 'sleep' },
    } }] };
    routes.set(queueUrl, response({ queue }));
    routes.set(backgroundUrl, response({ bg_consciousness_enabled: false, works: [projectedWork('sleeping-root', 'waiting', {
        because: [{ code: 'sleeping', text: 'Sleeping until it is woken' }] })] }));
    const activity = initActivity({ mount, ws });
    await activity.refresh();
    assert.match(section(mount, 'queue').textContent, /Sleeping/);
    assert.doesNotMatch(section(mount, 'queue').textContent, /paused \(budget\)/);
    assert.equal(section(mount, 'queue').querySelector('[data-act="task-control"]').dataset.budgetPaused, undefined);
    queue.budget_root_fences = [{ root_task_id: 'sleeping-root', status: 'paused', cause: 'owner_pause' }];
    routes.set(backgroundUrl, response({ bg_consciousness_enabled: false,
        works: [projectedWork('sleeping-root', 'paused')],
        active_chat_activities: [{ activity_id: 'sleeping-root', phase: 'budget_paused' }] }));
    await activity.refresh();
    assert.match(section(mount, 'queue').textContent, /Paused/);
    assert.doesNotMatch(section(mount, 'queue').textContent, /Sleeping|budget/);
});

// Census rows as gateway/state.py publishes them: the row's own stored status, axes and checkpoint.
const censusAxes = (status, execution) => ({ schema_version: 1, lifecycle: { status },
    execution: { status: execution, reason_code: execution === 'ok' ? 'final_message' : 'provider_unavailable' },
    artifacts: { status: 'not_applicable' }, objective: { status: 'not_evaluated', source: 'none' },
    review: { status: 'skipped', trigger: 'not_evaluated' } });
const outcomeRow = (activity_id, status, execution, phase, synthesis, extra = {}) => ({ activity_id, kind: 'managed_task',
    phase, status, outcome_axes: censusAxes(status, execution), root_phase_checkpoint: { post_task_synthesis: synthesis }, ...extra });
const chip = (row) => {
    const status = row.querySelector('[data-activity-phase]');
    const second = status?.nextElementSibling;
    return status && { text: status.textContent, phase: status.dataset.phase, label: status.getAttribute('aria-label'),
        secondary: second.hidden ? '' : second.textContent };
};
const rowOf = (mount, id) => section(mount, 'queue').querySelector(`[data-id="${id}"]`).parentElement.parentElement;

test('Activity shows a row\'s own known outcome first and its unfinished finalization beside it', async (t) => {
    const { mount, routes, ws } = setup(t);
    emptyActivity(routes);
    const now = Date.now;
    Date.now = () => 100000;
    t.after(() => { Date.now = now; });
    const running = (id, extra = {}) => ({ id, type: 'task', runtime_sec: 9, task: { id, root_task_id: id, title: `Work ${id}`, ...extra } });
    routes.set(queueUrl, response({ queue: {
        running: ['failed-root', 'done-root', 'warn-root', 'cancelled-root', 'live-root'].map((id) => running(id)).concat(
            running('child-1', { root_task_id: 'failed-root', parent_task_id: 'failed-root', title: 'Child of the failed root' })),
        pending: [{ id: 'queued-root', type: 'task', task: { id: 'queued-root', root_task_id: 'queued-root', title: 'Queued work' } }],
    } }));
    routes.set(backgroundUrl, response({ bg_consciousness_enabled: false, active_chat_activities_complete: true,
        works: [...['failed-root', 'done-root', 'warn-root', 'cancelled-root', 'live-root'].map((id) => projectedWork(id, 'working')),
            projectedWork('queued-root', 'queued'), projectedWork('late-root', 'paused'),
            projectedWork('tree', 'working', { members: { states: { 'child-1': { activity: 'working', confirmed: true } } } })],
        active_chat_activities: [
            outcomeRow('failed-root', 'failed', 'infra_failed', 'finalizing', 'running'),
            outcomeRow('done-root', 'completed', 'ok', 'finalizing', 'pending_once'),
            outcomeRow('warn-root', 'completed', 'degraded', 'working', 'completed'),
            outcomeRow('cancelled-root', 'cancelled', 'cancelled', 'working', ''),
            { activity_id: 'live-root', kind: 'managed_task', phase: 'working', status: 'running', outcome_axes: censusAxes('running', 'ok') },
            { activity_id: 'queued-root', kind: 'managed_task', phase: 'queued' },
            outcomeRow('answered-direct', 'completed', 'ok', 'finalizing', 'running', { kind: 'direct_chat', started_at: 90 }),
            outcomeRow('late-root', 'completed', 'ok', 'budget_paused', 'paused', { pause_cause: 'owner' }),
            { activity_id: 'thinking-direct', kind: 'direct_chat', phase: 'thinking', started_at: 95 },
        ] }));
    await initActivity({ mount, ws }).refresh();
    const queue = section(mount, 'queue');
    assert.equal(queue.querySelectorAll('.activity-row').length, 10, 'every occupied row stays listed');
    assert.deepEqual(chip(rowOf(mount, 'failed-root')), { text: 'Failed', phase: 'error',
        label: 'Task status: Failed, Finalizing…', secondary: 'Finalizing…' });
    assert.doesNotMatch(rowOf(mount, 'failed-root').textContent, /running/, 'the outcome replaces the occupancy word');
    assert.match(rowOf(mount, 'failed-root').textContent, /· task · 9s/);
    assert.deepEqual(chip(rowOf(mount, 'done-root')).text + chip(rowOf(mount, 'done-root')).secondary, 'DoneFinalizing…');
    assert.deepEqual(chip(rowOf(mount, 'warn-root')), { text: 'Done with warnings', phase: 'warn',
        label: 'Task status: Done with warnings', secondary: '' });
    assert.equal(chip(rowOf(mount, 'cancelled-root')).text, 'Stopped');
    // No known outcome: the queue's own words, exactly as before.
    for (const [id, word] of [['live-root', /Working · task · 9s/], ['queued-root', /Queued work[\s\S]*Queued · task/]]) {
        assert.equal(chip(rowOf(mount, id)), null, id);
        assert.match(rowOf(mount, id).textContent, word);
    }
    // A child never borrows its root's result, even while that root's census row is Failed.
    assert.equal(chip(rowOf(mount, 'child-1')), null);
    assert.match(rowOf(mount, 'child-1').textContent, /Child of the failed root[\s\S]*Working/);
    assert.doesNotMatch(rowOf(mount, 'child-1').textContent, /Failed/);
    // Census-only rows: a known answer with late work, a late Pause beside its Done, a plain turn.
    assert.deepEqual(chip(rowOf(mount, 'answered-direct')).text + '|' + chip(rowOf(mount, 'answered-direct')).secondary,
        'Done|Finalizing…');
    assert.match(rowOf(mount, 'answered-direct').textContent, /Direct turn[\s\S]*· 10s/);
    // The census's `owner` cause also covers Ouroboros's own Pause: without a work it never says who paused.
    assert.deepEqual(chip(rowOf(mount, 'late-root')), { text: 'Done', phase: 'done',
        label: 'Task status: Done, Paused', secondary: 'Paused' });
    assert.equal(chip(rowOf(mount, 'thinking-direct')), null);
    assert.equal(rowOf(mount, 'thinking-direct').querySelector('[data-i18n-skip]').textContent, 'Thinking');
    assert.match(rowOf(mount, 'thinking-direct').textContent, /· 5s/);
    // Known outcomes keep the shared chip in the cards' tones: Done with warnings green, Stopped neutral, Failed red.
    const look = (id) => {
        const status = rowOf(mount, id).querySelector('[data-activity-phase]');
        const mark = status.firstElementChild;
        return [status.dataset.chipTone, `${mark.dataset.shape}/${mark.dataset.hue}`];
    };
    assert.deepEqual(look('failed-root'), ['failed', 'dot/failed']);
    assert.deepEqual(look('warn-root'), ['ok', 'dot/ok']);
    assert.deepEqual(look('cancelled-root'), ['neutral', 'dot/neutral']);
    // Controls follow occupancy and the existing pause facts, never the outcome.
    const controls = Object.fromEntries(queue.querySelectorAll('[data-act="task-control"]').map((button) =>
        [button.dataset.id, [button.dataset.root, button.dataset.budgetPaused]]));
    assert.deepEqual(controls, {
        'failed-root': ['1', undefined], 'done-root': ['1', undefined], 'warn-root': ['1', undefined],
        'cancelled-root': ['1', undefined], 'live-root': ['1', undefined], 'child-1': [undefined, undefined],
        'queued-root': ['1', undefined], 'answered-direct': ['1', undefined], 'late-root': ['1', '1'],
        'thinking-direct': ['1', undefined],
    });
    assert.ok(queue.querySelectorAll('[data-activity-phase]').every((node) => node.closest('[data-i18n-skip]')),
        'chip words come from the translation seam, so the overlay leaves them alone');
});

test('Activity separates the pause word from confirmed work context and offers only declared controls', async (t) => {
    const { mount, routes, ws } = setup(t);
    emptyActivity(routes);
    const paused = (id, extra = {}) => outcomeRow(id, 'completed', 'ok', 'budget_paused', 'paused', { pause_cause: 'owner', ...extra });
    const pausedWork = (id, code, text, extra = {}) => ({ work_id: id, room: 1, activity: 'paused', confirmed: true,
        episode: { task_id: id, live: true }, because: [{ code, text }], tails: [], controls: [], ...extra });
    routes.set(backgroundUrl, response({ bg_consciousness_enabled: false, active_chat_activities_complete: true,
        active_chat_activities: [...['by-owner', 'by-mind', 'unnamed', 'unconfirmed'].map((id) => paused(id)),
            paused('reviewed', { finishing_reviews: true })],
        works: [pausedWork('by-owner', 'paused_by_owner', 'You paused it'),
            pausedWork('by-mind', 'paused_by_mind', 'Ouroboros: pause before the update'),
            pausedWork('unconfirmed', 'paused_by_owner', 'You paused it', { confirmed: false })], works_complete: true }));
    await initActivity({ mount, ws }).refresh();
    const rows = section(mount, 'queue').querySelectorAll('.activity-row');
    assert.equal(rows.length, 5, 'rows without controls remain visible');
    for (const text of ['You paused it', 'Ouroboros: pause before the update']) {
        const matches = rows.filter((row) => row.textContent.includes(text));
        assert.equal(matches.length, 1, text);
        const [row] = matches;
        assert.equal(chip(row).secondary, 'Paused', 'the chip keeps only the state word');
        assert.equal(row.querySelector('[data-act="task-control"]'), null, 'empty controls invent no action');
    }
    for (const id of ['unnamed', 'unconfirmed']) {
        const row = rowOf(mount, id);
        assert.equal(chip(row).secondary, 'Paused', id);
        assert.doesNotMatch(row.textContent, /You paused it|Ouroboros: pause before the update/);
    }
    assert.equal(chip(rowOf(mount, 'reviewed')).secondary, 'Paused · Review work finishing');
});

test('Activity keeps pause words for members and paints a host hold beside a known outcome as text', async (t) => {
    const { mount, routes, ws } = setup(t);
    emptyActivity(routes);
    const member = (id) => ({ id, task: { id, root_task_id: 'paused-root', parent_task_id: id === 'paused-root' ? '' : 'paused-root',
        type: 'task', title: id, _budget_pause: { reason: 'owner' } } });
    routes.set(queueUrl, response({ queue: { running: [], pending: [member('paused-root'), member('member-1')],
        budget_root_fences: [{ root_task_id: 'paused-root', status: 'paused', cause: 'owner_pause' }] } }));
    const hold = { label: '<img src=x onerror="globalThis.pwned=1">', detail: 'binding unavailable' };
    const activities = [outcomeRow('paused-root', 'completed', 'ok', 'budget_paused', 'paused', { pause_cause: 'owner' }),
        outcomeRow('held-root', 'failed', 'infra_failed', 'finalizing', 'running', { project_admission_hold: hold })];
    routes.set(backgroundUrl, response({ bg_consciousness_enabled: false, active_chat_activities: activities,
        active_chat_activities_complete: false, works: [projectedWork('paused-root', 'paused', {
            members: { states: { 'member-1': { activity: 'paused', confirmed: true } } } })] }));
    const activity = initActivity({ mount, ws });
    await activity.refresh();
    // The root's own row knows its answer; its member keeps the tree's pause word and Resume.
    assert.equal(chip(rowOf(mount, 'paused-root')).text, 'Done');
    assert.equal(chip(rowOf(mount, 'member-1')), null);
    assert.match(rowOf(mount, 'member-1').textContent, /Paused/);
    for (const id of ['paused-root', 'member-1']) {
        assert.equal(section(mount, 'queue').querySelector(`[data-id="${id}"]`).dataset.budgetPaused, '1', id);
    }
    // A partial census still describes its own rows; the host's hold text never becomes markup.
    assert.deepEqual(chip(rowOf(mount, 'held-root')).secondary, hold.label);
    assert.equal(section(mount, 'queue').querySelectorAll('img').length, 0);
    assert.doesNotMatch(section(mount, 'queue').textContent, /Nothing running/);
    // Without a census the rows stay visible, but their stop is unconfirmed.
    routes.set(backgroundUrl, response({}, 503));
    await activity.refresh();
    assert.equal(section(mount, 'queue').querySelectorAll('[data-activity-phase]').length, 0);
    assert.match(rowOf(mount, 'paused-root').textContent, /Activity unconfirmed/);
    assert.equal(section(mount, 'queue').querySelectorAll('.activity-row').length, 2);
});

test('Activity keeps a queue row\'s own wait beside its known outcome and shows no model-call telemetry', async (t) => {
    const { mount, routes, ws } = setup(t);
    emptyActivity(routes);
    const hold = { label: 'Waiting for Project verification', detail: 'binding unavailable' };
    const root = (id, title, extra = {}) => ({ id, type: 'task', task: { id, root_task_id: id, title, ...extra } });
    routes.set(queueUrl, response({ queue: {
        running: [
            { ...root('sleep-root', 'Sleeping answered work'), owner_wait: { state: 'waiting', reason: 'sleep', sleep: { mode: 'warm' } } },
            root('live-model', 'Finalizing with a model'),
            root('held-queue', 'Held work', { project_admission_hold: hold }),
        ],
        pending: [
            root('restart-root', 'Saved before Quit', { _budget_pause_hold: { reason: 'saved_work_hold' } }),
            root('pending-root', 'Requeued'),
        ],
    } }));
    routes.set(backgroundUrl, response({ bg_consciousness_enabled: false, active_chat_activities_complete: true,
        works: [projectedWork('sleep-root', 'waiting', { because: [{ code: 'sleeping', text: 'Sleeping until it is woken' }] }),
            projectedWork('live-model'), projectedWork('held-queue'), projectedWork('restart-root', 'paused', {
                because: [{ code: 'saved_on_stop', text: 'Saved when the app stopped' }] }), projectedWork('pending-root', 'queued')],
        active_chat_activities: [
            outcomeRow('sleep-root', 'completed', 'ok', 'finalizing', 'running'),
            outcomeRow('live-model', 'completed', 'ok', 'working', 'running',
                { model_activity: { phase: 'awaiting_response', task_attempt: 1 } }),
            outcomeRow('held-queue', 'failed', 'infra_failed', 'finalizing', 'running', { project_admission_hold: hold }),
            outcomeRow('restart-root', 'completed', 'ok', 'budget_paused', 'completed', { pause_cause: 'restart' }),
            outcomeRow('pending-root', 'failed', 'infra_failed', 'queued', 'completed'),
        ] }));
    await initActivity({ mount, ws }).refresh();
    const pair = (id) => [chip(rowOf(mount, id)).text, chip(rowOf(mount, id)).secondary];
    // A queue wait outranks the census's merely active phase; the outcome still leads.
    assert.deepEqual(pair('sleep-root'), ['Done', 'Waiting']);
    assert.deepEqual(pair('restart-root'), ['Done', 'Paused']);
    assert.deepEqual(pair('pending-root'), ['Failed', 'Queued']);
    assert.equal(chip(rowOf(mount, 'sleep-root')).label, 'Task status: Done, Waiting');
    // Activity carries no model-call telemetry: finalization stays a generic fact.
    assert.deepEqual(pair('live-model'), ['Done', 'Finalizing…']);
    assert.doesNotMatch(section(mount, 'queue').textContent, /request|response/i);
    // The queue line names its own hold once; the chip does not repeat it.
    assert.deepEqual(pair('held-queue'), ['Failed', 'Finalizing…']);
    assert.equal(rowOf(mount, 'held-queue').textContent.split(hold.label).length, 2);
    // Resume stays exactly where the queue facts put it.
    const paused = Object.fromEntries(section(mount, 'queue').querySelectorAll('[data-act="task-control"]')
        .map((button) => [button.dataset.id, button.dataset.budgetPaused]));
    assert.deepEqual(paused, { 'sleep-root': undefined, 'live-model': undefined, 'held-queue': undefined,
        'restart-root': '1', 'pending-root': undefined });
});

// ---------------------------------------------------------------------------
// Dashboard -> Storage: the real controller over the fixture's fetch, window and document.
// ---------------------------------------------------------------------------

const storageUrl = '/api/storage';
const measureUrl = '/api/storage/measure';
const cancelUrl = '/api/storage/measure/cancel';
const storageVolume = { free_bytes: 170e9, total_bytes: 4e12, observed_at: '2026-10-05T10:00:00Z' };
const storageBody = (overrides = {}) => ({ count: null, counting: null, error: null, volume: storageVolume, retention_days: 7, ...overrides });
const storageCounting = () => ({ started_at: '2026-10-05T10:00:00Z', place: 'model_calls', current: 'blobs', places_done: 0, places_total: 12,
    bytes: 1000, files: 1, done: [] });
const measuredBody = () => storageBody({ count: {
    started_at: '2026-10-05T10:00:00Z', finished_at: '2026-10-05T10:00:01Z', duration_seconds: 1, io_priority: 'utility', basis: 'allocated',
    bytes: 1400000, logical_bytes: 1400000, files: 3, unreadable: 0, volume: storageVolume,
    places: [{ id: 'task_results', group: 'work', roots: ['/data/task_results'], present: true, same_volume: true, bytes: 1400000,
        logical_bytes: 1400000, files: 3, unreadable: 0, parts: [], rest: { count: 0, bytes: 0 } }],
} });
const deferred = () => {
    const handle = {};
    handle.promise = new Promise((resolve, reject) => Object.assign(handle, { resolve, reject }));
    return handle;
};
const storageSummary = (mount) => `${mount.querySelector('[data-storage-summary]').textContent} ${mount.querySelector('[data-storage-status]').textContent}`;

/** Timers by hand: the poll is a scheduled callback the test can see, fire or see cleared. */
function manualTimers(t) {
    const pending = new Map();
    let next = 1;
    const realClear = globalThis.clearTimeout;
    globalThis.setTimeout = (fn, ms) => { pending.set(next, { fn, ms }); return next++; };
    globalThis.clearTimeout = (id) => { pending.delete(id); };
    t.after(() => { globalThis.clearTimeout = realClear; });
    return {
        polls: () => [...pending.values()].filter((timer) => timer.ms === 1000).length,
        async fire() {
            const due = [...pending.values()];
            pending.clear();
            for (const timer of due) await timer.fn();
        },
    };
}

function openStorage(t) {
    const fixture = setup(t);
    const timers = manualTimers(t);
    // The retention row's own settings read is dashboard_settings.test.js's; these flows count Storage's reads.
    const fetchStorage = globalThis.fetch;
    globalThis.fetch = async (url, init) => (url === '/api/settings' ? response({ OUROBOROS_GC_RETENTION_DAYS: 7 }) : fetchStorage(url, init));
    const state = { activePage: 'dashboard', dashboardActiveSubtab: 'storage' };
    const show = (tab) => {
        state.dashboardActiveSubtab = tab;
        fixture.window.dispatchEvent({ type: 'ouro:dashboard-subtab-shown', detail: { tab } });
    };
    return { ...fixture, timers, state, show, start: () => initStorage({ state, mount: fixture.mount }) };
}

test('Storage opens with one passive read; only the button measures, then polls while visible', async (t) => {
    const { mount, routes, calls, timers, start } = openStorage(t);
    routes.set(storageUrl, response(storageBody()));
    start();
    await settle();
    assert.deepEqual(calls, [storageUrl]);
    assert.match(storageSummary(mount), /Not measured yet/);
    assert.match(storageSummary(mount), /This disk: .* free of/, 'the free space is read before any count');
    assert.equal(timers.polls(), 0, 'nothing runs, so nothing is polled');
    assert.equal(mount.querySelector('[data-storage-places]').textContent, '', 'no list before a finished count');

    routes.set(measureUrl, (_url, init) => {
        assert.equal(init.method, 'POST');
        return response(storageBody({ counting: storageCounting(), started: true }), 202);
    });
    await mount.querySelector('[data-storage-action="measure"]').fire('click');
    await settle();
    assert.deepEqual(calls, [storageUrl, measureUrl]);
    assert.equal(timers.polls(), 1);
    assert.equal(mount.querySelector('[data-storage-action="stop"]').hidden, false, 'Stop is offered while a count runs');
    routes.set(storageUrl, response(measuredBody()));
    await timers.fire();
    await settle();
    assert.deepEqual(calls, [storageUrl, measureUrl, storageUrl]);
    assert.match(storageSummary(mount), /Space used by Ouroboros\s*1\.4 MB/);
    assert.equal(timers.polls(), 0, 'a finished count is not polled');
    assert.equal(mount.querySelector('[data-storage-panel]').hidden, false);
    assert.equal(mount.querySelector('[data-storage-action="stop"]').hidden, true);
});

test('Storage holds one count busy from the click to the finished walk, and names the count it holds on later reads', async (t) => {
    const { mount, routes, calls, timers, start } = openStorage(t);
    routes.set(storageUrl, response(storageBody()));
    start();
    await settle();
    const button = mount.querySelector('[data-storage-action="measure"]');
    const posted = deferred();
    routes.set(measureUrl, () => posted.promise);
    const click = button.fire('click');
    await settle();
    // The one button stays focusable (aria-disabled, never disabled) and a second click sends nothing.
    assert.equal(button.textContent, 'Measuring…');
    assert.equal(button.getAttribute('aria-disabled'), 'true');
    assert.equal(button.getAttribute('aria-busy'), 'true');
    assert.equal(button.disabled, false);
    await button.fire('click');
    assert.deepEqual(calls, [storageUrl, measureUrl]);

    posted.resolve(response(storageBody({ counting: storageCounting(), started: true }), 202));
    await click;
    await settle();
    assert.equal(button.getAttribute('aria-disabled'), 'true', 'a running count keeps it busy');
    await button.fire('click');
    assert.deepEqual(calls, [storageUrl, measureUrl]);
    assert.equal(timers.polls(), 1);

    routes.set(storageUrl, response(measuredBody()));
    await timers.fire();
    await settle();
    assert.equal(button.textContent, 'Refresh sizes');
    assert.equal(button.getAttribute('aria-disabled'), 'false');
    assert.equal(button.hasAttribute('aria-busy'), false);

    // A recount: the held count is named, and an answer without `count` keeps it on screen.
    routes.set(measureUrl, response({ ...measuredBody(), counting: storageCounting(), started: true }, 202));
    await button.fire('click');
    await settle();
    assert.match(storageSummary(mount), /1\.4 MB[\s\S]*Measuring/);
    const held = `${storageUrl}?count=${encodeURIComponent('2026-10-05T10:00:01Z')}`;
    routes.set(held, response({ counting: null, error: null, volume: storageVolume, retention_days: 7 }));
    await timers.fire();
    await settle();
    assert.equal(calls.at(-1), held);
    assert.match(storageSummary(mount), /1\.4 MB/);
    assert.doesNotMatch(storageSummary(mount), /Measuring/);
});

test('Storage says when a first count cannot be read, keeps polling and does not measure twice', async (t) => {
    const { mount, routes, calls, timers, start } = openStorage(t);
    routes.set(storageUrl, response(storageBody()));
    start();
    await settle();
    const button = mount.querySelector('[data-storage-action="measure"]');
    routes.set(measureUrl, response(storageBody({ counting: storageCounting(), started: true }), 202));
    await button.fire('click');
    await settle();
    assert.match(storageSummary(mount), /Measuring Model call records · blobs/);

    routes.set(storageUrl, response({ error: 'synthetic storage read refusal' }, 503));
    await timers.fire();
    await settle();
    assert.match(storageSummary(mount), /Measuring Model call records/);
    assert.match(storageSummary(mount), /Storage status could not be read\. Showing the last answer\./);
    assert.equal(timers.polls(), 1, 'the passive read is retried');
    assert.equal(button.getAttribute('aria-disabled'), 'true');
    await button.fire('click');
    await settle();
    assert.deepEqual(calls, [storageUrl, measureUrl, storageUrl], 'no second count');

    routes.set(storageUrl, response(measuredBody()));
    await timers.fire();
    await settle();
    assert.match(storageSummary(mount), /1\.4 MB/);
    assert.doesNotMatch(storageSummary(mount), /could not be read/);
    assert.equal(timers.polls(), 0);
    assert.equal(button.getAttribute('aria-disabled'), 'false');
});

test('Storage Stop asks the server to kill the count and keeps the saved count', async (t) => {
    const { mount, routes, calls, start } = openStorage(t);
    routes.set(storageUrl, response({ ...measuredBody(), counting: storageCounting() }));
    start();
    await settle();
    const stop = mount.querySelector('[data-storage-action="stop"]');
    assert.equal(stop.hidden, false);
    routes.set(cancelUrl, (_url, init) => {
        assert.equal(init.method, 'POST');
        return response({ ...measuredBody(), error: { at: 'x', message: 'Stopped before it finished.', stopped: true } }, 202);
    });
    await stop.fire('click');
    await settle();
    assert.deepEqual(calls, [storageUrl, cancelUrl]);
    assert.match(storageSummary(mount), /1\.4 MB[\s\S]*Stopped before it finished\. Showing the count from/);
    assert.equal(stop.hidden, true);
    assert.equal(mount.querySelector('[data-storage-action="measure"]').textContent, 'Refresh sizes');
});

test('Storage drops a read answered after its tab was left, and does not poll for it', async (t) => {
    const { mount, routes, calls, timers, show, start } = openStorage(t);
    const late = deferred();
    let signal = null;
    routes.set(storageUrl, (_url, init) => { signal = init.signal; return late.promise; });
    start();
    show('logs');
    assert.equal(signal.aborted, true, 'leaving the tab aborts its read');
    late.resolve(response(storageBody({ counting: storageCounting() })));  // a server that answered anyway
    await settle();
    assert.match(storageSummary(mount), /Loading/);
    assert.equal(timers.polls(), 0);
    assert.deepEqual(calls, [storageUrl]);
});

for (const outcome of ['answers', 'fails']) {
    test(`Storage reads nothing while hidden when a pending measure ${outcome} after the tab was left`, async (t) => {
        const { mount, routes, calls, timers, show, start, toasts } = openStorage(t);
        routes.set(storageUrl, response(storageBody()));
        start();
        await settle();
        const posted = deferred();
        routes.set(measureUrl, () => posted.promise);
        const click = mount.querySelector('[data-storage-action="measure"]').fire('click');
        await settle();
        show('logs');
        if (outcome === 'answers') posted.resolve(response(storageBody({ counting: storageCounting(), started: true }), 202));
        else posted.resolve(response({ error: 'The count could not be started.' }, 500));
        await click;
        await settle();
        assert.deepEqual(calls, [storageUrl, measureUrl], 'no read while the tab is hidden');
        assert.equal(timers.polls(), 0);
        if (outcome === 'fails') assert.deepEqual(toasts(), ['The count could not be started.']);

        routes.set(storageUrl, response(measuredBody()));
        show('storage');
        await settle();
        assert.deepEqual(calls, [storageUrl, measureUrl, storageUrl], 'showing the tab again reads once');
        assert.match(storageSummary(mount), /1\.4 MB/);
    });
}

test('Storage stops polling while the page is hidden and resumes when it returns', async (t) => {
    const { routes, calls, timers, document, start } = openStorage(t);
    routes.set(storageUrl, response(storageBody({ counting: storageCounting() })));
    start();
    await settle();
    assert.equal(timers.polls(), 1);
    document.hidden = true;
    document.dispatchEvent({ type: 'visibilitychange' });
    assert.equal(timers.polls(), 0, 'a hidden page keeps no poll');
    document.hidden = false;
    document.dispatchEvent({ type: 'visibilitychange' });
    await settle();
    assert.deepEqual(calls, [storageUrl, storageUrl]);
    assert.equal(timers.polls(), 1);
});

test('Storage destroy aborts its read and ignores late answers, tab events and clicks', async (t) => {
    const { mount, routes, calls, timers, show, start } = openStorage(t);
    const late = deferred();
    let signal = null;
    routes.set(storageUrl, (_url, init) => { signal = init.signal; return late.promise; });
    const storage = start();
    storage.destroy();
    assert.equal(signal.aborted, true);
    late.resolve(response(measuredBody()));
    await settle();
    assert.match(storageSummary(mount), /Loading/);
    show('storage');
    await mount.querySelector('[data-storage-action="measure"]').fire('click');
    await settle();
    assert.deepEqual(calls, [storageUrl]);
    assert.equal(timers.polls(), 0);
});

test('Activity keeps a held Continue blocker readable without offering Pause or Resume (#1721)', async (t) => {
    const { mount, routes, ws } = setup(t);
    emptyActivity(routes);
    routes.set(queueUrl, response({ queue: { running: [], pending: [{ id: 'pred-1-s1', task: {
        id: 'pred-1-s1', root_task_id: 'pred-1-s1', title: 'Continue Friday report',
        _budget_pause_hold: { reason: 'continuation_writer_unsettled', selected: false, dispatchable: false,
            detail: "the interrupted task's own work is not proven settled; this Continue waits",
            predecessor_task_id: 'pred-1', blockers: [{ kind: 'delegated_run', task_id: 'pred-1', run_id: '',
                invocation_id: 'unbound', state: 'stop_unknown', stop_outcome: 'pending_invocation_unbound',
                source: 'delegated_subagent' }] } } }] } }));
    routes.set(backgroundUrl, response({ bg_consciousness_enabled: false, works: [projectedWork('pred-1-s1', 'waiting', {
        because: [{ code: 'earlier_work_unconfirmed', text: "Continue starts when the earlier attempt's work is confirmed finished" }],
        controls: [{ action: 'stop', target: 'pred-1-s1' }] })] }));
    const activity = initActivity({ mount, ws });
    await activity.refresh();
    const row = section(mount, 'queue').querySelector('.activity-row');
    assert.match(row.querySelector('.activity-sub').textContent,
        /^Waiting · Continue starts when the earlier attempt&#39;s work is confirmed finished · a delegated start has an unknown outcome \(no run was recorded\)$/);
    assert.ok(row.classList.contains('activity-held'), 'its cause wraps instead of ending in an ellipsis');
    assert.doesNotMatch(row.textContent, /paused \(budget\)/);
    const control = row.querySelector('[data-act="task-control"]');
    assert.equal(control.getAttribute('data-resume'), null, 'the wait releases itself');
    assert.equal(control.getAttribute('data-budget-paused'), null, 'a writer wait is not a Pause');
    // An older queue reply cannot keep displaying a blocker after the work moved on.
    routes.set(backgroundUrl, response({ bg_consciousness_enabled: false,
        works: [projectedWork('pred-1-s1', 'queued')] }));
    await activity.refresh();
    assert.equal(section(mount, 'queue').querySelector('.activity-sub').textContent, 'Queued');
    activity.destroy();
});
