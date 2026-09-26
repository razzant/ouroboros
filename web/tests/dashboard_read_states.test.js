import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { initActivity, scheduleDeleteDialog } from '../modules/activity.js';
import { initLogs } from '../modules/logs.js';
import { initCosts } from '../modules/costs.js';
import { initDashboard } from '../modules/dashboard.js';

// DOM bookkeeping only: the production initializers, HTTP client, status helper,
// stream subscription and log deduplication all run unchanged. Layout/AT behavior
// remains a browser check; this fixture does not pretend to measure either.
class NodeStub {
    constructor(tag = 'div') {
        this.tagName = tag;
        this.children = [];
        this.attrs = {};
        this.dataset = {};
        this.events = new Map();
        this.scrollHeight = this.scrollTop = this.clientHeight = 0;
        this.hidden = this.disabled = false;
        this._text = '';
        this.classList = {
            contains: (name) => this.className.split(/\s+/).includes(name),
            toggle: (name, force) => {
                const names = new Set(this.className.split(/\s+/).filter(Boolean));
                const enabled = force ?? !names.has(name);
                if (enabled) names.add(name); else names.delete(name);
                this.className = [...names].join(' ');
                return enabled;
            },
            add: (name) => this.classList.toggle(name, true),
        };
    }
    get id() { return this.attrs.id || ''; }
    set id(value) { this.attrs.id = value; }
    get className() { return this.attrs.class || ''; }
    set className(value) { this.attrs.class = value; }
    setAttribute(key, value) {
        this.attrs[key] = String(value);
        if (key.startsWith('data-')) this.dataset[key.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = String(value);
        if (key === 'disabled' || key === 'hidden') this[key] = true;
    }
    getAttribute(key) { return this.attrs[key] ?? null; }
    hasAttribute(key) { return key in this.attrs; }
    removeAttribute(key) { delete this.attrs[key]; }
    get firstElementChild() { return this.children[0]; }
    get nextElementSibling() { return this.parentElement?.children[this.parentElement.children.indexOf(this) + 1]; }
    get textContent() { return this._text + this.children.map((child) => child.textContent).join(''); }
    set textContent(text) { this._text = String(text); this.children = []; }
    set innerHTML(html) {
        this.textContent = '';
        const stack = [this];
        for (const token of String(html).match(/<[^>]+>|[^<]+/g) || []) {
            if (token.startsWith('</')) { stack.pop(); continue; }
            if (!token.startsWith('<')) { stack.at(-1)._text += token; continue; }
            const tag = token.match(/^<([\w-]+)/)?.[1];
            if (!tag) continue;
            const node = new NodeStub(tag);
            for (const attr of token.slice(tag.length + 1, -1).matchAll(/([\w-]+)(?:="([^"]*)")?/g)) {
                node.setAttribute(attr[1], attr[2] ?? '');
            }
            stack.at(-1).appendChild(node);
            if (!['input', 'br', 'hr', 'img', 'meta', 'link'].includes(tag)) stack.push(node);
        }
    }
    appendChild(node) { node.remove(); node.parentElement = this; this.children.push(node); return node; }
    append(...nodes) { nodes.forEach((node) => this.appendChild(node)); }
    remove() {
        if (this.parentElement) this.parentElement.children.splice(this.parentElement.children.indexOf(this), 1);
        this.parentElement = null;
    }
    contains(node) { return this === node || this.children.some((child) => child.contains(node)); }
    matches(selector) {
        if (selector.includes(',')) return selector.split(',').some((part) => this.matches(part.trim()));
        if (selector.includes('][')) return (selector.match(/\[[^\]]+\]/g) || []).every((part) => this.matches(part));
        if (selector.startsWith('#')) return this.id === selector.slice(1);
        if (selector.startsWith('.')) return this.classList.contains(selector.slice(1));
        const attr = selector.match(/^\[([\w-]+)(?:="([^"]*)")?\]$/);
        return attr ? this.hasAttribute(attr[1]) && (attr[2] === undefined || this.getAttribute(attr[1]) === attr[2]) : this.tagName === selector;
    }
    querySelectorAll(selector) {
        return this.children.flatMap((child) => [...(child.matches(selector) ? [child] : []), ...child.querySelectorAll(selector)]);
    }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
    closest(selector) { return this.matches(selector) ? this : this.parentElement?.closest(selector) || null; }
    focus() { globalThis.document.activeElement = this; }
    addEventListener(type, fn) {
        if (!this.events.has(type)) this.events.set(type, []);
        this.events.get(type).push(fn);
    }
    async fire(type, event = {}) { for (const fn of this.events.get(type) || []) await fn({ currentTarget: this, ...event }); }
    dispatchEvent(event) { for (const fn of this.events.get(event.type) || []) fn(event); }
    removeEventListener(type, fn) { this.events.set(type, (this.events.get(type) || []).filter((listener) => listener !== fn)); }
}

function setup(t) {
    const mount = new NodeStub();
    const body = new NodeStub('body');
    const document = {
        body,
        createElement: (tag) => new NodeStub(tag),
        getElementById: (id) => mount.querySelector(`#${id}`) || body.querySelector(`#${id}`),
    };
    const window = new NodeStub();
    const routes = new Map();
    const calls = [];
    const wsHandlers = new Map();
    const ws = {
        on: (type, fn) => { wsHandlers.set(type, fn); },
        send() {},
        emit: (type, data = {}) => wsHandlers.get(type)?.(data),
    };
    const previous = Object.fromEntries(['document', 'window', 'fetch', 'requestAnimationFrame', 'setTimeout'].map((key) => [key, globalThis[key]]));
    // Timers still fire during the test; the handles are tracked so a toast's
    // dismissal timeout cannot keep the runner's event loop alive after it.
    const timers = new Set();
    const realSetTimeout = globalThis.setTimeout;
    Object.assign(globalThis, {
        document, window, requestAnimationFrame: (fn) => fn(),
        setTimeout: (fn, ms, ...rest) => {
            const handle = realSetTimeout(fn, ms, ...rest);
            timers.add(handle);
            return handle;
        },
        fetch: async (url, init) => {
            calls.push(url);
            if (!routes.has(url)) throw new Error(`Unexpected request ${url}`);
            const value = routes.get(url);
            if (typeof value === 'function') return value(url, init);
            if (value instanceof Error) throw value;
            return value;
        },
    });
    t.after(() => {
        timers.forEach((handle) => clearTimeout(handle));
        Object.assign(globalThis, previous);
    });
    const toasts = () => body.querySelectorAll('.toast').map((node) => node.textContent);
    return { mount, window, ws, routes, calls, toasts };
}

const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
const queueUrl = '/api/tasks?queue_only=1';
const backgroundUrl = '/api/state';
const schedulesUrl = '/api/schedules';
const logUrl = (name) => `/api/logs/${name}?limit=150`;
const settle = () => new Promise((resolve) => setImmediate(resolve));
const section = (mount, name) => mount.querySelector(`[data-activity-section="${name}"]`);
const message = (node) => node.querySelector('.ui-status').textContent;

function emptyActivity(routes) {
    routes.set(queueUrl, response({ queue: { running: [], pending: [] } }));
    routes.set(backgroundUrl, response({ bg_consciousness_enabled: false, active_chat_activities: [], active_chat_activities_complete: true }));
    routes.set(schedulesUrl, response({ tasks: [] }));
}

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
    { id: 'active', name: 'Active schedule', enabled: true, status: 'active', trigger: { type: 'cron', expr: '* * * * *' } },
    { id: 'disabled', name: 'Disabled schedule', enabled: false, status: 'disabled', trigger: { type: 'cron', expr: '* * * * *' } },
    { id: 'suppressed', name: 'Suppressed skill', enabled: false, status: 'suppressed', retained: true, restorable: true, source: 'skill_manifest', manual_override: 'disabled', trigger: { type: 'cron', expr: '* * * * *' } },
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

    assert.equal(history.hasAttribute('open'), false);
    assert.match(history.querySelector('summary').textContent, /suppressed \(2\)/);
    const retained = history.querySelectorAll('.activity-row');
    assert.equal(retained.length, 2);
    assert.match(retained[0].textContent, /Suppressed skill[\s\S]*suppressed/);
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
    for (const id of ['active', 'suppressed']) {
        routes.set(`/api/schedules/${id}/action`, (url, init) => {
            sent.push({ url, body: JSON.parse(init.body) });
            return response({ ok: true, changed: true, status: 'updated', audit: 'recorded', running_or_queued: false });
        });
    }
    await initActivity({ mount, ws }).refresh();
    const schedules = section(mount, 'schedules');
    const history = schedules.querySelector('[data-activity-history]');
    await mount.fire('click', { target: schedules.querySelector('[data-act="schedule-toggle"][data-id="active"]') });
    await mount.fire('click', { target: history.querySelector('[data-act="schedule-toggle"][data-id="suppressed"]') });
    assert.deepEqual(sent.map((entry) => [entry.url, entry.body.action]), [
        ['/api/schedules/active/action', 'disable'],
        ['/api/schedules/suppressed/action', 'restore'],
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
    assert.match(toasts().at(-1), /still running and was not stopped/);
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

test('Activity distinguishes a recorded restore that is not ready from audit failure', async (t) => {
    const { mount, routes, ws, toasts } = setup(t);
    emptyActivity(routes);
    routes.set(schedulesUrl, response({ tasks: lifecycleSchedules }));
    routes.set('/api/schedules/suppressed/action', () => response({
        ok: false, changed: true, status: 'restored_not_ready', audit: 'recorded',
        detail: 'skill_not_ready', running_or_queued: false,
    }));
    await initActivity({ mount, ws }).refresh();
    await mount.fire('click', { target: section(mount, 'schedules').querySelector(
        '[data-act="schedule-toggle"][data-id="suppressed"]') });
    assert.match(toasts().at(-1), /suppression lifted.*not ready to run: skill_not_ready/);
    assert.doesNotMatch(toasts().at(-1), /audit.*incomplete/);
});

test('Activity delete reads its outcome through the same seam as the toggles', async () => {
    // Delete opens a modal this fixture cannot drive, so the centralisation is
    // pinned at the source: ONE request helper, which reports every outcome, and
    // no branch that fires the action and ignores what came back.
    const source = await readFile(new URL('../modules/activity.js', import.meta.url), 'utf8');
    const requests = source.match(/fetchJson\(`\/api\/schedules\//g) || [];
    assert.equal(requests.length, 1, 'every lifecycle button goes through one request helper');
    const helper = source.split('async function scheduleAction(')[1].split('\n    }')[0];
    assert.match(helper, /reportScheduleOutcome\(action, outcome\)/);
    for (const branch of ["scheduleAction(id, 'delete'", 'scheduleAction(id, action']) {
        assert.ok(source.includes(branch), branch);
    }
    // No second, button-local reading of the response beside the shared one.
    assert.equal((source.match(/outcome\.ok/g) || []).length, 1);
});

test('Activity keeps the retained-history disclosure and focus the owner left open', async (t) => {
    const { mount, routes, ws } = setup(t);
    emptyActivity(routes);
    routes.set(schedulesUrl, response({ tasks: lifecycleSchedules }));
    const activity = initActivity({ mount, ws });
    await activity.refresh();
    const schedules = section(mount, 'schedules');
    assert.equal(schedules.querySelector('[data-activity-history]').hasAttribute('open'), false);

    // The owner opens the history and focuses a control inside it.
    const details = schedules.querySelector('[data-activity-history]');
    details.open = true;
    const restore = details.querySelector('[data-act="schedule-toggle"][data-id="suppressed"]');
    restore.focus();

    // A poll rebuilds the markup; it must not re-collapse or drop the focus.
    await activity.refresh();
    const rebuilt = section(mount, 'schedules').querySelector('[data-activity-history]');
    assert.equal(rebuilt.hasAttribute('open'), true, 'a refresh must not close what the owner opened');
    assert.equal(globalThis.document.activeElement.dataset.id, 'suppressed');
    assert.equal(globalThis.document.activeElement.dataset.act, 'schedule-toggle');

    // Collapsing it again is equally durable across the next refresh.
    rebuilt.open = false;
    await activity.refresh();
    assert.equal(section(mount, 'schedules').querySelector('[data-activity-history]').hasAttribute('open'), false);
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
    const oldControl = details.querySelector('[data-act="schedule-toggle"][data-id="suppressed"]');
    oldControl.focus();
    let finishSchedules;
    routes.set(schedulesUrl, () => new Promise((resolve) => { finishSchedules = resolve; }));
    const pending = activity.refresh();
    const moved = new NodeStub('button');
    globalThis.document.body.appendChild(moved);
    moved.focus();
    details.open = true;
    finishSchedules(response({ tasks: lifecycleSchedules }));
    await pending;

    const rebuilt = section(mount, 'schedules').querySelector('[data-activity-history]');
    assert.equal(rebuilt.hasAttribute('open'), true,
        'a render must retain disclosure state changed while its request was pending');
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
    routes.set(backgroundUrl, response({ bg_consciousness_enabled: true,
        active_chat_activities_complete: false, active_chat_activities: [
            { activity_id: 'direct-1', kind: 'direct_chat', phase: 'thinking', started_at: 88 },
            { activity_id: 'queued-1', kind: 'direct_chat', phase: 'thinking', started_at: 80 },
            { activity_id: 'paused-1', kind: 'managed_task', phase: 'running', started_at: 80 },
        ] }));
    routes.set(queueUrl, response({ queue: {
        running: [{ id: 'queued-1', type: 'task', runtime_sec: 9, task: { title: 'Queue title' } }],
        pending: [{ task: { id: 'paused-1', title: 'Paused work', _budget_pause: true } }],
    } }));
    await initActivity({ mount, ws }).refresh();
    const queue = section(mount, 'queue');
    const rows = queue.querySelectorAll('.activity-row');
    assert.equal(rows.length, 3, 'partial positive census adds exactly the missing direct identity');
    assert.match(rows[0].textContent, /Queue title[\s\S]*running · task · 9s/);
    assert.match(rows[1].textContent, /Paused work[\s\S]*paused \(budget\)/);
    assert.equal(rows[1].querySelector('button').dataset.budgetPaused, '1');
    assert.match(rows[2].textContent, /Direct turn[\s\S]*thinking · 12s/);
    assert.equal(rows[2].querySelector('button').dataset.id, 'direct-1');
    assert.equal(rows[2].querySelector('button').dataset.act, 'task-control');
    assert.doesNotMatch(queue.textContent, /Nothing running/);
    assert.deepEqual(calls, [queueUrl, backgroundUrl, schedulesUrl], 'reuse the existing state read');
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

test('Logs catches unavailable and malformed sources while retaining later live events', async (t) => {
    const { mount, routes, ws } = setup(t);
    routes.set(logUrl('events'), response(null));
    routes.set(logUrl('tools'), new Error('offline'));
    routes.set(logUrl('progress'), response({}, 503));
    routes.set(logUrl('supervisor'), { ok: true, json: async () => { throw new Error('not JSON'); } });
    initLogs({ mount, ws, state: {} });
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
    initLogs({ mount, ws, state: {} });
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

test('Costs numeric inputs have visible associated names and the task cap description', (t) => {
    const { mount } = setup(t);
    initCosts({ mount, state: {} });
    for (const [id, label] of [['s-budget', 'Total Budget ($)'], ['s-per-task-cost', 'Per-task Cost Cap ($)']]) {
        const field = mount.querySelector(`#${id}`);
        assert.ok(field);
        assert.equal(mount.querySelector(`[for="${id}"]`).textContent, label);
        assert.equal(field.classList.contains('ui-control'), true);
    }
    const input = mount.querySelector('#s-per-task-cost');
    const description = mount.querySelector(`#${input.getAttribute('aria-describedby')}`);
    assert.match(description.textContent, /whole root task tree/);
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


test('Activity names the typed no-change refusals and never calls a suppression "deleted"', async (t) => {
    const { mount, routes, ws, toasts } = setup(t);
    emptyActivity(routes);
    routes.set(schedulesUrl, response({ tasks: lifecycleSchedules }));
    const outcomes = [
        { ok: false, changed: false, status: 'manifest_absent', audit: 'recorded',
          detail: 'skill_absent: the skill no longer declares this schedule; suppression kept until it does' },
        { ok: false, changed: false, status: 'lock_timeout', audit: 'not_written',
          detail: 'Could not acquire schedule lock within 8s' },
        { ok: false, changed: false, status: 'not_suppressed', audit: 'recorded',
          detail: 'this skill schedule is not suppressed; it is disabled by skill readiness (skill_not_ready)' },
        { ok: true, changed: true, status: 'suppressed', audit: 'recorded', running_or_queued: false },
        { ok: true, changed: true, status: 'suppressed', audit: 'recorded', running_or_queued: null },
    ];
    let next = 0;
    routes.set('/api/schedules/suppressed/action', () => response(outcomes[next++]));
    await initActivity({ mount, ws }).refresh();
    const restore = () => mount.fire('click', { target: section(mount, 'schedules').querySelector(
        '[data-act="schedule-toggle"][data-id="suppressed"]') });
    await restore();
    assert.match(toasts().at(-1), /did not change anything: skill_absent: .*suppression kept/);
    assert.doesNotMatch(toasts().at(-1), /suppression lifted/);
    await restore();
    assert.match(toasts().at(-1), /did not change anything: Could not acquire schedule lock/);
    await restore();
    assert.match(toasts().at(-1), /did not change anything: this skill schedule is not suppressed/);
    // A suppression is reported as what it is, with and without an in-flight fact.
    await restore();
    assert.match(toasts().at(-1), /Schedule suppressed \(kept disabled until restored\)\./);
    assert.doesNotMatch(toasts().at(-1), /deleted/);
    await restore();
    assert.match(toasts().at(-1), /Schedule suppressed \(kept disabled until restored\)\. Whether a task/);
});

test('Activity offers no Enable on a skill row held back by readiness alone', async (t) => {
    const { mount, routes, ws } = setup(t);
    emptyActivity(routes);
    routes.set(schedulesUrl, response({ tasks: [
        { id: 'held', name: 'Held skill', enabled: false, status: 'disabled', source: 'skill_manifest',
          skill: 'demo', trigger: { type: 'cron', expr: '* * * * *' } },
    ] }));
    await initActivity({ mount, ws }).refresh();
    const row = section(mount, 'schedules').querySelector('.activity-row');
    assert.equal(row.querySelector('[data-act="schedule-toggle"]'), null);
    assert.match(row.textContent, /disabled by skill readiness/);
    // Delete stays (it suppresses) and is marked as a skill row for the dialog.
    assert.equal(row.querySelector('[data-act="schedule-delete"]').dataset.managed, '1');
});

test('Activity shows a notify row by its sentence with the notification tag and the owner controls', async (t) => {
    const { mount, routes, ws } = setup(t);
    emptyActivity(routes);
    routes.set(schedulesUrl, response({ tasks: [
        { id: 'notify-cal-evt-1-abc', name: 'Reminder from cal', kind: 'notify', source: 'skill:cal', enabled: true, status: 'active',
          trigger: { type: 'once', run_at: '2999-01-01T09:00:00+00:00' }, notification: { text: 'Dentist at 9', key: 'evt-1' } },
        { id: 'notify-cal-evt-2-def', name: 'Reminder from cal', kind: 'notify', source: 'skill:cal', enabled: false,
          manual_override: 'disabled', status: 'suppressed', retained: true, restorable: true, trigger: { type: 'once', run_at: '2999-01-02T09:00:00+00:00' },
          notification: { text: 'Standup', key: 'evt-2' } },
        { id: 'notify-cal-evt-3-ghi', name: 'Reminder from cal', kind: 'notify', source: 'skill:cal', enabled: false,
          completed_at: '2026-09-25T09:00:05+00:00', status: 'consumed', retained: true, trigger: { type: 'once', run_at: '2026-09-25T09:00:00+00:00' },
          notification: { text: 'Fired already', key: 'evt-3' } },
    ] }));
    await initActivity({ mount, ws }).refresh();
    const schedules = section(mount, 'schedules');
    const history = schedules.querySelector('[data-activity-history]');
    const standing = [...schedules.querySelectorAll('.activity-row')].filter((row) => !history.contains(row));
    assert.equal(standing.length, 1);
    // The sentence is the title; the row says it is a notification and names its skill,
    // never a skill-managed marker (no readiness caveat), and offers Disable.
    assert.match(standing[0].textContent, /Dentist at 9[\s\S]*notification · one-shot[\s\S]*· cal/);
    assert.doesNotMatch(standing[0].textContent, /Reminder from cal|readiness/);
    assert.equal(standing[0].querySelector('button').textContent, 'Disable');
    // The owner's disabled reminder is a suppressed record with Restore, like a skill row.
    const retained = history.querySelectorAll('.activity-row');
    assert.equal(retained.length, 2);
    assert.match(retained[0].textContent, /Standup[\s\S]*suppressed/);
    assert.equal(retained[0].querySelector('button').textContent, 'Restore');
    // Delete carries what the dialog needs to tell the truth: the first Delete of
    // an armed reminder suppresses it, deleting the retained record removes it,
    // and a reminder that already fired is a receipt Delete removes at once.
    const armedDelete = standing[0].querySelector('[data-act="schedule-delete"]');
    assert.equal(armedDelete.dataset.notify, '1');
    assert.equal(armedDelete.dataset.suppressed, '');
    assert.equal(armedDelete.dataset.consumed, '');
    assert.equal(retained[0].querySelector('[data-act="schedule-delete"]').dataset.suppressed, '1');
    const firedDelete = retained[1].querySelector('[data-act="schedule-delete"]');
    assert.match(retained[1].textContent, /Fired already[\s\S]*consumed once/);
    assert.equal(firedDelete.dataset.consumed, '1');
    // And the dialog each button opens says what the server will do.
    assert.equal(scheduleDeleteDialog(armedDelete.dataset).title, 'Suppress reminder');
    assert.equal(scheduleDeleteDialog(armedDelete.dataset).confirmLabel, 'Suppress');
    assert.equal(scheduleDeleteDialog(retained[0].querySelector('[data-act="schedule-delete"]').dataset).title, 'Delete schedule');
    assert.equal(scheduleDeleteDialog(firedDelete.dataset).title, 'Delete schedule');
    assert.equal(scheduleDeleteDialog(firedDelete.dataset).body, 'Delete this schedule?');
    assert.equal(scheduleDeleteDialog({ managed: '1' }).title, 'Suppress skill schedule');
    assert.equal(scheduleDeleteDialog({}).title, 'Delete schedule');
});
