// The Logs task card settles its headline only on the typed task_done frame.
// The supervisor bus serves control before telemetry, so a task's late tool or
// round frame — with its own `done` / `error` phase — can arrive after its
// task_done. Such a frame stays a timeline row; it neither latches the card
// mid-task (a finished tool is not a finished task) nor rewrites the settled
// headline. Runs the real module on a DOM stub; both directions are pinned.
import assert from 'node:assert/strict';
import test from 'node:test';

import { initLogs } from '../modules/logs.js';

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
    getClientRects() { return [{ width: 1, height: 1 }]; }  // drawn: the Logs reader asks whether its panel has a box
    get firstElementChild() { return this.children[0]; }
    get nextElementSibling() { return this.parentElement?.children[this.parentElement.children.indexOf(this) + 1]; }
    get textContent() { return this._text + this.children.map((child) => child.textContent).join(''); }
    set textContent(text) { this._text = String(text); this.children = []; }
    get innerHTML() { return this.textContent.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
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
    insertBefore(node, before) {  // as the DOM: before null appends
        node.remove();
        node.parentElement = this;
        this.children.splice(before ? this.children.indexOf(before) : this.children.length, 0, node);
        return node;
    }
    append(...nodes) { nodes.forEach((node) => this.appendChild(node)); }
    remove() {
        if (this.parentElement) this.parentElement.children.splice(this.parentElement.children.indexOf(this), 1);
        this.parentElement = null;
    }
    contains(node) { return this === node || this.children.some((child) => child.contains(node)); }
    matches(selector) {
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
    addEventListener(type, fn) {
        if (!this.events.has(type)) this.events.set(type, []);
        this.events.get(type).push(fn);
    }
    removeEventListener(type, fn) { this.events.set(type, (this.events.get(type) || []).filter((listener) => listener !== fn)); }
    dispatchEvent(event) { for (const fn of this.events.get(event.type) || []) fn(event); }
}

function setup(t) {
    const mount = new NodeStub();
    const body = new NodeStub('body');
    const document = {
        body,
        hidden: false,
        createElement: (tag) => new NodeStub(tag),
        getElementById: (id) => mount.querySelector(`#${id}`) || body.querySelector(`#${id}`),
        addEventListener() {},
        removeEventListener() {},
    };
    const wsHandlers = new Map();
    const ws = {
        on: (type, fn) => { wsHandlers.set(type, fn); },
        send() {},
        emit: (type, data = {}) => wsHandlers.get(type)?.(data),
    };
    const previous = Object.fromEntries(['document', 'window', 'fetch', 'requestAnimationFrame'].map((key) => [key, globalThis[key]]));
    Object.assign(globalThis, {
        document,
        window: new NodeStub(),
        requestAnimationFrame: (fn) => fn(),
        // The backfill of recent history: nothing recorded, every source available.
        fetch: async () => ({ ok: true, status: 200, json: async () => ({ entries: [] }) }),
    });
    t.after(() => { Object.assign(globalThis, previous); });
    initLogs({ mount, ws, state: { activePage: 'dashboard', dashboardActiveSubtab: 'logs' } });
    // Let the module's own history backfill (four empty sources) finish inside the test.
    const settled = (async () => { for (let i = 0; i < 4; i += 1) await new Promise((resolve) => setImmediate(resolve)); })();
    const card = () => mount.querySelectorAll('.log-task-card').find((node) => node.dataset.taskGroup === 't');
    return {
        settled,
        card,
        push: (evt) => ws.emit('log', { data: { ts: '2026-10-15T00:00:00+00:00', task_id: 't', ...evt } }),
        // The card's words (its latest headline, or its newest failure's own words while it holds one) and its failure
        // chip's word ('' while it holds no failure): the first of each in the card is its own.
        headline: () => card().querySelector('.log-words').textContent,
        choose: (kind) => mount.querySelectorAll('.filter-chip').find((chip) => chip.dataset.logKind === kind).dispatchEvent({ type: 'click' }),
        failure: () => (card().querySelector('.log-failure').hidden ? '' : card().querySelector('.log-failure-word').textContent),
        // A Timeline's rows exist while it is open: opened the way the browser does (its `toggle` reaches the list).
        timelineRows: () => {
            const details = card();
            details.open = true;
            mount.querySelector('#log-entries').dispatchEvent({ type: 'toggle', target: details });
            return card().querySelector('.log-task-timeline').querySelectorAll('.log-task-event').length;
        },
    };
}

test('a finished tool call never settles the card: the next heartbeat keeps it live', async (t) => {
    const logs = setup(t);
    await logs.settled;
    logs.push({ type: 'task_started', task_type: 'task', task_text: 'work' });
    logs.push({ type: 'tool_call_finished', tool: 'read_file', is_error: false, ts: '2026-10-15T00:00:01+00:00' });
    assert.equal(logs.failure(), '', 'a call that did not fail is not coloured');
    assert.equal(logs.headline(), 'Read file finished');
    logs.push({ type: 'task_heartbeat', phase: 'running', runtime_sec: 12, ts: '2026-10-15T00:00:02+00:00' });
    assert.equal(logs.failure(), '');
    assert.equal(logs.headline(), 'Still working');
    logs.push({ type: 'tool_call_finished', tool: 'run_command', is_error: true, ts: '2026-10-15T00:00:03+00:00' });
    assert.equal(logs.failure(), '1 step failed', 'a tool error is shown while the task runs, never as the task Failed');
    logs.push({ type: 'task_heartbeat', phase: 'running', runtime_sec: 15, ts: '2026-10-15T00:00:04+00:00' });
    assert.equal(logs.failure(), '1 step failed', 'the next heartbeat does not settle the card, and the failure it holds stays said');
    assert.equal(logs.headline(), 'Run command failed', 'in its own words');
    assert.equal(logs.timelineRows(), 5);
});

test('a failed task_done settles the headline; a late tool frame stays a timeline row', async (t) => {
    const logs = setup(t);
    await logs.settled;
    logs.push({ type: 'task_started', task_type: 'task', task_text: 'work' });
    logs.push({ type: 'task_done', status: 'failed', reason_code: 'tool_failure', ts: '2026-10-15T00:00:01+00:00' });
    assert.equal(logs.failure(), 'Failed', 'a failed task says the outcome\'s word');
    assert.equal(logs.headline(), '', 'once: its chip is its name, its line repeats nothing');
    logs.push({ type: 'tool_call_finished', tool: 'read_file', is_error: false, ts: '2026-10-15T00:00:02+00:00' });
    assert.deepEqual([logs.headline(), logs.failure()], ['', 'Failed'], 'the late done frame does not rewrite the card');
    logs.push({ type: 'task_heartbeat', phase: 'running', runtime_sec: 20, ts: '2026-10-15T00:00:03+00:00' });
    assert.deepEqual([logs.headline(), logs.failure()], ['', 'Failed'], 'nor does a late heartbeat reopen it');
    assert.equal(logs.timelineRows(), 4, 'late frames remain in the detail list');
    // A later typed terminal frame may still settle it again (a retry's own outcome).
    logs.push({ type: 'task_done', status: 'completed', ts: '2026-10-15T00:00:04+00:00' });
    assert.deepEqual([logs.headline(), logs.failure()], ['Done', ''], 'its own later outcome takes the failure back');
});

// Only a failure is coloured (Logs, owner 14.11): a settled task says its outcome in words, and only Failed takes the
// shared failure chip; a warning worth reading before it is its words, never a hue of its own.
test('a settled task says its outcome in words; only a failed one takes the failure chip', async (t) => {
    const logs = setup(t);
    await logs.settled;
    logs.push({ type: 'task_started', task_type: 'task', task_text: 'work' });
    logs.push({ type: 'task_checkpoint', checkpoint_kind: 'context_fit_low_retry', ts: '2026-10-15T00:00:01+00:00' });
    assert.equal(logs.failure(), '', 'a warning is said in its words');
    for (const [second, done, words, failure] of [
        [2, { status: 'cancelled' }, 'Stopped', ''],
        [3, { status: 'completed', outcome_axes: { execution: { status: 'degraded' } } }, 'Done with warnings', ''],
        [4, { status: 'failed' }, '', 'Failed'],  // said once, by the chip
        [5, { status: 'completed' }, 'Done', ''],
    ]) {
        logs.push({ type: 'task_done', ...done, ts: `2026-10-15T00:00:0${second}+00:00` });
        assert.deepEqual([logs.headline(), logs.failure()], [words, failure], JSON.stringify(done));
    }
});

// A card that holds a failure says it in the failure's own words whatever the strip shows (owner 14.11 sheets): its
// error, not its latest step, under All as under Errors, and still after the failure's row leaves the 30 rows the card
// keeps; nothing apologises for a row the card's words already hold.
test('a card holding a failure shows the failure\'s own words under every filter, after its row has left the timeline', async (t) => {
    const logs = setup(t);
    await logs.settled;
    logs.push({ type: 'task_started', task_type: 'task', task_text: 'work' });
    logs.push({ type: 'task_heartbeat', phase: 'running', runtime_sec: 1, ts: '2026-10-15T00:00:01+00:00' });
    assert.deepEqual([logs.headline(), logs.failure()], ['Still working', ''], 'before a failure: its latest step');
    logs.push({ type: 'tool_call_finished', tool: 'run_command', is_error: true, result_preview: 'TOOL_ERROR (run_command): exit 2',
        ts: '2026-10-15T00:00:02+00:00' });
    const said = ['TOOL_ERROR (run_command): exit 2', '1 step failed'];
    assert.deepEqual([logs.headline(), logs.failure()], said, 'its error, under All');
    for (let second = 3; second < 40; second += 1) {
        logs.push({ type: 'task_heartbeat', phase: 'running', runtime_sec: second, ts: `2026-10-15T00:00:${String(second).padStart(2, '0')}+00:00` });
    }
    assert.deepEqual([logs.headline(), logs.failure()], said, 'still, when 37 later steps came after it');
    assert.equal(logs.timelineRows(), 30, 'the failure\'s row has left the rows the card keeps');
    assert.doesNotMatch(logs.card().textContent, /older than|is not below/, 'and nothing apologises for it');
    await logs.choose('errors');
    assert.deepEqual([logs.headline(), logs.failure()], said, 'the same words under Errors');
    await logs.choose('');
    assert.deepEqual([logs.headline(), logs.failure()], said);
});
